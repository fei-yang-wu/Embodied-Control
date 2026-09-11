"""Replay an Isaac evaluation trace through the Embodied-Control pipeline.

The IsaacLab-Imitation evaluator (`evaluate_checkpoint.py --dump_trace`)
records, per control step of one environment: the raw robot state, the
observation terms the actor consumed, the published latent command, the macro
window the frozen encoder consumed, and the action. This script feeds ONLY the
robot state and the reference cursor into the deployment-side builders
(`fill_robot_anchored_window`, `ObservationAssembler`, the bundle's TorchScript
encoder and policy) and reports, per stage, how far EC's result is from what
Isaac produced from the same inputs:

  window   EC macro window (pelvis anchor, EC today; torso anchor, Isaac's
           `expert_anchor_body_name`) against Isaac's `hl/future_window[:-1]`
  z        EC encoder on the EC window, and on Isaac's own window, against the
           published `latent_command[:z_dim]`
  obs      EC assembler terms against Isaac's observation terms
  action   EC policy on Isaac's flat observation, and on EC's assembled
           observation, against Isaac's action

Usage (native env, from the Embodied-Control root):

    pixi run -e native python docs/evidence/isaac_replay_20260910/isaac_replay.py \
        <trace.npz> <bundle dir> <reference root> [--motion NAME] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.contracts import CommandSample, RobotState
from embodied_control.lowlevel.engine.torch_engine import TorchEngine
from embodied_control.lowlevel.macro_window import (
    fill_robot_anchored_window,
    window_indices,
)
from embodied_control.lowlevel.maths import wxyz_to_xyzw
from embodied_control.lowlevel.observation import ObservationAssembler
from embodied_control.lowlevel.reference import ReferenceArrays, ReferenceMotion


def _load_trace(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    raw = np.load(path, allow_pickle=True)
    rows: dict[str, np.ndarray] = {}
    static: dict[str, object] = {}
    for key in raw.files:
        if key.startswith("static/"):
            value = raw[key]
            static[key[len("static/") :]] = value.tolist() if value.ndim else value.item()
        else:
            rows[key] = raw[key]
    return rows, static


def _torso_motion(arrays: ReferenceArrays, motion: ReferenceMotion, body: str) -> ReferenceMotion:
    """The same motion re-anchored on `body` from the tracked-body arrays."""
    manifest = json.load(open(arrays.root / "reference_arrays_manifest.json"))
    spec = manifest["key"]["arrays"]
    index = arrays.motion_names.index(motion.name)
    start, end = arrays._starts[index], arrays._ends[index]
    body_index = arrays.body_names.index(body)
    pos = np.memmap(
        arrays.root / "body_pos_w.memmap",
        dtype=spec["body_pos_w"]["dtype"],
        mode="r",
        shape=tuple(spec["body_pos_w"]["shape"]),
    )[start:end, body_index]
    quat = np.memmap(
        arrays.root / "body_quat_w.memmap",
        dtype=spec["body_quat_w"]["dtype"],
        mode="r",
        shape=tuple(spec["body_quat_w"]["shape"]),
    )[start:end, body_index]
    if spec["body_quat_w"].get("quaternion_order") == "wxyz":
        quat = wxyz_to_xyzw(quat)
    return ReferenceMotion(
        name=motion.name,
        joint_qpos=motion.joint_qpos,
        anchor_pos_w=np.asarray(pos, dtype=np.float32),
        anchor_quat_w=np.asarray(quat, dtype=np.float32),
        joint_qvel=motion.joint_qvel,
        body_names=motion.body_names,
        body_pos_w=motion.body_pos_w,
    )


def _stat(name: str, err: np.ndarray, scale: np.ndarray | None = None) -> dict:
    err = np.asarray(err, dtype=np.float64)
    out = {
        "stage": name,
        "max_abs": float(np.nanmax(err)),
        "mean_abs": float(np.nanmean(err)),
        "p95_abs": float(np.nanpercentile(err, 95)),
    }
    if scale is not None:
        out["signal_rms"] = float(np.sqrt(np.nanmean(np.square(scale))))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("trace", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--motion", default=None, help="motion name; default: first traj_rank's name is not stored, so pass it")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None, help="limit to the first N steps")
    args = parser.parse_args()

    rows, static = _load_trace(args.trace)
    bundle = PolicyBundle.load(args.bundle)
    # `asset.data.joint_pos` is in the physics backend's joint order (Newton
    # permutes it); the observation terms and the bundle use the Isaac order
    # recorded in the action contract. Re-order the raw joint arrays once.
    dump_names = list(static["joint_names"])
    isaac_names = list(bundle.manifest.action.isaac_joint_names)
    perm = [dump_names.index(name) for name in isaac_names]
    for key in ("joint_pos", "joint_vel"):
        rows[key] = rows[key][:, perm]
    static["default_joint_pos"] = np.asarray(static["default_joint_pos"], np.float32)[perm]
    command = bundle.manifest.command
    obs_contract = bundle.manifest.obs
    z_dim = int(command.z_dim)
    state_dim = int(command.state_dim)
    window_steps = int(command.window_steps)
    stride = int(command.macro_frame_stride or 1)
    anchor_mode = str(command.macro_anchor_mode or "robot")

    arrays = ReferenceArrays(args.reference_root)
    if args.motion is None:
        raise SystemExit("--motion is required (the trace stores the rank, not the name)")
    motion_pelvis = arrays.motion(args.motion)
    motion_torso = _torso_motion(arrays, motion_pelvis, "torso_link")

    encoder = TorchEngine(bundle.encoder_path)
    policy = TorchEngine(bundle.policy_path)
    default_joint_pos = np.asarray(static["default_joint_pos"], dtype=np.float32)
    bundle_default = np.asarray(bundle.manifest.action.default_joint_pos, dtype=np.float32)
    assembler = ObservationAssembler(obs_contract, default_joint_pos=bundle_default)

    n = len(rows["step"]) if args.steps is None else min(args.steps, len(rows["step"]))
    # Stay inside the first episode of the trace: local_step must be monotone.
    local = rows["local_step"][:n]
    first_reset = np.flatnonzero(np.diff(local) < 0)
    if len(first_reset):
        n = int(first_reset[0] + 1)
        local = local[:n]

    # Isaac's live encoder input is `[state | future_window[:-1]]`: the current
    # frame followed by the next window_steps frames (the endpoint slot is
    # hidden in the `intermediate` window mode), the same slots EC fills from
    # `window_indices(cursor, ..., window_steps)`.
    hl_window = rows["hl_future_window"][:n]  # (n, H, state_dim), frames t+1..t+H
    isaac_state = rows["hl_state"][:n]
    isaac_window = np.concatenate(
        [isaac_state[:, None, :], hl_window[:, :-1, :]], axis=1
    ).reshape(n, -1)
    if isaac_window.shape[1] != state_dim * (window_steps + 1):
        raise SystemExit(
            f"isaac window width {isaac_window.shape[1]} != bundle "
            f"{state_dim} x {window_steps + 1}"
        )
    isaac_z = rows["obs/latent_command"][:n, :z_dim]
    isaac_action = rows["action"][:n]

    report: dict[str, object] = {
        "trace": str(args.trace),
        "bundle": str(args.bundle),
        "motion": args.motion,
        "steps": int(n),
        "default_joint_pos_max_diff_isaac_vs_bundle": float(
            np.abs(default_joint_pos - bundle_default).max()
        ),
        "window_slots": int(window_steps + 1),
        "stages": [],
    }

    # --- macro window -------------------------------------------------------
    ec_window = {"pelvis": np.zeros((n, isaac_window.shape[1]), np.float32),
                 "torso_link": np.zeros((n, isaac_window.shape[1]), np.float32)}
    motions = {"pelvis": motion_pelvis, "torso_link": motion_torso}
    for body, motion in motions.items():
        out = np.zeros(isaac_window.shape[1], np.float32)
        for i in range(n):
            indices = window_indices(int(local[i]), motion.length, window_steps, stride)
            fill_robot_anchored_window(
                motion,
                indices,
                rows[f"anchor_{body}_pos_w"][i],
                rows[f"anchor_{body}_quat_w"][i],
                state_dim,
                out,
                anchor_mode=anchor_mode,
            )
            ec_window[body][i] = out
        err = np.abs(ec_window[body] - isaac_window)
        # Per-feature breakdown of slot 0: [qpos 29 | anchor pos 3 | rot6d 6]
        slot0 = err[:, :state_dim]
        report["stages"].append(
            {
                **_stat(f"window[{body}] vs isaac", err, isaac_window),
                "slot0_qpos_max": float(slot0[:, :29].max()),
                "slot0_anchor_pos_max": float(slot0[:, 29:32].max()),
                "slot0_rot6d_max": float(slot0[:, 32:38].max()),
            }
        )

    # --- encoder ---------------------------------------------------------------
    def encode(windows: np.ndarray) -> np.ndarray:
        return np.stack([encoder.infer(w) for w in windows])

    z_from_isaac_window = encode(isaac_window)
    report["stages"].append(_stat("z: EC encoder on ISAAC window vs isaac z", np.abs(z_from_isaac_window - isaac_z), isaac_z))
    for body in motions:
        z_ec = encode(ec_window[body])
        report["stages"].append(_stat(f"z: EC encoder on EC window[{body}] vs isaac z", np.abs(z_ec - isaac_z), isaac_z))

    # --- observation assembly --------------------------------------------------
    obs_terms = [t for t in obs_contract.terms]
    isaac_obs_flat = np.concatenate(
        [rows[f"obs/{t.name}"][:n].reshape(n, -1) for t in obs_terms], axis=1
    )
    assembler.reset()
    ec_obs = np.zeros_like(isaac_obs_flat)
    last_action = np.zeros(29, np.float32)
    for i in range(n):
        state = RobotState(
            stamp=float(i) * 0.02,
            joint_pos=rows["joint_pos"][i],
            joint_vel=rows["joint_vel"][i],
            projected_gravity=rows["projected_gravity_b"][i],
            base_ang_vel=rows["root_ang_vel_b"][i],
        )
        sample = CommandSample(
            vector=rows["obs/latent_command"][i],
            age_ticks=0,
            renewed=True,
            terms={"latent_command": rows["obs/latent_command"][i]},
        )
        ec_obs[i] = assembler.assemble(state, sample, last_action)
        last_action = isaac_action[i]
    per_term = {}
    for t in obs_terms:
        sl = assembler._slices[t.name]
        per_term[t.name] = _stat(f"obs[{t.name}]", np.abs(ec_obs[:, sl] - isaac_obs_flat[:, sl]), isaac_obs_flat[:, sl])
    report["stages"].append({**_stat("obs: EC assembler vs isaac terms", np.abs(ec_obs - isaac_obs_flat), isaac_obs_flat), "per_term": per_term})

    # --- policy --------------------------------------------------------------------
    act_isaac_obs = np.stack([policy.infer(o) for o in isaac_obs_flat])
    report["stages"].append(_stat("action: EC policy on ISAAC obs vs isaac action", np.abs(act_isaac_obs - isaac_action), isaac_action))
    act_ec_obs = np.stack([policy.infer(o) for o in ec_obs])
    report["stages"].append(_stat("action: EC policy on EC obs vs isaac action", np.abs(act_ec_obs - isaac_action), isaac_action))

    for stage in report["stages"]:
        extra = {k: v for k, v in stage.items() if k not in ("stage", "max_abs", "mean_abs", "p95_abs", "signal_rms", "per_term")}
        print(f"{stage['stage']:58s} max {stage['max_abs']:.4g}  mean {stage['mean_abs']:.4g}  p95 {stage['p95_abs']:.4g}  (signal rms {stage.get('signal_rms', float('nan')):.3g}) {extra if extra else ''}")
        if "per_term" in stage:
            for name, st in stage["per_term"].items():
                print(f"    {name:20s} max {st['max_abs']:.4g}  mean {st['mean_abs']:.4g}  rms {st['signal_rms']:.3g}")
    print(f"default_joint_pos isaac vs bundle max diff {report['default_joint_pos_max_diff_isaac_vs_bundle']:.3g}; steps {n}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
