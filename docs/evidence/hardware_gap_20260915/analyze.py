"""Recompute matched-frame hardware/sim comparisons without controlling a robot.

Run from the repository root with:
    pixi run -e native python docs/evidence/hardware_gap_20260915/analyze.py
"""

from pathlib import Path
import hashlib
import json

import ec_native
import mujoco
import numpy as np

from embodied_control.lowlevel.reference import ReferenceArrays


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MOTION = "walking_quip_360_R_002__A428"
BUNDLE = "post_c1l1j1p1_f84500152320"
MODEL = ROOT / "assets/latent_playkit/model/g1_29dof_rev_1_0.xml"
REFERENCE = ReferenceArrays(ROOT / "assets/models/reference/bones")
REF = REFERENCE.motion(MOTION)
J = REFERENCE.joint_names
LEGS = [i for i, n in enumerate(J) if any(s in n for s in ("hip", "knee", "ankle"))]
CONTRACT = json.loads((ROOT / f"assets/models/controller/{BUNDLE}/action_contract.json").read_text())
assert J == CONTRACT["isaac_joint_names"]


def source(root, hardware):
    paths = [p for p in (ROOT / root).rglob("telemetry.npz")
             if ("hardware" in p.parts) == hardware and MOTION in str(p)]
    assert len(paths) == 1, paths
    return paths[0]


def read(path):
    with np.load(path) as raw:
        f = raw["reference_frames"]
        indices = np.flatnonzero((f > 0) & (f < REF.length))
        # Each recorded frame must be a distinct advancing tick; paused frame 0
        # and terminal -1 rows cannot enter either the dynamics or timing metrics.
        assert len(indices) and np.all(np.diff(f[indices]) == 1), path
        return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    frames=f[indices], q=raw["joint_position_log"][indices].astype(float),
                    cmd=raw["command_target_log"][indices].astype(float),
                    anchor=raw["anchor_pose_log"][indices].astype(float),
                    duration_ms=raw["tick_durations_ns"][indices] / 1e6,
                    all_q=raw["joint_position_log"].copy(),
                    all_anchor=raw["anchor_pose_log"].copy(), indices=indices)


def geometry(run):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    addresses = [model.joint(n).qposadr[0] for n in J]
    feet = [model.body(n).id for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    centers = np.array([[-.05, .025, -.03], [-.05, -.025, -.03],
                        [.12, .03, -.03], [.12, -.03, -.03]])
    soles = []
    ankle = []
    for q, pose in zip(run["q"], run["anchor"]):
        data.qpos[:] = 0
        data.qpos[3:7] = pose[[6, 3, 4, 5]]
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model, data)
        ankle.append(data.xpos[feet, 2].copy())
        soles.append([(centers @ data.xmat[b].reshape(3, 3).T + data.xpos[b])[:, 2] - .005 for b in feet])
    soles = np.array(soles)
    ankle = np.array(ankle)
    x, y, z, w = run["anchor"][:, 3:].T
    run["pitch_deg"] = np.rad2deg(np.arcsin(np.clip(2 * (w*y-z*x), -1, 1)))
    run["sole_height_difference_m"] = soles.min(2)[:, 0] - soles.min(2)[:, 1]
    run["ankle_height_difference_m"] = ankle[:, 0] - ankle[:, 1]


def stats(run, end):
    take = run["frames"] <= end
    q, cmd, fr = run["q"][take], run["cmd"][take], run["frames"][take]
    out = {"last_frame": int(fr[-1]), "leg_reference_mae_rad": float(np.abs(q-REF.joint_qpos[fr])[:, LEGS].mean()),
           "leg_command_mae_rad": float(np.abs(q-cmd)[:, LEGS].mean()),
           "compute_p99_ms": float(np.percentile(run["duration_ms"][take], 99)),
           "pitch_min_max_deg": [float(v) for v in np.percentile(run["pitch_deg"][take], [0, 100])]}
    for name in ("left_hip_pitch_joint", "left_knee_joint", "left_ankle_pitch_joint", "right_knee_joint"):
        j = J.index(name)
        out[name] = {"q_std": float(q[:, j].std()), "target_std": float(cmd[:, j].std()),
                     "std_ratio_not_transfer_gain": float(q[:, j].std()/cmd[:, j].std()),
                     "target_step_p99_rad": float(np.percentile(np.abs(np.diff(cmd[:, j])), 99)),
                     "q_min_max": [float(q[:, j].min()), float(q[:, j].max())]}
    return out


def estimator_comparison(run):
    estimator = ec_native.LegOdometry(str(MODEL), J)
    kin = np.array([estimator.update(np.ascontiguousarray(q, dtype=np.float32),
                                    np.ascontiguousarray(pose[3:], dtype=np.float32))
                    for q, pose in zip(run["all_q"], run["all_anchor"])])
    begin, end = run["indices"][[0, -1]]
    onboard = run["all_anchor"][[begin, end], :2]
    replay = kin[[begin, end], :2]
    return {"recorded_anchor_displacement_m": float(np.linalg.norm(onboard[1]-onboard[0])),
            "kinematic_replay_displacement_m": float(np.linalg.norm(replay[1]-replay[0])),
            "reference_displacement_m": float(np.linalg.norm(REF.anchor_pos_w[run["frames"][-1], :2]-REF.anchor_pos_w[run["frames"][0], :2])),
            "note": "Same recorded joints and orientation; replay is a no-slip estimate, not ground truth."}


def main():
    roots = {
        "f845_legkin": f"artifacts/post_echost_legkin/{BUNDLE}",
        "f845_odom": f"artifacts/post_echost/{BUNDLE}/{MOTION}",
        "f885_odom": f"artifacts/post_echost/post_c1l1j1p1_f88500338688/{MOTION}",
        "sonic_odom": f"artifacts/e5gap_echost/sonic_v1_1/{MOTION}",
    }
    runs = {}
    for name, root in roots.items():
        for hardware in (False, True):
            key = f"{name}_{'hw' if hardware else 'sim'}"
            runs[key] = read(source(root, hardware))
    calibration = ROOT / "artifacts/plant_calibration_20260915"
    for name in ("baseline", "fric_lo", "fric_hi", "lag15", "fric_lo_lag10"):
        runs[f"cal_{name}"] = read(source(str((calibration / name).relative_to(ROOT)), False))
    for run in runs.values():
        geometry(run)
    end = min(int(r["frames"][-1]) for r in runs.values())
    summary = {"common_prefix_frames": [1, end], "mujoco_version": mujoco.__version__,
               "runs": {}, "pairs": {}, "estimator_replay": {}}
    for name, run in runs.items():
        summary["runs"][name] = {"path": run["path"], "sha256": run["sha256"], **stats(run, end)}
        result_path = (ROOT / run["path"]).parents[2] / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            summary["runs"][name]["runtime"] = {k: result["control"][k] for k in
                ("control_ticks", "encoder_inferences", "deadline_misses", "planner_requests", "last_chunk_offset_steps", "scheduler_deadlines_missed")}
    for name in roots:
        hw, sim = runs[f"{name}_hw"], runs[f"{name}_sim"]
        n = len(hw["frames"])
        summary["pairs"][name] = {"frames": n, "target_mae_per_50_frames_rad": [], "joint_mae_per_50_frames_rad": []}
        for start in range(0, n, 50):
            sl = slice(start, min(start+50, n))
            for key, label in (("cmd", "target"), ("q", "joint")):
                summary["pairs"][name][f"{label}_mae_per_50_frames_rad"].append(
                    float(np.abs(hw[key][sl]-sim[key][sl])[:, LEGS].mean()))
        summary["estimator_replay"][name] = estimator_comparison(hw)
    summary["event"] = {}
    for name in ("f845_legkin_hw", "f845_legkin_sim", "cal_baseline", "cal_fric_hi", "cal_lag15"):
        run = runs[name]
        selected = (run["frames"] >= 170) & (run["frames"] <= 190)
        summary["event"][name] = {"frame": run["frames"][selected].tolist(),
            "left_knee_q": run["q"][selected, J.index("left_knee_joint")].tolist(),
            "left_knee_target": run["cmd"][selected, J.index("left_knee_joint")].tolist(),
            "left_sole_above_right_m": run["sole_height_difference_m"][selected].tolist(),
            "pitch_deg": run["pitch_deg"][selected].tolist()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_runs = ("f845_legkin_hw", "f845_legkin_sim", "cal_baseline")
    plot_keys = ("frames", "q", "cmd", "pitch_deg", "sole_height_difference_m", "ankle_height_difference_m")
    np.savez_compressed(OUT / "plot_data.npz", joint_names=J,
        **{f"{name}/{key}": runs[name][key] for name in plot_runs for key in plot_keys})
    print("common frames", [1, end])
    for name, row in summary["runs"].items():
        print(name, "leg_mae", round(row["leg_reference_mae_rad"], 4),
              "knee_target_step_p99", round(row["left_knee_joint"]["target_step_p99_rad"], 3),
              "pitch_range", np.round(row["pitch_min_max_deg"], 2))
    print("estimator replay", json.dumps(summary["estimator_replay"], indent=2))


if __name__ == "__main__":
    main()
