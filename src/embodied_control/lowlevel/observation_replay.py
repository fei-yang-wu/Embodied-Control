"""Observation replay: the policy side of the loop against a recording.

Step 3 of docs/evidence/hardware_gap_20260915: a run's telemetry carries what
the controller consumed on every tick (``state_log``, ``observation_log``,
``command_log``, ``encoder_window_log``, ``command_target_log``). Three
checks, each a deterministic function of the recording and the bundle:

* ``actor``: the recorded actor observation through ``policy_onnx`` and the
  action decode must give the recorded PD target (past blend-in, where the
  written target is a mix of the held pose and the policy's).
* ``encoder``: the recorded encoder window through ``encoder_onnx`` must give
  the latent slice of the recorded command on the same tick.
* ``assembly``: the recorded sensor state, command and decoded last action
  through the Python ``ObservationAssembler`` must give the recorded
  observation, so the native history buffers and term layout are the ones
  the policy was trained on.

A mismatch in ``actor`` or ``encoder`` is an inference fault (provider,
threads, non-determinism); a mismatch in ``assembly`` is an observation
pipeline fault. Both are zero on a plant run of the same build, which is the
floor a hardware recording is read against. None of this says whether the
observation itself was true to the robot; that is the plant side
(``command_replay``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.contracts import CommandSample, RobotState
from embodied_control.lowlevel.observation import ObservationAssembler

STANDARD_TERMS = (
    "projected_gravity", "base_ang_vel", "joint_pos_rel", "joint_vel_rel",
    "last_action",
)


@dataclass
class ObservationRecording:
    joint_position: np.ndarray      # ticks x 29
    joint_velocity: np.ndarray      # ticks x 29
    projected_gravity: np.ndarray   # ticks x 3
    base_angular_velocity: np.ndarray  # ticks x 3
    observation: np.ndarray         # ticks x obs width (NaN on ticks without a step)
    command: np.ndarray             # ticks x command width
    encoder_window: np.ndarray      # ticks x encoder width (empty when no encoder)
    target: np.ndarray              # ticks x 29
    stamps_ns: np.ndarray           # ticks x 2

    @property
    def ticks(self) -> int:
        return int(self.observation.shape[0])

    @property
    def controlled(self) -> np.ndarray:
        return np.isfinite(self.observation).all(axis=1)


def load_observation_recording(path: str | Path) -> ObservationRecording:
    data = np.load(path)
    missing = [k for k in ("observation_log", "state_log", "command_log",
                           "command_target_log", "joint_position_log") if k not in data]
    if missing:
        raise ValueError(
            f"{path} has no observation replay record ({', '.join(missing)} missing): "
            "it predates the per-tick observation log")
    state = np.asarray(data["state_log"], np.float64)
    window = np.asarray(data["encoder_window_log"], np.float64) if "encoder_window_log" in data else np.empty((state.shape[0], 0))
    stamps = np.asarray(data["tick_stamps_ns"], np.uint64) if "tick_stamps_ns" in data else np.zeros((state.shape[0], 2), np.uint64)
    return ObservationRecording(
        joint_position=np.asarray(data["joint_position_log"], np.float64),
        joint_velocity=state[:, :29],
        projected_gravity=state[:, 29:32],
        base_angular_velocity=state[:, 32:35],
        observation=np.asarray(data["observation_log"], np.float64),
        command=np.asarray(data["command_log"], np.float64),
        encoder_window=window.reshape(state.shape[0], -1),
        target=np.asarray(data["command_target_log"], np.float64),
        stamps_ns=stamps.reshape(-1, 2),
    )


def _engine(bundle: PolicyBundle, model_name: str, inference: dict | None):
    import ec_native

    artifact = bundle.manifest.models[model_name]
    engine = ec_native.OnnxEngine(
        str(bundle.model_path(model_name, expected_format="onnx")),
        artifact.input_name, artifact.output_name,
        artifact.input_shape[1], artifact.output_shape[1], 1, dict(inference or {}),
    )
    engine.warmup(4)
    return engine


@dataclass
class ReplayCheck:
    name: str
    ticks_compared: int
    max_abs: float
    p95_abs: float
    first_clean_tick: int          # first tick after which every tick is within tol
    tol: float
    per_tick_max: np.ndarray = field(repr=False)

    @property
    def ok(self) -> bool:
        return bool(self.ticks_compared) and self.max_abs <= self.tol

    def summary(self) -> dict:
        return {
            "ticks_compared": self.ticks_compared, "max_abs": self.max_abs,
            "p95_abs": self.p95_abs, "first_clean_tick": self.first_clean_tick,
            "tol": self.tol, "ok": self.ok,
        }


def _check(name: str, produced: np.ndarray, recorded: np.ndarray, mask: np.ndarray, tol: float) -> ReplayCheck:
    per_tick = np.full(mask.shape[0], np.nan)
    if mask.any():
        per_tick[mask] = np.abs(produced[mask] - recorded[mask]).max(axis=1)
    finite = per_tick[np.isfinite(per_tick)]
    # First tick from which the error never rises above tol again.
    first_clean = -1
    above = np.where(np.isfinite(per_tick) & (per_tick > tol))[0]
    if mask.any():
        first_clean = int(above[-1] + 1) if len(above) else int(np.where(mask)[0][0])
    return ReplayCheck(
        name=name, ticks_compared=int(mask.sum()),
        max_abs=float(finite.max()) if len(finite) else float("nan"),
        p95_abs=float(np.percentile(finite, 95)) if len(finite) else float("nan"),
        first_clean_tick=first_clean, tol=tol, per_tick_max=per_tick,
    )


def replay_actor(rec: ObservationRecording, bundle: PolicyBundle, *, blend_ticks: int = 0,
                 inference: dict | None = None, tol: float = 1e-3) -> tuple[ReplayCheck, np.ndarray]:
    """Recorded observation -> policy_onnx -> decode, against the recorded target.

    Blend-in ticks write ``held*(1-w) + policy*w``; they are skipped by
    ``blend_ticks`` (the job's value) and by the trailing-window rule
    ``first_clean_tick``.
    """
    engine = _engine(bundle, "policy_onnx", inference)
    action = bundle.manifest.action
    default = np.asarray(action.default_joint_pos, np.float64)
    scale = np.asarray(action.action_scale, np.float64)
    clip = action.raw_action_clip
    produced = np.full_like(rec.target, np.nan)
    actions = np.full((rec.ticks, action.width), np.nan)
    for t in np.where(rec.controlled)[0]:
        a = np.asarray(engine.infer(rec.observation[t].astype(np.float32)), np.float64)
        if clip is not None:
            a = np.clip(a, -clip, clip)
        actions[t] = a
        produced[t] = default + a * scale
    mask = rec.controlled.copy()
    mask[:blend_ticks] = False
    return _check("actor", produced, rec.target, mask, tol), actions


def replay_encoder(rec: ObservationRecording, bundle: PolicyBundle, *, inference: dict | None = None,
                   tol: float = 1e-3) -> ReplayCheck | None:
    """Recorded encoder window -> encoder_onnx, against the recorded latent."""
    if "encoder_onnx" not in bundle.manifest.models or rec.encoder_window.shape[1] == 0:
        return None
    engine = _engine(bundle, "encoder_onnx", inference)
    z_dim = int(bundle.manifest.command.z_dim)
    encoded = np.isfinite(rec.encoder_window).all(axis=1) & rec.controlled
    produced = np.full((rec.ticks, z_dim), np.nan)
    for t in np.where(encoded)[0]:
        produced[t] = np.asarray(engine.infer(rec.encoder_window[t].astype(np.float32)), np.float64)[:z_dim]
    return _check("encoder", produced, rec.command[:, :z_dim], encoded, tol)


def replay_assembly(rec: ObservationRecording, bundle: PolicyBundle, actions: np.ndarray, *,
                    blend_ticks: int = 0, tol: float = 1e-4) -> ReplayCheck:
    """Recorded state + command + last action -> ObservationAssembler, against
    the recorded observation. The last action fed back is the decoded action
    of the previous controlled tick (what the native loop reports past
    blend-in; during blend-in it reports ``action*w``, hence ``blend_ticks``).
    """
    contract = bundle.manifest.obs
    assembler = ObservationAssembler(
        contract, default_joint_pos=np.asarray(bundle.manifest.action.default_joint_pos, np.float32))
    command_terms = [t for t in contract.terms if t.name not in STANDARD_TERMS]
    produced = np.full_like(rec.observation, np.nan)
    last = np.zeros(bundle.manifest.action.width, np.float32)
    for t in range(rec.ticks):
        if not rec.controlled[t]:
            continue
        cursor = 0
        terms = {}
        for term in command_terms:
            terms[term.name] = rec.command[t, cursor:cursor + term.width].astype(np.float32)
            cursor += term.width
        state = RobotState(
            stamp=float(t), joint_pos=rec.joint_position[t].astype(np.float32),
            joint_vel=rec.joint_velocity[t].astype(np.float32),
            projected_gravity=rec.projected_gravity[t].astype(np.float32),
            base_ang_vel=rec.base_angular_velocity[t].astype(np.float32),
        )
        sample = CommandSample(vector=rec.command[t].astype(np.float32), age_ticks=0, renewed=True, terms=terms)
        produced[t] = assembler.assemble(state, sample, last)
        if np.isfinite(actions[t]).all():
            last = actions[t].astype(np.float32)
    mask = rec.controlled.copy()
    # The recorded last-action history reaches back history_length ticks past
    # blend-in; skip that span too so every compared term is post-blend.
    span = max((t.history_length * t.history_stride for t in contract.terms), default=1)
    mask[:blend_ticks + span] = False
    return _check("assembly", produced, rec.observation, mask, tol)


def replay(rec: ObservationRecording, bundle: PolicyBundle, *, blend_ticks: int = 0,
           inference: dict | None = None) -> dict[str, ReplayCheck]:
    actor, actions = replay_actor(rec, bundle, blend_ticks=blend_ticks, inference=inference)
    checks = {"actor": actor}
    encoder = replay_encoder(rec, bundle, inference=inference)
    if encoder is not None:
        checks["encoder"] = encoder
    checks["assembly"] = replay_assembly(rec, bundle, actions, blend_ticks=blend_ticks)
    return checks


def write_report(checks: dict[str, ReplayCheck], rec: ObservationRecording, output: str | Path, *,
                 label: str = "") -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stamps = rec.stamps_ns.astype(np.int64)
    tick_period_ms = np.diff(stamps[:, 0]) / 1e6 if rec.ticks > 1 else np.empty(0)
    sensor_age_ms = (stamps[:, 0] - stamps[:, 1]) / 1e6
    sensor_age_ms = sensor_age_ms[stamps[:, 1] > 0]
    summary = {
        "label": label, "ticks": rec.ticks, "controlled_ticks": int(rec.controlled.sum()),
        "checks": {k: v.summary() for k, v in checks.items()},
        "tick_period_ms": {"median": float(np.median(tick_period_ms)) if len(tick_period_ms) else float("nan"),
                           "max": float(tick_period_ms.max()) if len(tick_period_ms) else float("nan")},
        "sensor_age_ms": {"median": float(np.median(sensor_age_ms)) if len(sensor_age_ms) else float("nan"),
                          "p95": float(np.percentile(sensor_age_ms, 95)) if len(sensor_age_ms) else float("nan"),
                          "max": float(sensor_age_ms.max()) if len(sensor_age_ms) else float("nan")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(output / "per_tick.npz", **{f"{k}_max_abs": v.per_tick_max for k, v in checks.items()},
             tick_stamps_ns=rec.stamps_ns)
    return summary


__all__ = [
    "ObservationRecording", "ReplayCheck", "load_observation_recording", "replay",
    "replay_actor", "replay_assembly", "replay_encoder", "write_report",
]
