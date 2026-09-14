"""Tracking error of a plant rehearsal, scored from the plant's true state.

`ec lifecycle rehearse` leaves, per motion, the plant's true trajectory
(`plant.states.npz`, 500 Hz) and the controller's telemetry (`telemetry.npz`,
50 Hz, with the reference frame the controller was tracking on every tick).
The controller's own anchor pose is an estimate (odometry) or a frozen
start, so MPJPE-G and drift must come from the plant. This module joins the
two by joint vector (the plant serves the exact joints the controller
reads, up to the sensor noise it injects), maps the reference into the
plant world with the frame-0 heading alignment `eval_mpjpe` uses, and
reports per motion:

- MPJPE-L / MPJPE-G (mm) over the ticks that carried a reference frame,
- root drift (max planar distance between the true pelvis and the aligned
  reference anchor, m),
- the commanded target's dither (mean per-tick second difference, rad,
  ankles and all joints),
- the commanded target against the bundle's training joint limits: the
  ankles' largest excursion past a limit (rad), the share of (tick, joint)
  pairs whose target sits past a limit (all joints, and the ankles), and
  the measured joints' largest excursion past a limit on the plant rows
  (rad; the writer faults at 0.1). A target past the limit is ordinary (a
  PD target is a torque proxy); a MEASURED joint past the limit is the
  ankle-stop failure mode,
- the controller's anchor error (its anchor displacement against the plant
  truth, m; the odometry quality when the anchor is live),
- the anchor source the episode settled on (from lifecycle.jsonl),
- Isaac's smoothness metrics recomputed from the plant at the 50 Hz control
  step so the plant row can sit next to the clean-board row: body
  acceleration and jerk (mean over the reference bodies of |d v/dt| and
  |d a/dt| from finite differences of the FK body positions),
  tracking acceleration distance (|a_robot - a_ref| the same way),
  action_delta_l2 (norm of the raw-action step), and
- the actuator cost the PD loop pays on the 500 Hz plant rows: the applied
  torque `kp (target - q) - kd qdot` clamped to the effort limit (the plant's
  own law; it logs no torque), as mean sum tau^2 (Isaac's joint_torques_l2
  term) and mean sum |tau qdot| (Isaac's energy_consumption term), with the
  target of the last control tick held on every row, as the writer does.
"""

from __future__ import annotations

import json
from dataclasses import fields, asdict, dataclass
from pathlib import Path

import numpy as np
import yaml

from embodied_control.lowlevel.maths import quat_conjugate, quat_mul, quat_to_mat


@dataclass
class MotionGrade:
    motion: str
    seed: int
    passed: bool
    ticks: int
    reference_ticks: int
    mpjpe_l_mm: float
    mpjpe_g_mm: float
    drift_max_m: float
    ankle_target_dither: float
    all_target_dither: float
    ankle_target_excess_max_rad: float
    target_beyond_limit_pct: float
    ankle_target_beyond_limit_pct: float
    measured_excess_max_rad: float
    anchor_error_max_m: float
    body_acc_mps2: float
    body_jerk_mps3: float
    tracking_acceleration_distance_mps2: float
    action_delta_l2: float
    joint_torques_l2: float
    energy_consumption: float
    anchor_position_source: str
    live_anchor: bool
    directory: str


def _fk_factory(mjcf: str, isaac_joint_names: list[str], body_names: tuple[str, ...]):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(mjcf))
    data = mujoco.MjData(model)
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names]
    if min(bodies) < 0:
        missing = [n for n, b in zip(body_names, bodies) if b < 0]
        raise ValueError(f"MJCF lacks reference bodies {missing}")
    qpos_address = []
    for name in isaac_joint_names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint < 0:
            raise ValueError(f"MJCF lacks joint {name}")
        qpos_address.append(int(model.jnt_qposadr[joint]))

    def body_positions(root_pos, quat_xyzw, joints):
        data.qpos[:3] = root_pos
        data.qpos[3] = quat_xyzw[3]
        data.qpos[4:7] = quat_xyzw[:3]
        for address, value in zip(qpos_address, joints):
            data.qpos[address] = value
        mujoco.mj_kinematics(model, data)
        return data.xpos[bodies].copy()

    return body_positions


def _second_difference(values: np.ndarray) -> float:
    if values.shape[0] < 3:
        return float("nan")
    return float(np.abs(values[2:] - 2.0 * values[1:-1] + values[:-2]).mean())


def _anchor_evidence(run: Path) -> tuple[str, bool]:
    source = ""
    live = True
    job = run / "job.yaml"
    if job.is_file():
        spec = yaml.safe_load(job.read_text()) or {}
        if "live_anchor" not in spec:
            # A job written before 2026-09-11 has no live anchor: the
            # translation was frozen at the reference start frame.
            return "fixed_start", False
        live = bool(spec.get("live_anchor", True))
    log = run / "lifecycle.jsonl"
    if log.is_file():
        for line in log.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            values = event.get("values") or {}
            if "anchor_position_source" in values:
                source = str(values["anchor_position_source"])
    if not live:
        source = "expert_heading"
    return source or "unknown", live


def grade_run(
    run: Path,
    *,
    reference,
    motion_name: str,
    isaac_joint_names: list[str],
    mjcf: str,
    seed: int = 0,
    action=None,
) -> MotionGrade:
    """One `sim/seed_N` run directory."""
    result = json.loads((run / "result.json").read_text()) if (run / "result.json").is_file() else {}
    telemetry_paths = sorted((run / "episodes").glob("*/telemetry.npz"))
    states_path = run / "plant.states.npz"
    source, live = _anchor_evidence(run)
    nan = float("nan")
    if not telemetry_paths or not states_path.is_file():
        # No episode telemetry (the run faulted before the policy took
        # over): every metric is NaN, filled by name so a new column can
        # never shift the positional list again.
        blank = {
            f.name: float("nan") for f in fields(MotionGrade) if f.type in (float, "float")
        }
        return MotionGrade(
            **blank, motion=motion_name, seed=seed, passed=bool(result.get("passed", False)),
            ticks=0, reference_ticks=0, anchor_position_source=source, live_anchor=live,
            directory=str(run),
        )
    telemetry = np.load(telemetry_paths[-1])
    states = np.load(states_path)
    motion = reference.motion(motion_name)
    plant_names = [str(n) for n in states["joint_names"]]
    permutation = [plant_names.index(n) for n in isaac_joint_names]
    plant_joints = np.asarray(states["joint_pos"], np.float64)[:, permutation]
    controller_joints = np.asarray(telemetry["joint_position_log"], np.float64)
    reference_frames = np.asarray(telemetry["reference_frames"])
    targets = np.asarray(telemetry["command_target_log"], np.float64)
    anchor_log = np.asarray(telemetry["anchor_pose_log"], np.float64)
    ticks = np.where(reference_frames >= 0)[0]
    if ticks.size == 0:
        return MotionGrade(
            motion_name, seed, bool(result.get("passed", False)), int(len(controller_joints)), 0,
            nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, source, live, str(run),
        )
    # Join each control tick to the plant row with the same joint vector.
    rows = np.array([
        int(np.abs(plant_joints - controller_joints[k]).mean(axis=1).argmin()) for k in ticks
    ])
    frames = reference_frames[ticks]
    root_pos = np.asarray(states["root_pos"], np.float64)[rows]
    root_quat = np.asarray(states["root_quat_xyzw"], np.float32)[rows]
    fk = _fk_factory(mjcf, isaac_joint_names, motion.body_names)
    robot_bodies = np.array([
        fk(root_pos[i], root_quat[i], controller_joints[k]) for i, k in enumerate(ticks)
    ])
    # Frame-0 heading alignment: reference world -> plant world.
    delta = quat_mul(root_quat[0], quat_conjugate(motion.anchor_quat_w[frames[0]].astype(np.float32)))
    delta[0] = delta[1] = 0.0
    delta = delta / np.linalg.norm(delta)
    rotation = quat_to_mat(delta)
    translation = root_pos[0] - rotation @ motion.anchor_pos_w[frames[0]]
    reference_bodies = motion.body_pos_w[frames] @ rotation.T + translation
    reference_root = motion.anchor_pos_w[frames] @ rotation.T + translation
    local = np.linalg.norm(
        (robot_bodies - root_pos[:, None]) - (reference_bodies - reference_root[:, None]), axis=-1
    ).mean() * 1000.0
    global_ = np.linalg.norm(robot_bodies - reference_bodies, axis=-1).mean() * 1000.0
    drift = float(np.linalg.norm(root_pos[:, :2] - reference_root[:, :2], axis=1).max())
    estimate = anchor_log[ticks, :3] - anchor_log[ticks[0], :3]
    truth_in_reference = (root_pos - root_pos[0]) @ rotation  # plant world -> reference world
    anchor_error = float(np.linalg.norm(estimate[:, :2] - truth_in_reference[:, :2], axis=1).max())
    played = targets[ticks]
    ankles = [i for i, n in enumerate(isaac_joint_names) if "ankle" in n]
    # Isaac's smoothness metrics at the control step (dt = 20 ms): finite
    # differences of the FK body positions, mean over bodies, mean over ticks.
    dt = 1.0 / 50.0
    body_vel = np.diff(robot_bodies, axis=0) / dt
    body_acc = np.diff(body_vel, axis=0) / dt
    body_jerk = np.diff(body_acc, axis=0) / dt
    ref_vel = np.diff(reference_bodies, axis=0) / dt
    ref_acc = np.diff(ref_vel, axis=0) / dt
    acc_distance = np.linalg.norm(body_acc - ref_acc, axis=-1).mean(axis=-1)
    raw_action = (played - np.asarray(action.default_joint_pos)) / np.asarray(action.action_scale) if action is not None else None
    # Commanded target and measured joints against the training soft limits.
    ankle_excess_max = target_beyond_pct = ankle_target_beyond_pct = measured_excess_max = nan
    if action is not None and action.joint_limits_lower and action.joint_limits_upper:
        lower = np.asarray(action.joint_limits_lower, np.float64)
        upper = np.asarray(action.joint_limits_upper, np.float64)
        target_excess = np.maximum(np.maximum(played - upper, lower - played), 0.0)
        ankle_excess_max = float(target_excess[:, ankles].max()) if ankles else nan
        target_beyond_pct = float(100.0 * (target_excess > 0.0).mean())
        ankle_target_beyond_pct = float(100.0 * (target_excess[:, ankles] > 0.0).mean()) if ankles else nan
        rows_q = plant_joints[int(rows[0]):int(rows[-1]) + 1]
        measured_excess = np.maximum(np.maximum(rows_q - upper, lower - rows_q), 0.0)
        measured_excess_max = float(measured_excess.max())
    action_delta = float(np.linalg.norm(np.diff(raw_action, axis=0), axis=-1).mean()) if raw_action is not None else nan
    # PD torque on the 500 Hz plant rows between consecutive control ticks,
    # target of the last tick held (the writer republishes it every 2 ms).
    torque_l2 = energy = nan
    if action is not None and action.stiffness and action.damping:
        kp = np.asarray(action.stiffness, np.float64)
        kd = np.asarray(action.damping, np.float64)
        effort = np.asarray(action.effort_limit, np.float64) if action.effort_limit else None
        period = float(states["sample_period_seconds"]) if "sample_period_seconds" in states.files else 0.002
        first, last = int(rows[0]), int(rows[-1])
        q_rows = plant_joints[first:last + 1]
        qdot_rows = np.gradient(q_rows, period, axis=0)
        held = np.empty_like(q_rows)
        bounds = np.append(rows, last + 1)
        for k in range(ticks.size):
            held[bounds[k] - first:bounds[k + 1] - first] = targets[ticks[k]]
        tau = kp * (held - q_rows) - kd * qdot_rows
        if effort is not None:
            tau = np.clip(tau, -effort, effort)
        torque_l2 = float((tau ** 2).sum(axis=1).mean())
        energy = float(np.abs(tau * qdot_rows).sum(axis=1).mean())
    return MotionGrade(
        motion=motion_name,
        seed=seed,
        passed=bool(result.get("passed", False)),
        ticks=int(len(controller_joints)),
        reference_ticks=int(ticks.size),
        mpjpe_l_mm=float(local),
        mpjpe_g_mm=float(global_),
        drift_max_m=drift,
        ankle_target_dither=float(np.mean([_second_difference(played[:, j]) for j in ankles])),
        all_target_dither=float(np.mean([_second_difference(played[:, j]) for j in range(played.shape[1])])),
        ankle_target_excess_max_rad=ankle_excess_max,
        target_beyond_limit_pct=target_beyond_pct,
        ankle_target_beyond_limit_pct=ankle_target_beyond_pct,
        measured_excess_max_rad=measured_excess_max,
        anchor_error_max_m=anchor_error,
        body_acc_mps2=float(np.linalg.norm(body_acc, axis=-1).mean()),
        body_jerk_mps3=float(np.linalg.norm(body_jerk, axis=-1).mean()),
        tracking_acceleration_distance_mps2=float(acc_distance.mean()),
        action_delta_l2=action_delta,
        joint_torques_l2=torque_l2,
        energy_consumption=energy,
        anchor_position_source=source,
        live_anchor=live,
        directory=str(run),
    )


AGGREGATE_KEYS = (
    "mpjpe_l_mm", "mpjpe_g_mm", "drift_max_m", "ankle_target_dither", "all_target_dither",
    "ankle_target_excess_max_rad", "target_beyond_limit_pct", "ankle_target_beyond_limit_pct",
    "measured_excess_max_rad",
    "anchor_error_max_m", "body_acc_mps2", "body_jerk_mps3",
    "tracking_acceleration_distance_mps2", "action_delta_l2", "joint_torques_l2",
    "energy_consumption",
)
# Per-motion maxima that also get a board-wide maximum, next to the mean.
MAX_KEYS = ("ankle_target_excess_max_rad", "measured_excess_max_rad")


def grade_rehearsal(output: str | Path, *, reference_root: str = "", mjcf: str = "") -> dict:
    """Every motion under a `rehearse --output` directory; writes grade.tsv / grade.json."""
    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.reference import ReferenceArrays

    output = Path(output)
    summary_path = output / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text())
    grades: list[MotionGrade] = []
    reference = None
    names: list[str] | None = None
    action = None
    for row in summary["rows"]:
        job = yaml.safe_load(Path(row["job"]).read_text())
        root = reference_root or job["reference_root"]
        model = mjcf or job.get("mjcf", "")
        if not model:
            raise ValueError("the job carries no mjcf; pass --mjcf")
        if reference is None:
            reference = ReferenceArrays(root)
            action = PolicyBundle.load(job["bundle"]).manifest.action
            names = list(action.isaac_joint_names)
        motion_dir = Path(row["job"]).parent
        for run in sorted((motion_dir / "sim").glob("seed_*")):
            seed = int(run.name.split("_")[-1])
            grades.append(grade_run(
                run, reference=reference, motion_name=row["motion"], isaac_joint_names=names,
                mjcf=model, seed=seed, action=action,
            ))
    fields = [f for f in MotionGrade.__dataclass_fields__ if f != "directory"]
    lines = ["\t".join(fields + ["directory"])]
    for g in grades:
        d = asdict(g)
        lines.append("\t".join(
            f"{d[f]:.4f}" if isinstance(d[f], float) else str(d[f]) for f in fields
        ) + "\t" + d["directory"])
    (output / "grade.tsv").write_text("\n".join(lines) + "\n")

    def mean_of(key: str, subset: list[MotionGrade]) -> float:
        values = [getattr(g, key) for g in subset if np.isfinite(getattr(g, key))]
        return float(np.mean(values)) if values else float("nan")

    def max_of(key: str, subset: list[MotionGrade]) -> float:
        values = [getattr(g, key) for g in subset if np.isfinite(getattr(g, key))]
        return float(np.max(values)) if values else float("nan")

    passed = [g for g in grades if g.passed]
    aggregate = {
        "output": str(output),
        "motions": len(grades),
        "passed": len(passed),
        "anchor_position_sources": sorted({g.anchor_position_source for g in grades}),
        "all": {k: mean_of(k, grades) for k in AGGREGATE_KEYS},
        "passed_only": {k: mean_of(k, passed) for k in AGGREGATE_KEYS},
        "all_max": {k: max_of(k, grades) for k in MAX_KEYS},
        "passed_only_max": {k: max_of(k, passed) for k in MAX_KEYS},
        "rows": [asdict(g) for g in grades],
    }
    (output / "grade.json").write_text(json.dumps(aggregate, indent=1) + "\n")
    return aggregate
