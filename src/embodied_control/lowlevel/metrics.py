"""Tracking metrics that need a reference motion: MPJPE-L and MPJPE-G.

Formulas replicate the IsaacLab-Imitation protocol authority
(`imitation_experiments/lowlevel/evaluate_checkpoint.py:548-576`) exactly:

- MPJPE-G (global): mean over tracked bodies of the world-frame position
  error `|p_body - p_ref_body|`.
- MPJPE-L (root-relative, the paper's headline metric): subtract each side's
  own root (pelvis) position first — `|(p_body - p_root) -
  (p_ref_body - p_ref_root)|` — position subtraction only, no rotation into
  the root frame, mean over bodies. Micro-averaged over frames, reported in
  millimeters.

Robot body positions come from a forward-kinematics replay of the per-tick
telemetry (joint qpos + anchor pose) through the same MJCF the plant
simulated, entirely off the control path.
"""

from __future__ import annotations

from pathlib import Path
import os
import sys

import numpy as np

from embodied_control.lowlevel.bundle import ActionContract
from embodied_control.lowlevel.reference import ReferenceMotion

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def _set_replay_pose(data, qpos_address, joint_pos, anchor_pose_xyzw) -> None:
    data.qpos[0:3] = anchor_pose_xyzw[0:3]
    x, y, z, w = anchor_pose_xyzw[3:7]
    data.qpos[3:7] = [w, x, y, z]
    data.qpos[qpos_address] = joint_pos


def render_oracle_comparison_video(
    action: ActionContract,
    model_path: str | Path,
    motion: ReferenceMotion,
    telemetry: dict,
    output_path: str | Path,
    *,
    fps: int = 25,
    tick_stride: int = 2,
    width: int = 480,
    height: int = 360,
) -> dict:
    """Render full-horizon reference-left / policy-right native telemetry."""
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco

    from embodied_control.lowlevel.envs.mujoco import load_scene_model

    frames = np.asarray(telemetry["reference_frames"])
    joint_pos = np.asarray(telemetry["joint_position_log"])
    anchor = np.asarray(telemetry["anchor_pose_log"])
    valid = (
        (frames >= 0)
        & (frames < motion.length)
        & np.isfinite(joint_pos).all(axis=-1)
        & np.isfinite(anchor).all(axis=-1)
    )
    ticks = np.flatnonzero(valid)[:: max(1, int(tick_stride))]
    if not ticks.size:
        raise ValueError("telemetry has no frames that can be rendered")

    reference_model = load_scene_model(model_path)
    policy_model = load_scene_model(model_path)
    reference_data = mujoco.MjData(reference_model)
    policy_data = mujoco.MjData(policy_model)
    qpos_address = np.empty(action.width, dtype=np.int64)
    for isaac_index, name in enumerate(action.isaac_joint_names):
        joint_id = mujoco.mj_name2id(reference_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"model has no joint {name!r}")
        qpos_address[isaac_index] = reference_model.jnt_qposadr[joint_id]

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reference_renderer = mujoco.Renderer(
        reference_model, height=int(height), width=int(width)
    )
    policy_renderer = mujoco.Renderer(
        policy_model, height=int(height), width=int(width)
    )
    rendered = []
    try:
        for tick in ticks:
            frame = int(frames[tick])
            reference_pose = np.concatenate(
                [motion.anchor_pos_w[frame], motion.anchor_quat_w[frame]]
            )
            _set_replay_pose(
                reference_data,
                qpos_address,
                motion.joint_qpos[frame],
                reference_pose,
            )
            _set_replay_pose(policy_data, qpos_address, joint_pos[tick], anchor[tick])
            mujoco.mj_forward(reference_model, reference_data)
            mujoco.mj_forward(policy_model, policy_data)
            reference_renderer.update_scene(reference_data, camera="track")
            policy_renderer.update_scene(policy_data, camera="track")
            rendered.append(
                np.concatenate(
                    [reference_renderer.render(), policy_renderer.render()], axis=1
                ).copy()
            )
    finally:
        reference_renderer.close()
        policy_renderer.close()
    imageio.mimwrite(output_path, rendered, fps=int(fps), quality=8)
    return {
        "path": str(output_path),
        "frames": len(rendered),
        "fps": int(fps),
        "layout": "reference_left_policy_right",
    }


def fk_body_positions(
    model_path: str | Path,
    action: ActionContract,
    joint_pos: np.ndarray,
    anchor_pose_xyzw: np.ndarray,
    body_names: tuple[str, ...],
) -> np.ndarray:
    """Replay `[T, 29]` joint qpos + `[T, 7]` root pose; return `[T, B, 3]`."""
    import mujoco

    from embodied_control.lowlevel.envs.mujoco import load_scene_model

    model = load_scene_model(model_path)
    data = mujoco.MjData(model)
    qpos_address = np.empty(action.width, dtype=np.int64)
    for isaac_index, name in enumerate(action.isaac_joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"model has no joint {name!r}")
        qpos_address[isaac_index] = model.jnt_qposadr[joint_id]
    body_ids = []
    for name in body_names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"model has no body {name!r}")
        body_ids.append(body_id)
    body_ids = np.asarray(body_ids, dtype=np.int64)

    joint_pos = np.asarray(joint_pos, dtype=np.float64)
    anchor_pose_xyzw = np.asarray(anchor_pose_xyzw, dtype=np.float64)
    ticks = joint_pos.shape[0]
    positions = np.empty((ticks, len(body_ids), 3), dtype=np.float32)
    for tick in range(ticks):
        data.qpos[0:3] = anchor_pose_xyzw[tick, 0:3]
        # MuJoCo stores WXYZ; telemetry carries XYZW.
        x, y, z, w = anchor_pose_xyzw[tick, 3:7]
        data.qpos[3:7] = [w, x, y, z]
        data.qpos[qpos_address] = joint_pos[tick]
        mujoco.mj_kinematics(model, data)
        positions[tick] = data.xpos[body_ids]
    return positions


def compute_mpjpe(
    robot_body_pos: np.ndarray,
    robot_root_pos: np.ndarray,
    reference_body_pos: np.ndarray,
    reference_root_pos: np.ndarray,
) -> dict:
    """Per-frame and micro-averaged MPJPE-G / MPJPE-L in millimeters."""
    robot_body_pos = np.asarray(robot_body_pos, dtype=np.float64)
    reference_body_pos = np.asarray(reference_body_pos, dtype=np.float64)
    if robot_body_pos.shape != reference_body_pos.shape:
        raise ValueError(
            f"body arrays disagree: {robot_body_pos.shape} vs "
            f"{reference_body_pos.shape}"
        )
    global_mm = (
        np.linalg.norm(robot_body_pos - reference_body_pos, axis=-1).mean(axis=-1)
        * 1000.0
    )
    robot_rel = robot_body_pos - np.asarray(robot_root_pos)[:, None, :]
    reference_rel = reference_body_pos - np.asarray(reference_root_pos)[:, None, :]
    local_mm = np.linalg.norm(robot_rel - reference_rel, axis=-1).mean(axis=-1) * 1000.0
    return {
        "frames": int(global_mm.shape[0]),
        "mpjpe_g_mm": float(global_mm.mean()),
        "mpjpe_l_mm": float(local_mm.mean()),
        "mpjpe_g_mm_p95": float(np.percentile(global_mm, 95)),
        "mpjpe_l_mm_p95": float(np.percentile(local_mm, 95)),
        "per_frame_mpjpe_g_mm": global_mm.astype(np.float32),
        "per_frame_mpjpe_l_mm": local_mm.astype(np.float32),
    }


def compute_sonic_success(
    robot_body_pos: np.ndarray,
    robot_anchor_pose: np.ndarray,
    reference_body_pos: np.ndarray,
    reference_anchor_pos: np.ndarray,
    reference_anchor_quat: np.ndarray,
    body_names: tuple[str, ...],
) -> dict:
    """Apply the SONIC release thresholds to a full, aligned rollout."""
    robot_body_pos = np.asarray(robot_body_pos, dtype=np.float64)
    robot_anchor_pose = np.asarray(robot_anchor_pose, dtype=np.float64)
    reference_body_pos = np.asarray(reference_body_pos, dtype=np.float64)
    reference_anchor_pos = np.asarray(reference_anchor_pos, dtype=np.float64)
    reference_anchor_quat = np.asarray(reference_anchor_quat, dtype=np.float64)
    if robot_body_pos.shape != reference_body_pos.shape:
        raise ValueError("robot and reference body arrays disagree")
    if robot_anchor_pose.shape != (robot_body_pos.shape[0], 7):
        raise ValueError("robot anchor pose must have shape [T, 7]")

    anchor_z_error = np.abs(robot_anchor_pose[:, 2] - reference_anchor_pos[:, 2])
    robot_quat = robot_anchor_pose[:, 3:7]
    robot_quat /= np.linalg.norm(robot_quat, axis=-1, keepdims=True)
    reference_quat = reference_anchor_quat / np.linalg.norm(
        reference_anchor_quat, axis=-1, keepdims=True
    )
    dot = np.abs(np.sum(robot_quat * reference_quat, axis=-1)).clip(0.0, 1.0)
    anchor_ori_error = 2.0 * np.arccos(dot)

    ee_names = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    try:
        ee_indices = [body_names.index(name) for name in ee_names]
    except ValueError as exc:
        raise ValueError(
            "tracked bodies do not contain the four SONIC end effectors"
        ) from exc
    ee_z_error = np.abs(
        robot_body_pos[:, ee_indices, 2] - reference_body_pos[:, ee_indices, 2]
    ).max(axis=1)
    failures = {
        "anchor_pos": anchor_z_error > 0.25,
        "anchor_ori": np.square(anchor_ori_error) > 1.0,
        "ee_body_pos": ee_z_error > 0.25,
    }
    failed_ticks = [
        (int(np.flatnonzero(values)[0]), name)
        for name, values in failures.items()
        if values.any()
    ]
    first_failure = min(failed_ticks) if failed_ticks else None
    return {
        "success": first_failure is None,
        "failure_tick": None if first_failure is None else first_failure[0],
        "failure_cause": None if first_failure is None else first_failure[1],
        "max_anchor_z_error_m": float(anchor_z_error.max()),
        "max_anchor_ori_error_rad": float(anchor_ori_error.max()),
        "max_ee_z_error_m": float(ee_z_error.max()),
    }


def _aligned_oracle_kinematics(
    action: ActionContract,
    model_path: str | Path,
    motion: ReferenceMotion,
    telemetry: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if motion.body_pos_w is None or not motion.body_names:
        raise ValueError(
            "the reference tree carries no tracked-body positions; MPJPE needs "
            "body_pos_w — re-prepare the reference arrays with body tracking"
        )
    frames = np.asarray(telemetry["reference_frames"])
    joint_pos = np.asarray(telemetry["joint_position_log"])
    anchor = np.asarray(telemetry["anchor_pose_log"])
    valid = (
        (frames >= 0)
        & (frames < motion.length)
        & np.isfinite(joint_pos).all(axis=-1)
        & np.isfinite(anchor).all(axis=-1)
    )
    if not valid.any():
        raise ValueError("telemetry has no valid reference-aligned ticks")
    frames = frames[valid]
    robot_bodies = fk_body_positions(
        model_path, action, joint_pos[valid], anchor[valid], motion.body_names
    )
    return frames, anchor[valid], robot_bodies, valid


def oracle_tracking_metrics(
    action: ActionContract,
    model_path: str | Path,
    motion: ReferenceMotion,
    telemetry: dict,
) -> dict:
    """MPJPE-L/G for an oracle run, aligned by the recorded reference frames."""
    frames, anchor, robot_bodies, valid = _aligned_oracle_kinematics(
        action, model_path, motion, telemetry
    )
    result = compute_mpjpe(
        robot_bodies,
        anchor[:, 0:3],
        motion.body_pos_w[frames],
        motion.anchor_pos_w[frames],
    )
    result["ticks_evaluated"] = int(valid.sum())
    result["ticks_total"] = int(len(valid))
    result["tracked_bodies"] = list(motion.body_names)
    per_frame_g = result.pop("per_frame_mpjpe_g_mm")
    per_frame_l = result.pop("per_frame_mpjpe_l_mm")
    result["per_frame"] = {
        "reference_frames": frames.astype(np.int32),
        "mpjpe_g_mm": per_frame_g,
        "mpjpe_l_mm": per_frame_l,
    }
    return result


def sonic_success_metrics(
    action: ActionContract,
    model_path: str | Path,
    motion: ReferenceMotion,
    telemetry: dict,
) -> dict:
    """Score a full EC rollout with the SONIC release success criterion."""
    frames, anchor, robot_bodies, _ = _aligned_oracle_kinematics(
        action, model_path, motion, telemetry
    )
    expected_frames = np.arange(motion.length - 1, dtype=frames.dtype)
    complete = np.array_equal(frames, expected_frames)
    result = compute_sonic_success(
        robot_bodies,
        anchor,
        motion.body_pos_w[frames],
        motion.anchor_pos_w[frames],
        motion.anchor_quat_w[frames],
        motion.body_names,
    )
    result["complete_motion"] = bool(complete)
    if not complete:
        result["success"] = False
        result["failure_cause"] = "incomplete_motion"
    return result


__all__ = [
    "compute_mpjpe",
    "compute_sonic_success",
    "fk_body_positions",
    "oracle_tracking_metrics",
    "render_oracle_comparison_video",
    "sonic_success_metrics",
]
