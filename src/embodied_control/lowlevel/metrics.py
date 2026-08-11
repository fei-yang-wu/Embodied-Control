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

import numpy as np

from embodied_control.lowlevel.bundle import ActionContract
from embodied_control.lowlevel.reference import ReferenceMotion


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
    local_mm = (
        np.linalg.norm(robot_rel - reference_rel, axis=-1).mean(axis=-1) * 1000.0
    )
    return {
        "frames": int(global_mm.shape[0]),
        "mpjpe_g_mm": float(global_mm.mean()),
        "mpjpe_l_mm": float(local_mm.mean()),
        "mpjpe_g_mm_p95": float(np.percentile(global_mm, 95)),
        "mpjpe_l_mm_p95": float(np.percentile(local_mm, 95)),
        "per_frame_mpjpe_g_mm": global_mm.astype(np.float32),
        "per_frame_mpjpe_l_mm": local_mm.astype(np.float32),
    }


def oracle_tracking_metrics(
    action: ActionContract,
    model_path: str | Path,
    motion: ReferenceMotion,
    telemetry: dict,
) -> dict:
    """MPJPE-L/G for an oracle run, aligned by the recorded reference frames."""
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
    result = compute_mpjpe(
        robot_bodies,
        anchor[valid, 0:3],
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


__all__ = ["compute_mpjpe", "fk_body_positions", "oracle_tracking_metrics"]
