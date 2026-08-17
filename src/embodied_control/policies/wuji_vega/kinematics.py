"""Damped-least-squares IK and grasp waypoint generation."""

from __future__ import annotations

import mujoco
import numpy as np


def _joint_indices(
    model: mujoco.MjModel, joint_names: list[str]
) -> tuple[list[int], list[int], list[int]]:
    joint_ids = [model.joint(name).id for name in joint_names]
    qpos = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    dofs = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    return qpos, dofs, joint_ids


def _rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    delta = target @ current.T
    cosine = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ]
    )
    if angle < 1e-8:
        return 0.5 * skew
    sine = float(np.sin(angle))
    if abs(sine) < 1e-7:
        return np.zeros(3)
    return skew * (angle / (2.0 * sine))


def solve_position_ik(
    model: mujoco.MjModel,
    site_name: str,
    target: np.ndarray,
    joint_names: list[str],
    seed: np.ndarray | None = None,
) -> np.ndarray:
    data = mujoco.MjData(model)
    qpos, dofs, joint_ids = _joint_indices(model, joint_names)
    if seed is not None:
        data.qpos[qpos] = seed
    site_id = model.site(site_name).id
    for _ in range(500):
        mujoco.mj_forward(model, data)
        error = target - data.site_xpos[site_id]
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jacobian = jacp[:, dofs]
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.01**2 * np.eye(3), error
        )
        norm = float(np.linalg.norm(delta))
        if norm > 0.12:
            delta *= 0.12 / norm
        data.qpos[qpos] += delta
        data.qpos[qpos] = np.clip(
            data.qpos[qpos],
            model.jnt_range[joint_ids, 0],
            model.jnt_range[joint_ids, 1],
        )
    mujoco.mj_forward(model, data)
    residual = float(np.linalg.norm(target - data.site_xpos[site_id]))
    if residual > 0.008:
        raise RuntimeError(f"palm position IK residual too large: {residual:.4f} m")
    return data.qpos[qpos].copy()


def solve_pose_ik(
    model: mujoco.MjModel,
    site_name: str,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    joint_names: list[str],
    seed: np.ndarray,
) -> np.ndarray:
    data = mujoco.MjData(model)
    qpos, dofs, joint_ids = _joint_indices(model, joint_names)
    data.qpos[qpos] = seed
    site_id = model.site(site_name).id
    orientation_weight = 0.35
    for _ in range(900):
        mujoco.mj_forward(model, data)
        position_error = target_position - data.site_xpos[site_id]
        rotation_error = _rotation_error(
            target_rotation, data.site_xmat[site_id].reshape(3, 3)
        )
        if (
            np.linalg.norm(position_error) < 2e-5
            and np.linalg.norm(rotation_error) < 3e-4
        ):
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jacobian = np.vstack(
            (jacp[:, dofs], orientation_weight * jacr[:, dofs])
        )
        error = np.concatenate(
            (position_error, orientation_weight * rotation_error)
        )
        delta = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 0.015**2 * np.eye(6), error
        )
        norm = float(np.linalg.norm(delta))
        if norm > 0.08:
            delta *= 0.08 / norm
        data.qpos[qpos] += delta
        data.qpos[qpos] = np.clip(
            data.qpos[qpos],
            model.jnt_range[joint_ids, 0],
            model.jnt_range[joint_ids, 1],
        )
    mujoco.mj_forward(model, data)
    position_residual = float(
        np.linalg.norm(target_position - data.site_xpos[site_id])
    )
    rotation_residual = float(
        np.linalg.norm(
            _rotation_error(
                target_rotation, data.site_xmat[site_id].reshape(3, 3)
            )
        )
    )
    if position_residual > 0.008 or rotation_residual > np.deg2rad(4.0):
        raise RuntimeError(
            "palm pose IK residual too large: "
            f"position={position_residual:.4f} m, "
            f"rotation={np.rad2deg(rotation_residual):.2f} deg"
        )
    return data.qpos[qpos].copy()


def _site_pose(
    model: mujoco.MjModel,
    site_name: str,
    joint_names: list[str],
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    data = mujoco.MjData(model)
    qpos, _, _ = _joint_indices(model, joint_names)
    data.qpos[qpos] = q
    mujoco.mj_forward(model, data)
    site_id = model.site(site_name).id
    return (
        data.site_xpos[site_id].copy(),
        data.site_xmat[site_id].reshape(3, 3).copy(),
    )


def plan_right_arm(
    model: mujoco.MjModel, cube_position: np.ndarray
) -> dict[str, np.ndarray]:
    joint_names = [f"R_arm_j{i}" for i in range(1, 8)]
    grasp_position = cube_position + np.array([-0.09, 0.0, 0.085])
    grasp = solve_position_ik(model, "right_palm", grasp_position, joint_names)
    _, grasp_rotation = _site_pose(model, "right_palm", joint_names, grasp)
    pregrasp = solve_pose_ik(
        model,
        "right_palm",
        grasp_position + np.array([0.0, 0.0, 0.045]),
        grasp_rotation,
        joint_names,
        grasp,
    )
    above = solve_pose_ik(
        model,
        "right_palm",
        grasp_position + np.array([0.0, 0.0, 0.28]),
        grasp_rotation,
        joint_names,
        pregrasp,
    )
    transit = solve_position_ik(
        model,
        "right_palm",
        np.array([0.70, -0.27, 1.10]),
        joint_names,
        seed=above,
    )
    home = solve_position_ik(
        model,
        "right_palm",
        np.array([0.65, -0.30, 1.10]),
        joint_names,
        seed=transit,
    )
    lift = solve_pose_ik(
        model,
        "right_palm",
        grasp_position + np.array([0.0, 0.0, 0.14]),
        grasp_rotation,
        joint_names,
        grasp,
    )
    return {
        "home": home,
        "transit": transit,
        "above": above,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "lift": lift,
    }


def quintic_blend(progress: float) -> float:
    progress = float(np.clip(progress, 0.0, 1.0))
    return progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)


__all__ = [
    "plan_right_arm",
    "quintic_blend",
    "solve_pose_ik",
    "solve_position_ik",
]
