"""Quaternion and frame helpers. XYZW throughout, matching Isaac Lab 3.0.

NPZ/dataset quaternions are WXYZ and must pass through `wxyz_to_xyzw` exactly
once at load. rot6d is the first two rotation-matrix columns flattened
row-major: [R00, R01, R10, R11, R20, R21].
"""

from __future__ import annotations

import numpy as np


def wxyz_to_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    return quat[..., [1, 2, 3, 0]]


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    return quat / np.linalg.norm(quat, axis=-1, keepdims=True)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    out = quat.copy()
    out[..., :3] = -out[..., :3]
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float32), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float32), -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(quat_normalize(quat), -1, 0)
    row0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1)
    row1 = np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1)
    row2 = np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return np.stack([row0, row1, row2], axis=-2)


def rot6d_from_quat(quat: np.ndarray) -> np.ndarray:
    mat = quat_to_mat(quat)
    return mat[..., :, :2].reshape(*mat.shape[:-2], 6)


def rotate_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate `vec` from the world frame into the frame of `quat`."""
    mat = quat_to_mat(quat)
    return np.einsum("...ji,...j->...i", mat, np.asarray(vec, dtype=np.float32))


def subtract_frame(
    anchor_pos: np.ndarray,
    anchor_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Express a world target pose in the anchor frame (Isaac
    `subtract_frame_transforms` semantics)."""
    rel_pos = rotate_inverse(anchor_quat, np.asarray(target_pos) - np.asarray(anchor_pos))
    rel_quat = quat_mul(quat_conjugate(anchor_quat), target_quat)
    return rel_pos.astype(np.float32), rel_quat.astype(np.float32)


__all__ = [
    "quat_conjugate",
    "quat_mul",
    "quat_normalize",
    "quat_to_mat",
    "rot6d_from_quat",
    "rotate_inverse",
    "subtract_frame",
    "wxyz_to_xyzw",
]
