"""Macro-window construction for the DiffSR (latent) skill encoder.

The encoder input is one window of macro states: the frame at the renewal
cursor followed by `window_steps` lookahead frames taken at
`cursor + stride*k`, flattened slot-major. Two window flavours exist, and both
live here so the certified `OnboardEncoderPublisher` and the latent playground
build byte-identical inputs:

- precomputed (`macro_states`): the frames do not depend on the live robot
  (`expert_heading` bundles, synthetic plumbing runs);
- robot-anchored (`macro_anchor_mode == "robot"`): each slot is
  `[joint qpos J | expert anchor pos 3 | expert anchor ori rot6d 6]`, with the
  expert anchor world pose re-expressed in a live anchor frame.
"""

from __future__ import annotations

import numpy as np

from embodied_control.lowlevel.maths import (
    heading_anchor_frame,
    rot6d_from_quat,
    subtract_frame,
)
from embodied_control.lowlevel.reference import ReferenceMotion

ROBOT_ANCHOR_MODES = ("robot", "robot_heading")


def window_indices(
    cursor: int, length: int, window_steps: int, stride: int = 1
) -> list[int]:
    """Frame indices for one window; the tail clamps to the last frame."""
    last = int(length) - 1
    return [min(int(cursor) + int(stride) * k, last) for k in range(int(window_steps) + 1)]


def fill_precomputed_window(
    states: np.ndarray, indices: list[int], out: np.ndarray
) -> np.ndarray:
    """Write the selected precomputed macro states into `out`, slot-major."""
    width = int(states.shape[1])
    for slot, index in enumerate(indices):
        out[slot * width : (slot + 1) * width] = states[index]
    return out


def fill_robot_anchored_window(
    motion: ReferenceMotion,
    indices: list[int],
    anchor_pos_w: np.ndarray,
    anchor_quat_w: np.ndarray,
    state_dim: int,
    out: np.ndarray,
    anchor_mode: str = "robot",
) -> np.ndarray:
    """Write a robot-anchored window into `out`, slot-major.

    `anchor_pos_w` / `anchor_quat_w` (XYZW) are the frame the expert anchor is
    re-expressed in — the live robot anchor on a rollout, or the reference's
    own anchor when encoding offline under a perfect-tracking assumption.

    `anchor_mode="robot"` cancels the live anchor's full pose.
    `anchor_mode="robot_heading"` (SONIC v1.1) cancels only its heading twist
    and xy origin, so the reference keeps its height and its tilt relative to
    gravity in the encoder input. The two produce different windows from the
    same robot state, so the bundle's mode must drive this argument.
    """
    if anchor_mode not in ROBOT_ANCHOR_MODES:
        raise ValueError(
            f"anchor_mode must be one of {ROBOT_ANCHOR_MODES}, got {anchor_mode!r}"
        )
    if anchor_mode == "robot_heading":
        anchor_pos_w, anchor_quat_w = heading_anchor_frame(anchor_pos_w, anchor_quat_w)
    joints = motion.joint_qpos
    joint_count = int(joints.shape[1])
    if state_dim != joint_count + 9:
        raise ValueError(
            f"state_dim {state_dim} != qpos {joint_count} + 3 anchor pos + 6 rot6d"
        )
    for slot, index in enumerate(indices):
        rel_pos, rel_quat = subtract_frame(
            anchor_pos_w,
            anchor_quat_w,
            motion.anchor_pos_w[index],
            motion.anchor_quat_w[index],
        )
        base = slot * state_dim
        out[base : base + joint_count] = joints[index]
        out[base + joint_count : base + joint_count + 3] = rel_pos
        out[base + joint_count + 3 : base + state_dim] = rot6d_from_quat(rel_quat)
    return out


__all__ = [
    "ROBOT_ANCHOR_MODES",
    "fill_precomputed_window",
    "fill_robot_anchored_window",
    "window_indices",
]
