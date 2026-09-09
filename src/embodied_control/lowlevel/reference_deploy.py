"""Conservative endpoint screening, independent of checkpoint tracking quality.

Passing these checks does not establish dynamic balance or hardware readiness.
Missing measurements fail closed. A matching plant rehearsal remains required.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from embodied_control.lowlevel.reference import ReferenceMotion
from embodied_control.robot.gates import gravity_from_quaternion_xyzw, tilt_degrees

STANCE_PELVIS_HEIGHT_M = 0.79
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


@dataclass(frozen=True)
class DeployLimits:
    """Endpoint limits; passing them does not certify a motion."""

    root_speed_m_s: float = 0.15
    joint_speed_rad_s: float = 2.0
    pelvis_height_low_m: float = STANCE_PELVIS_HEIGHT_M - 0.09
    pelvis_height_high_m: float = STANCE_PELVIS_HEIGHT_M + 0.06
    tilt_degrees: float = 20.0
    # Ankle height symmetry is a screening proxy, not measured foot contact.
    foot_height_difference_m: float = 0.05


DEFAULT_LIMITS = DeployLimits()


@dataclass(frozen=True)
class DeployVerdict:
    name: str
    frames: int
    deployable: bool
    reasons: tuple[str, ...]
    stats: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"deployable": self.deployable, "reasons": list(self.reasons),
                "stats": {k: float(v) for k, v in self.stats.items()}}


def classify_motion(
    motion: ReferenceMotion,
    *,
    stance: list[float] | np.ndarray | None = None,
    limits: DeployLimits = DEFAULT_LIMITS,
    fps: float = 50.0,
) -> DeployVerdict:
    """Measure both endpoints; stance deviation is descriptive, not a gate."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    frames = motion.length
    if frames < 2:
        return DeployVerdict(motion.name, frames, False, ("at least two frames required",))
    for array in (motion.joint_qpos, motion.anchor_pos_w, motion.anchor_quat_w,
                  motion.joint_qvel, motion.body_pos_w):
        if array is not None and not np.isfinite(array).all():
            return DeployVerdict(motion.name, frames, False, ("non-finite reference data",))
    if np.any(np.abs(np.linalg.norm(motion.anchor_quat_w, axis=1) - 1.0) > .01):
        return DeployVerdict(motion.name, frames, False, ("invalid anchor quaternion",))
    displacement = np.linalg.norm(motion.anchor_pos_w[:, :2] - motion.anchor_pos_w[0, :2], axis=1)
    stats = {"frames": float(frames), "net_displacement_m": float(displacement[-1]),
             "max_displacement_m": float(displacement.max())}
    reasons = []
    if motion.joint_qvel is None:
        reasons.append("missing joint velocities")
    foot_ids = None
    if motion.body_pos_w is not None and all(name in motion.body_names for name in FOOT_BODIES):
        foot_ids = [motion.body_names.index(name) for name in FOOT_BODIES]
    else:
        reasons.append("missing foot positions for support screening")
    for label, index, adjacent in (("start", 0, 1), ("end", frames - 1, frames - 2)):
        speed = float(np.linalg.norm(motion.anchor_pos_w[index] - motion.anchor_pos_w[adjacent]) * fps)
        stats[f"{label}_root_speed_m_s"] = speed
        if speed > limits.root_speed_m_s:
            reasons.append(f"{label} root speed {speed:.2f} m/s exceeds {limits.root_speed_m_s:.2f}")
        if motion.joint_qvel is not None:
            joint_speed = float(np.abs(motion.joint_qvel[index]).max(initial=0))
            stats[f"{label}_joint_speed_rad_s"] = joint_speed
            if joint_speed > limits.joint_speed_rad_s:
                reasons.append(f"{label} joint speed {joint_speed:.2f} rad/s exceeds {limits.joint_speed_rad_s:.2f}")
        height = float(motion.anchor_pos_w[index, 2])
        stats[f"{label}_pelvis_height_m"] = height
        if not limits.pelvis_height_low_m <= height <= limits.pelvis_height_high_m:
            reasons.append(f"{label} pelvis height {height:.3f} m outside stance range")
        tilt = tilt_degrees(gravity_from_quaternion_xyzw(motion.anchor_quat_w[index].tolist()), [0., 0., -1.])
        stats[f"{label}_tilt_degrees"] = tilt
        if tilt > limits.tilt_degrees:
            reasons.append(f"{label} tilt {tilt:.1f} deg exceeds {limits.tilt_degrees:.1f}")
        if foot_ids is not None:
            difference = float(abs(motion.body_pos_w[index, foot_ids[0], 2] - motion.body_pos_w[index, foot_ids[1], 2]))
            stats[f"{label}_foot_height_difference_m"] = difference
            if difference > limits.foot_height_difference_m:
                reasons.append(f"{label} foot height difference {difference:.3f} m exceeds {limits.foot_height_difference_m:.3f}")
        if stance is not None:
            stance_array = np.asarray(stance)
            if stance_array.shape != motion.joint_qpos[index].shape:
                raise ValueError("stance and reference joint dimensions disagree")
            stats[f"{label}_stance_deviation_rad"] = float(np.abs(motion.joint_qpos[index] - stance_array).max(initial=0))
    return DeployVerdict(motion.name, frames, not reasons, tuple(reasons), stats)


def summarize(verdicts: list[DeployVerdict]) -> str:
    lines = [f"{'deployable' if v.deployable else 'training':12} {v.name}  {v.frames} frames"
             + (f"  ({'; '.join(v.reasons)})" if v.reasons else "")
             for v in sorted(verdicts, key=lambda v: (not v.deployable, v.name))]
    lines.append(f"{sum(v.deployable for v in verdicts)}/{len(verdicts)} pass endpoint screening")
    return '\n'.join(lines)
