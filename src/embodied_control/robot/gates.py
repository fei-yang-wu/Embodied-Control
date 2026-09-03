"""Pure gate arithmetic for the robot lifecycle.

Plain Python on plain lists: the light test env has no numpy, and every
check here is a max over 29 numbers. The lifecycle samples the robot and
hands the numbers in; nothing here touches DDS, time, or a thread.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GateResult:
    ok: bool
    detail: str
    values: dict[str, float | int | str | bool | list[float]] = field(
        default_factory=dict
    )


def max_abs(values: list[float]) -> float:
    return max((abs(float(v)) for v in values), default=0.0)


def max_abs_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return max((abs(float(x) - float(y)) for x, y in zip(a, b)), default=0.0)


def tilt_degrees(gravity_a: list[float], gravity_b: list[float]) -> float:
    """Angle between two projected-gravity vectors."""
    norm_a = math.sqrt(sum(float(v) * float(v) for v in gravity_a))
    norm_b = math.sqrt(sum(float(v) * float(v) for v in gravity_b))
    if norm_a == 0.0 or norm_b == 0.0:
        raise ValueError("gravity vector has zero norm")
    dot = sum(float(x) * float(y) for x, y in zip(gravity_a, gravity_b))
    cosine = max(-1.0, min(1.0, dot / (norm_a * norm_b)))
    return math.degrees(math.acos(cosine))


def gravity_from_quaternion_xyzw(quaternion: list[float]) -> list[float]:
    """Body-frame projected gravity for a world-frame orientation."""
    x, y, z, w = (float(v) for v in quaternion)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        raise ValueError("quaternion has zero norm")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return [
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        -(1.0 - 2.0 * (x * x + y * y)),
    ]


def pose_match(
    measured: list[float],
    target: list[float],
    tolerance: list[float],
) -> GateResult:
    """Per-joint |measured - target| against a per-joint tolerance."""
    if not (len(measured) == len(target) == len(tolerance)):
        raise ValueError("pose_match needs equal-length joint vectors")
    errors = [abs(float(m) - float(t)) for m, t in zip(measured, target)]
    worst = max(range(len(errors)), key=lambda i: errors[i] - float(tolerance[i]))
    exceeded = [i for i, e in enumerate(errors) if e > float(tolerance[i])]
    values = {
        "max_error_rad": max(errors),
        "worst_joint": worst,
        "exceeded_joints": [float(i) for i in exceeded],
        "errors_rad": errors,
    }
    if exceeded:
        return GateResult(
            False,
            f"{len(exceeded)} joint(s) outside tolerance; worst joint {worst} "
            f"off by {errors[worst]:.3f} rad (tolerance {float(tolerance[worst]):.3f})",
            values,
        )
    return GateResult(
        True, f"all joints within tolerance (max {max(errors):.3f} rad)", values
    )


def tilt_match(
    measured_gravity: list[float],
    reference_gravity: list[float],
    tolerance_degrees: float,
) -> GateResult:
    tilt = tilt_degrees(measured_gravity, reference_gravity)
    values = {"tilt_degrees": tilt}
    if tilt > tolerance_degrees:
        return GateResult(
            False,
            f"IMU tilt {tilt:.2f} deg from the reference frame exceeds "
            f"{tolerance_degrees:.2f} deg",
            values,
        )
    return GateResult(True, f"IMU tilt {tilt:.2f} deg", values)


def counter_advanced(
    before: int, after: int, minimum: int, name: str
) -> GateResult:
    delta = int(after) - int(before)
    values = {name: delta}
    if delta < minimum:
        return GateResult(
            False, f"{name} advanced by {delta}, needed {minimum}", values
        )
    return GateResult(True, f"{name} advanced by {delta}", values)


def unchanged(before: int, after: int, name: str) -> GateResult:
    delta = int(after) - int(before)
    if delta != 0:
        return GateResult(False, f"{name} rose by {delta}", {name: delta})
    return GateResult(True, f"no new {name}", {name: 0})
