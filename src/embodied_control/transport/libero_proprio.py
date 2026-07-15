"""Shared LIBERO raw-proprio extraction, used by both the OpenPI and GR00T
translation layers.

Both checkpoints' own reference eval scripts derive their state
representation from the *identical* LIBERO raw obs fields
(``robot0_eef_pos``/``robot0_eef_quat``/``robot0_gripper_qpos``) via the same
robosuite ``quat2axisangle`` conversion -- verified independently against
each project's own source (``examples/libero/main.py`` for OpenPI,
``gr00t/eval/sim/LIBERO/libero_env.py`` for GR00T, both 2026-07-15). Not a
coincidence worth duplicating: both wrap the same underlying LIBERO/robosuite
env and both need it in the same representation for the same reason (a
7-DoF end-effector delta-pose action space needs orientation as a compact
3-vector, not a 4-element quaternion).
"""

from __future__ import annotations

import math


def quat2axisangle(quat: list[float]) -> list[float]:
    """xyzw quaternion -> 3-vector axis-angle (robosuite's own conversion,
    copied via each project's reference eval script)."""
    qx, qy, qz, qw = quat
    qw = min(1.0, max(-1.0, qw))
    den = math.sqrt(1.0 - qw * qw)
    if math.isclose(den, 0.0):
        return [0.0, 0.0, 0.0]
    scale = (2.0 * math.acos(qw)) / den
    return [qx * scale, qy * scale, qz * scale]


def extract_named(names: list[str], values: list[float], prefix: str) -> list[float]:
    """Pull ``prefix[0]``, ``prefix[1]``, ... (or bare ``prefix`` if
    unindexed) out of our flattened proprio names/values, in index order --
    the inverse of libero_eval.py::_wire_observation's flattening."""
    indexed = sorted(
        ((int(n[len(prefix) + 1:-1]), v) for n, v in zip(names, values) if n.startswith(prefix + "[")),
    )
    if indexed:
        return [v for _, v in indexed]
    return [v for n, v in zip(names, values) if n == prefix]


def eef_pose_and_gripper(proprio: dict) -> tuple[list[float], list[float], list[float]]:
    """Our wire proprio -> ``(eef_pos[3], axisangle[3], gripper_qpos[2])``.
    Raises ``ValueError`` if any piece isn't present with the expected shape
    (a real, checkpoint-independent LIBERO wiring bug, not a translation
    ambiguity, if this fires)."""
    names, values = proprio.get("names") or [], proprio.get("values") or []
    eef_pos = extract_named(names, values, "robot0_eef_pos")
    eef_quat = extract_named(names, values, "robot0_eef_quat")
    gripper_qpos = extract_named(names, values, "robot0_gripper_qpos")
    if len(eef_pos) != 3 or len(eef_quat) != 4 or len(gripper_qpos) != 2:
        raise ValueError(
            "expected robot0_eef_pos(3)/robot0_eef_quat(4)/robot0_gripper_qpos(2) "
            f"in the wire proprio; got {len(eef_pos)}/{len(eef_quat)}/{len(gripper_qpos)} "
            f"-- names present: {names}"
        )
    return eef_pos, quat2axisangle(eef_quat), gripper_qpos
