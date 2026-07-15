"""Shared LIBERO proprio-extraction math (transport/libero_proprio.py) --
pure stdlib, used by both the OpenPI and GR00T translation layers. No numpy/
network needed, so this runs in the light default env unlike most of the
real-policy adapter tests."""

from __future__ import annotations

import math

import pytest

from embodied_control.transport.libero_proprio import (
    eef_pose_and_gripper,
    extract_named,
    quat2axisangle,
)


def test_quat2axisangle_identity_quaternion_is_zero_rotation():
    assert quat2axisangle([0.0, 0.0, 0.0, 1.0]) == [0.0, 0.0, 0.0]


def test_quat2axisangle_matches_robosuite_reference_values():
    # 90-degree rotation about the z-axis: xyzw = (0, 0, sin(45deg), cos(45deg))
    result = quat2axisangle([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    assert result[0] == pytest.approx(0.0, abs=1e-9)
    assert result[1] == pytest.approx(0.0, abs=1e-9)
    assert result[2] == pytest.approx(math.pi / 2, abs=1e-9)  # 90 degrees in radians


def test_extract_named_pulls_indexed_fields_in_order():
    names = ["robot0_eef_pos[0]", "robot0_eef_pos[1]", "robot0_eef_pos[2]", "unrelated[0]"]
    values = [0.1, 0.2, 0.3, 999.0]
    assert extract_named(names, values, "robot0_eef_pos") == [0.1, 0.2, 0.3]


def test_extract_named_falls_back_to_bare_key():
    assert extract_named(["state"], [7.0], "state") == [7.0]


def test_eef_pose_and_gripper_builds_verified_split():
    proprio = {
        "names": [
            "robot0_eef_pos[0]", "robot0_eef_pos[1]", "robot0_eef_pos[2]",
            "robot0_eef_quat[0]", "robot0_eef_quat[1]", "robot0_eef_quat[2]", "robot0_eef_quat[3]",
            "robot0_gripper_qpos[0]", "robot0_gripper_qpos[1]",
        ],
        "values": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.04, -0.04],
    }
    eef_pos, axisangle, gripper = eef_pose_and_gripper(proprio)
    assert eef_pos == [0.1, 0.2, 0.3]
    assert axisangle == [0.0, 0.0, 0.0]  # identity quat
    assert gripper == [0.04, -0.04]


def test_eef_pose_and_gripper_raises_on_missing_fields():
    with pytest.raises(ValueError, match="robot0_eef_pos"):
        eef_pose_and_gripper({"names": ["some_other_field"], "values": [1.0]})
