import json

import numpy as np
import pytest

from embodied_control.lowlevel.bundle import CommandContract
from embodied_control.lowlevel.command_buffer import InProcessCommandBuffer
from embodied_control.lowlevel.contracts import RobotState
from embodied_control.lowlevel.maths import (
    quat_mul,
    quat_to_mat,
    rot6d_from_quat,
    subtract_frame,
    wxyz_to_xyzw,
)
from embodied_control.lowlevel.publishers.onboard_encoder import OnboardEncoderPublisher
from embodied_control.lowlevel.reference import ReferenceArrays, ReferenceMotion

SQ2 = np.sqrt(0.5).astype(np.float32) if hasattr(np.sqrt(0.5), "astype") else np.sqrt(0.5)
YAW90_XYZW = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def test_wxyz_to_xyzw_roundtrip():
    wxyz = np.array([0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(wxyz_to_xyzw(wxyz), [0.2, 0.3, 0.4, 0.1])


def test_quat_to_mat_yaw90():
    mat = quat_to_mat(YAW90_XYZW)
    np.testing.assert_allclose(mat @ [1, 0, 0], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(mat @ [0, 1, 0], [-1, 0, 0], atol=1e-6)


def test_rot6d_layout_is_first_two_columns_row_major():
    mat = quat_to_mat(YAW90_XYZW)
    six = rot6d_from_quat(YAW90_XYZW)
    expected = [mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1], mat[2, 0], mat[2, 1]]
    np.testing.assert_allclose(six, expected, atol=1e-6)


def test_subtract_frame_translation_and_rotation():
    rel_pos, rel_quat = subtract_frame(
        np.array([1.0, 2.0, 0.0]), YAW90_XYZW, np.array([1.0, 3.0, 0.0]), YAW90_XYZW
    )
    # World offset +Y, anchor yawed +90: in the anchor frame that is +X.
    np.testing.assert_allclose(rel_pos, [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(rel_quat, IDENTITY_XYZW, atol=1e-6)

    _rel_pos, rel_quat = subtract_frame(
        np.zeros(3), IDENTITY_XYZW, np.zeros(3), YAW90_XYZW
    )
    np.testing.assert_allclose(rel_quat, YAW90_XYZW, atol=1e-6)
    np.testing.assert_allclose(
        quat_mul(rel_quat, np.array([0, 0, -np.sqrt(0.5), np.sqrt(0.5)], np.float32)),
        IDENTITY_XYZW,
        atol=1e-6,
    )


def _write_reference_tree(root, frames=12, joints=29):
    rng = np.random.default_rng(0)
    qpos = np.zeros((frames, 7 + joints), dtype=np.float32)
    qpos[:, 2] = 0.76
    qpos[:, 3] = 1.0  # root quat WXYZ identity
    qpos[:, 7:] = rng.standard_normal((frames, joints)).astype(np.float32) * 0.1
    anchor_pos = np.cumsum(
        np.full((frames, 3), [0.01, 0.0, 0.0], dtype=np.float32), axis=0
    )
    anchor_quat = np.tile(np.array([0, 0, 0, 1.0], np.float32), (frames, 1))
    arrays = {
        "qpos": qpos,
        "anchor_pos_w": anchor_pos,
        "anchor_quat_w": anchor_quat,
    }
    specs = {}
    for name, value in arrays.items():
        memmap = np.memmap(
            root / f"{name}.memmap", dtype="float32", mode="w+", shape=value.shape
        )
        memmap[:] = value
        memmap.flush()
        specs[name] = {
            "shape": list(value.shape),
            "dtype": "float32",
            "quaternion_order": "xyzw" if name == "anchor_quat_w" else None,
        }
    manifest = {
        "format_version": 1,
        "key": {
            "joint_names": [f"j{i}" for i in range(joints)],
            "anchor_body": "pelvis",
            "arrays": specs,
        },
        "traj_info": {
            "start_index": [0, 5],
            "end_index": [5, frames],
            "ordered_traj_list": [["d", "motion_a", "t0"], ["d", "motion_b", "t0"]],
        },
    }
    (root / "reference_arrays_manifest.json").write_text(json.dumps(manifest))
    return arrays


def test_reference_arrays_loader(tmp_path):
    arrays = _write_reference_tree(tmp_path)
    ref = ReferenceArrays(tmp_path)
    assert ref.motion_names == ["motion_a", "motion_b"]
    motion = ref.motion("motion_b")
    assert motion.length == 7
    np.testing.assert_allclose(motion.joint_qpos, arrays["qpos"][5:, 7:])
    np.testing.assert_allclose(motion.anchor_pos_w, arrays["anchor_pos_w"][5:])
    with pytest.raises(KeyError):
        ref.motion("nope")


class SpyEncoder:
    def __init__(self):
        self.inputs = []

    def infer(self, obs, out=None):
        self.inputs.append(np.array(obs, copy=True))
        return np.zeros(6, dtype=np.float32)


def _robot_state(anchor_pos, anchor_quat):
    return RobotState(
        stamp=0.0,
        joint_pos=np.zeros(29, np.float32),
        joint_vel=np.zeros(29, np.float32),
        projected_gravity=np.array([0, 0, -1], np.float32),
        base_ang_vel=np.zeros(3, np.float32),
        anchor_pos_w=np.asarray(anchor_pos, np.float32),
        anchor_quat_w=np.asarray(anchor_quat, np.float32),
    )


def test_robot_anchored_window(tmp_path):
    joints = 29
    frames = 15
    rng = np.random.default_rng(1)
    joint_qpos = rng.standard_normal((frames, joints)).astype(np.float32)
    anchor_pos = np.tile(np.array([2.0, 0.0, 0.7], np.float32), (frames, 1))
    anchor_quat = np.tile(IDENTITY_XYZW, (frames, 1))
    motion = ReferenceMotion("m", joint_qpos, anchor_pos, anchor_quat)
    command = CommandContract(
        z_dim=6, phase_mode="none", phase_dim=0, hold_steps=3,
        state_dim=38, window_steps=9, horizon_steps=10,
        encoder_window_mode="intermediate", macro_frame_stride=1,
        macro_anchor_mode="robot",
    )
    encoder = SpyEncoder()
    publisher = OnboardEncoderPublisher(
        InProcessCommandBuffer(), encoder, command, motion=motion
    )
    # Robot anchored 1 m behind the expert, yawed +90 degrees.
    state = _robot_state([1.0, 0.0, 0.7], YAW90_XYZW)
    publisher.tick(0, 0.0, state)
    window = encoder.inputs[0].reshape(10, 38)
    np.testing.assert_allclose(window[0, :29], joint_qpos[0], atol=1e-6)
    # Expert is +1 m world X from the robot; the robot's +90 yaw maps that to -Y.
    np.testing.assert_allclose(window[0, 29:32], [0.0, -1.0, 0.0], atol=1e-6)
    yaw_minus90 = quat_to_mat(
        np.array([0, 0, -np.sqrt(0.5), np.sqrt(0.5)], np.float32)
    )
    expected_6d = [
        yaw_minus90[0, 0], yaw_minus90[0, 1], yaw_minus90[1, 0],
        yaw_minus90[1, 1], yaw_minus90[2, 0], yaw_minus90[2, 1],
    ]
    np.testing.assert_allclose(window[0, 32:38], expected_6d, atol=1e-6)
    np.testing.assert_allclose(window[3, :29], joint_qpos[3], atol=1e-6)


def test_robot_anchored_requires_state(tmp_path):
    motion = ReferenceMotion(
        "m",
        np.zeros((12, 29), np.float32),
        np.zeros((12, 3), np.float32),
        np.tile(IDENTITY_XYZW, (12, 1)),
    )
    command = CommandContract(
        z_dim=6, phase_mode="none", phase_dim=0, hold_steps=3,
        state_dim=38, window_steps=9, horizon_steps=10,
        encoder_window_mode="intermediate", macro_frame_stride=1,
        macro_anchor_mode="robot",
    )
    publisher = OnboardEncoderPublisher(
        InProcessCommandBuffer(), SpyEncoder(), command, motion=motion
    )
    with pytest.raises(ValueError, match="robot state"):
        publisher.tick(0, 0.0, None)