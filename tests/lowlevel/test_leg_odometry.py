"""Kinematic leg odometry: pelvis translation from joints and IMU only."""

from pathlib import Path

import numpy as np
import pytest

ec_native = pytest.importorskip("ec_native")
mujoco = pytest.importorskip("mujoco")

from test_dds_plant import DEFAULT_POSE, G1_JOINT_NAMES  # noqa: E402
from test_native_core import _axis_quaternion, _g1_mjcf_path  # noqa: E402

SOLE = np.array([[-0.05, 0.025, -0.035], [-0.05, -0.025, -0.035],
                 [0.12, 0.03, -0.035], [0.12, -0.03, -0.035]])


def _sole_points(model, data, joints, quat_xyzw):
    data.qpos[:3] = 0.0
    data.qpos[3] = quat_xyzw[3]
    data.qpos[4:7] = quat_xyzw[:3]
    for name, value in zip(G1_JOINT_NAMES, joints):
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_kinematics(model, data)
    out = []
    for body in ("left_ankle_roll_link", "right_ankle_roll_link"):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        out.append(data.xpos[b] + SOLE @ data.xmat[b].reshape(3, 3).T)
    return np.concatenate(out)


def test_leg_odometry_stands_still_and_reports_height():
    odometry = ec_native.LegOdometry(str(_g1_mjcf_path()), G1_JOINT_NAMES)
    identity = np.array([0, 0, 0, 1], np.float32)
    pose = np.asarray(DEFAULT_POSE, np.float32)
    first = odometry.update(pose, identity)
    for _ in range(20):
        position = odometry.update(pose, identity)
    np.testing.assert_allclose(position[:2], 0.0, atol=1e-9)
    # Pelvis height above the sole of a bent-knee stance: about 0.7 m.
    assert 0.6 < position[2] < 0.8
    assert position[2] == pytest.approx(first[2])
    assert odometry.stance_switches == 0


def test_leg_odometry_integrates_pelvis_motion_from_planted_sole_points():
    model = mujoco.MjModel.from_xml_path(str(_g1_mjcf_path()))
    data = mujoco.MjData(model)
    odometry = ec_native.LegOdometry(str(_g1_mjcf_path()), G1_JOINT_NAMES)
    identity = np.array([0, 0, 0, 1], np.float32)
    # Right foot lifted (hip pitch back, knee bent): only the left sole is
    # planted, so a pelvis yaw must read as the left foot's arc, reversed.
    lifted = list(DEFAULT_POSE)
    lifted[6] = -1.0   # right_hip_pitch (SDK order: right leg is 6..11)
    lifted[9] = 1.4    # right_knee
    lifted = np.asarray(lifted, np.float32)
    odometry.update(lifted, identity)
    yaw = np.asarray(_axis_quaternion(2, 12.0), np.float32)
    position = odometry.update(lifted, yaw)
    before = _sole_points(model, data, lifted, identity)
    after = _sole_points(model, data, lifted, yaw)
    contact = (after[:, 2] <= after[:, 2].min() + 0.005) & (before[:, 2] <= before[:, 2].min() + 0.005)
    assert contact[:4].sum() >= 2 and not contact[4:].any(), "only the left sole should be planted"
    expected = -np.median((after - before)[contact, :2], axis=0)
    np.testing.assert_allclose(position[:2], expected, atol=1e-6)
    assert np.linalg.norm(expected) > 0.01
    assert odometry.stance_foot == 0


def test_leg_odometry_rejects_non_finite_input():
    odometry = ec_native.LegOdometry(str(_g1_mjcf_path()), G1_JOINT_NAMES)
    bad = np.asarray(DEFAULT_POSE, np.float32)
    bad[3] = np.nan
    with pytest.raises(RuntimeError):
        odometry.update(bad, np.array([0, 0, 0, 1], np.float32))
    with pytest.raises(RuntimeError):
        ec_native.LegOdometry(str(_g1_mjcf_path()), ["not_a_joint"] + G1_JOINT_NAMES[1:])


def test_leg_odometry_tracks_a_physics_walk_within_a_few_percent_of_its_path():
    """A retargeted reference clip slides its feet, so the oracle is a
    physics rollout: the plant's true pelvis against the estimate from the
    plant's joints and orientation."""
    root = Path(__file__).resolve().parents[2]
    bundle = root / "assets/models/controller/action01rate04_66500m_ect"
    reference = root / "assets/models/reference/bones"
    if not (bundle / "manifest.json").is_file() or not (reference / "reference_arrays_manifest.json").is_file():
        pytest.skip("local G1 bundle or bones reference tree is absent")
    pytest.importorskip("torch")
    from embodied_control.lowlevel.latent import LatentPlayground

    playground = LatentPlayground(str(bundle), str(_g1_mjcf_path()), str(reference))
    name = "injured_R_leg_turn_walk_360_001_A069"
    motion = playground.motion(name)
    rollout = playground.rollout(
        playground.make_reference_source(motion, start_frame=0), motion=motion,
        max_steps=200, start_pose_frame=0, record_video=False,
    )
    names = list(playground.bundle.manifest.action.isaac_joint_names)
    odometry = ec_native.LegOdometry(str(_g1_mjcf_path()), names)
    estimate = np.array([
        odometry.update(rollout.joint_pos[t].astype(np.float32), rollout.anchor_pose_xyzw[t, 3:7].astype(np.float32))
        for t in range(rollout.steps)
    ])
    truth = rollout.anchor_pose_xyzw[:rollout.steps, :2] - rollout.anchor_pose_xyzw[0, :2]
    path = np.linalg.norm(np.diff(truth, axis=0), axis=1).sum()
    error = np.linalg.norm(estimate[:, :2] - truth, axis=1)
    assert path > 1.0
    assert error.max() < 0.05 * path
