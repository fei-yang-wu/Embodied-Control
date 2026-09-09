"""MPJPE math, FK replay, and the non-blocking telemetry recorder."""

import json

import numpy as np
import pytest

from embodied_control.logging import EcLogger
from embodied_control.lowlevel.metrics import (
    _set_replay_pose,
    compute_mpjpe,
    compute_sonic_success,
    oracle_tracking_metrics,
)
from embodied_control.lowlevel.reference import ReferenceMotion
from embodied_control.lowlevel.telemetry import TelemetryRecorder


def test_mpjpe_definitions_position_only():
    frames, bodies = 4, 3
    reference = np.zeros((frames, bodies, 3))
    reference_root = np.zeros((frames, 3))
    robot = reference.copy()
    robot_root = reference_root.copy()
    # A pure world translation moves G but leaves L at exactly zero.
    robot = robot + np.array([0.1, 0.0, 0.0])
    robot_root = robot_root + np.array([0.1, 0.0, 0.0])
    result = compute_mpjpe(robot, robot_root, reference, reference_root)
    assert result["mpjpe_g_mm"] == pytest.approx(100.0)
    assert result["mpjpe_l_mm"] == pytest.approx(0.0, abs=1e-9)

    # A single-body offset with the root in place hits both, scaled by 1/bodies.
    robot = reference.copy()
    robot[:, 1, 2] += 0.03
    result = compute_mpjpe(robot, reference_root, reference, reference_root)
    assert result["mpjpe_g_mm"] == pytest.approx(10.0)
    assert result["mpjpe_l_mm"] == pytest.approx(10.0)
    assert result["frames"] == frames


def test_mpjpe_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="disagree"):
        compute_mpjpe(
            np.zeros((2, 3, 3)),
            np.zeros((2, 3)),
            np.zeros((2, 4, 3)),
            np.zeros((2, 3)),
        )


def test_replay_pose_converts_xyzw_to_mujoco_wxyz():
    class Data:
        qpos = np.zeros(12, dtype=np.float64)

    data = Data()
    _set_replay_pose(
        data,
        np.array([7, 9]),
        np.array([1.25, -2.5]),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    )
    np.testing.assert_allclose(data.qpos[:7], [0.1, 0.2, 0.3, 0.7, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(data.qpos[[7, 9]], [1.25, -2.5])


def test_sonic_success_uses_height_orientation_and_end_effector_thresholds():
    names = (
        "pelvis",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    )
    reference_bodies = np.zeros((3, len(names), 3), dtype=np.float32)
    reference_anchor = np.zeros((3, 3), dtype=np.float32)
    identity = np.tile(np.array([0, 0, 0, 1], np.float32), (3, 1))
    robot_pose = np.concatenate([reference_anchor, identity], axis=1)
    success = compute_sonic_success(
        reference_bodies,
        robot_pose,
        reference_bodies,
        reference_anchor,
        identity,
        names,
    )
    assert success["success"]

    robot_bodies = reference_bodies.copy()
    robot_bodies[1, names.index("left_wrist_yaw_link"), 2] = 0.26
    failed = compute_sonic_success(
        robot_bodies,
        robot_pose,
        reference_bodies,
        reference_anchor,
        identity,
        names,
    )
    assert not failed["success"]
    assert failed["failure_tick"] == 1
    assert failed["failure_cause"] == "ee_body_pos"


def test_oracle_tracking_metrics_requires_bodies_and_alignment():
    motion = ReferenceMotion(
        "m",
        np.zeros((10, 29), np.float32),
        np.zeros((10, 3), np.float32),
        np.tile(np.array([0, 0, 0, 1], np.float32), (10, 1)),
    )
    telemetry = {
        "reference_frames": np.zeros(5, np.int32),
        "joint_position_log": np.zeros((5, 29), np.float32),
        "anchor_pose_log": np.zeros((5, 7), np.float32),
    }
    with pytest.raises(ValueError, match="body_pos_w"):
        oracle_tracking_metrics(None, "unused.xml", motion, telemetry)


class _FakeRuntime:
    def __init__(self, ticks=6):
        self._ticks = ticks
        self._running = False

    def stats(self):
        return {
            "ticks": self._ticks,
            "mode": 2,
            "fault": 0,
            "deadline_misses": 0,
            "wake_late_ns_max": 1000,
        }

    def running(self):
        return self._running

    @property
    def base_height(self):
        if self._running:
            raise RuntimeError("native base height is available after the loop stops")
        return 0.76

    def joint_position_log(self):
        return np.arange(self._ticks * 29, dtype=np.float32)

    def anchor_pose_log(self):
        return np.arange(self._ticks * 7, dtype=np.float32)

    def reference_frames(self):
        return np.arange(self._ticks, dtype=np.int32)

    def base_heights(self):
        return np.full(self._ticks, 0.76, np.float32)

    def reference_joint_mae(self):
        return np.full(self._ticks, 0.05, np.float32)

    def tick_durations_ns(self):
        return np.full(self._ticks, 400_000, np.int64)


def test_telemetry_collect_and_save(tmp_path):
    recorder = TelemetryRecorder(_FakeRuntime(), logger=EcLogger.null(), sample_hz=0)
    record = recorder.collect()
    assert record["joint_position_log"].shape == (6, 29)
    assert record["anchor_pose_log"].shape == (6, 7)
    assert record["reference_frames"].shape == (6,)
    npz_path = recorder.save(tmp_path, record)
    stored = np.load(npz_path)
    assert stored["joint_position_log"].shape == (6, 29)
    summary = json.loads((tmp_path / "telemetry_summary.json").read_text())
    assert summary["ticks"] == 6
    assert summary["arrays"]["reference_frames"] == [6]


def test_telemetry_drain_thread_logs_and_exits(tmp_path):
    events = []

    class _CapturingLogger(EcLogger):
        def __init__(self):
            import logging

            super().__init__(logging.getLogger("test.telemetry"), "test")

        def event(self, event_type, phase="", severity="info", **fields):
            events.append((event_type, fields))

    runtime = _FakeRuntime()
    runtime._running = True
    recorder = TelemetryRecorder(runtime, logger=_CapturingLogger(), sample_hz=50.0)
    recorder.start()
    import time

    time.sleep(0.1)
    recorder.stop()
    assert events, "drain thread produced no samples"
    name, fields = events[0]
    assert name == "telemetry.sample"
    assert fields["ticks"] == 6
    assert fields["base_height"] == pytest.approx(0.76)
