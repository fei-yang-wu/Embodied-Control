"""MPJPE math, FK replay, and the non-blocking telemetry recorder."""

import json

import numpy as np
import pytest

from embodied_control.logging import EcLogger
from embodied_control.lowlevel.metrics import compute_mpjpe, oracle_tracking_metrics
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
            np.zeros((2, 3, 3)), np.zeros((2, 3)),
            np.zeros((2, 4, 3)), np.zeros((2, 3)),
        )


def test_oracle_tracking_metrics_requires_bodies_and_alignment():
    motion = ReferenceMotion(
        "m", np.zeros((10, 29), np.float32), np.zeros((10, 3), np.float32),
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
        return {"ticks": self._ticks, "mode": 2, "fault": 0,
                "deadline_misses": 0, "wake_late_ns_max": 1000}

    def running(self):
        return self._running

    @property
    def base_height(self):
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