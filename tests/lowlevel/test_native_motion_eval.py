from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_control.cli import build_parser
from embodied_control.lowlevel import native_motion_eval as motion_eval


def test_trajectory_metrics_reports_planar_motion_and_yaw():
    yaw = np.radians([0.0, 45.0, 90.0])
    telemetry = {
        "anchor_pose_log": np.column_stack(
            [
                np.asarray([[0.0, 0.0, 0.8], [1.0, 0.0, 0.8], [1.0, 1.0, 0.8]]),
                np.zeros(3),
                np.zeros(3),
                np.sin(yaw / 2.0),
                np.cos(yaw / 2.0),
            ]
        ),
        "base_heights": np.asarray([0.8, 0.78, 0.79]),
    }

    metrics, xy = motion_eval.trajectory_metrics(
        telemetry, control_hz=10, min_physics_height=0.77
    )

    assert xy.shape == (3, 2)
    assert metrics["net_displacement_m"] == pytest.approx(np.sqrt(2.0))
    assert metrics["path_length_m"] == pytest.approx(2.0)
    assert metrics["straightness"] == pytest.approx(np.sqrt(2.0) / 2.0)
    assert metrics["yaw_change_deg"] == pytest.approx(90.0)
    assert metrics["min_height_m"] == pytest.approx(0.77)
    assert not metrics["fell_below_0_4"]


def test_native_motion_eval_cli_collects_goals_and_service_command():
    args = build_parser().parse_args(
        [
            "lowlevel",
            "native-motion-eval",
            "--bundle",
            "bundle",
            "--model",
            "robot.xml",
            "--goal",
            "walk_forward",
            "--goal",
            "idle",
            "--repeats",
            "2",
            "--",
            "planner",
            "--checkpoint",
            "head.pt",
        ]
    )

    assert args.goal == ["walk_forward", "idle"]
    assert args.repeats == 2
    assert args.service_command == ["--", "planner", "--checkpoint", "head.pt"]


def _fake_bundle(root: Path):
    manifest_path = root / "manifest.json"
    manifest_path.write_text('{"bundle": "fake"}\n')
    command = SimpleNamespace(z_dim=64, hold_steps=10)
    manifest = SimpleNamespace(
        command=command,
        interface="latent",
        rates=SimpleNamespace(control_hz=50),
    )
    return SimpleNamespace(root=root, manifest=manifest)


class _FakeService:
    instances = 0
    goals = []

    def __init__(self, command, *, action_width, window_frames, goal):
        type(self).instances += 1
        self.command = command
        self.goal = goal
        self.head_ms = []
        self.ready = {
            "ready": True,
            "action_horizon": window_frames,
            "action_width": action_width,
            "window_frames": window_frames,
        }
        self.closed = False

    def close(self):
        self.closed = True


class _FakeWorker:
    def __init__(self, request_slot, response_slot, request_fn, **kwargs):
        self.service = request_fn
        self.request_ms = [4.0, 6.0]
        self.last_error = None

    def start(self):
        _FakeService.goals.append(self.service.goal)
        self.service.head_ms.extend([3.0, 5.0])

    def close(self):
        return None


class _FakeRuntime:
    instances = 0

    def __init__(self, bundle, model, **kwargs):
        type(self).instances += 1
        self.running = False
        self._ticks = 0

    def start(self, ticks, *, paced):
        self._ticks = ticks
        self.running = True

    def wait(self):
        self.running = False

    def stop(self):
        self.running = False

    def close(self):
        self.running = False

    def stats(self):
        return {
            "ticks": self._ticks,
            "control_ticks": self._ticks,
            "damp_ticks": 0,
            "planner_responses": 2,
            "deadline_misses": 0,
            "backend_deadline_misses": 0,
            "fault": 0,
        }

    def joint_position_log(self):
        return np.zeros((self._ticks, 29), dtype=np.float32)

    def anchor_pose_log(self):
        pose = np.zeros((self._ticks, 7), dtype=np.float32)
        pose[:, 0] = np.arange(self._ticks) * 0.1
        pose[:, 2] = 0.78
        pose[:, 6] = 1.0
        return pose

    def reference_frames(self):
        return np.arange(self._ticks, dtype=np.int32)

    def base_heights(self):
        return np.full(self._ticks, 0.78, dtype=np.float32)

    def reference_joint_mae(self):
        return np.zeros(self._ticks, dtype=np.float32)

    def tick_durations_ns(self):
        return np.full(self._ticks, 20_000_000, dtype=np.int64)

    @property
    def min_base_height(self):
        return 0.78


def test_native_motion_eval_reuses_service_and_writes_artifacts(tmp_path, monkeypatch):
    bundle_root = tmp_path / "bundle"
    bundle_root.mkdir()
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>")
    fake_bundle = _fake_bundle(bundle_root)
    _FakeService.instances = 0
    _FakeService.goals = []
    _FakeRuntime.instances = 0
    monkeypatch.setattr(motion_eval.PolicyBundle, "load", lambda path: fake_bundle)
    monkeypatch.setattr(motion_eval, "StdioChunkService", _FakeService)
    monkeypatch.setattr(motion_eval, "NativeLatentPlanWorker", _FakeWorker)
    monkeypatch.setattr(motion_eval, "NativeMujocoLoop", _FakeRuntime)

    run_dir, result = motion_eval.run_native_motion_evaluation(
        bundle_root,
        model,
        ["walk_forward", "idle"],
        ["fake-planner"],
        repeats=2,
        ticks=5,
        output_root=tmp_path / "runs",
    )

    assert result["status"]["succeeded"]
    assert _FakeService.instances == 1
    assert _FakeRuntime.instances == 4
    assert _FakeService.goals == ["walk_forward", "idle", "walk_forward", "idle"]
    required = {
        "job.yaml",
        "resolved_job.yaml",
        "manifest.json",
        "status.json",
        "validation.json",
        "metrics.json",
        "episodes.jsonl",
        "summary.csv",
        "trajectories.svg",
        "logs",
    }
    assert required <= {path.name for path in run_dir.iterdir()}
    rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
    assert len(rows) == 4
    assert all(row["planner"]["head_ms"]["mean"] == 4.0 for row in rows)
    assert all(row["success"] for row in rows)
    assert len(list((run_dir / "episodes").glob("*/telemetry/telemetry.npz"))) == 4


def test_native_motion_eval_writes_failed_contract_when_bundle_is_missing(tmp_path):
    model = tmp_path / "robot.xml"
    model.write_text("<mujoco/>")

    with pytest.raises(motion_eval.NativeMotionEvalError) as raised:
        motion_eval.run_native_motion_evaluation(
            tmp_path / "missing-bundle",
            model,
            ["idle"],
            ["fake-planner"],
            output_root=tmp_path / "runs",
        )

    run_dir = raised.value.run_dir
    assert json.loads((run_dir / "status.json").read_text())["state"] == "failed"
    assert not json.loads((run_dir / "validation.json").read_text())["valid"]
    for name in (
        "job.yaml",
        "resolved_job.yaml",
        "manifest.json",
        "status.json",
        "validation.json",
        "metrics.json",
        "episodes.jsonl",
    ):
        assert (run_dir / name).exists()
