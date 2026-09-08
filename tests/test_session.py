"""The experiment session on fakes: selection rules, planner, rebuilds, episodes."""

from __future__ import annotations

import json

import pytest

from embodied_control.robot import FakeRobotRuntime, RobotMode
from embodied_control.robot.lifecycle import (
    Lifecycle,
    LifecycleConfig,
    LifecycleLog,
    LifecycleState as S,
)
from embodied_control.robot.session import (
    ExperimentSession,
    Selection,
    SessionConfig,
    episode_summary,
    oracle_worker_argv,
    planner_worker_argv,
)
from embodied_control.robot.shell import build_session_bindings
from embodied_control.robot.tui import render
from test_lifecycle import POSE, FakeClock, FakeHoist, StubTracker

CATALOG = ["hurry_idle_001_A277", "walk_arc_cw_001", "wave_002"]
LENGTHS = {"hurry_idle_001_A277": 505, "walk_arc_cw_001": 467, "wave_002": 120}


class FakePlanner:
    started: list["FakePlanner"] = []

    def __init__(self, selection: Selection) -> None:
        self.selection = selection
        self._alive = True
        FakePlanner.started.append(self)

    def alive(self) -> bool:
        return self._alive

    def stop(self) -> None:
        self._alive = False

    def describe(self) -> str:
        return f"fake-{self.selection.mode} {'running' if self._alive else 'stopped'}"


class RecordingTracker(StubTracker):
    def __init__(self, clock, selection):
        super().__init__(clock)
        self.selection = selection
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def joint_position_log(self):
        return [[0.1] * 29 for _ in range(self.control_ticks)]

    def reference_frames(self):
        # Relative to the worker's start frame, like the native runtime.
        return list(range(self.control_ticks))

    def reference_joint_mae(self):
        return [0.05 + 0.001 * i for i in range(self.control_ticks)]

    def anchor_pose_log(self):
        return [[0.0] * 7 for _ in range(self.control_ticks)]

    def tick_durations_ns(self):
        return [1000] * self.control_ticks


def _session(tmp_path=None, *, wait_ok=True, planner_autostart=True):
    clock = FakeClock()
    FakePlanner.started = []
    built: list[RecordingTracker] = []

    def tracker_factory(selection):
        tracker = RecordingTracker(clock, selection)
        built.append(tracker)
        return tracker

    log = LifecycleLog(tmp_path)

    def lifecycle_factory(tracker, selection, session):
        return Lifecycle(
            tracker,
            FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY),
            LifecycleConfig(start_pose=list(POSE), ticks=100, blend_ticks=50),
            hoist=FakeHoist(),
            auto_ack=True,
            log=log,
            now=clock.now,
            sleep=clock.sleep,
        )

    session = ExperimentSession(
        SessionConfig(
            catalog=list(CATALOG),
            motion_lengths=dict(LENGTHS),
            artifacts_dir=str(tmp_path) if tmp_path else None,
            planner_autostart=planner_autostart,
        ),
        Selection(mode="oracle", motion=CATALOG[0], start_frame=0),
        planner_factory=FakePlanner,
        tracker_factory=tracker_factory,
        lifecycle_factory=lifecycle_factory,
        slot_names=["/ec_req", "/ec_res"],
        wait_slots=lambda names, timeout: wait_ok,
        unlink_slots=lambda names: None,
    )
    return session, built, clock


def _run_episode(session, clock):
    assert session.auto().ok, session.lifecycle.last_result
    assert session.go().ok
    while session.tracker.running:
        clock.sleep(0.1)
        session.poll()
    session.poll()
    assert session.lifecycle.state is S.HOLD


def test_selection_keys_cycle_motions_and_clamp_frames():
    session, built, clock = _session()
    assert session.step_motion(1).ok and session.selection.motion == CATALOG[1]
    assert session.step_motion(-2).ok and session.selection.motion == CATALOG[2]
    assert session.step_frame(500).ok and session.selection.start_frame == LENGTHS[CATALOG[2]] - 1
    assert session.step_frame(-1000).ok and session.selection.start_frame == 0
    assert session.toggle_mode().ok and session.selection.mode == "vla"
    assert session.toggle_mode().ok and session.selection.mode == "oracle"
    assert not session.select_mode("teleop").ok


def test_selection_cycles_trackers_and_marks_build_stale():
    session, built, clock = _session()
    session.config.trackers = ["sonic", "walk"]
    session.selection = Selection(
        mode="oracle", motion=CATALOG[0], start_frame=0, tracker="sonic"
    )
    assert session.rebuild().ok
    assert session.step_tracker(1).ok
    assert session.selection.tracker == "walk"
    assert session.snapshot()["session"]["built"] is False
    assert session.step_tracker(-1).ok
    assert session.selection.tracker == "sonic"


def test_rebuild_with_autostart_starts_planner_and_tracker():
    session, built, clock = _session()
    assert session.rebuild().ok
    assert session.planner is not None and session.planner.alive()
    assert len(built) == 1 and session.built_for == session.selection
    snapshot = session.snapshot()
    assert snapshot["session"]["built"] is True
    assert snapshot["session"]["planner_running"] is True

    assert session.step_motion(1).ok
    assert session.snapshot()["session"]["built"] is False
    # Any lifecycle verb rebuilds for the new selection first, planner too:
    # a new tracker numbers requests from 1 and the worker serves one frame.
    assert session.advance().ok
    assert len(built) == 2 and built[0].closed is True
    assert built[1].selection.motion == CATALOG[1]
    assert len(FakePlanner.started) == 2
    assert not FakePlanner.started[0].alive() and FakePlanner.started[1].alive()
    assert FakePlanner.started[1].selection.motion == CATALOG[1]


def test_selection_is_refused_while_the_tracker_owns_the_joints():
    session, built, clock = _session()
    assert session.auto(S.POSE_SETTLED).ok
    assert not session.step_motion(1).ok
    assert not session.toggle_mode().ok
    assert not session.stop_planner().ok
    assert not session.rebuild().ok
    assert session.selection.motion == CATALOG[0]
    assert session.abort().ok
    assert session.step_motion(1).ok


def test_planner_that_never_creates_slots_is_stopped():
    session, built, clock = _session(wait_ok=False)
    result = session.start_planner()
    assert not result.ok and "slots" in result.detail
    assert FakePlanner.started and not FakePlanner.started[0].alive()


def test_episode_telemetry_is_recorded_at_hold(tmp_path):
    session, built, clock = _session(tmp_path)
    _run_episode(session, clock)
    assert len(session.episodes) == 1
    summary = session.episodes[0]
    assert summary["motion"] == CATALOG[0]
    assert summary["ticks"] == summary["frames_tracked"] >= 100
    assert summary["first_frame"] == 0
    assert 0.05 <= summary["joint_mae_mean_rad"] <= 0.2
    directory = tmp_path / "episodes"
    written = list(directory.glob("ep001_oracle_*/summary.json"))
    assert len(written) == 1
    assert json.loads(written[0].read_text())["episode"] == 1

    # Second take on another motion and frame.
    assert session.recover().ok
    assert session.step_motion(1).ok and session.step_frame(25).ok
    _run_episode(session, clock)
    assert session.episodes[-1]["motion"] == CATALOG[1]
    assert session.episodes[-1]["start_frame"] == 25
    assert session.episodes[-1]["first_frame"] == 25
    assert session.episodes[-1]["episode"] == 2
    assert len(list((tmp_path / "episodes").glob("ep002_*/summary.json"))) == 1


def test_session_render_shows_selection_progress_and_comm():
    session, built, clock = _session()
    assert session.rebuild().ok
    assert session.auto().ok
    assert session.go().ok
    clock.sleep(1.0)
    snapshot = session.snapshot()
    rows = render(snapshot, build_session_bindings(session), [], width=140, rates={"state_hz": 500.0, "publish_hz": 500.0, "planner_hz": 5.0})
    text = "\n".join(rows)
    assert "MODE oracle" in text and CATALOG[0] in text
    assert "REFERENCE  [" in text and "%" in text
    assert "LOWSTATE    500 Hz" in text and "PLANNER    5.0 Hz" in text
    assert "o mode" in text and "p planner" in text


def test_worker_argv_builders():
    oracle = oracle_worker_argv("/b", "/ref", "m1", 25, "/req", "/res")
    assert "oracle-worker" in oracle and "--create-slots" in oracle
    assert oracle[oracle.index("--start-frame") + 1] == "25"
    vla = planner_worker_argv(["python", "svc.py"], "/req", "/res", reply="chunk")
    assert vla[-2:] == ["python", "svc.py"] and "chunk" in vla


def test_lifecycle_job_resolves_named_tracker_paths(tmp_path):
    from embodied_control.robot.lifecycle_job import load_lifecycle_job

    job_file = tmp_path / "job.yaml"
    job_file.write_text(
        "bundle: default-bundle\ntrackers:\n  controller-b: model/controller-b\n"
    )
    job = load_lifecycle_job(job_file)
    assert job.bundle == str(tmp_path / "default-bundle")
    assert job.trackers == {"controller-b": str(tmp_path / "model/controller-b")}


def test_rows_of_accepts_flat_and_nested_logs():
    from embodied_control.robot.session import rows_of

    assert rows_of([1.0, 2.0, 3.0, 4.0], 2) == [[1.0, 2.0], [3.0, 4.0]]
    assert rows_of([[1, 2], [3, 4]], 2) == [[1.0, 2.0], [3.0, 4.0]]
    assert rows_of([], 2) == []


def test_episode_summary_handles_empty_logs():
    summary = episode_summary([], [], [], {}, {}, Selection(), 3)
    assert summary["episode"] == 3 and summary["frames_tracked"] == 0
    assert summary["first_frame"] is None


def test_snapshot_never_blocks_on_the_plant_rpc():
    """The plant's hoist status is a DDS RPC with a timeout; the render thread
    reads a cache the watcher fills, so a hung plant cannot freeze the display."""
    import time

    class SlowHoist(FakeHoist):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def status(self):
            self.calls += 1
            time.sleep(0.4)
            return {"owned": False, "fsm_id": 1, "hoisted": False, "hoist_mode": 0}

    session, built, clock = _session()
    assert session.rebuild().ok
    slow_hoist = SlowHoist()
    session.hoist = slow_hoist
    session.lifecycle.hoist = slow_hoist

    started = time.monotonic()
    for _ in range(20):
        assert session.snapshot()["hoist"] is None
    assert time.monotonic() - started < 0.2
    assert slow_hoist.calls == 0

    session.refresh_hoist()
    assert session.snapshot()["hoist"]["hoist_mode"] == 0
    # And the watcher does not re-ask on every one of its 100 Hz ticks.
    for _ in range(10):
        session.refresh_hoist()
    assert slow_hoist.calls == 1


def test_sim_reset_closes_runtime_and_returns_to_no_tracker():
    class ResettableHoist(FakeHoist):
        def reset(self):
            self.calls.append("reset")

        def status(self):
            return {"owned": True, "fsm_id": 1, "hoisted": True, "hoist_mode": 1}

    session, built, clock = _session()
    session.hoist = ResettableHoist()
    assert session.rebuild().ok
    tracker = session.tracker
    assert session.reset_sim().ok
    assert tracker.closed is True
    assert session.tracker is None and session.lifecycle is None
    assert session.planner is None and session.built_for is None
    assert session.hoist.calls[-1] == "reset"
    assert session.snapshot()["state"] == "NO TRACKER"


def test_planner_child_is_isolated_from_the_robot_cores():
    from embodied_control.robot.isolation import (
        child_preexec,
        non_realtime_cores,
    )

    free = non_realtime_cores((2, 3))
    assert 2 not in free and 3 not in free or free == set()
    # A reservation that would leave nothing gives everything back.
    assert non_realtime_cores(range(1024))
    assert child_preexec(free) is not None
    assert child_preexec((), nice=0, die_with_parent=False) is None
    # A child that outlives a SIGKILLed console keeps the slots; the default
    # asks the kernel to take it down with us.
    assert child_preexec(()) is not None


def test_a_build_does_not_launch_a_vla_planner_by_default():
    """The VLA worker loads its own checkpoint, gigabytes on the GPU.
    Pressing a lifecycle key must not be what starts it."""
    session, built, clock = _session(planner_autostart=False)
    assert session.select_mode("vla").ok

    result = session.rebuild()

    assert not result.ok and "press p" in result.detail
    assert FakePlanner.started == []
    assert session.planner is None
    assert built == []


def test_a_lifecycle_key_asks_for_the_vla_planner_instead_of_starting_one():
    session, built, clock = _session(planner_autostart=False)
    assert session.select_mode("vla").ok

    with pytest.raises(RuntimeError, match="press p"):
        session.advance()

    assert FakePlanner.started == []


def test_the_oracle_worker_starts_with_the_tracker():
    """It memory-maps the reference and loads no weights, so an operator
    has nothing to decide and should not be asked."""
    session, built, clock = _session(planner_autostart=False)

    assert session.rebuild().ok
    assert len(FakePlanner.started) == 1
    assert len(built) == 1


def test_the_operator_starts_the_vla_planner_and_then_builds():
    session, built, clock = _session(planner_autostart=False)
    assert session.select_mode("vla").ok

    assert session.start_planner().ok
    assert len(FakePlanner.started) == 1

    assert session.rebuild().ok
    assert len(built) == 1
    # A fresh tracker numbers its requests from 1, so the planner restarts
    # with it. What must not happen is a start nobody asked for.
    assert len(FakePlanner.started) == 2
