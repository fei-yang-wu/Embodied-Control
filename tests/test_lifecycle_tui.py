"""The full-screen console, rendered without a terminal."""

from __future__ import annotations

from embodied_control.console import KEY_SPACE
from embodied_control.robot import FakeRobotRuntime, RobotMode
from embodied_control.robot.lifecycle import (
    Lifecycle,
    LifecycleConfig,
    LifecycleState as S,
)
from embodied_control.robot.shell import build_lifecycle_bindings
from embodied_control.robot.tui import DISPLAY_ORDER, LifecycleTui, render
from test_lifecycle import POSE, FakeClock, FakeHoist, StubTracker


def _lifecycle():
    clock = FakeClock()
    tracker = StubTracker(clock)
    vendor = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY)
    lifecycle = Lifecycle(
        tracker,
        vendor,
        LifecycleConfig(start_pose=list(POSE), ticks=100, blend_ticks=50),
        hoist=FakeHoist(),
        auto_ack=True,
        now=clock.now,
        sleep=clock.sleep,
    )
    return lifecycle, tracker, clock


def test_render_marks_the_current_rung_and_fits_the_width():
    lifecycle, tracker, clock = _lifecycle()
    assert lifecycle.auto(S.POSE_SETTLED).ok
    rows = render(lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), ["hello"], width=90, height=40)
    assert all(len(row) == 90 for row in rows)
    text = "\n".join(rows)
    assert "[>] POSE_SETTLED" in text
    assert "[x] START_POSE_RAMP" in text
    assert "[ ] LOWERED" in text
    assert "SPACE DAMP" in text
    assert "state: POSE_SETTLED" in text
    assert "vendor: released" in text
    assert "hello" in text


def test_render_shows_fault_banner_and_busy_line():
    lifecycle, tracker, clock = _lifecycle()
    lifecycle.fault_reason = "writer damped during RUNNING"
    lifecycle.state = S.FAULT
    rows = render(lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), [], busy="auto-advance to PRIMED", width=80, height=30)
    text = "\n".join(rows)
    assert "!! FAULT: writer damped during RUNNING" in text
    assert "BUSY  auto-advance to PRIMED" in text
    assert len(rows) <= 30


def test_display_order_covers_every_non_sink_state():
    shown = set(DISPLAY_ORDER)
    for state in S:
        if state in {S.IDLE, S.FAULT}:
            continue
        assert state in shown, state


def test_space_bypasses_the_lock_and_queues_the_transition():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    assert tui.handle(KEY_SPACE) is True
    # The writer got the damp before any worker ran.
    assert tracker.calls[-1] == "force_damp"
    assert tui._queue.qsize() == 1
    assert any("SPACE" in note for note in tui.notes)


def test_quit_is_refused_while_busy_and_unknown_keys_are_ignored():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    assert tui.handle("~") is True
    assert tui._queue.qsize() == 0
    tui._busy = "next: advance one state"
    assert tui.handle("q") is True
    assert tui.handle("n") is True
    assert tui._queue.qsize() == 0
    tui._busy = ""
    assert tui.handle("q") is False


def test_frame_renders_from_a_live_lifecycle():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    tui.note("started")
    rows = tui.frame(100, 40)
    assert rows and rows[0].startswith(" G1 LIFECYCLE")
    assert any("started" in row for row in rows)
