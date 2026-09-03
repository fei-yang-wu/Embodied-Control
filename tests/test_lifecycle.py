"""The lifecycle on fakes: every rung, every sink, every refused key.

No numpy, no DDS, no wall clock: a fake clock drives a stub tracker whose
writer stats evolve the way the native writer's do, so the gates see the
numbers they will see on the plant and the robot.
"""

from __future__ import annotations

import json

import pytest

from embodied_control.robot import FakeRobotRuntime, RobotMode
from embodied_control.robot.lifecycle import (
    Lifecycle,
    LifecycleConfig,
    LifecycleLog,
    LifecycleState as S,
    WRITER_CONTROL,
    WRITER_DAMP,
    WRITER_DISABLED,
    WRITER_HOLD,
    WRITER_INITIALIZE,
    WRITER_WAIT,
)

POSE = [0.1] * 29


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0
        self.listeners = []

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += dt
        for listener in self.listeners:
            listener(self.t, dt)


class StubTracker:
    def __init__(
        self,
        clock: FakeClock,
        *,
        vendor_service: str = "ai",
        planner_fault: int = 0,
        ramp_blocked: bool = False,
        first_action_error: float = 0.0,
        measured_pose: list[float] | None = None,
    ) -> None:
        self.clock = clock
        self.vendor_service = vendor_service
        self.planner_fault = planner_fault
        self.ramp_blocked = ramp_blocked
        self.first_action_error = first_action_error
        self.measured_pose = measured_pose
        self.mode = WRITER_DISABLED
        self.gate_open = False
        self.vendor_released = False
        self.publishes = 0
        self.publish_failures = 0
        self.ramp_faults = 0
        self.running = False
        self.control_ticks = 0
        self.planner_responses = 0
        self.fault = 0
        self.budget = 0
        self.speed = 0.0
        self.error = 0.0
        self.blend_remaining = 0
        self.target: list[float] | None = None
        self.joint_pos = [0.0] * 29
        self.ramp_end: float | None = None
        self.calls: list[str] = []
        self.restored_with = ""
        self.pending_writer_damp = False
        self.reference_paused: list[bool] = []
        clock.listeners.append(self.on_time)

    # -- time model -------------------------------------------------------

    def on_time(self, t: float, dt: float) -> None:
        if self.gate_open and self.mode != WRITER_DISABLED:
            self.publishes += max(1, int(dt * 500))
        if self.mode == WRITER_INITIALIZE and self.ramp_end is not None:
            if self.ramp_blocked and t >= self.ramp_end / 2:
                self.ramp_faults += 1
                self.mode = WRITER_DAMP
            elif t >= self.ramp_end:
                self.mode = WRITER_WAIT
                self.joint_pos = list(self.measured_pose or self.target)
                self.error = 0.0
        self.speed = max(0.0, self.speed - 0.5 * dt)
        if self.running:
            self.control_ticks += max(1, int(dt * 50))
            self.planner_responses += 1
            if self.control_ticks >= self.budget:
                self.running = False
            if self.mode == WRITER_CONTROL:
                self.blend_remaining = max(0, self.blend_remaining - int(dt * 500))
        if self.pending_writer_damp:
            self.mode = WRITER_DAMP

    # -- Tracker protocol -------------------------------------------------

    @property
    def unitree_mode(self) -> int:
        return self.mode

    def wait_for_state(self, timeout_seconds: float) -> bool:
        return True

    def writer_stats(self) -> dict:
        return {
            "mode": self.mode,
            "gate_open": self.gate_open,
            "vendor_released": self.vendor_released,
            "publishes": self.publishes,
            "publish_failures": self.publish_failures,
            "crc_errors": 0,
            "hardware_faults": 0,
            "watchdog_faults": self.ramp_faults,
            "ramp_faults": self.ramp_faults,
            "state_fault_reason": 0,
            "realtime_configured": True,
            "joint_speed_max": self.speed,
            "tracking_error_max": self.error,
            "command_target_error_max": self.first_action_error,
            "blend_ticks_remaining": self.blend_remaining,
        }

    def stats(self) -> dict:
        return {
            "control_ticks": self.control_ticks,
            "planner_requests": self.planner_responses,
            "planner_responses": self.planner_responses,
            "fault": self.fault,
            "mode": 2 if self.running else 0,
        }

    def vendor_mode(self) -> str:
        self.calls.append("vendor_mode")
        return self.vendor_service if not self.vendor_released else ""

    def open_damp_gate(self) -> None:
        if self.mode not in {WRITER_DISABLED, WRITER_DAMP}:
            raise RuntimeError("open_damp_gate needs DISABLED or DAMP")
        self.calls.append("open_damp_gate")
        self.mode = WRITER_DAMP
        self.gate_open = True

    def release_vendor(self) -> None:
        if not self.gate_open or self.mode != WRITER_DAMP:
            raise RuntimeError("release_vendor needs the gate open in DAMP")
        self.calls.append("release_vendor")
        self.vendor_released = True
        self.speed = 0.02

    def begin_initialization(
        self, duration_seconds, *, target_position=None, hold_current=False,
        skip_motion_switcher=False,
    ) -> None:
        if not self.gate_open or self.mode not in {WRITER_DAMP, WRITER_HOLD, WRITER_WAIT}:
            raise RuntimeError("initialization needs an open gate in DAMP/HOLD/WAIT")
        self.calls.append("begin_initialization")
        self.target = list(target_position) if target_position is not None else [0.0] * 29
        self.mode = WRITER_INITIALIZE
        self.ramp_end = self.clock.now() + duration_seconds
        self.speed = 0.4
        self.error = 0.2

    def wait_for_mode(self, expected: int, timeout_seconds: float) -> bool:
        deadline = self.clock.now() + timeout_seconds
        while self.mode != expected and self.clock.now() < deadline:
            self.clock.sleep(0.05)
        return self.mode == expected

    def start(self, ticks: int, paced: bool = True) -> None:
        if self.running:
            raise RuntimeError("already started")
        self.calls.append(f"start({ticks})")
        self.running = True
        self.budget = ticks
        self.control_ticks = 0
        self.fault = self.planner_fault

    def engage_control(self, blend_ticks: int = 0) -> None:
        if self.mode != WRITER_WAIT:
            raise RuntimeError("engage needs WAIT")
        self.calls.append(f"engage_control({blend_ticks})")
        self.mode = WRITER_CONTROL
        self.blend_remaining = blend_ticks

    def hold(self) -> None:
        if self.mode not in {WRITER_CONTROL, WRITER_WAIT}:
            raise RuntimeError("hold needs CONTROL or WAIT")
        self.calls.append("hold")
        self.mode = WRITER_HOLD

    def force_damp(self) -> None:
        self.calls.append("force_damp")
        self.mode = WRITER_DAMP

    def stop(self) -> None:
        self.running = False

    def wait(self) -> None:
        pass

    def set_reference_paused(self, paused: bool) -> None:
        self.reference_paused.append(paused)

    def close_gate(self) -> None:
        if self.mode not in {WRITER_DAMP, WRITER_DISABLED}:
            raise RuntimeError("close_gate needs DAMP")
        self.calls.append("close_gate")
        self.gate_open = False
        self.mode = WRITER_DISABLED

    def restore_vendor(self, name: str) -> None:
        if self.gate_open:
            raise RuntimeError("gate still open")
        self.calls.append(f"restore_vendor({name})")
        self.restored_with = name
        self.vendor_released = False

    def joint_position(self) -> list[float]:
        return list(self.joint_pos)

    def projected_gravity(self) -> list[float]:
        return [0.0, 0.0, -1.0]


class FakeHoist:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def hoist(self) -> None:
        self.calls.append("hoist")

    def lower(self) -> None:
        self.calls.append("lower")

    def slack(self) -> None:
        self.calls.append("slack")


def _lifecycle(tmp_path=None, *, auto_ack=True, tracker_kwargs=None, **config):
    clock = FakeClock()
    tracker = StubTracker(clock, **(tracker_kwargs or {}))
    vendor = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY)
    hoist = FakeHoist()
    cfg = LifecycleConfig(start_pose=list(POSE), ticks=100, blend_ticks=50, **config)
    lifecycle = Lifecycle(
        tracker,
        vendor,
        cfg,
        hoist=hoist,
        auto_ack=auto_ack,
        log=LifecycleLog(tmp_path),
        now=clock.now,
        sleep=clock.sleep,
    )
    return lifecycle, tracker, vendor, hoist, clock


def _run_to_hold(lifecycle, tracker, clock):
    assert lifecycle.go().ok, lifecycle.last_result
    assert lifecycle.state is S.RUNNING
    while tracker.running:
        clock.sleep(0.1)
        lifecycle.poll()
    lifecycle.poll()
    assert lifecycle.state is S.HOLD, lifecycle.last_result


def test_happy_path_hoist_to_standing(tmp_path):
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(tmp_path)

    result = lifecycle.auto()
    assert result.ok, result
    assert lifecycle.state is S.PRIMED
    assert lifecycle.vendor_name == "ai"
    assert vendor.mode() is RobotMode.DAMP
    assert tracker.calls[:4] == [
        "vendor_mode", "open_damp_gate", "release_vendor", "begin_initialization",
    ]
    assert "lower" in hoist.calls

    _run_to_hold(lifecycle, tracker, clock)
    assert tracker.mode == WRITER_HOLD
    assert not tracker.running
    # Pinned for the probe, pinned again at go, released once the blend is in.
    assert tracker.reference_paused == [True, True, False]

    result = lifecycle.recover()
    assert result.ok, result
    assert lifecycle.state is S.STANDING
    assert tracker.restored_with == "ai"
    assert tracker.mode == WRITER_DISABLED
    assert vendor.mode() is RobotMode.READY
    assert hoist.calls.count("hoist") >= 2 and hoist.calls[-1] == "lower"

    lines = (tmp_path / "lifecycle.jsonl").read_text().splitlines()
    states = [json.loads(line)["to_state"] for line in lines]
    assert states[:3] == ["PRECHECK", "VENDOR_DAMP_CONFIRMED", "SAFE_EXTERNAL_COMMAND_PRESENT"]
    assert states[-1] == "STANDING"
    assert all(json.loads(line)["ok"] for line in lines)
    assert (tmp_path / "pose_match.json").exists()


def test_next_episode_restarts_at_precheck(tmp_path):
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(tmp_path)
    assert lifecycle.auto().ok
    _run_to_hold(lifecycle, tracker, clock)
    assert lifecycle.recover().ok
    first_episode = lifecycle.episode

    assert lifecycle.advance().ok
    assert lifecycle.state is S.PRECHECK
    assert lifecycle.episode == first_episode + 1


def test_keys_are_refused_in_the_wrong_state():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle()
    assert not lifecycle.go().ok
    assert not lifecycle.hold().ok
    assert not lifecycle.retake().ok
    assert not lifecycle.recover().ok
    assert lifecycle.state is S.IDLE
    assert tracker.calls == []


def test_operator_acks_gate_precheck_and_lowering():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(auto_ack=False)
    result = lifecycle.advance()
    assert not result.ok and "hoisted" in result.detail
    assert lifecycle.state is S.IDLE

    lifecycle.ack_hoisted()
    result = lifecycle.auto()
    assert not result.ok and "press l" in result.detail
    assert lifecycle.state is S.POSE_SETTLED

    lifecycle.ack_lowered()
    assert lifecycle.auto().ok
    assert lifecycle.state is S.PRIMED
    assert hoist.calls == ["hoist", "lower"]


def test_ramp_guard_trips_into_fault_and_recovers():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(
        tracker_kwargs={"ramp_blocked": True}
    )
    result = lifecycle.auto()
    assert not result.ok
    assert lifecycle.state is S.FAULT
    assert "ramp guard" in lifecycle.fault_reason
    assert tracker.mode == WRITER_DAMP

    assert lifecycle.recover().ok
    assert lifecycle.state is S.STANDING
    assert tracker.restored_with == "ai"


def test_planner_fault_lands_in_fault_with_the_tracker_stopped():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(
        tracker_kwargs={"planner_fault": 3}
    )
    result = lifecycle.auto()
    assert not result.ok
    assert lifecycle.state is S.FAULT
    assert not tracker.running
    assert tracker.mode == WRITER_DAMP


def test_shutdown_from_fault_silences_writer_and_restores_vendor():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(
        tracker_kwargs={"planner_fault": 3}
    )
    assert not lifecycle.auto().ok
    assert lifecycle.state is S.FAULT
    assert tracker.gate_open

    lifecycle.shutdown()

    assert lifecycle.state is S.VENDOR_RESTORED
    assert tracker.mode == WRITER_DISABLED
    assert not tracker.gate_open
    assert tracker.restored_with == "ai"


def test_first_action_far_from_hold_is_refused_without_fault():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(
        tracker_kwargs={"first_action_error": 3.0}
    )
    result = lifecycle.auto()
    assert not result.ok and "first policy action" in result.detail
    assert lifecycle.state is S.POSE_MATCH_VERIFIED
    assert not tracker.running
    assert tracker.mode == WRITER_WAIT


def test_pose_mismatch_is_refused_and_reported(tmp_path):
    wrong = list(POSE)
    wrong[3] += 0.3
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(
        tmp_path, tracker_kwargs={"measured_pose": wrong}
    )
    result = lifecycle.auto()
    assert not result.ok
    assert lifecycle.state is S.LOWERED
    report = json.loads((tmp_path / "pose_match.json").read_text())
    assert report["ok"] is False
    assert report["pose"]["worst_joint"] == 3


def test_writer_damp_during_running_is_a_fault():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle()
    assert lifecycle.auto().ok
    assert lifecycle.go().ok
    tracker.pending_writer_damp = True
    clock.sleep(0.1)
    lifecycle.poll()
    assert lifecycle.state is S.FAULT
    assert "writer damped" in lifecycle.fault_reason


def test_damp_then_recover_to_damp_only_releases():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(end_state="damp")
    assert lifecycle.auto().ok
    assert lifecycle.go().ok
    assert lifecycle.damp().ok
    assert lifecycle.state is S.DAMP
    assert tracker.mode == WRITER_DAMP
    assert lifecycle.recover().ok
    assert lifecycle.state is S.RELEASED
    assert tracker.restored_with == ""


def test_rearm_from_hold_ramps_again_without_the_vendor():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle()
    assert lifecycle.auto().ok
    _run_to_hold(lifecycle, tracker, clock)
    episode = lifecycle.episode

    assert lifecycle.retake().ok, lifecycle.last_result
    assert lifecycle.state is S.START_POSE_RAMP
    assert tracker.calls.count("release_vendor") == 1
    assert lifecycle.auto().ok
    assert lifecycle.state is S.PRIMED
    assert lifecycle.episode == episode + 1
    # One probe start per POLICY_COMMAND_FRESH and one real start per go().
    assert tracker.calls.count("start(100)") == 3
    assert lifecycle.go().ok
    assert tracker.calls.count("start(100)") == 4


def test_abort_before_arming_hands_back_to_the_vendor():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle()
    assert lifecycle.auto(S.POSE_SETTLED).ok
    assert lifecycle.abort().ok
    assert lifecycle.state is S.STANDING
    assert tracker.restored_with == "ai"


def test_missing_vendor_service_fails_precheck():
    lifecycle, tracker, vendor, hoist, clock = _lifecycle(
        tracker_kwargs={"vendor_service": ""}
    )
    result = lifecycle.advance()
    assert not result.ok and "motion service" in result.detail
    assert lifecycle.state is S.IDLE


def test_config_rejects_bad_end_state():
    with pytest.raises(ValueError):
        LifecycleConfig(end_state="fly")
