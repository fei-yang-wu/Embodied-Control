import io

from embodied_control.console import KEY_SPACE, KeyConsole
from embodied_control.lowlevel.tracker_shell import (
    Pose,
    TrackerConsoleState,
    build_tracker_bindings,
    tracker_status,
)


class StubTracker:
    def __init__(self):
        self.running = False
        self.calls = []
        self.init_kwargs = None

    def force_damp(self):
        self.calls.append("force_damp")

    def stop(self):
        self.calls.append("stop")
        self.running = False

    def start(self, ticks, paced=True):
        self.calls.append(f"start({ticks})")
        self.running = True

    def engage_control(self):
        self.calls.append("engage_control")

    def begin_initialization(self, seconds, *, target_position, hold_current):
        self.calls.append("begin_initialization")
        self.init_kwargs = (seconds, target_position, hold_current)

    def stats(self):
        return {"control_ticks": 7, "fault": 0}


def _setup(poses=None):
    tracker = StubTracker()
    state = TrackerConsoleState(
        poses=poses or [Pose("default-stance"), Pose("hold", hold_current=True)],
        ticks=400,
    )
    console = KeyConsole(
        build_tracker_bindings(tracker, state), out=io.StringIO()
    )
    return tracker, state, console


def test_damp_does_not_route_through_stop():
    tracker, state, console = _setup()
    state.serving = True

    console.handle(KEY_SPACE)

    # force_damp lands on the independent writer thread even if the control
    # thread is wedged; requiring stop() first would defeat that.
    assert tracker.calls == ["force_damp"]
    assert state.serving is False


def test_pose_cycles_and_ramps_to_the_selection():
    tracker, state, console = _setup()

    assert state.pose.name == "default-stance"
    console.handle("p")
    assert state.pose.name == "hold"

    console.handle("i")
    seconds, target, hold = tracker.init_kwargs
    assert target is None and hold is True


def test_predefined_qpos_pose_is_passed_through():
    qpos = [0.1] * 29
    tracker, state, console = _setup(poses=[Pose("frame0", target=qpos)])

    console.handle("i")
    _seconds, target, hold = tracker.init_kwargs
    assert target == qpos and hold is False


def test_start_and_stop_serving():
    tracker, state, console = _setup()

    console.handle("g")
    assert tracker.running is True
    assert "start(400)" in tracker.calls

    console.handle("g")  # already serving: no second start
    assert tracker.calls.count("start(400)") == 1

    console.handle("e")
    assert tracker.running is False
    assert tracker.calls[-2:] == ["force_damp", "stop"]


def test_arm_updates_status():
    tracker, state, console = _setup()
    console.handle("a")
    assert state.engaged is True
    assert "engaged" in tracker_status(tracker, state)


def test_status_reports_serving_and_ticks():
    tracker, state, console = _setup()
    console.handle("g")
    text = tracker_status(tracker, state)
    assert "SERVING" in text
    assert "ticks: 7" in text


def test_quit_ends_the_loop():
    _tracker, _state, console = _setup()
    assert console.handle("q") is False
