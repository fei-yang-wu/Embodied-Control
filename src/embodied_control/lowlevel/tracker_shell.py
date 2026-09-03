"""Key bindings for the tracker operator console.

This is the terminal that owns ``rt/lowcmd``: the 500 Hz writer, the
initialization ramp, and serving. Predefined-qpos moves live here rather than
in the robot console because only this process may command joint targets — the
G1's sport service offers stand/squat/sit and nothing finer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from embodied_control.console import KEY_DAMP, ConsoleQuit, KeyBinding


@dataclass
class Pose:
    name: str
    # None target with hold_current False means the bundle's default stance.
    target: list[float] | None = None
    hold_current: bool = False


@dataclass
class TrackerConsoleState:
    poses: list[Pose]
    ticks: int
    init_seconds: float = 3.0
    selected: int = 0
    serving: bool = False
    engaged: bool = False
    log: list[str] = field(default_factory=list)

    @property
    def pose(self) -> Pose:
        return self.poses[self.selected]

    def next_pose(self) -> None:
        self.selected = (self.selected + 1) % len(self.poses)


def tracker_status(runtime, state: TrackerConsoleState) -> str:
    try:
        stats = runtime.stats()
        ticks = stats.get("control_ticks", 0)
        fault = stats.get("fault", 0)
    except Exception:  # console must render even mid-teardown
        ticks, fault = 0, 0
    serving = "SERVING" if getattr(runtime, "running", False) else "idle"
    engaged = "engaged" if state.engaged else "not engaged"
    return (
        f"{serving}  {engaged}  pose: {state.pose.name}  "
        f"ticks: {ticks}  fault: {fault}"
    )


def build_tracker_bindings(
    runtime,
    state: TrackerConsoleState,
    *,
    on_note: Callable[[str], None] = lambda _msg: None,
) -> list[KeyBinding]:
    def damp() -> None:
        # force_damp is a lock-free mode store the independent 500 Hz writer
        # picks up on its next tick, so it lands even if the control thread is
        # wedged. That is why it is the one key that never goes through stop().
        runtime.force_damp()
        state.serving = False
        state.engaged = False
        on_note("damped")

    def goto_pose() -> None:
        pose = state.pose
        runtime.begin_initialization(
            state.init_seconds,
            target_position=pose.target,
            hold_current=pose.hold_current,
        )
        on_note(f"ramping to {pose.name} over {state.init_seconds:g}s")

    def cycle_pose() -> None:
        state.next_pose()
        on_note(f"pose selected: {state.pose.name}")

    def engage() -> None:
        runtime.engage_control()
        state.engaged = True
        on_note("control engaged")

    def start_serving() -> None:
        if runtime.running:
            on_note("already serving")
            return
        runtime.start(state.ticks, paced=True)
        state.serving = True
        on_note(f"serving up to {state.ticks} ticks")

    def stop_serving() -> None:
        runtime.force_damp()
        runtime.stop()
        state.serving = False
        state.engaged = False
        on_note("serving stopped (damped)")

    def quit_console() -> None:
        raise ConsoleQuit

    return [
        KeyBinding(KEY_DAMP, "DAMP (safety stop)", damp, "safety"),
        KeyBinding("p", "cycle predefined pose", cycle_pose, "pose"),
        KeyBinding("i", "ramp to selected pose", goto_pose, "pose"),
        KeyBinding("a", "engage control", engage, "serving"),
        KeyBinding("g", "go / start serving", start_serving, "serving"),
        KeyBinding("e", "end serving", stop_serving, "serving"),
        KeyBinding("q", "quit (damps first)", quit_console, "console"),
    ]
