"""Key bindings for the robot operator console.

Bindings are derived from the runtime's capability protocols, so a robot
without legs simply has no motion keys — the console needs no per-robot code.
"""

from __future__ import annotations

from embodied_control.robot.base import (
    GestureControl,
    PostureControl,
    RobotRuntime,
    VelocityControl,
)
from embodied_control.console import (
    KEY_DOWN,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_SPACE,
    KEY_UP,
    ConsoleQuit,
    KeyBinding,
)


def robot_status(runtime: RobotRuntime) -> str:
    health = runtime.health()
    reach = "up" if health.reachable else "DOWN"
    detail = f"  {health.detail}" if health.detail else ""
    return f"mode: {health.mode}   link: {reach}{detail}"


def build_robot_bindings(
    runtime: RobotRuntime,
    *,
    vx_step: float = 0.2,
    vy_step: float = 0.2,
    vyaw_step: float = 0.3,
) -> list[KeyBinding]:
    def quit_console() -> None:
        raise ConsoleQuit

    bindings = [
        KeyBinding(KEY_SPACE, "DAMP (safety stop)", runtime.damp, "safety"),
        KeyBinding("z", "zero torque", runtime.zero_torque, "safety"),
        KeyBinding("r", "ready (stand + balance)", runtime.ready, "lifecycle"),
    ]

    if isinstance(runtime, PostureControl):
        bindings += [
            KeyBinding("s", "sit", runtime.sit, "posture"),
            KeyBinding("c", "crouch (squat)", runtime.squat, "posture"),
            KeyBinding("n", "stand", runtime.stand, "posture"),
        ]

    if isinstance(runtime, VelocityControl):
        # SetVelocity carries its own duration, so each press is one bounded
        # step rather than a held key - a terminal cannot report key release.
        bindings += [
            KeyBinding(
                KEY_UP,
                f"forward {vx_step} m/s",
                lambda: runtime.move(vx_step, 0.0, 0.0),
                "motion",
            ),
            KeyBinding(
                KEY_DOWN,
                f"back {vx_step} m/s",
                lambda: runtime.move(-vx_step, 0.0, 0.0),
                "motion",
            ),
            KeyBinding(
                KEY_LEFT,
                f"turn left {vyaw_step} rad/s",
                lambda: runtime.move(0.0, 0.0, vyaw_step),
                "motion",
            ),
            KeyBinding(
                KEY_RIGHT,
                f"turn right {vyaw_step} rad/s",
                lambda: runtime.move(0.0, 0.0, -vyaw_step),
                "motion",
            ),
            KeyBinding(
                "[",
                f"strafe left {vy_step} m/s",
                lambda: runtime.move(0.0, vy_step, 0.0),
                "motion",
            ),
            KeyBinding(
                "]",
                f"strafe right {vy_step} m/s",
                lambda: runtime.move(0.0, -vy_step, 0.0),
                "motion",
            ),
            KeyBinding("x", "stop moving", runtime.stop, "motion"),
        ]

    if isinstance(runtime, GestureControl):
        bindings += [
            KeyBinding("w", "wave hand", runtime.wave_hand, "gesture"),
            KeyBinding("k", "shake hand", runtime.shake_hand, "gesture"),
        ]

    bindings.append(KeyBinding("q", "quit (damps first)", quit_console, "console"))
    return bindings


def lifecycle_status(lifecycle) -> str:
    return lifecycle.status()


def build_session_bindings(session) -> list[KeyBinding]:
    """The command center: lifecycle keys plus selection and planner keys."""
    from embodied_control.robot.lifecycle import LifecycleState

    def quit_console() -> None:
        raise ConsoleQuit

    def report(result) -> None:
        if not result.ok:
            raise RuntimeError(result.detail)

    return [
        KeyBinding(KEY_SPACE, "DAMP (safety stop)", lambda: report(session.damp()), "safety"),
        KeyBinding("o", "mode: oracle / vla", lambda: report(session.toggle_mode()), "select"),
        KeyBinding("m", "motion: next", lambda: report(session.step_motion(1)), "select"),
        KeyBinding("M", "motion: previous", lambda: report(session.step_motion(-1)), "select"),
        KeyBinding("f", "start frame +25", lambda: report(session.step_frame(session.config.frame_step)), "select"),
        KeyBinding("F", "start frame -25", lambda: report(session.step_frame(-session.config.frame_step)), "select"),
        KeyBinding("p", "planner: start / stop", lambda: report(session.toggle_planner()), "select"),
        KeyBinding("r", "rebuild tracker for selection", lambda: report(session.rebuild()), "select"),
        KeyBinding("n", "next: advance one state", lambda: report(session.advance()), "ladder"),
        KeyBinding("a", "auto-advance to PRIMED", lambda: report(session.auto(LifecycleState.PRIMED)), "ladder"),
        KeyBinding("g", "go: PRIMED -> BLEND_IN -> RUNNING", lambda: report(session.go()), "episode"),
        KeyBinding("h", "hold: freeze on the last target", lambda: report(session.hold()), "episode"),
        KeyBinding("e", "retake: HOLD -> START_POSE_RAMP", lambda: report(session.retake()), "episode"),
        KeyBinding("l", "ack: lowered onto the feet", session.ack_lowered, "operator"),
        KeyBinding("H", "ack: hoist hooked, load taken", session.ack_hoisted, "operator"),
        KeyBinding("s", "recover to vendor stand (default end)", lambda: report(session.recover("vendor_stand")), "end"),
        KeyBinding("d", "release to vendor damp only", lambda: report(session.recover("vendor_damp")), "end"),
        KeyBinding("x", "abort the ladder (damp, hand back)", lambda: report(session.abort()), "end"),
        KeyBinding("q", "quit (damps, silences, restores)", quit_console, "console"),
    ]


def build_lifecycle_bindings(lifecycle) -> list[KeyBinding]:
    """Keys are requests; the lifecycle decides (docs/design/robot_lifecycle.md §5)."""
    from embodied_control.robot.lifecycle import LifecycleState

    def quit_console() -> None:
        raise ConsoleQuit

    def report(result) -> None:
        if not result.ok:
            raise RuntimeError(result.detail)

    return [
        KeyBinding(KEY_SPACE, "DAMP (safety stop)", lambda: report(lifecycle.damp()), "safety"),
        KeyBinding("n", "next: advance one state", lambda: report(lifecycle.advance()), "ladder"),
        KeyBinding("a", "auto-advance to PRIMED", lambda: report(lifecycle.auto(LifecycleState.PRIMED)), "ladder"),
        KeyBinding("g", "go: PRIMED -> BLEND_IN -> RUNNING", lambda: report(lifecycle.go()), "episode"),
        KeyBinding("h", "hold: freeze on the last target", lambda: report(lifecycle.hold()), "episode"),
        KeyBinding("e", "retake: HOLD -> START_POSE_RAMP", lambda: report(lifecycle.retake()), "episode"),
        KeyBinding("l", "ack: lowered onto the feet", lifecycle.ack_lowered, "operator"),
        KeyBinding("H", "ack: hoist hooked, load taken", lifecycle.ack_hoisted, "operator"),
        KeyBinding("s", "recover to vendor stand (default end)", lambda: report(lifecycle.recover("vendor_stand")), "end"),
        KeyBinding("d", "release to vendor damp only", lambda: report(lifecycle.recover("vendor_damp")), "end"),
        KeyBinding("x", "abort the ladder (damp, hand back)", lambda: report(lifecycle.abort()), "end"),
        KeyBinding("q", "quit (damps, silences, restores)", quit_console, "console"),
    ]
