"""Generic robot runtime contract: lifecycle and mode, not per-tick actuation.

This is the high-level command axis — what the vendor's joystick drives. Joint
targets stay in the native control path; nothing here is safe to call from a
control thread, because a realization's verbs are typically blocking RPCs that
can take their whole timeout to return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable


class RobotMode(StrEnum):
    UNKNOWN = "unknown"
    ZERO_TORQUE = "zero_torque"
    DAMP = "damp"
    READY = "ready"
    VENDOR_CONTROL = "vendor"
    USER_CONTROL = "user"  # our low-level runtime owns rt/lowcmd
    FAULT = "fault"


class Capability(StrEnum):
    POSTURE = "posture"
    VELOCITY = "velocity"
    GESTURE = "gesture"


@dataclass(frozen=True)
class RobotInfo:
    robot_id: str
    transport: str
    capabilities: frozenset[Capability] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RobotHealth:
    reachable: bool
    mode: RobotMode
    detail: str = ""


class RobotError(RuntimeError):
    """Base for every robot runtime failure."""


class RobotCommandError(RobotError):
    """The robot refused a command."""


class RobotTransitionError(RobotError):
    """The requested verb is not legal from the robot's current mode."""


class RobotWriteGateError(RobotError):
    """A state-changing verb was called on a read-only runtime."""


# Mode graph shared by every realization: `damp` is reachable from anywhere,
# including FAULT, which is what makes it the safety floor rather than just
# another verb.
Transitions = dict[RobotMode, frozenset[RobotMode]]


def check_transition(
    transitions: Transitions, current: RobotMode, target: RobotMode
) -> None:
    if target == RobotMode.DAMP:
        return
    allowed = transitions.get(current, frozenset())
    if target not in allowed:
        legal = ", ".join(sorted(allowed)) or "nothing"
        raise RobotTransitionError(
            f"cannot go from {current} to {target}; legal targets are {legal}"
        )


WRITE_CONFIRM_TOKEN = "ENABLE_G1_LOWLEVEL"


def check_write_gate(enabled: bool, verb: str) -> None:
    if not enabled:
        raise RobotWriteGateError(
            f"{verb} changes robot state; construct the runtime with "
            f"writes_enabled=True (CLI: --enable-writes --confirm "
            f"{WRITE_CONFIRM_TOKEN})"
        )


class RobotRuntime(Protocol):
    def info(self) -> RobotInfo: ...

    def mode(self) -> RobotMode: ...

    def health(self) -> RobotHealth: ...

    def damp(self) -> None: ...

    def zero_torque(self) -> None: ...

    def ready(self) -> None: ...

    def close(self, *, damp: bool = True) -> None: ...


@runtime_checkable
class PostureControl(Protocol):
    def sit(self) -> None: ...

    def squat(self) -> None: ...

    def stand(self, height: float | None = None) -> None: ...


@runtime_checkable
class VelocityControl(Protocol):
    def move(
        self, vx: float, vy: float, vyaw: float, *, continuous: bool = False
    ) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class GestureControl(Protocol):
    def wave_hand(self, *, turn: bool = False) -> None: ...

    def shake_hand(self, stage: int = -1) -> None: ...
