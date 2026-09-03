"""Hardware-free robot runtime: an in-memory mode machine on the same graph.

Per this repo's "prove new architecture with something cheap" rule, this is
what exercises the gate, the transition table and the capability branching in
the light test env. It honours the G1's transition graph, so a sequence that
passes here is a sequence the G1 will accept.
"""

from __future__ import annotations

from embodied_control.robot.base import (
    Capability,
    RobotCommandError,
    RobotHealth,
    RobotInfo,
    RobotMode,
    check_transition,
    check_write_gate,
)
from embodied_control.robot.g1 import G1_TRANSITIONS


class FakeRobotRuntime:
    def __init__(
        self,
        *,
        writes_enabled: bool = False,
        mode: RobotMode = RobotMode.DAMP,
        capabilities: frozenset[Capability] | None = None,
    ) -> None:
        self._writes_enabled = bool(writes_enabled)
        self._mode = mode
        self._capabilities = (
            capabilities
            if capabilities is not None
            else frozenset(
                {Capability.POSTURE, Capability.VELOCITY, Capability.GESTURE}
            )
        )
        self.calls: list[str] = []
        self.closed = False

    def info(self) -> RobotInfo:
        return RobotInfo(
            robot_id="fake_robot",
            transport="memory",
            capabilities=self._capabilities,
        )

    def mode(self) -> RobotMode:
        return self._mode

    def health(self) -> RobotHealth:
        return RobotHealth(reachable=True, mode=self._mode)

    def _command(self, verb: str, target: RobotMode) -> None:
        check_write_gate(self._writes_enabled, verb)
        check_transition(G1_TRANSITIONS, self._mode, target)
        self.calls.append(verb)
        self._mode = target

    def damp(self) -> None:
        check_write_gate(self._writes_enabled, "damp")
        self.calls.append("damp")
        self._mode = RobotMode.DAMP

    def zero_torque(self) -> None:
        self._command("zero_torque", RobotMode.ZERO_TORQUE)

    def ready(self) -> None:
        self._command("ready", RobotMode.READY)

    def close(self, *, damp: bool = True) -> None:
        if damp and self._writes_enabled:
            self.damp()
        self.closed = True

    def __enter__(self) -> FakeRobotRuntime:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def sit(self) -> None:
        self._command("sit", RobotMode.VENDOR_CONTROL)

    def squat(self) -> None:
        self._command("squat", RobotMode.VENDOR_CONTROL)

    def stand(self, height: float | None = None) -> None:
        self._command("stand", RobotMode.READY)

    def move(
        self, vx: float, vy: float, vyaw: float, *, continuous: bool = False
    ) -> None:
        check_write_gate(self._writes_enabled, "move")
        if self._mode is not RobotMode.READY:
            raise RobotCommandError(
                "move needs the robot in ready mode; call ready() first"
            )
        self.calls.append(f"move({vx},{vy},{vyaw})")

    def stop(self) -> None:
        check_write_gate(self._writes_enabled, "stop")
        self.calls.append("stop")

    def wave_hand(self, *, turn: bool = False) -> None:
        check_write_gate(self._writes_enabled, "wave_hand")
        self.calls.append("wave_hand")

    def shake_hand(self, stage: int = -1) -> None:
        check_write_gate(self._writes_enabled, "shake_hand")
        self.calls.append("shake_hand")
