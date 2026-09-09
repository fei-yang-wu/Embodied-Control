"""Unitree G1 realization of the robot runtime, over the SDK's "sport" service.

Every verb here is a blocking DDS RPC through ``ec_native.G1LocoClient``. Never
call one from the control thread: the native writer runs at 500 Hz with a
watchdog, and an RPC that blocks for its timeout starves the deadline and drops
the robot into damp.
"""

from __future__ import annotations

from embodied_control.robot.base import (
    Capability,
    RobotCommandError,
    RobotHealth,
    RobotInfo,
    RobotMode,
    Transitions,
    check_transition,
    check_write_gate,
)

# FSM ids, from the vendored SDK's g1_loco_client.hpp high-level wrappers.
_FSM_ZERO_TORQUE = 0
_FSM_DAMP = 1
_FSM_SQUAT = 2
_FSM_SIT = 3
_FSM_STAND_UP = 4
_FSM_START = 500

_FSM_TO_MODE = {
    _FSM_ZERO_TORQUE: RobotMode.ZERO_TORQUE,
    _FSM_DAMP: RobotMode.DAMP,
    _FSM_SQUAT: RobotMode.VENDOR_CONTROL,
    _FSM_SIT: RobotMode.VENDOR_CONTROL,
    _FSM_STAND_UP: RobotMode.READY,
    _FSM_START: RobotMode.READY,
}

# Takeover (MotionSwitcherClient::ReleaseMode, owned by the low-level tracker
# process) is only safe from the damp FSM state: the robot must be limp - and
# therefore hoisted - at handover. There is no "stand up, then take over" path,
# and this table is where that fact lives. USER_CONTROL is never entered
# through this runtime; the tracker reports it.
G1_TRANSITIONS: Transitions = {
    RobotMode.UNKNOWN: frozenset(
        {
            RobotMode.ZERO_TORQUE,
            RobotMode.DAMP,
            RobotMode.READY,
            RobotMode.VENDOR_CONTROL,
        }
    ),
    RobotMode.ZERO_TORQUE: frozenset({RobotMode.DAMP, RobotMode.READY}),
    RobotMode.DAMP: frozenset(
        {RobotMode.ZERO_TORQUE, RobotMode.READY, RobotMode.USER_CONTROL}
    ),
    RobotMode.READY: frozenset(
        {RobotMode.ZERO_TORQUE, RobotMode.VENDOR_CONTROL}
    ),
    RobotMode.VENDOR_CONTROL: frozenset(
        {RobotMode.READY, RobotMode.ZERO_TORQUE}
    ),
    RobotMode.USER_CONTROL: frozenset({RobotMode.VENDOR_CONTROL}),
    RobotMode.FAULT: frozenset(),
}


class G1Runtime:
    def __init__(
        self,
        network_interface: str,
        *,
        writes_enabled: bool = False,
        dds_domain: int = 0,
        timeout_seconds: float = 5.0,
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "G1Runtime needs the Unitree-enabled native build"
            ) from exc
        if not ec_native.WITH_UNITREE:
            raise RuntimeError(
                "ec_native was built without Unitree SDK2; set "
                "EC_UNITREE_SDK_ROOT and run `pixi run -e native build-native`"
            )
        self._writes_enabled = bool(writes_enabled)
        self._network_interface = str(network_interface)
        self._client = ec_native.G1LocoClient(
            network_interface=self._network_interface,
            dds_domain=int(dds_domain),
            timeout_seconds=float(timeout_seconds),
        )

    def info(self) -> RobotInfo:
        return RobotInfo(
            robot_id="unitree_g1",
            transport=f"dds:{self._network_interface}",
            capabilities=frozenset(
                {Capability.POSTURE, Capability.VELOCITY, Capability.GESTURE}
            ),
        )

    def mode(self) -> RobotMode:
        try:
            fsm_id = self._client.fsm_id
        except RuntimeError:
            return RobotMode.UNKNOWN
        return _FSM_TO_MODE.get(fsm_id, RobotMode.VENDOR_CONTROL)

    def health(self) -> RobotHealth:
        try:
            status = self._client.status()
        except RuntimeError as exc:
            return RobotHealth(
                reachable=False, mode=RobotMode.UNKNOWN, detail=str(exc)
            )
        return RobotHealth(
            reachable=True,
            mode=self.mode(),
            detail=f"fsm_id={status['fsm_id']} fsm_mode={status['fsm_mode']}",
        )

    def _command(self, verb: str, target: RobotMode) -> None:
        check_write_gate(self._writes_enabled, verb)
        check_transition(G1_TRANSITIONS, self.mode(), target)

    def damp(self) -> None:
        check_write_gate(self._writes_enabled, "damp")
        self._client.damp()

    def zero_torque(self) -> None:
        self._command("zero_torque", RobotMode.ZERO_TORQUE)
        self._client.zero_torque()

    def ready(self) -> None:
        self._command("ready", RobotMode.READY)
        self._client.stand_up()
        self._client.balance_stand()

    def close(self, *, damp: bool = True) -> None:
        if damp and self._writes_enabled:
            try:
                self.damp()
            except RuntimeError:
                pass

    def __enter__(self) -> G1Runtime:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # PostureControl

    def sit(self) -> None:
        self._command("sit", RobotMode.VENDOR_CONTROL)
        self._client.sit()

    def squat(self) -> None:
        self._command("squat", RobotMode.VENDOR_CONTROL)
        self._client.squat()

    def stand(self, height: float | None = None) -> None:
        self._command("stand", RobotMode.READY)
        self._client.stand_up()
        if height is not None:
            self._client.set_stand_height(float(height))

    # VelocityControl

    def move(
        self, vx: float, vy: float, vyaw: float, *, continuous: bool = False
    ) -> None:
        check_write_gate(self._writes_enabled, "move")
        if self.mode() is not RobotMode.READY:
            raise RobotCommandError(
                "move needs the robot in ready mode; call ready() first"
            )
        self._client.move(
            float(vx), float(vy), float(vyaw), continuous=bool(continuous)
        )

    def stop(self) -> None:
        check_write_gate(self._writes_enabled, "stop")
        self._client.stop_move()

    # GestureControl

    def wave_hand(self, *, turn: bool = False) -> None:
        check_write_gate(self._writes_enabled, "wave_hand")
        self._client.wave_hand(turn=bool(turn))

    def shake_hand(self, stage: int = -1) -> None:
        check_write_gate(self._writes_enabled, "shake_hand")
        self._client.shake_hand(stage=int(stage))
