import pytest

from embodied_control.cli import main
from embodied_control.robot import (
    Capability,
    FakeRobotRuntime,
    GestureControl,
    PostureControl,
    RobotCommandError,
    RobotMode,
    RobotTransitionError,
    RobotWriteGateError,
    VelocityControl,
    open_robot,
)


def test_getters_are_ungated():
    robot = FakeRobotRuntime()
    assert robot.info().robot_id == "fake_robot"
    assert robot.mode() is RobotMode.DAMP
    assert robot.health().reachable


def test_mutating_verbs_need_the_write_gate():
    robot = FakeRobotRuntime()
    with pytest.raises(RobotWriteGateError):
        robot.ready()
    with pytest.raises(RobotWriteGateError):
        robot.damp()


def test_damp_is_legal_from_every_mode_including_fault():
    for mode in RobotMode:
        robot = FakeRobotRuntime(writes_enabled=True, mode=mode)
        robot.damp()
        assert robot.mode() is RobotMode.DAMP


def test_takeover_is_only_legal_from_damp():
    from embodied_control.robot.base import check_transition
    from embodied_control.robot.g1 import G1_TRANSITIONS

    check_transition(G1_TRANSITIONS, RobotMode.DAMP, RobotMode.USER_CONTROL)
    for mode in (RobotMode.READY, RobotMode.VENDOR_CONTROL, RobotMode.FAULT):
        with pytest.raises(RobotTransitionError):
            check_transition(G1_TRANSITIONS, mode, RobotMode.USER_CONTROL)


def test_stand_is_illegal_when_already_ready():
    robot = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY)
    with pytest.raises(RobotTransitionError):
        robot.stand()


def test_move_requires_ready_mode():
    robot = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.DAMP)
    with pytest.raises(RobotCommandError):
        robot.move(0.2, 0.0, 0.0)

    robot.ready()
    robot.move(0.2, 0.0, 0.0)
    assert "move(0.2,0.0,0.0)" in robot.calls


def test_capabilities_are_declared_not_assumed():
    full = FakeRobotRuntime()
    assert isinstance(full, PostureControl)
    assert isinstance(full, VelocityControl)
    assert isinstance(full, GestureControl)

    arm = FakeRobotRuntime(capabilities=frozenset())
    assert arm.info().capabilities == frozenset()
    assert Capability.VELOCITY not in arm.info().capabilities


def test_close_damps_when_writes_are_enabled():
    robot = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY)
    robot.close()
    assert robot.mode() is RobotMode.DAMP
    assert robot.closed


def test_close_can_leave_the_robot_in_its_mode():
    robot = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY)
    robot.close(damp=False)
    assert robot.mode() is RobotMode.READY
    assert robot.closed


def test_open_robot_rejects_unknown_and_needs_network_for_g1():
    with pytest.raises(ValueError):
        open_robot("nosuchrobot")
    with pytest.raises(ValueError):
        open_robot("g1")
    assert isinstance(open_robot("fake"), FakeRobotRuntime)


def test_cli_status_is_ungated(capsys):
    assert main(["robot", "status", "--robot", "fake"]) == 0
    out = capsys.readouterr().out
    assert "fake_robot" in out
    assert "mode: damp" in out


def test_cli_mutating_verb_refuses_without_confirmation(capsys):
    assert main(["robot", "damp", "--robot", "fake"]) == 2
    assert "FAIL" in capsys.readouterr().out

    assert (
        main(["robot", "damp", "--robot", "fake", "--enable-writes"]) == 2
    )
    assert "ENABLE_G1_LOWLEVEL" in capsys.readouterr().out


def test_cli_mutating_verb_runs_when_confirmed(capsys):
    code = main(
        [
            "robot",
            "ready",
            "--robot",
            "fake",
            "--enable-writes",
            "--confirm",
            "ENABLE_G1_LOWLEVEL",
        ]
    )
    assert code == 0
    assert "ready: ok (mode ready)" in capsys.readouterr().out


def test_g1_health_reads_the_fsm_when_balance_readings_are_refused():
    """On the robot this morning: `ec robot status` said unreachable / unknown
    while the robot hung limp in damp, because the loco service refuses
    GetBalanceMode outside a standing FSM (7301, LocoState not available)."""
    from embodied_control.robot.g1 import G1Runtime

    class LimpLocoClient:
        fsm_id = 1
        fsm_mode = 0

        def status(self):
            raise RuntimeError("G1 sport service refused GetBalanceMode (status 7301)")

    runtime = G1Runtime.__new__(G1Runtime)
    runtime._client = LimpLocoClient()
    runtime._writes_enabled = False
    health = runtime.health()
    assert health.reachable and health.mode is RobotMode.DAMP
    assert "fsm_id=1" in health.detail and "7301" in health.detail

    class DeadLocoClient(LimpLocoClient):
        @property
        def fsm_id(self):
            raise RuntimeError("timeout")

    runtime._client = DeadLocoClient()
    assert not runtime.health().reachable
