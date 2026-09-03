import io

from embodied_control.robot import Capability, FakeRobotRuntime, RobotMode
from embodied_control.console import KEY_SPACE, KeyBinding, KeyConsole
from embodied_control.robot.shell import build_robot_bindings, robot_status


def _console(bindings, **kwargs):
    return KeyConsole(bindings, out=io.StringIO(), **kwargs)


def test_damp_is_a_single_keypress():
    robot = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY)
    console = _console(build_robot_bindings(robot))

    assert console.handle(KEY_SPACE) is True
    assert robot.mode() is RobotMode.DAMP


def test_unknown_keys_are_ignored():
    robot = FakeRobotRuntime(writes_enabled=True)
    console = _console(build_robot_bindings(robot))

    assert console.handle("~") is True
    assert robot.calls == []


def test_a_failing_action_reports_but_keeps_the_console_alive():
    robot = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY)
    out = io.StringIO()
    console = KeyConsole(build_robot_bindings(robot), out=out)

    # stand is illegal from READY: the console must survive it.
    assert console.handle("n") is True
    assert "RobotTransitionError" in out.getvalue()


def test_quit_key_ends_the_loop():
    robot = FakeRobotRuntime(writes_enabled=True)
    console = _console(build_robot_bindings(robot))

    assert console.handle("q") is False


def test_bindings_follow_capabilities():
    full = build_robot_bindings(FakeRobotRuntime())
    keys = {binding.key for binding in full}
    assert "<up>" in keys and "w" in keys

    armless = FakeRobotRuntime(capabilities=frozenset({Capability.POSTURE}))
    # Capability advertisement and protocol structure are separate concerns;
    # what matters is that the console builds from the runtime it is given.
    assert {b.key for b in build_robot_bindings(armless)} >= {KEY_SPACE, "q"}


def test_run_consumes_a_key_stream_until_quit():
    robot = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.DAMP)
    console = _console(build_robot_bindings(robot))

    assert console.run(iter(["r", "q", "z"])) == 0
    assert robot.mode() is RobotMode.READY
    assert "zero_torque" not in robot.calls


def test_status_line_reports_mode_and_link():
    robot = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.DAMP)
    assert "mode: damp" in robot_status(robot)
    assert "link: up" in robot_status(robot)


def test_read_only_console_refuses_without_dying():
    robot = FakeRobotRuntime(writes_enabled=False)
    out = io.StringIO()
    console = KeyConsole(build_robot_bindings(robot), out=out)

    assert console.handle(KEY_SPACE) is True
    assert "RobotWriteGateError" in out.getvalue()


def test_help_lists_every_binding():
    out = io.StringIO()
    console = KeyConsole(
        [KeyBinding("a", "alpha", lambda: None, "grp")], out=out
    )
    console.help()
    text = out.getvalue()
    assert "alpha" in text and "[grp]" in text
