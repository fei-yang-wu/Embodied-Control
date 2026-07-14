"""Runtime adapter command construction (pure; no docker daemon or subprocess
needed -- these test the argv-building logic, not actual process/container
launch, which the docker smoke tests cover manually)."""

from __future__ import annotations

from embodied_control.config.schemas import MountSpec
from embodied_control.runtime.docker import DockerRuntimeAdapter
from embodied_control.runtime.local import LocalRuntimeAdapter


def test_docker_bridge_network_publishes_port():
    adapter = DockerRuntimeAdapter(
        name="policy", image="img:latest", container_name="c1",
        command=["--foo", "bar"], log_path="/tmp/x.log",
        network="bridge", host_port=1234, container_port=8000,
    )
    cmd = adapter.run_command()
    assert "-p" in cmd
    assert cmd[cmd.index("-p") + 1] == "127.0.0.1:1234:8000"
    assert "--network" not in cmd
    assert cmd[-2:] == ["--foo", "bar"]


def test_docker_host_network_has_no_published_port():
    adapter = DockerRuntimeAdapter(
        name="sim", image="img:latest", container_name="c2",
        command=["--config", "/generated/x.json"], log_path="/tmp/x.log",
        network="host",
    )
    cmd = adapter.run_command()
    assert "--network" in cmd
    assert cmd[cmd.index("--network") + 1] == "host"
    assert "-p" not in cmd


def test_docker_bridge_without_ports_raises():
    try:
        DockerRuntimeAdapter(
            name="x", image="img", container_name="c", command=[],
            log_path="/tmp/x.log", network="bridge",
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_docker_mounts_and_env_flags():
    adapter = DockerRuntimeAdapter(
        name="sim", image="img:latest", container_name="c3",
        command=["run"], log_path="/tmp/x.log", network="host",
        mounts=[
            MountSpec(source="/host/generated", target="/generated", mode="ro"),
            MountSpec(source="/host/raw", target="/raw", mode="rw"),
        ],
        env={"FOO": "bar"},
    )
    cmd = adapter.run_command()
    joined = " ".join(cmd)
    assert "-v /host/generated:/generated:ro" in joined
    assert "-v /host/raw:/raw:rw" in joined
    assert "-e FOO=bar" in joined


def test_docker_run_command_is_detached_and_named():
    adapter = DockerRuntimeAdapter(
        name="policy", image="img:latest", container_name="my_container",
        command=[], log_path="/tmp/x.log", network="host",
    )
    cmd = adapter.run_command()
    assert cmd[1:4] == ["run", "-d", "--name"]
    assert "my_container" in cmd


def test_local_adapter_command_is_used_verbatim(tmp_path):
    adapter = LocalRuntimeAdapter(
        name="sim", command=["python3", "-c", "print(1)"],
        log_path=str(tmp_path / "x.log"),
    )
    assert adapter.command == ["python3", "-c", "print(1)"]


def test_local_adapter_wait_before_start_raises(tmp_path):
    adapter = LocalRuntimeAdapter(
        name="sim", command=["true"], log_path=str(tmp_path / "x.log"),
    )
    try:
        adapter.wait(timeout_s=1.0)
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_local_adapter_runs_and_waits_for_exit(tmp_path):
    adapter = LocalRuntimeAdapter(
        name="sim", command=["python3", "-c", "import sys; sys.exit(3)"],
        log_path=str(tmp_path / "x.log"),
    )
    handle = adapter.start()
    try:
        code = adapter.wait(timeout_s=5.0)
        assert code == 3
    finally:
        adapter.stop(handle)


def test_local_adapter_wait_timeout_raises(tmp_path):
    adapter = LocalRuntimeAdapter(
        name="sim", command=["python3", "-c", "import time; time.sleep(30)"],
        log_path=str(tmp_path / "x.log"),
    )
    handle = adapter.start()
    try:
        try:
            adapter.wait(timeout_s=0.2)
            assert False, "expected TimeoutError"
        except TimeoutError:
            pass
    finally:
        adapter.stop(handle)  # must clean up the still-running sleep process
