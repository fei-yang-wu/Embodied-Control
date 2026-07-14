"""Docker runtime adapter.

Generic: the caller supplies the container's command (appended to the image's
ENTRYPOINT), any bind mounts, and a network mode. Two shapes in use:

- A long-lived networked service (the policy): ``network="bridge"`` with a
  published port (design §18: explicit, localhost-bound network exposure;
  never mount the docker socket).
- A run-to-completion job (a delegated evaluator): ``network="host"`` so it
  can reach the policy container's published port at ``127.0.0.1:<host_port>``
  without container-to-container DNS (which Docker's bridge network provides
  but Apptainer/HPC does not -- host networking is the portable choice, see
  the design review's Apptainer-parity finding) -- and no port to publish
  itself, since it only makes outbound requests.

Requires access to the Docker daemon. On a machine where the current shell is not
yet in the ``docker`` group, run under ``newgrp docker`` / ``sg docker -c`` or
reboot after being added to the group.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Literal

from embodied_control.config.schemas import MountSpec
from embodied_control.logging.logger import EcLogger
from embodied_control.runtime.base import RuntimeHandle


class DockerRuntimeError(RuntimeError):
    pass


class DockerRuntimeAdapter:
    engine = "docker"

    def __init__(
        self,
        name: str,
        image: str,
        container_name: str,
        command: list[str],
        log_path: str,
        network: Literal["bridge", "host"] = "bridge",
        host_port: int | None = None,
        container_port: int | None = None,
        bind_host: str = "127.0.0.1",
        mounts: list[MountSpec] = (),
        env: dict[str, str] = None,
        shm_size: str | None = None,
        logger: EcLogger | None = None,
    ):
        self.name = name
        self.image = image
        self.container_name = container_name
        self.command = list(command)
        self.log_path = log_path
        self.network = network
        self.host_port = host_port
        self.container_port = container_port
        self.bind_host = bind_host
        self.mounts = list(mounts)
        self.env = dict(env or {})
        self.shm_size = shm_size
        self.logger = logger or EcLogger.null()

        if network == "bridge" and (host_port is None or container_port is None):
            raise ValueError("network='bridge' requires host_port and container_port")

    def _docker(self) -> str:
        exe = shutil.which("docker")
        if exe is None:
            raise DockerRuntimeError(
                "docker executable not found on PATH; install Docker or use runtime.type=local"
            )
        return exe

    def run_command(self) -> list[str]:
        cmd = [self._docker(), "run", "-d", "--name", self.container_name]
        if self.network == "host":
            cmd += ["--network", "host"]
        else:
            cmd += ["-p", f"{self.bind_host}:{self.host_port}:{self.container_port}"]
        for m in self.mounts:
            cmd += ["-v", f"{m.source}:{m.target}:{m.mode}"]
        for k, v in self.env.items():
            cmd += ["-e", f"{k}={v}"]
        if self.shm_size:
            cmd += ["--shm-size", self.shm_size]
        cmd += [self.image]
        cmd += self.command
        return cmd

    def start(self) -> RuntimeHandle:
        # Best-effort cleanup of a stale container with the same name.
        subprocess.run(
            [self._docker(), "rm", "-f", self.container_name],
            capture_output=True, text=True,
        )
        proc = subprocess.run(self.run_command(), capture_output=True, text=True)
        if proc.returncode != 0:
            self.logger.error(
                "runtime.docker.start_failed", image=self.image,
                container=self.container_name, stderr=proc.stderr.strip(),
            )
            raise DockerRuntimeError(
                f"failed to start container from image {self.image!r}:\n{proc.stderr.strip()}"
            )
        self.logger.event(
            "runtime.docker.started", phase="launch",
            image=self.image, container=self.container_name, network=self.network,
            host_port=self.host_port, container_port=self.container_port,
        )
        return RuntimeHandle(
            name=self.name,
            engine=self.engine,
            endpoint_host=self.bind_host if self.network == "bridge" else None,
            endpoint_port=self.host_port if self.network == "bridge" else None,
            log_path=self.log_path,
            container_name=self.container_name,
        )

    def logs(self, handle: RuntimeHandle) -> str:
        proc = subprocess.run(
            [self._docker(), "logs", self.container_name],
            capture_output=True, text=True,
        )
        text = (proc.stdout or "") + (proc.stderr or "")
        try:
            Path(handle.log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(handle.log_path).write_text(text)
        except OSError:
            pass
        return text

    def wait(self, timeout_s: float) -> int:
        """Block until the container exits (or `timeout_s` elapses) and return its exit code."""
        try:
            proc = subprocess.run(
                [self._docker(), "wait", self.container_name],
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            self.logger.warning(
                "runtime.docker.wait_timeout", container=self.container_name, timeout_s=timeout_s,
            )
            raise TimeoutError(
                f"container {self.container_name!r} did not exit within {timeout_s}s"
            ) from exc
        if proc.returncode != 0:
            raise DockerRuntimeError(
                f"'docker wait {self.container_name}' failed: {proc.stderr.strip()}"
            )
        return int(proc.stdout.strip())

    def stop(self, handle: RuntimeHandle) -> None:
        # Capture logs before removal so the run directory has the container's output.
        self.logs(handle)
        subprocess.run(
            [self._docker(), "rm", "-f", self.container_name],
            capture_output=True, text=True,
        )
        self.logger.event(
            "runtime.docker.stopped", phase="teardown", container=self.container_name,
        )
