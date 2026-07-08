"""Docker runtime adapter for the policy service.

Runs the blank policy service in its own container, publishing its port only on
127.0.0.1 (design §18: explicit, localhost-bound network exposure; never mount the
docker socket). The container image's ENTRYPOINT is the debug server; we append
the resolved args as the container command.

Requires access to the Docker daemon. On a machine where the current shell is not
yet in the ``docker`` group, run under ``newgrp docker`` / ``sg docker -c`` or
reboot after being added to the group.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

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
        policy_type: str,
        action_dim: int,
        action_schema_id: str,
        host_port: int,
        container_port: int,
        seed: int,
        max_action_horizon: int,
        log_path: str,
        bind_host: str = "127.0.0.1",
        shm_size: str | None = None,
        logger: EcLogger | None = None,
    ):
        self.name = name
        self.image = image
        self.container_name = container_name
        self.policy_type = policy_type
        self.action_dim = action_dim
        self.action_schema_id = action_schema_id
        self.host_port = host_port
        self.container_port = container_port
        self.seed = seed
        self.max_action_horizon = max_action_horizon
        self.log_path = log_path
        self.bind_host = bind_host
        self.shm_size = shm_size
        self.logger = logger or EcLogger.null()

    def _docker(self) -> str:
        exe = shutil.which("docker")
        if exe is None:
            raise DockerRuntimeError(
                "docker executable not found on PATH; install Docker or use runtime.type=local"
            )
        return exe

    def container_command(self) -> list[str]:
        # Args appended to the image ENTRYPOINT (the debug server). The service
        # binds 0.0.0.0 inside the container; Docker maps it to the host port.
        return [
            "--type", self.policy_type,
            "--action-dim", str(self.action_dim),
            "--action-schema-id", self.action_schema_id,
            "--host", "0.0.0.0",
            "--port", str(self.container_port),
            "--seed", str(self.seed),
            "--max-action-horizon", str(self.max_action_horizon),
        ]

    def run_command(self) -> list[str]:
        cmd = [
            self._docker(), "run", "-d",
            "--name", self.container_name,
            "-p", f"{self.bind_host}:{self.host_port}:{self.container_port}",
        ]
        if self.shm_size:
            cmd += ["--shm-size", self.shm_size]
        cmd += [self.image]
        cmd += self.container_command()
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
                f"failed to start policy container from image {self.image!r}:\n{proc.stderr.strip()}"
            )
        self.logger.event(
            "runtime.docker.started", phase="policy_launch",
            image=self.image, container=self.container_name,
            host_port=self.host_port, container_port=self.container_port,
        )
        return RuntimeHandle(
            name=self.name,
            engine=self.engine,
            endpoint_host=self.bind_host,
            endpoint_port=self.host_port,
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
