"""Local subprocess runtime adapter for the policy service.

Launches ``python -m embodied_control.policies.debug_server`` as a child process
bound to a host port. Used for CI/tests and any host without Docker access. The
transport is identical to the Docker path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from embodied_control.logging.logger import EcLogger
from embodied_control.runtime.base import RuntimeHandle


class LocalRuntimeAdapter:
    engine = "local"

    def __init__(
        self,
        name: str,
        policy_type: str,
        action_dim: int,
        action_schema_id: str,
        host: str,
        port: int,
        seed: int,
        max_action_horizon: int,
        log_path: str,
        logger: EcLogger | None = None,
    ):
        self.name = name
        self.policy_type = policy_type
        self.action_dim = action_dim
        self.action_schema_id = action_schema_id
        self.host = host
        self.port = port
        self.seed = seed
        self.max_action_horizon = max_action_horizon
        self.log_path = log_path
        self.logger = logger or EcLogger.null()
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    def command(self) -> list[str]:
        return [
            sys.executable, "-m", "embodied_control.policies.debug_server",
            "--type", self.policy_type,
            "--action-dim", str(self.action_dim),
            "--action-schema-id", self.action_schema_id,
            "--host", self.host,
            "--port", str(self.port),
            "--seed", str(self.seed),
            "--max-action-horizon", str(self.max_action_horizon),
        ]

    def start(self) -> RuntimeHandle:
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            self.command(),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )
        self.logger.event(
            "runtime.local.started", phase="policy_launch",
            pid=self._proc.pid, port=self.port, policy_type=self.policy_type,
        )
        return RuntimeHandle(
            name=self.name,
            engine=self.engine,
            endpoint_host=self.host,
            endpoint_port=self.port,
            log_path=self.log_path,
            pid=self._proc.pid,
        )

    def logs(self, handle: RuntimeHandle) -> str:
        try:
            return Path(handle.log_path).read_text()
        except OSError:
            return ""

    def stop(self, handle: RuntimeHandle) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)
            self.logger.event(
                "runtime.local.stopped", phase="teardown",
                pid=self._proc.pid, returncode=self._proc.returncode,
            )
        if self._log_fh is not None:
            self._log_fh.close()
