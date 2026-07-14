"""Local subprocess runtime adapter.

Generic: the caller supplies the full command to run (e.g. the supervisor
builds the policy debug-server invocation; the delegated rollout builds the
fake evaluator invocation). Used for CI/tests and any host without Docker
access -- the transport/contract is identical to the Docker path.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from embodied_control.logging.logger import EcLogger
from embodied_control.runtime.base import RuntimeHandle


class LocalRuntimeAdapter:
    engine = "local"

    def __init__(
        self,
        name: str,
        command: list[str],
        log_path: str,
        endpoint_host: str | None = None,
        endpoint_port: int | None = None,
        logger: EcLogger | None = None,
    ):
        self.name = name
        self.command = list(command)
        self.log_path = log_path
        self.endpoint_host = endpoint_host
        self.endpoint_port = endpoint_port
        self.logger = logger or EcLogger.null()
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    def start(self) -> RuntimeHandle:
        Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "w")
        self._proc = subprocess.Popen(
            self.command,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
        )
        self.logger.event(
            "runtime.local.started", phase="launch",
            pid=self._proc.pid, command=" ".join(self.command),
        )
        return RuntimeHandle(
            name=self.name,
            engine=self.engine,
            endpoint_host=self.endpoint_host,
            endpoint_port=self.endpoint_port,
            log_path=self.log_path,
            pid=self._proc.pid,
        )

    def logs(self, handle: RuntimeHandle) -> str:
        try:
            return Path(handle.log_path).read_text()
        except OSError:
            return ""

    def wait(self, timeout_s: float) -> int:
        """Block until the process exits and return its exit code.

        Raises ``TimeoutError`` if it doesn't exit within `timeout_s` -- this
        method only waits, it does not kill; call ``stop()`` for that (matches
        ``DockerRuntimeAdapter.wait()``'s contract so callers handle both
        uniformly, typically via ``try: ... finally: adapter.stop(handle)``).
        """
        assert self._proc is not None, "wait() called before start()"
        try:
            return self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            self.logger.warning("runtime.local.wait_timeout", pid=self._proc.pid, timeout_s=timeout_s)
            raise TimeoutError(
                f"process (pid={self._proc.pid}) did not exit within {timeout_s}s"
            ) from exc

    def stop(self, handle: RuntimeHandle) -> None:
        if self._proc is not None:
            if self._proc.poll() is None:  # still running: terminate (a completed run-to-exit
                self._proc.terminate()      # process, e.g. after wait(), is already stopped)
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
