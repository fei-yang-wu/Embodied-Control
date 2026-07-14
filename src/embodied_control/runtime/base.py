"""Common runtime handle + adapter protocol.

Runtime adapters are transport-agnostic launchers: they know how to start/stop
a command as a local process or Docker container and nothing about what that
command does. A long-lived networked service (the policy) has an endpoint; a
run-to-completion job (a delegated evaluator) does not -- both fields are
optional so ``RuntimeHandle`` covers either shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RuntimeHandle:
    name: str
    engine: str  # local | docker
    log_path: str
    endpoint_host: str | None = None
    endpoint_port: int | None = None
    container_name: str | None = None
    pid: int | None = None


class RuntimeAdapter(Protocol):
    def start(self) -> RuntimeHandle: ...

    def logs(self, handle: RuntimeHandle) -> str: ...

    def wait(self, timeout_s: float) -> int:
        """Block until the runtime exits (or `timeout_s` elapses) and return its exit code."""
        ...

    def stop(self, handle: RuntimeHandle) -> None: ...
