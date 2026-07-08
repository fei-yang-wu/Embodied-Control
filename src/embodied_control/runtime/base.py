"""Common runtime handle + adapter protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class RuntimeHandle:
    name: str
    engine: str  # local | docker
    endpoint_host: str
    endpoint_port: int
    log_path: str
    container_name: str | None = None
    pid: int | None = None


class RuntimeAdapter(Protocol):
    def start(self) -> RuntimeHandle: ...

    def logs(self, handle: RuntimeHandle) -> str: ...

    def stop(self, handle: RuntimeHandle) -> None: ...
