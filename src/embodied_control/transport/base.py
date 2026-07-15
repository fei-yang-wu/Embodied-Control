"""Structural interface every policy transport client implements.

Three implementations (``http``, ``openpi_websocket``, and ``gr00t_zmq`` once
M4 lands) is the point where this repo's "no abstraction for one instance"
convention flips -- see ``docs/design/real_policy_adapters.md``. A
``typing.Protocol`` documents the shared surface without forcing inheritance,
matching ``runtime/base.py::RuntimeAdapter`` for the same reason. Nothing
downstream (``orchestration/runner.py``, the delegated evaluators) imports
this directly -- it exists for readability and type-checking, not runtime
dispatch (that's ``transport/factory.py``).
"""

from __future__ import annotations

from typing import Protocol


class PolicyClientProtocol(Protocol):
    def health(self) -> dict: ...

    def wait_healthy(self, timeout_s: float = 30.0, interval_s: float = 0.2) -> dict: ...

    def describe(self) -> dict: ...

    def reset(self, episode_keys, seed: int) -> dict: ...

    def act(self, request_id, episode_keys, observations, requested_horizon) -> dict: ...
