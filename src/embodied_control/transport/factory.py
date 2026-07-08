"""Transport seam: map an endpoint ``scheme`` to a concrete policy client.

M1 ships only the ``http`` transport (stdlib, tiny container, no codegen). When
image observations and batching arrive and throughput matters, add ``grpc`` and
``zmq`` transports here — the runner, backend, and embodiment adapter call only
the ``PolicyClient`` surface (health/describe/reset/act) and never change.

This is the design's "internal transport interface" (D5): the wire format is a
detail behind ``endpoint.scheme``, not something the rollout loop knows about.
"""

from __future__ import annotations

from embodied_control.logging.logger import EcLogger
from embodied_control.transport.client import PolicyClient

SUPPORTED_SCHEMES = ("http",)
# Planned, not yet implemented — surfaced so config validation can give a clear error:
PLANNED_SCHEMES = ("grpc", "zmq")


def make_policy_client(
    scheme: str, host: str, port: int, timeout_s: float = 30.0, logger: EcLogger | None = None
):
    if scheme == "http":
        return PolicyClient(host, port, timeout_s=timeout_s, logger=logger)
    if scheme in PLANNED_SCHEMES:
        raise NotImplementedError(
            f"transport scheme {scheme!r} is planned for the image/high-throughput "
            f"milestone but not implemented yet; use 'http' for the blank-policy path"
        )
    raise ValueError(
        f"unknown transport scheme {scheme!r}; supported: {SUPPORTED_SCHEMES}, "
        f"planned: {PLANNED_SCHEMES}"
    )
