"""Transport seam: map an endpoint ``scheme`` to a concrete policy client.

``http`` is our own debug-policy transport (stdlib, tiny container, no
codegen). ``openpi_websocket`` and ``gr00t_zmq`` (M4) speak real VLA
policies' *native* protocols instead of forcing them through ours — see
``docs/design/real_policy_adapters.md``. Every scheme implements the same
surface (``transport.base.PolicyClientProtocol``); the runner, backend, and
embodiment adapter call only that surface and never change.

Non-``http`` client modules import ``websockets``/``msgpack``/``pyzmq``,
which are NOT dependencies of the light default host env (see the
``transports`` pixi feature) — imported lazily inside this function so
constructing an ``http`` client never requires them.

This is the design's "internal transport interface" (D5): the wire format is a
detail behind ``endpoint.scheme``, not something the rollout loop knows about.
"""

from __future__ import annotations

from typing import Any

from embodied_control.logging.logger import EcLogger
from embodied_control.transport.client import PolicyClient

SUPPORTED_SCHEMES = ("http", "openpi_websocket", "gr00t_zmq")
PLANNED_SCHEMES: tuple[str, ...] = ()


def make_policy_client(
    scheme: str,
    host: str,
    port: int,
    timeout_s: float = 30.0,
    logger: EcLogger | None = None,
    action_dim: int | None = None,
    observation_mapping: dict[str, Any] | None = None,
):
    if scheme == "http":
        return PolicyClient(host, port, timeout_s=timeout_s, logger=logger)
    if scheme == "openpi_websocket":
        from embodied_control.transport.openpi_client import OpenPIWebsocketClient
        from embodied_control.transport.openpi_translate import OpenPIObservationMapping

        if action_dim is None:
            raise ValueError("openpi_websocket requires action_dim (see EndpointSpec.action_dim)")
        mapping = OpenPIObservationMapping(**(observation_mapping or {}))
        return OpenPIWebsocketClient(
            host, port, action_dim=action_dim, mapping=mapping, timeout_s=timeout_s, logger=logger
        )
    if scheme == "gr00t_zmq":
        from embodied_control.transport.gr00t_client import Gr00tZmqClient
        from embodied_control.transport.gr00t_translate import Gr00tObservationMapping

        if action_dim is None:
            raise ValueError("gr00t_zmq requires action_dim (see EndpointSpec.action_dim)")
        mapping = Gr00tObservationMapping(**(observation_mapping or {}))
        return Gr00tZmqClient(
            host, port, action_dim=action_dim, mapping=mapping, timeout_s=timeout_s, logger=logger
        )
    if scheme in PLANNED_SCHEMES:
        raise NotImplementedError(
            f"transport scheme {scheme!r} is planned but not implemented yet "
            f"(see docs/design/real_policy_adapters.md)"
        )
    raise ValueError(
        f"unknown transport scheme {scheme!r}; supported: {SUPPORTED_SCHEMES}, "
        f"planned: {PLANNED_SCHEMES}"
    )
