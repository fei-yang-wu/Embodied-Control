"""OpenPI-native transport client: websocket + msgpack-numpy.

Speaks OpenPI's actual wire protocol, verified against
``src/openpi/serving/websocket_policy_server.py`` and
``packages/openpi-client/src/openpi_client/websocket_client_policy.py``
(Physical-Intelligence/openpi, 2026-07-14):

- A plain HTTP GET ``/healthz`` is intercepted *before* the websocket
  upgrade (``process_request`` on the real server) -- ``health()`` doesn't
  need to open a websocket session at all.
- On websocket connect, the server immediately sends one msgpack-numpy-packed
  metadata dict, unprompted. That becomes ``describe()``'s cached value.
- ``infer(obs)`` is the entire ``act()`` cycle: one msgpack-numpy ``obs``
  dict sent, one msgpack-numpy action dict received back (the real server
  adds a ``server_timing`` key before sending).
- ``reset()`` is a **no-op on the wire** on OpenPI's own client -- nothing is
  sent for it at all. There's no server-side per-episode state to clear;
  chunk-caching lives on our side (``transport.chunking.ChunkScheduler``),
  not the server's.
- On a server-side exception, the real server sends the traceback back as a
  plain **string** (a text frame, not a msgpack binary frame) before closing
  the connection -- ``recv()`` returning ``str`` instead of ``bytes`` is the
  error signal, exactly like the real client checks.

Scoped to a single env per client connection (see
``docs/design/real_policy_adapters.md``, "Single-env scoping for real
adapters") -- ``act``/``reset`` reject anything but exactly one episode key.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request

import msgpack_numpy
import websockets.sync.client

from embodied_control.logging.logger import EcLogger
from embodied_control.transport.client import PolicyClientError
from embodied_control.transport.openpi_translate import (
    OpenPIObservationMapping,
    observation_to_openpi,
    openpi_action_to_chunk,
)


class OpenPIWebsocketClient:
    def __init__(
        self,
        host: str,
        port: int,
        action_dim: int,
        mapping: OpenPIObservationMapping | None = None,
        api_key: str | None = None,
        timeout_s: float = 30.0,
        logger: EcLogger | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.action_dim = action_dim
        self.mapping = mapping or OpenPIObservationMapping()
        self._api_key = api_key
        self.timeout_s = timeout_s
        self.logger = logger or EcLogger.null()
        self._uri = f"ws://{host}:{self.port}"
        self._ws: websockets.sync.client.ClientConnection | None = None
        self._server_metadata: dict | None = None

    # --- low-level -------------------------------------------------------
    def _connect(self) -> None:
        if self._ws is not None:
            return
        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        self.logger.debug("openpi.connecting", uri=self._uri)
        try:
            self._ws = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None, additional_headers=headers,
                open_timeout=self.timeout_s,
            )
        except OSError as exc:
            raise PolicyClientError(f"cannot reach openpi policy service at {self._uri}: {exc}") from exc
        self._server_metadata = msgpack_numpy.unpackb(self._ws.recv())
        self.logger.debug("openpi.connected", metadata_keys=sorted(self._server_metadata))

    # --- operations ------------------------------------------------------
    def health(self) -> dict:
        url = f"http://{self.host}:{self.port}/healthz"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout_s) as resp:
                ok = resp.status == 200
        except (urllib.error.URLError, OSError) as exc:
            raise PolicyClientError(f"cannot reach openpi policy service at {url}: {exc}") from exc
        return {"status": "ok" if ok else "error", "model_loaded": ok}

    def wait_healthy(self, timeout_s: float = 30.0, interval_s: float = 0.2) -> dict:
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                h = self.health()
                if h.get("status") == "ok":
                    self.logger.debug("openpi.health_check_ok", endpoint=self._uri)
                    return h
            except PolicyClientError as exc:
                last_err = exc
                self.logger.debug("openpi.health_check_retry", endpoint=self._uri, error=str(exc))
            time.sleep(interval_s)
        self.logger.warning("openpi.health_check_timeout", endpoint=self._uri, timeout_s=timeout_s)
        raise PolicyClientError(
            f"openpi policy service at {self._uri} not healthy within {timeout_s}s"
            + (f" (last error: {last_err})" if last_err else "")
        )

    def describe(self) -> dict:
        self._connect()
        return {
            "action_dim": self.action_dim,
            "policy_id": "openpi-remote",
            "server_metadata": self._server_metadata,
        }

    def reset(self, episode_keys, seed: int) -> dict:
        if len(episode_keys) != 1:
            raise PolicyClientError(
                f"OpenPIWebsocketClient supports exactly one episode key per call, got {len(episode_keys)}"
            )
        self._connect()
        # Real OpenPI client's reset() is a client-side no-op -- nothing sent
        # over the wire (see module docstring). `seed` has no wire equivalent
        # for a trained checkpoint; only logged, never forwarded.
        self.logger.debug("openpi.reset_noop", episode_key=episode_keys[0], seed=seed)
        return {"status": "ok"}

    def act(self, request_id, episode_keys, observations, requested_horizon) -> dict:
        if len(episode_keys) != 1 or len(observations) != 1:
            raise PolicyClientError(
                "OpenPIWebsocketClient supports exactly one (episode_key, observation) pair per call, "
                f"got {len(episode_keys)} keys / {len(observations)} observations"
            )
        self._connect()
        payload = observation_to_openpi(observations[0], self.mapping)

        start = time.monotonic()
        self._ws.send(msgpack_numpy.packb(payload))
        response = self._ws.recv()
        total_ms = (time.monotonic() - start) * 1000.0
        if isinstance(response, str):
            raise PolicyClientError(f"openpi inference server error:\n{response}")

        action = msgpack_numpy.unpackb(response)
        chunk = openpi_action_to_chunk(action, self.mapping)
        # requested_horizon is advisory only for a real checkpoint -- its
        # chunk length is fixed by training config, not requestable per-call
        # (see docs/design/real_policy_adapters.md).
        return {
            "status": "ok",
            "actions": [{"action_chunk": chunk}],
            "timing": {"total_ms": total_ms, **action.get("server_timing", {})},
        }

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None
