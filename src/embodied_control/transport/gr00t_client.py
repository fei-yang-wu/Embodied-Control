"""GR00T-native transport client: ZeroMQ (REQ/REP) + msgpack.

Speaks GR00T's actual wire protocol, verified live against a real,
locally-running server (``NVIDIA/Isaac-GR00T``, checked out at a 2026-04-17
commit -- see ``gr00t_msgpack.py``'s docstring for why "verified against
source" isn't enough on its own; this project's ``main`` branch had already
diverged by the time it was researched):

- Raw ZeroMQ, ``zmq.REQ`` client / ``zmq.REP`` server, ``tcp://{host}:{port}``,
  default port 5555. REQ/REP is strictly synchronous -- one request in
  flight per socket, and after a timeout the socket is in an unusable state
  (must be recreated before the next send, exactly like the real client's
  ``ping()``/``call_endpoint()`` do).
- Request: ``{"endpoint": name, "data": {...} (if the endpoint takes input),
  "api_token": ... (if set)}``. Response: whatever the handler returned, or
  ``{"error": ...}``; a literal ``b"ERROR"`` byte string is also a hard
  error signal (not a msgpack frame at all).
- Endpoints used here: ``ping`` (health, no input), ``get_action`` (act;
  input ``{"observation": ..., "options": ...}``, returns a 2-list
  ``[action, info]``), ``reset`` (input ``{"options": ...}``),
  ``get_modality_config`` (describe, no input).
- Wire serialization is ``gr00t_msgpack.py`` -- a faithful port of this
  server's *actual* ``MsgSerializer`` (npy-embedded-in-msgpack for arrays),
  not the generic ``msgpack-numpy`` package this module used before a live
  round-trip proved it wire-incompatible (arrays arrived server-side as
  undecoded dicts, not a transport error).

**Deliberate scope-down from the real client**: GR00T's own
``MsgSerializer`` decodes a ``ModalityConfig`` marker into a real
``gr00t.data.types.ModalityConfig`` dataclass -- reconstructing that
would require depending on the full (heavy, GPU-oriented) ``gr00t``
package, not a small client library, defeating the point of a lightweight
adapter. ``gr00t_msgpack.py`` unwraps the same wire marker into a plain
dict instead of a live object -- enough to inspect/log/persist the
checkpoint's declared modality shapes, not enough to be a drop-in for code
written against the real ``PolicyClient``.

Scoped to a single env per client connection (see
``docs/design/real_policy_adapters.md``, "Single-env scoping for real
adapters").
"""

from __future__ import annotations

import time
from typing import Any

import zmq

from embodied_control.logging.logger import EcLogger
from embodied_control.transport import gr00t_msgpack
from embodied_control.transport.client import PolicyClientError
from embodied_control.transport.gr00t_translate import (
    Gr00tObservationMapping,
    gr00t_action_to_chunk,
    observation_to_gr00t,
)


class Gr00tZmqClient:
    def __init__(
        self,
        host: str,
        port: int,
        action_dim: int,
        mapping: Gr00tObservationMapping | None = None,
        api_token: str | None = None,
        timeout_s: float = 30.0,
        logger: EcLogger | None = None,
    ):
        self.host = host
        self.port = int(port)
        self.action_dim = action_dim
        self.mapping = mapping or Gr00tObservationMapping()
        self.api_token = api_token
        self.timeout_s = timeout_s
        self.logger = logger or EcLogger.null()
        self._addr = f"tcp://{host}:{self.port}"
        self._context: zmq.Context | None = None
        self._socket = None

    # --- low-level -------------------------------------------------------
    def _init_socket(self, timeout_s: float) -> None:
        if self._context is None:
            self._context = zmq.Context()
        if self._socket is not None:
            self._socket.close(linger=0)
        self._socket = self._context.socket(zmq.REQ)
        timeout_ms = max(1, int(timeout_s * 1000))
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(self._addr)

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True,
              timeout_s: float | None = None) -> Any:
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        if self._socket is None:
            self._init_socket(timeout_s)
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data or {}
        if self.api_token:
            request["api_token"] = self.api_token

        try:
            self._socket.send(gr00t_msgpack.to_bytes(request))
            message = self._socket.recv()
        except zmq.error.Again as exc:
            # REQ socket is now in an invalid state (waiting for a reply that
            # will never arrive) -- must be recreated before the next call,
            # exactly like the real client's ping()/call_endpoint() do.
            # `linger=0` matters: a REQ socket with an unflushed outbound
            # message and the default LINGER=-1 can hang the whole process at
            # interpreter/context teardown waiting to deliver it.
            self._socket.close(linger=0)
            self._socket = None
            raise PolicyClientError(
                f"gr00t policy service at {self._addr} timed out on {endpoint!r}: {exc}"
            ) from exc
        except zmq.error.ZMQError as exc:
            self._socket.close(linger=0)
            self._socket = None
            raise PolicyClientError(f"cannot reach gr00t policy service at {self._addr}: {exc}") from exc

        if message == b"ERROR":
            raise PolicyClientError(
                f"gr00t policy server returned a raw error on {endpoint!r} "
                "(malformed request or wrong policy loaded)"
            )
        response = gr00t_msgpack.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise PolicyClientError(f"gr00t policy server error on {endpoint!r}: {response['error']}")
        return response

    # --- operations ------------------------------------------------------
    def health(self) -> dict:
        # A short-lived, throwaway socket -- keeps a failed health probe from
        # blocking wait_healthy's polling loop for the full operational
        # timeout_s (ZMQ REQ doesn't fail fast like a TCP connect refusal).
        probe_timeout = min(2.0, self.timeout_s)
        saved_socket, self._socket = self._socket, None
        try:
            self._call("ping", requires_input=False, timeout_s=probe_timeout)
        finally:
            if self._socket is not None:
                self._socket.close(linger=0)
            self._socket = saved_socket
        return {"status": "ok", "model_loaded": True}

    def wait_healthy(self, timeout_s: float = 30.0, interval_s: float = 0.2) -> dict:
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                h = self.health()
                if h.get("status") == "ok":
                    self.logger.debug("gr00t.health_check_ok", endpoint=self._addr)
                    return h
            except PolicyClientError as exc:
                last_err = exc
                self.logger.debug("gr00t.health_check_retry", endpoint=self._addr, error=str(exc))
            time.sleep(interval_s)
        self.logger.warning("gr00t.health_check_timeout", endpoint=self._addr, timeout_s=timeout_s)
        raise PolicyClientError(
            f"gr00t policy service at {self._addr} not healthy within {timeout_s}s"
            + (f" (last error: {last_err})" if last_err else "")
        )

    def describe(self) -> dict:
        modality_config = self._call("get_modality_config", requires_input=False)
        return {
            "action_dim": self.action_dim,
            "policy_id": "gr00t-remote",
            "modality_config": modality_config,
        }

    def reset(self, episode_keys, seed: int) -> dict:
        if len(episode_keys) != 1:
            raise PolicyClientError(
                f"Gr00tZmqClient supports exactly one episode key per call, got {len(episode_keys)}"
            )
        self._call("reset", {"options": None})
        self.logger.debug("gr00t.reset", episode_key=episode_keys[0], seed=seed)
        return {"status": "ok"}

    def act(self, request_id, episode_keys, observations, requested_horizon) -> dict:
        if len(episode_keys) != 1 or len(observations) != 1:
            raise PolicyClientError(
                "Gr00tZmqClient supports exactly one (episode_key, observation) pair per call, "
                f"got {len(episode_keys)} keys / {len(observations)} observations"
            )
        obs_payload = observation_to_gr00t(observations[0], self.mapping)

        start = time.monotonic()
        response = self._call("get_action", {"observation": obs_payload, "options": None})
        total_ms = (time.monotonic() - start) * 1000.0

        action, _info = response  # (action, info) 2-list over the wire
        chunk = gr00t_action_to_chunk(action, self.mapping)
        # requested_horizon is advisory only for a real checkpoint -- same as
        # the OpenPI adapter (see docs/design/real_policy_adapters.md).
        return {
            "status": "ok",
            "actions": [{"action_chunk": chunk}],
            "timing": {"total_ms": total_ms},
        }

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        if self._context is not None:
            self._context.term()
            self._context = None
