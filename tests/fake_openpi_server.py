"""A minimal server speaking OpenPI's *real* wire protocol (its own
msgpack-numpy envelope over websocket, plus the /healthz HTTP intercept),
for testing ``OpenPIWebsocketClient`` without a real checkpoint or GPU.
Mirrors ``src/openpi/serving/websocket_policy_server.py`` closely enough to
be a faithful transport-layer stand-in -- see ``transport/openpi_client.py``'s
docstring for the exact behaviors this reproduces.

Uses ``transport.openpi_msgpack`` (OpenPI's own envelope, not the generic
``msgpack-numpy`` PyPI package) -- using the generic package here would have
made this fixture self-consistent with a client that had the same bug,
silently hiding the exact wire-incompatibility a live server round-trip
caught (see ``openpi_msgpack.py``'s docstring for the full story).
"""

from __future__ import annotations

import http
import threading

import websockets.sync.server as _server

from embodied_control.transport import openpi_msgpack


class FakeOpenPIServer:
    """``infer_fn(obs: dict) -> dict`` decides the response; pass one that
    raises to exercise the client's error path (a string response, not
    bytes, exactly like the real server on an internal error)."""

    def __init__(self, infer_fn, metadata: dict | None = None, host="127.0.0.1", port=0):
        self._infer_fn = infer_fn
        self._metadata = metadata or {}
        self._server = _server.serve(
            self._handler, host, port,
            compression=None, max_size=None, process_request=self._health_check,
        )
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def start_background(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._server.shutdown()
        self._thread.join(timeout=5)

    @staticmethod
    def _health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    def _handler(self, websocket) -> None:
        websocket.send(openpi_msgpack.packb(self._metadata))
        while True:
            try:
                obs = openpi_msgpack.unpackb(websocket.recv())
            except _server.ConnectionClosed:
                return
            try:
                action = self._infer_fn(obs)
            except Exception as exc:  # noqa: BLE001 - deliberately mirrors the real server
                websocket.send(f"{type(exc).__name__}: {exc}")
                websocket.close(code=1011, reason="internal error")
                return
            websocket.send(openpi_msgpack.packb(action))
