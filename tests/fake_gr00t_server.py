"""A minimal server speaking GR00T's *real* wire protocol (ZeroMQ REQ/REP +
msgpack), for testing ``Gr00tZmqClient`` without a real checkpoint or GPU.
Mirrors ``gr00t/policy/server_client.py::PolicyServer`` closely enough to be
a faithful transport-layer stand-in -- see ``transport/gr00t_client.py``'s
docstring for the exact behaviors this reproduces.
"""

from __future__ import annotations

import threading

import msgpack
import msgpack_numpy as mnp
import zmq


def _to_bytes(data):
    return msgpack.packb(data, default=mnp.encode)


def _from_bytes(data):
    return msgpack.unpackb(data, object_hook=mnp.decode, raw=False)


class FakeGr00tServer:
    """``handlers`` maps endpoint name -> ``(requires_input, fn)``. ``fn`` for
    a ``requires_input`` endpoint takes the ``data`` dict; otherwise takes no
    args. Always includes ``ping``. A handler that raises causes the server
    to send back ``{"error": str(exc)}``, exactly like the real server."""

    def __init__(self, handlers: dict, host="127.0.0.1", port=0):
        self._handlers = {"ping": (False, lambda: {"status": "ok", "message": "Server is running"})}
        self._handlers.update(handlers)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        if port == 0:
            self.port = self._socket.bind_to_random_port(f"tcp://{host}")
        else:
            self._socket.bind(f"tcp://{host}:{port}")
            self.port = port
        self._running = False
        self._thread = threading.Thread(target=self._serve_forever, daemon=True)

    def start_background(self) -> None:
        self._running = True
        self._thread.start()

    def _serve_forever(self) -> None:
        self._socket.setsockopt(zmq.RCVTIMEO, 200)
        while self._running:
            try:
                message = self._socket.recv()
            except zmq.error.Again:
                continue
            try:
                request = _from_bytes(message)
                endpoint = request.get("endpoint", "get_action")
                requires_input, fn = self._handlers[endpoint]
                result = fn(**request.get("data", {})) if requires_input else fn()
                self._socket.send(_to_bytes(result))
            except Exception as exc:  # noqa: BLE001 - deliberately mirrors the real server
                self._socket.send(_to_bytes({"error": str(exc)}))

    def shutdown(self) -> None:
        self._running = False
        self._thread.join(timeout=5)
        self._socket.close(linger=0)
        self._context.term()
