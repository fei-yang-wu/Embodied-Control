"""A tiny stdlib HTTP server hosting a policy (runs inside the policy container).

Imports only the standard library so the container image needs no third-party
Python dependencies. The server is transport-only; the actual behaviour lives in
the ``Policy`` object it is given.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class _Handler(BaseHTTPRequestHandler):
    # Silence default stderr access logging; the launcher captures stdout/stderr.
    def log_message(self, *args):  # noqa: D401,ANN001
        return

    def _send(self, code: int, obj) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else {}

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, self.server.policy.health())
        else:
            self._send(404, {"status": "error", "error": f"unknown path {self.path}"})

    def do_POST(self):  # noqa: N802
        try:
            req = self._read()
            policy = self.server.policy
            if self.path == "/describe":
                self._send(200, policy.describe())
            elif self.path == "/reset":
                self._send(200, policy.reset(req))
            elif self.path == "/act":
                self._send(200, policy.act(req))
            else:
                self._send(404, {"status": "error", "error": f"unknown path {self.path}"})
        except Exception as exc:  # noqa: BLE001 - report errors on the wire, don't crash the server
            self._send(500, {"status": "error", "error": f"{type(exc).__name__}: {exc}"})


class PolicyServer:
    """Owns a ThreadingHTTPServer bound to (host, port) serving ``policy``.

    ``port=0`` binds an ephemeral port; read ``.port`` afterwards.
    """

    def __init__(self, policy, host: str = "127.0.0.1", port: int = 0):
        self._httpd = ThreadingHTTPServer((host, port), _Handler)
        self._httpd.policy = policy
        self.host, self.port = self._httpd.server_address[0], self._httpd.server_address[1]
        self._thread: threading.Thread | None = None

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def start_background(self) -> threading.Thread:
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self._thread

    def shutdown(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
