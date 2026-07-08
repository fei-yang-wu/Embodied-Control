"""Host port allocation for policy service endpoints."""

from __future__ import annotations

import socket


def allocate_free_port(host: str = "127.0.0.1") -> int:
    """Bind an ephemeral port, release it, and return the number.

    There is a small TOCTOU window before the service binds it; acceptable for
    single-user local/dev runs. A shared-host lease scheme is future work.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return s.getsockname()[1]
