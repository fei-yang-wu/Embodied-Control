"""Entry point for the blank policy service (``python -m ...debug_server``).

This is what runs inside the policy container (or, for the ``local`` runtime, as
a host subprocess). Stdlib-only. Args are supplied by the host planner/supervisor
which knows the environment's action dimension.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from embodied_control.policies.base import make_policy
from embodied_control.transport.server import PolicyServer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Blank VLA policy debug server")
    p.add_argument("--type", default="zero", choices=["zero", "random"])
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--action-schema-id", default="ec.action.normalized/v1")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-action-horizon", type=int, default=16)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = make_policy(
        args.type,
        action_dim=args.action_dim,
        action_schema_id=args.action_schema_id,
        seed=args.seed,
        max_action_horizon=args.max_action_horizon,
    )
    server = PolicyServer(policy, host=args.host, port=args.port)
    # Print the bound endpoint so a launcher can discover an ephemeral port.
    print(f"policy_service_ready host={server.host} port={server.port} type={args.type}", flush=True)

    # server.shutdown() blocks until serve_forever() observes the shutdown flag
    # and exits (see stdlib BaseServer.shutdown docs) -- it MUST be called from a
    # different thread than the one running serve_forever(), or it deadlocks.
    # So: run serve_forever() in a background thread, and have the (SIGTERM-
    # triggered) main thread call shutdown() from outside it.
    stop_requested = threading.Event()

    def _term(signum, frame):  # noqa: ANN001
        stop_requested.set()

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    server.start_background()
    try:
        stop_requested.wait()
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
