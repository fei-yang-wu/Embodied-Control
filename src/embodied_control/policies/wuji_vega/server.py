"""HTTP policy-service entry point for the Vega-Wuji scripted IK oracle."""

from __future__ import annotations

import argparse
import signal
import sys
import threading

from embodied_control.policies.wuji_vega import WujiGraspOraclePolicy
from embodied_control.transport.server import PolicyServer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vega-Wuji grasp oracle service")
    parser.add_argument("--model", required=True)
    parser.add_argument("--control-dt", type=float, default=0.02)
    parser.add_argument(
        "--action-schema-id",
        default="ec.action.wuji_vega_joint_position_normalized/v1",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-action-horizon", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policy = WujiGraspOraclePolicy(
        model_path=args.model,
        control_dt=args.control_dt,
        action_schema_id=args.action_schema_id,
        seed=args.seed,
        max_action_horizon=args.max_action_horizon,
    )
    server = PolicyServer(policy, host=args.host, port=args.port)
    print(
        f"policy_service_ready host={server.host} port={server.port} "
        f"type={policy.policy_type}",
        flush=True,
    )
    stop_requested = threading.Event()

    def _term(signum, frame):  # noqa: ANN001
        del signum, frame
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
