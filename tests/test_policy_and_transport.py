"""Policy behaviour + transport round-trip (stdlib only; runs in the default env)."""

from __future__ import annotations

import signal
import subprocess
import sys
import time

from embodied_control.policies.base import make_policy
from embodied_control.transport.client import PolicyClient
from embodied_control.transport.protocol import episode_key
from embodied_control.transport.server import PolicyServer


def test_zero_policy_action_shape_and_reset_contract():
    p = make_policy("zero", action_dim=3)
    # act before reset -> FAILED_PRECONDITION (episode lifecycle contract)
    resp = p.act({"observations": [{"env_id": 0, "episode_id": 0}], "requested_horizon": 2})
    assert resp["status"] == "FAILED_PRECONDITION"

    p.reset({"episode_keys": [episode_key(0, 0)], "seed": 1})
    resp = p.act({"observations": [{"env_id": 0, "episode_id": 0}], "requested_horizon": 4})
    assert resp["status"] == "ok"
    chunk = resp["actions"][0]["action_chunk"]
    assert len(chunk) == 4
    assert all(a == [0.0, 0.0, 0.0] for a in chunk)


def test_random_policy_holds_command_across_chunk_and_is_seeded():
    p = make_policy("random", action_dim=2, seed=123)
    p.reset({"episode_keys": [episode_key(0, 0)], "seed": 123})
    resp = p.act({"observations": [{"env_id": 0, "episode_id": 0}], "requested_horizon": 5})
    chunk = resp["actions"][0]["action_chunk"]
    # one coherent command held across the chunk
    assert all(a == chunk[0] for a in chunk)
    assert all(-1.0 <= v <= 1.0 for v in chunk[0])

    # deterministic given the same seed
    p2 = make_policy("random", action_dim=2, seed=123)
    p2.reset({"episode_keys": [episode_key(0, 0)], "seed": 123})
    resp2 = p2.act({"observations": [{"env_id": 0, "episode_id": 0}], "requested_horizon": 5})
    assert resp2["actions"][0]["action_chunk"][0] == chunk[0]


def test_horizon_capped_at_max():
    p = make_policy("zero", action_dim=1, max_action_horizon=8)
    p.reset({"episode_keys": [episode_key(0, 0)], "seed": 0})
    resp = p.act({"observations": [{"env_id": 0, "episode_id": 0}], "requested_horizon": 100})
    assert len(resp["actions"][0]["action_chunk"]) == 8


def test_transport_round_trip_over_http():
    server = PolicyServer(make_policy("zero", action_dim=2), host="127.0.0.1", port=0)
    server.start_background()
    try:
        client = PolicyClient("127.0.0.1", server.port, timeout_s=5.0)
        health = client.wait_healthy(timeout_s=5.0)
        assert health["status"] == "ok"
        desc = client.describe()
        assert desc["action_dim"] == 2
        client.reset([episode_key(0, 0)], seed=7)
        resp = client.act("req-1", [episode_key(0, 0)],
                          [{"env_id": 0, "episode_id": 0}], requested_horizon=3)
        assert resp["status"] == "ok"
        assert len(resp["actions"][0]["action_chunk"]) == 3
        assert "total_ms" in resp["timing"]
    finally:
        server.shutdown()


def test_client_unreachable_raises_policy_client_error():
    from embodied_control.transport.client import PolicyClientError

    client = PolicyClient("127.0.0.1", 1, timeout_s=0.5)  # nothing listens on port 1
    start = time.time()
    try:
        client.wait_healthy(timeout_s=1.0)
        assert False, "expected failure"
    except PolicyClientError:
        pass
    assert time.time() - start < 5.0


def test_debug_server_process_exits_promptly_on_sigterm():
    """Regression guard: server.shutdown() called from the same thread as
    serve_forever() deadlocks (stdlib BaseServer.shutdown docs) until something
    force-kills it. debug_server.main() must run serve_forever() in a
    background thread and shut down from the (signal-handling) main thread so
    SIGTERM produces a fast, clean (returncode 0) exit -- not a ~10s hang."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "embodied_control.policies.debug_server",
         "--type", "zero", "--action-dim", "2", "--port", "0"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        ready_line = proc.stdout.readline()
        assert "policy_service_ready" in ready_line

        start = time.time()
        proc.send_signal(signal.SIGTERM)
        returncode = proc.wait(timeout=5.0)  # would previously hang ~10s then need SIGKILL
        elapsed = time.time() - start

        assert elapsed < 2.0, f"SIGTERM shutdown took {elapsed:.2f}s (expected a fast, clean exit)"
        assert returncode == 0
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)
