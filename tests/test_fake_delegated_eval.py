"""Fake delegated evaluator logic, exercised against a real in-process policy
service (stdlib only; default env, no mujoco/docker needed)."""

from __future__ import annotations

import json

from embodied_control.policies.base import make_policy
from embodied_control.sim.fake_delegated_eval import main, run_episode
from embodied_control.transport.client import PolicyClient
from embodied_control.transport.server import PolicyServer


def _live_client(policy_type="random", action_dim=4):
    server = PolicyServer(make_policy(policy_type, action_dim=action_dim), host="127.0.0.1", port=0)
    server.start_background()
    client = PolicyClient("127.0.0.1", server.port, timeout_s=5.0)
    client.wait_healthy(timeout_s=5.0)
    return server, client


def test_run_episode_produces_a_complete_raw_record():
    server, client = _live_client()
    try:
        config = {
            "max_steps_per_episode": 20,
            "requested_action_horizon": 4,
            "task_id": "unit_test_task",
        }
        record = run_episode(client, episode_id=0, seed=42, config=config)
    finally:
        server.shutdown()

    assert record["episode_id"] == 0
    assert record["seed"] == 42
    assert record["task_id"] == "unit_test_task"
    assert record["status"] == "completed"
    assert record["steps"] == 20
    assert isinstance(record["success"], bool)
    # chunk of 4 over 20 steps => 5 requests
    assert record["num_requests"] == 5
    assert record["mean_action_horizon"] == 4.0


def test_run_episode_success_is_seeded_and_reproducible():
    server, client = _live_client()
    try:
        config = {"max_steps_per_episode": 5, "requested_action_horizon": 1}
        r1 = run_episode(client, episode_id=0, seed=999, config=config)
        r2 = run_episode(client, episode_id=1, seed=999, config=config)
    finally:
        server.shutdown()
    assert r1["success"] == r2["success"]


def test_main_writes_one_json_file_per_episode(tmp_path):
    server, _ = _live_client(action_dim=2)
    try:
        config_path = tmp_path / "config.json"
        output_dir = tmp_path / "raw"
        config_path.write_text(json.dumps({
            "policy_host": "127.0.0.1",
            "policy_port": server.port,
            "seeds": [1, 2, 3],
            "max_steps_per_episode": 10,
            "action_dim": 2,
            "requested_action_horizon": 2,
            "task_id": "t",
        }))
        rc = main(["--config", str(config_path), "--output", str(output_dir)])
    finally:
        server.shutdown()

    assert rc == 0
    files = sorted(output_dir.glob("episode_*.json"))
    assert [f.name for f in files] == ["episode_0000.json", "episode_0001.json", "episode_0002.json"]


def test_main_returns_nonzero_when_policy_unreachable(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "policy_host": "127.0.0.1",
        "policy_port": 1,  # nothing listens here
        "health_timeout_s": 1.0,  # keep the test fast
        "seeds": [1],
        "max_steps_per_episode": 5,
        "action_dim": 2,
    }))
    rc = main(["--config", str(config_path), "--output", str(tmp_path / "raw")])
    assert rc != 0
