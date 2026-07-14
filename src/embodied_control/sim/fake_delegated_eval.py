"""Fake delegated evaluator: proves the two-container orchestration shape.

Entry point for a **delegated** sim runtime (``python -m
embodied_control.sim.fake_delegated_eval``). Unlike the stepped MuJoCo backend
(host-driven reset/step), a delegated evaluator owns its own rollout loop
end-to-end -- it is launched as a separate process/container, reads a
host-generated config describing how to reach the policy service, drives
episodes by calling that policy service directly (over the exact same
transport a stepped rollout uses), and writes one raw JSON record per episode
for the host to normalize. This mirrors the shape a real black-box evaluator
(e.g. LIBERO) needs (design D2/6.5) without any simulator dependency: there is
no physics here, just a policy-facing rollout loop and a fake, seeded success
signal so the pipeline has real (if synthetic) variance to normalize.

Stdlib + this package only (no pydantic/mujoco), so the container image needs
no pip install -- see containers/fake_delegated_eval/Dockerfile.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from embodied_control.transport.client import PolicyClient, PolicyClientError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fake delegated evaluator")
    p.add_argument("--config", required=True, help="path to the host-generated sim config JSON")
    p.add_argument("--output", required=True, help="directory to write per-episode raw JSON records")
    # Accepted for CLI-shape parity with other delegated evaluators (the host
    # always passes this) but unused: there is no simulator here to render.
    p.add_argument("--videos-dir", default=None, help="unused (no simulator to render)")
    return p


def _fake_observation(rng: random.Random, env_id: int, episode_id: int, task_id: str) -> dict:
    return {
        "env_id": env_id,
        "episode_id": episode_id,
        "proprio": {
            "names": ["fake_x", "fake_y"],
            "values": [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)],
        },
        "task": {"task_id": task_id, "language_instruction": "fake delegated task"},
    }


def run_episode(client: PolicyClient, episode_id: int, seed: int, config: dict) -> dict:
    key = [0, episode_id]
    client.reset([key], seed)
    rng = random.Random(seed)

    task_id = config.get("task_id", "fake_task")
    horizon = max(1, int(config.get("requested_action_horizon", 1)))
    max_steps = int(config["max_steps_per_episode"])

    buffer: list[list[float]] = []
    num_requests = 0
    horizons: list[int] = []
    total_return = 0.0
    steps = 0

    while steps < max_steps:
        if not buffer:
            obs = _fake_observation(rng, env_id=0, episode_id=episode_id, task_id=task_id)
            resp = client.act(f"{episode_id}:{steps}", [key], [obs], horizon)
            num_requests += 1
            chunk = resp["actions"][0]["action_chunk"]
            horizons.append(len(chunk))
            buffer = list(chunk)
        action = buffer.pop(0)
        # No physics to simulate -- a small illustrative reward so total_return
        # is non-trivial (smaller actions score slightly better).
        total_return -= sum(abs(a) for a in action) * 0.01
        steps += 1

    # Fake but seeded (reproducible) success signal, giving real pass/fail
    # variance for the metrics pipeline to aggregate -- same role the MuJoCo
    # reacher's distance-to-target threshold plays for the stepped path.
    success = rng.random() < 0.6

    return {
        "episode_id": episode_id,
        "seed": seed,
        "task_id": task_id,
        "status": "completed",
        "success": success,
        "steps": steps,
        "total_return": round(total_return, 6),
        "num_requests": num_requests,
        "mean_action_horizon": round(sum(horizons) / len(horizons), 4) if horizons else 0.0,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    health_timeout_s = float(config.get("health_timeout_s", 30.0))
    client = PolicyClient(
        config["policy_host"], config["policy_port"], timeout_s=30.0
    )
    try:
        client.wait_healthy(timeout_s=health_timeout_s)
    except PolicyClientError as exc:
        print(f"fake_delegated_eval: policy service unreachable: {exc}", flush=True)
        return 1

    for episode_id, seed in enumerate(config["seeds"]):
        record = run_episode(client, episode_id, seed, config)
        (output_dir / f"episode_{episode_id:04d}.json").write_text(json.dumps(record, indent=2))
        print(
            f"episode {episode_id} done: success={record['success']} steps={record['steps']}",
            flush=True,
        )

    print("fake_delegated_eval_complete", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
