import pytest
from pydantic import ValidationError

from embodied_control.config.schemas import EvalJob, PolicyBinding, RolloutSpec, SimSpec


def test_replay_dimension_and_timing_fail_before_run_creation():
    with pytest.raises(ValidationError):
        SimSpec(
            backend="twist2",
            mode="delegated",
            action_dim=53,
            backend_config={"replay_episode": "/episode/episode.json"},
        )
    sim = SimSpec(
        backend="twist2",
        mode="delegated",
        action_dim=47,
        backend_config={"replay_episode": "/episode/episode.json"},
    )
    with pytest.raises(ValidationError):
        EvalJob(
            name="wrong_horizon",
            sim=sim,
            policy=PolicyBinding(requested_action_horizon=15),
            rollout=RolloutSpec(video_fps=60),
        )
    with pytest.raises(ValidationError):
        EvalJob(
            name="wrong_fps",
            sim=sim,
            policy=PolicyBinding(requested_action_horizon=40),
            rollout=RolloutSpec(video_fps=30),
        )
    EvalJob(
        name="replay",
        sim=sim,
        policy=PolicyBinding(requested_action_horizon=40),
        rollout=RolloutSpec(video_fps=60),
    )


def test_delegated_latencies_reach_run_metrics():
    from embodied_control.config.schemas import EpisodePolicyStats, EpisodeRecord
    from embodied_control.metrics.aggregate import aggregate_run_metrics

    episode = EpisodeRecord(run_id="replay", episode_id=0, seed=42, task_id="replay",
                            status="completed", success=True, episode_length_steps=80,
                            total_return=0,
                            policy=EpisodePolicyStats(num_requests=2, latencies_ms=[100, 200]))
    metrics = aggregate_run_metrics("replay", 1, [episode], [])
    assert metrics.policy_latency_ms.count == 2
    assert metrics.policy_latency_ms.p50 == 150
    assert metrics.policy_latency_ms.p95 == 195
    assert aggregate_run_metrics("replay", 1, [episode], [50]).policy_latency_ms.count == 1
