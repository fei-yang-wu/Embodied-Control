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
