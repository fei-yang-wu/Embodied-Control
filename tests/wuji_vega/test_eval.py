"""End-to-end Embodied Control artifact-contract test."""

import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from embodied_control.artifacts.validation import validate_run_dir  # noqa: E402
from embodied_control.config.schemas import (  # noqa: E402
    EmbodimentBinding,
    EvalJob,
    OutputSpec,
    PolicyBinding,
    RolloutSpec,
    RuntimeSpec,
    SimSpec,
)
from embodied_control.orchestration.runner import run_eval  # noqa: E402
from embodied_control.sim.wuji_vega import ROBOT_ASSET, WUJI_HAND_ASSET  # noqa: E402


def test_eval_produces_valid_wuji_grasp_artifacts(tmp_path, scene_path):
    command = [
        sys.executable,
        "-m",
        "embodied_control.policies.wuji_vega.server",
        "--model",
        str(scene_path),
        "--control-dt",
        "0.02",
        "--host",
        "{host}",
        "--port",
        "{port}",
    ]
    job = EvalJob(
        name="test_wuji_vega_grasp",
        seed=0,
        sim=SimSpec(
            backend="wuji_vega_grasp",
            mode="stepped",
            model_path=str(scene_path),
            backend_config={"frame_skip": 10},
        ),
        policy=PolicyBinding(
            type="wuji_grasp_oracle",
            runtime=RuntimeSpec(type="local", command=command),
            requested_action_horizon=10,
        ),
        embodiment=EmbodimentBinding(
            observation_schema_id="ec.obs.wuji_vega_grasp_proprio/v1",
            action_schema_id="ec.action.wuji_vega_joint_position_normalized/v1",
        ),
        rollout=RolloutSpec(
            num_episodes=1,
            max_steps_per_episode=650,
            record_raw=True,
        ),
        outputs=OutputSpec(root_dir=str(tmp_path / "runs")),
    )
    result = run_eval(job)
    assert result.status == "succeeded"
    assert result.metrics.success_rate == 1.0
    assert result.metrics.num_episodes_completed == 1
    assert result.metrics.num_policy_requests < result.metrics.mean_episode_length_steps

    run_dir = Path(result.run_dir)
    report = validate_run_dir(run_dir)
    assert report.valid, report.errors
    episode = json.loads((run_dir / "episodes.jsonl").read_text())
    assert episode["task_id"] == "wuji_vega_pick_cube"
    assert episode["metrics"]["sustained_cube_lift"] > 0.04
    assert episode["metrics"]["right_hand_contact_steps"] > 0
    raw_path = run_dir / episode["artifacts"]["raw_record"]
    assert raw_path.is_file()
    raw = json.loads(raw_path.read_text())
    assert raw["robot_asset"] == ROBOT_ASSET
    assert raw["wuji_hand_asset"] == WUJI_HAND_ASSET
