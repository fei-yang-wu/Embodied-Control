"""Config loading, schema validation, embodiment mapping, and planning."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from embodied_control.config.loader import ConfigError, load_job
from embodied_control.config.schemas import EvalJob
from embodied_control.embodiments.passthrough import PassthroughController
from embodied_control.orchestration import planner as planner_mod

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.parametrize("name", [
    "mujoco_zero_local.yaml",
    "mujoco_random_local.yaml",
    "mujoco_zero.yaml",
    "mujoco_random_video.yaml",
    "wuji_vega_grasp_oracle_local.yaml",
])
def test_example_jobs_load_and_validate(name):
    job = load_job(EXAMPLES / name)
    assert isinstance(job, EvalJob)
    assert job.sim.mode == "stepped"


def test_wuji_vega_config_requires_scene_and_explicit_oracle_runtime():
    from pydantic import ValidationError

    from embodied_control.config.schemas import PolicyBinding, SimSpec

    with pytest.raises(ValidationError, match="requires sim.model_path"):
        SimSpec(backend="wuji_vega_grasp")
    with pytest.raises(ValidationError, match="requires sim.mode='stepped'"):
        SimSpec(
            backend="wuji_vega_grasp",
            mode="delegated",
            model_path="scene.xml",
            action_dim=59,
        )
    with pytest.raises(ValidationError, match="requires policy.runtime.command"):
        PolicyBinding(type="wuji_grasp_oracle")
    with pytest.raises(ValidationError, match="frame_skip must be >= 1"):
        SimSpec(
            backend="wuji_vega_grasp",
            model_path="scene.xml",
            backend_config={"frame_skip": 0},
        )


def test_video_example_enables_recording_and_debug_logging():
    job = load_job(EXAMPLES / "mujoco_random_video.yaml")
    assert job.rollout.record_video is True
    assert job.rollout.video_fps == 30
    assert job.outputs.log_level == "DEBUG"


def test_invalid_job_rejected():
    with pytest.raises(ConfigError):
        # missing required 'sim' section
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
            fh.write("api_version: ec.eval/v1alpha1\nname: bad\n")
            path = fh.name
        load_job(path)


def test_passthrough_maps_normalized_to_ctrlrange():
    ctrl = PassthroughController(ctrlrange=[(-2.0, 2.0), (0.0, 10.0)])
    assert ctrl.decode_action([0.0, 0.0]) == [0.0, 5.0]   # midpoints
    assert ctrl.decode_action([1.0, 1.0]) == [2.0, 10.0]  # upper
    assert ctrl.decode_action([-1.0, -1.0]) == [-2.0, 0.0]  # lower
    # clipping
    assert ctrl.decode_action([5.0, -5.0]) == [2.0, 0.0]
    # fallback = midpoints (normalized zero)
    assert ctrl.fallback() == [0.0, 5.0]


def test_planner_resolves_run_id_ports_and_seeds():
    job = load_job(EXAMPLES / "mujoco_zero_local.yaml")
    plan = planner_mod.build_plan(
        job, action_dim=2,
        action_schema_id="ec.action.mujoco_normalized/v1",
        observation_schema_id="ec.obs.mujoco_proprio/v1",
        now=datetime(2026, 7, 7, 21, 0, 0),
    )
    assert plan.run_id == "20260707_210000_mujoco_reacher_zero_local_42"
    assert plan.seeds == [42, 43, 44]  # num_episodes=3
    assert plan.policy_runtime.type == "local"
    assert plan.policy_endpoint.port > 0
    assert plan.action_dim == 2


def test_planner_docker_requires_image():
    job = load_job(EXAMPLES / "mujoco_zero.yaml")
    plan = planner_mod.build_plan(
        job, action_dim=2, action_schema_id="a", observation_schema_id="o",
        now=datetime(2026, 7, 7, 21, 0, 0),
    )
    assert plan.policy_runtime.type == "docker"
    assert plan.policy_runtime.image == "ec-policy-debug:latest"
    assert plan.policy_runtime.host_port and plan.policy_runtime.container_port == 8000


def test_planner_builds_debug_server_command_with_max_action_horizon(tmp_path):
    """Regression guard: the command persisted into resolved_job.yaml must be
    the exact command the supervisor launches (see
    orchestration/planner.py::_resolve_policy_command) -- these two used to
    drift, with the persisted one silently missing --max-action-horizon."""
    job = load_job(EXAMPLES / "mujoco_zero_local.yaml")
    plan = planner_mod.build_plan(
        job, action_dim=2, action_schema_id="a", observation_schema_id="o",
        now=datetime(2026, 7, 7, 21, 0, 0),
    )
    command = plan.policy_runtime.command
    assert "--max-action-horizon" in command
    assert command[0:1] == [__import__("sys").executable]
    assert "embodied_control.policies.debug_server" in command


def test_runtime_command_override_is_used_verbatim_with_host_port_substitution(tmp_path):
    """A job author's explicit runtime.command (needed for a real policy
    server whose CLI shape we can't derive -- see
    docs/design/real_policy_adapters.md) replaces the built-in debug-server
    command entirely, with {host}/{port} substituted."""
    job = load_job(EXAMPLES / "mujoco_zero_local.yaml")
    job.policy.runtime.command = ["fake-server", "--listen", "{host}:{port}", "--checkpoint", "/ckpt"]
    plan = planner_mod.build_plan(
        job, action_dim=2, action_schema_id="a", observation_schema_id="o",
        now=datetime(2026, 7, 7, 21, 0, 0),
    )
    assert plan.policy_runtime.command == [
        "fake-server", "--listen", f"127.0.0.1:{plan.policy_endpoint.port}", "--checkpoint", "/ckpt",
    ]


def test_planner_dedupes_run_id_when_run_dir_already_exists(tmp_path):
    """Two plans built for the same job/seed/timestamp must not collide on the
    same run_dir (which would silently overwrite the first run's artifacts and,
    since the logger's root name is keyed by run_id, its log handlers too)."""
    job = load_job(EXAMPLES / "mujoco_zero_local.yaml")
    job.outputs.root_dir = str(tmp_path)
    same_instant = datetime(2026, 7, 7, 21, 0, 0)

    plan1 = planner_mod.build_plan(
        job, action_dim=2, action_schema_id="a", observation_schema_id="o", now=same_instant
    )
    Path(plan1.run_dir).mkdir(parents=True)  # simulate the first run having started

    plan2 = planner_mod.build_plan(
        job, action_dim=2, action_schema_id="a", observation_schema_id="o", now=same_instant
    )

    assert plan1.run_id != plan2.run_id
    assert plan2.run_id == f"{plan1.run_id}-1"
    assert plan1.run_dir != plan2.run_dir
