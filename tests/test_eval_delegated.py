"""End-to-end delegated-mode eval via the local runtime (default env: no
mujoco/docker needed -- this is the point of the delegated/stepped split)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from embodied_control.artifacts.validation import validate_run_dir
from embodied_control.config.loader import load_job
from embodied_control.config.schemas import (
    EvalJob,
    OutputSpec,
    PolicyBinding,
    RolloutSpec,
    RuntimeSpec,
    SimSpec,
)
from embodied_control.orchestration.runner import run_eval

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def _job(tmp_path: Path, num_ep=3, steps=20, seed=5, action_dim=3) -> EvalJob:
    return EvalJob(
        name="test_delegated",
        seed=seed,
        sim=SimSpec(
            backend="fake_delegated", mode="delegated", action_dim=action_dim,
            runtime=RuntimeSpec(type="local"), timeout_s=30,
            backend_config={"task_id": "unit_test_task"},
        ),
        policy=PolicyBinding(type="random", runtime=RuntimeSpec(type="local"),
                             requested_action_horizon=3),
        rollout=RolloutSpec(num_episodes=num_ep, max_steps_per_episode=steps),
        outputs=OutputSpec(root_dir=str(tmp_path / "runs")),
    )


def test_delegated_local_eval_produces_valid_normalized_artifacts(tmp_path):
    result = run_eval(_job(tmp_path))
    assert result.status == "succeeded"
    assert result.metrics.num_episodes_completed == 3
    assert result.metrics.num_episodes_failed == 0
    assert result.metrics.num_policy_requests > 0

    run_dir = Path(result.run_dir)
    for f in ["job.yaml", "resolved_job.yaml", "manifest.json", "status.json",
              "metrics.json", "episodes.jsonl", "validation.json",
              "generated/sim_config.json", "logs/sim.log",
              "raw/episode_0000.json", "raw/episode_0001.json", "raw/episode_0002.json"]:
        assert (run_dir / f).is_file(), f"missing {f}"

    report = validate_run_dir(run_dir)
    assert report.valid, report.errors

    episodes = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 3
    assert [e["episode_id"] for e in episodes] == [0, 1, 2]
    assert [e["seed"] for e in episodes] == [5, 6, 7]  # per-episode seeds
    assert all(e["task_id"] == "unit_test_task" for e in episodes)
    assert all(e["artifacts"]["raw_record"] == f"raw/episode_{i:04d}.json" for i, e in enumerate(episodes))


def test_delegated_mode_requires_action_dim():
    # A schema-level error (like a malformed job.yaml), not a runtime failure --
    # raised at construction, before any plan/run_dir could even be created.
    with pytest.raises(ValidationError):
        SimSpec(backend="fake_delegated", mode="delegated", action_dim=None)


def test_delegated_mode_ignores_record_video_without_crashing(tmp_path):
    job = _job(tmp_path, num_ep=1, steps=5)
    job.rollout.record_video = True
    result = run_eval(job)
    assert result.status == "succeeded"  # best-effort: warns, doesn't fail


def test_example_delegated_jobs_load_and_validate():
    for name in ["fake_delegated_local.yaml", "fake_delegated_docker.yaml"]:
        job = load_job(EXAMPLES / name)
        assert job.sim.mode == "delegated"
        assert job.sim.backend == "fake_delegated"
        assert job.sim.action_dim is not None


def test_libero_example_job_loads_and_validates():
    # Schema-only: real LIBERO/robosuite/docker execution is covered by manual
    # smoke testing (`pixi run smoke-libero`), not automated pytest -- same
    # category as the other docker-runtime examples (see README "Tests").
    job = load_job(EXAMPLES / "libero_docker.yaml")
    assert job.sim.mode == "delegated"
    assert job.sim.backend == "libero"
    assert job.sim.action_dim == 7  # OSC_POSE: dx,dy,dz,droll,dpitch,dyaw,gripper
    assert job.sim.runtime.type == "docker"
    assert job.sim.backend_config["suite"] == "libero_spatial"
    assert isinstance(job.sim.backend_config["task_index"], int)


def test_real_policy_adapter_example_jobs_load_and_validate():
    # Schema-only, like the LIBERO example above: these point at an external
    # server that isn't actually running in CI (see docs/design/
    # real_policy_adapters.md M3), so real execution isn't automated here --
    # this just guards the YAML/schema contract itself from silently drifting.
    for name, scheme in [
        ("fake_delegated_openpi_external.yaml", "openpi_websocket"),
        ("fake_delegated_gr00t_external.yaml", "gr00t_zmq"),
    ]:
        job = load_job(EXAMPLES / name)
        assert job.policy.endpoint.scheme == scheme
        assert job.policy.endpoint.action_dim == job.sim.action_dim
        assert job.policy.endpoint.observation_mapping.get("proprio_key")


def test_libero_openpi_example_job_loads_and_matches_verified_checkpoint_contract():
    # Schema-only (see above) -- but the mapping values themselves are
    # verified against OpenPI's real src/openpi/policies/libero_policy.py,
    # not placeholders, per docs/design/real_policy_adapters.md M3. This
    # guards those specific verified values from silently drifting too.
    job = load_job(EXAMPLES / "libero_openpi_external.yaml")
    assert job.sim.backend == "libero"
    assert job.sim.action_dim == 7
    assert job.sim.backend_config["camera_height"] == 224
    assert job.sim.backend_config["camera_width"] == 224
    ep = job.policy.endpoint
    assert ep.scheme == "openpi_websocket"
    assert ep.action_dim == 7
    assert ep.observation_mapping["proprio_key"] == "observation/state"
    assert ep.observation_mapping["camera_keys"] == {
        "agentview_image": "observation/image",
        "robot0_eye_in_hand_image": "observation/wrist_image",
    }


def test_runtime_command_override_launches_and_produces_valid_run(tmp_path):
    """End-to-end proof that RuntimeSpec.command actually gets launched (not
    just resolved correctly in isolation, per test_config_and_planner.py) --
    stands in for a real policy server launch command using the debug server
    module itself as an arbitrary target, since the point being tested is
    'the supervisor launches exactly this command', not which server it is."""
    import sys

    job = _job(tmp_path, num_ep=1, steps=5)
    job.policy.runtime.command = [
        sys.executable, "-m", "embodied_control.policies.debug_server",
        "--type", "random", "--action-dim", "3", "--host", "{host}", "--port", "{port}",
    ]
    result = run_eval(job)
    assert result.status == "succeeded", result
    assert result.metrics.num_episodes_completed == 1
