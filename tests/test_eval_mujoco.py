"""End-to-end eval tests. Require MuJoCo, so they skip in the light default env
(run with: ``pixi run -e sim test-sim``). They use the local-subprocess runtime
so no Docker is needed."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Must run before the FIRST `import mujoco` anywhere in the process (including
# importorskip's own import below): MuJoCo picks its GL backend at first import
# and reuses that choice thereafter, so setting MUJOCO_GL any later is too late
# for the video test's offscreen renderer. See mujoco_backend._default_headless_gl_backend.
from embodied_control.sim.mujoco_backend import _default_headless_gl_backend  # noqa: E402

_default_headless_gl_backend()
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


def _job(tmp_path: Path, policy_type: str, horizon: int, num_ep: int, steps: int, seed: int,
         thr: float) -> EvalJob:
    return EvalJob(
        name=f"test_{policy_type}",
        seed=seed,
        sim=SimSpec(backend="mujoco", model="reacher2",
                    backend_config={"success_threshold": thr}),
        policy=PolicyBinding(type=policy_type, runtime=RuntimeSpec(type="local"),
                             requested_action_horizon=horizon),
        embodiment=EmbodimentBinding(),
        rollout=RolloutSpec(num_episodes=num_ep, max_steps_per_episode=steps),
        outputs=OutputSpec(root_dir=str(tmp_path / "runs")),
    )


def test_zero_eval_produces_valid_artifacts(tmp_path):
    result = run_eval(_job(tmp_path, "zero", horizon=1, num_ep=2, steps=60, seed=0, thr=0.03))
    assert result.status == "succeeded"
    assert result.metrics.num_episodes_completed == 2
    assert result.metrics.num_episodes_failed == 0
    # zero policy holds home pose -> essentially never reaches a random target
    assert result.metrics.success_rate == 0.0
    # horizon 1 => one policy request per step
    assert result.metrics.num_policy_requests == 2 * 60

    run_dir = Path(result.run_dir)
    for f in ["job.yaml", "resolved_job.yaml", "manifest.json", "status.json",
              "metrics.json", "episodes.jsonl", "validation.json",
              "logs/orchestrator.log", "logs/events.jsonl", "generated/eval_result.json"]:
        assert (run_dir / f).is_file(), f"missing {f}"

    report = validate_run_dir(run_dir)
    assert report.valid, report.errors

    episodes = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes) == 2
    assert episodes[0]["seed"] == 0 and episodes[1]["seed"] == 1  # per-episode seeds


def test_random_eval_reaches_target_and_chunks(tmp_path):
    result = run_eval(_job(tmp_path, "random", horizon=20, num_ep=6, steps=200, seed=3, thr=0.04))
    assert result.status == "succeeded"
    # a random position policy explores the workspace -> non-trivial success
    assert result.metrics.success_rate > 0.3
    # chunk of 20 over 200 steps => 10 requests/episode, far fewer than steps
    assert result.metrics.num_policy_requests == 6 * (200 // 20)


def test_action_chunking_reduces_requests(tmp_path):
    r1 = run_eval(_job(tmp_path, "zero", horizon=1, num_ep=1, steps=100, seed=0, thr=0.03))
    r8 = run_eval(_job(tmp_path, "zero", horizon=10, num_ep=1, steps=100, seed=0, thr=0.03))
    assert r1.metrics.num_policy_requests == 100
    assert r8.metrics.num_policy_requests == 10


def test_video_disabled_by_default_produces_no_video_artifact(tmp_path):
    result = run_eval(_job(tmp_path, "zero", horizon=1, num_ep=1, steps=20, seed=0, thr=0.03))
    run_dir = Path(result.run_dir)
    episodes = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines()]
    assert "video" not in episodes[0]["artifacts"]
    assert list((run_dir / "videos").iterdir()) == []  # dir exists (contract), stays empty


def test_video_recording_produces_playable_artifact(tmp_path):
    job = _job(tmp_path, "random", horizon=15, num_ep=1, steps=60, seed=1, thr=0.05)
    job.rollout.record_video = True
    job.rollout.video_fps = 30
    result = run_eval(job)
    assert result.status == "succeeded"

    run_dir = Path(result.run_dir)
    episodes = [json.loads(l) for l in (run_dir / "episodes.jsonl").read_text().splitlines()]
    video_rel = episodes[0]["artifacts"].get("video")
    if video_rel is None:
        pytest.skip("offscreen rendering unavailable in this environment (no EGL/OSMesa)")

    video_path = run_dir / video_rel
    assert video_path.is_file()
    assert video_path.stat().st_size > 1000  # not an empty/corrupt encode

    events = [json.loads(l) for l in (run_dir / "logs" / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "render.video.completed" for e in events)
