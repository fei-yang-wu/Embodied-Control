"""Metric aggregation + run-directory validation (default env; no sim/docker)."""

from __future__ import annotations

from embodied_control.artifacts.validation import validate_run_dir
from embodied_control.config.schemas import EpisodePolicyStats, EpisodeRecord
from embodied_control.metrics.aggregate import aggregate_run_metrics, latency_stats


def _ep(episode_id, success, length, ret, status="completed", requests=1):
    return EpisodeRecord(
        run_id="r", episode_id=episode_id, seed=episode_id, task_id="t",
        status=status, success=success, episode_length_steps=length, total_return=ret,
        policy=EpisodePolicyStats(num_requests=requests),
    )


def test_aggregate_success_rate_and_counts():
    eps = [_ep(0, True, 10, -1.0), _ep(1, False, 20, -2.0), _ep(2, True, 30, -3.0),
           _ep(3, False, 0, 0.0, status="failed")]
    m = aggregate_run_metrics("r", num_requested=4, episodes=eps, latencies_ms=[1, 2, 3, 4])
    assert m.num_episodes_completed == 3
    assert m.num_episodes_failed == 1
    assert m.success_rate == round(2 / 3, 6)  # 2 of 3 completed
    assert m.mean_episode_length_steps == 20.0
    assert m.num_policy_requests == 4


def test_latency_percentiles():
    s = latency_stats([float(x) for x in range(1, 101)])
    assert s.count == 100
    assert 49 <= s.p50 <= 51
    assert 94 <= s.p95 <= 96


def test_validation_detects_missing_and_corrupt(tmp_path):
    run = tmp_path / "runs" / "empty_run"
    run.mkdir(parents=True)
    report = validate_run_dir(run)
    assert not report.valid
    assert any("missing required artifact" in e for e in report.errors)

    # corrupt metrics.json
    (run / "metrics.json").write_text("{ not valid json ")
    report2 = validate_run_dir(run)
    assert not report2.valid


def test_validation_passes_on_wellformed_run(tmp_path):
    """Build a minimal well-formed run dir via the store and validate it."""
    from embodied_control.artifacts.store import ArtifactStore
    from embodied_control.config.schemas import (
        EvalJob, ExecutionPlan, HostFingerprint, ResolvedEndpoint, ResolvedRuntime,
        RunManifest, RunMetrics, RunStatus, SimSpec,
    )

    job = EvalJob(name="wf", sim=SimSpec(backend="mujoco"))
    host = HostFingerprint(platform="p", python_version="3.13", hostname="h", user="u")
    plan = ExecutionPlan(
        job=job, run_id="wf_run", run_dir=str(tmp_path / "runs" / "wf_run"),
        created_at="2026-07-07T00:00:00", host=host, seeds=[0], action_dim=2,
        action_schema_id="a", observation_schema_id="o",
        policy_endpoint=ResolvedEndpoint(host="127.0.0.1", port=1234),
        policy_runtime=ResolvedRuntime(name="policy", type="local", host_port=1234),
    )
    store = ArtifactStore(plan.run_dir)
    store.initialize()
    store.write_job(job)
    store.write_plan(plan)
    store.orchestrator_log_path.write_text("log\n")
    store.events_path.write_text("{}\n")
    store.write_episodes([])
    store.write_metrics(RunMetrics(
        run_id="wf_run", num_episodes_requested=0, num_episodes_completed=0,
        num_episodes_failed=0, success_rate=0.0, mean_episode_length_steps=0.0,
        mean_return=0.0, num_policy_requests=0,
    ))
    store.write_manifest(RunManifest(
        run_id="wf_run", job_name="wf", created_at="2026-07-07T00:00:00",
        status="succeeded", seed=0, host=host,
    ))
    store.write_status(RunStatus(
        status="succeeded", phase="done", started_at="2026-07-07T00:00:00",
    ))

    report = validate_run_dir(plan.run_dir)
    assert report.valid, report.errors
