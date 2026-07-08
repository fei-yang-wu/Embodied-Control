"""Stepped rollout runner: the host-driven eval loop.

Wiring (the separation model):

    host runner  --reset/act HTTP-->  policy service (separate process/container)
         |                                     returns normalized action chunk
         |  embodiment controller.decode_action(normalized) -> ctrl
         v
    MuJoCo stepped env  (host-driven reset/step)

The policy is blank (zero/random); the controller maps its normalized command to
actuation; the env simulates. Artifacts are written for every run, success or
failure.
"""

from __future__ import annotations

from datetime import datetime

from embodied_control.artifacts.store import ArtifactStore
from embodied_control.artifacts.validation import validate_run_dir
from embodied_control.config.schemas import (
    EpisodePolicyStats,
    EpisodeRecord,
    EvalJob,
    EvalResult,
    RunManifest,
    RunStatus,
)
from embodied_control.logging.config import LogConfig
from embodied_control.logging.logger import EcLogger
from embodied_control.logging.timeutil import utcnow_iso
from embodied_control.metrics.aggregate import aggregate_run_metrics
from embodied_control.orchestration import planner as planner_mod
from embodied_control.orchestration import registry
from embodied_control.orchestration.supervisor import PolicyServiceSupervisor
from embodied_control.transport.client import PolicyClientError


class RunFailure(RuntimeError):
    def __init__(self, phase: str, reason: str):
        super().__init__(f"[{phase}] {reason}")
        self.phase = phase
        self.reason = reason


def run_eval(job: EvalJob) -> EvalResult:
    started_at = utcnow_iso()

    # --- build backend (needs the sim env) to learn action dim + ctrlrange ---
    # Built before the plan/store/logger exist (its action_dim/ctrlrange are
    # needed to resolve the plan), so it starts with a null logger.
    backend = registry.get_sim_backend(job.sim.backend)(job)
    action_dim = backend.action_dim
    ctrlrange = backend.action_ctrlrange

    plan = planner_mod.build_plan(
        job,
        action_dim=action_dim,
        action_schema_id=job.embodiment.action_schema_id,
        observation_schema_id=job.embodiment.observation_schema_id,
        now=datetime.now(),
    )

    store = ArtifactStore(plan.run_dir)
    store.initialize()
    store.write_job(job)
    store.write_plan(plan)

    logger = EcLogger.create(
        LogConfig(level=job.outputs.log_level, log_dir=str(store.logs_dir)),
        run_id=plan.run_id,
    )
    backend.logger = logger.child("sim")
    logger.event("run.started", phase="init", run_dir=plan.run_dir,
                 backend=job.sim.backend, policy=job.policy.type,
                 runtime=plan.policy_runtime.type)

    controller = registry.get_embodiment(job.embodiment.adapter)(
        job, ctrlrange, logger=logger.child("embodiment")
    )
    supervisor = PolicyServiceSupervisor(
        job, plan, str(store.policy_log_path), logger=logger.child("policy")
    )

    episodes: list[EpisodeRecord] = []
    latencies_ms: list[float] = []
    phase = "policy_launch"
    status_str = "running"
    reason: str | None = None

    try:
        client = supervisor.start(health_timeout_s=30.0)
        logger.event("runtime.health.ready", phase="policy_launch",
                     endpoint=f"{plan.policy_endpoint.host}:{plan.policy_endpoint.port}")

        phase = "policy_describe"
        desc = client.describe()
        if int(desc.get("action_dim", -1)) != action_dim:
            raise RunFailure(
                "policy_describe",
                f"policy action_dim {desc.get('action_dim')} != env action_dim {action_dim}",
            )
        logger.event("policy.describe", phase="policy_describe",
                     policy_id=desc.get("policy_id"), action_dim=desc.get("action_dim"))

        phase = "rollout"
        rollout_logger = logger.child("rollout")
        for episode_id, seed in enumerate(plan.seeds):
            episodes.append(
                _run_episode(job, plan, backend, controller, client, rollout_logger, episode_id,
                             seed, latencies_ms, store)
            )

        phase = "aggregate"
        status_str = "succeeded"
    except RunFailure as exc:
        status_str, reason, phase = "failed", exc.reason, exc.phase
        logger.event("run.failed", severity="error", phase=phase, reason=reason)
    except Exception as exc:  # noqa: BLE001
        status_str, reason = "failed", f"{type(exc).__name__}: {exc}"
        logger.event("run.failed", severity="error", phase=phase, reason=reason,
                     policy_log_tail=supervisor.logs()[-500:])
    finally:
        supervisor.stop()
        backend.close()

    # --- write artifacts (always, even on failure) -----------------------
    metrics = aggregate_run_metrics(plan.run_id, len(plan.seeds), episodes, latencies_ms)
    store.write_episodes(episodes)
    store.write_metrics(metrics)

    manifest = RunManifest(
        run_id=plan.run_id,
        job_name=job.name,
        created_at=plan.created_at,
        completed_at=utcnow_iso(),
        status=status_str,
        seed=job.seed,
        host=plan.host,
        runtimes=[plan.policy_runtime],
        schemas={
            "artifact": plan.artifact_contract_version,
            "eval": plan.api_version,
            "policy_endpoint_scheme": plan.policy_endpoint.scheme,
            "observation": plan.observation_schema_id,
            "action": plan.action_schema_id,
        },
    )
    store.write_manifest(manifest)

    status = RunStatus(
        status=status_str if status_str != "running" else "failed",
        phase=phase,
        reason=reason,
        started_at=started_at,
        ended_at=utcnow_iso(),
        logs=["logs/orchestrator.log", "logs/events.jsonl", "logs/policy.log"],
    )
    store.write_status(status)

    report = validate_run_dir(plan.run_dir)
    store.write_validation(report)

    result = EvalResult(
        run_id=plan.run_id,
        run_dir=plan.run_dir,
        status=status.status,
        num_episodes=len(episodes),
        metrics=metrics,
    )
    store.write_result(result)
    logger.event("run.completed", phase="done", status=status.status,
                 success_rate=metrics.success_rate,
                 completed=metrics.num_episodes_completed, failed=metrics.num_episodes_failed)
    logger.close()
    return result


def _run_episode(job, plan, backend, controller, client, logger, episode_id, seed,
                 latencies_ms, store) -> EpisodeRecord:
    obs = backend.reset(seed, episode_id)
    key = [0, episode_id]
    client.reset([key], seed)
    logger.event("sim.episode.started", phase="rollout", episode_id=episode_id, seed=seed)

    horizon = max(1, int(job.policy.requested_action_horizon))
    max_steps = int(job.rollout.max_steps_per_episode)
    buffer: list[list[float]] = []
    num_requests = 0
    horizons: list[int] = []
    fallback_steps = 0
    steps = 0
    total_return = 0.0
    done = False
    failed = False

    # Video capture is best-effort: a render failure (e.g. no EGL/OSMesa on a
    # headless node) disables it for the rest of the episode and is logged,
    # but never fails the rollout -- the eval's actual output is the metrics.
    record_video = bool(job.rollout.record_video)
    frames = []
    if record_video:
        record_video = _try_capture_frame(backend, frames, logger)

    while steps < max_steps and not done:
        if not buffer:
            req_id = f"{plan.run_id}:{episode_id}:{steps}"
            try:
                resp = client.act(req_id, [key], [obs.to_wire()], horizon)
            except PolicyClientError as exc:
                # Bounded, recorded fallback: apply the controller's safe action.
                logger.event("policy.request.failed", severity="warning", phase="rollout",
                             episode_id=episode_id, step=steps, error=str(exc))
                ctrl = controller.fallback()
                result = backend.step(ctrl)
                obs, done = result.observation, result.done
                total_return += result.reward
                fallback_steps += 1
                steps += 1
                if record_video:
                    record_video = _try_capture_frame(backend, frames, logger)
                continue
            num_requests += 1
            timing = resp.get("timing", {})
            latencies_ms.append(float(timing.get("total_ms", 0.0)))
            chunk = resp["actions"][0]["action_chunk"]
            horizons.append(len(chunk))
            buffer = list(chunk)

        norm_action = buffer.pop(0)
        ctrl = controller.decode_action(norm_action)
        result = backend.step(ctrl)
        obs, done = result.observation, result.done
        total_return += result.reward
        steps += 1
        if done:  # non-finite state => blow-up
            failed = True
        if record_video:
            record_video = _try_capture_frame(backend, frames, logger)

    summary = backend.episode_summary()
    artifacts: dict[str, str] = {}
    if job.rollout.record_raw:
        raw = dict(summary)
        raw.update({"episode_id": episode_id, "seed": seed, "steps": steps,
                    "num_requests": num_requests, "fallback_steps": fallback_steps})
        artifacts["raw_record"] = store.write_raw_episode(episode_id, raw)

    if frames:
        try:
            video_rel = store.write_video(episode_id, frames, fps=job.rollout.video_fps)
            artifacts["video"] = video_rel
            logger.event("render.video.completed", phase="rollout", episode_id=episode_id,
                         path=video_rel, frames=len(frames))
        except Exception as exc:  # noqa: BLE001 - encoding failure must not fail the episode
            logger.warning("render.video.write_failed", episode_id=episode_id, error=str(exc))

    logger.event("sim.episode.completed", phase="rollout", episode_id=episode_id,
                 success=summary["success"], steps=steps, final_distance=summary["final_distance"])

    return EpisodeRecord(
        run_id=plan.run_id,
        episode_id=episode_id,
        env_id=0,
        seed=seed,
        task_id=backend.task_id(),
        status="failed" if failed else "completed",
        success=bool(summary["success"]) and not failed,
        episode_length_steps=steps,
        total_return=round(total_return, 6),
        metrics={
            "success": float(summary["success"]),
            "final_distance": summary["final_distance"],
            "min_distance": summary["min_distance"],
        },
        policy=EpisodePolicyStats(
            num_requests=num_requests,
            mean_action_horizon=round(sum(horizons) / len(horizons), 4) if horizons else 0.0,
            fallback_steps=fallback_steps,
            timeouts=0,
        ),
        artifacts=artifacts,
    )


def _try_capture_frame(backend, frames: list, logger) -> bool:
    """Append one rendered frame; return False (disabling video) on failure.

    Called at most once per step. A render failure is logged a single time
    (the caller stops calling once this returns False) rather than once per
    remaining step, so a headless-node failure doesn't flood the log.
    """
    try:
        frames.append(backend.render_frame())
        return True
    except Exception as exc:  # noqa: BLE001 - RenderError or any backend-specific GL failure
        logger.warning("render.frame_failed", error=str(exc))
        return False
