"""Delegated rollout: launch a runtime that owns its own rollout loop end-to-end
(unlike the stepped path, the host doesn't call reset()/step() at all), wait
for it to exit, and normalize its raw per-episode output into the standard
``EpisodeRecord`` contract.

This is intentionally the minimal shape that proves two-container
orchestration + delegated normalization work (design D2/6.5): two concrete
evaluators (``fake_delegated``, ``libero``), one fixed raw-JSON-per-episode
output format, no plugin registry for normalizers yet. A meaningfully
different third evaluator is what would justify factoring that out -- not
before.
"""

from __future__ import annotations

import json
import sys

from embodied_control.artifacts.store import ArtifactStore
from embodied_control.config.schemas import (
    EpisodePolicyStats,
    EpisodeRecord,
    EvalJob,
    ExecutionPlan,
    MountSpec,
)
from embodied_control.logging.logger import EcLogger
from embodied_control.orchestration.errors import RunFailure
from embodied_control.runtime.docker import DockerRuntimeAdapter
from embodied_control.runtime.local import LocalRuntimeAdapter

# Local-runtime module per backend. Only backends with no heavy/pinned
# dependencies (stdlib-only, like the host itself) can realistically run
# locally -- e.g. LIBERO's Python-3.8-only pinned deps make runtime.type=local
# a non-starter for it in practice, but we still dispatch on backend rather
# than silently defaulting to the wrong evaluator (see git history: this used
# to be hardcoded to fake_delegated_eval regardless of sim.backend).
_LOCAL_MODULES = {
    "fake_delegated": "embodied_control.sim.fake_delegated_eval",
    "libero": "embodied_control.sim.libero_eval",
}


def _generate_sim_config(job: EvalJob, plan: ExecutionPlan) -> dict:
    """The contract between host and delegated runtime: how to reach the
    policy service, and the rollout parameters the evaluator should run with.
    Written to ``generated/sim_config.json`` (design §10.2: every generated
    config lives under ``runs/<run_id>/generated/``).

    ``sim.backend_config`` is merged in wholesale rather than cherry-picked:
    the host stays generic about backend-specific keys (task/suite selectors,
    camera size, ...) -- a delegated evaluator is a black-box (design §6.5),
    so the host has no business knowing its config shape beyond this contract.
    """
    config = {
        "policy_scheme": plan.policy_endpoint.scheme,
        "policy_host": plan.policy_endpoint.host,
        "policy_port": plan.policy_endpoint.port,
        "seeds": plan.seeds,
        "max_steps_per_episode": job.rollout.max_steps_per_episode,
        "action_dim": plan.action_dim,
        "requested_action_horizon": job.policy.requested_action_horizon,
        "record_video": job.rollout.record_video,
        "video_fps": job.rollout.video_fps,
    }
    config.update(job.sim.backend_config)
    return config


def run_delegated_rollout(
    job: EvalJob, plan: ExecutionPlan, store: ArtifactStore, logger: EcLogger
) -> list[EpisodeRecord]:
    config = _generate_sim_config(job, plan)
    config_path = store.generated_dir / "sim_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    logger.event("sim.delegated.config_written", phase="sim_launch", path=str(config_path))

    rt = job.sim.runtime
    log_path = str(store.logs_dir / "sim.log")

    if rt.type == "docker":
        if not rt.image:
            raise RunFailure("sim_launch", "sim.runtime.type=docker requires sim.runtime.image")
        adapter = DockerRuntimeAdapter(
            name="sim",
            image=rt.image,
            container_name=f"ec_{plan.run_id}_sim",
            command=[
                "--config", "/generated/sim_config.json",
                "--output", "/raw",
                "--videos-dir", "/videos",
            ],
            # Host networking: the policy runtime (local or its own container)
            # publishes on 127.0.0.1:<port> on the actual host; a bridge-network
            # container can't reach that without container-to-container DNS,
            # which Apptainer/HPC don't provide. Host networking is the
            # portable choice for inter-runtime reachability (see the design
            # review's Apptainer-parity finding) -- this sim container makes
            # only outbound requests, so it needs no published port of its own.
            network="host",
            mounts=[
                MountSpec(source=str(store.generated_dir), target="/generated", mode="ro"),
                MountSpec(source=str(store.raw_dir), target="/raw", mode="rw"),
                MountSpec(source=str(store.videos_dir), target="/videos", mode="rw"),
            ],
            log_path=log_path,
            logger=logger,
        )
    else:  # local
        module = _LOCAL_MODULES.get(job.sim.backend)
        if module is None:
            raise RunFailure(
                "sim_launch",
                f"sim.runtime.type=local has no local entry point for backend {job.sim.backend!r} "
                f"(available: {sorted(_LOCAL_MODULES)}); use runtime.type=docker instead",
            )
        adapter = LocalRuntimeAdapter(
            name="sim",
            command=[
                sys.executable, "-m", module,
                "--config", str(config_path),
                "--output", str(store.raw_dir),
                "--videos-dir", str(store.videos_dir),
            ],
            log_path=log_path,
            logger=logger,
        )

    handle = adapter.start()
    logger.event("sim.delegated.started", phase="sim_launch", runtime=rt.type)
    try:
        exit_code = adapter.wait(timeout_s=job.sim.timeout_s)
    except TimeoutError as exc:
        raise RunFailure("sim_run", str(exc)) from exc
    finally:
        adapter.stop(handle)  # safe to call after a completed wait(): see adapter docstrings

    if exit_code != 0:
        tail = adapter.logs(handle)[-1000:]
        raise RunFailure("sim_run", f"delegated sim exited with code {exit_code}: {tail}")

    logger.event("sim.delegated.completed", phase="sim_run", exit_code=exit_code)
    episodes = _normalize_episodes(store, plan)

    if job.rollout.record_video and not any("video" in e.artifacts for e in episodes):
        # Best-effort, backend-dependent: fake_delegated has nothing to render
        # and simply ignores the flag; a backend that CAN render but didn't
        # (e.g. a real failure) would also land here -- either way this is a
        # signal worth surfacing, not a reason to fail a run that otherwise
        # produced valid metrics.
        logger.warning("rollout.record_video was requested but no episode produced a video "
                       "(backend may not support it, or rendering failed -- check sim.log)")

    return episodes


def _normalize_episodes(store: ArtifactStore, plan: ExecutionPlan) -> list[EpisodeRecord]:
    episodes = []
    for path in sorted(store.raw_dir.glob("episode_*.json")):
        raw = json.loads(path.read_text())
        artifacts = {"raw_record": f"raw/{path.name}"}
        video_path = store.videos_dir / f"episode_{raw['episode_id']:04d}.mp4"
        if video_path.is_file():
            artifacts["video"] = f"videos/{video_path.name}"
        episodes.append(
            EpisodeRecord(
                run_id=plan.run_id,
                episode_id=raw["episode_id"],
                env_id=0,
                seed=raw["seed"],
                task_id=raw["task_id"],
                status=raw.get("status", "completed"),
                success=bool(raw["success"]),
                episode_length_steps=raw["steps"],
                total_return=raw["total_return"],
                metrics={"success": float(raw["success"])},
                policy=EpisodePolicyStats(
                    num_requests=raw.get("num_requests", 0),
                    mean_action_horizon=raw.get("mean_action_horizon", 0.0),
                    fallback_steps=0,
                    timeouts=0,
                ),
                artifacts=artifacts,
            )
        )
    return episodes
