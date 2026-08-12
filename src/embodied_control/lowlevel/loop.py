"""50 Hz control loop: state machine, watchdogs, tick accounting, artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np

from embodied_control.logging import EcLogger
from embodied_control.lowlevel.job import LowLevelJob
from embodied_control.lowlevel.safety import SafetyFault, SafetyMonitor, damp_command
from embodied_control.lowlevel.tracker import LowLevelTracker


@dataclass
class EpisodeResult:
    episode_id: int
    steps: int
    status: str
    damp_cause: str | None = None
    renewals: int = 0
    unavailable_ticks: int = 0
    tick_ms_p50: float = 0.0
    tick_ms_p99: float = 0.0


@dataclass
class RunResult:
    episodes: list[EpisodeResult] = field(default_factory=list)
    engine_stats: dict | None = None

    @property
    def succeeded(self) -> bool:
        return bool(self.episodes) and all(
            e.status in {"completed", "reference_finished"} for e in self.episodes
        )


class ControlLoop:
    """Drive one tracker against one eval-env backend.

    States: CONTROL and DAMP. INIT ramp and WAIT are hardware concerns (M3);
    sim backends start at the default pose, matching the env's reset.
    """

    def __init__(self, job: LowLevelJob, tracker: LowLevelTracker, backend, publisher=None,
                 logger: EcLogger | None = None):
        self.job = job
        self.tracker = tracker
        self.backend = backend
        self.publisher = publisher
        self.logger = logger or EcLogger.null()
        self.state_logs: dict[int, dict[str, np.ndarray]] = {}

    def run_episode(self, episode_id: int, seed: int) -> EpisodeResult:
        control_hz = self.tracker.bundle.manifest.rates.control_hz
        clock = self.backend.clock()
        self.backend.reset(seed)
        state = self.backend.read_state()
        self.tracker.reset(state)
        if self.publisher is not None:
            self.publisher.reset()
        monitor = SafetyMonitor(spec=self.job.safety, control_hz=control_hz)
        self.logger.event("episode.started", phase="rollout", episode_id=episode_id, seed=seed)

        result = EpisodeResult(episode_id=episode_id, steps=0, status="completed")
        tick_ms: list[float] = []
        joint_log: list[np.ndarray] = []
        anchor_log: list[np.ndarray] = []
        for step in range(self.job.rollout.max_steps):
            clock.wait_for_tick(step, control_hz)
            started = time.perf_counter()
            state = self.backend.read_state()
            if (
                self.job.rollout.record_states
                and state.anchor_pos_w is not None
                and state.anchor_quat_w is not None
            ):
                joint_log.append(np.array(state.joint_pos, dtype=np.float32))
                anchor_log.append(
                    np.concatenate([state.anchor_pos_w, state.anchor_quat_w]).astype(
                        np.float32
                    )
                )
            now = clock.now()
            try:
                monitor.check_state(state, now)
                if self.publisher is not None:
                    self.publisher.tick(step, now, state)
                joint_command, sample = self.tracker.step(step, state)
                monitor.check_command(sample)
                if sample.renewed:
                    result.renewals += 1
                if joint_command is None:
                    result.unavailable_ticks += 1
                else:
                    self.backend.write_command(joint_command)
            except SafetyFault as fault:
                self._damp(fault.cause)
                result.status = "damped"
                result.damp_cause = fault.cause
                result.steps = step
                break
            except (RuntimeError, ValueError) as exc:
                self._damp("runtime_error")
                result.status = "damped"
                result.damp_cause = f"runtime_error: {exc}"
                result.steps = step
                break
            tick_ms.append((time.perf_counter() - started) * 1000.0)
            result.steps = step + 1
            if self.publisher is not None and self.publisher.exhausted:
                result.status = "reference_finished"
                break
        if tick_ms:
            result.tick_ms_p50 = float(np.percentile(tick_ms, 50))
            result.tick_ms_p99 = float(np.percentile(tick_ms, 99))
        if joint_log:
            self.state_logs[episode_id] = {
                "joint_pos": np.stack(joint_log),
                "anchor_pose_xyzw": np.stack(anchor_log),
            }
        self.logger.event(
            "episode.finished", phase="rollout", episode_id=episode_id,
            status=result.status, steps=result.steps, damp_cause=result.damp_cause or "",
        )
        return result

    def _damp(self, cause: str) -> None:
        self.logger.event("loop.damp", phase="rollout", severity="warning", cause=cause)
        try:
            self.backend.write_command(
                damp_command(self.tracker.bundle.manifest.action.width, self.job.safety.damp_kd)
            )
        except Exception:
            self.logger.event("loop.damp_write_failed", phase="rollout", severity="error")

    def run(self) -> RunResult:
        result = RunResult()
        for episode_id in range(self.job.rollout.episodes):
            result.episodes.append(
                self.run_episode(episode_id, self.job.rollout.seed + episode_id)
            )
        stats = self.tracker.engine.stats
        result.engine_stats = {
            "count": stats.count,
            "mean_ms": stats.mean_ms,
            "p50_ms": stats.p50_ms,
            "p95_ms": stats.p95_ms,
            "p99_ms": stats.p99_ms,
        }
        return result


def write_run_artifacts(run_dir: Path, job: LowLevelJob, bundle_manifest: dict,
                        result: RunResult) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_job.json").write_text(json.dumps(job.model_dump(), indent=2))
    (run_dir / "manifest.json").write_text(json.dumps(bundle_manifest, indent=2))
    with (run_dir / "episodes.jsonl").open("w") as stream:
        for episode in result.episodes:
            stream.write(json.dumps(episode.__dict__) + "\n")
    metrics = {
        "episodes": len(result.episodes),
        "succeeded": result.succeeded,
        "statuses": {e.episode_id: e.status for e in result.episodes},
        "engine": result.engine_stats,
        "tick_ms_p99_max": max((e.tick_ms_p99 for e in result.episodes), default=0.0),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (run_dir / "status.json").write_text(
        json.dumps({"status": "succeeded" if result.succeeded else "failed"}, indent=2)
    )


__all__ = ["ControlLoop", "EpisodeResult", "RunResult", "write_run_artifacts"]
