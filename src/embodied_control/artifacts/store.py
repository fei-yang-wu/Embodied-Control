"""Write the normalized run directory.

Layout (M1 subset of the design's artifact contract):

    runs/<run_id>/
      job.yaml              # user input copy
      resolved_job.yaml     # the immutable ExecutionPlan
      manifest.json         # provenance + runtimes + schema ids
      status.json           # final run status (always written, even on failure)
      validation.json       # artifact/schema validation report
      metrics.json          # aggregate metrics
      episodes.jsonl        # one row per episode (raw outcomes preserved)
      logs/
        orchestrator.log
        events.jsonl
        policy.log
      generated/
        eval_result.json    # typed EvalResult summary
      raw/
        episode_XXXX.json   # per-episode backend summary (if record_raw)
      videos/
        episode_XXXX.mp4    # per-episode render (if rollout.record_video)
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from pydantic import BaseModel

from embodied_control.config.schemas import (
    EpisodeRecord,
    EvalJob,
    EvalResult,
    ExecutionPlan,
    RunManifest,
    RunMetrics,
    RunStatus,
    ValidationReport,
)


def _dump_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=2)
    else:
        text = json.dumps(obj, indent=2)
    path.write_text(text + "\n")


class ArtifactStore:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.logs_dir = self.run_dir / "logs"
        self.raw_dir = self.run_dir / "raw"
        self.generated_dir = self.run_dir / "generated"
        self.videos_dir = self.run_dir / "videos"

    # --- paths -----------------------------------------------------------
    @property
    def job_path(self) -> Path:
        return self.run_dir / "job.yaml"

    @property
    def resolved_job_path(self) -> Path:
        return self.run_dir / "resolved_job.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"

    @property
    def validation_path(self) -> Path:
        return self.run_dir / "validation.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def episodes_path(self) -> Path:
        return self.run_dir / "episodes.jsonl"

    @property
    def orchestrator_log_path(self) -> Path:
        return self.logs_dir / "orchestrator.log"

    @property
    def events_path(self) -> Path:
        return self.logs_dir / "events.jsonl"

    @property
    def policy_log_path(self) -> Path:
        return self.logs_dir / "policy.log"

    # --- setup -----------------------------------------------------------
    def initialize(self) -> None:
        for d in (self.run_dir, self.logs_dir, self.raw_dir, self.generated_dir, self.videos_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- writers ---------------------------------------------------------
    def write_job(self, job: EvalJob) -> None:
        self.job_path.write_text(yaml.safe_dump(job.model_dump(mode="json"), sort_keys=False))

    def write_plan(self, plan: ExecutionPlan) -> None:
        self.resolved_job_path.write_text(
            yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False)
        )

    def write_manifest(self, manifest: RunManifest) -> None:
        _dump_json(manifest, self.manifest_path)

    def write_status(self, status: RunStatus) -> None:
        _dump_json(status, self.status_path)

    def write_validation(self, report: ValidationReport) -> None:
        _dump_json(report, self.validation_path)

    def write_metrics(self, metrics: RunMetrics) -> None:
        _dump_json(metrics, self.metrics_path)

    def write_result(self, result: EvalResult) -> None:
        _dump_json(result, self.generated_dir / "eval_result.json")

    def write_episodes(self, episodes: list[EpisodeRecord]) -> None:
        with open(self.episodes_path, "w") as fh:
            for e in episodes:
                fh.write(e.model_dump_json() + "\n")

    def write_raw_episode(self, episode_id: int, summary: dict) -> str:
        rel = f"raw/episode_{episode_id:04d}.json"
        _dump_json(summary, self.run_dir / rel)
        return rel

    def write_video(self, episode_id: int, frames: list, fps: int) -> str:
        """Encode collected RGB frames to an mp4 and return its run-relative path.

        Imports ``imageio`` lazily (a ``sim``-env-only dependency) so this
        module stays importable without it when video isn't used.
        """
        import imageio.v2 as imageio

        rel = f"videos/episode_{episode_id:04d}.mp4"
        imageio.mimwrite(str(self.run_dir / rel), frames, fps=fps, quality=8)
        return rel
