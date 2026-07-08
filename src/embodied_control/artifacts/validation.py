"""Validate a run directory against the required files and current schemas."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from embodied_control.config.schemas import (
    EpisodeRecord,
    ExecutionPlan,
    RunManifest,
    RunMetrics,
    RunStatus,
    ValidationReport,
)
from embodied_control.logging.timeutil import utcnow_iso

# (relative path, required?)
_REQUIRED_FILES = [
    "job.yaml",
    "resolved_job.yaml",
    "manifest.json",
    "status.json",
    "metrics.json",
    "episodes.jsonl",
    "logs/orchestrator.log",
    "logs/events.jsonl",
]

_JSON_SCHEMAS = {
    "manifest.json": RunManifest,
    "status.json": RunStatus,
    "metrics.json": RunMetrics,
}


def validate_run_dir(run_dir: str | Path) -> ValidationReport:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    errors: list[str] = []
    warnings: list[str] = []

    for rel in _REQUIRED_FILES:
        if not (run_dir / rel).is_file():
            errors.append(f"missing required artifact: {rel}")

    # resolved_job.yaml -> ExecutionPlan (round-trips the plan schema)
    rj = run_dir / "resolved_job.yaml"
    if rj.is_file():
        import yaml

        try:
            data = yaml.safe_load(rj.read_text())
            plan = ExecutionPlan.model_validate(data)
            run_id = plan.run_id
        except (ValidationError, Exception) as exc:  # noqa: BLE001
            errors.append(f"resolved_job.yaml does not validate as ExecutionPlan: {exc}")

    for rel, model in _JSON_SCHEMAS.items():
        p = run_dir / rel
        if not p.is_file():
            continue
        try:
            model.model_validate(json.loads(p.read_text()))
        except (ValidationError, json.JSONDecodeError) as exc:
            errors.append(f"{rel} does not validate as {model.__name__}: {exc}")

    # episodes.jsonl -> each row is an EpisodeRecord
    ep = run_dir / "episodes.jsonl"
    if ep.is_file():
        for i, line in enumerate(ep.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                EpisodeRecord.model_validate(json.loads(line))
            except (ValidationError, json.JSONDecodeError) as exc:
                errors.append(f"episodes.jsonl line {i + 1} invalid: {exc}")

    return ValidationReport(
        run_id=run_id,
        valid=not errors,
        checked_at=utcnow_iso(),
        errors=errors,
        warnings=warnings,
    )
