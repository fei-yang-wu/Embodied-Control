"""Load and validate an ``EvalJob`` from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import ValidationError

from embodied_control.config.schemas import EvalJob


class ConfigError(RuntimeError):
    pass


def load_job(path: str | Path) -> EvalJob:
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"job file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"job file {p} must be a YAML mapping, got {type(raw).__name__}")
    try:
        return EvalJob.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid job config {p}:\n{exc}") from exc
