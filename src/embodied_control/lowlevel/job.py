"""Job schema for `ec lowlevel run`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LOWLEVEL_API_VERSION = "ec.lowlevel/v1alpha1"


class JobModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvSpec(JobModel):
    backend: Literal["fake", "mujoco"] = "fake"
    model: str | None = None
    model_sha256: str | None = None
    realtime: bool = False
    lag_alpha: float = Field(default=1.0, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_backend(self) -> "EnvSpec":
        if self.backend == "mujoco" and self.model is None:
            raise ValueError("env.backend=mujoco requires env.model")
        return self


class Gr00tSpec(JobModel):
    service_cmd: list[str]
    service_cwd: str | None = None
    mode: Literal["latent", "chunk"] = "latent"
    hold_steps: int = Field(default=10, ge=1)
    slots: int = Field(default=3, ge=1)
    rtc: bool = False
    rtc_freeze_steps: int = Field(default=0, ge=0)
    rtc_ramp_rate: float = Field(default=5.0, gt=0.0)
    # Language chaining: [[start_tick, goal_name], ...] ascending; requests at
    # or after start_tick carry that goal to the service.
    goal_schedule: list[tuple[int, str]] | None = None


class CommandSpec(JobModel):
    topology: Literal["local", "push"] = "local"
    source: Literal[
        "onboard_encoder", "reference_playback", "gr00t_service", "none"
    ] = "onboard_encoder"
    reference: str | None = None
    motion: str | None = None
    hold_steps_override: int | None = Field(default=None, ge=1)
    buffer: Literal["zmq", "shm"] = "zmq"
    endpoint: str | None = None
    topic: str = ""
    shm_name: str | None = None
    gr00t: Gr00tSpec | None = None

    @model_validator(mode="after")
    def validate_topology(self) -> "CommandSpec":
        if self.topology == "push":
            if self.buffer == "zmq" and self.endpoint is None:
                raise ValueError("command.buffer=zmq requires command.endpoint")
            if self.buffer == "shm" and self.shm_name is None:
                raise ValueError("command.buffer=shm requires command.shm_name")
            if self.source != "none":
                raise ValueError("command.topology=push forbids a local source")
        if self.topology == "local" and self.source == "none":
            raise ValueError("command.topology=local requires a source")
        if self.source == "gr00t_service" and self.gr00t is None:
            raise ValueError("command.source=gr00t_service requires command.gr00t")
        return self


class RolloutSpec(JobModel):
    episodes: int = Field(default=1, ge=1)
    max_steps: int = Field(default=500, ge=1)
    seed: int = 0
    record_video: bool = False
    # Log per-tick joint positions + anchor pose to `states_ep{i}.npz` for
    # post-hoc tracking metrics (MPJPE via `eval_mpjpe`).
    record_states: bool = False


class SafetySpec(JobModel):
    state_late_ms: float = 50.0
    state_absent_ms: float = 500.0
    command_stale_ms: float = 200.0
    command_absent_ticks: int = Field(default=100, ge=1)
    damp_kd: float = 8.0
    joint_limit_margin_rad: float | None = None
    max_action_delta: float | None = None
    # Fall detection for sim evaluation: DAMP when the anchor (pelvis) world
    # height drops below this. Mirrors the training env's `base_too_low`.
    min_base_height_m: float | None = None


class OutputSpec(JobModel):
    root: str = "runs"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class LowLevelJob(JobModel):
    api_version: Literal["ec.lowlevel/v1alpha1"] = LOWLEVEL_API_VERSION
    bundle: str
    env: EnvSpec = Field(default_factory=EnvSpec)
    command: CommandSpec = Field(default_factory=CommandSpec)
    rollout: RolloutSpec = Field(default_factory=RolloutSpec)
    safety: SafetySpec = Field(default_factory=SafetySpec)
    outputs: OutputSpec = Field(default_factory=OutputSpec)


def load_lowlevel_job(path: str | Path) -> LowLevelJob:
    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"job file must be a YAML mapping: {path}")
    return LowLevelJob.model_validate(raw)


__all__ = ["LowLevelJob", "load_lowlevel_job", "LOWLEVEL_API_VERSION"]
