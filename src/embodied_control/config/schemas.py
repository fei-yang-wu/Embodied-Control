"""Versioned config + artifact schemas.

Two families of models:

- **Input**: ``EvalJob`` and its parts — the declarative job the user writes.
- **Resolved / output**: ``ExecutionPlan``, ``EpisodeRecord``, ``RunMetrics``,
  ``RunManifest``, ``RunStatus``, ``ValidationReport``, ``EvalResult`` — what the
  orchestrator produces and what ``ec eval validate`` checks against.

Kept intentionally narrow (design D8): only fields the M1 MuJoCo/blank-policy path
actually uses. Every public contract carries an ``api_version``-style id so it can
evolve without silently breaking older runs.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

EVAL_API_VERSION = "ec.eval/v1alpha1"
ARTIFACT_CONTRACT_VERSION = "ec.artifacts/v1alpha1"


# --------------------------------------------------------------------------- #
# Input job                                                                    #
# --------------------------------------------------------------------------- #
class MountSpec(BaseModel):
    source: str
    target: str
    mode: Literal["ro", "rw"] = "rw"


class RuntimeSpec(BaseModel):
    """How to launch a component. M1 supports local subprocess and Docker."""

    type: Literal["local", "docker"] = "local"
    image: str | None = None  # required for type=docker; digest-pin for serious runs
    env: dict[str, str] = Field(default_factory=dict)
    mounts: list[MountSpec] = Field(default_factory=list)
    container_port: int = 8000  # port the service binds *inside* a container
    shm_size: str | None = None  # docker-only; left None so it is honorable on all engines


class EndpointSpec(BaseModel):
    """Connect to an already-running policy service instead of launching one."""

    scheme: Literal["http", "grpc", "zmq"] = "http"
    host: str = "127.0.0.1"
    port: int


class SimSpec(BaseModel):
    backend: Literal["mujoco"] = "mujoco"
    mode: Literal["stepped"] = "stepped"  # MuJoCo is flexible; host drives reset/step
    model: str = "reacher2"  # builtin MJCF key (see sim/models.py)
    model_path: str | None = None  # optional external .xml/.mjcf
    backend_config: dict[str, Any] = Field(default_factory=dict)


class PolicyBinding(BaseModel):
    type: Literal["zero", "random"] = "zero"  # which blank policy the service runs
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)  # launched by the host
    endpoint: EndpointSpec | None = None  # OR: connect to an external service (no launch)
    requested_action_horizon: int = 1
    action_chunk_mode: Literal["sync"] = "sync"


class EmbodimentBinding(BaseModel):
    """The robot/task mapping boundary (the "lower-level controller").

    For M1 the controller maps a normalized policy command in [-1, 1] to MuJoCo
    actuator ``ctrl`` values. A real deployment can swap in IK, a residual
    controller, etc. behind the same adapter.
    """

    adapter: Literal["passthrough"] = "passthrough"
    observation_schema_id: str = "ec.obs.mujoco_proprio/v1"
    action_schema_id: str = "ec.action.mujoco_normalized/v1"
    clip_actions: bool = True
    fallback_action: Literal["zero", "hold"] = "zero"


class RolloutSpec(BaseModel):
    num_episodes: int = 3
    max_steps_per_episode: int = 200
    record_raw: bool = True
    record_video: bool = False
    video_fps: int = 30


class OutputSpec(BaseModel):
    root_dir: str = "./runs"
    run_name_template: str = "{date}_{name}_{seed}"
    artifact_contract_version: str = ARTIFACT_CONTRACT_VERSION
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class EvalJob(BaseModel):
    api_version: Literal["ec.eval/v1alpha1"] = EVAL_API_VERSION
    name: str
    seed: int = 0
    tags: dict[str, str] = Field(default_factory=dict)
    sim: SimSpec
    policy: PolicyBinding = Field(default_factory=PolicyBinding)
    embodiment: EmbodimentBinding = Field(default_factory=EmbodimentBinding)
    rollout: RolloutSpec = Field(default_factory=RolloutSpec)
    outputs: OutputSpec = Field(default_factory=OutputSpec)


# --------------------------------------------------------------------------- #
# Resolved execution plan                                                      #
# --------------------------------------------------------------------------- #
class HostFingerprint(BaseModel):
    platform: str
    python_version: str
    hostname: str
    user: str | None = None


class ResolvedEndpoint(BaseModel):
    scheme: str = "http"
    host: str
    port: int


class ResolvedRuntime(BaseModel):
    name: str
    type: str  # local | docker | external
    image: str | None = None
    command: list[str] = Field(default_factory=list)
    container_name: str | None = None
    host_port: int | None = None
    container_port: int | None = None


class ExecutionPlan(BaseModel):
    api_version: str = EVAL_API_VERSION
    job: EvalJob
    run_id: str
    run_dir: str
    created_at: str
    host: HostFingerprint
    seeds: list[int]
    action_dim: int
    action_schema_id: str
    observation_schema_id: str
    policy_endpoint: ResolvedEndpoint
    policy_runtime: ResolvedRuntime
    artifact_contract_version: str = ARTIFACT_CONTRACT_VERSION


# --------------------------------------------------------------------------- #
# Result artifacts                                                             #
# --------------------------------------------------------------------------- #
class EpisodePolicyStats(BaseModel):
    num_requests: int = 0
    mean_action_horizon: float = 0.0
    fallback_steps: int = 0
    timeouts: int = 0


class EpisodeRecord(BaseModel):
    run_id: str
    episode_id: int
    env_id: int = 0
    seed: int
    task_id: str
    status: Literal["completed", "failed"]
    success: bool
    episode_length_steps: int
    total_return: float
    metrics: dict[str, float] = Field(default_factory=dict)
    policy: EpisodePolicyStats = Field(default_factory=EpisodePolicyStats)
    artifacts: dict[str, str] = Field(default_factory=dict)


class LatencyStats(BaseModel):
    count: int = 0
    mean: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0


class RunMetrics(BaseModel):
    run_id: str
    num_episodes_requested: int
    num_episodes_completed: int
    num_episodes_failed: int
    success_rate: float
    mean_episode_length_steps: float
    mean_return: float
    num_policy_requests: int
    policy_latency_ms: LatencyStats = Field(default_factory=LatencyStats)
    extra: dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    run_id: str
    job_name: str
    created_at: str
    completed_at: str | None = None
    status: str
    seed: int
    host: HostFingerprint
    runtimes: list[ResolvedRuntime] = Field(default_factory=list)
    schemas: dict[str, str] = Field(default_factory=dict)
    artifact_contract_version: str = ARTIFACT_CONTRACT_VERSION


class RunStatus(BaseModel):
    status: Literal["succeeded", "failed", "running"]
    phase: str
    reason: str | None = None
    started_at: str
    ended_at: str | None = None
    logs: list[str] = Field(default_factory=list)


class ValidationReport(BaseModel):
    run_id: str
    valid: bool
    checked_at: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EvalResult(BaseModel):
    run_id: str
    run_dir: str
    status: str
    num_episodes: int
    metrics: RunMetrics
