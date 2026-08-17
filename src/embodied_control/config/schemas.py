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

from pydantic import BaseModel, Field, model_validator

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
    # Explicit launch command, verbatim (each element may reference "{host}"/
    # "{port}", substituted at launch time). Overrides the built-in debug-
    # server command builder -- required for launching anything other than
    # our own debug policy service, since a real server's CLI shape (e.g.
    # OpenPI's serve_policy.py, GR00T's start_server) isn't something we can
    # derive from job.policy.type. See docs/design/real_policy_adapters.md.
    command: list[str] | None = None


class EndpointSpec(BaseModel):
    """Connect to an already-running policy service instead of launching one."""

    scheme: Literal["http", "openpi_websocket", "gr00t_zmq"] = "http"
    host: str = "127.0.0.1"
    port: int
    # Real-policy adapters can't introspect this from the checkpoint the way
    # our own debug policies can -- the job author asserts it, and it's
    # checked against the env's action_dim the same way describe() already
    # is for http (see orchestration/runner.py's policy_describe phase).
    action_dim: int | None = None
    # Scheme-specific translation config (e.g. OpenPIObservationMapping's
    # fields for "openpi_websocket") -- see docs/design/real_policy_adapters.md.
    observation_mapping: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _non_http_requires_action_dim(self) -> "EndpointSpec":
        if self.scheme != "http" and self.action_dim is None:
            raise ValueError(
                f"endpoint.action_dim is required when scheme={self.scheme!r} "
                "(a real-policy adapter can't introspect this from the checkpoint)"
            )
        return self


class SimSpec(BaseModel):
    backend: Literal[
        "mujoco", "wuji_vega_grasp", "fake_delegated", "libero"
    ] = "mujoco"
    # stepped: host drives reset/step in-process (MuJoCo is flexible enough for this).
    # delegated: a separate runtime (local subprocess or Docker container) owns and
    # runs its own rollout loop end-to-end, calling the policy service directly; the
    # host only launches it, waits for exit, and normalizes its raw output. This is
    # the shape a black-box evaluator (e.g. LIBERO) needs (design D2/6.5).
    mode: Literal["stepped", "delegated"] = "stepped"
    model: str = "reacher2"  # builtin MJCF key (see sim/models.py); stepped only
    model_path: str | None = None  # optional external .xml/.mjcf; stepped only
    backend_config: dict[str, Any] = Field(default_factory=dict)

    # --- delegated mode only ---
    action_dim: int | None = None  # required for delegated (no in-process model to introspect)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    timeout_s: int = 120  # how long to wait for the delegated runtime to exit

    @model_validator(mode="after")
    def _delegated_requires_action_dim(self) -> "SimSpec":
        if self.mode == "delegated" and self.action_dim is None:
            raise ValueError("sim.action_dim is required when sim.mode='delegated'")
        if self.backend == "wuji_vega_grasp":
            if self.mode != "stepped":
                raise ValueError("sim.backend='wuji_vega_grasp' requires sim.mode='stepped'")
            if self.model_path is None:
                raise ValueError("sim.backend='wuji_vega_grasp' requires sim.model_path")
            cfg = self.backend_config
            if int(cfg.get("frame_skip", 10)) < 1:
                raise ValueError("sim.backend_config.frame_skip must be >= 1")
            if float(cfg.get("cube_xy_noise", 0.0)) < 0:
                raise ValueError("sim.backend_config.cube_xy_noise must be >= 0")
            if float(cfg.get("lift_threshold", 0.04)) <= 0:
                raise ValueError("sim.backend_config.lift_threshold must be > 0")
            if float(cfg.get("success_hold_s", 0.5)) <= 0:
                raise ValueError("sim.backend_config.success_hold_s must be > 0")
        return self


class PolicyBinding(BaseModel):
    type: Literal["zero", "random", "image_stats", "wuji_grasp_oracle"] = "zero"
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)  # launched by the host
    endpoint: EndpointSpec | None = None  # OR: connect to an external service (no launch)
    requested_action_horizon: int = 1
    action_chunk_mode: Literal["sync"] = "sync"

    @model_validator(mode="after")
    def _oracle_requires_explicit_runtime(self) -> "PolicyBinding":
        if (
            self.type == "wuji_grasp_oracle"
            and self.endpoint is None
            and self.runtime.command is None
        ):
            raise ValueError(
                "policy.type='wuji_grasp_oracle' requires policy.runtime.command "
                "or policy.endpoint"
            )
        return self


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
    action_dim: int | None = None
    observation_mapping: dict[str, Any] = Field(default_factory=dict)


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
    # Whatever the policy's describe() returned (policy_id, action_dim, and
    # for real-policy adapters the checkpoint's own server_metadata) -- so
    # "which checkpoint produced this success rate" is answerable from the
    # run directory itself, months later, not just from the job config's
    # intent (see docs/design/real_policy_adapters.md).
    policy_describe: dict[str, Any] = Field(default_factory=dict)
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
