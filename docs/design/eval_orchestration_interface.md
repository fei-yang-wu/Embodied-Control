# Modular VLA Inference and Evaluation Pipeline

Status: proposed v2.3
Audience: embodied-control developers, simulator owners, VLA policy owners, infra owners
Last updated: 2026-07-07
Primary goal: fast, reproducible, simulator-agnostic evaluation of VLA policies with isolated policy and simulator runtimes.

## 1. Executive Summary

Embodied-Control should be a thin orchestration and evaluation layer for embodied policies, not a simulator framework, not a ROS stack, and not a monolithic VLA runtime. The core product is a reliable way to launch policy services and simulator/evaluator jobs, wire them together through versioned adapter contracts, collect artifacts, normalize metrics, and compare policies across tasks and simulators.

The design should optimize for three things:

1. **Experiment velocity**: a developer can run a local smoke test with one command, then run a real delegated simulator evaluation without changing the top-level experiment interface.
2. **Isolation**: heavy simulator dependencies and VLA model dependencies live in Docker or Apptainer runtimes. The host package remains small and Pixi-managed.
3. **Contract stability**: all simulator backends, policy services, embodiment adapters, metrics, and artifacts conform to explicit versioned contracts.

The core evaluation path should be:

```text
Pixi-managed host orchestrator
    -> starts / monitors policy service runtime
    -> starts / monitors simulator evaluation runtime
    -> exposes only stable configs, RPC endpoints, mounted artifact dirs
    -> normalizes outputs into a common artifact contract
```

ROS should not be on the hot path for simulation evaluation. ROS remains a plugin boundary for real robots, rosbag replay, hardware drivers, existing ROS-native stacks, and safety infrastructure. For offline simulation evaluation, the default path should be direct container-to-container communication via a native RPC or messaging protocol.

### 1.1 Design Review Notes

The plan is directionally strong: it puts Embodied-Control at the right layer, keeps heavyweight dependencies out of the host package, and makes artifacts the durable output rather than logs or simulator-specific reports. The main risk is scope creep. As written, the design could accidentally require a full policy protocol, Docker and Apptainer parity, a stepped runner, runtime telemetry, plugin registries, and real VLA wrappers before the first useful evaluation.

The revised plan should keep two tracks separate:

1. **First useful evaluation path**: host CLI, job schema, artifact contract, wrappers around existing OpenPI and GR00T MuJoCo/LIBERO evaluation paths, Docker/local runtime supervisor, a thin Embodied-Control LIBERO harness for contract tests, and a report that can compare runs.
2. **Production hardening path**: Apptainer/HPC portability, reusable service registry, richer telemetry, external plugin entry points, IsaacLab-Arena as the second serious simulator backend, and a future common policy service only if native OpenPI/GR00T transports are not enough.

The first track should prove that Embodied-Control can launch an evaluator, wire it to a policy service, and normalize results. The second track should make that system portable, auditable, and extensible.

## 2. Design Decisions

### D1. Pixi owns the host developer workflow; containers own heavyweight runtime dependencies.

The host package should be installable and testable through Pixi. Pixi should manage Python, CLI tools, linters, schema validators, unit tests, local fake backends, and lightweight policy clients. It should not attempt to solve IsaacLab, GR00T, OpenPI, CUDA-heavy PyTorch stacks, Omniverse, or simulator GUI dependencies in the host environment.

Recommended host install path:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
pixi install
pixi run doctor
pixi run smoke
```

`doctor` and `smoke` are eval-orchestration commands. The current repository is now a Python-only Pixi workspace: no ROS workspace, no C++ extension, and no simulator imports on package import. The first implementation layer adds the eval CLI, schemas, artifact store, and MuJoCo/LIBERO verification backend on top of that baseline.

Recommended principle: one-line installation for host tooling, one-command runtime checks, containerized model/sim execution.

### D2. Evaluation backends have three modes: harness, delegated, and stepped.

The earlier two-mode split, delegated versus stepped, was too coarse. LIBERO itself is not a black-box evaluator service; it is a Python benchmark library on robosuite/MuJoCo. An Embodied-Control-owned LIBERO integration is therefore a **harness backend**: Embodied-Control owns a small rollout harness that is shipped into a container and imports LIBERO inside that runtime.

OpenPI and GR00T already ship LIBERO evaluation paths. Wrapping those paths is **delegated** from Embodied-Control's point of view, even if the upstream project internally owns a LIBERO harness.

Use these modes:

- **Harness**: Embodied-Control owns the rollout loop, but the loop runs inside a simulator runtime image. This is the default for MuJoCo/LIBERO.
- **Delegated**: an external evaluator owns the rollout loop. Embodied-Control only launches it, monitors it, and normalizes artifacts. This fits IsaacLab-Arena and existing upstream OpenPI/GR00T eval scripts.
- **Stepped**: the host process owns `reset()` / `step()` directly. Use only for lightweight local tests or simulators that are safe to import into the host Pixi environment.

Delegated wrappers around OpenPI and GR00T are the first implementation target. Harness mode is the first Embodied-Control-owned simulator backend and should be built for debug policies, fixtures, and reset/action contract tests.

### D3. Policy services are long-lived, versioned services.

A VLA policy service should load a checkpoint once, expose a health endpoint and a `describe()` endpoint, then serve batched observations into action chunks. The simulator runtime should call the policy service. The policy service should not know simulator internals.

### D4. Embodiment adapters are first-class, versioned plugins.

The adapter is the most important boundary in the system. It translates simulator observations into policy observations and policy actions into simulator actions. It owns joint ordering, action semantics, camera naming, control frequency, action scaling, proprioception layout, frame conventions, clipping, and fallback action behavior.

The adapter has an explicit home:

- In **harness mode**, the adapter executes inside the harness runtime, next to the simulator library. The runtime image must install a versioned `embodied-control-adapters` wheel or the main package with adapter extras.
- In **delegated mode**, adapter behavior is either provided by the delegated evaluator itself or by a first-class `policy_gateway` runtime. A gateway has its own `RuntimeSpec`, health check, logs, and manifest entry.
- In **stepped mode**, the adapter can execute in the host process only if the simulator-native observation/action objects are also host-safe.

The host orchestrator validates adapter config and versions, but it must not construct simulator-native objects in delegated or harness mode.

### D5. Use target-policy native transports first; add `ec.policy` only when needed.

The default VLA inference path should be a direct policy API, not ROS. OpenPI and GR00T already expose practical server/client transports. Treat those transports as first-class `PolicyAdapter` implementations:

- `openpi_websocket_msgpack`
- `gr00t_zmq_msgpack`
- `local_debug`
- optional future `ec_policy_grpc`

Do not make a new `ec.policy` gRPC protocol a blocker or the declared v1 end state. If a policy has no usable server, evaluate existing async inference servers before writing one. If `ec.policy` is added later, keep it unary first: `Health`, `Describe`, `Reset`, and `Act`. Do not add streaming or shared-memory references until the runtime model supports those semantics across container boundaries.

Avoid JSON/base64 in the image-heavy request path. If debugging JSON is needed, add a debug endpoint that records a small subset of examples rather than making the production path JSON-first. For any gRPC path, set and record message-size limits explicitly; default limits are too small for multi-camera batched RGB payloads.

### D6. Artifacts are the source of truth.

Every run must produce a manifest, resolved job config, logs, episode records, aggregate metrics, raw simulator output, and optional videos. A run is not complete unless artifacts validate against the schema.

### D7. Runtime specs should compile into an execution plan.

The user writes a concise job YAML. The runner resolves it into a concrete `ExecutionPlan` that includes image digests, mounted paths, ports, environment variables, resource leases, health checks, output locations, Git commit, Pixi lock hash, and backend-generated simulator config.

The resolved execution plan should be persisted so the run can be audited.

### D8. Versioned contracts should start narrow.

Version every public contract from the beginning, but keep the first schemas small. For v0, require only the fields needed to run a delegated eval, identify the policy endpoint, collect artifacts, and compute common metrics. Defer optional fields until a backend or policy actually needs them.

This applies especially to observation/action schemas. Named schemas are better than anonymous tensors, but the first LIBERO schema should describe only the cameras, proprioception, language task fields, and action groups that the selected LIBERO controller actually uses.

### D9. Portable runtime networking is host-port based.

The portable baseline is host networking with planner-allocated localhost ports injected into runtimes as environment variables or generated config. Docker bridge networking and service-name DNS are optional conveniences, not the core contract.

Runtime adapters must fail during planning if a requested field cannot be honored. In particular, rootless Apptainer should not silently accept Docker-only concepts such as bridge networks, container DNS, port mappings, or `shm_size`. Apptainer runs should use clean environment handling and explicit binds; Docker bridge can be enabled only when the selected runtime supports it.

### D10. Failure policy is the single source of truth.

Timeout, retry, fallback, and abort behavior must be decided by `FailurePolicy`. Chunk mode decides when policy requests happen. The adapter only supplies a safe fallback action when `FailurePolicy` allows one. A config that asks for `sync` execution, `fail_episode` on policy timeout, and `fallback_action: hold` is invalid unless the fallback is used only for cleanup after the episode has already failed.

### D11. Wrap prior art before rebuilding it.

The first VLA/LIBERO integrations should wrap existing OpenPI and GR00T evaluation paths and normalize their artifacts. Do not rebuild a LIBERO evaluator, transport stack, and policy server before proving that the artifact/reporting layer can reproduce or explain known reference behavior.

### D12. Record upstream contracts as evidence.

Every backend wrapper should name the upstream workflow it wraps and record the exact repo commit, image digest, task-suite arguments, checkpoint reference, and command template in `resolved_job.yaml`.

Reference prior art for v1:

- OpenPI LIBERO: upstream documents a recommended Docker Compose LIBERO workflow, a remote policy server on port 8000, checkpoint/task-suite arguments, and reference LIBERO numbers. See [OpenPI LIBERO README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) and [OpenPI remote inference](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md).
- GR00T LIBERO: upstream documents LIBERO suites, reported suite results, `LIBERO_PANDA`, `run_gr00t_server.py`, and a LIBERO rollout client with `--n-action-steps`. See [GR00T LIBERO README](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/LIBERO/README.md).
- GR00T action semantics: upstream distinguishes execution horizon from model action horizon and deprecates confusing `--action-horizon` naming. See [GR00T policy docs](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/policy.md).
- Apptainer portability: use explicit binds, clean environment handling, and `APPTAINERENV_*` for injected variables. See [Apptainer user docs](https://apptainer.org/docs/user/latest/).

## 3. Goals

The system should:

- Run comparable evaluations across many simulators and policies.
- Keep the host process small, deterministic, and simulator-agnostic.
- Isolate simulator and VLA dependencies through Docker or Apptainer.
- Support Docker for local development and Apptainer for shared/HPC environments.
- Support harness runtimes first, delegated evaluator runtimes second, and stepped backends only where useful.
- Provide versioned policy adapter contracts for OpenPI websocket/msgpack, GR00T ZMQ, local debug policies, and any future common protocol.
- Provide versioned embodiment adapters and action schemas.
- Normalize artifacts into a stable, queryable run directory.
- Produce cross-run reports with LIBERO suite/task breakdowns, confidence intervals, and exportable tables.
- Reserve trajectory-recording fields for later fine-tuning datasets.
- Make local smoke tests fast enough for CI.
- Make simple experiments easy: zero policy, replay policy, fake smoke tests, then MuJoCo/LIBERO evaluation.
- Reach one real MuJoCo/LIBERO evaluation before building the full production policy-service stack.
- Keep IsaacLab-Arena as the next delegated simulator backend after the MuJoCo/LIBERO path is stable.
- Keep ROS optional and isolated to ROS-specific plugins.

## 4. Non-Goals

The system should not:

- Replace LIBERO, MuJoCo, IsaacLab-Arena, Isaac Lab, Genesis, ManiSkill, Habitat, or any other simulator.
- Require every simulator to expose a Gym-like Python API.
- Require ROS for simulation evaluation.
- Provide hard real-time robot servo control through the VLA network API.
- Hide simulator-specific metrics. Common metrics should be normalized, but backend-specific metrics should be preserved under `metrics.extra`.
- Force one universal observation tensor layout across all models and embodiments.
- Make the host Pixi environment responsible for CUDA/simulator compatibility problems that belong in runtime images.
- Require every delegated backend to adopt the Embodied-Control policy protocol before it can run.
- Build a new policy serving protocol before wrapping the existing OpenPI and GR00T servers.
- Guarantee full Docker/Apptainer feature parity where the engines do not expose equivalent primitives.

## 5. Architecture

### 5.1 High-level system

```text
+-----------------------------------------------------------------------------------+
| Pixi-managed host                                                                  |
|                                                                                   |
|  CLI / API                                                                         |
|    -> Config loader and schema validator                                           |
|    -> Experiment planner                                                           |
|    -> Runtime supervisor                                                           |
|    -> Service registry                                                             |
|    -> Log/event collector                                                          |
|    -> Artifact normalizer                                                          |
|    -> Metrics aggregator                                                           |
|                                                                                   |
|  No heavyweight simulator imports in delegated mode                                |
+-----------------------------------+-----------------------------------------------+
                                    |
                                    | runtime launch, network endpoints, mounted dirs
                                    |
       +-------------------+        +-------------------+        +-------------------+
       | Policy runtime    |        | Policy gateway    |        | Sim/harness       |
       | Docker/Apptainer  |        | optional runtime  |        | Docker/Apptainer  |
       | OpenPI/GR00T/etc. |<------>| adapter + routing |<------>| LIBERO/MuJoCo/etc.|
       | native server     |        | if not in harness |        | rollout + metrics |
       +-------------------+        +-------------------+        +-------------------+
```

### 5.2 Main host components

`ConfigLoader` loads job YAML, applies defaults, validates schemas, and produces an immutable `EvalJob`.

`ExperimentPlanner` resolves job configs into an `ExecutionPlan`. The plan binds ports, resolves relative paths, checks artifact directories, selects runtime implementations, records image references, and generates simulator-specific configs.

`RuntimeSupervisor` starts and stops local processes, Docker containers, or Apptainer commands. It handles lifecycle, signals, health checks, log streaming, exit codes, retries, cleanup, and resource release.

`ServiceRegistry` tracks active policy and gateway services, their endpoints, health status, model identity, checkpoint path, and reuse policy. It is useful after v1; early commands can run a single job without a long-lived registry daemon.

`BackendDriver` is a plugin for each simulator backend. For harness backends, it packages or launches the Embodied-Control-owned rollout harness inside the simulator runtime. For delegated backends, it generates backend config, starts the runtime, and normalizes outputs. For stepped backends, it owns `reset()` / `step()` and calls a `PolicyAdapter` directly.

`ArtifactStore` writes run manifests, logs, JSONL records, metrics, raw outputs, videos, traces, and validation reports.

`MetricsAggregator` computes common aggregate metrics and preserves backend-specific metrics.

`ReportGenerator` creates cross-run summaries, LIBERO suite/task breakdowns, confidence intervals, CSV exports, and paper-friendly tables.

`AssetManager` prefetches simulator assets, benchmark files, model checkpoints, and normalization statistics before a GPU job starts.

`ResumeLedger` treats `episodes.jsonl` as the source of truth for interrupted runs and decides which episodes still need to be scheduled.

## 6. Core Abstractions

### 6.1 EvalJob

An `EvalJob` is the user-facing declaration of what to run.

```python
class EvalJob(BaseModel):
    api_version: Literal["ec.eval/v1alpha1"]
    name: str
    tags: dict[str, str] = {}
    seed: int
    services: dict[str, ServiceSpec]
    sim: SimSpec
    policy: PolicyBinding
    embodiment: EmbodimentBinding
    rollout: RolloutSpec
    outputs: OutputSpec
    resources: ResourceSpec = ResourceSpec()
    failure_policy: FailurePolicy = FailurePolicy()
```

Key rule: the job is declarative. It should not contain arbitrary Python code.

For v1, every registered backend and adapter must publish a versioned Pydantic model for its config block. `backend_config: dict` and `embodiment.config: dict` are acceptable only as parsed input before validation; they should not survive into `ExecutionPlan`.

Minimal v1 schema surface:

```python
class PolicyBinding(BaseModel):
    service: str | None = None
    endpoint: PolicyEndpointSpec
    request_mode: Literal["sync", "backend_native"] = "sync"


class PolicyEndpointSpec(BaseModel):
    scheme: Literal["openpi_websocket_msgpack", "gr00t_zmq_msgpack", "local_debug", "backend_native"]
    host: str | None = None
    port: int | None = None
    port_ref: str | None = None
    transport_version: str | None = None


class ServiceSpec(BaseModel):
    kind: Literal["policy_service", "policy_gateway", "sim_harness"]
    lifecycle: Literal["per_job", "reusable"] = "per_job"
    runtime: RuntimeSpec


class RolloutSpec(BaseModel):
    num_episodes: int
    max_steps_per_episode: int
    execute_steps: int = 1
    deterministic_reset: bool = True
    policy_sampling_seed: int | None = None


class FailurePolicy(BaseModel):
    policy_timeout: Literal["fail_episode", "fail_job", "use_cached_chunk", "hold_action"]
    service_crash: Literal["fail_job"]
    sim_crash: Literal["fail_job"]
    max_retries: int = 0


class EpisodeKey(BaseModel):
    run_id: str
    env_id: int
    episode_id: int
    reset_id: str
    seed: int | None = None


class SimSpec(BaseModel):
    backend: str
    mode: Literal["harness", "delegated", "stepped"]
    runtime: RuntimeSpec | None = None
    backend_config: BaseModel


class EmbodimentBinding(BaseModel):
    adapter: str
    schema_version: str
    observation_schema_id: str
    action_schema_id: str
    config: BaseModel


class OutputSpec(BaseModel):
    root_dir: Path
    run_name_template: str = "{date}_{name}_{seed}"
    artifact_contract_version: Literal["ec.artifacts/v1alpha1"]
    keep_raw: bool = True
    keep_policy_traces: bool = False


class PortSpec(BaseModel):
    name: str
    host: int | Literal["auto"]
    container: int | None = None             # Docker bridge only; invalid for rootless Apptainer.


class MountSpec(BaseModel):
    source: Path
    target: str
    mode: Literal["ro", "rw"] = "ro"


class NetworkSpec(BaseModel):
    mode: Literal["host", "bridge", "none"] = "host"


class GpuSpec(BaseModel):
    devices: list[str] | Literal["all"] | None = None


class ResourceSpec(BaseModel):
    gpus: GpuSpec | None = None
    cpu_limit: str | None = None
    memory_limit: str | None = None


class HealthcheckSpec(BaseModel):
    type: Literal["tcp", "http", "command"]
    port_ref: str | None = None
    command: list[str] | None = None
    timeout_s: int = 60


class RestartPolicy(BaseModel):
    mode: Literal["never", "on_failure"] = "never"
    max_restarts: int = 0
```

`EpisodeKey` is the only identity a stateful policy may use for episode-local cache. `Act` on an unknown key is a `FAILED_PRECONDITION`, not an implicit reset.

### 6.2 ExecutionPlan

An `ExecutionPlan` is the fully resolved, auditable plan generated from an `EvalJob`.

```python
class ExecutionPlan(BaseModel):
    job: EvalJob
    run_id: str
    run_dir: Path
    created_at: datetime
    host: HostFingerprint
    git: GitFingerprint | None
    pixi: PixiFingerprint | None
    runtimes: list[ResolvedRuntime]
    endpoints: dict[str, Endpoint]
    generated_configs: list[GeneratedConfig]
    artifact_contract_version: str
```

The `ExecutionPlan` should be saved as `resolved_job.yaml` before any runtime starts.

### 6.3 RuntimeSpec

A `RuntimeSpec` describes how to run a component, independent of the runtime engine.

```python
class RuntimeSpec(BaseModel):
    type: Literal["local", "docker", "apptainer"]
    image: str | None = None                 # Prefer digest-pinned images for serious runs.
    command: list[str]
    env: dict[str, str] = {}
    mounts: list[MountSpec] = []
    gpus: GpuSpec | None = None
    ports: list[PortSpec] = []
    network: NetworkSpec = NetworkSpec(mode="host")
    working_dir: str | None = None
    shm_size: str | None = None              # Docker-only unless a runtime adapter explicitly supports it.
    healthcheck: HealthcheckSpec | None = None
    restart: RestartPolicy = RestartPolicy()
    timeout_s: int | None = None
```

Runtime adapters must report unsupported fields at plan time. A `RuntimeSpec` that requests Docker bridge networking, service-name DNS, or `shm_size` cannot be silently translated to rootless Apptainer.

Runtime adapters expose a common interface:

```python
class RuntimeAdapter(Protocol):
    def prepare(self, plan: ExecutionPlan, spec: RuntimeSpec) -> PreparedRuntime: ...
    def start(self, prepared: PreparedRuntime) -> RuntimeHandle: ...
    def health(self, handle: RuntimeHandle) -> HealthStatus: ...
    def logs(self, handle: RuntimeHandle) -> Iterator[LogEvent]: ...
    def wait(self, handle: RuntimeHandle) -> RuntimeExit: ...
    def stop(self, handle: RuntimeHandle, reason: str) -> None: ...
```

### 6.4 PolicyService

A `PolicyService` is a long-lived model runtime.

Required operations:

- `Health`: service is reachable and model is loaded or loading.
- `Describe`: returns model identity, supported transport, expected observation schema, action schema, max batch size, trained action chunk length, control timestep, supported cameras, supported dtypes, normalization stats hash, and optional tokenizer/model metadata.
- `Reset`: clears per-episode recurrent state or cached action chunks.
- `Act`: maps one or more observations to one or more action chunks.

A policy service should expose model-level metadata:

```json
{
  "policy_id": "libero-debug-policy",
  "model_family": "debug",
  "checkpoint_ref": "sha256:...",
  "transport_versions": ["openpi_websocket_msgpack/v1"],
  "observation_schema_ids": ["ec.obs.libero/v1"],
  "action_schema_ids": ["ec.action.libero_continuous/v1"],
  "max_batch_size": 8,
  "chunk_length": 50,
  "preferred_control_dt_s": 0.05,
  "normalization_stats_hash": "sha256:...",
  "transport": "openpi_websocket_msgpack"
}
```

Normalization must have one owner. Either the policy service owns normalization and advertises the stats hash, or the gateway/harness owns normalization and records the same stats hash in the manifest. Double-normalization and missing normalization should be plan-time failures when detectable.

### 6.5 SimBackend

A backend is one of three modes.

```python
class HarnessBackend(Protocol):
    name: str
    def plan(self, job: EvalJob, services: ServiceRegistry) -> BackendPlan: ...
    def run(self, backend_plan: BackendPlan, supervisor: RuntimeSupervisor) -> BackendRunResult: ...
    def normalize(self, backend_result: BackendRunResult, store: ArtifactStore) -> EvalResult: ...


class DelegatedEvalBackend(Protocol):
    name: str
    def plan(self, job: EvalJob, services: ServiceRegistry) -> BackendPlan: ...
    def run(self, backend_plan: BackendPlan, supervisor: RuntimeSupervisor) -> BackendRunResult: ...
    def normalize(self, backend_result: BackendRunResult, store: ArtifactStore) -> EvalResult: ...
```

```python
class StepSimBackend(Protocol):
    name: str
    def make(self, job: EvalJob) -> SimSession: ...
```

A delegated backend should be treated as a black-box evaluator with a contract, not as a Python library import. A harness backend may import simulator libraries, but only inside the harness runtime, not in the host orchestrator.

Guarantee matrix:

| Guarantee | Harness | Delegated | Stepped |
| --- | --- | --- | --- |
| Host avoids heavyweight sim imports | yes | yes | no |
| EC owns reset/step loop | yes | no | yes |
| EC can enforce episode reset protocol | yes | backend-dependent | yes |
| EC can enforce action chunk policy | yes | backend-dependent | yes |
| EC can normalize artifacts | yes | yes | yes |
| EC can guarantee policy telemetry | yes | backend-dependent | yes |

Each backend must publish a capability declaration stating which events, metrics, failure classes, reset semantics, video outputs, and policy latency fields it can provide.

Minimal capability declaration:

```yaml
backend: openpi_libero
mode: delegated
capabilities:
  owns_rollout_loop: true
  explicit_episode_reset: backend_native
  vectorized_auto_reset: supported
  policy_latency: best_effort
  videos: supported
  raw_episode_records: supported
  can_enforce_failure_policy: partial
  supported_failure_actions:
    - fail_job
    - fail_episode
  unsupported_runtime_fields:
    apptainer:
      - network.mode=bridge
      - ports[].container
      - shm_size
```

`ExecutionPlan` must copy this declaration and mark any requested behavior as `enforced`, `best_effort`, or `unsupported`.

### 6.6 PolicyAdapter

A `PolicyAdapter` hides transport and policy-specific protocol details. It should not contain robot kinematics, joint ordering, simulator action slicing, or task-specific reward logic.

```python
class PolicyAdapter(Protocol):
    def describe(self) -> PolicyDescription: ...
    def reset(self, episodes: list[EpisodeKey]) -> None: ...
    def act(self, batch: PolicyBatchRequest) -> PolicyBatchResponse: ...
    def close(self) -> None: ...
```

### 6.7 EmbodimentAdapter

An `EmbodimentAdapter` is the robot/task mapping boundary.

```python
class EmbodimentAdapter(Protocol):
    schema_id: str
    robot_id: str
    policy_family: str | None

    def describe(self) -> EmbodimentDescription: ...
    def validate_sim_observation(self, obs: Any) -> ValidationReport: ...
    def encode_observation(self, sim_observation: Any, context: AdapterContext) -> PolicyObservation: ...
    def decode_action(self, policy_action: PolicyAction, sim_observation: Any, context: AdapterContext) -> SimAction: ...
    def hold_action(self, sim_observation: Any, context: AdapterContext) -> SimAction: ...
    def clip_or_reject(self, sim_action: SimAction, context: AdapterContext) -> SimAction | ActionReject: ...
```

The adapter must declare:

- robot name and version;
- joint names and order;
- actuator names and order;
- control mode: absolute joint position, delta joint position, joint velocity, Cartesian waypoint, hybrid action, gripper command, base command;
- units and ranges;
- camera names, image size, color format, intrinsics/extrinsics availability;
- proprioception fields;
- language/task fields;
- tactile fields, if any;
- fallback action semantics;
- safety clipping rules;
- schema version.

## 7. Policy Transport Contract v1

### 7.1 First-class transports

v1 should wrap the policy servers that already exist instead of inventing a new server first.

Required v1 adapters:

- `openpi_websocket_msgpack`: OpenPI-compatible websocket/msgpack-numpy client.
- `gr00t_zmq_msgpack`: GR00T-compatible ZMQ client.
- `local_debug`: deterministic local debug policy for smoke tests.

Optional later adapter:

- `ec_policy_grpc`: a common service for policies that do not already provide a usable server. If added, keep it unary with `Health`, `Describe`, `Reset`, and `Act`. Do not add streaming or shared-memory references until runtime IPC, mount, and namespace semantics are designed.

Every adapter must implement the same logical operations even if the wire format differs:

```text
Health
Describe
Reset(EpisodeKey[])
Act(PolicyBatchRequest) -> PolicyBatchResponse
Close
```

### 7.2 Planning handshake

Before any rollout starts, the planner must:

1. Start or connect to the policy endpoint.
2. Call `Describe`.
3. Check requested observation schema, action schema, control timestep, trained chunk length, normalization stats hash, max batch size, and transport payload limits.
4. Fail the plan with `FAILED_PRECONDITION` if compatibility is not established.

Schema mismatch should never be discovered mid-rollout after GPU time has already been allocated.

### 7.3 Reset and episode lifecycle

`Reset` is explicit and keyed by `EpisodeKey`.

Rules:

- The harness calls `Reset([episode_key])` before the first `Act` for each new episode.
- `Act` on an unknown key is `FAILED_PRECONDITION`.
- When vectorized envs auto-reset asynchronously, each observation carries its own `EpisodeKey`, step index, sim time, and seed.
- A policy adapter must drop recurrent state and cached chunks when it receives `Reset` for a key.
- Contract test: two consecutive episodes on one env must produce the same adapter state transitions as two fresh one-episode runs.

### 7.4 ActRequest

```text
ActRequest
  request_id
  job_id
  action_schema_id
  observation_schema_id
  execute_steps
  observations[]
  context
```

Each observation should be an envelope, not a fixed global tensor:

```text
PolicyObservation
  episode_key
  env_id
  episode_id
  step_index
  sim_time_ns
  control_dt_s
  task
    language_instruction
    task_id
    goal_image optional
  cameras[]
    name
    encoding: rgb8 | bgr8 | jpeg | png | nvjpeg | raw_tensor
    shape
    dtype
    timestamp_ns
    frame_id
    data bytes
  proprioception
    schema_id
    names[]
    values[]
    units[]
    timestamp_ns
  extras
```

### 7.5 ActResponse

```text
ActResponse
  request_id
  status
  actions[]
    episode_key
    env_id
    episode_id
    action_schema_id
    action_chunk
    chunk_length
    control_dt_s
    confidence optional
  timing
    server_receive_ns
    preprocess_ms
    inference_ms
    postprocess_ms
    total_ms
  diagnostics
```

### 7.6 Failure semantics

Policy failures must be explicit. The simulator/evaluator may only apply fallback actions if the job config allows it.

Failure classes:

- `UNAVAILABLE`: service down or not ready; retryable until timeout.
- `DEADLINE_EXCEEDED`: policy did not answer in time; optionally use cached action chunk or hold action.
- `INVALID_ARGUMENT`: schema mismatch; fail job.
- `FAILED_PRECONDITION`: checkpoint/model does not support requested schema, timestep, episode key, or normalization stats; fail job.
- `RESOURCE_EXHAUSTED`: payload too large, batch too large, or GPU memory exceeded; retry with smaller batch only if configured and record the subtype.
- `INTERNAL`: fail job unless explicitly marked retryable by service.

## 8. Action Chunking and Control Semantics

VLA inference is often slower than simulator control frequency, especially with multi-camera image inputs. The architecture should treat action chunks as a first-class primitive, but it must distinguish trained model output length from client execution policy.

Definitions:

- `chunk_length`: fixed property reported by policy `Describe`, for example the number of actions emitted by one model inference.
- `execute_steps`: rollout parameter controlling how many actions the harness executes before re-querying. Open-loop execution can be shorter than `chunk_length`.
- `control_dt_s`: simulator/control timestep. The job, policy description, and adapter must agree on this value or the planner fails with `FAILED_PRECONDITION`.

The runner records how many steps were served from fresh policy inference versus cached chunks, plus how many generated actions were discarded because `execute_steps < chunk_length`.

Recommended request modes:

- `sync`: query required envs, wait for actions, then execute up to `execute_steps`.
- `async_prefetch`: execute current chunk while requesting the next chunk.
- `use_cached_chunk`: if allowed by `FailurePolicy`, continue consuming an already-valid chunk on timeout.
- `fail_on_timeout`: terminate episode/job on policy timeout.

Recommended default for research evaluation: `sync` for correctness-first baselines, then `async_prefetch` for throughput experiments.

The adapter must declare whether actions are:

- absolute joint targets;
- delta joint targets;
- joint velocities;
- end-effector poses;
- base velocities;
- gripper open/close scalars;
- hand joint targets;
- hybrid named commands.

For LIBERO, prefer a named structured action schema over an anonymous action vector. Keep converters to benchmark-native and policy-native layouts, but do not let those raw vectors become the public contract.

Example:

```yaml
action_schema:
  id: ec.action.libero_continuous/v1
  control_dt_s: 0.05
  fields:
    end_effector_delta_position: {shape: [3], unit: m, mode: delta_position}
    end_effector_delta_rotation: {shape: [3], unit: rad, mode: delta_rotation}
    gripper_command: {shape: [1], unit: normalized, mode: continuous}
```

## 9. Runtime and Packaging Strategy

### 9.1 Host Pixi workspace

The host workspace should contain the orchestrator, schemas, fake backends, CLI, tests, docs, and lightweight clients.

Minimal `pixi.toml` shape for the Python-only eval tooling:

```toml
[workspace]
name = "embodied-control"
channels = ["conda-forge"]
platforms = ["linux-64"]

[tasks]
doctor = "ec doctor"
test = "pytest -q"
libero-verify = "ec libero verify --allow-missing"
smoke = "ec eval run examples/libero_mujoco_verify.yaml"

[dependencies]
python = ">=3.10,<3.13"
pytest = ">=8,<9"
pydantic = ">=2,<3"
pyyaml = ">=6,<7"

[pypi-dependencies]
embodied-control = { path = ".", editable = true }
```

Keep ROS, Isaac, OpenPI, GR00T, and other heavyweight stacks out of the default host environment. Add them later as optional runtime images or explicit optional environments, not as default install requirements.

The host package should expose a single CLI, for example:

```bash
ec doctor
ec eval run examples/libero_mujoco_zero.yaml
ec eval list-runs runs/
ec eval summarize runs/*/metrics.json
ec runtime check docker
ec runtime check apptainer
ec policy ping --endpoint localhost:5555
ec bench transport --endpoint localhost:5555 --image-size 224x224 --batch-sizes 1,4,8
```

### 9.2 Docker and Apptainer

Docker should be the default local developer runtime. Apptainer should be a first-class runtime for clusters and shared machines where Docker daemon access is unavailable or undesirable.

Use the same `RuntimeSpec` for both; implement engine-specific translation in the runtime adapter.

Portable networking model:

- The planner allocates host ports.
- Endpoints are injected as env vars or generated config, for example `POLICY_HOST=127.0.0.1` and `POLICY_PORT=43129`.
- Docker bridge networking and service-name DNS are opt-in.
- Apptainer uses host networking assumptions; do not rely on bridge networks, port mappings, or container-name DNS.
- Runtime adapters must error on fields they cannot honor.

For Docker:

- use digest-pinned images for real runs;
- support `--gpus` or NVIDIA visible device mapping;
- set `--shm-size` for image/simulator workloads;
- mount checkpoints/assets read-only when possible;
- mount run output read-write;
- never require mounting the Docker socket into policy or sim containers.

For Apptainer:

- support `.sif` images and OCI-derived images;
- support `--nv` for NVIDIA GPUs;
- support explicit bind mounts;
- assume stricter networking and home-directory policies on clusters;
- use clean environment handling and explicit `APPTAINERENV_*` variables;
- avoid implicit host package leakage through home-directory auto-binds where possible;
- do not accept `shm_size`, bridge networking, or service DNS unless the local Apptainer installation explicitly supports them;
- record the SIF checksum in the run manifest.

### 9.3 Docker Compose

Do not make Compose the primary orchestrator initially. Direct launch through `RuntimeSupervisor` gives tighter lifecycle control and a common abstraction for Docker, Apptainer, and local commands.

However, upstream Compose files are useful prior art. The OpenPI LIBERO integration can import or mirror its Compose service layout, then compile it into Embodied-Control `RuntimeSpec`s. Add `ec eval export-compose job.yaml` as a debugging tool. Compose is useful for reproducing local multi-service layouts and inspecting service/network/mount definitions, but it does not solve Apptainer or cluster execution.

## 10. Config Model

### 10.1 Example first-milestone job config

The first real eval config should wrap an existing OpenPI LIBERO path and normalize its artifacts.

```yaml
api_version: ec.eval/v1alpha1
name: libero_openpi_smoke
seed: 42
tags:
  benchmark: libero
  simulator: mujoco
  policy_family: openpi

services:
  policy:
    kind: policy_service
    lifecycle: per_job
    runtime:
      type: docker
      image: openpi-libero:local
      command:
        - uv
        - run
        - scripts/serve_policy.py
        - --port
        - "8000"
        - policy:checkpoint
        - --policy.config=pi05_libero
        - --policy.dir=/checkpoints/openpi-libero
      gpus:
        devices: ["0"]
      ports:
        - name: policy_ws
          host: 8000
      mounts:
        - source: ./checkpoints/openpi-libero
          target: /checkpoints/openpi-libero
          mode: ro
      healthcheck:
        type: tcp
        port_ref: policy_ws
        timeout_s: 180

sim:
  backend: openpi_libero
  mode: delegated
  runtime:
    type: docker
    image: openpi-libero:local
    command:
      - python
      - examples/libero/main.py
      - --host
      - "127.0.0.1"
      - --port
      - "8000"
      - --task-suite-name
      - libero_spatial
      - --num-trials-per-task
      - "10"
      - --video-out-path
      - /runs/${run_id}/raw/libero/videos
      - --seed
      - "42"
    mounts:
      - source: ./runs
        target: /runs
        mode: rw
  backend_config:
    upstream_workflow: examples/libero/compose.yml
    checkpoint_ref: gs://openpi-assets/checkpoints/pi05_libero
    suite: smoke
    task_selector: first
    robosuite_version: pinned-by-image
    num_envs: 1
    output_dir: /runs/${run_id}/raw/libero

policy:
  service: policy
  endpoint:
    scheme: openpi_websocket_msgpack
    host: 127.0.0.1
    port_ref: policy_ws
  request_mode: sync

embodiment:
  adapter: libero_openpi
  schema_version: v1
  observation_schema_id: ec.obs.libero/v1
  action_schema_id: ec.action.libero_continuous/v1
  config:
    use_language_instruction: true
    clip_actions: true
    normalization_stats: checkpoint_owned

rollout:
  num_episodes: 10
  max_steps_per_episode: 600
  execute_steps: 5
  deterministic_reset: true
  policy_sampling_seed: 420000
  record_video: true

outputs:
  root_dir: ./runs
  run_name_template: "{date}_{name}_{seed}"
  artifact_contract_version: ec.artifacts/v1alpha1
  keep_raw: true
  keep_policy_traces: true
  video:
    enabled: true
    max_episodes: 2

failure_policy:
  policy_timeout: fail_episode
  service_crash: fail_job
  sim_crash: fail_job
  artifact_validation_error: fail_job
```

### 10.2 Config resolution rules

- Every relative path resolves against the job file location unless explicitly marked as workspace-relative.
- Every generated config is written under `runs/<run_id>/generated/`.
- Every runtime port can be `auto`; the resolved port is written to `resolved_job.yaml`.
- Every image should be resolved to a digest for serious runs.
- Every adapter and schema must have a version.
- Environment variables from the user config are logged after redacting secrets.
- `${run_id}` is the only interpolation form inside generated runtime configs. User-facing run names use `run_name_template`.
- Job seed, episode seed, and policy sampling seed are distinct. The planner derives per-episode seeds deterministically and records them in `episodes.jsonl`.
- `policy_sampling_seed` must be forwarded to stochastic policy services when the transport supports it; otherwise the manifest records that policy sampling was uncontrolled.

### 10.3 Endpoint injection for delegated evaluators

Some delegated evaluators already know how to talk to specific policy servers. The config model should preserve the real transport name and let the backend generate simulator-native CLI flags, env vars, or config files from it.

```yaml
policy:
  service: policy
  endpoint:
    scheme: openpi_websocket_msgpack
    host: 127.0.0.1
    port_ref: policy_ws
  request_mode: sync

sim:
  backend_config:
    generated_client_args:
      - --host
      - "{policy.endpoint.host}"
      - --port
      - "{policy.endpoint.port}"
```

Rules:

- The backend may generate simulator-native policy config from this block, but it must not invent a protocol name for an upstream client that does not exist.
- `rollout.execute_steps` controls how many actions are consumed before re-querying; the endpoint only describes transport.
- The manifest must record the resolved transport name, adapter version, endpoint host/port, and any generated client arguments.
- The backend should still collect policy latency and timeout counts when the simulator exposes them.
- If a delegated evaluator completely owns policy loading and exposes no external endpoint, use `scheme: backend_native` and mark policy telemetry and reset enforcement according to the backend capability declaration.

## 11. Artifact Contract

Every run should produce:

```text
runs/<run_id>/
  job.yaml                         # user input copy
  resolved_job.yaml                # immutable resolved execution plan
  manifest.json                    # provenance and checksums
  status.json                      # final run status
  validation.json                  # artifact/schema validation report
  logs/
    orchestrator.log
    policy.log
    sim.log
    events.jsonl
  traces/
    policy_requests.jsonl
    latency.jsonl
    resources.jsonl
  episodes.jsonl
  metrics.json
  metrics_by_task.json
  videos/
  generated/
    simulator_config.yaml
    policy_config.yaml
  raw/
    simulator_specific_outputs/
```

### 11.1 manifest.json

```json
{
  "run_id": "20260707_libero_openpi_smoke_seed42",
  "job_name": "libero_openpi_smoke",
  "created_at": "2026-07-07T15:53:00Z",
  "completed_at": "2026-07-07T15:55:42Z",
  "status": "succeeded",
  "git": {
    "repo": "embodied-control",
    "commit": "abc123",
    "dirty": false
  },
  "pixi": {
    "lock_sha256": "..."
  },
  "runtimes": [
    {
      "name": "policy",
      "engine": "docker",
      "transport": "openpi_websocket_msgpack",
      "image": "openpi-libero:local",
      "source_repo": "Physical-Intelligence/openpi",
      "source_commit": "resolved-openpi-commit",
      "source_workflow": "examples/libero/compose.yml"
    },
    {
      "name": "sim",
      "engine": "docker",
      "image": "openpi-libero:local",
      "source_repo": "Physical-Intelligence/openpi",
      "source_commit": "resolved-openpi-commit",
      "source_workflow": "examples/libero/compose.yml",
      "libero_version": "pinned-by-image",
      "robosuite_version": "pinned-by-image",
      "sha256": "..."
    }
  ],
  "schemas": {
    "artifact": "ec.artifacts/v1alpha1",
    "policy_transport": "openpi_websocket_msgpack",
    "observation": "ec.obs.libero/v1",
    "action": "ec.action.libero_continuous/v1"
  },
  "normalization": {
    "owner": "policy",
    "stats_hash": "sha256:..."
  }
}
```

### 11.2 episodes.jsonl

One row per completed or failed episode:

```json
{
  "run_id": "20260707_libero_openpi_smoke_seed42",
  "job_name": "libero_openpi_smoke",
  "episode_id": 0,
  "env_id": 0,
  "seed": 420000,
  "policy_sampling_seed": 420000,
  "task_id": "libero_smoke_task_0",
  "task_description": "language instruction from the LIBERO task",
  "status": "completed",
  "success": true,
  "episode_length_steps": 184,
  "sim_time_s": 9.2,
  "metrics": {
    "success": true,
    "final_reward": 1.0
  },
  "policy": {
    "num_requests": 37,
    "chunk_length": 50,
    "execute_steps": 5,
    "timeouts": 0,
    "fallback_steps": 0
  },
  "artifacts": {
    "video": "videos/episode_000000.mp4",
    "raw_record": "raw/libero/episode_000000.json"
  },
  "extra": {}
}
```

### 11.3 metrics.json

```json
{
  "run_id": "20260707_libero_openpi_smoke_seed42",
  "job_name": "libero_openpi_smoke",
  "num_episodes_requested": 10,
  "num_episodes_completed": 10,
  "num_episodes_failed": 0,
  "success_rate": 0.7,
  "success_rate_ci95_bootstrap": [0.4, 1.0],
  "mean_episode_length_steps": 184.0,
  "policy_latency_ms": {
    "p50": 84.2,
    "p90": 115.9,
    "p95": 131.8,
    "p99": 188.4
  },
  "transport": {
    "num_requests": 370,
    "timeouts": 0,
    "retries": 0,
    "mean_payload_mb": 1.7
  },
  "sim": {
    "mean_fps": 52.3,
    "mean_env_steps_per_sec": 210.1
  },
  "resources": {
    "policy_gpu_peak_mem_gb": 27.4,
    "sim_gpu_peak_mem_gb": 14.2
  },
  "extra": {}
}
```

## 12. Metrics and Comparability

Use three metric namespaces:

1. `common`: success rate, episode length, return if available, policy latency, failures, resource usage.
2. `task`: task-defined metrics such as object distance, grasp success, collision count, drop count.
3. `extra`: backend-specific or policy-specific details preserved without forcing cross-simulator comparability.

For success rates, record raw per-episode outcomes and compute confidence intervals in the report. Do not only save aggregate success rate; it prevents later statistical analysis.

For latency, report both end-to-end request time and service-side breakdown. Record action chunk length and chunk utilization because chunking can hide high model latency.

### 12.1 Reports and Cross-Run Analysis

`ec report` is a v1 component, not a later convenience script. It should read only run artifacts, never simulator internals.

Required outputs:

- per-run summary table;
- per-suite and per-task breakdown for LIBERO, including spatial/object/goal/long when those suites are used;
- bootstrap confidence intervals over episode outcomes;
- seed-level comparison across policies;
- timeout, retry, fallback, and crash counts;
- policy latency and chunk-utilization summaries;
- CSV export;
- Markdown table export;
- optional LaTeX table export.

`metrics_by_task.json` should be specified as:

```json
{
  "run_id": "20260707_libero_openpi_smoke_seed42",
  "groups": [
    {
      "group_type": "libero_suite",
      "group_name": "smoke",
      "num_episodes": 2,
      "success_rate": 0.5,
      "success_rate_ci95_bootstrap": [0.0, 1.0]
    }
  ]
}
```

### 12.2 Trajectory Recording Reserve

The evaluation artifact contract should reserve fields for later training and fine-tuning data export, even if v1 only records summaries.

Reserved episode fields:

- `trajectory_ref`;
- `dataset_format`, for example `lerobot`;
- `observation_schema_id`;
- `action_schema_id`;
- `normalization_stats_hash`;
- `policy_sampling_seed`.

Large trajectories should be opt-in and governed by retention policy.

## 13. Failure Handling

The runner should use fail-closed semantics by default.

Recommended behavior:

- Policy health check timeout: fail job.
- Policy schema mismatch: fail job.
- Policy runtime exits unexpectedly: fail job.
- Simulator runtime exits nonzero: fail job.
- Missing required artifact: fail job.
- Episode-level policy timeout: configurable, default `fail_episode`.
- Retry policy: explicit, bounded, and recorded.
- Fallback actions: explicit, bounded, and recorded.
- Orchestrator receives SIGTERM/SIGINT: write `status.json` with `interrupted` if time allows, stop child runtimes, and leave `episodes.jsonl` usable as a resume ledger.

Precedence:

1. `FailurePolicy` decides fail, retry, cached chunk, hold action, or abort.
2. Request mode decides when policy calls occur.
3. Adapter supplies the safe action only when `FailurePolicy` permits fallback.
4. Backend capability declaration decides whether the requested behavior can be enforced. If not, planning fails or the behavior is marked best-effort in `resolved_job.yaml`.

The system should always write `status.json`, even on failure:

```json
{
  "status": "failed",
  "phase": "policy_healthcheck",
  "reason": "deadline_exceeded",
  "started_at": "...",
  "ended_at": "...",
  "logs": ["logs/orchestrator.log", "logs/policy.log"]
}
```

## 14. Logging and Telemetry

Every log event should include:

- timestamp;
- run id;
- component;
- severity;
- phase;
- message;
- optional structured fields.

Use a structured `events.jsonl` alongside human-readable logs. Human logs are for debugging; structured logs are for automation.

Important event types:

- `runtime.started`;
- `runtime.health.ready`;
- `policy.reset`;
- `policy.request.started`;
- `policy.request.completed`;
- `policy.request.timeout`;
- `sim.episode.started`;
- `sim.episode.completed`;
- `run.interrupted`;
- `run.resumed`;
- `artifact.validation.failed`;
- `run.completed`.

## 15. Registry and Plugins

Use registries for simulators, policies, embodiments, metrics, and runtimes.

```python
registry.register_sim_backend("libero_mujoco", LiberoMujocoBackend)
registry.register_sim_backend("openpi_libero", OpenPiLiberoBackend)
registry.register_sim_backend("gr00t_libero", Gr00tLiberoBackend)
registry.register_sim_backend("isaaclab_arena", IsaacLabArenaDelegatedBackend)
registry.register_sim_backend("external_command", ExternalCommandBackend)
registry.register_policy_adapter("openpi_websocket_msgpack", OpenPiWebsocketPolicyAdapter)
registry.register_policy_adapter("gr00t_zmq_msgpack", Gr00tZmqMsgpackPolicyAdapter)
registry.register_policy_adapter("local_debug", LocalDebugPolicyAdapter)
registry.register_embodiment("g1_inspire_gr00t", G1InspireGr00tAdapter)
registry.register_runtime("docker", DockerRuntimeAdapter)
registry.register_runtime("apptainer", ApptainerRuntimeAdapter)
```

Support Python entry points for external packages:

```toml
[project.entry-points."embodied_control.sim_backends"]
my_sim = "my_package.backend:MySimBackend"

[project.entry-points."embodied_control.embodiments"]
my_robot = "my_package.adapter:MyRobotAdapter"
```

## 16. Recommended Package Layout

This is the v1 layout. It extends the Python package, tests, examples, and docs without reintroducing a ROS workspace.

```text
src/embodied_control/
  cli/
    main.py

  config/
    schemas.py
    loader.py

  orchestration/
    planner.py
    supervisor.py
    status.py

  runtime/
    base.py
    local.py
    docker.py

  policies/
    base.py
    openpi.py
    gr00t.py
    debug.py

  sim/
    base.py
    openpi_libero.py
    gr00t_libero.py
    libero_mujoco.py
    isaaclab_arena.py

  embodiments/
    base.py
    libero.py
    g1_inspire.py

  artifacts/
    store.py
    jsonl.py
    validation.py

  report/
    summarize.py

  testing/
    golden_artifacts.py
```

Not building in v1:

- custom scheduler;
- Gymnasium backend;
- ROS backend;
- Docker Compose export;
- gRPC server/client/protobuf stack;
- streaming policy protocol;
- shared-memory tensor transport;
- plugin entry points for external packages;
- long-lived service registry daemon;
- full resource telemetry beyond basic latency and runtime metadata.

## 17. ROS Position

Do not make ROS central to the offline simulation evaluation stack. The default simulation path is:

```text
sim evaluator container <-> policy service container
```

ROS is a separate integration path:

```text
real robot / ROS-native sim
    -> ROS bridge or ROS backend
    -> optional policy service client
    -> artifact normalizer
```

ROS plugins can be valuable for:

- hardware drivers;
- tf2 frame trees;
- rosbag replay;
- robot state infrastructure;
- existing ROS-native simulator stacks;
- safety controllers;
- real robot operations.

This keeps the core stack clean while preserving a path to robot deployment.

For v1 offline evaluation, Python network clients over websocket/msgpack or ZMQ/msgpack are acceptable unless measurement proves otherwise. VLA inference, image preprocessing, and simulator stepping usually dominate per-request overhead, and `execute_steps` amortizes policy calls over multiple control steps. Add ROS, shared memory, or a custom binary transport only after latency traces show the Python client path is the bottleneck for a target benchmark.

## 18. Security and Isolation

Minimum rules:

- Do not mount the host Docker socket into policy or simulator containers.
- Mount checkpoints and assets read-only by default.
- Mount only the run output directory read-write.
- Redact secrets from logs and resolved configs.
- Record image digests and SIF checksums.
- Make network exposure explicit; default policy service ports should bind to localhost or an isolated container network.
- Prefer per-job runtime isolation for benchmark runs.
- Allow reusable policy services for development and throughput sweeps, but record reuse in the manifest.

## 19. Lab and HPC Operations

### SLURM model

The portable HPC model is to run the whole orchestrator inside an allocation:

```bash
sbatch --gres=gpu:1 --time=04:00:00 --wrap "pixi run ec eval run examples/libero_mujoco_zero.yaml"
```

The supervisor should launch child runtimes on localhost inside that allocation. The planner must compare job `timeout_s` with available walltime when available, and should leave enough cleanup time to write `status.json`.

If the orchestrator receives SIGTERM near walltime, it should:

1. mark `status.json` as `interrupted`;
2. flush logs and `episodes.jsonl`;
3. stop child processes/containers if possible;
4. make `ec eval resume <run_dir>` able to continue missing episodes.

### GPU arbitration

Respect `CUDA_VISIBLE_DEVICES`. If no explicit GPU is requested, select from visible devices only. For shared workstations, use advisory per-GPU lockfiles under the run root or a configurable lock directory. The manifest should record requested devices, resolved devices, and `CUDA_VISIBLE_DEVICES`.

### Assets, checkpoints, and images

Do not let GPU jobs discover missing assets at first use. Add:

```bash
ec assets fetch examples/libero_mujoco_zero.yaml
ec runtime prepare examples/libero_mujoco_zero.yaml
```

These commands should pre-stage LIBERO assets, BDDL/task files, model checkpoints, normalization stats, container images, and Apptainer SIF files. The manifest should record asset versions, checkpoint refs, normalization stats hashes, image digests, and SIF checksums.

For clusters without compute-node egress, image pulls and OCI-to-SIF conversion should happen before the scheduled job. Avoid relying on default home-directory caches for multi-GB images or checkpoints.

### Retention and garbage collection

Every run should declare a retention class:

- `summary_only`: keep manifest, status, metrics, reports, and logs;
- `debug`: keep raw per-episode records and selected videos;
- `training`: keep full trajectories;
- `temporary`: eligible for automatic deletion.

Add:

```bash
ec eval gc runs/ --policy configs/retention.yaml
```

If raw artifacts are deleted by retention policy or scratch purge, `status.json` or `validation.json` must reflect that the run is no longer fully materialized.

### Run index and tracking hooks

Runs should remain usable as plain directories, but a lab-wide index is useful. Add a thin post-run reporter that can append manifest and metrics summaries to a local SQLite/Parquet index and optionally mirror summaries to W&B, MLflow, or another tracker. Tracking should read artifacts after the run; it should not require policy or simulator containers to import tracking SDKs.

## 20. Testing Strategy

### Unit tests

- Config validation.
- Runtime spec resolution.
- Port allocation.
- Artifact schema validation.
- Metrics aggregation.
- Embodiment adapter round-trip tests.
- Policy adapter request/response codec tests for OpenPI websocket/msgpack and GR00T ZMQ.
- EpisodeKey reset-state tests.

### Contract tests

Every policy adapter should pass:

- health endpoint test;
- describe endpoint test;
- reset endpoint test;
- single observation act test;
- batched observation act test;
- timeout behavior test;
- invalid schema rejection test.
- act-on-unknown-episode-key rejection test.

Every backend should pass:

- plan generation test;
- run directory creation test;
- log capture test;
- minimal run test;
- artifact normalization test;
- failure artifact test.

### Integration tests

- artifact fixture + `ec report`;
- OpenPI LIBERO smoke;
- GR00T LIBERO smoke;
- Embodied-Control LIBERO harness with zero/debug policy;
- Docker LIBERO eval with host-port endpoint injection;
- interrupted run + resume;
- Apptainer LIBERO eval if available;
- IsaacLab-Arena delegated backend after v1.

### Performance tests

- policy ping latency;
- image payload throughput;
- batch size sweep;
- `execute_steps` sweep with fixed model `chunk_length`;
- simulator num-envs sweep;
- policy service warm-start time;
- artifact write overhead.

## 21. Fast Prototype Milestones

The prototype should be organized around gates. Each gate must leave behind a runnable command and validated artifacts. Avoid building the next layer until the previous layer produces a run directory that another person can inspect.

The first evaluation milestone is deliberately narrower than a full policy rollout: prove that the repo can verify a MuJoCo/LIBERO environment, resolve suites/tasks/assets, and write normalized artifacts from a Python-only Pixi workflow. After that foundation is stable, wrap existing OpenPI LIBERO and GR00T LIBERO evaluation paths, normalize their artifacts, and fill gaps with a thin Embodied-Control harness only where needed.

### Milestone 0 - Host foundation and artifact contract

Goal: make the repo runnable and make run artifacts concrete.

Deliverables:

- Keep the Python-only `pixi.toml`, `pixi.lock`, package skeleton, and tests working.
- Add `ec` CLI skeleton with `ec doctor`, `ec eval run`, `ec eval summarize`, `ec report`, and `ec eval resume`.
- Add minimal Pydantic schemas for `EvalJob`, `RuntimeSpec`, `ExecutionPlan`, `EvalResult`, `EpisodeKey`, episode records, and backend capability declarations.
- Add artifact writing for `job.yaml`, `resolved_job.yaml`, `manifest.json`, `status.json`, logs, `episodes.jsonl`, `metrics.json`, and `metrics_by_task.json`.
- Add golden artifact fixtures with coherent numbers.

Exit criteria:

- `pixi install && pixi run smoke` succeeds without Docker, Apptainer, ROS, MuJoCo, LIBERO, or VLA dependencies.
- `ec eval validate <run_dir>` validates generated artifacts, including logs and structured events.
- `ec eval resume <run_dir>` validates the run and reports non-resumable state for verification-only jobs.
- `ec report <run_dir>` prints a summary and exports CSV/Markdown.

### Milestone 1 - Local MuJoCo/LIBERO verification

Goal: make MuJoCo/LIBERO availability, suite metadata, task selection, and artifact production testable before adding VLA runners or container orchestration.

Deliverables:

- `libero_mujoco` verification backend that checks Python imports for `mujoco`, `robosuite`, and `libero.libero`.
- Suite/task inspection for `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`.
- Optional BDDL/init-state existence checks, init-state loading, and offscreen render smoke.
- Permissive smoke mode for lightweight CI and strict mode for machines with the simulator stack installed.
- Normalized artifacts: `job.yaml`, `resolved_job.yaml`, `manifest.json`, `status.json`, `validation.json`, logs, `episodes.jsonl`, `metrics.json`, and `metrics_by_task.json`.
- Example configs: `examples/libero_mujoco_verify.yaml`, `examples/libero_mujoco_verify_suites.yaml`, and `examples/libero_mujoco_verify_strict.yaml`.

Exit criteria:

- `pixi run smoke` produces a valid permissive run directory without MuJoCo/LIBERO installed.
- `pixi run smoke-suites` records per-suite verification rows for the main LIBERO suites.
- `pixi run smoke-strict` succeeds on a machine/container with MuJoCo, robosuite, LIBERO assets, and init states installed; on lightweight hosts it fails clearly while still writing valid failure artifacts.
- `pixi run libero-verify-strict` is the direct environment gate for a strict local simulator install.

### Milestone 2 - Wrap OpenPI LIBERO evaluator

Goal: get the first real VLA/LIBERO result by wrapping the upstream OpenPI LIBERO evaluation path.

Deliverables:

- `openpi_libero` backend using the upstream OpenPI LIBERO client/server shape.
- `openpi_websocket_msgpack` policy adapter.
- Docker/local runtime specs for OpenPI policy service and LIBERO evaluator.
- Endpoint injection using host/port env vars, not Docker service-name DNS.
- Normalizer from OpenPI/LIBERO outputs to `episodes.jsonl`, `metrics.json`, and `metrics_by_task.json`.
- Asset/checkpoint preflight for LIBERO assets, model checkpoint, and normalization stats.
- Aggregate `metrics.json` with requested episodes, completed episodes, failed episodes, success rate, and mean episode length.
- Example config: `examples/libero_openpi_smoke.yaml`.

Exit criteria:

- `ec eval run examples/libero_openpi_smoke.yaml` runs a tiny LIBERO subset through OpenPI.
- Artifacts validate and include policy transport metadata, normalization stats hash, LIBERO/robosuite versions, and raw output references.
- A small reproduction run is within expected noise of the published OpenPI LIBERO reference for the selected subset, or the discrepancy is explained in `status.json`.

### Milestone 3 - Wrap GR00T LIBERO evaluator

Goal: add the second target-policy transport without inventing a common serving protocol.

Deliverables:

- `gr00t_libero` backend or wrapper around the upstream GR00T LIBERO eval path.
- `gr00t_zmq_msgpack` policy adapter.
- Same artifact normalizer contract as OpenPI.
- Plan-time checks for control timestep, action schema, chunk length, normalization stats, and max payload/batch size.
- Sampling seed derivation for stochastic policies.

Exit criteria:

- `ec eval run examples/libero_gr00t_smoke.yaml` runs the same tiny LIBERO subset.
- `ec report` compares OpenPI and GR00T runs by task/suite, success rate, latency, and failures.

### Milestone 4 - Embodied-Control LIBERO rollout harness and CI mirror

Goal: own the minimal rollout harness only after verification and delegated upstream eval paths are stable.

Deliverables:

- `libero_mujoco` rollout backend for zero/debug policies and controlled contract tests.
- Explicit `EpisodeKey` reset lifecycle.
- Contract test: two consecutive episodes on one env equal two fresh one-episode runs.
- Fake LIBERO-output fixture that exercises the same normalizer as real OpenPI/GR00T runs.
- Resume ledger using `episodes.jsonl`.

Exit criteria:

- Core tests pass without MuJoCo/LIBERO installed by using golden raw-output fixtures.
- `ec eval run examples/libero_mujoco_zero.yaml` runs at least one LIBERO task for a tiny episode budget with a local debug policy.

### Milestone 5 - Reporting and first comparison sweep

Goal: produce the analysis artifact the project actually needs.

Deliverables:

- Config matrix for OpenPI, GR00T, seeds, LIBERO suites/tasks, `execute_steps`, and runtime engine.
- `ec report` with per-suite/per-task breakdown, bootstrap CIs, CSV, Markdown, and optional LaTeX export.
- Runtime metadata and failure summaries.
- Retention policy and `ec eval gc`.

Exit criteria:

- Run at least 3 seeds x 2 policies on a small LIBERO task subset.
- Report success rate, confidence intervals, policy latency, timeout/fallback counts, and artifact links.

### Milestone 6 - Operations hardening

Goal: make the same workflow survivable on lab machines and HPC.

Deliverables:

- Host-port endpoint injection for all runtimes.
- Advisory GPU lockfiles and `CUDA_VISIBLE_DEVICES` handling.
- SLURM/sbatch usage docs.
- `ec assets fetch` and `ec runtime prepare`.
- Interrupted status handling and `ec eval resume`.
- Apptainer adapter only after the Docker/local path is stable.

Exit criteria:

- A 500-episode run interrupted mid-way can resume without duplicating completed episodes.
- A Docker/local run and an Apptainer run, where available, produce the same artifact contract.

### Parallel Workstream - G1 Inspire red-ball schemas

This should not wait for the full LIBERO stack. It depends mostly on schema types and fake observations.

Deliverables:

- Named observation schema for cameras, proprioception, optional tactile data, and task text.
- Named action schema for arms, Inspire hands, base height, and base/navigation command groups.
- Dataset-native vector converter and simulator-native converter.
- Joint/action limit definitions.
- Round-trip, clipping, and missing-field tests.

Exit criteria:

- The adapter can encode a recorded or fake sim observation and decode a policy action chunk into simulator-native actions with validation.

### After v1

- IsaacLab-Arena delegated backend, with an explicit alpha-status risk entry.
- General `external_command` backend.
- Optional ROS bridge.
- Optional `ec_policy_grpc` if native OpenPI/GR00T transports are insufficient.

## 22. First Experiments to Run

Run these before attempting large-scale VLA benchmarks:

1. **Artifact fixture report**: run `ec report` on coherent LIBERO fixture artifacts before any simulator work.
2. **OpenPI LIBERO smoke**: tiny LIBERO subset through the upstream OpenPI client/server path, artifacts validated.
3. **GR00T LIBERO smoke**: same subset through the GR00T ZMQ path, artifacts validated.
4. **OpenPI reference check**: reproduce a small published/reference LIBERO slice within expected noise or document the discrepancy.
5. **LIBERO artifact fixture test**: normalize saved raw outputs without requiring MuJoCo/LIBERO installed in CI.
6. **Episode lifecycle test**: two consecutive episodes on one env must match two fresh one-episode runs at the adapter state level.
7. **Chunk execution sweep**: compare `execute_steps` values while keeping model `chunk_length` fixed.
8. **Small comparison benchmark**: 3 seeds x 2 policies on a small LIBERO task subset.
9. **Interrupted/resume test**: kill a sweep mid-run and resume from `episodes.jsonl`.

## 23. Main Risks and Mitigations

### Risk: adapter logic leaks into policy clients.

Mitigation: policy clients only implement transport and model-specific request formatting. Embodiment adapters own robot/task mapping and are tested independently.

### Risk: universal protocol work delays the first real eval.

Mitigation: make OpenPI websocket/msgpack and GR00T ZMQ first-class adapters. Add `ec.policy` only if a policy lacks a usable server.

### Risk: the adapter has no executable home.

Mitigation: run adapters inside the harness runtime for LIBERO; use a first-class `policy_gateway` runtime only when a delegated evaluator cannot host adapter logic.

### Risk: episode reset state silently corrupts policy evaluation.

Mitigation: define `EpisodeKey`, require explicit reset, reject act-on-unknown-key, and add the two-episodes-versus-two-fresh-runs contract test.

### Risk: Docker and Apptainer drift.

Mitigation: use host-port endpoint injection as the portable baseline. Runtime adapters must fail on unhonorable fields instead of silently accepting Docker-only settings.

### Risk: VLA latency dominates evaluation throughput.

Mitigation: action chunking, batch env observations, async prefetch mode, latency tracing, and policy service reuse for sweeps. Keep `chunk_length` and `execute_steps` separate.

### Risk: artifacts are inconsistent across simulators.

Mitigation: strict common schema plus `metrics.extra` and `raw/` preservation. Add backend-specific normalizer tests.

### Risk: one-line install cannot handle GPU driver/runtime setup.

Mitigation: Pixi handles host tooling; `ec doctor` detects Docker, NVIDIA Container Toolkit, Apptainer, GPU access, mount permissions, and explains missing system prerequisites.

### Risk: ROS becomes a hidden dependency.

Mitigation: keep ROS plugins in optional extras and optional tests. Core tests must pass without ROS installed.

### Risk: IsaacLab-Arena changes under us.

Mitigation: keep it after the LIBERO path, record IsaacLab-Arena/Isaac Sim/Python versions in manifests, and treat Arena backend failures as backend-specific until the alpha surface stabilizes.

## 24. Recommended Near-Term Build Order

Build in this order:

1. Host CLI + schemas + artifact store + report over fixtures.
2. Local MuJoCo/LIBERO verification backend and strict/permissive smoke gates.
3. OpenPI LIBERO wrapper and normalizer.
4. GR00T LIBERO wrapper and normalizer.
5. Embodied-Control LIBERO rollout harness for debug/contract tests.
6. First OpenPI-vs-GR00T LIBERO comparison report.
7. Operations hardening: assets, resume, retention, host-port networking, GPU locks.
8. Parallel G1 Inspire red-ball schemas and fake-observation adapter tests.
9. Apptainer runtime adapter only when a cluster run is needed.
10. IsaacLab-Arena delegated backend after LIBERO is stable.

This order avoids rebuilding infrastructure that target policy repos already provide while still proving the local simulator contract first. MuJoCo/LIBERO remains the first real evaluation target; IsaacLab-Arena becomes the second serious backend.

## 25. Pre-Build Decisions

Resolve these before Milestone 2 starts. They are intentionally small, but leaving them implicit will make the first VLA benchmark hard to reproduce.

1. Choose the exact OpenPI source commit and build source for `openpi-libero:local`. Prefer mirroring the upstream `examples/libero/compose.yml` into `RuntimeSpec`s instead of hand-copying its behavior.
2. Use `pi05_libero` from `gs://openpi-assets/checkpoints/pi05_libero` as the first OpenPI checkpoint unless a local checkpoint is explicitly under test.
3. Start with one `libero_spatial` task, 10 trials, fixed host port `8000`, and `execute_steps: 5`. Move to planner-allocated auto ports after the first smoke run is reproducible.
4. Choose the exact GR00T N1.7 LIBERO checkpoint/subfolder for Milestone 3, for example the shipped LIBERO checkpoint rather than a locally fine-tuned checkpoint.
5. Use `ec.artifacts/v1alpha1` as the canonical artifact schema string for all v1alpha fixtures, examples, and validators.
6. Keep W&B, MLflow, or lab dashboard upload as post-run reporters over normalized artifacts. Do not let benchmark containers publish their own incompatible run records.
7. Record `libero`, `robosuite`, MuJoCo, OpenPI, GR00T, image digest, and source commit versions in every serious run manifest.

## 26. Final Recommendation

The design should explicitly reject ROS as the central inference/evaluation middleware while still preserving a clean ROS bridge for the cases where ROS is genuinely valuable. The production architecture should be a Pixi-managed host orchestrator plus isolated local/Docker/Apptainer runtimes connected through first-class policy adapters for the transports the target policies already use.

For the first serious implementation, do not optimize prematurely for every simulator or every VLA. Land the Python-only MuJoCo/LIBERO verification path first so the local environment, suite metadata, and artifact contract are inspectable. Then wrap OpenPI's and GR00T's existing LIBERO evaluation paths, normalize their artifacts, and produce a comparison report. Build only the missing rollout harness pieces needed for contract tests, resume, and repeatability. Once LIBERO is stable, add operations hardening, pull the G1 Inspire adapter forward in parallel, and integrate IsaacLab-Arena as the next simulator backend.
