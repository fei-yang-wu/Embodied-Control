# Embodied Control

A thin host orchestrator for **isolated VLA policy / simulator evaluation**. This
is Milestone 1: a minimal MuJoCo eval that demonstrates the separation model from
[the design doc](docs/design/eval_orchestration_interface.md).

```text
 host orchestrator (Pixi `sim` env)                policy service (separate runtime)
 ┌───────────────────────────────┐   HTTP/JSON     ┌──────────────────────────────┐
 │ ec eval run                    │  reset / act    │  blank "VLA": zero | random  │
 │  → plan → launch policy svc ───┼────────────────▶│  emits normalized commands   │
 │  → drive MuJoCo reacher (step) │◀────────────────│  in [-1,1]  (action chunks)  │
 │  → embodiment controller maps  │  action chunk   └──────────────────────────────┘
 │    normalized cmd → ctrl       │                  Docker container  OR  local subprocess
 │  → normalize artifacts         │
 └───────────────────────────────┘
```

Three roles, cleanly separated:

- **Policy service** (the "VLA"): a blank server that emits `zero` or `random`
  commands in a normalized action space. Runs in its **own Docker container**
  (`runtime.type: docker`) or as a **local subprocess** (`runtime.type: local`) —
  same HTTP transport either way. Stdlib-only, so the container image is tiny.
- **Embodiment controller** (the "lower-level controller"): maps the policy's
  normalized command to simulator actuation (`decode_action`). Swap in IK, a
  residual controller, etc. here without touching the policy or the env.
- **MuJoCo env**: an in-process, host-driven *stepped* reacher (no delegated
  backend — MuJoCo is flexible enough to drive directly). LIBERO, which is
  service-shaped, is the delegated + Docker milestone.

The transport is behind a pluggable interface (`endpoint.scheme`): `http` now,
`grpc`/`zmq` later for the image-heavy / high-throughput path.

## Setup

```bash
pixi install            # light host env (orchestrator, CLI, stdlib policy service)
pixi install -e sim     # adds MuJoCo for the simulator backend
pixi run doctor         # check host + dependencies
```

The default env is intentionally light (no MuJoCo). The MuJoCo backend lives in
the `sim` feature env, so eval commands run with `pixi run -e sim ...`.

## Run an eval

**Policy as a local subprocess (no Docker needed):**

```bash
pixi run -e sim smoke          # zero policy  -> success_rate 0.0 (baseline)
pixi run -e sim smoke-random   # random policy -> ~0.8, with action chunking
```

**Policy in its own Docker container (the designed separation):**

```bash
pixi run build-policy-image    # build ec-policy-debug:latest (stdlib, ~tiny layer)
pixi run -e sim smoke-docker
```

> If your shell isn't in the `docker` group yet, run under `newgrp docker` /
> `sg docker -c '...'`, or reboot after being added.

**With a rendered video (offscreen MuJoCo render, top-down camera):**

```bash
pixi run -e sim smoke-video
```

Each run writes a normalized directory under `runs/<run_id>/`:

```text
job.yaml            # user input copy
resolved_job.yaml   # the immutable ExecutionPlan (auditable)
manifest.json       # provenance: host, runtimes, image, schema ids
status.json         # final status + phase (always written, even on failure)
validation.json     # schema/artifact validation report
metrics.json        # success_rate, mean_return, policy latency percentiles
episodes.jsonl      # one row per episode (raw per-episode outcomes preserved)
logs/               # orchestrator.log (human), events.jsonl (structured), policy.log
generated/          # eval_result.json (typed summary)
raw/                # per-episode backend summary
videos/             # per-episode mp4 (if rollout.record_video: true)
```

## Logging

A single structured logger is built once per run (`EcLogger.create`, IsaacLab-
style) and passed by constructor injection into every component — the MuJoCo
backend, the embodiment controller, the runtime adapters, the policy
supervisor, and the transport client each hold their own scoped `self.logger`
rather than reaching for a global. Every component gets its own view via
`logger.child("sim")`, `logger.child("policy")`, etc. — real stdlib child
loggers, so they propagate to the root's handlers with no extra wiring.

Every run produces two synchronized logs under `logs/`:

- `orchestrator.log` — human-readable, e.g.
  `2026-07-07 21:58:06 [INFO] ec.run.<id>.rollout: sim.episode.started (phase=rollout episode_id=1 seed=8)`
- `events.jsonl` — the same events as structured JSON rows (`run_id`, `logger`,
  `level`, `event`, `phase`, plus any extra fields), for automation/analysis.

The level is configurable per job (`outputs.log_level: DEBUG|INFO|WARNING|ERROR`,
default `INFO`) or overridden per invocation: `ec eval run job.yaml --log-level DEBUG`.
A component with no logger injected gets `EcLogger.null()` (a stdlib
`NullHandler`-backed logger) rather than crashing or reaching for a global.

## Inspect

```bash
pixi run -e sim ec eval validate runs/<run_id>     # validate against current schemas
pixi run ec eval list-runs runs                    # list runs + success rates
pixi run ec policy ping --endpoint 127.0.0.1:8756  # health + describe an endpoint
pixi run -e sim ec doctor                          # also checks imageio + offscreen renderer
```

## The task

A built-in 2-DOF planar **reacher** (shipped as an MJCF string — no asset
downloads). Position actuators: the normalized command maps to a target joint
angle. Success = the fingertip came within a threshold of a per-episode random
target at any point. The `zero` policy holds home (≈0% success); the `random`
policy explores and reaches (~80%) — a clear, honest signal that the metrics
pipeline works, and a demonstration of action chunking (one held command per
chunk → far fewer policy requests than steps).

The model also carries a fixed top-down `<camera>` for offscreen rendering
(`rollout.record_video: true`): each episode's frames are encoded to
`videos/episode_XXXX.mp4` via `imageio`/`ffmpeg`. Rendering is best-effort —
a failure (e.g. no EGL/OSMesa) disables video for the rest of that episode,
logs a warning, and never fails the eval; the run's actual output is the
metrics. On headless Linux (shared workstations, HPC compute nodes), MuJoCo's
GL backend defaults to EGL automatically (no `export MUJOCO_GL=egl` needed) —
`ec doctor` reports whether the offscreen renderer is actually usable on the
current host.

## Layout

```text
src/embodied_control/
  cli.py            ec entry point (doctor, eval run/validate/list-runs, policy serve/ping)
  config/           schemas.py (pydantic contracts) + loader.py
  logging/          config.py (LogConfig), logger.py (EcLogger: human log + events.jsonl)
  transport/        protocol.py, server.py (stdlib, in container), client.py, factory.py (scheme seam)
  policies/         base.py (zero/random) + debug_server.py (container entry point)
  sim/              base.py, models.py (reacher MJCF, topdown camera), mujoco_backend.py (stepped + render)
  embodiments/      passthrough.py (normalized cmd -> ctrl)
  runtime/          local.py, docker.py, ports.py (RuntimeSpec -> engine)
  orchestration/    planner.py, supervisor.py, runner.py, registry.py
  metrics/          aggregate.py
  artifacts/        store.py (incl. write_video), validation.py
containers/policy_debug/Dockerfile
examples/           mujoco_zero_local.yaml, mujoco_random_local.yaml, mujoco_zero.yaml (docker),
                    mujoco_random_video.yaml (record_video: true)
```

## Tests

```bash
pixi run test              # light env: schemas, transport, policies, metrics, validation (mujoco eval skipped)
pixi run -e sim test-sim   # full suite incl. end-to-end MuJoCo evals
```

## Not in this milestone

LIBERO (delegated + Docker), gRPC/ZMQ transports (image-heavy path), camera
observations, real VLA services (GR00T/OpenPI), Apptainer/HPC, and cross-run
comparison reporting. See the design doc for the sequencing.
