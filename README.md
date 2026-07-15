# Embodied Control

A thin host orchestrator for **isolated VLA policy / simulator evaluation**,
built from [the design doc](docs/design/eval_orchestration_interface.md).
Two rollout shapes are supported, chosen by `sim.mode`:

- **stepped** (Milestone 1): the host drives `reset()`/`step()` in-process.
  Used for a lightweight, flexible sim like MuJoCo.
- **delegated** (Milestone 2): a separate runtime — local subprocess or its
  own Docker container — owns its rollout loop end-to-end and calls the
  policy service directly; the host only launches it, waits for it to exit,
  and normalizes its raw output. This is the shape a black-box evaluator
  needs. Proven with a `fake_delegated` evaluator (no simulator dependency),
  then a real one: **LIBERO** manipulation tasks, containerized, driven by
  our own policy service over the same transport.

```text
 host orchestrator                                  policy service (separate runtime)
 ┌───────────────────────────────┐   HTTP/JSON      ┌──────────────────────────────┐
 │ ec eval run                    │  reset / act     │  blank "VLA": zero | random  │
 │  → plan → launch policy svc ───┼─────────────────▶│  emits normalized commands   │
 │                                 │◀─────────────────│  in [-1,1]  (action chunks) │
 │  stepped:                      │  action chunk     └──────────────────────────────┘
 │   drive MuJoCo reacher (step)  │                    Docker container OR local subprocess
 │   embodiment ctrl: cmd → ctrl  │
 │                                 │                   sim runtime (delegated only, separate)
 │  delegated:                    │   HTTP/JSON       ┌──────────────────────────────┐
 │   launch + wait for sim ───────┼──────────────────▶│  owns its own rollout loop;  │
 │   normalize its raw output     │                    │  calls the policy service    │
 │                                 │                    │  directly, same transport    │
 │  → normalize artifacts         │                    └──────────────────────────────┘
 └───────────────────────────────┘                     Docker container OR local subprocess
```

Roles, cleanly separated:

- **Policy service** (the "VLA"): a blank server that emits `zero` or `random`
  commands in a normalized action space. Runs in its **own Docker container**
  (`runtime.type: docker`) or as a **local subprocess** (`runtime.type: local`) —
  same HTTP transport either way. Stdlib-only, so the container image is tiny.
- **Embodiment controller** (the "lower-level controller", stepped mode only):
  maps the policy's normalized command to simulator actuation (`decode_action`).
  Swap in IK, a residual controller, etc. here without touching the policy or env.
- **MuJoCo env** (stepped): an in-process, host-driven reacher — MuJoCo is
  flexible enough to drive directly, no delegated runtime needed.
- **Delegated sim runtime**: a black-box evaluator that owns its own rollout
  loop, in its own local subprocess or Docker container. Reachable regardless
  of runtime combination via host networking (see "Delegated mode" below).

The policy transport is behind a pluggable interface (`endpoint.scheme`):
`http` now, `grpc`/`zmq` later for the image-heavy / high-throughput path.

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
pixi run -e sim smoke-video          # local policy + video
pixi run -e sim smoke-video-docker   # docker policy + video (orthogonal: rendering
                                      # always happens on the host, MuJoCo is in-process)
```

**Delegated mode — a separate runtime owns its own rollout loop** (no MuJoCo
needed; runs in the light default env):

```bash
pixi run smoke-delegated              # everything local: no Docker needed

pixi run build-policy-image           # two containers, wired over the host
pixi run build-fake-delegated-image   # network (see "Delegated mode" below)
pixi run smoke-delegated-docker
```

**Real LIBERO** (a real manipulation task, containerized; several minutes to
build the image, ~50s to run):

```bash
pixi run build-policy-image     # if not already built
pixi run build-libero-image     # several minutes: torch + robosuite + LIBERO's pinned deps
pixi run smoke-libero               # one libero_spatial task, zero policy -> success_rate 0.0
pixi run smoke-libero-video         # same task, random policy, + rendered video (~45s)
pixi run smoke-libero-image-stats   # same task, a policy that actually reads camera images (~15s)
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
logs/               # orchestrator.log (human), events.jsonl (structured), policy.log, sim.log (delegated)
generated/          # eval_result.json (typed summary); sim_config.json (delegated: host -> sim runtime)
raw/                # per-episode backend summary (stepped) or raw evaluator output (delegated)
videos/             # per-episode mp4 (if rollout.record_video: true; MuJoCo stepped and LIBERO)
```

## Delegated mode

`sim.mode: delegated` hands the whole rollout loop to a separate runtime
instead of the host stepping it. The host's job shrinks to: generate a config
telling that runtime how to reach the policy service (`generated/sim_config.json`,
merging in `sim.backend_config` wholesale so the host stays generic about
backend-specific keys), launch it (local subprocess or its own Docker
container, via `sim.runtime`, mirroring `policy.runtime`), wait for it to
exit, and normalize whatever raw per-episode JSON it wrote under `raw/` into
the same `EpisodeRecord`/`metrics.json` contract the stepped path produces.
`orchestration/delegated.py` is the whole host-side implementation — about
100 lines, no plugin registry: two concrete evaluators (`fake_delegated`,
`libero`) sharing one fixed raw-output format. A meaningfully different third
evaluator is what would justify factoring a normalizer registry out of this —
not before.

**`fake_delegated`** (`sim/fake_delegated_eval.py`) has no simulator dependency:
it drives fake episodes, calling the policy service over the *exact same*
transport (`transport.client.PolicyClient`) a stepped rollout uses, and writes
a seeded-but-synthetic success signal per episode. It exists purely to prove
the two-runtime orchestration end-to-end (both containerized, mounts, host
networking, exit-code handling, raw-output normalization) before taking on a
real evaluator's dependency risk.

**`libero`** (`sim/libero_eval.py`) is the real thing: actual LIBERO/robosuite
manipulation episodes, same host-side orchestration, same transport, same raw
JSON output contract, driven by our own policy service (blank zero/random, or
`image_stats`, which actually reads the camera images now sent over the wire —
see "LIBERO" below) rather than a real VLA yet.

**Two containers reaching each other.** The policy container publishes its
port on the host's `127.0.0.1` (`-p 127.0.0.1:<port>:8000`, unchanged from
stepped mode). The delegated sim container runs with `--network host` instead
of Docker's default bridge network: Docker's bridge network offers
container-to-container DNS, but Apptainer/HPC clusters don't, so relying on it
would break portability. A host-network container shares the host's actual
loopback interface, so `127.0.0.1:<port>` resolves correctly regardless of
whether the policy is itself a container or a local process — no
container-to-container DNS needed. The sim container makes only outbound
requests, so it needs no published port of its own.

## LIBERO

`containers/libero_eval/Dockerfile` builds a real LIBERO evaluation image —
robosuite 1.4.0 + LIBERO on Python 3.8 (LIBERO pins `numpy==1.22.4`, which only
has wheels for Python ≤3.10; 3.8 is what LIBERO's own docs recommend), isolated
entirely from the host's Python 3.13 and the `sim` env's MuJoCo 3.x — exactly
why this needs its own container rather than living in the host or `sim` env.
~11GB image; several minutes to build, dominated by compiling native
extensions (numba, scipy, egl_probe) and installing torch/robosuite/transformers.

A few things worth knowing if you touch this container:

- **No GPU passthrough is available in this deployment's Docker daemon** (no
  NVIDIA Container Toolkit configured), so LIBERO's usual EGL rendering path
  doesn't work here. The image sets `MUJOCO_GL=osmesa` (CPU software
  rendering) as a Dockerfile `ENV` — deliberately *not* as a Python-level
  `os.environ` assignment inside the harness, since MuJoCo picks its GL
  backend at the first `import mujoco` anywhere in the process and reuses that
  choice forever; only a container-level env var (present before any process
  starts) reliably beats that race. OSMesa is slower than EGL but needs no
  host/daemon configuration, which makes it the portable default until GPU
  passthrough is set up on a target machine.
- `libero.libero`'s package `__init__.py` runs an **interactive `input()`
  prompt** on its first-ever import if `~/.libero/config.yaml` doesn't exist —
  which would hang a non-interactive container run with no stdin attached.
  The Dockerfile triggers that first import at *build* time
  (`echo "n" | python3 -c "import libero.libero"`), which writes the config
  with the package's own computed default paths and answers the prompt once,
  permanently, so it never fires again at run time.
- The action space is LIBERO's default OSC_POSE controller: a 7-dim
  `[dx, dy, dz, droll, dpitch, dyaw, gripper]` delta, already normalized to
  `[-1, 1]` — the same convention this whole project uses, so no extra
  scaling/adapter layer was needed (`sim.action_dim: 7` in the job config).
- Determinism comes from LIBERO's own pre-generated per-task init-state array
  (`benchmark.get_task_init_states`); the per-episode seed selects which of
  those ~50 states to replay (`seed % len(init_states)`) rather than seeding a
  fresh randomization the harness doesn't otherwise control.
- **Camera observations ARE sent to the policy over the wire.** LIBERO's
  `OffScreenRenderEnv` renders `agentview_image`/`robot0_eye_in_hand_image`
  every step regardless (robosuite always computes them when
  `use_camera_obs=True`); the observation envelope now carries them alongside
  proprioception as a `cameras` list — `{name, encoding: "rgb8", shape,
  dtype, data}`, base64-encoded raw bytes (`transport.protocol.encode_camera`).
  At 128×128×3 that's ~65KB/frame, ~130KB for both cameras — trivial over
  plain HTTP/JSON, so there's no compression/format negotiation yet; that's a
  later optimization if a real policy's payload size ever becomes a real
  problem, not something worth solving speculatively now. The blank
  zero/random policies still never look at any of it — **`image_stats`** is a
  new debug policy that decodes the real bytes and sets its action from their
  mean brightness, which is what actually proves the round trip carries real,
  usable image data end to end (verified against a real LIBERO frame: exactly
  128×128×3 = 49152 bytes, byte-perfect, mean brightness 120.35 — a plausible
  value for a normally-lit scene, not a degenerate 0 or 255). This is exactly
  what a real vision-based VLA policy also needs — see "Real VLA policy
  transports" below for the adapters this made possible.
- **Video is captured separately, for human review, not for the policy.**
  `rollout.record_video: true` keeps the `agentview_image` frame LIBERO is
  already rendering each step (instead of discarding it) and encodes it with
  `imageio` to `videos/episode_XXXX.mp4` — same artifact contract as the
  MuJoCo stepped path. LIBERO's frames render upside down relative to a
  normal image (`frame[::-1]` corrects it — confirmed against LIBERO's own
  `benchmark_scripts/render_single_task.py`, which does the same flip).
  `pixi run smoke-libero-video` renders one.
- The real run: one `libero_spatial` task ("pick up the black bowl between
  the plate and the ramekin and place it on the plate"), 2 episodes × 200
  steps × 128×128 cameras, both containers, ran in **~50 seconds** end to end
  (~45s with video capture + encoding). `success_rate: 0.0` for both zero and
  random policies — expected and correct: unlike the toy MuJoCo reacher, a
  real multi-stage manipulation task is not something a policy that ignores
  observations can stumble into.

## Image registry

Three images so far (`ec-policy-debug`, `ec-fake-delegated-eval`,
`ec-libero-eval`), all currently built locally with `pixi run build-*-image`.
To share a built image instead of making every teammate rebuild it, push to
**GitHub Container Registry** (`ghcr.io`) — no new account needed if you
already have a GitHub login for this repo, and it's what the design doc's own
manifest examples assumed (`ghcr.io/org/...@sha256:...`).

One-time setup:

```bash
gh auth refresh -h github.com -s write:packages   # opens a device-code flow; approve the printed URL
gh auth token | docker login ghcr.io -u <your-github-username> --password-stdin
```

(The `gh` CLI's default token scopes don't include `write:packages` — pushing
will fail with `permission_denied: The token provided does not match expected
scopes` until you've run the `refresh` step above and re-logged in.)

Then, per image:

```bash
pixi run push-policy-image           # -> ghcr.io/<namespace>/ec-policy-debug:latest
pixi run push-fake-delegated-image   # -> ghcr.io/<namespace>/ec-fake-delegated-eval:latest
pixi run push-libero-image           # -> ghcr.io/<namespace>/ec-libero-eval:latest (~11GB, slow to push)
```

`scripts/push_image.sh` derives `ghcr.io/$GHCR_NAMESPACE/<repo>:<tag>` from
the local image name (`GHCR_NAMESPACE` defaults to `fei-yang-wu`; override it
for a different namespace/org). Once pushed, point a job's `runtime.image` at
the registry path instead of the local tag so teammates can run an eval
without building anything themselves — `docker pull` happens automatically on
first `docker run`. For reproducible runs, resolve and record the image
**digest** (`docker inspect --format '{{index .RepoDigests 0}}' <image>`)
rather than trusting a mutable `:latest` tag, matching this repo's own
`RuntimeSpec.image` convention ("digest-pin for serious runs").

Not set up yet: CI-based auto-build-and-push on Dockerfile changes. Manual
push is enough for now while the Dockerfiles are still actively changing (as
this session's LIBERO work showed); revisit once they stabilize. The
`ec-libero-eval` image's size (~11GB) is worth solving for deliberately if/when
this moves to CI — standard GitHub Actions runners have ~14GB of disk total.

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
  transport/        protocol.py (incl. encode_camera/decode_image_bytes/mean_brightness),
                    server.py (stdlib, in container), client.py, factory.py (scheme seam)
  policies/         base.py (zero/random/image_stats) + debug_server.py (policy container entry point)
  sim/              base.py, models.py (reacher MJCF, topdown camera), mujoco_backend.py (stepped + render),
                    fake_delegated_eval.py, libero_eval.py (delegated sim container entry points)
  embodiments/      passthrough.py (normalized cmd -> ctrl; stepped only)
  runtime/          local.py, docker.py (generic: command/mounts/network), ports.py
  orchestration/    planner.py, supervisor.py (policy runtime), delegated.py (sim runtime, stepped's
                    counterpart), runner.py (dispatches on sim.mode), registry.py, errors.py
  metrics/          aggregate.py
  artifacts/        store.py (incl. write_video), validation.py
containers/        policy_debug/Dockerfile, fake_delegated_eval/Dockerfile, libero_eval/Dockerfile
examples/           mujoco_zero_local.yaml, mujoco_random_local.yaml, mujoco_zero.yaml (docker),
                    mujoco_random_video.yaml, mujoco_video_docker.yaml (record_video: true),
                    fake_delegated_local.yaml, fake_delegated_docker.yaml (two containers),
                    libero_docker.yaml, libero_video_docker.yaml, libero_image_stats_docker.yaml
                    (real LIBERO task, two containers)
scripts/            build_*_image.sh, push_image.sh (-> ghcr.io)
```

## Tests

```bash
pixi run test                          # light env: schemas, transport, policies (incl. camera wire
                                        # protocol), metrics, validation, logging, runtime adapters,
                                        # delegated eval (mujoco + openpi/gr00t adapter tests skipped)
pixi run -e sim test-sim               # full suite incl. end-to-end MuJoCo evals
pixi run -e transports test-transports # OpenPI + GR00T client adapter tests (websockets/pyzmq/msgpack)
```

Delegated-mode tests run in the light default env (no mujoco/docker needed):
they use the local runtime for both the policy and the fake evaluator, and the
LIBERO example job is schema-validated (loads, correct backend/action_dim/
runtime type) without requiring the ~11GB image or a GPU. The two-container
Docker paths (`smoke-delegated-docker`, `smoke-docker`, `smoke-libero`) are
covered by manual smoke testing, not automated pytest — real LIBERO execution
in particular is too heavy (image build, CPU rendering) for a test suite that
should stay fast; it was verified manually end to end (zero and random
policies, real task, real success/failure signal, artifacts validated).

## Real VLA policy transports (OpenPI, GR00T)

`transport/openpi_client.py` and `transport/gr00t_client.py` speak OpenPI's
(websocket + msgpack-numpy) and GR00T's (ZeroMQ + msgpack) real wire
protocols directly, verified against each project's actual server/client
source — not our own HTTP/JSON forced onto them. Point a job at an
already-running server with no launch config needed:

```yaml
policy:
  endpoint:
    scheme: openpi_websocket   # or gr00t_zmq
    host: 127.0.0.1
    port: 8000
    action_dim: 7              # asserted by you; checked against the env's action_dim
    observation_mapping:       # translates our neutral observation into the
      proprio_key: "observation/state"   # checkpoint's expected keys — see
      prompt_key: "prompt"               # transport/openpi_translate.py /
      camera_keys:                       # transport/gr00t_translate.py
        agentview_image: "observation/image"
```

Test with `pixi run -e transports test-transports` (websockets/pyzmq/msgpack/
numpy — kept out of the light default env; see `pixi.toml`'s `transports`
feature). Both adapters are exercised against fake servers speaking the real
wire protocol (`tests/fake_openpi_server.py`, `tests/fake_gr00t_server.py`),
including a full `run_eval()` run.

**Both real checkpoints have actually been run and succeeded**: OpenPI's
public `pi05_libero` (`examples/libero_openpi_external.yaml`) and NVIDIA's
public `GR00T-N1.7-LIBERO` (`examples/libero_gr00t_external.yaml`), each
driven through this repo's own orchestrator against real LIBERO episodes
with video evidence. `docs/design/real_policy_adapters.md`'s M3/M4 sections
have the full results and the real, only-found-by-actually-running-them
bugs that surfaced along the way — a *different* wire-envelope mismatch for
each transport, neither one catchable by reading source or running the
fixture-based unit tests alone. **Important**: the process running
`ec eval run` needs the same transport deps as the job's
`policy.endpoint.scheme` — use `pixi run -e transports ec eval run
<job>.yaml` for any job with `scheme: openpi_websocket`/`gr00t_zmq`, not the
light default env.

## Not in this milestone

Managed launching of a real policy server (today you point at one that's
already running via `policy.endpoint`, or launch it with an explicit
`RuntimeSpec.command` override — see `docs/design/real_policy_adapters.md`'s
M5 notes; there's no automatic "build and run the real server image" step),
gRPC transport for *our own* protocol (not needed — HTTP/JSON payloads are
small at 128×128), JPEG/compressed image encoding (raw is small enough for
now), Apptainer/HPC, IsaacLab-Arena, CI-based image builds, and cross-run
comparison reporting. See the design doc for the sequencing.
