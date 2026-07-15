# AGENTS.md

Instructions for AI agents (and a decent quick-start for humans) working in
this repository. If you're a Claude Code agent, `CLAUDE.md` points here —
this file is the canonical source.

## What this is

A host orchestrator for evaluating VLA (vision-language-action) policies
across simulators, built for a robotics lab. The core commitment: **the
policy always runs as a separate process from the simulator**, communicating
over a small HTTP/JSON protocol, so heavyweight/conflicting dependencies
(different Python versions, different MuJoCo versions, GPU vs CPU rendering)
stay isolated in their own containers instead of fighting each other in one
environment.

Read `docs/architecture.md` before making any non-trivial change — it
explains the mental model (stepped vs delegated evaluation, why containers
are isolated the way they are, what's real vs. a blank debug policy). This
file is commands and conventions; that file is *why*.

## Environment

Pixi-managed, two environments:

```bash
pixi install            # light default env: orchestrator, CLI, stdlib policy service
pixi install -e sim     # adds MuJoCo, for the simulator backend
```

The default env is deliberately light — no MuJoCo, no heavy simulator deps.
LIBERO's dependencies live entirely inside its own Docker image
(`containers/libero_eval/`), never in any pixi environment; that image is
Python 3.8 with pinned old dependencies (`numpy==1.22.4`) unrelated to and
incompatible with the host's Python 3.13.

## Commands

```bash
pixi run test                          # default env: ~49 tests, ~6s — run this after every change
pixi run -e sim test-sim               # full suite incl. real MuJoCo evals: ~54 tests, ~11s
pixi run -e transports test-transports # OpenPI/GR00T client adapter tests (websockets/msgpack/numpy)
pixi run doctor                        # host + dependency check (imageio, offscreen renderer, docker, ...)
```

Docker-runtime examples (`smoke-docker`, `smoke-delegated-docker`,
`smoke-libero*`) are **not** covered by `pixi run test` — they need images
built first (`pixi run build-*-image`) and are verified manually, documented
in the README's "Tests" section. Don't assume they're covered by CI-style
testing; if you touch a Dockerfile or a container entry point, rebuild and
run the corresponding `pixi run smoke-*` task yourself before calling it
done.

If a `docker` command fails with a permission error, the current shell
session likely predates being added to the `docker` group — use
`sg docker -c '<command>'` or open a fresh terminal (see `docs/gotchas.md`
if the symptom is stranger than plain permission-denied).

## Conventions actually followed in this codebase

- **No comments explaining *what* code does.** Comments exist only for
  non-obvious *why* — a hidden constraint, a workaround for a specific bug,
  something that would surprise a reader. If you're tempted to write "# loop
  over episodes", don't.
- **Logger is constructor-injected, never a global.** Every component that
  logs takes an `EcLogger | None` parameter, defaulting to `EcLogger.null()`
  if not given. Get a component's own logger via `parent_logger.child("name")`
  — don't create ad-hoc loggers or import a module-level singleton.
- **Containers copy only the source subtree they need**, not the whole
  package — check an existing `containers/*/Dockerfile`'s `COPY` lines
  before adding a new container, and keep whatever you import from
  `transport/`/`logging/`/`policies/` genuinely stdlib-only if the container
  it ends up in is supposed to stay tiny (see `docs/gotchas.md`'s
  `LogConfig` story for what happens when that slips).
- **Every run produces the full artifact contract**, success or failure:
  `job.yaml`, `resolved_job.yaml`, `manifest.json`, `status.json` (always
  written, even on failure), `validation.json`, `metrics.json`,
  `episodes.jsonl`, `logs/`. Don't shortcut this for a new backend — if you
  add a delegated evaluator, its raw per-episode JSON needs a normalizer in
  `orchestration/delegated.py` producing the same `EpisodeRecord` shape
  everything else uses.
- **Schema/config validation errors belong in pydantic model validators**,
  not runtime checks buried in `orchestration/runner.py`. If a field
  combination is invalid (e.g. `sim.mode=delegated` without
  `sim.action_dim`), it should fail at `EvalJob` construction time, before
  any run directory exists — see the `model_validator` on `SimSpec` for the
  pattern. A runtime-only check for something the schema could have caught
  means a failed run silently skips writing `status.json` and friends,
  which breaks the "always produces artifacts" contract above.
- **Prove new architecture with something cheap before something real.**
  The delegated-evaluation pattern was proven with `fake_delegated_eval` (no
  simulator dependency at all) before LIBERO's real dependency risk was
  taken on. The camera wire protocol was proven with `image_stats` (reads
  real image bytes, does something trivially observable) before wiring in
  an actual vision policy. Follow this pattern for the next real
  capability, don't skip straight to the expensive/risky version.
- **Don't build the "proper" abstraction for one instance.** There is no
  plugin registry for delegated-evaluator normalizers — there are two
  evaluators (`fake_delegated`, `libero`) and one `if/else`. Add the
  registry when a third, meaningfully different evaluator actually needs
  it, not before (this is a repeated, deliberate choice in this codebase,
  not an oversight to "fix").

## Current status (2026-07-14)

Everything is still exercised only with our own blank debug policies
(`zero`/`random`/`image_stats`) and, as of M2, a fake OpenPI-protocol test
server — **no real trained VLA model has been wired in yet.** The adopted
plan is `docs/design/real_policy_adapters.md` (milestones M1–M4:
chunk-scheduler refactor → OpenPI adapter → real π0-LIBERO end-to-end →
GR00T adapter); M1 and M2 are done. `transport/openpi_client.py` +
`transport/openpi_translate.py` speak OpenPI's real websocket+msgpack-numpy
protocol (verified against its actual source, not assumed) and are wired all
the way through the orchestrator (`EndpointSpec.scheme="openpi_websocket"`,
`transport/factory.py`, `orchestration/supervisor.py`) — what's missing is a
real checkpoint's translation config (M3), not plumbing. The underlying
protocol research is `docs/transport-comparison.md`. Read both before
touching this area — the plan encodes decisions (neutral observations,
partial-vs-full handshake validation, why chunking stays stdlib-only) that
aren't obvious from the code alone.

Also not yet built: Apptainer/HPC support, IsaacLab-Arena backend, GPU
passthrough for containerized rendering (works today via CPU/OSMesa),
cross-run comparison/reporting tooling, CI-based image builds (still
manual, see README "Image registry"). See
`docs/design/eval_orchestration_interface.md` for the original sequencing
if picking up one of these.

## Docs map

- `README.md` — commands, artifact layout, per-feature usage docs.
- `docs/architecture.md` — as-built system design and rationale.
- `docs/gotchas.md` — specific bugs/traps already hit; check before
  re-touching rendering, container builds, process lifecycle, or the
  registry.
- `docs/transport-comparison.md` — OpenPI/GR00T/gRPC transport research for
  wiring in a real policy, validated against StarVLA and Isaac Lab-Arena.
- `docs/design/real_policy_adapters.md` — adopted plan for wiring in real
  VLA policies (the current next milestone).
- `docs/design/eval_orchestration_interface.md` — original design doc
  (aspirational/planning; some of it isn't built yet).
