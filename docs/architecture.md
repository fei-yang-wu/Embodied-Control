# Architecture: as-built

Status: describes what actually exists in the repo today, as opposed to
`docs/design/eval_orchestration_interface.md`, which is the original
planning document (aspirational, written before any of this was built — read
that for *why* the shape was chosen; read this for *what's actually there*).
Last updated: 2026-08-17.

For exact run commands see `README.md`. This document is the mental model:
what the pieces are, why they're separated the way they are, and where the
edges of what's built actually are.

## The core idea: policy, controller/evaluator, and simulator are three
different processes that don't import each other

The whole repo is built around one boundary: **the policy (the "VLA") is a
separate process from the simulator**, always, communicating over a small
HTTP/JSON protocol (`transport/`). Everything else is in service of that
boundary being real rather than aspirational — different Python versions on
either side, different dependency trees, containers that don't need each
other's packages installed.

There are two ways the simulator side can be structured:

- **stepped** — the host orchestrator (`orchestration/runner.py`) directly
  calls `reset()`/`step()` on an in-process simulator object. Used for the
  small reacher (`sim/mujoco_backend.py`) and the Vega U + Wuji Hand V2 Beta
  1-with-mount table grasp (`sim/wuji_vega/backend.py`): MuJoCo is lightweight
  enough to import into the host process, while each policy still runs as a
  separate service.
- **delegated** — a *separate* runtime (local subprocess or its own Docker
  container) owns the *entire* rollout loop and calls the policy service
  itself; the host only launches that runtime, waits for it to exit, and
  normalizes whatever raw JSON it wrote (`orchestration/delegated.py`). Used
  for anything too heavy or too black-box to import into the host: LIBERO
  (Python 3.8, `numpy==1.22.4`, robosuite) and `fake_delegated_eval` (a
  synthetic evaluator that exists purely to prove this pattern works before
  LIBERO's real dependency risk was taken on).

Both shapes produce the same artifact contract
(`artifacts/store.py`, `config/schemas.py::EpisodeRecord`) — `episodes.jsonl`,
`metrics.json`, `manifest.json`, `status.json`, `videos/`. Nothing downstream
of a run directory needs to know which shape produced it.

## Why delegated mode uses host networking, not Docker bridge

When both the policy and the sim are containers, the sim container is
launched with `--network host` (`orchestration/delegated.py`), not Docker's
default bridge network with published ports. Docker's bridge network offers
container-to-container DNS (each container can resolve the other by name);
Apptainer — the container runtime this repo's target HPC cluster actually
uses — does not offer that at all for unprivileged users (see
`docs/gotchas.md`). Host networking is the one model that works the same way
on both: a host-network container shares the host's actual network stack, so
`127.0.0.1:<port>` always resolves to whatever's bound there, regardless of
whether that's another container's published port or a local process. The
sim container only ever makes outbound requests, so it needs no published
port of its own — the asymmetry (policy uses bridge+publish, sim uses host)
is deliberate, not an inconsistency.

## The policy transport is deliberately not what a real VLA speaks

`transport/` (`protocol.py`, `client.py`, `server.py`) is a small,
stdlib-only HTTP/JSON protocol: `health`/`describe`/`reset`/`act`, JSON
bodies, camera images base64-encoded inline. It is used by exactly one thing:
our own debug policies (`policies/base.py` — `zero`, `random`,
`image_stats`), which exist to exercise every other part of the system
(orchestration, artifacts, containers, camera wire format) without needing a
real model. **Real policies (OpenPI, GR00T) each ship their own server
speaking their own protocol** (websocket+msgpack, ZeroMQ+msgpack
respectively) — wiring one in means writing a client adapter behind the same
`health/describe/reset/act` interface, not changing this protocol. See
`docs/transport-comparison.md` for the concrete comparison and recommended
shape.

## Logging: one injected logger, not a global

`logging/logger.py::EcLogger` is built once per run and passed into every
component's constructor (backend, controller, runtime adapters, supervisor,
transport client) rather than imported as a module-level singleton. Each
component gets its own scoped view via `.child("sim")`, `.child("policy")`,
etc. — real stdlib child loggers, so they propagate to the root's
handlers automatically. A component with no logger injected gets
`EcLogger.null()` (stdlib `NullHandler` pattern) instead of crashing or
reaching for a global. This exists specifically so tests, the CLI, and the
orchestrator can all construct the same objects without fighting over a
shared global logging config.

## Containers: three so far, each isolating a different dependency problem

| Image | Isolates | Size | Base |
|---|---|---|---|
| `ec-policy-debug` | nothing heavy — stdlib only | 176MB | `python:3.13-slim` |
| `ec-fake-delegated-eval` | nothing heavy — stdlib only, proves the delegated pattern | 176MB | `python:3.13-slim` |
| `ec-libero-eval` | LIBERO's pinned, Python-3.8-only dependency chain | ~11GB | `python:3.8-slim` |

All three copy in only the subtree of `src/embodied_control/` each actually
needs (not the whole package) — this is what keeps the first two tiny despite
being part of the same codebase as the 11GB LIBERO image. Pushed to
`ghcr.io/fei-yang-wu/` (see README "Image registry"); pulling doesn't require
building.

## What's real vs. what's still a blank policy

Everything in this repo — MuJoCo, both delegated backends, video capture,
camera observations — has been exercised with our own `zero`/`random`/
`image_stats` debug policies, never a real trained model. That's a
deliberate sequencing choice (design D8: prove the plumbing with something
cheap before taking on a real model's dependency risk), not an oversight, but
it means every "success rate" number produced so far reflects a policy that
either does nothing or reacts to trivial statistics — not evidence about any
real model's capability. The next real capability gap is a client for
whichever transport OpenPI/GR00T actually speak (see
`docs/transport-comparison.md`); after that, per-policy observation-format
translation (their trained normalization stats, action space conventions)
becomes the real work.

## Where to look for more detail

- `README.md` — exact commands, artifact directory layout, per-feature docs
  (Delegated mode, LIBERO, Image registry, Logging sections).
- `docs/design/eval_orchestration_interface.md` — the original design
  rationale (read this for *why*, not *what exists*; some of it describes
  milestones not yet built, e.g. Apptainer, IsaacLab-Arena).
- `docs/gotchas.md` — specific bugs/environment traps hit and fixed, worth
  reading before touching the same subsystem again.
- `docs/transport-comparison.md` — the real-VLA-policy transport research.
