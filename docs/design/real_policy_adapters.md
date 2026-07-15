# Plan: wiring in real VLA policies (OpenPI, GR00T)

Status: adopted plan, nothing implemented yet. Builds on the research in
`docs/transport-comparison.md` (protocol details verified against OpenPI and
GR00T source, then validated against how StarVLA and Isaac Lab-Arena solved
the same problem). Last updated: 2026-07-14.

## Goal

Run a real trained VLA policy (first target: OpenPI's LIBERO-finetuned π0
checkpoint) against our existing LIBERO delegated evaluator, producing the
same artifact contract every run produces today — without changing the
delegated architecture, the debug HTTP transport, or how simulators produce
observations.

## What we take from StarVLA / Isaac Lab-Arena, and where we go further

Both projects (see `docs/transport-comparison.md`, "Validation" section)
independently converged on the shape we already recommended: native
per-model-family transports behind one shared policy interface, per-family
containers. We copy three things outright:

- **Raw numpy bytes over msgpack for images** — no base64 on the real-policy
  wire.
- **Server-owned chunk length, advertised at handshake** — the checkpoint
  decides the action horizon; the client caches a chunk and re-requests only
  when it's exhausted. `requested_horizon` becomes advisory.
- **One centralized chunk scheduler** (Arena's `action_chunk_scheduler.py`
  idea) instead of per-evaluator copies of the consume/refill loop.

And we deliberately improve on four things neither does well:

1. **Handshake contract validation, fail-fast.** At connect time, compare
   what the server says it wants/provides (OpenPI `get_server_metadata()`,
   GR00T `get_modality_config()`) against what the job config says we'll
   send. Mismatch = hard error written to `validation.json` *before* episode
   0, not a warning scrolled past in a log (StarVLA warns; Arena doesn't
   check at all). A 2-hour eval that was silently resized wrong is worse
   than a run that refuses to start.
2. **Server metadata persisted into run artifacts.** Whatever
   `describe()` returns (checkpoint identity, chunk size, modality config)
   goes into `manifest.json`. Neither project records this; for a lab it's
   the difference between "success rate 0.62" and "success rate 0.62 *for
   which checkpoint, exactly?*" six months later.
3. **Translation parameterized by config, not by files.** StarVLA has one
   interface file per benchmark×model pair (`model2libero_interface.py`,
   `model2metaworld_interface.py`, ...); Arena has per-embodiment adapter
   classes. We keep exactly one translation layer per *policy family*,
   parameterized by config (camera-key mapping, resize target, proprio
   ordering) so sims × policies doesn't become a grid of near-identical
   files. Named custom transforms only when a real case actually demands
   one (repo convention: no registry for one instance).
4. **Unmanaged-server mode first.** Both projects assume you launch policy
   servers through their own scripts/orchestrators. Our first milestone
   adds `policy service: external` — connect to a server somebody already
   started (e.g. on the lab's GPU workstation) — so adapter work doesn't
   block on generalizing `PolicyServiceSupervisor`'s launch logic. Managed
   launching of real servers is a later, separate milestone.

## Target shape

```
evaluator loop (runner.py / fake_delegated_eval.py / libero_eval.py)
    └── ChunkScheduler            transport/chunking.py    stdlib-only, shared
          └── policy client       one of:
                PolicyClient          transport/client.py         http (exists)
                OpenPIWebsocketClient transport/openpi_client.py  websockets+msgpack-numpy
                Gr00tZmqClient        transport/gr00t_client.py   pyzmq+msgpack
                    └── per-family translation (config-parameterized)
```

Key structural decisions:

- **The shared surface stays `health / wait_healthy / describe / reset /
  act`** — formalized as a `typing.Protocol` in `transport/base.py` (three
  implementations is the threshold where the repo's "no abstraction for one
  instance" rule flips; same pattern as `runtime/base.py`).
- **Transport ≠ translation.** The transport class speaks the wire (fixed
  per server *software*); translation converts our neutral observation into
  that checkpoint's expected dict shape (fixed per *checkpoint*). Composed
  inside the client, kept as separable pieces, because the same OpenPI
  server code serves differently-configured checkpoints.
- **Encoding moves from evaluator to client.** Today
  `libero_eval._wire_observation` base64-encodes cameras before the client
  ever sees them; that forces every transport through our JSON envelope.
  Instead the evaluator builds a *neutral* observation (numpy arrays +
  proprio + task language) and each client encodes natively — HTTP client
  does the base64/JSON it does today (wire format unchanged), msgpack
  clients pack arrays directly. This is what makes new transports
  zero-touch on evaluators.
- **`transport/chunking.py` must stay stdlib-only** — it runs inside every
  delegated container, including the stdlib-only ones. It's list-index
  logic; this costs nothing. The msgpack clients are *not* copied into
  stdlib-only containers (`fake_delegated_eval` stays HTTP-only).
- **Scheme names**: `EndpointSpec.scheme` (config/schemas.py) currently
  allows `"http" | "grpc" | "zmq"`. The placeholders become
  `openpi_websocket` and `gr00t_zmq` — `"zmq"` alone is ambiguous, and
  `"grpc"` is dead (rejected in transport-comparison.md).
- **Single-env scoping for real adapters.** Our wire supports batched
  `episode_keys[]`; both real servers reset one session. Every backend
  built so far uses `env_id=0` anyway — the adapters map single-key calls
  and reject batches explicitly rather than pretending.

## Milestones

Ordering follows the repo's prove-cheap-before-real rule: refactors that
de-risk under test first, then the transport with a ready-made real
checkpoint for our proven backend, then the second transport.

### M1 — extract the chunk scheduler + neutral observation (pure refactor)

The consume/refill loop is duplicated three ways today
(`orchestration/runner.py:199-241`, `sim/fake_delegated_eval.py:60-74`,
`sim/libero_eval.py:130-147`), and observation encoding lives in the
evaluators. Two steps, no new dependencies:

1. `transport/chunking.py::ChunkScheduler` — owns buffer, refill-on-empty,
   per-request horizons bookkeeping (`num_requests`, `mean_action_horizon`).
   All three loops refactored onto it.
2. Neutral observation: evaluators stop calling `encode_camera` themselves;
   `PolicyClient.act()` encodes. HTTP wire bytes stay identical.

**Accept when**: `pixi run test` + `pixi run -e sim test-sim` green; fake
+ LIBERO images rebuilt and their `smoke-*` tasks pass with per-episode
records equivalent to before.

### M2 — OpenPI adapter

`transport/openpi_client.py` (websockets + msgpack-numpy), a fake
OpenPI-wire echo server as a test fixture (~60 lines, covered by
`pixi run test`), factory + schema scheme `openpi_websocket`, handshake
metadata → `manifest.json`, handshake validation → `validation.json`,
external-server mode (skip supervisor launch, just `wait_healthy` against a
configured host:port).

**Accept when**: unit tests round-trip real msgpack frames against the
fixture; a LIBERO smoke run against the fixture (blank actions) produces
the full artifact contract including recorded server metadata.

### M3 — real end-to-end: π0-LIBERO

OpenPI ships LIBERO-finetuned checkpoints and a serving script, and LIBERO
is our proven backend — this is the cheapest possible first real model.
Server runs on a GPU host (outside Docker until NVIDIA Container Toolkit is
configured — see `docs/gotchas.md`); LIBERO container connects out via host
networking, which it already uses. The real work here is translation
config: `agentview_image` / `robot0_eye_in_hand_image` / proprio / task
language → the checkpoint's expected keys (OpenPI normalizes server-side,
so the client sends raw observations with the right keys and resolution).

**Accept when**: a multi-episode LIBERO run against the real checkpoint
completes with a nonzero success rate and full artifacts — the first
number this repo produces that reflects an actual model.

### M4 — GR00T adapter

`transport/gr00t_client.py` (pyzmq + msgpack, GR00T's
`{"endpoint": ..., "data": ...}` request shape), ZMQ REP echo fixture,
scheme `gr00t_zmq`, `get_modality_config()` driving handshake validation
(GR00T tells us the expected shape — use it instead of hand-writing the
mapping blind). After OpenPI because there's no ready-made GR00T×LIBERO
checkpoint synergy.

### M5 — deferred, explicitly out of scope for now

Managed launching of real policy servers (generalizing
`PolicyServiceSupervisor` beyond the debug server's CLI), GPU passthrough
for containerized serving, batched multi-env against real servers,
IsaacLab-Arena as a sim backend.

## Risks / open questions

- **Python 3.8 in the LIBERO container** constrains client deps:
  `websockets` new enough for `websockets.sync.client` (≥12) but old enough
  for 3.8 — pin and verify at M2. `pyzmq`/`msgpack`/`msgpack-numpy` are fine
  on 3.8.
- **Official client packages vs our own thin clients**: prefer
  `openpi-client` (standalone, lightweight) if it installs cleanly in the
  py3.8 container; otherwise our own ~100-line client against the verified
  wire shape. Never depend on the full `openpi`/`Isaac-GR00T` repos client-
  side. Decide at M2 with the container build as the test.
- **Where the GPU server actually runs** (lab workstation vs HPC node) is a
  deployment question deferred to M3 — external-server mode is what keeps
  it from being a design question.
