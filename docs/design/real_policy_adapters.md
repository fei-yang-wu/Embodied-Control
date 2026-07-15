# Plan: wiring in real VLA policies (OpenPI, GR00T)

Status: M1, M2, M3, and M4 all done — a real trained checkpoint (OpenPI's
public `pi05_libero`) has been run end-to-end through this repo's own
orchestrator against real LIBERO episodes, with a genuine 100% success rate
on 2 episodes (see M3 below for the full result and the real bugs that
running it for real surfaced). Builds on the research in
`docs/transport-comparison.md` (protocol details verified against OpenPI and
GR00T source, then validated against how StarVLA and Isaac Lab-Arena solved
the same problem). Last updated: 2026-07-15.

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

### M1 — extract the chunk scheduler + neutral observation (pure refactor) — DONE

`transport/chunking.py::ChunkScheduler` (buffer, refill-on-empty, per-request
horizons bookkeeping) replaced the duplicated logic in the stepped runner,
`fake_delegated_eval`, and `libero_eval`. Evaluators stopped calling
`encode_camera` themselves; `PolicyClient.act()` encodes now. HTTP wire bytes
are byte-identical to before. Verified: `pixi run test` (49 passed) +
`pixi run -e sim test-sim` (54 passed) green; both Docker images rebuilt,
`smoke-delegated-docker`/`smoke-libero`/`smoke-libero-image-stats` all pass;
a live client/server round-trip confirmed a real numpy camera array still
produces the same bright→+1.0/dark→−1.0 result as the unit tests.

### M2 — OpenPI adapter — DONE

Built: `transport/openpi_translate.py` (pure functions —
`OpenPIObservationMapping`, `observation_to_openpi`, `openpi_action_to_chunk`
— no network, independently testable), `transport/openpi_client.py`
(`OpenPIWebsocketClient`, websockets + msgpack-numpy, wire behavior verified
against OpenPI's actual `websocket_policy_server.py` and
`websocket_client_policy.py` source rather than assumed — notably `reset()`
sends nothing over the wire at all, and `/healthz` is a plain HTTP intercept
before the websocket upgrade, so `health()` never needs to open a session),
`transport/base.py::PolicyClientProtocol` (the three-implementation
threshold that justifies formalizing the shared interface), `tests/
fake_openpi_server.py` (a small fixture speaking the real wire protocol,
used by `tests/test_openpi_client.py`'s 9 tests — translation unit tests,
transport round-trip, error path, multi-env rejection, and a full
`run_eval()` integration test against a real external endpoint). Scheme
`openpi_websocket` added to `EndpointSpec`/`transport/factory.py`; a new
`transports` pixi feature (numpy + websockets + msgpack + msgpack-numpy)
hosts these tests without adding weight to the light default env.

External-server mode needed no new code — `PolicyBinding.endpoint` +
`RuntimeSpec.type="external"` already existed end-to-end in
`planner.py`/`supervisor.py` for the HTTP scheme; it now also works for
`openpi_websocket` for free.

Two things adjusted from the original sketch, both found while implementing
rather than assumed upfront:

- **A real bug caught along the way**: `fake_delegated_eval.py` and
  `libero_eval.py` hardcoded `PolicyClient` (HTTP) regardless of
  `policy_endpoint.scheme` — a delegated evaluator would have silently tried
  to speak HTTP to a websocket server. Fixed by routing both through
  `transport/factory.py::make_policy_client`, same as the stepped path
  already did. This was a real gap, not a hypothetical: M3 (real LIBERO ×
  OpenPI) would have hit it immediately.
- **Handshake validation is partial, not full, and that's deliberate**: the
  existing `action_dim` mismatch check in `runner.py` (`RunFailure` before
  any episode runs) now also covers `openpi_websocket`, since
  `EndpointSpec.action_dim` is asserted by the job author and echoed back
  through `describe()`. Full validation against the *server's own* declared
  shape (comparing `observation_mapping` against real metadata fields) isn't
  built — we don't know the real metadata schema without a real checkpoint,
  and guessing it would mean hardcoding unverified field names. Deferred to
  M3, where a real server's actual metadata is available to validate against.
- **Server metadata now persists into artifacts**: `RunManifest.policy_describe`
  (new field) captures whatever `describe()` returns — verified end-to-end
  against the real `ec-libero-eval` Docker image, not just the fixture.

**Accept when**: unit tests round-trip real msgpack frames against the
fixture — done (9/9 passing in `pixi run -e transports test-transports`);
LIBERO smoke runs against blank actions produce the full artifact contract
— done (`smoke-libero`, `smoke-libero-image-stats`, both rebuilt images,
`manifest.json.policy_describe` confirmed populated).

### M3 — real end-to-end: π0-LIBERO — DONE (2026-07-15)

Ran for real: NVIDIA Container Toolkit configured on this host (it has a
real RTX 5080, previously just not passed through to Docker — see
`docs/gotchas.md`), OpenPI's real `scripts/docker/serve_policy.Dockerfile`
built and run with `--env LIBERO --port 8000` (auto-downloaded the public
`pi05_libero` checkpoint from GCS, ~11.6GB, no training needed), then
`examples/libero_openpi_external.yaml` run through this repo's own
orchestrator end to end.

**Result**: `run_id=20260715_073801_libero_spatial_task0_openpi_pi05_0`,
2/2 episodes succeeded (`success_rate=1.0`) on task
`pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`,
episodes terminating early on genuine success (82 and 71 steps, not hitting
the 200-step cap — `env.check_success()` firing for real, not an artifact),
17 total real inference calls, `mean_action_horizon=10.0` both episodes
(the real checkpoint's chunk length, matching what was requested), full
artifact contract validated (`validate_run_dir` → `valid: True`, zero
errors/warnings). This is the first number this repo has produced that
reflects an actual trained model, not a blank policy.

**Three real, previously-undetectable bugs found and fixed by actually
running this, not by more source-reading**:

1. **msgpack envelope mismatch (silent, not a transport error)**. OpenPI's
   server implements its own ndarray wire format
   (`__ndarray__`/`dtype`/`shape`/`data` keys), not the generic
   `msgpack-numpy` PyPI package's convention (`nd`/`type`/`kind`/`shape`/
   `data`) that `transport/openpi_client.py` was built against. The
   mismatch doesn't fail at the transport layer — the server just doesn't
   recognize the envelope, silently reconstructs the array as a 0-d object
   array, and the *first* symptom appears deep inside OpenPI's own LIBERO
   input transform as an unrelated-looking `IndexError`. Every unit test
   passed throughout, because the test fixture used the same (wrong, but
   self-consistent) envelope as the client. Fixed with
   `transport/openpi_msgpack.py` — a faithful, attributed port of OpenPI's
   actual encode/decode functions, used by both the real client and the
   test fixture now. (Considered depending on the real `openpi-client` PyPI
   package instead per this doc's earlier "Risks" note — it exists, is
   small, and even ships the `image_tools.resize_with_pad` this repo is
   still missing — but its `numpy<2.0.0` pin has no conda-forge build for
   this project's Python 3.13 host environment; reimplementing ~30 lines
   was simpler than a second Python toolchain just for one package.)
2. **`observation/state` must be a real ndarray, not a Python list.**
   OpenPI's normalization transform calls `.shape` on it generically (not
   LIBERO-specific) — a plain list survives the wire fine but fails
   server-side with `AttributeError: 'list' object has no attribute
   'shape'`. Fixed in `openpi_translate.py` for both the LIBERO preset and
   the generic proprio path.
3. **The host process running `ec eval run` needs the same transport deps
   as the job's `policy.endpoint.scheme`, not just the delegated
   container.** `orchestration/runner.py` always builds a policy client
   host-side for the pre-flight `policy_describe` health/action-dim check
   (deliberately — it's what makes a misconfigured job fail fast before
   launching an expensive LIBERO container), so `ec eval run` on a job with
   `scheme: openpi_websocket` needs `websockets`/`msgpack` importable in
   *that* process too. Not a bug to fix in the runner (the pre-flight check
   is genuinely valuable) — documented as a real operational requirement:
   run `ec eval run` via `pixi run -e transports ec eval run ...` for any
   job using a real-policy endpoint scheme, not the light default env.

**Resolved, was flagged as open**: the exact composition of the 8-dim
`observation/state` (`concat(robot0_eef_pos, quat2axisangle(robot0_eef_quat),
robot0_gripper_qpos)`) was found in OpenPI's own reference LIBERO eval
script (`examples/libero/main.py`, not the dataset converter this doc
originally checked) and confirmed correct by the real success above — see
`openpi_translate.py::_libero_pi0_state`/`_quat2axisangle`. Also implemented
from that same script: the 180° image flip
("IMPORTANT: rotate 180 degrees to match train preprocessing" — their
comment) via `OpenPIObservationMapping.flip_images_180`.

**Known remaining simplification, not yet revisited**: images are rendered
natively at 224×224 (`sim/libero_eval.py`'s `camera_height`/`camera_width`)
rather than OpenPI's own eval pipeline's render-at-256-then-resize-with-pad-
to-224. Since both are square aspect ratios this is a resize-only
(no letterboxing) difference, and the real result above suggests it's
close enough to work, but it hasn't been A/B'd against the exact reference
preprocessing.

### M4 — GR00T adapter — DONE

Built the same way as M2: `transport/gr00t_translate.py` (pure functions),
`transport/gr00t_client.py` (`Gr00tZmqClient`, pyzmq + msgpack, wire
behavior verified against GR00T's actual `server_client.py` source —
`{"endpoint": ..., "data": ...}` request shape, `ping`/`get_action`/`reset`/
`get_modality_config` endpoints, and the REQ-socket-becomes-unusable-after-
a-timeout recovery behavior its own client has), `tests/fake_gr00t_server.py`
+ `tests/test_gr00t_client.py` (8 tests: translation, transport round-trip,
error path, multi-env rejection, full `run_eval()` integration). Scheme
`gr00t_zmq` added to `transport/factory.py` (now fully implemented — the
`PLANNED_SCHEMES` placeholder is empty).

Two scope decisions, made deliberately rather than accidentally:

- **`get_modality_config()`'s `ModalityConfig` objects are unwrapped into
  plain dicts, not reconstructed as real objects.** GR00T's own client
  decodes them via `gr00t.data.types.ModalityConfig`, which lives in the
  full (GPU-oriented, heavy) `gr00t` package — depending on that would
  defeat the point of a lightweight client adapter. Our client unwraps the
  same wire marker (`__ModalityConfig__`/`as_json`) into its JSON payload
  directly. Good enough to inspect/log/persist; not a drop-in for code
  written against GR00T's real `PolicyClient`.
- **Full auto-validation from `get_modality_config()` isn't built** — same
  caveat as M2's OpenPI adapter: `describe()` persists the real modality
  config into `manifest.json` for inspection, but doesn't yet cross-check
  it against `observation_mapping`. Worth doing once a real GR00T
  checkpoint's actual modality config shape is known (mirrors M3's
  reasoning, deferred for the same cause: nothing to validate against yet).

A real bug caught while building the fixture, not the client: ZMQ REQ
sockets that time out must be closed with `linger=0` before being dropped —
leaving one open (even just dereferencing it) can hang the whole process at
interpreter/context teardown waiting to flush an unacknowledged outbound
message. Caught by a test that hung under pytest, passed standalone; fixed
in `_call`'s exception handlers, documented inline since it's exactly the
kind of thing that's silent until it isn't.

**Accept when**: unit tests round-trip real msgpack frames against the
fixture — done (8/8 passing); scheme wired through the full orchestrator —
done (verified via a `run_eval()` integration test against a fake server,
same pattern as M2).

### M5 — deferred, explicitly out of scope for now

GPU passthrough for containerized serving, batched multi-env against real
servers, IsaacLab-Arena as a sim backend.

**Partially done, ahead of schedule**: "managed launching of real policy
servers" turned out to have a cheap, testable slice that didn't need to wait
for M3's GPU/checkpoint access — `RuntimeSpec.command` (with `{host}`/
`{port}` substitution) lets a job override the built-in debug-server launch
command entirely, so a real OpenPI/GR00T server's actual CLI (once known)
can be launched the same way the debug server is today, through the same
`PolicyServiceSupervisor`/`DockerRuntimeAdapter`/`LocalRuntimeAdapter`
machinery. Building the *Docker images* that wrap real serving scripts (with
real GPU deps, real checkpoints mounted) is still M3's job, not this — this
just means the orchestrator no longer needs new code to launch them once
that image exists. Caught and fixed a real, separate bug while doing this:
`orchestration/planner.py` and `orchestration/supervisor.py` used to build
the launch command independently in two places, and the one persisted into
`resolved_job.yaml` (the planner's copy) had silently drifted from the one
actually executed (the supervisor's copy was missing
`--max-action-horizon`). Consolidated into one function in `planner.py`;
`supervisor.py` now just launches `plan.policy_runtime.command` instead of
recomputing it.

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
