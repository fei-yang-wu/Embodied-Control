# Policy transport comparison: what to adopt for real VLA policies

Status: research/recommendation, nothing implemented yet.
Last updated: 2026-07-14.

## The question

Everything built so far (stepped MuJoCo, delegated `fake_delegated`/LIBERO,
camera observations) talks to policies over our own `ec.policy` HTTP/JSON
transport (`transport/client.py`, `transport/server.py`). That transport is
only ever used by *our own* debug policies (`zero`/`random`/`image_stats`),
which we wrote, so we could pick anything. Real VLA policies are different:
**OpenPI and GR00T each already ship a server speaking their own protocol.**
Wiring one in is a client-adapter problem, not a protocol-design problem —
the question is which transport(s) to speak, not whether to invent a new one.

## The four candidates

### A. Our own `ec.policy` (HTTP + JSON, current)

- **Wire**: plain HTTP, JSON bodies, images base64-encoded inside the JSON
  (`transport/protocol.py::encode_camera`).
- **Who speaks it**: nobody but us. `policies/base.py`'s debug policies and
  the tiny stdlib-only `ec-policy-debug` container.
- **Dependencies**: stdlib only (`urllib`, `http.server`, `json`, `base64`) —
  this is *why* the debug-policy container is 176MB instead of gigabytes.
- **Message size**: no hard limit (HTTP has none by default); base64 adds
  ~33% overhead versus raw bytes, which was an accepted, deliberate tradeoff
  at 128×128 (see `docs/gotchas.md`) but would need revisiting at higher
  resolution or frame rate.
- **Verdict**: correct choice for what it's for (our own zero/random/
  image_stats debug policies), wrong choice to force onto OpenPI/GR00T —
  neither speaks it, so using it as the *only* supported scheme would mean
  writing a translation proxy in front of *their* servers, which is more
  work than just speaking their native protocols directly.

### B. OpenPI native: WebSocket + msgpack-numpy

Verified against `packages/openpi-client/src/openpi_client/websocket_client_policy.py`
(Physical-Intelligence/openpi, 2026-07-14):

- **Wire**: `websockets.sync.client`, URI `ws://{host}:{port}`, optional
  `Authorization: Api-Key {key}` header.
- **Serialization**: `msgpack_numpy` — packs NumPy arrays directly (dtype +
  shape + raw bytes), no base64 detour.
- **Connection model**: persistent connection; on connect, the server sends
  metadata first (`msgpack_numpy.unpackb(conn.recv())`), then the client
  calls `infer(obs: dict) -> dict` repeatedly. `reset()` and
  `get_server_metadata()` are also exposed.
- **Retry behavior**: the client itself retries the initial connection every
  5s if the server isn't up yet — similar spirit to our
  `PolicyClient.wait_healthy()`, built in rather than left to the caller.
- **Maps onto our contract almost directly**: `get_server_metadata()` ≈
  `describe()`, `infer()` ≈ `act()`, `reset()` ≈ `reset()`. No explicit
  `health()` — connecting (which the client already retries) is the
  closest analog; an adapter would treat "connected + got metadata" as
  healthy.
- **Dependencies**: `websockets`, `msgpack`, `msgpack-numpy`. All pure
  Python, no native extensions — cheap to add to a client-side container.

### C. GR00T native: ZeroMQ (REQ/REP) + msgpack + msgpack-numpy

Verified against `gr00t/policy/server_client.py` and
`gr00t/eval/run_gr00t_server.py` (NVIDIA/Isaac-GR00T, 2026-07-14):

- **Wire**: raw ZeroMQ, `zmq.REP` server / `zmq.REQ` client,
  `tcp://{host}:{port}`, default port **5555**.
- **Serialization**: msgpack, with a custom `MsgSerializer` layered on top of
  `msgpack_numpy` — notably it **hardens against pickle-based deserialization
  of object-dtype ndarrays** (refuses them on both encode and decode with an
  explicit `TypeError`/`ValueError`), i.e. GR00T's own maintainers have
  already treated "arbitrary pickle over the wire" as a real risk worth
  closing. Worth mirroring if we ever accept arrays from a source we don't
  fully trust.
- **Connection model**: request/reply, one request in flight at a time per
  socket (ZMQ REQ/REP is strictly synchronous — no pipelining without extra
  sockets). Request shape: `{"endpoint": name, "data": {...}, "api_token":
  ...}`; response is whatever the handler returns, or `{"error": ...}`.
- **Endpoints registered by default**: `ping` (health), `get_action` (act),
  `reset` (reset), `get_modality_config` (describe). **This maps onto our
  `health/act/reset/describe` split almost one-to-one** — a genuinely close
  match, not a coincidence worth ignoring when designing an adapter.
- **Dependencies**: `pyzmq`, `msgpack`, `msgpack-numpy`. `pyzmq` has a native
  extension (bundles libzmq) but is a very standard, widely-available wheel —
  not in the same league of pain as LIBERO's dependency chain.

### D. gRPC/protobuf (the original design doc's proposed `ec.policy` v1)

- **Wire**: HTTP/2, protobuf schemas, codegen required (`protoc`).
- **Who speaks it**: nobody we actually need to talk to. Neither OpenPI nor
  GR00T uses gRPC.
- **Known problems** (already flagged in the earlier design review of this
  repo): gRPC's **default max message size is 4MB**, which a multi-camera,
  higher-resolution batch would blow through without explicit
  reconfiguration (channel options on both client and server) — an easy
  silent failure mode if adopted without deliberately setting it.
- **Verdict**: solves a problem we don't have (schema evolution across
  languages/teams at scale) at a real cost (codegen build step, a protocol
  neither target policy speaks, a message-size footgun) with no offsetting
  benefit right now. Defer indefinitely unless a *third* policy shows up
  that speaks gRPC and nothing else.

## Recommendation

**Don't unify on one wire protocol. Add native client adapters for OpenPI
and GR00T, in addition to (not instead of) our own `ec.policy` HTTP/JSON.**

This is exactly the design doc's original D5 ("native pass-through... do not
make ec.policy a blocker") and what the earlier design review of this repo
already concluded — now confirmed concrete by reading both real protocols
end to end rather than assuming: both map cleanly onto the same
`health/describe/reset/act` shape our `PolicyClient` already exposes, so this
is client-adapter work behind an existing interface, not a redesign.

**Concrete shape**, when this work starts:

1. `transport/openpi_client.py` — a class with the same surface as
   `transport/client.py::PolicyClient` (`health`/`wait_healthy`/`describe`/
   `reset`/`act`), backed by `websockets` + `msgpack-numpy` instead of
   `urllib` + `json`.
2. `transport/gr00t_client.py` — same surface, backed by `pyzmq` +
   `msgpack`/`msgpack-numpy`, matching GR00T's `{"endpoint": ..., "data":
   ...}` request shape.
3. `transport/factory.py::make_policy_client` already dispatches on
   `endpoint.scheme` with `SUPPORTED_SCHEMES`/`PLANNED_SCHEMES` as an
   explicit extension point — add `openpi_websocket` and `gr00t_zmq` as real
   scheme names there (the current placeholders, generic `"grpc"`/`"zmq"`,
   should be renamed/replaced to match — `"zmq"` alone is ambiguous between
   GR00T's specific request shape and a hypothetical future one).
4. Camera observations already exist on our wire (`cameras[]`,
   `transport/protocol.py::encode_camera`) — an adapter's `act()` just needs
   to translate that shape into whichever `obs` dict shape OpenPI/GR00T's
   `infer`/`get_action` expects (their own conventions, e.g. flat keys like
   `observation/image`, `observation/state` for OpenPI-trained policies) —
   this is real per-policy mapping work, not a protocol problem.
5. ~~Neither adapter needs `image_stats`-style proof scaffolding~~ —
   partially superseded after the StarVLA/Arena review (below): no proof
   *policy* is needed, but each adapter does want a tiny **fake server test
   fixture** speaking the real wire shape (~60 lines: websocket+msgpack echo
   for OpenPI, ZMQ REP echo for GR00T), so the transport layer is covered by
   `pixi run test` without needing a GPU or a checkpoint. A real server being
   "reachable and answering" is still the only proof of the *model*, but the
   *client code* shouldn't be untestable until a checkpoint shows up.

**What this does *not* require**: no change to the delegated architecture,
no change to how LIBERO/MuJoCo produce observations, no gRPC, no change to
the artifact contract. It's additive.

## Validation against StarVLA and Isaac Lab-Arena (2026-07-14)

After the recommendation above was written, two independent VLA-eval
codebases were reviewed to check it against how others actually solved the
same problem:

- **StarVLA** (`github.com/starVLA/starVLA`) — websocket + msgpack-numpy
  policy server (`deployment/model_server/tools/websocket_policy_server.py`),
  per-benchmark thin client adapters
  (`examples/simBenchmarks/<Sim>/eval_files/model2<sim>_interface.py`),
  separate conda envs / a slim Docker server image for dependency isolation.
- **Isaac Lab-Arena** (`github.com/isaac-sim/IsaacLab-Arena`) — abstract
  `PolicyBase` with per-family remote clients (OpenPI websocket+msgpack,
  GR00T via its own ZMQ client, DreamZero websocket+msgpack), one Docker
  image per policy family, and a centralized
  `action_chunk_scheduler.py` that owns chunk replay for every backend.

Both independently landed on the same core recommendation as above:
**per-model-family native transports behind one shared policy interface,
with per-family containers for dependency isolation** — nobody forces a
single unified wire protocol onto foreign policy servers. Confirmations and
adjustments this review produced:

1. **Images cross the wire as raw numpy bytes inside msgpack**
   (`{data: tobytes(), dtype, shape}`), never base64 — both projects, both
   directions. Confirms the note under candidate A: base64/JSON stays
   correct for our own debug transport, but the real-policy adapters must
   hand arrays to msgpack directly, with no base64 detour in between.
2. **Chunk length is server-owned and advertised at handshake** — StarVLA
   sends `action_chunk_size` in its connect metadata (computed from the
   checkpoint's training config, not requestable per-call); clients cache a
   chunk and only re-request when it's exhausted. Confirms
   `requested_horizon` must become advisory for real-policy adapters.
3. **Chunk *scheduling* deserves one shared home** — Arena centralizes
   consume-from-chunk/refill-when-empty logic in a single scheduler used by
   every backend, instead of re-implementing it per adapter. Ours is
   currently duplicated three ways (`orchestration/runner.py`,
   `sim/fake_delegated_eval.py`, `sim/libero_eval.py`) — extracting it is
   now part of the adopted plan.
4. **msgpack-over-pickle for security is unanimous** — StarVLA explicitly
   rejected pickle (arbitrary code execution) in favor of msgpack+numpy;
   GR00T's serializer hardens against object-dtype ndarrays for the same
   reason. Third independent confirmation of the same conclusion.

Where we deliberately go further than either (see the adopted plan for
details): handshake-time contract validation as a hard failure instead of
StarVLA's warnings, persisting server-advertised metadata into run
artifacts, and config-parameterized translation instead of StarVLA's
N-sims × M-models grid of per-benchmark interface files.

**Adopted plan**: `docs/design/real_policy_adapters.md` — milestones,
acceptance criteria, and the concrete component design that came out of
this comparison.
