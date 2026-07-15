# Gotchas: hard-won lessons

Specific bugs and environment traps discovered while building this repo,
kept so the same subsystem doesn't have to be re-debugged from scratch next
time. Each entry: what happened, why, and what was actually done about it.
Last updated: 2026-07-15.

## Real-policy transport (OpenPI / GR00T)

**A wire-protocol bug can pass every unit test and still be completely
broken against the real server, if the test fixture shares the same bug.**
`transport/openpi_client.py` originally packed observations with the
generic `msgpack-numpy` PyPI package's ndarray envelope
(`nd`/`type`/`kind`/`shape`/`data` keys). OpenPI's real server implements
its *own* ndarray encoding (`__ndarray__`/`dtype`/`shape`/`data` keys,
`packages/openpi-client/src/openpi_client/msgpack_numpy.py`) — a different
convention, not a bug on either side individually. The generic package's
envelope arrives at OpenPI's server as an unrecognized plain dict; their
server doesn't error, it just doesn't reconstruct an array, and
`np.asarray()` on that dict later produces a 0-d object array — the first
visible symptom is an unrelated-looking `IndexError`/`AttributeError` deep
inside OpenPI's own input transform, not a transport-layer error. This was
invisible to `pixi run -e transports test-transports` because
`tests/fake_openpi_server.py` used the *same* generic package, so client and
fixture were self-consistently wrong together. Only caught by running
against a real, live OpenPI server. Fixed by `transport/openpi_msgpack.py`
— a faithful, attributed port of OpenPI's actual encode/decode functions,
used by both the real client and the test fixture now, so they can no
longer silently agree on the wrong thing. **Lesson**: a test fixture that
you also wrote is not independent verification of wire compatibility with a
*third party's* real implementation — it only proves self-consistency.

**Corollary, caught the same way for GR00T**: this doc originally claimed
GR00T's real server was *not* affected by the risk above, because its
`MsgSerializer` "calls the generic package's own `encode`/`decode`
functions directly" — that was true of `NVIDIA/Isaac-GR00T`'s `main` branch
as read via `gh api` during initial research, but the *actual,
locally-running* server used for real verification (a checkout pinned to an
April 2026 commit, already older than `main` by the time it was researched)
turned out to use a **third, different** envelope entirely:
`{"__ndarray_class__": True, "as_npy": <np.save bytes>}`. Same failure
shape as the OpenPI bug (`"Video key 'image' must be a numpy array. Got
<class 'dict'>"`, not a transport error), fixed the same way
(`transport/gr00t_msgpack.py`, a faithful port of the *actually running*
server's real `MsgSerializer`). **The lesson generalizes past "verify
against source"**: two clones of "the same" upstream project, or a local
checkout vs. that project's current `main`, can genuinely disagree — verify
wire compatibility against the specific server version actually being
talked to, not against whichever commit a research pass happened to land
on, even when that research quoted real source correctly at the time.

**GR00T's server rejects a flat `video.image`/`state.x`-style observation
by default — `--use-sim-policy-wrapper` is required, not optional, for the
shape this repo's LIBERO evaluator (and GR00T's own reference LIBERO env
wrapper) produces.** Without that flag, `Gr00tPolicy.check_observation`
expects a *nested* `{"video": {"image": ...}, "state": {"x": ...}}`
structure and every request fails immediately with `"Observation must
contain a 'video' key"`. `Gr00tSimPolicyWrapper` (enabled by the flag)
accepts the flat keys and reshapes them internally.

**Even with the sim policy wrapper enabled, every array needs explicit
batch and time dimensions, not just the right key names.**
`Gr00tSimPolicyWrapper.check_observation` asserts `ndim==3`, shape
`(B=1, T=1, D)`, dtype `float32` for every `state.*` key, and `ndim==5`,
shape `(B=1, T=1, H, W, 3)`, dtype `uint8` for every `video.*` key — a bare
`(H, W, 3)` image or an unbatched Python list of floats fails with a clear
`AssertionError`, not a silent wrong-shape bug (better than OpenPI's
silent-corruption failure mode, but still only discoverable by actually
running a request through). Fixed in `transport/gr00t_translate.py`
(`_batch_state`/`_batch_video` on the way in, `_unbatch` — a flat reshape,
not an assumed exact shape, since the wrapper's own docstring only commits
to `ndim==3` — on the way out).

**A real OpenPI server's normalization pipeline requires `observation/state`
to be a numpy ndarray, not a plain Python list** — `AttributeError: 'list'
object has no attribute 'shape'` inside their own `_normalize_quantile`
transform otherwise. msgpack-numpy only special-cases actual `np.ndarray`/
`np.generic` objects over the wire; a Python list survives the round-trip as
a list. Fixed in `transport/openpi_translate.py` (`_libero_pi0_state` and
the generic proprio path both now build `np.asarray(..., dtype=np.float32)`).

**GR00T's `run_gr00t_server` defaults to binding all network interfaces
(`host="*"`), not localhost.** Launching it without an explicit `--host
127.0.0.1` exposes an unauthenticated ZeroMQ inference listener to the
whole network, not just this machine — always pass `--host 127.0.0.1`
explicitly for local development/testing unless remote access is
genuinely intended and access-controlled some other way. Caught before
this mattered: the harness's own auto-mode classifier flagged the
default-bind launch command before it ran.

**`ec eval run` needs the same transport dependencies as the job's
`policy.endpoint.scheme`, in the process running the CLI itself — not just
inside a delegated container.** `orchestration/runner.py` always builds a
policy client host-side for the pre-flight `policy_describe` health/
action-dim check, deliberately (it's what makes a misconfigured job fail
fast before launching an expensive container) — but that means running
`ec eval run` on a job with `scheme: openpi_websocket` from the light
default pixi env fails with `ModuleNotFoundError: No module named
'websockets'`, even though the actual inference happens entirely inside a
separate LIBERO container that has the right deps. Not a bug to fix (the
pre-flight check is genuinely valuable) — use
`pixi run -e transports ec eval run <job>.yaml` for any job with a
real-policy endpoint scheme.

## Rendering / OpenGL

**MuJoCo's GL backend is chosen at the *first* `import mujoco` anywhere in
the process, and that choice is permanent for the life of the process.**
Setting `os.environ["MUJOCO_GL"]` later — even before your own code's first
render call, just not before mujoco's first import — does nothing. This bit
twice:

- On the host, `ec doctor`'s first check imported `mujoco` to report its
  version, *before* a later check tried to set the GL backend and test
  rendering — the later check silently used the wrong (default GLFW,
  display-requiring) backend. Fixed by moving the GL-backend default to
  `sim/mujoco_backend.py` module level (runs on that module's *own* first
  import) and calling it explicitly at the very top of `ec doctor`, before
  any other code path might import mujoco first.
- The same issue hit a test file that did `pytest.importorskip("mujoco")` —
  that line itself is a raw `import mujoco`, executed at collection time,
  before any of our own code runs.
- **Inside a container this whole class of bug doesn't apply**: a Dockerfile
  `ENV MUJOCO_GL=osmesa` is set before the *process* even starts, so there's
  no import-order race to lose. This is why `containers/libero_eval/Dockerfile`
  uses `ENV`, not a Python-level `os.environ` assignment in the harness.

**GPU passthrough was not configured for this deployment's Docker daemon
until 2026-07-15** — `docker run --gpus all ...` used to fail with
`permission_denied` / `no known GPU vendor found from CDI` (no NVIDIA
Container Toolkit runtime registered, only plain `runc`), even though the
*host* always had a working GPU (an RTX 5080) + EGL. Fixed by installing
`nvidia-container-toolkit` (`sudo apt-get install nvidia-container-toolkit`,
`sudo nvidia-ctk runtime configure --runtime=docker`,
`sudo systemctl restart docker` — see NVIDIA's install guide for exact
current commands) and verified with `docker run --rm --gpus all
nvidia/cuda:12.2.2-base-ubuntu22.04 nvidia-smi`. This is what made M3 (a
real GPU-served OpenPI checkpoint) possible — see
`docs/design/real_policy_adapters.md`.

`ec-libero-eval` still uses OSMesa (CPU software rendering) for the LIBERO
*simulator* itself, unrelated to this fix — LIBERO's own rendering has not
been switched to EGL/GPU, only the separate real-policy *server* container
(OpenPI's, run independently, not part of this repo's own images) uses the
GPU now. Nothing in `sim/libero_eval.py`/`containers/libero_eval/` currently
detects or uses GPU rendering; OSMesa remains the portable default there.

**OSMesa itself failed once with a confusing `PyOpenGL` error** (
`AttributeError: 'NoneType' object has no attribute 'glGetError'`) even
though `libOSMesa.so` was verifiably present and loadable via `ctypes.CDLL`
directly. The actual cause: `MUJOCO_GL=osmesa` was set via
`os.environ[...] = ...` inside the same Python process, *after* some other
import path had already triggered mujoco's GL backend selection (see above).
Setting it via `docker run -e MUJOCO_GL=osmesa` (a real process-level env var
present from PID 1) fixed it immediately — same underlying lesson as the
EGL case, just a more confusing symptom.

## LIBERO container build

**`libero.libero`'s package `__init__.py` runs an interactive `input()`
prompt** ("Do you want to specify a custom path for the dataset folder?
(Y/N)") on the *first ever* import, if `~/.libero/config.yaml` doesn't
already exist. In a non-interactive container with no stdin attached, this
would hang forever. Fixed by triggering that first import at **build** time,
piping in an answer: `RUN echo "n" | python3 -c "import libero.libero"` —
this writes the config (using the package's own computed default paths, not
hand-guessed ones) once, permanently, so it never prompts again at run time.

**`robomimic` (a LIBERO dependency) pulls in `egl_probe`, whose native
extension needs `cmake` to build** — not obvious from `requirements.txt`
alone (which just lists `robomimic==0.2.0`); the actual build failure only
surfaces partway through a multi-minute `pip install -r requirements.txt`.
Fixed by adding `cmake` to the Dockerfile's `apt-get install` list.

**`opencv-python==4.6.0.66` (LIBERO's pin — the full GUI-capable build, not
`opencv-python-headless`) needs `libgthread-2.0.so.0` and a few other
glib/X11 shared libraries at *import* time**, even though nothing in this
container ever opens a window. Fixed by adding
`libglib2.0-0 libsm6 libxext6 libxrender1` to the Dockerfile's apt packages.

**Unquoted `>=`/`<` in a Dockerfile `RUN pip install` line are shell
redirection operators, not version specifiers.**
`pip install imageio>=2.15,<3` silently breaks (bash tries to redirect stdin
from a file named `3`, tries to redirect stdout to `2.15,`). Always quote:
`pip install "imageio>=2.15,<3"`.

**Docker layer caching invalidates a `RUN pip install ...` layer (and
everything after it) if that instruction's text changes at all** — adding
one new package to an existing `pip install` line reruns the *entire*
install, including packages that didn't change (~95s for LIBERO's dependency
chain, every time). The mitigation actually used: order Dockerfile `COPY`
steps for *our own* frequently-edited source files **after** the expensive
`pip install`, so editing `libero_eval.py` and rebuilding only reruns the
cheap COPY + export steps (a few seconds), not the whole dependency install.

## Process lifecycle

**`http.server.BaseServer.shutdown()` deadlocks if called from the same
thread that's running `serve_forever()`** (this is documented stdlib
behavior, easy to miss). The original SIGTERM handler in
`policies/debug_server.py` called `server.shutdown()` directly from the
signal handler — which runs in the same thread as `serve_forever()` — so
every local eval hung for ~10s on teardown and then got hard-killed. Fixed
by running `serve_forever()` in a background thread and shutting down from
the (signal-handling) main thread instead. Local eval teardown went from
~10s+SIGKILL to <1.4s clean exit.

**`ExecutionPlan.run_id`'s default template only has second-level
granularity** (`{date}_{name}_{seed}`, date formatted to the second). Two
runs of the same job name/seed started within the same wall-clock second
collide on `run_dir` *and* on the logger's root name (`EcLogger` is keyed by
`run_id`) — silently overwriting the first run's artifacts. Fixed with a
dedup-with-counter-suffix scheme in `orchestration/planner.py`
(`run_id-1`, `run_id-2`, ... if the directory already exists).

## Dependencies leaking into "stdlib-only" containers

**`LogConfig` was originally a pydantic `BaseModel`.** That was fine until
`transport/client.py` (which every delegated evaluator container needs, to
talk to the policy service) started depending on `EcLogger`, which depended
on `LogConfig` — silently pulling pydantic into what were supposed to be
tiny, dependency-free containers (`fake_delegated_eval`, and it would have
broken `libero_eval` too if not caught). Fixed by converting `LogConfig` to
a plain `@dataclass` — same construction API, zero pydantic dependency. The
lesson generalizes: anything imported by `transport/` or `policies/` can end
up inside a minimal container, so it needs to stay stdlib-only transitively,
not just at the file you're editing.

## GHCR / registry

**`docker login ghcr.io` can succeed with a token that still can't actually
push.** Login only validates the credentials are real; it doesn't check
package-specific scope. The `gh` CLI's default OAuth token has `repo`,
`workflow`, etc. but not `write:packages` — pushing fails with
`permission_denied: The token provided does not match expected scopes` only
at the actual `docker push` step, after login already reported success.
Fix: `gh auth refresh -h github.com -s write:packages` (interactive
device-code flow — only a human can complete this, an agent can't do it
unattended), then re-run `docker login` with the refreshed token.

**Adding a new module under `transport/`/`sim/`/`logging/` doesn't make it
appear in a container just because a file already inside that container
imports it.** Extracting `ChunkScheduler` into `transport/chunking.py` and
importing it from `sim/fake_delegated_eval.py` / `sim/libero_eval.py` built
locally fine (host has the whole `src/` tree on `PYTHONPATH`) but would have
failed at container *runtime* with `ModuleNotFoundError` — each
`containers/*/Dockerfile` lists its `COPY` lines file-by-file (deliberately,
to keep images minimal), so a new source file needs its own `COPY` line
added by hand in every Dockerfile that imports it. Caught before a rebuild
by grepping the Dockerfile's `COPY` list while making the edit, not by
running the container and hitting the import error.

## Miscellaneous

**A local delegated-mode `runtime.type: local` branch once hardcoded the
`fake_delegated_eval` module regardless of `sim.backend`** — a real bug
(silent wrong-evaluator execution for any future backend using local mode)
that existed for a while before being caught, incidentally, while adding
video support and reading through that code path again. Nothing tested this
directly; the fix added a backend→module dispatch table
(`orchestration/delegated.py::_LOCAL_MODULES`) instead of a bare hardcoded
string. Worth remembering: a hardcoded "just this one case" shortcut in
otherwise-generic orchestration code is exactly the kind of thing that
survives silently until a second real case exists.

**Docker containers write files as root by default.** Video files LIBERO
writes under a host-mounted `/videos` directory end up owned by `root` on
the host side — readable by the host user (default `-rw-r--r--`), but not
writable/deletable without `sudo`. Not fixed, just a known wrinkle; would
need `--user $(id -u):$(id -g)` on the `docker run` if it ever becomes
annoying enough to matter (e.g. for `ec eval gc`-style cleanup tooling).
