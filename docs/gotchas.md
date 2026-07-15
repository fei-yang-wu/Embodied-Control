# Gotchas: hard-won lessons

Specific bugs and environment traps discovered while building this repo,
kept so the same subsystem doesn't have to be re-debugged from scratch next
time. Each entry: what happened, why, and what was actually done about it.
Last updated: 2026-07-14.

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

**No GPU passthrough is configured for this deployment's Docker daemon** —
`docker run --gpus all ...` fails with `permission_denied` /
`no known GPU vendor found from CDI` (no NVIDIA Container Toolkit runtime
registered, only plain `runc`). This means EGL rendering is not an option
inside any container here, even though the *host* has a working GPU + EGL.
`ec-libero-eval` uses OSMesa (CPU software rendering) instead — slower, but
needs no daemon configuration, which makes it the portable default. If a
target machine ever gets the NVIDIA Container Toolkit configured, EGL
becomes possible again and would be faster, but nothing here currently
detects or uses it.

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
