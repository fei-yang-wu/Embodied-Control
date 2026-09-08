"""Real LIBERO delegated evaluator.

Entry point for the delegated sim runtime (``python -m
embodied_control.sim.libero_eval``). Drives real LIBERO/robosuite manipulation
episodes, calling the policy service over whichever transport the resolved
plan's endpoint declares (``transport.factory.make_policy_client`` --
``policy_scheme`` in the generated sim config, defaulting to ``http`` for the
blank debug policies) -- LIBERO is a black-box evaluator (design D2/6.5): it
owns its own reset/step loop end-to-end, the host only launches it, waits for
exit, and normalizes its raw per-episode output.

Camera observations (``agentview_image``, ``robot0_eye_in_hand_image``) ARE
sent to the policy, in a ``cameras[]`` field alongside proprioception -- this
module builds a *neutral* observation (raw arrays, not wire-encoded); the
transport client owns turning that into whatever bytes-on-the-wire form it
uses (HTTP base64-encodes; ``openpi_websocket`` packs the same arrays
directly via msgpack-numpy, no base64 detour). At 128x128x3,
base64-over-HTTP is ~65KB/frame, trivial over plain HTTP/JSON, so there is no
compression/format negotiation here yet. The blank zero/random policies still
never look at any of it; ``image_stats`` (a policy that reads and decodes real
bytes from the observation) is what actually proves the round trip carries
usable data end to end, ahead of wiring in a real vision policy.

This harness intentionally does NOT do GPU/EGL rendering: no NVIDIA Container
Toolkit is configured for this deployment's Docker daemon, so the container
has no GPU device at all; MUJOCO_GL=osmesa (CPU software rendering, set in
the Dockerfile ENV, before any process starts) is the only rendering path
that reliably works here. It's slower than EGL but requires no host/daemon
configuration -- the more portable default until GPU passthrough is set up on
a target machine.

Determinism: LIBERO ships a fixed array of pre-generated initial states per
task (``benchmark.get_task_init_states``); the job's per-episode seed selects
which of those to replay (``seed % len(init_states)``) rather than seeding a
fresh randomization the harness can't otherwise control -- this is what makes
"same seed -> same episode" a meaningful claim for a LIBERO task (matches the
design's own suggested determinism mechanism, §Milestone 2).
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import unquote, urlparse

from embodied_control.transport.base import PolicyClientProtocol
from embodied_control.transport.chunking import ChunkScheduler
from embodied_control.transport.client import PolicyClientError
from embodied_control.transport.factory import make_policy_client

_IMAGE_KEY_SUFFIXES = ("_image", "_depth", "_segmentation")
_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
_VIDEO_CAMERA_KEY = "agentview_image"

_LIVE_VIEW_HTML = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Embodied-Control | LIBERO</title>
  <style>
    :root { color-scheme: dark; font-family: ui-monospace, monospace; }
    body { margin: 0; background: #0b0f14; color: #d7e0ea; }
    header { padding: 18px 22px; border-bottom: 1px solid #25303c; }
    h1 { margin: 0 0 8px; font-size: 18px; }
    #task { color: #9fb0c2; }
    main { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; padding: 16px; }
    section { overflow: hidden; border: 1px solid #25303c; border-radius: 10px; background: #111821; }
    h2 { margin: 0; padding: 10px 12px; font-size: 13px; color: #9fb0c2; }
    img { display: block; width: 100%; aspect-ratio: 1; object-fit: contain; background: #050709; }
    footer { padding: 0 18px 18px; color: #7f91a3; }
  </style>
</head>
<body>
  <header><h1>Embodied-Control / LIBERO live view</h1><div id="task">Waiting for simulation...</div></header>
  <main>
    <section><h2>Agent view</h2><img src="/stream/agentview_image.mjpg"></section>
    <section><h2>Wrist view</h2><img src="/stream/robot0_eye_in_hand_image.mjpg"></section>
  </main>
  <footer id="status">starting</footer>
  <script>
    async function refresh() {
      try {
        const s = await fetch('/status.json', {cache: 'no-store'}).then(r => r.json());
        document.getElementById('task').textContent = s.language || s.task || 'Waiting for simulation...';
        const bits = [s.phase, s.episode == null ? null : `episode ${s.episode}`,
          s.step == null ? null : `step ${s.step}`, s.success == null ? null : `success ${s.success}`];
        document.getElementById('status').textContent = bits.filter(Boolean).join(' | ');
      } catch (_) {}
    }
    refresh(); setInterval(refresh, 250);
  </script>
</body>
</html>
"""


class _LiveViewHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class LiberoLiveView:
    def __init__(self, host: str = "127.0.0.1", port: int = 8766, fps: int = 10):
        self.host = host
        self.port = int(port)
        self.fps = max(1, int(fps))
        self._condition = threading.Condition()
        self._frames: dict[str, bytes] = {}
        self._status: dict = {"phase": "starting"}
        self._sequence = 0
        self._closed = False
        self._next_frame_at = 0.0
        self._server: _LiveViewHttpServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/"

    def start(self) -> str:
        self._server = _LiveViewHttpServer((self.host, self.port), _make_live_view_handler(self))
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="libero-live-view", daemon=True
        )
        self._thread.start()
        return self.url

    def update_status(self, **status) -> None:
        with self._condition:
            self._status.update(status)

    def publish(self, obs: dict, *, force: bool = False, **status) -> None:
        now = time.monotonic()
        if not force and now < self._next_frame_at:
            self.update_status(**status)
            return
        self._next_frame_at = now + 1.0 / self.fps

        import cv2

        frames = {}
        for camera in _CAMERA_KEYS:
            image = obs.get(camera)
            if image is None:
                continue
            bgr = image[::-1, :, ::-1].copy()
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                frames[camera] = encoded.tobytes()
        self._publish_encoded(frames, status)

    def _publish_encoded(self, frames: dict[str, bytes], status: dict) -> None:
        with self._condition:
            self._frames.update(frames)
            self._status.update(status)
            self._sequence += 1
            self._condition.notify_all()

    def snapshot(self, camera: str) -> bytes | None:
        with self._condition:
            return self._frames.get(camera)

    def status(self) -> dict:
        with self._condition:
            return {
                **self._status,
                "cameras": sorted(self._frames),
                "viewer_url": self.url,
            }

    def wait_for_frame(self, camera: str, after: int, timeout_s: float = 1.0):
        with self._condition:
            changed = self._condition.wait_for(
                lambda: self._closed or self._sequence != after, timeout=timeout_s
            )
            if self._closed:
                return self._sequence, None, True
            if not changed:
                return self._sequence, None, False
            return self._sequence, self._frames.get(camera), False

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def _make_live_view_handler(view: LiberoLiveView):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self._send_bytes(200, "text/html; charset=utf-8", _LIVE_VIEW_HTML)
                return
            if path == "/status.json":
                payload = json.dumps(view.status(), ensure_ascii=False).encode()
                self._send_bytes(200, "application/json", payload)
                return
            if path.startswith("/snapshot/") and path.endswith(".jpg"):
                camera = path[len("/snapshot/"):-len(".jpg")]
                frame = view.snapshot(camera)
                if frame is None:
                    self._send_bytes(404, "text/plain", b"camera frame not available\n")
                else:
                    self._send_bytes(200, "image/jpeg", frame)
                return
            if path.startswith("/stream/") and path.endswith(".mjpg"):
                camera = path[len("/stream/"):-len(".mjpg")]
                if camera not in _CAMERA_KEYS:
                    self._send_bytes(404, "text/plain", b"unknown camera\n")
                    return
                self._stream(camera)
                return
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self._send_bytes(404, "text/plain", b"not found\n")

        def _send_bytes(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _stream(self, camera: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            while True:
                sequence, frame, closed = view.wait_for_frame(camera, sequence)
                if closed:
                    return
                if frame is None:
                    continue
                try:
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

        def log_message(self, _format: str, *args) -> None:
            return

    return Handler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real LIBERO delegated evaluator")
    p.add_argument("--config", required=True, help="path to the host-generated sim config JSON")
    p.add_argument("--output", required=True, help="directory to write per-episode raw JSON records")
    p.add_argument("--videos-dir", default=None,
                    help="directory to write per-episode mp4s (only used if the config enables it)")
    return p


def _make_env(suite_name: str, task_index: int, camera_height: int, camera_width: int):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_instance = benchmark.get_benchmark_dict()[suite_name]()
    task = benchmark_instance.get_task(task_index)
    bddl_file = benchmark_instance.get_task_bddl_file_path(task_index)
    init_states = benchmark_instance.get_task_init_states(task_index)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_heights=camera_height,
        camera_widths=camera_width,
    )
    return env, task, init_states


def _wire_observation(obs: dict, env_id: int, episode_id: int, language: str, task_id: str) -> dict:
    """Proprioception (flattened non-image fields) plus cameras -- a *neutral*
    observation carrying raw arrays; the transport client owns wire encoding
    (see module docstring)."""
    names: list[str] = []
    values: list[float] = []
    for key, val in obs.items():
        if key.endswith(_IMAGE_KEY_SUFFIXES):
            continue
        arr = val.ravel() if hasattr(val, "ravel") else [val]
        multi = len(arr) > 1
        for i, v in enumerate(arr):
            names.append(f"{key}[{i}]" if multi else key)
            values.append(float(v))
    cameras = [{"name": key, "array": obs[key]} for key in _CAMERA_KEYS if key in obs]
    return {
        "env_id": env_id,
        "episode_id": episode_id,
        "proprio": {"names": names, "values": values},
        "cameras": cameras,
        "task": {"task_id": task_id, "language_instruction": language},
    }


def _capture_frame(obs: dict, frames: list) -> None:
    """Append the agentview camera frame, best-effort (never raises).

    robosuite/MuJoCo render camera images upside down relative to a normal
    top-left-origin image -- ``[::-1]`` corrects that (confirmed against
    LIBERO's own ``benchmark_scripts/render_single_task.py``, which does the
    same flip before writing a PNG). imageio wants RGB, which is what
    robosuite already returns, so unlike that script's cv2 usage there is no
    channel-order swap to undo here.
    """
    image = obs.get(_VIDEO_CAMERA_KEY)
    if image is not None:
        frames.append(image[::-1])


def run_episode(client: PolicyClientProtocol, env, task, init_states, episode_id: int, seed: int,
                config: dict, videos_dir: Path | None,
                live_view: LiberoLiveView | None = None) -> dict:
    key = [0, episode_id]
    client.reset([key], seed)

    horizon = max(1, int(config.get("requested_action_horizon", 1)))
    max_steps = int(config["max_steps_per_episode"])

    env.reset()
    init_state = init_states[seed % len(init_states)]
    obs = env.set_init_state(init_state)

    frames: list = []
    if videos_dir is not None:
        _capture_frame(obs, frames)
    if live_view is not None:
        live_view.publish(
            obs,
            force=True,
            phase="running",
            episode=episode_id,
            step=0,
            task=task.name,
            language=task.language,
            success=False,
        )

    scheduler = ChunkScheduler(horizon)
    total_return = 0.0
    steps = 0
    success = False

    while steps < max_steps:
        if scheduler.empty:
            wire_obs = _wire_observation(obs, env_id=0, episode_id=episode_id,
                                          language=task.language, task_id=task.name)
            resp = client.act(f"{episode_id}:{steps}", [key], [wire_obs], scheduler.requested_horizon)
            scheduler.fill(resp["actions"][0]["action_chunk"])

        action = scheduler.pop()
        obs, reward, done, info = env.step(action)
        total_return += float(reward)
        steps += 1
        if videos_dir is not None:
            _capture_frame(obs, frames)
        success = bool(env.check_success())
        if live_view is not None:
            live_view.publish(
                obs,
                force=success or steps == max_steps,
                phase="running",
                episode=episode_id,
                step=steps,
                task=task.name,
                language=task.language,
                success=success,
            )
        if success:
            break
        if done:
            break

    if videos_dir is not None and frames:
        try:
            import imageio.v2 as imageio

            video_path = videos_dir / f"episode_{episode_id:04d}.mp4"
            imageio.mimwrite(str(video_path), frames, fps=int(config.get("video_fps", 30)), quality=8)
            print(f"episode {episode_id}: wrote video {video_path} ({len(frames)} frames)", flush=True)
        except Exception as exc:  # noqa: BLE001 - best-effort, must not fail the episode
            print(f"episode {episode_id}: video encode failed: {exc}", flush=True)

    if live_view is not None:
        live_view.update_status(phase="episode complete", success=success, step=steps)

    return {
        "episode_id": episode_id,
        "seed": seed,
        "task_id": task.name,
        "status": "completed",
        "success": success,
        "steps": steps,
        "total_return": round(total_return, 6),
        "num_requests": scheduler.num_requests,
        "mean_action_horizon": scheduler.mean_action_horizon,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = json.loads(Path(args.config).read_text())
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = None
    if config.get("record_video") and args.videos_dir:
        videos_dir = Path(args.videos_dir)
        videos_dir.mkdir(parents=True, exist_ok=True)

    # LIBERO env creation (compiling the MJCF scene) plus CPU rendering is much
    # slower to come up than the fake evaluator's instant health check.
    health_timeout_s = float(config.get("health_timeout_s", 60.0))
    client = make_policy_client(
        config.get("policy_scheme", "http"), config["policy_host"], config["policy_port"],
        timeout_s=30.0, action_dim=config.get("action_dim"),
        observation_mapping=config.get("policy_observation_mapping"),
    )
    try:
        client.wait_healthy(timeout_s=health_timeout_s)
    except PolicyClientError as exc:
        print(f"libero_eval: policy service unreachable: {exc}", flush=True)
        return 1

    suite_name = config.get("suite", "libero_spatial")
    task_index = int(config.get("task_index", 0))
    camera_height = int(config.get("camera_height", 128))
    camera_width = int(config.get("camera_width", 128))

    live_view = None
    if config.get("live_view"):
        live_view = LiberoLiveView(
            host=str(config.get("live_view_host", "127.0.0.1")),
            port=int(config.get("live_view_port", 8766)),
            fps=int(config.get("live_view_fps", 10)),
        )
        try:
            print(f"VIEWER_URL {live_view.start()}", flush=True)
        except OSError as exc:
            print(f"libero_eval: live view disabled: {exc}", flush=True)
            live_view = None

    env = None
    try:
        env, task, init_states = _make_env(suite_name, task_index, camera_height, camera_width)
        if live_view is not None:
            live_view.update_status(
                phase="ready", task=task.name, language=task.language, success=False
            )
        print(f"libero_eval: suite={suite_name} task={task.name!r} language={task.language!r} "
              f"init_states={len(init_states)} record_video={videos_dir is not None}", flush=True)
        for episode_id, seed in enumerate(config["seeds"]):
            record = run_episode(
                client, env, task, init_states, episode_id, seed, config, videos_dir, live_view
            )
            (output_dir / f"episode_{episode_id:04d}.json").write_text(json.dumps(record, indent=2))
            print(f"episode {episode_id} done: success={record['success']} steps={record['steps']}",
                  flush=True)
    finally:
        if live_view is not None:
            live_view.update_status(phase="complete")
            live_view.close()
        if env is not None:
            env.close()

    print("libero_eval_complete", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
