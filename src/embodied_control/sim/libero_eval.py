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
from pathlib import Path

from embodied_control.transport.base import PolicyClientProtocol
from embodied_control.transport.chunking import ChunkScheduler
from embodied_control.transport.client import PolicyClientError
from embodied_control.transport.factory import make_policy_client

_IMAGE_KEY_SUFFIXES = ("_image", "_depth", "_segmentation")
_CAMERA_KEYS = ("agentview_image", "robot0_eye_in_hand_image")
_VIDEO_CAMERA_KEY = "agentview_image"


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
                config: dict, videos_dir: Path | None) -> dict:
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
        if env.check_success():
            success = True
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

    env, task, init_states = _make_env(suite_name, task_index, camera_height, camera_width)
    print(f"libero_eval: suite={suite_name} task={task.name!r} language={task.language!r} "
          f"init_states={len(init_states)} record_video={videos_dir is not None}", flush=True)

    try:
        for episode_id, seed in enumerate(config["seeds"]):
            record = run_episode(client, env, task, init_states, episode_id, seed, config, videos_dir)
            (output_dir / f"episode_{episode_id:04d}.json").write_text(json.dumps(record, indent=2))
            print(f"episode {episode_id} done: success={record['success']} steps={record['steps']}",
                  flush=True)
    finally:
        env.close()

    print("libero_eval_complete", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
