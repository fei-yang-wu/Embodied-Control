"""Recorded G1 pipette inputs with predicted TWIST2 body commands in MuJoCo."""

import hashlib
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_control.sim.twist2_eval import Twist2Plant
from embodied_control.transport.factory import make_policy_client

BODY_DIMS = dict(left_leg=6, right_leg=6, waist=3, left_arm=7, right_arm=7)
STATE_DIMS = dict(**BODY_DIMS, imu_ang_vel=3, imu_rp=2, hand_left=6, hand_right=6)
ACTION_DIMS = dict(
    **BODY_DIMS,
    root_lin_vel_xy=2,
    root_height=1,
    root_rp=2,
    root_yaw_rate=1,
    hand_left=6,
    hand_right=6,
)


def mapping():
    return dict(
        camera_keys={k: f"video.{k}" for k in ("rgb", "wrist_left", "wrist_right")},
        state_keys={k: f"state.{k}" for k in STATE_DIMS},
        state_dims=STATE_DIMS,
        action_dims={f"action.{k}": v for k, v in ACTION_DIMS.items()},
        batched=True,
        prompt_key="annotation.human.task_description",
    )


def body_command(action):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (47,) or not np.isfinite(action).all():
        raise ValueError("Expected 47 finite absolute pipette action values")
    return np.r_[action[29:35], action[:29]]


def ticks_for_frame(frame, fps=60, controller_hz=100):
    return ((frame + 1) * controller_hz // fps) - (frame * controller_hz // fps)


def load_episode(path):
    path = Path(path)
    record = json.loads(path.read_text())
    if record["schema"] != "ec.g1-pipette-replay/v1" or record["fps"] != 60:
        raise ValueError("Require the versioned 60 Hz pipette replay contract")
    if record["root_command_names"] != ["vx_local", "vy_local", "z", "roll", "pitch", "yaw_rate"]:
        raise ValueError("Unexpected root command semantics")
    count = len(record["timestamps"])
    if count < 1 or len(record["language"]) != count:
        raise ValueError("Replay language/timestamp length mismatch")
    if not np.allclose(record["timestamps"], np.arange(count) / 60, atol=1e-4):
        raise ValueError("Replay timestamps must start at zero and be contiguous")
    for group, dims in (("state", STATE_DIMS), ("action", ACTION_DIMS)):
        if set(record[group]) != set(dims):
            raise ValueError(f"Unexpected {group} modalities")
        for key, width in dims.items():
            value = np.asarray(record[group][key])
            if value.shape != (count, width) or not np.isfinite(value).all():
                raise ValueError(f"Invalid {group}.{key}")
    if set(record["videos"]) != {"rgb", "wrist_left", "wrist_right"}:
        raise ValueError("Require all three recorded cameras")
    for spec in record["videos"].values():
        video = (path.parent / spec["path"]).resolve()
        if not video.is_relative_to(path.parent.resolve()):
            raise ValueError("Replay video must be inside its episode directory")
        with video.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != spec["sha256"]:
            raise ValueError("Replay video checksum mismatch")
    return record


def montage(sim, cameras, text):
    canvas = Image.new("RGB", (448, 490))
    for i, (label, array) in enumerate([("MuJoCo / TWIST2", sim), *cameras.items()]):
        tile = Image.fromarray(array)
        tile.thumbnail((224, 200))
        x, y = (i % 2) * 224, (i // 2) * 224
        canvas.paste(tile, (x + (224 - tile.width) // 2, y + 20))
        ImageDraw.Draw(canvas).text((x + 4, y + 3), label, fill="white")
    ImageDraw.Draw(canvas).text((4, 452), text[:140], fill="white")
    return np.asarray(canvas)


def run(config, output, videos):
    path = Path(config["replay_episode"])
    record = load_episode(path)
    if config["action_dim"] != 47 or config["requested_action_horizon"] != 40:
        raise ValueError("Pipette checkpoint requires action dimension 47 and horizon 40")
    source = config.get("action_source", "policy")
    if source not in ("policy", "recorded"):
        raise ValueError("action_source must be policy or recorded")
    client = (
        make_policy_client(
            config["policy_scheme"],
            config["policy_host"],
            config["policy_port"],
            timeout_s=120,
            action_dim=47,
            observation_mapping=mapping(),
        )
        if source == "policy"
        else None
    )
    ground_truth = np.concatenate([record["action"][k] for k in ACTION_DIMS], axis=1)
    count = min(len(ground_truth), config["max_steps_per_episode"])
    try:
        if client:
            client.wait_healthy(timeout_s=120)
        for episode, seed in enumerate(config["seeds"]):
            plant = Twist2Plant(config.get("twist2_root", "/twist2"))
            names = [plant.model.joint(i).name for i in range(1, plant.model.njnt)]
            if names != record["body_joint_names"]:
                plant.renderer.close()
                raise ValueError("Dataset joint order differs from the TWIST2 plant")
            readers = {
                k: imageio.get_reader(str(path.parent / v["path"]))
                for k, v in record["videos"].items()
            }
            writer = (
                imageio.get_writer(
                    str(videos / f"episode_{episode:04d}.mp4"), fps=60, macro_block_size=2
                )
                if config["record_video"]
                else None
            )
            rows, requests = [], []
            try:
                if client:
                    client.reset([[0, episode]], seed)
                plant.data.qpos[7:] = np.concatenate([record["state"][k][0] for k in BODY_DIMS])
                plant.data.qpos[2] = record["action"]["root_height"][0][0]
                roll, pitch = record["state"]["imu_rp"][0]
                plant.data.qpos[3:7] = [
                    np.cos(roll / 2) * np.cos(pitch / 2),
                    np.sin(roll / 2) * np.cos(pitch / 2),
                    np.cos(roll / 2) * np.sin(pitch / 2),
                    -np.sin(roll / 2) * np.sin(pitch / 2),
                ]
                mujoco.mj_forward(plant.model, plant.data)
                for _ in range(100):
                    plant.step(body_command(ground_truth[0]))
                start_time = plant.data.time
                chunk, chunk_start = None, 0
                for frame in range(count):
                    cameras = {k: reader.get_data(frame) for k, reader in readers.items()}
                    if source == "recorded":
                        action = ground_truth[frame]
                    else:
                        if chunk is None or frame >= chunk_start + len(chunk):
                            obs = dict(
                                cameras=[dict(name=k, array=v) for k, v in cameras.items()],
                                proprio={k: record["state"][k][frame] for k in STATE_DIMS},
                                task=dict(language_instruction=record["language"][frame]),
                            )
                            started = time.monotonic()
                            response = client.act(str(frame), [[0, episode]], [obs], 40)
                            chunk = np.asarray(response["actions"][0]["action_chunk"])
                            if chunk.shape != (40, 47) or not np.isfinite(chunk).all():
                                raise ValueError("Expected a finite 40 by 47 policy action chunk")
                            chunk_start = frame
                            requests.append(
                                dict(
                                    frame=frame,
                                    latency_s=time.monotonic() - started,
                                    language=record["language"][frame],
                                    action_chunk=chunk.tolist(),
                                )
                            )
                        action = chunk[frame - chunk_start]
                    command = body_command(action)
                    for _ in range(ticks_for_frame(frame)):
                        plant.step(command)
                    rows.append(
                        dict(
                            frame=frame,
                            source_time=record["timestamps"][frame],
                            sim_time=plant.data.time - start_time,
                            action=action.tolist(),
                            target=ground_truth[frame].tolist(),
                            command=command.tolist(),
                            qpos=plant.data.qpos.tolist(),
                        )
                    )
                    if writer:
                        writer.append_data(
                            montage(
                                plant.image(),
                                cameras,
                                f"{source} | frame {frame} | " + record["language"][frame],
                            )
                        )
                actions = np.asarray([r["action"] for r in rows])
                positions = np.asarray([r["qpos"] for r in rows])
                metrics = {}
                for name, sl in (
                    ("body_rad", slice(0, 29)),
                    ("root_velocity_m_s", slice(29, 31)),
                    ("root_height_m", slice(31, 32)),
                    ("root_rp_rad", slice(32, 34)),
                    ("root_yaw_rate_rad_s", slice(34, 35)),
                    ("hands", slice(35, 47)),
                ):
                    metrics[f"action_rmse_{name}"] = float(
                        np.sqrt(np.mean((actions[:, sl] - ground_truth[:count, sl]) ** 2))
                    )
                if requests:
                    latencies = [request["latency_s"] for request in requests]
                    metrics["inference_latency_mean_s"] = float(np.mean(latencies))
                    metrics["inference_latency_p95_s"] = float(np.percentile(latencies, 95))
                metrics["tracking_rmse_rad"] = float(
                    np.sqrt(np.mean((positions[:, 7:] - actions[:, :29]) ** 2))
                )
                metrics["minimum_base_height_m"] = float(positions[:, 2].min())
                metrics["final_base_height_m"] = float(positions[-1, 2])
                result = dict(
                    episode_id=episode,
                    seed=seed,
                    task_id="g1_pipette_recorded_input_replay",
                    status="completed",
                    success=bool(positions[:, 2].min() > 0.35),
                    steps=count,
                    total_return=0,
                    num_requests=len(requests),
                    mean_action_horizon=40 if requests else 0,
                    metrics=metrics,
                    source_episode=record["episode_index"],
                    action_source=source,
                    initialization="Recorded joints and IMU roll/pitch; commanded height; zero world xy/yaw and velocities; 100 controller ticks with first recorded command",
                    controller_ticks=plant.ticks,
                    controller_sha256=plant.controller_sha256,
                    requests=requests,
                    episode_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    semantics="Recorded inputs; success means finite execution without base height below 0.35m, not task success; hands unactuated; inference pauses simulated time",
                )
                (output / f"episode_{episode:04d}.json").write_text(json.dumps(result, indent=2))
                (output / f"trace_{episode:04d}.json").write_text(json.dumps(rows))
                print(json.dumps(result), flush=True)
            finally:
                if writer:
                    writer.close()
                for reader in readers.values():
                    reader.close()
                plant.renderer.close()
    finally:
        if client:
            client.close()
    return 0
