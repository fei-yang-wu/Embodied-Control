"""Headless, body-only GR00T/TWIST2 integration smoke (not a task benchmark).

TWIST2 observation, gains and PD equations follow deploy_real/
server_low_level_g1_sim.py at d5c7108. No Isaac Gym or Torch imports are needed
for its exported ONNX controller. GR00T remains in a separate process.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

from embodied_control.transport.factory import make_policy_client
from embodied_control.transport.chunking import ChunkScheduler

STATE_DIMS = dict(left_wrist_eef_9d=9, right_wrist_eef_9d=9, left_hand=7,
                  right_hand=7, left_arm=7, right_arm=7, waist=3)
ACTION_DIMS = {**STATE_DIMS, "base_height_command": 1, "navigate_command": 3}
DEFAULT = np.array([-0.2, 0, 0, .4, -.2, 0] * 2 + [0] * 3
                   + [0, .4, 0, 1.2, 0, 0, 0] + [0, -.4, 0, 1.2, 0, 0, 0])
KP = np.array([100, 100, 100, 150, 40, 40] * 2 + [150] * 3
              + [40, 40, 40, 40, 4, 4, 4] * 2)
KD = np.array([2, 2, 2, 4, 2, 2] * 2 + [4] * 3 + [5, 5, 5, 5, .2, .2, .2] * 2)


def mapping():
    return dict(camera_keys={"ego": "video.ego_view"}, batched=True,
                state_keys={k: "state." + k for k in STATE_DIMS},
                state_dims=STATE_DIMS,
                action_dims={"action." + k: v for k, v in ACTION_DIMS.items()},
                prompt_key="annotation.human.task_description")


def body_command(action, joints, joint_limits):
    """Consume decoded absolute arm/waist targets; bound displacement for smoke.

    EEF, hands and navigation are recorded but not mapped to actuators. Fixed
    root commands and nominal legs make the limited execution contract explicit.
    """
    action = np.asarray(action, dtype=float)
    if action.shape != (53,) or not np.isfinite(action).all():
        raise ValueError("Expected 53 finite base-G1 decoded action values")
    offsets = {}; start = 0
    for name, width in ACTION_DIMS.items():
        offsets[name] = action[start:start + width]; start += width
    targets = DEFAULT.copy()
    desired = np.concatenate([offsets["waist"], offsets["left_arm"], offsets["right_arm"]])
    targets[12:] = np.clip(desired, joints[12:] - .05, joints[12:] + .05)
    targets = np.clip(targets, joint_limits[:, 0], joint_limits[:, 1])
    return np.concatenate([[0, 0, .793, 0, 0, 0], targets])


class Twist2Plant:
    def __init__(self, root):
        root = Path(root)
        self.model = mujoco.MjModel.from_xml_path(str(root / "assets/g1/g1_sim2sim_29dof.xml"))
        self.data = mujoco.MjData(self.model)
        if self.model.nq != 36 or self.model.nu != 29:
            raise ValueError("TWIST2 smoke requires the pinned 29-DOF model")
        self.model.opt.timestep = .001
        self.data.qpos[:] = np.r_[0, 0, .793, 1, 0, 0, 0, DEFAULT]
        self.data.qpos[7 + 16] = .2; self.data.qpos[7 + 23] = -.2
        mujoco.mj_forward(self.model, self.data)
        checkpoint = root / "assets/ckpts/twist2_1017_20k.onnx"
        self.controller_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        options = ort.SessionOptions(); options.intra_op_num_threads = 2
        self.policy = ort.InferenceSession(str(checkpoint), options, providers=["CPUExecutionProvider"])
        self.input_name = self.policy.get_inputs()[0].name
        if self.policy.get_inputs()[0].shape[-1] != 1432:
            raise ValueError("TWIST2 controller requires 1432 observation features")
        self.last_action = np.zeros(29)
        self.history = deque([np.zeros(127) for _ in range(10)], maxlen=10)
        self.renderer = mujoco.Renderer(self.model, height=224, width=224)
        self.frames = deque(maxlen=21)
        self.ticks = 0

    def step(self, command):
        q = self.data.qpos[7:].copy(); dq = self.data.qvel[6:].copy()
        w, x, y, z = self.data.qpos[3:7]
        rp = [np.arctan2(2 * (w*x+y*z), 1-2*(x*x+y*y)),
              np.arcsin(np.clip(2*(w*y-z*x), -1, 1))]
        dq[[4, 5, 10, 11]] = 0
        proprio = np.r_[self.data.qvel[3:6] * .25, rp, q - DEFAULT, dq * .05, self.last_action]
        current = np.r_[command, proprio]
        observation = np.r_[current, np.asarray(self.history).ravel(), command].astype(np.float32)
        self.history.append(current)
        raw = self.policy.run(None, {self.input_name: observation[None]})[0].reshape(29)
        if not np.isfinite(raw).all():
            raise ValueError("Nonfinite TWIST2 output")
        self.last_action = raw
        target = np.clip(raw, -10, 10) * .5 + DEFAULT
        for _ in range(10):
            self.data.ctrl[:] = np.clip((target - self.data.qpos[7:]) * KP - self.data.qvel[6:] * KD, -KP, KP)
            mujoco.mj_step(self.model, self.data)
        self.ticks += 1
        if not np.isfinite(self.data.qpos).all():
            raise ValueError("Nonfinite MuJoCo state")

    def image(self):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = self.data.xpos[self.model.body("torso_link").id]
        camera.distance = 1.8; camera.azimuth = 180; camera.elevation = -15
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render().copy()

    def observation(self, prompt):
        image = self.image(); self.frames.append(image)
        torso = self.model.body("torso_link").id
        rotation = self.data.xmat[torso].reshape(3, 3)
        state = dict(left_hand=[0]*7, right_hand=[0]*7,
                     left_arm=self.data.qpos[22:29].tolist(),
                     right_arm=self.data.qpos[29:36].tolist(), waist=self.data.qpos[19:22].tolist())
        for side in ("left", "right"):
            wrist = self.model.body(side + "_wrist_yaw_link").id
            pos = rotation.T @ (self.data.xpos[wrist] - self.data.xpos[torso])
            rot = rotation.T @ self.data.xmat[wrist].reshape(3, 3)
            state[side + "_wrist_eef_9d"] = np.r_[pos, rot[:, :2].T.ravel()].tolist()
        return dict(cameras=[dict(name="ego", array=np.stack([self.frames[0], image]))],
                    proprio=state, task=dict(language_instruction=prompt))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--videos-dir", required=True)
    args = parser.parse_args(argv); config = json.loads(Path(args.config).read_text())
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    videos = Path(args.videos_dir); videos.mkdir(parents=True, exist_ok=True)
    if config.get("replay_episode"):
        from embodied_control.sim.twist2_replay import run
        return run(config, output, videos)
    client = make_policy_client(config["policy_scheme"], config["policy_host"], config["policy_port"],
                                timeout_s=120, action_dim=53, observation_mapping=mapping())
    try:
        client.wait_healthy(timeout_s=120)
        for episode, seed in enumerate(config["seeds"]):
            plant = Twist2Plant(config.get("twist2_root", "/twist2"))
            scheduler = ChunkScheduler(40); rows = []; latencies = []
            writer = imageio.get_writer(str(videos / f"episode_{episode:04d}.mp4"), fps=25)
            try:
                client.reset([[0, episode]], seed)
                neutral = np.r_[0, 0, .793, 0, 0, 0, DEFAULT]
                for _ in range(100):
                    plant.step(neutral)
                for step in range(config["max_steps_per_episode"]):
                    if scheduler.empty:
                        obs = plant.observation(config.get("prompt", "Stand still and raise your right arm."))
                        start = time.monotonic()
                        result = client.act(str(step), [[0, episode]], [obs], 40)
                        latencies.append(time.monotonic()-start)
                        scheduler.fill(result["actions"][0]["action_chunk"])
                    action = scheduler.pop()
                    command = body_command(action, plant.data.qpos[7:], plant.model.jnt_range[1:])
                    plant.step(command)
                    rows.append(dict(step=step, sim_time=plant.data.time, action=action,
                                     command=command.tolist(), qpos=plant.data.qpos.tolist()))
                    if step % 4 == 0:
                        frame = plant.image(); writer.append_data(frame); plant.frames.append(frame)
                record = dict(episode_id=episode, seed=seed, task_id="g1_twist2_body_integration_smoke",
                              status="completed", success=True, steps=len(rows), total_return=0,
                              num_requests=scheduler.num_requests, mean_action_horizon=scheduler.mean_action_horizon,
                              controller_sha256=plant.controller_sha256, inference_latency_s=latencies,
                              controller_ticks=plant.ticks, final_base_height=float(plant.data.qpos[2]),
                              semantics="Finite inference/controller/physics execution only; not task success")
                (output / f"episode_{episode:04d}.json").write_text(json.dumps(record, indent=2))
                (output / f"trace_{episode:04d}.json").write_text(json.dumps(rows))
                print(json.dumps(record), flush=True)
            finally:
                writer.close(); plant.renderer.close()
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
