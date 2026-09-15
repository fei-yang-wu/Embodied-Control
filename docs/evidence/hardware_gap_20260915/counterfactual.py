"""Exploratory synchronous simulation, with one changed factor per condition.

This starts at reference frame 0 and is not a lifecycle rehearsal or hardware gate.
Run: pixi run -e native python docs/evidence/hardware_gap_20260915/counterfactual.py
"""

import argparse
from dataclasses import replace
from pathlib import Path
import json

import ec_native
import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.contracts import JointCommand
from embodied_control.lowlevel.envs.mujoco import MujocoBackend
from embodied_control.lowlevel.maths import quat_mul, rotate_inverse
from embodied_control.lowlevel.native_core import NativeTracker
from embodied_control.lowlevel.reference import ReferenceArrays

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MOTION = "walking_quip_360_R_002__A428"
MODEL = ROOT / "assets/latent_playkit/model/g1_29dof_rev_1_0.xml"
CONDITIONS = ("baseline", "half_rate", "position_gain_0.6", "position_bias_5cm",
              "imu_pitch_2deg", "gyro_delay_20ms", "state_delay_20ms", "state_delay_40ms",
              "measured_noise", "training_noise", "command_delay_20ms")


def run(bundle_name, condition):
    bundle = PolicyBundle.load(ROOT / "assets/models/controller" / bundle_name)
    command, action = bundle.manifest.command, bundle.manifest.action
    reference = ReferenceArrays(ROOT / "assets/models/reference/bones")
    motion = reference.motion(MOTION)
    assert action.isaac_joint_names == reference.joint_names
    legs = [i for i, n in enumerate(reference.joint_names) if any(s in n for s in ("hip", "knee", "ankle"))]
    backend = MujocoBackend(action, MODEL)
    backend.set_pose(np.r_[motion.anchor_pos_w[0], motion.anchor_quat_w[0]], motion.joint_qpos[0])
    tracker = NativeTracker(bundle, intra_op_threads=1)
    enc = bundle.manifest.models["encoder_onnx"]
    encoder = ec_native.OnnxEngine(str(bundle.root / enc.path), enc.input_name, enc.output_name,
                                  enc.input_shape[-1], enc.output_shape[-1], 1)
    raw_root = np.c_[motion.joint_qpos, motion.anchor_pos_w, motion.anchor_quat_w].astype(np.float32)
    raw_sonic = np.c_[motion.joint_qpos, motion.joint_qvel, motion.anchor_quat_w].astype(np.float32)
    positions, targets, poses, torques = [], [], [], []
    states = []
    rng = np.random.default_rng(42)
    for frame in range(motion.length):
        state = backend.read_state()
        state = replace(state, base_ang_vel=state.base_ang_vel.copy())
        states.append(state)
        measured = state
        if condition.startswith("state_delay_"):
            delay = 2 if condition.endswith("40ms") else 1
            past = states[max(0, frame-delay)]
            measured = replace(past, stamp=state.stamp)
        elif condition == "gyro_delay_20ms":
            measured = replace(state, base_ang_vel=states[max(0, frame-1)].base_ang_vel)
        elif condition == "position_gain_0.6":
            pos = state.anchor_pos_w.copy()
            pos[:2] = motion.anchor_pos_w[0, :2] + .6 * (pos[:2]-motion.anchor_pos_w[0, :2])
            measured = replace(state, anchor_pos_w=pos)
        elif condition == "position_bias_5cm":
            pos = state.anchor_pos_w.copy()
            pos[0] += .05 * np.clip((frame-100)/50, 0, 1)
            measured = replace(state, anchor_pos_w=pos)
        elif condition == "imu_pitch_2deg":
            angle = np.deg2rad(2)
            quat = quat_mul(state.anchor_quat_w, np.array([0, np.sin(angle/2), 0, np.cos(angle/2)], dtype=np.float32))
            measured = replace(state, anchor_quat_w=quat,
                projected_gravity=rotate_inverse(quat, np.array([0, 0, -1], dtype=np.float32)))
        elif condition in ("measured_noise", "training_noise"):
            qp, qv, gyro, tilt = ((.0001, .025, .015, .0015) if condition == "measured_noise" else (.01, .5, .2, .05))
            dq = np.r_[rng.uniform(-tilt, tilt, 3)/2, 1.0].astype(np.float32)
            dq /= np.linalg.norm(dq)
            quat = quat_mul(state.anchor_quat_w, dq)
            measured = replace(state, joint_pos=state.joint_pos+rng.uniform(-qp, qp, 29),
                joint_vel=state.joint_vel+rng.uniform(-qv, qv, 29),
                base_ang_vel=state.base_ang_vel+rng.uniform(-gyro, gyro, 3),
                anchor_quat_w=quat,
                projected_gravity=rotate_inverse(quat, np.array([0, 0, -1], dtype=np.float32)))
        if frame == 0 or condition != "half_rate" or frame % 2 == 0:
            indices = np.minimum(frame+np.arange(10)*int(command.macro_frame_stride), motion.length-1)
            if command.encoder_state_interface == "root_qpos":
                window = ec_native.reexpress_root_qpos_window(raw_root[indices], measured.anchor_pos_w,
                                                              measured.anchor_quat_w, True).ravel()
            else:
                # The packer needs unstrided raw frames, including terminal padding.
                raw_indices = np.minimum(frame+np.arange(46), motion.length-1)
                window = ec_native.pack_joint_qpos_qvel_anchor_ori_window(raw_sonic[raw_indices], 0, 10, 5,
                                                                          measured.anchor_quat_w)
            z = encoder.infer(window)
        vector = np.r_[z, 0, 1].astype(np.float32) if command.phase_mode == "sin_cos" else z
        result = tracker.step_once(measured, vector)
        target = result["joint_target"]
        positions.append(state.joint_pos.copy())
        poses.append(np.r_[state.anchor_pos_w, state.anchor_quat_w])
        targets.append(target.copy())
        torques.append(backend.data.actuator_force.copy())
        applied = targets[max(0, frame-1)] if condition == "command_delay_20ms" else target
        backend.write_command(JointCommand(applied, np.asarray(action.stiffness), np.asarray(action.damping)))
        if backend.base_height < .4:
            break
    q, target, pose = np.array(positions), np.array(targets), np.array(poses)
    x, y, zq, w = pose[:, 3:].T
    pitch = np.rad2deg(np.arcsin(np.clip(2*(w*y-zq*x), -1, 1)))
    n = len(q)
    knee = reference.joint_names.index("left_knee_joint")
    take = min(n, 256)
    row = {"condition": condition, "steps": n, "completed": n == motion.length,
        "leg_reference_mae_rad": float(np.abs(q[1:take]-motion.joint_qpos[1:take])[:, legs].mean()),
        "left_knee_target_step_p99_rad": float(np.percentile(np.abs(np.diff(target[1:take, knee])), 99)),
        "pitch_min_max_deg": [float(pitch[1:take].min()), float(pitch[1:take].max())],
        "travel_m": float(np.linalg.norm(pose[-1,:2]-pose[0,:2])),
        "final_root_error_m": float(np.linalg.norm(pose[-1,:2]-motion.anchor_pos_w[n-1,:2]))}
    np.savez_compressed(OUT / f"counterfactual_{bundle_name}_{condition}.npz", q=q, target=target, pose=pose,
                        pitch_deg=pitch, torque=np.array(torques))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    args = parser.parse_args()
    rows = {}
    for bundle in ("post_c1l1j1p1_f84500152320", "sonic_v1_1"):
        rows[bundle] = []
        for condition in args.conditions:
            row = run(bundle, condition)
            rows[bundle].append(row)
            print(bundle, json.dumps(row), flush=True)
    (OUT / "counterfactual.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
