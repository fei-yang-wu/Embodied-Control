"""Offline stance/bridge candidates; dynamics must be checked in rehearsal."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from embodied_control.lowlevel.reference import ReferenceArrays
from embodied_control.lowlevel.reference_catalog import motion_sha256
from embodied_control.lowlevel.reference_deploy import classify_motion


def _hermite(a, b, va, vb, seconds, frames):
    u = np.linspace(0, 1, frames + 1)[:, None]
    return ((2*u**3 - 3*u**2 + 1)*a + (u**3 - 2*u**2 + u)*seconds*va
            + (-2*u**3 + 3*u**2)*b + (u**3 - u**2)*seconds*vb)


def _yaw_quaternion(q):
    w, x, y, z = q
    yaw = np.arctan2(2*(w*z + x*y), 1-2*(y*y + z*z))
    return np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])


def compose_reference(reference_root, motion_name, bundle, model_path, destination,
                      *, hold_seconds=2.0, bridge_seconds=3.0):
    """Write one derived motion with a stationary encoder window at both ends.

    Cubic Hermite bridges match positional endpoint velocities. Quaternion
    curves use normalized Hermite interpolation with tangent-space derivatives.
    This is kinematic synthesis, not a contact-aware motion planner.
    """
    import mujoco
    from embodied_control.lowlevel.envs.mujoco import load_scene_model

    if not np.isfinite([hold_seconds, bridge_seconds]).all() or min(hold_seconds, bridge_seconds) <= 0:
        raise ValueError('hold and bridge durations must be positive and finite')
    ref = ReferenceArrays(reference_root)
    motion = ref.motion(motion_name)
    action = bundle.manifest.action
    if ref.joint_names != list(action.isaac_joint_names):
        raise ValueError('bundle and reference joint order disagree')
    command = bundle.manifest.command
    lookahead = command.horizon_steps * command.macro_frame_stride
    hold_frames = int(np.ceil(hold_seconds * ref.fps))
    bridge_frames = int(np.ceil(bridge_seconds * ref.fps))
    if hold_frames <= lookahead:
        raise ValueError(f'hold needs more than {lookahead} frames for this encoder')
    if motion.length < 2 or motion.joint_qvel is None:
        raise ValueError('composition requires two or more frames and joint velocities')
    qpos = np.concatenate([motion.anchor_pos_w, motion.anchor_quat_w[:, [3, 0, 1, 2]], motion.joint_qpos], axis=1).astype(np.float64)
    if not np.isfinite(qpos).all() or not np.isfinite(motion.joint_qvel).all():
        raise ValueError('cannot compose non-finite reference')
    for i in range(1, len(qpos)):
        if np.dot(qpos[i-1, 3:7], qpos[i, 3:7]) < 0:
            qpos[i, 3:7] *= -1
    velocity = np.gradient(qpos, 1/ref.fps, axis=0)
    velocity[:, 7:] = motion.joint_qvel
    for i in (0, -1):
        q = qpos[i, 3:7]
        velocity[i, 3:7] -= q * np.dot(q, velocity[i, 3:7])
    start, end = qpos[0].copy(), qpos[-1].copy()
    for stance in (start, end):
        stance[2] = .79
        stance[3:7] = _yaw_quaternion(stance[3:7])
        stance[7:] = action.default_joint_pos
    if np.dot(start[3:7], qpos[0, 3:7]) < 0:
        start[3:7] *= -1
    if np.dot(end[3:7], qpos[-1, 3:7]) < 0:
        end[3:7] *= -1
    seconds = bridge_frames / ref.fps
    bridge_in = _hermite(start, qpos[0], np.zeros_like(start), velocity[0], seconds, bridge_frames)
    bridge_out = _hermite(qpos[-1], end, velocity[-1], np.zeros_like(end), seconds, bridge_frames)
    combined = np.concatenate([np.tile(start, (hold_frames, 1)), bridge_in[:-1], qpos,
                               bridge_out[1:], np.tile(end, (hold_frames, 1))])
    norms = np.linalg.norm(combined[:, 3:7], axis=1, keepdims=True)
    if np.any(norms < .1):
        raise ValueError('bridge quaternion curve is singular; choose another reference')
    combined[:, 3:7] /= norms
    model = load_scene_model(model_path)
    data = mujoco.MjData(model)
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ref.joint_names]
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ref.body_names]
    if min(joint_ids + body_ids) < 0 or ref.anchor_body != 'pelvis':
        raise ValueError('composition requires a pelvis anchor and matching MJCF joints/bodies')
    addresses = model.jnt_qposadr[joint_ids]
    dofs = model.jnt_dofadr[joint_ids]
    replay = np.tile(model.qpos0, (len(combined), 1))
    replay[:, :7] = combined[:, :7]
    replay[:, addresses] = combined[:, 7:]
    velocities = np.zeros((len(combined), model.nv))
    body_pos = np.empty((len(combined), len(body_ids), 3), dtype=np.float32)
    for i in range(len(combined)):
        left, right = max(0, i-1), min(len(combined)-1, i+1)
        mujoco.mj_differentiatePos(model, velocities[i], (right-left)/ref.fps, replay[left], replay[right])
        data.qpos[:] = replay[i]
        mujoco.mj_kinematics(model, data)
        body_pos[i] = data.xpos[body_ids]
    arrays = {'qpos': combined.astype(np.float32),
              'qvel': np.concatenate([velocities[:, :6], velocities[:, dofs]], axis=1).astype(np.float32),
              'anchor_pos_w': combined[:, :3].astype(np.float32),
              'anchor_quat_w': combined[:, [4, 5, 6, 3]].astype(np.float32),
              'body_pos_w': body_pos}
    source_sha = motion_sha256(ref, motion_name)
    recipe = {'version': 1, 'source_sha256': source_sha, 'stance': list(action.default_joint_pos),
              'hold_frames': hold_frames, 'bridge_frames': bridge_frames, 'fps': ref.fps,
              'model_sha256': hashlib.sha256(Path(model_path).read_bytes()).hexdigest()}
    recipe_sha = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    name = f'{motion_name}__stand_{recipe_sha[:12]}'
    manifest = {'format_version': 1, 'fps': ref.fps,
                'key': {'joint_names': ref.joint_names, 'body_names': list(ref.body_names),
                        'anchor_body': ref.anchor_body, 'source': {'dataset_id': 'bones', 'recipe': recipe},
                        'arrays': {key: {'shape': list(a.shape), 'dtype': 'float32',
                                         'quaternion_order': 'xyzw' if key == 'anchor_quat_w' else None}
                                   for key, a in arrays.items()}},
                'traj_info': {'start_index': [0], 'end_index': [len(combined)],
                              'ordered_traj_list': [['bones', name]], 'capacity': len(combined), 'written': len(combined)}}
    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.compose-', dir=target.parent) as temporary:
        staging = Path(temporary) / 'tree'
        staging.mkdir()
        for key, array in arrays.items():
            array.tofile(staging / f'{key}.memmap')
        path = staging / 'reference_arrays_manifest.json'
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        built = ReferenceArrays(staging)
        verdict = classify_motion(built.motion(name), stance=action.default_joint_pos, fps=ref.fps)
        manifest['motions'] = {name: {'dataset_id': 'bones', 'parent_motion': motion_name, 'recipe': recipe,
                                     'sha256': motion_sha256(built, name), 'frames': len(combined),
                                     'hold_frames': hold_frames, 'bridge_frames': bridge_frames,
                                     'source_start_frame': hold_frames + bridge_frames,
                                     'requires_rehearsal': True, **verdict.to_json()}}
        path.write_text(json.dumps(manifest, indent=2) + '\n')
        staging.rename(target)
    return name
