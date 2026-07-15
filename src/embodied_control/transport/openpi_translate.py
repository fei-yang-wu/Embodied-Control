"""Neutral observation <-> OpenPI ``infer(obs)`` dict translation.

Pure functions, no network -- fully unit-testable without a real server. Kept
separate from ``openpi_client.py`` deliberately: the wire mechanics
(websocket + msgpack-numpy) are fixed per OpenPI's server *software*, but the
key names below are fixed per *trained checkpoint* (a pi0-LIBERO checkpoint
and a pi0-DROID checkpoint can plausibly expect different keys even though
both are served by the same OpenPI server code). See
``docs/design/real_policy_adapters.md`` M3 for wiring an actual checkpoint.

``OpenPIObservationMapping``'s defaults are placeholders, not verified
against any specific checkpoint -- they must be set from the checkpoint's
own training/deploy config (typically visible in its policy config or the
serving script that mounts it) before pointing this at a real model.

``libero_pi0_proprio``/``flip_images_180`` encode two checkpoint-specific
preprocessing steps verified against OpenPI's own reference LIBERO eval
script (``examples/libero/main.py``, Physical-Intelligence/openpi,
2026-07-15), not this repo's own convention: images are flipped 180 degrees
("IMPORTANT: rotate 180 degrees to match train preprocessing" -- their
comment, not ours) and ``observation/state`` is NOT a raw proprio dump but
specifically ``concat(robot0_eef_pos, quat2axisangle(robot0_eef_quat),
robot0_gripper_qpos)`` -- 8 floats, in that exact order. Sending our full
flattened proprio vector instead would silently feed the model a
differently-shaped, differently-ordered, wrong-representation (quaternion
instead of axis-angle) input -- the pipeline would still "run", just not
mean anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from embodied_control.transport.libero_proprio import eef_pose_and_gripper


@dataclass
class OpenPIObservationMapping:
    # our camera name (e.g. "agentview_image") -> the key the served
    # checkpoint expects (e.g. "observation/image")
    camera_keys: dict[str, str] = field(default_factory=dict)
    proprio_key: str = "observation/state"
    prompt_key: str = "prompt"
    # key in the dict returned by infer() holding the action chunk
    action_key: str = "actions"
    # Checkpoint-specific preprocessing, verified for pi05_libero specifically
    # (see module docstring) -- not a generic feature, don't set for other
    # checkpoints without re-verifying against their own reference eval code.
    flip_images_180: bool = False
    libero_pi0_proprio: bool = False


def _libero_pi0_state(proprio: dict) -> np.ndarray:
    eef_pos, axisangle, gripper_qpos = eef_pose_and_gripper(proprio)
    # Must be a real ndarray, not a plain list: verified live against a real
    # server that its normalization pipeline calls `.shape` on this value
    # (AttributeError: 'list' object has no attribute 'shape' otherwise) --
    # msgpack-numpy only special-cases actual ndarrays over the wire.
    return np.asarray(eef_pos + axisangle + gripper_qpos, dtype=np.float32)


def _flip_180(array):
    return array[::-1, ::-1]


def observation_to_openpi(obs: dict, mapping: OpenPIObservationMapping) -> dict:
    """Our neutral observation (``env_id``/``episode_id``/``proprio``/
    ``cameras``/``task``) -> the ``obs`` dict ``infer()`` expects."""
    payload: dict = {}
    for camera in obs.get("cameras") or []:
        their_key = mapping.camera_keys.get(camera["name"])
        if their_key is not None:
            array = camera["array"]
            payload[their_key] = _flip_180(array) if mapping.flip_images_180 else array

    proprio = obs.get("proprio") or {}
    if mapping.libero_pi0_proprio:
        payload[mapping.proprio_key] = _libero_pi0_state(proprio)
    elif proprio.get("values"):
        # ndarray, not a plain list -- see _libero_pi0_state's comment; this
        # is a real OpenPI server requirement (its normalization transform
        # calls .shape), not specific to the libero_pi0_proprio preset.
        payload[mapping.proprio_key] = np.asarray(proprio["values"], dtype=np.float32)

    task = obs.get("task") or {}
    if task.get("language_instruction"):
        payload[mapping.prompt_key] = task["language_instruction"]

    return payload


def openpi_action_to_chunk(action: dict, mapping: OpenPIObservationMapping) -> list[list[float]]:
    """The dict returned by ``infer()`` -> our ``action_chunk`` wire shape
    (list of per-timestep float lists), regardless of whether the checkpoint
    returned a numpy array or a nested list for the chunk."""
    if mapping.action_key not in action:
        raise KeyError(
            f"openpi response missing {mapping.action_key!r}; keys present: {sorted(action)}"
        )
    chunk = action[mapping.action_key]
    return [[float(v) for v in row] for row in chunk]
