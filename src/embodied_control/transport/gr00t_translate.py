"""Neutral observation <-> GR00T ``get_action`` dict translation.

Pure functions, no network -- see ``openpi_translate.py``'s docstring for why
this stays separate from ``gr00t_client.py`` (transport is fixed per server
software; key names are fixed per trained checkpoint). ``Gr00tObservationMapping``'s
defaults are placeholders, not verified against any specific checkpoint.

``libero_gr00t_proprio``/``libero_gr00t_action``/``flip_images_180`` encode
checkpoint-specific preprocessing verified against GR00T's own reference
LIBERO env wrapper (``gr00t/eval/sim/LIBERO/libero_env.py``,
NVIDIA/Isaac-GR00T, 2026-07-15) for the public `nvidia/GR00T-N1.7-LIBERO`
checkpoint. **Structurally different from OpenPI's LIBERO preset, not just a
renamed copy**: GR00T-LIBERO splits proprioception into seven individual
``state.x``/``state.y``/``state.z``/``state.roll``/``state.pitch``/
``state.yaw``/``state.gripper`` keys (not one ``observation/state`` array),
and the action side is the mirror image -- seven separate ``action.*`` keys
in the response, concatenated back into a single 7-vector here, with the
gripper dimension needing a `[0,1]->[-1,1]` normalize-and-binarize plus a
sign flip (`normalize_gripper_action`/`invert_gripper_action` in GR00T's own
``libero_env.py`` -- copied here, not invented) before it matches what
`sim/libero_eval.py`'s OSC_POSE controller expects. Images are flipped 180
degrees, same as OpenPI's checkpoint and for the same reason (LIBERO/
robosuite rendering vs. training-time preprocessing convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from embodied_control.transport.libero_proprio import eef_pose_and_gripper

_LIBERO_ACTION_KEYS = ("action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw")


@dataclass
class Gr00tObservationMapping:
    # our camera name -> the key the served checkpoint expects
    # (GR00T convention is commonly "video.<name>", e.g. "video.ego_view")
    camera_keys: dict[str, str] = field(default_factory=dict)
    proprio_key: str = "state.joint_position"
    prompt_key: str = "annotation.human.action.task_description"
    # key in the action dict returned by get_action() holding the chunk.
    # GR00T checkpoints often split action into several named parts (e.g.
    # "action.joint_position", "action.gripper") -- if a real checkpoint does
    # that, this needs to become a list and the concat order configured
    # (deferred until a real checkpoint's actual output shape is known, same
    # as OpenPI's action_key caveat). Unused when libero_gr00t_action=True.
    action_key: str = "action"
    # Checkpoint-specific preprocessing verified for nvidia/GR00T-N1.7-LIBERO
    # specifically (see module docstring) -- not a generic feature, don't set
    # for other checkpoints without re-verifying against their own reference
    # eval code.
    flip_images_180: bool = False
    libero_gr00t_proprio: bool = False
    libero_gr00t_action: bool = False


def _flip_180(array):
    return array[::-1, ::-1]


def _libero_gr00t_state(proprio: dict) -> dict:
    eef_pos, axisangle, gripper_qpos = eef_pose_and_gripper(proprio)
    return {
        "state.x": [eef_pos[0]], "state.y": [eef_pos[1]], "state.z": [eef_pos[2]],
        "state.roll": [axisangle[0]], "state.pitch": [axisangle[1]], "state.yaw": [axisangle[2]],
        "state.gripper": gripper_qpos,
    }


def observation_to_gr00t(obs: dict, mapping: Gr00tObservationMapping) -> dict:
    """Our neutral observation -> the ``observation`` dict ``get_action()``
    expects."""
    payload: dict = {}
    for camera in obs.get("cameras") or []:
        their_key = mapping.camera_keys.get(camera["name"])
        if their_key is not None:
            array = camera["array"]
            payload[their_key] = _flip_180(array) if mapping.flip_images_180 else array

    proprio = obs.get("proprio") or {}
    if mapping.libero_gr00t_proprio:
        payload.update(_libero_gr00t_state(proprio))
    elif proprio.get("values"):
        payload[mapping.proprio_key] = proprio["values"]

    task = obs.get("task") or {}
    if task.get("language_instruction"):
        payload[mapping.prompt_key] = [task["language_instruction"]]

    return payload


def _normalize_and_invert_gripper(raw: float) -> float:
    """`[0,1] -> [-1,1]`, binarize, then flip sign -- exactly GR00T's own
    ``normalize_gripper_action`` + ``invert_gripper_action``
    (``gr00t/eval/sim/LIBERO/libero_env.py``), not invented here. Without
    this the gripper dimension is inverted and unbinarized relative to what
    ``sim/libero_eval.py``'s OSC_POSE controller expects."""
    normalized = 2.0 * (raw - 0.0) / (1.0 - 0.0) - 1.0
    binarized = 1.0 if normalized > 0 else (-1.0 if normalized < 0 else 0.0)
    return binarized * -1.0


def _libero_gr00t_action_to_chunk(action: dict) -> list[list[float]]:
    columns = []
    for key in _LIBERO_ACTION_KEYS:
        if key not in action:
            raise KeyError(f"gr00t response missing {key!r}; keys present: {sorted(action)}")
        columns.append([float(v) for v in action[key]])
    if "action.gripper" not in action:
        raise KeyError(f"gr00t response missing 'action.gripper'; keys present: {sorted(action)}")
    gripper = [_normalize_and_invert_gripper(float(v)) for v in action["action.gripper"]]

    horizon = len(columns[0])
    if any(len(col) != horizon for col in columns) or len(gripper) != horizon:
        raise ValueError(
            f"libero_gr00t_action expects all seven action.* keys to share one horizon; "
            f"got lengths {[len(c) for c in columns]} and gripper={len(gripper)}"
        )
    return [[col[t] for col in columns] + [gripper[t]] for t in range(horizon)]


def gr00t_action_to_chunk(action: dict, mapping: Gr00tObservationMapping) -> list[list[float]]:
    """The action dict half of ``get_action()``'s ``(action, info)`` return
    -> our ``action_chunk`` wire shape."""
    if mapping.libero_gr00t_action:
        return _libero_gr00t_action_to_chunk(action)
    if mapping.action_key not in action:
        raise KeyError(
            f"gr00t response missing {mapping.action_key!r}; keys present: {sorted(action)}"
        )
    chunk = action[mapping.action_key]
    return [[float(v) for v in row] for row in chunk]
