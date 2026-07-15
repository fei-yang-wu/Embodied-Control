"""Neutral observation <-> GR00T ``get_action`` dict translation.

Pure functions, no network -- see ``openpi_translate.py``'s docstring for why
this stays separate from ``gr00t_client.py`` (transport is fixed per server
software; key names are fixed per trained checkpoint). ``Gr00tObservationMapping``'s
defaults are placeholders, not verified against any specific checkpoint.

``libero_gr00t_proprio``/``libero_gr00t_action``/``flip_images_180`` encode
checkpoint-specific preprocessing verified against GR00T's own reference
LIBERO env wrapper (``gr00t/eval/sim/LIBERO/libero_env.py``,
NVIDIA/Isaac-GR00T, 2026-07-15) for the public `nvidia/GR00T-N1.7-LIBERO`
checkpoint, served with ``--use-sim-policy-wrapper`` (``gr00t/policy/
gr00t_policy.py::Gr00tSimPolicyWrapper``). **Structurally different from
OpenPI's LIBERO preset, not just a renamed copy**: GR00T-LIBERO splits
proprioception into seven individual ``state.x``/``state.y``/``state.z``/
``state.roll``/``state.pitch``/``state.yaw``/``state.gripper`` keys (not one
``observation/state`` array), each a real ``float32`` ndarray with explicit
batch+time dims -- ``Gr00tSimPolicyWrapper.check_observation`` asserts
``ndim==3``, shape ``(B, T, D)``, dtype ``float32`` for every state key, and
``ndim==5``, shape ``(B, T, H, W, 3)``, dtype ``uint8`` for every video key
(``B=T=1`` here -- one env, one current-timestep observation, no history).
A Python list/scalar or a mismatched dtype fails validation server-side with
an ``AssertionError``, not a silent wrong-shape bug -- verified live against
the real server, not assumed from source alone. The action side is the
mirror image -- seven separate batched ``action.*`` keys in the response
(shape roughly ``(1, horizon, 1)``, squeezed here via a flat reshape rather
than assumed exact), concatenated back into a single 7-vector, with the
gripper dimension needing a `[0,1]->[-1,1]` normalize-and-binarize plus a
sign flip (`normalize_gripper_action`/`invert_gripper_action` in GR00T's own
``libero_env.py`` -- copied here, not invented) before it matches what
`sim/libero_eval.py`'s OSC_POSE controller expects. Images are flipped 180
degrees, same as OpenPI's checkpoint and for the same reason (LIBERO/
robosuite rendering vs. training-time preprocessing convention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

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


def _batch_state(values: list[float]) -> np.ndarray:
    """(D,) -> (B=1, T=1, D) float32 -- Gr00tSimPolicyWrapper.check_observation
    asserts ndim==3 and dtype float32 for every state.* key (see module
    docstring)."""
    return np.asarray(values, dtype=np.float32).reshape(1, 1, -1)


def _batch_video(array) -> np.ndarray:
    """(H, W, C) -> (B=1, T=1, H, W, C) uint8 -- Gr00tSimPolicyWrapper.
    check_observation asserts ndim==5 and dtype uint8 for every video.* key."""
    return np.asarray(array, dtype=np.uint8)[np.newaxis, np.newaxis]


def _libero_gr00t_state(proprio: dict) -> dict:
    eef_pos, axisangle, gripper_qpos = eef_pose_and_gripper(proprio)
    return {
        "state.x": _batch_state([eef_pos[0]]), "state.y": _batch_state([eef_pos[1]]),
        "state.z": _batch_state([eef_pos[2]]),
        "state.roll": _batch_state([axisangle[0]]), "state.pitch": _batch_state([axisangle[1]]),
        "state.yaw": _batch_state([axisangle[2]]),
        "state.gripper": _batch_state(gripper_qpos),
    }


def observation_to_gr00t(obs: dict, mapping: Gr00tObservationMapping) -> dict:
    """Our neutral observation -> the ``observation`` dict ``get_action()``
    expects."""
    payload: dict = {}
    for camera in obs.get("cameras") or []:
        their_key = mapping.camera_keys.get(camera["name"])
        if their_key is not None:
            array = camera["array"]
            if mapping.flip_images_180:
                array = _flip_180(array)
            payload[their_key] = _batch_video(array) if mapping.libero_gr00t_proprio else array

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


def _unbatch(value) -> list[float]:
    """Flatten whatever batch/time/feature dims Gr00tSimPolicyWrapper's
    response carries (roughly (B=1, horizon, D=1), not asserted exact here
    since the wrapper's own check_action docstring only commits to ndim==3,
    not precise sizes) down to the flat per-horizon-step sequence -- correct
    as long as B=1 and D=1 per key, true for this single-env, scalar-per-key
    LIBERO action space."""
    return np.asarray(value, dtype=np.float64).reshape(-1).tolist()


def _libero_gr00t_action_to_chunk(action: dict) -> list[list[float]]:
    columns = []
    for key in _LIBERO_ACTION_KEYS:
        if key not in action:
            raise KeyError(f"gr00t response missing {key!r}; keys present: {sorted(action)}")
        columns.append(_unbatch(action[key]))
    if "action.gripper" not in action:
        raise KeyError(f"gr00t response missing 'action.gripper'; keys present: {sorted(action)}")
    gripper = [_normalize_and_invert_gripper(v) for v in _unbatch(action["action.gripper"])]

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
