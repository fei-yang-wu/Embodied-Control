"""Neutral observation <-> GR00T ``get_action`` dict translation.

Pure functions, no network -- see ``openpi_translate.py``'s docstring for why
this stays separate from ``gr00t_client.py`` (transport is fixed per server
software; key names are fixed per trained checkpoint). ``Gr00tObservationMapping``'s
defaults are placeholders, not verified against any specific checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # as OpenPI's action_key caveat).
    action_key: str = "action"


def observation_to_gr00t(obs: dict, mapping: Gr00tObservationMapping) -> dict:
    """Our neutral observation -> the ``observation`` dict ``get_action()``
    expects."""
    payload: dict = {}
    for camera in obs.get("cameras") or []:
        their_key = mapping.camera_keys.get(camera["name"])
        if their_key is not None:
            payload[their_key] = camera["array"]

    proprio = obs.get("proprio") or {}
    if proprio.get("values"):
        payload[mapping.proprio_key] = proprio["values"]

    task = obs.get("task") or {}
    if task.get("language_instruction"):
        payload[mapping.prompt_key] = [task["language_instruction"]]

    return payload


def gr00t_action_to_chunk(action: dict, mapping: Gr00tObservationMapping) -> list[list[float]]:
    """The action dict half of ``get_action()``'s ``(action, info)`` return
    -> our ``action_chunk`` wire shape."""
    if mapping.action_key not in action:
        raise KeyError(
            f"gr00t response missing {mapping.action_key!r}; keys present: {sorted(action)}"
        )
    chunk = action[mapping.action_key]
    return [[float(v) for v in row] for row in chunk]
