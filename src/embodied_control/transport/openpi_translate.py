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
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OpenPIObservationMapping:
    # our camera name (e.g. "agentview_image") -> the key the served
    # checkpoint expects (e.g. "observation/image")
    camera_keys: dict[str, str] = field(default_factory=dict)
    proprio_key: str = "observation/state"
    prompt_key: str = "prompt"
    # key in the dict returned by infer() holding the action chunk
    action_key: str = "actions"


def observation_to_openpi(obs: dict, mapping: OpenPIObservationMapping) -> dict:
    """Our neutral observation (``env_id``/``episode_id``/``proprio``/
    ``cameras``/``task``) -> the ``obs`` dict ``infer()`` expects."""
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
