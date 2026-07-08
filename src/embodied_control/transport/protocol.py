"""Wire message helpers for the policy protocol (stdlib only).

The protocol has four operations, mirroring the design's PolicyService:

- ``GET  /health``   -> liveness + model-loaded
- ``POST /describe`` -> model identity, action dim, schema ids, max horizon
- ``POST /reset``    -> clear per-episode state for the given episode keys
- ``POST /act``      -> map observations to action chunks

Messages are plain JSON dicts. We keep small builders/validators here rather than
a schema library so the container image needs no third-party dependencies.

An ``EpisodeKey`` is the tuple ``(env_id, episode_id)`` serialized as a 2-list.
The rollout driver (host) MUST call ``/reset`` for an episode key before the
first ``/act`` that references it; ``/act`` on an unknown key is an error.
"""

from __future__ import annotations

PROTOCOL_VERSION = "ec.policy/v1alpha1"


def episode_key(env_id: int, episode_id: int) -> list[int]:
    return [int(env_id), int(episode_id)]


def key_str(key) -> str:
    return f"{int(key[0])}:{int(key[1])}"


def build_act_request(request_id, episode_keys, observations, requested_horizon):
    """observations: list of {env_id, episode_id, proprio: {names, values}, task}."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "episode_keys": episode_keys,
        "requested_horizon": int(requested_horizon),
        "observations": observations,
    }


def build_reset_request(episode_keys, seed):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "episode_keys": episode_keys,
        "seed": int(seed),
    }
