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

An observation MAY carry a ``cameras`` list alongside ``proprio``/``task``:
``[{name, encoding, shape, dtype, data}]``, where ``data`` is base64-encoded
raw bytes (``encoding: "rgb8"`` for now -- JPEG/other encodings are a later
optimization, not needed at the payload sizes this milestone deals with: a
128x128x3 uint8 frame is ~65KB base64-encoded, trivial over plain HTTP/JSON).
``encode_camera``/``decode_image_bytes`` stay stdlib-only (no numpy import
here) so this module keeps working inside the tiny policy container -- callers
that HAVE numpy (e.g. the LIBERO harness) pass arrays in; callers that don't
(e.g. a policy reading images back out) can still compute real statistics
directly over raw bytes (see ``mean_brightness``).
"""

from __future__ import annotations

import base64

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


def encode_camera(name: str, array, encoding: str = "rgb8") -> dict:
    """Build a wire camera entry from an array-like object (numpy or similar --
    only ``.shape``/``.dtype``/``.tobytes()`` are used, so this function itself
    never needs numpy installed)."""
    return {
        "name": name,
        "encoding": encoding,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def decode_image_bytes(camera: dict) -> bytes:
    return base64.b64decode(camera["data"])


def mean_brightness(raw: bytes) -> float:
    """Mean byte value in [0, 255] -- stdlib-only real image statistic (no
    numpy needed): treats the raw bytes as an unstructured pile of uint8
    samples, which for an ``rgb8`` frame is exactly the per-channel pixel
    values. Used by ``ImageStatsPolicy`` to prove it decoded real image data,
    not just that a ``cameras`` field was present."""
    return sum(raw) / len(raw) if raw else 0.0
