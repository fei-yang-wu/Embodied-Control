"""Camera observation wire protocol: encode/decode round-trip, and the
`image_stats` policy that proves a policy actually decodes real image bytes
from the observation (stdlib only; default env, no numpy/mujoco needed)."""

from __future__ import annotations

from embodied_control.policies.base import make_policy
from embodied_control.transport.protocol import (
    decode_image_bytes,
    encode_camera,
    episode_key,
    mean_brightness,
)


class _FakeArray:
    """Duck-typed stand-in for a numpy array: encode_camera only ever touches
    .shape/.dtype/.tobytes(), so a real numpy dependency isn't needed to test
    it (this module intentionally stays numpy-free, including in tests, since
    it must keep working inside the tiny stdlib-only policy container)."""

    def __init__(self, data: bytes, shape, dtype="uint8"):
        self._data = data
        self.shape = shape
        self.dtype = dtype

    def tobytes(self) -> bytes:
        return self._data


def test_encode_decode_round_trips_exact_bytes():
    raw = bytes([10, 20, 30, 40, 50, 60])
    camera = encode_camera("agentview_image", _FakeArray(raw, shape=(1, 2, 3)))
    assert camera["name"] == "agentview_image"
    assert camera["encoding"] == "rgb8"
    assert camera["shape"] == [1, 2, 3]
    assert camera["dtype"] == "uint8"
    assert decode_image_bytes(camera) == raw


def test_mean_brightness_matches_hand_computed_values():
    assert mean_brightness(bytes([0, 0, 0, 0])) == 0.0
    assert mean_brightness(bytes([255, 255, 255])) == 255.0
    assert mean_brightness(bytes([0, 100, 200])) == 100.0
    assert mean_brightness(b"") == 0.0


def test_image_stats_policy_falls_back_to_zero_without_a_camera():
    p = make_policy("image_stats", action_dim=3)
    p.reset({"episode_keys": [episode_key(0, 0)], "seed": 0})
    resp = p.act({"observations": [{"env_id": 0, "episode_id": 0}], "requested_horizon": 1})
    assert resp["status"] == "ok"
    assert resp["actions"][0]["action_chunk"] == [[0.0, 0.0, 0.0]]


def test_image_stats_policy_reacts_to_real_decoded_image_content():
    p = make_policy("image_stats", action_dim=2)
    p.reset({"episode_keys": [episode_key(0, 0)], "seed": 0})

    dark = encode_camera("agentview_image", _FakeArray(bytes([0] * 100), shape=(10, 10, 1)))
    bright = encode_camera("agentview_image", _FakeArray(bytes([255] * 100), shape=(10, 10, 1)))

    dark_resp = p.act({
        "observations": [{"env_id": 0, "episode_id": 0, "cameras": [dark]}],
        "requested_horizon": 1,
    })
    bright_resp = p.act({
        "observations": [{"env_id": 0, "episode_id": 0, "cameras": [bright]}],
        "requested_horizon": 1,
    })

    dark_action = dark_resp["actions"][0]["action_chunk"][0]
    bright_action = bright_resp["actions"][0]["action_chunk"][0]

    # Real decoded content drives the action, not just presence of a camera:
    # an all-black frame maps to -1, an all-white frame to +1 (see
    # ImageStatsPolicy's [0,255] -> [-1,1] mapping), and both dims agree
    # since the policy applies the same signal uniformly.
    assert dark_action == [-1.0, -1.0]
    assert bright_action == [1.0, 1.0]
    assert dark_action != bright_action
