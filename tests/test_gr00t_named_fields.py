"""Named multi-dimensional contracts used by G1 policies."""
import pytest

np = pytest.importorskip("numpy")
from embodied_control.transport.gr00t_translate import (
    Gr00tObservationMapping, observation_to_gr00t, gr00t_action_to_chunk,
)


def mapping():
    return Gr00tObservationMapping(
        camera_keys={"ego": "video.ego_view"}, batched=True,
        state_keys={"arm": "state.left_arm"}, state_dims={"arm": 7},
        action_dims={"action.arm": 7, "action.height": 1},
    )


def test_named_states_and_camera_history():
    obs = {"cameras": [{"name": "ego", "array": np.zeros((2, 8, 9, 3), np.uint8)}],
           "proprio": {"arm": list(range(7))}}
    result = observation_to_gr00t(obs, mapping())
    assert result["video.ego_view"].shape == (1, 2, 8, 9, 3)
    np.testing.assert_array_equal(result["state.left_arm"], np.arange(7).reshape(1, 1, 7))
    obs["proprio"]["arm"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        observation_to_gr00t(obs, mapping())


def test_action_fields_preserve_time_and_configured_order():
    arm = np.arange(21).reshape(1, 3, 7)
    height = np.array([[[0.7], [0.8], [0.9]]])
    result = gr00t_action_to_chunk({"action.height": height, "action.arm": arm}, mapping())
    np.testing.assert_allclose(result, np.concatenate([arm, height], axis=2)[0])
    with pytest.raises(ValueError, match="horizon"):
        gr00t_action_to_chunk({"action.arm": arm, "action.height": height[:, :2]}, mapping())
    with pytest.raises(ValueError, match="shape"):
        gr00t_action_to_chunk({"action.arm": arm[0], "action.height": height}, mapping())


def test_missing_camera_is_rejected():
    with pytest.raises(ValueError, match="Missing GR00T cameras"):
        observation_to_gr00t({"proprio": {"arm": [0] * 7}}, mapping())


def test_current_server_numeric_wire_format():
    import msgpack
    from embodied_control.transport.gr00t_msgpack import from_bytes
    expected = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    envelope = {b"nd": True, b"type": expected.dtype.str, b"kind": b"",
                b"shape": expected.shape, b"data": expected.tobytes()}
    np.testing.assert_array_equal(from_bytes(msgpack.packb(envelope)), expected)
    envelope[b"kind"] = b"O"
    with pytest.raises(ValueError, match="numeric"):
        from_bytes(msgpack.packb(envelope))
    assert from_bytes(msgpack.packb({"__ModalityConfig__": True,
                                    "as_json": {"delta_indices": [0]}})) == {"delta_indices": [0]}
