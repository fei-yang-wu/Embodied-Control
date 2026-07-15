"""OpenPI transport adapter: translation (pure functions) + real websocket
round-trip against a fake OpenPI-protocol server. Requires websockets/msgpack/
numpy (the `transports` pixi feature) -- run with `pixi run -e transports
test-transports`, not covered by the light default `pixi run test`."""

from __future__ import annotations

import json
import math

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("websockets")
pytest.importorskip("msgpack_numpy")

from embodied_control.transport.client import PolicyClientError  # noqa: E402
from embodied_control.transport.openpi_client import OpenPIWebsocketClient  # noqa: E402
from embodied_control.transport.openpi_translate import (  # noqa: E402
    OpenPIObservationMapping,
    _quat2axisangle,
    observation_to_openpi,
    openpi_action_to_chunk,
)

from fake_openpi_server import FakeOpenPIServer  # noqa: E402

from embodied_control.config.schemas import (  # noqa: E402
    EndpointSpec,
    EvalJob,
    OutputSpec,
    PolicyBinding,
    RolloutSpec,
    RuntimeSpec,
    SimSpec,
)
from embodied_control.orchestration.runner import run_eval  # noqa: E402


# --- translation (pure functions, no server needed) ----------------------

def test_observation_to_openpi_maps_cameras_proprio_and_prompt():
    mapping = OpenPIObservationMapping(
        camera_keys={"agentview_image": "observation/image"},
        proprio_key="observation/state",
        prompt_key="prompt",
    )
    obs = {
        "env_id": 0, "episode_id": 0,
        "proprio": {"names": ["x", "y"], "values": [1.0, 2.0]},
        "cameras": [{"name": "agentview_image", "array": np.zeros((4, 4, 3), dtype=np.uint8)}],
        "task": {"task_id": "t", "language_instruction": "pick up the cup"},
    }
    payload = observation_to_openpi(obs, mapping)
    # a real ndarray, not a plain list -- verified live against a real
    # server that its normalization pipeline requires .shape (see
    # openpi_translate.py's comment)
    assert isinstance(payload["observation/state"], np.ndarray)
    assert payload["observation/state"].tolist() == [1.0, 2.0]
    assert payload["prompt"] == "pick up the cup"
    assert payload["observation/image"].shape == (4, 4, 3)


def test_observation_to_openpi_skips_unmapped_cameras():
    mapping = OpenPIObservationMapping(camera_keys={})  # nothing mapped
    obs = {"cameras": [{"name": "wrist_cam", "array": np.zeros((2, 2))}]}
    payload = observation_to_openpi(obs, mapping)
    assert payload == {}


def test_quat2axisangle_identity_quaternion_is_zero_rotation():
    # xyzw identity quaternion -> zero rotation
    assert _quat2axisangle([0.0, 0.0, 0.0, 1.0]) == [0.0, 0.0, 0.0]


def test_quat2axisangle_matches_robosuite_reference_values():
    # 90-degree rotation about the z-axis: xyzw = (0, 0, sin(45deg), cos(45deg))
    result = _quat2axisangle([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])
    assert result[0] == pytest.approx(0.0, abs=1e-9)
    assert result[1] == pytest.approx(0.0, abs=1e-9)
    assert result[2] == pytest.approx(math.pi / 2, abs=1e-9)  # 90 degrees in radians


def test_observation_to_openpi_libero_pi0_proprio_builds_verified_8dim_state():
    # Verified against OpenPI's own examples/libero/main.py: concat(eef_pos,
    # quat2axisangle(eef_quat), gripper_qpos) = 8 floats, exactly this order.
    mapping = OpenPIObservationMapping(libero_pi0_proprio=True)
    obs = {
        "proprio": {
            "names": [
                "robot0_eef_pos[0]", "robot0_eef_pos[1]", "robot0_eef_pos[2]",
                "robot0_eef_quat[0]", "robot0_eef_quat[1]", "robot0_eef_quat[2]", "robot0_eef_quat[3]",
                "robot0_gripper_qpos[0]", "robot0_gripper_qpos[1]",
                "unrelated_field[0]",  # must be ignored, not swept into the state vector
            ],
            "values": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.04, -0.04, 999.0],
        },
    }
    state = observation_to_openpi(obs, mapping)["observation/state"]
    assert isinstance(state, np.ndarray)
    assert len(state) == 8
    assert state[:3].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert state[3:6].tolist() == pytest.approx([0.0, 0.0, 0.0])  # identity quat -> zero axis-angle
    assert state[6:].tolist() == pytest.approx([0.04, -0.04])


def test_observation_to_openpi_libero_pi0_proprio_raises_on_missing_fields():
    mapping = OpenPIObservationMapping(libero_pi0_proprio=True)
    obs = {"proprio": {"names": ["some_other_field"], "values": [1.0]}}
    with pytest.raises(ValueError, match="robot0_eef_pos"):
        observation_to_openpi(obs, mapping)


def test_observation_to_openpi_flips_images_180_when_enabled():
    mapping = OpenPIObservationMapping(
        camera_keys={"agentview_image": "observation/image"}, flip_images_180=True,
    )
    array = np.arange(9).reshape(3, 3, 1)
    obs = {"cameras": [{"name": "agentview_image", "array": array}]}
    result = observation_to_openpi(obs, mapping)["observation/image"]
    assert np.array_equal(result, array[::-1, ::-1])
    assert not np.array_equal(result, array)  # sanity: this array isn't 180-rotation-symmetric


def test_openpi_action_to_chunk_converts_ndarray_to_float_lists():
    mapping = OpenPIObservationMapping(action_key="actions")
    action = {"actions": np.array([[0.1, 0.2], [0.3, 0.4]])}
    chunk = openpi_action_to_chunk(action, mapping)
    assert chunk == [[0.1, 0.2], [0.3, 0.4]]
    assert all(isinstance(v, float) for row in chunk for v in row)


def test_openpi_action_to_chunk_raises_on_missing_key():
    mapping = OpenPIObservationMapping(action_key="actions")
    with pytest.raises(KeyError):
        openpi_action_to_chunk({"wrong_key": []}, mapping)


# --- real wire round-trip against a fake OpenPI-protocol server -----------

def _echo_infer(obs: dict) -> dict:
    # Chunk length 3, values derived from the state so the test can assert
    # real content flowed through, not just "some response came back".
    state = obs.get("observation/state", [0.0])
    return {"actions": np.array([[state[0] + i for _ in range(2)] for i in range(3)])}


def test_act_round_trips_real_msgpack_frames():
    server = FakeOpenPIServer(_echo_infer, metadata={"policy_name": "fake-pi0"})
    server.start_background()
    try:
        client = OpenPIWebsocketClient(
            "127.0.0.1", server.port, action_dim=2,
            mapping=OpenPIObservationMapping(proprio_key="observation/state"),
            timeout_s=5.0,
        )
        health = client.wait_healthy(timeout_s=5.0)
        assert health["status"] == "ok"

        desc = client.describe()
        assert desc["action_dim"] == 2
        assert desc["server_metadata"] == {"policy_name": "fake-pi0"}

        client.reset([[0, 0]], seed=7)  # no-op on the wire, must not raise

        obs = {"env_id": 0, "episode_id": 0, "proprio": {"names": ["x"], "values": [10.0]}}
        resp = client.act("req-1", [[0, 0]], [obs], requested_horizon=3)
        assert resp["status"] == "ok"
        chunk = resp["actions"][0]["action_chunk"]
        assert chunk == [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]]
        assert "total_ms" in resp["timing"]
    finally:
        server.shutdown()


def test_act_raises_policy_client_error_on_server_exception():
    def _raising_infer(obs):
        raise ValueError("boom")

    server = FakeOpenPIServer(_raising_infer)
    server.start_background()
    try:
        client = OpenPIWebsocketClient("127.0.0.1", server.port, action_dim=2, timeout_s=5.0)
        client.wait_healthy(timeout_s=5.0)
        with pytest.raises(PolicyClientError, match="boom"):
            client.act("req-1", [[0, 0]], [{"env_id": 0, "episode_id": 0}], requested_horizon=1)
    finally:
        server.shutdown()


def test_act_rejects_multi_env_batches():
    server = FakeOpenPIServer(_echo_infer)
    server.start_background()
    try:
        client = OpenPIWebsocketClient("127.0.0.1", server.port, action_dim=2, timeout_s=5.0)
        client.wait_healthy(timeout_s=5.0)
        with pytest.raises(PolicyClientError, match="exactly one"):
            client.act("req-1", [[0, 0], [0, 1]], [{"env_id": 0}, {"env_id": 1}], requested_horizon=1)
    finally:
        server.shutdown()


def test_wait_healthy_raises_when_unreachable():
    client = OpenPIWebsocketClient("127.0.0.1", 1, action_dim=2, timeout_s=0.5)  # nothing listens on port 1
    with pytest.raises(PolicyClientError):
        client.wait_healthy(timeout_s=1.0)


# --- end-to-end: run_eval() against a real openpi_websocket external endpoint ---

def test_run_eval_against_external_openpi_endpoint_produces_full_artifacts(tmp_path):
    """Proves the whole plumbing (planner -> supervisor -> factory -> client ->
    runner -> manifest), not just the client in isolation -- the M2 acceptance
    criterion from docs/design/real_policy_adapters.md."""
    server = FakeOpenPIServer(_echo_infer, metadata={"checkpoint": "fake-pi0-v0"})
    server.start_background()
    try:
        job = EvalJob(
            name="test_openpi_external",
            seed=1,
            sim=SimSpec(
                backend="fake_delegated", mode="delegated", action_dim=2,
                runtime=RuntimeSpec(type="local"), timeout_s=30,
                backend_config={"task_id": "openpi_smoke_task"},
            ),
            policy=PolicyBinding(
                endpoint=EndpointSpec(
                    scheme="openpi_websocket", host="127.0.0.1", port=server.port,
                    action_dim=2, observation_mapping={"proprio_key": "observation/state"},
                ),
                requested_action_horizon=3,
            ),
            rollout=RolloutSpec(num_episodes=1, max_steps_per_episode=6),
            outputs=OutputSpec(root_dir=str(tmp_path / "runs")),
        )
        result = run_eval(job)
    finally:
        server.shutdown()

    assert result.status == "succeeded", result
    assert result.metrics.num_episodes_completed == 1

    run_dir = tmp_path / "runs" / result.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schemas"]["policy_endpoint_scheme"] == "openpi_websocket"
    assert manifest["policy_describe"]["action_dim"] == 2
    assert manifest["policy_describe"]["server_metadata"] == {"checkpoint": "fake-pi0-v0"}
