"""GR00T transport adapter: translation (pure functions) + real ZeroMQ
round-trip against a fake GR00T-protocol server. Requires pyzmq/msgpack/numpy
(the `transports` pixi feature) -- run with `pixi run -e transports
test-transports`, not covered by the light default `pixi run test`."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("zmq")
pytest.importorskip("msgpack_numpy")

from embodied_control.transport.client import PolicyClientError  # noqa: E402
from embodied_control.transport.gr00t_client import Gr00tZmqClient  # noqa: E402
from embodied_control.transport.gr00t_translate import (  # noqa: E402
    Gr00tObservationMapping,
    gr00t_action_to_chunk,
    observation_to_gr00t,
)

from fake_gr00t_server import FakeGr00tServer  # noqa: E402

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

def test_observation_to_gr00t_maps_cameras_proprio_and_prompt():
    mapping = Gr00tObservationMapping(
        camera_keys={"agentview_image": "video.ego_view"},
        proprio_key="state.joint_position",
        prompt_key="annotation.human.action.task_description",
    )
    obs = {
        "env_id": 0, "episode_id": 0,
        "proprio": {"names": ["x", "y"], "values": [1.0, 2.0]},
        "cameras": [{"name": "agentview_image", "array": np.zeros((4, 4, 3), dtype=np.uint8)}],
        "task": {"task_id": "t", "language_instruction": "pick up the cup"},
    }
    payload = observation_to_gr00t(obs, mapping)
    assert payload["state.joint_position"] == [1.0, 2.0]
    assert payload["annotation.human.action.task_description"] == ["pick up the cup"]
    assert payload["video.ego_view"].shape == (4, 4, 3)


def test_gr00t_action_to_chunk_converts_ndarray_to_float_lists():
    mapping = Gr00tObservationMapping(action_key="action")
    action = {"action": np.array([[0.1, 0.2], [0.3, 0.4]])}
    chunk = gr00t_action_to_chunk(action, mapping)
    assert chunk == [[0.1, 0.2], [0.3, 0.4]]


def test_gr00t_action_to_chunk_raises_on_missing_key():
    mapping = Gr00tObservationMapping(action_key="action")
    with pytest.raises(KeyError):
        gr00t_action_to_chunk({"wrong_key": []}, mapping)


# --- libero_gr00t_proprio / libero_gr00t_action presets (nvidia/GR00T-N1.7-LIBERO) ---

def test_observation_to_gr00t_libero_preset_splits_proprio_into_seven_keys():
    # Verified against gr00t/eval/sim/LIBERO/libero_env.py::_process_observation --
    # NOT one array like OpenPI's checkpoint, seven individual state.* keys.
    mapping = Gr00tObservationMapping(libero_gr00t_proprio=True)
    obs = {
        "proprio": {
            "names": [
                "robot0_eef_pos[0]", "robot0_eef_pos[1]", "robot0_eef_pos[2]",
                "robot0_eef_quat[0]", "robot0_eef_quat[1]", "robot0_eef_quat[2]", "robot0_eef_quat[3]",
                "robot0_gripper_qpos[0]", "robot0_gripper_qpos[1]",
            ],
            "values": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.04, -0.04],
        },
    }
    payload = observation_to_gr00t(obs, mapping)
    # (B=1, T=1, D) float32 -- Gr00tSimPolicyWrapper.check_observation
    # requires this exact shape/dtype, verified live against a real server
    # (see gr00t_translate.py's module docstring).
    for key in ("state.x", "state.y", "state.z", "state.roll", "state.pitch", "state.yaw"):
        assert payload[key].shape == (1, 1, 1)
        assert payload[key].dtype == np.float32
    assert payload["state.gripper"].shape == (1, 1, 2)
    assert payload["state.x"][0, 0, 0] == pytest.approx(0.1)
    assert payload["state.y"][0, 0, 0] == pytest.approx(0.2)
    assert payload["state.z"][0, 0, 0] == pytest.approx(0.3)
    assert payload["state.roll"][0, 0, 0] == pytest.approx(0.0)
    assert payload["state.pitch"][0, 0, 0] == pytest.approx(0.0)
    assert payload["state.yaw"][0, 0, 0] == pytest.approx(0.0)
    assert payload["state.gripper"][0, 0].tolist() == pytest.approx([0.04, -0.04])


def test_libero_gr00t_action_to_chunk_concatenates_and_transforms_gripper():
    # Verified against gr00t/eval/sim/LIBERO/libero_env.py::step +
    # normalize_gripper_action/invert_gripper_action -- gripper needs
    # [0,1]->[-1,1], binarize, then sign flip; the other six dims pass through.
    mapping = Gr00tObservationMapping(libero_gr00t_action=True)
    action = {
        "action.x": [0.1, 0.2], "action.y": [0.0, 0.0], "action.z": [0.0, 0.0],
        "action.roll": [0.0, 0.0], "action.pitch": [0.0, 0.0], "action.yaw": [0.0, 0.0],
        "action.gripper": [1.0, 0.0],  # 1.0 (open, raw) -> normalize/binarize/-1 -> -1.0
                                        # 0.0 (close, raw) -> ... -> +1.0
    }
    chunk = gr00t_action_to_chunk(action, mapping)
    assert chunk == [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


def test_libero_gr00t_action_to_chunk_raises_on_missing_key():
    mapping = Gr00tObservationMapping(libero_gr00t_action=True)
    action = {"action.x": [0.1]}  # missing the rest
    with pytest.raises(KeyError):
        gr00t_action_to_chunk(action, mapping)


def test_observation_to_gr00t_flips_images_180_when_enabled():
    mapping = Gr00tObservationMapping(camera_keys={"agentview_image": "video.image"}, flip_images_180=True)
    array = np.arange(9).reshape(3, 3, 1)
    obs = {"cameras": [{"name": "agentview_image", "array": array}]}
    result = observation_to_gr00t(obs, mapping)["video.image"]
    assert np.array_equal(result, array[::-1, ::-1])


def test_observation_to_gr00t_libero_preset_batches_video_with_flip():
    # Gr00tSimPolicyWrapper.check_observation requires (B=1, T=1, H, W, 3)
    # uint8 for every video.* key -- verified live against a real server.
    mapping = Gr00tObservationMapping(
        camera_keys={"agentview_image": "video.image"},
        flip_images_180=True, libero_gr00t_proprio=True,
    )
    array = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    obs = {
        "cameras": [{"name": "agentview_image", "array": array}],
        "proprio": {
            "names": [
                "robot0_eef_pos[0]", "robot0_eef_pos[1]", "robot0_eef_pos[2]",
                "robot0_eef_quat[0]", "robot0_eef_quat[1]", "robot0_eef_quat[2]", "robot0_eef_quat[3]",
                "robot0_gripper_qpos[0]", "robot0_gripper_qpos[1]",
            ],
            "values": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0, 0.04, -0.04],
        },
    }
    result = observation_to_gr00t(obs, mapping)["video.image"]
    assert result.shape == (1, 1, 3, 3, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result[0, 0], array[::-1, ::-1])


# --- real wire round-trip against a fake GR00T-protocol server ------------

def _get_action(observation, options=None):
    state = observation.get("state.joint_position", [0.0])
    chunk = np.array([[state[0] + i for _ in range(2)] for i in range(3)])
    return [{"action": chunk}, {"server_timing_ms": 0.0}]


def _get_modality_config():
    # Real marker/shape verified live against gr00t/policy/server_client.py::
    # MsgSerializer -- "__ModalityConfig_class__" (not "__ModalityConfig__"),
    # "as_json" a plain dict (not a JSON string).
    return {"__ModalityConfig_class__": True, "as_json": {"video.ego_view": {"shape": [224, 224, 3]}}}


def test_act_round_trips_real_msgpack_frames():
    server = FakeGr00tServer({
        "get_action": (True, _get_action),
        "reset": (True, lambda options=None: {"status": "ok"}),
        "get_modality_config": (False, _get_modality_config),
    })
    server.start_background()
    try:
        client = Gr00tZmqClient(
            "127.0.0.1", server.port, action_dim=2,
            mapping=Gr00tObservationMapping(proprio_key="state.joint_position"),
            timeout_s=5.0,
        )
        health = client.wait_healthy(timeout_s=5.0)
        assert health["status"] == "ok"

        desc = client.describe()
        assert desc["action_dim"] == 2
        assert desc["modality_config"] == {"video.ego_view": {"shape": [224, 224, 3]}}

        client.reset([[0, 0]], seed=7)

        obs = {"env_id": 0, "episode_id": 0, "proprio": {"names": ["x"], "values": [10.0]}}
        resp = client.act("req-1", [[0, 0]], [obs], requested_horizon=3)
        assert resp["status"] == "ok"
        chunk = resp["actions"][0]["action_chunk"]
        assert chunk == [[10.0, 10.0], [11.0, 11.0], [12.0, 12.0]]
        assert "total_ms" in resp["timing"]
    finally:
        server.shutdown()


def test_act_raises_policy_client_error_on_server_exception():
    def _raising_get_action(observation, options=None):
        raise ValueError("boom")

    server = FakeGr00tServer({"get_action": (True, _raising_get_action)})
    server.start_background()
    try:
        client = Gr00tZmqClient("127.0.0.1", server.port, action_dim=2, timeout_s=5.0)
        client.wait_healthy(timeout_s=5.0)
        with pytest.raises(PolicyClientError, match="boom"):
            client.act("req-1", [[0, 0]], [{"env_id": 0, "episode_id": 0}], requested_horizon=1)
    finally:
        server.shutdown()


def test_act_rejects_multi_env_batches():
    server = FakeGr00tServer({"get_action": (True, _get_action)})
    server.start_background()
    try:
        client = Gr00tZmqClient("127.0.0.1", server.port, action_dim=2, timeout_s=5.0)
        client.wait_healthy(timeout_s=5.0)
        with pytest.raises(PolicyClientError, match="exactly one"):
            client.act("req-1", [[0, 0], [0, 1]], [{"env_id": 0}, {"env_id": 1}], requested_horizon=1)
    finally:
        server.shutdown()


def test_wait_healthy_raises_when_unreachable():
    client = Gr00tZmqClient("127.0.0.1", 1, action_dim=2, timeout_s=0.5)
    with pytest.raises(PolicyClientError):
        client.wait_healthy(timeout_s=1.0)


# --- end-to-end: run_eval() against a real gr00t_zmq external endpoint ----

def test_run_eval_against_external_gr00t_endpoint_produces_full_artifacts(tmp_path):
    server = FakeGr00tServer({
        "get_action": (True, _get_action),
        "reset": (True, lambda options=None: {"status": "ok"}),
        "get_modality_config": (False, _get_modality_config),
    })
    server.start_background()
    try:
        job = EvalJob(
            name="test_gr00t_external",
            seed=1,
            sim=SimSpec(
                backend="fake_delegated", mode="delegated", action_dim=2,
                runtime=RuntimeSpec(type="local"), timeout_s=30,
                backend_config={"task_id": "gr00t_smoke_task"},
            ),
            policy=PolicyBinding(
                endpoint=EndpointSpec(
                    scheme="gr00t_zmq", host="127.0.0.1", port=server.port,
                    action_dim=2, observation_mapping={"proprio_key": "state.joint_position"},
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
    assert manifest["schemas"]["policy_endpoint_scheme"] == "gr00t_zmq"
    assert manifest["policy_describe"]["action_dim"] == 2
