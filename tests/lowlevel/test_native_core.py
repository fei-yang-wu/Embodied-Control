"""C++ ONNX step_once parity for observation, inference, and action decode."""

import json
import os
from pathlib import Path
import threading
import time
import uuid

import numpy as np
import pytest

ec_native = pytest.importorskip("ec_native")
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from embodied_control.lowlevel.bundle import (  # noqa: E402
    ActionContract,
    ModelArtifact,
    PolicyBundle,
    sha256_file,
)
from embodied_control.lowlevel.contracts import RobotState  # noqa: E402
from embodied_control.lowlevel.native_core import (  # noqa: E402
    NativeFakeLoop,
    NativeMujocoLoop,
    NativeTracker,
    NativeUnitreeLoop,
    verify_native_bundle,
)
from embodied_control.lowlevel.publishers.native_pull import (  # noqa: E402
    NativeChunkWorker,
    NativeLatentPlanWorker,
)
from embodied_control.lowlevel.publishers.native_oracle import (  # noqa: E402
    NativeOracleWorker,
)
from embodied_control.cli import build_parser  # noqa: E402
from embodied_control.lowlevel.maths import (  # noqa: E402
    rotate_inverse,
    rot6d_from_quat,
    subtract_frame,
    quat_mul,
    quat_to_mat,
)


def _axis_quaternion(axis, degrees):
    half = np.radians(degrees) / 2
    result = np.zeros(4, dtype=np.float32)
    result[axis], result[3] = np.sin(half), np.cos(half)
    return result


@pytest.mark.parametrize("start_yaw", [-179.0, -90.0, 0.0, 88.0, 179.0])
@pytest.mark.parametrize("turn", [0.0, 35.0])
def test_reference_heading_preserves_tilt_and_relative_turn(start_yaw, turn):
    ref_heading = _axis_quaternion(2, 47)
    ref_start = quat_mul(ref_heading, _axis_quaternion(1, -8))
    initial = quat_mul(_axis_quaternion(2, start_yaw), _axis_quaternion(0, 12))
    current = quat_mul(
        _axis_quaternion(2, start_yaw + turn), _axis_quaternion(1, 19)
    )
    aligned = ec_native.align_heading_to_reference(initial, ref_start, current)
    expected = quat_mul(_axis_quaternion(2, 47 + turn), _axis_quaternion(1, 19))
    np.testing.assert_allclose(quat_to_mat(aligned), quat_to_mat(expected), atol=1e-6)
    np.testing.assert_allclose(
        ec_native.projected_gravity_from_xyzw(aligned),
        ec_native.projected_gravity_from_xyzw(current), atol=1e-6,
    )
    opposite_sign = ec_native.align_heading_to_reference(-initial, -ref_start, -current)
    np.testing.assert_allclose(quat_to_mat(opposite_sign), quat_to_mat(expected), atol=1e-6)

    # A forward offset in the reference must give the same encoder input
    # regardless of the IMU world's arbitrary starting yaw.
    origin = np.array([2.0, -3.0, 0.76], dtype=np.float32)
    raw = np.zeros((10, 36), dtype=np.float32)
    raw[:, 29:32] = origin + quat_to_mat(ref_heading) @ [0.2, 0.0, 0.03]
    raw[:, 32:36] = ref_start
    packed = ec_native.reexpress_root_qpos_window(raw, origin, aligned)
    expected_pos, expected_ori = subtract_frame(origin, expected, raw[0, 29:32], ref_start)
    np.testing.assert_allclose(packed[:, 29:32], np.tile(expected_pos, (10, 1)), atol=1e-6)
    np.testing.assert_allclose(packed[:, 32:38], np.tile(rot6d_from_quat(expected_ori), (10, 1)), atol=1e-6)
    joint_raw = np.zeros((10, 62), dtype=np.float32)
    joint_raw[:, 58:62] = ref_start
    np.testing.assert_allclose(
        ec_native.pack_joint_qpos_qvel_anchor_ori_window(joint_raw, 0, 10, 1, aligned),
        ec_native.pack_joint_qpos_qvel_anchor_ori_window(joint_raw, 0, 10, 1, expected),
        atol=1e-6,
    )


@pytest.mark.parametrize("bad", [[0, 0, 0, 0], [float("nan"), 0, 0, 1], [1, 0, 0, 0]])
def test_reference_heading_rejects_undefined_initial_heading(bad):
    with pytest.raises(RuntimeError, match="heading alignment quaternion is invalid"):
        ec_native.align_heading_to_reference(bad, [0, 0, 0, 1], [0, 0, 0, 1])


def _g1_mjcf_path() -> Path:
    mjcf = (
        Path(__file__).resolve().parents[2]
        / "assets/latent_playkit/model/g1_29dof_rev_1_0.xml"
    )
    if not mjcf.is_file():
        pytest.skip(
            "G1 MJCF is absent; run ./scripts/setup_latent_lab.sh "
            f"(expected {mjcf})"
        )
    return mjcf


class _FirstActionTerms(torch.nn.Module):
    def forward(self, observation):
        return 0.5 * observation[:, :29]


class _MarkerEncoder(torch.nn.Module):
    def forward(self, window):
        return window[:, :6]


def _native_bundle(tmp_path, latent_manifest, *, with_encoder=False, model=None):
    root = tmp_path / "native_bundle"
    root.mkdir()
    model = (_FirstActionTerms() if model is None else model).eval()
    torch.onnx.export(
        model,
        torch.zeros(1, latent_manifest.obs.total_width),
        root / "policy.onnx",
        input_names=["obs"],
        output_names=["action"],
        opset_version=18,
        dynamo=False,
    )
    if with_encoder:
        encoder_width = int(
            latent_manifest.command.state_dim
            * (int(latent_manifest.command.window_steps) + 1)
        )
        torch.onnx.export(
            _MarkerEncoder().eval(),
            torch.zeros(1, encoder_width),
            root / "encoder.onnx",
            input_names=["window"],
            output_names=["z"],
            opset_version=18,
            dynamo=False,
        )
    (root / "policy.pt").write_bytes(b"diagnostic-placeholder")
    (root / "encoder.pt").write_bytes(b"diagnostic-placeholder")
    (root / "obs_contract.json").write_text(latent_manifest.obs.model_dump_json())
    (root / "action_contract.json").write_text(latent_manifest.action.model_dump_json())
    np.savez(
        root / "golden_trace.npz",
        obs=np.zeros((1, latent_manifest.obs.total_width), dtype=np.float32),
        action=np.zeros((1, 29), dtype=np.float32),
    )
    raw = latent_manifest.model_dump()
    raw["models"] = {
        "policy_onnx": ModelArtifact(
            format="onnx",
            path="policy.onnx",
            input_name="obs",
            output_name="action",
            input_shape=[1, latent_manifest.obs.total_width],
            output_shape=[1, 29],
            opset=18,
            parity_atol=1e-5,
            max_abs_error=0.0,
        ).model_dump()
    }
    if with_encoder:
        raw["models"]["encoder_onnx"] = ModelArtifact(
            format="onnx",
            path="encoder.onnx",
            input_name="window",
            output_name="z",
            input_shape=[1, encoder_width],
            output_shape=[1, 6],
            opset=18,
            parity_atol=1e-5,
            max_abs_error=0.0,
        ).model_dump()
    names = [
        "policy.onnx",
        "policy.pt",
        "encoder.pt",
        "obs_contract.json",
        "action_contract.json",
        "golden_trace.npz",
    ]
    if with_encoder:
        names.append("encoder.onnx")
    raw["files"] = {name: sha256_file(root / name) for name in names}
    if with_encoder:
        raw["command"].update(
            {
                "activation": "mish",
                "layer_norm": False,
                "encoder_sha256": raw["files"]["encoder.pt"],
            }
        )
    (root / "manifest.json").write_text(json.dumps(raw))
    return PolicyBundle.load(root)


def _state():
    return RobotState(
        stamp=0.0,
        joint_pos=np.full(29, 0.2, dtype=np.float32),
        joint_vel=np.linspace(-0.2, 0.2, 29, dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_ang_vel=np.array([0.1, -0.2, 0.3], dtype=np.float32),
    )


def _shm_name(label):
    return f"/ec_{label}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def test_mujoco_native_cli_accepts_latent_plan_geometry():
    args = build_parser().parse_args(
        [
            "lowlevel",
            "mujoco-native",
            "bundle",
            "--model",
            "robot.xml",
            "--response-slot",
            "/response",
            "--latent-plan",
            "--plan-slots",
            "3",
            "--viewer",
            "--viewer-port",
            "9000",
        ]
    )
    assert args.latent_plan
    assert args.plan_slots == 3
    assert args.viewer
    assert args.viewer_port == 9000


def _write_reference_tree(root, joint_names, frames=20):
    root.mkdir()
    qpos = np.zeros((frames, 7 + len(joint_names)), dtype=np.float32)
    qpos[:, 2] = 0.76
    qpos[:, 3] = 1.0
    qpos[:, 7:] = np.arange(frames, dtype=np.float32)[:, None] * 0.01
    qvel = np.zeros((frames, 6 + len(joint_names)), dtype=np.float32)
    qvel[:, 6:] = np.arange(frames, dtype=np.float32)[:, None] * 0.02
    anchor_pos = np.zeros((frames, 3), dtype=np.float32)
    anchor_pos[:, 0] = np.arange(frames, dtype=np.float32) * 0.002
    anchor_pos[:, 2] = 0.76
    anchor_quat = np.zeros((frames, 4), dtype=np.float32)
    anchor_quat[:, 3] = 1.0
    specs = {}
    for name, values, quaternion_order in (
        ("qpos", qpos, None),
        ("qvel", qvel, None),
        ("anchor_pos_w", anchor_pos, None),
        ("anchor_quat_w", anchor_quat, "xyzw"),
    ):
        mapped = np.memmap(
            root / f"{name}.memmap",
            dtype="float32",
            mode="w+",
            shape=values.shape,
        )
        mapped[:] = values
        mapped.flush()
        specs[name] = {
            "shape": list(values.shape),
            "dtype": "float32",
            "quaternion_order": quaternion_order,
        }
    manifest = {
        "format_version": 1,
        "key": {
            "joint_names": list(joint_names),
            "anchor_body": "pelvis",
            "arrays": specs,
        },
        "traj_info": {
            "start_index": [0],
            "end_index": [frames],
            "ordered_traj_list": [["test", "motion", "trajectory_0"]],
        },
    }
    (root / "reference_arrays_manifest.json").write_text(json.dumps(manifest))
    return qpos, anchor_pos, anchor_quat


def test_native_root_qpos_reexpression_matches_python():
    rng = np.random.default_rng(7)
    raw = np.zeros((10, 36), dtype=np.float32)
    raw[:, :29] = rng.standard_normal((10, 29)).astype(np.float32)
    raw[:, 29:32] = rng.standard_normal((10, 3)).astype(np.float32)
    raw[:, 32:36] = rng.standard_normal((10, 4)).astype(np.float32)
    raw[:, 32:36] /= np.linalg.norm(raw[:, 32:36], axis=1, keepdims=True)
    anchor_position = np.array([0.3, -0.2, 0.7], dtype=np.float32)
    anchor_quaternion = np.array(
        [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32
    )

    actual = ec_native.reexpress_root_qpos_window(
        raw, anchor_position, anchor_quaternion
    )
    expected = np.empty((10, 38), dtype=np.float32)
    expected[:, :29] = raw[:, :29]
    for frame in range(10):
        relative_position, relative_quaternion = subtract_frame(
            anchor_position,
            anchor_quaternion,
            raw[frame, 29:32],
            raw[frame, 32:36],
        )
        expected[frame, 29:32] = relative_position
        expected[frame, 32:38] = rot6d_from_quat(relative_quaternion)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_native_projected_gravity_matches_python():
    quaternion = np.array([0.2, -0.3, 0.1, 0.9], dtype=np.float32)
    actual = ec_native.projected_gravity_from_xyzw(quaternion)
    expected = rotate_inverse(quaternion, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_native_joint_reference_window_matches_term_major_sonic_layout():
    raw = np.zeros((55, 62), dtype=np.float32)
    for frame in range(55):
        raw[frame, :29] = frame * 100 + np.arange(29)
        raw[frame, 29:58] = frame * 1000 + np.arange(29)
        raw[frame, 61] = 1.0
    actual = ec_native.pack_joint_qpos_qvel_anchor_ori_window(
        raw, 2, 10, 5, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    )
    orientation = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    selected = raw[2 : 2 + 10 * 5 : 5]
    blocks = []
    for block in range(5):
        blocks.extend(
            [selected[2 * block, :29], selected[2 * block + 1, :29], orientation]
        )
    for block in range(5):
        blocks.extend(
            [
                selected[2 * block, 29:58],
                selected[2 * block + 1, 29:58],
                orientation,
            ]
        )
    np.testing.assert_array_equal(actual, np.concatenate(blocks))


def test_native_oracle_worker_streams_raw_world_frames(tmp_path, latent_manifest):
    command = latent_manifest.command.model_copy(
        update={
            "state_dim": 38,
            "encoder_state_interface": "root_qpos",
            "macro_anchor_mode": "robot",
        }
    )
    manifest = latent_manifest.model_copy(update={"command": command})
    bundle = _native_bundle(tmp_path, manifest, with_encoder=True)
    reference_root = tmp_path / "reference"
    qpos, anchor_pos, anchor_quat = _write_reference_tree(
        reference_root, bundle.manifest.action.isaac_joint_names
    )
    request_name = _shm_name("oracle_request")
    response_name = _shm_name("oracle_response")
    worker = NativeOracleWorker(
        request_name,
        response_name,
        bundle,
        reference_root,
        "motion",
        start_frame=2,
        horizon=30,
        create_slots=True,
    )
    request = ec_native.ShmCommandSlot(request_name, False)
    response = ec_native.ShmCommandSlot(response_name, False)
    worker.start()
    try:
        request.publish(1, 11, np.array([3.0, 5.0], np.float32), 0.0)
        deadline = time.monotonic() + 2.0
        result = None
        while result is None and time.monotonic() < deadline:
            result = response.snapshot(0)
            time.sleep(0.001)
    finally:
        worker.close()

    assert worker.last_error is None
    assert result is not None
    sequence, tag, values, _, _ = result
    assert sequence == 1
    assert tag == 3
    np.testing.assert_array_equal(values[:3], [3.0, 5.0, 13.0])
    frames = values[3:].reshape(30, 36)
    np.testing.assert_array_equal(frames[0, :29], qpos[7, 7:])
    np.testing.assert_array_equal(frames[0, 29:32], anchor_pos[7])
    np.testing.assert_array_equal(frames[0, 32:36], anchor_quat[7])
    np.testing.assert_array_equal(frames[-1, :29], qpos[-1, 7:])
    assert worker.padded_frames == 17


def test_native_oracle_encodes_stride_five_reference_on_every_control_tick(
    tmp_path, latent_manifest
):
    command = latent_manifest.command.model_copy(
        update={
            "state_dim": 64,
            "encoder_state_interface": "joint_qpos_qvel_anchor_ori",
            "macro_anchor_mode": "robot_heading",
            "macro_frame_stride": 5,
            "encoder_trigger": "every_control_tick",
        }
    )
    manifest = latent_manifest.model_copy(update={"command": command})
    bundle = _native_bundle(tmp_path, manifest, with_encoder=True)
    reference_root = tmp_path / "reference_stride5"
    qpos, anchor_pos, anchor_quat = _write_reference_tree(
        reference_root, bundle.manifest.action.isaac_joint_names, frames=60
    )
    request_name = _shm_name("oracle_stride5_request")
    response_name = _shm_name("oracle_stride5_response")
    worker = NativeOracleWorker(
        request_name,
        response_name,
        bundle,
        reference_root,
        "motion",
        create_slots=True,
    )
    worker.start()
    loop = NativeFakeLoop(
        bundle,
        request_slot=request_name,
        response_slot=response_name,
        create_slots=False,
        command_source="oracle",
        hold_steps=5,
        lead_ticks=2,
        command_stale_ms=1000.0,
    )
    loop.set_initial_pose(
        np.concatenate([anchor_pos[0], anchor_quat[0], qpos[0, 7:]])
    )
    try:
        loop.start(12, paced=True)
        loop.wait()
    finally:
        worker.close()
    stats = loop.stats()
    assert worker.last_error is None
    assert stats["fault"] == 0
    assert stats["control_ticks"] == 12
    assert stats["encoder_inferences"] == 12
    assert stats["planner_requests"] >= 3


def test_native_step_assembles_and_decodes(tmp_path, latent_manifest):
    tracker = NativeTracker(_native_bundle(tmp_path, latent_manifest))
    tracker.warmup()
    command = np.arange(8, dtype=np.float32)
    result = tracker.step_once(_state(), command)
    expected_obs = np.concatenate(
        [
            command,
            [0.0, 0.0, -1.0],
            [0.1, -0.2, 0.3],
            np.full(29, 0.1, dtype=np.float32),
            np.linspace(-0.2, 0.2, 29, dtype=np.float32),
            np.zeros(29, dtype=np.float32),
        ]
    ).astype(np.float32)
    np.testing.assert_allclose(result["observation"], expected_obs)
    np.testing.assert_allclose(result["action"], 0.5 * expected_obs[:29])
    np.testing.assert_allclose(result["joint_target"], 0.1 + 0.25 * result["action"])

    second = tracker.step_once(_state(), command)
    np.testing.assert_allclose(second["observation"][-29:], result["action"])


def test_native_step_assembles_term_major_strided_history(
    tmp_path, latent_manifest
):
    raw = latent_manifest.model_dump()
    raw["obs"]["terms"][2].update(
        {
            "history_length": 3,
            "history_stride": 2,
            "history_order": "oldest_first",
            "reset_fill": "repeat_first",
        }
    )
    raw["obs"]["total_width"] += 6
    manifest = type(latent_manifest).model_validate(raw)
    tracker = NativeTracker(_native_bundle(tmp_path, manifest))
    command = np.arange(8, dtype=np.float32)
    for tick in range(5):
        state = _state()
        state.base_ang_vel[:] = [tick, tick + 0.1, tick + 0.2]
        result = tracker.step_once(state, command)
    history = result["observation"][11:20].reshape(3, 3)
    np.testing.assert_allclose(history[:, 0], [0.0, 2.0, 4.0])
    tracker.reset()
    state = _state()
    state.base_ang_vel[:] = [7.0, 7.1, 7.2]
    reset = tracker.step_once(state, command)
    np.testing.assert_allclose(
        reset["observation"][11:20].reshape(3, 3)[:, 0], [7.0, 7.0, 7.0]
    )


def test_native_step_rejects_nonfinite_state(tmp_path, latent_manifest):
    tracker = NativeTracker(_native_bundle(tmp_path, latent_manifest))
    state = _state()
    state.joint_pos[0] = np.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        tracker.step_once(state, np.zeros(8, dtype=np.float32))


def test_native_onnx_engine_rejects_non_float32_contract(tmp_path):
    path = tmp_path / "double_policy.onnx"
    torch.onnx.export(
        _FirstActionTerms().double().eval(),
        torch.zeros(1, 101, dtype=torch.float64),
        path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=18,
        dynamo=False,
    )
    with pytest.raises(RuntimeError, match="float32"):
        ec_native.OnnxEngine(str(path), "obs", "action", 101, 29)


def test_native_step_clamps_joint_targets_to_bundle_limits(tmp_path, latent_manifest):
    action = latent_manifest.action.model_copy(
        update={
            "joint_limits_lower": [-0.2] * 29,
            "joint_limits_upper": [0.2] * 29,
        }
    )
    manifest = latent_manifest.model_copy(update={"action": action})
    tracker = NativeTracker(_native_bundle(tmp_path, manifest))
    result = tracker.step_once(_state(), np.full(8, 10.0, dtype=np.float32))
    assert np.all(result["joint_target"] >= -0.2)
    assert np.all(result["joint_target"] <= 0.2)
    assert result["joint_target"][0] == pytest.approx(0.2)


def test_native_onnx_engine_replays_bundle_trace(tmp_path, latent_manifest):
    report = verify_native_bundle(_native_bundle(tmp_path, latent_manifest))
    assert report["rows"] == 1
    assert report["policy_max_abs_error"] == 0.0


def test_native_fake_loop_runs_without_python_tick_callbacks(tmp_path, latent_manifest):
    bundle = _native_bundle(tmp_path, latent_manifest)
    response_name = _shm_name("direct")
    loop = NativeFakeLoop(
        bundle,
        response_slot=response_name,
        lead_ticks=2,
        command_stale_ms=1000.0,
    )
    publisher = ec_native.ShmCommandSlot(response_name, False)
    publisher.publish(1, 1, np.ones(8, dtype=np.float32), 0.0)
    loop.start(5000, paced=False)
    loop.wait()

    stats = loop.stats()
    assert stats["ticks"] == 5000
    assert stats["control_ticks"] == 5000
    assert stats["damp_ticks"] == 0
    assert stats["fault"] == 0
    durations_ms = loop.tick_durations_ns().astype(np.float64) / 1e6
    assert np.percentile(durations_ms, 99) < 5.0
    assert np.isfinite(loop.state()["joint_position"]).all()


def test_native_chunk_mailbox_builds_causal_history_and_encodes_off_thread(
    tmp_path, latent_manifest
):
    bundle = _native_bundle(tmp_path, latent_manifest, with_encoder=True)
    request_name = _shm_name("request")
    response_name = _shm_name("response")
    loop = NativeFakeLoop(
        bundle,
        request_slot=request_name,
        response_slot=response_name,
        control_hz=100,
        hold_steps=5,
        lead_ticks=2,
        command_absent_ticks=20,
        command_stale_ms=1000.0,
    )
    histories = []
    contexts = []

    def service(history, context):
        histories.append(np.array(history, copy=True))
        contexts.append(dict(context))
        marker = float(len(histories))
        return np.full((30, 4), marker, dtype=np.float32)

    worker = NativeChunkWorker(
        request_name,
        response_name,
        service,
        hold_steps=5,
        lead_ticks=2,
        state_width=4,
    )
    worker.start()
    try:
        loop.start(40, paced=True)
        loop.wait()
    finally:
        worker.close()

    assert worker.last_error is None
    assert worker.requests >= 5
    assert histories
    assert all(context["prev_chunk"] is None for context in contexts)
    first = histories[0].reshape(10, 93)
    np.testing.assert_allclose(first, np.repeat(first[:1], 10, axis=0))
    np.testing.assert_allclose(first[0, :58], 0.0)
    np.testing.assert_allclose(first[0, 58:61], 0.0)
    np.testing.assert_allclose(first[0, 61:64], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(first[0, 64:], 0.0)
    stats = loop.stats()
    assert stats["control_ticks"] >= 35
    assert worker.requests - stats["planner_responses"] in {0, 1}
    assert stats["planner_requests"] in {worker.requests, worker.requests + 1}
    assert stats["deadline_misses"] == 0
    assert stats["damp_ticks"] == 0
    assert stats["fault"] == 0


def test_native_late_chunk_uses_actual_elapsed_frame_offset(tmp_path, latent_manifest):
    bundle = _native_bundle(tmp_path, latent_manifest, with_encoder=True)
    request_name = _shm_name("late_request")
    response_name = _shm_name("late_response")
    loop = NativeFakeLoop(
        bundle,
        request_slot=request_name,
        response_slot=response_name,
        control_hz=100,
        hold_steps=5,
        lead_ticks=2,
        command_absent_ticks=100,
        command_stale_ms=2000.0,
    )
    release_second = threading.Event()
    release_later = threading.Event()
    calls = 0

    def service(history, context):
        nonlocal calls
        calls += 1
        if calls == 2:
            release_second.wait(timeout=5)
        elif calls > 2:
            release_later.wait(timeout=5)
        frames = np.arange(30, dtype=np.float32)[:, None]
        return np.repeat(frames, 4, axis=1)

    worker = NativeChunkWorker(
        request_name,
        response_name,
        service,
        hold_steps=5,
        lead_ticks=2,
        state_width=4,
    )
    worker.start()
    loop.start(100, paced=True)
    try:
        deadline = time.monotonic() + 3.0
        while loop.stats()["deadline_misses"] == 0 and time.monotonic() < deadline:
            time.sleep(0.002)
        assert loop.stats()["deadline_misses"] == 1
        release_second.set()
        deadline = time.monotonic() + 3.0
        while (
            loop.stats()["last_chunk_offset_steps"] != 7 and time.monotonic() < deadline
        ):
            time.sleep(0.002)
        assert loop.stats()["last_chunk_offset_steps"] == 7
        assert loop.stats()["fault"] == 0
    finally:
        release_second.set()
        release_later.set()
        loop.stop()
        loop.wait()
        worker.close()


def test_native_encoder_bundle_accepts_direct_latent_commands(
    tmp_path, latent_manifest
):
    bundle = _native_bundle(tmp_path, latent_manifest, with_encoder=True)
    response_name = _shm_name("direct_latent_with_encoder")
    loop = NativeFakeLoop(
        bundle,
        response_slot=response_name,
        hold_steps=5,
        lead_ticks=2,
        command_stale_ms=1000.0,
    )
    publisher = ec_native.ShmCommandSlot(response_name, False)
    publisher.publish(1, 1, np.ones(8, dtype=np.float32), 0.0)
    loop.start(4, paced=False)
    loop.wait()
    assert loop.stats()["control_ticks"] == 4
    assert loop.stats()["fault"] == 0


def test_native_encoder_accepts_recorded_macro_stride(tmp_path, latent_manifest):
    command = latent_manifest.command.model_copy(update={"macro_frame_stride": 5})
    manifest = latent_manifest.model_copy(update={"command": command})
    bundle = _native_bundle(tmp_path, manifest, with_encoder=True)
    loop = NativeFakeLoop(
        bundle,
        response_slot=_shm_name("stride"),
        hold_steps=5,
        lead_ticks=2,
    )
    assert loop.tracker.observation_width == bundle.manifest.obs.total_width


def test_native_mujoco_loop_runs_independent_physics_schedule(
    tmp_path, latent_manifest
):
    joint_names = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    action = ActionContract(
        isaac_joint_names=joint_names,
        sdk_joint_names=joint_names,
        isaac_to_sdk=list(range(29)),
        default_joint_pos=[0.0] * 29,
        action_scale=[0.25] * 29,
        stiffness=[40.0] * 29,
        damping=[1.0] * 29,
        armature=[0.01] * 29,
        effort_limit=[100.0] * 29,
    )
    manifest = latent_manifest.model_copy(update={"action": action})
    bundle = _native_bundle(tmp_path, manifest)
    response_name = _shm_name("mujoco")
    loop = NativeMujocoLoop(
        bundle,
        str(_g1_mjcf_path()),
        response_slot=response_name,
        lead_ticks=2,
        command_stale_ms=1000.0,
    )
    publisher = ec_native.ShmCommandSlot(response_name, False)
    assert loop.latest_state() is None
    loop.start(25, paced=True)
    time.sleep(0.05)
    live_pose = loop.latest_state()
    assert live_pose is not None
    assert live_pose.shape == (36,)
    assert live_pose[2] > 0.1
    publisher.publish(1, 1, np.ones(8, dtype=np.float32), 0.0)
    loop.wait()

    stats = loop.stats()
    assert stats["wait_ticks"] >= 1
    assert stats["control_ticks"] + stats["wait_ticks"] == 25
    assert stats["damp_ticks"] == 0
    assert stats["fault"] == 0
    assert 97 <= stats["backend_steps"] <= 101
    assert stats["backend_steps"] > 3 * stats["ticks"]
    assert loop.backend_time == pytest.approx(stats["backend_steps"] * 0.005)
    assert stats["backend_deadline_misses"] == 0
    assert stats["backend_wake_late_ns_max"] > 0
    assert stats["backend_realtime_configured"]
    assert np.isfinite(loop.base_height)
    assert loop.base_height > 0.1

    with pytest.raises(RuntimeError, match="requires paced"):
        loop.start(1, paced=False)

    loop.start(250, paced=True)
    publisher.publish(2, 1, np.ones(8, dtype=np.float32), 0.0)
    time.sleep(0.06)
    loop.stop()
    loop.wait()
    stopped_time = loop.backend_time
    assert 0 < loop.stats()["ticks"] < 250
    assert loop.stats()["backend_steps"] > 0
    time.sleep(0.02)
    assert loop.backend_time == stopped_time

    with pytest.raises(RuntimeError, match="timestep times decimation"):
        NativeMujocoLoop(
            bundle,
            str(_g1_mjcf_path()),
            response_slot=_shm_name("bad_rate"),
            control_hz=100,
            lead_ticks=2,
        )


def test_native_loop_latches_damp_when_command_is_absent(tmp_path, latent_manifest):
    loop = NativeFakeLoop(
        _native_bundle(tmp_path, latent_manifest),
        response_slot=_shm_name("absent"),
        command_absent_ticks=2,
        lead_ticks=2,
    )
    loop.start(6, paced=False)
    loop.wait()
    stats = loop.stats()
    assert stats["fault"] == 1
    assert stats["damp_ticks"] >= 3


def test_native_loop_rejects_bad_command_width(tmp_path, latent_manifest):
    bundle = _native_bundle(tmp_path, latent_manifest)
    response_name = _shm_name("bad_width")
    loop = NativeFakeLoop(bundle, response_slot=response_name, lead_ticks=2)
    publisher = ec_native.ShmCommandSlot(response_name, False)
    publisher.publish(1, 1, np.zeros(7, dtype=np.float32), 0.0)
    loop.start(3, paced=False)
    loop.wait()
    assert loop.stats()["fault"] == 3


def test_native_loop_latches_damp_for_stale_command(tmp_path, latent_manifest):
    bundle = _native_bundle(tmp_path, latent_manifest)
    response_name = _shm_name("stale")
    loop = NativeFakeLoop(
        bundle,
        response_slot=response_name,
        lead_ticks=2,
        command_stale_ms=0.1,
    )
    publisher = ec_native.ShmCommandSlot(response_name, False)
    publisher.publish(1, 1, np.zeros(8, dtype=np.float32), 0.0)
    time.sleep(0.01)
    loop.start(3, paced=False)
    loop.wait()
    assert loop.stats()["fault"] == 2


@pytest.mark.skipif(
    not ec_native.WITH_UNITREE, reason="native extension was built without Unitree SDK2"
)
def test_unitree_writer_stays_closed_before_initialization(tmp_path, latent_manifest):
    action = latent_manifest.action.model_copy(
        update={
            "joint_limits_lower": [-3.0] * 29,
            "joint_limits_upper": [3.0] * 29,
        }
    )
    manifest = latent_manifest.model_copy(update={"action": action})
    loop = NativeUnitreeLoop(
        _native_bundle(tmp_path, manifest),
        "lo",
        response_slot=_shm_name("unitree_gate"),
        writes_enabled=True,
        control_cpu=-1,
        writer_cpu=-1,
        control_fifo_priority=0,
        writer_fifo_priority=0,
        lock_memory=False,
        require_realtime=False,
    )
    time.sleep(0.05)
    assert loop.writer_stats()["publishes"] == 0
    loop.force_damp()
    time.sleep(0.05)
    assert loop.writer_stats()["publishes"] == 0


# --------------------------------------------------------------------------
# Latent plan: one planner reply covers `plan_slots` holds. The controller
# walks the plan without calling the planner again, which is the cadence the
# Isaac board's leading row uses (30 slots, hold 1).
# --------------------------------------------------------------------------


class _PlanServer:
    """Answer native planner requests with a fixed-size latent plan.

    Slot k of every plan is filled with the constant `k + 1`, so a rollout's
    joint targets show one plateau per consumed slot.
    """

    REQUEST_TAG = 10
    PLAN_TAG = 4

    def __init__(
        self,
        request_slot,
        response_slot,
        *,
        slots,
        z_dim,
        reply_delay_s=0.0,
        max_replies=None,
    ):
        self._request = ec_native.ShmCommandSlot(request_slot, False)
        self._response = ec_native.ShmCommandSlot(response_slot, False)
        self.slots = int(slots)
        self.z_dim = int(z_dim)
        self.reply_delay_s = float(reply_delay_s)
        self.max_replies = max_replies
        self.replies = 0
        self._last_sequence = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _plan(self):
        plan = np.zeros((self.slots, self.z_dim), dtype=np.float32)
        for slot in range(self.slots):
            plan[slot, :] = float(slot + 1)
        return plan.reshape(-1)

    def _run(self):
        while not self._stop.is_set():
            raw = self._request.snapshot(self._last_sequence)
            if raw is None:
                self._stop.wait(0.001)
                continue
            sequence, tag, _, _, _ = raw
            if int(sequence) <= self._last_sequence or int(tag) != self.REQUEST_TAG:
                self._stop.wait(0.001)
                continue
            self._last_sequence = int(sequence)
            if self.max_replies is not None and self.replies >= self.max_replies:
                continue
            if self.reply_delay_s > 0.0:
                self._stop.wait(self.reply_delay_s)
            self._response.publish(
                int(sequence), self.PLAN_TAG, self._plan(), time.monotonic()
            )
            self.replies += 1


def test_native_latent_plan_serves_every_slot_from_one_reply(
    tmp_path, latent_manifest
):
    bundle = _native_bundle(tmp_path, latent_manifest)
    response_name = _shm_name("plan_response")
    request_name = _shm_name("plan_request")
    slots, hold, plans = 4, 5, 3
    ticks = slots * hold * plans
    loop = NativeFakeLoop(
        bundle,
        response_slot=response_name,
        request_slot=request_name,
        hold_steps=hold,
        lead_ticks=2,
        plan_slots=slots,
        latent_plan=True,
        command_stale_ms=5000.0,
    )
    server = _PlanServer(
        request_name, response_name, slots=slots, z_dim=latent_manifest.command.z_dim
    ).start()
    try:
        loop.start(ticks, paced=True)
        loop.wait()
    finally:
        server.stop()

    stats = loop.stats()
    assert stats["fault"] == 0
    assert stats["deadline_misses"] == 0
    assert stats["plan_late_starts"] == 0
    # Most holds are served from a plan already in hand, so the planner is
    # called about once per plan instead of once per hold.
    assert stats["plan_slot_advances"] >= (slots - 1) * (plans - 1)
    assert stats["planner_requests"] <= plans + 1
    assert stats["planner_requests"] < ticks // hold

    # Slot k drives the tracker for `hold` ticks, so the first joint target
    # holds one value per slot and steps up as the plan is walked.
    # The log is flat (ticks x joints); take joint 0 of every tick.
    joint_log = np.asarray(loop.joint_position_log()).reshape(-1, 29)[:, 0]
    plateaus = [joint_log[0]]
    for value in joint_log[1:]:
        if not np.isclose(value, plateaus[-1]):
            plateaus.append(value)
    assert len(plateaus) >= slots


def test_native_latent_plan_allows_a_lead_longer_than_one_hold(
    tmp_path, latent_manifest
):
    bundle = _native_bundle(tmp_path, latent_manifest)
    # The lead counts down to plan exhaustion, so it may span several holds.
    loop = NativeFakeLoop(
        bundle,
        response_slot=_shm_name("plan_long_lead"),
        request_slot=_shm_name("plan_long_lead_req"),
        hold_steps=5,
        lead_ticks=12,
        plan_slots=4,
        latent_plan=True,
    )
    assert loop.tracker.command_width == bundle.manifest.command.z_dim + 2
    # Without a plan the historical rule still holds: the lead must fit inside
    # the single hold that one reply covers.
    with pytest.raises(RuntimeError):
        NativeFakeLoop(
            bundle,
            response_slot=_shm_name("plan_bad_lead"),
            request_slot=_shm_name("plan_bad_lead_req"),
            hold_steps=5,
            lead_ticks=12,
            plan_slots=1,
            latent_plan=True,
        )


def test_native_latent_plan_holds_the_last_slot_on_a_deadline_miss(
    tmp_path, latent_manifest
):
    bundle = _native_bundle(tmp_path, latent_manifest)
    response_name = _shm_name("plan_miss_response")
    request_name = _shm_name("plan_miss_request")
    slots, hold = 2, 5
    loop = NativeFakeLoop(
        bundle,
        response_slot=response_name,
        request_slot=request_name,
        hold_steps=hold,
        lead_ticks=2,
        plan_slots=slots,
        latent_plan=True,
        command_stale_ms=5000.0,
    )
    # One plan only: after it is walked the planner is silent, and the
    # controller must hold the last command and count the miss, not fault.
    server = _PlanServer(
        request_name,
        response_name,
        slots=slots,
        z_dim=latent_manifest.command.z_dim,
        max_replies=1,
    ).start()
    try:
        loop.start(slots * hold * 3, paced=True)
        loop.wait()
    finally:
        server.stop()
    stats = loop.stats()
    assert stats["fault"] == 0
    assert stats["deadline_misses"] > 0
    assert stats["control_ticks"] > slots * hold


def test_native_latent_plan_time_aligns_a_late_reply(tmp_path, latent_manifest):
    bundle = _native_bundle(tmp_path, latent_manifest)
    response_name = _shm_name("plan_late_response")
    request_name = _shm_name("plan_late_request")
    slots, hold = 2, 5
    loop = NativeFakeLoop(
        bundle,
        response_slot=response_name,
        request_slot=request_name,
        hold_steps=hold,
        lead_ticks=2,
        plan_slots=slots,
        latent_plan=True,
        command_stale_ms=5000.0,
    )
    # A reply that lands after the whole plan's window has passed is used at
    # its last slot and recorded, never replayed from slot 0.
    server = _PlanServer(
        request_name,
        response_name,
        slots=slots,
        z_dim=latent_manifest.command.z_dim,
        reply_delay_s=(slots * hold + 2) / 50.0,
    ).start()
    try:
        loop.start(slots * hold * 2, paced=True)
        loop.wait()
    finally:
        server.stop()
    stats = loop.stats()
    assert stats["fault"] == 0
    assert stats["plan_late_starts"] >= 1
    assert stats["last_chunk_offset_steps"] >= slots * hold


def test_native_latent_plan_worker_drives_the_controller(tmp_path, latent_manifest):
    """The worker + controller pair: one head call per plan, no encoder."""
    bundle = _native_bundle(tmp_path, latent_manifest)
    response_name = _shm_name("plan_worker_response")
    request_name = _shm_name("plan_worker_request")
    slots, hold = 3, 5
    z_dim = int(latent_manifest.command.z_dim)
    calls = []

    def _head(history, context):
        assert history.shape == (930,)
        calls.append(context)
        plan = np.zeros((slots, z_dim), dtype=np.float32)
        for slot in range(slots):
            plan[slot, :] = float(slot + 1)
        return plan

    loop = NativeFakeLoop(
        bundle,
        response_slot=response_name,
        request_slot=request_name,
        hold_steps=hold,
        lead_ticks=4,
        plan_slots=slots,
        latent_plan=True,
        command_stale_ms=5000.0,
    )
    worker = NativeLatentPlanWorker(
        request_name,
        response_name,
        _head,
        z_dim=z_dim,
        plan_slots=slots,
        hold_steps=hold,
        lead_ticks=4,
    )
    worker.start()
    try:
        loop.start(slots * hold * 2, paced=True)
        loop.wait()
    finally:
        worker.close()

    stats = loop.stats()
    assert worker.last_error is None
    assert stats["fault"] == 0
    assert stats["deadline_misses"] == 0
    assert stats["plan_slot_advances"] >= slots - 1
    # One call per plan, not one per hold.
    assert 0 < worker.requests <= 3
    assert len(calls) == worker.requests


def test_native_latent_plan_keeps_requesting_at_hold_one(tmp_path, latent_manifest):
    """Hold 1 with a 30-slot plan: the leading Isaac row's cadence.

    The countdown to plan exhaustion steps 1 -> 0 at hold 1, so a scheduler
    that waits for the lead EXACTLY never asks for another plan and starves
    the controller after the first one.
    """
    bundle = _native_bundle(tmp_path, latent_manifest)
    response_name = _shm_name("plan_hold1_response")
    request_name = _shm_name("plan_hold1_request")
    slots, hold, lead = 30, 1, 5
    ticks = slots * 4
    loop = NativeFakeLoop(
        bundle,
        response_slot=response_name,
        request_slot=request_name,
        hold_steps=hold,
        lead_ticks=lead,
        plan_slots=slots,
        latent_plan=True,
        command_stale_ms=5000.0,
    )
    server = _PlanServer(
        request_name, response_name, slots=slots, z_dim=latent_manifest.command.z_dim
    ).start()
    try:
        loop.start(ticks, paced=True)
        loop.wait()
    finally:
        server.stop()
    stats = loop.stats()
    assert stats["fault"] == 0
    assert stats["control_ticks"] > ticks - 5
    assert stats["planner_requests"] >= 3  # one per plan, not one per hold
    assert stats["planner_requests"] <= ticks // slots + 2
    assert stats["plan_slot_advances"] >= (slots - 1) * 2
    assert stats["deadline_misses"] == 0


def test_the_oracle_horizon_covers_the_window_the_control_thread_reads(
    tmp_path, latent_manifest
):
    """SONIC v1.1's stride of 5 needs a chunk the encoder can read past.

    `encode_active_reference` reads from offset `o` out to
    `o + (window_frames - 1) * stride` strictly inside the chunk, and `o`
    reaches `hold_steps` on the tick a new chunk is due. A horizon equal to
    that sum is one frame short and faults on a command contract mid-run.
    """
    command = latent_manifest.command.model_copy(
        update={
            "state_dim": 64,
            "encoder_state_interface": "joint_qpos_qvel_anchor_ori",
            "macro_anchor_mode": "robot_heading",
            "macro_frame_stride": 5,
            "encoder_trigger": "every_control_tick",
        }
    )
    manifest = latent_manifest.model_copy(update={"command": command})
    bundle = _native_bundle(tmp_path, manifest, with_encoder=True)
    reference_root = tmp_path / "reference_horizon"
    _write_reference_tree(
        reference_root, bundle.manifest.action.isaac_joint_names, frames=200
    )
    window_frames = int(command.window_steps + 1)
    stride = int(command.macro_frame_stride)
    hold = int(command.hold_steps)
    minimum = (window_frames - 1) * stride + hold + 1

    def _worker(horizon):
        return NativeOracleWorker(
            _shm_name("oracle_horizon_request"),
            _shm_name("oracle_horizon_response"),
            bundle,
            reference_root,
            "motion",
            horizon=horizon,
            create_slots=True,
        )

    with pytest.raises(ValueError, match="shorter than the encoder window"):
        _worker(minimum - 1)

    exact = _worker(minimum)
    try:
        assert exact.horizon == minimum
    finally:
        exact.close()

    # The default carries one hold of slack, so a late reply is a deadline
    # miss rather than a fault.
    default = _worker(None)
    try:
        assert default.horizon == minimum + hold
    finally:
        default.close()
