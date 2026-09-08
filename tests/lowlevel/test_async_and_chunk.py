"""Async pull client scheduling + chunk-slot decoding semantics."""

import threading
import sys

import numpy as np
import pytest

from embodied_control.lowlevel.bundle import CommandContract
from embodied_control.lowlevel.command_buffer import InProcessCommandBuffer
from embodied_control.lowlevel.contracts import CommandPacket, RobotState
from embodied_control.lowlevel.decoders import ChunkCommandSource
from embodied_control.lowlevel.maths import (
    quat_conjugate,
    quat_mul,
    quat_to_mat,
    rotate_inverse,
)
from embodied_control.lowlevel.publishers.pull_client import AsyncChunkPullClient
from embodied_control.lowlevel.publishers.native_pull import StdioChunkService
from embodied_control.lowlevel.tracker import BufferedCommandSource

FRAME_DIM = 4
HOLD = 10
LEAD = 3
YAW90 = np.array([0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32)
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _contract() -> CommandContract:
    return CommandContract(
        z_dim=6,
        phase_mode="sin_cos",
        phase_dim=2,
        hold_steps=HOLD,
        state_dim=FRAME_DIM,
        window_steps=9,
        horizon_steps=10,
        encoder_window_mode="intermediate",
        macro_frame_stride=1,
        macro_anchor_mode="robot",
    )


class MarkerEncoder:
    """z = [window mean, 0...]: identifies which chunk was encoded."""

    def infer(self, obs, out=None):
        z = np.zeros(6, dtype=np.float32)
        z[0] = float(np.mean(obs))
        return z


class FirstFrameEncoder:
    def infer(self, obs, out=None):
        window = np.asarray(obs, dtype=np.float32).reshape(-1, FRAME_DIM)
        z = np.zeros(6, dtype=np.float32)
        z[0] = window[0, 0]
        return z


class ControlledService:
    def __init__(self):
        self.release = threading.Event()
        self.done = threading.Event()
        self.contexts = []
        self.counter = 0

    def __call__(self, history, context):
        self.contexts.append(
            {
                "prev": None
                if context["prev_chunk"] is None
                else context["prev_chunk"].copy(),
                "freeze": context["freeze_steps"],
            }
        )
        self.release.wait()
        self.counter += 1
        chunk = np.full((30, FRAME_DIM), float(self.counter), dtype=np.float32)
        self.done.set()
        return chunk


def _drive(client, service, ticks):
    """Advance ticks deterministically: wait out worker completions."""
    buffer_values = []
    history = np.zeros(930, dtype=np.float32)
    for tick in range(ticks):
        service.release.set()
        client.tick(tick, float(tick), history)
        if tick == 0 or client._steps_remaining == LEAD - 1:
            # Cold start, or a request was just issued at steps==lead: wait
            # for the worker so the swap at expiry is deterministic.
            service.done.wait(timeout=5)
            service.done.clear()
        packet = client.buffer.snapshot().packet
        buffer_values.append(np.array(packet.values, copy=True))
    return buffer_values


def test_pull_client_on_time_schedule():
    service = ControlledService()
    client = AsyncChunkPullClient(
        InProcessCommandBuffer(),
        MarkerEncoder(),
        _contract(),
        service,
        hold_steps=HOLD,
        lead_ticks=LEAD,
        window_frames=10,
    )
    try:
        values = _drive(client, service, 30)
    finally:
        client.close()
    assert client.deadline_misses == 0
    assert client.request_overruns == 0
    z_markers = [v[0] for v in values]
    assert z_markers[0] == pytest.approx(1.0)  # cold-start chunk
    assert z_markers[9] == pytest.approx(1.0)  # held through the hold
    assert z_markers[10] == pytest.approx(2.0)  # swap exactly at expiry
    assert z_markers[20] == pytest.approx(3.0)
    assert service.contexts[0]["prev"] is None
    assert service.contexts[1]["prev"] is not None
    np.testing.assert_allclose(service.contexts[1]["prev"], 1.0)
    assert service.contexts[1]["freeze"] == LEAD
    # Phase restarts at each renewal: sin(0)=0, cos(0)=1.
    assert values[10][6] == pytest.approx(0.0, abs=1e-6)
    assert values[10][7] == pytest.approx(1.0, abs=1e-6)


def test_pull_client_deadline_miss_holds_stale_and_recovers():
    service = ControlledService()
    client = AsyncChunkPullClient(
        InProcessCommandBuffer(),
        MarkerEncoder(),
        _contract(),
        service,
        hold_steps=HOLD,
        lead_ticks=LEAD,
        window_frames=10,
    )
    history = np.zeros(930, dtype=np.float32)
    try:
        service.release.set()  # cold start succeeds
        client.tick(0, 0.0, history)
        service.done.wait(timeout=5)
        service.done.clear()
        service.release.clear()  # second request will hang
        markers = []
        for tick in range(1, 15):
            client.tick(tick, float(tick), history)
            markers.append(client.buffer.snapshot().packet.values[0])
        assert client.deadline_misses == 1
        assert all(m == pytest.approx(1.0) for m in markers)  # stale z held
        service.release.set()  # late reply lands
        service.done.wait(timeout=5)
        for tick in range(15, 32):
            client.tick(tick, float(tick), history)
        assert client.buffer.snapshot().packet.values[0] >= 2.0
    finally:
        service.release.set()
        client.close()


def test_pull_client_late_chunk_uses_elapsed_frame_offset():
    service = ControlledService()

    def indexed_service(history, context):
        service.contexts.append(context)
        service.release.wait()
        service.counter += 1
        service.done.set()
        base = 100.0 * service.counter
        return np.repeat(
            (base + np.arange(30, dtype=np.float32))[:, None], FRAME_DIM, axis=1
        )

    client = AsyncChunkPullClient(
        InProcessCommandBuffer(),
        FirstFrameEncoder(),
        _contract(),
        indexed_service,
        hold_steps=HOLD,
        lead_ticks=LEAD,
        window_frames=10,
    )
    history = np.zeros(930, dtype=np.float32)
    try:
        service.release.set()
        client.tick(0, 0.0, history)
        assert client.buffer.snapshot().packet.values[0] == pytest.approx(100.0)
        service.done.wait(timeout=5)
        service.done.clear()
        service.release.clear()
        for tick in range(1, 12):
            client.tick(tick, float(tick), history)
        assert client.deadline_misses == 1
        service.release.set()
        service.done.wait(timeout=5)
        for tick in range(12, 21):
            client.tick(tick, float(tick), history)
        assert client.buffer.snapshot().packet.values[0] == pytest.approx(
            200.0 + LEAD + HOLD
        )
    finally:
        service.release.set()
        client.close()


def test_pull_client_lead_bounds():
    with pytest.raises(ValueError, match="lead_ticks"):
        AsyncChunkPullClient(
            InProcessCommandBuffer(),
            MarkerEncoder(),
            _contract(),
            lambda h, c: None,
            hold_steps=HOLD,
            lead_ticks=HOLD,
        )


def test_stdio_chunk_service_checks_contract_and_round_trips():
    script = r"""
import json
import sys

print(json.dumps({
    "ready": True,
    "action_horizon": 10,
    "state_history": 10,
    "state_width": 93,
    "action_width": 38,
    "window_frames": 10,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("stop"):
        break
    print(json.dumps({
        "chunk": list(range(10 * 38)),
        "head_ms": 12.5,
        "echo_goal": request.get("goal"),
    }), flush=True)
"""
    service = StdioChunkService(
        [sys.executable, "-u", "-c", script], goal="walk_straight"
    )
    try:
        chunk = service(
            np.zeros(930, dtype=np.float32),
            {"prev_chunk": None, "freeze_steps": 2, "hold_steps": 5},
        )
    finally:
        service.close()
    assert chunk.shape == (10, 38)
    np.testing.assert_array_equal(chunk.reshape(-1), np.arange(10 * 38))
    assert service.head_ms == [12.5]


def test_stdio_chunk_service_accepts_latent_plan_contract():
    script = r"""
import json
import sys

print(json.dumps({
    "ready": True,
    "action_horizon": 3,
    "state_history": 10,
    "state_width": 93,
    "action_width": 64,
    "window_frames": 3,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("stop"):
        break
    print(json.dumps({"chunk": list(range(3 * 64))}), flush=True)
"""
    service = StdioChunkService(
        [sys.executable, "-u", "-c", script],
        action_width=64,
        window_frames=3,
    )
    try:
        plan = service(np.zeros(930, dtype=np.float32), {})
    finally:
        service.close()
    assert plan.shape == (3, 64)


def test_stdio_chunk_service_switches_goal_file_without_restarting(tmp_path):
    script = r"""
import json
import sys

print(json.dumps({
    "ready": True,
    "action_horizon": 1,
    "state_history": 10,
    "state_width": 93,
    "action_width": 2,
    "window_frames": 1,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("stop"):
        break
    value = 1.0 if request.get("goal") == "walk" else 2.0
    print(json.dumps({"chunk": [value, value], "head_ms": value}), flush=True)
"""
    goal_file = tmp_path / "goal.txt"
    goal_file.write_text("walk\n")
    service = StdioChunkService(
        [sys.executable, "-u", "-c", script],
        action_width=2,
        window_frames=1,
        goal_file=goal_file,
    )
    try:
        walk = service(np.zeros(930, dtype=np.float32), {})
        goal_file.write_text("idle\n")
        idle = service(np.zeros(930, dtype=np.float32), {})
    finally:
        service.close()
    np.testing.assert_array_equal(walk, [[1.0, 1.0]])
    np.testing.assert_array_equal(idle, [[2.0, 2.0]])
    assert service.head_ms == [1.0, 2.0]


def test_stdio_chunk_service_rejects_wrong_response_width():
    script = r"""
import json
import sys

print(json.dumps({
    "ready": True,
    "action_horizon": 10,
    "state_history": 10,
    "state_width": 93,
    "action_width": 38,
    "window_frames": 10,
}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("stop"):
        break
    print(json.dumps({"chunk": [0.0] * 379}), flush=True)
"""
    service = StdioChunkService([sys.executable, "-u", "-c", script])
    try:
        with pytest.raises(RuntimeError, match="379 values; expected 380"):
            service(np.zeros(930, dtype=np.float32), {})
    finally:
        service.close()


def test_stdio_chunk_service_rejects_stdout_before_ready_record():
    script = "print('upstream diagnostic on stdout', flush=True)"
    with pytest.raises(RuntimeError, match="non-JSON data.*upstream diagnostic"):
        StdioChunkService([sys.executable, "-u", "-c", script])


# --------------------------------------------------------------------------- #
# Chunk decoder
# --------------------------------------------------------------------------- #

COMPONENTS = [
    ("expert_motion", 58),
    ("expert_anchor_pos_b", 3),
    ("expert_anchor_ori_b", 6),
]


def _chunk_packet(sequence=1, anchor_pos=(0, 0, 0), anchor_quat=IDENTITY):
    motion = np.stack([np.full(58, j, dtype=np.float32) for j in range(10)])
    pos = np.stack([[0.1 * j, 0.0, 0.0] for j in range(10)]).astype(np.float32)
    ori = np.tile(np.array([1, 0, 0, 1, 0, 0], np.float32), (10, 1))
    values = np.concatenate([motion.reshape(-1), pos.reshape(-1), ori.reshape(-1)])
    assert values.shape == (670,)
    return CommandPacket(
        interface="chunk",
        values=values,
        sequence=sequence,
        stamp=0.0,
        metadata={
            "anchor_pos_w": np.asarray(anchor_pos, np.float32),
            "anchor_quat_w": np.asarray(anchor_quat, np.float32),
        },
    )


def _state(pos, quat):
    return RobotState(
        stamp=0.0,
        joint_pos=np.zeros(29, np.float32),
        joint_vel=np.zeros(29, np.float32),
        projected_gravity=np.array([0, 0, -1], np.float32),
        base_ang_vel=np.zeros(3, np.float32),
        anchor_pos_w=np.asarray(pos, np.float32),
        anchor_quat_w=np.asarray(quat, np.float32),
    )


def test_chunk_slots_consumed_once_and_overrun_repeats():
    buffer = InProcessCommandBuffer()
    decoder = ChunkCommandSource(BufferedCommandSource(buffer), COMPONENTS)
    decoder.reset(None)
    buffer.publish(_chunk_packet())
    state = _state([0, 0, 0], IDENTITY)
    for j in range(10):
        sample = decoder.update(j, state)
        assert sample.metadata["slot"] == j
        assert not sample.metadata["slot_overrun"]
        np.testing.assert_allclose(sample.terms["expert_motion"], j)
        np.testing.assert_allclose(
            sample.terms["expert_anchor_pos_b"], [0.1 * j, 0, 0], atol=1e-6
        )
    overrun = decoder.update(10, state)
    assert overrun.metadata["slot"] == 9
    assert overrun.metadata["slot_overrun"]

    buffer.publish(_chunk_packet(sequence=2))
    renewed = decoder.update(11, state)
    assert renewed.metadata["slot"] == 0
    assert renewed.renewed


def test_chunk_reexpression_matches_frame_math():
    buffer = InProcessCommandBuffer()
    decoder = ChunkCommandSource(BufferedCommandSource(buffer), COMPONENTS)
    decoder.reset(None)
    buffer.publish(_chunk_packet())
    moved = _state([0.5, 0.0, 0.0], YAW90)
    sample = decoder.update(0, moved)

    delta_quat = quat_mul(quat_conjugate(YAW90), IDENTITY)
    delta_pos = rotate_inverse(
        YAW90, np.array([0, 0, 0], np.float32) - np.array([0.5, 0, 0], np.float32)
    )
    expected_pos = quat_to_mat(delta_quat) @ np.array([0.0, 0, 0]) + delta_pos
    np.testing.assert_allclose(
        sample.terms["expert_anchor_pos_b"], expected_pos, atol=1e-6
    )
    # Slot-0 packet position was the origin of the publish anchor; the robot
    # moved +0.5 m world X and yawed +90, so in its frame the old anchor sits
    # 0.5 m along -(-Y) -> +Y... verified numerically via the oracle above;
    # spot-check one literal: rotate_inverse(yaw90, [-0.5,0,0]) = [0, 0.5, 0].
    np.testing.assert_allclose(expected_pos, [0.0, 0.5, 0.0], atol=1e-6)
    ori = sample.terms["expert_anchor_ori_b"].reshape(3, 2)
    np.testing.assert_allclose(ori, quat_to_mat(delta_quat)[:, :2], atol=1e-6)
    np.testing.assert_allclose(sample.terms["expert_motion"], 0.0, atol=1e-6)


def test_chunk_wrong_width_rejected():
    buffer = InProcessCommandBuffer()
    decoder = ChunkCommandSource(BufferedCommandSource(buffer), COMPONENTS)
    decoder.reset(None)
    buffer.publish(
        CommandPacket(
            interface="chunk", values=np.zeros(100, np.float32), sequence=1, stamp=0.0
        )
    )
    with pytest.raises(ValueError, match="670"):
        decoder.update(0, _state([0, 0, 0], IDENTITY))
