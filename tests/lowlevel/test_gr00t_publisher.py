"""Gr00tServicePublisher schedule, protocol, and encoding semantics."""

from __future__ import annotations

import json
import sys
import textwrap

import numpy as np

from embodied_control.lowlevel.bundle import CommandContract
from embodied_control.lowlevel.command_buffer import InProcessCommandBuffer
from embodied_control.lowlevel.contracts import RobotState
from embodied_control.lowlevel.publishers.gr00t_service import Gr00tServicePublisher

# Stub service: replies with a chunk whose value encodes the request index in
# every element of row r as `100 * index + r`, and records each request line
# to stdout metadata-free. Horizon/width come from argv.
_STUB = textwrap.dedent(
    """
    import json, sys
    horizon, width = int(sys.argv[1]), int(sys.argv[2])
    print(json.dumps({
        "ready": True, "action_horizon": horizon, "action_width": width,
        "state_history": 10, "state_width": 93,
    }), flush=True)
    index = 0
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        chunk = []
        for row in range(horizon):
            chunk.extend([100.0 * index + row] * width)
        reply = {"chunk": chunk, "head_ms": 1.0, "echo_rtc": {
            "has_prev": request.get("prev_chunk") is not None,
            "overlap": request.get("rtc_overlap_steps"),
        }}
        print(json.dumps(reply), flush=True)
        index += 1
    """
)


class _TrackerStub:
    def __init__(self):
        self.last_action = np.zeros(29, dtype=np.float32)


class _EncoderStub:
    """Returns the first z_dim values of its input — enough to prove wiring."""

    def __init__(self, z_dim: int):
        self.z_dim = z_dim
        self.calls: list[np.ndarray] = []

    def infer(self, flat: np.ndarray, out=None) -> np.ndarray:
        self.calls.append(np.array(flat, copy=True))
        return np.asarray(flat[: self.z_dim], dtype=np.float32)


def _state(value: float = 0.0) -> RobotState:
    return RobotState(
        stamp=0.0,
        joint_pos=np.full(29, value, dtype=np.float32),
        joint_vel=np.zeros(29, dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
    )


def _contract(z_dim: int) -> CommandContract:
    return CommandContract(
        z_dim=z_dim,
        phase_mode="sin_cos",
        phase_dim=2,
        hold_steps=10,
        window_steps=9,
        state_dim=38,
        macro_frame_stride=1,
        macro_anchor_mode="robot",
    )


def _publisher(mode: str, z_dim: int, horizon: int, width: int, **kwargs):
    return Gr00tServicePublisher(
        InProcessCommandBuffer(),
        _contract(z_dim),
        [sys.executable, "-c", _STUB, str(horizon), str(width)],
        mode=mode,
        tracker=_TrackerStub(),
        default_joint_pos=np.zeros(29, dtype=np.float32),
        hold_steps=4,
        **kwargs,
    )


def test_latent_openloop_consumes_slots_between_requests():
    publisher = _publisher("latent", z_dim=6, horizon=3, width=6, slots=3)
    buffer = publisher.buffer
    seen = []
    try:
        for tick in range(24):
            publisher.tick(tick, float(tick), _state())
            packet = buffer.snapshot().packet
            seen.append(float(packet.values[0]))
    finally:
        publisher.close()
    # Request 0 covers ticks 0-11 (slots 0,1,2 x hold 4), request 1 covers 12-23.
    expected = [0.0] * 4 + [1.0] * 4 + [2.0] * 4 + [100.0] * 4 + [101.0] * 4 + [102.0] * 4
    assert seen == expected
    # Phase must be sin/cos over the hold.
    packet = buffer.snapshot().packet
    assert packet.values.shape == (8,)


def test_latent_rtc_requests_every_hold_with_prev():
    publisher = _publisher("latent", z_dim=6, horizon=3, width=6, slots=3, rtc=True)
    buffer = publisher.buffer
    seen = []
    try:
        for tick in range(12):
            publisher.tick(tick, float(tick), _state())
            seen.append(float(buffer.snapshot().packet.values[0]))
    finally:
        publisher.close()
    # A fresh request every hold: slot 0 of requests 0,1,2.
    assert seen == [0.0] * 4 + [100.0] * 4 + [200.0] * 4


def test_chunk_mode_encodes_window_through_encoder():
    encoder = _EncoderStub(z_dim=6)
    publisher = _publisher(
        "chunk", z_dim=6, horizon=30, width=38, encoder=encoder
    )
    buffer = publisher.buffer
    try:
        publisher.tick(0, 0.0, _state())
    finally:
        publisher.close()
    assert len(encoder.calls) == 1
    # Encoder input = frames 0..9 flattened (window_steps 9 + state frame).
    assert encoder.calls[0].shape == (10 * 38,)
    assert encoder.calls[0][0] == 0.0  # request 0, row 0
    assert encoder.calls[0][38] == 1.0  # row 1
    packet = buffer.snapshot().packet
    assert float(packet.values[0]) == 0.0


def test_history_frame_layout_uses_tracker_last_action():
    publisher = _publisher("latent", z_dim=6, horizon=3, width=6, slots=3)
    try:
        publisher.tracker.last_action[:] = 7.0
        publisher.tick(0, 0.0, _state(value=0.25))
        history = publisher._history
        assert np.allclose(history[-1][0:29], 0.25)
        assert np.allclose(history[-1][64:93], 7.0)
        # Reset fill: every history row equals the first frame.
        assert np.allclose(history[0], history[-1])
    finally:
        publisher.close()


def test_min_base_height_damps():
    from embodied_control.lowlevel.job import SafetySpec
    from embodied_control.lowlevel.safety import SafetyFault, SafetyMonitor

    monitor = SafetyMonitor(
        spec=SafetySpec(min_base_height_m=0.4), control_hz=50
    )
    low = RobotState(
        stamp=0.0,
        joint_pos=np.zeros(29, dtype=np.float32),
        joint_vel=np.zeros(29, dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
        anchor_pos_w=np.array([0.0, 0.0, 0.3], dtype=np.float32),
        anchor_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    try:
        monitor.check_state(low, now=0.0)
    except SafetyFault as fault:
        assert fault.cause == "base_too_low"
    else:
        raise AssertionError("expected base_too_low fault")
    monitor.check_state(_state(), now=0.0)  # no anchor -> no check
