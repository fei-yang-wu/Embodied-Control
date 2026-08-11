"""Relay semantics against a fake line-JSON chunk service."""

import os
import sys
import time
import uuid

import numpy as np
import pytest

ec_native = pytest.importorskip("ec_native")

from embodied_control.lowlevel.publishers.stdio_chunk_relay import (  # noqa: E402
    CHUNK_RESPONSE_TAG,
    PLANNER_REQUEST_TAG,
    StdioChunkRelay,
)

_FAKE_SERVICE = r"""
import json, sys
print(json.dumps({"ready": True, "action_horizon": 30}), flush=True)
print("banner noise that must be ignored", flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("stop"):
        break
    state = request["state"]
    chunk = [float(state[0])] * (30 * 38)
    print(json.dumps({"chunk": chunk, "head_ms": 1.0}), flush=True)
"""


def _name(prefix):
    return f"/{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def test_relay_pairs_sequences_and_truncates_chunks():
    request_name = _name("relay_req")
    response_name = _name("relay_resp")
    relay = StdioChunkRelay(
        request_name, response_name,
        [sys.executable, "-u", "-c", _FAKE_SERVICE],
        create_slots=True,
    )
    assert relay.ready["ready"]
    relay.start()
    requester = ec_native.ShmCommandSlot(request_name, False)
    responder = ec_native.ShmCommandSlot(response_name, False)
    try:
        for sequence in (5, 9):
            state = np.full(930, float(sequence), dtype=np.float32)
            requester.publish(sequence, PLANNER_REQUEST_TAG, state, 0.0)
            deadline = time.monotonic() + 10
            raw = None
            while time.monotonic() < deadline:
                raw = responder.snapshot()
                if raw is not None and raw[0] == sequence:
                    break
                time.sleep(0.002)
            assert raw is not None and raw[0] == sequence
            got_sequence, tag, values, _recv, _sender = raw
            assert tag == CHUNK_RESPONSE_TAG
            assert np.asarray(values).shape == (20 * 38,)  # truncated to slot budget
            np.testing.assert_allclose(values, float(sequence))
        assert relay.requests_relayed == 2
        assert relay.head_ms == [1.0, 1.0]
    finally:
        relay.close()


def test_relay_ignores_foreign_tags():
    request_name = _name("relay_req2")
    response_name = _name("relay_resp2")
    relay = StdioChunkRelay(
        request_name, response_name,
        [sys.executable, "-u", "-c", _FAKE_SERVICE],
        create_slots=True,
    )
    relay.start()
    requester = ec_native.ShmCommandSlot(request_name, False)
    try:
        requester.publish(3, 11, np.zeros(2, dtype=np.float32), 0.0)  # oracle tag
        time.sleep(0.2)
        assert relay.requests_relayed == 0
    finally:
        relay.close()