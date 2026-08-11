"""ec_native shared-memory buffer: pinned CommandBuffer semantics + cross-process.

The whole module skips when the extension is not built
(`pixi run -e native build-native`).
"""

import os
import subprocess
import sys
import time
import uuid

import numpy as np
import pytest

ec_native = pytest.importorskip("ec_native")

from embodied_control.lowlevel.contracts import CommandPacket  # noqa: E402
from embodied_control.lowlevel.native_buffer import ShmCommandBuffer  # noqa: E402


def _name() -> str:
    return f"/ec_native_test_{os.getpid()}_{uuid.uuid4().hex[:8]}"


def _packet(values, sequence=0, interface="latent", stamp=0.0):
    values = np.asarray(values, dtype=np.float32)
    return CommandPacket(
        interface=interface,
        values=values,
        sequence=sequence,
        stamp=stamp,
        terms={"latent_command": values},
    )


def test_empty_snapshot_is_unavailable():
    buffer = ShmCommandBuffer(_name(), create=True)
    snap = buffer.snapshot()
    assert snap.packet is None
    assert snap.age_seconds == float("inf")
    assert not snap.renewed


def test_shared_memory_slot_has_exactly_one_owner():
    name = _name()
    owner = ec_native.ShmCommandSlot(name, True)
    with pytest.raises(RuntimeError, match="already exists"):
        ec_native.ShmCommandSlot(name, True)
    connector = ec_native.ShmCommandSlot(name, False)
    owner.publish(1, 1, np.ones(1, dtype=np.float32), 0.0)
    assert connector.snapshot()[0] == 1


def test_latest_wins_and_renewed_once():
    buffer = ShmCommandBuffer(_name(), create=True)
    buffer.publish(_packet([1.0, 2.0]))
    buffer.publish(_packet([3.0, 4.0]))
    first = buffer.snapshot()
    assert first.renewed
    assert first.packet.sequence == 2
    np.testing.assert_array_equal(first.packet.values, [3.0, 4.0])
    np.testing.assert_array_equal(first.packet.terms["latent_command"], [3.0, 4.0])
    second = buffer.snapshot()
    assert not second.renewed


def test_explicit_stale_sequence_dropped():
    buffer = ShmCommandBuffer(_name(), create=True)
    buffer.publish(_packet([1.0], sequence=7))
    buffer.publish(_packet([2.0], sequence=3))
    snap = buffer.snapshot()
    assert snap.packet.sequence == 7
    assert snap.packet.values[0] == 1.0


def test_age_uses_receive_time_not_sender_stamp():
    buffer = ShmCommandBuffer(_name(), create=True)
    buffer.publish(_packet([1.0], stamp=-12345.0))
    time.sleep(0.05)
    snap = buffer.snapshot()
    assert 0.05 <= snap.age_seconds < 0.5
    assert snap.packet.stamp == -12345.0


def test_interface_tags_round_trip():
    buffer = ShmCommandBuffer(_name(), create=True)
    buffer.publish(_packet(np.zeros(670), interface="chunk"))
    snap = buffer.snapshot()
    assert snap.packet.interface == "chunk"
    assert snap.packet.terms == {}
    assert snap.packet.values.shape == (670,)


def test_oversize_payload_refused():
    buffer = ShmCommandBuffer(_name(), create=True)
    with pytest.raises(RuntimeError, match="kMaxValues"):
        buffer.publish(_packet(np.zeros(ec_native.MAX_VALUES + 1)))


def test_float32_values_bit_exact():
    buffer = ShmCommandBuffer(_name(), create=True)
    rng = np.random.default_rng(0)
    values = rng.standard_normal(258).astype(np.float32)
    buffer.publish(_packet(values))
    np.testing.assert_array_equal(buffer.snapshot().packet.values, values)


def test_native_slot_skips_payload_when_sequence_is_not_new():
    slot = ec_native.ShmCommandSlot(_name(), True)
    slot.publish(7, 1, np.ones(930, dtype=np.float32), 0.0)
    assert slot.snapshot(7) is None
    assert slot.snapshot(6)[0] == 7


_CHILD_PUBLISHER = """
import sys
import numpy as np
import ec_native

slot = ec_native.ShmCommandSlot(sys.argv[1], False)
for sequence in range(1, 51):
    values = np.full(8, float(sequence), dtype=np.float32)
    slot.publish(sequence, 1, values, 0.0)
print("done")
"""


def test_cross_process_delivery():
    name = _name()
    reader = ShmCommandBuffer(name, create=True)
    result = subprocess.run(
        [sys.executable, "-c", _CHILD_PUBLISHER, name],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    snap = reader.snapshot()
    assert snap.packet is not None
    assert snap.packet.sequence == 50
    np.testing.assert_array_equal(snap.packet.values, np.full(8, 50.0, np.float32))


def test_round_trip_latency_smoke():
    buffer = ShmCommandBuffer(_name(), create=True)
    values = np.zeros(258, dtype=np.float32)
    durations = []
    for index in range(2000):
        started = time.perf_counter()
        buffer.publish(_packet(values, sequence=index + 1))
        buffer.snapshot()
        durations.append(time.perf_counter() - started)
    p99_us = float(np.percentile(np.asarray(durations) * 1e6, 99))
    assert p99_us < 1000.0, f"publish+snapshot p99 {p99_us:.1f} us"


def test_monotonic_clock_matches_python():
    native = ec_native.monotonic_now()
    python = time.monotonic()
    assert abs(native - python) < 1.0
