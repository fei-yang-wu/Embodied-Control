import numpy as np
import pytest

from embodied_control.lowlevel.command_buffer import (
    InProcessCommandBuffer,
    ZmqCommandBuffer,
    publish_zmq_command,
)
from embodied_control.lowlevel.contracts import CommandPacket


class ManualClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _packet(values, sequence=0, interface="latent", stamp=0.0):
    return CommandPacket(
        interface=interface,
        values=np.asarray(values, dtype=np.float32),
        sequence=sequence,
        stamp=stamp,
        terms={"latent_command": np.asarray(values, dtype=np.float32)},
    )


def test_in_process_empty_snapshot_is_unavailable():
    buffer = InProcessCommandBuffer(clock=ManualClock())
    snap = buffer.snapshot()
    assert snap.packet is None
    assert snap.age_seconds == float("inf")
    assert not snap.renewed


def test_in_process_latest_wins_and_renewed_fires_once():
    clock = ManualClock()
    buffer = InProcessCommandBuffer(clock=clock)
    buffer.publish(_packet([1.0]))
    buffer.publish(_packet([2.0]))
    first = buffer.snapshot()
    assert first.renewed
    assert first.packet.values[0] == 2.0
    second = buffer.snapshot()
    assert not second.renewed
    assert second.packet.values[0] == 2.0


def test_in_process_age_uses_receive_time():
    clock = ManualClock()
    buffer = InProcessCommandBuffer(clock=clock)
    clock.t = 5.0
    buffer.publish(_packet([1.0], stamp=123456.0))
    clock.t = 5.25
    snap = buffer.snapshot()
    assert snap.age_seconds == pytest.approx(0.25)


def test_in_process_rejects_stale_explicit_sequence():
    buffer = InProcessCommandBuffer(clock=ManualClock())
    buffer.publish(_packet([1.0], sequence=7))
    buffer.publish(_packet([2.0], sequence=3))
    snap = buffer.snapshot()
    assert snap.packet.sequence == 7
    assert snap.packet.values[0] == 1.0


def test_in_process_autoassigns_monotonic_sequences():
    buffer = InProcessCommandBuffer(clock=ManualClock())
    buffer.publish(_packet([1.0]))
    buffer.publish(_packet([2.0]))
    assert buffer.snapshot().packet.sequence == 2


def _zmq_pair():
    zmq = pytest.importorskip("zmq")
    context = zmq.Context.instance()
    receiver = context.socket(zmq.PULL)
    port = receiver.bind_to_random_port("tcp://127.0.0.1")
    sender = context.socket(zmq.PUSH)
    sender.connect(f"tcp://127.0.0.1:{port}")
    return zmq, sender, receiver


def _drain_until(buffer, predicate, tries=200):
    import time

    for _ in range(tries):
        snap = buffer.snapshot()
        if predicate(snap):
            return snap
        time.sleep(0.005)
    raise AssertionError("condition not reached")


def test_zmq_round_trip_and_renewed_semantics():
    _zmq, sender, receiver = _zmq_pair()
    clock = ManualClock()
    buffer = ZmqCommandBuffer("unused", socket=receiver, clock=clock)
    publish_zmq_command(sender, _packet([1.0, 2.0], sequence=1, stamp=99.0))
    snap = _drain_until(buffer, lambda s: s.packet is not None)
    assert snap.renewed
    assert snap.packet.interface == "latent"
    np.testing.assert_allclose(snap.packet.values, [1.0, 2.0])
    np.testing.assert_allclose(snap.packet.terms["latent_command"], [1.0, 2.0])
    again = buffer.snapshot()
    assert not again.renewed
    assert again.packet is not None


def test_zmq_age_uses_receive_time_not_sender_stamp():
    _zmq, sender, receiver = _zmq_pair()
    clock = ManualClock()
    clock.t = 10.0
    buffer = ZmqCommandBuffer("unused", socket=receiver, clock=clock)
    publish_zmq_command(sender, _packet([1.0], sequence=1, stamp=-500.0))
    _drain_until(buffer, lambda s: s.packet is not None)
    clock.t = 10.5
    snap = buffer.snapshot()
    assert snap.age_seconds == pytest.approx(0.5, abs=1e-6)


def test_zmq_drops_stale_sequences_keeps_latest():
    _zmq, sender, receiver = _zmq_pair()
    buffer = ZmqCommandBuffer("unused", socket=receiver, clock=ManualClock())
    publish_zmq_command(sender, _packet([1.0], sequence=5))
    publish_zmq_command(sender, _packet([2.0], sequence=4))
    publish_zmq_command(sender, _packet([3.0], sequence=6))
    snap = _drain_until(
        buffer, lambda s: s.packet is not None and s.packet.sequence == 6
    )
    assert snap.packet.values[0] == 3.0


def test_zmq_publish_refused():
    _zmq, _sender, receiver = _zmq_pair()
    buffer = ZmqCommandBuffer("unused", socket=receiver)
    with pytest.raises(RuntimeError):
        buffer.publish(_packet([1.0]))


def test_payload_rejects_unknown_interface():
    from embodied_control.lowlevel.command_buffer import _payload_to_packet

    with pytest.raises(ValueError):
        _payload_to_packet({"interface": "telepathy", "values": []})
