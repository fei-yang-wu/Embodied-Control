"""Command-buffer seams shared by VLA publishers and tracker consumers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import threading
import time
from typing import Protocol

import numpy as np

from embodied_control.lowlevel.contracts import CommandPacket, CommandSnapshot


class CommandBuffer(Protocol):
    """Non-blocking latest-command interface.

    The publisher owns command production. The tracker owns consumption. A
    consumer never waits for a planner response on its 50 Hz control tick.

    Pinned semantics, identical for every implementation:

    - ``sequence`` is monotonic per publisher; stale or repeated sequences
      are dropped.
    - ``snapshot().renewed`` is true when a sequence not seen by a previous
      ``snapshot()`` call has arrived. One consumer per buffer.
    - ``snapshot().age_seconds`` measures time since the packet was
      *received*, on the consumer's clock. The sender stamp in the packet is
      diagnostic only: monotonic clocks do not compare across hosts.
    """

    def publish(self, packet: CommandPacket) -> None: ...
    def snapshot(self, now: float | None = None) -> CommandSnapshot: ...


class InProcessCommandBuffer:
    """Thread-safe single-slot buffer for local VLA and tracker processes."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._packet: CommandPacket | None = None
        self._received_at = 0.0
        self._next_sequence = 0
        self._last_snapshot_sequence = -1

    def publish(self, packet: CommandPacket) -> None:
        with self._lock:
            if packet.sequence <= 0:
                self._next_sequence += 1
                packet = CommandPacket(
                    interface=packet.interface,
                    values=packet.values,
                    sequence=self._next_sequence,
                    stamp=packet.stamp,
                    terms=packet.terms,
                    metadata=packet.metadata,
                )
            elif packet.sequence <= self._next_sequence:
                return
            else:
                self._next_sequence = packet.sequence
            self._packet = packet.copy()
            self._received_at = self._clock()

    def snapshot(self, now: float | None = None) -> CommandSnapshot:
        current = self._clock() if now is None else float(now)
        with self._lock:
            packet = None if self._packet is None else self._packet.copy()
            received_at = self._received_at
            renewed = packet is not None and packet.sequence != self._last_snapshot_sequence
            if packet is not None:
                self._last_snapshot_sequence = packet.sequence
        age = float("inf") if packet is None else max(0.0, current - received_at)
        return CommandSnapshot(packet=packet, age_seconds=age, renewed=renewed)


def _packet_to_payload(packet: CommandPacket) -> dict:
    return {
        "interface": packet.interface,
        "values": np.asarray(packet.values, dtype=np.float32).tolist(),
        "sequence": int(packet.sequence),
        "stamp": float(packet.stamp),
        "terms": {
            name: np.asarray(value, dtype=np.float32).tolist()
            for name, value in packet.terms.items()
        },
        "metadata": dict(packet.metadata),
    }


def _payload_to_packet(payload: Mapping) -> CommandPacket:
    if not isinstance(payload, Mapping):
        raise TypeError("command payload must be a mapping")
    interface = str(payload.get("interface", ""))
    if interface not in {"explicit", "latent", "chunk"}:
        raise ValueError(f"unsupported command interface: {interface!r}")
    return CommandPacket(
        interface=interface,  # type: ignore[arg-type]
        values=np.asarray(payload.get("values", []), dtype=np.float32),
        sequence=int(payload.get("sequence", 0)),
        stamp=float(payload.get("stamp", time.monotonic())),
        terms={
            name: np.asarray(value, dtype=np.float32)
            for name, value in dict(payload.get("terms", {})).items()
        },
        metadata=dict(payload.get("metadata", {})),
    )


class ZmqCommandBuffer:
    """Latest-command ZMQ SUB buffer using a msgpack-compatible payload.

    JSON is retained as a dependency-light fallback. When ``msgpack`` is
    installed, the wire format is a single msgpack mapping, which is also the
    shape used by the GR00T service boundary.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        topic: bytes = b"",
        socket=None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - optional feature
            raise ImportError("ZMQ command buffers require pyzmq") from exc
        self._zmq = zmq
        self._clock = clock
        self._context = None
        if socket is None:
            self._context = zmq.Context.instance()
            socket = self._context.socket(zmq.SUB)
            socket.setsockopt(zmq.SUBSCRIBE, topic)
            socket.connect(endpoint)
        self._socket = socket
        self._topic = topic
        self._packet: CommandPacket | None = None
        self._received_at = 0.0
        self._last_sequence = -1
        self._last_snapshot_sequence = -1

    def publish(self, packet: CommandPacket) -> None:
        raise RuntimeError("ZmqCommandBuffer is a consumer; use a VLA publisher socket")

    def _decode(self, message: bytes) -> CommandPacket:
        payload = (
            message[len(self._topic) :]
            if self._topic and message.startswith(self._topic)
            else message
        )
        try:
            import msgpack

            decoded = msgpack.unpackb(payload, raw=False)
        except ImportError:
            decoded = json.loads(payload.decode("utf-8"))
        return _payload_to_packet(decoded)

    def snapshot(self, now: float | None = None) -> CommandSnapshot:
        current = self._clock() if now is None else float(now)
        while True:
            try:
                message = self._socket.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            packet = self._decode(message)
            if packet.sequence <= self._last_sequence:
                continue
            self._last_sequence = packet.sequence
            self._packet = packet.copy()
            self._received_at = self._clock() if now is None else float(now)
        packet = self._packet
        renewed = packet is not None and packet.sequence != self._last_snapshot_sequence
        if packet is not None:
            self._last_snapshot_sequence = packet.sequence
        age = float("inf") if packet is None else max(0.0, current - self._received_at)
        return CommandSnapshot(
            packet=None if packet is None else packet.copy(),
            age_seconds=age,
            renewed=renewed,
        )


def publish_zmq_command(socket, packet: CommandPacket, *, topic: bytes = b"") -> None:
    """Publish one command packet from a VLA process."""
    payload = _packet_to_payload(packet)
    try:
        import msgpack

        encoded = msgpack.packb(payload, use_bin_type=True)
    except ImportError:
        encoded = json.dumps(payload).encode("utf-8")
    socket.send(topic + encoded)


__all__ = [
    "CommandBuffer",
    "InProcessCommandBuffer",
    "ZmqCommandBuffer",
    "publish_zmq_command",
]
