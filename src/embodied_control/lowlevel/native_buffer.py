"""Shared-memory command buffer backed by the optional `ec_native` extension.

Broker-less same-host cross-process delivery: a VLA process publishes into a
named POSIX shm slot; the tracker process snapshots it. Implements the pinned
`CommandBuffer` semantics (publisher-lifetime monotonic sequences, latest
wins, `renewed` = new sequence since the previous snapshot, age from receive
time). The C++ side stamps CLOCK_MONOTONIC at publish, which is the same
system-wide clock Python's `time.monotonic()` reads on Linux.

v0 wire format is values-only: named terms are reconstructed for the latent
interface (`latent_command`); explicit/chunk multi-term splitting by contract
widths is an M2 decoder concern.
"""

from __future__ import annotations

from collections.abc import Callable
import time

import numpy as np

from embodied_control.lowlevel.contracts import CommandPacket, CommandSnapshot

_INTERFACE_TO_TAG = {"explicit": 0, "latent": 1, "chunk": 2}
_TAG_TO_INTERFACE = {tag: name for name, tag in _INTERFACE_TO_TAG.items()}


def native_available() -> bool:
    try:
        import ec_native  # noqa: F401
    except ImportError:
        return False
    return True


class ShmCommandBuffer:
    """`CommandBuffer` over a named shm slot; `create=True` on the owner side."""

    def __init__(
        self,
        name: str,
        *,
        create: bool,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - env without the build
            raise ImportError(
                "ShmCommandBuffer needs the ec_native extension; build it with "
                "`pixi run -e native build-native`"
            ) from exc
        self._slot = ec_native.ShmCommandSlot(name, create)
        self._clock = clock
        self._next_sequence = 0
        self._last_snapshot_sequence = -1

    def publish(self, packet: CommandPacket) -> None:
        if packet.sequence <= 0:
            self._next_sequence += 1
            sequence = self._next_sequence
        elif packet.sequence <= self._next_sequence:
            return
        else:
            self._next_sequence = packet.sequence
            sequence = packet.sequence
        values = np.ascontiguousarray(packet.values, dtype=np.float32).reshape(-1)
        self._slot.publish(
            sequence, _INTERFACE_TO_TAG[packet.interface], values, packet.stamp
        )

    def snapshot(self, now: float | None = None) -> CommandSnapshot:
        current = self._clock() if now is None else float(now)
        raw = self._slot.snapshot()
        if raw is None:
            return CommandSnapshot(packet=None, age_seconds=float("inf"), renewed=False)
        sequence, tag, values, recv_stamp, sender_stamp = raw
        interface = _TAG_TO_INTERFACE.get(int(tag))
        if interface is None:
            raise ValueError(f"unknown interface tag {tag}")
        values = np.asarray(values, dtype=np.float32)
        terms = {"latent_command": values} if interface == "latent" else {}
        renewed = sequence != self._last_snapshot_sequence
        self._last_snapshot_sequence = sequence
        packet = CommandPacket(
            interface=interface,  # type: ignore[arg-type]
            values=values,
            sequence=int(sequence),
            stamp=float(sender_stamp),
            terms=terms,
        )
        age = max(0.0, current - float(recv_stamp))
        return CommandSnapshot(packet=packet, age_seconds=age, renewed=renewed)


__all__ = ["ShmCommandBuffer", "native_available"]
