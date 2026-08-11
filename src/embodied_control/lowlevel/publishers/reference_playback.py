"""Explicit publisher: per-tick reference command terms from arrays."""

from __future__ import annotations

import numpy as np

from embodied_control.lowlevel.command_buffer import CommandBuffer
from embodied_control.lowlevel.contracts import CommandPacket


class ReferencePlaybackPublisher:
    """Publish precomputed anchor-frame command terms slot by slot.

    `term_arrays` maps term name to `[T, width]`; `term_order` fixes the
    concatenation order (the bundle's command-term prefix). The arrays must
    already be expressed in the frame the tracker expects (`anchor_b`);
    world-frame re-expression is an M2 decoder concern.
    """

    def __init__(
        self,
        buffer: CommandBuffer,
        term_arrays: dict[str, np.ndarray],
        term_order: list[str],
    ):
        if not term_order:
            raise ValueError("term_order must not be empty")
        self.buffer = buffer
        self.term_order = list(term_order)
        self.arrays = {}
        length = None
        for name in self.term_order:
            if name not in term_arrays:
                raise KeyError(f"term_arrays is missing {name!r}")
            value = np.asarray(term_arrays[name], dtype=np.float32)
            if value.ndim != 2:
                raise ValueError(f"term {name!r} must be [T, width], got {value.shape}")
            if length is None:
                length = value.shape[0]
            elif value.shape[0] != length:
                raise ValueError("all term arrays must share the same length")
            self.arrays[name] = value
        self.length = int(length or 0)
        self._sequence = 0
        self.reset()

    def reset(self) -> None:
        self.cursor = 0

    @property
    def exhausted(self) -> bool:
        return self.cursor >= self.length - 1

    def tick(self, tick: int, stamp: float, state=None) -> None:
        terms = {name: self.arrays[name][self.cursor] for name in self.term_order}
        values = np.concatenate([terms[name] for name in self.term_order])
        self._sequence += 1
        self.buffer.publish(
            CommandPacket(
                interface="explicit",
                values=values,
                sequence=self._sequence,
                stamp=stamp,
                terms=terms,
                metadata={"cursor": self.cursor, "frame": "anchor_b"},
            )
        )
        if not self.exhausted:
            self.cursor += 1


__all__ = ["ReferencePlaybackPublisher"]
