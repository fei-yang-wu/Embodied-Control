"""Per-interface consumption of buffered command packets.

`ChunkCommandSource` reproduces the env's explicit-packet semantics
(`mdp/commands/actor.py:365-390, 631-710`): a packet holds a frame-major
window of explicit command frames expressed in the **publish-time** robot
anchor frame. Each control tick consumes one slot: the window is shifted by
the ticks elapsed since renewal (repeating the final frame past the end),
and position / rot6d components are rigidly re-expressed from the
publish-time anchor into the **live** robot anchor frame
(`delta_quat = current^-1 x renewal`,
`delta_pos = current^-1 . (renewal_pos - current_pos)`). Joint-space
components are frame-invariant.

Packet layout (paper convention, term-major): for each component a
`[window, width]` block flattened row-major, concatenated in term order —
e.g. `expert_motion 10x58 | expert_anchor_pos_b 10x3 | expert_anchor_ori_b
10x6` = 670 values. The publish-time anchor pose rides in packet metadata
(`anchor_pos_w`, `anchor_quat_w`, XYZW); without it the pose is captured at
first consume, one tick late — allowed only when the publisher runs in the
same tick loop.
"""

from __future__ import annotations

import numpy as np

from embodied_control.lowlevel.contracts import CommandSample, RobotState
from embodied_control.lowlevel.maths import (
    quat_conjugate,
    quat_mul,
    quat_to_mat,
    rotate_inverse,
)
from embodied_control.lowlevel.tracker import BufferedCommandSource


class ChunkCommandSource:
    """Adapt chunk packets to per-tick explicit command terms."""

    def __init__(
        self,
        source: BufferedCommandSource,
        components: list[tuple[str, int]],
        *,
        window_steps: int = 10,
    ):
        self.source = source
        self.components = list(components)
        self.window_steps = int(window_steps)
        self.packet_width = self.window_steps * sum(w for _, w in self.components)
        self.reset(None)

    def reset(self, state: RobotState | None) -> None:
        self.source.reset(state)
        self._window: dict[str, np.ndarray] | None = None
        self._anchor: tuple[np.ndarray, np.ndarray] | None = None
        self._phase = 0

    def _split(self, values: np.ndarray) -> dict[str, np.ndarray]:
        if values.shape != (self.packet_width,):
            raise ValueError(
                f"chunk packet must have {self.packet_width} values, got {values.shape}"
            )
        window: dict[str, np.ndarray] = {}
        cursor = 0
        for name, width in self.components:
            block = values[cursor : cursor + self.window_steps * width]
            window[name] = block.reshape(self.window_steps, width).copy()
            cursor += self.window_steps * width
        return window

    def _reexpress(
        self, name: str, frame: np.ndarray, state: RobotState
    ) -> np.ndarray:
        is_position = "_pos" in name
        is_orientation = "_ori" in name
        if self._anchor is None or not (is_position or is_orientation):
            return frame
        if state.anchor_pos_w is None or state.anchor_quat_w is None:
            raise ValueError("chunk re-expression needs the live robot anchor pose")
        renewal_pos, renewal_quat = self._anchor
        delta_quat = quat_mul(quat_conjugate(state.anchor_quat_w), renewal_quat)
        delta_pos = rotate_inverse(
            state.anchor_quat_w, renewal_pos - state.anchor_pos_w
        )
        if is_position:
            vectors = frame.reshape(-1, 3)
            mat = quat_to_mat(delta_quat)
            rotated = vectors @ mat.T
            return (rotated + delta_pos[None, :]).reshape(frame.shape)
        columns = frame.reshape(-1, 3, 2)
        mat = quat_to_mat(delta_quat)
        rotated = np.einsum("ij,njk->nik", mat, columns)
        return rotated.reshape(frame.shape).astype(np.float32)

    def update(self, tick: int, state: RobotState) -> CommandSample:
        sample = self.source.update(tick, state)
        if sample.available and sample.renewed:
            self._window = self._split(np.asarray(sample.vector, dtype=np.float32))
            self._phase = 0
            meta = sample.metadata
            if "anchor_pos_w" in meta and "anchor_quat_w" in meta:
                self._anchor = (
                    np.asarray(meta["anchor_pos_w"], np.float32),
                    np.asarray(meta["anchor_quat_w"], np.float32),
                )
            elif state.anchor_pos_w is not None and state.anchor_quat_w is not None:
                self._anchor = (
                    state.anchor_pos_w.copy(),
                    state.anchor_quat_w.copy(),
                )
            else:
                self._anchor = None
        if self._window is None:
            return CommandSample(
                vector=np.empty(0, np.float32),
                age_ticks=sample.age_ticks,
                renewed=False,
                available=False,
            )
        slot = min(self._phase, self.window_steps - 1)
        overrun = self._phase >= self.window_steps
        terms = {
            name: self._reexpress(name, self._window[name][slot], state)
            for name, _ in self.components
        }
        self._phase += 1
        vector = np.concatenate([terms[name] for name, _ in self.components])
        return CommandSample(
            vector=vector.astype(np.float32),
            age_ticks=sample.age_ticks,
            renewed=sample.renewed,
            terms=terms,
            available=True,
            metadata={**dict(sample.metadata), "slot": slot, "slot_overrun": overrun},
        )


__all__ = ["ChunkCommandSource"]
