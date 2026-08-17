"""Latent command publisher whose z comes from a pluggable source.

`OnboardEncoderPublisher` is the certified reference path: reference window ->
encoder -> z, on the frozen hold schedule. This module keeps that schedule but
lets the z itself come from anywhere, which is what latent-space experiments
need:

- `ReferenceEncoderSource` reproduces the certified behaviour (parity is
  asserted in `tests/lowlevel/test_latent_playground.py`);
- `TransformedLatentSource` wraps any source with `fn(z, renewal_index) -> z`
  (add noise, move along a direction, scale, zero dimensions);
- `ConstantLatentSource` holds one z forever, with no reference at all;
- `SequenceLatentSource` plays a precomputed list of z, one per renewal.

The published vector is `z ++ [sin(2*pi*phase), cos(2*pi*phase)]` when the
bundle's command contract carries a sin/cos phase, else `z` alone. Phase runs
`0 -> (hold-1)/hold` across the hold window, exactly as in the frozen sampler.
"""

from __future__ import annotations

import math
from typing import Callable, Protocol

import numpy as np

from embodied_control.lowlevel.bundle import CommandContract
from embodied_control.lowlevel.command_buffer import CommandBuffer
from embodied_control.lowlevel.contracts import CommandPacket, RobotState
from embodied_control.lowlevel.engine.base import Engine
from embodied_control.lowlevel.macro_window import (
    ROBOT_ANCHOR_MODES,
    fill_precomputed_window,
    fill_robot_anchored_window,
    window_indices,
)
from embodied_control.lowlevel.reference import ReferenceMotion


class LatentSource(Protocol):
    """Produce one z per renewal; optionally end the episode."""

    def reset(self) -> None: ...

    def z(self, renewal_index: int, state: RobotState | None) -> np.ndarray: ...

    def after_tick(self) -> None: ...

    @property
    def exhausted(self) -> bool: ...


class ReferenceEncoderSource:
    """Encode the reference macro window at the live cursor — the oracle z.

    Pass either `motion` (robot-anchored windows, `macro_anchor_mode="robot"`
    or `"robot_heading"`) or `macro_states` (`[T, state_dim]` precomputed
    frames).
    """

    def __init__(
        self,
        encoder: Engine,
        command: CommandContract,
        *,
        motion: ReferenceMotion | None = None,
        macro_states: np.ndarray | None = None,
        start_frame: int = 0,
    ):
        if (motion is None) == (macro_states is None):
            raise ValueError("pass exactly one of motion or macro_states")
        for field in ("window_steps", "z_dim", "state_dim"):
            if getattr(command, field) is None:
                raise ValueError(f"command contract is missing {field}")
        self.encoder = encoder
        self.command = command
        self.stride = int(command.macro_frame_stride or 1)
        self.window_steps = int(command.window_steps)
        self.state_dim = int(command.state_dim)
        self.z_dim = int(command.z_dim)
        self.motion = motion
        self.start_frame = int(start_frame)
        if macro_states is not None:
            states = np.asarray(macro_states, dtype=np.float32)
            if states.ndim != 2 or states.shape[1] != self.state_dim:
                raise ValueError(
                    f"macro_states must be [T, {self.state_dim}], got {states.shape}"
                )
            self.states: np.ndarray | None = states
            self.length = int(states.shape[0])
        else:
            assert motion is not None
            if command.macro_anchor_mode not in ROBOT_ANCHOR_MODES:
                raise ValueError(
                    "motion-driven windows are implemented for "
                    f"macro_anchor_mode in {ROBOT_ANCHOR_MODES}; bundle says "
                    f"{command.macro_anchor_mode!r}"
                )
            self.states = None
            self.length = motion.length
        self.anchor_mode = str(command.macro_anchor_mode or "robot")
        self._encoder_in = np.empty(
            self.state_dim * (self.window_steps + 1), dtype=np.float32
        )
        self.reset()

    def reset(self) -> None:
        self.cursor = min(self.start_frame, self.length - 1)

    @property
    def exhausted(self) -> bool:
        return self.cursor >= self.length - 1

    def window(self, state: RobotState | None) -> np.ndarray:
        indices = window_indices(self.cursor, self.length, self.window_steps, self.stride)
        if self.states is not None:
            return fill_precomputed_window(self.states, indices, self._encoder_in)
        assert self.motion is not None
        if state is None or state.anchor_pos_w is None or state.anchor_quat_w is None:
            raise ValueError(
                "robot-anchored encoding needs state.anchor_pos_w/anchor_quat_w"
            )
        return fill_robot_anchored_window(
            self.motion,
            indices,
            state.anchor_pos_w,
            state.anchor_quat_w,
            self.state_dim,
            self._encoder_in,
            anchor_mode=self.anchor_mode,
        )

    def z(self, renewal_index: int, state: RobotState | None) -> np.ndarray:
        del renewal_index
        return np.asarray(self.encoder.infer(self.window(state)), dtype=np.float32)

    def after_tick(self) -> None:
        if not self.exhausted:
            self.cursor += 1


class ConstantLatentSource:
    """Publish one fixed z on every renewal; never exhausts."""

    def __init__(self, z: np.ndarray):
        self._z = np.asarray(z, dtype=np.float32).copy()
        if self._z.ndim != 1:
            raise ValueError(f"z must be 1-D, got {self._z.shape}")
        self.z_dim = int(self._z.shape[0])

    def reset(self) -> None:
        return

    def z(self, renewal_index: int, state: RobotState | None) -> np.ndarray:
        del renewal_index, state
        return self._z

    def after_tick(self) -> None:
        return

    @property
    def exhausted(self) -> bool:
        return False


class SequenceLatentSource:
    """Play `[K, z_dim]` latents, one per renewal.

    After the last entry the source either exhausts (`hold_last=False`) or
    keeps republishing the final z.
    """

    def __init__(self, latents: np.ndarray, *, hold_last: bool = True):
        values = np.asarray(latents, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"latents must be [K, z_dim], got {values.shape}")
        self.latents = values
        self.z_dim = int(values.shape[1])
        self.hold_last = bool(hold_last)
        self.reset()

    def reset(self) -> None:
        self._index = 0

    @property
    def exhausted(self) -> bool:
        return not self.hold_last and self._index >= self.latents.shape[0]

    def z(self, renewal_index: int, state: RobotState | None) -> np.ndarray:
        del state
        index = min(renewal_index, self.latents.shape[0] - 1)
        self._index = renewal_index + 1
        return self.latents[index]

    def after_tick(self) -> None:
        return


class TransformedLatentSource:
    """Apply `fn(z, renewal_index) -> z` to another source's output."""

    def __init__(
        self,
        inner: LatentSource,
        fn: Callable[[np.ndarray, int], np.ndarray],
    ):
        self.inner = inner
        self.fn = fn
        self.z_dim = int(getattr(inner, "z_dim", 0))

    def __getattr__(self, name: str):
        """Expose the wrapped source's state (`cursor`, `motion`, ...).

        A rollout reads `source.cursor` to record which reference frame each
        tick tracked; a wrapper must not hide it.
        """
        if name in {"inner", "fn"}:
            raise AttributeError(name)
        return getattr(self.__dict__["inner"], name)

    def reset(self) -> None:
        self.inner.reset()

    @property
    def exhausted(self) -> bool:
        return self.inner.exhausted

    def z(self, renewal_index: int, state: RobotState | None) -> np.ndarray:
        base = self.inner.z(renewal_index, state)
        out = np.asarray(self.fn(base, renewal_index), dtype=np.float32)
        if out.shape != base.shape:
            raise ValueError(
                f"transform changed the z shape: {base.shape} -> {out.shape}"
            )
        return out

    def after_tick(self) -> None:
        self.inner.after_tick()


class LatentPublisher:
    """Publish `z ++ phase` on the contract's hold schedule.

    Records every published z in `z_trace` (one row per renewal) so an
    experiment can report exactly what the tracker consumed.
    """

    def __init__(
        self,
        buffer: CommandBuffer,
        source: LatentSource,
        command: CommandContract,
        *,
        hold_steps: int | None = None,
        max_renewals: int | None = None,
    ):
        if command.z_dim is None:
            raise ValueError("command contract is missing z_dim")
        self.buffer = buffer
        self.source = source
        self.command = command
        self.z_dim = int(command.z_dim)
        self.hold_steps = int(
            hold_steps if hold_steps is not None else command.hold_steps
        )
        if self.hold_steps < 1:
            raise ValueError("hold_steps must be >= 1")
        self.max_renewals = None if max_renewals is None else int(max_renewals)
        self._sequence = 0
        self.reset()

    def reset(self) -> None:
        self.source.reset()
        self._steps_remaining = 0
        self._renewals = 0
        self._z: np.ndarray | None = None
        self.z_trace: list[np.ndarray] = []

    @property
    def renewals(self) -> int:
        return self._renewals

    @property
    def exhausted(self) -> bool:
        if self.max_renewals is not None and self._renewals >= self.max_renewals:
            return True
        return bool(self.source.exhausted)

    def tick(self, tick: int, stamp: float, state: RobotState | None = None) -> None:
        del tick
        if self._z is None or self._steps_remaining <= 0:
            z = np.asarray(self.source.z(self._renewals, state), dtype=np.float32)
            if z.shape != (self.z_dim,):
                raise ValueError(
                    f"source produced {z.shape}, expected ({self.z_dim},)"
                )
            self._z = z
            self.z_trace.append(z.copy())
            self._renewals += 1
            self._steps_remaining = self.hold_steps
        if self.command.phase_mode == "sin_cos":
            phase = (self.hold_steps - self._steps_remaining) / float(self.hold_steps)
            values = np.concatenate(
                [
                    self._z,
                    np.array(
                        [
                            math.sin(2.0 * math.pi * phase),
                            math.cos(2.0 * math.pi * phase),
                        ],
                        dtype=np.float32,
                    ),
                ]
            )
        else:
            values = self._z
        self._sequence += 1
        self.buffer.publish(
            CommandPacket(
                interface="latent",
                values=values,
                sequence=self._sequence,
                stamp=stamp,
                terms={"latent_command": values},
                metadata={
                    "renewal": self._renewals - 1,
                    "steps_remaining": self._steps_remaining,
                    "cursor": getattr(self.source, "cursor", -1),
                },
            )
        )
        self._steps_remaining -= 1
        self.source.after_tick()


__all__ = [
    "ConstantLatentSource",
    "LatentPublisher",
    "LatentSource",
    "ReferenceEncoderSource",
    "SequenceLatentSource",
    "TransformedLatentSource",
]
