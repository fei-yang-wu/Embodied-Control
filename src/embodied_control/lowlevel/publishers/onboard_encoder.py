"""Latent publisher: macro window -> encoder engine -> z ++ phase."""

from __future__ import annotations

import math

import numpy as np

from embodied_control.lowlevel.bundle import CommandContract
from embodied_control.lowlevel.command_buffer import CommandBuffer
from embodied_control.lowlevel.contracts import CommandPacket, RobotState
from embodied_control.lowlevel.engine.base import Engine
from embodied_control.lowlevel.macro_window import (
    fill_precomputed_window,
    fill_robot_anchored_window,
    window_indices,
)
from embodied_control.lowlevel.reference import ReferenceMotion


class OnboardEncoderPublisher:
    """Reproduce the frozen sampler's hold/renewal schedule outside Isaac.

    Two window sources:

    - `macro_states` (`[T, state_dim]`): precomputed frames, valid when the
      frames do not depend on the live robot (`expert_heading` bundles, or
      synthetic plumbing runs).
    - `motion` (`ReferenceMotion`) with `macro_anchor_mode == "robot"`: the
      window is built at encode time — per frame `[joint qpos 29 |
      expert anchor pos 3 | expert anchor ori rot6d 6]`, with the expert
      anchor world pose re-expressed in the LIVE robot anchor frame
      (`expert_data_plane` rollout context). Needs `state.anchor_pos_w` /
      `anchor_quat_w` from the backend.

    At renewal the window takes frames `cursor + stride*k` for
    `k in 0..window_steps`, the first as `state` and the rest flattened
    behind it — the `intermediate` window mode. Every tick publishes
    `z ++ [sin(2*pi*phase), cos(2*pi*phase)]` when the contract has a
    sin_cos phase, else `z` alone.
    """

    def __init__(
        self,
        buffer: CommandBuffer,
        encoder: Engine,
        command: CommandContract,
        macro_states: np.ndarray | None = None,
        *,
        motion: ReferenceMotion | None = None,
        hold_steps: int | None = None,
    ):
        if (macro_states is None) == (motion is None):
            raise ValueError("pass exactly one of macro_states or motion")
        if command.window_steps is None:
            raise ValueError("command contract is missing window_steps")
        if command.z_dim is None:
            raise ValueError("command contract is missing z_dim")
        if command.state_dim is None:
            raise ValueError("command contract is missing state_dim")
        self.buffer = buffer
        self.encoder = encoder
        self.command = command
        self.stride = int(command.macro_frame_stride or 1)
        self.window_steps = int(command.window_steps)
        self.hold_steps = int(hold_steps if hold_steps is not None else command.hold_steps)
        self.motion = motion
        if macro_states is not None:
            states = np.asarray(macro_states, dtype=np.float32)
            if states.ndim != 2:
                raise ValueError(f"macro_states must be [T, state_dim], got {states.shape}")
            if states.shape[1] != command.state_dim:
                raise ValueError(
                    f"macro_states width {states.shape[1]} != contract state_dim "
                    f"{command.state_dim}"
                )
            self.states: np.ndarray | None = states
            self.length = states.shape[0]
        else:
            assert motion is not None
            if command.macro_anchor_mode != "robot":
                raise ValueError(
                    "motion-driven windows are implemented for macro_anchor_mode="
                    f"'robot'; bundle says {command.macro_anchor_mode!r}. Use "
                    "precomputed macro_states for other modes."
                )
            if command.state_dim != motion.joint_qpos.shape[1] + 9:
                raise ValueError(
                    f"state_dim {command.state_dim} != qpos {motion.joint_qpos.shape[1]} "
                    "+ 3 anchor pos + 6 rot6d"
                )
            self.states = None
            self.length = motion.length
        self._encoder_in = np.empty(
            command.state_dim * (self.window_steps + 1), dtype=np.float32
        )
        self._sequence = 0
        self.reset()

    def reset(self) -> None:
        self.cursor = 0
        self._steps_remaining = 0
        self._z: np.ndarray | None = None

    @property
    def exhausted(self) -> bool:
        return self.cursor >= self.length - 1

    def _window_indices(self, cursor: int) -> list[int]:
        return window_indices(cursor, self.length, self.window_steps, self.stride)

    def _fill_precomputed(self, cursor: int) -> None:
        assert self.states is not None
        fill_precomputed_window(
            self.states, self._window_indices(cursor), self._encoder_in
        )

    def _fill_robot_anchored(self, cursor: int, state: RobotState) -> None:
        assert self.motion is not None
        if state.anchor_pos_w is None or state.anchor_quat_w is None:
            raise ValueError(
                "robot-anchored encoding needs state.anchor_pos_w/anchor_quat_w; "
                "the backend does not provide an anchor pose"
            )
        fill_robot_anchored_window(
            self.motion,
            self._window_indices(cursor),
            state.anchor_pos_w,
            state.anchor_quat_w,
            int(self.command.state_dim),
            self._encoder_in,
        )

    def tick(self, tick: int, stamp: float, state: RobotState | None = None) -> None:
        if self._z is None or self._steps_remaining <= 0:
            if self.states is not None:
                self._fill_precomputed(self.cursor)
            else:
                if state is None:
                    raise ValueError("robot-anchored encoding needs the robot state")
                self._fill_robot_anchored(self.cursor, state)
            self._z = np.asarray(self.encoder.infer(self._encoder_in), dtype=np.float32)
            if self._z.shape != (self.command.z_dim,):
                raise ValueError(
                    f"encoder produced {self._z.shape}, expected ({self.command.z_dim},)"
                )
            self._steps_remaining = self.hold_steps
        if self.command.phase_mode == "sin_cos":
            phase = (self.hold_steps - self._steps_remaining) / float(self.hold_steps)
            values = np.concatenate(
                [
                    self._z,
                    np.array(
                        [math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)],
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
                metadata={"cursor": self.cursor, "steps_remaining": self._steps_remaining},
            )
        )
        self._steps_remaining -= 1
        if not self.exhausted:
            self.cursor += 1


__all__ = ["OnboardEncoderPublisher"]
