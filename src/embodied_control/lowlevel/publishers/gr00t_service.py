"""GR00T-head publisher: causal history -> stdio service -> latent commands.

Drives the line-JSON chunk service
(`imitation_experiments.planner.gr00t_chunk_service`) synchronously from the
publisher tick — valid only for stepped (non-realtime) evaluation, where a
blocking ~60 ms GPU inference inside a tick costs wall-clock but not
correctness.

Two head target modes:

- ``latent``: the head predicts `slots` consecutive published latents
  (`[slots, z_dim]`), each held `hold_steps` ticks. Open-loop (the basic
  arm): one inference per `slots * hold_steps` ticks, slots consumed
  sequentially. RTC: one inference per `hold_steps` ticks with the previous
  prediction seeding the `slots - 1` overlapping slots; slot 0 of the fresh
  prediction is published.
- ``chunk``: the head predicts a `[horizon, 38]` root_qpos window anchored
  at the request-time robot anchor. Each publication requests fresh, feeds
  frames `0..window_steps` to the bundle's tracker-side encoder
  (`[state; window]` flat), and publishes the resulting z. RTC passes the
  previous window with overlap `horizon - hold_steps` (frame 0 aligns with
  its request state).

Every tick publishes ``z ++ [sin, cos]`` phase over the hold, matching the
frozen sampler (`onboard_encoder.py`). FSQ snapping stays in the tracker
(consume time); this publisher forwards the head's pre-quantization values.

The 10x93 causal history mirrors Isaac's `g1_causal_robot_history` spec:
consecutive control ticks, frame = `[joint_pos_rel 29 | joint_vel 29 |
base_ang_vel 3 | projected_gravity 3 | last_action 29]`, oldest -> newest,
reset-filled with the first frame.
"""

from __future__ import annotations

import json
import math
import subprocess
import time

import numpy as np

from embodied_control.lowlevel.bundle import CommandContract
from embodied_control.lowlevel.command_buffer import CommandBuffer
from embodied_control.lowlevel.contracts import CommandPacket, RobotState
from embodied_control.lowlevel.engine.base import Engine
from embodied_control.lowlevel.tracker import LowLevelTracker

STATE_FRAMES = 10
STATE_WIDTH = 93


class Gr00tServicePublisher:
    def __init__(
        self,
        buffer: CommandBuffer,
        command: CommandContract,
        service_command: list[str],
        *,
        mode: str,
        tracker: LowLevelTracker,
        default_joint_pos: np.ndarray,
        encoder: Engine | None = None,
        hold_steps: int = 10,
        slots: int = 3,
        rtc: bool = False,
        rtc_freeze_steps: int = 0,
        rtc_ramp_rate: float = 5.0,
        service_cwd: str | None = None,
        ready_timeout_s: float = 600.0,
        max_publications: int | None = None,
        goal_schedule: list[tuple[int, str]] | None = None,
    ):
        if mode not in {"latent", "chunk"}:
            raise ValueError(f"mode must be latent|chunk, got {mode!r}")
        if command.z_dim is None:
            raise ValueError("command contract is missing z_dim")
        if mode == "chunk" and encoder is None:
            raise ValueError("chunk mode needs the bundle's encoder engine")
        self.buffer = buffer
        self.command = command
        self.mode = mode
        self.tracker = tracker
        self.encoder = encoder
        self.hold_steps = int(hold_steps)
        self.slots = int(slots)
        self.rtc = bool(rtc)
        self.rtc_freeze_steps = int(rtc_freeze_steps)
        self.rtc_ramp_rate = float(rtc_ramp_rate)
        self.max_publications = max_publications
        # Ascending [(start_tick, goal_name)]; requests at or after start_tick
        # carry that goal. Empty/None = the service's startup goal throughout.
        self.goal_schedule = sorted(goal_schedule or [], key=lambda item: item[0])
        self._default_joint_pos = np.asarray(default_joint_pos, dtype=np.float32)
        self.head_ms: list[float] = []

        self._service = subprocess.Popen(
            service_command,
            cwd=service_cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready = self._read_json(timeout_s=ready_timeout_s)
        if not self.ready.get("ready"):
            raise RuntimeError(f"gr00t service did not become ready: {self.ready}")
        self.horizon = int(self.ready["action_horizon"])
        self.action_width = int(self.ready["action_width"])
        if mode == "latent":
            if self.horizon < self.slots:
                raise ValueError(
                    f"service horizon {self.horizon} < requested slots {self.slots}"
                )
            if self.action_width != int(command.z_dim):
                raise ValueError(
                    f"latent head width {self.action_width} != contract z_dim "
                    f"{command.z_dim}"
                )
        else:
            window = int(command.window_steps or 9)
            if self.horizon < window + 1:
                raise ValueError(
                    f"chunk horizon {self.horizon} < encoder window {window + 1}"
                )
        self._sequence = 0
        self.reset()

    # -- service I/O ----------------------------------------------------
    def _read_json(self, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        assert self._service.stdout is not None
        while time.monotonic() < deadline:
            line = self._service.stdout.readline()
            if not line:
                raise RuntimeError("gr00t service exited")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise TimeoutError("gr00t service produced no JSON before the timeout")

    def _goal_at(self, tick: int) -> str | None:
        goal = None
        for start_tick, name in self.goal_schedule:
            if tick >= start_tick:
                goal = name
        return goal

    def _request(self, history_flat: np.ndarray, tick: int = 0) -> np.ndarray:
        request: dict = {"state": [float(v) for v in history_flat]}
        goal = self._goal_at(tick)
        if goal is not None:
            request["goal"] = goal
        if self.rtc and self._prev_prediction is not None:
            consumed = self.hold_steps if self.mode == "chunk" else 1
            request["prev_chunk"] = [
                float(v) for v in self._prev_prediction.reshape(-1)
            ]
            request["rtc_overlap_steps"] = max(self.horizon - consumed, 0)
            request["freeze_steps"] = self.rtc_freeze_steps
            request["rtc_ramp_rate"] = self.rtc_ramp_rate
        assert self._service.stdin is not None
        self._service.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        reply = self._read_json(timeout_s=120.0)
        if "head_ms" in reply:
            self.head_ms.append(float(reply["head_ms"]))
        prediction = np.asarray(reply["chunk"], dtype=np.float32).reshape(
            self.horizon, self.action_width
        )
        self._prev_prediction = prediction
        return prediction

    def close(self) -> None:
        if self._service.stdin is not None:
            try:
                self._service.stdin.write(json.dumps({"stop": True}) + "\n")
                self._service.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        try:
            self._service.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self._service.kill()

    # -- causal history --------------------------------------------------
    def _frame(self, state: RobotState) -> np.ndarray:
        frame = np.empty(STATE_WIDTH, dtype=np.float32)
        frame[0:29] = np.asarray(state.joint_pos) - self._default_joint_pos
        frame[29:58] = np.asarray(state.joint_vel)
        frame[58:61] = np.asarray(state.base_ang_vel)
        frame[61:64] = np.asarray(state.projected_gravity)
        frame[64:93] = self.tracker.last_action
        return frame

    def _push_history(self, state: RobotState) -> None:
        frame = self._frame(state)
        if not self._history_filled:
            self._history[:] = frame
            self._history_filled = True
        else:
            self._history[:-1] = self._history[1:]
            self._history[-1] = frame

    # -- publisher protocol ----------------------------------------------
    def reset(self) -> None:
        self._history = np.zeros((STATE_FRAMES, STATE_WIDTH), dtype=np.float32)
        self._history_filled = False
        self._prev_prediction: np.ndarray | None = None
        self._z: np.ndarray | None = None
        self._slot_queue: list[np.ndarray] = []
        self._steps_into_hold = 0
        self._publications = 0

    @property
    def exhausted(self) -> bool:
        return (
            self.max_publications is not None
            and self._publications >= self.max_publications
            and self._steps_into_hold == 0
        )

    def _encode_chunk(self, prediction: np.ndarray) -> np.ndarray:
        assert self.encoder is not None
        window = int(self.command.window_steps or 9) + 1
        flat = np.ascontiguousarray(prediction[:window].reshape(-1))
        z = np.asarray(self.encoder.infer(flat), dtype=np.float32)
        if z.shape != (int(self.command.z_dim),):
            raise ValueError(
                f"encoder produced {z.shape}, expected ({self.command.z_dim},)"
            )
        return z

    def tick(self, tick: int, stamp: float, state: RobotState | None = None) -> None:
        if state is None:
            raise ValueError("the gr00t publisher needs the robot state")
        self._push_history(state)
        if self._z is None or self._steps_into_hold >= self.hold_steps:
            if self._z is not None:
                self._publications += 1
            self._steps_into_hold = 0
            if self._slot_queue:
                self._z = self._slot_queue.pop(0)
            else:
                history_flat = np.ascontiguousarray(self._history.reshape(-1))
                prediction = self._request(history_flat, tick)
                if self.mode == "chunk":
                    self._z = self._encode_chunk(prediction)
                elif self.rtc:
                    self._z = prediction[0].copy()
                else:
                    self._z = prediction[0].copy()
                    self._slot_queue = [
                        prediction[slot].copy() for slot in range(1, self.slots)
                    ]
        if self.command.phase_mode == "sin_cos":
            phase = self._steps_into_hold / float(self.hold_steps)
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
                    "mode": self.mode,
                    "rtc": self.rtc,
                    "steps_into_hold": self._steps_into_hold,
                },
            )
        )
        self._steps_into_hold += 1


__all__ = ["Gr00tServicePublisher"]
