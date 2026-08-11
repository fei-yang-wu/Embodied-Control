"""Deterministic kinematic backend for tests and plumbing runs."""

from __future__ import annotations

import numpy as np

from embodied_control.lowlevel.bundle import ActionContract
from embodied_control.lowlevel.contracts import JointCommand, RobotState


class _FreeRunningClock:
    def __init__(self, backend: "FakeBackend"):
        self._backend = backend

    def now(self) -> float:
        return self._backend.now

    def wait_for_tick(self, tick: int, control_hz: int) -> None:
        return


class FakeBackend:
    """First-order joint lag toward `q_target`; no dynamics, no contact.

    `lag_alpha=1.0` teleports to the target each control step. The state
    stamp advances by the control period on every read, so freshness
    watchdogs see a live robot.
    """

    def __init__(self, action: ActionContract, *, control_hz: int = 50, lag_alpha: float = 1.0):
        self._action = action
        self._dt = 1.0 / float(control_hz)
        self._alpha = float(lag_alpha)
        self._clock = _FreeRunningClock(self)
        self.reset()

    def reset(self, seed: int = 0) -> None:
        del seed
        self._time = 0.0
        self._joint_pos = np.asarray(self._action.default_joint_pos, dtype=np.float32).copy()
        self._joint_vel = np.zeros(self._action.width, dtype=np.float32)

    def read_state(self) -> RobotState:
        return RobotState(
            stamp=self._time,
            joint_pos=self._joint_pos.copy(),
            joint_vel=self._joint_vel.copy(),
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            anchor_pos_w=np.zeros(3, dtype=np.float32),
            anchor_quat_w=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def write_command(self, cmd: JointCommand) -> None:
        target = np.asarray(cmd.q_target, dtype=np.float32)
        if target.shape != self._joint_pos.shape:
            raise ValueError(
                f"q_target must have shape {self._joint_pos.shape}, got {target.shape}"
            )
        new_pos = self._joint_pos + self._alpha * (target - self._joint_pos)
        self._joint_vel = (new_pos - self._joint_pos) / self._dt
        self._joint_pos = new_pos
        self._time += self._dt

    def clock(self) -> _FreeRunningClock:
        return self._clock

    @property
    def now(self) -> float:
        return self._time


__all__ = ["FakeBackend"]
