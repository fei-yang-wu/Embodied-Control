"""Safety envelope: freshness watchdogs and the damp command."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from embodied_control.lowlevel.contracts import CommandSample, JointCommand, RobotState
from embodied_control.lowlevel.job import SafetySpec


class SafetyFault(RuntimeError):
    def __init__(self, cause: str, detail: str = ""):
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause = cause
        self.detail = detail


@dataclass
class SafetyMonitor:
    """Per-episode watchdog state. One instance per episode."""

    spec: SafetySpec
    control_hz: int
    stale_command_ticks: int = 0

    def check_state(self, state: RobotState, now: float) -> None:
        state.validate(state.joint_pos.shape[0])
        age_ms = max(0.0, (now - state.stamp) * 1000.0)
        if age_ms > self.spec.state_absent_ms:
            raise SafetyFault("state_absent", f"robot state is {age_ms:.0f} ms old")

    def state_late(self, state: RobotState, now: float) -> bool:
        return (now - state.stamp) * 1000.0 > self.spec.state_late_ms

    def check_command(self, command: CommandSample) -> None:
        if not command.available:
            self.stale_command_ticks += 1
        elif command.renewed:
            self.stale_command_ticks = 0
        if self.stale_command_ticks > self.spec.command_absent_ticks:
            raise SafetyFault(
                "command_absent",
                f"no command for {self.stale_command_ticks} control ticks",
            )

    def check_action(self, action: np.ndarray, previous: np.ndarray) -> None:
        if not np.isfinite(action).all():
            raise SafetyFault("nan_action", "policy action contains non-finite values")
        if self.spec.max_action_delta is not None:
            delta = float(np.abs(action - previous).max())
            if delta > self.spec.max_action_delta:
                raise SafetyFault("action_delta", f"max action delta {delta:.3f}")


def damp_command(width: int, kd: float) -> JointCommand:
    return JointCommand(
        q_target=np.zeros(width, dtype=np.float32),
        kp=np.zeros(width, dtype=np.float32),
        kd=np.full(width, float(kd), dtype=np.float32),
    )


__all__ = ["SafetyFault", "SafetyMonitor", "damp_command"]
