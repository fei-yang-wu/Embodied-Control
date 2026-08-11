"""Runtime data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class RobotState:
    stamp: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    projected_gravity: np.ndarray
    base_ang_vel: np.ndarray
    anchor_pos_w: np.ndarray | None = None
    anchor_quat_w: np.ndarray | None = None

    def validate(self, width: int = 29) -> None:
        expected = {
            "joint_pos": (width,),
            "joint_vel": (width,),
            "projected_gravity": (3,),
            "base_ang_vel": (3,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"RobotState.{name} must have shape {shape}, got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"RobotState.{name} contains non-finite values")
        for name in ("anchor_pos_w", "anchor_quat_w"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(np.asarray(value)).all():
                raise ValueError(f"RobotState.{name} contains non-finite values")


@dataclass(frozen=True)
class JointCommand:
    q_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray | None = None


@dataclass(frozen=True)
class CommandSample:
    vector: np.ndarray
    age_ticks: int
    renewed: bool
    terms: Mapping[str, np.ndarray] = field(default_factory=dict)
    available: bool = True
    done: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandPacket:
    """One VLA-to-tracker command published to a command buffer."""

    interface: Literal["explicit", "latent", "chunk"]
    values: np.ndarray
    sequence: int
    stamp: float
    terms: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def copy(self) -> "CommandPacket":
        return CommandPacket(
            interface=self.interface,
            values=np.array(self.values, dtype=np.float32, copy=True),
            sequence=int(self.sequence),
            stamp=float(self.stamp),
            terms={
                name: np.array(value, dtype=np.float32, copy=True)
                for name, value in self.terms.items()
            },
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class CommandSnapshot:
    packet: CommandPacket | None
    age_seconds: float
    renewed: bool


class LoopClock(Protocol):
    def now(self) -> float: ...
    def wait_for_tick(self, tick: int, control_hz: int) -> None: ...


class RobotBackend(Protocol):
    def reset(self, seed: int = 0) -> None: ...
    def read_state(self) -> RobotState: ...
    def write_command(self, cmd: JointCommand) -> None: ...
    def clock(self) -> LoopClock: ...
