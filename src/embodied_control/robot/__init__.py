"""Robot runtimes: the high-level command axis (lifecycle and mode).

Per this repo's standing convention, ``open_robot`` is an if/else and not a
plugin registry — the registry arrives with the third robot, if ever.
"""

from __future__ import annotations

from embodied_control.robot.base import (
    Capability,
    GestureControl,
    PostureControl,
    RobotCommandError,
    RobotError,
    RobotHealth,
    RobotInfo,
    RobotMode,
    RobotRuntime,
    RobotTransitionError,
    RobotWriteGateError,
    VelocityControl,
)
from embodied_control.robot.fake import FakeRobotRuntime

__all__ = [
    "Capability",
    "FakeRobotRuntime",
    "GestureControl",
    "PostureControl",
    "RobotCommandError",
    "RobotError",
    "RobotHealth",
    "RobotInfo",
    "RobotMode",
    "RobotRuntime",
    "RobotTransitionError",
    "RobotWriteGateError",
    "VelocityControl",
    "open_robot",
]


def open_robot(
    robot: str = "g1",
    *,
    network_interface: str = "",
    writes_enabled: bool = False,
    dds_domain: int = 0,
    timeout_seconds: float = 5.0,
) -> RobotRuntime:
    if robot == "fake":
        return FakeRobotRuntime(writes_enabled=writes_enabled)
    if robot == "g1":
        if not network_interface:
            raise ValueError("the G1 runtime needs --network")
        from embodied_control.robot.g1 import G1Runtime

        return G1Runtime(
            network_interface,
            writes_enabled=writes_enabled,
            dds_domain=dds_domain,
            timeout_seconds=timeout_seconds,
        )
    raise ValueError(f"unknown robot '{robot}'")
