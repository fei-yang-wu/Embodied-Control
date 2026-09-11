from __future__ import annotations

from . import _ec_native
from ._ec_native import (
    MAX_VALUES,
    LegOdometry,
    NativeFakeRuntime,
    NativeMujocoRuntime,
    NativeTrackerCore,
    OnnxEngine,
    ShmCommandSlot,
    WITH_UNITREE,
    __version__,
    align_heading_to_reference,
    monotonic_now,
    pack_joint_qpos_qvel_anchor_ori_window,
    projected_gravity_from_xyzw,
    reexpress_root_qpos_window,
)

__all__ = [
    "MAX_VALUES",
    "LegOdometry",
    "NativeFakeRuntime",
    "NativeMujocoRuntime",
    "NativeTrackerCore",
    "OnnxEngine",
    "ShmCommandSlot",
    "WITH_UNITREE",
    "__version__",
    "align_heading_to_reference",
    "monotonic_now",
    "pack_joint_qpos_qvel_anchor_ori_window",
    "projected_gravity_from_xyzw",
    "reexpress_root_qpos_window",
]

if WITH_UNITREE:
    G1LocoClient = _ec_native.G1LocoClient
    MujocoDdsPlant = _ec_native.MujocoDdsPlant
    PlantClient = _ec_native.PlantClient
    NativeUnitreeRuntime = _ec_native.NativeUnitreeRuntime
    UnitreeStateProbe = _ec_native.UnitreeStateProbe
    OdometryProbe = _ec_native.OdometryProbe
    __all__.extend(
        [
            "G1LocoClient",
            "MujocoDdsPlant",
            "PlantClient",
            "NativeUnitreeRuntime",
            "UnitreeStateProbe",
        ]
    )
