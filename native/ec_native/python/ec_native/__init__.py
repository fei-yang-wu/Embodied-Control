from __future__ import annotations

import ctypes
import glob
import os
import sysconfig


def preload_gpu_runtime() -> list[str]:
    """Load the CUDA / cuDNN / TensorRT runtime libraries the ONNX Runtime GPU
    providers dlopen by soname, from the pip wheels in this environment
    (`nvidia-*-cu12`, `tensorrt-cu12-libs`), before any session asks for them.
    The onnxruntime-gpu wheel does the same. Returns the paths loaded; empty
    when no wheel is installed, which leaves the CPU provider untouched."""
    site = sysconfig.get_paths()["purelib"]
    patterns = [
        os.path.join(site, "nvidia", "*", "lib", "lib*.so*"),
        os.path.join(site, "tensorrt_libs", "lib*.so*"),
    ]
    loaded: list[str] = []
    # Dependencies first: cudart / cublas before cudnn / nvinfer.
    order = ("cudart", "nvrtc", "cublasLt", "cublas", "cufft", "curand", "cudnn",
             "nvinfer", "nvonnxparser")
    paths = sorted({path for pattern in patterns for path in glob.glob(pattern)})
    paths.sort(key=lambda p: next((i for i, k in enumerate(order) if k in os.path.basename(p)), len(order)))
    for path in paths:
        if "_static" in path or "builder_resource" in path:
            continue
        try:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            loaded.append(path)
        except OSError:
            continue
    return loaded


if os.environ.get("EC_PRELOAD_GPU_RUNTIME", "1") != "0":
    GPU_RUNTIME_LIBRARIES = preload_gpu_runtime()
else:  # pragma: no cover - opt-out
    GPU_RUNTIME_LIBRARIES = []

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
    "GPU_RUNTIME_LIBRARIES",
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
    "preload_gpu_runtime",
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
