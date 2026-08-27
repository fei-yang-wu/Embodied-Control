from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from embodied_control.lowlevel.bundle import ActionContract, PolicyBundle
from embodied_control.lowlevel.unitree_probe import run_probe

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def sdk_to_isaac(action: ActionContract, sdk_values: np.ndarray) -> np.ndarray:
    values = np.asarray(sdk_values, dtype=np.float64)
    if values.shape != (action.width,):
        raise ValueError(
            f"SDK values must have shape ({action.width},), got {values.shape}"
        )
    if len(action.isaac_to_sdk) != action.width:
        raise ValueError("bundle has no complete isaac_to_sdk mapping")
    return values[np.asarray(action.isaac_to_sdk, dtype=np.int64)]


def isaac_to_sdk(action: ActionContract, isaac_values: np.ndarray) -> np.ndarray:
    values = np.asarray(isaac_values, dtype=np.float64)
    if values.shape != (action.width,):
        raise ValueError(
            f"Isaac values must have shape ({action.width},), got {values.shape}"
        )
    result = np.empty_like(values)
    result[np.asarray(action.isaac_to_sdk, dtype=np.int64)] = values
    return result


def _normalized_wxyz(values: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("IMU quaternion must contain four finite WXYZ values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        raise ValueError("IMU quaternion has near-zero norm")
    return quaternion / norm


def render_pose_comparison(
    action: ActionContract,
    model_path: str | Path,
    sdk_joint_position: np.ndarray,
    imu_quaternion_wxyz: np.ndarray,
    output_path: str | Path,
    *,
    root_height: float = 0.76,
    width: int = 640,
    height: int = 480,
) -> dict[str, Any]:
    import imageio.v2 as imageio
    import mujoco

    from embodied_control.lowlevel.envs.mujoco import load_scene_model

    if not np.isfinite(root_height) or root_height <= 0:
        raise ValueError("root_height must be a positive finite number")
    isaac_joint_position = sdk_to_isaac(action, sdk_joint_position)
    reconstructed_sdk = isaac_to_sdk(action, isaac_joint_position)
    roundtrip_error = float(
        np.max(np.abs(reconstructed_sdk - np.asarray(sdk_joint_position)))
    )

    model = load_scene_model(model_path)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[:3] = [0.0, 0.0, root_height]
    data.qpos[3:7] = _normalized_wxyz(imu_quaternion_wxyz)

    mapping = []
    mismatches = []
    qpos_address = np.empty(action.width, dtype=np.int64)
    for isaac_index, isaac_name in enumerate(action.isaac_joint_names):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, isaac_name
        )
        if joint_id < 0:
            raise ValueError(f"MJCF has no joint {isaac_name!r}")
        address = int(model.jnt_qposadr[joint_id])
        qpos_address[isaac_index] = address
        data.qpos[address] = isaac_joint_position[isaac_index]
        sdk_index = int(action.isaac_to_sdk[isaac_index])
        sdk_name = (
            action.sdk_joint_names[sdk_index]
            if action.sdk_joint_names
            else None
        )
        if sdk_name is not None and sdk_name != isaac_name:
            mismatches.append(
                {
                    "isaac_index": isaac_index,
                    "sdk_index": sdk_index,
                    "isaac_name": isaac_name,
                    "sdk_name": sdk_name,
                }
            )
        mapping.append(
            {
                "isaac_index": isaac_index,
                "sdk_index": sdk_index,
                "isaac_name": isaac_name,
                "sdk_name": sdk_name,
                "q_rad": float(isaac_joint_position[isaac_index]),
            }
        )

    mujoco.mj_forward(model, data)
    readback = np.asarray(data.qpos[qpos_address], dtype=np.float64)
    readback_error = float(np.max(np.abs(readback - isaac_joint_position)))

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=int(height), width=int(width))
    try:
        renderer.update_scene(data, camera="track")
        imageio.imwrite(output, renderer.render())
    finally:
        renderer.close()

    checks = {
        "roundtrip": roundtrip_error <= 1e-12,
        "joint_names": not mismatches,
        "mujoco_readback": readback_error <= 1e-12,
    }
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "image": str(output),
        "root_position": [0.0, 0.0, float(root_height)],
        "root_position_source": "fixed_for_visualization",
        "imu_quaternion_wxyz": _normalized_wxyz(
            imu_quaternion_wxyz
        ).tolist(),
        "roundtrip_max_error": roundtrip_error,
        "mujoco_readback_max_error": readback_error,
        "mapping_name_mismatches": mismatches,
        "sdk_joint_position": np.asarray(
            sdk_joint_position, dtype=np.float64
        ).tolist(),
        "isaac_joint_position": isaac_joint_position.tolist(),
        "joint_mapping": mapping,
    }


def compare_live_pose(
    bundle_path: str | Path,
    model_path: str | Path,
    network: str,
    output_dir: str | Path,
    *,
    samples: int = 500,
    timeout: float = 5.0,
    root_height: float = 0.76,
    dds_domain: int = 0,
) -> dict[str, Any]:
    bundle = PolicyBundle.load(bundle_path)
    connection = run_probe(
        network,
        samples=samples,
        timeout=timeout,
        dds_domain=dds_domain,
    )
    if connection["status"] != "pass":
        raise RuntimeError("G1 connection check failed; refusing to render stale state")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    pose = render_pose_comparison(
        bundle.manifest.action,
        model_path,
        np.asarray(connection["state"]["joint_position"]),
        np.asarray(connection["state"]["quaternion"]),
        output / "mujoco_pose.png",
        root_height=root_height,
    )
    report = {
        "status": pose["status"],
        "read_only": True,
        "bundle": str(bundle.root),
        "model": str(Path(model_path).resolve()),
        "connection": {
            key: connection[key]
            for key in (
                "status",
                "network",
                "topic",
                "samples",
                "rate_hz",
                "max_gap_ms",
                "crc_errors",
                "motor_error_joints",
                "first_tick",
                "last_tick",
            )
        },
        "pose": pose,
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["report"] = str(report_path)
    return report
