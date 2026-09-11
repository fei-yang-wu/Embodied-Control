"""Policy-independent MuJoCo plant serving the G1 DDS protocol."""

from __future__ import annotations

import numpy as np

from embodied_control.robot.plant import PlantConfig


class NativeDdsPlant:
    """MuJoCo physics serving the exact G1 hardware DDS protocol.

    The Digit-style unified plant interface: the controller runs the one
    hardware code path (``NativeUnitreeLoop``) against this plant on
    interface ``lo`` and against the robot on its NIC; nothing else changes.
    Plant configuration owns joint order and physical parameters. DDS motor
    IDs map that order onto the wire independently of any controller.
    """

    def __init__(
        self,
        robot: PlantConfig,
        model_path: str,
        network_interface: str = "lo",
        *,
        timestep: float = 0.002,
        mode_machine: int = 5,
        physics_cpu: int = -1,
        physics_fifo_priority: int = 0,
        lock_memory: bool = False,
        require_realtime: bool = False,
        sensor_noise: dict[str, float] | None = None,
        noise_seed: int = 0,
        state_log_capacity: int = 0,
        dds_domain: int = 0,
        freeze_until_command: bool = False,
        vendor: bool = False,
        vendor_name: str = "ai",
        hoist: bool = False,
        hoist_clearance: float = 0.10,
        odometry_topic: str = "",
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "NativeDdsPlant needs the Unitree-enabled native build"
            ) from exc
        if not ec_native.WITH_UNITREE:
            raise RuntimeError(
                "ec_native was built without Unitree SDK2; set "
                "EC_UNITREE_SDK_ROOT and run `pixi run -e native build-native`"
            )
        self.robot = robot
        self.joint_names = robot.joint_names
        self._joint_to_sdk = [joint.motor_id for joint in robot.joints]
        sdk_joint_names = [""] * len(robot.joints)
        for joint in robot.joints:
            sdk_joint_names[joint.motor_id] = joint.name

        def values(field):
            return self._to_sdk([getattr(joint, field) for joint in robot.joints])

        self._plant = ec_native.MujocoDdsPlant(
            str(model_path),
            str(network_interface),
            sdk_joint_names,
            values("nominal_position"),
            values("armature"),
            values("effort_limit"),
            values("vendor_stiffness"),
            values("vendor_damping"),
            float(timestep),
            int(mode_machine),
            int(physics_cpu),
            int(physics_fifo_priority),
            bool(lock_memory),
            bool(require_realtime),
            # A real G1 does not serve clean state; the plant puts the sensor
            # noise on the wire so the controller sees it exactly as it will
            # on hardware.
            float((sensor_noise or {}).get("joint_pos", 0.0)),
            float((sensor_noise or {}).get("joint_vel", 0.0)),
            float((sensor_noise or {}).get("base_ang_vel", 0.0)),
            float((sensor_noise or {}).get("imu_tilt_rad", 0.0)),
            int(noise_seed),
            int(state_log_capacity),
            int(dds_domain),
            bool(freeze_until_command),
            bool(vendor),
            str(vendor_name),
            bool(hoist),
            float(hoist_clearance),
        )
        if odometry_topic:
            self._plant.publish_odometry(str(odometry_topic))

    def hoist(self) -> None:
        self._plant.hoist()

    def lower(self) -> None:
        self._plant.lower()

    def slack(self) -> None:
        self._plant.slack()

    def _to_sdk(self, values) -> np.ndarray:
        configured = np.asarray(values, dtype=np.float32)
        sdk = np.empty_like(configured)
        sdk[self._joint_to_sdk] = configured
        return sdk

    def set_initial_pose(self, pose) -> None:
        """Plant-config-order start pose [pos 3 | quat XYZW 4 | joints 29]."""
        values = np.asarray(pose, dtype=np.float32)
        if values.size == 0:
            self._plant.set_initial_pose(np.zeros(0, np.float32))
            return
        if values.shape != (36,):
            raise ValueError("initial pose must have 36 values")
        converted = np.concatenate([values[:7], self._to_sdk(values[7:])])
        self._plant.set_initial_pose(converted)

    def latest_state(self) -> np.ndarray:
        """Newest true state as [pos 3 | quat XYZW 4 | joints 29], plant-config order."""
        row = np.asarray(self._plant.latest_state(), dtype=np.float32)
        return np.concatenate([row[:7], row[7:][self._joint_to_sdk]])

    def state_log(self) -> np.ndarray:
        """True simulator state, rows x [pos 3 | quat XYZW 4 | joints 29].

        The hardware wire protocol carries no root pose, so scoring MPJPE on
        this tier has to read the plant's own state, not the controller's.
        The joint order is recorded alongside saved state arrays; consumers
        must map by name when comparing with a reference or controller.
        """
        rows = np.asarray(self._plant.state_log(), dtype=np.float32)
        if rows.size == 0:
            return rows.reshape(0, 36)
        configured = np.empty_like(rows)
        configured[:, :7] = rows[:, :7]
        configured[:, 7:] = rows[:, 7:][:, self._joint_to_sdk]
        return configured

    def hoist_log(self) -> np.ndarray:
        """Rows of spreader endpoints (6), release gain, and strap tensions (2)."""
        return np.asarray(self._plant.hoist_log(), dtype=np.float32).reshape(-1, 9)

    def hoist_attachment_points(self) -> np.ndarray:
        """Shoulder buckle locations in torso_link coordinates."""
        return np.asarray(self._plant.hoist_attachment_points(), dtype=np.float64)

    def reset(self) -> None:
        self._plant.reset()

    def start(self) -> None:
        self._plant.start()

    def stop(self) -> None:
        self._plant.stop()

    def wait_for_stop(self) -> None:
        self._plant.wait_for_stop()

    @property
    def running(self) -> bool:
        return bool(self._plant.running)

    def stats(self) -> dict[str, float | int | bool]:
        return dict(self._plant.stats())
