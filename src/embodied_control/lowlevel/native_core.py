"""Thin Python construction API for the native allocation-free tracker core."""

from __future__ import annotations

import os
import time
import uuid

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.contracts import RobotState

_PROPRIO_TERMS = {
    "projected_gravity",
    "base_ang_vel",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
}


def _direct_command_tag(bundle: PolicyBundle) -> int:
    if bundle.manifest.interface == "latent":
        return 1
    if bundle.manifest.interface == "explicit":
        return 0
    raise ValueError(
        "native runtime supports latent or explicit tracker bundles, got "
        f"{bundle.manifest.interface!r}"
    )


def _require_supported_encoder_cadence(bundle: PolicyBundle) -> None:
    del bundle


def _oracle_enabled(bundle: PolicyBundle, command_source: str) -> bool:
    if command_source not in {"vla", "oracle"}:
        raise ValueError("command_source must be 'vla' or 'oracle'")
    oracle = command_source == "oracle"
    command = bundle.manifest.command
    expected_anchor = {
        "root_qpos": "robot",
        "joint_qpos_qvel_anchor_ori": "robot_heading",
    }.get(command.encoder_state_interface)
    if oracle and (
        expected_anchor is None
        or command.macro_anchor_mode != expected_anchor
        or command.macro_frame_stride is None
        or bundle.manifest.models.get("encoder_onnx") is None
    ):
        raise ValueError(
            "oracle command_source needs a supported robot-anchored ONNX encoder"
        )
    return oracle


class NativeTracker:
    """Configure C++ once; only fixed float arrays cross during diagnostic steps."""

    def __init__(self, bundle: PolicyBundle, *, intra_op_threads: int = 4):
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - depends on optional build
            raise ImportError(
                "NativeTracker needs the native Pixi environment: "
                "pixi run -e native ..."
            ) from exc

        artifact = bundle.manifest.models.get("policy_onnx")
        if artifact is None:
            raise ValueError("native tracker requires a policy_onnx model artifact")
        terms = [
            (
                term.name,
                term.width,
                term.history_length,
                term.history_stride,
                term.history_order,
                term.reset_fill,
            )
            for term in bundle.manifest.obs.terms
        ]
        command_width = sum(
            term.width
            for term in bundle.manifest.obs.terms
            if term.name not in _PROPRIO_TERMS
        )
        action = bundle.manifest.action
        lower = action.joint_limits_lower or []
        upper = action.joint_limits_upper or []
        command = bundle.manifest.command
        fsq_half = command.fsq_half_levels or []
        fsq_z_dim = int(command.z_dim or 0) if command.quantizer == "fsq" else 0
        self.bundle = bundle
        self._core = ec_native.NativeTrackerCore(
            str(bundle.policy_onnx_path),
            artifact.input_name,
            artifact.output_name,
            terms,
            command_width,
            np.asarray(action.default_joint_pos, dtype=np.float32),
            np.asarray(action.action_scale, dtype=np.float32),
            np.asarray(lower, dtype=np.float32),
            np.asarray(upper, dtype=np.float32),
            np.asarray(fsq_half, dtype=np.float32),
            fsq_z_dim,
            float(action.raw_action_clip or 0.0),
            int(intra_op_threads),
        )

    def reset(self) -> None:
        self._core.reset()

    def warmup(self, iterations: int = 8) -> None:
        self._core.warmup(iterations)

    def step_once(
        self, state: RobotState, command: np.ndarray
    ) -> dict[str, np.ndarray]:
        return self._core.step_once(
            np.asarray(state.joint_pos, dtype=np.float32),
            np.asarray(state.joint_vel, dtype=np.float32),
            np.asarray(state.projected_gravity, dtype=np.float32),
            np.asarray(state.base_ang_vel, dtype=np.float32),
            np.asarray(command, dtype=np.float32),
        )

    @property
    def observation_width(self) -> int:
        return int(self._core.observation_width)

    @property
    def command_width(self) -> int:
        return int(self._core.command_width)


class NativeFakeLoop:
    """Construct the fully native fake-backend control and planner loop."""

    def __init__(
        self,
        bundle: PolicyBundle,
        *,
        response_slot: str,
        request_slot: str = "",
        create_slots: bool = True,
        control_hz: int | None = None,
        lag_alpha: float = 1.0,
        command_absent_ticks: int = 100,
        command_stale_ms: float = 500.0,
        hold_steps: int | None = None,
        lead_ticks: int = 4,
        cpu: int = -1,
        fifo_priority: int = 0,
        lock_memory: bool = False,
        require_realtime: bool = False,
        policy_threads: int = 4,
        command_source: str = "vla",
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "NativeFakeLoop needs the native Pixi environment"
            ) from exc

        self.tracker = NativeTracker(bundle, intra_op_threads=policy_threads)
        _require_supported_encoder_cadence(bundle)
        command = bundle.manifest.command
        oracle_reference = _oracle_enabled(bundle, command_source)
        encoder = bundle.manifest.models.get("encoder_onnx")
        encoder_path = ""
        encoder_input_name = ""
        encoder_output_name = ""
        encoder_input_width = 0
        encoder_output_width = 0
        if encoder is not None:
            encoder_path = str(bundle.encoder_onnx_path)
            encoder_input_name = encoder.input_name
            encoder_output_name = encoder.output_name
            encoder_input_width = encoder.input_shape[1]
            encoder_output_width = encoder.output_shape[1]
        self._runtime = ec_native.NativeFakeRuntime(
            self.tracker._core,
            response_slot,
            request_slot,
            create_slots,
            int(control_hz or bundle.manifest.rates.control_hz),
            float(lag_alpha),
            int(command_absent_ticks),
            float(command_stale_ms),
            int(hold_steps or command.hold_steps),
            int(lead_ticks),
            int(command.state_dim or 38),
            int((command.window_steps or 9) + 1),
            int(command.z_dim or self.tracker.command_width),
            command.phase_mode == "sin_cos",
            _direct_command_tag(bundle),
            oracle_reference,
            str(command.encoder_state_interface or "root_qpos"),
            int(command.macro_frame_stride or 1),
            str(command.encoder_trigger),
            encoder_path,
            encoder_input_name,
            encoder_output_name,
            encoder_input_width,
            encoder_output_width,
            int(cpu),
            int(fifo_priority),
            bool(lock_memory),
            bool(require_realtime),
        )

    def start(self, max_ticks: int, *, paced: bool = True) -> None:
        self._runtime.start(int(max_ticks), bool(paced))

    def stop(self) -> None:
        self._runtime.stop()

    def wait(self) -> None:
        self._runtime.wait()

    @property
    def running(self) -> bool:
        return bool(self._runtime.running)

    def stats(self) -> dict[str, int | bool]:
        return dict(self._runtime.stats())

    def state(self) -> dict[str, np.ndarray]:
        return self._runtime.state()

    def set_initial_pose(self, pose: np.ndarray) -> None:
        """Frame-0 start: [root pos 3 | root quat XYZW 4 | joints 29]."""
        self._runtime.set_initial_pose(np.ascontiguousarray(pose, dtype=np.float32))

    def reference_frames(self) -> np.ndarray:
        return np.asarray(self._runtime.reference_frames())

    def joint_position_log(self) -> np.ndarray:
        return np.asarray(self._runtime.joint_position_log())

    def anchor_pose_log(self) -> np.ndarray:
        return np.asarray(self._runtime.anchor_pose_log())

    def tick_durations_ns(self) -> np.ndarray:
        return np.asarray(self._runtime.tick_durations_ns(), dtype=np.uint64)

    def base_heights(self) -> np.ndarray:
        return np.asarray(self._runtime.base_heights(), dtype=np.float32)

    def reference_joint_mae(self) -> np.ndarray:
        return np.asarray(self._runtime.reference_joint_mae(), dtype=np.float32)

    @property
    def backend_time(self) -> float:
        return float(self._runtime.backend_time)

    @property
    def base_height(self) -> float:
        return float(self._runtime.base_height)

    @property
    def min_base_height(self) -> float:
        return float(self._runtime.min_base_height)


class NativeMujocoLoop(NativeFakeLoop):
    """Run independent MuJoCo physics and policy schedules in C++."""

    def __init__(
        self,
        bundle: PolicyBundle,
        model_path: str,
        *,
        response_slot: str,
        request_slot: str = "",
        create_slots: bool = True,
        control_hz: int | None = None,
        command_absent_ticks: int = 100,
        command_stale_ms: float = 500.0,
        hold_steps: int | None = None,
        lead_ticks: int = 4,
        cpu: int = -1,
        fifo_priority: int = 0,
        lock_memory: bool = False,
        require_realtime: bool = False,
        physics_cpu: int = -1,
        physics_fifo_priority: int = 0,
        physics_lock_memory: bool = False,
        physics_require_realtime: bool = False,
        policy_threads: int = 4,
        command_source: str = "vla",
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "NativeMujocoLoop needs the native Pixi environment"
            ) from exc

        action = bundle.manifest.action
        if not action.armature or not action.effort_limit:
            raise ValueError(
                "native MuJoCo needs armature and effort_limit in the bundle"
            )
        self.tracker = NativeTracker(bundle, intra_op_threads=policy_threads)
        _require_supported_encoder_cadence(bundle)
        command = bundle.manifest.command
        oracle_reference = _oracle_enabled(bundle, command_source)
        encoder = bundle.manifest.models.get("encoder_onnx")
        encoder_path = ""
        encoder_input_name = ""
        encoder_output_name = ""
        encoder_input_width = 0
        encoder_output_width = 0
        if encoder is not None:
            encoder_path = str(bundle.encoder_onnx_path)
            encoder_input_name = encoder.input_name
            encoder_output_name = encoder.output_name
            encoder_input_width = encoder.input_shape[1]
            encoder_output_width = encoder.output_shape[1]
        self.bundle = bundle
        self._runtime = ec_native.NativeMujocoRuntime(
            self.tracker._core,
            str(model_path),
            list(action.isaac_joint_names),
            np.asarray(action.default_joint_pos, dtype=np.float32),
            np.asarray(action.stiffness, dtype=np.float32),
            np.asarray(action.damping, dtype=np.float32),
            np.asarray(action.armature, dtype=np.float32),
            np.asarray(action.effort_limit, dtype=np.float32),
            float(bundle.manifest.rates.physics_dt),
            int(bundle.manifest.rates.decimation),
            response_slot,
            request_slot,
            create_slots,
            int(control_hz or bundle.manifest.rates.control_hz),
            int(command_absent_ticks),
            float(command_stale_ms),
            int(hold_steps or command.hold_steps),
            int(lead_ticks),
            int(command.state_dim or 38),
            int((command.window_steps or 9) + 1),
            int(command.z_dim or self.tracker.command_width),
            command.phase_mode == "sin_cos",
            _direct_command_tag(bundle),
            oracle_reference,
            str(command.encoder_state_interface or "root_qpos"),
            int(command.macro_frame_stride or 1),
            str(command.encoder_trigger),
            encoder_path,
            encoder_input_name,
            encoder_output_name,
            encoder_input_width,
            encoder_output_width,
            int(cpu),
            int(fifo_priority),
            bool(lock_memory),
            bool(require_realtime),
            int(physics_cpu),
            int(physics_fifo_priority),
            bool(physics_lock_memory),
            bool(physics_require_realtime),
        )


class NativeUnitreeLoop(NativeFakeLoop):
    """C++ G1 DDS runtime with explicit write and operator gates."""

    DISABLED = 0
    INITIALIZE = 1
    WAIT = 2
    CONTROL = 3
    DAMP = 4

    def __init__(
        self,
        bundle: PolicyBundle,
        network_interface: str,
        *,
        response_slot: str,
        request_slot: str = "",
        writes_enabled: bool = False,
        create_slots: bool = True,
        control_hz: int | None = None,
        command_absent_ticks: int = 100,
        command_stale_ms: float = 500.0,
        state_absent_ms: float = 500.0,
        hold_steps: int | None = None,
        lead_ticks: int = 4,
        control_cpu: int = 2,
        writer_cpu: int = 3,
        control_fifo_priority: int = 80,
        writer_fifo_priority: int = 90,
        lock_memory: bool = True,
        require_realtime: bool = True,
        policy_threads: int = 1,
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "NativeUnitreeLoop needs the Unitree-enabled native build"
            ) from exc
        if not ec_native.WITH_UNITREE:
            raise RuntimeError(
                "ec_native was built without Unitree SDK2; set "
                "EC_UNITREE_SDK_ROOT and run `pixi run -e native build-native`"
            )
        self.tracker = NativeTracker(bundle, intra_op_threads=policy_threads)
        _require_supported_encoder_cadence(bundle)
        action = bundle.manifest.action
        if not action.joint_limits_lower or not action.joint_limits_upper:
            raise ValueError(
                "native Unitree control requires joint limits in the bundle"
            )
        command = bundle.manifest.command
        encoder = bundle.manifest.models.get("encoder_onnx")
        encoder_path = ""
        encoder_input_name = ""
        encoder_output_name = ""
        encoder_input_width = 0
        encoder_output_width = 0
        if encoder is not None:
            encoder_path = str(bundle.encoder_onnx_path)
            encoder_input_name = encoder.input_name
            encoder_output_name = encoder.output_name
            encoder_input_width = encoder.input_shape[1]
            encoder_output_width = encoder.output_shape[1]
        config = {
            "control_hz": int(control_hz or bundle.manifest.rates.control_hz),
            "command_absent_ticks": int(command_absent_ticks),
            "command_stale_ms": float(command_stale_ms),
            "state_absent_ms": float(state_absent_ms),
            "hold_steps": int(hold_steps or command.hold_steps),
            "lead_ticks": int(lead_ticks),
            "root_qpos_width": int(command.state_dim or 38),
            "window_frames": int((command.window_steps or 9) + 1),
            "z_dim": int(command.z_dim or self.tracker.command_width),
            "sin_cos_phase": command.phase_mode == "sin_cos",
            "direct_tag": _direct_command_tag(bundle),
            "reference_encoder_layout": str(
                command.encoder_state_interface or "root_qpos"
            ),
            "encoder_frame_stride": int(command.macro_frame_stride or 1),
            "encoder_trigger": str(command.encoder_trigger),
            "encoder_path": encoder_path,
            "encoder_input_name": encoder_input_name,
            "encoder_output_name": encoder_output_name,
            "encoder_input_width": encoder_input_width,
            "encoder_output_width": encoder_output_width,
            "cpu": int(control_cpu),
            "fifo_priority": int(control_fifo_priority),
            "writer_cpu": int(writer_cpu),
            "writer_fifo_priority": int(writer_fifo_priority),
            "lock_memory": bool(lock_memory),
            "require_realtime": bool(require_realtime),
        }
        self.bundle = bundle
        self._runtime = ec_native.NativeUnitreeRuntime(
            self.tracker._core,
            str(network_interface),
            list(action.isaac_to_sdk),
            np.asarray(action.default_joint_pos, dtype=np.float32),
            np.asarray(action.stiffness, dtype=np.float32),
            np.asarray(action.damping, dtype=np.float32),
            np.asarray(action.joint_limits_lower, dtype=np.float32),
            np.asarray(action.joint_limits_upper, dtype=np.float32),
            bool(writes_enabled),
            response_slot,
            request_slot,
            bool(create_slots),
            config,
        )

    @property
    def state_ready(self) -> bool:
        return bool(self._runtime.state_ready)

    @property
    def unitree_mode(self) -> int:
        return int(self._runtime.unitree_mode)

    def wait_for_state(self, timeout_seconds: float) -> bool:
        return bool(self._runtime.wait_for_state(float(timeout_seconds)))

    def begin_initialization(self, duration_seconds: float = 3.0) -> None:
        self._runtime.begin_initialization(float(duration_seconds))

    def wait_for_mode(self, expected: int, timeout_seconds: float) -> bool:
        return bool(self._runtime.wait_for_mode(int(expected), float(timeout_seconds)))

    def arm_control(self) -> None:
        self._runtime.arm_control()

    def force_damp(self) -> None:
        self._runtime.force_damp()

    def writer_stats(self) -> dict[str, int | bool]:
        return dict(self._runtime.writer_stats())


class NativeDdsPlant:
    """MuJoCo physics serving the exact G1 hardware DDS protocol.

    The Digit-style unified plant interface: the controller runs the one
    hardware code path (``NativeUnitreeLoop``) against this plant on
    interface ``lo`` and against the robot on its NIC; nothing else changes.
    Per-joint data crosses the wire in SDK motor order, so this wrapper
    derives the SDK-ordered tables from the bundle's Isaac-order contract.
    """

    def __init__(
        self,
        bundle: PolicyBundle,
        model_path: str,
        network_interface: str = "lo",
        *,
        timestep: float = 0.002,
        mode_machine: int = 5,
        physics_cpu: int = -1,
        physics_fifo_priority: int = 0,
        lock_memory: bool = False,
        require_realtime: bool = False,
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
        action = bundle.manifest.action
        if not action.isaac_to_sdk:
            raise ValueError("the DDS plant requires isaac_to_sdk in the bundle")
        if not action.armature or not action.effort_limit:
            raise ValueError(
                "the DDS plant requires armature and effort_limit in the bundle"
            )
        self._isaac_to_sdk = [int(v) for v in action.isaac_to_sdk]
        count = len(self._isaac_to_sdk)
        sdk_joint_names = [""] * count
        for isaac, sdk in enumerate(self._isaac_to_sdk):
            sdk_joint_names[sdk] = action.isaac_joint_names[isaac]
        self.bundle = bundle
        self._plant = ec_native.MujocoDdsPlant(
            str(model_path),
            str(network_interface),
            sdk_joint_names,
            self._to_sdk(action.default_joint_pos),
            self._to_sdk(action.armature),
            self._to_sdk(action.effort_limit),
            self._to_sdk(action.stiffness),
            self._to_sdk(action.damping),
            float(timestep),
            int(mode_machine),
            int(physics_cpu),
            int(physics_fifo_priority),
            bool(lock_memory),
            bool(require_realtime),
        )

    def _to_sdk(self, values) -> np.ndarray:
        isaac = np.asarray(values, dtype=np.float32)
        sdk = np.empty_like(isaac)
        sdk[self._isaac_to_sdk] = isaac
        return sdk

    def set_initial_pose(self, pose) -> None:
        """Isaac-order start pose [pos 3 | quat XYZW 4 | joints 29]."""
        values = np.asarray(pose, dtype=np.float32)
        if values.size == 0:
            self._plant.set_initial_pose(np.zeros(0, np.float32))
            return
        if values.shape != (36,):
            raise ValueError("initial pose must have 36 values")
        converted = np.concatenate([values[:7], self._to_sdk(values[7:])])
        self._plant.set_initial_pose(converted)

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


def verify_native_bundle(
    bundle: PolicyBundle, *, warmup_iterations: int = 8
) -> dict[str, float | int]:
    """Replay bundle golden traces in the C++ ONNX Runtime engine."""
    try:
        import ec_native
    except ImportError as exc:  # pragma: no cover - depends on optional build
        raise ImportError(
            "native bundle verification needs the native Pixi environment"
        ) from exc

    trace = np.load(bundle.root / "golden_trace.npz")
    report: dict[str, float | int] = {"rows": int(trace["obs"].shape[0])}
    for model_name, input_key, output_key, report_key in (
        ("policy_onnx", "obs", "action", "policy_max_abs_error"),
        ("encoder_onnx", "encoder_in", "encoder_out", "encoder_max_abs_error"),
    ):
        artifact = bundle.manifest.models.get(model_name)
        if artifact is None:
            if model_name == "policy_onnx":
                raise ValueError("native bundle verification requires policy_onnx")
            continue
        if input_key not in trace or output_key not in trace:
            raise ValueError(f"golden trace is missing {input_key!r} or {output_key!r}")
        engine = ec_native.OnnxEngine(
            str(bundle.model_path(model_name, expected_format="onnx")),
            artifact.input_name,
            artifact.output_name,
            artifact.input_shape[1],
            artifact.output_shape[1],
        )
        engine.warmup(warmup_iterations)
        outputs = np.stack(
            [
                engine.infer(np.asarray(row, dtype=np.float32))
                for row in trace[input_key]
            ]
        )
        worst = float(np.abs(outputs - trace[output_key]).max())
        if worst > artifact.parity_atol:
            raise ValueError(
                f"{model_name} native parity failed: {worst} > {artifact.parity_atol}"
            )
        report[report_key] = worst
    return report


def benchmark_native_bundle(
    bundle: PolicyBundle,
    *,
    ticks: int = 10_000,
    policy_threads: int = 4,
    lead_ticks: int = 4,
    paced: bool = False,
    cpu: int = -1,
    fifo_priority: int = 0,
    lock_memory: bool = False,
    require_realtime: bool = False,
) -> dict[str, float | int]:
    """Measure the warmed C++ control hot path without Python tick callbacks."""
    try:
        import ec_native
    except ImportError as exc:  # pragma: no cover - optional build
        raise ImportError("native benchmark needs the native Pixi environment") from exc
    if ticks < 100:
        raise ValueError("native benchmark needs at least 100 ticks")
    slot_name = f"/ec_bench_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    loop = NativeFakeLoop(
        bundle,
        response_slot=slot_name,
        lead_ticks=lead_ticks,
        command_stale_ms=60_000.0,
        policy_threads=policy_threads,
        cpu=cpu,
        fifo_priority=fifo_priority,
        lock_memory=lock_memory,
        require_realtime=require_realtime,
    )
    publisher = ec_native.ShmCommandSlot(slot_name, False)
    command = bundle.manifest.command
    if bundle.manifest.models.get("encoder_onnx") is not None:
        window_frames = int((command.window_steps or 9) + 1)
        width = int(command.state_dim or 38)
        payload = np.zeros((lead_ticks + window_frames) * width, np.float32)
        tag = 2
    else:
        payload = np.zeros(loop.tracker.command_width, np.float32)
        tag = 1 if bundle.manifest.interface == "latent" else 0
    publisher.publish(1, tag, payload, time.monotonic())
    loop.start(ticks, paced=paced)
    loop.wait()
    durations_ms = loop.tick_durations_ns().astype(np.float64) / 1e6
    stats = loop.stats()
    return {
        "ticks": int(stats["ticks"]),
        "policy_threads": int(policy_threads),
        "paced": bool(paced),
        "cpu": int(cpu),
        "fifo_priority": int(fifo_priority),
        "realtime_configured": bool(stats["realtime_configured"]),
        "tick_ms_p50": float(np.percentile(durations_ms, 50)),
        "tick_ms_p95": float(np.percentile(durations_ms, 95)),
        "tick_ms_p99": float(np.percentile(durations_ms, 99)),
        "tick_ms_p999": float(np.percentile(durations_ms, 99.9)),
        "tick_ms_max": float(durations_ms.max()),
        "scheduler_deadlines_missed": int(stats["scheduler_deadlines_missed"]),
        "wake_late_ms_max": float(stats["wake_late_ns_max"]) / 1e6,
        "fault": int(stats["fault"]),
    }


__all__ = [
    "NativeDdsPlant",
    "NativeFakeLoop",
    "NativeMujocoLoop",
    "NativeTracker",
    "NativeUnitreeLoop",
    "benchmark_native_bundle",
    "verify_native_bundle",
]
