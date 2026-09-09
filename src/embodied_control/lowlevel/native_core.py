"""Thin Python construction API for the native allocation-free tracker core."""

from __future__ import annotations

import os
import time
import uuid

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.contracts import RobotState
from embodied_control.sim.dds_plant import NativeDdsPlant

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


def _native_reference_layout(bundle: PolicyBundle) -> str:
    command = bundle.manifest.command
    if (
        command.encoder_state_interface == "root_qpos"
        and command.macro_anchor_mode == "robot_heading"
    ):
        return "root_qpos_heading"
    return str(command.encoder_state_interface or "root_qpos")


def _oracle_enabled(bundle: PolicyBundle, command_source: str) -> bool:
    if command_source not in {"vla", "oracle"}:
        raise ValueError("command_source must be 'vla' or 'oracle'")
    oracle = command_source == "oracle"
    command = bundle.manifest.command
    expected_anchor = {
        "root_qpos": {"robot", "robot_heading"},
        "joint_qpos_qvel_anchor_ori": {"robot_heading"},
    }.get(command.encoder_state_interface)
    if oracle and (
        expected_anchor is None
        or command.macro_anchor_mode not in expected_anchor
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
        plan_slots: int = 1,
        latent_plan: bool = False,
        cpu: int = -1,
        fifo_priority: int = 0,
        lock_memory: bool = False,
        require_realtime: bool = False,
        policy_threads: int = 4,
        command_source: str = "vla",
        sensor_noise: dict[str, float] | None = None,
        noise_seed: int = 0,
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
        # A latent plan carries the head's own latents, so the tracker-side
        # encoder stays out of the loop entirely.
        if encoder is not None and not latent_plan:
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
            int(plan_slots),
            bool(latent_plan),
            int(command.state_dim or 38),
            int((command.window_steps or 9) + 1),
            int(command.z_dim or self.tracker.command_width),
            command.phase_mode == "sin_cos",
            _direct_command_tag(bundle),
            oracle_reference,
            _native_reference_layout(bundle),
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

    def close(self) -> None:
        """Stop the loop and drop the native runtime (its writer joins in the
        destructor). The object is unusable afterwards; the session builds a
        new one for the next selection."""
        runtime = getattr(self, "_runtime", None)
        if runtime is None:
            return
        force_damp = getattr(runtime, "force_damp", None)
        if force_damp is not None:
            force_damp()
        try:
            if runtime.running:
                runtime.stop()
            runtime.wait()
        except RuntimeError:
            pass
        self._runtime = None

    def set_reference_paused(self, paused: bool) -> None:
        """Pin the reference clock at frame 0 until the policy owns the joints."""
        self._runtime.set_reference_paused(bool(paused))

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
        return np.asarray(self._runtime.joint_position_log()).reshape(-1, 29)

    def anchor_pose_log(self) -> np.ndarray:
        return np.asarray(self._runtime.anchor_pose_log()).reshape(-1, 7)

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
        plan_slots: int = 1,
        latent_plan: bool = False,
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
        sensor_noise: dict[str, float] | None = None,
        noise_seed: int = 0,
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
        # A latent plan carries the head's own latents, so the tracker-side
        # encoder stays out of the loop entirely.
        if encoder is not None and not latent_plan:
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
            int(plan_slots),
            bool(latent_plan),
            int(command.state_dim or 38),
            int((command.window_steps or 9) + 1),
            int(command.z_dim or self.tracker.command_width),
            command.phase_mode == "sin_cos",
            _direct_command_tag(bundle),
            oracle_reference,
            _native_reference_layout(bundle),
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
            # SONIC's observation noise on the controller's view only; the
            # metric logs keep the clean state.
            float((sensor_noise or {}).get("joint_pos", 0.0)),
            float((sensor_noise or {}).get("joint_vel", 0.0)),
            float((sensor_noise or {}).get("base_ang_vel", 0.0)),
            float((sensor_noise or {}).get("projected_gravity", 0.0)),
            int(noise_seed),
        )


class NativeUnitreeLoop(NativeFakeLoop):
    """C++ G1 DDS runtime with explicit write and operator gates."""

    DISABLED = 0
    INITIALIZE = 1
    WAIT = 2
    CONTROL = 3
    DAMP = 4
    HOLD = 5

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
        plan_slots: int = 1,
        latent_plan: bool = False,
        command_source: str = "vla",
        fixed_anchor_position: np.ndarray | None = None,
        fixed_anchor_quaternion: np.ndarray | None = None,
        control_cpu: int = 2,
        writer_cpu: int = 3,
        control_fifo_priority: int = 80,
        writer_fifo_priority: int = 90,
        lock_memory: bool = True,
        require_realtime: bool = True,
        policy_threads: int = 1,
        dds_domain: int = 0,
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
        oracle_reference = _oracle_enabled(bundle, command_source)
        if oracle_reference and fixed_anchor_position is None:
            raise ValueError(
                "Unitree oracle control requires a fixed initial anchor position"
            )
        if not oracle_reference and fixed_anchor_position is not None:
            raise ValueError(
                "fixed initial anchor position is only valid for oracle control"
            )
        fixed_anchor = None
        fixed_quaternion = None
        if fixed_anchor_position is not None:
            fixed_anchor = np.asarray(fixed_anchor_position, dtype=np.float32)
            if fixed_anchor.shape != (3,) or not np.isfinite(fixed_anchor).all():
                raise ValueError("fixed initial anchor position must be 3 finite values")
            if fixed_anchor_quaternion is None:
                raise ValueError(
                    "a fixed initial anchor needs its reference orientation (XYZW)"
                )
            fixed_quaternion = np.asarray(fixed_anchor_quaternion, dtype=np.float32)
            if fixed_quaternion.shape != (4,) or not np.isfinite(fixed_quaternion).all():
                raise ValueError("fixed initial anchor quaternion must be 4 finite values")
        encoder = bundle.manifest.models.get("encoder_onnx")
        encoder_path = ""
        encoder_input_name = ""
        encoder_output_name = ""
        encoder_input_width = 0
        encoder_output_width = 0
        # A latent plan carries the head's own latents, so the tracker-side
        # encoder stays out of the loop entirely.
        if encoder is not None and not latent_plan:
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
            "plan_slots": int(plan_slots),
            "latent_plan": bool(latent_plan),
            "oracle_reference": oracle_reference,
            "root_qpos_width": int(command.state_dim or 38),
            "window_frames": int((command.window_steps or 9) + 1),
            "z_dim": int(command.z_dim or self.tracker.command_width),
            "sin_cos_phase": command.phase_mode == "sin_cos",
            "direct_tag": _direct_command_tag(bundle),
            "reference_encoder_layout": _native_reference_layout(bundle),
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
            # 0 is the robot's domain; a simulated plant pair may isolate
            # itself so two rig processes on `lo` never cross-talk.
            "dds_domain": int(dds_domain),
        }
        if fixed_anchor is not None:
            config["fixed_anchor_position"] = fixed_anchor.tolist()
            config["fixed_anchor_quaternion"] = fixed_quaternion.tolist()
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

    def begin_initialization(
        self,
        duration_seconds: float = 3.0,
        *,
        target_position: np.ndarray | None = None,
        hold_current: bool = False,
        skip_motion_switcher: bool = False,
    ) -> None:
        """Ramp to a pose: the bundle default stance, the current pose, or a
        caller-supplied predefined qpos.

        The default stance is the hardware sequence. A rehearsal episode that
        starts ON a reference frame passes ``hold_current=True`` so the ramp
        does not drag the robot off that frame before the planner takes over.
        ``target_position`` (29 values, Isaac order) overrides both.
        """
        target: list[float] = []
        if target_position is not None:
            values = np.asarray(target_position, dtype=np.float32)
            if values.shape != (29,) or not np.isfinite(values).all():
                raise ValueError(
                    "initialization target must be 29 finite values"
                )
            target = values.tolist()
        self._runtime.begin_initialization(
            float(duration_seconds),
            target,
            bool(hold_current),
            bool(skip_motion_switcher),
        )

    def wait_for_mode(self, expected: int, timeout_seconds: float) -> bool:
        return bool(self._runtime.wait_for_mode(int(expected), float(timeout_seconds)))

    def engage_control(self, blend_ticks: int = 0) -> None:
        self._runtime.engage_control(int(blend_ticks))

    def force_damp(self) -> None:
        self._runtime.force_damp()

    def writer_stats(self) -> dict[str, int | bool]:
        return dict(self._runtime.writer_stats())

    # The lifecycle sequence (docs/design/robot_lifecycle.md). Every call is
    # a blocking RPC or a guarded mode store; none is safe on a control thread.

    def vendor_mode(self) -> str:
        return str(self._runtime.vendor_mode())

    def open_damp_gate(self) -> None:
        self._runtime.open_damp_gate()

    def release_vendor(self) -> None:
        self._runtime.release_vendor()

    def hold(self) -> None:
        self._runtime.hold()

    def set_ramp_guard(self, rad: float, ticks: int) -> None:
        self._runtime.set_ramp_guard(float(rad), int(ticks))

    def set_hold_gain_scale(self, scale: float) -> None:
        self._runtime.set_hold_gain_scale(float(scale))

    def clear_latched_fault(self) -> None:
        self._runtime.clear_latched_fault()

    def close_gate(self) -> None:
        self._runtime.close_gate()

    def restore_vendor(self, name: str) -> None:
        self._runtime.restore_vendor(str(name))

    def latest_state(self) -> dict[str, np.ndarray | bool]:
        """Fresh rt/lowstate snapshot for the operator thread.

        ``state()`` is the control thread's cached view and stops updating
        when that thread is not running; the lifecycle's pose gates read
        this instead.
        """
        raw = self._runtime.latest_state()
        return {
            "valid": bool(raw["valid"]),
            "joint_position": np.asarray(raw["joint_position"], dtype=np.float32),
            "joint_velocity": np.asarray(raw["joint_velocity"], dtype=np.float32),
            "projected_gravity": np.asarray(
                raw["projected_gravity"], dtype=np.float32
            ),
        }

    def command_target_error(self) -> np.ndarray:
        """Per-joint |policy target - held pose| measured in WAIT, Isaac order."""
        return np.asarray(self._runtime.command_target_error(), dtype=np.float32)

    def joint_position(self) -> np.ndarray:
        return self.latest_state()["joint_position"]

    def joint_velocity(self) -> np.ndarray:
        return self.latest_state()["joint_velocity"]

    def projected_gravity(self) -> np.ndarray:
        return self.latest_state()["projected_gravity"]


class NativePlantClient:
    """Client for plant-only gantry, reset, and status controls.

    Hardware has no RPC for the gantry, so the lifecycle only reaches for
    this when it is rehearsing against the plant; on the robot the same step
    is an operator acknowledgement.
    """

    def __init__(
        self,
        network_interface: str = "lo",
        *,
        dds_domain: int = 0,
        timeout_seconds: float = 2.0,
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "NativePlantClient needs the Unitree-enabled native build"
            ) from exc
        if not ec_native.WITH_UNITREE:
            raise RuntimeError("ec_native was built without Unitree SDK2")
        self._client = ec_native.PlantClient(
            str(network_interface), int(dds_domain), float(timeout_seconds)
        )

    def hoist(self) -> None:
        self._client.hoist()

    def lower(self) -> None:
        self._client.lower()

    def slack(self) -> None:
        self._client.slack()

    def reset(self) -> None:
        self._client.reset()
        deadline = time.monotonic() + 2.0
        while self.status().get("reset_pending", True):
            if time.monotonic() >= deadline:
                raise RuntimeError("plant did not apply reset before timeout")
            time.sleep(0.005)

    def status(self) -> dict:
        import json

        return json.loads(self._client.status())


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
