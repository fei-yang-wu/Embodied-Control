"""The hoist-to-run lifecycle against the vendored MuJoCo plant on lo.

Same code path as hardware: the plant serves the G1's sport and
motion-switcher RPCs, owns the joints until ReleaseMode, hangs the robot from
a virtual gantry, and rejects rt/lowcmd sent in the wrong mode. The lifecycle
under test is the exact object `ec lifecycle` drives on the robot; only the
network interface differs.
"""

import json
import threading
import time

import numpy as np
import pytest

ec_native = pytest.importorskip("ec_native")
pytest.importorskip("torch")
pytest.importorskip("onnx")

if not ec_native.WITH_UNITREE:  # pragma: no cover - depends on optional build
    pytest.skip("ec_native was built without Unitree SDK2", allow_module_level=True)

from test_dds_plant import (  # noqa: E402
    DEFAULT_POSE,
    _finish_plant,
    _g1_bundle,
    _spawn_plant,
)
from test_native_core import _g1_mjcf_path, _shm_name  # noqa: E402

from embodied_control.lowlevel.native_core import (  # noqa: E402
    NativePlantClient,
    NativeUnitreeLoop,
)
from embodied_control.robot import RobotMode  # noqa: E402
from embodied_control.robot.g1 import G1Runtime  # noqa: E402
from embodied_control.robot.lifecycle import (  # noqa: E402
    Lifecycle,
    LifecycleConfig,
    LifecycleLog,
    LifecycleState as S,
)

# The SDK's DDS factory is process-global. Other native tests initialize the
# host process on the robot's default domain, so the loopback uses that same
# domain; the plant itself remains isolated in its own subprocess.
DOMAIN = 0
QUIET_PLANT = [
    "--noise-joint-pos", "0", "--noise-joint-vel", "0",
    "--noise-base-ang-vel", "0", "--noise-imu-tilt-rad", "0",
]


class _Feeder:
    """A stand-in planner: publishes the zero latent so the toy policy holds."""

    def __init__(self, slot_name, width):
        self._publisher = ec_native.ShmCommandSlot(slot_name, False)
        self._payload = np.zeros(width, dtype=np.float32)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        sequence = 1
        while not self._stop.is_set():
            self._publisher.publish(sequence, 1, self._payload, time.monotonic())
            sequence += 1
            self._stop.wait(0.05)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


def _runtime(bundle, response_name):
    return NativeUnitreeLoop(
        bundle,
        "lo",
        response_slot=response_name,
        writes_enabled=True,
        control_cpu=-1,
        writer_cpu=-1,
        control_fifo_priority=0,
        writer_fifo_priority=0,
        lock_memory=False,
        require_realtime=False,
        command_stale_ms=1000.0,
        dds_domain=DOMAIN,
    )


def _blocked_mjcf(tmp_path):
    source = _g1_mjcf_path().read_text()
    source = source.replace(
        'meshdir="meshes"',
        f'meshdir="{_g1_mjcf_path().parent / "meshes"}"',
        1,
    )
    source = source.replace(
        "</mujoco>",
        '<equality><weld body1="left_hip_pitch_link" body2="pelvis"/></equality></mujoco>',
        1,
    )
    model = tmp_path / "blocked_g1.xml"
    model.write_text(source)
    return model


def test_vendor_owns_the_joints_until_released(tmp_path, latent_manifest):
    bundle = _g1_bundle(tmp_path, latent_manifest)
    report_path = tmp_path / "plant_report.json"
    # Hung 10 deg nose-down, so the hardware path's projected gravity has a
    # sign to get wrong: a forward pitch puts gravity along +x of the body.
    half = np.radians(5.0)
    pose = np.concatenate([
        [0.0, 0.0, 0.76], [0.0, np.sin(half), 0.0, np.cos(half)],
        np.asarray(DEFAULT_POSE, dtype=np.float32),
    ]).astype(np.float32)
    pose_path = tmp_path / "pitched_pose.npy"
    np.save(pose_path, pose)
    process = _spawn_plant(
        bundle.root, _g1_mjcf_path(), 60.0, report_path,
        extra=[*QUIET_PLANT, "--vendor", "--hoist", "--dds-domain", str(DOMAIN),
               "--initial-pose", str(pose_path)],
    )
    runtime = None
    try:
        vendor = G1Runtime("lo", writes_enabled=True, dds_domain=DOMAIN, timeout_seconds=2.0)
        assert vendor.mode() is RobotMode.DAMP
        vendor.stand()
        assert vendor.mode() is RobotMode.READY

        runtime = _runtime(bundle, _shm_name("lifecycle_owned"))
        assert runtime.wait_for_state(10.0)
        gravity = runtime.latest_state()["projected_gravity"]
        # The strap levels the pelvis toward upright, so only part of the
        # 10 deg survives; the sign is the regression under test (the old
        # formula reported a forward pitch as gravity along -x).
        assert 0.03 < gravity[0] < np.sin(np.radians(10.0)) + 0.03, gravity
        assert gravity[2] < -0.95, gravity
        assert runtime.vendor_mode() == "ai"
        runtime.open_damp_gate()
        time.sleep(0.3)
        # ReleaseMode from a standing vendor is refused, exactly as on the G1.
        with pytest.raises(RuntimeError):
            runtime.release_vendor()
        assert runtime.writer_stats()["vendor_released"] is False
        assert NativePlantClient("lo", dds_domain=DOMAIN).status()["owned"] is True

        vendor.damp()
        runtime.release_vendor()
        assert runtime.writer_stats()["vendor_released"] is True
        assert runtime.vendor_mode() == ""
        time.sleep(0.3)
        runtime.force_damp()
        runtime.close_gate()
        runtime.restore_vendor("ai")
        assert runtime.vendor_mode() == "ai"
        assert vendor.mode() is RobotMode.DAMP
        report = _finish_plant(process, report_path)
    finally:
        if runtime is not None:
            runtime.force_damp()
        if process.poll() is None:
            process.kill()
            process.communicate()
    # Our damp frames before the release were counted and dropped; the ones
    # after it were applied; none carried the wrong mode_machine.
    assert report["rejected_commands"] > 50
    assert report["commands_received"] > 50
    assert report["mode_machine_rejections"] == 0
    assert report["vendor_owned"] is True
    assert report["physics_fault"] is False


def test_blocked_joint_trips_native_ramp_guard(tmp_path, latent_manifest):
    bundle = _g1_bundle(tmp_path, latent_manifest)
    blocked_target = list(DEFAULT_POSE)
    blocked_target[0] += 0.4
    report_path = tmp_path / "plant_report.json"
    process = _spawn_plant(
        bundle.root,
        _blocked_mjcf(tmp_path),
        30.0,
        report_path,
        extra=[*QUIET_PLANT, "--vendor", "--hoist", "--dds-domain", str(DOMAIN)],
    )
    runtime = None
    lifecycle = None
    report = None
    try:
        response_name = _shm_name("lifecycle_blocked")
        runtime = _runtime(bundle, response_name)
        vendor = G1Runtime("lo", writes_enabled=True, dds_domain=DOMAIN, timeout_seconds=2.0)
        lifecycle = Lifecycle(
            runtime,
            vendor,
            LifecycleConfig(
                start_pose=blocked_target,
                ramp_seconds=1.0,
                ramp_fault_rad=0.1,
                ramp_fault_ms=10.0,
                allow_non_realtime=True,
                settle_seconds=0.1,
                drift_rad=0.1,
                settle_timeout_seconds=10.0,
            ),
            hoist=NativePlantClient("lo", dds_domain=DOMAIN),
            auto_ack=True,
            log=LifecycleLog(tmp_path / "lifecycle"),
            note=lambda msg: print("  --", msg),
        )
        result = lifecycle.auto()
        assert not result.ok
        assert lifecycle.state is S.FAULT
        assert "ramp guard" in lifecycle.fault_reason
        stats = runtime.writer_stats()
        assert stats["ramp_faults"] >= 1, stats
        assert stats["ramp_error_joint"] == 0, stats
        runtime.close_gate()
        runtime.restore_vendor("ai")
        lifecycle.shutdown()
        report = _finish_plant(process, report_path)
    finally:
        if lifecycle is not None:
            lifecycle.damp()
        if runtime is not None:
            runtime.force_damp()
            try:
                runtime.close_gate()
            except RuntimeError:
                pass
            if runtime.running:
                runtime.stop()
                runtime.wait()
        if process.poll() is None:
            process.kill()
            process.communicate()
    assert report is not None
    assert report["physics_fault"] is False
    assert report["vendor_owned"] is True


def test_lifecycle_hoist_to_standing_on_the_plant(tmp_path, latent_manifest):
    bundle = _g1_bundle(tmp_path, latent_manifest)
    report_path = tmp_path / "plant_report.json"
    process = _spawn_plant(
        bundle.root, _g1_mjcf_path(), 120.0, report_path,
        extra=[*QUIET_PLANT, "--vendor", "--hoist", "--dds-domain", str(DOMAIN)],
    )
    runtime = None
    feeder = None
    try:
        response_name = _shm_name("lifecycle")
        runtime = _runtime(bundle, response_name)
        feeder = _Feeder(response_name, runtime.tracker.command_width)
        feeder.start()
        vendor = G1Runtime("lo", writes_enabled=True, dds_domain=DOMAIN, timeout_seconds=2.0)
        hoist = NativePlantClient("lo", dds_domain=DOMAIN)
        config = LifecycleConfig(
            start_pose=list(DEFAULT_POSE),
            ramp_seconds=1.0,
            ticks=150,
            blend_ticks=50,
            allow_non_realtime=True,
            # The toy bundle's 300 N m/rad hold sags under gravity once the
            # feet carry the weight; this test is about the sequence, not the
            # gains, so the pose gates are loose here.
            settle_position_rad=0.15,
            pose_tolerance_rad=0.15,
            settle_timeout_seconds=15.0,
            hoist_release_seconds=1.5,
        )
        lifecycle = Lifecycle(
            runtime, vendor, config, hoist=hoist, auto_ack=True,
            log=LifecycleLog(tmp_path / "lifecycle"),
            note=lambda msg: print("  --", msg),
        )

        result = lifecycle.auto()
        assert result.ok, (result, lifecycle.state)
        assert lifecycle.state is S.PRIMED
        assert lifecycle.vendor_name == "ai"
        status = hoist.status()
        assert (status["owned"], status["fsm_id"], status["hoist_mode"]) == (False, 1, 2)
        assert runtime.unitree_mode == NativeUnitreeLoop.WAIT

        result = lifecycle.go()
        assert result.ok, (result, lifecycle.state)
        assert lifecycle.state is S.RUNNING
        deadline = time.monotonic() + 20.0
        while runtime.running and time.monotonic() < deadline:
            lifecycle.poll()
            time.sleep(0.01)
        lifecycle.poll()
        assert lifecycle.state is S.HOLD, (lifecycle.state, lifecycle.fault_reason)
        assert runtime.unitree_mode == NativeUnitreeLoop.HOLD
        stats = runtime.stats()
        assert stats["fault"] == 0, stats
        assert stats["control_ticks"] >= 140
        # The reference clock stood still through the 50-writer-tick blend.
        assert stats["reference_ticks"] <= stats["control_ticks"] - 3, stats

        result = lifecycle.recover()
        assert result.ok, (result, lifecycle.state)
        assert lifecycle.state is S.STANDING
        assert vendor.mode() is RobotMode.READY
        assert runtime.unitree_mode == NativeUnitreeLoop.DISABLED
        assert runtime.writer_stats()["gate_open"] is False
        assert hoist.status()["owned"] is True and hoist.status()["hoist_mode"] == 2

        # A second take without handing back: HOLD -> ramp -> PRIMED again.
        assert lifecycle.advance().ok
        assert lifecycle.state is S.PRECHECK
        assert lifecycle.episode == 2

        lifecycle.shutdown()
        report = _finish_plant(process, report_path)
    finally:
        if feeder is not None:
            feeder.stop()
        if runtime is not None:
            runtime.force_damp()
            if runtime.running:
                runtime.stop()
                runtime.wait()
        if process.poll() is None:
            process.kill()
            process.communicate()

    lines = (tmp_path / "lifecycle" / "lifecycle.jsonl").read_text().splitlines()
    transitions = [json.loads(line) for line in lines]
    assert [t["to_state"] for t in transitions][:11] == [
        "PRECHECK", "VENDOR_DAMP_CONFIRMED", "SAFE_EXTERNAL_COMMAND_PRESENT",
        "USER_CONTROL_CONFIRMED", "START_POSE_RAMP", "POSE_SETTLED", "LOWERED",
        "POSE_MATCH_VERIFIED", "POLICY_COMMAND_FRESH", "PRIMED", "BLEND_IN",
    ]
    assert all(t["ok"] for t in transitions), [t for t in transitions if not t["ok"]]
    assert (tmp_path / "lifecycle" / "pose_match.json").exists()
    assert report["rejected_commands"] > 0
    assert report["mode_machine_rejections"] == 0
    assert report["crc_errors"] == 0
    assert report["physics_fault"] is False
    assert report["vendor_owned"] is True
    # It ended standing on its feet under the vendor, not on the floor.
    assert report["base_height"] > 0.5, report
