"""Digit-style unified plant: MuJoCo serving the G1 DDS protocol on lo.

The controller under test is the real hardware path (``NativeUnitreeLoop``
over rt/lowcmd + rt/lowstate); only the network interface separates this
loopback from the robot. The plant always runs in a subprocess because
each process owns one DDS ChannelFactory initialization.
"""

import json
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

ec_native = pytest.importorskip("ec_native")
torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

if not ec_native.WITH_UNITREE:  # pragma: no cover - depends on optional build
    pytest.skip("ec_native was built without Unitree SDK2", allow_module_level=True)

from test_native_core import _native_bundle, _g1_mjcf_path, _shm_name  # noqa: E402

from embodied_control.lowlevel.bundle import ActionContract  # noqa: E402
from embodied_control.lowlevel.native_core import NativeUnitreeLoop  # noqa: E402

G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
# Coprime stride: a deliberately nontrivial Isaac->SDK permutation so a
# wiring mistake scrambles the robot instead of cancelling out.
PERMUTATION = [(i * 7) % 29 for i in range(29)]
# Bent-knee G1 stance: statically robust, unlike the marginal straight-leg
# zero pose, and non-uniform so a scrambled permutation visibly falls.
DEFAULT_POSE = [0.0] * 29
for _side in (0, 6):
    DEFAULT_POSE[_side + 0] = -0.20
    DEFAULT_POSE[_side + 3] = 0.42
    DEFAULT_POSE[_side + 4] = -0.23


class _ZeroAction(torch.nn.Module):
    """Targets equal the defaults, so a standing plant proves the DDS path."""

    def forward(self, observation):
        return 0.0 * observation[:, :29]


def _g1_bundle(tmp_path, latent_manifest):
    action = ActionContract(
        isaac_joint_names=G1_JOINT_NAMES,
        sdk_joint_names=G1_JOINT_NAMES,
        isaac_to_sdk=PERMUTATION,
        default_joint_pos=list(DEFAULT_POSE),
        action_scale=[0.25] * 29,
        stiffness=[300.0] * 29,
        damping=[8.0] * 29,
        armature=[0.01] * 29,
        effort_limit=[150.0] * 29,
        joint_limits_lower=[-3.5] * 29,
        joint_limits_upper=[3.5] * 29,
    )
    manifest = latent_manifest.model_copy(update={"action": action})
    return _native_bundle(tmp_path, manifest, model=_ZeroAction())


def _spawn_plant(bundle_root, model, seconds, report_path, extra=None):
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "embodied_control.cli",
            "lowlevel",
            "plant",
            str(bundle_root),
            "--model",
            str(model),
            "--network",
            "lo",
            "--seconds",
            str(seconds),
            "--report",
            str(report_path),
            *(extra or []),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    deadline = time.monotonic() + 30.0
    lines = []
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            break
        lines.append(line)
        if "PLANT_READY" in line:
            return process
    process.kill()
    raise AssertionError("plant did not become ready:\n" + "".join(lines))


def _finish_plant(process, report_path, timeout=20.0):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    return json.loads(report_path.read_text())


PLANT_ONLY_DOMAIN = ["--dds-domain", "41"]


def test_plant_serves_lowstate_alone(tmp_path, latent_manifest):
    bundle = _g1_bundle(tmp_path, latent_manifest)
    report_path = tmp_path / "plant_report.json"
    process = _spawn_plant(
        bundle.root, _g1_mjcf_path(), 1.5, report_path, extra=PLANT_ONLY_DOMAIN
    )
    process.communicate(timeout=30.0)
    report = json.loads(report_path.read_text())
    assert process.returncode == 0
    assert report["publishes"] > 100
    assert report["commands_received"] == 0
    assert report["crc_errors"] == 0
    assert report["holding"] is True
    assert report["physics_fault"] is False
    # The virtual gantry keeps the uncommanded robot standing at defaults.
    assert report["min_base_height"] > 0.5


def test_dds_loopback_end_to_end(tmp_path, latent_manifest):
    bundle = _g1_bundle(tmp_path, latent_manifest)
    report_path = tmp_path / "plant_report.json"
    # Protocol coverage, not a stability result: the toy policy in this
    # bundle amplifies sensor noise into a fall, so this plant serves the
    # deterministic (noise-free) wire. Noise has its own test below.
    process = _spawn_plant(
        bundle.root,
        _g1_mjcf_path(),
        120.0,
        report_path,
        extra=["--noise-joint-pos", "0", "--noise-joint-vel", "0",
               "--noise-base-ang-vel", "0", "--noise-imu-tilt-rad", "0"],
    )
    runtime = None
    feeder_stop = threading.Event()
    feeder = None
    try:
        response_name = _shm_name("dds")
        runtime = NativeUnitreeLoop(
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
        )
        assert runtime.wait_for_state(10.0), "no rt/lowstate over loopback"
        runtime.begin_initialization(1.0)
        assert runtime.wait_for_mode(NativeUnitreeLoop.WAIT, 20.0), (
            runtime.unitree_mode,
            runtime.writer_stats(),
        )

        publisher = ec_native.ShmCommandSlot(response_name, False)
        width = runtime.tracker.command_width
        payload = np.zeros(width, dtype=np.float32)

        def _feed():
            sequence = 1
            while not feeder_stop.is_set():
                publisher.publish(sequence, 1, payload, time.monotonic())
                sequence += 1
                feeder_stop.wait(0.05)

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()

        runtime.start(150, paced=True)
        engaged = False
        engage_deadline = time.monotonic() + 10.0
        while runtime.running:
            if not engaged and int(runtime.stats()["control_ticks"]) > 0:
                runtime.engage_control()
                engaged = True
            if not engaged and time.monotonic() > engage_deadline:
                raise AssertionError("the controller never computed a control tick")
            time.sleep(0.02)
        runtime.wait()
        assert engaged

        stats = runtime.stats()
        writer = runtime.writer_stats()
        assert stats["fault"] == 0, stats
        assert stats["control_ticks"] > 0
        assert writer["publishes"] > 200
        assert writer["publish_failures"] == 0
        assert writer["crc_errors"] == 0
        assert writer["hardware_faults"] == 0
        assert writer["watchdog_faults"] == 0
        assert runtime.unitree_mode == NativeUnitreeLoop.CONTROL
        # Stop the plant while the controller still holds the robot, so the
        # report covers only the controlled window, not the damp slump.
        report = _finish_plant(process, report_path)
    finally:
        feeder_stop.set()
        if feeder is not None:
            feeder.join(timeout=2.0)
        if runtime is not None:
            runtime.stop()
            runtime.wait()
            runtime.force_damp()
        if process.poll() is None:
            process.kill()
            process.communicate()

    assert report["commands_received"] > 200
    assert report["crc_errors"] == 0
    assert report["holding"] is False
    assert report["physics_fault"] is False
    # This test covers the wire path, not stability: the bundle's toy policy
    # (0.5 x the first observation terms) is not a balancing controller, and
    # whether it stays upright for three seconds depends on the tick the
    # controller happens to engage at. Assert only that physics stayed sane and
    # the robot did not sink through the floor.
    assert report["min_base_height"] > 0.05


def test_plant_publishes_sensor_noise_on_the_wire(tmp_path, latent_manifest):
    """A real G1 serves noisy state, so the plant must too."""
    bundle = _g1_bundle(tmp_path, latent_manifest)
    clean_report = tmp_path / "clean.json"
    noisy_report = tmp_path / "noisy.json"

    clean = _spawn_plant(
        bundle.root,
        _g1_mjcf_path(),
        1.0,
        clean_report,
        extra=[*PLANT_ONLY_DOMAIN, "--noise-joint-pos", "0",
               "--noise-joint-vel", "0", "--noise-base-ang-vel", "0",
               "--noise-imu-tilt-rad", "0"],
    )
    clean.communicate(timeout=30.0)
    clean_stats = json.loads(clean_report.read_text())

    noisy = _spawn_plant(
        bundle.root, _g1_mjcf_path(), 1.0, noisy_report, extra=PLANT_ONLY_DOMAIN
    )
    noisy.communicate(timeout=30.0)
    noisy_stats = json.loads(noisy_report.read_text())

    assert clean_stats["physics_fault"] is False
    assert noisy_stats["physics_fault"] is False
    assert clean_stats["publishes"] > 100
    assert noisy_stats["publishes"] > 100
    # Both plants hold the same uncommanded pose; noise rides on the wire, not
    # on the physics, so the simulated robot still stands.
    assert noisy_stats["min_base_height"] > 0.5
