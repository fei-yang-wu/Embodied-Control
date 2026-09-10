import pytest

pytest.importorskip("numpy")

from embodied_control.lowlevel.unitree_probe import finalize_probe_report, run_probe


def _raw(**overrides):
    raw = {
        "samples": 500,
        "crc_errors": 0,
        "nonfinite_samples": 0,
        "motor_error_samples": 0,
        "duplicate_ticks": 0,
        "first_receive_ns": 1_000_000_000,
        "last_receive_ns": 2_000_000_000,
        "max_gap_ns": 3_000_000,
        "first_tick": 100,
        "last_tick": 599,
        "motor_error_mask": 0,
        "mode_machine": 5,
        "snapshot_consistent": True,
        "joint_position": [0.0] * 29,
        "joint_velocity": [0.0] * 29,
        "joint_torque": [0.0] * 29,
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "gyroscope": [0.0] * 3,
        "accelerometer": [0.0, 0.0, 9.81],
    }
    raw.update(overrides)
    return raw


def _report(raw):
    return finalize_probe_report(
        raw,
        network="eth0",
        requested_samples=500,
        min_rate_hz=100.0,
        max_gap_ms=100.0,
    )


def test_probe_report_passes_healthy_stream():
    report = _report(_raw())

    assert report["status"] == "pass"
    assert report["rate_hz"] == 499.0
    assert report["writes_enabled"] is False


def test_probe_report_identifies_data_and_motor_failures():
    report = _report(
        _raw(
            crc_errors=2,
            motor_error_samples=4,
            motor_error_mask=(1 << 3) | (1 << 9),
        )
    )

    assert report["status"] == "fail"
    assert report["motor_error_joints"] == [3, 9]
    assert report["checks"]["crc"] is False
    assert report["checks"]["motors"] is False


def test_probe_report_fails_when_tick_does_not_advance():
    report = _report(
        _raw(first_tick=100, last_tick=100, duplicate_ticks=499)
    )

    assert report["status"] == "fail"
    assert report["checks"]["tick_progress"] is False


def test_probe_preserves_transient_firmware_error_after_motor_recovers():
    codes = [0] * 29
    codes[0] = 0x80000001
    report = _report(_raw(
        motor_error_samples=1,
        motor_error_mask=1,
        first_motor_error_code=codes,
        motor_state=[0] * 29,
    ))

    assert report["status"] == "fail"
    assert report["motor_errors"] == [{
        "sdk_index": 0,
        "first_code": 0x80000001,
        "first_code_hex": "0x80000001",
        "latest_code": 0,
    }]


def test_probe_does_not_invent_raw_codes_from_an_older_native_build():
    report = _report(_raw(motor_error_samples=1, motor_error_mask=1))

    assert report["motor_errors"][0]["first_code"] is None
    assert report["motor_errors"][0]["latest_code"] is None


def test_probe_report_fails_short_slow_or_stalled_stream():
    report = _report(
        _raw(
            samples=20,
            last_receive_ns=3_000_000_000,
            max_gap_ns=150_000_000,
        )
    )

    assert report["status"] == "fail"
    assert report["checks"]["sample_count"] is False
    assert report["checks"]["rate"] is False
    assert report["checks"]["max_gap"] is False


def test_probe_fails_when_no_lowstate_arrives(monkeypatch):
    ec_native = pytest.importorskip("ec_native")

    class NoMessages:
        def __init__(self, network, dds_domain):
            assert network == "eth0"
            assert dds_domain == 0

        def wait_for_samples(self, samples, timeout):
            assert samples == 10
            assert timeout == 0.1
            return False

        def snapshot(self):
            return _raw(
                samples=0,
                first_receive_ns=0,
                last_receive_ns=0,
                max_gap_ns=0,
                snapshot_consistent=False,
            )

    monkeypatch.setattr(ec_native, "UnitreeStateProbe", NoMessages)
    report = run_probe("eth0", samples=10, timeout=0.1)

    assert report["status"] == "fail"
    assert report["samples"] == 0
    assert report["checks"]["sample_count"] is False
    assert report["checks"]["rate"] is False
    assert report["checks"]["snapshot"] is False


@pytest.mark.parametrize(
    ("network", "kwargs"),
    [
        ("", {}),
        ("eth0", {"samples": 0}),
        ("eth0", {"timeout": 0.0}),
        ("eth0", {"min_rate_hz": float("nan")}),
        ("eth0", {"max_gap_ms": -1.0}),
        ("eth0", {"dds_domain": -1}),
    ],
)
def test_probe_rejects_invalid_options(network, kwargs):
    with pytest.raises(ValueError):
        run_probe(network, **kwargs)


def test_capture_recovers_the_plants_injected_sensor_noise(tmp_path):
    """The probe's raw capture is the hardware side of the noise comparison;
    against the plant it must read back the `--noise-*` half-ranges it injects.

    The probe runs in a child process: the SDK's ChannelFactory is a
    per-process singleton, so a probe opened after another test's plant on a
    different domain would silently stay on that domain and see nothing."""
    pytest.importorskip("ec_native")
    import json
    import subprocess
    import sys

    import numpy as np
    from test_dds_plant import _finish_plant, _spawn_plant
    from test_native_core import _g1_mjcf_path

    from embodied_control.lowlevel.unitree_probe import noise_summary

    report_path = tmp_path / "plant_report.json"
    process = _spawn_plant(
        _g1_mjcf_path(), 20.0, report_path,
        ["--vendor", "--hoist", "--dds-domain", "45", "--noise-seed", "3",
         "--noise-joint-pos", "0.02", "--noise-joint-vel", "0.4", "--noise-base-ang-vel", "0.1"],
    )
    capture = tmp_path / "capture.npz"
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "embodied_control.cli", "lowlevel", "check-unitree",
             "--network", "lo", "--dds-domain", "45", "--samples", "1500", "--timeout", "10",
             "--capture", str(capture), "--json"],
            capture_output=True, text=True, timeout=60,
        )
    finally:
        _finish_plant(process, report_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout[completed.stdout.index("{"):])
    assert report["capture"]["rows"] == 1500 and capture.is_file()
    data = np.load(capture)
    assert data["joint_position"].shape == (1500, 29) and data["quaternion_wxyz"].shape == (1500, 4)
    assert np.all(np.diff(data["receive_ns"].astype(np.int64)) > 0)
    noise = noise_summary(capture)
    assert report["noise"]["rate_hz"] == pytest.approx(500.0, rel=0.2)
    # A hanging robot barely moves at 500 Hz, so the residual is the injection.
    assert noise["joint_position"]["implied_uniform_half_range"] == pytest.approx(0.02, rel=0.2)
    assert noise["joint_velocity"]["implied_uniform_half_range"] == pytest.approx(0.4, rel=0.2)
    assert noise["gyroscope"]["implied_uniform_half_range"] == pytest.approx(0.1, rel=0.25)
