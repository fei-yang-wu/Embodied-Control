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
