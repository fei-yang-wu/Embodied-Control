from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def _validate_probe_options(
    network: str,
    samples: int,
    timeout: float,
    min_rate_hz: float,
    max_gap_ms: float,
    dds_domain: int,
) -> None:
    if not network.strip():
        raise ValueError("network interface must not be empty")
    if samples <= 0:
        raise ValueError("samples must be positive")
    for name, value in (
        ("timeout", timeout),
        ("min_rate_hz", min_rate_hz),
        ("max_gap_ms", max_gap_ms),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if dds_domain < 0:
        raise ValueError("dds_domain must be non-negative")


def finalize_probe_report(
    raw: dict[str, Any],
    *,
    network: str,
    requested_samples: int,
    min_rate_hz: float,
    max_gap_ms: float,
) -> dict[str, Any]:
    elapsed_ns = max(0, int(raw["last_receive_ns"]) - int(raw["first_receive_ns"]))
    intervals = max(0, int(raw["samples"]) - 1)
    rate_hz = intervals * 1e9 / elapsed_ns if elapsed_ns and intervals else 0.0
    observed_gap_ms = float(raw["max_gap_ns"]) / 1e6
    checks = {
        "sample_count": int(raw["samples"]) >= requested_samples,
        "rate": rate_hz >= min_rate_hz,
        "max_gap": observed_gap_ms <= max_gap_ms,
        "crc": int(raw["crc_errors"]) == 0,
        "finite": int(raw["nonfinite_samples"]) == 0,
        "motors": int(raw["motor_error_samples"]) == 0,
        "tick_progress": (
            int(raw["samples"]) > 1
            and int(raw["last_tick"]) != int(raw["first_tick"])
        ),
        "snapshot": bool(raw["snapshot_consistent"]),
    }
    error_joints = [
        index for index in range(29) if int(raw["motor_error_mask"]) & (1 << index)
    ]
    return {
        "status": "pass" if all(checks.values()) else "fail",
        "network": network,
        "topic": "rt/lowstate",
        "samples": int(raw["samples"]),
        "rate_hz": rate_hz,
        "max_gap_ms": observed_gap_ms,
        "crc_errors": int(raw["crc_errors"]),
        "nonfinite_samples": int(raw["nonfinite_samples"]),
        "motor_error_samples": int(raw["motor_error_samples"]),
        "motor_error_joints": error_joints,
        "duplicate_ticks": int(raw["duplicate_ticks"]),
        "first_tick": int(raw["first_tick"]),
        "last_tick": int(raw["last_tick"]),
        "mode_machine": int(raw["mode_machine"]),
        "writes_enabled": False,
        "checks": checks,
        "state": {
            "joint_position": list(raw["joint_position"]),
            "joint_velocity": list(raw["joint_velocity"]),
            "joint_torque": list(raw["joint_torque"]),
            "quaternion": list(raw["quaternion"]),
            "gyroscope": list(raw["gyroscope"]),
            "accelerometer": list(raw["accelerometer"]),
        },
    }


def run_probe(
    network: str,
    *,
    samples: int = 500,
    timeout: float = 5.0,
    min_rate_hz: float = 100.0,
    max_gap_ms: float = 100.0,
    dds_domain: int = 0,
) -> dict[str, Any]:
    _validate_probe_options(
        network, samples, timeout, min_rate_hz, max_gap_ms, dds_domain
    )
    try:
        import ec_native
    except ImportError as exc:
        raise RuntimeError(
            "Unitree check needs the native environment: pixi run -e native"
        ) from exc
    if not ec_native.WITH_UNITREE:
        raise RuntimeError(
            "ec_native was built without Unitree SDK2; set EC_UNITREE_SDK_ROOT "
            "and run pixi run -e native build-native"
        )
    probe = ec_native.UnitreeStateProbe(network, dds_domain)
    probe.wait_for_samples(samples, timeout)
    return finalize_probe_report(
        probe.snapshot(),
        network=network,
        requested_samples=samples,
        min_rate_hz=min_rate_hz,
        max_gap_ms=max_gap_ms,
    )


def write_report(report: dict[str, Any], output: str) -> None:
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
