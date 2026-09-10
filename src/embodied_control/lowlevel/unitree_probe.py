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
    # Keep reports from older native builds usable, but never invent a zero
    # (healthy) status when that build did not expose the firmware code.
    first_codes = raw.get("first_motor_error_code", [None] * 29)
    motor_states = raw.get("motor_state", [None] * 29)
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
        "motor_errors": [
            {
                "sdk_index": index,
                "first_code": first_codes[index],
                "first_code_hex": (
                    f"0x{first_codes[index]:08x}"
                    if first_codes[index] is not None else None
                ),
                "latest_code": motor_states[index],
            }
            for index in error_joints
        ],
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
            "motor_state": list(motor_states),
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
    capture: str = "",
) -> dict[str, Any]:
    """Read-only link check; `capture` also saves the first `samples` raw rows.

    The capture is what a sensor-noise comparison against the plant needs
    (`noise_summary`): the tracker's telemetry logs positions at 50 Hz only,
    while the plant injects noise on velocity and gyro too.
    """
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
    probe = (
        ec_native.UnitreeStateProbe(network, dds_domain, samples)
        if capture
        else ec_native.UnitreeStateProbe(network, dds_domain)
    )
    probe.wait_for_samples(samples, timeout)
    report = finalize_probe_report(
        probe.snapshot(),
        network=network,
        requested_samples=samples,
        min_rate_hz=min_rate_hz,
        max_gap_ms=max_gap_ms,
    )
    if capture:
        report["capture"] = write_capture(probe.samples(), capture)
        report["noise"] = noise_summary(capture)
    return report


CAPTURE_FIELDS = (
    ("joint_position", 0, 29), ("joint_velocity", 29, 58), ("joint_torque", 58, 87),
    ("quaternion_wxyz", 87, 91), ("gyroscope", 91, 94), ("accelerometer", 94, 97),
)


def write_capture(raw: dict[str, Any], output: str | Path) -> dict[str, Any]:
    """Raw LowState rows from the probe as a named npz."""
    import numpy as np

    values = np.asarray(raw["values"], dtype=np.float32)
    times = np.asarray(raw["receive_ns"], dtype=np.uint64)
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: values[:, lo:hi] for name, lo, hi in CAPTURE_FIELDS}
    np.savez(path, receive_ns=times, tick=np.asarray(raw["tick"], dtype=np.uint32), **arrays)
    seconds = float(times[-1] - times[0]) / 1e9 if len(times) > 1 else 0.0
    return {"path": str(path), "rows": int(len(values)), "seconds": seconds}


def noise_summary(capture: str | Path) -> dict[str, Any]:
    """Implied white-noise std per channel from a raw capture.

    Second-difference residual r[t] = x[t] - (x[t-1] + x[t+1]) / 2: a signal
    smooth at the wire rate barely excites it, white noise of std s excites it
    at s * sqrt(1.5). Comparable with the plant's `--noise-*` half-ranges
    (uniform half-range h has std h / sqrt(3)).
    """
    import numpy as np

    data = np.load(capture)
    out: dict[str, Any] = {"rows": int(len(data["receive_ns"]))}
    if out["rows"] < 8:
        out["error"] = "too few rows"
        return out
    intervals = np.diff(data["receive_ns"].astype(np.float64)) / 1e9
    out["rate_hz"] = float(1.0 / max(np.median(intervals), 1e-9))
    for name, _, _ in CAPTURE_FIELDS:
        x = data[name].astype(np.float64)
        residual = x[1:-1] - 0.5 * (x[:-2] + x[2:])
        per_channel = residual.std(axis=0) / np.sqrt(1.5)
        out[name] = {
            "implied_noise_std": float(per_channel.mean()),
            "implied_noise_std_max": float(per_channel.max()),
            "implied_uniform_half_range": float(per_channel.mean() * np.sqrt(3.0)),
            "step_p99": float(np.percentile(np.abs(np.diff(x, axis=0)), 99)),
        }
    return out


def write_report(report: dict[str, Any], output: str) -> None:
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
