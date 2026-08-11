"""Timing-SLO evaluation for native runtime reports.

Consumes the report dict the native runtimes emit (`--report` on
`ec lowlevel mujoco-native` / the loop bindings' `stats()` +
`tick_durations_ns()`), computes percentiles, and grades them against the
v2 architecture's service-level objectives. Quantities the native layer
exposes only as maxima are graded against max budgets and labeled so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Initial engineering budgets from wiki/tracker-runtime-v2-architecture.md.
# Replace with recorded baselines once R1 measurements stabilize.
SLO_TARGETS: dict[str, float] = {
    "control_tick_compute_p99_ms": 2.0,
    "control_tick_compute_max_ms": 5.0,
    "control_wake_late_max_us": 1000.0,
    "plant_wake_late_max_us": 1000.0,
    "deadline_misses": 0,
    "scheduler_deadlines_missed": 0,
    "response_overruns": 0,
    "damp_ticks": 0,
    "fault": 0,
}


def evaluate_report(
    report: dict[str, Any],
    tick_durations_ns: np.ndarray | None = None,
    *,
    targets: dict[str, float] | None = None,
) -> dict[str, Any]:
    slo = dict(SLO_TARGETS if targets is None else targets)
    control = report.get("control", report)
    measured: dict[str, float] = {
        "control_tick_compute_max_ms": control["tick_ns_max"] / 1e6,
        "control_wake_late_max_us": control["wake_late_ns_max"] / 1e3,
        "plant_wake_late_max_us": control.get("backend_wake_late_ns_max", 0) / 1e3,
        "deadline_misses": control["deadline_misses"]
        + control.get("backend_deadline_misses", 0),
        "scheduler_deadlines_missed": control.get("scheduler_deadlines_missed", 0),
        "response_overruns": control.get("response_overruns", 0),
        "damp_ticks": control.get("damp_ticks", 0),
        "fault": control.get("fault", 0),
    }
    if tick_durations_ns is not None and len(tick_durations_ns):
        ticks_ms = np.asarray(tick_durations_ns, dtype=np.float64) / 1e6
        measured["control_tick_compute_p50_ms"] = float(np.percentile(ticks_ms, 50))
        measured["control_tick_compute_p99_ms"] = float(np.percentile(ticks_ms, 99))
    checks = {
        name: bool(measured[name] <= limit)
        for name, limit in slo.items()
        if name in measured
    }
    unmeasured = sorted(set(slo) - set(measured))
    return {
        "measured": measured,
        "targets": slo,
        "checks": checks,
        "unmeasured": unmeasured,
        "max_based": [
            "control_tick_compute_max_ms",
            "control_wake_late_max_us",
            "plant_wake_late_max_us",
        ],
        "pass": all(checks.values()),
        "realtime_configured": bool(control.get("realtime_configured", False))
        and bool(control.get("backend_realtime_configured", True)),
    }


def certify_report_file(
    report_path: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    report = json.loads(Path(report_path).read_text())
    verdict = evaluate_report(report)
    certificate = {"source_report": str(report_path), **verdict}
    if output_path is not None:
        Path(output_path).write_text(json.dumps(certificate, indent=2))
    return certificate


__all__ = ["SLO_TARGETS", "certify_report_file", "evaluate_report"]
