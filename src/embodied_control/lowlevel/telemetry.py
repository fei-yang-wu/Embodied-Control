"""Non-blocking telemetry for the native runtimes.

The control thread never logs: it writes fixed-size per-tick records into
preallocated C++ arrays (joint positions, anchor pose, reference frame,
base height, tick duration — a memcpy on the hot path). This recorder is
the off-path consumer, following the house logger conventions: an
`EcLogger` is constructor-injected (never a global, `EcLogger.null()` when
absent), and a normal-priority drain thread samples the runtime's atomic
stats at a low rate for live visibility. `collect()`/`save()` pull the full
arrays once the run is over and write `telemetry.npz` plus a JSON summary.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import numpy as np

from embodied_control.logging import EcLogger

JOINT_COUNT = 29
ANCHOR_POSE_WIDTH = 7


class TelemetryRecorder:
    def __init__(
        self,
        runtime,
        *,
        logger: EcLogger | None = None,
        sample_hz: float = 1.0,
    ) -> None:
        self.runtime = runtime
        self.logger = logger or EcLogger.null()
        self.sample_hz = float(sample_hz)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self.sample_hz <= 0:
            return
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _drain(self) -> None:
        period = 1.0 / self.sample_hz
        while not self._stop.wait(period):
            try:
                stats = self.runtime.stats()
            except Exception:
                return
            # `base_height` throws while the loop is running; the per-tick log does not.
            heights = self.runtime.base_heights()
            self.logger.event(
                "telemetry.sample",
                phase="rollout",
                ticks=int(stats.get("ticks", 0)),
                mode=int(stats.get("mode", -1)),
                fault=int(stats.get("fault", 0)),
                deadline_misses=int(stats.get("deadline_misses", 0)),
                wake_late_ns_max=int(stats.get("wake_late_ns_max", 0)),
                base_height=float(heights[-1]) if len(heights) else float("nan"),
            )
            # The native runtime exposes `running` as a property, the Python
            # wrappers as a method; accept both.
            running = self.runtime.running
            if not (running() if callable(running) else running):
                return

    def collect(self) -> dict:
        """Pull the complete per-tick record after (or during) a run."""
        joint = np.asarray(self.runtime.joint_position_log(), dtype=np.float32)
        anchor = np.asarray(self.runtime.anchor_pose_log(), dtype=np.float32)
        record = {
            "joint_position_log": joint.reshape(-1, JOINT_COUNT),
            "anchor_pose_log": anchor.reshape(-1, ANCHOR_POSE_WIDTH),
            "reference_frames": np.asarray(
                self.runtime.reference_frames(), dtype=np.int32
            ),
            "base_heights": np.asarray(self.runtime.base_heights(), dtype=np.float32),
            "reference_joint_mae": np.asarray(
                self.runtime.reference_joint_mae(), dtype=np.float32
            ),
            "tick_durations_ns": np.asarray(
                self.runtime.tick_durations_ns(), dtype=np.int64
            ),
        }
        record["stats"] = dict(self.runtime.stats())
        return record

    def save(self, output_dir: str | Path, record: dict | None = None) -> Path:
        record = record if record is not None else self.collect()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays = {k: v for k, v in record.items() if isinstance(v, np.ndarray)}
        npz_path = output_dir / "telemetry.npz"
        np.savez_compressed(npz_path, **arrays)
        summary = {
            "ticks": int(record["stats"].get("ticks", 0)),
            "stats": {k: int(v) if isinstance(v, (bool, int)) else v
                      for k, v in record["stats"].items()},
            "arrays": {k: list(v.shape) for k, v in arrays.items()},
            "written_monotonic": time.monotonic(),
        }
        (output_dir / "telemetry_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        self.logger.event(
            "telemetry.saved", phase="finalize", path=str(npz_path),
            ticks=summary["ticks"],
        )
        return npz_path


__all__ = ["TelemetryRecorder"]
