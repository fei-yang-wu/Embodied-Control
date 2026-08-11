"""Reference-streaming worker for the native oracle command source.

The worker does disk and array work outside the control deadline. It sends raw
world-frame records. The native 50 Hz thread owns live-pelvis re-expression and
ONNX encoding at command consume time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import threading
import time

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.reference import ReferenceArrays

ORACLE_REQUEST_TAG = 11
RAW_REFERENCE_RESPONSE_TAG = 3
RAW_REFERENCE_WIDTH = 36
REFERENCE_HEADER_WIDTH = 3


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NativeOracleWorker:
    """Stream one validated reference motion through native mailboxes."""

    def __init__(
        self,
        request_slot: str,
        response_slot: str,
        bundle: PolicyBundle,
        reference_root: str | Path,
        motion: str,
        *,
        start_frame: int = 0,
        horizon: int = 30,
        poll_seconds: float = 0.001,
        create_slots: bool = False,
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "NativeOracleWorker needs the native Pixi environment"
            ) from exc
        command = bundle.manifest.command
        if (
            command.encoder_state_interface != "root_qpos"
            or command.macro_anchor_mode != "robot"
            or command.macro_frame_stride != 1
            or command.state_dim != 38
        ):
            raise ValueError(
                "oracle worker needs root_qpos, robot anchor, stride 1, width 38"
            )
        if horizon < int((command.window_steps or 9) + 1):
            raise ValueError("oracle horizon is shorter than the encoder window")
        if start_frame < 0:
            raise ValueError("start_frame must be non-negative")

        arrays = ReferenceArrays(reference_root)
        if arrays.joint_names != list(bundle.manifest.action.isaac_joint_names):
            raise ValueError("reference and bundle Isaac joint orders differ")
        selected = arrays.motion(motion)
        if start_frame >= selected.length:
            raise ValueError(
                f"start_frame {start_frame} is outside motion length {selected.length}"
            )

        self._request = ec_native.ShmCommandSlot(request_slot, create_slots)
        self._response_slot = ec_native.ShmCommandSlot(response_slot, create_slots)
        self._motion = selected
        self.start_frame = int(start_frame)
        self.horizon = int(horizon)
        self.poll_seconds = float(poll_seconds)
        self.requests = 0
        self.padded_frames = 0
        self.request_ms: list[float] = []
        self.last_error: BaseException | None = None
        self._last_sequence = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        manifest_path = Path(reference_root) / "reference_arrays_manifest.json"
        self.provenance = {
            "source": "oracle",
            "bundle": str(bundle.root.resolve()),
            "bundle_manifest_sha256": _sha256(bundle.root / "manifest.json"),
            "encoder_onnx_sha256": bundle.manifest.files.get("encoder.onnx"),
            "reference_root": str(Path(reference_root).resolve()),
            "reference_manifest_sha256": _sha256(manifest_path),
            "motion": selected.name,
            "motion_length": selected.length,
            "start_frame": self.start_frame,
            "horizon": self.horizon,
            "raw_reference_width": RAW_REFERENCE_WIDTH,
        }

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("native oracle worker is already started")
        self._thread = threading.Thread(
            target=self.run, name="native-oracle-worker", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    @staticmethod
    def _nonnegative_integer(value: float, name: str) -> int:
        if not np.isfinite(value):
            raise ValueError(f"{name} must be an exact non-negative integer")
        result = int(value)
        if result < 0 or float(result) != float(value):
            raise ValueError(f"{name} must be an exact non-negative integer")
        return result

    def _build_response(self, generation: int, reference_frame: int) -> np.ndarray:
        cursor = self.start_frame + reference_frame
        remaining = max(0, self._motion.length - cursor)
        valid_frames = min(self.horizon, remaining)
        if valid_frames == 0:
            raise ValueError(
                f"reference exhausted at frame {cursor}; length={self._motion.length}"
            )
        indices = np.minimum(
            cursor + np.arange(self.horizon, dtype=np.int64),
            self._motion.length - 1,
        )
        raw = np.empty((self.horizon, RAW_REFERENCE_WIDTH), dtype=np.float32)
        raw[:, :29] = self._motion.joint_qpos[indices]
        raw[:, 29:32] = self._motion.anchor_pos_w[indices]
        raw[:, 32:36] = self._motion.anchor_quat_w[indices]
        response = np.empty(
            REFERENCE_HEADER_WIDTH + self.horizon * RAW_REFERENCE_WIDTH,
            dtype=np.float32,
        )
        response[:REFERENCE_HEADER_WIDTH] = [
            float(generation),
            float(reference_frame),
            float(valid_frames),
        ]
        response[REFERENCE_HEADER_WIDTH:] = raw.reshape(-1)
        self.padded_frames += self.horizon - valid_frames
        return response

    def run(self) -> None:
        while not self._stop.is_set():
            raw_request = self._request.snapshot(self._last_sequence)
            if raw_request is None:
                self._stop.wait(self.poll_seconds)
                continue
            sequence, tag, values, _, _ = raw_request
            if int(sequence) <= self._last_sequence:
                self._stop.wait(self.poll_seconds)
                continue
            self._last_sequence = int(sequence)
            started = time.perf_counter()
            try:
                if int(tag) != ORACLE_REQUEST_TAG:
                    raise ValueError(f"unexpected oracle request tag {tag}")
                request = np.asarray(values, dtype=np.float32)
                if request.shape != (2,):
                    raise ValueError(
                        f"oracle request must have shape (2,), got {request.shape}"
                    )
                generation = self._nonnegative_integer(request[0], "generation")
                reference_frame = self._nonnegative_integer(
                    request[1], "reference_frame"
                )
                response = self._build_response(generation, reference_frame)
                self._response_slot.publish(
                    int(sequence),
                    RAW_REFERENCE_RESPONSE_TAG,
                    response,
                    time.monotonic(),
                )
            except Exception as exc:  # keep disk/data faults outside the RT loop
                self.last_error = exc
                return
            self.requests += 1
            self.request_ms.append((time.perf_counter() - started) * 1000.0)


__all__ = [
    "NativeOracleWorker",
    "ORACLE_REQUEST_TAG",
    "RAW_REFERENCE_RESPONSE_TAG",
]
