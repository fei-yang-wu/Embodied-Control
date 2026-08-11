"""Bridge a line-JSON chunk service onto the native planner mailboxes.

The native control loop publishes planner requests (tag 10, the flat 10x93
causal history) into its request slot and expects the paired response (same
sequence, tag 2) in its response slot. This relay owns a chunk service as a
subprocess speaking the stdio protocol (`{"state": [...]}\n` ->
`{"chunk": [...], "head_ms": ...}\n`, non-JSON lines ignored) and forwards
between the two, staying entirely off the control deadline.

The service's chunk starts one frame after the request state. The native
loop slices the consumed window by the measured elapsed ticks, so the relay
publishes the leading `published_frames` frames (default 20 = 760 floats,
within the slot's 1024-float capacity) — enough to absorb a full hold of
reply lateness.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time

import numpy as np

PLANNER_REQUEST_TAG = 10
CHUNK_RESPONSE_TAG = 2
PLANNER_HISTORY_WIDTH = 930


class StdioChunkRelay:
    def __init__(
        self,
        request_slot: str,
        response_slot: str,
        service_command: list[str],
        *,
        cwd: str | None = None,
        frame_width: int = 38,
        published_frames: int = 20,
        create_slots: bool = False,
        ready_timeout_s: float = 600.0,
        poll_seconds: float = 0.0005,
        service_process: subprocess.Popen | None = None,
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - env without the build
            raise ImportError("the relay needs the ec_native extension") from exc
        self._request = ec_native.ShmCommandSlot(request_slot, create_slots)
        self._response = ec_native.ShmCommandSlot(response_slot, create_slots)
        self.frame_width = int(frame_width)
        self.published_frames = int(published_frames)
        self.poll_seconds = float(poll_seconds)
        self.requests_relayed = 0
        self.head_ms: list[float] = []
        self._stop = threading.Event()
        self._owns_service = service_process is None
        if service_process is None:
            self._service = subprocess.Popen(
                service_command, cwd=cwd, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True, bufsize=1,
            )
            self.ready = self._read_json(timeout_s=ready_timeout_s)
            if not self.ready.get("ready"):
                raise RuntimeError(f"chunk service did not become ready: {self.ready}")
        else:
            self._service = service_process
            self.ready = {"ready": True, "shared": True}
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _read_json(self, timeout_s: float) -> dict:
        deadline = time.monotonic() + timeout_s
        assert self._service.stdout is not None
        while time.monotonic() < deadline:
            line = self._service.stdout.readline()
            if not line:
                raise RuntimeError("chunk service exited")
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
        raise TimeoutError("chunk service produced no JSON before the timeout")

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10)
        if not self._owns_service:
            return
        if self._service.stdin is not None:
            try:
                self._service.stdin.write(json.dumps({"stop": True}) + "\n")
                self._service.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        self._service.wait(timeout=30)

    def _run(self) -> None:
        last_sequence = 0
        assert self._service.stdin is not None
        while not self._stop.is_set():
            raw = self._request.snapshot()
            if raw is None:
                time.sleep(self.poll_seconds)
                continue
            sequence, tag, values, _recv, _sender = raw
            if sequence == last_sequence or tag != PLANNER_REQUEST_TAG:
                time.sleep(self.poll_seconds)
                continue
            last_sequence = sequence
            state = np.asarray(values, dtype=np.float32)
            if state.shape != (PLANNER_HISTORY_WIDTH,):
                continue
            self._service.stdin.write(
                json.dumps({"state": [float(v) for v in state]}) + "\n"
            )
            reply = self._read_json(timeout_s=60.0)
            chunk = np.asarray(reply["chunk"], dtype=np.float32).reshape(
                -1, self.frame_width
            )
            payload = np.ascontiguousarray(
                chunk[: self.published_frames].reshape(-1)
            )
            self._response.publish(
                int(sequence), CHUNK_RESPONSE_TAG, payload, time.monotonic()
            )
            self.requests_relayed += 1
            if "head_ms" in reply:
                self.head_ms.append(float(reply["head_ms"]))


__all__ = ["StdioChunkRelay"]
