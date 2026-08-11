"""Asynchronous lead-time chunk client (the pull topology's tracker side).

The VLA service owns inference; this client owns the schedule. `lead_ticks`
before the current hold expires it snapshots the causal state history and
hands a request to a single worker thread; the control tick never blocks. At
expiry the reply is swapped in; a late reply is a **deadline miss**: the
stale command is re-armed for another hold (phase restarts, staying in the
training distribution) and the late chunk, when it lands, is used at the
next expiry. Requests never overlap — if one is still in flight when the
next would be issued, the issue is skipped and counted as an overrun.

The service contract is a callable `request_fn(state_history, context) ->
chunk [H, frame_dim]` where context carries `prev_chunk` and `freeze_steps`
for RTC. The chunk is the head's denormalized prediction starting one frame
after the snapshot. At swap, the client measures how many control ticks have
elapsed since that request. It consumes
`chunk[elapsed : elapsed + window_frames]`. Thus, a cold start uses offset 0,
an on-time reply normally uses `lead_ticks`, and a late reply does not replay
frames whose time has passed.
"""

from __future__ import annotations

from collections.abc import Callable
import math
import threading
import time

import numpy as np

from embodied_control.lowlevel.bundle import CommandContract
from embodied_control.lowlevel.command_buffer import CommandBuffer
from embodied_control.lowlevel.contracts import CommandPacket
from embodied_control.lowlevel.engine.base import Engine


class AsyncChunkPullClient:
    def __init__(
        self,
        buffer: CommandBuffer,
        encoder: Engine,
        command: CommandContract,
        request_fn: Callable[[np.ndarray, dict], np.ndarray],
        *,
        hold_steps: int | None = None,
        lead_ticks: int = 4,
        window_frames: int | None = None,
    ):
        if command.z_dim is None or command.state_dim is None:
            raise ValueError("command contract must define z_dim and state_dim")
        self.buffer = buffer
        self.encoder = encoder
        self.command = command
        self.request_fn = request_fn
        self.hold_steps = int(
            hold_steps if hold_steps is not None else command.hold_steps
        )
        self.lead_ticks = int(lead_ticks)
        if not 0 < self.lead_ticks < self.hold_steps:
            raise ValueError("lead_ticks must be in (0, hold_steps)")
        self.window_frames = int(
            window_frames
            if window_frames is not None
            else (command.window_steps or 9) + 1
        )
        self.deadline_misses = 0
        self.request_overruns = 0
        self.request_ms: list[float] = []
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._request_ready = threading.Event()
        self._result_ready = threading.Event()
        self._lock = threading.Lock()
        self._pending: tuple[np.ndarray, dict, int] | None = None
        self._result: tuple[np.ndarray, int] | None = None
        self._busy = False
        self._stop = False
        self._z: np.ndarray | None = None
        self._prev_chunk: np.ndarray | None = None
        self._steps_remaining = 0
        self._sequence = 0
        self.exhausted = False
        self._worker.start()

    def close(self) -> None:
        self._stop = True
        self._request_ready.set()
        self._worker.join(timeout=5)

    def _worker_loop(self) -> None:
        while True:
            self._request_ready.wait()
            self._request_ready.clear()
            if self._stop:
                return
            with self._lock:
                pending = self._pending
                self._pending = None
                self._busy = pending is not None
            if pending is None:
                continue
            history, context, request_tick = pending
            started = time.perf_counter()
            chunk = np.asarray(self.request_fn(history, context), dtype=np.float32)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            with self._lock:
                self._result = (chunk, request_tick)
                self.request_ms.append(elapsed_ms)
                self._busy = False
            self._result_ready.set()

    def _issue_request(self, history: np.ndarray, tick: int) -> None:
        with self._lock:
            if self._pending is not None or self._busy:
                self.request_overruns += 1
                return
            context = {
                "prev_chunk": None
                if self._prev_chunk is None
                else self._prev_chunk.copy(),
                "freeze_steps": self.lead_ticks,
                "hold_steps": self.hold_steps,
            }
            self._pending = (np.array(history, copy=True), context, int(tick))
        self._request_ready.set()

    def _take_result(self) -> tuple[np.ndarray, int] | None:
        if not self._result_ready.is_set():
            return None
        with self._lock:
            chunk = self._result
            self._result = None
        self._result_ready.clear()
        return chunk

    def _swap_in(self, chunk: np.ndarray, elapsed_ticks: int) -> None:
        start = max(int(elapsed_ticks), 0)
        window = chunk[start : start + self.window_frames]
        if window.shape[0] == 0:
            raise ValueError(
                f"planner chunk has no frame at elapsed offset {start}; "
                f"horizon is {chunk.shape[0]}"
            )
        if window.shape[0] < self.window_frames:
            pad = np.repeat(window[-1:], self.window_frames - window.shape[0], axis=0)
            window = np.concatenate([window, pad], axis=0)
        flat = window.reshape(-1).astype(np.float32)
        self._z = np.asarray(self.encoder.infer(flat), dtype=np.float32)
        self._prev_chunk = chunk
        self._steps_remaining = self.hold_steps

    def tick(self, tick: int, stamp: float, state_history: np.ndarray) -> None:
        """Advance one control tick. `state_history` is the flat 10x93 history."""
        if self._z is None:
            # Cold start: block once for the first chunk (WAIT-state behavior).
            self._issue_request(state_history, tick)
            self._result_ready.wait()
            first = self._take_result()
            if first is None:
                raise RuntimeError("chunk service produced no first chunk")
            chunk, request_tick = first
            self._swap_in(chunk, tick - request_tick)
        elif self._steps_remaining == self.lead_ticks:
            # A late reply that already landed but was never consumed swaps at
            # the coming expiry instead of being replaced by a fresh request.
            if not self._result_ready.is_set():
                self._issue_request(state_history, tick)
        elif self._steps_remaining <= 0:
            result = self._take_result()
            if result is None:
                self.deadline_misses += 1
                self._steps_remaining = self.hold_steps  # re-arm stale z
            else:
                chunk, request_tick = result
                self._swap_in(chunk, tick - request_tick)
        assert self._z is not None
        phase = (self.hold_steps - self._steps_remaining) / float(self.hold_steps)
        if self.command.phase_mode == "sin_cos":
            values = np.concatenate(
                [
                    self._z,
                    np.array(
                        [math.sin(2 * math.pi * phase), math.cos(2 * math.pi * phase)],
                        dtype=np.float32,
                    ),
                ]
            )
        else:
            values = self._z
        self._sequence += 1
        self.buffer.publish(
            CommandPacket(
                interface="latent",
                values=values.astype(np.float32),
                sequence=self._sequence,
                stamp=stamp,
                terms={"latent_command": values.astype(np.float32)},
                metadata={"steps_remaining": self._steps_remaining},
            )
        )
        self._steps_remaining -= 1


__all__ = ["AsyncChunkPullClient"]
