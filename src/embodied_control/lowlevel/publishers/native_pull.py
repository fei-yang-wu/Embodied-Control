"""Non-real-time planner worker for the native request/response mailboxes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
import subprocess
import threading
import time

import numpy as np

_PLANNER_REQUEST_TAG = 10
_CHUNK_RESPONSE_TAG = 2
_LATENT_PLAN_RESPONSE_TAG = 4


class NativeChunkWorker:
    """Run a slow planner away from the native control thread.

    The native runtime publishes the latest 10 x 93 causal history to one
    shared-memory slot. This worker calls the planner and publishes its full
    root_qpos chunk to a second slot. It can run in the constructor process
    or in a separate process with the same two slot names.
    """

    def __init__(
        self,
        request_slot: str,
        response_slot: str,
        request_fn: Callable[[np.ndarray, dict], np.ndarray],
        *,
        hold_steps: int = 10,
        lead_ticks: int = 4,
        state_width: int = 38,
        poll_seconds: float = 0.001,
        rtc_enabled: bool = False,
        create_slots: bool = False,
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "NativeChunkWorker needs the native Pixi environment"
            ) from exc
        self._request = ec_native.ShmCommandSlot(request_slot, create_slots)
        self._response = ec_native.ShmCommandSlot(response_slot, create_slots)
        self._request_fn = request_fn
        self.hold_steps = int(hold_steps)
        self.lead_ticks = int(lead_ticks)
        self.state_width = int(state_width)
        self.poll_seconds = float(poll_seconds)
        self.rtc_enabled = bool(rtc_enabled)
        self.request_ms: list[float] = []
        self.requests = 0
        self.last_error: BaseException | None = None
        self._last_sequence = 0
        self._previous_chunk: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("native chunk worker is already started")
        self._thread = threading.Thread(
            target=self.run, name="native-chunk-worker", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def run(self) -> None:
        while not self._stop.is_set():
            raw = self._request.snapshot(self._last_sequence)
            if raw is None:
                self._stop.wait(self.poll_seconds)
                continue
            sequence, tag, values, _, _ = raw
            if int(sequence) <= self._last_sequence:
                self._stop.wait(self.poll_seconds)
                continue
            if int(tag) != _PLANNER_REQUEST_TAG:
                self.last_error = ValueError(f"unexpected planner request tag {tag}")
                return
            self._last_sequence = int(sequence)
            history = np.asarray(values, dtype=np.float32)
            if history.shape != (930,):
                self.last_error = ValueError(
                    f"planner history must have shape (930,), got {history.shape}"
                )
                return
            context = {
                "prev_chunk": self._previous_chunk if self.rtc_enabled else None,
                "freeze_steps": self.lead_ticks,
                "hold_steps": self.hold_steps,
            }
            started = time.perf_counter()
            try:
                chunk = np.asarray(self._request_fn(history, context), dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != self.state_width:
                    raise ValueError(
                        "planner chunk must have shape [H, "
                        f"{self.state_width}], got {chunk.shape}"
                    )
                if not np.isfinite(chunk).all():
                    raise ValueError("planner chunk contains a non-finite value")
                flat = np.ascontiguousarray(chunk.reshape(-1), dtype=np.float32)
                self._response.publish(
                    int(sequence), _CHUNK_RESPONSE_TAG, flat, time.monotonic()
                )
            except Exception as exc:  # keep the fault outside the RT loop
                self.last_error = exc
                return
            self.request_ms.append((time.perf_counter() - started) * 1000.0)
            self.requests += 1
            self._previous_chunk = chunk.copy()


class NativeLatentPlanWorker:
    """Answer native planner requests with a latent plan.

    The chunk worker (:class:`NativeChunkWorker`) replies with a root_qpos
    window that the tracker-side encoder turns into one latent. A latent head
    predicts the latents itself, so this worker forwards the head's plan
    unchanged: `[slots, z_dim]` published under the latent-plan tag. The
    controller walks the plan slot by slot, one head call per
    `slots * hold_steps` control ticks, and owns the lead-time schedule.

    The plan is NOT re-based on arrival: the controller time-aligns it against
    the tick the request was published at, so slots whose time passed while the
    head was thinking are never replayed.
    """

    def __init__(
        self,
        request_slot: str,
        response_slot: str,
        request_fn: Callable[[np.ndarray, dict], np.ndarray],
        *,
        z_dim: int,
        plan_slots: int,
        hold_steps: int = 1,
        lead_ticks: int = 5,
        poll_seconds: float = 0.001,
        rtc_enabled: bool = False,
        create_slots: bool = False,
    ) -> None:
        try:
            import ec_native
        except ImportError as exc:  # pragma: no cover - optional build
            raise ImportError(
                "NativeLatentPlanWorker needs the native Pixi environment"
            ) from exc
        self._request = ec_native.ShmCommandSlot(request_slot, create_slots)
        self._response = ec_native.ShmCommandSlot(response_slot, create_slots)
        self._request_fn = request_fn
        self.z_dim = int(z_dim)
        self.plan_slots = int(plan_slots)
        self.hold_steps = int(hold_steps)
        self.lead_ticks = int(lead_ticks)
        self.poll_seconds = float(poll_seconds)
        self.rtc_enabled = bool(rtc_enabled)
        if self.z_dim <= 0 or self.plan_slots <= 0 or self.hold_steps <= 0:
            raise ValueError("latent plan geometry must be positive")
        self.request_ms: list[float] = []
        self.requests = 0
        self.last_error: BaseException | None = None
        self._last_sequence = 0
        self._previous_plan: np.ndarray | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("native latent plan worker is already started")
        self._thread = threading.Thread(
            target=self.run, name="native-latent-plan-worker", daemon=True
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def run(self) -> None:
        while not self._stop.is_set():
            raw = self._request.snapshot(self._last_sequence)
            if raw is None:
                self._stop.wait(self.poll_seconds)
                continue
            sequence, tag, values, _, _ = raw
            if int(sequence) <= self._last_sequence:
                self._stop.wait(self.poll_seconds)
                continue
            if int(tag) != _PLANNER_REQUEST_TAG:
                self.last_error = ValueError(f"unexpected planner request tag {tag}")
                return
            self._last_sequence = int(sequence)
            history = np.asarray(values, dtype=np.float32)
            if history.shape != (930,):
                self.last_error = ValueError(
                    f"planner history must have shape (930,), got {history.shape}"
                )
                return
            context = {
                "prev_chunk": self._previous_plan if self.rtc_enabled else None,
                "freeze_steps": self.lead_ticks,
                "hold_steps": self.hold_steps,
            }
            started = time.perf_counter()
            try:
                plan = np.asarray(self._request_fn(history, context), dtype=np.float32)
                if plan.ndim != 2 or plan.shape[1] != self.z_dim:
                    raise ValueError(
                        f"latent plan must have shape [slots, {self.z_dim}], "
                        f"got {plan.shape}"
                    )
                if plan.shape[0] > self.plan_slots:
                    # Never publish more slots than the controller was built
                    # for; the tail would be silently dropped mid-packet.
                    plan = plan[: self.plan_slots]
                if not np.isfinite(plan).all():
                    raise ValueError("latent plan contains a non-finite value")
                flat = np.ascontiguousarray(plan.reshape(-1), dtype=np.float32)
                self._response.publish(
                    int(sequence), _LATENT_PLAN_RESPONSE_TAG, flat, time.monotonic()
                )
            except Exception as exc:  # keep the fault outside the RT loop
                self.last_error = exc
                return
            self.request_ms.append((time.perf_counter() - started) * 1000.0)
            self.requests += 1
            self._previous_plan = plan.copy()


class StdioChunkService:
    """Callable adapter for ``gr00t_chunk_service`` and its Pixi process."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        action_width: int = 38,
        window_frames: int = 10,
        goal: str | None = None,
        goal_file: str | Path | None = None,
    ) -> None:
        # A chunk head predicts root_qpos frames (38 wide); a latent head
        # predicts the commands themselves (z_dim wide). Both speak this
        # protocol, so the expected width is a parameter, not a constant.
        self.action_width = int(action_width)
        self.window_frames = int(window_frames)
        # Per-episode goal switch: the service re-selects its cached language
        # features when a request carries a "goal" key.
        self.goal = goal
        self.goal_file = Path(goal_file) if goal_file else None
        self.head_ms: list[float] = []
        self._process = subprocess.Popen(
            list(command),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        assert self._process.stdout is not None
        ready_line = self._process.stdout.readline()
        if not ready_line:
            self._process.wait(timeout=5)
            raise RuntimeError("GR00T chunk service exited before readiness")
        try:
            ready = json.loads(ready_line)
        except json.JSONDecodeError as exc:
            self._process.terminate()
            self._process.wait(timeout=5)
            raise RuntimeError(
                "GR00T chunk service wrote non-JSON data to protocol stdout: "
                f"{ready_line.rstrip()!r}"
            ) from exc
        if not ready.get("ready"):
            self._process.terminate()
            self._process.wait(timeout=5)
            raise RuntimeError(f"GR00T chunk service did not become ready: {ready}")
        expected = {
            "state_history": 10,
            "state_width": 93,
            "action_width": self.action_width,
            "window_frames": self.window_frames,
        }
        mismatches = {
            key: (ready.get(key), value)
            for key, value in expected.items()
            if ready.get(key) != value
        }
        if mismatches:
            self._process.terminate()
            self._process.wait(timeout=5)
            raise RuntimeError(f"GR00T chunk service contract mismatch: {mismatches}")
        try:
            horizon = int(ready["action_horizon"])
        except (KeyError, TypeError, ValueError):
            horizon = 0
        if horizon < expected["window_frames"]:
            self._process.terminate()
            self._process.wait(timeout=5)
            raise RuntimeError(
                "GR00T chunk service action_horizon must be at least "
                f"{expected['window_frames']}, got {ready.get('action_horizon')}"
            )
        self.ready = ready

    def __call__(self, history: np.ndarray, context: dict) -> np.ndarray:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"GR00T chunk service exited with code {self._process.returncode}"
            )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        previous = context.get("prev_chunk")
        request = {
            "state": np.asarray(history, dtype=np.float32).reshape(-1).tolist(),
            "prev_chunk": (
                None
                if previous is None
                else np.asarray(previous, dtype=np.float32).reshape(-1).tolist()
            ),
            "freeze_steps": int(context.get("freeze_steps", 0)),
            "hold_steps": int(context.get("hold_steps", 10)),
        }
        goal = context.get("goal")
        if goal is None and self.goal_file is not None:
            try:
                goal = self.goal_file.read_text().strip()
            except OSError as exc:
                raise RuntimeError(
                    f"cannot read planner goal file {self.goal_file}: {exc}"
                ) from exc
            if not goal:
                raise RuntimeError(f"planner goal file {self.goal_file} is empty")
            self.goal = goal
        if goal is None:
            goal = self.goal
        if goal is not None:
            request["goal"] = str(goal)
        self._process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError("GR00T chunk service closed its output")
        response = json.loads(line)
        if "chunk" not in response:
            raise RuntimeError(f"GR00T response has no chunk: {response}")
        if response.get("head_ms") is not None:
            head_ms = float(response["head_ms"])
            if not np.isfinite(head_ms):
                raise RuntimeError(f"GR00T response has invalid head_ms: {head_ms}")
            self.head_ms.append(head_ms)
        horizon = int(self.ready["action_horizon"])
        chunk = np.asarray(response["chunk"], dtype=np.float32)
        if chunk.size != horizon * self.action_width:
            raise RuntimeError(
                f"GR00T response chunk has {chunk.size} values; expected "
                f"{horizon * self.action_width}"
            )
        return chunk.reshape(horizon, self.action_width)

    def close(self) -> None:
        if self._process.poll() is not None:
            return
        assert self._process.stdin is not None
        try:
            self._process.stdin.write('{"stop": true}\n')
            self._process.stdin.flush()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)


__all__ = ["NativeChunkWorker", "StdioChunkService"]
