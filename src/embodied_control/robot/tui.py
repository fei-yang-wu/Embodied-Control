"""Full-screen operator display for the lifecycle console.

The ladder on the left, the writer's numbers on the right, the keys along
the bottom, the last gate results underneath: one glance says where the
robot is and what the next key does. Rendering is a pure function of a
lifecycle snapshot so it is tested without a terminal; curses only paints.

Two rules the layout serves. SPACE reaches the writer before anything else:
it bypasses the lifecycle lock (a gate may hold it for its whole timeout) and
stores DAMP straight into the writer. And a key never blocks the screen:
actions run on one worker thread, one at a time, while the display keeps
refreshing from a lock-free snapshot.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable

from embodied_control.console import KEY_SPACE, ConsoleQuit, KeyBinding
from embodied_control.robot.lifecycle import (
    LADDER,
    WRITER_MODE_NAMES,
    Lifecycle,
    LifecycleState,
)

DISPLAY_ORDER: tuple[LifecycleState, ...] = tuple(LADDER[1:]) + (
    LifecycleState.BLEND_IN,
    LifecycleState.RUNNING,
    LifecycleState.HOLD,
    LifecycleState.DAMP,
    LifecycleState.RELEASED,
    LifecycleState.VENDOR_RESTORED,
    LifecycleState.VENDOR_STAND,
    LifecycleState.STANDING,
)

HOIST_MODE_NAMES = {0: "slack", 1: "hoisted", 2: "lowered"}


def _fit(text: str, width: int) -> str:
    return text[:width].ljust(width)


def progress_bar(fraction: float, width: int = 30) -> str:
    fraction = 0.0 if fraction != fraction else max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {fraction * 100:3.0f}%"


def comm_lines(snapshot: dict, rates: dict | None) -> list[str]:
    """Link health: the robot's state stream, our command stream, the planner."""
    ws = snapshot.get("writer", {})
    st = snapshot.get("control", {})
    rates = rates or {}
    state_hz = rates.get("state_hz")
    lowcmd_hz = rates.get("publish_hz")
    planner_hz = rates.get("planner_hz")
    gap_ms = float(ws.get("state_gap_ns_max", 0)) / 1.0e6
    age = st.get("command_age_ms", -1.0)
    age_text = "-" if age is None or float(age) < 0 else f"{float(age):.0f} ms"
    lowstate = f"lowstate {state_hz:5.0f} Hz" if state_hz is not None else "lowstate    -- Hz"
    lowcmd = f"lowcmd {lowcmd_hz:5.0f} Hz" if lowcmd_hz is not None else "lowcmd    -- Hz"
    planner = f"planner {planner_hz:4.1f} Hz" if planner_hz is not None else "planner   -- Hz"
    ok_state = state_hz is None or state_hz > 400.0
    ok_cmd = int(ws.get("publish_failures", 0)) == 0 and int(ws.get("crc_errors", 0)) == 0
    return [
        f" COMM   {lowstate}  gap max {gap_ms:5.1f} ms  crc {ws.get('crc_errors', 0)}  {'ok' if ok_state and ok_cmd else 'CHECK'}"
        f"   |  {lowcmd}  fail {ws.get('publish_failures', 0)}"
        f"   |  {planner}  reply age {age_text}  stale {st.get('stale_responses', 0)}  overrun {st.get('response_overruns', 0)}",
    ]


def reference_line(snapshot: dict) -> str | None:
    session = snapshot.get("session")
    if not session:
        return None
    if session.get("mode") != "oracle":
        return " REFERENCE  planner-driven (no reference trajectory)"
    length = int(session.get("motion_length", 0))
    start = int(session.get("start_frame", 0))
    total = max(1, length - start)
    played = int(snapshot.get("control", {}).get("reference_ticks", 0))
    played = max(0, min(total, played))
    return (
        f" REFERENCE  {progress_bar(played / total)}  frame {start + played}/{length}"
        f"  ({played}/{total} played)"
    )


def render(
    snapshot: dict,
    bindings: list[KeyBinding],
    notes: list[str],
    *,
    busy: str = "",
    width: int = 100,
    height: int = 36,
    rates: dict | None = None,
) -> list[str]:
    """Rows of exactly `width` characters, at most `height` of them."""
    width = max(60, width)
    state = snapshot.get("state", "?")
    ws = snapshot.get("writer", {})
    st = snapshot.get("control", {})
    hoist = snapshot.get("hoist")
    writer_mode = WRITER_MODE_NAMES.get(int(ws.get("mode", -1)), "?")
    vendor = "released" if ws.get("vendor_released") else "owns joints"
    if not snapshot.get("vendor_name"):
        vendor = "unknown"
    rows: list[str] = []
    title = f" G1 LIFECYCLE   episode {snapshot.get('episode', 0)}   state: {state}"
    right = f"vendor: {vendor}   writer: {writer_mode} "
    rows.append(_fit(title.ljust(width - len(right)) + right, width))
    if snapshot.get("fault_reason"):
        rows.append(_fit(f" !! FAULT: {snapshot['fault_reason']}", width))
    rows.append("-" * width)
    session = snapshot.get("session")
    if session:
        motion = session.get("motion") or "default stance"
        index = session.get("motion_index", 0)
        rows.append(_fit(
            f" MODE {session.get('mode', '?'):<7} MOTION {motion} ({index}/{session.get('catalog_size', 0)})"
            f"   START FRAME {session.get('start_frame', 0)}/{session.get('motion_length', 0)}"
            f"   PLANNER {session.get('planner', '?')}"
            f"   TRACKER {'built' if session.get('built') else 'REBUILD (r)'}",
            width,
        ))
        reference = reference_line(snapshot)
        if reference:
            rows.append(_fit(reference, width))
        for entry in session.get("episodes", []):
            mpjpe = f"   MPJPE_l {entry['mpjpe_l_mm']:.1f} mm" if "mpjpe_l_mm" in entry else ""
            rows.append(_fit(
                f"   ep{entry['episode']:>3} {entry['mode']} {entry['motion'] or 'default'}@{entry['start_frame']}"
                f"   ticks {entry['ticks']}   frames {entry['first_frame']}-{entry['last_frame']}"
                f"   joint MAE {entry['joint_mae_mean_rad']:.3f} rad (p95 {entry['joint_mae_p95_rad']:.3f})"
                f"{mpjpe}   faults {entry['runtime_fault']}/{entry['hardware_faults']}/{entry['watchdog_faults']}",
                width,
            ))
        rows.append("-" * width)
    for line in comm_lines(snapshot, rates):
        rows.append(_fit(line, width))
    rows.append("-" * width)

    left_width = 34
    ladder = [" LADDER"]
    try:
        current = DISPLAY_ORDER.index(LifecycleState(state))
    except ValueError:
        current = -1
    for index, entry in enumerate(DISPLAY_ORDER):
        if index < current:
            mark = "[x]"
        elif index == current:
            mark = "[>]"
        else:
            mark = "[ ]"
        ladder.append(f"  {mark} {entry}")

    live = [" LIVE"]
    hw = ws.get("hardware_faults", 0)
    wd = ws.get("watchdog_faults", 0)
    reason = ws.get("state_fault_reason", 0)
    live += [
        f"  publishes      {ws.get('publishes', 0):>8}   failures {ws.get('publish_failures', 0)}",
        f"  gate           {'open' if ws.get('gate_open') else 'closed':>8}   crc errors {ws.get('crc_errors', 0)}",
        f"  control ticks  {st.get('control_ticks', 0):>8}   reference {st.get('reference_ticks', 0)}",
        f"  planner        {st.get('planner_responses', 0):>8} replies   runtime fault {st.get('fault', 0)}",
        f"  faults hw/wd   {hw:>4}/{wd:<4}   reason {reason}",
        f"  joint speed    {float(ws.get('joint_speed_max', 0.0)):>8.3f} rad/s",
        f"  tracking err   {float(ws.get('tracking_error_max', 0.0)):>8.3f} rad",
        f"  ramp err       {float(ws.get('ramp_error_max', 0.0)):>8.3f} rad (joint {ws.get('ramp_error_joint', '-')})",
        f"  first action   {float(ws.get('command_target_error_max', 0.0)):>8.3f} rad",
        f"  blend left     {ws.get('blend_ticks_remaining', 0):>8} ticks",
        f"  acks           hoisted {'yes' if snapshot.get('hoisted_ack') else 'no ':<3}  lowered {'yes' if snapshot.get('lowered_ack') else 'no'}",
    ]
    if hoist:
        live.append(
            f"  plant hoist    {HOIST_MODE_NAMES.get(int(hoist.get('hoist_mode', -1)), '?'):>8}"
            f"   vendor owns {'yes' if hoist.get('owned') else 'no'}  fsm {hoist.get('fsm_id', '-')}"
        )
    for index in range(max(len(ladder), len(live))):
        left = ladder[index] if index < len(ladder) else ""
        right_text = live[index] if index < len(live) else ""
        rows.append(_fit(_fit(left, left_width) + right_text, width))

    rows.append("-" * width)
    legend: list[str] = []
    line = " KEYS "
    for binding in bindings:
        key = "SPACE" if binding.key == KEY_SPACE else binding.key
        chunk = f"{key} {binding.label.split(':')[0].split(' (')[0]}   "
        if len(line) + len(chunk) > width:
            legend.append(_fit(line, width))
            line = "      " + chunk
        else:
            line += chunk
    legend.append(_fit(line, width))
    rows += legend
    rows.append("-" * width)
    if busy:
        rows.append(_fit(f" BUSY  {busy} ...   (SPACE still damps)", width))
    else:
        last_ok = snapshot.get("last_ok")
        tag = "" if last_ok is None else ("ok    " if last_ok else "FAIL  ")
        rows.append(_fit(f" LAST  {tag}{snapshot.get('last_detail', '')}", width))
    rows.append(_fit(" LOG", width))
    remaining = max(0, height - len(rows))
    for note in list(notes)[-remaining:]:
        rows.append(_fit(f"  {note}", width))
    return rows[:height]


class LifecycleTui:
    """`model` is a Lifecycle or an ExperimentSession: anything with
    `snapshot()` and `emergency_damp()`."""

    def __init__(
        self,
        lifecycle,
        bindings: list[KeyBinding],
        *,
        refresh_seconds: float = 0.1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.lifecycle = lifecycle
        self.bindings = {binding.key: binding for binding in bindings}
        self.ordered = list(bindings)
        self.refresh_seconds = refresh_seconds
        self.notes: deque[str] = deque(maxlen=200)
        self._queue: queue.Queue[KeyBinding | None] = queue.Queue()
        self._busy = ""
        self._quit = threading.Event()
        self._clock = clock
        self._t0 = clock()
        self._worker = threading.Thread(target=self._work, daemon=True)
        self._last_sample: tuple[float, dict] | None = None
        self._rates: dict = {}

    # -- model ---------------------------------------------------------

    def note(self, message: str) -> None:
        self.notes.append(f"{self._clock() - self._t0:7.1f}s  {message}")

    def handle(self, key: str) -> bool:
        """Dispatch one key. Returns False when the console should exit."""
        if key == KEY_SPACE:
            # Straight to the writer, ahead of whatever gate holds the lock.
            self.lifecycle.emergency_damp()
            self.note("SPACE: damp stored in the writer")
            self._queue.put(self.bindings[KEY_SPACE])
            return True
        if key == "q":
            if self._busy:
                self.note("busy: wait for the action to finish, or SPACE")
                return True
            return False
        binding = self.bindings.get(key)
        if binding is None:
            return True
        if self._busy:
            self.note(f"busy with '{self._busy}'; '{key}' ignored")
            return True
        self._queue.put(binding)
        return True

    def _work(self) -> None:
        while not self._quit.is_set():
            try:
                binding = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if binding is None:
                return
            self._busy = binding.label
            try:
                binding.action()
            except ConsoleQuit:
                pass
            except Exception as exc:  # the console outlives any one refusal
                self.note(f"!! {type(exc).__name__}: {exc}")
            finally:
                self._busy = ""

    def rates(self, snapshot: dict) -> dict:
        """Per-second rates from counter deltas, refreshed every 0.5 s."""
        now = self._clock()
        counters = {
            "state_frames": int(snapshot.get("writer", {}).get("state_frames", 0)),
            "publishes": int(snapshot.get("writer", {}).get("publishes", 0)),
            "planner_responses": int(snapshot.get("control", {}).get("planner_responses", 0)),
        }
        if self._last_sample is None:
            self._last_sample = (now, counters)
            return self._rates
        then, previous = self._last_sample
        dt = now - then
        if dt >= 0.5:
            self._rates = {
                "state_hz": (counters["state_frames"] - previous["state_frames"]) / dt,
                "publish_hz": (counters["publishes"] - previous["publishes"]) / dt,
                "planner_hz": (counters["planner_responses"] - previous["planner_responses"]) / dt,
            }
            self._last_sample = (now, counters)
        return self._rates

    def frame(self, width: int, height: int) -> list[str]:
        snapshot = self.lifecycle.snapshot()
        return render(
            snapshot,
            self.ordered,
            list(self.notes),
            busy=self._busy,
            width=width,
            height=height,
            rates=self.rates(snapshot),
        )

    # -- terminal ------------------------------------------------------

    def run(self) -> int:
        import curses

        self._worker.start()
        try:
            curses.wrapper(self._loop)
        finally:
            self._quit.set()
            self._queue.put(None)
            self._worker.join(timeout=2.0)
        return 0

    def _loop(self, screen) -> None:
        import curses

        curses.curs_set(0)
        screen.nodelay(True)
        screen.timeout(int(self.refresh_seconds * 1000))
        while True:
            height, width = screen.getmaxyx()
            for row, text in enumerate(self.frame(width - 1, height)):
                try:
                    screen.addstr(row, 0, text)
                except curses.error:
                    pass
            screen.refresh()
            code = screen.getch()
            if code == -1:
                continue
            if code == curses.KEY_RESIZE:
                screen.erase()
                continue
            try:
                key = chr(code)
            except ValueError:
                continue
            if not self.handle(key):
                return
