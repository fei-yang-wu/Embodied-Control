"""Full-screen operator display for the lifecycle console.

The ladder on the left, the writer's numbers on the right, the keys along
the bottom, the last gate results underneath: one glance says where the
robot is and what the next key does.

Rendering builds rows of styled spans, so a label can be dim while its value
is bright and one link's dot can be red while its neighbours stay green;
`render` flattens the same rows to plain text, which is what the tests read
and what a pipe gets. Curses only maps a style name to an attribute.

Two rules the layout serves. SPACE reaches the writer before anything else:
it bypasses the lifecycle lock (a gate may hold it for its whole timeout) and
stores DAMP straight into the writer. And a key never blocks the screen:
actions run on one worker thread, one at a time, while the display keeps
refreshing from a lock-free snapshot.
"""

from __future__ import annotations

import queue
import re
import textwrap
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from embodied_control.console import (
    KEY_BACKSPACE,
    KEY_DAMP,
    KEY_DELETE,
    KEY_DOWN,
    KEY_END,
    KEY_ENTER,
    KEY_ESCAPE,
    KEY_HOME,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_SPACE,
    KEY_TAB,
    KEY_UP,
    ConsoleQuit,
    KeyBinding,
)
from embodied_control.robot.lifecycle import (
    LADDER,
    WRITER_MODE_NAMES,
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

SLASH_KEYS = {
    "damp": KEY_DAMP,
    "mode": "o",
    "motion": "m",
    "motion-prev": "M",
    "tracker": "t",
    "tracker-prev": "T",
    "frame-next": "f",
    "frame-prev": "F",
    "planner": "p",
    "rebuild": "r",
    "reset-sim": "R",
    "next": "n",
    "auto": "a",
    "go": "g",
    "hold": "h",
    "retake": "e",
    "lowered": "l",
    "hoisted": "H",
    "stand": "s",
    "release": "d",
    "abort": "x",
    "quit": "q",
}

NEXT_ACTION = {
    "NO TRACKER": "/rebuild",
    "IDLE": "/next",
    "PRECHECK": "/next",
    "VENDOR_DAMP_CONFIRMED": "/next",
    "SAFE_EXTERNAL_COMMAND_PRESENT": "/next",
    "USER_CONTROL_CONFIRMED": "/next",
    "START_POSE_RAMP": "/next",
    "POSE_SETTLED": "/lowered",
    "LOWERED": "/next",
    "POSE_MATCH_VERIFIED": "/next",
    "POLICY_COMMAND_FRESH": "/next",
    "PRIMED": "/go",
    "BLEND_IN": "wait",
    "RUNNING": "/hold",
    "HOLD": "/hoisted, then /damp",
    "DAMP": "/hoisted, then /damp",
    "FAULT": "/hoisted, then /damp",
    "RELEASED": "/next for the next episode",
    "VENDOR_RESTORED": "/next for the next episode",
    "VENDOR_STAND": "/lowered",
    "STANDING": "/next for the next episode",
}

DISPLAY_LABELS = {
    LifecycleState.SAFE_EXTERNAL_COMMAND_PRESENT: "SAFE COMMAND PRESENT",
}

# How each state colours its badge: urgency should read before words. States
# not listed here are rungs on the way up, and take the accent.
STATE_TONE = {
    "FAULT": "bad",
    "DAMP": "warn",
    "HOLD": "warn",
    "RUNNING": "ok",
    "BLEND_IN": "ok",
    "PRIMED": "ok",
    "STANDING": "accent",
    "NO TRACKER": "dim",
    "IDLE": "dim",
}

SPARK = "▁▂▃▄▅▆▇█"
# Partial blocks: a bar then moves every eighth of a cell, not every cell.
BLOCKS = " ▏▎▍▌▋▊▉█"

# Anything that has to line up column-wise is ASCII or Latin-1. Dingbats and
# Geometric Shapes (✓ U+2713, ● U+25CF) are missing from many monospace fonts,
# and the fallback glyph arrives at a different advance width, which walks the
# column separator sideways row by row. Blocks and box drawing stay: they are
# the one non-Latin range a terminal font reliably ships, and they only ever
# appear at the end of a field.
# The console targets an English terminal, so any single-cell glyph is fine
# here. Every row is padded to an exact cell count, and a mark that is not
# one cell wide would shift the column separator.
MARK_DONE = "✓"
MARK_NOW = "▸"
MARK_TODO = "·"
MARK_LINK_OK = "•"
MARK_LINK_BAD = "!"


class CommandLine:
    """A real prompt: a cursor, editing keys, history and completion.

    The console is a harness, so its prompt behaves like one. Everything here
    is pure, so the editing rules are tested without a terminal; the curses
    loop only turns key codes into the tokens this understands.
    """

    EDIT_KEYS = {
        KEY_BACKSPACE, KEY_DELETE, KEY_LEFT, KEY_RIGHT, KEY_HOME, KEY_END,
        KEY_UP, KEY_DOWN, KEY_TAB, KEY_ENTER, KEY_ESCAPE,
        "\x01", "\x05", "\x0b", "\x15", "\x17",
    }

    def __init__(self, commands: Iterable[str] = ()) -> None:
        self.commands = sorted(commands)
        self.text = ""
        self.cursor = 0
        self.history: list[str] = []
        self._recall: int | None = None
        self._draft = ""

    # -- editing -------------------------------------------------------

    def handle(self, key: str) -> str | None:
        """Apply one key. Returns "run", "cancel", or None to keep editing."""
        if key == KEY_ENTER or key in {"\r", "\n"}:
            text = self.text.strip().lower()
            if text and (not self.history or self.history[-1] != text):
                self.history.append(text)
            return "run"
        if key == KEY_ESCAPE:
            return "cancel"
        if key in {KEY_BACKSPACE, "\b", "\x7f"}:
            if not self.text:
                # Backspace on an empty prompt closes it, the way a shell's
                # own line editor gives the terminal back.
                return "cancel"
            if self.cursor:
                self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                self.cursor -= 1
        elif key == KEY_DELETE:
            self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
        elif key == KEY_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif key == KEY_RIGHT:
            self.cursor = min(len(self.text), self.cursor + 1)
        elif key in {KEY_HOME, "\x01"}:
            self.cursor = 0
        elif key in {KEY_END, "\x05"}:
            self.cursor = len(self.text)
        elif key == "\x15":  # ctrl-u
            self.text, self.cursor = self.text[self.cursor :], 0
        elif key == "\x0b":  # ctrl-k
            self.text = self.text[: self.cursor]
        elif key == "\x17":  # ctrl-w
            head = self.text[: self.cursor].rstrip()
            cut = max(head.rfind(" "), head.rfind("-")) + 1
            self.text = self.text[:cut] + self.text[self.cursor :]
            self.cursor = cut
        elif key == KEY_TAB:
            self._complete()
        elif key == KEY_UP:
            self._recall_history(-1)
        elif key == KEY_DOWN:
            self._recall_history(1)
        elif len(key) == 1 and key.isprintable() and len(self.text) < 40:
            self.text = self.text[: self.cursor] + key + self.text[self.cursor :]
            self.cursor += 1
        return None

    def _complete(self) -> None:
        matches = self.candidates()
        if not matches:
            return
        shared = matches[0]
        for candidate in matches[1:]:
            while not candidate.startswith(shared):
                shared = shared[:-1]
        if len(shared) > len(self.text):
            self.text, self.cursor = shared, len(shared)
        elif len(matches) == 1:
            self.text, self.cursor = matches[0], len(matches[0])

    def _recall_history(self, step: int) -> None:
        if not self.history:
            return
        if self._recall is None:
            if step > 0:
                return
            self._draft = self.text
            self._recall = len(self.history)
        index = self._recall + step
        if index >= len(self.history):
            self._recall, self.text = None, self._draft
        else:
            self._recall = max(0, index)
            self.text = self.history[self._recall]
        self.cursor = len(self.text)

    # -- view ----------------------------------------------------------

    def candidates(self) -> list[str]:
        return [name for name in self.commands if name.startswith(self.text.lower())]

    def ghost(self) -> str:
        """The rest of the only matching command, shown dim behind the cursor."""
        matches = self.candidates()
        if len(matches) == 1 and matches[0] != self.text:
            return matches[0][len(self.text) :]
        return ""


@dataclass(frozen=True)
class Span:
    """One run of characters that share a style name."""

    text: str
    style: str = "text"


Row = list[Span]


def _len(row: Iterable[Span]) -> int:
    return sum(len(span.text) for span in row)


def _cell_text(value: str) -> str:
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    return re.sub(r"[\x00-\x1f\x7f]", " ", text)


def pad(row: Row, width: int, style: str = "text") -> Row:
    """Truncate or pad a row to exactly `width` cells."""
    total = 0
    out: Row = []
    for span in row:
        if total >= width:
            break
        text = _cell_text(span.text)[: width - total]
        out.append(Span(text, span.style))
        total += len(text)
    if total < width:
        out.append(Span(" " * (width - total), style))
    return out


def flatten(row: Row) -> str:
    return "".join(span.text for span in row)


class _Note(str):
    """A log line that remembers its level; still a string everywhere else."""

    level = "info"


# A console log where every line looks the same is a log an operator scans
# instead of reads. Each note carries a level, and the level picks the style.
LOG_STYLES = {
    "fail": "bad",
    "damp": "warn",
    "gate": "ok",
    "info": "dim",
}


def classify_note(message: str) -> str:
    """The level a lifecycle note belongs to, from the text it already has.

    The lifecycle writes its own transition lines, so this reads them rather
    than asking every call site to label what it prints.
    """
    text = str(message)
    lowered = text.lower()
    if text.lstrip().startswith("!!") or "fault" in lowered or "failed" in lowered:
        return "fail"
    if "refus" in lowered or "timed out" in lowered or "ignored" in lowered:
        return "fail"
    if "damp" in lowered:
        return "damp"
    if "-> " in text and ": ok" in text:
        return "gate"
    return "info"


def _visual_log_lines(
    notes: Iterable[str], width: int
) -> list[tuple[str, str]]:
    """Terminal-safe wrapped lines, each with the level of the note it came from."""
    visual: list[tuple[str, str]] = []
    content_width = max(1, width - 1)
    # Bound work per 10 Hz redraw even when one dependency emits a traceback
    # as one giant message. Newest visual rows are the only rows displayable.
    for note in list(notes)[-64:]:
        level = getattr(note, "level", None) or classify_note(note)
        clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(note)[-4096:])
        physical = clean.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        for line in physical:
            line = line.expandtabs(4)
            line = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", line)
            wrapped = (
                textwrap.wrap(
                    line,
                    width=content_width,
                    subsequent_indent="  ",
                    replace_whitespace=False,
                )
                or [""]
            )
            visual.extend((text, level) for text in wrapped)
    return visual[-256:]


def _log_rows(
    notes: Iterable[str],
    width: int,
    height: int,
    visual: list[tuple[str, str]] | None = None,
) -> list[Row]:
    """Newest terminal-safe visual log lines within an exact row budget."""
    if height <= 0:
        return []
    visual = _visual_log_lines(notes, width) if visual is None else visual
    if height == 1:
        return [_rule("LOG", width)]
    visible = visual[-(height - 1) :]
    return [_rule("LOG", width)] + [
        pad([Span(f" {line}", LOG_STYLES.get(level, "dim"))], width)
        for line, level in visible
    ]


def sparkline(values: Iterable[float], width: int = 8) -> str:
    """A tiny trend, newest on the right; blank while there is no history."""
    series = [float(value) for value in values][-width:]
    if not series:
        return " " * width
    low, high = min(series), max(series)
    # A stream sitting at zero is a stream that is not running. Half-height
    # blocks there read as traffic, which is the opposite of the truth.
    if high <= 0.0:
        return " " * width
    span = high - low
    marks = "".join(
        SPARK[min(len(SPARK) - 1, int((value - low) / span * (len(SPARK) - 1)))]
        if span > 1e-9
        else SPARK[3]
        for value in series
    )
    return marks.rjust(width)


def bar_cells(fraction: float, width: int) -> str:
    """`width` cells filled to `fraction`, to an eighth of a cell."""
    fraction = 0.0 if fraction != fraction else max(0.0, min(1.0, fraction))
    full, remainder = divmod(int(round(fraction * width * 8)), 8)
    cells = "█" * min(full, width)
    # BLOCKS[0] is a space, which would read as a hole in the track.
    if full < width and remainder:
        cells += BLOCKS[remainder]
    return cells.ljust(width, "·")[:width]


def progress_bar(fraction: float, width: int = 30) -> str:
    """Plain-text form, for a caller that wants one string."""
    fraction = 0.0 if fraction != fraction else max(0.0, min(1.0, fraction))
    return f"[{bar_cells(fraction, width)}] {fraction * 100:3.0f}%"


def gauge(value: float, limit: float, width: int = 7) -> tuple[str, str]:
    """Bar cells plus the tone they should be drawn in."""
    fraction = 0.0 if limit <= 0 else value / limit
    tone = "ok" if fraction < 0.6 else ("warn" if fraction < 1.0 else "bad")
    return bar_cells(fraction, width), tone


def _rule(title: str, width: int, *, style: str = "rule") -> Row:
    """`── TITLE ──────`: a section header that is also the separator."""
    tail = max(0, width - 4 - len(title))
    return [
        Span("── ", style),
        Span(title, "section"),
        Span(" " + "─" * tail, style),
    ]


def _link(
    name: str, hz: float | None, healthy: bool, trend: Iterable[float], detail: Row
) -> Row:
    if hz is None:
        rate = "   -- Hz"
    else:
        # A planner runs at single-digit Hz; a wire runs at hundreds.
        rate = f"{hz:4.1f} Hz" if hz < 20.0 else f"{hz:5.0f} Hz"
    return [
        Span(f" {MARK_LINK_OK if healthy else MARK_LINK_BAD} ",
             "ok" if healthy else "bad"),
        # Fixed-width name and rate: three links in a column are compared by
        # their numbers, and numbers that do not line up are not compared.
        Span(f"{name:<9}", "label"),
        Span(f" {rate} ", "value"),
        Span(sparkline(trend, 6), "accent"),
    ] + detail


# Under this width the three links stop sharing a row: a truncated link is
# worse than a taller panel, because the number that got cut is the one an
# operator is looking for.
LINKS_ONE_ROW = 118


def comm_rows(
    snapshot: dict, rates: dict | None, trends: dict | None, width: int = 999
) -> list[Row]:
    """Link health: the robot's state stream, our command stream, the planner."""
    ws = snapshot.get("writer", {})
    st = snapshot.get("control", {})
    rates = rates or {}
    trends = trends or {}
    gap_ms = float(ws.get("state_gap_ns_max", 0)) / 1.0e6
    age = st.get("command_age_ms", -1.0)
    age_text = "--" if age is None or float(age) < 0 else f"{float(age):.0f} ms"
    crc = int(ws.get("crc_errors", 0))
    fails = int(ws.get("publish_failures", 0))
    stale = int(st.get("stale_responses", 0))
    state_hz = rates.get("state_hz")
    lowcmd_hz = rates.get("publish_hz")
    planner_hz = rates.get("planner_hz")

    def count(label: str, value: int) -> Row:
        return [
            Span(f" {label} ", "label"),
            Span(str(value), "value" if value == 0 else "bad"),
        ]

    links = [
        _link(
            "LOWSTATE",
            state_hz,
            (state_hz is None or state_hz > 400.0) and crc == 0,
            trends.get("state_hz", ()),
            [Span(f" gap {gap_ms:.1f}ms", "dim")] + count("crc", crc),
        ),
        _link(
            "LOWCMD",
            lowcmd_hz,
            fails == 0,
            trends.get("publish_hz", ()),
            count("fail", fails),
        ),
        _link(
            "PLANNER",
            planner_hz,
            stale == 0 and (age is None or float(age) < 500.0),
            trends.get("planner_hz", ()),
            [Span(f" age {age_text}", "dim")] + count("stale", stale),
        ),
    ]
    if width < LINKS_ONE_ROW:
        return links
    joined: Row = []
    for index, link in enumerate(links):
        joined += ([Span("  │", "rule")] if index else []) + link
    return [joined]


def comm_lines(snapshot: dict, rates: dict | None) -> list[str]:
    return [flatten(row) for row in comm_rows(snapshot, rates, None)]


def reference_row(snapshot: dict, width: int) -> Row | None:
    session = snapshot.get("session")
    if not session:
        return None
    if session.get("mode") != "oracle":
        return [
            Span(" VLA", "label"),
            Span("  planner-driven; no reference trajectory", "dim"),
        ]
    length = int(session.get("motion_length", 0))
    start = int(session.get("start_frame", 0))
    total = max(1, length - start)
    played = max(
        0, min(total, int(snapshot.get("control", {}).get("reference_ticks", 0)))
    )
    fraction = played / total
    cells = max(12, min(44, width - 46))
    return [
        Span(" REFERENCE  ", "label"),
        Span("[", "rule"),
        Span(bar_cells(fraction, cells), "accent"),
        Span("]", "rule"),
        Span(f" {fraction * 100:3.0f}%", "value"),
        Span(f"  frame {start + played}/{length}", "dim"),
        Span(f"  ({played}/{total} played)", "dim"),
    ]


def reference_line(snapshot: dict) -> str | None:
    row = reference_row(snapshot, 100)
    return None if row is None else flatten(row)


def _state_cell(entry: LifecycleState, current: int, index: int, width: int) -> Row:
    if index < current:
        mark, tone, text_style = MARK_DONE, "ok", "dim"
    elif index == current:
        mark, tone, text_style = MARK_NOW, "accent", "state.now"
    else:
        mark, tone, text_style = MARK_TODO, "rule", "dim"
    label = DISPLAY_LABELS.get(entry, str(entry).replace("_", " "))
    # The rung number is what an operator says out loud ("we are at 8"), and
    # it makes the climb legible when every label is dim.
    rung = index < len(LADDER) - 1
    number = f"{index + 1:>2} " if rung else "   "
    if not rung and index > current:
        mark, tone = " ", "rule"
    return pad(
        [Span(f" {mark} ", tone), Span(number, "rule"), Span(label, text_style)],
        width,
    )


def _fit(groups: list[Row], width: int) -> Row:
    """As many whole groups as fit. A dropped field beats a field cut in half."""
    out: Row = []
    for group in groups:
        if _len(out) + _len(group) > width:
            break
        out += group
    return out


def _metric(
    label: str, value: str, *, tone: str = "value", trail: Row | None = None
) -> Row:
    return [Span(f" {label:<15}", "label"), Span(f"{value:>9}", tone)] + (trail or [])


def _live_rows(snapshot: dict) -> list[Row]:
    ws = snapshot.get("writer", {})
    st = snapshot.get("control", {})
    hardware = int(ws.get("hardware_faults", 0))
    watchdog = int(ws.get("watchdog_faults", 0))
    fails = int(ws.get("publish_failures", 0))
    crc = int(ws.get("crc_errors", 0))
    gate_open = bool(ws.get("gate_open"))
    tracking = float(ws.get("tracking_error_max", 0.0))
    speed = float(ws.get("joint_speed_max", 0.0))
    ramp = float(ws.get("ramp_error_max", 0.0))
    # Full bar at the writer's own guards: 1 rad of tracking error is the
    # ramp-fault scale, 2 rad/s is twice the init ramp cap.
    track_cells, track_tone = gauge(tracking, 1.0)
    speed_cells, speed_tone = gauge(speed, 2.0)
    rows = [
        _metric(
            "publishes",
            f"{ws.get('publishes', 0)}",
            trail=[
                Span("  fail ", "label"),
                Span(str(fails), "value" if fails == 0 else "bad"),
            ],
        ),
        _metric(
            "gate",
            "OPEN" if gate_open else "closed",
            tone="ok" if gate_open else "dim",
            trail=[
                Span("  crc ", "label"),
                Span(str(crc), "value" if crc == 0 else "bad"),
            ],
        ),
        _metric("control ticks", f"{st.get('control_ticks', 0)}"),
        _metric("reference", f"{st.get('reference_ticks', 0)}"),
        _metric("planner replies", f"{st.get('planner_responses', 0)}"),
        [
            Span(" faults hw/wd   ", "label"),
            Span(f"{hardware:>4}", "value" if hardware == 0 else "bad"),
            Span("/", "rule"),
            Span(f"{watchdog:<4}", "value" if watchdog == 0 else "bad"),
            Span("  reason ", "label"),
            Span(str(ws.get("state_fault_reason", 0)), "dim"),
        ],
        _metric(
            "joint speed",
            f"{speed:.3f}",
            trail=[Span(" rad/s ", "dim"), Span(speed_cells, speed_tone)],
        ),
        _metric(
            "tracking error",
            f"{tracking:.3f}",
            trail=[Span(" rad   ", "dim"), Span(track_cells, track_tone)],
        ),
        _metric(
            "ramp error",
            f"{ramp:.3f}",
            trail=[
                Span(
                    f" rad  j{ws.get('ramp_error_joint', '-')}" if ramp > 0 else " rad",
                    "dim",
                )
            ],
        ),
        _metric(
            "first action",
            f"{float(ws.get('command_target_error_max', 0.0)):.3f}",
            trail=[Span(" rad", "dim")],
        ),
        _metric(
            "blend left",
            f"{ws.get('blend_ticks_remaining', 0)}",
            trail=[Span(" ticks", "dim")],
        ),
        [
            Span(" ack hoist/lower", "label"),
            Span(
                "  yes" if snapshot.get("hoisted_ack") else "   no",
                "ok" if snapshot.get("hoisted_ack") else "dim",
            ),
            Span("/", "rule"),
            Span(
                "yes" if snapshot.get("lowered_ack") else "no",
                "ok" if snapshot.get("lowered_ack") else "dim",
            ),
        ],
    ]
    hoist = snapshot.get("hoist")
    if hoist:
        owned = bool(hoist.get("owned"))
        rows.append(
            [
                Span(" plant           ", "label"),
                Span(
                    HOIST_MODE_NAMES.get(int(hoist.get("hoist_mode", -1)), "?"), "value"
                ),
                Span("  vendor ", "label"),
                Span("owns" if owned else "released", "warn" if owned else "ok"),
                Span(f"  fsm {hoist.get('fsm_id', '-')}", "dim"),
            ]
        )
    return rows


def _help_rows(
    bindings: list[KeyBinding], width: int, height: int, agent_label: str
) -> list[Row]:
    keys = {binding.key for binding in bindings}
    rows: list[Row] = [
        pad(
            [Span(" ▌EMBODIED-CONTROL", "bar"), Span("  COMMAND PALETTE ", "bar.dim")],
            width,
            "bar.dim",
        ),
        _rule("COMMANDS", width),
    ]
    commands = [
        (name, key) for name, key in SLASH_KEYS.items() if key in keys or key == "q"
    ]
    commands += [
        ("status", "show current snapshot result"),
        ("clear", "clear console log"),
    ]
    if agent_label != "off":
        commands += [
            ("diagnose", f"read-only analysis with {agent_label}"),
            ("agent", "alias for /diagnose"),
        ]
    columns = 2 if width >= 84 else 1
    column_width = width // columns
    half = (len(commands) + columns - 1) // columns
    for row_index in range(half):
        row: Row = []
        for column in range(columns):
            index = row_index + column * half
            if index >= len(commands):
                continue
            name, value = commands[index]
            label = next(
                (item.label for item in bindings if item.key == value), str(value)
            )
            row += pad(
                [
                    Span(f"  /{name:<13}", "accent"),
                    Span(label.split(":")[0].split(" (")[0], "dim"),
                ],
                column_width,
            )
        rows.append(pad(row, width))
    rows += [
        pad([], width),
        pad(
            [
                Span(" Safety: ", "label"),
                Span(
                    "Ctrl-D always damps, including while this palette is open "
                    "or the prompt has text.",
                    "warn",
                ),
            ],
            width,
        ),
        pad(
            [
                Span(" Press ", "dim"),
                Span("?", "accent"),
                Span(" or ", "dim"),
                Span("ESC", "accent"),
                Span(" to close, ", "dim"),
                Span("/", "accent"),
                Span(" to open the prompt: ", "dim"),
                Span("TAB", "accent"),
                Span(" completes, ", "dim"),
                Span("↑↓", "accent"),
                Span(" recall, ", "dim"),
                Span("^U ^W ^A ^E", "accent"),
                Span(" edit.", "dim"),
            ],
            width,
        ),
    ]
    return rows[:height]


def _header_rows(snapshot: dict, width: int) -> list[Row]:
    state = str(snapshot.get("state", "?"))
    ws = snapshot.get("writer", {})
    writer_mode = WRITER_MODE_NAMES.get(int(ws.get("mode", -1)), "?")
    vendor = "released" if ws.get("vendor_released") else "owns joints"
    if not snapshot.get("vendor_name"):
        vendor = "unknown"
    badge: Row = [
        Span(f" ep {snapshot.get('episode', 0)} ", "bar.dim"),
        Span(f" {state} ", f"pill.{STATE_TONE.get(state, 'accent')}"),
    ]
    title: Row = [
        Span(" ▌EMBODIED-CONTROL", "bar"),
        Span("  G1 LIFECYCLE ", "bar.dim"),
    ]
    gap = max(0, width - _len(title) - _len(badge))
    rows = [pad(title + [Span(" " * gap, "bar.dim")] + badge, width, "bar.dim")]
    if snapshot.get("fault_reason"):
        rows.append(
            pad(
                [
                    Span(" ! FAULT  ", "bad"),
                    Span(str(snapshot["fault_reason"]), "bad"),
                ],
                width,
            )
        )
    status: Row = [
        Span(" vendor ", "label"),
        Span(vendor, "ok" if vendor == "released" else "value"),
        Span("   writer ", "label"),
        Span(writer_mode, "warn" if writer_mode == "damp" else "value"),
    ]
    # The robot may boot facing any direction. Say by how much the fixed
    # anchor turned the reference to meet it, so an operator can see that
    # the heading was captured rather than assumed.
    if ws.get("anchor_heading_captured"):
        status += [
            Span("   boot yaw ", "label"),
            Span(f"{float(ws.get('anchor_yaw_offset_degrees', 0.0)):+.0f}°", "value"),
        ]
    status += [
        Span("   next ", "label"),
        Span(NEXT_ACTION.get(state, "/help"), "accent"),
    ]
    rows.append(pad(status, width))
    return rows


def _session_rows(snapshot: dict, width: int) -> list[Row]:
    session = snapshot.get("session")
    if not session:
        return []
    rows = [_rule("SESSION", width)]
    built = bool(session.get("built"))
    planner = str(session.get("planner", "?"))
    # Whole fields, so a narrow terminal drops the last one instead of
    # cutting a word in half. The first three are the selection an operator
    # is about to run, and they never drop.
    fields: list[Row] = [
        [
            Span(" MODE ", "label"),
            Span(str(session.get("mode", "?")), "accent"),
        ],
        # The planner is the operator's to start, so its state sits where a
        # narrow terminal cannot drop it.
        [
            Span("   planner ", "label"),
            Span(planner, "ok" if "running" in planner else "warn"),
        ],
        [
            Span("   TRACKER ", "label"),
            Span(str(session.get("tracker") or "default"), "accent"),
            Span(
                f" {session.get('tracker_index', 0)}/{session.get('tracker_count', 0)}",
                "dim",
            ),
        ],
        [
            Span("   MOTION ", "label"),
            Span(session.get("motion") or "default stance", "value"),
            Span(
                f" {session.get('motion_index', 0)}/{session.get('catalog_size', 0)}",
                "dim",
            ),
        ],
        [
            Span("   frame ", "label"),
            Span(
                f"{session.get('start_frame', 0)}/{session.get('motion_length', 0)}",
                "value",
            ),
        ],
        [
            Span("   tracker ", "label"),
            Span("ready" if built else "needs /rebuild", "ok" if built else "warn"),
        ],
    ]
    rows.append(pad(_fit(fields, width), width))
    reference = reference_row(snapshot, width)
    if reference:
        rows.append(pad(reference, width))
    for entry in (session.get("episodes", [])[-1:] if width >= 96 else []):
        row: Row = [
            Span(" LAST EP ", "label"),
            Span(f"{entry['episode']:>3}", "value"),
            Span(
                f" {entry['mode']} {entry['motion'] or 'default'}@{entry['start_frame']}",
                "dim",
            ),
            Span("   ticks ", "label"),
            Span(str(entry["ticks"]), "value"),
            Span(f"   frames {entry['first_frame']}-{entry['last_frame']}", "dim"),
            Span("   joint MAE ", "label"),
            Span(f"{entry['joint_mae_mean_rad']:.3f}", "value"),
            Span(f" rad (p95 {entry['joint_mae_p95_rad']:.3f})", "dim"),
        ]
        if entry.get("mpjpe_l_mm") is not None:
            row += [
                Span("   MPJPE_l ", "label"),
                Span(f"{entry['mpjpe_l_mm']:.1f} mm", "value"),
            ]
        faults = (
            int(entry["runtime_fault"])
            + int(entry["hardware_faults"])
            + int(entry["watchdog_faults"])
        )
        row += [
            Span("   faults ", "label"),
            Span(
                f"{entry['runtime_fault']}/{entry['hardware_faults']}"
                f"/{entry['watchdog_faults']}",
                "value" if faults == 0 else "bad",
            ),
        ]
        rows.append(pad(row, width))
    return rows


def _body_rows(snapshot: dict, width: int) -> list[Row]:
    state = str(snapshot.get("state", "?"))
    try:
        current = DISPLAY_ORDER.index(LifecycleState(state))
    except ValueError:
        current = -1
    live = _live_rows(snapshot)
    rows: list[Row] = []
    wide = width >= 96
    ladder_width = min(62, max(54, width - 40)) if wide else 34
    columns = 2 if wide else 1
    per_column = (len(DISPLAY_ORDER) + columns - 1) // columns
    cell_width = (ladder_width - 1) // columns
    rows.append(
        pad(
            pad(_rule("LIFECYCLE", ladder_width - 1), ladder_width - 1)
            + [Span("│", "rule")]
            + _rule("TELEMETRY", width - ladder_width),
            width,
        )
    )
    for index in range(max(per_column, len(live))):
        left: Row = []
        for column in range(columns):
            entry_index = index + column * per_column
            room = (
                cell_width
                if column + 1 < columns
                else ladder_width - 1 - cell_width * column
            )
            # `index` runs past per_column when telemetry is the longer side;
            # without the first test the tail of column one repeats column two.
            left += (
                _state_cell(DISPLAY_ORDER[entry_index], current, entry_index, room)
                if index < per_column and entry_index < len(DISPLAY_ORDER)
                else pad([], room)
            )
        right = live[index] if index < len(live) else []
        rows.append(pad(left + [Span("│", "rule")] + right, width))
    return rows


def _key_hint(key: str, label: str) -> Row:
    """A reverse-video keycap and its verb; the keycap carries its own padding."""
    return [Span(f" {key} ", "key"), Span(f"{label} ", "dim")]


#: The prompt is always this tall, open or closed, matching or not. A footer
#: that grows by a row the moment a command matches moves every row above it,
#: and ncurses realises that with an insert-line: on a short terminal the
#: bottom row falls off and the prompt lands inside the key legend. Reserving
#: the row costs one line and keeps every frame the same shape.
PROMPT_ROWS = 2


def _prompt_rows(line: "CommandLine | None", width: int) -> list[Row]:
    """The prompt, plus what it would complete to and what else matches."""
    if line is None:
        return [
            pad(
                [
                    Span(" > ", "accent"),
                    Span("Type / for commands · /diagnose explains current failure", "dim"),
                ],
                width,
            ),
            pad([], width),
        ]
    ghost = line.ghost()
    rows = [
        pad(
            [
                Span(" > ", "accent"),
                Span(f"/{line.text}", "value"),
                Span(ghost, "dim"),
                Span("" if ghost else "_", "accent"),
            ],
            width,
        )
    ]
    matches = line.candidates()
    if line.text and matches:
        row: Row = [Span("   ", "text")]
        for name in matches[:8]:
            row += [Span(f"/{name}", "accent" if name == matches[0] else "dim"),
                    Span("  ", "text")]
        if len(matches) > 8:
            row.append(Span(f"+{len(matches) - 8} more", "dim"))
        rows.append(pad(row, width))
    elif line.text:
        rows.append(pad([Span("   no command matches; TAB completes, ESC cancels",
                              "warn")], width))
    else:
        rows.append(pad([Span("   TAB completes · ESC cancels · Enter runs", "dim")],
                        width))
    assert len(rows) == PROMPT_ROWS
    return rows


def _footer_rows(
    snapshot: dict, width: int, busy: str, line: "CommandLine | None"
) -> list[Row]:
    rows = [_rule("CONTROLS", width)]
    groups: list[Row] = [[Span(" ^D DAMP ", "pill.bad")]]
    groups += [_key_hint(key, label) for key, label in (("/", "commands"), ("?", "help"))]
    groups.append([Span("  │", "rule")])
    groups += [
        _key_hint(key, label)
        for key, label in (
            ("o", "mode"),
            ("t/T", "tracker"),
            ("m/M", "motion"),
            ("f/F", "frame"),
            ("p", "planner"),
            ("r", "rebuild"),
            ("R", "reset sim"),
        )
    ]
    controls: Row = _fit(groups, width)
    rows.append(pad(controls, width))
    second: list[Row] = [
        _key_hint(key, label)
        for key, label in (
            ("n", "next"),
            ("a", "auto"),
            ("g", "go"),
            ("h", "hold"),
            ("e", "retake"),
            ("H/l", "hoist/lower"),
            ("s", "stand"),
            ("d", "release"),
            ("x", "abort"),
        )
    ]
    rows.append(pad(_fit(second, width), width))
    rows += _prompt_rows(line, width)
    if busy:
        rows.append(
            pad(
                [
                    Span(" ~ ", "warn"),
                    Span(f"{busy}…", "warn"),
                    Span("   ^D still damps", "dim"),
                ],
                width,
            )
        )
    else:
        last_ok = snapshot.get("last_ok")
        tone = "dim" if last_ok is None else ("ok" if last_ok else "bad")
        tag = "" if last_ok is None else ("ok " if last_ok else "!! ")
        rows.append(
            pad(
                [Span(f" {tag}", tone), Span(str(snapshot.get("last_detail", "")), tone)],
                width,
            )
        )
    return rows


def render_rows(
    snapshot: dict,
    bindings: list[KeyBinding],
    notes: list[str],
    *,
    busy: str = "",
    width: int = 100,
    height: int = 36,
    rates: dict | None = None,
    trends: dict | None = None,
    command: "CommandLine | str | None" = None,
    help_open: bool = False,
    assistant: list[str] | None = None,
    agent_label: str = "off",
) -> list[Row]:
    """Styled rows of exactly `width` cells, at most `height` of them."""
    width = max(60, width)
    line = CommandLine() if isinstance(command, str) else command
    if isinstance(command, str):
        line.text, line.cursor = command, len(command)
    if help_open:
        return _help_rows(bindings, width, height, agent_label)

    rows = _header_rows(snapshot, width)
    rows += _session_rows(snapshot, width)
    rows.append(_rule("LINKS", width))
    rows += [pad(row, width) for row in comm_rows(snapshot, rates, trends, width)]

    assistant = assistant or []
    if assistant:
        rows.append(_rule("DIAGNOSIS", width, style="diagnosis"))
        footer = [
            _rule("CONTROLS", width),
            pad(
                [
                    Span(" ^D DAMP ", "pill.bad"),
                    Span("   /clear close    /diagnose refresh    ? help", "dim"),
                ],
                width,
            ),
            pad(
                [
                    Span(" > ", "accent"),
                    Span(
                        f"/{line.text}" if line is not None else "Type /clear to return",
                        "value" if line is not None else "dim",
                    ),
                ],
                width,
            ),
        ]
        for source_line in assistant:
            wrapped = textwrap.wrap(source_line, width=max(20, width - 2)) or [""]
            for line in wrapped:
                if len(rows) >= height - len(footer):
                    break
                style = "diagnosis" if "read-only" in line else "text"
                rows.append(pad([Span(" ", "text"), Span(line, style)], width))
            if len(rows) >= height - len(footer):
                break
        return (rows + footer)[:height]

    footer = _footer_rows(snapshot, width, busy, line)
    log_lines = _visual_log_lines(notes, width)
    log_height = min(8, len(log_lines) + 1)
    room = max(0, height - len(rows) - len(footer) - log_height)
    # A lone panel header helps nobody; below three rows the ladder is out.
    if room >= 3:
        rows += _body_rows(snapshot, width)[:room]
    rows += footer
    if len(rows) < height:
        rows += _log_rows(notes, width, height - len(rows), log_lines)
    return rows[:height]


def render(*args, **kwargs) -> list[str]:
    """Plain-text rows: what a pipe, a test, or a colourless terminal sees."""
    return [flatten(row) for row in render_rows(*args, **kwargs)]


def _ensure_terminfo() -> None:
    """Make sure `curses.setupterm()` can succeed before the console starts.

    Two independent failure modes have shown up in practice, both giving the
    same unhelpful `setupterm: could not find terminfo database`:

    1. conda-forge's ncurses build doesn't reliably locate its own bundled
       terminfo directory (`pixi.toml` sets `TERMINFO` for the `native`
       feature to work around this).
    2. A terminal with a very new `TERM` (Ghostty's `xterm-ghostty`, and
       likely others) whose terminfo entry exists on the system but was
       written by a newer ncurses than the one in this environment, which
       fails to read it - even when pointed at it directly. There is no
       directory to fix; the entry itself is unreadable here.

    (1) is fixed by trying more `TERMINFO` directories. (2) has no fix on
    that axis, so if every directory still fails to resolve the terminal's
    own `TERM`, this falls back to `xterm-256color` outright: the console
    uses no capability beyond basic colour and cursor movement, so losing
    whatever is specific to the real terminal costs nothing here.
    """
    import curses
    import os
    import sys

    def probe(term: str) -> bool:
        try:
            curses.setupterm(term=term)
            return True
        except curses.error:
            return False

    term = os.environ.get("TERM", "xterm")
    if probe(term):
        return
    for candidate in (
        os.environ.get("TERMINFO", ""),
        os.path.join(sys.prefix, "lib", "terminfo"),
        os.path.join(sys.prefix, "share", "terminfo"),
        "/usr/share/terminfo",
        "/etc/terminfo",
        "/lib/terminfo",
    ):
        if candidate and os.path.isdir(candidate):
            os.environ["TERMINFO"] = candidate
            if probe(term):
                return
    # Nothing could read this TERM's own entry. Force a value every terminfo
    # database ships (this is an override, not setdefault: the point is that
    # the real TERM value is the one that just failed).
    for fallback in ("xterm-256color", "xterm", "vt100"):
        os.environ["TERM"] = fallback
        if probe(fallback):
            return


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
        diagnose: Callable[[dict, list[str]], object] | None = None,
        agent_label: str = "off",
        pin: Callable[[], object] | None = None,
        theme: str = "auto",
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
        self._trends: dict[str, deque[float]] = {
            name: deque(maxlen=8) for name in ("state_hz", "publish_hz", "planner_hz")
        }
        self._line: CommandLine | None = None
        self._help_open = False
        self._history: list[str] = []
        self._diagnose = diagnose
        # Called once on each console thread: the display and the key handler
        # must not land on a core a SCHED_FIFO robot thread owns.
        self._pin = pin
        self.agent_label = agent_label
        self.theme = resolve_theme(theme)
        self.assistant: list[str] = []

    # -- model ---------------------------------------------------------

    def note(self, message: str, level: str | None = None) -> None:
        """One log line. `level` overrides what the text itself implies."""
        note = _Note(f"{self._clock() - self._t0:7.1f}s  {message}")
        note.level = level or classify_note(message)
        self.notes.append(note)

    def command_names(self) -> list[str]:
        keys = set(self.bindings)
        names = [name for name, key in SLASH_KEYS.items() if key in keys or key == "q"]
        names += ["help", "status", "clear"]
        if self._diagnose is not None:
            names += ["diagnose", "agent"]
        return names

    def handle(self, key: str) -> bool:
        """Dispatch one key. Returns False when the console should exit."""
        if key == KEY_DAMP:
            # Ahead of the prompt, the palette and the worker queue: the chord
            # is never text and never waits.
            self._line = None
            self._help_open = False
            self.lifecycle.emergency_damp()
            self.note("^D: damp stored in the writer", "damp")
            self._queue.put(self.bindings[KEY_DAMP])
            return True
        if self._line is not None:
            return self._handle_command_key(key)
        if key == "/":
            self._line = CommandLine(self.command_names())
            self._line.history = list(self._history)
            self._help_open = False
            return True
        if key == "?":
            self._help_open = not self._help_open
            return True
        if key in {KEY_ESCAPE, "\x1b"} and self._help_open:
            self._help_open = False
            return True
        if key == KEY_SPACE:
            self.note("SPACE does nothing here; damp is Ctrl-D")
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

    def _handle_command_key(self, key: str) -> bool:
        assert self._line is not None
        outcome = self._line.handle(key)
        if outcome == "cancel":
            self._line = None
            return True
        if outcome == "run":
            command = self._line.text.strip().lower()
            history = self._line.history
            self._line = None
            result = self._execute_command(command)
            # The prompt closes on Enter, but its history outlives it.
            self._history = history
            return result
        return True

    def _execute_command(self, command: str) -> bool:
        if not command:
            self._help_open = True
            return True
        if command == "help":
            self._help_open = True
            return True
        if command == "clear":
            self.notes.clear()
            self.assistant.clear()
            return True
        if command == "status":
            snapshot = self.lifecycle.snapshot()
            self.note(
                f"state {snapshot.get('state', '?')}; "
                f"{snapshot.get('last_detail', 'no gate result')}"
            )
            return True
        if command in {"diagnose", "agent"}:
            if self._diagnose is None:
                detail = "disabled" if self.agent_label == "off" else "unavailable"
                self.note(
                    f"diagnostic agent {detail}; choose an installed CLI "
                    "with --diagnostic-agent"
                )
                return True
            if self._busy:
                self.note(f"busy with '{self._busy}'; /{command} ignored")
                return True
            self.assistant.clear()
            self._queue.put(
                KeyBinding("", f"diagnosing with {self.agent_label}", self._run_diagnosis)
            )
            return True
        key = SLASH_KEYS.get(command)
        if key is None or (key != "q" and key not in self.bindings):
            self.note(f"unknown command /{command}; use /help")
            return True
        return self.handle(key)

    def _run_diagnosis(self) -> None:
        assert self._diagnose is not None
        result = self._diagnose(self.lifecycle.snapshot(), list(self.notes))
        provider = getattr(result, "provider", self.agent_label)
        text = getattr(result, "text", str(result))
        clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
        clean = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", clean)
        lines: list[str] = [f"{provider} · read-only"]
        for source_line in clean.splitlines():
            if not source_line.strip():
                lines.append("")
                continue
            lines.append(source_line)
        self.assistant[:] = lines[:80]
        self.note(f"{provider} diagnosis ready; /clear returns to log")

    def _work(self) -> None:
        if self._pin is not None:
            self._pin()
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
                self.note(f"!! {type(exc).__name__}: {exc}", "fail")
            finally:
                self._busy = ""

    def rates(self, snapshot: dict) -> dict:
        """Per-second rates from counter deltas, refreshed every 0.5 s."""
        now = self._clock()
        counters = {
            "state_frames": int(snapshot.get("writer", {}).get("state_frames", 0)),
            "publishes": int(snapshot.get("writer", {}).get("publishes", 0)),
            "planner_responses": int(
                snapshot.get("control", {}).get("planner_responses", 0)
            ),
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
                "planner_hz": (
                    counters["planner_responses"] - previous["planner_responses"]
                )
                / dt,
            }
            for name, value in self._rates.items():
                self._trends[name].append(value)
            self._last_sample = (now, counters)
        return self._rates

    def frame_rows(self, width: int, height: int) -> list[Row]:
        snapshot = self.lifecycle.snapshot()
        return render_rows(
            snapshot,
            self.ordered,
            list(self.notes),
            busy=self._busy,
            width=width,
            height=height,
            rates=self.rates(snapshot),
            trends={name: list(values) for name, values in self._trends.items()},
            command=self._line,
            help_open=self._help_open,
            assistant=self.assistant,
            agent_label=self.agent_label,
        )

    def frame(self, width: int, height: int) -> list[str]:
        return [flatten(row) for row in self.frame_rows(width, height)]

    # -- terminal ------------------------------------------------------

    def run(self) -> int:
        import curses

        _ensure_terminfo()
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

        try:
            curses.curs_set(0)
        except curses.error:
            pass
        styles = build_styles(curses, self.theme)
        screen.keypad(True)  # arrows, Home/End and Backspace arrive as codes
        screen.nodelay(True)
        screen.timeout(int(self.refresh_seconds * 1000))
        while True:
            height, width = screen.getmaxyx()
            screen.erase()
            rows = self.frame_rows(width - 1, height)
            paint_rows(screen, rows, width - 1, height, styles, curses.error)
            try:
                curses.curs_set(1 if self._line is not None else 0)
                if self._line is not None:
                    prompt_row = next(
                        index
                        for index, row in enumerate(rows)
                        if flatten(row).startswith(" > /")
                    )
                    screen.move(prompt_row, min(width - 2, 4 + self._line.cursor))
            except (curses.error, StopIteration):
                pass
            screen.refresh()
            code = screen.getch()
            if code == -1:
                continue
            if code == curses.KEY_RESIZE:
                screen.erase()
                continue
            key = decode_key(curses, code)
            if key is None:
                continue
            if not self.handle(key):
                return


def paint_rows(
    screen,
    rows: list[Row],
    width: int,
    height: int,
    styles: dict[str, int],
    curses_error: type[BaseException],
) -> None:
    """Paint bounded single-line spans; no text may move curses' cursor."""
    for row_index, row in enumerate(rows[: max(0, height)]):
        column = 0
        for span in row:
            if column >= width:
                break
            text = _cell_text(span.text)[: width - column]
            if not text:
                continue
            try:
                screen.addnstr(
                    row_index,
                    column,
                    text,
                    len(text),
                    styles.get(span.style, styles["text"]),
                )
            except curses_error:
                pass
            column += len(text)


def decode_key(curses, code: int) -> str | None:
    """One curses key code to the token the console understands.

    Backspace is the reason this exists: with `keypad(True)` it arrives as
    KEY_BACKSPACE (263), and `chr(263)` is a letter, so the old loop inserted
    a character instead of deleting one.
    """
    named = {
        curses.KEY_BACKSPACE: KEY_BACKSPACE,
        curses.KEY_DC: KEY_DELETE,
        curses.KEY_LEFT: KEY_LEFT,
        curses.KEY_RIGHT: KEY_RIGHT,
        curses.KEY_UP: KEY_UP,
        curses.KEY_DOWN: KEY_DOWN,
        curses.KEY_HOME: KEY_HOME,
        curses.KEY_END: KEY_END,
        curses.KEY_ENTER: KEY_ENTER,
        8: KEY_BACKSPACE,
        127: KEY_BACKSPACE,
        9: KEY_TAB,
        10: KEY_ENTER,
        13: KEY_ENTER,
        27: KEY_ESCAPE,
    }
    if code in named:
        return named[code]
    if 0 <= code < 0x110000:
        return chr(code)
    return None


# 256 colours first, because a grey that is actually grey is what separates a
# label from its value; the 8-colour fallback keeps the same meanings.
# Two palettes, because a colour that reads on black is invisible on white and
# the reverse. `value` is the number an operator came to read, so it takes the
# strongest contrast in each theme and `dim` and `rule` step back from it.
# Reverse-video styles (bar, pill, key) swap fg and bg, so their colour is the
# chip's background: it has to be dark enough for light text either way.
PALETTE_256_DARK = {
    "accent": 44,
    "ok": 78,
    "warn": 214,
    "bad": 203,
    "dim": 244,
    "value": 253,
    "section": 45,
    "rule": 238,
    "diagnosis": 176,
    "key": 250,
}
PALETTE_256_LIGHT = {
    "accent": 24,
    "ok": 22,
    "warn": 130,
    "bad": 124,
    "dim": 240,
    "value": 234,
    "section": 25,
    "rule": 250,
    "diagnosis": 90,
    "key": 238,
}
PALETTE_8_DARK = {
    "accent": "CYAN",
    "ok": "GREEN",
    "warn": "YELLOW",
    "bad": "RED",
    "dim": "WHITE",
    "value": "WHITE",
    "section": "CYAN",
    "rule": "BLUE",
    "diagnosis": "MAGENTA",
    "key": "WHITE",
}
# Eight-colour yellow is unreadable on white, so `warn` borrows magenta here;
# a terminal with 256 colours gets the amber it should have.
PALETTE_8_LIGHT = {
    "accent": "BLUE",
    "ok": "GREEN",
    "warn": "MAGENTA",
    "bad": "RED",
    "dim": "BLACK",
    "value": "BLACK",
    "section": "BLUE",
    "rule": "BLACK",
    "diagnosis": "MAGENTA",
    "key": "BLACK",
}
PALETTE_256 = PALETTE_256_DARK
PALETTE_8 = PALETTE_8_DARK
THEMES = ("dark", "light")


def resolve_theme(requested: str = "auto", environ: dict | None = None) -> str:
    """Which palette to paint with.

    `auto` reads COLORFGBG, which terminals that set it publish as
    `<foreground>;<background>`: a background of 7 or 15 is a light terminal.
    Nothing else is guessable from inside curses, so anything unset stays
    dark, and `EC_TUI_THEME` overrides the lot for a terminal that lies.
    """
    import os

    environ = os.environ if environ is None else environ
    override = str(environ.get("EC_TUI_THEME", "")).strip().lower()
    if override in THEMES:
        return override
    wanted = str(requested or "auto").strip().lower()
    if wanted in THEMES:
        return wanted
    colours = str(environ.get("COLORFGBG", "")).strip()
    if colours:
        background = colours.split(";")[-1]
        if background.isdigit() and int(background) in {7, 15}:
            return "light"
    return "dark"


def build_styles(curses, theme: str = "dark") -> dict[str, int]:
    """Style name to a curses attribute, resolved once per session."""
    plain = {
        "text": curses.A_NORMAL,
        "dim": curses.A_DIM,
        "label": curses.A_DIM,
        "value": curses.A_NORMAL,
        "section": curses.A_BOLD,
        "rule": curses.A_DIM,
        "accent": curses.A_BOLD,
        "ok": curses.A_NORMAL,
        "warn": curses.A_BOLD,
        "bad": curses.A_BOLD,
        "diagnosis": curses.A_BOLD,
        "key": curses.A_REVERSE,
        "state.now": curses.A_BOLD,
        "bar": curses.A_REVERSE | curses.A_BOLD,
        "bar.dim": curses.A_REVERSE | curses.A_BOLD,
        "pill.ok": curses.A_REVERSE | curses.A_BOLD,
        "pill.warn": curses.A_REVERSE | curses.A_BOLD,
        "pill.bad": curses.A_REVERSE | curses.A_BOLD,
        "pill.accent": curses.A_REVERSE | curses.A_BOLD,
        "pill.dim": curses.A_REVERSE | curses.A_BOLD,
    }
    if not curses.has_colors():
        return plain
    curses.start_color()
    try:
        curses.use_default_colors()
    except curses.error:
        return plain
    wide = curses.COLORS >= 256
    light = str(theme).lower() == "light"
    if wide:
        source = PALETTE_256_LIGHT if light else PALETTE_256_DARK
    else:
        source = PALETTE_8_LIGHT if light else PALETTE_8_DARK
    pairs: dict[str, int] = {}
    for index, (name, colour) in enumerate(source.items(), start=1):
        value = colour if wide else getattr(curses, f"COLOR_{colour}")
        try:
            curses.init_pair(index, value, -1)
        except curses.error:
            continue
        pairs[name] = curses.color_pair(index)
    styles = dict(plain)
    styles["text"] = pairs.get("value", curses.A_NORMAL)
    styles["value"] = pairs.get("value", curses.A_NORMAL) | curses.A_BOLD
    styles["dim"] = pairs.get("dim", curses.A_DIM)
    styles["label"] = pairs.get("dim", curses.A_DIM)
    styles["rule"] = pairs.get("rule", curses.A_DIM)
    styles["section"] = pairs.get("section", 0) | curses.A_BOLD
    styles["accent"] = pairs.get("accent", 0) | curses.A_BOLD
    styles["ok"] = pairs.get("ok", 0)
    styles["warn"] = pairs.get("warn", 0) | curses.A_BOLD
    styles["bad"] = pairs.get("bad", 0) | curses.A_BOLD
    styles["diagnosis"] = pairs.get("diagnosis", 0) | curses.A_BOLD
    styles["key"] = pairs.get("key", 0) | curses.A_REVERSE
    styles["state.now"] = pairs.get("value", 0) | curses.A_BOLD
    styles["bar"] = pairs.get("accent", 0) | curses.A_REVERSE | curses.A_BOLD
    styles["bar.dim"] = pairs.get("accent", 0) | curses.A_REVERSE | curses.A_BOLD
    for tone in ("ok", "warn", "bad", "accent", "dim"):
        styles[f"pill.{tone}"] = pairs.get(tone, 0) | curses.A_REVERSE | curses.A_BOLD
    return styles
