"""The full-screen console, rendered without a terminal."""

from __future__ import annotations

from embodied_control.console import (
    KEY_BACKSPACE,
    KEY_DAMP,
    KEY_DELETE,
    KEY_END,
    KEY_ENTER,
    KEY_ESCAPE,
    KEY_HOME,
    KEY_LEFT,
    KEY_SPACE,
    KEY_TAB,
    KEY_UP,
)
from embodied_control.robot import FakeRobotRuntime, RobotMode
from embodied_control.robot.lifecycle import (
    Lifecycle,
    LifecycleConfig,
    LifecycleState as S,
)
from embodied_control.robot.shell import build_lifecycle_bindings
from embodied_control.robot.diagnostics import DiagnosticResult
from embodied_control.robot.tui import DISPLAY_ORDER, LifecycleTui, render
from test_lifecycle import POSE, FakeClock, FakeHoist, StubTracker


def _lifecycle():
    clock = FakeClock()
    tracker = StubTracker(clock)
    vendor = FakeRobotRuntime(writes_enabled=True, mode=RobotMode.READY)
    lifecycle = Lifecycle(
        tracker,
        vendor,
        LifecycleConfig(start_pose=list(POSE), ticks=100, blend_ticks=50),
        hoist=FakeHoist(),
        auto_ack=True,
        now=clock.now,
        sleep=clock.sleep,
    )
    return lifecycle, tracker, clock


def test_render_marks_the_current_rung_and_fits_the_width():
    lifecycle, tracker, clock = _lifecycle()
    assert lifecycle.auto(S.POSE_SETTLED).ok
    rows = render(lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), ["hello"], width=90, height=40)
    assert all(len(row) == 90 for row in rows)
    text = "\n".join(rows)
    # The current rung is marked and numbered; the ones behind it are ticked.
    assert "▸  6 POSE SETTLED" in text
    assert "✓  5 START POSE RAMP" in text
    assert "·  7 LOWERED" in text
    assert "^D DAMP" in text
    assert " POSE_SETTLED " in text
    assert "vendor released" in text
    assert "hello" in text


def test_render_shows_fault_banner_and_busy_line():
    lifecycle, tracker, clock = _lifecycle()
    lifecycle.fault_reason = "writer damped during RUNNING"
    lifecycle.state = S.FAULT
    rows = render(
        lifecycle.snapshot(),
        build_lifecycle_bindings(lifecycle),
        [],
        busy="auto-advance to PRIMED",
        width=80,
        height=30,
    )
    text = "\n".join(rows)
    assert "! FAULT  writer damped during RUNNING" in text
    assert "auto-advance to PRIMED" in text
    assert len(rows) <= 30


def test_compact_terminal_keeps_emergency_control_and_command_prompt_visible():
    lifecycle, tracker, clock = _lifecycle()
    rows = render(
        lifecycle.snapshot(),
        build_lifecycle_bindings(lifecycle),
        [],
        width=72,
        height=16,
    )
    text = "\n".join(rows)
    assert len(rows) <= 16
    assert "^D DAMP" in text
    assert "Type / for commands" in text


def test_display_order_covers_every_non_sink_state():
    shown = set(DISPLAY_ORDER)
    for state in S:
        if state in {S.IDLE, S.FAULT}:
            continue
        assert state in shown, state


def test_damp_chord_bypasses_the_lock_and_space_is_inert():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    assert tui.handle(KEY_DAMP) is True
    # The writer got the damp before any worker ran.
    assert tracker.calls[-1] == "force_damp"
    assert tui._queue.qsize() == 1
    assert any("^D" in note for note in tui.notes)

    # SPACE is the one key an operator hits by accident; it must do nothing.
    assert tui.handle(KEY_SPACE) is True
    assert tui._queue.qsize() == 1
    assert any("SPACE does nothing" in note for note in tui.notes)


def test_quit_is_refused_while_busy_and_unknown_keys_are_ignored():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    assert tui.handle("~") is True
    assert tui._queue.qsize() == 0
    tui._busy = "next: advance one state"
    assert tui.handle("q") is True
    assert tui.handle("n") is True
    assert tui._queue.qsize() == 0
    tui._busy = ""
    assert tui.handle("q") is False


def test_frame_renders_from_a_live_lifecycle():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    tui.note("started")
    rows = tui.frame(100, 40)
    assert rows and "G1 LIFECYCLE" in rows[0]
    assert any("started" in row for row in rows)


def test_slash_commands_dispatch_bindings_and_render_help():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    for key in "/next":
        assert tui.handle(key) is True
    assert tui.handle(KEY_ENTER) is True
    assert tui._queue.qsize() == 1
    for key in "/help":
        assert tui.handle(key) is True
    assert tui.handle(KEY_ENTER) is True
    text = "\n".join(tui.frame(100, 40))
    assert "COMMAND PALETTE" in text
    assert "/damp" in text
    assert "Ctrl-D always damps" in text


def test_damp_chord_wins_over_a_half_typed_command():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    tui.handle("/")
    tui.handle("a")
    assert tui.handle(KEY_DAMP) is True
    assert tui._line is None
    assert tracker.calls[-1] == "force_damp"


def test_prompt_edits_completes_and_recalls():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    tui.handle("/")
    for key in "sttaus":
        tui.handle(key)
    # Backspace deletes; it used to arrive as KEY_BACKSPACE and get inserted.
    for _ in range(5):
        tui.handle(KEY_BACKSPACE)
    for key in "tatus":
        tui.handle(key)
    assert tui._line.text == "status"
    tui.handle(KEY_HOME)
    tui.handle(KEY_DELETE)
    assert tui._line.text == "tatus" and tui._line.cursor == 0
    tui.handle("s")
    tui.handle(KEY_END)
    assert tui._line.text == "status"
    assert tui.handle(KEY_ENTER) is True
    assert tui._line is None
    assert any("state IDLE" in note for note in tui.notes)

    # The prompt is gone but its history is not.
    tui.handle("/")
    tui.handle(KEY_UP)
    assert tui._line.text == "status"
    tui.handle(KEY_ESCAPE)
    assert tui._line is None

    # TAB completes to the only match and shows it as ghost text first.
    tui.handle("/")
    for key in "rel":
        tui.handle(key)
    assert tui._line.ghost() == "ease"
    tui.handle(KEY_TAB)
    assert tui._line.text == "release"


def test_backspace_on_an_empty_prompt_closes_it():
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now)
    tui.handle("/")
    tui.handle("a")
    tui.handle(KEY_BACKSPACE)
    assert tui._line is not None and tui._line.text == ""
    tui.handle(KEY_BACKSPACE)
    assert tui._line is None


def test_prompt_shows_candidates_and_flags_an_unknown_command():
    from embodied_control.robot.tui import CommandLine, render

    lifecycle, tracker, clock = _lifecycle()
    line = CommandLine(["go", "hold", "help"])
    for key in "h":
        line.handle(key)
    text = "\n".join(render(
        lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), [],
        width=100, command=line,
    ))
    assert "/hold" in text and "/help" in text
    line.handle("z")
    text = "\n".join(render(
        lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), [],
        width=100, command=line,
    ))
    assert "no command matches" in text


def test_line_editor_kill_keys():
    from embodied_control.robot.tui import CommandLine

    line = CommandLine(["rebuild"])
    for key in "frame-next":
        line.handle(key)
    line.handle("\x17")  # ctrl-w
    assert line.text == "frame-"
    line.handle("\x15")  # ctrl-u
    assert line.text == "" and line.cursor == 0
    for key in "abcd":
        line.handle(key)
    line.handle(KEY_LEFT)
    line.handle("\x0b")  # ctrl-k
    assert line.text == "abc"


def test_the_prompt_keeps_the_frame_the_same_shape_however_it_is_typed():
    """A footer that grows when a command matches moves every row above it.

    ncurses realises that shift with an insert-line, and on a short terminal
    the bottom row falls off and the prompt is drawn inside the key legend
    (reproduced in a pty at 24 rows). The prompt therefore reserves its
    completion row whether it is closed, empty, matching or not.
    """
    lifecycle, tracker, clock = _lifecycle()
    assert lifecycle.auto(S.POSE_SETTLED).ok
    snapshot, bindings = lifecycle.snapshot(), build_lifecycle_bindings(lifecycle)
    for height in (22, 24, 30, 40):
        shapes = set()
        for command in (None, "", "s", "zzz"):
            rows = render(
                snapshot, bindings, ["one", "two"],
                width=100, height=height, command=command,
            )
            controls = next(i for i, row in enumerate(rows) if "CONTROLS" in row)
            shapes.add((len(rows), controls))
        assert len(shapes) == 1, (height, shapes)


def test_diagnosis_gets_snapshot_and_has_its_own_panel():
    lifecycle, tracker, clock = _lifecycle()
    seen = {}

    def diagnose(snapshot, notes):
        seen["state"] = snapshot["state"]
        seen["notes"] = notes
        return DiagnosticResult("codex", "Cause\nA stale command\n\nSafe next checks\nInspect planner")

    tui = LifecycleTui(
        lifecycle,
        build_lifecycle_bindings(lifecycle),
        clock=clock.now,
        diagnose=diagnose,
        agent_label="codex",
    )
    tui.note("planner timed out")
    tui._run_diagnosis()
    text = "\n".join(tui.frame(100, 40))
    assert seen["state"] == "IDLE"
    assert any("planner timed out" in note for note in seen["notes"])
    assert "─ DIAGNOSIS" in text
    assert "codex · read-only" in text
    assert "A stale command" in text


def test_bars_and_sparklines_are_exact():
    from embodied_control.robot.tui import bar_cells, gauge, sparkline

    assert bar_cells(0.0, 4) == "····"
    assert bar_cells(1.0, 4) == "████"
    assert bar_cells(0.5, 4) == "██··"
    # Eighths, so a bar moves before a whole cell is earned.
    assert bar_cells(0.55, 4) == "██▎·"
    assert sparkline([], 5) == "     "
    assert sparkline([1, 2, 3], 5) == "  ▁▄█"
    assert sparkline([2, 2, 2], 3) == "▄▄▄"
    assert gauge(0.1, 1.0)[1] == "ok"
    assert gauge(0.7, 1.0)[1] == "warn"
    assert gauge(1.4, 1.0)[1] == "bad"


def test_rows_are_styled_spans_that_pad_to_the_width():
    from embodied_control.robot.tui import Span, flatten, pad, render_rows

    lifecycle, tracker, clock = _lifecycle()
    rows = render_rows(
        lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), [], width=100
    )
    assert all(len(flatten(row)) == 100 for row in rows)
    assert all(isinstance(span, Span) for row in rows for span in row)
    assert len(flatten(pad([Span("abcdef")], 4))) == 4


def test_lifecycle_header_and_current_state_are_fully_bold():
    from embodied_control.robot.tui import build_styles, paint_rows, render_rows

    class FakeCurses:
        A_NORMAL = 0
        A_DIM = 1
        A_BOLD = 2
        A_REVERSE = 4

        @staticmethod
        def has_colors():
            return False

    class Screen:
        def __init__(self):
            self.calls = []

        def addnstr(self, row, column, text, count, attribute):
            self.calls.append((row, column, text, count, attribute))

    lifecycle, tracker, clock = _lifecycle()
    assert lifecycle.auto(S.POSE_SETTLED).ok
    rows = render_rows(
        lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), [],
        width=100, height=40,
    )
    styles = build_styles(FakeCurses)
    screen = Screen()
    paint_rows(screen, rows, 100, 40, styles, RuntimeError)
    header = [call for call in screen.calls if call[0] == 0 and call[2].strip()]
    assert header and all(call[4] & FakeCurses.A_BOLD for call in header)
    current = [call for call in screen.calls if "POSE SETTLED" in call[2]]
    assert len(current) == 1
    assert current[0][2].strip() == "POSE SETTLED"
    assert current[0][4] & FakeCurses.A_BOLD


def test_many_multiline_logs_stay_inside_screen_without_cursor_controls():
    from embodied_control.robot.tui import paint_rows, render_rows

    lifecycle, tracker, clock = _lifecycle()
    notes = [
        f"line {index}\ncontinuation\t\x1b[31mred\x1b[0m"
        for index in range(400)
    ]
    rows = render_rows(
        lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), notes,
        width=72, height=30,
    )
    assert len(rows) <= 30
    text = "\n".join("".join(span.text for span in row) for row in rows)
    assert "line 399" in text and "continuation" in text
    assert "\x1b" not in text and "\t" not in text and "\r" not in text

    class Screen:
        def __init__(self):
            self.calls = []

        def addnstr(self, row, column, value, count, attribute):
            self.calls.append((row, column, value, count, attribute))

    screen = Screen()
    paint_rows(screen, rows, 72, 30, {"text": 0, "dim": 1}, RuntimeError)
    occupied = {}
    for row, column, value, count, _ in screen.calls:
        assert 0 <= row < 30 and column + len(value) <= 72
        assert count == len(value)
        assert not any(char in value for char in "\n\r\t\x1b")
        assert column >= occupied.get(row, 0)
        occupied[row] = column + len(value)


def test_fault_and_links_carry_their_own_tone():
    from embodied_control.robot.tui import comm_rows, render_rows

    lifecycle, tracker, clock = _lifecycle()
    lifecycle.fault_reason = "writer damped during RUNNING"
    lifecycle.state = S.FAULT
    rows = render_rows(
        lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), [], width=120
    )
    styles = {span.style for row in rows for span in row}
    assert "bad" in styles and "pill.bad" in styles
    # One bad link does not colour its neighbours.
    snapshot = {"writer": {"crc_errors": 4}, "control": {}}
    row = comm_rows(snapshot, {"state_hz": 500.0, "publish_hz": 500.0}, None)[0]
    dots = [span.style for span in row if span.text.strip() in {"•", "!"}]
    assert dots == ["bad", "ok", "ok"]


def test_links_stack_when_the_terminal_is_narrow():
    from embodied_control.robot.tui import comm_rows

    snapshot = {"writer": {}, "control": {}}
    rates = {"state_hz": 500.0, "publish_hz": 500.0, "planner_hz": 5.0}
    assert len(comm_rows(snapshot, rates, None, width=140)) == 1
    assert len(comm_rows(snapshot, rates, None, width=90)) == 3


def test_short_terminal_drops_the_ladder_instead_of_half_drawing_it():
    lifecycle, tracker, clock = _lifecycle()
    text = "\n".join(
        render(lifecycle.snapshot(), build_lifecycle_bindings(lifecycle), [],
               width=100, height=14)
    )
    assert "─ TELEMETRY" not in text
    assert "^D DAMP" in text


def test_console_threads_are_pinned_once_each():
    """`apply` pins the calling thread, so the test restores its own mask:
    a child process inherits the affinity of the thread that spawned it, and
    a pinned pytest runner slows every subprocess test after this one."""
    import os

    from embodied_control.robot.isolation import ThreadPinner, non_realtime_cores

    before = os.sched_getaffinity(0)
    try:
        pinner = ThreadPinner(non_realtime_cores((0,)))
        assert pinner.apply() is True
        assert pinner.apply() is True  # once per thread, not once per call
        assert os.sched_getaffinity(0) == pinner.cores
    finally:
        os.sched_setaffinity(0, before)
    assert ThreadPinner({0}).describe() == "cores 0"

    calls = []
    lifecycle, tracker, clock = _lifecycle()
    tui = LifecycleTui(
        lifecycle, build_lifecycle_bindings(lifecycle), clock=clock.now,
        pin=lambda: calls.append("pinned"),
    )
    tui._quit.set()
    tui._work()
    assert calls == ["pinned"]


def test_theme_resolution_reads_the_terminal_then_the_override():
    from embodied_control.robot.tui import resolve_theme

    # An explicit request wins over anything the terminal says.
    assert resolve_theme("light", {}) == "light"
    assert resolve_theme("dark", {"COLORFGBG": "0;15"}) == "dark"
    # `auto` reads COLORFGBG: the last field is the background.
    assert resolve_theme("auto", {"COLORFGBG": "0;15"}) == "light"
    assert resolve_theme("auto", {"COLORFGBG": "0;7"}) == "light"
    assert resolve_theme("auto", {"COLORFGBG": "15;0"}) == "dark"
    # Unset, malformed, or a terminal that does not publish it: dark.
    assert resolve_theme("auto", {}) == "dark"
    assert resolve_theme("auto", {"COLORFGBG": "default;default"}) == "dark"
    # The environment override beats the flag, for a terminal that lies.
    assert resolve_theme("dark", {"EC_TUI_THEME": "light"}) == "light"
    assert resolve_theme("auto", {"EC_TUI_THEME": "nonsense"}) == "dark"


def _xterm_luminance(index: int) -> float:
    """Relative luminance, 0 black to 1 white, of an xterm-256 colour.

    The index is not a brightness: 16-231 is a 6x6x6 cube and 232-255 is a
    greyscale ramp, so 238 is nearly black while 44 is a bright cyan. A
    palette test has to compare the colours, not their numbers.
    """
    if index >= 232:
        level = (8 + (index - 232) * 10) / 255.0
        return level
    index -= 16
    steps = [0, 95, 135, 175, 215, 255]
    red = steps[index // 36] / 255.0
    green = steps[(index % 36) // 6] / 255.0
    blue = steps[index % 6] / 255.0
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def test_the_light_palette_inverts_the_contrast_it_needs_to():
    """`value` is what an operator reads and must carry the most contrast
    against the paper; reverse-video chips take their colour as a background,
    so on light they must be dark enough to hold white text."""
    from embodied_control.robot.tui import (
        PALETTE_256_DARK,
        PALETTE_256_LIGHT,
    )

    dark = {k: _xterm_luminance(v) for k, v in PALETTE_256_DARK.items()}
    light = {k: _xterm_luminance(v) for k, v in PALETTE_256_LIGHT.items()}

    # Near-white on a dark terminal, near-black on a light one.
    assert dark["value"] > 0.85
    assert light["value"] < 0.15
    # `dim` and `rule` step back from `value`, towards the paper.
    assert dark["dim"] < dark["value"] and dark["rule"] < dark["dim"]
    assert light["dim"] > light["value"] and light["rule"] > light["dim"]
    # `value` and `bad` carry the contrast that matters most: the number and
    # the failure. WCAG ratio against the paper, (1.05)/(L + 0.05).
    assert 1.05 / (light["value"] + 0.05) > 4.5
    assert 1.05 / (light["bad"] + 0.05) > 4.0
    # Every light-theme hue stays dark: readable as text on white, and dark
    # enough to be a chip behind white text. Green and amber bottom out around
    # 3:1 and 2.3:1 in the 6x6x6 cube, which is why they are always bold.
    for name in ("accent", "ok", "warn", "bad", "section", "key"):
        assert light[name] < 0.45, (name, light[name])
        assert 1.05 / (light[name] + 0.05) > 2.2, (name, light[name])
    # The dark theme's hues are bright for the same reason, reversed.
    for name in ("accent", "ok", "warn", "bad", "section"):
        assert dark[name] > 0.35, (name, dark[name])
    # The three tones stay apart from each other, or a red gauge reads green.
    for a, b in (("ok", "warn"), ("warn", "bad"), ("ok", "bad")):
        assert abs(light[a] - light[b]) > 0.05, (a, b)


def test_a_long_log_never_eats_the_ladder_on_a_thirty_row_terminal():
    """The log used to grow to eight rows and push the ladder's bottom rungs off."""
    lifecycle, tracker, clock = _lifecycle()
    assert lifecycle.auto(S.POSE_SETTLED).ok
    snapshot, bindings = lifecycle.snapshot(), build_lifecycle_bindings(lifecycle)
    quiet = render(snapshot, bindings, ["one"], width=100, height=30)
    noisy = render(snapshot, bindings, [f"note {i}" for i in range(200)], width=100, height=30)
    ladder = [row for row in quiet if "VENDOR_RESTORED" in row or "RUNNING" in row]
    assert ladder, quiet
    for row in ladder:
        assert row in noisy
    assert "note 199" in "\n".join(noisy)
