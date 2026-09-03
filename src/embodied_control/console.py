"""Single-keypress operator console.

An operator console is not a place to type words. Damp is a safety stop: it has
to be one key, reachable without looking, and it must work while the terminal is
mid-redraw. So keys dispatch immediately in cbreak mode with no Enter, and the
write gate is taken once at process start rather than per keystroke.

The dispatch table is deliberately separate from the terminal I/O: `handle` is
pure and testable, `run` is the part that needs a tty.
"""

from __future__ import annotations

import select
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass

KEY_ESCAPE = "\x1b"
KEY_SPACE = " "
KEY_UP = "<up>"
KEY_DOWN = "<down>"
KEY_LEFT = "<left>"
KEY_RIGHT = "<right>"

_ARROWS = {"A": KEY_UP, "B": KEY_DOWN, "C": KEY_RIGHT, "D": KEY_LEFT}


@dataclass(frozen=True)
class KeyBinding:
    key: str
    label: str
    action: Callable[[], None]
    group: str = "general"


class ConsoleQuit(Exception):
    """Raised by a binding to end the console loop."""


class KeyConsole:
    def __init__(
        self,
        bindings: list[KeyBinding],
        *,
        status: Callable[[], str] | None = None,
        out=None,
    ) -> None:
        self._bindings = {binding.key: binding for binding in bindings}
        self._order = list(bindings)
        self._status = status
        self._out = out if out is not None else sys.stdout

    def _print(self, text: str = "") -> None:
        print(text, file=self._out, flush=True)

    def help(self) -> None:
        self._print("")
        current = ""
        for binding in self._order:
            if binding.group != current:
                current = binding.group
                self._print(f"  [{current}]")
            shown = {KEY_SPACE: "SPACE"}.get(binding.key, binding.key)
            self._print(f"    {shown:<8} {binding.label}")
        self._print("    ?        this help")
        self._print("")

    def show_status(self) -> None:
        if self._status is not None:
            self._print(f"  {self._status()}")

    def handle(self, key: str) -> bool:
        """Dispatch one key. Returns False when the console should exit."""
        if key == "?":
            self.help()
            return True
        binding = self._bindings.get(key)
        if binding is None:
            return True
        try:
            binding.action()
        except ConsoleQuit:
            return False
        except Exception as exc:  # operator console: report, never die
            self._print(f"  !! {type(exc).__name__}: {exc}")
            return True
        self.show_status()
        return True

    def run(self, keys: Iterator[str] | None = None) -> int:
        self.help()
        self.show_status()
        source = keys if keys is not None else read_keys()
        for key in source:
            if not self.handle(key):
                break
        return 0


def read_keys(stream=None) -> Iterator[str]:
    """Yield single keypresses, decoding arrow escape sequences.

    Falls back to line-at-a-time when stdin is not a tty, so the console stays
    drivable from a pipe or a test.
    """
    import termios
    import tty

    source = stream if stream is not None else sys.stdin
    if not source.isatty():
        for line in source:
            token = line.strip()
            if token:
                yield token
        return

    descriptor = source.fileno()
    saved = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        while True:
            char = source.read(1)
            if not char:
                return
            if char == KEY_ESCAPE:
                ready, _, _ = select.select([source], [], [], 0.05)
                if not ready:
                    yield KEY_ESCAPE
                    continue
                if source.read(1) != "[":
                    continue
                yield _ARROWS.get(source.read(1), "")
                continue
            yield char
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)
