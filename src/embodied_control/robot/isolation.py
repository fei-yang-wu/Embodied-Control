"""Keep the operator console off the cores the robot runs on.

The 50 Hz control thread and the 500 Hz writer are C++ threads inside this
process, pinned to their own cores at SCHED_FIFO 80 and 90. A SCHED_OTHER
thread that lands on one of those cores does not slow the robot down; the
robot starves the thread, and the thread here is the display and the key
handler. So the console's own threads are pinned to everything else, and
child processes (the planner, the diagnostic agent) inherit the same mask.

The planner is not renice'd: it answers the control loop inside a deadline.
The diagnostic agent is, hard, because it is an LLM CLI that will otherwise
take every core it can while a robot is standing.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterable


def available_cores() -> set[int]:
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # pragma: no cover - not Linux
        count = os.cpu_count() or 1
        return set(range(count))


def non_realtime_cores(reserved: Iterable[int]) -> set[int]:
    """Every core we may use, minus the ones the robot threads are pinned to.

    Returns the full set unchanged when the reservation would leave nothing:
    a console with no core at all is worse than a shared one.
    """
    cores = available_cores()
    free = cores - {int(core) for core in reserved if int(core) >= 0}
    return free or cores


def pin_current_thread(cores: Iterable[int]) -> bool:
    """Pin the calling thread (pid 0 is this thread on Linux)."""
    mask = {int(core) for core in cores}
    if not mask:
        return False
    try:
        os.sched_setaffinity(0, mask)
        return True
    except (AttributeError, OSError):  # pragma: no cover - not Linux
        return False


PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:  # pragma: no cover - runs in the forked child
    """Ask the kernel to SIGTERM this child when the spawning thread dies.

    The children run in their own session so a stray Ctrl-C in the operator's
    shell cannot reach them, which also means a console killed with SIGKILL
    would leave an oracle worker holding the shared-memory slots forever. This
    closes that hole from the other side.
    """
    import ctypes
    import signal

    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(
            PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0
        )
    except OSError:
        pass


def child_preexec(
    cores: Iterable[int], nice: int = 0, die_with_parent: bool = True
) -> Callable[[], None] | None:
    """A `preexec_fn` that puts a child on `cores`, optionally renice'd."""
    mask = {int(core) for core in cores}
    if not mask and nice <= 0 and not die_with_parent:
        return None

    def apply() -> None:  # pragma: no cover - runs in the forked child
        if die_with_parent:
            _die_with_parent()
        if mask:
            try:
                os.sched_setaffinity(0, mask)
            except OSError:
                pass
        if nice > 0:
            try:
                os.nice(nice)
            except OSError:
                pass

    return apply


class ThreadPinner:
    """Pin whichever thread calls `apply`, once per thread."""

    def __init__(self, cores: Iterable[int]) -> None:
        self.cores = {int(core) for core in cores}
        self._done: set[int] = set()
        self._lock = threading.Lock()

    def apply(self) -> bool:
        native = threading.get_ident()
        with self._lock:
            if native in self._done:
                return True
            self._done.add(native)
        return pin_current_thread(self.cores)

    def describe(self) -> str:
        return "cores " + ",".join(str(core) for core in sorted(self.cores))
