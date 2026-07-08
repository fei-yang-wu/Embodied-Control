"""Structured, configurable logging (IsaacLab-style) for the host orchestrator.

Two outputs per run, both under the run's ``logs/`` artifact directory:

- ``orchestrator.log``: human-readable text, one line per event.
- ``events.jsonl``: the same events as structured JSON rows for automation.

A single ``EcLogger`` is constructed once per run (:func:`EcLogger.create`) and
handed to every component via constructor injection; components obtain their
own scoped view with ``logger.child("sim")`` etc. Children are plain stdlib
child loggers that propagate to the root's handlers, so no extra wiring is
needed below the root.
"""

from embodied_control.logging.config import LogConfig
from embodied_control.logging.logger import EcLogger
from embodied_control.logging.timeutil import utcnow_iso

__all__ = ["EcLogger", "LogConfig", "utcnow_iso"]
