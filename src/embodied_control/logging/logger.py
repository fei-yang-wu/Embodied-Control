"""``EcLogger``: the structured logger instance passed across host objects.

Wraps a stdlib ``logging.Logger``. One root logger is built per run via
:meth:`EcLogger.create`, named ``ec.run.<run_id>`` so concurrent/sequential
runs in the same process never share or leak handlers. Every component that
needs to log receives its own view via :meth:`EcLogger.child`, e.g.
``logger.child("sim")`` -> ``ec.run.<run_id>.sim`` -- a real stdlib child
logger that propagates up to the root's console/file/JSONL handlers, so no
handler wiring is needed anywhere below the root.

Components default to :meth:`EcLogger.null` (a stdlib pattern: a logger with
only a ``NullHandler`` attached) when no logger is injected, so call sites
never need ``if logger is not None`` guards.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from embodied_control.logging.config import LogConfig
from embodied_control.logging.timeutil import utcnow_iso

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

_HUMAN_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

_NULL_LOGGER_NAME = "embodied_control.null"


class _HumanFormatter(logging.Formatter):
    """Base human format, with any structured ``phase``/``fields`` appended."""

    def __init__(self) -> None:
        super().__init__(_HUMAN_FORMAT)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        bits = []
        phase = getattr(record, "phase", "")
        if phase:
            bits.append(f"phase={phase}")
        fields = getattr(record, "fields", None)
        if fields:
            bits.extend(f"{k}={v}" for k, v in fields.items())
        return f"{base} ({' '.join(bits)})" if bits else base


class _JsonlEventHandler(logging.Handler):
    """Appends one self-describing JSON row per record to a ``.jsonl`` file."""

    def __init__(self, path: Path, run_id: str):
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a")
        self.run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:
        row = {
            "ts": utcnow_iso(),
            "run_id": self.run_id,
            "logger": record.name,
            "level": record.levelname,
            "event": getattr(record, "event", record.getMessage()),
            "phase": getattr(record, "phase", ""),
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            row.update(fields)
        self._fh.write(json.dumps(row, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        finally:
            super().close()


class EcLogger:
    def __init__(self, logger: logging.Logger, run_id: str = ""):
        self._logger = logger
        self.run_id = run_id

    @classmethod
    def create(cls, config: LogConfig, run_id: str) -> "EcLogger":
        logger = logging.getLogger(f"ec.run.{run_id}")
        logger.setLevel(_LEVELS[config.level])
        logger.propagate = False
        # run_id is unique per run (planner dedupes run_dir), so this is a
        # fresh logger; clear defensively in case a run_id is ever reused
        # within a process (e.g. tests) so handlers/files are not duplicated.
        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

        formatter = _HumanFormatter()

        if config.console:
            console = logging.StreamHandler(sys.stderr)
            console.setLevel(_LEVELS[config.console_level or config.level])
            console.setFormatter(formatter)
            logger.addHandler(console)

        if config.file:
            if not config.log_dir:
                raise ValueError("LogConfig.file=True requires log_dir")
            fh = logging.FileHandler(Path(config.log_dir) / config.file_name)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if config.json_events:
            if not config.log_dir:
                raise ValueError("LogConfig.json_events=True requires log_dir")
            logger.addHandler(
                _JsonlEventHandler(Path(config.log_dir) / config.events_file_name, run_id)
            )

        return cls(logger, run_id=run_id)

    @classmethod
    def null(cls) -> "EcLogger":
        """A logger that discards everything -- the default when none is injected."""
        logger = logging.getLogger(_NULL_LOGGER_NAME)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        logger.propagate = False
        return cls(logger, run_id="")

    def child(self, component: str) -> "EcLogger":
        return EcLogger(self._logger.getChild(component), run_id=self.run_id)

    # --- plain logging -----------------------------------------------------
    def debug(self, msg: str, **fields) -> None:
        self._logger.debug(msg, extra={"fields": fields} if fields else None)

    def info(self, msg: str, **fields) -> None:
        self._logger.info(msg, extra={"fields": fields} if fields else None)

    def warning(self, msg: str, **fields) -> None:
        self._logger.warning(msg, extra={"fields": fields} if fields else None)

    def error(self, msg: str, **fields) -> None:
        self._logger.error(msg, extra={"fields": fields} if fields else None)

    # --- structured lifecycle events ----------------------------------------
    def event(self, event_type: str, phase: str = "", severity: str = "info", **fields) -> None:
        level = _LEVELS.get(severity.upper(), logging.INFO)
        self._logger.log(level, event_type, extra={"event": event_type, "phase": phase, "fields": fields})

    def close(self) -> None:
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
            h.close()
