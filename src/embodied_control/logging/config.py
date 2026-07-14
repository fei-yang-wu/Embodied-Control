"""Configuration for the base logger.

Every field here is a knob a job/CLI invocation can set — this is the
"configurable base logger" the orchestrator constructs once per run.

Deliberately a plain dataclass, not a pydantic model: ``EcLogger`` (and the
``PolicyClient`` it can be attached to) must stay stdlib-only so it can be
reused inside minimal, non-pip-installed containers -- e.g. the fake delegated
evaluator, which imports ``transport.client.PolicyClient`` -> ``EcLogger`` ->
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


@dataclass
class LogConfig:
    level: LogLevel = "INFO"

    console: bool = True
    console_level: LogLevel | None = None  # defaults to `level` when unset

    file: bool = True
    file_name: str = "orchestrator.log"

    json_events: bool = True
    events_file_name: str = "events.jsonl"

    log_dir: str | None = None  # required when file or json_events is True
