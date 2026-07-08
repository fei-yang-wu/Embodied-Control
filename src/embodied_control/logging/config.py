"""Configuration for the base logger.

Every field here is a knob a job/CLI invocation can set — this is the
"configurable base logger" the orchestrator constructs once per run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class LogConfig(BaseModel):
    level: LogLevel = "INFO"

    console: bool = True
    console_level: LogLevel | None = None  # defaults to `level` when unset

    file: bool = True
    file_name: str = "orchestrator.log"

    json_events: bool = True
    events_file_name: str = "events.jsonl"

    log_dir: str | None = None  # required when file or json_events is True
