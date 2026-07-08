"""Timestamp helper shared by the logger and artifact schemas."""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
