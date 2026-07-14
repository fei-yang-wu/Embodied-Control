"""Shared orchestration failure type (fail-closed: every raise carries a phase)."""

from __future__ import annotations


class RunFailure(RuntimeError):
    def __init__(self, phase: str, reason: str):
        super().__init__(f"[{phase}] {reason}")
        self.phase = phase
        self.reason = reason
