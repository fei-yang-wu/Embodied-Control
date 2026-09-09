"""Read-only coding-agent bridge for operator-console diagnostics."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


PROVIDERS = ("auto", "codex", "claude", "off")


@dataclass(frozen=True)
class DiagnosticResult:
    provider: str
    text: str


class DiagnosticAgent:
    """Run one bounded, non-interactive, read-only diagnostic turn."""

    def __init__(
        self,
        provider: str = "auto",
        *,
        cwd: str | Path,
        timeout_seconds: float = 90.0,
        which: Callable[[str], str | None] = shutil.which,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
        preexec: Callable[[], None] | None = None,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown diagnostic agent {provider!r}")
        self.cwd = Path(cwd).resolve()
        self.timeout_seconds = timeout_seconds
        self._run = run
        self._preexec = preexec
        self.provider, self.executable = self._resolve(provider, which)

    @staticmethod
    def _resolve(
        requested: str, which: Callable[[str], str | None]
    ) -> tuple[str, str | None]:
        if requested == "off":
            return "off", None
        candidates = (requested,) if requested != "auto" else ("codex", "claude")
        for candidate in candidates:
            executable = which(candidate)
            if executable:
                return candidate, executable
        return requested, None

    @property
    def available(self) -> bool:
        return self.executable is not None

    @property
    def label(self) -> str:
        return self.provider if self.available else "unavailable"

    def diagnose(self, snapshot: dict, notes: Sequence[str]) -> DiagnosticResult:
        if not self.available:
            requested = (
                "Codex or Claude Code" if self.provider == "auto" else self.provider
            )
            raise RuntimeError(f"{requested} CLI not found")
        prompt = diagnostic_prompt(snapshot, notes)
        command = self._command()
        try:
            completed = self._run(
                command,
                cwd=self.cwd,
                input=prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                preexec_fn=self._preexec,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"{self.provider} diagnosis timed out after {self.timeout_seconds:.0f}s"
            ) from exc
        output = completed.stdout.strip()
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            reason = detail[-1] if detail else f"exit code {completed.returncode}"
            raise RuntimeError(f"{self.provider} diagnosis failed: {reason}")
        if not output:
            raise RuntimeError(f"{self.provider} diagnosis returned no text")
        return DiagnosticResult(self.provider, output[:12_000])

    def _command(self) -> list[str]:
        assert self.executable is not None
        if self.provider == "codex":
            return [
                self.executable,
                "exec",
                "--sandbox",
                "read-only",
                "--ephemeral",
                "--ignore-user-config",
                "--color",
                "never",
                "--cd",
                str(self.cwd),
                "-",
            ]
        return [
            self.executable,
            "--print",
            "--restricted",
            "--strict-mcp-config",
            "--tools",
            "Read,Grep,Glob",
            "--permission-mode",
            "dontAsk",
            "--permission-prompts",
            "none",
            "--no-session-persistence",
            "--output-format",
            "text",
        ]


def diagnostic_prompt(snapshot: dict, notes: Sequence[str]) -> str:
    """Keep agent context useful, small, and free of opaque native arrays."""
    context = {
        "state": snapshot.get("state"),
        "fault_reason": snapshot.get("fault_reason"),
        "last_ok": snapshot.get("last_ok"),
        "last_detail": snapshot.get("last_detail"),
        "writer": snapshot.get("writer", {}),
        "control": snapshot.get("control", {}),
        "hoist": snapshot.get("hoist"),
        "session": snapshot.get("session"),
        "recent_console_log": list(notes)[-30:],
    }
    return (
        "You are embedded in the embodied-control robot operator console as a "
        "read-only diagnostic assistant. Diagnose the current failure or abnormal "
        "runtime state. You may inspect this repository when needed. Do not edit "
        "files, run robot commands, start services, or suggest bypassing a safety "
        "gate. Separate confirmed evidence from hypotheses. Return concise sections "
        "named Cause, Evidence, and Safe next checks. If evidence is insufficient, "
        "say what operator-visible measurement is missing.\n\n"
        "Runtime context:\n"
        + json.dumps(context, indent=2, default=str)
    )
