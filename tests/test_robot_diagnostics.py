from pathlib import Path
from subprocess import CompletedProcess

import pytest

from embodied_control.robot.diagnostics import DiagnosticAgent, diagnostic_prompt


def test_auto_prefers_codex_and_uses_stdin_in_read_only_ephemeral_mode(tmp_path):
    seen = {}

    def which(name):
        return f"/bin/{name}" if name in {"codex", "claude"} else None

    def run(command, **kwargs):
        seen["command"] = command
        seen["kwargs"] = kwargs
        return CompletedProcess(command, 0, "Cause\nplanner is stale\n", "")

    agent = DiagnosticAgent("auto", cwd=tmp_path, which=which, run=run)
    result = agent.diagnose({"state": "FAULT"}, ["timeout"])

    assert result.provider == "codex"
    assert result.text == "Cause\nplanner is stale"
    assert seen["command"] == [
        "/bin/codex",
        "exec",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "--ignore-user-config",
        "--color",
        "never",
        "--cd",
        str(Path(tmp_path).resolve()),
        "-",
    ]
    assert "FAULT" in seen["kwargs"]["input"]
    assert "shell" not in seen["kwargs"]


def test_claude_is_restricted_to_read_tools(tmp_path):
    seen = {}

    def run(command, **kwargs):
        seen["command"] = command
        return CompletedProcess(command, 0, "Evidence\nok", "")

    agent = DiagnosticAgent(
        "claude", cwd=tmp_path, which=lambda _: "/bin/claude", run=run
    )
    agent.diagnose({}, [])
    command = seen["command"]
    assert "--restricted" in command
    assert "--strict-mcp-config" in command
    assert "Read,Grep,Glob" in command
    assert "--permission-mode" in command
    assert "dontAsk" in command
    assert "--no-session-persistence" in command


def test_missing_agent_and_failed_agent_are_reported(tmp_path):
    missing = DiagnosticAgent("codex", cwd=tmp_path, which=lambda _: None)
    with pytest.raises(RuntimeError, match="CLI not found"):
        missing.diagnose({}, [])

    failed = DiagnosticAgent(
        "codex",
        cwd=tmp_path,
        which=lambda _: "/bin/codex",
        run=lambda command, **kwargs: CompletedProcess(command, 1, "", "auth failed\n"),
    )
    with pytest.raises(RuntimeError, match="auth failed"):
        failed.diagnose({}, [])


def test_prompt_contains_only_recent_log_entries():
    prompt = diagnostic_prompt(
        {"state": "FAULT", "opaque": [1, 2, 3]}, [str(i) for i in range(40)]
    )
    assert '"state": "FAULT"' in prompt
    assert '"opaque"' not in prompt
    assert '"recent_console_log"' in prompt
    assert '"0"' not in prompt
    assert '"10"' in prompt
