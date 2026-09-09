"""The operator manual is generated, so it cannot drift from the keys."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "build_operator_manual.py"
MANUAL = REPO_ROOT / "docs" / "operator_manual.md"


def test_the_manual_on_disk_matches_the_console_bindings():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_the_manual_names_the_rehearsal_and_the_damp_end():
    text = MANUAL.read_text()

    assert "Rehearse first" in text
    assert "limp under the vendor" in text
    # Ctrl-D is the damp chord, and the manual must not teach SPACE.
    assert "Ctrl-D" in text
