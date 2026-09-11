"""Rehearsal grading from the plant's true state (needs a local rehearsal)."""

from pathlib import Path

import pytest

pytest.importorskip("mujoco")

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "output, expected_source",
    [
        ("artifacts/anchor_modes_leg_kinematics", "leg_kinematics"),
        ("artifacts/anchor_modes_fixed_start", "fixed_start"),
    ],
)
def test_grade_reports_source_and_finite_metrics(output, expected_source):
    from embodied_control.robot.rehearsal_grade import grade_rehearsal

    path = ROOT / output
    if not (path / "summary.json").is_file():
        pytest.skip(f"no rehearsal at {path}")
    aggregate = grade_rehearsal(path)
    assert aggregate["motions"] >= 1
    assert aggregate["anchor_position_sources"] == [expected_source]
    row = aggregate["rows"][0]
    assert row["reference_ticks"] > 100
    assert 5.0 < row["mpjpe_l_mm"] < 100.0
    assert row["mpjpe_g_mm"] >= row["mpjpe_l_mm"]
    assert (path / "grade.tsv").is_file()
