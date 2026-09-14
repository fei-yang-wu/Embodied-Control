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
    # Isaac-style smoothness and actuator cost from the plant rows.
    for key in ("body_acc_mps2", "body_jerk_mps3", "tracking_acceleration_distance_mps2",
                "action_delta_l2", "joint_torques_l2", "energy_consumption"):
        assert row[key] == row[key] and row[key] > 0.0, key
    assert row["body_jerk_mps3"] > row["body_acc_mps2"]
    # Joint-limit columns: targets past the training limit are ordinary and
    # measured in percent; the measured excursion is what the writer faults on.
    assert 0.0 <= row["target_beyond_limit_pct"] <= 100.0
    assert 0.0 <= row["ankle_target_beyond_limit_pct"] <= 100.0
    assert row["ankle_target_excess_max_rad"] >= 0.0
    assert 0.0 <= row["measured_excess_max_rad"] < 0.2
    assert aggregate["all_max"]["measured_excess_max_rad"] >= row["measured_excess_max_rad"]
    assert (path / "grade.tsv").is_file()
