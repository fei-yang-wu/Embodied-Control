"""Plant configuration is a robot contract, with no checkpoint dependency."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from embodied_control.cli import build_parser
from embodied_control.robot.plant import PlantConfig, load_plant_config


PROFILE = Path(__file__).resolve().parents[1] / "examples/g1_plant.yaml"


def test_plant_profile_and_cli_need_only_robot_and_model():
    robot = load_plant_config(PROFILE)
    assert robot.joints[0].name == "left_hip_pitch_joint"
    assert [joint.motor_id for joint in robot.joints] == list(range(29))
    args = build_parser().parse_args([
        "lowlevel", "plant", str(PROFILE), "--model", "robot.xml", "--vendor", "--hoist",
    ])
    assert args.robot == str(PROFILE)
    assert not hasattr(args, "bundle")


@pytest.mark.parametrize("field,value", [
    ("motor_id", 1), ("name", "left_hip_roll_joint"), ("effort_limit", 0),
    ("armature", -1), ("vendor_stiffness", -1), ("vendor_damping", -1),
    ("nominal_position", float("nan")),
])
def test_invalid_plant_joint_contract_is_rejected(field, value):
    raw = load_plant_config(PROFILE).model_dump()
    raw["joints"][0][field] = value
    with pytest.raises(ValidationError):
        PlantConfig.model_validate(raw)


def test_plant_profile_rejects_checkpoint_fields():
    raw = load_plant_config(PROFILE).model_dump()
    raw["bundle"] = "checkpoint-directory"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PlantConfig.model_validate(raw)
