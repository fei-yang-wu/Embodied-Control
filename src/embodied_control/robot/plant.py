"""Robot parameters for the simulated DDS plant, independent of controllers."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlantJoint(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(min_length=1)
    motor_id: int = Field(ge=0, lt=29)
    nominal_position: float
    armature: float = Field(ge=0)
    effort_limit: float = Field(gt=0)
    vendor_stiffness: float = Field(ge=0)
    vendor_damping: float = Field(ge=0)


class PlantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_version: Literal["ec.plant/v1alpha1"] = "ec.plant/v1alpha1"
    joints: list[PlantJoint] = Field(min_length=29, max_length=29)

    @model_validator(mode="after")
    def validate_joint_mapping(self) -> "PlantConfig":
        if len({joint.name for joint in self.joints}) != 29:
            raise ValueError("plant joint names must be unique")
        if {joint.motor_id for joint in self.joints} != set(range(29)):
            raise ValueError("plant motor IDs must cover 0 through 28 exactly once")
        return self

    @property
    def joint_names(self) -> list[str]:
        return [joint.name for joint in self.joints]


def load_plant_config(path: str | Path) -> PlantConfig:
    source = Path(path)
    raw = yaml.safe_load(source.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{source} is not a mapping")
    return PlantConfig.model_validate(raw)
