"""Job schema for `ec lifecycle`: one YAML per rehearsal or hardware session."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

LIFECYCLE_API_VERSION = "ec.lifecycle/v1alpha1"


class JobModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ThresholdSpec(JobModel):
    settle_seconds: float = Field(default=0.5, gt=0.0)
    settle_timeout_seconds: float = Field(default=10.0, gt=0.0)
    drift_rad: float = Field(default=0.03, gt=0.0)
    settle_position_rad: float | None = None
    ramp_fault_rad: float = Field(default=0.5, gt=0.0)
    ramp_fault_ms: float = Field(default=100.0, gt=0.0)
    hold_gain_scale: float = Field(default=3.0, gt=0.0)
    # None derives per-joint tolerances from the bundle's joint names: legs
    # tight, waist medium, arms loose (they sag under the policy's PD gains).
    pose_tolerance_rad: float | list[float] | None = None
    tilt_tolerance_degrees: float = Field(default=10.0, gt=0.0)
    first_action_rad: float = Field(default=2.5, gt=0.0)
    first_action_torque_ratio: float = Field(default=3.0, gt=0.0)
    damp_publish_frames: int = Field(default=100, ge=1)
    hoist_release_seconds: float = Field(default=1.5, ge=0.0)
    command_timeout_seconds: float = Field(default=10.0, gt=0.0)
    vendor_timeout_seconds: float = Field(default=5.0, gt=0.0)


class PlannerSpec(JobModel):
    oracle_horizon: int = Field(default=30, ge=1)
    # The VLA planner service (`ec lowlevel planner-worker -- <command>`).
    vla_service_command: list[str] = Field(default_factory=list)
    vla_reply: Literal["latent_plan", "chunk"] = "latent_plan"
    vla_z_dim: int = Field(default=256, ge=1)
    vla_plan_slots: int = Field(default=1, ge=1)
    vla_hold_steps: int = Field(default=10, ge=1)


class RealtimeSpec(JobModel):
    control_cpu: int = 2
    writer_cpu: int = 3
    control_priority: int = 80
    writer_priority: int = 90
    lock_memory: bool = True
    policy_threads: int = 1


class LifecycleJob(JobModel):
    api_version: Literal["ec.lifecycle/v1alpha1"] = LIFECYCLE_API_VERSION
    bundle: str
    # Additional deployable policy bundles. Keys are operator-facing tracker
    # names; values are bundle directories. `bundle` remains the default and
    # keeps existing jobs valid.
    trackers: dict[str, str] = Field(default_factory=dict)
    network: str = "lo"
    dds_domain: int = 0
    request_slot: str = "/ec_g1_request"
    response_slot: str = "/ec_g1_response"
    connect_slots: bool = True
    command_source: Literal["vla", "oracle"] = "vla"
    reference_root: str = ""
    motion: str = ""
    start_frame: int = Field(default=0, ge=0)
    fixed_initial_anchor: bool = False
    # Legacy compatibility knob. Displacement is now reported, never rejected:
    # curated moving references intentionally deploy from a fixed start anchor.
    fixed_anchor_max_displacement: float = Field(default=0.05, gt=0.0)
    # "default" is the bundle stance, "motion" is reference frame
    # `start_frame`, a list is an explicit 29-value Isaac-order qpos.
    start_pose: Literal["default", "motion"] | list[float] = "default"
    ramp_seconds: float = Field(default=3.0, gt=0.0)
    ticks: int = Field(default=500, ge=1)
    blend_ticks: int = Field(default=250, ge=0)
    lead_ticks: int = Field(default=4, ge=0)
    command_stale_ms: float = Field(default=500.0, gt=0.0)
    state_absent_ms: float = Field(default=500.0, gt=0.0)
    end_state: Literal["vendor_stand", "vendor_damp", "damp"] = "vendor_stand"
    # Empty reads the active service from CheckMode at PRECHECK.
    vendor_name: str = ""
    require_vendor: bool = True
    vendor_rpc_timeout_seconds: float = Field(default=5.0, gt=0.0)
    # Rehearsal against the plant: acknowledge hoist/lower by driving the
    # plant's gantry instead of waiting for an operator.
    sim_hoist: bool = False
    slack_on_run: bool = True
    pin_reference: bool = True
    thresholds: ThresholdSpec = Field(default_factory=ThresholdSpec)
    realtime: RealtimeSpec = Field(default_factory=RealtimeSpec)
    planner: PlannerSpec = Field(default_factory=PlannerSpec)
    # MJCF for in-line MPJPE after each episode (forward kinematics); empty
    # skips it and leaves `eval_mpjpe` to grade the saved telemetry.
    mjcf: str = ""
    artifacts_dir: str = ""

    @model_validator(mode="after")
    def validate_sources(self) -> "LifecycleJob":
        if self.command_source == "oracle":
            if not self.reference_root or not self.motion:
                raise ValueError(
                    "command_source=oracle needs reference_root and motion"
                )
            if not self.fixed_initial_anchor:
                raise ValueError(
                    "command_source=oracle on the Unitree path needs "
                    "fixed_initial_anchor=true (no external localization)"
                )
        if self.start_pose == "motion" and (
            not self.reference_root or not self.motion
        ):
            raise ValueError("start_pose=motion needs reference_root and motion")
        if isinstance(self.start_pose, list) and len(self.start_pose) != 29:
            raise ValueError("an explicit start_pose needs 29 joint values")
        if isinstance(self.thresholds.pose_tolerance_rad, list) and len(
            self.thresholds.pose_tolerance_rad
        ) != 29:
            raise ValueError("pose_tolerance_rad list needs 29 values")
        return self


def load_lifecycle_job(path: str | Path) -> LifecycleJob:
    source = Path(path)
    raw = yaml.safe_load(source.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source} is not a mapping")
    job = LifecycleJob.model_validate(raw)
    base = source.resolve().parent
    for name in ("bundle", "reference_root", "artifacts_dir", "mjcf"):
        value = getattr(job, name)
        if value and not Path(value).is_absolute():
            setattr(job, name, str((base / value).resolve()))
    job.trackers = {
        name: str((base / value).resolve()) if not Path(value).is_absolute() else value
        for name, value in job.trackers.items()
    }
    return job
