"""Job schema for `ec lifecycle`: one YAML per rehearsal or hardware session."""

from __future__ import annotations

import json
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
    # Reference frames per oracle reply. 0 means the bundle's own encoder
    # window, which is the only value that is right for every stride.
    oracle_horizon: int = Field(default=0, ge=0)
    # The VLA planner service (`ec lowlevel planner-worker -- <command>`).
    vla_service_command: list[str] = Field(default_factory=list)
    vla_reply: Literal["latent_plan", "chunk"] = "latent_plan"
    vla_z_dim: int = Field(default=256, ge=1)
    vla_plan_slots: int = Field(default=1, ge=1)
    vla_hold_steps: int = Field(default=10, ge=1)


class GainScale(JobModel):
    """Deployment-time multipliers on the bundle's PD gains.

    An experiment knob, not a tuning: the policy learned the closed loop at
    the bundle's own gains, so a scale is part of the rehearsal identity and
    hardware needs a rehearsal at the same scale.
    """

    stiffness: float = Field(default=1.0, gt=0.0)
    damping: float = Field(default=1.0, gt=0.0)


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
    ticks: int | Literal["auto"] = 500
    blend_ticks: int = Field(default=250, ge=0)
    # Arming engages the policy on the reference's first frame and stops there;
    # the motion plays on a second, separate request, after this countdown.
    play_countdown_seconds: float = Field(default=3.0, ge=0.0)
    arm_timeout_seconds: float = Field(default=60.0, gt=0.0)
    stand_hold_seconds: float = Field(default=0.0, ge=0.0)
    lead_ticks: int = Field(default=4, ge=0)
    command_stale_ms: float = Field(default=500.0, gt=0.0)
    state_absent_ms: float = Field(default=500.0, gt=0.0)
    # An episode rests limp under the vendor, so the next one climbs
    # PRECHECK again. `vendor_stand` stays available as the `s` request.
    end_state: Literal["vendor_stand", "vendor_damp", "damp"] = "vendor_damp"
    # `damp` hands the joints back to the vendor once the kd-only frames are
    # on the wire and the hoist is acknowledged.
    damp_hands_back: bool = True
    # A retake re-reads the link before driving the robot a second time.
    retake_precheck: bool = True
    # Hardware runs what the plant already ran: PRECHECK refuses a non-loopback
    # interface until a passing rehearsal of the same bundle and motion exists.
    require_rehearsal: bool = True
    # Where to look for it. Empty means the parent of `artifacts_dir`, which is
    # where a sim job of the same campaign writes its own run.
    rehearsal_root: str = ""
    rehearsal_max_age_days: float = Field(default=14.0, gt=0.0)
    # Kinematic endpoint screening of the reference before a hardware run.
    # `start` refuses a moving or off-stance first frame (the ramp and the
    # blend assume a stationary start) and only reports the last frame: the
    # run ends in HOLD under the hoist, which is the recovery for any ending.
    # `both` is the old behaviour; `off` reports everything.
    endpoint_screening: Literal["start", "both", "off"] = "start"
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
    gain_scale: GainScale = Field(default_factory=GainScale)
    realtime: RealtimeSpec = Field(default_factory=RealtimeSpec)
    planner: PlannerSpec = Field(default_factory=PlannerSpec)
    # MJCF for in-line MPJPE after each episode (forward kinematics); empty
    # skips it and leaves `eval_mpjpe` to grade the saved telemetry.
    mjcf: str = ""
    artifacts_dir: str = ""

    @model_validator(mode="after")
    def validate_sources(self) -> "LifecycleJob":
        if not self.pin_reference:
            raise ValueError("arm/play requires pin_reference=true")
        if self.stand_hold_seconds > 0 and (self.command_source != "oracle" or self.ticks != "auto"):
            raise ValueError("stand_hold_seconds requires oracle playback with ticks=auto")
        if isinstance(self.ticks, int) and self.ticks < 1:
            raise ValueError("ticks must be positive or auto")
        if self.ticks == "auto" and self.command_source != "oracle":
            raise ValueError("ticks=auto requires oracle playback")
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
        manifest_path = Path(self.reference_root) / "reference_arrays_manifest.json"
        if self.reference_root and self.motion and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            info = manifest["traj_info"]
            names = [entry[1] for entry in info["ordered_traj_list"]]
            if self.motion not in names:
                raise ValueError(f"motion {self.motion!r} not in reference tree {self.reference_root}")
            index = names.index(self.motion)
            length = info["end_index"][index] - info["start_index"][index]
            if self.start_frame >= length - 1:
                raise ValueError("start_frame must leave at least one reference transition")
        return self


SIM_DDS_DOMAIN = 51


def apply_target(
    job: LifecycleJob,
    target: str,
    *,
    network: str = "",
    dds_domain: int | None = None,
) -> LifecycleJob:
    """One job, two targets: the same deployment against the plant or the robot.

    Only the fields that name *where* the run happens change, so the sim run
    and the hardware run share a rehearsal identity. A sim run writes under
    `<artifacts_dir>/sim`, a hardware run under `<artifacts_dir>/hardware`
    and looks for its rehearsal beside it.
    """
    if target not in ("sim", "hardware"):
        raise ValueError(f"target must be sim or hardware, not {target!r}")
    root = job.artifacts_dir
    if not root:
        raise ValueError("--target needs artifacts_dir in the job")
    base = job.model_dump(mode="json")
    stem = lambda slot: slot.removesuffix("_sim").removesuffix("_hw")  # noqa: E731
    if target == "sim":
        base.update(
            network="lo",
            dds_domain=SIM_DDS_DOMAIN if dds_domain is None else int(dds_domain),
            sim_hoist=True,
            artifacts_dir=str(Path(root) / "sim"),
            rehearsal_root="",
            request_slot=stem(job.request_slot) + "_sim",
            response_slot=stem(job.response_slot) + "_sim",
            realtime={
                **base["realtime"],
                "control_cpu": -1,
                "writer_cpu": -1,
                "control_priority": 0,
                "writer_priority": 0,
                "lock_memory": False,
            },
        )
    else:
        if not network and job.network in ("", "lo", "localhost"):
            raise ValueError("--target hardware needs --network <robot NIC>")
        base.update(
            network=network or job.network,
            dds_domain=0 if dds_domain is None else int(dds_domain),
            sim_hoist=False,
            artifacts_dir=str(Path(root) / "hardware"),
            rehearsal_root=job.rehearsal_root or root,
            request_slot=stem(job.request_slot) + "_hw",
            response_slot=stem(job.response_slot) + "_hw",
        )
    return LifecycleJob.model_validate(base)


def load_lifecycle_job(path: str | Path) -> LifecycleJob:
    source = Path(path)
    raw = yaml.safe_load(source.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source} is not a mapping")
    base = source.resolve().parent
    for name in ("bundle", "reference_root", "artifacts_dir", "mjcf", "rehearsal_root"):
        value = raw.get(name)
        if value and not Path(value).is_absolute():
            raw[name] = str((base / value).resolve())
    raw["trackers"] = {
        name: str((base / value).resolve()) if not Path(value).is_absolute() else value
        for name, value in raw.get("trackers", {}).items()
    }
    return LifecycleJob.model_validate(raw)
