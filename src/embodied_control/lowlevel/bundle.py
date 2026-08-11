"""Policy-bundle schemas and verification."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA = "ec.bundle/v1"


class BundleModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class ObservationTerm(BundleModel):
    name: str
    width: int = Field(gt=0)
    normalize: bool = True
    history_length: int = Field(default=1, ge=1)
    history_stride: int = Field(default=1, ge=1)
    history_order: Literal["oldest_first", "newest_first"] = "oldest_first"
    reset_fill: Literal["repeat_first", "zero"] = "repeat_first"

    @property
    def flat_width(self) -> int:
        return self.width * self.history_length


class ObservationContract(BundleModel):
    terms: list[ObservationTerm] = Field(min_length=1)
    total_width: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_width(self) -> "ObservationContract":
        actual = sum(term.flat_width for term in self.terms)
        if actual != self.total_width:
            raise ValueError(
                f"observation width is {self.total_width}, but terms total {actual}"
            )
        if len({term.name for term in self.terms}) != len(self.terms):
            raise ValueError("observation term names must be unique")
        return self


class ActionContract(BundleModel):
    width: int = 29
    isaac_joint_names: list[str] = Field(min_length=1)
    sdk_joint_names: list[str] = Field(default_factory=list)
    isaac_to_sdk: list[int] = Field(default_factory=list)
    default_joint_pos: list[float]
    default_joint_vel: list[float] = Field(default_factory=list)
    action_scale: list[float]
    stiffness: list[float]
    damping: list[float]
    armature: list[float] = Field(default_factory=list)
    effort_limit: list[float] = Field(default_factory=list)
    torque_ff: list[float] = Field(default_factory=list)
    last_action_is_raw: bool = True
    raw_action_clip: float | None = Field(default=None, gt=0)
    joint_limits_lower: list[float] | None = None
    joint_limits_upper: list[float] | None = None

    @model_validator(mode="after")
    def validate_widths(self) -> "ActionContract":
        if self.width != len(self.isaac_joint_names):
            raise ValueError("action width must equal the Isaac joint-name count")
        fields = (
            "default_joint_pos",
            "action_scale",
            "stiffness",
            "damping",
        )
        for field_name in fields:
            if len(getattr(self, field_name)) != self.width:
                raise ValueError(f"{field_name} must have width {self.width}")
        for field_name in (
            "default_joint_vel",
            "torque_ff",
            "armature",
            "effort_limit",
        ):
            values = getattr(self, field_name)
            if values and len(values) != self.width:
                raise ValueError(f"{field_name} must have width {self.width}")
        if self.sdk_joint_names:
            if len(self.sdk_joint_names) != self.width:
                raise ValueError("sdk_joint_names must have the action width")
            if len(self.isaac_to_sdk) != self.width:
                raise ValueError("isaac_to_sdk must have the action width")
            if sorted(self.isaac_to_sdk) != list(range(self.width)):
                raise ValueError("isaac_to_sdk must be a permutation")
        for field_name in ("joint_limits_lower", "joint_limits_upper"):
            values = getattr(self, field_name)
            if values is not None and len(values) != self.width:
                raise ValueError(f"{field_name} must have width {self.width}")
        if (self.joint_limits_lower is None) != (self.joint_limits_upper is None):
            raise ValueError("joint lower and upper limits must be declared together")
        if self.joint_limits_lower is not None and any(
            lower > upper
            for lower, upper in zip(
                self.joint_limits_lower, self.joint_limits_upper, strict=True
            )
        ):
            raise ValueError("joint lower limit cannot exceed upper limit")
        for field_name in ("stiffness", "damping", "armature"):
            if any(value < 0.0 for value in getattr(self, field_name)):
                raise ValueError(f"{field_name} values must be non-negative")
        if self.effort_limit and any(value <= 0.0 for value in self.effort_limit):
            raise ValueError("effort_limit values must be positive")
        return self

    def decode(
        self, raw_action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        action = np.asarray(raw_action, dtype=np.float32)
        if action.shape != (self.width,):
            raise ValueError(
                f"raw action must have shape ({self.width},), got {action.shape}"
            )
        if self.raw_action_clip is not None:
            action = np.clip(action, -self.raw_action_clip, self.raw_action_clip)
        q_target = np.asarray(
            self.default_joint_pos, dtype=np.float32
        ) + action * np.asarray(self.action_scale, dtype=np.float32)
        if self.joint_limits_lower is not None:
            q_target = np.maximum(
                q_target, np.asarray(self.joint_limits_lower, dtype=np.float32)
            )
        if self.joint_limits_upper is not None:
            q_target = np.minimum(
                q_target, np.asarray(self.joint_limits_upper, dtype=np.float32)
            )
        kp = np.asarray(self.stiffness, dtype=np.float32)
        kd = np.asarray(self.damping, dtype=np.float32)
        return q_target, kp, kd

    def sdk_values(self, values: np.ndarray) -> np.ndarray:
        """Convert Isaac-order values to SDK order at the hardware boundary."""
        value = np.asarray(values)
        if value.shape != (self.width,):
            raise ValueError(
                f"values must have shape ({self.width},), got {value.shape}"
            )
        if not self.isaac_to_sdk:
            return value.copy()
        result = np.empty_like(value)
        result[np.asarray(self.isaac_to_sdk)] = value
        return result


class CommandContract(BundleModel):
    z_dim: int | None = Field(default=None, gt=0)
    phase_mode: Literal["sin_cos", "none"] = "none"
    phase_dim: int = 0
    hold_steps: int = Field(default=1, ge=1)
    state_dim: int | None = Field(default=None, gt=0)
    encoder_state_interface: Literal[
        "root_qpos", "full_body", "joint_qpos_qvel_anchor_ori"
    ] | None = None
    window_steps: int | None = Field(default=None, ge=0)
    horizon_steps: int | None = Field(default=None, gt=0)
    encoder_window_mode: Literal["full", "intermediate"] | None = None
    macro_frame_stride: int | None = Field(default=None, gt=0)
    macro_anchor_mode: Literal[
        "robot", "robot_heading", "expert_heading"
    ] | None = None
    encoder_trigger: Literal["on_acceptance", "every_control_tick"] = (
        "on_acceptance"
    )
    activation: str | None = None
    layer_norm: bool | None = None
    encoder_sha256: str | None = None
    quantizer: Literal["none", "fsq"] = "none"
    fsq_half_levels: list[float] | None = None
    components: list[ObservationTerm] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_phase(self) -> "CommandContract":
        expected = 2 if self.phase_mode == "sin_cos" else 0
        if self.phase_dim != expected:
            raise ValueError(f"phase_dim must be {expected} for {self.phase_mode}")
        if self.quantizer == "fsq":
            if not self.fsq_half_levels:
                raise ValueError("quantizer=fsq requires fsq_half_levels")
            if self.z_dim is not None and len(self.fsq_half_levels) != self.z_dim:
                raise ValueError("fsq_half_levels length must equal z_dim")
            if any(half < 1.0 for half in self.fsq_half_levels):
                raise ValueError("fsq_half_levels entries must be >= 1")
        elif self.fsq_half_levels:
            raise ValueError("fsq_half_levels is only valid with quantizer=fsq")
        if self.encoder_window_mode is not None:
            if self.horizon_steps is None or self.window_steps is None:
                raise ValueError(
                    "encoder window mode requires horizon_steps and window_steps"
                )
            expected = self.horizon_steps - (
                1 if self.encoder_window_mode == "intermediate" else 0
            )
            if self.window_steps != expected:
                raise ValueError(
                    f"window_steps must be {expected} for "
                    f"encoder_window_mode={self.encoder_window_mode!r}"
                )
        return self


class RateContract(BundleModel):
    control_hz: int = Field(default=50, gt=0)
    physics_dt: float = Field(default=0.005, gt=0)
    decimation: int = Field(default=4, gt=0)


class ModelArtifact(BundleModel):
    format: Literal["onnx", "torchscript"]
    path: str
    input_name: str
    output_name: str
    input_shape: list[int] = Field(min_length=2, max_length=2)
    output_shape: list[int] = Field(min_length=2, max_length=2)
    opset: int | None = Field(default=None, gt=0)
    parity_atol: float = Field(gt=0)
    max_abs_error: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_static_shapes(self) -> "ModelArtifact":
        artifact_path = Path(self.path)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ValueError("model artifact path must stay inside the bundle")
        if not self.input_name or not self.output_name:
            raise ValueError("model artifact input and output names cannot be empty")
        if any(value <= 0 for value in (*self.input_shape, *self.output_shape)):
            raise ValueError("model artifact shapes must be positive and static")
        if self.input_shape[0] != 1 or self.output_shape[0] != 1:
            raise ValueError("low-level model artifacts require static batch one")
        if self.format == "onnx" and self.opset is None:
            raise ValueError("ONNX model artifacts require an opset")
        if self.format == "onnx" and self.opset != 18:
            raise ValueError("native low-level ONNX models require opset 18")
        if self.format == "torchscript" and self.opset is not None:
            raise ValueError("TorchScript model artifacts cannot declare an opset")
        if self.max_abs_error > self.parity_atol:
            raise ValueError("model artifact maximum error exceeds its tolerance")
        return self


class BundleManifest(BundleModel):
    api_version: Literal["ec.bundle/v1"] = SCHEMA
    source: dict[str, Any]
    interface: Literal["latent", "explicit", "chunk", "privileged-teacher"]
    obs: ObservationContract
    action: ActionContract
    command: CommandContract = Field(default_factory=CommandContract)
    rates: RateContract = Field(default_factory=RateContract)
    models: dict[str, ModelArtifact] = Field(default_factory=dict)
    files: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_interface(self) -> "BundleManifest":
        if self.interface == "latent" and self.command.z_dim is None:
            raise ValueError("latent bundles require command.z_dim")
        if self.interface == "latent":
            latent_terms = [
                term for term in self.obs.terms if term.name == "latent_command"
            ]
            expected_width = int(self.command.z_dim or 0) + self.command.phase_dim
            if len(latent_terms) != 1 or latent_terms[0].width != expected_width:
                raise ValueError(
                    "latent_command width must equal command z_dim plus phase_dim"
                )
        if self.interface == "privileged-teacher" and self.source.get(
            "primary_policy_role"
        ) not in {
            "teacher",
            "privileged-teacher",
        }:
            raise ValueError(
                "privileged-teacher bundles must identify the teacher role"
            )
        policy = self.models.get("policy_onnx")
        if policy is not None and (
            policy.format != "onnx"
            or policy.input_shape != [1, self.obs.total_width]
            or policy.output_shape != [1, self.action.width]
        ):
            raise ValueError(
                "policy_onnx shapes must match the observation and action contracts"
            )
        encoder = self.models.get("encoder_onnx")
        if encoder is not None:
            interface_width = {
                "root_qpos": 38,
                "full_body": 67,
                "joint_qpos_qvel_anchor_ori": 64,
            }.get(self.command.encoder_state_interface)
            if (
                interface_width is not None
                and self.command.state_dim != interface_width
            ):
                raise ValueError(
                    "encoder_state_interface disagrees with command.state_dim"
                )
            missing = [
                name
                for name in (
                    "state_dim",
                    "window_steps",
                    "horizon_steps",
                    "encoder_window_mode",
                    "macro_frame_stride",
                    "macro_anchor_mode",
                    "activation",
                    "layer_norm",
                    "encoder_sha256",
                )
                if getattr(self.command, name) is None
            ]
            if missing:
                raise ValueError(
                    f"encoder_onnx requires command provenance fields: {missing}"
                )
            expected_input = None
            if (
                self.command.state_dim is not None
                and self.command.window_steps is not None
            ):
                expected_input = self.command.state_dim * (
                    self.command.window_steps + 1
                )
            if (
                encoder.format != "onnx"
                or self.command.z_dim is None
                or encoder.output_shape != [1, self.command.z_dim]
                or (
                    expected_input is not None
                    and encoder.input_shape != [1, expected_input]
                )
            ):
                raise ValueError("encoder_onnx shapes must match the command contract")
            if self.files.get("encoder.pt") != self.command.encoder_sha256:
                raise ValueError(
                    "command encoder_sha256 must match the encoder.pt file hash"
                )
        return self


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PolicyBundle:
    root: Path
    manifest: BundleManifest

    @property
    def policy_path(self) -> Path:
        return self.root / "policy.pt"

    @property
    def encoder_path(self) -> Path:
        return self.root / "encoder.pt"

    @property
    def policy_onnx_path(self) -> Path:
        return self.model_path("policy_onnx", expected_format="onnx")

    @property
    def encoder_onnx_path(self) -> Path:
        return self.model_path("encoder_onnx", expected_format="onnx")

    def model_path(self, name: str, *, expected_format: str | None = None) -> Path:
        if name not in self.manifest.models:
            raise KeyError(f"policy bundle has no model artifact {name!r}")
        artifact = self.manifest.models[name]
        if expected_format is not None and artifact.format != expected_format:
            raise ValueError(
                f"model artifact {name!r} is {artifact.format}, not {expected_format}"
            )
        path = self.root / artifact.path
        if not path.is_file():
            raise FileNotFoundError(f"model artifact is missing: {path}")
        return path

    @classmethod
    def load(cls, root: str | Path, *, diagnostic: bool = False) -> "PolicyBundle":
        path = Path(root).expanduser().resolve()
        manifest_path = path / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"policy bundle manifest not found: {manifest_path}"
            )
        try:
            raw = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid policy bundle manifest: {manifest_path}"
            ) from exc
        manifest = BundleManifest.model_validate(raw)
        if manifest.api_version != SCHEMA:
            raise ValueError(
                f"unsupported policy bundle schema: {manifest.api_version}"
            )
        if manifest.interface == "privileged-teacher" and not diagnostic:
            raise PermissionError("privileged-teacher bundles require diagnostic=True")
        required = [
            "policy.pt",
            "obs_contract.json",
            "action_contract.json",
            "golden_trace.npz",
        ]
        if manifest.interface == "latent":
            required.append("encoder.pt")
        for name, artifact in manifest.models.items():
            if artifact.path not in manifest.files:
                raise ValueError(
                    f"model artifact {name!r} is not covered by a bundle hash"
                )
        for rel in manifest.files:
            relative = Path(rel)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"bundle hash path must stay inside the bundle: {rel}")
        for rel in required:
            if not (path / rel).is_file():
                raise FileNotFoundError(f"policy bundle is missing {rel}: {path}")
            if rel not in manifest.files:
                raise ValueError(f"required bundle file has no hash: {rel}")
        for rel, expected in manifest.files.items():
            target = path / rel
            if not target.resolve().is_relative_to(path):
                raise ValueError(f"bundle hash target escapes the bundle: {rel}")
            if not target.is_file():
                raise FileNotFoundError(f"manifest hash target is missing: {rel}")
            actual = sha256_file(target)
            if actual != expected:
                raise ValueError(
                    f"policy bundle hash mismatch for {rel}: {actual} != {expected}"
                )
        obs_file = ObservationContract.model_validate_json(
            (path / "obs_contract.json").read_text()
        )
        action_file = ActionContract.model_validate_json(
            (path / "action_contract.json").read_text()
        )
        if obs_file != manifest.obs:
            raise ValueError("obs_contract.json disagrees with manifest.obs")
        if action_file != manifest.action:
            raise ValueError("action_contract.json disagrees with manifest.action")
        return cls(path, manifest)

    def verify(self) -> dict[str, Any]:
        loaded = self.load(
            self.root, diagnostic=self.manifest.interface == "privileged-teacher"
        )
        return {
            "valid": True,
            "api_version": loaded.manifest.api_version,
            "interface": loaded.manifest.interface,
            "observation_width": loaded.manifest.obs.total_width,
            "action_width": loaded.manifest.action.width,
            "files": dict(loaded.manifest.files),
        }


__all__ = [
    "ActionContract",
    "BundleManifest",
    "CommandContract",
    "ModelArtifact",
    "ObservationContract",
    "ObservationTerm",
    "PolicyBundle",
    "RateContract",
    "SCHEMA",
    "sha256_file",
]
