import json

import numpy as np
import pytest
from pydantic import ValidationError

from embodied_control.lowlevel.bundle import (
    ActionContract,
    BundleManifest,
    ModelArtifact,
    PolicyBundle,
    sha256_file,
)
from embodied_control.lowlevel.job import LowLevelJob, load_lowlevel_job


def test_obs_contract_width_mismatch_fails(latent_manifest):
    raw = latent_manifest.model_dump()
    raw["obs"]["total_width"] = 100
    with pytest.raises(ValidationError):
        BundleManifest.model_validate(raw)


def test_obs_contract_counts_recorded_history(latent_manifest):
    raw = latent_manifest.model_dump()
    raw["obs"]["terms"][2].update(
        {"history_length": 10, "history_stride": 1, "history_order": "oldest_first"}
    )
    raw["obs"]["total_width"] += 9 * 3
    manifest = BundleManifest.model_validate(raw)
    term = manifest.obs.terms[2]
    assert term.width == 3
    assert term.flat_width == 30
    assert manifest.obs.total_width == 128


def test_latent_manifest_requires_z_dim(latent_manifest):
    raw = latent_manifest.model_dump()
    raw["command"]["z_dim"] = None
    with pytest.raises(ValidationError):
        BundleManifest.model_validate(raw)


def test_action_decode_and_permutation(action_contract):
    raw = np.ones(29, dtype=np.float32)
    q_target, kp, kd = action_contract.decode(raw)
    np.testing.assert_allclose(q_target, 0.1 + 0.25 * raw)
    np.testing.assert_allclose(kp, 100.0)
    np.testing.assert_allclose(kd, 2.0)

    isaac = np.arange(29, dtype=np.float32)
    sdk = action_contract.sdk_values(isaac)
    for isaac_index, sdk_index in enumerate(action_contract.isaac_to_sdk):
        assert sdk[sdk_index] == isaac[isaac_index]


def test_action_contract_rejects_bad_permutation(action_contract):
    raw = action_contract.model_dump()
    raw["isaac_to_sdk"] = [0] * 29
    with pytest.raises(ValidationError):
        ActionContract.model_validate(raw)


def test_action_contract_requires_paired_ordered_joint_limits(action_contract):
    raw = action_contract.model_dump()
    raw["joint_limits_lower"] = [-1.0] * 29
    with pytest.raises(ValidationError, match="declared together"):
        ActionContract.model_validate(raw)
    raw["joint_limits_upper"] = [1.0] * 29
    raw["joint_limits_lower"][0] = 2.0
    with pytest.raises(ValidationError, match="cannot exceed"):
        ActionContract.model_validate(raw)


def test_action_contract_rejects_unsafe_gains(action_contract):
    raw = action_contract.model_dump()
    raw["stiffness"][0] = -1.0
    with pytest.raises(ValidationError, match="non-negative"):
        ActionContract.model_validate(raw)


def test_model_artifact_requires_static_batch_one():
    with pytest.raises(ValidationError, match="static batch one"):
        ModelArtifact(
            format="onnx",
            path="policy.onnx",
            input_name="obs",
            output_name="action",
            input_shape=[2, 101],
            output_shape=[2, 29],
            opset=18,
            parity_atol=1e-5,
            max_abs_error=0.0,
        )


def test_model_artifact_rejects_failed_parity_and_unsafe_path():
    base = {
        "format": "onnx",
        "path": "policy.onnx",
        "input_name": "obs",
        "output_name": "action",
        "input_shape": [1, 101],
        "output_shape": [1, 29],
        "opset": 18,
        "parity_atol": 1e-5,
        "max_abs_error": 2e-5,
    }
    with pytest.raises(ValidationError, match="exceeds"):
        ModelArtifact.model_validate(base)
    base["max_abs_error"] = 0.0
    base["path"] = "../policy.onnx"
    with pytest.raises(ValidationError, match="inside the bundle"):
        ModelArtifact.model_validate(base)


def _write_bundle(tmp_path, manifest: BundleManifest, tamper: bool = False):
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "policy.pt").write_bytes(b"not-a-real-policy")
    (root / "encoder.pt").write_bytes(b"not-a-real-encoder")
    (root / "obs_contract.json").write_text(manifest.obs.model_dump_json())
    (root / "action_contract.json").write_text(manifest.action.model_dump_json())
    np.savez(
        root / "golden_trace.npz", obs=np.zeros((1, 101)), action=np.zeros((1, 29))
    )
    files = {
        name: sha256_file(root / name)
        for name in [
            "policy.pt",
            "encoder.pt",
            "obs_contract.json",
            "action_contract.json",
            "golden_trace.npz",
        ]
    }
    if tamper:
        files["policy.pt"] = "0" * 64
    raw = manifest.model_dump()
    raw["files"] = files
    (root / "manifest.json").write_text(json.dumps(raw))
    return root


def test_bundle_load_round_trip(tmp_path, latent_manifest):
    root = _write_bundle(tmp_path, latent_manifest)
    bundle = PolicyBundle.load(root)
    assert bundle.manifest.interface == "latent"
    assert bundle.manifest.obs.total_width == 101


def test_bundle_hash_mismatch_refused(tmp_path, latent_manifest):
    root = _write_bundle(tmp_path, latent_manifest, tamper=True)
    with pytest.raises(ValueError, match="hash mismatch"):
        PolicyBundle.load(root)


def test_bundle_hash_target_cannot_escape_root(tmp_path, latent_manifest):
    root = _write_bundle(tmp_path, latent_manifest)
    raw = json.loads((root / "manifest.json").read_text())
    raw["files"]["../outside"] = "0" * 64
    (root / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="stay inside"):
        PolicyBundle.load(root)


def test_bundle_contract_file_must_match_manifest(tmp_path, latent_manifest):
    root = _write_bundle(tmp_path, latent_manifest)
    raw = json.loads((root / "manifest.json").read_text())
    contract = json.loads((root / "obs_contract.json").read_text())
    contract["terms"][0]["normalize"] = True
    (root / "obs_contract.json").write_text(json.dumps(contract))
    raw["files"]["obs_contract.json"] = sha256_file(root / "obs_contract.json")
    (root / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="disagrees"):
        PolicyBundle.load(root)


def test_bundle_missing_encoder_refused_for_latent(tmp_path, latent_manifest):
    root = _write_bundle(tmp_path, latent_manifest)
    (root / "encoder.pt").unlink()
    with pytest.raises(FileNotFoundError):
        PolicyBundle.load(root)


def test_privileged_teacher_requires_diagnostic(tmp_path, latent_manifest):
    raw = latent_manifest.model_dump()
    raw["interface"] = "privileged-teacher"
    raw["source"]["primary_policy_role"] = "teacher"
    manifest = BundleManifest.model_validate(raw)
    root = _write_bundle(tmp_path, manifest)
    with pytest.raises(PermissionError):
        PolicyBundle.load(root)
    assert (
        PolicyBundle.load(root, diagnostic=True).manifest.interface
        == "privileged-teacher"
    )


def test_job_defaults_and_validators(tmp_path):
    job = LowLevelJob(bundle="bundles/x")
    assert job.command.topology == "local"
    with pytest.raises(ValidationError):
        LowLevelJob(bundle="x", command={"topology": "push"})
    with pytest.raises(ValidationError):
        LowLevelJob(bundle="x", env={"backend": "mujoco"})
    path = tmp_path / "job.yaml"
    path.write_text("api_version: ec.lowlevel/v1alpha1\nbundle: bundles/x\n")
    assert load_lowlevel_job(path).bundle == "bundles/x"
