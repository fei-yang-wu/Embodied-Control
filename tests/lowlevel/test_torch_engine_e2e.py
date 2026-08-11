import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from embodied_control.lowlevel.bundle import PolicyBundle, sha256_file  # noqa: E402
from embodied_control.lowlevel.engine.torch_engine import TorchEngine  # noqa: E402
from embodied_control.lowlevel.job import LowLevelJob  # noqa: E402
from embodied_control.lowlevel.runner import run_lowlevel_job, verify_bundle  # noqa: E402


def _scripted_linear(in_features, out_features, seed):
    generator = torch.Generator().manual_seed(seed)
    module = torch.nn.Linear(in_features, out_features)
    with torch.no_grad():
        module.weight.copy_(0.01 * torch.randn(out_features, in_features, generator=generator))
        module.bias.zero_()
    return torch.jit.script(module.eval())


@pytest.fixture
def torch_bundle(tmp_path, latent_manifest):
    root = tmp_path / "bundle"
    root.mkdir()
    policy = _scripted_linear(101, 29, seed=0)
    encoder = _scripted_linear(40, 6, seed=1)
    policy.save(str(root / "policy.pt"))
    encoder.save(str(root / "encoder.pt"))
    (root / "obs_contract.json").write_text(latent_manifest.obs.model_dump_json())
    (root / "action_contract.json").write_text(latent_manifest.action.model_dump_json())

    rng = np.random.default_rng(0)
    obs = rng.standard_normal((16, 101)).astype(np.float32)
    with torch.inference_mode():
        action = policy(torch.from_numpy(obs)).numpy()
    encoder_in = rng.standard_normal((8, 40)).astype(np.float32)
    with torch.inference_mode():
        encoder_out = encoder(torch.from_numpy(encoder_in)).numpy()
    np.savez(
        root / "golden_trace.npz",
        obs=obs, action=action, encoder_in=encoder_in, encoder_out=encoder_out,
    )

    raw = latent_manifest.model_dump()
    raw["files"] = {
        name: sha256_file(root / name)
        for name in [
            "policy.pt", "encoder.pt", "obs_contract.json",
            "action_contract.json", "golden_trace.npz",
        ]
    }
    (root / "manifest.json").write_text(json.dumps(raw))
    return root


def test_torch_engine_determinism_and_stats(torch_bundle, latent_manifest):
    engine = TorchEngine(torch_bundle / "policy.pt")
    engine.warmup(input_width=latent_manifest.obs.total_width)
    obs = np.linspace(-1.0, 1.0, 101, dtype=np.float32)
    first = engine.infer(obs).copy()
    for _ in range(10):
        np.testing.assert_array_equal(engine.infer(obs), first)
    assert engine.stats.count == 11
    assert engine.stats.p99_ms >= 0.0


def test_torch_engine_warmup_requires_width(torch_bundle):
    engine = TorchEngine(torch_bundle / "policy.pt")
    with pytest.raises(ValueError, match="input_width"):
        engine.warmup()


def test_verify_bundle_passes_and_detects_tamper(torch_bundle):
    report = verify_bundle(torch_bundle)
    assert report["valid"]
    assert report["policy_rows"] == 16
    assert report["policy_max_abs_err"] <= 1e-5
    assert report["encoder_rows"] == 8

    trace = dict(np.load(torch_bundle / "golden_trace.npz"))
    trace["action"] = trace["action"] + 0.1
    np.savez(torch_bundle / "golden_trace.npz", **trace)
    raw = json.loads((torch_bundle / "manifest.json").read_text())
    raw["files"]["golden_trace.npz"] = sha256_file(torch_bundle / "golden_trace.npz")
    (torch_bundle / "manifest.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="golden trace mismatch"):
        verify_bundle(torch_bundle)


def test_run_lowlevel_job_end_to_end(torch_bundle, tmp_path):
    rng = np.random.default_rng(2)
    macro = rng.standard_normal((120, 4)).astype(np.float32)
    reference = tmp_path / "reference.npz"
    np.savez(reference, macro_states=macro)
    job = {
        "api_version": "ec.lowlevel/v1alpha1",
        "bundle": str(torch_bundle),
        "command": {"topology": "local", "source": "onboard_encoder",
                    "reference": str(reference)},
        "rollout": {"episodes": 2, "max_steps": 60},
        "outputs": {"root": str(tmp_path / "runs")},
    }
    job_path = tmp_path / "job.yaml"
    job_path.write_text(json.dumps(job))

    run_dir, result = run_lowlevel_job(job_path)
    assert result.succeeded
    for episode in result.episodes:
        assert episode.status == "completed"
        assert episode.steps == 60
        assert episode.renewals == 60  # the onboard publisher republishes every tick
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["succeeded"]
    assert (run_dir / "logs" / "events.jsonl").is_file()
    lines = (run_dir / "episodes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_mujoco_backend_needs_env_and_model(torch_bundle, tmp_path):
    job = LowLevelJob(bundle=str(torch_bundle), env={"backend": "mujoco", "model": "x.xml"})
    job_path = tmp_path / "job.yaml"
    job_path.write_text(json.dumps(job.model_dump()))
    with pytest.raises((ImportError, FileNotFoundError, ValueError)):
        run_lowlevel_job(job_path)


def test_bundle_verify_via_policy_bundle_api(torch_bundle):
    bundle = PolicyBundle.load(torch_bundle)
    report = bundle.verify()
    assert report["observation_width"] == 101
    assert report["action_width"] == 29
