"""Assemble and run a lowlevel job; verify bundles against golden traces."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import numpy as np

from embodied_control.logging import EcLogger, LogConfig
from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.command_buffer import InProcessCommandBuffer, ZmqCommandBuffer
from embodied_control.lowlevel.envs.fake import FakeBackend
from embodied_control.lowlevel.job import LowLevelJob, load_lowlevel_job
from embodied_control.lowlevel.loop import ControlLoop, RunResult, write_run_artifacts
from embodied_control.lowlevel.tracker import BufferedCommandSource, LowLevelTracker


def _load_torch_engine(path: Path, device: str = "cpu"):
    from embodied_control.lowlevel.engine.torch_engine import TorchEngine

    return TorchEngine(path, device=device)


def verify_bundle(root: str | Path, *, atol: float = 1e-5) -> dict:
    """Replay the exporter's golden traces through the shipped TorchScript."""
    bundle = PolicyBundle.load(root)
    report = bundle.verify()
    trace = np.load(bundle.root / "golden_trace.npz")
    engine = _load_torch_engine(bundle.policy_path)
    engine.warmup(input_width=bundle.manifest.obs.total_width)
    obs, expected = trace["obs"], trace["action"]
    worst = 0.0
    for row_obs, row_expected in zip(obs, expected, strict=True):
        actual = engine.infer(np.asarray(row_obs, dtype=np.float32))
        worst = max(worst, float(np.abs(actual - row_expected).max()))
    report["policy_rows"] = int(obs.shape[0])
    report["policy_max_abs_err"] = worst
    if worst > atol:
        raise ValueError(f"policy golden trace mismatch: max abs err {worst} > {atol}")
    if "encoder_in" in trace:
        encoder = _load_torch_engine(bundle.encoder_path)
        enc_in, enc_out = trace["encoder_in"], trace["encoder_out"]
        encoder.warmup(input_width=int(enc_in.shape[1]))
        enc_worst = 0.0
        for row_in, row_out in zip(enc_in, enc_out, strict=True):
            actual = encoder.infer(np.asarray(row_in, dtype=np.float32))
            enc_worst = max(enc_worst, float(np.abs(actual - row_out).max()))
        report["encoder_rows"] = int(enc_in.shape[0])
        report["encoder_max_abs_err"] = enc_worst
        if enc_worst > atol:
            raise ValueError(f"encoder golden trace mismatch: max abs err {enc_worst} > {atol}")
    return report


def _build_publisher(job: LowLevelJob, bundle: PolicyBundle, buffer, encoder_engine, tracker):
    from embodied_control.lowlevel.publishers.onboard_encoder import OnboardEncoderPublisher
    from embodied_control.lowlevel.publishers.reference_playback import (
        ReferencePlaybackPublisher,
    )

    if job.command.topology == "push":
        return None
    if job.command.source == "gr00t_service":
        from embodied_control.lowlevel.publishers.gr00t_service import (
            Gr00tServicePublisher,
        )

        spec = job.command.gr00t
        return Gr00tServicePublisher(
            buffer,
            bundle.manifest.command,
            list(spec.service_cmd),
            mode=spec.mode,
            tracker=tracker,
            default_joint_pos=np.asarray(
                bundle.manifest.action.default_joint_pos, dtype=np.float32
            ),
            encoder=encoder_engine,
            hold_steps=spec.hold_steps,
            slots=spec.slots,
            rtc=spec.rtc,
            rtc_freeze_steps=spec.rtc_freeze_steps,
            rtc_ramp_rate=spec.rtc_ramp_rate,
            service_cwd=spec.service_cwd,
            goal_schedule=spec.goal_schedule,
            goal_sequence=spec.goal_sequence,
        )
    if job.command.reference is None:
        raise ValueError("command.topology=local requires command.reference")
    reference = Path(job.command.reference)
    if job.command.source == "onboard_encoder":
        if encoder_engine is None:
            raise ValueError("latent playback needs the bundle's encoder.pt")
        if reference.is_dir():
            from embodied_control.lowlevel.reference import ReferenceArrays

            arrays = ReferenceArrays(reference)
            if arrays.joint_names != bundle.manifest.action.isaac_joint_names:
                raise ValueError(
                    "reference-array joint order does not match the action contract"
                )
            motion = arrays.motion(job.command.motion if job.command.motion else 0)
            return OnboardEncoderPublisher(
                buffer,
                encoder_engine,
                bundle.manifest.command,
                motion=motion,
                hold_steps=job.command.hold_steps_override,
            )
        data = np.load(reference)
        return OnboardEncoderPublisher(
            buffer,
            encoder_engine,
            bundle.manifest.command,
            np.asarray(data["macro_states"], dtype=np.float32),
            hold_steps=job.command.hold_steps_override,
        )
    data = np.load(reference)
    proprio = {"projected_gravity", "base_ang_vel", "joint_pos_rel", "joint_vel_rel", "last_action"}
    order = [t.name for t in bundle.manifest.obs.terms if t.name not in proprio]
    return ReferencePlaybackPublisher(buffer, {name: data[name] for name in order}, order)


def run_lowlevel_job(job_path: str | Path, *, device: str = "cpu") -> tuple[Path, RunResult]:
    job = load_lowlevel_job(job_path)
    bundle = PolicyBundle.load(job.bundle)
    control_hz = bundle.manifest.rates.control_hz

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_lowlevel")
    run_dir = Path(job.outputs.root) / run_id
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    logger = EcLogger.create(
        LogConfig(level=job.outputs.log_level, log_dir=str(run_dir / "logs")), run_id
    )

    engine = _load_torch_engine(bundle.policy_path, device)
    engine.warmup(input_width=bundle.manifest.obs.total_width)
    encoder_engine = None
    if bundle.manifest.interface == "latent" and bundle.encoder_path.is_file():
        encoder_engine = _load_torch_engine(bundle.encoder_path, device)

    if job.env.backend == "fake":
        backend = FakeBackend(
            bundle.manifest.action, control_hz=control_hz, lag_alpha=job.env.lag_alpha
        )
    elif job.env.backend == "mujoco":
        try:
            from embodied_control.lowlevel.envs.mujoco import MujocoBackend
        except ImportError as exc:
            raise ImportError(
                "the mujoco backend needs the lowlevel-sim env: "
                "pixi run -e lowlevel-sim ec lowlevel run ..."
            ) from exc
        backend = MujocoBackend(
            bundle.manifest.action,
            job.env.model,
            control_hz=control_hz,
            timestep=bundle.manifest.rates.physics_dt,
            decimation=bundle.manifest.rates.decimation,
            record_video=job.rollout.record_video,
        )
    else:
        raise NotImplementedError(f"env backend {job.env.backend!r} is not built yet")

    if job.command.topology == "push":
        if job.command.buffer == "shm":
            from embodied_control.lowlevel.native_buffer import ShmCommandBuffer

            buffer = ShmCommandBuffer(job.command.shm_name, create=True)
        else:
            buffer = ZmqCommandBuffer(
                job.command.endpoint, topic=job.command.topic.encode()
            )
    else:
        buffer = InProcessCommandBuffer()
    source = BufferedCommandSource(buffer, control_hz=control_hz)
    tracker = LowLevelTracker(bundle, engine, source)
    publisher = _build_publisher(job, bundle, buffer, encoder_engine, tracker)

    loop = ControlLoop(job, tracker, backend, publisher, logger=logger.child("loop"))
    try:
        result = loop.run()
    finally:
        if hasattr(publisher, "close"):
            publisher.close()
    write_run_artifacts(
        run_dir, job, json.loads((bundle.root / "manifest.json").read_text()), result
    )
    for episode_id, arrays in loop.state_logs.items():
        np.savez_compressed(run_dir / f"states_ep{episode_id}.npz", **arrays)
    episode_goals = getattr(publisher, "episode_goals", None)
    if episode_goals:
        # Which goal each episode ran, so a multi-goal process can be scored
        # per goal without re-deriving it from the episode order downstream.
        (run_dir / "episode_goals.json").write_text(json.dumps(episode_goals, indent=2))
    head_ms = getattr(publisher, "head_ms", None)
    if head_ms:
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text())
        metrics["planner_head_ms"] = {
            "count": len(head_ms),
            "p50": float(np.percentile(head_ms, 50)),
            "p95": float(np.percentile(head_ms, 95)),
            "max": float(np.max(head_ms)),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
    frames = getattr(backend, "frames", None)
    if job.rollout.record_video and frames:
        import imageio.v2 as imageio

        video_dir = run_dir / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = (video_dir / "last_episode.mp4").resolve()
        imageio.mimwrite(video_path, frames, fps=25)
        print(f"video retained: {video_path}")
    logger.event("run.finished", phase="finalize", succeeded=result.succeeded, run_dir=str(run_dir))
    return run_dir, result


__all__ = ["run_lowlevel_job", "verify_bundle"]
