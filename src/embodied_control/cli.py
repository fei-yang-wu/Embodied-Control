"""The ``ec`` command-line entry point.

Subcommands:
    ec doctor                         host + dependency check
    ec eval run <job.yaml>            resolve, launch policy service, run rollout, write artifacts
    ec eval validate <run_dir>        validate a run directory against the schemas
    ec eval list-runs <root>          list runs under a root directory
    ec policy serve ...               run a blank policy service standalone
    ec policy ping --endpoint host:p  health + describe an endpoint

Kept dependency-light (argparse). Heavy imports (mujoco, the runner) are done
inside the handlers so ``ec doctor`` / ``ec policy`` work in the light host env.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import time


# Pinned model assets live beside the code they are run from, so a relative
# default works from the repository root the operating card already assumes.
DEFAULT_MODEL_ROOT = "assets/models"


def _models_token(args) -> str | None:
    return getattr(args, "token", "") or None


def _cmd_models_list(args) -> int:
    from embodied_control.models import find_pins, load_pin, verify

    directories = find_pins(args.root)
    if not directories:
        print(f"no pinned models under {args.root}")
        return 0
    for directory in directories:
        pin = load_pin(directory)
        missing = verify(directory, pin)
        state = "here" if not missing else f"{len(missing)}/{len(pin.files)} missing"
        remote = f"{pin.repo}@{pin.revision[:8]}"
        if pin.path:
            remote += f":{pin.path}"
        print(f"{pin.kind:<10} {directory}  {remote}  {state}")
    return 0


def _cmd_models_pull(args) -> int:
    from embodied_control.models import ModelStoreError, find_pins, materialize

    if args.all:
        targets = find_pins(args.root)
    elif args.directory:
        targets = [Path(args.directory)]
    else:
        print("FAIL: name a directory or pass --all")
        return 2
    for directory in targets:
        try:
            materialize(directory, token=_models_token(args))
        except ModelStoreError as exc:
            print(f"FAIL: {exc}")
            return 1
        print(f"ok: {directory}")
    return 0


def _cmd_models_verify(args) -> int:
    from embodied_control.models import load_pin, verify

    pin = load_pin(args.directory)
    missing = verify(args.directory, pin)
    if missing:
        print(f"FAIL: {len(missing)} file(s) do not match the pin:")
        for name in sorted(missing):
            print(f"  {name}")
        return 1
    print(f"ok: {len(pin.files)} file(s) match {pin.repo}@{pin.revision[:8]}")
    return 0


def _cmd_models_pin(args) -> int:
    from embodied_control.models import ModelStoreError, pin_local_directory

    try:
        pin = pin_local_directory(
            args.directory,
            kind=args.kind,
            repo=args.repo,
            path_in_repo=args.path,
            revision=args.revision or None,
            repo_type=args.repo_type,
            token=_models_token(args),
        )
    except ModelStoreError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"pinned {len(pin.files)} file(s) to {pin.repo}@{pin.revision[:8]}")
    return 0


def _cmd_models_fetch(args) -> int:
    from embodied_control.models import ModelStoreError, fetch_and_pin

    try:
        pin = fetch_and_pin(
            args.directory,
            kind=args.kind,
            repo=args.repo,
            path_in_repo=args.path,
            revision=args.revision or None,
            repo_type=args.repo_type,
            token=_models_token(args),
        )
    except ModelStoreError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(
        f"fetched and pinned {len(pin.files)} file(s) from "
        f"{pin.repo}@{pin.revision[:8]}"
    )
    return 0


def _cmd_models_push(args) -> int:
    from embodied_control.models import ModelStoreError, push_model

    try:
        pin = push_model(
            args.directory,
            kind=args.kind,
            repo=args.repo,
            path_in_repo=args.path,
            private=not args.public,
            repo_type=args.repo_type,
            token=_models_token(args),
            message=args.message or None,
        )
    except ModelStoreError as exc:
        print(f"FAIL: {exc}")
        return 1
    visibility = "public" if args.public else "private"
    print(
        f"published {len(pin.files)} file(s) to {visibility} {pin.repo}"
        f"@{pin.revision[:8]}"
    )
    return 0


def _cmd_doctor(args) -> int:
    # Must run before the FIRST `import mujoco` anywhere in this process: MuJoCo
    # selects its GL backend (GLFW/EGL/OSMesa) the first time it is imported, and
    # later imports reuse that choice regardless of MUJOCO_GL changing afterward.
    try:
        from embodied_control.sim.mujoco_backend import _default_headless_gl_backend

        _default_headless_gl_backend()
    except ImportError:
        pass

    print("embodied-control doctor")
    print(f"  python           : {sys.version.split()[0]}")
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        mark = "ok " if ok else "MISS"
        print(f"  [{mark}] {name}{(' - ' + detail) if detail else ''}")

    try:
        import pydantic  # noqa: F401

        check("pydantic", True)
    except Exception as exc:  # noqa: BLE001
        check("pydantic", False, str(exc))
    try:
        import yaml  # noqa: F401

        check("pyyaml", True)
    except Exception as exc:  # noqa: BLE001
        check("pyyaml", False, str(exc))
    try:
        import mujoco  # noqa: F401

        check("mujoco (sim env)", True, mujoco.__version__)
    except Exception:  # noqa: BLE001
        check(
            "mujoco (sim env)", False, "run in the `sim` pixi env: pixi run -e sim ..."
        )

    try:
        import imageio  # noqa: F401

        check("imageio (video, sim env)", True, imageio.__version__)
    except Exception:  # noqa: BLE001
        check(
            "imageio (video, sim env)",
            False,
            "needed for rollout.record_video; run in the `sim` pixi env",
        )

    try:
        import mujoco  # noqa: F401

        renderer = mujoco.Renderer(
            mujoco.MjModel.from_xml_string("<mujoco/>"), height=64, width=64
        )
        renderer.close()
        check(
            "mujoco offscreen renderer",
            True,
            f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}",
        )
    except ImportError:
        check("mujoco offscreen renderer", False, "requires mujoco (sim env)")
    except Exception as exc:  # noqa: BLE001
        check(
            "mujoco offscreen renderer",
            False,
            f"{exc} (no EGL/OSMesa on this host? rollout.record_video will be disabled)",
        )

    import shutil

    docker = shutil.which("docker")
    check(
        "docker (for containerized policy)",
        docker is not None,
        docker or "optional; runtime.type=local works without it",
    )

    # The blank policy service is stdlib-only:
    try:
        from embodied_control.transport.server import PolicyServer  # noqa: F401
        from embodied_control.policies.base import make_policy  # noqa: F401

        check("policy service (stdlib)", True)
    except Exception as exc:  # noqa: BLE001
        check("policy service (stdlib)", False, str(exc))

    print(
        "  note: MuJoCo eval requires the `sim` env; the policy service runs in Docker "
        "(runtime.type=docker) or as a local subprocess (runtime.type=local)."
    )
    return 0 if all(checks) else 0  # doctor never fails the process; it reports


def _cmd_eval_run(args) -> int:
    from embodied_control.config.loader import load_job
    from embodied_control.orchestration.runner import run_eval

    job = load_job(args.job)
    if args.log_level:
        job.outputs.log_level = args.log_level
    result = run_eval(job)
    m = result.metrics
    print(f"run_id      : {result.run_id}")
    print(f"run_dir     : {result.run_dir}")
    print(f"status      : {result.status}")
    print(
        f"episodes    : completed={m.num_episodes_completed} failed={m.num_episodes_failed}"
    )
    print(f"success_rate: {m.success_rate:.3f}")
    print(f"mean_return : {m.mean_return:.4f}")
    print(
        f"policy reqs : {m.num_policy_requests}  "
        f"latency p50={m.policy_latency_ms.p50:.2f}ms p95={m.policy_latency_ms.p95:.2f}ms"
    )
    return 0 if result.status == "succeeded" else 1


def _cmd_eval_validate(args) -> int:
    from embodied_control.artifacts.validation import validate_run_dir

    report = validate_run_dir(args.run_dir)
    print(json.dumps(report.model_dump(), indent=2))
    return 0 if report.valid else 1


def _cmd_eval_list_runs(args) -> int:
    root = Path(args.root)
    if not root.is_dir():
        print(f"no such directory: {root}")
        return 1
    for run_dir in sorted(p for p in root.iterdir() if (p / "manifest.json").is_file()):
        try:
            manifest = json.loads((run_dir / "manifest.json").read_text())
            metrics = json.loads((run_dir / "metrics.json").read_text())
            print(
                f"{run_dir.name}  status={manifest.get('status')}  "
                f"success_rate={metrics.get('success_rate')}  "
                f"episodes={metrics.get('num_episodes_completed')}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{run_dir.name}  <unreadable: {exc}>")
    return 0


def _cmd_policy_serve(args) -> int:
    from embodied_control.policies.debug_server import main as serve_main

    argv = [
        "--type",
        args.type,
        "--action-dim",
        str(args.action_dim),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--seed",
        str(args.seed),
    ]
    return serve_main(argv)


def _cmd_policy_ping(args) -> int:
    from embodied_control.transport.client import PolicyClient, PolicyClientError

    host, _, port = args.endpoint.rpartition(":")
    client = PolicyClient(host or "127.0.0.1", int(port))
    try:
        print("health  :", json.dumps(client.wait_healthy(timeout_s=args.timeout)))
        print("describe:", json.dumps(client.describe()))
    except PolicyClientError as exc:
        print(f"error: {exc}")
        return 1
    return 0


def _cmd_lowlevel_run(args) -> int:
    from embodied_control.lowlevel.runner import run_lowlevel_job

    run_dir, result = run_lowlevel_job(
        args.job,
        device=args.device,
        viewer=args.viewer,
        viewer_host=args.viewer_host,
        viewer_port=args.viewer_port,
        viewer_fps=args.viewer_fps,
    )
    print(f"run_dir: {run_dir}")
    for episode in result.episodes:
        print(
            f"episode {episode.episode_id}: {episode.status} steps={episode.steps}"
            + (f" damp_cause={episode.damp_cause}" if episode.damp_cause else "")
        )
    print(f"engine: {json.dumps(result.engine_stats)}")
    return 0 if result.succeeded else 1


def _cmd_lowlevel_verify_bundle(args) -> int:
    from embodied_control.lowlevel.runner import verify_bundle

    try:
        report = verify_bundle(args.bundle, atol=args.atol)
    except (
        KeyError,
        RuntimeError,
        ValueError,
        FileNotFoundError,
        PermissionError,
    ) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


def _cmd_lowlevel_verify_native_bundle(args) -> int:
    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.native_core import verify_native_bundle

    try:
        report = verify_native_bundle(PolicyBundle.load(args.bundle))
    except (
        KeyError,
        RuntimeError,
        ValueError,
        FileNotFoundError,
        PermissionError,
    ) as exc:
        print(f"FAIL: {exc}")
        return 1
    print(json.dumps(report, indent=2))
    return 0


def _cmd_lowlevel_bench_native(args) -> int:
    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.native_core import benchmark_native_bundle

    report = benchmark_native_bundle(
        PolicyBundle.load(args.bundle),
        ticks=args.ticks,
        policy_threads=args.policy_threads,
        lead_ticks=args.lead_ticks,
        paced=args.paced,
        cpu=args.cpu,
        fifo_priority=args.fifo_priority,
        lock_memory=args.lock_memory,
        require_realtime=args.require_realtime,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["fault"] == 0 else 1


def _cmd_lowlevel_mujoco_native(args) -> int:
    import numpy as np

    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.native_core import NativeMujocoLoop

    if args.plan_slots < 1:
        raise SystemExit("--plan-slots must be at least 1")
    if args.plan_slots != 1 and not args.latent_plan:
        raise SystemExit("--plan-slots > 1 needs --latent-plan")
    if args.latent_plan and args.command_source != "vla":
        raise SystemExit("--latent-plan needs --command-source vla")
    bundle = PolicyBundle.load(args.bundle)
    runtime = NativeMujocoLoop(
        bundle,
        args.model,
        response_slot=args.response_slot,
        request_slot=args.request_slot,
        create_slots=not args.connect_slots,
        command_absent_ticks=args.command_absent_ticks,
        command_stale_ms=args.command_stale_ms,
        lead_ticks=args.lead_ticks,
        plan_slots=args.plan_slots,
        latent_plan=args.latent_plan,
        cpu=args.cpu,
        fifo_priority=args.fifo_priority,
        lock_memory=args.lock_memory,
        require_realtime=args.require_realtime,
        physics_cpu=args.physics_cpu,
        physics_fifo_priority=args.physics_fifo_priority,
        physics_lock_memory=args.physics_lock_memory,
        physics_require_realtime=args.physics_require_realtime,
        policy_threads=args.policy_threads,
        command_source=args.command_source,
    )
    from embodied_control.logging import EcLogger, LogConfig
    from embodied_control.lowlevel.telemetry import TelemetryRecorder

    telemetry_dir = Path(args.telemetry_dir).resolve() if args.telemetry_dir else None
    logger = EcLogger.null()
    if telemetry_dir is not None:
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        logger = EcLogger.create(
            LogConfig(level="INFO", console=False, log_dir=str(telemetry_dir)),
            telemetry_dir.name,
        )
    recorder = TelemetryRecorder(
        runtime, logger=logger.child("telemetry"), sample_hz=args.telemetry_hz
    )
    mpjpe_motion = None
    if args.mpjpe:
        if args.command_source != "oracle":
            raise SystemExit("--mpjpe needs --command-source oracle (a reference motion)")
        if not args.reference_root or not args.motion:
            raise SystemExit("--mpjpe needs --reference-root and --motion")
        from embodied_control.lowlevel.reference import ReferenceArrays

        mpjpe_motion = ReferenceArrays(args.reference_root).motion(args.motion)
        # Match the Isaac protocol: the episode starts ON reference frame 0.
        runtime.set_initial_pose(
            np.concatenate(
                [
                    mpjpe_motion.anchor_pos_w[0],
                    mpjpe_motion.anchor_quat_w[0],
                    mpjpe_motion.joint_qpos[0],
                ]
            )
        )
    view = None
    try:
        if args.viewer:
            from embodied_control.lowlevel.plant_view import watch

            def start_runtime() -> None:
                recorder.start()
                runtime.start(args.ticks, paced=True)

            view = watch(
                runtime,
                args.model,
                bundle.manifest.action.isaac_joint_names,
                live=True,
                fps=args.viewer_fps,
                host=args.viewer_host,
                port=args.viewer_port,
                stats=runtime.stats,
                on_ready=start_runtime,
                should_stop=lambda: not runtime.running,
            )
        else:
            recorder.start()
            runtime.start(args.ticks, paced=True)
        runtime.wait()
    except KeyboardInterrupt:
        runtime.stop()
        runtime.wait()
    except Exception:
        runtime.stop()
        runtime.wait()
        raise
    finally:
        recorder.stop()
    telemetry_record = recorder.collect()
    if telemetry_dir is not None:
        recorder.save(telemetry_dir, telemetry_record)
    heights = runtime.base_heights()
    heights = heights[np.isfinite(heights)]
    reference_errors = runtime.reference_joint_mae()
    reference_errors = reference_errors[np.isfinite(reference_errors)]
    report = {
        "command_source": args.command_source,
        "bundle": str(Path(args.bundle).resolve()),
        "control": runtime.stats(),
        "simulation_time": runtime.backend_time,
        "base_height": {
            "min_physics": runtime.min_base_height,
            "mean_control": float(np.mean(heights)) if heights.size else None,
            "final": runtime.base_height,
            "fell_below_0_4": runtime.min_base_height < 0.4,
        },
        "reference_joint_mae_rad": (
            {
                "samples": int(reference_errors.size),
                "mean": float(np.mean(reference_errors)),
                "p95": float(np.percentile(reference_errors, 95)),
                "final": float(reference_errors[-1]),
            }
            if reference_errors.size
            else None
        ),
    }
    if view is not None:
        report["viewer"] = view
    if args.mpjpe:
        from embodied_control.lowlevel.metrics import oracle_tracking_metrics

        motion = mpjpe_motion
        tracking = oracle_tracking_metrics(
            bundle.manifest.action,
            args.model,
            motion,
            telemetry_record,
            args.start_frame,
        )
        per_frame = tracking.pop("per_frame")
        if telemetry_dir is not None:
            np.savez_compressed(telemetry_dir / "mpjpe_per_frame.npz", **per_frame)
        report["tracking_mpjpe"] = {**tracking, "motion": motion.name}
    if args.report:
        output = Path(args.report).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(output)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return (
        0
        if report["control"]["fault"] == 0 and report["control"]["control_ticks"] > 0
        else 1
    )


def _cmd_lowlevel_planner_worker(args) -> int:
    from embodied_control.lowlevel.publishers.native_pull import (
        NativeChunkWorker,
        NativeLatentPlanWorker,
        StdioChunkService,
    )

    command = list(args.service_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("FAIL: planner-worker needs a service command after --")
        return 2
    service = StdioChunkService(
        command,
        action_width=args.z_dim if args.reply == "latent_plan" else args.state_width,
        window_frames=args.plan_slots if args.reply == "latent_plan" else 10,
    )
    if args.reply == "latent_plan":
        # A latent head predicts the commands themselves: forward its plan and
        # let the controller walk it, one head call per plan.
        worker = NativeLatentPlanWorker(
            args.request_slot,
            args.response_slot,
            service,
            z_dim=args.z_dim,
            plan_slots=args.plan_slots,
            hold_steps=args.hold_steps,
            lead_ticks=args.lead_ticks,
            rtc_enabled=args.rtc,
            create_slots=args.create_slots,
        )
    else:
        worker = NativeChunkWorker(
            args.request_slot,
            args.response_slot,
            service,
            hold_steps=args.hold_steps,
            lead_ticks=args.lead_ticks,
            state_width=args.state_width,
            rtc_enabled=args.rtc,
            create_slots=args.create_slots,
        )
    interrupted = False
    try:
        worker.run()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        worker.close()
        service.close()
    report = {
        "requests": worker.requests,
        "reply": args.reply,
        "request_ms_mean": (
            round(sum(worker.request_ms) / len(worker.request_ms), 3)
            if worker.request_ms
            else None
        ),
        "request_ms_max": (
            round(max(worker.request_ms), 3) if worker.request_ms else None
        ),
        "rtc_enabled": args.rtc,
        "interrupted": interrupted,
        "error": None if worker.last_error is None else str(worker.last_error),
    }
    if args.report:
        output = Path(args.report).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(output)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if worker.last_error is None else 1


def _cmd_lowlevel_certify_report(args) -> int:
    from embodied_control.lowlevel.slo import certify_report_file

    certificate = certify_report_file(args.report, args.output or None)
    print(json.dumps(certificate, indent=2))
    return 0 if certificate["pass"] else 1


def _cmd_lowlevel_oracle_worker(args) -> int:
    import numpy as np

    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.publishers.native_oracle import (
        NativeOracleWorker,
    )

    worker = NativeOracleWorker(
        args.request_slot,
        args.response_slot,
        PolicyBundle.load(args.bundle),
        args.reference_root,
        args.motion,
        start_frame=args.start_frame,
        # 0 means the bundle's own minimum: the encoder window is
        # (window_steps + 1 - 1) * frame_stride + hold_steps frames, and
        # SONIC v1.1's stride of 5 needs far more than a root_qpos bundle.
        horizon=args.horizon or None,
        create_slots=args.create_slots,
    )
    interrupted = False
    try:
        worker.run()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        worker.close()
    durations = np.asarray(worker.request_ms, dtype=np.float64)
    report = {
        **worker.provenance,
        "requests": worker.requests,
        "padded_frames": worker.padded_frames,
        "request_ms_p50": (
            float(np.percentile(durations, 50)) if durations.size else None
        ),
        "request_ms_p99": (
            float(np.percentile(durations, 99)) if durations.size else None
        ),
        "interrupted": interrupted,
        "error": None if worker.last_error is None else str(worker.last_error),
    }
    if args.report:
        output = Path(args.report).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        report["report_path"] = str(output)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if worker.last_error is None else 1


def _cmd_lowlevel_compare_unitree_pose(args) -> int:
    from embodied_control.lowlevel.unitree_pose import compare_live_pose

    try:
        report = compare_live_pose(
            args.bundle,
            args.model,
            args.network,
            args.output,
            samples=args.samples,
            timeout=args.timeout,
            root_height=args.root_height,
            dds_domain=args.dds_domain,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    print(f"{report['status'].upper()}: G1 joint mapping to MuJoCo")
    print(f"image: {report['pose']['image']}")
    print(f"report: {report['report']}")
    print(
        "roundtrip max error: "
        f"{report['pose']['roundtrip_max_error']:.3g}"
    )
    print(
        "MuJoCo readback max error: "
        f"{report['pose']['mujoco_readback_max_error']:.3g}"
    )
    print(
        "mapping name mismatches: "
        f"{len(report['pose']['mapping_name_mismatches'])}"
    )
    print("writes: disabled")
    return 0 if report["status"] == "pass" else 1


def _cmd_lowlevel_check_unitree(args) -> int:
    from embodied_control.lowlevel.unitree_probe import run_probe, write_report

    try:
        report = run_probe(
            args.network,
            samples=args.samples,
            timeout=args.timeout,
            min_rate_hz=args.min_rate,
            max_gap_ms=args.max_gap_ms,
            dds_domain=args.dds_domain,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    if args.report:
        write_report(report, args.report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{report['status'].upper()}: Unitree G1 rt/lowstate")
        print(f"network: {report['network']}")
        print(f"samples: {report['samples']}")
        print(f"rate: {report['rate_hz']:.1f} Hz")
        print(f"max gap: {report['max_gap_ms']:.2f} ms")
        print(f"CRC errors: {report['crc_errors']}")
        print(f"motor error joints: {report['motor_error_joints']}")
        print("writes: disabled")
    return 0 if report["status"] == "pass" else 1


def _unitree_write_gate_error(
    *, enable_writes: bool, allow_non_realtime: bool, confirm: str
) -> str | None:
    if not enable_writes:
        return None
    required = (
        "ENABLE_G1_LOWLEVEL_NON_REALTIME"
        if allow_non_realtime
        else "ENABLE_G1_LOWLEVEL"
    )
    if confirm != required:
        return f"--enable-writes requires --confirm {required}"
    return None


def _unitree_stationary_anchor(
    bundle, reference_root: str, motion: str, start_frame: int,
    max_displacement: float,
):
    import numpy as np

    from embodied_control.lowlevel.reference import ReferenceArrays

    if max_displacement <= 0:
        raise ValueError("fixed-anchor maximum displacement must be positive")
    reference = ReferenceArrays(reference_root)
    if reference.joint_names != list(bundle.manifest.action.isaac_joint_names):
        raise ValueError("reference and bundle Isaac joint orders differ")
    selected = reference.motion(motion)
    if start_frame < 0 or start_frame >= selected.length:
        raise ValueError(
            f"start frame {start_frame} is outside motion length {selected.length}"
        )
    positions = np.asarray(selected.anchor_pos_w[start_frame:], dtype=np.float64)
    displacement = np.linalg.norm(positions - positions[0], axis=1)
    maximum = float(displacement.max(initial=0.0))
    anchor = FixedAnchor(
        position=positions[0].astype(np.float32),
        quaternion_xyzw=np.asarray(
            selected.anchor_quat_w[start_frame], dtype=np.float32
        ),
    )
    return anchor, maximum, selected.length - start_frame - 1


class FixedAnchor:
    """Reference start-frame anchor pose: the encoder's frame for the episode."""

    def __init__(self, position, quaternion_xyzw) -> None:
        self.position = position
        self.quaternion_xyzw = quaternion_xyzw


def _cmd_lowlevel_unitree(args) -> int:
    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.native_core import NativeUnitreeLoop

    gate_error = _unitree_write_gate_error(
        enable_writes=args.enable_writes,
        allow_non_realtime=args.allow_non_realtime,
        confirm=args.confirm,
    )
    if gate_error is not None:
        print(f"FAIL: {gate_error}")
        return 2
    if args.enable_writes and args.allow_non_realtime:
        print(
            "WARNING: G1 writes use best-effort timing; SCHED_FIFO setup "
            "failures will not block control"
        )
    bundle = PolicyBundle.load(args.bundle)
    fixed_anchor = None
    if args.command_source == "oracle":
        if not args.fixed_initial_anchor:
            print("FAIL: Unitree oracle control requires --fixed-initial-anchor")
            return 2
        if not args.reference_root or not args.motion or not args.request_slot:
            print(
                "FAIL: Unitree oracle control requires --reference-root, "
                "--motion, and --request-slot"
            )
            return 2
        try:
            fixed_anchor, displacement, available_ticks = _unitree_stationary_anchor(
                bundle,
                args.reference_root,
                args.motion,
                args.start_frame,
                args.fixed_anchor_max_displacement,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            print(f"FAIL: {exc}")
            return 2
        if args.ticks > available_ticks:
            print(
                f"FAIL: --ticks {args.ticks} exceeds the remaining "
                f"{available_ticks} motion ticks"
            )
            return 2
        print(
            "Fixed initial anchor enabled: "
            f"max reference displacement {displacement:.4f} m"
        )
    elif args.fixed_initial_anchor:
        print("FAIL: --fixed-initial-anchor requires --command-source oracle")
        return 2
    runtime = NativeUnitreeLoop(
        bundle,
        args.network,
        response_slot=args.response_slot,
        request_slot=args.request_slot,
        writes_enabled=args.enable_writes,
        create_slots=not args.connect_slots,
        lead_ticks=args.lead_ticks,
        command_source=args.command_source,
        fixed_anchor_position=None if fixed_anchor is None else fixed_anchor.position,
        fixed_anchor_quaternion=(
            None if fixed_anchor is None else fixed_anchor.quaternion_xyzw
        ),
        control_cpu=args.control_cpu,
        writer_cpu=args.writer_cpu,
        control_fifo_priority=args.control_priority,
        writer_fifo_priority=args.writer_priority,
        lock_memory=not args.no_lock_memory,
        require_realtime=not args.allow_non_realtime,
        policy_threads=args.policy_threads,
    )
    if not runtime.wait_for_state(args.state_timeout):
        print("FAIL: no fresh rt/lowstate data before the timeout")
        return 1
    if args.enable_writes:
        try:
            runtime.begin_initialization(args.init_seconds)
        except RuntimeError as exc:
            runtime.force_damp()
            print(f"FAIL: {exc}")
            return 1
        if not runtime.wait_for_mode(NativeUnitreeLoop.WAIT, args.init_seconds + 5.0):
            runtime.force_damp()
            print("FAIL: G1 did not reach the WAIT state after initialization")
            return 1

    if args.console:
        return _run_tracker_console(runtime, args, bundle)

    runtime.start(args.ticks, paced=True)
    engaged = not args.enable_writes
    failed = False
    interrupted = False
    command_deadline = time.monotonic() + args.command_timeout
    try:
        while runtime.running:
            if args.enable_writes and not engaged:
                if int(runtime.stats()["control_ticks"]) > 0:
                    runtime.engage_control()
                    engaged = True
                    print("G1 native control engaged")
                elif time.monotonic() >= command_deadline:
                    raise RuntimeError(
                        "no valid planner command arrived before engage timeout"
                    )
            time.sleep(0.02)
    except KeyboardInterrupt:
        interrupted = True
        # Damp before stopping: force_damp() is a lock-free mode store that the
        # independent 500 Hz writer picks up on its next tick, while stop() and
        # wait() only unwind the control thread. Damping after the join leaves a
        # window in which the robot still tracks, and a hung control thread
        # would keep it tracking forever.
        runtime.force_damp()
        runtime.stop()
    except RuntimeError as exc:
        failed = True
        runtime.force_damp()
        runtime.stop()
        print(f"FAIL: {exc}")
    finally:
        runtime.force_damp()
        runtime.wait()
    report = {"control": runtime.stats(), "writer": runtime.writer_stats()}
    print(json.dumps(report, indent=2))
    return (
        0 if not failed and not interrupted and report["control"]["fault"] == 0 else 1
    )


def _run_tracker_console(runtime, args, bundle) -> int:
    import numpy as np

    from embodied_control.console import KeyConsole
    from embodied_control.lowlevel.tracker_shell import (
        Pose,
        TrackerConsoleState,
        build_tracker_bindings,
        tracker_status,
    )

    poses = [Pose("default-stance"), Pose("hold-current", hold_current=True)]
    if args.reference_root and args.motion:
        try:
            from embodied_control.lowlevel.reference import ReferenceArrays

            reference = ReferenceArrays(args.reference_root)
            selected = reference.motion(args.motion)
            frame = np.asarray(
                selected.joint_qpos[args.start_frame], dtype=np.float32
            )
            poses.append(
                Pose(f"{args.motion}@{args.start_frame}", target=frame.tolist())
            )
        except (FileNotFoundError, KeyError, ValueError, AttributeError) as exc:
            print(f"WARNING: reference pose unavailable: {exc}")

    state = TrackerConsoleState(
        poses=poses, ticks=args.ticks, init_seconds=args.init_seconds
    )
    console = KeyConsole(
        build_tracker_bindings(
            runtime, state, on_note=lambda msg: print(f"  -- {msg}")
        ),
        status=lambda: tracker_status(runtime, state),
    )
    print("tracker console (owns rt/lowcmd)")
    if not args.enable_writes:
        print("READ-ONLY: pass --enable-writes --confirm to command the robot")
    try:
        console.run()
    except KeyboardInterrupt:
        pass
    finally:
        runtime.force_damp()
        runtime.stop()
        runtime.wait()
    print(json.dumps({"control": runtime.stats(), "writer": runtime.writer_stats()}, indent=2))
    return 0


def _lifecycle_reference_pose(job, *, motion=None, start_frame=None):
    """Start-pose joints (Isaac order) and start-frame projected gravity."""
    import numpy as np

    from embodied_control.lowlevel.reference import ReferenceArrays
    from embodied_control.robot.gates import gravity_from_quaternion_xyzw

    motion = motion or job.motion
    start_frame = job.start_frame if start_frame is None else int(start_frame)
    reference = ReferenceArrays(job.reference_root)
    selected = reference.motion(motion)
    if start_frame >= selected.length:
        raise ValueError(
            f"start frame {start_frame} is outside motion length {selected.length}"
        )
    joints = np.asarray(selected.joint_qpos[start_frame], dtype=np.float64)
    quaternion = np.asarray(
        selected.anchor_quat_w[start_frame], dtype=np.float64
    )
    return [float(v) for v in joints], gravity_from_quaternion_xyzw(
        [float(v) for v in quaternion]
    )


# Measured on the plant with the SONIC bundle's own PD gains, robot lowered
# onto its feet: legs hold within 0.06 rad; the 28 N m/rad waist pitch sags
# 0.42 rad and the 14 N m/rad shoulders 0.16 rad under gravity. The gate is
# for gross mismatches (wrong frame, joint order, tilt), not for sag the
# policy was trained against, so the gain-limited groups are loose.
# Ankles carry the stance under 85 N m/rad (3x hold): 0.17 rad off on a
# flexed start frame, so they get their own bound.
POSE_TOLERANCE_BY_GROUP = (("hip", 0.1), ("knee", 0.1), ("ankle", 0.2), ("waist", 0.5))
POSE_TOLERANCE_ARM = 0.5


def _pose_tolerances(bundle, spec):
    """Per-joint pose-match tolerances (Isaac order)."""
    if isinstance(spec, list):
        return [float(v) for v in spec]
    if spec is not None:
        return [float(spec)] * 29
    out = []
    for name in bundle.manifest.action.isaac_joint_names:
        tolerance = POSE_TOLERANCE_ARM
        for key, value in POSE_TOLERANCE_BY_GROUP:
            if key in name:
                tolerance = value
                break
        out.append(tolerance)
    return out


def _tracker_bundle_paths(job) -> dict[str, str]:
    paths = {Path(job.bundle).name: job.bundle}
    paths.update(job.trackers)
    return paths


def _lifecycle_selection(job, tracker: str = ""):
    from embodied_control.robot.session import Selection

    return Selection(
        mode=job.command_source,
        motion=job.motion,
        start_frame=int(job.start_frame),
        tracker=tracker,
    )


def _tracker_for(job, args, bundle, selection):
    """Runtime plus the start pose and reference gravity for one selection."""
    from embodied_control.lowlevel.native_core import NativeUnitreeLoop

    network = args.network or job.network
    fixed_anchor = None
    if selection.mode == "oracle":
        fixed_anchor, displacement, available = _unitree_stationary_anchor(
            bundle,
            job.reference_root,
            selection.motion,
            selection.start_frame,
            job.fixed_anchor_max_displacement,
        )
        # A later start frame leaves fewer frames; the episode budget follows.
        ticks = min(int(job.ticks), int(available))
        print(f"Fixed initial anchor: max reference displacement {displacement:.4f} m; budget {ticks} ticks")
    else:
        ticks = int(job.ticks)
    start_pose = None
    reference_gravity = None
    if job.start_pose == "motion" or (selection.mode == "oracle" and job.start_pose != "default" and not isinstance(job.start_pose, list)):
        start_pose, reference_gravity = _lifecycle_reference_pose(
            job, motion=selection.motion, start_frame=selection.start_frame
        )
    elif job.start_pose == "default":
        start_pose = [float(v) for v in bundle.manifest.action.default_joint_pos]
    else:
        start_pose = [float(v) for v in job.start_pose]
    runtime = NativeUnitreeLoop(
        bundle,
        network,
        response_slot=job.response_slot,
        request_slot=job.request_slot,
        writes_enabled=args.enable_writes,
        create_slots=not job.connect_slots,
        lead_ticks=job.lead_ticks,
        command_source=selection.mode,
        fixed_anchor_position=None if fixed_anchor is None else fixed_anchor.position,
        fixed_anchor_quaternion=(
            None if fixed_anchor is None else fixed_anchor.quaternion_xyzw
        ),
        command_stale_ms=job.command_stale_ms,
        state_absent_ms=job.state_absent_ms,
        control_cpu=job.realtime.control_cpu,
        writer_cpu=job.realtime.writer_cpu,
        control_fifo_priority=job.realtime.control_priority,
        writer_fifo_priority=job.realtime.writer_priority,
        lock_memory=job.realtime.lock_memory,
        require_realtime=not args.allow_non_realtime,
        policy_threads=job.realtime.policy_threads,
        dds_domain=job.dds_domain,
        plan_slots=job.planner.vla_plan_slots if selection.mode == "vla" else 1,
        latent_plan=selection.mode == "vla" and job.planner.vla_reply == "latent_plan",
    )
    return runtime, start_pose, reference_gravity, ticks


def _rehearsal_root(job) -> str:
    """Where to look for a plant rehearsal of this job."""
    if job.rehearsal_root:
        return job.rehearsal_root
    # A sim job and its hardware twin write beside each other, so the parent
    # of this run's artifacts is where the rehearsal lands.
    return str(Path(job.artifacts_dir).parent) if job.artifacts_dir else ""


def _run_identity(job, bundle, network: str) -> dict:
    from embodied_control.robot.rehearsal import run_identity

    source = bundle.manifest.source or {}
    return run_identity(
        bundle_sha=str(source.get("checkpoint_sha256", "")),
        bundle_name=Path(job.bundle).name,
        motion=job.motion,
        command_source=job.command_source,
        network=network,
        start_frame=job.start_frame,
        ticks=job.ticks,
    )


def _lifecycle_config(job, args, bundle, start_pose, reference_gravity, ticks=None):
    from embodied_control.robot.lifecycle import LifecycleConfig

    thresholds = job.thresholds
    return LifecycleConfig(
        start_pose=start_pose,
        reference_gravity=reference_gravity,
        ramp_seconds=job.ramp_seconds,
        ticks=int(job.ticks if ticks is None else ticks),
        blend_ticks=job.blend_ticks,
        end_state=job.end_state,
        damp_hands_back=job.damp_hands_back,
        retake_precheck=job.retake_precheck,
        require_rehearsal=job.require_rehearsal,
        rehearsal_root=_rehearsal_root(job),
        rehearsal_max_age_days=job.rehearsal_max_age_days,
        vendor_name=job.vendor_name,
        require_vendor=job.require_vendor,
        allow_non_realtime=args.allow_non_realtime,
        damp_publish_frames=thresholds.damp_publish_frames,
        settle_seconds=thresholds.settle_seconds,
        settle_timeout_seconds=thresholds.settle_timeout_seconds,
        drift_rad=thresholds.drift_rad,
        settle_position_rad=thresholds.settle_position_rad,
        ramp_fault_rad=thresholds.ramp_fault_rad,
        ramp_fault_ms=thresholds.ramp_fault_ms,
        hold_gain_scale=thresholds.hold_gain_scale,
        slack_on_run=job.slack_on_run,
        pin_reference=job.pin_reference,
        hoist_release_seconds=thresholds.hoist_release_seconds,
        control_hz=float(bundle.manifest.rates.control_hz),
        pose_tolerance_rad=_pose_tolerances(bundle, thresholds.pose_tolerance_rad),
        tilt_tolerance_degrees=thresholds.tilt_tolerance_degrees,
        first_action_rad=thresholds.first_action_rad,
        first_action_torque_ratio=thresholds.first_action_torque_ratio,
        stiffness=[float(v) for v in bundle.manifest.action.stiffness],
        effort_limit=(
            [float(v) for v in bundle.manifest.action.effort_limit]
            if bundle.manifest.action.effort_limit
            else None
        ),
        command_timeout_seconds=thresholds.command_timeout_seconds,
        vendor_timeout_seconds=thresholds.vendor_timeout_seconds,
    )


def _lifecycle_peers(job, args):
    """The vendor runtime and the sim hoist client, shared across rebuilds."""
    from embodied_control.lowlevel.native_core import NativePlantClient
    from embodied_control.robot import open_robot

    network = args.network or job.network
    vendor = None
    if job.require_vendor or job.vendor_name:
        vendor = open_robot(
            "g1",
            network_interface=network,
            writes_enabled=args.enable_writes,
            dds_domain=job.dds_domain,
            timeout_seconds=job.vendor_rpc_timeout_seconds,
        )
    hoist = NativePlantClient(network, dds_domain=job.dds_domain) if job.sim_hoist else None
    return vendor, hoist


def _resolved_bundle(path, args):
    """A bundle directory, fetched first when it carries a model pin."""
    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.models import ensure_model

    return PolicyBundle.load(
        ensure_model(
            path,
            offline=getattr(args, "offline", False),
            token=getattr(args, "token", "") or None,
        )
    )


def _load_lifecycle_job(args):
    from embodied_control.robot.lifecycle_job import load_lifecycle_job

    job = load_lifecycle_job(args.job)
    gate_error = _unitree_write_gate_error(
        enable_writes=args.enable_writes,
        allow_non_realtime=args.allow_non_realtime,
        confirm=args.confirm,
    )
    if gate_error is not None:
        raise ValueError(gate_error)
    return job, _resolved_bundle(job.bundle, args)


def _build_lifecycle(args):
    """Job -> (job, lifecycle, runtime, vendor) for the scripted `run`."""
    from embodied_control.robot.lifecycle import Lifecycle, LifecycleLog

    job, bundle = _load_lifecycle_job(args)
    selection = _lifecycle_selection(job)
    runtime, start_pose, reference_gravity, ticks = _tracker_for(job, args, bundle, selection)
    vendor, hoist = _lifecycle_peers(job, args)
    config = _lifecycle_config(job, args, bundle, start_pose, reference_gravity, ticks)
    artifacts = args.artifacts or job.artifacts_dir or None
    lifecycle = Lifecycle(
        runtime,
        vendor,
        config,
        hoist=hoist,
        auto_ack=bool(job.sim_hoist or args.auto_ack),
        identity=_run_identity(job, bundle, args.network or job.network),
        log=LifecycleLog(artifacts),
        note=lambda msg: print(f"  -- {msg}", flush=True),
    )
    return job, lifecycle, runtime, vendor


def _episode_mpjpe(job, bundle_for):
    """In-line MPJPE from an episode's telemetry, when MuJoCo and an MJCF exist."""
    if not job.mjcf or not job.reference_root:
        return None

    def grade(directory, selection):
        import numpy as np

        from embodied_control.lowlevel.eval_mpjpe import _align_reference
        from embodied_control.lowlevel.metrics import compute_mpjpe, fk_body_positions
        from embodied_control.lowlevel.reference import ReferenceArrays

        if selection.mode != "oracle":
            return None
        telemetry = np.load(directory / "telemetry.npz")
        joint = telemetry["joint_position_log"]
        anchor = telemetry["anchor_pose_log"]
        frames = telemetry["reference_frames"]
        valid = np.isfinite(joint).all(axis=1) & np.isfinite(anchor).all(axis=1) & (frames >= 0)
        if valid.sum() < 2:
            return None
        joint, anchor, frames = joint[valid], anchor[valid], frames[valid]
        arrays = ReferenceArrays(job.reference_root)
        motion = arrays.motion(selection.motion)
        if motion.body_pos_w is None or not arrays.body_names:
            return None
        bundle = bundle_for(selection)
        robot_body = fk_body_positions(job.mjcf, bundle.manifest.action, joint, anchor, arrays.body_names)
        reference_body = motion.body_pos_w[frames]
        aligned_body = _align_reference(
            anchor[0], motion.anchor_pos_w[frames[0]], motion.anchor_quat_w[frames[0]],
            reference_body.reshape(-1, 3),
        ).reshape(reference_body.shape)
        aligned_root = _align_reference(
            anchor[0], motion.anchor_pos_w[frames[0]], motion.anchor_quat_w[frames[0]],
            motion.anchor_pos_w[frames],
        )
        record = compute_mpjpe(robot_body, anchor[:, 0:3], aligned_body, aligned_root)
        return {k: v for k, v in record.items() if isinstance(v, (int, float))}

    return grade


def _build_session(args):
    """Job -> ExperimentSession: the command center behind the console."""
    from embodied_control.lowlevel.reference import ReferenceArrays
    from embodied_control.robot.lifecycle import Lifecycle, LifecycleLog
    from embodied_control.robot.session import (
        ExperimentSession,
        SessionConfig,
        SubprocessPlanner,
        oracle_worker_argv,
        planner_worker_argv,
    )

    from embodied_control.robot.isolation import (
        ThreadPinner,
        child_preexec,
        non_realtime_cores,
    )

    job, default_bundle = _load_lifecycle_job(args)
    tracker_paths = _tracker_bundle_paths(job)
    default_tracker = next(iter(tracker_paths))
    bundles = {default_tracker: default_bundle}

    def bundle_for(selection):
        name = selection.tracker or default_tracker
        if name not in tracker_paths:
            raise ValueError(f"unknown tracker {name}")
        if name not in bundles:
            bundles[name] = _resolved_bundle(tracker_paths[name], args)
        return bundles[name]
    # The control thread and the writer own their cores at SCHED_FIFO 80/90.
    # Everything the operator touches lives on the rest.
    free_cores = non_realtime_cores(
        (job.realtime.control_cpu, job.realtime.writer_cpu)
    )
    pinner = ThreadPinner(free_cores)
    planner_preexec = child_preexec(free_cores)
    catalog: list[str] = []
    lengths: dict[str, int] = {}
    if job.reference_root:
        arrays = ReferenceArrays(job.reference_root)
        catalog = list(arrays.motion_names)
        lengths = {name: int(arrays.motion(name).length) for name in catalog}
    artifacts = args.artifacts or job.artifacts_dir or None
    artifacts_path = Path(artifacts) if artifacts else None
    if artifacts_path is not None:
        artifacts_path.mkdir(parents=True, exist_ok=True)
    vendor, hoist = _lifecycle_peers(job, args)
    log = LifecycleLog(artifacts)
    notes: list = []

    def note(message: str) -> None:
        for sink in notes:
            sink(message)

    def planner_factory(selection):
        report = str(artifacts_path / f"planner_{selection.mode}.json") if artifacts_path else ""
        planner_log = artifacts_path / "planner.log" if artifacts_path else None
        if selection.mode == "oracle":
            selected_bundle = bundle_for(selection)
            argv = oracle_worker_argv(
                str(selected_bundle.root), job.reference_root, selection.motion, selection.start_frame,
                job.request_slot, job.response_slot,
                horizon=job.planner.oracle_horizon or None, report=report,
            )
        else:
            if not job.planner.vla_service_command:
                raise ValueError("job.planner.vla_service_command is empty")
            argv = planner_worker_argv(
                job.planner.vla_service_command, job.request_slot, job.response_slot,
                reply=job.planner.vla_reply, z_dim=job.planner.vla_z_dim,
                plan_slots=job.planner.vla_plan_slots, hold_steps=job.planner.vla_hold_steps,
                lead_ticks=job.lead_ticks, report=report,
            )
        return SubprocessPlanner(argv, planner_log, preexec=planner_preexec)

    pending: dict = {}

    def tracker_factory(selection):
        bundle = bundle_for(selection)
        runtime, start_pose, reference_gravity, ticks = _tracker_for(job, args, bundle, selection)
        pending["config"] = _lifecycle_config(job, args, bundle, start_pose, reference_gravity, ticks)
        # The console can switch bundle and motion between episodes, so the
        # identity is rebuilt with the tracker rather than read once.
        pending["identity"] = _run_identity(
            job, bundle, args.network or job.network
        ) | {"motion": selection.motion or job.motion}
        return runtime

    def lifecycle_factory(tracker, selection, session):
        return Lifecycle(
            tracker,
            vendor,
            pending["config"],
            hoist=hoist,
            auto_ack=bool(job.sim_hoist or args.auto_ack),
            identity=pending.get("identity", {}),
            log=log,
            note=note,
        )

    session = ExperimentSession(
        SessionConfig(
            catalog=catalog,
            motion_lengths=lengths,
            trackers=list(tracker_paths),
            artifacts_dir=artifacts,
            planner_autostart=bool(getattr(args, "planner_autostart", False)),
        ),
        _lifecycle_selection(job, default_tracker),
        hoist=hoist,
        planner_factory=planner_factory,
        tracker_factory=tracker_factory,
        lifecycle_factory=lifecycle_factory,
        slot_names=[job.request_slot, job.response_slot] if job.connect_slots else [],
        note=note,
        mpjpe=_episode_mpjpe(job, bundle_for),
    )
    session.note_sinks = notes
    session.pinner = pinner
    return job, session, vendor


def _cmd_lifecycle_console(args) -> int:
    import threading

    from embodied_control.robot.isolation import child_preexec

    from embodied_control.console import KeyConsole
    from embodied_control.robot.shell import build_session_bindings

    try:
        job, session, vendor = _build_session(args)
    except (ImportError, RuntimeError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"FAIL: {exc}")
        return 2
    stop = threading.Event()
    watch_note = [lambda message: print(f"  !! {message}", flush=True)]

    def watch() -> None:
        session.pinner.apply()
        while not stop.is_set():
            try:
                session.poll()
            except Exception as exc:  # the watcher must outlive any one fault
                watch_note[0](f"poll: {exc}")
            stop.wait(0.01)

    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    bindings = build_session_bindings(session)
    use_tui = not args.plain and sys.stdin.isatty() and sys.stdout.isatty()
    try:
        if use_tui:
            from embodied_control.robot.diagnostics import DiagnosticAgent
            from embodied_control.robot.tui import LifecycleTui

            # The agent is an LLM CLI: off the robot's cores and renice'd, so
            # a diagnosis never competes with the control loop.
            agent = DiagnosticAgent(
                args.diagnostic_agent,
                cwd=Path.cwd(),
                preexec=child_preexec(session.pinner.cores, nice=10),
            )
            diagnose = agent.diagnose if agent.available else None
            session.pinner.apply()
            tui = LifecycleTui(
                session,
                bindings,
                diagnose=diagnose,
                agent_label=agent.label if args.diagnostic_agent != "off" else "off",
                pin=session.pinner.apply,
                theme=getattr(args, "theme", "auto"),
            )
            watch_note[0] = tui.note
            tui.note(
                f"console threads on {session.pinner.describe()}; robot threads keep "
                f"cpu {job.realtime.control_cpu} and {job.realtime.writer_cpu}"
            )
            session.note_sinks.append(tui.note)
            if not args.enable_writes:
                tui.note("READ-ONLY: pass --enable-writes --confirm to command the robot")
            if args.diagnostic_agent != "off" and not agent.available:
                tui.note("diagnostic agent unavailable: install or authenticate Codex/Claude Code")
            tui.note(
                f"job {args.job}: {session.selection.label()}",
                "info",
            )
            tui.note(
                "press r to build the tracker; in vla mode press p first, "
                "because that planner loads its own checkpoint",
                "info",
            )
            tui.run()
        else:
            session.note_sinks.append(lambda msg: print(f"  -- {msg}", flush=True))
            console = KeyConsole(bindings, status=lambda: session.snapshot()["state"])
            print(f"lifecycle console  bundle={job.bundle}  network={args.network or job.network}")
            if not args.enable_writes:
                print("READ-ONLY: pass --enable-writes --confirm to command the robot")
            console.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        session.close()
        if vendor is not None:
            vendor.close(damp=False)
    print(json.dumps({"episodes": session.episodes, "selection": session.selection.label()}, indent=2))
    return 0


def _cmd_lifecycle_run(args) -> int:
    from embodied_control.robot.lifecycle import LifecycleState

    try:
        until = LifecycleState(args.until)
    except ValueError:
        print(f"FAIL: unknown state {args.until}")
        return 2
    try:
        job, lifecycle, runtime, vendor = _build_lifecycle(args)
    except (ImportError, RuntimeError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"FAIL: {exc}")
        return 2
    failed = False
    try:
        result = lifecycle.auto(until)
        if not result.ok:
            failed = True
            print(f"FAIL at {lifecycle.state}: {result.detail}")
        elif args.go and lifecycle.state is LifecycleState.PRIMED:
            result = lifecycle.go()
            if not result.ok:
                failed = True
                print(f"FAIL at {lifecycle.state}: {result.detail}")
            while runtime.running and lifecycle.state is LifecycleState.RUNNING:
                lifecycle.poll()
                time.sleep(0.01)
            lifecycle.poll()
            if lifecycle.state is LifecycleState.FAULT:
                failed = True
                print(f"FAULT: {lifecycle.fault_reason}")
        if args.recover and lifecycle.state in {
            LifecycleState.HOLD, LifecycleState.DAMP, LifecycleState.FAULT,
        }:
            result = lifecycle.recover()
            if not result.ok:
                failed = True
                print(f"FAIL recovering at {lifecycle.state}: {result.detail}")
    except KeyboardInterrupt:
        failed = True
        lifecycle.damp()
    finally:
        if not args.recover:
            lifecycle.shutdown()
        else:
            lifecycle.log.finish(lifecycle.summary())
        runtime.force_damp()
        if runtime.running:
            runtime.stop()
            runtime.wait()
        if vendor is not None:
            vendor.close(damp=False)
    summary = lifecycle.summary()
    summary["control"] = runtime.stats()
    summary["writer"] = runtime.writer_stats()
    print(json.dumps(summary, indent=2))
    return 1 if failed or lifecycle.state is LifecycleState.FAULT else 0


def _cmd_lowlevel_plant(args) -> int:
    import numpy as np

    from embodied_control.robot.plant import load_plant_config
    from embodied_control.sim.dds_plant import NativeDdsPlant

    robot = load_plant_config(args.robot)
    plant = NativeDdsPlant(
        robot,
        args.model,
        args.network,
        timestep=args.timestep,
        mode_machine=args.mode_machine,
        physics_cpu=args.physics_cpu,
        physics_fifo_priority=args.physics_fifo_priority,
        lock_memory=args.lock_memory,
        require_realtime=args.require_realtime,
        sensor_noise={
            "joint_pos": args.noise_joint_pos,
            "joint_vel": args.noise_joint_vel,
            "base_ang_vel": args.noise_base_ang_vel,
            "imu_tilt_rad": args.noise_imu_tilt_rad,
        },
        noise_seed=args.noise_seed,
        dds_domain=args.dds_domain,
        freeze_until_command=args.freeze_until_command,
        vendor=args.vendor,
        vendor_name=args.vendor_name,
        hoist=args.hoist,
        hoist_clearance=args.hoist_clearance,
        # One row per publish; the plant serves at 1 / timestep.
        state_log_capacity=(
            int(args.seconds / args.timestep) + 1024 if args.states else 0
        ),
    )
    if args.initial_pose:
        plant.set_initial_pose(np.load(args.initial_pose))
        plant.reset()
    plant.start()
    print("PLANT_READY", flush=True)
    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    view = {}

    def finished() -> bool:
        return not plant.running or (deadline is not None and time.monotonic() >= deadline)

    try:
        if args.viewer or args.video:
            from embodied_control.lowlevel.plant_view import watch

            view = watch(
                plant,
                args.model,
                robot.joint_names,
                live=args.viewer,
                video=args.video,
                fps=args.view_fps,
                width=args.view_width,
                height=args.view_height,
                camera=args.view_camera,
                host=args.viewer_host,
                port=args.viewer_port,
                should_stop=finished,
            )
        else:
            while not finished():
                time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        plant.stop()
        plant.wait_for_stop()
    report = plant.stats()
    report.update(view)
    if args.states:
        rows = plant.state_log()
        states_path = Path(args.states).resolve()
        states_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            states_path,
            root_pos=rows[:, 0:3],
            root_quat_xyzw=rows[:, 3:7],
            joint_pos=rows[:, 7:],
            joint_names=np.asarray(robot.joint_names),
            publish_hz=np.asarray(1.0 / args.timestep, dtype=np.float64),
        )
        report["states_path"] = str(states_path)
        report["state_rows"] = int(rows.shape[0])
    if args.report:
        output = Path(args.report).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if not report["physics_fault"] and report["publishes"] > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ec", description="Embodied-Control eval orchestrator"
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check host + dependencies").set_defaults(
        func=_cmd_doctor
    )

    ev = sub.add_parser("eval", help="evaluation commands")
    evs = ev.add_subparsers(dest="eval_command", required=True)
    run = evs.add_parser("run", help="run an eval job")
    run.add_argument("job")
    run.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="override outputs.log_level from the job file",
    )
    run.set_defaults(func=_cmd_eval_run)
    val = evs.add_parser("validate", help="validate a run directory")
    val.add_argument("run_dir")
    val.set_defaults(func=_cmd_eval_validate)
    lst = evs.add_parser("list-runs", help="list runs under a root")
    lst.add_argument("root")
    lst.set_defaults(func=_cmd_eval_list_runs)

    pol = sub.add_parser("policy", help="policy service commands")
    pols = pol.add_subparsers(dest="policy_command", required=True)
    serve = pols.add_parser("serve", help="run a blank policy service")
    serve.add_argument(
        "--type", default="zero", choices=["zero", "random", "image_stats"]
    )
    serve.add_argument("--action-dim", type=int, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8756)
    serve.add_argument("--seed", type=int, default=0)
    serve.set_defaults(func=_cmd_policy_serve)
    ping = pols.add_parser("ping", help="health + describe an endpoint")
    ping.add_argument("--endpoint", required=True, help="host:port")
    ping.add_argument("--timeout", type=float, default=10.0)
    ping.set_defaults(func=_cmd_policy_ping)

    low = sub.add_parser("lowlevel", help="50 Hz tracker runtime commands")
    lows = low.add_subparsers(dest="lowlevel_command", required=True)
    lrun = lows.add_parser("run", help="run a lowlevel job (needs the lowlevel env)")
    lrun.add_argument("job")
    lrun.add_argument("--device", default="cpu")
    lrun.add_argument(
        "--viewer",
        action="store_true",
        help="serve a live browser viewer and pace MuJoCo to simulation time",
    )
    lrun.add_argument("--viewer-host", default="127.0.0.1")
    lrun.add_argument("--viewer-port", type=int, default=8765)
    lrun.add_argument("--viewer-fps", type=int, default=25)
    lrun.set_defaults(func=_cmd_lowlevel_run)
    lver = lows.add_parser("verify-bundle", help="replay a bundle's golden traces")
    lver.add_argument("bundle")
    lver.add_argument("--atol", type=float, default=1e-5)
    lver.set_defaults(func=_cmd_lowlevel_verify_bundle)
    lnver = lows.add_parser(
        "verify-native-bundle", help="replay golden traces in C++ ONNX Runtime"
    )
    lnver.add_argument("bundle")
    lnver.set_defaults(func=_cmd_lowlevel_verify_native_bundle)
    lnbench = lows.add_parser(
        "bench-native", help="measure the callback-free native control hot path"
    )
    lnbench.add_argument("bundle")
    lnbench.add_argument("--ticks", type=int, default=10_000)
    lnbench.add_argument("--policy-threads", type=int, default=4)
    lnbench.add_argument("--lead-ticks", type=int, default=4)
    lnbench.add_argument("--paced", action="store_true")
    lnbench.add_argument("--cpu", type=int, default=-1)
    lnbench.add_argument("--fifo-priority", type=int, default=0)
    lnbench.add_argument("--lock-memory", action="store_true")
    lnbench.add_argument("--require-realtime", action="store_true")
    lnbench.set_defaults(func=_cmd_lowlevel_bench_native)
    lnmj = lows.add_parser(
        "mujoco-native", help="run the C++ ONNX tracker and MuJoCo loop"
    )
    lnmj.add_argument("bundle")
    lnmj.add_argument("--model", required=True)
    lnmj.add_argument("--response-slot", required=True)
    lnmj.add_argument("--request-slot", default="")
    lnmj.add_argument("--connect-slots", action="store_true")
    lnmj.add_argument("--command-source", choices=["vla", "oracle"], default="vla")
    lnmj.add_argument("--ticks", type=int, default=500)
    lnmj.add_argument("--lead-ticks", type=int, default=4)
    lnmj.add_argument(
        "--latent-plan",
        action="store_true",
        help="consume planner replies as [plan_slots, z_dim] latent plans",
    )
    lnmj.add_argument("--plan-slots", type=int, default=1)
    lnmj.add_argument("--policy-threads", type=int, default=4)
    lnmj.add_argument("--command-absent-ticks", type=int, default=100)
    lnmj.add_argument("--command-stale-ms", type=float, default=500.0)
    lnmj.add_argument("--cpu", type=int, default=-1)
    lnmj.add_argument("--fifo-priority", type=int, default=0)
    lnmj.add_argument("--lock-memory", action="store_true")
    lnmj.add_argument("--require-realtime", action="store_true")
    lnmj.add_argument("--physics-cpu", type=int, default=-1)
    lnmj.add_argument("--physics-fifo-priority", type=int, default=0)
    lnmj.add_argument("--physics-lock-memory", action="store_true")
    lnmj.add_argument("--physics-require-realtime", action="store_true")
    lnmj.add_argument("--report", default="")
    lnmj.add_argument("--mpjpe", action="store_true",
                      help="compute MPJPE-L/G; needs oracle source + --reference-root/--motion")
    lnmj.add_argument("--reference-root", default="")
    lnmj.add_argument("--motion", default="")
    lnmj.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help=(
            "reference frame the oracle worker was started at; must match the "
            "worker's --start-frame or the MPJPE row scores the wrong frames"
        ),
    )
    lnmj.add_argument("--telemetry-dir", default="")
    lnmj.add_argument("--telemetry-hz", type=float, default=1.0)
    lnmj.add_argument("--viewer", action="store_true")
    lnmj.add_argument("--viewer-host", default="127.0.0.1")
    lnmj.add_argument("--viewer-port", type=int, default=8765)
    lnmj.add_argument("--viewer-fps", type=int, default=25)
    lnmj.set_defaults(func=_cmd_lowlevel_mujoco_native)
    lnplanner = lows.add_parser(
        "planner-worker",
        help="bridge native planner slots to a GR00T stdio service",
    )
    lnplanner.add_argument("--request-slot", required=True)
    lnplanner.add_argument("--response-slot", required=True)
    lnplanner.add_argument("--create-slots", action="store_true")
    lnplanner.add_argument("--hold-steps", type=int, default=10)
    lnplanner.add_argument("--lead-ticks", type=int, default=4)
    lnplanner.add_argument("--state-width", type=int, default=38)
    lnplanner.add_argument(
        "--reply",
        choices=("chunk", "latent_plan"),
        default="chunk",
        help=(
            "chunk: a root_qpos window the tracker-side encoder turns into one "
            "latent (one head call per hold). latent_plan: the head's own "
            "[slots, z_dim] plan, walked by the controller (one head call per "
            "plan_slots holds)."
        ),
    )
    lnplanner.add_argument("--z-dim", type=int, default=256)
    lnplanner.add_argument("--plan-slots", type=int, default=1)
    lnplanner.add_argument("--rtc", action="store_true")
    lnplanner.add_argument("--report", default="")
    lnplanner.add_argument("service_command", nargs=argparse.REMAINDER)
    lnplanner.set_defaults(func=_cmd_lowlevel_planner_worker)
    lnoracle = lows.add_parser(
        "oracle-worker",
        help="stream a fixed reference motion to the native root_qpos encoder",
    )
    lnoracle.add_argument("bundle")
    lnoracle.add_argument("--request-slot", required=True)
    lnoracle.add_argument("--response-slot", required=True)
    lnoracle.add_argument("--reference-root", required=True)
    lnoracle.add_argument("--motion", required=True)
    lnoracle.add_argument("--start-frame", type=int, default=0)
    lnoracle.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="reference frames per reply; 0 uses the bundle's encoder window",
    )
    lnoracle.add_argument("--create-slots", action="store_true")
    lnoracle.add_argument("--report", default="")
    lnoracle.set_defaults(func=_cmd_lowlevel_oracle_worker)

    lcert = lows.add_parser(
        "certify-report", help="grade a native runtime report against the timing SLOs"
    )
    lcert.add_argument("report")
    lcert.add_argument("--output", default="")
    lcert.set_defaults(func=_cmd_lowlevel_certify_report)
    lcompare = lows.add_parser(
        "compare-unitree-pose",
        help="render a live read-only G1 joint snapshot in MuJoCo",
    )
    lcompare.add_argument("bundle")
    lcompare.add_argument("--model", required=True)
    lcompare.add_argument("--network", required=True)
    lcompare.add_argument("--output", required=True)
    lcompare.add_argument("--samples", type=int, default=500)
    lcompare.add_argument("--timeout", type=float, default=5.0)
    lcompare.add_argument("--root-height", type=float, default=0.76)
    lcompare.add_argument("--dds-domain", type=int, default=0)
    lcompare.set_defaults(func=_cmd_lowlevel_compare_unitree_pose)
    lcheck = lows.add_parser(
        "check-unitree", help="read-only health check for a live G1 DDS link"
    )
    lcheck.add_argument("--network", required=True)
    lcheck.add_argument("--samples", type=int, default=500)
    lcheck.add_argument("--timeout", type=float, default=5.0)
    lcheck.add_argument("--min-rate", type=float, default=100.0)
    lcheck.add_argument("--max-gap-ms", type=float, default=100.0)
    lcheck.add_argument("--dds-domain", type=int, default=0)
    lcheck.add_argument("--report", default="")
    lcheck.add_argument("--json", action="store_true")
    lcheck.set_defaults(func=_cmd_lowlevel_check_unitree)
    lunitree = lows.add_parser(
        "unitree", help="run the native G1 DDS tracker (writes off by default)"
    )
    lunitree.add_argument("bundle")
    lunitree.add_argument("--network", required=True)
    lunitree.add_argument("--response-slot", required=True)
    lunitree.add_argument("--request-slot", default="")
    lunitree.add_argument("--connect-slots", action="store_true")
    lunitree.add_argument("--ticks", type=int, default=500)
    lunitree.add_argument("--lead-ticks", type=int, default=4)
    lunitree.add_argument(
        "--command-source", choices=["vla", "oracle"], default="vla"
    )
    lunitree.add_argument("--fixed-initial-anchor", action="store_true")
    lunitree.add_argument("--reference-root", default="")
    lunitree.add_argument("--motion", default="")
    lunitree.add_argument("--start-frame", type=int, default=0)
    lunitree.add_argument(
        "--fixed-anchor-max-displacement", type=float, default=0.05
    )
    lunitree.add_argument("--policy-threads", type=int, default=1)
    lunitree.add_argument("--control-cpu", type=int, default=2)
    lunitree.add_argument("--writer-cpu", type=int, default=3)
    lunitree.add_argument("--control-priority", type=int, default=80)
    lunitree.add_argument("--writer-priority", type=int, default=90)
    lunitree.add_argument("--state-timeout", type=float, default=10.0)
    lunitree.add_argument("--command-timeout", type=float, default=10.0)
    lunitree.add_argument("--init-seconds", type=float, default=3.0)
    lunitree.add_argument("--no-lock-memory", action="store_true")
    lunitree.add_argument("--allow-non-realtime", action="store_true")
    lunitree.add_argument("--enable-writes", action="store_true")
    lunitree.add_argument("--confirm", default="")
    lunitree.add_argument(
        "--console",
        action="store_true",
        help="drive the tracker from a single-keypress operator console",
    )
    lunitree.set_defaults(func=_cmd_lowlevel_unitree)
    lplant = lows.add_parser(
        "plant", help="serve MuJoCo physics on the G1 hardware DDS protocol"
    )
    lplant.add_argument("robot", help="standalone robot/plant YAML; no policy bundle")
    lplant.add_argument("--model", required=True)
    lplant.add_argument("--network", default="lo")
    lplant.add_argument("--timestep", type=float, default=0.002)
    lplant.add_argument("--mode-machine", type=int, default=5)
    lplant.add_argument(
        "--seconds", type=float, default=0.0, help="0 serves until Ctrl-C"
    )
    lplant.add_argument(
        "--initial-pose",
        default="",
        help=".npy with the 36-value start pose in plant-config joint order",
    )
    # SONIC's policy-group observation noise, as uniform half-ranges. The
    # rehearsal protocol runs WITH noise by default (user directive
    # 2026-08-17): a clean rehearsal hid a real fall-free drop.
    lplant.add_argument("--noise-joint-pos", type=float, default=0.01)
    lplant.add_argument("--noise-joint-vel", type=float, default=0.5)
    lplant.add_argument("--noise-base-ang-vel", type=float, default=0.2)
    lplant.add_argument("--noise-imu-tilt-rad", type=float, default=0.05)
    lplant.add_argument("--noise-seed", type=int, default=0)
    lplant.add_argument(
        "--dds-domain",
        type=int,
        default=0,
        help="0 is the robot's domain; isolate simulated plant pairs above it",
    )
    lplant.add_argument(
        "--states",
        default="",
        help=(
            ".npz for the plant's TRUE state trajectory (pos, quat XYZW, "
            "joints in plant-config order, with joint_names). The DDS wire carries no root pose, so "
            "this is the only ground truth for MPJPE on this tier."
        ),
    )
    lplant.add_argument(
        "--freeze-until-command",
        action="store_true",
        help=(
            "hold the initial pose (serve state, integrate nothing) until the "
            "first controller command arrives; a mid-stride reference frame "
            "falls over long before a controller finishes booting"
        ),
    )
    lplant.add_argument(
        "--vendor",
        action="store_true",
        help=(
            "serve the G1's sport and motion_switcher RPC services and own "
            "the joints until ReleaseMode, rejecting rt/lowcmd until then"
        ),
    )
    lplant.add_argument("--vendor-name", default="ai")
    lplant.add_argument(
        "--hoist",
        action="store_true",
        help="hang the pelvis from a virtual gantry until `lower` is requested",
    )
    lplant.add_argument(
        "--hoist-clearance",
        type=float,
        default=0.10,
        help=(
            "metres of floor clearance the gantry hangs the robot at, the "
            "number the operator manual asks for on hardware; the ramp to a "
            "start pose extends the legs ~4 cm, so a smaller value stands "
            "the rehearsal on the floor. 0 hangs it where it spawns"
        ),
    )
    lplant.add_argument(
        "--viewer",
        action="store_true",
        help=(
            "serve an interactive 3D view of the plant's true state over "
            "HTTP (mjviser/Viser; a render-only copy, so a slow frame never "
            "reaches the physics thread) with native mouse orbit/pan/zoom; "
            "open the printed VIEWER_URL in a browser, or "
            "`ssh -L <port>:localhost:<port>` first if the plant is remote"
        ),
    )
    lplant.add_argument(
        "--video", default="", help="also record the same view to this mp4"
    )
    lplant.add_argument("--view-fps", type=int, default=30)
    lplant.add_argument("--view-width", type=int, default=960)
    lplant.add_argument("--view-height", type=int, default=540)
    lplant.add_argument(
        "--view-camera", default="", help="named MJCF camera (default: free)"
    )
    lplant.add_argument(
        "--viewer-host",
        default="127.0.0.1",
        help="bind address for --viewer's HTTP server (default: loopback only)",
    )
    lplant.add_argument(
        "--viewer-port", type=int, default=8765, help="port for --viewer's HTTP server"
    )
    lplant.add_argument("--physics-cpu", type=int, default=-1)
    lplant.add_argument("--physics-fifo-priority", type=int, default=0)
    lplant.add_argument("--lock-memory", action="store_true")
    lplant.add_argument("--require-realtime", action="store_true")
    lplant.add_argument("--report", default="")
    lplant.set_defaults(func=_cmd_lowlevel_plant)

    mdl = sub.add_parser(
        "models",
        help="pinned controller and planner assets under assets/models/",
    )
    mdls = mdl.add_subparsers(dest="models_command", required=True)
    mlist = mdls.add_parser("list", help="every pinned model and whether it is here")
    mlist.add_argument("--root", default=DEFAULT_MODEL_ROOT)
    mlist.set_defaults(func=_cmd_models_list)
    mpull = mdls.add_parser("pull", help="fetch a pinned model's missing files")
    mpull.add_argument("directory", nargs="?", default="")
    mpull.add_argument("--root", default=DEFAULT_MODEL_ROOT)
    mpull.add_argument("--all", action="store_true", help="every pin under --root")
    mpull.add_argument("--token", default="", help="Hugging Face token")
    mpull.set_defaults(func=_cmd_models_pull)
    mver = mdls.add_parser("verify", help="hash the local files against the pin")
    mver.add_argument("directory")
    mver.set_defaults(func=_cmd_models_verify)
    mpin = mdls.add_parser(
        "pin", help="record an existing repository's files as this directory's pin"
    )
    mpin.add_argument("directory")
    mpin.add_argument("--repo", required=True, help="Hugging Face repo id")
    mpin.add_argument("--path", default="", help="subdirectory inside the repo")
    mpin.add_argument(
        "--kind", choices=("controller", "planner", "reference"), default="controller"
    )
    mpin.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    mpin.add_argument(
        "--revision", default="", help="commit sha; default is the repo head"
    )
    mpin.add_argument("--token", default="")
    mpin.set_defaults(func=_cmd_models_pin)
    mfetch = mdls.add_parser(
        "fetch", help="download a repository folder and pin it here"
    )
    mfetch.add_argument("directory")
    mfetch.add_argument("--repo", required=True, help="Hugging Face repo id")
    mfetch.add_argument("--path", default="", help="subdirectory inside the repo")
    mfetch.add_argument(
        "--kind", choices=("controller", "planner", "reference"), default="controller"
    )
    mfetch.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    mfetch.add_argument(
        "--revision", default="", help="commit sha; default is the repo head"
    )
    mfetch.add_argument("--token", default="")
    mfetch.set_defaults(func=_cmd_models_fetch)
    mpush = mdls.add_parser("push", help="upload a bundle and pin the new commit")
    mpush.add_argument("directory")
    mpush.add_argument("--repo", required=True, help="Hugging Face repo id")
    mpush.add_argument("--path", default="", help="subdirectory inside the repo")
    mpush.add_argument(
        "--kind", choices=("controller", "planner", "reference"), default="controller"
    )
    mpush.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    mpush.add_argument(
        "--public",
        action="store_true",
        help="create the repository public; the default is private",
    )
    mpush.add_argument("--message", default="")
    mpush.add_argument("--token", default="")
    mpush.set_defaults(func=_cmd_models_push)

    life = sub.add_parser(
        "lifecycle",
        help="the gated hoist-to-run lifecycle (docs/design/robot_lifecycle.md)",
    )
    lifes = life.add_subparsers(dest="lifecycle_command", required=True)
    for name, helptext in (
        ("console", "single-keypress operator console over the lifecycle"),
        ("run", "scripted: advance to a state, optionally go and recover"),
    ):
        parser = lifes.add_parser(name, help=helptext)
        parser.add_argument("job", help="lifecycle job YAML")
        parser.add_argument(
            "--offline",
            action="store_true",
            help="refuse to fetch a pinned bundle; require it on disk already",
        )
        parser.add_argument("--network", default="", help="overrides the job")
        parser.add_argument("--artifacts", default="", help="overrides the job")
        parser.add_argument("--allow-non-realtime", action="store_true")
        parser.add_argument("--enable-writes", action="store_true")
        parser.add_argument("--confirm", default="")
        parser.add_argument(
            "--auto-ack",
            action="store_true",
            help="acknowledge hoist/lower steps without an operator",
        )
        parser.add_argument(
            "--planner-autostart",
            action="store_true",
            help="start the planner with the tracker; off by default because "
            "a VLA planner loads gigabytes of its own",
        )
        if name == "console":
            parser.add_argument(
                "--plain",
                action="store_true",
                help="line-mode console instead of the full-screen display",
            )
            parser.add_argument(
                "--theme",
                choices=("auto", "light", "dark"),
                default="auto",
                help="palette for the full-screen display; auto reads "
                "COLORFGBG and falls back to dark. EC_TUI_THEME overrides it",
            )
            parser.add_argument(
                "--diagnostic-agent",
                choices=("auto", "codex", "claude", "off"),
                default="auto",
                help="read-only agent used by /diagnose (default: Codex, then Claude Code)",
            )
            parser.set_defaults(func=_cmd_lifecycle_console)
        else:
            parser.add_argument("--until", default="PRIMED")
            parser.add_argument("--go", action="store_true")
            parser.add_argument("--recover", action="store_true")
            parser.set_defaults(func=_cmd_lifecycle_run)

    rob = sub.add_parser(
        "robot", help="high-level robot lifecycle commands (start, damp, ...)"
    )
    rob.add_argument(
        "verb",
        choices=[
            "status",
            "damp",
            "zero-torque",
            "ready",
            "sit",
            "squat",
            "stand",
            "move",
            "stop",
            "wave-hand",
            "shake-hand",
        ],
    )
    rob.add_argument("--robot", default="g1", choices=["g1", "fake"])
    rob.add_argument("--network", default="")
    rob.add_argument("--dds-domain", type=int, default=0)
    rob.add_argument("--timeout", type=float, default=5.0)
    rob.add_argument("--height", type=float, default=-1.0)
    rob.add_argument("--vx", type=float, default=0.0)
    rob.add_argument("--vy", type=float, default=0.0)
    rob.add_argument("--vyaw", type=float, default=0.0)
    rob.add_argument("--continuous", action="store_true")
    rob.add_argument("--enable-writes", action="store_true")
    rob.add_argument("--confirm", default="")
    rob.set_defaults(func=_cmd_robot)

    robsh = sub.add_parser(
        "robot-shell",
        help="single-keypress operator console for the high-level robot layer",
    )
    robsh.add_argument("--robot", default="g1", choices=["g1", "fake"])
    robsh.add_argument("--network", default="")
    robsh.add_argument("--dds-domain", type=int, default=0)
    robsh.add_argument("--timeout", type=float, default=5.0)
    robsh.add_argument("--vx-step", type=float, default=0.2)
    robsh.add_argument("--vy-step", type=float, default=0.2)
    robsh.add_argument("--vyaw-step", type=float, default=0.3)
    robsh.add_argument("--enable-writes", action="store_true")
    robsh.add_argument("--confirm", default="")
    robsh.set_defaults(func=_cmd_robot_shell)

    return p


def _cmd_robot_shell(args) -> int:
    from embodied_control.robot import open_robot
    from embodied_control.console import KeyConsole
    from embodied_control.robot.shell import build_robot_bindings, robot_status

    gate_error = _unitree_write_gate_error(
        enable_writes=args.enable_writes,
        allow_non_realtime=False,
        confirm=args.confirm,
    )
    if gate_error is not None:
        print(f"FAIL: {gate_error}")
        return 2
    try:
        runtime = open_robot(
            args.robot,
            network_interface=args.network,
            writes_enabled=args.enable_writes,
            dds_domain=args.dds_domain,
            timeout_seconds=args.timeout,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2

    info = runtime.info()
    print(f"{info.robot_id} operator console  ({info.transport})")
    if not args.enable_writes:
        print("READ-ONLY: pass --enable-writes --confirm to command the robot")
    bindings = build_robot_bindings(
        runtime,
        vx_step=args.vx_step,
        vy_step=args.vy_step,
        vyaw_step=args.vyaw_step,
    )
    console = KeyConsole(bindings, status=lambda: robot_status(runtime))
    try:
        return console.run()
    except KeyboardInterrupt:
        return 0
    finally:
        runtime.close()
        print("\nconsole closed (robot damped)" if args.enable_writes else "")


def _cmd_robot(args) -> int:
    from embodied_control.robot import (
        Capability,
        GestureControl,
        PostureControl,
        RobotError,
        VelocityControl,
        open_robot,
    )

    mutating = args.verb not in {"status"}
    if mutating:
        gate_error = _unitree_write_gate_error(
            enable_writes=args.enable_writes,
            allow_non_realtime=False,
            confirm=args.confirm,
        )
        if gate_error is not None:
            print(f"FAIL: {gate_error}")
            return 2
        if not args.enable_writes:
            print(f"FAIL: {args.verb} changes robot state; pass --enable-writes")
            return 2
    try:
        runtime = open_robot(
            args.robot,
            network_interface=args.network,
            writes_enabled=args.enable_writes,
            dds_domain=args.dds_domain,
            timeout_seconds=args.timeout,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2
    try:
        if args.verb == "status":
            info = runtime.info()
            health = runtime.health()
            print(f"robot: {info.robot_id}")
            print(f"transport: {info.transport}")
            print(
                "capabilities: "
                + (", ".join(sorted(info.capabilities)) or "none")
            )
            print(f"reachable: {health.reachable}")
            print(f"mode: {health.mode}")
            if health.detail:
                print(f"detail: {health.detail}")
            return 0 if health.reachable else 1
        if args.verb == "damp":
            runtime.damp()
        elif args.verb == "zero-torque":
            runtime.zero_torque()
        elif args.verb == "ready":
            runtime.ready()
        elif args.verb in {"sit", "squat", "stand"}:
            if not isinstance(runtime, PostureControl):
                print(f"FAIL: {args.robot} has no posture control")
                return 2
            if args.verb == "sit":
                runtime.sit()
            elif args.verb == "squat":
                runtime.squat()
            else:
                runtime.stand(args.height if args.height >= 0 else None)
        elif args.verb in {"move", "stop"}:
            if not isinstance(runtime, VelocityControl):
                print(f"FAIL: {args.robot} has no velocity control")
                return 2
            if args.verb == "move":
                runtime.move(
                    args.vx, args.vy, args.vyaw, continuous=args.continuous
                )
            else:
                runtime.stop()
        elif args.verb in {"wave-hand", "shake-hand"}:
            if not isinstance(runtime, GestureControl):
                print(f"FAIL: {args.robot} has no gesture control")
                return 2
            if args.verb == "wave-hand":
                runtime.wave_hand()
            else:
                runtime.shake_hand()
        else:
            print(f"FAIL: unknown verb {args.verb}")
            return 2
    except RobotError as exc:
        print(f"FAIL: {exc}")
        return 1
    finally:
        # A one-shot verb leaves the robot in the mode it asked for. Damping on
        # exit here would stand the robot up and drop it in the same command;
        # only the interactive shell damps when it closes.
        runtime.close(damp=False)
    print(f"{args.verb}: ok (mode {runtime.mode()})")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
