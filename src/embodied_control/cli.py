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

    run_dir, result = run_lowlevel_job(args.job, device=args.device)
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
    try:
        runtime.start(args.ticks, paced=True)
        runtime.wait()
    except KeyboardInterrupt:
        runtime.stop()
        runtime.wait()
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
        StdioChunkService,
    )

    command = list(args.service_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("FAIL: planner-worker needs a service command after --")
        return 2
    service = StdioChunkService(command)
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
        horizon=args.horizon,
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


def _cmd_lowlevel_unitree(args) -> int:
    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.native_core import NativeUnitreeLoop

    if args.enable_writes and args.confirm != "ENABLE_G1_LOWLEVEL":
        print("FAIL: --enable-writes requires --confirm ENABLE_G1_LOWLEVEL")
        return 2
    if args.enable_writes and args.allow_non_realtime:
        print("FAIL: hardware writes require strict real-time setup")
        return 2
    runtime = NativeUnitreeLoop(
        PolicyBundle.load(args.bundle),
        args.network,
        response_slot=args.response_slot,
        request_slot=args.request_slot,
        writes_enabled=args.enable_writes,
        create_slots=not args.connect_slots,
        lead_ticks=args.lead_ticks,
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

    runtime.start(args.ticks, paced=True)
    armed = not args.enable_writes
    failed = False
    interrupted = False
    command_deadline = time.monotonic() + args.command_timeout
    try:
        while runtime.running:
            if args.enable_writes and not armed:
                if int(runtime.stats()["control_ticks"]) > 0:
                    runtime.arm_control()
                    armed = True
                    print("G1 native control armed")
                elif time.monotonic() >= command_deadline:
                    raise RuntimeError(
                        "no valid planner command arrived before arm timeout"
                    )
            time.sleep(0.02)
    except KeyboardInterrupt:
        interrupted = True
        runtime.stop()
    except RuntimeError as exc:
        failed = True
        runtime.stop()
        print(f"FAIL: {exc}")
    finally:
        runtime.wait()
        runtime.force_damp()
    report = {"control": runtime.stats(), "writer": runtime.writer_stats()}
    print(json.dumps(report, indent=2))
    return (
        0 if not failed and not interrupted and report["control"]["fault"] == 0 else 1
    )


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
    lnoracle.add_argument("--horizon", type=int, default=30)
    lnoracle.add_argument("--create-slots", action="store_true")
    lnoracle.add_argument("--report", default="")
    lnoracle.set_defaults(func=_cmd_lowlevel_oracle_worker)

    lcert = lows.add_parser(
        "certify-report", help="grade a native runtime report against the timing SLOs"
    )
    lcert.add_argument("report")
    lcert.add_argument("--output", default="")
    lcert.set_defaults(func=_cmd_lowlevel_certify_report)
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
    lunitree.set_defaults(func=_cmd_lowlevel_unitree)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
