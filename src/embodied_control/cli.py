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
        check("mujoco (sim env)", False, "run in the `sim` pixi env: pixi run -e sim ...")

    try:
        import imageio  # noqa: F401
        check("imageio (video, sim env)", True, imageio.__version__)
    except Exception:  # noqa: BLE001
        check("imageio (video, sim env)", False,
              "needed for rollout.record_video; run in the `sim` pixi env")

    try:
        import mujoco  # noqa: F401

        renderer = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>"), height=64, width=64)
        renderer.close()
        check("mujoco offscreen renderer", True, f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}")
    except ImportError:
        check("mujoco offscreen renderer", False, "requires mujoco (sim env)")
    except Exception as exc:  # noqa: BLE001
        check("mujoco offscreen renderer", False,
              f"{exc} (no EGL/OSMesa on this host? rollout.record_video will be disabled)")

    import shutil
    docker = shutil.which("docker")
    check("docker (for containerized policy)", docker is not None,
          docker or "optional; runtime.type=local works without it")

    # The blank policy service is stdlib-only:
    try:
        from embodied_control.transport.server import PolicyServer  # noqa: F401
        from embodied_control.policies.base import make_policy  # noqa: F401
        check("policy service (stdlib)", True)
    except Exception as exc:  # noqa: BLE001
        check("policy service (stdlib)", False, str(exc))

    print("  note: MuJoCo eval requires the `sim` env; the policy service runs in Docker "
          "(runtime.type=docker) or as a local subprocess (runtime.type=local).")
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
    print(f"episodes    : completed={m.num_episodes_completed} failed={m.num_episodes_failed}")
    print(f"success_rate: {m.success_rate:.3f}")
    print(f"mean_return : {m.mean_return:.4f}")
    print(f"policy reqs : {m.num_policy_requests}  "
          f"latency p50={m.policy_latency_ms.p50:.2f}ms p95={m.policy_latency_ms.p95:.2f}ms")
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
            print(f"{run_dir.name}  status={manifest.get('status')}  "
                  f"success_rate={metrics.get('success_rate')}  "
                  f"episodes={metrics.get('num_episodes_completed')}")
        except Exception as exc:  # noqa: BLE001
            print(f"{run_dir.name}  <unreadable: {exc}>")
    return 0


def _cmd_policy_serve(args) -> int:
    from embodied_control.policies.debug_server import main as serve_main

    argv = [
        "--type", args.type,
        "--action-dim", str(args.action_dim),
        "--host", args.host,
        "--port", str(args.port),
        "--seed", str(args.seed),
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ec", description="Embodied-Control eval orchestrator")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="check host + dependencies").set_defaults(func=_cmd_doctor)

    ev = sub.add_parser("eval", help="evaluation commands")
    evs = ev.add_subparsers(dest="eval_command", required=True)
    run = evs.add_parser("run", help="run an eval job")
    run.add_argument("job")
    run.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
                      help="override outputs.log_level from the job file")
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
    serve.add_argument("--type", default="zero", choices=["zero", "random", "image_stats"])
    serve.add_argument("--action-dim", type=int, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8756)
    serve.add_argument("--seed", type=int, default=0)
    serve.set_defaults(func=_cmd_policy_serve)
    ping = pols.add_parser("ping", help="health + describe an endpoint")
    ping.add_argument("--endpoint", required=True, help="host:port")
    ping.add_argument("--timeout", type=float, default=10.0)
    ping.set_defaults(func=_cmd_policy_ping)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
