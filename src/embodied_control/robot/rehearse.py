"""Rehearse a tracker on the plant, unattended, one motion at a time.

`ec lifecycle rehearse <bundle> <motions...>` is the whole preparation for a
hardware session in one command: for every motion it writes the canonical
job, spawns its own vendored and hoisted plant on a private DDS domain,
drives the console's session through build, prepare, arm, play, hoist and
damp exactly as an operator would, and leaves the evidence the hardware
gate reads (`episodes/*/lifecycle.json`), the plant's true state trajectory
and a video of the whole run from hoist to hand-back.

Each episode runs in a child process. DDS entity teardown can abort inside
the vendor SDK after the robot is already safe, and one aborted motion must
not take the rest of the sweep with it.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from embodied_control.robot.lifecycle_job import (
    LIFECYCLE_API_VERSION,
    LifecycleJob,
    apply_target,
    load_lifecycle_job,
)

DEFAULT_REFERENCE_ROOT = "assets/models/reference/bones"
DEFAULT_MODEL = "assets/latent_playkit/model/g1_29dof_rev_1_0.xml"
DEFAULT_PLANT = "examples/g1_plant.yaml"
NON_REALTIME_CONFIRM = "ENABLE_G1_LOWLEVEL_NON_REALTIME"
#: Seconds the plant and the episode get on top of the reference's own length.
PREPARATION_ALLOWANCE_SECONDS = 120.0
ARMED_PAUSE_SECONDS = 2.0
FINAL_HOLD_SECONDS = 2.0
HOISTED_SETTLE_SECONDS = 3.0
#: Cores per lane: control, writer, plant physics, and the rest for Python.
LANE_CORES = 4
#: Plant sensor-noise profiles, as `--noise-*` uniform half-ranges.
#: `training` is the plant default (SONIC's policy-group ranges); `measured`
#: is the G1's own LowState at rest on 2026-09-10 (hardware_noise_20260910),
#: rounded up; `off` is a clean sensor.
PLANT_NOISE = {
    "training": {},
    "measured": {"joint_pos": 0.0001, "joint_vel": 0.025, "base_ang_vel": 0.015, "imu_tilt_rad": 0.0015},
    "off": {"joint_pos": 0.0, "joint_vel": 0.0, "base_ang_vel": 0.0, "imu_tilt_rad": 0.0},
}
MAX_DDS_DOMAIN = 232


@dataclass
class RehearsePlan:
    bundle: str
    motions: list[str]
    output: Path
    reference_root: str = DEFAULT_REFERENCE_ROOT
    model: str = DEFAULT_MODEL
    plant_config: str = DEFAULT_PLANT
    seeds: int = 1
    lanes: int = 1
    dds_domain_base: int = 190
    template: dict = field(default_factory=dict)
    video: bool = True
    offline: bool = True
    plant_noise: str = "training"
    # Serve the plant's true pelvis position on this odometry topic (the G1
    # message layout), so the controller's odometry anchor path runs against
    # a perfect estimator. Empty: the controller falls back to its own leg
    # odometry, the sim stand-in for the vendor estimator.
    plant_odometry: str = ""


# ------------------------------------------------------------------ jobs


def _composed_hold_frames(reference_root: str, motion: str) -> int:
    manifest = Path(reference_root) / "reference_arrays_manifest.json"
    if not manifest.is_file():
        return 0
    entry = json.loads(manifest.read_text()).get("motions", {}).get(motion, {})
    return int(entry.get("hold_frames", 0))


def canonical_job(plan: RehearsePlan, motion: str, bundle_manifest) -> dict:
    """The hardware-shaped job for one motion; `--target sim` derives the rehearsal.

    Strict real-time, an operator-supplied interface, the bundle's default
    stance, no planner lookahead: the profile the hardware test card uses,
    so the rehearsal identity matches the run it vouches for.
    """
    command = bundle_manifest.command
    lookahead = command.horizon_steps * command.macro_frame_stride
    composed = _composed_hold_frames(plan.reference_root, motion) > lookahead
    # A one-tick-hold tracker wants its reply on the tick (lead 0); a
    # ten-step-hold tracker starves at lead 0 and rehearsed clean at 4.
    lead_ticks = min(4, max(0, int(command.hold_steps) - 1))
    root = plan.output / motion
    job = {
        "api_version": LIFECYCLE_API_VERSION,
        "bundle": str(Path(plan.bundle).resolve()),
        "trackers": {},
        "network": "",
        "dds_domain": 0,
        "request_slot": f"/ec_{Path(plan.bundle).name}_request",
        "response_slot": f"/ec_{Path(plan.bundle).name}_response",
        "connect_slots": True,
        "command_source": "oracle",
        "reference_root": str(Path(plan.reference_root).resolve()),
        "motion": motion,
        "start_frame": 0,
        "lead_ticks": lead_ticks,
        "fixed_initial_anchor": True,
        "start_pose": "default",
        "ramp_seconds": 3.0,
        "ticks": "auto",
        "blend_ticks": 250,
        "play_countdown_seconds": 3.0,
        "arm_timeout_seconds": 60.0,
        "stand_hold_seconds": 60.0 if composed else 0.0,
        "end_state": "vendor_damp",
        "sim_hoist": False,
        "require_vendor": True,
        "require_rehearsal": True,
        "artifacts_dir": str(root.resolve()),
        "mjcf": str(Path(plan.model).resolve()),
        "realtime": {
            "control_cpu": 2, "writer_cpu": 3, "control_priority": 80,
            "writer_priority": 90, "lock_memory": True, "policy_threads": 1,
        },
        "thresholds": {"settle_timeout_seconds": 15.0},
    }
    for key, value in plan.template.items():
        if key in ("bundle", "trackers", "reference_root", "motion", "artifacts_dir", "mjcf"):
            continue
        job[key] = value
    return LifecycleJob.model_validate(job).model_dump(mode="json")


def template_from(path: str | Path) -> dict:
    """Deployment settings to carry from an existing job into every motion."""
    job = load_lifecycle_job(path).model_dump(mode="json")
    keep = (
        "start_pose", "ramp_seconds", "blend_ticks", "play_countdown_seconds",
        "arm_timeout_seconds", "lead_ticks", "command_stale_ms", "state_absent_ms",
        "end_state", "damp_hands_back", "retake_precheck", "rehearsal_max_age_days",
        "endpoint_screening", "thresholds", "realtime", "planner", "slack_on_run",
        "fixed_anchor_max_displacement", "gain_scale",
    )
    return {key: job[key] for key in keep}


def episode_job(job_path: Path, *, seed: int, domain: int, cores: list[int]) -> LifecycleJob:
    """The sim job one episode actually runs: `--target sim` plus its own seed dir."""
    job = apply_target(load_lifecycle_job(job_path), "sim", dds_domain=domain)
    realtime = job.realtime.model_copy(update={
        "control_cpu": cores[0] if len(cores) > 1 else -1,
        "writer_cpu": cores[1] if len(cores) > 1 else -1,
    })
    return job.model_copy(update={
        "artifacts_dir": str(Path(job.artifacts_dir) / f"seed_{seed}"),
        "request_slot": f"/ec_rehearse_{domain}_request",
        "response_slot": f"/ec_rehearse_{domain}_response",
        "realtime": realtime,
    })


# --------------------------------------------------------------- episode


def _wait_plant_ready(process, log: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while "PLANT_READY" not in log.read_text():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError("plant did not become ready; see plant.log")
        time.sleep(0.1)


def _stop_plant(process, timeout: float = 15.0) -> None:
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def run_episode(
    job_path: str | Path,
    *,
    seed: int,
    domain: int,
    cores: list[int],
    model: str,
    plant_config: str,
    video: bool = True,
    offline: bool = True,
    plant_noise: str = "training",
    plant_odometry: str = "",
) -> dict:
    """One motion, one seed, one plant: the operator's keys, unattended.

    Returns the `result.json` payload and leaves the run directory with the
    resolved job, plant log/report/state trajectory, lifecycle evidence,
    episode telemetry and (when asked) the video.
    """
    from embodied_control.robot.build import BuildOptions, build_session
    from embodied_control.robot.lifecycle import LifecycleState as S

    if cores:
        os.sched_setaffinity(0, set(cores))
    job = episode_job(Path(job_path), seed=seed, domain=domain, cores=cores)
    run = Path(job.artifacts_dir)
    run.mkdir(parents=True, exist_ok=True)
    resolved = run / "job.yaml"
    resolved.write_text(yaml.safe_dump(job.model_dump(mode="json"), sort_keys=False))
    frames = _motion_length(job.reference_root, job.motion)
    budget = frames / 50.0 + PREPARATION_ALLOWANCE_SECONDS + job.stand_hold_seconds
    result = {
        "motion": job.motion, "bundle": Path(job.bundle).name, "seed": seed,
        "dds_domain": domain, "frames": frames, "started_at": time.time(),
        "passed": False, "failure_stage": "plant_start", "directory": str(run),
        "plant_noise": plant_noise,
        "plant_odometry": plant_odometry,
        "live_anchor": job.live_anchor,
        "anchor_position_source": job.anchor_position_source,
    }
    plant_log = run / "plant.log"
    with plant_log.open("w") as stream:
        plant = subprocess.Popen(
            [
                sys.executable, "-m", "embodied_control.cli", "lowlevel", "plant",
                str(plant_config), "--model", str(model), "--network", "lo",
                "--vendor", "--hoist", "--noise-seed", str(seed),
                "--dds-domain", str(domain),
                "--physics-cpu", str(cores[2] if len(cores) > 2 else -1),
                "--seconds", f"{budget:.0f}", "--report", str(run / "plant.json"),
                *plant_noise_argv(plant_noise),
                *(["--odometry-topic", plant_odometry] if plant_odometry else []),
            ],
            stdout=stream, stderr=subprocess.STDOUT,
        )
    session = vendor = None
    timeline: list[dict] = []
    try:
        _wait_plant_ready(plant, plant_log)
        options = BuildOptions(
            job=str(resolved), enable_writes=True, allow_non_realtime=True,
            confirm=NON_REALTIME_CONFIRM, offline=offline, planner_autostart=True,
            auto_ack=True,
        )
        _, session, vendor = build_session(options)
        session.note_sinks.append(lambda text: print(f"  -- {text}", flush=True))
        for stage, operation in (
            ("build", session.rebuild), ("prepare", session.auto), ("arm", session.arm),
        ):
            result["failure_stage"] = stage
            gate = operation()
            if stage == "build" and gate.ok:
                _watch_transitions(session.lifecycle, timeline)
            if not gate.ok:
                raise RuntimeError(gate.detail)
        time.sleep(ARMED_PAUSE_SECONDS)
        session.poll()
        if session.lifecycle.state is not S.ARMED:
            raise RuntimeError(f"armed pause ended at {session.lifecycle.state}")
        result["failure_stage"] = "play"
        gate = session.play()
        if not gate.ok:
            raise RuntimeError(gate.detail)
        result["failure_stage"] = "running"
        deadline = time.monotonic() + frames / 50.0 + 10.0
        while session.lifecycle.state is S.RUNNING and time.monotonic() < deadline:
            session.poll()
            time.sleep(0.01)
        expected = S.STAND_HOLD if job.stand_hold_seconds > 0 else S.HOLD
        if session.lifecycle.state is not expected:
            raise RuntimeError(f"{session.lifecycle.state}: {session.lifecycle.fault_reason}")
        result["control"] = session.tracker.stats()
        result["reference_complete"] = True
        result["failure_stage"] = "recovery"
        if expected is S.STAND_HOLD:
            time.sleep(FINAL_HOLD_SECONDS)
            session.poll()
            if session.lifecycle.state is not S.STAND_HOLD:
                raise RuntimeError("final stance hold failed")
            session.ack_hoisted()
            time.sleep(HOISTED_SETTLE_SECONDS)
            gate = session.damp()
        else:
            gate = session.recover("vendor_damp")
        if not gate.ok:
            raise RuntimeError(gate.detail)
        summary = session.lifecycle.summary()
        result["end_state"] = summary["state"]
        result["failed_transitions"] = summary["failed_transitions"]
        result["fault_reason"] = summary["fault_reason"]
        result["passed"] = (
            session.lifecycle.state is S.VENDOR_RESTORED
            and summary["failed_transitions"] == 0
            and not summary["fault_reason"]
            and int(result["control"].get("fault", 0)) == 0
        )
        if result["passed"]:
            result["failure_stage"] = None
    except Exception as error:
        result["error"] = str(error)
        if session is not None and session.lifecycle is not None:
            result["state_at_failure"] = str(session.lifecycle.state)
            result["fault_reason"] = session.lifecycle.fault_reason
            if session.tracker is not None:
                result["control"] = session.tracker.stats()
    finally:
        if session is not None:
            try:
                session.close()
            except Exception as error:
                result["cleanup_error"] = str(error)
                result["passed"] = False
        if vendor is not None:
            vendor.close(damp=False)
        _stop_plant(plant)
        if session is not None and session.episodes:
            result["episode"] = {
                key: session.episodes[-1].get(key)
                for key in ("ticks", "joint_mae_mean_rad", "joint_mae_p95_rad", "mpjpe_l_mm", "directory")
            }
        (run / "timeline.json").write_text(json.dumps(timeline, indent=2) + "\n")
        result["finished_at"] = time.time()
        if video:
            try:
                result["video"] = _render(run, job, timeline)
            except Exception as error:  # a missing video must not hide a passed run
                result["video_error"] = str(error)
        (run / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("RESULT " + json.dumps(result), flush=True)
    return result


def plant_noise_argv(profile: str) -> list[str]:
    if profile not in PLANT_NOISE:
        raise ValueError(f"plant noise profile must be one of {sorted(PLANT_NOISE)}")
    argv = []
    for key, value in PLANT_NOISE[profile].items():
        argv += [f"--noise-{key.replace('_', '-')}", str(value)]
    return argv


def _watch_transitions(lifecycle, timeline: list[dict]) -> None:
    """Wall-clock every transition, so the video can name the state."""
    downstream = lifecycle.on_transition

    def hook(transition):
        timeline.append({
            "wall": time.time(), "to_state": transition.to_state,
            "from_state": transition.from_state, "ok": transition.ok,
            "detail": transition.detail,
        })
        if downstream is not None:
            downstream(transition)

    lifecycle.on_transition = hook


def _motion_length(reference_root: str, motion: str) -> int:
    manifest = json.loads((Path(reference_root) / "reference_arrays_manifest.json").read_text())
    info = manifest["traj_info"]
    names = [entry[1] for entry in info["ordered_traj_list"]]
    index = names.index(motion)
    return int(info["end_index"][index] - info["start_index"][index])


def _render(run: Path, job: LifecycleJob, timeline: list[dict]) -> str:
    from embodied_control.lowlevel.plant_render import render_plant_states

    states = run / "plant.states.npz"
    if not states.is_file():
        raise FileNotFoundError(states)
    report = json.loads((run / "plant.json").read_text())
    output = run / "video.mp4"
    render_plant_states(
        states, output, job.mjcf,
        title=f"{Path(job.bundle).name}  |  {job.motion}  |  seed {Path(run).name.split('_')[-1]}",
        timeline=timeline, started_at=report.get("started_at"),
    )
    return str(output)


# ----------------------------------------------------------------- sweep


def _lanes(count: int) -> list[list[int]]:
    """Disjoint core blocks per lane, from what this process may run on."""
    available = sorted(os.sched_getaffinity(0))
    if len(available) < LANE_CORES:
        return [[] for _ in range(count)]
    lanes = []
    for lane in range(count):
        block = available[lane * LANE_CORES:(lane + 1) * LANE_CORES]
        lanes.append(block if len(block) == LANE_CORES else [])
    return lanes


def _worker_argv(plan: RehearsePlan, job_path: Path, seed: int, domain: int, cores: list[int]) -> list[str]:
    argv = [
        sys.executable, "-m", "embodied_control.cli", "lifecycle", "rehearse-episode",
        str(job_path), "--seed", str(seed), "--dds-domain", str(domain),
        "--model", str(plan.model), "--plant", str(plan.plant_config),
    ]
    if cores:
        argv += ["--cores", ",".join(map(str, cores))]
    if not plan.video:
        argv.append("--no-video")
    if not plan.offline:
        argv.append("--fetch")
    argv += ["--plant-noise", plan.plant_noise]
    if plan.plant_odometry:
        argv += ["--plant-odometry", plan.plant_odometry]
    return argv


def sweep(plan: RehearsePlan, *, note=print) -> dict:
    """Every motion times every seed; finished episodes are kept, not rerun."""
    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.reference import ReferenceArrays
    from embodied_control.lowlevel.reference_catalog import reference_compatibility
    from embodied_control.models import ensure_model

    bundle = PolicyBundle.load(ensure_model(plan.bundle, offline=plan.offline))
    reference_compatibility(bundle, ReferenceArrays(plan.reference_root))
    plan.output.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[str, Path]] = []
    for motion in plan.motions:
        directory = plan.output / motion
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "job.yaml"
        if not path.exists():
            path.write_text(yaml.safe_dump(canonical_job(plan, motion, bundle.manifest), sort_keys=False))
        jobs.append((motion, path))
    pending = [
        (motion, path, seed)
        for motion, path in jobs
        for seed in range(plan.seeds)
        if not (path.parent / "sim" / f"seed_{seed}" / "result.json").exists()
    ]
    # CycloneDDS refuses domain ids above 232, and every episode takes one.
    if pending and plan.dds_domain_base + len(pending) - 1 > MAX_DDS_DOMAIN:
        raise ValueError(
            f"{len(pending)} episodes from domain {plan.dds_domain_base} would pass "
            f"{MAX_DDS_DOMAIN}; lower --dds-domain-base"
        )
    lanes = _lanes(max(1, plan.lanes))
    note(f"{len(pending)} episode(s) to run on {len(lanes)} lane(s); output {plan.output}")

    def run_one(item):
        index, (motion, path, seed) = item
        lane = index % len(lanes)
        domain = plan.dds_domain_base + index
        run = path.parent / "sim" / f"seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        frames = _motion_length(plan.reference_root, motion)
        timeout = frames / 50.0 + PREPARATION_ALLOWANCE_SECONDS + 120.0
        with (run / "console.log").open("w") as log:
            process = subprocess.Popen(
                _worker_argv(plan, path, seed, domain, lanes[lane]),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        result_path = run / "result.json"
        if not result_path.exists():
            result_path.write_text(json.dumps({
                "motion": motion, "seed": seed, "passed": False, "failure_stage": "worker",
                "error": "worker failed or timed out; inspect console.log",
                "directory": str(run),
            }, indent=2) + "\n")
        result = json.loads(result_path.read_text())
        note(
            f"{motion} seed {seed}: {'PASS' if result.get('passed') else 'FAIL'}"
            + (f" ({result['error']})" if result.get("error") else "")
        )
        return result

    with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
        list(pool.map(run_one, enumerate(pending)))
    return summarize(plan.output)


def summarize(output: Path) -> dict:
    """`summary.json` and `REPORT.md` from every `result.json` under `output`."""
    output = Path(output).resolve()
    results = []
    for path in sorted(output.glob("*/sim/seed_*/result.json")):
        payload = json.loads(path.read_text())
        payload.setdefault("motion", path.parents[2].name)
        results.append(payload)
    by_motion: dict[str, list[dict]] = {}
    for result in results:
        by_motion.setdefault(result["motion"], []).append(result)
    rows = []
    for motion, runs in sorted(by_motion.items()):
        passed = sum(1 for run in runs if run.get("passed"))
        rows.append({
            "motion": motion, "seeds": len(runs), "passed": passed,
            "all_passed": passed == len(runs),
            "job": str(output / motion / "job.yaml"),
            "failures": [run.get("error", "") for run in runs if not run.get("passed")],
            "joint_mae_mean_rad": [
                run.get("episode", {}).get("joint_mae_mean_rad") for run in runs
            ],
            "videos": [run.get("video", "") for run in runs],
        })
    summary = {
        "output": str(output), "written_at": time.time(),
        "motions": len(rows), "motions_all_passed": sum(1 for row in rows if row["all_passed"]),
        "episodes": len(results), "episodes_passed": sum(1 for r in results if r.get("passed")),
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "REPORT.md").write_text(_report_markdown(summary, results))
    return summary


def _relative(path: str, root: str) -> str:
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path


def _report_markdown(summary: dict, results: list[dict]) -> str:
    bundle = next((r.get("bundle", "") for r in results if r.get("bundle")), "")
    lines = [
        f"# Rehearsal sweep: {bundle}",
        "",
        f"Written {time.strftime('%Y-%m-%d %H:%M', time.localtime(summary['written_at']))}. "
        f"{summary['episodes_passed']}/{summary['episodes']} episodes clean; "
        f"{summary['motions_all_passed']}/{summary['motions']} motions clean on every seed.",
        "",
        "A clean episode reached `VENDOR_RESTORED` with no refused transition and no",
        "runtime fault, hoisted preparation through playback and hoisted hand-back.",
        "Joint MAE is joint-space tracking against the reference; the video is the",
        "plant's true root. Neither is a hardware result.",
        "",
        "| Motion | Clean | Joint MAE (rad) | Failures | Video |",
        "|---|---:|---|---|---|",
    ]
    for row in summary["rows"]:
        mae = ", ".join(f"{v:.3f}" for v in row["joint_mae_mean_rad"] if v is not None) or "-"
        videos = ", ".join(
            f"[seed {i}]({_relative(v, summary['output'])})"
            for i, v in enumerate(row["videos"]) if v
        ) or "-"
        failures = "; ".join(f for f in row["failures"] if f) or "-"
        lines.append(f"| `{row['motion']}` | {row['passed']}/{row['seeds']} | {mae} | {failures} | {videos} |")
    lines += [
        "",
        "Each motion directory holds `job.yaml`, the canonical hardware-shaped job.",
        "Rehearse it again with `--target sim`; run it on the robot with",
        "`--target hardware --network <NIC>`, which finds this evidence beside it.",
        "",
    ]
    return "\n".join(lines)


def matrix_report(root: str | Path) -> dict:
    """Motion x tracker table across the per-bundle sweeps under `root`.

    Each `<root>/<bundle>/summary.json` is one tracker; a cell is clean seeds
    over seeds, and a motion row links every video, so one page answers
    "which tracker for which motion" before anybody opens a run directory.
    """
    root = Path(root).resolve()
    sweeps = {}
    for path in sorted(root.glob("*/summary.json")):
        sweeps[path.parent.name] = {row["motion"]: row for row in json.loads(path.read_text())["rows"]}
    if not sweeps:
        raise FileNotFoundError(f"no */summary.json under {root}")
    motions = sorted({motion for rows in sweeps.values() for motion in rows})
    bundles = sorted(sweeps, key=lambda name: -sum(r["passed"] for r in sweeps[name].values()))
    lines = [
        f"# Rehearsal matrix: {len(bundles)} trackers x {len(motions)} motions",
        "",
        f"Written {time.strftime('%Y-%m-%d %H:%M')}. A cell is clean episodes / seeds "
        "(`VENDOR_RESTORED`, no refused transition, no runtime fault); a dash is not run. "
        "Links open the seed-0 video. Simulation only; not hardware evidence.",
        "",
        "| Tracker | Clean motions |",
        "|---|---:|",
    ]
    for bundle in bundles:
        rows = sweeps[bundle]
        lines.append(f"| `{bundle}` | {sum(r['all_passed'] for r in rows.values())}/{len(rows)} |")
    lines += ["", "| Motion | " + " | ".join(bundles) + " |", "|---|" + "---:|" * len(bundles)]
    cells = {}
    for motion in motions:
        row = []
        for bundle in bundles:
            entry = sweeps[bundle].get(motion)
            if entry is None:
                row.append("-")
                continue
            video = next((v for v in entry["videos"] if v), "")
            mark = f"{entry['passed']}/{entry['seeds']}"
            row.append(f"[{mark}]({_relative(video, str(root))})" if video else mark)
            cells[(motion, bundle)] = entry["all_passed"]
        lines.append(f"| `{motion}` | " + " | ".join(row) + " |")
    payload = {
        "root": str(root), "bundles": bundles, "motions": motions,
        "clean": {b: sum(r["all_passed"] for r in sweeps[b].values()) for b in bundles},
        "clean_everywhere": [m for m in motions if all(cells.get((m, b)) for b in bundles)],
        "clean_nowhere": [m for m in motions if not any(cells.get((m, b)) for b in bundles)],
    }
    lines += [
        "",
        f"Clean on every tracker ({len(payload['clean_everywhere'])}): "
        + ", ".join(f"`{m}`" for m in payload["clean_everywhere"]),
        "",
        f"Clean on no tracker ({len(payload['clean_nowhere'])}): "
        + ", ".join(f"`{m}`" for m in payload["clean_nowhere"]),
        "",
    ]
    (root / "REPORT.md").write_text("\n".join(lines))
    (root / "matrix.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload
