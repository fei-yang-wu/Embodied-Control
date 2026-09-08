"""Repeatable multi-goal evaluation for the native latent-plan MuJoCo path."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import yaml

from embodied_control.logging import EcLogger, LogConfig
from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.native_core import NativeMujocoLoop
from embodied_control.lowlevel.publishers.native_pull import (
    NativeLatentPlanWorker,
    StdioChunkService,
)
from embodied_control.lowlevel.telemetry import TelemetryRecorder


class NativeMotionEvalError(RuntimeError):
    def __init__(self, message: str, run_dir: Path):
        super().__init__(message)
        self.run_dir = run_dir


def _timestamp() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:96] or "goal"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def trajectory_metrics(
    telemetry: dict,
    *,
    control_hz: int,
    min_physics_height: float | None,
) -> tuple[dict, np.ndarray]:
    poses = np.asarray(telemetry.get("anchor_pose_log", []), dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 7:
        poses = np.empty((0, 7), dtype=np.float64)
    poses = poses[np.isfinite(poses).all(axis=1)]
    min_height = (
        float(min_physics_height)
        if min_physics_height is not None and np.isfinite(min_physics_height)
        else None
    )
    fell = min_height is not None and min_height < 0.4
    if poses.size == 0:
        return {
            "samples": 0,
            "duration_s": 0.0,
            "net_displacement_m": None,
            "path_length_m": None,
            "mean_speed_mps": None,
            "straightness": None,
            "yaw_change_deg": None,
            "yaw_travel_deg": None,
            "min_height_m": min_height,
            "mean_height_m": None,
            "final_height_m": None,
            "fell_below_0_4": fell,
        }, np.empty((0, 2), dtype=np.float64)

    position = poses[:, :3]
    xy = position[:, :2]
    segments = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    path_length = float(np.sum(segments))
    displacement = float(np.linalg.norm(xy[-1] - xy[0]))
    duration = float(len(poses) / control_hz)
    quaternion = poses[:, 3:7]
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = quaternion / np.maximum(norms, 1.0e-12)
    x, y, z, w = quaternion.T
    yaw = np.unwrap(
        np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )
    heights = np.asarray(telemetry.get("base_heights", []), dtype=np.float64)
    heights = heights[np.isfinite(heights)]
    return {
        "samples": int(len(poses)),
        "duration_s": duration,
        "start_xy_m": [float(value) for value in xy[0]],
        "final_xy_m": [float(value) for value in xy[-1]],
        "net_displacement_m": displacement,
        "path_length_m": path_length,
        "mean_speed_mps": path_length / duration if duration > 0 else None,
        "straightness": displacement / path_length if path_length > 1.0e-9 else 0.0,
        "yaw_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "yaw_travel_deg": float(np.degrees(np.sum(np.abs(np.diff(yaw))))),
        "min_height_m": min_height,
        "mean_height_m": float(np.mean(heights)) if heights.size else None,
        "final_height_m": float(heights[-1]) if heights.size else None,
        "fell_below_0_4": fell,
    }, xy


def _latency_summary(values: Sequence[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _planner_metrics(request_ms: Sequence[float], head_ms: Sequence[float]) -> dict:
    return {
        "request_ms": _latency_summary(request_ms),
        "head_ms": _latency_summary(head_ms),
    }


def aggregate_motion_metrics(episodes: Sequence[dict]) -> dict:
    fields = (
        "net_displacement_m",
        "path_length_m",
        "mean_speed_mps",
        "straightness",
        "yaw_change_deg",
        "yaw_travel_deg",
        "min_height_m",
    )
    grouped: dict[str, list[dict]] = {}
    for episode in episodes:
        grouped.setdefault(str(episode["goal"]), []).append(episode)
    by_goal = {}
    for goal, rows in grouped.items():
        trajectory = {}
        for field in fields:
            numeric = np.asarray(
                [
                    row["trajectory"][field]
                    for row in rows
                    if row["trajectory"].get(field) is not None
                ],
                dtype=float,
            )
            trajectory[field] = {
                "mean": float(np.mean(numeric)) if numeric.size else None,
                "std": float(np.std(numeric)) if numeric.size else None,
                "min": float(np.min(numeric)) if numeric.size else None,
                "max": float(np.max(numeric)) if numeric.size else None,
            }
        successes = sum(bool(row.get("success")) for row in rows)
        planner_latency = [
            row["planner"]["request_ms"]["mean"]
            for row in rows
            if row["planner"]["request_ms"]["mean"] is not None
        ]
        by_goal[goal] = {
            "episodes": len(rows),
            "successes": successes,
            "success_rate": successes / len(rows),
            "planner_request_ms_mean": (
                float(np.mean(planner_latency)) if planner_latency else None
            ),
            "trajectory": trajectory,
        }
    successes = sum(bool(row.get("success")) for row in episodes)
    return {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "by_goal": by_goal,
    }


def _write_summary_csv(path: Path, episodes: Sequence[dict]) -> None:
    columns = (
        "episode_id",
        "goal",
        "repeat",
        "success",
        "fault",
        "deadline_misses",
        "planner_responses",
        "planner_request_ms_mean",
        "planner_head_ms_mean",
        "net_displacement_m",
        "path_length_m",
        "mean_speed_mps",
        "straightness",
        "yaw_change_deg",
        "yaw_travel_deg",
        "min_height_m",
        "fell_below_0_4",
        "error",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for episode in episodes:
            trajectory = episode["trajectory"]
            control = episode["control"]
            writer.writerow(
                {
                    "episode_id": episode["episode_id"],
                    "goal": episode["goal"],
                    "repeat": episode["repeat"],
                    "success": episode["success"],
                    "fault": control.get("fault"),
                    "deadline_misses": control.get("deadline_misses"),
                    "planner_responses": control.get("planner_responses"),
                    "planner_request_ms_mean": episode["planner"]["request_ms"][
                        "mean"
                    ],
                    "planner_head_ms_mean": episode["planner"]["head_ms"]["mean"],
                    **{
                        key: trajectory.get(key)
                        for key in columns
                        if key in trajectory
                    },
                    "error": episode.get("error"),
                }
            )


def _write_trajectory_svg(
    path: Path,
    trajectories: Sequence[tuple[dict, np.ndarray]],
) -> None:
    plot_width, height, margin = 620, 700, 70
    goals = list(dict.fromkeys(str(episode["goal"]) for episode, _ in trajectories))
    legend_columns = max(1, (len(goals) + 19) // 20)
    legend_column_width = 300
    width = plot_width + 2 * margin + legend_columns * legend_column_width
    plot_height = height - 2 * margin
    available = [xy for _, xy in trajectories if len(xy)]
    if available:
        all_xy = np.concatenate(available, axis=0)
        minimum = np.min(all_xy, axis=0)
        maximum = np.max(all_xy, axis=0)
    else:
        minimum = np.array([-1.0, -1.0])
        maximum = np.array([1.0, 1.0])
    center = (minimum + maximum) / 2.0
    span = np.maximum(maximum - minimum, 1.0)
    scale = min(plot_width / span[0], plot_height / span[1]) * 0.9
    colors = (
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#9467bd",
        "#ff7f0e",
        "#17becf",
    )
    color_by_goal = {
        goal: colors[index % len(colors)] for index, goal in enumerate(goals)
    }

    def project(xy: np.ndarray) -> np.ndarray:
        result = np.empty_like(xy, dtype=float)
        result[:, 0] = margin + plot_width / 2.0 + (xy[:, 0] - center[0]) * scale
        result[:, 1] = margin + plot_height / 2.0 - (xy[:, 1] - center[1]) * scale
        return result

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
        '<text x="70" y="34" font-family="sans-serif" font-size="22" font-weight="600">Native GR00T motion trajectories</text>',
        f'<rect x="{margin}" y="{margin}" width="{plot_width}" height="{plot_height}" fill="white" stroke="#c8c8d0"/>',
    ]
    origin = project(np.asarray([[0.0, 0.0]]))[0]
    if margin <= origin[0] <= margin + plot_width:
        lines.append(
            f'<line x1="{origin[0]:.2f}" y1="{margin}" x2="{origin[0]:.2f}" y2="{margin + plot_height}" stroke="#e3e3e8"/>'
        )
    if margin <= origin[1] <= margin + plot_height:
        lines.append(
            f'<line x1="{margin}" y1="{origin[1]:.2f}" x2="{margin + plot_width}" y2="{origin[1]:.2f}" stroke="#e3e3e8"/>'
        )
    for episode, xy in trajectories:
        if len(xy) == 0:
            continue
        sampled = xy[:: max(1, len(xy) // 1000)]
        points = project(sampled)
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        color = color_by_goal[str(episode["goal"])]
        lines.append(
            f'<polyline points="{encoded}" fill="none" stroke="{color}" stroke-width="2.2" stroke-opacity="0.72"/>'
        )
        lines.append(
            f'<circle cx="{points[-1, 0]:.2f}" cy="{points[-1, 1]:.2f}" r="3.5" fill="{color}"/>'
        )
    legend_x = margin + plot_width + 28
    lines.append(
        f'<text x="{legend_x}" y="{margin + 5}" font-family="sans-serif" font-size="14" font-weight="600">Goals</text>'
    )
    for index, goal in enumerate(goals):
        column, row = divmod(index, 20)
        x_pos = legend_x + column * legend_column_width
        y_pos = margin + 34 + row * 28
        color = color_by_goal[goal]
        label = html.escape(goal)
        lines.extend(
            [
                f'<line x1="{x_pos}" y1="{y_pos}" x2="{x_pos + 24}" y2="{y_pos}" stroke="{color}" stroke-width="4"/>',
                f'<text x="{x_pos + 32}" y="{y_pos + 4}" font-family="monospace" font-size="11">{label}</text>',
            ]
        )
    lines.extend(
        [
            f'<text x="{margin + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">world X (m)</text>',
            f'<text x="18" y="{margin + plot_height / 2}" transform="rotate(-90 18 {margin + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">world Y (m)</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _job_payload(
    *,
    bundle: str | Path,
    model: str | Path,
    goals: Sequence[str],
    repeats: int,
    ticks: int,
    output_root: str | Path,
    service_command: Sequence[str],
    plan_slots: int,
    lead_ticks: int,
    hold_steps: int | None,
    command_stale_ms: float,
    policy_threads: int,
    telemetry_hz: float,
) -> dict:
    return {
        "api_version": "ec.native-motion-eval/v1",
        "bundle": str(bundle),
        "model": str(model),
        "goals": list(goals),
        "repeats": repeats,
        "ticks": ticks,
        "output_root": str(output_root),
        "planner": {
            "service_command": list(service_command),
            "plan_slots": plan_slots,
            "lead_ticks": lead_ticks,
            "hold_steps": hold_steps,
            "command_stale_ms": command_stale_ms,
        },
        "policy_threads": policy_threads,
        "telemetry_hz": telemetry_hz,
    }


def _status(
    *,
    state: str,
    succeeded: bool,
    started_at: str,
    expected: int,
    completed: int,
    error: str | None,
) -> dict:
    return {
        "state": state,
        "succeeded": succeeded,
        "started_at": started_at,
        "completed_at": _timestamp() if state != "running" else None,
        "episodes_expected": expected,
        "episodes_completed": completed,
        "error": error,
    }


def _error_text(error: BaseException | None) -> str | None:
    return None if error is None else f"{type(error).__name__}: {error}"


def run_native_motion_evaluation(
    bundle: str | Path,
    model: str | Path,
    goals: Sequence[str],
    service_command: Sequence[str],
    *,
    repeats: int = 1,
    ticks: int = 500,
    output_root: str | Path = "runs",
    plan_slots: int = 3,
    lead_ticks: int = 4,
    hold_steps: int | None = None,
    command_stale_ms: float = 500.0,
    policy_threads: int = 4,
    telemetry_hz: float = 0.0,
) -> tuple[Path, dict]:
    started_at = _timestamp()
    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f_native_motion_eval")
    run_dir = Path(output_root).resolve() / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "episodes").mkdir()
    (run_dir / "episodes.jsonl").write_text("")
    expected = len(goals) * repeats
    common = {
        "goals": goals,
        "repeats": repeats,
        "ticks": ticks,
        "service_command": service_command,
        "plan_slots": plan_slots,
        "lead_ticks": lead_ticks,
        "hold_steps": hold_steps,
        "command_stale_ms": command_stale_ms,
        "policy_threads": policy_threads,
        "telemetry_hz": telemetry_hz,
    }
    (run_dir / "job.yaml").write_text(
        yaml.safe_dump(
            _job_payload(bundle=bundle, model=model, output_root=output_root, **common),
            sort_keys=False,
        )
    )
    (run_dir / "resolved_job.yaml").write_text(
        yaml.safe_dump(
            _job_payload(
                bundle=Path(bundle).resolve(),
                model=Path(model).resolve(),
                output_root=Path(output_root).resolve(),
                **common,
            ),
            sort_keys=False,
        )
    )
    _write_json(run_dir / "metrics.json", aggregate_motion_metrics([]))
    _write_summary_csv(run_dir / "summary.csv", [])
    _write_trajectory_svg(run_dir / "trajectories.svg", [])
    _write_json(
        run_dir / "validation.json",
        {"valid": False, "checks": [], "error": "validation not completed"},
    )
    _write_json(
        run_dir / "status.json",
        _status(
            state="running",
            succeeded=False,
            started_at=started_at,
            expected=expected,
            completed=0,
            error=None,
        ),
    )
    _write_json(
        run_dir / "manifest.json",
        {
            "api_version": "ec.native-motion-eval/run-v1",
            "bundle": str(Path(bundle).resolve()),
            "model": str(Path(model).resolve()),
            "planner_ready": None,
        },
    )

    logger = EcLogger.create(
        LogConfig(level="INFO", console=False, log_dir=str(run_dir / "logs")),
        run_id,
    )
    episodes: list[dict] = []
    trajectories: list[tuple[dict, np.ndarray]] = []
    service: StdioChunkService | None = None

    def save_progress() -> None:
        _write_json(run_dir / "metrics.json", aggregate_motion_metrics(episodes))
        _write_summary_csv(run_dir / "summary.csv", episodes)
        _write_trajectory_svg(run_dir / "trajectories.svg", trajectories)
        _write_json(
            run_dir / "status.json",
            _status(
                state="running",
                succeeded=False,
                started_at=started_at,
                expected=expected,
                completed=len(episodes),
                error=None,
            ),
        )

    try:
        checks = [
            {"name": "goals", "passed": bool(goals), "detail": f"{len(goals)} goals"},
            {
                "name": "goal_values",
                "passed": all(isinstance(goal, str) and bool(goal.strip()) for goal in goals),
                "detail": "goals must be non-empty strings",
            },
            {
                "name": "unique_goals",
                "passed": len(set(goals)) == len(goals),
                "detail": "use repeats instead of duplicate goals",
            },
            {"name": "repeats", "passed": repeats > 0, "detail": str(repeats)},
            {"name": "ticks", "passed": ticks > 0, "detail": str(ticks)},
            {"name": "plan_slots", "passed": plan_slots > 0, "detail": str(plan_slots)},
            {"name": "lead_ticks", "passed": lead_ticks >= 0, "detail": str(lead_ticks)},
            {
                "name": "hold_steps",
                "passed": hold_steps is None or hold_steps > 0,
                "detail": str(hold_steps),
            },
            {
                "name": "command_stale_ms",
                "passed": command_stale_ms > 0,
                "detail": str(command_stale_ms),
            },
            {
                "name": "policy_threads",
                "passed": policy_threads > 0,
                "detail": str(policy_threads),
            },
            {
                "name": "telemetry_hz",
                "passed": telemetry_hz >= 0,
                "detail": str(telemetry_hz),
            },
            {
                "name": "service_command",
                "passed": bool(service_command),
                "detail": " ".join(service_command),
            },
            {
                "name": "model",
                "passed": Path(model).is_file(),
                "detail": str(Path(model).resolve()),
            },
        ]
        try:
            bundle_object = PolicyBundle.load(bundle)
        except Exception as exc:
            checks.append(
                {"name": "bundle", "passed": False, "detail": _error_text(exc)}
            )
            _write_json(
                run_dir / "validation.json",
                {"valid": False, "checks": checks, "error": _error_text(exc)},
            )
            raise
        command = bundle_object.manifest.command
        checks.extend(
            [
                {
                    "name": "bundle",
                    "passed": True,
                    "detail": str(bundle_object.root),
                },
                {
                    "name": "latent_interface",
                    "passed": bundle_object.manifest.interface == "latent",
                    "detail": bundle_object.manifest.interface,
                },
                {
                    "name": "latent_width",
                    "passed": bool(command.z_dim),
                    "detail": str(command.z_dim),
                },
            ]
        )
        validation = {
            "valid": all(check["passed"] for check in checks),
            "checks": checks,
            "error": None,
        }
        _write_json(run_dir / "validation.json", validation)
        if not validation["valid"]:
            failed = [check["name"] for check in checks if not check["passed"]]
            raise ValueError(f"native motion eval validation failed: {failed}")

        bundle_manifest_path = bundle_object.root / "manifest.json"
        manifest = {
            "api_version": "ec.native-motion-eval/run-v1",
            "bundle": str(bundle_object.root),
            "bundle_manifest_sha256": _sha256(bundle_manifest_path),
            "bundle_manifest": json.loads(bundle_manifest_path.read_text()),
            "model": str(Path(model).resolve()),
            "model_sha256": _sha256(Path(model).resolve()),
            "planner_ready": None,
        }
        service = StdioChunkService(
            service_command,
            action_width=int(command.z_dim),
            window_frames=plan_slots,
            goal=str(goals[0]),
        )
        manifest["planner_ready"] = service.ready
        _write_json(run_dir / "manifest.json", manifest)
        control_hz = int(bundle_object.manifest.rates.control_hz)
        effective_hold = int(command.hold_steps if hold_steps is None else hold_steps)
        episode_index = 0

        for repeat in range(repeats):
            for goal in goals:
                episode_id = f"{episode_index:04d}_{_slug(goal)}_r{repeat:02d}"
                episode_dir = run_dir / "episodes" / episode_id
                episode_dir.mkdir()
                episode_started = _timestamp()
                service.goal = str(goal)
                head_start = len(service.head_ms)
                request_slot = f"/ec_motion_eval_{os.getpid()}_{episode_index}_request"
                response_slot = f"/ec_motion_eval_{os.getpid()}_{episode_index}_response"
                worker: NativeLatentPlanWorker | None = None
                runtime: NativeMujocoLoop | None = None
                recorder: TelemetryRecorder | None = None
                failure: BaseException | None = None
                telemetry: dict = {}
                control: dict = {}
                min_height: float | None = None
                try:
                    worker = NativeLatentPlanWorker(
                        request_slot,
                        response_slot,
                        service,
                        z_dim=int(command.z_dim),
                        plan_slots=plan_slots,
                        hold_steps=effective_hold,
                        lead_ticks=lead_ticks,
                        create_slots=True,
                    )
                    runtime = NativeMujocoLoop(
                        bundle_object,
                        str(model),
                        request_slot=request_slot,
                        response_slot=response_slot,
                        create_slots=False,
                        command_stale_ms=command_stale_ms,
                        lead_ticks=lead_ticks,
                        plan_slots=plan_slots,
                        latent_plan=True,
                        policy_threads=policy_threads,
                    )
                    recorder = TelemetryRecorder(
                        runtime,
                        logger=logger.child(episode_id),
                        sample_hz=telemetry_hz,
                    )
                    worker.start()
                    recorder.start()
                    runtime.start(ticks, paced=True)
                    runtime.wait()
                except BaseException as exc:
                    failure = exc
                    if runtime is not None:
                        try:
                            runtime.stop()
                            runtime.wait()
                        except Exception:
                            pass
                finally:
                    if recorder is not None:
                        recorder.stop()
                    if worker is not None:
                        worker.close()

                if runtime is not None and recorder is not None:
                    try:
                        control = runtime.stats()
                        min_height = runtime.min_base_height
                        telemetry = recorder.collect()
                        recorder.save(episode_dir / "telemetry", telemetry)
                    except Exception as exc:
                        if failure is None:
                            failure = exc
                    finally:
                        runtime.close()
                if (
                    failure is None
                    and worker is not None
                    and worker.last_error is not None
                ):
                    failure = worker.last_error

                trajectory, xy = trajectory_metrics(
                    telemetry,
                    control_hz=control_hz,
                    min_physics_height=min_height,
                )
                planner = _planner_metrics(
                    worker.request_ms if worker is not None else [],
                    service.head_ms[head_start:],
                )
                deadlines = int(control.get("deadline_misses", 0)) + int(
                    control.get("backend_deadline_misses", 0)
                ) + int(
                    control.get("scheduler_deadlines_missed", 0)
                )
                success = (
                    failure is None
                    and int(control.get("fault", 0)) == 0
                    and int(control.get("ticks", 0)) == ticks
                    and int(control.get("control_ticks", 0)) > 0
                    and int(control.get("damp_ticks", 0)) == 0
                    and int(control.get("planner_responses", 0)) > 0
                    and deadlines == 0
                    and not trajectory["fell_below_0_4"]
                )
                episode = {
                    "episode_id": episode_id,
                    "goal": str(goal),
                    "repeat": repeat,
                    "success": success,
                    "started_at": episode_started,
                    "completed_at": _timestamp(),
                    "control": control,
                    "planner": planner,
                    "trajectory": trajectory,
                    "error": _error_text(failure),
                    "artifacts": {
                        "report": f"episodes/{episode_id}/report.json",
                        "telemetry": (
                            f"episodes/{episode_id}/telemetry/telemetry.npz"
                            if telemetry
                            else None
                        ),
                    },
                }
                _write_json(episode_dir / "report.json", episode)
                with (run_dir / "episodes.jsonl").open("a") as stream:
                    stream.write(json.dumps(episode, default=str) + "\n")
                episodes.append(episode)
                trajectories.append((episode, xy))
                save_progress()
                logger.event(
                    "native_motion_eval.episode",
                    phase="rollout",
                    episode_id=episode_id,
                    goal=goal,
                    repeat=repeat,
                    succeeded=success,
                )
                episode_index += 1
                if isinstance(failure, KeyboardInterrupt):
                    raise failure

        metrics = aggregate_motion_metrics(episodes)
        status = _status(
            state="completed",
            succeeded=bool(episodes) and all(row["success"] for row in episodes),
            started_at=started_at,
            expected=expected,
            completed=len(episodes),
            error=None,
        )
        _write_json(run_dir / "metrics.json", metrics)
        _write_json(run_dir / "status.json", status)
        logger.event(
            "native_motion_eval.completed",
            phase="finalize",
            succeeded=status["succeeded"],
            episodes=len(episodes),
            run_dir=str(run_dir),
        )
        return run_dir, {"status": status, "metrics": metrics}
    except BaseException as exc:
        state = "cancelled" if isinstance(exc, KeyboardInterrupt) else "failed"
        error = _error_text(exc)
        _write_json(run_dir / "metrics.json", aggregate_motion_metrics(episodes))
        _write_summary_csv(run_dir / "summary.csv", episodes)
        _write_trajectory_svg(run_dir / "trajectories.svg", trajectories)
        _write_json(
            run_dir / "status.json",
            _status(
                state=state,
                succeeded=False,
                started_at=started_at,
                expected=expected,
                completed=len(episodes),
                error=error,
            ),
        )
        raise NativeMotionEvalError(str(exc) or state, run_dir) from exc
    finally:
        if service is not None:
            service.close()
        logger.close()
