"""The experiment session behind the lifecycle console.

The lifecycle drives one episode of one motion under one command source. A
session is what the operator sits in front of for an afternoon: pick oracle
or planner, pick the motion and the start frame, start the planner process,
build the tracker for that choice, run episodes, and keep the per-episode
telemetry so MPJPE can be graded afterwards. Reconfiguring is only allowed
while nothing owns the joints.

Everything that touches a process or the native runtime is injected, so the
session's rules are tested on fakes in the light env.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

from embodied_control.robot.gates import GateResult
from embodied_control.robot.lifecycle import (
    OWNING_STATES,
    Lifecycle,
    LifecycleState,
    Transition,
)
from embodied_control.robot.rehearsal import ACCEPTED_END_STATES

MODES = ("oracle", "vla")


@dataclass(frozen=True)
class Selection:
    mode: str = "oracle"
    motion: str = ""
    start_frame: int = 0
    tracker: str = ""

    def label(self) -> str:
        suffix = f" [{self.tracker}]" if self.tracker else ""
        if self.mode == "oracle":
            return f"oracle {self.motion}@{self.start_frame}{suffix}"
        return (
            f"vla planner (start pose {self.motion or 'default'}@{self.start_frame})"
            f"{suffix}"
        )


class PlannerHandle(Protocol):
    """A running planner process for one selection."""

    def alive(self) -> bool: ...

    def stop(self) -> None: ...

    def describe(self) -> str: ...


class TrackerHandle(Protocol):
    def close(self) -> None: ...


class SubprocessPlanner:
    """oracle-worker or planner-worker as a child of the console.

    Its own process, so its ONNX session, its disk reads and its Python GC
    cannot touch the control loop's deadline or the console's key handler.
    `preexec` keeps it off the cores the robot threads are pinned to; it is
    never renice'd, because the control loop waits on its replies.
    """

    def __init__(
        self,
        argv: list[str],
        log_path: Path | None = None,
        preexec: Callable[[], None] | None = None,
    ) -> None:
        self.argv = list(argv)
        self.log_path = log_path
        self._log = open(log_path, "ab") if log_path else subprocess.DEVNULL
        self._process = subprocess.Popen(
            self.argv,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            preexec_fn=preexec,
            start_new_session=True,
        )

    def alive(self) -> bool:
        return self._process.poll() is None

    def stop(self) -> None:
        if self._process.poll() is None:
            self._process.send_signal(2)
            try:
                self._process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5.0)
        if self._log not in (subprocess.DEVNULL, None):
            self._log.close()

    def describe(self) -> str:
        name = self.argv[self.argv.index("lowlevel") + 1] if "lowlevel" in self.argv else self.argv[0]
        state = "running" if self.alive() else f"exited {self._process.returncode}"
        return f"{name} pid {self._process.pid} {state}"


def oracle_worker_argv(
    bundle: str,
    reference_root: str,
    motion: str,
    start_frame: int,
    request_slot: str,
    response_slot: str,
    *,
    # 0 lets the worker read the bundle's own encoder window, which is the
    # only value that is right for every frame stride.
    horizon: int | None = None,
    report: str = "",
) -> list[str]:
    argv = [
        sys.executable, "-m", "embodied_control.cli", "lowlevel", "oracle-worker",
        bundle, "--reference-root", reference_root, "--motion", motion,
        "--start-frame", str(start_frame), "--request-slot", request_slot,
        "--response-slot", response_slot, "--horizon", str(horizon or 0),
        "--create-slots",
    ]
    if report:
        argv += ["--report", report]
    return argv


def planner_worker_argv(
    service_command: list[str],
    request_slot: str,
    response_slot: str,
    *,
    reply: str = "latent_plan",
    z_dim: int = 256,
    plan_slots: int = 1,
    hold_steps: int = 10,
    lead_ticks: int = 4,
    report: str = "",
) -> list[str]:
    argv = [
        sys.executable, "-m", "embodied_control.cli", "lowlevel", "planner-worker",
        "--request-slot", request_slot, "--response-slot", response_slot,
        "--create-slots", "--reply", reply, "--z-dim", str(z_dim),
        "--plan-slots", str(plan_slots), "--hold-steps", str(hold_steps),
        "--lead-ticks", str(lead_ticks),
    ]
    if report:
        argv += ["--report", report]
    return argv + ["--", *service_command]


def unlink_slot_files(names: list[str]) -> None:
    for name in names:
        try:
            (Path("/dev/shm") / name.lstrip("/")).unlink()
        except FileNotFoundError:
            pass


def wait_for_slots(
    names: list[str], timeout_seconds: float, *, sleep=time.sleep, exists=None
) -> bool:
    """Block until the planner has created its shared-memory mailboxes."""
    probe = exists or (lambda name: (Path("/dev/shm") / name.lstrip("/")).exists())
    deadline = time.monotonic() + timeout_seconds
    while True:
        if all(probe(name) for name in names):
            return True
        if time.monotonic() >= deadline:
            return False
        sleep(0.1)


def rows_of(values, width: int) -> list[list[float]]:
    """Rows from a per-tick log that may arrive flat (ticks * width) or 2-D."""
    items = list(values)
    if not items:
        return []
    first = items[0]
    if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
        return [[float(v) for v in row] for row in items]
    flat = [float(v) for v in items]
    return [flat[i:i + width] for i in range(0, len(flat) - width + 1, width)]


def episode_summary(
    joint_position_log: list[list[float]],
    reference_frames: list[int],
    reference_joint_mae: list[float],
    stats: dict,
    writer: dict,
    selection: Selection,
    episode: int,
) -> dict:
    """Tracking numbers for one episode, without numpy.

    joint MAE is the runtime's per-tick mean |q - reference| over 29 joints;
    MPJPE proper needs forward kinematics and is graded offline from the
    saved telemetry (`eval_mpjpe`), or in-line when MuJoCo is importable.
    """
    mae = [float(v) for v in reference_joint_mae if v is not None and not math.isnan(float(v))]
    frames = [int(f) for f in reference_frames if int(f) >= 0]
    ordered = sorted(mae)
    p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else float("nan")
    return {
        "episode": episode,
        "mode": selection.mode,
        "tracker": selection.tracker,
        "motion": selection.motion,
        "start_frame": selection.start_frame,
        "ticks": int(stats.get("control_ticks", 0)),
        "reference_ticks": int(stats.get("reference_ticks", 0)),
        "runtime_fault": int(stats.get("fault", 0)),
        "hardware_faults": int(writer.get("hardware_faults", 0)),
        "watchdog_faults": int(writer.get("watchdog_faults", 0)),
        "frames_tracked": len(frames),
        "first_frame": min(frames) if frames else None,
        "last_frame": max(frames) if frames else None,
        "joint_mae_mean_rad": sum(mae) / len(mae) if mae else float("nan"),
        "joint_mae_p95_rad": p95,
        "joint_mae_max_rad": max(mae) if mae else float("nan"),
        "rows": len(joint_position_log),
    }


@dataclass
class SessionConfig:
    catalog: list[str]
    motion_lengths: dict[str, int]
    trackers: list[str] = field(default_factory=list)
    modes: tuple[str, ...] = MODES
    frame_step: int = 25
    slot_timeout_seconds: float = 20.0
    artifacts_dir: str | None = None
    # Whether building a tracker may start a planner that has to be paid for.
    # Off by default; see `_planner_is_cheap` for the modes this never
    # applies to.
    planner_autostart: bool = False


class ExperimentSession:
    def __init__(
        self,
        config: SessionConfig,
        selection: Selection,
        *,
        hoist: Hoist | None = None,
        planner_factory: Callable[[Selection], PlannerHandle],
        tracker_factory: Callable[[Selection], TrackerHandle],
        lifecycle_factory: Callable[[TrackerHandle, Selection, "ExperimentSession"], Lifecycle],
        slot_names: list[str] | None = None,
        wait_slots: Callable[[list[str], float], bool] | None = None,
        unlink_slots: Callable[[list[str]], None] | None = None,
        note: Callable[[str], None] = lambda _msg: None,
        mpjpe: Callable[[Path, Selection], dict | None] | None = None,
    ) -> None:
        self.config = config
        self.selection = selection
        self.hoist = hoist
        self._planner_factory = planner_factory
        self._tracker_factory = tracker_factory
        self._lifecycle_factory = lifecycle_factory
        self._slot_names = list(slot_names or [])
        self._wait_slots = wait_slots or wait_for_slots
        self._unlink_slots = unlink_slots or unlink_slot_files
        self._note = note
        self._mpjpe = mpjpe
        self.planner: PlannerHandle | None = None
        self.tracker: TrackerHandle | None = None
        self.lifecycle: Lifecycle | None = None
        self.built_for: Selection | None = None
        self.episodes: list[dict] = []
        self._episode_written: tuple | None = None
        self._episode_directory: Path | None = None
        self.note_sinks: list = []
        # The plant's hoist status is a blocking DDS RPC. `poll` refreshes it
        # from the watcher thread; `snapshot` only ever reads the cache, so a
        # hung plant cannot freeze the display.
        self._hoist_status: dict | None = None
        self._hoist_checked = 0.0
        self._hoist_period = 0.5

    # ---------------------------------------------------------- selection

    def can_reconfigure(self) -> bool:
        if self.lifecycle is None:
            return True
        if self.lifecycle.state in OWNING_STATES:
            return False
        return not getattr(self.tracker, "running", False)

    def _reconfigure(self, change: Selection) -> GateResult:
        if not self.can_reconfigure():
            return self._refuse(
                f"cannot change the selection in {self.lifecycle.state if self.lifecycle else '?'}; "
                "hold, hand back, then choose"
            )
        self.selection = change
        stale = self.built_for is not None and self.built_for != change
        self._note(f"selected {change.label()}" + (" (press r to rebuild)" if stale else ""))
        return GateResult(True, change.label())

    def select_mode(self, mode: str) -> GateResult:
        if mode not in self.config.modes:
            return self._refuse(f"unknown mode {mode}")
        return self._reconfigure(replace(self.selection, mode=mode))

    def toggle_mode(self) -> GateResult:
        index = self.config.modes.index(self.selection.mode)
        return self.select_mode(self.config.modes[(index + 1) % len(self.config.modes)])

    def step_motion(self, step: int) -> GateResult:
        if not self.config.catalog:
            return self._refuse("no reference catalog")
        try:
            index = self.config.catalog.index(self.selection.motion)
        except ValueError:
            index = -1
        index = (index + step) % len(self.config.catalog)
        motion = self.config.catalog[index]
        frame = min(self.selection.start_frame, self.motion_length(motion) - 1)
        return self._reconfigure(replace(self.selection, motion=motion, start_frame=max(0, frame)))

    def step_frame(self, delta: int) -> GateResult:
        length = self.motion_length(self.selection.motion)
        frame = max(0, min(length - 1, self.selection.start_frame + delta))
        return self._reconfigure(replace(self.selection, start_frame=frame))

    def step_tracker(self, step: int) -> GateResult:
        if not self.config.trackers:
            return self._refuse("no tracker catalog")
        try:
            index = self.config.trackers.index(self.selection.tracker)
        except ValueError:
            index = -1
        tracker = self.config.trackers[(index + step) % len(self.config.trackers)]
        return self._reconfigure(replace(self.selection, tracker=tracker))

    def motion_length(self, motion: str) -> int:
        return int(self.config.motion_lengths.get(motion, 1))

    # ------------------------------------------------------------ planner

    def start_planner(self) -> GateResult:
        if self.planner is not None and self.planner.alive():
            return self._refuse("planner already running; press p to stop it")
        if self.selection.mode == "oracle" and not self.selection.motion:
            return self._refuse("pick a motion first")
        # A previous worker's mailboxes outlive it, and a new one refuses to
        # create a slot that already exists.
        if self._slot_names:
            self._unlink_slots(self._slot_names)
        try:
            self.planner = self._planner_factory(self.selection)
        except Exception as exc:
            return self._refuse(f"planner failed to start: {exc}")
        if self._slot_names and not self._wait_slots(self._slot_names, self.config.slot_timeout_seconds):
            self.planner.stop()
            return self._refuse("planner did not create its slots in time")
        self._note(f"planner started: {self.planner.describe()}")
        return GateResult(True, self.planner.describe())

    def stop_planner(self) -> GateResult:
        if self.planner is None:
            return self._refuse("no planner running")
        if not self.can_reconfigure():
            return self._refuse("the tracker owns the joints; hold and hand back first")
        self.planner.stop()
        described = self.planner.describe()
        self.planner = None
        self._note(f"planner stopped: {described}")
        return GateResult(True, described)

    def _planner_is_cheap(self) -> bool:
        """Whether this selection's planner costs nothing worth asking about.

        The oracle worker memory-maps the reference arrays and copies frame
        windows out of them; it loads no weights and measures about 58 MB
        with no GPU. The VLA worker launches a service that loads its own
        checkpoint, several gigabytes on the GPU, which is the one an
        operator should have to ask for.
        """
        return self.selection.mode == "oracle"

    def toggle_planner(self) -> GateResult:
        if self.planner is not None and self.planner.alive():
            return self.stop_planner()
        return self.start_planner()

    # ------------------------------------------------------------ tracker

    def rebuild(self) -> GateResult:
        """Build the tracker and lifecycle for the current selection."""
        if not self.can_reconfigure():
            return self._refuse("cannot rebuild while the tracker owns the joints")
        if self.lifecycle is not None:
            self.lifecycle.shutdown()
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        # A fresh tracker numbers its requests from 1 again and an oracle
        # worker serves one start frame, so the planner restarts with the
        # tracker: stop it, drop the old mailboxes, start it for this
        # selection, then connect.
        had_planner = self.planner is not None
        if self.planner is not None:
            self.planner.stop()
            self.planner = None
        if self._slot_names:
            self._unlink_slots(self._slot_names)
        # The planner owns the mailboxes and the tracker connects to them, so
        # a tracker that needs slots cannot be built before one is running.
        # Starting a cheap one is a detail; starting a multi-gigabyte one
        # because somebody pressed `n` is not.
        may_start = (
            had_planner or self.config.planner_autostart or self._planner_is_cheap()
        )
        if not may_start and self._slot_names:
            return self._refuse(
                "no planner running: press p to start it for this "
                "selection, then r to build the tracker"
            )
        result = self.start_planner()
        if not result.ok:
            return result
        try:
            self.tracker = self._tracker_factory(self.selection)
            self.lifecycle = self._lifecycle_factory(self.tracker, self.selection, self)
        except Exception as exc:
            self.tracker = None
            self.lifecycle = None
            return self._refuse(f"tracker build failed: {exc}")
        self.lifecycle.on_transition = self._on_transition
        self.built_for = self.selection
        self._note(f"tracker built for {self.selection.label()}")
        return GateResult(True, f"built for {self.selection.label()}")

    def ensure_built(self) -> GateResult:
        if self.lifecycle is not None and self.built_for == self.selection:
            return GateResult(True, "built")
        return self.rebuild()

    def close(self) -> None:
        if self.lifecycle is not None:
            self.lifecycle.shutdown()
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        if self.planner is not None:
            self.planner.stop()
            self.planner = None

    def reset_sim(self) -> GateResult:
        """Drop controller state, then ask simulated plant for a clean boot."""
        reset = getattr(self.hoist, "reset", None)
        if reset is None:
            return self._refuse("sim reset is unavailable for this session")
        if self.tracker is not None:
            self.emergency_damp()
            self.tracker.close()
            self.tracker = None
        if self.planner is not None:
            self.planner.stop()
            self.planner = None
        if self.lifecycle is not None:
            self.lifecycle.log.finish(self.lifecycle.summary())
            self.lifecycle = None
        self.built_for = None
        self._episode_written = None
        try:
            reset()
            self.refresh_hoist(now=self._hoist_checked + self._hoist_period)
        except Exception as exc:
            return self._refuse(f"sim reset failed: {exc}")
        self._note("sim reset to nominal pose; rebuild tracker when ready")
        return GateResult(True, "sim reset to nominal pose")

    # ------------------------------------------------------------ episodes

    def _on_transition(self, transition: Transition) -> None:
        if not transition.ok or self.lifecycle is None:
            return
        if transition.to_state in ACCEPTED_END_STATES:
            self._record_evidence()
            return
        if transition.to_state not in {str(LifecycleState.HOLD), str(LifecycleState.FAULT), str(LifecycleState.DAMP)}:
            return
        if transition.from_state not in {str(LifecycleState.RUNNING), str(LifecycleState.BLEND_IN), str(LifecycleState.ARMED), str(LifecycleState.STAND_HOLD)}:
            return
        # The lifecycle counts episodes per build; the session counts them
        # for the afternoon, so directories never collide across rebuilds.
        marker = (id(self.lifecycle), self.lifecycle.episode)
        if marker == self._episode_written:
            return
        self._episode_written = marker
        episode = len(self.episodes) + 1
        try:
            self._record_episode(episode)
        except Exception as exc:  # telemetry must never break the lifecycle
            self._note(f"episode record failed: {exc}")

    def _record_evidence(self) -> None:
        """The rehearsal gate reads `lifecycle.json`; one per episode, not per directory.

        The shared `lifecycle.json` at the artifacts root is rewritten by
        every rebuild, so a session that rehearses four motions used to leave
        evidence for the last one only.
        """
        directory = self._episode_directory
        if directory is None or self.lifecycle is None:
            return
        summary = self.lifecycle.summary()
        (directory / "lifecycle.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (directory / "lifecycle.jsonl").open("w") as stream:
            for transition in self.lifecycle.transitions:
                stream.write(json.dumps(transition.__dict__) + "\n")
        self._episode_directory = None

    def _record_episode(self, episode: int) -> None:
        tracker = self.tracker
        assert tracker is not None
        joint_log = rows_of(tracker.joint_position_log(), 29)
        # The runtime's reference frames count from the worker's start frame.
        frames = [
            int(v) + self.selection.start_frame if int(v) >= 0 else int(v)
            for v in tracker.reference_frames()
        ]
        mae = [float(v) for v in tracker.reference_joint_mae()]
        summary = episode_summary(
            joint_log, frames, mae, tracker.stats(), tracker.writer_stats(),
            self.selection, episode,
        )
        directory = None
        if self.config.artifacts_dir:
            tracker_prefix = f"{self.selection.tracker}_" if self.selection.tracker else ""
            directory = Path(self.config.artifacts_dir) / "episodes" / (
                f"ep{episode:03d}_{tracker_prefix}{self.selection.mode}_"
                f"{self.selection.motion or 'default'}"
                f"@{self.selection.start_frame}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            try:
                import numpy as np

                np.savez(
                    directory / "telemetry.npz",
                    joint_position_log=np.asarray(joint_log, dtype=np.float32),
                    reference_frames=np.asarray(frames, dtype=np.int32),
                    reference_joint_mae=np.asarray(mae, dtype=np.float32),
                    anchor_pose_log=np.asarray(rows_of(tracker.anchor_pose_log(), 7), dtype=np.float32),
                    command_target_log=np.asarray(
                        rows_of(tracker.command_target_log(), 29) if hasattr(tracker, "command_target_log") else [],
                        dtype=np.float32,
                    ).reshape(-1, 29),
                    anchor_pose_source=np.asarray("controller_fixed_translation_imu_orientation"),
                    tick_durations_ns=np.asarray(tracker.tick_durations_ns(), dtype=np.uint64),
                )
            except ImportError:
                pass
            if self._mpjpe is not None:
                try:
                    mpjpe = self._mpjpe(directory, self.selection)
                    if mpjpe:
                        summary.update(mpjpe)
                except Exception as exc:
                    summary["mpjpe_error"] = str(exc)
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            summary["directory"] = str(directory)
            self._episode_directory = directory
        self.episodes.append(summary)
        self._note(
            f"episode {episode}: {summary['ticks']} ticks, joint MAE "
            f"{summary['joint_mae_mean_rad']:.3f} rad"
            + (f", MPJPE_l {summary['mpjpe_l_mm']:.1f} mm" if "mpjpe_l_mm" in summary else "")
        )

    # ------------------------------------------------------------- view

    def refresh_hoist(self, now: float | None = None) -> None:
        """Blocking plant RPC, called from the watcher thread only."""
        status = getattr(self.hoist, "status", None)
        if status is None:
            return
        moment = time.monotonic() if now is None else now
        if moment - self._hoist_checked < self._hoist_period:
            return
        self._hoist_checked = moment
        try:
            self._hoist_status = status()
        except Exception:
            self._hoist_status = None

    def snapshot(self) -> dict:
        base = self.lifecycle.snapshot() if self.lifecycle is not None else {
            "state": "NO TRACKER", "fault_reason": "", "vendor_name": "",
            "episode": 0, "writer": {}, "control": {}, "hoist": None,
            "last_ok": None,
            "last_detail": "press r to build the tracker",
        }
        planner = "not running"
        if self.planner is not None:
            planner = self.planner.describe()
        base["session"] = {
            "mode": self.selection.mode,
            "tracker": self.selection.tracker,
            "tracker_index": (self.config.trackers.index(self.selection.tracker) + 1)
            if self.selection.tracker in self.config.trackers else 0,
            "tracker_count": len(self.config.trackers),
            "motion": self.selection.motion,
            "motion_index": (self.config.catalog.index(self.selection.motion) + 1)
            if self.selection.motion in self.config.catalog else 0,
            "catalog_size": len(self.config.catalog),
            "start_frame": self.selection.start_frame,
            "motion_length": self.motion_length(self.selection.motion),
            "planner": planner,
            "built": self.built_for == self.selection and self.lifecycle is not None,
            "built_for": self.built_for.label() if self.built_for else "",
            "episodes": self.episodes[-3:],
        }
        if self._hoist_status is not None:
            base["hoist"] = self._hoist_status
        return base

    def emergency_damp(self) -> None:
        if self.lifecycle is not None:
            self.lifecycle.emergency_damp()

    def _refuse(self, detail: str) -> GateResult:
        self._note(detail)
        return GateResult(False, detail)

    # Lifecycle verbs forwarded to whatever lifecycle is built now.

    def _current(self) -> Lifecycle:
        if self.lifecycle is None or self.built_for != self.selection:
            result = self.ensure_built()
            if not result.ok:
                raise RuntimeError(result.detail)
        assert self.lifecycle is not None
        return self.lifecycle

    def advance(self) -> GateResult:
        return self._current().advance()

    def auto(self, until: LifecycleState = LifecycleState.PRIMED) -> GateResult:
        return self._current().auto(until)

    def arm(self) -> GateResult:
        return self._current().arm()

    def play(self) -> GateResult:
        return self._current().play()

    def go(self) -> GateResult:
        return self._current().go()

    def hold(self) -> GateResult:
        return self._current().hold()

    def damp(self) -> GateResult:
        if self.lifecycle is None:
            return self._refuse("no tracker built; nothing to damp")
        return self.lifecycle.damp()

    def retake(self) -> GateResult:
        return self._current().retake()

    def recover(self, end_state: str | None = None) -> GateResult:
        if self.lifecycle is None:
            return self._refuse("no tracker built")
        return self.lifecycle.recover(end_state)

    def abort(self) -> GateResult:
        if self.lifecycle is None:
            return self._refuse("no tracker built")
        return self.lifecycle.abort()

    def ack_hoisted(self) -> None:
        if self.lifecycle is not None:
            self.lifecycle.ack_hoisted()

    def ack_lowered(self) -> None:
        if self.lifecycle is not None:
            self.lifecycle.ack_lowered()

    def poll(self) -> None:
        self.refresh_hoist()
        if self.lifecycle is not None:
            self.lifecycle.poll()
