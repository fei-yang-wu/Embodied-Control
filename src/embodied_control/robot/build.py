"""Build a lifecycle, a tracker or a whole session from a job.

Everything `ec lifecycle console`, `ec lifecycle run` and `ec lifecycle
rehearse` share lives here rather than in `cli.py`, so a campaign script can
import public names instead of the CLI's private helpers.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BuildOptions:
    """What a build needs from the command line, without argparse."""

    network: str = ""
    enable_writes: bool = False
    allow_non_realtime: bool = False
    confirm: str = ""
    artifacts: str = ""
    auto_ack: bool = False
    planner_autostart: bool = False
    offline: bool = False
    token: str = ""
    job: str = ""
    target: str = ""
    dds_domain: int | None = None


def write_gate_error(
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


def stationary_anchor(
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




def reference_pose(job, *, motion=None, start_frame=None):
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


def pose_tolerances(bundle, spec):
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


def tracker_bundle_paths(job) -> dict[str, str]:
    paths = {Path(job.bundle).name: job.bundle}
    paths.update(job.trackers)
    return paths


def job_selection(job, tracker: str = ""):
    from embodied_control.robot.session import Selection

    return Selection(
        mode=job.command_source,
        motion=job.motion,
        start_frame=int(job.start_frame),
        tracker=tracker,
    )


def screening_split(reasons, policy: str) -> tuple[list[str], list[str]]:
    """(blocking, reported) endpoint-screening reasons under a job's policy.

    Reasons name their endpoint ("start ..." / "end ..."); anything else is
    a broken reference (missing arrays, non-finite data) and blocks unless
    screening is off.
    """
    reasons = [str(r) for r in reasons]
    if policy == "off":
        return [], reasons
    if policy == "both":
        return reasons, []
    blocking = [r for r in reasons if not r.startswith("end ")]
    return blocking, [r for r in reasons if r.startswith("end ")]


def build_tracker(job, args, bundle, selection):
    """Runtime plus the start pose and reference gravity for one selection."""
    from embodied_control.lowlevel.native_core import NativeUnitreeLoop

    network = args.network or job.network
    fixed_anchor = None
    if selection.mode == "oracle":
        from embodied_control.lowlevel.reference import ReferenceArrays
        from embodied_control.lowlevel.reference_deploy import classify_motion
        from embodied_control.robot.rehearsal import SIM_NETWORKS
        reference = ReferenceArrays(job.reference_root)
        from embodied_control.lowlevel.reference_catalog import reference_compatibility
        reference_compatibility(bundle, reference)
        selected_motion = reference.motion(selection.motion)
        metadata = reference.manifest.get("motions", {}).get(selection.motion, {})
        if job.stand_hold_seconds > 0:
            hold_frames = int(metadata.get("hold_frames", 0))
            lookahead = bundle.manifest.command.horizon_steps * bundle.manifest.command.macro_frame_stride
            if hold_frames <= lookahead or selection.start_frame != 0:
                raise ValueError("final policy hold requires a composed stance reference covering this encoder's lookahead")
            import numpy as np
            for segment in (slice(0, hold_frames), slice(-hold_frames, None)):
                for array in (selected_motion.joint_qpos, selected_motion.anchor_pos_w, selected_motion.anchor_quat_w):
                    if not np.allclose(array[segment], array[segment][0], atol=1e-6):
                        raise ValueError("composed stance segment is not stationary")
                if selected_motion.joint_qvel is None or not np.allclose(selected_motion.joint_qvel[segment], 0, atol=1e-6):
                    raise ValueError("composed stance segment has nonzero joint velocity")
        if network not in SIM_NETWORKS:
            if selection.start_frame != 0:
                raise ValueError("hardware playback requires a screened start at frame 0")
            verdict = classify_motion(selected_motion, fps=reference.fps)
            blocking, reported = screening_split(verdict.reasons, job.endpoint_screening)
            if blocking:
                raise ValueError("reference fails deployment endpoint screening: " + "; ".join(blocking))
            for reason in reported:
                print(f"Endpoint screening (diagnostic, not a gate): {reason}")
        fixed_anchor, displacement, available = stationary_anchor(
            bundle,
            job.reference_root,
            selection.motion,
            selection.start_frame,
            job.fixed_anchor_max_displacement,
        )
        # A later start frame leaves fewer frames; the episode budget follows.
        ticks = int(available) if job.ticks == "auto" else min(int(job.ticks), int(available))
        print(f"Fixed initial anchor: max reference displacement {displacement:.4f} m; budget {ticks} ticks")
    else:
        if job.ticks == "auto":
            raise ValueError("ticks=auto requires oracle playback")
        ticks = int(job.ticks)
    start_pose = None
    reference_gravity = None
    if job.start_pose == "motion" or (selection.mode == "oracle" and job.start_pose != "default" and not isinstance(job.start_pose, list)):
        start_pose, reference_gravity = reference_pose(
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
        stiffness_scale=job.gain_scale.stiffness,
        damping_scale=job.gain_scale.damping,
        plan_slots=job.planner.vla_plan_slots if selection.mode == "vla" else 1,
        latent_plan=selection.mode == "vla" and job.planner.vla_reply == "latent_plan",
    )
    return runtime, start_pose, reference_gravity, ticks


def rehearsal_root(job) -> str:
    """Where to look for a plant rehearsal of this job."""
    if job.rehearsal_root:
        return job.rehearsal_root
    # A sim job and its hardware twin write beside each other, so the parent
    # of this run's artifacts is where the rehearsal lands.
    return str(Path(job.artifacts_dir).parent) if job.artifacts_dir else ""


def run_identity_for(job, bundle, network: str, selection=None, ticks=None) -> dict:
    from embodied_control.robot.rehearsal import run_identity

    source = bundle.manifest.source or {}
    motion = selection.motion if selection is not None else job.motion
    start_frame = selection.start_frame if selection is not None else job.start_frame
    mode = selection.mode if selection is not None else job.command_source
    reference_sha = ""
    if job.reference_root and motion:
        from embodied_control.lowlevel.reference import ReferenceArrays
        from embodied_control.lowlevel.reference_catalog import motion_sha256
        reference_sha = motion_sha256(ReferenceArrays(job.reference_root), motion)
    import hashlib
    deployment = {key: getattr(job, key) for key in (
        "start_pose", "fixed_initial_anchor", "pin_reference", "ramp_seconds", "lead_ticks",
        "blend_ticks", "play_countdown_seconds", "arm_timeout_seconds", "stand_hold_seconds",
    )}
    deployment["gain_scale"] = job.gain_scale.model_dump()
    deployment["rehearsal_hoist_contract"] = "g1_shoulder_straps_v1"
    deployment["startup_contract"] = "diagnostic_pose_operator_play_v1"
    deployment["thresholds"] = job.thresholds.model_dump()
    deployment["bundle_manifest"] = bundle.manifest.model_dump(mode="json")
    deployment_sha = hashlib.sha256(json.dumps(deployment, sort_keys=True).encode()).hexdigest()
    return run_identity(
        bundle_sha=str(source.get("checkpoint_sha256", "")),
        deployment_sha=deployment_sha,
        bundle_name=bundle.root.name,
        motion=motion,
        reference_sha=reference_sha,
        command_source=mode,
        network=network,
        start_frame=start_frame,
        ticks=int(ticks if ticks is not None else (0 if job.ticks == "auto" else job.ticks)),
    )


def lifecycle_config(job, args, bundle, start_pose, reference_gravity, ticks=None, selection=None):
    from embodied_control.robot.lifecycle import LifecycleConfig

    thresholds = job.thresholds
    terminal_hold_frames = 0
    if job.stand_hold_seconds > 0:
        manifest = json.loads((Path(job.reference_root) / "reference_arrays_manifest.json").read_text())
        motion = selection.motion if selection is not None else job.motion
        terminal_hold_frames = int(manifest["motions"][motion]["hold_frames"])
    return LifecycleConfig(
        start_pose=start_pose,
        reference_gravity=reference_gravity,
        ramp_seconds=job.ramp_seconds,
        ticks=int(job.ticks if ticks is None else ticks),
        blend_ticks=job.blend_ticks,
        play_countdown_seconds=job.play_countdown_seconds,
        arm_timeout_seconds=job.arm_timeout_seconds,
        stand_hold_seconds=job.stand_hold_seconds,
        terminal_hold_frames=terminal_hold_frames,
        end_state=job.end_state,
        damp_hands_back=job.damp_hands_back,
        retake_precheck=job.retake_precheck,
        require_rehearsal=job.require_rehearsal,
        rehearsal_root=rehearsal_root(job),
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
        pose_tolerance_rad=pose_tolerances(bundle, thresholds.pose_tolerance_rad),
        tilt_tolerance_degrees=thresholds.tilt_tolerance_degrees,
        first_action_rad=thresholds.first_action_rad,
        first_action_torque_ratio=thresholds.first_action_torque_ratio,
        stiffness=[float(v) * job.gain_scale.stiffness for v in bundle.manifest.action.stiffness],
        effort_limit=(
            [float(v) for v in bundle.manifest.action.effort_limit]
            if bundle.manifest.action.effort_limit
            else None
        ),
        command_timeout_seconds=thresholds.command_timeout_seconds,
        vendor_timeout_seconds=thresholds.vendor_timeout_seconds,
    )


def lifecycle_peers(job, args):
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


def resolved_bundle(path, args):
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


def load_job(args):
    from embodied_control.robot.lifecycle_job import apply_target, load_lifecycle_job

    job = load_lifecycle_job(args.job)
    target = getattr(args, "target", "") or ""
    if target:
        job = apply_target(
            job, target, network=args.network, dds_domain=getattr(args, "dds_domain", None)
        )
    gate_error = write_gate_error(
        enable_writes=args.enable_writes,
        allow_non_realtime=args.allow_non_realtime,
        confirm=args.confirm,
    )
    if gate_error is not None:
        raise ValueError(gate_error)
    return job, resolved_bundle(job.bundle, args)


def build_lifecycle(args):
    """Job -> (job, lifecycle, runtime, vendor) for the scripted `run`."""
    from embodied_control.robot.lifecycle import Lifecycle, LifecycleLog

    job, bundle = load_job(args)
    selection = job_selection(job)
    runtime, start_pose, reference_gravity, ticks = build_tracker(job, args, bundle, selection)
    vendor, hoist = lifecycle_peers(job, args)
    config = lifecycle_config(job, args, bundle, start_pose, reference_gravity, ticks, selection)
    artifacts = args.artifacts or job.artifacts_dir or None
    lifecycle = Lifecycle(
        runtime,
        vendor,
        config,
        hoist=hoist,
        auto_ack=bool(job.sim_hoist or args.auto_ack),
        identity=run_identity_for(job, bundle, args.network or job.network, selection, ticks),
        log=LifecycleLog(artifacts),
        note=lambda msg: print(f"  -- {msg}", flush=True),
    )
    return job, lifecycle, runtime, vendor


def episode_mpjpe(job, bundle_for):
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
        if str(telemetry.get("anchor_pose_source", "unknown")) not in {
            "simulator_ground_truth", "measured_root",
        }:
            return None
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


def build_session(args):
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

    job, default_bundle = load_job(args)
    tracker_paths = tracker_bundle_paths(job)
    default_tracker = next(iter(tracker_paths))
    bundles = {default_tracker: default_bundle}

    def bundle_for(selection):
        name = selection.tracker or default_tracker
        if name not in tracker_paths:
            raise ValueError(f"unknown tracker {name}")
        if name not in bundles:
            bundles[name] = resolved_bundle(tracker_paths[name], args)
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
    vendor, hoist = lifecycle_peers(job, args)
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
        runtime, start_pose, reference_gravity, ticks = build_tracker(job, args, bundle, selection)
        pending["config"] = lifecycle_config(job, args, bundle, start_pose, reference_gravity, ticks, selection)
        # The console can switch bundle and motion between episodes, so the
        # identity is rebuilt with the tracker rather than read once.
        pending["identity"] = run_identity_for(
            job, bundle, args.network or job.network, selection, ticks
        )
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
        job_selection(job, default_tracker),
        hoist=hoist,
        planner_factory=planner_factory,
        tracker_factory=tracker_factory,
        lifecycle_factory=lifecycle_factory,
        slot_names=[job.request_slot, job.response_slot] if job.connect_slots else [],
        note=note,
        mpjpe=episode_mpjpe(job, bundle_for),
    )
    session.note_sinks = notes
    session.pinner = pinner
    return job, session, vendor


