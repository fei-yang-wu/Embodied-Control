"""Oracle-driven MPJPE-L/G sweep over every motion in a reference tree.

For each motion: the native oracle worker streams the reference over shm,
the C++ MuJoCo rehearsal rig tracks it in real time from the frame-0 pose,
and FK replay of the recorded telemetry yields root-relative (MPJPE-L) and
world-frame (MPJPE-G) errors under the frozen protocol definitions.

Run inside the native environment:

    pixi run -e native python scripts/oracle_mpjpe_eval.py \
        --bundle <bundle_dir> --model <g1_xml> --reference-root <root_qpos_v1> \
        --output oracle_mpjpe_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.metrics import oracle_tracking_metrics
from embodied_control.lowlevel.native_core import NativeMujocoLoop
from embodied_control.lowlevel.publishers.native_oracle import NativeOracleWorker
from embodied_control.lowlevel.reference import ReferenceArrays
from embodied_control.lowlevel.slo import evaluate_report
from embodied_control.lowlevel.telemetry import TelemetryRecorder


def evaluate_motion(bundle, model, reference_root, name, *, cpu, physics_cpu):
    arrays = ReferenceArrays(reference_root)
    motion = arrays.motion(name)
    ticks = motion.length - 1
    slot = f"/ec_om_{uuid.uuid4().hex[:8]}"
    worker = NativeOracleWorker(
        f"{slot}_req", f"{slot}_resp", bundle, reference_root, name,
        create_slots=True,
    )
    worker.start()
    loop = NativeMujocoLoop(
        bundle, model, response_slot=f"{slot}_resp", request_slot=f"{slot}_req",
        create_slots=False, command_source="oracle", lead_ticks=4,
        cpu=cpu, physics_cpu=physics_cpu,
    )
    loop.set_initial_pose(
        np.concatenate(
            [motion.anchor_pos_w[0], motion.anchor_quat_w[0], motion.joint_qpos[0]]
        )
    )
    recorder = TelemetryRecorder(loop, sample_hz=0)
    loop.start(ticks, paced=True)
    loop.wait()
    record = recorder.collect()
    worker.close()
    stats = record["stats"]
    verdict = evaluate_report({"control": stats}, record["tick_durations_ns"])
    tracking = oracle_tracking_metrics(bundle.manifest.action, model, motion, record)
    tracking.pop("per_frame")
    tracking.pop("tracked_bodies")
    heights = record["base_heights"]
    heights = heights[np.isfinite(heights)]
    return {
        "motion": name,
        "ticks": ticks,
        "no_fall": bool((heights > 0.4).all()),
        "base_height_min": round(float(heights.min()), 4),
        "mae_rad": round(float(np.nanmean(record["reference_joint_mae"])), 4),
        **{
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in tracking.items()
        },
        "deadline_misses": stats["deadline_misses"],
        "fault": stats["fault"],
        "tick_p99_ms": round(
            verdict["measured"].get("control_tick_compute_p99_ms", -1), 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--motions", nargs="*", default=None,
        help="subset of motion names; default is every motion in the tree",
    )
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--physics-cpu", type=int, default=3)
    args = parser.parse_args()

    bundle = PolicyBundle.load(args.bundle)
    names = args.motions or ReferenceArrays(args.reference_root).motion_names
    rows = []
    for name in names:
        row = evaluate_motion(
            bundle, args.model, args.reference_root, name,
            cpu=args.cpu, physics_cpu=args.physics_cpu,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)

    frames = [row["frames"] for row in rows]
    l_values = [row["mpjpe_l_mm"] for row in rows]
    g_values = [row["mpjpe_g_mm"] for row in rows]
    total = sum(frames)
    report = {
        "protocol": {
            "stack": (
                "native C++ 50 Hz control + C++ MuJoCo plant (paced RTF 1.0) "
                "+ shm oracle worker"
            ),
            "bundle": str(Path(args.bundle).name),
            "reference": str(args.reference_root),
            "start": "frame 0, robot initialized on reference frame-0 pose",
            "randomization": "none (deterministic plant reset)",
            "seeds": "single pass per motion",
            "mpjpe_defs": (
                "protocol formulas from evaluate_checkpoint.py:548-576; "
                "micro-averaged by frame"
            ),
            "note": (
                "sim2sim deployment signal on the rehearsal rig, not an Isaac "
                "paper number; single seed = preliminary"
            ),
        },
        "motions": rows,
        "aggregate": {
            "motions": len(rows),
            "survival": f"{sum(row['no_fall'] for row in rows)}/{len(rows)}",
            "mpjpe_l_mm_micro": round(
                sum(a * b for a, b in zip(l_values, frames)) / total, 2
            ),
            "mpjpe_g_mm_micro": round(
                sum(a * b for a, b in zip(g_values, frames)) / total, 2
            ),
            "mpjpe_l_mm_range": [round(min(l_values), 2), round(max(l_values), 2)],
            "mpjpe_g_mm_range": [round(min(g_values), 2), round(max(g_values), 2)],
            "deadline_misses_total": int(
                sum(row["deadline_misses"] for row in rows)
            ),
            "faults_total": int(sum(row["fault"] != 0 for row in rows)),
        },
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"REPORT: {output}")
    print(json.dumps(report["aggregate"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
