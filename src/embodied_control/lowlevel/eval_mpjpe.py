"""Post-hoc MPJPE for a lowlevel run's recorded states vs a reference motion.

Reads `states_ep{i}.npz` written by a `rollout.record_states=true` run,
forward-kinematics the robot body positions through the MJCF, and compares
against the reference tree's stored world body positions frame-by-frame
(tick t vs reference frame min(t, length-1) — frame-0 starts at 50 Hz).

Needs MuJoCo, so run it in the `lowlevel-sim` environment:

    pixi run -e lowlevel-sim python -m embodied_control.lowlevel.eval_mpjpe \
        --run-dir <run> --reference <arrays tree> --motion <name> --mjcf <xml>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.maths import quat_conjugate, quat_mul, quat_to_mat
from embodied_control.lowlevel.metrics import compute_mpjpe, fk_body_positions
from embodied_control.lowlevel.reference import ReferenceArrays


def _align_reference(
    anchor0_pose_xyzw: np.ndarray,
    reference_pos0: np.ndarray,
    reference_quat0_xyzw: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Map reference world points into the robot's world frame.

    The MuJoCo episode spawns the robot at the origin while the reference
    lives in the dataset's world frame; without the frame-0 anchor rigid
    alignment (`R = robot0 * ref0^-1`, `t = robot_pos0 - R @ ref_pos0`) both
    MPJPE variants measure the spawn offset, not tracking.
    """
    delta = quat_mul(
        np.asarray(anchor0_pose_xyzw[3:7], np.float32),
        quat_conjugate(np.asarray(reference_quat0_xyzw, np.float32)),
    )
    rotation = quat_to_mat(delta)
    translation = anchor0_pose_xyzw[0:3] - rotation @ np.asarray(reference_pos0)
    return points @ rotation.T + translation


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--motion", required=True)
    parser.add_argument("--mjcf", type=Path, required=True)
    parser.add_argument(
        "--render",
        action="store_true",
        help="also render a reference-left / policy-right comparison video per episode",
    )
    args = parser.parse_args(argv)

    manifest = json.loads((args.run_dir / "manifest.json").read_text())
    bundle_root = json.loads((args.run_dir / "resolved_job.json").read_text())["bundle"]
    bundle = PolicyBundle.load(bundle_root)
    arrays = ReferenceArrays(args.reference)
    if not arrays.body_names:
        raise ValueError("reference tree stores no body positions")
    motion = arrays.motion(args.motion)
    reference_body = motion.body_pos_w
    if reference_body is None:
        raise ValueError("reference motion has no body_pos_w")

    results: dict[str, dict] = {}
    for states_path in sorted(args.run_dir.glob("states_ep*.npz")):
        states = np.load(states_path)
        joint_pos = states["joint_pos"]
        anchor = states["anchor_pose_xyzw"]
        ticks = joint_pos.shape[0]
        frames = np.minimum(np.arange(ticks), motion.length - 1)
        robot_body = fk_body_positions(
            args.mjcf,
            bundle.manifest.action,
            joint_pos,
            anchor,
            arrays.body_names,
        )
        aligned_body = _align_reference(
            anchor[0],
            motion.anchor_pos_w[0],
            motion.anchor_quat_w[0],
            reference_body[frames].reshape(-1, 3),
        ).reshape(reference_body[frames].shape)
        aligned_root = _align_reference(
            anchor[0],
            motion.anchor_pos_w[0],
            motion.anchor_quat_w[0],
            motion.anchor_pos_w[frames],
        )
        record = compute_mpjpe(
            robot_body,
            anchor[:, 0:3],
            aligned_body,
            aligned_root,
        )
        record.pop("per_frame_mpjpe_g_mm", None)
        record.pop("per_frame_mpjpe_l_mm", None)
        if args.render:
            from embodied_control.lowlevel.metrics import render_oracle_comparison_video

            video_path = args.run_dir / f"{states_path.stem}_comparison.mp4"
            render_oracle_comparison_video(
                bundle.manifest.action,
                args.mjcf,
                motion,
                {
                    "reference_frames": frames,
                    "joint_position_log": joint_pos,
                    "anchor_pose_log": anchor,
                },
                video_path,
            )
            record["video"] = str(video_path.resolve())
            print(f"video: {video_path.resolve()}")
        results[states_path.stem] = record

    if not results:
        raise FileNotFoundError(f"no states_ep*.npz under {args.run_dir}")
    output = args.run_dir / "mpjpe.json"
    output.write_text(json.dumps({"manifest_sha": manifest.get("sha256"), **results}, indent=2))
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
