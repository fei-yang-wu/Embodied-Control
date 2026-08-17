"""Latent playground: encode z, perturb it, roll it out in MuJoCo, render it.

This is a research surface for asking what a DiffSR latent (`z256`) actually
means to a frozen tracker. It reuses the exported policy bundle unchanged —
same TorchScript policy, same encoder, same observation contract, same hold
schedule — and only replaces the source of z.

Scope and honesty notes, kept here because every number this module produces
inherits them:

- The plant is MuJoCo, not the Newton/PhysX training simulator. MuJoCo
  actuator dynamics differ (the 2026-08-03 sim2sim verdict), so a rollout here
  is a deployment-style signal, not a paper metric.
- Each `rollout()` is one deterministic episode with no domain randomization
  and no pushes. One episode is a single sample; call it preliminary and
  repeat before you believe a difference.
- `encode_motion()` encodes offline under a perfect-tracking assumption: the
  live robot anchor is replaced by the reference's own anchor at the cursor.
  During a closed-loop rollout the same encoder sees the real robot anchor, so
  the oracle z of a drifting robot differs from the offline bank.
- An FSQ bundle (`command.quantizer == "fsq"`) is supported: its embedded
  encoder already emits lattice values, a perturbation is free to move z off
  the lattice, and the tracker snaps back at consume time (the SONIC
  convention). `z_trace` records what was *published*; use `snap()` to see
  what the tracker actually consumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.command_buffer import InProcessCommandBuffer
from embodied_control.lowlevel.contracts import RobotState
from embodied_control.lowlevel.publishers.latent_perturbation import (
    LatentPublisher,
    LatentSource,
    ReferenceEncoderSource,
)
from embodied_control.lowlevel.reference import ReferenceArrays, ReferenceMotion
from embodied_control.lowlevel.tracker import BufferedCommandSource, LowLevelTracker

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


def snap_fsq(z: np.ndarray, half_levels: np.ndarray) -> np.ndarray:
    """Snap `[..., z_dim]` onto the FSQ lattice.

    Must stay identical to `LowLevelTracker._snap_fsq`, which applies this at
    consume time; `tests/lowlevel/test_latent_playground.py` asserts parity.
    """
    z = np.asarray(z, dtype=np.float32)
    half = np.asarray(half_levels, dtype=np.float32)
    return np.clip(np.rint(z * half), -half, half - 1.0) / half


def fsq_codes(z: np.ndarray, half_levels: np.ndarray) -> np.ndarray:
    """Integer lattice code per dimension, each in `[-half, half-1]`."""
    half = np.asarray(half_levels, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    return np.clip(np.rint(z * half), -half, half - 1.0).astype(np.int32)


@dataclass(frozen=True)
class ZBank:
    """Offline-encoded latents for one motion."""

    motion: str
    z: np.ndarray        # [K, z_dim]
    cursors: np.ndarray  # [K] reference frame each z was encoded at

    def __len__(self) -> int:
        return int(self.z.shape[0])


@dataclass
class LatentRollout:
    """One closed-loop episode driven by a latent source."""

    steps: int
    status: str
    joint_pos: np.ndarray            # [T, 29] Isaac order
    anchor_pose_xyzw: np.ndarray     # [T, 7]
    base_height: np.ndarray          # [T]
    reference_frame: np.ndarray      # [T] cursor per tick, -1 without a reference
    z_trace: np.ndarray              # [renewals, z_dim] what the tracker consumed
    action: np.ndarray               # [T, 29] raw policy action
    frames: list[np.ndarray] = field(default_factory=list)
    video_fps: float = 25.0
    motion: str | None = None

    @property
    def survived(self) -> bool:
        return self.status == "completed"

    @property
    def min_base_height(self) -> float:
        return float(self.base_height.min()) if self.base_height.size else float("nan")


class LatentPlayground:
    """Load one bundle once, then run many latent experiments against it."""

    def __init__(
        self,
        bundle_dir: str | Path,
        model_path: str | Path,
        reference_root: str | Path | None = None,
        *,
        device: str = "cpu",
    ):
        from embodied_control.lowlevel.engine.torch_engine import TorchEngine

        self.bundle = PolicyBundle.load(bundle_dir)
        manifest = self.bundle.manifest
        if manifest.interface != "latent":
            raise ValueError(
                f"the playground needs a latent bundle, got {manifest.interface!r}"
            )
        self.fsq_half: np.ndarray | None = None
        if manifest.command.quantizer == "fsq":
            self.fsq_half = np.asarray(
                manifest.command.fsq_half_levels, dtype=np.float32
            )
        self.model_path = Path(model_path)
        self.policy = TorchEngine(self.bundle.policy_path, device=device)
        self.policy.warmup(input_width=manifest.obs.total_width)
        if not self.bundle.encoder_path.is_file():
            raise FileNotFoundError(f"bundle has no encoder.pt: {self.bundle.root}")
        self.encoder = TorchEngine(self.bundle.encoder_path, device=device)
        self.encoder.warmup(
            input_width=int(manifest.command.state_dim)
            * (int(manifest.command.window_steps) + 1)
        )
        self.arrays: ReferenceArrays | None = None
        if reference_root is not None:
            self.arrays = ReferenceArrays(reference_root)
            if self.arrays.joint_names != manifest.action.isaac_joint_names:
                raise ValueError(
                    "reference-array joint order does not match the action contract"
                )

    # ---------------------------------------------------------------- basics

    @property
    def command(self):
        return self.bundle.manifest.command

    @property
    def z_dim(self) -> int:
        return int(self.command.z_dim)

    @property
    def hold_steps(self) -> int:
        return int(self.command.hold_steps)

    @property
    def quantized(self) -> bool:
        return self.fsq_half is not None

    def snap(self, z: np.ndarray) -> np.ndarray:
        """What the tracker consumes for a published z (FSQ bundles only)."""
        if self.fsq_half is None:
            raise ValueError("this bundle has no quantizer; snap() is FSQ-only")
        return snap_fsq(z, self.fsq_half)

    def codes(self, z: np.ndarray) -> np.ndarray:
        """Integer lattice codes for a z (FSQ bundles only)."""
        if self.fsq_half is None:
            raise ValueError("this bundle has no quantizer; codes() is FSQ-only")
        return fsq_codes(z, self.fsq_half)

    @property
    def control_hz(self) -> int:
        return int(self.bundle.manifest.rates.control_hz)

    @property
    def motion_names(self) -> list[str]:
        if self.arrays is None:
            raise ValueError("the playground was built without a reference tree")
        return list(self.arrays.motion_names)

    def motion(self, name_or_index: str | int) -> ReferenceMotion:
        if self.arrays is None:
            raise ValueError("the playground was built without a reference tree")
        return self.arrays.motion(name_or_index)

    # --------------------------------------------------------------- encoding

    def encode_frame(
        self,
        motion: ReferenceMotion,
        cursor: int,
        *,
        anchor_pos_w: np.ndarray | None = None,
        anchor_quat_w: np.ndarray | None = None,
    ) -> np.ndarray:
        """Encode the macro window starting at `cursor` into one z.

        Without an explicit anchor the reference's own anchor at `cursor` is
        used, which is the perfect-tracking case.
        """
        source = ReferenceEncoderSource(self.encoder, self.command, motion=motion)
        source.cursor = int(np.clip(cursor, 0, motion.length - 1))
        pos = motion.anchor_pos_w[source.cursor] if anchor_pos_w is None else anchor_pos_w
        quat = (
            motion.anchor_quat_w[source.cursor]
            if anchor_quat_w is None
            else anchor_quat_w
        )
        width = self.bundle.manifest.action.width
        state = RobotState(
            stamp=0.0,
            joint_pos=np.zeros(width, dtype=np.float32),
            joint_vel=np.zeros(width, dtype=np.float32),
            projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
            base_ang_vel=np.zeros(3, dtype=np.float32),
            anchor_pos_w=np.asarray(pos, dtype=np.float32),
            anchor_quat_w=np.asarray(quat, dtype=np.float32),
        )
        return source.z(0, state)

    def encode_motion(
        self,
        name_or_motion: str | int | ReferenceMotion,
        *,
        stride: int | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> ZBank:
        """Encode a motion into a bank of latents, one per `stride` frames.

        The default stride is the bundle's hold, so the bank is exactly the
        sequence of latents a perfectly tracking rollout would consume.
        """
        motion = (
            name_or_motion
            if isinstance(name_or_motion, ReferenceMotion)
            else self.motion(name_or_motion)
        )
        step = int(stride if stride is not None else self.hold_steps)
        last = motion.length - 1 if end is None else int(end)
        cursors = np.arange(int(start), max(int(start) + 1, last), step, dtype=np.int64)
        latents = np.stack([self.encode_frame(motion, int(c)) for c in cursors])
        return ZBank(motion=motion.name, z=latents.astype(np.float32), cursors=cursors)

    # --------------------------------------------------------------- rollouts

    def make_reference_source(
        self, motion: ReferenceMotion, *, start_frame: int = 0
    ) -> ReferenceEncoderSource:
        return ReferenceEncoderSource(
            self.encoder, self.command, motion=motion, start_frame=start_frame
        )

    def rollout(
        self,
        source: LatentSource,
        *,
        motion: ReferenceMotion | None = None,
        max_steps: int = 500,
        hold_steps: int | None = None,
        max_renewals: int | None = None,
        start_pose_frame: int | None = 0,
        fall_height_m: float = 0.4,
        record_video: bool = True,
        video_stride: int = 2,
        video_size: tuple[int, int] = (480, 368),
        camera_distance: float = 2.5,
        camera_azimuth: float = 135.0,
        camera_elevation: float = -15.0,
        follow_camera: bool = True,
    ) -> LatentRollout:
        """Run one episode with `source` driving the latent command.

        `motion` is only needed to place the robot on the reference start pose
        (`start_pose_frame`); pass `start_pose_frame=None` to start from the
        bundle's default stance instead, which is what a reference-free
        experiment (constant z, interpolation) usually wants.
        """
        import mujoco

        from embodied_control.lowlevel.envs.mujoco import MujocoBackend

        manifest = self.bundle.manifest
        backend = MujocoBackend(
            manifest.action,
            self.model_path,
            control_hz=self.control_hz,
            timestep=manifest.rates.physics_dt,
            decimation=manifest.rates.decimation,
        )
        backend.reset()
        if motion is not None and start_pose_frame is not None:
            self._place_on_reference(backend, motion, int(start_pose_frame))

        buffer = InProcessCommandBuffer()
        publisher = LatentPublisher(
            buffer,
            source,
            self.command,
            hold_steps=hold_steps,
            max_renewals=max_renewals,
        )
        tracker = LowLevelTracker(
            self.bundle,
            self.policy,
            BufferedCommandSource(buffer, control_hz=self.control_hz),
        )
        tracker.reset(backend.read_state())

        renderer = None
        camera = None
        if record_video:
            renderer = mujoco.Renderer(
                backend.model, height=int(video_size[1]), width=int(video_size[0])
            )
            camera = mujoco.MjvCamera()
            camera.distance = float(camera_distance)
            camera.azimuth = float(camera_azimuth)
            camera.elevation = float(camera_elevation)

        joint_log: list[np.ndarray] = []
        anchor_log: list[np.ndarray] = []
        height_log: list[float] = []
        frame_log: list[int] = []
        action_log: list[np.ndarray] = []
        frames: list[np.ndarray] = []
        status = "completed"
        steps = 0
        try:
            for step in range(int(max_steps)):
                state = backend.read_state()
                joint_log.append(np.asarray(state.joint_pos, dtype=np.float32).copy())
                anchor_log.append(
                    np.concatenate([state.anchor_pos_w, state.anchor_quat_w]).astype(
                        np.float32
                    )
                )
                height_log.append(backend.base_height)
                frame_log.append(int(getattr(source, "cursor", -1)))
                publisher.tick(step, backend.now, state)
                joint_command, _ = tracker.step(step, state)
                if joint_command is None:
                    status = "no_command"
                    break
                action_log.append(tracker.last_action.copy())
                backend.write_command(joint_command)
                steps = step + 1
                if renderer is not None and step % max(1, int(video_stride)) == 0:
                    if follow_camera:
                        camera.lookat[:] = backend.data.qpos[0:3]
                    renderer.update_scene(backend.data, camera=camera)
                    frames.append(renderer.render().copy())
                if backend.base_height < float(fall_height_m):
                    status = "fell"
                    break
                if publisher.exhausted:
                    status = "reference_finished"
                    break
        finally:
            if renderer is not None:
                renderer.close()

        fps = self.control_hz / max(1, int(video_stride))
        return LatentRollout(
            steps=steps,
            status=status,
            joint_pos=np.stack(joint_log) if joint_log else np.empty((0, 29)),
            anchor_pose_xyzw=np.stack(anchor_log) if anchor_log else np.empty((0, 7)),
            base_height=np.asarray(height_log, dtype=np.float32),
            reference_frame=np.asarray(frame_log, dtype=np.int64),
            z_trace=(
                np.stack(publisher.z_trace)
                if publisher.z_trace
                else np.empty((0, self.z_dim), dtype=np.float32)
            ),
            action=np.stack(action_log) if action_log else np.empty((0, 29)),
            frames=frames,
            video_fps=float(fps),
            motion=None if motion is None else motion.name,
        )

    def _place_on_reference(
        self, backend, motion: ReferenceMotion, frame: int
    ) -> None:
        frame = int(np.clip(frame, 0, motion.length - 1))
        backend.set_pose(
            np.concatenate([motion.anchor_pos_w[frame], motion.anchor_quat_w[frame]]),
            motion.joint_qpos[frame],
        )

    # ---------------------------------------------------------------- scoring

    def tracking_error(
        self, rollout: LatentRollout, motion: ReferenceMotion
    ) -> dict:
        """MPJPE-L/G against the reference frames the rollout was aligned to.

        Only meaningful for a reference-driven rollout — a constant-z or
        interpolated rollout has no per-tick reference to compare against, and
        this raises instead of inventing an alignment.
        """
        from embodied_control.lowlevel.metrics import compute_mpjpe, fk_body_positions

        frames = rollout.reference_frame
        if frames.size == 0 or (frames < 0).any():
            raise ValueError(
                "this rollout has no per-tick reference frame; tracking error is "
                "undefined for reference-free latent sources"
            )
        if motion.body_pos_w is None or not motion.body_names:
            raise ValueError(
                "the reference tree carries no body_pos_w; MPJPE needs tracked bodies"
            )
        bodies = fk_body_positions(
            self.model_path,
            self.bundle.manifest.action,
            rollout.joint_pos,
            rollout.anchor_pose_xyzw,
            motion.body_names,
        )
        result = compute_mpjpe(
            bodies,
            rollout.anchor_pose_xyzw[:, 0:3],
            motion.body_pos_w[frames],
            motion.anchor_pos_w[frames],
        )
        result.pop("per_frame_mpjpe_g_mm")
        result.pop("per_frame_mpjpe_l_mm")
        result["joint_mae_rad"] = float(
            np.abs(rollout.joint_pos - motion.joint_qpos[frames]).mean()
        )
        return result

    def summary(
        self, rollout: LatentRollout, motion: ReferenceMotion | None = None
    ) -> dict:
        row = {
            "motion": rollout.motion,
            "steps": rollout.steps,
            "status": rollout.status,
            "survived": rollout.survived,
            "min_base_height_m": round(rollout.min_base_height, 3),
            "renewals": int(rollout.z_trace.shape[0]),
        }
        if (
            motion is not None
            and rollout.reference_frame.size > 0
            and (rollout.reference_frame >= 0).all()
        ):
            error = self.tracking_error(rollout, motion)
            row["mpjpe_l_mm"] = round(error["mpjpe_l_mm"], 1)
            row["mpjpe_g_mm"] = round(error["mpjpe_g_mm"], 1)
            row["joint_mae_rad"] = round(error["joint_mae_rad"], 4)
        return row


def save_video(rollout: LatentRollout, path: str | Path, *, fps: float | None = None) -> Path:
    """Write a rollout's rendered frames to mp4."""
    import imageio.v2 as imageio

    if not rollout.frames:
        raise ValueError("this rollout has no frames; run it with record_video=True")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path, rollout.frames, fps=float(fps or rollout.video_fps), quality=8
    )
    return path


def save_grid_video(
    rollouts: dict[str, LatentRollout],
    path: str | Path,
    *,
    columns: int = 3,
    fps: float | None = None,
) -> Path:
    """Tile several rollouts into one mp4, padded to the longest one."""
    import imageio.v2 as imageio

    items = [(label, roll) for label, roll in rollouts.items() if roll.frames]
    if not items:
        raise ValueError("no rollout in this set has frames")
    height, width = items[0][1].frames[0].shape[:2]
    length = max(len(roll.frames) for _, roll in items)
    columns = max(1, int(columns))
    rows = (len(items) + columns - 1) // columns
    canvas_frames = []
    for index in range(length):
        tiles = []
        for _, roll in items:
            frame = roll.frames[min(index, len(roll.frames) - 1)]
            tiles.append(frame)
        while len(tiles) < rows * columns:
            tiles.append(np.zeros((height, width, 3), dtype=np.uint8))
        grid = np.concatenate(
            [
                np.concatenate(tiles[row * columns : (row + 1) * columns], axis=1)
                for row in range(rows)
            ],
            axis=0,
        )
        canvas_frames.append(grid)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        path, canvas_frames, fps=float(fps or items[0][1].video_fps), quality=8
    )
    return path


def video_html(path: str | Path, *, width: int = 480) -> "object":
    """Inline an mp4 in a notebook cell (base64, so it survives export)."""
    import base64

    from IPython.display import HTML

    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return HTML(
        f'<video width="{int(width)}" controls loop autoplay muted '
        f'src="data:video/mp4;base64,{data}"></video>'
    )


__all__ = [
    "LatentPlayground",
    "LatentRollout",
    "ZBank",
    "fsq_codes",
    "save_grid_video",
    "save_video",
    "snap_fsq",
    "video_html",
]
