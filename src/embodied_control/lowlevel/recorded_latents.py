"""Recorded-latent playback: reuse a run's latent stream instead of encoding live.

The native runtime records, on every control tick, the command the actor
stepped with (``command_log``, the latent plus phase) and the reference
frame that tick tracked (``reference_frames``). A table keyed by reference
frame rebuilt from a clean plant run can be handed to a later run --
plant or hardware -- which then serves ``z[frame]`` from the table and
never encodes the live window. The policy still closes its loop on the
robot's own proprioception; only the latent is canned.

What that tests: a deployment that needs no position estimate at all (the
latent carries the plant's tracking-error pattern, not the robot's) with
the weights exactly as trained. It sits between the live robot-heading
window (real error, needs localization) and an expert-heading fine-tune
(zero error, needs new weights).

The table is bound to the bundle checkpoint, the motion and the start
frame it was recorded from; loading refuses anything else. Frames the
recording never reached hold the previous latent in the runtime and count
as misses in its stats.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

TABLE_API_VERSION = "ec.recorded_latents/v1"


@dataclass
class LatentTable:
    frames: np.ndarray          # int32 [N] absolute reference frames (sorted, unique)
    latents: np.ndarray         # float32 [N, z_dim]
    bundle_sha256: str
    motion: str
    start_frame: int
    z_dim: int
    source: str = ""

    def dense(self, start_frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Runtime-local table (frame - start_frame) and per-frame valid flags."""
        local = self.frames.astype(np.int64) - int(start_frame)
        keep = local >= 0
        if not keep.any():
            raise ValueError("recorded latents hold no frame at or after the start frame")
        length = int(local[keep].max()) + 1
        table = np.zeros((length, self.z_dim), dtype=np.float32)
        valid = np.zeros(length, dtype=np.uint8)
        table[local[keep]] = self.latents[keep]
        valid[local[keep]] = 1
        return table, valid


def table_from_telemetry(
    telemetry_path: str | Path,
    *,
    bundle_sha256: str,
    motion: str,
    start_frame: int,
    z_dim: int,
    pick: str = "last",
) -> LatentTable:
    """Rebuild the per-frame latent stream a run consumed.

    Several control ticks can share one reference frame (the pinned frame 0
    while arming; a held frame after a late reply). ``pick='last'`` keeps
    the tick closest to the frame advancing, the one taken under running
    conditions; ``'first'`` keeps the earliest.
    """
    data = np.load(telemetry_path)
    for key in ("command_log", "reference_frames"):
        if key not in data:
            raise ValueError(f"{telemetry_path} has no {key}: recorded before the observation record")
    command = np.asarray(data["command_log"], np.float32)
    frames = np.asarray(data["reference_frames"], np.int64)
    if command.ndim != 2 or command.shape[1] < z_dim:
        raise ValueError(f"command_log is {command.shape}, narrower than z_dim {z_dim}")
    if frames.shape[0] != command.shape[0]:
        raise ValueError("reference_frames and command_log disagree on the tick count")
    controlled = np.isfinite(command[:, :z_dim]).all(axis=1) & (frames >= 0)
    if not controlled.any():
        raise ValueError("the recording has no controlled tick with a finite latent")
    order = np.where(controlled)[0]
    per_frame: dict[int, int] = {}
    for tick in order:
        frame = int(frames[tick])
        if pick == "first" and frame in per_frame:
            continue
        per_frame[frame] = int(tick)
    keys = np.array(sorted(per_frame), dtype=np.int32)
    ticks = np.array([per_frame[int(k)] for k in keys], dtype=np.int64)
    return LatentTable(
        frames=keys,
        latents=np.ascontiguousarray(command[ticks, :z_dim]),
        bundle_sha256=str(bundle_sha256),
        motion=str(motion),
        start_frame=int(start_frame),
        z_dim=int(z_dim),
        source=str(telemetry_path),
    )


def save_table(table: LatentTable, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "api_version": TABLE_API_VERSION,
        "bundle_sha256": table.bundle_sha256,
        "motion": table.motion,
        "start_frame": table.start_frame,
        "z_dim": table.z_dim,
        "source": table.source,
        "frames": int(table.frames.shape[0]),
        "frame_min": int(table.frames.min()),
        "frame_max": int(table.frames.max()),
    }
    np.savez(path, frames=table.frames, latents=table.latents, meta=np.asarray(json.dumps(meta)))
    return path


def load_table(path: str | Path) -> LatentTable:
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    if meta.get("api_version") != TABLE_API_VERSION:
        raise ValueError(f"{path}: unsupported recorded-latent table {meta.get('api_version')!r}")
    return LatentTable(
        frames=np.asarray(data["frames"], np.int32),
        latents=np.asarray(data["latents"], np.float32),
        bundle_sha256=str(meta["bundle_sha256"]),
        motion=str(meta["motion"]),
        start_frame=int(meta["start_frame"]),
        z_dim=int(meta["z_dim"]),
        source=str(meta.get("source", "")),
    )


def check_table(table: LatentTable, *, bundle_sha256: str, motion: str, z_dim: int) -> None:
    """Refuse a table recorded from another bundle, motion or latent width."""
    problems = []
    if table.bundle_sha256 != str(bundle_sha256):
        problems.append(f"bundle {table.bundle_sha256[:12]} != {str(bundle_sha256)[:12]}")
    if table.motion != str(motion):
        problems.append(f"motion {table.motion!r} != {motion!r}")
    if table.z_dim != int(z_dim):
        problems.append(f"z_dim {table.z_dim} != {z_dim}")
    if table.latents.shape != (table.frames.shape[0], table.z_dim):
        problems.append("latents shape does not match frames x z_dim")
    if problems:
        raise ValueError("recorded latents do not fit this job: " + "; ".join(problems))


__all__ = [
    "LatentTable", "TABLE_API_VERSION", "check_table", "load_table", "save_table",
    "table_from_telemetry",
]
