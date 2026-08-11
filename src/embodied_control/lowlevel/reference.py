"""Reader for the packed root_qpos_v1 reference-array trees.

Layout (produced by IsaacLab-Imitation's reference-buffer workflow): one
float32 memmap per array plus `reference_arrays_manifest.json` carrying
shapes, joint names (Isaac articulation order), per-trajectory start/end
indices, and quaternion orders. `anchor_quat_w` is already XYZW; `qpos`
carries `[root pos 3 | root quat WXYZ 4 | joints 29]` per frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from embodied_control.lowlevel.maths import wxyz_to_xyzw


@dataclass(frozen=True)
class ReferenceMotion:
    name: str
    joint_qpos: np.ndarray     # [T, 29] Isaac order
    anchor_pos_w: np.ndarray   # [T, 3]
    anchor_quat_w: np.ndarray  # [T, 4] XYZW
    joint_qvel: np.ndarray | None = None  # [T, 29] Isaac order
    body_names: tuple[str, ...] = ()
    body_pos_w: np.ndarray | None = None  # [T, B, 3] tracked-body world positions

    @property
    def length(self) -> int:
        return int(self.joint_qpos.shape[0])


class ReferenceArrays:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        manifest_path = self.root / "reference_arrays_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"not a reference-array tree: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        # The tree is produced by IsaacLab-Imitation's reference-buffer
        # pipeline (ImitationLearningTools side); this reader is the frozen
        # consumer contract, pinned by the format version.
        version = manifest.get("format_version")
        if version != 1:
            raise ValueError(
                f"unsupported reference-array format_version {version!r}; this "
                "reader understands version 1"
            )
        self.manifest = manifest
        key = manifest["key"]
        self.joint_names: list[str] = list(key["joint_names"])
        self.anchor_body: str = key["anchor_body"]
        self.body_names: tuple[str, ...] = tuple(key.get("body_names", ()))
        arrays = key["arrays"]
        self._qpos = self._open(arrays, "qpos")
        self._qvel = self._open(arrays, "qvel") if "qvel" in arrays else None
        self._anchor_pos = self._open(arrays, "anchor_pos_w")
        self._body_pos = (
            self._open(arrays, "body_pos_w") if "body_pos_w" in arrays else None
        )
        anchor_quat = self._open(arrays, "anchor_quat_w")
        order = arrays["anchor_quat_w"].get("quaternion_order")
        if order == "wxyz":
            anchor_quat = wxyz_to_xyzw(anchor_quat)
        elif order != "xyzw":
            raise ValueError(f"anchor_quat_w has unknown quaternion order {order!r}")
        self._anchor_quat = anchor_quat
        info = manifest["traj_info"]
        self._starts = [int(v) for v in info["start_index"]]
        self._ends = [int(v) for v in info["end_index"]]
        self.motion_names = [entry[1] for entry in info["ordered_traj_list"]]
        joint_count = len(self.joint_names)
        if self._qpos.shape[1] != 7 + joint_count:
            raise ValueError(
                f"qpos width {self._qpos.shape[1]} != 7 + {joint_count} joints"
            )
        if self._qvel is not None and self._qvel.shape[1] != 6 + joint_count:
            raise ValueError(
                f"qvel width {self._qvel.shape[1]} != 6 + {joint_count} joints"
            )

    def _open(self, arrays: dict, name: str) -> np.ndarray:
        spec = arrays[name]
        shape = tuple(int(v) for v in spec["shape"])
        return np.memmap(
            self.root / f"{name}.memmap", dtype=spec["dtype"], mode="r", shape=shape
        )

    def motion(self, name_or_index: str | int) -> ReferenceMotion:
        if isinstance(name_or_index, int):
            index = name_or_index
        else:
            try:
                index = self.motion_names.index(name_or_index)
            except ValueError as exc:
                raise KeyError(
                    f"motion {name_or_index!r} not in {self.motion_names}"
                ) from exc
        start, end = self._starts[index], self._ends[index]
        return ReferenceMotion(
            name=self.motion_names[index],
            joint_qpos=np.asarray(self._qpos[start:end, 7:], dtype=np.float32),
            anchor_pos_w=np.asarray(self._anchor_pos[start:end], dtype=np.float32),
            anchor_quat_w=np.asarray(self._anchor_quat[start:end], dtype=np.float32),
            joint_qvel=(
                None
                if self._qvel is None
                else np.asarray(self._qvel[start:end, 6:], dtype=np.float32)
            ),
            body_names=self.body_names,
            body_pos_w=(
                None
                if self._body_pos is None
                else np.asarray(self._body_pos[start:end], dtype=np.float32)
            ),
        )


__all__ = ["ReferenceArrays", "ReferenceMotion"]
