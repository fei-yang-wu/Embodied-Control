"""Render a recorded plant run: true root pose, straps, and the lifecycle state.

The DDS wire carries no root pose, so the only ground truth for what the
robot did is the plant's own state log (`plant.states.npz`). This draws it
with the shoulder straps at their recorded attachment points and, when the
run's transitions are supplied, the lifecycle state the robot was in at each
moment, so one video shows hoisted preparation, lowering, playback and the
hoisted recovery end to end.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)
STRAP_COLOR = (0.12, 0.12, 0.12, 1.0)
SPREADER_COLOR = (0.85, 0.15, 0.08, 1.0)


def _font(size: int):
    from PIL import ImageFont

    for path in FONT_PATHS:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def state_at(timeline: list[dict], when: float) -> str:
    """The newest successful transition at wall time `when`, or ''."""
    current = ""
    for entry in timeline:
        if not entry.get("ok", True):
            continue
        if float(entry["wall"]) <= when:
            current = str(entry["to_state"])
        else:
            break
    return current


def render_plant_states(
    states_path: str | Path,
    output: str | Path,
    model_path: str | Path,
    *,
    title: str = "plant ground truth",
    timeline: list[dict] | None = None,
    started_at: float | None = None,
    fps: int = 25,
    width: int = 960,
    height: int = 560,
) -> dict:
    """Write `output` (mp4) and a poster PNG beside it; return a small report."""
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from PIL import Image, ImageDraw

    from embodied_control.lowlevel.plant_view import PlantPuppet

    states_path, output = Path(states_path), Path(output)
    with np.load(states_path) as source:
        if str(source.get("root_pose_source", "unknown")) != "simulator_ground_truth":
            raise ValueError("rendering needs labelled simulator ground truth, not a controller anchor")
        names = source["joint_names"].tolist()
        rows = np.concatenate(
            [source["root_pos"], source["root_quat_xyzw"], source["joint_pos"]], axis=1
        )
        hz = float(source["publish_hz"])
        hoist = None
        if "hoist_hook_pos" in source:
            hoist = {
                key: source[key].copy()
                for key in (
                    "hoist_hook_pos", "hoist_gain", "hoist_tension",
                    "hoist_attachment_points", "hoist_attachment_body",
                )
            }
    if len(rows) == 0 or not np.isfinite(rows).all() or hz <= 0:
        raise ValueError("state recording is empty or invalid")

    puppet = PlantPuppet(model_path, names)
    puppet.model.vis.global_.offwidth = max(puppet.model.vis.global_.offwidth, width)
    puppet.model.vis.global_.offheight = max(puppet.model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(puppet.model, height=height, width=width)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.7
    camera.azimuth = 135
    camera.elevation = -15
    indices = np.rint(np.arange(0, len(rows) / hz, 1 / fps) * hz).astype(int)
    indices = indices[indices < len(rows)]
    output.parent.mkdir(parents=True, exist_ok=True)
    font, small = _font(22), _font(17)
    header = 80
    footer = 40
    timeline = sorted(timeline or [], key=lambda entry: float(entry["wall"]))
    writer = imageio.get_writer(
        output, fps=fps, quality=8, macro_block_size=1,
        ffmpeg_params=["-movflags", "+faststart"],
    )
    poster = None
    states_seen: list[str] = []
    try:
        for index, tick in enumerate(indices):
            row = rows[tick]
            seconds = tick / hz
            puppet.pose(row)
            camera.lookat[:] = [row[0], row[1], 0.75]
            renderer.update_scene(puppet.data, camera=camera)
            if hoist is not None and hoist["hoist_gain"][tick] > 0:
                body = mujoco.mj_name2id(
                    puppet.model, mujoco.mjtObj.mjOBJ_BODY, str(hoist["hoist_attachment_body"])
                )
                rotation = puppet.data.xmat[body].reshape(3, 3)
                attachment = hoist["hoist_attachment_points"] @ rotation.T + puppet.data.xpos[body]
                hooks = hoist["hoist_hook_pos"][tick]
                for start, end, color in (
                    (attachment[0], hooks[0], STRAP_COLOR),
                    (attachment[1], hooks[1], STRAP_COLOR),
                    (hooks[0], hooks[1], SPREADER_COLOR),
                ):
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(
                        geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).ravel(), np.asarray(color, dtype=np.float32),
                    )
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.006, start, end)
                    renderer.scene.ngeom += 1
            frame = Image.new("RGB", (width, height + header + footer), "#f4f6fa")
            frame.paste(Image.fromarray(renderer.render().copy()), (0, header))
            draw = ImageDraw.Draw(frame)
            draw.text((16, 10), title, font=font, fill="#182335")
            line = (
                f"t={seconds:.2f}s   pelvis xyz {row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f} m"
            )
            if hoist is not None:
                gain = float(hoist["hoist_gain"][tick])
                line += "   strap " + ("hoisted" if gain >= 0.99 else "slack" if gain <= 0 else f"gain {gain:.2f}")
            draw.text((16, 44), line, font=small, fill="#182335")
            if timeline and started_at is not None:
                state = state_at(timeline, started_at + seconds)
                if state and (not states_seen or states_seen[-1] != state):
                    states_seen.append(state)
                if state:
                    draw.rectangle((width - 300, 10, width - 16, 44), fill="#182335")
                    draw.text((width - 290, 16), state, font=font, fill="#ffffff")
            draw.text(
                (16, height + header + 10),
                "true root from the plant; camera follows the pelvis; straps drawn at their recorded attachments",
                font=small, fill="#182335",
            )
            writer.append_data(np.asarray(frame))
            if index == len(indices) * 2 // 3:
                poster = frame.copy()
    finally:
        writer.close()
        renderer.close()
    if poster is not None:
        poster.save(output.with_suffix(".png"))
    reader = imageio.get_reader(output)
    count = reader.count_frames()
    reader.close()
    if count != len(indices):
        raise RuntimeError("encoded video has incomplete frame coverage")
    report = {
        "source": str(states_path),
        "root_pose_source": "simulator_ground_truth",
        "video": str(output),
        "frames": int(count),
        "fps": fps,
        "duration_seconds": count / fps,
        "root_min_xyz": rows[:, :3].min(axis=0).tolist(),
        "root_max_xyz": rows[:, :3].max(axis=0).tolist(),
        "states_seen": states_seen,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
