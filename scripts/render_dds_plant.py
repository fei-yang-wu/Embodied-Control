"""Render recorded DDS plant truth without substituting controller root estimates."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MUJOCO_GL', 'egl')
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('states', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', type=Path, default=Path('assets/latent_playkit/model/g1_29dof_rev_1_0.xml'))
    parser.add_argument('--title', default='DDS plant ground truth')
    args = parser.parse_args()
    import imageio.v2 as imageio
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from embodied_control.lowlevel.plant_view import PlantPuppet

    with np.load(args.states) as source:
        if str(source.get('root_pose_source', 'unknown')) != 'simulator_ground_truth':
            raise ValueError('Rendering requires labelled simulator ground truth, not a controller anchor')
        names = source['joint_names'].tolist()
        rows = np.concatenate([source['root_pos'], source['root_quat_xyzw'], source['joint_pos']], axis=1)
        hz = float(source['publish_hz'])
        hoist = None
        if 'hoist_hook_pos' in source:
            hoist = {key: source[key].copy() for key in (
                'hoist_hook_pos', 'hoist_gain', 'hoist_tension',
                'hoist_attachment_points', 'hoist_attachment_body')}

    if len(rows) == 0 or not np.isfinite(rows).all() or hz <= 0:
        raise ValueError('State recording is empty or invalid')
    puppet = PlantPuppet(args.model, names)
    puppet.model.vis.global_.offwidth = 960
    puppet.model.vis.global_.offheight = 560
    renderer = mujoco.Renderer(puppet.model, height=560, width=960)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.7
    camera.azimuth = 135
    camera.elevation = -15
    fps = 25
    indices = np.rint(np.arange(0, len(rows)/hz, 1/fps)*hz).astype(int)
    indices = indices[indices < len(rows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 22)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    writer = imageio.get_writer(args.output, fps=fps, quality=8, macro_block_size=1, ffmpeg_params=['-movflags', '+faststart'])
    try:
        for index, tick in enumerate(indices):
            row = rows[tick]
            puppet.pose(row)
            camera.lookat[:] = [row[0], row[1], .75]
            renderer.update_scene(puppet.data, camera=camera)
            if hoist is not None and hoist['hoist_gain'][tick] > 0:
                body = mujoco.mj_name2id(puppet.model, mujoco.mjtObj.mjOBJ_BODY, str(hoist['hoist_attachment_body']))
                attachment = hoist['hoist_attachment_points'] @ puppet.data.xmat[body].reshape(3, 3).T + puppet.data.xpos[body]
                hooks = hoist['hoist_hook_pos'][tick]
                for start, end, color in [(attachment[0], hooks[0], [.12,.12,.12,1]),
                                           (attachment[1], hooks[1], [.12,.12,.12,1]),
                                           (hooks[0], hooks[1], [.85,.15,.08,1])]:
                    geom = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).ravel(), np.asarray(color,dtype=np.float32))
                    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .006, start, end)
                    renderer.scene.ngeom += 1
            frame = Image.new('RGB', (960,680), '#f4f6fa')
            frame.paste(Image.fromarray(renderer.render().copy()), (0,80))
            draw = ImageDraw.Draw(frame)
            draw.text((16,10), args.title+' | true root position + orientation', font=font, fill='#182335')
            draw.text((16,44), f'Simulator t={tick/hz:.2f}s | pelvis xyz = {row[0]:.2f}, {row[1]:.2f}, {row[2]:.2f} m', font=small, fill='#182335')
            draw.text((16,648), 'Full run: hoisted preparation, playback, recovery. Camera follows horizontal root motion.', font=small, fill='#182335')
            writer.append_data(np.asarray(frame))
            if index == len(indices)*2//3:
                frame.save(args.output.with_suffix('.png'))
    finally:
        writer.close()
        renderer.close()
    reader = imageio.get_reader(args.output)
    count = reader.count_frames()
    reader.get_data(count-1)
    reader.close()
    if count != len(indices):
        raise RuntimeError('Encoded video has incomplete frame coverage')
    report = {'source': str(args.states), 'root_pose_source': 'simulator_ground_truth',
              'video': str(args.output), 'frames': count, 'fps': fps, 'duration_seconds':count/fps,
              'root_min_xyz': rows[:,:3].min(axis=0).tolist(), 'root_max_xyz':rows[:,:3].max(axis=0).tolist()}
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
