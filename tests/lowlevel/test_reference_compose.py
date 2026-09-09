import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('mujoco')

from embodied_control.lowlevel.reference import ReferenceArrays
from embodied_control.lowlevel.reference_compose import compose_reference


def test_composition_preserves_motion_and_has_stationary_encoder_windows(tmp_path):
    model = tmp_path / 'robot.xml'
    model.write_text('''<mujoco><worldbody><body name="pelvis"><freejoint/>
    <geom type="sphere" size=".1" mass="1"/>
    <body name="left_ankle_roll_link" pos="0 .1 -.7"><joint name="left"/>
    <geom type="sphere" size=".05" mass="1"/></body>
    <body name="right_ankle_roll_link" pos="0 -.1 -.7"><joint name="right"/>
    <geom type="sphere" size=".05" mass="1"/></body>
    </body></worldbody></mujoco>''')
    source = tmp_path / 'source'
    source.mkdir()
    qpos = np.tile([0., 0., .79, 1., 0., 0., 0., .1, .1], (10, 1)).astype(np.float32)
    qpos[:, 7] += np.linspace(0, .05, 10)
    arrays = {'qpos': qpos, 'qvel': np.zeros((10, 8), np.float32),
              'anchor_pos_w': qpos[:, :3], 'anchor_quat_w': qpos[:, [4, 5, 6, 3]]}
    manifest = {'format_version': 1, 'key': {'joint_names': ['left', 'right'],
                'body_names': ['left_ankle_roll_link', 'right_ankle_roll_link'], 'anchor_body': 'pelvis',
                'arrays': {key: {'shape': list(a.shape), 'dtype': 'float32',
                                 'quaternion_order': 'xyzw' if key == 'anchor_quat_w' else None}
                           for key, a in arrays.items()}},
                'traj_info': {'start_index': [0], 'end_index': [10], 'ordered_traj_list': [['bones', 'wave']]}}
    for key, array in arrays.items():
        array.tofile(source / f'{key}.memmap')
    (source / 'reference_arrays_manifest.json').write_text(json.dumps(manifest))
    bundle = SimpleNamespace(manifest=SimpleNamespace(
        action=SimpleNamespace(isaac_joint_names=['left', 'right'], default_joint_pos=[.1, .1]),
        command=SimpleNamespace(horizon_steps=3, macro_frame_stride=1)))
    name = compose_reference(source, 'wave', bundle, model, tmp_path / 'derived')
    ref = ReferenceArrays(tmp_path / 'derived')
    motion = ref.motion(name)
    assert motion.length == 510
    np.testing.assert_allclose(motion.joint_qpos[250:260], qpos[:, 7:])
    np.testing.assert_allclose(motion.joint_qpos[:100], .1)
    np.testing.assert_allclose(motion.joint_qpos[-100:], .1)
    np.testing.assert_allclose(motion.joint_qvel[:100], 0, atol=1e-6)
    np.testing.assert_allclose(motion.joint_qvel[-100:], 0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(motion.anchor_quat_w, axis=1), 1)
    assert ref.manifest['motions'][name]['requires_rehearsal']
    assert name == compose_reference(source, 'wave', bundle, model, tmp_path / 'repeat')
    with pytest.raises(FileExistsError):
        compose_reference(source, 'wave', bundle, model, tmp_path / 'derived')
    with pytest.raises(ValueError, match='encoder'):
        compose_reference(source, 'wave', bundle, model, tmp_path / 'short', hold_seconds=.02)
