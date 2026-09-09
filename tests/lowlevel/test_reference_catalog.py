import json
from dataclasses import replace

import numpy as np
import pytest

from embodied_control.lowlevel.reference import ReferenceArrays, ReferenceMotion
from embodied_control.lowlevel.reference_catalog import merge_reference_trees, motion_sha256
from embodied_control.lowlevel.reference_deploy import classify_motion
from test_maths_and_reference import _write_reference_tree


def standing_motion():
    return ReferenceMotion(
        'stand', np.zeros((5, 29), np.float32),
        np.tile([0., 0., .79], (5, 1)), np.tile([0., 0., 0., 1.], (5, 1)),
        np.zeros((5, 29), np.float32),
        ('left_ankle_roll_link', 'right_ankle_roll_link'),
        np.zeros((5, 2, 3), np.float32),
    )


def test_endpoint_screening_fails_closed():
    motion = standing_motion()
    assert classify_motion(motion).deployable
    assert not classify_motion(replace(motion, body_pos_w=None)).deployable
    assert not classify_motion(replace(motion, joint_qvel=None)).deployable
    motion.joint_qpos[0, 0] = np.nan
    assert not classify_motion(motion).deployable


def test_endpoint_speed_uses_sample_rate():
    motion = standing_motion()
    motion.anchor_pos_w[:, 0] = np.arange(5) * .002
    assert classify_motion(motion, fps=50).deployable
    assert not classify_motion(motion, fps=100).deployable


def test_merge_preserves_every_array_and_content_identity(tmp_path):
    sources = [tmp_path / 'first', tmp_path / 'second']
    for idx, path in enumerate(sources):
        path.mkdir()
        _write_reference_tree(path)
        manifest_path = path / 'reference_arrays_manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for entry in manifest['traj_info']['ordered_traj_list']:
            entry[1] += str(idx)
        manifest_path.write_text(json.dumps(manifest))
    target = tmp_path / 'bones'
    merged = merge_reference_trees(sources, target)
    reference = ReferenceArrays(target)
    assert len(reference.motion_names) == 4
    for source in sources:
        original = ReferenceArrays(source)
        for name in original.motion_names:
            assert motion_sha256(original, name) == motion_sha256(reference, name)
            np.testing.assert_array_equal(original.motion(name).joint_qpos, reference.motion(name).joint_qpos)
            assert merged['motions'][name]['source_tree'] == source.name
    with pytest.raises(FileExistsError):
        merge_reference_trees(sources, target)


def test_duplicate_names_are_refused_without_partial_output(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    _write_reference_tree(source)
    with pytest.raises(ValueError, match='duplicate motion'):
        merge_reference_trees([source, source], tmp_path / 'merged')
    assert not (tmp_path / 'merged').exists()


def test_reference_contract_does_not_guess_training_distribution(tmp_path):
    from types import SimpleNamespace
    from embodied_control.lowlevel.reference_catalog import reference_compatibility
    _write_reference_tree(tmp_path)
    reference = ReferenceArrays(tmp_path)
    command = SimpleNamespace(encoder_state_interface='root_qpos', state_dim=38,
                              macro_anchor_mode='robot_heading', macro_frame_stride=1)
    bundle = SimpleNamespace(manifest=SimpleNamespace(command=command,
        action=SimpleNamespace(isaac_joint_names=reference.joint_names)))
    assert reference_compatibility(bundle, reference)['training_distribution'] == 'unknown'
    command.encoder_state_interface = 'joint_qpos_qvel_anchor_ori'
    command.state_dim = 64
    with pytest.raises(ValueError, match='qvel'):
        reference_compatibility(bundle, reference)
