"""Build a shared motion namespace without discarding source provenance."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from embodied_control.lowlevel.reference import ReferenceArrays
from embodied_control.lowlevel.reference_deploy import DEFAULT_LIMITS, classify_motion


def motion_sha256(reference: ReferenceArrays, name: str) -> str:
    index = reference.motion_names.index(name)
    start, end = reference._starts[index], reference._ends[index]
    digest = hashlib.sha256()
    contract = {key: reference.manifest['key'].get(key) for key in
                ('joint_names', 'body_names', 'dataset_body_names', 'anchor_body')}
    contract['fps'] = reference.fps
    digest.update(json.dumps(contract, sort_keys=True).encode())
    for key, spec in sorted(reference.manifest['key']['arrays'].items()):
        array = reference._open(reference.manifest['key']['arrays'], key)[start:end]
        digest.update(json.dumps([key, spec['dtype'], spec.get('quaternion_order'), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def merge_reference_trees(sources: list[str | Path], destination: str | Path) -> dict:
    """Publish a new tree atomically; refuse collisions and incompatible layouts."""
    if not sources:
        raise ValueError('at least one reference tree is required')
    refs = [ReferenceArrays(source) for source in sources]
    target = Path(destination)
    if target.exists():
        raise FileExistsError(target)
    key = deepcopy(refs[0].manifest['key'])
    specs = key['arrays']
    entries = []
    names = set()
    for ref in refs:
        for field in ('joint_names', 'body_names', 'dataset_body_names', 'anchor_body'):
            if ref.manifest['key'].get(field) != key.get(field):
                raise ValueError(f'incompatible {field} in {ref.root}')
        other = ref.manifest['key']['arrays']
        if set(other) != set(specs) or ref.fps != refs[0].fps:
            raise ValueError(f'incompatible arrays or fps in {ref.root}')
        for name, spec in specs.items():
            if any(other[name].get(k) != spec.get(k) for k in ('dtype', 'quaternion_order')) or other[name]['shape'][1:] != spec['shape'][1:]:
                raise ValueError(f'incompatible array {name} in {ref.root}')
        for name in ref.motion_names:
            if name in names:
                raise ValueError(f'duplicate motion {name!r}; resolve source variants explicitly')
            names.add(name)
            entries.append((ref, name))
    lengths = [ref.motion(name).length for ref, name in entries]
    total = sum(lengths)
    starts = np.cumsum([0] + lengths[:-1]).tolist()
    ends = np.cumsum(lengths).tolist()
    key['source'] = {'dataset_id': 'bones', 'sources': [ref.manifest['key'].get('source') for ref in refs]}
    key.pop('manifest', None)
    key.pop('manifest_sha256', None)
    manifest = {'format_version': 1, 'fps': refs[0].fps, 'key': key,
                'traj_info': {'capacity': total, 'written': total, 'start_index': starts,
                              'end_index': ends, 'ordered_traj_list': [['bones', name] for _, name in entries]},
                'deployment_limits': asdict(DEFAULT_LIMITS), 'motions': {}}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.reference-', dir=target.parent) as temporary:
        staging = Path(temporary) / 'tree'
        staging.mkdir()
        for name, spec in specs.items():
            spec['shape'][0] = total
            out = np.memmap(staging / f'{name}.memmap', dtype=spec['dtype'], mode='w+', shape=tuple(spec['shape']))
            for (ref, motion), start, end in zip(entries, starts, ends):
                idx = ref.motion_names.index(motion)
                array = ref._open(ref.manifest['key']['arrays'], name)
                out[start:end] = array[ref._starts[idx]:ref._ends[idx]]
            out.flush()
            del out
        for ref, name in entries:
            verdict = classify_motion(ref.motion(name), fps=ref.fps)
            manifest['motions'][name] = {
                'dataset_id': 'bones', 'source_tree': ref.root.name,
                'source': ref.manifest['key'].get('source'),
                'source_pin': json.loads((ref.root / 'model.pin.json').read_text()) if (ref.root / 'model.pin.json').exists() else None,
                'sha256': motion_sha256(ref, name), 'frames': ref.motion(name).length,
                'fps': ref.fps, 'mirrored': name.endswith('_M'),
                'root_stationary': verdict.stats.get('max_displacement_m', float('inf')) <= .05,
                **verdict.to_json(),
            }
        (staging / 'reference_arrays_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (staging / 'README.md').write_text(
            '# BONES reference motions\n\n'
            'Unified motion namespace, preserving source variants and provenance.\n'
            '`deployable` means endpoint screening passed; it is not evidence of balance, '
            'tracking success, or hardware safety. Rehearse the selected checkpoint and exact '
            'reference before hardware use. Training motions remain available for simulation.\n'
        )
        staging.rename(target)
    return manifest


def reference_compatibility(bundle, reference: ReferenceArrays) -> dict:
    """Check the consumer contract without guessing the training dataset."""
    if reference.joint_names != list(bundle.manifest.action.isaac_joint_names):
        raise ValueError('reference joint order does not match the selected bundle')
    if reference.fps != 50.0:
        raise ValueError('the 50 Hz tracker requires a 50 Hz reference')
    command = bundle.manifest.command
    interface = command.encoder_state_interface
    expected = {'root_qpos': (38, {'robot', 'robot_heading'}),
                'joint_qpos_qvel_anchor_ori': (64, {'robot_heading'})}
    if interface not in expected:
        raise ValueError(f'unsupported reference encoder interface {interface!r}')
    width, anchors = expected[interface]
    if command.state_dim != width or command.macro_anchor_mode not in anchors or not command.macro_frame_stride:
        raise ValueError('reference encoder interface, width, anchor and stride disagree')
    required = ['qpos', 'anchor_pos_w', 'anchor_quat_w']
    if interface == 'joint_qpos_qvel_anchor_ori':
        required.append('qvel')
    missing = set(required) - set(reference.manifest['key']['arrays'])
    if missing:
        raise ValueError(f'reference missing required arrays: {sorted(missing)}')
    return {'compatible': True, 'fps': reference.fps, 'required_arrays': required,
            'encoder_state_interface': interface, 'macro_frame_stride': command.macro_frame_stride,
            'macro_anchor_mode': command.macro_anchor_mode, 'training_distribution': 'unknown',
            'distribution_note': 'Exported bundles do not record trained reference dataset IDs.'}
