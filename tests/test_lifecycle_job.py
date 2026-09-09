import json

import pytest
from pydantic import ValidationError

from embodied_control.robot.lifecycle_job import LifecycleJob, load_lifecycle_job


def test_relative_reference_is_validated_after_path_resolution(tmp_path):
    reference = tmp_path / 'reference'
    reference.mkdir()
    (reference / 'reference_arrays_manifest.json').write_text(json.dumps({
        'traj_info': {'ordered_traj_list': [['bones', 'walk']], 'start_index': [0], 'end_index': [20]}
    }))
    job = tmp_path / 'job.yaml'
    job.write_text('bundle: model\nreference_root: reference\nmotion: absent\ncommand_source: oracle\nfixed_initial_anchor: true\nticks: auto\n')
    with pytest.raises(ValidationError, match='not in reference tree'):
        load_lifecycle_job(job)
    job.write_text(job.read_text().replace('absent', 'walk'))
    assert load_lifecycle_job(job).ticks == 'auto'
    job.write_text(job.read_text() + 'start_frame: 19\n')
    with pytest.raises(ValidationError, match='leave at least one'):
        load_lifecycle_job(job)


@pytest.mark.parametrize('fields', [{'ticks': 0}, {'ticks': 'auto'}, {'pin_reference': False}, {'stand_hold_seconds': 10}])
def test_invalid_playback_combinations_are_schema_errors(fields):
    with pytest.raises(ValidationError):
        LifecycleJob(bundle='model', **fields)
