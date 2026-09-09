"""Hardware runs what the plant already ran."""

import json
import time

import pytest

from embodied_control.robot.rehearsal import (
    ACCEPTED_END_STATES,
    find_rehearsals,
    is_simulated,
    rehearsal_evidence,
    run_identity,
)

SHA = "f" * 64


def _identity(**overrides):
    values = {
        "bundle_sha": SHA,
        "bundle_name": "sonic_v1_1",
        "motion": "hurry_idle_001_A277",
        "command_source": "oracle",
        "network": "enp128s31f6",
        "start_frame": 0,
        "ticks": 504,
    }
    values.update(overrides)
    return run_identity(**values)


def _write_run(directory, identity, *, state="VENDOR_RESTORED", failed=0, age_days=0.0):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "lifecycle.json").write_text(
        json.dumps(
            {
                "state": state,
                "fault_reason": "",
                "episode": 1,
                "transitions": 16,
                "failed_transitions": failed,
                "rehearsal": identity,
                "finished_at": time.time() - age_days * 86400.0,
            }
        )
    )
    return directory


def test_a_loopback_run_is_the_rehearsal_not_the_thing_gated():
    assert is_simulated(_identity(network="lo"))
    assert not is_simulated(_identity())


def test_the_gate_passes_on_a_clean_rehearsal_of_the_same_bundle(tmp_path):
    _write_run(tmp_path / "lifecycle_sim", _identity(network="lo"))

    result = rehearsal_evidence(tmp_path, _identity())

    assert result.ok, result.detail
    assert result.values["rehearsal_state"] == "VENDOR_RESTORED"


@pytest.mark.parametrize("end_state", sorted(ACCEPTED_END_STATES))
def test_every_accepted_end_state_counts(tmp_path, end_state):
    _write_run(tmp_path / "sim", _identity(network="lo"), state=end_state)

    assert rehearsal_evidence(tmp_path, _identity()).ok


def test_a_rehearsal_of_a_different_bundle_does_not_count(tmp_path):
    _write_run(tmp_path / "sim", _identity(network="lo", bundle_sha="a" * 64))

    result = rehearsal_evidence(tmp_path, _identity())

    assert not result.ok and "no plant rehearsal of bundle" in result.detail


def test_a_rehearsal_of_a_different_motion_does_not_count(tmp_path):
    _write_run(tmp_path / "sim", _identity(network="lo", motion="walk_arc_cw"))

    assert not rehearsal_evidence(tmp_path, _identity()).ok


def test_a_hardware_run_cannot_vouch_for_another_hardware_run(tmp_path):
    _write_run(tmp_path / "hardware", _identity())

    result = rehearsal_evidence(tmp_path, _identity())

    assert not result.ok and "no plant rehearsal" in result.detail


def test_a_rehearsal_that_faulted_does_not_count(tmp_path):
    _write_run(tmp_path / "sim", _identity(network="lo"), state="FAULT")

    result = rehearsal_evidence(tmp_path, _identity())

    assert not result.ok and "ended in FAULT" in result.detail


def test_a_rehearsal_with_a_failed_gate_does_not_count(tmp_path):
    _write_run(tmp_path / "sim", _identity(network="lo"), failed=1)

    result = rehearsal_evidence(tmp_path, _identity())

    assert not result.ok and "failed transition" in result.detail


def test_a_stale_rehearsal_is_refused(tmp_path):
    _write_run(tmp_path / "sim", _identity(network="lo"), age_days=30.0)

    result = rehearsal_evidence(tmp_path, _identity(), max_age_days=14.0)

    assert not result.ok and "days old" in result.detail
    assert result.values["rehearsal_age_days"] == pytest.approx(30.0, abs=0.1)


def test_the_newest_rehearsal_is_the_one_judged(tmp_path):
    _write_run(tmp_path / "old", _identity(network="lo"), age_days=3.0)
    _write_run(tmp_path / "new", _identity(network="lo"), state="FAULT", age_days=0.0)

    result = rehearsal_evidence(tmp_path, _identity())

    assert not result.ok and "ended in FAULT" in result.detail


def test_an_empty_root_says_what_to_do(tmp_path):
    result = rehearsal_evidence(tmp_path / "nothing", _identity())

    assert not result.ok and "run this job against the plant" in result.detail


def test_unreadable_runs_are_skipped_not_fatal(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "lifecycle.json").write_text("{not json")
    _write_run(tmp_path / "sim", _identity(network="lo"))

    assert len(find_rehearsals(tmp_path)) == 1
    assert rehearsal_evidence(tmp_path, _identity()).ok


@pytest.mark.parametrize('changed', [dict(reference_sha='b'*64), dict(start_frame=2), dict(ticks=100)])
def test_different_reference_content_or_playback_range_needs_new_rehearsal(tmp_path, changed):
    _write_run(tmp_path / 'sim', _identity(network='lo', reference_sha='a'*64))
    assert not rehearsal_evidence(tmp_path, _identity(**({'reference_sha': 'a'*64} | changed))).ok


def test_legacy_rehearsal_cannot_vouch_for_hashed_reference(tmp_path):
    _write_run(tmp_path / 'sim', _identity(network='lo'))
    assert not rehearsal_evidence(tmp_path, _identity(reference_sha='a'*64)).ok


def test_identity_follows_selected_tracker_mode_and_start_frame():
    from pathlib import Path
    from types import SimpleNamespace
    from embodied_control.cli import _run_identity
    from embodied_control.robot.lifecycle_job import LifecycleJob
    job = LifecycleJob(bundle='/models/default')
    manifest = SimpleNamespace(source={'checkpoint_sha256': SHA}, model_dump=lambda **kwargs: {'files': {'policy.onnx': SHA}})
    bundle = SimpleNamespace(root=Path('/models/selected'), manifest=manifest)
    selection = SimpleNamespace(motion='selected_motion', start_frame=7, mode='oracle')
    identity = _run_identity(job, bundle, 'lo', selection, 13)
    assert identity['bundle_name'] == 'selected'
    assert identity['motion'] == 'selected_motion'
    assert identity['command_source'] == 'oracle'
    assert identity['start_frame'] == 7
    assert identity['ticks'] == 13
    changed_timing = job.model_copy(update={'lead_ticks': job.lead_ticks + 1})
    assert _run_identity(changed_timing, bundle, 'lo', selection, 13)['deployment_sha'] != identity['deployment_sha']
    changed = job.model_copy(update={'start_pose': 'motion'})
    assert _run_identity(changed, bundle, 'lo', selection, 13)['deployment_sha'] != identity['deployment_sha']


def test_recovery_does_not_clear_a_runtime_fault(tmp_path):
    directory = _write_run(tmp_path / "sim", _identity(network="lo"))
    path = directory / "lifecycle.json"
    run = json.loads(path.read_text())
    run["fault_reason"] = "command stale"
    path.write_text(json.dumps(run))
    assert not rehearsal_evidence(tmp_path, _identity()).ok
