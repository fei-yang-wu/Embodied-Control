"""Observation replay on a loop's own recording must reproduce it exactly."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("ec_native")

from embodied_control.lowlevel.native_core import NativeFakeLoop  # noqa: E402
from embodied_control.lowlevel.observation_replay import (  # noqa: E402
    load_observation_recording, replay,
)
from embodied_control.lowlevel.publishers.native_oracle import (  # noqa: E402
    NativeOracleWorker,
)

from test_native_core import _native_bundle, _shm_name, _write_reference_tree  # noqa: E402


@pytest.fixture
def recorded_run(tmp_path, latent_manifest):
    command = latent_manifest.command.model_copy(
        update={
            "state_dim": 64,
            "encoder_state_interface": "joint_qpos_qvel_anchor_ori",
            "macro_anchor_mode": "robot_heading",
            "macro_frame_stride": 5,
            "encoder_trigger": "every_control_tick",
        }
    )
    manifest = latent_manifest.model_copy(update={"command": command})
    bundle = _native_bundle(tmp_path, manifest, with_encoder=True)
    reference_root = tmp_path / "reference"
    qpos, anchor_pos, anchor_quat = _write_reference_tree(
        reference_root, bundle.manifest.action.isaac_joint_names, frames=80
    )
    request_name = _shm_name("obs_replay_request")
    response_name = _shm_name("obs_replay_response")
    worker = NativeOracleWorker(
        request_name, response_name, bundle, reference_root, "motion", create_slots=True,
    )
    worker.start()
    loop = NativeFakeLoop(
        bundle, request_slot=request_name, response_slot=response_name,
        create_slots=False, command_source="oracle", hold_steps=5, lead_ticks=2,
        command_stale_ms=1000.0,
    )
    loop.set_initial_pose(np.concatenate([anchor_pos[0], anchor_quat[0], qpos[0, 7:]]))
    try:
        loop.start(30, paced=True)
        loop.wait()
    finally:
        worker.close()
    assert loop.stats()["fault"] == 0
    np.savez(
        tmp_path / "telemetry.npz",
        joint_position_log=loop.joint_position_log(),
        command_target_log=loop.command_target_log(),
        state_log=loop.state_log(),
        observation_log=loop.observation_log(),
        command_log=loop.command_log(),
        encoder_window_log=loop.encoder_window_log(),
        tick_stamps_ns=loop.tick_stamps_ns(),
    )
    return bundle, tmp_path / "telemetry.npz"


def test_observation_replay_reproduces_the_loops_own_record(recorded_run):
    bundle, path = recorded_run
    rec = load_observation_recording(path)
    assert rec.controlled.sum() == 30
    checks = replay(rec, bundle)
    assert set(checks) == {"actor", "encoder", "assembly"}
    for name, check in checks.items():
        assert check.ticks_compared > 0, name
        assert check.ok, (name, check.max_abs)
    # The assembly check compares only once the last-action history is fully
    # post-blend; with no blend it still skips the history span.
    assert checks["assembly"].ticks_compared < checks["actor"].ticks_compared


def test_observation_replay_flags_a_corrupted_observation(recorded_run):
    bundle, path = recorded_run
    rec = load_observation_recording(path)
    rec.observation[20, :3] += 0.5  # a wrong projected gravity on one tick
    checks = replay(rec, bundle)
    assert not checks["assembly"].ok
    # Tick 20 disagrees on the term itself; tick 21 on the last action the
    # corrupted observation produced; from 22 the rebuild matches again.
    assert checks["assembly"].first_clean_tick == 22
    assert not checks["actor"].ok  # the recorded target came from the true observation
