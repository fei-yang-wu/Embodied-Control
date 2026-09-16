"""Recorded-latent playback: the runtime serves z[frame] from a table."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("ec_native")

from embodied_control.lowlevel.native_core import NativeFakeLoop  # noqa: E402
from embodied_control.lowlevel.publishers.native_oracle import (  # noqa: E402
    NativeOracleWorker,
)
from embodied_control.lowlevel.recorded_latents import (  # noqa: E402
    check_table, load_table, save_table, table_from_telemetry,
)

from test_native_core import _native_bundle, _shm_name, _write_reference_tree  # noqa: E402


def _oracle_loop(tmp_path, latent_manifest, label, frames=80):
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
    (tmp_path / label).mkdir(exist_ok=True)
    bundle = _native_bundle(tmp_path / label, manifest, with_encoder=True)
    reference_root = tmp_path / label / "reference"
    qpos, anchor_pos, anchor_quat = _write_reference_tree(
        reference_root, bundle.manifest.action.isaac_joint_names, frames=frames
    )
    request_name = _shm_name(f"{label}_request")
    response_name = _shm_name(f"{label}_response")
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
    return bundle, loop, worker


def test_runtime_serves_recorded_latents_and_skips_the_encoder(tmp_path, latent_manifest):
    bundle, loop, worker = _oracle_loop(tmp_path, latent_manifest, "recorded")
    z_dim = int(bundle.manifest.command.z_dim)
    ticks = 30
    rng = np.random.default_rng(0)
    table = rng.standard_normal((ticks + 5, z_dim)).astype(np.float32)
    valid = np.ones(ticks + 5, dtype=np.uint8)
    valid[7] = 0  # one hole: the previous latent is held and a miss is counted
    loop.set_recorded_latents(table, valid)
    assert loop.recorded_latents_enabled
    try:
        loop.start(ticks, paced=True)
        loop.wait()
    finally:
        worker.close()
    stats = loop.stats()
    assert stats["fault"] == 0
    assert stats["control_ticks"] == ticks
    assert stats["encoder_inferences"] == 0
    assert stats["recorded_latent_hits"] + stats["recorded_latent_misses"] == ticks
    assert stats["recorded_latent_misses"] >= 1
    frames = loop.reference_frames()
    command = loop.command_log()[:, :z_dim]
    for tick in range(ticks):
        frame = int(frames[tick])
        expected = table[frame] if valid[frame] else table[frame - 1]
        np.testing.assert_allclose(command[tick], expected, atol=1e-6)


def test_set_recorded_latents_rejects_the_wrong_width(tmp_path, latent_manifest):
    bundle, loop, worker = _oracle_loop(tmp_path, latent_manifest, "badwidth")
    try:
        with pytest.raises(RuntimeError):
            loop.set_recorded_latents(np.zeros((10, 3), np.float32), np.ones(10, np.uint8))
    finally:
        worker.close()


def test_table_round_trip_from_a_recording(tmp_path, latent_manifest):
    bundle, loop, worker = _oracle_loop(tmp_path, latent_manifest, "source")
    try:
        loop.start(24, paced=True)
        loop.wait()
    finally:
        worker.close()
    assert loop.stats()["fault"] == 0
    z_dim = int(bundle.manifest.command.z_dim)
    sha = "a" * 64
    telemetry = tmp_path / "telemetry.npz"
    np.savez(telemetry, command_log=loop.command_log(), reference_frames=loop.reference_frames())
    table = table_from_telemetry(
        telemetry, bundle_sha256=sha, motion="motion", start_frame=0, z_dim=z_dim,
    )
    # One entry per reference frame, taken from the last tick on that frame.
    frames = loop.reference_frames()
    command = loop.command_log()[:, :z_dim]
    assert list(table.frames) == sorted(set(int(f) for f in frames if f >= 0))
    last = int(np.where(frames == table.frames[-1])[0][-1])
    np.testing.assert_allclose(table.latents[-1], command[last])
    path = save_table(table, tmp_path / "latents.npz")
    loaded = load_table(path)
    np.testing.assert_allclose(loaded.latents, table.latents)
    check_table(loaded, bundle_sha256=sha, motion="motion", z_dim=z_dim)
    with pytest.raises(ValueError):
        check_table(loaded, bundle_sha256="b" * 64, motion="motion", z_dim=z_dim)
    with pytest.raises(ValueError):
        check_table(loaded, bundle_sha256=sha, motion="other", z_dim=z_dim)
    dense, valid = loaded.dense(0)
    assert dense.shape == (int(table.frames.max()) + 1, z_dim)
    assert valid.sum() == table.frames.shape[0]
    # Playing the table back reproduces the source run's latents exactly.
    bundle2, loop2, worker2 = _oracle_loop(tmp_path, latent_manifest, "replay")
    loop2.set_recorded_latents(dense, valid)
    try:
        loop2.start(24, paced=True)
        loop2.wait()
    finally:
        worker2.close()
    assert loop2.stats()["encoder_inferences"] == 0
    replay = loop2.command_log()[:, :z_dim]
    frames2 = loop2.reference_frames()
    for tick in range(24):
        f = int(frames2[tick])
        if f in set(int(x) for x in table.frames):
            np.testing.assert_allclose(replay[tick], dense[f], atol=1e-6)
