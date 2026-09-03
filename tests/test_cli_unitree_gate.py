from pathlib import Path

import pytest

from embodied_control.cli import (
    _unitree_stationary_anchor,
    _unitree_write_gate_error,
)


def test_unitree_writes_require_strict_confirmation():
    assert _unitree_write_gate_error(
        enable_writes=True,
        allow_non_realtime=False,
        confirm="ENABLE_G1_LOWLEVEL",
    ) is None
    assert _unitree_write_gate_error(
        enable_writes=True,
        allow_non_realtime=False,
        confirm="ENABLE_G1_LOWLEVEL_NON_REALTIME",
    ) is not None


def test_unitree_best_effort_writes_require_distinct_confirmation():
    assert _unitree_write_gate_error(
        enable_writes=True,
        allow_non_realtime=True,
        confirm="ENABLE_G1_LOWLEVEL_NON_REALTIME",
    ) is None
    assert _unitree_write_gate_error(
        enable_writes=True,
        allow_non_realtime=True,
        confirm="ENABLE_G1_LOWLEVEL",
    ) is not None


def test_unitree_read_only_needs_no_confirmation():
    assert _unitree_write_gate_error(
        enable_writes=False,
        allow_non_realtime=True,
        confirm="",
    ) is None


def test_hurry_idle_is_accepted_for_fixed_initial_anchor():
    pytest.importorskip("numpy")
    from embodied_control.lowlevel.bundle import PolicyBundle

    root = Path(__file__).resolve().parents[1]
    bundle = PolicyBundle.load(
        root / "assets/latent_playkit/bundles/fsq64_sonic_4500m"
    )
    anchor, displacement, ticks = _unitree_stationary_anchor(
        bundle,
        str(root / "assets/latent_playkit/reference/root_qpos_v1"),
        "hurry_idle_001_A277",
        0,
        0.05,
    )
    assert anchor.position.shape == (3,)
    assert anchor.quaternion_xyzw.shape == (4,)
    assert displacement < 0.05
    assert ticks == 504


def test_fixed_anchor_accepts_every_curated_motion():
    pytest.importorskip("numpy")
    from embodied_control.lowlevel.bundle import PolicyBundle

    root = Path(__file__).resolve().parents[1]
    bundle = PolicyBundle.load(
        root / "assets/latent_playkit/bundles/fsq64_sonic_4500m"
    )
    reference_root = root / "assets/latent_playkit/reference/root_qpos_v1"
    from embodied_control.lowlevel.reference import ReferenceArrays

    arrays = ReferenceArrays(reference_root)
    for motion in arrays.motion_names:
        anchor, displacement, ticks = _unitree_stationary_anchor(
            bundle,
            str(reference_root),
            motion,
            0,
            0.001,
        )
        assert anchor.position.shape == (3,)
        assert displacement >= 0.0
        assert ticks == arrays.motion(motion).length - 1
