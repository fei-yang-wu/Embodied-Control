"""Fixtures for the lowlevel runtime suite.

The whole directory needs numpy, which the light default env deliberately
lacks -- skip everything there so `pixi run test` stays green.
"""

import pytest

np = pytest.importorskip("numpy")

from embodied_control.lowlevel.bundle import (  # noqa: E402
    ActionContract,
    BundleManifest,
    CommandContract,
    ObservationContract,
    ObservationTerm,
)

JOINT_NAMES = [f"joint_{i}" for i in range(29)]
SDK_NAMES = [f"sdk_{i}" for i in range(29)]
PERMUTATION = list(reversed(range(29)))


@pytest.fixture
def action_contract() -> ActionContract:
    return ActionContract(
        isaac_joint_names=JOINT_NAMES,
        sdk_joint_names=SDK_NAMES,
        isaac_to_sdk=PERMUTATION,
        default_joint_pos=[0.1] * 29,
        action_scale=[0.25] * 29,
        stiffness=[100.0] * 29,
        damping=[2.0] * 29,
    )


@pytest.fixture
def latent_obs_contract() -> ObservationContract:
    return ObservationContract(
        terms=[
            ObservationTerm(name="latent_command", width=8, normalize=False),
            ObservationTerm(name="projected_gravity", width=3),
            ObservationTerm(name="base_ang_vel", width=3),
            ObservationTerm(name="joint_pos_rel", width=29),
            ObservationTerm(name="joint_vel_rel", width=29),
            ObservationTerm(name="last_action", width=29),
        ],
        total_width=101,
    )


@pytest.fixture
def latent_manifest(action_contract, latent_obs_contract) -> BundleManifest:
    return BundleManifest(
        source={"checkpoint_sha256": "0" * 64, "task": "test"},
        interface="latent",
        obs=latent_obs_contract,
        action=action_contract,
        command=CommandContract(
            z_dim=6,
            phase_mode="sin_cos",
            phase_dim=2,
            hold_steps=5,
            state_dim=4,
            window_steps=9,
            horizon_steps=10,
            encoder_window_mode="intermediate",
            macro_frame_stride=1,
            macro_anchor_mode="expert_heading",
        ),
    )


@pytest.fixture
def fsq_manifest(action_contract) -> BundleManifest:
    return BundleManifest(
        source={"checkpoint_sha256": "0" * 64, "task": "test-fsq"},
        interface="latent",
        obs=ObservationContract(
            terms=[
                ObservationTerm(name="latent_command", width=10, normalize=False),
                ObservationTerm(name="projected_gravity", width=3),
                ObservationTerm(name="base_ang_vel", width=3),
                ObservationTerm(name="joint_pos_rel", width=29),
                ObservationTerm(name="joint_vel_rel", width=29),
                ObservationTerm(name="last_action", width=29),
            ],
            total_width=103,
        ),
        action=action_contract,
        command=CommandContract(
            z_dim=8,
            phase_mode="sin_cos",
            phase_dim=2,
            hold_steps=5,
            state_dim=4,
            window_steps=9,
            horizon_steps=10,
            encoder_window_mode="intermediate",
            macro_frame_stride=1,
            macro_anchor_mode="robot",
            quantizer="fsq",
            fsq_half_levels=[16.0] * 8,
        ),
    )


class FakeEngine:
    """Deterministic engine: action = first 29 obs values times 0.5."""

    def __init__(self):
        self.calls = 0

    def load(self, path, device="cpu"):
        return None

    def warmup(self, iters=3, input_width=None):
        return None

    def infer(self, obs, out=None):
        self.calls += 1
        action = 0.5 * np.asarray(obs[:29], dtype=np.float32)
        if out is not None:
            np.copyto(out, action)
            return out
        return action

    @property
    def stats(self):
        from embodied_control.lowlevel.engine.base import LatencyStats

        return LatencyStats(count=self.calls)


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()
