"""Latent-source publisher semantics, and parity with the certified path."""

import pytest

np = pytest.importorskip("numpy")

from embodied_control.lowlevel.bundle import CommandContract  # noqa: E402
from embodied_control.lowlevel.command_buffer import InProcessCommandBuffer  # noqa: E402
from embodied_control.lowlevel.contracts import RobotState  # noqa: E402
from embodied_control.lowlevel.publishers.latent_perturbation import (  # noqa: E402
    ConstantLatentSource,
    LatentPublisher,
    ReferenceEncoderSource,
    SequenceLatentSource,
    TransformedLatentSource,
)
from embodied_control.lowlevel.publishers.onboard_encoder import (  # noqa: E402
    OnboardEncoderPublisher,
)
from embodied_control.lowlevel.reference import ReferenceMotion  # noqa: E402

IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
Z_DIM = 6


class HashEncoder:
    """Deterministic, window-sensitive stand-in for the DiffSR encoder."""

    def __init__(self):
        self.inputs = []

    def load(self, path, device="cpu"):
        return None

    def warmup(self, iters=3, input_width=None):
        return None

    def infer(self, obs, out=None):
        obs = np.asarray(obs, dtype=np.float32)
        self.inputs.append(obs.copy())
        return np.array(
            [float(obs[k :: Z_DIM].sum()) for k in range(Z_DIM)], dtype=np.float32
        )


def _command(**overrides) -> CommandContract:
    fields = dict(
        z_dim=Z_DIM,
        phase_mode="sin_cos",
        phase_dim=2,
        hold_steps=3,
        state_dim=38,
        window_steps=9,
        horizon_steps=10,
        encoder_window_mode="intermediate",
        macro_frame_stride=1,
        macro_anchor_mode="robot",
    )
    fields.update(overrides)
    return CommandContract(**fields)


def _motion(frames: int = 24, seed: int = 0) -> ReferenceMotion:
    rng = np.random.default_rng(seed)
    return ReferenceMotion(
        "m",
        rng.standard_normal((frames, 29)).astype(np.float32),
        rng.standard_normal((frames, 3)).astype(np.float32),
        np.tile(IDENTITY_XYZW, (frames, 1)),
    )


def _state(step: int = 0) -> RobotState:
    return RobotState(
        stamp=float(step),
        joint_pos=np.zeros(29, np.float32),
        joint_vel=np.zeros(29, np.float32),
        projected_gravity=np.array([0, 0, -1], np.float32),
        base_ang_vel=np.zeros(3, np.float32),
        anchor_pos_w=np.array([0.05 * step, 0.0, 0.75], np.float32),
        anchor_quat_w=IDENTITY_XYZW,
    )


def _drain(publisher, buffer, ticks: int) -> list[np.ndarray]:
    published = []
    for step in range(ticks):
        publisher.tick(step, float(step), _state(step))
        packet = buffer.snapshot().packet
        published.append(np.array(packet.values, copy=True))
    return published


def test_reference_source_matches_certified_publisher():
    """LatentPublisher + ReferenceEncoderSource == OnboardEncoderPublisher."""
    command = _command()
    motion = _motion()
    ticks = 20

    certified_buffer = InProcessCommandBuffer()
    certified_encoder = HashEncoder()
    certified = OnboardEncoderPublisher(
        certified_buffer, certified_encoder, command, motion=motion
    )
    expected = _drain(certified, certified_buffer, ticks)

    playground_buffer = InProcessCommandBuffer()
    playground_encoder = HashEncoder()
    publisher = LatentPublisher(
        playground_buffer,
        ReferenceEncoderSource(playground_encoder, command, motion=motion),
        command,
    )
    actual = _drain(publisher, playground_buffer, ticks)

    assert len(actual) == len(expected)
    for step, (left, right) in enumerate(zip(actual, expected)):
        np.testing.assert_array_equal(left, right, err_msg=f"tick {step}")
    assert len(playground_encoder.inputs) == len(certified_encoder.inputs)
    for left, right in zip(playground_encoder.inputs, certified_encoder.inputs):
        np.testing.assert_array_equal(left, right)


def test_hold_schedule_and_phase():
    command = _command(hold_steps=4)
    buffer = InProcessCommandBuffer()
    z = np.arange(Z_DIM, dtype=np.float32)
    publisher = LatentPublisher(buffer, ConstantLatentSource(z), command)
    published = _drain(publisher, buffer, 8)

    for values in published:
        np.testing.assert_array_equal(values[:Z_DIM], z)
    phases = [np.arctan2(values[Z_DIM], values[Z_DIM + 1]) for values in published]
    expected = [2.0 * np.pi * (k % 4) / 4.0 for k in range(8)]
    np.testing.assert_allclose(
        np.mod(phases, 2 * np.pi), np.mod(expected, 2 * np.pi), atol=1e-6
    )
    assert publisher.renewals == 2
    assert publisher.z_trace[0].shape == (Z_DIM,)


def test_constant_source_never_exhausts_but_max_renewals_stops():
    command = _command(hold_steps=2)
    buffer = InProcessCommandBuffer()
    publisher = LatentPublisher(
        buffer,
        ConstantLatentSource(np.ones(Z_DIM, dtype=np.float32)),
        command,
        max_renewals=3,
    )
    _drain(publisher, buffer, 5)
    assert publisher.renewals == 3
    assert publisher.exhausted


def test_transform_sees_every_renewal():
    command = _command(hold_steps=2)
    buffer = InProcessCommandBuffer()
    seen = []

    def shift(z, renewal_index):
        seen.append(renewal_index)
        return z + float(renewal_index)

    publisher = LatentPublisher(
        buffer,
        TransformedLatentSource(
            ConstantLatentSource(np.zeros(Z_DIM, dtype=np.float32)), shift
        ),
        command,
    )
    published = _drain(publisher, buffer, 6)
    assert seen == [0, 1, 2]
    np.testing.assert_array_equal(published[0][:Z_DIM], np.zeros(Z_DIM))
    np.testing.assert_array_equal(published[2][:Z_DIM], np.ones(Z_DIM))
    np.testing.assert_array_equal(published[4][:Z_DIM], 2.0 * np.ones(Z_DIM))


def test_transform_must_keep_the_z_shape():
    command = _command()
    publisher = LatentPublisher(
        InProcessCommandBuffer(),
        TransformedLatentSource(
            ConstantLatentSource(np.zeros(Z_DIM, dtype=np.float32)),
            lambda z, index: z[:-1],
        ),
        command,
    )
    with pytest.raises(ValueError, match="changed the z shape"):
        publisher.tick(0, 0.0, _state())


def test_sequence_source_holds_or_exhausts():
    command = _command(hold_steps=1)
    latents = np.stack([np.full(Z_DIM, k, dtype=np.float32) for k in range(3)])

    buffer = InProcessCommandBuffer()
    holding = LatentPublisher(buffer, SequenceLatentSource(latents), command)
    published = _drain(holding, buffer, 5)
    np.testing.assert_array_equal(published[4][:Z_DIM], latents[-1])
    assert not holding.exhausted

    buffer = InProcessCommandBuffer()
    ending = LatentPublisher(
        buffer, SequenceLatentSource(latents, hold_last=False), command
    )
    _drain(ending, buffer, 3)
    assert ending.exhausted


def test_reference_source_start_frame_and_exhaustion():
    command = _command(hold_steps=1)
    motion = _motion(frames=8)
    encoder = HashEncoder()
    source = ReferenceEncoderSource(encoder, command, motion=motion, start_frame=5)
    buffer = InProcessCommandBuffer()
    publisher = LatentPublisher(buffer, source, command)
    assert source.cursor == 5
    _drain(publisher, buffer, 2)
    assert source.cursor == 7
    assert publisher.exhausted


def test_reference_source_rejects_missing_anchor():
    command = _command()
    source = ReferenceEncoderSource(HashEncoder(), command, motion=_motion())
    publisher = LatentPublisher(InProcessCommandBuffer(), source, command)
    with pytest.raises(ValueError, match="anchor"):
        publisher.tick(0, 0.0, None)


def test_snap_fsq_matches_tracker_consume_path(fsq_manifest, fake_engine, tmp_path):
    """`snap_fsq` must equal what `LowLevelTracker._snap_fsq` consumes."""
    from embodied_control.lowlevel.bundle import PolicyBundle
    from embodied_control.lowlevel.contracts import CommandPacket
    from embodied_control.lowlevel.latent import fsq_codes, snap_fsq
    from embodied_control.lowlevel.tracker import (
        BufferedCommandSource,
        LowLevelTracker,
    )

    half = np.asarray(fsq_manifest.command.fsq_half_levels, dtype=np.float32)
    z = np.array([0.03, 0.51, -0.99, 1.7, -1.4, 0.0, 0.999, -0.001], np.float32)
    values = np.concatenate([z, np.zeros(2, np.float32)])

    bundle = PolicyBundle(root=tmp_path, manifest=fsq_manifest)
    buffer = InProcessCommandBuffer()
    tracker = LowLevelTracker(bundle, fake_engine, BufferedCommandSource(buffer))
    state = _state()
    tracker.reset(state)
    buffer.publish(
        CommandPacket(
            interface="latent", values=values, sequence=0, stamp=0.0,
            terms={"latent_command": values},
        )
    )
    tracker.step(0, state)

    snapped = snap_fsq(z, half)
    np.testing.assert_allclose(tracker.observation.buffer[: z.size], snapped)
    np.testing.assert_array_equal(snap_fsq(snapped, half), snapped)
    np.testing.assert_array_equal(fsq_codes(z, half), (snapped * half).astype(np.int32))


def test_precomputed_window_source():
    command = _command(macro_anchor_mode="expert_heading")
    states = np.arange(20 * 38, dtype=np.float32).reshape(20, 38)
    encoder = HashEncoder()
    source = ReferenceEncoderSource(encoder, command, macro_states=states)
    publisher = LatentPublisher(InProcessCommandBuffer(), source, command)
    publisher.tick(0, 0.0, None)
    window = encoder.inputs[0].reshape(10, 38)
    np.testing.assert_array_equal(window, states[:10])
