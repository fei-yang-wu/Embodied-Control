import json

import numpy as np
import pytest

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.command_buffer import InProcessCommandBuffer
from embodied_control.lowlevel.contracts import CommandPacket, RobotState
from embodied_control.lowlevel.envs.fake import FakeBackend
from embodied_control.lowlevel.job import LowLevelJob
from embodied_control.lowlevel.loop import ControlLoop, write_run_artifacts
from embodied_control.lowlevel.observation import ObservationAssembler
from embodied_control.lowlevel.tracker import BufferedCommandSource, LowLevelTracker


def _state(width=29):
    return RobotState(
        stamp=0.0,
        joint_pos=np.full(width, 0.1, dtype=np.float32),
        joint_vel=np.zeros(width, dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
    )


def _latent_packet(z, sequence=0):
    values = np.asarray(z, dtype=np.float32)
    return CommandPacket(
        interface="latent", values=values, sequence=sequence, stamp=0.0,
        terms={"latent_command": values},
    )


def _tracker(latent_manifest, fake_engine, tmp_path):
    bundle = PolicyBundle(root=tmp_path, manifest=latent_manifest)
    buffer = InProcessCommandBuffer()
    source = BufferedCommandSource(buffer, control_hz=50)
    return LowLevelTracker(bundle, fake_engine, source), buffer


def test_observation_order_and_values(latent_manifest):
    assembler = ObservationAssembler(
        latent_manifest.obs,
        default_joint_pos=np.asarray(latent_manifest.action.default_joint_pos, dtype=np.float32),
    )
    state = _state()
    buffer = InProcessCommandBuffer()
    buffer.publish(_latent_packet(np.arange(8)))
    source = BufferedCommandSource(buffer)
    sample = source.update(0, state)
    obs = assembler.assemble(state, sample, np.full(29, 0.5, dtype=np.float32))
    np.testing.assert_allclose(obs[:8], np.arange(8, dtype=np.float32))
    np.testing.assert_allclose(obs[8:11], [0.0, 0.0, -1.0])
    np.testing.assert_allclose(obs[14:43], 0.0)   # joint_pos_rel: pos == default
    np.testing.assert_allclose(obs[72:101], 0.5)  # last_action passthrough


def test_observation_rejects_wrong_width(latent_manifest):
    assembler = ObservationAssembler(
        latent_manifest.obs,
        default_joint_pos=np.asarray(latent_manifest.action.default_joint_pos, dtype=np.float32),
    )
    buffer = InProcessCommandBuffer()
    buffer.publish(_latent_packet(np.arange(5)))
    sample = BufferedCommandSource(buffer).update(0, _state())
    with pytest.raises(ValueError, match="width"):
        assembler.assemble(_state(), sample, np.zeros(29, dtype=np.float32))


def test_tracker_step_and_last_action(latent_manifest, fake_engine, tmp_path):
    tracker, buffer = _tracker(latent_manifest, fake_engine, tmp_path)
    state = _state()
    tracker.reset(state)
    joint_command, sample = tracker.step(0, state)
    assert joint_command is None
    assert not sample.available

    buffer.publish(_latent_packet(np.ones(8)))
    joint_command, sample = tracker.step(1, state)
    assert sample.renewed
    expected_action = 0.5 * np.concatenate([np.ones(8), [0.0, 0.0, -1.0]])[:29]
    padded = np.zeros(29, dtype=np.float32)
    padded[:11] = expected_action[:11]
    np.testing.assert_allclose(tracker.last_action[:11], padded[:11])
    np.testing.assert_allclose(
        joint_command.q_target,
        0.1 + 0.25 * tracker.last_action,
    )

    _, second = tracker.step(2, state)
    assert not second.renewed
    assert second.age_ticks == 1


def _loop_fixture(latent_manifest, fake_engine, tmp_path, publisher=None, max_steps=20):
    bundle = PolicyBundle(root=tmp_path, manifest=latent_manifest)
    buffer = InProcessCommandBuffer()
    tracker = LowLevelTracker(
        bundle, fake_engine, BufferedCommandSource(buffer, control_hz=50)
    )
    backend = FakeBackend(latent_manifest.action, control_hz=50)
    job = LowLevelJob(
        bundle=str(tmp_path),
        rollout={"episodes": 1, "max_steps": max_steps},
        safety={"command_absent_ticks": 5},
    )
    return ControlLoop(job, tracker, backend, publisher), buffer, job, bundle


class ScriptedPublisher:
    def __init__(self, buffer, packets):
        self.buffer = buffer
        self.packets = list(packets)
        self.exhausted = False
        self._cursor = 0

    def reset(self):
        self._cursor = 0
        self.exhausted = False

    def tick(self, tick, stamp, state=None):
        if self._cursor < len(self.packets):
            self.buffer.publish(self.packets[self._cursor])
            self._cursor += 1
        else:
            self.exhausted = True


def test_loop_damps_when_no_command_ever_arrives(latent_manifest, fake_engine, tmp_path):
    loop, _buffer, _job, _bundle = _loop_fixture(latent_manifest, fake_engine, tmp_path)
    result = loop.run()
    episode = result.episodes[0]
    assert episode.status == "damped"
    assert episode.damp_cause == "command_absent"


def test_loop_completes_with_scripted_publisher(latent_manifest, fake_engine, tmp_path):
    loop, buffer, job, bundle = _loop_fixture(latent_manifest, fake_engine, tmp_path)
    loop.publisher = ScriptedPublisher(
        buffer, [_latent_packet(np.ones(8)) for _ in range(40)]
    )
    result = loop.run()
    episode = result.episodes[0]
    assert episode.status == "completed"
    assert episode.steps == job.rollout.max_steps
    assert episode.renewals >= 1

    run_dir = tmp_path / "run"
    write_run_artifacts(run_dir, job, {"source": "test"}, result)
    for name in ["resolved_job.json", "manifest.json", "episodes.jsonl", "metrics.json", "status.json"]:
        assert (run_dir / name).is_file()
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "succeeded"


def test_loop_determinism(latent_manifest, fake_engine, tmp_path):
    def run_once():
        loop, buffer, _job, _bundle = _loop_fixture(latent_manifest, fake_engine, tmp_path)
        loop.publisher = ScriptedPublisher(
            buffer, [_latent_packet(np.full(8, 0.3)) for _ in range(40)]
        )
        result = loop.run()
        return [(e.status, e.steps, e.renewals) for e in result.episodes]

    assert run_once() == run_once()


def test_fsq_snap_quantizes_and_preserves_phase(fsq_manifest, fake_engine, tmp_path):
    bundle = PolicyBundle(root=tmp_path, manifest=fsq_manifest)
    buffer = InProcessCommandBuffer()
    tracker = LowLevelTracker(bundle, fake_engine, BufferedCommandSource(buffer))
    state = _state()
    tracker.reset(state)

    z = np.array([0.03, 0.51, -0.99, 1.7, -1.4, 0.0, 0.999, -0.001], dtype=np.float32)
    phase = np.array([0.123, 0.456], dtype=np.float32)
    values = np.concatenate([z, phase])
    buffer.publish(
        CommandPacket(
            interface="latent", values=values, sequence=0, stamp=0.0,
            terms={"latent_command": values},
        )
    )
    _, sample = tracker.step(0, state)
    assembled = tracker.observation.buffer
    expected_z = np.clip(np.rint(z * 16.0), -16.0, 15.0) / 16.0
    np.testing.assert_allclose(assembled[:8], expected_z)
    np.testing.assert_allclose(assembled[8:10], phase)
    np.testing.assert_allclose(sample.terms["latent_command"][:8], expected_z)
    assert expected_z[3] == pytest.approx(15.0 / 16.0)   # clipped above
    assert expected_z[4] == pytest.approx(-1.0)          # clipped below


def test_fsq_snap_idempotent_on_lattice(fsq_manifest, fake_engine, tmp_path):
    bundle = PolicyBundle(root=tmp_path, manifest=fsq_manifest)
    buffer = InProcessCommandBuffer()
    tracker = LowLevelTracker(bundle, fake_engine, BufferedCommandSource(buffer))
    lattice = np.arange(-16, -8, dtype=np.float32) / 16.0
    values = np.concatenate([lattice, np.zeros(2, dtype=np.float32)])
    buffer.publish(
        CommandPacket(
            interface="latent", values=values, sequence=0, stamp=0.0,
            terms={"latent_command": values},
        )
    )
    tracker.reset(_state())
    tracker.step(0, _state())
    np.testing.assert_array_equal(tracker.observation.buffer[:8], lattice)


def test_fsq_contract_validators():
    from pydantic import ValidationError

    from embodied_control.lowlevel.bundle import CommandContract

    with pytest.raises(ValidationError, match="requires fsq_half_levels"):
        CommandContract(z_dim=8, quantizer="fsq")
    with pytest.raises(ValidationError, match="length must equal z_dim"):
        CommandContract(z_dim=8, quantizer="fsq", fsq_half_levels=[16.0] * 4)
    with pytest.raises(ValidationError, match="only valid with quantizer=fsq"):
        CommandContract(z_dim=8, fsq_half_levels=[16.0] * 8)


def test_onboard_encoder_hold_and_phase(latent_manifest):
    from embodied_control.lowlevel.publishers.onboard_encoder import OnboardEncoderPublisher

    class CountingEncoder:
        def __init__(self):
            self.calls = 0

        def infer(self, obs, out=None):
            self.calls += 1
            return np.full(6, float(self.calls), dtype=np.float32)

    encoder = CountingEncoder()
    buffer = InProcessCommandBuffer()
    macro = np.arange(120 * 4, dtype=np.float32).reshape(120, 4)
    publisher = OnboardEncoderPublisher(
        buffer, encoder, latent_manifest.command, macro
    )
    phases = []
    for tick in range(60):
        publisher.tick(tick, stamp=float(tick))
        packet = buffer.snapshot().packet
        assert packet.values.shape == (8,)
        phases.append((packet.values[6], packet.values[7]))
    assert encoder.calls == 12  # hold_steps=5 over 60 ticks
    np.testing.assert_allclose(phases[0], [0.0, 1.0], atol=1e-6)  # phase 0 at renewal
    assert phases[1] != phases[0]
    np.testing.assert_allclose(phases[5], [0.0, 1.0], atol=1e-6)  # next renewal restarts
    assert not publisher.exhausted

    for tick in range(60, 125):
        publisher.tick(tick, stamp=float(tick))
    assert publisher.exhausted


def test_loop_damps_on_nan_action(latent_manifest, tmp_path):
    class NanEngine:
        def infer(self, obs, out=None):
            value = np.full(29, np.nan, dtype=np.float32)
            if out is not None:
                np.copyto(out, value)
                return out
            return value

        @property
        def stats(self):
            from embodied_control.lowlevel.engine.base import LatencyStats

            return LatencyStats()

    loop, buffer, _job, _bundle = _loop_fixture(latent_manifest, NanEngine(), tmp_path)
    loop.publisher = ScriptedPublisher(buffer, [_latent_packet(np.ones(8))] * 5)
    result = loop.run()
    assert result.episodes[0].status == "damped"
    assert "runtime_error" in result.episodes[0].damp_cause
