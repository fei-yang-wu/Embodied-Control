"""Single-step low-level tracker independent of the robot backend."""

from __future__ import annotations

import numpy as np

from embodied_control.lowlevel.bundle import PolicyBundle
from embodied_control.lowlevel.command_buffer import CommandBuffer
from embodied_control.lowlevel.contracts import CommandSample, JointCommand, RobotState
from embodied_control.lowlevel.engine.base import Engine
from embodied_control.lowlevel.observation import ObservationAssembler


class BufferedCommandSource:
    """Adapt the latest packet in a command buffer to the tracker contract."""

    def __init__(self, buffer: CommandBuffer, *, control_hz: int = 50):
        self.buffer = buffer
        self.control_hz = int(control_hz)
        self._last_sequence = -1
        self._age_ticks = 0

    def reset(self, _state: RobotState) -> None:
        self._last_sequence = -1
        self._age_ticks = 0

    def update(self, _tick: int, _state: RobotState) -> CommandSample:
        snapshot = self.buffer.snapshot()
        packet = snapshot.packet
        if packet is None:
            self._age_ticks += 1
            return CommandSample(
                vector=np.empty(0, dtype=np.float32),
                age_ticks=self._age_ticks,
                renewed=False,
                available=False,
            )
        renewed = packet.sequence != self._last_sequence
        if renewed:
            self._last_sequence = packet.sequence
            self._age_ticks = 0
        else:
            self._age_ticks += 1
        return CommandSample(
            vector=np.asarray(packet.values, dtype=np.float32),
            age_ticks=self._age_ticks,
            renewed=renewed,
            terms=packet.terms,
            metadata={
                **packet.metadata,
                "interface": packet.interface,
                "sequence": packet.sequence,
            },
        )


class LowLevelTracker:
    """Decode one command-buffer snapshot into one robot joint command."""

    def __init__(self, bundle: PolicyBundle, engine: Engine, source: BufferedCommandSource):
        self.bundle = bundle
        self.engine = engine
        self.source = source
        self.observation = ObservationAssembler(
            bundle.manifest.obs,
            default_joint_pos=np.asarray(
                bundle.manifest.action.default_joint_pos, dtype=np.float32
            ),
        )
        self._last_action = np.zeros(bundle.manifest.action.width, dtype=np.float32)
        self._action_out = np.empty(bundle.manifest.action.width, dtype=np.float32)
        command_contract = bundle.manifest.command
        self._fsq_half: np.ndarray | None = None
        self._fsq_z_dim = 0
        if command_contract.quantizer == "fsq":
            self._fsq_half = np.asarray(command_contract.fsq_half_levels, dtype=np.float32)
            self._fsq_z_dim = int(command_contract.z_dim or len(self._fsq_half))

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action

    def reset(self, state: RobotState) -> None:
        self._last_action.fill(0.0)
        self.source.reset(state)

    def _snap_fsq(self, command: CommandSample) -> CommandSample:
        """Snap the z slice onto the FSQ lattice; idempotent for lattice inputs.

        SONIC convention: a planner regresses the PRE-quantized bounded
        vector, so the tracker quantizes at consume time. Phase dims pass
        through untouched.
        """
        half = self._fsq_half
        if half is None:
            return command
        values = np.array(command.vector, dtype=np.float32, copy=True)
        z = values[: self._fsq_z_dim]
        snapped = np.clip(np.rint(z * half), -half, half - 1.0) / half
        values[: self._fsq_z_dim] = snapped
        terms = dict(command.terms)
        if "latent_command" in terms:
            terms["latent_command"] = values
        return CommandSample(
            vector=values,
            age_ticks=command.age_ticks,
            renewed=command.renewed,
            terms=terms,
            available=command.available,
            done=command.done,
            metadata=command.metadata,
        )

    def step(self, tick: int, state: RobotState) -> tuple[JointCommand | None, CommandSample]:
        command = self.source.update(tick, state)
        if not command.available:
            return None, command
        command = self._snap_fsq(command)
        obs = self.observation.assemble(state, command, self._last_action)
        action = self.engine.infer(obs, self._action_out)
        if not np.isfinite(action).all():
            raise ValueError("tracker engine produced a non-finite action")
        q_target, kp, kd = self.bundle.manifest.action.decode(action)
        joint_command = JointCommand(q_target=q_target, kp=kp, kd=kd)
        np.copyto(self._last_action, action)
        return joint_command, command

