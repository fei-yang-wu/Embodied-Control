"""Contract-driven observation assembly."""

from __future__ import annotations

import numpy as np

from embodied_control.lowlevel.bundle import ObservationContract
from embodied_control.lowlevel.contracts import CommandSample, RobotState


class ObservationAssembler:
    def __init__(self, contract: ObservationContract, *, default_joint_pos: np.ndarray | None = None):
        self.contract = contract
        self._buffer = np.empty(contract.total_width, dtype=np.float32)
        self._default_joint_pos = np.zeros(29, dtype=np.float32) if default_joint_pos is None else np.asarray(
            default_joint_pos, dtype=np.float32
        )
        if self._default_joint_pos.shape != (29,):
            raise ValueError("default_joint_pos must have shape (29,)")
        self._slices = {}
        cursor = 0
        for term in contract.terms:
            self._slices[term.name] = slice(cursor, cursor + term.width)
            cursor += term.width

    @property
    def buffer(self) -> np.ndarray:
        return self._buffer

    def assemble(
        self,
        state: RobotState,
        command: CommandSample,
        last_action: np.ndarray,
    ) -> np.ndarray:
        state.validate(len(self._default_joint_pos))
        values = {
            "projected_gravity": np.asarray(state.projected_gravity),
            "base_ang_vel": np.asarray(state.base_ang_vel),
            "joint_pos_rel": np.asarray(state.joint_pos) - self._default_joint_pos,
            "joint_vel_rel": np.asarray(state.joint_vel),
            "last_action": np.asarray(last_action),
        }
        values.update({name: np.asarray(value) for name, value in command.terms.items()})
        for term in self.contract.terms:
            if term.name not in values:
                raise KeyError(f"observation term {term.name!r} has no runtime value")
            value = values[term.name].reshape(-1)
            if value.shape != (term.width,):
                raise ValueError(
                    f"observation term {term.name!r} must have width {term.width}, got {value.shape}"
                )
            if not np.isfinite(value).all():
                raise ValueError(f"observation term {term.name!r} contains non-finite values")
            np.copyto(self._buffer[self._slices[term.name]], value.astype(np.float32, copy=False))
        if not np.isfinite(self._buffer).all():
            raise ValueError("assembled observation contains non-finite values")
        return self._buffer


__all__ = ["ObservationAssembler"]
