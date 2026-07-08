"""Embodiment adapter contract.

The adapter owns the mapping between the policy's normalized command space and
the simulator's concrete actuation. It is the extensibility point the design
calls the "most important boundary": for M1 it is a passthrough scaler, but a
real robot swaps in IK, joint-order remapping, safety clipping, etc. here —
without touching the policy service or the rollout loop.
"""

from __future__ import annotations

from typing import Protocol


class EmbodimentController(Protocol):
    schema_id: str
    action_schema_id: str

    def decode_action(self, normalized_action: list[float]) -> list[float]:
        """Map a normalized command in [-1, 1]^n to simulator ctrl values."""
        ...

    def fallback(self) -> list[float]:
        """Safe action to apply when the policy cannot produce one."""
        ...
