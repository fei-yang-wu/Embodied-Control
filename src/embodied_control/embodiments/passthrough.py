"""Passthrough controller: scale normalized policy commands to actuator ctrlrange.

This is the minimal ``decode_action``: it clips the policy's [-1, 1] command and
affinely maps it onto each actuator's ``ctrlrange``. It carries no task reward or
transport knowledge — only the robot-side action mapping.
"""

from __future__ import annotations

from embodied_control.logging.logger import EcLogger


class PassthroughController:
    schema_id = "ec.obs.mujoco_proprio/v1"

    def __init__(
        self,
        ctrlrange: list[tuple[float, float]],
        action_schema_id: str = "ec.action.mujoco_normalized/v1",
        clip_actions: bool = True,
        fallback_action: str = "zero",
        logger: EcLogger | None = None,
    ):
        self.ctrlrange = [(float(lo), float(hi)) for lo, hi in ctrlrange]
        self.action_schema_id = action_schema_id
        self.clip_actions = clip_actions
        self.fallback_action = fallback_action
        self.action_dim = len(self.ctrlrange)
        self.logger = logger or EcLogger.null()
        self.logger.debug("embodiment.controller_constructed", action_dim=self.action_dim,
                          clip_actions=self.clip_actions, fallback_action=self.fallback_action)

    def decode_action(self, normalized_action: list[float]) -> list[float]:
        out: list[float] = []
        for i, (lo, hi) in enumerate(self.ctrlrange):
            a = float(normalized_action[i]) if i < len(normalized_action) else 0.0
            if self.clip_actions:
                a = max(-1.0, min(1.0, a))
            # [-1, 1] -> [lo, hi]
            out.append(lo + (a + 1.0) * 0.5 * (hi - lo))
        return out

    def fallback(self) -> list[float]:
        self.logger.warning("embodiment.fallback_applied", fallback_action=self.fallback_action)
        if self.fallback_action == "zero":
            # Normalized zero -> midpoint of each ctrlrange.
            return self.decode_action([0.0] * self.action_dim)
        # "hold": also midpoint here (no previous action state kept in M1).
        return self.decode_action([0.0] * self.action_dim)
