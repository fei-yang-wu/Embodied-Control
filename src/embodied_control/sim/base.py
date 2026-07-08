"""Stepped simulator backend contract.

A stepped backend is host-driven: the runner owns the rollout loop and calls
``reset``/``step`` directly. This is scoped (design D2) to lightweight in-host
sims like plain MuJoCo — no heavyweight simulator import is forced on a delegated
evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Observation:
    """A single-env observation passed to the policy (normalized wire form)."""

    env_id: int
    episode_id: int
    proprio_names: list[str]
    proprio_values: list[float]
    task: dict = field(default_factory=dict)

    def to_wire(self) -> dict:
        return {
            "env_id": self.env_id,
            "episode_id": self.episode_id,
            "proprio": {"names": self.proprio_names, "values": self.proprio_values},
            "task": self.task,
        }


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class SteppedBackend(Protocol):
    name: str

    @property
    def action_dim(self) -> int: ...

    @property
    def action_ctrlrange(self) -> list[tuple[float, float]]: ...

    def reset(self, seed: int, episode_id: int) -> Observation: ...

    def step(self, ctrl: list[float]) -> StepResult: ...

    def episode_summary(self) -> dict: ...

    def task_id(self) -> str: ...

    def close(self) -> None: ...
