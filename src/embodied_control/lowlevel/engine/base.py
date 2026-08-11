"""Inference engine protocol and latency accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    count: int = 0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0


class Engine(Protocol):
    def load(self, path, device: str = "cpu") -> None: ...
    def warmup(self, iters: int = 3, input_width: int | None = None) -> None: ...
    def infer(self, obs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray: ...
    @property
    def stats(self) -> LatencyStats: ...

