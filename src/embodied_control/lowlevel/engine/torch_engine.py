"""Torch eager/TorchScript inference engine."""

from __future__ import annotations

import time

import numpy as np
import torch

from embodied_control.lowlevel.engine.base import LatencyStats


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values), q * 100.0))


class TorchEngine:
    def __init__(self, path=None, *, device: str = "cpu"):
        self.device = torch.device(device)
        self.module = None
        self._latencies: list[float] = []
        if path is not None:
            self.load(path, device)

    def load(self, path, device: str = "cpu") -> None:
        self.device = torch.device(device)
        try:
            module = torch.jit.load(str(path), map_location=self.device)
        except RuntimeError as exc:
            raise ValueError(f"policy.pt must be a TorchScript module: {path}") from exc
        self.module = module.eval().to(self.device)
        self._latencies.clear()

    def warmup(self, iters: int = 3, input_width: int | None = None) -> None:
        if self.module is None:
            raise RuntimeError("engine is not loaded")
        if input_width is None:
            raise ValueError("warmup requires input_width (the bundle's obs total width)")
        sample = torch.zeros(1, int(input_width), device=self.device)
        with torch.inference_mode():
            for _ in range(max(0, int(iters))):
                self.module(sample)
        self._sync()
        self._latencies.clear()

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def infer(self, obs: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
        if self.module is None:
            raise RuntimeError("engine is not loaded")
        value = np.asarray(obs, dtype=np.float32)
        if value.ndim != 1:
            raise ValueError(f"engine observation must be 1-D, got {value.shape}")
        tensor = torch.from_numpy(value).to(self.device).unsqueeze(0)
        self._sync()
        start = time.perf_counter()
        with torch.inference_mode():
            result = self.module(tensor)
        self._sync()
        self._latencies.append((time.perf_counter() - start) * 1000.0)
        if isinstance(result, (tuple, list)):
            result = result[0]
        array = result.detach().reshape(-1).to("cpu").numpy().astype(np.float32, copy=False)
        if out is not None:
            if out.shape != array.shape:
                raise ValueError(f"engine output shape {array.shape} != requested {out.shape}")
            np.copyto(out, array)
            return out
        return np.array(array, copy=True)

    @property
    def stats(self) -> LatencyStats:
        values = self._latencies
        return LatencyStats(
            count=len(values),
            mean_ms=float(np.mean(values)) if values else 0.0,
            p50_ms=_percentile(values, 0.50),
            p95_ms=_percentile(values, 0.95),
            p99_ms=_percentile(values, 0.99),
        )

