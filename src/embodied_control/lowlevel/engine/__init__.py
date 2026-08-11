"""Inference engines for policy bundles.

`TorchEngine` needs torch, which only the `lowlevel` feature env carries; the
light default env must still import this package, so the symbol is loaded
lazily on first attribute access.
"""

from embodied_control.lowlevel.engine.base import Engine, LatencyStats

__all__ = ["Engine", "LatencyStats", "TorchEngine"]


def __getattr__(name: str):
    if name == "TorchEngine":
        from embodied_control.lowlevel.engine.torch_engine import TorchEngine

        return TorchEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
