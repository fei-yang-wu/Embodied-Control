"""Plugin registries for sim backends and embodiment controllers.

Plain module-level dicts (design-appropriate for a single-lab, one-consumer repo).
Entry-point discovery for external packages is deferred.
"""

from __future__ import annotations

from typing import Callable

_SIM_BACKENDS: dict[str, Callable] = {}
_EMBODIMENTS: dict[str, Callable] = {}


def register_sim_backend(name: str, factory: Callable) -> None:
    _SIM_BACKENDS[name] = factory


def get_sim_backend(name: str) -> Callable:
    try:
        return _SIM_BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown sim backend {name!r}; registered: {sorted(_SIM_BACKENDS)}"
        ) from None


def register_embodiment(name: str, factory: Callable) -> None:
    _EMBODIMENTS[name] = factory


def get_embodiment(name: str) -> Callable:
    try:
        return _EMBODIMENTS[name]
    except KeyError:
        raise ValueError(
            f"unknown embodiment adapter {name!r}; registered: {sorted(_EMBODIMENTS)}"
        ) from None


def _register_builtins() -> None:
    if _SIM_BACKENDS:
        return

    def _mujoco(job):
        from embodied_control.sim.mujoco_backend import MujocoStepBackend

        cfg = job.sim.backend_config or {}
        kwargs = {}
        if "success_threshold" in cfg:
            kwargs["success_threshold"] = float(cfg["success_threshold"])
        if "target_radius_range" in cfg:
            lo, hi = cfg["target_radius_range"]
            kwargs["target_radius_range"] = (float(lo), float(hi))
        return MujocoStepBackend(
            model=job.sim.model, model_path=job.sim.model_path, **kwargs
        )

    def _passthrough(job, ctrlrange, logger=None):
        from embodied_control.embodiments.passthrough import PassthroughController

        return PassthroughController(
            ctrlrange=ctrlrange,
            action_schema_id=job.embodiment.action_schema_id,
            clip_actions=job.embodiment.clip_actions,
            fallback_action=job.embodiment.fallback_action,
            logger=logger,
        )

    register_sim_backend("mujoco", _mujoco)
    register_embodiment("passthrough", _passthrough)


_register_builtins()
