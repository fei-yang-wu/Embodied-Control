"""In-process MuJoCo stepped backend (the reacher task).

Imports mujoco/numpy lazily so the module can be imported in the light host env
for type-checking/tests; the heavy import only happens when a backend is built
(in the ``sim`` pixi environment).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from embodied_control.logging.logger import EcLogger
from embodied_control.sim.base import Observation, StepResult
from embodied_control.sim.models import get_model_xml


class RenderError(RuntimeError):
    """Offscreen rendering unavailable (e.g. no EGL/OSMesa on a headless node)."""


def _default_headless_gl_backend() -> None:
    """Default MuJoCo's GL backend to EGL on headless Linux (shared workstations,
    HPC compute nodes) so offscreen rendering works without a display or an
    ``export MUJOCO_GL=egl`` step. GLFW (mujoco's default) requires a real
    display/X server and is the common failure mode on lab machines and CI.
    ``setdefault`` respects an explicit user override; harmless if unused
    (rollout.record_video defaults to False).

    MuJoCo selects its GL backend the first time it is imported anywhere in
    the process and reuses that choice thereafter, so this must run before
    that first import -- called both at THIS module's import time (below) and
    from ``ec doctor`` (which otherwise imports raw ``mujoco`` first). Any
    other code path that needs a guarantee (e.g. a test importing ``mujoco``
    directly) should call this before doing so.
    """
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")


_default_headless_gl_backend()  # set as early as possible: as soon as this module loads


class MujocoStepBackend:
    name = "mujoco"

    def __init__(
        self,
        model: str = "reacher2",
        model_path: str | None = None,
        success_threshold: float = 0.03,
        target_radius_range: tuple[float, float] = (0.05, 0.19),
        logger: EcLogger | None = None,
    ):
        import mujoco  # local import: only needed in the sim env
        import numpy as np

        # The backend is constructed before the run's ExecutionPlan/ArtifactStore
        # exist (its action_dim/ctrlrange are needed to build the plan), so it
        # starts with a null logger; the runner attaches the real scoped logger
        # (``backend.logger = logger.child("sim")``) once one exists.
        self.logger = logger or EcLogger.null()

        self._mj = mujoco
        self._np = np
        if model_path:
            xml = Path(model_path).read_text()
        else:
            xml = get_model_xml(model)
        self._model_name = model if not model_path else Path(model_path).stem
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.success_threshold = float(success_threshold)
        self.target_radius_range = target_radius_range

        self._target_bid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._fingertip_sid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "fingertip"
        )
        self._target_sid = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "target"
        )

        self._episode_id = -1
        self._steps = 0
        self._min_dist = math.inf
        self._last_dist = math.inf

        self._renderer = None
        self._render_size: tuple[int, int] | None = None

    # --- spec ------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return int(self.mj_model.nu)

    @property
    def action_ctrlrange(self) -> list[tuple[float, float]]:
        rng = self.mj_model.actuator_ctrlrange
        return [(float(lo), float(hi)) for lo, hi in rng]

    def task_id(self) -> str:
        return f"reacher2_reach"

    # --- lifecycle -------------------------------------------------------
    def reset(self, seed: int, episode_id: int) -> Observation:
        mujoco, np = self._mj, self._np
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        # Small random initial joint angles.
        self.mj_data.qpos[:] = rng.uniform(-0.2, 0.2, size=self.mj_model.nq)
        self.mj_data.qvel[:] = 0.0
        # Randomize the target within a reachable annulus.
        lo, hi = self.target_radius_range
        radius = float(rng.uniform(lo, hi))
        angle = float(rng.uniform(-math.pi, math.pi))
        self.mj_model.body_pos[self._target_bid][0] = radius * math.cos(angle)
        self.mj_model.body_pos[self._target_bid][1] = radius * math.sin(angle)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._episode_id = episode_id
        self._steps = 0
        self._min_dist = self._distance()
        self._last_dist = self._min_dist
        return self._observation()

    def step(self, ctrl: list[float]) -> StepResult:
        mujoco, np = self._mj, self._np
        self.mj_data.ctrl[:] = np.asarray(ctrl, dtype=float)[: self.mj_model.nu]
        mujoco.mj_step(self.mj_model, self.mj_data)
        self._steps += 1
        dist = self._distance()
        self._last_dist = dist
        self._min_dist = min(self._min_dist, dist)
        reward = -dist
        # Fixed-horizon task; the runner enforces max_steps. Detect blow-ups.
        done = bool(not np.all(np.isfinite(self.mj_data.qpos)))
        if done:
            self.logger.warning(
                "sim.instability_detected", episode_id=self._episode_id, step=self._steps
            )
        return StepResult(
            observation=self._observation(),
            reward=float(reward),
            done=done,
            info={"distance": dist, "failed": done},
        )

    def episode_summary(self) -> dict:
        # Success = the fingertip came within threshold of the target at any point
        # in the episode. A "reached at least once" criterion is the fair measure
        # for a non-holding baseline (a random policy explores but won't park on
        # the target), and cleanly separates random from the zero baseline.
        success = bool(self._min_dist < self.success_threshold)
        metrics = {
            "success": float(success),
            "final_distance": float(self._last_dist),
            "min_distance": float(self._min_dist),
        }
        return {
            "success": success,
            "final_distance": float(self._last_dist),
            "min_distance": float(self._min_dist),
            "steps": int(self._steps),
            "success_threshold": self.success_threshold,
            "metrics": metrics,
        }

    def render_frame(self, width: int = 320, height: int = 240):
        """Render the current sim state via a fixed top-down camera (RGB uint8 HxWx3).

        The renderer is created lazily and cached (offscreen contexts are
        expensive) so this is cheap to call every step during a rollout.
        Raises :class:`RenderError` if offscreen rendering is unavailable
        (e.g. no EGL/OSMesa on a headless node) -- callers should catch this
        once and disable video for the rest of the run rather than fail it.
        """
        mujoco = self._mj
        size = (width, height)
        if self._renderer is None or self._render_size != size:
            if self._renderer is not None:
                self._renderer.close()
            try:
                self._renderer = mujoco.Renderer(self.mj_model, height=height, width=width)
            except Exception as exc:  # noqa: BLE001 - GL/EGL init failures vary by platform
                self.logger.error("sim.renderer_init_failed", reason=str(exc))
                raise RenderError(
                    f"failed to create MuJoCo offscreen renderer "
                    f"(no EGL/OSMesa on this host?): {exc}"
                ) from exc
            self._render_size = size
        self._renderer.update_scene(self.mj_data, camera="topdown")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.mj_data = None
        self.mj_model = None

    # --- helpers ---------------------------------------------------------
    def _distance(self) -> float:
        np = self._np
        fp = self.mj_data.site_xpos[self._fingertip_sid][:2]
        tg = self.mj_data.site_xpos[self._target_sid][:2]
        return float(np.linalg.norm(fp - tg))

    def _observation(self) -> Observation:
        q = self.mj_data.qpos
        qd = self.mj_data.qvel
        fp = self.mj_data.site_xpos[self._fingertip_sid][:2]
        tg = self.mj_data.site_xpos[self._target_sid][:2]
        names = [
            "cos_j1", "sin_j1", "cos_j2", "sin_j2",
            "vel_j1", "vel_j2",
            "target_x", "target_y",
            "fingertip_x", "fingertip_y",
            "delta_x", "delta_y",
        ]
        values = [
            math.cos(float(q[0])), math.sin(float(q[0])),
            math.cos(float(q[1])), math.sin(float(q[1])),
            float(qd[0]), float(qd[1]),
            float(tg[0]), float(tg[1]),
            float(fp[0]), float(fp[1]),
            float(tg[0] - fp[0]), float(tg[1] - fp[1]),
        ]
        return Observation(
            env_id=0,
            episode_id=self._episode_id,
            proprio_names=names,
            proprio_values=values,
            task={"task_id": self.task_id(), "language_instruction": "reach the red target"},
        )
