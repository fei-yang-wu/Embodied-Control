"""Stepped MuJoCo backend for the Vega U + Wuji Hand table-cube task."""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

from embodied_control.logging.logger import EcLogger
from embodied_control.sim.base import Observation, StepResult
from embodied_control.sim.mujoco_backend import RenderError
from embodied_control.sim.wuji_vega.constants import (
    LEFT_STANDBY,
    RIGHT_HOME,
    ROBOT_ASSET,
    WUJI_HAND_ASSET,
)


class WujiVegaGraspBackend:
    name = "wuji_vega_grasp"

    def __init__(
        self,
        model_path: str,
        frame_skip: int = 10,
        cube_xy_noise: float = 0.0,
        lift_threshold: float = 0.04,
        success_hold_s: float = 0.5,
        camera_name: str = "front_camera",
        logger: EcLogger | None = None,
    ):
        import mujoco
        import numpy as np

        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Vega-Wuji scene not found: {path}")
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if cube_xy_noise < 0:
            raise ValueError("cube_xy_noise must be >= 0")
        if lift_threshold <= 0 or success_hold_s <= 0:
            raise ValueError("lift_threshold and success_hold_s must be > 0")

        self.logger = logger or EcLogger.null()
        self._mj = mujoco
        self._np = np
        self.model_path = path
        self.mj_model = mujoco.MjModel.from_xml_path(str(path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.frame_skip = int(frame_skip)
        self.control_dt = float(self.mj_model.opt.timestep * self.frame_skip)
        self.cube_xy_noise = float(cube_xy_noise)
        self.lift_threshold = float(lift_threshold)
        self.success_hold_s = float(success_hold_s)
        self.success_hold_steps = max(1, math.ceil(success_hold_s / self.control_dt))
        self.camera_name = camera_name

        self._cube_bid = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "grasp_cube")
        cube_jid = self._required_id(
            mujoco.mjtObj.mjOBJ_JOINT, "grasp_cube_freejoint"
        )
        self._cube_qadr = int(self.mj_model.jnt_qposadr[cube_jid])
        self._cube_geom_id = self._required_id(
            mujoco.mjtObj.mjOBJ_GEOM, "grasp_cube_geom"
        )
        self._right_palm_sid = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, "right_palm"
        )
        self._left_palm_sid = self._required_id(
            mujoco.mjtObj.mjOBJ_SITE, "left_palm"
        )
        self._right_ee_bid = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "R_ee")
        self._left_ee_bid = self._required_id(mujoco.mjtObj.mjOBJ_BODY, "L_ee")
        self._actuator_joint_ids = self.mj_model.actuator_trnid[:, 0].astype(int)
        self._actuator_qpos = self.mj_model.jnt_qposadr[
            self._actuator_joint_ids
        ].astype(int)
        self._actuator_dofs = self.mj_model.jnt_dofadr[
            self._actuator_joint_ids
        ].astype(int)
        self._actuator_names = [
            mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            for actuator_id in range(self.mj_model.nu)
        ]
        self._joint_names = [
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(jid))
            for jid in self._actuator_joint_ids
        ]
        self._right_hand_geom_ids = {
            geom_id
            for geom_id in range(self.mj_model.ngeom)
            if (
                name := mujoco.mj_id2name(
                    self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                )
            )
            and name.startswith("r_")
        }
        cameras = {
            mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(self.mj_model.ncam)
        }
        if camera_name not in cameras:
            raise ValueError(f"camera {camera_name!r} not found in {path}")

        self._renderer = None
        self._render_size: tuple[int, int] | None = None
        self._render_options = mujoco.MjvOption()
        self._render_options.sitegroup[:] = 0
        self._render_options.geomgroup[4] = 0
        self._episode_id = -1
        self._steps = 0
        self._initial_cube_z = 0.0
        self._last_cube_lift = 0.0
        self._max_cube_lift = 0.0
        self._min_palm_cube_distance = math.inf
        self._right_hand_contact_steps = 0
        self._cube_contact_steps = 0
        self._lift_window: deque[float] = deque(maxlen=self.success_hold_steps)
        self._success = False

    @property
    def action_dim(self) -> int:
        return int(self.mj_model.nu)

    @property
    def action_ctrlrange(self) -> list[tuple[float, float]]:
        return [
            (float(lower), float(upper))
            for lower, upper in self.mj_model.actuator_ctrlrange
        ]

    def task_id(self) -> str:
        return "wuji_vega_pick_cube"

    def reset(self, seed: int, episode_id: int) -> Observation:
        np, mujoco = self._np, self._mj
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        initial_ctrl = np.zeros(self.mj_model.nu, dtype=float)
        for side, values in (("L", LEFT_STANDBY), ("R", RIGHT_HOME)):
            for index, value in enumerate(values, start=1):
                actuator_id = self.mj_model.actuator(
                    f"{side}_arm_j{index}_position"
                ).id
                initial_ctrl[actuator_id] = value
        self.mj_data.qpos[self._actuator_qpos] = initial_ctrl
        self.mj_data.qvel[:] = 0.0
        self.mj_data.ctrl[:] = initial_ctrl

        cube_qpos = self.mj_data.qpos[self._cube_qadr : self._cube_qadr + 7]
        cube_qpos[:2] += rng.uniform(-self.cube_xy_noise, self.cube_xy_noise, size=2)
        cube_qpos[3:] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._episode_id = int(episode_id)
        self._steps = 0
        self._initial_cube_z = float(self.mj_data.xpos[self._cube_bid, 2])
        self._last_cube_lift = 0.0
        self._max_cube_lift = 0.0
        self._min_palm_cube_distance = self._palm_cube_distance()
        self._right_hand_contact_steps = 0
        self._cube_contact_steps = 0
        self._lift_window.clear()
        self._success = False
        return self._observation()

    def step(self, ctrl: list[float]) -> StepResult:
        np, mujoco = self._np, self._mj
        command = np.asarray(ctrl, dtype=float)
        if command.shape != (self.mj_model.nu,):
            raise ValueError(
                f"expected {self.mj_model.nu} controls, got shape {command.shape}"
            )
        self.mj_data.ctrl[:] = np.clip(
            command,
            self.mj_model.actuator_ctrlrange[:, 0],
            self.mj_model.actuator_ctrlrange[:, 1],
        )

        cube_contact = False
        right_hand_contact = False
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.mj_model, self.mj_data)
            sub_cube, sub_hand = self._contacts()
            cube_contact = cube_contact or sub_cube
            right_hand_contact = right_hand_contact or sub_hand

        self._steps += 1
        self._cube_contact_steps += int(cube_contact)
        self._right_hand_contact_steps += int(right_hand_contact)
        previous_lift = self._last_cube_lift
        self._last_cube_lift = float(
            self.mj_data.xpos[self._cube_bid, 2] - self._initial_cube_z
        )
        self._max_cube_lift = max(self._max_cube_lift, self._last_cube_lift)
        distance = self._palm_cube_distance()
        self._min_palm_cube_distance = min(self._min_palm_cube_distance, distance)
        self._lift_window.append(self._last_cube_lift)
        self._success = bool(
            len(self._lift_window) == self.success_hold_steps
            and self._sustained_lift() > self.lift_threshold
            and self._right_hand_contact_steps > 0
        )
        unstable = bool(not np.all(np.isfinite(self.mj_data.qpos)))
        if unstable:
            self.logger.warning(
                "sim.instability_detected",
                episode_id=self._episode_id,
                step=self._steps,
            )
        reward = (
            -distance
            + 20.0 * (self._last_cube_lift - previous_lift)
            + (0.1 if right_hand_contact else 0.0)
            + (5.0 if self._success else 0.0)
        )
        return StepResult(
            observation=self._observation(),
            reward=float(reward),
            done=self._success or unstable,
            info={
                "cube_lift": self._last_cube_lift,
                "palm_cube_distance": distance,
                "right_hand_contact": right_hand_contact,
                "success": self._success,
                "unstable": unstable,
            },
        )

    def episode_summary(self) -> dict:
        metrics = {
            "success": float(self._success),
            "final_cube_lift": float(self._last_cube_lift),
            "max_cube_lift": float(self._max_cube_lift),
            "sustained_cube_lift": float(self._sustained_lift()),
            "min_palm_cube_distance": float(self._min_palm_cube_distance),
            "right_hand_contact_steps": float(self._right_hand_contact_steps),
            "cube_contact_steps": float(self._cube_contact_steps),
        }
        return {
            **metrics,
            "success": self._success,
            "steps": self._steps,
            "lift_threshold": self.lift_threshold,
            "success_hold_s": self.success_hold_s,
            "control_dt": self.control_dt,
            "robot_asset": ROBOT_ASSET,
            "wuji_hand_asset": WUJI_HAND_ASSET,
            "metrics": metrics,
        }

    def render_frame(self, width: int = 640, height: int = 360):
        size = (width, height)
        if self._renderer is None or self._render_size != size:
            if self._renderer is not None:
                self._renderer.close()
            try:
                self._renderer = self._mj.Renderer(
                    self.mj_model, height=height, width=width
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.error("sim.renderer_init_failed", reason=str(exc))
                raise RenderError(f"failed to create MuJoCo renderer: {exc}") from exc
            self._render_size = size
        self._renderer.update_scene(
            self.mj_data,
            camera=self.camera_name,
            scene_option=self._render_options,
        )
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.mj_data = None
        self.mj_model = None

    def mount_errors(self) -> dict[str, float]:
        return {
            "left": float(
                self._np.linalg.norm(
                    self.mj_data.site_xpos[self._left_palm_sid]
                    - self.mj_data.xpos[self._left_ee_bid]
                )
            ),
            "right": float(
                self._np.linalg.norm(
                    self.mj_data.site_xpos[self._right_palm_sid]
                    - self.mj_data.xpos[self._right_ee_bid]
                )
            ),
        }

    def _observation(self) -> Observation:
        qpos = self.mj_data.qpos[self._actuator_qpos]
        qvel = self.mj_data.qvel[self._actuator_dofs]
        cube = self.mj_data.xpos[self._cube_bid]
        right_palm = self.mj_data.site_xpos[self._right_palm_sid]
        left_palm = self.mj_data.site_xpos[self._left_palm_sid]
        names = [f"joint/{name}/position" for name in self._joint_names]
        names += [f"joint/{name}/velocity" for name in self._joint_names]
        names += ["cube/x", "cube/y", "cube/z"]
        names += ["right_palm/x", "right_palm/y", "right_palm/z"]
        names += ["left_palm/x", "left_palm/y", "left_palm/z"]
        values = qpos.tolist() + qvel.tolist()
        values += cube.tolist() + right_palm.tolist() + left_palm.tolist()
        return Observation(
            env_id=0,
            episode_id=self._episode_id,
            proprio_names=names,
            proprio_values=[float(value) for value in values],
            task={
                "task_id": self.task_id(),
                "language_instruction": (
                    "pick up the blue cube from the table with the right hand"
                ),
                "control_dt": self.control_dt,
                "actuator_names": self._actuator_names,
                "robot_asset": ROBOT_ASSET,
                "wuji_hand_asset": WUJI_HAND_ASSET,
            },
        )

    def _contacts(self) -> tuple[bool, bool]:
        cube_contact = False
        right_hand_contact = False
        for index in range(self.mj_data.ncon):
            contact = self.mj_data.contact[index]
            if contact.geom1 == self._cube_geom_id:
                other = int(contact.geom2)
            elif contact.geom2 == self._cube_geom_id:
                other = int(contact.geom1)
            else:
                continue
            cube_contact = True
            right_hand_contact = right_hand_contact or other in self._right_hand_geom_ids
        return cube_contact, right_hand_contact

    def _palm_cube_distance(self) -> float:
        return float(
            self._np.linalg.norm(
                self.mj_data.site_xpos[self._right_palm_sid]
                - self.mj_data.xpos[self._cube_bid]
            )
        )

    def _sustained_lift(self) -> float:
        return (
            float(min(self._lift_window))
            if self._lift_window
            else self._last_cube_lift
        )

    def _required_id(self, object_type, name: str) -> int:
        object_id = int(self._mj.mj_name2id(self.mj_model, object_type, name))
        if object_id < 0:
            raise ValueError(
                f"required MuJoCo object {name!r} missing from {self.model_path}"
            )
        return object_id


__all__ = ["WujiVegaGraspBackend"]
