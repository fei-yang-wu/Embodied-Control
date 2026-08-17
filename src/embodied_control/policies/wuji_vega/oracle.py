"""Observation-driven heuristic oracle for the Vega-Wuji grasp task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from embodied_control.policies.base import Policy
from embodied_control.policies.wuji_vega.kinematics import (
    plan_right_arm,
    quintic_blend,
)
from embodied_control.transport.protocol import key_str


@dataclass(frozen=True)
class MotionPhase:
    name: str
    duration: float
    start_q: np.ndarray
    end_q: np.ndarray
    start_close: float
    end_close: float


@dataclass
class EpisodePlan:
    phases: list[MotionPhase]
    left_q: np.ndarray
    step: int = 0


def _motion_phases(
    current_right: np.ndarray, waypoints: dict[str, np.ndarray]
) -> list[MotionPhase]:
    q = waypoints
    return [
        MotionPhase("settle", 0.8, current_right, q["home"], 0.0, 0.0),
        MotionPhase("reach_transit", 1.5, q["home"], q["transit"], 0.0, 0.0),
        MotionPhase("reach_above", 1.5, q["transit"], q["above"], 0.0, 0.0),
        MotionPhase("descend", 1.2, q["above"], q["pregrasp"], 0.0, 0.0),
        MotionPhase(
            "final_approach", 0.8, q["pregrasp"], q["grasp"], 0.0, 0.0
        ),
        MotionPhase("grasp_settle", 0.7, q["grasp"], q["grasp"], 0.0, 0.0),
        MotionPhase("close", 0.7, q["grasp"], q["grasp"], 0.0, 0.95),
        MotionPhase("secure", 0.6, q["grasp"], q["grasp"], 0.95, 0.95),
        MotionPhase("lift", 3.0, q["grasp"], q["lift"], 0.95, 0.95),
        MotionPhase("hold", 0.8, q["lift"], q["lift"], 0.95, 0.95),
    ]


class WujiGraspOraclePolicy(Policy):
    policy_type = "wuji_grasp_oracle"

    def __init__(
        self,
        model_path: str,
        control_dt: float = 0.02,
        action_schema_id: str = "ec.action.wuji_vega_joint_position_normalized/v1",
        seed: int = 0,
        max_action_horizon: int = 32,
    ):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Vega-Wuji scene not found: {self.model_path}")
        if control_dt <= 0:
            raise ValueError("control_dt must be > 0")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.control_dt = float(control_dt)
        super().__init__(
            action_dim=self.model.nu,
            action_schema_id=action_schema_id,
            seed=seed,
            max_action_horizon=max_action_horizon,
        )
        self._actuator_names = [
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            for actuator_id in range(self.model.nu)
        ]
        self._plans: dict[str, EpisodePlan] = {}

    def describe(self) -> dict:
        description = super().describe()
        description.update(
            {
                "model_family": "scripted_ik_oracle",
                "model_path": str(self.model_path),
                "control_dt": self.control_dt,
                "planner": "clearance waypoints + sequential damped least-squares IK",
            }
        )
        return description

    def reset(self, req: dict) -> dict:
        response = super().reset(req)
        for key in req.get("episode_keys", []):
            self._plans.pop(key_str(key), None)
        return response

    def _action(
        self, env_id: int, episode_id: int, obs: dict
    ) -> list[float]:
        return self._action_chunk(env_id, episode_id, 1, obs)[0]

    def _action_chunk(
        self,
        env_id: int,
        episode_id: int,
        horizon: int,
        obs: dict,
    ) -> list[list[float]]:
        episode_key = key_str([env_id, episode_id])
        plan = self._plans.get(episode_key)
        if plan is None:
            plan = self._build_episode_plan(obs)
            self._plans[episode_key] = plan
        chunk = []
        for _ in range(horizon):
            chunk.append(self._command_at(plan, plan.step * self.control_dt))
            plan.step += 1
        return chunk

    def _build_episode_plan(self, obs: dict) -> EpisodePlan:
        proprio = obs.get("proprio") or {}
        state = dict(
            zip(proprio.get("names", []), proprio.get("values", []), strict=True)
        )
        task = obs.get("task") or {}
        if task.get("actuator_names") != self._actuator_names:
            raise ValueError("observation actuator order does not match oracle model")
        observation_dt = float(task.get("control_dt", -1.0))
        if not np.isclose(observation_dt, self.control_dt, atol=1e-9):
            raise ValueError(
                f"observation control_dt {observation_dt} does not match oracle "
                f"control_dt {self.control_dt}"
            )
        cube = np.array([state[f"cube/{axis}"] for axis in "xyz"], dtype=float)
        current_right = np.array(
            [state[f"joint/R_arm_j{i}/position"] for i in range(1, 8)],
            dtype=float,
        )
        current_left = np.array(
            [state[f"joint/L_arm_j{i}/position"] for i in range(1, 8)],
            dtype=float,
        )
        return EpisodePlan(
            phases=_motion_phases(current_right, plan_right_arm(self.model, cube)),
            left_q=current_left,
        )

    def _command_at(self, plan: EpisodePlan, time_s: float) -> list[float]:
        phase, phase_time = self._phase_at(plan.phases, time_s)
        blend = quintic_blend(phase_time / phase.duration)
        right_q = phase.start_q + blend * (phase.end_q - phase.start_q)
        close = phase.start_close + blend * (phase.end_close - phase.start_close)
        targets = np.zeros(self.model.nu, dtype=float)
        for side, values in (("L", plan.left_q), ("R", right_q)):
            for index, value in enumerate(values, start=1):
                targets[
                    self.model.actuator(f"{side}_arm_j{index}_position").id
                ] = value
        closure = {
            "THJ": (0.8, 0.0, 0.8, 0.7),
            "FFJ": (1.2, 0.0, 1.5, 1.0),
            "MFJ": (1.2, 0.0, 1.5, 1.0),
            "RFJ": (1.2, 0.0, 1.5, 1.0),
            "LFJ": (1.2, 0.0, 1.5, 1.0),
        }
        for joint, values in closure.items():
            for index, value in enumerate(values):
                targets[self.model.actuator(f"r_{joint}{index}").id] = value * close
        lower = self.model.actuator_ctrlrange[:, 0]
        upper = self.model.actuator_ctrlrange[:, 1]
        normalized = 2.0 * (targets - lower) / (upper - lower) - 1.0
        return np.clip(normalized, -1.0, 1.0).tolist()

    @staticmethod
    def _phase_at(
        phases: list[MotionPhase], time_s: float
    ) -> tuple[MotionPhase, float]:
        elapsed = 0.0
        for phase in phases:
            if time_s < elapsed + phase.duration:
                return phase, time_s - elapsed
            elapsed += phase.duration
        return phases[-1], phases[-1].duration


__all__ = ["MotionPhase", "WujiGraspOraclePolicy"]
