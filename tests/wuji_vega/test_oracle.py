"""IK and closed-loop physics tests for the scripted grasp oracle."""

import pytest

from embodied_control.sim.mujoco_backend import _default_headless_gl_backend

_default_headless_gl_backend()
mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from embodied_control.embodiments.passthrough import (  # noqa: E402
    PassthroughController,
)
from embodied_control.policies.wuji_vega import WujiGraspOraclePolicy  # noqa: E402
from embodied_control.policies.wuji_vega.kinematics import (  # noqa: E402
    plan_right_arm,
)
from embodied_control.sim.wuji_vega import WujiVegaGraspBackend  # noqa: E402
from embodied_control.transport.protocol import episode_key  # noqa: E402


def _wire_action(policy, observation, episode_id: int, horizon: int = 1):
    response = policy.act(
        {
            "request_id": f"test:{episode_id}",
            "episode_keys": [episode_key(0, episode_id)],
            "observations": [observation.to_wire()],
            "requested_horizon": horizon,
        }
    )
    assert response["status"] == "ok"
    return response["actions"][0]["action_chunk"]


def test_ik_waypoints_reach_requested_grasp_and_lift(scene_path):
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    cube = np.array([0.86, -0.24, 0.735])
    solutions = plan_right_arm(model, cube)
    qpos = [model.jnt_qposadr[model.joint(f"R_arm_j{i}").id] for i in range(1, 8)]
    site = model.site("right_palm").id

    data = mujoco.MjData(model)
    data.qpos[qpos] = solutions["grasp"]
    mujoco.mj_forward(model, data)
    assert data.site_xpos[site] == pytest.approx(
        cube + np.array([-0.09, 0.0, 0.085]), abs=0.008
    )

    data.qpos[qpos] = solutions["lift"]
    mujoco.mj_forward(model, data)
    assert data.site_xpos[site] == pytest.approx(
        cube + np.array([-0.09, 0.0, 0.225]), abs=0.008
    )


def test_action_chunks_drive_successful_physics_grasp(scene_path):
    backend = WujiVegaGraspBackend(str(scene_path))
    policy = WujiGraspOraclePolicy(str(scene_path), control_dt=backend.control_dt)
    controller = PassthroughController(backend.action_ctrlrange)
    try:
        episode_id = 3
        observation = backend.reset(seed=3, episode_id=episode_id)
        policy.reset({"episode_keys": [episode_key(0, episode_id)], "seed": 3})
        done = False
        steps = 0
        while steps < 650 and not done:
            chunk = _wire_action(policy, observation, episode_id, horizon=10)
            assert all(len(action) == backend.action_dim for action in chunk)
            assert all(-1.0 <= value <= 1.0 for action in chunk for value in action)
            for action in chunk:
                result = backend.step(controller.decode_action(action))
                observation = result.observation
                steps += 1
                if result.done:
                    done = True
                    break
        summary = backend.episode_summary()
        assert done
        assert summary["success"] is True
        assert summary["sustained_cube_lift"] > 0.04
        assert summary["right_hand_contact_steps"] > 0
        assert steps < 650
    finally:
        backend.close()
