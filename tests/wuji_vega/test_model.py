"""Portable MJCF composition and simulator-contract tests."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from embodied_control.sim.mujoco_backend import RenderError, _default_headless_gl_backend

_default_headless_gl_backend()
mujoco = pytest.importorskip("mujoco")
np = pytest.importorskip("numpy")

from embodied_control.sim.wuji_vega import (  # noqa: E402
    ROBOT_ASSET,
    WUJI_HAND_ASSET,
    WujiVegaGraspBackend,
)


def test_robot_and_scene_are_separate_portable_assets(scene_path):
    scene_root = ET.parse(scene_path).getroot()
    include = scene_root.find("include")
    assert include is not None
    assert include.attrib["file"] == "vega_u_wuji_v2_beta1_with_mount.xml"
    assert scene_root.find("visual") is not None
    assert scene_root.findall("./asset/mesh") == []
    assert [
        texture.attrib["name"] for texture in scene_root.findall("./asset/texture")
    ] == ["sky_gradient"]
    assert scene_root.find("actuator") is None
    assert len(scene_root.findall("./worldbody/light")) == 3
    scene_bodies = {
        body.attrib["name"] for body in scene_root.findall("./worldbody/body")
    }
    assert scene_bodies == {"desk", "grasp_cube"}

    robot_path = scene_path.parent / include.attrib["file"]
    robot_root = ET.parse(robot_path).getroot()
    assert robot_root.attrib["model"] == "vega_u_wuji_v2_beta1_with_mount"
    assert robot_root.find("visual") is None
    assert robot_root.findall("./worldbody/light") == []
    custom_text = {
        text.attrib["name"]: text.attrib["data"]
        for text in robot_root.findall("./custom/text")
    }
    assert custom_text["robot_assembly"] == ROBOT_ASSET
    assert custom_text["wuji_hand_asset"] == WUJI_HAND_ASSET
    assert custom_text["wuji_hand_source_models"] == (
        "left_with_mount.xml; right_with_mount.xml"
    )
    for mesh in robot_root.findall("./asset/mesh"):
        relative = Path(mesh.attrib["file"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        assert (scene_path.parent / relative).is_file()

    robot_model = mujoco.MjModel.from_xml_path(str(robot_path))
    assert robot_model.nu == 59
    assert robot_model.nlight == 0
    assert mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_BODY, "desk") == -1
    assert (
        mujoco.mj_name2id(robot_model, mujoco.mjtObj.mjOBJ_BODY, "grasp_cube") == -1
    )


def test_backend_mounts_and_seeded_reset(scene_path):
    backend = WujiVegaGraspBackend(str(scene_path), cube_xy_noise=0.01)
    try:
        first = backend.reset(seed=7, episode_id=0)
        first_state = dict(zip(first.proprio_names, first.proprio_values, strict=True))
        repeat = backend.reset(seed=7, episode_id=1)
        repeat_state = dict(zip(repeat.proprio_names, repeat.proprio_values, strict=True))
        other = backend.reset(seed=8, episode_id=2)
        other_state = dict(zip(other.proprio_names, other.proprio_values, strict=True))

        assert backend.action_dim == 59
        assert len(first.proprio_names) == 2 * backend.action_dim + 9
        assert first.task["actuator_names"][0] == "Lift_position"
        assert first.task["actuator_names"][-1] == "r_LFJ3"
        assert first.task["robot_asset"] == ROBOT_ASSET
        assert first.task["wuji_hand_asset"] == WUJI_HAND_ASSET
        assert backend.mount_errors() == pytest.approx({"left": 0.0, "right": 0.0})
        expected_mount_rotation = np.diag([1.0, -1.0, -1.0])
        for side in ("L", "R"):
            ee_rotation = backend.mj_data.xmat[
                backend.mj_model.body(f"{side}_ee").id
            ].reshape(3, 3)
            palm_rotation = backend.mj_data.site_xmat[
                backend.mj_model.site(
                    "left_palm" if side == "L" else "right_palm"
                ).id
            ].reshape(3, 3)
            assert ee_rotation.T @ palm_rotation == pytest.approx(
                expected_mount_rotation, abs=1e-8
            )
        assert [first_state[f"cube/{axis}"] for axis in "xyz"] == pytest.approx(
            [repeat_state[f"cube/{axis}"] for axis in "xyz"]
        )
        assert [first_state[f"cube/{axis}"] for axis in "xy"] != pytest.approx(
            [other_state[f"cube/{axis}"] for axis in "xy"]
        )

        desk = backend.mj_model.geom("desk_top").id
        desk_body = backend.mj_model.geom_bodyid[desk]
        desk_top = (
            backend.mj_model.body_pos[desk_body, 2]
            + backend.mj_model.geom_pos[desk, 2]
            + backend.mj_model.geom_size[desk, 2]
        )
        desk_near_edge = (
            backend.mj_model.body_pos[desk_body, 0]
            + backend.mj_model.geom_pos[desk, 0]
            - backend.mj_model.geom_size[desk, 0]
        )
        assert desk_top == pytest.approx(0.70)
        assert desk_near_edge == pytest.approx(0.25)
        assert backend.mj_model.nlight >= 3
        assert other_state["cube/z"] == pytest.approx(0.735)

        hold = backend.mj_data.ctrl.copy().tolist()
        for _ in range(30):
            result = backend.step(hold)
        assert not result.done
        assert backend.episode_summary()["success"] is False
    finally:
        backend.close()


def test_backend_camera_renders_lit_scene_without_collision_tint(scene_path):
    backend = WujiVegaGraspBackend(str(scene_path))
    try:
        backend.reset(seed=0, episode_id=0)
        try:
            frame = backend.render_frame(width=320, height=180)
        except RenderError as exc:
            pytest.skip(f"offscreen rendering unavailable: {exc}")
        assert frame.shape == (180, 320, 3)
        assert frame.dtype == np.uint8
        assert 10.0 < float(frame.mean()) < 245.0
        assert float(frame.std()) > 10.0
    finally:
        backend.close()
