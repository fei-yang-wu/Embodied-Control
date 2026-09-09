import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pytest

pytest.importorskip("mujoco")

from embodied_control.lowlevel.bundle import ActionContract
from embodied_control.lowlevel.unitree_pose import (
    isaac_to_sdk,
    render_pose_comparison,
    sdk_to_isaac,
)


def _action():
    return ActionContract(
        width=2,
        isaac_joint_names=["alpha", "beta"],
        sdk_joint_names=["beta", "alpha"],
        isaac_to_sdk=[1, 0],
        default_joint_pos=[0.0, 0.0],
        action_scale=[1.0, 1.0],
        stiffness=[1.0, 1.0],
        damping=[0.1, 0.1],
    )


def test_sdk_isaac_mapping_roundtrip():
    sdk = np.array([0.25, -0.5])

    isaac = sdk_to_isaac(_action(), sdk)

    np.testing.assert_array_equal(isaac, [-0.5, 0.25])
    np.testing.assert_array_equal(isaac_to_sdk(_action(), isaac), sdk)


def test_render_pose_checks_names_and_mujoco_readback(tmp_path):
    model = tmp_path / "two_joint.xml"
    model.write_text(
        """
<mujoco>
  <worldbody>
    <body name="pelvis">
      <freejoint/>
      <geom type="sphere" size="0.08"/>
      <body pos="0 0 0.1">
        <joint name="alpha" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
        <body pos="0 0 0.2">
          <joint name="beta" type="hinge" axis="1 0 0"/>
          <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    )
    image = tmp_path / "pose.png"

    report = render_pose_comparison(
        _action(),
        model,
        np.array([0.25, -0.5]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        image,
    )

    assert report["status"] == "pass"
    assert report["mapping_name_mismatches"] == []
    assert report["roundtrip_max_error"] == 0.0
    assert report["mujoco_readback_max_error"] == 0.0
    assert image.is_file()
