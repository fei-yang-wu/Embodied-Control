"""Built-in MJCF models, shipped as strings so the smoke test needs no asset downloads.

``reacher2`` is a 2-DOF planar arm with a randomizable target site. It is small,
deterministic under a fixed seed, and gives a non-trivial success metric (did the
fingertip reach the target?) so the metrics/artifact pipeline has real signal —
without pulling in robosuite/LIBERO assets.
"""

from __future__ import annotations

# Planar 2-link reacher. Gravity is disabled so the arm stays in-plane and a
# zero-torque (zero policy) baseline holds pose; a random policy explores.
REACHER2 = """
<mujoco model="reacher2">
  <!-- MuJoCo defaults to angle="degree"; we author joint ranges in radians. -->
  <compiler angle="radian"/>
  <option timestep="0.01" gravity="0 0 0" integrator="implicitfast"/>
  <default>
    <!-- armature regularizes the (otherwise near-zero) rotational inertia of the
         thin links; damping keeps the position servo well-behaved. -->
    <joint type="hinge" axis="0 0 1" damping="1.5" armature="0.05"/>
    <geom contype="0" conaffinity="0" density="1000"/>
  </default>
  <worldbody>
    <light pos="0 0 1"/>
    <!-- Fixed world camera looking straight down (-z), for offscreen rendering. -->
    <camera name="topdown" pos="0 0 0.55" xyaxes="1 0 0 0 1 0"/>
    <geom name="ground" type="plane" size="0.5 0.5 0.05" rgba="0.9 0.9 0.9 1"/>
    <body name="link1" pos="0 0 0.02">
      <joint name="j1" limited="true" range="-3.14159 3.14159"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.1 0 0" size="0.012" rgba="0.2 0.4 0.8 1"/>
      <body name="link2" pos="0.1 0 0">
        <joint name="j2" limited="true" range="-3.0 3.0"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.1 0 0" size="0.012" rgba="0.3 0.6 0.9 1"/>
        <site name="fingertip" pos="0.1 0 0" size="0.01" rgba="0 0.8 0.2 1"/>
      </body>
    </body>
    <body name="target" pos="0.12 0.08 0.02">
      <geom name="target_marker" type="sphere" size="0.012" rgba="0.9 0.1 0.1 1"/>
      <site name="target" pos="0 0 0" size="0.012" rgba="0.9 0.1 0.1 1"/>
    </body>
  </worldbody>
  <!-- Position actuators: the normalized policy command in [-1, 1] maps to a
       target joint angle (design's "absolute joint position" action mode). -->
  <actuator>
    <position name="m1" joint="j1" kp="20" ctrlrange="-3.14159 3.14159"/>
    <position name="m2" joint="j2" kp="20" ctrlrange="-3.0 3.0"/>
  </actuator>
</mujoco>
"""

BUILTIN_MODELS = {"reacher2": REACHER2}


def get_model_xml(key: str) -> str:
    try:
        return BUILTIN_MODELS[key]
    except KeyError:
        raise ValueError(
            f"unknown builtin model {key!r}; choices: {sorted(BUILTIN_MODELS)}"
        ) from None
