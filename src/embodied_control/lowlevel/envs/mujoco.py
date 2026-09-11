"""MuJoCo sim2sim backend for the G1 tracker.

Follows the conventions established by IsaacLab-Imitation's
`scripts/bench/mujoco_reference_tracking_baseline.py`: the vendored MJCF's
`<motor>` actuators are rewritten into position servos
(`gainprm[0]=kp, biasprm[1]=-kp, biasprm[2]=-kd`, force-limited to the
effort limit) with gains, armature, and limits taken from the bundle's
action contract; passive joint damping and frictionloss are zeroed (the
training runtime's values); integrator implicitfast (what MJWarp runs);
timestep 0.005 with 4 substeps per 50 Hz control step.

Conventions verified empirically or against the training stack:
- The MJCF actuator order is the Unitree SDK order; the contract's joint
  name lists map it to Isaac order. Nothing is derived by position.
- MuJoCo free-joint `qvel[3:6]` is BODY-frame angular velocity, which is
  exactly Isaac's `root_ang_vel_b` — no rotation needed.
- MuJoCo quaternions are WXYZ; `RobotState` carries XYZW.

Known gap, by design: MuJoCo's actuator dynamics differ from Newton/PhysX
(the 2026-08-03 sim2sim verdict). This backend is a deployment signal, not
a paper metric.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from embodied_control.lowlevel.bundle import ActionContract
from embodied_control.lowlevel.contracts import JointCommand, RobotState
from embodied_control.lowlevel.maths import rotate_inverse, wxyz_to_xyzw

_SCENE_TEMPLATE = """
<mujoco>
  <include file="{model}"/>
  <statistic center="0 0 0.8" extent="1.6"/>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true"/>
    <camera name="track" pos="2.2 -2.2 1.6" xyaxes="0.7 0.7 0 -0.35 0.35 0.87"/>
  </worldbody>
</mujoco>
"""


def load_scene_model(model_path: str | Path):
    """Load the G1 MJCF wrapped in the light/camera scene.

    Mesh paths in the MJCF resolve relative to the main model file, so the
    wrapper scene is written beside it while loading.
    """
    import os
    import tempfile

    import mujoco

    model_path = Path(model_path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {model_path}")
    wrapper = tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=model_path.parent, delete=False
    )
    try:
        wrapper.write(_SCENE_TEMPLATE.format(model=model_path.name))
        wrapper.close()
        return mujoco.MjModel.from_xml_path(wrapper.name)
    finally:
        os.unlink(wrapper.name)


class _SimClock:
    def __init__(self, backend: "MujocoBackend"):
        self._backend = backend

    def now(self) -> float:
        return self._backend.now

    def wait_for_tick(self, tick: int, control_hz: int) -> None:
        return


class MujocoBackend:
    def __init__(
        self,
        action: ActionContract,
        model_path: str | Path,
        *,
        control_hz: int = 50,
        timestep: float = 0.005,
        decimation: int = 4,
        record_video: bool = False,
        video_every_ticks: int = 2,
    ):
        import mujoco

        self._mujoco = mujoco
        if not action.armature or not action.effort_limit:
            raise ValueError(
                "the MuJoCo backend needs armature and effort_limit in the "
                "action contract; re-export the bundle"
            )
        self.model = load_scene_model(model_path)
        self.data = mujoco.MjData(self.model)
        self._action = action
        self._decimation = int(decimation)
        self._dt = float(timestep)
        del control_hz
        self.model.opt.timestep = self._dt
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

        if self.model.nu != action.width:
            raise ValueError(f"model has {self.model.nu} actuators, expected {action.width}")
        self._actuator_to_isaac = np.empty(action.width, dtype=np.int64)
        self._qposadr = np.empty(action.width, dtype=np.int64)
        self._dofadr = np.empty(action.width, dtype=np.int64)
        for actuator_id in range(self.model.nu):
            joint_id = self.model.actuator_trnid[actuator_id, 0]
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name not in action.isaac_joint_names:
                raise ValueError(f"model joint {name!r} is not in the action contract")
            isaac_index = action.isaac_joint_names.index(name)
            self._actuator_to_isaac[actuator_id] = isaac_index
            self._qposadr[actuator_id] = self.model.jnt_qposadr[joint_id]
            dof = self.model.jnt_dofadr[joint_id]
            self._dofadr[actuator_id] = dof
            self.model.dof_armature[dof] = action.armature[isaac_index]
            self.model.dof_damping[dof] = 0.0
            self.model.dof_frictionloss[dof] = 0.0
            # Joint limits as stiff as Isaac Lab's Newton/MJWarp model carries
            # them (`solreflimit="-10000 -10"`, `solimplimit="0.9 0.95 0.001"`
            # in the MJCF Newton dumps). The vendor MJCF's default soft limit
            # (`solref 0.02 1`) lets a driven ankle overshoot its hard range by
            # a few hundredths of a radian, which the Unitree writer's
            # measured-position guard reports as a fault; the training
            # simulator never lets the joint get there.
            self.model.jnt_solref[joint_id] = [-10000.0, -10.0]
            self.model.jnt_solimp[joint_id] = [0.9, 0.95, 0.001, 0.5, 2.0]
            self.model.actuator_forcelimited[actuator_id] = 1
            effort = action.effort_limit[isaac_index]
            self.model.actuator_forcerange[actuator_id] = [-effort, effort]
            self.model.actuator_ctrllimited[actuator_id] = 0
        self._applied_kp = np.full(action.width, np.nan)
        self._applied_kd = np.full(action.width, np.nan)
        self._write_gains(
            np.asarray(action.stiffness, dtype=np.float64),
            np.asarray(action.damping, dtype=np.float64),
        )

        pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis < 0:
            raise ValueError("model has no 'pelvis' body")
        self._pelvis_body = pelvis
        self._clock = _SimClock(self)

        self._renderer = None
        self._video_every = max(1, int(video_every_ticks))
        self._tick = 0
        self.frames: list[np.ndarray] = []
        if record_video:
            try:
                self._renderer = mujoco.Renderer(self.model, height=480, width=640)
            except Exception as exc:  # rendering is best-effort, never fatal
                print(f"video disabled: offscreen renderer unavailable ({exc})")
                self._renderer = None
        self.reset()

    def _write_gains(self, kp: np.ndarray, kd: np.ndarray) -> None:
        mujoco = self._mujoco
        for actuator_id in range(self.model.nu):
            isaac_index = self._actuator_to_isaac[actuator_id]
            if (
                self._applied_kp[isaac_index] == kp[isaac_index]
                and self._applied_kd[isaac_index] == kd[isaac_index]
            ):
                continue
            self.model.actuator_gaintype[actuator_id] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[actuator_id] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gainprm[actuator_id, :] = 0.0
            self.model.actuator_biasprm[actuator_id, :] = 0.0
            self.model.actuator_gainprm[actuator_id, 0] = kp[isaac_index]
            self.model.actuator_biasprm[actuator_id, 1] = -kp[isaac_index]
            self.model.actuator_biasprm[actuator_id, 2] = -kd[isaac_index]
        self._applied_kp = kp.copy()
        self._applied_kd = kd.copy()

    def reset(self, seed: int = 0) -> None:
        del seed
        mujoco = self._mujoco
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[0:3] = [0.0, 0.0, 0.76]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        for actuator_id in range(self.model.nu):
            isaac_index = self._actuator_to_isaac[actuator_id]
            self.data.qpos[self._qposadr[actuator_id]] = self._action.default_joint_pos[
                isaac_index
            ]
        self.data.ctrl[:] = 0.0
        for actuator_id in range(self.model.nu):
            isaac_index = self._actuator_to_isaac[actuator_id]
            self.data.ctrl[actuator_id] = self._action.default_joint_pos[isaac_index]
        mujoco.mj_forward(self.model, self.data)
        self.data.time = 0.0
        self._tick = 0
        self.frames = []

    def set_pose(self, root_pose_xyzw: np.ndarray, joint_pos: np.ndarray) -> None:
        """Place the robot on `[pos 3 | quat XYZW 4]` plus Isaac-ordered joints.

        Velocities are zeroed and the position targets are seeded with the same
        pose, so the first control tick does not fight a stale target. Used to
        start an episode on a reference frame instead of the default stance.
        """
        mujoco = self._mujoco
        root_pose_xyzw = np.asarray(root_pose_xyzw, dtype=np.float64)
        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        if root_pose_xyzw.shape != (7,):
            raise ValueError(f"root pose must have shape (7,), got {root_pose_xyzw.shape}")
        if joint_pos.shape != (self._action.width,):
            raise ValueError(
                f"joint_pos must have shape ({self._action.width},), got {joint_pos.shape}"
            )
        self.data.qpos[0:3] = root_pose_xyzw[0:3]
        x, y, z, w = root_pose_xyzw[3:7]
        self.data.qpos[3:7] = [w, x, y, z]
        for actuator_id in range(self.model.nu):
            isaac_index = self._actuator_to_isaac[actuator_id]
            self.data.qpos[self._qposadr[actuator_id]] = joint_pos[isaac_index]
            self.data.ctrl[actuator_id] = joint_pos[isaac_index]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    @property
    def now(self) -> float:
        return float(self.data.time)

    def read_state(self) -> RobotState:
        joint_pos = np.empty(self._action.width, dtype=np.float32)
        joint_vel = np.empty(self._action.width, dtype=np.float32)
        for actuator_id in range(self.model.nu):
            isaac_index = self._actuator_to_isaac[actuator_id]
            joint_pos[isaac_index] = self.data.qpos[self._qposadr[actuator_id]]
            joint_vel[isaac_index] = self.data.qvel[self._dofadr[actuator_id]]
        root_quat_xyzw = wxyz_to_xyzw(np.asarray(self.data.qpos[3:7], dtype=np.float32))
        gravity = rotate_inverse(
            root_quat_xyzw, np.array([0.0, 0.0, -1.0], dtype=np.float32)
        )
        pelvis_quat_xyzw = wxyz_to_xyzw(
            np.asarray(self.data.xquat[self._pelvis_body], dtype=np.float32)
        )
        return RobotState(
            stamp=self.now,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            projected_gravity=gravity.astype(np.float32),
            base_ang_vel=np.asarray(self.data.qvel[3:6], dtype=np.float32),
            anchor_pos_w=np.asarray(
                self.data.xpos[self._pelvis_body], dtype=np.float32
            ),
            anchor_quat_w=pelvis_quat_xyzw,
        )

    def write_command(self, cmd: JointCommand) -> None:
        mujoco = self._mujoco
        kp = np.asarray(cmd.kp, dtype=np.float64)
        kd = np.asarray(cmd.kd, dtype=np.float64)
        if not (
            np.array_equal(kp, self._applied_kp) and np.array_equal(kd, self._applied_kd)
        ):
            self._write_gains(kp, kd)
        target = np.asarray(cmd.q_target, dtype=np.float64)
        for actuator_id in range(self.model.nu):
            self.data.ctrl[actuator_id] = target[self._actuator_to_isaac[actuator_id]]
        for _ in range(self._decimation):
            mujoco.mj_step(self.model, self.data)
        self._tick += 1
        if self._renderer is not None and self._tick % self._video_every == 0:
            self._renderer.update_scene(self.data, camera="track")
            self.frames.append(self._renderer.render().copy())

    def clock(self) -> _SimClock:
        return self._clock

    @property
    def base_height(self) -> float:
        return float(self.data.qpos[2])


__all__ = ["MujocoBackend"]
