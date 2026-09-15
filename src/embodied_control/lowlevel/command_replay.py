"""Replay recorded PD targets through a MuJoCo replica of the rehearsal plant.

The hardware and the plant receive the same 50 Hz joint-position targets from
the same policy, and the writer republishes each target at 500 Hz, so a
recording's ``command_target_log`` is exactly the input the actuators saw.
Driving the plant model with that input from the hardware's own measured
state, in short windows re-synchronised to the recording, isolates the plant
side of the loop: identical commands, different joint motion means the model
of the actuator or of the contact is wrong, and the per-joint, per-window
divergence says which joint and when. The policy is not in the loop.

The replica mirrors ``MujocoDdsPlant``: position servos with
``kp (q_des - q) - kd dq`` clipped to the effort limit, the plant profile's
armature, stiff joint limits, and the profile's optional Coulomb friction,
viscous damping and first-order target lag. Sensor noise is not modelled (the
replay compares measured positions, the recording already contains the
hardware's own noise).

The window re-synchronisation resets the joint positions to the recording and
the joint velocities to their finite difference, so the first ticks of every
window carry a transient from the velocity estimate; the statistics skip
them (``settle_ticks``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONTROL_DT = 0.02
SOLE_RADIUS = 0.005
LEG_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)


@dataclass
class Recording:
    """One episode's telemetry: measured joints and commanded targets at 50 Hz."""

    joint_names: list[str]
    measured: np.ndarray  # [T, 29] rad, isaac order
    targets: np.ndarray  # [T, 29] rad, isaac order
    quat_xyzw: np.ndarray  # [T, 4] pelvis orientation from the anchor log
    anchor_pos: np.ndarray  # [T, 3] pelvis translation estimate from the anchor log
    reference_frames: np.ndarray  # [T]
    source: str
    # Optional simulator ground truth (plant recordings only), aligned to the
    # ticks: root position and orientation, used instead of the anchor log to
    # test the replica itself.
    true_root_pos: np.ndarray | None = None
    true_root_quat_xyzw: np.ndarray | None = None

    @property
    def ticks(self) -> int:
        return int(self.measured.shape[0])


def load_recording(telemetry: str | Path, joint_names: list[str]) -> Recording:
    data = np.load(telemetry)
    measured = np.asarray(data["joint_position_log"], np.float64)
    targets = np.asarray(data["command_target_log"], np.float64)
    anchor = np.asarray(data["anchor_pose_log"], np.float64)
    frames = np.asarray(data["reference_frames"])
    valid = np.isfinite(measured).all(axis=1) & np.isfinite(targets).all(axis=1)
    keep = int(np.argmin(valid)) if not valid.all() else measured.shape[0]
    if keep < 3:
        raise ValueError(f"{telemetry}: fewer than three finite ticks")
    return Recording(
        joint_names=list(joint_names),
        measured=measured[:keep],
        targets=targets[:keep],
        quat_xyzw=anchor[:keep, 3:7],
        anchor_pos=anchor[:keep, 0:3],
        reference_frames=frames[:keep],
        source=str(telemetry),
    )


def attach_plant_truth(recording: Recording, states: str | Path) -> None:
    """Align a plant run's 500 Hz state log to the recording's 50 Hz ticks."""
    z = np.load(states)
    names = [str(n) for n in z["joint_names"]]
    perm = [names.index(n) for n in recording.joint_names]
    q = np.asarray(z["joint_pos"], np.float64)[:, perm]
    hz = float(z["publish_hz"]) if "publish_hz" in z.files else 500.0
    per_tick = int(round(hz * CONTROL_DT))
    T = recording.ticks
    best = None
    for offset in range(0, q.shape[0] - T * per_tick):
        d = float(np.abs(q[offset : offset + T * per_tick : per_tick] - recording.measured).mean())
        if best is None or d < best[1]:
            best = (offset, d)
    if best is None or best[1] > 0.01:
        raise ValueError(f"{states}: no alignment with the recording (best {best})")
    rows = np.arange(best[0], best[0] + T * per_tick, per_tick)
    recording.true_root_pos = np.asarray(z["root_pos"], np.float64)[rows]
    recording.true_root_quat_xyzw = np.asarray(z["root_quat_xyzw"], np.float64)[rows]


@dataclass
class ReplayPlant:
    """The rehearsal plant's MuJoCo model, configured like ``MujocoDdsPlant``."""

    mjcf: str
    joint_names: list[str]
    stiffness: np.ndarray
    damping: np.ndarray
    armature: np.ndarray
    effort_limit: np.ndarray
    frictionloss: np.ndarray | None = None
    joint_damping: np.ndarray | None = None
    actuator_lag_s: float = 0.0
    timestep: float = 0.002
    model: object = field(init=False, repr=False)
    data: object = field(init=False, repr=False)
    _actuator: list[int] = field(init=False, repr=False)
    _qpos: list[int] = field(init=False, repr=False)
    _dof: list[int] = field(init=False, repr=False)
    _sole_geoms: list[int] = field(init=False, repr=False)
    _foot_bodies: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(self.mjcf))
        model.opt.timestep = float(self.timestep)
        if model.nq < 7 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("the plant model needs a free base joint first")
        actuator, qpos, dof = [], [], []
        for index, name in enumerate(self.joint_names):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"MJCF lacks joint {name}")
            act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if act < 0:
                raise ValueError(f"MJCF lacks actuator {name}")
            actuator.append(act)
            qpos.append(int(model.jnt_qposadr[joint]))
            d = int(model.jnt_dofadr[joint])
            dof.append(d)
            # Position servo: force = kp (ctrl - q) - kd qdot, clipped.
            model.actuator_gaintype[act] = mujoco.mjtGain.mjGAIN_FIXED
            model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_gainprm[act, :] = 0.0
            model.actuator_biasprm[act, :] = 0.0
            model.actuator_gainprm[act, 0] = self.stiffness[index]
            model.actuator_biasprm[act, 1] = -self.stiffness[index]
            model.actuator_biasprm[act, 2] = -self.damping[index]
            model.actuator_forcelimited[act] = 1
            model.actuator_forcerange[act, 0] = -self.effort_limit[index]
            model.actuator_forcerange[act, 1] = self.effort_limit[index]
            model.actuator_ctrllimited[act] = 0
            model.dof_armature[d] = self.armature[index]
            model.dof_damping[d] = (
                float(self.joint_damping[index]) if self.joint_damping is not None else 0.0
            )
            model.dof_frictionloss[d] = (
                float(self.frictionloss[index]) if self.frictionloss is not None else 0.0
            )
            model.jnt_solref[joint, :] = (-10000.0, -10.0)
            model.jnt_solimp[joint, :3] = (0.9, 0.95, 0.001)
        self._actuator, self._qpos, self._dof = actuator, qpos, dof
        feet = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ("left_ankle_roll_link", "right_ankle_roll_link")
        ]
        if min(feet) < 0:
            raise ValueError("MJCF lacks the ankle_roll_link bodies")
        self._foot_bodies = feet
        self._sole_geoms = [
            g
            for g in range(model.ngeom)
            if int(model.geom_bodyid[g]) in feet
            and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        ]
        self.model = model
        self.data = mujoco.MjData(model)

    # -- state -------------------------------------------------------------
    def set_state(
        self,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        quat_xyzw: np.ndarray,
        root_lin_vel: np.ndarray | None = None,
        root_ang_vel: np.ndarray | None = None,
        root_pos: np.ndarray | None = None,
    ) -> None:
        """Place the robot at the recorded joints and orientation, feet on the floor.

        Root translation is unobserved on hardware; the pelvis is put where the
        lowest sole sphere touches the floor. Root velocities come from the
        anchor log's finite differences (world linear, body angular); without
        them the body would stop dead at every re-sync and every joint would
        absorb the momentum change.
        """
        import mujoco

        d = self.data
        mujoco.mj_resetData(self.model, d)
        q = np.asarray(quat_xyzw, np.float64)
        q = q / max(np.linalg.norm(q), 1e-9)
        d.qpos[0:3] = 0.0
        d.qpos[3] = q[3]
        d.qpos[4:7] = q[0:3]
        for i, adr in enumerate(self._qpos):
            d.qpos[adr] = joint_pos[i]
        d.qvel[:] = 0.0
        if root_lin_vel is not None:
            d.qvel[0:3] = root_lin_vel
        if root_ang_vel is not None:
            d.qvel[3:6] = root_ang_vel
        for i, adr in enumerate(self._dof):
            d.qvel[adr] = joint_vel[i]
        mujoco.mj_kinematics(self.model, d)
        if root_pos is not None:
            d.qpos[0:3] = root_pos
        else:
            lowest = min(float(d.geom_xpos[g][2]) for g in self._sole_geoms) - SOLE_RADIUS
            d.qpos[2] = -lowest
        for i, act in enumerate(self._actuator):
            d.ctrl[act] = joint_pos[i]
        mujoco.mj_forward(self.model, d)

    def step_tick(self, target: np.ndarray, lagged: np.ndarray | None) -> np.ndarray:
        """Hold one 50 Hz target for a control tick; returns the lagged target used."""
        import mujoco

        substeps = int(round(CONTROL_DT / self.timestep))
        alpha = (
            self.timestep / (self.actuator_lag_s + self.timestep)
            if self.actuator_lag_s > 0.0
            else 1.0
        )
        target = np.asarray(target, np.float64)
        current = (
            np.array(lagged, np.float64)
            if (lagged is not None and self.actuator_lag_s > 0.0)
            else target.copy()
        )
        for _ in range(substeps):
            if self.actuator_lag_s > 0.0:
                current = current + alpha * (target - current)
            for i, act in enumerate(self._actuator):
                self.data.ctrl[act] = current[i]
            mujoco.mj_step(self.model, self.data)
        return current

    def joint_pos(self) -> np.ndarray:
        return np.array([self.data.qpos[a] for a in self._qpos])

    def joint_vel(self) -> np.ndarray:
        return np.array([self.data.qvel[a] for a in self._dof])

    def actuator_force(self) -> np.ndarray:
        return np.array([self.data.actuator_force[a] for a in self._actuator])

    def foot_contacts(self) -> np.ndarray:
        """Per foot: any sole sphere in contact."""
        touched = {b: False for b in self._foot_bodies}
        d = self.data
        for k in range(d.ncon):
            c = d.contact[k]
            for g in (c.geom1, c.geom2):
                body = int(self.model.geom_bodyid[g])
                if body in touched and g in self._sole_geoms:
                    touched[body] = True
        return np.array([touched[b] for b in self._foot_bodies], bool)


def plant_from_profile(
    mjcf: str, profile, joint_names: list[str], stiffness: list[float], damping: list[float]
) -> ReplayPlant:
    """Build the replica from a ``PlantConfig`` (armature, effort, friction, lag)
    and the bundle's servo gains, all in the recording's joint order."""
    by_name = {joint.name: joint for joint in profile.joints}
    missing = [n for n in joint_names if n not in by_name]
    if missing:
        raise ValueError(f"plant profile lacks joints {missing}")
    return ReplayPlant(
        mjcf=mjcf,
        joint_names=list(joint_names),
        stiffness=np.asarray(stiffness, np.float64),
        damping=np.asarray(damping, np.float64),
        armature=np.array([by_name[n].armature for n in joint_names], np.float64),
        effort_limit=np.array([by_name[n].effort_limit for n in joint_names], np.float64),
        frictionloss=np.array([by_name[n].frictionloss for n in joint_names], np.float64),
        joint_damping=np.array([by_name[n].damping for n in joint_names], np.float64),
        actuator_lag_s=float(profile.actuator_lag_ms) / 1000.0,
    )


@dataclass
class ReplayResult:
    joint_names: list[str]
    window_ticks: int
    settle_ticks: int
    sim_pos: np.ndarray  # [T, 29], NaN where no window covered the tick
    sim_force: np.ndarray  # [T, 29] Nm
    sim_contact: np.ndarray  # [T, 2] bool
    hw_pos: np.ndarray  # [T, 29]
    targets: np.ndarray  # [T, 29]
    window_start: np.ndarray  # [T] the window each tick belongs to
    valid: np.ndarray  # [T] bool, ticks past the settle transient

    def per_joint(self) -> dict[str, dict[str, float]]:
        out = {}
        err = self.sim_pos - self.hw_pos
        for i, name in enumerate(self.joint_names):
            e = err[self.valid, i]
            e = e[np.isfinite(e)]
            hw_amp = float(np.std(self.hw_pos[self.valid, i]))
            sim_amp = float(np.std(self.sim_pos[self.valid, i][np.isfinite(self.sim_pos[self.valid, i])]))
            out[name] = {
                "mae_rad": float(np.mean(np.abs(e))) if e.size else float("nan"),
                "p95_rad": float(np.percentile(np.abs(e), 95)) if e.size else float("nan"),
                "bias_rad": float(np.mean(e)) if e.size else float("nan"),
                "amp_ratio_sim_over_hw": sim_amp / hw_amp if hw_amp > 1e-6 else float("nan"),
                "force_p95_nm": float(np.percentile(np.abs(self.sim_force[self.valid, i]), 95)),
            }
        return out

    def per_window(self, joints: tuple[str, ...] = LEG_JOINTS) -> list[dict[str, float]]:
        ids = [self.joint_names.index(j) for j in joints]
        rows = []
        for start in np.unique(self.window_start[self.window_start >= 0]):
            mask = (self.window_start == start) & self.valid
            if not mask.any():
                continue
            e = np.abs(self.sim_pos[mask][:, ids] - self.hw_pos[mask][:, ids])
            rows.append(
                {
                    "start_tick": int(start),
                    "leg_mae_rad": float(np.nanmean(e)),
                    "leg_max_rad": float(np.nanmax(e)),
                    "worst_joint": joints[int(np.nanargmax(np.nanmean(e, axis=0)))],
                }
            )
        return rows


def _body_angular_velocity(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """Body-frame angular velocity that takes q0 to q1 over dt (small-angle)."""
    def wxyz(q):
        q = np.asarray(q, np.float64); q = q / max(np.linalg.norm(q), 1e-9)
        return np.array([q[3], q[0], q[1], q[2]])
    a, b = wxyz(q0_xyzw), wxyz(q1_xyzw)
    # relative rotation r = conj(a) * b, expressed in the body frame of a
    w0, x0, y0, z0 = a[0], -a[1], -a[2], -a[3]
    w1, x1, y1, z1 = b
    r = np.array([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ])
    if r[0] < 0:
        r = -r
    angle = 2.0 * np.arctan2(np.linalg.norm(r[1:]), r[0])
    axis = r[1:] / max(np.linalg.norm(r[1:]), 1e-12)
    return axis * angle / dt


def replay(
    recording: Recording,
    plant: ReplayPlant,
    *,
    window_ticks: int = 25,
    settle_ticks: int = 3,
    stride: int | None = None,
) -> ReplayResult:
    """Drive the replica with the recorded targets, window by window."""
    T = recording.ticks
    stride = int(stride or window_ticks)
    sim_pos = np.full((T, len(recording.joint_names)), np.nan)
    sim_force = np.zeros((T, len(recording.joint_names)))
    sim_contact = np.zeros((T, 2), bool)
    window_start = np.full(T, -1, int)
    valid = np.zeros(T, bool)
    for start in range(1, T - 1, stride):
        end = min(start + window_ticks, T - 1)
        vel = (recording.measured[start] - recording.measured[start - 1]) / CONTROL_DT
        if recording.true_root_pos is not None:
            pos_log, quat_log = recording.true_root_pos, recording.true_root_quat_xyzw
        else:
            pos_log, quat_log = recording.anchor_pos, recording.quat_xyzw
        lin = (pos_log[start] - pos_log[start - 1]) / CONTROL_DT
        ang = _body_angular_velocity(quat_log[start - 1], quat_log[start], CONTROL_DT)
        plant.set_state(
            recording.measured[start], vel, quat_log[start],
            root_lin_vel=np.where(np.isfinite(lin), lin, 0.0),
            root_ang_vel=ang,
            root_pos=pos_log[start] if recording.true_root_pos is not None else None,
        )
        lagged = None
        for t in range(start, end):
            lagged = plant.step_tick(recording.targets[t], lagged)
            # The plant's joint position after tick t is what the hardware
            # reports at tick t + 1.
            sim_pos[t + 1] = plant.joint_pos()
            sim_force[t + 1] = plant.actuator_force()
            sim_contact[t + 1] = plant.foot_contacts()
            window_start[t + 1] = start
            valid[t + 1] = (t - start) >= settle_ticks
    return ReplayResult(
        joint_names=list(recording.joint_names),
        window_ticks=window_ticks,
        settle_ticks=settle_ticks,
        sim_pos=sim_pos,
        sim_force=sim_force,
        sim_contact=sim_contact,
        hw_pos=recording.measured.copy(),
        targets=recording.targets.copy(),
        window_start=window_start,
        valid=valid,
    )


def write_report(result: ReplayResult, output: str | Path, *, label: str = "") -> dict:
    """grade-style files: per-joint TSV, per-window TSV, traces NPZ, summary JSON."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    per_joint = result.per_joint()
    per_window = result.per_window()
    with (out / "per_joint.tsv").open("w") as fh:
        fh.write("joint\tmae_rad\tp95_rad\tbias_rad\tamp_ratio_sim_over_hw\tforce_p95_nm\n")
        for name, row in per_joint.items():
            fh.write(
                f"{name}\t{row['mae_rad']:.4f}\t{row['p95_rad']:.4f}\t{row['bias_rad']:+.4f}\t"
                f"{row['amp_ratio_sim_over_hw']:.3f}\t{row['force_p95_nm']:.1f}\n"
            )
    with (out / "per_window.tsv").open("w") as fh:
        fh.write("start_tick\tleg_mae_rad\tleg_max_rad\tworst_joint\n")
        for row in per_window:
            fh.write(
                f"{row['start_tick']}\t{row['leg_mae_rad']:.4f}\t{row['leg_max_rad']:.4f}\t{row['worst_joint']}\n"
            )
    np.savez(
        out / "traces.npz",
        joint_names=np.array(result.joint_names),
        sim_pos=result.sim_pos,
        sim_force=result.sim_force,
        sim_contact=result.sim_contact,
        hw_pos=result.hw_pos,
        targets=result.targets,
        window_start=result.window_start,
        valid=result.valid,
    )
    leg_ids = [result.joint_names.index(j) for j in LEG_JOINTS]
    e = np.abs(result.sim_pos[result.valid][:, leg_ids] - result.hw_pos[result.valid][:, leg_ids])
    # Divergence as a function of ticks since the re-sync: a transient decays,
    # a model mismatch grows.
    offsets = np.arange(result.hw_pos.shape[0]) - result.window_start
    profile = []
    for k in range(1, result.window_ticks):
        mask = (result.window_start >= 0) & (offsets == k)
        if mask.any():
            ek = np.abs(result.sim_pos[mask][:, leg_ids] - result.hw_pos[mask][:, leg_ids])
            profile.append(float(np.nanmean(ek)))
    summary = {
        "label": label,
        "ticks": int(result.hw_pos.shape[0]),
        "window_ticks": result.window_ticks,
        "settle_ticks": result.settle_ticks,
        "leg_mae_rad": float(np.nanmean(e)),
        "leg_p95_rad": float(np.nanpercentile(e, 95)),
        "leg_mae_by_tick_in_window": profile,
        "worst_windows": sorted(per_window, key=lambda r: -r["leg_mae_rad"])[:5],
        "per_joint": per_joint,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
