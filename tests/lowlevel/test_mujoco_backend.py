"""MuJoCo backend mechanics on a tiny synthetic 2-joint model.

Runs only where mujoco is installed (`-e lowlevel-sim` / `-e sim`). The full
G1 stand/tracking checks run through `ec lowlevel run` with a real bundle.
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from embodied_control.cli import build_parser  # noqa: E402
from embodied_control.lowlevel.bundle import ActionContract  # noqa: E402
from embodied_control.lowlevel.contracts import JointCommand  # noqa: E402
from embodied_control.lowlevel.envs.mujoco import MujocoBackend  # noqa: E402

_TINY = """
<mujoco>
  <worldbody>
    <body name="pelvis" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" mass="5"/>
      <body name="upper" pos="0.2 0 0">
        <joint name="beta_joint" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="1"/>
        <body name="lower" pos="0.2 0 0">
          <joint name="alpha_joint" axis="0 1 0"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="beta_joint" joint="beta_joint"/>
    <motor name="alpha_joint" joint="alpha_joint"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def contract(tmp_path):
    # Isaac order deliberately differs from the model's actuator order.
    return ActionContract(
        width=2,
        isaac_joint_names=["alpha_joint", "beta_joint"],
        sdk_joint_names=["beta_joint", "alpha_joint"],
        isaac_to_sdk=[1, 0],
        default_joint_pos=[0.3, -0.2],
        default_joint_vel=[0.0, 0.0],
        action_scale=[0.25, 0.25],
        stiffness=[60.0, 80.0],
        damping=[2.0, 3.0],
        armature=[0.01, 0.02],
        effort_limit=[30.0, 40.0],
    )


@pytest.fixture
def backend(tmp_path, contract):
    model_file = tmp_path / "tiny.xml"
    model_file.write_text(_TINY)
    return MujocoBackend(contract, model_file, timestep=0.005, decimation=4)


def test_servo_params_written_in_isaac_order(backend, contract):
    model = backend.model
    for actuator_id in range(model.nu):
        joint_id = model.actuator_trnid[actuator_id, 0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        isaac_index = contract.isaac_joint_names.index(name)
        assert model.actuator_gainprm[actuator_id, 0] == pytest.approx(
            contract.stiffness[isaac_index]
        )
        assert model.actuator_biasprm[actuator_id, 1] == pytest.approx(
            -contract.stiffness[isaac_index]
        )
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(
            -contract.damping[isaac_index]
        )
        assert model.actuator_forcerange[actuator_id, 1] == pytest.approx(
            contract.effort_limit[isaac_index]
        )
        dof = model.jnt_dofadr[joint_id]
        assert model.dof_armature[dof] == pytest.approx(contract.armature[isaac_index])
        assert model.dof_damping[dof] == 0.0


def test_reset_and_state_in_isaac_order(backend, contract):
    state = backend.read_state()
    np.testing.assert_allclose(state.joint_pos, contract.default_joint_pos, atol=1e-9)
    np.testing.assert_allclose(state.projected_gravity, [0, 0, -1], atol=1e-6)
    assert state.anchor_quat_w[3] == pytest.approx(1.0)  # XYZW identity


def test_commanded_isaac_joint_moves_the_right_model_joint(backend, contract):
    target = np.asarray(contract.default_joint_pos, dtype=np.float32)
    target[0] += 0.4  # alpha_joint in Isaac order
    for _ in range(150):
        backend.write_command(
            JointCommand(
                q_target=target,
                kp=np.asarray(contract.stiffness, np.float32),
                kd=np.asarray(contract.damping, np.float32),
            )
        )
    state = backend.read_state()
    assert state.joint_pos[0] == pytest.approx(target[0], abs=0.05)
    assert state.joint_pos[1] == pytest.approx(contract.default_joint_pos[1], abs=0.05)


def test_damp_rewrites_gains(backend, contract):
    damp = JointCommand(
        q_target=np.zeros(2, np.float32),
        kp=np.zeros(2, np.float32),
        kd=np.full(2, 8.0, np.float32),
    )
    backend.write_command(damp)
    model = backend.model
    for actuator_id in range(model.nu):
        assert model.actuator_gainprm[actuator_id, 0] == 0.0
        assert model.actuator_biasprm[actuator_id, 2] == pytest.approx(-8.0)


def test_video_camera_follows_pelvis(tmp_path, contract, monkeypatch):
    class FakeRenderer:
        def __init__(self, model, *, height, width):
            self.camera = None
            self.closed = False

        def update_scene(self, data, *, camera):
            self.camera = camera

        def render(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

        def close(self):
            self.closed = True

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    model_file = tmp_path / "tiny.xml"
    model_file.write_text(_TINY)
    recorded = MujocoBackend(
        contract,
        model_file,
        timestep=0.005,
        decimation=1,
        record_video=True,
        video_every_ticks=1,
    )
    recorded.data.qpos[0:3] = [2.0, -1.0, 0.76]
    mujoco.mj_forward(recorded.model, recorded.data)
    recorded.write_command(
        JointCommand(
            q_target=np.asarray(contract.default_joint_pos, np.float32),
            kp=np.asarray(contract.stiffness, np.float32),
            kd=np.asarray(contract.damping, np.float32),
        )
    )

    np.testing.assert_allclose(
        recorded._renderer.camera.lookat,
        recorded.data.xpos[recorded._pelvis_body],
    )
    renderer = recorded._renderer
    recorded.close()
    assert renderer.closed


def test_live_view_streams_state_and_paces_clock(tmp_path, contract, monkeypatch):
    from embodied_control.lowlevel import plant_view
    from embodied_control.lowlevel.envs import mujoco as backend_module

    class FakeStreamer:
        def __init__(self, model, *, host, port):
            self.url = f"http://{host}:{port}/"
            self.positions = []
            self.closed = False

        def capture(self, data):
            self.positions.append(data.xpos.copy())

        def close(self):
            self.closed = True

    monkeypatch.setattr(plant_view, "MujocoStreamer", FakeStreamer)
    model_file = tmp_path / "tiny.xml"
    model_file.write_text(_TINY)
    viewed = MujocoBackend(
        contract,
        model_file,
        timestep=0.005,
        decimation=1,
        viewer=True,
        viewer_port=0,
        viewer_fps=50,
    )
    viewed.data.qpos[0:3] = [2.0, -1.0, 0.76]
    mujoco.mj_forward(viewed.model, viewed.data)
    viewed.write_command(
        JointCommand(
            q_target=np.asarray(contract.default_joint_pos, np.float32),
            kp=np.asarray(contract.stiffness, np.float32),
            kd=np.asarray(contract.damping, np.float32),
        )
    )
    np.testing.assert_allclose(
        viewed._viewer.positions[-1][viewed._pelvis_body],
        viewed.data.xpos[viewed._pelvis_body],
    )

    viewed._clock._started_at = 100.0
    monkeypatch.setattr(backend_module.time, "monotonic", lambda: 100.01)
    delays = []
    monkeypatch.setattr(backend_module.time, "sleep", delays.append)
    viewed.clock().wait_for_tick(1, 50)
    assert delays == [pytest.approx(0.01)]

    streamer = viewed._viewer
    viewed.close()
    assert streamer.closed


def test_watch_forwards_native_status_and_reports_trajectory(monkeypatch):
    from embodied_control.lowlevel import plant_view

    class FakePlant:
        calls = 0

        def latest_state(self):
            self.calls += 1
            return np.zeros(36, dtype=np.float32)

    class FakePuppet:
        def __init__(self, model_path, joint_names):
            self.model = object()
            self.data = object()
            self.poses = []

        def pose(self, row):
            self.poses.append(row)

    class FakeStreamer:
        def __init__(self, model, *, host, port):
            self.url = f"http://{host}:{port}/"
            self.trajectory_samples = 1
            self.path_length = 2.5
            self.statuses = []
            self.closed = False

        def capture(self, data, *, status=None):
            self.statuses.append(status)

        def close(self):
            self.closed = True

    monkeypatch.setattr(plant_view, "PlantPuppet", FakePuppet)
    monkeypatch.setattr(plant_view, "MujocoStreamer", FakeStreamer)
    plant = FakePlant()
    ready = []
    view = plant_view.watch(
        plant,
        "robot.xml",
        ["joint"],
        live=True,
        port=0,
        stats=lambda: {"ticks": 7},
        on_ready=lambda: ready.append(True),
        should_stop=lambda: plant.calls >= 1,
        sleep=lambda _: None,
    )
    assert view["trajectory_samples"] == 1
    assert view["path_length_m"] == pytest.approx(2.5)
    assert ready == [True]


def test_lowlevel_run_viewer_cli_defaults():
    args = build_parser().parse_args(["lowlevel", "run", "job.yaml", "--viewer"])
    assert args.viewer
    assert args.viewer_host == "127.0.0.1"
    assert args.viewer_port == 8765
    assert args.viewer_fps == 25


def test_missing_armature_is_refused(tmp_path, contract):
    raw = contract.model_dump()
    raw["armature"] = []
    stripped = ActionContract.model_validate(raw)
    model_file = tmp_path / "tiny.xml"
    model_file.write_text(_TINY)
    with pytest.raises(ValueError, match="armature"):
        MujocoBackend(stripped, model_file)
