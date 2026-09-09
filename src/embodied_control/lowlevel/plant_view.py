"""Watch the DDS plant: an interactive 3D web view, a recording, or both.

The plant's own `mjModel`/`mjData` live in C++ and are stepped by a real-time
thread; MuJoCo's Python bindings cannot wrap an existing `mjData`, and handing
a renderer a pointer into state that a 500 Hz thread is writing would tear.

So the view is a puppet: a second model loaded from the same MJCF, whose pose
is overwritten from the plant's published true state each frame and settled
with `mj_forward`. It never steps physics, never touches the plant's memory,
and costs the plant nothing but one 36-float read per drawn frame.

The live view is served by `mjviser` (`PlantStreamer`): real geometry pushed
to a three.js client over Viser's own HTTP/WebSocket server, so orbit / pan /
zoom are native mouse controls in the browser, not something reimplemented
server-side. Works from a headless/remote host: `ssh -L` the port to your
workstation and open it in a browser. No GLFW/X11.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np


class PlantPuppet:
    """A render-only copy of the robot, posed from the plant's true state."""

    def __init__(self, model_path: str | Path, joint_names: Sequence[str]):
        import mujoco

        from embodied_control.lowlevel.envs.mujoco import load_scene_model

        self.model = load_scene_model(model_path)
        self.data = mujoco.MjData(self.model)
        self._mujoco = mujoco
        self._qpos_address = []
        for name in joint_names:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise ValueError(f"{name} is not a joint in {model_path}")
            self._qpos_address.append(int(self.model.jnt_qposadr[joint]))

    def pose(self, row: np.ndarray) -> None:
        """`row` is [pos 3 | quat XYZW 4 | joints 29] in the supplied joint-name order."""
        values = np.asarray(row, dtype=np.float64)
        if values.shape != (7 + len(self._qpos_address),):
            raise ValueError(f"expected {7 + len(self._qpos_address)} values")
        if not np.isfinite(values).all():
            return
        self.data.qpos[0:3] = values[0:3]
        # The wire carries XYZW; MuJoCo's free joint is WXYZ.
        self.data.qpos[3] = values[6]
        self.data.qpos[4:7] = values[3:6]
        for address, angle in zip(self._qpos_address, values[7:]):
            self.data.qpos[address] = angle
        self._mujoco.mj_forward(self.model, self.data)


class PlantRecorder:
    """Offscreen frames to an mp4, for a headless host or a shared artifact."""

    def __init__(
        self,
        puppet: PlantPuppet,
        path: str | Path,
        *,
        fps: int = 30,
        width: int = 960,
        height: int = 540,
        camera: str = "",
    ) -> None:
        import imageio.v2 as imageio
        import mujoco

        self.puppet = puppet
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(fps)
        # The MJCF's offscreen framebuffer defaults to 640x480 and the renderer
        # refuses anything larger; grow it before the context is made.
        puppet.model.vis.global_.offwidth = max(puppet.model.vis.global_.offwidth, width)
        puppet.model.vis.global_.offheight = max(puppet.model.vis.global_.offheight, height)
        self._renderer = mujoco.Renderer(puppet.model, height=height, width=width)
        self._camera = camera
        self._writer = imageio.get_writer(
            self.path, fps=self.fps, macro_block_size=None
        )
        self.frames = 0

    def capture(self) -> None:
        if self._camera:
            self._renderer.update_scene(self.puppet.data, camera=self._camera)
        else:
            self._renderer.update_scene(self.puppet.data)
        self._writer.append_data(self._renderer.render())
        self.frames += 1

    def close(self) -> None:
        self._writer.close()
        self._renderer.close()


class PlantStreamer:
    """Serves the puppet as an interactive 3D scene via mjviser (Viser).

    Real geometry is pushed to a three.js client over Viser's own
    HTTP/WebSocket server, so orbit/pan/zoom are the browser's native mouse
    controls — nothing server-side to reimplement. Binds to loopback by
    default — reach it through an SSH port forward
    (`ssh -L 8765:localhost:8765 host`), not by exposing the port itself.
    """

    def __init__(
        self,
        puppet: PlantPuppet,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        import viser
        from mjviser import ViserMujocoScene

        self.puppet = puppet
        self._server = viser.ViserServer(host=host, port=port, verbose=False)
        self._scene = ViserMujocoScene(self._server, puppet.model, num_envs=1)
        self._scene.create_visualization_gui()
        self.host = host
        self.port = self._server.get_port()
        self.url = f"http://{self.host}:{self.port}/"

    def capture(self) -> None:
        self._scene.update_from_mjdata(self.puppet.data)

    def close(self) -> None:
        self._server.stop()


def watch(
    plant,
    model_path: str | Path,
    joint_names: Sequence[str],
    *,
    live: bool = False,
    video: str | Path = "",
    fps: int = 30,
    width: int = 960,
    height: int = 540,
    camera: str = "",
    host: str = "127.0.0.1",
    port: int = 8765,
    should_stop=lambda: False,
    sleep=None,
):
    """Drive a stream and/or a recorder until `should_stop` says otherwise.

    Runs on the caller's thread, which is the plant process's main thread: the
    physics thread keeps its own core and its own schedule, so a slow GPU
    drops frames here and changes nothing the controller sees.
    """
    import time

    sleep = sleep or time.sleep
    puppet = PlantPuppet(model_path, joint_names)
    recorder = (
        PlantRecorder(puppet, video, fps=fps, width=width, height=height, camera=camera)
        if video
        else None
    )
    streamer = None
    if live:
        streamer = PlantStreamer(puppet, host=host, port=port)
        print(f"VIEWER_URL {streamer.url}", flush=True)
    period = 1.0 / max(1, fps)
    next_frame = time.monotonic()
    try:
        while not should_stop():
            puppet.pose(plant.latest_state())
            if recorder is not None:
                recorder.capture()
            if streamer is not None:
                streamer.capture()
            next_frame += period
            delay = next_frame - time.monotonic()
            if delay > 0:
                sleep(delay)
            else:  # a slow frame never accumulates debt
                next_frame = time.monotonic()
    finally:
        if streamer is not None:
            streamer.close()
        if recorder is not None:
            recorder.close()
    return {
        "video": str(recorder.path) if recorder else "",
        "frames": recorder.frames if recorder else 0,
        "stream_url": streamer.url if streamer else "",
    }
