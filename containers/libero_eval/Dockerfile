# Real LIBERO delegated evaluator. Unlike fake_delegated_eval/policy_debug this
# is NOT stdlib-only or tiny -- LIBERO pins an old, specific dependency set
# (numpy==1.22.4, gym==0.25.2, robosuite==1.4.0, ...) that only resolves on
# Python 3.8, which is exactly why it needs to be isolated in its own
# container rather than installed into the host or sim pixi env.
#
# Build (from repo root, takes several minutes -- torch + robosuite + LIBERO):
#     docker build -f containers/libero_eval/Dockerfile -t ec-libero-eval:latest .
FROM python:3.8-slim

# libosmesa6-dev: CPU software rendering (MUJOCO_GL=osmesa below). No NVIDIA
# Container Toolkit is configured for this deployment's Docker daemon, so the
# container has no GPU device -- EGL is not an option here; OSMesa needs no
# host/daemon configuration, which makes it the portable default.
# git: LIBERO's bddl/init-state/asset files are bundled in its git repo, not
# published as a separate downloadable dataset.
# build-essential/gfortran/cmake: numba/scipy/egl_probe (LIBERO's robomimic dep
# pulls in egl_probe, whose native extension needs cmake) build from source.
# libglib2.0-0/libsm6/libxext6/libxrender1: opencv-python==4.6.0.66 (LIBERO pin,
# the non-headless GUI-capable build) needs these at import time even though
# nothing here ever opens a window.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    gfortran \
    cmake \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set before ANY Python process starts (not via os.environ in our harness):
# MuJoCo selects its GL backend at first import and reuses that choice for the
# life of the process, so this must be a real container-level env var, not a
# Python-level workaround (see sim/mujoco_backend.py's host-side equivalent,
# which needs the workaround precisely because it CAN'T rely on container ENV).
ENV MUJOCO_GL=osmesa \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /libero_src
RUN git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git .

# Order matches LIBERO's documented install (README): requirements.txt, then
# torch (CPU build here -- no GPU passthrough, see above), then the package
# itself in editable mode.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 \
       --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e . \
    && pip install --no-cache-dir "imageio>=2.15,<3" "imageio-ffmpeg>=0.4"

# libero.libero's __init__.py interactively prompts for a custom dataset path
# on its FIRST import ever, if ~/.libero/config.yaml doesn't exist yet -- which
# would hang a non-interactive container run with no stdin. Trigger that first
# import now, at build time, feeding "n" so it writes the default path config
# (computed from the package's own logic, not hand-guessed paths here) and
# never prompts again.
RUN echo "n" | python3 -c "import libero.libero"

# Only the modules the evaluator imports from our package (stdlib-only
# subtree: transport client + the logging package it depends on).
ENV PYTHONPATH=/app/src
COPY src/embodied_control/__init__.py              /app/src/embodied_control/__init__.py
COPY src/embodied_control/sim/__init__.py           /app/src/embodied_control/sim/__init__.py
COPY src/embodied_control/sim/libero_eval.py        /app/src/embodied_control/sim/libero_eval.py
COPY src/embodied_control/transport/__init__.py     /app/src/embodied_control/transport/__init__.py
COPY src/embodied_control/transport/protocol.py     /app/src/embodied_control/transport/protocol.py
COPY src/embodied_control/transport/client.py       /app/src/embodied_control/transport/client.py
COPY src/embodied_control/logging/                  /app/src/embodied_control/logging/

# The host mounts generated/sim_config.json read-only at /generated and the
# raw/ output directory read-write at /raw (see orchestration/delegated.py).
ENTRYPOINT ["python3", "-m", "embodied_control.sim.libero_eval"]
CMD ["--config", "/generated/sim_config.json", "--output", "/raw"]
