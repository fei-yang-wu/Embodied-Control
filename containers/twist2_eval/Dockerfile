FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==1.26.4 mujoco==3.3.7 onnxruntime==1.23.2 imageio==2.37.0 imageio-ffmpeg==0.6.0 pyzmq==27.1.0 msgpack==1.1.2
COPY repos/Embodied-Control/src/embodied_control /app/embodied_control
COPY external/TWIST2/assets /twist2/assets
ENV PYTHONPATH=/app MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-m", "embodied_control.sim.twist2_eval"]
