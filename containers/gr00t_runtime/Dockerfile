# Local smoke runtime: mounts the already provisioned Python 3.12 native env
# read-only at /workspace. installed.txt is recorded with each campaign.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends python3.12 libgl1 libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
ENV PYTHONPATH=/workspace/.runtime/gr00t17/lib/python3.12/site-packages:/workspace/repos/Isaac-GR00T \
    HF_HUB_CACHE=/workspace/data/hf-cache/hub HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
ENTRYPOINT ["python3.12", "-m", "gr00t.eval.run_gr00t_server"]
