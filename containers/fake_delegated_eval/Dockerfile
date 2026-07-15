# Minimal image for the fake delegated evaluator -- proves the two-container
# orchestration shape (a separate sim runtime that owns its own rollout loop
# and calls the policy service directly) that a real black-box evaluator like
# LIBERO will need later.
#
# Stdlib-only (no pydantic/mujoco), same policy of the policy_debug image: copy
# the source, put it on PYTHONPATH, no pip install. Result: a tiny, fast-building
# image.
#
# Build (from repo root):
#     docker build -f containers/fake_delegated_eval/Dockerfile -t ec-fake-delegated-eval:latest .
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app
# Only the modules the evaluator imports (stdlib-only subtree: transport client
# + the logging package it depends on, plus this evaluator itself).
COPY src/embodied_control/__init__.py            /app/src/embodied_control/__init__.py
COPY src/embodied_control/sim/__init__.py        /app/src/embodied_control/sim/__init__.py
COPY src/embodied_control/sim/fake_delegated_eval.py /app/src/embodied_control/sim/fake_delegated_eval.py
COPY src/embodied_control/transport/__init__.py  /app/src/embodied_control/transport/__init__.py
COPY src/embodied_control/transport/protocol.py  /app/src/embodied_control/transport/protocol.py
COPY src/embodied_control/transport/client.py    /app/src/embodied_control/transport/client.py
COPY src/embodied_control/transport/chunking.py  /app/src/embodied_control/transport/chunking.py
COPY src/embodied_control/transport/base.py      /app/src/embodied_control/transport/base.py
COPY src/embodied_control/transport/factory.py   /app/src/embodied_control/transport/factory.py
COPY src/embodied_control/logging/               /app/src/embodied_control/logging/

# The host mounts generated/sim_config.json read-only at /generated and the
# raw/ output directory read-write at /raw (see orchestration/delegated.py).
ENTRYPOINT ["python", "-m", "embodied_control.sim.fake_delegated_eval"]
CMD ["--config", "/generated/sim_config.json", "--output", "/raw"]
