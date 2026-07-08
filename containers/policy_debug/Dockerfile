# Minimal image for the blank VLA policy service.
#
# The policy service is stdlib-only, so we do NOT pip-install the package (no
# pydantic, no mujoco, no network beyond the base image). We just copy the source
# and put it on PYTHONPATH. Result: a tiny, fast-building image.
#
# Build (from repo root):
#     docker build -f containers/policy_debug/Dockerfile -t ec-policy-debug:latest .
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app
# Only the modules the policy service imports (stdlib-only subtree).
COPY src/embodied_control/__init__.py            /app/src/embodied_control/__init__.py
COPY src/embodied_control/policies/              /app/src/embodied_control/policies/
COPY src/embodied_control/transport/__init__.py  /app/src/embodied_control/transport/__init__.py
COPY src/embodied_control/transport/protocol.py  /app/src/embodied_control/transport/protocol.py
COPY src/embodied_control/transport/server.py    /app/src/embodied_control/transport/server.py

# Default policy port inside the container; overridden by --port at run time.
EXPOSE 8000

# The host appends args: --type ... --action-dim ... --host 0.0.0.0 --port ... --seed ...
ENTRYPOINT ["python", "-m", "embodied_control.policies.debug_server"]
CMD ["--type", "zero", "--action-dim", "2", "--host", "0.0.0.0", "--port", "8000"]
