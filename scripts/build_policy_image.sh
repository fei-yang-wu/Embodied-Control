#!/usr/bin/env bash
# Build the blank policy-service container image.
#
# If your shell is not yet in the `docker` group, run this under `newgrp docker`
# or `sg docker -c 'pixi run build-policy-image'`, or reboot after being added.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${EC_POLICY_IMAGE:-ec-policy-debug:latest}"

cd "$REPO_ROOT"
echo "Building policy image: $IMAGE"
docker build -f containers/policy_debug/Dockerfile -t "$IMAGE" .
echo "Done: $IMAGE"
echo "Smoke test the image directly:"
echo "  docker run --rm -p 127.0.0.1:8000:8000 $IMAGE --type zero --action-dim 2 --host 0.0.0.0 --port 8000"
