#!/usr/bin/env bash
# Build the fake delegated evaluator container image.
#
# If your shell is not yet in the `docker` group, run this under `newgrp docker`
# or `sg docker -c 'pixi run build-fake-delegated-image'`, or reboot after being added.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${EC_FAKE_DELEGATED_IMAGE:-ec-fake-delegated-eval:latest}"

cd "$REPO_ROOT"
echo "Building fake delegated evaluator image: $IMAGE"
docker build -f containers/fake_delegated_eval/Dockerfile -t "$IMAGE" .
echo "Done: $IMAGE"
