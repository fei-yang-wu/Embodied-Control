#!/usr/bin/env bash
# Build the real LIBERO delegated evaluator container image. Takes several
# minutes (torch + robosuite + LIBERO's pinned dependency set).
#
# If your shell is not yet in the `docker` group, run this under `newgrp docker`
# or `sg docker -c 'pixi run build-libero-image'`, or reboot after being added.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${EC_LIBERO_IMAGE:-ec-libero-eval:latest}"

cd "$REPO_ROOT"
echo "Building LIBERO evaluator image: $IMAGE (this takes a while)"
docker build -f containers/libero_eval/Dockerfile -t "$IMAGE" .
echo "Done: $IMAGE"
