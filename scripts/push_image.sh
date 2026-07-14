#!/usr/bin/env bash
# Tag and push a locally-built image to GitHub Container Registry (ghcr.io).
#
# One-time setup:
#   1. Grant the `write:packages` scope (the default `gh auth token` used
#      elsewhere in this repo's scripts does NOT have it):
#          gh auth refresh -h github.com -s write:packages
#      This opens a device-code flow -- visit the printed URL and approve it.
#   2. Log docker in using that token:
#          gh auth token | docker login ghcr.io -u <your-github-username> --password-stdin
#
# Usage: scripts/push_image.sh <local-image:tag> [registry-image:tag]
# If registry-image is omitted, it's derived as
# ghcr.io/$GHCR_NAMESPACE/<repo>:<tag> from the local image name.
set -euo pipefail

GHCR_NAMESPACE="${GHCR_NAMESPACE:-fei-yang-wu}"
LOCAL_IMAGE="${1:?usage: $0 <local-image:tag> [registry-image:tag]}"
REPO="${LOCAL_IMAGE%%:*}"
TAG="${LOCAL_IMAGE##*:}"
REGISTRY_IMAGE="${2:-ghcr.io/${GHCR_NAMESPACE}/${REPO}:${TAG}}"

echo "Tagging $LOCAL_IMAGE -> $REGISTRY_IMAGE"
docker tag "$LOCAL_IMAGE" "$REGISTRY_IMAGE"
echo "Pushing $REGISTRY_IMAGE"
docker push "$REGISTRY_IMAGE"
echo "Done: $REGISTRY_IMAGE"
echo "Pull with: docker pull $REGISTRY_IMAGE"
