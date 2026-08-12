#!/usr/bin/env bash
# Package everything the latent playground notebook needs into one directory
# (and a tarball) that can be handed to somebody with no access to the
# training repo: the exported policy bundle, the reference-array tree, and the
# G1 MJCF with its meshes.
#
# Run from the IsaacLab-Imitation side, where those artifacts live:
#
#   external/Embodied-Control/scripts/make_latent_playkit.sh \
#     --bundle    logs/policy_bundles/rollout24_gamma097_3500m \
#     --reference data/bones_seed_language10_v1/reference_arrays/root_qpos_v1 \
#     --mjcf      source/isaaclab_imitation/isaaclab_imitation/assets/unitree/g1_description/g1_29dof_rev_1_0.xml \
#     --output    /tmp/z256_latent_playkit
#
# The notebook then reads it through EC_LATENT_PLAYKIT=<output>.
set -euo pipefail

BUNDLE=""
REFERENCE=""
MJCF=""
OUTPUT=""
TARBALL=1

usage() {
  sed -n '2,18p' "$0"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle) BUNDLE="$2"; shift 2 ;;
    --reference) REFERENCE="$2"; shift 2 ;;
    --mjcf) MJCF="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --no-tarball) TARBALL=0; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

for name in BUNDLE REFERENCE MJCF OUTPUT; do
  if [[ -z "${!name}" ]]; then
    echo "missing --${name,,}" >&2
    usage
  fi
done

[[ -f "${BUNDLE}/manifest.json" ]] || { echo "not a policy bundle: ${BUNDLE}" >&2; exit 1; }
[[ -f "${REFERENCE}/reference_arrays_manifest.json" ]] || {
  echo "not a reference-array tree: ${REFERENCE}" >&2; exit 1; }
[[ -f "${MJCF}" ]] || { echo "MJCF not found: ${MJCF}" >&2; exit 1; }

MJCF_DIR="$(cd "$(dirname "${MJCF}")" && pwd)"
MESH_DIR="${MJCF_DIR}/meshes"
[[ -d "${MESH_DIR}" ]] || { echo "MJCF has no meshes/ beside it: ${MESH_DIR}" >&2; exit 1; }

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}/bundle" "${OUTPUT}/reference" "${OUTPUT}/model"

cp -r "${BUNDLE}/." "${OUTPUT}/bundle/"
cp -r "${REFERENCE}" "${OUTPUT}/reference/$(basename "${REFERENCE}")"
cp "${MJCF}" "${OUTPUT}/model/"
cp -r "${MESH_DIR}" "${OUTPUT}/model/meshes"

BUNDLE_SHA="$(python3 - "$OUTPUT/bundle/manifest.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["source"]["checkpoint_sha256"])
PY
)"

cat > "${OUTPUT}/playkit.json" <<JSON
{
  "kit": "ec.latent_playkit/v1",
  "bundle": "$(basename "${BUNDLE}")",
  "checkpoint_sha256": "${BUNDLE_SHA}",
  "reference": "$(basename "${REFERENCE}")",
  "model": "$(basename "${MJCF}")",
  "source_host": "$(hostname)",
  "created_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

cat > "${OUTPUT}/README.md" <<'MD'
# z256 latent playkit

Self-contained inputs for `notebooks/z256_latent_perturbation.ipynb` in the
Embodied-Control repository.

```
bundle/     exported policy bundle: TorchScript tracker + DiffSR z256 encoder,
            observation/action contracts, normalizer, golden trace, provenance
reference/  reference-array tree (root_qpos_v1) with the motions to encode
model/      G1 MJCF plus its meshes, for the MuJoCo plant and the renderer
playkit.json  what this kit was built from
```

Use it:

```bash
export EC_LATENT_PLAYKIT=<path to this directory>
pixi run -e latent-lab latent-lab
```

Nothing here is a paper metric: the plant is MuJoCo, not the training
simulator, and the notebook runs single deterministic episodes.
MD

echo "playkit: ${OUTPUT}"
du -sh "${OUTPUT}"

if [[ "${TARBALL}" == "1" ]]; then
  TAR_PATH="${OUTPUT%/}.tar.gz"
  tar -czf "${TAR_PATH}" -C "$(dirname "${OUTPUT}")" "$(basename "${OUTPUT}")"
  echo "tarball: ${TAR_PATH}"
  du -sh "${TAR_PATH}"
fi
