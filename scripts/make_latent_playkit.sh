#!/usr/bin/env bash
# Package everything the latent-playground notebooks need into one directory
# (and a tarball) that can be handed to somebody with no access to the
# training repo: one or more exported policy bundles, the reference-array
# tree, and the G1 MJCF with its meshes.
#
# Run from the IsaacLab-Imitation side, where those artifacts live:
#
#   external/Embodied-Control/scripts/make_latent_playkit.sh \
#     --bundle    logs/policy_bundles/rollout24_gamma097_3500m \
#     --bundle    logs/policy_bundles/fsq64_sonic_4500m \
#     --reference data/bones_seed_language10_v1/reference_arrays/root_qpos_v1 \
#     --mjcf      source/isaaclab_imitation/isaaclab_imitation/assets/unitree/g1_description/g1_29dof_rev_1_0.xml \
#     --output    /tmp/latent_playkit
#
# The notebooks then read it through EC_LATENT_PLAYKIT=<output>.
set -euo pipefail

BUNDLES=()
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
    --bundle) BUNDLES+=("$2"); shift 2 ;;
    --reference) REFERENCE="$2"; shift 2 ;;
    --mjcf) MJCF="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --no-tarball) TARBALL=0; shift ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

[[ ${#BUNDLES[@]} -gt 0 ]] || { echo "missing --bundle" >&2; usage; }
for name in REFERENCE MJCF OUTPUT; do
  if [[ -z "${!name}" ]]; then
    echo "missing --${name,,}" >&2
    usage
  fi
done

for bundle in "${BUNDLES[@]}"; do
  [[ -f "${bundle}/manifest.json" ]] || { echo "not a policy bundle: ${bundle}" >&2; exit 1; }
done
[[ -f "${REFERENCE}/reference_arrays_manifest.json" ]] || {
  echo "not a reference-array tree: ${REFERENCE}" >&2; exit 1; }
[[ -f "${MJCF}" ]] || { echo "MJCF not found: ${MJCF}" >&2; exit 1; }

MJCF_DIR="$(cd "$(dirname "${MJCF}")" && pwd)"
MESH_DIR="${MJCF_DIR}/meshes"
[[ -d "${MESH_DIR}" ]] || { echo "MJCF has no meshes/ beside it: ${MESH_DIR}" >&2; exit 1; }

rm -rf "${OUTPUT}"
mkdir -p "${OUTPUT}/bundles" "${OUTPUT}/reference" "${OUTPUT}/model"

for bundle in "${BUNDLES[@]}"; do
  cp -r "${bundle}" "${OUTPUT}/bundles/$(basename "${bundle}")"
done
cp -r "${REFERENCE}" "${OUTPUT}/reference/$(basename "${REFERENCE}")"
cp "${MJCF}" "${OUTPUT}/model/"
cp -r "${MESH_DIR}" "${OUTPUT}/model/meshes"

python3 - "${OUTPUT}" "$(basename "${REFERENCE}")" "$(basename "${MJCF}")" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

output = Path(sys.argv[1])
bundles = {}
for bundle_dir in sorted(output.glob("bundles/*")):
    manifest = json.load(open(bundle_dir / "manifest.json"))
    command = manifest["command"]
    bundles[bundle_dir.name] = {
        "checkpoint_sha256": manifest["source"]["checkpoint_sha256"],
        "skill_checkpoint_sha256": manifest["source"].get("skill_checkpoint_sha256"),
        "interface": manifest["interface"],
        "quantizer": command.get("quantizer", "none"),
        "z_dim": command.get("z_dim"),
        "hold_steps": command.get("hold_steps"),
        "macro_anchor_mode": command.get("macro_anchor_mode"),
    }

files = {}
for path in sorted(p for p in output.rglob("*") if p.is_file()):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files[str(path.relative_to(output))] = digest

kit = {
    "kit": "ec.latent_playkit/v2",
    "bundles": bundles,
    "reference": sys.argv[2],
    "model": sys.argv[3],
    "source_host": subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip(),
    "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "file_sha256": files,
}
(output / "playkit.json").write_text(json.dumps(kit, indent=2) + "\n")
PY

cat > "${OUTPUT}/README.md" <<'MD'
# Latent playkit

Self-contained inputs for the latent-playground notebooks in the
Embodied-Control repository:

- `notebooks/z256_latent_perturbation.ipynb` — continuous 256-dim latent
  (`bundles/rollout24_gamma097_3500m`)
- `notebooks/fsq64_latent_perturbation.ipynb` — quantized 64-dim FSQ latent
  (`bundles/fsq64_sonic_4500m`)

```
bundles/<name>/  exported policy bundle: TorchScript tracker + skill encoder,
                 observation/action contracts, normalizer, golden trace,
                 provenance (training-checkpoint SHA)
reference/       reference-array tree (root_qpos_v1) with the motions to encode
model/           G1 MJCF plus its meshes, for the MuJoCo plant and the renderer
playkit.json     what this kit was built from + per-file sha256
```

Use it:

```bash
export EC_LATENT_PLAYKIT=<path to this directory>
pixi run -e latent-lab latent-lab
```

Distribution: upload this directory to the private Hugging Face dataset
`GeorgiaTech/ec-latent-playkit` and re-pin `PLAYKIT_REVISION` in both
notebooks' Inputs cell to the new commit; the notebooks auto-download the
pinned revision.

Nothing here is a paper metric: the plant is MuJoCo, not the training
simulator, and the notebooks run single deterministic episodes.
MD

echo "playkit: ${OUTPUT}"
du -sh "${OUTPUT}"

if [[ "${TARBALL}" == "1" ]]; then
  TAR_PATH="${OUTPUT%/}.tar.gz"
  tar -czf "${TAR_PATH}" -C "$(dirname "${OUTPUT}")" "$(basename "${OUTPUT}")"
  echo "tarball: ${TAR_PATH}"
  du -sh "${TAR_PATH}"
fi
