#!/usr/bin/env bash
# One-shot setup for the latent playground notebooks, on Linux or an
# Apple-Silicon mac. Run it from anywhere inside the repository:
#
#   ./scripts/setup_latent_lab.sh              # set up, then launch Jupyter
#   ./scripts/setup_latent_lab.sh --no-launch  # set up only
#
# It installs pixi if missing, builds the latent-lab environment, checks
# Hugging Face access (the playkit lives in a private dataset), pre-downloads
# the playkit at the exact revision the notebooks pin, and starts Jupyter.
# Every step is idempotent — rerunning is always safe.
set -euo pipefail

LAUNCH=1
if [ "${1:-}" = "--no-launch" ]; then
  LAUNCH=0
elif [ -n "${1:-}" ]; then
  echo "unknown argument: $1 (only --no-launch is accepted)" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
echo "==> repository: ${REPO_ROOT}"

# --- 1. platform ------------------------------------------------------------
OS="$(uname -s)"
ARCH="$(uname -m)"
if [ "${OS}" = "Darwin" ] && [ "${ARCH}" != "arm64" ]; then
  echo "error: this environment is locked for Apple-Silicon (arm64) macs only," >&2
  echo "       and this machine is ${ARCH}. Ask for an osx-64 lock if needed." >&2
  exit 1
fi

# --- 2. pixi ----------------------------------------------------------------
export PATH="${HOME}/.pixi/bin:${PATH}"
if ! command -v pixi >/dev/null 2>&1; then
  echo "==> pixi not found - installing to ~/.pixi/bin"
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="${HOME}/.pixi/bin:${PATH}"
fi
echo "==> pixi $(pixi --version)"

# --- 3. environment ---------------------------------------------------------
echo "==> installing the latent-lab environment (first run downloads ~1 GB)"
pixi install --locked -e latent-lab

# --- 4. Hugging Face access -------------------------------------------------
# The playkit (policy bundles + reference motions + robot model) is a private
# dataset; reading it needs a token that can see the GeorgiaTech org.
if ! pixi run -e latent-lab python - <<'PY'
from huggingface_hub import whoami

try:
    info = whoami()
except Exception:
    raise SystemExit(1)
print(f"==> Hugging Face: logged in as {info['name']}")
PY
then
  echo ""
  echo "==> No Hugging Face login found. Paste a token that can read the"
  echo "    GeorgiaTech org (https://huggingface.co/settings/tokens):"
  pixi run -e latent-lab python -c "from huggingface_hub import login; login()"
fi

# --- 5. playkit, at the revision the notebooks pin --------------------------
pixi run -e latent-lab python - <<'PY'
import json
from pathlib import Path

notebook = json.loads(Path("notebooks/fsq64_latent_perturbation.ipynb").read_text())
revision = None
for cell in notebook["cells"]:
    for line in cell["source"]:
        if line.startswith("PLAYKIT_REVISION"):
            revision = line.split('"')[1]
if revision is None:
    raise SystemExit("no PLAYKIT_REVISION found in the notebook")

kit = Path("assets/latent_playkit")
if (kit / "playkit.json").exists():
    print(f"==> playkit already at {kit}")
else:
    from huggingface_hub import snapshot_download

    print(f"==> downloading the playkit at revision {revision[:12]} (~150 MB)")
    snapshot_download(
        "GeorgiaTech/ec-latent-playkit",
        repo_type="dataset",
        revision=revision,
        local_dir=kit,
    )
    print(f"==> playkit ready at {kit}")
PY

# --- 6. launch --------------------------------------------------------------
echo ""
echo "Setup complete. The notebooks are:"
echo "  notebooks/z256_latent_perturbation.ipynb   (continuous 256-dim latent)"
echo "  notebooks/fsq64_latent_perturbation.ipynb  (quantized 64-dim FSQ latent)"
if [ "${LAUNCH}" = "1" ]; then
  echo "==> launching Jupyter (Ctrl-C to stop; relaunch any time with:"
  echo "    pixi run -e latent-lab latent-lab)"
  exec pixi run -e latent-lab latent-lab
else
  echo "Launch later with: pixi run -e latent-lab latent-lab"
fi
