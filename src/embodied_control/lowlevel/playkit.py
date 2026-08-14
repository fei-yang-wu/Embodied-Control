"""Fetch the latent playkit at a pinned revision, with no account needed.

The kit lives in a public Hugging Face dataset, so there are two equivalent
paths and neither needs a token:

- `huggingface_hub.snapshot_download` when the package is importable
  (it ships in the `latent-lab` environment and caches nicely);
- otherwise plain stdlib HTTPS against the `resolve/<revision>` endpoint,
  verified against the per-file sha256 map in `playkit.json`.

Everything is pinned by revision: what a notebook downloads is byte-for-byte
what was validated when that notebook was committed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Callable

DEFAULT_REPO = "GeorgiaTech/ec-latent-playkit"
RESOLVE_URL = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"


def fetch_playkit(
    dest: str | Path,
    revision: str,
    *,
    repo: str = DEFAULT_REPO,
    progress: Callable[[str], None] = print,
) -> Path:
    """Ensure the playkit exists at `dest`; download it if it does not."""
    dest = Path(dest)
    if (dest / "playkit.json").exists():
        return dest
    progress(f"playkit not found at {dest} - downloading {repo}@{revision[:12]}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        _fetch_https(dest, revision, repo=repo, progress=progress)
    else:
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=dest)
    progress(f"playkit ready at {dest}")
    return dest


def _fetch_https(
    dest: Path,
    revision: str,
    *,
    repo: str,
    progress: Callable[[str], None],
    url_template: str = RESOLVE_URL,
) -> None:
    def url_for(path: str) -> str:
        return url_template.format(repo=repo, revision=revision, path=path)

    with urllib.request.urlopen(url_for("playkit.json")) as response:
        kit_bytes = response.read()
    files = dict(json.loads(kit_bytes)["file_sha256"])
    # The builder hashes the kit's files before writing playkit.json and the
    # README, so those two are absent from the map; playkit.json is written
    # last below as the completion marker.
    files.pop("playkit.json", None)
    files.pop("README.md", None)
    dest.mkdir(parents=True, exist_ok=True)
    for index, (path, digest) in enumerate(sorted(files.items()), start=1):
        target = dest / path
        if target.exists() and _sha256(target) == digest:
            continue
        progress(f"  [{index}/{len(files)}] {path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url_for(path)) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        actual = _sha256(partial)
        if actual != digest:
            partial.unlink()
            raise IOError(
                f"sha256 mismatch for {path}: expected {digest}, got {actual}"
            )
        partial.replace(target)
    try:
        with urllib.request.urlopen(url_for("README.md")) as response:
            (dest / "README.md").write_bytes(response.read())
    except OSError:
        pass
    (dest / "playkit.json").write_bytes(kit_bytes)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["DEFAULT_REPO", "fetch_playkit"]
