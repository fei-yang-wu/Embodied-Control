"""The pin file: a repository, one immutable revision, and per-file hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MODEL_PIN_API_VERSION = "ec.model_pin/v1"
PIN_FILENAME = "model.pin.json"
# Files the pin describes the bundle with, never itself.
EXCLUDED_FROM_PIN = frozenset({PIN_FILENAME})


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PinnedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sha256: str
    size: int = Field(ge=0)

    @field_validator("sha256")
    @classmethod
    def _hex(cls, value: str) -> str:
        text = value.strip().lower()
        if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
            raise ValueError("sha256 must be 64 hex characters")
        return text


class ModelPin(BaseModel):
    """Where a bundle comes from, and which bytes are the right ones."""

    model_config = ConfigDict(extra="forbid")

    api_version: Literal["ec.model_pin/v1"] = MODEL_PIN_API_VERSION
    # `controller` is a deployable policy bundle the tracker loads;
    # `planner` is what a planner worker loads. The console lists them apart.
    kind: Literal["controller", "planner"]
    name: str
    repo: str
    repo_type: Literal["model", "dataset"] = "model"
    # A commit sha, never a branch: a branch moves and a pin must not.
    revision: str
    # Subdirectory inside the repository; empty means the repository root.
    path: str = ""
    files: dict[str, PinnedFile]

    @field_validator("revision")
    @classmethod
    def _commit_sha(cls, value: str) -> str:
        text = value.strip().lower()
        if len(text) != 40 or any(c not in "0123456789abcdef" for c in text):
            raise ValueError(
                "revision must be a 40-character commit sha; a branch or tag "
                "moves under the pin"
            )
        return text

    @field_validator("files")
    @classmethod
    def _relative_paths(cls, value: dict) -> dict:
        if not value:
            raise ValueError("a pin needs at least one file")
        for name in value:
            candidate = Path(name)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"pinned file '{name}' escapes the directory")
        return value

    def remote_path(self, name: str) -> str:
        return f"{self.path}/{name}" if self.path else name


def load_pin(directory: str | Path) -> ModelPin:
    path = Path(directory)
    if path.is_dir():
        path = path / PIN_FILENAME
    return ModelPin.model_validate(json.loads(path.read_text()))


def write_pin(directory: str | Path, pin: ModelPin) -> Path:
    path = Path(directory) / PIN_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pin.model_dump(), indent=2, sort_keys=True) + "\n")
    return path


def has_pin(directory: str | Path) -> bool:
    return (Path(directory) / PIN_FILENAME).is_file()


def find_pins(root: str | Path) -> list[Path]:
    """Every pinned model directory under `root`, sorted by path."""
    base = Path(root)
    if not base.is_dir():
        return []
    return sorted(p.parent for p in base.rglob(PIN_FILENAME))
