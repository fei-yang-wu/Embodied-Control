"""Fetch, verify, and publish pinned model directories.

The store never guesses. A pinned file is present only when its sha256
matches; anything else is downloaded again at the pinned revision, and an
offline caller is told what is missing instead of being handed a partial
bundle.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from embodied_control.models.pin import (
    EXCLUDED_FROM_PIN,
    MODEL_PIN_API_VERSION,
    ModelPin,
    PinnedFile,
    has_pin,
    load_pin,
    sha256_of,
    write_pin,
)


class ModelStoreError(RuntimeError):
    """A pinned model could not be produced as pinned."""


def _hub():
    try:
        import huggingface_hub
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ModelStoreError(
            "huggingface_hub is required to fetch or publish a pinned model; "
            "install it in this environment"
        ) from exc
    return huggingface_hub


def verify(directory: str | Path, pin: ModelPin | None = None) -> list[str]:
    """Names of the pinned files that are missing or have the wrong bytes."""
    path = Path(directory)
    pin = pin or load_pin(path)
    wrong: list[str] = []
    for name, expected in pin.files.items():
        target = path / name
        if not target.is_file() or target.stat().st_size != expected.size:
            wrong.append(name)
            continue
        if sha256_of(target) != expected.sha256:
            wrong.append(name)
    return wrong


def materialize(
    directory: str | Path,
    *,
    offline: bool = False,
    token: str | None = None,
) -> Path:
    """Make `directory` hold exactly the pinned bytes, and return it."""
    path = Path(directory)
    pin = load_pin(path)
    missing = verify(path, pin)
    if not missing:
        return path
    if offline:
        raise ModelStoreError(
            f"{path} is missing {len(missing)} pinned file(s) "
            f"({', '.join(sorted(missing)[:3])}...) and offline was requested"
        )
    hub = _hub()
    for name in missing:
        try:
            fetched = hub.hf_hub_download(
                repo_id=pin.repo,
                filename=pin.remote_path(name),
                revision=pin.revision,
                repo_type=pin.repo_type,
                token=token,
            )
        except Exception as exc:
            raise ModelStoreError(
                f"could not fetch {pin.repo}@{pin.revision[:8]}:"
                f"{pin.remote_path(name)}: {exc}"
            ) from exc
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        # The hub keeps its own cache; the pin directory owns a real copy so
        # a cleared cache cannot empty a bundle the robot is about to run.
        shutil.copyfile(fetched, target)
    still_wrong = verify(path, pin)
    if still_wrong:
        raise ModelStoreError(
            f"{path} still does not match its pin after fetching: "
            f"{', '.join(sorted(still_wrong))}"
        )
    return path


def ensure_model(
    directory: str | Path,
    *,
    offline: bool = False,
    token: str | None = None,
) -> Path:
    """Materialize `directory` when it is pinned; pass it through otherwise.

    A bundle checked out by hand stays usable: only a directory carrying a
    pin file is fetched or verified.
    """
    path = Path(directory)
    if not has_pin(path):
        return path
    return materialize(path, offline=offline, token=token)


def fetch_and_pin(
    directory: str | Path,
    *,
    kind: str,
    repo: str,
    path_in_repo: str = "",
    revision: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
) -> ModelPin:
    """Adopt a folder that already lives in a repository.

    A pin carries sha256 per file and the hub does not publish one for every
    object, so the files are downloaded once, hashed here, and pinned to the
    commit the download resolved to.
    """
    target = Path(directory)
    hub = _hub()
    api = hub.HfApi(token=token)
    try:
        resolved = revision or api.repo_info(repo_id=repo, repo_type=repo_type).sha
        prefix = path_in_repo.strip("/")
        snapshot = Path(
            hub.snapshot_download(
                repo_id=repo,
                repo_type=repo_type,
                revision=resolved,
                token=token,
                allow_patterns=[f"{prefix}/*"] if prefix else None,
            )
        )
    except Exception as exc:
        raise ModelStoreError(f"could not fetch {repo}: {exc}") from exc
    source = snapshot / prefix if prefix else snapshot
    if not source.is_dir():
        raise ModelStoreError(f"{repo}@{resolved[:8]} has no directory '{prefix}'")
    target.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.rglob("*")):
        if not item.is_file():
            continue
        destination = target / item.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, destination)
    return pin_local_directory(
        target,
        kind=kind,
        repo=repo,
        path_in_repo=prefix,
        revision=resolved,
        repo_type=repo_type,
        token=token,
    )


def _pinnable_files(directory: Path) -> dict[str, PinnedFile]:
    files: dict[str, PinnedFile] = {}
    for item in sorted(directory.rglob("*")):
        if not item.is_file():
            continue
        name = item.relative_to(directory).as_posix()
        if name in EXCLUDED_FROM_PIN:
            continue
        files[name] = PinnedFile(sha256=sha256_of(item), size=item.stat().st_size)
    if not files:
        raise ModelStoreError(f"{directory} holds no files to pin")
    return files


def pin_local_directory(
    directory: str | Path,
    *,
    kind: str,
    repo: str,
    path_in_repo: str = "",
    revision: str | None = None,
    repo_type: str = "model",
    token: str | None = None,
) -> ModelPin:
    """Record what is already on disk as a pin against an existing repo.

    `revision` defaults to the repository's current head commit, resolved
    once and written down, so the pin does not follow the branch.
    """
    source = Path(directory)
    if revision is None:
        hub = _hub()
        try:
            info = hub.HfApi(token=token).repo_info(
                repo_id=repo, repo_type=repo_type
            )
        except Exception as exc:
            raise ModelStoreError(f"could not read {repo}: {exc}") from exc
        revision = info.sha
    pin = ModelPin(
        api_version=MODEL_PIN_API_VERSION,
        kind=kind,
        name=source.name,
        repo=repo,
        repo_type=repo_type,
        revision=revision,
        path=path_in_repo.strip("/"),
        files=_pinnable_files(source),
    )
    write_pin(source, pin)
    return pin


def push_model(
    directory: str | Path,
    *,
    kind: str,
    repo: str,
    path_in_repo: str = "",
    private: bool = True,
    repo_type: str = "model",
    token: str | None = None,
    message: str | None = None,
) -> ModelPin:
    """Upload a local bundle and pin the commit the upload produced."""
    source = Path(directory)
    if not source.is_dir():
        raise ModelStoreError(f"{source} is not a directory")
    files = _pinnable_files(source)
    hub = _hub()
    api = hub.HfApi(token=token)
    try:
        api.create_repo(
            repo_id=repo, repo_type=repo_type, private=private, exist_ok=True
        )
        commit = api.upload_folder(
            folder_path=str(source),
            path_in_repo=path_in_repo.strip("/"),
            repo_id=repo,
            repo_type=repo_type,
            # The pin describes the upload; it is not part of it.
            ignore_patterns=sorted(EXCLUDED_FROM_PIN),
            commit_message=message or f"Publish {kind} bundle {source.name}",
        )
    except Exception as exc:
        raise ModelStoreError(f"could not publish to {repo}: {exc}") from exc
    revision = getattr(commit, "oid", None) or api.repo_info(
        repo_id=repo, repo_type=repo_type
    ).sha
    pin = ModelPin(
        api_version=MODEL_PIN_API_VERSION,
        kind=kind,
        name=source.name,
        repo=repo,
        repo_type=repo_type,
        revision=revision,
        path=path_in_repo.strip("/"),
        files=files,
    )
    write_pin(source, pin)
    return pin
