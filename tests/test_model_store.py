"""Pinned model directories: what is fetched, what is refused, what is recorded."""

import json

import pytest

from embodied_control.models import (
    ModelPin,
    ModelStoreError,
    ensure_model,
    find_pins,
    load_pin,
    materialize,
    sha256_of,
    verify,
    write_pin,
)
from embodied_control.models.store import pin_local_directory, push_model

REVISION = "a" * 40


def _bundle(directory, contents=None):
    directory.mkdir(parents=True, exist_ok=True)
    contents = contents or {"manifest.json": b'{"api_version": "ec.bundle/v1"}'}
    for name, payload in contents.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return directory


def _pin_for(directory, **overrides):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "model.pin.json":
            name = path.relative_to(directory).as_posix()
            files[name] = {"sha256": sha256_of(path), "size": path.stat().st_size}
    payload = {
        "kind": "controller",
        "name": directory.name,
        "repo": "acme/bundles",
        "revision": REVISION,
        "path": "controller/toy",
        "files": files,
    }
    payload.update(overrides)
    return ModelPin.model_validate(payload)


class _FakeHub:
    """A hub that serves one repository from a local directory."""

    def __init__(self, source, *, fail=()):
        self.source = source
        self.fail = set(fail)
        self.requested = []

    def hf_hub_download(self, *, repo_id, filename, revision, repo_type, token):
        self.requested.append((repo_id, filename, revision))
        if filename in self.fail:
            raise RuntimeError("404")
        return str(self.source / filename)


def _install_hub(monkeypatch, hub):
    monkeypatch.setattr("embodied_control.models.store._hub", lambda: hub)


def test_a_pin_refuses_a_branch_for_a_revision():
    with pytest.raises(ValueError, match="commit sha"):
        ModelPin.model_validate(
            {
                "kind": "controller",
                "name": "toy",
                "repo": "acme/bundles",
                "revision": "main",
                "files": {"a.bin": {"sha256": "0" * 64, "size": 1}},
            }
        )


def test_a_pin_refuses_a_path_that_escapes_the_directory():
    with pytest.raises(ValueError, match="escapes"):
        ModelPin.model_validate(
            {
                "kind": "controller",
                "name": "toy",
                "repo": "acme/bundles",
                "revision": REVISION,
                "files": {"../secret": {"sha256": "0" * 64, "size": 1}},
            }
        )


def test_verify_names_the_files_that_do_not_match(tmp_path):
    bundle = _bundle(tmp_path / "toy", {"a.bin": b"one", "b.bin": b"two"})
    write_pin(bundle, _pin_for(bundle))

    assert verify(bundle) == []

    (bundle / "b.bin").write_bytes(b"different")
    assert verify(bundle) == ["b.bin"]

    (bundle / "a.bin").unlink()
    assert sorted(verify(bundle)) == ["a.bin", "b.bin"]


def test_materialize_fetches_only_what_is_missing(tmp_path, monkeypatch):
    remote = _bundle(
        tmp_path / "remote" / "controller" / "toy",
        {"a.bin": b"one", "b.bin": b"two"},
    )
    local = _bundle(tmp_path / "toy", {"a.bin": b"one"})
    write_pin(local, _pin_for(remote))
    hub = _FakeHub(tmp_path / "remote")
    _install_hub(monkeypatch, hub)

    materialize(local)

    assert (local / "b.bin").read_bytes() == b"two"
    # a.bin was already right, so only the missing file crossed the network.
    assert [name for _, name, _ in hub.requested] == ["controller/toy/b.bin"]
    assert hub.requested[0][2] == REVISION


def test_materialize_replaces_a_file_whose_bytes_are_wrong(tmp_path, monkeypatch):
    remote = _bundle(tmp_path / "remote" / "controller" / "toy", {"a.bin": b"one"})
    local = _bundle(tmp_path / "toy", {"a.bin": b"tampered"})
    write_pin(local, _pin_for(remote))
    _install_hub(monkeypatch, _FakeHub(tmp_path / "remote"))

    materialize(local)

    assert (local / "a.bin").read_bytes() == b"one"


def test_materialize_refuses_bytes_that_do_not_match_the_pin(tmp_path, monkeypatch):
    remote = _bundle(tmp_path / "remote" / "controller" / "toy", {"a.bin": b"one"})
    local = tmp_path / "toy"
    local.mkdir()
    pin = _pin_for(remote)
    write_pin(local, pin)
    (remote / "a.bin").write_bytes(b"something else entirely")
    _install_hub(monkeypatch, _FakeHub(tmp_path / "remote"))

    with pytest.raises(ModelStoreError, match="does not match its pin"):
        materialize(local)


def test_offline_says_what_is_missing_instead_of_fetching(tmp_path, monkeypatch):
    remote = _bundle(tmp_path / "remote" / "controller" / "toy", {"a.bin": b"one"})
    local = tmp_path / "toy"
    local.mkdir()
    write_pin(local, _pin_for(remote))
    hub = _FakeHub(tmp_path / "remote")
    _install_hub(monkeypatch, hub)

    with pytest.raises(ModelStoreError, match="offline"):
        materialize(local, offline=True)
    assert hub.requested == []


def test_ensure_model_passes_an_unpinned_directory_through(tmp_path):
    bundle = _bundle(tmp_path / "hand_built", {"a.bin": b"one"})
    assert ensure_model(bundle) == bundle


def test_find_pins_lists_every_pinned_directory(tmp_path):
    for kind, name in (("controller", "one"), ("controller", "two"), ("planner", "p")):
        bundle = _bundle(tmp_path / kind / name, {"a.bin": b"x"})
        write_pin(bundle, _pin_for(bundle, kind=kind))

    found = [p.relative_to(tmp_path).as_posix() for p in find_pins(tmp_path)]

    assert found == ["controller/one", "controller/two", "planner/p"]


def test_pin_local_directory_records_the_head_commit(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "toy", {"a.bin": b"one", "nested/b.bin": b"two"})

    class _Api:
        def __init__(self, token=None):
            pass

        def repo_info(self, repo_id, repo_type):
            return type("Info", (), {"sha": "b" * 40})()

    _install_hub(monkeypatch, type("Hub", (), {"HfApi": _Api}))

    pin = pin_local_directory(
        bundle, kind="controller", repo="acme/bundles", path_in_repo="controller/toy"
    )

    assert pin.revision == "b" * 40
    assert sorted(pin.files) == ["a.bin", "nested/b.bin"]
    assert load_pin(bundle).revision == pin.revision
    # The pin describes the bundle; it is never one of the pinned files.
    assert "model.pin.json" not in pin.files


def test_push_model_uploads_then_pins_the_new_commit(tmp_path, monkeypatch):
    bundle = _bundle(tmp_path / "toy", {"a.bin": b"one"})
    calls = {}

    class _Api:
        def __init__(self, token=None):
            pass

        def create_repo(self, repo_id, repo_type, private, exist_ok):
            calls["private"] = private

        def upload_folder(self, **kwargs):
            calls["upload"] = kwargs
            return type("Commit", (), {"oid": "c" * 40})()

    _install_hub(monkeypatch, type("Hub", (), {"HfApi": _Api}))

    pin = push_model(
        bundle, kind="controller", repo="acme/bundles", path_in_repo="controller/toy"
    )

    assert calls["private"] is True
    assert calls["upload"]["path_in_repo"] == "controller/toy"
    assert calls["upload"]["ignore_patterns"] == ["model.pin.json"]
    assert pin.revision == "c" * 40
    assert json.loads((bundle / "model.pin.json").read_text())["repo"] == "acme/bundles"
