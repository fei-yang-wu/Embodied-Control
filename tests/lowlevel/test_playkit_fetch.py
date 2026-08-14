"""Playkit fetching: stdlib-HTTPS fallback, sha256 gating, idempotence."""

import hashlib
import json

import pytest

from embodied_control.lowlevel.playkit import _fetch_https, fetch_playkit


def _make_source_kit(root, files):
    shas = {}
    for path, payload in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        shas[path] = hashlib.sha256(payload).hexdigest()
    (root / "README.md").write_text("readme\n")
    (root / "playkit.json").write_text(json.dumps({"file_sha256": shas}))
    return shas


def _template(root):
    return "file://" + str(root) + "/{path}"


def test_https_fallback_downloads_and_verifies(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    files = {
        "bundles/b/policy.pt": b"policy-bytes",
        "reference/tree/qpos.memmap": b"qpos-bytes",
    }
    _make_source_kit(source, files)

    _fetch_https(
        dest, "rev", repo="r", progress=lambda _: None, url_template=_template(source)
    )

    for path, payload in files.items():
        assert (dest / path).read_bytes() == payload
    assert json.loads((dest / "playkit.json").read_text())["file_sha256"]
    assert (dest / "README.md").read_text() == "readme\n"


def test_https_fallback_rejects_sha_mismatch(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    _make_source_kit(source, {"bundles/b/policy.pt": b"payload"})
    kit = json.loads((source / "playkit.json").read_text())
    kit["file_sha256"]["bundles/b/policy.pt"] = "0" * 64
    (source / "playkit.json").write_text(json.dumps(kit))

    with pytest.raises(IOError, match="sha256 mismatch"):
        _fetch_https(
            dest, "rev", repo="r", progress=lambda _: None,
            url_template=_template(source),
        )
    assert not (dest / "playkit.json").exists()
    assert not (dest / "bundles/b/policy.pt").exists()


def test_fetch_playkit_is_idempotent_once_marker_exists(tmp_path, monkeypatch):
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "playkit.json").write_text("{}")

    def explode(*args, **kwargs):
        raise AssertionError("must not download when the marker exists")

    monkeypatch.setattr("embodied_control.lowlevel.playkit._fetch_https", explode)
    assert fetch_playkit(dest, "rev", progress=lambda _: None) == dest
