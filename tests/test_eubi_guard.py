"""The import guard against old eubi (plan 8.28, E2): ``import micro_reader`` raises an
ImportError beside eubi-bridge 0.1.3 <= v < 0.3.0.dev0 or eubi-bridge-lite v < 0.3.0.dev0,
which read the old index name "scene".

Each case puts fake ``*.dist-info`` folders (METADATA with Name and Version, optionally a
``direct_url.json``) in a temporary folder first on ``sys.path`` and imports micro_reader in a
fresh ``python -I -B``, so no real eubi is needed and the venv is untouched. The folder also
holds a fake ``eubi_bridge`` package that raises when imported: the guard must read versions
from metadata only."""
import importlib.metadata
import json
import os
import pathlib
import subprocess
import sys
import zipfile
from importlib.machinery import PathFinder
from urllib.request import url2pathname

import pytest

import micro_reader
from micro_reader import _eubi_guard
from micro_reader._eubi_guard import FIRST_SERIES_EUBI, SKIP_ENV, version_key

SRC = os.path.dirname(os.path.dirname(os.path.abspath(micro_reader.__file__)))
NAMES = ("eubi-bridge", "eubi-bridge-lite")

PROBE = r"""
import json, sys, time, warnings
sys.path[:0] = [sys.argv[1], sys.argv[2]]
out = {}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    t0 = time.perf_counter()
    try:
        import micro_reader
        out["result"] = "ok"
    except ImportError as exc:
        out["result"], out["message"] = "ImportError", str(exc)
    out["seconds"] = time.perf_counter() - t0
out["warnings"] = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
out["eubi_imported"] = "eubi_bridge" in sys.modules
out["numpy_imported"] = "numpy" in sys.modules
out["backends_imported"] = any(m.startswith("micro_reader.backends") for m in sys.modules)
print(json.dumps(out))
"""


def _fake(root, dist, version, editable=False):
    folder = os.path.join(root, f"{dist.replace('-', '_')}-{version}.dist-info")
    os.makedirs(folder)
    with open(os.path.join(folder, "METADATA"), "w", encoding="utf-8") as fh:
        fh.write(f"Metadata-Version: 2.1\nName: {dist}\nVersion: {version}\n")
    if editable:
        with open(os.path.join(folder, "direct_url.json"), "w", encoding="utf-8") as fh:
            json.dump({"url": "file:///C:/work/eubi-bridge", "dir_info": {"editable": True}}, fh)


def _real_eubi():
    """The eubi distributions really installed here (they would shadow the 'not faked' ones)."""
    found = set()
    for name in NAMES:
        for spelling in (name, name.replace("-", "_")):
            try:
                importlib.metadata.distribution(spelling)
                found.add(name)
            except importlib.metadata.PackageNotFoundError:
                pass
    return found


def _probe(tmp_path, dists, env=None):
    real = _real_eubi() - {d[0] for d in dists}
    if real:
        pytest.skip(f"a real {', '.join(sorted(real))} is installed here; run in a venv without eubi")
    for dist in dists:
        _fake(str(tmp_path), *dist)
    pkg = tmp_path / "eubi_bridge"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("raise RuntimeError('the guard imported eubi_bridge')\n")
    environ = {k: v for k, v in os.environ.items() if k != SKIP_ENV}
    environ.update(env or {})
    p = subprocess.run([sys.executable, "-I", "-B", "-c", PROBE, str(tmp_path), SRC],
                       capture_output=True, text=True, env=environ, timeout=120)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert not out["eubi_imported"], "the guard imported eubi_bridge"
    return out


def _assert_refused(out, dist, version):
    assert out["result"] == "ImportError", out
    msg = out["message"]
    for part in ("does not support micro-reader", f"{dist} {version}", f"micro-reader {micro_reader.__version__}",
                 f"pip install -U '{dist}>={FIRST_SERIES_EUBI}'", "pip install 'micro-reader<0.1'",
                 f"{SKIP_ENV}=1"):
        assert part in msg, (part, msg)
    # a refused import is cheap and loads nothing else
    assert out["seconds"] < 0.5, out["seconds"]
    assert not out["numpy_imported"] and not out["backends_imported"], out


REFUSED = [("eubi-bridge", "0.1.3"), ("eubi-bridge", "0.1.7"), ("eubi-bridge", "0.2.0"),
           ("eubi-bridge-lite", "0.1.4"), ("eubi-bridge-lite", "0.2.0"), ("eubi-bridge-lite", "0.2.1")]
SILENT = [("eubi-bridge", "0.1.2"), ("eubi-bridge", "0.3.0.dev0"), ("eubi-bridge", "0.3.0rc1"),
          ("eubi-bridge", "0.3.0"), ("eubi-bridge-lite", "0.3.0.dev0"), ("eubi-bridge-lite", "0.3.0")]


@pytest.mark.parametrize("dist, version", REFUSED, ids=[f"{d}-{v}" for d, v in REFUSED])
def test_old_eubi_is_refused(tmp_path, dist, version):
    out = _probe(tmp_path, [(dist, version)])
    _assert_refused(out, dist, version)
    assert "editable install" not in out["message"]


@pytest.mark.parametrize("dist, version", SILENT, ids=[f"{d}-{v}" for d, v in SILENT])
def test_eubi_that_reads_series_or_never_reads_micro_reader_imports_silently(tmp_path, dist, version):
    out = _probe(tmp_path, [(dist, version)])
    assert out["result"] == "ok", out
    assert out["warnings"] == [], out


def test_no_eubi_imports_silently(tmp_path):
    out = _probe(tmp_path, [])
    assert out["result"] == "ok" and out["warnings"] == [], out


def test_one_refused_distribution_is_enough(tmp_path):
    out = _probe(tmp_path, [("eubi-bridge", "0.3.0"), ("eubi-bridge-lite", "0.2.0")])
    _assert_refused(out, "eubi-bridge-lite", "0.2.0")


def test_an_editable_install_is_checked_and_told_to_reinstall(tmp_path):
    out = _probe(tmp_path, [("eubi-bridge", "0.1.3", True)])
    _assert_refused(out, "eubi-bridge", "0.1.3")
    assert "eubi-bridge is an editable install" in out["message"]
    assert "reinstall it" in out["message"]
    assert "pip install -e " + url2pathname("/C:/work/eubi-bridge") + ")" in out["message"]


def test_an_editable_install_that_reads_series_imports_silently(tmp_path):
    out = _probe(tmp_path, [("eubi-bridge", "0.3.0.dev0", True)])
    assert out["result"] == "ok" and out["warnings"] == [], out


def test_the_variable_skips_with_one_warning(tmp_path):
    out = _probe(tmp_path, [("eubi-bridge", "0.2.0")], {SKIP_ENV: "1"})
    assert out["result"] == "ok", out
    assert len(out["warnings"]) == 1, out
    assert f"{SKIP_ENV}=1" in out["warnings"][0] and "skipped" in out["warnings"][0]


@pytest.mark.parametrize("value", ["0", "", "true", "yes"])
def test_any_other_value_does_not_skip(tmp_path, value):
    out = _probe(tmp_path, [("eubi-bridge", "0.2.0")], {SKIP_ENV: value})
    _assert_refused(out, "eubi-bridge", "0.2.0")


def test_an_unreadable_version_warns_once_and_imports(tmp_path):
    out = _probe(tmp_path, [("eubi-bridge-lite", "not-a-version")])
    assert out["result"] == "ok", out
    assert len(out["warnings"]) == 1 and "'not-a-version'" in out["warnings"][0], out


def _in_process(tmp_path, monkeypatch, metadata, direct_url=None):
    """Point the guard at one fake eubi-bridge dist-info and nothing else, in this process."""
    folder = tmp_path / "site" / "eubi_bridge-0.2.0.dist-info"
    folder.mkdir(parents=True)
    (folder / "METADATA").write_text(metadata, encoding="utf-8")
    if direct_url is not None:
        (folder / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")
    monkeypatch.setattr(sys, "meta_path", [PathFinder])
    monkeypatch.setattr(sys, "path", [str(tmp_path / "site")])
    monkeypatch.setattr(_eubi_guard, "_warned", set())
    monkeypatch.delenv(SKIP_ENV, raising=False)


def test_a_missing_version_is_said_in_words(tmp_path, monkeypatch):
    _in_process(tmp_path, monkeypatch, "Metadata-Version: 2.1\nName: eubi-bridge\n")
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _eubi_guard.check()
    texts = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(texts) == 1, texts
    assert "the installed eubi-bridge records no version" in texts[0], texts[0]
    assert "None" not in texts[0], texts[0]


def test_an_editable_install_without_a_recorded_folder_names_a_bare_placeholder(tmp_path, monkeypatch):
    _in_process(tmp_path, monkeypatch, "Metadata-Version: 2.1\nName: eubi-bridge\nVersion: 0.2.0\n",
                {"dir_info": {"editable": True}})
    with pytest.raises(ImportError, match="does not support micro-reader") as info:
        _eubi_guard.check()
    msg = str(info.value)
    assert "(pip install -e <its folder>)" in msg, msg
    assert '"<its folder>"' not in msg, msg


def test_editable_folders_are_quoted_only_when_they_hold_a_space():
    for folder, shown in (("", "<its folder>"), (r"C:\work\eubi", r"C:\work\eubi"),
                          (r"C:\My Work\eubi", r'"C:\My Work\eubi"')):
        msg = _eubi_guard.refusal("eubi-bridge", "0.2.0", folder)
        assert "does not support micro-reader" in msg
        assert f"(pip install -e {shown})" in msg, msg
    assert "editable install" not in _eubi_guard.refusal("eubi-bridge", "0.2.0")


def test_the_underscore_spelling_is_found(tmp_path):
    out = _probe(tmp_path, [("eubi_bridge_lite", "0.2.1")])
    _assert_refused(out, "eubi-bridge-lite", "0.2.1")


def test_version_ordering():
    ordered = ["0.1.2", "0.1.3", "0.1.4", "0.1.7", "0.2.0", "0.2.0.post1", "0.2.1",
               "0.3.0.dev0", "0.3.0a1.dev0", "0.3.0a1", "0.3.0b2", "0.3.0rc1.dev3", "0.3.0rc1",
               "0.3.0", "0.3.0.post1.dev0", "0.3.0.post1", "0.3.1", "1.0", "1!0.0.1"]
    keys = [version_key(v) for v in ordered]
    assert None not in keys
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)
    assert version_key("0.3") == version_key("0.3.0") == version_key("v0.3.0") == version_key(" 0.3.0 ")
    assert version_key("0.2.0+local.1") == version_key("0.2.0")
    assert version_key("0.3.0RC1") == version_key("0.3.0rc1") == version_key("0.3.0-rc.1")
    assert version_key("0.3.0-1") == version_key("0.3.0.post1")
    below = version_key(FIRST_SERIES_EUBI + ".dev0")
    for v in ("0.1.3", "0.1.7", "0.2.0", "0.2.1", "0.1.4"):
        assert version_key(v) < below
    for v in ("0.3.0.dev0", "0.3.0rc1", "0.3.0"):
        assert version_key(v) >= below
    for bad in ("", "not-a-version", "0.3.0 foo", "0..3", None):
        assert version_key(bad) is None


# The folder listing that spares importing importlib.metadata when no eubi can be installed.

def test_the_listing_finds_every_spelling(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", [PathFinder])
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(sys, "path", [str(empty), str(tmp_path / "missing")])
    assert not _eubi_guard.may_be_installed()
    for name in ("EuBI_Bridge-0.2.0.dist-info", "eubi.bridge.lite-0.2.1.dist-info", "eubi_bridge.egg-info",
                 "eubi_bridge-0.1.3-py3.11.egg"):
        folder = tmp_path / name.replace(".", "_")
        (folder / name).mkdir(parents=True)
        monkeypatch.setattr(sys, "path", [str(empty), str(folder)])
        assert _eubi_guard.may_be_installed(), name
    egg = tmp_path / "eubi_bridge-0.1.3-py3.11.egg"
    egg.mkdir()
    monkeypatch.setattr(sys, "path", [str(egg)])
    assert _eubi_guard.may_be_installed()


def test_the_listing_says_maybe_for_what_it_cannot_see(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "meta_path", [PathFinder])
    archive = tmp_path / "packages.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("eubi_bridge-0.2.0.dist-info/METADATA", "Name: eubi-bridge\nVersion: 0.2.0\n")
    monkeypatch.setattr(sys, "path", [str(archive)])
    assert _eubi_guard.may_be_installed()
    with pytest.raises(ImportError, match="eubi-bridge 0.2.0 does not support micro-reader"):
        _eubi_guard.check()


class _Finder(importlib.metadata.DistributionFinder):
    """An import finder that reports a distribution no folder listing can see."""

    def __init__(self, dist_info):
        self.dist_info = dist_info

    def find_spec(self, *args, **kwargs):
        return None

    def find_distributions(self, context=importlib.metadata.DistributionFinder.Context()):
        if context.name and context.name.replace("_", "-").lower() == "eubi-bridge":
            yield importlib.metadata.PathDistribution(pathlib.Path(self.dist_info))


def test_a_finder_that_reports_distributions_is_asked(tmp_path, monkeypatch):
    hidden = tmp_path / "hidden"
    _fake(str(hidden), "eubi-bridge", "0.2.0")
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(sys, "path", [str(empty)])
    monkeypatch.setattr(sys, "meta_path", [PathFinder, _Finder(hidden / "eubi_bridge-0.2.0.dist-info")])
    assert _eubi_guard.may_be_installed()
    with pytest.raises(ImportError, match="eubi-bridge 0.2.0 does not support micro-reader"):
        _eubi_guard.check()
