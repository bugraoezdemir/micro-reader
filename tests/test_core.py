"""micro-reader-core, the native read engine (Workplan MR23): the module builds,
imports and links the codecs crate of this checkout.

Skips where micro_reader_core is not installed: micro-reader does not depend on it
yet.  CI builds its wheels beside the codecs' and installs them in the test jobs.
"""
import pathlib
import tomllib

import pytest

core = pytest.importorskip("micro_reader_core")

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _crate_version(folder: str) -> str:
    with open(ROOT / folder / "Cargo.toml", "rb") as f:
        return tomllib.load(f)["package"]["version"]


def test_versions_match_the_crates_of_this_checkout():
    assert core.__version__ == _crate_version("core")
    # the codecs crate compiled into core is this checkout's, not a released one
    assert core.CODECS_VERSION == _crate_version("codecs")


def test_core_links_the_c_decoders_the_codecs_module_has():
    codecs = pytest.importorskip("micro_reader_codecs")
    assert core.HAS_LIBJPEG_TURBO == codecs.HAS_LIBJPEG_TURBO
    assert core.HAS_JXRLIB == codecs.HAS_JXRLIB
    assert core.ZSTD_VERSION == codecs.zstd_version()
