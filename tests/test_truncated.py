"""Cut files fail loudly: a copy cut anywhere either raises ``CorruptFileError`` or reads
exactly the pixels of the intact file (the cut part was not needed).  Never fewer pages,
other pages, zeros, or an ``IndexError`` / ``KeyError`` / ``struct.error`` from inside a parser
(real-data round, row F: truncated OME-TIFF, Micro-Manager, MetaMorph, NDPI, SVS, LSM, CZI)."""
from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402
from micro_reader import CorruptFileError  # noqa: E402

from _czi_writer import raw_payload, write_czi  # noqa: E402

try:
    import imagecodecs  # noqa: F401
    HAVE_CODECS = True
except ImportError:
    HAVE_CODECS = False


def _data(shape, dtype="u2", seed=0):
    return np.random.default_rng(seed).integers(0, 60000, size=shape).astype(dtype)


def _read_all(path):
    with micro_reader.open(path) as f:
        return [np.asarray(im[...]) for im in f.images]


def _cuts(size):
    """Byte counts to cut at: the header, every 7 %, and just before the end."""
    return sorted(n for n in {16, 200, 4096, *range(size // 14, size, size // 14), size - 64, size - 1}
                  if 0 < n < size)


def _check_every_cut(path, tmp_path):
    intact = _read_all(path)
    data = open(path, "rb").read()
    for n in _cuts(len(data)):
        cut = tmp_path / f"cut_{n}{path.suffix}"
        cut.write_bytes(data[:n])
        try:
            got = _read_all(cut)
        except CorruptFileError:
            continue
        assert len(got) == len(intact) and all(
            a.shape == b.shape and np.array_equal(a, b) for a, b in zip(got, intact)), \
            f"cut at byte {n} of {len(data)} read silently as something else"


def test_cut_multipage_tiff(tmp_path):
    path = tmp_path / "pages.tif"
    with tifffile.TiffWriter(path) as tw:
        for k in range(6):
            tw.write(_data((40, 50), seed=k), photometric="minisblack", description=f"page {k}")
    _check_every_cut(path, tmp_path)


def test_cut_ome_tiff(tmp_path):
    path = tmp_path / "stack.ome.tif"
    tifffile.imwrite(path, _data((3, 4, 32, 40)), ome=True, metadata={"axes": "CZYX"})
    _check_every_cut(path, tmp_path)


def test_cut_imagej_hyperstack(tmp_path):
    path = tmp_path / "hyper.tif"
    tifffile.imwrite(path, _data((2, 3, 32, 32)), imagej=True, metadata={"axes": "ZCYX"})
    _check_every_cut(path, tmp_path)


@pytest.mark.skipif(not HAVE_CODECS, reason="imagecodecs writes the compressed tiles")
def test_cut_tiled_compressed_tiff(tmp_path):
    path = tmp_path / "tiled.tif"
    tifffile.imwrite(path, _data((128, 160)), tile=(32, 32), compression="zlib")
    _check_every_cut(path, tmp_path)


def test_cut_czi(tmp_path):
    path = tmp_path / "tiles.czi"
    a, b = _data((24, 30), seed=1), _data((24, 30), seed=2)
    write_czi(path, [dict(x=0, y=0, width=30, height=24, dtype="u2", compression=0, payload=raw_payload(a)),
                     dict(x=0, y=0, width=30, height=24, dtype="u2", compression=0, payload=raw_payload(b),
                          plane={"C": 1})])
    _check_every_cut(path, tmp_path)
