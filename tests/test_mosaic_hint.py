"""The file summary points to tiles="mosaic" where it would succeed (Workplan MR28).

With the default tiles="separate", a slide scan opens as its camera tiles, and its stored
pyramid is reached only through tiles="mosaic". print(f) names the option on each series
whose tiles can be placed on one canvas: CZI, Leica tile scans, FluoView tiling sets and
BigDataViewer without turned or z-offset views. It never points at an option that would
then raise, and it uses only what opening read.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

import micro_reader
from micro_reader._outline import outline_lines

from _czi_writer import raw_payload, write_czi
from _lif_writer import image, write_lif

RNG = np.random.default_rng(28)
HINT = 'tiles="mosaic"'
DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))


def _series_lines(path, **kwargs):
    with micro_reader.open(path, **kwargs) as f:
        return [line for line in outline_lines(f) if line.startswith("series")], repr(f)


def _czi_tiles(levels: bool):
    """Two 16 x 16 tiles side by side in series 0 (mosaic 0, 1); with *levels*, the
    series' pyramid subblock at zoom 0.5 (no mosaic index, as ZEN writes it)."""
    a = RNG.integers(0, 4000, (16, 32)).astype("u2")
    tiles = [dict(series=0, mosaic=m, plane={"C": 0}, x=16 * m, y=0, height=16, width=16,
                  dtype="u2", compression=0, payload=raw_payload(a[:, 16 * m:16 * (m + 1)]))
             for m in range(2)]
    if levels:
        tiles.append(dict(series=0, plane={"C": 0}, x=0, y=0, height=16, width=32,
                          stored=(16, 8), pyramid=2, dtype="u2", compression=0,
                          payload=raw_payload(a[::2, ::2])))
    return tiles


@pytest.mark.parametrize("levels", [False, True], ids=["no-levels", "stored-levels"])
def test_a_czi_mosaic_shows_the_hint(tmp_path, levels):
    path = str(tmp_path / "scan.czi")
    write_czi(path, _czi_tiles(levels))
    (line,), shown = _series_lines(path)
    assert "tiles 0-1" in line and HINT in line and HINT in shown
    # the levels are named only where the mosaic has more than a tile
    assert ("with its 2 stored levels" in line) == levels
    with micro_reader.open(path, tiles="mosaic") as f:          # and it does succeed
        assert len(f.images[0].resolutions) == (2 if levels else 1)


def test_no_hint_when_opened_as_a_mosaic(tmp_path):
    path = str(tmp_path / "scan.czi")
    write_czi(path, _czi_tiles(True))
    (line,), _ = _series_lines(path, tiles="mosaic")
    assert HINT not in line


def test_a_leica_tile_scan_shows_the_hint(tmp_path):
    data = RNG.integers(0, 4000, (1, 1, 1, 3, 8, 9)).astype("u2")
    grid = [(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0), (2, 0, 2e-4, 0.0)]
    path = str(tmp_path / "scan.lif")
    write_lif(path, [image("scan", data, "tczmyx", tiles=grid)])
    (line,), _ = _series_lines(path)
    assert HINT in line and "stored levels" not in line
    with micro_reader.open(path, tiles="mosaic") as f:
        assert len(f.images) == 1


def test_a_leica_tile_scan_without_positions_shows_no_hint(tmp_path):
    """FlipX: the tiles have no stage position, and tiles="mosaic" refuses them."""
    data = RNG.integers(0, 4000, (1, 1, 1, 2, 8, 9)).astype("u2")
    path = str(tmp_path / "flipped.lif")
    write_lif(path, [image("flipped", data, "tczmyx", tiles=[(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0)],
                           tile_flags={"FlipX": "1"})])
    (line,), _ = _series_lines(path)
    assert "tiles 0-1" in line and HINT not in line


def test_tiles_of_another_format_show_no_hint(tmp_path):
    """TIFF tiles (a J K grid) cannot be composed: no hint."""
    data = RNG.integers(0, 4000, (2, 4, 9, 11)).astype("u2")
    path = str(tmp_path / "grid.tif")
    tifffile.imwrite(path, data, metadata={"axes": "JKYX"}, photometric="minisblack")
    (line,), _ = _series_lines(path)
    assert "tiles 0-7" in line and HINT not in line


@pytest.mark.parametrize("name, hinted", [
    (os.path.join("osf-bufza-bigstitcher", "grid-2d-stitched-h5", "dataset.xml"), True),
    # tiles at several z positions: tiles="mosaic" refuses them (Workplan MR21)
    (os.path.join("zenodo-8060528", "CH00_000000.xml"), False),
])
def test_bigdataviewer_hint_follows_what_the_mosaic_accepts(name, hinted):
    path = os.path.join(DATA, "formats", "h5", "bdv", name)
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path} (set MICROSCOPY_DATA)")
    lines, _ = _series_lines(path)
    assert any(HINT in line for line in lines) == hinted
    if hinted:
        with micro_reader.open(path, tiles="mosaic"):
            pass
    else:
        with pytest.raises(micro_reader.MicroReaderError, match="lie apart"):
            micro_reader.open(path, tiles="mosaic")
