"""Reading after ``File.close()`` raises ``ValueError`` in every backend, from any
thread, and closing releases every image's handles (review of 2026-10-18: VSI
reopened its ETS files on a thread that had not read yet; file sets reopened
their members; TIFF levels left out by ``resolutions="base"`` kept their handles)."""
from __future__ import annotations

import threading

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402

import test_cv7000  # noqa: E402
from test_vsi import _levels, _simple  # noqa: E402


def _read_in_new_thread(image):
    """What reading *image* on a thread that never read it raises (None: it read)."""
    caught = []

    def run():
        try:
            np.asarray(image[...])
        except Exception as exc:                        # noqa: BLE001 - reported to the test
            caught.append(exc)
    t = threading.Thread(target=run)
    t.start()
    t.join()
    return caught[0] if caught else None


def test_vsi_levels_raise_after_close_on_any_thread(tmp_path):
    path = _simple(tmp_path, _levels((45, 70), 3))
    with micro_reader.open(path, resolutions="base") as f:
        levels = list(f.images[0].resolutions)
        np.asarray(levels[0][...])                      # this thread has a handle now
    assert len(levels) == 3
    for level in levels:
        with pytest.raises(ValueError, match="closed"):
            level[...]
        assert isinstance(_read_in_new_thread(level), ValueError)


def test_plate_fields_raise_after_close_on_any_thread(tmp_path):
    test_cv7000._write(str(tmp_path))
    with micro_reader.open(tmp_path / "P1.wpi") as f:
        field = f.images[0]
        np.asarray(field[...])
    with pytest.raises(ValueError, match="closed"):
        field[...]
    assert isinstance(_read_in_new_thread(field), ValueError)


def test_tiff_releases_the_handles_of_levels_it_does_not_list(tmp_path):
    path = str(tmp_path / "pyramid.ome.tif")
    base = np.random.default_rng(1).integers(0, 60000, (64, 96)).astype("u2")
    with tifffile.TiffWriter(path) as tw:
        tw.write(base, subifds=1, tile=(32, 32), metadata={"axes": "YX"})
        tw.write(base[::2, ::2], subfiletype=1, tile=(32, 32))
    with micro_reader.open(path, resolutions="base") as f:
        level = f.images[0].resolutions[1]
        np.asarray(level[...])
        handles = level._handles
        assert handles._all                             # the level opened its own file handle
    assert not handles._all                             # closing the file closed it
    with pytest.raises(ValueError, match="closed"):
        level[...]
