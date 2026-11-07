"""A plate's plane file opens the plate, as every other member of a set opens its set
(micro-reader 0.4.0; before, it depended on the format), and only itself with
``open(multifile=False)``.  The name alone says nothing: a file named like a plane
with no index near it is a TIFF."""
from __future__ import annotations

import glob
import os

import numpy as np
import pytest

import micro_reader

tifffile = pytest.importorskip("tifffile")

import test_columbus  # noqa: E402
import test_cv7000  # noqa: E402
import test_harmony  # noqa: E402
import test_incell  # noqa: E402
import test_jdce  # noqa: E402
import test_metaxpress  # noqa: E402

WRITERS = {
    "cv7000": lambda d: test_cv7000._write(d),
    "harmony": lambda d: test_harmony._write_harmony(d),
    "columbus": lambda d: test_columbus._write(d),
    "jdce": lambda d: test_jdce._write(d),
    "incell": lambda d: test_incell._write(d, [("FITC", [100.0]), ("DAPI", [100.0])]),
    "metaxpress": lambda d: test_metaxpress._write(d, [(1, 1)]),
}
LONE_NAMES = ["P_B02_T0001F001L01A01Z01C01.tif", "r03c03f01p01-ch1sk1fk1fl1.tiff", "012013-1.tif",
              "A - 1(fld 1 wv Cy5 - Cy5).tif", "t1_B02_s1_w1_z1.tif", "P_A01_s1_w1.TIF"]


def _plane(folder):
    planes = glob.glob(os.path.join(glob.escape(folder), "**", "*.tif*"), recursive=True)
    return sorted(planes)[0]


@pytest.mark.parametrize("name", WRITERS)
def test_a_plane_opens_its_plate(tmp_path, name):
    folder = str(tmp_path / "plate")
    os.makedirs(folder)
    WRITERS[name](folder)
    plane = _plane(folder)
    with micro_reader.open(plane) as f:
        assert len(f.plates) == 1 and f.path != plane
    with micro_reader.open(plane, multifile=False) as f:
        assert f.path == plane and not f.plates


@pytest.mark.parametrize("name", LONE_NAMES)
def test_a_file_named_like_a_plane_without_an_index_is_a_tiff(tmp_path, name):
    path = str(tmp_path / name)
    tifffile.imwrite(path, np.zeros((6, 8), np.uint8))
    with micro_reader.open(path) as f:
        assert f.backend == "tiff" and not f.plates
