"""Image names: one rule for every format and entry point (decision MR-2, micro-reader
0.4.0; _naming.py).  Formats' own names and plates are covered in their backends' tests
(Leica element names in test_lif, slide roles in test_tiff_variants, plates in test_hcs,
the BigDataViewer entry points in test_hdf5); here the rule's own parts."""
from __future__ import annotations

import os

import numpy as np
import pytest

import micro_reader

tifffile = pytest.importorskip("tifffile")

import test_companion  # noqa: E402

NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def _ome(path, names, shape=(6, 8)):
    """An OME-TIFF with one image per name (None: an Image without a Name)."""
    images = []
    for k, name in enumerate(names):
        named = f' Name="{name}"' if name is not None else ""
        images.append(f'<Image ID="Image:{k}"{named}><Pixels ID="Pixels:{k}" DimensionOrder="XYZCT" '
                      f'Type="uint8" SizeX="{shape[1]}" SizeY="{shape[0]}" SizeZ="1" SizeC="1" '
                      f'SizeT="1"><Channel ID="Channel:{k}" SamplesPerPixel="1"/>'
                      f'<TiffData IFD="{k}" PlaneCount="1"/></Pixels></Image>')
    xml = f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="{NS}">{"".join(images)}</OME>'
    with tifffile.TiffWriter(path) as tw:
        for k in range(len(names)):
            tw.write(np.full(shape, k, np.uint8), description=xml if k == 0 else None,
                     metadata=None)


def test_one_image_without_a_name_is_named_after_its_file(tmp_path):
    path = tmp_path / "cells_01.tif"
    tifffile.imwrite(path, np.zeros((6, 8), np.uint8))
    with micro_reader.open(str(path)) as f:
        assert f.images[0].name == "cells_01"                   # the stem, no extension


def test_the_formats_own_name_wins(tmp_path):
    path = str(tmp_path / "plate_scan.ome.tif")
    _ome(path, ["nuclei", "membrane"])
    with micro_reader.open(path) as f:
        assert [im.name for im in f.images] == ["nuclei", "membrane"]


def test_images_the_format_does_not_name_are_named_by_their_index(tmp_path):
    path = str(tmp_path / "two.ome.tif")
    _ome(path, [None, None])
    with micro_reader.open(path) as f:
        assert [im.name for im in f.images] == ["series 0", "series 1"]


def test_two_images_of_one_name_get_what_differs_appended(tmp_path):
    path = str(tmp_path / "same.ome.tif")
    _ome(path, ["scan", "scan", "other"])
    with micro_reader.open(path) as f:
        assert [im.name for im in f.images] == ["scan, series 0", "scan, series 1", "other"]


def test_every_level_carries_its_images_name(tmp_path):
    path = str(tmp_path / "pyramid.ome.tif")
    with tifffile.TiffWriter(path) as tw:
        tw.write(np.zeros((64, 64), np.uint8), subifds=1, tile=(16, 16),
                 metadata={"axes": "YX", "Name": "slide"})
        tw.write(np.zeros((32, 32), np.uint8), subfiletype=1, tile=(16, 16))
    with micro_reader.open(path, resolutions="all") as f:
        assert {im.name for im in f.images} == {"slide"} and len(f.images) == 2


def test_a_set_is_named_by_its_index_file_whichever_file_is_opened(tmp_path):
    # MR-2: the same name from every entry point (the gap hunt's Luxendo case)
    companion, _ = test_companion._write(str(tmp_path), images=1)
    member = next(os.path.join(str(tmp_path), n) for n in sorted(os.listdir(tmp_path))
                  if n.endswith(".tif"))
    with micro_reader.open(companion) as f, micro_reader.open(member) as g:
        assert [im.name for im in g.images] == [im.name for im in f.images]


def test_is_pyramid_is_deprecated_for_level(tmp_path):
    path = tmp_path / "a.tif"
    tifffile.imwrite(path, np.zeros((6, 8), np.uint8))
    with micro_reader.open(str(path)) as f:
        image = f.images[0]
        with pytest.warns(DeprecationWarning, match="image.level == 0"):
            assert image.is_pyramid
        assert image.level == 0 and not image.is_pyramidal
