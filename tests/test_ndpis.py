"""Hamamatsu NDPIS sets (backends/ndpis.py) against the members written
(tests/_ndpis_writer.py)."""
from __future__ import annotations

import os

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, MissingImagesWarning, UnsupportedFormatError
from micro_reader.backends import ndpis

from _ndpis_writer import write_set


def test_one_image_with_a_channel_per_member(tmp_path):
    path, bases, _ = write_set(str(tmp_path))
    with micro_reader.open(path) as f:
        assert (f.backend, f.format) == ("ndpis", "ndpis")
        image = f.images[0]
        assert (image.axes, image.shape, image.dtype) == ("cyx", (3, 96, 128), np.uint8)
        for c, band in enumerate((2, 1, 0)):                       # DAPI blue, FITC green, TRITC red
            np.testing.assert_array_equal(image[c], bases[c][..., band])
            np.testing.assert_array_equal(image[c, 10:50, 7:99], bases[c][10:50, 7:99, band])
        level = image.resolutions[1]
        assert level.shape == (3, 48, 64)
        np.testing.assert_array_equal(level[1], bases[1][::2, ::2, 1])
        channels = image.metadata.channels
        assert [c.name for c in channels] == ["DAPI 2 (387)", "FITC 2 (485)", "TRITC 2 (560)"]
        assert [c.emission_wavelength for c in channels] == \
            [(455.0, "nanometer"), (519.0, "nanometer"), (622.0, "nanometer")]
        # the first member's associated picture follows as a further series
        assert [im.index["series"] for im in f.images] == [0, 1] and f.images[1].shape == (20, 60, 3)


def test_all_levels_as_images(tmp_path):
    path, _, _ = write_set(str(tmp_path))
    with micro_reader.open(path, resolutions="all") as f:
        assert [im.index for im in f.images[:2]] == [{"series": 0, "resolution": 0},
                                                      {"series": 0, "resolution": 1}]


def test_members_open_alone(tmp_path):
    """A pointed set, index only: a member is an ordinary slide."""
    _, bases, paths = write_set(str(tmp_path))
    with micro_reader.open(paths[1]) as f:
        assert f.backend == "tiff" and f.images[0].axes == "yxs"
        np.testing.assert_array_equal(f.images[0][...], bases[1])


def test_a_missing_member_reads_as_zeros(tmp_path):
    path, bases, _ = write_set(str(tmp_path), skip=(1,))
    with pytest.warns(MissingImagesWarning, match="1 of its 3 NDPI files"):
        f = micro_reader.open(path)
    with f:
        image = f.images[0]
        assert not image[1].any()
        np.testing.assert_array_equal(image[2], bases[2][..., 0])
        assert image.metadata.channels[1].name is None


def test_every_member_missing(tmp_path):
    path, _, _ = write_set(str(tmp_path), skip=(0, 1, 2))
    with pytest.raises(CorruptFileError, match="none of its NDPI files"):
        micro_reader.open(path)


def test_a_colour_member_is_refused(tmp_path):
    path, _, _ = write_set(str(tmp_path), channels=(("DAPI", 455.0, 2), ("brightfield", 0.0, 0)))
    with pytest.raises(UnsupportedFormatError, match="names no colour band"):
        micro_reader.open(path)


def test_members_of_different_sizes_are_refused(tmp_path):
    path, _, _ = write_set(str(tmp_path), channels=(("a", 455.0, 2), ("b", 519.0, 1)),
                           shapes=[(96, 128), (96, 120)])
    with pytest.raises(CorruptFileError, match="at level 0"):
        micro_reader.open(path)


def test_set_file_errors(tmp_path):
    bad = tmp_path / "x.ndpis"
    bad.write_text("[NanoZoomer Digital Pathology Image Set]\nNoImages=2\nImage0=a.ndpi\n")
    with pytest.raises(CorruptFileError, match="Image1 is missing"):
        ndpis.read_set(str(bad))
    bad.write_bytes(b"\xef\xbb\xbf[NanoZoomer Digital Pathology Image Set]\r\nNoImages=1\r\nImage0=a b (1) .ndpi\r\n")
    assert ndpis.read_set(str(bad)) == ["a b (1) .ndpi"]


@pytest.mark.parametrize("wave,band", [(455.0, 2), (490.0, 2), (519.0, 1), (580.0, 1), (622.0, 0)])
def test_band_of_a_wavelength(wave, band):
    assert ndpis._band(wave, "x") == band


def test_group_files(tmp_path):
    path, _, paths = write_set(str(tmp_path))
    g = micro_reader.group_files(str(tmp_path))
    (group,) = g.groups
    assert (group.kind, group.format) == ("index", "ndpis") and os.path.samefile(group.main, path)
    assert {os.path.normcase(p) for p in group.members} == {os.path.normcase(p) for p in paths}
    assert not g.singles
