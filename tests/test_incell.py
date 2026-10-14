"""InCell plates (backends/incell.py, docs/HCS.md) and ragged images
(docs/AXES.md): synthetic ``.xdce`` files with one small TIFF per plane."""
from __future__ import annotations

import os

import numpy as np
import pytest

import micro_reader

tifffile = pytest.importorskip("tifffile")
RNG = np.random.default_rng(41)
Y, X = 8, 12


def _write(folder, waves, fields=1, wells=((3, 5),), times=1, skip=()):
    """*waves*: [(name, focus z of each plane in um)]; a wave with one entry is
    2-D.  *wells*: 1-based (row, column).  *skip*: (row, col, field, t, wave, z)
    not written.  -> (xdce path, {key: array})."""
    os.makedirs(folder, exist_ok=True)
    wave_xml = "".join(
        f'<Wavelength index="{i}" name="{name}">'
        f'<ExcitationFilter name="{name}" unit="nm" wavelength="{400 + 50 * i}"/>'
        f'<EmissionFilter name="{name}" unit="nm" wavelength="{450 + 50 * i}"/></Wavelength>'
        for i, (name, _z) in enumerate(waves))
    images, data = [], {}
    for r, c in wells:
        for f in range(fields):
            for t in range(times):
                for w, (name, zs) in enumerate(waves):
                    for z, focus in enumerate(zs):
                        key = (r, c, f, t, w, z)
                        fn = f"{chr(64 + r)} - {c:02d}(fld {f + 1} wv {name} z {z + 1:02d} t {t}).tif"
                        if key not in skip:
                            a = RNG.integers(1, 4000, (Y, X)).astype("u2")
                            tifffile.imwrite(os.path.join(folder, fn), a)
                            data[key] = a
                        images.append(
                            f'<Image filename="{fn}"><Well label="x"><Row number="{r}"/>'
                            f'<Column number="{c}"/></Well>'
                            f'<FocusPosition unit="um" z="{focus}"/>'
                            f'<Identifier field_index="{f}" time_index="{t}" wave_index="{w}" '
                            f'z_index="{z}"/></Image>')
    xml = ('<?xml version="1.0" encoding="utf-8"?><ImageStack><AutoLeadAcquisitionProtocol>'
           '<ObjectiveCalibration pixel_height="1.3" pixel_width="1.3" unit="um"/>'
           f"<Wavelengths>{wave_xml}</Wavelengths>"
           '<Plate columns="12" name="Greiner 96 uClear" rows="8"/></AutoLeadAcquisitionProtocol>'
           f"<Images>{''.join(images)}</Images></ImageStack>")
    path = os.path.join(folder, "Plate_1.xdce")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return path, data


STACK = [100.0 + 4 * k for k in range(4)]                    # a 3-D channel: 4 slices, 4 um


def test_a_ragged_field_has_parts_and_a_padded_whole(tmp_path):
    # FITC 2-D at 108.0 (slice 2); DAPI 3-D; Cy3 2-D at 105.9 (nearest slice 1)
    path, data = _write(str(tmp_path), [("FITC", [108.0]), ("DAPI", STACK), ("Cy3", [105.9])],
                        fields=2)
    with micro_reader.open(path) as f:
        assert f.backend == "incell" and f.plates[0].name == "Plate_1"
        assert [w.name for w in f.plates[0].wells] == ["C5"]
        assert f.indices == [{"plate": 0, "acquisition": 0, "well": 0, "field": fd}
                             for fd in range(2)]
        whole = f.plates[0].wells[0].fields[1].image
        assert whole.is_ragged and whole.shape == (1, 3, 4, Y, X)
        flat, stack = whole.parts
        assert flat.shape == (1, 2, 1, Y, X) and stack.shape == (1, 1, 4, Y, X)
        assert flat.whole is whole and stack.whole is whole
        assert flat.index == {**whole.index, "part": 0} and stack.index["part"] == 1
        assert [c.name for c in flat.metadata.channels] == ["FITC", "Cy3"]
        assert [c.name for c in whole.metadata.channels] == ["FITC", "DAPI", "Cy3"]
        assert stack.metadata.pixel_sizes["z"] == micro_reader.Quantity(4.0, "micrometer")
        assert "z" not in flat.metadata.pixel_sizes
        # the parts: exact
        np.testing.assert_array_equal(flat[0, 1, 0], data[(3, 5, 1, 0, 2, 0)])
        for z in range(4):
            np.testing.assert_array_equal(stack[0, 0, z], data[(3, 5, 1, 0, 1, z)])
        # the whole: DAPI at its slices; FITC at slice 2, Cy3 at slice 1, zeros elsewhere
        np.testing.assert_array_equal(whole[0, 1], np.stack([data[(3, 5, 1, 0, 1, z)] for z in range(4)]))
        for c, at, w in ((0, 2, 0), (2, 1, 2)):
            np.testing.assert_array_equal(whole[0, c, at], data[(3, 5, 1, 0, w, 0)])
            assert not np.delete(whole[0, c], at, axis=0).any()
        assert "ragged: (1, 2, 1, 8, 12) + (1, 1, 4, 8, 12)" in f.outline()


def test_an_image_that_is_not_ragged_is_its_own_part(tmp_path):
    path, _ = _write(str(tmp_path), [("FITC", [100.0]), ("DAPI", [100.0])])
    with micro_reader.open(path) as f:
        (image,) = f.images
        assert not image.is_ragged and image.parts == [image] and image.whole is image
        assert image.shape == (1, 2, 1, Y, X) and "part" not in image.index


def test_parts_without_a_common_grid_are_read_only_as_parts(tmp_path):
    """A second stack with a 3 um step does not lie on the 4 um grid."""
    path, data = _write(str(tmp_path), [("DAPI", STACK), ("GFP", [100.0, 103.0, 106.0])])
    with micro_reader.open(path) as f:
        (whole,) = f.images
        assert whole.is_ragged
        with pytest.raises(micro_reader.UnsupportedFormatError, match="read its parts"):
            whole[0, 0, 0]
        np.testing.assert_array_equal(whole.parts[1][0, 0, 2], data[(3, 5, 0, 0, 1, 2)])


def test_a_single_plane_beyond_half_a_step_breaks_the_grid(tmp_path):
    path, _ = _write(str(tmp_path), [("DAPI", STACK), ("FITC", [130.0])])   # 18 um past the stack
    with micro_reader.open(path) as f:
        with pytest.raises(micro_reader.UnsupportedFormatError, match="read its parts"):
            f.images[0][...]


def test_missing_planes_follow_the_file_set_rule(tmp_path):
    with pytest.warns(micro_reader.MissingImagesWarning, match="1 fields miss some planes"):
        path, data = _write(str(tmp_path), [("FITC", [108.0]), ("DAPI", STACK)],
                            skip={(3, 5, 0, 0, 1, 3)})
        f = micro_reader.open(path)
    with f:
        whole = f.images[0]
        assert not whole[0, 1, 3].any() and not whole.parts[1][0, 0, 3].any()
        np.testing.assert_array_equal(whole[0, 1, 2], data[(3, 5, 0, 0, 1, 2)])


def test_channels_at_different_timepoints_are_refused(tmp_path):
    path, _ = _write(str(tmp_path), [("FITC", [100.0])], times=2)
    # a second channel imaged only at timepoint 0
    with open(path, encoding="utf-8") as fh:
        xml = fh.read()
    xml = xml.replace("</Images>", '<Image filename="extra.tif"><Well label="x"><Row number="3"/>'
                      '<Column number="5"/></Well><FocusPosition unit="um" z="100"/>'
                      '<Identifier field_index="0" time_index="0" wave_index="1" z_index="0"/>'
                      "</Image></Images>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    with pytest.raises(micro_reader.UnsupportedFormatError, match="different timepoints"):
        micro_reader.open(path)
