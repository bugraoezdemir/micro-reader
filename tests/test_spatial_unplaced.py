"""The formats micro-reader places nowhere (plan 4.12, ``conftest.UNPLACED_FORMATS``):
a file declares no coordinate system and no transformation, every image it gives
out (levels and parts too) has position {} and no coordinate system, and
declaring reads no metadata.  One synthetic file per format, from its writer
(``tests/_<format>_writer.py``) or the helper of its own test module."""
from __future__ import annotations

import os

import numpy as np
import pytest

import micro_reader
from micro_reader._base import Image
from micro_reader._spatial import check_invariants

tifffile = pytest.importorskip("tifffile")

import conftest  # noqa: E402
import test_dv  # noqa: E402
import test_incell  # noqa: E402
import test_metaxpress  # noqa: E402
import test_mrc  # noqa: E402
import test_tiff_variants  # noqa: E402
import test_vsi  # noqa: E402
from _ics_writer import write_ics  # noqa: E402
from _metamorph_writer import write_set, write_stk  # noqa: E402
from _micromanager_writer import write_acquisition  # noqa: E402
from _nd2_legacy_writer import write_legacy_nd2  # noqa: E402
from _oif_writer import write_oib, write_oif  # noqa: E402
from _oir_writer import write_oir  # noqa: E402
from _raster_writer import write_bmp, write_png  # noqa: E402
from _tiff_writer import write_tiff  # noqa: E402

RNG = np.random.default_rng(12)


def _u2(shape):
    return RNG.integers(1, 4000, size=shape).astype(np.uint16)


def _h5py():
    pytest.importorskip("h5py")
    import _hdf5_writer
    return _hdf5_writer


# -- one synthetic file per format: name -> (tmp_path -> path) ---------------------------

def _metamorph(tmp_path):
    nd, _, _ = write_set(str(tmp_path), stages=["B2", "C4"], times=2,
                         stage_xy=lambda s: (70360.2 - 39143.2 * s, 9839.7 - 18184.3 * s))
    return nd


def _stk(tmp_path):
    path = str(tmp_path / "lone.STK")
    write_stk(path, _u2((3, 8, 12)), z=[0.0, 2.0, 4.0])
    return path


def _incell(tmp_path):
    path, _ = test_incell._write(str(tmp_path), [("DAPI", [0.0]), ("FITC", [0.0])], fields=2)
    return path


def _oif(tmp_path):
    return write_oif(str(tmp_path), "a", _u2((2, 2, 3, 20, 24)))


def _oib(tmp_path):
    return write_oib(tmp_path / "a.oib", _u2((2, 2, 3, 20, 24)))


def _nd2_legacy(tmp_path):
    path = str(tmp_path / "legacy.nd2")
    write_legacy_nd2(path, _u2((4, 2, 12, 14)), [("T", 2), ("P", 2)])
    return path


def _oir(tmp_path):
    path = str(tmp_path / "plain.oir")
    write_oir(path, _u2((1, 2, 3, 16, 20)))
    return path


def _vsi(tmp_path):
    return test_vsi._simple(tmp_path, test_vsi._levels((45, 70), 3))


def _bif(tmp_path):
    path = str(tmp_path / "slide.bif")
    test_tiff_variants._bif(path)
    return path


def _bif_iscan(tmp_path):
    path = str(tmp_path / "iscan.bif")
    test_tiff_variants._iscan_bif(path)
    return path


def _philips(tmp_path):
    path = str(tmp_path / "slide.tiff")
    rgb = test_tiff_variants._rgb
    with tifffile.TiffWriter(path) as tw:
        tw.write(rgb((64, 96)), metadata=None, photometric="rgb", tile=(32, 32),
                 software="Philips DP v1.0", description=test_tiff_variants.PHILIPS_XML)
        tw.write(rgb((32, 48)), metadata=None, photometric="rgb", tile=(32, 32),
                 software="Philips DP v1.0", description="level=1 mag=20 quality=80")
        tw.write(rgb((20, 40)), metadata=None, photometric="rgb", software="Philips DP v1.0",
                 description="Macro -offset=(0,0)-pixelsize=(0.0315,0.0315)")
    return path


def _dv(tmp_path):
    path = str(tmp_path / "a.dv")
    test_dv.write_dv(path, test_dv._data())
    return path


def _mrc(tmp_path):
    path = str(tmp_path / "a.mrc")
    test_mrc.write_mrc(path, test_mrc._data("u2"))
    return path


def _tiff(tmp_path):
    path = str(tmp_path / "plain.tif")
    write_tiff(path, _u2((20, 24)), 16)
    return path


def _bigtiff(tmp_path):
    path = str(tmp_path / "big.tif")
    tifffile.imwrite(path, _u2((3, 20, 24)), bigtiff=True, metadata=None, photometric="minisblack")
    return path


def _imagej(tmp_path):
    path = str(tmp_path / "ij.tif")
    tifffile.imwrite(path, _u2((3, 20, 24)), imagej=True, resolution=(2.0, 2.0),
                     metadata={"axes": "ZYX", "unit": "um", "spacing": 0.5})
    return path


def _ndtiff(tmp_path):
    return test_tiff_variants._ndtiff(str(tmp_path / "acq"))[0]


def _luxendo(tmp_path):
    path = tmp_path / "a.lux.h5"
    _h5py().write_luxendo(path, _u2((4, 16, 16)))
    return path


def _fiji(tmp_path):
    path = tmp_path / "stack.h5"
    _h5py().write_fiji(path, {f"t0/channel{c}": _u2((3, 16, 20)) for c in range(2)})
    return path


def _ilastik(tmp_path):
    path = tmp_path / "export.h5"
    _h5py().write_ilastik(str(path), _u2((3, 16, 20, 2)), "zyxc")
    return path


def _metaxpress(tmp_path):
    htd, _ = test_metaxpress._write(str(tmp_path), wells=[(0, 1), (1, 0)], sites=[[True, True]])
    return htd


def _ics(tmp_path):
    return write_ics(tmp_path, _u2((2, 3, 16, 20)), "czyx")


def _micromanager(tmp_path):
    _, files = write_acquisition(str(tmp_path / "acq"), positions=2, prefix="",
                                 tiff_pixel_size=0.5, stage=lambda p: (10.0 + p, -20.0, 3.5))
    return files[0]


def _png(tmp_path):
    path = str(tmp_path / "v.png")
    write_png(path, RNG.integers(0, 255, (12, 16)).astype(np.uint8))
    return path


def _bmp(tmp_path):
    path = str(tmp_path / "p.bmp")
    write_bmp(path, RNG.integers(0, 255, (12, 16, 3)).astype(np.uint8), bpp=24)
    return path


#: case -> (writer, the (backend, format) it opens as); the plan's list (4.12) first
CASES = {
    "metamorph": (_metamorph, ("metamorph", "metamorph")),
    "incell": (_incell, ("incell", "incell")),
    "oif": (_oif, ("oif", "oif")),
    "oib": (_oib, ("oif", "oib")),
    "nd2-legacy": (_nd2_legacy, ("nd2", "nd2-legacy")),
    "oir": (_oir, ("oir", "oir")),
    "vsi": (_vsi, ("vsi", "vsi")),
    "bif": (_bif, ("tiff", "bif")),
    "bif-iscan": (_bif_iscan, ("tiff", "bif")),
    "philips": (_philips, ("tiff", "philips")),
    "dv": (_dv, ("dv", "dv")),
    "mrc": (_mrc, ("mrc", "mrc")),
    "tiff": (_tiff, ("tiff", "tiff")),
    "bigtiff": (_bigtiff, ("tiff", "bigtiff")),
    "luxendo": (_luxendo, ("hdf5", "luxendo")),
    "fiji-hdf5": (_fiji, ("hdf5", "fiji-hdf5")),
    "ilastik": (_ilastik, ("ilastik", "ilastik")),
    "metaxpress": (_metaxpress, ("metaxpress", "metaxpress")),
    # not in the plan's list, no position either
    "stk": (_stk, ("tiff", "stk")),
    "imagej": (_imagej, ("tiff", "imagej")),
    "ndtiff": (_ndtiff, ("tiff", "ndtiff")),
    "ics": (_ics, ("ics", "ics")),
    "micromanager-files": (_micromanager, ("micromanager", "micromanager-files")),
    "png": (_png, ("raster", "png")),
    "bmp": (_bmp, ("raster", "bmp")),
}

#: the formats of UNPLACED_FORMATS no synthetic file here stands for (no writer):
#: the session check still covers the suite's files of them
WITHOUT_A_WRITER = {("vsi", "ets"), ("raster", "jpeg"), ("tiff", "lsm"), ("tiff", "eer"),
                    ("tiff", "ndpi"), ("tiff", "micromanager"), ("tiff", "scanimage")}


def _metadata_reads_while_declaring(f, monkeypatch) -> int:
    """How often the declaration (``coordinate_systems``,
    ``coordinate_transformations``) touches any image's metadata, cached or
    not: declared afresh under a counting ``Image.metadata``."""
    count = [0]
    real = Image.metadata

    def counting(self):
        count[0] += 1
        return real.fget(self)

    monkeypatch.setattr(Image, "metadata", property(counting))
    f._graph = None
    f.coordinate_systems
    f.coordinate_transformations
    monkeypatch.setattr(Image, "metadata", real)
    return count[0]


@pytest.mark.parametrize("case", list(CASES))
def test_an_unplaced_format_declares_nothing_and_places_nothing(tmp_path, monkeypatch, case):
    write, pair = CASES[case]
    path = write(tmp_path)
    with micro_reader.open(path, resolutions="all") as f:
        assert (f.backend, f.format) == pair
        assert pair in conftest.UNPLACED_FORMATS
        assert _metadata_reads_while_declaring(f, monkeypatch) == 0
        assert f.coordinate_systems == () and f.coordinate_transformations == ()
        images = f._every_image()
        assert images
        for image in images:
            md = image.metadata
            assert md.position == {}, image.name
            assert md.coordinate_system is None, image.name
        check_invariants(f)                             # I1-I3, I5 hold trivially


@pytest.mark.parametrize("case", ["metamorph", "micromanager-files"])
def test_a_stage_position_is_kept_as_recorded_but_places_nothing(tmp_path, case):
    """MetaMorph and Micro-Manager image-per-file record stage coordinates
    whose point in the field is unverified: ``stage_position`` keeps them as
    recorded (3.27), ``position`` and ``coordinate_system`` stay empty."""
    write, _ = CASES[case]
    with micro_reader.open(write(tmp_path)) as f:
        for image in f.images:
            md = image.metadata
            assert md.stage_position and md.position == {} and md.coordinate_system is None


def test_every_unplaced_format_has_a_case_or_is_named_without_one():
    """``UNPLACED_FORMATS`` is backed here: each pair is opened by a case
    above, or listed in ``WITHOUT_A_WRITER``."""
    covered = {pair for _, pair in CASES.values()}
    assert covered <= conftest.UNPLACED_FORMATS
    assert conftest.UNPLACED_FORMATS == covered | WITHOUT_A_WRITER
    assert not covered & WITHOUT_A_WRITER
