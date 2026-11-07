"""Composite formats (docs/COMPOSITE.md): opening any part opens the whole
image; a folder opens the image it is a part of, or the one it holds."""
from __future__ import annotations

import os

import numpy as np
import pytest

import micro_reader
from micro_reader import (CompositeWarning, CorruptFileError, MicroReaderError,
                          UnsupportedFormatError, _parts)

import test_companion
import test_oir
from _ndtiff_writer import write_ndtiff
from test_dv import _data as dv_data, write_dv

ACQUISITION_LOG = "#\n#Tag 0x00FF0408\n\nResolve3D Image File: /data/{name}\n"


# -- parts open the whole -----------------------------------------------------------

def test_oir_continuation_opens_the_main_file(tmp_path):
    path, expected = test_oir._write(tmp_path, "companion_file")
    part = os.path.splitext(path)[0] + "_00001"
    with micro_reader.open(part) as f:
        assert os.path.samefile(f.path, path) and f.opened == part
        np.testing.assert_array_equal(f.images[0][...], expected)


def test_oir_continuation_without_its_main_file_raises(tmp_path):
    path, _ = test_oir._write(tmp_path, "companion_file")
    part = os.path.splitext(path)[0] + "_00001"
    os.remove(path)
    with pytest.raises(CorruptFileError, match="which is missing"):
        micro_reader.open(part)


def test_an_explicit_backend_reads_the_part_itself(tmp_path):
    path, _ = test_oir._write(tmp_path, "companion_file")
    with pytest.raises(UnsupportedFormatError, match="without image data"):
        micro_reader.open(os.path.splitext(path)[0] + "_00001", backend="oir")


@pytest.mark.parametrize("log_name", ["a_R3D.dv.log", "a_R3D_log.txt"])
def test_deltavision_logs_open_their_image(tmp_path, log_name):
    data = dv_data((1, 2, 3, 9, 11))
    image = write_dv(tmp_path / "a_R3D.dv", data)
    log = tmp_path / log_name
    log.write_text(ACQUISITION_LOG.format(name="a_R3D.dv"))
    with micro_reader.open(str(log)) as f:
        assert f.backend == "dv" and os.path.samefile(f.path, image)
        np.testing.assert_array_equal(f.images[0].as_axes("tczyx")[...], data)


def test_a_deltavision_log_without_its_image_raises(tmp_path):
    log = tmp_path / "gone_R3D.dv.log"
    log.write_text(ACQUISITION_LOG.format(name="gone_R3D.dv"))
    with pytest.raises(CorruptFileError, match="gone_R3D.dv"):
        micro_reader.open(str(log))


def test_other_logs_are_not_parts(tmp_path):
    log = tmp_path / "server.log"
    log.write_text("started\n")
    with pytest.raises(UnsupportedFormatError, match="no micro-reader backend"):
        micro_reader.open(str(log))


def _frames():
    y, x = np.mgrid[:6, :8]
    return [({"position": 0, "time": t, "z": 0}, (t * 100 + y * 8 + x).astype(np.uint16))
            for t in range(3)]


def test_ndtiff_index_and_folder_open_the_dataset(tmp_path):
    paths = write_ndtiff(str(tmp_path / "acq"), _frames(), per_file=2)
    for target in (str(tmp_path / "acq" / "NDTiff.index"), str(tmp_path / "acq")):
        with micro_reader.open(target) as f:
            assert f.format == "ndtiff" and os.path.samefile(f.path, paths[0])
            assert f.opened == target
            assert f.images[0].as_axes("tczyx").shape[0] == 3


# -- folders ----------------------------------------------------------------------------

def test_a_folder_with_one_image_and_its_parts_opens_it(tmp_path):
    folder = tmp_path / "scan"
    folder.mkdir()
    image = write_dv(folder / "a_R3D.dv", dv_data((1, 1, 2, 5, 6)))
    (folder / "a_R3D.dv.log").write_text(ACQUISITION_LOG.format(name="a_R3D.dv"))
    (folder / "notes.txt").write_text("not an image")
    with micro_reader.open(str(folder)) as f:
        assert os.path.samefile(f.path, image) and f.opened == str(folder)


def test_a_folder_with_several_images_lists_them(tmp_path):
    for name in ("a.dv", "b.dv"):
        write_dv(tmp_path / name, dv_data((1, 1, 1, 4, 4)))
    with pytest.raises(MicroReaderError, match=r"2 images: a\.dv, b\.dv"):
        micro_reader.open(str(tmp_path))


def test_a_folder_without_images_raises(tmp_path):
    (tmp_path / "notes.txt").write_text("not an image")
    with pytest.raises(UnsupportedFormatError, match="no file in this folder"):
        micro_reader.open(str(tmp_path))


def test_a_sample_folder_opens_the_one_dataset_below_it(tmp_path):
    # MR-6: a folder with no image of its own opens the one dataset in its
    # sub-folders (38 of 54 plate sample folders failed: their export sits below)
    deep = tmp_path / "sample" / "export" / "scan"
    deep.mkdir(parents=True)
    image = write_dv(deep / "a_R3D.dv", dv_data((1, 1, 2, 5, 6)))
    (tmp_path / "sample" / "README.txt").write_text("from the depositors")
    with micro_reader.open(str(tmp_path / "sample")) as f:
        assert os.path.samefile(f.path, image)


def test_a_sample_folder_takes_a_companion_below_it_as_one_set(tmp_path):
    # each sub-folder is taken as open would take it: its index file wins over
    # the member TIFFs next to it, so the set is one dataset
    companion, _ = test_companion._write(str(tmp_path / "sample" / "set"))
    with micro_reader.open(str(tmp_path / "sample")) as f:
        assert f.backend == "companion" and os.path.samefile(f.path, companion)


def test_a_folder_of_several_datasets_below_lists_them(tmp_path):
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        write_dv(tmp_path / name / f"{name}.dv", dv_data((1, 1, 1, 4, 4)))
    with pytest.raises(MicroReaderError, match=r"2 datasets in its sub-folders: "
                                               r"one[\\/]one\.dv, two[\\/]two\.dv"):
        micro_reader.open(str(tmp_path))


def test_an_ome_zarr_store_is_refused_naming_ome_zarr_pyramid(tmp_path):
    # gap hunt B-06: the generic "no file in this folder is an image" before
    store = tmp_path / "cells.ome.zarr"
    store.mkdir()
    (store / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    for path in (store, store / "zarr.json"):
        with pytest.raises(UnsupportedFormatError, match="OME-Zarr store.*ome_zarr_pyramid"):
            micro_reader.open(str(path))
    with pytest.raises(UnsupportedFormatError, match="holds OME-Zarr stores"):
        micro_reader.open(str(tmp_path))


@pytest.mark.parametrize("name, what", [("scan.ptu", "PicoQuant time-tagged"),
                                        ("scan.phu", "PicoQuant histogram")])
def test_formats_known_but_not_read_are_named(tmp_path, name, what):
    # gap hunt B-07: they got the generic "no micro-reader backend recognises"
    (tmp_path / name).write_bytes(b"\x00" * 64)
    with pytest.raises(UnsupportedFormatError, match=f"{what}.*does not read yet"):
        micro_reader.open(str(tmp_path / name))


def test_a_folder_with_an_index_file_opens_the_set(tmp_path):
    """A companion's member TIFFs are images too; the index file wins."""
    companion, _ = test_companion._write(str(tmp_path / "set"))
    with micro_reader.open(str(tmp_path / "set")) as f:
        assert f.backend == "companion" and os.path.samefile(f.path, companion)


# -- the helpers -----------------------------------------------------------------------

class _FakeComposite:
    """A backend whose parts are images on their own (as OIF planes)."""
    NAME = "fake"

    @staticmethod
    def owner(path, head):
        if path.endswith(".part.dv"):
            return _parts.Part(None, path.replace(".part.dv", ".main"), standalone=True)
        return None

    @staticmethod
    def sniff(head, path):
        return False


def test_a_standalone_part_without_its_main_file_is_read_with_a_warning(tmp_path):
    part = write_dv(tmp_path / "x.part.dv", dv_data((1, 1, 1, 4, 4)))
    with pytest.warns(CompositeWarning, match="x.main"):
        assert _parts.main_file(part, [_FakeComposite]) == part


def test_find_matches_letter_case(tmp_path):
    (tmp_path / "Image.OIR").write_bytes(b"")
    assert os.path.samefile(_parts.find(str(tmp_path), "image.oir"), tmp_path / "Image.OIR")
    assert _parts.find(str(tmp_path), "other.oir") is None


def test_find_refuses_two_names_that_differ_only_in_case(tmp_path):
    (tmp_path / "a.oir").write_bytes(b"")
    if os.path.exists(tmp_path / "A.OIR"):
        pytest.skip("case-insensitive file system")
    (tmp_path / "A.OIR").write_bytes(b"")
    with pytest.warns(CompositeWarning, match="letter case"):
        assert _parts.find(str(tmp_path), "a.Oir") is None


def test_micro_manager_stack_files_are_parts_of_the_first(tmp_path):
    """Files named <prefix>_MMStack*.tif are one acquisition (tifffile's
    rule); the first in natural order is the main file, so a folder of them
    is one image.  Plain TIFFs here: the rule goes by the names."""
    tifffile = pytest.importorskip("tifffile")
    names = ["run_MMStack_Pos0.ome.tif", "run_MMStack_Pos0_1.ome.tif", "run_MMStack_Pos1.ome.tif",
             "run_MMStack_Pos10.ome.tif", "run_MMStack_Pos2.ome.tif"]
    for name in names:
        tifffile.imwrite(tmp_path / name, np.zeros((4, 5), np.uint8))
    from micro_reader.backends import tiff
    for name in names[1:]:
        path = str(tmp_path / name)
        part = _parts.part_of(path, _parts.read_head(path), [tiff])
        assert part is not None and os.path.basename(part.main) == names[0]
    first = str(tmp_path / names[0])
    assert _parts.part_of(first, _parts.read_head(first), [tiff]) is None
    side = tmp_path / "run_MMStack_Pos0_metadata.txt"           # the side file opens the stack
    side.write_text("{}")
    part = _parts.part_of(str(side), _parts.read_head(str(side)), [tiff])
    assert os.path.basename(part.main) == names[0]
    with micro_reader.open(str(tmp_path)) as f:
        assert os.path.basename(f.path) == names[0]


# -- side files, messages (gap hunt 2026-10-29, closed for 0.4.0) ------------------------

def test_leica_files_of_settings_are_parts_of_their_experiment(tmp_path):
    from micro_reader.backends import lif
    (tmp_path / "scan" / "Metadata").mkdir(parents=True)
    xlef = tmp_path / "scan.xlef"
    xlef.write_text("<x/>")
    for path in (tmp_path / "scan" / "FrameProperties.lof",
                 tmp_path / "scan" / "Metadata" / "FrameProperties.xlif",
                 tmp_path / "scan" / "Metadata" / "IOManagerConfiguation.xlif"):
        path.write_text("<x/>")
        assert os.path.normcase(lif.owner(str(path), b"<x/>").main) == os.path.normcase(str(xlef))
    assert not lif.may_index("FrameProperties.xlif") and lif.may_index("scan.xlif")


def test_a_lifext_and_an_xml_are_parts_of_their_lif(tmp_path):
    from micro_reader.backends import lif
    main = tmp_path / "scan.lif"
    main.write_bytes(b"x")
    ext, xml = tmp_path / "scan.lifext", tmp_path / "scan.xml"
    ext.write_bytes(b"x")
    xml.write_text("<LMSDataContainerHeader/>")
    assert lif.owner(str(ext), b"").member                      # alone with multifile=False
    part = lif.owner(str(xml), b"<LMSDataContainerHeader/>")
    assert part.main == str(main) and not part.member
    assert lif.owner(str(xml), b"<other/>") is None             # an XML of something else


def test_a_folder_says_what_it_skips_beside_its_images(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    tifffile.imwrite(tmp_path / "img.tif", np.zeros((6, 8), np.uint8))
    (tmp_path / "img.ptu").write_bytes(b"x")
    with pytest.warns(micro_reader.CompositeWarning, match="img.ptu"):
        micro_reader.open(tmp_path).close()


def test_a_path_has_one_spelling(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = tmp_path / "a.tif"
    tifffile.imwrite(path, np.zeros((6, 8), np.uint8))
    with micro_reader.open(str(path).replace(os.sep, "/")) as f:
        assert f.path == str(path)


def test_an_orphan_log_names_the_deconvolved_image_beside_it(tmp_path):
    from micro_reader.backends import dv
    log = tmp_path / "cell_R3D.dv.log"
    log.write_bytes(dv.LOG_SIGNATURES[0])
    (tmp_path / "cell_R3D_D3D.dv").write_bytes(b"x")
    part = dv.owner(str(log), dv.LOG_SIGNATURES[0])
    assert part.main is None and "cell_R3D_D3D.dv" in part.hint


def test_a_leica_element_opens_its_experiment(tmp_path):
    from micro_reader.backends import lif
    (tmp_path / "Metadata").mkdir()
    xlef = tmp_path / "exp.xlef"
    xlef.write_text("<x/>")
    lof, xlif = tmp_path / "scan.lof", tmp_path / "Metadata" / "scan.xlif"
    lof.write_text("<x/>")
    xlif.write_text("<x/>")
    for path in (lof, xlif):
        part = lif.owner(str(path), b"<x/>")
        assert part.member and os.path.normcase(part.main) == os.path.normcase(str(xlef))
    lone = tmp_path / "other.lof"                       # no .xlif of it: no element of the experiment
    lone.write_text("<x/>")
    assert lif.owner(str(lone), b"<x/>") is None
