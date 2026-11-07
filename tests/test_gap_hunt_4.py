"""What the fourth gap hunt (2026-11-01) found in micro-reader."""
from types import SimpleNamespace

import numpy as np
import pytest

from micro_reader._metadata import Quantity
from micro_reader._registry import _check_rename
from micro_reader.backends import _tiff_metadata, bdv, ims


def _setup(unit, voxel):
    return SimpleNamespace(unit=unit, voxel=voxel)


def test_a_voxel_size_in_pixels_is_kept_as_relative():
    sizes = bdv._pixel_sizes(_setup("pixel", (1.0, 1.0, 5.0)))
    assert sizes == {"x": Quantity(1.0, "pixel"), "y": Quantity(1.0, "pixel"),
                     "z": Quantity(5.0, "pixel")}


def test_a_voxel_size_of_ones_in_pixels_says_nothing():
    assert bdv._pixel_sizes(_setup("pixel", (1.0, 1.0, 1.0))) == {}


def test_a_known_unit_is_as_before():
    assert bdv._pixel_sizes(_setup("micron", (0.5, 0.5, 2.0)))["z"] == Quantity(2.0, "micrometer")


def _described(text):
    return SimpleNamespace(pages=[SimpleNamespace(description=text)])


def test_the_z_step_of_an_imaris_tiff_export():
    text = "[ImarisDataSet]\r\nNumberOfImages=1\r\n\r\n[Image]\r\nDx = 0.14\r\nDz = 0.2\r\nUnit = um\r\n"
    assert _tiff_metadata._imaris_z_step(_described(text)) == Quantity(0.2, "micrometer")


@pytest.mark.parametrize("text", ["", "ImageJ=1.53", "[ImarisDataSet]\nDx = 1\n"])
def test_no_z_step_where_none_is_written(text):
    assert _tiff_metadata._imaris_z_step(_described(text)) is None


def test_a_rename_that_is_no_dict_says_so():
    with pytest.raises(TypeError, match="sequence"):
        _check_rename("zyxc")


def test_an_h5_copy_of_an_ims_belongs_to_it(tmp_path):
    (tmp_path / "set.ims").write_bytes(b"\x89HDF\r\n\x1a\n")
    (tmp_path / "set.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
    (tmp_path / "other.h5").write_bytes(b"\x89HDF\r\n\x1a\n")
    part = ims.owner(str(tmp_path / "set.h5"), b"\x89HDF\r\n\x1a\n")
    assert part is not None and part.main.endswith("set.ims")
    assert ims.owner(str(tmp_path / "other.h5"), b"\x89HDF\r\n\x1a\n") is None
    assert ims.owner(str(tmp_path / "set.ims"), b"\x89HDF\r\n\x1a\n") is None


def test_a_stack_of_two_pixel_types_is_not_ragged():
    from micro_reader._base import Image
    from micro_reader._stack import ChannelStack
    parts =[Image(k, "cyx", (1, 4, 4), np.dtype(t), name="p", index={})
             for k, t in enumerate(("u1", "u2"))]
    whole = ChannelStack(0, parts, [(parts[0], 0), (parts[1], 0)], "w", {})
    assert len(whole.parts) == 2 and whole.is_ragged is False
