"""DeltaVision backend, on synthetic files written to the Priism / IVE layout
(the real OME DeltaVision samples were compared with Bio-Formats plane by
plane: docs/PLAN.md, "DeltaVision DV")."""
from __future__ import annotations

import struct

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, UnsupportedFormatError

MODE_OF = {"u1": 0, "i2": 1, "f4": 2, "u2": 6, "i4": 7}
#: imgSequence -> section order, slowest first
ORDER = {0: "ctz", 1: "tzc", 2: "tcz"}
N_INTS, N_FLOATS = 8, 32


def write_dv(path, data, *, sequence=0, order="<", mode=None, d=(0.1, 0.1, 0.3),
             waves=(457, 528, 617, 685, 0), stamps=None, excitation=None):
    """*data*: (t, c, z, y, x) in display orientation (row 0 at the top); it is
    stored bottom-up, sections in the *sequence* interleave.  *stamps*:
    (t) seconds; *excitation*: (c) nm -- written to every section's extended
    header."""
    data = np.asarray(data)
    nt, nc, nz, ny, nx = data.shape
    mode = MODE_OF[data.dtype.str[1:]] if mode is None else mode
    sections = nt * nc * nz
    head = bytearray(1024)
    struct.pack_into(order + "4i", head, 0, nx, ny, sections, mode)
    struct.pack_into(order + "3i", head, 28, 1, 1, 1)
    struct.pack_into(order + "3f", head, 40, *d)
    struct.pack_into(order + "i", head, 92, sections * (N_INTS + N_FLOATS) * 4)
    struct.pack_into(order + "h", head, 96, -16224)
    struct.pack_into(order + "2h", head, 128, N_INTS, N_FLOATS)
    struct.pack_into(order + "2h", head, 180, nt, sequence)
    struct.pack_into(order + "h", head, 196, nc)
    struct.pack_into(order + "5h", head, 198, *waves)
    axes = ORDER[sequence]
    stored = data.transpose([{"t": 0, "c": 1, "z": 2}[a] for a in axes] + [3, 4])
    ext = bytearray()
    for i, j, k in np.ndindex(stored.shape[:3]):
        index = dict(zip(axes, (i, j, k)))
        floats = [0.0] * N_FLOATS
        if stamps is not None:
            floats[1] = stamps[index["t"]]
        if excitation is not None:
            floats[10] = excitation[index["c"]]
        ext += struct.pack(order + f"{N_INTS}i{N_FLOATS}f", *([0] * N_INTS), *floats)
    with open(path, "wb") as fh:
        fh.write(head)
        fh.write(ext)
        fh.write(stored[..., ::-1, :].astype(data.dtype.newbyteorder(order)).tobytes())
    return str(path)


def _data(shape=(3, 2, 4, 21, 33), dtype="u2", seed=0):
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype) if np.dtype(dtype).kind in "iu" else None
    if info is None:
        return rng.normal(size=shape).astype(dtype)
    return rng.integers(info.min, info.max, size=shape, endpoint=True).astype(dtype)


@pytest.mark.parametrize("sequence", [0, 1, 2])
@pytest.mark.parametrize("order", ["<", ">"])
def test_sections_in_every_interleave_and_byte_order(tmp_path, sequence, order):
    data = _data()
    path = write_dv(tmp_path / "a.dv", data, sequence=sequence, order=order)
    with micro_reader.open(path) as f:
        assert f.backend == "dv" and f.format == "dv"
        im = f.images[0]
        assert im.axes == ORDER[sequence] + "yx"                 # the file's own order
        a = im.as_axes("tczyx")
        np.testing.assert_array_equal(a[...], data)
        np.testing.assert_array_equal(a[1:3, :, 2:4, 5:17, 3:30], data[1:3, :, 2:4, 5:17, 3:30])
        np.testing.assert_array_equal(a[2, 1, 0, 20:21, :], data[2, 1, 0, 20:21, :])


@pytest.mark.parametrize("dtype", ["u1", "i2", "f4", "i4"])
def test_pixel_types(tmp_path, dtype):
    data = _data((1, 1, 2, 9, 11), dtype)
    path = write_dv(tmp_path / "a.r3d", data)
    with micro_reader.open(path) as f:
        im = f.images[0]
        assert im.dtype == np.dtype(dtype)                       # mode 0 is unsigned in DV
        np.testing.assert_array_equal(im.as_axes("tczyx")[...], data)


def test_recognised_by_content_not_extension(tmp_path):
    path = write_dv(tmp_path / "scan.r3d_d3d", _data((1, 1, 1, 5, 6)))
    with micro_reader.open(path) as f:
        assert f.backend == "dv"


def test_metadata(tmp_path):
    data = _data((3, 2, 2, 5, 6))
    path = write_dv(tmp_path / "a.dv", data, d=(0.065, 0.065, 0.2), waves=(457, 617, 0, 0, 0),
                    stamps=[0.0, 12.5, 25.0], excitation=[360.0, 555.0])
    with micro_reader.open(path) as f:
        meta = f.images[0].metadata
    assert meta.pixel_sizes["x"].value == pytest.approx(0.065)
    assert meta.pixel_sizes["z"].value == pytest.approx(0.2)
    assert meta.pixel_sizes["t"].value == pytest.approx(12.5)
    assert meta.pixel_sizes["t"].unit == "second"
    assert [(c.emission_wavelength.value, c.excitation_wavelength.value) for c in meta.channels] \
        == [(457.0, 360.0), (617.0, 555.0)]
    assert all(c.name is None for c in meta.channels)            # no softWoRx log here
    assert not meta.position


def test_junk_time_stamps_give_no_time_step(tmp_path):
    path = write_dv(tmp_path / "a.dv", _data((2, 1, 1, 5, 6)), stamps=[-1.0e9, 7.75])
    with micro_reader.open(path) as f:
        assert "t" not in f.images[0].metadata.pixel_sizes


def test_truncated_file_is_corrupt(tmp_path):
    path = write_dv(tmp_path / "a.dv", _data((1, 1, 3, 9, 11)))
    with open(path, "r+b") as fh:
        fh.truncate(fh.seek(0, 2) - 10)
    with pytest.raises(CorruptFileError, match="truncated"):
        micro_reader.open(path)


def test_sections_not_waves_times_z_is_corrupt(tmp_path):
    path = write_dv(tmp_path / "a.dv", _data((1, 1, 3, 9, 11)))
    with open(path, "r+b") as fh:                                # 2 waves of 3 sections
        fh.seek(196)
        fh.write(struct.pack("<h", 2))
    with pytest.raises(CorruptFileError, match="sections"):
        micro_reader.open(path)


def test_complex_int16_refused(tmp_path):
    path = write_dv(tmp_path / "a.dv", _data((1, 1, 1, 4, 4), "u1"), mode=3)
    with pytest.raises(UnsupportedFormatError, match="mode 3"):
        micro_reader.open(path)


def test_mrc_files_stay_mrc(tmp_path):
    from test_mrc import write_mrc
    path = write_mrc(tmp_path / "a.mrc", _data((1, 1, 3, 9, 11)).reshape(3, 9, 11))
    with micro_reader.open(path) as f:
        assert f.backend == "mrc"


# -- channel names from the softWoRx logs (docs/COMPOSITE.md, "DeltaVision") ----------

def acquisition_log(image_name, names, sections=None):
    """An acquisition log for *image_name*: ``#KEY CHANNEL`` lines for
    *names* (wave order), then one ``Image N.`` block per section, its EM
    filter from *sections* (section -> name) when given."""
    lines = ["#", "#Tag 0x00FF0408", "", f"Resolve3D Image File: /data/{image_name}", ""]
    lines += [f"#KEY CHANNEL 0.1000,{n},{n},100%" for n in names]
    lines += ["", "Experiment Record:", "------------------"]
    for k, name in enumerate(sections or []):
        lines += [f"Image {k + 1}.", "           Time Point: 0.000 secs",
                  f"           EX filter:  {name}", f"           EM filter:  {name}", ""]
    return "\r\n".join(lines) + "\r\n"


def _names(path):
    with micro_reader.open(str(path)) as f:
        return [c.name for c in f.images[0].metadata.channels]


@pytest.mark.parametrize("sequence", [0, 1, 2])
def test_names_from_the_files_own_log_follow_its_sections(tmp_path, sequence):
    """Two waves share a filter name (the #KEY lines would be ambiguous):
    each name is the EM filter of the channel's first section."""
    data = _data((2, 3, 2, 5, 6))
    path = write_dv(tmp_path / "x_R3D.dv", data, sequence=sequence)
    order = ORDER[sequence]
    sizes = {"t": 2, "c": 3, "z": 2}
    channel_names = ["DAPI", "GFP", "GFP-long"]
    sections = []
    for i, j, k in np.ndindex(*(sizes[a] for a in order)):
        sections.append(channel_names[dict(zip(order, (i, j, k)))["c"]])
    (tmp_path / "x_R3D.dv.log").write_text(acquisition_log("x_R3D.dv", ["wrong"] * 3, sections))
    assert _names(path) == channel_names


def test_a_processed_file_takes_the_raw_files_log(tmp_path):
    path = write_dv(tmp_path / "x_R3D_D3D_PRJ.dv", _data((1, 2, 1, 5, 6)))
    (tmp_path / "x_R3D.dv.log").write_text(acquisition_log("x_R3D.dv", ["DAPI", "FITC"]))
    assert _names(path) == ["DAPI", "FITC"]


@pytest.mark.parametrize("processing", ["frap", "deconvolution"])
def test_a_processing_log_leads_to_the_acquisition_log(tmp_path, processing):
    path = write_dv(tmp_path / "result.dv", _data((1, 2, 1, 5, 6)))
    if processing == "frap":
        (tmp_path / "result.dv.log").write_text(
            "FRAP Results:   Wed Apr  2 09:08:15 2003\n\n  Input Image:   \t/d/raw.r3d\n")
    else:
        (tmp_path / "result_log.txt").write_text(
            " DECON3D: 3D deconvolution.\n\nImage Name..........\t///d/raw.r3d\n")
    (tmp_path / "raw.r3d.log").write_text(acquisition_log("raw.r3d", ["DAPI", "CY-5"]))
    assert _names(path) == ["DAPI", "CY-5"]


def test_a_log_of_another_image_is_not_used(tmp_path):
    path = write_dv(tmp_path / "x_R3D.dv", _data((1, 2, 1, 5, 6)))
    (tmp_path / "x_R3D.dv.log").write_text(acquisition_log("y_R3D.dv", ["DAPI", "FITC"]))
    assert _names(path) == [None, None]


def test_a_log_with_another_channel_count_warns(tmp_path):
    path = write_dv(tmp_path / "x_R3D.dv", _data((1, 2, 1, 5, 6)))
    (tmp_path / "x_R3D.dv.log").write_text(acquisition_log("x_R3D.dv", ["DAPI", "FITC", "CY-5"]))
    with pytest.warns(micro_reader.CompositeWarning, match="lists 3 channel"):
        assert _names(path) == [None, None]
