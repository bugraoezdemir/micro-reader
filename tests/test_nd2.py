"""ND2 backend on synthetic files (tests/_nd2_writer.py), checked against the
known data and, where installed, against the nd2 library reading the same
file (which also validates the writer)."""
from __future__ import annotations

import os
import struct
import threading

import numpy as np
import pytest

import micro_reader
from micro_reader import MicroReaderError, UnsupportedFormatError
from micro_reader.backends import nd2 as nd2_backend

from _nd2_writer import write_nd2


def _native(*args, **kwargs):
    """``micro_reader.open`` with RGB samples kept as an ``s`` axis: these tests
    check the backend's own layout (open's default folds them into channels)."""
    return micro_reader.open(*args, **{"samples": "keep", **kwargs})

try:
    import nd2 as nd2lib
except ImportError:                   # the cross-check is optional
    nd2lib = None


def _frames(n, h=37, w=53, channels=2, samples=1, dtype="u2", seed=0):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return rng.normal(size=(n, h, w, channels, samples)).astype(dtype)
    return rng.integers(0, np.iinfo(dtype).max, size=(n, h, w, channels, samples),
                        endpoint=True).astype(dtype)


def _expected(frames, loops, missing=()):
    """Per series (XY position): t c z y x [s] from frames in loop order."""
    frames = frames.copy()
    for i in missing:
        frames[i] = 0
    counts = [c for _, c in loops]
    kinds = ["T" if k == "NE" else k for k, _ in loops]
    arr = frames.reshape(*counts, *frames.shape[1:])       # loops..., y, x, c, s
    for k in "TPZ":
        if k not in kinds:
            arr = np.expand_dims(arr, 0)
            kinds.insert(0, k)
    n = len(kinds)
    order = [kinds.index("P"), kinds.index("T"), n + 2, kinds.index("Z"), n, n + 1, n + 3]
    arr = arr.transpose(order)                             # p, t, c, z, y, x, s
    return [a if a.shape[-1] > 1 else a[..., 0] for a in arr]


CASES = {
    "z_rgb_padded":   dict(frames=_frames(5, channels=1, samples=3, dtype="u1"),
                           loops=[("Z", 5)], pad_bytes=3),
    "t_p_z":          dict(frames=_frames(24), loops=[("T", 3), ("P", 2), ("Z", 4)]),
    "p_outer":        dict(frames=_frames(6), loops=[("P", 3), ("T", 2)]),
    "lossless":       dict(frames=_frames(6), loops=[("T", 2), ("Z", 3)],
                           compression="lossless"),
    "lossless_padded_rgb": dict(frames=_frames(4, channels=2, samples=3, dtype="u1"),
                                loops=[("T", 4)], compression="lossless", pad_bytes=5),
    "padded_multichannel_rgb": dict(frames=_frames(3, channels=2, samples=3, dtype="u1"),
                                    loops=[("Z", 3)], pad_bytes=7),
    "float32":        dict(frames=_frames(3, dtype="f4"), loops=[("Z", 3)]),
    "ne_time":        dict(frames=_frames(5), loops=[("NE", 5)], ne_periods=[2, 3]),
    "invalid_position": dict(frames=_frames(4), loops=[("T", 2), ("P", 2)],
                             valid_positions=[1, 0, 1]),
    "missing_frame":  dict(frames=_frames(4), loops=[("Z", 4)], missing=(2,)),
    "single_frame":   dict(frames=_frames(1, channels=3), loops=[]),
}


def _write(tmp_path, name):
    case = dict(CASES[name])
    path = str(tmp_path / f"{name}.nd2")
    frames, loops = case.pop("frames"), case.pop("loops")
    write_nd2(path, frames, loops, **case)
    return path, _expected(frames, loops, case.get("missing", ()))


def _keys(shape, n=20, seed=3):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,), (0,), (-1, 0)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            b = int(rng.integers(a + 1, size + 1))
            choice = rng.integers(0, 4)
            key.append(a if choice == 0 else slice(a, b, 2) if choice == 1 else slice(a, b))
        keys.append(tuple(key))
    return keys


@pytest.fixture(params=[1, 4], ids=["serial", "pool"])
def decode_threads(request):
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(request.param)
    yield
    micro_reader.set_decode_threads(old)


@pytest.mark.parametrize("name", list(CASES))
def test_matches_known_data(tmp_path, name, decode_threads):
    path, expected = _write(tmp_path, name)
    with _native(path) as img:
        assert img.backend == "nd2"
        assert len(img.images) == len(expected)
        for series, ref in zip(img.images, expected):
            assert series.shape == ref.shape and series.dtype == ref.dtype
            assert series.axes == ("tczyxs" if ref.ndim == 6 else "tczyx")
            for key in _keys(ref.shape):
                np.testing.assert_array_equal(series[key], ref[key], err_msg=f"{name} {key}")


#: nd2 0.11 reads padded rows with more than one channel of RGB components
#: wrongly: it steps between channels by one component instead of one pixel's
#: components (strides (widthBytes, n_comp*bpc, bpc, bpc)).  Our reader is
#: checked against the known data above; strict, so a fix upstream shows up.
_ND2_PADDED_MULTICHANNEL_RGB_BUG = pytest.mark.xfail(
    strict=True, reason="nd2 library: wrong channel stride for padded multi-channel RGB")


@pytest.mark.skipif(nd2lib is None, reason="nd2 library not installed")
@pytest.mark.parametrize("name", [
    pytest.param(n, marks=_ND2_PADDED_MULTICHANNEL_RGB_BUG)
    if n in ("lossless_padded_rgb", "padded_multichannel_rgb") else n
    for n in CASES])
def test_writer_agrees_with_nd2_library(tmp_path, name):
    """The nd2 library reads the synthetic files as the known data too, so
    the writer (and with it the tests above) follows the real format."""
    path, expected = _write(tmp_path, name)
    with nd2lib.ND2File(path) as f:
        sizes = dict(f.sizes)
        arr = f.asarray()
    assert sizes.get("P", 1) == len(expected)
    axes = list(sizes)
    for p, ref in enumerate(expected):
        a, ax = (arr[(slice(None),) * axes.index("P") + (p,)], [x for x in axes if x != "P"]) \
            if "P" in axes else (arr, list(axes))
        for k in "TCZ":
            if k not in ax:
                a, ax = np.expand_dims(a, 0), [k] + ax
        order = [ax.index(k) for k in "TCZYX"] + ([ax.index("S")] if "S" in ax else [])
        np.testing.assert_array_equal(a.transpose(order), ref)


def test_uncompressed_reads_only_the_rows_asked_for(tmp_path, monkeypatch):
    # the exact path: no skipped columns are read (see test_*_spans for merged rows)
    monkeypatch.setattr(nd2_backend, "SEPARATE_READ_BYTES", 0)
    path, expected = _write(tmp_path, "z_rgb_padded")
    with _native(path) as img:
        series = img.images[0]
        assert series.read_unit is None
        np.testing.assert_array_equal(series[0, 0, 2, 10:14], expected[0][0, 0, 2, 10:14])
        assert series.stats.pixels_read == 4 * 53 * 3                 # 4 whole rows
        series.stats.reset()
        np.testing.assert_array_equal(series[0, 0, 2, 10:14, 5:9], expected[0][0, 0, 2, 10:14, 5:9])
        assert series.stats.pixels_read == 4 * 4 * 3                  # narrow: only its bytes


def test_lossless_frames_decoded_once(tmp_path):
    path, expected = _write(tmp_path, "lossless")
    with _native(path) as img:
        series = img.images[0]
        assert series.read_unit == (1, 2, 1, 37, 53)
        series[0, 0, 1, 0:5, 0:5]
        assert series.stats.pixels_read == 37 * 53 * 2                # the whole frame
        series[0, 1, 1, 20:30, 10:40]
        assert series.stats.pixels_read == 37 * 53 * 2                # cached


def test_legacy_nd2_is_refused(tmp_path):
    path = str(tmp_path / "old.nd2")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<I", 0x0C000000) + b"\x00" * 200)
    with pytest.raises((UnsupportedFormatError, MicroReaderError)):
        _native(path, backend="nd2")


def test_truncated_file_is_refused(tmp_path):
    path, _ = _write(tmp_path, "t_p_z")
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 100)
    with pytest.raises(MicroReaderError):
        _native(path)


def test_close_releases_the_file(tmp_path, decode_threads):
    path, _ = _write(tmp_path, "lossless")
    img = _native(path)
    img.images[0][...]
    img.close()
    os.remove(path)
    with pytest.raises(ValueError):
        img.images[0][0]


def test_concurrent_reads(tmp_path, decode_threads):
    path, expected = _write(tmp_path, "t_p_z")
    errors = []
    with _native(path) as img:
        def worker(i):
            try:
                for p, (series, ref) in enumerate(zip(img.images, expected)):
                    for key in _keys(ref.shape, n=10, seed=i * 7 + p):
                        if not np.array_equal(series[key], ref[key]):
                            errors.append((p, key))
            except Exception as exc:        # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []


def test_uncompressed_narrow_reads_merge_rows(tmp_path):
    """Narrow columns: the few skipped bytes per row cost less than one read per row,
    so whole rows are read (as in LIF); pixels stay exact."""
    path, expected = _write(tmp_path, "z_rgb_padded")
    with _native(path) as img:
        series = img.images[0]
        np.testing.assert_array_equal(series[0, 0, 2, 10:14, 5:9], expected[0][0, 0, 2, 10:14, 5:9])
        assert series.stats.pixels_read == 4 * 53 * 3                 # 4 whole rows


# -- coordinate systems (plan 4.5) --------------------------------------------------------

CAMERA_TURNED = {"dStgLgCT11": -1.0, "dStgLgCT12": 0.0, "dStgLgCT21": 0.0, "dStgLgCT22": -1.0}


@pytest.fixture
def metadata_reads(monkeypatch):
    """How often an ND2 image's ``_read_metadata`` ran."""
    count = [0]
    real = nd2_backend._Nd2Image._read_metadata

    def counting(self):
        count[0] += 1
        return real(self)
    monkeypatch.setattr(nd2_backend._Nd2Image, "_read_metadata", counting)
    return count


def test_positions_are_in_the_stage_system(tmp_path, metadata_reads):
    from micro_reader._spatial import check_invariants
    path = str(tmp_path / "a.nd2")
    write_nd2(path, _frames(3, channels=1), [("P", 3)], picture=CAMERA_TURNED)
    with _native(path) as f:
        assert [s.id for s in f.coordinate_systems] == ["stage"]
        assert f.coordinate_systems[0].axes is None and f.coordinate_transformations == ()
        assert metadata_reads[0] == 0                       # declared without metadata
        check_invariants(f)                                 # I1-I3 on every image
        for image in f.images:
            assert image.metadata.position and image.metadata.coordinate_system == "stage"
        # every position of the file in one frame: the points 1 um apart in stage x
        # are 1 um apart along the image's x (turned by 180 degrees: -1)
        xs = [image.metadata.position["x"].value for image in f.images]
        assert xs[1] - xs[0] == pytest.approx(-1.0) and xs[2] - xs[1] == pytest.approx(-1.0)


@pytest.mark.parametrize("picture", [None, {"bCalibrated": False}, dict(CAMERA_TURNED, dStgLgCT22=0.0)],
                         ids=["no_camera_matrix", "uncalibrated", "singular_matrix"])
def test_files_without_a_position_declare_no_system(tmp_path, picture, metadata_reads):
    from micro_reader._spatial import check_invariants
    path = str(tmp_path / "a.nd2")
    if picture is not None and "dStgLgCT11" not in picture:
        picture = dict(CAMERA_TURNED, **picture)
    write_nd2(path, _frames(2, channels=1), [("P", 2)], picture=picture)
    with _native(path) as f:
        assert f.coordinate_systems == ()
        assert metadata_reads[0] == 0
        check_invariants(f)
        for image in f.images:
            assert image.metadata.position == {} and image.metadata.coordinate_system is None


def test_legacy_nd2_declares_no_system(tmp_path):
    pytest.importorskip("imagecodecs")             # encodes the JPEG 2000 frames
    from _nd2_legacy_writer import write_legacy_nd2
    from micro_reader._spatial import check_invariants
    path = str(tmp_path / "old.nd2")
    write_legacy_nd2(path, np.zeros((2, 1, 8, 10), np.uint16), [("P", 2)])
    with _native(path) as f:
        assert f.backend == "nd2" and f.coordinate_systems == ()
        check_invariants(f)
        assert all(image.metadata.position == {} and image.metadata.coordinate_system is None
                   for image in f.images)


def test_nd2_is_checked_strictly_by_the_session():
    import conftest
    assert "nd2" not in conftest.POSITIONS_WITHOUT_A_SYSTEM


# -- XY points named after wells: plates=True ---------------------------------------

def _named_points(tmp_path, names, valid=None):
    path = str(tmp_path / "plate.nd2")
    n = len(names) if valid is None else sum(valid)
    frames = np.arange(n * 6 * 8, dtype=np.uint16).reshape(n, 6, 8, 1, 1)
    write_nd2(path, frames, [("P", n)], point_names=names, valid_positions=valid)
    return path, frames


def test_points_named_after_wells_read_as_a_plate(tmp_path):
    # NIS-Elements JOBS names: well, then a counter; wells out of order
    names = ["B03_0000", "B02_0000", "B02_0001", "B03_0001", "C02_0000"]
    path, frames = _named_points(tmp_path, names)
    with micro_reader.open(path, plates=True) as f:
        (plate,) = f.plates
        assert plate.name == "plate"
        assert (len(plate.row_names), len(plate.column_names)) == (3, 4)   # the smallest: 12-well
        assert [(w.name, [x.index for x in w.fields]) for w in plate.wells] == [
            ("B2", [0, 1]), ("B3", [0, 1]), ("C2", [0])]
        # fields in acquisition order; images stay in the file's order
        assert [(im.index["well"], im.index["field"]) for im in f.images] == [
            (1, 0), (0, 0), (0, 1), (1, 1), (2, 0)]
        np.testing.assert_array_equal(plate.wells[0].fields[1].image[0, 0, 0], frames[2, ..., 0, 0])
    with micro_reader.open(path) as f:                         # without plates=True: as before
        assert f.plates == [] and f.images[0].index == {"series": 0}


@pytest.mark.parametrize("names, rows", [(["E8.1", "E8.2", "D5.1"], 6),    # 48-well
                                          (["A10_0000", "H12_0000"], 8)])  # 96-well
def test_other_nis_point_names(tmp_path, names, rows):
    path, _frames = _named_points(tmp_path, names)
    with micro_reader.open(path, plates=True) as f:
        assert len(f.plates[0].row_names) == rows


@pytest.mark.parametrize("names", [["P1", "P2"], ["A1", "A2"], ["B02_0000", "p1"],
                                   ["Position 1", "Position 2"]])
def test_points_that_are_not_all_wells_stay_positions(tmp_path, names):
    """A bare 'P1' or 'A1' names a position as often as a well: NIS-Elements'
    well points carry a counter or a zero-padded column; one point that is no
    well keeps the whole file as positions."""
    path, _frames = _named_points(tmp_path, names)
    with micro_reader.open(path, plates=True) as f:
        assert f.plates == [] and f.images[0].index == {"series": 0}


def test_only_the_valid_points_are_wells(tmp_path):
    path, _frames = _named_points(tmp_path, ["A01_0000", "B01_0000", "C01_0000"],
                                  valid=[1, 0, 1])
    with micro_reader.open(path, plates=True) as f:
        assert [w.name for w in f.plates[0].wells] == ["A1", "C1"]


ND2_PLATES = os.path.join(os.environ.get("MICROSCOPY_DATA",
                                         os.path.expanduser("~/Desktop/data/microscopy")),
                          "plates", "nd2", "zenodo-21162526")


@pytest.mark.parametrize("name, wells, fields", [
    ("nested_loop", 24, [1] * 24),          # A10_0000 ...: one point per well
    ("nd2jobs", 3, [4, 4, 4]),              # B02_0000 .. B02_0003: JOBS
    ("NDacquisition", 2, [2, 3]),           # E8.1 ..: ND acquisition (D5: 2, E8: 3)
])
def test_zenodo_21162526_point_names_make_plates(name, wells, fields):
    """Zenodo 21162526 (CC BY 4.0): NIS-Elements files whose XY points are
    named after wells in three conventions; the files name no plate format."""
    path = os.path.join(ND2_PLATES, f"{name}.nd2")
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path} (set MICROSCOPY_DATA)")
    with micro_reader.open(path, plates=True) as f:
        (plate,) = f.plates
        assert len(plate.wells) == wells
        assert [len(w.fields) for w in plate.wells] == fields


def test_the_job_s_well_plate_is_the_layout(tmp_path):
    """A file NIS-Elements acquired with a job holds the job's well plate
    (JobDefinitionV1_0): that is the layout, not the smallest standard plate."""
    path = str(tmp_path / "plate.nd2")
    frames = np.arange(2 * 6 * 8, dtype=np.uint16).reshape(2, 6, 8, 1, 1)
    write_nd2(path, frames, [("P", 2)], point_names=["B02_0000", "B03_0000"],
              wellplate=(16, 24, "384 well 1.5H Cellvis"))
    with micro_reader.open(path, plates=True) as f:
        assert (len(f.plates[0].row_names), len(f.plates[0].column_names)) == (16, 24)


def test_points_without_a_job_guess_the_layout_and_say_so(tmp_path):
    path, _frames = _named_points(tmp_path, ["B02_0000", "B03_0000"])
    with pytest.warns(micro_reader.PlateLayoutWarning, match="smallest standard plate"):
        micro_reader.open(path, plates=True).close()


# -- NIS-Elements well-plate runs: a folder of ND2 files ------------------------------

def _run(folder, files, *, wellplate=(8, 12, "Cellvis P96-1.5H-N"), positions=1, t=1):
    """A run folder: *files* {name: value}; each file one position (or
    *positions*) of t x 6 x 8 uint16 filled with value + frame number."""
    os.makedirs(folder, exist_ok=True)
    out = {}
    for name, value in files.items():
        n = t * positions
        frames = (np.arange(n * 6 * 8, dtype=np.uint16).reshape(n, 6, 8, 1, 1) % 97 + value)
        loops = [("T", t)] if positions == 1 else [("T", t), ("P", positions)]
        write_nd2(os.path.join(folder, name), frames, loops, wellplate=wellplate)
        out[name] = frames
    return out


def test_a_run_of_files_one_per_well_is_a_plate(tmp_path):
    folder = str(tmp_path / "20220701_140147")
    data = _run(folder, {"WellB02_ChannelDIC_Seq0000.nd2": 100, "WellB03_ChannelDIC_Seq0001.nd2": 200,
                         "WellF04_ChannelDIC_Seq0014.nd2": 300}, wellplate=(16, 24, "384 well"), t=3)
    first = os.path.join(folder, "WellB02_ChannelDIC_Seq0000.nd2")
    with micro_reader.open(os.path.join(folder, "WellF04_ChannelDIC_Seq0014.nd2"), plates=True) as f:
        assert f.format == "nd2-run"
        (plate,) = f.plates
        assert plate.name == "20220701_140147"
        assert (len(plate.row_names), len(plate.column_names)) == (16, 24)
        assert [(w.name, [x.index for x in w.fields]) for w in plate.wells] == [
            ("B2", [0]), ("B3", [0]), ("F4", [0])]
        im = plate.wells[2].fields[0].image
        assert im.index == {"plate": 0, "acquisition": 0, "well": 2, "field": 0}
        assert (im.axes, im.shape) == ("tczyx", (3, 1, 1, 6, 8))
        np.testing.assert_array_equal(im[:, 0, 0], data["WellF04_ChannelDIC_Seq0014.nd2"][..., 0, 0])
        assert [c.name for c in im.metadata.channels] == ["DIC"] or im.metadata.channels[0].name
    for kwargs in ({}, {"plates": True, "multifile": False}):         # the one file
        with micro_reader.open(first, **kwargs) as f:
            assert len(f.images) == 1 and f.plates == []
    g = micro_reader.group_files(folder, plates=True)
    (group,) = g.groups
    assert not g.singles and len(group.members) == 2            # the run's other files
    assert os.path.basename(group.main) in data
    assert len(micro_reader.group_files(folder).singles) == 3       # without plates: files


def test_a_run_of_files_one_per_point_has_the_points_as_fields(tmp_path):
    folder = str(tmp_path / "run")
    data = _run(folder, {"WellA01_PointA01_0001_ChannelFITC BP_Seq0001.nd2": 10,
                         "WellA01_PointA01_0000_ChannelFITC BP_Seq0000.nd2": 20,
                         "WellB01_PointB01_0000_ChannelFITC BP_Seq0002.nd2": 30})
    with micro_reader.open(os.path.join(folder, "WellB01_PointB01_0000_ChannelFITC BP_Seq0002.nd2"),
                           plates=True) as f:
        (plate,) = f.plates
        assert [(w.name, [x.index for x in w.fields]) for w in plate.wells] == [
            ("A1", [0, 1]), ("B1", [0])]
        im = plate.wells[0].fields[1].image                         # point counter 1
        np.testing.assert_array_equal(im[0, 0, 0], data[
            "WellA01_PointA01_0001_ChannelFITC BP_Seq0001.nd2"][0, ..., 0, 0])


def test_channels_in_files_of_their_own_are_stacked(tmp_path):
    folder = str(tmp_path / "run")
    data = _run(folder, {"WellC03_ChannelDAPI_Seq0000.nd2": 1000, "WellC03_ChannelGFP_Seq0001.nd2": 2000})
    with micro_reader.open(os.path.join(folder, "WellC03_ChannelGFP_Seq0001.nd2"), plates=True) as f:
        (im,) = f.images
        assert im.shape == (1, 2, 1, 6, 8)
        # its channel files are not ragged parts: one sampling, one part
        assert not im.is_ragged and im.parts == [im]
        np.testing.assert_array_equal(im[0, 1, 0], data["WellC03_ChannelGFP_Seq0001.nd2"][0, ..., 0, 0])
        np.testing.assert_array_equal(im[0, :, 0, 2:4, 1:5][0],
                                      data["WellC03_ChannelDAPI_Seq0000.nd2"][0, 2:4, 1:5, 0, 0])


def test_a_run_micro_reader_cannot_place_is_refused(tmp_path):
    folder = str(tmp_path / "twice")
    _run(folder, {"WellA01_ChannelDIC_Seq0000.nd2": 1, "WellA01_ChannelDIC_Seq0001.nd2": 2})
    with pytest.raises(UnsupportedFormatError, match="several files of one channel.*multifile=False"):
        micro_reader.open(os.path.join(folder, "WellA01_ChannelDIC_Seq0000.nd2"), plates=True)
    folder = str(tmp_path / "positions")
    _run(folder, {"WellA01_ChannelDIC_Seq0000.nd2": 1}, positions=2)
    with pytest.raises(UnsupportedFormatError, match="holds 2 XY positions"):
        micro_reader.open(os.path.join(folder, "WellA01_ChannelDIC_Seq0000.nd2"), plates=True)


def test_a_run_of_more_files_than_stay_open_reads_every_file(tmp_path, monkeypatch):
    """The run keeps OPEN_FILES (32) members open; a closed one opens again on its next
    read. It did not: every field of a 96-well run past the 32nd failed with 'I/O
    operation on closed file'."""
    from micro_reader.backends import _nd2_run
    monkeypatch.setattr(_nd2_run, "OPEN_FILES", 2)
    folder = str(tmp_path / "run")
    names = {f"Well{r}{c:02d}_ChannelDIC_Seq{i:04d}.nd2": 100 * (i + 1)
             for i, (r, c) in enumerate([("A", 1), ("A", 2), ("B", 1), ("B", 2), ("C", 1)])}
    data = _run(folder, names)
    with micro_reader.open(os.path.join(folder, next(iter(names))), plates=True) as f:
        assert len(f.images) == 5
        for _ in range(2):                                      # every member closed once
            for im, name in zip(f.images, names):
                np.testing.assert_array_equal(im[0, 0, 0], data[name][0, ..., 0, 0])


def test_a_run_of_rgb_files_is_refused_by_name(tmp_path):
    folder = str(tmp_path / "rgb")
    os.makedirs(folder)
    frames = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(1, 6, 8, 1, 3)
    for name in ("WellA01_ChannelBF_Seq0000.nd2", "WellA02_ChannelBF_Seq0001.nd2"):
        write_nd2(os.path.join(folder, name), frames, [("T", 1)], wellplate=(8, 12, "96 well"))
    with pytest.raises(UnsupportedFormatError, match="RGB samples.*multifile=False"):
        micro_reader.open(os.path.join(folder, "WellA01_ChannelBF_Seq0000.nd2"), plates=True)


ND2_RUNS = os.path.join(os.environ.get("MICROSCOPY_DATA",
                                       os.path.expanduser("~/Desktop/data/microscopy")),
                        "plates", "nd2")


@pytest.mark.parametrize("folder, layout, wells, fields, shape", [
    ("bia-S-BIAD3591/4x_overview_20250306_171942_094", (8, 12), 24, 24, (1, 1, 6, 576, 576)),
    ("zenodo-18927821-96well/20260116_094944_372", (8, 12), 40, 360, (1, 1, 1, 1024, 1024)),
    ("zenodo-10139789-384well/microchamber/384well_220701/20220701_140147", (16, 24), 4, 4,
     (33, 1, 1, 2304, 2304)),
])
def test_real_nis_runs_read_as_plates(folder, layout, wells, fields, shape):
    """BIA S-BIAD3591 (one file per well), Zenodo 18927821 (one per point, 9 per
    well), Zenodo 10139789 (one per well, 4 of a 384-well plate, t 33): the
    layout from each file's job (phenoplate-96, Cellvis P96, 384 well Cellvis)."""
    path = os.path.join(ND2_RUNS, *folder.split("/"))
    if not os.path.exists(path):
        pytest.skip(f"sample not found: {path} (set MICROSCOPY_DATA)")
    first = sorted(e.path for e in os.scandir(path) if e.name.endswith(".nd2"))[0]
    with micro_reader.open(first, plates=True) as f:
        (plate,) = f.plates
        assert (len(plate.row_names), len(plate.column_names)) == layout
        assert len(plate.wells) == wells and sum(len(w.fields) for w in plate.wells) == fields
        im = plate.wells[0].fields[0].image
        assert im.shape == shape and im.metadata.coordinate_system == "stage"
        with micro_reader.open(first) as alone:
            np.testing.assert_array_equal(im[0, 0, 0, :64, :64], alone.images[0][0, 0, 0, :64, :64])
