"""LIF backend on synthetic files (``_lif_writer``): every layout the reader
distinguishes, checked against the written arrays, and against readlif where
readlif can read the layout."""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

import micro_reader
from micro_reader import CorruptFileError, UnsupportedFormatError

from _lif_writer import folder, image, write_lif


def _native(*args, **kwargs):
    """``micro_reader.open`` with RGB samples kept as an ``s`` axis: these tests
    check the backend's own layout (open's default folds them into channels)."""
    return micro_reader.open(*args, **{"samples": "keep", **kwargs})

RNG = np.random.default_rng(0)


def _ints(shape, dtype="u2", top=None):
    top = top or np.iinfo(dtype).max
    return RNG.integers(0, top, size=shape, endpoint=True).astype(dtype)


# name -> (elements, expected [(image name, axes, array)], write kwargs); each
# tile is its own image (docs/AXES.md)
def _cases():
    stack = _ints((2, 3, 4, 20, 30), top=4095)
    u8 = _ints((1, 2, 5, 17, 23), "u1")
    f32 = RNG.normal(size=(1, 1, 3, 16, 12)).astype("f4")
    f16 = RNG.normal(size=(1, 1, 2, 16, 12)).astype("f2")
    u32 = _ints((1, 1, 2, 9, 11), "u4")
    tl = _ints((3, 2, 2, 4, 10, 12))                                  # t c z l y x
    tiles = _ints((1, 1, 2, 3, 16, 20))                               # t c z m y x
    rgb = _ints((2, 3, 18, 25, 3), "u1")                              # t z y x s (RGB)
    padded = _ints((1, 2, 3, 15, 21))
    order = _ints((2, 3, 4, 12, 14))
    return {
        "stack_12bit": ([image("stack", stack, resolution=12)], [("stack", "tczyx", stack)], {}),
        "uint8": ([image("u8", u8)], [("u8", "tczyx", u8)], {}),
        "float32": ([image("f32", f32)], [("f32", "tczyx", f32)], {}),
        "float16_flim": ([image("Fast Flim", f16)], [("Fast Flim", "tczyx", f16)], {}),
        "uint32": ([image("mask", u32)], [("mask", "tczyx", u32)], {}),
        # a lambda scan: one image per emission band (index "emission")
        "time_lambda": ([image("lambda", tl, "tczlyx")],
                        [("lambda", "tczyx", tl[:, :, :, k]) for k in range(4)], {}),
        "tiles": ([image("scan", tiles, "tczmyx",
                         tiles=[(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0), (2, 0, 2e-4, 0.0)])],
                  [("scan", "tczyx", tiles[:, :, :, m]) for m in range(3)], {}),
        "rgb": ([image("colour", rgb, "tzyxs", rgb=True)],
                [("colour", "tczyxs", rgb[:, None])], {}),
        "padded_rows": ([image("pad", padded, row_pad=6)], [("pad", "tczyx", padded)], {}),
        "z_before_c": ([image("zc", order, order="xyzct")], [("zc", "tczyx", order)], {}),
        "version1": ([image("v1", u8)], [("v1", "tczyx", u8)], {"version": 1}),
        "nested": ([image("a", u8), folder("group", [image("b", padded),
                                                     folder("empty"),
                                                     image("c", u8[:, :1])])],
                   [("a", "tczyx", u8), ("group/b", "tczyx", padded),
                    ("group/c", "tczyx", u8[:, :1])], {}),
    }


CASES = _cases()


def _keys(shape, n=20, seed=3):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,)]
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


def _write(tmp_path, name):
    elements, expected, kwargs = CASES[name]
    path = str(tmp_path / f"{name}.lif")
    write_lif(path, elements, **kwargs)
    return path, expected


@pytest.mark.parametrize("name", list(CASES))
def test_matches_written_data(tmp_path, name, decode_threads):
    path, expected = _write(tmp_path, name)
    with _native(path) as img:
        assert img.backend == "lif"
        assert [(s.name, s.axes) for s in img.images] == [(n, a) for n, a, _ in expected]
        for series, (_, _, ref) in zip(img.images, expected):
            assert series.shape == ref.shape and series.dtype == ref.dtype
            for key in _keys(ref.shape):
                got = series[key]
                assert got.dtype.isnative
                np.testing.assert_array_equal(got, ref[key], err_msg=f"{name} {key}")


@pytest.mark.parametrize("name", ["stack_12bit", "uint8", "tiles", "z_before_c", "version1",
                                  "nested", "float16_flim"])
def test_agrees_with_readlif(tmp_path, name):
    """readlif (eubi-bridge's LIF path, via bioio-lif) reads the same planes;
    it returns half floats as their uint16 bits.  Not compared: lambda
    (readlif has no lambda axis and returns the wrong planes) and padded
    rows (readlif ignores the row stride) -- both readlif limitations,
    checked against Bio-Formats with harness/conformance_lif.py."""
    readlif = pytest.importorskip("readlif.reader")
    path, expected = _write(tmp_path, name)
    lf = readlif.LifFile(path)
    with _native(path) as img:
        assert list(dict.fromkeys(s.name for s in img.images)) == \
            [i["name"] for i in lf.image_list]
        for series in img.images:
            i, m = series.index["series"], series.index.get("tile")
            for t in range(series.shape[0]):
                for c in range(series.shape[1]):
                    for z in range(series.shape[2]):
                        kw = {"m": m} if m is not None else {}
                        ref = np.asarray(lf.get_image(i).get_frame(z=z, t=t, c=c, **kw))
                        got = series.read(t=t, c=c, z=z)
                        if got.dtype == np.float16:
                            got = got.view(np.uint16)
                        np.testing.assert_array_equal(got, ref, err_msg=f"{name} {i} {t} {c} {z}")


def test_rgb_samples_follow_the_channel_tags(tmp_path):
    """Colour stored B, G, R (tags 3, 2, 1) comes out R, G, B; stored R, G, B
    (tags 1, 2, 3) too."""
    rgb = _ints((1, 1, 6, 7, 3), "u1")
    for stored_bgr in (True, False):
        path = str(tmp_path / f"rgb_{stored_bgr}.lif")
        el = image("c", rgb, "tzyxs", rgb=stored_bgr,
                   tags=None if stored_bgr else [1, 2, 3])
        write_lif(path, [el])
        with _native(path) as img:
            np.testing.assert_array_equal(img.images[0][0, 0, 0], rgb[0, 0])     # rgb: t z y x s
            np.testing.assert_array_equal(img.images[0][0, 0, 0, :, :, 2], rgb[0, 0, :, :, 2])


def test_tiles_are_images_with_positions(tmp_path):
    path, _ = _write(tmp_path, "tiles")
    with _native(path) as img:
        assert img.indices == [{"series": 0, "tile": m} for m in range(3)]
        with pytest.warns(DeprecationWarning, match="metadata.position"):
            assert [s.tile_position for s in img.images] == [(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0),
                                                             (2, 0, 2e-4, 0.0)]
        # the replacement: each tile's position, along the image's axes (um)
        assert [round(s.metadata.position["x"].value) for s in img.images] == [0, 100, 200]


def test_tiles_compose_a_mosaic_at_their_stage_positions(tmp_path):
    """tiles="mosaic": one image per scan, each tile at its stage position (9 px wide
    tiles of 12.5 um pixels, 100 um apart: 8 px steps, one column of overlap that the
    later tile supplies), placed at the first tile's position."""
    data = _ints((1, 1, 1, 3, 8, 9))
    grid = [(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0), (2, 0, 2e-4, 0.0)]
    path = str(tmp_path / "scan.lif")
    write_lif(path, [image("scan", data, "tczmyx", tiles=grid)])
    with _native(path, tiles="mosaic") as f:
        (mosaic,) = f.images
        assert dict(mosaic.index) == {"series": 0} and mosaic.shape == (1, 1, 1, 8, 25)
        canvas = mosaic[0, 0, 0]
        expected = np.zeros((8, 25), data.dtype)
        for m in range(3):                                       # later tiles on top
            expected[:, 8 * m:8 * m + 9] = data[0, 0, 0, m]
        np.testing.assert_array_equal(canvas, expected)
        assert round(mosaic.metadata.position["x"].value) == 0
        assert mosaic.metadata.pixel_sizes["x"].to("micrometer").value == pytest.approx(12.5)


def test_a_mosaic_of_unplaced_tiles_is_refused(tmp_path):
    data = _ints((1, 1, 1, 2, 8, 9))
    path = str(tmp_path / "flipped.lif")
    write_lif(path, [image("flipped", data, "tczmyx", tiles=[(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0)],
                           tile_flags={"FlipX": "1"})])
    with pytest.raises(UnsupportedFormatError, match="no stage position"):
        _native(path, tiles="mosaic").images[0].metadata


def test_untiled_images_of_a_tiled_file_get_tile_0(tmp_path):
    tiles = _ints((1, 1, 1, 2, 8, 9))
    u8 = _ints((1, 1, 1, 8, 9), "u1")
    path = str(tmp_path / "mixed.lif")
    write_lif(path, [image("plain", u8), image("scan", tiles, "tczmyx")])
    with _native(path) as img:
        assert img.indices == [{"series": 0, "tile": 0}, {"series": 1, "tile": 0},
                                  {"series": 1, "tile": 1}]
        assert img.index_range == {"series": 2, "tile": 2}
        np.testing.assert_array_equal(img.select(series=1, tile=1)[0][...], tiles[:, :, :, 1])


def _patch(path, old, new):
    data = open(path, "rb").read()
    old, new = old.encode("utf-16-le"), new.encode("utf-16-le")
    assert old in data
    with open(path, "wb") as fh:
        fh.write(data.replace(old, new))


def test_rotations_are_images(tmp_path):
    """DimID 6 (rotation): one image per rotation, like tiles."""
    path, expected = _write(tmp_path, "tiles")
    _patch(path, 'DimID="10"', 'DimID="6" ')                     # same length
    with _native(path) as img:
        assert img.indices == [{"series": 0, "rotation": r} for r in range(3)]
        for r, image_ in enumerate(img.images):
            np.testing.assert_array_equal(image_[...], expected[r][2])


def test_lambda_scans_are_emission_and_excitation_indices(tmp_path):
    path, expected = _write(tmp_path, "time_lambda")
    with _native(path) as img:
        assert img.indices == [{"series": 0, "emission": k} for k in range(4)]
    _patch(path, 'DimID="5"', 'DimID="9"')
    with _native(path) as img:
        assert img.indices == [{"series": 0, "excitation": k} for k in range(4)]
        for k, image_ in enumerate(img.images):
            assert image_.axes == "tczyx"
            np.testing.assert_array_equal(image_[...], expected[k][2])


def test_reads_only_the_rows_asked(tmp_path):
    """A window is one contiguous read from its first to its last pixel."""
    path, expected = _write(tmp_path, "stack_12bit")
    with _native(path) as img:
        series = img.images[0]
        np.testing.assert_array_equal(series[1, 2, 3, 5:9, 4:10], expected[0][2][1, 2, 3, 5:9, 4:10])
        assert series.stats.pixels_read == 3 * 30 + 6       # rows 5..8, columns 4..9


def test_unsupported_dimension_is_named(tmp_path):
    path, _ = _write(tmp_path, "uint8")
    data = open(path, "rb").read()
    text = 'DimID="3" NumberOfElements="5"'.encode("utf-16-le")
    assert text in data
    patched = data.replace(text, 'DimID="7" NumberOfElements="5"'.encode("utf-16-le"))
    with open(path, "wb") as fh:
        fh.write(patched)
    with pytest.raises(UnsupportedFormatError, match="XT slice"):
        _native(path)


def test_bit_packed_samples_are_refused(tmp_path):
    """A non-zero BitInc means samples not aligned to bytes: refused, never
    read as if aligned."""
    path, _ = _write(tmp_path, "uint8")
    data = open(path, "rb").read()
    text = 'BitInc="0"'.encode("utf-16-le")
    assert text in data
    with open(path, "wb") as fh:
        fh.write(data.replace(text, 'BitInc="4"'.encode("utf-16-le"), 1))
    with pytest.raises(UnsupportedFormatError, match="bit-packed"):
        _native(path)


def test_truncated_file_is_reported(tmp_path):
    path, _ = _write(tmp_path, "stack_12bit")
    with open(path, "r+b") as fh:
        fh.truncate(os.path.getsize(path) - 100)
    with pytest.raises(CorruptFileError, match="truncated"):
        _native(path)


def test_close_releases_the_file(tmp_path, decode_threads):
    path, _ = _write(tmp_path, "stack_12bit")
    img = _native(path)
    img.images[0][...]
    img.close()
    os.remove(path)                                        # Windows: fails if still open
    with pytest.raises(ValueError):
        img.images[0][0]


def test_concurrent_reads(tmp_path, decode_threads):
    path, expected = _write(tmp_path, "time_lambda")
    ref = expected[0][2]
    errors = []
    with _native(path) as img:
        series = img.images[0]

        def work(seed):
            for key in _keys(ref.shape, n=15, seed=seed):
                if not np.array_equal(series[key], ref[key]):
                    errors.append(key)
        threads = [threading.Thread(target=work, args=(s,)) for s in range(6)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    assert not errors


def test_one_metre_per_pixel_is_index_space(tmp_path):
    """LAS stores FLIM phasor plots with Length = N - 1 m (1 m per pixel):
    no pixel size, while the image beside it keeps its own."""
    plot = np.ones((30, 51), np.float32)
    path = str(tmp_path / "flim.lif")
    write_lif(path, [image("Fast Flim", np.ones((8, 10), np.float16), "yx"),
                     image("Phasor Plot", plot, "yx", lengths={"x": 50, "y": 29})])
    with _native(path) as f:
        fast, phasor = f.images
        assert set(fast.metadata.pixel_sizes) == {"x", "y"}
        assert phasor.metadata.pixel_sizes == {}
        np.testing.assert_array_equal(phasor[...].reshape(plot.shape), plot)


# -- coordinate systems (plan 4.4, 4.14, 4.15) ----------------------------------------------

#: a FILETIME (2019-07-19, as in a real SP8 file) and one hour in its ticks
T0, HOUR = 0x1d53e3b64eca2c0, 3600 * 10 ** 7
SERIAL = "8100000388"


def _run_id(clock_seq=0x04b5, node=0xeccd6d7240e6):
    """A version-1 UniqueID of one LAS X run (its PC and clock sequence)."""
    import uuid
    return str(uuid.uuid1(node=node, clock_seq=clock_seq))


def _placed(name, x_um, *, start=T0, serial=SERIAL, run=None, stamps=True, **kwargs):
    """An image element with a stage position (*x_um*, 0) and what ties it to a
    session: the microscope's serial, the run's UniqueID (*run* False: none),
    two time stamps from *start* (*stamps* False: none)."""
    setting = {"StagePosX": x_um * 1e-6, "StagePosY": 0.0}
    if serial is not None:
        setting["SystemSerialNumber"] = serial
    return image(name, _ints((1, 1, 1, 8, 9), "u1"), setting=setting,
                 unique_id=_run_id() if run is None else run or None,
                 timestamps=[int(start), int(start) + 10 ** 7] if stamps else None, **kwargs)


@pytest.fixture
def metadata_reads(monkeypatch):
    """How often a LIF image's ``_read_metadata`` ran."""
    from micro_reader.backends import lif as lif_backend
    count = [0]
    real = lif_backend._LifImage._read_metadata

    def counting(self):
        count[0] += 1
        return real(self)
    monkeypatch.setattr(lif_backend._LifImage, "_read_metadata", counting)
    return count


def _systems(path, metadata_reads):
    """(declared ids, each image's system) of a LIF, checking every invariant,
    that declaring reads no metadata, and that the backend's test of which
    series are placed (``has_position``) agrees with the positions it gives."""
    from micro_reader._spatial import check_invariants
    from micro_reader.backends._lif_metadata import has_position
    metadata_reads[0] = 0
    with _native(path) as f:
        ids = [s.id for s in f.coordinate_systems]
        assert metadata_reads[0] == 0                        # declared without metadata
        assert all(s.axes is None and not s.estimated for s in f.coordinate_systems)
        assert f.coordinate_transformations == ()
        check_invariants(f)                                  # I1-I3 on every image
        for series in sorted({dict.get(i.index, "series") for i in f.images}):
            images = f.select(series=series)
            tiles = images[0]._layout.size.get("tile", 1)
            assert has_position(images[0]._element, tiles) == \
                any(i.metadata.position for i in images)
        return ids, [i.metadata.coordinate_system for i in f.images]


def test_one_session_is_one_stage_system(tmp_path, metadata_reads):
    path = str(tmp_path / "a.lif")
    write_lif(path, [_placed("a", 100.0), _placed("b", 250.0, start=T0 + HOUR),
                     folder("f", [_placed("c", 400.0, start=T0 + 2 * HOUR)])])
    assert _systems(path, metadata_reads) == (["stage"], ["stage"] * 3)
    with _native(path) as f:                      # positions as before
        assert [round(i.metadata.position["x"].value) for i in f.images] == [100, 250, 400]


@pytest.mark.parametrize("third", [
    dict(start=T0 + 3 * HOUR),                              # a break of 2.5 h: another session
    dict(serial="8100000783"),                              # another microscope
    dict(run=_run_id(clock_seq=0x2fa5)),                    # another LAS X run
    dict(run=_run_id(node=0x00506220276f)),                 # another acquisition PC
], ids=["gap", "serial", "clock_seq", "node"])
def test_each_session_of_a_file_gets_its_own_system(tmp_path, third, metadata_reads):
    """Decision K1-2: one system per session, stage-r<k> in series order; the
    two series of the first session stay in one frame."""
    path = str(tmp_path / "a.lif")
    write_lif(path, [_placed("a", 100.0), _placed("b", 250.0, start=T0 + HOUR / 2),
                     _placed("c", 400.0, **{"start": T0 + HOUR, **third})])
    assert _systems(path, metadata_reads) == (["stage-r0", "stage-r1"],
                                              ["stage-r0", "stage-r0", "stage-r1"])


@pytest.mark.parametrize("third", [
    dict(serial=None),                                      # no serial: cannot tell
    dict(run=False),                                        # no UniqueID: cannot tell
    dict(run="not-a-uuid"),
    dict(run="1b4e28ba-2fa1-41d2-883f-0016d3cca427"),       # version 4: no run in it
    dict(stamps=False),                                     # no time stamps: cannot tell
], ids=["no_serial", "no_uid", "bad_uid", "uuid4", "no_time"])
def test_a_series_that_cannot_be_tied_to_a_session_gets_its_own_system(tmp_path, third,
                                                                      metadata_reads):
    """Plan 4.15 / 8.17 for that series alone: stage-s<series>; the others keep
    their session's system."""
    path = str(tmp_path / "a.lif")
    write_lif(path, [_placed("a", 100.0), _placed("b", 250.0, start=T0 + HOUR / 2),
                     _placed("c", 400.0, **{"start": T0 + HOUR, **third})])
    assert _systems(path, metadata_reads) == (["stage-r0", "stage-s2"],
                                              ["stage-r0", "stage-r0", "stage-s2"])


def test_a_session_split_by_a_gap_and_rejoined_by_its_run_stays_apart(tmp_path, metadata_reads):
    """Sessions are cut at breaks within one run, then numbered in series order."""
    path = str(tmp_path / "a.lif")
    write_lif(path, [_placed("a", 100.0, start=T0 + 5 * HOUR), _placed("b", 250.0),
                     _placed("c", 400.0, start=T0 + 5.5 * HOUR)])
    assert _systems(path, metadata_reads) == (["stage-r0", "stage-r1"],
                                              ["stage-r0", "stage-r1", "stage-r0"])


def test_a_break_inside_the_gap_keeps_one_session(tmp_path, metadata_reads):
    path = str(tmp_path / "a.lif")
    write_lif(path, [_placed("a", 100.0), _placed("b", 250.0, start=T0 + 1.9 * HOUR)])
    assert _systems(path, metadata_reads)[0] == ["stage"]


def test_unplaced_images_have_no_system_and_do_not_decide(tmp_path, metadata_reads):
    """An image without a stage position (an analysis result) is unplaced; its
    time stamps neither split one session nor bridge two."""
    result = _ints((1, 1, 1, 8, 9), "u1")
    path = str(tmp_path / "a.lif")
    write_lif(path, [_placed("a", 100.0), image("result", result, timestamps=[T0 + 9 * HOUR]),
                     _placed("b", 250.0, start=T0 + HOUR)])
    assert _systems(path, metadata_reads) == (["stage"], ["stage", None, "stage"])
    bridged = str(tmp_path / "b.lif")
    write_lif(bridged, [_placed("a", 100.0),
                        image("result", result, unique_id=_run_id(),
                              timestamps=[T0 + int(1.5 * HOUR), T0 + 3 * HOUR]),
                        _placed("b", 250.0, start=T0 + 3.2 * HOUR)])
    assert _systems(bridged, metadata_reads) == (["stage-r0", "stage-r1"],
                                                 ["stage-r0", None, "stage-r1"])


def test_a_file_without_positions_declares_no_system(tmp_path, metadata_reads):
    path, _ = _write(tmp_path, "stack_12bit")
    assert _systems(path, metadata_reads) == ([], [None])


def test_tiles_of_a_swapped_or_flipped_scan_are_unplaced(tmp_path, metadata_reads):
    """4.4: tile positions of a scan with SwapXY / FlipX / FlipY are in a frame
    nobody has verified: no position, no system.  An unflagged scan in the
    same file keeps its tiles' positions."""
    tiles = _ints((1, 1, 1, 2, 8, 9))
    grid = [(0, 0, 0.0, 0.0), (1, 0, 1e-4, 0.0)]
    session = dict(setting={"SystemSerialNumber": SERIAL, "StagePosX": 0.0, "StagePosY": 0.0},
                   unique_id=_run_id(), timestamps=[T0, T0 + 10 ** 7])
    for flag in ("SwapXY", "FlipX", "FlipY"):
        path = str(tmp_path / f"{flag}.lif")
        write_lif(path, [image("flagged", tiles, "tczmyx", tiles=grid, tile_flags={flag: "1"},
                               **session),
                         image("plain", tiles, "tczmyx", tiles=grid, **session)])
        assert _systems(path, metadata_reads) == (["stage"], [None, None, "stage", "stage"])
        with _native(path) as f:
            assert all(i.metadata.position == {} for i in f.select(series=0))


def test_tiles_share_their_series_system(tmp_path, metadata_reads):
    """The synthetic tile scan records no microscope or run: one system for
    its series (4.15), every tile placed in it."""
    path, _ = _write(tmp_path, "tiles")
    assert _systems(path, metadata_reads) == (["stage-s0"], ["stage-s0"] * 3)


def test_systems_survive_open_with_axes(tmp_path):
    path = str(tmp_path / "a.lif")
    write_lif(path, [_placed("a", 100.0), _placed("b", 250.0, start=T0 + HOUR)])
    with _native(path, axes="tczyx") as f:
        assert [s.id for s in f.coordinate_systems] == ["stage"]
        assert [i.metadata.coordinate_system for i in f.images] == ["stage", "stage"]


def test_lif_is_checked_strictly_by_the_session():
    import conftest
    assert "lif" not in conftest.POSITIONS_WITHOUT_A_SYSTEM
