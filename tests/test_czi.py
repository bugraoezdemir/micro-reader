"""CZI backend on synthetic files written with pylibCZIrw (test-only), checked
against the known arrays and against pylibCZIrw reading the same files.

Most tests check composition, so they open files with ``tiles="mosaic"``;
image keys and separate tiles are in test_czi_keys.py."""
from __future__ import annotations

import os
import struct
import threading
import warnings

import numpy as np
import pytest

pyczi = pytest.importorskip("pylibCZIrw.czi")

import micro_reader  # noqa: E402
from micro_reader import CorruptFileError, DamagedFileWarning  # noqa: E402
from micro_reader.backends.czi import _directory  # noqa: E402

RNG = np.random.default_rng(0)


def _plane(h=40, w=56, dtype="u2", samples=1):
    shape = (h, w) if samples == 1 else (h, w, samples)
    return RNG.integers(1, np.iinfo(dtype).max, size=shape, endpoint=True).astype(dtype)


def write_czi(path, tiles, dtype="u2", samples=1, compression=None):
    """tiles: [(scene, plane dict, (x, y), array)] in writing (= mosaic) order.
    Returns {scene: expected composite} as t c z y x [s], plus the scene
    origins: later tiles on top, background 0, box = union of the tiles.
    *compression*: pylibCZIrw options, e.g. "zstd1:ExplicitLevel=2"."""
    with pyczi.create_czi(str(path), exist_ok=True, compression_options=compression) as czi:
        for scene, plane, (x, y), data in tiles:
            czi.write(data=data if samples > 1 else data[..., None], location=(x, y),
                      plane=plane, scene=scene)
    expected, origins = {}, {}
    for scene in sorted({t[0] for t in tiles}):
        mine = [t for t in tiles if t[0] == scene]
        x0 = min(x for _, _, (x, _), _ in mine)
        y0 = min(y for _, _, (_, y), _ in mine)
        x1 = max(x + d.shape[1] for _, _, (x, _), d in mine)
        y1 = max(y + d.shape[0] for _, _, (_, y), d in mine)
        nt = max(p.get("T", 0) for _, p, _, _ in tiles) + 1
        nc = max(p.get("C", 0) for _, p, _, _ in tiles) + 1
        nz = max(p.get("Z", 0) for _, p, _, _ in tiles) + 1
        out = np.zeros((nt, nc, nz, y1 - y0, x1 - x0) + ((samples,) if samples > 1 else ()),
                       dtype)
        for _, plane, (x, y), data in mine:
            rgb = data[..., ::-1] if samples > 1 else data       # CZI stores B, G, R
            out[plane.get("T", 0), plane.get("C", 0), plane.get("Z", 0),
                y - y0:y - y0 + data.shape[0], x - x0:x - x0 + data.shape[1]] = rgb
        expected[scene], origins[scene] = out, (x0, y0)
    return expected, origins


def _stack(scene=0, t=2, c=2, z=3, at=(0, 0)):
    return [(scene, {"T": ti, "C": ci, "Z": zi}, at, _plane())
            for ti in range(t) for ci in range(c) for zi in range(z)]


def _mosaic(scene=0, c=2, overlap=10, at=(100, 50)):
    """3 x 2 tiles of 40 x 56 overlapping by *overlap* pixels."""
    tiles = []
    for ci in range(c):
        for j in range(2):
            for i in range(3):
                tiles.append((scene, {"T": 0, "C": ci, "Z": 0},
                              (at[0] + i * (56 - overlap), at[1] + j * (40 - overlap)), _plane()))
    return tiles


CASES = {
    "stack":         lambda: dict(tiles=_stack()),
    "mosaic_overlap": lambda: dict(tiles=_mosaic()),
    "scenes":        lambda: dict(tiles=_stack(0, 1, 2, 2, (0, 0)) + _stack(1, 1, 2, 2, (500, 300))),
    "scene_mosaics": lambda: dict(tiles=_mosaic(0, 1) + _mosaic(1, 1, 0, (900, 900))),
    "uint8":         lambda: dict(tiles=[(0, {"C": 0}, (0, 0), _plane(dtype="u1"))], dtype="u1"),
    "rgb":           lambda: dict(tiles=[(0, {"T": 0, "C": 0, "Z": 0}, (0, 0),
                                          _plane(dtype="u1", samples=3))], dtype="u1", samples=3),
    # compressed subblocks (decoded by micro-reader-codecs when installed)
    "zstd0_stack":   lambda: dict(tiles=_stack(), compression="zstd0:ExplicitLevel=2"),
    "zstd1_mosaic":  lambda: dict(tiles=_mosaic(), compression="zstd1:ExplicitLevel=2"),
    "zstd1_hilo_scenes": lambda: dict(
        tiles=_stack(0, 1, 2, 2, (0, 0)) + _mosaic(1, 1, 5, (400, 400)),
        compression="zstd1:ExplicitLevel=2;PreProcess=HiLoByteUnpack"),
    "zstd1_rgb":     lambda: dict(tiles=[(0, {"T": 0, "C": 0, "Z": 0}, (0, 0),
                                          _plane(dtype="u1", samples=3))], dtype="u1", samples=3,
                                  compression="zstd1:ExplicitLevel=1"),
}


def _write(tmp_path, name):
    case = CASES[name]()
    path = str(tmp_path / f"{name}.czi")
    expected, origins = write_czi(path, **case)
    return path, expected, origins


def _keys(shape, n=15, seed=5):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            b = int(rng.integers(a + 1, size + 1))
            key.append(a if rng.integers(0, 4) == 0 else slice(a, b))
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
    path, expected, origins = _write(tmp_path, name)
    with micro_reader.open(path, tiles="mosaic") as img:
        assert img.backend == "czi" and not img.damage
        assert len(img.images) == len(expected)
        for scene, sid in zip(img.images, sorted(expected)):
            ref = expected[sid]
            assert scene.origin == origins[sid]
            assert scene.shape == ref.shape and scene.dtype == ref.dtype
            for key in _keys(ref.shape):
                np.testing.assert_array_equal(scene[key], ref[key], err_msg=f"{name} {key}")


@pytest.mark.parametrize("name", list(CASES))
def test_agrees_with_pylibczirw(tmp_path, name):
    """pylibCZIrw (eubi-bridge's CZI path) composes the same pixels."""
    path, expected, _ = _write(tmp_path, name)
    with micro_reader.open(path, tiles="mosaic") as img, pyczi.open_czi(path) as czi:
        rects = czi.scenes_bounding_rectangle_no_pyramid
        for i, scene in enumerate(img.images):
            sid = scene.index["scene"]
            box = rects[sid] if rects else czi.total_bounding_rectangle_no_pyramid
            ref = czi.read(roi=(box.x, box.y, box.w, box.h), plane={"T": 0, "C": 0, "Z": 0},
                           scene=sid if rects else None)
            ref = ref[..., 0] if ref.shape[-1] == 1 else ref[..., ::-1]
            np.testing.assert_array_equal(scene[0, 0, 0], ref)


def test_tiles_are_returned_one_by_one(tmp_path):
    """Each tile on its own is exactly what was written for it -- including
    the strips that later, overlapping tiles hide in the composite."""
    tiles_written = _mosaic(overlap=10)
    path = str(tmp_path / "m.czi")
    write_czi(path, tiles_written)
    written = {(p["C"], xy): data for _, p, xy, data in tiles_written}
    with micro_reader.open(path) as img, micro_reader.open(path, tiles="mosaic") as st:
        tiles = img.select(scene=0)
        assert len(tiles) == 6 and img.indices == [{"scene": 0, "tile": m} for m in range(6)]
        composite = st.images[0]
        hidden = 0
        for tile in tiles:
            for c in range(tile.shape[1]):
                own = written[(c, tile.origin)]
                np.testing.assert_array_equal(tile[0, c, 0], own)
                ox, oy = composite.origin
                y, x = tile.origin[1] - oy, tile.origin[0] - ox
                in_composite = composite[0, c, 0, y:y + 40, x:x + 56]
                hidden += not np.array_equal(in_composite, own)
        assert hidden > 0            # overlaps really covered parts of earlier tiles


@pytest.mark.parametrize("name", [n for n in CASES if n.startswith("zstd")])
def test_compressed_files_via_the_reference_decoder(tmp_path, name, monkeypatch):
    """Without the native module (monkeypatched away) the same files decode
    to the same pixels through the reference path."""
    from micro_reader import _codecs
    pytest.importorskip("zstandard")
    monkeypatch.setattr(_codecs, "_native", None)
    path, expected, _ = _write(tmp_path, name)
    with micro_reader.open(path, tiles="mosaic") as img:
        assert {s.compression for s in img._level0} <= {5, 6}
        for scene, sid in zip(img.images, sorted(expected)):
            np.testing.assert_array_equal(scene[...], expected[sid])


def test_compressed_tiles_are_decoded_once(tmp_path):
    path, _, _ = _write(tmp_path, "zstd1_mosaic")
    with micro_reader.open(path, tiles="mosaic") as img:
        scene = img.images[0]
        assert {s.compression for s in img._level0} == {6}
        scene[0, 0, 0, 0:10, 0:10]
        first = scene.stats.pixels_read
        scene[0, 0, 0, 5:15, 5:15]                       # same tile again: cached
        assert scene.stats.pixels_read == first == 40 * 56


def test_exact_reads_count_only_what_is_asked(tmp_path, monkeypatch):
    """With separate reads per row forced, only the rectangle is read."""
    from micro_reader.backends import czi
    monkeypatch.setattr(czi, "SEPARATE_READ_BYTES", 0)
    path, expected, _ = _write(tmp_path, "stack")
    with micro_reader.open(path, tiles="mosaic") as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[1, 0, 2, 5:15, 10:30], expected[0][1, 0, 2, 5:15, 10:30])
        assert scene.stats.pixels_read == 10 * 20


def test_narrow_windows_are_one_contiguous_read(tmp_path):
    """A window narrower than its tile is one read from its first pixel to
    its last (one read per row costs far more than the bytes skipped), and
    what that read spans is counted."""
    path, expected, _ = _write(tmp_path, "stack")
    with micro_reader.open(path, tiles="mosaic") as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[1, 0, 2, 5:15, 10:30], expected[0][1, 0, 2, 5:15, 10:30])
        assert scene.stats.pixels_read == 9 * 56 + 20


def _scattered(n=25, seed=3, scenes=1):
    """Tiles of random sizes at random places, heavily overlapping, with a
    tile written later that hides an earlier one entirely."""
    rng = np.random.default_rng(seed)
    tiles = []
    for k in range(n):
        h, w = int(rng.integers(8, 50)), int(rng.integers(8, 60))
        x, y = int(rng.integers(0, 120)), int(rng.integers(0, 90))
        tiles.append((k % scenes, {"T": 0, "C": 0, "Z": 0}, (x, y), _plane(h, w)))
    tiles.append((0, {"T": 0, "C": 0, "Z": 0}, (30, 20), _plane(10, 12)))
    tiles.append((0, {"T": 0, "C": 0, "Z": 0}, (25, 15), _plane(30, 30)))   # hides it
    return tiles


@pytest.mark.parametrize("compression,separate", [(None, 32 << 10), (None, 0),
                                                  ("zstd1:ExplicitLevel=1", 32 << 10)],
                         ids=["raw", "raw_separate_reads", "zstd1"])
def test_overlapping_tiles_compose_exactly(tmp_path, compression, separate, decode_threads,
                                           monkeypatch):
    from micro_reader.backends import czi as backend
    monkeypatch.setattr(backend, "SEPARATE_READ_BYTES", separate)
    path = str(tmp_path / "scattered.czi")
    expected, _ = write_czi(path, _scattered(), compression=compression)
    with micro_reader.open(path, tiles="mosaic") as img, pyczi.open_czi(path) as czi:
        scene, ref = img.images[0], expected[0]
        for key in _keys(ref.shape, n=40):
            np.testing.assert_array_equal(scene[key], ref[key], err_msg=str(key))
        box = czi.total_bounding_rectangle_no_pyramid
        theirs = czi.read(roi=(box.x, box.y, box.w, box.h), plane={"T": 0, "C": 0, "Z": 0})
        np.testing.assert_array_equal(scene[0, 0, 0], theirs[..., 0])


def test_overlaps_are_read_once(tmp_path, monkeypatch):
    """Uncompressed tiles: only the part no later tile covers is read, so
    with separate reads forced a region costs exactly its covered pixels,
    however much the tiles overlap."""
    from micro_reader.backends import czi as backend
    monkeypatch.setattr(backend, "SEPARATE_READ_BYTES", 0)
    path = str(tmp_path / "scattered.czi")
    expected, _ = write_czi(path, _scattered())
    covered = np.zeros(expected[0].shape[-2:], bool)
    with micro_reader.open(path, tiles="mosaic") as img:
        scene = img.images[0]
        ox, oy = scene.origin
        for s in img._level0:
            covered[s.y - oy:s.y - oy + s.height, s.x - ox:s.x - ox + s.width] = True
        for key in _keys(covered.shape, n=30, seed=9):
            key = tuple(slice(k, k + 1) if isinstance(k, int) else k for k in key)
            scene.stats.reset()
            scene[(0, 0, 0) + key]
            assert scene.stats.pixels_read == covered[key].sum(), key


def test_hidden_compressed_tile_is_not_decoded(tmp_path):
    path = str(tmp_path / "hidden.czi")
    plane = {"T": 0, "C": 0, "Z": 0}
    expected, _ = write_czi(path, [(0, plane, (10, 10), _plane(20, 20)),
                                   (0, plane, (0, 0), _plane(40, 56))],
                            compression="zstd1:ExplicitLevel=1")
    with micro_reader.open(path, tiles="mosaic") as img:
        scene = img.images[0]
        np.testing.assert_array_equal(scene[...], expected[0])
        assert scene.stats.pixels_read == 40 * 56          # the hidden tile never decoded


def test_spatial_index_finds_tiles_on_a_large_grid(tmp_path):
    """A 30 x 30 grid of small tiles: region reads touch only their tiles."""
    tiles = [(0, {"T": 0, "C": 0, "Z": 0}, (i * 8, j * 6), _plane(6, 8))
             for j in range(30) for i in range(30)]
    path = str(tmp_path / "grid.czi")
    expected, _ = write_czi(path, tiles)
    with micro_reader.open(path, tiles="mosaic") as img:
        scene = img.images[0]
        tiles_index = scene._planes[(0, 0, 0)]
        assert len(tiles_index.query((60, 66, 80, 88))) == 1
        assert len(tiles_index.query((59, 67, 79, 89))) == 9
        for key in _keys(expected[0].shape, n=20):
            np.testing.assert_array_equal(scene[key], expected[0][key])


@pytest.mark.parametrize("name", ["scenes", "scene_mosaics", "zstd1_hilo_scenes", "stack"])
def test_canvas_matches_pylibczirw(tmp_path, name):
    """The whole file on one canvas, as pylibCZIrw reads it without a scene."""
    path, _, _ = _write(tmp_path, name)
    with micro_reader.open(path, tiles="mosaic") as img, pyczi.open_czi(path) as czi:
        canvas = img.canvas()
        box = czi.total_bounding_rectangle_no_pyramid
        assert canvas.origin == (box.x, box.y)
        assert canvas.shape[-2:] == (box.h, box.w)
        for c in range(canvas.shape[1]):
            ref = czi.read(roi=(box.x, box.y, box.w, box.h), plane={"T": 0, "C": c, "Z": 0})
            np.testing.assert_array_equal(canvas[0, c, 0], ref[..., 0])


def test_canvas_of_overlapping_scenes_matches_pylibczirw(tmp_path):
    path = str(tmp_path / "overlap_scenes.czi")
    write_czi(path, _scattered(n=30, scenes=3))
    with micro_reader.open(path, tiles="mosaic") as img, pyczi.open_czi(path) as czi:
        box = czi.total_bounding_rectangle_no_pyramid
        ref = czi.read(roi=(box.x, box.y, box.w, box.h), plane={"T": 0, "C": 0, "Z": 0})
        np.testing.assert_array_equal(img.canvas()[0, 0, 0], ref[..., 0])


def test_canvas_draws_scene_then_mosaic_then_directory_order(tmp_path):
    """libCZI stacks tiles by scene, then mosaic index, then directory order.
    pylibCZIrw writes its directory grouped by scene, which hides the rule,
    so this file interleaves scenes and mosaic indices in the directory."""
    from _czi_writer import raw_payload, write_czi as write_raw
    rng = np.random.default_rng(4)
    tiles = []
    for scene, mosaic in [(2, 0), (0, 3), (1, 1), (0, 0), (2, 2), (1, 0), (0, 1)]:
        tiles.append(dict(scene=scene, mosaic=mosaic, plane={"C": 0}, dtype="u2",
                          x=int(rng.integers(0, 8)), y=int(rng.integers(0, 8)),
                          height=30, width=30, compression=0,
                          payload=raw_payload(_plane(30, 30))))
    path = str(tmp_path / "interleaved.czi")
    write_raw(path, tiles)
    with micro_reader.open(path, tiles="mosaic") as img, pyczi.open_czi(path) as czi:
        box = czi.total_bounding_rectangle_no_pyramid
        ref = czi.read(roi=(box.x, box.y, box.w, box.h), plane={"C": 0})
        np.testing.assert_array_equal(img.canvas()[0, 0, 0], ref[..., 0])


@pytest.mark.parametrize("name", ["mosaic_overlap", "scene_mosaics", "rgb", "zstd1_mosaic"])
def test_read_region_in_file_coordinates(tmp_path, name):
    """Regions in the file's own coordinates, partly or wholly outside the
    scene box, give what pylibCZIrw gives for the same roi and scene."""
    path, _, _ = _write(tmp_path, name)
    rng = np.random.default_rng(1)
    with micro_reader.open(path, tiles="mosaic") as img, pyczi.open_czi(path) as czi:
        for i, scene in enumerate(img.images):
            sid = scene.index["scene"]
            ox, oy = scene.origin
            ny, nx = scene.shape[3], scene.shape[4]
            rois = [(ox - 7, oy - 5, nx + 20, ny + 9), (ox + nx + 3, oy, 10, 10),
                    (ox, oy, nx, ny), (ox + 3, oy + 2, 0, 5)]
            for _ in range(10):
                rois.append((ox + int(rng.integers(-30, nx)), oy + int(rng.integers(-30, ny)),
                             int(rng.integers(1, 60)), int(rng.integers(1, 60))))
            for x, y, w, h in rois:
                mine = scene.read_region(x, y, w, h, t=0, c=0, z=0)
                if w == 0:
                    assert mine.shape[:2] == (h, 0)
                    continue
                ref = czi.read(roi=(x, y, w, h), plane={"T": 0, "C": 0, "Z": 0}, scene=sid)
                ref = ref[..., 0] if ref.shape[-1] == 1 else ref[..., ::-1]
                np.testing.assert_array_equal(mine, ref, err_msg=f"{name} {(x, y, w, h)}")
                if scene.axes.endswith("s"):
                    np.testing.assert_array_equal(
                        scene.read_region(x, y, w, h, t=0, c=0, z=0, s=1), ref[..., 1])


# -- damage and recovery --------------------------------------------------------------

def _insert_bytes(path, at, count):
    """Insert *count* junk bytes at *at* (as happened to the CD7 plate) and keep
    the file header's own pointers right: only the subblock directory is off."""
    with open(path, "rb") as fh:
        data = bytearray(fh.read())
    data[at:at] = bytes(range(256)) * (count // 256) + bytes(count % 256)
    for field in (84, 92, 104):                    # directory, metadata, attachments
        value, = struct.unpack_from("<q", data, field)
        if value > at:
            struct.pack_into("<q", data, field, value + count)
    with open(path, "wb") as fh:
        fh.write(data)


def test_misplaced_subblocks_are_recovered_loudly(tmp_path, decode_threads):
    path, expected, _ = _write(tmp_path, "scenes")
    with open(path, "rb") as fh:
        entries = _directory(fh)
    first_of_scene1 = min(e.position for e in entries if e.scene == 1)
    _insert_bytes(path, first_of_scene1, 4096 + 13)
    with pytest.warns(DamagedFileWarning, match=r"DAMAGED CZI FILE.*recovered the real position"):
        img = micro_reader.open(path, tiles="mosaic")
    with img:
        assert img.damage["recovered"] == img.damage["misplaced"] > 0
        assert img.damage["shifts"] == [4096 + 13] and img.damage["unrecoverable"] == 0
        assert img.damage["scenes_affected"] == [1]
        for scene, sid in zip(img.images, sorted(expected)):
            np.testing.assert_array_equal(scene[...], expected[sid])


def test_unrecoverable_subblock_is_named_and_refused(tmp_path):
    path, expected, _ = _write(tmp_path, "scenes")
    with open(path, "rb") as fh:
        entries = _directory(fh)
    victim = max((e for e in entries if e.scene == 1), key=lambda e: e.position)
    with open(path, "r+b") as fh:                  # destroy that subblock's header
        fh.seek(victim.position)
        fh.write(b"\x00" * 64)
    with pytest.warns(DamagedFileWarning, match=r"could NOT be found \(scenes \[1\]\)"):
        img = micro_reader.open(path, tiles="mosaic")
    with img:
        assert img.damage["unrecoverable"] == 1
        np.testing.assert_array_equal(img.images[0][...], expected[0])   # scene 0 intact
        with pytest.raises(CorruptFileError, match="could not be found"):
            img.images[1][...]


def test_sound_file_gives_no_warning(tmp_path):
    path, _, _ = _write(tmp_path, "scenes")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DamagedFileWarning)
        with micro_reader.open(path, tiles="mosaic") as img:
            assert img.damage == {}


def test_close_releases_the_file(tmp_path, decode_threads):
    path, _, _ = _write(tmp_path, "mosaic_overlap")
    img = micro_reader.open(path, tiles="mosaic")
    img.images[0][...]
    img.close()
    os.remove(path)
    with pytest.raises(ValueError):
        img.images[0][0]


def test_concurrent_reads(tmp_path, decode_threads):
    path, expected, _ = _write(tmp_path, "scene_mosaics")
    errors = []
    with micro_reader.open(path, tiles="mosaic") as img:
        def worker(i):
            try:
                for scene, sid in zip(img.images, sorted(expected)):
                    for key in _keys(expected[sid].shape, n=10, seed=i):
                        if not np.array_equal(scene[key], expected[sid][key]):
                            errors.append((sid, key))
            except Exception as exc:        # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []


# -- channels of different pixel types; the slide's label and preview ----------------

from _czi_writer import raw_payload, write_czi as write_raw  # noqa: E402

def _mixed_plane(c, dtype, value_range, scene=0, size=(20, 30), seed=0):
    rng = np.random.default_rng(seed + c)
    a = rng.integers(*value_range, size).astype(dtype)
    return a, dict(scene=scene, plane={"C": c}, x=0, y=0, height=size[0], width=size[1],
                   dtype=dtype, compression=0, payload=raw_payload(a))


def test_channels_of_different_pixel_types(tmp_path):
    """gray8 + gray16 + gray8 (as pylibCZIrw's c2_gray8_gray16.czi, one more
    channel): one image in uint16 (values unchanged) whose parts are exact."""
    a0, t0 = _mixed_plane(0, "u1", (0, 255))
    a1, t1 = _mixed_plane(1, "u2", (0, 60000))
    a2, t2 = _mixed_plane(2, "u1", (0, 255))
    meta = ("<Information><Image><Dimensions><Channels>"
            '<Channel Name="A"/><Channel Name="B"/><Channel Name="C"/>'
            "</Channels></Dimensions></Image></Information>")
    path = str(tmp_path / "mixed.czi")
    write_raw(path, [t0, t1, t2], meta)
    with micro_reader.open(path) as f:
        image = f.images[0]
        assert image.dtype == np.uint16 and image.shape == (1, 3, 1, 20, 30)
        np.testing.assert_array_equal(image.read(t=0, z=0), np.stack([a0, a1, a2]))
        np.testing.assert_array_equal(image[0, 1:3, 0, 5:9, 7:20], np.stack([a1, a2])[:, 5:9, 7:20])
        assert image.is_ragged
        parts = image.parts
        assert [(p.dtype, p.shape[1], p.index["part"]) for p in parts] == \
            [(np.uint8, 2, 0), (np.uint16, 1, 1)]
        np.testing.assert_array_equal(parts[0].read(t=0, z=0), np.stack([a0, a2]))
        assert all(p.whole is image for p in parts)
        assert [c.name for c in image.metadata.channels] == ["A", "B", "C"]
        assert [c.name for c in parts[0].metadata.channels] == ["A", "C"]
        assert "parts: (1, 2, 1, 20, 30) uint8 + (1, 1, 1, 20, 30) uint16" in f.outline()


def test_grey_and_colour_channels_do_not_mix(tmp_path):
    a0, t0 = _mixed_plane(0, "u1", (0, 255))
    rgb = np.zeros((20, 30, 3), np.uint8)
    t1 = dict(scene=0, plane={"C": 1}, x=0, y=0, height=20, width=30, dtype="u1", samples=3,
              compression=0, payload=raw_payload(rgb))
    path = str(tmp_path / "mix.czi")
    write_raw(path, [t0, t1])
    with pytest.raises(micro_reader.UnsupportedFormatError, match="grey and colour"):
        micro_reader.open(path)


def test_label_and_slide_preview_are_scenes(tmp_path):
    """A slide scan's Label and SlidePreview attachments (each a CZI of its own)
    are further scenes; the Thumbnail (a JPEG copy of the image) is not."""
    label, tl = _mixed_plane(0, "u1", (0, 255), size=(12, 16), seed=4)
    preview, tp = _mixed_plane(0, "u2", (0, 9000), size=(10, 14), seed=7)
    write_raw(str(tmp_path / "label.czi"), [tl])
    write_raw(str(tmp_path / "preview.czi"), [tp])
    image, t0 = _mixed_plane(0, "u2", (0, 4000))
    attachments = [("Thumbnail", "JPG", b"\xff\xd8\xff\xe0 not decoded"),
                   ("SlidePreview", "CZI", (tmp_path / "preview.czi").read_bytes()),
                   ("Label", "CZI", (tmp_path / "label.czi").read_bytes())]
    path = str(tmp_path / "slide.czi")
    write_raw(path, [t0], attachments=attachments)
    with micro_reader.open(path) as f:
        assert [im.index for im in f.images] == [{"scene": 0}, {"scene": 1}, {"scene": 2}]
        assert [im.name.split(" ", 1)[1] for im in f.images[1:]] == ["label", "slide preview"]
        np.testing.assert_array_equal(f.images[0].read(t=0, c=0, z=0), image)
        np.testing.assert_array_equal(f.images[1].read(t=0, c=0, z=0), label)
        np.testing.assert_array_equal(f.images[2][0, 0, 0, 2:8, 3:11], preview[2:8, 3:11])
    f.images[1]                                       # closed with the file: reads raise
    with pytest.raises(ValueError):
        f.images[1].read()


def test_a_compressed_label_reads_from_its_own_offsets(tmp_path):
    """A label whose subblock is compressed is read through the embedded file's
    handles (``_ShiftedHandles.read``), which once added the attachment's offset
    twice and read past it (CorruptFileError) -- raw subblocks took another path."""
    imagecodecs = pytest.importorskip("imagecodecs")
    label = np.random.default_rng(5).integers(0, 255, (12, 16)).astype("u1")
    write_raw(str(tmp_path / "label.czi"),
              [dict(scene=0, plane={"C": 0}, x=0, y=0, height=12, width=16, dtype="u1",
                    compression=5, payload=imagecodecs.zstd_encode(raw_payload(label)))])
    image, t0 = _mixed_plane(0, "u2", (0, 4000))
    path = str(tmp_path / "slide.czi")
    write_raw(path, [t0], attachments=[("Label", "CZI", (tmp_path / "label.czi").read_bytes())])
    with micro_reader.open(path) as f:
        np.testing.assert_array_equal(f.images[1].read(t=0, c=0, z=0), label)


def test_stitched_is_the_deprecated_name_of_mosaic(tmp_path):
    path = str(tmp_path / "m.czi")
    write_czi(path, _mosaic(overlap=10))
    with micro_reader.open(path, tiles="mosaic") as new:
        expected = new.images[0][...]
    with pytest.warns(DeprecationWarning, match="tiles='mosaic'"):
        old = micro_reader.open(path, tiles="stitched")
    with old:
        assert len(old.images) == 1
        np.testing.assert_array_equal(old.images[0][...], expected)
