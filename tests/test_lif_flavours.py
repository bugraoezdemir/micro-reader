"""The other Leica flavours -- LOF, LIFEXT, XLIF (frames in PNG, TIFF,
multi-page TIFF, LOF, BMP, JPEG files), XLEF, XLCF -- on synthetic files
(``_lif_writer``), against the written arrays and against liffile
(Gohlke, BSD-3; the reference for these flavours)."""
from __future__ import annotations

import os
import threading

import numpy as np
import pytest

pytest.importorskip("imagecodecs")             # writes the frame files
pytest.importorskip("tifffile")

import micro_reader  # noqa: E402
from micro_reader import CorruptFileError  # noqa: E402
from micro_reader.backends import lif  # noqa: E402

from _lif_writer import image, write_lifext, write_lof, write_xlef, write_xlif  # noqa: E402


def _native(*args, **kwargs):
    """``micro_reader.open`` with RGB samples kept as an ``s`` axis: these tests
    check the backend's own layout (open's default folds them into channels)."""
    return micro_reader.open(*args, **{"samples": "keep", **kwargs})

RNG = np.random.default_rng(7)


def _grey(dtype="u2", shape=(2, 3, 20, 30)):                          # c z y x
    top = 255 if np.dtype(dtype).itemsize == 1 else 4000
    return RNG.integers(0, top, shape).astype(dtype)


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """exp.xlef -> a.xlif (PNG frames), b.xlif (TIFF frames, UTF-16 XML),
    sub/coll.xlcf -> d (multi-page TIFF), e (LOF), f (RGB BMP), g (JPEG)."""
    root = tmp_path_factory.mktemp("flavours")
    os.makedirs(root / "sub")
    truth = {
        "a": _grey(), "b": _grey("u1"), "d": _grey(), "e": _grey(),
        "f": RNG.integers(0, 255, (2, 20, 30, 3)).astype("u1"),       # z y x s
        "g": _grey("u1"),
    }
    write_xlif(str(root / "a.xlif"), image("a", truth["a"], "czyx"), frames="png")
    write_xlif(str(root / "b.xlif"), image("b", truth["b"], "czyx"), frames="tif",
               encoding="utf-16")
    write_xlif(str(root / "sub" / "d.xlif"), image("d", truth["d"], "czyx"), frames="ometif")
    write_xlif(str(root / "sub" / "e.xlif"), image("e", truth["e"], "czyx"), frames="lof")
    # LAS X keeps the LIF channel tags (B, G, R) for frame files, which decode R, G, B
    write_xlif(str(root / "sub" / "f.xlif"), image("f", truth["f"], "zyxs", tags=[3, 2, 1]),
               frames="bmp")
    write_xlif(str(root / "sub" / "g.xlif"), image("g", truth["g"], "czyx"), frames="jpg")
    write_xlef(str(root / "sub" / "coll.xlcf"), ["d.xlif", "e.xlif", "f.xlif", "g.xlif"],
               name="coll", collection=True)
    write_xlef(str(root / "exp.xlef"), ["a.xlif", "b.xlif", "sub\\coll.xlcf"])
    return root, truth


def _expected(truth, key, series):
    """The written array in the series's axes."""
    want = truth[key]
    return want[None] if series.axes == "tczyx" else want[None, None]   # rgb: t c z y x s


def _keys(shape, n=12, seed=2):
    rng = np.random.default_rng(seed)
    keys = [(Ellipsis,)]
    for _ in range(n):
        key = []
        for size in shape:
            a = int(rng.integers(0, size))
            key.append(slice(a, int(rng.integers(a + 1, size + 1))))
        keys.append(tuple(key))
    return keys


@pytest.fixture(params=[1, 4], ids=["serial", "pool"])
def decode_threads(request):
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(request.param)
    yield
    micro_reader.set_decode_threads(old)


def test_experiment_tree(tree, decode_threads):
    root, truth = tree
    with _native(str(root / "exp.xlef")) as img:
        assert img.backend == "lif" and img.format == "xlef"
        assert [s.name for s in img.images] == ["a", "b", "coll/d", "coll/e", "coll/f", "coll/g"]
        for series in img.images:
            want = _expected(truth, series.name[-1], series)
            assert series.shape == want.shape and series.dtype == want.dtype
            for key in _keys(want.shape):
                np.testing.assert_array_equal(series[key], want[key], err_msg=f"{series.name} {key}")


@pytest.mark.parametrize("name,flavour,names", [
    ("sub/coll.xlcf", "XLCF", ["coll/d", "coll/e", "coll/f", "coll/g"]),
    ("sub/f.xlif", "XLIF", ["f"]),
])
def test_opening_a_member(tree, name, flavour, names):
    root, truth = tree
    with _native(str(root / name)) as img:
        assert img.format == flavour.lower()
        assert [s.name for s in img.images] == names
        for series in img.images:
            np.testing.assert_array_equal(series[...], _expected(truth, series.name[-1], series))


def test_agrees_with_liffile(tree):
    liffile = pytest.importorskip("liffile")
    root, _ = tree
    path = str(root / "exp.xlef")
    with _native(path) as img, liffile.LifFile(path) as lf:
        assert [s.name for s in img.images] == [i.path for i in lf.images]
        for series, ref_image in zip(img.images, lf.images):
            ref = np.asarray(ref_image.asarray())
            dims = [d.lower() for d in ref_image.dims]
            ref = ref.transpose([dims.index(a) for a in series.axes if a in dims])
            np.testing.assert_array_equal(series[...].squeeze(), ref.squeeze(), err_msg=series.name)


@pytest.mark.parametrize("legacy", [False, True], ids=["lof", "legacy_lof"])
def test_lof(tmp_path, legacy, decode_threads):
    data = _grey()
    path = str(tmp_path / "one.lof")
    write_lof(path, image("one", data, "czyx"), legacy=legacy)
    with _native(path) as img:
        assert img.format == "lof" and [s.name for s in img.images] == ["one"]
        for key in _keys((1,) + data.shape):
            np.testing.assert_array_equal(img.images[0][key], data[None][key])


def test_lifext(tmp_path):
    a, b = _grey(), _grey("u1")
    path = str(tmp_path / "x.lifext")
    write_lifext(path, [("MemBlock_7", [image("x1", a, "czyx"), image("x2", b, "czyx")])])
    with _native(path) as img:
        assert img.format == "lifext"
        assert [s.name for s in img.images] == ["MemBlock_7/x1", "MemBlock_7/x2"]
        np.testing.assert_array_equal(img.images[0][...], a[None])
        np.testing.assert_array_equal(img.images[1][...], b[None])


def test_rgb_frame_for_a_grey_image_gives_its_first_sample(tmp_path):
    """A frame file decoded as RGB where the block holds grey (as liffile
    reads it): its first sample."""
    import imagecodecs
    data = _grey("u1", (1, 2, 20, 30))
    path = str(tmp_path / "g.xlif")
    write_xlif(path, image("g", data, "czyx"), frames="png")
    plane = tmp_path / "g_frames" / "plane_001.png"
    with open(plane, "wb") as fh:                              # replace plane 1 by RGB
        fh.write(imagecodecs.png_encode(np.stack([data[0, 1]] * 3, axis=-1)))
    with _native(path) as img:
        np.testing.assert_array_equal(img.images[0][...], data[None])


def test_float16_in_tiff_frames(tmp_path):
    """LAS X FLIM results in TIFF frames: half floats, read as such."""
    data = RNG.normal(size=(1, 2, 16, 12)).astype("f2")
    path = str(tmp_path / "flim.xlif")
    write_xlif(path, image("Fast Flim", data, "czyx"), frames="tif")
    with _native(path) as img:
        assert img.images[0].dtype == np.float16
        np.testing.assert_array_equal(img.images[0][...], data[None])


def test_missing_frame_file_is_named(tmp_path):
    path = str(tmp_path / "a.xlif")
    write_xlif(path, image("a", _grey(), "czyx"), frames="png")
    os.remove(tmp_path / "a_frames" / "plane_002.png")
    with pytest.raises(CorruptFileError, match="plane_002.png"):
        _native(path)


def test_reference_cycle_is_refused(tmp_path):
    write_xlef(str(tmp_path / "loop.xlef"), ["loop.xlef"])
    with pytest.raises(CorruptFileError, match="cycle"):
        _native(str(tmp_path / "loop.xlef"))


def test_sniff_claims_only_leica_xml():
    assert lif.sniff('<?xml version="1.0"?><LMSDataContainerHeader Version="2">'.encode(), "x")
    assert lif.sniff("<LMSDataContainerHeader>".encode("utf-16"), "x")
    assert not lif.sniff('<?xml version="1.0"?><OME xmlns="...">'.encode(), "x.companion.ome")


def test_close_releases_frame_files(tree, decode_threads):
    root, _ = tree
    img = _native(str(root / "exp.xlef"))
    for series in img.images:
        series[...]
    img.close()
    assert not img._handles and not img._tiffs


def test_concurrent_reads_of_frames(tree, decode_threads):
    root, truth = tree
    errors = []
    with _native(str(root / "exp.xlef")) as img:
        def work(seed):
            for series in img.images:
                want = _expected(truth, series.name[-1], series)
                for key in _keys(want.shape, n=6, seed=seed):
                    if not np.array_equal(series[key], want[key]):
                        errors.append((series.name, key))
        threads = [threading.Thread(target=work, args=(s,)) for s in range(5)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
    assert not errors
