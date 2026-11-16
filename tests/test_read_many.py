"""``micro_reader.read_many``: many images' regions in one go (Workplan MR23,
joins), against one ``image[key]`` per image, on the engine and on the Python
path."""
from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

import micro_reader  # noqa: E402
from micro_reader import _engine  # noqa: E402

from test_tiff import HAVE_CODECS, _data  # noqa: E402

# name -> (plane shape, dtype, imwrite kwargs, needs imagecodecs to write)
STACKS = {
    "zstd_tiles": ((300, 500), "u2", {"compression": "zstd", "tile": (64, 64)}, True),
    "zlib_strips": ((300, 500), "u2", {"compression": "zlib", "rowsperstrip": 37}, False),
    "raw_strips": ((300, 500), "u2", {"rowsperstrip": 50}, False),
    "rgb_tiles": ((200, 300, 3), "u1", {"photometric": "rgb", "compression": "zlib",
                                       "tile": (64, 64)}, False),
}


@pytest.fixture(scope="module")
def stacks(tmp_path_factory):
    """name -> (paths of 6 one-plane files, their planes)."""
    root = tmp_path_factory.mktemp("many")
    out = {}
    for name, (shape, dtype, kwargs, needs) in STACKS.items():
        if needs and not HAVE_CODECS:
            continue
        paths, planes = [], []
        for z in range(6):
            data = _data(shape, dtype, seed=z)
            path = str(root / f"{name}_z{z:02d}.tif")
            tifffile.imwrite(path, data, **kwargs)
            paths.append(path)
            planes.append(data)
        out[name] = (paths, planes)
    return out


@pytest.fixture(params=["engine", "python"])
def path_kind(request):
    if request.param == "engine" and not _engine.enabled():
        pytest.skip("micro-reader-core is not installed")
    _engine.set_engine(request.param == "engine")
    yield request.param
    _engine.set_engine(None)


@pytest.fixture
def engine_calls(monkeypatch):
    seen = {"calls": 0}
    run = _engine.run_plans

    def counting(plans, out, counts=None):
        seen["calls"] += 1
        return run(plans, out, counts)
    monkeypatch.setattr(_engine, "run_plans", counting)
    return seen


def _open_all(paths, **kwargs):
    return [micro_reader.open(p, **kwargs) for p in paths]


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("name", sorted(STACKS))
def test_a_region_across_files_reads_as_one_by_one(stacks, name, threads, path_kind,
                                                   engine_calls):
    if name not in stacks:
        pytest.skip("needs imagecodecs to write")
    paths, planes = stacks[name]
    old = micro_reader.decode_threads()
    micro_reader.set_decode_threads(threads)
    files = _open_all(paths, axes="canonical", samples="channels")
    try:
        images = [f.images[0] for f in files]
        shape = images[0].shape                      # t c z y x
        for key in [(0, slice(None), 0, slice(None), slice(None)),
                    (0, slice(None), 0, slice(17, 200), slice(33, 290)),
                    (0, slice(0, 1), 0, slice(64, 128), slice(64, 128))]:
            want = np.stack([image[key] for image in images])
            region = np.zeros_like(want)
            engine_calls["calls"] = 0
            micro_reader.read_many([(image, key, region[k]) for k, image in enumerate(images)])
            np.testing.assert_array_equal(region, want, err_msg=f"{name} {key}")
            if not (name == "rgb_tiles" and key[1] == slice(0, 1)):   # one sample of 3:
                assert engine_calls["calls"] == 1    # read one by one; else one call
        # the planes as written (RGB samples folded into channels: c = 3)
        full = np.zeros((len(images),) + shape[1:2] + shape[3:], images[0].dtype)
        micro_reader.read_many([(image, (0, slice(None), 0), full[k])
                                for k, image in enumerate(images)])
        for k, plane in enumerate(planes):
            want = np.moveaxis(plane, -1, 0) if plane.ndim == 3 else plane[None]
            np.testing.assert_array_equal(full[k], want)
    finally:
        for f in files:
            f.close()
        micro_reader.set_decode_threads(old)


def test_stats_count_as_reading_one_by_one(stacks, path_kind):
    paths, _ = stacks["zlib_strips"]
    key = (0, 0, 0, slice(10, 120), slice(5, 400))
    counts = []
    for many in (False, True):
        files = _open_all(paths, axes="canonical")
        try:
            images = [f.images[0] for f in files]
            if many:
                region = np.empty((len(images), 110, 395), images[0].dtype)
                micro_reader.read_many([(im, key, region[k]) for k, im in enumerate(images)])
            else:
                for im in images:
                    im[key]
            counts.append([(im.stats.reads, im.stats.pixels_returned, im.stats.pixels_read)
                           for im in images])
        finally:
            for f in files:
                f.close()
    assert counts[0] == counts[1]
    assert all(read > 0 for _, _, read in counts[1])


def test_keys_that_cannot_be_planned_are_read_one_by_one(stacks, path_kind):
    """A stepped key, an empty one, and an image that cannot plan into an array:
    read one by one, with the same result."""
    paths, _ = stacks["zlib_strips"]
    files = _open_all(paths, axes="canonical")
    try:
        a, b, c = (f.images[0] for f in files[:3])
        stepped = (0, 0, 0, slice(0, 300, 7), slice(1, 500, 3))
        empty = (0, 0, 0, slice(5, 5), slice(None))
        out_a = np.empty(a[stepped].shape, a.dtype)
        out_b = np.empty(b[empty].shape, b.dtype)
        out_c = np.empty(c[...].shape, c.dtype)

        class NoPlans:                              # any image whose backend cannot plan
            def __init__(self, image):
                self._image = image
                self.shape, self.dtype, self.name = image.shape, image.dtype, image.name
                self.stats = image.stats

            def _plan_into(self, block, out, stack):
                return None

            def __getitem__(self, key):
                return self._image[key]
        micro_reader.read_many([(a, stepped, out_a), (b, empty, out_b),
                                (NoPlans(c), Ellipsis, out_c)])
        np.testing.assert_array_equal(out_a, a[stepped])
        assert out_b.shape == (0, 500)
        np.testing.assert_array_equal(out_c, c[...])
    finally:
        for f in files:
            f.close()


def test_arrays_of_different_buffers(stacks, path_kind):
    """Requests whose arrays are not views of one array: each buffer is run apart."""
    paths, planes = stacks["raw_strips"]
    files = _open_all(paths[:3], axes="canonical")
    try:
        outs = [np.empty((300, 500), "u2") for _ in files]
        micro_reader.read_many([(f.images[0], (0, 0, 0), out) for f, out in zip(files, outs)])
        for out, plane in zip(outs, planes):
            np.testing.assert_array_equal(out, plane)
    finally:
        for f in files:
            f.close()


def test_folded_channels_not_on_a_pixel_boundary(stacks, path_kind):
    """Channels 1:2 of an RGB image folded into channels: not plannable as a
    view, read and copied."""
    paths, planes = stacks["rgb_tiles"]
    with micro_reader.open(paths[0], axes="canonical", samples="channels") as f:
        image = f.images[0]
        out = np.empty((1, 200, 300), image.dtype)
        micro_reader.read_many([(image, (0, slice(1, 2), 0), out)])
        np.testing.assert_array_equal(out[0], planes[0][..., 1])


def test_refused_outputs(stacks):
    paths, _ = stacks["raw_strips"]
    with micro_reader.open(paths[0], axes="canonical") as f:
        image = f.images[0]
        key = (0, 0, 0)
        with pytest.raises(ValueError, match="shape"):
            micro_reader.read_many([(image, key, np.empty((300, 499), "u2"))])
        with pytest.raises(ValueError, match="dtype"):
            micro_reader.read_many([(image, key, np.empty((300, 500), "u4"))])
        ro = np.empty((300, 500), "u2")
        ro.setflags(write=False)
        with pytest.raises(ValueError, match="read-only"):
            micro_reader.read_many([(image, key, ro)])
        with pytest.raises(TypeError, match="NumPy array"):
            micro_reader.read_many([(image, key, bytearray(300 * 500 * 2))])
    with pytest.raises(ValueError, match="closed"):
        micro_reader.read_many([(image, key, np.empty((300, 500), "u2"))])


def test_a_dyna_join_of_images_reads_in_one_call(stacks, engine_calls):
    """dyna's concatenation finds ``Image.read_many`` on its sources' type."""
    dyna = pytest.importorskip("dyna_zarr")
    from dyna_zarr.operations import structural
    if not hasattr(structural, "_read_many"):
        pytest.skip("this dyna-zarr has no read_many hook")
    paths, planes = stacks["zlib_strips"]
    files = _open_all(paths, axes="canonical")
    try:
        assert all(type(f.images[0]).read_many is micro_reader.Image.read_many for f in files)
        joined = dyna.operations.concatenate([dyna.from_array(f.images[0]) for f in files],
                                             axis=2)
        engine_calls["calls"] = 0
        got = np.asarray(joined[0, 0, 1:5, 20:200, 10:300])
        np.testing.assert_array_equal(got, np.stack(planes)[1:5, 20:200, 10:300])
        assert engine_calls["calls"] == 1
    finally:
        for f in files:
            f.close()
