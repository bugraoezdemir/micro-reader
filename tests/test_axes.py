"""Scenes presented with chosen axes: ``Image.as_axes`` and ``open(axes=...)``.

The view logic on an array-backed series (every axis layout, every policy,
checked against numpy), then through real backends (TIFF, MRC, ND2 RGB)."""
from __future__ import annotations

import numpy as np
import pytest

import micro_reader
from micro_reader import Image


class ArrayImage(Image):
    """A series over an in-memory array, counting what it reads."""

    def __init__(self, data, axes, unit=None):
        self.data = np.asarray(data)
        super().__init__(0, axes, self.data.shape, self.data.dtype, name="array",
                         read_unit=unit)

    def _read_block(self, block):
        out = self.data[block]
        return out.copy(), out.size


RNG = np.random.default_rng(0)


def _keys(shape, n=25, seed=4):
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


def _check(view, expected):
    assert view.shape == expected.shape and view.dtype == expected.dtype
    for key in _keys(expected.shape):
        got = view[key]
        if not any(isinstance(k, slice) and k.step not in (None, 1) for k in key):
            assert got.flags.c_contiguous             # steps are applied after reading
        np.testing.assert_array_equal(got, expected[key], err_msg=str(key))


def test_reorders_and_inserts_missing_axes():
    data = RNG.integers(0, 1000, (4, 3, 10, 12))                  # z c y x
    view = ArrayImage(data, "zcyx").as_axes("tczyx")
    assert view.axes == "tczyx" and view.source.axes == "zcyx"
    _check(view, data.transpose(1, 0, 2, 3)[None])


def test_mrc_like_zyx_to_tczyx():
    data = RNG.integers(0, 1000, (5, 9, 11))
    _check(ArrayImage(data, "zyx").as_axes("tczyx"), data[None, None])


def test_samples_kept_trailing():
    data = RNG.integers(0, 255, (2, 8, 9, 3))                      # c y x s
    _check(ArrayImage(data, "cyxs").as_axes("tczyxs"), data[None, :, None])


@pytest.mark.parametrize("axes,make", [
    ("cyxs", lambda d: d),                                          # channels then samples
    ("scyx", lambda d: d.transpose(3, 0, 1, 2)),                     # samples first (planar)
])
def test_samples_folded_into_channels(axes, make):
    data = RNG.integers(0, 255, (2, 8, 9, 3))                      # c y x s
    view = ArrayImage(make(data), axes).as_axes("tczyx", samples="channels")
    # channel k = sample k % 3 of channel k // 3
    expected = data.transpose(0, 3, 1, 2).reshape(6, 8, 9)[None, :, None]
    _check(view, expected)


def test_rgb_without_channels_folds_to_three_channels():
    data = RNG.integers(0, 255, (8, 9, 3))                         # y x s
    view = ArrayImage(data, "yxs").as_axes("tczyx", samples="channels")
    _check(view, data.transpose(2, 0, 1)[None, :, None])


def test_axis_that_does_not_fit_is_named():
    series = ArrayImage(np.zeros((2, 3, 4, 5)), "tzyx")
    with pytest.raises(ValueError, match=r"'t'"):
        series.as_axes("zyx")
    with pytest.raises(ValueError, match="samples='channels'"):
        ArrayImage(np.zeros((4, 5, 3)), "yxs").as_axes("tczyx")


@pytest.mark.parametrize("axes,kwargs,match", [
    ("tczyx", dict(samples="planes"), "samples="),
    ("tqzyx", {}, "not array axes"),                   # arrays have t c z y x (s) only
    ("zzyx", {}, "each once"),
])
def test_bad_requests_are_errors(axes, kwargs, match):
    series = ArrayImage(np.zeros((2, 3, 4, 5)), "czyx")
    with pytest.raises(ValueError, match=match):
        series.as_axes(axes, **kwargs)


def test_as_axes_has_no_select_or_rename():
    """Non-standard dimensions are image indices now, not axes to fix or name."""
    series = ArrayImage(np.zeros((2, 4, 5)), "zyx")
    with pytest.raises(TypeError):
        series.as_axes("tczyx", select={"z": 0})
    with pytest.raises(TypeError):
        series.as_axes("tczyx", rename={"z": "t"})


def test_reads_stay_exact_and_units_follow():
    data = RNG.integers(0, 1000, (4, 3, 10, 12))
    view = ArrayImage(data, "zcyx", unit=(1, 1, 5, 6)).as_axes("tczyx")
    assert view.read_unit == (1, 1, 1, 5, 6)
    view[0, 1, 2, 0:4, 0:5]
    assert view.stats.pixels_read == 4 * 5
    assert view.as_axes("czyx").source is view.source              # views do not stack


# -- through real backends ------------------------------------------------------------

def test_tiff_open_with_axes(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = RNG.integers(0, 4000, (4, 3, 20, 30)).astype("u2")      # Z C Y X
    path = str(tmp_path / "zc.ome.tif")
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "ZCYX"})
    with micro_reader.open(path) as img:
        native = img.images[0]
        assert native.axes == "zcyx"
    with micro_reader.open(path, axes="tczyx") as img:
        view = img.images[0]
        assert view.axes == "tczyx" and view.source.axes == "zcyx"
        _check(view, data.transpose(1, 0, 2, 3)[None])


def test_rgb_tiff_samples_as_channels(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = RNG.integers(0, 255, (20, 30, 3)).astype("u1")
    path = str(tmp_path / "rgb.tif")
    tifffile.imwrite(path, data, photometric="rgb")
    with micro_reader.open(path, axes="tczyx", samples="channels") as img:
        _check(img.images[0], data.transpose(2, 0, 1)[None, :, None])


def test_mrc_open_with_axes(tmp_path):
    from test_mrc import write_mrc
    data = RNG.integers(0, 1000, (3, 16, 20)).astype("i2")
    path = str(tmp_path / "v.mrc")
    write_mrc(path, data)
    with micro_reader.open(path, axes="tczyx") as img:
        view = img.images[0]
        assert view.source.axes == "zyx"
        _check(view, view.source[...][None, None])


def test_open_rename_makes_an_index_an_axis(tmp_path):
    """A plain multi-page TIFF does not say what its pages are: an index
    ``sequence`` (one image per page), or an axis with rename."""
    tifffile = pytest.importorskip("tifffile")
    data = RNG.integers(0, 255, (4, 5, 6)).astype("u1")
    path = str(tmp_path / "pages.tif")
    with tifffile.TiffWriter(path) as tw:
        for page in data:
            tw.write(page, metadata=None)
    with micro_reader.open(path) as f:
        assert f.index_range == {"series": 1, "sequence": 4}
        assert all(image.axes == "yx" for image in f.images)
        np.testing.assert_array_equal(f.select(sequence=2)[0][...], data[2])
    with micro_reader.open(path, rename={"sequence": "z"}) as f:
        (image,) = f.images
        assert image.axes == "zyx" and "sequence" not in image.index
        np.testing.assert_array_equal(image[...], data)


@pytest.mark.parametrize("rename,match", [
    ({"series": "z"}, "not an index that can become an axis"),
    ({"sequence": "y"}, "not 't', 'c' or 'z'"),
    ({"sequence": "z", "phase": "z"}, "two indices onto one axis"),
])
def test_bad_renames_are_errors(tmp_path, rename, match):
    tifffile = pytest.importorskip("tifffile")
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, np.zeros((5, 6), "u1"))
    with pytest.raises(ValueError, match=match):
        micro_reader.open(path, rename=rename)


def test_rename_of_an_index_the_file_lacks_is_harmless(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, np.zeros((5, 6), "u1"))
    with micro_reader.open(path, rename={"sequence": "z"}) as f:
        assert f.images[0].axes == "yx"


def test_policies_without_axes_keep_the_order(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = RNG.integers(0, 255, (20, 30, 3)).astype("u1")
    path = str(tmp_path / "rgb.tif")
    tifffile.imwrite(path, data, photometric="rgb")
    with micro_reader.open(path, samples="channels") as img:
        assert img.images[0].axes == "cyx"
        _check(img.images[0], data.transpose(2, 0, 1))


def test_open_folds_rgb_samples_into_channels_by_default(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = RNG.integers(0, 255, (20, 30, 3)).astype("u1")
    rgb, grey = str(tmp_path / "rgb.tif"), str(tmp_path / "grey.tif")
    tifffile.imwrite(rgb, data, photometric="rgb")
    tifffile.imwrite(grey, data[..., 0])
    with micro_reader.open(rgb) as f:
        (im,) = f.images
        assert (im.axes, im.shape) == ("cyx", (3, 20, 30))
        _check(im, data.transpose(2, 0, 1))
        # one unnamed channel: unnamed folded channels, tinted red / green / blue
        assert [(c.name, c.color) for c in im.metadata.channels] == [
            (None, "#FF0000"), (None, "#00FF00"), (None, "#0000FF")]
        assert im.source.axes == "yxs"
    with micro_reader.open(rgb, samples="keep") as f:
        assert f.images[0].axes == "yxs" and len(f.images[0].metadata.channels) == 1
    with micro_reader.open(grey) as f:                  # no samples: the native image
        assert not hasattr(f.images[0], "source") and f.images[0].axes == "yx"
    with pytest.raises(ValueError, match="samples='rgb'"):
        micro_reader.open(rgb, samples="rgb")


def test_a_folded_image_keeps_its_backend_attributes(tmp_path):
    """A backend's own public attributes (a CZI series' ``origin``) stay readable
    on an RGB image that open's default folds; private ones are not passed on."""
    tifffile = pytest.importorskip("tifffile")
    rgb = str(tmp_path / "rgb.tif")
    tifffile.imwrite(rgb, RNG.integers(0, 255, (20, 30, 3)).astype("u1"), photometric="rgb")
    with micro_reader.open(rgb) as f:
        im = f.images[0]
        im.source.vendor_origin = (5, 7)
        assert im.vendor_origin == (5, 7)
        with pytest.raises(AttributeError):
            im._not_passed_on


def test_folded_channels_are_named_as_eubi_bridge_names_them():
    from micro_reader._axes import folded_channels
    from micro_reader._metadata import Channel
    nm = micro_reader.Quantity(520, "nanometer")
    named = folded_channels((Channel(name="Bright", emission_wavelength=nm),), 1, 3)
    assert [(c.name, c.color) for c in named] == [("Bright R", "#FF0000"),
                                                  ("Bright G", "#00FF00"),
                                                  ("Bright B", "#0000FF")]
    assert {c.emission_wavelength for c in named} == {nm}
    two = folded_channels((Channel(name="a"), Channel()), 2, 3)
    assert [c.name for c in two] == ["a R", "a G", "a B",
                                     "Channel 1 R", "Channel 1 G", "Channel 1 B"]
    listed = tuple(Channel(name=n) for n in "RGB")     # one channel per sample already
    assert folded_channels(listed, 1, 3) == listed
    rgba = folded_channels((Channel(name="x"),), 1, 4)
    assert [c.name for c in rgba][-1] == "x A" and rgba[-1].color == "#FFFFFF"
    pair = folded_channels((Channel(name="x"),), 1, 2)
    assert [(c.name, c.color) for c in pair] == [("x 0", None), ("x 1", None)]


# -- canonical order and squeeze ------------------------------------------------------

def test_canonical_order():
    data = RNG.integers(0, 1000, (3, 4, 9, 11))                    # z c y x
    view = ArrayImage(data, "zcyx").as_axes("canonical")
    assert view.axes == "tczyx"
    _check(view, data.transpose(1, 0, 2, 3)[None])


def test_canonical_keeps_samples_innermost_or_folds_them():
    data = RNG.integers(0, 255, (8, 9, 3))                          # y x s
    assert ArrayImage(data, "yxs").as_axes("canonical").axes == "tczyxs"
    view = ArrayImage(data, "yxs").as_axes("canonical", samples="channels")
    assert view.axes == "tczyx"
    _check(view, data.transpose(2, 0, 1)[None, :, None])


def test_squeeze_drops_size_one_axes_but_not_the_plane():
    data = RNG.integers(0, 1000, (1, 3, 1, 1, 7))                   # t c z y x
    view = ArrayImage(data, "tczyx").squeeze()
    assert view.axes == "cyx" and view.source.axes == "tczyx"
    _check(view, data[0, :, 0])


def test_squeeze_named_axes_only():
    data = RNG.integers(0, 1000, (1, 3, 1, 6, 7))
    series = ArrayImage(data, "tczyx")
    _check(series.squeeze("t"), data[0])
    with pytest.raises(ValueError, match="has 3, not 1"):
        series.squeeze("c")


def test_squeeze_a_view_rebuilds_over_the_source():
    data = RNG.integers(0, 1000, (4, 9, 11))                         # z y x
    view = ArrayImage(data, "zyx").as_axes("canonical")
    assert view.axes == "tczyx"
    squeezed = view.squeeze()
    assert squeezed.axes == "zyx" and squeezed.source is view.source
    _check(squeezed, data)
    _check(view.squeeze("t"), data[None])                          # c z y x


def test_squeeze_with_explicit_axes_keeps_their_order():
    data = RNG.integers(0, 1000, (4, 2, 9, 11))                      # z c y x
    view = ArrayImage(data, "zcyx").as_axes("tczyx", squeeze=True)
    assert view.axes == "czyx"
    _check(view, data.transpose(1, 0, 2, 3))


def test_open_canonical_squeezed(tmp_path):
    from test_mrc import write_mrc
    data = RNG.integers(0, 1000, (3, 16, 20)).astype("i2")
    path = str(tmp_path / "v.mrc")
    write_mrc(path, data)
    with micro_reader.open(path, axes="canonical") as img:
        assert img.images[0].axes == "tczyx"
    with micro_reader.open(path, axes="canonical", squeeze=True) as img:
        assert img.images[0].axes == "zyx"
        _check(img.images[0], data)


# -- image keys -----------------------------------------------------------------------

def test_indices_select_and_range():
    from micro_reader import File
    keys = [{"series": 0, "tile": 0}, {"series": 0, "tile": 1}, {"series": 1, "tile": 0}]
    images = []
    for i, key in enumerate(keys):
        image = ArrayImage(np.zeros((2, 3)), "yx")
        image.index = key
        images.append(image)
    f = File("x", images)
    assert f.indices == keys
    assert f.index_range == {"series": 2, "tile": 2}    # widest range: series 0 has 2 tiles, series 1 one
    assert f.select(series=0) == images[:2]
    assert f.select(series=1, tile=0) == [images[2]]
    assert images[1].as_axes("canonical").index == keys[1]           # axes images keep it
    with pytest.raises(KeyError, match="unknown indices"):
        f.select(position=0)
    assert not hasattr(f, "find")                                    # renamed select


# -- one File for every format ------------------------------------------------------

@pytest.mark.parametrize("make, backend, fmt", [
    ("ome", "tiff", "ome-tiff"), ("imagej", "tiff", "imagej"), ("plain", "tiff", "tiff"),
    ("bigtiff", "tiff", "bigtiff")])
def test_every_format_opens_as_a_file_with_backend_and_format(tmp_path, make, backend, fmt):
    import tifffile
    import micro_reader
    path = str(tmp_path / f"{make}.tif")
    data = np.zeros((2, 8, 8), np.uint8)
    kwargs = {"ome": dict(ome=True), "imagej": dict(imagej=True), "plain": dict(metadata=None),
              "bigtiff": dict(bigtiff=True, metadata=None)}[make]
    tifffile.imwrite(path, data, photometric="minisblack", **kwargs)
    with micro_reader.open(path) as f:
        assert isinstance(f, micro_reader.File)
        assert (f.backend, f.format) == (backend, fmt)
        assert repr(f).startswith("<micro_reader.File ") and f"format={fmt!r}" in repr(f)
        assert repr(f.images[0]).startswith("<micro_reader.Image ")


def test_resolution_index_follows_the_file_not_the_image():
    """A file with one pyramidal and one flat image: both have a resolution
    in their index (the file has that dimension), the flat one 0."""
    from micro_reader import File
    pyramidal, level1, flat = (ArrayImage(np.zeros((4, 4)), "yx") for _ in range(3))
    pyramidal.index, level1.index = {"series": 0}, {"series": 0, "resolution": 1}
    pyramidal._resolutions = level1._resolutions = [pyramidal, level1]
    flat.index = {"series": 1}
    f = File("x", [pyramidal, flat])
    assert pyramidal.is_pyramidal and not flat.is_pyramidal
    assert f.indices == [{"series": 0, "resolution": 0}, {"series": 1, "resolution": 0}]
    assert level1.index == {"series": 0, "resolution": 1}
