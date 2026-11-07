"""Lambda scans are one image in every format (decision MR-5, _lambda.py): the bands along
``c`` by default, each channel with its wavelength; a real axis ``l`` with ``axes="native"``.
What each backend reads is in its own tests (the bands, index ``emission``)."""
from __future__ import annotations

import numpy as np
import pytest

import micro_reader
from micro_reader._registry import open as real_open

tifffile = pytest.importorskip("tifffile")

from test_oir import _data, write_oir  # noqa: E402


@pytest.fixture
def scan(tmp_path):
    path = str(tmp_path / "scan.oir")
    data = _data((3, 2, 2, 1, 16, 20))                       # l t c z y x
    write_oir(path, data, frame_axes={"LAMBDA": lambda lam, t, z: 500.0 + 25 * lam,
                                      "TIMELAPSE": lambda lam, t, z: 1500.0 * t})
    return path, data


def test_a_scan_is_one_image_with_its_bands_along_c(scan):
    path, data = scan
    with real_open(path) as f:
        (image,) = f.images
        assert dict(image.index) == {"series": 0} and image.name == "scan"
        assert (image.axes, image.shape) == ("tczyx", (2, 6, 1, 16, 20))
        # a channel's bands next to each other, in band order
        for k, (j, band) in enumerate((j, b) for j in range(2) for b in range(3)):
            np.testing.assert_array_equal(image[:, k], data[band, :, j])
        channels = image.metadata.channels
        assert [c.emission_wavelength.value for c in channels] == [500.0, 525.0, 550.0] * 2
        assert channels[1].name.endswith("525 nm") and channels[0].name != channels[3].name
        assert "emission" not in image.metadata.index_values
        assert image.metadata.pixel_sizes["t"] == (1500.0, "millisecond")   # the bands' own


def test_native_has_a_real_lambda_axis(scan):
    path, data = scan
    with real_open(path, axes="native") as f:
        (image,) = f.images
        assert (image.axes, image.shape) == ("tlczyx", (2, 3, 2, 1, 16, 20))
        np.testing.assert_array_equal(image[...], np.moveaxis(data, 0, 1))
        np.testing.assert_array_equal(image[1, 1:3, 0, 0, 4:9, 2:7],
                                      np.moveaxis(data, 0, 1)[1, 1:3, 0, 0, 4:9, 2:7])
        assert [q.value for q in image.metadata.index_values["emission"]] == [500.0, 525.0, 550.0]


def test_axes_that_name_l_give_the_axis(scan):
    path, data = scan
    with real_open(path, axes="lctzyx") as f:
        image = f.images[0]
        assert image.axes == "lctzyx"
        np.testing.assert_array_equal(image[...], np.moveaxis(data, 2, 1))


def test_a_native_scan_asked_for_other_axes_says_so(scan):
    with real_open(scan[0], axes="native") as f:
        with pytest.raises(ValueError, match="without axes='native'"):
            f.images[0].as_axes("tczyx")


def test_scans_of_other_sizes_stay_apart():
    from types import SimpleNamespace

    from micro_reader._lambda import gather

    class Band(SimpleNamespace):
        level, is_pyramidal = 0, False
    a = Band(index={"series": 0, "emission": 0}, axes="cyx", shape=(1, 4, 4), dtype=np.uint8)
    b = Band(index={"series": 0, "emission": 1}, axes="cyx", shape=(1, 5, 4), dtype=np.uint8)
    f = SimpleNamespace(images=[a, b], _bind_plates=lambda: None)
    gather(f, False)
    assert f.images == [a, b]


def test_lsm_channels_carry_their_detection_bands(tmp_path):
    from micro_reader.backends._tiff_metadata import _band_wavelengths
    bands = np.array([[4.1e-7, 4.9e-7], [-1.0, -1.0], [0.0, 0.0]])
    assert [q.value if q else None for q in _band_wavelengths(bands)] == [450.0, None, None]
    assert _band_wavelengths(None) == []
