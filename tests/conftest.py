"""Session-wide checks shared by every test module."""
import pytest


@pytest.fixture(autouse=True, scope="session")
def every_index_is_an_image_index():
    """Every image ``micro_reader.open`` gives out in the whole session -- its
    images, their resolution levels and parts, also with ``axes=``,
    ``samples=`` and ``resolutions="all"`` -- has an ``ImageIndex`` (the index
    guard, plan 3.24): a backend that stores a plain dict bypasses the guard."""
    import micro_reader
    from micro_reader._base import ImageIndex

    real_open = micro_reader.open
    bad, checked = [], [0]

    def checking_open(*args, **kwargs):
        f = real_open(*args, **kwargs)
        for image in list(f.images) + f._every_image():
            checked[0] += 1
            if type(image.index) is not ImageIndex:
                bad.append(f"{f.path}: {image.name!r} has a {type(image.index).__name__}")
        return f

    micro_reader.open = checking_open
    try:
        yield
    finally:
        micro_reader.open = real_open
    assert not bad, "images without an ImageIndex:\n" + "\n".join(bad[:20])
