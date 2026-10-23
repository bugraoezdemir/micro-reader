"""OME companion files (backends/companion.py, docs/COMPANION.md, CO-1):
synthetic sets of a companion plus small OME-TIFF members, one plane per file
or one multi-page file per timepoint."""
from __future__ import annotations

import os
import types
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import micro_reader
from micro_reader.backends import companion

tifffile = pytest.importorskip("tifffile")
RNG = np.random.default_rng(7)
NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
Y, X = 20, 24
COLOURS = [16711935, -16776961]                 # OME RGBA: green, red


def _stub(own, points_to, metadata_file="set.companion.ome"):
    """A member's OME-XML: its own UUID and a BinaryOnly stub (None: no stub)."""
    stub = (f'<BinaryOnly MetadataFile="{metadata_file}" UUID="{points_to}"/>'
            if points_to else "")
    own = f' UUID="{own}"' if own else ""
    return f'<OME xmlns="{NS}"{own}>{stub}</OME>'


def _write(folder, *, T=2, C=2, Z=3, order="XYZCT", layout="planes", images=1, own="listed",
           points_to="this", list_uuids=True, physical_z=True, positions_z=None, skip=(),
           name="set.companion.ome", absolute=False, member_shape=None, time=True,
           channel_info=True, metadata_file="set.companion.ome", dpi96=False,
           positions_xy=None):
    """A companion set in *folder*. *layout*: "planes" (one file per plane) or
    "stacks" (one file per timepoint, its planes in DimensionOrder; T must be
    slowest). *time* / *channel_info*: the companion gives a TimeIncrement /
    channel names and colours. *metadata_file*: what members' stubs point to;
    *dpi96*: members carry 96 dpi resolution tags. *own*: each member's own
    UUID "listed" (as in the companion),
    "other" or None; *points_to*: its stub points to "this" companion,
    "other", or None (no stub). *skip*: (image, file number) not written.
    *positions_xy*: {t: (x, y)} in um, the PositionX / Y of every plane of that
    timepoint (the field's centre), each image 1000 um further in x; every
    plane gets a Plane element, the other timepoints' without a position.
    -> (companion path, {(image, t, c, z): plane})."""
    os.makedirs(folder, exist_ok=True)
    this = "urn:uuid:companion-0"
    dims = order[2:].lower()
    sizes = {"t": T, "c": C, "z": Z}

    def place(i):
        at = {}
        for a in dims:
            at[a] = i % sizes[a]
            i //= sizes[a]
        return at["t"], at["c"], at["z"]

    n = T * C * Z
    data, xml_images = {}, []
    for im in range(images):
        if layout == "planes":
            groups = [[i] for i in range(n)]
        else:
            per = C * Z
            groups = [list(range(t * per, (t + 1) * per)) for t in range(T)]
        tiffdata = []
        for g, flats in enumerate(groups):
            fname = f"img{im}_part{g:03d}.tif"
            listed = f"urn:uuid:member-{im}-{g}"
            mine = {"listed": listed, "other": f"urn:uuid:else-{im}-{g}", None: None}[own]
            target = {"this": this, "other": "urn:uuid:another-companion", None: None}[points_to]
            desc = _stub(mine, target, metadata_file) if (mine or target) else None
            pages = []
            for i in flats:
                a = RNG.integers(1, 60000, member_shape or (Y, X)).astype("u2")
                data[(im, *place(i))] = a
                pages.append(a)
            if (im, g) not in skip:
                with tifffile.TiffWriter(os.path.join(folder, fname)) as w:
                    for k, a in enumerate(pages):
                        w.write(a, description=desc if k == 0 else None, metadata=None,
                                compression="zlib", tile=(16, 16), contiguous=False,
                                resolution=(96, 96) if dpi96 else None,
                                resolutionunit="INCH" if dpi96 else None)
            t, c, z = place(flats[0])
            # an absolute path from the acquisition computer, which does not exist here
            path = (os.path.join(os.path.abspath(os.sep), "acquisition-pc-x7", fname)
                    if absolute else fname)
            uuid = listed if list_uuids else ""
            tiffdata.append(f'<TiffData FirstT="{t}" FirstC="{c}" FirstZ="{z}" IFD="0" '
                            f'PlaneCount="{len(flats)}"><UUID FileName="{path}">{uuid}</UUID>'
                            f'</TiffData>')
        info = (lambda c: f' Name="ch{c}" Color="{COLOURS[c % 2]}"') if channel_info else             (lambda c: "")
        chans = "".join(f'<Channel ID="Channel:{im}:{c}"{info(c)} SamplesPerPixel="1"/>'
                        for c in range(C))
        planes = ""
        if positions_z is not None:
            planes = "".join(f'<Plane TheT="0" TheC="0" TheZ="{z}" PositionZ="{p}" '
                             f'PositionZUnit="µm"/>' for z, p in enumerate(positions_z))
        if positions_xy is not None:
            def at(t):
                if t not in positions_xy:
                    return ""
                x, y = positions_xy[t]
                return (f' PositionX="{x + 1000 * im}" PositionXUnit="µm" PositionY="{y}" '
                        f'PositionYUnit="µm"')
            planes = "".join(f'<Plane TheT="{t}" TheC="{c}" TheZ="{z}"{at(t)}/>'
                             for t in range(T) for c in range(C) for z in range(Z))
        pz = ' PhysicalSizeZ="2.5" PhysicalSizeZUnit="µm"' if physical_z else ""
        tinc = ' TimeIncrement="60" TimeIncrementUnit="s"' if time else ""
        xml_images.append(
            f'<Image ID="Image:{im}" Name="series {im}"><Pixels ID="Pixels:{im}" '
            f'DimensionOrder="{order}" Type="uint16" SizeX="{X}" SizeY="{Y}" SizeZ="{Z}" '
            f'SizeC="{C}" SizeT="{T}" PhysicalSizeX="0.5" PhysicalSizeY="0.5"{pz} '
            f'{tinc}>{chans}{"".join(tiffdata)}{planes}'
            f'</Pixels></Image>')
    xml = (f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="{NS}" UUID="{this}">'
           f'{"".join(xml_images)}</OME>')
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(xml)
    return path, data


def _check(image, data, im=0):
    T, C, Z = image.shape[:3]
    got = image[...]
    for t in range(T):
        for c in range(C):
            for z in range(Z):
                np.testing.assert_array_equal(got[t, c, z], data[(im, t, c, z)])


@pytest.mark.parametrize("layout", ["planes", "stacks"])
@pytest.mark.parametrize("order", ["XYZCT", "XYCZT"])
def test_a_set_reads_as_one_image_with_its_metadata(tmp_path, layout, order):
    path, data = _write(str(tmp_path), layout=layout, order=order)
    with micro_reader.open(path) as f:
        assert (f.backend, f.format) == ("companion", "ome-companion")
        assert f.indices == [{"series": 0}]
        im = f.images[0]
        assert im.axes == "tczyx" and im.shape == (2, 2, 3, Y, X) and im.dtype == np.uint16
        assert im.name == "series 0"
        _check(im, data)
        # a region across planes and members, against the planes themselves
        part = im[1, :, 1:3, 3:17, 5:21]
        for c in range(2):
            for z in (1, 2):
                np.testing.assert_array_equal(part[c, z - 1], data[(0, 1, c, z)][3:17, 5:21])
        m = im.metadata
        q = micro_reader.Quantity
        assert m.pixel_sizes == {"x": q(0.5, "micrometer"), "y": q(0.5, "micrometer"),
                                 "z": q(2.5, "micrometer"), "t": q(60.0, "second")}
        assert [(c.name, c.color) for c in m.channels] == [("ch0", "#00FF00"), ("ch1", "#FF0000")]


def test_two_images_in_one_companion(tmp_path):
    path, data = _write(str(tmp_path), images=2, layout="stacks")
    with micro_reader.open(path) as f:
        assert f.indices == [{"series": 0}, {"series": 1}]
        assert [im.name for im in f.images] == ["series 0", "series 1"]
        _check(f.images[1], data, im=1)


def test_a_companion_named_ome_xml_is_recognised(tmp_path):
    path, data = _write(str(tmp_path), name="meta.ome.xml")
    with micro_reader.open(path) as f:
        assert f.backend == "companion"
        _check(f.images[0], data)


def test_missing_members_read_as_zeros_and_an_image_without_files_is_left_out(tmp_path):
    path, data = _write(str(tmp_path), images=2, layout="stacks",
                        skip={(0, 1), (1, 0), (1, 1)})
    with pytest.warns(micro_reader.MissingImagesWarning):
        f = micro_reader.open(path)
    with f:
        assert f.indices == [{"series": 0}]               # image 1 has none of its files
        got = f.images[0][...]
        np.testing.assert_array_equal(got[0, 1, 2], data[(0, 0, 1, 2)])
        assert not got[1].any()                          # timepoint 1's file is missing


# -- matching: a member belongs when either UUID check passes -----------------

@pytest.mark.parametrize("own, points_to", [
    ("listed", None),          # its own UUID is the companion's entry
    ("other", "this"),         # Viventis: another UUID, but its stub points here
    ("listed", "other"),       # the entry matches; the stub's other target is outvoted
    (None, None),              # no UUIDs: the file name decides
])
def test_a_member_that_passes_a_check_belongs(tmp_path, own, points_to):
    path, data = _write(str(tmp_path), own=own, points_to=points_to)
    with micro_reader.open(path) as f:
        _check(f.images[0], data)


@pytest.mark.parametrize("own, points_to", [("other", None), ("other", "other"),
                                            (None, "other")])
def test_a_member_that_fails_and_passes_no_check_is_an_error(tmp_path, own, points_to):
    path, _data = _write(str(tmp_path), own=own, points_to=points_to)
    with pytest.raises(micro_reader.CorruptFileError, match="is not part of the set"):
        micro_reader.open(path)


def test_without_uuids_in_the_companion_the_name_decides(tmp_path):
    path, data = _write(str(tmp_path), own="other", points_to=None, list_uuids=False)
    with micro_reader.open(path) as f:
        _check(f.images[0], data)


def test_a_member_of_another_size_is_an_error(tmp_path):
    path, _data = _write(str(tmp_path), member_shape=(Y, X + 1))
    with pytest.raises(micro_reader.CorruptFileError, match="expects"):
        micro_reader.open(path)


def test_an_absolute_path_that_does_not_exist_falls_back_to_the_folder(tmp_path):
    path, data = _write(str(tmp_path), absolute=True)
    with pytest.warns(micro_reader.CompanionWarning, match="absolute"):
        f = micro_reader.open(path)
    with f:
        _check(f.images[0], data)


# -- the z step ---------------------------------------------------------------

def test_without_physical_size_z_the_even_position_z_steps_give_it(tmp_path):
    path, _data = _write(str(tmp_path), physical_z=False, positions_z=[10.0, 13.0, 16.0])
    with micro_reader.open(path) as f, pytest.warns(micro_reader.CompanionWarning,
                                                    match="PositionZ"):
        sizes = f.images[0].metadata.pixel_sizes
    assert sizes["z"] == micro_reader.Quantity(3.0, "micrometer")


def test_uneven_positions_give_no_z_and_members_without_metadata_neither(tmp_path):
    path, _data = _write(str(tmp_path), physical_z=False, positions_z=[10.0, 13.0, 19.0])
    with micro_reader.open(path) as f, warnings.catch_warnings():
        warnings.simplefilter("error")                   # nothing taken from elsewhere
        assert "z" not in f.images[0].metadata.pixel_sizes


def test_the_members_own_metadata_fills_a_missing_z(tmp_path):
    path, _data = _write(str(tmp_path), physical_z=False, layout="stacks", C=1,
                         own=None, points_to=None)
    for t in range(2):                                   # members with ImageJ z spacing
        member = os.path.join(str(tmp_path), f"img0_part{t:03d}.tif")
        stack = tifffile.imread(member)
        tifffile.imwrite(member, stack, imagej=True, metadata={"spacing": 1.5, "unit": "um"})
    with micro_reader.open(path) as f, pytest.warns(micro_reader.CompanionWarning,
                                                    match="member file"):
        sizes = f.images[0].metadata.pixel_sizes
    assert sizes["z"] == micro_reader.Quantity(1.5, "micrometer")
    assert sizes["x"] == micro_reader.Quantity(0.5, "micrometer")    # the companion wins


# -- refused ------------------------------------------------------------------

@pytest.mark.parametrize("edit, error", [
    (lambda x: x.replace('SamplesPerPixel="1"', 'SamplesPerPixel="3"'),
     micro_reader.UnsupportedFormatError),
    (lambda x: x.replace("</Pixels>", '<BinData Length="0"/></Pixels>'),
     micro_reader.UnsupportedFormatError),
    (lambda x: x.replace('<UUID FileName="img0_part000.tif">urn:uuid:member-0-0</UUID>', ""),
     micro_reader.CorruptFileError),
    (lambda x: x.replace('FirstZ="1"', 'FirstZ="0"', 1), micro_reader.CorruptFileError),
])
def test_what_it_cannot_read_is_refused(tmp_path, edit, error):
    path, _data = _write(str(tmp_path))
    text = open(path, encoding="utf-8").read()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(edit(text))
    with pytest.raises(error):
        micro_reader.open(path)


# -- a companion whose bytes are not in the encoding it declares ------------------

def _rewrite_bytes(path, edit):
    """The companion at *path* with its bytes changed by *edit*."""
    with open(path, "rb") as fh:
        raw = fh.read()
    with open(path, "wb") as fh:
        fh.write(edit(raw))


def _mis_encoded(raw: bytes, byte: bytes = b"\xb5") -> bytes:
    """*raw* as Prairie View writes it: a UTF-8 BOM, the UTF-8 declaration
    kept, every ``µ`` as the one Windows *byte*."""
    return b"\xef\xbb\xbf" + raw.replace("µ".encode("utf-8"), byte)


def test_a_companion_with_windows_bytes_under_a_utf8_declaration_is_read(tmp_path):
    """Prairie View writes ``µ`` as the one byte 0xB5 into every unit attribute
    of a companion that declares UTF-8 (Bruker, zenodo 15224926): it is read as
    cp1252 with one CompanionWarning naming the file and the encoding, and the
    set reads as it would with proper UTF-8, its members too."""
    path, data = _write(str(tmp_path), positions_z=[1.0, 3.5, 6.0])
    _rewrite_bytes(path, _mis_encoded)
    assert b"\xb5" in open(path, "rb").read() and b"\xc2\xb5" not in open(path, "rb").read()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with micro_reader.open(path) as f:
            (im,) = f.images
            assert im.shape == (2, 2, 3, Y, X)
            _check(im, data)
            z = im.metadata.pixel_sizes["z"]
            assert (z.value, z.unit) == (2.5, "micrometer")      # the µm unit was read
        with micro_reader.open(_member(str(tmp_path), 2)) as f:  # a member opens the whole
            assert f.backend == "companion" and f.images[0].shape == (2, 2, 3, Y, X)
        with micro_reader.open(_member(str(tmp_path), 2), multifile=False) as f:
            assert f.images[0].metadata.pixel_sizes["z"].unit == "micrometer"
    told = [w for w in caught if issubclass(w.category, micro_reader.CompanionWarning)]
    assert len(told) == 1                           # parsed once, warned once
    message = str(told[0].message)
    assert repr(path) in message and "'utf-8'" in message and "'cp1252'" in message


def test_a_byte_cp1252_does_not_have_falls_back_to_latin_1(tmp_path):
    path, _data = _write(str(tmp_path))
    _rewrite_bytes(path, lambda raw: _mis_encoded(raw, b"\x81"))   # undefined in cp1252
    with pytest.warns(micro_reader.CompanionWarning, match="read as 'latin-1'"):
        with micro_reader.open(path) as f:
            (im,) = f.images
            assert im.shape == (2, 2, 3, Y, X)
            assert im.metadata.pixel_sizes["x"].value == 0.5    # "\x81m" is no unit: no z


@pytest.mark.parametrize("edit", [
    lambda raw: _mis_encoded(raw).replace(b"</OME>", b""),     # bad bytes and broken XML
    lambda raw: raw.replace(b"</OME>", b""),                    # fine bytes, broken XML
    lambda raw: b"<OME \xb5 this is not XML at all",             # sniffed, not parsed
], ids=["bad-bytes-broken-xml", "broken-xml", "no-xml"])
def test_what_is_no_xml_in_any_encoding_stays_an_error(tmp_path, edit):
    path, _data = _write(str(tmp_path))
    _rewrite_bytes(path, edit)
    with pytest.raises(micro_reader.CorruptFileError, match="not readable XML"):
        micro_reader.open(path)


# -- caching and member files ---------------------------------------------------

def test_a_companion_is_parsed_once_while_it_does_not_change(tmp_path, monkeypatch):
    path, _data = _write(str(tmp_path))
    parsed = []
    real = companion.Core
    monkeypatch.setattr(companion, "Core", lambda p: parsed.append(p) or real(p))
    companion._cores.clear()
    for _ in range(3):
        micro_reader.open(path).close()
    assert len(parsed) == 1
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    micro_reader.open(path).close()
    assert len(parsed) == 2


def test_few_members_stay_open_and_threads_read_safely(tmp_path, monkeypatch):
    monkeypatch.setattr(companion, "OPEN_MEMBERS", 2)
    path, data = _write(str(tmp_path), T=3, C=2, Z=4)
    with micro_reader.open(path) as f:
        im = f.images[0]

        def region(y0):
            got = im[:, :, :, y0:y0 + 5, :]
            for (_i, t, c, z), plane in data.items():
                np.testing.assert_array_equal(got[t, c, z], plane[y0:y0 + 5])
            return True

        with ThreadPoolExecutor(8) as pool:
            assert all(pool.map(region, [0, 5, 10, 15] * 3))
        assert len(f._members._open) <= 2


# -- CO-2, multifile=False: one member file, its metadata from its companion -------

def _member(folder, number=0):
    return os.path.join(folder, f"img0_part{number:03d}.tif")


def test_a_plane_opened_alone_takes_its_metadata_from_the_companion(tmp_path):
    path, data = _write(str(tmp_path), dpi96=True)       # its 96 dpi is no pixel size
    with micro_reader.open(_member(str(tmp_path), 4), multifile=False) as f:
        assert f.backend == "tiff" and f.companion == path
        im = f.images[0]
        # plane 4 in XYZCT order: z 1, c 1, t 0; t, c and z are explicit, so kept
        assert im.axes == "tczyx" and im.shape == (1, 1, 1, Y, X)
        np.testing.assert_array_equal(im[0, 0, 0], data[(0, 0, 1, 1)])
        m = im.metadata
        q = micro_reader.Quantity
        assert m.pixel_sizes == {"x": q(0.5, "micrometer"), "y": q(0.5, "micrometer"),
                                 "z": q(2.5, "micrometer"), "t": q(60.0, "second")}
        assert [(c.name, c.color) for c in m.channels] == [("ch1", "#FF0000")]


def test_only_axes_the_companion_says_something_about_are_added(tmp_path):
    path, _data = _write(str(tmp_path), physical_z=False, time=False)
    with micro_reader.open(_member(str(tmp_path)), multifile=False) as f:
        assert f.images[0].axes == "cyx"                 # channel information only
    path, _data = _write(str(tmp_path / "bare"), physical_z=False, time=False,
                         channel_info=False)
    with micro_reader.open(_member(str(tmp_path / "bare")), multifile=False) as f:
        im = f.images[0]
        assert im.axes == "yx" and im.metadata.pixel_sizes["x"].value == 0.5


def test_a_multi_page_member_opens_as_its_block_of_the_image(tmp_path):
    path, data = _write(str(tmp_path), layout="stacks")  # one file per t: 2 c x 3 z pages
    with micro_reader.open(_member(str(tmp_path), 1), multifile=False) as f:
        im = f.images[0]
        assert im.axes == "tczyx" and im.shape == (1, 2, 3, Y, X)
        got = im[...]
        for c in range(2):
            for z in range(3):
                np.testing.assert_array_equal(got[0, c, z], data[(0, 1, c, z)])
        assert [c.name for c in im.metadata.channels] == ["ch0", "ch1"]


def test_without_its_companion_a_stub_opens_plainly_with_a_loud_warning(tmp_path):
    path, data = _write(str(tmp_path), dpi96=True)
    os.remove(path)
    with pytest.warns(micro_reader.CompanionWarning, match="is not there"):
        f = micro_reader.open(_member(str(tmp_path)), multifile=False)
    with f:
        im = f.images[0]
        assert im.axes == "yx" and im.metadata.pixel_sizes == {}   # never the 96 dpi
        np.testing.assert_array_equal(im[...], data[(0, 0, 0, 0)])


@pytest.mark.parametrize("own, points_to, recognised", [
    ("other", "this", True),           # Viventis
    ("listed", None, None),            # no stub: not a pointer, opens as it is
    ("other", "other", False),         # neither check passes
])
def test_a_member_is_recognised_when_a_check_passes(tmp_path, own, points_to, recognised):
    path, _data = _write(str(tmp_path), own=own, points_to=points_to)
    with warnings.catch_warnings(record=True) as told:
        warnings.simplefilter("always")
        f = micro_reader.open(_member(str(tmp_path)), multifile=False)
    with f:
        assert (getattr(f, "companion", None) == path) is bool(recognised)
        refused = [w for w in told if issubclass(w.category, micro_reader.CompanionWarning)]
        assert bool(refused) is (recognised is False)


def test_a_stub_may_point_to_an_ome_tiff(tmp_path):
    """The full OME-XML in another member's first page (no .companion.ome):
    a stub pointing to that OME-TIFF takes its metadata from it."""
    path, data = _write(str(tmp_path), metadata_file="img0_part000.tif")
    xml = open(path, encoding="utf-8").read()
    os.remove(path)
    tifffile.imwrite(_member(str(tmp_path), 0), data[(0, 0, 0, 0)],
                     description=xml.encode("utf-8"), metadata=None)   # the master: XML + plane 0
    with warnings.catch_warnings():
        warnings.simplefilter("error", micro_reader.CompanionWarning)
        f = micro_reader.open(_member(str(tmp_path), 1), multifile=False)
    with f:
        assert os.path.samefile(f.companion, _member(str(tmp_path), 0))
        im = f.images[0]
        assert im.metadata.pixel_sizes["x"].value == pytest.approx(0.5)
        np.testing.assert_array_equal(np.asarray(im[...]).reshape(Y, X), data[(0, 0, 0, 1)])


def _xml_in_every_file(folder, T=3):
    """A multi-file OME-TIFF: one file per timepoint, the full OME-XML in
    every file's first page, no companion.  -> (paths, data)."""
    path, data = _write(folder, layout="stacks", T=T)
    xml = open(path, encoding="utf-8").read().encode("utf-8")
    os.remove(path)
    paths = [_member(folder, g) for g in range(T)]
    for p in paths:
        with tifffile.TiffFile(p) as tif:
            pages = [page.asarray() for page in tif.pages]
        with tifffile.TiffWriter(p) as w:
            for k, a in enumerate(pages):
                w.write(a, description=xml if k == 0 else None, metadata=None)
    return paths, data


def test_multifile_false_opens_one_file_of_a_multi_file_ome_tiff(tmp_path):
    paths, data = _xml_in_every_file(str(tmp_path))
    with micro_reader.open(paths[1]) as f:                       # default: the whole set
        assert f.images[0].shape[0] == 3
    with micro_reader.open(paths[1], multifile=False) as f:      # only its own timepoint
        (im,) = f.images
        assert im.shape[:3] == (1, 2, 3)
        got = im[...]
        for c in range(2):
            for z in range(3):
                np.testing.assert_array_equal(got[0, c, z], data[(0, 1, c, z)])
        assert im.metadata.pixel_sizes["x"].value == pytest.approx(0.5)
        assert [ch.name for ch in im.metadata.channels] == ["ch0", "ch1"]


def test_multifile_false_leaves_micro_manager_stacks_whole(tmp_path, monkeypatch):
    """Micro-Manager splits its files by size, mid-timepoint (61 / 2 planes
    of one timepoint on the OME 4.7 GB sample): a file is no block of the
    image, so multifile=False opens the whole stack, as by default."""
    paths, _data = _xml_in_every_file(str(tmp_path))
    # tifffile takes the file's flags from its first page
    monkeypatch.setattr(tifffile.TiffPage, "is_micromanager", property(lambda self: True))
    with micro_reader.open(paths[1], multifile=False) as f:
        assert f.images[0].shape[0] == 3


def test_multifile_false_leaves_single_file_ome_tiffs_and_others_alone(tmp_path):
    single = str(tmp_path / "single.ome.tif")
    tifffile.imwrite(single, np.zeros((2, Y, X), "u2"), metadata={"axes": "ZYX"})
    plain = str(tmp_path / "plain.tif")
    tifffile.imwrite(plain, np.zeros((Y, X), "u2"))
    for p in (single, plain):
        with micro_reader.open(p) as a, micro_reader.open(p, multifile=False) as b:
            assert [im.shape for im in a.images] == [im.shape for im in b.images]


def test_a_renamed_member_is_found_by_its_uuid(tmp_path):
    path, data = _write(str(tmp_path), own="listed", points_to="this")
    renamed = os.path.join(str(tmp_path), "renamed.tif")
    os.rename(_member(str(tmp_path)), renamed)
    with micro_reader.open(renamed, multifile=False) as f:
        assert f.companion == path
        np.testing.assert_array_equal(f.images[0][0, 0, 0], data[(0, 0, 0, 0)])


def test_an_absolute_pointer_that_does_not_exist_falls_back_to_the_folder(tmp_path):
    gone = os.path.join(os.path.abspath(os.sep), "acquisition-pc-x7", "set.companion.ome")
    path, _data = _write(str(tmp_path), metadata_file=gone)
    with pytest.warns(micro_reader.CompanionWarning, match="does not exist"):
        f = micro_reader.open(_member(str(tmp_path)), multifile=False)
    with f:
        assert f.companion == path


def test_a_companion_that_does_not_fit_the_file_is_not_used(tmp_path):
    _path, _data = _write(str(tmp_path), member_shape=(Y, X + 1))
    with pytest.warns(micro_reader.CompanionWarning, match="another size"):
        f = micro_reader.open(_member(str(tmp_path)), multifile=False)
    with f:
        assert f.images[0].axes == "yx" and f.images[0].metadata.pixel_sizes == {}


def test_pages_that_are_not_one_block_are_refused(tmp_path):
    path, _data = _write(str(tmp_path), layout="stacks", C=1)    # pages z 0, 1, 2
    text = open(path, encoding="utf-8").read()
    # file 0 holds z 0 and z 2 of t 0, file 1 the rest
    text = text.replace(
        '<TiffData FirstT="0" FirstC="0" FirstZ="0" IFD="0" PlaneCount="3">'
        '<UUID FileName="img0_part000.tif">urn:uuid:member-0-0</UUID></TiffData>',
        '<TiffData FirstT="0" FirstC="0" FirstZ="0" IFD="0" PlaneCount="1">'
        '<UUID FileName="img0_part000.tif">urn:uuid:member-0-0</UUID></TiffData>'
        '<TiffData FirstT="0" FirstC="0" FirstZ="2" IFD="2" PlaneCount="1">'
        '<UUID FileName="img0_part000.tif">urn:uuid:member-0-0</UUID></TiffData>'
        '<TiffData FirstT="0" FirstC="0" FirstZ="1" IFD="0" PlaneCount="1">'
        '<UUID FileName="img0_part001.tif">urn:uuid:member-0-1</UUID></TiffData>')
    text = text.replace('FirstT="1" FirstC="0" FirstZ="0" IFD="0" PlaneCount="3"',
                        'FirstT="1" FirstC="0" FirstZ="0" IFD="1" PlaneCount="2"')
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    companion._cores.clear()
    with pytest.raises(micro_reader.UnsupportedFormatError, match="not one block"):
        micro_reader.open(_member(str(tmp_path)), multifile=False)


# -- multifile=True (the default): a member opens its whole set ---------------------
# (docs/COMPANION.md, "DECIDED: multifile chooses between CO-1 and CO-2")

@pytest.mark.parametrize("own, points_to", [("listed", "this"), ("other", "this")])
def test_a_member_opens_the_whole_set_as_its_companion_would(tmp_path, own, points_to):
    path, data = _write(str(tmp_path), own=own, points_to=points_to, dpi96=True)
    member = _member(str(tmp_path), 4)
    with micro_reader.open(member) as f, micro_reader.open(path) as g:
        assert f.backend == "companion" and f.path == path and f.opened == member
        (im,) = f.images
        assert im.shape == g.images[0].shape == (2, 2, 3, Y, X)
        _check(im, data)
        # the companion's metadata wins (it is opened as the companion)
        assert im.metadata.pixel_sizes == g.images[0].metadata.pixel_sizes
        assert [c.name for c in im.metadata.channels] == ["ch0", "ch1"]


def _master_set(folder, compressed_members=False):
    """A set whose full OME-XML is in the first member (no .companion.ome), the
    others BinaryOnly stubs pointing to it; every file one uncompressed strip
    unless *compressed_members*.  -> (master path, data)."""
    path, data = _write(folder, metadata_file="img0_part000.tif")
    xml = open(path, encoding="utf-8").read()
    os.remove(path)
    for g in range(12):
        p = _member(folder, g)
        with tifffile.TiffFile(p) as tif:
            plane, desc = tif.pages[0].asarray(), tif.pages[0].description
        if g == 0:
            tifffile.imwrite(p, plane, description=xml.encode("utf-8"), metadata=None)
        elif not compressed_members:
            tifffile.imwrite(p, plane, description=desc.encode("utf-8"), metadata=None)
    return _member(folder, 0), data


def test_a_member_pointing_to_an_ome_tiff_opens_that_files_set(tmp_path):
    master, data = _master_set(str(tmp_path))
    with micro_reader.open(_member(str(tmp_path), 1)) as f:
        assert os.path.samefile(f.path, master) and f.backend == "tiff"
        assert f.images[0].shape[:3] == (2, 2, 3)
        _check(f.images[0], data)


def test_a_set_whose_files_are_stored_differently_is_refused_not_misread(tmp_path):
    """Real-data round: pages of a multi-file OME-TIFF are decoded as its first
    page is; members stored otherwise (here tiled zlib after an uncompressed
    master) read as wrong pixels before, and are refused now."""
    master, data = _master_set(str(tmp_path), compressed_members=True)
    with micro_reader.open(master) as f:
        np.testing.assert_array_equal(f.images[0][0, 0, 0], data[(0, 0, 0, 0)])
        with pytest.raises(micro_reader.UnsupportedFormatError, match="stored differently"):
            f.images[0][0, 0, 1]


@pytest.mark.parametrize("break_it, why", [
    ("remove", "is not there"),
    ("garble", "cannot be read"),
    ("other set", "describes another set"),
    ("size", "another size"),
])
def test_a_metadata_file_that_cannot_be_followed_is_an_error(tmp_path, break_it, why):
    kw = {"own": "other", "points_to": "other"} if break_it == "other set" else \
         {"member_shape": (Y, X + 1)} if break_it == "size" else {}
    path, _data = _write(str(tmp_path), **kw)
    if break_it == "remove":
        os.remove(path)
    elif break_it == "garble":
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("<OME")
    companion._cores.clear()
    with pytest.raises(micro_reader.CorruptFileError, match=f"{why}.*multifile=False"):
        micro_reader.open(_member(str(tmp_path)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", micro_reader.CompanionWarning)
        with micro_reader.open(_member(str(tmp_path)), multifile=False) as f:   # the file alone
            assert f.images[0].axes == "yx"


def test_a_missing_member_warns_and_the_rest_reads(tmp_path):
    """Missing MEMBERS are no error (a subset, as for every file set): only a
    metadata file that cannot be followed is."""
    path, data = _write(str(tmp_path), skip={(0, 7)})
    with pytest.warns(micro_reader.MissingImagesWarning):
        f = micro_reader.open(_member(str(tmp_path), 4))
    with f:
        assert f.path == path
        got = f.images[0][...]
        np.testing.assert_array_equal(got[0, 0, 1], data[(0, 0, 0, 1)])


def test_a_member_without_a_stub_opens_as_it_is(tmp_path):
    path, _data = _write(str(tmp_path), own="listed", points_to=None)
    with micro_reader.open(_member(str(tmp_path))) as f:
        assert f.backend == "tiff" and f.path == _member(str(tmp_path))


# -- the stage system (plan 4.6) ---------------------------------------------------

def _first_pixel(centre):
    """PositionX / Y (the field's centre, um) -> the first pixel's centre."""
    x, y = centre
    return pytest.approx(x - (X / 2 - 0.5) * 0.5), pytest.approx(y - (Y / 2 - 0.5) * 0.5)


def test_a_sets_positions_are_in_its_one_stage_system(tmp_path, monkeypatch):
    """Every image of a companion set is placed in ``stage``; declaring it
    reads no image's metadata; PositionZ is no part of the position."""
    from micro_reader._spatial import check_invariants
    reads = []
    real = companion._CompanionImage._read_metadata
    monkeypatch.setattr(companion._CompanionImage, "_read_metadata",
                        lambda self: reads.append(self) or real(self))
    path, _data = _write(str(tmp_path), images=2, positions_xy={0: (100.0, 50.0)},
                         positions_z=[10.0, 13.0, 16.0])
    with micro_reader.open(path) as f:
        assert [s.id for s in f.coordinate_systems] == ["stage"]
        assert f.coordinate_systems[0].axes is None and not f.coordinate_systems[0].estimated
        assert reads == []
        for k, im in enumerate(f.images):
            m = im.metadata
            assert m.coordinate_system == "stage" and set(m.position) == {"x", "y"}
            x, y = _first_pixel((100.0 + 1000 * k, 50.0))
            assert (m.position["x"].value, m.position["y"].value) == (x, y)
            assert m.position["x"].unit == "micrometer"
        check_invariants(f)


def test_a_set_without_positions_is_unplaced(tmp_path):
    from micro_reader._spatial import check_invariants
    path, _data = _write(str(tmp_path))
    with micro_reader.open(path) as f:
        assert [s.id for s in f.coordinate_systems] == ["stage"]
        m = f.images[0].metadata
        assert m.position == {} and m.coordinate_system is None
        check_invariants(f)


def test_a_member_opened_alone_is_placed_by_its_own_first_plane(tmp_path):
    """CO-2: the block a member holds is placed by its first plane, in
    ``stage``; a block whose first plane has no position is unplaced."""
    from micro_reader._spatial import check_invariants
    path, _data = _write(str(tmp_path), T=3, layout="stacks",
                         positions_xy={0: (100.0, 50.0), 1: (130.0, 50.0)})
    for number, centre in ((1, (130.0, 50.0)), (2, None)):
        with micro_reader.open(_member(str(tmp_path), number), multifile=False) as f:
            assert (f.backend, f.format) == ("tiff", "ome-tiff")
            assert [s.id for s in f.coordinate_systems] == ["stage"]
            m = f.images[0].metadata
            if centre is None:
                assert m.position == {} and m.coordinate_system is None
            else:
                assert m.coordinate_system == "stage"
                assert (m.position["x"].value, m.position["y"].value) == _first_pixel(centre)
            check_invariants(f)
    with micro_reader.open(_member(str(tmp_path), 1)) as f:      # the whole set
        assert f.backend == "companion"
        assert f.images[0].metadata.coordinate_system == "stage"


DATA = os.environ.get("MICROSCOPY_DATA", os.path.expanduser("~/Desktop/data/microscopy"))
BRUKER = os.path.join(DATA, "formats", "companion", "zenodo-15224926-bruker")


def test_the_bruker_prairie_view_t_series_reads_as_one_time_series():
    """Zenodo 15224926 (Bruker, CC BY 4.0): a Prairie View T-series, a
    companion declaring UTF-8 with Windows ``µ`` bytes in 30,000 Plane units,
    8 OME-TIFF members of 4089 pages (the last 1377) and Prairie's own .xml /
    .env side files.  It opens as one image t=30000 (c=1, z=1, 112 x 128
    uint16) with one CompanionWarning; a member opens the whole set, alone it
    is its block of 4089 timepoints; the side files are no images."""
    companion_path = os.path.join(BRUKER, "TSeries-02282025-1111-071.companion.ome")
    if not os.path.exists(companion_path):
        pytest.skip(f"sample not found: {companion_path} (set MICROSCOPY_DATA)")
    stem = os.path.join(BRUKER, "TSeries-02282025-1111-071")
    first_member, last_member = f"{stem}_Cycle00001_Ch2_000001.ome.tif", f"{stem}_Cycle00001_Ch2_000008.ome.tif"
    companion._cores.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with micro_reader.open(companion_path) as f:
            assert f.backend == "companion" and len(f.images) == 1
            (im,) = f.images
            assert (im.axes, im.shape, im.dtype) == ("tczyx", (30000, 1, 1, 112, 128), np.uint16)
            meta = im.metadata
            assert meta.pixel_sizes["x"].value == meta.pixel_sizes["y"].value == 2.1484375
            assert meta.pixel_sizes["x"].unit == "micrometer"      # µm read, not "?m"
            assert [c.name for c in meta.channels] == ["Ch2"]
            assert meta.coordinate_system == "stage" and set(meta.position) == {"x", "y"}
            first, last = im[0, 0, 0], im[29999, 0, 0]
            with tifffile.TiffFile(first_member) as tif:
                np.testing.assert_array_equal(first, tif.pages[0].asarray())
            with tifffile.TiffFile(last_member) as tif:
                assert len(tif.pages) == 1377
                np.testing.assert_array_equal(last, tif.pages[1376].asarray())
            assert first.max() > 0 and last.max() > 0
    told = [w for w in caught if issubclass(w.category, micro_reader.CompanionWarning)]
    assert len(told) == 1 and "'cp1252'" in str(told[0].message)
    with micro_reader.open(first_member) as f:                    # any part opens the whole
        assert f.backend == "companion" and f.images[0].shape == (30000, 1, 1, 112, 128)
    with micro_reader.open(last_member, multifile=False) as f:
        assert f.backend == "tiff" and f.images[0].shape == (1377, 1, 1, 112, 128)
        assert f.images[0].metadata.pixel_sizes["x"].unit == "micrometer"
    for side in (f"{stem}.xml", f"{stem}.env"):
        with open(side, "rb") as fh:
            assert not companion.sniff(fh.read(4096), side)
        with pytest.raises(micro_reader.UnsupportedFormatError):
            micro_reader.open(side)


def test_companion_sets_and_ome_tiffs_are_checked_strictly():
    """Plan 4.6 landed: the session check no longer tolerates a position
    without a system for companion sets or OME-TIFFs (members too)."""
    import conftest
    assert "companion" not in conftest.POSITIONS_WITHOUT_A_SYSTEM
    assert "tiff" not in conftest.POSITIONS_WITHOUT_A_SYSTEM
    f = types.SimpleNamespace(backend="tiff", format="ome-tiff")
    assert conftest.strict_placement(f)
    assert conftest.strict_placement(types.SimpleNamespace(backend="companion",
                                                           format="ome-companion"))
