"""A minimal Leica LIF writer for tests: the layouts our real files lack
(time, lambda, tiles, interleaved RGB, 12-bit, half float, padded rows,
version 1 headers, nested images, empty folders).

Structure as Leica documents it (and as real files show): header block
``0x70, length = 1 + 4 + 2 * chars, 0x2A, chars, UTF-16 XML``; memory
blocks ``0x70, length = header remainder, 0x2A, size (int32 in version 1,
int64 in version 2), 0x2A, chars, UTF-16 ID, data``.  readlif and
Bio-Formats read these files, which checks the writer.
"""
from __future__ import annotations

import os
import struct
from xml.sax.saxutils import quoteattr

import numpy as np

DIM_IDS = {"x": 1, "y": 2, "z": 3, "t": 4, "l": 5, "m": 10}
#: memory order of the plane axes, fastest first (Leica's usual order)
DEFAULT_ORDER = "xyczltm"


def _dim(dim_id, n, inc, unit="m", length=1e-4):
    return (f'<DimensionDescription DimID="{dim_id}" NumberOfElements="{n}" Origin="0" '
            f'Length="{length}" Unit="{unit}" BitInc="0" BytesInc="{inc}"/>')


def image(name, data, axes="tczyx", *, resolution=None, data_type=None, rgb=False,
          tags=None, order=DEFAULT_ORDER, row_pad=0, tiles=None, lengths=None, children=(),
          setting=None, unique_id=None, timestamps=None, tile_flags=None):
    """One image element.  *data* has *axes* (a subset of t c z l m y x, plus
    s for interleaved colour, samples in R, G, B order); it is stored with
    *order* (fastest first), rows padded by *row_pad* bytes.  *rgb*: store
    the samples B, G, R interleaved in each pixel, tagged 3, 2, 1 (as LAS
    files store colour); *tags* overrides the ChannelTags.  *tiles*:
    [(FieldX, FieldY, PosX, PosY)] for the TileScanInfo attachment.
    *lengths*: {axis: Length in m} instead of the default 1e-4.
    *setting*: attributes of the image's confocal setting (a HardwareSetting
    attachment), e.g. ``StagePosX`` / ``StagePosY`` (m) and
    ``SystemSerialNumber``; *unique_id*: the element's ``UniqueID``;
    *timestamps*: FILETIME ticks for its ``TimeStampList``; *tile_flags*:
    {"SwapXY" / "FlipX" / "FlipY": "1"} on the TileScanInfo attachment."""
    return dict(name=name, data=np.asarray(data), axes=axes, resolution=resolution,
                data_type=data_type, rgb=rgb, tags=tags, order=order, row_pad=row_pad,
                tiles=tiles, lengths=lengths or {}, children=list(children),
                setting=setting, unique_id=unique_id, timestamps=timestamps,
                tile_flags=tile_flags or {})


def folder(name, children=()):
    """An element without pixels (an empty memory block)."""
    return dict(name=name, data=None, children=list(children))


def _layout(img):
    """(XML for the image description, memory block bytes)."""
    data, axes = img["data"], img["axes"]
    full = "tczlmyx" + ("s" if "s" in axes else "")
    for a in full:
        if a not in axes:
            data, axes = data[None], a + axes
    data = data.transpose([axes.index(a) for a in full])     # t c z l m y x [s]
    dtype = data.dtype
    item = dtype.itemsize
    nt, nc, nz, nl, nm, ny, nx = data.shape[:7]
    ns = data.shape[7] if data.ndim == 8 else 1
    data_type = img["data_type"] if img["data_type"] is not None else int(dtype.kind == "f")
    resolution = img["resolution"] or 8 * item
    sizes = {"x": nx, "y": ny, "c": nc, "z": nz, "l": nl, "t": nt, "m": nm}
    strides, pos = {}, 0
    pixel = item * ns
    order = img["order"] + "".join(a for a in DEFAULT_ORDER if a not in img["order"])
    for a in order:
        if a == "x":
            strides[a], pos = pixel, pixel * nx
        elif a == "y":
            strides[a] = pos + img["row_pad"]
            pos = strides[a] * ny
        else:
            strides[a], pos = pos, pos * sizes[a]
    if img["order"] == DEFAULT_ORDER and not img["row_pad"] and ns == 1:
        # Leica's usual layout is C order over m t l z c y x: no scattering
        # (fast enough for benchmark-sized files)
        block = np.ascontiguousarray(data.astype(dtype.newbyteorder("<"))
                                     .transpose(4, 0, 3, 2, 1, 5, 6)).view(np.uint8).reshape(-1)
        assert block.size == pos
    else:                                         # any layout: each sample to its offset
        block = np.zeros(pos, np.uint8)
        stored = [2, 1, 0] if img["rgb"] else list(range(ns))   # file sample order
        grids = np.meshgrid(*(np.arange(n) for n in (nt, nc, nz, nl, nm, ny, nx)), indexing="ij")
        offs = sum(g * strides[a] for g, a in zip(grids, "tczlmyx"))
        for k in range(ns):
            values = data[..., k] if ns > 1 else data
            at = (offs + stored.index(k) * item).reshape(-1)
            raw = np.ascontiguousarray(values.astype(dtype.newbyteorder("<"))).view(np.uint8)
            raw = raw.reshape(-1, item)
            for b in range(item):
                block[at + b] = raw[:, b]
    if img["tags"] is not None:
        tags = img["tags"]
    elif img["rgb"]:
        tags = [3, 2, 1]                                         # stored B, G, R
    else:
        tags = [0] * nc
    if ns > 1:
        channels = [(tags[k], k * item) for k in range(ns)]
    else:
        channels = [(tags[c] if c < len(tags) else 0, c * strides["c"]) for c in range(nc)]
    ch_xml = "".join(
        f'<ChannelDescription DataType="{data_type}" ChannelTag="{tag}" Resolution="{resolution}" '
        f'NameOfMeasuredQuantity="" Min="0" Max="0" Unit="" LUTName="Gray" IsLUTInverted="0" '
        f'BytesInc="{inc}" BitInc="0"/>' for tag, inc in channels)
    dims_xml = "".join(_dim(DIM_IDS[a], sizes[a], strides[a], unit="s" if a == "t" else "m",
                            length=img.get("lengths", {}).get(a, 1e-4))
                       for a in "xyztlm" if a in "xy" or sizes[a] > 1)
    tiles_xml = ""
    if img["tiles"]:
        flags = {"FlipX": "0", "FlipY": "0", "SwapXY": "0", **img.get("tile_flags", {})}
        tiles_xml = ('<Attachment Name="TileScanInfo" Application="LAS AF" '
                     + " ".join(f'{k}="{v}"' for k, v in flags.items()) + ">" + "".join(
                         f'<Tile FieldX="{fx}" FieldY="{fy}" PosX="{px}" PosY="{py}"/>'
                         for fx, fy, px, py in img["tiles"]) + "</Attachment>")
    setting_xml = ""
    if img.get("setting") is not None:          # as LAS X writes it (SystemTypeName on the attachment)
        attrs = " ".join(f"{k}={quoteattr(str(v))}" for k, v in img["setting"].items())
        setting_xml = ('<Attachment Name="HardwareSetting" Application="LAS AF" '
                       f'SystemTypeName="TCS SP8"><ATLConfocalSettingDefinition {attrs}/>'
                       "</Attachment>")
    stamps_xml = ""
    if img.get("timestamps") is not None:
        stamps = img["timestamps"]
        stamps_xml = (f'<TimeStampList NumberOfTimeStamps="{len(stamps)}">'
                      + " ".join(f"{t:x}" for t in stamps) + "</TimeStampList>")
    xml = (f"<Image TextDescription=\"\"><ImageDescription><Channels>{ch_xml}</Channels>"
           f"<Dimensions>{dims_xml}</Dimensions></ImageDescription>{tiles_xml}{setting_xml}"
           f"{stamps_xml}</Image>")
    return xml, block.tobytes()


def write_lif(path, elements, version=2, file_name="test.lif"):
    """*elements*: images / folders (nested through ``children``)."""
    blocks = []

    def element_xml(el):
        block_id = f"MemBlock_{len(blocks) + 1}"
        if el.get("data") is None:
            data_xml, raw = "<Data><Experiment/></Data>", b""
        else:
            image_xml, raw = _layout(el)
            data_xml = f"<Data>{image_xml}</Data>"
        blocks.append((block_id, raw))
        kids = "".join(element_xml(c) for c in el["children"])
        uid = f" UniqueID={quoteattr(el['unique_id'])}" if el.get("unique_id") else ""
        return (f"<Element Name={quoteattr(el['name'])} Visibility=\"1\" CopyOption=\"1\"{uid}>"
                f"{data_xml}<Memory Size=\"{len(raw)}\" MemoryBlockID=\"{block_id}\"/>"
                f"<Children>{kids}</Children></Element>")

    root = folder(file_name, elements)
    xml = f'<LMSDataContainerHeader Version="{version}">{element_xml(root)}</LMSDataContainerHeader>'
    text = xml.encode("utf-16-le")
    out = [struct.pack("<iiBi", 0x70, 1 + 4 + len(text), 0x2A, len(text) // 2), text]
    size_fmt = "<q" if version >= 2 else "<i"
    for block_id, raw in blocks:
        name = block_id.encode("utf-16-le")
        rest = (struct.pack("<B", 0x2A) + struct.pack(size_fmt, len(raw))
                + struct.pack("<Bi", 0x2A, len(name) // 2) + name)
        out += [struct.pack("<ii", 0x70, len(rest)), rest, raw]
    with open(path, "wb") as fh:
        for part in out:
            fh.write(part)


# -- the other flavours: LOF, LIFEXT, XLIF (+ frame files), XLEF / XLCF ------------------
# (structure as liffile, Gohlke, BSD-3, reads them; liffile is the reference in the tests)

LOF_TEXT = "LMS_Object_File"


def _block_header(text: str) -> bytes:
    raw = text.encode("utf-16-le")
    return struct.pack("<iiBi", 0x70, 1 + 4 + len(raw), 0x2A, len(raw) // 2) + raw


def _single_element(img, memory_xml, image_xml):
    return (f"<Element Name={quoteattr(img['name'])} Visibility=\"1\" CopyOption=\"1\">"
            f"<Data>{image_xml}</Data>{memory_xml}<Children/></Element>")


def write_lof(path, img, legacy=False):
    """One image as a LOF file: header block 'LMS_Object_File', two version
    fields, the memory block, then the XML block.  *legacy*: XML that is
    only <Data>...</Data> (older LOF files)."""
    image_xml, raw = _layout(img)
    if legacy:
        xml = f"<Data>{image_xml}</Data>"
    else:
        memory = f'<Memory Size="{len(raw)}" MemoryBlockID="MemBlock_1"/>'
        xml = (f'<LMSDataContainerHeader Version="2">{_single_element(img, memory, image_xml)}'
               "</LMSDataContainerHeader>")
    with open(path, "wb") as fh:
        fh.write(_block_header(LOF_TEXT))
        fh.write(struct.pack("<BIBI", 0x2A, 1, 0x2A, 2))
        fh.write(struct.pack("<BQ", 0x2A, len(raw)))
        fh.write(raw)
        fh.write(_block_header(xml))


def write_lifext(path, groups):
    """A LIFEXT file: *groups* = [(MemoryBlockID of the LIF element they
    extend, [images])]."""
    blocks, parts = [], []
    for group_id, images in groups:
        els = []
        for img in images:
            image_xml, raw = _layout(img)
            block_id = f"MemBlock_{900 + len(blocks)}"
            blocks.append((block_id, raw))
            memory = f'<Memory Size="{len(raw)}" MemoryBlockID="{block_id}"/>'
            els.append(_single_element(img, memory, image_xml))
        parts.append(f'<ChildrenOf MemoryBlockID="{group_id}">{"".join(els)}</ChildrenOf>')
    xml = f'<LMSDataContainerEnhancedHeader Version="2">{"".join(parts)}</LMSDataContainerEnhancedHeader>'
    with open(path, "wb") as fh:
        fh.write(_block_header(xml))
        for block_id, raw in blocks:
            name = block_id.encode("utf-16-le")
            rest = (struct.pack("<BqBi", 0x2A, len(raw), 0x2A, len(name) // 2) + name)
            fh.write(struct.pack("<ii", 0x70, len(rest)) + rest + raw)


def _xml_file(path, xml, encoding):
    head = f'<?xml version="1.0" encoding="{"utf-16" if "16" in encoding else "utf-8"}"?>'
    with open(path, "wb") as fh:
        fh.write((head + xml).encode(encoding))


def write_xlif(path, img, frames="png", encoding="utf-8-sig"):
    """One image as an XLIF with its memory block in frame files next to it
    (folder '<stem>_frames', referenced with backslashes as LAS X on Windows
    writes them): one file per plane ('png', 'jpg' lossless, 'bmp', 'tif'),
    one multi-page TIFF ('ometif') or one LOF ('lof') for the whole block.
    Default layout only (planes contiguous)."""
    import imagecodecs
    import tifffile
    assert img["order"] == DEFAULT_ORDER and not img["row_pad"]
    image_xml, raw = _layout(img)
    data, axes = img["data"], img["axes"]
    ns = data.shape[-1] if "s" in axes else 1
    ny, nx = data.shape[axes.index("y")], data.shape[axes.index("x")]
    dtype = data.dtype
    plane_bytes = ny * nx * ns * dtype.itemsize
    stem = os.path.splitext(os.path.basename(path))[0]
    folder_name = f"{stem}_frames"
    os.makedirs(os.path.join(os.path.dirname(path), folder_name), exist_ok=True)
    planes = np.frombuffer(raw, dtype.newbyteorder("<")).reshape(
        (-1, ny, nx, ns) if ns > 1 else (-1, ny, nx))
    refs = []                                            # (relative file, offset, size)
    if frames in ("png", "jpg", "bmp", "tif"):
        for k, plane in enumerate(planes):
            name = f"plane_{k:03d}.{frames}"
            target = os.path.join(os.path.dirname(path), folder_name, name)
            if frames == "tif":
                tifffile.imwrite(target, plane, photometric="rgb" if ns > 1 else "minisblack")
            else:
                encode = {"png": imagecodecs.png_encode, "bmp": imagecodecs.bmp_encode,
                          "jpg": lambda a: imagecodecs.jpeg8_encode(a, lossless=True)}[frames]
                with open(target, "wb") as fh:
                    fh.write(encode(plane))
            refs.append((f"{folder_name}\\{name}", k * plane_bytes, plane_bytes))
    elif frames == "ometif":
        name = f"{stem}.ome.tif"
        tifffile.imwrite(os.path.join(os.path.dirname(path), folder_name, name), planes,
                         photometric="rgb" if ns > 1 else "minisblack")
        refs.append((f"{folder_name}\\{name}", 0, len(raw)))
    elif frames == "lof":
        name = f"{stem}.lof"
        write_lof(os.path.join(os.path.dirname(path), folder_name, name), img)
        refs.append((f"{folder_name}\\{name}", 0, len(raw)))
    else:
        raise ValueError(frames)
    frame_xml = "".join(f'<Frame File={quoteattr(f)} Offset="{o}" Size="{s}" UUID=""/>'
                        for f, o, s in refs)
    memory = f'<Memory Size="{len(raw)}" MemoryBlockID="MemBlock_1">{frame_xml}</Memory>'
    _xml_file(path, f'<LMSDataContainerHeader Version="2">{_single_element(img, memory, image_xml)}'
                    "</LMSDataContainerHeader>", encoding)


def write_xlef(path, refs, name="experiment", collection=False, encoding="utf-8-sig"):
    """An XLEF (experiment) or XLCF (collection) referencing *refs*
    (paths relative to *path*'s folder)."""
    kind = "<Collection/>" if collection else "<Experiment/>"
    children = "".join(f'<Reference Name={quoteattr(os.path.splitext(os.path.basename(r))[0])} '
                       f'File={quoteattr(r)} UUID=""/>' for r in refs)
    xml = (f'<LMSDataContainerHeader Version="2"><Element Name={quoteattr(name)}>'
           f'<Data>{kind}</Data><Memory Size="0" MemoryBlockID=""/>'
           f"<Children>{children}</Children></Element></LMSDataContainerHeader>")
    _xml_file(path, xml, encoding)
