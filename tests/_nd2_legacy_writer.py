"""A minimal legacy (JPEG 2000) ND2 writer for tests, modelled on the real
files OME publishes (aryeh/*.nd2, CC-BY): JP2 signature, ftyp, jp2h/ihdr,
per-frame VIMD / VCAL XML boxes, one ``LUNK`` JPEG 2000 codestream per
channel per frame, the experiment loops (AIM1), the text description (TINF),
and the box map at the end.  The nd2 library reads these files, which
checks the writer.
"""
from __future__ import annotations

import struct

import numpy as np

JP2_SIGNATURE = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
MAP_SIGNATURE = b"LABORATORY IMAGING ND BOX MAP 00"
COMMENT = b"<!--Laboratory Imaging(http://www.lim.cz) generated xml-->\n"


def _v(tag, runtype, value):
    return f'<{tag} runtype="{runtype}" value="{value}"/>'


def _time_loop(count):
    period = (f'<_00 runtype="RLxExperiment.RLxExpTimeLoop">{_v("uiCount", "lx_uint32", count)}'
              f'{_v("dStart", "double", 0.0)}{_v("dPeriod", "double", 1000.0)}'
              f'{_v("dDuration", "double", 1000.0 * count)}</_00>')
    return (f'{_v("eType", "lx_uint32", 8)}<uLoopPars runtype="CLxListVariant">'
            f'<no_name runtype="RLxExperiment.RLxExpNETimeLoop">{_v("uiCount", "lx_uint32", count)}'
            f'{_v("uiPeriodCount", "lx_uint32", 1)}<pPeriod runtype="CLxListVariant">{period}'
            f'</pPeriod></no_name></uLoopPars>')


def _position_loop(count):
    def points(name, values):
        items = "".join(_v(f"_{i:05d}", "double", v) for i, v in enumerate(values))
        return f'<{name} runtype="CLxListVariant">{items}</{name}>'
    xs = [100.0 * i for i in range(count)]
    return (f'{_v("eType", "lx_uint32", 2)}<uLoopPars runtype="CLxListVariant">'
            f'<no_name runtype="RLxExperiment.RLxExpXYPosLoop">{_v("uiCount", "lx_uint32", count)}'
            f'{points("dPosX", xs)}{points("dPosY", xs)}{points("dPosZ", [0.0] * count)}'
            f'{points("dPFSOffset", [0.0] * count)}{_v("bUseZ", "bool", "false")}'
            f'</no_name></uLoopPars>')


def _z_loop(count):
    return (f'{_v("eType", "lx_uint32", 4)}<uLoopPars runtype="CLxListVariant">'
            f'<no_name runtype="RLxExperiment.RLxExpZStackLoop">{_v("uiCount", "lx_uint32", count)}'
            f'{_v("dZLow", "double", 0.0)}{_v("dZHigh", "double", float(count - 1))}'
            f'{_v("dZHome", "double", 0.0)}{_v("dZStep", "double", 1.0)}'
            f'{_v("bAbsolute", "bool", "true")}</no_name></uLoopPars>')


def _loops_xml(loops):
    """Nested experiment levels, outermost first (NextLevelEx)."""
    make = {"T": _time_loop, "P": _position_loop, "Z": _z_loop}
    inner = ""
    for axis, count in reversed(loops):
        next_level = (f'<ppNextLevelEx runtype="CLxListVariant"><_00 runtype="RLxExperiment">'
                      f'{inner}</_00></ppNextLevelEx>') if inner else ""
        inner = make[axis](count) + next_level
    return (f'<Metadata_V1.2 version="1.0"><variant version="1.0">'
            f'<no_name runtype="CLxListVariant"><vMetadata runtype="RLxExperiment">{inner}'
            f'</vMetadata></no_name></variant></Metadata_V1.2>')


def _frame_xml(index, channels, per_channel):
    planes = "".join(
        f'<a{c} runtype="RLxPicturePlaneDescV1_1">{_v("uiCompCount", "lx_uint32", per_channel)}'
        f'{_v("sOpticalConfigName", "CLxStringW", f"ch{c}")}{_v("uiColor", "lx_uint32", 255)}</a{c}>'
        for c in range(channels))
    return (f'<MetadataSeq _SEQUENCE_INDEX="{index}"><variant version="1.0">'
            f'<no_name runtype="RLxPictureMetadata">{_v("dTimeMSec", "double", float(index))}'
            f'{_v("dCalibration", "double", 0.5)}{_v("bCalibrated", "bool", "true")}'
            f'<sPicturePlanes runtype="RLxPicturePlanes">{_v("uiCount", "lx_uint32", channels)}'
            f'{_v("uiCompCount", "lx_uint32", channels * per_channel)}'
            f'<sPlane runtype="CLxListVariant">{planes}</sPlane></sPicturePlanes>'
            f'</no_name></variant></MetadataSeq>')


def _description(loops, channels):
    names = {"T": "T'", "P": "XY", "Z": "Z"}
    parts = [f"{names[a]}({n})" for a, n in loops] + ([f"λ({channels})"] if channels > 1 else [])
    text = "Metadata:\r\nDimensions: " + " x ".join(parts) + "\r\n"
    text = text.replace("\r", "&#x000d;").replace("\n", "&#x000a;")
    return (f'<TextInfo Count="6"><TextInfoItem Text="" Index="0"/>'
            f'<TextInfoItem Text="{text}" Index="5"/></TextInfo>')


def write_legacy_nd2(path, frames, loops, *, extra_frames=0):
    """*frames*: (n_frames, channels, height, width[, 3]) uint8 / uint16, frame
    order = the loops' (outermost first); *loops*: [(axis 'T'/'P'/'Z',
    count)].  *extra_frames*: frames stored beyond the loops (as seen in
    real files), filled with 7s."""
    import imagecodecs
    frames = np.asarray(frames)
    n, channels, height, width = frames.shape[:4]
    per_channel = frames.shape[4] if frames.ndim == 5 else 1
    assert n == int(np.prod([c for _, c in loops] or [1]))
    if extra_frames:
        frames = np.concatenate([frames, np.full((extra_frames,) + frames.shape[1:], 7,
                                                 frames.dtype)])
    bits = 8 * frames.dtype.itemsize
    out = bytearray(JP2_SIGNATURE)
    entries = [(b"jP  ", b"LUNK", 0)]

    def box(kind, nikon, payload):
        entries.append((kind, nikon, len(out)))
        out.extend(struct.pack(">I4s", 8 + len(payload), kind) + payload)

    box(b"ftyp", b"LUNK", b"jp2 \x00\x00\x00\x00jp2 ")
    ihdr = struct.pack(">I4siihBBBB", 22, b"ihdr", height, width, per_channel, bits - 1, 7, 0, 0)
    box(b"jp2h", b"LUNK", ihdr)
    for index, frame in enumerate(frames):
        for c in range(channels):
            stream = imagecodecs.jpeg2k_encode(frame[c], level=0, codecformat="J2K")
            box(b"jp2c", b"LUNK", bytes(stream))
        box(b"xml ", b"VCAL", COMMENT + f'<Calib _SEQUENCE_INDEX="{index}"/>'.encode())
        box(b"xml ", b"VIMD", COMMENT + _frame_xml(index, channels, per_channel).encode())
    box(b"xml ", b"TINF", COMMENT + _description(loops, channels).encode())
    # a single frame: a one-element time loop (nd2 needs an experiment; loops
    # of one element are left out of the dimensions by both readers)
    box(b"xml ", b"AIM1", COMMENT + _loops_xml(loops or [("T", 1)]).encode())
    map_at = len(out)
    out.extend(struct.pack(">I", len(entries)))
    for kind, nikon, offset in entries:
        out.extend(struct.pack("<4s4sII", kind, nikon, offset, 0))
    out.extend(MAP_SIGNATURE + struct.pack("<Q", len(out) - map_at + 40))
    with open(path, "wb") as fh:
        fh.write(out)
