"""MetaMorph ``.nd`` sets for tests: the ``.nd`` text file, MetaSeries TIFF
members (tifffile, one page per z plane, the ``<MetaData>`` XML in each
page's description) and STK members (written here: tifffile reads STK but
does not write it -- one IFD, the planes stored one after the other, the
UIC1 / UIC2 / UIC4 tags).  Layout as on the real samples
(``formats/metamorph``): ``"Key", value`` lines ending ``\\r\\n``, a
multi-line ``Description``, members ``<base>[_w<k><name>][_s<p>][_t<t>].TIF``.
"""
from __future__ import annotations

import os
import struct

import numpy as np


def write_nd(path, *, times=None, stages=None, waves=None, nz=None, z_step=None,
             waves_in_name=True, description="", extra=()):
    """An ``.nd`` file.  *times*: NTimePoints (None: DoTimelapse FALSE);
    *stages*: stage names (None: DoStage FALSE); *waves*: [(name, has z)]
    (None: DoWave FALSE); *nz*: NZSteps (None: DoZSeries FALSE);
    *waves_in_name*: WaveInFileName (None: the key left out); *extra*: more
    (key, raw value) lines before ``EndFile``."""
    lines = ['"NDInfoFile", Version 2.0', '"Description", ' + description.replace("\n", "\r\r\n"),
             '"StartTime1", 20240816 09:49:28']
    lines += ['"DoTimelapse", ' + ("TRUE" if times else "FALSE")]
    if times:
        lines.append(f'"NTimePoints", {times}')
    lines.append('"DoStage", ' + ("TRUE" if stages else "FALSE"))
    if stages:
        lines.append(f'"NStagePositions", {len(stages)}')
        lines += [f'"Stage{k}", "{name}"' for k, name in enumerate(stages, 1)]
    lines.append('"DoWave", ' + ("TRUE" if waves else "FALSE"))
    lines.append(f'"NWavelengths", {len(waves) if waves else 0}')
    for k, (name, doz) in enumerate(waves or (), 1):
        lines += [f'"WaveName{k}", "{name}"', f'"WaveDoZ{k}", ' + ("TRUE" if doz else "FALSE")]
    lines.append('"DoZSeries", ' + ("TRUE" if nz else "FALSE"))
    if nz:
        lines += [f'"NZSteps", {nz}', f'"ZStepSize", {z_step if z_step is not None else 1}']
    if waves_in_name is not None:
        lines.append('"WaveInFileName", ' + ("TRUE" if waves_in_name else "FALSE"))
    lines += [f'"{k}", {v}' for k, v in extra]
    lines += ['"NEvents", 0', '"EndFile"']
    with open(path, "w", encoding="latin-1", newline="") as fh:
        fh.write("\r\n".join(lines) + "\r\n")
    return path


def metaseries_xml(*, calibration=0.65, units="um", z=None, illumination=None,
                   calibrated=True, stage=None) -> str:
    props = [("spatial-calibration-state", "bool", "on" if calibrated else "off"),
             ("spatial-calibration-x", "float", calibration),
             ("spatial-calibration-y", "float", calibration),
             ("spatial-calibration-units", "string", units)]
    if z is not None:
        props.append(("z-position", "float", z))
    if illumination is not None:
        props.append(("_IllumSetting_", "string", illumination))
    if stage is not None:
        props += [("stage-position-x", "float", stage[0]), ("stage-position-y", "float", stage[1])]
    plane = "\n".join(f'<prop id="{k}" type="{t}" value="{v}"/>' for k, t, v in props)
    return ('<MetaData>\n<prop id="Description" type="string" value=""/>\n'
            '<prop id="MetaDataVersion" type="float" value="1"/>\n'
            '<prop id="ApplicationName" type="string" value="MetaMorph"/>\n'
            f"<PlaneInfo>\n{plane}\n</PlaneInfo>\n"
            '<SetInfo>\n<prop id="number-of-planes" type="int" value="1"/>\n</SetInfo>\n'
            "</MetaData>")


def write_metaseries(path, planes, *, z=None, **xml):
    """A MetaSeries TIFF: *planes* (n, y, x), one page each; *z*: the focus
    position of each page (or None)."""
    import tifffile
    planes = np.asarray(planes)
    with tifffile.TiffWriter(path) as tw:
        for k, plane in enumerate(planes):
            tw.write(plane, description=metaseries_xml(z=None if z is None else z[k], **xml),
                     software="MetaSeries", metadata=None, contiguous=False)
    return path


def write_stk(path, planes, *, z=None, calibration=1.25, units="um"):
    """A MetaMorph STK: *planes* (n, y, x) uint16 in one IFD; *z*: the
    planes' AbsoluteZ (UIC4, valid) or None.  Plane times differ, so tifffile
    calls the planes T (the ``.nd`` file says what they are)."""
    planes = np.ascontiguousarray(planes, dtype="<u2")
    n, rows, cols = planes.shape
    pixels = planes.tobytes()
    blobs = bytearray()
    base = 8 + len(pixels)

    def blob(data: bytes) -> int:
        offset = base + len(blobs)
        blobs.extend(data)
        if len(blobs) % 2:
            blobs.append(0)
        return offset

    units_b = units.encode() + b"\0"
    xcal = blob(struct.pack("<2I", int(calibration * 1000), 1000))
    ycal = blob(struct.pack("<2I", int(calibration * 1000), 1000))
    unit_off = blob(struct.pack("<I", len(units_b)) + units_b)
    uic1 = [(4, xcal), (5, ycal), (6, unit_off)]
    uic1_off = blob(b"".join(struct.pack("<2I", k, v) for k, v in uic1))
    uic2 = b"".join(struct.pack("<6I", 0, 1, 2459000, 1000 * (k + 1), 2459000, 1000 * (k + 1))
                    for k in range(n))
    uic2_off = blob(uic2)
    uic4 = b""
    if z is not None:
        uic4 += struct.pack("<H", 40) + b"".join(struct.pack("<2I", int(round(v * 1000)), 1000)
                                                 for v in z)
        uic4 += struct.pack("<H", 41) + struct.pack(f"<{n}I", *([1] * n))
    uic4_off = blob(uic4 + struct.pack("<H", 0))

    # tag: (type, count, value or offset); types 3 SHORT, 4 LONG, 5 RATIONAL
    tags = {256: (4, 1, cols), 257: (4, 1, rows), 258: (3, 1, 16), 259: (3, 1, 1),
            262: (3, 1, 1), 273: (4, 1, 8), 277: (3, 1, 1), 278: (4, 1, rows),
            279: (4, 1, rows * cols * 2), 33628: (4, len(uic1), uic1_off),
            33629: (5, n, uic2_off), 33631: (4, n, uic4_off)}
    ifd_off = base + len(blobs)
    ifd = struct.pack("<H", len(tags))
    for tag in sorted(tags):
        kind, count, value = tags[tag]
        if kind == 3 and count == 1:
            ifd += struct.pack("<HHIHH", tag, kind, count, value, 0)
        else:
            ifd += struct.pack("<HHII", tag, kind, count, value)
    ifd += struct.pack("<I", 0)
    with open(path, "wb") as fh:
        fh.write(b"II*\0" + struct.pack("<I", ifd_off) + pixels + bytes(blobs) + ifd)
    return path


def member_name(base, s=None, t=None, w=None, wave_name="", ext=".TIF"):
    """``<base>[_w<w><wave_name>][_s<s>][_t<t>]<ext>`` (1-based numbers)."""
    return (base + (f"_w{w}{wave_name}" if w is not None else "")
            + (f"_s{s}" if s is not None else "") + (f"_t{t}" if t is not None else "") + ext)


def write_set(folder, base="exp", *, times=None, stages=None, waves=None, nz=None, z_step=2.0,
              z0=100.0, waves_in_name=True, stk=False, skip=(), shape=(8, 12), seed=7,
              illumination=True, stage_xy=None):
    """A whole set: the ``.nd`` file and every member, random uint16 planes.
    *skip*: (s, t, w) (0-based) not written.  *illumination*: the members'
    ``_IllumSetting_`` -- True: the wavelength's name, a string: that, None:
    none.  *stage_xy*: s -> (x, y) the members record.
    -> (nd path, {(s, t, w): (n, y, x) array}, {(s, t, w): member path})."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    nd = write_nd(os.path.join(folder, base + ".nd"), times=times, stages=stages, waves=waves,
                  nz=nz, z_step=z_step, waves_in_name=waves_in_name)
    data, paths = {}, {}
    n_s, n_t, n_w = len(stages or [None]), times or 1, len(waves or [None])
    for s in range(n_s):
        for t in range(n_t):
            for w in range(n_w):
                name, doz = waves[w] if waves else (None, True)
                depth = nz if (nz and doz) else 1
                # a plane without z sits at the middle slice of the stack
                zs = ([z0 + k * z_step for k in range(depth)] if depth > 1 or not nz
                      else [z0 + z_step * ((nz - 1) // 2)])
                fn = member_name(base, s + 1 if stages else None, t + 1 if times else None,
                                 w + 1 if waves and waves_in_name else None, name or "",
                                 ".STK" if stk else ".TIF")
                path = os.path.join(folder, fn)
                planes = rng.integers(1, 60000, (depth,) + tuple(shape)).astype("u2")
                data[(s, t, w)], paths[(s, t, w)] = planes, path
                if (s, t, w) in skip:
                    continue
                if stk:
                    write_stk(path, planes, z=zs)
                else:
                    write_metaseries(path, planes, z=zs,
                                     illumination=name if illumination is True else illumination,
                                     stage=stage_xy(s) if stage_xy else None)
    return nd, data, paths
