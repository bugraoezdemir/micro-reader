"""Olympus OIF folders and OIB files for tests, laid out as FV1000 writes them
(checked against the OME FV1000 sample): a UTF-16 INI main file, one TIFF
per plane named by its axes, LUT files, ``OibInfo.txt`` naming the streams
of an OIB."""
from __future__ import annotations

import io
import os

import numpy as np

from _cfb_writer import write_cfb


def _ini(sections: dict) -> bytes:
    lines = []
    for name, values in sections.items():
        lines.append(f"[{name}]")
        lines += [f"{k}={v}" for k, v in values.items()]
    return b"\xff\xfe" + ("\r\n".join(lines) + "\r\n").encode("utf-16-le")


def _plane_tiff(plane: np.ndarray, rows_per_strip=None, compression=None) -> bytes:
    import tifffile
    buf = io.BytesIO()
    tifffile.imwrite(buf, plane, rowsperstrip=rows_per_strip, compression=compression,
                     metadata=None)
    return buf.getvalue()


def _lut(rgb) -> bytes:
    table = np.zeros((256, 4), np.uint8)
    table[:, 0] = np.linspace(0, rgb[2], 256)              # B G R A
    table[:, 1] = np.linspace(0, rgb[1], 256)
    table[:, 2] = np.linspace(0, rgb[0], 256)
    text = "[Blue]\r\nBrightness=0\r\n[ColorLUTData]\r\n".encode("utf-16-le")
    return b"\xff\xfe" + text + table.tobytes()


def oif_files(data, *, pixel="0,207.0", z_interval=510, t_interval=0, names=("CH1", "CH3"),
              emission=(461, 618), excitation=(405, 543), colors=((0, 0, 255), (255, 0, 0)),
              rows_per_strip=None, compression=None, skip=(), extra_axis=None):
    """*data*: (t, c, z, y, x) -> (main settings bytes, {file name: bytes}).
    *skip*: (t, c, z) planes left out."""
    data = np.asarray(data)
    nt, nc, nz, ny, nx = data.shape
    axes = [("X", nx, "um"), ("Y", ny, "um"), ("C", nc, "Ch"), ("Z", nz, "nm"), ("T", nt, "ms")]
    if extra_axis:
        axes.append(extra_axis)
    main = {"Axis Parameter Common": {"AxisCount": 4, "AxisOrder": '"XYCZ"'}}
    for k, (code, size, unit) in enumerate(axes):
        main[f"Axis {k} Parameters Common"] = {
            "AxisCode": f'"{code}"', "MaxSize": size if (code not in "ZT" or size > 1) else 0,
            "PixUnit": f'"{unit}"', "StartPosition": "0.0", "EndPosition": "211,761.0",
            "Interval": z_interval if code == "Z" else t_interval if code == "T" else 0}
    main["Reference Image Parameter"] = {
        "ImageWidth": nx, "ImageHeight": ny, "WidthConvertValue": pixel,
        "HeightConvertValue": pixel, "WidthUnit": '"um"', "HeightUnit": '"um"'}
    for c in range(nc):
        main[f"Channel {c + 1} Parameters"] = {
            "CH Name": f'"{names[c]}"', "DyeName": f'"dye{c}"',
            "EmissionWavelength": emission[c], "ExcitationWavelength": excitation[c]}
    files = {}
    for t in range(nt):
        for c in range(nc):
            for z in range(nz):
                if (t, c, z) in skip:
                    continue
                name = f"s_C{c + 1:03d}" + (f"Z{z + 1:03d}" if nz > 1 else "") + \
                    (f"T{t + 1:03d}" if nt > 1 else "")
                files[name + ".tif"] = _plane_tiff(data[t, c, z], rows_per_strip, compression)
                files[name + ".pty"] = _ini({"Acquisition Parameters Common": {"ZoomValue": "1.0"}})
    for c in range(nc):
        files[f"s_LUT{c + 1}.lut"] = _lut(colors[c])
    files["s_Thumb.bmp"] = b"BM" + b"\x00" * 60
    return _ini(main), files


def write_oif(folder, name, data, **kwargs) -> str:
    main, files = oif_files(data, **kwargs)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".oif")
    with open(path, "wb") as fh:
        fh.write(main)
    storage = path + ".files"
    os.makedirs(storage, exist_ok=True)
    for file_name, content in files.items():
        with open(os.path.join(storage, file_name), "wb") as fh:
            fh.write(content)
    return path


def write_oib(path, data, interleave=False, **kwargs) -> str:
    name = os.path.splitext(os.path.basename(str(path)))[0]
    main, files = oif_files(data, **kwargs)
    streams, info = {}, {"Version": "2.0.0.0", "Name": "COFAImages", "Compression": "None"}
    for k, (file_name, content) in enumerate(files.items(), start=1):
        key = f"Stream{k:05d}"
        streams[f"Storage00001/{key}"] = content
        info[key] = f"Storage00001/{file_name}"
    main_key = f"Stream{len(files) + 1:05d}"
    streams[main_key] = main
    info[main_key] = name + ".oif"
    info["Storage00001"] = name + ".oif.files"
    info["MainFileName"] = main_key
    streams["OibInfo.txt"] = _ini({"OibSaveInfo": info})
    write_cfb(str(path), streams, interleave=interleave)
    return str(path)
