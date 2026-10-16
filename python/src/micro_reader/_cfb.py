"""Compound File Binary (OLE2) containers: Olympus OIB, later Zeiss ZVI.

Written from Microsoft's published specification [MS-CFB] (Compound File
Binary File Format); oiffile's ``CompoundFile`` (Gohlke, BSD-3) was a
reference.  A container is a little file system: fixed-size sectors, a FAT
chaining them, a directory of storages (folders) and streams (files); streams
smaller than the mini-stream cutoff live in 64-byte mini sectors inside the
root entry's stream.

Only what reading needs: the directory (stream paths ``Storage/name``), small
streams read whole, and for a large stream its *runs* -- (offset in the
stream, offset in the file, length) of each stretch of consecutive sectors --
so that any byte range of the stream is read with positioned reads, never the
whole stream.
"""
from __future__ import annotations

import bisect
import builtins
import struct

from ._errors import CorruptFileError

MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
FREESECT, ENDOFCHAIN, FATSECT, DIFSECT = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFC
MAXREGSECT = 0xFFFFFFFA
NOSTREAM = 0xFFFFFFFF
STORAGE, STREAM, ROOT = 1, 2, 5
#: directory entries are 128 bytes
DIR_ENTRY = struct.Struct("<64sHBBIII16sIQQIQ")


class Stream:
    """A stream of the container: *size* bytes, read through *runs*
    (stream offset, file offset, length), or held in memory (*data*:
    streams in the mini stream, which are small)."""

    def __init__(self, path: str, size: int, runs=None, data: bytes = None, on_disk: str = None):
        self.path, self.size, self.runs, self.data = path, size, runs, data
        #: the compound file the stream is in
        self.path_on_disk = on_disk
        self._starts = [r[0] for r in runs] if runs else []

    def read_all(self) -> bytes:
        """The whole stream (for small ones: settings, text)."""
        if self.data is not None:
            return self.data
        out = bytearray()
        with builtins.open(self.path_on_disk, "rb") as fh:
            for at, n in self.pieces(0, self.size):
                fh.seek(at)
                out += fh.read(n)
        return bytes(out)

    def pieces(self, offset: int, size: int):
        """(file offset, length) pieces that hold bytes offset:offset+size."""
        if offset < 0 or offset + size > self.size:
            raise CorruptFileError(f"{self.path!r}: bytes {offset}:{offset + size} of a "
                                   f"{self.size}-byte stream")
        k = bisect.bisect_right(self._starts, offset) - 1
        end = offset + size
        while offset < end:
            start, at, length = self.runs[k]
            take = min(end, start + length) - offset
            yield at + offset - start, take
            offset += take
            k += 1


class CompoundFile:
    """The directory of compound file *path*: ``streams`` maps each stream's
    path (``Storage00001/s_C001Z001.tif``, ``/`` between levels) to a
    ``Stream``."""

    def __init__(self, path: str):
        self.path = path
        with builtins.open(path, "rb") as fh:
            head = fh.read(512)
            if head[:8] != MAGIC:
                raise CorruptFileError(f"{path!r} is no compound (OLE) file")
            (_minor, major, byte_order, sector_shift, mini_shift, n_fat, first_dir, self.cutoff,
             first_minifat, n_minifat, first_difat, n_difat) = self._header(head)
            #: version 3 files keep only the low 32 bits of a stream size meaningful
            self._size_mask = 0xFFFFFFFF if major == 3 else (1 << 64) - 1
            if byte_order != 0xFFFE:
                raise CorruptFileError(f"{path!r}: compound file byte order {byte_order:#x}")
            self.sector = 1 << sector_shift
            self.mini = 1 << mini_shift
            fh.seek(0, 2)
            self._file_size = fh.tell()
            difat = list(struct.unpack_from("<109I", head, 76))
            sect = first_difat
            for _ in range(n_difat):
                if sect >= MAXREGSECT:
                    break
                data = self._sector(fh, sect)
                values = struct.unpack(f"<{self.sector // 4}I", data)
                difat.extend(values[:-1])
                sect = values[-1]
            fat_sectors = [s for s in difat[:n_fat] if s < MAXREGSECT]
            fat = []
            for s in fat_sectors:
                fat.extend(struct.unpack(f"<{self.sector // 4}I", self._sector(fh, s)))
            self._fat = fat
            entries = []
            for chunk in self._chain_data(fh, first_dir):
                for k in range(0, len(chunk), 128):
                    entries.append(DIR_ENTRY.unpack_from(chunk, k))
            minifat = []
            if n_minifat and first_minifat < MAXREGSECT:
                for chunk in self._chain_data(fh, first_minifat):
                    minifat.extend(struct.unpack(f"<{len(chunk) // 4}I", chunk))
            self._minifat = minifat
            root = entries[0]
            root_size = root[12] & self._size_mask
            mini_stream = b"".join(self._chain_data(fh, root[11]))[:root_size] if root_size else b""
            self.streams = {}
            self._walk(fh, entries, entries[0][6], "", mini_stream, set())

    @staticmethod
    def _header(head: bytes):
        minor, major, byte_order, sector_shift, mini_shift = struct.unpack_from("<HHHHH", head, 24)
        n_fat, first_dir = struct.unpack_from("<II", head, 44)
        cutoff, first_minifat, n_minifat, first_difat, n_difat = struct.unpack_from(
            "<IIIII", head, 56)
        return (minor, major, byte_order, sector_shift, mini_shift, n_fat, first_dir, cutoff,
                first_minifat, n_minifat, first_difat, n_difat)

    def _sector(self, fh, sect: int) -> bytes:
        fh.seek((sect + 1) * self.sector)
        data = fh.read(self.sector)
        if len(data) != self.sector:
            raise CorruptFileError(f"{self.path!r}: sector {sect} beyond the end of the file")
        return data

    def _chain(self, start: int) -> list:
        chain, sect, seen = [], start, set()
        while sect < MAXREGSECT:
            if sect in seen or sect >= len(self._fat):
                raise CorruptFileError(f"{self.path!r}: broken sector chain at {sect}")
            seen.add(sect)
            chain.append(sect)
            sect = self._fat[sect]
        return chain

    def _chain_data(self, fh, start: int):
        for sect in self._chain(start):
            yield self._sector(fh, sect)

    def _runs(self, start: int, size: int) -> list:
        """(stream offset, file offset, length) of each stretch of consecutive sectors."""
        runs, offset = [], 0
        for sect in self._chain(start):
            at = (sect + 1) * self.sector
            length = min(self.sector, size - offset)
            if length <= 0:
                break
            if runs and runs[-1][1] + runs[-1][2] == at:
                s, a, n = runs[-1]
                runs[-1] = (s, a, n + length)
            else:
                runs.append((offset, at, length))
            offset += length
        if offset < size:
            raise CorruptFileError(f"{self.path!r}: a stream's sectors end before its size")
        return runs

    def _mini_data(self, mini_stream: bytes, start: int, size: int) -> bytes:
        out, sect, seen = bytearray(), start, set()
        while sect < MAXREGSECT and len(out) < size:
            if sect in seen or sect >= len(self._minifat):
                raise CorruptFileError(f"{self.path!r}: broken mini sector chain at {sect}")
            seen.add(sect)
            out += mini_stream[sect * self.mini:(sect + 1) * self.mini]
            sect = self._minifat[sect]
        return bytes(out[:size])

    def _walk(self, fh, entries, sid: int, prefix: str, mini_stream: bytes, seen: set):
        """Every entry of the red-black tree at *sid* (siblings left / right,
        a storage's children below it)."""
        stack = [sid]
        while stack:
            sid = stack.pop()
            if sid == NOSTREAM or sid in seen or sid >= len(entries):
                continue
            seen.add(sid)
            (raw, name_len, kind, _color, left, right, child, _clsid, _state, _times, _times2,
             start, size) = entries[sid]
            size &= self._size_mask
            stack += [left, right]
            name = raw[:max(0, name_len - 2)].decode("utf-16-le", "replace")
            path = prefix + name
            if kind == STORAGE:
                self._walk(fh, entries, child, path + "/", mini_stream, seen)
            elif kind == STREAM:
                if size < self.cutoff:
                    self.streams[path] = Stream(path, size, data=self._mini_data(mini_stream, start, size))
                else:
                    self.streams[path] = Stream(path, size, runs=self._runs(start, size),
                                               on_disk=self.path)
