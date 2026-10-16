"""A minimal Compound File Binary (OLE2, version 3) writer for tests, after
Microsoft's [MS-CFB]: 512-byte sectors, 64-byte mini sectors below the
4096-byte cutoff, one level of storages.  *interleave* allocates the sectors
of the large streams in turn, so their chains are fragmented (several runs)."""
from __future__ import annotations

import struct

SECTOR, MINI, CUTOFF = 512, 64, 4096
FREESECT, ENDOFCHAIN, FATSECT, NOSTREAM = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF


def _pad(data: bytes, size: int) -> bytes:
    return data + b"\x00" * (-len(data) % size)


def _entry(name, kind, left=NOSTREAM, right=NOSTREAM, child=NOSTREAM, start=0, size=0) -> bytes:
    raw = name.encode("utf-16-le")
    return struct.pack("<64sHBBIII16sIQQIQ", raw.ljust(64, b"\x00"), len(raw) + 2, kind, 1,
                       left, right, child, b"\x00" * 16, 0, 0, 0, start, size)


def write_cfb(path, streams: dict, interleave: bool = False) -> None:
    """*streams*: {"Storage/name" or "name": bytes}."""
    small = {p: d for p, d in streams.items() if len(d) < CUTOFF}
    large = {p: d for p, d in streams.items() if len(d) >= CUTOFF}

    # the mini stream and its FAT
    mini_data, mini_start, minifat = bytearray(), {}, []
    for p, d in small.items():
        first = len(mini_data) // MINI
        n = max(1, -(-len(d) // MINI))
        mini_start[p] = first
        mini_data += _pad(d, MINI) if d else b"\x00" * MINI
        minifat += [first + k + 1 for k in range(n - 1)] + [ENDOFCHAIN]

    # directory entries: root, storages, streams
    storages = sorted({p.split("/")[0] for p in streams if "/" in p})
    names = ["Root Entry"] + storages + list(streams)
    n_dir_sectors = -(-len(names) * 128 // SECTOR)

    # sectors in order: [mini stream] [minifat] [directory] [large streams]; FAT after
    chains, sectors = {}, []

    def allocate(key, data):
        chains[key] = []
        for k in range(0, max(len(data), 1), SECTOR):
            chains[key].append(len(sectors))
            sectors.append(_pad(data[k:k + SECTOR], SECTOR))

    allocate("ministream", bytes(mini_data))
    allocate("minifat", struct.pack(f"<{len(minifat)}I", *minifat) if minifat else b"")
    chains["dir"] = list(range(len(sectors), len(sectors) + n_dir_sectors))
    sectors += [b""] * n_dir_sectors                     # filled below
    if interleave and len(large) > 1:
        pieces = {p: [d[k:k + SECTOR] for k in range(0, len(d), SECTOR)] for p, d in large.items()}
        for p in large:
            chains[p] = []
        while any(pieces.values()):
            for p in large:
                if pieces[p]:
                    chains[p].append(len(sectors))
                    sectors.append(_pad(pieces[p].pop(0), SECTOR))
    else:
        for p, d in large.items():
            allocate(p, d)

    entries = []
    top_first = (storages + [p for p in streams if "/" not in p])[:1]
    root_child = 1 + names[1:].index(top_first[0]) if top_first else NOSTREAM
    entries.append(_entry("Root Entry", 5, child=root_child, start=chains["ministream"][0],
                          size=len(mini_data)))
    top = storages + [p for p in streams if "/" not in p]
    index = {n: k for k, n in enumerate(names)}
    for k, name in enumerate(storages):
        children = [p for p in streams if p.startswith(name + "/")]
        child = index[children[0]] if children else NOSTREAM
        right = index[top[top.index(name) + 1]] if top.index(name) + 1 < len(top) else NOSTREAM
        entries.append(_entry(name, 1, right=right, child=child))
    for p, d in streams.items():
        storage, _, leaf = p.rpartition("/")
        siblings = [q for q in streams if q.rpartition("/")[0] == storage] if storage else top
        pos = siblings.index(p if storage else p)
        right = index[siblings[pos + 1]] if pos + 1 < len(siblings) else NOSTREAM
        start = mini_start[p] if p in small else chains[p][0]
        entries.append(_entry(leaf, 2, right=right, start=start, size=len(d)))
    directory = _pad(b"".join(entries), SECTOR)
    for k, sect in enumerate(chains["dir"]):
        sectors[sect] = directory[k * SECTOR:(k + 1) * SECTOR]

    # the FAT (with its own sectors at the end)
    n_data = len(sectors)
    n_fat = 1
    while n_fat * SECTOR // 4 < n_data + n_fat:
        n_fat += 1
    fat = [FREESECT] * (n_fat * SECTOR // 4)
    for key, chain in chains.items():
        if key == "minifat" and not minifat:
            continue
        for a, b in zip(chain, chain[1:] + [ENDOFCHAIN]):
            fat[a] = b
    fat_sectors = list(range(n_data, n_data + n_fat))
    for s in fat_sectors:
        fat[s] = FATSECT
    fat_bytes = struct.pack(f"<{len(fat)}I", *fat)

    head = bytearray(SECTOR)
    head[:8] = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    struct.pack_into("<HHHHH", head, 24, 0x3E, 3, 0xFFFE, 9, 6)
    struct.pack_into("<III", head, 40, 0, n_fat, chains["dir"][0])
    struct.pack_into("<IIIII", head, 56, CUTOFF, chains["minifat"][0] if minifat else ENDOFCHAIN,
                     -(-len(minifat) * 4 // SECTOR) if minifat else 0, ENDOFCHAIN, 0)
    difat = fat_sectors + [FREESECT] * (109 - len(fat_sectors))
    struct.pack_into("<109I", head, 76, *difat)
    with open(path, "wb") as fh:
        fh.write(head)
        for s in sectors:
            fh.write(s)
        fh.write(fat_bytes)
