"""The decoded-block caches of a process share one byte limit (``set_cache_limit``):
many open files do not hold many times the memory one file's cache may."""
from __future__ import annotations

import numpy as np
import pytest

import micro_reader
from micro_reader import _io
from micro_reader._io import DecodeCache

MB = 1 << 20


@pytest.fixture(autouse=True)
def _fresh_limit():
    micro_reader.set_cache_limit(0)          # empty: other tests' files share the caches
    micro_reader.set_cache_limit(None)
    yield
    micro_reader.set_cache_limit(None)


def _seg(mb):
    return np.zeros(int(mb * MB), dtype=np.uint8)


def test_many_caches_share_one_limit():
    micro_reader.set_cache_limit("40MB")
    caches = [DecodeCache(64 * MB) for _ in range(10)]        # 10 files, 64 MB each
    for c in caches:
        for k in range(4):
            c.put(k, _seg(4))                                  # 16 MB per file: 160 MB
    assert micro_reader.cached_bytes() <= 40 * MB
    assert sum(c._bytes for c in caches) == micro_reader.cached_bytes()
    # least recently used went first: the last file still has its blocks
    assert caches[-1].get(3) is not None and caches[0].get(0) is None


def test_reading_a_block_keeps_it():
    micro_reader.set_cache_limit("12MB")
    a, b = DecodeCache(64 * MB), DecodeCache(64 * MB)
    a.put("x", _seg(4))
    b.put("y", _seg(4))
    a.get("x")                                                 # x is now the most recent
    b.put("z", _seg(4))
    b.put("w", _seg(4))                                        # evicts y, not x
    assert a.get("x") is not None and b.get("y") is None


def test_the_per_cache_cap_still_holds():
    c = DecodeCache(8 * MB)
    for k in range(4):
        c.put(k, _seg(2))
    assert c._bytes <= 8 * MB and micro_reader.cached_bytes() == c._bytes


def test_lowering_the_limit_evicts_and_clear_releases():
    caches = [DecodeCache(64 * MB) for _ in range(3)]
    for i, c in enumerate(caches):
        c.put(i, _seg(8))
    assert micro_reader.cached_bytes() == 24 * MB
    micro_reader.set_cache_limit("10MB")
    assert micro_reader.cached_bytes() <= 10 * MB
    for c in caches:
        c.clear()
    assert micro_reader.cached_bytes() == 0


def test_zero_keeps_nothing_and_blocks_above_the_limit_are_not_kept():
    micro_reader.set_cache_limit(0)
    c = DecodeCache(64 * MB)
    c.put(0, _seg(1))
    assert c.get(0) is None and micro_reader.cached_bytes() == 0
    micro_reader.set_cache_limit("2MB")
    c.put(1, _seg(3))                                          # larger than the limit
    assert c.get(1) is None


def test_default_and_environment(monkeypatch):
    assert micro_reader.cache_limit() == _io.DEFAULT_CACHE_LIMIT
    monkeypatch.setenv("MICRO_READER_CACHE_LIMIT", "1GB")
    micro_reader.set_cache_limit(None)
    assert micro_reader.cache_limit() == 1 << 30
