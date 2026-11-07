# Memory and threads

micro-reader keeps the memory used for decoding within limits that you set. This page explains the three process-wide settings and how reads use memory.

## The settings at a glance

| | Default | Get | Set | Environment variable |
| --- | --- | --- | --- | --- |
| Memory limit for decoding | ¼ of the computer's RAM (min. 64 MB) | `memory_limit()` | `set_memory_limit(x)` | `MICRO_READER_MEMORY_LIMIT` |
| Cache of decoded blocks | 256 MB | `cache_limit()`, `cached_bytes()` | `set_cache_limit(x)` | `MICRO_READER_CACHE_LIMIT` |
| Decode threads | number of CPUs | `decode_threads()` | `set_decode_threads(n)` | `MICRO_READER_DECODE_THREADS` |

Sizes are a number of bytes or a string with binary units: `"8GB"`, `"512 MB"`, `"1.5g"`, `"64MiB"`. `GB` means 2³⁰ bytes, the same as `GiB`. Passing `None` to `set_memory_limit` or `set_cache_limit` restores the default, which is the environment variable if set and the built-in default otherwise.

```python
import micro_reader

micro_reader.set_memory_limit("8GB")
micro_reader.set_cache_limit("512MB")
micro_reader.set_decode_threads(4)

micro_reader.memory_limit()     # 8589934592
micro_reader.cached_bytes()     # what the caches hold right now
```

```bash
export MICRO_READER_MEMORY_LIMIT=8GB
export MICRO_READER_CACHE_LIMIT=512MB
export MICRO_READER_DECODE_THREADS=4
```

All three apply to the whole **process**: every open file, every thread. Worker processes each have their own, so set them in each worker.

## The memory limit

Compressed data is stored in blocks: TIFF strips and tiles, CZI subblocks, ND2 frames, Imaris chunks, a whole PNG. micro-reader decodes each block in one of two ways.

- **Small blocks (up to 16 MB) are decoded whole.** While it is being decoded, a block reserves its size from the memory limit. The blocks in flight never exceed the limit together, whichever threads decode them. A decode that does not fit waits until others finish.
- **Larger blocks are read in parts** where the codec allows it (see below). Only the rows a read needs are kept.

A block that can only be decoded whole and is larger than the limit raises `MemoryLimitError`, instead of letting the process run out of memory:

```text
MemoryLimitError: a block must be decoded whole (12,288 MB), more than the memory
limit (8,192 MB).  Raise it with micro_reader.set_memory_limit(...) or MICRO_READER_MEMORY_LIMIT
```

!!! note "What the limit covers"
    The limit covers decoding: the blocks being decoded at once. The arrays your reads return are yours, and they are not counted. Reading a 20 GB region still needs 20 GB for the result. To keep your own memory flat, read in pieces (see [Reading efficiently](reading.md#reading-efficiently)).

## The decoded-block cache

Decoded blocks are kept for reuse, so that a block shared by neighbouring reads is not decoded twice. All open files share one cache budget (256 MB by default), and the least recently used blocks are dropped first. A block larger than a quarter of a file's own cache is not kept.

```python
micro_reader.set_cache_limit("1GB")   # more reuse for viewers that pan around
micro_reader.set_cache_limit(0)       # keep nothing: lowest memory, more decoding
```

At most, a process uses the memory limit plus the cache limit for decoding, however many files are open. One exception: a single picture (PNG, JPEG, ... up to 256 MB) is kept decoded by its own open file until you close it.

## Decode threads

A read decodes its blocks on a shared thread pool. Codecs and file reads release the GIL, so one read uses all cores. The pool is shared by every file and every read, so the number of decoding threads stays fixed however many reads run at once.

- `set_decode_threads(1)` decodes on the calling thread, with no pool.
- If your program already runs many reads in parallel (a writer with a thread pool, dask), lower the decode threads so that the two pools together match the cores. For example, 8 writer threads × 2 decode threads on 16 cores.

## Streamed decoding

Deflate, LZW, zstd, PackBits and LZMA blocks larger than 16 MB are decoded from their start, keeping only the rows the read needs. This covers TIFF strips and tiles, ND2 lossless frames, CZI subblocks and non-interlaced PNG, among others.

- Reading a block from top to bottom is a single pass. Each read continues where the last one stopped.
- Jumping back, as a viewer does, resumes from a checkpoint for deflate. For LZW and zstd, the block is decoded again from its start.

**Region decoding.** JPEG 2000 and JPEG XL decode only the requested region. JPEG blocks with restart markers decode only the restart intervals that cover the requested rows.

**Whole-block codecs.** Progressive and lossless JPEG, JPEG without restart markers, interlaced PNG, JPEG XR, and CZI zstd with hi-lo packing can only be decoded whole. They count fully against the memory limit.

## Uncompressed data

Uncompressed data is not decoded. Only the requested rows are read from the file, directly into the result.

## Tuning recipes

=== "Converter with its own writer threads"

    ```python
    import os, micro_reader
    writers = 8
    micro_reader.set_decode_threads(max(1, os.cpu_count() // writers))
    micro_reader.set_memory_limit("4GB")
    ```

=== "Interactive viewer"

    ```python
    micro_reader.set_cache_limit("2GB")    # panning back is free
    # read lower resolution levels when zoomed out
    ```

=== "Small machine or many worker processes"

    ```python
    # per worker process
    micro_reader.set_memory_limit("1GB")
    micro_reader.set_cache_limit("128MB")
    micro_reader.set_decode_threads(2)
    ```

=== "Cluster job via environment"

    ```bash
    export MICRO_READER_MEMORY_LIMIT=$((SLURM_MEM_PER_NODE / 4))MB
    export MICRO_READER_DECODE_THREADS=$SLURM_CPUS_PER_TASK
    ```
