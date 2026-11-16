# Memory and threads

A read returns the requested region as a NumPy array. Its memory requirements depend on the size of that region, not the size of the image file. Only the strips, tiles, subblocks, or chunks that overlap the requested region are read and decoded.

This page explains how decoding uses memory and describes the three process-wide settings that control it.

## The settings at a glance

| Setting | Default | Get | Set | Environment variable |
| --- | --- | --- | --- | --- |
| Memory limit for decoding | ¼ of system RAM (minimum 64 MB) | `memory_limit()` | `set_memory_limit(x)` | `MICRO_READER_MEMORY_LIMIT` |
| Decoded-block cache limit | 256 MB | `cache_limit()`, `cached_bytes()` | `set_cache_limit(x)` | `MICRO_READER_CACHE_LIMIT` |
| Decode threads | Number of CPUs | `decode_threads()` | `set_decode_threads(n)` | `MICRO_READER_DECODE_THREADS` |

Memory sizes can be specified as a number of bytes or a string with binary units, such as `"8GB"`, `"512 MB"`, `"1.5g"`, or `"64MiB"`. `GB` means 2³⁰ bytes, the same as `GiB`.

Passing `None` to `set_memory_limit()` or `set_cache_limit()` restores the default. For each setting, the default is taken from its environment variable if set, or from the built-in value otherwise.

```python
import micro_reader

micro_reader.set_memory_limit("8GB")
micro_reader.set_cache_limit("512MB")
micro_reader.set_decode_threads(4)

micro_reader.memory_limit()    # 8589934592
micro_reader.cached_bytes()    # current cache usage
```

You can also configure the settings through environment variables:

```bash
export MICRO_READER_MEMORY_LIMIT=8GB
export MICRO_READER_CACHE_LIMIT=512MB
export MICRO_READER_DECODE_THREADS=4
```

All three settings apply to the entire **process**, including every open file and every thread. Each worker process has its own settings, so configure them separately in each worker.

## The memory limit

Compressed image data is stored in blocks, such as TIFF strips and tiles, CZI subblocks, ND2 frames, Imaris chunks, or an entire PNG image. A read decodes only the blocks that overlap the requested region.

Some codecs can decode a large compressed block in parts. These include Deflate, LZW, zstd, JPEG 2000, and JPEG with restart markers. For these codecs, reading a small region does not require keeping the entire decoded block in memory.

Decoding is subject to a process-wide memory limit, configured with `set_memory_limit()`. The limit applies to blocks being decoded concurrently across all threads. If a decode cannot fit within the available budget, it waits for other decodes to finish.

Blocks up to 16 MB are decoded in full. Their decoded size is reserved against the memory limit while decoding. Larger blocks are decoded in parts when the codec supports it, and only the rows needed by the read are retained.

A `MemoryLimitError` is raised only when a single block must be decoded in full and its size exceeds the entire memory limit:

```text
MemoryLimitError: a block must be decoded whole (12,288 MB), more than the memory
limit (8,192 MB). Raise it with micro_reader.set_memory_limit(...) or MICRO_READER_MEMORY_LIMIT
```

!!! note "What the limit covers"
    The limit applies to blocks being decoded concurrently. It does not include the arrays returned by reads. Reading a 20 GB region still requires 20 GB for the result. To keep your own memory use bounded, read the image in smaller regions (see [Reading efficiently](reading.md#reading-efficiently)).

## The decoded-block cache

Decoded blocks are cached for reuse, so neighbouring reads that overlap the same block do not need to decode it again.

All open files share one cache budget, which defaults to 256 MB. Blocks are evicted using a least-recently-used policy. A block larger than one quarter of the cache budget is not retained.

```python
micro_reader.set_cache_limit("1GB")  # more reuse for viewers that pan around
micro_reader.set_cache_limit(0)      # disable caching: less memory, more decoding
```

Decoding uses at most the memory limit plus the cache limit, regardless of how many files are open.

There is one exception: a decoded image such as a PNG or JPEG, up to 256 MB, is retained by its own open file until that file is closed.

## Decode threads

Reads decode their blocks using a shared thread pool. File reads and codec operations release the GIL, allowing a single read to use multiple cores.

The pool is shared by all files and reads. Its number of threads remains fixed regardless of how many reads run concurrently.

Setting `set_decode_threads(1)` disables the pool and performs decoding on the calling thread.

If your application already runs multiple reads in parallel, for example through a writer thread pool or Dask, reduce the number of decode threads to avoid oversubscribing the available cores. For example, on a 16-core machine, you could use 8 writer threads with 2 decode threads each.

## Streamed decoding

Deflate, LZW, zstd, PackBits, and LZMA blocks larger than 16 MB are decoded from the beginning of the block, while retaining only the rows needed by the read. This applies to TIFF strips and tiles, ND2 lossless frames, CZI subblocks, and non-interlaced PNG, among other formats.

Reading a block from top to bottom requires a single pass. Each read can continue from where the previous one stopped. If a read jumps back to an earlier region, as an interactive viewer might, the decoder resumes from a checkpoint for Deflate. For LZW and zstd, the block is decoded again from the beginning.

**Region decoding:** JPEG 2000 and JPEG XL decode only the requested region. JPEG blocks with restart markers decode only the restart intervals that cover the requested rows.

**Whole-block decoding:** Progressive and lossless JPEG, JPEG without restart markers, interlaced PNG, JPEG XR, and CZI zstd with hi-lo packing must be decoded in full. Their decoded size counts against the memory limit.

## Uncompressed data

Uncompressed data does not need decoding. The requested rows are read directly from the file into the result array.

## Tuning recipes

=== "Converter with its own writer threads"

    ```python
    import os
    import micro_reader

    writers = 8
    micro_reader.set_decode_threads(max(1, os.cpu_count() // writers))
    micro_reader.set_memory_limit("4GB")
    ```

=== "Interactive viewer"

    ```python
    micro_reader.set_cache_limit("2GB")  # reuse blocks when panning back
    # Read lower-resolution levels when zoomed out
    ```

=== "Small machine or many worker processes"

    ```python
    # Configure each worker process separately
    micro_reader.set_memory_limit("1GB")
    micro_reader.set_cache_limit("128MB")
    micro_reader.set_decode_threads(2)
    ```

=== "Cluster job via environment"

    ```bash
    export MICRO_READER_MEMORY_LIMIT=$((SLURM_MEM_PER_NODE / 4))MB
    export MICRO_READER_DECODE_THREADS=$SLURM_CPUS_PER_TASK
    ```