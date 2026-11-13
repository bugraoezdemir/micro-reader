# micro-reader-core

The native read engine of [micro-reader](https://github.com/bugraoezdemir/micro-reader)
(Python module `micro_reader_core`). It is not meant to be used on its own:
micro-reader depends on it and calls it.

micro-reader keeps a file's metadata, its discovery and its layout in Python, and
builds a read plan once per layout: where each piece of a region lies in the file,
how it is compressed and where it goes in the result. The engine then reads, decodes
and places every piece of a request in one call, on its own threads and without
Python's global lock, so many small pieces cost what they cost in the file and not a
Python call each.

It links the decoders of
[micro-reader-codecs](https://pypi.org/project/micro-reader-codecs/) as a Rust
library. Its wheels are abi3: one per platform for every Python from 3.11.

Building from source needs a Rust toolchain, a C compiler, CMake, and NASM on x86 (for
libjpeg-turbo's SIMD), as micro-reader-codecs does.

MIT licence.
