//! Streaming decoders: a compressed block decoded from its start, keeping only
//! the bytes a read asks for (micro-reader PLAN.md, "memory-bounded reads of
//! huge compressed blocks").
//!
//! A `StreamDecoder` reads the block's compressed bytes from the file itself,
//! 64 KiB at a time, so neither the compressed nor the decoded block is ever
//! held whole: memory is the codec's window (32 KiB for deflate, the frame's
//! window for zstd, the 4096-entry table for LZW) plus the caller's buffer.
//! It keeps its place (a cursor): a read further on continues where the last
//! one stopped.  For deflate / zlib, a ``BlockIndex`` shared by a block's
//! decoders keeps checkpoints -- the inflater's state every N decoded bytes --
//! so a read before the cursor (or far ahead of it) resumes from the nearest
//! one instead of starting the block again.  Other codecs start again.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read, Seek, SeekFrom, Take};
use std::sync::{Arc, Mutex};

use miniz_oxide::inflate::stream::{inflate, InflateState};
use miniz_oxide::{DataFormat, MZError, MZFlush, MZStatus};

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyOSError, PyValueError};
use pyo3::prelude::*;

const CHUNK: usize = 1 << 16;

type Source = BufReader<Take<File>>;

fn open_source(path: &str, offset: u64, length: u64) -> io::Result<Source> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(offset))?;
    Ok(BufReader::with_capacity(CHUNK, file.take(length)))
}

/// TIFF-style LZW (MSB first, early code-size switch) as a `Read`.
struct LzwReader {
    decoder: weezl::decode::Decoder,
    src: Source,
    done: bool,
}

impl Read for LzwReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.done || out.is_empty() {
            return Ok(0);
        }
        loop {
            let input = self.src.fill_buf()?;
            let eof = input.is_empty();
            let result = self.decoder.decode_bytes(input, out);
            self.src.consume(result.consumed_in);
            match result.status {
                Ok(weezl::LzwStatus::Done) => {
                    self.done = true;
                    return Ok(result.consumed_out);
                }
                Ok(weezl::LzwStatus::Ok) => {
                    if result.consumed_out > 0 {
                        return Ok(result.consumed_out);
                    }
                    if eof {
                        // TIFF writers may omit the end code: the data simply stops
                        self.done = true;
                        return Ok(0);
                    }
                }
                Ok(weezl::LzwStatus::NoProgress) => {
                    self.done = true;
                    return Ok(result.consumed_out);
                }
                Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("lzw: {e}"))),
            }
        }
    }
}

/// PackBits (Apple / TIFF run-length) as a `Read`.
struct PackBitsReader {
    src: Source,
    literal: usize,
    run: usize,
    run_byte: u8,
}

impl Read for PackBitsReader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        loop {
            if self.literal > 0 {
                let want = self.literal.min(out.len());
                let n = self.src.read(&mut out[..want])?;
                if n == 0 {
                    self.literal = 0;
                    return Ok(0);
                }
                self.literal -= n;
                return Ok(n);
            }
            if self.run > 0 {
                let n = self.run.min(out.len());
                out[..n].fill(self.run_byte);
                self.run -= n;
                return Ok(n);
            }
            let mut header = [0u8; 1];
            if self.src.read(&mut header)? == 0 {
                return Ok(0);
            }
            let n = header[0] as i8;
            if n >= 0 {
                self.literal = n as usize + 1;
            } else if n != -128 {
                let mut byte = [0u8; 1];
                if self.src.read(&mut byte)? == 0 {
                    return Ok(0);
                }
                self.run = (1 - n as isize) as usize;
                self.run_byte = byte[0];
            }
        }
    }
}

/// A file seen from *base* on: position 0 is the file's byte *base* (a PNG
/// inside a BMP starts there).
struct Window {
    file: File,
    base: u64,
}

impl Read for Window {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        self.file.read(out)
    }
}

impl Seek for Window {
    fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
        let at = match to {
            SeekFrom::Start(n) => self.file.seek(SeekFrom::Start(self.base + n))?,
            other => self.file.seek(other)?,
        };
        Ok(at.saturating_sub(self.base))
    }
}

/// The rows of a PNG (not interlaced), as stored: packed below 8 bits and for
/// palette indices, 16-bit samples little-endian (PNG stores them big-endian).
struct PngRows {
    reader: png::Reader<BufReader<Window>>,
    pending: Vec<u8>,
    at: usize,
    swap16: bool,
}

impl PngRows {
    fn open(path: &str, offset: u64) -> io::Result<Self> {
        let mut file = File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let window = Window { file, base: offset };
        let mut decoder = png::Decoder::new(BufReader::with_capacity(CHUNK, window));
        decoder.set_transformations(png::Transformations::IDENTITY);
        let reader = decoder.read_info()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("png: {e}")))?;
        if reader.info().interlaced {
            return Err(io::Error::new(io::ErrorKind::InvalidInput,
                                      "png: interlaced images are not streamed by row"));
        }
        let swap16 = reader.info().bit_depth == png::BitDepth::Sixteen;
        Ok(Self { reader, pending: Vec::new(), at: 0, swap16 })
    }
}

impl Read for PngRows {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        loop {
            if self.at < self.pending.len() {
                let n = (self.pending.len() - self.at).min(out.len());
                out[..n].copy_from_slice(&self.pending[self.at..self.at + n]);
                self.at += n;
                return Ok(n);
            }
            match self.reader.next_row() {
                Ok(Some(row)) => {
                    self.pending.clear();
                    self.pending.extend_from_slice(row.data());
                    if self.swap16 {
                        for pair in self.pending.chunks_exact_mut(2) {
                            pair.swap(0, 1);
                        }
                    }
                    self.at = 0;
                }
                Ok(None) => return Ok(0),
                Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("png: {e}"))),
            }
        }
    }
}

/// Raw deflate or zlib, decoded with miniz_oxide's streaming inflater, whose
/// state can be copied: a checkpoint is that state plus where in the
/// compressed and decoded bytes it was taken.
struct Deflate {
    src: Source,
    state: Box<InflateState>,
    /// compressed bytes consumed so far
    in_pos: u64,
    done: bool,
}

impl Deflate {
    fn open(path: &str, offset: u64, length: u64, zlib: bool) -> io::Result<Self> {
        let format = if zlib { DataFormat::Zlib } else { DataFormat::Raw };
        Ok(Self { src: open_source(path, offset, length)?, state: InflateState::new_boxed(format),
                  in_pos: 0, done: false })
    }

    fn resume(path: &str, offset: u64, length: u64, point: &Checkpoint) -> io::Result<Self> {
        let src = open_source(path, offset + point.in_pos, length - point.in_pos)?;
        Ok(Self { src, state: point.state.clone(), in_pos: point.in_pos, done: false })
    }
}

impl Read for Deflate {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.done || out.is_empty() {
            return Ok(0);
        }
        loop {
            let input = self.src.fill_buf()?;
            let eof = input.is_empty();
            let result = inflate(&mut self.state, input, out, MZFlush::None);
            self.src.consume(result.bytes_consumed);
            self.in_pos += result.bytes_consumed as u64;
            match result.status {
                Ok(MZStatus::StreamEnd) => {
                    self.done = true;
                    return Ok(result.bytes_written);
                }
                Ok(_) | Err(MZError::Buf) => {
                    if result.bytes_written > 0 {
                        return Ok(result.bytes_written);
                    }
                    if eof || (result.bytes_consumed == 0 && result.status.is_err()) {
                        self.done = true; // the data simply stops (a short segment)
                        return Ok(0);
                    }
                }
                Err(e) => {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, format!("inflate: {e:?}")))
                }
            }
        }
    }
}

/// A decoder's state at one point of a block, to resume from.
struct Checkpoint {
    out_pos: u64,
    in_pos: u64,
    state: Box<InflateState>,
}

/// The checkpoints of one block, shared by all its decoders: any decoder
/// records one every `every` decoded bytes as it passes (at most `max`),
/// and any decoder resumes from the nearest one before where it must start.
pub struct IndexInner {
    every: u64,
    max: usize,
    points: Mutex<Vec<Arc<Checkpoint>>>, // sorted by out_pos
}

impl IndexInner {
    fn best(&self, start: u64) -> Option<Arc<Checkpoint>> {
        let points = self.points.lock().unwrap_or_else(|p| p.into_inner());
        let k = points.partition_point(|p| p.out_pos <= start);
        if k == 0 { None } else { Some(points[k - 1].clone()) }
    }

    /// Record a checkpoint at *out_pos* unless one is within `every` of it.
    fn record(&self, out_pos: u64, deflate: &Deflate) {
        let mut points = self.points.lock().unwrap_or_else(|p| p.into_inner());
        if points.len() >= self.max {
            return;
        }
        let k = points.partition_point(|p| p.out_pos <= out_pos);
        let near = (k > 0 && out_pos - points[k - 1].out_pos < self.every)
            || (k < points.len() && points[k].out_pos - out_pos < self.every);
        if !near {
            points.insert(k, Arc::new(Checkpoint { out_pos, in_pos: deflate.in_pos,
                                                   state: deflate.state.clone() }));
        }
    }
}

/// The shared checkpoints of one deflate / zlib block (``StreamDecoder(...,
/// index=)``): a checkpoint every *every* decoded bytes, at most *max*.
#[pyclass(module = "micro_reader_codecs")]
pub struct BlockIndex {
    inner: Arc<IndexInner>,
}

#[pymethods]
impl BlockIndex {
    #[new]
    #[pyo3(signature = (every, max = 256))]
    fn new(every: u64, max: usize) -> Self {
        Self { inner: Arc::new(IndexInner { every: every.max(1), max, points: Mutex::new(Vec::new()) }) }
    }

    /// The checkpoints recorded so far.
    fn __len__(&self) -> usize {
        self.inner.points.lock().unwrap_or_else(|p| p.into_inner()).len()
    }
}

enum Reader {
    Deflate(Deflate),
    Other(Box<dyn Read + Send>),
}

impl Reader {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        match self {
            Reader::Deflate(d) => d.read(out),
            Reader::Other(r) => r.read(out),
        }
    }
}

fn open_reader(codec: &str, path: &str, offset: u64, length: u64) -> io::Result<Reader> {
    if codec == "zlib" || codec == "deflate" {
        return Ok(Reader::Deflate(Deflate::open(path, offset, length, codec == "zlib")?));
    }
    if codec == "png" {
        return Ok(Reader::Other(Box::new(PngRows::open(path, offset)?)));
    }
    let src = open_source(path, offset, length)?;
    Ok(Reader::Other(match codec {
        "lzw" => Box::new(LzwReader {
            decoder: weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8),
            src,
            done: false,
        }),
        "zstd" => Box::new(zstd::stream::read::Decoder::with_buffer(src)?),
        "packbits" => Box::new(PackBitsReader { src, literal: 0, run: 0, run_byte: 0 }),
        "none" => Box::new(src),
        other => {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, format!("no streaming codec {other:?}")))
        }
    }))
}

/// Read until `out` is full or the stream ends; the bytes read.
fn read_full(reader: &mut Reader, out: &mut [u8]) -> io::Result<usize> {
    let mut done = 0;
    while done < out.len() {
        match reader.read(&mut out[done..]) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// The decoder's moving parts: the open stream and where it is.
struct State {
    reader: Option<Reader>,
    /// decoded bytes produced so far (the cursor)
    pos: u64,
    restarts: u64,
    resumes: u64,
}

/// One compressed block of a file, decoded as a stream.  Calls on one decoder
/// are serialised (a mutex): micro-reader gives each read its own decoder.
#[pyclass(module = "micro_reader_codecs")]
pub struct StreamDecoder {
    path: String,
    offset: u64,
    length: u64,
    codec: String,
    index: Option<Arc<IndexInner>>,
    state: Mutex<State>,
}

impl StreamDecoder {
    /// Decoded bytes *start*.. into *output*.  The decoder continues from its
    /// cursor, or from a checkpoint nearer to *start* (also one ahead of the
    /// cursor), or starts the block again.
    fn advance(&self, state: &mut State, start: u64, output: &mut [u8]) -> io::Result<usize> {
        let point = self.index.as_ref().and_then(|ix| ix.best(start));
        let behind = state.reader.is_none() || start < state.pos;
        if let Some(point) = point.filter(|p| behind || p.out_pos > state.pos) {
            state.reader = Some(Reader::Deflate(Deflate::resume(&self.path, self.offset, self.length, &point)?));
            state.pos = point.out_pos;
            state.resumes += 1;
        } else if behind {
            if state.reader.is_some() {
                state.restarts += 1;
            }
            state.reader = Some(open_reader(&self.codec, &self.path, self.offset, self.length)?);
            state.pos = 0;
        }
        let index = self.index.clone();
        let reader = state.reader.as_mut().unwrap();
        let every = index.as_ref().map_or(u64::MAX, |ix| ix.every);
        // decode in pieces no longer than the checkpoint spacing, recording
        // a checkpoint after each piece that passes new ground
        let mut scratch = vec![0u8; CHUNK.min((start - state.pos) as usize).max(1)];
        while state.pos < start {
            let want = ((start - state.pos) as usize).min(scratch.len()).min(every as usize);
            let n = read_full(reader, &mut scratch[..want])?;
            state.pos += n as u64;
            if let (Some(ix), Reader::Deflate(d)) = (&index, &*reader) {
                ix.record(state.pos, d);
            }
            if n < want {
                return Ok(0); // the stream ends before *start*
            }
        }
        let mut written = 0;
        while written < output.len() {
            let want = (output.len() - written).min(every as usize);
            let n = read_full(reader, &mut output[written..written + want])?;
            written += n;
            state.pos += n as u64;
            if let (Some(ix), Reader::Deflate(d)) = (&index, &*reader) {
                ix.record(state.pos, d);
            }
            if n < want {
                break;
            }
        }
        Ok(written)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

#[pymethods]
impl StreamDecoder {
    /// The block of *length* bytes at *offset* of the file *path*, compressed
    /// with *codec*: "zlib", "deflate", "lzw", "zstd", "packbits", "none", or
    /// "png" (a whole PNG file from *offset*: its rows as stored, not
    /// interlaced).  *index*: the block's shared checkpoints (zlib / deflate).
    #[new]
    #[pyo3(signature = (path, offset, length, codec, index = None))]
    fn new(path: String, offset: u64, length: u64, codec: String,
           index: Option<PyRef<'_, BlockIndex>>) -> PyResult<Self> {
        if !["zlib", "deflate", "lzw", "zstd", "packbits", "none", "png"].contains(&codec.as_str()) {
            return Err(PyValueError::new_err(format!("no streaming codec {codec:?}")));
        }
        let index = if codec == "zlib" || codec == "deflate" {
            index.map(|ix| ix.inner.clone())
        } else {
            None
        };
        let state = State { reader: None, pos: 0, restarts: 0, resumes: 0 };
        Ok(Self { path, offset, length, codec, index, state: Mutex::new(state) })
    }

    /// The decoded bytes produced so far: where the next read continues.
    #[getter]
    fn position(&self) -> u64 {
        self.lock().pos
    }

    /// How often a read before the cursor started the block again.
    #[getter]
    fn restarts(&self) -> u64 {
        self.lock().restarts
    }

    /// How often a read resumed from a checkpoint.
    #[getter]
    fn resumes(&self) -> u64 {
        self.lock().resumes
    }

    /// Decoded bytes from *start* into *out* (C-contiguous, writable); returns
    /// the bytes written -- fewer than ``len(out)`` where the stream ends.
    fn read_into(&self, py: Python<'_>, start: u64, out: &Bound<'_, PyAny>) -> PyResult<usize> {
        let dst: PyBuffer<u8> = PyBuffer::get(out)?;
        if dst.readonly() {
            return Err(PyBufferError::new_err("output buffer is read-only"));
        }
        if !dst.is_c_contiguous() {
            return Err(PyBufferError::new_err("buffers must be C-contiguous"));
        }
        let (ptr, len) = (dst.buf_ptr() as usize, dst.len_bytes());
        let result = py.detach(|| {
            let output = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) };
            let mut state = self.lock();
            self.advance(&mut state, start, output)
        });
        result.map_err(|e| match e.kind() {
            io::ErrorKind::InvalidData | io::ErrorKind::InvalidInput | io::ErrorKind::UnexpectedEof => {
                PyValueError::new_err(format!("{}: {e}", self.codec))
            }
            _ => PyOSError::new_err(format!("{}: {e}", self.path)),
        })
    }
}
