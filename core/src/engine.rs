//! The read engine: one call reads, decodes and places every piece of a read plan.
//!
//! A plan is a table of pieces (one row of `NCOL` integers each, columns below) plus
//! the codec specs and byte tables its rows point to.  A piece is a rectangle of one
//! stored segment (a TIFF strip or tile): rows `y0..y1`, columns `x0..x1` and samples
//! `c0..c1` of a segment `seg_rows` x `seg_cols` x `samples`, written to the
//! destination at address `dst` with byte strides (row, column, sample).
//!
//! - Uncompressed segments (`Kind::Raw`) are read in place: only the rows and
//!   columns asked for, straight into the destination when its memory has the
//!   file's layout.
//! - Compressed segments are read whole, decoded once however many pieces of the
//!   call need them (pieces with the same file and offset), predictors undone and
//!   samples brought to native byte order, kept in the cache when small, and each
//!   piece's rectangle copied out.
//!
//! Every destination rectangle is checked to lie inside the output buffer before
//! anything is written.  Pieces write disjoint memory (the plan's promise: they
//! come from slicing one output array), so they run on the threads in any order.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use micro_reader_codecs::decode::{self, EerParams, JpegColour};
use rayon::prelude::*;

use crate::cache::Cache;
use crate::files::{quoted, Files, Source};

pub const FILE: usize = 0;
pub const OFFSET: usize = 1;
pub const COUNT: usize = 2;
pub const CODEC: usize = 3;
pub const TABLE: usize = 4;
pub const SEG_ROWS: usize = 5;
pub const SEG_COLS: usize = 6;
pub const Y0: usize = 7;
pub const Y1: usize = 8;
pub const X0: usize = 9;
pub const X1: usize = 10;
pub const C0: usize = 11;
pub const C1: usize = 12;
pub const DST: usize = 13;
pub const DST_ROW: usize = 14;
pub const DST_COL: usize = 15;
pub const DST_SAMPLE: usize = 16;
pub const NCOL: usize = 17;

/// Uncompressed reads: bytes read at a time when rows are buffered.
const CHUNK_BYTES: usize = 4 << 20;

/// Work below this (bytes read raw, plus 8 x the bytes of segments to decode) runs
/// on the caller's thread: waking the pool's threads costs more than they save.  The
/// Python path reads a request below 8 MB of raw rows as one task.
const SMALL_WORK: usize = 2 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Raw,
    None,
    Lzw,
    Zlib,
    PackBits,
    Zstd,
    Jpeg,
    Jpeg2k,
    WebP,
    Jxl,
    Png,
    Eer,
    /// JPEG XR (CZI slide scans), through jxrlib
    JpegXr,
    /// CZI's zstd with a header (compression 6): the header's first byte is its
    /// length; hi-lo byte packing of 16-bit samples when its flag is set
    ZstdCzi,
}

impl Kind {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "raw" => Self::Raw,
            "none" => Self::None,
            "lzw" => Self::Lzw,
            "zlib" => Self::Zlib,
            "packbits" => Self::PackBits,
            "zstd" => Self::Zstd,
            "jpeg" => Self::Jpeg,
            "jpeg2k" => Self::Jpeg2k,
            "webp" => Self::WebP,
            "jxl" => Self::Jxl,
            "png" => Self::Png,
            "eer" => Self::Eer,
            "jpegxr" => Self::JpegXr,
            "zstd-czi" => Self::ZstdCzi,
            _ => return None,
        })
    }

    /// Codecs that give the stored bytes (the predictor and byte order apply after).
    fn bytes(self) -> bool {
        matches!(self, Self::None | Self::Lzw | Self::Zlib | Self::PackBits | Self::Zstd
                       | Self::ZstdCzi)
    }
}

/// How one layout's segments decode.
#[derive(Clone, Debug)]
pub struct Codec {
    pub kind: Kind,
    /// bytes per sample (1, 2, 4 or 8)
    pub itemsize: usize,
    /// samples per pixel stored together (TIFF contig samples)
    pub samples: usize,
    /// TIFF predictor: 1 none, 2 horizontal, 3 floating point
    pub predictor: u8,
    /// stored samples are big-endian (byte codecs and raw reads; image codecs give
    /// little-endian)
    pub big_endian: bool,
    pub colour: JpegColour,
    /// JPEG: the table is a header the segment follows (NDPI: header + bytes + EOI)
    /// rather than JPEGTables spliced in front of it
    pub jpeg_header: bool,
    /// EER: (skip, horizontal, vertical) bits
    pub eer: (u32, u32, u32),
    /// raw reads: bytes worth skipping between rows rather than issuing one more
    /// read per row
    pub gap: usize,
    /// the segment holds B, G, R(, A) samples (CZI): placed as R, G, B(, A)
    pub bgr: bool,
}

impl Codec {
    pub fn check(&self) -> Result<(), String> {
        if ![1, 2, 4, 8].contains(&self.itemsize) || self.samples == 0 {
            return Err(format!("codec {:?}: {} samples of {} bytes", self.kind, self.samples,
                               self.itemsize));
        }
        if !matches!(self.predictor, 1..=3)
            || (self.predictor != 1 && !self.kind.bytes()) {
            return Err(format!("codec {:?}: predictor {}", self.kind, self.predictor));
        }
        if self.kind == Kind::Eer {
            let (skip, horz, vert) = self.eer;
            EerParams { height: 1, width: 1, skipbits: skip, horzbits: horz, vertbits: vert,
                        superres: 0, itemsize: self.itemsize }.check()?;
            if self.samples != 1 {
                return Err("codec eer: one sample per pixel".into());
            }
        }
        if self.bgr && !matches!(self.samples, 3 | 4) {
            return Err(format!("codec {:?}: B, G, R order with {} samples", self.kind,
                               self.samples));
        }
        Ok(())
    }

    /// The stored sample that destination sample `c` takes.
    #[inline]
    fn stored(&self, c: usize) -> usize {
        if self.bgr && c < 3 { 2 - c } else { c }
    }

    /// Whether stored samples need their bytes reversed to native order.
    fn swap(&self) -> bool {
        self.itemsize > 1 && self.big_endian != cfg!(target_endian = "big")
    }
}

/// One row of the plan, checked.
#[derive(Clone, Copy, Debug)]
struct Piece {
    file: u64,
    offset: u64,
    count: usize,
    codec: usize,
    table: Option<usize>,
    seg_rows: usize,
    seg_cols: usize,
    y0: usize,
    y1: usize,
    x0: usize,
    x1: usize,
    c0: usize,
    c1: usize,
    dst: usize,
    dst_row: usize,
    dst_col: usize,
    dst_sample: usize,
}

fn nonneg(v: i64, what: &str, row: usize) -> Result<usize, String> {
    usize::try_from(v).map_err(|_| format!("read plan row {row}: {what} {v} is negative"))
}

impl Piece {
    fn parse(r: &[i64], row: usize, codecs: &[Codec], tables: usize, out: (usize, usize))
        -> Result<Self, String> {
        let n = |c: usize, what: &str| nonneg(r[c], what, row);
        let table = if r[TABLE] < 0 { None } else { Some(n(TABLE, "table")?) };
        let p = Piece {
            file: n(FILE, "file")? as u64, offset: n(OFFSET, "offset")? as u64,
            count: n(COUNT, "count")?, codec: n(CODEC, "codec")?, table,
            seg_rows: n(SEG_ROWS, "segment rows")?, seg_cols: n(SEG_COLS, "segment columns")?,
            y0: n(Y0, "y0")?, y1: n(Y1, "y1")?, x0: n(X0, "x0")?, x1: n(X1, "x1")?,
            c0: n(C0, "c0")?, c1: n(C1, "c1")?, dst: n(DST, "destination")?,
            dst_row: n(DST_ROW, "row stride")?, dst_col: n(DST_COL, "column stride")?,
            dst_sample: n(DST_SAMPLE, "sample stride")?,
        };
        let codec = codecs.get(p.codec)
            .ok_or_else(|| format!("read plan row {row}: no codec {}", p.codec))?;
        if p.table.is_some_and(|t| t >= tables) {
            return Err(format!("read plan row {row}: no table {}", r[TABLE]));
        }
        let rows_ok = codec.kind == Kind::Raw || p.y1 <= p.seg_rows;
        if p.y0 >= p.y1 || p.x0 >= p.x1 || p.c0 >= p.c1 || !rows_ok || p.x1 > p.seg_cols
            || p.c1 > codec.samples {
            return Err(format!("read plan row {row}: rectangle rows {}..{}, columns {}..{}, \
                                samples {}..{} outside its {} x {} x {} segment",
                               p.y0, p.y1, p.x0, p.x1, p.c0, p.c1, p.seg_rows, p.seg_cols,
                               codec.samples));
        }
        // the last byte written, from the strides: inside the output buffer
        let last = (p.y1 - p.y0 - 1).checked_mul(p.dst_row)
            .and_then(|a| (p.x1 - p.x0 - 1).checked_mul(p.dst_col).and_then(|b| a.checked_add(b)))
            .and_then(|a| (p.c1 - p.c0 - 1).checked_mul(p.dst_sample).and_then(|b| a.checked_add(b)))
            .and_then(|a| a.checked_add(codec.itemsize))
            .and_then(|a| a.checked_add(p.dst));
        match last {
            Some(end) if p.dst >= out.0 && end <= out.0 + out.1 => {}
            _ => return Err(format!("read plan row {row}: destination outside the output buffer")),
        }
        // the stride of an axis of one element is never stepped (numpy gives such axes
        // any stride, often 0): set it to what a contiguous layout has, so the fast
        // paths see contiguous memory as contiguous
        let mut p = p;
        if p.c1 - p.c0 == 1 {
            p.dst_sample = codec.itemsize;
        }
        if p.x1 - p.x0 == 1 {
            p.dst_col = (p.c1 - p.c0) * p.dst_sample;
        }
        if p.y1 - p.y0 == 1 {
            p.dst_row = (p.x1 - p.x0) * p.dst_col;
        }
        Ok(p)
    }

    fn rows(&self) -> usize {
        self.y1 - self.y0
    }
}

/// The engine: its files, its decoded-segment cache, and thread pools by size.
pub struct Engine {
    pub files: Files,
    pub cache: Mutex<Cache>,
    pools: Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>,
}

/// What one call reads: checked pieces and what they point to.
pub struct Plan {
    pieces: Vec<Piece>,
    codecs: Vec<Codec>,
    tables: Vec<Vec<u8>>,
}

impl Plan {
    /// `rows`: the plan table, `NCOL` integers per piece; `out`: (address, length)
    /// of the output buffer every destination must lie in.
    pub fn new(rows: &[i64], codecs: Vec<Codec>, tables: Vec<Vec<u8>>, out: (usize, usize))
        -> Result<Self, String> {
        if !rows.len().is_multiple_of(NCOL) {
            return Err(format!("read plan: {} integers, not rows of {NCOL}", rows.len()));
        }
        for c in &codecs {
            c.check()?;
        }
        let pieces = rows.as_chunks::<NCOL>().0.iter().enumerate()
            .map(|(i, r)| Piece::parse(r, i, &codecs, tables.len(), out))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { pieces, codecs, tables })
    }
}

/// A unit of work: an uncompressed piece, or a segment and every piece it feeds.
enum Job {
    Raw(usize),
    Segment(Vec<usize>),
}

impl Engine {
    pub fn new(cache_limit: usize, item_limit: usize) -> Self {
        Self { files: Files::default(), cache: Mutex::new(Cache::new(cache_limit, item_limit)),
               pools: Mutex::new(HashMap::new()) }
    }

    fn pool(&self, threads: usize) -> Result<Arc<rayon::ThreadPool>, String> {
        let mut pools = self.pools.lock().unwrap();
        if let Some(p) = pools.get(&threads) {
            return Ok(p.clone());
        }
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads)
            .thread_name(|i| format!("micro-reader-core-{i}")).build()
            .map_err(|e| format!("read engine: no thread pool: {e}"))?);
        pools.insert(threads, pool.clone());
        Ok(pool)
    }

    /// Run `plan` on `threads` threads (1: on the caller's); returns, per job, the
    /// job's first piece and the samples it read or decoded (a segment several
    /// pieces need counts once, on its first piece).
    pub fn read_jobs(&self, plan: &Plan, threads: usize) -> Result<Vec<(usize, usize)>, String> {
        // the files, looked up once
        let mut sources: HashMap<u64, Arc<Source>> = HashMap::new();
        for p in &plan.pieces {
            if let std::collections::hash_map::Entry::Vacant(e) = sources.entry(p.file) {
                let s = self.files.get(p.file)
                    .ok_or_else(|| format!("read engine: file {} is not open", p.file))?;
                e.insert(s);
            }
        }
        // pieces of one segment are one job: decoded once
        let mut jobs: Vec<Job> = Vec::new();
        let mut at: HashMap<(u64, u64), usize> = HashMap::new();
        for (i, p) in plan.pieces.iter().enumerate() {
            if plan.codecs[p.codec].kind == Kind::Raw {
                jobs.push(Job::Raw(i));
                continue;
            }
            match at.get(&(p.file, p.offset)) {
                Some(&j) => {
                    if let Job::Segment(v) = &mut jobs[j] {
                        v.push(i)
                    }
                }
                None => {
                    at.insert((p.file, p.offset), jobs.len());
                    jobs.push(Job::Segment(vec![i]));
                }
            }
        }
        let run = |job: &Job| -> Result<usize, String> {
            match job {
                Job::Raw(i) => {
                    let p = &plan.pieces[*i];
                    raw(&sources[&p.file], p, &plan.codecs[p.codec])
                }
                Job::Segment(v) => self.segment(plan, &sources, v),
            }
        };
        let counted = |job: &Job| -> Result<(usize, usize), String> {
            let first = match job {
                Job::Raw(i) => *i,
                Job::Segment(v) => v[0],
            };
            run(job).map(|n| (first, n))
        };
        let work: usize = jobs.iter().map(|job| match job {
            Job::Raw(i) => {
                let p = &plan.pieces[*i];
                let c = &plan.codecs[p.codec];
                p.rows() * (p.x1 - p.x0) * c.samples * c.itemsize
            }
            Job::Segment(v) => {
                let p = &plan.pieces[v[0]];
                let c = &plan.codecs[p.codec];
                8 * p.seg_rows * p.seg_cols * c.samples * c.itemsize
            }
        }).sum();
        if threads <= 1 || jobs.len() < 2 || work < SMALL_WORK {
            return jobs.iter().map(counted).collect();
        }
        self.pool(threads)?.install(|| jobs.par_iter().map(counted).collect())
    }

    /// Decode (or take from the cache) the segment of pieces `v`, copy each out.
    fn segment(&self, plan: &Plan, sources: &HashMap<u64, Arc<Source>>, v: &[usize])
        -> Result<usize, String> {
        let first = &plan.pieces[v[0]];
        let codec = &plan.codecs[first.codec];
        let source = &sources[&first.file];
        let key = (first.file, first.offset);
        let need = first.seg_rows * first.seg_cols * codec.samples * codec.itemsize;
        let (cached, keep) = {
            let mut cache = self.cache.lock().unwrap();
            (cache.get(key), need <= cache.item_limit && need <= cache.limit)
        };
        if let Some(seg) = cached {
            if seg.len() != need {
                // a cached segment decoded for a plan that saw another shape
                return Err(format!("read engine: segment at byte {} of {} holds {} bytes,                                     the plan expects {need}", first.offset,
                                   quoted(&source.path), seg.len()));
            }
            for &i in v {
                place(&seg, &plan.pieces[i], codec);
            }
            return Ok(0);
        }
        let table = first.table.map(|t| plan.tables[t].as_slice());
        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let (data, spare) = &mut *scratch;
            let result = if keep {
                // decoded into its own memory, which the cache keeps
                let mut seg = Vec::new();
                decode_segment(source, first, codec, table, data, &mut seg).map(|()| {
                    for &i in v {
                        place(&seg, &plan.pieces[i], codec);
                    }
                    self.cache.lock().unwrap().put(key, Arc::new(seg));
                })
            } else {
                // not kept: this thread's buffer, reused (no fresh pages per segment)
                decode_segment(source, first, codec, table, data, spare).map(|()| {
                    for &i in v {
                        place(spare, &plan.pieces[i], codec);
                    }
                })
            };
            for buf in [data, spare] {
                if buf.capacity() > SCRATCH_KEEP {
                    *buf = Vec::new();               // a rare large segment: not held on to
                }
            }
            result.map(|()| need / codec.itemsize)
        })
    }
}

/// Bytes of scratch memory a thread keeps between segments.
const SCRATCH_KEEP: usize = 4 << 20;

thread_local! {
    /// (compressed bytes, decoded segment not cached) of the thread's current segment
    static SCRATCH: std::cell::RefCell<(Vec<u8>, Vec<u8>)> =
        const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
}

/// Read `p.count` bytes of a segment and decode them to native-order samples,
/// `seg_rows` x `seg_cols` x `samples`.
fn decode_segment(source: &Source, p: &Piece, codec: &Codec, table: Option<&[u8]>,
                  data: &mut Vec<u8>, seg: &mut Vec<u8>) -> Result<(), String> {
    data.clear();
    data.resize(p.count, 0);
    if !source.read_exact_at(data, p.offset).map_err(|e| io_error(source, p.offset, e))? {
        return Err(source.ends_early(p.offset));
    }
    let data: &[u8] = data;
    let (rows, cols, ns, item) = (p.seg_rows, p.seg_cols, codec.samples, codec.itemsize);
    seg.clear();
    seg.resize(rows * cols * ns * item, 0);
    let seg: &mut [u8] = seg;
    let corrupt = |e: String| format!("{}: segment at byte {}: {e}", quoted(&source.path), p.offset);
    let check = |got: (usize, usize, usize)| -> Result<(), String> {
        if got != (cols, rows, ns) {
            return Err(format!("{}: segment at byte {} decodes to {} x {} x {}, the TIFF tags \
                                say {rows} x {cols} x {ns}", quoted(&source.path), p.offset,
                               got.1, got.0, got.2));
        }
        Ok(())
    };
    match codec.kind {
        Kind::Raw => unreachable!("raw pieces are read in place"),
        Kind::None => {
            let n = data.len().min(seg.len());
            seg[..n].copy_from_slice(&data[..n]);
        }
        Kind::Zstd => {
            let n = decode::zstd(data, seg).map_err(|e| corrupt(format!("corrupt zstd data: {e}")))?;
            if n != seg.len() {
                return Err(corrupt(format!("zstd data decodes to {n} bytes, expected {}", seg.len())));
            }
        }
        // a short segment (LZW without EOI, ...) leaves zeros, as TIFF readers do
        Kind::Lzw => {
            decode::lzw(data, seg).map_err(|e| corrupt(format!("corrupt LZW data: {e}")))?;
        }
        Kind::Zlib => {
            decode::zlib(data, seg).map_err(|e| corrupt(format!("corrupt deflate data: {e}")))?;
        }
        Kind::PackBits => {
            decode::packbits(data, seg).map_err(|e| corrupt(format!("corrupt PackBits data: {e}")))?;
        }
        Kind::Jpeg => {
            let joined;
            let data = match table {
                Some(header) if codec.jpeg_header => {
                    joined = [header, data, b"\xff\xd9"].concat();
                    &joined[..]
                }
                // abbreviated stream: JPEGTables without their EOI, the segment without its SOI
                Some(tables) if tables.len() >= 2 && data.len() >= 2 => {
                    joined = [&tables[..tables.len() - 2], &data[2..]].concat();
                    &joined[..]
                }
                _ => data,
            };
            let (w, h, n, _) = decode::jpeg(data, seg, codec.colour)
                .map_err(|e| corrupt(format!("corrupt JPEG data: {e}")))?;
            check((w, h, n))?;
        }
        Kind::Jpeg2k => {
            let (w, h, n, _, _) = decode::jpeg2k(data, seg).map_err(corrupt)?;
            check((w, h, n))?;
        }
        Kind::WebP => {
            let (w, h, n, _) = decode::webp(data, seg).map_err(corrupt)?;
            check((w, h, n))?;
        }
        Kind::Jxl => {
            let (w, h, n, _) = decode::jxl(data, seg).map_err(corrupt)?;
            check((w, h, n))?;
        }
        Kind::Png => {
            let (w, h, n, _) = decode::png(data, seg).map_err(corrupt)?;
            check((w, h, n))?;
        }
        Kind::Eer => {
            let (skipbits, horzbits, vertbits) = codec.eer;
            decode::eer(data, seg, EerParams { height: rows, width: cols, skipbits,
                                                       horzbits, vertbits, superres: 0,
                                                       itemsize: item })
                .map_err(|e| corrupt(format!("corrupt EER data: {e}")))?;
        }
        Kind::ZstdCzi => {
            // header: its length, then (from 3 bytes on) chunk 1 = preprocessing,
            // whose bit 0 is hi-lo byte packing (all low bytes, then all high bytes)
            let header = *data.first().ok_or_else(|| corrupt("empty zstd subblock".into()))?
                as usize;
            if header > data.len() {
                return Err(corrupt(format!("zstd header of {header} bytes in {}", data.len())));
            }
            let hilo = header >= 3 && data[1] == 1 && data[2] & 1 == 1;
            let body = &data[header..];
            let zstd = |out: &mut [u8]| -> Result<(), String> {
                let n = decode::zstd(body, out)
                    .map_err(|e| corrupt(format!("corrupt zstd data: {e}")))?;
                if n != out.len() {
                    return Err(corrupt(format!("zstd data decodes to {n} bytes, expected {}",
                                               out.len())));
                }
                Ok(())
            };
            if hilo && item == 2 {
                let mut packed = vec![0u8; seg.len()];
                zstd(&mut packed)?;
                let half = seg.len() / 2;
                let (lo, hi) = packed.split_at(half);
                for ((v, &l), &h) in seg.as_chunks_mut::<2>().0.iter_mut().zip(lo).zip(hi) {
                    *v = [l, h];
                }
            } else {
                zstd(seg)?;
            }
        }
        Kind::JpegXr => {
            jpegxr(data, seg, codec, ns, item, &corrupt, &check)?;
        }
    }
    if codec.kind.bytes() {
        match codec.predictor {
            3 => floating_point_predictor(seg, rows, cols, ns, item),
            2 => {
                if codec.swap() {
                    swap_bytes(seg, item);
                }
                horizontal_predictor(seg, rows, cols, ns, item);
            }
            _ => {
                if codec.swap() {
                    swap_bytes(seg, item);
                }
            }
        }
    } else if item > 1 && cfg!(target_endian = "big") {
        swap_bytes(seg, item);                   // image codecs write little-endian
    }
    Ok(())
}

/// A JPEG XR segment through jxrlib.  With `codec.bgr` the samples end up B, G,
/// R(, A) as stored (8-bit Zeiss tiles decode that way without a conversion pass;
/// others come out R, G, B and are swapped here).
#[cfg(feature = "jxrlib")]
fn jpegxr(data: &[u8], seg: &mut [u8], codec: &Codec, ns: usize, item: usize,
          corrupt: &dyn Fn(String) -> String,
          check: &dyn Fn((usize, usize, usize)) -> Result<(), String>) -> Result<(), String> {
    let (w, h, n, bytes, _kind, order) = decode::jpegxr(data, seg, codec.bgr)
        .map_err(|e| corrupt(format!("corrupt JPEG XR data: {e}")))?;
    check((w, h, n))?;
    if bytes != item {
        return Err(corrupt(format!("JPEG XR samples of {bytes} bytes, the file says {item}")));
    }
    if codec.bgr && order != "bgr" && ns >= 3 {
        for px in seg.chunks_exact_mut(ns * item) {
            for b in 0..item {
                px.swap(b, 2 * item + b);
            }
        }
    }
    Ok(())
}

#[cfg(not(feature = "jxrlib"))]
fn jpegxr(_data: &[u8], _seg: &mut [u8], _codec: &Codec, _ns: usize, _item: usize,
          corrupt: &dyn Fn(String) -> String,
          _check: &dyn Fn((usize, usize, usize)) -> Result<(), String>) -> Result<(), String> {
    Err(corrupt("JPEG XR: micro-reader-core was built without jxrlib".into()))
}

fn io_error(source: &Source, offset: u64, e: std::io::Error) -> String {
    format!("{}: reading at byte {offset}: {e}", quoted(&source.path))
}

fn swap_bytes(buf: &mut [u8], item: usize) {
    for v in buf.chunks_exact_mut(item) {
        v.reverse();
    }
}

/// TIFF predictor 2: each sample is the difference to the one a pixel left of it
/// (native-order samples, wrapping integer sums).
fn horizontal_predictor(seg: &mut [u8], rows: usize, cols: usize, ns: usize, item: usize) {
    macro_rules! run {
        ($t:ty) => {{
            let row_len = cols * ns * item;
            for row in seg.chunks_exact_mut(row_len).take(rows) {
                let mut prev = vec![0 as $t; ns];
                for (k, v) in row.chunks_exact_mut(item).enumerate() {
                    let s = k % ns;
                    let x = <$t>::from_ne_bytes(v.try_into().unwrap()).wrapping_add(prev[s]);
                    v.copy_from_slice(&x.to_ne_bytes());
                    prev[s] = x;
                }
            }
        }};
    }
    match item {
        1 => run!(u8),
        2 => run!(u16),
        4 => run!(u32),
        _ => run!(u64),
    }
}

/// TIFF predictor 3 (libtiff `fpAcc`): per row, bytes summed with a stride of one
/// pixel (`ns` bytes), then the byte planes (most significant first) put back
/// together as native-order samples.  The file's byte order plays no part.
fn floating_point_predictor(seg: &mut [u8], rows: usize, cols: usize, ns: usize, item: usize) {
    let values = cols * ns;
    let row_len = values * item;
    let mut tmp = vec![0u8; row_len];
    for row in seg.chunks_exact_mut(row_len).take(rows) {
        for j in ns..row_len {
            row[j] = row[j].wrapping_add(row[j - ns]);
        }
        for v in 0..values {
            for k in 0..item {
                // plane k holds byte k of every value, most significant first
                let b = row[k * values + v];
                let at = if cfg!(target_endian = "big") { k } else { item - 1 - k };
                tmp[v * item + at] = b;
            }
        }
        row.copy_from_slice(&tmp);
    }
}

/// Copy samples of `item` bytes from `src` to `dst`, reversing each when `swap`.
///
/// # Safety
/// `dst` must be valid for `n * item` bytes.
#[inline]
unsafe fn copy_items(src: &[u8], dst: *mut u8, item: usize, swap: bool) {
    if !swap {
        std::ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len());
        return;
    }
    for (k, v) in src.chunks_exact(item).enumerate() {
        for b in 0..item {
            *dst.add(k * item + b) = v[item - 1 - b];
        }
    }
}

/// Write rows `y0..y1` (relative to the piece's first row: `rows` holds them, each
/// `row_len` bytes, starting at column 0 of the segment) to the piece's destination,
/// columns `x0..x1` and samples `c0..c1` (B, G, R segments reversed: `codec.bgr`).
#[allow(clippy::too_many_arguments)]
fn scatter(rows: &[u8], row_len: usize, first: usize, n: usize, p: &Piece, codec: &Codec,
           swap: bool) {
    let (ns, item) = (codec.samples, codec.itemsize);
    let width = p.x1 - p.x0;
    let nc = p.c1 - p.c0;
    let px = ns * item;
    // contiguous destination rows holding every sample: one copy per row
    let whole = !codec.bgr && p.c0 == 0 && p.c1 == ns && p.dst_sample == item && p.dst_col == px;
    for r in 0..n {
        let src = &rows[r * row_len..(r + 1) * row_len];
        // SAFETY: Piece::parse checked the rectangle against the output buffer
        let base = p.dst + (first + r) * p.dst_row;
        unsafe {
            if whole {
                copy_items(&src[p.x0 * px..p.x1 * px], base as *mut u8, item, swap);
            } else if !codec.bgr && p.dst_sample == item {
                for x in 0..width {
                    let s = (p.x0 + x) * px + p.c0 * item;
                    copy_items(&src[s..s + nc * item], (base + x * p.dst_col) as *mut u8, item, swap);
                }
            } else {
                for x in 0..width {
                    for c in 0..nc {
                        let s = (p.x0 + x) * px + codec.stored(p.c0 + c) * item;
                        copy_items(&src[s..s + item],
                                   (base + x * p.dst_col + c * p.dst_sample) as *mut u8, item, swap);
                    }
                }
            }
        }
    }
}

/// Copy piece `p`'s rectangle out of its decoded (native-order) segment.
fn place(seg: &[u8], p: &Piece, codec: &Codec) {
    let row_len = p.seg_cols * codec.samples * codec.itemsize;
    scatter(&seg[p.y0 * row_len..p.y1 * row_len], row_len, 0, p.rows(), p, codec, false);
}

/// Read piece `p` of an uncompressed segment (stride `seg_cols` pixels) in place:
/// only its rows and columns; returns the samples read.
fn raw(source: &Source, p: &Piece, codec: &Codec) -> Result<usize, String> {
    let (ns, item, swap) = (codec.samples, codec.itemsize, codec.swap());
    let px = ns * item;
    let row_len = p.seg_cols * px;
    let rows = p.rows();
    let width = p.x1 - p.x0;
    let at = |r: usize, x: usize| p.offset + (r * row_len + x * px) as u64;
    let read = |buf: &mut [u8], offset: u64| -> Result<(), String> {
        match source.read_exact_at(buf, offset) {
            Ok(true) => Ok(()),
            Ok(false) => Err(source.ends_early(offset)),
            Err(e) => Err(io_error(source, offset, e)),
        }
    };
    let whole_pixels = !codec.bgr && p.c0 == 0 && p.c1 == ns && p.dst_sample == item
        && p.dst_col == px;
    let direct = whole_pixels && !swap;
    if direct && p.x0 == 0 && p.x1 == p.seg_cols && p.dst_row == row_len {
        // the destination has the file's layout: one read straight into it
        // SAFETY: Piece::parse checked the rectangle against the output buffer
        let dst = unsafe { std::slice::from_raw_parts_mut(p.dst as *mut u8, rows * row_len) };
        read(dst, at(p.y0, 0))?;
        return Ok(rows * p.seg_cols * ns);
    }
    if rows > 1 && (p.x0, p.x1) != (0, p.seg_cols) && (p.seg_cols - width) * px <= codec.gap {
        // a few skipped bytes per row cost less than one more read per row: read
        // from the first row's columns to the last row's, in chunks
        let per_chunk = (CHUNK_BYTES / row_len).max(1).min(rows);
        return raw_buffer(per_chunk * row_len, |buf| {
            let mut r = 0;
            while r < rows {
                let n = per_chunk.min(rows - r);
                let span = (n - 1) * row_len + width * px;
                read(&mut buf[p.x0 * px..p.x0 * px + span], at(p.y0 + r, p.x0))?;
                scatter(&buf[..n * row_len], row_len, r, n, p, codec, swap);
                r += n;
            }
            Ok((rows - 1) * p.seg_cols * ns + width * ns)
        });
    }
    if p.x0 == 0 && p.x1 == p.seg_cols {
        // whole rows are contiguous: read them in chunks
        let per_chunk = (CHUNK_BYTES / row_len).max(1).min(rows);
        return raw_buffer(per_chunk * row_len, |buf| {
            let mut r = 0;
            while r < rows {
                let n = per_chunk.min(rows - r);
                read(&mut buf[..n * row_len], at(p.y0 + r, 0))?;
                scatter(&buf[..n * row_len], row_len, r, n, p, codec, swap);
                r += n;
            }
            Ok(rows * p.seg_cols * ns)
        });
    }
    // row by row: straight into the destination when its rows have the file's layout
    if direct {
        for r in 0..rows {
            // SAFETY: as above, one row of the checked rectangle
            let dst = unsafe {
                std::slice::from_raw_parts_mut((p.dst + r * p.dst_row) as *mut u8, width * px)
            };
            read(dst, at(p.y0 + r, p.x0))?;
        }
        return Ok(rows * width * ns);
    }
    raw_buffer(row_len, |buf| {
        for r in 0..rows {
            read(&mut buf[p.x0 * px..p.x1 * px], at(p.y0 + r, p.x0))?;
            scatter(buf, row_len, r, 1, p, codec, swap);
        }
        Ok(rows * width * ns)
    })
}

thread_local! {
    /// the thread's buffer for uncompressed rows, reused between pieces
    static RAW: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `f` on `n` bytes of this thread's raw-read buffer (grown, never zeroed
/// again once grown; let go when a rare piece made it larger than a chunk).
fn raw_buffer<R>(n: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
    RAW.with(|buf| {
        let mut buf = buf.borrow_mut();
        if buf.len() < n {
            buf.resize(n, 0);
        }
        let result = f(&mut buf[..n]);
        if buf.len() > CHUNK_BYTES + (1 << 20) {
            *buf = Vec::new();
        }
        result
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn horizontal_predictor_sums_per_sample() {
        let mut seg: Vec<u8> = vec![1, 10, 1, 10, 1, 10];      // 1 row, 3 px, 2 samples
        horizontal_predictor(&mut seg, 1, 3, 2, 1);
        assert_eq!(seg, vec![1, 10, 2, 20, 3, 30]);
    }

    #[test]
    fn floating_point_predictor_restores_values() {
        // 1 row of two float32 values, encoded as libtiff's fpDiff does
        let values = [1.5f32, -2.25f32];
        let mut planes = vec![0u8; 8];
        for (v, x) in values.iter().enumerate() {
            for (k, b) in x.to_be_bytes().iter().enumerate() {
                planes[k * 2 + v] = *b;
            }
        }
        for j in (1..8).rev() {
            planes[j] = planes[j].wrapping_sub(planes[j - 1]);
        }
        floating_point_predictor(&mut planes, 1, 2, 1, 4);
        let got: Vec<f32> = planes.as_chunks::<4>().0.iter()
            .map(|c| f32::from_ne_bytes(*c)).collect();
        assert_eq!(got, values);
    }
}
