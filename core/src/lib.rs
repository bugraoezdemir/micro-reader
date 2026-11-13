//! micro-reader's native read engine, shipped as the package `micro-reader-core`
//! (Python module `micro_reader_core`); Workplan MR23, decisions MR-7 (this crate)
//! and MR-8 (it links micro-reader-codecs as a Rust library).
//!
//! micro-reader builds a read plan per request in Python (file, offset, length,
//! codec, source rectangle and destination per piece); the engine reads, decodes
//! and places every piece in one call, on its own threads and without Python's
//! lock (`engine.rs`).  The Python path stays as the reference the tests compare
//! with bit for bit.

mod cache;
mod engine;
mod files;

use micro_reader_codecs::decode::{self, JpegColour};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// How one layout's segments decode (see `engine::Codec`).
#[pyclass(frozen, from_py_object, module = "micro_reader_core")]
#[derive(Clone)]
struct Codec(engine::Codec);

#[pymethods]
impl Codec {
    #[new]
    #[pyo3(signature = (kind, itemsize, samples, *, predictor=1, big_endian=false,
                        colour="auto", jpeg_header=false, eer=(7, 2, 2), gap=32768))]
    #[allow(clippy::too_many_arguments)]
    fn new(kind: &str, itemsize: usize, samples: usize, predictor: u8, big_endian: bool,
           colour: &str, jpeg_header: bool, eer: (u32, u32, u32), gap: usize)
        -> PyResult<Self> {
        let kind = engine::Kind::parse(kind)
            .ok_or_else(|| PyValueError::new_err(format!("unknown codec {kind:?}")))?;
        let colour = JpegColour::parse(colour)
            .ok_or_else(|| PyValueError::new_err(format!("unknown JPEG colour {colour:?}")))?;
        let c = engine::Codec { kind, itemsize, samples, predictor, big_endian, colour,
                                jpeg_header, eer, gap };
        c.check().map_err(PyValueError::new_err)?;
        Ok(Self(c))
    }

    fn __repr__(&self) -> String {
        format!("Codec({:?}, itemsize={}, samples={}, predictor={})", self.0.kind,
                self.0.itemsize, self.0.samples, self.0.predictor)
    }
}

/// The read engine: open files, a decoded-segment cache, thread pools.
#[pyclass(frozen, module = "micro_reader_core")]
struct Engine(engine::Engine);

#[pymethods]
impl Engine {
    /// `cache_limit`: bytes all cached segments may hold; `item_limit`: larger
    /// segments are not kept.
    #[new]
    #[pyo3(signature = (cache_limit=256 << 20, item_limit=16 << 20))]
    fn new(cache_limit: usize, item_limit: usize) -> Self {
        Self(engine::Engine::new(cache_limit, item_limit))
    }

    /// Open `path` for reading; returns its file number for read plans.
    fn open(&self, py: Python<'_>, path: &str) -> PyResult<u64> {
        py.detach(|| self.0.files.open(path))
            .map_err(|e| PyOSError::new_err(format!("{}: {e}", files::quoted(path))))
    }

    /// Close file `file` and forget its cached segments; False if it was not open.
    fn close(&self, file: u64) -> bool {
        self.0.cache.lock().unwrap().forget(file);
        self.0.files.close(file)
    }

    /// Run a read plan: `pieces`, a C-contiguous int64 buffer of rows of `NCOL`
    /// (`PIECE_COLUMNS` names them); `codecs` and `tables` (bytes), what its rows
    /// point to; `out`, the writable buffer every destination address lies in.
    /// Returns the number of samples read or decoded.
    #[pyo3(signature = (pieces, codecs, tables, out, threads=1))]
    fn read(&self, py: Python<'_>, pieces: &Bound<'_, PyAny>, codecs: Vec<Codec>,
            tables: Vec<Bound<'_, PyBytes>>, out: &Bound<'_, PyAny>, threads: usize)
        -> PyResult<usize> {
        let rows: PyBuffer<i64> = PyBuffer::get(pieces)?;
        if !rows.is_c_contiguous() {
            return Err(PyBufferError::new_err("the plan must be C-contiguous"));
        }
        let rows = rows.to_vec(py)?;
        let dst: PyBuffer<u8> = PyBuffer::get(out).or_else(|_| {
            // any numpy dtype: its bytes
            let view = out.call_method1("view", ("u1",))?;
            PyBuffer::get(&view)
        })?;
        if dst.readonly() {
            return Err(PyBufferError::new_err("the output buffer is read-only"));
        }
        let region = (dst.buf_ptr() as usize, dst.len_bytes());
        let tables = tables.iter().map(|t| t.as_bytes().to_vec()).collect();
        let plan = engine::Plan::new(&rows, codecs.into_iter().map(|c| c.0).collect(), tables,
                                     region).map_err(PyValueError::new_err)?;
        // `dst` stays alive (and the buffer exported) until the read has ended
        let result = py.detach(|| self.0.read(&plan, threads.max(1)));
        drop(dst);
        result.map_err(PyValueError::new_err)
    }

    /// Bytes the decoded-segment cache holds now.
    fn cached_bytes(&self) -> usize {
        self.0.cache.lock().unwrap().bytes()
    }

    /// Set what the cache may hold (bytes); older segments go at once.
    fn set_cache_limit(&self, limit: usize) {
        let mut cache = self.0.cache.lock().unwrap();
        cache.limit = limit;
        cache.shrink(limit);
    }

    fn clear_cache(&self) {
        self.0.cache.lock().unwrap().clear();
    }

    /// The number of files open.
    fn open_files(&self) -> usize {
        self.0.files.count()
    }
}

#[pymodule]
fn micro_reader_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // the micro-reader-codecs crate compiled in (not the installed Python package)
    m.add("CODECS_VERSION", decode::VERSION)?;
    m.add("ZSTD_VERSION", decode::zstd_version())?;
    m.add("HAS_LIBJPEG_TURBO", decode::HAS_LIBJPEG_TURBO)?;
    m.add("HAS_JXRLIB", decode::HAS_JXRLIB)?;
    let columns: [&str; engine::NCOL] = [
        "file", "offset", "count", "codec", "table", "seg_rows", "seg_cols", "y0", "y1", "x0",
        "x1", "c0", "c1", "dst", "dst_row", "dst_col", "dst_sample"];
    m.add("PIECE_COLUMNS", columns.to_vec())?;
    m.add_class::<Codec>()?;
    m.add_class::<Engine>()?;
    Ok(())
}
