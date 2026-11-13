//! The Python module `micro_reader_codecs` (feature "python", on by default): thin
//! wrappers over `decode`, which take Python buffers and run the decoder without
//! Python's lock, so the shared decode pool decodes in parallel.
//!
//! Every function here has a reference path in Python (`micro_reader._codecs`)
//! that the tests compare with bit for bit (PLAN.md, "Native rule").

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;

use crate::cdecoders;
use crate::decode;
use crate::stream;

/// Run `decode(input, output)` on the bytes of `data` and `out` without the
/// GIL.  The buffer objects stay exported until this returns, so the memory
/// cannot move or be freed meanwhile; addresses cross the release as integers.
fn with_buffers<T, F>(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>, decode: F)
    -> PyResult<T>
where
    T: Send,
    F: FnOnce(&[u8], &mut [u8]) -> Result<T, String> + Send,
{
    let src: PyBuffer<u8> = PyBuffer::get(data)?;
    let dst: PyBuffer<u8> = PyBuffer::get(out)?;
    if dst.readonly() {
        return Err(PyBufferError::new_err("output buffer is read-only"));
    }
    if !src.is_c_contiguous() || !dst.is_c_contiguous() {
        return Err(PyBufferError::new_err("buffers must be C-contiguous"));
    }
    let (src_ptr, src_len) = (src.buf_ptr() as usize, src.len_bytes());
    let (dst_ptr, dst_len) = (dst.buf_ptr() as usize, dst.len_bytes());
    py.detach(move || {
        let input = unsafe { std::slice::from_raw_parts(src_ptr as *const u8, src_len) };
        let output = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, dst_len) };
        decode(input, output)
    })
    .map_err(PyValueError::new_err)
}

/// One zstd frame -> `out`; returns the bytes written.
#[pyfunction]
fn zstd_decompress_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<usize> {
    with_buffers(py, data, out, decode::zstd)
}

/// One LZ4 block (no frame) -> `out`; returns the bytes written.
#[pyfunction]
fn lz4_block_decompress_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<usize> {
    with_buffers(py, data, out, decode::lz4_block)
}

/// LZF (liblzf's format; h5py's HDF5 filter 32000) -> `out`; returns the bytes
/// written.  An error when the data needs more room than `out` or is damaged.
#[pyfunction]
fn lzf_decompress_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<usize> {
    with_buffers(py, data, out, decode::lzf)
}

/// TIFF-style LZW (MSB first, 8-bit symbols, early code-size switch) -> `out`;
/// returns the bytes written (the stream may end before `out` is full).
#[pyfunction]
fn lzw_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<usize> {
    with_buffers(py, data, out, decode::lzw)
}

/// A JPEG (baseline, progressive or lossless) -> `out`.  Returns
/// (width, height, samples, bits): samples 1 (grey) or 3 (R, G, B); 16-bit
/// samples are written little-endian.  *colour*: "auto" (from the JFIF / Adobe
/// markers, as JPEG files say), "ycbcr" or "rgb" (no transform: what TIFF's
/// Photometric RGB means).  See `decode::jpeg` for the decoders tried.
#[pyfunction]
#[pyo3(signature = (data, out, colour = "auto"))]
fn jpeg_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>, colour: &str)
    -> PyResult<(usize, usize, usize, usize)> {
    let colour = decode::JpegColour::parse(colour)
        .ok_or_else(|| PyValueError::new_err(format!("jpeg: unknown colour {colour:?}")))?;
    with_buffers(py, data, out, move |input, output| decode::jpeg(input, output, colour))
}

/// A JPEG XL codestream (first frame) -> `out` as 8- or 16-bit integer
/// samples, interleaved.  Returns (width, height, channels, bits).
#[pyfunction]
fn jxl_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>)
    -> PyResult<(usize, usize, usize, usize)> {
    with_buffers(py, data, out, decode::jxl)
}

/// The area *x0*..*x1*, *y0*..*y1* of a JPEG XL codestream (first frame)
/// -> `out`, like `jxl_decode_into`: jxl-oxide renders only the groups
/// covering it.  Returns (width, height, channels, bits) of the area.
#[pyfunction]
fn jxl_decode_area_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>,
                        x0: u32, y0: u32, x1: u32, y1: u32) -> PyResult<(usize, usize, usize, usize)> {
    decode::jxl_area_check(x0, y0, x1, y1).map_err(PyValueError::new_err)?;
    with_buffers(py, data, out, move |input, output| decode::jxl_area(input, output, x0, y0, x1, y1))
}

/// A zlib stream (TIFF / HDF5 "deflate") -> `out`; returns the bytes written.
#[pyfunction]
fn zlib_decompress_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<usize> {
    with_buffers(py, data, out, decode::zlib)
}

/// PackBits (Apple / TIFF run-length) -> `out`; returns the bytes written.
/// Stops when `out` is full, as TIFF readers do.
#[pyfunction]
fn packbits_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>) -> PyResult<usize> {
    with_buffers(py, data, out, decode::packbits)
}

/// A PNG (first image, nothing expanded) -> `out`.  Returns (width, height,
/// samples, bits); 16-bit samples little-endian.  Depths below 8 bits and
/// palette indices come out as stored (see `decode::png`).
#[pyfunction]
fn png_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>)
    -> PyResult<(usize, usize, usize, usize)> {
    with_buffers(py, data, out, decode::png)
}

/// A WebP (lossless or lossy) -> `out` as 8-bit R, G, B (A).  Returns
/// (width, height, samples, 8).
#[pyfunction]
fn webp_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>)
    -> PyResult<(usize, usize, usize, usize)> {
    with_buffers(py, data, out, decode::webp)
}

/// A JPEG 2000 codestream or JP2 file -> `out`, components interleaved,
/// each sample 1, 2 or 4 bytes little-endian by its precision (two's
/// complement when signed).  Returns (width, height, components, bits, signed).
#[pyfunction]
fn jpeg2k_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>)
    -> PyResult<(usize, usize, usize, usize, bool)> {
    with_buffers(py, data, out, decode::jpeg2k)
}

/// The rectangle x0 <= x < x1, y0 <= y < y1 of a JPEG 2000 codestream or JP2
/// -> *out*, components interleaved: OpenJPEG decodes only the code-blocks
/// that cover it.  Returns (width, height, components, bits, signed) of the region.
#[pyfunction]
fn jpeg2k_decode_area_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>,
                           x0: u32, y0: u32, x1: u32, y1: u32)
    -> PyResult<(usize, usize, usize, usize, bool)> {
    with_buffers(py, data, out, move |input, output| decode::jpeg2k_area(input, output, x0, y0, x1, y1))
}

/// Samples packed at `bits` bits each (1-32), most significant bit first,
/// every run of `runlen` samples (a TIFF row) starting on a byte boundary
/// -> `out` as unsigned integers of `itemsize` bytes (1, 2 or 4),
/// little-endian.  Returns the number of samples written (whole runs).
#[pyfunction]
fn unpack_bits_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>,
                    bits: u32, runlen: usize, itemsize: usize) -> PyResult<usize> {
    decode::unpack_bits_check(bits, runlen, itemsize).map_err(PyValueError::new_err)?;
    with_buffers(py, data, out, move |input, output| {
        decode::unpack_bits(input, output, bits, runlen, itemsize)
    })
}

/// EER (Electron Event Representation; Thermo Fisher Falcon cameras) ->
/// electron counts per pixel in `out` (`itemsize` 1, 2 or 4 bytes,
/// little-endian, zeroed here), `height` x `width` (super-resolved when
/// `superres` > 0).  Returns the number of electrons (see `decode::eer`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn eer_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>,
                   height: usize, width: usize, skipbits: u32, horzbits: u32, vertbits: u32,
                   superres: u32, itemsize: usize) -> PyResult<usize> {
    let params = decode::EerParams { height, width, skipbits, horzbits, vertbits, superres, itemsize };
    params.check().map_err(PyValueError::new_err)?;
    with_buffers(py, data, out, move |src, dst| decode::eer(src, dst, params))
}

/// A JPEG XR codestream -> `out` through jxrlib, in the layout imagecodecs'
/// `jpegxr_decode` gives.  `bgr=True`: an 8-bit B, G, R(, A) codestream (Zeiss CZI
/// tiles) comes out as stored.  Returns (width, height, samples, bytes per sample,
/// kind, order); see `decode::jpegxr`.
#[cfg(feature = "jxrlib")]
#[pyfunction]
#[pyo3(signature = (data, out, bgr = false))]
fn jpegxr_decode_into(py: Python<'_>, data: &Bound<'_, PyAny>, out: &Bound<'_, PyAny>, bgr: bool)
    -> PyResult<decode::JxrInfo> {
    with_buffers(py, data, out, move |input, output| decode::jpegxr(input, output, bgr))
}

/// Version of the zstd library compiled in.
#[pyfunction]
fn zstd_version() -> String {
    decode::zstd_version()
}

#[pymodule]
fn micro_reader_codecs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(zstd_decompress_into, m)?)?;
    m.add_function(wrap_pyfunction!(lz4_block_decompress_into, m)?)?;
    m.add_function(wrap_pyfunction!(lzf_decompress_into, m)?)?;
    m.add_function(wrap_pyfunction!(lzw_decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg_decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(jxl_decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(jxl_decode_area_into, m)?)?;
    m.add_function(wrap_pyfunction!(zlib_decompress_into, m)?)?;
    m.add_function(wrap_pyfunction!(packbits_decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(png_decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(webp_decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg2k_decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(jpeg2k_decode_area_into, m)?)?;
    m.add_function(wrap_pyfunction!(unpack_bits_into, m)?)?;
    m.add_function(wrap_pyfunction!(eer_decode_into, m)?)?;
    m.add_function(wrap_pyfunction!(zstd_version, m)?)?;
    #[cfg(feature = "jxrlib")]
    m.add_function(wrap_pyfunction!(jpegxr_decode_into, m)?)?;
    // which C decoders this build links (micro-reader picks its routes by these)
    m.add("HAS_LIBJPEG_TURBO", cdecoders::HAS_LIBJPEG_TURBO)?;
    m.add("HAS_JXRLIB", cdecoders::HAS_JXRLIB)?;
    m.add_class::<stream::StreamDecoder>()?;
    m.add_class::<stream::BlockIndex>()?;
    Ok(())
}
