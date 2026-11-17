//! Native decoders for micro-reader, shipped as the separate package
//! `micro-reader-codecs` (Python module `micro_reader_codecs`); micro-reader
//! depends on it directly.
//!
//! Two faces of one crate:
//! - `decode`: the decoders as plain Rust functions on byte slices.  The read
//!   engine in micro-reader-core links this crate as a Rust library and calls
//!   them directly (Workplan decision MR-8), without the "python" feature.
//! - `python` (feature "python", on by default): the Python module, wrappers that
//!   take Python buffers and decode without Python's lock.  Every function there
//!   has a reference path in Python (`micro_reader._codecs`) that the tests
//!   compare with bit for bit (PLAN.md, "Native rule").
//!
//! Decoders write into a buffer the caller sized from the file's own
//! description of the pixels, and return what the codestream said it holds,
//! so the caller can check the two agree and put samples in R, G, B order.

mod cdecoders;
pub mod decode;
mod jpeg;
#[cfg(feature = "python")]
mod python;
#[cfg(feature = "python")]
mod stream;
