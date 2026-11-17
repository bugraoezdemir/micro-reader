//! The decoders as plain Rust functions on byte slices, without Python.
//!
//! The Python module (`python.rs`, feature "python") wraps each of them; the read
//! engine in micro-reader-core calls them directly (Workplan decision MR-8: core
//! links this crate as a Rust library).  Each writes into `output`, a buffer the
//! caller sized from the file's own description of the pixels, and returns what the
//! codestream said it holds, so the caller can check the two agree.  Errors are
//! messages naming the codec ("zstd: ...", "jpeg: ...").

use std::io::Cursor;

use crate::cdecoders;
use crate::jpeg;

pub use crate::cdecoders::{HAS_JXRLIB, HAS_LIBJPEG_TURBO};

/// This crate's version (micro-reader-codecs), for a crate that links it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// One zstd frame -> `output`; returns the bytes written.
pub fn zstd(input: &[u8], output: &mut [u8]) -> Result<usize, String> {
    ::zstd::bulk::decompress_to_buffer(input, output).map_err(|e| format!("zstd: {e}"))
}

/// One LZ4 block (no frame) -> `output`; returns the bytes written.
pub fn lz4_block(input: &[u8], output: &mut [u8]) -> Result<usize, String> {
    lz4_flex::block::decompress_into(input, output).map_err(|e| format!("lz4: {e}"))
}

/// LZF (liblzf's format; h5py's HDF5 filter 32000) -> `output`; returns the bytes
/// written.  An error when the data needs more room than `output` or is damaged.
pub fn lzf(input: &[u8], output: &mut [u8]) -> Result<usize, String> {
    let (mut i, mut o) = (0usize, 0usize);
    while i < input.len() {
        let ctrl = input[i] as usize;
        i += 1;
        if ctrl < 32 {                                    // ctrl + 1 literal bytes
            let n = ctrl + 1;
            if i + n > input.len() {
                return Err("lzf: literal run past the end of the data".into());
            }
            if o + n > output.len() {
                return Err("lzf: decodes to more bytes than expected".into());
            }
            output[o..o + n].copy_from_slice(&input[i..i + n]);
            i += n;
            o += n;
        } else {                                          // a copy of earlier output
            let mut n = ctrl >> 5;
            if n == 7 {
                n += *input.get(i).ok_or("lzf: truncated back reference")? as usize;
                i += 1;
            }
            let low = *input.get(i).ok_or("lzf: truncated back reference")? as usize;
            i += 1;
            n += 2;
            let distance = ((ctrl & 0x1f) << 8) + low + 1;
            if distance > o {
                return Err("lzf: back reference before the start".into());
            }
            if o + n > output.len() {
                return Err("lzf: decodes to more bytes than expected".into());
            }
            let from = o - distance;
            if distance >= n {
                output.copy_within(from..from + n, o);
            } else {                                      // overlapping: a repeating pattern
                for k in 0..n {
                    output[o + k] = output[from + k];
                }
            }
            o += n;
        }
    }
    Ok(o)
}

/// TIFF-style LZW (MSB first, 8-bit symbols, early code-size switch) -> `output`;
/// returns the bytes written (the stream may end before `output` is full).
pub fn lzw(mut input: &[u8], output: &mut [u8]) -> Result<usize, String> {
    let mut decoder = weezl::decode::Decoder::with_tiff_size_switch(weezl::BitOrder::Msb, 8);
    let mut written = 0;
    loop {
        let result = decoder.decode_bytes(input, &mut output[written..]);
        written += result.consumed_out;
        input = &input[result.consumed_in..];
        match result.status {
            Ok(weezl::LzwStatus::Done) => break,
            Ok(weezl::LzwStatus::Ok) => {
                if written == output.len() || (result.consumed_in == 0 && result.consumed_out == 0) {
                    break;
                }
            }
            Ok(weezl::LzwStatus::NoProgress) => break,
            Err(e) => return Err(format!("lzw: {e}")),
        }
    }
    Ok(written)
}

/// How a JPEG's colour is read: from its JFIF / Adobe markers ("auto", as JPEG files
/// say), as YCbCr, or as RGB without a transform (what TIFF's Photometric RGB means).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JpegColour {
    Auto,
    YCbCr,
    Rgb,
}

impl JpegColour {
    /// "auto", "ycbcr" or "rgb"; None for anything else.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "auto" => Some(Self::Auto),
            "ycbcr" => Some(Self::YCbCr),
            "rgb" => Some(Self::Rgb),
            _ => None,
        }
    }
}

/// A JPEG (baseline, progressive or lossless) -> `output`.  Returns
/// (width, height, samples, bits): samples 1 (grey) or 3 (R, G, B); 16-bit
/// samples are written little-endian (jpeg-decoder gives native order).
///
/// libjpeg-turbo itself decodes first (with its SIMD; 8-bit, 12-bit and lossless, 1
/// or 3 components; `cdecoders.rs`), 12-bit samples as 16 bits.  What it declines (4
/// components), warns about (damaged data it would repair) or refuses goes on as
/// before: baseline and extended sequential JPEG to micro-reader's own port of
/// libjpeg-turbo's decoder (`jpeg.rs`), other kinds to jpeg-decoder.
pub fn jpeg(input: &[u8], output: &mut [u8], colour: JpegColour)
    -> Result<(usize, usize, usize, usize), String> {
    let (transform, own, code) = match colour {
        JpegColour::Auto => (None, jpeg::Colour::Auto, 0),
        JpegColour::YCbCr => (Some(jpeg_decoder::ColorTransform::YCbCr), jpeg::Colour::YCbCr, 1),
        JpegColour::Rgb => (Some(jpeg_decoder::ColorTransform::RGB), jpeg::Colour::Rgb, 2),
    };
    let turbo = match cdecoders::jpeg(input, output, code) {
        cdecoders::Jpeg::Decoded(w, h, n, bits) => return Ok((w, h, n, bits)),
        cdecoders::Jpeg::Failed(e) => format!("; libjpeg-turbo: {e}"),
        cdecoders::Jpeg::Declined => String::new(),
    };
    // progressive, lossless, ...: jpeg-decoder below
    let why = match jpeg::decode_into(input, output, own) {
        Ok((w, h, n)) => return Ok((w, h, n, 8)),
        Err(jpeg::JpegError::Corrupt(e)) => return Err(format!("jpeg: {e}{turbo}")),
        Err(jpeg::JpegError::Unsupported(why)) => why,
    };
    let mut decoder = jpeg_decoder::Decoder::new(input);
    if let Some(transform) = transform {
        decoder.set_color_transform(transform);
    }
    let pixels = decoder.decode().map_err(|e| format!("jpeg: {e} ({why}){turbo}"))?;
    let info = decoder.info().ok_or("jpeg: no image information")?;
    let (samples, bits) = match info.pixel_format {
        jpeg_decoder::PixelFormat::L8 => (1, 8),
        jpeg_decoder::PixelFormat::L16 => (1, 16),
        jpeg_decoder::PixelFormat::RGB24 => (3, 8),
        jpeg_decoder::PixelFormat::CMYK32 => return Err("jpeg: CMYK is not supported".into()),
    };
    if pixels.len() != output.len() {
        return Err(format!("jpeg: decodes to {} bytes, expected {}", pixels.len(), output.len()));
    }
    if bits == 16 && cfg!(target_endian = "big") {
        // jpeg-decoder writes 16-bit samples in native order; we return little-endian
        for (dst, src) in output.chunks_exact_mut(2).zip(pixels.chunks_exact(2)) {
            dst[0] = src[1];
            dst[1] = src[0];
        }
    } else {
        output.copy_from_slice(&pixels);
    }
    Ok((info.width as usize, info.height as usize, samples, bits))
}

/// A JPEG XL codestream (first frame) -> `output` as 8- or 16-bit integer
/// samples, interleaved.  Returns (width, height, channels, bits).
pub fn jxl(input: &[u8], output: &mut [u8]) -> Result<(usize, usize, usize, usize), String> {
    jxl_write(input, output, None)
}

/// The area *x0*..*x1*, *y0*..*y1* of a JPEG XL codestream (first frame)
/// -> `output`, like `jxl`: jxl-oxide renders only the groups covering it.
/// Returns (width, height, channels, bits) of the area.
pub fn jxl_area(input: &[u8], output: &mut [u8], x0: u32, y0: u32, x1: u32, y1: u32)
    -> Result<(usize, usize, usize, usize), String> {
    jxl_area_check(x0, y0, x1, y1)?;
    let area = jxl_oxide::CropInfo { left: x0, top: y0, width: x1 - x0, height: y1 - y0 };
    jxl_write(input, output, Some(area))
}

/// The argument check of `jxl_area`, on its own so the Python wrapper can run it
/// before taking the buffers.
pub fn jxl_area_check(x0: u32, y0: u32, x1: u32, y1: u32) -> Result<(), String> {
    if x1 <= x0 || y1 <= y0 {
        return Err("jpeg xl: empty area".into());
    }
    Ok(())
}

fn jxl_write(input: &[u8], output: &mut [u8], area: Option<jxl_oxide::CropInfo>)
    -> Result<(usize, usize, usize, usize), String> {
        let mut image = jxl_oxide::JxlImage::builder()
            .pool(jxl_oxide::JxlThreadPool::none())   // one thread: the pool runs tiles in parallel
            .read(Cursor::new(input))
            .map_err(|e| format!("jpeg xl: {e}"))?;
        if let Some(area) = area {
            let (w, h) = (image.width(), image.height());
            if area.left + area.width > w || area.top + area.height > h {
                return Err(format!("jpeg xl: area outside the {w} x {h} image"));
            }
            image.set_image_region(area);
        }
        let bits = image.image_header().metadata.bit_depth.bits_per_sample() as usize;
        let render = image.render_frame(0).map_err(|e| format!("jpeg xl: {e}"))?;
        let mut stream = render.stream();
        let (w, h, c) = (stream.width() as usize, stream.height() as usize, stream.channels() as usize);
        let sample_bytes = if bits <= 8 { 1 } else { 2 };
        if w * h * c * sample_bytes != output.len() {
            return Err(format!("jpeg xl: {w} x {h} x {c} at {bits} bits does not fill {} bytes",
                               output.len()));
        }
        if sample_bytes == 1 {
            stream.write_to_buffer::<u8>(output);
        } else {
            let (head, samples, tail) = unsafe { output.align_to_mut::<u16>() };
            if !head.is_empty() || !tail.is_empty() {
                return Err("jpeg xl: output buffer not aligned for 16-bit samples".into());
            }
            stream.write_to_buffer::<u16>(samples);
            for s in samples.iter_mut() {
                *s = s.to_le();
            }
        }
        Ok((w, h, c, bits))
}

/// A zlib stream (TIFF / HDF5 "deflate") -> `output`; returns the bytes written.
pub fn zlib(input: &[u8], output: &mut [u8]) -> Result<usize, String> {
    libdeflater::Decompressor::new().zlib_decompress(input, output)
        .map_err(|e| format!("zlib: {e:?}"))
}

/// PackBits (Apple / TIFF run-length) -> `output`; returns the bytes written.
/// Stops when `output` is full, as TIFF readers do.
pub fn packbits(input: &[u8], output: &mut [u8]) -> Result<usize, String> {
    let (mut i, mut o) = (0usize, 0usize);
    while i < input.len() && o < output.len() {
        let n = input[i] as i8;
        i += 1;
        if n >= 0 {                                   // n + 1 literal bytes
            let count = (n as usize + 1).min(input.len() - i).min(output.len() - o);
            output[o..o + count].copy_from_slice(&input[i..i + count]);
            i += n as usize + 1;
            o += count;
        } else if n != -128 {                         // the next byte, 1 - n times
            if i >= input.len() {
                break;
            }
            let count = ((1 - n as isize) as usize).min(output.len() - o);
            output[o..o + count].fill(input[i]);
            i += 1;
            o += count;
        }
    }
    Ok(o)
}

/// A PNG (first image, nothing expanded) -> `output`.  Returns (width, height,
/// samples, bits); 16-bit samples little-endian.  Depths below 8 bits and
/// palette indices come out as stored: rows of packed samples, most
/// significant bit first, each row starting on a byte (the caller unpacks
/// them and applies the palette, if it wants to).
pub fn png(input: &[u8], output: &mut [u8]) -> Result<(usize, usize, usize, usize), String> {
    let mut decoder = ::png::Decoder::new(Cursor::new(input));
    decoder.set_transformations(::png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().map_err(|e| format!("png: {e}"))?;
    let size = reader.output_buffer_size().ok_or("png: image too large")?;
    let mut pixels = vec![0u8; size];
    let frame = reader.next_frame(&mut pixels).map_err(|e| format!("png: {e}"))?;
    let bits = frame.bit_depth as usize;
    let samples = frame.color_type.samples();
    let used = frame.buffer_size();
    if used != output.len() {
        return Err(format!("png: decodes to {used} bytes, expected {}", output.len()));
    }
    if bits == 16 {                                   // PNG is big-endian
        for (d, s) in output.chunks_exact_mut(2).zip(pixels[..used].chunks_exact(2)) {
            d[0] = s[1];
            d[1] = s[0];
        }
    } else {
        output.copy_from_slice(&pixels[..used]);
    }
    Ok((frame.width as usize, frame.height as usize, samples, bits))
}

/// A WebP (lossless or lossy) -> `output` as 8-bit R, G, B (A).  Returns
/// (width, height, samples, 8).
pub fn webp(input: &[u8], output: &mut [u8]) -> Result<(usize, usize, usize, usize), String> {
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(input))
        .map_err(|e| format!("webp: {e}"))?;
    let (w, h) = decoder.dimensions();
    let samples = if decoder.has_alpha() { 4 } else { 3 };
    let size = decoder.output_buffer_size().ok_or("webp: image too large")?;
    if size != output.len() {
        return Err(format!("webp: decodes to {size} bytes, expected {}", output.len()));
    }
    decoder.read_image(output).map_err(|e| format!("webp: {e}"))?;
    Ok((w as usize, h as usize, samples, 8))
}

/// OpenJPEG's sYCC -> RGB of one pixel (color.c `sycc_to_rgb`, which
/// imagecodecs uses): the arithmetic in double precision, truncated to int,
/// clamped to 0 ..= upb.
fn sycc_to_rgb(offset: i32, upb: i32, y: i32, cb: i32, cr: i32) -> [i32; 3] {
    let (cb, cr) = ((cb - offset) as f32 as f64, (cr - offset) as f32 as f64);
    let r = y + (1.402 * cr) as i32;
    let g = y - (0.344 * cb + 0.714 * cr) as i32;
    let b = y + (1.772 * cb) as i32;
    [r.clamp(0, upb), g.clamp(0, upb), b.clamp(0, upb)]
}

/// The three components of an sYCC image -> full-resolution R, G, B, chroma
/// upsampled as OpenJPEG does it (color.c `sycc444/422/420_to_rgb`): pixel
/// (row, col) takes the chroma sample (row / dy, col / dx); when the image
/// origin (x0, y0) is odd, the first column / row has no chroma sample and
/// uses Cb = Cr = 0, as OpenJPEG does.  None when the layout is not one of
/// 4:4:4, 4:2:2 or 4:2:0.
fn sycc_planes(comps: &[jpeg2k::ImageComponent], x0: usize, y0: usize)
    -> Option<Result<[Vec<i32>; 3], String>> {
    let (w, h) = (comps[0].width() as usize, comps[0].height() as usize);
    let (cw, ch) = (comps[1].width() as usize, comps[1].height() as usize);
    if (comps[2].width() as usize, comps[2].height() as usize) != (cw, ch) || cw == 0 || ch == 0 {
        return None;
    }
    let factor = |full: usize, part: usize| if part == full { Some(1) }
                                            else if part < full && 2 * part >= full { Some(2) }
                                            else { None };
    let (dx, dy) = (factor(w, cw)?, factor(h, ch)?);
    if dx == 1 && dy == 2 {
        return None;                                   // 4:4:0: OpenJPEG leaves it as it is
    }
    let bits = comps[0].precision();
    if bits == 0 || bits > 16 || comps.iter().any(|c| c.precision() != bits || c.is_signed()) {
        return None;
    }
    let (offset, upb) = (1i32 << (bits - 1), (1i32 << bits) - 1);
    let offx = if dx == 2 { x0 & 1 } else { 0 };
    let offy = if dy == 2 { y0 & 1 } else { 0 };
    let (luma, cb, cr) = (comps[0].data(), comps[1].data(), comps[2].data());
    let mut out = [vec![0i32; w * h], vec![0i32; w * h], vec![0i32; w * h]];
    for row in 0..h {
        for col in 0..w {
            let at = row * w + col;
            let (b, r) = if (offx == 1 && col == 0) || (offy == 1 && row == 0) {
                (0, 0)
            } else {
                let k = ((row - offy) / dy) * cw + (col - offx) / dx;
                match (cb.get(k), cr.get(k)) {
                    (Some(&b), Some(&r)) => (b, r),
                    _ => return Some(Err("jpeg 2000: chroma smaller than its luma".into())),
                }
            };
            let rgb = sycc_to_rgb(offset, upb, luma[at], b, r);
            out[0][at] = rgb[0];
            out[1][at] = rgb[1];
            out[2][at] = rgb[2];
        }
    }
    Some(Ok(out))
}

/// The components of a decoded JPEG 2000 *image*, interleaved into *output*,
/// leaving out its first *skip* = (columns, rows); (width, height, components,
/// bits, signed) of what is written.  Three components in sYCC -- a JP2 that
/// says so, or (as OpenJPEG and imagecodecs decide) a codestream whose chroma
/// is subsampled horizontally, e.g. Aperio's TIFF compression 33003 -- are
/// upsampled and converted to RGB, as imagecodecs does.
fn j2k_write(image: &jpeg2k::Image, output: &mut [u8], skip: (usize, usize))
    -> Result<(usize, usize, usize, usize, bool), String> {
    let comps = image.components();
    let first = comps.first().ok_or("jpeg 2000: no components")?;
    let (fw, fh) = (first.width() as usize, first.height() as usize);
    let (bits, signed) = (first.precision() as usize, first.is_signed());
    let differ = comps.iter().any(|c| c.width() as usize != fw || c.height() as usize != fh
                                      || c.precision() as usize != bits || c.is_signed() != signed);
    let sycc = comps.len() == 3
        && (matches!(image.color_space(), jpeg2k::ColorSpace::SYCC)
            || (comps[1].width() as usize) < fw);
    let converted = if sycc {
        match sycc_planes(comps, image.x_offset() as usize, image.y_offset() as usize) {
            Some(planes) => Some(planes?),
            None if differ => {
                return Err("jpeg 2000: components differ in size or precision in a way not \
                            supported (only 4:4:4, 4:2:2 and 4:2:0 sYCC)".into());
            }
            None => None,
        }
    } else if differ {
        return Err("jpeg 2000: components differ in size or precision (subsampled)".into());
    } else {
        None
    };
    let planes: Vec<&[i32]> = match &converted {
        Some(p) => p.iter().map(|v| v.as_slice()).collect(),
        None => comps.iter().map(|c| c.data()).collect(),
    };
    let (sx, sy) = skip;
    if sx > fw || sy > fh {
        return Err("jpeg 2000: decoded area smaller than its offset".into());
    }
    let (w, h, n) = (fw - sx, fh - sy, planes.len());
    let item = if bits <= 8 { 1 } else if bits <= 16 { 2 } else { 4 };
    if w * h * n * item != output.len() {
        return Err(format!("jpeg 2000: {w} x {h} x {n} at {bits} bits does not fill {} bytes",
                           output.len()));
    }
    // typed writes (vectorised when there is one component and no skip):
    // OpenJPEG gives each component as i32 samples
    macro_rules! interleave {
        ($t:ty) => {{
            let (head, out, tail) = unsafe { output.align_to_mut::<$t>() };
            if !head.is_empty() || !tail.is_empty() {
                return Err("jpeg 2000: output buffer not aligned".into());
            }
            if n == 1 && skip == (0, 0) {
                for (dst, &v) in out.iter_mut().zip(planes[0]) {
                    *dst = (v as $t).to_le();
                }
            } else {
                for (k, plane) in planes.iter().enumerate() {
                    for row in 0..h {
                        let src = &plane[(row + sy) * fw + sx..(row + sy) * fw + fw];
                        let dst = &mut out[row * w * n + k..(row + 1) * w * n];
                        for (d, &v) in dst.iter_mut().step_by(n).zip(src) {
                            *d = (v as $t).to_le();
                        }
                    }
                }
            }
        }};
    }
    match item {
        1 => interleave!(u8),
        2 => interleave!(u16),
        _ => interleave!(u32),
    }
    Ok((w, h, n, bits, signed))
}

/// A JPEG 2000 codestream or JP2 file -> `output`, components interleaved,
/// each sample 1, 2 or 4 bytes little-endian by its precision (two's
/// complement when signed).  Returns (width, height, components, bits, signed).
pub fn jpeg2k(input: &[u8], output: &mut [u8]) -> Result<(usize, usize, usize, usize, bool), String> {
    let image = jpeg2k::Image::from_bytes(input).map_err(|e| format!("jpeg 2000: {e}"))?;
    j2k_write(&image, output, (0, 0))
}

/// The rectangle x0 <= x < x1, y0 <= y < y1 of a JPEG 2000 codestream or JP2
/// -> *output*, components interleaved: OpenJPEG decodes only the code-blocks
/// that cover it, so a region of a huge single-tile codestream costs the
/// region, not the image.  Returns (width, height, components, bits, signed)
/// of the region.
pub fn jpeg2k_area(input: &[u8], output: &mut [u8], x0: u32, y0: u32, x1: u32, y1: u32)
    -> Result<(usize, usize, usize, usize, bool), String> {
    // decoded from even coordinates and cropped: subsampled chroma (4:2:2,
    // 4:2:0) has no sample for an odd first column / row of an area, so an
    // area then equals the same rectangle of the whole image
    let (ax, ay) = (x0 & !1, y0 & !1);
    let area: jpeg2k::DecodeArea = format!("{ax}:{ay}:{x1}:{y1}").parse()
        .map_err(|e| format!("jpeg 2000: area: {e}"))?;
    let params = jpeg2k::DecodeParameters::new().decode_area(Some(area));
    let image = jpeg2k::Image::from_bytes_with(input, params)
        .map_err(|e| format!("jpeg 2000: {e}"))?;
    j2k_write(&image, output, ((x0 - ax) as usize, (y0 - ay) as usize))
}

/// The argument check of `unpack_bits`, on its own so the Python wrapper can run it
/// before taking the buffers.
pub fn unpack_bits_check(bits: u32, runlen: usize, itemsize: usize) -> Result<(), String> {
    if !(1..=32).contains(&bits) || ![1, 2, 4].contains(&itemsize) || bits as usize > 8 * itemsize
        || runlen == 0 {
        return Err(format!("unpack bits: {bits} bits into {itemsize}-byte samples, runs of {runlen}"));
    }
    Ok(())
}

/// Samples packed at `bits` bits each (1-32), most significant bit first,
/// every run of `runlen` samples (a TIFF row) starting on a byte boundary
/// -> `output` as unsigned integers of `itemsize` bytes (1, 2 or 4),
/// little-endian.  Returns the number of samples written (whole runs; the
/// data may hold fewer than `output` asks for).
pub fn unpack_bits(input: &[u8], output: &mut [u8], bits: u32, runlen: usize, itemsize: usize)
    -> Result<usize, String> {
    unpack_bits_check(bits, runlen, itemsize)?;
    let run_bytes = (runlen * bits as usize).div_ceil(8);
    let runs = (input.len() / run_bytes).min(output.len() / (runlen * itemsize));
    let mask: u64 = (1u64 << bits) - 1;
    for r in 0..runs {
        let src = &input[r * run_bytes..(r + 1) * run_bytes];
        let dst = &mut output[r * runlen * itemsize..(r + 1) * runlen * itemsize];
        let (mut acc, mut have, mut next) = (0u64, 0u32, 0usize);
        for k in 0..runlen {
            while have < bits {
                acc = (acc << 8) | src[next] as u64;
                next += 1;
                have += 8;
            }
            let value = ((acc >> (have - bits)) & mask) as u32;
            have -= bits;
            acc &= (1u64 << have) - 1;
            dst[k * itemsize..(k + 1) * itemsize].copy_from_slice(&value.to_le_bytes()[..itemsize]);
        }
    }
    Ok(runs * runlen)
}

/// The parameters of an EER stream (see `eer`).
#[derive(Clone, Copy, Debug)]
pub struct EerParams {
    pub height: usize,
    pub width: usize,
    pub skipbits: u32,
    pub horzbits: u32,
    pub vertbits: u32,
    pub superres: u32,
    pub itemsize: usize,
}

impl EerParams {
    /// The argument check of `eer`, on its own so the Python wrapper can run it
    /// before taking the buffers.
    pub fn check(&self) -> Result<(), String> {
        let EerParams { height, width, skipbits, horzbits, vertbits, itemsize, .. } = *self;
        let nbits = skipbits + horzbits + vertbits;
        if height < 1 || width < 1 || nbits > 16 || nbits <= 8 || skipbits < 4 || horzbits < 1
            || vertbits < 1 || ![1, 2, 4].contains(&itemsize) {
            return Err(format!(
                "eer: invalid parameters (skip {skipbits}, horz {horzbits}, vert {vertbits} bits, \
                 {height} x {width}, itemsize {itemsize})"));
        }
        Ok(())
    }
}

/// EER (Electron Event Representation; Thermo Fisher Falcon cameras) ->
/// electron counts per pixel in `dst` (`itemsize` 1, 2 or 4 bytes,
/// little-endian, zeroed here), `height` x `width` (super-resolved when
/// `superres` > 0).  Returns the number of electrons.
///
/// The stream is little-endian bits: a `skipbits` run of pixels without an
/// electron (all ones: skip that many and continue), then, when not all
/// ones, `horzbits` + `vertbits` sub-pixel bits of the electron at the next
/// pixel.  Ported from imagecodecs' `imcd_eer_decode` (Copyright (c)
/// 2008-2026, Christoph Gohlke, BSD-3-Clause), which follows the EER file
/// format documentation 3.0, section 4 (M. Leichsenring, 2023).
pub fn eer(src: &[u8], dst: &mut [u8], p: EerParams) -> Result<usize, String> {
    p.check()?;
    let EerParams { height, width, skipbits, horzbits, vertbits, superres, itemsize } = p;
    let nbits = skipbits + horzbits + vertbits;
    let dstsize = height * width;
    if dst.len() != dstsize * itemsize {
        return Err(format!("eer: output holds {} bytes, expected {}", dst.len(),
                           dstsize * itemsize));
    }
    if src.len() % 2 != 0 {
        return Err("eer: segment of odd length".into());
    }
    let mask = |bits: u32| -> u32 { if bits >= 32 { u32::MAX } else { (1u32 << bits) - 1 } };
    let (skipmask, horzmask, vertmask) = (mask(skipbits), mask(horzbits), mask(vertbits));
    let horzshift = horzbits.saturating_sub(superres);
    let vertshift = vertbits.saturating_sub(superres);
    let horzsize = ((horzmask as usize) + 1) >> horzshift;
    let vertsize = ((vertmask as usize) + 1) >> vertshift;
    if superres > 0 && (width % horzsize != 0 || height % vertsize != 0) {
        return Err("eer: super-resolved size not a multiple of the sub-pixel grid".into());
    }
    let width2 = width / horzsize;
    let pixelsize = dstsize / (horzsize * vertsize);
    let srcbits = (src.len() * 8) as isize - nbits as isize;
    dst.fill(0);
    let (mut bitindex, mut pixelindex, mut events) = (0isize, 0usize, 0usize);
    while bitindex < srcbits {
        let at = (bitindex / 8) as usize;
        let mut bytes = [0u8; 4];
        let n = (src.len() - at).min(4);
        bytes[..n].copy_from_slice(&src[at..at + n]);
        let mut word = u32::from_le_bytes(bytes) >> (bitindex % 8);
        let skip = word & skipmask;
        pixelindex += skip as usize;
        if pixelindex == pixelsize {
            break;
        }
        if pixelindex > pixelsize {
            return Err("eer: events beyond the image (output too small)".into());
        }
        if skip == skipmask {
            bitindex += skipbits as isize;
            continue;
        }
        let dstindex = if superres == 0 {
            pixelindex
        } else {
            word >>= skipbits;
            let h = (((word & horzmask) ^ horzbits) >> horzshift) as usize
                + (pixelindex % width2) * horzsize;
            word >>= horzbits;
            let v = (((word & vertmask) ^ vertbits) >> vertshift) as usize
                + (pixelindex / width2) * vertsize;
            v * width + h
        };
        if dstindex >= dstsize {
            return Err("eer: event beyond the image (output too small)".into());
        }
        let at = dstindex * itemsize;
        match itemsize {
            1 => dst[at] = dst[at].wrapping_add(1),
            2 => {
                let v = u16::from_le_bytes([dst[at], dst[at + 1]]).wrapping_add(1);
                dst[at..at + 2].copy_from_slice(&v.to_le_bytes());
            }
            _ => {
                let v = u32::from_le_bytes([dst[at], dst[at + 1], dst[at + 2], dst[at + 3]])
                    .wrapping_add(1);
                dst[at..at + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        pixelindex += 1;
        bitindex += nbits as isize;
        events += 1;
    }
    Ok(events)
}

/// What `jpegxr` wrote: (width, height, samples, bytes per sample, kind, order);
/// kind "uint", "float", "half" or "bool"; order "bgr" or "rgb" (the colour order
/// written, "rgb" also for one sample).
pub type JxrInfo = (usize, usize, usize, usize, &'static str, &'static str);

/// A JPEG XR codestream -> `output` through jxrlib, in the layout imagecodecs'
/// `jpegxr_decode` gives (its pixel-format table and jxrlib's converter): colour as
/// R, G, B(, A); fixed point as float32.  `output` must hold exactly the decoded
/// image.  `bgr`: an 8-bit B, G, R(, A) codestream (Zeiss CZI tiles) comes out as
/// stored, with no conversion pass.
#[cfg(feature = "jxrlib")]
pub fn jpegxr(input: &[u8], output: &mut [u8], bgr: bool) -> Result<JxrInfo, String> {
    // one pass: the shim checks the size against `output` before decoding
    let (w, h, kind, bytes, samples, is_bgr) = cdecoders::jxr_decode(input, output, bgr)?;
    let kind = match kind { 1 => "float", 2 => "half", 3 => "bool", _ => "uint" };
    Ok((w, h, samples, bytes, kind, if is_bgr { "bgr" } else { "rgb" }))
}

/// Version of the zstd library compiled in.
pub fn zstd_version() -> String {
    let v = ::zstd::zstd_safe::version_number();
    format!("{}.{}.{}", v / 10000, v / 100 % 100, v % 100)
}
