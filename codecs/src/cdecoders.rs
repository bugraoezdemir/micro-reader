//! The C decoders build.rs links: libjpeg-turbo (JPEG) and jxrlib (JPEG XR), each
//! behind a small shim in src/c that never longjmps or prints across this boundary.

#[cfg(any(feature = "libjpeg-turbo", feature = "jxrlib"))]
use std::os::raw::{c_char, c_int, c_uchar};

#[cfg(any(feature = "libjpeg-turbo", feature = "jxrlib"))]
fn text(buf: &[c_char]) -> String {
    let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// What libjpeg-turbo made of a codestream.
pub enum Jpeg {
    /// (width, height, samples, bits)
    Decoded(usize, usize, usize, usize),
    /// Not for this path (4 components), or libjpeg warned (damaged data it would
    /// repair): the caller's other decoder decides, as before.
    Declined,
    Failed(String),
}

#[cfg(feature = "libjpeg-turbo")]
extern "C" {
    fn mr_jpeg_decode(data: *const c_uchar, len: usize, out: *mut c_uchar, out_len: usize,
                      colour: c_int, width: *mut c_int, height: *mut c_int,
                      samples: *mut c_int, bits: *mut c_int, errbuf: *mut c_char,
                      errlen: usize) -> c_int;
}

/// libjpeg-turbo is linked.
pub const HAS_LIBJPEG_TURBO: bool = cfg!(feature = "libjpeg-turbo");
/// jxrlib is linked.
pub const HAS_JXRLIB: bool = cfg!(feature = "jxrlib");

/// *colour*: 0 as the file says, 1 YCbCr, 2 RGB (no conversion).
#[cfg(feature = "libjpeg-turbo")]
pub fn jpeg(input: &[u8], output: &mut [u8], colour: i32) -> Jpeg {
    let (mut w, mut h, mut s, mut b) = (0, 0, 0, 0);
    let mut err = [0 as c_char; 256];
    let code = unsafe {
        mr_jpeg_decode(input.as_ptr(), input.len(), output.as_mut_ptr(), output.len(), colour,
                       &mut w, &mut h, &mut s, &mut b, err.as_mut_ptr(), err.len())
    };
    match code {
        0 => Jpeg::Decoded(w as usize, h as usize, s as usize, b as usize),
        2 | 3 => Jpeg::Declined,
        _ => Jpeg::Failed(text(&err)),
    }
}

#[cfg(not(feature = "libjpeg-turbo"))]
pub fn jpeg(_input: &[u8], _output: &mut [u8], _colour: i32) -> Jpeg {
    Jpeg::Declined
}

#[cfg(feature = "jxrlib")]
extern "C" {
    fn mr_jxr_decode(data: *const c_uchar, len: usize, out: *mut c_uchar, out_len: usize,
                     width: *mut c_int, height: *mut c_int, kind: *mut c_int,
                     bytes: *mut c_int, samples: *mut c_int,
                     errbuf: *mut c_char, errlen: usize) -> c_int;
}

/// A JPEG XR codestream into *output*, which must hold exactly the decoded image.
/// Returns (width, height, kind, bytes per sample, samples): kind 0 unsigned, 1
/// float, 2 half float, 3 bool (the shim's `mr_jxr_info` reads the same without
/// decoding).
#[cfg(feature = "jxrlib")]
pub fn jxr_decode(input: &[u8], output: &mut [u8])
    -> Result<(usize, usize, i32, usize, usize), String> {
    let (mut w, mut h, mut k, mut b, mut s) = (0, 0, 0, 0, 0);
    let mut err = [0 as c_char; 256];
    let code = unsafe {
        mr_jxr_decode(input.as_ptr(), input.len(), output.as_mut_ptr(), output.len(),
                      &mut w, &mut h, &mut k, &mut b, &mut s, err.as_mut_ptr(), err.len())
    };
    if code != 0 {
        return Err(format!("jpegxr: {}", text(&err)));
    }
    Ok((w as usize, h as usize, k, b as usize, s as usize))
}
