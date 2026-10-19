//! Baseline JPEG decoding that gives libjpeg-turbo's pixels (what imagecodecs,
//! tifffile, Pillow and most viewers decode to), bit for bit.
//!
//! JPEG leaves a decoder free in three places, and decoders differ there by a
//! few grey levels: the inverse DCT, chroma upsampling and YCbCr -> RGB.  This
//! decoder does each the way libjpeg-turbo does by default:
//!
//! * the inverse DCT is libjpeg's accurate integer one (jidctint.c,
//!   `jpeg_idct_islow`), with its range limiting (jdmaster.c);
//! * chroma is upsampled with libjpeg-turbo's "fancy" triangle filters for 2:1
//!   (jdsample.c: `h2v1_fancy_upsample`, `h1v2_fancy_upsample`,
//!   `h2v2_fancy_upsample`; edge rows replicated as jdmainct.c does) and
//!   replicated for other ratios (`int_upsample`);
//! * YCbCr -> RGB uses libjpeg's rounded fixed-point products (jdcolor.c's
//!   tables, computed inline).
//!
//! Supported: baseline and extended sequential Huffman JPEG, 8-bit, 1 or 3
//! components, any whole-number sampling factors, restart markers, interleaved
//! and single-component scans.  Anything else (progressive, lossless,
//! arithmetic coding, 12-bit, 4 components) is `Unsupported`, and the caller
//! decodes it another way.
//!
//! The algorithms follow libjpeg (Copyright (C) the Independent JPEG Group)
//! and libjpeg-turbo (BSD-style licence); see THIRD_PARTY_LICENSES.txt.

/// Why a JPEG was not decoded here.
#[derive(Debug)]
pub enum JpegError {
    /// A kind of JPEG this decoder does not handle (the caller falls back).
    Unsupported(String),
    /// Damaged or invalid data.
    Corrupt(String),
}

type Result<T> = std::result::Result<T, JpegError>;

fn corrupt<T>(what: impl Into<String>) -> Result<T> {
    Err(JpegError::Corrupt(what.into()))
}

fn unsupported<T>(what: impl Into<String>) -> Result<T> {
    Err(JpegError::Unsupported(what.into()))
}

/// How the components are to be read: libjpeg's own choice from the markers,
/// or as the caller says (TIFF's Photometric).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Colour {
    Auto,
    YCbCr,
    Rgb,
}

/// The zig-zag order: coefficient k of a block is at natural index ORDER[k];
/// 16 extra entries absorb a run past the end in damaged data (libjpeg's
/// `jpeg_natural_order`).
const ORDER: [usize; 80] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
    20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58,
    59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    63, 63, 63, 63, 63, 63, 63,
];

// -- Huffman tables -----------------------------------------------------------------

const LOOKAHEAD: u32 = 9;

#[derive(Clone)]
struct Huffman {
    /// (length << 8 | value) for every LOOKAHEAD-bit prefix of a short code; 0: longer
    fast: Vec<u16>,
    /// the largest code of each length (-1: none), lengths 1..=16 at index 1..=16
    maxcode: [i32; 18],
    /// value index = code + valoffset[length]
    valoffset: [i32; 17],
    values: Vec<u8>,
}

impl Huffman {
    fn new(counts: &[u8; 16], values: &[u8]) -> Result<Self> {
        let mut maxcode = [-1i32; 18];
        let mut valoffset = [0i32; 17];
        let mut fast = vec![0u16; 1 << LOOKAHEAD];
        let mut code: i32 = 0;
        let mut p: i32 = 0;
        for len in 1..=16usize {
            let n = counts[len - 1] as i32;
            if n > 0 {
                valoffset[len] = p - code;
                for _ in 0..n {
                    if (len as u32) <= LOOKAHEAD {
                        let shift = LOOKAHEAD - len as u32;
                        let first = (code as usize) << shift;
                        let entry = ((len as u16) << 8) | values[p as usize] as u16;
                        for slot in &mut fast[first..first + (1 << shift)] {
                            *slot = entry;
                        }
                    }
                    code += 1;
                    p += 1;
                }
                maxcode[len] = code - 1;
            }
            if code > (1 << len) {
                return corrupt("a Huffman table with too many codes");
            }
            code <<= 1;
        }
        maxcode[17] = i32::MAX;                       // ends the long-code search
        Ok(Huffman { fast, maxcode, valoffset, values: values.to_vec() })
    }
}

// -- the entropy-coded data -------------------------------------------------------------

/// Bits of an entropy-coded segment, MSB first: 0xFF 0x00 is a 0xFF byte; any
/// other 0xFF xx is a marker, where the data ends -- as in libjpeg, the bits
/// after it read as zeros.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    acc: u64,
    count: u32,
    at_marker: bool,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Bits { data, pos, acc: 0, count: 0, at_marker: false }
    }

    #[inline]
    fn fill(&mut self) {
        while self.count <= 56 {
            let mut byte = 0u8;
            if !self.at_marker {
                if self.pos < self.data.len() {
                    let b = self.data[self.pos];
                    if b == 0xFF {
                        let mut next = self.pos + 1;
                        while next < self.data.len() && self.data[next] == 0xFF {
                            next += 1;                // fill bytes before a marker
                        }
                        if next < self.data.len() && self.data[next] == 0x00 {
                            byte = 0xFF;
                            self.pos = next + 1;
                        } else {
                            self.at_marker = true;   // pos stays at the marker's 0xFF
                        }
                    } else {
                        byte = b;
                        self.pos += 1;
                    }
                } else {
                    self.at_marker = true;
                }
            }
            self.acc |= (byte as u64) << (56 - self.count);
            self.count += 8;
        }
    }

    #[inline]
    fn peek(&mut self, n: u32) -> u32 {
        if self.count < n {
            self.fill();
        }
        (self.acc >> (64 - n)) as u32
    }

    #[inline]
    fn skip(&mut self, n: u32) {
        self.acc <<= n;
        self.count -= n;
    }

    #[inline]
    fn bits(&mut self, n: u32) -> i32 {
        if n == 0 {
            return 0;
        }
        let v = self.peek(n);
        self.skip(n);
        v as i32
    }

    #[inline]
    fn decode(&mut self, table: &Huffman) -> u8 {
        let look = self.peek(16);
        let entry = table.fast[(look >> (16 - LOOKAHEAD)) as usize];
        if entry != 0 {
            self.skip((entry >> 8) as u32);
            return entry as u8;
        }
        let mut len = LOOKAHEAD as usize + 1;
        while len <= 16 && ((look >> (16 - len)) as i32) > table.maxcode[len] {
            len += 1;
        }
        if len > 16 {
            self.skip(16);
            return 0;                                  // libjpeg: a bad code decodes to 0
        }
        self.skip(len as u32);
        let index = (look >> (16 - len)) as i32 + table.valoffset[len];
        table.values.get(index as usize).copied().unwrap_or(0)
    }

    /// After a restart interval: the rest of the byte is dropped and the next
    /// RSTn marker consumed (any n; libjpeg checks the number and resyncs).
    fn restart(&mut self) {
        self.acc = 0;
        self.count = 0;
        let mut p = self.pos;
        while p + 1 < self.data.len() && !(self.data[p] == 0xFF && self.data[p + 1] != 0xFF
                                            && self.data[p + 1] != 0x00) {
            p += 1;
        }
        if p + 1 < self.data.len() && (0xD0..=0xD7).contains(&self.data[p + 1]) {
            p += 2;
        }
        self.pos = p;
        self.at_marker = false;
    }

    /// Where the next marker starts, after the scan.
    fn marker_pos(&self) -> usize {
        let mut p = self.pos;
        while p + 1 < self.data.len() && !(self.data[p] == 0xFF && self.data[p + 1] != 0x00
                                            && self.data[p + 1] != 0xFF) {
            p += 1;
        }
        p
    }
}

#[inline]
fn extend(v: i32, t: u32) -> i32 {
    if t == 0 {
        0
    } else if v < (1 << (t - 1)) {
        v - (1 << t) + 1
    } else {
        v
    }
}

// -- the inverse DCT: jidctint.c, jpeg_idct_islow --------------------------------------

const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;
const FIX_0_298631336: i32 = 2446;
const FIX_0_390180644: i32 = 3196;
const FIX_0_541196100: i32 = 4433;
const FIX_0_765366865: i32 = 6270;
const FIX_0_899976223: i32 = 7373;
const FIX_1_175875602: i32 = 9633;
const FIX_1_501321110: i32 = 12299;
const FIX_1_847759065: i32 = 15137;
const FIX_1_961570560: i32 = 16069;
const FIX_2_053119869: i32 = 16819;
const FIX_2_562915447: i32 = 20995;
const FIX_3_072711026: i32 = 25172;

#[inline]
fn descale(x: i32, n: i32) -> i32 {
    (x + (1 << (n - 1))) >> n
}

/// libjpeg's post-IDCT range limit: the low 10 bits as a signed value, plus
/// the centre 128, clamped to 0..=255 (`IDCT_range_limit` with RANGE_MASK).
#[inline]
fn range_limit(x: i32) -> u8 {
    let v = x & 1023;
    let s = if v >= 512 { v - 1024 } else { v };
    (s + 128).clamp(0, 255) as u8
}

/// `coef` (natural order) times `quant` (natural order) -> 8 x 8 samples at
/// `out[row * stride + col]`.
fn idct_islow(coef: &[i32; 64], quant: &[u16; 64], out: &mut [u8], stride: usize) {
    let mut ws = [0i32; 64];
    for col in 0..8 {
        let c = |r: usize| coef[r * 8 + col] * quant[r * 8 + col] as i32;
        if (1..8).all(|r| coef[r * 8 + col] == 0) {
            let dc = c(0) << PASS1_BITS;
            for r in 0..8 {
                ws[r * 8 + col] = dc;
            }
            continue;
        }
        // even part
        let (z2, z3) = (c(2), c(6));
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * -FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let (z2, z3) = (c(0), c(4));
        let tmp0 = (z2 + z3) << CONST_BITS;
        let tmp1 = (z2 - z3) << CONST_BITS;
        let (tmp10, tmp13, tmp11, tmp12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
        // odd part
        let (mut t0, mut t1, mut t2, mut t3) = (c(7), c(5), c(3), c(1));
        let (z1, z2, z3, z4) = (t0 + t3, t1 + t2, t0 + t2, t1 + t3);
        let z5 = (z3 + z4) * FIX_1_175875602;
        t0 *= FIX_0_298631336;
        t1 *= FIX_2_053119869;
        t2 *= FIX_3_072711026;
        t3 *= FIX_1_501321110;
        let z1 = z1 * -FIX_0_899976223;
        let z2 = z2 * -FIX_2_562915447;
        let z3 = z3 * -FIX_1_961570560 + z5;
        let z4 = z4 * -FIX_0_390180644 + z5;
        t0 += z1 + z3;
        t1 += z2 + z4;
        t2 += z2 + z3;
        t3 += z1 + z4;
        let n = CONST_BITS - PASS1_BITS;
        ws[col] = descale(tmp10 + t3, n);
        ws[7 * 8 + col] = descale(tmp10 - t3, n);
        ws[8 + col] = descale(tmp11 + t2, n);
        ws[6 * 8 + col] = descale(tmp11 - t2, n);
        ws[2 * 8 + col] = descale(tmp12 + t1, n);
        ws[5 * 8 + col] = descale(tmp12 - t1, n);
        ws[3 * 8 + col] = descale(tmp13 + t0, n);
        ws[4 * 8 + col] = descale(tmp13 - t0, n);
    }
    for row in 0..8 {
        let w = &ws[row * 8..row * 8 + 8];
        let o = &mut out[row * stride..row * stride + 8];
        if w[1..].iter().all(|&v| v == 0) {
            let v = range_limit(descale(w[0], PASS1_BITS + 3));
            o.fill(v);
            continue;
        }
        let (z2, z3) = (w[2], w[6]);
        let z1 = (z2 + z3) * FIX_0_541196100;
        let tmp2 = z1 + z3 * -FIX_1_847759065;
        let tmp3 = z1 + z2 * FIX_0_765366865;
        let tmp0 = (w[0] + w[4]) << CONST_BITS;
        let tmp1 = (w[0] - w[4]) << CONST_BITS;
        let (tmp10, tmp13, tmp11, tmp12) = (tmp0 + tmp3, tmp0 - tmp3, tmp1 + tmp2, tmp1 - tmp2);
        let (mut t0, mut t1, mut t2, mut t3) = (w[7], w[5], w[3], w[1]);
        let (z1, z2, z3, z4) = (t0 + t3, t1 + t2, t0 + t2, t1 + t3);
        let z5 = (z3 + z4) * FIX_1_175875602;
        t0 *= FIX_0_298631336;
        t1 *= FIX_2_053119869;
        t2 *= FIX_3_072711026;
        t3 *= FIX_1_501321110;
        let z1 = z1 * -FIX_0_899976223;
        let z2 = z2 * -FIX_2_562915447;
        let z3 = z3 * -FIX_1_961570560 + z5;
        let z4 = z4 * -FIX_0_390180644 + z5;
        t0 += z1 + z3;
        t1 += z2 + z4;
        t2 += z2 + z3;
        t3 += z1 + z4;
        let n = CONST_BITS + PASS1_BITS + 3;
        o[0] = range_limit(descale(tmp10 + t3, n));
        o[7] = range_limit(descale(tmp10 - t3, n));
        o[1] = range_limit(descale(tmp11 + t2, n));
        o[6] = range_limit(descale(tmp11 - t2, n));
        o[2] = range_limit(descale(tmp12 + t1, n));
        o[5] = range_limit(descale(tmp12 - t1, n));
        o[3] = range_limit(descale(tmp13 + t0, n));
        o[4] = range_limit(descale(tmp13 - t0, n));
    }
}

// -- the frame ----------------------------------------------------------------------------

#[derive(Clone)]
struct Component {
    id: u8,
    h: usize,
    v: usize,
    tq: usize,
    /// samples of this component across and down the image (libjpeg's
    /// downsampled_width / _height)
    width: usize,
    height: usize,
    /// its decoded samples: whole MCUs, `stride` wide
    plane: Vec<u8>,
    stride: usize,
    dc_pred: i32,
}

struct Decoder<'a> {
    data: &'a [u8],
    quant: [Option<[u16; 64]>; 4],
    dc: [Option<Huffman>; 4],
    ac: [Option<Huffman>; 4],
    restart_interval: usize,
    jfif: bool,
    adobe: Option<u8>,
    width: usize,
    height: usize,
    hmax: usize,
    vmax: usize,
    mcus_x: usize,
    mcus_y: usize,
    components: Vec<Component>,
    frame_seen: bool,
}

fn be16(data: &[u8], at: usize) -> Result<usize> {
    match data.get(at..at + 2) {
        Some(b) => Ok(((b[0] as usize) << 8) | b[1] as usize),
        None => corrupt("the data ends inside a marker"),
    }
}

impl<'a> Decoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Decoder {
            data,
            quant: [None; 4],
            dc: [None, None, None, None],
            ac: [None, None, None, None],
            restart_interval: 0,
            jfif: false,
            adobe: None,
            width: 0,
            height: 0,
            hmax: 1,
            vmax: 1,
            mcus_x: 0,
            mcus_y: 0,
            components: Vec::new(),
            frame_seen: false,
        }
    }

    /// Reads every marker segment and scan; the decoded samples end in the
    /// components' planes.
    fn run(&mut self) -> Result<()> {
        let data = self.data;
        if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
            return corrupt("not a JPEG (no SOI marker)");
        }
        let mut p = 2;
        loop {
            // find the next marker (skipping fill bytes and stray data)
            while p < data.len() && data[p] != 0xFF {
                p += 1;
            }
            while p < data.len() && data[p] == 0xFF {
                p += 1;
            }
            if p >= data.len() {
                if self.frame_seen {
                    return Ok(());                    // no EOI: what was decoded stands
                }
                return corrupt("the data ends before the image");
            }
            let marker = data[p];
            p += 1;
            match marker {
                0xD9 => return Ok(()),                 // EOI
                0xD0..=0xD7 | 0x01 => continue,        // stray RSTn, TEM: no length
                _ => {}
            }
            let len = be16(data, p)?;
            if len < 2 || p + len > data.len() {
                return corrupt("a marker segment runs past the data");
            }
            let seg = &data[p + 2..p + len];
            match marker {
                0xC0 | 0xC1 => self.frame(seg)?,
                0xC2 | 0xC6 | 0xCA | 0xCE => return unsupported("progressive JPEG"),
                0xC3 | 0xC7 | 0xCB | 0xCF => return unsupported("lossless JPEG"),
                0xC5 | 0xC9 | 0xCD => return unsupported("arithmetic-coded JPEG"),
                0xC4 => self.huffman(seg)?,
                0xCC => return unsupported("arithmetic-coded JPEG"),
                0xDB => self.quantization(seg)?,
                0xDD => self.restart_interval = be16(seg, 0)?,
                0xE0 => {
                    if seg.len() >= 5 && &seg[..5] == b"JFIF\0" {
                        self.jfif = true;
                    }
                }
                0xEE => {
                    if seg.len() >= 12 && &seg[..5] == b"Adobe" {
                        self.adobe = Some(seg[11]);
                    }
                }
                0xDA => {
                    p = self.scan(seg, p + len)?;
                    continue;
                }
                0xDC => return unsupported("a JPEG whose height comes in a DNL marker"),
                _ => {}                                // APPn, COM, ...: nothing to do
            }
            p += len;
        }
    }

    fn quantization(&mut self, mut seg: &[u8]) -> Result<()> {
        while !seg.is_empty() {
            let (pq, tq) = ((seg[0] >> 4) as usize, (seg[0] & 15) as usize);
            if tq > 3 || pq > 1 {
                return corrupt("a bad quantization table");
            }
            let n = if pq == 0 { 64 } else { 128 };
            if seg.len() < 1 + n {
                return corrupt("a quantization table runs short");
            }
            let mut table = [0u16; 64];
            for k in 0..64 {
                table[ORDER[k]] = if pq == 0 {
                    seg[1 + k] as u16
                } else {
                    ((seg[1 + 2 * k] as u16) << 8) | seg[2 + 2 * k] as u16
                };
            }
            self.quant[tq] = Some(table);
            seg = &seg[1 + n..];
        }
        Ok(())
    }

    fn huffman(&mut self, mut seg: &[u8]) -> Result<()> {
        while !seg.is_empty() {
            if seg.len() < 17 {
                return corrupt("a Huffman table runs short");
            }
            let (tc, th) = (seg[0] >> 4, (seg[0] & 15) as usize);
            if tc > 1 || th > 3 {
                return corrupt("a bad Huffman table");
            }
            let mut counts = [0u8; 16];
            counts.copy_from_slice(&seg[1..17]);
            let total: usize = counts.iter().map(|&c| c as usize).sum();
            if total > 256 || seg.len() < 17 + total {
                return corrupt("a Huffman table runs short");
            }
            let table = Huffman::new(&counts, &seg[17..17 + total])?;
            if tc == 0 {
                self.dc[th] = Some(table);
            } else {
                self.ac[th] = Some(table);
            }
            seg = &seg[17 + total..];
        }
        Ok(())
    }

    fn frame(&mut self, seg: &[u8]) -> Result<()> {
        if self.frame_seen {
            return corrupt("two frames in one JPEG");
        }
        if seg.len() < 6 {
            return corrupt("a frame header runs short");
        }
        if seg[0] != 8 {
            return unsupported(format!("{}-bit JPEG", seg[0]));
        }
        self.height = be16(seg, 1)?;
        self.width = be16(seg, 3)?;
        let n = seg[5] as usize;
        if self.height == 0 {
            return unsupported("a JPEG whose height comes in a DNL marker");
        }
        if self.width == 0 || seg.len() < 6 + 3 * n {
            return corrupt("a bad frame header");
        }
        if n != 1 && n != 3 {
            return unsupported(format!("a JPEG with {n} components"));
        }
        for i in 0..n {
            let c = &seg[6 + 3 * i..9 + 3 * i];
            let (mut h, mut v) = ((c[1] >> 4) as usize, (c[1] & 15) as usize);
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) || c[2] > 3 {
                return corrupt("bad sampling factors or quantization table");
            }
            if n == 1 {
                (h, v) = (1, 1);                       // one component: one block per MCU
            }
            self.components.push(Component {
                id: c[0], h, v, tq: c[2] as usize, width: 0, height: 0, plane: Vec::new(),
                stride: 0, dc_pred: 0,
            });
        }
        self.hmax = self.components.iter().map(|c| c.h).max().unwrap_or(1);
        self.vmax = self.components.iter().map(|c| c.v).max().unwrap_or(1);
        if self.components.iter().any(|c| self.hmax % c.h != 0 || self.vmax % c.v != 0) {
            return unsupported("fractional chroma sampling");
        }
        self.mcus_x = self.width.div_ceil(8 * self.hmax);
        self.mcus_y = self.height.div_ceil(8 * self.vmax);
        let (w, h, hm, vm) = (self.width, self.height, self.hmax, self.vmax);
        let (mx, my) = (self.mcus_x, self.mcus_y);
        for c in &mut self.components {
            c.width = (w * c.h).div_ceil(hm);
            c.height = (h * c.v).div_ceil(vm);
            c.stride = mx * c.h * 8;
            c.plane = vec![0u8; c.stride * my * c.v * 8];
        }
        self.frame_seen = true;
        Ok(())
    }

    /// One scan; returns where the data after it starts.
    fn scan(&mut self, seg: &[u8], data_start: usize) -> Result<usize> {
        if !self.frame_seen {
            return corrupt("a scan before the frame header");
        }
        let ns = *seg.first().ok_or(JpegError::Corrupt("an empty scan header".into()))? as usize;
        if ns == 0 || seg.len() < 1 + 2 * ns + 3 {
            return corrupt("a bad scan header");
        }
        let mut members = Vec::with_capacity(ns);
        for i in 0..ns {
            let (cs, t) = (seg[1 + 2 * i], seg[2 + 2 * i]);
            let ci = self.components.iter().position(|c| c.id == cs)
                .ok_or(JpegError::Corrupt("a scan names an unknown component".into()))?;
            let (td, ta) = ((t >> 4) as usize, (t & 15) as usize);
            if td > 3 || ta > 3 || self.dc[td].is_none() || self.ac[ta].is_none() {
                return corrupt("a scan uses a missing Huffman table");
            }
            if self.quant[self.components[ci].tq].is_none() {
                return corrupt("a component uses a missing quantization table");
            }
            members.push((ci, td, ta));
        }
        let (ss, se, a) = (seg[1 + 2 * ns], seg[2 + 2 * ns], seg[3 + 2 * ns]);
        if ss != 0 || se != 63 || a != 0 {
            return unsupported("a scan that is not sequential");
        }
        for &(ci, _, _) in &members {
            self.components[ci].dc_pred = 0;
        }
        let tables: Vec<(Huffman, Huffman, [u16; 64])> = members.iter().map(|&(ci, td, ta)| {
            (self.dc[td].clone().unwrap(), self.ac[ta].clone().unwrap(),
             self.quant[self.components[ci].tq].unwrap())
        }).collect();
        let mut bits = Bits::new(self.data, data_start);
        let mut coef = [0i32; 64];
        // (component, block column, block row) of every block of each MCU
        let (units_x, units_y, interleaved) = if ns == 1 {
            let c = &self.components[members[0].0];
            (c.width.div_ceil(8), c.height.div_ceil(8), false)
        } else {
            (self.mcus_x, self.mcus_y, true)
        };
        let mut left = self.restart_interval;
        for my in 0..units_y {
            for mx in 0..units_x {
                if self.restart_interval > 0 {
                    if left == 0 {
                        bits.restart();
                        for &(ci, _, _) in &members {
                            self.components[ci].dc_pred = 0;
                        }
                        left = self.restart_interval;
                    }
                    left -= 1;
                }
                for (m, &(ci, _, _)) in members.iter().enumerate() {
                    let (dc, ac, quant) = &tables[m];
                    let comp = &mut self.components[ci];
                    let (bh, bv) = if interleaved { (comp.h, comp.v) } else { (1, 1) };
                    for by in 0..bv {
                        for bx in 0..bh {
                            coef.fill(0);
                            let t = bits.decode(dc) as u32;
                            let diff = extend(bits.bits(t.min(16)), t.min(16));
                            comp.dc_pred += diff;
                            coef[0] = comp.dc_pred;
                            let mut k = 1usize;
                            while k < 64 {
                                let rs = bits.decode(ac);
                                let (r, s) = ((rs >> 4) as usize, (rs & 15) as u32);
                                if s != 0 {
                                    k += r;
                                    coef[ORDER[k]] = extend(bits.bits(s), s);
                                    k += 1;
                                } else {
                                    if r != 15 {
                                        break;
                                    }
                                    k += 16;
                                }
                            }
                            let col = (mx * bh + bx) * 8;
                            let row = (my * bv + by) * 8;
                            let stride = comp.stride;
                            let rows = comp.plane.len() / stride;
                            if col + 8 <= stride && row + 8 <= rows {
                                idct_islow(&coef, quant, &mut comp.plane[row * stride + col..],
                                           stride);
                            }
                        }
                    }
                }
            }
        }
        Ok(bits.marker_pos())
    }

    /// Whether the three components are YCbCr, as libjpeg decides it
    /// (jdapimin.c, `default_decompress_parms`) or the caller says.
    fn is_ycbcr(&self, colour: Colour) -> bool {
        match colour {
            Colour::YCbCr => true,
            Colour::Rgb => false,
            Colour::Auto => {
                if self.jfif {
                    true
                } else if let Some(t) = self.adobe {
                    t != 0
                } else {
                    let ids: Vec<u8> = self.components.iter().map(|c| c.id).collect();
                    !(ids == [b'R', b'G', b'B'])
                }
            }
        }
    }
}

// -- upsampling: jdsample.c -----------------------------------------------------------

/// Component `c`'s samples brought to the image's size, row `y` -> `out`: `width`
/// samples, and `out` must hold `width + 1` (2:1 upsampling writes whole pairs).
fn upsampled_row(c: &Component, hexp: usize, vexp: usize, y: usize, width: usize, out: &mut [u8]) {
    let row = |r: usize| &c.plane[r.min(c.height - 1) * c.stride..][..c.stride];
    let dw = c.width;
    match (hexp, vexp) {
        (1, 1) => out[..width].copy_from_slice(&row(y)[..width]),
        (2, 1) if dw > 2 => {
            // h2v1_fancy_upsample
            let inp = &row(y)[..dw];
            let o = &mut out[..2 * dw];
            o[0] = inp[0];
            o[1] = ((inp[0] as u32 * 3 + inp[1] as u32 + 2) >> 2) as u8;
            for i in 1..dw - 1 {
                let v = inp[i] as u32 * 3;
                o[2 * i] = ((v + inp[i - 1] as u32 + 1) >> 2) as u8;
                o[2 * i + 1] = ((v + inp[i + 1] as u32 + 2) >> 2) as u8;
            }
            o[2 * dw - 2] = ((inp[dw - 1] as u32 * 3 + inp[dw - 2] as u32 + 1) >> 2) as u8;
            o[2 * dw - 1] = inp[dw - 1];
        }
        (1, 2) => {
            // h1v2_fancy_upsample
            let (r, v) = (y / 2, y % 2);
            let other = if v == 0 { r.saturating_sub(1) } else { r + 1 };
            let (a, b) = (row(r), row(other));
            let bias = if v == 0 { 1 } else { 2 };
            for x in 0..width {
                out[x] = ((a[x] as u32 * 3 + b[x] as u32 + bias) >> 2) as u8;
            }
        }
        (2, 2) if dw > 2 => {
            // h2v2_fancy_upsample
            let (r, v) = (y / 2, y % 2);
            let other = if v == 0 { r.saturating_sub(1) } else { r + 1 };
            let (a, b) = (&row(r)[..dw], &row(other)[..dw]);
            let o = &mut out[..2 * dw];
            let sum = |i: usize| a[i] as i32 * 3 + b[i] as i32;
            let mut this = sum(0);
            let mut next = sum(1);
            o[0] = ((this * 4 + 8) >> 4) as u8;
            o[1] = ((this * 3 + next + 7) >> 4) as u8;
            let mut last = this;
            this = next;
            for i in 2..dw {
                next = sum(i);
                o[2 * i - 2] = ((this * 3 + last + 8) >> 4) as u8;
                o[2 * i - 1] = ((this * 3 + next + 7) >> 4) as u8;
                last = this;
                this = next;
            }
            o[2 * dw - 2] = ((this * 3 + last + 8) >> 4) as u8;
            o[2 * dw - 1] = ((this * 4 + 7) >> 4) as u8;
        }
        _ => {
            // h2v1_upsample, h2v2_upsample, int_upsample: replicated samples
            let inp = row(y / vexp);
            for x in 0..width {
                out[x] = inp[x / hexp];
            }
        }
    }
}

// -- YCbCr -> RGB: jdcolor.c -------------------------------------------------------------

/// jdcolor.c's tables, as products: entry `i` is `(FIX(k) * (i - 128) + ONE_HALF) >> 16`.
/// Computed inline, so the row loop vectorises; the results equal the table lookups.
const FIX_1_40200: i32 = 91881;
const FIX_1_77200: i32 = 116130;
const FIX_0_71414: i32 = 46802;
const FIX_0_34414: i32 = 22554;
const ONE_HALF: i32 = 1 << 15;

fn ycbcr_row(ys: &[u8], cbs: &[u8], crs: &[u8], dst: &mut [u8]) {
    for (((px, &y), &cb), &cr) in dst.chunks_exact_mut(3).zip(ys).zip(cbs).zip(crs) {
        let (y, cb, cr) = (y as i32, cb as i32 - 128, cr as i32 - 128);
        px[0] = (y + ((FIX_1_40200 * cr + ONE_HALF) >> 16)).clamp(0, 255) as u8;
        px[1] = (y + ((-FIX_0_34414 * cb - FIX_0_71414 * cr + ONE_HALF) >> 16)).clamp(0, 255) as u8;
        px[2] = (y + ((FIX_1_77200 * cb + ONE_HALF) >> 16)).clamp(0, 255) as u8;
    }
}

// -- the API ------------------------------------------------------------------------------

/// Decodes `data` -> `out` (rows x cols x components, 8-bit, R G B order).
/// Returns (width, height, components).
pub fn decode_into(data: &[u8], out: &mut [u8], colour: Colour) -> Result<(usize, usize, usize)> {
    let mut d = Decoder::new(data);
    d.run()?;
    if !d.frame_seen {
        return corrupt("no frame header");
    }
    let (w, h, n) = (d.width, d.height, d.components.len());
    if out.len() != w * h * n {
        return corrupt(format!("decodes to {} bytes, expected {}", w * h * n, out.len()));
    }
    // one more sample than the image is wide: 2:1 upsampling writes whole pairs
    let mut rows: Vec<Vec<u8>> = vec![vec![0u8; w + 1]; n];
    let ycbcr = n == 3 && d.is_ycbcr(colour);
    for y in 0..h {
        // full-size components are read in place, the others upsampled into `rows`
        for (i, c) in d.components.iter().enumerate() {
            if c.h != d.hmax || c.v != d.vmax {
                upsampled_row(c, d.hmax / c.h, d.vmax / c.v, y, w, &mut rows[i]);
            }
        }
        let line = |i: usize| -> &[u8] {
            let c = &d.components[i];
            if c.h == d.hmax && c.v == d.vmax {
                &c.plane[y * c.stride..][..w]
            } else {
                &rows[i][..w]
            }
        };
        let dst = &mut out[y * w * n..(y + 1) * w * n];
        if n == 1 {
            dst.copy_from_slice(line(0));
        } else if ycbcr {
            ycbcr_row(line(0), line(1), line(2), dst);
        } else {
            let (r, g, b) = (line(0), line(1), line(2));
            for (((px, &r), &g), &b) in dst.chunks_exact_mut(3).zip(r).zip(g).zip(b) {
                px[0] = r;
                px[1] = g;
                px[2] = b;
            }
        }
    }
    Ok((w, h, n))
}
