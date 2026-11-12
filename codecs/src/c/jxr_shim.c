/* JPEG XR -> a caller's buffer, through Microsoft's jxrlib (BSD-2, vendor/jxrlib).
 *
 * Follows imagecodecs' jpegxr_decode (imagecodecs/_jpegxr.pyx, BSD-3) step by step so
 * the pixels are the same: the decoder from bytes, the output pixel format chosen from
 * the stored one by the same table (fp2int=False), jxrlib's format converter, and the
 * same channel fix-ups afterwards.  Colour images come out R, G, B(, A), or, when the
 * caller asks for B, G, R order (keep_bgr) and the codestream stores 8-bit B, G, R(, A)
 * as Zeiss CZI tiles do, as stored: no conversion pass and no copy after decoding.
 *
 * mr_jxr_info() reads the size and output layout without decoding, so the caller can
 * size the buffer; mr_jxr_decode() decodes into it.
 */
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "JXRGlue.h"

/* sample kinds returned to Rust */
#define MR_UINT 0
#define MR_FLOAT 1
#define MR_HALF 2
#define MR_BOOL 3

typedef struct {
    int kind;        /* MR_UINT / MR_FLOAT / MR_HALF / MR_BOOL */
    int bytes;       /* bytes per sample */
    int samples;     /* samples per pixel in the converter's output */
    int alpha;       /* uAlphaMode */
    int strip;       /* 0 none, 1 drop last, 2 drop last + BGR -> RGB, 3 BGRA -> RGBA */
    int bgr;         /* 1: colour comes out B, G, R(, A) */
} mr_layout;

/* imagecodecs' jxr_decode_guid with fp2int=False: the output format for a stored
 * format (changed in place) and its layout.  keep_bgr: 8-bit B, G, R(, A) stays as
 * stored instead of going through the converter to R, G, B(, A). */
static ERR layout_for(PKPixelFormatGUID* pf, mr_layout* o, int keep_bgr)
{
#define IS(g) IsEqualGUID(pf, &GUID_PKPixelFormat##g)
#define TO(g) (*pf = GUID_PKPixelFormat##g)
#define OUT(k, b, s, a, st) do { o->kind = k; o->bytes = b; o->samples = s; \
        o->alpha = a; o->strip = st; o->bgr = 0; return WMP_errSuccess; } while (0)
#define OUT_BGR(s, a, st) do { o->kind = MR_UINT; o->bytes = 1; o->samples = s; \
        o->alpha = a; o->strip = st; o->bgr = 1; return WMP_errSuccess; } while (0)
    if (keep_bgr) {
        if (IS(24bppBGR)) OUT_BGR(3, 0, 0);
        if (IS(32bppBGR)) OUT_BGR(4, 0, 1);      /* drop the padding byte */
        if (IS(32bppBGRA)) OUT_BGR(4, 2, 0);
    }
    if (IS(BlackWhite)) { TO(8bppGray); OUT(MR_BOOL, 1, 1, 0, 0); }
    /* uint8 */
    if (IS(8bppGray)) OUT(MR_UINT, 1, 1, 0, 0);
    if (IS(24bppRGB)) OUT(MR_UINT, 1, 3, 0, 0);
    if (IS(16bppRGB555) || IS(16bppRGB565) || IS(24bppBGR) || IS(32bppRGB)) {
        TO(24bppRGB); OUT(MR_UINT, 1, 3, 0, 0);
    }
    if (IS(32bppBGR)) OUT(MR_UINT, 1, 4, 0, 2);
    if (IS(32bppBGRA)) { TO(32bppRGBA); OUT(MR_UINT, 1, 4, 2, 0); }
    if (IS(32bppRGBA)) OUT(MR_UINT, 1, 4, 2, 0);
    if (IS(32bppPBGRA)) OUT(MR_UINT, 1, 4, 2, 3);
    if (IS(32bppPRGBA)) OUT(MR_UINT, 1, 4, 2, 0);
    if (IS(32bppRGBE)) OUT(MR_UINT, 1, 4, 0, 0);
    if (IS(32bppCMYK)) OUT(MR_UINT, 1, 4, 0, 0);
    if (IS(40bppCMYKAlpha)) OUT(MR_UINT, 1, 5, 2, 0);
    if (IS(24bpp3Channels)) OUT(MR_UINT, 1, 3, 0, 0);
    if (IS(32bpp4Channels)) OUT(MR_UINT, 1, 4, 0, 0);
    if (IS(40bpp5Channels)) OUT(MR_UINT, 1, 5, 0, 0);
    if (IS(48bpp6Channels)) OUT(MR_UINT, 1, 6, 0, 0);
    if (IS(56bpp7Channels)) OUT(MR_UINT, 1, 7, 0, 0);
    if (IS(64bpp8Channels)) OUT(MR_UINT, 1, 8, 0, 0);
    if (IS(32bpp3ChannelsAlpha)) OUT(MR_UINT, 1, 4, 2, 0);
    if (IS(40bpp4ChannelsAlpha)) OUT(MR_UINT, 1, 5, 2, 0);
    if (IS(48bpp5ChannelsAlpha)) OUT(MR_UINT, 1, 6, 2, 0);
    if (IS(56bpp6ChannelsAlpha)) OUT(MR_UINT, 1, 7, 2, 0);
    if (IS(64bpp7ChannelsAlpha)) OUT(MR_UINT, 1, 8, 2, 0);
    if (IS(72bpp8ChannelsAlpha)) OUT(MR_UINT, 1, 9, 2, 0);
    /* uint16 */
    if (IS(16bppGray)) OUT(MR_UINT, 2, 1, 0, 0);
    if (IS(48bppRGB)) OUT(MR_UINT, 2, 3, 0, 0);
    if (IS(64bppRGBA) || IS(64bppPRGBA)) OUT(MR_UINT, 2, 4, 2, 0);
    if (IS(64bppCMYK)) OUT(MR_UINT, 2, 4, 0, 0);
    if (IS(80bppCMYKAlpha)) OUT(MR_UINT, 2, 5, 2, 0);
    if (IS(32bppRGB101010)) { TO(48bppRGB); OUT(MR_UINT, 2, 3, 0, 0); }
    if (IS(48bpp3Channels)) OUT(MR_UINT, 2, 3, 0, 0);
    if (IS(64bpp4Channels)) OUT(MR_UINT, 2, 4, 0, 0);
    if (IS(80bpp5Channels)) OUT(MR_UINT, 2, 5, 0, 0);
    if (IS(96bpp6Channels)) OUT(MR_UINT, 2, 6, 0, 0);
    if (IS(112bpp7Channels)) OUT(MR_UINT, 2, 7, 0, 0);
    if (IS(128bpp8Channels)) OUT(MR_UINT, 2, 8, 0, 0);
    if (IS(64bpp3ChannelsAlpha)) OUT(MR_UINT, 2, 4, 2, 0);
    if (IS(80bpp4ChannelsAlpha)) OUT(MR_UINT, 2, 5, 2, 0);
    if (IS(96bpp5ChannelsAlpha)) OUT(MR_UINT, 2, 6, 2, 0);
    if (IS(112bpp6ChannelsAlpha)) OUT(MR_UINT, 2, 7, 2, 0);
    if (IS(128bpp7ChannelsAlpha)) OUT(MR_UINT, 2, 8, 2, 0);
    if (IS(144bpp8ChannelsAlpha)) OUT(MR_UINT, 2, 9, 2, 0);
    /* float32 */
    if (IS(32bppGrayFloat)) OUT(MR_FLOAT, 4, 1, 0, 0);
    if (IS(96bppRGBFloat)) OUT(MR_FLOAT, 4, 3, 0, 0);
    if (IS(128bppRGBFloat)) { TO(96bppRGBFloat); OUT(MR_FLOAT, 4, 3, 0, 0); }
    if (IS(128bppRGBAFloat) || IS(128bppPRGBAFloat)) OUT(MR_FLOAT, 4, 4, 2, 0);
    /* float16 */
    if (IS(16bppGrayHalf)) OUT(MR_HALF, 2, 1, 0, 0);
    if (IS(48bppRGBHalf)) OUT(MR_HALF, 2, 3, 0, 0);
    if (IS(64bppRGBHalf)) { TO(48bppRGBHalf); OUT(MR_HALF, 2, 3, 0, 0); }
    if (IS(64bppRGBAHalf)) OUT(MR_HALF, 2, 4, 2, 0);
    /* fixed point -> float32 */
    if (IS(16bppGrayFixedPoint) || IS(32bppGrayFixedPoint)) {
        TO(32bppGrayFloat); OUT(MR_FLOAT, 4, 1, 0, 0);
    }
    if (IS(48bppRGBFixedPoint) || IS(64bppRGBFixedPoint) || IS(96bppRGBFixedPoint)
            || IS(128bppRGBFixedPoint)) {
        TO(96bppRGBFloat); OUT(MR_FLOAT, 4, 3, 0, 0);
    }
    if (IS(64bppRGBAFixedPoint) || IS(128bppRGBAFixedPoint)) {
        TO(128bppRGBAFloat); OUT(MR_FLOAT, 4, 4, 2, 0);
    }
#undef IS
#undef TO
#undef OUT
#undef OUT_BGR
    return WMP_errUnsupportedFormat;
}

static ERR open_decoder(const unsigned char* data, size_t len, PKImageDecode** out)
{
    const PKIID* iid = NULL;
    struct WMPStream* stream = NULL;
    PKImageDecode* decoder = NULL;
    ERR err = GetImageDecodeIID(".jxr", &iid);
    if (err) return err;
    err = CreateWS_Memory(&stream, (void*)data, len);
    if (err) return err;
    err = PKCodecFactory_CreateCodec(iid, (void**)&decoder);
    if (err) { stream->Close(&stream); return err; }
    err = decoder->Initialize(decoder, stream);
    decoder->fStreamOwner = 1;               /* the decoder closes the stream */
    *out = decoder;
    return err;
}

static void message(char* buf, size_t n, const char* what, ERR err)
{
    if (buf && n) snprintf(buf, n, "%s failed (jxrlib error %d)", what, (int)err);
}

/* The image's width, height and the layout mr_jxr_decode gives (after strip):
 * kind, bytes per sample, samples per pixel.  0 on success. */
int mr_jxr_info(const unsigned char* data, size_t len, int* width, int* height,
                int* kind, int* bytes, int* samples, char* errbuf, size_t errlen)
{
    PKImageDecode* decoder = NULL;
    PKPixelFormatGUID pf;
    mr_layout lay;
    I32 w = 0, h = 0;
    ERR err = open_decoder(data, len, &decoder);
    if (err) { message(errbuf, errlen, "opening the codestream", err); goto done; }
    err = decoder->GetSize(decoder, &w, &h);
    if (err) { message(errbuf, errlen, "reading the size", err); goto done; }
    err = decoder->GetPixelFormat(decoder, &pf);
    if (err) { message(errbuf, errlen, "reading the pixel format", err); goto done; }
    err = layout_for(&pf, &lay, 0);
    if (err) { message(errbuf, errlen, "an unsupported pixel format", err); goto done; }
    *width = (int)w; *height = (int)h; *kind = lay.kind; *bytes = lay.bytes;
    *samples = lay.strip == 1 || lay.strip == 2 ? lay.samples - 1 : lay.samples;
done:
    if (decoder) PKImageDecode_Release(&decoder);
    return err ? (int)err : 0;
}

/* Decode into out (out_len bytes, exactly height x width x samples x bytes after
 * strip), giving the same layout as mr_jxr_info.  keep_bgr: 8-bit B, G, R(, A) as
 * stored; *bgr says which order came out.  0 on success. */
int mr_jxr_decode(const unsigned char* data, size_t len, unsigned char* out, size_t out_len,
                  int keep_bgr, int* width, int* height, int* kind, int* bytes, int* samples,
                  int* bgr, char* errbuf, size_t errlen)
{
    PKImageDecode* decoder = NULL;
    PKFormatConverter* converter = NULL;
    PKPixelFormatGUID pf;
    mr_layout lay;
    PKRect rect;
    I32 w = 0, h = 0;
    unsigned char* buf = NULL;
    size_t pixel, kept, row, need, i;
    ERR err = open_decoder(data, len, &decoder);
    if (err) { message(errbuf, errlen, "opening the codestream", err); goto done; }
    err = decoder->GetSize(decoder, &w, &h);
    if (err) { message(errbuf, errlen, "reading the size", err); goto done; }
    err = decoder->GetPixelFormat(decoder, &pf);
    if (err) { message(errbuf, errlen, "reading the pixel format", err); goto done; }
    err = layout_for(&pf, &lay, keep_bgr);
    if (err) { message(errbuf, errlen, "an unsupported pixel format", err); goto done; }
    *bgr = lay.bgr;
    decoder->WMP.wmiSCP.uAlphaMode = (U8)lay.alpha;
    err = PKCodecFactory_CreateFormatConverter(&converter);
    if (err) { message(errbuf, errlen, "creating the format converter", err); goto done; }
    err = PKFormatConverter_Initialize(converter, decoder, NULL, pf);
    if (err) { message(errbuf, errlen, "initialising the format converter", err); goto done; }

    pixel = (size_t)lay.samples * (size_t)lay.bytes;
    kept = (lay.strip == 1 || lay.strip == 2) ? pixel - (size_t)lay.bytes : pixel;
    need = (size_t)w * (size_t)h * kept;
    *width = (int)w; *height = (int)h; *kind = lay.kind; *bytes = lay.bytes;
    *samples = (int)(kept / (size_t)lay.bytes);
    if (need != out_len) {
        if (errbuf && errlen)
            snprintf(errbuf, errlen, "decodes to %zu bytes (%d x %d x %d x %d), expected %zu",
                     need, (int)h, (int)w, *samples, lay.bytes, out_len);
        err = WMP_errBufferOverflow;
        goto done;
    }
    rect.X = 0; rect.Y = 0; rect.Width = w; rect.Height = h;
    if (lay.strip == 0) {
        err = PKFormatConverter_Copy(converter, &rect, out, (U32)((size_t)w * pixel));
        if (err) { message(errbuf, errlen, "decoding", err); goto done; }
    } else {
        row = (size_t)w * pixel;
        buf = (unsigned char*)malloc(row * (size_t)h);
        if (!buf) { err = WMP_errOutOfMemory; message(errbuf, errlen, "allocating", err); goto done; }
        err = PKFormatConverter_Copy(converter, &rect, buf, (U32)row);
        if (err) { message(errbuf, errlen, "decoding", err); goto done; }
        for (i = 0; i < (size_t)w * (size_t)h; i++) {
            const unsigned char* s = buf + i * pixel;
            unsigned char* d = out + i * kept;
            size_t b = (size_t)lay.bytes;
            if (lay.strip == 1) {
                memcpy(d, s, kept);
            } else if (lay.strip == 2) {            /* B G R x -> R G B */
                memcpy(d, s + 2 * b, b); memcpy(d + b, s + b, b); memcpy(d + 2 * b, s, b);
            } else {                                /* B G R A -> R G B A */
                memcpy(d, s + 2 * b, b); memcpy(d + b, s + b, b); memcpy(d + 2 * b, s, b);
                memcpy(d + 3 * b, s + 3 * b, b);
            }
        }
    }
done:
    free(buf);
    if (converter) PKFormatConverter_Release(&converter);
    if (decoder) PKImageDecode_Release(&decoder);
    return err ? (int)err : 0;
}
