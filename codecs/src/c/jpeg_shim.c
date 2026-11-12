/* JPEG -> a caller's buffer, through libjpeg-turbo (vendor/libjpeg-turbo; IJG,
 * BSD-3 and zlib licences), with its SIMD: the decoder tifffile / imagecodecs, Pillow
 * and OpenSlide use, so the pixels are theirs and so is the speed.
 *
 * 8-bit, 12-bit and lossless (2..16-bit) codestreams, baseline, extended, progressive
 * and arithmetic; 1 or 3 components.  12- and 16-bit samples are written as native
 * uint16.  libjpeg's defaults otherwise: islow IDCT, fancy upsampling.
 *
 * colour: 0 = as the file says (JFIF / Adobe markers), 1 = YCbCr (convert to RGB),
 * 2 = RGB (no conversion: what TIFF's Photometric RGB means).
 *
 * Returns 0 on success; MR_JPEG_UNSUPPORTED for what this path does not take
 * (4 components); MR_JPEG_WARNED when libjpeg warned (damaged data it would repair:
 * the caller decides); MR_JPEG_ERROR with a message otherwise.
 */
#include <setjmp.h>
#include <stdio.h>
#include <string.h>

#include "jpeglib.h"

#define MR_JPEG_ERROR 1
#define MR_JPEG_UNSUPPORTED 2
#define MR_JPEG_WARNED 3
#define MR_JPEG_SIZE 4

typedef struct {
    struct jpeg_error_mgr pub;
    jmp_buf jump;
} mr_error;

static void mr_error_exit(j_common_ptr cinfo)
{
    mr_error* err = (mr_error*)cinfo->err;
    longjmp(err->jump, 1);
}

static void mr_output_message(j_common_ptr cinfo)
{
    (void)cinfo;                                /* never print to stderr */
}

int mr_jpeg_decode(const unsigned char* data, size_t len, unsigned char* out,
                   size_t out_len, int colour, int* width, int* height, int* samples,
                   int* bits, char* errbuf, size_t errlen)
{
    struct jpeg_decompress_struct cinfo;
    mr_error jerr;
    volatile int result = 0;
    size_t row_bytes, sample_bytes;
    int precision;

    memset(&cinfo, 0, sizeof(cinfo));
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = mr_error_exit;
    jerr.pub.output_message = mr_output_message;
    if (setjmp(jerr.jump)) {
        if (errbuf && errlen) {
            char text[JMSG_LENGTH_MAX];
            (*cinfo.err->format_message)((j_common_ptr)&cinfo, text);
            snprintf(errbuf, errlen, "%s", text);
        }
        jpeg_destroy_decompress(&cinfo);
        return result ? result : MR_JPEG_ERROR;
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, (unsigned long)len);
    jpeg_read_header(&cinfo, TRUE);

    if (cinfo.num_components != 1 && cinfo.num_components != 3) {
        jpeg_destroy_decompress(&cinfo);
        return MR_JPEG_UNSUPPORTED;
    }
    if (cinfo.num_components == 3) {
        if (colour == 1) cinfo.jpeg_color_space = JCS_YCbCr;
        else if (colour == 2) cinfo.jpeg_color_space = JCS_RGB;
        cinfo.out_color_space = JCS_RGB;
    } else {
        cinfo.out_color_space = JCS_GRAYSCALE;
    }
    precision = cinfo.data_precision;
    jpeg_start_decompress(&cinfo);

    *width = (int)cinfo.output_width;
    *height = (int)cinfo.output_height;
    *samples = cinfo.output_components;
    *bits = precision <= 8 ? 8 : 16;
    sample_bytes = precision <= 8 ? 1 : 2;
    row_bytes = (size_t)cinfo.output_width * (size_t)cinfo.output_components * sample_bytes;
    if (row_bytes * (size_t)cinfo.output_height != out_len) {
        if (errbuf && errlen)
            snprintf(errbuf, errlen, "decodes to %zu bytes, expected %zu",
                     row_bytes * (size_t)cinfo.output_height, out_len);
        jpeg_abort_decompress(&cinfo);
        jpeg_destroy_decompress(&cinfo);
        return MR_JPEG_SIZE;
    }
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char* row = out + (size_t)cinfo.output_scanline * row_bytes;
        if (precision <= 8) {
            JSAMPROW rows[1];
            rows[0] = (JSAMPROW)row;
            jpeg_read_scanlines(&cinfo, rows, 1);
        } else if (precision <= 12) {
            J12SAMPROW rows[1];
            rows[0] = (J12SAMPROW)row;
            jpeg12_read_scanlines(&cinfo, rows, 1);
        } else {
            J16SAMPROW rows[1];
            rows[0] = (J16SAMPROW)row;
            jpeg16_read_scanlines(&cinfo, rows, 1);
        }
    }
    jpeg_finish_decompress(&cinfo);
    if (jerr.pub.num_warnings > 0) result = MR_JPEG_WARNED;
    jpeg_destroy_decompress(&cinfo);
    return result;
}
