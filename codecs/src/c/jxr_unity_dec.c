/* jxrlib's codec core and decoder as ONE translation unit, so the compiler can inline
 * its small per-bit helpers (load4BE, _byteswap_ulong, getBit16, ...) across files:
 * what imagecodecs' jxrlib gets from whole-program optimisation (/GL).  Measured on
 * Zeiss JPEG XR tiles (2026-11-10): separate files decoded ~18 % slower than
 * imagecodecs.  The encoder stays separate (build.rs). */
#if defined(_MSC_VER)
/* built with __ANSI__ on Windows, jxrlib's bit reader calls _byteswap_ulong without
 * a declaration: a CRT call per load.  Declared here, it is the bswap instruction. */
#include <stdlib.h>
#pragma intrinsic(_byteswap_ulong)
#endif
#include "image/sys/adapthuff.c"
#include "image/sys/image.c"
#include "image/sys/strcodec.c"
#include "image/sys/strPredQuant.c"
#include "image/sys/strTransform.c"
#include "image/sys/perfTimerANSI.c"
#include "image/decode/decode.c"
#include "image/decode/postprocess.c"
#include "image/decode/strdec.c"
#include "image/decode/strPredQuantDec.c"
#include "image/decode/JXRTranscode.c"
#include "jxrgluelib/JXRGlue.c"
#include "jxrgluelib/JXRGlueJxr.c"
#include "jxrgluelib/JXRGluePFC.c"
#include "jxrgluelib/JXRMeta.c"
/* segdec.c gives LOAD16 its own meaning (strcodec.c defined it too) */
#undef LOAD16
#include "image/decode/segdec.c"
/* last: it #defines mbX, mbY, tileX ... as fields of pSC to its end */
#include "image/decode/strInvTransform.c"
