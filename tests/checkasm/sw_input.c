/*
 * Copyright (C) 2026 Open Broadcast Systems Ltd
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include <string.h>

#include "libavutil/common.h"
#include "libavutil/cpu.h"
#include "libavutil/mem.h"
#include "libavutil/pixdesc.h"

#include "libswscale/swscale.h"
#include "libswscale/swscale_internal.h"

#include "checkasm.h"

/* Input conversion (lumToYV12/chrToYV12) of the semi-planar formats, as
 * selected by sws_init_context() for each source format. */

#define MAX_W      2048
/* bytes the versions that don't mask the tail may write past the row */
#define SLACK      64
#define GUARD      128
#define SRC_SIZE   (MAX_W * 4 + GUARD + 64)
#define DST_SIZE   (GUARD + MAX_W * 2 + GUARD)

/* every width up to NB_SMALL, then these */
#define NB_SMALL   200
static const int large_widths[] = {
    255, 256, 257, 359, 360, 361, 511, 512, 513, 719, 720, 721, 960,
    1023, 1024, 1025, 1279, 1280, 1919, 1920, 1921, 2047, 2048,
};
#define NB_WIDTHS  (NB_SMALL + FF_ARRAY_ELEMS(large_widths))

static const struct {
    enum AVPixelFormat fmt;
    const char *name;
} formats[] = {
    { AV_PIX_FMT_NV12,      "nv12"      },
    { AV_PIX_FMT_NV21,      "nv21"      },
    { AV_PIX_FMT_NV16,      "nv16"      },
    { AV_PIX_FMT_NV20LE,    "nv20le"    },
    { AV_PIX_FMT_NV20BE,    "nv20be"    },
    { AV_PIX_FMT_NV12_10LE, "nv12_10le" },
};

static int width(int i)
{
    return i < NB_SMALL ? i + 1 : large_widths[i - NB_SMALL];
}

static void randomize(uint8_t *buf, int size)
{
    int i;

    for (i = 0; i < size; i++)
        buf[i] = rnd();
}

/* The AVX-512 versions mask the tail and write nothing past the row; the
 * others may write up to SLACK bytes past it. At the AVX-512 level only the
 * AVX-512 versions are new, so only they are tested there. */
static int tail_slack(void)
{
    return (av_get_cpu_flags() & AV_CPU_FLAG_AVX512) ? 0 : SLACK;
}

/* Check a destination buffer: [GUARD, GUARD + len) against the reference
 * output, everything outside [GUARD, GUARD + len + slack) against the
 * initial contents. The reference is the version tested before, which may
 * itself have written past len. */
static int cmp_dst(const uint8_t *ref, const uint8_t *new, const uint8_t *init,
                   int len, int slack)
{
    if (memcmp(init, new, GUARD))
        return 1;
    if (memcmp(ref + GUARD, new + GUARD, len))
        return 1;
    len += slack;
    return memcmp(init + GUARD + len, new + GUARD + len,
                  DST_SIZE - GUARD - len) != 0;
}

static void check_input(enum AVPixelFormat fmt, const char *name)
{
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(fmt);
    const int bps = desc->comp[0].depth > 8 ? 2 : 1;
    struct SwsContext *c;
    int i;
    uint8_t *src   = av_malloc(SRC_SIZE);
    uint8_t *dst0  = av_malloc(DST_SIZE);
    uint8_t *dst1  = av_malloc(DST_SIZE);
    uint8_t *dst2  = av_malloc(DST_SIZE);
    uint8_t *dst3  = av_malloc(DST_SIZE);
    uint8_t *init0 = av_malloc(DST_SIZE);
    uint8_t *init1 = av_malloc(DST_SIZE);

    if (!src || !dst0 || !dst1 || !dst2 || !dst3 || !init0 || !init1)
        goto end;

    /* scaled (not the unscaled shortcut), so the input functions are set */
    c = sws_getContext(64, 16, fmt, 96, 16, AV_PIX_FMT_YUV420P,
                       SWS_BILINEAR, NULL, NULL, NULL);
    if (!c) {
        fail();
        goto end;
    }

    if (c->lumToYV12) {
        declare_func(void, uint8_t *dst, const uint8_t *src,
                     const uint8_t *unused1, const uint8_t *unused2,
                     int w, uint32_t *pal);

        if (check_func(c->lumToYV12, "%s_to_y", name)) {
            const int slack = tail_slack();

            for (i = 0; i < NB_WIDTHS; i++) {
                const int w   = width(i);
                const int off = (i & 1) ? (rnd() % 32) * 2 : 0;
                const uint8_t *s = src + off;

                randomize(src, SRC_SIZE);
                randomize(init0, DST_SIZE);
                memcpy(dst0, init0, DST_SIZE);
                memcpy(dst1, init0, DST_SIZE);

                call_ref(dst0 + GUARD, s, NULL, NULL, w, NULL);
                call_new(dst1 + GUARD, s, NULL, NULL, w, NULL);
                if (cmp_dst(dst0, dst1, init0, w * bps, slack)) {
                    fail();
                    fprintf(stderr, "w %d src+%d\n", w, off);
                    break;
                }
            }
            bench_new(dst1 + GUARD, src, NULL, NULL, 1920, NULL);
        }
    }

    if (c->chrToYV12) {
        declare_func(void, uint8_t *dstU, uint8_t *dstV,
                     const uint8_t *unused0, const uint8_t *src1,
                     const uint8_t *src2, int w, uint32_t *pal);

        if (check_func(c->chrToYV12, "%s_to_uv", name)) {
            const int slack = tail_slack();

            for (i = 0; i < NB_WIDTHS; i++) {
                const int w   = width(i);
                /* the SSE2/AVX NV12 versions have an unaligned source path */
                const int off = (i & 1) ? 1 + rnd() % 63 : 0;
                const uint8_t *s = src + off;

                randomize(src, SRC_SIZE);
                randomize(init0, DST_SIZE);
                randomize(init1, DST_SIZE);
                memcpy(dst0, init0, DST_SIZE);
                memcpy(dst1, init0, DST_SIZE);
                memcpy(dst2, init1, DST_SIZE);
                memcpy(dst3, init1, DST_SIZE);

                call_ref(dst0 + GUARD, dst2 + GUARD, NULL, s, s, w, NULL);
                call_new(dst1 + GUARD, dst3 + GUARD, NULL, s, s, w, NULL);
                if (cmp_dst(dst0, dst1, init0, w * bps, slack) ||
                    cmp_dst(dst2, dst3, init1, w * bps, slack)) {
                    fail();
                    fprintf(stderr, "w %d src+%d\n", w, off);
                    break;
                }
            }
            bench_new(dst1 + GUARD, dst3 + GUARD, NULL, src, src, 1920, NULL);
        }
    }

    sws_freeContext(c);
end:
    av_free(src);
    av_free(dst0);
    av_free(dst1);
    av_free(dst2);
    av_free(dst3);
    av_free(init0);
    av_free(init1);
}

void checkasm_check_sw_input(void)
{
    int i;

    for (i = 0; i < FF_ARRAY_ELEMS(formats); i++)
        check_input(formats[i].fmt, formats[i].name);
    report("input");
}
