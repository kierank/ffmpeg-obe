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

#include "libavfilter/drawutils.h"

#include "checkasm.h"

/* The 8-bit blending kernels of FFDrawContext, against C versions of their
 * definitions in drawutils.h, which is what blend_line() and blend_pixel()
 * in drawutils.c compute for each byte. */

#define MAX_PIX   400
#define MAX_H     3
#define DST_LS    1024
#define GUARD     64
#define DST_SIZE  (GUARD + MAX_H * DST_LS + GUARD)
#define MASK_LS   (2 * MAX_PIX + 64)
#define MASK_SIZE ((2 * MAX_H + 1) * MASK_LS)

static void blend_rect8_c(uint8_t *dst, ptrdiff_t dst_linesize, int w, int h,
                          const FFBlendRect8Params *p)
{
    int x, y;

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++)
            dst[x] = (dst[x] * p->tau + ((x & 1) ? p->s_odd : p->s_even)) >> 24;
        dst += dst_linesize;
    }
}

static av_always_inline void blend_mask8_c(uint8_t *dst, ptrdiff_t dst_linesize,
                                           const uint8_t *mask, int w, int h,
                                           const FFBlendMask8Params *p,
                                           int cols, int step)
{
    int x, y, c, b;

    for (y = 0; y < h; y++) {
        const uint8_t *m = mask;

        for (x = 0; x < w; x += step) {
            unsigned t = 0, a;

            for (c = 0; c < cols; c++)
                t += m[c] + m[p->row2 + c];
            a = (t >> p->shift) * p->alpha;
            for (b = 0; b < step; b++)
                dst[x + b] = ((0x1010101 - a) * dst[x + b] +
                              a * p->src[(x + b) & 1]) >> 24;
            m += cols;
        }
        dst  += dst_linesize;
        mask += p->adv;
    }
}

#define BLEND_MASK8_C(H, S)                                                   \
static void blend_mask8_h ## H ## s ## S ## _c(uint8_t *dst,                  \
                                               ptrdiff_t dst_linesize,        \
                                               const uint8_t *mask, int w,    \
                                               int h,                         \
                                               const FFBlendMask8Params *p)   \
{                                                                             \
    blend_mask8_c(dst, dst_linesize, mask, w, h, p, H, S);                    \
}

BLEND_MASK8_C(1, 1)
BLEND_MASK8_C(2, 1)
BLEND_MASK8_C(1, 2)
BLEND_MASK8_C(2, 2)

/* blend_rect8_params() in drawutils.c */
static void rect8_params(FFBlendRect8Params *bp, unsigned alpha,
                         unsigned src_even, unsigned src_odd)
{
    unsigned tau = 0x1010101 - alpha;
    unsigned se  = src_even * alpha;
    unsigned so  = src_odd  * alpha;

    bp->t0     = (tau & 0xffff) * 0x10001;
    bp->t1     = (tau >> 16) * 0x10001;
    bp->s1     = (se >> 16) | (so >> 16) << 16;
    bp->ns0    = (~se & 0xffff) | (~so & 0xffff) << 16;
    bp->tau    = tau;
    bp->s_even = se;
    bp->s_odd  = so;
}

/* An alpha byte as drawtext gives it: mostly random, sometimes 0 or 255 */
static unsigned rnd_alpha(void)
{
    switch (rnd() % 8) {
    case 0:  return 0;
    case 1:  return 255;
    default: return rnd() & 0xff;
    }
}

static void init_draw(FFDrawContext *draw)
{
    int flags = av_get_cpu_flags();

    /* The AVX2 kernels need BMI2 as well, which checkasm has no level for:
     * give the AVX2 levels the host's BMI1/BMI2. */
    if (ARCH_X86 && (flags & AV_CPU_FLAG_AVX2)) {
        int host;

        av_force_cpu_flags(-1);
        host = av_get_cpu_flags();
        av_force_cpu_flags(flags | (host & (AV_CPU_FLAG_BMI1 | AV_CPU_FLAG_BMI2)));
    }
    if (ff_draw_init(draw, AV_PIX_FMT_YUV420P, 0) < 0)
        memset(draw, 0, sizeof(*draw));
    av_force_cpu_flags(flags);
}

static void randomize(uint8_t *buf, int size)
{
    int i;

    for (i = 0; i < size; i++)
        buf[i] = rnd();
}

/* The rectangle of new (h rows of w bytes at GUARD + off) against the C
 * output, the rest against the initial contents: nothing outside the
 * rectangle may be written. */
static int cmp_rect(const uint8_t *ref, const uint8_t *new, const uint8_t *init,
                    int off, int w, int h)
{
    int pos = 0, y;

    for (y = 0; y < h; y++) {
        const int start = GUARD + off + y * DST_LS;

        if (memcmp(init + pos, new + pos, start - pos) ||
            memcmp(ref + start, new + start, w))
            return 1;
        pos = start + w;
    }
    return memcmp(init + pos, new + pos, DST_SIZE - pos) != 0;
}

static void check_blend_rect8(const FFDrawContext *draw)
{
    uint8_t *dst0 = av_malloc(DST_SIZE);
    uint8_t *dst1 = av_malloc(DST_SIZE);
    uint8_t *init = av_malloc(DST_SIZE);
    int w, step;

    declare_func(void, uint8_t *dst, ptrdiff_t dst_linesize, int w, int h,
                 const FFBlendRect8Params *p);

    if (!dst0 || !dst1 || !init)
        goto end;

    if (check_func(draw->blend_rect8 ? draw->blend_rect8 : blend_rect8_c,
                   "blend_rect8")) {
        FFBlendRect8Params bp;

        for (step = 1; step <= 2; step++) {
            for (w = step; w <= MAX_PIX * step; w += step) {
                const int h   = 1 + rnd() % MAX_H;
                const int off = rnd() % GUARD;
                /* full alpha, half for a subsampled edge row or column,
                 * quarter for a subsampled corner */
                const unsigned alpha = (0x10203 * rnd_alpha() + 2) >> (rnd() % 3);
                const unsigned se = rnd() & 0xff;
                const unsigned so = step == 2 ? rnd() & 0xff : se;

                rect8_params(&bp, alpha, se, so);
                randomize(init, DST_SIZE);
                memcpy(dst0, init, DST_SIZE);
                memcpy(dst1, init, DST_SIZE);

                blend_rect8_c(dst0 + GUARD + off, DST_LS, w, h, &bp);
                call_new(dst1 + GUARD + off, DST_LS, w, h, &bp);
                if (cmp_rect(dst0, dst1, init, off, w, h)) {
                    fail();
                    fprintf(stderr, "w %d h %d step %d dst+%d alpha 0x%x\n",
                            w, h, step, off, alpha);
                    goto bench;
                }
            }
        }
bench:
        rect8_params(&bp, 0x10203 * 128 + 2, 16, 128);
        bench_new(dst1 + GUARD, DST_LS, 384, MAX_H, &bp);
    }

end:
    av_free(dst0);
    av_free(dst1);
    av_free(init);
}

static void check_blend_mask8(const FFDrawContext *draw, int hsub, int step)
{
    static void (* const ref[2][2])(uint8_t *dst, ptrdiff_t dst_linesize,
                                    const uint8_t *mask, int w, int h,
                                    const FFBlendMask8Params *p) = {
        { blend_mask8_h1s1_c, blend_mask8_h1s2_c },
        { blend_mask8_h2s1_c, blend_mask8_h2s2_c },
    };
    const int cols = 1 + hsub;
    uint8_t *dst0 = av_malloc(DST_SIZE);
    uint8_t *dst1 = av_malloc(DST_SIZE);
    uint8_t *init = av_malloc(DST_SIZE);
    uint8_t *mask = av_malloc(MASK_SIZE);
    int npix;

    declare_func(void, uint8_t *dst, ptrdiff_t dst_linesize,
                 const uint8_t *mask, int w, int h,
                 const FFBlendMask8Params *p);

    if (!dst0 || !dst1 || !init || !mask)
        goto end;

    if (check_func(draw->blend_mask8[hsub][step - 1] ?
                   draw->blend_mask8[hsub][step - 1] : ref[hsub][step - 1],
                   "blend_mask8_h%ds%d", cols, step)) {
        FFBlendMask8Params bp = { 0 };

        for (npix = 1; npix <= MAX_PIX; npix++) {
            const int w    = npix * step;
            const int h    = 1 + rnd() % MAX_H;
            const int off  = rnd() % GUARD;
            const int moff = rnd() % 64;
            const int vsub = rnd() & 1;
            /* one mask row (read twice) or two */
            const int two  = rnd() & 1;

            bp.alpha  = (0x10307 * rnd_alpha() + 3) >> 8;
            bp.src[0] = rnd() & 0xff;
            bp.src[1] = step == 2 ? rnd() & 0xff : bp.src[0];
            /* blend_mask8_band() shifts the sum of 2 * cols mask values by
             * hsub + vsub + (one row), at least log2(2 * cols); the
             * 1-column kernels also take the partial pixels of subsampled
             * planes, with up to 2 more */
            bp.shift  = cols + (rnd() % (hsub ? 2 : 3));
            bp.row2   = two ? MASK_LS : 0;
            bp.adv    = (ptrdiff_t)MASK_LS << vsub;

            randomize(mask, MASK_SIZE);
            randomize(init, DST_SIZE);
            memcpy(dst0, init, DST_SIZE);
            memcpy(dst1, init, DST_SIZE);

            ref[hsub][step - 1](dst0 + GUARD + off, DST_LS, mask + moff,
                                w, h, &bp);
            call_new(dst1 + GUARD + off, DST_LS, mask + moff, w, h, &bp);
            if (cmp_rect(dst0, dst1, init, off, w, h)) {
                fail();
                fprintf(stderr, "w %d h %d dst+%d alpha 0x%x shift %u row2 %td adv %td\n",
                        w, h, off, bp.alpha, bp.shift, bp.row2, bp.adv);
                goto bench;
            }
        }
bench:
        bp.alpha  = (0x10307 * 255 + 3) >> 8;
        bp.src[0] = 16;
        bp.src[1] = step == 2 ? 128 : 16;
        bp.shift  = cols;
        bp.row2   = MASK_LS;
        bp.adv    = 2 * MASK_LS;
        bench_new(dst1 + GUARD, DST_LS, mask, 384, MAX_H, &bp);
    }

end:
    av_free(dst0);
    av_free(dst1);
    av_free(init);
    av_free(mask);
}

void checkasm_check_drawutils(void)
{
    FFDrawContext draw;
    int hsub, step;

    init_draw(&draw);

    check_blend_rect8(&draw);
    report("blend_rect8");

    for (hsub = 0; hsub < 2; hsub++)
        for (step = 1; step <= 2; step++)
            check_blend_mask8(&draw, hsub, step);
    report("blend_mask8");
}
