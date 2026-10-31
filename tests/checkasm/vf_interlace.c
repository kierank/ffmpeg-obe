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

#include "config.h"
#include "libavutil/common.h"
#include "libavutil/cpu.h"
#include "libavutil/mem.h"
#if ARCH_X86
#include "libavutil/x86/cpu.h"
#endif

#include "libavfilter/interlace.h"

#include "checkasm.h"

#define STRIDE  2048            /* source line stride, a multiple of 64 */
#define GUARD   64
#define DST_SIZE (GUARD + STRIDE + GUARD)

static const int widths[] = { 255, 256, 257, 719, 720, 1023, 1024, 1920 };

/* The C versions in libavfilter/vf_interlace.c */
static void lowpass_line_c(uint8_t *dstp, ptrdiff_t linesize,
                           const uint8_t *srcp,
                           ptrdiff_t mref, ptrdiff_t pref)
{
    const uint8_t *srcp_above = srcp + mref;
    const uint8_t *srcp_below = srcp + pref;
    int i;
    for (i = 0; i < linesize; i++)
        dstp[i] = (1 + srcp[i] + srcp[i] + srcp_above[i] + srcp_below[i]) >> 2;
}

static void lowpass_line_complex_c(uint8_t *dstp, ptrdiff_t linesize,
                                   const uint8_t *srcp,
                                   ptrdiff_t mref, ptrdiff_t pref)
{
    const uint8_t *srcp_above = srcp + mref;
    const uint8_t *srcp_below = srcp + pref;
    const uint8_t *srcp_above2 = srcp + mref * 2;
    const uint8_t *srcp_below2 = srcp + pref * 2;
    int i;
    for (i = 0; i < linesize; i++)
        dstp[i] = av_clip_uint8((4 + (srcp[i] << 2)
                  + ((srcp[i] + srcp_above[i] + srcp_below[i]) << 1)
                  - srcp_above2[i] - srcp_below2[i]) >> 3);
}

/* What the version selected at the current cpu level may assume and do:
 * round  - it writes FFALIGN(w, round) bytes (1: exactly w)
 * align  - it needs 16-byte aligned pointers and line offsets */
static void lowpass_props(int lowpass, int *round, int *align)
{
    *round = 1;
    *align = 0;
#if ARCH_X86
    {
        int cpu_flags = av_get_cpu_flags();

        if (lowpass == VLPF_LIN) {
            if (ARCH_X86_64 && EXTERNAL_AVX2_FAST(cpu_flags)) {
                *round = 1;     /* AVX2, AVX-512: exact width, unaligned */
            } else if (EXTERNAL_SSE2(cpu_flags)) {
                *round = 32;    /* SSE2, AVX: two xmm per iteration */
                *align = 1;
            }
        } else {
            if (EXTERNAL_AVX2_FAST(cpu_flags))
                *round = 32;
            else if (EXTERNAL_SSE2(cpu_flags))
                *round = 16;
        }
    }
#endif
}

static void check_lowpass_line(int lowpass)
{
    InterlaceContext s = { .lowpass = lowpass };
    /* rows -2..2 of the source, the current line at row 2 */
    uint8_t *src  = av_malloc(5 * STRIDE + GUARD);
    uint8_t *dst0 = av_malloc(DST_SIZE);
    uint8_t *dst1 = av_malloc(DST_SIZE);
    uint8_t *init = av_malloc(DST_SIZE);
    static const ptrdiff_t refs[][2] = {
        { -STRIDE, STRIDE }, { STRIDE, -STRIDE }, { 0, STRIDE }, { -STRIDE, 0 },
        { STRIDE, 0 }, { 0, -STRIDE }, { 0, 0 },
    };
    void (*const ref)(uint8_t *dstp, ptrdiff_t linesize, const uint8_t *srcp,
                      ptrdiff_t mref, ptrdiff_t pref) =
        lowpass == VLPF_LIN ? lowpass_line_c : lowpass_line_complex_c;
    int i, j, w, r;

    declare_func(void, uint8_t *dstp, ptrdiff_t linesize, const uint8_t *srcp,
                 ptrdiff_t mref, ptrdiff_t pref);

    if (!src || !dst0 || !dst1 || !init)
        goto end;

    s.lowpass_line = ref;
#if ARCH_X86
    ff_interlace_init_x86(&s);
#endif

    if (check_func(s.lowpass_line, "lowpass_line%s",
                   lowpass == VLPF_CMP ? "_complex" : "")) {
        int round, align;

        lowpass_props(lowpass, &round, &align);

        for (i = 0; i < 160 + FF_ARRAY_ELEMS(widths); i++) {
            w = i < 160 ? i + 1 : widths[i - 160];
            for (r = 0; r < FF_ARRAY_ELEMS(refs); r++) {
                const int soff = align ? 0 : rnd() % 64;
                const int doff = align ? 0 : rnd() % 64;
                const int wr   = FFALIGN(w, round);
                const uint8_t *srcp = src + 2 * STRIDE + soff;
                const int wend = GUARD + doff + wr;
                uint8_t *d0 = dst0 + GUARD + doff;
                uint8_t *d1 = dst1 + GUARD + doff;

                for (j = 0; j < 5 * STRIDE + GUARD; j++)
                    src[j] = rnd();
                for (j = 0; j < DST_SIZE; j++)
                    init[j] = dst0[j] = dst1[j] = rnd();

                /* against C: the version tested before may need aligned
                 * pointers that this one doesn't */
                ref(d0, w, srcp, refs[r][0], refs[r][1]);
                call_new(d1, w, srcp, refs[r][0], refs[r][1]);
                /* the line, then nothing written outside
                 * [0, FFALIGN(w, round)) */
                if (memcmp(d0, d1, w) ||
                    memcmp(init, dst1, GUARD + doff) ||
                    memcmp(init + wend, dst1 + wend, DST_SIZE - wend)) {
                    fail();
                    fprintf(stderr, "w %d mref %td pref %td src+%d dst+%d\n",
                            w, refs[r][0], refs[r][1], soff, doff);
                    goto bench;
                }
            }
        }
bench:
        bench_new(dst1 + GUARD, 1920, src + 2 * STRIDE, -STRIDE, STRIDE);
    }

end:
    av_free(src);
    av_free(dst0);
    av_free(dst1);
    av_free(init);
}

void checkasm_check_vf_interlace(void)
{
    check_lowpass_line(VLPF_LIN);
    report("lowpass_line");

    check_lowpass_line(VLPF_CMP);
    report("lowpass_line_complex");
}
