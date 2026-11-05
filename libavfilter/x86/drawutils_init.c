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

#include "libavutil/attributes.h"
#include "libavutil/cpu.h"
#include "libavutil/x86/cpu.h"
#include "libavfilter/drawutils.h"

#define BLEND_PROTOS(isa)                                                     \
void ff_blend_rect8_ ## isa(uint8_t *dst, ptrdiff_t dst_linesize, int w,     \
                            int h, const FFBlendRect8Params *p);              \
void ff_blend_mask8_h1s1_ ## isa(uint8_t *dst, ptrdiff_t dst_linesize,       \
                                 const uint8_t *mask, int w, int h,           \
                                 const FFBlendMask8Params *p);                \
void ff_blend_mask8_h2s1_ ## isa(uint8_t *dst, ptrdiff_t dst_linesize,       \
                                 const uint8_t *mask, int w, int h,           \
                                 const FFBlendMask8Params *p);                \
void ff_blend_mask8_h1s2_ ## isa(uint8_t *dst, ptrdiff_t dst_linesize,       \
                                 const uint8_t *mask, int w, int h,           \
                                 const FFBlendMask8Params *p);                \
void ff_blend_mask8_h2s2_ ## isa(uint8_t *dst, ptrdiff_t dst_linesize,       \
                                 const uint8_t *mask, int w, int h,           \
                                 const FFBlendMask8Params *p);

BLEND_PROTOS(avx2)
BLEND_PROTOS(avx512)

#define BLEND_SET(isa)                                  \
    do {                                                \
        draw->blend_rect8       = ff_blend_rect8_ ## isa;      \
        draw->blend_mask8[0][0] = ff_blend_mask8_h1s1_ ## isa; \
        draw->blend_mask8[1][0] = ff_blend_mask8_h2s1_ ## isa; \
        draw->blend_mask8[0][1] = ff_blend_mask8_h1s2_ ## isa; \
        draw->blend_mask8[1][1] = ff_blend_mask8_h2s2_ ## isa; \
    } while (0)

av_cold void ff_draw_init_x86(FFDrawContext *draw)
{
#if ARCH_X86_64
    int cpu_flags = av_get_cpu_flags();

    /* the AVX2 tails use shrx */
    if (EXTERNAL_AVX2_FAST(cpu_flags) && (cpu_flags & AV_CPU_FLAG_BMI2))
        BLEND_SET(avx2);
    if (EXTERNAL_AVX512(cpu_flags))
        BLEND_SET(avx512);
#endif
}
