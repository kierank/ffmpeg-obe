/*
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

#ifndef AVFILTER_DRAWUTILS_H
#define AVFILTER_DRAWUTILS_H

/**
 * @file
 * misc drawing utilities
 */

#include <stdint.h>
#include "avfilter.h"
#include "libavutil/pixfmt.h"

int ff_fill_rgba_map(uint8_t *rgba_map, enum AVPixelFormat pix_fmt);

int ff_fill_line_with_color(uint8_t *line[4], int pixel_step[4], int w,
                            uint8_t dst_color[4],
                            enum AVPixelFormat pix_fmt, uint8_t rgba_color[4],
                            int *is_packed_rgba, uint8_t rgba_map[4]);

void ff_draw_rectangle(uint8_t *dst[4], int dst_linesize[4],
                       uint8_t *src[4], int pixelstep[4],
                       int hsub, int vsub, int x, int y, int w, int h);

void ff_copy_rectangle(uint8_t *dst[4], int dst_linesize[4],
                       uint8_t *src[4], int src_linesize[4], int pixelstep[4],
                       int hsub, int vsub, int x, int y, int y2, int w, int h);

#define MAX_PLANES 4

/**
 * Constants for blending one colour over a rectangle of 8-bit samples:
 * dst = (dst * tau + s) >> 24, with s taken from s_even or s_odd by the
 * parity of the byte offset within the row (both equal for planar data).
 * t0..ns0 hold the same values split into 16-bit halves, two per dword
 * (even byte in the low half), for the SIMD versions.
 */
typedef struct FFBlendRect8Params {
    uint32_t t0;      ///< (tau & 0xffff), twice
    uint32_t t1;      ///< (tau >> 16), twice
    uint32_t s1;      ///< s_even >> 16 | (s_odd >> 16) << 16
    uint32_t ns0;     ///< ~s_even & 0xffff | (~s_odd & 0xffff) << 16
    uint32_t tau;
    uint32_t s_even;
    uint32_t s_odd;
} FFBlendRect8Params;

/**
 * Constants for blending one colour through a 8-bit grey mask.
 * For each byte of the row the mask is summed over 1 or 2 columns (by the
 * kernel) and over the mask row and the row at +row2, then shifted right by
 * shift; a = sum * alpha and dst = ((0x1010101 - a) * dst + a * src) >> 24.
 */
typedef struct FFBlendMask8Params {
    uint32_t  alpha;
    uint16_t  src[2];  ///< colour for even and odd bytes
    uint32_t  shift;
    uint32_t  reserved;
    ptrdiff_t row2;    ///< offset of the second mask row (0: the same row)
    ptrdiff_t adv;     ///< mask advance per destination row
} FFBlendMask8Params;

typedef struct FFDrawContext {
    const struct AVPixFmtDescriptor *desc;
    enum AVPixelFormat format;
    unsigned nb_planes;
    int pixelstep[MAX_PLANES]; /*< offset between pixels */
    uint8_t comp_mask[MAX_PLANES]; /*< bitmask of used non-alpha components */
    uint8_t hsub[MAX_PLANES];  /*< horizontal subsampling */
    uint8_t vsub[MAX_PLANES];  /*< vertical subsampling */
    uint8_t hsub_max;
    uint8_t vsub_max;
    unsigned flags;
    /* SIMD kernels for 8-bit planes, NULL if unavailable.
     * w is in bytes, h in rows (both > 0). */
    void (*blend_rect8)(uint8_t *dst, ptrdiff_t dst_linesize, int w, int h,
                        const FFBlendRect8Params *p);
    /* [hsub][pixelstep - 1]: hsub 0 reads one mask column per pixel,
     * hsub 1 two; pixelstep 2 is interleaved chroma (one mask value for both
     * bytes of a pixel). */
    void (*blend_mask8[2][2])(uint8_t *dst, ptrdiff_t dst_linesize,
                              const uint8_t *mask, int w, int h,
                              const FFBlendMask8Params *p);
} FFDrawContext;

void ff_draw_init_x86(FFDrawContext *draw);

typedef struct FFDrawColor {
    uint8_t rgba[4];
    union {
        uint32_t u32[4];
        uint16_t u16[8];
        uint8_t  u8[16];
    } comp[MAX_PLANES];
} FFDrawColor;

/**
  * Process alpha pixel component.
  */
#define FF_DRAW_PROCESS_ALPHA 1

/**
 * Init a draw context.
 *
 * Only a limited number of pixel formats are supported, if format is not
 * supported the function will return an error.
 * flags is combination of FF_DRAW_* flags.
 * @return  0 for success, < 0 for error
 */
int ff_draw_init(FFDrawContext *draw, enum AVPixelFormat format, unsigned flags);

/**
 * Prepare a color.
 */
void ff_draw_color(FFDrawContext *draw, FFDrawColor *color, const uint8_t rgba[4]);

/**
 * Copy a rectangle from an image to another.
 *
 * The coordinates must be as even as the subsampling requires.
 */
void ff_copy_rectangle2(FFDrawContext *draw,
                        uint8_t *dst[], int dst_linesize[],
                        uint8_t *src[], int src_linesize[],
                        int dst_x, int dst_y, int src_x, int src_y,
                        int w, int h);

/**
 * Fill a rectangle with an uniform color.
 *
 * The coordinates must be as even as the subsampling requires.
 * The color needs to be inited with ff_draw_color.
 */
void ff_fill_rectangle(FFDrawContext *draw, FFDrawColor *color,
                       uint8_t *dst[], int dst_linesize[],
                       int dst_x, int dst_y, int w, int h);

/**
 * Blend a rectangle with an uniform color.
 */
void ff_blend_rectangle(FFDrawContext *draw, FFDrawColor *color,
                        uint8_t *dst[], int dst_linesize[],
                        int dst_w, int dst_h,
                        int x0, int y0, int w, int h);

/**
 * Blend an alpha mask with an uniform color.
 *
 * @param draw           draw context
 * @param color          color for the overlay;
 * @param dst            destination image
 * @param dst_linesize   line stride of the destination
 * @param dst_w          width of the destination image
 * @param dst_h          height of the destination image
 * @param mask           mask
 * @param mask_linesize  line stride of the mask
 * @param mask_w         width of the mask
 * @param mask_h         height of the mask
 * @param l2depth        log2 of depth of the mask (0 for 1bpp, 3 for 8bpp)
 * @param endianness     bit order of the mask (0: MSB to the left)
 * @param x0             horizontal position of the overlay
 * @param y0             vertical position of the overlay
 */
void ff_blend_mask(FFDrawContext *draw, FFDrawColor *color,
                   uint8_t *dst[], int dst_linesize[], int dst_w, int dst_h,
                   const uint8_t *mask, int mask_linesize, int mask_w, int mask_h,
                   int l2depth, unsigned endianness, int x0, int y0);

/**
 * Round a dimension according to subsampling.
 *
 * @param draw       draw context
 * @param sub_dir    0 for horizontal, 1 for vertical
 * @param round_dir  0 nearest, -1 round down, +1 round up
 * @param value      value to round
 * @return  the rounded value
 */
int ff_draw_round_to_sub(FFDrawContext *draw, int sub_dir, int round_dir,
                         int value);

/**
 * Return the list of pixel formats supported by the draw functions.
 *
 * The flags are the same as ff_draw_init, i.e., none currently.
 */
AVFilterFormats *ff_draw_supported_pixel_formats(unsigned flags);

#endif /* AVFILTER_DRAWUTILS_H */
