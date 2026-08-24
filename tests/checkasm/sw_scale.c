/*
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with FFmpeg; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#include <string.h>

#include "libavutil/common.h"
#include "libavutil/intreadwrite.h"
#include "libavutil/mem.h"

#include "libswscale/swscale.h"
#include "libswscale/swscale_internal.h"

#include "checkasm.h"

#define randomize_buffers(buf, size)      \
    do {                                  \
        int j;                            \
        for (j = 0; j < size; j+=4)       \
            AV_WN32(buf + j, rnd());      \
    } while (0)

static void yuv2planeX_8_ref(const int16_t *filter, int filterSize,
                             const int16_t **src, uint8_t *dest, int dstW,
                             const uint8_t *dither, int offset)
{
    int i, j;
    for (i = 0; i < dstW; i++) {
        int val = dither[(i + offset) & 7] << 12;
        for (j = 0; j < filterSize; j++)
            val += src[j][i] * filter[j];
        dest[i] = av_clip_uint8(val >> 19);
    }
}

static int cmp_off_by_n_8(const uint8_t *ref, const uint8_t *test, size_t n, int accuracy)
{
    size_t i;
    for (i = 0; i < n; i++) {
        if (abs(ref[i] - test[i]) > accuracy)
            return 1;
    }
    return 0;
}

static void show_differences_8(const uint8_t *ref, const uint8_t *test, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        if (ref[i] != test[i]) {
            printf("both:\n");
            for (i = 0; i < n; i++)
                printf("%3d, ", ref[i]);
            printf("\nnew:\n");
            for (i = 0; i < n; i++)
                printf("%3d, ", test[i]);
            printf("\n");
            return;
        }
    }
}

static int cmp_off_by_n_16(const uint16_t *ref, const uint16_t *test, size_t n, int accuracy)
{
    size_t i;
    for (i = 0; i < n; i++) {
        if (abs(ref[i] - test[i]) > accuracy)
            return 1;
    }
    return 0;
}

static void show_differences_16(const uint16_t *ref, const uint16_t *test, size_t n)
{
    size_t i;
    for (i = 0; i < n; i++) {
        if (ref[i] != test[i]) {
            printf("first difference at %d: both %d, new %d\n",
                   (int)i, ref[i], test[i]);
            return;
        }
    }
}

static void check_yuv2yuvX(int accurate, int bpc, enum AVPixelFormat dstFormat)
{
    struct SwsContext *c;
    int fsi, osi, isi, i, j;
    int dstW;
#define LARGEST_FILTER 16
    // ff_yuv2planeX_8_sse2 can't handle odd filter sizes
    const int filter_sizes[] = {2, 4, 8, 16};
    const int FILTER_SIZES = sizeof(filter_sizes)/sizeof(filter_sizes[0]);
#define LARGEST_INPUT_SIZE 512
    // 17 and 129 leave a single pixel over for the tail of the wider scalers,
    // 24 a whole unit of 8, the rest nothing at all.  The width is taken down
    // by the offset, which covers the sizes in between as well.
    static const int input_sizes[] = {8, 17, 24, 128, 129, 144, 256, 512};
    // the only two offsets swscale.c ever passes, one per chroma plane
    static const int offsets[] = {0, 3};
    const char *accurate_str = (accurate) ? "accurate" : "approximate";
    // only the 8 bit output has an mmx vfilter to dodge; the 9 and 10 bit
    // functions are exact and take the plain filter in both rounding modes,
    // so those can be compared against the C version directly
    const int pixel_size = (bpc == 8) ? 1 : 2;

    declare_func_emms(AV_CPU_FLAG_MMX, void, const int16_t *filter,
                      int filterSize, const int16_t **src, uint8_t *dest,
                      int dstW, const uint8_t *dither, int offset);

    const int16_t **src;
    LOCAL_ALIGNED_16(int16_t, src_pixels, [LARGEST_FILTER * LARGEST_INPUT_SIZE]);
    LOCAL_ALIGNED_16(int16_t, filter_coeff, [LARGEST_FILTER]);
    LOCAL_ALIGNED_16(uint16_t, dst0, [LARGEST_INPUT_SIZE]);
    LOCAL_ALIGNED_16(uint16_t, dst1, [LARGEST_INPUT_SIZE]);
    LOCAL_ALIGNED_16(uint8_t, dither, [8]);
    union VFilterData{
        const int16_t *src;
        uint16_t coeff[8];
    } *vFilterData;
    // eight distinct values, so that the offset rotation is covered.  This is
    // only sound because the offsets tested are the 0 and 3 of the API: the
    // asm rotates the dither by 3 for any non-zero offset.
    randomize_buffers(dither, 8);
    randomize_buffers((uint8_t*)src_pixels, LARGEST_FILTER * LARGEST_INPUT_SIZE * sizeof(int16_t));
    c = sws_alloc_context();
    if (accurate)
        c->flags |= SWS_ACCURATE_RND;
    if (sws_init_context(c, NULL, NULL) < 0)
        fail();

    c->dstBpc = bpc;
    c->dstFormat = dstFormat;
    ff_getSwsFunc(c);
    for(isi = 0; isi < FF_ARRAY_ELEMS(input_sizes); ++isi){
        dstW = input_sizes[isi];
        for(osi = 0; osi < FF_ARRAY_ELEMS(offsets); ++osi){
            int offset = offsets[osi];
            for (fsi = 0; fsi < FILTER_SIZES; ++fsi) {
                // Generate filter coefficients for the given filter size,
                // with some properties:
                // - The coefficients add up to the intended sum (4096, 1<<12)
                // - The coefficients contain negative values
                // - The filter intermediates don't overflow for worst case
                //   inputs (all positive coefficients are coupled with
                //   input_max and all negative coefficients with input_min,
                //   or vice versa).
                // Produce a filter with all coefficients set to
                // -((1<<12)/(filter_size-1)) except for one (randomly chosen)
                // which is set to ((1<<13)-1).
                for (i = 0; i < filter_sizes[fsi]; ++i)
                    filter_coeff[i] = -((1 << 12) / (filter_sizes[fsi] - 1));
                filter_coeff[rnd() % filter_sizes[fsi]] = (1 << 13) - 1;

                src = av_malloc(sizeof(int16_t*) * filter_sizes[fsi]);
                vFilterData = av_malloc((filter_sizes[fsi] + 2) * sizeof(union VFilterData));
                memset(vFilterData, 0, (filter_sizes[fsi] + 2) * sizeof(union VFilterData));
                for (i = 0; i < filter_sizes[fsi]; ++i) {
                    src[i] = &src_pixels[i * LARGEST_INPUT_SIZE];
                    // the mmx filter reads from offset onwards
                    vFilterData[i].src = src[i] - offset;
                    for(j = 0; j < 4; ++j)
                        vFilterData[i].coeff[j + 4] = filter_coeff[i];
                }
                if (check_func(c->yuv2planeX, "yuv2yuvX_%d_%d_%d_%d_%s", bpc, filter_sizes[fsi], offset, dstW, accurate_str)) {
                    // use vFilterData for the mmx function
                    const int16_t *filter = c->use_mmx_vfilter ? (const int16_t*)vFilterData : &filter_coeff[0];
                    int px      = dstW - offset;
                    // the scalers work in units of 8 pixels and may run into the
                    // last one, but no further
                    int written = FFALIGN(px, 8);
                    memset(dst0, 0, LARGEST_INPUT_SIZE * sizeof(dst0[0]));
                    memset(dst1, 0, LARGEST_INPUT_SIZE * sizeof(dst1[0]));

                    if (bpc == 8) {
                        // We can't use call_ref here, because we don't know if use_mmx_vfilter was set for that
                        // function or not, so we can't pass it the parameters correctly.
                        yuv2planeX_8_ref(&filter_coeff[0], filter_sizes[fsi], src, (uint8_t*)dst0, px, dither, offset);
                        call_new(filter, filter_sizes[fsi], src, (uint8_t*)dst1, px, dither, offset);

                        if (cmp_off_by_n_8((uint8_t*)dst0, (uint8_t*)dst1, px, accurate ? 0 : 2)) {
                            fail();
                            printf("failed: yuv2yuvX_%d_%d_%d_%d_%s\n", bpc, filter_sizes[fsi], offset, dstW, accurate_str);
                            show_differences_8((uint8_t*)dst0, (uint8_t*)dst1, px);
                        }
                    } else {
                        call_ref(filter, filter_sizes[fsi], src, (uint8_t*)dst0, px, dither, offset);
                        call_new(filter, filter_sizes[fsi], src, (uint8_t*)dst1, px, dither, offset);

                        if (cmp_off_by_n_16(dst0, dst1, px, 0)) {
                            fail();
                            printf("failed: yuv2yuvX_%d_%d_%d_%d_%s\n", bpc, filter_sizes[fsi], offset, dstW, accurate_str);
                            show_differences_16(dst0, dst1, px);
                        }
                    }

                    if (cmp_off_by_n_8((uint8_t*)dst0 + written * pixel_size, (uint8_t*)dst1 + written * pixel_size,
                                       LARGEST_INPUT_SIZE * sizeof(dst0[0]) - written * pixel_size, 0)) {
                        fail();
                        printf("failed: yuv2yuvX_%d_%d_%d_%d_%s wrote past dstW\n", bpc, filter_sizes[fsi], offset, dstW, accurate_str);
                    }

                    if (dstW == LARGEST_INPUT_SIZE)
                        bench_new(filter, filter_sizes[fsi], src, (uint8_t*)dst1, px, dither, offset);
                }
                av_freep(&src);
                av_freep(&vFilterData);
            }
        }
    }
    sws_freeContext(c);
#undef LARGEST_INPUT_SIZE
#undef LARGEST_FILTER
}

void checkasm_check_sw_scale(void)
{
    check_yuv2yuvX(0, 8, AV_PIX_FMT_YUV420P);
    check_yuv2yuvX(1, 8, AV_PIX_FMT_YUV420P);
    report("yuv2yuvX_8");
    check_yuv2yuvX(0, 9, AV_PIX_FMT_YUV420P9LE);
    check_yuv2yuvX(1, 9, AV_PIX_FMT_YUV420P9LE);
    report("yuv2yuvX_9LE");
    check_yuv2yuvX(0, 10, AV_PIX_FMT_YUV420P10LE);
    check_yuv2yuvX(1, 10, AV_PIX_FMT_YUV420P10LE);
    report("yuv2yuvX_10LE");
}
