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

static void check_yuv2yuvX(int accurate)
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

    c->dstBpc = 8;
    c->dstFormat = AV_PIX_FMT_YUV420P;
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
                if (check_func(c->yuv2planeX, "yuv2yuvX_8_%d_%d_%d_%s", filter_sizes[fsi], offset, dstW, accurate_str)) {
                    // use vFilterData for the mmx function
                    const int16_t *filter = c->use_mmx_vfilter ? (const int16_t*)vFilterData : &filter_coeff[0];
                    memset(dst0, 0, LARGEST_INPUT_SIZE * sizeof(dst0[0]));
                    memset(dst1, 0, LARGEST_INPUT_SIZE * sizeof(dst1[0]));

                    // We can't use call_ref here, because we don't know if use_mmx_vfilter was set for that
                    // function or not, so we can't pass it the parameters correctly.
                    yuv2planeX_8_ref(&filter_coeff[0], filter_sizes[fsi], src, (uint8_t*)dst0, dstW - offset, dither, offset);
                    call_new(filter, filter_sizes[fsi], src, (uint8_t*)dst1, dstW - offset, dither, offset);

                    if (cmp_off_by_n_8((uint8_t*)dst0, (uint8_t*)dst1, dstW - offset, accurate ? 0 : 2)) {
                        fail();
                        printf("failed: yuv2yuvX_8_%d_%d_%d_%s\n", filter_sizes[fsi], offset, dstW, accurate_str);
                        show_differences_8((uint8_t*)dst0, (uint8_t*)dst1, dstW - offset);
                    }
                    // the scalers work in units of 8 pixels and may run into the
                    // last one, but no further
                    if (cmp_off_by_n_8((uint8_t*)dst0 + FFALIGN(dstW - offset, 8), (uint8_t*)dst1 + FFALIGN(dstW - offset, 8),
                                       LARGEST_INPUT_SIZE * sizeof(dst0[0]) - FFALIGN(dstW - offset, 8), 0)) {
                        fail();
                        printf("failed: yuv2yuvX_8_%d_%d_%d_%s wrote past dstW\n", filter_sizes[fsi], offset, dstW, accurate_str);
                    }

                    if (dstW == LARGEST_INPUT_SIZE)
                        bench_new(filter, filter_sizes[fsi], src, (uint8_t*)dst1, dstW - offset, dither, offset);
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

#define SRC_PIXELS 512
#define MAX_FILTER_WIDTH 40
// rounded up: randomize_buffers() writes in units of four bytes
#define SRC_SAMPLES (SRC_PIXELS + 4 + MAX_FILTER_WIDTH)

static void randomize_src(uint16_t *src, int n, int bpc, int bits)
{
    int i;
    if (bpc == 8) {
        randomize_buffers((uint8_t*)src, n * sizeof(*src));
        return;
    }
    // the filter coefficients below are calibrated for a full range input of
    // this many bits: anything wider makes the 15 bit output leave int16_t,
    // where the C version wraps and the asm saturates
    for (i = 0; i < n; i++)
        src[i] = rnd() & ((1 << bits) - 1);
}

static void check_hscale(void)
{
    // 16 and 32 are whole 16 tap chunks, 24 and 40 leave an odd 8 tap one, and
    // 20 and 28 are the same two cases with the X4 split tail on the end
    static const int filter_sizes[] = { 4, 8, 12, 16, 20, 24, 28, 32, 40 };
    // 13 and 33 leave a pixel over for the tail of the wider scalers, 12 a
    // whole unit of 4, the rest nothing at all
    static const int dst_widths[] = { 8, 12, 13, 16, 33, 128, 144, 256, 512 };
    static const struct {
        int bpc;
        int bits;
        enum AVPixelFormat fmt;
    } src_types[] = {
        {  8,  8, AV_PIX_FMT_GRAY8       },
        {  9,  9, AV_PIX_FMT_YUV420P9LE  },
        { 10, 10, AV_PIX_FMT_YUV420P10LE },
        { 12, 12, AV_PIX_FMT_YUV420P12LE },
        { 14, 14, AV_PIX_FMT_YUV420P14LE },
        { 16, 16, AV_PIX_FMT_YUV420P16LE },
        // packed RGB comes in as 16 bit but scales with the 14 bit shift
        { 16, 14, AV_PIX_FMT_RGB24       },
    };
    static const int dst_bpcs[] = { 14, 16 };
    // the step between the filter positions of neighbouring output pixels: 1 is
    // a mild scale, which is what the widest kernels are written for, and 5
    // spreads the taps of one iteration of the avx512 8 bit one over more than
    // the 64 byte source window it loads, so its fallback is covered as well
    static const int pos_steps[] = { 1, 5 };

    struct SwsContext *c;
    int sti, fsi, dwi, bi, psi, i, j;

    // filter and filterPos carry the three entries of padding initFilter()
    // leaves for the scalers that read past dstW
    LOCAL_ALIGNED_32(uint16_t, src, [SRC_SAMPLES]);
    LOCAL_ALIGNED_32(int16_t, filter, [(SRC_PIXELS + 3) * MAX_FILTER_WIDTH]);
    LOCAL_ALIGNED_32(int32_t, filterPos, [SRC_PIXELS + 3]);
    LOCAL_ALIGNED_32(uint32_t, dst0, [SRC_PIXELS + 8]);
    LOCAL_ALIGNED_32(uint32_t, dst1, [SRC_PIXELS + 8]);

    // dst is int16_t or int32_t depending on dstBpc
    declare_func_emms(AV_CPU_FLAG_MMX, void, struct SwsContext *c, int16_t *dst,
                      int dstW, const uint8_t *src, const int16_t *filter,
                      const int32_t *filterPos, int filterSize);

    c = sws_alloc_context();
    if (sws_init_context(c, NULL, NULL) < 0)
        fail();

    for (sti = 0; sti < FF_ARRAY_ELEMS(src_types); sti++) {
        randomize_src(src, SRC_SAMPLES,
                      src_types[sti].bpc, src_types[sti].bits);

        for (fsi = 0; fsi < FF_ARRAY_ELEMS(filter_sizes); fsi++) {
            int width = filter_sizes[fsi];

            // As the yuv2yuvX filters: all coefficients negative except one
            // random positive one, so that both negative output and the
            // positive clip at (1 << 15) - 1 are reached.  The row has to sum
            // to exactly 1 << 14, since the 16 bit input functions fold
            // 0x8000 * sum(coeffs) back in as a constant.
            for (i = 0; i < SRC_PIXELS + 3; i++) {
                int k = rnd() % width;
                int sum = 0;
                for (j = 0; j < width; j++)
                    filter[i * width + j] = -((1 << 14) / (width - 1));
                filter[i * width + k] = (1 << 15) - 1;
                for (j = 0; j < width; j++)
                    sum += filter[i * width + j];
                filter[i * width + (k ? k - 1 : k + 1)] += (1 << 14) - sum;
            }

            for (bi = 0; bi < FF_ARRAY_ELEMS(dst_bpcs); bi++) {
                int dst_bpc  = dst_bpcs[bi];
                int dst_size = dst_bpc <= 14 ? sizeof(int16_t) : sizeof(int32_t);

                c->srcFormat        = src_types[sti].fmt;
                c->srcBpc           = src_types[sti].bpc;
                c->dstBpc           = dst_bpc;
                c->hLumFilterSize   = c->hChrFilterSize = width;
                ff_getSwsFunc(c);

                for (dwi = 0; dwi < FF_ARRAY_ELEMS(dst_widths); dwi++) {
                for (psi = 0; psi < FF_ARRAY_ELEMS(pos_steps); psi++) {
                    int dstW = dst_widths[dwi];
                    int step = pos_steps[psi];
                    // the scalers work in units of 4 pixels and may run into
                    // the last one, but no further
                    int written = FFALIGN(dstW, 4);

                    // the taps of the last pixel, and what the widest kernels
                    // read past them, have to stay inside src
                    if (step * (dstW - 1) + MAX_FILTER_WIDTH + 4 > SRC_SAMPLES)
                        continue;
                    for (i = 0; i < SRC_PIXELS + 3; i++)
                        filterPos[i] = i * step;

                    // the format has to be part of the name: two of the source
                    // types are 16 bpc but dispatch to different functions, and
                    // checkasm takes the last registered version as the
                    // reference, so a shared name compares one against the other
                    if (!check_func(c->hcScale, "hscale_%s_%d_to_%d__fs_%d_dstW_%d_step_%d",
                                    av_get_pix_fmt_name(src_types[sti].fmt),
                                    src_types[sti].bpc, dst_bpc <= 14 ? 15 : 19,
                                    width, dstW, step))
                        continue;

                    memset(dst0, 0, (SRC_PIXELS + 8) * sizeof(dst0[0]));
                    memset(dst1, 0, (SRC_PIXELS + 8) * sizeof(dst1[0]));

                    call_ref(c, (int16_t*)dst0, dstW, (const uint8_t*)src,
                             filter, filterPos, width);
                    call_new(c, (int16_t*)dst1, dstW, (const uint8_t*)src,
                             filter, filterPos, width);

                    if (memcmp(dst0, dst1, dstW * dst_size)) {
                        fail();
                        printf("failed: hscale_%s_%d_to_%d__fs_%d_dstW_%d_step_%d\n",
                               av_get_pix_fmt_name(src_types[sti].fmt),
                               src_types[sti].bpc, dst_bpc <= 14 ? 15 : 19,
                               width, dstW, step);
                    }
                    if (memcmp((uint8_t*)dst0 + written * dst_size,
                               (uint8_t*)dst1 + written * dst_size,
                               (SRC_PIXELS + 8) * sizeof(dst0[0]) - written * dst_size)) {
                        fail();
                        printf("failed: hscale_%s_%d_to_%d__fs_%d_dstW_%d_step_%d wrote past dstW\n",
                               av_get_pix_fmt_name(src_types[sti].fmt),
                               src_types[sti].bpc, dst_bpc <= 14 ? 15 : 19,
                               width, dstW, step);
                    }

                    if (dstW == SRC_PIXELS)
                        bench_new(c, (int16_t*)dst1, dstW, (const uint8_t*)src,
                                  filter, filterPos, width);
                }
                }
            }
        }
    }
    sws_freeContext(c);
#undef SRC_PIXELS
#undef SRC_SAMPLES
#undef MAX_FILTER_WIDTH
}

void checkasm_check_sw_scale(void)
{
    check_hscale();
    report("hscale");
    check_yuv2yuvX(0);
    check_yuv2yuvX(1);
    report("yuv2yuvX_8");
}
