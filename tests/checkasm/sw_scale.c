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

// The yuv2planeX of use_mmx_vfilter: bit exact with the inline asm
// yuv2yuvX_sse3/mmxext, which round differently from the C version.  The
// dither is rotated by 3 for any non-zero offset, each pixel starts at
// (dither + 8 * (filterSize - 1)) >> 4 and adds pmulhw(src, coeff) in 16 bits.
static void yuv2yuvX_mmx_ref(const int16_t *filter, int filterSize,
                             const int16_t **src, uint8_t *dest, int dstW,
                             const uint8_t *dither, int offset)
{
    int i, j;
    for (i = 0; i < dstW; i++) {
        int16_t val = (int16_t)(dither[(i + (offset ? 3 : 0)) & 7] + 8 * (filterSize - 1)) >> 4;
        for (j = 0; j < filterSize; j++)
            val = (int16_t)(val + ((src[j][i] * filter[j]) >> 16));
        dest[i] = av_clip_uint8(val >> 3);
    }
}

static void check_yuv2yuvX(int accurate, int bpc, enum AVPixelFormat dstFormat)
{
    struct SwsContext *c;
    int fsi, osi, isi, i, j, mis;
    int dstW;
#define LARGEST_FILTER 34
    // ff_yuv2planeX_8_sse2 can't handle odd filter sizes; 18 and 34 are
    // thumbnail ones
    const int filter_sizes[] = {2, 4, 6, 8, 16, 18, 34};
    const int FILTER_SIZES = sizeof(filter_sizes)/sizeof(filter_sizes[0]);
#define LARGEST_INPUT_SIZE 2048
#define DST_SIZE (LARGEST_INPUT_SIZE + 64)
    // 17 and 129 leave a single pixel over for the tail of the wider scalers,
    // 24 a whole unit of 8, 120/240 and 720-1920 are the obed widths.  The
    // width is taken down by the offset, which covers the sizes in between.
    static const int input_sizes[] = {8, 17, 24, 120, 128, 129, 144, 240, 256,
                                      512, 720, 960, 1280, 1920};
    // 0 and 3 are what swscale.c passes for the non-mmx functions; with the
    // mmx filter the V plane passes c->uv_offx2 >> 1, which 37 stands for
    static const int offsets[] = {0, 3, 37};
    const char *accurate_str = (accurate) ? "accurate" : "approximate";
    // only the 8 bit output has an mmx vfilter to dodge; the 9 and 10 bit
    // functions are exact and take the plain filter in both rounding modes,
    // so those can be compared against the C version directly
    const int pixel_size = (bpc == 8) ? 1 : 2;

    declare_func_emms(AV_CPU_FLAG_MMX, void, const int16_t *filter,
                      int filterSize, const int16_t **src, uint8_t *dest,
                      int dstW, const uint8_t *dither, int offset);

    const int16_t **src;
    int16_t  *src_pixels = av_malloc(LARGEST_FILTER * LARGEST_INPUT_SIZE * sizeof(int16_t));
    uint16_t *dst0 = av_malloc(DST_SIZE * sizeof(uint16_t));
    uint16_t *dst1 = av_malloc(DST_SIZE * sizeof(uint16_t));
    LOCAL_ALIGNED_16(int16_t, filter_coeff, [LARGEST_FILTER]);
    LOCAL_ALIGNED_16(uint8_t, dither, [8]);
    union VFilterData{
        const int16_t *src;
        uint16_t coeff[8];
    } *vFilterData;
    if (!src_pixels || !dst0 || !dst1) {
        fail();
        goto end;
    }
    // eight distinct values, so that the offset rotation is covered
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
            // the others rotate the dither by 3 for any non-zero offset as well,
            // but the C version by the offset itself
            if (offset >= dstW || (offset > 3 && !c->use_mmx_vfilter))
                continue;
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
                // the mmx versions take another path for an unaligned dest
                // (the 9 and 10 bit sse2 ones store aligned, as their dest always is)
                for (mis = 0; mis < 1 + (bpc == 8); mis++) {
                if (check_func(c->yuv2planeX, "yuv2yuvX_%d_%d_%d_%d_%s%s", bpc, filter_sizes[fsi], offset, dstW, accurate_str, mis ? "_unaligned" : "")) {
                    // use vFilterData for the mmx function
                    const int16_t *filter = c->use_mmx_vfilter ? (const int16_t*)vFilterData : &filter_coeff[0];
                    int px      = dstW - offset;
                    // the scalers work in units of 16 pixels and may run into the
                    // last one, but no further
                    int written = FFALIGN(px, 16);
                    uint8_t *d0 = (uint8_t*)dst0 + mis * pixel_size;
                    uint8_t *d1 = (uint8_t*)dst1 + mis * pixel_size;
                    memset(dst0, 0, DST_SIZE * sizeof(dst0[0]));
                    memset(dst1, 0, DST_SIZE * sizeof(dst1[0]));

                    if (bpc == 8) {
                        // We can't use call_ref here, because we don't know if use_mmx_vfilter was set for that
                        // function or not, so we can't pass it the parameters correctly.
                        if (c->use_mmx_vfilter)
                            yuv2yuvX_mmx_ref(&filter_coeff[0], filter_sizes[fsi], src, d0, px, dither, offset);
                        else
                            yuv2planeX_8_ref(&filter_coeff[0], filter_sizes[fsi], src, d0, px, dither, offset);
                        call_new(filter, filter_sizes[fsi], src, d1, px, dither, offset);

                        // the mmx rounding is its own, but exact
                        if (cmp_off_by_n_8(d0, d1, px, accurate || c->use_mmx_vfilter ? 0 : 2)) {
                            fail();
                            printf("failed: yuv2yuvX_%d_%d_%d_%d_%s%s\n", bpc, filter_sizes[fsi], offset, dstW, accurate_str, mis ? " unaligned" : "");
                            show_differences_8(d0, d1, px);
                        }
                    } else {
                        call_ref(filter, filter_sizes[fsi], src, d0, px, dither, offset);
                        call_new(filter, filter_sizes[fsi], src, d1, px, dither, offset);

                        if (memcmp(d0, d1, px * 2)) {
                            fail();
                            printf("failed: yuv2yuvX_%d_%d_%d_%d_%s%s\n", bpc, filter_sizes[fsi], offset, dstW, accurate_str, mis ? " unaligned" : "");
                            show_differences_16(dst0, dst1, px + mis);
                        }
                    }

                    if (cmp_off_by_n_8(d0 + written * pixel_size, d1 + written * pixel_size,
                                       DST_SIZE * sizeof(dst0[0]) - (written + mis) * pixel_size, 0)) {
                        fail();
                        printf("failed: yuv2yuvX_%d_%d_%d_%d_%s wrote past dstW\n", bpc, filter_sizes[fsi], offset, dstW, accurate_str);
                    }

                    if ((dstW == 512 || dstW == 240 || dstW == 1920) && !offset && !mis)
                        bench_new(filter, filter_sizes[fsi], src, d1, px, dither, offset);
                }
                }
                av_freep(&src);
                av_freep(&vFilterData);
            }
        }
    }
    sws_freeContext(c);
end:
    av_free(src_pixels);
    av_free(dst0);
    av_free(dst1);
#undef LARGEST_INPUT_SIZE
#undef LARGEST_FILTER
#undef DST_SIZE
}

static void check_yuv2nv12cX(int accurate)
{
    struct SwsContext *c;
#define LARGEST_FILTER 16
    const int filter_sizes[] = {2, 4, 8, 16};
#define LARGEST_INPUT_SIZE 512
    static const int input_sizes[] = {8, 24, 128, 144, 256, 512};
    const char *accurate_str = (accurate) ? "accurate" : "approximate";
    int isi, fsi, i;

    declare_func_emms(AV_CPU_FLAG_MMX, void, enum AVPixelFormat dstFormat,
                      const uint8_t *chrDither, const int16_t *chrFilter,
                      int chrFilterSize, const int16_t **chrUSrc,
                      const int16_t **chrVSrc, uint8_t *dest, int dstW);

    const int16_t *srcU[LARGEST_FILTER], *srcV[LARGEST_FILTER];
    LOCAL_ALIGNED_16(int16_t, srcU_pixels, [LARGEST_FILTER * LARGEST_INPUT_SIZE]);
    LOCAL_ALIGNED_16(int16_t, srcV_pixels, [LARGEST_FILTER * LARGEST_INPUT_SIZE]);
    LOCAL_ALIGNED_16(int16_t, filter_coeff, [LARGEST_FILTER]);
    LOCAL_ALIGNED_16(uint8_t, dst0, [LARGEST_INPUT_SIZE * 2]);
    LOCAL_ALIGNED_16(uint8_t, dst1, [LARGEST_INPUT_SIZE * 2]);
    LOCAL_ALIGNED_16(uint8_t, dither, [LARGEST_INPUT_SIZE]);
    uint8_t d_val = rnd();
    memset(dither, d_val, LARGEST_INPUT_SIZE);
    randomize_buffers((uint8_t*)srcU_pixels, LARGEST_FILTER * LARGEST_INPUT_SIZE * sizeof(int16_t));
    randomize_buffers((uint8_t*)srcV_pixels, LARGEST_FILTER * LARGEST_INPUT_SIZE * sizeof(int16_t));
    for (i = 0; i < LARGEST_FILTER; i++) {
        srcU[i] = &srcU_pixels[i * LARGEST_INPUT_SIZE];
        srcV[i] = &srcV_pixels[i * LARGEST_INPUT_SIZE];
    }

    c = sws_alloc_context();
    if (accurate)
        c->flags |= SWS_ACCURATE_RND;
    if (sws_init_context(c, NULL, NULL) < 0)
        fail();

    c->dstFormat = AV_PIX_FMT_NV12;
    ff_getSwsFunc(c);
    for (isi = 0; isi < FF_ARRAY_ELEMS(input_sizes); isi++){
        const int dstW = input_sizes[isi];
        for (fsi = 0; fsi < FF_ARRAY_ELEMS(filter_sizes); fsi++) {
            const int filter_size = filter_sizes[fsi];
            for (i = 0; i < filter_size; i++)
                filter_coeff[i] = -((1 << 12) / (filter_size - 1));
            filter_coeff[rnd() % filter_size] = (1 << 13) - 1;

            if (check_func(c->yuv2nv12cX, "yuv2nv12cX_%d_%d_%s", filter_size, dstW, accurate_str)){
                memset(dst0, 0, LARGEST_INPUT_SIZE * sizeof(dst0[0]));
                memset(dst1, 0, LARGEST_INPUT_SIZE * sizeof(dst1[0]));

                call_ref(c->dstFormat, dither, &filter_coeff[0], filter_size, srcU, srcV, dst0, dstW);
                call_new(c->dstFormat, dither, &filter_coeff[0], filter_size, srcU, srcV, dst1, dstW);

                if (cmp_off_by_n_8(dst0, dst1, dstW * 2 * sizeof(dst0[0]), accurate ? 0 : 2)) {
                    fail();
                    printf("failed: yuv2nv12wX_%d_%d_%s\n", filter_size, dstW, accurate_str);
                    show_differences_8(dst0, dst1, dstW * 2 * sizeof(dst0[0]));
                }
                if (dstW == LARGEST_INPUT_SIZE)
                    bench_new(c->dstFormat, dither, &filter_coeff[0], filter_size, srcU, srcV, dst1, dstW);

            }
        }
    }
    sws_freeContext(c);
}
#undef LARGEST_FILTER
#undef LARGEST_INPUT_SIZE

typedef void (*yuv2nv_fn)(enum AVPixelFormat dstFormat, const uint8_t *chrDither,
                          const int16_t *chrFilter, int chrFilterSize,
                          const int16_t **chrUSrc, const int16_t **chrVSrc,
                          uint8_t *dest, int dstW);

#define YUV2NV_PROTO(name) \
    void name(enum AVPixelFormat dstFormat, const uint8_t *chrDither, \
              const int16_t *chrFilter, int chrFilterSize, \
              const int16_t **chrUSrc, const int16_t **chrVSrc, \
              uint8_t *dest, int dstW)

// As the hscale kernels: tested by name as well as through the dispatch, so
// that the ones not dispatched on this cpu are covered too.  unit is how far
// the kernel may write past dstW, in pixels: the upstream avx2 one stores 8
// at a time, the others exactly dstW.
typedef struct NvKernel {
    const char *name;
    yuv2nv_fn fn;
    int cpu_flag;
    int bits;
    int swapped;
    int unit;
} NvKernel;

#if ARCH_X86_64 && HAVE_X86ASM
YUV2NV_PROTO(ff_yuv2nv12cX_avx2);
YUV2NV_PROTO(ff_yuv2nv21cX_avx2);
YUV2NV_PROTO(ff_yuv2nv12cX_madd_avx2);
YUV2NV_PROTO(ff_yuv2nv21cX_madd_avx2);
YUV2NV_PROTO(ff_yuv2nv20cX_avx2);
static const NvKernel nv_kernels[] = {
#if HAVE_AVX2_EXTERNAL
    { "upstream",        ff_yuv2nv12cX_avx2,           AV_CPU_FLAG_AVX2,       8, 0, 8 },
    { "upstream",        ff_yuv2nv21cX_avx2,           AV_CPU_FLAG_AVX2,       8, 1, 8 },
    { "madd",            ff_yuv2nv12cX_madd_avx2,      AV_CPU_FLAG_AVX2,       8, 0, 1 },
    { "madd",            ff_yuv2nv21cX_madd_avx2,      AV_CPU_FLAG_AVX2,       8, 1, 1 },
    { "madd",            ff_yuv2nv20cX_avx2,           AV_CPU_FLAG_AVX2,      10, 0, 1 },
#endif
    { NULL },
};
#else
static const NvKernel nv_kernels[] = { { NULL } };
#endif

static void check_yuv2nvcX(enum AVPixelFormat fmt)
{
#define NV_MAX_FILTER 16
#define NV_MAX_W 1024
// what the kernels may read past dstW in the source lines; swscale leaves
// more than this after each line
#define NV_SRC_STRIDE (NV_MAX_W + 64)
#define NV_DST_SIZE (NV_MAX_W * 4 + 64)
    static const int filter_sizes[] = { 1, 2, 3, 4, 5, 6, 8, 9, 12, 16 };
    // tails of every length for 16 and 32 pixel units, a whole unit, and the
    // chroma widths of 240-1920 wide pictures
    static const int widths[] = { 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                                  47, 63, 64, 65, 120, 180, 240, 360, 640, 960 };
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(fmt);
    const int bits    = desc->comp[0].depth;
    const int swapped = fmt == AV_PIX_FMT_NV21;
    const int px_size = bits > 8 ? 4 : 2;
    yuv2planar1_fn p1;
    yuv2planarX_fn pX;
    yuv2packed1_fn pk1;
    yuv2packed2_fn pk2;
    yuv2packedX_fn pkX;
    yuv2anyX_fn any;
    yuv2interleavedX_fn c_fn = NULL;
    struct SwsContext *c;
    int fsi, wi, ki, i;

    declare_func_emms(AV_CPU_FLAG_MMX, void, enum AVPixelFormat dstFormat,
                      const uint8_t *chrDither, const int16_t *chrFilter,
                      int chrFilterSize, const int16_t **chrUSrc,
                      const int16_t **chrVSrc, uint8_t *dest, int dstW);

    const int16_t *srcU[NV_MAX_FILTER], *srcV[NV_MAX_FILTER];
    int16_t *pixU = av_malloc(NV_MAX_FILTER * NV_SRC_STRIDE * sizeof(int16_t));
    int16_t *pixV = av_malloc(NV_MAX_FILTER * NV_SRC_STRIDE * sizeof(int16_t));
    uint8_t *dst0 = av_malloc(NV_DST_SIZE);
    uint8_t *dst1 = av_malloc(NV_DST_SIZE);
    LOCAL_ALIGNED_16(int16_t, filter, [NV_MAX_FILTER]);
    LOCAL_ALIGNED_16(uint8_t, dither, [8]);

    if (!pixU || !pixV || !dst0 || !dst1) {
        fail();
        goto end;
    }
    // eight distinct values, so that a dither applied to the wrong pixel
    // shows; swscale passes rows of ff_dither_8x8_128 for >8 bit input
    randomize_buffers(dither, 8);
    // 15 bit, as the horizontal scalers produce for 8-14 bit output
    for (i = 0; i < NV_MAX_FILTER * NV_SRC_STRIDE; i++) {
        pixU[i] = rnd() & 0x7fff;
        pixV[i] = rnd() & 0x7fff;
    }
    for (i = 0; i < NV_MAX_FILTER; i++) {
        srcU[i] = pixU + i * NV_SRC_STRIDE;
        srcV[i] = pixV + i * NV_SRC_STRIDE;
    }

    c = sws_alloc_context();
    if (sws_init_context(c, NULL, NULL) < 0)
        fail();
    c->dstFormat = fmt;
    c->dstBpc    = bits;
    ff_sws_init_output_funcs(c, &p1, &pX, &c_fn, &pk1, &pk2, &pkX, &any);
    ff_getSwsFunc(c);
    if (!c_fn) {
        fail();
        goto done;
    }

    for (fsi = 0; fsi < FF_ARRAY_ELEMS(filter_sizes); fsi++) {
        const int fs = filter_sizes[fsi];
        // as the yuv2yuvX test: negative taps and one large positive one, so
        // that the output clips at both ends
        if (fs == 1) {
            filter[0] = (rnd() % (3 << 12)) - (1 << 12);
        } else {
            for (i = 0; i < fs; i++)
                filter[i] = -((1 << 12) / (fs - 1));
            filter[rnd() % fs] = (1 << 13) - 1;
        }

        for (wi = 0; wi < FF_ARRAY_ELEMS(widths); wi++) {
            const int w = widths[wi];
            for (ki = -1; ki < 0 || nv_kernels[ki].name; ki++) {
                const NvKernel *k = ki < 0 ? NULL : &nv_kernels[ki];
                const char *kname = k ? k->name : "dispatch";
                int unit = 1;
                if (k) {
                    if (k->bits != (bits > 8 ? 10 : 8) || k->swapped != swapped ||
                        !(av_get_cpu_flags() & k->cpu_flag))
                        continue;
                    if (!check_func(k->fn, "yuv2%scX_%d_%d_%s",
                                    desc->name, fs, w, k->name))
                        continue;
                    func_ref = c_fn;
                    unit = k->unit;
                } else {
                    if (!check_func(c->yuv2nv12cX, "yuv2%scX_%d_%d",
                                    desc->name, fs, w))
                        continue;
                    for (i = 0; nv_kernels[i].name; i++)
                        if (nv_kernels[i].fn == c->yuv2nv12cX)
                            unit = nv_kernels[i].unit;
                }

                memset(dst0, 0xAA, NV_DST_SIZE);
                memset(dst1, 0xAA, NV_DST_SIZE);
                call_ref(fmt, dither, filter, fs, srcU, srcV, dst0, w);
                call_new(fmt, dither, filter, fs, srcU, srcV, dst1, w);

                if (memcmp(dst0, dst1, w * px_size)) {
                    fail();
                    printf("failed: yuv2%scX_%d_%d %s\n", desc->name, fs, w, kname);
                    if (bits > 8)
                        show_differences_16((uint16_t *)dst0, (uint16_t *)dst1, w * 2);
                    else
                        show_differences_8(dst0, dst1, w * 2);
                }
                if (memcmp(dst0 + FFALIGN(w, unit) * px_size,
                           dst1 + FFALIGN(w, unit) * px_size,
                           NV_DST_SIZE - FFALIGN(w, unit) * px_size)) {
                    fail();
                    printf("failed: yuv2%scX_%d_%d %s wrote past dstW\n",
                           desc->name, fs, w, kname);
                }
                if ((w == 640 || w == 960) && (fs == 2 || fs == 4 || fs == 8))
                    bench_new(fmt, dither, filter, fs, srcU, srcV, dst1, w);
            }
        }
    }
done:
    sws_freeContext(c);
end:
    av_free(pixU);
    av_free(pixV);
    av_free(dst0);
    av_free(dst1);
#undef NV_MAX_FILTER
#undef NV_MAX_W
#undef NV_SRC_STRIDE
#undef NV_DST_SIZE
}
#define MAX_DST_W 1920
#define MAX_FILTER_WIDTH 40
// rounded up: randomize_buffers() writes in units of four bytes
#define SRC_SAMPLES 4096

typedef void (*hscale_fn)(struct SwsContext *c, int16_t *dst, int dstW,
                          const uint8_t *src, const int16_t *filter,
                          const int32_t *filterPos, int filterSize);

#define HSCALE_PROTO(name) \
    void name(struct SwsContext *c, int16_t *dst, int dstW, const uint8_t *src, \
              const int16_t *filter, const int32_t *filterPos, int filterSize)

// Kernels tested by name as well as through the dispatch, so that the ones
// written but not dispatched (and the ones a faster version shadows on this
// cpu) are covered too.  fs is the filter size, or -4/-8 for X4/X8.
typedef struct HscaleKernel {
    const char *name;
    hscale_fn fn;
    int cpu_flag;
    int depth;      // the source depth the function is for
    int to19;
    int fs;
} HscaleKernel;

#if ARCH_X86_64 && HAVE_X86ASM
#define DECL1(d, fs, opt) \
    HSCALE_PROTO(ff_hscale ## d ## to15_ ## fs ## _ ## opt); \
    HSCALE_PROTO(ff_hscale ## d ## to19_ ## fs ## _ ## opt);
#define DECL16(fs, opt) \
    DECL1(9, fs, opt) DECL1(10, fs, opt) DECL1(12, fs, opt) \
    DECL1(14, fs, opt) DECL1(16, fs, opt)
DECL1(8, 8,  avx2)
DECL1(8, X4, avx2)
DECL1(8, X8, avx2)
DECL1(8, X4_masked, avx2)
DECL1(8, X8_masked, avx2)
DECL1(8, 8,  avx512icl)
DECL1(8, X4, avx512icl)
DECL1(8, X8, avx512icl)
DECL16(X4,     avx2)
DECL16(X8,     avx2)
DECL16(X4_2px, avx2)
DECL16(X8_2px, avx2)
DECL16(X4_masked, avx2)
DECL16(X8_masked, avx2)
DECL16(8,      avx512icl)
DECL16(X4,     avx512icl)
DECL16(X8,     avx512icl)
#define K1(d, fs, fsn, opt, flag) \
    { #d "to15_" #fs "_" #opt, ff_hscale ## d ## to15_ ## fs ## _ ## opt, flag, d, 0, fsn }, \
    { #d "to19_" #fs "_" #opt, ff_hscale ## d ## to19_ ## fs ## _ ## opt, flag, d, 1, fsn }
#define K16(fs, fsn, opt, flag) \
    K1(9, fs, fsn, opt, flag), K1(10, fs, fsn, opt, flag), K1(12, fs, fsn, opt, flag), \
    K1(14, fs, fsn, opt, flag), K1(16, fs, fsn, opt, flag)
static const HscaleKernel hscale_kernels[] = {
#if HAVE_AVX2_EXTERNAL
    K1(8, 8,  8,  avx2, AV_CPU_FLAG_AVX2),
    K1(8, X4, -4, avx2, AV_CPU_FLAG_AVX2),
    K1(8, X8, -8, avx2, AV_CPU_FLAG_AVX2),
    K1(8, X4_masked, -4, avx2, AV_CPU_FLAG_AVX2),
    K1(8, X8_masked, -8, avx2, AV_CPU_FLAG_AVX2),
    K16(X4,     -4, avx2, AV_CPU_FLAG_AVX2),
    K16(X8,     -8, avx2, AV_CPU_FLAG_AVX2),
    // the 2 pixel kernels were only ever dispatched for 16 taps and up
    K16(X4_2px, -16 - 4, avx2, AV_CPU_FLAG_AVX2),
    K16(X8_2px, -16 - 8, avx2, AV_CPU_FLAG_AVX2),
    K16(X4_masked, -4, avx2, AV_CPU_FLAG_AVX2),
    K16(X8_masked, -8, avx2, AV_CPU_FLAG_AVX2),
#endif
#if HAVE_AVX512ICL_EXTERNAL
    K1(8, 8,  8,  avx512icl, AV_CPU_FLAG_AVX512ICL),
    K1(8, X4, -4, avx512icl, AV_CPU_FLAG_AVX512ICL),
    K1(8, X8, -8, avx512icl, AV_CPU_FLAG_AVX512ICL),
    K16(8,  8,  avx512icl, AV_CPU_FLAG_AVX512ICL),
    K16(X4, -4, avx512icl, AV_CPU_FLAG_AVX512ICL),
    K16(X8, -8, avx512icl, AV_CPU_FLAG_AVX512ICL),
#endif
    { NULL },
};
#else
static const HscaleKernel hscale_kernels[] = { { NULL } };
#endif

// fs -4/-8: X4/X8, for filters over 8 taps; -20/-24: the same from 16 taps up
static int kernel_fits(const HscaleKernel *k, int depth, int to19, int fs)
{
    int min = 9;
    int x   = -k->fs;
    if (!k->fn || k->depth != depth || k->to19 != to19)
        return 0;
    if (k->fs > 0)
        return k->fs == fs;
    if (x > 16) {
        min = 16;
        x  -= 16;
    }
    if (fs < min)
        return 0;
    return x == 4 ? (fs & 4) != 0 : !(fs & 4);
}

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
    // 20 and 28 are the same two cases with the X4 split tail on the end; 12
    // is a lone 8 + 4
    static const int filter_sizes[] = { 4, 8, 12, 16, 20, 24, 28, 32, 40 };
    // the obed scales (1920, 1280, 960, 720 and the 240/256 thumbnails), the
    // units of 4/8/16 the kernels work in, one over, and the odd and tiny
    static const int dst_widths[] = { 1, 3, 8, 12, 13, 16, 17, 33, 128, 144, 240,
                                      256, 512, 720, 960, 1280, 1919, 1920 };
    static const struct {
        int bpc;
        int bits;   // also the depth of the function swscale picks
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
    // the step between the filter positions of neighbouring output pixels, as
    // the scale ratio srcW/dstW: 1 is a mild scale, 5 spreads the taps of one
    // iteration of the avx512 window kernels over more than the window, so
    // that their fallback is covered, and the rest are the obed scales
    static const struct {
        int num, den;
        const char *name;
        int bench_w;    // the dstW obed uses at this ratio, or 0
    } pos_steps[] = {
        {  1, 1, "1",    512 },
        {  5, 1, "5",    256 },     // 1280x720 thumbnail
        {  3, 8, "3o8", 1920 },     // 720 -> 1920
        {  2, 3, "2o3", 1920 },     // 1280 -> 1920
        {  3, 2, "3o2", 1280 },     // 1920 -> 1280
        { 16, 9, "16o9", 720 },     // 1280 -> 720
        {  8, 3, "8o3",  720 },     // 1920 -> 720
        {  3, 1, "3",    240 },     // 720x576 thumbnail
        {  8, 1, "8",    240 },     // 1920x1080 thumbnail
    };
    // the C function per source type and output width, for the kernels
    // tested by name; the C pass of checkasm always runs first
    static hscale_fn c_fn[FF_ARRAY_ELEMS(src_types)][2];

    struct SwsContext *c;
    int sti, fsi, dwi, bi, psi, i, j, ki;

    LOCAL_ALIGNED_32(uint16_t, src, [SRC_SAMPLES]);
    int16_t *filter  = av_malloc((MAX_DST_W + 3) * MAX_FILTER_WIDTH * sizeof(*filter));
    int16_t *tfilter = av_malloc((MAX_DST_W + 3) * MAX_FILTER_WIDTH * sizeof(*filter));
    int32_t *filterPos = av_malloc((MAX_DST_W + 3) * sizeof(*filterPos));
    uint32_t *dst0 = av_malloc((MAX_DST_W + 8) * sizeof(*dst0));
    uint32_t *dst1 = av_malloc((MAX_DST_W + 8) * sizeof(*dst1));

    // dst is int16_t or int32_t depending on dstBpc
    declare_func_emms(AV_CPU_FLAG_MMX, void, struct SwsContext *c, int16_t *dst,
                      int dstW, const uint8_t *src, const int16_t *filter,
                      const int32_t *filterPos, int filterSize);

    if (!filter || !tfilter || !filterPos || !dst0 || !dst1) {
        fail();
        goto end;
    }

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
            for (i = 0; i < MAX_DST_W; i++) {
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
                int to19     = dst_bpc > 14;
                int dst_size = to19 ? sizeof(int32_t) : sizeof(int16_t);

                c->srcFormat        = src_types[sti].fmt;
                c->srcBpc           = src_types[sti].bpc;
                c->dstBpc           = dst_bpc;
                c->hLumFilterSize   = c->hChrFilterSize = width;
                ff_getSwsFunc(c);
                if (!av_get_cpu_flags())
                    c_fn[sti][to19] = c->hcScale;

                for (dwi = 0; dwi < FF_ARRAY_ELEMS(dst_widths); dwi++) {
                for (psi = 0; psi < FF_ARRAY_ELEMS(pos_steps); psi++) {
                    int dstW = dst_widths[dwi];
                    int num  = pos_steps[psi].num, den = pos_steps[psi].den;
                    // as initFilter(): centred taps, clamped to the line
                    int srcW = FFMAX((dstW * num + den - 1) / den, width);
                    // the scalers work in units of 4 pixels and may run into
                    // the last one, but no further
                    int written = FFALIGN(dstW, 4);
                    int do_bench = dstW == pos_steps[psi].bench_w;

                    if (srcW > SRC_SAMPLES)
                        continue;
                    for (i = 0; i < dstW; i++) {
                        int pos = (int)(((2LL * i + 1) * num) / (2 * den)) - width / 2;
                        filterPos[i] = av_clip(pos, 0, srcW - width);
                    }
                    // the padding initFilter() leaves for the scalers that
                    // read past dstW
                    memcpy(tfilter, filter, dstW * width * sizeof(*filter));
                    for (i = dstW; i < dstW + 3; i++) {
                        filterPos[i] = filterPos[dstW - 1];
                        memcpy(tfilter + i * width, filter + (dstW - 1) * width,
                               width * sizeof(*filter));
                    }

                    for (ki = -1; ki < 0 || hscale_kernels[ki].name; ki++) {
                        const HscaleKernel *k = ki < 0 ? NULL : &hscale_kernels[ki];
                        char kname[32];
                        if (k) {
                            if (!kernel_fits(k, src_types[sti].bits, to19, width) ||
                                !(av_get_cpu_flags() & k->cpu_flag) ||
                                !c_fn[sti][to19])
                                continue;
                            snprintf(kname, sizeof(kname), "%s", k->name);
                            if (!check_func(k->fn, "hscale_%d_to_%d__fs_%d_dstW_%d_step_%s__%s__%s",
                                            src_types[sti].bpc, to19 ? 19 : 15, width, dstW,
                                            pos_steps[psi].name,
                                            av_get_pix_fmt_name(src_types[sti].fmt), kname))
                                continue;
                            // the kernel registers itself first, so the C one is the reference
                            func_ref = c_fn[sti][to19];
                        } else {
                            snprintf(kname, sizeof(kname), "dispatch");
                            if (!check_func(c->hcScale, "hscale_%d_to_%d__fs_%d_dstW_%d_step_%s__%s",
                                            src_types[sti].bpc, to19 ? 19 : 15, width, dstW,
                                            pos_steps[psi].name,
                                            av_get_pix_fmt_name(src_types[sti].fmt)))
                                continue;
                        }

                        memset(dst0, 0, (MAX_DST_W + 8) * sizeof(dst0[0]));
                        memset(dst1, 0, (MAX_DST_W + 8) * sizeof(dst1[0]));

                        call_ref(c, (int16_t*)dst0, dstW, (const uint8_t*)src,
                                 tfilter, filterPos, width);
                        call_new(c, (int16_t*)dst1, dstW, (const uint8_t*)src,
                                 tfilter, filterPos, width);

                        if (memcmp(dst0, dst1, dstW * dst_size)) {
                            fail();
                            printf("failed: hscale_%d_to_%d__fs_%d_dstW_%d_step_%s__%s %s\n",
                                   src_types[sti].bpc, to19 ? 19 : 15,
                                   width, dstW, pos_steps[psi].name,
                                   av_get_pix_fmt_name(src_types[sti].fmt), kname);
                        }
                        if (memcmp((uint8_t*)dst0 + written * dst_size,
                                   (uint8_t*)dst1 + written * dst_size,
                                   (MAX_DST_W + 8) * sizeof(dst0[0]) - written * dst_size)) {
                            fail();
                            printf("failed: hscale_%d_to_%d__fs_%d_dstW_%d_step_%s__%s %s wrote past dstW\n",
                                   src_types[sti].bpc, to19 ? 19 : 15,
                                   width, dstW, pos_steps[psi].name,
                                   av_get_pix_fmt_name(src_types[sti].fmt), kname);
                        }

                        if (do_bench)
                            bench_new(c, (int16_t*)dst1, dstW, (const uint8_t*)src,
                                      tfilter, filterPos, width);
                    }
                }
                }
            }
        }
    }
    sws_freeContext(c);
end:
    av_free(filter);
    av_free(tfilter);
    av_free(filterPos);
    av_free(dst0);
    av_free(dst1);
#undef MAX_DST_W
#undef SRC_SAMPLES
#undef MAX_FILTER_WIDTH
}

void checkasm_check_sw_scale(void)
{
    check_hscale();
    report("hscale");
    check_yuv2yuvX(0, 8, AV_PIX_FMT_YUV420P);
    check_yuv2yuvX(1, 8, AV_PIX_FMT_YUV420P);
    report("yuv2yuvX_8");
    check_yuv2yuvX(0, 9, AV_PIX_FMT_YUV420P9LE);
    check_yuv2yuvX(1, 9, AV_PIX_FMT_YUV420P9LE);
    report("yuv2yuvX_9LE");
    check_yuv2yuvX(0, 10, AV_PIX_FMT_YUV420P10LE);
    check_yuv2yuvX(1, 10, AV_PIX_FMT_YUV420P10LE);
    report("yuv2yuvX_10LE");
    check_yuv2nv12cX(0);
    check_yuv2nv12cX(1);
    report("yuv2nv12cX");
    check_yuv2nvcX(AV_PIX_FMT_NV12);
    check_yuv2nvcX(AV_PIX_FMT_NV16);
    check_yuv2nvcX(AV_PIX_FMT_NV21);
    report("yuv2nv12cX_8");
    check_yuv2nvcX(AV_PIX_FMT_NV20LE);
    check_yuv2nvcX(AV_PIX_FMT_NV12_10LE);
    report("yuv2nv20cX");
}
