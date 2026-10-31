;******************************************************************************
;* AVX2 horizontal line scaling functions for 9 to 16 bit input
;* Copyright (c) 2011 Ronald S. Bultje <rsbultje@gmail.com>
;*
;* This file is part of FFmpeg.
;*
;* FFmpeg is free software; you can redistribute it and/or
;* modify it under the terms of the GNU Lesser General Public
;* License as published by the Free Software Foundation; either
;* version 2.1 of the License, or (at your option) any later version.
;*
;* FFmpeg is distributed in the hope that it will be useful,
;* but WITHOUT ANY WARRANTY; without even the implied warranty of
;* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
;* Lesser General Public License for more details.
;*
;* You should have received a copy of the GNU Lesser General Public
;* License along with FFmpeg; if not, write to the Free Software
;* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
;******************************************************************************

%include "libavutil/x86/x86util.asm"

%if ARCH_X86_64 && HAVE_AVX2_EXTERNAL

SECTION_RODATA 32

minshort_ymm:  times 16 dw 0x8000
unicoeff_ymm:  times  8 dd 0x20000000
max_19bit_ymm: times  8 dd 0x7ffff
; the 8 tap tree ends with one pixel per lane half: 0 2 4 6 | 1 3 5 7
swizzle_8tap:  dd 0, 4, 1, 5, 2, 6, 3, 7
; the last four of eight words, for the 4 tap remainder of the X4 kernels
hi4_words:     dw 0, 0, 0, 0, -1, -1, -1, -1
; a sliding window of dword masks: the n at mask_dw + 32 - 4n are -1
mask_dw:       times 8 dd -1
               times 8 dd 0

SECTION .text

;-----------------------------------------------------------------------------
; horizontal line scaling, 16 bit wide input
;
; void hscale<source_width>to<intermediate_nbits>_<filterSize>_avx2
;                               (SwsContext *c, int{16,32}_t *dst,
;                                int dstW, const uint16_t *src,
;                                const int16_t *filter,
;                                const int32_t *filterPos, int filterSize);
;
; Same contract as scale.asm.  The fixed 4 and 8 tap filters below do eight
; output pixels per iteration; the arbitrary filterSize ones are further down.
; The taps of neighbouring output pixels start at unrelated source positions,
; so a ymm of source has to be built from two 128 bit loads; the filter rows on
; the other hand are contiguous, and eight taps of two pixels (or four taps of
; four) are exactly the 32 bytes one ymm operand wants.  That halves the
; pmaddwd and phaddd count for the same number of loads, and no gather is
; needed.
;
; The wide loop only runs while eight whole pixels are left.  The tail is the
; 128 bit code, so the read past dstW stays inside the three filterPos[] and
; filter[] entries initFilter() pads, and the write inside the FFALIGN(dstW, 4)
; the existing scalers already claim.
;-----------------------------------------------------------------------------

; 8 taps of two pixels, one per lane
%macro LOAD_8TAP 2 ; dst, first pixel
    movsxd     pos0q, dword [fltposq+cntq*4+%2*4+0]
    movsxd     pos1q, dword [fltposq+cntq*4+%2*4+4]
    movu        xm%1, [srcq+pos0q*2]
    vinserti128  m%1, m%1, [srcq+pos1q*2], 1
%endmacro

; 4 taps of four pixels, two per lane.  m5 is scratch.
%macro LOAD_4TAP 2 ; dst, first pixel
    movsxd     pos0q, dword [fltposq+cntq*4+%2*4+0]
    movsxd     pos1q, dword [fltposq+cntq*4+%2*4+4]
    movq        xm%1, [srcq+pos0q*2]
    movhps      xm%1, [srcq+pos1q*2]
    movsxd     pos0q, dword [fltposq+cntq*4+%2*4+8]
    movsxd     pos1q, dword [fltposq+cntq*4+%2*4+12]
    movq         xm5, [srcq+pos0q*2]
    movhps       xm5, [srcq+pos1q*2]
    vinserti128  m%1, m%1, xm5, 1
%endmacro

; SCALE_FUNC16 source_width, intermediate_nbits, filtersize
%macro SCALE_FUNC16 3
cglobal hscale%1to%2_%3, 6, 8, 10, pos0, dst, w, src, filter, fltpos, pos1, cnt
    movsxd        wq, wd
    xor         cntq, cntq
%if %1 == 16
    mova          m8, [minshort_ymm]
    mova          m9, [unicoeff_ymm]
%endif
%if %2 == 19
    mova          m6, [max_19bit_ymm]
%if %3 == 8
    mova          m7, [swizzle_8tap]
%endif
%endif
    cmp           wq, 8
    jl .tail_loop
    sub           wq, 8

.loop:
%if %3 == 8
    LOAD_8TAP      0, 0
    LOAD_8TAP      1, 2
    LOAD_8TAP      2, 4
    LOAD_8TAP      3, 6
%if %1 == 16 ; pmaddwd needs signed adds, so this moves unsigned -> signed,
             ; the 0x8000 * sum(coeffs) comes back after the horizontal add
    psubw         m0, m8
    psubw         m1, m8
    psubw         m2, m8
    psubw         m3, m8
%endif
    pmaddwd       m0, [filterq+ 0]
    pmaddwd       m1, [filterq+32]
    pmaddwd       m2, [filterq+64]
    pmaddwd       m3, [filterq+96]

    phaddd        m0, m1                  ; px0 halves, px2 halves | px1, px3
    phaddd        m2, m3                  ; px4, px6 | px5, px7
    phaddd        m0, m2                  ; px0 px2 px4 px6 | px1 px3 px5 px7
%else ; %3 == 4
    LOAD_4TAP      0, 0
    LOAD_4TAP      1, 4
%if %1 == 16
    psubw         m0, m8
    psubw         m1, m8
%endif
    pmaddwd       m0, [filterq+ 0]
    pmaddwd       m1, [filterq+32]

    phaddd        m0, m1                  ; px0 px1 px4 px5 | px2 px3 px6 px7
%endif ; %3 == 8/4

%if %1 == 16
    paddd         m0, m9
%endif
    psrad         m0, 14 + %1 - %2
%if %2 == 15
    packssdw      m0, m0
    vextracti128 xm4, m0, 1
%if %3 == 8
    punpcklwd    xm0, xm0, xm4            ; the lanes interleave word by word
%else
    punpckldq    xm0, xm0, xm4            ; ... and pairwise for 4 taps
%endif
    movu [dstq+cntq*2], xm0
%else ; %2 == 19
%if %3 == 8
    vpermd        m0, m7, m0
%else
    vpermq        m0, m0, q3120
%endif
    PMINSD        m0, m6, m4
    movu [dstq+cntq*4], m0
%endif ; %2 == 15/19
    add      filterq, %3*16
    add         cntq, 8
    cmp         cntq, wq
    jle .loop

    add           wq, 8
    cmp         cntq, wq
    jge .end

    ; 1-7 pixels left, or dstW < 8 to start with
.tail_loop:
%if %3 == 8
    movsxd     pos0q, dword [fltposq+cntq*4+ 0]
    movsxd     pos1q, dword [fltposq+cntq*4+ 4]
    movu         xm0, [srcq+pos0q*2]
    movu         xm1, [srcq+pos1q*2]
    movsxd     pos0q, dword [fltposq+cntq*4+ 8]
    movsxd     pos1q, dword [fltposq+cntq*4+12]
    movu         xm2, [srcq+pos0q*2]
    movu         xm3, [srcq+pos1q*2]
%if %1 == 16
    psubw        xm0, xm8
    psubw        xm1, xm8
    psubw        xm2, xm8
    psubw        xm3, xm8
%endif
    pmaddwd      xm0, [filterq+ 0]
    pmaddwd      xm1, [filterq+16]
    pmaddwd      xm2, [filterq+32]
    pmaddwd      xm3, [filterq+48]

    phaddd       xm0, xm1
    phaddd       xm2, xm3
    phaddd       xm0, xm2                 ; px0 px1 px2 px3
%else ; %3 == 4
    movsxd     pos0q, dword [fltposq+cntq*4+ 0]
    movsxd     pos1q, dword [fltposq+cntq*4+ 4]
    movq         xm0, [srcq+pos0q*2]
    movhps       xm0, [srcq+pos1q*2]
    movsxd     pos0q, dword [fltposq+cntq*4+ 8]
    movsxd     pos1q, dword [fltposq+cntq*4+12]
    movq         xm1, [srcq+pos0q*2]
    movhps       xm1, [srcq+pos1q*2]
%if %1 == 16
    psubw        xm0, xm8
    psubw        xm1, xm8
%endif
    pmaddwd      xm0, [filterq+ 0]
    pmaddwd      xm1, [filterq+16]

    phaddd       xm0, xm1                 ; px0 px1 px2 px3
%endif ; %3 == 8/4

%if %1 == 16
    paddd        xm0, xm9
%endif
    psrad        xm0, 14 + %1 - %2
%if %2 == 15
    packssdw     xm0, xm0
    movq [dstq+cntq*2], xm0
%else ; %2 == 19
    PMINSD       xm0, xm6, xm4
    movu [dstq+cntq*4], xm0
%endif
    add      filterq, %3*8
    add         cntq, 4
    cmp         cntq, wq
    jl .tail_loop
.end:
    RET
%endmacro

;-----------------------------------------------------------------------------
; The same for an arbitrary filterSize: two output pixels in flight, sixteen
; taps of each per iteration.  A pixel row of the filter is contiguous along
; the tap dimension, so both the source and the coefficients are one 32 byte
; load, and the loop does twice the taps of the ssse3 one for the same four
; memory operands.
;
; The loop part is filterSize-dlt taps, always a multiple of 8, so at most one
; odd 8 tap chunk is left over; that is exactly filterSize & 8.  The X4 tail is
; the ssse3 one unchanged: the last four taps of pixel 0 and the first four of
; pixel 1 are one 16 byte filter load, and land in the layout the folded
; accumulator already has.
;-----------------------------------------------------------------------------

; SCALE_FUNC16_X source_width, intermediate_nbits, filtersuffix
%macro SCALE_FUNC16_X 3
%ifidn %3, X4
%define dlt 4
%else ; X8
%define dlt 0
%endif
cglobal hscale%1to%2_%3_2px, 7, 10, 9, pos0, dst, w, srcmem, filter, fltpos, fltsize
%define srcq      r7
%define pos1q     r8
%define srcend16q r9
    movsxd        wq, wd
    movsxd  fltsizeq, fltsized
%if %1 == 16
    mova          m6, [minshort_ymm]
    mova          m7, [unicoeff_ymm]
%endif
%if %2 == 19
    mova          m8, [max_19bit_ymm]
%endif
    lea   srcend16q, [srcmemq+fltsizeq*2-(dlt+16)*2] ; &src[filterSize-dlt-16]
    lea      fltposq, [fltposq+wq*4]
%if %2 == 15
    lea         dstq, [dstq+wq*2]
%else ; %2 == 19
    lea         dstq, [dstq+wq*4]
%endif
    neg           wq

.loop:
    movsxd     pos0q, dword [fltposq+wq*4+0]   ; filterPos[0]
    movsxd     pos1q, dword [fltposq+wq*4+4]   ; filterPos[1]
    pxor          m4, m4
    pxor          m5, m5
    mov         srcq, srcmemq
    cmp         srcq, srcend16q
    jg .chunk8                                 ; fewer than 16 taps to loop over

.innerloop:
    movu          m0, [srcq+ pos0q     *2]     ; src[filterPos[0] + {0..15}]
    movu          m1, [srcq+(pos1q+dlt)*2]     ; src[filterPos[1] + {0..15}]
%if %1 == 16
    psubw         m0, m6
    psubw         m1, m6
%endif
    pmaddwd       m0, [filterq]
    pmaddwd       m1, [filterq+(fltsizeq+dlt)*2]
    paddd         m4, m0
    paddd         m5, m1
    add      filterq, 32
    add         srcq, 32
    cmp         srcq, srcend16q
    jle .innerloop

.chunk8:
    test    fltsized, 8
    jz .taps_done
    ; 8 taps left.  The xmm loads leave the upper lane zeroed, so the 16 bytes
    ; of filter the ymm operand takes from the next row multiply by nothing.
    movu         xm0, [srcq+ pos0q     *2]
    movu         xm1, [srcq+(pos1q+dlt)*2]
%if %1 == 16
    psubw        xm0, xm6
    psubw        xm1, xm6
%endif
    pmaddwd       m0, [filterq]
    pmaddwd       m1, [filterq+(fltsizeq+dlt)*2]
    paddd         m4, m0
    paddd         m5, m1
    add      filterq, 16
    add         srcq, 16

.taps_done:
    phaddd        m4, m5                       ; px0 pairs, px1 pairs, per lane
    vextracti128 xm2, m4, 1
    paddd        xm4, xm2                      ; fold the lanes together
%if dlt == 4
    sub        pos1q, fltsizeq                 ; split last 4 srcpx of dstpx[0]
    movq         xm0, [srcq+ pos0q     *2]     ; and first 4 srcpx of dstpx[1]
    movhps       xm0, [srcq+(pos1q+dlt)*2]
%if %1 == 16
    psubw        xm0, xm6
%endif
    pmaddwd      xm0, [filterq]
    paddd        xm4, xm0
%endif ; dlt == 4
    lea      filterq, [filterq+(fltsizeq+dlt)*2]
    phaddd       xm4, xm4                      ; dstpx[0], dstpx[1]

%if %1 == 16
    paddd        xm4, xm7
%endif
    psrad        xm4, 14 + %1 - %2
%if %2 == 15
    packssdw     xm4, xm4
    movd [dstq+wq*2], xm4
%else ; %2 == 19
    PMINSD       xm4, xm8, xm2
    movq [dstq+wq*4], xm4
%endif ; %2 == 15/19
    add           wq, 2
    jl .loop
    RET
%endmacro

;-----------------------------------------------------------------------------
; An arbitrary filterSize again, four output pixels in flight.  The kernel
; above (kept as _2px) folds its two accumulators after every pair of pixels,
; and that fold, not the multiplies, is what bounds it: with a 16 tap filter it
; is no faster than ssse3.  Four accumulators fold in one HADD4D tree, half the
; shuffles a pixel, and the four source rows and filter rows are walked by one
; shared offset, since a tap is two bytes of either.
;
; After the whole 16 tap chunks, an 8 tap one (filterSize & 8) and, for X4, a
; 4 tap one.  The 4 tap one loads the eight words that end at the last tap and
; masks off the four already done, so neither the source nor the filter row
; is read past its end.  This also covers filterSize 12, which the _2px kernel
; leaves to ssse3.  Pixels are done in fours up to FFALIGN(dstW, 4), as the
; ssse3 kernels do.
;-----------------------------------------------------------------------------

; Per 128 bit lane, the sums of the four dwords of each of %1-%4, in that
; order, to %1.  %5 is scratch.
%macro HADD4D 5
    punpckldq       %5, %1, %2
    punpckhdq       %1, %2
    paddd           %1, %5              ; a02 b02 a13 b13
    punpckldq       %5, %3, %4
    punpckhdq       %3, %4
    paddd           %3, %5              ; c02 d02 c13 d13
    punpcklqdq      %5, %1, %3
    punpckhqdq      %1, %3
    paddd           %1, %5              ; a b c d
%endmacro

; 16 bit input is moved to signed for pmaddwd, as above; %1 = register, %2 = minshort
%macro TO_SIGNED16 2
%if src_width == 16
    psubw           %1, %2
%endif
%endmacro

; SCALE_FUNC16_XQ source_width, intermediate_nbits, X4/X8
%macro SCALE_FUNC16_XQ 3
%assign src_width %1
cglobal hscale%1to%2_%3, 7, 15, 12, f1, dst, w, src, filter, fltpos, fltsize, s0, s1, s2, s3, f2, f3, tap, lim
    movsxd  fltsizeq, fltsized
    lea         limq, [fltsizeq*2-32]     ; the last byte a 16 tap chunk may start at
%if %1 == 16
    mova         m10, [minshort_ymm]
    mova         m11, [unicoeff_ymm]
%endif
%if %2 == 19
    mova          m9, [max_19bit_ymm]
%endif
%ifidn %3, X4
    mova         xm8, [hi4_words]
%endif

.loop:
    movsxd       s0q, dword [fltposq+ 0]
    movsxd       s1q, dword [fltposq+ 4]
    movsxd       s2q, dword [fltposq+ 8]
    movsxd       s3q, dword [fltposq+12]
    lea          s0q, [srcq+s0q*2]
    lea          s1q, [srcq+s1q*2]
    lea          s2q, [srcq+s2q*2]
    lea          s3q, [srcq+s3q*2]
    lea          f1q, [filterq+fltsizeq*2]
    lea          f2q, [filterq+fltsizeq*4]
    lea          f3q, [f1q+fltsizeq*4]
    pxor          m0, m0
    pxor          m1, m1
    pxor          m2, m2
    pxor          m3, m3
    xor         tapd, tapd
    test        limq, limq
    js .rem8

.inner:
    movu          m4, [s0q+tapq]
    movu          m5, [s1q+tapq]
    movu          m6, [s2q+tapq]
    movu          m7, [s3q+tapq]
    TO_SIGNED16   m4, m10
    TO_SIGNED16   m5, m10
    TO_SIGNED16   m6, m10
    TO_SIGNED16   m7, m10
    pmaddwd       m4, [filterq+tapq]
    pmaddwd       m5, [f1q+tapq]
    pmaddwd       m6, [f2q+tapq]
    pmaddwd       m7, [f3q+tapq]
    paddd         m0, m4
    paddd         m1, m5
    paddd         m2, m6
    paddd         m3, m7
    add         tapq, 32
    cmp         tapq, limq
    jle .inner

.rem8:
    test    fltsized, 8
    jz .rem4
    ; the xmm results leave the upper lane zeroed
    movu         xm4, [s0q+tapq]
    movu         xm5, [s1q+tapq]
    movu         xm6, [s2q+tapq]
    movu         xm7, [s3q+tapq]
    TO_SIGNED16  xm4, xm10
    TO_SIGNED16  xm5, xm10
    TO_SIGNED16  xm6, xm10
    TO_SIGNED16  xm7, xm10
    pmaddwd      xm4, [filterq+tapq]
    pmaddwd      xm5, [f1q+tapq]
    pmaddwd      xm6, [f2q+tapq]
    pmaddwd      xm7, [f3q+tapq]
    paddd         m0, m4
    paddd         m1, m5
    paddd         m2, m6
    paddd         m3, m7
    add         tapq, 16
.rem4:
%ifidn %3, X4
    movu         xm4, [s0q+tapq-8]
    movu         xm5, [s1q+tapq-8]
    movu         xm6, [s2q+tapq-8]
    movu         xm7, [s3q+tapq-8]
    TO_SIGNED16  xm4, xm10
    TO_SIGNED16  xm5, xm10
    TO_SIGNED16  xm6, xm10
    TO_SIGNED16  xm7, xm10
    pand         xm4, xm8
    pand         xm5, xm8
    pand         xm6, xm8
    pand         xm7, xm8
    pmaddwd      xm4, [filterq+tapq-8]
    pmaddwd      xm5, [f1q+tapq-8]
    pmaddwd      xm6, [f2q+tapq-8]
    pmaddwd      xm7, [f3q+tapq-8]
    paddd         m0, m4
    paddd         m1, m5
    paddd         m2, m6
    paddd         m3, m7
%endif

    HADD4D        m0, m1, m2, m3, m4      ; px0-3 | px0-3
    vextracti128 xm1, m0, 1
    paddd        xm0, xm1
%if %1 == 16
    paddd        xm0, xm11
%endif
    psrad        xm0, 14 + %1 - %2
%if %2 == 15
    packssdw     xm0, xm0
    movq      [dstq], xm0
    add         dstq, 8
%else ; %2 == 19
    pminsd       xm0, xm9
    movu      [dstq], xm0
    add         dstq, 16
%endif ; %2 == 15/19
    lea      filterq, [f3q+fltsizeq*2]
    add       fltposq, 16
    sub           wd, 4
    jg .loop
    RET
%endmacro

;-----------------------------------------------------------------------------
; The masked version: eight output pixels in flight, a pixel its whole 16 tap
; chunks and then one chunk of the 4, 8, 12 or 16 taps left, loaded with
; vpmaskmovd for both source and filter (a tap is a word of either, and the 4
; tap granularity of filterSize a dword one), so no remainder steps and
; nothing past the row read.  For 16 bit input the taps past the row become
; -0x8000, but the masked filter load has zeroed their coefficients.  As
; scale8.asm's X4_masked.
;-----------------------------------------------------------------------------

; one pixel: %1 accumulator, %2 pixel, %3 the whole chunks before the masked
; last one at lastoff: 0, 1 (each unrolled), or 2 for nfull of them in a loop.
; m9 is the mask of the last chunk, rowq the bytes of a filter row, lastq
; those of the last chunk.  m10 scratch.
%macro X16_PIXEL_AVX2 3
    movsxd      posq, dword [fltposq+%2*4]
    lea         posq, [srcq+posq*2]
    ; masked loads do not train the hardware prefetcher (Zen 5): a line
    ; fresh from memory is ~2x slower without this
%if %2 == 0
    prefetcht0  [posq+512]
%endif
    vpmaskmovd   m%1, m9, [posq+lastoffq]
    TO_SIGNED16  m%1, m12
    vpmaskmovd    m10, m9, [filterq+lastoffq]
    pmaddwd      m%1, m10
%if %3 == 1
    movu         m10, [posq]
    TO_SIGNED16  m10, m12
    pmaddwd      m10, [filterq]
    paddd        m%1, m10
%elif %3 == 2
    mov      chunksq, nfullq
%%chunk:
    movu         m10, [posq]
    TO_SIGNED16  m10, m12
    pmaddwd      m10, [filterq]
    paddd        m%1, m10
    add         posq, 32
    add      filterq, 32
    dec      chunksq
    jnz %%chunk
%endif
%if %3 == 2
    add      filterq, lastq
%else
    add      filterq, rowq
%endif
%endmacro

; a whole line: %1 source_width, %2 intermediate_nbits, %3 chunk mode
%macro X16_LINE_AVX2 3
    sub           wd, 4
    jle .tail%3
.loop%3:
%assign %%i 0
%rep 8
    X16_PIXEL_AVX2 %%i, %%i, %3
%assign %%i %%i+1
%endrep
    HADD4D        m0, m1, m2, m3, m10     ; px 0-3 | px 0-3
    HADD4D        m4, m5, m6, m7, m10     ; px 4-7 | px 4-7
    vperm2i128   m10, m0, m4, 0x31        ; the high lanes
    vinserti128   m0, m0, xm4, 1          ; the low lanes
    paddd         m0, m10
%if %1 == 16
    paddd         m0, m13
%endif
    psrad         m0, 14 + %1 - %2
%if %2 == 15
    vextracti128 xm1, m0, 1
    packssdw     xm0, xm1
    movu      [dstq], xm0
    add         dstq, 16
%else
    pminsd        m0, m11
    movu      [dstq], m0
    add         dstq, 32
%endif
    add       fltposq, 32
    sub           wd, 8
    jg .loop%3
.tail%3:
    cmp           wd, -4
    jle .end
    ; 1-4 pixels left
%assign %%i 0
%rep 4
    X16_PIXEL_AVX2 %%i, %%i, %3
%assign %%i %%i+1
%endrep
    HADD4D        m0, m1, m2, m3, m10
    vextracti128 xm1, m0, 1
    paddd        xm0, xm1
%if %1 == 16
    paddd        xm0, xm13
%endif
    psrad        xm0, 14 + %1 - %2
%if %2 == 15
    packssdw     xm0, xm0
    movq      [dstq], xm0
%else
    pminsd       xm0, xm11
    movu      [dstq], xm0
%endif
    jmp .end
%endmacro

; SCALE_FUNC16_XM source_width, intermediate_nbits, X4/X8
%macro SCALE_FUNC16_XM 3
%assign src_width %1
cglobal hscale%1to%2_%3_masked, 7, 13, 14, pos, dst, w, src, filter, fltpos, fltsize, chunks, nfull, lastoff, last, tmp, row
    movsxd  fltsizeq, fltsized
%if %1 == 16
    mova         m12, [minshort_ymm]
    mova         m13, [unicoeff_ymm]
%endif
%if %2 == 19
    mova         m11, [max_19bit_ymm]
%endif
    lea         rowq, [fltsizeq*2]
    lea       nfullq, [fltsizeq-1]
    shr       nfullq, 4                   ; whole 16 tap chunks before the last
    mov     lastoffq, nfullq
    shl     lastoffq, 5                   ; the byte offset of the last chunk
    mov        lastq, rowq
    sub        lastq, lastoffq            ; its bytes: 8, 16, 24 or 32
    lea         tmpq, [mask_dw+32]
    sub         tmpq, lastq
    movu          m9, [tmpq]              ; taps/2 dwords
    cmp       nfulld, 1
    jl .chunks0
    je .chunks1
    X16_LINE_AVX2 %1, %2, 2
.chunks1:
    X16_LINE_AVX2 %1, %2, 1
.chunks0:
    X16_LINE_AVX2 %1, %2, 0
.end:
    RET
%endmacro

; SCALE_FUNCS16 source_width
%macro SCALE_FUNCS16 1
SCALE_FUNC16    %1, 15, 4
SCALE_FUNC16    %1, 15, 8
SCALE_FUNC16_X  %1, 15, X4
SCALE_FUNC16_X  %1, 15, X8
SCALE_FUNC16_XQ %1, 15, X4
SCALE_FUNC16_XQ %1, 15, X8
SCALE_FUNC16_XM %1, 15, X4
SCALE_FUNC16_XM %1, 15, X8
SCALE_FUNC16    %1, 19, 4
SCALE_FUNC16    %1, 19, 8
SCALE_FUNC16_X  %1, 19, X4
SCALE_FUNC16_X  %1, 19, X8
SCALE_FUNC16_XQ %1, 19, X4
SCALE_FUNC16_XQ %1, 19, X8
SCALE_FUNC16_XM %1, 19, X4
SCALE_FUNC16_XM %1, 19, X8
%endmacro

INIT_YMM avx2
SCALE_FUNCS16  9
SCALE_FUNCS16 10
SCALE_FUNCS16 12
SCALE_FUNCS16 14
SCALE_FUNCS16 16

%endif ; ARCH_X86_64 && HAVE_AVX2_EXTERNAL
