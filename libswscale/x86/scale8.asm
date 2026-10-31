;******************************************************************************
;* AVX2 and AVX-512ICL horizontal line scaling functions for 8 bit input
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

%if ARCH_X86_64

SECTION_RODATA 64

; the dwords of the offset vector are all < 64, so the value of offset i is the
; byte at 4i; vpermb spreads it over the four taps of that pixel
spread_offset: db  0,  0,  0,  0,  4,  4,  4,  4,  8,  8,  8,  8, 12, 12, 12, 12
               db 16, 16, 16, 16, 20, 20, 20, 20, 24, 24, 24, 24, 28, 28, 28, 28
               db 32, 32, 32, 32, 36, 36, 36, 36, 40, 40, 40, 40, 44, 44, 44, 44
               db 48, 48, 48, 48, 52, 52, 52, 52, 56, 56, 56, 56, 60, 60, 60, 60
tap_offset:    times 16 db 0, 1, 2, 3
; pmaddwd leaves the two halves of a pixel in a pair of dwords, so the sums are
; the even ones of the two accumulators
even_dwords:   dd 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
; 8 taps: the word at 8p+t of group g is tap t of pixel 4g+p, so its index is
; the low byte of offset dword 4g+p, plus t.  The odd bytes are zeroed by the
; vpermb mask, so they are don't-cares.
spread8_g0:    db  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0,  0, 0
               db  4, 4,  4, 4,  4, 4,  4, 4,  4, 4,  4, 4,  4, 4,  4, 4
               db  8, 8,  8, 8,  8, 8,  8, 8,  8, 8,  8, 8,  8, 8,  8, 8
               db 12,12, 12,12, 12,12, 12,12, 12,12, 12,12, 12,12, 12,12
spread8_g1:    times 16 db 16
               times 16 db 20
               times 16 db 24
               times 16 db 28
spread8_g2:    times 16 db 32
               times 16 db 36
               times 16 db 40
               times 16 db 44
spread8_g3:    times 16 db 48
               times 16 db 52
               times 16 db 56
               times 16 db 60
tap8_offset:   times 4 db 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7
; lane l of group g is pixel 4g+l, and the transpose leaves pixel 4g+l in
; dword 4l+g
transpose16:   dd 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
; the eight pixel reduction leaves pixel l of 0-3 in dword 4l, 4+l in 4l+1
collect8:      dd 0, 4, 8, 12, 1, 5, 9, 13
; the 8 tap tree ends with one pixel per lane half: 0 2 4 6 | 1 3 5 7
swizzle_8tap:  dd 0, 4, 1, 5, 2, 6, 3, 7
; the constants above keep this one 32 byte aligned for mova
max_19bit_ymm: times 8 dd 0x7ffff
; a sliding window of dword masks: the n at mask_dw + 32 - 4n are -1
mask_dw:       times 8 dd -1
               times 8 dd 0
even_dwords_x: dd 0, 2, 4, 6
; the last four of eight words, for the 4 tap remainder of the X4 kernels
hi4_words:     dw 0, 0, 0, 0, -1, -1, -1, -1
max_offset:    dd 60
max_offset8:   dd 56
max_19bit:     dd 0x7ffff

SECTION .text

; Per 128 bit lane, the sums of the four dwords of each of %1-%4, in that
; order, to %1.  The shuffle and add count of a phaddd tree, but with an EVEX
; form, so the avx2 and avx512 kernels share it.  %5 is scratch.
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

; four pixels of eight taps at 128 bits, for the tails of the 8 tap kernels.
; %1 intermediate_nbits, %2 the register holding max_19bit.  xm0-xm4 scratch.
%macro HSCALE8_8_TAIL 2
    movsxd     pos0q, dword [fltposq+cntq*4+ 0]
    movsxd     pos1q, dword [fltposq+cntq*4+ 4]
    pmovzxbw     xm0, [srcq+pos0q]
    pmovzxbw     xm1, [srcq+pos1q]
    movsxd     pos0q, dword [fltposq+cntq*4+ 8]
    movsxd     pos1q, dword [fltposq+cntq*4+12]
    pmovzxbw     xm2, [srcq+pos0q]
    pmovzxbw     xm3, [srcq+pos1q]
    pmaddwd      xm0, [filterq+ 0]
    pmaddwd      xm1, [filterq+16]
    pmaddwd      xm2, [filterq+32]
    pmaddwd      xm3, [filterq+48]
    HADD4D       xm0, xm1, xm2, xm3, xm4
    psrad        xm0, 14 + 8 - %1
%if %1 == 15
    packssdw     xm0, xm0
    movq [dstq+cntq*2], xm0
%else
    pminsd       xm0, xm%2
    movu [dstq+cntq*4], xm0
%endif
    add      filterq, 64
    add         cntq, 4
%endmacro

;-----------------------------------------------------------------------------
; horizontal line scaling, 8 bit input, 4 tap filter
;
; void hscale8to<intermediate_nbits>_4_<opt>
;                               (SwsContext *c, int{16,32}_t *dst,
;                                int dstW, const uint8_t *src,
;                                const int16_t *filter,
;                                const int32_t *filterPos, int filterSize);
;
; Same contract as scale.asm.  Four taps of one output pixel are four source
; bytes at filterPos[i], and the taps of neighbouring pixels start at unrelated
; positions, so the work is all in collecting the bytes: the arithmetic that
; follows is one pmaddwd and one horizontal add per four pixels either way.
;
; avx2 builds the four dwords of a ymm's worth of pixels with pinsrd and widens
; them in one pmovzxbw, which keeps the pmaddwd/phaddd count at half of the
; ssse3 one for the same number of loads, and needs no gather.
;
; The wide loop only runs while eight whole pixels are left.  The tail is the
; 128 bit code, so the read past dstW stays inside the three filterPos[] and
; filter[] entries initFilter() pads, and the write inside the FFALIGN(dstW, 4)
; the existing scalers already claim.
;-----------------------------------------------------------------------------

; four output pixels of filter taps, byte -> word
%macro LOAD_4PX_8BIT 2 ; dst, first pixel
    movsxd     pos0q, dword [fltposq+cntq*4+%2*4+ 0]
    movsxd     pos1q, dword [fltposq+cntq*4+%2*4+ 4]
    movd        xm%1, [srcq+pos0q]
    pinsrd      xm%1, xm%1, [srcq+pos1q], 1
    movsxd     pos0q, dword [fltposq+cntq*4+%2*4+ 8]
    movsxd     pos1q, dword [fltposq+cntq*4+%2*4+12]
    pinsrd      xm%1, xm%1, [srcq+pos0q], 2
    pinsrd      xm%1, xm%1, [srcq+pos1q], 3
    pmovzxbw     m%1, xm%1
%endmacro

; the 128 bit tail, four output pixels; also the fallback of the avx512 loop.
; phaddd has no EVEX form, so under avx512 the pairs fold with a shift and the
; sums are collected by the same permute the wide loop uses.
%macro HSCALE8_4_TAIL 3 ; intermediate_nbits, max_19bit reg, even_dwords reg
    movsxd     pos0q, dword [fltposq+cntq*4+ 0]
    movsxd     pos1q, dword [fltposq+cntq*4+ 4]
    movd         xm0, [srcq+pos0q]
    pinsrd       xm0, xm0, [srcq+pos1q], 1
    movsxd     pos0q, dword [fltposq+cntq*4+ 8]
    movsxd     pos1q, dword [fltposq+cntq*4+12]
    movd         xm1, [srcq+pos0q]
    pinsrd       xm1, xm1, [srcq+pos1q], 1
    pmovzxbw     xm0, xm0
    pmovzxbw     xm1, xm1
    pmaddwd      xm0, [filterq+ 0]
    pmaddwd      xm1, [filterq+16]
%if cpuflag(avx512)
    psrlq        xm2, xm0, 32
    paddd        xm0, xm2
    psrlq        xm2, xm1, 32
    paddd        xm1, xm2
    vpermt2d     xm0, xm%3, xm1           ; px0 px1 px2 px3
%else
    phaddd       xm0, xm1                 ; px0 px1 px2 px3
%endif
    psrad        xm0, 14 + 8 - %1
%if %1 == 15
    packssdw     xm0, xm0
    movq [dstq+cntq*2], xm0
%else
    pminsd       xm0, xm%2
    movu [dstq+cntq*4], xm0
%endif
    add      filterq, 32
    add         cntq, 4
%endmacro

%if HAVE_AVX2_EXTERNAL
INIT_YMM avx2

; SCALE8_FUNC intermediate_nbits
%macro SCALE8_FUNC 1
cglobal hscale8to%1_4, 6, 8, 6, pos0, dst, w, src, filter, fltpos, pos1, cnt
    movsxd        wq, wd
    xor         cntq, cntq
%if %1 == 19
    mova          m3, [max_19bit_ymm]
%endif
    cmp           wq, 8
    jl .tail_loop
    sub           wq, 8

.loop:
    LOAD_4PX_8BIT  0, 0
    LOAD_4PX_8BIT  1, 4
    pmaddwd       m0, [filterq+ 0]
    pmaddwd       m1, [filterq+32]

    phaddd        m0, m1                  ; px0 px1 px4 px5 | px2 px3 px6 px7
    psrad         m0, 14 + 8 - %1
%if %1 == 15
    packssdw      m0, m0
    vextracti128 xm4, m0, 1
    punpckldq    xm0, xm0, xm4            ; the lanes interleave pairwise
    movu [dstq+cntq*2], xm0
%else ; %1 == 19
    vpermq        m0, m0, q3120
    pminsd        m0, m3
    movu [dstq+cntq*4], m0
%endif ; %1 == 15/19
    add      filterq, 64
    add         cntq, 8
    cmp         cntq, wq
    jle .loop

    add           wq, 8
    cmp         cntq, wq
    jge .end

    ; 1-7 pixels left, or dstW < 8 to start with
.tail_loop:
    HSCALE8_4_TAIL %1, 3, 0
    cmp         cntq, wq
    jl .tail_loop
.end:
    RET
%endmacro

SCALE8_FUNC 15
SCALE8_FUNC 19

;-----------------------------------------------------------------------------
; 8 taps, eight output pixels per iteration.  The eight bytes of a pixel are
; one movq, so a pair of pixels is one xmm and one pmovzxbw puts them in the
; two lanes of a ymm, in the order the filter rows are in memory: each pmaddwd
; takes 32 contiguous bytes of filter.  The tail is four pixels at 128 bits,
; with the same reads and writes as the ssse3 kernel.
;-----------------------------------------------------------------------------

; two pixels of eight taps, one per lane
%macro LOAD_8TAP_8BIT 2 ; dst, first pixel
    movsxd     pos0q, dword [fltposq+cntq*4+%2*4+0]
    movsxd     pos1q, dword [fltposq+cntq*4+%2*4+4]
    movq        xm%1, [srcq+pos0q]
    movhps      xm%1, [srcq+pos1q]
    pmovzxbw     m%1, xm%1
%endmacro

; SCALE8_8_FUNC intermediate_nbits
%macro SCALE8_8_FUNC 1
cglobal hscale8to%1_8, 6, 8, 8, pos0, dst, w, src, filter, fltpos, pos1, cnt
    movsxd        wq, wd
    xor         cntq, cntq
%if %1 == 19
    mova          m6, [max_19bit_ymm]
    mova          m7, [swizzle_8tap]
%endif
    cmp           wq, 8
    jl .tail_loop
    sub           wq, 8

.loop:
    LOAD_8TAP_8BIT 0, 0
    LOAD_8TAP_8BIT 1, 2
    LOAD_8TAP_8BIT 2, 4
    LOAD_8TAP_8BIT 3, 6
    pmaddwd       m0, [filterq+ 0]
    pmaddwd       m1, [filterq+32]
    pmaddwd       m2, [filterq+64]
    pmaddwd       m3, [filterq+96]
    HADD4D        m0, m1, m2, m3, m4  ; px0 px2 px4 px6 | px1 px3 px5 px7
    psrad         m0, 14 + 8 - %1
%if %1 == 15
    packssdw      m0, m0
    vextracti128 xm4, m0, 1
    punpcklwd    xm0, xm0, xm4            ; the lanes interleave word by word
    movu [dstq+cntq*2], xm0
%else ; %1 == 19
    vpermd        m0, m7, m0
    pminsd        m0, m6
    movu [dstq+cntq*4], m0
%endif ; %1 == 15/19
    add      filterq, 128
    add         cntq, 8
    cmp         cntq, wq
    jle .loop

    add           wq, 8
    cmp         cntq, wq
    jge .end

    ; 1-7 pixels left, or dstW < 8 to start with
.tail_loop:
    HSCALE8_8_TAIL %1, 6
    cmp         cntq, wq
    jl .tail_loop
.end:
    RET
%endmacro

SCALE8_8_FUNC 15
SCALE8_8_FUNC 19

;-----------------------------------------------------------------------------
; An arbitrary filterSize (a multiple of 4), four output pixels in flight.
; Each pixel's taps are one contiguous run of source and one of filter, so a
; 16 tap chunk is a pmovzxbw of 16 bytes and a pmaddwd of 32 bytes of filter,
; twice the taps of the ssse3 inner loop per instruction, with no shuffle
; besides the zero extension.  What is left after the whole chunks is an 8
; tap chunk (filterSize & 8) and, for X4, a 4 tap one.  The 4 tap one loads
; the eight bytes that end at the last tap and masks off the four already
; done, so neither the source nor the filter row is read past its end.
;
; The four accumulators fold with the HADD4D tree, which leaves the two lane
; halves of each pixel next to each other; one more add finishes them.
; Pixels are done in fours up to FFALIGN(dstW, 4), as the ssse3 kernels do.
;-----------------------------------------------------------------------------

; SCALE8_X_FUNC intermediate_nbits, X4/X8
%macro SCALE8_X_FUNC 2
cglobal hscale8to%1_%2, 7, 15, 10, f1, dst, w, src, filter, fltpos, fltsize, s0, s1, s2, s3, f2, f3, tap, lim
    movsxd  fltsizeq, fltsized
    lea         limq, [fltsizeq-16]       ; the last tap a 16 tap chunk may start at
%if %1 == 19
    mova          m9, [max_19bit_ymm]
%endif
%ifidn %2, X4
    mova         xm8, [hi4_words]
%endif

.loop:
    movsxd       s0q, dword [fltposq+ 0]
    movsxd       s1q, dword [fltposq+ 4]
    movsxd       s2q, dword [fltposq+ 8]
    movsxd       s3q, dword [fltposq+12]
    add          s0q, srcq
    add          s1q, srcq
    add          s2q, srcq
    add          s3q, srcq
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
    pmovzxbw      m4, [s0q+tapq]
    pmovzxbw      m5, [s1q+tapq]
    pmovzxbw      m6, [s2q+tapq]
    pmovzxbw      m7, [s3q+tapq]
    pmaddwd       m4, [filterq+tapq*2]
    pmaddwd       m5, [f1q+tapq*2]
    pmaddwd       m6, [f2q+tapq*2]
    pmaddwd       m7, [f3q+tapq*2]
    paddd         m0, m4
    paddd         m1, m5
    paddd         m2, m6
    paddd         m3, m7
    add         tapq, 16
    cmp         tapq, limq
    jle .inner

.rem8:
    test    fltsized, 8
    jz .rem4
    ; the xmm results leave the upper lane zeroed
    pmovzxbw     xm4, [s0q+tapq]
    pmovzxbw     xm5, [s1q+tapq]
    pmovzxbw     xm6, [s2q+tapq]
    pmovzxbw     xm7, [s3q+tapq]
    pmaddwd      xm4, [filterq+tapq*2]
    pmaddwd      xm5, [f1q+tapq*2]
    pmaddwd      xm6, [f2q+tapq*2]
    pmaddwd      xm7, [f3q+tapq*2]
    paddd         m0, m4
    paddd         m1, m5
    paddd         m2, m6
    paddd         m3, m7
    add         tapq, 8
.rem4:
%ifidn %2, X4
    pmovzxbw     xm4, [s0q+tapq-4]
    pmovzxbw     xm5, [s1q+tapq-4]
    pmovzxbw     xm6, [s2q+tapq-4]
    pmovzxbw     xm7, [s3q+tapq-4]
    pand         xm4, xm8
    pand         xm5, xm8
    pand         xm6, xm8
    pand         xm7, xm8
    pmaddwd      xm4, [filterq+tapq*2-8]
    pmaddwd      xm5, [f1q+tapq*2-8]
    pmaddwd      xm6, [f2q+tapq*2-8]
    pmaddwd      xm7, [f3q+tapq*2-8]
    paddd         m0, m4
    paddd         m1, m5
    paddd         m2, m6
    paddd         m3, m7
%endif

    HADD4D        m0, m1, m2, m3, m4  ; px0-3 | px0-3
    vextracti128 xm1, m0, 1
    paddd        xm0, xm1
    psrad        xm0, 14 + 8 - %1
%if %1 == 15
    packssdw     xm0, xm0
    movq      [dstq], xm0
    add         dstq, 8
%else ; %1 == 19
    pminsd       xm0, xm9
    movu      [dstq], xm0
    add         dstq, 16
%endif ; %1 == 15/19
    lea      filterq, [f3q+fltsizeq*2]
    add       fltposq, 16
    sub           wd, 4
    jg .loop
    RET
%endmacro

SCALE8_X_FUNC 15, X4
SCALE8_X_FUNC 15, X8
SCALE8_X_FUNC 19, X4
SCALE8_X_FUNC 19, X8

;-----------------------------------------------------------------------------
; The same with eight output pixels in flight and no remainder steps: a pixel
; is its whole 16 tap chunks and then one chunk of the 4, 8, 12 or 16 taps
; left, loaded with vpmaskmovd, which reads only the dwords its mask selects
; (a tap is a byte of source and a word of filter, so the 4 tap granularity of
; filterSize is a dword granularity of both).  A 12 tap filter is then one
; masked load of source, one of filter, a pmovzxbw and a pmaddwd a pixel.  The
; line is specialised on the whole chunk count (0, 1, or a loop of them), and
; the eight accumulators fold with two HADD4D trees and one lane swap.
;
; Eight pixels while at least five are left, then four, so filterPos[] and
; filter[] are read and dst is written up to FFALIGN(dstW, 4) and no further.
; It wins with 12 taps (the 8 + 4 remainder steps are all the X4 kernel above
; does then); from 16 taps up the four pixel kernel is faster on Zen 5.
;-----------------------------------------------------------------------------

; one pixel: %1 accumulator, %2 pixel, %3 the whole chunks before the masked
; last one at lastoff: 0, 1 (each unrolled), or 2 for nfull of them in a loop.
; xm8/m9 are the source/filter masks of the last chunk, rowq the bytes of a
; filter row, lastq those of the last chunk.  m10 scratch.
%macro X8_PIXEL_AVX2 3
    movsxd      posq, dword [fltposq+%2*4]
    add         posq, srcq
    vpmaskmovd   xm10, xm8, [posq+lastoffq]
    pmovzxbw     m%1, xm10
    vpmaskmovd    m10, m9, [filterq+lastoffq*2]
    pmaddwd      m%1, m10
%if %3 == 1
    pmovzxbw     m10, [posq]
    pmaddwd      m10, [filterq]
    paddd        m%1, m10
%elif %3 == 2
    mov      chunksq, nfullq
%%chunk:
    pmovzxbw     m10, [posq]
    pmaddwd      m10, [filterq]
    paddd        m%1, m10
    add         posq, 16
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

; a whole line: %1 intermediate_nbits, %2 the X8_PIXEL_AVX2 chunk mode
%macro X8_LINE_AVX2 2
    sub           wd, 4
    jle .tail%2
.loop%2:
%assign %%i 0
%rep 8
    X8_PIXEL_AVX2 %%i, %%i, %2
%assign %%i %%i+1
%endrep
    HADD4D        m0, m1, m2, m3, m10     ; px 0-3 | px 0-3
    HADD4D        m4, m5, m6, m7, m10     ; px 4-7 | px 4-7
    vperm2i128   m10, m0, m4, 0x31        ; the high lanes
    vinserti128   m0, m0, xm4, 1          ; the low lanes
    paddd         m0, m10
    psrad         m0, 14 + 8 - %1
%if %1 == 15
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
    jg .loop%2
.tail%2:
    cmp           wd, -4
    jle .end
    ; 1-4 pixels left
%assign %%i 0
%rep 4
    X8_PIXEL_AVX2 %%i, %%i, %2
%assign %%i %%i+1
%endrep
    HADD4D        m0, m1, m2, m3, m10
    vextracti128 xm1, m0, 1
    paddd        xm0, xm1
    psrad        xm0, 14 + 8 - %1
%if %1 == 15
    packssdw     xm0, xm0
    movq      [dstq], xm0
%else
    pminsd       xm0, xm11
    movu      [dstq], xm0
%endif
    jmp .end
%endmacro

; SCALE8_X_FUNC2 intermediate_nbits, X4/X8
%macro SCALE8_X_FUNC2 2
cglobal hscale8to%1_%2_masked, 7, 13, 12, pos, dst, w, src, filter, fltpos, fltsize, chunks, nfull, lastoff, last, tmp, row
    movsxd  fltsizeq, fltsized
%if %1 == 19
    mova         m11, [max_19bit_ymm]
%endif
    lea         rowq, [fltsizeq*2]
    lea       nfullq, [fltsizeq-1]
    shr       nfullq, 4                   ; whole 16 tap chunks before the last
    mov     lastoffq, nfullq
    shl     lastoffq, 4                   ; the source offset of the last chunk
    mov        lastq, fltsizeq
    sub        lastq, lastoffq            ; taps in it: 4, 8, 12 or 16
    lea         tmpq, [mask_dw+32]
    sub         tmpq, lastq
    movu         xm8, [tmpq]              ; taps/4 dwords of source
    sub         tmpq, lastq
    movu          m9, [tmpq]              ; taps/2 dwords of filter
    add        lastq, lastq               ; bytes of filter in it

    cmp       nfulld, 1
    jl .chunks0
    je .chunks1
    X8_LINE_AVX2  %1, 2
.chunks1:
    X8_LINE_AVX2  %1, 1
.chunks0:
    X8_LINE_AVX2  %1, 0
.end:
    RET
%endmacro

SCALE8_X_FUNC2 15, X4
SCALE8_X_FUNC2 15, X8
SCALE8_X_FUNC2 19, X4
SCALE8_X_FUNC2 19, X8
%endif ; HAVE_AVX2_EXTERNAL

;-----------------------------------------------------------------------------
; The same with avx512icl, sixteen output pixels at a time.  Collecting the
; bytes one pixel at a time is what costs, and a byte permute does all sixteen
; at once: the taps of a whole iteration come from a window of source no wider
; than 64 bytes for any scaling ratio that keeps a 4 tap filter, so one zmm
; load and one vpermb replace the whole insert tree.
;
; The window is placed at the END of the taps of the last pixel of the group,
; at filterPos[15] + 4 - 64, so that it reads no further than filterPos[15] + 3.
; Where that start would be negative it moves up, but never past
; filterPos[dstW-1] + 4 - 64: the highest byte any window reads is then
; filterPos[dstW-1] + 3, which is exactly what the ssse3 kernel reads on the
; same line, so the source needs no padding beyond what swscale already relies
; on.  (A line whose last filterPos is under 60 has less than 64 bytes of
; source in total and stays on the tail code.)
;
; The offsets into the window are filterPos[i] - base, and the iteration is
; valid iff all sixteen are within [0, 60]; that is one vpcmpud away, and it
; proves the window covers every tap without assuming filterPos[] is monotonic.
; A group that fails it (a 4 tap filter downscaling by more than ~3.75, which
; initFilter() does not produce) falls back to the 128 bit tail code, which has
; no such constraint.
;-----------------------------------------------------------------------------

%if HAVE_AVX512ICL_EXTERNAL
INIT_ZMM avx512icl

; SCALE8_FUNC_ICL intermediate_nbits
%macro SCALE8_FUNC_ICL 1
cglobal hscale8to%1_4, 6, 11, 12, pos0, dst, w, src, filter, fltpos, pos1, cnt, winbase, hibase, zero
    movsxd        wq, wd
    xor         cntq, cntq
    xor        zerod, zerod
    mov      hibased, dword [fltposq+wq*4-4]  ; filterPos[dstW-1]
    sub      hibased, 60                      ; the last window this line may load
    mova          m6, [spread_offset]
    mova          m7, [tap_offset]
    mova          m8, [even_dwords]
    mova         xm9, [even_dwords_x]
    vpbroadcastd m10, [max_offset]
%if %1 == 19
    vpbroadcastd m11, [max_19bit]
%endif
    cmp           wq, 16
    jl .tail_loop
    test     hibased, hibased
    js .tail_loop                           ; under 64 bytes of source in all
    sub           wq, 16

.loop:
    mov     winbased, dword [fltposq+cntq*4+60] ; filterPos[15]
    sub     winbased, 60
    cmovl   winbased, zerod
    cmp     winbased, hibased
    cmovg   winbased, hibased
    vpbroadcastd  m1, winbased
    movu          m0, [fltposq+cntq*4]
    psubd         m0, m1                    ; filterPos[i] - base
    vpcmpud       k1, m0, m10, 2            ; unsigned, so negatives fail too
    kortestw      k1, k1
    jnc .narrow                             ; a tap lies outside the window

    vpermb        m2, m6, m0                ; offset of the pixel each tap is in
    paddb         m2, m7                    ; + the tap's own 0..3
    movu          m3, [srcq+winbaseq]
    vpermb        m3, m2, m3                ; 16 px * 4 taps of source
    vextracti32x8 ym4, m3, 1
    pmovzxbw      m3, ym3                   ; px0-7
    pmovzxbw      m4, ym4                   ; px8-15
    pmaddwd       m3, [filterq+ 0]
    pmaddwd       m4, [filterq+64]

    psrlq         m5, m3, 32
    paddd         m3, m5
    psrlq         m5, m4, 32
    paddd         m4, m5
    vpermt2d      m3, m8, m4                ; px0 .. px15
    psrad         m3, 14 + 8 - %1
%if %1 == 15
    vpmovsdw     ym3, m3
    movu [dstq+cntq*2], ym3
%else ; %1 == 19
    pminsd        m3, m11
    movu [dstq+cntq*4], m3
%endif ; %1 == 15/19
    add      filterq, 128
    add         cntq, 16
    cmp         cntq, wq
    jle .loop
    jmp .after_loop

.narrow:
    HSCALE8_4_TAIL %1, 11, 9
    cmp         cntq, wq
    jle .loop

.after_loop:
    add           wq, 16
    cmp         cntq, wq
    jge .end

    ; 1-15 pixels left, or dstW < 16 to start with
.tail_loop:
    HSCALE8_4_TAIL %1, 11, 9
    cmp         cntq, wq
    jl .tail_loop
.end:
    RET
%endmacro

SCALE8_FUNC_ICL 15
SCALE8_FUNC_ICL 19

;-----------------------------------------------------------------------------
; 8 taps with avx512icl, sixteen output pixels per iteration, the same way:
; the taps of sixteen pixels of an 8 tap filter come from a window no wider
; than 64 bytes whenever the filter positions advance by at most 3.5 a pixel,
; which every 8 tap filter initFilter() makes does.  The window is placed at
; filterPos[15] + 8 - 64, and never past filterPos[dstW-1] + 8 - 64, so the
; highest byte it reads is the last tap of the line, as for the 4 tap kernel.
;
; Here the permute does the zero extension too: each of the four vpermb
; writes the bytes of four pixels into the even bytes of a zmm and zeroes the
; odd ones by mask, so it yields words directly, with no pmovzxbw.  Group g is
; pixels 4g..4g+3, one per lane, so its filter rows are one contiguous 64 byte
; operand; HADD4D then leaves pixel 4g+l in dword 4l+g, and one vpermd puts
; the sixteen in order.
;-----------------------------------------------------------------------------

; SCALE8_8_FUNC_ICL intermediate_nbits
%macro SCALE8_8_FUNC_ICL 1
cglobal hscale8to%1_8, 6, 11, 14, pos0, dst, w, src, filter, fltpos, pos1, cnt, winbase, hibase, zero
    movsxd        wq, wd
    xor         cntq, cntq
    xor        zerod, zerod
    mov      hibased, dword [fltposq+wq*4-4]  ; filterPos[dstW-1]
    sub      hibased, 56                      ; the last window this line may load
    mova          m6, [spread8_g0]
    mova          m7, [spread8_g1]
    mova          m8, [spread8_g2]
    mova          m9, [spread8_g3]
    mova         m10, [tap8_offset]
    mova         m11, [transpose16]
    vpbroadcastd m12, [max_offset8]
%if %1 == 19
    vpbroadcastd m13, [max_19bit]
%endif
    mov        pos0q, 0x5555555555555555
    kmovq         k2, pos0q                 ; the even bytes
    cmp           wq, 16
    jl .tail_loop
    test     hibased, hibased
    js .tail_loop                           ; under 64 bytes of source in all
    sub           wq, 16

.loop:
    mov     winbased, dword [fltposq+cntq*4+60] ; filterPos[15]
    sub     winbased, 56
    cmovl   winbased, zerod
    cmp     winbased, hibased
    cmovg   winbased, hibased
    vpbroadcastd  m1, winbased
    movu          m0, [fltposq+cntq*4]
    psubd         m0, m1                    ; filterPos[i] - base
    vpcmpud       k1, m0, m12, 2            ; unsigned, so negatives fail too
    kortestw      k1, k1
    jnc .narrow                             ; a tap lies outside the window

    movu          m5, [srcq+winbaseq]       ; the 64 byte window
    vpermb        m1, m6, m0
    vpermb        m2, m7, m0
    vpermb        m3, m8, m0
    vpermb        m4, m9, m0
    paddb         m1, m10                   ; + the tap's own 0..7
    paddb         m2, m10
    paddb         m3, m10
    paddb         m4, m10
    vpermb        m1{k2}{z}, m1, m5         ; px 0-3, 8 taps each, as words
    vpermb        m2{k2}{z}, m2, m5         ; px 4-7
    vpermb        m3{k2}{z}, m3, m5         ; px 8-11
    vpermb        m4{k2}{z}, m4, m5         ; px 12-15
    pmaddwd       m1, [filterq+  0]
    pmaddwd       m2, [filterq+ 64]
    pmaddwd       m3, [filterq+128]
    pmaddwd       m4, [filterq+192]
    HADD4D        m1, m2, m3, m4, m0
    vpermd        m1, m11, m1               ; px0 .. px15
    psrad         m1, 14 + 8 - %1
%if %1 == 15
    vpmovsdw     ym1, m1
    movu [dstq+cntq*2], ym1
%else ; %1 == 19
    pminsd        m1, m13
    movu [dstq+cntq*4], m1
%endif ; %1 == 15/19
    add      filterq, 256
    add         cntq, 16
    cmp         cntq, wq
    jle .loop
    jmp .after_loop

.narrow:
    HSCALE8_8_TAIL %1, 13
    cmp         cntq, wq
    jle .loop

.after_loop:
    add           wq, 16
    cmp         cntq, wq
    jge .end

    ; 1-15 pixels left, or dstW < 16 to start with
.tail_loop:
    HSCALE8_8_TAIL %1, 13
    cmp         cntq, wq
    jl .tail_loop
.end:
    RET
%endmacro

SCALE8_8_FUNC_ICL 15
SCALE8_8_FUNC_ICL 19

;-----------------------------------------------------------------------------
; An arbitrary filterSize with avx512icl, eight output pixels in flight, one
; ymm each.  A pixel is its whole 16 tap chunks and then one masked chunk of
; the 1-16 taps left, so the 12 and 16 tap lanczos SD <-> HD filters are one
; masked load of source, one of filter, one pmovzxbw and one pmaddwd a pixel,
; with none of the 8 and 4 tap remainder steps the avx2 kernel takes, and
; nothing past the row is read, of either.
;
; Wider registers do not pay here: a zmm per pixel halves the multiplies of
; the 20 and 32 tap thumbnail filters, but then has twice the dwords to fold,
; and measured slower than this on Zen 5.
;
; Eight pixels while at least five are left, then four, so that, as for the
; other kernels, filterPos[] and filter[] are read and dst is written up to
; FFALIGN(dstW, 4) and no further.
;-----------------------------------------------------------------------------

; one pixel: %1 accumulator, %2 pixel, %3 the whole chunks before the masked
; last one at lastoff: 0, 1 (each unrolled), or 2 for nfull of them in a loop.
; rowq is the bytes of a filter row, lastq those of the last chunk.
%macro X8_PIXEL_ICL 3
    movsxd      posq, dword [fltposq+%2*4]
    add         posq, srcq
    vmovdqu8    xm14{k1}{z}, [posq+lastoffq]
    vmovdqu16   ym15{k1}{z}, [filterq+lastoffq*2]
    pmovzxbw    ym%1, xm14
    pmaddwd     ym%1, ym15
%if %3 == 1
    pmovzxbw    ym14, [posq]
    vpdpwssd    ym%1, ym14, [filterq]
%elif %3 == 2
    mov      chunksq, nfullq
%%chunk:
    pmovzxbw    ym14, [posq]
    vpdpwssd    ym%1, ym14, [filterq]
    add         posq, 16
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

; m0-m7, one pixel each -> ym0 = px 0-7.  m8 scratch.
%macro REDUCE8 0
    HADD4D       ym0, ym1, ym2, ym3, ym8  ; px 0-3 | px 0-3
    HADD4D       ym4, ym5, ym6, ym7, ym8  ; px 4-7 | px 4-7
    vshufi64x2   ym8, ym0, ym4, 3         ; the high lanes
    vshufi64x2   ym0, ym0, ym4, 0         ; the low lanes
    paddd        ym0, ym8
%endmacro

; m0-m3 -> xm0 = px 0-3.  m8 scratch.
%macro REDUCE4 0
    HADD4D       ym0, ym1, ym2, ym3, ym8
    vextracti32x4 xm8, ym0, 1
    paddd        xm0, xm8
%endmacro

; %1 intermediate_nbits, %2 pixels (8 or 4)
%macro X8_STORE 2
%if %2 == 8
    psrad        ym0, 14 + 8 - %1
%if %1 == 15
    vpmovsdw     xm0, ym0
    movu      [dstq], xm0
    add         dstq, 16
%else
    pminsd       ym0, ym11
    movu      [dstq], ym0
    add         dstq, 32
%endif
%else
    psrad        xm0, 14 + 8 - %1
%if %1 == 15
    packssdw     xm0, xm0
    movq      [dstq], xm0
%else
    pminsd       xm0, xm11
    movu      [dstq], xm0
%endif
%endif
%endmacro

; a whole line: %1 intermediate_nbits, %2 the X8_PIXEL_ICL chunk mode
%macro X8_LINE_ICL 2
    sub           wd, 4
    jle .tail%2
.loop%2:
%assign %%i 0
%rep 8
    X8_PIXEL_ICL  %%i, %%i, %2
%assign %%i %%i+1
%endrep
    REDUCE8
    X8_STORE      %1, 8
    add       fltposq, 32
    sub           wd, 8
    jg .loop%2
.tail%2:
    cmp           wd, -4
    jle .end
    ; 1-4 pixels left
%assign %%i 0
%rep 4
    X8_PIXEL_ICL  %%i, %%i, %2
%assign %%i %%i+1
%endrep
    REDUCE4
    X8_STORE      %1, 4
    jmp .end
%endmacro

; SCALE8_X_FUNC_ICL intermediate_nbits, X4/X8
%macro SCALE8_X_FUNC_ICL 2
cglobal hscale8to%1_%2, 7, 13, 16, pos, dst, w, src, filter, fltpos, fltsize, chunks, nfull, lastoff, last, tmp, row
    movsxd  fltsizeq, fltsized
%if %1 == 19
    vpbroadcastd m11, [max_19bit]
%endif
    lea         rowq, [fltsizeq*2]
    lea       nfullq, [fltsizeq-1]
    shr       nfullq, 4                   ; whole 16 tap chunks before the last
    mov     lastoffq, nfullq
    shl     lastoffq, 4                   ; the source offset of the last chunk
    mov        lastq, fltsizeq
    sub        lastq, lastoffq            ; taps in it, 1-16
    mov         tmpd, -1
    bzhi        tmpd, tmpd, lastd
    kmovd         k1, tmpd
    add        lastq, lastq               ; bytes of filter in it

    cmp       nfulld, 1
    jl .chunks0
    je .chunks1
    X8_LINE_ICL   %1, 2
.chunks1:
    X8_LINE_ICL   %1, 1
.chunks0:
    X8_LINE_ICL   %1, 0
.end:
    RET
%endmacro

SCALE8_X_FUNC_ICL 15, X4
SCALE8_X_FUNC_ICL 15, X8
SCALE8_X_FUNC_ICL 19, X4
SCALE8_X_FUNC_ICL 19, X8
%endif ; HAVE_AVX512ICL_EXTERNAL

%endif ; ARCH_X86_64
