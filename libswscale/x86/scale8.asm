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
; the three 64 byte constants above keep this one 32 byte aligned for mova
max_19bit_ymm: times 8 dd 0x7ffff
even_dwords_x: dd 0, 2, 4, 6
max_offset:    dd 60
max_19bit:     dd 0x7ffff

SECTION .text

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
    pinsrd      xm%1, [srcq+pos1q], 1
    movsxd     pos0q, dword [fltposq+cntq*4+%2*4+ 8]
    movsxd     pos1q, dword [fltposq+cntq*4+%2*4+12]
    pinsrd      xm%1, [srcq+pos0q], 2
    pinsrd      xm%1, [srcq+pos1q], 3
    pmovzxbw     m%1, xm%1
%endmacro

; the 128 bit tail, four output pixels; also the fallback of the avx512 loop.
; phaddd has no EVEX form, so under avx512 the pairs fold with a shift and the
; sums are collected by the same permute the wide loop uses.
%macro HSCALE8_4_TAIL 3 ; intermediate_nbits, max_19bit reg, even_dwords reg
    movsxd     pos0q, dword [fltposq+cntq*4+ 0]
    movsxd     pos1q, dword [fltposq+cntq*4+ 4]
    movd         xm0, [srcq+pos0q]
    pinsrd       xm0, [srcq+pos1q], 1
    movsxd     pos0q, dword [fltposq+cntq*4+ 8]
    movsxd     pos1q, dword [fltposq+cntq*4+12]
    movd         xm1, [srcq+pos0q]
    pinsrd       xm1, [srcq+pos1q], 1
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
%endif ; HAVE_AVX512ICL_EXTERNAL

%endif ; ARCH_X86_64
