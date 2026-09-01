;******************************************************************************
;* AVX-512ICL horizontal line scaling functions for 9 to 16 bit input
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

%if ARCH_X86_64 && HAVE_AVX512ICL_EXTERNAL

SECTION_RODATA 64

; the offset of output pixel i is the word at 2i of the filterPos vector, since
; the offsets are all under 64; vpermw spreads it over that pixel's four taps
spread_lo:     dw  0,  0,  0,  0,  2,  2,  2,  2,  4,  4,  4,  4,  6,  6,  6,  6
               dw  8,  8,  8,  8, 10, 10, 10, 10, 12, 12, 12, 12, 14, 14, 14, 14
spread_hi:     dw 16, 16, 16, 16, 18, 18, 18, 18, 20, 20, 20, 20, 22, 22, 22, 22
               dw 24, 24, 24, 24, 26, 26, 26, 26, 28, 28, 28, 28, 30, 30, 30, 30
tap_offset:    times 8 dw 0, 1, 2, 3
; pmaddwd leaves the two halves of a pixel in a pair of dwords, so the sums are
; the even ones of the two accumulators
even_dwords:   dd 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
minshort_zmm:  times 32 dw 0x8000
unicoeff_zmm:  times 16 dd 0x20000000
max_19bit_zmm: times 16 dd 0x7ffff
even_dwords_x: dd 0, 2, 4, 6
max_offset:    dd 60

SECTION .text

;-----------------------------------------------------------------------------
; horizontal line scaling, 16 bit wide input, 4 tap filter
;
; void hscale<source_width>to<intermediate_nbits>_4_avx512icl
;                               (SwsContext *c, int{16,32}_t *dst,
;                                int dstW, const uint16_t *src,
;                                const int16_t *filter,
;                                const int32_t *filterPos, int filterSize);
;
; Same contract as scale.asm, and the same shape as the 8 bit avx512 kernel in
; scale8.asm: the taps of sixteen output pixels come from a window of source no
; wider than 64 samples for any ratio that still leaves a 4 tap filter, so two
; zmm loads and two vpermi2w collect all sixty four taps, in place of the eight
; movq/movhps/vinserti128 groups the avx2 kernel needs for the same pixels.
;
; The window is placed at the end of the taps of the last pixel of the group,
; at filterPos[15]+4-64, and never moved past filterPos[dstW-1]+4-64, so the
; highest sample it reads is filterPos[dstW-1]+3 -- exactly what the existing
; 4 tap kernels read on the same line, and no new source padding is assumed.
; A vpcmpud of the sixteen offsets then proves every tap is inside the window
; before the permute runs, without assuming filterPos[] is monotonic; a group
; that fails it falls back to the 128 bit tail code, which has no constraint.
;-----------------------------------------------------------------------------

; the 128 bit tail, four output pixels.  phaddd has no EVEX form, so the pairs
; fold with a shift and the sums are collected by the same permute the wide
; loop uses.  m12/m13/m14 hold minshort/unicoeff/max_19bit, m10 the permute.
%macro HSCALE16_4_TAIL 2 ; source_width, intermediate_nbits
    movsxd     pos0q, dword [fltposq+cntq*4+ 0]
    movsxd     pos1q, dword [fltposq+cntq*4+ 4]
    movq         xm0, [srcq+pos0q*2]
    movq         xm2, [srcq+pos1q*2]
    punpcklqdq   xm0, xm0, xm2
    movsxd     pos0q, dword [fltposq+cntq*4+ 8]
    movsxd     pos1q, dword [fltposq+cntq*4+12]
    movq         xm1, [srcq+pos0q*2]
    movq         xm2, [srcq+pos1q*2]
    punpcklqdq   xm1, xm1, xm2
%if %1 == 16
    psubw        xm0, xm12
    psubw        xm1, xm12
%endif
    pmaddwd      xm0, [filterq+ 0]
    pmaddwd      xm1, [filterq+16]
    psrlq        xm2, xm0, 32
    paddd        xm0, xm2
    psrlq        xm2, xm1, 32
    paddd        xm1, xm2
    vpermt2d     xm0, xm10, xm1           ; px0 px1 px2 px3
%if %1 == 16
    paddd        xm0, xm13
%endif
    psrad        xm0, 14 + %1 - %2
%if %2 == 15
    packssdw     xm0, xm0
    movq [dstq+cntq*2], xm0
%else
    pminsd       xm0, xm14
    movu [dstq+cntq*4], xm0
%endif
    add      filterq, 32
    add         cntq, 4
%endmacro

INIT_ZMM avx512icl

; SCALE_FUNC16_ICL source_width, intermediate_nbits
%macro SCALE_FUNC16_ICL 2
cglobal hscale%1to%2_4, 6, 11, 16, pos0, dst, w, src, filter, fltpos, pos1, cnt, winbase, hibase, zero
    movsxd        wq, wd
    xor         cntq, cntq
    xor        zerod, zerod
    mov      hibased, dword [fltposq+wq*4-4]   ; filterPos[dstW-1]
    sub      hibased, 60                       ; the last window this line may load
    mova          m6, [spread_lo]
    mova          m7, [spread_hi]
    mova          m8, [tap_offset]
    mova          m9, [even_dwords]
    mova        xm10, [even_dwords_x]
    vpbroadcastd m11, [max_offset]
%if %1 == 16
    mova         m12, [minshort_zmm]
    mova         m13, [unicoeff_zmm]
%endif
%if %2 == 19
    mova         m14, [max_19bit_zmm]
%endif
    cmp           wq, 16
    jl .tail_loop
    test     hibased, hibased
    js .tail_loop                            ; under 64 samples of source in all
    sub           wq, 16

.loop:
    mov     winbased, dword [fltposq+cntq*4+60] ; filterPos[15]
    sub     winbased, 60
    cmovl   winbased, zerod
    cmp     winbased, hibased
    cmovg   winbased, hibased
    vpbroadcastd  m1, winbased
    movu          m0, [fltposq+cntq*4]
    psubd         m0, m1                     ; filterPos[i] - base
    vpcmpud       k1, m0, m11, 2             ; unsigned, so negatives fail too
    kortestw      k1, k1
    jnc .narrow                              ; a tap lies outside the window

    vpermw        m2, m6, m0                 ; offset of the pixel each tap is in
    vpermw        m3, m7, m0
    paddw         m2, m8                     ; + the tap's own 0..3
    paddw         m3, m8
    movu          m4, [srcq+winbaseq*2]      ; the 64 sample window
    movu          m5, [srcq+winbaseq*2+64]
    vpermi2w      m2, m4, m5                 ; px0-7,  4 taps each
    vpermi2w      m3, m4, m5                 ; px8-15, 4 taps each
%if %1 == 16 ; pmaddwd needs signed adds, so this moves unsigned -> signed,
             ; the 0x8000 * sum(coeffs) comes back after the horizontal add
    psubw         m2, m12
    psubw         m3, m12
%endif
    pmaddwd       m2, [filterq+ 0]
    pmaddwd       m3, [filterq+64]

    psrlq         m0, m2, 32
    paddd         m2, m0
    psrlq         m1, m3, 32
    paddd         m3, m1
    vpermt2d      m2, m9, m3                 ; px0 .. px15
%if %1 == 16
    paddd         m2, m13
%endif
    psrad         m2, 14 + %1 - %2
%if %2 == 15
    vpmovsdw     ym2, m2
    movu [dstq+cntq*2], ym2
%else ; %2 == 19
    pminsd        m2, m14
    movu [dstq+cntq*4], m2
%endif ; %2 == 15/19
    add      filterq, 128
    add         cntq, 16
    cmp         cntq, wq
    jle .loop
    jmp .after_loop

.narrow:
    HSCALE16_4_TAIL %1, %2
    cmp         cntq, wq
    jle .loop

.after_loop:
    add           wq, 16
    cmp         cntq, wq
    jge .end

    ; 1-15 pixels left, or dstW < 16 to start with
.tail_loop:
    HSCALE16_4_TAIL %1, %2
    cmp         cntq, wq
    jl .tail_loop
.end:
    RET
%endmacro

; SCALE_FUNCS16_ICL source_width
%macro SCALE_FUNCS16_ICL 1
SCALE_FUNC16_ICL %1, 15
SCALE_FUNC16_ICL %1, 19
%endmacro

SCALE_FUNCS16_ICL  9
SCALE_FUNCS16_ICL 10
SCALE_FUNCS16_ICL 12
SCALE_FUNCS16_ICL 14
SCALE_FUNCS16_ICL 16

%endif ; ARCH_X86_64 && HAVE_AVX512ICL_EXTERNAL
