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
; Same contract as scale.asm, eight output pixels per iteration.  The taps of
; neighbouring output pixels start at unrelated source positions, so a ymm of
; source has to be built from two 128 bit loads; the filter rows on the other
; hand are contiguous, and eight taps of two pixels (or four taps of four) are
; exactly the 32 bytes one ymm operand wants.  That halves the pmaddwd and
; phaddd count for the same number of loads, and no gather is needed.
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

; SCALE_FUNCS16 source_width
%macro SCALE_FUNCS16 1
SCALE_FUNC16 %1, 15, 4
SCALE_FUNC16 %1, 15, 8
SCALE_FUNC16 %1, 19, 4
SCALE_FUNC16 %1, 19, 8
%endmacro

INIT_YMM avx2
SCALE_FUNCS16  9
SCALE_FUNCS16 10
SCALE_FUNCS16 12
SCALE_FUNCS16 14
SCALE_FUNCS16 16

%endif ; ARCH_X86_64 && HAVE_AVX2_EXTERNAL
