;******************************************************************************
;* x86-optimized vertical line scaling functions
;* Copyright (c) 2011 Ronald S. Bultje <rsbultje@gmail.com>
;*                    Kieran Kunhya <kieran@kunhya.com>
;*           (c) 2020 Nelson Gomez <nelson.gomez@microsoft.com>
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

SECTION_RODATA 32

minshort:      times 8 dw 0x8000
yuv2yuvX_16_start:  times 4 dd 0x4000 - 0x40000000
yuv2yuvX_10_start:  times 4 dd 0x10000
yuv2yuvX_9_start:   times 4 dd 0x20000
yuv2yuvX_10_upper:  times 8 dw 0x3ff
yuv2yuvX_9_upper:   times 8 dw 0x1ff
pd_4:          times 4 dd 4
pd_4min0x40000:times 4 dd 4 - (0x40000)
pw_16:         times 8 dw 16
pw_32:         times 8 dw 32
pd_255:        times 8 dd 255
pw_512:        times 8 dw 512
pw_1024:       times 8 dw 1024
; the packs of the 512 bit 8 bit output leave the qwords of 0-31 and 32-63
; interleaved per lane
pack_order:    dq 0, 2, 4, 6, 1, 3, 5, 7

yuv2nv12_shuffle_mask: times 2 db 0,  4,  8, 12, \
                                 -1, -1, -1, -1, \
                                 -1, -1, -1, -1, \
                                 -1, -1, -1, -1
yuv2nv21_shuffle_mask: times 2 db 4,  0, 12,  8, \
                                 -1, -1, -1, -1, \
                                 -1, -1, -1, -1, \
                                 -1, -1, -1, -1
yuv2nv12_permute_mask: dd 0, 4, 1, 2, 3, 5, 6, 7

SECTION .text

;-----------------------------------------------------------------------------
; vertical line scaling
;
; void yuv2plane1_<output_size>_<opt>(const int16_t *src, uint8_t *dst, int dstW,
;                                     const uint8_t *dither, int offset)
; and
; void yuv2planeX_<output_size>_<opt>(const int16_t *filter, int filterSize,
;                                     const int16_t **src, uint8_t *dst, int dstW,
;                                     const uint8_t *dither, int offset)
;
; Scale one or $filterSize lines of source data to generate one line of output
; data. The input is 15-bit in int16_t if $output_size is [8,10] and 19-bit in
; int32_t if $output_size is 16. $filter is 12-bits. $filterSize is a multiple
; of 2. $offset is either 0 or 3. $dither holds 8 values.
;-----------------------------------------------------------------------------

%macro yuv2planeX_fn 3

%if ARCH_X86_32
%define cntr_reg fltsizeq
%define movsx mov
%else
%define cntr_reg r7
%define movsx movsxd
%endif

cglobal yuv2planeX_%1, %3, 8, %2, filter, fltsize, src, dst, w, dither, offset
%if %1 == 8 || %1 == 9 || %1 == 10
    pxor            m6,  m6
%endif ; %1 == 8/9/10

%if %1 == 8
%if ARCH_X86_32
%assign pad 0x2c - (stack_offset & 15)
    SUB             rsp, pad
%define m_dith m7
%else ; x86-64
%define m_dith m9
%endif ; x86-32

    ; create registers holding dither
    movq        m_dith, [ditherq]        ; dither
    test        offsetd, offsetd
    jz              .no_rot
%if mmsize == 16
    punpcklqdq  m_dith,  m_dith
%endif ; mmsize == 16
    PALIGNR     m_dith,  m_dith,  3,  m0
.no_rot:
%if mmsize == 16
    punpcklbw   m_dith,  m6
%if ARCH_X86_64
    punpcklwd       m8,  m_dith,  m6
    pslld           m8,  12
%else ; x86-32
    punpcklwd       m5,  m_dith,  m6
    pslld           m5,  12
%endif ; x86-32/64
    punpckhwd   m_dith,  m6
    pslld       m_dith,  12
%if ARCH_X86_32
    mova      [rsp+ 0],  m5
    mova      [rsp+16],  m_dith
%endif
%else ; mmsize == 8
    punpcklbw       m5,  m_dith,  m6
    punpckhbw   m_dith,  m6
    punpcklwd       m4,  m5,  m6
    punpckhwd       m5,  m6
    punpcklwd       m3,  m_dith,  m6
    punpckhwd   m_dith,  m6
    pslld           m4,  12
    pslld           m5,  12
    pslld           m3,  12
    pslld       m_dith,  12
    mova      [rsp+ 0],  m4
    mova      [rsp+ 8],  m5
    mova      [rsp+16],  m3
    mova      [rsp+24],  m_dith
%endif ; mmsize == 8/16
%endif ; %1 == 8

    xor             r5,  r5

.pixelloop:
%assign %%i 0
    ; the rep here is for the 8bit output mmx case, where dither covers
    ; 8 pixels but we can only handle 2 pixels per register, and thus 4
    ; pixels per iteration. In order to not have to keep track of where
    ; we are w.r.t. dithering, we unroll the mmx/8bit loop x2.
%if %1 == 8
%assign %%repcnt 16/mmsize
%else
%assign %%repcnt 1
%endif

%rep %%repcnt

%if %1 == 8
%if ARCH_X86_32
    mova            m2, [rsp+mmsize*(0+%%i)]
    mova            m1, [rsp+mmsize*(1+%%i)]
%else ; x86-64
    mova            m2,  m8
    mova            m1,  m_dith
%endif ; x86-32/64
%else ; %1 == 9/10/16
    mova            m1, [yuv2yuvX_%1_start]
    mova            m2,  m1
%endif ; %1 == 8/9/10/16
    movsx     cntr_reg,  fltsizem
.filterloop_ %+ %%i:
    ; input pixels
    mov             r6, [srcq+gprsize*cntr_reg-2*gprsize]
%if %1 == 16
    mova            m3, [r6+r5*4]
    mova            m5, [r6+r5*4+mmsize]
%else ; %1 == 8/9/10
    mova            m3, [r6+r5*2]
%endif ; %1 == 8/9/10/16
    mov             r6, [srcq+gprsize*cntr_reg-gprsize]
%if %1 == 16
    mova            m4, [r6+r5*4]
    mova            m6, [r6+r5*4+mmsize]
%else ; %1 == 8/9/10
    mova            m4, [r6+r5*2]
%endif ; %1 == 8/9/10/16

    ; coefficients
    movd            m0, [filterq+2*cntr_reg-4] ; coeff[0], coeff[1]
%if %1 == 16
    pshuflw         m7,  m0,  0          ; coeff[0]
    pshuflw         m0,  m0,  0x55       ; coeff[1]
    pmovsxwd        m7,  m7              ; word -> dword
    pmovsxwd        m0,  m0              ; word -> dword

    pmulld          m3,  m7
    pmulld          m5,  m7
    pmulld          m4,  m0
    pmulld          m6,  m0

    paddd           m2,  m3
    paddd           m1,  m5
    paddd           m2,  m4
    paddd           m1,  m6
%else ; %1 == 10/9/8
    punpcklwd       m5,  m3,  m4
    punpckhwd       m3,  m4
    SPLATD          m0

    pmaddwd         m5,  m0
    pmaddwd         m3,  m0

    paddd           m2,  m5
    paddd           m1,  m3
%endif ; %1 == 8/9/10/16

    sub       cntr_reg,  2
    jg .filterloop_ %+ %%i

%if %1 == 16
    psrad           m2,  31 - %1
    psrad           m1,  31 - %1
%else ; %1 == 10/9/8
    psrad           m2,  27 - %1
    psrad           m1,  27 - %1
%endif ; %1 == 8/9/10/16

%if %1 == 8
    packssdw        m2,  m1
    packuswb        m2,  m2
    movh   [dstq+r5*1],  m2
%else ; %1 == 9/10/16
%if %1 == 16
    packssdw        m2,  m1
    paddw           m2, [minshort]
%else ; %1 == 9/10
%if cpuflag(sse4)
    packusdw        m2,  m1
%else ; mmxext/sse2
    packssdw        m2,  m1
    pmaxsw          m2,  m6
%endif ; mmxext/sse2/sse4/avx
    pminsw          m2, [yuv2yuvX_%1_upper]
%endif ; %1 == 9/10/16
    mova   [dstq+r5*2],  m2
%endif ; %1 == 8/9/10/16

    add             r5,  mmsize/2
    sub             wd,  mmsize/2

%assign %%i %%i+2
%endrep
    jg .pixelloop

%if %1 == 8
%if ARCH_X86_32
    ADD             rsp, pad
    RET
%else ; x86-64
    REP_RET
%endif ; x86-32/64
%else ; %1 == 9/10/16
    REP_RET
%endif ; %1 == 8/9/10/16
%endmacro

%if ARCH_X86_32
INIT_MMX mmxext
yuv2planeX_fn  8,  0, 7
yuv2planeX_fn  9,  0, 5
yuv2planeX_fn 10,  0, 5
%endif

INIT_XMM sse2
yuv2planeX_fn  8, 10, 7
yuv2planeX_fn  9,  7, 5
yuv2planeX_fn 10,  7, 5

INIT_XMM sse4
yuv2planeX_fn  8, 10, 7
yuv2planeX_fn  9,  7, 5
yuv2planeX_fn 10,  7, 5
yuv2planeX_fn 16,  8, 5

%if HAVE_AVX_EXTERNAL
INIT_XMM avx
yuv2planeX_fn  8, 10, 7
yuv2planeX_fn  9,  7, 5
yuv2planeX_fn 10,  7, 5
%endif

%if ARCH_X86_64 && HAVE_AVX2_EXTERNAL
INIT_YMM avx2
; 16 pixels per iteration. The lane split means the accumulators hold pixels
; 0-3/8-11 (m2) and 4-7/12-15 (m1), which packssdw puts back in order; only the
; final packuswb needs a lane fixup. The dither has period 8, so it survives
; both the broadcast and the in-lane PALIGNR.
cglobal yuv2planeX_8, 7, 8, 10, filter, fltsize, src, dst, w, dither, offset
    pxor            m6,  m6

    vpbroadcastq    m9, [ditherq]
    test        offsetd, offsetd
    jz .no_rot
    PALIGNR         m9,  m9,  3,  m0
.no_rot:
    punpcklbw       m9,  m6                 ; px 0-7 | px 8-15
    punpcklwd       m8,  m9,  m6            ; px 0-3 | px 8-11
    punpckhwd       m9,  m6                 ; px 4-7 | px 12-15
    pslld           m8,  12
    pslld           m9,  12

    xor             r5,  r5
    cmp             wd,  16
    jl .tail

.pixelloop:
    mova            m2,  m8
    mova            m1,  m9
    movsxd          r7,  fltsizem
.filterloop:
    mov             r6, [srcq+gprsize*r7-2*gprsize]
    movu            m3, [r6+r5*2]
    mov             r6, [srcq+gprsize*r7-gprsize]
    movu            m4, [r6+r5*2]

    vpbroadcastd    m0, [filterq+2*r7-4]    ; coeff[0], coeff[1]
    punpcklwd       m5,  m3,  m4
    punpckhwd       m3,  m4

    pmaddwd         m5,  m0
    pmaddwd         m3,  m0

    paddd           m2,  m5
    paddd           m1,  m3

    sub             r7,  2
    jg .filterloop

    psrad           m2,  19
    psrad           m1,  19
    packssdw        m2,  m1                 ; px 0-7 | px 8-15
    packuswb        m2,  m2
    vpermq          m2,  m2,  q3120
    movu   [dstq+r5*1], xm2

    add             r5,  16
    sub             wd,  16
    cmp             wd,  16
    jge .pixelloop

    test            wd,  wd
    jle .end

    ; 1-15 pixels left; r5 is a multiple of 16 so the dither phase is unchanged
    ; and the low lanes of m8/m9 are still the right coefficients.
.tail:
    mova           xm2,  xm8
    mova           xm1,  xm9
    movsxd          r7,  fltsizem
.tailfilterloop:
    mov             r6, [srcq+gprsize*r7-2*gprsize]
    movu           xm3, [r6+r5*2]
    mov             r6, [srcq+gprsize*r7-gprsize]
    movu           xm4, [r6+r5*2]

    vpbroadcastd   xm0, [filterq+2*r7-4]
    punpcklwd      xm5,  xm3,  xm4
    punpckhwd      xm3,  xm4

    pmaddwd        xm5,  xm0
    pmaddwd        xm3,  xm0

    paddd          xm2,  xm5
    paddd          xm1,  xm3

    sub             r7,  2
    jg .tailfilterloop

    psrad          xm2,  19
    psrad          xm1,  19
    packssdw       xm2,  xm1
    packuswb       xm2,  xm2
    movq   [dstq+r5*1], xm2

    add             r5,  8
    sub             wd,  8
    jg .tail
.end:
    RET

; 32 pixels per iteration.  The lane split leaves the accumulators holding
; pixels 0-3|8-11 and 4-7|12-15, which is what packusdw wants: it puts them
; back in order, so unlike the 8 bit case there is no lane fixup before the
; store.  Two accumulator pairs are worth it because the source pointers and
; the coefficient broadcast are then amortised over 32 pixels rather than 16 -
; the inner loop is otherwise short of the load ports.  A third pair is not:
; the unpacks already saturate the shuffle port at two pairs.
;
; The rounding offset and the upper clip come out of the existing xmm
; constants with a broadcast, so nothing here needs 32 byte rodata.
%macro yuv2planeX_avx2_fn 1
cglobal yuv2planeX_%1, 5, 8, 12, filter, fltsize, src, dst, w
    vpbroadcastd   m10, [yuv2yuvX_%1_start]
    vpbroadcastw   m11, [yuv2yuvX_%1_upper]

    xor             r5,  r5
    cmp             wd,  32
    jl .check16

.pixelloop:
    mova            m1,  m10
    mova            m2,  m10
    mova            m3,  m10
    mova            m4,  m10
    movsxd          r7,  fltsizem
.filterloop:
    mov             r6, [srcq+gprsize*r7-2*gprsize]
    movu            m5, [r6+r5*2]
    movu            m6, [r6+r5*2+mmsize]
    mov             r6, [srcq+gprsize*r7-gprsize]
    movu            m7, [r6+r5*2]
    movu            m8, [r6+r5*2+mmsize]
    vpbroadcastd    m0, [filterq+2*r7-4]    ; coeff[0], coeff[1]

    punpcklwd       m9,  m5,  m7
    punpckhwd       m5,  m7
    pmaddwd         m9,  m0
    pmaddwd         m5,  m0
    paddd           m1,  m9
    paddd           m2,  m5

    punpcklwd       m9,  m6,  m8
    punpckhwd       m6,  m8
    pmaddwd         m9,  m0
    pmaddwd         m6,  m0
    paddd           m3,  m9
    paddd           m4,  m6

    sub             r7,  2
    jg .filterloop

    psrad           m1,  27 - %1
    psrad           m2,  27 - %1
    psrad           m3,  27 - %1
    psrad           m4,  27 - %1
    packusdw        m1,  m2                 ; px 0-15
    packusdw        m3,  m4                 ; px 16-31
    pminsw          m1,  m11
    pminsw          m3,  m11
    movu   [dstq+r5*2       ],  m1
    movu   [dstq+r5*2+mmsize],  m3

    add             r5,  32
    sub             wd,  32
    cmp             wd,  32
    jge .pixelloop

.check16:
    cmp             wd,  16
    jl .check8
    ; wd < 32 here, so this runs at most once
    mova            m1,  m10
    mova            m2,  m10
    movsxd          r7,  fltsizem
.filterloop16:
    mov             r6, [srcq+gprsize*r7-2*gprsize]
    movu            m5, [r6+r5*2]
    mov             r6, [srcq+gprsize*r7-gprsize]
    movu            m7, [r6+r5*2]

    vpbroadcastd    m0, [filterq+2*r7-4]
    punpcklwd       m9,  m5,  m7
    punpckhwd       m5,  m7
    pmaddwd         m9,  m0
    pmaddwd         m5,  m0
    paddd           m1,  m9
    paddd           m2,  m5

    sub             r7,  2
    jg .filterloop16

    psrad           m1,  27 - %1
    psrad           m2,  27 - %1
    packusdw        m1,  m2
    pminsw          m1,  m11
    movu   [dstq+r5*2],  m1

    add             r5,  16
    sub             wd,  16

.check8:
    test            wd,  wd
    jle .end

    ; 1-15 pixels left, in units of 8 as the xmm functions write them
.tail:
    mova           xm1,  xm10
    mova           xm2,  xm10
    movsxd          r7,  fltsizem
.tailfilterloop:
    mov             r6, [srcq+gprsize*r7-2*gprsize]
    movu           xm5, [r6+r5*2]
    mov             r6, [srcq+gprsize*r7-gprsize]
    movu           xm7, [r6+r5*2]

    vpbroadcastd   xm0, [filterq+2*r7-4]
    punpcklwd      xm9,  xm5,  xm7
    punpckhwd      xm5,  xm7
    pmaddwd        xm9,  xm0
    pmaddwd        xm5,  xm0
    paddd          xm1,  xm9
    paddd          xm2,  xm5

    sub             r7,  2
    jg .tailfilterloop

    psrad          xm1,  27 - %1
    psrad          xm2,  27 - %1
    packusdw       xm1,  xm2
    pminsw         xm1,  xm11
    movu   [dstq+r5*2], xm1

    add             r5,  8
    sub             wd,  8
    jg .tail
.end:
    RET
%endmacro

yuv2planeX_avx2_fn  9
yuv2planeX_avx2_fn 10
%endif

%if ARCH_X86_64 && HAVE_AVX512ICL_EXTERNAL
INIT_ZMM avx512icl

; 64 pixels per iteration.  Zen 4 splits a 512 bit operation into two 256 bit
; ones, so the width on its own buys little beyond the loop overhead; what pays
; here is vpdpwssd, which is the pmaddwd and the paddd of the avx2 inner loop in
; one instruction.  It is the non-saturating form, so it is bit exact with them.
;
; The lane split works out as it does at 256 bits: the accumulators hold pixels
; 0-3|8-11|16-19|24-27 and 4-7|12-15|20-23|28-31 of each 32, and packusdw is
; per lane too, so it puts them back in order with no fixup.
;
; The tail is one masked pass instead of the 16 and 8 pixel ladder, so it writes
; exactly dstW pixels rather than running into the next unit of 16.
%macro yuv2planeX_avx512icl_fn 1
cglobal yuv2planeX_%1, 5, 10, 16, filter, fltsize, src, dst, w
    vpbroadcastd   m14, [yuv2yuvX_%1_start]
    vpbroadcastw   m15, [yuv2yuvX_%1_upper]

    xor             r5,  r5
    cmp             wd,  64
    jl .tail

.pixelloop:
    mova            m1,  m14
    mova            m2,  m14
    mova            m3,  m14
    mova            m4,  m14
    movsxd          r7,  fltsizem
.filterloop:
    mov             r6, [srcq+gprsize*r7-2*gprsize]
    movu            m5, [r6+r5*2]
    movu            m6, [r6+r5*2+mmsize]
    mov             r6, [srcq+gprsize*r7-gprsize]
    movu            m7, [r6+r5*2]
    movu            m8, [r6+r5*2+mmsize]
    vpbroadcastd    m0, [filterq+2*r7-4]    ; coeff[0], coeff[1]

    punpcklwd       m9,  m5,  m7
    punpckhwd       m5,  m7
    vpdpwssd        m1,  m9,  m0
    vpdpwssd        m2,  m5,  m0

    punpcklwd       m9,  m6,  m8
    punpckhwd       m6,  m8
    vpdpwssd        m3,  m9,  m0
    vpdpwssd        m4,  m6,  m0

    sub             r7,  2
    jg .filterloop

    psrad           m1,  27 - %1
    psrad           m2,  27 - %1
    psrad           m3,  27 - %1
    psrad           m4,  27 - %1
    packusdw        m1,  m2                 ; px 0-31
    packusdw        m3,  m4                 ; px 32-63
    pminsw          m1,  m15
    pminsw          m3,  m15
    movu   [dstq+r5*2       ],  m1
    movu   [dstq+r5*2+mmsize],  m3

    add             r5,  64
    sub             wd,  64
    cmp             wd,  64
    jge .pixelloop

    test            wd,  wd
    jle .end

    ; 1-63 pixels left, or dstW < 64 to start with: at most two masked passes
.tail:
    mov            r8d,  wd
    cmp            r8d,  32
    jl .have_count
    mov            r8d,  32
.have_count:
    ; the mask is per word, and rax is x86inc's r6, which the loop below needs
    mov             r9, -1
    bzhi            r9,  r9, r8
    kmovd           k1,  r9d

    mova            m1,  m14
    mova            m2,  m14
    movsxd          r7,  fltsizem
.tailfilterloop:
    mov             r6, [srcq+gprsize*r7-2*gprsize]
    vmovdqu16 m5{k1}{z}, [r6+r5*2]
    mov             r6, [srcq+gprsize*r7-gprsize]
    vmovdqu16 m7{k1}{z}, [r6+r5*2]

    vpbroadcastd    m0, [filterq+2*r7-4]
    punpcklwd       m9,  m5,  m7
    punpckhwd       m5,  m7
    vpdpwssd        m1,  m9,  m0
    vpdpwssd        m2,  m5,  m0

    sub             r7,  2
    jg .tailfilterloop

    psrad           m1,  27 - %1
    psrad           m2,  27 - %1
    packusdw        m1,  m2
    pminsw          m1,  m15
    vmovdqu16 [dstq+r5*2]{k1}, m1

    add             r5,  32
    sub             wd,  32
    jg .tail
.end:
    RET
%endmacro

yuv2planeX_avx512icl_fn  9
yuv2planeX_avx512icl_fn 10

; 8 bit output, 64 pixels per iteration, with vpdpwssd as for 9/10 bits.  The
; dither, shifted up by 12, starts the accumulators as in the avx2 version;
; it has period 8, which a 128 bit lane holds, so the broadcast and the in-lane
; rotation work at 512 bits too.  The two packs leave the qwords of pixels 0-31
; and 32-63 interleaved per lane, and one vpermq sorts them.  The tail is one
; masked pass, so exactly dstW pixels are written.
cglobal yuv2planeX_8, 7, 11, 13, filter, fltsize, src, dst, w, dither, offset, x, line, cnt, tmp
    vpbroadcastq    m9, [ditherq]
    test       offsetd, offsetd
    jz .no_rot
    PALIGNR         m9, m9, 3, m0
.no_rot:
    pxor            m6, m6
    punpcklbw       m9, m6
    punpcklwd       m8, m9, m6              ; px 0-3 of each lane
    punpckhwd       m9, m6                  ; px 4-7
    pslld           m8, 12
    pslld           m9, 12
    movu           m10, [pack_order]

    movsxd          wq, wd
    xor             xq, xq
    cmp             wq, 64
    jl .tail

.pixelloop:
    mova            m1, m8
    mova            m2, m9
    mova            m3, m8
    mova            m4, m9
    movsxd        cntq, fltsized
.filterloop:
    mov          lineq, [srcq+gprsize*cntq-2*gprsize]
    movu            m5, [lineq+xq*2]
    movu            m6, [lineq+xq*2+mmsize]
    mov          lineq, [srcq+gprsize*cntq-gprsize]
    movu            m7, [lineq+xq*2]
    movu           m11, [lineq+xq*2+mmsize]
    vpbroadcastd    m0, [filterq+2*cntq-4]  ; coeff[0], coeff[1]
    punpcklwd      m12, m5, m7
    punpckhwd       m5, m7
    vpdpwssd        m1, m12, m0
    vpdpwssd        m2, m5, m0
    punpcklwd      m12, m6, m11
    punpckhwd       m6, m11
    vpdpwssd        m3, m12, m0
    vpdpwssd        m4, m6, m0
    sub           cntq, 2
    jg .filterloop

    psrad           m1, 19
    psrad           m2, 19
    psrad           m3, 19
    psrad           m4, 19
    packssdw        m1, m2                  ; px 0-31
    packssdw        m3, m4                  ; px 32-63
    packuswb        m1, m3
    vpermq          m1, m10, m1
    movu    [dstq+xq], m1

    add             xq, 64
    sub             wq, 64
    cmp             wq, 64
    jge .pixelloop

    test            wq, wq
    jle .end

    ; 1-63 pixels left, or dstW < 64 to start with
.tail:
    mov           tmpq, -1
    bzhi          tmpq, tmpq, wq
    kmovq           k3, tmpq                ; bytes of output
    kmovq           k1, tmpq                ; words of px 0-31
    kshiftrq        k2, k1, 32              ; words of px 32-63
    mova            m1, m8
    mova            m2, m9
    mova            m3, m8
    mova            m4, m9
    movsxd        cntq, fltsized
.tailfilterloop:
    mov          lineq, [srcq+gprsize*cntq-2*gprsize]
    vmovdqu16       m5{k1}{z}, [lineq+xq*2]
    vmovdqu16       m6{k2}{z}, [lineq+xq*2+mmsize]
    mov          lineq, [srcq+gprsize*cntq-gprsize]
    vmovdqu16       m7{k1}{z}, [lineq+xq*2]
    vmovdqu16      m11{k2}{z}, [lineq+xq*2+mmsize]
    vpbroadcastd    m0, [filterq+2*cntq-4]
    punpcklwd      m12, m5, m7
    punpckhwd       m5, m7
    vpdpwssd        m1, m12, m0
    vpdpwssd        m2, m5, m0
    punpcklwd      m12, m6, m11
    punpckhwd       m6, m11
    vpdpwssd        m3, m12, m0
    vpdpwssd        m4, m6, m0
    sub           cntq, 2
    jg .tailfilterloop

    psrad           m1, 19
    psrad           m2, 19
    psrad           m3, 19
    psrad           m4, 19
    packssdw        m1, m2
    packssdw        m3, m4
    packuswb        m1, m3
    vpermq          m1, m10, m1
    vmovdqu8 [dstq+xq]{k3}, m1
.end:
    RET
%endif

; %1=outout-bpc, %2=alignment (u/a)
%macro yuv2plane1_mainloop 2
.loop_%2:
%if %1 == 8
    paddsw          m0, m2, [srcq+wq*2+mmsize*0]
    paddsw          m1, m3, [srcq+wq*2+mmsize*1]
    psraw           m0, 7
    psraw           m1, 7
    packuswb        m0, m1
    mov%2    [dstq+wq], m0
%elif %1 == 16
    paddd           m0, m4, [srcq+wq*4+mmsize*0]
    paddd           m1, m4, [srcq+wq*4+mmsize*1]
    paddd           m2, m4, [srcq+wq*4+mmsize*2]
    paddd           m3, m4, [srcq+wq*4+mmsize*3]
    psrad           m0, 3
    psrad           m1, 3
    psrad           m2, 3
    psrad           m3, 3
%if cpuflag(sse4) ; avx/sse4
    packusdw        m0, m1
    packusdw        m2, m3
%else ; mmx/sse2
    packssdw        m0, m1
    packssdw        m2, m3
    paddw           m0, m5
    paddw           m2, m5
%endif ; mmx/sse2/sse4/avx
    mov%2    [dstq+wq*2+mmsize*0], m0
    mov%2    [dstq+wq*2+mmsize*1], m2
%else ; %1 == 9/10
    paddsw          m0, m2, [srcq+wq*2+mmsize*0]
    paddsw          m1, m2, [srcq+wq*2+mmsize*1]
    psraw           m0, 15 - %1
    psraw           m1, 15 - %1
    pmaxsw          m0, m4
    pmaxsw          m1, m4
    pminsw          m0, m3
    pminsw          m1, m3
    mov%2    [dstq+wq*2+mmsize*0], m0
    mov%2    [dstq+wq*2+mmsize*1], m1
%endif
    add             wq, mmsize
    jl .loop_%2
%endmacro

%macro yuv2plane1_fn 3
cglobal yuv2plane1_%1, %3, %3, %2, src, dst, w, dither, offset
    movsxdifnidn    wq, wd
    add             wq, mmsize - 1
    and             wq, ~(mmsize - 1)
%if %1 == 8
    add           dstq, wq
%else ; %1 != 8
    lea           dstq, [dstq+wq*2]
%endif ; %1 == 8
%if %1 == 16
    lea           srcq, [srcq+wq*4]
%else ; %1 != 16
    lea           srcq, [srcq+wq*2]
%endif ; %1 == 16
    neg             wq

%if %1 == 8
    pxor            m4, m4               ; zero

    ; create registers holding dither
    movq            m3, [ditherq]        ; dither
    test       offsetd, offsetd
    jz              .no_rot
%if mmsize == 16
    punpcklqdq      m3, m3
%endif ; mmsize == 16
    PALIGNR         m3, m3, 3, m2
.no_rot:
%if mmsize == 8
    mova            m2, m3
    punpckhbw       m3, m4               ; byte->word
    punpcklbw       m2, m4               ; byte->word
%else
    punpcklbw       m3, m4
    mova            m2, m3
%endif
%elif %1 == 9
    pxor            m4, m4
    mova            m3, [pw_512]
    mova            m2, [pw_32]
%elif %1 == 10
    pxor            m4, m4
    mova            m3, [pw_1024]
    mova            m2, [pw_16]
%else ; %1 == 16
%if cpuflag(sse4) ; sse4/avx
    mova            m4, [pd_4]
%else ; mmx/sse2
    mova            m4, [pd_4min0x40000]
    mova            m5, [minshort]
%endif ; mmx/sse2/sse4/avx
%endif ; %1 == ..

    ; actual pixel scaling
%if mmsize == 8
    yuv2plane1_mainloop %1, a
%else ; mmsize == 16
    test          dstq, 15
    jnz .unaligned
    yuv2plane1_mainloop %1, a
    REP_RET
.unaligned:
    yuv2plane1_mainloop %1, u
%endif ; mmsize == 8/16
    REP_RET
%endmacro

%if ARCH_X86_32
INIT_MMX mmx
yuv2plane1_fn  8, 0, 5
yuv2plane1_fn 16, 0, 3

INIT_MMX mmxext
yuv2plane1_fn  9, 0, 3
yuv2plane1_fn 10, 0, 3
%endif

INIT_XMM sse2
yuv2plane1_fn  8, 5, 5
yuv2plane1_fn  9, 5, 3
yuv2plane1_fn 10, 5, 3
yuv2plane1_fn 16, 6, 3

INIT_XMM sse4
yuv2plane1_fn 16, 5, 3

%if HAVE_AVX_EXTERNAL
INIT_XMM avx
yuv2plane1_fn  8, 5, 5
yuv2plane1_fn  9, 5, 3
yuv2plane1_fn 10, 5, 3
yuv2plane1_fn 16, 5, 3
%endif

%undef movsx

;-----------------------------------------------------------------------------
; AVX2 yuv2nv12cX implementation
;
; void ff_yuv2nv12cX_avx2(enum AVPixelFormat format, const uint8_t *dither,
;                         const int16_t *filter, int filterSize,
;                         const int16_t **u, const int16_t **v,
;                         uint8_t *dst, int dstWidth)
;
; void ff_yuv2nv21cX_avx2(enum AVPixelFormat format, const uint8_t *dither,
;                         const int16_t *filter, int filterSize,
;                         const int16_t **u, const int16_t **v,
;                         uint8_t *dst, int dstWidth)
;-----------------------------------------------------------------------------

%if ARCH_X86_64
%macro yuv2nv12cX_fn 1
cglobal %1cX, 8, 11, 13, tmp1, dither, filter, filterSize, u, v, dst, dstWidth

    mov tmp1q, qword [ditherq]
    movq xm0, tmp1q
    ror tmp1q, 24
    movq xm1, tmp1q

    pmovzxbd m0, xm0
    pslld m0, m0, 12                        ; ditherLo
    pmovzxbd m1, xm1
    pslld m1, m1, 12                        ; ditherHi

    pxor m9, m9                             ; uint8_min dwords
    mova m10, [pd_255]                      ; uint8_max dwords
    mova m11, [%1_shuffle_mask]             ; shuffle_mask
    mova m12, [yuv2nv12_permute_mask]       ; permute mask

    DEFINE_ARGS tmp1, tmp2, filter, filterSize, u, v, dst, dstWidth

    xor r8q, r8q

nv12_outer_%1:
    mova m2, m0                             ; resultLo
    mova m3, m1                             ; resultHi
    xor r9q, r9q

nv12_inner_%1:
    movsx r10d, word [filterq + (2 * r9q)]
    movd xm4, r10d
    vpbroadcastd m4, xm4                    ; filter

    mov tmp1q, [uq + (gprsize * r9q)]
    mova xm7, oword [tmp1q + 2 * r8q]

    mov tmp2q, [vq + (gprsize * r9q)]
    mova xm8, oword [tmp2q + 2 * r8q]

    punpcklwd xm5, xm7, xm8
    pmovsxwd m5, xm5                        ; multiplicandsLo
    punpckhwd xm6, xm7, xm8
    pmovsxwd m6, xm6                        ; multiplicandsHi

    pmulld m7, m5, m4                       ; mulResultLo
    pmulld m8, m6, m4                       ; mulResultHi
    paddd m2, m2, m7                        ; resultLo += mulResultLo
    paddd m3, m3, m8                        ; resultHi += mulResultHi

    inc r9d
    cmp r9d, filterSized
    jl nv12_inner_%1
    ; end of inner loop

    psrad m2, m2, 19
    psrad m3, m3, 19

    ; Vectorized av_clip_uint8
    pmaxsd m2, m2, m9
    pmaxsd m3, m3, m9
    pminsd m2, m2, m10
    pminsd m3, m3, m10

    ; At this point we have clamped uint8s arranged in this order:
    ;     m2: u1  0  0  0  v1  0  0  0  [...]
    ;     m3: u5  0  0  0  v5  0  0  0  [...]
    ;
    ; First, we shuffle the bytes to make the bytes semi-contiguous.
    ; AVX-2 doesn't have cross-lane shuffling, so we'll end up with:
    ;     m2: u1  v1  u2  v2  0  0  0  0  0  0  0  0  u3  v3  u4  v4
    ;     m3: u5  v5  u6  v6  0  0  0  0  0  0  0  0  u7  v7  u8  v8
    pshufb m2, m2, m11
    pshufb m3, m3, m11

    ; To fix the cross-lane shuffling issue, we'll then use cross-lane
    ; permutation to combine the two segments
    vpermd m2, m12, m2
    vpermd m3, m12, m3

    ; Now we have the final results in the lower 8 bytes of each register
    movq [dstq], xm2
    movq [dstq + 8], xm3

    add r8d, 8
    add dstq, 16

    cmp r8d, dstWidthd
    jl nv12_outer_%1
    RET
%endmacro

INIT_YMM avx2
yuv2nv12cX_fn yuv2nv12
yuv2nv12cX_fn yuv2nv21
%endif ; ARCH_X86_64
