;******************************************************************************
;* AVX2 and AVX-512 vertical scaling with the MMX filter layout
;* Copyright (c) 2026 Open Broadcast Systems Ltd
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

SECTION .text

;-----------------------------------------------------------------------------
; void yuv2yuvX_<opt>(const int16_t *filter, int filterSize,
;                     const int16_t **src, uint8_t *dest, int dstW,
;                     const uint8_t *dither, int offset);
;
; The yuv2planeX of use_mmx_vfilter, i.e. 8 bit output without
; SWS_ACCURATE_RND or SWS_BITEXACT, and bit exact with the inline asm
; yuv2yuvX_sse3/yuv2yuvX_mmxext in swscale.c, not with the C yuv2planeX_8:
;
;  - filter is c->lumMmxFilter/chrMmxFilter: per line a 16 byte entry of
;    { const int16_t *line; int16_t coeff[4] }, ending at a NULL pointer;
;    src is not used
;  - the line is read from element offset on and written from dest[0] on:
;    for the V plane the U lines are passed, and offset reaches over to V
;  - the dither is rotated by 3 for any non-zero offset
;  - each pixel starts at (dither + 8 * (filterSize - 1)) >> 4, adds
;    pmulhw(line, coeff) for every line, in 16 bits, and ends at
;    packuswb(acc >> 3)
;
; The avx2 version does 32 pixels an iteration and the last 1-31 in units of
; 16, so it writes up to FFALIGN(dstW, 16), as the sse3 one does; the avx512
; one does 64 and masks the tail, writing and reading nothing past dstW.
;-----------------------------------------------------------------------------

; the start value of every pixel, in m7 (sixteen words per lane pair, the
; dither has period 8); m0 m1 m5 scratch
%macro YUV2YUVX_START 0
    vpbroadcastq    m7, [ditherq]
    test       offsetd, offsetd
    jz .no_rot
    psrlq           m5, m7, 24
    psllq           m7, 40
    por             m7, m5
.no_rot:
    pxor            m0, m0
    punpcklbw       m7, m0
    dec        fltsized
    movd           xm1, fltsized
    vpbroadcastw    m1, xm1
    psllw           m1, 3
    paddw           m7, m1
    psraw           m7, 4
%endmacro

%macro YUV2YUVX_FN 0
cglobal yuv2yuvX, 7, 10, 8, filter, fltsize, src, dst, w, dither, offset, fptr, line, x
    movsxd          wq, wd
    movsxd     offsetq, offsetd
    YUV2YUVX_START
%if cpuflag(avx512)
    mova            m6, [pack_order]
%endif
    lea             xq, [offsetq*2]         ; the byte offset into the lines
    add           dstq, wq
    neg             wq
%if cpuflag(avx512)
    add             wq, mmsize
    jg .tail
%else
    add             wq, mmsize
    jg .check16
%endif

.loop:
    mova            m3, m7
    mova            m4, m7
    mov          fptrq, filterq
    mov          lineq, [filterq]
.filterloop:
    vpbroadcastq    m0, [fptrq+8]
    pmulhw          m1, m0, [lineq+xq]
    pmulhw          m2, m0, [lineq+xq+mmsize]
    paddw           m3, m1
    paddw           m4, m2
    add          fptrq, 16
    mov          lineq, [fptrq]
    test         lineq, lineq
    jnz .filterloop
    psraw           m3, 3
    psraw           m4, 3
    packuswb        m3, m4
%if cpuflag(avx512)
    vpermq          m3, m6, m3
%else
    vpermq          m3, m3, q3120
%endif
    movu [dstq+wq-mmsize], m3
    add             xq, mmsize*2
    add             wq, mmsize
    jle .loop

%if cpuflag(avx512)
.tail:
    sub             wq, mmsize              ; -(pixels left)
    jz .end
    ; the masks for the pixels left: words of each half, bytes of the output
    mov          fptrq, wq
    neg          fptrq
    mov          lineq, -1
    bzhi         lineq, lineq, fptrq
    kmovq           k3, lineq
    kmovq           k1, lineq
    kshiftrq        k2, k1, 32

    mova            m3, m7
    mova            m4, m7
    mov          fptrq, filterq
    mov          lineq, [filterq]
.tailfilterloop:
    vpbroadcastq    m0, [fptrq+8]
    vmovdqu16       m1{k1}{z}, [lineq+xq]
    vmovdqu16       m2{k2}{z}, [lineq+xq+mmsize]
    pmulhw          m1, m0
    pmulhw          m2, m0
    paddw           m3, m1
    paddw           m4, m2
    add          fptrq, 16
    mov          lineq, [fptrq]
    test         lineq, lineq
    jnz .tailfilterloop
    psraw           m3, 3
    psraw           m4, 3
    packuswb        m3, m4
    vpermq          m3, m6, m3
    vmovdqu8 [dstq+wq]{k3}, m3
.end:
    RET
%else
.check16:
    sub             wq, mmsize
    jge .end
    ; 1-31 pixels left, in units of 16
.tail:
    mova           xm3, xm7
    mova           xm4, xm7
    mov          fptrq, filterq
    mov          lineq, [filterq]
.tailfilterloop:
    vpbroadcastq   xm0, [fptrq+8]
    pmulhw         xm1, xm0, [lineq+xq]
    pmulhw         xm2, xm0, [lineq+xq+16]
    paddw          xm3, xm1
    paddw          xm4, xm2
    add          fptrq, 16
    mov          lineq, [fptrq]
    test         lineq, lineq
    jnz .tailfilterloop
    psraw          xm3, 3
    psraw          xm4, 3
    packuswb       xm3, xm4
    movu   [dstq+wq], xm3
    add             xq, 32
    add             wq, 16
    jl .tail
.end:
    RET
%endif
%endmacro

%if HAVE_AVX2_EXTERNAL
INIT_YMM avx2
YUV2YUVX_FN
%endif

%if HAVE_AVX512_EXTERNAL
SECTION_RODATA 64
; packuswb leaves the qwords of 0-31 and 32-63 interleaved per lane
pack_order: dq 0, 2, 4, 6, 1, 3, 5, 7

SECTION .text
INIT_ZMM avx512
YUV2YUVX_FN
%endif

%endif ; ARCH_X86_64
