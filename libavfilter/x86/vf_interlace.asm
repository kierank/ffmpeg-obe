;*****************************************************************************
;* x86-optimized functions for interlace filter
;*
;* Copyright (C) 2014 Kieran Kunhya <kierank@obe.tv>
;* Copyright (c) 2014 Michael Niedermayer <michaelni@gmx.at>
;* Copyright (c) 2017 Thomas Mundt <tmundt75@gmail.com>
;*
;* This file is part of FFmpeg.
;*
;* FFmpeg is free software; you can redistribute it and/or modify
;* it under the terms of the GNU General Public License as published by
;* the Free Software Foundation; either version 2 of the License, or
;* (at your option) any later version.
;*
;* FFmpeg is distributed in the hope that it will be useful,
;* but WITHOUT ANY WARRANTY; without even the implied warranty of
;* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;* GNU General Public License for more details.
;*
;* You should have received a copy of the GNU General Public License along
;* with FFmpeg; if not, write to the Free Software Foundation, Inc.,
;* 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
;******************************************************************************

%include "libavutil/x86/x86util.asm"

SECTION_RODATA 32

pw_4: times 16 dw 4

SECTION .text

%macro LOWPASS_LINE 0
cglobal lowpass_line, 5, 5, 7, dst, h, src, mref, pref
    add dstq, hq
    add srcq, hq
    add mrefq, srcq
    add prefq, srcq
    neg hq

    pcmpeqb m6, m6

.loop:
    mova m0, [mrefq+hq]
    mova m1, [mrefq+hq+mmsize]
    pavgb m0, [prefq+hq]
    pavgb m1, [prefq+hq+mmsize]
    pxor m0, m6
    pxor m1, m6
    pxor m2, m6, [srcq+hq]
    pxor m3, m6, [srcq+hq+mmsize]
    pavgb m0, m2
    pavgb m1, m3
    pxor m0, m6
    pxor m1, m6
    mova [dstq+hq], m0
    mova [dstq+hq+mmsize], m1

    add hq, 2*mmsize
    jl .loop
REP_RET

%endmacro

%macro LOWPASS_LINE_COMPLEX 0
cglobal lowpass_line_complex, 5, 5, 7, dst, h, src, mref, pref
    pxor m6, m6
.loop:
    movu m0, [srcq+mrefq]
    movu m2, [srcq+prefq]
    mova m1, m0
    mova m3, m2
    punpcklbw m0, m6
    punpcklbw m2, m6
    punpckhbw m1, m6
    punpckhbw m3, m6
    paddw m0, m2
    paddw m1, m3
    movu m2, [srcq+mrefq*2]
    movu m4, [srcq+prefq*2]
    mova m3, m2
    mova m5, m4
    punpcklbw m2, m6
    punpcklbw m4, m6
    punpckhbw m3, m6
    punpckhbw m5, m6
    paddw m2, m4
    paddw m3, m5
    movu m4, [srcq]
    mova m5, m4
    punpcklbw m4, m6
    punpckhbw m5, m6
    paddw m0, m4
    paddw m1, m5
    psllw m0, 1
    psllw m1, 1
    psllw m4, 2
    psllw m5, 2
    paddw m0, m4
    paddw m1, m5
    paddw m0, [pw_4]
    paddw m1, [pw_4]
    psubusw m0, m2
    psubusw m1, m3
    psrlw m0, 3
    psrlw m1, 3
    packuswb m0, m1
    movu [dstq], m0

    add dstq, mmsize
    add srcq, mmsize
    sub hd, mmsize
    jg .loop
REP_RET

%endmacro

INIT_XMM sse2
LOWPASS_LINE

INIT_XMM avx
LOWPASS_LINE

INIT_XMM sse2
LOWPASS_LINE_COMPLEX

%if HAVE_AVX2_EXTERNAL
INIT_YMM avx2
LOWPASS_LINE_COMPLEX
%endif

; Exact width: nothing is written past dst[w-1].
%macro LOWPASS_LINE_EXACT_BODY 1 ; offset
    movu      m0, [mrefq+%1]
    pavgb     m0, [prefq+%1]
    pxor      m0, m3
    pxor      m1, m3, [srcq+%1]
    pavgb     m0, m1
    pxor      m0, m3
    movu      [dstq+%1], m0
%endmacro

%macro LOWPASS_LINE_EXACT 0
cglobal lowpass_line, 5, 8, 4, dst, w, src, mref, pref, pos, t, u
    add       mrefq, srcq
    add       prefq, srcq
%if cpuflag(avx512)
    vpternlogd m3, m3, m3, 0xff
%else
    pcmpeqb   m3, m3
%endif
    xor       posd, posd
%if cpuflag(avx512)
    mov       tq, wq
    and       tq, mmsize-1
    or        uq, -1
    bzhi      uq, uq, tq
    kmovq     k1, uq
%endif
    lea       tq, [posq+mmsize]
    cmp       tq, wq
    jg        .tail
.loop:
    LOWPASS_LINE_EXACT_BODY posq
    add       posq, mmsize
    lea       tq, [posq+mmsize]
    cmp       tq, wq
    jle       .loop
.tail:
    cmp       posq, wq
    jge       .end
%if cpuflag(avx512)
    vmovdqu8  m0{k1}{z}, [mrefq+posq]
    vmovdqu8  m2{k1}{z}, [prefq+posq]
    vmovdqu8  m1{k1}{z}, [srcq+posq]
    pavgb     m0, m2
    pxor      m0, m3
    pxor      m1, m3
    pavgb     m0, m1
    pxor      m0, m3
    vmovdqu8  [dstq+posq]{k1}, m0
%else
    cmp       wq, mmsize
    jl        .scalar
    ; last vector overlaps the previous one; same inputs give the same bytes
    lea       posq, [wq-mmsize]
    LOWPASS_LINE_EXACT_BODY posq
    jmp       .end
.scalar:
    movzx     td, byte [srcq+posq]
    movzx     ud, byte [mrefq+posq]
    lea       td, [td*2+ud+1]
    movzx     ud, byte [prefq+posq]
    add       td, ud
    shr       td, 2
    mov       [dstq+posq], tb
    inc       posq
    cmp       posq, wq
    jl        .scalar
%endif
.end:
    RET
%endmacro

%if ARCH_X86_64
%if HAVE_AVX2_EXTERNAL
INIT_YMM avx2
LOWPASS_LINE_EXACT
%endif
%if HAVE_AVX512_EXTERNAL
INIT_ZMM avx512
LOWPASS_LINE_EXACT
%endif
%endif
