;******************************************************************************
;* AVX2 8 bit to 16 bit plane copy for the unscaled path
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

%if ARCH_X86_64 && HAVE_AVX2_EXTERNAL

SECTION .text

;-----------------------------------------------------------------------------
; 8 bit to 16 bit little endian plane copy
;
; void ff_plane_copy_8to16[_shiftonly]_<shift>_avx2(uint16_t *dst,
;                                                   const uint8_t *src,
;                                                   int length);
;
; planarCopyWrapper()'s promotion of an 8 bit plane to a 9 to 16 bit one, which
; an unscaled conversion between two planar formats of the same subsampling
; falls into whatever the depths are.  The shiftonly variant is the plain
; left shift the chroma planes and the limited range luma plane want; the other
; replicates the high bits into the low ones, as the alpha plane and full range
; luma want.
;
; pmovzxbw is used rather than punpcklbw so that a 32 byte store comes out of
; one load with the pixels in order: punpcklbw on a ymm works per 128 bit lane
; and would need a permute to put the halves back.  Nothing here is aligned,
; the caller's length is a picture width and the strides come from whatever
; allocated the planes.
;-----------------------------------------------------------------------------

; PLANE_COPY_8TO16 name, shift, replicate high bits into low
%macro PLANE_COPY_8TO16 3
cglobal %1, 3, 6, 2, dst, src, len, cnt, tmp, tmp2
    movsxd      lenq, lend
    test        lenq, lenq
    jle .end                        ; the tail below runs before it tests
    xor         cntq, cntq
    cmp         lenq, mmsize/2
    jl .tail_loop
    sub         lenq, mmsize/2

.loop:
    pmovzxbw      m0, [srcq+cntq]
%if %3
    psllw         m1, m0, %2
    psrlw         m0, m0, 8-%2
    por           m0, m1
%else
    psllw         m0, m0, %2
%endif
    movu [dstq+cntq*2], m0
    add         cntq, mmsize/2
    cmp         cntq, lenq
    jle .loop

    add         lenq, mmsize/2
    cmp         cntq, lenq
    jge .end

    ; 1 to mmsize/2 - 1 pixels left, or a line shorter than one vector
.tail_loop:
    movzx       tmpd, byte [srcq+cntq]
%if %3
    mov        tmp2d, tmpd
    shl         tmpd, %2
    shr        tmp2d, 8-%2
    or          tmpd, tmp2d
%else
    shl         tmpd, %2
%endif
    mov  [dstq+cntq*2], tmpw
    inc         cntq
    cmp         cntq, lenq
    jl .tail_loop
.end:
    RET
%endmacro

; PLANE_COPY_8TO16_BOTH shift
%macro PLANE_COPY_8TO16_BOTH 1
PLANE_COPY_8TO16 plane_copy_8to16_%1,           %1, 1
PLANE_COPY_8TO16 plane_copy_8to16_shiftonly_%1, %1, 0
%endmacro

INIT_YMM avx2
PLANE_COPY_8TO16_BOTH 1 ; 9 bit
PLANE_COPY_8TO16_BOTH 2 ; 10 bit
PLANE_COPY_8TO16_BOTH 4 ; 12 bit
PLANE_COPY_8TO16_BOTH 6 ; 14 bit
PLANE_COPY_8TO16_BOTH 8 ; 16 bit

%endif ; ARCH_X86_64 && HAVE_AVX2_EXTERNAL
