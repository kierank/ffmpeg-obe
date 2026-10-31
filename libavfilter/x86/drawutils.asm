;*****************************************************************************
;* x86-optimized blending for drawutils (8-bit planes)
;*
;* Copyright (C) 2026 Open Broadcast Systems Ltd
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

SECTION_RODATA 64

pb_1:   times 64 db 1
pw_1:   times 32 dw 1
swap_w: db 2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13
dup_w:  db 0, 1, 0, 1, 4, 5, 4, 5,  8,  9, 8, 9, 12, 13, 12, 13

SECTION .text

%if ARCH_X86_64

; Both kernels match the C code in libavfilter/drawutils.c bit for bit.
; The AVX2 versions finish each row with a scalar loop, the AVX-512 versions
; with a masked vector; neither touches bytes outside [dst, dst + w).

;-----------------------------------------------------------------------------
; void blend_rect8(uint8_t *dst, ptrdiff_t dst_linesize, int w, int h,
;                  const FFBlendRect8Params *p)
;
; dst = (dst * tau + s) >> 24 with tau = t1 << 16 | t0, s = s1 << 16 | s0:
; dst * tau + s = (dst * t1 + hi16(dst * t0) + s1 + carry) << 16 + low bits,
; carry = lo16(dst * t0) + s0 > 0xffff = lo16(dst * t0) > ~s0.
; The sum fits in 16 bits because dst * tau + s < 2^32.
;-----------------------------------------------------------------------------

; m6 t0, m7 t1, m8 s1, m9 ~s0, m10 zero, m11 pw_1
%macro RECT8_WORDS 4 ; register prefix, words, tmp, tmp
    pmulhuw   %1 %+ %3, %1 %+ %2, %1 %+ 6
    pmullw    %1 %+ %4, %1 %+ %2, %1 %+ 6
    pmullw    %1 %+ %2, %1 %+ 7
    paddw     %1 %+ %2, %1 %+ %3
    paddw     %1 %+ %2, %1 %+ 8
    psubusw   %1 %+ %4, %1 %+ 9
    pminuw    %1 %+ %4, %1 %+ 11
    paddw     %1 %+ %2, %1 %+ %4
    psrlw     %1 %+ %2, 8
%endmacro

; bytes in and out in %1 0
%macro RECT8_VEC 1 ; register prefix
    punpckhbw %1 %+ 1, %1 %+ 0, %1 %+ 10
    punpcklbw %1 %+ 0, %1 %+ 10
    RECT8_WORDS %1, 0, 2, 3
    RECT8_WORDS %1, 1, 4, 5
    packuswb  %1 %+ 0, %1 %+ 1
%endmacro

%macro BLEND_RECT8 0
cglobal blend_rect8, 5, 8, 12, dst, ls, w, h, p, pos, t, u
    movsxd    wq, wd
    vpbroadcastd m6, [pq+ 0]
    vpbroadcastd m7, [pq+ 4]
    vpbroadcastd m8, [pq+ 8]
    vpbroadcastd m9, [pq+12]
    pxor      m10, m10
    movu      m11, [pw_1]
%if cpuflag(avx512)
    mov       tq, wq
    and       tq, mmsize-1
    or        uq, -1
    bzhi      uq, uq, tq
    kmovq     k1, uq
%endif
.row:
    xor       posd, posd
    jmp       .check
.loop:
    movu      m0, [dstq+posq]
    RECT8_VEC m
    movu      [dstq+posq], m0
    add       posq, mmsize
.check:
    lea       tq, [posq+mmsize]
    cmp       tq, wq
    jle       .loop
%if cpuflag(avx512)
    cmp       posq, wq
    jge       .next
    vmovdqu8  m0{k1}{z}, [dstq+posq]
    RECT8_VEC m
    vmovdqu8  [dstq+posq]{k1}, m0
%else
    lea       tq, [posq+16]
    cmp       tq, wq
    jg        .scalar
    movu      xm0, [dstq+posq]
    RECT8_VEC xm
    movu      [dstq+posq], xm0
    add       posq, 16
.scalar: ; x is even here: one even and one odd byte per iteration
    cmp       posq, wq
    jge       .next
    movzx     td, byte [dstq+posq]
    imul      td, [pq+16]
    add       td, [pq+20]
    shr       td, 24
    mov       [dstq+posq], tb
    inc       posq
    cmp       posq, wq
    jge       .next
    movzx     td, byte [dstq+posq]
    imul      td, [pq+16]
    add       td, [pq+24]
    shr       td, 24
    mov       [dstq+posq], tb
    inc       posq
    jmp       .scalar
%endif
.next:
    add       dstq, lsq
    dec       hd
    jg        .row
    RET
%endmacro

;-----------------------------------------------------------------------------
; void blend_mask8_h<H>s<S>(uint8_t *dst, ptrdiff_t dst_linesize,
;                           const uint8_t *mask, int w, int h,
;                           const FFBlendMask8Params *p)
;
; H: mask columns summed per pixel (1 or 2)
; S: bytes per pixel (1 planar, 2 interleaved chroma)
; m = (sum over H columns of row and row + row2) >> shift, a = m * alpha,
; dst = (dst * 0x1010101 + a * (src - dst)) >> 24 (mod 2^32, as in C).
; a * (src - dst) is done as alpha * (m * (src - dst)), the inner product
; exact in 32 bits from pmullw/pmulhw.
;-----------------------------------------------------------------------------

; m8 alpha, m9 src words, xm10 shift, m11 shuffle or pb_1
%macro MASK8_SUM 3 ; H, S, masked
%if %1 == 1 && %2 == 1
  %if %3
    vmovdqu8  ym1{k2}{z}, [mpq]
    vmovdqu8  ym2{k2}{z}, [mpq+row2q]
    pmovzxbw  m1, ym1
    pmovzxbw  m2, ym2
  %else
    pmovzxbw  m1, [mpq]
    pmovzxbw  m2, [mpq+row2q]
  %endif
    paddw     m1, m2
%elif %1 == 2 && %2 == 1
  %if %3
    vmovdqu8  m1{k2}{z}, [mpq]
    vmovdqu8  m2{k2}{z}, [mpq+row2q]
  %else
    movu      m1, [mpq]
    movu      m2, [mpq+row2q]
  %endif
    pmaddubsw m1, m11
    pmaddubsw m2, m11
    paddw     m1, m2
%elif %1 == 2 ; 2 columns, interleaved: byte j uses columns j & ~1 and j | 1
  %if %3
    vmovdqu8  ym1{k2}{z}, [mpq]
    vmovdqu8  ym2{k2}{z}, [mpq+row2q]
    pmovzxbw  m1, ym1
    pmovzxbw  m2, ym2
  %else
    pmovzxbw  m1, [mpq]
    pmovzxbw  m2, [mpq+row2q]
  %endif
    paddw     m1, m2
    pshufb    m2, m1, m11
    paddw     m1, m2
%else ; 1 column, interleaved: byte j uses column j >> 1
  %if %3
    vmovdqu8  xm1{k2}{z}, [mpq]
    vmovdqu8  xm2{k2}{z}, [mpq+row2q]
    pmovzxbd  m1, xm1
    pmovzxbd  m2, xm2
  %else
    pmovzxbd  m1, [mpq]
    pmovzxbd  m2, [mpq+row2q]
  %endif
    paddd     m1, m2
    pshufb    m1, m11
%endif
    psrlw     m1, xm10
%endmacro

%macro MASK8_VEC 3 ; H, S, masked
    MASK8_SUM %1, %2, %3
%if %3
    vmovdqu8  ym0{k1}{z}, [dstq+posq]
    pmovzxbw  m0, ym0
%else
    pmovzxbw  m0, [dstq+posq]
%endif
    psubw     m2, m9, m0
    pmullw    m3, m1, m2
    pmulhw    m1, m2
    punpcklwd m2, m3, m1
    punpckhwd m3, m1
    pmulld    m2, m8
    pmulld    m3, m8
    psllw     m1, m0, 8
    por       m0, m1
    punpcklwd m1, m0, m0
    punpckhwd m0, m0
    paddd     m1, m2
    paddd     m0, m3
    psrld     m1, 24
    psrld     m0, 24
    packusdw  m1, m0
%if mmsize == 64
    vpmovwb   ym1, m1
  %if %3
    vmovdqu8  [dstq+posq]{k1}, ym1
  %else
    vmovdqu8  [dstq+posq], ym1
  %endif
%else
    vextracti128 xm0, m1, 1
    packuswb  xm1, xm0
    movu      [dstq+posq], xm1
%endif
%endmacro

; t = a, writes dst[x + %1] using the colour at p + %2
%macro MASK8_SCALAR 2
    movzx     ud, byte [dstq+posq+%1]
    mov       vd, 0x1010101
    sub       vd, td
    imul      ud, vd
    movzx     vd, word [pq+%2]
    imul      vd, td
    add       ud, vd
    shr       ud, 24
    mov       [dstq+posq+%1], ub
%endmacro

%macro BLEND_MASK8 2 ; H, S
%assign VW mmsize/2      ; destination bytes per vector
%assign MW VW*%1/%2      ; mask bytes per vector
cglobal blend_mask8_h%1s%2, 6, 13, 12, dst, ls, mask, w, h, p, pos, mp, row2, t, u, v, sh
    movsxd    wq, wd
    mov       row2q, [pq+16]
    mov       shd, [pq+8]
    vpbroadcastd m8, [pq]
    vpbroadcastd m9, [pq+4]
    movd      xm10, shd
%if %1 == 2 && %2 == 1
    movu      m11, [pb_1]
%elif %1 == 2
    vbroadcasti128 m11, [swap_w]
%elif %2 == 2
    vbroadcasti128 m11, [dup_w]
%endif
%if cpuflag(avx512)
    mov       tq, wq
    and       tq, VW-1
    or        uq, -1
    bzhi      vq, uq, tq
    kmovq     k1, vq
  %if MW > VW
    add       tq, tq
    bzhi      vq, uq, tq
  %elif MW < VW
    shr       tq, 1
    bzhi      vq, uq, tq
  %endif
    kmovq     k2, vq
%endif
.row:
    xor       posd, posd
    mov       mpq, maskq
    jmp       .check
.loop:
    MASK8_VEC %1, %2, 0
    add       posq, VW
    add       mpq, MW
.check:
    lea       tq, [posq+VW]
    cmp       tq, wq
    jle       .loop
%if cpuflag(avx512)
    cmp       posq, wq
    jge       .next
    MASK8_VEC %1, %2, 1
%else
.scalar: ; one pixel (S bytes) per iteration
    cmp       posq, wq
    jge       .next
    movzx     td, byte [mpq]
    movzx     ud, byte [mpq+row2q]
    add       td, ud
  %if %1 == 2
    movzx     ud, byte [mpq+1]
    add       td, ud
    movzx     ud, byte [mpq+row2q+1]
    add       td, ud
  %endif
    shrx      td, td, shd
    imul      td, [pq]
    MASK8_SCALAR 0, 4
  %if %2 == 2
    MASK8_SCALAR 1, 6
  %endif
    add       posq, %2
    add       mpq, %1
    jmp       .scalar
%endif
.next:
    add       dstq, lsq
    add       maskq, [pq+24]
    dec       hd
    jg        .row
    RET
%endmacro

%macro BLEND_FUNCS 0
BLEND_RECT8
BLEND_MASK8 1, 1
BLEND_MASK8 2, 1
BLEND_MASK8 1, 2
BLEND_MASK8 2, 2
%endmacro

%if HAVE_AVX2_EXTERNAL
INIT_YMM avx2
BLEND_FUNCS
%endif

%if HAVE_AVX512_EXTERNAL
INIT_ZMM avx512
BLEND_FUNCS
%endif

%endif ; ARCH_X86_64
