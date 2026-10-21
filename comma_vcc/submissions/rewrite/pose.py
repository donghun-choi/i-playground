"""pose: pos4 섹션 풀기, 짝수 프레임 = 아핀(이전 홀수 프레임) + 기저 합성 (GUIDE 3.5, 4.6)."""

from __future__ import annotations

import numpy as np
import torch


def rice_decode(buf: bytes, r: int, n: int) -> np.ndarray:
    """[i32 평균 m][u8 kk] × r 행 + 비트열 (MSB 먼저) → (r, n) int64.
    값마다 1 이 q 개, 0 하나, kk 비트 나머지 → z = (q << kk) | 나머지 → zigzag 되돌리기 (짝수 z/2, 홀수 -(z+1)/2) + m."""
    raise NotImplementedError("TODO: rice_decode")


def parse_pose4(buf: bytes) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """pos4 섹션 → (B (k,C,bh,bw), c (n,k), a (n,6)) float32 텐서.

    헤더 [u16 n][u16 k][u8 C][u16 bh][u16 bw] + f32 c_step (k) + f32 a_step (6) + f32 B_scale (k)
    + 기저 정수 ([u8 qmax][r × k·C][u32 단어 수][단어들], 행 = (기저, 채널), 행마다 bh·bw 개) + rice_decode (k+6 행: c 의 k 행, a 의 6 행)
    B = B_q (float32) × B_scale, c = float32(c_int × float64(c_step)), a = float32(a_int × float64(a_step))
    """
    raise NotImplementedError("TODO: parse_pose4")


def affine(img: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """img (n,3,SH,SW), a (n,6) → bilinear 재샘플 (padding border, align_corners=False).
    a' = a × [0.01, 0.01, 2/512, 0.01, 0.01, 2/384], θ = [[1+a'0, a'1, a'2], [a'3, 1+a'4, a'5]]."""
    raise NotImplementedError("TODO: affine")


def even_frames(prev: torch.Tensor, a: torch.Tensor, c: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """prev (n,3,SH,SW) 이전 쌍 홀수 프레임 (첫 쌍은 렌더러(M_pre)), a (n,6), c (n,k), B (k,1,bh,bw)
    → clamp(affine(prev, a) + bicubic 확대(Σ_k c_k B_k → (SH,SW)), 0, 255)  (n,3,SH,SW). 기저는 회색이라 RGB 에 같은 값."""
    raise NotImplementedError("TODO: even_frames")
