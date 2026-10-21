"""512x384 float 프레임 → 1164x874 uint8 서브픽셀 확장 (GUIDE 4.7)."""

from __future__ import annotations

import numpy as np

SH, SW = 384, 512
H, W = 874, 1164


def expand_fine(x: np.ndarray) -> np.ndarray:
    """x (n,SH,SW,3) float32 0..255 → (n,H,W,3) uint8.

    평가 코드의 bilinear 축소 (H,W → SH,SW, align_corners=False) 에서 작은 이미지 한 칸은 원본 2x2 칸의 가중합이고 칸끼리 겹치지 않는다.
    칸마다 base = clamp(floor(x), 1, 253) 에 b ∈ {-1,0,1,2}^4 (256 조합) 를 더한 2x2 중 가중합이 x 에 가장 가까운 것을 쓴다.
    축소가 읽지 않는 원본 행·열은 가까운 읽히는 행·열을 복사. (가중치·조합 표는 한 번만 만들어 둔다)
    """
    raise NotImplementedError("TODO: expand_fine")
