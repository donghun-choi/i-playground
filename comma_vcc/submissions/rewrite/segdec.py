"""seg 맵 스트림 복호화: 패스 마스크, 23채널 입력, stride-32 카운트 모델, 복호화 루프 (GUIDE 3.3, 4.1, 4.3, 4.4)."""

from __future__ import annotations

import numpy as np

from ctxnet import IntCNN

SH, SW = 384, 512
LEVELS = (32, 16, 8, 4, 2)  # 레벨 s, 격자 간격 h = s/2
PASSES = ("A1", "A2", "B")
K = 5


def masks(h: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """stride-h 격자 (SH/h, SW/h) 에서 패스 kind 의 (아는 칸, 보낼 칸) bool 배열.

    even = (i, j 둘 다 짝수), odd = (둘 다 홀수)
    A1: 보낼 = odd & (i%4 == j%4), 아는 = even / A2: 보낼 = odd & ~A1, 아는 = even | A1 / B: 보낼 = ~(even|odd), 아는 = even|odd
    """
    raise NotImplementedError("TODO: masks")


def build_input(cur: np.ndarray, s: int, kind: str, prev: np.ndarray | None, prev2: np.ndarray | None):
    """문맥 CNN 입력 (23, SH/h, SW/h), 값은 0..64 정수 (float32).

    cur: 지금 맵 (SH,SW) uint8 (보낼 칸·아직 모르는 칸 값은 무엇이든 결과가 같아야 한다)
    prev, prev2: 이전·전전 맵 (없으면 None → 그 채널 0)
    채널: cur one-hot × 아는 칸 × 64 (5) | 아는 칸 × 64 (1) | prev 클래스 비율 (5) | prev2 클래스 비율 (5)
          | A 패스 64 (1) | 레벨 one-hot × 64 (5) | A2 패스 64 (1)
    클래스 비율 = floor(격자점이 왼쪽 위인 h×h 블록 안 그 클래스 픽셀 수 × 64 / h²)
    (decode_maps 에서는 프레임마다 비율을 한 번만 계산해 두는 게 빠르다. 이 함수 모양은 검사용.)
    """
    raise NotImplementedError("TODO: build_input")


class CoarseCounts:
    """stride-32 격자 192칸의 적응형 카운트. 표 (6,5) 처음 전부 1, 문맥 = 이전 맵의 같은 칸 클래스 (이전 맵이 없으면 5)."""

    def __init__(self):
        raise NotImplementedError("TODO: CoarseCounts.__init__")

    def probs(self, ctx: np.ndarray) -> np.ndarray:
        """ctx (192,) → (192,5) float64 = 표[ctx] / 행 합."""
        raise NotImplementedError("TODO: CoarseCounts.probs")

    def update(self, ctx: np.ndarray, sym: np.ndarray) -> None:
        """맵 하나를 다 보낸 뒤 칸마다 표[ctx][sym] += 16 (같은 (ctx, sym) 이 여러 번이면 여러 번 더한다)."""
        raise NotImplementedError("TODO: CoarseCounts.update")


def decode_maps(segs: bytes, net: IntCNN, n_max: int | None = None) -> np.ndarray:
    """segs 섹션 → seg 맵 (n,SH,SW) uint8. n = 헤더의 u32 (601), n_max 를 주면 앞 n_max 장만 (검사용).

    맵마다: stride-32 격자 192칸 (행 우선, CoarseCounts) → 레벨 32..2 × 패스 A1, A2, B 마다
    build_input → net.run → 보낼 칸 (행 우선) 의 확률 (ctxnet.probs) 로 한 번에 복호화해서 맵에 써 넣는다.
    """
    raise NotImplementedError("TODO: decode_maps")
