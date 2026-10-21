"""정수 문맥 CNN: ctxe 섹션 풀기, 층 읽기, 정수 산술, 확률표 (GUIDE 3.2, 4.2)."""

from __future__ import annotations

import numpy as np

# 층 하나 = (W int64 (o,i,k,k), B int64 (o,), M int64 (o,), dilation int, relu bool)
Layer = tuple


def unpack_ctxe(ctxe: bytes) -> bytes:
    """ctxe 섹션 → QNet 바이트 ([u8 층 수] + 층마다 [7바이트 헤더][W int8][B int32][M int32]).

    [u32 L] + xz(헤더·qmax·r·B·M) + range coder (W). 층마다 W 를 laplace_rows(qmax, r, i·k·k) 로 (o 행).
    """
    raise NotImplementedError("TODO: unpack_ctxe")


def parse_qnet(buf: bytes) -> list[Layer]:
    """QNet 바이트 → 층 목록. 첫 층 relu 바이트의 비트 1·2 (플래그, 값 6) 는 떼어 내고 켜져 있는지 확인."""
    raise NotImplementedError("TODO: parse_qnet")


class IntCNN:
    """정수 CNN. 층마다 acc = conv(x, W) (정확한 정수), y = floor(f32(f32(acc + B) × f32(M·2^-16))), relu 층은 clamp(0, 255)."""

    def __init__(self, layers: list[Layer]):
        raise NotImplementedError("TODO: IntCNN.__init__")

    def self_check(self) -> bool:
        """이 기계의 float32 합성곱이 정확한지 (레벨별 격자 크기마다) float64 와 비교. 틀리면 float64 합성곱으로 전환하고 False."""
        raise NotImplementedError("TODO: IntCNN.self_check")

    def run(self, x) -> np.ndarray:
        """x: (n,23,gh,gw) 정수값 (float32 numpy 또는 torch) → logit (n,5,gh,gw) 정수값 (1/16 nat 단위)."""
        raise NotImplementedError("TODO: IntCNN.run")


def probs(logits) -> np.ndarray:
    """logits (m,5) 정수값 → 확률 (m,5) float64, C 연속.

    d = min(행 최댓값 - logit, 255), freq = FREQ[d] (FREQ[d] = max(1, round(2^20·exp(-d/16)))), p = freq / Σ freq.
    """
    raise NotImplementedError("TODO: probs")
