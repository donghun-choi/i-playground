"""여러 섹션이 같이 쓰는 엔트로피 복호화 도구 (GUIDE 3 머리말, 3.6)."""

from __future__ import annotations

import numpy as np


def unxz(data: bytes) -> bytes:
    """lzma raw (LZMA2, preset 9 | PRESET_EXTREME) 풀기. 압축은 lab 쪽 lzma.compress(..., format=FORMAT_RAW, filters=...)."""
    raise NotImplementedError("TODO: unxz")


def laplace_probs(r: int, qmax: int) -> np.ndarray:
    """정수 q ∈ [-qmax, qmax] 의 이산 라플라스 확률 (2·qmax+1,) float64, C 연속 (GUIDE 3.6).

    f(0) = 2^24, f(j+1) = max(1, (f(j)·r) >> 8) 를 정수로, P(q) = f(|q|) / Σ f.  같은 (r, qmax) 는 자주 나오니 캐시해도 된다.
    """
    raise NotImplementedError("TODO: laplace_probs")


class Decoder:
    """range coder 단어열 하나를 앞에서부터 차례로 푸는 복호기.

    constriction 을 쓴다면: constriction.stream.queue.RangeDecoder(np.uint32 배열),
    모델 family = constriction.stream.model.Categorical(perfect=False), dec.decode(family, probs (m,K)) → (m,) 기호.
    probs 는 반드시 C 연속 float64 (constriction 은 stride 를 무시한다).
    """

    def __init__(self, words: bytes | np.ndarray):
        """words: little-endian u32 단어들 (bytes 면 np.frombuffer(..., np.uint32))."""
        raise NotImplementedError("TODO: Decoder.__init__")

    def decode(self, probs: np.ndarray) -> np.ndarray:
        """probs (m,K) float64 → 기호 (m,) 정수. 기호 i 의 확률이 probs[i]."""
        raise NotImplementedError("TODO: Decoder.decode")

    def laplace_rows(self, qmax: int, r_bytes: bytes, m: int) -> np.ndarray:
        """행마다 m 개, 행 i 의 확률 = laplace_probs(r_bytes[i], qmax), 기호 - qmax 가 값 → (len(r_bytes), m) int64.
        qmax == 0 이면 아무것도 읽지 않고 0 행렬."""
        raise NotImplementedError("TODO: Decoder.laplace_rows")
