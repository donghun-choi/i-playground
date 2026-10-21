"""archive 의 파일 'p' 풀기 (GUIDE 3.1)."""

from __future__ import annotations


def read_sections(p: bytes) -> dict[str, bytes]:
    """b"SCP1" + [이름 4바이트][u32 길이 (little-endian)][데이터] 나열 → {이름: 데이터}.

    v6.26 은 ctxe, segs, rend, pos4 네 개. 맨 앞 4바이트가 b"SCP1" 이 아니면 에러.
    """
    raise NotImplementedError("TODO: read_sections")
