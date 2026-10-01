"""archive 의 'p' 파일 포맷: 섹션 [이름 4바이트][u32 길이][데이터] 의 나열."""

from __future__ import annotations

import lzma
import struct

import numpy as np
import torch

MAGIC = b"SCP1"
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


def pack(sections: dict[str, bytes]) -> bytes:
    out = [MAGIC]
    for name, data in sections.items():
        assert len(name) == 4
        out.append(name.encode() + struct.pack("<I", len(data)) + data)
    return b"".join(out)


def unpack(buf: bytes) -> dict[str, bytes]:
    assert buf[:4] == MAGIC, "알 수 없는 archive"
    off, out = 4, {}
    while off < len(buf):
        name = buf[off : off + 4].decode()
        (n,) = struct.unpack_from("<I", buf, off + 4)
        out[name] = buf[off + 8 : off + 8 + n]
        off += 8 + n
    return out


def xz(data: bytes) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)


def unxz(data: bytes) -> bytes:
    return lzma.decompress(data, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)


# ---------------------------------------------------------------- carrier
# kind 0: 학습 기저 B (k,1,bh,bw) int8 + 기저별 스케일,  kind 1: 고정 DCT 기저 (저장 안 함)
# 계수 c (n,k): 차원별 양자화 간격 step[k] 의 정수배. 시간 방향 차분 후 int16 로 저장.
def pack_carrier(kind: int, B_q: np.ndarray | None, B_scale: np.ndarray | None, c_int: np.ndarray, step: np.ndarray, bh: int, bw: int) -> bytes:
    n, k = c_int.shape
    head = struct.pack("<BHHHH", kind, n, k, bh, bw)
    body = [step.astype(np.float32).tobytes()]
    if kind == 0:
        body += [B_scale.astype(np.float32).tobytes(), B_q.astype(np.int8).tobytes()]
    d = np.diff(c_int, axis=0, prepend=np.zeros((1, k), c_int.dtype)).astype(np.int16)
    # 차원 우선 (같은 차원 계수끼리 붙어 있어야 lzma 가 잘 줄인다)
    body.append(np.ascontiguousarray(d.T).tobytes())
    return head + xz(b"".join(body))


def unpack_carrier(buf: bytes):
    kind, n, k, bh, bw = struct.unpack_from("<BHHHH", buf, 0)
    body = unxz(buf[9:])
    off = 0
    step = np.frombuffer(body, np.float32, k, off)
    off += 4 * k
    B = None
    if kind == 0:
        scale = np.frombuffer(body, np.float32, k, off)
        off += 4 * k
        B_q = np.frombuffer(body, np.int8, k * bh * bw, off).reshape(k, 1, bh, bw)
        off += k * bh * bw
        B = torch.from_numpy(B_q.astype(np.float32) * scale.reshape(k, 1, 1, 1))
    d = np.frombuffer(body, np.int16, n * k, off).reshape(k, n).T
    c_int = np.cumsum(d.astype(np.int64), axis=0)
    c = torch.from_numpy((c_int * step.astype(np.float64)).astype(np.float32))
    if kind == 1:
        B = dct_basis(k, bh, bw)
    return B, c


def dct_basis(k: int, bh: int, bw: int) -> torch.Tensor:
    """저주파부터 k 개의 2D DCT-II 기저 (k,1,bh,bw), 최대 진폭 30."""
    freqs = sorted(((u, v) for u in range(bh) for v in range(bw)), key=lambda f: (f[0] + f[1], f[0]))[:k]
    y = (torch.arange(bh, dtype=torch.float64) + 0.5) / bh
    x = (torch.arange(bw, dtype=torch.float64) + 0.5) / bw
    B = torch.stack([(torch.cos(torch.pi * u * y)[:, None] * torch.cos(torch.pi * v * x)[None, :])[None] for u, v in freqs])
    return (B * 30).float()
