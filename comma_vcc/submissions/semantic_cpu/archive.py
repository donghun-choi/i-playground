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
# 헤더: kind(0 학습 기저 / 1 DCT), base(0 odd / 1 gray), mode(0 bilinear / 1 bicubic), n, k, C, bh, bw
# 학습 기저 B (k,C,bh,bw): int8 + 기저별 스케일.  계수 c (n,k): 차원별 간격 step[k] 의 정수배, 시간 차분 후 int16.
BASES = ("odd", "gray")
MODES = ("bilinear", "bicubic")


def pack_carrier(kind, base, mode, B_q, B_scale, c_int, step, C, bh, bw) -> bytes:
    n, k = c_int.shape
    head = struct.pack("<BBBHHBHH", kind, BASES.index(base), MODES.index(mode), n, k, C, bh, bw)
    body = [step.astype(np.float32).tobytes()]
    if kind == 0:
        body += [B_scale.astype(np.float32).tobytes(), B_q.astype(np.int8).tobytes()]
    d = np.diff(c_int, axis=0, prepend=np.zeros((1, k), c_int.dtype)).astype(np.int16)
    body.append(np.ascontiguousarray(d.T).tobytes())  # 차원 우선 (lzma 가 잘 줄이도록)
    return head + xz(b"".join(body))


_HEAD = struct.calcsize("<BBBHHBHH")


def unpack_carrier(buf: bytes):
    """→ dict(B, c, base, mode)"""
    kind, base, mode, n, k, C, bh, bw = struct.unpack_from("<BBBHHBHH", buf, 0)
    body = unxz(buf[_HEAD:])
    off = 0
    step = np.frombuffer(body, np.float32, k, off)
    off += 4 * k
    if kind == 0:
        scale = np.frombuffer(body, np.float32, k, off)
        off += 4 * k
        B_q = np.frombuffer(body, np.int8, k * C * bh * bw, off).reshape(k, C, bh, bw)
        off += k * C * bh * bw
        B = torch.from_numpy(B_q.astype(np.float32) * scale.reshape(k, 1, 1, 1))
    else:
        B = dct_basis(k, bh, bw)
    d = np.frombuffer(body, np.int16, n * k, off).reshape(k, n).T
    c_int = np.cumsum(d.astype(np.int64), axis=0)
    c = torch.from_numpy((c_int * step.astype(np.float64)).astype(np.float32))
    return {"B": B, "c": c, "base": BASES[base], "mode": MODES[mode]}


def dct_basis(k: int, bh: int, bw: int) -> torch.Tensor:
    """저주파부터 k 개의 2D DCT-II 기저 (k,1,bh,bw), 최대 진폭 30."""
    freqs = sorted(((u, v) for u in range(bh) for v in range(bw)), key=lambda f: (f[0] + f[1], f[0]))[:k]
    y = (torch.arange(bh, dtype=torch.float64) + 0.5) / bh
    x = (torch.arange(bw, dtype=torch.float64) + 0.5) / bw
    B = torch.stack([(torch.cos(torch.pi * u * y)[:, None] * torch.cos(torch.pi * v * x)[None, :])[None] for u, v in freqs])
    return (B * 30).float()


# ---------------------------------------------------------------- 렌더러
# 헤더: arch(1 = v1 Renderer, 2 = RendererV2), v2 이면 width, fdim, n_frames, dilation 4개
def pack_renderer(cfg, weights: bytes) -> bytes:
    if not cfg:
        head = struct.pack("<B", 1)
    else:
        width, fdim, n_frames, dils = cfg
        head = struct.pack("<BHBH4B", 2, width, fdim, n_frames, *dils)
    return head + xz(weights)


def unpack_renderer(buf: bytes):
    """→ (cfg, weights bytes)"""
    if buf[0] == 1:
        return None, unxz(buf[1:])
    _, width, fdim, n_frames, *dils = struct.unpack_from("<BHBH4B", buf, 0)
    return (width, fdim, n_frames, tuple(dils)), unxz(buf[struct.calcsize("<BHBH4B"):])


# ---------------------------------------------------------------- pose v2: 이전 렌더 + 아핀 + carrier
# 헤더: n, k, C, bh, bw. 본문(xz): c 간격(k) + a 간격(6) + B 스케일(k) + B int8 + c int16 차분 + a int16 차분
def pack_pose2(B_q, B_scale, c_int, c_step, a_int, a_step) -> bytes:
    n, k = c_int.shape
    _, C, bh, bw = B_q.shape
    head = struct.pack("<HHBHH", n, k, C, bh, bw)

    def deltas(x):
        d = np.diff(x, axis=0, prepend=np.zeros((1, x.shape[1]), x.dtype))
        assert np.abs(d).max() < 32768
        return np.ascontiguousarray(d.astype(np.int16).T).tobytes()

    body = [c_step.astype(np.float32).tobytes(), a_step.astype(np.float32).tobytes(),
            B_scale.astype(np.float32).tobytes(), B_q.astype(np.int8).tobytes(), deltas(c_int), deltas(a_int)]
    return head + xz(b"".join(body))


def unpack_pose2(buf: bytes):
    n, k, C, bh, bw = struct.unpack_from("<HHBHH", buf, 0)
    body = unxz(buf[struct.calcsize("<HHBHH"):])
    off = 0

    def take(dtype, count):
        nonlocal off
        x = np.frombuffer(body, dtype, count, off)
        off += x.nbytes
        return x

    c_step, a_step, B_scale = take(np.float32, k), take(np.float32, 6), take(np.float32, k)
    B_q = take(np.int8, k * C * bh * bw).reshape(k, C, bh, bw)
    c_int = np.cumsum(take(np.int16, n * k).reshape(k, n).T.astype(np.int64), axis=0)
    a_int = np.cumsum(take(np.int16, n * 6).reshape(6, n).T.astype(np.int64), axis=0)
    return {
        "B": torch.from_numpy(B_q.astype(np.float32) * B_scale.reshape(k, 1, 1, 1)),
        "c": torch.from_numpy((c_int * c_step.astype(np.float64)).astype(np.float32)),
        "a": torch.from_numpy((a_int * a_step.astype(np.float64)).astype(np.float32)),
    }
