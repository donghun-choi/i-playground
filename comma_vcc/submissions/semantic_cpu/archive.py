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
# 헤더: arch(1 = v1 Renderer, 2 = RendererV2, 3 = 폭을 바꾼 v1, 4 = 폭을 바꾼 v1 + 프레임별 FiLM), v2 이면 width, fdim, n_frames, dilation 4개,
# 3 이면 c1 c2 c3, 4 이면 c1 c2 c3 fdim n_frames
def pack_renderer(cfg, weights: bytes) -> bytes:
    """weights: pack_state 바이트 (xz 로 압축) 또는 wcodec 엔트로피 부호 (그대로, 헤더 코드 +10)."""
    import wcodec

    if wcodec.is_ec(weights):
        assert cfg and len(cfg) in (3, 5)
        return struct.pack(f"<B{len(cfg)}H", 13 if len(cfg) == 3 else 14, *cfg) + weights
    if not cfg:
        head = struct.pack("<B", 1)
    elif len(cfg) == 3:
        head = struct.pack("<B3H", 3, *cfg)
    elif len(cfg) == 5:
        head = struct.pack("<B5H", 4, *cfg)
    else:
        width, fdim, n_frames, dils = cfg
        head = struct.pack("<BHBH4B", 2, width, fdim, n_frames, *dils)
    return head + xz(weights)


def unpack_renderer(buf: bytes):
    """→ (cfg, weights bytes)"""
    if buf[0] == 1:
        return None, unxz(buf[1:])
    if buf[0] == 3:
        return tuple(struct.unpack_from("<3H", buf, 1)), unxz(buf[struct.calcsize("<B3H"):])
    if buf[0] == 4:
        return tuple(struct.unpack_from("<5H", buf, 1)), unxz(buf[struct.calcsize("<B5H"):])
    if buf[0] in (13, 14):  # wcodec 엔트로피 부호 (unpack_state 전에 wcodec.ec_unpack 으로 되돌린다)
        nc = 3 if buf[0] == 13 else 5
        return tuple(struct.unpack_from(f"<{nc}H", buf, 1)), buf[struct.calcsize(f"<B{nc}H"):]
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


# ---------------------------------------------------------------- pose v3: pose v2 와 같은 값, 계수만 Rice 부호
# 계수는 시간 상관이 거의 없어서 (차분하면 오히려 분산이 2배) 차분 + xz 대신 행별 (값 - 평균) 을 Rice 부호로 쓴다.
def rice_encode(rows: np.ndarray) -> bytes:
    """(r, n) 정수 → 행마다 [i32 평균][u8 k] + 비트열 (zigzag → q 개의 1, 0, k 비트 나머지)."""
    head, bits = [], []
    for row in rows.astype(np.int64):
        m = int(np.round(row.mean()))
        v = row - m
        z = np.where(v >= 0, 2 * v, -2 * v - 1)
        k = min(range(16), key=lambda kk: int(((z >> kk) + 1 + kk).sum()))
        head.append(struct.pack("<iB", m, k))
        for x in z.tolist():
            bits.extend([1] * (x >> k) + [0] + [(x >> (k - 1 - j)) & 1 for j in range(k)])
    return b"".join(head) + np.packbits(np.array(bits, np.uint8)).tobytes()


def rice_decode(buf: bytes, r: int, n: int) -> np.ndarray:
    hs = struct.calcsize("<iB")
    params = [struct.unpack_from("<iB", buf, i * hs) for i in range(r)]
    bits = np.unpackbits(np.frombuffer(buf, np.uint8, offset=r * hs)).tolist()
    out = np.zeros((r, n), np.int64)
    pos = 0
    for i, (m, k) in enumerate(params):
        for j in range(n):
            q = 0
            while bits[pos]:
                q += 1
                pos += 1
            pos += 1
            rem = 0
            for _ in range(k):
                rem = (rem << 1) | bits[pos]
                pos += 1
            z = (q << k) | rem
            out[i, j] = m + (z >> 1 if z % 2 == 0 else -((z + 1) >> 1))
    return out


def pose2_to_pose3(blob2: bytes) -> bytes:
    """pose2 blob → pose3 blob (무손실 재포장)."""
    n, k, C, bh, bw = struct.unpack_from("<HHBHH", blob2, 0)
    hl = struct.calcsize("<HHBHH")
    body = unxz(blob2[hl:])
    fixed = 4 * (k + 6 + k) + k * C * bh * bw
    d = np.frombuffer(body, np.int16, (k + 6) * n, fixed).reshape(k + 6, n)
    vals = np.cumsum(d.astype(np.int64), axis=1)
    xzpart = xz(body[:fixed])
    return blob2[:hl] + struct.pack("<I", len(xzpart)) + xzpart + rice_encode(vals)


def unpack_pose3(buf: bytes):
    n, k, C, bh, bw = struct.unpack_from("<HHBHH", buf, 0)
    hl = struct.calcsize("<HHBHH")
    (lx,) = struct.unpack_from("<I", buf, hl)
    body = unxz(buf[hl + 4 : hl + 4 + lx])
    vals = rice_decode(buf[hl + 4 + lx :], k + 6, n)
    c_step = np.frombuffer(body, np.float32, k, 0)
    a_step = np.frombuffer(body, np.float32, 6, 4 * k)
    B_scale = np.frombuffer(body, np.float32, k, 4 * (k + 6))
    B_q = np.frombuffer(body, np.int8, k * C * bh * bw, 4 * (2 * k + 6)).reshape(k, C, bh, bw)
    c_int, a_int = vals[:k].T, vals[k:].T
    return {
        "B": torch.from_numpy(B_q.astype(np.float32) * B_scale.reshape(k, 1, 1, 1)),
        "c": torch.from_numpy((c_int * c_step.astype(np.float64)).astype(np.float32)),
        "a": torch.from_numpy((a_int * a_step.astype(np.float64)).astype(np.float32)),
    }
