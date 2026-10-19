"""archive.zip 안의 파일 'p' 형식: b"SCP1" + 섹션 [이름 4바이트][u32 길이][데이터] 나열.

섹션: ctxe 문맥 CNN (wcodec.ec_pack_qnet) / segs seg 맵 스트림 (segcodec.encode)
      rend 렌더러 (아래) / pos4 pose 파라미터 (아래).  바이트 단위 명세는 comma_vcc/GUIDE.md
"""

from __future__ import annotations

import struct

import numpy as np
import torch

import wcodec

MAGIC = b"SCP1"


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


# ---------------------------------------------------------------- rend: [u8 13][u16 c1][u16 c2][u16 c3] + wcodec.ec_pack(pack_state 바이트)
REND_CODE = 13  # (예전 형식 번호를 피해 13: 폭 3개 + 엔트로피 부호 가중치)


def pack_renderer(cfg, ec: bytes) -> bytes:
    return struct.pack("<B3H", REND_CODE, *cfg) + ec


def unpack_renderer(buf: bytes):
    """→ (폭 (c1,c2,c3), wcodec 바이트)"""
    assert buf[0] == REND_CODE, "예전 렌더러 형식은 지원하지 않는다"
    return tuple(struct.unpack_from("<3H", buf, 1)), buf[struct.calcsize("<B3H"):]


# ---------------------------------------------------------------- pos4: pose 파라미터
# [u16 n][u16 k][u8 C][u16 bh][u16 bw] + f32 c_step (k) + f32 a_step (6) + f32 B_scale (k)
#   + wcodec.ec_pack_rows(B_q (k·C, bh·bw)) + Rice(계수 정수 (k+6, n): c 의 k 행, 그다음 a 의 6 행)
# 값: B = B_q · B_scale (기저마다),  c = c_int · c_step,  a = a_int · a_step (차원마다)
POSE_HEAD = "<HHBHH"


def rice_encode(rows: np.ndarray) -> bytes:
    """(r, n) 정수 → 행마다 [i32 평균 m][u8 k] + 비트열 (MSB 먼저, 마지막 바이트는 0 으로 채움).
    값 v - m 을 zigzag (z = 2v 또는 -2v-1) → (z >> k) 개의 1, 0 하나, z 의 아래 k 비트. k 는 그 행에서 가장 짧은 것."""
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


def pack_pose4(B_q, B_scale, c_int, c_step, a_int, a_step) -> bytes:
    """B_q (k,C,bh,bw) 정수, B_scale (k,), c_int (n,k), c_step (k,), a_int (n,6), a_step (6,)"""
    n, k = c_int.shape
    _, C, bh, bw = B_q.shape
    steps = [np.asarray(x).astype(np.float32).tobytes() for x in (c_step, a_step, B_scale)]
    vals = np.concatenate([np.asarray(c_int).T, np.asarray(a_int).T]).astype(np.int64)
    return (struct.pack(POSE_HEAD, n, k, C, bh, bw) + b"".join(steps)
            + wcodec.ec_pack_rows(np.asarray(B_q).reshape(k * C, bh * bw)) + rice_encode(vals))


def unpack_pose4(buf: bytes) -> dict:
    """→ {"B": (k,C,bh,bw), "c": (n,k), "a": (n,6)} float32 텐서"""
    n, k, C, bh, bw = struct.unpack_from(POSE_HEAD, buf, 0)
    hl = struct.calcsize(POSE_HEAD)
    c_step = np.frombuffer(buf, np.float32, k, hl)
    a_step = np.frombuffer(buf, np.float32, 6, hl + 4 * k)
    B_scale = np.frombuffer(buf, np.float32, k, hl + 4 * (k + 6))
    B_q, off = wcodec.ec_unpack_rows(buf, k * C, bh * bw, hl + 4 * (2 * k + 6))
    vals = rice_decode(buf[off:], k + 6, n)
    c_int, a_int = vals[:k].T, vals[k:].T
    return {
        "B": torch.from_numpy(B_q.reshape(k, C, bh, bw).astype(np.float32) * B_scale.reshape(k, 1, 1, 1)),
        "c": torch.from_numpy((c_int * c_step.astype(np.float64)).astype(np.float32)),
        "a": torch.from_numpy((a_int * a_step.astype(np.float64)).astype(np.float32)),
    }
