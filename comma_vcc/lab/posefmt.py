"""pose 중간 형식 (pose_fit / pose_refine 의 결과 파일, archive 에는 pos4 로 다시 포장해서 넣는다).

pose2 blob: [u16 n][u16 k][u8 C][u16 bh][u16 bw] + xz( f32 c_step (k) + f32 a_step (6) + f32 B_scale (k)
            + int8 B_q (k,C,bh,bw) + int16 c 차분 (k 행 × n, 시간 차분) + int16 a 차분 (6 행 × n) )
"""

from __future__ import annotations

import lzma
import struct

import numpy as np
import torch

import archive

HEAD = "<HHBHH"
_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


def xz(data: bytes) -> bytes:
    return lzma.compress(data, format=lzma.FORMAT_RAW, filters=_FILTERS)


def unxz(data: bytes) -> bytes:
    return lzma.decompress(data, format=lzma.FORMAT_RAW, filters=_FILTERS)


def pack_pose2(B_q, B_scale, c_int, c_step, a_int, a_step) -> bytes:
    n, k = c_int.shape
    _, C, bh, bw = B_q.shape

    def deltas(x):
        d = np.diff(x, axis=0, prepend=np.zeros((1, x.shape[1]), x.dtype))
        assert np.abs(d).max() < 32768
        return np.ascontiguousarray(d.astype(np.int16).T).tobytes()

    body = [c_step.astype(np.float32).tobytes(), a_step.astype(np.float32).tobytes(),
            B_scale.astype(np.float32).tobytes(), B_q.astype(np.int8).tobytes(), deltas(c_int), deltas(a_int)]
    return struct.pack(HEAD, n, k, C, bh, bw) + xz(b"".join(body))


def read_pose2(buf: bytes) -> dict:
    """→ 정수·격자 그대로: B_q (k,C,bh,bw) int8, B_scale/c_step/a_step float32, c_int (n,k)/a_int (n,6) int64"""
    n, k, C, bh, bw = struct.unpack_from(HEAD, buf, 0)
    body = unxz(buf[struct.calcsize(HEAD):])
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
    return {"B_q": B_q, "B_scale": B_scale, "c_int": c_int, "c_step": c_step, "a_int": a_int, "a_step": a_step}


def unpack_pose2(buf: bytes) -> dict:
    """→ {"B", "c", "a"} float32 텐서 (archive.unpack_pose4 와 같은 값)"""
    r = read_pose2(buf)
    k = len(r["c_step"])
    return {
        "B": torch.from_numpy(r["B_q"].astype(np.float32) * r["B_scale"].reshape(k, 1, 1, 1)),
        "c": torch.from_numpy((r["c_int"] * r["c_step"].astype(np.float64)).astype(np.float32)),
        "a": torch.from_numpy((r["a_int"] * r["a_step"].astype(np.float64)).astype(np.float32)),
    }


def pose2_to_pose4(buf: bytes) -> bytes:
    """pose2 blob → archive 의 pos4 섹션 (같은 값, 무손실)."""
    r = read_pose2(buf)
    return archive.pack_pose4(r["B_q"], r["B_scale"], r["c_int"], r["c_step"], r["a_int"], r["a_step"])
