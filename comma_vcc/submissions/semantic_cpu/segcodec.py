"""seg 맵 무손실 코덱 (제출물에 그대로 들어가는 자체 완결 모듈: numpy, torch, constriction 만 사용).

순서: 프레임마다 stride-32 격자(coarse) → 레벨 s=32..2 에서 A 패스, B 패스.
확률: coarse 는 적응형 정수 카운트, 나머지는 정수 CNN (float64 로 계산하지만 값이 전부 정수라 결과가 기계와 무관).

정수 CNN 규약
  입력 x_q: 0..64 정수 (이진 채널은 0/64, 클래스 비율은 floor(count*64/h^2))
  층마다: acc = conv(x_q, W_q) (정확한 정수) → a_q = clamp(((acc + B_q) * M) >> S, 0, 255)  (ReLU 포함)
  마지막 층: logit_q = ((acc + B_q) * M) >> S  (1/16 nat 단위)
  확률: freq_c = EXP_TABLE[min(max_logit - logit_c, 255)]
"""

from __future__ import annotations

import math
import struct

import numpy as np
import torch
import torch.nn.functional as F

SH, SW = 384, 512
LEVELS = [32, 16, 8, 4, 2]
K = 5
C_IN = 22
Q_IN = 64  # 입력 스케일
LOGIT_UNIT = 16  # logit 1 nat = 16
SHIFT = 16
EXP_TABLE = np.array([max(1, round(32768 * math.exp(-d / LOGIT_UNIT))) for d in range(256)], dtype=np.int64)


# ---------------------------------------------------------------- 입력
def onehot_np(m: np.ndarray) -> np.ndarray:
    """(n,H,W) uint8 → (n,5,H,W) int64 0/1"""
    return (m[:, None, :, :] == np.arange(K, dtype=m.dtype)[None, :, None, None]).astype(np.int64)


def frac_q(m: np.ndarray | None, h: int, n: int) -> np.ndarray:
    """h×h 블록의 클래스 개수 → floor(count*64/h^2). (n,5,SH/h,SW/h) int64"""
    gh, gw = SH // h, SW // h
    if m is None:
        return np.zeros((n, K, gh, gw), np.int64)
    oh = onehot_np(m)
    cnt = oh.reshape(n, K, gh, h, gw, h).sum((3, 5))
    return (cnt * Q_IN) // (h * h)


def masks(h: int, kind: str):
    gi = (np.arange(SH // h) % 2)[:, None]
    gj = (np.arange(SW // h) % 2)[None, :]
    if kind == "A":
        known, target = (gi == 0) & (gj == 0), (gi == 1) & (gj == 1)
    else:
        known, target = gi == gj, gi != gj
    return known, target


def build_input_q(cur: np.ndarray, prev, prev2, s: int, kind: str) -> np.ndarray:
    """cur: (n,384,512) uint8 (target 칸은 아무 값이어도 됨). → (n,22,gh,gw) int64 (0..64)"""
    h = s // 2
    n = cur.shape[0]
    g = cur[:, ::h, ::h]
    known, _ = masks(h, kind)
    gh, gw = known.shape
    lvl = np.zeros((n, len(LEVELS), gh, gw), np.int64)
    lvl[:, LEVELS.index(s)] = Q_IN
    return np.concatenate([
        onehot_np(g) * known * Q_IN,
        np.broadcast_to(known * Q_IN, (n, 1, gh, gw)),
        frac_q(prev, h, n),
        frac_q(prev2, h, n),
        np.full((n, 1, gh, gw), Q_IN if kind == "A" else 0, np.int64),
        lvl,
    ], 1)


# ---------------------------------------------------------------- 정수 네트워크
class QNet:
    """layers: [(W_q int (o,i,kh,kw), B_q int (o,), M int (o,), pad, relu)]"""

    def __init__(self, layers):
        self.layers = layers
        # float32 합성곱이 정확하려면 모든 부분합의 절댓값이 2^24 미만이어야 한다
        in_max = Q_IN
        for W, _, _, _, relu in layers:
            assert in_max * np.abs(W).sum((1, 2, 3)).max() < 2**24, "float32 정확 범위 초과"
            in_max = 255
        self._t = [(torch.from_numpy(W.astype(np.float32)), torch.from_numpy(B).view(1, -1, 1, 1),
                    torch.from_numpy(M).view(1, -1, 1, 1), pad, relu) for W, B, M, pad, relu in layers]
        self.dtype = torch.float32

    def self_check(self, x_q: np.ndarray) -> bool:
        """이 기계의 float32 합성곱이 정확한지 float64 결과와 비교. 아니면 float64 로 전환."""
        self.dtype = torch.float64
        ref = self(x_q)
        self.dtype = torch.float32
        ok = np.array_equal(ref, self(x_q))
        if not ok:
            self.dtype = torch.float64
        return ok

    def __call__(self, x_q: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(x_q).to(self.dtype)
        for W, B, M, pad, relu in self._t:
            acc = F.conv2d(x, W.to(self.dtype), padding=pad)
            acc = acc.round().to(torch.int64)  # 정확한 정수
            y = ((acc + B) * M) >> SHIFT
            if relu:
                y = y.clamp(0, 255)
            x = y.to(self.dtype)
        return x.to(torch.int64).numpy()

    # 직렬화: 층 수, 각 층 (o,i,k,pad,relu) + W(int8) + B(int32) + M(int32)
    def to_bytes(self) -> bytes:
        out = [struct.pack("<B", len(self.layers))]
        for W, B, M, pad, relu in self.layers:
            o, i, k, _ = W.shape
            out.append(struct.pack("<HHBBB", o, i, k, pad, int(relu)))
            out.append(W.astype(np.int8).tobytes())
            out.append(B.astype(np.int32).tobytes())
            out.append(M.astype(np.int32).tobytes())
        return b"".join(out)

    @classmethod
    def from_bytes(cls, buf: bytes, off: int = 0):
        (n,) = struct.unpack_from("<B", buf, off)
        off += 1
        layers = []
        for _ in range(n):
            o, i, k, pad, relu = struct.unpack_from("<HHBBB", buf, off)
            off += 7
            W = np.frombuffer(buf, np.int8, o * i * k * k, off).reshape(o, i, k, k).astype(np.int64)
            off += o * i * k * k
            B = np.frombuffer(buf, np.int32, o, off).astype(np.int64)
            off += 4 * o
            M = np.frombuffer(buf, np.int32, o, off).astype(np.int64)
            off += 4 * o
            layers.append((W, B, M, pad, bool(relu)))
        return cls(layers), off


def quantize_ctxnet(state_dict: dict, calib_inputs: list[np.ndarray], wbits: int = 8) -> QNet:
    """float CtxNet (Conv-ReLU 반복 + 마지막 1x1) → QNet. calib_inputs: build_input_q 결과 몇 개."""
    convs = [(k[: -len(".weight")], v) for k, v in state_dict.items() if k.endswith(".weight")]
    x_float = [torch.from_numpy(x.astype(np.float32)) / Q_IN for x in calib_inputs]  # float 모델 입력 (0..1)
    s_in = 1.0 / Q_IN  # 정수 입력 1 단위의 실수값
    layers = []
    qmax = 2 ** (wbits - 1) - 1
    for li, (name, Wf) in enumerate(convs):
        bf = state_dict[name + ".bias"]
        last = li == len(convs) - 1
        pad = Wf.shape[-1] // 2
        s_w = Wf.abs().amax((1, 2, 3)).clamp_min(1e-8) / qmax  # 출력 채널별
        W_q = (Wf / s_w.view(-1, 1, 1, 1)).round().clamp(-qmax, qmax)
        # float 모델로 다음 활성값 범위 측정
        ys = [F.conv2d(x, Wf, bf, padding=pad) for x in x_float]
        if last:
            s_out = 1.0 / LOGIT_UNIT
        else:
            ys = [y.relu() for y in ys]
            hi = torch.quantile(torch.cat([y.flatten()[:: max(1, y.numel() // 200000)] for y in ys]), 0.9999).item()
            s_out = max(hi, 1e-3) / 255
        # 정수 경로: y_q = (acc + B) * M >> S,  acc 단위 = s_in * s_w
        unit = s_in * s_w
        B_q = (bf / unit).round()
        M = (unit / s_out * 2**SHIFT).round()
        layers.append((W_q.numpy().astype(np.int64), B_q.numpy().astype(np.int64), M.numpy().astype(np.int64), pad, not last))
        x_float = [y for y in ys] if not last else x_float
        s_in = s_out
    return QNet(layers)


def probs_from_logits(logit_q: np.ndarray) -> np.ndarray:
    """(n,5) 정수 logit → (n,5) float64 확률 (정수 빈도의 정확한 나눗셈)."""
    d = np.minimum(logit_q.max(1, keepdims=True) - logit_q, 255)
    f = EXP_TABLE[d].astype(np.float64)
    return f / f.sum(1, keepdims=True)


# ---------------------------------------------------------------- coarse 격자 (적응형 카운트)
def coarse_pos():
    ys, xs = np.mgrid[0:SH:LEVELS[0], 0:SW:LEVELS[0]]
    return ys.ravel(), xs.ravel()


class CoarseModel:
    def __init__(self):
        self.c = np.ones((K + 1, K), np.int64)  # 문맥 = 이전 프레임 같은 칸 (5 = 없음)

    def probs(self, ctx):
        c = self.c[ctx].astype(np.float64)
        return c / c.sum(1, keepdims=True)

    def update(self, ctx, sym):
        np.add.at(self.c, (ctx, sym), 16)


# ---------------------------------------------------------------- 부호화 / 복호화
def _family():
    import constriction

    return constriction.stream.model.Categorical(perfect=False)


def encode(seg: np.ndarray, qnet: QNet) -> bytes:
    import constriction

    n = len(seg)
    enc = constriction.stream.queue.RangeEncoder()
    fam = _family()
    cm = CoarseModel()
    cy, cx = coarse_pos()
    for t in range(n):
        cur = seg[t : t + 1]
        prev = seg[t - 1 : t] if t >= 1 else None
        prev2 = seg[t - 2 : t - 1] if t >= 2 else None
        sym = cur[0, cy, cx].astype(np.int32)
        ctx = prev[0, cy, cx].astype(np.int64) if prev is not None else np.full(len(cy), K)
        enc.encode(sym, fam, cm.probs(ctx))
        cm.update(ctx, sym)
        for s in LEVELS:
            h = s // 2
            for kind in "AB":
                _, target = masks(h, kind)
                logit = qnet(build_input_q(cur, prev, prev2, s, kind))[0]  # (5,gh,gw)
                p = probs_from_logits(logit[:, target].T)
                enc.encode(cur[0, ::h, ::h][target].astype(np.int32), fam, p)
    words = enc.get_compressed()
    return struct.pack("<I", n) + words.tobytes()


def decode(buf: bytes, qnet: QNet, progress=None) -> np.ndarray:
    import constriction

    (n,) = struct.unpack_from("<I", buf, 0)
    words = np.frombuffer(buf, np.uint32, offset=4)
    dec = constriction.stream.queue.RangeDecoder(words)
    fam = _family()
    cm = CoarseModel()
    cy, cx = coarse_pos()
    seg = np.zeros((n, SH, SW), np.uint8)
    for t in range(n):
        cur = seg[t : t + 1]
        prev = seg[t - 1 : t] if t >= 1 else None
        prev2 = seg[t - 2 : t - 1] if t >= 2 else None
        ctx = prev[0, cy, cx].astype(np.int64) if prev is not None else np.full(len(cy), K)
        sym = dec.decode(fam, cm.probs(ctx))
        cur[0, cy, cx] = sym
        cm.update(ctx, sym.astype(np.int64))
        for s in LEVELS:
            h = s // 2
            for kind in "AB":
                _, target = masks(h, kind)
                logit = qnet(build_input_q(cur, prev, prev2, s, kind))[0]
                p = probs_from_logits(logit[:, target].T)
                g = cur[0, ::h, ::h]  # view
                g[target] = dec.decode(fam, p)
        if progress:
            progress(t)
    return seg
