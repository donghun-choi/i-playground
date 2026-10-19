"""seg 맵 무손실 코덱: 계층 순서 + 정수 문맥 CNN + range coder (numpy, torch, constriction 만 사용).

프레임 t (0..n-1, 0 은 첫 쌍의 '이전 프레임' 용 맵) 하나의 부호화 순서
  1. stride-32 격자 12x16 칸: 적응형 카운트 (문맥 = 이전 맵의 같은 칸)
  2. 레벨 s = 32, 16, 8, 4, 2 마다 stride-h 격자 (h = s/2, 격자점 (i,j) = 픽셀 (i·h, j·h)) 에서 세 패스
       A1: (홀,홀) 칸 중 i ≡ j (mod 4)    아는 칸: (짝,짝)
       A2: 나머지 (홀,홀) 칸               아는 칸: (짝,짝) + A1
       B : 한쪽만 홀수인 칸                아는 칸: (짝,짝) + (홀,홀)
     패스 안의 칸들은 서로를 문맥으로 쓰지 않는다 → 한 번의 CNN 으로 패스 전체의 확률이 나온다.
확률은 정수 문맥 CNN (QNet) 의 logit → 정수 빈도표 → 정확한 나눗셈. 모든 값이 기계와 무관하게 같도록 짰다.
형식 명세: comma_vcc/GUIDE.md
"""

from __future__ import annotations

import math
import struct

import numpy as np
import torch
import torch.nn.functional as F

SH, SW = 384, 512
LEVELS = [32, 16, 8, 4, 2]
PASSES = ("A1", "A2", "B")
HS = [s // 2 for s in LEVELS]  # 레벨별 격자 간격
K = 5  # 클래스 수
C_IN = 23  # 문맥 CNN 입력 채널
Q_IN = 64  # 입력값 1.0 = 64
SHIFT = 16  # 재양자화 배율 M 의 소수 비트
LOGIT_UNIT = 16  # logit 1 nat = 16
FLAGS = 6  # QNet 바이트 첫 층 relu 바이트의 비트 1 (float32 재양자화) | 비트 2 (빈도표 최대 2^20). 이 형식만 쓴다.
# logit 이 최댓값보다 d 작은 클래스의 빈도 (최대 2^20, 최소 1)
FREQ = torch.tensor([max(1, round(2**20 * math.exp(-d / LOGIT_UNIT))) for d in range(256)], dtype=torch.float64)


# ---------------------------------------------------------------- 패스와 입력
def masks(h: int, kind: str):
    """stride-h 격자에서 패스 kind 의 (아는 칸, 보낼 칸) bool 배열 (SH/h, SW/h)."""
    i = np.arange(SH // h)[:, None]
    j = np.arange(SW // h)[None, :]
    even = (i % 2 == 0) & (j % 2 == 0)
    odd = (i % 2 == 1) & (j % 2 == 1)
    a1 = odd & (i % 4 == j % 4)
    if kind == "A1":
        return even, a1
    if kind == "A2":
        return even | a1, odd & ~a1
    return even | odd, ~(even | odd)


class Inputs:
    """문맥 CNN 입력 (n, 23, SH/h, SW/h) float32 (값은 0..64 정수).

    채널: 현재 맵 one-hot × 아는 칸 (5) | 아는 칸 (1) | 이전 맵 클래스 비율 (5) | 전전 맵 클래스 비율 (5)
          | A 패스 표시 (1) | 레벨 one-hot (5, LEVELS 순서) | A2 패스 표시 (1)     (표시 값은 64)
    클래스 비율 = floor(h×h 블록 안 그 클래스 픽셀 수 × 64 / h²), 블록은 격자점이 왼쪽 위 모서리.
    """

    def __init__(self):
        self.known, self.target = {}, {}
        for h in HS:
            for kind in PASSES:
                kn, tg = masks(h, kind)
                self.known[h, kind] = torch.from_numpy(kn).float()
                self.target[h, kind] = torch.from_numpy(tg)

    @staticmethod
    def fracs(m: torch.Tensor | None, n: int = 1, hs=HS) -> dict:
        """맵 (n,SH,SW) uint8 → {h: (n,5,SH/h,SW/h) 클래스 비율}. 맵이 없으면 (None) 0."""
        if m is None:
            return {h: torch.zeros(n, K, SH // h, SW // h) for h in hs}
        oh = F.one_hot(m.long(), K).permute(0, 3, 1, 2).float()
        return {h: torch.floor((F.avg_pool2d(oh, h) if h > 1 else oh) * Q_IN) for h in hs}

    def ctx_fracs(self, seg_t: torch.Tensor, idx: list) -> dict:
        """프레임 번호 목록 (None = 없음) → 그 맵들의 비율을 쌓은 것 (없는 자리는 0)."""
        out = self.fracs(None, len(idx))
        valid = [i for i, k in enumerate(idx) if k is not None]
        if valid:
            f = self.fracs(seg_t[[idx[i] for i in valid]], len(valid))
            for h in HS:
                out[h][valid] = f[h]
        return out

    def build(self, cur: torch.Tensor, s: int, kind: str, fp: dict, fp2: dict) -> torch.Tensor:
        """cur: (n,SH,SW) uint8 현재 맵 (보낼 칸은 아무 값이어도 된다), fp/fp2: 이전·전전 맵의 fracs."""
        h = s // 2
        n = cur.shape[0]
        kn = self.known[h, kind]
        gh, gw = kn.shape
        g = cur[:, ::h, ::h]
        lvl = torch.zeros(n, len(LEVELS), gh, gw)
        lvl[:, LEVELS.index(s)] = Q_IN

        def flag(on):
            return torch.full((n, 1, gh, gw), float(Q_IN if on else 0))

        onehot = F.one_hot(g.long(), K).permute(0, 3, 1, 2).float() * (kn * Q_IN)
        return torch.cat([onehot, (kn * Q_IN).expand(n, 1, gh, gw), fp[h], fp2[h], flag(kind != "B"), lvl, flag(kind == "A2")], 1)


# ---------------------------------------------------------------- 정수 문맥 CNN
class QNet:
    """layers: [(W 정수 (o,i,k,k), B 정수 (o,), M 정수 (o,), dilation, relu)], padding = dilation·(k//2).

    층마다  acc = conv(x, W)                                (정확한 정수: 모든 부분합이 2^24 미만이라 float32 합성곱이 정확)
           y   = floor(f32(f32(acc + B) × f32(M · 2^-16)))   (덧셈·곱셈 각각 float32 IEEE 반올림 → 기계와 무관)
           relu 층이면 y = clamp(y, 0, 255)
    마지막 층 (relu 없음) 의 y 가 logit (1/16 nat 단위).
    """

    def __init__(self, layers):
        self.layers = layers
        in_max = Q_IN
        for W, B, M, _, _ in layers:
            assert in_max * np.abs(W).sum((1, 2, 3)).max() + np.abs(B).max() < 2**24 and np.abs(M).max() < 2**24, "float32 정확 범위 초과"
            in_max = 255
        self._g = [(torch.from_numpy(W.astype(np.float32)).contiguous(memory_format=torch.channels_last),
                    torch.from_numpy(B.astype(np.float32)).view(1, -1, 1, 1),
                    torch.from_numpy((M * 2.0**-SHIFT).astype(np.float32)).view(1, -1, 1, 1), dil, relu)
                   for W, B, M, dil, relu in layers]
        self.dtype = torch.float32  # 합성곱 dtype (self_check 가 실패하면 float64)

    def self_check(self) -> bool:
        """이 기계의 float32 합성곱이 정확한지 레벨별 격자 크기마다 float64 결과와 비교. 아니면 float64 합성곱으로 전환.
        (합성곱 구현이 격자 크기마다 달라질 수 있어서 모든 크기를 본다)"""
        rng = np.random.default_rng(0)
        xs = [torch.from_numpy(rng.integers(0, Q_IN + 1, (1, C_IN, SH // h, SW // h))).float() for h in HS]
        self.dtype = torch.float64
        ref = [self.run(x) for x in xs]
        self.dtype = torch.float32
        ok = all(torch.equal(r, self.run(x)) for r, x in zip(ref, xs))
        if not ok:
            self.dtype = torch.float64
        return ok

    def run(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n,23,gh,gw) 정수값 float 텐서 → (n,5,gh,gw) 정수값 logit (float32)."""
        x = x.to(self.dtype).contiguous(memory_format=torch.channels_last)
        for W, B, Ms, dil, relu in self._g:
            y = F.conv2d(x, W.to(self.dtype), padding=dil * (W.shape[-1] // 2), dilation=dil).float()  # 정확한 정수
            y.add_(B).mul_(Ms).floor_()
            if relu:
                y.clamp_(0, 255)
            x = y.to(self.dtype)
        return x.float().contiguous()

    # 직렬화: [u8 층 수] + 층마다 [u16 o][u16 i][u8 k][u8 dil][u8 relu|FLAGS] + W int8 + B int32 + M int32
    def to_bytes(self) -> bytes:
        out = [struct.pack("<B", len(self.layers))]
        for li, (W, B, M, dil, relu) in enumerate(self.layers):
            o, i, k, _ = W.shape
            out.append(struct.pack("<HHBBB", o, i, k, dil, int(relu) | (FLAGS if li == 0 else 0)))
            out += [W.astype(np.int8).tobytes(), B.astype(np.int32).tobytes(), M.astype(np.int32).tobytes()]
        return b"".join(out)

    @classmethod
    def from_bytes(cls, buf: bytes, off: int = 0):
        (n,) = struct.unpack_from("<B", buf, off)
        off += 1
        layers = []
        for li in range(n):
            o, i, k, dil, relu = struct.unpack_from("<HHBBB", buf, off)
            off += 7
            if li == 0:
                assert relu & FLAGS == FLAGS, "예전 형식 (float64 재양자화 / 최대 빈도 2^15) 은 지원하지 않는다"
                relu &= 1
            W = np.frombuffer(buf, np.int8, o * i * k * k, off).reshape(o, i, k, k).astype(np.int64)
            off += o * i * k * k
            B = np.frombuffer(buf, np.int32, o, off).astype(np.int64)
            off += 4 * o
            M = np.frombuffer(buf, np.int32, o, off).astype(np.int64)
            off += 4 * o
            layers.append((W, B, M, dil, bool(relu)))
        return cls(layers), off


def probs(logit: torch.Tensor) -> np.ndarray:
    """(m,5) 정수값 logit → (m,5) float64 확률: freq_c = FREQ[min(max logit - logit_c, 255)], p_c = freq_c / Σ freq.
    (빈도가 정수라 합과 나눗셈이 정확 → 기계와 무관. constriction 은 stride 를 무시하므로 C 연속 배열로 넘긴다)"""
    d = (logit.amax(1, keepdim=True) - logit).clamp_(max=255).long()
    f = FREQ[d]
    return (f / f.sum(1, keepdim=True)).contiguous().numpy()


# ---------------------------------------------------------------- stride-32 격자 (적응형 카운트)
def coarse_pos():
    """stride-32 격자 칸의 (행, 열) 픽셀 좌표, 행 우선 192개."""
    ys, xs = np.mgrid[0:SH:LEVELS[0], 0:SW:LEVELS[0]]
    return ys.ravel(), xs.ravel()


class CoarseModel:
    """문맥 (이전 맵의 같은 칸 클래스, 5 = 이전 맵 없음) 마다 클래스 카운트. 처음 1, 프레임을 보낼 때마다 +16."""

    def __init__(self):
        self.c = np.ones((K + 1, K), np.int64)

    def probs(self, ctx):
        c = self.c[ctx].astype(np.float64)
        return np.ascontiguousarray(c / c.sum(1, keepdims=True))

    def update(self, ctx, sym):
        np.add.at(self.c, (ctx, sym), 16)


# ---------------------------------------------------------------- 부호화 / 복호화
def _family():
    import constriction

    return constriction.stream.model.Categorical(perfect=False)


def encode(seg: np.ndarray, qnet: QNet, chunk: int = 16) -> bytes:
    """seg (n,SH,SW) uint8 → [u32 n] + range coder 단어 (u32).
    확률은 chunk 장씩 묶어 한 번에 계산하고 (인코더는 모든 맵을 안다), 부호화는 디코드 순서 (프레임 → 격자 → 레벨 → 패스) 대로."""
    import constriction

    n = len(seg)
    enc = constriction.stream.queue.RangeEncoder()
    fam = _family()
    cm = CoarseModel()
    cy, cx = coarse_pos()
    fi = Inputs()
    seg_t = torch.from_numpy(seg)
    for t0 in range(0, n, chunk):
        frames = list(range(t0, min(t0 + chunk, n)))
        cur = seg_t[frames[0] : frames[-1] + 1]
        fp = fi.ctx_fracs(seg_t, [t - 1 if t >= 1 else None for t in frames])
        fp2 = fi.ctx_fracs(seg_t, [t - 2 if t >= 2 else None for t in frames])
        per_pass = []  # [(logit (b,m,5), 기호 (b,m))]
        with torch.inference_mode():
            for s in LEVELS:
                h = s // 2
                for kind in PASSES:
                    tg = fi.target[h, kind]
                    logit = qnet.run(fi.build(cur, s, kind, fp, fp2))
                    per_pass.append((logit[:, :, tg].permute(0, 2, 1), cur[:, ::h, ::h][:, tg].numpy().astype(np.int32)))
        for j, t in enumerate(frames):
            sym = seg[t, cy, cx].astype(np.int32)
            cctx = seg[t - 1, cy, cx].astype(np.int64) if t >= 1 else np.full(len(cy), K)
            enc.encode(sym, fam, cm.probs(cctx))
            cm.update(cctx, sym)
            for lt, sy in per_pass:
                enc.encode(np.ascontiguousarray(sy[j]), fam, probs(lt[j]))
    return struct.pack("<I", n) + enc.get_compressed().tobytes()


def decode(buf: bytes, qnet: QNet) -> np.ndarray:
    """encode 의 역. → (n,SH,SW) uint8"""
    import constriction

    (n,) = struct.unpack_from("<I", buf, 0)
    dec = constriction.stream.queue.RangeDecoder(np.frombuffer(buf, np.uint32, offset=4))
    fam = _family()
    cm = CoarseModel()
    cy, cx = coarse_pos()
    fi = Inputs()
    seg = np.zeros((n, SH, SW), np.uint8)
    seg_t = torch.from_numpy(seg)  # 같은 메모리
    zero = fi.fracs(None)
    fr = {}  # 복호화한 맵의 클래스 비율 (다음 두 프레임의 문맥)
    with torch.inference_mode():
        for t in range(n):
            cur = seg_t[t : t + 1]
            fp, fp2 = fr.get(t - 1, zero), fr.get(t - 2, zero)
            cctx = seg[t - 1, cy, cx].astype(np.int64) if t >= 1 else np.full(len(cy), K)
            sym = dec.decode(fam, cm.probs(cctx))
            seg[t, cy, cx] = sym
            cm.update(cctx, sym.astype(np.int64))
            for s in LEVELS:
                h = s // 2
                g = cur[0, ::h, ::h]  # view: 복호화한 값이 바로 seg 에 들어간다
                for kind in PASSES:
                    tg = fi.target[h, kind]
                    logit = qnet.run(fi.build(cur, s, kind, fp, fp2))[0]  # (5,gh,gw)
                    g[tg] = torch.from_numpy(dec.decode(fam, probs(logit[:, tg].T)).astype(np.uint8))
            fr[t] = fi.fracs(cur)
            fr.pop(t - 2, None)
    return seg
