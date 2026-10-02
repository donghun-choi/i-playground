"""렌더러, pose carrier, 원해상도 확장. 압축(lab)과 inflate 가 같은 코드를 쓴다."""

from __future__ import annotations

import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SH, SW = 384, 512  # 평가 네트워크가 보는 해상도
H, W = 874, 1164  # 원본 해상도


# ---------------------------------------------------------------- 렌더러: seg 맵 → 홀수 프레임
def coords(n: int) -> torch.Tensor:
    ys = torch.linspace(-1, 1, SH).view(1, 1, SH, 1).expand(n, 1, SH, SW)
    xs = torch.linspace(-1, 1, SW).view(1, 1, 1, SW).expand(n, 1, SH, SW)
    return torch.cat([ys, xs], 1)


class Renderer(nn.Module):
    """작은 U-Net. 입력: one-hot(5) + 좌표(2). 출력: 0~255 RGB."""

    def __init__(self, c1: int = 16, c2: int = 24, c3: int = 32):
        super().__init__()
        self.cfg = (c1, c2, c3)
        self.e1 = nn.Sequential(nn.Conv2d(7, c1, 3, padding=1), nn.ReLU(), nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU())
        self.e2 = nn.Sequential(nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(), nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU())
        self.e3 = nn.Sequential(nn.Conv2d(c2, c3, 3, padding=1), nn.ReLU(), nn.Conv2d(c3, c3, 3, padding=2, dilation=2), nn.ReLU())
        self.d2 = nn.Sequential(nn.Conv2d(c3 + c2, c2, 3, padding=1), nn.ReLU())
        self.d1 = nn.Sequential(nn.Conv2d(c2 + c1, c1, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(c1, 3, 1)

    def forward(self, seg: torch.Tensor) -> torch.Tensor:
        x = torch.cat([F.one_hot(seg.long(), 5).permute(0, 3, 1, 2).float(), coords(seg.shape[0])], 1)
        f1 = self.e1(x)
        f2 = self.e2(F.avg_pool2d(f1, 2))
        f3 = self.e3(F.avg_pool2d(f2, 2))
        u2 = self.d2(torch.cat([F.interpolate(f3, scale_factor=2, mode="bilinear"), f2], 1))
        u1 = self.d1(torch.cat([F.interpolate(u2, scale_factor=2, mode="bilinear"), f1], 1))
        return torch.sigmoid(self.out(u1)) * 255


class _Block(nn.Module):
    """depthwise(dilation) → pointwise → GroupNorm → 프레임별 FiLM → GELU, 잔차."""

    def __init__(self, w: int, fdim: int, dil: int):
        super().__init__()
        self.dw = nn.Conv2d(w, w, 3, padding=dil, dilation=dil, groups=w)
        self.pw = nn.Conv2d(w, w, 1)
        self.norm = nn.GroupNorm(max(1, w // 8), w)
        self.film = nn.Linear(fdim, 2 * w)

    def forward(self, x, f):
        r = self.norm(self.pw(self.dw(x)))
        scale, shift = self.film(f).chunk(2, dim=1)
        return x + F.gelu(r * (1 + scale[:, :, None, None]) + shift[:, :, None, None])


class RendererV2(nn.Module):
    """전해상도 렌더러 (구조는 PR #130 의 semantic renderer 를 참고): 클래스 임베딩 + 좌표 → 블록 4개 → RGB.

    프레임마다 작은 임베딩(fdim)으로 FiLM 변조 → 같은 맵이라도 프레임별로 미세 조정 가능.
    """

    def __init__(self, width: int = 64, fdim: int = 8, n_frames: int = 600, dils=(1, 1, 2, 4)):
        super().__init__()
        self.cfg = (width, fdim, n_frames, tuple(dils))
        self.embed = nn.Conv2d(5 + 4, width, 1)  # one-hot + (x, y, x², y²)
        self.frame = nn.Embedding(n_frames, fdim)
        nn.init.normal_(self.frame.weight, std=0.1)
        self.blocks = nn.ModuleList([_Block(width, fdim, d) for d in dils])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, seg: torch.Tensor, idx: torch.Tensor | None = None) -> torch.Tensor:
        n = seg.shape[0]
        c = coords(n)
        x = torch.cat([F.one_hot(seg.long(), 5).permute(0, 3, 1, 2).float(), c, c * c], 1)
        x = self.embed(x.contiguous(memory_format=torch.channels_last))
        f = self.frame(idx if idx is not None else torch.zeros(n, dtype=torch.long))
        for b in self.blocks:
            x = b(x, f)
        return torch.sigmoid(self.head(F.gelu(x))) * 255


def make_renderer(cfg) -> nn.Module:
    """cfg: None/() → v1 Renderer, (width, fdim, n_frames, dils) → RendererV2"""
    return RendererV2(*cfg) if cfg else Renderer()


def render(G: nn.Module, seg: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return G(seg, idx) if isinstance(G, RendererV2) else G(seg)


# ---------------------------------------------------------------- 가중치 직렬화 (int8, 출력 채널별 스케일)
def pack_state(sd: dict, bits: int = 8) -> bytes:
    """weight(2차원 이상): bits 비트 정수(int8 바이트에 저장) + 출력 채널별 fp16 스케일, 1차원: fp16.

    값의 가짓수가 적을수록 lzma 가 잘 줄인다 (archive 에서 섹션 전체를 xz 로 압축).
    """
    out = [struct.pack("<HB", len(sd), bits)]
    qmax = 2 ** (bits - 1) - 1
    for k, v in sd.items():
        v = v.detach().float()
        if v.ndim >= 2:
            s = (v.abs().reshape(v.shape[0], -1).amax(1).clamp_min(1e-8) / qmax).half().float()
            q = (v / s.view(-1, *[1] * (v.ndim - 1))).round().clamp(-qmax, qmax).to(torch.int8)
            out.append(struct.pack("<B", 1) + s.half().numpy().tobytes() + q.numpy().tobytes())
        else:
            out.append(struct.pack("<B", 0) + v.half().numpy().tobytes())
    return b"".join(out)


def unpack_state(buf: bytes, template: dict) -> dict:
    n, _bits = struct.unpack_from("<HB", buf, 0)
    off = 3
    sd = {}
    assert n == len(template)
    for k, v in template.items():
        (kind,) = struct.unpack_from("<B", buf, off)
        off += 1
        if kind == 1:
            s = np.frombuffer(buf, np.float16, v.shape[0], off).astype(np.float32)
            off += 2 * v.shape[0]
            q = np.frombuffer(buf, np.int8, v.numel(), off).reshape(v.shape)
            off += v.numel()
            sd[k] = torch.from_numpy(q.astype(np.float32) * s.reshape(-1, *[1] * (v.ndim - 1)))
        else:
            sd[k] = torch.from_numpy(np.frombuffer(buf, np.float16, v.numel(), off).astype(np.float32).reshape(v.shape))
            off += 2 * v.numel()
    return sd


def fake_quant_(module: nn.Module, bits: int) -> None:
    """양자화 인지 학습용: 2차원 이상 가중치를 bits 비트 격자 값으로 덮어쓴다 (pack_state 와 같은 규칙)."""
    qmax = 2 ** (bits - 1) - 1
    with torch.no_grad():
        for p in module.parameters():
            if p.ndim >= 2:
                s = (p.abs().reshape(p.shape[0], -1).amax(1).clamp_min(1e-8) / qmax).half().float()
                p.copy_((p / s.view(-1, *[1] * (p.ndim - 1))).round().clamp(-qmax, qmax) * s.view(-1, *[1] * (p.ndim - 1)))
            else:
                p.copy_(p.half().float())


def quantize_roundtrip(module: nn.Module, bits: int = 8) -> None:
    """모듈 가중치를 저장/복원 왕복 값으로 바꾼다 (inflate 와 같은 가중치로 평가/미세조정하기 위해)."""
    module.load_state_dict(unpack_state(pack_state(module.state_dict(), bits), module.state_dict()))


# ---------------------------------------------------------------- pose carrier: 짝수 프레임
GRAY = 127.5


def carrier_delta(c: torch.Tensor, B: torch.Tensor, mode: str = "bicubic") -> torch.Tensor:
    """c (n,k), B (k,C,bh,bw) → (n,C,384,512) 섭동 (C=1 이면 회색)."""
    return F.interpolate(torch.einsum("nk,kchw->nchw", c, B), size=(SH, SW), mode=mode)


def even_frames(odd: torch.Tensor, c: torch.Tensor, B: torch.Tensor, base: str = "gray", mode: str = "bicubic") -> torch.Tensor:
    """짝수 프레임 (float, 0..255). base: 'gray' (127.5) 또는 'odd' (같은 쌍 홀수 프레임)."""
    b = odd if base == "odd" else torch.full_like(odd, GRAY)
    return (b + carrier_delta(c, B, mode)).clamp(0, 255)


AFF_SCALE = torch.tensor([0.01, 0.01, 2 / SW, 0.01, 0.01, 2 / SH])  # 행렬 원소 0.01 단위, 이동 픽셀 단위


def affine(img: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """img (n,3,H,W), a (n,6) = [a11-1, a12, tx(px), a21, a22-1, ty(px)] (스케일 전) → bilinear 재샘플."""
    a = a * AFF_SCALE
    th = torch.stack([torch.stack([1 + a[:, 0], a[:, 1], a[:, 2]], 1), torch.stack([a[:, 3], 1 + a[:, 4], a[:, 5]], 1)], 1)
    g = F.affine_grid(th, list(img.shape), align_corners=False)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=False)


def even_frames_prev(prev: torch.Tensor, a: torch.Tensor, c: torch.Tensor, B: torch.Tensor, mode: str = "bicubic") -> torch.Tensor:
    """짝수 프레임 = 아핀(이전 쌍 홀수 프레임) + carrier. PoseNet 을 자연스러운 운동 상태에 둔다."""
    return (affine(prev, a) + carrier_delta(c, B, mode)).clamp(0, 255)


# ---------------------------------------------------------------- 512x384 → 1164x874
def _src_index(n_in: int, n_out: int):
    s = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, None)
    i0 = np.floor(s).astype(np.int64)
    return i0, np.minimum(i0 + 1, n_in - 1)


def expand_index(n_in: int, n_out: int) -> np.ndarray:
    """원해상도 각 행(열)이 어느 작은 이미지 행(열)을 복사할지.

    평가 코드의 bilinear 축소는 작은 이미지 한 칸마다 원해상도 2x2 칸만 읽는다.
    그 2x2 를 같은 값으로 채우면 축소 결과가 정확히 그 값이 된다. 안 읽히는 줄은 가까운 줄로 채운다.
    """
    i0, i1 = _src_index(n_in, n_out)
    idx = np.full(n_in, -1, np.int64)
    idx[i0] = np.arange(n_out)
    idx[i1] = np.arange(n_out)
    used = np.flatnonzero(idx >= 0)
    for i in np.flatnonzero(idx < 0):
        idx[i] = idx[used[np.argmin(np.abs(used - i))]]
    return idx


ROWS = expand_index(H, SH)
COLS = expand_index(W, SW)


def expand(img_u8: np.ndarray) -> np.ndarray:
    """(n,384,512,3) uint8 → (n,874,1164,3) uint8"""
    return img_u8[:, ROWS][:, :, COLS]


# ---------------------------------------------------------------- 서브픽셀 정밀 확장
# 평가 코드의 bilinear 축소에서 작은 이미지 (i,j) = Σ_ab wr[i,a]·wc[j,b]·원본[y_a(i), x_b(j)].
# 2x2 네 칸을 서로 다른 정수로 채우면 정수 사이 값도 만들 수 있다 (가중치가 칸마다 달라서).
def _bilinear_taps(n_in: int, n_out: int):
    s = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, None)
    i0 = np.floor(s).astype(np.int64)
    lam = s - i0
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1, np.stack([1 - lam, lam], 1)  # (n_out, 2)


_R0, _R1, _WR = _bilinear_taps(H, SH)
_C0, _C1, _WC = _bilinear_taps(W, SW)
_STEPS = np.array([-1, 0, 1, 2], np.int64)
_COMBOS = np.stack(np.meshgrid(_STEPS, _STEPS, _STEPS, _STEPS, indexing="ij"), -1).reshape(-1, 4)  # (256,4) b00,b01,b10,b11
_TABLE = None


def _table():
    """위치별로 256 조합의 가중합을 정렬해 둔다: (SH*SW,256) 정렬된 합, 조합 번호."""
    global _TABLE
    if _TABLE is None:
        w = torch.from_numpy((_WR[:, None, :, None] * _WC[None, :, None, :]).reshape(SH * SW, 4)).float()
        sums = w @ torch.from_numpy(_COMBOS).float().T  # (P,256)
        sorted_sums, order = sums.sort(1)
        _TABLE = (sorted_sums.contiguous(), order)
    return _TABLE


def fine_blocks(x: np.ndarray) -> np.ndarray:
    """x: (n,384,512,3) float (0..255) → (n,384,512,4,3) uint8 : 각 칸의 2x2 정수값 (b00,b01,b10,b11).

    기준값 base = clamp(floor(x), 1, 253) 에 b ∈ {-1,0,1,2}^4 를 더한 조합 중 가중합이 x 에 가장 가까운 것.
    (기준값을 1..253 으로 묶으면 모든 조합이 0..255 안에 있다)
    """
    sorted_sums, order = _table()
    combos = torch.from_numpy(_COMBOS).to(torch.int16)
    n = x.shape[0]
    P = SH * SW
    xf = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).reshape(n, P, 3).permute(1, 0, 2).reshape(P, n * 3)
    base = xf.floor().clamp(1, 253)
    r = (xf - base).contiguous()  # (P, n*3) in [-1, 2]
    j = torch.searchsorted(sorted_sums, r).clamp(1, 255)
    lo = sorted_sums.gather(1, j - 1)
    hi = sorted_sums.gather(1, j)
    j = torch.where((r - lo).abs() <= (hi - r).abs(), j - 1, j)
    k = order.gather(1, j)  # (P, n*3) 조합 번호
    vals = base.to(torch.int16)[:, :, None] + combos[k]  # (P, n*3, 4)
    return vals.view(P, n, 3, 4).permute(1, 0, 3, 2).reshape(n, SH, SW, 4, 3).to(torch.uint8).numpy()


def _full_index() -> np.ndarray:
    """원해상도 (874*1164) 각 픽셀이 fine_blocks 결과 (384*512*4) 의 어느 값을 복사할지."""
    def sub(n_in, r0, r1):
        i_of, a_of = np.full(n_in, -1), np.full(n_in, -1)
        i_of[r0], a_of[r0] = np.arange(len(r0)), 0
        i_of[r1], a_of[r1] = np.arange(len(r1)), 1
        used = np.flatnonzero(i_of >= 0)
        for r in np.flatnonzero(i_of < 0):  # 안 읽히는 줄: 가까운 줄 값 (아무 값이어도 됨)
            u = used[np.argmin(np.abs(used - r))]
            i_of[r], a_of[r] = i_of[u], a_of[u]
        return i_of, a_of

    ri, ra = sub(H, _R0, _R1)
    ci, ca = sub(W, _C0, _C1)
    return ((ri[:, None] * SW + ci[None, :]) * 4 + ra[:, None] * 2 + ca[None, :]).ravel()


_FULL = None


def expand_fine(x: np.ndarray) -> np.ndarray:
    """(n,384,512,3) float → (n,874,1164,3) uint8. 축소하면 x 에 아주 가깝게 돌아온다."""
    global _FULL
    if _FULL is None:
        _FULL = torch.from_numpy(_full_index())
    blk = torch.from_numpy(fine_blocks(x)).reshape(x.shape[0], SH * SW * 4, 3)
    return blk[:, _FULL].reshape(x.shape[0], H, W, 3).numpy()
