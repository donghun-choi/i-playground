"""렌더러 (seg 맵 → 홀수 프레임), 짝수 프레임 합성, 512x384 → 1164x874 서브픽셀 확장, 렌더러 가중치 형식.

압축 쪽 (lab) 과 inflate 가 같은 코드를 쓴다.
"""

from __future__ import annotations

import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SH, SW = 384, 512  # 평가 네트워크가 보는 해상도
H, W = 874, 1164  # 원본 해상도


# ---------------------------------------------------------------- 렌더러: seg 맵 → 홀수 프레임
def coords(n: int, device=None) -> torch.Tensor:
    ys = torch.linspace(-1, 1, SH, device=device).view(1, 1, SH, 1).expand(n, 1, SH, SW)
    xs = torch.linspace(-1, 1, SW, device=device).view(1, 1, 1, SW).expand(n, 1, SH, SW)
    return torch.cat([ys, xs], 1)


class Renderer(nn.Module):
    """작은 U-Net (폭 c1·c2·c3). 입력: one-hot(5) + 좌표(y, x ∈ [-1, 1]). 출력: sigmoid × 255 RGB (float)."""

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
        """seg (n,SH,SW) 정수 → (n,3,SH,SW) float 0..255"""
        x = torch.cat([F.one_hot(seg.long(), 5).permute(0, 3, 1, 2).float(), coords(seg.shape[0], seg.device)], 1)
        f1 = self.e1(x)
        f2 = self.e2(F.avg_pool2d(f1, 2))
        f3 = self.e3(F.avg_pool2d(f2, 2))
        u2 = self.d2(torch.cat([F.interpolate(f3, scale_factor=2, mode="bilinear"), f2], 1))
        u1 = self.d1(torch.cat([F.interpolate(u2, scale_factor=2, mode="bilinear"), f1], 1))
        return torch.sigmoid(self.out(u1)) * 255


# ---------------------------------------------------------------- 가중치 직렬화 (출력 채널별 fp16 스케일 + int8 정수)
# [u16 텐서 수][u8 비트 (0 = 채널별 학습 비트)] + state_dict 순서로 텐서마다
#   2차원 이상: [u8 1] + fp16 스케일 (출력 채널 수) + int8 정수 (텐서 모양 그대로)  → 값 = 정수 × 스케일
#   1차원:     [u8 0] + fp16 값
def pack_state(sd: dict, bits: int = 8) -> bytes:
    """float state_dict → 위 형식 (출력 채널마다 max|w| / qmax 스케일로 반올림). self-compression 은 selfcomp.export 가 같은 형식을 만든다."""
    out = [struct.pack("<HB", len(sd), bits)]
    qmax = 2 ** (bits - 1) - 1
    for v in sd.values():
        v = v.detach().float().cpu()
        if v.ndim >= 2:
            s = (v.abs().reshape(v.shape[0], -1).amax(1).clamp_min(1e-8) / qmax).half().float().clamp_min(1e-8)  # 0 행 (fp16 에서 0) 대비
            q = (v / s.view(-1, *[1] * (v.ndim - 1))).round().clamp(-qmax, qmax).to(torch.int8)
            out.append(struct.pack("<B", 1) + s.half().numpy().tobytes() + q.numpy().tobytes())
        else:
            out.append(struct.pack("<B", 0) + v.half().numpy().tobytes())
    return b"".join(out)


def unpack_state(buf: bytes, template: dict) -> dict:
    """pack_state 형식 → float state_dict (template 은 모양과 순서만 쓴다)."""
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


# ---------------------------------------------------------------- 짝수 프레임 = 아핀(이전 쌍 홀수 프레임) + 기저 합성
AFF_SCALE = torch.tensor([0.01, 0.01, 2 / SW, 0.01, 0.01, 2 / SH])  # 행렬 원소 0.01 단위, 이동은 픽셀 단위


def affine(img: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """img (n,3,H,W), a (n,6) = [a11-1, a12, tx(px), a21, a22-1, ty(px)] (AFF_SCALE 곱하기 전) → bilinear 재샘플 (가장자리 복제)."""
    a = a * AFF_SCALE.to(a.device)
    th = torch.stack([torch.stack([1 + a[:, 0], a[:, 1], a[:, 2]], 1), torch.stack([a[:, 3], 1 + a[:, 4], a[:, 5]], 1)], 1)
    g = F.affine_grid(th, list(img.shape), align_corners=False)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=False)


def even_frames(prev: torch.Tensor, a: torch.Tensor, c: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """prev (n,3,SH,SW) 이전 쌍 홀수 프레임, a (n,6), c (n,k), B (k,1,bh,bw) 공유 기저 (회색: RGB 에 같은 값)
    → clamp(아핀(prev) + bicubic 확대(Σ_k c_k B_k), 0, 255). PoseNet 이 자연스러운 운동 상태에서 pose 를 내게 한다."""
    d = F.interpolate(torch.einsum("nk,kchw->nchw", c, B), size=(SH, SW), mode="bicubic")
    return (affine(prev, a) + d).clamp(0, 255)


# ---------------------------------------------------------------- 512x384 float → 1164x874 uint8 서브픽셀 확장
# 평가 코드의 bilinear 축소 (1164x874 → 512x384) 에서 작은 이미지 (i,j) = Σ_ab wr[i,a]·wc[j,b]·원본[y_a(i), x_b(j)] 이고,
# 축소가 읽는 원본 2x2 칸들은 서로 겹치지 않는다. 그 2x2 네 칸을 서로 다른 정수로 채우면 정수 사이 값도 만들 수 있다.
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
    """(n,384,512,3) float → (n,874,1164,3) uint8. 평가 코드가 축소하면 x 에 아주 가깝게 (평균 오차 0.0035 레벨) 돌아온다."""
    global _FULL
    if _FULL is None:
        _FULL = torch.from_numpy(_full_index())
    blk = torch.from_numpy(fine_blocks(x)).reshape(x.shape[0], SH * SW * 4, 3)
    return blk[:, _FULL].reshape(x.shape[0], H, W, 3).numpy()
