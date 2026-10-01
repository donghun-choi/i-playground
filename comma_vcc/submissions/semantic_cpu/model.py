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


# ---------------------------------------------------------------- 가중치 직렬화 (int8, 출력 채널별 스케일)
def pack_state(sd: dict, bits: int = 8) -> bytes:
    """weight: int8 + 채널별 fp32 스케일, bias: fp32. 키 순서는 state_dict 순서를 따른다."""
    qmax = 2 ** (bits - 1) - 1
    out = [struct.pack("<H", len(sd))]
    for k, v in sd.items():
        v = v.detach().float()
        if v.ndim >= 2:
            s = v.abs().reshape(v.shape[0], -1).amax(1).clamp_min(1e-12) / qmax
            q = (v / s.view(-1, *[1] * (v.ndim - 1))).round().clamp(-qmax, qmax).to(torch.int8)
            out.append(struct.pack("<B", 1) + s.numpy().astype(np.float32).tobytes() + q.numpy().tobytes())
        else:
            out.append(struct.pack("<B", 0) + v.numpy().astype(np.float32).tobytes())
    return b"".join(out)


def unpack_state(buf: bytes, template: dict) -> dict:
    (n,) = struct.unpack_from("<H", buf, 0)
    off = 2
    sd = {}
    assert n == len(template)
    for k, v in template.items():
        (kind,) = struct.unpack_from("<B", buf, off)
        off += 1
        if kind == 1:
            s = np.frombuffer(buf, np.float32, v.shape[0], off)
            off += 4 * v.shape[0]
            q = np.frombuffer(buf, np.int8, v.numel(), off).reshape(v.shape)
            off += v.numel()
            sd[k] = torch.from_numpy(q.astype(np.float32) * s.reshape(-1, *[1] * (v.ndim - 1)))
        else:
            sd[k] = torch.from_numpy(np.frombuffer(buf, np.float32, v.numel(), off).reshape(v.shape).copy())
            off += 4 * v.numel()
    return sd


def quantize_roundtrip(module: nn.Module) -> None:
    """모듈 가중치를 int8 왕복 값으로 바꾼다 (inflate 와 같은 가중치로 평가/미세조정하기 위해)."""
    module.load_state_dict(unpack_state(pack_state(module.state_dict()), module.state_dict()))


# ---------------------------------------------------------------- pose carrier: 짝수 프레임
def carrier_delta(c: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """c (n,k), B (k,1,bh,bw) → (n,1,384,512) 회색 섭동."""
    return F.interpolate(torch.einsum("nk,kchw->nchw", c, B), size=(SH, SW), mode="bilinear")


def even_frames(odd: torch.Tensor, c: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """odd: (n,3,384,512) 정수값 float. → 짝수 프레임 (반올림 전 float)."""
    return (odd + carrier_delta(c, B)).clamp(0, 255)


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
