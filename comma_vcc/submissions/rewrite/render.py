"""렌더러: rend 섹션 풀기, U-Net, seg 맵 → 홀수 프레임 (GUIDE 3.4, 4.5)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def unpack_renderer(rend: bytes) -> tuple[tuple[int, int, int], list[np.ndarray]]:
    """rend 섹션 → (폭 (c1,c2,c3), 텐서 18개 float32 (형식 순서: e1.0.weight, e1.0.bias, e1.2.weight, ...)).

    [u8 13][u16 c1 c2 c3] + [u16 T][u8 0xEC][u32 L] + xz(bits, 텐서별 kind·qmax·fp16 스케일·r / fp16 bias) + range coder (정수).
    weight = int q × fp16 스케일 (출력 채널마다), bias = fp16. 텐서 모양은 폭에서 정해진다 (GUIDE 3.4 목록).
    """
    raise NotImplementedError("TODO: unpack_renderer")


class Renderer(nn.Module):
    """U-Net (폭 c1·c2·c3). 입력 one-hot(5) + 좌표 (y, x) → sigmoid × 255. 층 구성은 GUIDE 4.5."""

    def __init__(self, c1: int, c2: int, c3: int):
        super().__init__()
        raise NotImplementedError("TODO: Renderer.__init__")

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        """maps (n,SH,SW) 정수 → (n,3,SH,SW) float32 0..255"""
        raise NotImplementedError("TODO: Renderer.forward")


def load_renderer(rend: bytes) -> Renderer:
    """unpack_renderer 의 폭으로 Renderer 를 만들고 텐서를 형식 순서대로 넣는다 (eval 모드)."""
    raise NotImplementedError("TODO: load_renderer")


def render_all(G: Renderer, maps: np.ndarray, batch: int = 8) -> torch.Tensor:
    """maps (n,SH,SW) uint8 → (n,3,SH,SW) float32. inference_mode 로 batch 장씩 (참조 구현은 8장씩)."""
    raise NotImplementedError("TODO: render_all")
