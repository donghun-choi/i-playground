"""Self-compression (Cséfalvay 2023, "Self-Compressing Neural Networks"): 출력 채널마다 비트 수 b 와 스케일 지수 e 를 학습.

가중치 W (C_out, ...) 의 채널 c:
    m_c = relu(2^(b_c - 1) - 1)                       # 허용 정수 크기 (b=4 → 7, b=1 → 0 = 채널 제거)
    s_c = fp16(2^e_c)                                 # 저장되는 스케일 그대로 (STE)
    q   = round(clamp(W / s_c, -m_c, m_c))            # STE, b 에는 clamp 경계로 기울기가 흐른다
    W'  = q * s_c
손실에 '가중치당 평균 비트' (Σ_c relu(b_c)·n_c / Σ n_c) 를 더하면 덜 중요한 채널은 비트가 줄고 0 이 되면 사라진다.

저장 형식은 model.pack_state 와 같다 (채널별 fp16 스케일 + int8 값, 1차원은 fp16) → inflate 쪽 unpack_state 그대로.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (x.round() - x).detach()


class SelfCompress(nn.Module):
    def __init__(self, model: nn.Module, init_bits: float = 4.0, skip=()):
        super().__init__()
        self.names = [k for k, p in model.named_parameters() if p.ndim >= 2 and not any(k.startswith(s) for s in skip)]
        self.e = nn.ParameterDict()
        self.b = nn.ParameterDict()
        params = dict(model.named_parameters())
        for k in self.names:
            W = params[k].detach()
            m0 = 2 ** (init_bits - 1) - 1
            amax = W.abs().reshape(W.shape[0], -1).amax(1).clamp_min(1e-3)  # 잘린 채널 (전부 0) 도 쓸 수 있는 스케일로 (아니면 fp16 에서 0 → 0/0)
            key = k.replace(".", "__")
            self.e[key] = nn.Parameter(torch.log2(amax / m0))
            self.b[key] = nn.Parameter(torch.full((W.shape[0],), float(init_bits), device=W.device))
        self.n_per_ch = {k: params[k][0].numel() for k in self.names}
        self.total = sum(params[k].numel() for k in self.names)

    def _q(self, k: str, W: torch.Tensor):
        key = k.replace(".", "__")
        e, b = self.e[key], self.b[key]
        shape = (-1,) + (1,) * (W.ndim - 1)
        s = 2.0 ** e
        s = s + (s.half().float().clamp_min(2**-14) - s).detach()  # 저장되는 fp16 값으로 (정상 범위 안)
        m = torch.relu(2.0 ** (b - 1) - 1).view(shape)
        x = W / s.view(shape)
        q = _ste_round(torch.maximum(torch.minimum(x, m), -m))
        return q, s

    def params(self, model: nn.Module) -> dict:
        """functional_call 용 파라미터 (가짜 양자화). 1차원은 fp16 (pack_state 와 같게)."""
        out = {}
        for k, p in model.named_parameters():
            if k in self.n_per_ch:
                q, s = self._q(k, p)
                out[k] = q * s.view((-1,) + (1,) * (p.ndim - 1))
            else:
                out[k] = p + (p.half().float() - p).detach()
        return out

    def bits_per_weight(self) -> torch.Tensor:
        """크기 손실: 가중치당 평균 비트 (채널 비트 × 채널 가중치 수)."""
        tot = 0.0
        for k in self.names:
            tot = tot + torch.relu(self.b[k.replace(".", "__")]).sum() * self.n_per_ch[k]
        return tot / self.total

    def rate_bits(self, model: nn.Module) -> torch.Tensor:
        """엔트로피 부호 크기 (wcodec: 출력 채널마다 이산 라플라스) 의 미분 가능한 추정, 비트 단위.

        채널마다 m = mean|q| → 양쪽 기하분포 P(q) = (1-ρ)/(1+ρ) ρ^|q| 의 ρ = (√(1+m²)-1)/m,  H = -log2 P(0) - m log2 ρ.
        (c8 렌더러에서 28,501 B vs 실제 부호 28,364 B). |q| 는 STE 라서 W 를 0 쪽으로, 스케일을 크게 미는 기울기가 흐른다.
        """
        params = dict(model.named_parameters())
        tot = 0.0
        for k in self.names:
            q, _ = self._q(k, params[k])
            m = q.abs().reshape(q.shape[0], -1).mean(1).clamp_min(1e-4)
            rho = m / (torch.sqrt(1 + m * m) + 1)  # = (√(1+m²)-1)/m, 작은 m 에서 상쇄 없이
            H = -torch.log2((1 - rho) / (1 + rho)) - m * torch.log2(rho)
            tot = tot + (H * self.n_per_ch[k]).sum()
        return tot

    @torch.no_grad()
    def export(self, model: nn.Module):
        """→ (역양자화 float state_dict = inflate 와 같은 값, pack 바이트). 정수가 int8 범위를 넘지 않게 자른다."""
        sd, packed = {}, []
        import struct

        state = model.state_dict()
        packed.append(struct.pack("<HB", len(state), 0))  # 비트 칸은 0 = 채널별 학습 비트
        for k, v in state.items():
            v = v.detach().float().cpu()
            if k in self.n_per_ch:
                q, s = self._q(k, v.to(self.e[k.replace(".", "__")].device))
                q = q.clamp(-127, 127).to(torch.int8).cpu()
                s16 = s.detach().half().cpu()
                sd[k] = q.float() * s16.float().view((-1,) + (1,) * (v.ndim - 1))
                packed.append(struct.pack("<B", 1) + s16.numpy().tobytes() + q.numpy().tobytes())
            elif v.ndim >= 2:  # self-compress 대상이 아닌 2차원 이상 (skip) 은 8비트 max-abs
                sc = (v.abs().reshape(v.shape[0], -1).amax(1).clamp_min(1e-8) / 127).half().float().clamp_min(1e-8)
                q = (v / sc.view((-1,) + (1,) * (v.ndim - 1))).round().clamp(-127, 127).to(torch.int8)
                sd[k] = q.float() * sc.view((-1,) + (1,) * (v.ndim - 1))
                packed.append(struct.pack("<B", 1) + sc.half().numpy().tobytes() + q.numpy().tobytes())
            else:
                sd[k] = v.half().float()
                packed.append(struct.pack("<B", 0) + v.half().numpy().tobytes())
        return sd, b"".join(packed)

    @torch.no_grad()
    def summary(self) -> str:
        bs = torch.cat([torch.relu(self.b[k.replace(".", "__")]).detach().cpu() for k in self.names])
        alive = (2.0 ** (bs - 1) - 1 >= 0.5).float()
        return f"평균 {self.bits_per_weight().item():.2f} bit/가중치, 살아있는 채널 {int(alive.sum())}/{len(bs)}, 비트 분포 {np.percentile(bs.numpy(), [10, 50, 90]).round(2)}"
