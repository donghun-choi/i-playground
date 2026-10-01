"""pose carrier 실험: 짝수 = 홀수 + Σ_k c[i,k] · f_k(홀수)  (f: 공유 작은 CNN, 출력 K 채널 회색 섭동)

고주파 구조는 홀수 프레임 내용(경계)에서 나오고, 저장할 것은 작은 CNN 가중치 + 쌍별 계수뿐.

    python carrier_cnn_exp.py --pairs 64 --steps 300 --k 12 --ch 16
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import SH, SW, load_gt, nets, pose_out
from model import coords  # noqa: E402


class PerturbNet(nn.Module):
    def __init__(self, k: int, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(),
            nn.Conv2d(ch, ch, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(ch, k, 3, padding=1),
        )

    def forward(self, odd):
        x = torch.cat([(odd - 64) / 64, coords(odd.shape[0])], 1)
        return self.net(x) * 10  # (n,k,H,W)


def ste_round(x):
    return x + (x.round() - x).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fp32-after", type=int, default=100)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--port", type=int, default=8007)
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    small, seg, pose = load_gt()
    idx = np.linspace(1, 599, args.pairs).astype(int)
    odd = torch.from_numpy(np.stack([small[2 * i + 1] for i in idx])).round()
    target = torch.from_numpy(pose[idx])
    f = PerturbNet(args.k, args.ch)
    print(f"perturb net params {sum(p.numel() for p in f.parameters()):,}", flush=True)
    c = torch.zeros(len(idx), args.k)
    c[:, 0] = 1.0
    c.requires_grad_(True)
    opt = torch.optim.Adam([{"params": f.parameters(), "lr": args.lr}, {"params": [c], "lr": 0.02}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.02)
    viz = LiveVis(f"comma vcc · cnn carrier k={args.k} ch={args.ch}", port=args.port).start()

    def make_even(sel):
        return (odd[sel] + torch.einsum("nk,nkhw->nhw", c[sel], f(odd[sel]))[:, None]).clamp(0, 255)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = torch.randperm(len(idx))[: args.bs]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=step < args.fp32_after):
            even = ste_round(make_even(sel).float())
            out = pose_out(net, even, odd[sel])
        loss = ((out.float() - target[sel]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        viz.log(step, train_posenet_dist=loss.item())
        if step % 50 == 0 or step == args.steps:
            with torch.inference_mode():
                d = ((pose_out(net, make_even(torch.arange(len(idx))).round(), odd) - target) ** 2).mean().item()
            viz.log(step, posenet_dist=d, pose_term=float(np.sqrt(10 * d)))
            print(f"step {step}: posenet_dist {d:.6f} term {np.sqrt(10 * d):.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
