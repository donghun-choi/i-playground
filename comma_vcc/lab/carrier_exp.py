"""pose carrier 실험: 짝수 프레임 = base + up(Σ_k c[i,k] B_k) (회색 = Y 채널만 변화).

B (공유 기저, 저해상도) 와 c (쌍별 계수) 를 PoseNet 출력 MSE 로 함께 최적화한다.
base 후보: odd(같은 쌍 홀수 프레임) / gray / prevodd(이전 쌍 홀수 프레임)

    python carrier_exp.py --base odd --pairs 64 --steps 300
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import SH, SW, load_gt, nets, pose_out


def ste_round(x):
    return x + (x.round() - x).detach()


def dct_basis(k: int, bh: int, bw: int) -> torch.Tensor:
    """저주파부터 k 개의 2D DCT-II 기저 (k,1,bh,bw), 각 최대 진폭 1."""
    freqs = sorted(((u, v) for u in range(bh) for v in range(bw)), key=lambda f: (f[0] + f[1], f[0]))[:k]
    y = (torch.arange(bh) + 0.5) / bh
    x = (torch.arange(bw) + 0.5) / bw
    return torch.stack([(torch.cos(torch.pi * u * y)[:, None] * torch.cos(torch.pi * v * x)[None, :])[None] for u, v in freqs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="odd")
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--bres", type=int, default=4, help="기저 해상도 = 512x384 / bres")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--dimw", type=float, default=0.0, help="차원별 가중 (1/분산)^dimw")
    ap.add_argument("--rgb", action="store_true", help="기저를 RGB 3채널로 (기본: 회색 1채널)")
    ap.add_argument("--binit", type=float, default=5.0)
    ap.add_argument("--dct", action="store_true", help="학습 기저 대신 고정 DCT 기저 (저장 비용 0)")
    ap.add_argument("--fp32-after", type=int, default=10**9, help="이 스텝부터 fp32 로 학습")
    ap.add_argument("--port", type=int, default=8003)
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    small, seg, pose = load_gt()
    idx = np.linspace(1, 599, args.pairs).astype(int)
    odd = torch.from_numpy(np.stack([small[2 * i + 1] for i in idx])).round()
    shift = None
    if args.base in ("oddshift", "oddaffine"):
        base = odd.clone()
        # 쌍별 아핀: [a11-1, a12, tx(px), a21, a22-1, ty(px)] (oddshift 는 tx, ty 만 학습)
        shift = torch.zeros(len(idx), 6, requires_grad=True)
    elif args.base == "odd":
        base = odd.clone()
    elif args.base == "prevodd":
        base = torch.from_numpy(np.stack([small[2 * i - 1] for i in idx])).round()
    else:
        base = torch.full_like(odd, 128.0)
    target = torch.from_numpy(pose[idx])
    wdim = torch.from_numpy(pose).var(0).clamp_min(1e-8) ** (-args.dimw)
    wdim = wdim / wdim.mean()
    bh, bw = SH // args.bres, SW // args.bres
    if args.dct:
        B = dct_basis(args.k, bh, bw) * 30
        c = torch.zeros(len(idx), args.k, requires_grad=True)
        opt = torch.optim.Adam([c], lr=args.lr)
    else:
        B = (torch.randn(args.k, 3 if args.rgb else 1, bh, bw) * args.binit).requires_grad_(True)
        c = (torch.randn(len(idx), args.k) * 0.3).requires_grad_(True)
        opt = torch.optim.Adam([{"params": [B], "lr": args.lr * 20}, {"params": [c], "lr": args.lr}])
    if shift is not None:
        opt.add_param_group({"params": [shift], "lr": 0.02})
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.02)

    def base_of(sel):
        if shift is None:
            return base[sel]
        n_ = len(sel)
        a = shift[sel]
        if args.base == "oddshift":
            a = a * torch.tensor([0, 0, 1, 0, 0, 1.0])
        sc = torch.tensor([0.01, 0.01, 2 / SW, 0.01, 0.01, 2 / SH])  # 행렬 원소는 0.01 단위, 이동은 픽셀 단위
        a = a * sc
        th = torch.stack([torch.stack([1 + a[:, 0], a[:, 1], a[:, 2]], 1), torch.stack([a[:, 3], 1 + a[:, 4], a[:, 5]], 1)], 1)
        grid = F.affine_grid(th, (n_, 3, SH, SW), align_corners=False)
        return F.grid_sample(base[sel], grid, mode="bilinear", padding_mode="border", align_corners=False)
    viz = LiveVis(f"comma vcc · carrier base={args.base} k={args.k} bres={args.bres}", port=args.port).start()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        perm = torch.randperm(len(idx))[: args.bs]
        delta = F.interpolate(torch.einsum("nk,kchw->nchw", c[perm], B), size=(SH, SW), mode="bicubic")
        even = ste_round((base_of(perm) + delta).clamp(0, 255))
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=step < args.fp32_after):
            out = pose_out(net, even, odd[perm])
        loss = (((out.float() - target[perm]) ** 2) * wdim).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        viz.log(step, train_posenet_dist=loss.item())
        if step % 50 == 0 or step == args.steps:
            with torch.inference_mode():
                delta = F.interpolate(torch.einsum("nk,kchw->nchw", c, B), size=(SH, SW), mode="bicubic")
                even = (base_of(torch.arange(len(idx))) + delta).clamp(0, 255).round()
                err = (pose_out(net, even, odd) - target) ** 2
                d = err.mean().item()
                print("   차원별 RMS", err.mean(0).sqrt().numpy().round(4), flush=True)
            viz.log(step, posenet_dist_fp32=d, pose_term=float(np.sqrt(10 * d)))
            print(f"step {step}: posenet_dist {d:.6f} term {np.sqrt(10 * d):.4f} |c| {c.abs().mean():.3f} "
                  f"|delta| {delta.abs().mean():.2f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
