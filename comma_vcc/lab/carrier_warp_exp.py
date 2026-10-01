"""pose carrier 실험: 짝수 프레임 = warp(홀수 프레임, θ) [+ DCT 잔차].

θ (쌍마다 8개): 원근 변환 3x3 행렬의 8자유도를 항등 변환 기준 작은 값으로.
전진 운동 ≈ 소실점 기준 확대, 회전 ≈ 이동/기울임 이라 PoseNet 이 자연스럽게 반응할 것으로 기대.

    python carrier_warp_exp.py --pairs 64 --steps 300 [--dct 24]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import SH, SW, load_gt, nets, pose_out
import archive  # noqa: E402

NP = 8


def warp(img: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """img (n,3,H,W), theta (n,8) → 원근 변환으로 다시 샘플 (출력 좌표 → 입력 좌표)."""
    n = img.shape[0]
    Hm = torch.cat([theta, torch.zeros(n, 1)], 1).view(n, 3, 3) + torch.eye(3)
    ys = torch.linspace(-1, 1, SH)
    xs = torch.linspace(-1, 1, SW)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    pts = torch.stack([gx, gy, torch.ones_like(gx)], -1).view(1, -1, 3)  # (1,HW,3)
    src = pts @ Hm.transpose(1, 2)  # (n,HW,3)
    src = src[..., :2] / src[..., 2:3]
    return F.grid_sample(img, src.view(n, SH, SW, 2), mode="bilinear", padding_mode="border", align_corners=True)


def ste_round(x):
    return x + (x.round() - x).detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--fp32-after", type=int, default=100)
    ap.add_argument("--dct", type=int, default=0, help="DCT 잔차 기저 개수 (0 = 없음)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--port", type=int, default=8006)
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    small, seg, pose = load_gt()
    idx = np.linspace(1, 599, args.pairs).astype(int)
    odd = torch.from_numpy(np.stack([small[2 * i + 1] for i in idx])).round()
    target = torch.from_numpy(pose[idx])
    theta = torch.zeros(len(idx), NP, requires_grad=True)
    groups = [{"params": [theta], "lr": args.lr}]
    if args.dct:
        B = archive.dct_basis(args.dct, SH, SW)
        c = torch.zeros(len(idx), args.dct, requires_grad=True)
        groups.append({"params": [c], "lr": 0.02})
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr * 0.02)
    viz = LiveVis(f"comma vcc · warp carrier dct={args.dct}", port=args.port).start()

    def make_even(sel):
        e = warp(odd[sel], theta[sel])
        if args.dct:
            e = e + torch.einsum("nk,kchw->nchw", c[sel], B)
        return e.clamp(0, 255)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = torch.randperm(len(idx))[: args.bs]
        even = ste_round(make_even(sel))
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=step < args.fp32_after):
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
            print(f"step {step}: posenet_dist {d:.6f} term {np.sqrt(10 * d):.4f} |θ| {theta.abs().mean(0).detach().numpy().round(4)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
