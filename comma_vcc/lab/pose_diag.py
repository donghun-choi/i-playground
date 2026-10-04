"""pose 진단: 회전/횡방향 차원(1~5)이 왜 안 맞는지, 짝수 프레임 방식별로 작은 부분집합에서 비교.

  gray : 회색(127.5) + 진폭 32 · Σ c_k · 정규화 기저 (상위 제출물과 같은 형태)
  prev : 이전 쌍 렌더에 아핀 + 기저 (현재 v4b, pose2 에서 시작)
  free : 쌍마다 자유 저해상도 이미지 (48x64, 상한 확인용)

    python pose_diag.py --mode gray --pairs 24 --steps 400
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, SH, SW, load_gt, nets, pose_out
import archive  # noqa: E402
from model import even_frames_prev, parse_rcfg  # noqa: E402
from pose_fit import renders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gray", choices=("gray", "prev", "free"))
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--renderer", default=str(CACHE / "renderer_w243240ft.pt"))
    ap.add_argument("--renderer-cfg", default="24,32,40")
    ap.add_argument("--pose2", default=str(CACHE / "pose2_w24b.bin"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8016)
    ap.add_argument("--b-lr", type=float, default=0.02, help="prev 모드 기저 학습률 (0 이면 고정)")
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    _, seg, pose = load_gt()
    n = len(seg)
    sel = np.linspace(0, n - 1, args.pairs).round().astype(int)
    target_all = torch.from_numpy(pose)
    scale = target_all.std(0)
    target = target_all[sel]
    pre = np.load(CACHE / "seg_pre.npy")
    odd_all, prev_all = renders(args.renderer, parse_rcfg(args.renderer_cfg, n), seg, pre, 6)
    odd, prev = odd_all[sel], prev_all[sel]
    del odd_all, prev_all
    viz = LiveVis(f"comma vcc · pose 진단 ({args.mode})", port=args.port).start()

    if args.mode == "gray":
        raw = (torch.randn(args.k, 3, 24, 32)).requires_grad_(True)
        c = torch.zeros(len(sel), args.k, requires_grad=True)
        groups = [{"params": [raw], "lr": 0.003}, {"params": [c], "lr": 0.03}]

        def even():
            b = F.interpolate(raw, size=(SH, SW), mode="bicubic", align_corners=False)
            b = b - b.mean((1, 2, 3), keepdim=True)
            b = b / b.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
            return (127.5 + 32 * torch.einsum("nk,kchw->nchw", c, b) / math.sqrt(args.k)).clamp(0, 255)
    elif args.mode == "prev":
        pos = archive.unpack_pose2(open(args.pose2, "rb").read())
        B = pos["B"].clone().requires_grad_(args.b_lr > 0)
        a = pos["a"][sel].clone().requires_grad_(True)
        c = pos["c"][sel].clone().requires_grad_(True)
        groups = [{"params": [a, c], "lr": 0.02}] + ([{"params": [B], "lr": args.b_lr}] if args.b_lr > 0 else [])

        def even():
            return even_frames_prev(prev, a, c, B)
    else:
        img = torch.full((len(sel), 3, 48, 64), 127.5, requires_grad=True)
        groups = [{"params": [img], "lr": 2.0}]

        def even():
            return F.interpolate(img, size=(SH, SW), mode="bicubic", align_corners=False).clamp(0, 255)

    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=1e-5)
    t0 = time.time()
    with torch.no_grad():
        err0 = pose_out(net, even(), odd) - target
    print("시작 차원별 RMS/std", (err0.pow(2).mean(0).sqrt() / scale).numpy().round(3), f"dist {err0.pow(2).mean():.6f}", flush=True)
    for step in range(1, args.steps + 1):
        out = pose_out(net, even(), odd)
        res = out - target
        loss = (res / scale).square().mean() + 0.02 * res.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        d = res.detach().pow(2).mean().item()
        viz.log(step, posenet_dist=d, loss=loss.item())
        if step % 50 == 0:
            r = res.detach().pow(2).mean(0).sqrt() / scale
            print(f"step {step}: dist {d:.7f} term {math.sqrt(10 * d):.4f} 차원별 RMS/std {r.numpy().round(3)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
