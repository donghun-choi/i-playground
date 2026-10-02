"""pose 실험 2: 짝수 = affine_i(render(M_{i-1})) + bicubic(Σ c[i,k] B_k)

PoseNet 을 '자연스러운 운동' 상태에 두고 (이전 쌍 렌더 → 실제 2프레임 운동),
아핀(6) 과 carrier(k) 로 미세조정. 홀수/이전 프레임은 실제 inflate 처럼 렌더러 출력.

    python carrier_exp2.py --pairs 64 --steps 400 --dimw 0.5
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, SH, SW, load_gt, nets, pose_out
from model import carrier_delta, make_renderer, quantize_roundtrip, render  # noqa: E402

AFF_SCALE = torch.tensor([0.01, 0.01, 2 / SW, 0.01, 0.01, 2 / SH])  # 행렬 원소 0.01 단위, 이동 픽셀 단위


def affine(img, a):
    a = a * AFF_SCALE
    th = torch.stack([torch.stack([1 + a[:, 0], a[:, 1], a[:, 2]], 1), torch.stack([a[:, 3], 1 + a[:, 4], a[:, 5]], 1)], 1)
    g = F.affine_grid(th, img.shape, align_corners=False)
    return F.grid_sample(img, g, mode="bilinear", padding_mode="border", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--fp32-after", type=int, default=100)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--dimw", type=float, default=0.5)
    ap.add_argument("--no-carrier", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    _, seg, pose = load_gt()
    idx = np.linspace(1, 599, args.pairs).astype(int)
    G = make_renderer(None)
    G.load_state_dict(torch.load(CACHE / "renderer_v1.pt"))
    quantize_roundtrip(G)
    G.eval()
    with torch.no_grad():
        odd = torch.cat([render(G, torch.from_numpy(seg[i : i + 1]), torch.tensor([i])) for i in idx])
        prv = torch.cat([render(G, torch.from_numpy(seg[i - 1 : i]), torch.tensor([i - 1])) for i in idx])
    t = torch.from_numpy(pose[idx])
    w = torch.from_numpy(pose).var(0) ** (-args.dimw)
    w = w / w.mean()
    n = len(idx)
    a = torch.zeros(n, 6, requires_grad=True)
    B = (torch.randn(args.k, 3, 24, 32) * 3).requires_grad_(True)
    c = torch.zeros(n, args.k, requires_grad=True)
    groups = [{"params": [a], "lr": 0.02}]
    if not args.no_carrier:
        groups += [{"params": [B], "lr": 0.5}, {"params": [c], "lr": 0.02}]
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=1e-4)

    def even(sel):
        e = affine(prv[sel], a[sel])
        if not args.no_carrier:
            e = e + carrier_delta(c[sel], B)
        return e.clamp(0, 255)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = torch.randperm(n)[: args.bs]
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=step < args.fp32_after):
            out = pose_out(net, even(sel), odd[sel])
        loss = (((out.float() - t[sel]) ** 2) * w).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                err = torch.cat([pose_out(net, even(torch.arange(i, min(i + 16, n))), odd[i : i + 16]) for i in range(0, n, 16)]) - t
            print(f"step {step}: posenet_dist {err.pow(2).mean():.6f} 차원별 RMS {err.pow(2).mean(0).sqrt().numpy().round(4)} |a| {a.abs().mean(0).detach().numpy().round(2)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
