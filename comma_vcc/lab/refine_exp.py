"""inflate 시점 보정 실험: 렌더한 홀수 프레임을 SegNet 그래디언트로 몇 번 다듬는다 (목표 맵은 디코더가 안다).

손실: 각 픽셀에서 정답 logit 이 다른 최대 logit 보다 margin 이상 크게 (hinge). 경계 근처만 움직인다.
반복 수별 disagreement 와 걸린 시간을 본다.

    python refine_exp.py --frames 16 --steps 4 --lr 2.0
"""

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, load_gt, nets
from model import Renderer, quantize_roundtrip  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2.0)
    ap.add_argument("--margin", type=float, default=1.0)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--renderer", default=str(CACHE / "renderer_v1.pt"))
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    net = nets()
    _, seg, _ = load_gt()
    G = Renderer()
    G.load_state_dict(torch.load(args.renderer))
    quantize_roundtrip(G)
    G.eval()
    idx = np.linspace(0, 599, args.frames).astype(int)
    errs = np.zeros(args.steps + 1)
    t_step = 0.0
    for i in range(0, len(idx), args.bs):
        m = torch.from_numpy(seg[idx[i : i + args.bs]]).long()
        with torch.no_grad():
            x = G(m)
        for s in range(args.steps + 1):
            x.requires_grad_(True)
            t = time.time()
            logits = net.segnet(x)
            with torch.no_grad():
                errs[s] += (logits.argmax(1) != m).float().mean((1, 2)).sum().item()
            if s == args.steps:
                break
            true = logits.gather(1, m[:, None])[:, 0]
            other = logits.scatter(1, m[:, None], -1e4).amax(1)
            loss = F.relu(args.margin - (true - other)).sum()
            (g,) = torch.autograd.grad(loss, x)
            with torch.no_grad():
                # 이미지별 최대 그래디언트로 정규화: 가장 많이 움직이는 픽셀이 lr 만큼
                x = (x - args.lr * g / g.abs().amax((1, 2, 3), keepdim=True).clamp_min(1e-12)).clamp(0, 255)
            t_step += time.time() - t
    errs /= len(idx)
    print(" → ".join(f"{e:.6f}" for e in errs), f"| 반복 1회 {t_step / args.steps / len(idx):.3f}s/frame")


if __name__ == "__main__":
    main()
