"""pose carrier 실험: 짝수 = 홀수 + Σ_steps J(e_s)^T a_s  (J: PoseNet pose 출력의 짝수 프레임 Jacobian)

inflate 때 J^T a 는 (a · pose) 를 짝수 프레임으로 한 번 미분하면 나온다 → 저장할 것은 쌍마다 6×steps 개 숫자.
압축 때는 J 를 전부 구해 Gauss-Newton 으로 a 를 푼다.

    python vjp_exp.py --pairs 32 --steps 3
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import load_gt, nets, pose_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--gn-iters", type=int, default=8)
    ap.add_argument("--gray", action="store_true", help="섭동을 회색(RGB 동일)으로 제한")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--base", default="odd", help="odd | blend(이전 홀수와 반반) | extrap(2*이전-현재)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    net = nets()
    small, seg, pose = load_gt()
    idx = np.linspace(1, 599, args.pairs).astype(int)
    odd = torch.from_numpy(np.stack([small[2 * i + 1] for i in idx])).round()
    t = torch.from_numpy(pose[idx])
    n = len(idx)
    bs = 16

    def f(even):
        with torch.inference_mode():
            return torch.cat([pose_out(net, even[i : i + bs], odd[i : i + bs]) for i in range(0, n, bs)])

    def jac(even):
        J = torch.zeros(n, 6, 3, *even.shape[2:])
        for i in range(0, n, bs):
            e = even[i : i + bs].clone().requires_grad_(True)
            out = pose_out(net, e, odd[i : i + bs])
            for d in range(6):
                (g,) = torch.autograd.grad(out[:, d].sum(), e, retain_graph=d < 5)
                if args.gray:
                    g = g.mean(1, keepdim=True).expand_as(g)
                J[i : i + bs, d] = g
        return J

    def dist(p):
        return ((p - t) ** 2).mean(1)

    prev = torch.from_numpy(np.stack([small[2 * i - 1] for i in idx])).round()
    e = {"odd": odd, "blend": (0.5 * prev + 0.5 * odd).round(), "extrap": (2 * prev - odd).clamp(0, 255).round()}[args.base].clone()
    t0 = time.time()
    print(f"시작: posenet_dist {dist(f(e)).mean():.4f}", flush=True)
    for s in range(args.steps):
        J = jac(e)
        Jf = J.flatten(2)
        G = Jf @ Jf.transpose(1, 2)  # (n,6,6)
        Ginv = torch.linalg.inv(G + 1e-9 * torch.eye(6))
        a = torch.zeros(n, 6)
        for it in range(args.gn_iters):
            cand = (e + torch.einsum("nd,nd...->n...", a, J)).clamp(0, 255)
            r = t - f(cand)
            a = a + (Ginv @ r[:, :, None])[:, :, 0]
        delta = torch.einsum("nd,nd...->n...", a, J)
        cont = dist(f((e + delta).clamp(0, 255))).mean().item()
        e = (e + delta).clamp(0, 255).round()
        d = dist(f(e))
        print(f"step {s + 1}: 연속값 {cont:.6f} / 반올림 후 {d.mean():.6f} (term {np.sqrt(10 * d.mean()):.4f}) "
              f"|delta| 평균 {delta.abs().mean():.2f} 최대 {delta.abs().max():.1f} |a| {a.abs().mean(0).numpy()} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
