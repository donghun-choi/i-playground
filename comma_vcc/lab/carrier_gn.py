"""carrier 계수 쌍별 Gauss-Newton 다듬기 (기저 B 는 고정).

쌍마다 c (k차원) → pose (6차원). J (6×k) 를 중앙 유한차분으로 구해 Levenberg-Marquardt 스텝,
반복 후 양자화 격자에 맞추고, 격자 위에서 한 번 더 GN(반올림) + ±1 탐욕 탐색.

    python carrier_gn.py --carrier ../cache/carrier_v1.bin --renderer ../cache/renderer_v1.pt --out ../cache/carrier_v1gn.bin
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import CACHE, load_gt, nets, pose_out
import archive  # noqa: E402
from carrier_fit import exact_pose_dist, render_odd
from model import even_frames  # noqa: E402


@torch.inference_mode()
def poses(net, odd, c, B, base, bs=40):
    return torch.cat([pose_out(net, even_frames(odd[i : i + bs], c[i : i + bs], B, base), odd[i : i + bs]) for i in range(0, len(odd), bs)])


@torch.inference_mode()
def jacobian(net, odd, c, B, base, h):
    """중앙 차분 J: (n,6,k). h: (k,) 차원별 간격."""
    n, k = c.shape
    J = torch.zeros(n, 6, k)
    for d in range(k):
        e = torch.zeros(k)
        e[d] = h[d]
        J[:, :, d] = (poses(net, odd, c + e, B, base) - poses(net, odd, c - e, B, base)) / (2 * h[d])
    return J


def lm_step(J, r, lam):
    """min ||J δ - r||² + lam ||δ||²  → δ = J^T (J J^T + lam I)^{-1} r   (6x6 풀이)"""
    A = J @ J.transpose(1, 2) + lam * torch.eye(6)
    return (J.transpose(1, 2) @ torch.linalg.solve(A, r[:, :, None]))[:, :, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrier", default=str(CACHE / "carrier_v1.bin"))
    ap.add_argument("--renderer", default=str(CACHE / "renderer_v1.pt"))
    ap.add_argument("--renderer-cfg", default=None)
    ap.add_argument("--out", default=str(CACHE / "carrier_gn.bin"))
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    net = nets()
    _, seg, pose = load_gt()
    n = args.pairs
    target = torch.from_numpy(pose[:n])
    cfg = None
    if args.renderer_cfg:
        w, fd = (int(v) for v in args.renderer_cfg.split(","))
        cfg = (w, fd, len(seg), (1, 1, 2, 4))
    t0 = time.time()
    odd = render_odd(args.renderer, seg, cfg)[:n]
    blob = open(args.carrier, "rb").read()
    car = archive.unpack_carrier(blob)
    B, c, base = car["B"], car["c"][:n].clone(), car["base"]
    # 양자화 간격 (원래 carrier 의 것 그대로)
    k = c.shape[1]
    qs = torch.from_numpy(np.frombuffer(archive.unxz(blob[archive._HEAD:]), np.float32, k).copy())

    def dist(cc):
        return ((poses(net, odd, cc, B, base) - target) ** 2).mean(1)

    d = dist(c)
    print(f"시작 posenet_dist {d.mean():.7f} ({time.time() - t0:.0f}s)", flush=True)
    lam = torch.full((n, 1, 1), 1e-3)
    for it in range(args.iters):
        J = jacobian(net, odd, c, B, base, qs * 4)
        r = target - poses(net, odd, c, B, base)
        step = lm_step(J, r, lam)
        cand = c + step
        dn = dist(cand)
        better = dn < d
        c[better] = cand[better]
        d[better] = dn[better]
        lam[better] *= 0.3
        lam[~better] *= 10
        print(f"GN {it + 1}: posenet_dist {d.mean():.7f} (개선된 쌍 {better.float().mean():.0%}) ({time.time() - t0:.0f}s)", flush=True)

    # 격자에 맞추고, 격자 위에서 다듬기
    c_int = (c / qs).round()
    d = dist(c_int * qs)
    print(f"양자화 후 {d.mean():.7f}", flush=True)
    for rnd in range(2):
        J = jacobian(net, odd, c_int * qs, B, base, qs * 4)
        r = target - poses(net, odd, c_int * qs, B, base)
        cand = (c_int + (lm_step(J, r, 1e-3 * torch.ones(n, 1, 1)) / qs).round())
        dn = dist(cand * qs)
        better = dn < d
        c_int[better] = cand[better]
        d[better] = dn[better]
        for kd in range(k):
            for sgn in (1, -1):
                cand = c_int.clone()
                cand[:, kd] += sgn
                dn = dist(cand * qs)
                better = dn < d
                c_int[better] = cand[better]
                d[better] = dn[better]
        print(f"격자 다듬기 {rnd + 1}: {d.mean():.7f} term {np.sqrt(10 * d.mean()):.4f} ({time.time() - t0:.0f}s)", flush=True)

    if n == len(seg):
        kind, base_i, mode_i, _, kk, C, bh, bw = __import__("struct").unpack_from("<BBBHHBHH", blob, 0)
        body = archive.unxz(blob[archive._HEAD:])
        scale = np.frombuffer(body, np.float32, kk, 4 * kk)
        B_q = np.frombuffer(body, np.int8, kk * C * bh * bw, 8 * kk).reshape(kk, C, bh, bw)
        out = archive.pack_carrier(kind, archive.BASES[base_i], archive.MODES[mode_i], B_q, scale,
                                   c_int.numpy().astype(np.int64), qs.numpy(), C, bh, bw)
        open(args.out, "wb").write(out)
        car2 = archive.unpack_carrier(out)
        d2 = exact_pose_dist(net, odd, car2["c"], car2["B"], target, car2["base"]).mean().item()
        print(f"carrier {len(out):,} B, 평가 경로 posenet_dist {d2:.7f} term {np.sqrt(10 * d2):.4f}", flush=True)


if __name__ == "__main__":
    main()
