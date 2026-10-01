"""600쌍 전체 pose carrier 피팅 → archive 의 carrier 섹션.

홀수 프레임은 inflate 와 똑같이 만든다 (int8 왕복 렌더러 + 반올림).
  1) bf16 으로 기저 B 와 계수 c 공동 학습 → 2) fp32 로 계속
  3) B 를 int8 로 고정, c 를 양자화 격자에서 미세조정 (STE) → 4) 쌍별 격자 탐욕 탐색

    python carrier_fit.py --renderer ../cache/renderer.pt --out ../cache/carrier.bin
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, SH, SW, load_gt, nets, pose_out
import archive  # noqa: E402
from model import Renderer, carrier_delta, quantize_roundtrip  # noqa: E402


def ste_round(x):
    return x + (x.round() - x).detach()


@torch.inference_mode()
def render_odd(renderer_path, seg, bs=8):
    G = Renderer()
    G.load_state_dict(torch.load(renderer_path))
    quantize_roundtrip(G)
    G.eval()
    out = torch.zeros(len(seg), 3, SH, SW)
    for i in range(0, len(seg), bs):
        out[i : i + bs] = G(torch.from_numpy(seg[i : i + bs])).round().clamp(0, 255)
    return out


def pose_dist(net, odd, c, B, target, bs=32):
    """inflate 와 같은 계산으로 쌍별 posenet 왜곡."""
    d = []
    with torch.inference_mode():
        for i in range(0, len(odd), bs):
            even = (odd[i : i + bs] + carrier_delta(c[i : i + bs], B)).clamp(0, 255).round()
            d.append(((pose_out(net, even, odd[i : i + bs]) - target[i : i + bs]) ** 2).mean(1))
    return torch.cat(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default=str(CACHE / "renderer.pt"))
    ap.add_argument("--out", default=str(CACHE / "carrier.bin"))
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--bres", type=int, default=16)
    ap.add_argument("--dct", action="store_true")
    ap.add_argument("--bf16-steps", type=int, default=1500)
    ap.add_argument("--fp32-steps", type=int, default=1500)
    ap.add_argument("--q-steps", type=int, default=600)
    ap.add_argument("--cbits", type=int, default=10, help="계수 양자화: 차원별 범위를 2^cbits 칸으로")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8005)
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    _, seg, pose = load_gt()
    target = torch.from_numpy(pose)
    n = len(seg)
    viz = LiveVis(f"comma vcc · carrier 피팅 k={args.k} {'dct' if args.dct else f'bres={args.bres}'}", port=args.port).start()
    t0 = time.time()
    odd = render_odd(args.renderer, seg)
    print(f"홀수 프레임 렌더 {time.time() - t0:.0f}s", flush=True)

    bh, bw = SH // args.bres, SW // args.bres
    if args.dct:
        B = archive.dct_basis(args.k, bh, bw)
        params = []
    else:
        B = (torch.randn(args.k, 1, bh, bw) * 5).requires_grad_(True)
        params = [{"params": [B], "lr": args.lr * 20}]
    c = torch.zeros(n, args.k, requires_grad=True)
    params.append({"params": [c], "lr": args.lr})

    step = 0

    def train(n_steps, bf16, opt_params, c_quant=None):
        nonlocal step
        opt = torch.optim.Adam(opt_params)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=opt_params[-1]["lr"] * 0.02)
        for _ in range(n_steps):
            step += 1
            perm = torch.randperm(n)[: args.bs]
            cc = c[perm] if c_quant is None else c_quant(c[perm])
            even = ste_round((odd[perm] + carrier_delta(cc, B)).clamp(0, 255))
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                out = pose_out(net, even, odd[perm])
            loss = ((out.float() - target[perm]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            viz.log(step, train_posenet_dist=loss.item())
            if step % 250 == 0:
                d = pose_dist(net, odd, c.detach() if c_quant is None else c_quant(c.detach()), B.detach(), target).mean().item()
                viz.log(step, posenet_dist=d, pose_term=float(np.sqrt(10 * d)))
                print(f"step {step}: posenet_dist {d:.6f} term {np.sqrt(10 * d):.4f} ({time.time() - t0:.0f}s)", flush=True)

    train(args.bf16_steps, True, params)
    train(args.fp32_steps, False, params)

    # ---- 기저 int8 고정
    with torch.no_grad():
        if args.dct:
            B_q = B_scale = None
            Bf = B
        else:
            B_scale = B.abs().amax((1, 2, 3)) / 127
            B_q = (B / B_scale.view(-1, 1, 1, 1)).round().clamp(-127, 127)
            Bf = B_q * B_scale.view(-1, 1, 1, 1)
            B_q, B_scale = B_q.numpy().astype(np.int8), B_scale.numpy()
    B = Bf.detach()
    # ---- 계수 양자화 격자: 차원별 범위 / 2^cbits
    with torch.no_grad():
        rng = (c.amax(0) - c.amin(0)).clamp_min(1e-6) * 1.1
        qstep = (rng / 2**args.cbits).numpy().astype(np.float32)
    qs = torch.from_numpy(qstep)

    def c_quant(x):
        return ste_round(x / qs) * qs

    train(args.q_steps, False, [{"params": [c], "lr": args.lr * 0.2}], c_quant=c_quant)

    # ---- 쌍별 탐욕 탐색: 격자 ±1 칸씩 바꿔 보며 나아지면 채택
    with torch.no_grad():
        c_int = (c / qs).round()
        d = pose_dist(net, odd, c_int * qs, B, target)
        print(f"양자화 후 posenet_dist {d.mean():.6f}", flush=True)
        for rnd in range(2):
            worst = torch.argsort(d, descending=True)[: n // 2]
            for kdim in range(args.k):
                for sgn in (1, -1):
                    cand = c_int.clone()
                    cand[worst, kdim] += sgn
                    dn = pose_dist(net, odd[worst], cand[worst] * qs, B, target[worst])
                    better = dn < d[worst]
                    c_int[worst[better]] = cand[worst[better]]
                    d[worst[better]] = dn[better]
            viz.log(step + rnd + 1, posenet_dist=d.mean().item(), pose_term=float(np.sqrt(10 * d.mean().item())))
            print(f"탐욕 탐색 {rnd + 1}: posenet_dist {d.mean():.6f} term {np.sqrt(10 * d.mean()):.4f}", flush=True)

    blob = archive.pack_carrier(1 if args.dct else 0, B_q, B_scale, c_int.numpy().astype(np.int64), qstep, bh, bw)
    open(args.out, "wb").write(blob)
    # 왕복 확인: inflate 가 읽는 값으로 다시 계산
    B2, c2 = archive.unpack_carrier(blob)
    d2 = pose_dist(net, odd, c2, B2, target).mean().item()
    print(f"carrier {len(blob):,} bytes, 왕복 후 posenet_dist {d2:.6f} term {np.sqrt(10 * d2):.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
