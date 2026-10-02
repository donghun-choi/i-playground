"""600쌍 전체 pose carrier 피팅 → archive 의 carrier 섹션.

짝수 = base + bicubic(Σ_k c[i,k] B_k)   (base: 회색 127.5 또는 같은 쌍 홀수 프레임)
홀수 프레임은 inflate 와 똑같이 만든다 (int8 왕복 렌더러, float 출력 → 서브픽셀 확장).
  1) bf16 으로 B, c 공동 학습 → 2) fp32 로 계속
  3) B 를 int8 로 고정, c 를 양자화 격자에서 미세조정 → 4) 쌍별 격자 탐욕 탐색
  5) 실제 평가 경로(서브픽셀 확장 → 축소 → PoseNet)로 확인

    python carrier_fit.py --renderer ../cache/renderer.pt --out ../cache/carrier.bin
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import CACHE, SH, SW, downsample, load_gt, nets, pose_out
import archive  # noqa: E402
from model import even_frames, expand_fine, make_renderer, quantize_roundtrip, render  # noqa: E402


@torch.inference_mode()
def render_odd(renderer_path, seg, cfg=None, bs=8):
    """inflate 와 같은 홀수 프레임: int8 왕복 렌더러, float 출력."""
    G = make_renderer(cfg)
    G.load_state_dict(torch.load(renderer_path))
    quantize_roundtrip(G)
    G.eval()
    out = torch.zeros(len(seg), 3, SH, SW)
    for i in range(0, len(seg), bs):
        out[i : i + bs] = render(G, torch.from_numpy(seg[i : i + bs]), torch.arange(i, min(i + bs, len(seg))))
    return out


def pose_dist(net, odd, c, B, target, base, bs=32):
    d = []
    with torch.inference_mode():
        for i in range(0, len(odd), bs):
            even = even_frames(odd[i : i + bs], c[i : i + bs], B, base)
            d.append(((pose_out(net, even, odd[i : i + bs]) - target[i : i + bs]) ** 2).mean(1))
    return torch.cat(d)


@torch.inference_mode()
def exact_pose_dist(net, odd, c, B, target, base, bs=20):
    """평가와 같은 경로: float → 서브픽셀 정수 프레임 → bilinear 축소 → PoseNet."""
    d = []
    for i in range(0, len(odd), bs):
        o = odd[i : i + bs]
        e = even_frames(o, c[i : i + bs], B, base)
        pair = torch.stack([e, o], 1).flatten(0, 1).permute(0, 2, 3, 1).numpy()
        back = downsample(torch.from_numpy(expand_fine(pair))).view(-1, 2, 3, SH, SW)
        d.append(((pose_out(net, back[:, 0], back[:, 1]) - target[i : i + bs]) ** 2).mean(1))
    return torch.cat(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default=str(CACHE / "renderer.pt"))
    ap.add_argument("--out", default=str(CACHE / "carrier.bin"))
    ap.add_argument("--base", default="gray")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--ch", type=int, default=3, help="기저 채널 (3=RGB, 1=회색)")
    ap.add_argument("--bh", type=int, default=24)
    ap.add_argument("--bw", type=int, default=32)
    ap.add_argument("--binit", type=float, default=30.0)
    ap.add_argument("--bf16-steps", type=int, default=1500)
    ap.add_argument("--fp32-steps", type=int, default=3000)
    ap.add_argument("--q-steps", type=int, default=800)
    ap.add_argument("--cbits", type=int, default=12, help="계수 양자화: 차원별 범위를 2^cbits 칸으로")
    ap.add_argument("--greedy-rounds", type=int, default=2)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8005)
    ap.add_argument("--dimw", type=float, default=0.0, help="차원별 가중치 = (1/분산)^dimw 정규화. 0 이면 평가와 같은 MSE")
    ap.add_argument("--init", default=None, help="기존 carrier.bin 에서 B, c 로 시작 (렌더러가 바뀌었을 때)")
    ap.add_argument("--renderer-cfg", default=None, help="RendererV2 이면 'width,fdim'")
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    _, seg, pose = load_gt()
    target = torch.from_numpy(pose)
    n = len(seg)
    viz = LiveVis(f"comma vcc · carrier 피팅 base={args.base} k={args.k} {args.ch}x{args.bh}x{args.bw}", port=args.port).start()
    t0 = time.time()
    cfg = None
    if args.renderer_cfg:
        w, fd = (int(v) for v in args.renderer_cfg.split(","))
        cfg = (w, fd, n, (1, 1, 2, 4))
    odd = render_odd(args.renderer, seg, cfg)
    print(f"홀수 프레임 렌더 {time.time() - t0:.0f}s", flush=True)

    if args.init:
        car = archive.unpack_carrier(open(args.init, "rb").read())
        B = car["B"].clone().requires_grad_(True)
        c = car["c"].clone().requires_grad_(True)
        print(f"초기값: {args.init} posenet_dist {pose_dist(net, odd, c.detach(), B.detach(), target, args.base).mean():.7f}", flush=True)
    else:
        B = (torch.randn(args.k, args.ch, args.bh, args.bw) * args.binit).requires_grad_(True)
        c = (torch.randn(n, args.k) * 0.3).requires_grad_(True)
    step = 0
    # 차원별 가중치: pose 0번(전진)의 분산이 나머지보다 ~1000배 커서 그냥 MSE 면 회전 차원을 못 배운다
    wdim = target.var(0).clamp_min(1e-8) ** (-args.dimw)
    wdim = wdim / wdim.mean()

    def train(n_steps, bf16, groups, cq=None, weighted=True):
        nonlocal step
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=groups[-1]["lr"] * 0.01)
        for _ in range(n_steps):
            step += 1
            perm = torch.randperm(n)[: args.bs]
            cc = c[perm] if cq is None else cq(c[perm])
            even = even_frames(odd[perm], cc, B, args.base)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                out = pose_out(net, even, odd[perm])
            w = wdim if weighted else torch.ones(6)
            loss = (((out.float() - target[perm]) ** 2) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            viz.log(step, train_posenet_dist=loss.item())
            if step % 250 == 0:
                d = pose_dist(net, odd, c.detach() if cq is None else cq(c.detach()), B.detach(), target, args.base).mean().item()
                viz.log(step, posenet_dist=d, pose_term=float(np.sqrt(10 * d)))
                print(f"step {step}: posenet_dist {d:.7f} term {np.sqrt(10 * d):.4f} ({time.time() - t0:.0f}s)", flush=True)

    train(args.bf16_steps, True, [{"params": [B], "lr": args.lr * 20}, {"params": [c], "lr": args.lr}])
    train(args.fp32_steps, False, [{"params": [B], "lr": args.lr * 10}, {"params": [c], "lr": args.lr * 0.5}])

    # ---- 기저 int8 고정
    with torch.no_grad():
        B_scale = B.abs().amax((1, 2, 3)) / 127
        B_q = (B / B_scale.view(-1, 1, 1, 1)).round().clamp(-127, 127)
        B = (B_q * B_scale.view(-1, 1, 1, 1)).detach()
        B_q, B_scale = B_q.numpy().astype(np.int8), B_scale.numpy()
        rng = (c.amax(0) - c.amin(0)).clamp_min(1e-6) * 1.2
        qstep = (rng / 2**args.cbits).numpy().astype(np.float32)
    qs = torch.from_numpy(qstep)

    def cq(x):  # 격자 양자화 (STE)
        return x + ((x / qs).round() * qs - x).detach()

    train(args.q_steps, False, [{"params": [c], "lr": args.lr * 0.1}], cq=cq, weighted=False)

    # ---- 쌍별 탐욕 탐색: 격자 ±1 칸씩
    with torch.no_grad():
        c_int = (c / qs).round()
        d = pose_dist(net, odd, c_int * qs, B, target, args.base)
        print(f"양자화 후 posenet_dist {d.mean():.7f}", flush=True)
        for rnd in range(args.greedy_rounds):
            worst = torch.argsort(d, descending=True)[: n // 2]
            for kdim in range(args.k):
                for sgn in (1, -1):
                    cand = c_int.clone()
                    cand[worst, kdim] += sgn
                    dn = pose_dist(net, odd[worst], cand[worst] * qs, B, target[worst], args.base)
                    better = dn < d[worst]
                    c_int[worst[better]] = cand[worst[better]]
                    d[worst[better]] = dn[better]
            viz.log(step + rnd + 1, posenet_dist=d.mean().item(), pose_term=float(np.sqrt(10 * d.mean().item())))
            print(f"탐욕 탐색 {rnd + 1}: posenet_dist {d.mean():.7f} term {np.sqrt(10 * d.mean()):.4f}", flush=True)

    blob = archive.pack_carrier(0, args.base, "bicubic", B_q, B_scale, c_int.numpy().astype(np.int64), qstep, args.ch, args.bh, args.bw)
    open(args.out, "wb").write(blob)
    car = archive.unpack_carrier(blob)
    d2 = exact_pose_dist(net, odd, car["c"], car["B"], target, car["base"]).mean().item()
    print(f"carrier {len(blob):,} bytes, 평가 경로 posenet_dist {d2:.7f} term {np.sqrt(10 * d2):.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
