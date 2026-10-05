"""pose v2 피팅 (600쌍): 짝수 = 아핀_i(이전 쌍 렌더) + bicubic(Σ c[i,k] B_k)

첫 쌍의 '이전 프레임' 은 GT 짝수 프레임 0 의 SegNet 맵을 렌더한 것 (seg 스트림 맨 앞에 한 장 더 보낸다).
  1) bf16 (차원 가중) → 2) fp32 (차원 가중 → 일반 MSE) → 3) B int8 고정, c/a 격자 양자화 + STE 미세조정
  4) 쌍별 격자 ±1 탐욕 탐색 → 5) 평가 경로(서브픽셀 확장)로 확인

    python pose_fit.py --renderer ../cache/renderer_v2w48.pt --renderer-cfg 48,8 --out ../cache/pose2.bin
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common import CACHE, SH, SW, downsample, load_gt, nets, pose_out
import archive  # noqa: E402
from model import parse_rcfg  # noqa: E402
from model import even_frames_prev, expand_fine, make_renderer, quantize_roundtrip, render  # noqa: E402


def renders(renderer_path, cfg, seg, pre, bits=8, bs=8, device="cpu"):
    """inflate 와 같은 홀수 프레임과 '이전 프레임' 들."""
    G = make_renderer(cfg)
    G.load_state_dict(torch.load(renderer_path, map_location="cpu"))
    quantize_roundtrip(G, bits)
    G.eval().to(device)
    n = len(seg)
    odd = torch.zeros(n, 3, SH, SW, device=device)
    with torch.inference_mode():
        for i in range(0, n, bs):
            odd[i : i + bs] = render(G, torch.from_numpy(seg[i : i + bs]).to(device), torch.arange(i, min(i + bs, n)))
        prev0 = render(G, torch.from_numpy(pre).to(device), torch.zeros(1, dtype=torch.long))
    return odd, torch.cat([prev0, odd[:-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default=str(CACHE / "renderer_v1.pt"))
    ap.add_argument("--renderer-cfg", default=None, help="렌더러 구성: v1 폭 'c1,c2,c3' 또는 RendererV2 'width,fdim'")
    ap.add_argument("--out", default=str(CACHE / "pose2.bin"))
    ap.add_argument("--rbits", type=int, default=8, help="렌더러 저장 비트 수 (build_archive 와 같게)")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--bf16-steps", type=int, default=1200)
    ap.add_argument("--fp32-steps", type=int, default=2400)
    ap.add_argument("--q-steps", type=int, default=600)
    ap.add_argument("--cbits", type=int, default=12)
    ap.add_argument("--bbits", type=int, default=6, help="기저 B 저장 비트 수")
    ap.add_argument("--dimw", type=float, default=0.5)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--greedy-rounds", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8013)
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    small, seg, pose = load_gt()
    n = len(seg)
    target = torch.from_numpy(pose)
    t0 = time.time()
    with torch.inference_mode():
        pre = net.segnet(torch.from_numpy(np.array(small[0:1]))).argmax(1).numpy().astype(np.uint8)
    np.save(CACHE / "seg_pre.npy", pre)
    cfg = parse_rcfg(args.renderer_cfg, n)
    odd, prev = renders(args.renderer, cfg, seg, pre, args.rbits)
    print(f"렌더 {time.time() - t0:.0f}s", flush=True)
    viz = LiveVis(f"comma vcc · pose v2 피팅 (이전 렌더 + 아핀 + carrier k={args.k})", port=args.port).start()

    a = torch.zeros(n, 6, requires_grad=True)
    c = torch.zeros(n, args.k, requires_grad=True)
    B = (torch.randn(args.k, 3, 24, 32) * 3).requires_grad_(True)
    wdim = target.var(0) ** (-args.dimw)
    wdim = wdim / wdim.mean()
    step = 0

    def evaluate(aa, cc, BB, bs=40):
        with torch.inference_mode():
            return torch.cat([pose_out(net, even_frames_prev(prev[i : i + bs], aa[i : i + bs], cc[i : i + bs], BB), odd[i : i + bs])
                              for i in range(0, n, bs)]) - target

    def train(n_steps, bf16, groups, weighted, q=None):
        nonlocal step
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=1e-5)
        for _ in range(n_steps):
            step += 1
            sel = torch.randperm(n)[: args.bs]
            aa, cc = (a[sel], c[sel]) if q is None else q(a[sel], c[sel])
            e = even_frames_prev(prev[sel], aa, cc, B)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                out = pose_out(net, e, odd[sel])
            sq = (out.float() - target[sel]) ** 2
            loss = (sq * wdim).mean() if weighted else sq.mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            viz.log(step, train_loss=loss.item())
            if step % 200 == 0:
                aa, cc = (a.detach(), c.detach()) if q is None else q(a.detach(), c.detach())
                err = evaluate(aa, cc, B.detach())
                d = err.pow(2).mean().item()
                viz.log(step, posenet_dist=d, pose_term=float(np.sqrt(10 * d)))
                print(f"step {step}: posenet_dist {d:.7f} term {np.sqrt(10 * d):.4f} 차원별 RMS {err.pow(2).mean(0).sqrt().numpy().round(4)} ({time.time() - t0:.0f}s)", flush=True)

    groups = [{"params": [a], "lr": 0.02}, {"params": [c], "lr": 0.02}, {"params": [B], "lr": 0.5}]
    train(args.bf16_steps, True, groups, True)
    train(args.fp32_steps // 2, False, [dict(g, lr=g["lr"] * 0.5) for g in groups], True)
    train(args.fp32_steps // 2, False, [dict(g, lr=g["lr"] * 0.25) for g in groups], False)

    # ---- B int8 고정, 격자 양자화
    with torch.no_grad():
        bq = 2 ** (args.bbits - 1) - 1
        B_scale = B.abs().amax((1, 2, 3)) / bq
        B_q = (B / B_scale.view(-1, 1, 1, 1)).round().clamp(-bq, bq)
        B.data = B_q * B_scale.view(-1, 1, 1, 1)
        c_step = ((c.amax(0) - c.amin(0)).clamp_min(1e-6) * 1.2 / 2**args.cbits)
        a_step = ((a.amax(0) - a.amin(0)).clamp_min(1e-6) * 1.2 / 2**args.cbits)
    B.requires_grad_(False)

    def q(aa, cc):  # STE 격자
        return aa + ((aa / a_step).round() * a_step - aa).detach(), cc + ((cc / c_step).round() * c_step - cc).detach()

    train(args.q_steps, False, [{"params": [a], "lr": 0.002}, {"params": [c], "lr": 0.002}], False, q=q)

    # ---- 쌍별 탐욕 탐색
    with torch.no_grad():
        a_int, c_int = (a / a_step).round(), (c / c_step).round()
        d = evaluate(a_int * a_step, c_int * c_step, B).pow(2).mean(1)
        print(f"양자화 후 posenet_dist {d.mean():.7f}", flush=True)
        for rnd in range(args.greedy_rounds):
            for which, vec, stp in (("a", a_int, a_step), ("c", c_int, c_step)):
                for kd in range(vec.shape[1]):
                    for sgn in (1, -1):
                        cand = vec.clone()
                        cand[:, kd] += sgn
                        aa = cand * a_step if which == "a" else a_int * a_step
                        cc = cand * c_step if which == "c" else c_int * c_step
                        dn = evaluate(aa, cc, B).pow(2).mean(1)
                        better = dn < d
                        vec[better] = cand[better]
                        d[better] = dn[better]
            print(f"탐욕 탐색 {rnd + 1}: posenet_dist {d.mean():.7f} term {np.sqrt(10 * d.mean()):.4f} ({time.time() - t0:.0f}s)", flush=True)

    blob = archive.pack_pose2(B_q.numpy().astype(np.int8), B_scale.numpy(), c_int.numpy().astype(np.int64), c_step.numpy(),
                              a_int.numpy().astype(np.int64), a_step.numpy())
    open(args.out, "wb").write(blob)
    pos = archive.unpack_pose2(blob)
    # 평가 경로: float → 서브픽셀 정수 프레임 → 축소 → PoseNet
    dd = []
    with torch.inference_mode():
        for i in range(0, n, 20):
            e = even_frames_prev(prev[i : i + 20], pos["a"][i : i + 20], pos["c"][i : i + 20], pos["B"])
            pair = torch.stack([e, odd[i : i + 20]], 1).flatten(0, 1).permute(0, 2, 3, 1).numpy()
            back = downsample(torch.from_numpy(expand_fine(pair))).view(-1, 2, 3, SH, SW)
            dd.append(((pose_out(net, back[:, 0], back[:, 1]) - target[i : i + 20]) ** 2).mean(1))
    d2 = torch.cat(dd).mean().item()
    print(f"pose2 {len(blob):,} B, 평가 경로 posenet_dist {d2:.7f} term {np.sqrt(10 * d2):.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
