"""pose v2 쌍별 파라미터 다듬기: 기저 B 고정, 쌍들을 고정 배치로 나눠 배치마다 따로 Adam → 매 에폭 모든 쌍이 한 번씩 갱신.

(전체 피팅에서는 배치가 무작위라 쌍마다 25스텝에 한 번만 그래디언트를 받아 수렴이 느렸다.)
마지막에 격자 양자화 + STE 미세조정 + ±1 탐욕 탐색, 평가 경로로 확인.

    python pose_refine.py --pose2 ../cache/pose2_v1r.bin --renderer ../cache/renderer_v1.pt --rbits 6 --out ../cache/pose2_v1r_ref.bin
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, SH, SW, downsample, fine_q, fine_q_ste, load_gt, nets, pose_out, setup_device
import archive  # noqa: E402
from model import parse_rcfg  # noqa: E402
from model import even_frames_prev, expand_fine  # noqa: E402
from pose_fit import renders


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose2", required=True)
    ap.add_argument("--renderer", default=str(CACHE / "renderer_v1.pt"))
    ap.add_argument("--renderer-cfg", default=None)
    ap.add_argument("--rbits", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--q-epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--cbits", type=int, default=12)
    ap.add_argument("--greedy-rounds", type=int, default=2)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8014)
    ap.add_argument("--train-b", type=float, default=0.0, help="> 0 이면 기저 B 도 이 학습률로 함께 학습 (렌더러가 바뀌어 회전 차원이 나빠졌을 때)")
    ap.add_argument("--bbits", type=int, default=6)
    ap.add_argument("--b-res", default=None, help="기저 B 해상도를 바꿔서 시작 (예: 12,16). --train-b 필요")
    ap.add_argument("--b-qat", action="store_true", help="기저 학습 때 B 를 --bbits 격자로 가짜 양자화 (STE) → 저비트 기저에 적응")
    ap.add_argument("--plain-epochs", type=int, default=0, help="본 학습(가중/B 학습) 뒤 B 고정 + 일반 MSE 로 더 학습할 에폭")
    ap.add_argument("--dimw", type=float, default=0.0, help="본 학습 손실의 차원별 가중 (1/분산)^dimw (양자화 단계는 일반 MSE)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (Colab GPU)")
    ap.add_argument("--limit", type=int, default=0, help="> 0 이면 앞 N쌍만 (드라이버 점검용, 결과 blob 은 쓸모없다)")
    ap.add_argument("--qf", action="store_true", help="학습/탐욕 탐색을 평가 경로 (서브픽셀 정수화 후 축소, fine_q) 로. 홀수는 fine_q, 짝수는 STE")
    ap.add_argument("--resume", action="store_true", help="--out + '.state.pt' 체크포인트에서 이어서 (컨테이너 재시작 대비, 10 에폭마다 저장)")
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    dev = setup_device(args.device)
    net = nets(dev)
    small, seg, pose = load_gt()
    if args.limit:
        seg, pose = seg[: args.limit], pose[: args.limit]
    n = len(seg)
    target = torch.from_numpy(pose).to(dev)
    wdim = target.var(0) ** (-args.dimw)
    wdim = wdim / wdim.mean()
    pre = np.load(CACHE / "seg_pre.npy")
    cfg = parse_rcfg(args.renderer_cfg, n)
    t0 = time.time()
    odd, prev = renders(args.renderer, cfg, seg, pre, args.rbits, device=dev)
    odd_in = fine_q(odd) if args.qf else odd  # PoseNet 에 들어가는 홀수 프레임 (prev 는 inflate 처럼 float 렌더)
    evq = fine_q_ste if args.qf else (lambda x: x)
    blob = open(args.pose2, "rb").read()
    pos = archive.unpack_pose2(blob)
    B = pos["B"].clone().to(dev)
    if args.b_res:
        assert args.train_b > 0, "--b-res 는 기저를 다시 학습해야 한다 (--train-b)"
        bh, bw = (int(v) for v in args.b_res.split(","))
        B = F.interpolate(B, size=(bh, bw), mode="area" if bh <= B.shape[2] else "bicubic")
        print(f"기저 해상도 {tuple(pos['B'].shape[2:])} → {(bh, bw)}", flush=True)
    opt_B = None
    if args.train_b > 0:
        B.requires_grad_(True)
        opt_B = torch.optim.Adam([B], lr=args.train_b)
    batches = [torch.arange(i, min(i + args.bs, n)) for i in range(0, n, args.bs)]
    A = [pos["a"][b].clone().to(dev).requires_grad_(True) for b in batches]
    C = [pos["c"][b].clone().to(dev).requires_grad_(True) for b in batches]
    batches = [b.to(dev) for b in batches]
    viz = LiveVis("comma vcc · pose 쌍별 다듬기", port=args.port).start()

    def evaluate(a, c):
        with torch.inference_mode():
            return torch.cat([pose_out(net, evq(even_frames_prev(prev[b], a[j], c[j], B.detach())), odd_in[b]) for j, b in enumerate(batches)]) - target

    state_path = args.out + ".state.pt"

    def save_state(phase, ep):
        torch.save({"phase": phase, "ep": ep, "A": [x.detach().cpu() for x in A], "C": [x.detach().cpu() for x in C], "B": B.detach().cpu(), "B_new": B_new}, state_path)

    def fq_b(Bt):
        """B 를 저장 격자(기저별 스케일, --bbits)로 가짜 양자화. 기울기는 그대로 통과."""
        qm = 2 ** (args.bbits - 1) - 1
        sc = Bt.detach().abs().amax((1, 2, 3), keepdim=True) / qm
        return Bt + ((Bt / sc).round().clamp(-qm, qm) * sc - Bt).detach()

    def run(epochs, lr, q=None, weighted=True, with_b=True, phase=None, start=0):
        opts = [torch.optim.Adam([A[j], C[j]], lr=lr) for j in range(len(batches))]
        for ep in range(start, epochs):
            cur_lr = lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * ep / max(epochs, 1))))
            for j, b in enumerate(batches):
                for g in opts[j].param_groups:
                    g["lr"] = cur_lr
                train_b = opt_B is not None and q is None and with_b
                a, c = (A[j], C[j]) if q is None else q(A[j], C[j])
                Bf = (fq_b(B) if args.b_qat else B) if train_b else B.detach()
                out = pose_out(net, evq(even_frames_prev(prev[b], a, c, Bf)), odd_in[b])
                sq = (out - target[b]) ** 2
                loss = (sq * wdim).sum() if (q is None and weighted) else sq.sum()
                opts[j].zero_grad()
                if train_b:
                    opt_B.zero_grad()
                    for g in opt_B.param_groups:
                        g["lr"] = args.train_b * cur_lr / lr
                loss.backward()
                opts[j].step()
                if train_b:
                    opt_B.step()
            if ep % 5 == 4 or ep == epochs - 1:
                aa = [x.detach() for x in A]
                cc = [x.detach() for x in C]
                if q is not None:
                    aa, cc = zip(*[q(x, y) for x, y in zip(aa, cc)])
                err = evaluate(aa, cc)
                d = err.pow(2).mean().item()
                viz.log(ep, posenet_dist=d, pose_term=float(np.sqrt(10 * d)))
                print(f"epoch {ep + 1}: posenet_dist {d:.7f} term {np.sqrt(10 * d):.4f} 차원별 RMS {err.pow(2).mean(0).sqrt().cpu().numpy().round(4)} ({time.time() - t0:.0f}s)", flush=True)
            if phase and (ep % 10 == 9 or ep == epochs - 1):
                save_state(phase, ep + 1)

    B_new = None
    phase, start = "main", 0
    if args.resume:
        st = torch.load(state_path, weights_only=False, map_location="cpu")
        phase, start = st["phase"], st["ep"]
        with torch.no_grad():
            for x, y in zip(A + C, st["A"] + st["C"]):
                x.copy_(y)
        B, B_new = st["B"].clone().to(dev).requires_grad_(B.requires_grad and phase == "main"), st["B_new"]
        if opt_B is not None and phase == "main":
            opt_B = torch.optim.Adam([B], lr=args.train_b)
        print(f"이어서: {phase} 에폭 {start} 부터", flush=True)
    if phase == "main":
        run(args.epochs, args.lr, phase="main", start=start)
        start = 0
    if opt_B is not None and phase == "main":  # 기저를 저장 비트로 고정
        with torch.no_grad():
            bq = 2 ** (args.bbits - 1) - 1
            B_scale_new = B.abs().amax((1, 2, 3)) / bq
            B_q_new = (B / B_scale_new.view(-1, 1, 1, 1)).round().clamp(-bq, bq)
            B = (B_q_new * B_scale_new.view(-1, 1, 1, 1)).detach()
            B_new = (B_q_new.cpu().numpy().astype(np.int8), B_scale_new.cpu().numpy().astype(np.float32))
    if args.plain_epochs:
        run(args.plain_epochs, args.lr, weighted=False, with_b=False, phase="plain", start=start)

    # 격자 양자화
    with torch.no_grad():
        a_all, c_all = torch.cat([x.detach() for x in A]), torch.cat([x.detach() for x in C])
        a_step = (a_all.amax(0) - a_all.amin(0)).clamp_min(1e-6) * 1.2 / 2**args.cbits
        c_step = (c_all.amax(0) - c_all.amin(0)).clamp_min(1e-6) * 1.2 / 2**args.cbits

    def q(a, c):
        return a + ((a / a_step).round() * a_step - a).detach(), c + ((c / c_step).round() * c_step - c).detach()

    run(args.q_epochs, args.lr * 0.1, q=q)

    with torch.no_grad():
        a_int = torch.cat([(x.detach() / a_step).round() for x in A])
        c_int = torch.cat([(x.detach() / c_step).round() for x in C])

        def dist(ai, ci):
            return evaluate([ai[b] * a_step for b in batches], [ci[b] * c_step for b in batches]).pow(2).mean(1)

        d = dist(a_int, c_int)
        print(f"양자화 후 posenet_dist {d.mean().item():.7f}", flush=True)
        for rnd in range(args.greedy_rounds):
            for which in ("a", "c"):
                vec = a_int if which == "a" else c_int
                for kd in range(vec.shape[1]):
                    for sgn in (1, -1):
                        cand = vec.clone()
                        cand[:, kd] += sgn
                        dn = dist(cand, c_int) if which == "a" else dist(a_int, cand)
                        better = dn < d
                        vec[better] = cand[better]
                        d[better] = dn[better]
            print(f"탐욕 탐색 {rnd + 1}: posenet_dist {d.mean().item():.7f} term {np.sqrt(10 * d.mean().item()):.4f} ({time.time() - t0:.0f}s)", flush=True)

    # B 는 원래 blob 의 정수 그대로 다시 쓴다
    import struct

    nn_, k, Cc, bh, bw = struct.unpack_from("<HHBHH", blob, 0)
    body = archive.unxz(blob[struct.calcsize("<HHBHH"):])
    B_scale = np.frombuffer(body, np.float32, k, 4 * (k + 6))
    B_q = np.frombuffer(body, np.int8, k * Cc * bh * bw, 4 * (2 * k + 6)).reshape(k, Cc, bh, bw)
    if B_new is not None:
        B_q, B_scale = B_new
    out = archive.pack_pose2(B_q, B_scale, c_int.cpu().numpy().astype(np.int64), c_step.cpu().numpy(), a_int.cpu().numpy().astype(np.int64), a_step.cpu().numpy())
    open(args.out, "wb").write(out)
    pos2 = archive.unpack_pose2(out)
    assert torch.allclose(pos2["B"], B.cpu())
    pos2 = {k: v.to(dev) for k, v in pos2.items()}
    dd = []
    with torch.inference_mode():
        for i in range(0, n, 20):
            e = even_frames_prev(prev[i : i + 20], pos2["a"][i : i + 20], pos2["c"][i : i + 20], pos2["B"])
            pair = torch.stack([e, odd[i : i + 20]], 1).flatten(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            back = downsample(torch.from_numpy(expand_fine(pair))).view(-1, 2, 3, SH, SW).to(dev)
            dd.append(((pose_out(net, back[:, 0], back[:, 1]) - target[i : i + 20]) ** 2).mean(1))
    d2 = torch.cat(dd).mean().item()
    print(f"pose2 {len(out):,} B, 평가 경로 posenet_dist {d2:.7f} term {np.sqrt(10 * d2):.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
