"""seg 맵 계층 부호화용 학습 문맥 모델 (float 학습 → 비트 수 추정).

패스 (stride s, h=s/2):
  A: stride-h 격자에서 (홀,홀) 칸. 아는 것: (짝,짝) 칸 = stride-s 격자 전체.
  B: 한쪽만 홀수인 칸. 아는 것: (짝,짝) + (홀,홀).
입력 (stride-h 격자 해상도):
  현재 맵 one-hot × known(5) + known(1) + 이전 맵 클래스 비율(5) + 전전 맵 비율(5) + is_A(1) + 레벨 one-hot(5) = 22
출력: 각 칸 5클래스 logits. 손실은 target 칸에서만.
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CACHE, SH, SW, load_gt
import segcodec as sc  # noqa: E402

LEVELS = sc.LEVELS
K = sc.K
C_IN = sc.C_IN


def build_input(cur, prev, prev2, s: int, kind: str):
    """디코더와 같은 정수 입력 (segcodec.build_input_q) 을 float (0..1) 로. cur/prev/prev2: (n,384,512) uint8 numpy."""
    h = s // 2
    x = torch.from_numpy(sc.build_input_q(cur, prev, prev2, s, kind)).float() / sc.Q_IN
    _, target = sc.masks(h, kind)
    return x, torch.from_numpy(target), torch.from_numpy(cur[:, ::h, ::h].astype(np.int64))


class CtxNet(nn.Module):
    def __init__(self, ch: int = 16, layers: int = 4, dils=None):
        super().__init__()
        dils = list(dils) if dils else [1] * layers
        assert len(dils) == layers
        self.dils = dils
        mods = [nn.Conv2d(C_IN, ch, 3, padding=dils[0], dilation=dils[0]), nn.ReLU()]
        for d in dils[1:]:
            mods += [nn.Conv2d(ch, ch, 3, padding=d, dilation=d), nn.ReLU()]
        mods += [nn.Conv2d(ch, K, 1)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def load_ctx(path):
    """체크포인트 → (CtxNet, state_dict, dils). 예전 형식(state_dict 만)도 읽는다."""
    ck = torch.load(path)
    sd, dils = (ck["sd"], ck["dils"]) if "sd" in ck else (ck, None)
    ch = sd["net.0.weight"].shape[0]
    layers = sum(1 for k in sd if k.endswith(".weight")) - 1
    m = CtxNet(ch, layers, dils)
    m.load_state_dict(sd)
    return m, sd, dils


def frame_maps(seg, t):
    return seg[t : t + 1], (seg[t - 1 : t] if t >= 1 else None), (seg[t - 2 : t - 1] if t >= 2 else None)


@torch.inference_mode()
def eval_bits(model, seg, frames):
    """전체 프레임에서 패스별 비트 수 (coarse 격자 제외)."""
    tot = {}
    for t in frames:
        cur, prev, prev2 = frame_maps(seg, t)
        for s in LEVELS:
            for kind in "AB":
                x, target, g = build_input(cur, prev, prev2, s, kind)
                logp = F.log_softmax(model(x), 1)
                nll = -logp.gather(1, g[:, None])[:, 0][:, target].sum().item() / math.log(2)
                tot[f"{kind}{s}"] = tot.get(f"{kind}{s}", 0.0) + nll
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--crop", type=int, default=96)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default=str(CACHE / "ctxnet.pt"))
    ap.add_argument("--init", default=None)
    ap.add_argument("--dils", default=None, help="층별 dilation, 예: 1,2,4,2,1")
    args = ap.parse_args()

    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    _, seg, _ = load_gt()
    dils = [int(d) for d in args.dils.split(",")] if args.dils else None
    model = CtxNet(args.ch, args.layers, dils)
    if args.init:
        ck = torch.load(args.init)
        model.load_state_dict(ck["sd"] if "sd" in ck else ck)
    print(f"ctxnet params {sum(p.numel() for p in model.parameters()):,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    viz = LiveVis(f"comma vcc · seg 문맥 모델 ch{args.ch} L{args.layers}", port=args.port).start()
    eval_frames = list(range(3, 600, 60))  # 10장
    # 레벨별 샘플링 비중: 세밀한 레벨에 비트가 몰려 있다
    level_p = np.array([0.04, 0.06, 0.15, 0.3, 0.45])
    t0 = time.time()
    for step in range(1, args.steps + 1):
        s = LEVELS[rng.choice(len(LEVELS), p=level_p)]
        h = s // 2
        gh, gw = SH // h, SW // h
        loss_sum, n_sum = 0.0, 0
        ts = rng.integers(0, 600, 4)
        xs, tg, gs = [], [], []
        for t in ts:
            cur, prev, prev2 = frame_maps(seg, int(t))
            kind = "A" if rng.random() < 0.5 else "B"
            x, target, g = build_input(cur, prev, prev2, s, kind)
            c = min(args.crop, gh, gw)
            # 크롭 시작은 짝수 칸 (A/B 패턴 유지)
            i0 = int(rng.integers(0, (gh - c) // 2 + 1)) * 2
            j0 = int(rng.integers(0, (gw - c) // 2 + 1)) * 2
            xs.append(x[:, :, i0 : i0 + c, j0 : j0 + c])
            tg.append(target[i0 : i0 + c, j0 : j0 + c].expand(1, c, c))
            gs.append(g[:, i0 : i0 + c, j0 : j0 + c])
        x = torch.cat(xs)
        tmask = torch.cat(tg)
        g = torch.cat(gs)
        logits = model(x)
        nll = F.cross_entropy(logits, g, reduction="none")[tmask]
        loss = nll.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        viz.log(step, train_bits_per_target=loss.item() / math.log(2))
        if step % args.eval_every == 0 or step == args.steps:
            tot = eval_bits(model, seg, eval_frames)
            bpf = sum(tot.values()) / 8 / len(eval_frames)
            viz.log(step, eval_bytes_per_frame=bpf, **{f"B/frame {k}": v / 8 / len(eval_frames) for k, v in tot.items()})
            print(f"step {step}: {bpf:.0f} B/frame (coarse 제외) ({time.time() - t0:.0f}s) "
                  + " ".join(f"{k}:{v / 8 / len(eval_frames):.0f}" for k, v in tot.items()), flush=True)
            torch.save({"sd": model.state_dict(), "dils": model.dils}, args.out)


if __name__ == "__main__":
    main()
