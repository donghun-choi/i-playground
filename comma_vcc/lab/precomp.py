"""경계 사전보정: 렌더러 입력 맵 M' 를 조정해서 SegNet(render(M')) 이 원래 맵 M 과 더 일치하게.

전송하는 것은 M' (디코더는 M 이 필요 없다). 오류의 99% 가 경계 1픽셀이라 경계를 국소적으로 밀어 본다.
라운드마다:
  1) out = argmax SegNet(render(M'))   (fp32, 평가와 같은 수치)
  2) 틀린 픽셀 p (정답 c) 주변 3x3 에서 M' 를 c 로 넓히는 후보 변경
  3) 다시 평가해서, 변경한 픽셀마다 주변 5x5 오류가 줄었으면 채택, 아니면 되돌림

    python precomp.py --renderer ../cache/renderer_v1.pt --rounds 6 --out ../cache/seg_precomp.npy
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, load_gt, nets
from model import Renderer, quantize_roundtrip  # noqa: E402


def box(x: torch.Tensor, k: int) -> torch.Tensor:
    """(n,H,W) bool/float → k×k 창 안의 합 (n,H,W)."""
    return F.avg_pool2d(x.float()[:, None], k, stride=1, padding=k // 2, count_include_pad=False)[:, 0] * k * k


@torch.inference_mode()
def segout(G, net, m: torch.Tensor) -> torch.Tensor:
    return net.segnet(G(m)).argmax(1)


@torch.inference_mode()
def refine(G, net, M: torch.Tensor, rounds: int):
    """M: (n,H,W) 원래 맵 (uint8). → M' 와 라운드별 오류 수."""
    Mp = M.clone()
    out = segout(G, net, Mp)
    hist = [(out != M).sum().item()]
    for _ in range(rounds):
        err = out != M
        cand = Mp.clone()
        claimed = torch.zeros_like(err)
        for c in range(5):
            grow = (box(err & (M == c), 3) > 0) & (Mp != c) & ~claimed
            cand[grow] = c
            claimed |= grow
        changed = cand != Mp
        if not changed.any():
            break
        out2 = segout(G, net, cand)
        err2 = out2 != M
        better = box(err2, 5) < box(err, 5)
        keep = changed & better
        Mp[keep] = cand[keep]
        out = segout(G, net, Mp)
        hist.append((out != M).sum().item())
    return Mp, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default=str(CACHE / "renderer_v1.pt"))
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default=str(CACHE / "seg_precomp.npy"))
    ap.add_argument("--port", type=int, default=8010)
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    net = nets()
    _, seg, _ = load_gt()
    G = Renderer()
    G.load_state_dict(torch.load(args.renderer))
    quantize_roundtrip(G)
    G.eval()
    viz = LiveVis("comma vcc · 경계 사전보정", port=args.port).start()
    n = args.frames
    out = seg[:n].copy()
    before = after = 0
    changed = 0
    t0 = time.time()
    for i in range(0, n, args.bs):
        M = torch.from_numpy(seg[i : i + args.bs])
        Mp, hist = refine(G, net, M, args.rounds)
        out[i : i + args.bs] = Mp.numpy()
        before += hist[0]
        after += hist[-1]
        changed += (Mp != M).sum().item()
        f = i + len(M)
        viz.log(f, disagreement_before=before / (f * 196608), disagreement_after=after / (f * 196608),
                changed_px_per_frame=changed / f)
        if (i // args.bs) % 10 == 0:
            print(f"{f}/{n}: 틀린 픽셀 {before / f:.0f} → {after / f:.0f} /frame, 바꾼 픽셀 {changed / f:.0f}/frame, 라운드별 {hist} ({time.time() - t0:.0f}s)", flush=True)
    np.save(args.out, out)
    print(f"완료: disagreement {before / (n * 196608):.6f} → {after / (n * 196608):.6f}, 바꾼 픽셀 {changed / n:.0f}/frame ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
