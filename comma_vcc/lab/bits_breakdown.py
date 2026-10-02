"""seg 맵 부호 길이 분해: 정답 클래스별, 이전 프레임과 같은지별, 레벨별 (float 문맥 모델 기준)."""
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, load_gt
from ctxmodel import CtxNet, build_input, frame_maps
import segcodec as sc  # noqa: E402

torch.set_num_threads(1)
ckpt = sys.argv[1] if len(sys.argv) > 1 else str(CACHE / "ctxnet_c24l5.pt")
sd = torch.load(ckpt)
model = CtxNet(sd["net.0.weight"].shape[0], sum(1 for k in sd if k.endswith(".weight")) - 1)
model.load_state_dict(sd)
_, seg, _ = load_gt()
NAMES = ["road", "lane", "bg", "car", "hood"]
by_cls = np.zeros(5)
by_same = np.zeros(2)
cnt_same = np.zeros(2)
by_lvl = {}
frames = range(3, 600, 30)
with torch.inference_mode():
    for t in frames:
        cur, prev, prev2 = frame_maps(seg, t)
        for s in sc.LEVELS:
            h = s // 2
            for kind in "AB":
                x, target, g = build_input(cur, prev, prev2, s, kind)
                nll = -F.log_softmax(model(x), 1).gather(1, g[:, None])[:, 0][0] / math.log(2)
                tv, nv = g[0][target].numpy(), nll[target].numpy()
                pv = prev[0, ::h, ::h][target.numpy()]
                for c in range(5):
                    by_cls[c] += nv[tv == c].sum()
                same = (pv == tv)
                by_same[1] += nv[same].sum(); by_same[0] += nv[~same].sum()
                cnt_same[1] += same.sum(); cnt_same[0] += (~same).sum()
                by_lvl[f"{kind}{s}"] = by_lvl.get(f"{kind}{s}", 0) + nv.sum()
n = len(frames)
tot = by_cls.sum()
print(f"평균 {tot / 8 / n:.0f} B/frame")
print("정답 클래스별:", ", ".join(f"{NAMES[c]} {by_cls[c] / tot:.1%}" for c in range(5)))
print(f"이전 프레임과 같은 칸: 비트 {by_same[1] / tot:.1%} (칸 {cnt_same[1]:.0f}, 칸당 {by_same[1] / cnt_same[1]:.4f} bit)")
print(f"이전 프레임과 다른 칸: 비트 {by_same[0] / tot:.1%} (칸 {cnt_same[0]:.0f}, 칸당 {by_same[0] / cnt_same[0]:.3f} bit)")
print("레벨별 B/frame:", " ".join(f"{k}:{v / 8 / n:.0f}" for k, v in by_lvl.items()))
