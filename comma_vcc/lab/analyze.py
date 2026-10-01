"""문제 구조 분석. 숫자는 stdout, 그림은 livevis 로.

1) 512x384 이미지 → 원해상도 프레임 복원이 정확한가 (bilinear 축소가 읽는 픽셀만 채우면 되는가)
2) GT SegNet 맵: 클래스 분포, 경계 픽셀 수, 시간 변화, 단순 압축 크기
3) GT pose: 차원별 분산, 상수 예측 시 점수
4) 네트워크 CPU 속도 (forward / backward)
"""

import bz2
import io
import lzma
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import CACHE, H, SH, SW, W, downsample, load_gt, nets, pose_out, score
from livevis import LiveVis

torch.set_num_threads(4)
viz = LiveVis("comma vcc · 분석").start()
small, seg, pose = load_gt()
margin = np.load(CACHE / "gt_margin.npy")
net = nets()


def png(arr):
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


# ---------- 1) 축소가 실제로 읽는 원해상도 픽셀 ----------
def src_index(n_in, n_out):
    scale = n_in / n_out
    s = (np.arange(n_out) + 0.5) * scale - 0.5
    s = np.clip(s, 0, None)
    i0 = np.floor(s).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    return i0, i1


r0, r1 = src_index(H, SH)
c0, c1 = src_index(W, SW)
used_rows = np.union1d(r0, r1)
used_cols = np.union1d(c0, c1)
print(f"[1] 축소가 읽는 행 {len(used_rows)}/{H}, 열 {len(used_cols)}/{W}, 겹치는 행 {len(r0) * 2 - len(used_rows)}")

# 512x384 정수 이미지를 원해상도로 펼친 뒤 다시 축소하면 그대로 돌아오는가
rng = np.random.default_rng(0)
img = rng.integers(0, 256, (SH, SW, 3)).astype(np.uint8)
row_of = np.zeros(H, int)
col_of = np.zeros(W, int)
row_of[r0], row_of[r1] = np.arange(SH), np.arange(SH)
col_of[c0], col_of[c1] = np.arange(SW), np.arange(SW)
# 안 읽히는 행/열은 가까운 것으로 (아무 값이어도 됨)
for arr, used in ((row_of, used_rows), (col_of, used_cols)):
    for i in range(len(arr)):
        if i not in set(used):
            arr[i] = arr[used[np.argmin(np.abs(used - i))]]
full = img[row_of][:, col_of]
back = downsample(torch.from_numpy(full)[None])[0].permute(1, 2, 0).numpy()
print(f"[1] 펼침→축소 최대 오차: {np.abs(back - img).max():.6f}")

# ---------- 2) SegNet 맵 ----------
counts = np.bincount(seg.ravel(), minlength=5) / seg.size
print("[2] 클래스 비율:", " ".join(f"{c}:{p:.3f}" for c, p in enumerate(counts)))
edge = (seg[:, 1:, :] != seg[:, :-1, :]).sum((1, 2)) + (seg[:, :, 1:] != seg[:, :, :-1]).sum((1, 2))
print(f"[2] 프레임당 경계 길이(4-이웃 불일치 수): 평균 {edge.mean():.0f}, 최소 {edge.min()}, 최대 {edge.max()}")
tchange = (seg[1:] != seg[:-1]).mean((1, 2))
print(f"[2] 이전 맵 대비 바뀐 픽셀 비율: 평균 {tchange.mean():.4f}, 최대 {tchange.max():.4f}")
low = (margin.astype(np.float32) < 1.0).mean()
print(f"[2] SegNet top1-top2 logit 차 < 1 인 픽셀 비율: {low:.4f}")

raw = seg.tobytes()
t = time.time()
sizes = {
    "lzma(raw)": len(lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)),
    "bz2(raw)": len(bz2.compress(raw, 9)),
}
xor = np.concatenate([seg[:1], (seg[1:] != seg[:-1]) * (seg[1:] + 1)]).astype(np.uint8)
sizes["lzma(시간차)"] = len(lzma.compress(xor.tobytes(), preset=9 | lzma.PRESET_EXTREME))
sizes["png 합"] = sum(len(png(s)) for s in seg)
for k, v in sizes.items():
    print(f"[2] {k}: {v:,} bytes → rate 항 {score(0, 0, v)['rate_term']:.3f}")
print(f"    ({time.time() - t:.0f}s)")

palette = np.array([[64, 64, 64], [230, 230, 230], [42, 120, 214], [12, 163, 12], [208, 59, 59]], np.uint8)
for k in (0, 300, 599):
    m = np.clip(margin[k].astype(np.float32) / 8, 0, 1)
    viz.image(f"seg map #{k}", png(np.concatenate([palette[seg[k]], (np.stack([m] * 3, -1) * 255).astype(np.uint8)], 1)),
              caption="왼쪽: GT SegNet argmax / 오른쪽: top1-top2 logit 차 (어두울수록 경계에 민감)")

# ---------- 3) pose ----------
var = pose.var(0)
print("[3] pose 차원별 표준편차:", np.round(np.sqrt(var), 4))
print(f"[3] 평균 pose 만 내보낼 때 posenet_dist {var.mean():.5f} → pose 항 {score(0, var.mean(), 0)['pose_term']:.3f}")
for step, p in enumerate(pose):
    viz.log(step, **{f"pose_{d}": float(p[d]) for d in range(6)}, seg_temporal_change=float(tchange[step - 1]) if step else 0.0,
            seg_edge=float(edge[step]))

# ---------- 4) 속도 ----------
x = torch.from_numpy(np.array(small[1:9:2]))  # 4장
x.requires_grad_(True)
for name, fn in (("segnet", lambda z: net.segnet(z).sum()),):
    t = time.time(); fn(x.detach()); tf = time.time() - t
    t = time.time(); fn(x).backward(); tb = time.time() - t
    print(f"[4] {name} 4장: forward {tf:.2f}s, forward+backward {tb:.2f}s")
e = torch.from_numpy(np.array(small[0:32:2])).requires_grad_(True)
o = torch.from_numpy(np.array(small[1:32:2]))
t = time.time(); pose_out(net, e.detach(), o); tf = time.time() - t
t = time.time(); pose_out(net, e, o).sum().backward(); tb = time.time() - t
print(f"[4] posenet 16쌍: forward {tf:.2f}s, forward+backward {tb:.2f}s")
print("분석 끝. 페이지는 Ctrl+C 전까지 유지")
viz.wait()
