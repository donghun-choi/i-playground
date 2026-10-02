"""seg 맵 움직임 보상 실험: 노면 평면 전진 운동 모델로 이전 맵을 워핑해 현재 맵을 얼마나 맞히나.

노면 위 점 (u,v) 는 소실점 (u0,v0) 기준으로 (u-u0, v-v0) · (1 + k·(v-v0)) 로 이동 (k ∝ 속도).
현재 픽셀 (u,v) 의 값은 이전 맵에서 역방향 위치를 샘플: 이전 위치 ≈ 소실점 + (현재 - 소실점) / (1 + k·(v-v0)).
보닛(class 4) 은 차와 같이 움직이므로 워핑하지 않는다.
"""

import numpy as np
import torch
import torch.nn.functional as F

from common import SH, SW, load_gt

torch.set_num_threads(1)
_, seg, _ = load_gt()


def warp(prev: torch.Tensor, k: float, u0: float, v0: float) -> torch.Tensor:
    """prev (n,H,W) long → 워핑한 맵 (nearest)."""
    n = prev.shape[0]
    v, u = torch.meshgrid(torch.arange(SH, dtype=torch.float32), torch.arange(SW, dtype=torch.float32), indexing="ij")
    dv = (v - v0).clamp_min(0)  # 수평선 위(하늘)는 움직이지 않는다고 둔다
    s = 1 + k * dv
    pu = u0 + (u - u0) / s
    pv = v0 + (v - v0) / s
    grid = torch.stack([pu / (SW - 1) * 2 - 1, pv / (SH - 1) * 2 - 1], -1)[None].expand(n, SH, SW, 2)
    w = F.grid_sample(prev[:, None].float(), grid, mode="nearest", padding_mode="border", align_corners=True)[:, 0].long()
    return torch.where(prev == 4, prev, w)  # 보닛 자리는 그대로 (대충)


frames = torch.from_numpy(seg[100:400:10].astype(np.int64))
prevs = torch.from_numpy(seg[99:399:10].astype(np.int64))
changed = frames != prevs
print(f"이전 맵과 다른 픽셀 {changed.float().mean():.4f} ({changed.sum(0).float().mean() * 0:.0f})")
best = None
for v0 in (150, 165, 180, 195):
    for u0 in (236, 256, 276):
        for k in (0.001, 0.002, 0.004, 0.006, 0.008):
            w = warp(prevs, k, u0, v0)
            agree_changed = ((w == frames) & changed).sum().item() / changed.sum().item()
            wrong_new = ((w != frames) & ~changed).sum().item() / changed.sum().item()
            score = agree_changed - wrong_new
            if best is None or score > best[0]:
                best = (score, v0, u0, k, agree_changed, wrong_new)
print(f"최적 (전역 k): v0={best[1]} u0={best[2]} k={best[3]}: 바뀐 픽셀 중 워핑이 맞힌 비율 {best[4]:.3f}, 새로 틀리게 된 픽셀 비율 {best[5]:.3f}")
# 프레임마다 k 를 고르면?
_, v0, u0 = best[:3]
tot_fix = tot_new = 0
for i in range(len(frames)):
    cand = []
    for k in np.linspace(0, 0.012, 13):
        w = warp(prevs[i : i + 1], float(k), u0, v0)[0]
        fix = ((w == frames[i]) & changed[i]).sum().item()
        new = ((w != frames[i]) & ~changed[i]).sum().item()
        cand.append((fix - new, fix, new, k))
    c = max(cand)
    tot_fix += c[1]
    tot_new += c[2]
print(f"프레임별 k: 바뀐 픽셀 중 맞힌 비율 {tot_fix / changed.sum().item():.3f}, 새로 틀린 {tot_new / changed.sum().item():.3f}")
