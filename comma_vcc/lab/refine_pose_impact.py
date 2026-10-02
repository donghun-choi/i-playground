"""inflate 시점 SegNet 보정이 pose 에 주는 영향: 보정 전/후 홀수 프레임으로 seg/pose 왜곡 비교 (pose 파라미터는 그대로)."""

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, load_gt, nets, pose_out
import archive  # noqa: E402
from model import even_frames_prev  # noqa: E402
from pose_fit import renders

torch.set_num_threads(4)
net = nets()
net.segnet.to(memory_format=torch.channels_last)
_, seg, pose = load_gt()
pre = np.load(CACHE / "seg_pre.npy")
odd, prev = renders(str(CACHE / "renderer_v1.pt"), None, seg, pre, 6)
pos = archive.unpack_pose2(open(CACHE / "pose2_v1r.bin", "rb").read())
idx = np.arange(100, 132)  # 연속 32쌍 (이전 프레임도 보정된 것을 써야 하므로 연속 구간)


def refine(x, m, steps=2, lr=8.0, decay=0.5, margin=1.0):
    for s in range(steps):
        x = x.clone().requires_grad_(True)
        logits = net.segnet(x)
        true = logits.gather(1, m[:, None])[:, 0]
        other = logits.scatter(1, m[:, None], -1e4).amax(1)
        (g,) = torch.autograd.grad(F.relu(margin - (true - other)).sum(), x)
        x = (x - lr * decay**s * g / g.abs().amax((1, 2, 3), keepdim=True).clamp_min(1e-12)).clamp(0, 255).detach()
    return x


lo, hi = idx[0] - 1, idx[-1] + 1
m = torch.from_numpy(seg[lo:hi]).long()
o = odd[lo:hi]
ro = torch.cat([refine(o[i : i + 4], m[i : i + 4]) for i in range(0, len(o), 4)])
t = torch.from_numpy(pose[idx])
with torch.inference_mode():
    for name, oo in (("보정 전", o), ("보정 2스텝", ro)):
        seg_err = (net.segnet(oo[1:]).argmax(1) != m[1:]).float().mean().item()
        e = even_frames_prev(oo[:-1], pos["a"][idx], pos["c"][idx], pos["B"])
        pd = ((pose_out(net, e, oo[1:]) - t) ** 2).mean().item()
        print(f"{name}: seg {seg_err:.6f}, pose {pd:.7f}  (|변화| 평균 {(oo - o).abs().mean():.3f} 최대 {(oo - o).abs().max():.1f})")
