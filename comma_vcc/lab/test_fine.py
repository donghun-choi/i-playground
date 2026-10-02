"""서브픽셀 정밀 확장 검증: GT float 이미지를 정수 반올림 vs 정밀 확장으로 만들었을 때 축소 오차와 네트워크 왜곡."""
import time

import numpy as np
import torch

from common import downsample, load_gt, nets, pose_out
from model import expand, expand_fine  # noqa: E402

torch.set_num_threads(1)
net = nets()
small, seg, pose = load_gt()
idx = np.arange(1, 600, 20)
x = np.stack([np.stack([small[2 * i], small[2 * i + 1]]) for i in idx])  # (n,2,3,H,W)
flat = x.reshape(-1, *x.shape[2:]).transpose(0, 2, 3, 1)  # (2n,H,W,3)
for name, fn in (("정수 반올림", lambda a: expand(np.clip(np.round(a), 0, 255).astype(np.uint8))), ("서브픽셀 정밀", expand_fine)):
    t = time.time()
    full = fn(flat)
    dt = time.time() - t
    back = downsample(torch.from_numpy(full)).numpy()
    err = np.abs(back - flat.transpose(0, 3, 1, 2))
    b = torch.from_numpy(back).view(len(idx), 2, *back.shape[1:])
    with torch.inference_mode():
        p = pose_out(net, b[:, 0], b[:, 1])
        sg = net.segnet(b[:, 1]).argmax(1).numpy()
    pd = ((p - torch.from_numpy(pose[idx])) ** 2).mean().item()
    sd = (sg != seg[idx]).mean()
    print(f"{name}: 축소 오차 평균 {err.mean():.4f} 최대 {err.max():.3f} | posenet {pd:.2e} (term {np.sqrt(10 * pd):.4f}) | segnet {sd:.2e} | {dt:.1f}s")
