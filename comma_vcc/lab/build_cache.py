"""GT 를 한 번 디코드해서 네트워크 입력/출력을 캐시한다.

cache/gt_small.npy   (1200,3,384,512) float32  네트워크가 보는 원본 프레임
cache/gt_seg.npy     (600,384,512)    uint8    홀수 프레임 SegNet argmax
cache/gt_margin.npy  (600,384,512)    float16  SegNet top1-top2 logit 차 (경계 민감도)
cache/gt_pose.npy    (600,6)          float32  PoseNet pose 평균 6차원
cache/gt_pose12.npy  (600,12)         float32  PoseNet 전체 출력
"""

import numpy as np
import torch

from common import CACHE, N_FRAMES, N_PAIRS, SH, SW, Timer, downsample, gt_frames, nets

CACHE.mkdir(exist_ok=True)
torch.set_num_threads(4)
net = nets()
small = np.lib.format.open_memmap(CACHE / "gt_small.npy", mode="w+", dtype=np.float32, shape=(N_FRAMES, 3, SH, SW))
seg = np.zeros((N_PAIRS, SH, SW), np.uint8)
margin = np.zeros((N_PAIRS, SH, SW), np.float16)
pose12 = np.zeros((N_PAIRS, 12), np.float32)

B = 8
buf = []
with Timer("cache"), torch.inference_mode():
    for i, pair in enumerate(gt_frames()):
        x = downsample(pair)  # (2,3,384,512)
        small[2 * i : 2 * i + 2] = x.numpy()
        buf.append(x)
        if len(buf) == B or i == N_PAIRS - 1:
            xs = torch.stack(buf)  # (b,2,3,384,512)
            j0 = i + 1 - len(buf)
            pin = net.posenet.preprocess_input(xs)
            pose12[j0 : i + 1] = net.posenet(pin)["pose"].numpy()
            logits = net.segnet(xs[:, 1])
            top2 = logits.topk(2, dim=1).values
            seg[j0 : i + 1] = logits.argmax(1).numpy()
            margin[j0 : i + 1] = (top2[:, 0] - top2[:, 1]).numpy()
            buf = []
            if (i + 1) % 80 == 0:
                print(f"{i + 1}/{N_PAIRS}", flush=True)

small.flush()
np.save(CACHE / "gt_seg.npy", seg)
np.save(CACHE / "gt_margin.npy", margin)
np.save(CACHE / "gt_pose12.npy", pose12)
np.save(CACHE / "gt_pose.npy", pose12[:, :6])
print("done")
