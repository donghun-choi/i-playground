"""실험 공용 도구: 경로, 평가 네트워크, GT 캐시.

평가 네트워크는 1164x874 프레임을 받자마자 512x384 로 bilinear 축소한다.
그래서 실험은 대부분 '네트워크가 보는 512x384 float 이미지' 공간에서 한다.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
VCC = LAB.parent
CHALLENGE = VCC / "challenge"
CACHE = VCC / "cache"
SUB = VCC / "submissions" / "semantic_cpu"
sys.path[:0] = [str(CHALLENGE), str(VCC.parent), str(SUB)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from frame_utils import AVVideoDataset, camera_size, rgb_to_yuv6, segnet_model_input_size  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402

W, H = camera_size  # 1164, 874
SW, SH = segnet_model_input_size  # 512, 384
N_FRAMES = 1200
N_PAIRS = 600
ORIGINAL_BYTES = 37_545_489


def score(seg: float, pose: float, nbytes: int) -> dict:
    t = {"seg_term": 100 * seg, "pose_term": math.sqrt(10 * max(pose, 0.0)), "rate_term": 25 * nbytes / ORIGINAL_BYTES}
    return {"score": sum(t.values()), **t}


_NET = None


def nets(device="cpu") -> DistortionNet:
    global _NET
    if _NET is None:
        net = DistortionNet().eval()
        net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
        for p in net.parameters():
            p.requires_grad_(False)
        _NET = net
    return _NET.to(device)


def setup_device(name: str) -> torch.device:
    """--device 인자 → torch.device. GPU 에서는 TF32 를 끈다 (flip 손실은 margin 근처 정밀도에 민감)."""
    dev = torch.device(name)
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
    return dev


def downsample(frames_u8: torch.Tensor) -> torch.Tensor:
    """(N,H,W,3) uint8 원해상도 → (N,3,384,512) float. 평가 코드와 같은 연산."""
    x = frames_u8.permute(0, 3, 1, 2).float()
    return F.interpolate(x, size=(SH, SW), mode="bilinear")


def fine_q(x: torch.Tensor) -> torch.Tensor:
    """(N,3,384,512) float → expand_fine 으로 원해상도 uint8 을 만든 뒤 평가 코드가 다시 축소한 값 (원해상도를 만들지 않고 바로).

    fine_blocks 와 같은 선택: 기준값 + 위치별 256 조합 가중합 중 가장 가까운 것. 평가 경로와 1e-4 이내로 같다.
    """
    from model import _table

    ss, _ = _table()
    ss = ss.to(x.device)
    n = x.shape[0]
    xf = x.detach().permute(2, 3, 0, 1).reshape(SH * SW, n * 3)
    base = xf.floor().clamp(1, 253)
    r = (xf - base).contiguous()
    j = torch.searchsorted(ss, r).clamp(1, 255)
    lo, hi = ss.gather(1, j - 1), ss.gather(1, j)
    v = torch.where((r - lo).abs() <= (hi - r).abs(), lo, hi)
    return (base + v).reshape(SH, SW, n, 3).permute(2, 3, 0, 1).contiguous()


def fine_q_ste(x: torch.Tensor) -> torch.Tensor:
    """앞으로는 fine_q, 뒤로는 그대로 (STE)."""
    return x + (fine_q(x) - x).detach()


def yuv6(rgb: torch.Tensor) -> torch.Tensor:
    """frame_utils.rgb_to_yuv6 와 같은 수식인데 미분 가능 (원본은 no_grad + in-place)."""
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    Y = (R * 0.299 + G * 0.587 + B * 0.114).clamp(0.0, 255.0)
    U = ((B - Y) / 1.772 + 128.0).clamp(0.0, 255.0)
    V = ((R - Y) / 1.402 + 128.0).clamp(0.0, 255.0)

    def sub(c):
        return (c[:, 0::2, 0::2] + c[:, 1::2, 0::2] + c[:, 0::2, 1::2] + c[:, 1::2, 1::2]) * 0.25

    return torch.stack([Y[:, 0::2, 0::2], Y[:, 1::2, 0::2], Y[:, 0::2, 1::2], Y[:, 1::2, 1::2], sub(U), sub(V)], dim=1)


def posenet_in(even: torch.Tensor, odd: torch.Tensor) -> torch.Tensor:
    """512x384 float RGB 두 장 (B,3,384,512) → PoseNet 입력 (B,12,192,256)."""
    return torch.cat([yuv6(even), yuv6(odd)], dim=1)


def pose_out(net, even, odd) -> torch.Tensor:
    return net.posenet(posenet_in(even, odd))["pose"][:, :6]


def seg_logits(net, odd) -> torch.Tensor:
    return net.segnet(odd)


def gt_frames():
    """원본 영상을 평가 코드와 똑같이 디코드해서 (2,H,W,3) uint8 쌍을 하나씩 낸다."""
    ds = AVVideoDataset(["0.mkv"], data_dir=CHALLENGE / "videos", batch_size=1, device=torch.device("cpu"))
    for _, _, batch in ds:
        yield batch[0]


def load_gt():
    """캐시된 GT: small (1200,3,384,512) float32 memmap, seg (600,384,512) uint8, pose (600,6) float32."""
    small = np.load(CACHE / "gt_small.npy", mmap_mode="r") if (CACHE / "gt_small.npy").exists() else None  # Colab 경량 캐시에는 없다
    seg = np.load(CACHE / "gt_seg.npy")
    pose = np.load(CACHE / "gt_pose.npy")
    return small, seg, pose


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        print(f"[{self.name}] {time.time() - self.t:.2f}s", flush=True)
