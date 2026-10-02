"""렌더러 오류 분석: 틀린 픽셀이 경계에서 얼마나 떨어져 있나, 어떤 클래스 쌍인가, 프레임별 분포.

    python seg_errors.py --renderer ../cache/renderer_v1.pt
"""

import argparse
import io

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import CACHE, load_gt, nets
from model import Renderer, quantize_roundtrip  # noqa: E402

NAMES = ["road", "lane", "bg", "car", "hood"]
PALETTE = np.array([[64, 64, 64], [230, 230, 230], [42, 120, 214], [12, 163, 12], [208, 59, 59]], np.uint8)


def boundary_distance(m: torch.Tensor, maxd: int = 4) -> torch.Tensor:
    """각 픽셀에서 가장 가까운 '다른 클래스' 픽셀까지의 체비셰프 거리 (maxd 이상은 maxd)."""
    oh = F.one_hot(m.long(), 5).permute(0, 3, 1, 2).float()
    dist = torch.full(m.shape, float(maxd))
    for d in range(maxd, 0, -1):
        k = 2 * d + 1
        present = F.max_pool2d(oh, k, stride=1, padding=d)  # 반경 d 안에 그 클래스가 있나
        other = (present.sum(1) > 1.5)  # 자기 말고 다른 클래스도 있다
        dist[other] = d
    return dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renderer", default=str(CACHE / "renderer_v1.pt"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--port", type=int, default=8009)
    args = ap.parse_args()
    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    net = nets()
    _, seg, _ = load_gt()
    G = Renderer()
    G.load_state_dict(torch.load(args.renderer))
    quantize_roundtrip(G)
    G.eval()
    viz = LiveVis("comma vcc · 렌더러 오류 분석", port=args.port).start()
    conf = np.zeros((5, 5), np.int64)
    by_dist = np.zeros(5, np.int64)
    px_by_dist = np.zeros(5, np.int64)
    per_frame = []
    worst = (0, None)
    with torch.inference_mode():
        for i in range(0, 600, 8):
            m = torch.from_numpy(seg[i : i + 8])
            img = G(m)
            out = net.segnet(img).argmax(1)
            wrong = out != m
            dist = boundary_distance(m)
            for d in range(5):
                by_dist[d] += (wrong & (dist == d)).sum().item()
                px_by_dist[d] += (dist == d).sum().item()
            np.add.at(conf, (m[wrong].numpy(), out[wrong].numpy()), 1)
            for j in range(len(m)):
                e = wrong[j].float().mean().item()
                per_frame.append(e)
                viz.log(i + j, frame_disagreement=e)
                if e > worst[0]:
                    worst = (e, (i + j, img[j], out[j], m[j]))
    per_frame = np.array(per_frame)
    total = conf.sum()
    print(f"전체 disagreement {per_frame.mean():.6f} (seg 항 {100 * per_frame.mean():.4f}), 프레임당 틀린 픽셀 {total / 600:.0f}")
    print("경계 거리별 (0=?, 1=경계 바로 옆 ...):")
    for d in range(1, 5):
        print(f"  거리 {d}{'+' if d == 4 else ''}: 틀린 {by_dist[d] / total:6.1%}  (그 거리 픽셀 중 틀린 비율 {by_dist[d] / max(px_by_dist[d], 1):.4f})")
    print("클래스 쌍 (정답 → SegNet 출력) 상위:")
    pairs = sorted(((conf[a, b], a, b) for a in range(5) for b in range(5) if a != b), reverse=True)[:8]
    for cnt, a, b in pairs:
        print(f"  {NAMES[a]:>5} → {NAMES[b]:<5} {cnt / total:6.1%}")
    q = np.quantile(per_frame, [0.5, 0.9, 0.99])
    print(f"프레임별 disagreement 중앙값 {q[0]:.5f}, 90% {q[1]:.5f}, 99% {q[2]:.5f}, 최대 {per_frame.max():.5f}")
    e, (k, img, out, m) = worst
    vis = np.concatenate([img.round().clamp(0, 255).permute(1, 2, 0).byte().numpy(), PALETTE[out.numpy()]], 1)
    vis[:, 512:][(out != m).numpy()] = [255, 0, 255]
    b = io.BytesIO()
    Image.fromarray(vis).save(b, format="PNG")
    viz.image("가장 나쁜 프레임", b.getvalue(), caption=f"#{k} disagreement {e:.5f} (분홍 = 불일치)")
    np.save(CACHE / "renderer_v1_frame_err.npy", per_frame)


if __name__ == "__main__":
    main()
