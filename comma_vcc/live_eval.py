"""공식 evaluate.py 와 같은 계산을 하면서 진행 상황을 브라우저에 실시간으로 그린다.

    comma_vcc/.venv/bin/python comma_vcc/live_eval.py --submission-dir comma_vcc/challenge/submissions/baseline_fast

- 데이터로더, DistortionNet, 점수식은 챌린지 코드를 그대로 import 해서 쓴다 (CPU 경로).
- 배치마다: 배치 평균 PoseNet/SegNet 왜곡, 지금까지의 누적 점수와 항별 기여.
- 몇 배치마다: 원본 vs 복원 프레임, SegNet 클래스 맵과 불일치 픽셀.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE / "challenge"
sys.path[:0] = [str(CHALLENGE), str(HERE.parent)]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from frame_utils import AVVideoDataset, TensorVideoDataset, camera_size, seq_len  # noqa: E402
from livevis import LiveVis  # noqa: E402
from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # noqa: E402

# SegNet 5개 클래스 색 (구분만 되면 된다)
SEG_COLORS = np.array([[64, 64, 64], [230, 230, 230], [42, 120, 214], [12, 163, 12], [208, 59, 59]], dtype=np.uint8)


def score_terms(seg: float, pose: float, rate: float) -> dict:
    terms = {"segnet_term": 100 * seg, "posenet_term": math.sqrt(10 * pose), "rate_term": 25 * rate}
    return {"score": sum(terms.values()), **terms}


def comparison_png(gt: torch.Tensor, comp: torch.Tensor, seg_gt: torch.Tensor, seg_comp: torch.Tensor) -> bytes:
    """2x2: 원본 | 복원 / 원본 SegNet 맵 | 복원 위에 불일치 픽셀(빨강)."""
    w, h = camera_size[0] // 2, camera_size[1] // 2
    gt_img = Image.fromarray(gt.numpy()).resize((w, h), Image.BILINEAR)
    comp_img = Image.fromarray(comp.numpy()).resize((w, h), Image.BILINEAR)
    seg_img = Image.fromarray(SEG_COLORS[seg_gt.numpy()]).resize((w, h), Image.NEAREST)
    diff = (seg_gt != seg_comp).numpy()
    overlay = np.asarray(comp_img.resize(diff.shape[::-1], Image.BILINEAR)).copy()
    overlay[diff] = (0.3 * overlay[diff] + 0.7 * np.array([255, 0, 0])).astype(np.uint8)
    diff_img = Image.fromarray(overlay).resize((w, h), Image.NEAREST)

    canvas = Image.new("RGB", (2 * w, 2 * h))
    for i, (img, label) in enumerate(
        [(gt_img, "original"), (comp_img, "reconstructed"), (seg_img, "segnet(original)"), (diff_img, "segnet disagreement")]
    ):
        x, y = (i % 2) * w, (i // 2) * h
        canvas.paste(img, (x, y))
        ImageDraw.Draw(canvas).text((x + 6, y + 4), label, fill=(255, 255, 0))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--submission-dir", type=Path, required=True)
    ap.add_argument("--uncompressed-dir", type=Path, default=CHALLENGE / "videos")
    ap.add_argument("--video-names-file", type=Path, default=CHALLENGE / "public_test_video_names.txt")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--image-every", type=int, default=3, help="몇 배치마다 비교 이미지를 갱신할지")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--no-wait", action="store_true", help="끝나면 서버를 내리고 바로 종료")
    args = ap.parse_args()

    sub = args.submission_dir.resolve()
    viz = LiveVis(f"comma vcc · {sub.name}", port=args.port).start()

    device = torch.device("cpu")
    net = DistortionNet().eval().to(device)
    net.load_state_dicts(posenet_sd_path, segnet_sd_path, device)

    names = [line.strip() for line in args.video_names_file.read_text().splitlines() if line.strip()]
    common = dict(batch_size=args.batch_size, device=device, num_threads=2, seed=1234, prefetch_queue_depth=4)
    ds_gt = AVVideoDataset(names, data_dir=args.uncompressed_dir, **common)
    ds_comp = TensorVideoDataset(names, data_dir=sub / "inflated", **common)
    ds_gt.prepare_data()
    ds_comp.prepare_data()

    compressed = (sub / "archive.zip").stat().st_size
    original = sum(f.stat().st_size for f in args.uncompressed_dir.rglob("*") if f.is_file())
    rate = compressed / original

    pose_sum = seg_sum = 0.0
    n = 0
    t0 = time.time()
    with torch.inference_mode():
        for b, ((_, _, gt), (_, _, comp)) in enumerate(zip(ds_gt, ds_comp)):
            assert list(comp.shape)[1:] == [seq_len, camera_size[1], camera_size[0], 3], f"unexpected shape {comp.shape}"
            assert gt.shape == comp.shape, f"shape mismatch {gt.shape} vs {comp.shape}"
            pose_gt, seg_out_gt = net(gt)
            pose_comp, seg_out_comp = net(comp)
            pose_d = net.posenet.compute_distortion(pose_gt, pose_comp)
            seg_d = net.segnet.compute_distortion(seg_out_gt, seg_out_comp)

            pose_sum += pose_d.sum().item()
            seg_sum += seg_d.sum().item()
            n += gt.shape[0]
            viz.log(
                n,
                batch_posenet=pose_d.mean().item(),
                batch_segnet=seg_d.mean().item(),
                **score_terms(seg_sum / n, pose_sum / n, rate),
                samples_per_sec=n / (time.time() - t0),
            )
            if b % args.image_every == 0:
                i = int(seg_d.argmax())  # 배치에서 SegNet 이 가장 많이 틀린 샘플을 보여준다
                png = comparison_png(gt[i, -1], comp[i, -1], seg_out_gt[i].argmax(0), seg_out_comp[i].argmax(0))
                viz.image(
                    "worst sample in batch",
                    png,
                    step=n,
                    caption=f"pair #{n - gt.shape[0] + i}: segnet {seg_d[i].item():.4f} · posenet {pose_d[i].item():.4f}",
                )

    result = score_terms(seg_sum / n, pose_sum / n, rate)
    report = "\n".join(
        [
            f"=== {sub.name} · {n} samples · {time.time() - t0:.0f}s ===",
            f"  Average PoseNet Distortion: {pose_sum / n:.8f}",
            f"  Average SegNet Distortion: {seg_sum / n:.8f}",
            f"  Submission file size: {compressed:,} bytes",
            f"  Original uncompressed size: {original:,} bytes",
            f"  Compression Rate: {rate:.8f}",
            f"  Score terms: segnet {result['segnet_term']:.4f} + posenet {result['posenet_term']:.4f} + rate {result['rate_term']:.4f}",
            f"  Final score: {result['score']:.4f}",
        ]
    )
    print(report, flush=True)
    (sub / "live_report.txt").write_text(report + "\n")
    if not args.no_wait:
        viz.wait()
    viz.stop()


if __name__ == "__main__":
    main()
