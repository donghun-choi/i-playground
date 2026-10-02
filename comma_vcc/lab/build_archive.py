"""학습 산출물 → submissions/semantic_cpu/archive.zip

  ctxn: 정수 문맥 CNN (lzma)
  segs: seg 맵 600장 range coder 스트림
  rend: 렌더러 int8 가중치 (lzma)
  carr: pose carrier (carrier_fit.py 결과)

    python build_archive.py --ctx ../cache/ctxnet_c24l5.pt --renderer ../cache/renderer.pt --carrier ../cache/carrier.bin
"""

from __future__ import annotations

import argparse
import time
import zipfile

import numpy as np
import torch

from common import CACHE, SUB, load_gt, score
import archive  # noqa: E402
import segcodec as sc  # noqa: E402
from model import pack_state  # noqa: E402


def build_qnet(ckpt: str, seg) -> sc.QNet:
    from ctxmodel import load_ctx

    _, sd, dils = load_ctx(ckpt)
    calib = [sc.build_input_q(seg[t : t + 1], seg[t - 1 : t], seg[t - 2 : t - 1], s, k)
             for t in (50, 200, 350, 500) for s in sc.LEVELS for k in "AB"]
    return sc.quantize_ctxnet(sd, calib, dils=dils)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default=str(CACHE / "ctxnet.pt"))
    ap.add_argument("--renderer", default=str(CACHE / "renderer.pt"))
    ap.add_argument("--carrier", default=None, help="v1 carrier (회색 기준)")
    ap.add_argument("--pose2", default=None, help="pose v2 (이전 렌더 + 아핀 + carrier). 이 경우 seg 스트림 맨 앞에 첫 쌍용 맵을 더한다")
    ap.add_argument("--segs", default=None, help="이미 인코드한 seg 스트림 재사용 (ctx 모델이 같을 때만)")
    ap.add_argument("--out", default=str(SUB / "archive.zip"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--renderer-cfg", default=None, help="RendererV2 이면 'width,fdim'")
    ap.add_argument("--rbits", type=int, default=8, help="렌더러 저장 비트 수 (pose_fit 과 같게)")
    ap.add_argument("--no-verify", action="store_true", help="seg 디코드 왕복 확인 생략")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    _, seg, _ = load_gt()

    t = time.time()
    q = build_qnet(args.ctx, seg)
    ctxn = archive.xz(q.to_bytes())
    if args.pose2:
        seg = np.concatenate([np.load(CACHE / "seg_pre.npy"), seg])
    if args.segs:
        segs = open(args.segs, "rb").read()
    else:
        segs = sc.encode(seg, q)
        open(CACHE / ("segs_pre.bin" if args.pose2 else "segs.bin"), "wb").write(segs)
    print(f"seg 스트림 {len(segs):,} B ({len(segs) / len(seg):.0f} B/frame), 문맥모델 {len(ctxn):,} B ({time.time() - t:.0f}s)", flush=True)
    cfg = None
    if args.renderer_cfg:
        w, fd = (int(v) for v in args.renderer_cfg.split(","))
        cfg = (w, fd, len(seg), (1, 1, 2, 4))
    rend = archive.pack_renderer(cfg, pack_state(torch.load(args.renderer), args.rbits))
    if args.pose2:
        pose_sec = ("pos2", open(args.pose2, "rb").read())
    else:
        pose_sec = ("carr", open(args.carrier, "rb").read())
    p = archive.pack({"ctxn": ctxn, "segs": segs, "rend": rend, pose_sec[0]: pose_sec[1]})
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(zipfile.ZipInfo("p", date_time=(2026, 1, 1, 0, 0, 0)), p)
    size = len(open(args.out, "rb").read())
    for name, b in (("ctxn", ctxn), ("segs", segs), ("rend", rend), pose_sec):
        print(f"  {name}: {len(b):>8,} B  (rate 항 {score(0, 0, len(b))['rate_term']:.4f})")
    print(f"archive.zip {size:,} B → rate 항 {score(0, 0, size)['rate_term']:.4f}")

    if args.no_verify:
        return
    # 디코드 검증 (inflate 와 같은 함수)
    t = time.time()
    q2, _ = sc.QNet.from_bytes(archive.unxz(ctxn))
    dec = sc.decode(segs, q2)
    print(f"seg 왕복 일치: {np.array_equal(dec, seg)} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
