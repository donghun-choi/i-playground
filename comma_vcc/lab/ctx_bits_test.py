"""문맥 모델 가중치 비트 수별: 모델 크기(xz) 와 실제 seg 스트림 크기 (일부 프레임)."""
import sys

import numpy as np
import torch

from common import CACHE, load_gt
import archive  # noqa: E402
import segcodec as sc  # noqa: E402
from ctxmodel import load_ctx

torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
ckpt = sys.argv[1]
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 60
_, seg, _ = load_gt()
_, sd, dils = load_ctx(ckpt)
calib = [sc.build_input_q(seg[t : t + 1], seg[t - 1 : t], seg[t - 2 : t - 1], s, k) for t in (50, 200, 350, 500) for s in sc.LEVELS for k in "AB"]
sub = seg[200 : 200 + nf]
for wb in (8, 7, 6):
    q = sc.quantize_ctxnet(sd, calib, wbits=wb, dils=dils)
    model_b = len(archive.xz(q.to_bytes()))
    stream_b = len(sc.encode(sub, q))
    print(f"{wb}비트: 모델 {model_b:,} B, 스트림 {stream_b / nf:.1f} B/frame → 600장 환산 합계 {model_b + stream_b / nf * 600:,.0f} B", flush=True)
