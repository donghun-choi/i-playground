"""segcodec 정수화 손실과 인코드/디코드 왕복 확인.  python test_segcodec.py [프레임 수]"""
import sys
import time

import numpy as np
import torch

from common import CACHE, load_gt
import segcodec as sc  # noqa: E402  (common 이 경로를 잡아준다)
from ctxmodel import CtxNet, eval_bits

torch.set_num_threads(1)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ckpt = sys.argv[2] if len(sys.argv) > 2 else str(CACHE / "ctxnet.pt")
_, seg, _ = load_gt()
sd = torch.load(ckpt)
ch = sd["net.0.weight"].shape[0]
layers = sum(1 for k in sd if k.endswith(".weight")) - 1
model = CtxNet(ch, layers)
model.load_state_dict(sd)

calib = []
for t in (50, 250, 450):
    for s in sc.LEVELS:
        for kind in "AB":
            calib.append(sc.build_input_q(seg[t : t + 1], seg[t - 1 : t], seg[t - 2 : t - 1], s, kind))
q = sc.quantize_ctxnet(sd, calib)
blob = q.to_bytes()
print(f"정수 모델 {len(blob):,} bytes")

frames = list(range(0, n))
fb = sum(eval_bits(model, seg, frames).values()) / 8
t0 = time.time()
enc = sc.encode(seg[:n], q)
te = time.time() - t0
print(f"float 모델 추정 {fb / n:.0f} B/frame (coarse 제외) / 정수 모델 실제 {len(enc) / n:.0f} B/frame (coarse 포함), 인코드 {te:.1f}s")
q2, _ = sc.QNet.from_bytes(blob)
t0 = time.time()
dec = sc.decode(enc, q2)
td = time.time() - t0
print(f"디코드 {td:.1f}s ({td / n:.2f}s/frame), 왕복 일치: {np.array_equal(dec, seg[:n])}")
