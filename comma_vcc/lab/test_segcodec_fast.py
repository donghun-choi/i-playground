"""새 (빠른) segcodec 이 기존 스트림과 비트 단위로 같은지: 인코드 바이트 비교 + 기존 segs.bin 앞부분 디코드."""
import sys
import time

import numpy as np
import torch

from common import CACHE, load_gt
import segcodec as sc  # noqa: E402
from build_archive import build_qnet

torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 1)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
_, seg, _ = load_gt()
q = build_qnet(str(CACHE / "ctxnet_c24l5.pt"), seg)

# 기존 (느린) 경로로 n 프레임 인코드한 것과 새 경로 인코드가 같은가
def old_encode(seg_, qnet):
    import constriction, struct
    enc = constriction.stream.queue.RangeEncoder(); fam = sc._family(); cm = sc.CoarseModel(); cy, cx = sc.coarse_pos()
    for t in range(len(seg_)):
        cur = seg_[t:t+1]; prev = seg_[t-1:t] if t >= 1 else None; prev2 = seg_[t-2:t-1] if t >= 2 else None
        sym = cur[0, cy, cx].astype(np.int32); ctx = prev[0, cy, cx].astype(np.int64) if prev is not None else np.full(len(cy), sc.K)
        enc.encode(sym, fam, cm.probs(ctx)); cm.update(ctx, sym)
        for s in sc.LEVELS:
            h = s // 2
            for kind in "AB":
                _, target = sc.masks(h, kind)
                logit = qnet(sc.build_input_q(cur, prev, prev2, s, kind))[0]
                enc.encode(cur[0, ::h, ::h][target].astype(np.int32), fam, sc.probs_from_logits(logit[:, target].T))
    return struct.pack("<I", len(seg_)) + enc.get_compressed().tobytes()

t = time.time(); a = old_encode(seg[:n], q); ta = time.time() - t
t = time.time(); b = sc.encode(seg[:n], q); tb = time.time() - t
print(f"인코드 바이트 동일: {a == b} (기존 {ta:.1f}s, 새 {tb:.1f}s)")
t = time.time(); d = sc.decode(b, q); td = time.time() - t
print(f"새 디코드 왕복: {np.array_equal(d, seg[:n])} ({td / n:.3f}s/frame)")
