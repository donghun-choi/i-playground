"""배관 테스트용 carrier: 계수 0 (pose 는 엉망이지만 포맷/inflate 경로 검증용)."""
import numpy as np

from common import CACHE
import archive  # noqa: E402

k, C, bh, bw = 12, 3, 24, 32
B_q = np.zeros((k, C, bh, bw), np.int8)
blob = archive.pack_carrier(0, "gray", "bicubic", B_q, np.ones(k, np.float32), np.zeros((600, k), np.int64), np.ones(k, np.float32), C, bh, bw)
open(CACHE / "carrier_dummy.bin", "wb").write(blob)
print(len(blob), "bytes")
