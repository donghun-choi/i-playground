"""archive/p → <out_dir>/<name>.raw  (uint8 RGB 1164x874 프레임 1200장: 짝수, 홀수, 짝수, ...)

1. 문맥 CNN 복원 → seg 맵 601장 복호화 (0 번은 첫 쌍의 '이전 프레임' 용, 1..600 은 홀수 프레임 맵)
2. 렌더러 복원 → 홀수 프레임 i = 렌더러(맵 i+1), 첫 쌍의 이전 프레임 = 렌더러(맵 0)
3. 짝수 프레임 i = 아핀_i(홀수 프레임 i-1) + 기저 합성 (pose 파라미터)
4. 512x384 float → 1164x874 uint8 서브픽셀 확장해서 기록
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import archive  # noqa: E402
import segcodec  # noqa: E402
import wcodec  # noqa: E402
from model import SH, SW, Renderer, even_frames, expand_fine, unpack_state  # noqa: E402


def log(msg):
    print(f"[inflate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_renderer(sec: bytes) -> Renderer:
    cfg, ec = archive.unpack_renderer(sec)
    G = Renderer(*cfg)
    tmpl = G.state_dict()
    G.load_state_dict(unpack_state(wcodec.ec_unpack(ec, tmpl), tmpl))
    return G.eval()


def reconstruct(p: bytes, batch: int = 8):
    """archive 바이트 → (even, odd) float32 (600,384,512,3) 두 배열 (평가 네트워크가 볼 값)."""
    sec = archive.unpack(p)
    qnet, _ = segcodec.QNet.from_bytes(wcodec.ec_unpack_qnet(sec["ctxe"]))
    if not qnet.self_check():
        log("float32 합성곱이 정확하지 않아 float64 로 디코드합니다 (느림)")
    t = time.time()
    seg = segcodec.decode(sec["segs"], qnet)
    log(f"seg 맵 {len(seg)}장 복호화 {time.time() - t:.0f}s")

    G = load_renderer(sec["rend"])
    pos = archive.unpack_pose4(sec["pos4"])
    t = time.time()
    pre, seg = seg[:1], seg[1:]
    n = len(seg)
    with torch.inference_mode():
        odd = torch.zeros(n, 3, SH, SW)
        for i in range(0, n, batch):
            odd[i : i + batch] = G(torch.from_numpy(seg[i : i + batch]))
        prev0 = G(torch.from_numpy(pre))
        even = np.zeros((n, SH, SW, 3), np.float32)
        for i in range(0, n, batch):
            j = min(i + batch, n)
            prev = torch.cat([prev0, odd[: j - 1]]) if i == 0 else odd[i - 1 : j - 1]
            even[i:j] = even_frames(prev, pos["a"][i:j], pos["c"][i:j], pos["B"]).permute(0, 2, 3, 1).numpy()
    log(f"렌더링 {time.time() - t:.0f}s")
    return even, odd.permute(0, 2, 3, 1).contiguous().numpy()


def main(archive_dir: str, out_dir: str, list_file: str):
    torch.set_num_threads(int(os.environ.get("INFLATE_THREADS", os.cpu_count() or 4)))
    names = [line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]
    assert len(names) == 1, "이 archive 는 영상 하나용"
    even, odd = reconstruct((Path(archive_dir) / "p").read_bytes())
    out = Path(out_dir) / (Path(names[0]).stem + ".raw")
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(out, "wb") as f:
        for i in range(0, len(odd), 20):
            pair = np.stack([even[i : i + 20], odd[i : i + 20]], 1).reshape(-1, SH, SW, 3)
            f.write(expand_fine(pair).tobytes())
    log(f"{out} 기록 {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
