"""archive/p → <out_dir>/<name>.raw (uint8 RGB 1164x874 프레임 1200장: 짝수, 홀수, 짝수, ...). 흐름만 있고 단계는 다른 파일에.

1. ctxe → 정수 문맥 CNN,  segs → seg 맵 601장 (0 번 = M_pre, i+1 번 = 홀수 프레임 i 의 맵)
2. rend → 렌더러,  홀수 프레임 i = 렌더러(맵 i+1),  첫 쌍의 이전 프레임 = 렌더러(맵 0)
3. pos4 → 기저·계수·아핀,  짝수 프레임 i = 아핀_i(홀수 프레임 i-1) + 기저 합성
4. 쌍마다 (짝수, 홀수) 를 서브픽셀 확장해서 기록
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

import container  # noqa: E402
import ctxnet  # noqa: E402
import pose  # noqa: E402
import render  # noqa: E402
import segdec  # noqa: E402
import upscale  # noqa: E402


def log(msg):
    print(f"[inflate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reconstruct(p: bytes):
    """→ (even, odd) torch (600,3,384,512) float32"""
    sec = container.read_sections(p)
    net = ctxnet.IntCNN(ctxnet.parse_qnet(ctxnet.unpack_ctxe(sec["ctxe"])))
    if not net.self_check():
        log("float32 합성곱이 정확하지 않아 float64 로")
    t = time.time()
    maps = segdec.decode_maps(sec["segs"], net)
    log(f"seg 맵 {len(maps)}장 {time.time() - t:.0f}s")

    t = time.time()
    G = render.load_renderer(sec["rend"])
    odd = render.render_all(G, maps[1:])
    prev = torch.cat([render.render_all(G, maps[:1]), odd[:-1]])
    B, c, a = pose.parse_pose4(sec["pos4"])
    with torch.inference_mode():
        even = pose.even_frames(prev, a, c, B)
    log(f"렌더링 {time.time() - t:.0f}s")
    return even, odd


def main(archive_dir: str, out_dir: str, list_file: str):
    torch.set_num_threads(int(os.environ.get("INFLATE_THREADS", os.cpu_count() or 4)))
    names = [line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]
    assert len(names) == 1, "영상 하나용"
    even, odd = reconstruct((Path(archive_dir) / "p").read_bytes())
    out = Path(out_dir) / (Path(names[0]).stem + ".raw")
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(out, "wb") as f:
        for i in range(0, len(odd), 20):
            pair = torch.stack([even[i : i + 20], odd[i : i + 20]], 1).flatten(0, 1)  # 짝수, 홀수, 짝수, ...
            f.write(upscale.expand_fine(pair.permute(0, 2, 3, 1).contiguous().numpy()).tobytes())
    log(f"{out} 기록 {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
