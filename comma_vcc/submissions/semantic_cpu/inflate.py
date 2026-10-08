"""archive/p → <out_dir>/<name>.raw  (uint8 RGB 1164x874 프레임 나열)

1. seg 맵 600장 복호화 (정수 문맥 CNN + range coder)
2. 렌더러로 홀수 프레임 생성
3. pose carrier 로 짝수 프레임 생성
4. 512x384 → 1164x874 확장해서 기록
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
from model import even_frames, even_frames_prev, expand_fine, make_renderer, render, unpack_state  # noqa: E402


def log(msg):
    print(f"[inflate {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def reconstruct(p: bytes, batch: int = 8):
    """archive 바이트 → (even, odd) float32 (600,384,512,3) 두 배열 (평가 네트워크가 볼 값)."""
    sec = archive.unpack(p)
    qnet, _ = segcodec.QNet.from_bytes(wcodec.ec_unpack_qnet(sec["ctxe"]) if "ctxe" in sec else archive.unxz(sec["ctxn"]))
    probe = np.random.default_rng(0).integers(0, segcodec.Q_IN + 1, (1, segcodec.C_IN, 48, 64))
    if not qnet.self_check(probe):
        log("float32 합성곱이 정확하지 않아 float64 로 디코드합니다 (느림)")
    t = time.time()
    seg = segcodec.decode(sec["segs"], qnet)
    log(f"seg 맵 {len(seg)}장 복호화 {time.time() - t:.0f}s")

    cfg, wbytes = archive.unpack_renderer(sec["rend"])
    G = make_renderer(cfg)
    if wcodec.is_ec(wbytes):
        wbytes = wcodec.ec_unpack(wbytes, G.state_dict())
    G.load_state_dict(unpack_state(wbytes, G.state_dict()))
    G.eval()
    t = time.time()
    with torch.inference_mode():
        if "pos2" in sec or "pos3" in sec:
            # seg 스트림 맨 앞 한 장은 첫 쌍의 '이전 프레임' 용 맵
            pre, seg = seg[:1], seg[1:]
            n = len(seg)
            odd_t = torch.zeros(n, 3, 384, 512)
            for i in range(0, n, batch):
                odd_t[i : i + batch] = render(G, torch.from_numpy(seg[i : i + batch]), torch.arange(i, min(i + batch, n)))
            prev0 = render(G, torch.from_numpy(pre), torch.zeros(1, dtype=torch.long))
            pos = archive.unpack_pose3(sec["pos3"]) if "pos3" in sec else archive.unpack_pose2(sec["pos2"])
            even = np.zeros((n, 384, 512, 3), np.float32)
            for i in range(0, n, batch):
                prev = torch.cat([prev0, odd_t[: min(i + batch, n) - 1]]) if i == 0 else odd_t[i - 1 : min(i + batch, n) - 1]
                e = even_frames_prev(prev, pos["a"][i : i + batch], pos["c"][i : i + batch], pos["B"])
                even[i : i + batch] = e.permute(0, 2, 3, 1).numpy()
            odd = odd_t.permute(0, 2, 3, 1).contiguous().numpy()
        else:
            car = archive.unpack_carrier(sec["carr"])
            n = len(seg)
            odd = np.zeros((n, 384, 512, 3), np.float32)
            even = np.zeros((n, 384, 512, 3), np.float32)
            for i in range(0, n, batch):
                o = render(G, torch.from_numpy(seg[i : i + batch]), torch.arange(i, min(i + batch, n)))
                e = even_frames(o, car["c"][i : i + batch], car["B"], car["base"], car["mode"])
                odd[i : i + batch] = o.permute(0, 2, 3, 1).numpy()
                even[i : i + batch] = e.permute(0, 2, 3, 1).numpy()
    log(f"렌더링 {time.time() - t:.0f}s")
    return even, odd


def main(archive_dir: str, out_dir: str, list_file: str):
    torch.set_num_threads(int(os.environ.get("INFLATE_THREADS", os.cpu_count() or 4)))
    names = [line.strip() for line in Path(list_file).read_text().splitlines() if line.strip()]
    assert len(names) == 1, "이 archive 는 영상 하나용"
    p = (Path(archive_dir) / "p").read_bytes()
    even, odd = reconstruct(p)
    out = Path(out_dir) / (Path(names[0]).stem + ".raw")
    out.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(out, "wb") as f:
        for i in range(0, len(odd), 20):
            pair = np.stack([even[i : i + 20], odd[i : i + 20]], 1).reshape(-1, 384, 512, 3)
            f.write(expand_fine(pair).tobytes())
    log(f"{out} 기록 {time.time() - t:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:4])
