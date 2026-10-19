"""재작성 검증용 기준값: 참조 디코더 (submissions/semantic_cpu) 로 archive 를 풀면서 단계별 중간 결과를 cache/golden/ 에 저장한다.

    python golden.py dump [--archive ../submissions/semantic_cpu/archive.zip]   # 약 5분
    python golden.py diff 내_결과.npy [내_결과.npz ...]                          # cache/golden/ 의 같은 이름 파일과 비교

저장하는 것 (단계 순서)
  sections.json            섹션별 길이·sha256, QNet 층 모양·dilation, 렌더러 폭, pose 모양, 단계별 sha256
  qnet_layers.npz          QNet 정수 가중치 W{i} B{i} M{i} dil relu                      (ctxe 섹션 풀기)
  qnet_probe_in.npy        무작위 정수 입력 (1,23,192,256)                                (정수 CNN 산술)
  qnet_probe_out.npy       그 logit
  inputs_t5_s4_A2.npy      맵 5, 레벨 4, A2 패스의 23채널 입력 (1,23,192,256)              (입력 만들기)
  logits_t5_s4_A2.npy      그 입력의 logit (1,5,192,256)
  probs_t5_s4_A2.npy       보낼 칸들의 확률 (m,5) float64 (행 우선 칸 순서)               (확률표)
  seg.npz                  복호화한 seg 맵 (601,384,512) uint8                              (segs 섹션 전체)
  renderer.npz             복원한 렌더러 float 가중치 (state_dict 이름)                     (rend 섹션)
  pose.npz                 B (8,1,24,32), c (600,8), a (600,6) float32                     (pos4 섹션)
  odd_0_3.npy / even_0_3.npy   홀수/짝수 프레임 0..3 (4,384,512,3) float32
  frame_sums.npy           (600,2,3) float64 프레임별 채널 합 (짝수, 홀수) — 전체 프레임 대략 비교
  raw_pair0.npy            첫 쌍 (짝수, 홀수) 의 1164x874 uint8 (2,874,1164,3)
  raw.sha256               전체 .raw (1200 프레임) 의 sha256
seg 맵·QNet 출력·확률은 비트 단위로 같아야 한다 (하나라도 다르면 range decoder 가 어긋난다).
렌더 프레임은 합성곱 구현에 따라 1e-4 레벨 차이가 날 수 있다 (점수에는 영향 없음, README 의 AVX2 실험).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from common import CACHE, SUB
import archive  # noqa: E402
import segcodec as sc  # noqa: E402
import wcodec  # noqa: E402
from inflate import load_renderer  # noqa: E402
from model import SH, SW, even_frames, expand_fine  # noqa: E402

GOLDEN = CACHE / "golden"


def sha(x) -> str:
    return hashlib.sha256(x if isinstance(x, bytes) else np.ascontiguousarray(x).tobytes()).hexdigest()


def dump(archive_zip: str, batch: int = 8):
    GOLDEN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    p = zipfile.ZipFile(archive_zip).read("p")
    sec = archive.unpack(p)
    info = {"archive_sha256": sha(p), "sections": {k: {"bytes": len(v), "sha256": sha(v)} for k, v in sec.items()}}
    t0 = time.time()

    # ctxe → QNet
    qb = wcodec.ec_unpack_qnet(sec["ctxe"])
    qnet, _ = sc.QNet.from_bytes(qb)
    info["qnet_bytes_sha256"] = sha(qb)
    info["qnet_layers"] = [{"W": list(W.shape), "dil": dil, "relu": relu} for W, _, _, dil, relu in qnet.layers]
    np.savez(GOLDEN / "qnet_layers.npz", **{f"{n}{i}": a for i, L in enumerate(qnet.layers) for n, a in zip("WBM", L[:3])},
             dil=np.array([L[3] for L in qnet.layers]), relu=np.array([L[4] for L in qnet.layers]))
    info["float32_conv_exact"] = qnet.self_check()
    x = torch.from_numpy(np.random.default_rng(0).integers(0, sc.Q_IN + 1, (1, sc.C_IN, SH // 2, SW // 2))).float()
    np.save(GOLDEN / "qnet_probe_in.npy", x.numpy())
    np.save(GOLDEN / "qnet_probe_out.npy", qnet.run(x).numpy())

    # segs → seg 맵
    t = time.time()
    seg = sc.decode(sec["segs"], qnet)
    print(f"seg 맵 {len(seg)}장 복호화 {time.time() - t:.0f}s", flush=True)
    np.savez_compressed(GOLDEN / "seg.npz", seg=seg)
    info["seg_sha256"] = sha(seg)

    # 맵 5, 레벨 4, A2 패스의 입력·logit·확률
    fi = sc.Inputs()
    tt, s, kind = 5, 4, "A2"
    xin = fi.build(torch.from_numpy(seg[tt : tt + 1]), s, kind, fi.fracs(torch.from_numpy(seg[tt - 1 : tt])), fi.fracs(torch.from_numpy(seg[tt - 2 : tt - 1])))
    logit = qnet.run(xin)
    np.save(GOLDEN / "inputs_t5_s4_A2.npy", xin.numpy())
    np.save(GOLDEN / "logits_t5_s4_A2.npy", logit.numpy())
    np.save(GOLDEN / "probs_t5_s4_A2.npy", sc.probs(logit[0][:, fi.target[s // 2, kind]].T))

    # rend → 렌더러, pos4 → pose
    G = load_renderer(sec["rend"])
    np.savez(GOLDEN / "renderer.npz", **{k: v.numpy() for k, v in G.state_dict().items()})
    info["renderer_widths"] = list(G.cfg)
    pos = archive.unpack_pose4(sec["pos4"])
    np.savez(GOLDEN / "pose.npz", **{k: v.numpy() for k, v in pos.items()})
    info["pose_shapes"] = {k: list(v.shape) for k, v in pos.items()}

    # 프레임 (inflate.reconstruct 와 같은 배치 순서)
    t = time.time()
    pre, maps = seg[:1], seg[1:]
    n = len(maps)
    with torch.inference_mode():
        odd = torch.zeros(n, 3, SH, SW)
        for i in range(0, n, batch):
            odd[i : i + batch] = G(torch.from_numpy(maps[i : i + batch]))
        prev0 = G(torch.from_numpy(pre))
        even = torch.zeros(n, 3, SH, SW)
        for i in range(0, n, batch):
            j = min(i + batch, n)
            prev = torch.cat([prev0, odd[: j - 1]]) if i == 0 else odd[i - 1 : j - 1]
            even[i:j] = even_frames(prev, pos["a"][i:j], pos["c"][i:j], pos["B"])
    even, odd = even.permute(0, 2, 3, 1).contiguous().numpy(), odd.permute(0, 2, 3, 1).contiguous().numpy()
    print(f"렌더링 {time.time() - t:.0f}s", flush=True)
    np.save(GOLDEN / "odd_0_3.npy", odd[:4])
    np.save(GOLDEN / "even_0_3.npy", even[:4])
    np.save(GOLDEN / "frame_sums.npy", np.stack([even.sum((1, 2), dtype=np.float64), odd.sum((1, 2), dtype=np.float64)], 1))
    info["odd_sha256"], info["even_sha256"] = sha(odd), sha(even)

    # 원해상도 기록 (파일 대신 sha256 만)
    t = time.time()
    h = hashlib.sha256()
    for i in range(0, n, 20):
        full = expand_fine(np.stack([even[i : i + 20], odd[i : i + 20]], 1).reshape(-1, SH, SW, 3))
        if i == 0:
            np.save(GOLDEN / "raw_pair0.npy", full[:2])
        h.update(full.tobytes())
    (GOLDEN / "raw.sha256").write_text(h.hexdigest() + "\n")
    info["raw_sha256"] = h.hexdigest()
    print(f"확장 + sha256 {time.time() - t:.0f}s", flush=True)
    (GOLDEN / "sections.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"{GOLDEN} 에 저장 (총 {time.time() - t0:.0f}s)\n{json.dumps(info, indent=2, ensure_ascii=False)}")


def _cmp(name: str, a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        print(f"  {name}: 모양 다름 {a.shape} vs 기준 {b.shape}")
        return False
    same = np.array_equal(a, b)
    if same:
        print(f"  {name}: 완전히 같음 {a.shape} {a.dtype}")
    elif np.issubdtype(b.dtype, np.integer) or b.dtype == bool:
        bad = np.argwhere(a != b)
        print(f"  {name}: 다른 원소 {len(bad):,}/{a.size:,}, 첫 위치 {tuple(bad[0])}: {a[tuple(bad[0])]} vs 기준 {b[tuple(bad[0])]}")
    else:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print(f"  {name}: 최대 차이 {d.max():.3g}, 평균 {d.mean():.3g}, 다른 원소 {int((d != 0).sum()):,}/{d.size:,}")
    return same


def diff(paths: list[str]) -> bool:
    ok = True
    for path in paths:
        ref = GOLDEN / Path(path).name
        print(f"{path} vs {ref}")
        if path.endswith(".npz"):
            mine, gold = np.load(path), np.load(ref)
            for k in gold.files:
                ok &= k in mine.files and _cmp(k, mine[k], gold[k])
        else:
            ok &= _cmp(Path(path).stem, np.load(path), np.load(ref))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--archive", default=str(SUB / "archive.zip"))
    c = sub.add_parser("diff")
    c.add_argument("files", nargs="+")
    args = ap.parse_args()
    if args.cmd == "dump":
        dump(args.archive)
    else:
        sys.exit(0 if diff(args.files) else 1)
