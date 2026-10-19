"""학습 결과 → submissions/semantic_cpu/archive.zip (+ seg 스트림 왕복 확인). 기본값은 v6.26 산출물.

  ctxe: 문맥 CNN (ctxmodel.py) → 정수화 (quantize_ctxnet) → QNet 바이트 → wcodec
  segs: seg 맵 601장 (cache/seg_pre.npy + 홀수 프레임 600장) 을 그 QNet 으로 부호화 (--segs 로 이미 만든 스트림 재사용)
  rend: 렌더러 정수 가중치 (--rbits 0 = self-compression 학습이 만든 '.q' 바이트, 아니면 pack_state(--rbits)) → wcodec
  pos4: pose_fit / pose_refine 의 pose2 blob 을 다시 포장 (posefmt.pose2_to_pose4)

    python build_archive.py                                         # v6.26 (seg 스트림을 다시 부호화 5~10분 + 왕복 확인 3분)
    python build_archive.py --segs ../cache/segs_pre_sa6p4_w.bin    # 스트림 재사용 (같은 문맥 모델일 때만)
"""

from __future__ import annotations

import argparse
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

from common import CACHE, SUB, load_gt, parse_widths, score
import archive  # noqa: E402
import segcodec as sc  # noqa: E402
from model import Renderer, pack_state  # noqa: E402
import wcodec  # noqa: E402
from posefmt import pose2_to_pose4


def quantize_ctxnet(state_dict: dict, calib_inputs: list, wbits: int = 8, dils=None, wq: dict | None = None) -> sc.QNet:
    """float CtxNet (Conv-ReLU 반복 + 마지막 1x1) → QNet. calib_inputs: segcodec.Inputs 결과 몇 개 (정수값 float, 0..64).

    층마다 활성 범위를 float 모델로 재서 (99.99% 분위수 → 255) 출력 스케일을 정하고, 정수 경로를 맞춘다:
      y_q = floor((acc + B_q) · M / 2^16),  acc 단위 = s_in · s_w  →  B_q = round(b / (s_in s_w)),  M = round(s_in s_w / s_out · 2^16)
    wq: 층 이름 → (정수 가중치, 출력 채널별 스케일). 주면 그 양자화를 그대로 쓴다 (self-compression 학습 결과).
    """
    convs = [(k[: -len(".weight")], v) for k, v in state_dict.items() if k.endswith(".weight")]
    x_float = [x / sc.Q_IN for x in calib_inputs]  # float 모델 입력 (0..1)
    s_in = 1.0 / sc.Q_IN  # 정수 입력 1 단위의 실수값
    layers = []
    qmax = 2 ** (wbits - 1) - 1
    for li, (name, Wf) in enumerate(convs):
        bf = state_dict[name + ".bias"]
        last = li == len(convs) - 1
        dil = 1 if (dils is None or li >= len(dils)) else dils[li]
        pad = dil * (Wf.shape[-1] // 2)
        if wq is not None and name in wq:
            W_q, s_w = (torch.as_tensor(v).float() for v in wq[name])
            Wf = W_q * s_w.view(-1, 1, 1, 1)  # 활성 범위도 학습한 (양자화된) 가중치로 잰다
        else:
            s_w = Wf.abs().amax((1, 2, 3)).clamp_min(1e-8) / qmax  # 출력 채널별
            W_q = (Wf / s_w.view(-1, 1, 1, 1)).round().clamp(-qmax, qmax)
        ys = [F.conv2d(x, Wf, bf, padding=pad, dilation=dil) for x in x_float]
        if last:
            s_out = 1.0 / sc.LOGIT_UNIT
        else:
            ys = [y.relu() for y in ys]
            hi = torch.quantile(torch.cat([y.flatten()[:: max(1, y.numel() // 200000)] for y in ys]), 0.9999).item()
            s_out = max(hi, 1e-3) / 255
        unit = s_in * s_w
        B_q = (bf / unit).round()
        M = (unit / s_out * 2**sc.SHIFT).round()
        layers.append((W_q.numpy().astype(np.int64), B_q.numpy().astype(np.int64), M.numpy().astype(np.int64), dil, not last))
        x_float = ys if not last else x_float
        s_in = s_out
    return sc.QNet(layers)


def build_qnet(ckpt: str, seg, wbits: int = 8) -> sc.QNet:
    """문맥 CNN 체크포인트 → QNet. 보정 입력: 프레임 50/200/350/500 (홀수 프레임 맵 번호) 의 모든 레벨·패스."""
    from ctxmodel import load_ctx

    _, sd, dils = load_ctx(ckpt)
    fi = sc.Inputs()
    calib = []
    for t in (50, 200, 350, 500):
        fp, fp2 = (fi.fracs(torch.from_numpy(seg[u : u + 1])) for u in (t - 1, t - 2))
        calib += [fi.build(torch.from_numpy(seg[t : t + 1]), s, k, fp, fp2) for s in sc.LEVELS for k in sc.PASSES]
    wq = torch.load(ckpt).get("wq")  # self-compression 학습 결과 (채널별 비트) 면 그 양자화를 그대로
    return quantize_ctxnet(sd, calib, wbits=wbits, dils=dils, wq=wq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default=str(CACHE / "ctxnet_c24sa6p4.pt"))
    ap.add_argument("--cbits", type=int, default=5, help="문맥 모델 가중치 비트 수 (self-compression 체크포인트면 쓰이지 않는다)")
    ap.add_argument("--renderer", default=str(CACHE / "renderer_c8_sc7g2pt.pt"))
    ap.add_argument("--renderer-cfg", default="24,32,40", help="렌더러 폭 c1,c2,c3")
    ap.add_argument("--rbits", type=int, default=0, help="렌더러 저장 비트 수 (pose 피팅 때와 같게). 0 = self-compression ('.q' 바이트)")
    ap.add_argument("--pose2", default=str(CACHE / "pose2_pt_gray3.bin"))
    ap.add_argument("--segs", default=None, help="이미 부호화한 seg 스트림 재사용 (같은 문맥 모델일 때만)")
    ap.add_argument("--out", default=str(SUB / "archive.zip"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--segs-only", action="store_true", help="seg 스트림만 부호화해서 cache/segs_pre.bin 에 저장")
    ap.add_argument("--no-verify", action="store_true", help="seg 복호화 왕복 확인 생략")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    _, seg, _ = load_gt()

    t = time.time()
    q = build_qnet(args.ctx, seg, args.cbits)
    ctxe = wcodec.ec_pack_qnet(q.to_bytes())
    seg = np.concatenate([np.load(CACHE / "seg_pre.npy"), seg])  # 맨 앞 = 첫 쌍의 '이전 프레임' 용 맵 (pose_fit.py 가 만든다)
    if args.segs:
        segs = open(args.segs, "rb").read()
    else:
        segs = sc.encode(seg, q)
        open(CACHE / "segs_pre.bin", "wb").write(segs)
    print(f"seg 스트림 {len(segs):,} B ({len(segs) / len(seg):.1f} B/frame), 문맥 모델 {len(ctxe):,} B ({time.time() - t:.0f}s)", flush=True)
    if args.segs_only:
        return

    widths = parse_widths(args.renderer_cfg)
    if args.rbits == 0:  # self-compression: 학습 때 만든 바이트 (채널별 비트) 그대로
        wbytes = open(args.renderer + ".q", "rb").read()
    else:
        wbytes = pack_state(torch.load(args.renderer), args.rbits)
    tmpl = Renderer(*widths).state_dict()
    ec = wcodec.ec_pack(wbytes, tmpl)
    assert wcodec.ec_unpack(ec, tmpl) == wbytes
    sections = {"ctxe": ctxe, "segs": segs, "rend": archive.pack_renderer(widths, ec), "pos4": pose2_to_pose4(open(args.pose2, "rb").read())}
    p = archive.pack(sections)
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_STORED) as z:
        z.writestr(zipfile.ZipInfo("p", date_time=(2026, 1, 1, 0, 0, 0)), p)
    size = len(open(args.out, "rb").read())
    for name, b in sections.items():
        print(f"  {name}: {len(b):>8,} B  (rate 항 {score(0, 0, len(b))['rate_term']:.4f})")
    print(f"{args.out} {size:,} B → rate 항 {score(0, 0, size)['rate_term']:.4f}")

    if args.no_verify:
        return
    t = time.time()
    q2, _ = sc.QNet.from_bytes(wcodec.ec_unpack_qnet(ctxe))
    q2.self_check()
    print(f"seg 왕복 일치: {np.array_equal(sc.decode(segs, q2), seg)} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
