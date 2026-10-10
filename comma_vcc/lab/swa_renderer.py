"""렌더러 가중치 평균 (SWA): self-compression 체크포인트 여러 개의 float 가중치와 비트/스케일을 평균해서
그 양자화로 600장 불일치와 엔트로피 부호 크기를 잰다. 좋으면 renderer.py --resume 과 같은 형식으로 저장."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from common import load_gt, nets
import archive  # noqa: E402
import wcodec  # noqa: E402
from model import make_renderer  # noqa: E402
from renderer import evaluate
from selfcomp import SelfCompress


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="renderer .pt 경로들 (각각 .sc.pt 가 있어야 함)")
    ap.add_argument("--widths", default="24,32,40")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    widths = tuple(int(v) for v in args.widths.split(","))
    sts = [torch.load(c + ".sc.pt", map_location="cpu") for c in args.ckpts]
    G = make_renderer(widths)
    G.load_state_dict({k: sum(st["float"][k] for st in sts) / len(sts) for k in sts[0]["float"]})
    sc = SelfCompress(G, sts[0]["init_bits"])
    sc.load_state_dict({k: sum(st["sc"][k] for st in sts) / len(sts) for k in sts[0]["sc"]})
    sd_exp, packed = sc.export(G)
    nbytes = len(archive.pack_renderer(G.cfg, wcodec.ec_pack(packed, G.state_dict())))
    net = nets()
    net.segnet.to(memory_format=torch.channels_last)
    _, seg, _ = load_gt()
    errs = evaluate(G, net, seg, np.arange(len(seg)), sd=sd_exp)
    d = errs.mean().item()
    print(f"평균 {len(sts)}개: 600장 불일치 {d:.6f}, 렌더러 {nbytes:,} B → seg + 렌더러 rate {100 * d + 25 * nbytes / 37_545_489:.4f}", flush=True)
    if args.out:
        torch.save(sd_exp, args.out)
        open(args.out + ".q", "wb").write(packed)
        torch.save({"float": G.state_dict(), "sc": sc.state_dict(), "init_bits": sts[0]["init_bits"]}, args.out + ".sc.pt")


if __name__ == "__main__":
    main()
