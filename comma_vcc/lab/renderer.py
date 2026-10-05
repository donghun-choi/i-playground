"""seg 맵 → 홀수 프레임(512x384 RGB) 렌더러.

SegNet(render(M)) 의 argmax 가 M 과 같아지도록 학습한다. 렌더러 가중치는 archive 에 들어가므로 작게 유지한다.

    python renderer.py --epochs 20 --out ../cache/renderer.pt
"""

from __future__ import annotations

import argparse
import io
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import CACHE, SH, SW, load_gt, nets
import copy

from model import Renderer, fake_quant_, make_renderer, quantize_roundtrip, render, widen_renderer  # noqa: E402  (제출물 패키지와 같은 정의)

PALETTE = np.array([[64, 64, 64], [230, 230, 230], [42, 120, 214], [12, 163, 12], [208, 59, 59]], np.uint8)


def quantize_ste(x: torch.Tensor) -> torch.Tensor:
    """반올림 (gradient 는 그대로 통과). 실제 프레임은 uint8 이라 학습 때부터 반올림을 넣는다."""
    return x + (x.round() - x).detach()


def png(arr: np.ndarray) -> bytes:
    b = io.BytesIO()
    Image.fromarray(arr).save(b, format="PNG")
    return b.getvalue()


def ste_params(G, bits):
    """QAT: pack_state 와 같은 격자로 가짜 양자화한 파라미터 (기울기는 float 원본으로 그대로 흐른다)."""
    qmax = 2 ** (bits - 1) - 1
    out = {}
    for k, p in G.named_parameters():
        if p.ndim >= 2:
            s = (p.detach().abs().reshape(p.shape[0], -1).amax(1).clamp_min(1e-8) / qmax).half().float().clamp_min(1e-8).view(-1, *[1] * (p.ndim - 1))
            q = (p / s).round().clamp(-qmax, qmax) * s
        else:
            q = p.half().float()
        out[k] = p + (q - p).detach()
    return out


@torch.inference_mode()
def evaluate(G, net, seg, idx, bs=8, rounded=False, bits=8):
    """inflate 와 같은 조건: 저장/복원 왕복 가중치, float 출력 (서브픽셀 확장으로 거의 그대로 전달된다)."""
    Gq = copy.deepcopy(G)
    quantize_roundtrip(Gq, bits)
    errs = []
    for i in range(0, len(idx), bs):
        m = torch.from_numpy(seg[idx[i : i + bs]])
        img = render(Gq, m, torch.from_numpy(np.asarray(idx[i : i + bs])))
        out = net.segnet(img.round() if rounded else img).argmax(1)
        errs.append((out != m).float().mean((1, 2)))
    return torch.cat(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=str(CACHE / "renderer.pt"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--round", action="store_true", help="학습 때 출력 반올림 (정수 프레임용, 서브픽셀 확장을 쓰면 불필요)")
    ap.add_argument("--fp32", action="store_true", help="SegNet 을 bf16 대신 fp32 로 (느리지만 평가와 같은 수치)")
    ap.add_argument("--cosine", action="store_true", help="OneCycle 대신 cosine 감쇠 (이어서 학습할 때)")
    ap.add_argument("--arch", default="v1", help="v1 | v2 (전해상도 + 프레임별 FiLM)")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--fdim", type=int, default=8)
    ap.add_argument("--widths", default=None, help="v1 폭 c1,c2,c3 (예: 24,32,40)")
    ap.add_argument("--film", type=int, default=0, help="> 0 이면 v1 에 프레임별 FiLM (이 차원의 프레임 코드). --resume 이 FiLM 없는 체크포인트면 FiLM 0 에서 시작")
    ap.add_argument("--widen-from", default=None, help="이 v1 체크포인트를 --widths 로 넓혀서 시작 (처음엔 같은 출력)")
    ap.add_argument("--bits", type=int, default=8, help="저장 비트 수 (평가에 반영)")
    ap.add_argument("--project", action="store_true", help="스텝마다 가중치를 --bits 격자로 투영 (양자화 인지 미세조정)")
    ap.add_argument("--qat", action="store_true", help="STE 가짜 양자화로 학습 (--bits 격자, float 원본 가중치 유지)")
    ap.add_argument("--full-eval", action="store_true", help="끝나고 600장 전체 평가")
    ap.add_argument("--start-epoch", type=int, default=0, help="--resume 와 함께: 이 에폭부터 (학습률 일정을 그만큼 진행시킨다, 컨테이너 재시작 대비)")
    ap.add_argument("--loss", default="cehinge", choices=("cehinge", "flip"),
                    help="cehinge: CE + margin 2 hinge / flip: 앞 절반 softplus margin(τ 0.2), 뒤 절반 sigmoid(-margin/τ) 로 '뒤집힐 확률' 을 직접 줄인다 (τ 0.15→0.05)")
    args = ap.parse_args()

    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    net = nets()
    _, seg, _ = load_gt()
    widths = tuple(int(v) for v in args.widths.split(",")) if args.widths else None
    if args.film:
        widths = (widths or (16, 24, 32)) + (args.film, len(seg))
    G = make_renderer((args.width, args.fdim, len(seg), (1, 1, 2, 4)) if args.arch == "v2" else widths)
    if args.widen_from:
        sd_old = torch.load(args.widen_from)
        old = Renderer(*(sd_old[f"{k}.0.weight"].shape[0] for k in ("e1", "e2", "e3")))  # 폭은 체크포인트에서 읽는다
        old.load_state_dict(sd_old)
        G = widen_renderer(old, *widths)
    net.segnet.to(memory_format=torch.channels_last)
    if args.resume:
        missing, unexpected = G.load_state_dict(torch.load(args.resume), strict=False)
        assert not unexpected and all(k.startswith(("frame.", "film")) for k in missing), (missing, unexpected)
        if missing:
            print(f"체크포인트에 없는 FiLM 파라미터는 초기값(변조 없음)으로 시작: {missing}")
    print(f"renderer params: {sum(p.numel() for p in G.parameters()):,}")
    opt = torch.optim.Adam(G.parameters(), lr=args.lr)
    steps = args.epochs * (len(seg) // args.bs)
    if args.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=args.lr * 0.02)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    viz = LiveVis("comma vcc · 렌더러 학습", port=args.port).start()
    val_idx = np.arange(5, 600, 25)  # 24장 고정 검증

    step = 0
    t0 = time.time()
    for _ in range(args.start_epoch * (len(seg) // args.bs)):
        sched.step()
        step += 1
    for ep in range(args.start_epoch, args.epochs):
        perm = np.random.default_rng(ep).permutation(len(seg))
        for b in range(0, len(perm) - args.bs + 1, args.bs):
            m = torch.from_numpy(seg[perm[b : b + args.bs]]).long()
            if args.qat:
                img = torch.func.functional_call(G, ste_params(G, args.bits), (m, torch.from_numpy(perm[b : b + args.bs])))
            else:
                img = render(G, m, torch.from_numpy(perm[b : b + args.bs]))
            if args.round:
                img = quantize_ste(img)
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=not args.fp32):
                logits = net.segnet(img)
            logits = logits.float()
            true = logits.gather(1, m[:, None])[:, 0]
            other = logits.scatter(1, m[:, None], -1e4).amax(1)
            margin = true - other
            if args.loss == "flip":
                prog = step / max(steps - 1, 1)
                if prog < 0.5:
                    loss = (F.softplus(-margin / 0.2) * 0.2).mean()
                else:
                    loss = torch.sigmoid(-margin / (0.15 - 0.2 * (prog - 0.5))).mean()
            else:
                # argmax 를 뒤집는 데 직접 관여하는 margin 손실: 정답 logit 이 다른 것보다 2 이상 크게
                loss = F.cross_entropy(logits, m) + F.relu(2.0 - margin).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            if args.project:
                fake_quant_(G, args.bits)
            sched.step()
            step += 1
            with torch.no_grad():
                err = (logits.argmax(1) != m).float().mean().item()
            viz.log(step, loss=loss.item(), train_disagreement_bf16=err, lr=sched.get_last_lr()[0])
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} err {err:.5f} ({time.time() - t0:.0f}s)", flush=True)
        errs = evaluate(G, net, seg, val_idx, bits=args.bits)
        viz.log(step, val_disagreement_fp32=errs.mean().item(), val_seg_term=100 * errs.mean().item())
        with torch.inference_mode():
            m = torch.from_numpy(seg[val_idx[:1]])
            img = render(G, m, torch.from_numpy(val_idx[:1]))
            out = net.segnet(img).argmax(1)
        vis = np.concatenate([img[0].round().clamp(0, 255).permute(1, 2, 0).byte().numpy(), PALETTE[out[0].numpy()]], 1)
        vis[:, SW:][(out[0] != m[0]).numpy()] = [255, 0, 255]
        viz.image("render", png(vis), step=step, caption=f"epoch {ep}: 왼쪽 렌더, 오른쪽 SegNet 결과(분홍=불일치). val {errs.mean():.5f}")
        torch.save(G.state_dict(), args.out)
        print(f"== epoch {ep} val disagreement {errs.mean():.5f} (max {errs.max():.5f})", flush=True)
    if args.full_eval:
        errs = evaluate(G, net, seg, np.arange(len(seg)), bits=args.bits)
        print(f"== 전체 600장 disagreement {errs.mean():.6f} → seg 항 {100 * errs.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
