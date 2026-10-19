"""seg 맵 → 홀수 프레임(512x384 RGB) 렌더러 학습.

SegNet(render(M)) 의 argmax 가 M 과 같아지도록 학습한다. 렌더러 가중치는 archive 에 들어가므로 작게 유지한다
(--qat: 고정 비트 가짜 양자화, --self-compress: 채널별 비트·스케일 학습 = selfcomp.py).

    python renderer.py --widths 24,32,40 --epochs 20 --out ../cache/renderer.pt
결과: --out (inflate 와 같은 float 가중치), self-compression 이면 --out.q (archive 에 넣을 pack_state 바이트) 와
      --out.sc.pt (이어서 학습용 float 가중치 + 비트/스케일) 도 쓴다.
"""

from __future__ import annotations

import argparse
import os
import io
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import CACHE, SH, SW, load_gt, nets, parse_widths, quantize_roundtrip, setup_device
import copy

from model import Renderer  # noqa: E402  (제출물 패키지와 같은 정의)

PALETTE = np.array([[64, 64, 64], [230, 230, 230], [42, 120, 214], [12, 163, 12], [208, 59, 59]], np.uint8)


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


def widen_renderer(old: Renderer, c1: int, c2: int, c3: int) -> Renderer:
    """렌더러를 더 넓게 만들되 처음엔 같은 함수가 되도록 (새 채널의 나가는 가중치 = 0).

    새 채널은 기본 초기화된 들어오는 가중치를 가지므로 학습이 시작되면 바로 쓰이기 시작한다.
    """
    o1, o2, o3 = old.cfg
    new = Renderer(c1, c2, c3)
    r = lambda a, b: list(range(a, b))  # noqa: E731
    # 층별 (이전 입력 인덱스 → 새 입력 인덱스)
    in_maps = {
        "e1.0": (r(0, 7), r(0, 7)),
        "e1.2": (r(0, o1), r(0, o1)),
        "e2.0": (r(0, o1), r(0, o1)),
        "e2.2": (r(0, o2), r(0, o2)),
        "e3.0": (r(0, o2), r(0, o2)),
        "e3.2": (r(0, o3), r(0, o3)),
        "d2.0": (r(0, o3 + o2), r(0, o3) + r(c3, c3 + o2)),
        "d1.0": (r(0, o2 + o1), r(0, o2) + r(c2, c2 + o1)),
        "out": (r(0, o1), r(0, o1)),
    }
    olds, news = dict(old.named_modules()), dict(new.named_modules())
    with torch.no_grad():
        for name, (src_in, dst_in) in in_maps.items():
            a, b = olds[name], news[name]
            n_out = a.weight.shape[0]
            new_cols = sorted(set(range(b.weight.shape[1])) - set(dst_in))
            b.weight[:n_out, new_cols] = 0  # 기존 출력은 새 입력을 보지 않는다
            b.weight[:n_out][:, dst_in] = a.weight[:, src_in]
            b.bias[:n_out] = a.bias
    return new


@torch.inference_mode()
def evaluate(G, net, seg, idx, bs=8, bits=8, sd=None):
    """inflate 와 같은 조건: 저장/복원 왕복 가중치, float 출력 (서브픽셀 확장으로 거의 그대로 전달된다).

    sd 를 주면 그 (이미 양자화된) 가중치를 그대로 쓴다 (self-compression 의 export).
    """
    Gq = copy.deepcopy(G)
    if sd is None:
        quantize_roundtrip(Gq, bits)
    else:
        Gq.load_state_dict(sd)
    dev = next(Gq.parameters()).device
    errs = []
    for i in range(0, len(idx), bs):
        m = torch.from_numpy(seg[idx[i : i + bs]]).to(dev)
        out = net.segnet(Gq(m)).argmax(1)
        errs.append((out != m).float().mean((1, 2)).cpu())
    return torch.cat(errs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cpu", help="cpu | cuda (Colab GPU)")
    ap.add_argument("--limit", type=int, default=0, help="> 0 이면 앞 N장만 (드라이버 점검용)")
    ap.add_argument("--self-compress", type=float, default=0.0,
                    help="> 0 이면 채널별 비트 수를 학습 (selfcomp.py). 1.0 = 점수 공식에 맞춘 크기/불일치 trade-off (--bits 는 시작 비트)")
    ap.add_argument("--sc-lr", type=float, default=1e-2, help="self-compression 의 비트/스케일 학습률")
    ap.add_argument("--sc-rate", default="range", choices=("range", "laplace"),
                    help="크기 손실: range = 채널별 비트 수 (원 논문) / laplace = 엔트로피 부호 (wcodec) 비트 추정 (selfcomp.rate_bits)")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0, help="학습 순서·무작위 시드 (사이클을 다른 경로로 다시 돌릴 때)")
    ap.add_argument("--flip-start", type=float, default=0.5, help="flip 손실에서 sigmoid 단계가 시작하는 진행률 (앞은 softplus)")
    ap.add_argument("--tau", default="0.15,0.05", help="flip 손실 sigmoid 단계의 τ 시작,끝 (선형)")
    ap.add_argument("--out", default=str(CACHE / "renderer.pt"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--fp32", action="store_true", help="SegNet 을 bf16 대신 fp32 로 (느리지만 평가와 같은 수치)")
    ap.add_argument("--cosine", action="store_true", help="OneCycle 대신 cosine 감쇠 (이어서 학습할 때)")
    ap.add_argument("--widths", default=None, help="폭 c1,c2,c3 (예: 24,32,40, 기본 16,24,32)")
    ap.add_argument("--widen-from", default=None, help="이 체크포인트를 --widths 로 넓혀서 시작 (처음엔 같은 출력)")
    ap.add_argument("--bits", type=int, default=8, help="저장 비트 수 (평가에 반영). self-compression 이면 시작 비트")
    ap.add_argument("--qat", action="store_true", help="STE 가짜 양자화로 학습 (--bits 격자, float 원본 가중치 유지)")
    ap.add_argument("--full-eval", action="store_true", help="끝나고 600장 전체 평가")
    ap.add_argument("--start-epoch", type=int, default=0, help="--resume 와 함께: 이 에폭부터 (학습률 일정을 그만큼 진행시킨다, 컨테이너 재시작 대비)")
    ap.add_argument("--loss", default="cehinge", choices=("cehinge", "flip"),
                    help="cehinge: CE + margin 2 hinge / flip: 앞 절반 softplus margin(τ 0.2), 뒤 절반 sigmoid(-margin/τ) 로 '뒤집힐 확률' 을 직접 줄인다 (τ 0.15→0.05)")
    args = ap.parse_args()

    from livevis import LiveVis

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dev = setup_device(args.device)
    net = nets(dev)
    _, seg, _ = load_gt()
    if args.limit:
        seg = seg[: args.limit]
    widths = parse_widths(args.widths)
    G = Renderer(*widths)
    if args.widen_from:
        sd_old = torch.load(args.widen_from, map_location="cpu")
        old = Renderer(*(sd_old[f"{k}.0.weight"].shape[0] for k in ("e1", "e2", "e3")))  # 폭은 체크포인트에서 읽는다
        old.load_state_dict(sd_old)
        G = widen_renderer(old, *widths)
    net.segnet.to(memory_format=torch.channels_last)
    if args.resume:
        G.load_state_dict(torch.load(args.resume, map_location="cpu"))
    G.to(dev)
    print(f"renderer params: {sum(p.numel() for p in G.parameters()):,} ({dev})")
    sc = None
    groups = [{"params": list(G.parameters()), "lr": args.lr}]
    if args.self_compress > 0:
        from selfcomp import SelfCompress

        sc_state = (args.resume + ".sc.pt") if args.resume else None
        if sc_state and os.path.exists(sc_state):  # self-compression 을 이어서: float 가중치와 비트/스케일
            st = torch.load(sc_state, map_location="cpu")
            G.load_state_dict(st["float"])
            sc = SelfCompress(G, st["init_bits"]).to(dev)
            sc.load_state_dict(st["sc"])
        else:
            sc = SelfCompress(G, float(args.bits)).to(dev)
        # 점수 = 100·불일치 + 25·바이트/원본 → 손실(≈불일치) 단위로 가중치당 1비트의 값 (xz 가 비트 수의 약 0.75 로 줄인다)
        gamma = args.self_compress * 0.75 * 25 * sc.total / 8 / 37_545_489 / 100
        if args.sc_rate == "laplace":  # 손실(≈불일치) 단위로 1비트의 값: 25/8/원본/100
            gamma = args.self_compress * 25 / 8 / 37_545_489 / 100
        groups.append({"params": list(sc.parameters()), "lr": args.sc_lr})
        print(f"self-compression: γ {gamma:.3g} (가중치 {sc.total:,}), {sc.summary()}", flush=True)
    opt = torch.optim.Adam(groups)
    steps = args.epochs * (len(seg) // args.bs)
    if args.cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=args.lr * 0.02)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    viz = LiveVis("comma vcc · 렌더러 학습", port=args.port).start()
    val_idx = np.arange(5, len(seg), 25) if len(seg) > 25 else np.arange(len(seg))  # 24장 고정 검증

    step = 0
    t0 = time.time()
    for _ in range(args.start_epoch * (len(seg) // args.bs)):
        sched.step()
        step += 1
    for ep in range(args.start_epoch, args.epochs):
        perm = np.random.default_rng(ep + 1000 * args.seed).permutation(len(seg))
        for b in range(0, len(perm) - args.bs + 1, args.bs):
            m = torch.from_numpy(seg[perm[b : b + args.bs]]).long().to(dev)
            if sc is not None:
                img = torch.func.functional_call(G, sc.params(G), (m,))
            elif args.qat:
                img = torch.func.functional_call(G, ste_params(G, args.bits), (m,))
            else:
                img = G(m)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=not args.fp32):
                logits = net.segnet(img)
            logits = logits.float()
            true = logits.gather(1, m[:, None])[:, 0]
            other = logits.scatter(1, m[:, None], -1e4).amax(1)
            margin = true - other
            if args.loss == "flip":
                prog = step / max(steps - 1, 1)
                if prog < args.flip_start:
                    loss = (F.softplus(-margin / 0.2) * 0.2).mean()
                else:
                    t0_, t1_ = (float(v) for v in args.tau.split(","))
                    frac = (prog - args.flip_start) / max(1 - args.flip_start, 1e-6)
                    loss = torch.sigmoid(-margin / (t0_ + (t1_ - t0_) * frac)).mean()
            else:
                # argmax 를 뒤집는 데 직접 관여하는 margin 손실: 정답 logit 이 다른 것보다 2 이상 크게
                loss = F.cross_entropy(logits, m) + F.relu(2.0 - margin).mean()
            if sc is not None:
                loss = loss + gamma * (sc.rate_bits(G) if args.sc_rate == "laplace" else sc.bits_per_weight())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            with torch.no_grad():
                err = (logits.argmax(1) != m).float().mean().item()
            viz.log(step, loss=loss.item(), train_disagreement_bf16=err, lr=sched.get_last_lr()[0])
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} err {err:.5f} ({time.time() - t0:.0f}s)", flush=True)
        sd_exp = None
        if sc is not None:
            sd_exp, packed = sc.export(G)
            import archive
            import wcodec

            nbytes = len(archive.pack_renderer(G.cfg, wcodec.ec_pack(packed, G.state_dict())))
            with torch.no_grad():
                est = sc.rate_bits(G).item() / 8
            viz.log(step, renderer_bytes=nbytes, bits_per_weight=sc.bits_per_weight().item())
            print(f"   self-compression: 렌더러 {nbytes:,} B (추정 {est:,.0f} B), {sc.summary()}", flush=True)
        errs = evaluate(G, net, seg, val_idx, bits=args.bits, bs=16 if dev.type == "cuda" else 8, sd=sd_exp)
        viz.log(step, val_disagreement_fp32=errs.mean().item(), val_seg_term=100 * errs.mean().item())
        with torch.inference_mode():
            m = torch.from_numpy(seg[val_idx[:1]]).to(dev)
            img = G(m)
            out = net.segnet(img).argmax(1)
            img, out, m = img.cpu(), out.cpu(), m.cpu()
        vis = np.concatenate([img[0].round().clamp(0, 255).permute(1, 2, 0).byte().numpy(), PALETTE[out[0].numpy()]], 1)
        vis[:, SW:][(out[0] != m[0]).numpy()] = [255, 0, 255]
        viz.image("render", png(vis), step=step, caption=f"epoch {ep}: 왼쪽 렌더, 오른쪽 SegNet 결과(분홍=불일치). val {errs.mean():.5f}")
        if sc is not None:  # out = inflate 와 같은 (양자화된) 가중치, .q = archive 에 넣을 바이트, .sc.pt = 이어서 학습용
            torch.save(sd_exp, args.out)
            open(args.out + ".q", "wb").write(packed)
            torch.save({"float": {k: v.detach().cpu() for k, v in G.state_dict().items()},
                        "sc": {k: v.detach().cpu() for k, v in sc.state_dict().items()}, "init_bits": float(args.bits)}, args.out + ".sc.pt")
        else:
            torch.save({k: v.detach().cpu() for k, v in G.state_dict().items()}, args.out)
        print(f"== epoch {ep} val disagreement {errs.mean():.5f} (max {errs.max():.5f})", flush=True)
    if args.full_eval:
        errs = evaluate(G, net, seg, np.arange(len(seg)), bits=args.bits, bs=16 if dev.type == "cuda" else 8, sd=sd_exp)
        print(f"== 전체 600장 disagreement {errs.mean():.6f} → seg 항 {100 * errs.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
