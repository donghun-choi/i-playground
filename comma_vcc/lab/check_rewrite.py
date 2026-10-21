"""다시 짠 디코더 (기본 submissions/rewrite) 를 단계마다 golden 기준값과 비교한다. 먼저 `python golden.py dump`.

    python check_rewrite.py                    # 모든 단계
    python check_rewrite.py inputs probs       # 고른 단계만
    python check_rewrite.py --full decode      # seg 맵 601장 전부 (기본은 앞 3장)
    python check_rewrite.py --dir ../submissions/다른이름

단계마다 입력은 golden 값을 넣는다 → 앞 단계가 틀려도 뒤 단계를 따로 확인할 수 있다.
결과: PASS / FAIL (무엇이 다른지) / TODO (NotImplementedError) / ERROR (그 밖의 예외)
참조 구현 (submissions/semantic_cpu) 은 import 하지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VCC = Path(__file__).resolve().parents[1]
GOLDEN = VCC / "cache" / "golden"
MODULES = ("container", "entropy", "ctxnet", "segdec", "render", "pose", "upscale")
STAGES = {}


def stage(f):
    STAGES[f.__name__] = f
    return f


class Fail(Exception):
    pass


def arr(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def nchw(x: np.ndarray) -> torch.Tensor:
    """golden 프레임 (n,384,512,3) → torch (n,3,384,512)"""
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).contiguous()


def compare(name: str, mine, gold: np.ndarray, tol: float = 0.0) -> str:
    """같으면 짧은 설명, 다르면 Fail. tol = 0 이면 값이 정확히 같아야 한다."""
    a = arr(mine)
    if a.shape != gold.shape:
        if a.size != gold.size:
            raise Fail(f"{name}: 모양 {a.shape}, 기준 {gold.shape}")
        a = a.reshape(gold.shape)
    if tol == 0:
        bad = a != gold
        if bad.any():
            idx = tuple(int(i) for i in np.argwhere(bad)[0])
            raise Fail(f"{name}: 다른 원소 {int(bad.sum()):,}/{a.size:,}, 첫 위치 {idx}: {a[idx].item()!r} vs 기준 {gold[idx].item()!r}")
        return f"{name} 같음"
    d = np.abs(a.astype(np.float64) - gold.astype(np.float64))
    if not d.max() <= tol:
        raise Fail(f"{name}: 최대 차이 {d.max():.3g} > {tol:g} (평균 {d.mean():.3g})")
    return f"{name} 최대 차이 {d.max():.2g}"


class Golden:
    """기준값을 필요할 때 읽는다."""

    def __init__(self, full: bool):
        if not (GOLDEN / "sections.json").exists():
            sys.exit(f"{GOLDEN} 가 없다: 먼저 python golden.py dump")
        self.full = full
        self.info = json.loads((GOLDEN / "sections.json").read_text())
        self._cache = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name):
        if name == "p":
            return zipfile.ZipFile(VCC / "submissions" / "semantic_cpu" / "archive.zip").read("p")
        if name == "sec":  # 섹션은 직접 잘라서 (container 단계와 독립)
            p, off, out = self.p, 4, {}
            while off < len(p):
                n = int.from_bytes(p[off + 4 : off + 8], "little")
                out[p[off : off + 4].decode()] = p[off + 8 : off + 8 + n]
                off += 8 + n
            return out
        if name == "layers":
            z = np.load(GOLDEN / "qnet_layers.npz")
            return [(z[f"W{i}"], z[f"B{i}"], z[f"M{i}"], int(z["dil"][i]), bool(z["relu"][i])) for i in range(len(z["dil"]))]
        if name == "seg":
            return np.load(GOLDEN / "seg.npz")["seg"]
        if name in ("masks", "pose", "renderer"):
            return np.load(GOLDEN / f"{name}.npz")
        return np.load(GOLDEN / f"{name}.npy")


# ---------------------------------------------------------------- 단계 (GUIDE 5절 순서)
@stage
def container(M, g):
    sec = M.container.read_sections(g.p)
    want = g.info["sections"]
    if list(sec) != list(want):
        raise Fail(f"섹션 {list(sec)}, 기준 {list(want)}")
    for k, v in want.items():
        if len(sec[k]) != v["bytes"] or hashlib.sha256(sec[k]).hexdigest() != v["sha256"]:
            raise Fail(f"{k}: {len(sec[k]):,} B, 기준 {v['bytes']:,} B (또는 sha256 다름)")
    return " ".join(f"{k} {v['bytes']:,} B" for k, v in want.items())


@stage
def ctxe(M, g):
    qb = M.ctxnet.unpack_ctxe(g.sec["ctxe"])
    if hashlib.sha256(qb).hexdigest() != g.info["qnet_bytes_sha256"]:
        ref = (GOLDEN / "qnet.bin").read_bytes()
        i = next((i for i, (x, y) in enumerate(zip(qb, ref)) if x != y), min(len(qb), len(ref)))
        raise Fail(f"QNet 바이트 {len(qb):,} B (기준 {len(ref):,} B), 처음 다른 오프셋 {i}")
    return f"QNet 바이트 {len(qb):,} B 같음"


@stage
def parse_qnet(M, g):
    layers = M.ctxnet.parse_qnet((GOLDEN / "qnet.bin").read_bytes())
    if len(layers) != len(g.layers):
        raise Fail(f"층 {len(layers)}개, 기준 {len(g.layers)}개")
    for i, (mine, gold) in enumerate(zip(layers, g.layers)):
        for name, a, b in zip(("W", "B", "M"), mine[:3], gold[:3]):
            compare(f"층 {i} {name}", a, b)
        if int(mine[3]) != gold[3] or bool(mine[4]) != gold[4]:
            raise Fail(f"층 {i} (dil, relu) = {mine[3:5]}, 기준 {gold[3:5]}")
    return f"{len(layers)}층 같음"


@stage
def intcnn(M, g):
    net = M.ctxnet.IntCNN(g.layers)
    ok = net.self_check()
    a = compare("무작위 입력 logit", net.run(g.qnet_probe_in), g.qnet_probe_out)
    b = compare("맵 5 레벨 4 A2 logit", net.run(g.inputs_t5_s4_A2), g.logits_t5_s4_A2)
    return f"{a}, {b}, self_check {ok}"


@stage
def masks(M, g):
    for h in (16, 8, 4, 2, 1):
        for kind in ("A1", "A2", "B"):
            kn, tg = M.segdec.masks(h, kind)
            compare(f"h{h} {kind} 아는 칸", np.asarray(kn, bool), g.masks[f"known_h{h}_{kind}"])
            compare(f"h{h} {kind} 보낼 칸", np.asarray(tg, bool), g.masks[f"target_h{h}_{kind}"])
    return "15개 같음"


@stage
def inputs(M, g):
    seg = g.seg
    a = compare("맵 5 레벨 4 A2 입력", M.segdec.build_input(seg[5], 4, "A2", seg[4], seg[3]), g.inputs_t5_s4_A2)
    # 아직 모르는 칸 (stride-2 격자의 아는 칸 밖, 격자 밖 픽셀) 값이 바뀌어도 입력은 같아야 한다
    cur = np.random.default_rng(0).integers(0, 5, seg[5].shape).astype(np.uint8)
    kn = g.masks["known_h2_A2"]
    grid = cur[::2, ::2]
    grid[kn] = seg[5][::2, ::2][kn]
    compare("모르는 칸을 바꾼 입력", M.segdec.build_input(cur, 4, "A2", seg[4], seg[3]), g.inputs_t5_s4_A2)
    b = compare("이전 맵 없는 입력 (비율 채널 0)", arr(M.segdec.build_input(seg[0], 32, "A1", None, None)).reshape(-1, 24, 32)[6:16],
                np.zeros((10, 24, 32), np.float32))
    return f"{a}, 모르는 칸 무시, {b}"


@stage
def probs(M, g):
    rows = np.ascontiguousarray(g.logits_t5_s4_A2[0][:, g.masks["target_h2_A2"]].T)  # (m,5) float32 정수값
    p = M.ctxnet.probs(rows)
    if not (isinstance(p, np.ndarray) and p.dtype == np.float64 and p.flags["C_CONTIGUOUS"]):
        raise Fail(f"C 연속 float64 numpy 여야 한다 (지금 {type(p).__name__} {getattr(p, 'dtype', '')})")
    return compare(f"확률 {len(rows):,}칸", p, g.probs_t5_s4_A2)


@stage
def decode(M, g):
    n = None if g.full else 3
    net = M.ctxnet.IntCNN(g.layers)
    net.self_check()
    maps = M.segdec.decode_maps(g.sec["segs"], net, n_max=n)
    want = g.seg if n is None else g.seg[:n]
    return compare(f"seg 맵 {len(want)}장", maps, want)


@stage
def renderer_weights(M, g):
    widths, ts = M.render.unpack_renderer(g.sec["rend"])
    if tuple(widths) != tuple(g.info["renderer_widths"]):
        raise Fail(f"폭 {widths}, 기준 {g.info['renderer_widths']}")
    names = g.renderer.files
    if len(ts) != len(names):
        raise Fail(f"텐서 {len(ts)}개, 기준 {len(names)}개")
    for t, k in zip(ts, names):
        compare(k, t, g.renderer[k])
    return f"폭 {tuple(widths)}, 텐서 {len(ts)}개 같음"


@stage
def renderer_forward(M, g):
    G = M.render.load_renderer(g.sec["rend"])
    a = compare("홀수 프레임 0..3", arr(M.render.render_all(G, g.seg[1:5])).transpose(0, 2, 3, 1), g.odd_0_3, tol=1e-3)
    b = compare("렌더러(M_pre)", arr(M.render.render_all(G, g.seg[:1])).transpose(0, 2, 3, 1), g.prev0, tol=1e-3)
    return f"{a}, {b}"


@stage
def pose4(M, g):
    B, c, a = M.pose.parse_pose4(g.sec["pos4"])
    return ", ".join(compare(k, v, g.pose[k]) for k, v in (("B", B), ("c", c), ("a", a)))


@stage
def even(M, g):
    prev = torch.cat([nchw(g.prev0), nchw(g.odd_0_3[:3])])
    B, c, a = (torch.from_numpy(g.pose[k]) for k in ("B", "c", "a"))
    with torch.inference_mode():
        e = M.pose.even_frames(prev, a[:4], c[:4], B)
    return compare("짝수 프레임 0..3", arr(e).transpose(0, 2, 3, 1), g.even_0_3, tol=1e-3)


@stage
def expand(M, g):
    x = np.stack([g.even_0_3[0], g.odd_0_3[0]])
    out = arr(M.upscale.expand_fine(x))
    if out.shape != g.raw_pair0.shape or out.dtype != np.uint8:
        raise Fail(f"모양/자료형 {out.shape} {out.dtype}, 기준 {g.raw_pair0.shape} uint8")
    diff = int((out != g.raw_pair0).sum())
    down = F.interpolate(torch.from_numpy(out).permute(0, 3, 1, 2).float(), size=(384, 512), mode="bilinear")
    ref = F.interpolate(torch.from_numpy(g.raw_pair0).permute(0, 3, 1, 2).float(), size=(384, 512), mode="bilinear")
    dd = (down - ref).abs().max().item()
    if dd > 1e-4:
        raise Fail(f"평가 축소 결과가 다르다 (최대 {dd:.3g}), 다른 원소 {diff:,}")
    err = np.abs(down.permute(0, 2, 3, 1).numpy() - x).mean()
    same = "원소 전부 같음" if diff == 0 else f"2x2 값 {diff:,}개 다르지만 축소 결과 같음 (같은 합 조합, 점수 같음)"
    return f"{same}, 축소 오차 평균 {err:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stages", nargs="*", help=f"단계 (기본 전부): {' '.join(STAGES)}")
    ap.add_argument("--dir", default=str(VCC / "submissions" / "rewrite"))
    ap.add_argument("--full", action="store_true", help="decode 단계에서 601장 전부")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    unknown = [s for s in args.stages if s not in STAGES]
    if unknown:
        sys.exit(f"모르는 단계 {unknown}. 단계: {' '.join(STAGES)}")
    sys.path.insert(0, str(Path(args.dir).resolve()))

    class Mods:
        pass

    M = Mods()
    for name in MODULES:
        try:
            setattr(M, name, importlib.import_module(name))
        except Exception as e:  # 문법 오류 등: 그 모듈을 쓰는 단계만 ERROR
            print(f"[import 실패] {name}: {type(e).__name__}: {e}")
    g = Golden(args.full)
    counts = {}
    for name in args.stages or STAGES:
        t = time.time()
        try:
            msg, res = STAGES[name](M, g), "PASS"
        except NotImplementedError as e:
            msg, res = str(e) or "TODO", "TODO"
        except Fail as e:
            msg, res = str(e), "FAIL"
        except Exception as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            msg, res = f"{type(e).__name__}: {e}  ({Path(tb.filename).name}:{tb.lineno})", "ERROR"
        counts[res] = counts.get(res, 0) + 1
        print(f"{res:5} {name:17} {msg}  ({time.time() - t:.1f}s)", flush=True)
    print(" ".join(f"{k} {v}" for k, v in counts.items()))
    sys.exit(0 if set(counts) == {"PASS"} else 1)


if __name__ == "__main__":
    main()
