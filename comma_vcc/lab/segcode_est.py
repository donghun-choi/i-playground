"""seg 맵 무손실 부호화: 문맥 모델별 이상적 부호 길이 추정 (실제 산술부호화 전 단계).

계층적 순서: stride 32 격자 → 각 레벨에서 (A) 칸 중심, (B) 변 중점 순으로 채운다.
한 패스 안의 픽셀은 서로 독립이라 디코드도 패스 단위로 벡터화된다.
확률은 문맥별 적응형 정수 카운트 (기계마다 결과가 같도록 정수만 쓴다).
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from common import SH, SW, load_gt

S0 = 32
K = 5


def passes():
    """[(이름, 위치 flat index, 이웃 4개 flat index (n,4))] 순서대로."""
    out = []
    ys, xs = np.mgrid[0:SH:S0, 0:SW:S0]
    out.append(("coarse", (ys * SW + xs).ravel(), None))
    s = S0
    while s > 1:
        h = s // 2
        # A: 중심 (y+h, x+h), 이웃은 대각 네 꼭짓점
        ys, xs = np.mgrid[h:SH:s, h:SW:s]
        ys, xs = ys.ravel(), xs.ravel()
        nb = np.stack([
            (ys - h) * SW + (xs - h),
            (ys - h) * SW + np.minimum(xs + h, SW - s + 0),
            np.minimum(ys + h, SH - s) * SW + (xs - h),
            np.minimum(ys + h, SH - s) * SW + np.minimum(xs + h, SW - s),
        ], 1)
        out.append((f"A{s}", ys * SW + xs, nb))
        # B: 변 중점. (y, x+h) 와 (y+h, x). 이웃은 위/아래/왼/오른쪽 h 거리
        ya, xa = np.mgrid[0:SH:s, h:SW:s]
        yb, xb = np.mgrid[h:SH:s, 0:SW:s]
        ys = np.concatenate([ya.ravel(), yb.ravel()])
        xs = np.concatenate([xa.ravel(), xb.ravel()])

        def at(y, x):
            # 경계 밖이면 반대편 이웃을 쓴다 (이미 알고 있는 칸)
            return y, x

        up = np.where(ys - h >= 0, ys - h, ys + h)
        dn = np.where(ys + h < SH, ys + h, ys - h)
        lf = np.where(xs - h >= 0, xs - h, xs + h)
        rt = np.where(xs + h < SW, xs + h, xs - h)
        nb = np.stack([up * SW + xs, dn * SW + xs, ys * SW + lf, ys * SW + rt], 1)
        out.append((f"B{s}", ys * SW + xs, nb))
        s = h
    return out


def check_order(ps):
    known = np.zeros(SH * SW, bool)
    for name, pos, nb in ps:
        if nb is not None:
            assert known[nb].all(), f"{name}: 아직 모르는 이웃 사용"
        assert not known[pos].any(), f"{name}: 중복"
        known[pos] = True
    assert known.all()


class Counts:
    def __init__(self, n_ctx: int, alpha: int = 1, scale: int = 16):
        # 카운트는 정수. 관측 1회 = scale, 초기값 alpha. (나중에 산술부호화에 그대로 쓴다)
        self.c = np.full((n_ctx, K), alpha, np.int64)
        self.scale = scale

    def bits(self, ctx, sym):
        c = self.c[ctx]
        return -np.log2(c[np.arange(len(sym)), sym] / c.sum(1))

    def update(self, ctx, sym):
        np.add.at(self.c, (ctx, sym), self.scale)


def run(seg, model: str, verbose=True):
    ps = passes()
    check_order(ps)
    # 문맥: 패스 종류 × 이웃 4개 × 이전 프레임 값 (+ 옵션)
    n_prev = 6  # 0..4, 5 = 이전 프레임 없음
    extra = {"base": 1, "prevnb": 2}[model]
    tables = [Counts(K**4 * n_prev * extra if nb is not None else n_prev * 9) for _, _, nb in ps]
    total = 0.0
    per_pass = np.zeros(len(ps))
    t0 = time.time()
    prev = None
    for t in range(len(seg)):
        m = seg[t].ravel().astype(np.int64)
        for k, (name, pos, nb) in enumerate(ps):
            sym = m[pos]
            pv = prev[pos] if prev is not None else np.full(len(pos), 5)
            if nb is None:
                # coarse: 이전 프레임 값 + 이전 프레임 3x3 (stride 32 격자에서) 이 균일한지
                ctx = pv * 9
            else:
                n = m[nb]
                ctx = ((n[:, 0] * K + n[:, 1]) * K + n[:, 2]) * K + n[:, 3]
                ctx = ctx * n_prev + pv
                if model == "prevnb":
                    # 이전 프레임에서 같은 이웃 위치 값이 지금 이웃과 모두 같은가 (움직임이 없는 곳)
                    same = (prev[nb] == n).all(1) if prev is not None else np.zeros(len(pos), bool)
                    ctx = ctx * 2 + same
            b = tables[k].bits(ctx, sym)
            per_pass[k] += b.sum()
            tables[k].update(ctx, sym)
        prev = m
        total = per_pass.sum()
        if verbose and (t + 1) % 100 == 0:
            print(f"  {model} {t + 1}/{len(seg)}: {total / 8 / (t + 1):.0f} B/frame ({time.time() - t0:.0f}s)", flush=True)
    return total / 8, {name: per_pass[k] / 8 for k, (name, _, _) in enumerate(ps)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="base,prevnb")
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()
    _, seg, _ = load_gt()
    seg = seg[: args.frames]
    for model in args.models.split(","):
        nbytes, br = run(seg, model)
        print(f"{model}: {nbytes:,.0f} bytes ({nbytes / len(seg):.0f} B/frame)")
        print("   " + " ".join(f"{k}:{v / 1000:.1f}K" for k, v in br.items()))
