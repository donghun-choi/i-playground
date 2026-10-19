"""제출물 코덱의 형식 단위 테스트 (캐시·평가 네트워크 없이 몇 초).

    cd comma_vcc/lab && ../.venv/bin/python -m unittest test_codec -v
다시 짠 디코더도 같은 테스트를 통과해야 한다 (모듈 이름만 바꿔서).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "submissions" / "semantic_cpu"))

import archive  # noqa: E402
import segcodec as sc  # noqa: E402
import wcodec  # noqa: E402
from model import SH, SW, Renderer, expand_fine, pack_state, unpack_state  # noqa: E402


def random_qnet(seed=0, ch=8):
    rng = np.random.default_rng(seed)
    layers = []
    for c_in, c_out, k, dil, relu in ((sc.C_IN, ch, 3, 1, True), (ch, ch, 3, 2, True), (ch, sc.K, 1, 1, False)):
        W = rng.integers(-3, 4, (c_out, c_in, k, k))
        B = rng.integers(-200, 200, c_out)
        M = rng.integers(2000, 6000, c_out)
        layers.append((W, B, M, dil, relu))
    return sc.QNet(layers)


def random_maps(n, seed=0):
    """매끈한 무작위 맵 (경계가 있는 덩어리) + 프레임마다 조금씩 이동."""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(1, 1, 12, 16, generator=g)
    out = []
    for t in range(n):
        f = F.interpolate(torch.roll(base, t, 3), size=(SH, SW), mode="bilinear", align_corners=False)
        out.append((f[0, 0] * 5).clamp(0, 4.999).floor().to(torch.uint8))
    return torch.stack(out).numpy()


class TestPasses(unittest.TestCase):
    def test_partition(self):
        """레벨마다 A1·A2·B 의 보낼 칸이 겹치지 않고, 아는 칸 (= 이전 레벨까지) 과 합치면 격자 전체가 된다."""
        for h in sc.HS:
            known0, _ = sc.masks(h, "A1")
            seen = known0.copy()
            for kind in sc.PASSES:
                kn, tg = sc.masks(h, kind)
                self.assertTrue(np.array_equal(kn, seen), (h, kind))  # 그 패스가 아는 칸 = 지금까지 보낸 칸
                self.assertFalse((seen & tg).any())
                seen |= tg
            self.assertTrue(seen.all())
            i, j = np.mgrid[0 : SH // h, 0 : SW // h]
            self.assertTrue(np.array_equal(known0, (i % 2 == 0) & (j % 2 == 0)))  # 이전 레벨 격자 = (짝,짝)

    def test_inputs_layout(self):
        fi = sc.Inputs()
        m = random_maps(3)
        cur = torch.from_numpy(m[2:3])
        x = fi.build(cur, 8, "A2", fi.fracs(torch.from_numpy(m[1:2])), fi.fracs(None))
        self.assertEqual(tuple(x.shape), (1, sc.C_IN, SH // 4, SW // 4))
        self.assertTrue(torch.equal(x, x.round()) and x.min() >= 0 and x.max() <= sc.Q_IN)
        self.assertTrue((x[0, 11:16] == 0).all())  # 전전 맵 없음
        self.assertTrue((x[0, 16] == 64).all() and (x[0, 22] == 64).all() and (x[0, 17 + sc.LEVELS.index(8)] == 64).all())
        # 이전 맵 비율: 블록 안 클래스 개수 × 64 / h² 의 내림 (h = 4)
        oh = (m[1, :4, :4, None] == np.arange(5)).sum((0, 1))
        self.assertTrue(np.array_equal(x[0, 6:11, 0, 0].numpy(), np.floor(oh * 64 / 16)))


class TestQNet(unittest.TestCase):
    def test_bytes_roundtrip(self):
        q = random_qnet()
        q2, off = sc.QNet.from_bytes(q.to_bytes())
        self.assertEqual(off, len(q.to_bytes()))
        self.assertEqual(q2.to_bytes(), q.to_bytes())
        self.assertEqual(q.to_bytes()[1 + 6] & sc.FLAGS, sc.FLAGS)

    def test_arithmetic(self):
        """run() = 정수 합성곱 → float32 (acc + B) → × float32(M·2^-16) → floor (→ clamp) 를 손으로 계산한 것."""
        q = random_qnet(1)
        x = torch.from_numpy(np.random.default_rng(1).integers(0, 65, (1, sc.C_IN, 24, 32))).float()
        y = x.double()
        for W, B, M, dil, relu in q.layers:
            acc = F.conv2d(y, torch.from_numpy(W).double(), padding=dil * (W.shape[-1] // 2), dilation=dil)
            t = (acc.float() + torch.from_numpy(B.astype(np.float32)).view(1, -1, 1, 1))
            t = (t * torch.from_numpy((M * 2.0**-16).astype(np.float32)).view(1, -1, 1, 1)).floor()
            y = (t.clamp(0, 255) if relu else t).double()
        self.assertTrue(torch.equal(q.run(x), y.float()))
        self.assertTrue(q.self_check())

    def test_probs(self):
        p = sc.probs(torch.tensor([[0.0, -16, -32, -300, 5]]))
        f = np.array([sc.FREQ[5].item(), sc.FREQ[21].item(), sc.FREQ[37].item(), 1, 2**20])
        self.assertTrue(np.array_equal(p, (f / f.sum())[None]))


class TestStreams(unittest.TestCase):
    def test_seg_roundtrip(self):
        seg = random_maps(4)
        q = random_qnet(2)
        buf = sc.encode(seg, q, chunk=3)
        self.assertEqual(int(np.frombuffer(buf[:4], np.uint32)[0]), 4)
        self.assertTrue(np.array_equal(sc.decode(buf, q), seg))

    def test_rice(self):
        rows = np.random.default_rng(0).integers(-300, 300, (5, 50))
        self.assertTrue(np.array_equal(archive.rice_decode(archive.rice_encode(rows), 5, 50), rows))

    def test_pose4(self):
        rng = np.random.default_rng(0)
        B_q, B_scale = rng.integers(-15, 16, (8, 1, 24, 32)), rng.random(8).astype(np.float32)
        c_int, a_int = rng.integers(-128, 128, (30, 8)), rng.integers(-128, 128, (30, 6))
        c_step, a_step = rng.random(8).astype(np.float32), rng.random(6).astype(np.float32)
        pos = archive.unpack_pose4(archive.pack_pose4(B_q, B_scale, c_int, c_step, a_int, a_step))
        self.assertTrue(torch.equal(pos["B"], torch.from_numpy(B_q.astype(np.float32) * B_scale.reshape(8, 1, 1, 1))))
        self.assertTrue(torch.equal(pos["c"], torch.from_numpy((c_int * c_step.astype(np.float64)).astype(np.float32))))
        self.assertEqual(tuple(pos["a"].shape), (30, 6))

    def test_renderer_weights(self):
        G = Renderer(8, 8, 8)
        buf = pack_state(G.state_dict(), 4)
        tmpl = G.state_dict()
        ec = wcodec.ec_pack(buf, tmpl)
        self.assertEqual(wcodec.ec_unpack(ec, tmpl), buf)
        cfg, ec2 = archive.unpack_renderer(archive.pack_renderer(G.cfg, ec))
        self.assertEqual((cfg, ec2), ((8, 8, 8), ec))
        sd = unpack_state(buf, tmpl)
        self.assertLess(max((sd[k] - v).abs().max().item() for k, v in tmpl.items()), 0.1)

    def test_qnet_ec(self):
        q = random_qnet(3)
        self.assertEqual(wcodec.ec_unpack_qnet(wcodec.ec_pack_qnet(q.to_bytes())), q.to_bytes())


class TestExpand(unittest.TestCase):
    def test_expand_fine(self):
        """평가 코드처럼 bilinear 축소하면 거의 원래 float 값 (균일 무작위 값에서 평균 0.013, 실제 프레임은 0.0035)."""
        x = (torch.rand(1, SH, SW, 3) * 255).numpy()
        full = torch.from_numpy(expand_fine(x)).permute(0, 3, 1, 2).float()
        back = F.interpolate(full, size=(SH, SW), mode="bilinear").permute(0, 2, 3, 1).numpy()
        self.assertLess(np.abs(back - x).mean(), 0.02)
        self.assertLessEqual(np.abs(back - x).max(), 0.5 + 1e-3)  # 탭이 한쪽에 몰린 위치는 정수 반올림 수준


if __name__ == "__main__":
    unittest.main()
