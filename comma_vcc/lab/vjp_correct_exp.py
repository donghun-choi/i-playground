"""carrier 결과 위에 VJP 보정: e1 = e0 + J(e0)^T a,  a = (J J^T)^{-1} (target - pose(e0)).

inflate 때는 (a · pose) 를 짝수 프레임으로 한 번 미분하면 J^T a 가 나온다 → 쌍마다 a (6개) 만 저장.
32쌍에서 보정 전/후 posenet_dist (float, 반올림 없음 = 서브픽셀 확장 가정) 와 a 의 양자화 영향을 본다.
"""

import numpy as np
import torch

from common import CACHE, load_gt, nets, pose_out
import archive  # noqa: E402
from model import even_frames, make_renderer, quantize_roundtrip, render  # noqa: E402

torch.set_num_threads(1)
net = nets()
_, seg, pose = load_gt()
idx = np.linspace(0, 599, 32).astype(int)
G = make_renderer(None)
G.load_state_dict(torch.load(CACHE / "renderer_v1.pt"))
quantize_roundtrip(G)
G.eval()
car = archive.unpack_carrier(open(CACHE / "carrier_v1.bin", "rb").read())
with torch.inference_mode():
    odd = torch.cat([render(G, torch.from_numpy(seg[i : i + 1]), torch.tensor([i])) for i in idx])
    e0 = even_frames(odd, car["c"][idx], car["B"], car["base"])
t = torch.from_numpy(pose[idx])


def jac(e):
    J = torch.zeros(len(e), 6, *e.shape[1:])
    for i in range(len(e)):
        x = e[i : i + 1].clone().requires_grad_(True)
        out = pose_out(net, x, odd[i : i + 1])[0]
        for d in range(6):
            (g,) = torch.autograd.grad(out[d], x, retain_graph=d < 5)
            J[i, d] = g[0]
    return J


def vjp(e, a):
    """inflate 와 같은 계산: (a · pose) 를 e 로 미분."""
    x = e.clone().requires_grad_(True)
    out = pose_out(net, x, odd)
    (g,) = torch.autograd.grad((out * a).sum(), x)
    return g


def dist(e):
    with torch.inference_mode():
        return ((pose_out(net, e, odd) - t) ** 2).mean(1)


e = e0
print(f"carrier 만: {dist(e).mean():.7f}  차원별 RMS {((pose_out(net, e, odd).detach() - t) ** 2).mean(0).sqrt().numpy().round(4)}")
A = []
for step in range(2):
    J = jac(e).flatten(2)
    r = t - pose_out(net, e, odd).detach()
    a = torch.linalg.solve(J @ J.transpose(1, 2), r[:, :, None])[:, :, 0]
    A.append(a)
    e_new = (e + vjp(e, a)).clamp(0, 255).detach()
    d = dist(e_new)
    print(f"VJP {step + 1}: {d.mean():.7f}  |a| {a.abs().mean():.1f}  |δ| 평균 {(e_new - e).abs().mean():.3f} 최대 {(e_new - e).abs().max():.1f}")
    # a 를 float16 으로 저장하면?
    e16 = (e + vjp(e, a.half().float())).clamp(0, 255).detach()
    print(f"        a float16: {dist(e16).mean():.7f}")
    e = e_new
