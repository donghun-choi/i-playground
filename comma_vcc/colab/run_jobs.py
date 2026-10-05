"""Colab GPU 작업 드라이버: 렌더러 장기 학습 사이클 → pose 재피팅, 결과를 colab-results 브랜치로 push.

결과는 comma_vcc/colab/results/ 에 쌓인다 (summary.json, renderer_best.pt, pose2_best.bin, logs/).
런타임이 끊겨도 다시 실행하면 summary.json 을 보고 이어서 한다.

    python comma_vcc/colab/run_jobs.py --jobs renderer,pose --cycles 6 --push
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VCC = HERE.parent
ROOT = VCC.parent
LAB = VCC / "lab"
CACHE = VCC / "cache"
SEED = HERE / "seed"
RES = HERE / "results"
LOGS = RES / "logs"
SUMMARY = RES / "summary.json"


def load_summary() -> dict:
    return json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}


def save_summary(s: dict) -> None:
    SUMMARY.write_text(json.dumps(s, indent=2, ensure_ascii=False))


def run(cmd: list[str], log: Path) -> str:
    """lab/ 에서 실행, 출력은 화면과 로그 파일에 동시에. 실패하면 예외."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    out = []
    with open(log, "a") as f:
        p = subprocess.Popen(cmd, cwd=LAB, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
            f.flush()
            out.append(line)
        if p.wait():
            raise RuntimeError(f"실패 (종료 코드 {p.returncode}): {log}")
    return "".join(out)


def push(args, msg: str) -> None:
    """results/ 만 커밋해서 colab-results 브랜치로 push (커밋 작성자는 노트북에서 설정)."""
    if not args.push:
        return
    subprocess.run(["git", "add", "-A", str(RES)], cwd=ROOT, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        return
    subprocess.run(["git", "commit", "-q", "-m", msg], cwd=ROOT, check=True)
    for i in range(4):
        if subprocess.run(["git", "push", "-q", "origin", "HEAD:colab-results"], cwd=ROOT).returncode == 0:
            print(f"push 완료: {msg}", flush=True)
            return
        time.sleep(2 ** (i + 1))
    print("push 실패 (결과는 로컬 results/ 에 남아 있다)", flush=True)


RENDERER_TRACKS = ("renderer", "renderer3", "renderer_sc")


def renderer_bytes(path: Path, widths: str, bits: int) -> int:
    """archive 에 들어갈 렌더러 섹션 크기 (inflate 와 같은 포장). bits 0 = self-compression (.q 바이트)."""
    sys.path[:0] = [str(VCC / "submissions" / "semantic_cpu")]
    import torch
    import archive
    from model import pack_state

    cfg = tuple(int(v) for v in widths.split(","))
    if bits == 0:
        return len(archive.pack_renderer(cfg, Path(str(path) + ".q").read_bytes()))
    return len(archive.pack_renderer(cfg, pack_state(torch.load(path, map_location="cpu"), bits)))


def seg_plus_rate(d: float, nbytes: int) -> float:
    return 100 * d + 25 * nbytes / 37_545_489


def copy_renderer(src: Path, dst: Path) -> None:
    """체크포인트와 함께 다니는 파일 (.q 바이트, .sc.pt 학습 상태) 까지 복사."""
    shutil.copy(src, dst)
    for ext in (".q", ".sc.pt"):
        if Path(str(src) + ext).exists():
            shutil.copy(str(src) + ext, str(dst) + ext)


def renderer_job(args, s: dict, track="renderer", bits=None, cycles=None, start: Path | None = None) -> None:
    """flip 손실 사이클을 반복. 사이클마다 600장 전체 평가, seg 항 + 렌더러 rate 가 가장 좋은 것을 이어서 학습한다.

    renderer: --bits(4) QAT, 시드에서 / renderer3: 3비트 QAT, 4비트 최고에서 /
    renderer_sc: self-compression (채널별 비트 학습, 시작 4비트), 4비트 최고에서. 저장 비트 0 = .q 바이트 사용.
    """
    bits = args.bits if bits is None else bits
    cycles = cycles or args.cycles
    sc = track == "renderer_sc"
    tr = s.setdefault(track, {"widths": args.widths, "bits": 0 if sc else bits, "cycles": []})
    best_file = RES / f"{track}_best.pt"
    if not best_file.exists():
        copy_renderer(start or SEED / "renderer_seed.pt", best_file)
        seed_d = s.get("seed", {}).get("renderer_disagreement") if track == "renderer" else None
        tr["best"] = {"cycle": 0, "disagreement": seed_d,
                      "score": seg_plus_rate(seed_d, renderer_bytes(best_file, args.widths, bits)) if seed_d else None}
    if tr.get("best", {}).get("score") is None and tr.get("best", {}).get("disagreement") is not None:  # 예전 요약 (score 없음)
        tr["best"]["score"] = round(seg_plus_rate(tr["best"]["disagreement"], renderer_bytes(best_file, args.widths, tr["bits"])), 5)
    for c in range(len(tr["cycles"]) + 1, cycles + 1):
        t = time.time()
        out = CACHE / f"gpu_{track}_c{c}.pt"
        quant = ["--bits", "4", "--self-compress", str(args.sc_weight)] if sc else ["--bits", str(bits), "--qat"]
        txt = run([sys.executable, "renderer.py", "--device", args.device, "--widths", args.widths, "--resume", str(best_file),
                   "--epochs", str(args.epochs), "--lr", str(args.lr), "--bs", str(args.bs), "--cosine", "--fp32", *quant,
                   "--loss", "flip", "--full-eval", "--threads", "2", "--port", str(args.renderer_port),
                   "--out", str(out), *args.limit], LOGS / f"{track}_c{c}.log")
        d = float(re.findall(r"전체 600장 disagreement ([0-9.]+)", txt)[-1])
        nb = renderer_bytes(out, args.widths, tr["bits"])
        rec = {"cycle": c, "disagreement": d, "seg_term": 100 * d, "bytes": nb, "score": round(seg_plus_rate(d, nb), 5),
               "minutes": round((time.time() - t) / 60, 1)}
        tr["cycles"].append(rec)
        prev = tr.get("best", {}).get("score")
        if prev is None or rec["score"] < prev:
            copy_renderer(out, best_file)
            tr["best"] = rec
        save_summary(s)
        print(f"== {track} 사이클 {c}: 600장 불일치 {d:.6f}, {nb:,} B → seg+rate {rec['score']:.4f} (최고 {tr['best'].get('score')}) {rec['minutes']}분", flush=True)
        push(args, f"colab: {track} 사이클 {c} 불일치 {d:.6f} {nb} B")


def pick_renderer(args, s: dict):
    """seg 항 + 렌더러 rate 항이 가장 작은 렌더러 (트랙별 최고 중에서)."""
    cands = []
    for track in RENDERER_TRACKS:
        f = RES / f"{track}_best.pt"
        tr = s.get(track, {})
        d = tr.get("best", {}).get("disagreement")
        if f.exists() and d is not None:
            nb = renderer_bytes(f, args.widths, tr["bits"])
            cands.append((seg_plus_rate(d, nb), f, tr["bits"], d, nb))
    if not cands:
        return SEED / "renderer_seed.pt", args.bits
    cands.sort(key=lambda x: x[0])
    for sc_, f, b, d, nb in cands:
        print(f"후보 {f.name}: {b or '채널별'}비트, 불일치 {d:.6f}, {nb:,} B → seg+렌더러 rate {sc_:.4f}", flush=True)
    s["renderer_choice"] = {"file": cands[0][1].name, "bits": cands[0][2], "disagreement": cands[0][3], "bytes": cands[0][4],
                            "seg_plus_rate": round(cands[0][0], 5)}
    return cands[0][1], cands[0][2]


def pose_bytes(path: Path) -> int:
    """archive 에 들어갈 pose 섹션 (pos3) 크기."""
    sys.path[:0] = [str(VCC / "submissions" / "semantic_cpu")]
    import archive

    return len(archive.pose2_to_pose3(path.read_bytes()))


def pose_job(args, s: dict) -> None:
    """고른 렌더러에 맞춰 pose 재피팅 (기저 해상도 변형마다): 정규화 손실 + 기저 학습 → 일반 MSE → 낮은 학습률 다듬기.

    변형 중 pose 항 + pose 섹션 rate 가 가장 작은 것을 pose2_best.bin 으로.
    """
    renderer, rbits = pick_renderer(args, s)
    start = SEED / "pose2_seed.bin"
    common = ["--device", args.device, "--renderer", str(renderer), "--renderer-cfg", args.widths, "--rbits", str(rbits),
              "--cbits", str(args.pose_cbits), "--q-epochs", str(args.q_epochs), "--greedy-rounds", "1", "--threads", "2", "--port", str(args.pose_port), *args.limit]
    found = re.compile(r"평가 경로 posenet_dist ([0-9.]+)")
    ps = s.setdefault("pose", {})
    if ps.get("renderer") != renderer.name or ps.get("rbits") != rbits:  # 렌더러가 바뀌면 처음부터
        ps.clear()
        ps.update({"renderer": renderer.name, "rbits": rbits, "variants": {}})
    variants = ps.setdefault("variants", {})
    for res in [v.strip() for v in args.pose_variants.split(",") if v.strip()]:
        key = f"{res}_b{args.pose_bbits}"
        if key in variants:
            continue
        t = time.time()
        bres = [] if res == "24x32" else ["--b-res", res.replace("x", ",")]
        main_out, pol_out = CACHE / f"gpu_pose_{key}_main.bin", CACHE / f"gpu_pose_{key}_pol.bin"
        txt1 = run([sys.executable, "pose_refine.py", "--pose2", str(start), *common, "--epochs", str(args.pose_epochs), "--lr", "0.01",
                    "--train-b", "0.02", "--dimw", "1.0", "--plain-epochs", str(args.plain_epochs), "--bbits", str(args.pose_bbits),
                    *(["--b-qat"] if args.pose_bbits < 6 else []), *bres, "--out", str(main_out)], LOGS / f"pose_{key}_main.log")
        txt2 = run([sys.executable, "pose_refine.py", "--pose2", str(main_out), *common, "--epochs", str(args.polish_epochs), "--lr", "0.001",
                    "--out", str(pol_out)], LOGS / f"pose_{key}_polish.log")
        d1, d2 = float(found.findall(txt1)[-1]), float(found.findall(txt2)[-1])
        best, d = (pol_out, d2) if d2 <= d1 else (main_out, d1)
        shutil.copy(best, RES / f"pose2_{key}.bin")
        nb = pose_bytes(best)
        sc = (10 * d) ** 0.5 + 25 * nb / 37_545_489
        variants[key] = {"posenet": d, "pose_term": round((10 * d) ** 0.5, 5), "bytes": nb, "pose_plus_rate": round(sc, 5),
                         "minutes": round((time.time() - t) / 60, 1)}
        bk = min(variants, key=lambda k: variants[k]["pose_plus_rate"])
        shutil.copy(RES / f"pose2_{bk}.bin", RES / "pose2_best.bin")
        ps["best"] = bk
        save_summary(s)
        print(f"== pose {key}: posenet {d:.7f} (항 {(10 * d) ** 0.5:.4f}), {nb:,} B → pose+rate {sc:.4f} (최고 {bk})", flush=True)
        push(args, f"colab: pose {key} posenet {d:.7f} {nb} B")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default="renderer,pose", help="renderer, pose 중 쉼표로")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--widths", default="24,32,40")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--cycles", type=int, default=6, help="렌더러 사이클 수 (사이클마다 flip 일정을 처음부터)")
    ap.add_argument("--epochs", type=int, default=10, help="사이클당 에폭")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--pose-epochs", type=int, default=200)
    ap.add_argument("--pose-bbits", type=int, default=5, help="pose 기저 B 저장 비트 (6 미만이면 기저 QAT)")
    ap.add_argument("--pose-variants", default="24x32,12x16", help="기저 해상도 변형 (쉼표로)")
    ap.add_argument("--pose-cbits", type=int, default=9, help="pose 계수 격자 비트 (9 는 10 과 같은 pose 항에 -1.35KB)")
    ap.add_argument("--r3-cycles", type=int, default=4, help="3비트 렌더러 트랙 사이클 수")
    ap.add_argument("--sc-cycles", type=int, default=4, help="self-compression 렌더러 트랙 사이클 수")
    ap.add_argument("--sc-weight", type=float, default=1.0, help="self-compression 크기 손실 배율 (1 = 점수 공식 기준)")
    ap.add_argument("--push", action="store_true", help="결과를 colab-results 브랜치로 push")
    ap.add_argument("--renderer-port", type=int, default=8020, help="렌더러 학습 livevis 포트")
    ap.add_argument("--pose-port", type=int, default=8014, help="pose 피팅 livevis 포트")
    ap.add_argument("--smoke", action="store_true", help="흐름 점검: 앞 8장, 1에폭, 2사이클 (결과는 results_smoke/)")
    args = ap.parse_args()
    args.plain_epochs, args.polish_epochs, args.q_epochs, args.limit = 30, 40, 10, []
    if args.smoke:
        global RES, LOGS, SUMMARY
        RES = HERE / "results_smoke"
        LOGS, SUMMARY = RES / "logs", RES / "summary.json"
        args.epochs, args.cycles, args.pose_epochs, args.r3_cycles, args.sc_cycles = 1, 2, 2, 1, 1
        args.plain_epochs, args.polish_epochs, args.q_epochs, args.limit = 1, 1, 1, ["--limit", "8"]

    RES.mkdir(parents=True, exist_ok=True)
    s = load_summary()
    s.setdefault("seed", json.loads((SEED / "seed.json").read_text()) if (SEED / "seed.json").exists() else {})
    save_summary(s)
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    if "renderer" in jobs:
        renderer_job(args, s)
    if "renderer3" in jobs:
        renderer_job(args, s, track="renderer3", bits=3, cycles=args.r3_cycles, start=RES / "renderer_best.pt")
    if "renderer_sc" in jobs:
        renderer_job(args, s, track="renderer_sc", bits=4, cycles=args.sc_cycles, start=RES / "renderer_best.pt")
    if "pose" in jobs:
        pose_job(args, s)
    print("\n모든 작업 끝:", json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
