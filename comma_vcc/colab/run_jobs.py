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


def renderer_job(args, s: dict) -> None:
    """flip 손실 + QAT 사이클을 반복. 사이클마다 600장 전체 평가, 가장 좋은 것을 이어서 학습한다."""
    tr = s.setdefault("renderer", {"widths": args.widths, "bits": args.bits, "cycles": []})
    best_file = RES / "renderer_best.pt"
    if not best_file.exists():
        shutil.copy(SEED / "renderer_seed.pt", best_file)
        tr["best"] = {"cycle": 0, "disagreement": s.get("seed", {}).get("renderer_disagreement")}
    for c in range(len(tr["cycles"]) + 1, args.cycles + 1):
        t = time.time()
        out = CACHE / f"gpu_r{c}.pt"
        txt = run([sys.executable, "renderer.py", "--device", args.device, "--widths", args.widths, "--resume", str(best_file),
                   "--epochs", str(args.epochs), "--lr", str(args.lr), "--bs", str(args.bs), "--cosine", "--fp32",
                   "--bits", str(args.bits), "--qat", "--loss", "flip", "--full-eval", "--threads", "2", "--port", str(args.renderer_port),
                   "--out", str(out), *args.limit], LOGS / f"renderer_c{c}.log")
        d = float(re.findall(r"전체 600장 disagreement ([0-9.]+)", txt)[-1])
        rec = {"cycle": c, "disagreement": d, "seg_term": 100 * d, "minutes": round((time.time() - t) / 60, 1)}
        tr["cycles"].append(rec)
        prev = tr.get("best", {}).get("disagreement")
        if prev is None or d < prev:
            shutil.copy(out, best_file)
            tr["best"] = rec
        save_summary(s)
        print(f"== 사이클 {c}: 600장 불일치 {d:.6f} (최고 {tr['best']['disagreement']}) {rec['minutes']}분", flush=True)
        push(args, f"colab: 렌더러 사이클 {c} 불일치 {d:.6f}")


def pose_job(args, s: dict) -> None:
    """최고 렌더러에 맞춰 pose 재피팅: 정규화 손실 + 기저 학습 → 일반 MSE → 낮은 학습률 다듬기."""
    renderer = RES / "renderer_best.pt" if (RES / "renderer_best.pt").exists() else SEED / "renderer_seed.pt"
    start = SEED / "pose2_seed.bin"
    common = ["--device", args.device, "--renderer", str(renderer), "--renderer-cfg", args.widths, "--rbits", str(args.bits),
              "--cbits", "10", "--q-epochs", str(args.q_epochs), "--greedy-rounds", "1", "--threads", "2", "--port", str(args.pose_port), *args.limit]
    main_out, pol_out = CACHE / "gpu_pose_main.bin", CACHE / "gpu_pose_pol.bin"
    t = time.time()
    txt1 = run([sys.executable, "pose_refine.py", "--pose2", str(start), *common, "--epochs", str(args.pose_epochs), "--lr", "0.01",
                "--train-b", "0.02", "--dimw", "1.0", "--plain-epochs", str(args.plain_epochs), "--out", str(main_out)], LOGS / "pose_main.log")
    txt2 = run([sys.executable, "pose_refine.py", "--pose2", str(main_out), *common, "--epochs", str(args.polish_epochs), "--lr", "0.001",
                "--out", str(pol_out)], LOGS / "pose_polish.log")
    found = re.compile(r"평가 경로 posenet_dist ([0-9.]+)")
    d1, d2 = float(found.findall(txt1)[-1]), float(found.findall(txt2)[-1])
    best, d = (pol_out, d2) if d2 <= d1 else (main_out, d1)
    shutil.copy(best, RES / "pose2_best.bin")
    s["pose"] = {"renderer": renderer.name, "renderer_cycle": s.get("renderer", {}).get("best", {}).get("cycle", 0),
                 "posenet_main": d1, "posenet_polish": d2, "posenet_best": d, "pose_term": round((10 * d) ** 0.5, 5),
                 "minutes": round((time.time() - t) / 60, 1)}
    save_summary(s)
    print(f"== pose: 평가 경로 posenet {d:.7f} (항 {(10 * d) ** 0.5:.4f})", flush=True)
    push(args, f"colab: pose 재피팅 posenet {d:.7f}")


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
        args.epochs, args.cycles, args.pose_epochs = 1, 2, 2
        args.plain_epochs, args.polish_epochs, args.q_epochs, args.limit = 1, 1, 1, ["--limit", "8"]

    RES.mkdir(parents=True, exist_ok=True)
    s = load_summary()
    s.setdefault("seed", json.loads((SEED / "seed.json").read_text()) if (SEED / "seed.json").exists() else {})
    save_summary(s)
    jobs = [j.strip() for j in args.jobs.split(",") if j.strip()]
    if "renderer" in jobs:
        renderer_job(args, s)
    if "pose" in jobs:
        pose_job(args, s)
    print("\n모든 작업 끝:", json.dumps(s, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
