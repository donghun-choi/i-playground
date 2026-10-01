#!/usr/bin/env bash
# 제출물 하나를 압축 → 풀기 → inflate → 실시간 평가까지 돌린다.
#   bash comma_vcc/run.sh <제출물 이름> [--recompress] [--no-wait]
# 제출물은 comma_vcc/challenge/submissions/<이름> (우리 것은 setup.sh 가 링크해 둔다).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CH="$HERE/challenge"
PY="$HERE/.venv/bin/python"

NAME="${1:?사용법: run.sh <제출물 이름> [--recompress] [--no-wait]}"; shift
RECOMPRESS=0; EVAL_ARGS=()
for a in "$@"; do
  case "$a" in
    --recompress) RECOMPRESS=1 ;;
    *) EVAL_ARGS+=("$a") ;;
  esac
done

SUB="$CH/submissions/$NAME"
[ -d "$SUB" ] || { echo "제출물이 없습니다: $SUB (setup.sh 를 다시 돌려 링크하세요)" >&2; exit 1; }
export PATH="$HERE/.venv/bin:$PATH"  # 제출물 스크립트의 python 이 우리 venv 를 쓰게

cd "$CH"
if [ "$RECOMPRESS" = 1 ] || [ ! -f "$SUB/archive.zip" ]; then
  [ -f "$SUB/compress.sh" ] || { echo "archive.zip 도 compress.sh 도 없습니다" >&2; exit 1; }
  rm -f "$SUB/archive.zip"
  start=$(date +%s); bash "$SUB/compress.sh"; echo "압축: $(( $(date +%s) - start ))s"
fi

rm -rf "$SUB/archive" "$SUB/inflated"
mkdir -p "$SUB/archive"
unzip -q -o "$SUB/archive.zip" -d "$SUB/archive"
start=$(date +%s)
bash "$SUB/inflate.sh" "$SUB/archive" "$SUB/inflated" "$CH/public_test_video_names.txt"
echo "inflate: $(( $(date +%s) - start ))s (공식 평가는 inflate+평가 합쳐 30분 제한)"

"$PY" "$HERE/live_eval.py" --submission-dir "$SUB" ${EVAL_ARGS[@]+"${EVAL_ARGS[@]}"}
