#!/usr/bin/env bash
# comma video compression challenge 작업 환경을 만든다.
#   - 챌린지 저장소를 comma_vcc/challenge 에 클론 (git 에는 올리지 않음)
#   - LFS 파일(영상, 모델)을 받는다. git lfs 가 막혀 있으면 media.githubusercontent.com 로 받는다.
#   - comma_vcc/.venv 에 의존성 설치 (download.pytorch.org 가 막힌 환경이라 PyPI 에서 받는다)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="commaai/comma_video_compression_challenge"
CH="$HERE/challenge"

if [ ! -d "$CH/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 "https://github.com/$REPO.git" "$CH"
fi

for f in videos/0.mkv models/segnet.safetensors models/posenet.safetensors; do
  oid="$(git -C "$CH" show "HEAD:$f" | sed -n 's/^oid sha256://p')"
  if [ "$(sha256sum "$CH/$f" | cut -d' ' -f1)" = "$oid" ]; then continue; fi
  echo "LFS 파일 받는 중: $f"
  if ! (cd "$CH" && git lfs pull --include "$f" 2>/dev/null) || [ "$(sha256sum "$CH/$f" | cut -d' ' -f1)" != "$oid" ]; then
    branch="$(git -C "$CH" rev-parse --abbrev-ref HEAD)"
    curl -sSfL -o "$CH/$f.tmp" "https://media.githubusercontent.com/media/$REPO/$branch/$f"
    mv "$CH/$f.tmp" "$CH/$f"
  fi
  [ "$(sha256sum "$CH/$f" | cut -d' ' -f1)" = "$oid" ] || { echo "해시 불일치: $f" >&2; exit 1; }
done

if [ ! -x "$HERE/.venv/bin/python" ]; then
  uv venv -q "$HERE/.venv" --python 3.11
fi
VIRTUAL_ENV="$HERE/.venv" uv pip install -q -r "$HERE/requirements.txt"

# 우리 제출물을 챌린지 submissions/ 아래에 링크해야 inflate.sh 의 `python -m submissions.<이름>.inflate` 가 동작한다.
for d in "$HERE"/submissions/*/; do
  [ -d "$d" ] || continue
  ln -sfn "$d" "$CH/submissions/$(basename "$d")"
done

echo "준비 완료. 예) bash comma_vcc/run.sh baseline_fast"
