#!/usr/bin/env bash
# Colab(GPU) 학습 환경: 챌린지 코드 + 평가 네트워크 가중치 + 경량 캐시.
# 원본 영상은 받지 않는다 (학습에는 GT seg 맵 / pose 만 있으면 된다. colab/seed 에 들어 있다).
#   bash comma_vcc/colab/setup_colab.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VCC="$(dirname "$HERE")"
CH="$VCC/challenge"
REPO="commaai/comma_video_compression_challenge"

if [ ! -d "$CH/.git" ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 "https://github.com/$REPO.git" "$CH"
fi

# 평가 네트워크 (CPU 에서 쓰던 것과 같은 파일인지 해시로 확인)
declare -A SHA=(
  [models/segnet.safetensors]=68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6
  [models/posenet.safetensors]=0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576
)
for f in "${!SHA[@]}"; do
  if [ -f "$CH/$f" ] && [ "$(sha256sum "$CH/$f" | cut -d' ' -f1)" = "${SHA[$f]}" ]; then continue; fi
  echo "받는 중: $f"
  curl -sSfL -o "$CH/$f" "https://media.githubusercontent.com/media/$REPO/master/$f"
  [ "$(sha256sum "$CH/$f" | cut -d' ' -f1)" = "${SHA[$f]}" ] || { echo "해시 불일치: $f" >&2; exit 1; }
done

pip install -q timm einops segmentation-models-pytorch safetensors constriction av

# 경량 캐시: GT seg 맵, pose, 첫 쌍용 맵
mkdir -p "$VCC/cache"
python - "$HERE/seed" "$VCC/cache" <<'EOF'
import lzma, shutil, sys
from pathlib import Path
seed, cache = Path(sys.argv[1]), Path(sys.argv[2])
for f in seed.iterdir():
    out = cache / f.name.removesuffix(".xz")
    if out.exists():
        continue
    if f.suffix == ".xz":
        out.write_bytes(lzma.decompress(f.read_bytes()))
    else:
        shutil.copy(f, out)
    print("캐시:", out.name)
EOF

python - <<'EOF'
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")
EOF
echo "준비 완료"
