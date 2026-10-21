#!/usr/bin/env bash
# 사용: inflate.sh <archive_dir> <output_dir> <video_names_file>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$HERE/inflate.py" "$1" "$2" "$3"
