#!/bin/bash
# 이 저장소의 커밋 작성자를 donghun-choi로 고정하고 커밋 메시지 검사 훅을 켠다.
set -euo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

git config user.name "donghun-choi"
git config user.email "97453378+donghun-choi@users.noreply.github.com"
git config core.hooksPath .githooks
