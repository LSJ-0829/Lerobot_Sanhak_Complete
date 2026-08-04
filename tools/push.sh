#!/usr/bin/env bash
# LSJ 디렉터리의 변경을 GitHub(LSJ-0829/Lerobot_Sanhak_Complete)에 저장한다.
# 인증은 이 머신의 배포키(~/.ssh/id_ed25519, 저장소 쓰기권한)로 이뤄진다 — 대화형 로그인 불필요.
#
#   bash tools/push.sh                  # 자동 메시지
#   bash tools/push.sh "메시지"          # 직접 메시지
#
# models/ 는 .gitignore 로 제외된다(865MB 체크포인트).
set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ -z "$(git status --porcelain)" ]; then
  echo "변경 없음 — 저장할 것이 없습니다."
  exit 0
fi
echo "--- 변경된 파일 ---"
git status --short
git add -A
git commit -q -m "${1:-chore: 시연 본체에서 자동 저장 $(date '+%Y-%m-%d %H:%M')}"
git push -q origin main
echo "✅ 저장 완료: $(git log --oneline -1)"
