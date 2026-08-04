#!/usr/bin/env bash
# csi-agent 의 camera_0 규칙이 우리 상공캠을 가로채는 문제를 고친다.
#
# 무슨 일이 있었나
#   99-lerobot.rules 의 camera_0(top 카메라) 규칙은 모델 문자열로 매칭한다:
#     ENV{ID_SERIAL}=="Sonix_Technology_Co.__Ltd._USB_2.0_Camera_SN0001"
#   규칙에 달린 주석은 "its model string is unique" 라고 적고 있는데, 상공캠을 꽂는 순간
#   그 전제가 깨진다 — 상공캠도 같은 Sonix / 같은 SN0001 이다. 그래서 나중에 꽂힌 상공캠이
#   /dev/lerobot/camera_0 을 가져가고, 수건개기 rollout 이 top 뷰 대신 세탁기를 보게 된다.
#
# 고치는 방법
#   camera_1/camera_2 가 이미 쓰는 방식(포트 = ID_PATH)으로 camera_0 도 바꾼다.
#   top 카메라는 포트 usb-0:7, 상공캠은 usb-0:6.3 (2026-08-04 실측).
#
# 이 스크립트는 csi-agent(lhwdev) 쪽 설정을 건드린다. 원본은 .bak 으로 백업하고,
# 바꾼 내용은 csi-agent/tools/99-lerobot.rules 에도 반영해 두는 게 좋다.
#
# 사용:  sudo bash tools/fix_camera0_conflict.sh [--top-path <ID_PATH>]

set -euo pipefail

RULES=/etc/udev/rules.d/99-lerobot.rules
TOP_PATH="${2:-pci-0000:00:14.0-usb-0:7:1.0}"   # top 카메라가 꽂힌 포트

if [ "$(id -u)" -ne 0 ]; then
  echo "root 권한이 필요합니다:  sudo bash $0" >&2
  exit 1
fi
if [ ! -f "$RULES" ]; then
  echo "✗ $RULES 가 없습니다. csi-agent 규칙이 설치돼 있는지 확인하세요." >&2
  exit 1
fi

if grep -q 'SYMLINK+="lerobot/camera_0"' "$RULES" && grep -q "ID_PATH.*camera_0" "$RULES"; then
  echo "이미 포트 기반으로 고쳐져 있습니다. 변경 없음."
  exit 0
fi

BAK="$RULES.bak.$(date +%Y%m%d_%H%M%S)"
cp -a "$RULES" "$BAK"
echo "백업: $BAK"

# camera_0 매칭을 ID_SERIAL → ID_PATH 로 교체
python3 - "$RULES" "$TOP_PATH" <<'PY'
import re, sys
path, top = sys.argv[1], sys.argv[2]
src = open(path, encoding="utf-8").read()
old = re.compile(r'^SUBSYSTEM=="video4linux",\s*ENV\{ID_SERIAL\}=="Sonix[^\n]*camera_0"\s*$', re.M)
new = (f'SUBSYSTEM=="video4linux", ENV{{ID_PATH}}=="{top}", ATTR{{index}}=="0", '
       f'SYMLINK+="lerobot/camera_0"')
out, n = old.subn(
    '# camera_0 = top. 상공캠도 같은 Sonix/SN0001 이라 모델 문자열로는 구분이 안 된다 →\n'
    '# camera_1/2 와 같이 물리 포트로 고정한다.\n' + new, src)
if n == 0:
    print("⚠️  camera_0 의 Sonix 매칭 줄을 못 찾았습니다. 수동 확인이 필요합니다.")
    sys.exit(2)
open(path, "w", encoding="utf-8").write(out)
print(f"camera_0 규칙 {n}개를 포트 기반({top})으로 교체")
PY

udevadm control --reload
udevadm trigger --action=add
sleep 1
echo "--- 결과 ---"
ls -l /dev/lerobot/ | grep -E "camera|overhead" || true
