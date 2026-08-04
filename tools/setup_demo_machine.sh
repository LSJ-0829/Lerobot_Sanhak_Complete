#!/usr/bin/env bash
# 시연 본체(SO101 bimanual + LeKiwi 를 한 대에서 다 돌리는 머신)에 필요한 것들을 채운다.
#
#   1) 상공 카메라 udev ID  (/dev/lerobot/overhead)  — sudo 필요
#   2) camera_0 충돌 수정                            — sudo 필요
#   3) open_clip_torch 설치 (CLIP grasp probe 용)
#   4) SmolVLA 체크포인트 내려받기 (HF, 약 865MB)
#   5) 검증
#
# 네트워크 설정은 건드리지 않는다. Jetson USB 이더넷은 별도로 잡아야 한다(README 참고).
#
# 사용:  bash tools/setup_demo_machine.sh [--skip-udev] [--skip-model]

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${LEROBOT_PYTHON:-/home/lerobot/miniconda3/envs/lerobot/bin/python}"
HF_REPO="${HF_REPO:-HyeonseokE/smolvla_lekiwi_spin_cycle}"
CKPT_DIR="$HERE/models/smolvla_lekiwi_spin_cycle"

SKIP_UDEV=0; SKIP_MODEL=0
for a in "$@"; do
  case "$a" in
    --skip-udev) SKIP_UDEV=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
  esac
done

echo "════════ 시연 본체 셋업 ════════"
echo "번들   : $HERE"
echo "python : $PY"

# ── 1) udev ───────────────────────────────────────────────────────────────────
if [ "$SKIP_UDEV" -eq 0 ]; then
  echo
  echo "[1] 상공 카메라 udev ID (/dev/lerobot/overhead)"
  if [ -e /dev/lerobot/overhead ]; then
    echo "    이미 있음 → $(readlink -f /dev/lerobot/overhead)"
  else
    sudo cp "$HERE/tools/99-lekiwi-overhead.rules" /etc/udev/rules.d/ \
      && sudo udevadm control --reload && sudo udevadm trigger --action=add && sleep 2
    echo "    → $(readlink -f /dev/lerobot/overhead 2>/dev/null || echo '생성 실패')"
  fi

  echo "[2] camera_0 충돌 수정 (상공캠이 top 카메라 이름을 뺏는 문제)"
  sudo bash "$HERE/tools/fix_camera0_conflict.sh" || echo "    ⚠️ 수동 확인 필요"
fi

# ── 3) open_clip ──────────────────────────────────────────────────────────────
echo
echo "[3] open_clip_torch (CLIP grasp probe)"
if "$PY" -c "import open_clip" 2>/dev/null; then
  echo "    이미 설치됨"
else
  "$PY" -m pip install -q open_clip_torch && echo "    설치 완료" || echo "    ✗ 설치 실패"
fi

# ── 4) SmolVLA 체크포인트 ─────────────────────────────────────────────────────
if [ "$SKIP_MODEL" -eq 0 ]; then
  echo
  echo "[4] SmolVLA 체크포인트 ($HF_REPO → models/)"
  if [ -f "$CKPT_DIR/config.json" ]; then
    echo "    이미 있음: $CKPT_DIR"
  else
    "$PY" -c "import huggingface_hub" 2>/dev/null || "$PY" -m pip install -q huggingface_hub
    mkdir -p "$CKPT_DIR"
    "$PY" -m huggingface_hub.commands.huggingface_cli download "$HF_REPO" \
        --local-dir "$CKPT_DIR" 2>&1 | tail -3 \
      && echo "    → $CKPT_DIR" || echo "    ✗ 내려받기 실패 (HF 로그인/네트워크 확인)"
  fi
fi

# ── 5) 검증 ───────────────────────────────────────────────────────────────────
echo
echo "[5] 검증"
"$PY" "$HERE/examples/lekiwi/find_overhead_cam.py" 2>&1 | tail -4
echo
"$PY" "$HERE/scripts/skills/laundry_task4.py" --check 2>&1 | tail -22
