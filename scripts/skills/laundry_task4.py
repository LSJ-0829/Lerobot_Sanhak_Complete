# !/usr/bin/env python
"""세탁 파이프라인 오케스트레이터 — LeKiwi(laundry_task3) → SO101 수건개기(csi-agent).

두 본체를 순서대로 잇는다. 로봇 제어는 각 스크립트가 하고, 여기서는 '실행'만 이어붙인다.

  [A]  preflight : 장비·경로 확인 — 로봇이 움직이기 '전에' 먼저 확인해서, 수건을 전달해 놓고
                   다음 단계가 안 뜨는 상황을 막는다.
  [B0] prewarm   : 수건개기 rollout 을 IDLE 로 미리 띄워 ACT 4단계 + classifier 를 올려 둔다.
  [B]  task3     : laundry_task3.py 실행 (자기 preload → Enter → 실행).
                   approach → 문열기 → VLA 집기 → 복귀주행 → throw → (마지막 후퇴 후) 전달 모션.
  [C]  handoff   : task3 가 성공(exit 0)하면 미리 올려둔 rollout 이 그대로 이어받는다.

■ Enter 는 딱 한 번이다.
  무거운 로딩(SmolVLA + CLIP probe + ACT 4단계 + classifier)은 전부 그 Enter '앞'에서 끝난다.
  Enter 이후로는 추가 입력도, 로딩 스톨도 없다. (--pause 를 주면 한 번 더 받는다.)

■ 원격(csi-agent, https://github.com/lhwdev/csi-agent) 실행 규약:
  bi_so_follower(SO101 2팔) + top/left/right 카메라. clothing/ 을 CWD 로 두고 실행해야 한다
  (clothing/scripts/setup_script.py 가 CWD 이름을 검사함). 기본 커맨드는 scripts/rollout_auto.py
  (ACT 0~3단계 + classifier 자동 전이).

■ 아직 원격 환경 설정(장치 심볼릭 링크 /dev/lerobot/*, conda env, 체크포인트)은 안 돼 있다.
  그건 SSH 로 직접 붙어서 잡고, 여기서는 --csi-host / --csi-dir / --csi-python 만 맞춰주면 된다.

■ ⚠️ 경로는 본체마다 다르다:
  - 상공 카메라 번호(/dev/videoN)는 본체마다, 그리고 USB 를 다시 꽂을 때마다 바뀐다.
    그래서 기본값이 'auto' 다 — examples/lekiwi/find_overhead_cam.py 가 내장 카메라를 빼고
    실제로 프레임이 나오는 USB 노드를 찾는다. 캠이 여러 개면 --overhead-match C920 처럼 좁힌다.
  - Jetson IP 는 유선 192.168.55.1 / 무선 192.168.0.19 → --jetson-ip
  - 원격 SO101 본체는 csi-agent 규칙대로 /dev/lerobot/{follower_1,2, camera_0,1,2} 심볼릭 링크를 쓴다
  로컬/원격 preflight 가 이 경로들을 먼저 확인하고, 안 맞으면 로봇을 움직이기 전에 멈춘다.

실행 (기본 원격 호스트: lerobot@115.145.179.95):
  # 연결/장비만 점검 — 로봇 안 움직임. 제일 먼저 이걸 돌려볼 것
  python scripts/skills/laundry_task4.py --check

  # 전체: LeKiwi 세탁물 회수 → SO101 수건개기
  python scripts/skills/laundry_task4.py

  # LeKiwi 는 이미 끝났고 SO101 만 트리거
  python scripts/skills/laundry_task4.py --skip-task3

  # 이 본체의 실제 경로를 지정하며 실행
  python scripts/skills/laundry_task4.py --overhead-cam /dev/v4l/by-id/usb-XXXX-video-index0 \
      --jetson-ip 192.168.55.1 -- --record --skip-approach   # `--` 뒤는 전부 task3 인자

환경변수 기본값: CSI_HOST, CSI_DIR, CSI_PYTHON, CSI_CMD, CSI_ENV,
                LEROBOT_PYTHON, OVERHEAD_CAM, REMOTE_IP, HANDOFF_MOTION
"""

import argparse
import json
import math
import os
import re
import select
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]  # ~/lerobot

# ───────────────────────── 기본 설정(env 로 조절) ─────────────────────────
# 원격 SO101 본체
# 원격 실측값(2026-08-04, lerobot-H310M-H): 저장소는 lerobot2/csi-agent/lhwdev 아래,
# conda 는 miniconda3(랩탑의 miniforge3 와 다름). rollout_auto.py 는 clothing/ 기준 ../train/... 을 본다.
CSI_HOST = os.environ.get("CSI_HOST", "lerobot@115.145.179.95")  # ~/.ssh/config 별칭 'so101' 도 가능
CSI_DIR = os.environ.get("CSI_DIR", "/home/lerobot/lerobot2/csi-agent/lhwdev")
CSI_PYTHON = os.environ.get("CSI_PYTHON", "/home/lerobot/miniconda3/envs/lerobot/bin/python")
# 기본은 --no_step0 = step0(펼치기)를 빼고 step1(접기)부터. step0 정책이 아직 불완전해서다.
# 이러면 classifier 의 IDLE 탈출 후보가 class 2~3 으로 좁혀져서(rollout.py: start_step = 2),
# '수건이 펴져 있다'고 인식될 때까지 IDLE 에서 기다렸다가 접기부터 시작한다.
# 펼치기 단계를 되살리려면 --csi-with-step0.
CSI_CMD = os.environ.get("CSI_CMD", "scripts/rollout_auto.py --no_step0")
CSI_ENV = os.environ.get("CSI_ENV", "")                        # 예: "DISPLAY=:0 CUDA_VISIBLE_DEVICES=0"

# 로컬 LeKiwi 쪽 — ⚠️ 아래 두 값은 '본체마다 다르다'. 다른 PC 로 옮기면 반드시 다시 확인할 것.
#   OVERHEAD_CAM : 기본 'auto' — /dev/videoN 번호는 본체마다, 그리고 USB 를 다시 꽂을 때마다
#                  달라지므로 번호를 쓰지 않고 find_overhead_cam 이 실제로 프레임이 나오는
#                  USB 노드를 찾는다. 특정 캠을 못 박고 싶으면 /dev/v4l/by-id/... 를 준다.
#   JETSON_IP    : 유선(USB-C 이더넷) 192.168.55.1 / 무선 192.168.0.19 — 네트워크마다 다름.
LOCAL_PYTHON = os.environ.get("LEROBOT_PYTHON", "")
DEFAULT_TASK3 = REPO / "scripts" / "skills" / "laundry_task3.py"
OVERHEAD_CAM = os.environ.get("OVERHEAD_CAM", "auto")
OVERHEAD_MATCH = os.environ.get("OVERHEAD_MATCH", "")  # 캠이 여러 개일 때 이름으로 좁히기
JETSON_IP = os.environ.get("REMOTE_IP", "192.168.55.1")  # 유선 USB 링크 기본값. --wireless 는 task3 쪽
HANDOFF_MOTION = os.environ.get("HANDOFF_MOTION", "")  # motions/<이름>.json, 비우면 전달 모션 생략

# Jetson 자동 준비 — 아무 인자 없이 `python laundry_task4.py` 만 쳐도 되게 하는 값들.
#   JETSON_USER  : Jetson 로그인 계정(홈이 /home/<user>). 키 인증이 걸려 있어야 한다.
#   USBNET_UNIT  : USB 이더넷 정적 IP 를 붙이는 systemd unit. IP 가 없을 때만 재시작한다.
JETSON_USER = os.environ.get("JETSON_USER", "comnet02")
USBNET_UNIT = os.environ.get("USBNET_UNIT", "lekiwi-usbnet.service")
HOST_PORTS = (5555, 5556)
GRIP_REGS_PY = "lekiwi_grip_regs.py"   # Jetson 홈에 두는 6번 순응 레지스터 설정 도구
ARM_SPEED_PY = "lekiwi_arm_speed.py"   # 팔 1~6번 Goal_Velocity/Acceleration 설정 도구

# Jetson 에 심어두는 host 자동 재기동 래퍼.
#   host 는 클라이언트가 연결을 끊으면 함께 종료된다 → 매 실행마다 사람이 다시 띄워야 했다.
#   이 래퍼가 죽을 때마다 3초 뒤 다시 올려서 그 수고를 없앤다(systemd Restart=always 대용, sudo 불필요).
KEEPALIVE_PATH = "~/lekiwi_host_keepalive.sh"
KEEPALIVE_SH = r"""#!/bin/bash
# lekiwi_host 자동 재기동 래퍼 (laundry_task4.py 가 생성/관리).
#
# ⚠️ host 는 반드시 '한 번에 하나'만 떠야 한다.
#    두 개가 뜨면 뒤에 뜬 쪽은 5555 바인드에 실패하고도 루프를 계속 돌면서,
#    명령을 한 번도 못 받으니 워치독이 0.5초마다 stop_base() 를 같은 모터 버스에
#    써 넣는다 → 앞의 host 가 굴리던 바퀴가 0.5초 주기로 끊긴다.
#    flock 으로 상호배제한다. 중복 실행은 조용히 종료된다(경합 없음).
LOCK="$HOME/.lekiwi_host.lock"
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
  echo "[keepalive] $(date +%F' '%T) 이미 실행 중 — 중복 기동 취소" >> "$HOME/lekiwi_host.log"
  exit 0
fi

# 여기 왔다는 건 다른 keepalive 가 없다는 뜻 → 살아있는 host 는 죽은 래퍼가 남긴 고아다.
if pkill -f 'lerobot\.robots\.lekiwi\.lekiwi_host' 2>/dev/null; then
  echo "[keepalive] $(date +%F' '%T) 고아 host 정리함" >> "$HOME/lekiwi_host.log"
  sleep 2
fi

cd "$HOME/lerobot" || exit 1
PY="$HOME/miniforge3/envs/lerobot/bin/python"
while true; do
  echo "[keepalive] $(date +%F' '%T) host 기동" >> "$HOME/lekiwi_host.log"
  # 9>&- : 잠금 fd 를 자식에게 물려주지 않는다. 안 그러면 래퍼가 죽고 host 만 남았을 때
  #        그 고아가 잠금을 계속 쥐고 있어 새 래퍼가 영영 못 뜬다.
  "$PY" -m lerobot.robots.lekiwi.lekiwi_host \
      --robot.id=my_awesome_kiwi --host.connection_time_s=7200 >> "$HOME/lekiwi_host.log" 2>&1 9>&-
  echo "[keepalive] $(date +%F' '%T) host 종료(코드 $?) — 3초 후 재기동" >> "$HOME/lekiwi_host.log"
  sleep 3
done
"""

SSH_OPTS = ["-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]


def local_python() -> str:
    """conda lerobot env 파이썬(이 머신엔 uv 가 없다). 없으면 현재 인터프리터."""
    if LOCAL_PYTHON:
        return os.path.expanduser(LOCAL_PYTHON)
    cand = Path.home() / "miniforge3" / "envs" / "lerobot" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def run_streaming(cmd, prefix, cwd=None, env=None, show=None, sink=None):
    """자식 프로세스 출력을 prefix 붙여 실시간으로 흘리고 returncode 를 돌려준다.

    줄 단위로만 읽으면 안 된다: input() 의 프롬프트는 줄바꿈이 없어서 버퍼에 갇히고,
    사용자는 "Enter 를 기다리는 중"인지 "아직 로딩 중"인지 알 수 없게 된다.
    그래서 출력이 잠시 멎으면(=상대가 입력을 기다리는 상태) 남은 조각을 그대로 내보낸다.

    show 를 주면 그 문자열을 대신 표시한다(ssh 원격 스크립트는 따옴표 때문에 원문이 읽기 어렵다).
    sink 를 주면(list) 흘려보낸 텍스트를 거기에도 쌓는다 — 실패 사유를 사후에 판정하려고.
    """
    print(f"  $ {show or ' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, bufsize=0)
    fd = p.stdout.fileno()
    buf = b""

    def emit_lines():
        nonlocal buf
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace").rstrip()
            if sink is not None:
                sink.append(text)
            print(f"{prefix}{text}", flush=True)

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0.3)
            if ready:
                chunk = os.read(fd, 8192)
                if not chunk:            # EOF
                    buf += b""
                    break
                buf += chunk
                emit_lines()
            else:
                # 0.3초간 조용하다 = 줄바꿈 없이 멈춘 것(프롬프트). 있는 그대로 흘려준다.
                if buf:
                    sys.stdout.write(f"{prefix}{buf.decode('utf-8', 'replace')}")
                    sys.stdout.flush()
                    buf = b""
                if p.poll() is not None:
                    break
        emit_lines()
        if buf:
            print(f"{prefix}{buf.decode('utf-8', 'replace').rstrip()}", flush=True)
    except KeyboardInterrupt:
        print(f"\n{prefix}[중단] Ctrl+C → 자식 프로세스 종료 중...", flush=True)
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        raise
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
    return p.wait()


# ─────────────────────── [C] 원격(csi-agent) 실행 ───────────────────────
def remote_script(args, preflight_only=False, feed_enter=False):
    """원격에서 돌릴 bash 스크립트. clothing/ 을 CWD 로 잡는 게 핵심."""
    d = args.csi_dir.rstrip("/")
    py = args.csi_python
    lines = [
        "set -e",
        f'CSI_DIR="$(eval echo {shlex.quote(d)})"',
        f'CSI_PY="$(eval echo {shlex.quote(py)})"',
        'if [ ! -d "$CSI_DIR/clothing" ]; then echo "MISSING_DIR: $CSI_DIR/clothing"; exit 21; fi',
        'if [ ! -x "$CSI_PY" ]; then echo "MISSING_PYTHON: $CSI_PY"; exit 22; fi',
        'cd "$CSI_DIR/clothing"',
    ]
    if preflight_only:
        lines += [
            'echo "dir  : $CSI_DIR/clothing"',
            'echo "py   : $("$CSI_PY" -V 2>&1)"',
            '"$CSI_PY" -c "import lerobot" 2>/dev/null && echo "lerobot: OK" '
            '|| echo "lerobot: ⚠️ import 실패(원격 env 설정 필요)"',
            f'test -f {shlex.quote(args.csi_cmd.split()[0])} '
            f'&& echo "cmd  : {args.csi_cmd} OK" || echo "cmd  : ⚠️ {args.csi_cmd} 없음"',
            # ⚠️ 원격 본체는 장치 경로 규칙이 다르다(csi-agent 는 /dev/lerobot/* 심볼릭 링크를 쓴다).
            'ls -d /dev/lerobot/follower_1 /dev/lerobot/follower_2 2>/dev/null '
            '|| echo "dev  : ⚠️ /dev/lerobot/follower_* 없음(99-lerobot.rules 적용 필요)"',
            'ls -d /dev/lerobot/camera_0 /dev/lerobot/camera_1 /dev/lerobot/camera_2 2>/dev/null '
            '|| echo "cam  : ⚠️ /dev/lerobot/camera_* 없음(top/left/right 링크 필요)"',
            'echo PREFLIGHT_OK',
        ]
    else:
        if args.csi_env:
            lines.append("export " + args.csi_env)
        # feed_enter: 캘리브레이션 프롬프트에 빈 줄을 계속 먹여 준다(무인 실행용).
        lines.append(f'yes "" | exec "$CSI_PY" {args.csi_cmd}' if feed_enter
                     else f'exec "$CSI_PY" {args.csi_cmd}')
    return "\n".join(lines)


def ssh_cmd(args, script, tty=False):
    base = ["ssh", *SSH_OPTS]
    if tty:
        base.append("-tt")
    else:
        base += ["-o", "BatchMode=yes"]
    return [*base, args.csi_host, f"bash -lc {shlex.quote(script)}"]


def csi_is_local(args):
    """SO101 단계를 이 머신에서 바로 돌리는가(=여기가 SO101 본체인가)."""
    return args.csi_host.strip().lower() in ("", "local", "localhost")


class Prewarm:
    """수건개기 rollout 을 '첫 Enter 전에' 미리 띄워 모델을 다 올려 두는 핸들.

    왜 필요한가: 그냥 순서대로 돌리면 task3 가 끝난 '뒤'에 ACT 정책 4개 + classifier 를
    로드하느라 수건을 든 채로 수십 초를 멈춰 있게 된다. 그래서 task3 의 preload 와 동시에
    rollout 프로세스를 띄워 두고, 로딩이 끝난 상태로 IDLE 에서 대기시킨다.

    IDLE 대기가 전제다 — 미리 띄우는 이상 --immediate_start 를 쓰면 수건이 오기도 전에
    팔이 움직인다. 그래서 prewarm 은 항상 IDLE(classifier 판단) 모드로 돈다.

    로딩 중 출력은 로그 파일로 보낸다. task3 의 Enter 프롬프트가 rollout 로그에 묻히면
    안 되기 때문이다. task3 가 끝난 뒤 모아서 흘려준다.
    """

    def __init__(self, cmd, cwd, logpath, label):
        self.logpath = logpath
        self.label = label
        self._fh = open(logpath, "wb")
        # stdin 을 터미널에서 떼어낸다. 안 그러면 rollout 의 캘리브레이션 프롬프트
        # ("Press ENTER to use provided calibration file...")가 task3 의 시작 Enter 를
        # 가로챈다 — 같은 tty 를 두 프로세스가 읽으면 누가 먹을지 정해지지 않는다.
        # start_new_session: 자체 프로세스 그룹을 준다. `yes '' | python ...` 은 bash 를 거치므로
        # terminate() 로 bash 만 죽이면 python 이 로봇을 쥔 채 살아남는다 → 다음 실행이 장치를
        # 못 연다. 그룹 전체에 시그널을 보내야 확실히 정리된다.
        self.p = subprocess.Popen(cmd, cwd=cwd, stdout=self._fh, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, start_new_session=True)

    def alive(self):
        return self.p.poll() is None

    def _signal_group(self, sig):
        try:
            os.killpg(os.getpgid(self.p.pid), sig)
        except (ProcessLookupError, PermissionError):
            pass

    def stop(self, why=""):
        if self.alive():
            print(f"  ⏹ 미리 띄운 수건개기 종료{(' — ' + why) if why else ''}")
            self._signal_group(signal.SIGTERM)
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    self.p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if not self._fh.closed:
            self._fh.close()

    def follow(self, prefix):
        """지금까지의 로그를 뱉고, 프로세스가 끝날 때까지 이어서 흘린다.

        로그가 쓰이는 도중에는 readline() 이 줄바꿈 없는 반쪽 줄을 돌려준다. 그대로 찍으면
        한 줄이 여러 줄로 쪼개져 보이므로, 줄바꿈이 올 때까지 모았다가 내보낸다.
        """
        pending = ""
        try:
            with open(self.logpath, "r", errors="replace") as f:
                while True:
                    chunk = f.readline()
                    if chunk:
                        pending += chunk
                        if pending.endswith("\n"):
                            print(f"{prefix}{pending.rstrip()}", flush=True)
                            pending = ""
                        continue
                    if self.p.poll() is not None:
                        pending += f.read()
                        for ln in pending.splitlines():
                            print(f"{prefix}{ln}", flush=True)
                        break
                    time.sleep(0.2)
        except KeyboardInterrupt:
            self.stop("Ctrl+C")
            raise
        finally:
            if not self._fh.closed:
                self._fh.close()
        return self.p.returncode


def local_csi_cmd(args, feed_enter=True):
    """SSH 없이 로컬에서 돌릴 커맨드 (CWD=<csi_dir>/clothing).

    feed_enter=True 면 `yes '' |` 로 빈 줄을 계속 흘려 넣는다. lerobot 은 팔마다
      "Press ENTER to use provided calibration file ... or type 'c' to run calibration:"
    을 물어보는데, 여기서 'c' 가 아닌 빈 줄은 '기존 캘리브레이션 파일을 쓴다'는 뜻이라
    자동으로 넘겨도 안전하다(캘리브레이션을 새로 하지 않는다).
    prewarm 경로는 이미 이렇게 하고 있었는데 --skip-task3 로 직접 실행하는 경로에는
    빠져 있어서, 그 경우에만 사람이 엔터를 쳐야 했다.
    """
    py = os.path.expanduser(args.csi_python)
    cwd = os.path.join(os.path.expanduser(args.csi_dir), "clothing")
    if not feed_enter:
        return [py, *shlex.split(args.csi_cmd)], cwd
    inner = f"{shlex.quote(py)} {args.csi_cmd}"
    return ["bash", "-c", f"yes '' | PYTHONUNBUFFERED=1 {inner}"], cwd


def start_prewarm(args):
    """수건개기 rollout 을 IDLE 모드로 미리 띄운다(모델 로딩을 첫 Enter 앞으로 당기기)."""
    idle_cmd = args.csi_cmd.replace("--immediate_start", "").strip()
    log = os.path.join(tempfile.gettempdir(), f"laundry_task4_csi_{os.getpid()}.log")
    print("─" * 60)
    print("[B0] 수건개기 정책 미리 로딩 (IDLE 대기) — task3 preload 와 동시에 진행")
    print(f"     `{idle_cmd}`   로그: {log}")
    print("     ⚠️ 이 프로세스는 지금부터 top 카메라를 보고 있습니다. 접는 판에 이미 수건이 있으면")
    print("        task3 가 끝나기 전에도 SO101 이 움직입니다. 판을 비워 두고 시작하세요.")
    # `yes '' |` 로 빈 줄을 계속 먹여 준다 — lerobot 의 로봇 connect 는
    # "Press ENTER to use provided calibration file..." 로 Enter 를 요구하는데,
    # 빈 줄이면 기존 캘리브레이션을 그대로 쓴다. (ensure_host 도 같은 방식을 쓴다.)
    if csi_is_local(args):
        inner = " ".join(shlex.quote(c) for c in
                         [os.path.expanduser(args.csi_python), *shlex.split(idle_cmd)])
        # 로그 파일로 나가므로 더더욱 버퍼링을 꺼야 follow() 가 실시간으로 보여줄 수 있다.
        cmd = ["bash", "-c", f"yes '' | PYTHONUNBUFFERED=1 {inner}"]
        cwd = os.path.join(os.path.expanduser(args.csi_dir), "clothing")
    else:
        saved, args.csi_cmd = args.csi_cmd, idle_cmd
        cmd, cwd = ssh_cmd(args, remote_script(args, feed_enter=True)), None
        args.csi_cmd = saved
    return Prewarm(cmd, cwd, log, "csi")


def preflight(args):
    print("─" * 60)
    where = "이 머신(로컬)" if csi_is_local(args) else args.csi_host
    print(f"[A] preflight: SO101 쪽 확인 — {where}")

    # 순서가 중요하다: 팔을 기준 자세로 먼저 세운 뒤에 카메라를 배정한다.
    # 좌/우 팔캠 구분은 '화면에 접는 판이 어느 쪽으로 보이는지'로 하는데, 팔이 제멋대로
    # 있으면 그 화면이 달라져 배정이 흔들린다. 자세를 고정해 놓고 봐야 재현된다.
    # 실패하면 preflight 를 통과시키지 않는다 — Enter 를 받기 전에 멈춰야
    # '수건은 전달됐는데 폴딩이 안 되는' 상황을 피할 수 있다.
    # 모델 확인이 맨 앞이다: 로봇을 건드리지 않고 즉시 끝나므로, 파일이 문제면
    # 팔을 움직이기도 전에 멈추는 게 낫다.
    pre_ok = True
    if csi_is_local(args):
        pre_ok &= check_models(args)
        pre_ok &= home_arms(args)
        pre_ok &= resolve_folding_cams(args)

    if csi_is_local(args):
        # SSH 를 돌 이유가 없다. 같은 것들을 로컬에서 직접 확인한다.
        ok = True
        cdir = Path(os.path.expanduser(args.csi_dir)) / "clothing"
        py = Path(os.path.expanduser(args.csi_python))
        print(f"  | dir  : {cdir}{'' if cdir.is_dir() else '  ✗ 없음'}")
        ok &= cdir.is_dir()
        print(f"  | py   : {py}{'' if py.exists() else '  ✗ 없음'}")
        ok &= py.exists()
        if cdir.is_dir():
            script = cdir / shlex.split(args.csi_cmd)[0]
            print(f"  | cmd  : {args.csi_cmd}{'  OK' if script.exists() else '  ✗ 없음'}")
            ok &= script.exists()
        for d in ("follower_1", "follower_2", "camera_0", "camera_1", "camera_2"):
            p = Path("/dev/lerobot") / d
            print(f"  | dev  : {p}{'' if p.exists() else '  ✗ 없음(99-lerobot.rules 필요)'}")
            ok &= p.exists()
        if ok:
            if not pre_ok:
                print("  ✗ preflight 실패 — 모델 파일/홈 자세/카메라 배정 확인")
                return False
            print("  ✅ preflight OK")
        else:
            print("  ✗ preflight 실패 — 위 ✗ 항목 확인")
        return ok

    if not args.csi_host:
        print("  ✗ --csi-host 가 없습니다 (또는 CSI_HOST env). SO101 단계를 못 돌립니다.")
        return False
    rc = run_streaming(ssh_cmd(args, remote_script(args, preflight_only=True)), "  | ",
                       show=f"ssh {args.csi_host} '<preflight: dir/python/lerobot/cmd/dev 확인>'")
    if rc == 0:
        print("  ✅ preflight OK")
        return True
    print(f"  ✗ preflight 실패(exit {rc}) — SSH 키/경로/파이썬 확인 필요")
    return False


# 모터가 '가끔 이유 없이' 인식되지 않는다. 하드웨어 고장이 아니라 버스가 아직 정리되지
# 않은 상태에서 연결을 시도해 생기는 일이라, 다시 띄우면 대개 그냥 된다(실측 확인:
# 실패 직후 1~6번 전부 정상 응답). 아래 흔적이 보이면 프로세스를 통째로 다시 띄운다.
# 연결 호출 하나만 감싸지 않고 프로세스를 재시작하는 이유: 실패 지점이 rollout.py 일 수도
# setup_devices.py 일 수도 있어서(실제로 후자에서 터졌다), 지점을 특정하지 않는 편이 확실하다.
MOTOR_FAIL_MARKS = (
    "no status packet",
    "Failed to write",
    "Failed to sync read",
    "motor check failed",
    "Missing motor IDs",
    "ConnectionError",
    "Failed to connect",
)

# 카메라도 같은 성격으로 '가끔' 안 잡힌다. 앞 프로세스가 아직 장치를 놓지 않았거나,
# USB 에는 붙어 있는데 드라이버가 안 잡혀 노드 자체가 없는 경우다. 후자는 기다린다고
# 낫지 않아서 USB 전원 리셋이 필요하다 — 그래서 모터 실패와 따로 본다.
CAM_FAIL_MARKS = (
    "could not be opened",
    "device busy",
    "Failed to open",
    "OpenCVCamera",
    "VIDIOC_",
)

MOTOR_FAIL_MARKS = MOTOR_FAIL_MARKS + CAM_FAIL_MARKS


def _looks_like_motor_failure(lines) -> str:
    for text in reversed(lines[-400:]):
        for m in MOTOR_FAIL_MARKS:
            if m in text:
                return text.strip()[:160]
    return ""


def _looks_like_camera_failure(why: str) -> bool:
    return any(m in why for m in CAM_FAIL_MARKS)


def ensure_cams_valid(args, quiet=False):
    """링크가 아직 유효한 캡처 노드를 가리키는지 확인하고, 아니면 다시 배정한다.

    USB 가 재열거되면 /dev/videoN 번호가 통째로 밀린다. 그러면 예전에 캡처 노드였던
    번호가 메타데이터 노드(index 1)가 되어, 파일은 있는데 열리지는 않는다. preflight
    때 맞춰 놓아도 실제 실행 시점에 어긋날 수 있으므로, 띄우기 직전에 다시 확인한다.
    """
    if args.no_cam_resolve:
        return True
    try:
        sys.path.insert(0, str(REPO / "examples" / "lekiwi"))
        import usb_reset
        from resolve_cameras import links_valid
        # 링크가 가리키는 노드가 멀쩡해 보여도, USB 쪽에서 드라이버가 빠져 있으면
        # 열리지 않는다. 링크보다 먼저 본다(리셋하면 어차피 링크를 다시 걸어야 한다).
        if usb_reset.recover(expect=args.cam_expect, attempts=1, verbose=False) and links_valid():
            return True
        if not quiet:
            print("  ⚠️ 카메라 상태가 어긋났습니다(USB 재열거/드라이버 빠짐) — 다시 배정합니다")
        return resolve_folding_cams(args)
    except Exception as e:
        if not quiet:
            print(f"  ⚠️ 카메라 재확인 건너뜀({type(e).__name__}: {e})")
        return True


def _wait_cams_free(args, timeout=15.0):
    """폴딩 캠 3대가 다시 열리는 상태가 될 때까지 기다린다.

    앞 프로세스(prewarm 또는 직전 시도)가 카메라를 놓는 데 시간이 걸린다. 곧바로 새
    프로세스를 띄우면 device busy 로 죽는다 — 재시도가 매번 같은 이유로 실패하게 된다.
    """
    if args.no_cam_resolve:
        return True
    try:
        sys.path.insert(0, str(REPO / "examples" / "lekiwi"))
        from resolve_cameras import LINKS, wait_released
        devs = [str(LINKS / r) for r in ("top", "left_cam", "right_cam")
                if (LINKS / r).exists()]
        if not devs:
            return True
        ok = wait_released(devs, timeout=timeout, verbose=False)
        if not ok:
            print(f"  ⚠️ 카메라가 {timeout:.0f}초 안에 풀리지 않았습니다 — 그대로 진행합니다")
        return ok
    except Exception:
        return True


def run_csi(args):
    print("─" * 60)
    local = csi_is_local(args)
    where = (f"{cwd_of(args)} 에서" if local
             else f"{args.csi_host}:{args.csi_dir}/clothing 에서")
    print(f"[C] SO101 수건개기{'(로컬)' if local else ''}: {where} `{args.csi_cmd}`")

    for attempt in range(1, args.csi_retries + 1):
        if attempt > 1:
            print(f"\n  ↻ 재시도 {attempt}/{args.csi_retries} — {args.csi_retry_delay:.0f}초 후 다시 로딩")
            time.sleep(args.csi_retry_delay)
        # 앞 시도가 카메라를 놓을 때까지 기다리고, 링크가 아직 유효한지도 확인한다.
        # (USB 재열거로 /dev/videoN 번호가 밀리면 링크가 메타데이터 노드를 가리키게 된다)
        _wait_cams_free(args)
        ensure_cams_valid(args)
        lines = []
        try:
            if local:
                cmd, cwd = local_csi_cmd(args)
                rc = run_streaming(cmd, "  ▸ ", cwd=cwd, sink=lines)
            else:
                tty = sys.stdin.isatty() and not args.no_tty
                rc = run_streaming(ssh_cmd(args, remote_script(args), tty=tty), "  ▸ ",
                                   show=f"ssh {args.csi_host} 'cd {args.csi_dir}/clothing && "
                                        f"{args.csi_python} {args.csi_cmd}'", sink=lines)
        except KeyboardInterrupt:
            raise

        if rc == 0:
            print(f"  ✅ 수건개기 완료{'' if attempt == 1 else f' ({attempt}번째 시도)'}")
            return 0
        if rc == 130:
            print("  ✗ 사용자 중단")
            return rc

        why = _looks_like_motor_failure(lines)
        if not why:
            print(f"  ✗ 수건개기 실패/중단 (exit {rc}) — 모터 인식 실패가 아니라 재시도하지 않습니다")
            return rc
        print(f"  ✗ 인식 실패 (exit {rc}): {why}")
        if _looks_like_camera_failure(why):
            # 카메라가 USB 에는 붙어 있는데 드라이버가 안 잡힌 상태는 기다린다고 낫지 않는다.
            # 뽑았다 꽂는 것과 같은 일(authorized 0→1)을 하고 다시 배정한다.
            print("  카메라 쪽으로 보입니다 — USB 전원 리셋 후 다시 배정합니다")
            sys.path.insert(0, str(REPO / "examples" / "lekiwi"))
            import usb_reset
            usb_reset.recover(expect=args.cam_expect, attempts=args.cam_resets,
                              verbose=True, force=True)
            resolve_folding_cams(args)
        if attempt == args.csi_retries:
            print(f"  ✗ {args.csi_retries}회 모두 실패 — 팔 전원/USB 케이블을 확인하세요")
            return rc
    return rc


def cwd_of(args):
    return os.path.join(os.path.expanduser(args.csi_dir), "clothing")


# ─────────────────────── Jetson 자동 준비 ───────────────────────
def _ping(ip, timeout=2) -> bool:
    return subprocess.run(["ping", "-c", "1", "-W", str(timeout), ip],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def _port_open(ip, port, timeout=2) -> bool:
    import socket
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except OSError:
        return False


def _jetson_ssh(args, script, timeout=25):
    """Jetson 에 짧은 명령 실행. (rc, stdout) 반환."""
    cmd = ["ssh", *SSH_OPTS, "-o", "BatchMode=yes",
           f"{args.jetson_user}@{args.jetson_ip}", "bash -s"]
    try:
        p = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def ensure_jetson_link(args) -> bool:
    """유선 USB 링크가 죽어 있으면 정적 IP 서비스를 다시 올린다.

    Jetson 이 USB 를 재열거하면(케이블 흔들림·재부팅) 정적 IP 가 날아가는데, 그 서비스는
    oneshot 이라 스스로 복구하지 않는다. 여기서 한 번 되살려 본다.
    ⚠️ 이 서비스는 enx* 만 건드리고 본체 인터넷 회선(enp2s0)은 손대지 않는다.
    """
    if _ping(args.jetson_ip):
        return True
    print(f"  Jetson : {args.jetson_ip} 응답 없음 → {args.usbnet_unit} 재시작 시도")
    r = subprocess.run(["sudo", "-n", "systemctl", "restart", args.usbnet_unit],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print("           ✗ 무암호 sudo 가 없어 자동 복구 불가. 본체 터미널에서:")
        print(f"             sudo systemctl restart {args.usbnet_unit}")
        return False
    time.sleep(3)
    if _ping(args.jetson_ip):
        print("           ✅ 링크 복구됨")
        return True
    print("           ✗ 여전히 응답 없음 — Jetson 전원/USB 케이블 확인")
    return False


def ensure_grip_regs(args):
    """6번(gripper) 순응제어 레지스터를 보장한다. **host 가 안 떠 있을 때만** 호출할 것.

    host 가 /dev/ttyACM0 를 점유하므로 동시 접근은 패킷이 섞여 위험하다. 반대로 host 기동
    직전은 버스가 비어 있는 유일하게 안전한 창이라, 여기서 한 번 맞춰 둔다.
    EEPROM 이라 값이 유지되고, 스크립트는 다를 때만 쓰므로 매번 불러도 마모가 없다.

    P_Coefficient 가 핵심이다. 16 이면 손잡이에 막힌 위치오차에도 토크를 약하게 내
    Torque_Limit 에 한참 못 미치는 데서 평형이 잡혀 그립이 약하다 → 당길 때 놓친다.
    """
    if args.no_grip_regs:
        return
    rc, out = _jetson_ssh(
        args, f"~/miniforge3/envs/lerobot/bin/python ~/{GRIP_REGS_PY} --set 2>&1 | grep -v Warning",
        timeout=40)
    if rc != 0:
        print(f"  그립설정: ⚠️ 실패(rc={rc}) — 순응제어는 위치제어만으로 동작 (치명적 아님)")
        return
    if "변경 없음" in out:
        print("  그립설정: 6번 순응 레지스터 이미 맞음 (P게인/토크상한/과부하)")
    else:
        changed = [ln.strip() for ln in out.splitlines() if "→" in ln]
        print(f"  그립설정: 6번 순응 레지스터 {len(changed)}개 조정")
        for ln in changed:
            print(f"           {ln}")


def ensure_arm_speed(args):
    """팔 1~6번의 Goal_Velocity/Acceleration 을 원본 play_motion 값(800/40)으로 맞춘다.

    **왜 매번 해야 하나:** 둘 다 SRAM 이라 전원을 껐다 켜면 공장값(0/254)으로 돌아간다.
    Goal_Velocity=0 은 '속도 무제한', Acceleration=254 는 거의 즉시 가속이라, 27.5Hz 로
    들어오는 웨이포인트마다 모터가 튀어가 멈추고 다음 것까지 36ms 를 가만히 있는다
    → 초당 27번 순간이동하는 꼴이라 뚝뚝 끊겨 보인다(FPS 가 낮아 보이는 현상의 정체).
    800/40 은 속도에 상한을 둬서 다음 웨이포인트가 올 때까지 계속 이동 중이게 만든다.
    host 가 버스를 점유하므로 host 기동 '전'에만 안전하다.
    """
    if args.no_grip_regs:
        return
    rc, out = _jetson_ssh(
        args, f"~/miniforge3/envs/lerobot/bin/python ~/{ARM_SPEED_PY} --set 2>&1 | grep -v Warning",
        timeout=60)
    if rc != 0:
        print(f"  팔속도  : ⚠️ 실패(rc={rc}) — 모션이 끊겨 보일 수 있음")
        return
    if "변경 없음" in out or "0개 레지스터" in out:
        print("  팔속도  : Goal_Velocity=800 / Acceleration=40 이미 적용됨")
    else:
        print("  팔속도  : 팔 1~6번 Goal_Velocity=800 / Acceleration=40 적용 "
              "(원본 play_motion 과 동일 — 웨이포인트 사이를 끊김 없이 이동)")


def ensure_jetson_host(args) -> bool:
    """lekiwi_host 가 안 떠 있으면 Jetson 에 접속해 keepalive 래퍼로 띄운다."""
    if all(_port_open(args.jetson_ip, p) for p in HOST_PORTS):
        print(f"  host   : {args.jetson_ip}:{HOST_PORTS[0]}/{HOST_PORTS[1]} 리슨 중")
        return True

    print("  host   : 안 떠 있음 → Jetson 에서 기동")
    # 버스가 비어 있는 지금이 레지스터를 만질 수 있는 유일한 창이다(host 기동 전).
    ensure_grip_regs(args)
    ensure_arm_speed(args)
    # 래퍼가 없거나 내용이 바뀌었으면 새로 심는다(멱등).
    rc, out = _jetson_ssh(args, f"cat > {KEEPALIVE_PATH} <<'__EOF__'\n{KEEPALIVE_SH}__EOF__\n"
                                f"chmod +x {KEEPALIVE_PATH} && echo ok")
    if rc != 0:
        print(f"           ✗ Jetson SSH 실패(rc={rc}): {out.strip()[:200]}")
        print(f"           ssh {args.jetson_user}@{args.jetson_ip} 키 인증이 되는지 확인하세요.")
        return False

    # 중복 기동 방지는 래퍼 안의 flock 이 최종 방어선이다(경쟁 상태에서도 안전).
    # 여기 pgrep 은 불필요한 프로세스 생성을 줄이는 1차 필터일 뿐이다.
    rc, out = _jetson_ssh(
        args,
        f"if pgrep -f '[l]ekiwi_host_keepalive.sh' > /dev/null; then echo already; else "
        f"setsid nohup {KEEPALIVE_PATH} > /dev/null 2>&1 < /dev/null & echo launched; fi")
    if rc != 0:
        print(f"           ✗ 기동 실패(rc={rc}): {out.strip()[:200]}")
        return False
    print(f"           {'이미 실행 중' if 'already' in out else '기동함'} — 리슨 대기")

    for i in range(30):
        if all(_port_open(args.jetson_ip, p, timeout=1) for p in HOST_PORTS):
            print(f"           ✅ host 준비됨 ({i + 1}초)")
            return True
        time.sleep(1)
    _, log = _jetson_ssh(args, "tail -15 ~/lekiwi_host.log 2>&1")
    print("           ✗ 30초 내 리슨 안 함. Jetson 로그 마지막:")
    for line in log.strip().splitlines()[-8:]:
        print(f"             {line}")
    return False


# ─────────────────────── 모델 파일 확인 ───────────────────────
# 모델은 lhwdev 계정이 학습·복사하고 우리는 lerobot 으로 읽기만 한다. 복사 방식에 따라
# 원본 권한이 그대로 따라와 lerobot 이 못 읽는 일이 생긴다(2026-08-05 classifier 가
# -rw------- 로 들어와 롤아웃이 죽었다. train/ 에 default ACL 이 걸려 있지만
# cp -p / rsync -a 는 그 기본값을 덮어써서 재발할 수 있다).
# 폴딩은 파이프라인의 '끝'이라, 여기서 실패하면 수건을 이미 전달받은 뒤다. 그래서
# Enter 를 받기 전에 5개 모델을 전부 확인한다.
CLASSIFIER_REL = "train/towel_fold01_nextlevel"
MODEL_FILES = ("config.json", "model.safetensors")


def _safetensors_ok(p: Path) -> str:
    """safetensors 가 '끝까지' 있는지 본다. 헤더에 적힌 마지막 오프셋과 실제 크기를 비교.

    복사가 중간에 끊긴 파일은 열리기는 하고 로드할 때야 터진다. 실제로 train/ 아래에
    model.safetensors.tmp 가 남아 있던 적이 있어서(2026-08-05) 크기까지 확인한다.
    빈 문자열이면 정상, 아니면 사유.
    """
    try:
        with p.open("rb") as f:
            n = int.from_bytes(f.read(8), "little")
            if n <= 0 or n > 100_000_000:
                return f"헤더 길이가 이상함({n})"
            head = json.loads(f.read(n).decode("utf-8"))
        end = max((v["data_offsets"][1] for v in head.values()
                   if isinstance(v, dict) and "data_offsets" in v), default=0)
        want, have = 8 + n + end, p.stat().st_size
        if want != have:
            return f"잘림 — {have:,}바이트인데 {want:,}바이트여야 함"
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return ""


def _make_readable(p: Path) -> bool:
    """읽을 수 있게 만든다. 파일 주인이 우리가 아니면 chmod 가 안 되니 sudo -n 도 시도한다.
    (-n 이라 비밀번호를 물으며 멈추지 않는다 — 안 되면 그냥 실패로 두고 사람에게 알린다)"""
    for fn in (
        lambda: os.chmod(p, p.stat().st_mode | 0o444),
        lambda: subprocess.run(["sudo", "-n", "chmod", "a+r", str(p)],
                               check=True, capture_output=True, timeout=10),
    ):
        try:
            fn()
            if os.access(p, os.R_OK):
                return True
        except Exception:
            pass
    return False


def _model_paths(args):
    """rollout_auto.py 가 실제로 여는 경로를 그대로 계산한다(하드코딩하지 않는다).
    --no_step0 이면 step0 은 아예 안 열리므로 확인 대상에서도 뺀다."""
    cdir = Path(os.path.expanduser(args.csi_dir)) / "clothing"
    name, idxs = "towel_fold01", [0, 1, 2, 3]
    info = cdir / "clothing_info.py"
    try:
        src = info.read_text()
        m = re.search(r'clothing_name\s*=\s*"([^"]+)"', src)
        if m:
            name = m.group(1)
        found = sorted({int(i) for i in re.findall(r'_step(\d+)"', src)})
        if found:
            idxs = found
    except Exception:
        pass  # 못 읽으면 기본값으로 진행 — 경로가 틀리면 어차피 아래에서 '없음' 으로 잡힌다
    start = 1 if "--no_step0" in args.csi_cmd else 0
    out = [(f"step{i}", cdir.parent / "train" / f"rollout_hil_{name}_step{i}" / "checkpoints" / "last")
           for i in idxs if i >= start]
    out.append(("classifier", cdir.parent / CLASSIFIER_REL))
    return out


def check_models(args) -> bool:
    ok = True
    print("  모델    : 폴딩 정책/분류기 파일 확인")
    for label, d in _model_paths(args):
        # checkpoints/last 는 심볼릭 링크다. 학습을 다시 돌리면 없는 번호를 가리킨 채
        # 남아 있는 일이 있다(실제로 step0 이 지워진 074740 을 가리키고 있었다).
        if d.is_symlink() and not d.exists():
            print(f"           ✗ {label}: 링크가 끊김 {d.name} → {os.readlink(d)}")
            ok = False
            continue
        if not d.is_dir():
            print(f"           ✗ {label}: 경로 없음 {d}")
            ok = False
            continue
        bad = []
        for fn in MODEL_FILES:
            f = d / fn
            if not f.exists():
                bad.append(f"{fn} 없음")
                continue
            if not os.access(f, os.R_OK):
                if _make_readable(f):
                    print(f"           ⚠️ {label}/{fn}: 읽기 권한이 없어 a+r 로 고쳤습니다")
                else:
                    st = f.stat()
                    bad.append(f"{fn} 읽기 불가 (mode={oct(st.st_mode)[-4:]} uid={st.st_uid})")
                    continue
            if fn.endswith(".safetensors"):
                why = _safetensors_ok(f)
                if why:
                    bad.append(f"{fn} {why}")
        if bad:
            ok = False
            for b in bad:
                print(f"           ✗ {label}: {b}")
        else:
            tgt = os.readlink(d) if d.is_symlink() else d.name
            size = (d / "model.safetensors").stat().st_size
            print(f"           {label:<10} OK  ({tgt}, {size / 1e6:.0f}MB)")
    if not ok:
        print("           소유자가 다른 계정이라 못 고치는 경우 본체에서:")
        print("             sudo chmod -R a+rX /home/lerobot/lerobot2/csi-agent/lhwdev/train")
    return ok


IDLE_POSTURE = "idle_posture.json"


def home_arms(args):
    """SO101 양팔을 기준 자세(idle_posture.json)로 세운다. 카메라 배정 '전에' 한다.

    csi-agent 의 RobotHoming 은 clothing/idle_posture.json 을 찾는데 그 파일이 아예
    없어서(2026-08-05 확인) 홈 복귀가 한 번도 돌지 않았다 — '팔이 제자리로 안 잡힌다'의 원인.
    파일은 examples/lekiwi/record_idle_posture.py 로 만든다.

    카메라를 열지 않는 SOFollower 를 직접 쓴다. setup_devices 의 robot 을 쓰면 카메라까지
    함께 열려서, 자세를 잡기도 전에 카메라 문제로 실패한다.
    """
    if args.no_home:
        return True
    conf = Path(os.path.expanduser(args.csi_dir)) / "clothing" / IDLE_POSTURE
    if not conf.exists():
        print(f"  홈자세  : ⚠️ {IDLE_POSTURE} 없음 — 건너뜀 "
              "(examples/lekiwi/record_idle_posture.py 로 현재 자세를 저장할 수 있습니다)")
        return False
    try:
        target = json.loads(conf.read_text())["joint_positions"]
    except Exception as e:
        print(f"  홈자세  : ⚠️ {IDLE_POSTURE} 읽기 실패({e}) — 건너뜀")
        return False

    try:
        sys.path.insert(0, str(REPO / "examples" / "lekiwi"))
        from record_idle_posture import connect_with_retry, mk_arms
    except Exception as e:
        print(f"  홈자세  : ⚠️ 모듈 로드 실패({e}) — 건너뜀")
        return False

    print(f"  홈자세  : 양팔을 기준 자세로 이동 ({args.home_seconds:.1f}s)")
    ok = True
    for side, arm in mk_arms().items():
        try:
            connect_with_retry(arm, attempts=args.csi_retries, delay=3.0)
            obs = arm.get_observation()
            keys = [k for k in obs if k.endswith(".pos")]
            start = {k: float(obs[k]) for k in keys}
            goal = {k: float(target[f"{side}_{k}"]) for k in keys if f"{side}_{k}" in target}
            n = max(1, int(args.home_seconds * 30))
            for i in range(1, n + 1):
                # 코사인 프로파일 — 시작/끝에서 부드럽게(csi-agent RobotHoming 과 동일한 방식)
                t = (1.0 - math.cos(math.pi * i / n)) / 2.0
                arm.send_action({k: start[k] + (goal[k] - start[k]) * t for k in goal})
                time.sleep(1 / 30)
            err = max(abs(float(arm.get_observation()[k]) - goal[k]) for k in goal)
            print(f"           {side:<5} 도달 (최대 오차 {err:.1f})")
            if err > 5.0:
                print(f"           ⚠️ {side} 오차가 큽니다 — 걸림/과부하 확인")
                ok = False
            arm.disconnect()
        except Exception as e:
            print(f"           ✗ {side} 실패: {str(e).splitlines()[0][:110]}")
            ok = False
    return ok


def _resolve_cams_once(args):
    """한 번만 배정해 본다. (성공여부, 사유) 를 돌려준다."""
    sys.path.insert(0, str(REPO / "examples" / "lekiwi"))
    from resolve_cameras import (resolve as _resolve_cams,
                                 wait_released as _wait_released,
                                 write_links as _write_cam_links)
    try:
        mapping, _cams, why = _resolve_cams(verbose=False)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    need = ("top", "left_cam", "right_cam")
    if not all(k in mapping for k in need):
        return False, ", ".join(w for w in why if "⚠️" in w) or "카메라 부족"
    _write_cam_links(mapping)
    # 배정하느라 우리가 4대를 열었다 닫았다. V4L2 는 close 직후 곧바로 재오픈되지 않는
    # 경우가 있어, 넘겨주기 전에 다시 열리는지 확인한다.
    # (이걸 빼먹어 롤아웃이 device busy 로 죽는 일이 있었다)
    _wait_released([mapping[k] for k in need], timeout=15.0, verbose=False)
    print("  폴딩캠 : " + "  ".join(f"{k}→{os.path.basename(mapping[k])}" for k in need))
    return True, ""


def resolve_folding_cams(args):
    """폴딩 캠 3대를 모델+화면내용으로 배정한다. LeKiwi 단계와 무관하므로
    --skip-task3(폴딩만 실행)에서도 반드시 돌아야 한다.

    udev 이름(/dev/lerobot/camera_N)은 '물리 포트'로 붙는다. 카메라 4대가 2쌍씩 벤더·모델·
    시리얼이 완전히 같아 udev 단서가 포트뿐이기 때문이다. 자리를 옮기면 이름이 뒤바뀐다
    (2026-08-05: 폴딩 top 이 세탁기 상공캠을 가리킨 채로 돌았다).
    그래서 매 실행마다 모델+화면내용으로 다시 배정한다.

    배정에 실패하면 USB 전원 리셋 후 다시 해 본다 — 카메라가 USB 에는 붙어 있는데
    드라이버가 안 잡혀 /dev/video 노드가 아예 없는 상태가 실제로 생긴다(뽑았다 꽂으면 낫는다).
    """
    if args.no_cam_resolve:
        return True
    sys.path.insert(0, str(REPO / "examples" / "lekiwi"))
    import usb_reset

    for i in range(1, args.cam_resets + 2):
        # 먼저 USB 쪽이 멀쩡한지 본다. 노드가 없으면 배정은 해 볼 것도 없다.
        # (첫 바퀴는 문제 있을 때만 리셋 — 멀쩡한 걸 흔들면 devnum 만 바뀐다)
        if not usb_reset.recover(expect=args.cam_expect, attempts=1, verbose=(i > 1)):
            why = "USB 인식 실패"
        else:
            ok, why = _resolve_cams_once(args)
            if ok:
                if i > 1:
                    print(f"           (USB 리셋 {i - 1}회 후 성공)")
                return True
        if i > args.cam_resets:
            break
        print(f"  폴딩캠 : 배정 실패({why}) — USB 전원 리셋 후 재시도 {i}/{args.cam_resets}")
        usb_reset.recover(expect=args.cam_expect, attempts=1, verbose=True, force=True)

    print(f"  폴딩캠 : ✗ USB 리셋 {args.cam_resets}회 후에도 실패 — 케이블/허브를 확인하세요")
    return False


# ─────────────────────── [B] 로컬 task3 실행 ───────────────────────
def local_preflight(args):
    """LeKiwi 쪽 장비 확인. 경로는 본체마다 다르므로 '없으면 경고'만 하고 진행 여부는 사용자가 정한다."""
    print("─" * 60)
    print("[A2] 로컬 preflight: LeKiwi 장비 확인 (경로는 본체마다 다름)")
    ok = True
    py = local_python()
    print(f"  python : {py}{'' if os.path.exists(py) else '  ✗ 없음'}")
    ok &= os.path.exists(py)

    # 상공캠: 번호(/dev/videoN)는 본체마다·재연결마다 바뀌므로 실제로 프레임이 나오는 노드를 찾는다.
    print(f"  상공캠 : 탐색 중 (지정={args.overhead_cam}"
          + (f", match='{args.overhead_match}'" if args.overhead_match else "") + ")")
    sys.path.insert(0, str(REPO / "examples" / "lekiwi"))
    from find_overhead_cam import resolve_overhead_cam  # cv2 만 쓰는 가벼운 모듈

    cam, log = resolve_overhead_cam(args.overhead_cam, args.overhead_match)
    for line in log:
        print(f"           {line}")
    if cam:
        print(f"           → 사용: {cam}")
        args.overhead_cam = cam  # task3 에는 확정된 경로를 넘긴다
    else:
        ok = False
        print("           ✗ 상공 카메라를 못 찾음 (USB 연결 확인)")

    # Jetson: 링크 → host 순으로 '확인하고, 안 되면 되살린다'. 사람이 미리 띄워둘 필요가 없다.
    if args.jetson_ip:
        link = ensure_jetson_link(args)
        if link:
            print(f"  Jetson : {args.jetson_ip}  OK")
            ok &= ensure_jetson_host(args) if not args.no_host_autostart else True
        else:
            ok = False
    else:
        print("  Jetson : (미지정 — task3 기본값/--wireless 에 맡김)")

    if args.handoff_motion:
        m = REPO / "motions" / f"{args.handoff_motion}.json"
        print(f"  전달모션: {m}  {'OK' if m.exists() else '✗ 없음'}")
        ok &= m.exists()
    else:
        print("  전달모션: (미설정 — throw 후 후퇴까지만 하고 끝남)")
    return ok


def run_task3(args, passthrough):
    print("─" * 60)
    print(f"[B] LeKiwi 세탁물 회수: {args.task3}")
    env = dict(os.environ)
    # stdout 이 파이프면 파이썬이 블록 버퍼링을 한다 → flush=True 가 없는 print 는 수 KB 쌓일
    # 때까지 화면에 안 나온다. approach 진행 로그처럼 '실시간으로 봐야 하는' 출력이 뭉쳐서
    # 나오면 쓸모가 없으므로 버퍼링을 끈다.
    env["PYTHONUNBUFFERED"] = "1"
    env["OVERHEAD_CAM"] = args.overhead_cam          # 본체마다 다른 카메라 경로를 여기서 주입
    if args.jetson_ip:
        env["REMOTE_IP"] = args.jetson_ip            # 지정 시 task3 의 유/무선 기본값보다 우선
    if args.handoff_motion:
        env["HANDOFF_MOTION"] = args.handoff_motion  # 마지막 후퇴 뒤 전달 모션
    cmd = [local_python(), str(args.task3), *passthrough]
    rc = run_streaming(cmd, "  | ", cwd=str(REPO), env=env)
    codes = {0: "성공", 3: "approach 실패", 4: "문열기 실패", 5: "사용자 abort",
             6: "throw 모션 실패", 7: "전달 모션 실패", 8: "상공 카메라 못 찾음", 130: "Ctrl+C 중단"}
    print(f"  → task3 exit {rc} ({codes.get(rc, '알 수 없는 실패')})")
    return rc


# ─────────────────────── main ───────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="LeKiwi 세탁물 회수(laundry_task3) → SO101 수건개기(csi-agent) 이어붙이기.",
        epilog="`--` 뒤의 인자는 전부 laundry_task3.py 로 그대로 넘어갑니다.")
    ap.add_argument("--csi-host", default=CSI_HOST,
                    help="SO101 본체 SSH 대상 (user@host). 'local' 이면 SSH 없이 이 머신에서 실행. "
                         "csi-agent 가 로컬에 있으면 자동으로 local 이 된다. env CSI_HOST")
    ap.add_argument("--csi-local", action="store_true", help="SO101 단계를 이 머신에서 직접 실행(SSH 안 씀)")
    ap.add_argument("--csi-dir", default=CSI_DIR, help="원격 csi-agent 저장소 경로 (기본 ~/csi-agent)")
    ap.add_argument("--csi-python", default=CSI_PYTHON, help="원격 파이썬 (기본 conda lerobot env)")
    ap.add_argument("--csi-cmd", default=CSI_CMD,
                    help="clothing/ 기준 실행 커맨드 (기본 rollout_auto.py --no_step0)")
    ap.add_argument("--csi-with-step0", action="store_true",
                    help="step0(펼치기)까지 포함. 기본은 --no_step0 — step0 정책이 불완전해 제외하고, "
                         "'펴져 있음'이 인식될 때까지 IDLE 에서 기다렸다 접기(step1)부터 시작한다")
    ap.add_argument("--csi-immediate", action="store_true",
                    help="classifier 대기 없이 즉시 접기 시작(--immediate_start). prewarm 과 같이 못 쓴다")
    ap.add_argument("--csi-env", default=CSI_ENV, help='원격 export 할 env (예: "DISPLAY=:0")')
    ap.add_argument("--task3", type=Path, default=DEFAULT_TASK3, help="실행할 laundry_task3.py 경로")
    ap.add_argument("--overhead-cam", default=OVERHEAD_CAM,
                    help="상공 카메라. 기본 'auto' — 번호(/dev/videoN)는 본체마다·재연결마다 바뀌므로 "
                         "실제로 프레임이 나오는 USB 노드를 찾는다. 못 박으려면 /dev/v4l/by-id/... 사용. "
                         "env OVERHEAD_CAM")
    ap.add_argument("--overhead-match", default=OVERHEAD_MATCH,
                    help="USB 캠이 여러 개일 때 이름/by-id 문자열로 좁히기(예: C920). env OVERHEAD_MATCH")
    ap.add_argument("--jetson-ip", default=JETSON_IP,
                    help="LeKiwi Jetson IP. ⚠️ 본체/네트워크마다 다름(유선 192.168.55.1 / 무선 192.168.0.19). "
                         "비우면 task3 기본값·--wireless 를 따름. env REMOTE_IP")
    ap.add_argument("--jetson-user", default=JETSON_USER,
                    help="Jetson 로그인 계정(host 자동 기동용, 키 인증 필요). env JETSON_USER")
    ap.add_argument("--usbnet-unit", default=USBNET_UNIT,
                    help="USB 이더넷 정적 IP systemd unit — Jetson 이 ping 에 응답 안 할 때만 재시작. env USBNET_UNIT")
    ap.add_argument("--no-host-autostart", action="store_true",
                    help="lekiwi_host 자동 기동 안 함(이미 직접 띄운 경우)")
    ap.add_argument("--no-grip-regs", action="store_true",
                    help="6번 순응제어 레지스터(P게인·토크상한) 자동 설정 안 함")
    ap.add_argument("--no-cam-resolve", action="store_true",
                    help="폴딩 캠 자동 배정 안 함(udev 이름을 그대로 씀)")
    ap.add_argument("--cam-resets", type=int, default=int(os.environ.get("CAM_RESETS", "5")),
                    help="카메라 인식 실패 시 USB 전원 리셋을 걸 최대 횟수(기본 5). env CAM_RESETS")
    ap.add_argument("--cam-expect", type=int, default=int(os.environ.get("CAM_EXPECT", "4")),
                    help="있어야 하는 카메라 대수(폴딩 3 + 세탁기 상공 1 = 4). env CAM_EXPECT")
    ap.add_argument("--no-home", action="store_true",
                    help="카메라 배정 전 양팔 홈 자세 이동을 건너뜀")
    ap.add_argument("--home-seconds", type=float, default=float(os.environ.get("HOME_SECONDS", "3.0")),
                    help="홈 자세로 이동하는 시간(초). env HOME_SECONDS")
    ap.add_argument("--csi-retries", type=int, default=int(os.environ.get("CSI_RETRIES", "10")),
                    help="모터 인식 실패 시 수건개기 프로세스를 다시 띄울 최대 횟수(기본 10). env CSI_RETRIES")
    ap.add_argument("--csi-retry-delay", type=float, default=float(os.environ.get("CSI_RETRY_DELAY", "5")),
                    help="재시도 전 대기 초(기본 5). 버스가 정리될 시간을 준다. env CSI_RETRY_DELAY")
    ap.add_argument("--handoff-motion", default=HANDOFF_MOTION,
                    help="마지막 후퇴 뒤 재생할 전달 모션 이름(motions/<이름>.json). 비우면 생략. env HANDOFF_MOTION")
    ap.add_argument("--check", action="store_true", help="preflight 만 하고 종료(로봇 안 움직임)")
    ap.add_argument("--skip-task3", action="store_true", help="LeKiwi 단계 건너뛰고 SO101 만 실행")
    ap.add_argument("--skip-csi", action="store_true", help="SO101 단계 건너뛰기(task3 만)")
    ap.add_argument("--skip-preflight", action="store_true", help="preflight 생략")
    ap.add_argument("--force-csi", action="store_true", help="원격 preflight 실패/ task3 실패에도 SO101 실행")
    ap.add_argument("--force-local", action="store_true", help="로컬 preflight 실패해도 task3 강행")
    ap.add_argument("--no-prewarm", dest="prewarm", action="store_false",
                    help="수건개기 모델을 미리 올리지 않는다(=task3 끝난 뒤 로딩. 스톨 생김)")
    ap.add_argument("--pause", action="store_true",
                    help="task3 종료 후 Enter 를 한 번 더 받는다(기본은 Enter 한 번뿐)")
    ap.add_argument("--handoff-wait", type=float, default=float(os.environ.get("HANDOFF_WAIT", "0")),
                    help="전달 모션 후 SO101 시작까지 대기 초(기본 0)")
    ap.add_argument("--no-tty", action="store_true", help="원격에 TTY 할당 안 함(비대화 실행)")
    args, passthrough = ap.parse_known_args()

    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    # 이 머신이 곧 SO101 본체면(csi-agent 가 로컬에 있으면) SSH 를 돌 이유가 없다.
    # 실제 시연은 SO101 본체 한 대에서 전 과정을 돌리므로 이게 기본 동작이 된다.
    if args.csi_local:
        args.csi_host = "local"
    elif "--csi-host" not in sys.argv and not csi_is_local(args):
        if (Path(os.path.expanduser(args.csi_dir)) / "clothing").is_dir():
            print("  ℹ️ 이 머신에 csi-agent 가 있어 로컬 모드로 전환합니다 (SSH 안 씀). "
                  "원격으로 강제하려면 --csi-host 를 명시하세요.")
            args.csi_host = "local"

    # 원격 rollout 플래그 조정
    if args.csi_with_step0:
        args.csi_cmd = args.csi_cmd.replace("--no_step0", "").strip()
    if args.csi_immediate and "--immediate_start" not in args.csi_cmd:
        args.csi_cmd += " --immediate_start"
    # prewarm 이 켜져 있으면(기본) 즉시시작은 성립하지 않는다 — 미리 띄우는 이상
    # --immediate_start 는 수건이 도착하기도 전에 팔을 움직인다. IDLE 대기를 강제한다.
    if args.prewarm and not args.skip_csi and not args.skip_task3:
        if "--immediate_start" in args.csi_cmd:
            print("  ℹ️ prewarm 과 --csi-immediate 는 같이 못 씁니다 → IDLE 대기로 진행 "
                  "(즉시 시작하려면 --no-prewarm 을 함께 주세요)")
        args.csi_cmd = args.csi_cmd.replace("--immediate_start", "").strip()

    if not args.task3.exists():
        print(f"✗ task3 스크립트 없음: {args.task3}")
        return 2

    print("=" * 60)
    print("세탁 파이프라인:  LeKiwi(laundry_task3) → SO101(csi-agent 수건개기)")
    print(f"  task3    : {args.task3}{' [skip]' if args.skip_task3 else ''}")
    print(f"  task3 인자: {' '.join(passthrough) if passthrough else '(없음)'}")
    print(f"  상공캠   : {args.overhead_cam}   Jetson: {args.jetson_ip or '(task3 기본값)'}")
    _where = "로컬" if csi_is_local(args) else args.csi_host
    print(f"  SO101    : {_where}:{args.csi_dir}/clothing "
          f"→ {args.csi_cmd}{' [skip]' if args.skip_csi else ''}")

    # [A] preflight — 로봇이 움직이기 전에 원격/로컬 장비부터 확인
    remote_ok = local_ok = True
    if not args.skip_preflight:
        if not args.skip_csi:
            remote_ok = preflight(args)
        if not args.skip_task3:
            local_ok = local_preflight(args)
    if args.check:
        print("─" * 60)
        print(f"[check] 원격={'OK' if remote_ok else 'NG'}  로컬={'OK' if local_ok else 'NG'} (로봇은 움직이지 않음)")
        return 0 if (remote_ok and local_ok) else 1
    if not args.skip_preflight:
        if not remote_ok and not args.force_csi:
            print("\n✗ 원격 준비가 안 됐습니다. 수건을 전달해놓고 못 넘기는 상황을 막기 위해 중단합니다.")
            print("  (원격 설정 전에 LeKiwi 만 돌리려면 --skip-csi, 무시하고 강행하려면 --force-csi)")
            return 1
        if not local_ok and not args.force_local:
            print("\n✗ 로컬 장비 확인 실패(카메라/Jetson 경로는 본체마다 다릅니다).")
            print("  --overhead-cam / --jetson-ip 로 이 본체의 실제 경로를 지정하거나, --force-local 로 강행하세요.")
            return 1

    # [B0] prewarm — 무거운 로딩(SmolVLA·probe / ACT 4단계·classifier)을 전부 첫 Enter '앞'으로.
    #      task3 는 자기 preload 를 끝낸 뒤에야 Enter 를 묻고, 수건개기는 그 사이 IDLE 로 올라온다.
    #      Enter 이후에는 추가 입력도, 로딩 스톨도 없다.
    warm = None
    if args.prewarm and not args.skip_csi and not args.skip_task3:
        warm = start_prewarm(args)

    # [B] LeKiwi
    rc3 = 0
    if not args.skip_task3:
        try:
            rc3 = run_task3(args, passthrough)
        except KeyboardInterrupt:
            if warm:
                warm.stop("파이프라인 중단")
            print("\n[중단] 파이프라인 Ctrl+C")
            return 130
        if rc3 != 0 and not args.force_csi:
            if warm:
                warm.stop("task3 실패 — 수건이 오지 않음")
            print("\n✗ task3 가 성공하지 못해 SO101 단계를 시작하지 않습니다 (--force-csi 로 강행 가능).")
            return rc3

    # [C] SO101
    if args.skip_csi:
        print("\n✅ task3 까지 완료 (SO101 단계는 --skip-csi 로 생략).")
        return rc3
    try:
        if args.handoff_wait > 0:
            print(f"  ⏳ 전달 후 {args.handoff_wait}s 대기...")
            time.sleep(args.handoff_wait)
        if args.pause:
            if warm is not None:
                print("  (참고: 미리 띄운 수건개기는 classifier 가 수건을 인식하면 이 Enter 와")
                print("   무관하게 이미 시작했을 수 있습니다. 이 Enter 는 로그 출력을 여는 것입니다.)")
            input("  >>> 계속하려면 Enter: ")
    except KeyboardInterrupt:
        if warm:
            warm.stop("파이프라인 중단")
        print("\n[중단] 파이프라인 Ctrl+C")
        return 130

    try:
        if warm is not None:
            print("─" * 60)
            if warm.alive():
                print("[C] SO101 수건개기 — 이미 로딩 완료된 프로세스가 이어받습니다(로딩 스톨 없음)")
            else:
                print("[C] ⚠️ 미리 띄운 수건개기 프로세스가 이미 종료됨 — 로그를 확인하세요")
            rc_csi = warm.follow("  ▸ ")
            if rc_csi == 0:
                print("  ✅ 수건개기 완료")
            elif rc_csi == 130:
                print("  ✗ 사용자 중단")
            else:
                # prewarm 경로에는 재시도가 없었다. 미리 띄워 둔 프로세스가 실패하면
                # 그대로 끝나 버려서, 모터·카메라가 '가끔' 안 잡히는 것에 무방비였다.
                # 여기서 run_csi 의 재시도 루프로 넘긴다. 넘기기 전에 그 프로세스를 확실히
                # 죽여야 한다 — 살아 있으면 카메라를 쥔 채라 새 프로세스가 device busy 로 죽는다.
                print(f"  ✗ 미리 띄운 프로세스 실패 (exit {rc_csi}) — 다시 로딩해 재시도합니다")
                warm.stop("실패 — 재시도를 위해 정리")
                _wait_cams_free(args)
                rc_csi = run_csi(args)
        else:
            rc_csi = run_csi(args)
    except KeyboardInterrupt:
        print("\n[중단] 파이프라인 Ctrl+C")
        return 130

    if rc_csi == 0 and rc3 == 0:
        print("\n✅ 전체 완료: 세탁물 회수 → 전달 → 수건개기")
    return rc_csi


if __name__ == "__main__":
    sys.exit(main())
