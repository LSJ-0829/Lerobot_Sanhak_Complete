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
import os
import shlex
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
# rollout_auto.py 는 기본적으로 IDLE 에서 classifier 가 '시작해도 되는 상태'라고 판단할 때까지 기다린다.
# 우리는 task3 가 exit 0(집기 확정 + 전달까지 완료)일 때만 여기 오므로 --immediate_start 로 바로 시작한다.
# 기다렸다 시작하게 하려면 --csi-idle (이러면 수건이 보일 때까지 팔이 안 움직여서 더 안전하다).
CSI_CMD = os.environ.get("CSI_CMD", "scripts/rollout_auto.py --immediate_start")
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
JETSON_IP = os.environ.get("REMOTE_IP", "")           # 비우면 task3 의 --wireless/기본값에 맡김
HANDOFF_MOTION = os.environ.get("HANDOFF_MOTION", "")  # motions/<이름>.json, 비우면 전달 모션 생략

SSH_OPTS = ["-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4"]


def local_python() -> str:
    """conda lerobot env 파이썬(이 머신엔 uv 가 없다). 없으면 현재 인터프리터."""
    if LOCAL_PYTHON:
        return os.path.expanduser(LOCAL_PYTHON)
    cand = Path.home() / "miniforge3" / "envs" / "lerobot" / "bin" / "python"
    return str(cand) if cand.exists() else sys.executable


def run_streaming(cmd, prefix, cwd=None, env=None, show=None):
    """자식 프로세스 출력을 prefix 붙여 실시간으로 흘리고 returncode 를 돌려준다.

    show 를 주면 그 문자열을 대신 표시한다(ssh 원격 스크립트는 따옴표 때문에 원문이 읽기 어렵다).
    """
    print(f"  $ {show or ' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in p.stdout:
            print(f"{prefix}{line.rstrip()}", flush=True)
    except KeyboardInterrupt:
        print(f"\n{prefix}[중단] Ctrl+C → 자식 프로세스 종료 중...", flush=True)
        p.terminate()
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
        raise
    finally:
        if p.stdout:
            p.stdout.close()
    return p.wait()


# ─────────────────────── [C] 원격(csi-agent) 실행 ───────────────────────
def remote_script(args, preflight_only=False):
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
        lines.append(f'exec "$CSI_PY" {args.csi_cmd}')
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
        self.p = subprocess.Popen(cmd, cwd=cwd, stdout=self._fh, stderr=subprocess.STDOUT)

    def alive(self):
        return self.p.poll() is None

    def stop(self, why=""):
        if self.alive():
            print(f"  ⏹ 미리 띄운 수건개기 종료{(' — ' + why) if why else ''}")
            self.p.terminate()
            try:
                self.p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.p.kill()
        self._fh.close()

    def follow(self, prefix):
        """지금까지의 로그를 뱉고, 프로세스가 끝날 때까지 이어서 흘린다."""
        try:
            with open(self.logpath, "r", errors="replace") as f:
                while True:
                    line = f.readline()
                    if line:
                        print(f"{prefix}{line.rstrip()}", flush=True)
                        continue
                    if self.p.poll() is not None:
                        rest = f.read()
                        for ln in rest.splitlines():
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


def local_csi_cmd(args):
    """SSH 없이 로컬에서 돌릴 커맨드 (CWD=<csi_dir>/clothing)."""
    py = os.path.expanduser(args.csi_python)
    return [py, *shlex.split(args.csi_cmd)], os.path.join(os.path.expanduser(args.csi_dir), "clothing")


def start_prewarm(args):
    """수건개기 rollout 을 IDLE 모드로 미리 띄운다(모델 로딩을 첫 Enter 앞으로 당기기)."""
    idle_cmd = args.csi_cmd.replace("--immediate_start", "").strip()
    log = os.path.join(tempfile.gettempdir(), f"laundry_task4_csi_{os.getpid()}.log")
    print("─" * 60)
    print("[B0] 수건개기 정책 미리 로딩 (IDLE 대기) — task3 preload 와 동시에 진행")
    print(f"     `{idle_cmd}`   로그: {log}")
    if csi_is_local(args):
        cmd = [os.path.expanduser(args.csi_python), *shlex.split(idle_cmd)]
        cwd = os.path.join(os.path.expanduser(args.csi_dir), "clothing")
    else:
        saved, args.csi_cmd = args.csi_cmd, idle_cmd
        cmd, cwd = ssh_cmd(args, remote_script(args)), None
        args.csi_cmd = saved
    return Prewarm(cmd, cwd, log, "csi")


def preflight(args):
    print("─" * 60)
    where = "이 머신(로컬)" if csi_is_local(args) else args.csi_host
    print(f"[A] preflight: SO101 쪽 확인 — {where}")

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


def run_csi(args):
    print("─" * 60)
    if csi_is_local(args):
        cmd, cwd = local_csi_cmd(args)
        print(f"[C] SO101 수건개기(로컬): {cwd} 에서 `{args.csi_cmd}`")
        rc = run_streaming(cmd, "  ▸ ", cwd=cwd)
    else:
        print(f"[C] SO101 수건개기: {args.csi_host}:{args.csi_dir}/clothing 에서 `{args.csi_cmd}`")
        tty = sys.stdin.isatty() and not args.no_tty
        rc = run_streaming(ssh_cmd(args, remote_script(args), tty=tty), "  ▸ ",
                           show=f"ssh {args.csi_host} 'cd {args.csi_dir}/clothing && "
                                f"{args.csi_python} {args.csi_cmd}'")
    if rc == 0:
        print("  ✅ 수건개기 완료")
    else:
        print(f"  ✗ 수건개기 실패/중단 (exit {rc})")
    return rc


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

    if args.jetson_ip:
        r = subprocess.run(["ping", "-c", "1", "-W", "2", args.jetson_ip],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  Jetson : {args.jetson_ip}  {'OK' if r.returncode == 0 else '✗ 응답 없음'}")
        ok &= r.returncode == 0
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
                    help="clothing/ 기준 실행 커맨드 (기본 rollout_auto.py --immediate_start)")
    ap.add_argument("--csi-idle", action="store_true",
                    help="--immediate_start 를 빼고 IDLE 에서 classifier 판단을 기다리게 한다(더 안전)")
    ap.add_argument("--csi-no-step0", action="store_true",
                    help="step0(펼치기) 정책 없이 step1 부터 실행(rollout_auto --no_step0)")
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

    # 원격 rollout 플래그 조정.
    # prewarm 이 켜져 있으면(기본) IDLE 이 강제된다 — 미리 띄우는 이상 --immediate_start 를 쓰면
    # 수건이 도착하기도 전에 팔이 움직인다. 대신 로딩이 전부 첫 Enter 앞에서 끝난다.
    if args.csi_idle or (args.prewarm and not args.skip_csi and not args.skip_task3):
        args.csi_cmd = args.csi_cmd.replace("--immediate_start", "").strip()
    if args.csi_no_step0 and "--no_step0" not in args.csi_cmd:
        args.csi_cmd += " --no_step0"

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
    if args.handoff_wait > 0:
        print(f"  ⏳ 전달 후 {args.handoff_wait}s 대기...")
        time.sleep(args.handoff_wait)
    if args.pause:
        input("SO101 수건개기를 시작하려면 Enter: ")

    try:
        if warm is not None:
            print("─" * 60)
            if warm.alive():
                print("[C] SO101 수건개기 — 이미 로딩 완료된 프로세스가 이어받습니다(로딩 스톨 없음)")
            else:
                print("[C] ⚠️ 미리 띄운 수건개기 프로세스가 이미 종료됨 — 로그를 확인하세요")
            rc_csi = warm.follow("  ▸ ")
            print("  ✅ 수건개기 완료" if rc_csi == 0 else f"  ✗ 수건개기 실패/중단 (exit {rc_csi})")
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
