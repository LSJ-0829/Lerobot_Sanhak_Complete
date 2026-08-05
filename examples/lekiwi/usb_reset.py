"""카메라가 USB 에 붙어 있는데 인식되지 않을 때 전원 리셋으로 되살린다.

증상(2026-08-05 실제로 걸림): USB 에는 4대가 다 열거되는데 그중 2대는
`bConfigurationValue` 가 비어 있고 uvcvideo 가 붙지 않아 `/dev/video*` 노드가 아예
생기지 않는다. 케이블을 뽑았다 꽂으면 낫는다 — 즉 커널이 설정(configuration)을
고르지 못한 채로 열거만 끝낸 상태다. 뽑았다 꽂는 것과 같은 일을 소프트웨어로 한다.

방법은 두 가지고, 순서대로 시도한다.

1. `authorized` 0→1 — 장치를 껐다 켠 것과 사실상 같다(재열거 + 설정 재선택 + 드라이버 재바인딩).
   **이것만이 위 증상을 고친다.** 실측: 드라이버 미바인딩 2대가 1회 토글로 전부 복구.
2. `USBDEVFS_RESET` ioctl — 포트 리셋만 한다. 위 증상에는 듣지 않았지만(설정이 여전히 비어 있었다),
   장치가 응답만 안 하는 흔한 경우에는 이쪽이 덜 거칠어서 보조로 남겨둔다.

**sudo 가 필요 없다.** 이 본체는 udev 규칙이 카메라의 `authorized` 를 0666,
`/dev/bus/usb/*` 노드를 0666 plugdev 로 열어 둔다(`99-lerobot.rules:47`). 카메라가 아닌
장치(Jetson, 모터 시리얼)는 0644 라 실수로 건드릴 수도 없다 — 그래도 VID:PID 로 한 번 더 거른다.

csi-agent 에도 같은 일을 하는 `tools/usb_reset.py` 가 있고, `setup_devices.py` 와
`rollout.py` 가 연결에 실패했을 때 `reset_usb_cameras(power_cycle=True, reset_all=True)` 로
부른다. 다만 그건 **실행 중에 실패한 뒤**에만 돌고, 포트를 `["1-3","1-5","1-7"]` 로 박아
두었다 — 로봇이 이동해 USB 자리가 바뀌면 엉뚱한 포트를 리셋한다(현재도 4번째 카메라
1-6.3 은 그 목록에 없다). 여기서는 **시작 전에**, 포트가 아니라 VID:PID 로 찾아서 건다.

⚠️ 리셋하면 devnum 과 `/dev/videoN` 번호가 바뀐다. 반드시 뒤이어 카메라를 다시 배정할 것
(`resolve_cameras.py`). 그리고 열려 있는 프로세스가 있으면 그 핸들이 끊긴다.

사용:
  python usb_reset.py            # 상태만 본다
  python usb_reset.py --recover  # 필요하면 리셋해서 되살린다
  python usb_reset.py --force    # 멀쩡해도 4대 전부 리셋
"""
import argparse
import fcntl
import os
import sys
import time
from pathlib import Path

SYSUSB = Path("/sys/bus/usb/devices")

# 카메라 4대가 전부 같은 칩이다(Innomaker 도 속은 Sonix). 그래서 VID:PID 로는 서로
# 구분되지 않지만, '카메라인지 아닌지'를 가리는 데는 이걸로 충분하다.
CAM_IDS = {("0c45", "6366")}

USBDEVFS_RESET = ord("U") << 8 | 20  # _IO('U', 20)

EXPECT_DEFAULT = int(os.environ.get("CAM_EXPECT", "4"))
RESETS_DEFAULT = int(os.environ.get("CAM_RESETS", "5"))

# 전원을 내린 채 두는 시간. 0.5초로도 실측 복구가 됐지만, csi-agent 쪽은 3초를 쓴다
# ("장치 펌웨어가 재부팅될 시간"). 그쪽이 실제로 검증한 값이라 비슷하게 잡는다.
POWER_OFF_SEC = float(os.environ.get("CAM_POWER_OFF_SEC", "2.0"))


def _read(p: Path, default="") -> str:
    try:
        return p.read_text().strip()
    except Exception:
        return default


class Cam:
    def __init__(self, path: Path):
        self.path = path
        self.name = path.name
        self.product = _read(path / "product", "?")

    @property
    def configured(self) -> bool:
        # 설정이 안 골라진 장치는 이 값이 '빈 문자열'이다(0 이 아니라).
        return _read(self.path / "bConfigurationValue") not in ("", "0")

    @property
    def bound(self) -> bool:
        return any((i / "driver").exists() for i in self.path.glob("*:*"))

    @property
    def videos(self):
        out = []
        v4l = Path("/sys/class/video4linux")
        if not v4l.is_dir():
            return out
        for v in sorted(v4l.iterdir()):
            try:
                if str(self.path.resolve()) in str((v / "device").resolve()):
                    out.append(v.name)
            except Exception:
                pass
        return out

    @property
    def ready(self) -> bool:
        # 세 가지가 다 맞아야 '쓸 수 있다'. 노드까지 봐야 하는 이유: 설정과 드라이버가
        # 붙었는데도 노드가 안 생기는 중간 상태를 실제로 봤다.
        return self.configured and self.bound and bool(self.videos)

    @property
    def node(self) -> str:
        b, d = _read(self.path / "busnum"), _read(self.path / "devnum")
        return f"/dev/bus/usb/{int(b):03d}/{int(d):03d}" if b and d else ""

    def desc(self) -> str:
        vids = ",".join(self.videos) or "노드없음"
        return f"{self.name:<8} {self.product:<26} {vids}"

    def reset(self, verbose=True) -> str:
        """되살리기를 시도하고 무엇을 했는지 돌려준다(빈 문자열이면 아무것도 못 했다)."""
        auth = self.path / "authorized"
        try:
            auth.write_text("0")
            time.sleep(POWER_OFF_SEC)   # 내려놓은 채로 두는 시간
            auth.write_text("1")
            return "authorized 0→1"
        except Exception as e:
            first = f"authorized 실패({type(e).__name__})"
        try:
            fd = os.open(self.node, os.O_WRONLY)
            try:
                fcntl.ioctl(fd, USBDEVFS_RESET, 0)
                return f"{first} → 포트 리셋"
            finally:
                os.close(fd)
        except Exception as e:
            if verbose:
                print(f"    ✗ {self.name}: {first}, 포트 리셋도 실패({type(e).__name__}: {e})")
            return ""


def cameras():
    """USB 에 열거된 카메라들. 열거조차 안 된 장치는 여기 안 잡힌다(리셋으로도 못 살린다)."""
    out = []
    if not SYSUSB.is_dir():
        return out
    for d in sorted(SYSUSB.iterdir()):
        if not (d / "idVendor").exists():
            continue
        if (_read(d / "idVendor"), _read(d / "idProduct")) in CAM_IDS:
            out.append(Cam(d))
    return out


def wait_ready(cams, timeout=8.0, poll=0.3) -> bool:
    """리셋 뒤 노드가 다시 생길 때까지 기다린다. 고정 sleep 대신 폴링하는 이유:
    빠르면 1초, 느리면 5초라 고정값을 잡으면 둘 중 하나는 손해다."""
    end = time.time() + timeout
    while time.time() < end:
        # Cam 객체는 sysfs 를 매번 읽으므로 다시 만들 필요는 없지만,
        # 리셋 뒤 경로가 사라졌다 다시 생기는 경우가 있어 새로 열거한다.
        fresh = {c.name: c for c in cameras()}
        if all(fresh.get(c.name) and fresh[c.name].ready for c in cams):
            return True
        time.sleep(poll)
    return False


def settle(extra=1.5):
    """노드가 생긴 '뒤'에도 곧바로 쓸 수 있는 건 아니다.

    udev 가 속성을 채우고 /dev/lerobot 링크를 다시 걸어야 하고, uvc 장치 자체도 첫
    open 을 받기까지 잠깐 걸린다. 이걸 짧게 잡았더니(0.7초) 리셋 직후 배정이 한 번
    실패하고 두 번째에야 됐다 — 리셋 한 바퀴를 통째로 낭비했다.
    """
    try:
        import subprocess
        subprocess.run(["udevadm", "settle", "--timeout=5"],
                       capture_output=True, timeout=8)
    except Exception:
        pass
    time.sleep(extra)


def status(expect=EXPECT_DEFAULT, verbose=True):
    cams = cameras()
    bad = [c for c in cams if not c.ready]
    ok = len(cams) >= expect and not bad
    if verbose:
        for c in cams:
            print(f"    {'OK ' if c.ready else '✗  '}{c.desc()}")
        if len(cams) < expect:
            print(f"    ⚠️ USB 에 {len(cams)}대만 보입니다(기대 {expect}대) — "
                  "안 보이는 건 리셋으로 못 살립니다. 케이블/허브 확인")
    return ok, cams, bad


def recover(expect=EXPECT_DEFAULT, attempts=RESETS_DEFAULT, verbose=True, force=False) -> bool:
    """멀쩡하면 그냥 True. 아니면 최대 attempts 회 리셋하며 되살린다."""
    for i in range(1, attempts + 1):
        ok, cams, bad = status(expect, verbose=verbose and i == 1)
        if ok and not (force and i == 1):
            if verbose and i > 1:
                print(f"    ✅ USB 리셋 {i - 1}회로 복구 — " +
                      ", ".join(f"{c.name}:{','.join(c.videos)}" for c in cams))
            return True
        # 고장난 것만 리셋한다 — 멀쩡한 걸 건드리면 devnum 만 흔들린다.
        # 다만 고장난 게 없는데 대수가 모자라면(=열거 자체가 덜 됐다) 전부 흔들어 본다.
        targets = bad if bad else cams
        if not targets:
            if verbose:
                print("    ✗ USB 에 카메라가 하나도 안 보입니다 — 케이블 확인")
            return False
        if not bad and not force and i > 1:
            # 보이는 건 전부 멀쩡한데 대수가 모자란다 = 그 장치는 USB 열거조차 안 됐다.
            # 멀쩡한 옆 카메라를 몇 번을 리셋해도 없는 장치가 생기지는 않는다. 여기서 멈춘다.
            if verbose:
                print(f"    ✗ 보이는 {len(cams)}대는 전부 정상인데 {expect}대가 아닙니다 — "
                      "안 보이는 카메라는 소프트웨어로 못 살립니다. 케이블/허브를 확인하세요")
            return False
        if verbose:
            print(f"    USB 리셋 {i}/{attempts}: " + ", ".join(t.name for t in targets))
        did = False
        for t in targets:
            what = t.reset(verbose=verbose)
            if what:
                did = True
                if verbose:
                    print(f"      {t.name} {what}")
        if not did:
            if verbose:
                print("    ✗ 리셋 자체가 안 됩니다 — 권한 확인 "
                      "(카메라의 /sys/.../authorized 가 0666 여야 합니다)")
            return False
        wait_ready(targets, timeout=8.0)
        settle()

    ok, _, _ = status(expect, verbose=verbose)
    if verbose and ok:
        print(f"    ✅ {attempts}회 안에 복구")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recover", action="store_true", help="문제가 있으면 리셋해서 되살린다")
    ap.add_argument("--force", action="store_true", help="멀쩡해도 전부 리셋")
    ap.add_argument("--expect", type=int, default=EXPECT_DEFAULT, help="있어야 하는 카메라 수")
    ap.add_argument("--attempts", type=int, default=RESETS_DEFAULT, help="최대 리셋 횟수")
    a = ap.parse_args()

    if a.recover or a.force:
        ok = recover(a.expect, a.attempts, verbose=True, force=a.force)
    else:
        ok, _, _ = status(a.expect, verbose=True)
        if not ok:
            print("  → 되살리려면: python usb_reset.py --recover")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
