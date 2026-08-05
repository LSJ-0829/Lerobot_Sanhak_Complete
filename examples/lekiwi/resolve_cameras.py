"""USB 포트가 바뀌어도 카메라 4대를 특정한다.

왜 udev 만으로는 안 되나
  4대가 '완전히 같은 정체성'을 가진 2쌍이다(2026-08-05 실측):
      Sonix   USB_2.0_Camera            SN0001  ← top, overhead
      Innomaker U20CAM-1080p-S1         SN0001  ← left_cam, right_cam
  벤더·모델·시리얼·리비전이 쌍 안에서 전부 동일해서, udev 가 구분할 수 있는 건
  ID_PATH(물리 포트)뿐이다. 그래서 자리를 옮기면 이름이 뒤바뀐다
  (실제로 폴딩 top 이 세탁기 상공캠을 가리키는 사고가 났다).

그래서 두 단계로 특정한다
  1) 모델 문자열로 쌍을 가른다 — 이건 포트와 무관하다.
  2) 쌍 안은 '화면 내용'으로 가른다. 실측 분리도:
       top vs overhead : 어두운 픽셀 비율 0.172 vs 0.000
         (top 은 접는 판의 검은 프레임과 양팔이 크게 잡힌다.
          overhead 는 흰 세탁기를 보고 있어 거의 전부 밝다)
       left vs right   : 검은 패널의 x 중심 0.729 vs 0.360
         (두 팔캠이 접는 판을 서로 반대쪽에서 본다)

결과는 ~/.lerobot/cams/ 아래 심볼릭 링크로 만든다 — **root 권한이 필요 없다**.
    ~/.lerobot/cams/{top,left_cam,right_cam,overhead} → /dev/videoN

좌/우 팔캠의 정답 방향은 기계적으로 알 수 없어 설정 파일에 저장한다
(`--set-left-x-side left|right`). 기본값은 2026-08-05 배치 기준이다.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

CONF = Path.home() / ".lerobot" / "cams.json"
LINKS = Path.home() / ".lerobot" / "cams"

# 쌍을 가르는 모델 문자열(포트 무관)
SONIX = "USB_2.0_Camera"
INNOMAKER = "Innomaker-U20CAM-1080p-S1"

# 실측 기반 임계값. 두 값 사이가 넓어 여유가 크다.
DARK_LEVEL = 90          # 이 밝기 미만을 '어둡다'로 본다
DARK_FRAC_SPLIT = 0.05   # top(0.172) 과 overhead(0.000) 사이
DEFAULT_LEFT_X_SIDE = "right"   # 왼팔 캠은 검은 패널이 화면 '오른쪽'에 잡힌다(2026-08-05)


def udev(dev: str, key: str) -> str:
    try:
        out = subprocess.run(["udevadm", "info", "-q", "property", "-n", dev],
                             capture_output=True, text=True, timeout=5).stdout
        for line in out.splitlines():
            if line.startswith(key + "="):
                return line.split("=", 1)[1]
    except Exception:
        pass
    return ""


def capture(dev: str, warmup: int = 10, tries: int = 40, delay: float = 0.08):
    """프레임 한 장. 이 카메라들은 꽂은 직후/재열거 직후 몇 프레임을 버려야 나온다.

    ⚠️ 재시도 사이에 반드시 쉬어야 한다. sleep 없이 read() 만 반복하면 실패가
    수십 번 순식간에 지나가 버려 '프레임 없음'으로 오판한다(실제로 팔캠 한 대를
    놓쳐 좌/우 배정이 실패했다). 실패 프레임은 거의 즉시 돌아온다.
    """
    import time as _t
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    got = None
    for i in range(tries):
        ok, f = cap.read()
        if ok:
            got = f
            if i >= warmup:
                break
        _t.sleep(delay)
    cap.release()
    return got


def metrics(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dark = g < DARK_LEVEL
    xs = np.nonzero(dark)[1]
    return {
        "mean": float(g.mean()),
        "dark_frac": float(dark.mean()),
        "dark_cx": float(xs.mean() / g.shape[1]) if len(xs) else 0.5,
    }


def enumerate_cams():
    """index==0 인 캡처 노드만 모은다(카메라마다 노드가 2개씩 생긴다)."""
    cams = []
    for dev in sorted(Path("/dev").glob("video*"), key=lambda p: int(p.name[5:])):
        idx = Path(f"/sys/class/video4linux/{dev.name}/index")
        if not idx.exists() or idx.read_text().strip() != "0":
            continue
        model = udev(str(dev), "ID_MODEL")
        if not model:
            continue
        cams.append({"dev": str(dev), "model": model,
                     "path": udev(str(dev), "ID_PATH")})
    return cams


def resolve(verbose=True):
    cfg = json.loads(CONF.read_text()) if CONF.exists() else {}
    left_side = cfg.get("left_x_side", DEFAULT_LEFT_X_SIDE)

    cams = enumerate_cams()
    for c in cams:
        img = capture(c["dev"])
        c["ok"] = img is not None
        if img is not None:
            c.update(metrics(img))

    usable = [c for c in cams if c["ok"]]
    if verbose:
        print(f"{'장치':<14} {'모델':<28} {'포트':<34} {'어두운비율':>9} {'x중심':>7}")
        print("─" * 100)
        for c in cams:
            if c["ok"]:
                print(f"{c['dev']:<14} {c['model']:<28} {c['path']:<34} "
                      f"{c['dark_frac']:9.3f} {c['dark_cx']:7.3f}")
            else:
                print(f"{c['dev']:<14} {c['model']:<28} {c['path']:<34}  프레임 실패")

    result, why = {}, []

    sonix = [c for c in usable if c["model"] == SONIX]
    if len(sonix) == 2:
        # 어두운 비율이 큰 쪽이 접는 판(검은 프레임 + 양팔), 작은 쪽이 흰 세탁기.
        s = sorted(sonix, key=lambda c: c["dark_frac"])
        result["overhead"], result["top"] = s[0]["dev"], s[1]["dev"]
        why.append(f"Sonix 2대: 어두운비율 {s[0]['dark_frac']:.3f}→overhead, "
                   f"{s[1]['dark_frac']:.3f}→top")
        if s[1]["dark_frac"] - s[0]["dark_frac"] < DARK_FRAC_SPLIT:
            why.append("  ⚠️ 두 값이 가깝다 — 조명/장면이 평소와 다를 수 있으니 확인할 것")
    elif len(sonix) == 1:
        result["top"] = sonix[0]["dev"]
        why.append("Sonix 1대뿐 — top 으로 배정(overhead 미연결)")
    else:
        why.append(f"⚠️ Sonix 카메라가 {len(sonix)}대 — top/overhead 배정 불가")

    inno = [c for c in usable if c["model"] == INNOMAKER]
    if len(inno) == 2:
        # 검은 패널이 화면 오른쪽에 잡히는 쪽 / 왼쪽에 잡히는 쪽으로 갈린다.
        a, b = sorted(inno, key=lambda c: c["dark_cx"])   # a=왼쪽, b=오른쪽
        if left_side == "right":
            result["left_cam"], result["right_cam"] = b["dev"], a["dev"]
        else:
            result["left_cam"], result["right_cam"] = a["dev"], b["dev"]
        why.append(f"Innomaker 2대: x중심 {a['dark_cx']:.3f} / {b['dark_cx']:.3f}"
                   f" (설정: 왼팔캠은 패널이 {left_side} 쪽)")
        if b["dark_cx"] - a["dark_cx"] < 0.15:
            why.append("  ⚠️ x중심 차이가 작다 — 판이 비어 있으면 구분이 흐려진다")
    else:
        why.append(f"⚠️ Innomaker 카메라가 {len(inno)}대 — 좌/우팔 배정 불가")

    if verbose:
        print()
        for w in why:
            print(f"  {w}")
    return result, cams, why


ROLES = ("top", "left_cam", "right_cam", "overhead")


def write_links(result):
    """배정된 링크를 새로 만들고, 배정 못 한 역할의 '묵은 링크'는 지운다.

    지우는 게 중요하다: /dev/videoN 번호는 재열거마다 바뀌어서, 남겨 두면 사라진 노드를
    가리키는 링크가 된다. 그 상태로 열면 'device busy or missing' 으로 죽는데,
    원인이 busy 인지 missing 인지 알기 어려워 진단이 오래 걸린다(실제로 겪음).
    """
    LINKS.mkdir(parents=True, exist_ok=True)
    for role in ROLES:
        link = LINKS / role
        dev = result.get(role)
        if link.is_symlink() or link.exists():
            link.unlink()
        if dev:
            link.symlink_to(dev)
    return LINKS


def wait_released(devs, timeout=15.0, verbose=True):
    """방금 우리가 열었던 카메라들이 '다시 열리는' 상태가 될 때까지 기다린다.

    이걸 안 하면 배정 직후 롤아웃이 카메라를 열다 실패한다:
        ConnectionError: ... could not be opened (device busy or missing)
    V4L2 장치는 close() 직후 곧바로 재오픈되지 않는 경우가 있다. 측정하느라 우리가
    열었던 것이 원인이므로, 넘겨주기 전에 여기서 확인하고 기다리는 게 맞다.
    """
    import time as _t
    deadline = _t.time() + timeout
    pending = list(devs)
    while pending and _t.time() < deadline:
        still = []
        for d in pending:
            cap = cv2.VideoCapture(d, cv2.CAP_V4L2)
            ok = cap.isOpened()
            cap.release()
            if not ok:
                still.append(d)
        if not still:
            pending = []
            break
        pending = still
        _t.sleep(0.4)
    if verbose and pending:
        print(f"  ⚠️ 아직 열리지 않는 장치: {', '.join(pending)}")
    return not pending


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", action="store_true", help="측정값만 보고 링크는 만들지 않음")
    ap.add_argument("--set-left-x-side", choices=["left", "right"],
                    help="왼팔 캠에서 검은 패널이 어느 쪽에 잡히는지 저장(좌/우 뒤바뀌면 반대로)")
    ap.add_argument("--json", action="store_true", help="결과를 JSON 으로 출력")
    args = ap.parse_args()

    CONF.parent.mkdir(parents=True, exist_ok=True)
    if args.set_left_x_side:
        cfg = json.loads(CONF.read_text()) if CONF.exists() else {}
        cfg["left_x_side"] = args.set_left_x_side
        CONF.write_text(json.dumps(cfg, indent=1))
        print(f"저장됨: 왼팔 캠은 검은 패널이 '{args.set_left_x_side}' 쪽 → {CONF}")

    result, cams, why = resolve(verbose=not args.json)

    if args.json:
        print(json.dumps({"mapping": result, "cameras": cams, "why": why},
                         ensure_ascii=False, indent=1))
        return 0

    print("\n=== 배정 결과 ===")
    for role in ("top", "left_cam", "right_cam", "overhead"):
        dev = result.get(role)
        port = next((c["path"] for c in cams if c["dev"] == dev), "")
        print(f"  {role:<10} → {dev or '✗ 없음':<14} {port}")

    if args.scan:
        print("\n(--scan 이라 링크는 만들지 않았다)")
        return 0

    d = write_links(result)
    print(f"\n심볼릭 링크 생성: {d}/  (root 권한 불필요)")
    for role in sorted(result):
        print(f"  {d}/{role} → {os.readlink(d / role)}")
    print("\n좌/우가 반대라면:  python resolve_cameras.py --set-left-x-side left")
    return 0


if __name__ == "__main__":
    sys.exit(main())
