# !/usr/bin/env python
"""상공 USB 카메라를 번호 대신 '실제로 프레임이 나오는 USB 캡처 노드'로 찾아준다.

왜 필요한가: `/dev/videoN` 번호는 본체마다 다르고, 같은 본체에서도 USB 를 다시 꽂거나
부팅 순서가 바뀌면 달라진다. 이 랩탑은 내장 카메라(Intel IPU6)가 `/dev/video0~31` 을
전부 차지하고 있어서, USB 캠은 매번 남는 번호(32, 33, ...)에 붙는다.

탐색 규칙:
  1) 명시된 경로가 열리고 프레임이 나오면 그대로 쓴다.
  2) 아니면 sysfs 로 'USB 에 물린' video 노드만 추린다(내장 PCI 카메라는 자동 제외).
  3) 그중 실제로 프레임이 읽히는 첫 노드를 고른다(메타데이터 전용 노드는 여기서 걸러진다).
  4) 고른 노드의 안정적인 이름(/dev/v4l/by-id/...)도 같이 돌려준다 — 이걸 고정해두면
     번호가 바뀌어도 그대로 쓸 수 있다.

단독 실행:
  python examples/lekiwi/find_overhead_cam.py            # 자동 탐색 결과 출력
  python examples/lekiwi/find_overhead_cam.py --list     # 후보 전부 나열
  python examples/lekiwi/find_overhead_cam.py --match C920  # 이름/by-id 에 문자열이 든 것만
"""

import argparse
import os
from pathlib import Path

SYS_V4L = Path("/sys/class/video4linux")
STABLE_DIRS = ("/dev/v4l/by-id", "/dev/v4l/by-path")


def _node_index(p: Path) -> int:
    try:
        return int(p.name.replace("video", ""))
    except ValueError:
        return 1 << 30


def list_video_nodes():
    """/dev/video* 를 sysfs 정보와 함께 나열. usb=True 면 USB 에 물린 장치."""
    nodes = []
    if not SYS_V4L.is_dir():
        return nodes
    for d in sorted(SYS_V4L.glob("video*"), key=_node_index):
        dev = f"/dev/{d.name}"
        if not os.path.exists(dev):
            continue
        name = ""
        if (d / "name").exists():
            name = (d / "name").read_text(errors="replace").strip()
        real = os.path.realpath(d / "device") if (d / "device").exists() else ""
        nodes.append({"dev": dev, "name": name, "usb": "/usb" in real, "sysdev": real})
    return nodes


def stable_names(dev):
    """dev 를 가리키는 /dev/v4l/by-id, by-path 링크들(번호가 바뀌어도 유지되는 이름)."""
    target = os.path.realpath(dev)
    out = []
    for base in STABLE_DIRS:
        b = Path(base)
        if not b.is_dir():
            continue
        for link in sorted(b.iterdir()):
            if os.path.realpath(link) == target:
                out.append(str(link))
    return out


def probe(dev, warmup=3):
    """실제로 열어서 프레임이 나오는지 확인. (h, w) 를 돌려주고, 실패하면 None."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return None
        frame = None
        for _ in range(warmup):  # UVC 는 첫 프레임이 비는 경우가 있어 몇 장 버린다
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        return None if frame is None else (frame.shape[0], frame.shape[1])
    finally:
        cap.release()


def candidates(match=""):
    """USB 캡처 후보 목록. match 가 있으면 이름/by-id 에 그 문자열이 든 것만."""
    out = []
    for n in list_video_nodes():
        if not n["usb"]:
            continue  # 내장 PCI 카메라(Intel IPU6 등) 제외
        n["stable"] = stable_names(n["dev"])
        if match:
            hay = (n["name"] + " " + " ".join(n["stable"])).lower()
            if match.lower() not in hay:
                continue
        out.append(n)
    return out


# udev 로 고정해 둔 이름들. 리그에 카메라가 여러 대면 자동 탐색이 엉뚱한 걸 고를 수 있으므로
# 이게 있으면 항상 먼저 쓴다(tools/99-lekiwi-overhead.rules 가 만든다).
WELL_KNOWN = ("/dev/lerobot/overhead", "/dev/lekiwi/overhead")


def resolve_overhead_cam(spec="auto", match="", verbose=True):
    """상공 카메라 경로를 확정한다.

    우선순위: 명시 경로 → udev 고정 이름(/dev/lerobot/overhead) → USB 자동 탐색.
    반환: (경로 or None, 안내 메시지 리스트)
    """
    log = []
    if spec and spec != "auto":
        if os.path.exists(spec) and probe(spec):
            log.append(f"지정 경로 사용: {spec}")
            return spec, log
        log.append(f"지정 경로 {spec} 를 못 씀(없거나 프레임 안 나옴) → 자동 탐색으로 전환")

    for wk in WELL_KNOWN:
        if os.path.exists(wk) and probe(wk):
            log.append(f"udev 고정 이름 사용: {wk} → {os.path.realpath(wk)}")
            return wk, log

    cands = candidates(match)
    if len(cands) > 2 and not match:
        # 리그에 카메라가 여러 대다(예: SO101 top/left/right + 상공). 자동 탐색은 '프레임이
        # 나오는 첫 번째'를 고르므로 엉뚱한 걸 잡을 수 있다.
        log.append(f"⚠️ USB 카메라가 여러 대({len(cands)//2 or len(cands)}대로 추정) 보인다. "
                   "udev 로 /dev/lerobot/overhead 를 만들어 두거나 --overhead-match 로 좁히는 걸 권함")
    if not cands:
        log.append("USB 카메라 후보 없음 — 케이블/허브 연결 확인 필요"
                   + (f" (match='{match}' 조건에 걸렸을 수도 있음)" if match else ""))
        return None, log

    for n in cands:
        shape = probe(n["dev"])
        if shape is None:
            if verbose:
                log.append(f"  {n['dev']:<14} {n['name'][:40]:<40} → 프레임 없음(메타데이터 노드로 추정)")
            continue
        log.append(f"  {n['dev']:<14} {n['name'][:40]:<40} → {shape[1]}x{shape[0]} ✅")
        if n["stable"]:
            log.append(f"  번호가 바뀌어도 되는 고정 경로: {n['stable'][0]}")
        return n["dev"], log
    log.append("USB 노드는 보이는데 프레임이 나오는 게 없음 — 다른 프로세스가 쓰고 있는지 확인")
    return None, log


def main():
    ap = argparse.ArgumentParser(description="상공 USB 카메라 자동 탐색(번호 대신).")
    ap.add_argument("--match", default="", help="이름/by-id 에 포함될 문자열로 후보 좁히기")
    ap.add_argument("--spec", default="auto", help="먼저 시도할 경로(기본 auto)")
    ap.add_argument("--list", action="store_true", help="후보만 나열하고 종료")
    args = ap.parse_args()

    if args.list:
        nodes = list_video_nodes()
        print(f"video 노드 {len(nodes)}개 (USB 만 상공캠 후보):")
        for n in nodes:
            tag = "USB" if n["usb"] else "내장"
            st = stable_names(n["dev"])
            print(f"  [{tag}] {n['dev']:<14} {n['name'][:45]:<45} {st[0] if st else ''}")
        return 0

    dev, log = resolve_overhead_cam(args.spec, args.match)
    for line in log:
        print(line)
    if dev is None:
        print("✗ 상공 카메라를 못 찾았습니다.")
        return 1
    print(f"\nOVERHEAD_CAM={dev}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
