# !/usr/bin/env python
"""LeKiwi front/wrist 원본 영상을 그대로 띄운다 — 로봇 위치 잡기·approach 재보정용.

화면에 나오는 것은 `obs["front"]` 원본 프레임 그대로다(리사이즈·회전·보정 없음).
표시만 겹쳐 그리고, 저장되는 이미지는 항상 원본이다.

키:
  o     좌표/면적 오버레이 켜고 끄기 (기본 켬)
  m     빨강 마스크 보기 (red_approach.json 의 HSV 로 검출한 것)
  s     현재 원본 프레임 저장 (/tmp/front_<시각>.jpg)
  c     지금 보이는 손잡이 좌표를 red_approach.json 목표값으로 저장할 값으로 출력
  q/ESC 종료

주의: 이 스크립트가 연결을 끊으면 Jetson lekiwi_host 도 함께 종료된다. 종료 후 태스크를
      다시 돌리려면 host 를 다시 띄워야 한다(laundry_task3 의 ensure_host 가 자동으로 한다).

  python tools/stream_front.py                 # 유선 192.168.55.1
  python tools/stream_front.py --wireless      # 무선 192.168.0.19
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.pop("SESSION_MANAGER", None)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "examples" / "lekiwi"))

import cv2
import numpy as np

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_red():
    cal = {"target_y": 20, "center_x": None, "dead_x": 50, "min_area": 300,
           "ranges": [((0, 120, 80), (10, 255, 255)), ((170, 120, 80), (179, 255, 255))]}
    p = REPO / "red_approach.json"
    if p.exists():
        c = json.load(open(p))
        hsv = c.get("HSV", {})
        if hsv:
            cal["ranges"] = [(tuple(hsv["lower_red1"]), tuple(hsv["upper_red1"])),
                             (tuple(hsv["lower_red2"]), tuple(hsv["upper_red2"]))]
        cal.update(target_y=c.get("TARGET_Y", 20), center_x=c.get("CENTER_X"),
                   dead_x=c.get("DEAD_ZONE_X", 50), min_area=c.get("DETECT_AREA_MIN", 300))
    return cal


def detect(bgr, cal):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in cal["ranges"]:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, 0, mask
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < cal["min_area"]:
        return None, None, area, mask
    M = cv2.moments(c)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), area, mask


def main():
    ap = argparse.ArgumentParser(description="LeKiwi front/wrist 원본 스트리밍")
    ap.add_argument("--wireless", action="store_true")
    ap.add_argument("--ip", default=None)
    args = ap.parse_args()
    ip = args.ip or os.environ.get("REMOTE_IP") or ("192.168.0.19" if args.wireless else "192.168.55.1")

    cal = load_red()
    print(f"[stream] 연결 {ip} ...")
    r = LeKiwiClient(LeKiwiClientConfig(remote_ip=ip, id="lekiwi", connect_timeout_s=15))
    r.connect()
    print("[stream] 연결됨. 키: o=오버레이 m=마스크 s=저장 c=보정값 q=종료")
    print(f"[stream] 현재 red_approach.json 목표: CENTER_X={cal['center_x']} TARGET_Y={cal['target_y']}")

    win = "LeKiwi front (raw)  [o]verlay [m]ask [s]ave [c]alib [q]uit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    overlay, show_mask = True, False
    try:
        while True:
            obs = r.get_observation()
            fr = obs.get("front")
            if not isinstance(fr, np.ndarray):
                time.sleep(0.05)
                continue
            raw = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)     # 원본 그대로 (RGB→BGR 표시 변환만)
            cx, cy, area, mask = detect(raw, cal)
            disp = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else raw.copy()

            if overlay:
                h, w = raw.shape[:2]
                cx0 = cal["center_x"] if cal["center_x"] is not None else w // 2
                cv2.line(disp, (cx0, 0), (cx0, h), (255, 200, 0), 1)          # 목표 x
                cv2.line(disp, (0, cal["target_y"]), (w, cal["target_y"]), (0, 200, 255), 1)  # 목표 y
                if cx is not None:
                    cv2.drawMarker(disp, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
                    txt = f"cx={cx} cy={cy} area={int(area)}  (목표 {cx0},{cal['target_y']})"
                else:
                    txt = f"미검출 (최대면적 {int(area)} < min {cal['min_area']})"
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 26), (0, 0, 0), -1)
                cv2.putText(disp, txt, (8, 18), _FONT, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(disp, f"{raw.shape[1]}x{raw.shape[0]}", (disp.shape[1] - 70, 18),
                            _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow(win, disp)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            elif k == ord("o"):
                overlay = not overlay
            elif k == ord("m"):
                show_mask = not show_mask
            elif k == ord("s"):
                fn = f"/tmp/front_{time.strftime('%H%M%S')}.jpg"
                cv2.imwrite(fn, raw)                       # 저장은 항상 원본
                print(f"[stream] 저장: {fn}")
            elif k == ord("c") and cx is not None:
                print("\n" + "=" * 56)
                print("  지금 이 위치를 approach 목표로 삼으려면 red_approach.json 에:")
                print(f'    "CENTER_X": {cx},')
                print(f'    "TARGET_Y": {cy},')
                print(f'    "AREA_THRESHOLD": {int(area * 0.9)},')
                print("=" * 56 + "\n")
    except KeyboardInterrupt:
        pass
    finally:
        r.disconnect()
        cv2.destroyAllWindows()
        print("[stream] 종료. ⚠️ lekiwi_host 도 함께 내려갑니다 — 다음 실행 시 자동 재기동됩니다.")


if __name__ == "__main__":
    main()
