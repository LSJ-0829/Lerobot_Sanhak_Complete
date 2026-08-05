"""빨간 손잡이의 cx/cy/area 를 실시간으로 찍는다. 로봇을 손으로 밀며 거리별 값을 본다.

왜 필요한가: approach 는 cy 로 거리를 재는데, cy 가 거리에 따라 충분히 변하지 않으면
(=민감도가 낮으면) 어떤 TARGET_Y 를 넣어도 엉뚱한 곳에서 멈춘다. area 는 거리의 제곱에
반비례해 변하므로 대개 훨씬 잘 구분된다. 둘 중 무엇이 실제로 거리를 구분하는지 본다.

사용: python measure_approach.py [--ip 192.168.55.1]
  1) 로봇을 '문 열기에 딱 맞는 위치'에 두고 s 를 눌러 기록
  2) 20~30cm 뒤로 물린 뒤 s 를 다시 눌러 기록
  3) q 로 종료 — 두 지점의 cy/area 차이를 비교해 준다
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "skills"))

from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ip", default="192.168.55.1")
ap.add_argument("--cal", default=str(REPO / "red_approach.json"))
args = ap.parse_args()

cal = json.loads(Path(args.cal).read_text())
hsv = cal["HSV"]
ranges = [(np.array(hsv["lower_red1"]), np.array(hsv["upper_red1"])),
          (np.array(hsv["lower_red2"]), np.array(hsv["upper_red2"]))]
min_area = cal["DETECT_AREA_MIN"]


def detect(bgr):
    h = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(h, lo, hi)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, 0
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    if a < min_area:
        return None, None, a
    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, None, a
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), a


robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=args.ip, id="lekiwi", connect_timeout_s=15))
robot.connect()
print(f"현재 설정값: CENTER_X={cal['CENTER_X']} TARGET_Y={cal['TARGET_Y']} "
      f"AREA_THRESHOLD={cal['AREA_THRESHOLD']}")
print("s = 현재 위치 기록,  q = 종료   (창을 클릭해 포커스를 준 뒤 키를 누르세요)\n")

marks = []
try:
    while True:
        obs = robot.get_observation()
        fr = obs.get("front")
        if not isinstance(fr, np.ndarray):
            time.sleep(0.05)
            continue
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        cx, cy, area = detect(bgr)
        txt = f"cx={cx} cy={cy} area={int(area)}" if cx is not None else f"미검출 (area={int(area)})"
        print(f"\r  {txt}          ", end="", flush=True)

        vis = bgr.copy()
        if cx is not None:
            cv2.circle(vis, (cx, cy), 6, (0, 255, 0), -1)
            cv2.line(vis, (cal["CENTER_X"], 0), (cal["CENTER_X"], vis.shape[0]), (255, 0, 0), 1)
            cv2.line(vis, (0, cal["TARGET_Y"]), (vis.shape[1], cal["TARGET_Y"]), (0, 0, 255), 1)
        cv2.putText(vis, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("approach measure (s=기록 q=종료)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and cx is not None:
            marks.append((cx, cy, int(area)))
            print(f"\n  [{len(marks)}] 기록: cx={cx} cy={cy} area={int(area)}")
finally:
    cv2.destroyAllWindows()
    robot.disconnect()

print("\n\n=== 기록 ===")
for i, (cx, cy, a) in enumerate(marks, 1):
    print(f"  [{i}] cx={cx:4d}  cy={cy:4d}  area={a:6d}")
if len(marks) >= 2:
    (x1, y1, a1), (x2, y2, a2) = marks[0], marks[-1]
    print(f"\n  cy 변화   : {y1} → {y2}  (차이 {y2 - y1:+d}px)")
    print(f"  area 변화 : {a1} → {a2}  (배율 {a2 / max(a1, 1):.2f}x)")
    if abs(y2 - y1) < 15:
        print("\n  ⚠️ cy 가 거리에 거의 안 변한다 → cy 로는 거리를 못 잰다.")
        print("     area 기반 정지로 바꾸는 게 맞다.")
    else:
        print(f"\n  cy 가 거리를 구분한다 → TARGET_Y 를 정답 위치 값({y1})으로 두면 된다.")
