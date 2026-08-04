# !/usr/bin/env python
"""세탁물 회수 태스크 v3 (랩탑 ZMQ 단일 프로세스) — 전체 파이프라인.

approach → 문열기 → grab(VLA+2단계 게이트+압착) → carry → 복귀주행 → throw 를
전부 노트북에서 LeKiwiClient(ZMQ)로 실행한다(호스트는 계속 UP). 기존 Jetson 직접버스
구현(move_to_destination / open_door2 / return_home / grab_place)을 ZMQ base 속도로 포팅했다.

■ 왜 랩탑 단일 프로세스인가:
  VLA(SmolVLA)는 랩탑 GPU 필수 → host 가 계속 떠 있어야 함(Jetson 직접버스 못 씀).
  그래서 approach/문열기/주행/throw 도 ZMQ base 속도(x=m/s 전후, y=m/s 좌우, theta=rad/s)로
  옮겼다. Jetson 은 lekiwi_host(모터·카메라 서빙)만 담당.

■ 모든 무거운 로딩은 시작 Enter '전'에 끝낸다(요청):
  SmolVLA+프로세서, 상공/손목 probe, 상공 카메라, 포즈/모션 변환, red_approach 파라미터.

■ 플래그는 둘뿐:  --wireless (없으면 유선),  --record (front/wrist/상공 녹화).
  나머지(속도·시간·부호)는 전부 env — 실기에서 재보정한다(⚠️ CALIBRATION 섹션).

■ Jetson 에서 트리거:  scripts/skills/run_task3_from_jetson.sh 를 Jetson 에서 실행하면
  host 를 띄우고 랩탑으로 SSH 해 이 스크립트를 실행한다.

전제: 상공 USB 카메라(기본 /dev/video32) 랩탑 연결. red_approach.json / poses / motions 존재.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.pop("SESSION_MANAGER", None)  # Qt(ICE) 자살 방지

REPO = Path(__file__).resolve().parents[2]           # ~/lerobot
sys.path.insert(0, str(REPO / "examples" / "lekiwi"))  # grasp_clip / lekiwi_pose

import cv2
import numpy as np
import torch

from grasp_clip import GraspGate, open_v4l2_camera
from lekiwi_pose import ARM_JOINTS, load_pose, move_to_pose, pose_to_action, raw_to_norm
from lerobot.common.control_utils import predict_action
from lerobot.policies import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.robots.lekiwi import LeKiwiClient, LeKiwiClientConfig
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

# ─────────────────────── 기본 설정 ───────────────────────
OVERHEAD_CAM = os.environ.get("OVERHEAD_CAM", "auto")  # 'auto'=번호 대신 USB 노드 자동 탐색
TASK_DESCRIPTION = "grabbing clothes from washer"
FPS = int(os.environ.get("FPS", "30"))
RENAME = {"front": "camera1", "wrist": "camera2"}
BASE_KEYS = ("x.vel", "y.vel", "theta.vel")
_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_CKPT = str(REPO / "models" / "smolvla_lekiwi_spin_cycle")

# 포즈 이름
DRIVE_POSE = os.environ.get("DRIVE_POSE", "laundry_default")     # 접근 주행 자세
OPENREADY = os.environ.get("OPENREADY_POSE", "laundry_openready")
OPENCATCH = os.environ.get("OPENCATCH_POSE", "laundry_opencatch")
NOTHING_POSE = os.environ.get("NOTHING_POSE", "nothing")
BACK_POSE = os.environ.get("BACK_POSE", "laundry_back")
READY_POSE = os.environ.get("READY_POSE", "laundry_grabready")   # VLA 집기 자세
CARRY_POSE = os.environ.get("CARRY_POSE", "laundry_default")     # 집은 뒤 주행 자세
HOME_DURATION = float(os.environ.get("HOME_DURATION", "3.0"))
# 마지막 후퇴 뒤에 재생할 '전달 모션'(SO101 인계용). motions/<이름>.json. 비우면 생략.
HANDOFF_MOTION = os.environ.get("HANDOFF_MOTION", "")

# grasp 게이트 / 손목 확정 / 압착
GRASP_THRESHOLD = float(os.environ.get("GRASP_THRESHOLD", "0.55"))
GRASP_HOLD = int(os.environ.get("GRASP_HOLD", "3"))
GRASP_CHECK_EVERY = int(os.environ.get("GRASP_CHECK_EVERY", "3"))
WRIST_THRESHOLD = float(os.environ.get("WRIST_THRESHOLD", "0.5"))
GRIP_SQUEEZE = float(os.environ.get("GRIP_SQUEEZE", "0"))
GRIP_SQUEEZE_SEC = float(os.environ.get("GRIP_SQUEEZE_SEC", "1.2"))

# ══════════════════ ⚠️ CALIBRATION (실기 재보정) ══════════════════
# ZMQ base 속도는 m/s(x,y)·rad/s(theta) 물리단위라 기존 직접버스 wheel-speed/시간과 다르다.
# 부호(Y_RIGHT/THETA_CW)는 실기에서 방향 확인 후 뒤집는다.
# 아래 세 값은 검증된 직접버스 구현(~/Lerobot_Sanhak, laundry/open_door2.py, grab_place.py)의
# wheel-speed(drive 500 / strafe 400 / rotate 300)에서 역산했다. 시간 기반 이동은 그 속도 기준으로
# 보정돼 있어서, 속도가 다르면 거리가 통째로 어긋난다.
#   raw = wheel_linear(m/s) × 13038
#   전진  raw500 → 500/(0.866×13038) = 0.044 m/s   (open_door2/grab_place drive_speed=500)
#   strafe raw400 → 400/(1.000×13038) = 0.031 m/s   (open_door2 strafe_speed=400)
#   회전  raw300 → 300/(0.125×13038) = 0.184 rad/s
# 검산: 0.184 rad/s × 16.9s = 178도 — Lerobot_Sanhak/laundry_task3 의 "16.9초(speed 300 실측)"와 일치.
V_FWD = float(os.environ.get("V_FWD", "0.044"))       # 전후진 속도 m/s
V_STRAFE = float(os.environ.get("V_STRAFE", "0.031"))  # 좌우 strafe 속도 m/s
V_ROT = float(os.environ.get("V_ROT", "0.184"))        # 회전 속도 rad/s
# 실기 확인(2026-08-04): 좌우가 반대로 나가서 뒤집었다.
Y_RIGHT = float(os.environ.get("Y_RIGHT", "-1"))      # y.vel 부호: +값이 '오른쪽'이면 1
THETA_CW = float(os.environ.get("THETA_CW", "-1"))    # theta.vel 부호: 시계(우)회전이 음수면 -1
STEP = float(os.environ.get("STEP", "0.15"))          # approach 펄스 최대 길이(초)
# approach
APPROACH_TIMEOUT = float(os.environ.get("APPROACH_TIMEOUT", "40"))
# approach 전용 저속. V_FWD/V_STRAFE 를 낮추면 [7] 복귀 주행처럼 '시간 기반' 이동거리가 같이
# 줄어들기 때문에 여기서만 따로 쓴다(V_FWD=0.15 로 한 펄스가 약 2cm 라 목표 근처에서 튄다).
V_APPROACH = float(os.environ.get("V_APPROACH", "0.05"))     # strafe 최대
V_APPROACH_Y = float(os.environ.get("V_APPROACH_Y", "0.05"))  # 전후진 최대
V_APPROACH_MIN = float(os.environ.get("V_APPROACH_MIN", "0.03"))  # 이보다 느리면 모터가 안 돈다
# 이 오차(px)에서 최대 속도. 그보다 작으면 비례해서 느려진다.
APPROACH_FULL_PX = float(os.environ.get("APPROACH_FULL_PX", "80"))
# 실측(2026-08-04, 시연 본체): y.vel=+0.08 을 0.5초 주면 화면상 cx 가 +41px 커진다.
# 즉 cx 를 '줄이려면' y.vel 은 음수여야 한다. 배치가 바뀌어 반대가 되면 -1 로 뒤집는다.
Y_VEL_TO_CX = float(os.environ.get("Y_VEL_TO_CX", "-1"))
# 실측: x.vel=+0.10 전진 시 cy 가 -11px 작아진다 → cy 를 줄이려면 x.vel 은 양수.
X_VEL_TO_CY = float(os.environ.get("X_VEL_TO_CY", "-1"))
# 전후진 허용오차(px). 4 는 너무 빡빡해서 한 STEP 이 그보다 크게 움직이면 영원히 진동한다.
APPROACH_Y_TOL = float(os.environ.get("APPROACH_Y_TOL", "8"))
# 문열기 base 이동 시간(초) — 직접버스 기본과 같은 값, 속도가 달라 재보정 필요
OPEN_STRAFE_SEC = float(os.environ.get("OPEN_STRAFE_SEC", "0.5"))    # 오른쪽 정렬
OPEN_FORWARD_SEC = float(os.environ.get("OPEN_FORWARD_SEC", "0.7"))  # 손잡이쪽 전진
BACKDRIVE_SEC = float(os.environ.get("BACKDRIVE_SEC", "2.0"))        # 후진+1번회전
POST_OPEN_STRAFE_SEC = float(os.environ.get("POST_OPEN_STRAFE_SEC", "2.0"))  # 모션후 왼쪽
POST_OPEN_FORWARD_SEC = float(os.environ.get("POST_OPEN_FORWARD_SEC", "1.0"))
# 복귀 주행(집은 뒤): 후진→180도 회전→직진→(throw 후)후진
BACKUP_SEC = float(os.environ.get("BACKUP_SEC", "5.0"))
ROTATE_SEC = float(os.environ.get("ROTATE_SEC", "16.9"))   # 180도 되게 실측
FORWARD_SEC = float(os.environ.get("FORWARD_SEC", "5.0"))
RETREAT_SEC = float(os.environ.get("RETREAT_SEC", "2.0"))
RECORD_STRIDE = int(os.environ.get("RECORD_STRIDE", "3"))
# ═════════════════════════════════════════════════════════════════


# ─────────────────────── base(ZMQ) 이동 헬퍼 ───────────────────────
# ⚠️ 액션에는 팔 자세(.pos)를 '반드시' 같이 실어야 한다.
#    Jetson host 의 send_action 은 이렇게 나뉜다:
#        arm_goal_pos  = {k: v for k, v in action.items() if k.endswith(".pos")}
#        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}
#        ...
#        self.bus.sync_write("Goal_Position", arm_goal_pos_raw)
#    속도만 보내면 arm_goal_pos 가 빈 dict 이 되고, 빈 dict 로 sync_write 하다 예외가 난다.
#    host 는 그 예외를 "Message fetching failed" 로 삼키고 **액션 전체를 버린다** → 바퀴도 안 돈다.
#    2026-08-04 실측: 속도만 25회 전송 → host 예외 25건, 로봇 미동작.
#                     팔 자세를 같이 보내니 예외 0건, 정상 주행.
def _arm_hold(robot):
    """현재 팔 자세(.pos)를 읽어 둔다 — 주행 중 이 자세를 유지시키는 용도."""
    try:
        obs = robot.get_observation()
        return {k: float(v) for k, v in obs.items()
                if isinstance(v, (int, float)) and k.startswith("arm_") and k.endswith(".pos")}
    except Exception:
        return {}


def _send_base(robot, x=0.0, y=0.0, theta=0.0, arm=None):
    act = dict(arm) if arm else {}
    act.update({"x.vel": float(x), "y.vel": float(y), "theta.vel": float(theta)})
    robot.send_action(act)


def _stop(robot, arm=None):
    if arm is None:
        arm = _arm_hold(robot)
    for _ in range(3):
        _send_base(robot, 0, 0, 0, arm)
        time.sleep(1.0 / FPS)


def move_base(robot, x=0.0, y=0.0, theta=0.0, seconds=0.0, rec=None, ohcap=None, arm=None):
    """body 속도로 seconds 초 이동 후 정지. arm 을 주면 그 자세를 유지한 채 주행한다."""
    if arm is None:
        arm = _arm_hold(robot)      # 매 펄스마다 읽으면 느리므로 시작 때 한 번만
    for i in range(max(1, int(seconds * FPS))):
        _send_base(robot, x, y, theta, arm)
        if rec is not None and i % RECORD_STRIDE == 0:
            oh = None
            if ohcap is not None:
                ok, f = ohcap.read(); oh = f if ok else None
            rec.add(robot.get_observation(), oh)
        time.sleep(1.0 / FPS)
    _stop(robot, arm)


def fwd(robot, direction, seconds, **kw):   # +1 전진 / -1 후진
    move_base(robot, x=direction * V_FWD, seconds=seconds, **kw)


def strafe(robot, direction, seconds, **kw):  # +1 오른쪽 / -1 왼쪽
    move_base(robot, y=direction * Y_RIGHT * V_STRAFE, seconds=seconds, **kw)


def rotate(robot, clockwise, seconds, **kw):  # True=시계(우)
    w = (THETA_CW if clockwise else -THETA_CW) * V_ROT
    move_base(robot, theta=w, seconds=seconds, **kw)


def _p_vel(err_px, v_max, axis_sign):
    """오차(px)에 비례한 속도(m/s). 펄스+정지를 반복하지 않으므로 움직임이 부드럽다.

    axis_sign: 그 축의 '오차를 줄이는' 부호(실측으로 확정. Y_VEL_TO_CX / X_VEL_TO_CY).
    너무 느리면 모터가 안 도므로 V_APPROACH_MIN 밑으로는 안 내려간다.
    """
    if err_px == 0:
        return 0.0
    frac = min(1.0, abs(err_px) / max(1.0, APPROACH_FULL_PX))
    v = max(V_APPROACH_MIN, v_max * frac)
    return -axis_sign * (1 if err_px > 0 else -1) * v


def replay_motion(robot, name, rec=None, ohcap=None):
    """motions/<name>.json(raw frames) 을 ZMQ 로 재생(정규화 액션 순차 전송)."""
    path = REPO / "motions" / f"{name}.json"
    if not path.exists():
        print(f"  ⚠️ 모션 없음: {path}"); return False
    d = json.load(open(path))
    interval = float(d.get("interval", 0.3))
    for fr in d.get("frames", []):
        act = {f"arm_{j}.pos": raw_to_norm(j, fr[j]) for j in ARM_JOINTS if j in fr}
        act.update({k: 0.0 for k in BASE_KEYS})
        robot.send_action(act)
        if rec is not None:
            oh = None
            if ohcap is not None:
                ok, f = ohcap.read(); oh = f if ok else None
            rec.add(robot.get_observation(), oh)
        time.sleep(interval)
    return True


# ─────────────────────── approach(빨간블롭) ───────────────────────
def load_red_params():
    p = REPO / "red_approach.json"
    cal = {"target_y": 20, "center_x": None, "dead_x": 50, "min_area": 300,
           "ranges": [((0, 120, 80), (10, 255, 255)), ((170, 120, 80), (179, 255, 255))]}
    if p.exists():
        c = json.load(open(p))
        hsv = c.get("HSV", {})
        if hsv:
            cal["ranges"] = [(tuple(hsv["lower_red1"]), tuple(hsv["upper_red1"])),
                             (tuple(hsv["lower_red2"]), tuple(hsv["upper_red2"]))]
        cal["target_y"] = c.get("TARGET_Y", 20)
        cal["center_x"] = c.get("CENTER_X", None)
        cal["dead_x"] = c.get("DEAD_ZONE_X", 50)
        cal["min_area"] = c.get("DETECT_AREA_MIN", 300)
    return cal


def detect_blob(frame_bgr, ranges, min_area):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in ranges:
        m = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None, None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area:
        return None, None, None
    M = cv2.moments(c)
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]), area


def approach(robot, cal, y_tol=None, rec=None, ohcap=None, gui_win=None):
    """front(obs) 빨간블롭을 strafe 로 중심 맞추고 전후진으로 target_y 까지 접근.

    진행 상황을 주기적으로 찍는다. 예전엔 40초간 아무 것도 안 찍고 '타임아웃'만 나와서,
    무엇이 잘못됐는지(못 찾는 건지 / 엉뚱한 방향으로 가는 건지) 알 수가 없었다.
    """
    y_tol = APPROACH_Y_TOL if y_tol is None else y_tol
    cx0_cfg = cal["center_x"]
    print(f"[approach] 빨간 손잡이 접근 — 목표 cx={cx0_cfg} cy={cal['target_y']} "
          f"(허용 x±{cal['dead_x']} y±{y_tol}), 최대 {APPROACH_TIMEOUT}s")
    print(f"[approach] 연속 비례 제어 — 최대 strafe {V_APPROACH} / 전후 {V_APPROACH_Y} m/s, "
          f"최소 {V_APPROACH_MIN} m/s, 오차 {APPROACH_FULL_PX}px 에서 최대속도")
    deadline = time.time() + APPROACH_TIMEOUT
    last_log = 0.0
    first = None       # 처음 관측한 (cx, cy) — 방향이 맞는지 판단용
    last = None
    n_miss = 0
    while time.time() < deadline:
        obs = robot.get_observation()
        fr = obs.get("front")
        if not isinstance(fr, np.ndarray):
            time.sleep(0.05); continue
        bgr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        cx0 = cx0_cfg if cx0_cfg is not None else bgr.shape[1] // 2
        cx, cy, area = detect_blob(bgr, cal["ranges"], cal["min_area"])
        if rec is not None:
            rec.add(obs, (ohcap.read()[1] if ohcap else None))
        if gui_win:
            cv2.imshow(gui_win, cv2.resize(bgr, (480, 360)))
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                print("[approach] abort"); return False
        # 주행 액션에 실어 보낼 현재 팔 자세 — 이미 읽은 obs 에서 뽑으므로 추가 통신이 없다.
        arm = {k: float(v) for k, v in obs.items()
               if isinstance(v, (int, float)) and k.startswith("arm_") and k.endswith(".pos")}

        if cx is None:
            n_miss += 1
            if time.time() - last_log > 1.5:
                print(f"[approach] 손잡이 미검출 ({n_miss}회) — 시야에 없거나 HSV/조명 문제")
                last_log = time.time()
            _stop(robot, arm); time.sleep(0.05); continue

        last = (cx, cy)
        if first is None:
            first = (cx, cy)
        dx, dy = cx - cx0, cy - cal["target_y"]

        # 두 축을 '동시에', '연속으로' 제어한다. 예전처럼 펄스를 주고 멈추기를 반복하면
        # 휙휙 끊겨 보인다. 매 주기 오차에 비례한 속도를 갱신해 보내면 부드럽게 수렴한다.
        aligned_x = abs(dx) <= cal["dead_x"]
        aligned_y = abs(dy) <= y_tol
        if aligned_x and aligned_y:
            _stop(robot, arm)
            print(f"[approach] ✅ 도착 cx={cx}(목표 {cx0}) cy={cy}(목표 {cal['target_y']}) area={int(area)}")
            return True

        vy = 0.0 if aligned_x else _p_vel(dx, V_APPROACH, Y_VEL_TO_CX)
        vx = 0.0 if aligned_y else _p_vel(dy, V_APPROACH_Y, X_VEL_TO_CY)
        _send_base(robot, x=vx, y=vy, arm=arm)

        if time.time() - last_log > 1.0:
            print(f"[approach] cx={cx}{'✓' if aligned_x else f'(차이 {dx:+d})'} "
                  f"cy={cy}{'✓' if aligned_y else f'(차이 {dy:+d})'} area={int(area)} "
                  f"→ x={vx:+.3f} y={vy:+.3f} m/s")
            last_log = time.time()
        time.sleep(1.0 / FPS)

    _stop(robot)
    print(f"[approach] ⚠️ 타임아웃({APPROACH_TIMEOUT}s)")
    if last is None:
        print("[approach]   손잡이를 한 번도 못 찾았다 → 로봇이 세탁기를 안 보고 있거나 HSV/조명 문제.")
        print("[approach]   tools/stream_front.py 로 화면을 확인할 것.")
    else:
        print(f"[approach]   마지막 cx={last[0]}(목표 {cx0_cfg}) cy={last[1]}(목표 {cal['target_y']})")
        if first is not None:
            moved = (abs(first[0] - last[0]), abs(first[1] - last[1]))
            if max(moved) < 5:
                print("[approach]   시작과 끝의 좌표가 거의 같다 → 바퀴가 안 움직이는 것으로 보인다"
                      " (ZMQ base 명령/모터 확인).")
            else:
                near_x = abs(last[0] - (cx0_cfg or 0))
                near_x0 = abs(first[0] - (cx0_cfg or 0))
                if near_x > near_x0:
                    print("[approach]   목표에서 오히려 멀어졌다 → 이동 방향 부호가 반대다"
                          " (Y_RIGHT 또는 V_FWD 부호를 뒤집을 것).")
                else:
                    print("[approach]   접근은 하는데 시간이 부족했다 → APPROACH_TIMEOUT 을 늘리거나"
                          " STEP/V_FWD 를 키울 것.")
    return False


# ─────────────────────── 문열기(open_door2 포팅) ───────────────────────
def open_door(robot, poses, rec=None, ohcap=None):
    """openready→오른쪽strafe→전진→opencatch(손잡이 물기)→후진+1번회전→laundry_back 복귀
    +상쇄주행→openseasame→왼쪽strafe+전진. (open_door2.py ZMQ 포팅)"""
    print(f"[door] 준비자세 '{OPENREADY}'")
    move_to_pose(poses[OPENREADY], robot=robot, duration=HOME_DURATION, fps=FPS)
    print(f"[door] 오른쪽 strafe {OPEN_STRAFE_SEC}s + 전진 {OPEN_FORWARD_SEC}s")
    strafe(robot, +1, OPEN_STRAFE_SEC, rec=rec, ohcap=ohcap)
    fwd(robot, +1, OPEN_FORWARD_SEC, rec=rec, ohcap=ohcap)

    print(f"[door] 손잡이 무는 자세 '{OPENCATCH}'")
    move_to_pose(poses[OPENCATCH], robot=robot, duration=HOME_DURATION, fps=FPS, hold_gripper=False)

    # 후진 + shoulder_pan(1번)만 nothing 값까지 회전(나머지 opencatch 유지)
    target_pan = load_pose(NOTHING_POSE)["shoulder_pan"]
    tgt_norm = raw_to_norm("shoulder_pan", target_pan)
    print(f"[door] 후진 {BACKDRIVE_SEC}s + 1번→{NOTHING_POSE}({target_pan}) 회전")
    obs = robot.get_observation()
    start_pan = float(obs.get("arm_shoulder_pan.pos", 0.0))
    hold = {k: float(v) for k, v in obs.items()
            if isinstance(v, (int, float)) and k.startswith("arm_") and k.endswith(".pos")}
    n = max(1, int(BACKDRIVE_SEC * FPS))
    for i in range(n):
        a = (i + 1) / n
        g = dict(hold)
        g["arm_shoulder_pan.pos"] = start_pan + (tgt_norm - start_pan) * a
        g.update({"x.vel": -V_FWD, "y.vel": 0.0, "theta.vel": 0.0})
        robot.send_action(g)
        if rec is not None and i % RECORD_STRIDE == 0:
            rec.add(robot.get_observation(), (ohcap.read()[1] if ohcap else None))
        time.sleep(1.0 / FPS)
    _stop(robot)

    # laundry_back 자세 → 시작 위치 상쇄 주행(net strafe/drive)
    print(f"[door] 복귀 자세 '{BACK_POSE}' + 시작 위치 상쇄")
    move_to_pose(poses[BACK_POSE], robot=robot, duration=HOME_DURATION, fps=FPS, hold_gripper=True)
    net_strafe = OPEN_STRAFE_SEC              # 오른쪽으로 간 것
    net_drive = OPEN_FORWARD_SEC - BACKDRIVE_SEC  # +면 순전진
    strafe(robot, -1 if net_strafe > 0 else +1, abs(net_strafe), rec=rec, ohcap=ohcap)
    fwd(robot, -1 if net_drive > 0 else +1, abs(net_drive), rec=rec, ohcap=ohcap)

    print("[door] openseasame 재생")
    if not replay_motion(robot, "openseasame", rec=rec, ohcap=ohcap):
        return False
    print(f"[door] 모션 후 왼쪽 strafe {POST_OPEN_STRAFE_SEC}s + 전진 {POST_OPEN_FORWARD_SEC}s")
    strafe(robot, -1, POST_OPEN_STRAFE_SEC, rec=rec, ohcap=ohcap)
    fwd(robot, +1, POST_OPEN_FORWARD_SEC, rec=rec, ohcap=ohcap)
    return True


# ─────────────────────── 녹화 ───────────────────────
class Recorder:
    def __init__(self, root, fps=10):
        self.root, self.fps = root, fps
        self.dirs = {n: os.path.join(root, n) for n in ("front", "wrist", "overhead")}
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        self.n = 0
        print(f"  🎥 녹화 → {root}/")

    def add(self, obs, overhead):
        ts = f"{time.time():017.6f}.jpg"
        for nm in ("front", "wrist"):
            fr = obs.get(nm)
            if isinstance(fr, np.ndarray):
                cv2.imwrite(os.path.join(self.dirs[nm], ts), cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
        if overhead is not None:
            cv2.imwrite(os.path.join(self.dirs["overhead"], ts), overhead)
        self.n += 1

    def close(self):
        print(f"  🎥 녹화 {self.n}프레임. mp4 합치는 중...")
        for nm, d in self.dirs.items():
            if not any(f.endswith(".jpg") for f in os.listdir(d)):
                continue
            out = os.path.join(self.root, f"{nm}.mp4")
            subprocess.run(["ffmpeg", "-y", "-nostdin", "-framerate", str(self.fps),
                            "-pattern_type", "glob", "-i", os.path.join(d, "*.jpg"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", out],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"  📂 {self.root}")


def show(win, obs, overhead, note):
    tiles = []
    for nm in ("front", "wrist"):
        fr = obs.get(nm)
        t = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR) if isinstance(fr, np.ndarray) else np.zeros((480, 480, 3), np.uint8)
        if t.shape[0] != 480:
            t = cv2.resize(t, (int(t.shape[1] * 480 / t.shape[0]), 480))
        tiles.append(t)
    oh = overhead if overhead is not None else np.zeros((480, 640, 3), np.uint8)
    tiles.append(cv2.resize(oh, (640, 480)))
    disp = cv2.hconcat(tiles)
    cv2.rectangle(disp, (0, disp.shape[0] - 30), (disp.shape[1], disp.shape[0]), (0, 0, 0), -1)
    cv2.putText(disp, note, (10, disp.shape[0] - 9), _FONT, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
    cv2.imshow(win, disp)
    return cv2.waitKey(1) & 0xFF


def ensure_host(ip, timeout=30):
    """Jetson 에 lekiwi_host 가 떠 있는지 확인하고, 없으면 SSH 로 띄운다(시작 전 preload).
    graceful 아닌 kill 로 나중에 내려도 되게, 여기선 '떠 있게'만 보장한다."""
    uh = f"comnet02@{ip}"
    script = (
        "if pgrep -f 'lekiwi_host --robot' >/dev/null; then echo ALREADY_UP; exit 0; fi\n"
        "cd ~/lerobot\n"
        "cat > /tmp/run_host.sh <<'INNER'\n"
        "#!/bin/bash\n"
        "cd ~/lerobot\n"
        "yes '' | ~/miniforge3/envs/lerobot/bin/python -m lerobot.robots.lekiwi.lekiwi_host "
        "--robot.id=my_awesome_kiwi --host.connection_time_s=7200\n"
        "INNER\n"
        "chmod +x /tmp/run_host.sh\n"
        "setsid /tmp/run_host.sh > /tmp/lekiwi_host.log 2>&1 < /dev/null &\n"
        "echo STARTED\n"
    )
    try:
        r = subprocess.run(["ssh", "-o", "ConnectTimeout=6", uh, "bash -s"],
                           input=script, capture_output=True, text=True, timeout=20)
    except Exception as e:
        print(f"[preload] ⚠️ host SSH 실패({e}) — Jetson 에서 수동으로 host 를 띄우세요."); return False
    if "ALREADY_UP" in r.stdout:
        print("[preload] host 이미 UP"); return True
    print("[preload] host 시작(SSH) — 부팅 대기...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            c = subprocess.run(["ssh", "-o", "ConnectTimeout=6", uh,
                                "pgrep -f 'lekiwi_host --robot' >/dev/null && echo UP"],
                               capture_output=True, text=True, timeout=10)
            if "UP" in c.stdout:
                print("[preload] host UP"); time.sleep(2); return True
        except Exception:
            pass
        time.sleep(2)
    print("[preload] ⚠️ host 확인 실패"); return False


# ─────────────────────── main ───────────────────────
def main():
    ap = argparse.ArgumentParser(description="세탁물 태스크 v3(랩탑 ZMQ) 전체 파이프라인.")
    ap.add_argument("checkpoint", nargs="?", default=DEFAULT_CKPT)
    ap.add_argument("--wireless", action="store_true", help="무선(192.168.0.19). 기본 유선(192.168.55.1)")
    ap.add_argument("--record", action="store_true", help="front/wrist/상공 녹화(runs/<ts>/)")
    ap.add_argument("--record-dir", default=None)
    ap.add_argument("--skip-approach", action="store_true", help="이미 세탁기 앞이면 접근 생략")
    ap.add_argument("--skip-door", action="store_true", help="문 열려있으면 문열기 생략")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    remote_ip = os.environ.get("REMOTE_IP") or (
        "192.168.0.19" if (args.wireless or os.environ.get("WIRELESS")) else "192.168.55.1")

    # ─────── preload (시작 전 전부: host + VLA + VLM) ───────
    print("=" * 60)
    print(f"[preload] 통신={remote_ip}  녹화={'ON' if args.record else 'OFF'}  "
          f"approach={'skip' if args.skip_approach else 'on'}  door={'skip' if args.skip_door else 'on'}")
    print("[preload] host 확인/기동 (Jetson lekiwi_host)...")
    ensure_host(remote_ip)  # host 를 먼저 띄우고, 부팅되는 동안 아래 모델을 로드
    print("[preload] SmolVLA...")
    policy = SmolVLAPolicy.from_pretrained(args.checkpoint); policy.eval()
    device = torch.device(policy.config.device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": str(device)}})
    robot = LeKiwiClient(LeKiwiClientConfig(remote_ip=remote_ip, id="lekiwi", connect_timeout_s=15))
    renamed = {RENAME.get(k, k): v for k, v in dict(robot.observation_features).items()}
    ds_features = {**hw_to_dataset_features(renamed, OBS_STR),
                   **hw_to_dataset_features(robot.action_features, ACTION)}
    print("[preload] 상공/손목 probe...")
    overhead_gate = GraspGate(str(REPO / "grasp_probe_overhead.pt"), "grabbed", GRASP_THRESHOLD, GRASP_HOLD, device=str(device))
    wrist_gate = GraspGate(str(REPO / "grasp_probe_wrist.pt"), "grabbed", WRIST_THRESHOLD, 1, device="cpu")
    # /dev/videoN 번호는 본체마다·재연결마다 바뀐다 → 'auto' 면 실제로 프레임 나오는 USB 노드를 찾는다.
    cam_path = OVERHEAD_CAM
    if cam_path == "auto":
        from find_overhead_cam import resolve_overhead_cam
        cam_path, cam_log = resolve_overhead_cam("auto", os.environ.get("OVERHEAD_MATCH", ""))
        for line in cam_log:
            print(f"[preload]   {line}")
        if cam_path is None:
            print("[preload] ✗ 상공 카메라를 못 찾음 — USB 연결 확인"); return 8
    print(f"[preload] 상공 카메라 {cam_path}...")
    ohcap = open_v4l2_camera(cam_path)
    print("[preload] 포즈/모션/red 파라미터...")
    pose_names = {DRIVE_POSE, OPENREADY, OPENCATCH, BACK_POSE, READY_POSE, CARRY_POSE}
    poses = {n: pose_to_action(n) for n in pose_names}
    _ = load_pose(NOTHING_POSE)  # shoulder_pan 참조
    cal = load_red_params()
    # throw/openseasame 존재 확인
    for m in ("laundry_throw", "openseasame"):
        if not (REPO / "motions" / f"{m}.json").exists():
            print(f"  ⚠️ 모션 없음: {m}")

    robot.connect(); policy.reset()
    win = "laundry_task3  [Q/ESC] abort"
    gui = True
    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error:
        gui = False
    print("[preload] 완료. 로딩 스톨 없음.\n")

    rec = Recorder(args.record_dir or str(REPO / "runs" / time.strftime("%Y%m%d_%H%M%S")),
                   fps=max(1, FPS // RECORD_STRIDE)) if args.record else None

    if not args.yes:
        # 배너는 반드시 줄바꿈으로 끝내고 flush 한다 — input() 의 프롬프트는 줄바꿈이 없어서
        # 파이프로 읽는 쪽(laundry_task4)에서 버퍼에 갇혀 화면에 안 보인다.
        print("\n" + "=" * 60, flush=True)
        print("  ⏸  준비 완료. 여기서 대기 중입니다 — 로딩이 아닙니다.", flush=True)
        print("  ▶  손 떼고 Enter 를 누르면 로봇이 스스로 움직입니다.", flush=True)
        print("=" * 60, flush=True)
        input("  >>> Enter 를 누르세요: ")
        print("  ▶ 시작합니다.", flush=True)

    def read_oh():
        ok, f = ohcap.read(); return f if ok else None

    code = 0
    try:
        # [0] 주행 자세
        move_to_pose(poses[DRIVE_POSE], robot=robot, duration=HOME_DURATION, fps=FPS)

        # [1] approach
        if not args.skip_approach:
            if not approach(robot, cal, rec=rec, ohcap=ohcap, gui_win=win if gui else None):
                print("접근 실패/중단 → 종료"); return 3

        # [2] 문열기
        if not args.skip_door:
            if not open_door(robot, poses, rec=rec, ohcap=ohcap):
                print("문열기 실패 → 종료"); return 4

        # [3] grab: grabready → VLA(상공 게이트 자동 정지)
        print(f"[3] grab: '{READY_POSE}' 복귀 후 VLA 집기")
        move_to_pose(poses[READY_POSE], robot=robot, duration=HOME_DURATION, fps=FPS)
        policy.reset(); overhead_gate.reset()
        stop_obs = None; frame_i = 0
        while True:
            t0 = time.perf_counter()
            obs = robot.get_observation(); oh = read_oh()
            obs_frame = build_dataset_frame(ds_features, {RENAME.get(k, k): v for k, v in obs.items()}, prefix=OBS_STR)
            at = predict_action(observation=obs_frame, policy=policy, device=device,
                                preprocessor=preprocessor, postprocessor=postprocessor,
                                use_amp=device.type == "cuda", task=TASK_DESCRIPTION, robot_type=robot.name)
            action = make_robot_action(at, ds_features)
            for k in BASE_KEYS:
                action[k] = 0.0
            robot.send_action(action)
            frame_i += 1
            if rec is not None and frame_i % RECORD_STRIDE == 0:
                rec.add(obs, oh)
            if oh is not None and frame_i % GRASP_CHECK_EVERY == 0:
                grabbed, prob, _ = overhead_gate.update(cv2.cvtColor(oh, cv2.COLOR_BGR2RGB))
                if grabbed:
                    print(f"[3] ▶ 상공 GRABBED (p={prob:.2f}) → VLA 정지"); stop_obs = obs; break
            if gui and show(win, obs, oh, "grab: VLA running") in (ord("q"), 27):
                print("[중단]"); return 5
            time.sleep(max(1.0 / FPS - (time.perf_counter() - t0), 0.0))

        # [4] 손목 확정 + [5] 압착
        confirmed = True
        wf = stop_obs.get("wrist") if stop_obs else None
        if isinstance(wf, np.ndarray):
            wtop, _, wsc = wrist_gate.predict(wf)
            confirmed = (wtop == "grabbed" and wsc.get("grabbed", 0.0) >= WRIST_THRESHOLD)
            print(f"[4] 손목 확정: {wtop}(p={wsc.get('grabbed',0.0):.2f}) → " + ("✅잡음" if confirmed else "✗헛집음"))
        if confirmed and GRIP_SQUEEZE_SEC > 0:
            arm = {k: float(v) for k, v in stop_obs.items()
                   if isinstance(v, (int, float)) and k.startswith("arm_") and k.endswith(".pos")}
            print(f"[5] squeeze {GRIP_SQUEEZE_SEC}s")
            for _ in range(int(GRIP_SQUEEZE_SEC * FPS)):
                g = dict(arm); g["arm_gripper.pos"] = GRIP_SQUEEZE
                g.update({k: 0.0 for k in BASE_KEYS})
                robot.send_action(g); time.sleep(1.0 / FPS)

        # [6] carry(laundry_default, 그리퍼 유지)
        print(f"[6] carry '{CARRY_POSE}'")
        move_to_pose(poses[CARRY_POSE], robot=robot, duration=HOME_DURATION, fps=FPS, hold_gripper=True)

        # [7] 복귀 주행
        print(f"[7] 복귀: 후진{BACKUP_SEC}s→회전{ROTATE_SEC}s→직진{FORWARD_SEC}s (⚠️재보정)")
        fwd(robot, -1, BACKUP_SEC, rec=rec, ohcap=ohcap)
        rotate(robot, True, ROTATE_SEC, rec=rec, ohcap=ohcap)
        fwd(robot, +1, FORWARD_SEC, rec=rec, ohcap=ohcap)

        # [8] throw
        print("[8] throw (laundry_throw)")
        if not replay_motion(robot, "laundry_throw", rec=rec, ohcap=ohcap):
            print("throw 모션 실패 → 종료"); return 6
        time.sleep(0.5)
        if RETREAT_SEC > 0:
            fwd(robot, -1, RETREAT_SEC, rec=rec, ohcap=ohcap)

        # [9] 전달 모션 — 마지막 후퇴 후, SO101 이 수건을 이어받을 수 있게 내미는 동작.
        #     HANDOFF_MOTION 이 비어 있으면(기본) 생략한다.
        if HANDOFF_MOTION:
            print(f"[9] 전달 모션 '{HANDOFF_MOTION}' (SO101 인계)")
            if not replay_motion(robot, HANDOFF_MOTION, rec=rec, ohcap=ohcap):
                print("전달 모션 실패 → 종료"); return 7
            print("\n✅ approach→문열기→grab→carry→복귀→throw→전달 완료.")
        else:
            print("\n✅ approach→문열기→grab→carry→복귀→throw 완료. (HANDOFF_MOTION 미설정 → 전달 모션 생략)")
    except KeyboardInterrupt:
        print("\n[중단] Ctrl+C")
        code = 130
    finally:
        try:
            _stop(robot)
        except Exception:
            pass
        robot.disconnect()
        if ohcap is not None:
            ohcap.release()
        if gui:
            cv2.destroyAllWindows()
        if rec is not None:
            rec.close()
        print("완료.")
    return code


if __name__ == "__main__":
    with torch.inference_mode():
        # 종료코드로 성공/실패를 알린다(laundry_task4.py 가 이 값으로 다음 단계 진행 여부를 결정).
        # 0=성공 3=approach 4=문열기 5=abort 6=throw 7=전달모션 130=Ctrl+C
        sys.exit(main() or 0)
