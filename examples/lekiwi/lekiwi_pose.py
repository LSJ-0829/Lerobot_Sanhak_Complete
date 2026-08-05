# !/usr/bin/env python
"""저장된 포즈(JSON, 원시 모터 스텝)를 LeKiwi 클라이언트/리더암으로 재현하는 공용 모듈.

Jetson의 `scripts/arm/set_pose.py` 는 `/dev/ttyACM0` 을 직접 잡기 때문에
`lekiwi_host` 가 떠 있는 동안에는 쓸 수 없다. 이 모듈은 대신 **노트북(클라이언트)
쪽에서** 원시 스텝값을 lerobot 정규화 좌표로 변환해 `robot.send_action()` /
`leader.send_feedback()` 으로 보낸다. 호스트를 끄지 않고도 포즈를 잡을 수 있다.

포즈 JSON 형식 (`poses/<이름>.json`, Jetson `save_current_pose.py` 가 만든 것):
    {"shoulder_pan": 2011, ..., "gripper": 2759}   ← Present_Position 원시 스텝

정규화 규칙은 `lerobot/motors/motors_bus.py::_normalize` 와 동일:
  - DEGREES (팔 5축)      : (raw - (min+max)/2) * 360 / 4095
  - RANGE_0_100 (gripper) : (raw - min) / (max - min) * 100
`min/max` 는 LeKiwi 캘리브레이션(`lekiwi_calibration.json`, Jetson에서 복사본)에서 읽는다.

리더암도 같은 정규화 좌표를 쓰므로(캘리 값만 다를 뿐) 동일한 목표값을 그대로
보내면 리더/팔로워가 같은 자세가 된다.
"""

import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
POSES_DIR = REPO_ROOT / "poses"
CALIB_PATH = Path(__file__).resolve().parent / "lekiwi_calibration.json"

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# 정규화 모드: gripper 만 0~100, 나머지는 각도(deg). LeKiwi/SO101 리더 둘 다 동일.
GRIPPER = "gripper"
BASE_KEYS = ("x.vel", "y.vel", "theta.vel")
MAX_RES = 4095  # sts3215 resolution(4096) - 1

_calib_cache = None


def _calib() -> dict:
    global _calib_cache
    if _calib_cache is None:
        if not CALIB_PATH.exists():
            raise FileNotFoundError(
                f"LeKiwi 캘리브레이션이 없습니다: {CALIB_PATH}\n"
                "Jetson에서 복사하세요:  scp jetson:~/.cache/huggingface/lerobot/"
                "calibration/robots/lekiwi/my_awesome_kiwi.json "
                f"{CALIB_PATH}"
            )
        _calib_cache = json.loads(CALIB_PATH.read_text())
    return _calib_cache


def raw_to_norm(joint: str, raw: float) -> float:
    """원시 스텝 → lerobot 정규화 값(팔=deg, gripper=0~100)."""
    entry = _calib()[f"arm_{joint}"]
    lo, hi = entry["range_min"], entry["range_max"]
    if joint == GRIPPER:
        bounded = min(hi, max(lo, raw))
        return (bounded - lo) / (hi - lo) * 100.0
    mid = (lo + hi) / 2
    return (raw - mid) * 360.0 / MAX_RES


def load_pose(name: str) -> dict[str, int]:
    """`poses/<name>.json` 의 원시 스텝 dict 를 읽는다."""
    path = Path(name) if str(name).endswith(".json") else POSES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"포즈 파일 없음: {path}")
    return json.loads(path.read_text())


def pose_to_action(name: str) -> dict[str, float]:
    """포즈 이름 → LeKiwi 액션 dict(`arm_*.pos` 정규화 + 베이스 속도 0)."""
    raw = load_pose(name)
    missing = [j for j in ARM_JOINTS if j not in raw]
    if missing:
        raise ValueError(f"포즈 '{name}' 에 관절이 빠졌습니다: {missing}")
    action = {f"arm_{j}.pos": raw_to_norm(j, raw[j]) for j in ARM_JOINTS}
    action.update({k: 0.0 for k in BASE_KEYS})
    return action


def _leader_goals(action: dict[str, float]) -> dict[str, float]:
    """LeKiwi 액션 키(`arm_shoulder_pan.pos`) → 리더암 키(`shoulder_pan.pos`)."""
    return {f"{j}.pos": action[f"arm_{j}.pos"] for j in ARM_JOINTS}


def move_to_pose(
    action: dict[str, float],
    robot=None,
    leader=None,
    duration: float = 3.0,
    fps: int = 30,
    verbose: bool = True,
    hold_gripper: bool = False,
    settle_s: float = 2.0,
    settle_tol: float = 1.5,
) -> None:
    """현재 자세 → 목표 포즈로 선형 보간해 천천히 이동한다.

    robot / leader 둘 다 주면 **동시에** 같은 목표로 움직여 핸드오버 점프를 막는다.
    리더암은 이동하는 동안만 토크를 켠다(끄는 시점은 호출자가 정하도록 여기선 유지하지 않음).

    hold_gripper=True 면 목표 포즈의 그리퍼 값을 무시하고 **현재 그리퍼 위치를 유지**한다.
    VLA 로 수건을 잡은 뒤 준비자세(grabready)로 팔만 복귀시킬 때, 준비자세의 열린
    그리퍼 목표(예: laundry_grabready=2762≈열림)로 덮어써서 수건을 놓치는 것을 막는다.
    (robot 관측에서 현재 그리퍼 값을 읽어 목표로 삼으므로 robot 이 있어야 동작.)
    """
    if robot is None and leader is None:
        return
    steps = max(1, int(duration * fps))
    arm_keys = [f"arm_{j}.pos" for j in ARM_JOINTS]

    start_r = None
    if robot is not None:
        obs = robot.get_observation()
        start_r = {k: float(obs[k]) for k in arm_keys}
        if hold_gripper:
            # 그리퍼는 현재(닫힘) 위치로 목표를 고정 → 잡은 수건을 놓지 않게 한다.
            action = {**action, "arm_gripper.pos": start_r["arm_gripper.pos"]}

    start_l = None
    if leader is not None:
        cur = leader.get_action()
        start_l = {f"{j}.pos": float(cur[f"{j}.pos"]) for j in ARM_JOINTS}
        leader.enable_torque()

    target_l = _leader_goals(action)
    if verbose:
        who = " + ".join(x for x in ("팔로워" if robot else "", "리더" if leader else "") if x)
        print(f"[포즈] {who} → 목표 자세로 {duration:.1f}초간 이동")

    for i in range(1, steps + 1):
        a = i / steps
        if robot is not None:
            goal = {k: start_r[k] + (action[k] - start_r[k]) * a for k in arm_keys}
            goal.update({k: 0.0 for k in BASE_KEYS})
            try:
                robot.send_action(goal)
            except Exception as e:  # 통신 튐은 무시하고 계속(다음 틱에 목표를 다시 보냄)
                print(f"[포즈] send_action 실패(무시): {e}")
        if leader is not None:
            lg = {k: start_l[k] + (target_l[k] - start_l[k]) * a for k in target_l}
            try:
                leader.send_feedback(lg)
            except Exception as e:
                print(f"[포즈] leader send_feedback 실패(무시): {e}")
        time.sleep(1.0 / fps)

    # ── 도달 대기 ──────────────────────────────────────────────────────────
    # 보간 웨이포인트를 다 보냈다고 팔이 거기 있는 건 아니다. 모터에 Goal_Velocity
    # 상한(800)과 Acceleration(40)이 걸려 있어 마지막 목표까지 따라오는 데 시간이
    # 걸린다(감속 구간만 ~0.2s). 예전엔 Goal_Velocity=0(무제한)이라 즉시 도달해
    # 이 문제가 안 보였는데, 끊김을 없애려 속도를 제한하면서 드러났다.
    # 확인 없이 반환하면 '지정 자세가 아닌 채로' 다음 단계(접근·문열기)가 시작된다.
    if robot is not None and settle_s > 0:
        deadline = time.time() + settle_s
        worst_j, worst = None, 0.0
        while time.time() < deadline:
            goal = {k: action[k] for k in arm_keys}
            goal.update({k: 0.0 for k in BASE_KEYS})
            try:
                robot.send_action(goal)       # 목표를 계속 눌러 준다
            except Exception:
                pass
            time.sleep(1.0 / fps)
            try:
                obs = robot.get_observation()
            except Exception:
                continue
            worst_j, worst = None, 0.0
            for k in arm_keys:
                if k not in obs:
                    continue
                err = abs(float(obs[k]) - action[k])
                if err > worst:
                    worst_j, worst = k, err
            if worst <= settle_tol:
                break
        if worst > settle_tol:
            print(f"[포즈] ⚠️ {settle_s:.1f}s 안에 도달 못함 — 최대 오차 "
                  f"{worst:.1f}({worst_j}). duration 을 늘리거나 Goal_Velocity 를 올릴 것")
        elif verbose:
            print(f"[포즈] 도달 (최대 오차 {worst:.2f})")
    elif verbose:
        print("[포즈] 도달")


def release_leader(leader, hold_s: float = 2.0, verbose: bool = True) -> None:
    """리더암 토크를 끊기 전에 사람이 잡을 시간을 준다(놓으면 중력에 떨어짐)."""
    if leader is None:
        return
    if verbose and hold_s > 0:
        print(f"[포즈] 리더암을 잡으세요 — {hold_s:.0f}초 뒤 토크를 끕니다.")
    time.sleep(max(hold_s, 0.0))
    try:
        leader.disable_torque()
    except Exception as e:
        print(f"[포즈] 리더 토크 해제 실패(무시): {e}")
    if verbose:
        print("[포즈] 리더암 토크 해제 — 이제 조종 가능")


if __name__ == "__main__":  # 변환값 확인용: python lekiwi_pose.py laundry_grabready
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "laundry_grabready"
    raw = load_pose(name)
    act = pose_to_action(name)
    print(f"포즈 '{name}'")
    for j in ARM_JOINTS:
        unit = "%" if j == GRIPPER else "deg"
        print(f"  {j:<14} raw={raw[j]:>5}  →  {act[f'arm_{j}.pos']:>8.2f} {unit}")
