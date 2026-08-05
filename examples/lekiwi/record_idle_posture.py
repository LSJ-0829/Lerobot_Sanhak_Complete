"""현재 SO101 양팔 자세를 idle_posture.json 으로 저장한다(홈 복귀 기준 자세).

csi-agent 의 RobotHoming.load_idle_posture() 는 clothing/idle_posture.json 을 찾는데
그 파일이 아예 없었다 → 항상 None 이라 홈 복귀 루틴이 한 번도 돌지 않았다
(2026-08-05 확인. '팔이 제자리로 세팅되지 않는다'의 원인).

사용:
  팔을 원하는 기준 자세로 둔 뒤   python record_idle_posture.py
  확인만                          python record_idle_posture.py --show

⚠️ 연결하면 토크가 켜져 팔이 현재 자세를 유지한다. 다시 손으로 옮기려면 토크를 꺼야 한다.
"""
import argparse
import json
import sys
import time
from pathlib import Path

def out_path() -> Path:
    """저장 위치. 모듈을 읽는 시점이 아니라 쓸 때 정한다 —
    laundry_task4 가 mk_arms/connect_with_retry 만 쓰려고 이 파일을 import 하는데,
    그때 csi-agent 를 못 찾는다고 import 자체가 실패하면 곤란하다."""
    from csi_paths import clothing  # 경로는 csi_paths 한 곳에서 정한다
    return clothing() / "idle_posture.json"


def _hard_disconnect(robot):
    """로봇/팔/버스를 단계적으로 닫는다.

    부분적으로 열린 상태로 두면 다음 시도가 'FeetechMotorsBus is already connected' 로
    실패해서, 재시도가 전부 같은 이유로 헛돈다(실제로 겪음).
    """
    targets = [robot, getattr(robot, "bus", None)]
    for attr in ("left_arm", "right_arm"):
        arm = getattr(robot, attr, None)
        if arm is not None:
            targets += [arm, getattr(arm, "bus", None)]
    for t in targets:
        if t is None:
            continue
        try:
            if getattr(t, "is_connected", False):
                t.disconnect()
        except Exception:
            pass



def mk_arms():
    """카메라 없이 팔만 여는 SOFollower 2개.

    id 는 SOFollowerConfig 가 아니라 RobotConfig 쪽 필드라 둘을 합친 SOFollowerRobotConfig 를 쓴다.
    id 는 캘리브레이션 파일 이름과 같아야 한다(lhwdev_follower_bimanual_{left,right}.json).
    disable_torque_on_disconnect=False : 끊은 뒤에도 팔이 자세를 유지하게 둔다
    (기본 True 면 끊는 순간 토크가 풀려 중력으로 떨어진다).
    """
    from lerobot.robots.so_follower import SOFollower
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

    def mk(port, cal_id):
        return SOFollower(SOFollowerRobotConfig(
            port=port, id=cal_id, cameras={}, disable_torque_on_disconnect=False))

    return {
        "left": mk("/dev/lerobot/follower_1", "lhwdev_follower_bimanual_left"),
        "right": mk("/dev/lerobot/follower_2", "lhwdev_follower_bimanual_right"),
    }


def connect_with_retry(robot, attempts=10, delay=3.0):
    """모터가 가끔 첫 연결에서 안 잡힌다. 될 때까지 다시 붙는다."""
    last = None
    for i in range(1, attempts + 1):
        try:
            robot.connect()
            if i > 1:
                print(f"  {i}번째 시도에서 연결 성공")
            return
        except Exception as e:
            last = e
            print(f"  연결 실패 {i}/{attempts}: {str(e).splitlines()[0][:110]}")
            _hard_disconnect(robot)
            if i < attempts:
                time.sleep(delay)
    raise last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="저장하지 않고 현재 값만 출력")
    args = ap.parse_args()

    # ⚠️ setup_devices 의 robot 을 쓰지 않는다. 그건 카메라까지 함께 여는데, 자세를 읽는 데는
    #    카메라가 필요 없고 오히려 카메라 문제로 실패한다(실제로 device busy 로 막혔다).
    #    팔만 여는 SOFollower 를 직접 만든다. 캘리브레이션 id 는 csi-agent 와 동일하게 맞춘다.
    arms = mk_arms()
    joints = {}
    for side, arm in arms.items():
        connect_with_retry(arm)
        obs = arm.get_observation()
        for k, v in obs.items():
            if k.endswith(".pos"):
                joints[f"{side}_{k}"] = float(v)   # rollout 의 관측 키와 같은 형식
        arm.disconnect()

    print(f"\n현재 자세 ({len(joints)}축):")
    for k in sorted(joints):
        print(f"  {k:<28} {joints[k]:8.2f}")

    if args.show:
        print("\n(--show 라 저장하지 않았다)")
        return 0
    if len(joints) != 12:
        print(f"\n✗ 12축이 아니라 {len(joints)}축이다 — 팔 연결을 확인하고 다시 시도할 것")
        return 1

    out = out_path()
    if out.exists():
        bak = out.with_suffix(".json.bak")
        bak.write_text(out.read_text())
        print(f"\n기존 파일 백업: {bak.name}")
    out.write_text(json.dumps({"joint_positions": joints}, indent=2))
    print(f"저장됨: {out}")
    print("이제 rollout 의 RobotHoming 이 이 자세로 복귀한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
