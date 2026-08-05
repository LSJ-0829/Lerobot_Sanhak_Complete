"""csi-agent 의 rollout.py 에 모터 연결 재시도를 심는다(멱등, 백업 생성).

왜: SO101 팔이 가끔 첫 연결에서 인식되지 않는다(시리얼 버스가 아직 정리되지 않았거나
모터 응답이 늦는 경우). 지금은 그 자리에서 예외로 죽어 파이프라인이 통째로 끝난다.
수건을 이미 전달받은 뒤라 특히 손해가 크다.

무엇을: robot.connect() 를 _connect_with_retry() 로 감싼다.
  - 실패하면 부분 연결 상태를 정리(disconnect)한 뒤 잠깐 쉬고 다시 시도
  - 대기는 점증(2s, 3s, 4.5s…)해서, 장치가 재열거 중이면 그 사이에 준비되게 한다
  - 매 시도의 실패 사유를 남긴다 — 전원/케이블 문제인지 일시적 문제인지 구분하려고
  - 전부 실패하면 마지막 예외를 그대로 올린다(조용히 넘어가면 더 위험하다)

사용: 시연 본체에서  python patch_csi_motor_retry.py [--check|--revert]
"""
import pathlib
import sys

import sys as _sys
_sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from csi_paths import clothing  # noqa: E402  경로는 csi_paths 한 곳에서 정한다

TARGET = clothing() / "rollout.py"
BACKUP = TARGET.with_suffix(".py.bak.motorretry")

# 참고: setup_devices.py:76 의 robot.connect() 는 여기서 감싸지 않는다.
#   실제 시연에서 터진 건 그쪽이었지만(ConnectionError: no status packet on id_=3),
#   남의 파일을 여러 군데 고치는 대신 laundry_task4 가 rollout 프로세스 자체를
#   최대 10회 재시작하도록 했다. 실패 지점이 어디든 덮이고 되돌리기도 쉽다.

ANCHOR = "from lerobot.robots import make_robot_from_config"

HELPER = '''

# ── 모터 연결 재시도 (LSJ 패치) ────────────────────────────────────────────────
# SO101 팔이 가끔 첫 connect() 에서 인식되지 않는다. 예전엔 그 자리에서 죽어
# 파이프라인이 통째로 끝났다(수건을 이미 전달받은 뒤라 손해가 크다).
def _connect_with_retry(robot, attempts: int = 5, delay: float = 2.0, backoff: float = 1.5):
    """robot.connect() 를 최대 attempts 회 시도한다. 전부 실패하면 마지막 예외를 올린다."""
    import logging
    import time as _time

    log = logging.getLogger(__name__)
    last = None
    for i in range(1, attempts + 1):
        try:
            robot.connect()
            if i > 1:
                log.info("[motor-retry] %d번째 시도에서 연결 성공", i)
            return robot
        except Exception as e:  # noqa: BLE001 - 어떤 버스 예외든 재시도 대상
            last = e
            log.warning("[motor-retry] 연결 실패 %d/%d: %s", i, attempts,
                        str(e).splitlines()[0][:200])
            # 부분적으로 열린 포트를 닫아 둔다. 안 닫으면 다음 시도가 '포트 사용 중'으로 또 실패한다.
            try:
                if getattr(robot, "is_connected", False):
                    robot.disconnect()
            except Exception:
                pass
            if i < attempts:
                log.info("[motor-retry] %.1f초 후 재시도", delay)
                _time.sleep(delay)
                delay *= backoff
    log.error("[motor-retry] %d회 모두 실패 — 전원/USB 케이블을 확인하세요", attempts)
    raise last
'''

OLD_CALL = "        robot.connect()\n"
NEW_CALL = "        _connect_with_retry(robot)\n"
MARKER = "_connect_with_retry"

if not TARGET.exists():
    print(f"✗ 대상 없음: {TARGET}")
    sys.exit(1)

if "--revert" in sys.argv:
    if BACKUP.exists():
        TARGET.write_text(BACKUP.read_text())
        print(f"원복 완료 ({BACKUP.name} → {TARGET.name})")
    else:
        print("✗ 백업이 없어 원복할 수 없다")
    sys.exit(0)

src = TARGET.read_text()

if MARKER in src:
    n = src.count(NEW_CALL.strip())
    print(f"이미 적용됨 (connect 호출 {n}곳이 재시도 경유)")
    sys.exit(0)

if "--check" in sys.argv:
    print(f"미적용 — connect() 호출 {src.count(OLD_CALL)}곳을 감쌀 수 있다")
    sys.exit(0)

if ANCHOR not in src:
    print(f"✗ 삽입 위치를 못 찾음: {ANCHOR!r}")
    sys.exit(1)
if OLD_CALL not in src:
    print("✗ robot.connect() 호출을 못 찾음 (들여쓰기가 다를 수 있다)")
    sys.exit(1)

if not BACKUP.exists():
    BACKUP.write_text(src)

# ⚠️ 순서와 매칭 방식이 중요하다.
#   - 호출부를 '먼저' 바꾸고 그 뒤에 헬퍼를 넣는다. 반대로 하면 헬퍼 안의
#     robot.connect() 까지 바뀌어 자기 자신을 부르는 무한 재귀가 된다(실제로 겪음).
#   - str.replace 는 부분문자열을 잡으므로 들여쓰기가 더 깊은 호출까지 걸린다.
#     줄 전체(^...$)로 앵커링해 8칸 들여쓰기 호출만 정확히 바꾼다.
import re  # noqa: E402

pat = re.compile(r"^        robot\.connect\(\)$", re.M)
n = len(pat.findall(src))
if n == 0:
    print("✗ 8칸 들여쓰기 robot.connect() 호출을 못 찾음")
    sys.exit(1)
src = pat.sub("        _connect_with_retry(robot)", src)
src = src.replace(ANCHOR, ANCHOR + HELPER, 1)
TARGET.write_text(src)
print(f"패치 적용됨 — connect() 호출 {n}곳을 재시도로 감쌌다 (백업: {BACKUP.name})")
print("  최대 5회, 대기 2.0→3.0→4.5→6.75초, 매 시도 실패 사유를 로그에 남긴다")
