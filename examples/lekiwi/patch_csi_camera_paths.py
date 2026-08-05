"""csi-agent 의 setup_devices.py 가 포트 고정 경로 대신 리졸버 결과를 쓰게 한다(멱등, 백업).

왜: /dev/lerobot/camera_N 은 udev 가 **물리 포트**로 만든다. 카메라 4대가 2쌍씩
    벤더·모델·시리얼이 완전히 같아서 udev 가 쓸 수 있는 단서가 포트뿐이기 때문이다.
    그래서 자리를 옮기면 이름이 뒤바뀐다 — 실제로 폴딩 top 이 세탁기 상공캠을
    가리킨 채로 돌아간 사고가 있었다(2026-08-05).

무엇을: camera_0/1/2 → ~/.lerobot/cams/{top,left_cam,right_cam} 으로 바꾼다.
    그 링크는 resolve_cameras.py 가 '모델로 쌍을 가르고 화면 내용으로 쌍 안을 가려'
    만든다. 포트가 바뀌어도 유지된다. 링크가 없으면 예전 경로로 자동 폴백한다.

사용: python patch_csi_camera_paths.py [--check|--revert]
"""
import pathlib
import sys

TARGET = pathlib.Path("/home/lerobot/lerobot2/csi-agent/lhwdev/clothing/scripts/setup_devices.py")
BACKUP = TARGET.with_suffix(".py.bak.campaths")
MARKER = "_cam_path"

HELPER = '''import os as _os

# ── 카메라 경로 해석 (LSJ 패치) ──────────────────────────────────────────────
# /dev/lerobot/camera_N 은 udev 가 물리 포트로 만든다. 카메라 4대가 2쌍씩 벤더·모델·
# 시리얼이 완전히 동일해 udev 단서가 포트뿐이라, 자리를 옮기면 이름이 뒤바뀐다.
# resolve_cameras.py 가 만든 ~/.lerobot/cams/ 링크를 우선 쓰고, 없으면 예전 경로로 돌아간다.
def _cam_path(role, fallback):
    p = _os.path.expanduser(f"~/.lerobot/cams/{role}")
    return p if _os.path.exists(p) else fallback


'''

REPLACEMENTS = [
    ('index_or_path="/dev/lerobot/camera_1"',
     'index_or_path=_cam_path("left_cam", "/dev/lerobot/camera_1")'),
    ('index_or_path="/dev/lerobot/camera_2"',
     'index_or_path=_cam_path("right_cam", "/dev/lerobot/camera_2")'),
    ('index_or_path="/dev/lerobot/camera_0"',
     'index_or_path=_cam_path("top", "/dev/lerobot/camera_0")'),
]

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
    n = sum(src.count(new) for _, new in REPLACEMENTS)
    print(f"이미 적용됨 (카메라 경로 {n}곳이 리졸버 경유)")
    sys.exit(0)

missing = [old for old, _ in REPLACEMENTS if old not in src]
if missing:
    print("✗ 다음 경로를 못 찾음(파일이 바뀌었을 수 있다):")
    for m in missing:
        print(f"    {m}")
    sys.exit(1)

if "--check" in sys.argv:
    print(f"미적용 — 카메라 경로 {len(REPLACEMENTS)}곳을 바꿀 수 있다")
    sys.exit(0)

if not BACKUP.exists():
    BACKUP.write_text(src)

for old, new in REPLACEMENTS:
    src = src.replace(old, new)
src = HELPER + src
TARGET.write_text(src)
print(f"패치 적용됨 — 카메라 경로 {len(REPLACEMENTS)}곳이 ~/.lerobot/cams/ 를 먼저 본다")
print(f"  백업: {BACKUP.name}")
print("  링크가 없으면 예전 /dev/lerobot/camera_N 으로 자동 폴백한다")
