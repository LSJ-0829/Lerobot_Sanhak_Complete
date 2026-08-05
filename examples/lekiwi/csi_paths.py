"""csi-agent 가 어디 있는지 한 곳에서 정한다.

경로를 파일마다 박아 두었더니(`/home/lerobot/lerobot2/csi-agent/lhwdev`) 계정이나
본체가 바뀌는 순간 전부 깨진다. laundry_task4 는 이미 `CSI_DIR` 환경변수로 받고 있었지만
보조 도구들(record_idle_posture, patch_csi_*, verify_*, scan_models)은 그렇지 않았다.

찾는 순서 — 명시한 값이 항상 이긴다:
  1. `CSI_ROOT`  — csi-agent 루트(그 아래 lhwdev/, tools/, LSJ/ 가 있는 곳)
  2. `CSI_DIR`   — lhwdev 디렉터리. laundry_task4 가 쓰는 것과 같은 변수라 하나만 정하면 된다
  3. 이 파일 위치에서 위로 훑기 — LSJ/examples/lekiwi/ 아래에 있으면 바로 찾아진다
  4. 흔한 자리 몇 곳

찾지 못하면 조용히 넘어가지 않고 예외를 낸다. 없는 경로로 계속 진행하면
'파일이 없다'가 엉뚱한 곳에서 터져서 원인을 찾기 어렵다.
"""
import os
from pathlib import Path

_MARKERS = ("lhwdev/clothing", "clothing")   # 이게 있으면 csi-agent 루트로 본다


def _looks_like_root(p: Path) -> bool:
    return p.is_dir() and any((p / m).is_dir() for m in _MARKERS)


def _candidates():
    env_root = os.environ.get("CSI_ROOT")
    if env_root:
        yield Path(os.path.expanduser(env_root))
    env_dir = os.environ.get("CSI_DIR")
    if env_dir:
        # CSI_DIR 은 lhwdev 를 가리킨다 — 루트는 그 부모다.
        yield Path(os.path.expanduser(env_dir)).parent
    # 이 파일이 csi-agent 안에 있으면(LSJ/examples/lekiwi/…) 위로 올라가다 만난다
    for up in Path(__file__).resolve().parents:
        yield up
    home = Path.home()
    for rel in ("lerobot2/csi-agent", "csi-agent", "lerobot/csi-agent"):
        yield home / rel


def csi_root() -> Path:
    for c in _candidates():
        try:
            if _looks_like_root(c):
                return c
        except OSError:
            continue
    raise FileNotFoundError(
        "csi-agent 를 못 찾았습니다. 위치를 알려주세요:\n"
        "  export CSI_ROOT=/home/<계정>/lerobot2/csi-agent\n"
        "  (또는 CSI_DIR 로 lhwdev 디렉터리를 직접)")


def work_dir() -> Path:
    """모델·코드가 실제로 있는 곳. 보통 <root>/lhwdev 이지만, 루트에 clothing/ 이
    바로 있는 배치도 있어서 둘 다 받는다."""
    root = csi_root()
    return root / "lhwdev" if (root / "lhwdev" / "clothing").is_dir() else root


def clothing() -> Path:
    return work_dir() / "clothing"


def train() -> Path:
    return work_dir() / "train"


if __name__ == "__main__":
    print(f"root     : {csi_root()}")
    print(f"work_dir : {work_dir()}")
    print(f"clothing : {clothing()}{'' if clothing().is_dir() else '   ✗ 없음'}")
    print(f"train    : {train()}{'' if train().is_dir() else '   ✗ 없음'}")
