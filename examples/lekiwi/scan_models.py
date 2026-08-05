"""csi-agent 안의 모든 학습 산출물을 스캔해 '실제로 쓸 수 있는' 체크포인트를 가린다.

읽기 전용이다. 로봇은 건드리지 않는다.
판정 항목: last 링크 해석 / 필수 파일 존재 / 읽기 권한 / 설정 파싱 / mtime.
"""
import json
import os
import time
from pathlib import Path

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_paths import csi_root, train as _train  # noqa: E402  경로는 csi_paths 한 곳에서

ROOT = csi_root()
TRAIN = _train()

NEEDED = ["config.json", "model.safetensors"]


def readable(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            f.read(8)
        return True
    except OSError:
        return False


def ck_info(ck: Path) -> dict:
    """체크포인트 디렉터리 하나를 조사한다. lerobot 은 pretrained_model/ 하위에 두기도 한다."""
    d = ck / "pretrained_model" if (ck / "pretrained_model").is_dir() else ck
    info = {"path": str(d), "ok": True, "why": []}
    for n in NEEDED:
        f = d / n
        if not f.exists():
            info["ok"] = False
            info["why"].append(f"{n} 없음")
        elif not readable(f):
            info["ok"] = False
            info["why"].append(f"{n} 읽기 거부(권한 {oct(f.stat().st_mode)[-3:]}, 소유 {f.owner()})")
    cfg = d / "config.json"
    if cfg.exists() and readable(cfg):
        try:
            c = json.loads(cfg.read_text())
            info["type"] = c.get("type", "?")
        except Exception as e:
            info["ok"] = False
            info["why"].append(f"config 파싱 실패: {e}")
    sm = d / "model.safetensors"
    if sm.exists():
        info["size_mb"] = sm.stat().st_size / 1e6
        info["mtime"] = sm.stat().st_mtime
    return info


rows = []
for tdir in sorted(TRAIN.iterdir()):
    cks = tdir / "checkpoints"
    if not cks.is_dir():
        continue
    last = cks / "last"
    entry = {"name": tdir.name, "last_raw": None, "resolved": None}
    if last.is_symlink():
        entry["last_raw"] = os.readlink(last)
    if last.exists():
        entry["resolved"] = last.resolve().name
        entry.update(ck_info(last))
    else:
        entry["ok"] = False
        entry["why"] = ["last 없음/깨진 링크" if last.is_symlink() else "last 자체가 없음"]
        # 대안: 번호가 가장 큰 체크포인트
        nums = sorted([p for p in cks.iterdir() if p.name.isdigit()], key=lambda p: int(p.name))
        if nums:
            entry["fallback"] = nums[-1].name
            fb = ck_info(nums[-1])
            entry["fallback_ok"] = fb["ok"]
            entry["fallback_why"] = fb["why"]
    rows.append(entry)

print(f"{'디렉터리':<48} {'last→':<10} {'상태':<6} {'크기MB':>8}  {'수정시각':<16} 비고")
print("─" * 130)
for r in sorted(rows, key=lambda r: -(r.get("mtime") or 0)):
    t = time.strftime("%m-%d %H:%M", time.localtime(r["mtime"])) if r.get("mtime") else "-"
    size = f"{r['size_mb']:.0f}" if r.get("size_mb") else "-"
    status = "✅ OK" if r.get("ok") else "✗ 불가"
    note = "; ".join(r.get("why", []))
    if r.get("fallback"):
        note += f"  → 대안 {r['fallback']} ({'OK' if r.get('fallback_ok') else '역시 불가'})"
    print(f"{r['name']:<48} {str(r.get('resolved') or '-'):<10} {status:<6} {size:>8}  {t:<16} {note}")

print("\n\n=== rollout_auto 가 실제로 참조하는 경로 ===")
for i in range(4):
    p = TRAIN / f"rollout_hil_towel_fold01_step{i}" / "checkpoints" / "last"
    print(f"  정책 step{i + 1} ← {p.parent.parent.name}/checkpoints/last", end="  ")
    if p.exists():
        print(f"→ {p.resolve().name}  {'✅' if ck_info(p)['ok'] else '✗ ' + str(ck_info(p)['why'])}")
    else:
        print("✗ 없음")

print("\n=== 분류기 ===")
for c in sorted(TRAIN.glob("towel_fold01_nextlevel*")):
    i = ck_info(c)
    t = time.strftime("%m-%d %H:%M", time.localtime(i["mtime"])) if i.get("mtime") else "-"
    print(f"  {c.name:<40} {'✅ OK' if i['ok'] else '✗ ' + str(i['why'])}  {t}")
