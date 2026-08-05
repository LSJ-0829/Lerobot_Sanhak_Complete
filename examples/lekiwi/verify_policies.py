"""각 체크포인트를 실제로 GPU 에 올리고 더미 추론까지 돌려 '작동'을 확인한다.

파일이 있다고 쓸 수 있는 게 아니다. config 스키마가 어긋나거나 입력 키가 안 맞으면
로딩은 되고 추론에서 죽는다 — 시연 중에 터지는 게 그런 경우다. 여기서 미리 가른다.
읽기 전용이고 로봇은 건드리지 않는다.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_paths import train as _train  # noqa: E402  경로는 csi_paths 한 곳에서 정한다

TRAIN = _train()

# 검증 대상: 파이프라인이 실제로 쓰는 것 + 같은 단계의 대안들
TARGETS = []
for i in range(4):
    TARGETS.append((f"정책step{i + 1} (사용중)", TRAIN / f"rollout_hil_towel_fold01_step{i}" / "checkpoints" / "last"))
TARGETS += [
    ("대안 step0-last깨짐→027000", TRAIN / "rollout_hil_towel_fold01_step0" / "checkpoints" / "027000"),
    ("대안 towel_fold01_step1", TRAIN / "towel_fold01_step1" / "checkpoints" / "last"),
    ("대안 towel_fold01_step2", TRAIN / "towel_fold01_step2" / "checkpoints" / "last"),
    ("대안 towel_fold01_step3", TRAIN / "towel_fold01_step3" / "checkpoints" / "last"),
    ("대안 towel_fold01_step0", TRAIN / "towel_fold01_step0" / "checkpoints" / "last"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}\n")
print(f"{'대상':<34} {'로드':<8} {'추론':<8} {'입력키':<44} 비고")
print("─" * 132)

results = []
for label, path in TARGETS:
    row = {"label": label, "path": str(path), "load": "✗", "infer": "✗", "note": ""}
    if not path.exists():
        row["note"] = "경로 없음"
        results.append(row)
        print(f"{label:<34} {'✗':<8} {'-':<8} {'-':<44} 경로 없음")
        continue
    try:
        d = path / "pretrained_model" if (path / "pretrained_model").is_dir() else path
        t0 = time.time()
        cfg = PreTrainedConfig.from_pretrained(d, local_files_only=True)
        cfg.pretrained_path = d
        policy = make_policy(cfg, ds_meta=None, env_cfg=None) if False else None
        # ds_meta 없이 만들 수 없는 구현이 있어, 정책 클래스에서 직접 로드한다.
        from lerobot.policies.factory import get_policy_class
        policy = get_policy_class(cfg.type).from_pretrained(d, config=cfg)
        policy.to(device).eval()
        row["load"] = f"✅ {time.time() - t0:.1f}s"
        feats = getattr(cfg, "input_features", {}) or {}
        keys = sorted(feats)
        row["keys"] = keys

        # 더미 배치 구성 — config 의 input_features 모양을 그대로 따른다.
        batch = {}
        for k, f in feats.items():
            shape = tuple(f.shape) if hasattr(f, "shape") else tuple(f["shape"])
            batch[k] = torch.zeros((1, *shape), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = policy.select_action(batch)
        row["infer"] = f"✅ {tuple(out.shape)}"
        del policy
        torch.cuda.empty_cache()
    except Exception as e:
        row["note"] = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
        if "--trace" in sys.argv:
            traceback.print_exc()
        torch.cuda.empty_cache()
    results.append(row)
    ks = ",".join(x.replace("observation.", "o.") for x in row.get("keys", []))[:43]
    print(f"{label:<34} {row['load']:<8} {row['infer']:<8} {ks:<44} {row['note']}")

print("\n\n=== 요약 ===")
ok = [r for r in results if r["infer"].startswith("✅")]
bad = [r for r in results if not r["infer"].startswith("✅")]
print(f"  추론까지 성공: {len(ok)}개")
for r in ok:
    print(f"    ✅ {r['label']}")
if bad:
    print(f"  실패: {len(bad)}개")
    for r in bad:
        print(f"    ✗ {r['label']}  — {r['note']}")

Path("/tmp/policy_verify.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str))
print("\n결과 저장: /tmp/policy_verify.json")
