"""분류기 4종을 실제 top 카메라 프레임으로 돌려 비교한다.

'시작을 안 한다'의 원인이 분류기였으므로, 로드만이 아니라 **현재 장면에 어떤 클래스를
얼마의 확신으로 주는지** 봐야 한다. IDLE 에 머물지, step1 로 넘어갈지가 여기서 갈린다.
카메라만 읽는다 — 로봇은 건드리지 않는다.
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

sys.path.insert(0, str(Path(__file__).resolve().parent))
from csi_paths import train as _train  # noqa: E402  경로는 csi_paths 한 곳에서 정한다

TRAIN = _train()
# 분류기 입력은 폴딩 top 캠. resolve_cameras 가 만든 링크를 먼저 쓴다 —
# /dev/lerobot/camera_0 은 물리 포트로 붙어서 자리가 바뀌면 엉뚱한 카메라를 가리킨다.
_LINK = Path.home() / ".lerobot" / "cams" / "top"
CAM = str(_LINK) if _LINK.exists() else "/dev/lerobot/camera_0"
N_FRAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 10

cap = cv2.VideoCapture(CAM)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frames = []
for _ in range(N_FRAMES + 5):
    ok, f = cap.read()
    if ok:
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    time.sleep(0.05)
cap.release()
frames = frames[5:]                     # 워밍업 프레임 버림
if not frames:
    print("✗ 카메라에서 프레임을 못 읽음")
    sys.exit(1)
print(f"top 캠에서 {len(frames)}프레임 획득 {frames[0].shape}\n")

device = "cuda" if torch.cuda.is_available() else "cpu"
for cdir in sorted(TRAIN.glob("towel_fold01_nextlevel*")):
    try:
        model = AutoModelForImageClassification.from_pretrained(cdir).to(device).eval()
        proc = AutoImageProcessor.from_pretrained(cdir)
    except Exception as e:
        print(f"{cdir.name:<30} ✗ 로드 실패: {str(e).splitlines()[0][:80]}")
        continue

    counts, confs = {}, []
    for fr in frames:
        with torch.no_grad():
            inp = proc(images=fr, return_tensors="pt").to(device)
            logits = model(**inp).logits
            p = torch.softmax(logits, -1)[0]
            i = int(p.argmax())
        lab = model.config.id2label[i]
        counts[lab] = counts.get(lab, 0) + 1
        confs.append(float(p[i]))
    top = max(counts, key=counts.get)
    dist = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items(), key=lambda x: -x[1]))
    mark = " ← 현재 사용중" if cdir.name == "towel_fold01_nextlevel" else ""
    print(f"{cdir.name:<30} 판정 '{top}' ({counts[top]}/{len(frames)})  "
          f"평균확신 {np.mean(confs):.2f}   [{dist}]{mark}")
    del model
    torch.cuda.empty_cache()

print("\n해석: 수건이 아직 펴져 있지 않다면 IDLE 이 나와야 정상이다.")
print("      펴서 놓았는데도 IDLE 이면 대기 상태로 남아 폴딩이 시작되지 않는다.")
