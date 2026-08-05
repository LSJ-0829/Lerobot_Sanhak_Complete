"""수건 grasp 성공/실패를 CLIP 선형 probe 로 판별하는 공용 모듈.

구조(washer_probe.pt / grasp_probe.pt 와 완전 호환):
  - open_clip(ViT-B-32) 로 이미지 feature(512d, L2 정규화)를 뽑고
  - 그 위에 로지스틱 회귀 probe(W[C,512], b[C]) 를 얹어 분류한다.
  - probe 체크포인트 dict:
      {W, b, classes, model_name, pretrained, cv_acc, cv_conf, counts, scene_counts}

세 스크립트가 공유:
  collect_grasp_data.py  → 카메라 프레임 수집
  train_grasp_probe.py   → encode_images 로 feature 뽑아 probe 학습
  deploy_policy.py       → GraspGate 로 실시간 판별(잡으면 VLA 정지)
"""

import os
import time

import numpy as np
import torch
from PIL import Image

import open_clip

DEFAULT_MODEL = "ViT-B-32"
# 이 머신엔 로컬 safetensors 가 없으므로 표준 laion 가중치를 받아 쓴다.
# (probe 학습 시 이 값이 체크포인트에 저장되고, 배포 때 동일 가중치로 로드된다.)
DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


def open_v4l2_camera(cam_device, width=640, height=480, warmup=15):
    """raw cv2 V4L2 + MJPG 로 카메라를 연다(세탁기 위 별도 grasp 카메라용).

    lerobot OpenCVCamera 가 이 카메라들에서 해상도 검증에 실패하는 문제를 회피하려고
    washer_clip_check.py 와 동일하게 raw cv2 를 쓴다. BUFFERSIZE=1 로 최신 프레임을 받는다.
    """
    import cv2

    cam = int(cam_device) if str(cam_device).isdigit() else cam_device
    cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: {cam_device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(warmup):  # 자동노출/화이트밸런스 안정화
        cap.read()
        time.sleep(0.03)
    return cap


def load_clip(model_name=DEFAULT_MODEL, pretrained=DEFAULT_PRETRAINED, device=None):
    """CLIP 모델 + 전처리(transform) 로드. (model, preprocess, device) 반환."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    return model, preprocess, device


def to_pil(image) -> Image.Image:
    """np.ndarray(RGB) 또는 PIL.Image → PIL.Image(RGB)."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image)).convert("RGB")


@torch.no_grad()
def encode_images(model, preprocess, images, device, batch_size=64) -> torch.Tensor:
    """이미지(경로/np/PIL) 리스트 → L2 정규화된 CLIP feature 텐서 [N, 512] (cpu)."""
    feats, batch = [], []

    def flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        feats.append(f.float().cpu())
        batch.clear()

    for img in images:
        if isinstance(img, str):
            img = Image.open(img)
        batch.append(preprocess(to_pil(img)))
        if len(batch) >= batch_size:
            flush()
    flush()
    return torch.cat(feats, dim=0) if feats else torch.empty(0, 512)


class GraspGate:
    """학습된 grasp probe 로 실시간 판별하는 게이트.

    deploy 루프에서 매 프레임 update(frame) 를 호출한다.
    'grabbed' 클래스가 threshold 이상 확률로 hold_frames 번 '연속' 나오면
    grabbed() 가 True 를 돌려준다(노이즈로 인한 조기 정지 방지).
    """

    def __init__(
        self,
        probe_path,
        target_class="grabbed",
        threshold=0.85,
        hold_frames=5,
        device=None,
    ):
        if not (probe_path and os.path.exists(probe_path)):
            raise FileNotFoundError(f"probe 를 찾을 수 없습니다: {probe_path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(probe_path, map_location=self.device, weights_only=False)
        self.W = ckpt["W"].to(self.device).float()
        self.b = ckpt["b"].to(self.device).float()
        self.classes = list(ckpt["classes"])
        if target_class not in self.classes:
            raise ValueError(f"target_class '{target_class}' 가 probe classes {self.classes} 에 없습니다.")
        self.target_idx = self.classes.index(target_class)
        self.target_class = target_class
        self.threshold = threshold
        self.hold_frames = hold_frames
        self._streak = 0

        model_name = ckpt.get("model_name") or DEFAULT_MODEL
        pretrained = ckpt.get("pretrained") or DEFAULT_PRETRAINED
        # 학습 때 로컬 safetensors 경로가 저장돼 있으나 이 머신엔 없을 수 있음 → 표준 가중치로 폴백
        if isinstance(pretrained, str) and pretrained.endswith(".safetensors") and not os.path.exists(pretrained):
            pretrained = DEFAULT_PRETRAINED
        self.model, self.preprocess, self.device = load_clip(model_name, pretrained, self.device)

    @torch.no_grad()
    def predict(self, frame):
        """단일 프레임 → (top_class, target_prob, scores_dict). streak 갱신 안 함."""
        x = self.preprocess(to_pil(frame)).unsqueeze(0).to(self.device)
        f = self.model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        proba = (f.float() @ self.W.t() + self.b).softmax(dim=-1)[0]
        scores = {c: float(proba[i]) for i, c in enumerate(self.classes)}
        top = self.classes[int(proba.argmax())]
        return top, float(proba[self.target_idx]), scores

    def update(self, frame):
        """프레임 하나 처리하고 연속 카운트 갱신. (grabbed_bool, target_prob, scores)."""
        top, p, scores = self.predict(frame)
        if top == self.target_class and p >= self.threshold:
            self._streak += 1
        else:
            self._streak = 0
        return self.grabbed(), p, scores

    def grabbed(self) -> bool:
        return self._streak >= self.hold_frames

    def reset(self):
        self._streak = 0
