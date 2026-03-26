[이전: 비디오 이해](./30_Video_Understanding.md)

---

# 31. Optical Flow

## 학습 목표

이 수업을 완료하면 다음을 수행할 수 있습니다:

1. 프레임 간 픽셀 수준 동작 추정으로서의 Optical Flow 설명
2. 희소 Optical Flow를 위한 Lucas-Kanade 방법 구현
3. 딥러닝 접근법 설명: FlowNet, PWC-Net, RAFT
4. 표준 색상 코딩을 사용한 Optical Flow 필드 시각화
5. 동작 추정, 비디오 안정화, 행동 인식에 Optical Flow 적용

---

## 목차

1. [Optical Flow 기초](#1-optical-flow-기초)
2. [Lucas-Kanade 방법](#2-lucas-kanade-방법)
3. [Horn-Schunck 밀집 흐름](#3-horn-schunck-밀집-흐름)
4. [딥 Optical Flow (FlowNet)](#4-딥-optical-flow-flownet)
5. [RAFT: 최신 기술](#5-raft-최신-기술)
6. [흐름 시각화 및 평가](#6-흐름-시각화-및-평가)
7. [응용 분야](#7-응용-분야)
8. [연습문제](#8-연습문제)

---

## 1. Optical Flow 기초

### 1.1 Optical Flow란?

```
Optical Flow: 연속 두 프레임 간 각 픽셀의 겉보기 동작을
설명하는 밀집 벡터 필드.

프레임 I₁의 각 픽셀 (x, y)에 대해:
  흐름 (u, v) = 프레임 I₂의 대응 픽셀까지의 변위

  프레임 1의 픽셀 (x, y) → 프레임 2의 픽셀 (x+u, y+v)

밝기 항상성 가정:
  I(x, y, t) = I(x + u, y + v, t + 1)
  "픽셀의 밝기는 프레임 간에 변하지 않는다"

  테일러 전개 → Optical Flow 방정식:
  Ix·u + Iy·v + It = 0

  여기서 Ix, Iy = 공간 기울기, It = 시간 기울기
  하나의 방정식, 두 개의 미지수 (u, v) → 추가 제약 조건 필요!
```

---

## 2. Lucas-Kanade 방법

### 2.1 LK 희소 흐름

```python
import cv2
import numpy as np


def lucas_kanade_demo(video_path):
    """Lucas-Kanade 희소 Optical Flow 추적."""
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # 추적할 좋은 특징 탐지
    feature_params = dict(maxCorners=100, qualityLevel=0.3,
                         minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)

    # LK 매개변수
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))

    colors = np.random.randint(0, 255, (100, 3))
    mask = np.zeros_like(old_frame)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow 계산
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        # 좋은 점 선택
        if p1 is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]

        # 궤적 그리기
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel().astype(int)
            c, d = old.ravel().astype(int)
            mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 2)
            frame = cv2.circle(frame, (a, b), 5, colors[i].tolist(), -1)

        output = cv2.add(frame, mask)
        cv2.imshow('Optical Flow - Lucas-Kanade', output)

        if cv2.waitKey(30) & 0xFF == 27:
            break

        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    cv2.destroyAllWindows()
```

---

## 3. Horn-Schunck 밀집 흐름

### 3.1 Farneback을 사용한 밀집 흐름

```python
def farneback_dense_flow(video_path):
    """Farneback 방법을 사용한 밀집 Optical Flow."""
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 밀집 Optical Flow 계산
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # HSV 색상 코딩을 사용한 시각화
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(old_frame)
        hsv[..., 0] = angle * 180 / np.pi / 2  # 색상 = 방향
        hsv[..., 1] = 255                        # 채도 = 최대
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255,
                                     cv2.NORM_MINMAX)  # 명도 = 크기

        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imshow('Dense Optical Flow', flow_rgb)

        if cv2.waitKey(30) & 0xFF == 27:
            break

        old_gray = gray

    cap.release()
    cv2.destroyAllWindows()
```

---

## 4. 딥 Optical Flow (FlowNet)

### 4.1 FlowNet 아키텍처

```
FlowNet (2015): 최초의 CNN 기반 Optical Flow 추정.

FlowNetS (Simple):
  두 프레임을 연결 → 인코더-디코더 → 흐름 예측
  입력: [I₁, I₂] (6 채널) → 출력: 픽셀당 (u, v)

FlowNetC (Correlation):
  각 프레임에 별도 인코더 → Correlation 레이어 → 디코더
  Correlation이 프레임 간 매칭 캡처

아키텍처:
  I₁ ─┐
      ├─ Concat ─→ 인코더 ─→ 디코더 ─→ Flow (u, v)
  I₂ ─┘

학습: 합성 데이터로 지도 학습 (Flying Chairs, FlyingThings3D)
  손실: EPE (End-Point Error) = ||flow_pred - flow_gt||₂
```

### 4.2 간소화된 FlowNet

```python
import torch
import torch.nn as nn


class FlowNetS(nn.Module):
    """간소화된 FlowNet-Simple 아키텍처."""

    def __init__(self):
        super().__init__()
        # 인코더
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
        )

        # 스킵 연결이 있는 디코더
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),  # 2 채널: (u, v)
        )

    def forward(self, img1, img2):
        x = torch.cat([img1, img2], dim=1)  # (B, 6, H, W)
        features = self.encoder(x)
        flow = self.decoder(features)
        return flow  # (B, 2, H, W)
```

---

## 5. RAFT: 최신 기술

### 5.1 RAFT 아키텍처

```
RAFT (Recurrent All-Pairs Field Transforms, 2020):
  최신 Optical Flow 추정 기술.

핵심 구성요소:
  1. Feature encoder: 두 프레임에서 특징 추출
  2. Correlation volume: 특징 간 모든 쌍의 상관관계
  3. GRU 기반 반복적 정제: 흐름 추정을 반복적으로 업데이트

  I₁ → 인코더 → Features₁ ──┐
                                ├── Correlation Volume
  I₂ → 인코더 → Features₂ ──┘       │
                                       ▼
  초기 흐름 (영) → GRU → 업데이트 → GRU → 업데이트 → ... → 최종 흐름
                          ↑              ↑
                     Correlation     Correlation
                     조회            조회

  각 GRU 반복이 흐름 추정을 정제.
  학습 시 보통 12-32회 반복, 테스트 시 임의의 횟수.
```

### 5.2 RAFT 사용

```python
# torchvision의 사전 학습된 RAFT 사용
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def compute_raft_flow(img1, img2):
    """사전 학습된 RAFT를 사용하여 Optical Flow 계산."""
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights)
    model.eval()

    # 전처리
    transforms = weights.transforms()
    img1_t, img2_t = transforms(img1, img2)

    with torch.no_grad():
        flow_predictions = model(img1_t.unsqueeze(0), img2_t.unsqueeze(0))

    # 마지막 예측이 최종 흐름
    flow = flow_predictions[-1][0]  # (2, H, W)
    return flow
```

---

## 6. 흐름 시각화 및 평가

### 6.1 흐름 색상 코딩

```python
def flow_to_color(flow, max_flow=None):
    """Middlebury 색상 코딩을 사용하여 Optical Flow를 컬러 이미지로 변환."""
    u = flow[..., 0]
    v = flow[..., 1]

    if max_flow is None:
        max_flow = max(np.abs(u).max(), np.abs(v).max())

    magnitude = np.sqrt(u**2 + v**2)
    angle = np.arctan2(-v, -u) / np.pi  # [-1, 1]

    # HSV로 매핑
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + 1) / 2 * 179).astype(np.uint8)  # 색상
    hsv[..., 1] = 255  # 채도
    hsv[..., 2] = np.minimum(magnitude / max_flow * 255, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def end_point_error(pred_flow, gt_flow):
    """End-Point Error: 예측 흐름과 정답 흐름 간의 L2 거리."""
    epe = np.sqrt(((pred_flow - gt_flow) ** 2).sum(axis=-1))
    return epe.mean()
```

---

## 7. 응용 분야

### 7.1 비디오 안정화

```python
def stabilize_video(frames, flows):
    """Optical Flow를 사용한 간단한 비디오 안정화."""
    H, W = frames[0].shape[:2]
    cumulative_dx = 0
    cumulative_dy = 0
    stabilized = [frames[0]]

    for i in range(1, len(frames)):
        # 평균 흐름이 전역 동작을 제공
        dx = flows[i-1][..., 0].mean()
        dy = flows[i-1][..., 1].mean()

        cumulative_dx += dx
        cumulative_dy += dy

        # 역변환 생성
        M = np.float32([[1, 0, -cumulative_dx],
                        [0, 1, -cumulative_dy]])

        stabilized_frame = cv2.warpAffine(frames[i], M, (W, H))
        stabilized.append(stabilized_frame)

    return stabilized
```

---

## 8. 연습문제

### 연습문제 1: Lucas-Kanade 특징 추적

LK 추적 구현 및 평가:
1. LK Optical Flow를 사용하여 비디오에서 50개 특징 추적
2. 전방-후방 일관성 검사 구현
3. 어노테이션된 데이터셋에서 추적 정확도 측정
4. 추적된 점의 궤적 시각화
5. 특징 손실 처리 및 재탐지

### 연습문제 2: 밀집 흐름 시각화

밀집 Optical Flow 시각화기 구축:
1. 비디오에서 Farneback 흐름 계산
2. Middlebury 색상 코딩 구현
3. 흐름 크기 히트맵 생성
4. 원본 이미지에 흐름 화살표 오버레이
5. 비교: Farneback, 보간된 LK, RAFT

### 연습문제 3: FlowNet 학습

FlowNet을 처음부터 학습:
1. 합성 학습 데이터 생성 (Flying Chairs 스타일)
2. FlowNetS 아키텍처 구현
3. EPE 손실과 다중 스케일 감독으로 학습
4. Sintel 또는 KITTI 벤치마크에서 평가
5. 동일 데이터에서 전통적 방법과 비교

### 연습문제 4: 동작 세그멘테이션

동작 세그멘테이션에 Optical Flow 사용:
1. 연속 프레임 간 밀집 흐름 계산
2. 움직이는 객체를 분리하기 위해 흐름 벡터 클러스터링
3. 움직이는 영역 vs 정적 영역의 이진 마스크 생성
4. 비디오에서 움직이는 객체 추적
5. 측정: 세그멘테이션 품질 vs 단순 배경 차분

### 연습문제 5: 비디오 안정화 시스템

완전한 비디오 안정화 파이프라인 구축:
1. Optical Flow 평균을 사용하여 전역 동작 계산
2. 경로 스무딩 구현 (누적 동작에 이동 평균)
3. 안정화를 위한 역변환 적용
4. 처리: 검은 테두리를 피하기 위한 확대
5. OpenCV의 비디오 안정화 모듈과 품질 비교

---

*31강 끝*
