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

### 이론: 밝기 불변성 제약

Optical flow의 기본 가정: **물리적 한 점의 밝기는 연속 프레임 간에 변하지 않는다**. 시간 `t`의 장면 점이 픽셀 `(x, y)`로 밝기 `I(x, y, t)`로 투영되고, 시간 `t + dt`에는 `(x + dx, y + dy)`로 이동했다면:

```
I(x + dx, y + dy, t + dt) = I(x, y, t)
```

좌변을 `(x, y, t)` 주변으로 1차 Taylor 전개:

```
I(x, y, t) + (∂I/∂x)·dx + (∂I/∂y)·dy + (∂I/∂t)·dt  =  I(x, y, t)
```

`I(x, y, t)`를 지우고 `dt`로 나누면(속도 `u = dx/dt`, `v = dy/dt`):

```
(∂I/∂x)·u + (∂I/∂y)·v + (∂I/∂t) = 0       ← 밝기 불변성 제약식
```

보통 `I_x · u + I_y · v + I_t = 0` 또는 `∇I · (u, v) + I_t = 0`로 씁니다. 이것은 **두 미지수 `u`, `v`에 대한 단 하나의 선형 방정식**입니다. 픽셀당 방정식 1개, 미지수 2개이므로 추가 정보 없이는 미결정 시스템.

밝기 불변성 가정은 실전에서 그림자, 반사광, 주변광 변화에 의해 깨집니다. 더 현대적 공식은 기울기 불변성 또는 학습된 특징 불변성을 써서 완화하지만, 기본 틀은 동일합니다.

### 이론: Aperture 문제

기하학적으로 단일 제약식은 다음과 같이 말합니다: **이미지 기울기 방향의 움직임 성분**은 결정되지만, 기울기에 수직한 성분은 결정되지 않습니다. 방정식으로부터:

```
∇I 방향의 (u, v) 성분          =  -I_t / |∇I|       (결정됨)
∇I에 수직한 (u, v) 성분                              (미결정)
```

이것이 **aperture 문제**: 작은 구멍으로 움직이는 에지를 볼 때, 에지가 *자기 자신에 수직하게* 어떻게 움직이는지만 보이고, 평행한 움직임은 보이지 않습니다. 수평으로 미끄러지는 수평 에지는 움직이지 않는 에지와 똑같아 보입니다.

유용한 모든 optical flow 알고리즘은 수직 성분을 결정하기 위한 추가 정보를 더해야 합니다. 두 고전적 해결은 무엇을 더할지에 대한 두 가지 서로 다른 선택입니다.

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

### 이론: Lucas-Kanade: 국소 일정성

**가정**: 작은 창 내 픽셀들이 모두 같은 방식으로 움직인다. 이 가정 아래에서는 창 내 모든 픽셀이 *같은* `(u, v)`를 가진 밝기 불변성 방정식 한 개씩을 기여합니다:

```
창 W 내 픽셀 i에 대해:
    I_x(i) · u + I_y(i) · v = -I_t(i)
```

`N`-픽셀 창의 `N`개 방정식을 쌓으면:

```
⎡ I_x(1)  I_y(1) ⎤ ⎡ u ⎤     ⎡ -I_t(1) ⎤
⎢ I_x(2)  I_y(2) ⎥ ⎢   ⎥  =  ⎢ -I_t(2) ⎥
⎢   ...    ...   ⎥ ⎣ v ⎦     ⎢   ...   ⎥
⎣ I_x(N)  I_y(N) ⎦           ⎣ -I_t(N) ⎦

       A            (u,v)         b
```

이것은 과결정 선형 시스템. 최소제곱 해는 `(u, v) = (AᵀA)⁻¹ · Aᵀ · b`. 전개:

```
⎡ u ⎤     ⎡ Σ I_x²    Σ I_x·I_y ⎤⁻¹  ⎡ -Σ I_x·I_t ⎤
⎣ v ⎦  =  ⎣ Σ I_x·I_y  Σ I_y²   ⎦    ⎣ -Σ I_y·I_t ⎦
```

좌측 행렬은 **정확히 §13(Feature Detection)의 구조 텐서(Structure Tensor)**. Lucas-Kanade의 국소 시스템은 구조 텐서가 두 개의 큰 고유값을 가질 때 정확히 잘 조건화(invertible)됩니다 — 즉 **코너**에서. 에지에서는 한 고유값이 작고 에지 방향으로 해가 불안정해집니다(aperture 문제가 국소적으로 지속). 평탄 영역에서는 두 고유값 모두 작아 흐름을 복원할 수 없습니다.

이것이 Lucas-Kanade가 보통 희소 **키포인트**(Harris/Shi-Tomasi가 검출한 코너)에만 적용되는 이유입니다: 거기서 수치적으로 잘 조건화되고, 다른 곳에서는 어차피 실패하기 때문.

### 이론: Coarse-to-Fine 피라미드: 큰 움직임 다루기

밝기 불변성의 Taylor 전개는 `(dx, dy)`가 작다고 — 1 픽셀 정도 — 가정합니다. 그보다 큰 움직임에서는 1차 근사가 무효가 되고 Lucas-Kanade와 Horn-Schunck 모두 실패합니다.

**해결**: 두 프레임의 가우시안 피라미드를 만들고, 움직임이 작은 가장 거친(가장 축소된) 레벨에서 optical flow를 풀고, 그 결과를 다음 더 정밀한 레벨로 **전파**(흐름을 2배로 스케일링, 이 추정으로 두 번째 프레임을 워핑해 잔여 움직임이 다시 작아지도록), 정제. 풀 해상도까지 반복.

이것이 OpenCV의 Lucas-Kanade 변형이 `calcOpticalFlowPyrLK`인 이유 — `Pyr`가 피라미드 트릭 — 이며, 가장 작은 움직임을 제외한 모든 경우에 필수입니다.

### 2.1 LK 희소 흐름

```python
import cv2
import numpy as np


def lucas_kanade_demo(video_path):
    """Lucas-Kanade 희소 Optical Flow 추적."""
    cap = cv2.VideoCapture(video_path)
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Detect good features to track
    feature_params = dict(maxCorners=100, qualityLevel=0.3,
                         minDistance=7, blockSize=7)
    p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)

    # LK parameters
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

        # Calculate optical flow
        p1, status, error = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        # Select good points
        if p1 is not None:
            good_new = p1[status == 1]
            good_old = p0[status == 1]

        # Draw tracks
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

### 이론: Horn-Schunck: 전역 매끄러움

**가정**: 흐름장이 전역적으로 매끄럽다. 각 창에서 일정한 흐름을 가정하는 대신, Horn-Schunck는 목적함수에 **매끄러움 정칙화** 항을 추가합니다:

```
E(u, v) = ∫∫ [ (I_x u + I_y v + I_t)² + α²·( |∇u|² + |∇v|² ) ] dx dy
          └───── 데이터 항 ─────┘    └── 매끄러움 ──┘
```

첫 항이 밝기 불변성 위반에 페널티(§1의 밝기 불변성 제약식에서)를 주고, 둘째 항이 매끄럽지 않은 흐름장에 페널티를 줍니다. 가중치 `α`는 하이퍼파라미터: 큰 `α`는 매우 매끄러운 흐름을 강제(완만한 움직임에 좋음), 작은 `α`는 급격한 변화를 허용(움직임 경계에 더 잘 맞지만 잡음이 많음).

`E` 최소화는 결합된 PDE 시스템을 줍니다. 표준 해법은 이산화된 이미지 위의 Gauss-Seidel 반복 — 각 반복이 모든 픽셀에서 이웃에 기반해 `(u, v)`를 갱신. 결과는 **밀집 흐름장** — 모든 픽셀이 흐름 추정을 얻음, Lucas-Kanade의 희소 코너와 달리.

Horn-Schunck는 밀집 출력을 만들지만 움직임 불연속을 가로질러 과하게 매끄럽게 합니다(서로 다르게 움직이는 인접 두 픽셀이 중간으로 평균됨). 현대 변분법은 강건한 비이차 페널티와 total-variation 정칙화를 써서 불연속을 더 잘 다룹니다.

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

        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Visualize using HSV color coding
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        hsv = np.zeros_like(old_frame)
        hsv[..., 0] = angle * 180 / np.pi / 2  # Hue = direction
        hsv[..., 1] = 255                        # Saturation = max
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255,
                                     cv2.NORM_MINMAX)  # Value = magnitude

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

### 이론: 현대 학습 기반 방법이 바꾼 것

딥러닝 optical flow(FlowNet, PWC-Net, RAFT)는 수작업 데이터 및 매끄러움 항을 학습된 표현으로 대체하지만, 문제의 기본 구조는 동일합니다:

- **데이터 항**: 원시 밝기 차이 대신 워핑된 패치들 간 학습된 특징 유사도(밝기 변화, 그림자 등에 강건).
- **매끄러움 사전분포**: 컨볼루션 아키텍처에 내재(공간적 매끄러움은 CNN의 강한 귀납 편향) + 정제 모듈에.
- **피라미드**: PWC-Net에서 명시적(피라미드 + 워핑 + cost volume), RAFT에서 암묵적(GRU 기반 반복 정제).
- **Aperture 문제**: 고전 방법과 같은 방식으로 해결 — 이웃 위에 정보를 집계, 이제 고정 대신 학습으로.

RAFT와 고전 변분법 사이의 벤치마크 격차는 크지만, 훈련 데이터 없는 도메인(과학 이미지, 새로운 센서)에서는 현대 정칙화기를 가진 고전 Lucas-Kanade나 Horn-Schunck도 여전히 경쟁력 있습니다.

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
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(6, 64, 7, stride=2, padding=3), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, 5, stride=2, padding=2), nn.LeakyReLU(0.1),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), nn.LeakyReLU(0.1),
        )

        # Decoder with skip connections
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(32, 2, 4, stride=2, padding=1),  # 2 channels: (u, v)
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
# Using pretrained RAFT from torchvision
import torch
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

def compute_raft_flow(img1, img2):
    """사전 학습된 RAFT를 사용하여 Optical Flow 계산."""
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights)
    model.eval()

    # Preprocess
    transforms = weights.transforms()
    img1_t, img2_t = transforms(img1, img2)

    with torch.no_grad():
        flow_predictions = model(img1_t.unsqueeze(0), img2_t.unsqueeze(0))

    # Last prediction is the final flow
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

    # Map to HSV
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + 1) / 2 * 179).astype(np.uint8)  # Hue
    hsv[..., 1] = 255  # Saturation
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
        # Average flow gives global motion
        dx = flows[i-1][..., 0].mean()
        dy = flows[i-1][..., 1].mean()

        cumulative_dx += dx
        cumulative_dy += dy

        # Create inverse transformation
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
