[이전: 3D Gaussian Splatting](./29_3D_Gaussian_Splatting.md)

---

# 30. 비디오 이해

## 학습 목표

이 수업을 완료하면 다음을 수행할 수 있습니다:

1. 비디오에서의 행동 인식과 시간적 모델링 설명
2. 비디오 특징을 위한 Two-Stream 네트워크와 3D 합성곱 구현
3. 시공간 어텐션을 사용한 비디오 Transformer (TimeSformer, ViViT) 구축
4. 다중 속도 시간적 모델링을 위한 SlowFast 아키텍처 설명
5. 행동 탐지와 시간적 위치 추정에 비디오 이해 적용

---

## 목차

1. [비디오 이해 개요](#1-비디오-이해-개요)
2. [시간적 모델링 접근법](#2-시간적-모델링-접근법)
3. [3D 합성곱 (C3D, I3D)](#3-3d-합성곱-c3d-i3d)
4. [SlowFast 네트워크](#4-slowfast-네트워크)
5. [비디오 Transformer](#5-비디오-transformer)
6. [시간적 행동 탐지](#6-시간적-행동-탐지)
7. [실용적 비디오 파이프라인](#7-실용적-비디오-파이프라인)
8. [연습문제](#8-연습문제)

---

## 1. 비디오 이해 개요

### 1.1 비디오 이해의 과제

```
Video Classification:
  입력: 비디오 클립 → 출력: 행동 레이블
  "이 비디오는 수영을 보여줍니다"

Temporal Action Detection:
  입력: 편집되지 않은 비디오 → 출력: (행동, 시작_시간, 종료_시간)
  "수영 5.2초~12.8초, 다이빙 13.0초~15.5초"

Action Recognition:
  입력: 짧은 클립 (3-10초) → 출력: 행동 클래스
  실시간 응용을 위한 온라인 설정

Video Captioning:
  입력: 비디오 → 출력: 자연어 설명
  "한 사람이 수영장에 다이빙하여 건너편으로 수영합니다"

Video Question Answering:
  입력: 비디오 + 질문 → 출력: 답변
  "몇 명이 수영하고 있나요?" → "세 명"
```

---

## 2. 시간적 모델링 접근법

### 2.1 접근법 분류

```
비디오에서 시간 차원을 어떻게 처리할 것인가?

1. 프레임 레벨 (2D CNN + 집계):
   각 프레임을 독립적으로 처리한 후 집계
   단순하지만 동작 정보 손실

2. Two-Stream (RGB + Optical Flow):
   공간 스트림: RGB 프레임에서 외형
   시간 스트림: Optical Flow에서 동작
   두 스트림의 예측을 융합

3. 3D 합성곱:
   2D conv를 3D로 확장: 공간과 시간을 함께 합성곱
   시공간 패턴을 직접 캡처

4. 비디오 Transformer:
   공간과 시간에 걸친 셀프 어텐션
   처음부터 전역 수용 영역

5. 순환 (LSTM/GRU):
   메모리를 가지고 프레임을 순차적으로 처리
   가변 길이 비디오에 적합
```

### 2.2 Two-Stream 아키텍처

```python
import torch
import torch.nn as nn
import torchvision.models as models


class TwoStreamNetwork(nn.Module):
    """행동 인식을 위한 Two-Stream 아키텍처."""

    def __init__(self, n_classes, n_flow_frames=10):
        super().__init__()
        # 공간 스트림: 단일 RGB 프레임
        self.spatial = models.resnet50(pretrained=True)
        self.spatial.fc = nn.Linear(2048, n_classes)

        # 시간 스트림: 쌓인 Optical Flow
        self.temporal = models.resnet50(pretrained=True)
        # Flow 입력을 위해 첫 번째 conv 수정 (2*n_flow_frames 채널)
        self.temporal.conv1 = nn.Conv2d(
            2 * n_flow_frames, 64, kernel_size=7, stride=2, padding=3
        )
        self.temporal.fc = nn.Linear(2048, n_classes)

    def forward(self, rgb_frame, flow_stack):
        """
        Args:
            rgb_frame: (B, 3, H, W) 단일 RGB 프레임
            flow_stack: (B, 2*T, H, W) 쌓인 Optical Flow
        Returns:
            logits: (B, n_classes) 융합된 예측
        """
        spatial_logits = self.spatial(rgb_frame)
        temporal_logits = self.temporal(flow_stack)

        # 후기 융합: 로짓 평균
        return 0.5 * spatial_logits + 0.5 * temporal_logits
```

---

## 3. 3D 합성곱 (C3D, I3D)

### 3.1 3D 합성곱 개념

```
2D Conv: 커널 (k, k)이 (H, W) 위를 이동 → 공간 특징
3D Conv: 커널 (k, k, k)이 (T, H, W) 위를 이동 → 시공간 특징

  비디오에 대한 2D Conv (프레임별):
  프레임 간 동작을 캡처할 수 없음.

  비디오에 대한 3D Conv:
  커널이 여러 프레임에 걸침 → 동작 패턴 캡처!
  예: 3×3×3 커널은 3프레임 × 3×3 공간 윈도우를 봄
```

### 3.2 I3D (Inflated 3D)

```python
class I3DBlock(nn.Module):
    """Inflated 3D 합성곱 블록 (2D 필터를 3D로 확장)."""

    def __init__(self, in_channels, out_channels, temporal_kernel=3):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=(temporal_kernel, 3, 3),
            padding=(temporal_kernel // 2, 1, 1),
            bias=False
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, C, T, H, W)
        return self.relu(self.bn(self.conv(x)))


class SimpleI3D(nn.Module):
    """행동 인식을 위한 간소화된 I3D."""

    def __init__(self, n_classes=400, in_channels=3):
        super().__init__()
        self.features = nn.Sequential(
            I3DBlock(in_channels, 64, temporal_kernel=7),
            nn.MaxPool3d((1, 2, 2)),
            I3DBlock(64, 128),
            nn.MaxPool3d((2, 2, 2)),
            I3DBlock(128, 256),
            I3DBlock(256, 256),
            nn.MaxPool3d((2, 2, 2)),
            I3DBlock(256, 512),
            I3DBlock(512, 512),
            nn.MaxPool3d((2, 2, 2)),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(512, n_classes)

    def forward(self, x):
        # x: (B, 3, T, H, W) - 비디오 클립
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
```

---

## 4. SlowFast 네트워크

### 4.1 SlowFast 개념

```
SlowFast: 서로 다른 시간적 속도로 동작하는 두 경로.

Slow 경로:
  낮은 프레임 속도 (예: 4 FPS)
  높은 채널 용량
  공간 의미론과 외형 캡처

Fast 경로:
  높은 프레임 속도 (예: 32 FPS)
  낮은 채널 용량 (8배 적은 채널)
  세밀한 시간 패턴과 동작 캡처

  Slow: ████████████  (4 프레임, 64 채널)
  Fast: ████████████████████████████████  (32 프레임, 8 채널)

측면 연결: Fast → Slow (시간 정보를 공간에 융합)
```

### 4.2 SlowFast 구현

```python
class SlowFastNetwork(nn.Module):
    """간소화된 SlowFast 네트워크."""

    def __init__(self, n_classes=400, alpha=8, beta=8):
        super().__init__()
        self.alpha = alpha  # 프레임 속도 비율 (fast/slow)
        self.beta = beta    # 채널 비율 (slow/fast)

        # Slow 경로
        self.slow_conv1 = nn.Conv3d(3, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3))
        self.slow_pool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res = self._make_layer(64 + 64 // beta, 256, n_blocks=3)

        # Fast 경로
        self.fast_conv1 = nn.Conv3d(3, 64 // beta, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3))
        self.fast_pool = nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res = self._make_layer(64 // beta, 256 // beta, n_blocks=3)

        # 측면 연결 (fast → slow)
        self.lateral = nn.Conv3d(64 // beta, 64 // beta, (5, 1, 1),
                                stride=(alpha, 1, 1), padding=(2, 0, 0))

        # 분류기
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256 + 256 // beta, n_classes)

    def _make_layer(self, in_ch, out_ch, n_blocks):
        layers = [nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.ReLU()]
        for _ in range(n_blocks - 1):
            layers.extend([nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.ReLU()])
        return nn.Sequential(*layers)

    def forward(self, video):
        """
        Args:
            video: (B, 3, T, H, W) - 전체 프레임 속도 비디오
        Returns:
            logits: (B, n_classes)
        """
        # Slow 경로를 위한 서브샘플링
        slow_input = video[:, :, ::self.alpha]  # alpha번째 프레임마다

        # Slow 경로
        slow = self.slow_conv1(slow_input)
        slow = self.slow_pool(slow)

        # Fast 경로
        fast = self.fast_conv1(video)
        fast = self.fast_pool(fast)

        # 측면 연결: fast → slow
        lateral = self.lateral(fast)
        slow = torch.cat([slow, lateral], dim=1)

        # 계속 처리
        slow = self.slow_res(slow)
        fast = self.fast_res(fast)

        # 병합 및 분류
        slow = self.pool(slow).flatten(1)
        fast = self.pool(fast).flatten(1)
        x = torch.cat([slow, fast], dim=1)

        return self.fc(x)
```

---

## 5. 비디오 Transformer

### 5.1 TimeSformer

```python
class TimeSformerBlock(nn.Module):
    """TimeSformer: 분할된 시공간 어텐션."""

    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        # 시간 어텐션 (동일 공간 위치에서 시간에 걸쳐 어텐션)
        self.temporal_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.temporal_norm = nn.LayerNorm(d_model)

        # 공간 어텐션 (동일 시간 단계에서 공간에 걸쳐 어텐션)
        self.spatial_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.spatial_norm = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x, T, S):
        """
        Args:
            x: (B, T*S, D) 평탄화된 비디오 토큰
            T: 시간 단계 수
            S: 프레임당 공간 토큰 수
        """
        B, N, D = x.shape

        # 1. 시간 어텐션
        x_t = x.reshape(B * S, T, D)  # 공간 위치별 그룹화
        attn_out, _ = self.temporal_attn(x_t, x_t, x_t)
        x = x + attn_out.reshape(B, N, D)
        x = self.temporal_norm(x)

        # 2. 공간 어텐션
        x_s = x.reshape(B * T, S, D)  # 시간 단계별 그룹화
        attn_out, _ = self.spatial_attn(x_s, x_s, x_s)
        x = x + attn_out.reshape(B, N, D)
        x = self.spatial_norm(x)

        # 3. FFN
        x = x + self.ffn(x)
        x = self.ffn_norm(x)

        return x
```

---

## 6. 시간적 행동 탐지

### 6.1 행동 탐지 파이프라인

```python
def temporal_action_detection(video_features, model, threshold=0.5):
    """
    편집되지 않은 비디오에서 행동 탐지.

    출력: (행동_클래스, 시작_시간, 종료_시간, 신뢰도) 리스트
    """
    # 1. 시간적 후보 생성 (후보 구간)
    proposals = generate_proposals(video_features)

    # 2. 각 후보 분류
    detections = []
    for start, end in proposals:
        segment_features = video_features[start:end]
        pooled = segment_features.mean(dim=0)

        class_scores = model.classify(pooled)
        action_class = class_scores.argmax().item()
        confidence = class_scores.max().item()

        if confidence > threshold:
            detections.append({
                'class': action_class,
                'start': start / fps,
                'end': end / fps,
                'confidence': confidence,
            })

    # 3. 비최대 억제
    detections = temporal_nms(detections, iou_threshold=0.5)

    return detections
```

---

## 7. 실용적 비디오 파이프라인

### 7.1 비디오 데이터셋 로딩

```python
import cv2
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """클립 샘플링을 사용하는 비디오 데이터셋."""

    def __init__(self, video_paths, labels, clip_length=16,
                 frame_rate=4, transform=None):
        self.videos = video_paths
        self.labels = labels
        self.clip_length = clip_length
        self.frame_rate = frame_rate
        self.transform = transform

    def __getitem__(self, idx):
        # 비디오 로드
        cap = cv2.VideoCapture(self.videos[idx])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # 대상 속도로 프레임 샘플링
        sample_interval = max(1, int(video_fps / self.frame_rate))
        frame_indices = list(range(0, total_frames, sample_interval))

        # 랜덤 시간적 크롭
        if len(frame_indices) > self.clip_length:
            start = np.random.randint(0, len(frame_indices) - self.clip_length)
            frame_indices = frame_indices[start:start + self.clip_length]

        frames = []
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        # 필요 시 패딩
        while len(frames) < self.clip_length:
            frames.append(frames[-1])

        clip = np.stack(frames[:self.clip_length])  # (T, H, W, 3)

        if self.transform:
            clip = self.transform(clip)

        # (T, H, W, 3) → (3, T, H, W)
        clip = torch.FloatTensor(clip).permute(3, 0, 1, 2) / 255.0

        return clip, self.labels[idx]

    def __len__(self):
        return len(self.videos)
```

---

## 8. 연습문제

### 연습문제 1: 3D CNN 행동 인식

행동 인식 시스템 구축:
1. 비디오 분류를 위한 3D CNN (C3D 스타일) 구현
2. Kinetics-400 서브셋 또는 UCF-101에서 학습
3. 비교: 프레임별 2D CNN vs 3D CNN
4. 시각화: 3D conv가 학습하는 시간 패턴은?
5. 보고: top-1 및 top-5 정확도

### 연습문제 2: Two-Stream 네트워크

Two-Stream 아키텍처 구현:
1. OpenCV를 사용하여 Optical Flow 계산 (Farneback 또는 RAFT)
2. 공간 스트림 (RGB 프레임)과 시간 스트림 (Flow 스택) 구축
3. 초기 및 후기 융합 전략 구현
4. 비교: 공간만, 시간만, Two-Stream
5. 분석: 어떤 행동이 외형 vs 동작에 더 의존하는가?

### 연습문제 3: SlowFast 처음부터

간소화된 SlowFast 네트워크 구축:
1. Slow 및 Fast 경로 구현
2. 측면 연결 추가 (fast → slow)
3. 비디오 데이터셋에서 학습하고 단일 경로와 비교
4. Alpha (프레임 속도 비율) 변경: {4, 8, 16}하고 영향 측정
5. 시각화: 각 경로가 무엇에 집중하는가?

### 연습문제 4: 비디오 Transformer

분할된 시공간 어텐션 구현:
1. 비디오 프레임 패치 임베딩 (16×16 패치)
2. 시간 어텐션과 공간 어텐션을 별도로 구현
3. 비디오 분류 작업에서 학습
4. 동일 데이터셋에서 3D CNN과 비교
5. 어텐션 패턴 시각화: 시간적 및 공간적

### 연습문제 5: 실시간 행동 인식

실시간 행동 인식 시스템 구축:
1. 웹캠을 입력으로 사용
2. 최근 N 프레임에 대한 슬라이딩 윈도우 구현
3. 경량 모델 (MobileNet 기반) 실행하여 분류
4. 표시: 현재 행동, 신뢰도, FPS
5. 처리: 행동 간 전환, "행동 없음" 클래스

---

*30강 끝*
