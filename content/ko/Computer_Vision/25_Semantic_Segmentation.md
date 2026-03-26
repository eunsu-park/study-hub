[이전: SLAM 입문](./23_SLAM_Introduction.md)

---

# 25. 시맨틱 세그멘테이션

## 학습 목표

이 수업을 완료하면 다음을 수행할 수 있습니다:

1. 시맨틱 세그멘테이션과 분류 및 탐지와의 차이점 설명
2. 픽셀 단위 예측을 위한 Fully Convolutional Networks (FCN) 구현
3. 정밀한 세그멘테이션을 위해 스킵 연결이 있는 U-Net 아키텍처 구축
4. Atrous 합성곱과 ASPP 모듈을 사용하는 DeepLab v3+ 설명
5. IoU, mIoU, 픽셀 정확도를 사용한 세그멘테이션 모델 평가

---

## 목차

1. [세그멘테이션 개요](#1-세그멘테이션-개요)
2. [Fully Convolutional Networks (FCN)](#2-fully-convolutional-networks-fcn)
3. [U-Net 아키텍처](#3-u-net-아키텍처)
4. [DeepLab v3+](#4-deeplab-v3)
5. [세그멘테이션 손실 함수](#5-세그멘테이션-손실-함수)
6. [평가 지표](#6-평가-지표)
7. [실습 구현](#7-실습-구현)
8. [연습문제](#8-연습문제)

---

## 1. 세그멘테이션 개요

### 1.1 세그멘테이션의 유형

```
Image Classification:
  입력: 이미지 → 출력: 단일 레이블
  "이것은 고양이입니다"

Object Detection:
  입력: 이미지 → 출력: 바운딩 박스 + 레이블
  "고양이가 (x1,y1,x2,y2)에 있음"

Semantic Segmentation:
  입력: 이미지 → 출력: 모든 픽셀에 대한 클래스 레이블
  "픽셀 (i,j)는 고양이, 픽셀 (i,j+1)은 배경"

Instance Segmentation:
  입력: 이미지 → 출력: 모든 픽셀에 대한 클래스 + 인스턴스 ID
  "픽셀 (i,j)는 고양이 #1, 픽셀 (i,j+5)는 고양이 #2"

Panoptic Segmentation:
  모든 클래스(stuff + things)에 대한 Semantic + Instance
```

### 1.2 응용 분야

```
시맨틱 세그멘테이션 응용 분야:

자율 주행:
  도로, 인도, 차량, 보행자, 하늘, 건물
  입력: 1920×1080 카메라 → 출력: 픽셀별 레이블

의료 영상:
  CT/MRI에서 종양, 장기, 조직 세그멘테이션
  진단 및 수술 계획에 필수적

위성/항공 영상:
  토지 이용 분류: 산림, 수역, 도시, 농업

로봇공학:
  내비게이션과 물체 조작을 위한 장면 이해

증강 현실:
  실시간 인물/배경 세그멘테이션 (화상 통화)
```

---

## 2. Fully Convolutional Networks (FCN)

### 2.1 분류에서 세그멘테이션으로

```
핵심 아이디어: 완전 연결 레이어를 합성곱 레이어로 대체.

Classification CNN:
  이미지 → Conv 레이어 → FC 레이어 → [cat, dog, ...]
                         ↑ 공간 정보 손실!

FCN:
  이미지 → Conv 레이어 → 1×1 Conv → 업샘플링 → 픽셀별 레이블
                                    ↑ 공간 정보 보존!
```

### 2.2 FCN 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN8s(nn.Module):
    """FCN-8s: 8배 업샘플링을 사용하는 Fully Convolutional Network."""

    def __init__(self, n_classes=21):
        super().__init__()
        # 인코더 (VGG-16 백본)
        self.conv1 = self._make_block(3, 64, 2)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = self._make_block(64, 128, 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = self._make_block(128, 256, 3)
        self.pool3 = nn.MaxPool2d(2, 2)    # 1/8 해상도

        self.conv4 = self._make_block(256, 512, 3)
        self.pool4 = nn.MaxPool2d(2, 2)    # 1/16 해상도

        self.conv5 = self._make_block(512, 512, 3)
        self.pool5 = nn.MaxPool2d(2, 2)    # 1/32 해상도

        # FCN 헤드 (FC를 1x1 conv로 대체)
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.score = nn.Conv2d(4096, n_classes, 1)

        # 스킵 연결
        self.score_pool4 = nn.Conv2d(512, n_classes, 1)
        self.score_pool3 = nn.Conv2d(256, n_classes, 1)

        # 업샘플링 레이어
        self.upscore2 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, padding=1)
        self.upscore4 = nn.ConvTranspose2d(n_classes, n_classes, 4, stride=2, padding=1)
        self.upscore8 = nn.ConvTranspose2d(n_classes, n_classes, 16, stride=8, padding=4)

    def _make_block(self, in_ch, out_ch, n_convs):
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 인코더
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        pool3_out = x                          # 1/8

        x = self.pool4(self.conv4(x))
        pool4_out = x                          # 1/16

        x = self.pool5(self.conv5(x))          # 1/32

        # FCN 헤드
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = self.score(x)                      # 1/32, n_classes 채널

        # FCN-8s: pool3, pool4, fc7 융합
        x = self.upscore2(x)                   # 1/16
        x = x + self.score_pool4(pool4_out)    # 스킵 연결

        x = self.upscore4(x)                   # 1/8
        x = x + self.score_pool3(pool3_out)    # 스킵 연결

        x = self.upscore8(x)                   # 1/1 (원본 해상도)

        return x
```

---

## 3. U-Net 아키텍처

### 3.1 U-Net 설계

```
U-Net: 모든 레벨에서 스킵 연결이 있는 인코더-디코더.

  인코더 (수축 경로)            디코더 (확장 경로)
  ┌────────────────────┐       ┌────────────────────┐
  │  64 ch, 256×256    │━━━━━━▶│  64 ch, 256×256    │ → 출력
  │  ↓ MaxPool         │       │  ↑ UpConv           │
  │ 128 ch, 128×128    │━━━━━━▶│ 128 ch, 128×128    │
  │  ↓ MaxPool         │       │  ↑ UpConv           │
  │ 256 ch, 64×64      │━━━━━━▶│ 256 ch, 64×64      │
  │  ↓ MaxPool         │       │  ↑ UpConv           │
  │ 512 ch, 32×32      │━━━━━━▶│ 512 ch, 32×32      │
  │  ↓ MaxPool         │       │  ↑ UpConv           │
  │ 1024 ch, 16×16     │───────┘                     │
  └────────────────────┘  병목 레이어                  │
                          ━━━━━▶ = 스킵 연결 (연결, concatenate)
```

### 3.2 U-Net 구현

```python
class DoubleConv(nn.Module):
    """연속적인 두 개의 conv-bn-relu 블록."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """시맨틱 세그멘테이션을 위한 U-Net."""

    def __init__(self, in_channels=3, n_classes=21, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        # 인코더 (다운샘플링 경로)
        for feat in features:
            self.encoder.append(DoubleConv(in_channels, feat))
            in_channels = feat

        # 병목 레이어
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 디코더 (업샘플링 경로)
        for feat in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feat * 2, feat, 2, stride=2)
            )
            self.decoder.append(DoubleConv(feat * 2, feat))

        # 출력
        self.final = nn.Conv2d(features[0], n_classes, 1)

    def forward(self, x):
        skip_connections = []

        # 인코더
        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # 디코더
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)      # 업샘플링
            skip = skip_connections[i // 2]

            # 크기 불일치 처리
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)  # 연결
            x = self.decoder[i + 1](x)       # Double conv

        return self.final(x)
```

---

## 4. DeepLab v3+

### 4.1 Atrous (팽창) 합성곱

```
표준 3×3 conv: 수용 영역 = 3×3
Atrous conv (rate=2): 수용 영역 = 5×5 (간격 포함)
Atrous conv (rate=4): 수용 영역 = 9×9 (간격 포함)

  표준 (rate=1):         Atrous (rate=2):
  ■ ■ ■                  ■ ○ ■ ○ ■
  ■ ■ ■                  ○ ○ ○ ○ ○
  ■ ■ ■                  ■ ○ ■ ○ ■
                          ○ ○ ○ ○ ○
  3×3 RF                  ■ ○ ■ ○ ■
                          5×5 RF (동일한 파라미터 수!)

장점: 추가 파라미터나 다운샘플링 없이 더 큰 수용 영역 확보.
```

### 4.2 ASPP 모듈

```python
class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""

    def __init__(self, in_channels, out_channels=256, rates=[6, 12, 18]):
        super().__init__()
        modules = []

        # 1×1 합성곱
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))

        # 다양한 비율의 Atrous 합성곱
        for rate in rates:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

        # 전역 평균 풀링 분기
        modules.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ))

        self.branches = nn.ModuleList(modules)

        # 연결된 특징 투영
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * (len(rates) + 2), out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        outputs = []
        for branch in self.branches[:-1]:
            outputs.append(branch(x))

        # 전역 풀링 분기: 입력 크기로 업샘플링
        gap = self.branches[-1](x)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        outputs.append(gap)

        x = torch.cat(outputs, dim=1)
        return self.project(x)
```

---

## 5. 세그멘테이션 손실 함수

### 5.1 일반적인 손실 함수

```python
def cross_entropy_loss(pred, target, ignore_index=255):
    """세그멘테이션을 위한 표준 교차 엔트로피."""
    return F.cross_entropy(pred, target, ignore_index=ignore_index)


def dice_loss(pred, target, smooth=1.0):
    """Dice 손실: 불균형 클래스에 적합."""
    pred = F.softmax(pred, dim=1)
    n_classes = pred.shape[1]
    total_loss = 0

    for c in range(n_classes):
        pred_c = pred[:, c]
        target_c = (target == c).float()

        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()

        dice = (2 * intersection + smooth) / (union + smooth)
        total_loss += (1 - dice)

    return total_loss / n_classes


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal 손실: 쉬운 샘플의 가중치를 줄임."""
    ce = F.cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()


class CombinedLoss(nn.Module):
    """최상의 결과를 위해 CE + Dice를 결합."""

    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred, target):
        ce = cross_entropy_loss(pred, target)
        dl = dice_loss(pred, target)
        return self.ce_weight * ce + self.dice_weight * dl
```

---

## 6. 평가 지표

### 6.1 세그멘테이션 지표

```python
import numpy as np


def compute_iou(pred, target, n_classes):
    """
    클래스별 Intersection over Union (IoU).
    Jaccard Index라고도 함.

    IoU = TP / (TP + FP + FN)
    """
    ious = []
    for c in range(n_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()

        if union == 0:
            ious.append(float('nan'))  # 클래스가 존재하지 않음
        else:
            ious.append(intersection / union)

    return ious


def mean_iou(pred, target, n_classes):
    """Mean IoU (mIoU): 모든 클래스에 대한 평균 IoU."""
    ious = compute_iou(pred, target, n_classes)
    valid = [iou for iou in ious if not np.isnan(iou)]
    return np.mean(valid) if valid else 0.0


def pixel_accuracy(pred, target):
    """전체 픽셀 정확도."""
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def evaluate_segmentation(model, dataloader, n_classes, device='cpu'):
    """데이터셋에 대한 전체 평가."""
    total_iou = np.zeros(n_classes)
    total_count = np.zeros(n_classes)
    total_correct = 0
    total_pixels = 0

    model.eval()
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for c in range(n_classes):
                pred_c = (preds == c)
                target_c = (targets == c)

                intersection = (pred_c & target_c).sum().item()
                union = (pred_c | target_c).sum().item()

                if union > 0:
                    total_iou[c] += intersection / union
                    total_count[c] += 1

            total_correct += (preds == targets).sum().item()
            total_pixels += targets.numel()

    # mIoU 계산
    class_ious = []
    for c in range(n_classes):
        if total_count[c] > 0:
            class_ious.append(total_iou[c] / total_count[c])

    miou = np.mean(class_ious) if class_ious else 0.0
    pixel_acc = total_correct / total_pixels

    print(f"mIoU: {miou:.4f}")
    print(f"Pixel Accuracy: {pixel_acc:.4f}")
    return miou, pixel_acc
```

---

## 7. 실습 구현

### 7.1 학습 파이프라인

```python
def train_segmentation(model, train_loader, val_loader, n_classes,
                        epochs=50, lr=1e-3, device='cuda'):
    """세그멘테이션을 위한 완전한 학습 파이프라인."""
    criterion = CombinedLoss(ce_weight=1.0, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model = model.to(device)
    best_miou = 0

    for epoch in range(epochs):
        # 학습
        model.train()
        total_loss = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device).long()

            outputs = model(images)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # 검증
        miou, pixel_acc = evaluate_segmentation(
            model, val_loader, n_classes, device
        )

        print(f"Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, "
              f"mIoU={miou:.4f}, PixAcc={pixel_acc:.4f}")

        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_segmentation.pth')

    return best_miou
```

---

## 8. 연습문제

### 연습문제 1: FCN 구현

FCN을 처음부터 구축하세요:
1. FCN-32s (단일 32배 업샘플링) 구현
2. FCN-16s와 FCN-8s를 위한 스킵 연결 추가
3. Pascal VOC 2012 세그멘테이션 데이터셋으로 학습
4. FCN-32s, FCN-16s, FCN-8s 비교: 스킵 연결로 인한 mIoU 개선
5. 이미지 위에 오버레이된 세그멘테이션 예측 시각화

### 연습문제 2: 의료 영상을 위한 U-Net

의료 세그멘테이션 작업을 위한 U-Net 구축:
1. 의료 영상 데이터셋 다운로드 (예: 폐 CT, 세포 세그멘테이션)
2. 구성 가능한 깊이를 가진 U-Net 구현
3. CE + Dice 결합 손실로 학습
4. 장기/구조별 IoU와 Dice 점수로 평가
5. 데이터 증강 적용: 회전, 뒤집기, 탄성 변형

### 연습문제 3: ASPP를 사용한 DeepLab v3+

DeepLab v3+ 아키텍처 구현:
1. 비율 [6, 12, 18]의 ASPP 모듈 구축
2. ResNet-50 백본 사용 (ImageNet에서 사전 학습)
3. 인코더-디코더 구조 구현
4. Cityscapes 데이터셋 (도시 장면 세그멘테이션)으로 학습
5. 동일 데이터셋에서 U-Net과 비교

### 연습문제 4: 손실 함수 비교

세그멘테이션 손실 함수 비교:
1. 구현: 교차 엔트로피, Dice, Focal, Lovasz-softmax
2. 불균형 데이터셋에서 각 손실로 동일 모델 학습
3. 측정: mIoU, 클래스별 IoU, 수렴 속도
4. 의도적으로 불균형한 데이터셋 생성 (1개의 희귀 클래스)
5. Dice/Focal 손실이 불균형을 더 잘 처리함을 보여주기

### 연습문제 5: 실시간 세그멘테이션

실시간 세그멘테이션 시스템 구축:
1. 경량 모델 구현 (예: BiSeNet 또는 ENet)
2. 속도 최적화: 채널 수 줄이기, 깊이별 분리 가능 합성곱 사용
3. CPU와 GPU에서 FPS 측정
4. 웹캠 피드에 실시간 세그멘테이션 적용
5. 속도-정확도 트레이드오프 비교: U-Net vs 경량 모델

---

*25강 끝*
