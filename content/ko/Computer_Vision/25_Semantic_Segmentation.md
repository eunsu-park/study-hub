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

참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 픽셀 단위 분류로서의 세그멘테이션, FCN → U-Net → DeepLab 아키텍처 진화, receptive-field 문제, 그리고 IoU / Dice 손실 공식화를 다룹니다.

1. [세그멘테이션 개요](#1-세그멘테이션-개요)
2. [Fully Convolutional Networks (FCN)](#2-fully-convolutional-networks-fcn)
3. [U-Net 아키텍처](#3-u-net-아키텍처)
4. [DeepLab v3+](#4-deeplab-v3)
5. [세그멘테이션 손실 함수](#5-세그멘테이션-손실-함수)
6. [평가 지표](#6-평가-지표)
7. [실습 구현](#7-실습-구현)
8. [연습문제](#8-연습문제)

---

## 이론과 원리

시맨틱 세그멘테이션은 이미지의 **모든 픽셀**을 고정된 클래스 집합 중 하나에 할당합니다. 경계 상자 없음, 객체 인스턴스 없음 — 단지 "도로의 모든 픽셀은 도로 클래스, 하늘의 모든 픽셀은 하늘 클래스". 분류의 픽셀 단위 극단: 이미지당 한 레이블이 아니라 픽셀당 한 레이블.

이 섹션은 다음을 다룹니다:

- **(A) 픽셀 단위 분류 프레이밍** — 세그멘테이션을 검출과 근본적으로 다르게 만드는 것.
- **(B) FCN이 돌파구였던 이유** — 분류 대신 밀집 예측을 위해 CNN 사용.
- **(C) Receptive-field 문제와 skip connection** — U-Net의 아키텍처적 답.
- **(D) Dilated / atrous 컨볼루션** — 같은 문제에 대한 DeepLab의 답.
- **(E) 손실 함수** — cross-entropy, dice loss, focal loss, 그리고 각각이 맞는 때.
- **(F) 평가 메트릭** — IoU와 mIoU, pixel accuracy, accuracy가 오해를 불러일으키는 이유.

### A. 픽셀 단위 분류

시맨틱 세그멘테이션 출력 모양: `(H, W, K)` logits, `K`가 클래스 수. 픽셀당 클래스 확률을 얻기 위해 `K` 축을 따라 softmax 적용. 네트워크는 구조적으로 **모든 픽셀에 적용된 분류기처럼 생김** — 따라서 "픽셀 단위 분류".

이 프레이밍은 즉시 문제를 시사: 분류 네트워크는 픽셀당이 아니라 이미지당 한 예측을 출력하도록 설계됨. 네트워크 내내 공격적으로 다운샘플링(max-pool, strided conv), `224×224` 이미지를 끝에서 `1×1` 특징 벡터로 축소. 세그멘테이션에는 출력이 입력과 같은 해상도여야 함.

세그멘테이션의 전체 아키텍처 역사는 픽셀 레이블을 올바로 분류할 충분한 전역 컨텍스트를 집계하면서도 픽셀 레이블을 국소화할 만큼 공간 해상도를 높게 유지하는 방법에 관한 것.

### B. FCN: 돌파구

Long, Shelhamer & Darrell(2015)가 Fully Convolutional Network를 도입. 두 아이디어:

1. **Fully-connected 분류기 헤드 제거**하고 `K` 채널을 생성하는 1×1 컨볼루션으로 대체. 네트워크가 이제 단일 벡터 대신 저해상도 클래스 확률 맵을 출력.
2. Transposed convolution(또는 bilinear 보간 + 컨볼루션)을 통해 출력을 입력 해상도로 **업샘플**.

아키텍처: 사전 훈련된 분류 네트워크(VGG)를 가져와, FC layer를 벗기고, conv layer로 대체, 세그멘테이션 데이터로 훈련. 네트워크가 이제 인간의 눈이 기대하는 것을 수행: 밀집 픽셀 단위 클래스 예측.

바닐라 FCN의 문제: 특징이 최종 분류기에 도달할 때쯤 공간 해상도가 입력보다 32× 낮음. 그 32×를 원 해상도로 다시 업샘플하면 세밀한 디테일이 없는 흐릿한 세그멘테이션 — 필요한 곳(경계)에서 바로.

### C. U-Net: 해상도를 위한 Skip Connection

U-Net(Ronneberger 등, 2015, 원래 바이오메디컬 이미지용)은 FCN의 해상도 손실을 **대칭 인코더-디코더 구조와 skip connection**으로 해결:

```
입력 (572×572)
  │
  │  인코더(다운샘플링 경로): 4단계 conv + pool
  │
  ▼    ─────────► skip connection ─────────┐
Level 1 (64 ch)                             │
  │                                         │
  │                                         ▼  디코더 출력 (64 ch)
  ▼    ─────────► skip connection ──────┐    ▲
Level 2 (128 ch)                        │    │ 업샘플
  │                                     │    │
  │                                     ▼    │
  ▼    ─────────► skip connection ──┐   ... (각 레벨에서 같은 패턴)
Level 3 (256 ch)                    │
  │                                 │
  │                                 │
  ▼                                 │
Bottleneck (1024 ch, 32×32)         │
```

핵심 통찰: 인코더가 더 깊이 갈수록 공간 정보를 버림; skip connection이 그 공간 정보를 **직접** 디코더로 라우팅, 거기서 업샘플된 특징과 연결됨. 디코더는 따라서 **의미 정보**(전체 이미지를 본 bottleneck에서)와 **공간 정보**(세밀한 디테일을 보존하는 skip connection에서) 둘 다 가짐.

U-Net은 의료 영상, 위성 이미지, 많은 다른 세그멘테이션 작업의 템플릿이 됨. 변형(nnU-Net, TransUNet) 모두 skip-connection 아이디어 공유.

### D. DeepLab: 다운샘플링 없는 Atrous 컨볼루션

DeepLab(Chen 등, 2015-2018)은 다른 접근. 다운샘플링과 업샘플링(U-Net 스타일) 대신, 특징 맵을 더 높은 해상도로 유지하고 **atrous(dilated) 컨볼루션**을 써서 receptive field를 키움:

- dilation rate 1의 일반 3×3 conv는 3×3 영역을 봄.
- dilation rate 2의 3×3 conv는 5×5 영역을 보지만 9점만(하나 걸러) 샘플링.
- dilation rate 4의 3×3 conv는 같은 9 파라미터로 9×9 영역을 봄.

증가하는 rate의 atrous conv를 쌓으면 다운샘플링 없이 큰 효과적 receptive field를 얻음. DeepLab v3+의 **Atrous Spatial Pyramid Pooling(ASPP)** 모듈은 서로 다른 dilation rate를 가진 여러 병렬 atrous conv를 적용하고 연결, 다중 스케일 컨텍스트 포착.

DeepLab v3+는 atrous 백본 위에 skip connection을 가진 작은 디코더를 추가, 두 접근 모두 결합.

### E. 손실 함수

#### E.1 Cross-entropy

기본: 픽셀별 범주형 cross-entropy. 분류와 같음, 단지 모든 픽셀에 적용. 문제: **클래스 불균형**. 주행 장면에서 60% 픽셀이 도로이고 0.5%가 교통 표지일 수 있음. Cross-entropy는 모든 픽셀을 동등하게 취급, 네트워크가 도로에는 매우 능숙해지지만 표지는 거의 학습하지 않음.

#### E.2 Dice / IoU 손실

Dice loss는 예측과 ground-truth 마스크 간 겹침을 직접 최적화:

```
Dice(A, B) = 2 · |A ∩ B| / (|A| + |B|)
Loss = 1 - Dice
```

이진 마스크: `Dice = 2 · Σ(p · g) / (Σp + Σg)` (`p`, `g`는 예측 및 ground-truth 확률). 클래스 불균형에 둔감 — 배경이 아니라 전경 겹침만 신경 쓰기 때문. 관심 클래스(종양)가 배경에 비해 작은 의료 영상에 인기.

#### E.3 Focal loss

Cross-entropy에 **잘 분류된 쉬운 픽셀을 다운가중**하는 추가 `(1 - p)^γ` 인자, 어려운 픽셀에 훈련 집중. 클래스 불균형과 싸우는 또 다른 방법, RetinaNet이 도입, 세그멘테이션에도 인기.

전형 실천: cross-entropy와 dice loss 결합(합 또는 가중 평균). Cross-entropy가 안정적 기울기 제공; dice가 메트릭의 직접 최적화 제공.

### F. 평가 메트릭

#### F.1 Pixel accuracy

가장 단순: 올바로 분류된 픽셀의 비율. 클래스 불균형 때문에 오해를 불러일으킴 — 주행 장면 데이터셋에서 절반 클래스에 대해 "모든 곳 도로"를 예측해도 95% pixel accuracy를 얻을 수 있음.

#### F.2 Intersection-over-Union (IoU)

클래스별 IoU:

```
IoU_c = |pred_c ∩ true_c| / |pred_c ∪ true_c|
```

클래스 `c`에 대해: 예측 및 진짜 `c`-픽셀의 합집합 중 둘 다에 있는 비율. IoU 1이 완벽, 0이 겹침 없음.

#### F.3 Mean IoU (mIoU)

모든 `K` 클래스에 걸친 평균 IoU, 각 클래스가 픽셀 수와 무관하게 동등하게 가중. **이것이 표준 세그멘테이션 메트릭** — Cityscapes, ADE20K, Pascal VOC, 모든 벤치마크에서. 희귀 클래스가 흔한 클래스와 같게 가중되어 클래스 불균형에 강건.

부수 이점: mIoU가 개선되고 pixel accuracy가 감소하면, 희귀 클래스 세그멘테이션에서 약간 지배적 클래스를 희생해 나아지고 있음 — 보통 옳은 트레이드오프.

### 이론에서 아래 함수들로

- 현대 라이브러리(PyTorch: `torchvision.models.segmentation`, `segmentation_models_pytorch`)가 한 줄 로딩으로 사전 훈련된 FCN, U-Net, DeepLab 모델 제공.
- OpenCV의 DNN 모듈은 내보낸 ONNX 세그멘테이션 모델 실행 가능; 추론 파이프라인은 §19를 따름.
- 주요 하이퍼파라미터: 입력 크기(클수록 = 더 많은 컨텍스트지만 더 느림), 백본(ResNet, EfficientNet), 손실 함수 조합(CE + Dice).
- 후처리: 최종 레이블 맵을 위해 클래스 축에 argmax, 에지 정제를 위한 선택적 CRF(conditional random field).

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
