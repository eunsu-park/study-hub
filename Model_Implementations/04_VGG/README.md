# 04. VGG

## 개요

VGGNet은 2014년 ILSVRC에서 2위를 차지한 모델로, Karen Simonyan과 Andrew Zisserman이 제안했습니다. "Very Deep Convolutional Networks for Large-Scale Image Recognition" 논문에서 **3x3 작은 필터를 깊게 쌓는 것**이 효과적임을 보여주었습니다.

---

## 수학적 배경

### 1. 3x3 필터 스택의 효과

```
왜 3x3 필터를 여러 개 쌓는가?

2개의 3x3 conv ≈ 1개의 5x5 conv (같은 receptive field)
3개의 3x3 conv ≈ 1개의 7x7 conv

장점:
1. 파라미터 수 감소:
   - 7x7: 49C² 파라미터
   - 3x3 × 3: 27C² 파라미터 (45% 감소)

2. 비선형성 증가:
   - 7x7: 1개의 ReLU
   - 3x3 × 3: 3개의 ReLU → 더 복잡한 함수 학습 가능
```

### 2. Receptive Field 계산

```
레이어가 쌓일수록 receptive field 증가:

RF = (RF_prev - 1) × stride + kernel_size

예시 (stride=1, kernel=3):
- Layer 1: RF = 3
- Layer 2: RF = 5
- Layer 3: RF = 7
- Layer 4: RF = 9
...

MaxPool (kernel=2, stride=2) 후:
- RF가 2배로 확장
```

### 3. Feature Map 크기 변화

```
Conv (stride=1, padding=1, kernel=3):
  H_out = H_in  (크기 유지)

MaxPool (kernel=2, stride=2):
  H_out = H_in / 2  (크기 절반)

224 → [Conv×2] → 224 → Pool → 112 → [Conv×2] → 112 → Pool → 56 → ...
```

---

## VGG 아키텍처

### VGG 변형 비교

| 구성 | VGG11 | VGG13 | VGG16 | VGG19 |
|------|-------|-------|-------|-------|
| Conv Layers | 8 | 10 | 13 | 16 |
| FC Layers | 3 | 3 | 3 | 3 |
| Total Layers | 11 | 13 | 16 | 19 |
| Parameters | 133M | 133M | 138M | 144M |

### VGG16 상세 구조

```
입력: 224×224×3 RGB 이미지

Block 1: [Conv3-64] × 2 + MaxPool
  (224×224×3) → (224×224×64) → (112×112×64)

Block 2: [Conv3-128] × 2 + MaxPool
  (112×112×64) → (112×112×128) → (56×56×128)

Block 3: [Conv3-256] × 3 + MaxPool
  (56×56×128) → (56×56×256) → (28×28×256)

Block 4: [Conv3-512] × 3 + MaxPool
  (28×28×256) → (28×28×512) → (14×14×512)

Block 5: [Conv3-512] × 3 + MaxPool
  (14×14×512) → (14×14×512) → (7×7×512)

Classifier:
  Flatten: 7×7×512 = 25,088
  FC1: 25088 → 4096 + ReLU + Dropout
  FC2: 4096 → 4096 + ReLU + Dropout
  FC3: 4096 → 1000 (classes)

파라미터 분포:
- Conv layers: ~15M (11%)
- FC layers: ~124M (89%)  ← 대부분!
```

### VGG 설정 (Configuration)

```python
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
# 'M' = MaxPool
```

---

## 파일 구조

```
04_VGG/
├── README.md                      # 이 파일
├── pytorch_lowlevel/
│   └── vgg_lowlevel.py           # F.conv2d, F.linear 사용
├── paper/
│   └── vgg_paper.py              # 논문 아키텍처 정확 재현
└── exercises/
    ├── 01_feature_visualization.md   # 각 블록 feature map 시각화
    └── 02_transfer_learning.md       # 사전학습 가중치 활용
```

---

## 핵심 개념

### 1. Deep & Narrow vs Shallow & Wide

```
VGG 이전: 큰 필터 + 얕은 네트워크
  - AlexNet: 11×11, 5×5 필터
  - 적은 레이어

VGG: 작은 필터 + 깊은 네트워크
  - 오직 3×3 필터 (+ 일부 1×1)
  - 16~19 레이어

결론: 깊이가 성능에 매우 중요
```

### 2. 균일한 구조

```
VGG의 설계 원칙:

1. 모든 Conv는 3×3, stride=1, padding=1
2. 모든 MaxPool은 2×2, stride=2
3. 블록마다 채널 수 2배 증가 (64→128→256→512)
4. 간단하고 규칙적 → 이해/구현 용이
```

### 3. VGG의 한계

```
단점:
1. 파라미터 과다 (138M, ResNet-50: 25M)
2. 메모리 소비 큼 (FC 레이어)
3. 학습 느림
4. Gradient vanishing (깊어질수록)

후속 연구:
- GoogLeNet: Inception 모듈로 효율성
- ResNet: Skip connection으로 더 깊게
- MobileNet: Depthwise separable conv
```

### 4. VGG as Feature Extractor

```
VGG는 특징 추출기로 널리 사용:

1. Style Transfer
   - 콘텐츠: block4_conv2
   - 스타일: block1~5_conv1

2. Perceptual Loss
   - 픽셀 손실 대신 VGG 특징 비교

3. Object Detection
   - VGG backbone + detection head
```

---

## 구현 레벨

### Level 2: PyTorch Low-Level (pytorch_lowlevel/)

- F.conv2d, F.max_pool2d, F.linear 사용
- nn.Conv2d, nn.Linear 미사용
- 파라미터 수동 초기화 및 관리
- 블록 단위 모듈화

### Level 3: Paper Implementation (paper/)

- 논문의 모든 설정 재현
- Batch Normalization 추가 (VGG-BN)
- 다양한 VGG 변형 지원

---

## 학습 체크리스트

- [ ] 3×3 필터 스택의 장점 이해
- [ ] Receptive field 계산 방법 숙지
- [ ] VGG16 아키텍처 암기
- [ ] 파라미터 분포 이해 (Conv vs FC)
- [ ] VGG를 feature extractor로 활용하는 방법
- [ ] VGG의 한계와 후속 모델 비교

---

## 참고 자료

- Simonyan & Zisserman (2014). "Very Deep Convolutional Networks for Large-Scale Image Recognition"
- [torchvision VGG](https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py)
- [CS231n: ConvNets](https://cs231n.github.io/convolutional-networks/)
- [../03_CNN_LeNet/README.md](../03_CNN_LeNet/README.md)
