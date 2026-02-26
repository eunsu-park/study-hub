# 07. 전이학습 (Transfer Learning)

## 학습 목표

- 전이학습의 개념과 이점
- 사전 학습 모델 활용
- 미세 조정(Fine-tuning) 전략
- 실전 이미지 분류 프로젝트

---

## 1. 전이학습이란?

### 개념

초기 CNN 레이어는 엣지(Edge), 텍스처(Texture), 색상처럼 과제에 관계없이 유용한 보편적 특성(Universal Feature)을 학습합니다. 최종 레이어만이 원래 데이터셋에 특화됩니다. 초기 레이어를 동결하고 최종 레이어를 재학습하면, 소규모 데이터셋에서도 이러한 보편적 특성을 활용할 수 있습니다. 이것이 가능한 이유는 시각 특성이 계층적이기 때문입니다: 엣지가 모여 텍스처가 되고, 텍스처가 모여 부품이 되고, 부품이 모여 객체가 됩니다 — 이 계층의 하위 수준은 거의 모든 이미지 인식 과제에서 공유됩니다.

```
Model trained on ImageNet
        ↓
    Low-level features (edges, textures) → Reuse
        ↓
    High-level features → Adapt to new data
        ↓
    New classification task
```

### 이점

- 적은 데이터로도 높은 성능
- 빠른 학습
- 더 나은 일반화

---

## 2. 전이학습 전략

**미세 조정(Fine-tuning) vs 특성 추출(Feature Extraction) 선택 기준**: 소규모 데이터셋 + 유사한 도메인(예: ImageNet 모델로 개 품종 분류)이라면 대부분의 레이어를 동결하고 헤드(Head)만 학습하세요. 대규모 데이터셋 + 다른 도메인(예: 의료 X선)이라면 작은 학습률로 모든 레이어를 미세 조정하여 네트워크가 새로운 이미지 분포에 맞게 저수준 특성을 조정할 수 있도록 하세요. 그 중간이라면 점진적 해동(Gradual Unfreezing)이 안전한 절충안을 제공합니다.

### 전략 1: 특성 추출 (Feature Extraction)

```python
# Freeze: prevents gradient updates to pretrained weights, preserving
# the learned universal features (edges, textures, shapes)
for param in model.parameters():
    param.requires_grad = False

# Replace the classification head to match our number of classes;
# pretrained weights for everything else remain intact
model.fc = nn.Linear(2048, num_classes)
```

- 사전 학습된 특징 그대로 사용
- 마지막 분류층만 학습
- 데이터가 적을 때 적합

### 전략 2: 미세 조정 (Fine-tuning)

```python
# Unfreeze all layers for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Use a very low learning rate (1e-5) — large updates would destroy
# the pretrained features. The goal is to gently nudge the weights
# toward the new domain, not to learn from scratch.
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
```

- 사전 학습 가중치를 시작점으로
- 전체 네트워크 미세 조정
- 데이터가 충분할 때 적합

### 전략 3: 점진적 해동 (Gradual Unfreezing)

```python
# Step 1: Last layer only
for param in model.parameters():
    param.requires_grad = False
model.fc.requires_grad_(True)
train_for_epochs(5)

# Step 2: Last block too
model.layer4.requires_grad_(True)
train_for_epochs(5)

# Step 3: Entire network
model.requires_grad_(True)
train_for_epochs(10)
```

---

## 3. PyTorch 구현

### 기본 전이학습

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets

# 1. Load pretrained model — weights were learned from 1.2M ImageNet images
model = models.resnet50(weights='IMAGENET1K_V2')

# 2. Freeze all backbone weights — only the new head will be trained,
# so we optimize far fewer parameters (much faster, less overfitting)
for param in model.parameters():
    param.requires_grad = False

# 3. Replace the classification head to match our number of classes;
# the new layers are randomly initialized and will be the only trainable params
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),                    # Regularize the high-dimensional backbone output
    nn.Linear(num_features, 256),       # Reduce from 2048 to 256 — bottleneck prevents overfitting
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
```

### 데이터 전처리

```python
# Use ImageNet normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])
```

---

## 4. 학습 전략

### 차등 학습률 (Discriminative Learning Rates)

```python
# Different learning rates for each layer group — earlier layers learn
# universal features that need minimal adjustment (tiny LR), while later
# layers and the FC head are more task-specific and need larger updates
optimizer = torch.optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},   # Edges, textures — nearly universal
    {'params': model.layer2.parameters(), 'lr': 5e-5},   # Low-level combinations
    {'params': model.layer3.parameters(), 'lr': 1e-4},   # Mid-level features
    {'params': model.layer4.parameters(), 'lr': 5e-4},   # High-level, task-specific features
    {'params': model.fc.parameters(), 'lr': 1e-3},       # New head — learns from scratch
])
```

### 학습률 스케줄링

```python
# Warmup + Cosine Decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=epochs,
    steps_per_epoch=len(train_loader),
    pct_start=0.1  # 10% warmup
)
```

---

## 5. 다양한 사전 학습 모델

### torchvision 모델

```python
# Classification
resnet50 = models.resnet50(weights='IMAGENET1K_V2')
efficientnet = models.efficientnet_b0(weights='IMAGENET1K_V1')
vit = models.vit_b_16(weights='IMAGENET1K_V1')

# Object detection
fasterrcnn = models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

# Segmentation
deeplabv3 = models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
```

### timm 라이브러리

```python
import timm

# Check available models
print(timm.list_models('*efficientnet*'))

# Load model
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=10)
```

---

## 6. 실전 프로젝트: 꽃 분류

### 데이터 준비

```python
# Flowers102 dataset
from torchvision.datasets import Flowers102

train_data = Flowers102(
    root='data',
    split='train',
    transform=train_transform,
    download=True
)

test_data = Flowers102(
    root='data',
    split='test',
    transform=val_transform
)
```

### 모델 및 학습

```python
class FlowerClassifier(nn.Module):
    def __init__(self, num_classes=102):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')

        # Replace last layer
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# Training
model = FlowerClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
```

---

## 7. 주의사항

### 데이터 크기별 전략

| 데이터 크기 | 전략 | 설명 |
|-------------|------|------|
| 매우 적음 (<1000) | 특성 추출 | 마지막 층만 학습 |
| 적음 (1000-10000) | 점진적 해동 | 후반 층부터 해동 |
| 보통 (10000+) | 전체 미세 조정 | 낮은 학습률로 전체 학습 |

### 도메인 유사성

```
Similar to ImageNet (animals, objects):
    → Can use shallow layers as-is

Different from ImageNet (medical, satellite):
    → Need to fine-tune deeper layers
```

### 일반적인 실수

1. ImageNet 정규화 누락
2. 너무 높은 학습률
3. 훈련/평가 모드 전환 잊음
4. 가중치 고정 후 optimizer에 포함

---

## 8. 성능 향상 팁

### 데이터 증강

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    normalize
])
```

### Label Smoothing

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

### Mixup / CutMix

```python
def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam
```

---

## 정리

### 핵심 개념

1. **특성 추출**: 사전 학습 특징 재사용
2. **미세 조정**: 낮은 학습률로 전체 조정
3. **점진적 해동**: 후반 층부터 순차적 학습

### 체크리스트

- [ ] ImageNet 정규화 사용
- [ ] 적절한 학습률 선택 (1e-4 ~ 1e-5)
- [ ] model.train() / model.eval() 전환
- [ ] 데이터 증강 적용
- [ ] 조기 종료 설정

---

## 연습 문제

### 연습 1: 특성 추출(Feature Extraction) vs 미세 조정(Fine-tuning) 비교

소규모 데이터셋에서 특성 추출과 전체 미세 조정 전략을 비교하세요.

1. Flowers102 데이터셋을 사용하여 학습 샘플을 500개로 제한하세요.
2. 전략 A: ResNet-18의 모든 레이어를 동결하고 최종 FC 레이어만 10 에포크 학습하세요.
3. 전략 B: 모든 레이어를 해동하고 `lr=1e-5`로 10 에포크 미세 조정하세요.
4. 두 전략의 최종 검증 정확도를 기록하세요.
5. 트레이드오프를 설명하세요: 특성 추출이 유리한 경우와 미세 조정이 유리한 경우는?

### 연습 2: 점진적 해동(Gradual Unfreezing) 스케줄

수업에서 배운 3단계 점진적 해동 전략을 구현하세요.

1. 사전 학습된 ResNet-18을 불러오세요.
2. 1단계 (에포크 1~5): 최종 FC 레이어만 학습하세요.
3. 2단계 (에포크 6~10): `layer4`도 해동하여 학습하세요.
4. 3단계 (에포크 11~20): `lr=1e-5`로 모든 레이어를 해동하여 학습하세요.
5. 20 에포크 전체에 걸친 검증 정확도를 그래프로 표시하고, 각 단계 경계를 수직선으로 표시하세요.

### 연습 3: 차별적 학습률(Discriminative Learning Rate)

레이어 그룹별로 다른 학습률을 적용하고 효과를 관찰하세요.

1. 사전 학습된 ResNet-18을 불러오세요.
2. 5개의 파라미터 그룹으로 옵티마이저를 설정하세요: `layer1` (lr=1e-5), `layer2` (lr=3e-5), `layer3` (lr=1e-4), `layer4` (lr=3e-4), `fc` (lr=1e-3).
3. CIFAR-10의 소규모 샘플(2000개)에서 15 에포크 학습하세요.
4. 동일한 에포크 동안 균일한 lr=1e-4로 학습한 결과와 비교하세요.
5. 직관을 설명하세요: 초기 레이어에 더 작은 학습률을 사용해야 하는 이유는?

### 연습 4: 도메인 간격(Domain Gap) 조사

소스(ImageNet)와 타겟 도메인의 유사성이 전이 학습 품질에 미치는 영향을 탐구하세요.

1. 두 가지 타겟 데이터셋을 선택하세요: CIFAR-10(자연 이미지, ImageNet과 유사)과 의료 또는 텍스처 데이터셋.
2. 각 데이터셋에 대해 (a) 처음부터 학습, (b) 특성 추출만, (c) 전체 미세 조정의 성능을 비교하세요.
3. 결과를 표로 정리하세요.
4. 도메인 간격으로 인해 데이터셋마다 다른 전략이 유리한 이유를 설명하세요.

### 연습 5: ImageNet 정규화 — 생략하면 어떻게 되나

전이 학습에서 ImageNet 정규화(Normalization)가 필수적임을 경험적으로 검증하세요.

1. 사전 학습된 EfficientNet-B0을 불러오세요.
2. 100개 샘플 평가 세트에서 두 가지 실험을 실행하세요: (a) 올바른 ImageNet 정규화 `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]`, (b) 정규화 없이(또는 `mean=0.5, std=0.5` 등 잘못된 정규화).
3. 두 경우의 Top-1 정확도를 비교하세요.
4. 두 전처리 방식에서 샘플 이미지를 시각화하고, 입력 분포가 달라질 때 사전 학습된 모델의 특성 감지기가 왜 작동하지 않는지 설명하세요.

---

## 다음 단계

[RNN 기초 (Recurrent Neural Networks)](./13_RNN_Basics.md)에서 순환 신경망을 학습합니다.
