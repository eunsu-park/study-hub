# 05. CNN 기초 (Convolutional Neural Networks)

## 학습 목표

- 합성곱 연산의 원리 이해
- 풀링, 패딩, 스트라이드 개념
- PyTorch로 CNN 구현
- MNIST/CIFAR-10 분류

---

## 1. 합성곱 (Convolution) 연산

### 개념

이미지는 지역적 구조를 가지고 있습니다 — 인접 픽셀들은 서로 상관관계가 있습니다. 완전 연결 층(Fully Connected Layer)은 이미지를 평탄한 벡터로 취급하여 이를 무시합니다. 합성곱(Convolution)은 이미지를 슬라이딩하는 작은 필터를 사용해 지역성을 활용하고, 파라미터를 크게 줄이면서 공간 패턴을 포착합니다. 이것이 CNN이 **공간 계층 구조(Spatial Hierarchy)**를 만드는 이유입니다: 초기 층은 작은 영역에서 에지와 텍스처를 감지하고, 더 깊은 층은 이를 복잡한 형태와 객체로 조합합니다.

이미지의 지역적 패턴(에지, 텍스처)을 감지합니다.

```
Input Image     Filter(Kernel)     Output
[1 2 3 4]       [1 0]              [?]
[5 6 7 8]  *    [0 1]   =
[9 0 1 2]
```

### 수식

```
Output[i,j] = Σ Σ Input[i+m, j+n] × Filter[m, n]
```

**곱하고 더하기가 왜 패턴을 감지할까?** 이 연산은 필터와 로컬 이미지 패치 사이의 *내적(Dot Product)*입니다. 내적은 **유사도**를 측정합니다: 패치의 픽셀 패턴이 필터 가중치와 일치하면(같은 위치에서 둘 다 양수, 둘 다 음수) 합이 커집니다. 일치하지 않으면 양수와 음수 항이 상쇄되어 출력은 0에 가깝습니다. 따라서 각 필터는 본질적으로 *템플릿*이며, 출력 맵은 이미지가 해당 템플릿과 국소적으로 유사한 곳에서 활성화됩니다.

### 차원 계산

```
Output size = (Input - Kernel + 2×Padding) / Stride + 1

Example: Input 32×32, Kernel 3×3, Padding 1, Stride 1
         = (32 - 3 + 2) / 1 + 1 = 32
```

---

## 2. 주요 개념

### 패딩 (Padding)

```
Add zeros to input borders to maintain output size

padding='same': Output = Input size
padding='valid': No padding (Output < Input)
```

### 스트라이드 (Stride)

```
Filter movement interval

stride=1: Move one pixel at a time (default)
stride=2: Move two pixels at a time → Output size halved
```

### 풀링 (Pooling)

점진적인 다운샘플링(Downsampling)은 이동 불변성(Translation Invariance)을 제공하고(이동된 고양이도 여전히 고양이) 더 깊은 층의 연산량을 줄입니다. 각 지역 영역 내의 정확한 위치 정보를 버림으로써, 네트워크는 특징이 *어디에서* 감지되었는지보다 *감지되었는지 여부*에 집중합니다.

```
Reduce spatial size, increase invariance

Max Pooling: Maximum value in region
Avg Pooling: Average value in region
```

맥스 풀링(Max Pooling)은 각 영역에서 가장 강한 활성화를 선택합니다 — "이 이웃 영역 *어딘가에서* 이 특징이 감지되었나?"라고 묻는 것과 같습니다. 수학적으로 2x2 풀 윈도우의 경우: `y = max(x_{i,j}, x_{i+1,j}, x_{i,j+1}, x_{i+1,j+1})`. 맥스 연산은 구간별 선형 함수로, 그 기울기는 최댓값 원소에 대해 1이고 나머지는 0입니다 — 역전파 시 가장 활성화된 위치만 업데이트됩니다.

---

## 3. CNN 구조

### 기본 구조

```
Input → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

### LeNet-5 (1998)

```
Input (32×32×1)
  ↓
Conv1 (5×5, 6 channels) → 28×28×6
  ↓
MaxPool (2×2) → 14×14×6
  ↓
Conv2 (5×5, 16 channels) → 10×10×16
  ↓
MaxPool (2×2) → 5×5×16
  ↓
Flatten → 400
  ↓
FC → 120 → 84 → 10
```

---

## 4. PyTorch Conv2d

### 기본 사용법

```python
import torch.nn as nn

# Conv2d(in_channels, out_channels, kernel_size, stride, padding)
conv = nn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=64,    # 64 filters
    kernel_size=3,      # 3×3 kernel
    stride=1,
    padding=1           # same padding
)

# Input: (batch, channels, height, width)
x = torch.randn(1, 3, 32, 32)
out = conv(x)  # (1, 64, 32, 32)
```

### MaxPool2d

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# 32×32 → 16×16
```

---

## 5. MNIST CNN 구현

### 모델 정의

```python
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv block 1
        # kernel_size=3: 3×3 is the smallest kernel that captures all 8 neighbors
        # — the sweet spot between expressiveness and efficiency (VGG principle)
        # padding=1: "Same" padding so output spatial size equals input,
        # preventing information loss at borders
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Conv block 2
        # Double the channels (32→64): deeper layers need more filters to
        # represent the combinatorial explosion of higher-level features
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # FC block
        # 64*7*7 = 3136: after two 2×2 pools, 28→14→7 spatially, with 64 channels
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))  # (batch, 32, 28, 28)
        x = self.pool1(x)          # (batch, 32, 14, 14)

        x = F.relu(self.conv2(x))  # (batch, 64, 14, 14)
        x = self.pool2(x)          # (batch, 64, 7, 7)

        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

### 학습 코드

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Load data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Model, loss, optimizer
model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(5):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 6. 특징 맵 시각화

```python
def visualize_feature_maps(model, image):
    """Visualize feature maps from the first Conv layer"""
    model.eval()
    with torch.no_grad():
        # First Conv output
        x = model.conv1(image)
        x = F.relu(x)

    # Display in grid
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < x.shape[1]:
            ax.imshow(x[0, i].cpu().numpy(), cmap='viridis')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('feature_maps.png')
```

---

## 7. NumPy로 합성곱 이해 (참고)

```python
def conv2d_numpy(image, kernel):
    """2D convolution implementation with NumPy (educational)"""
    h, w = image.shape
    kh, kw = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1

    output = np.zeros((oh, ow))

    for i in range(oh):
        for j in range(ow):
            # Extract region
            region = image[i:i+kh, j:j+kw]
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)

    return output

# Sobel edge detection example
# Why these specific values?  The Sobel-x kernel computes a weighted
# horizontal difference: the right column is positive (+1, +2, +1) and
# the left column is negative (-1, -2, -1).  When convolved with a
# region that transitions from dark (left) to bright (right), the
# positives dominate → large positive output.  In a uniform region,
# left and right cancel → output ≈ 0.  The center row has double
# weight (±2) to emphasize the pixel directly adjacent to the edge.
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

edges = conv2d_numpy(image, sobel_x)
```

> **참고**: 실제 CNN에서는 PyTorch의 최적화된 구현을 사용합니다. 핵심 통찰은 학습된 CNN에서 네트워크가 역전파를 통해 필터 값을 *학습*한다는 것입니다 — Sobel의 수작업 가중치가 에지를 감지하는 것처럼, 학습된 필터는 해당 태스크에 가장 유용한 패턴을 자동으로 발견합니다.

---

## 8. 배치 정규화와 Dropout

### CNN에서 사용

```python
class CNNWithBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BN for Conv
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)  # 2D Dropout

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.bn_fc = nn.BatchNorm1d(128)  # BN for FC
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(-1, 32 * 14 * 14)
        x = self.fc1(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x
```

---

## 9. CIFAR-10 분류

### 데이터

- 32×32 RGB 이미지
- 10개 클래스: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### 모델

```python
class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Two consecutive 3×3 convs give a 5×5 effective receptive field
            # with fewer parameters: 2×(3×3)=18 vs 1×(5×5)=25 weights per channel
            nn.Conv2d(3, 64, 3, padding=1),   # padding=1: same padding preserves 32×32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32→16: halves spatial dims, doubles receptive field

            # Channel doubling (64→128): as spatial resolution decreases,
            # we increase channels to maintain representational capacity
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16→8
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),     # 50% dropout: aggressive regularization before FC
                                 # layers which have the most parameters
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        return x
```

---

## 10. 정리

### 핵심 개념

1. **합성곱**: 지역 패턴 추출, 파라미터 공유
2. **풀링**: 공간 축소, 불변성 증가
3. **채널**: 다양한 특징 학습
4. **계층적 학습**: 저수준 → 고수준 특징

### CNN vs MLP

| 항목 | MLP | CNN |
|------|-----|-----|
| 연결 | 완전 연결 | 지역 연결 |
| 파라미터 | 많음 | 적음 (공유) |
| 공간 정보 | 무시 | 보존 |
| 이미지 | 비효율적 | 효율적 |

**매개변수 공유가 중요한 이유**: 224×224×3 이미지를 평탄화한 벡터로 처리하는 MLP는 첫 번째 층의 *뉴런당* 224×224×3 = 150,528개의 가중치가 필요합니다. 3×3 합성곱 필터는 단 3×3×3 = 27개의 가중치만 사용하고 전체 이미지를 슬라이딩합니다 — 동일한 27개의 가중치가 위치에 관계없이 같은 패턴을 감지합니다. 이것이 CNN을 매개변수 효율적이고 이동 불변(Translation Invariant)하게 만듭니다.

### 다음 단계

[CNN 심화 - 유명 아키텍처](./08_CNN_Advanced.md)에서 ResNet, VGG 등 유명 아키텍처를 학습합니다.

---

## 연습 문제

### 연습 1: 출력 차원 계산

코드를 실행하지 않고 각 시나리오의 출력 공간 차원을 계산하세요.

1. 입력: 28×28, Conv2d(kernel=5, stride=1, padding=0). 출력 크기는?
2. 입력: 64×64, Conv2d(kernel=3, stride=2, padding=1). 출력 크기는?
3. 입력: 32×32, Conv2d(kernel=3, stride=1, padding=1) 세 번 연속 후 MaxPool2d(2,2). 최종 크기는?
4. `torch.randn(1, C, H, W)`을 해당 레이어에 통과시켜 각 답을 검증하세요.

### 연습 2: NumPy로 2D 합성곱(Convolution) 구현

NumPy로 합성곱을 직접 구현하여 이미지에 적용하세요.

1. 수업의 `conv2d_numpy` 함수를 사용하세요.
2. 원하는 값으로 5×5 테스트 이미지를 만드세요.
3. Sobel-x 필터와 Sobel-y 필터(`[[-1,-2,-1],[0,0,0],[1,2,1]]`)를 적용하세요.
4. matplotlib으로 원본 이미지와 두 에지 필터 결과를 시각화하세요.
5. 각 필터가 감지하는 패턴을 설명하세요.

### 연습 3: MNISTNet 학습 및 특성 맵(Feature Map) 시각화

`MNISTNet` 모델을 학습하고 첫 번째 합성곱 층이 학습한 내용을 확인하세요.

1. MNIST에서 `MNISTNet`을 3 에포크 학습하세요.
2. 학습 후 테스트 이미지 하나를 선택하고, 포워드 훅(Forward Hook)을 사용하여 `conv1`의 출력(ReLU 전)을 추출하세요.
3. 32개 특성 맵 전체를 격자로 시각화하세요.
4. 수평 에지, 수직 에지 또는 다른 패턴에 강하게 반응하는 필터를 찾아보세요.

### 연습 4: CIFAR-10에서 배치 정규화(Batch Normalization) 유무 비교

배치 정규화가 학습 안정성에 미치는 영향을 비교하세요.

1. `CIFAR10Net`의 두 버전을 정의하세요: 각 합성곱 레이어 뒤에 `nn.BatchNorm2d`를 추가한 버전과 추가하지 않은 버전.
2. 동일한 하이퍼파라미터로 두 모델을 15 에포크 학습하세요.
3. 학습 손실과 테스트 정확도 곡선을 나란히 그리세요.
4. 배치 정규화가 학습 수렴을 더 빠르고 안정적으로 만드는 이유를 2~3문장으로 설명하세요.

### 연습 5: 파라미터 수 분석

CNN과 MLP의 파라미터 수를 분석적으로 도출하고 검증하세요.

1. `MNISTNet`의 파라미터를 수동으로 계산하세요 (합성곱 레이어와 FC 레이어 각각).
2. 동일한 입력(28×28=784)과 출력(10), 크기 128의 은닉층 하나를 가진 완전 연결 네트워크의 파라미터 수를 계산하세요.
3. `sum(p.numel() for p in model.parameters())`로 두 수치를 검증하세요.
4. 이미지 태스크에서 CNN이 MLP보다 훨씬 적은 파라미터를 사용하는 이유를 자신의 말로 설명하세요.
