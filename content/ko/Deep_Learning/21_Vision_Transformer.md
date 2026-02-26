# 19. Vision Transformer (ViT)

## 학습 목표

- Vision Transformer 아키텍처 이해
- Patch Embedding 원리
- CLS 토큰과 Position Embedding
- ViT 변형 모델들 (DeiT, Swin Transformer)
- PyTorch 구현 및 활용

---

## 1. Vision Transformer 개요

### 핵심 아이디어

```
기존 CNN: 지역적 특징 → 전역 특징 (계층적)
ViT: 이미지를 패치 시퀀스로 변환 → Transformer로 처리

이미지 (224×224) → 16×16 패치 196개 → Transformer Encoder
```

### 왜 Transformer를 Vision에?

```
1. Self-Attention의 장점
   - 장거리 의존성 포착
   - 전역적 컨텍스트 고려

2. 확장성
   - 대규모 데이터셋에서 CNN 능가
   - 스케일링이 용이

3. 아키텍처 통합
   - Vision + Language 통합 가능
   - 멀티모달 학습에 유리
```

---

## 2. ViT 아키텍처

### 전체 구조

```
입력 이미지 (224×224×3)
        ↓
[Patch Embedding] → 196개 패치 벡터 (각 768차원)
        ↓
[CLS Token 추가] → 197개 토큰
        ↓
[Position Embedding 추가]
        ↓
[Transformer Encoder × L layers]
        ↓
[CLS Token 출력 추출]
        ↓
[MLP Head] → 분류 결과
```

### 수식 정리

```
# 입력
x ∈ R^(H×W×C)  # 예: 224×224×3

# 패치 분할
P = patch_size  # 예: 16
N = (H/P) × (W/P)  # 패치 개수: 196

# Patch Embedding
x_p ∈ R^(N×(P²·C))  # 196×768 (16×16×3 = 768)
z_0 = [x_class; x_p·E] + E_pos  # E: 투영 행렬

# Transformer
z_l = MSA(LN(z_{l-1})) + z_{l-1}  # Multi-Head Self-Attention
z_l = MLP(LN(z_l)) + z_l         # Feed Forward

# 출력
y = LN(z_L^0)  # CLS 토큰의 최종 표현
```

---

## 3. Patch Embedding

### 개념

```python
# 이미지를 패치로 분할
# (B, 3, 224, 224) → (B, 196, 768)

# 방법 1: reshape
patches = image.reshape(B, N, P*P*C)  # 직접 재구성

# 방법 2: Conv2d (더 효율적)
# stride=kernel_size로 겹치지 않는 패치 추출
conv = nn.Conv2d(3, 768, kernel_size=16, stride=16)
patches = conv(image)  # (B, 768, 14, 14)
patches = patches.flatten(2).transpose(1, 2)  # (B, 196, 768)
```

### PyTorch 구현

```python
class PatchEmbedding(nn.Module):
    """Patch Embedding Layer (⭐⭐)"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Conv2d로 패치 추출 + 임베딩
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)        # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)   # (B, num_patches, embed_dim)
        return x
```

---

## 4. CLS Token과 Position Embedding

### CLS Token

```python
# BERT에서 차용한 개념
# 전체 이미지의 표현을 학습하는 특별 토큰

class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
# 배치에 브로드캐스트
cls_tokens = class_token.expand(batch_size, -1, -1)  # (B, 1, D)
# 패치 임베딩 앞에 연결
x = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (B, N+1, D)
```

### Position Embedding

```python
# 패치의 위치 정보 제공 (Transformer는 위치 정보 없음)

class PositionEmbedding(nn.Module):
    """Learnable Position Embedding (⭐⭐)"""
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        # +1 for CLS token
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )

    def forward(self, x):
        return x + self.pos_embedding
```

### 위치 임베딩 시각화

```python
def visualize_position_embedding(pos_embed, img_size=224, patch_size=16):
    """위치 임베딩 유사도 시각화 (⭐⭐)"""
    # pos_embed: (1, N+1, D)
    # CLS 토큰 제외
    pos_embed = pos_embed[0, 1:]  # (N, D)

    # 유사도 행렬
    similarity = torch.mm(pos_embed, pos_embed.T)  # (N, N)

    # 특정 패치와의 유사도
    num_patches = (img_size // patch_size)
    center_idx = num_patches * (num_patches // 2) + (num_patches // 2)
    center_sim = similarity[center_idx].reshape(num_patches, num_patches)

    return center_sim  # 중앙 패치와의 유사도 맵
```

---

## 5. Vision Transformer 전체 구현

### 기본 ViT

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention (⭐⭐⭐)"""
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape

        # QKV 계산
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # Output
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP Block (⭐⭐)"""
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block (⭐⭐⭐)"""
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) (⭐⭐⭐⭐)"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0
    ):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim
        )

        # CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position Embedding
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classification Head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch Embedding
        x = self.patch_embed(x)  # (B, N, D)

        # Add CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # Add Position Embedding
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # CLS Token만 추출하여 분류
        cls_output = x[:, 0]
        return self.head(cls_output)
```

### ViT 모델 변형

```python
# ViT-Base (ViT-B/16)
vit_base = VisionTransformer(
    img_size=224, patch_size=16,
    embed_dim=768, depth=12, num_heads=12
)

# ViT-Large (ViT-L/16)
vit_large = VisionTransformer(
    img_size=224, patch_size=16,
    embed_dim=1024, depth=24, num_heads=16
)

# ViT-Huge (ViT-H/14)
vit_huge = VisionTransformer(
    img_size=224, patch_size=14,
    embed_dim=1280, depth=32, num_heads=16
)
```

---

## 6. DeiT (Data-efficient Image Transformer)

### 핵심 개선점

```
문제: ViT는 대규모 데이터 필요 (JFT-300M 등)
해결: 지식 증류 + 강력한 데이터 증강으로 ImageNet만으로 학습

1. Distillation Token: CNN 교사 모델의 지식 학습
2. 강력한 Data Augmentation
3. Regularization (Stochastic Depth, Dropout)
```

### Distillation Token

```python
class DeiT(nn.Module):
    """Data-efficient Image Transformer (⭐⭐⭐⭐)"""
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        # CLS Token + Distillation Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Position Embedding (+2 for CLS and DIST)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 2, embed_dim)
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 두 개의 Head
        self.head = nn.Linear(embed_dim, num_classes)
        self.head_dist = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_tokens = self.dist_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, dist_tokens, x], dim=1)

        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # CLS와 DIST 토큰 모두 사용
        cls_output = self.head(x[:, 0])
        dist_output = self.head_dist(x[:, 1])

        if self.training:
            return cls_output, dist_output
        else:
            # 추론 시 평균
            return (cls_output + dist_output) / 2
```

### DeiT 학습

```python
def train_deit_with_distillation(student, teacher, dataloader, epochs=100):
    """DeiT 지식 증류 학습 (⭐⭐⭐)"""
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)
    criterion_ce = nn.CrossEntropyLoss()
    criterion_dist = nn.CrossEntropyLoss()

    teacher.eval()

    for epoch in range(epochs):
        for images, labels in dataloader:
            # Teacher prediction (soft labels)
            with torch.no_grad():
                teacher_output = teacher(images)

            # Student predictions
            cls_output, dist_output = student(images)

            # Losses
            loss_cls = criterion_ce(cls_output, labels)
            loss_dist = criterion_dist(dist_output, teacher_output.argmax(dim=1))

            loss = 0.5 * loss_cls + 0.5 * loss_dist

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 7. Swin Transformer

### 핵심 아이디어

```
문제: ViT의 O(n²) 복잡도 → 고해상도 이미지 처리 어려움
해결: 계층적 구조 + Shifted Window Attention

특징:
1. Window Attention: 지역 윈도우 내에서만 attention
2. Shifted Windows: 윈도우 간 정보 교환
3. 계층적 구조: 특징 맵 해상도 점진적 감소
```

### Window Attention

```python
def window_partition(x, window_size):
    """이미지를 윈도우로 분할 (⭐⭐⭐)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """윈도우를 다시 이미지로 합침 (⭐⭐⭐)"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention (⭐⭐⭐⭐)"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # 상대 위치 인덱스 생성
        self._create_relative_position_index()

    def _create_relative_position_index(self):
        coords = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords, coords], indexing='ij'))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn + mask

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x
```

### Shifted Window

```python
class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block with (Shifted) Window Attention (⭐⭐⭐⭐)"""
    def __init__(self, dim, num_heads, window_size=7, shift_size=0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, x, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows)

        # Window reverse
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x
```

---

## 8. 사전학습 모델 활용

### torchvision 사용

```python
from torchvision.models import vit_b_16, vit_l_16, swin_t, swin_s

# ViT-B/16 (pretrained)
model = vit_b_16(weights='IMAGENET1K_V1')

# 특징 추출기로 사용
model.heads = nn.Identity()
features = model(image)  # (B, 768)

# Fine-tuning
model = vit_b_16(weights='IMAGENET1K_V1')
model.heads = nn.Linear(768, num_classes)

# 학습률 차등 적용
params = [
    {'params': model.encoder.parameters(), 'lr': 1e-5},  # backbone
    {'params': model.heads.parameters(), 'lr': 1e-3}     # head
]
optimizer = torch.optim.AdamW(params)
```

### timm 라이브러리

```python
import timm

# 사용 가능한 ViT 모델 목록
vit_models = timm.list_models('vit*', pretrained=True)
print(f"Available ViT models: {len(vit_models)}")

# 모델 로드
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# 커스텀 분류 헤드
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=10  # 자동으로 head 교체
)

# DeiT 모델
deit_model = timm.create_model('deit_base_patch16_224', pretrained=True)

# Swin Transformer
swin_model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
```

---

## 9. 실전 Fine-tuning

### CIFAR-10 Fine-tuning

```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm

def finetune_vit_cifar10(epochs=10):
    """ViT CIFAR-10 Fine-tuning (⭐⭐⭐)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 데이터 전처리 (ViT 입력 크기에 맞게)
    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # 데이터셋
    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10('data', train=False, transform=transform_test)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4)

    # 모델
    model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=10)
    model = model.to(device)

    # 옵티마이저 (차등 학습률)
    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
    head_params = [p for n, p in model.named_parameters() if 'head' in n]

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},
        {'params': head_params, 'lr': 1e-3}
    ], weight_decay=0.01)

    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 학습
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 평가
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={accuracy:.2f}%')

        scheduler.step()

    return model
```

---

## 10. ViT vs CNN 비교

### 특성 비교

| 특성 | CNN | ViT |
|-----|-----|-----|
| 귀납적 편향 | 지역성, 등변성 | 없음 |
| 데이터 요구량 | 적음 | 많음 |
| 계산 복잡도 | O(n) | O(n²) |
| 장거리 의존성 | 어려움 | 용이 |
| 해석 가능성 | 필터 시각화 | Attention 시각화 |

### 사용 가이드라인

```
CNN 선호:
- 소규모 데이터셋
- 제한된 계산 리소스
- 실시간 추론 필요

ViT 선호:
- 대규모 데이터셋 또는 사전학습 모델 활용
- 전역 컨텍스트가 중요한 태스크
- 멀티모달 학습 계획
```

---

## 정리

### 핵심 개념

1. **Patch Embedding**: 이미지를 패치 시퀀스로 변환
2. **CLS Token**: 전체 이미지 표현 학습
3. **Position Embedding**: 패치 위치 정보 제공
4. **DeiT**: 데이터 효율적 학습 (지식 증류)
5. **Swin**: 윈도우 기반 효율적 attention

### 모델 선택 가이드

```
일반 분류: ViT-B/16 또는 DeiT
고해상도: Swin Transformer
제한된 자원: ViT-Small, DeiT-Tiny
최고 성능: ViT-Large, Swin-Large
```

### PyTorch 실전 팁

```python
# 1. timm 사용 권장
import timm
model = timm.create_model('vit_base_patch16_224', pretrained=True)

# 2. 차등 학습률 필수
optimizer = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 1e-5},
    {'params': head_params, 'lr': 1e-3}
])

# 3. 입력 크기 주의 (224, 384, 등)

# 4. 강력한 데이터 증강 사용
```

---

## 연습 문제

### 연습 1: 패치 임베딩(Patch Embedding) 형태 검증

패치 임베딩이 예상된 텐서 형태를 생성하는지 검증하세요.

1. `PatchEmbedding(img_size=224, patch_size=16, in_channels=3, embed_dim=768)`을 인스턴스화하세요.
2. 난수 입력 `torch.randn(2, 3, 224, 224)`를 전달하고 출력 형태를 출력하세요.
3. 출력이 `(2, 196, 768)`임을 확인하세요. `(224 // 16) ** 2`로 196을 직접 계산하세요.
4. `patch_size=32`와 `patch_size=8`로 반복하세요. 각 경우에 `num_patches`는 얼마인가요?
5. 트레이드오프를 설명하세요: 더 작은 패치는 더 많은 토큰을 생성하지만(해상도 향상) O(n²) 어텐션 비용이 증가합니다.

### 연습 2: CLS 토큰의 역할 검증

CLS 토큰이 모든 패치로부터 전역 정보를 집계하는지 검증하세요.

1. 소형 ViT (depth=2, `d_model=64`, `nhead=4`)를 구성하고 난수 이미지 배치를 전달하세요.
2. 인코더 이후 CLS 토큰 출력 `x[:, 0]`과 각 패치 토큰 출력 `x[:, 1:]`을 추출하세요.
3. CLS 토큰과 각 패치 토큰 사이의 코사인 유사도(Cosine Similarity)를 계산하세요.
4. 패치 유사도를 `(H/P, W/P)` 격자로 재형성하여 유사도 맵을 시각화하세요.
5. 모든 패치 토큰의 평균 풀링 대신 CLS 토큰이 분류 표현으로 사용되는 이유를 설명하세요.

### 연습 3: CIFAR-10에서 ViT 미세 조정(Fine-tuning)

차별적 학습률을 사용하여 CIFAR-10에서 사전 학습된 ViT를 미세 조정하세요.

1. `timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=10)`을 불러오세요.
2. 백본(Backbone) 학습률은 `1e-5`, 헤드(Head) 학습률은 `1e-3`으로 설정하세요.
3. 올바른 ImageNet 정규화를 적용하고 CIFAR-10 이미지를 224×224로 리사이즈하세요.
4. `CosineAnnealingLR`로 10 에포크 학습하고 테스트 정확도를 보고하세요.
5. 동일한 아키텍처(무작위 초기화)로 처음부터 10 에포크 학습한 결과와 비교하세요.

### 연습 4: 윈도우 어텐션(Window Attention) 복잡도 분석

Swin의 윈도우 어텐션이 전체 어텐션 대비 계산 이점을 정량화하세요.

1. `d_model=96`의 H=56, W=56 입력 특성 맵에 대해:
   - 전체 자기 어텐션의 연산 횟수를 계산하세요: O(N²), N = H × W.
   - `window_size=7` 윈도우 어텐션의 연산 횟수를 계산하세요: 각 윈도우는 49개 토큰이고, 윈도우가 `(56/7)² = 64`개입니다.
2. `전체_연산수 / 윈도우_연산수` 비율로 표현하세요.
3. 일반적인 공식을 도출하세요: 이미지 크기 H×W와 윈도우 크기 W_s에 대한 복잡도 비율.
4. 윈도우 어텐션이 전체 어텐션보다 100배 이상 저렴해지는 이미지 크기는 얼마인가요?

### 연습 5: DeiT 지식 증류(Knowledge Distillation) 학습

CNN 교사(Teacher) 모델로 DeiT 학생(Student) 모델을 학습시키세요.

1. 사전 학습된 ResNet-18을 교사 모델로 불러오세요.
2. 소형 DeiT 스타일 모델(depth=4, `d_model=192`)을 학생 모델로 구성하세요.
3. 결합 손실을 구현하세요: `0.5 * CrossEntropy(cls_output, labels) + 0.5 * CrossEntropy(dist_output, teacher_argmax)`.
4. CIFAR-10에서 20 에포크 학습하고 검증 정확도를 기록하세요.
5. 증류 토큰 없이 하드 레이블만으로 학습한 동일 학생 모델과 비교하고, 관찰된 정확도 차이를 설명하세요.

---

## 참고 자료

- ViT 원본: https://arxiv.org/abs/2010.11929
- DeiT: https://arxiv.org/abs/2012.12877
- Swin Transformer: https://arxiv.org/abs/2103.14030
- timm 라이브러리: https://github.com/huggingface/pytorch-image-models
