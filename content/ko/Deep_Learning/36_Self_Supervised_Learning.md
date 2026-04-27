[이전: CLIP (Contrastive Language-Image Pre-training)](./35_Impl_CLIP.md) | [다음: 현대 딥러닝 아키텍처](./37_Modern_Architectures.md)

---

# 36. Self-Supervised Learning

## 학습 목표

- Self-Supervised Learning 개념과 필요성 이해
- Contrastive Learning (SimCLR, MoCo, BYOL)
- Masked Image Modeling (MAE)
- 사전학습 표현의 전이 학습
- PyTorch 구현 및 실습

## 1. Self-Supervised Learning 개요

### 이론: 왜 SSL

레이블 데이터는 비싸고; 비레이블 데이터는 본질적으로 무료. SSL은 *라벨이 필요 없는* pretext 작업으로 학습하여 이를 활용:

- **단어 수준**: 마스킹된 토큰 예측(MLM, BERT에서 사용) 또는 다음 토큰 예측(causal LM, GPT에서 사용).
- **이미지 수준**: 같은 이미지의 두 증강 뷰가 같은 인스턴스인지 예측(contrastive); 마스킹된 패치 재구성(MAE); 직소 퍼즐, 채색 등.

희망은 pretext 작업 풀이가 네트워크가 다운스트림 작업으로 *전이*되는 특징을 학습하도록 강제하는 것. 실험적으로 이는 놀랍도록 잘 작동 — ImageNet의 현대 SSL 표현이 다운스트림 전이에 대해 지도 사전학습을 일치시키거나 능가.


### 정의와 필요성

```
정의: 레이블 없이 데이터 자체에서 학습 신호 생성

왜 필요한가?
1. 레이블링 비용: ImageNet (1400만장 레이블링) 비용 막대
2. 풍부한 비레이블 데이터: 인터넷의 대부분 데이터
3. 일반화 능력: 다양한 다운스트림 태스크에 전이

패러다임 변화:
지도학습: 데이터 + 레이블 → 모델
자기지도학습: 데이터 → Pretext Task → 표현 → 다운스트림 태스크
```

### SSL의 종류

```
┌─────────────────────────────────────────────────────────────┐
│              Self-Supervised Learning Methods               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Contrastive Learning          Masked Modeling               │
│  ├── SimCLR                   ├── MAE (Vision)               │
│  ├── MoCo                     ├── BERT (NLP)                 │
│  ├── BYOL                     └── BEiT                       │
│  └── SimSiam                                                 │
│                                                              │
│  Clustering                    Generative                    │
│  ├── DeepCluster              ├── VAE                        │
│  └── SwAV                     └── GAN                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Contrastive Learning 기초

### 이론: Contrastive 방법: SimCLR, MoCo

SimCLR (Chen et al. 2020)와 MoCo (He et al. 2019) 둘 다 증강 뷰와 InfoNCE 손실을 사용:

```
- 각 이미지에 두 무작위 증강 적용: x_a = aug(x), x_b = aug(x).
- 둘 다 인코딩: z_a = f(x_a), z_b = f(x_b).
- (z_a, z_b)를 양성 쌍으로 다룸; 배치 내 모든 다른 예제를 음성으로.
- InfoNCE가 (z_a, z_b)를 함께 끌고, 다른 모두를 멀리 밂.
```

SimCLR은 배치 내 음성만 사용 — 충분한 음성을 위해 거대한 배치(8192) 필요. MoCo는 **메모리 뱅크**(최근 특징의 큐)를 유지하고 키에 천천히 움직이는 "momentum encoder"를 사용해 음성을 배치에서 분리, 거대한 배치 크기 없이 훨씬 많은 음성 허용.

핵심 귀납 편향: 증강은 이미지의 *정체성*은 보존하지만 표면 디테일은 변경 — 색, 크롭, 조명. 따라서 contrastive 손실은 암묵적으로 "이 표면 특징은 정체성에 중요하지 않아야"라고 말함. 증강 선택이 결정적(SimCLR 레시피: 무작위 크롭 + 색 지터링 + 가우시안 블러).


### 핵심 아이디어

```
같은 이미지의 다른 augmentation → 가깝게 (Positive)
다른 이미지 → 멀게 (Negative)

x ──┬── Aug1 → view1 ──┐
    │                   │ → 유사도 최대화 (positive pair)
    └── Aug2 → view2 ──┘

x1 ── Aug → view1 ─┐
                    │ → 유사도 최소화 (negative pair)
x2 ── Aug → view2 ─┘
```

### InfoNCE Loss

```python
import torch
import torch.nn.functional as F

def info_nce_loss(features, temperature=0.5):
    """InfoNCE (NT-Xent) Loss (⭐⭐⭐)

    features: (2N, D) - N개 이미지의 두 augmentation

    Returns:
        loss: contrastive loss
    """
    batch_size = features.shape[0] // 2

    # Normalize
    features = F.normalize(features, dim=1)

    # Similarity matrix
    similarity_matrix = features @ features.T / temperature

    # Mask diagonal and same-image pairs
    labels = torch.cat([torch.arange(batch_size) for _ in range(2)])
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # Mask self-similarity (diagonal)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # Positives: same image, different augmentation
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # Negatives: different images
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    # Logits
    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)

    return F.cross_entropy(logits, labels.to(logits.device))
```

---

## 3. SimCLR (Simple Framework for Contrastive Learning)

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     SimCLR Framework                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│     Image x                                                  │
│        │                                                     │
│        ▼                                                     │
│  ┌─────────────┐                                            │
│  │ Data Aug    │ → 두 개의 view 생성 (t, t')                │
│  └─────────────┘                                            │
│      │     │                                                 │
│     t(x)  t'(x)                                             │
│      │     │                                                 │
│      ▼     ▼                                                 │
│  ┌─────────────┐                                            │
│  │  Encoder f  │ → 특징 추출 (ResNet 등)                    │
│  └─────────────┘                                            │
│      │     │                                                 │
│     h_i   h_j                                                │
│      │     │                                                 │
│      ▼     ▼                                                 │
│  ┌─────────────┐                                            │
│  │Projection g │ → MLP로 저차원 임베딩                      │
│  └─────────────┘                                            │
│      │     │                                                 │
│     z_i   z_j                                                │
│      │     │                                                 │
│      └──┬──┘                                                │
│         ▼                                                    │
│    NT-Xent Loss                                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch 구현

```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50

class SimCLR(nn.Module):
    """SimCLR Model (⭐⭐⭐⭐)"""
    def __init__(self, base_encoder=resnet50, projection_dim=128):
        super().__init__()

        # Encoder (pretrained ResNet without FC)
        self.encoder = base_encoder(weights=None)
        self.encoder_dim = self.encoder.fc.in_features
        self.encoder.fc = nn.Identity()

        # Projection Head (MLP)
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.ReLU(),
            nn.Linear(self.encoder_dim, projection_dim)
        )

    def forward(self, x):
        h = self.encoder(x)  # 특징 벡터
        z = self.projection_head(h)  # 투영 벡터
        return h, z

    def get_features(self, x):
        """다운스트림 태스크용 특징 추출"""
        return self.encoder(x)


class SimCLRAugmentation:
    """SimCLR Data Augmentation (⭐⭐⭐)"""
    def __init__(self, size=224):
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=int(0.1 * size) // 2 * 2 + 1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
```

### SimCLR 학습

```python
def train_simclr(model, train_loader, epochs=100, lr=0.3, temperature=0.5):
    """SimCLR Training Loop (⭐⭐⭐)"""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0

        for (x_i, x_j), _ in train_loader:
            x_i, x_j = x_i.to(device), x_j.to(device)

            # Forward
            _, z_i = model(x_i)
            _, z_j = model(x_j)

            # Concatenate
            z = torch.cat([z_i, z_j], dim=0)

            # Loss
            loss = info_nce_loss(z, temperature)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss/len(train_loader):.4f}")

    return model
```

---

## 4. MoCo (Momentum Contrast)

### 핵심 아이디어

```
SimCLR 문제: 큰 배치 사이즈 필요 (4096+)
MoCo 해결: Momentum Encoder + Queue로 많은 negative 확보

특징:
1. Queue: 이전 배치의 임베딩 저장 (65536개)
2. Momentum Encoder: 천천히 업데이트되는 인코더
```

### MoCo 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     MoCo Framework                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Query                          Key                         │
│     │                              │                         │
│     ▼                              ▼                         │
│ ┌─────────┐                  ┌─────────┐                    │
│ │Encoder f│                  │Encoder  │                    │
│ │  (q)    │                  │f_k (mom)│ ← momentum update  │
│ └────┬────┘                  └────┬────┘                    │
│      │                            │                          │
│      ▼                            ▼                          │
│     q_i                         k_i ────────────────┐       │
│      │                            │                 │        │
│      │                            ▼                 ▼        │
│      │                       ┌─────────┐     ┌──────────┐   │
│      │                       │  Queue  │ ←── │ enqueue  │   │
│      │                       │ (k-)    │     └──────────┘   │
│      │                       └────┬────┘                    │
│      │                            │                          │
│      └────────────┬───────────────┘                         │
│                   ▼                                          │
│            InfoNCE Loss                                      │
│        (q·k+ / q·k-)                                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch 구현

```python
import copy

class MoCo(nn.Module):
    """Momentum Contrast (MoCo v2) (⭐⭐⭐⭐)"""
    def __init__(self, base_encoder=resnet50, dim=128, K=65536, m=0.999, T=0.07):
        super().__init__()
        self.K = K  # Queue size
        self.m = m  # Momentum
        self.T = T  # Temperature

        # Query encoder
        self.encoder_q = base_encoder(weights=None)
        self.encoder_q.fc = nn.Sequential(
            nn.Linear(self.encoder_q.fc.in_features, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim)
        )

        # Key encoder (momentum)
        self.encoder_k = copy.deepcopy(self.encoder_q)
        for param_k in self.encoder_k.parameters():
            param_k.requires_grad = False  # 역전파 X

        # Queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Key encoder의 momentum 업데이트"""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Queue 업데이트"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace oldest
        if ptr + batch_size > self.K:
            # Wrap around
            self.queue[:, ptr:] = keys[:self.K - ptr].T
            self.queue[:, :batch_size - (self.K - ptr)] = keys[self.K - ptr:].T
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """
        Args:
            im_q: query image
            im_k: key image (different augmentation)
        """
        # Query
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # Key (no gradient)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)
            k = F.normalize(k, dim=1)

        # Positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)

        # Negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T

        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


def train_moco(model, train_loader, epochs=200):
    """MoCo Training (⭐⭐⭐)"""
    optimizer = torch.optim.SGD(model.encoder_q.parameters(),
                                lr=0.03, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for (im_q, im_k), _ in train_loader:
            logits, labels = model(im_q.cuda(), im_k.cuda())
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 5. BYOL (Bootstrap Your Own Latent)

### 이론: 비-Contrastive: BYOL, SimSiam, 그리고 붕괴

BYOL (Grill et al. 2020)과 SimSiam (Chen & He 2021)은 음성을 완전히 떨어뜨립니다. 양성 쌍만 가지고 **표현 붕괴**(encoder가 모든 것을 같은 벡터로 매핑하는 사소한 해)를 막는 구조적 트릭을 가짐.

BYOL: 두 네트워크(online + target). Online이 target의 출력을 예측; target은 online의 EMA. 비대칭 + EMA가 붕괴 방지.

SimSiam: 더 단순 — 두 뷰, 하지만 예측 측 네트워크가 target 측에 stop-gradient. Stop-gradient 없이는 네트워크가 상수 출력으로 붕괴; 그것과 함께면 학습이 안정.

여기서 붕괴가 일어나지 않는 이유는 다소 실험적 — 이론적 논증(Tian et al. 2021)은 비대칭 + stop-gradient가 명시적 음성 없이도 contrastive 힘 같은 것을 암묵적으로 만든다는 것을 보임.


### 핵심 아이디어

```
문제: Negative samples이 정말 필요한가?
BYOL: Negative 없이 학습! (Online + Target network)

핵심: Predictor 네트워크 + EMA로 collapse 방지
```

### BYOL 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     BYOL Framework                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Online Network                 Target Network              │
│   (학습됨)                       (EMA로 업데이트)            │
│                                                              │
│  ┌─────────┐                    ┌─────────┐                 │
│  │Encoder  │                    │Encoder  │                 │
│  │(θ)      │                    │(ξ)      │ ← EMA           │
│  └────┬────┘                    └────┬────┘                 │
│       │                              │                       │
│       ▼                              ▼                       │
│  ┌─────────┐                    ┌─────────┐                 │
│  │Projector│                    │Projector│                 │
│  └────┬────┘                    └────┬────┘                 │
│       │                              │                       │
│       ▼                              │                       │
│  ┌─────────┐                         │                       │
│  │Predictor│ (Online만)              │                       │
│  └────┬────┘                         │                       │
│       │                              │                       │
│      q_θ                           z_ξ                       │
│       │                              │                       │
│       └──────────┬───────────────────┘                      │
│                  ▼                                           │
│         MSE Loss (q_θ, sg(z_ξ))                             │
│         sg = stop gradient                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch 구현

```python
class BYOL(nn.Module):
    """Bootstrap Your Own Latent (⭐⭐⭐⭐)"""
    def __init__(self, base_encoder=resnet50, hidden_dim=4096, proj_dim=256, pred_dim=256):
        super().__init__()

        # Online network
        encoder = base_encoder(weights=None)
        encoder_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()

        self.online_encoder = encoder
        self.online_projector = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, proj_dim)
        )

        # Target network (EMA)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Freeze target
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target(self, tau=0.99):
        """Target network EMA 업데이트"""
        for online, target in zip(self.online_encoder.parameters(),
                                   self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

        for online, target in zip(self.online_projector.parameters(),
                                   self.target_projector.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

    def forward(self, x1, x2):
        # Online predictions
        online_proj_1 = self.online_projector(self.online_encoder(x1))
        online_proj_2 = self.online_projector(self.online_encoder(x2))

        online_pred_1 = self.predictor(online_proj_1)
        online_pred_2 = self.predictor(online_proj_2)

        # Target projections (no gradient)
        with torch.no_grad():
            target_proj_1 = self.target_projector(self.target_encoder(x1))
            target_proj_2 = self.target_projector(self.target_encoder(x2))

        return online_pred_1, online_pred_2, target_proj_1, target_proj_2


def byol_loss(pred, target):
    """BYOL Loss: Negative Cosine Similarity (⭐⭐⭐)"""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 2 - 2 * (pred * target).sum(dim=-1).mean()


def train_byol(model, train_loader, epochs=100):
    """BYOL Training (⭐⭐⭐)"""
    optimizer = torch.optim.Adam(
        list(model.online_encoder.parameters()) +
        list(model.online_projector.parameters()) +
        list(model.predictor.parameters()),
        lr=3e-4
    )

    for epoch in range(epochs):
        for (x1, x2), _ in train_loader:
            x1, x2 = x1.cuda(), x2.cuda()

            pred1, pred2, target1, target2 = model(x1, x2)

            # 대칭적 손실
            loss = byol_loss(pred1, target2) + byol_loss(pred2, target1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Target 업데이트
            model.update_target(tau=0.99)
```

---

## 6. MAE (Masked Autoencoder)

### 이론: Masked Image Modeling (MAE)

MAE (He et al. 2022)는 BERT의 MLM 아이디어를 이미지로 이식. 이미지를 가져와 (ViT 같이) 패치로 나누고, 패치의 75%를 무작위로 마스킹, encoder에게 보이는 25%를 인코딩하도록 요청, 그 다음 작은 decoder에게 encoder 출력에서 마스킹된 것을 재구성하도록 요청.

```
loss = MSE(reconstructed_patches, original_masked_patches)
```

핵심 설계 포인트:

- **비대칭 encoder/decoder**: encoder는 전체 ViT; decoder는 작음(대부분 작업이 컨텍스트 인코딩이기 때문). Encoder가 보이는 패치만 보고, decoder는 전체 시퀀스를 봄(누락된 것에 마스크 토큰).
- **높은 마스크 비율(75%)**: encoder가 *맥락적* 정보를 사용하도록 강제 — BERT의 MLM의 15% 마스킹보다 훨씬 어려움. 이미지 도메인이 텍스트보다 더 많은 중복성을 가짐; 높은 마스킹이 보상.
- **픽셀 재구성**: 원시 픽셀에 단순 MSE; 화려한 정규화나 패치별 손실 균형 없음.

MAE는 매우 강한 표현을 만듦 — 음성 샘플 메커니즘이 필요 없기 때문에, 훨씬 낮은 계산에서 contrastive 방법과 경쟁력.


### 핵심 아이디어

```
NLP의 BERT 아이디어를 Vision에 적용:
- 이미지의 일부(75%)를 마스킹
- 마스킹된 부분을 복원하도록 학습

장점:
1. 높은 마스킹 비율 → 강한 학습 신호
2. 효율적: 마스킹되지 않은 패치만 인코딩
3. ViT와 자연스럽게 결합
```

### MAE 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     MAE Architecture                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Image → Patches                                            │
│         ↓                                                    │
│   ┌─────────────────────────────────────────────┐           │
│   │ [P1] [P2] [P3] ... [P196]                   │           │
│   └─────────────────────────────────────────────┘           │
│         ↓ Random Masking (75%)                               │
│   ┌─────────────────────────────────────────────┐           │
│   │ [P1] [M] [P3] [M] [M] ... [P196]            │           │
│   └─────────────────────────────────────────────┘           │
│         ↓ (visible patches only)                             │
│   ┌─────────────────────────────────────────────┐           │
│   │ Encoder (ViT) - only on visible patches     │           │
│   │ [P1] [P3] ... [P196]                        │           │
│   └─────────────────────────────────────────────┘           │
│         ↓ + Mask tokens + Position Embedding                 │
│   ┌─────────────────────────────────────────────┐           │
│   │ Decoder (small ViT)                         │           │
│   │ [P1] [M] [P3] [M] [M] ... [P196]            │           │
│   └─────────────────────────────────────────────┘           │
│         ↓                                                    │
│   Reconstruct masked patches (MSE Loss)                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PyTorch 구현

```python
import random

class MAE(nn.Module):
    """Masked Autoencoder (⭐⭐⭐⭐)"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_heads=16,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        num_patches = (img_size // patch_size) ** 2

        # Encoder
        self.patch_embed = nn.Conv2d(in_channels, encoder_embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, encoder_embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, encoder_embed_dim))

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(encoder_embed_dim, encoder_heads, batch_first=True),
            num_layers=encoder_depth
        )
        self.encoder_norm = nn.LayerNorm(encoder_embed_dim)

        # Decoder
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, decoder_embed_dim))

        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(decoder_embed_dim, decoder_heads, batch_first=True),
            num_layers=decoder_depth
        )
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_channels)

    def random_masking(self, x, mask_ratio):
        """Random masking (⭐⭐⭐)"""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))

        # 랜덤 셔플
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first len_keep
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Generate mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # Add position embedding (without cls)
        x = x + self.pos_embed[:, 1:, :]

        # Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)

        # Append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x = torch.cat([cls_tokens, x], dim=1)

        # Encoder
        x = self.encoder(x)
        x = self.encoder_norm(x)

        # Decoder embed
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # Add position embedding
        x = x + self.decoder_pos_embed

        # Decoder
        x = self.decoder(x)
        x = self.decoder_norm(x)

        # Predictor
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # remove cls token

        return x, mask

    def loss(self, pred, target, mask):
        """MAE Loss: MSE on masked patches only (⭐⭐⭐)"""
        target = self.patchify(target)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # mean per patch
        loss = (loss * mask).sum() / mask.sum()  # mean on masked
        return loss

    def patchify(self, imgs):
        """이미지를 패치로 변환"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        x = x.reshape(B, h * w, p * p * C)
        return x
```

### MAE 학습

```python
def train_mae(model, train_loader, epochs=400, lr=1.5e-4):
    """MAE Training (⭐⭐⭐)"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.95), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        total_loss = 0

        for images, _ in train_loader:
            images = images.cuda()

            pred, mask = model(images)
            loss = model.loss(pred, images, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs}: Loss = {total_loss/len(train_loader):.4f}")
```

---

## 7. Linear Evaluation

### 표현 품질 평가

```python
def linear_evaluation(encoder, train_loader, test_loader, num_classes=10):
    """SSL 표현의 Linear Evaluation (⭐⭐⭐)"""
    # Encoder freeze
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Linear classifier
    feature_dim = 2048  # ResNet-50 기준
    classifier = nn.Linear(feature_dim, num_classes).cuda()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9)

    # Extract features
    def extract_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for images, targets in loader:
                feat = encoder(images.cuda())
                features.append(feat.cpu())
                labels.append(targets)
        return torch.cat(features), torch.cat(labels)

    train_features, train_labels = extract_features(train_loader)
    test_features, test_labels = extract_features(test_loader)

    # Train linear classifier
    train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True)

    for epoch in range(100):
        for features, labels in train_loader:
            features, labels = features.cuda(), labels.cuda()
            output = classifier(features)
            loss = F.cross_entropy(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    classifier.eval()
    with torch.no_grad():
        output = classifier(test_features.cuda())
        pred = output.argmax(dim=1).cpu()
        accuracy = (pred == test_labels).float().mean().item()

    return accuracy * 100
```

---

## 8. 방법론 비교

### 특성 비교

| 방법 | Negative | 배치 사이즈 | 주요 특징 |
|------|----------|-------------|----------|
| SimCLR | 필요 | 4096+ | 단순, 강력한 augmentation |
| MoCo | 필요 (Queue) | 256 | 메모리 효율적 |
| BYOL | 불필요 | 256 | Predictor + EMA |
| SimSiam | 불필요 | 256 | BYOL 단순화 (EMA 없음) |
| MAE | 불필요 | 256 | Reconstruction |

### 성능 비교 (ImageNet Linear Probe)

```
SimCLR (ResNet-50, 8192 batch): 69.3%
MoCo v2 (ResNet-50): 71.1%
BYOL (ResNet-50): 74.3%
MAE (ViT-Base): 67.8% → Fine-tune: 83.6%
```

---

## 정리

### 핵심 개념

1. **Contrastive Learning**: Positive/Negative 쌍으로 학습
2. **InfoNCE Loss**: 대조 손실 함수
3. **Momentum Encoder**: 느리게 업데이트되는 target
4. **Masked Modeling**: 일부 입력을 복원하도록 학습
5. **Linear Evaluation**: 고정된 표현 위의 선형 분류기 성능

### 선택 가이드

```
대규모 배치 가능: SimCLR (단순하고 효과적)
제한된 리소스: MoCo (Queue로 효율적)
Negative 없이: BYOL, SimSiam
ViT 기반: MAE (복원 기반)
```

### 실전 팁

```python
# 1. Data Augmentation이 핵심
# - 강한 augmentation = 더 좋은 표현

# 2. 온도 파라미터 주의
# - 너무 낮으면 학습 불안정
# - 너무 높으면 학습 신호 약함

# 3. 긴 학습 필요
# - 최소 200-800 epochs 권장

# 4. Linear eval vs Fine-tune
# - Linear: 표현 품질 평가
# - Fine-tune: 실제 성능 (더 높음)
```

---

## 연습 문제

### 연습 1: 대조 학습(Contrastive Learning) 증강 이해

데이터 증강(data augmentation)은 대조 학습 품질에서 가장 중요한 요소입니다.
1. `SimCLRAugmentation`을 사용하여 같은 CIFAR-10 이미지의 8개의 증강된 뷰를 생성하고 시각화하세요.
2. 어떤 증강(crop, flip, color jitter, grayscale, blur)이 시각적으로 가장 극적인지 파악하세요.
3. 대조 학습에서 색상 왜곡(color jitter)이 특히 중요한 이유를 설명하세요: color jitter가 제거되면 모델이 어떤 시각적 단서를 활용할까요?
4. 증강으로서 `GaussianBlur`의 목적은 무엇인가요? 어떤 불변성(invariance)을 인코더가 학습하도록 장려하나요?

### 연습 2: SimCLR 학습 구현 및 실행

제공된 `SimCLR`과 `SimCLRAugmentation` 코드를 사용하여:
1. `__getitem__`이 `SimCLRAugmentation`으로 `(증강_뷰_1, 증강_뷰_2)`를 반환하도록 CIFAR-10 데이터셋을 래핑하세요.
2. `ResNet-18` 백본으로 `batch_size=256`, `temperature=0.5`로 20 에폭 `SimCLR` 모델을 학습하세요.
3. 학습 후 인코더를 고정하고 그 위에 선형 분류기를 30 에폭 학습하세요 (선형 평가).
4. CIFAR-10 테스트 세트에서 선형 프로브 정확도를 보고하세요. 동일한 아키텍처로 완전 지도 학습한 결과와 어떻게 비교되나요?

### 연습 3: MoCo의 모멘텀 인코더(Momentum Encoder) 분석

모멘텀 인코더는 SimCLR보다 더 작은 배치 크기를 사용할 수 있게 하는 MoCo의 핵심 혁신입니다.
1. `MoCo._momentum_update_key_encoder`에서 업데이트 규칙: `param_k = m * param_k + (1-m) * param_q`. 키 인코더 파라미터가 역전파를 통해 업데이트되지 않는 이유를 자신의 말로 설명하세요.
2. 실험: 소규모 네트워크의 두 복사본을 초기화하세요. 하나는 SGD로, 다른 하나는 모멘텀 업데이트(m=0.999)로 100 스텝 업데이트하세요. 시간에 따른 파라미터 차이의 노름(norm)을 플롯하세요.
3. `m=0.0` (모멘텀 없음)이면 어떻게 되나요? `m=1.0`이면? 실전에서 1에 가까운 값(예: 0.999)이 사용되는 이유는?

### 연습 4: MAE 마스킹 시각화 구현

`MAE.random_masking` 함수를 사용하여:
1. 224×224 이미지를 가져와 크기 16×16의 14×14 = 196 패치로 나누세요.
2. `mask_ratio=0.25, 0.50, 0.75, 0.90`으로 랜덤 마스킹을 적용하고 결과 가시 패치를 시각화하세요 (마스킹된 패치는 회색으로 표시).
3. 원래 MAE 논문은 75% 마스킹을 사용합니다. 이렇게 높은 마스킹 비율이 왜 유익한지 설명하세요: 모델이 단순한 국부 텍스처 패턴을 사용하여 패치를 재구성하는 것을 어떻게 방지하나요?
4. MAE의 복원 기반 접근법과 SimCLR의 대조 학습 접근법을 비교하세요. 어떤 시나리오에서 MAE 스타일 사전학습이 선호되나요? 대조 학습이 선호되는 경우는?

### 연습 5: SSL 방법들 평가 및 비교

사전학습된 모델 가중치(예: torchvision 또는 timm)를 사용하여:
1. SimCLR 사전학습된 ResNet-50과 지도학습으로 학습된 ResNet-50을 불러와 모두 고정하세요.
2. 레이블이 적은 STL-10 데이터셋을 사용하여 각각의 위에 100 에폭 선형 분류기를 학습하세요.
3. 두 접근법의 선형 프로브 정확도를 비교하세요.
4. 이제 10%의 레이블 데이터로 두 모델 모두 전체 미세조정하세요. 어느 접근법이 미세조정을 통해 더 많은 혜택을 받나요? 레이블이 적은 환경에서 자기지도 사전학습이 특히 빛나는 이유는 무엇인가요?

---

## 참고 자료

- SimCLR: https://arxiv.org/abs/2002.05709
- MoCo: https://arxiv.org/abs/1911.05722
- BYOL: https://arxiv.org/abs/2006.07733
- MAE: https://arxiv.org/abs/2111.06377
- SimSiam: https://arxiv.org/abs/2011.10566
