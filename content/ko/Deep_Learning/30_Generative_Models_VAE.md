[이전: 생성적 적대 신경망(GAN)](./29_Impl_GAN.md) | [다음: Variational Autoencoder (VAE)](./31_Impl_VAE.md)

---

# 30. 생성 모델 - VAE (Variational Autoencoder)

## 학습 목표

- VAE의 이론적 기반 (Variational Inference)
- Latent Space와 확률적 생성 이해
- ELBO 손실 함수 유도
- Reparameterization Trick
- Beta-VAE와 Disentanglement
- PyTorch 구현 및 시각화

---

## 1. VAE 이론

### Autoencoder vs VAE

```
Autoencoder:
    입력 → Encoder → 잠재 벡터 z (결정론적) → Decoder → 재구성

VAE:
    입력 → Encoder → 평균 mu, 분산 sigma → 샘플링 z ~ N(mu, sigma) → Decoder → 재구성
                         ↓
                    잠재 공간이 연속적이고 정규 분포를 따름
                    → 새로운 이미지 생성 가능
```

### 왜 확률적인가?

```
일반 Autoencoder의 문제:
- 잠재 공간이 불연속적
- 학습 데이터에 없는 z 입력 시 이상한 출력
- 생성 모델로 사용하기 어려움

VAE의 해결:
- 잠재 공간을 정규 분포로 정규화
- 연속적인 잠재 공간
- 임의의 z ~ N(0, I)에서 샘플링하여 생성 가능
```

### 그래피컬 모델

```
생성 과정 (Generative Process):
    z ~ p(z) = N(0, I)           # 사전 분포
    x ~ p_theta(x|z)             # 디코더

추론 과정 (Inference):
    q_phi(z|x) ≈ p(z|x)          # 인코더가 사후 분포 근사

목표:
    log p(x) 최대화 (데이터의 우도)
    → ELBO (Evidence Lower Bound) 최대화
```

---

## 2. ELBO 손실 함수

### 유도

```
log p(x) = log ∫ p(x, z) dz

         = log ∫ p(x|z) p(z) dz

         = log ∫ q(z|x) * [p(x|z) p(z) / q(z|x)] dz

         ≥ ∫ q(z|x) log[p(x|z) p(z) / q(z|x)] dz    (Jensen's inequality)

         = E_q[log p(x|z)] - KL(q(z|x) || p(z))

         = ELBO (Evidence Lower Bound)
```

### 두 항의 의미

```python
# ELBO = Reconstruction - KL Divergence

# 1. Reconstruction Term: E_q[log p(x|z)]
#    - 디코더가 z로부터 x를 얼마나 잘 복원하는가
#    - 재구성 손실 (MSE 또는 BCE)

# 2. KL Divergence: KL(q(z|x) || p(z))
#    - 인코딩된 분포가 사전 분포(N(0,I))와 얼마나 가까운가
#    - 잠재 공간 정규화
```

### 손실 함수 구현

```python
def vae_loss(x, x_recon, mu, log_var):
    """VAE 손실 함수 (⭐⭐⭐)

    Args:
        x: 원본 이미지 (batch, ...)
        x_recon: 재구성 이미지 (batch, ...)
        mu: 평균 (batch, latent_dim)
        log_var: 로그 분산 (batch, latent_dim)

    Returns:
        total_loss, recon_loss, kl_loss
    """
    # 재구성 손실 (이진 이미지: BCE, 연속 이미지: MSE)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # 또는 MSE
    # recon_loss = F.mse_loss(x_recon, x, reduction='sum')

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + kl_loss

    return total_loss, recon_loss, kl_loss
```

---

## 3. Reparameterization Trick

### 문제점

```
z ~ q(z|x) = N(mu, sigma)에서 샘플링

문제: 샘플링은 미분 불가 → 역전파 불가
```

### 해결: Reparameterization

```python
def reparameterize(mu, log_var):
    """Reparameterization Trick (⭐⭐⭐)

    z = mu + sigma * epsilon
    epsilon ~ N(0, I)

    이렇게 하면 랜덤성이 epsilon에 있고,
    mu, sigma에 대해 미분 가능
    """
    std = torch.exp(0.5 * log_var)  # sigma = exp(0.5 * log(sigma^2))
    eps = torch.randn_like(std)     # epsilon ~ N(0, I)
    z = mu + std * eps
    return z
```

### 시각적 이해

```
[미분 불가]
mu, sigma → 샘플링 → z → Decoder

[미분 가능 - Reparameterization]
mu ──────────────┐
                 │
                 ▼
sigma ────────▶ (mu + sigma * eps) ──▶ z ──▶ Decoder
                       ▲
eps ~ N(0, I) ─────────┘ (상수 취급)
```

---

## 4. VAE 모델 구현

### Encoder

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAEEncoder(nn.Module):
    """VAE Encoder (⭐⭐⭐)

    이미지 → mu, log_var
    """
    def __init__(self, in_channels=1, latent_dim=20):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),  # 28 → 14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # 14 → 7
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        h = self.conv_layers(x)
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
```

### Decoder

```python
class VAEDecoder(nn.Module):
    """VAE Decoder (⭐⭐⭐)

    z → 이미지
    """
    def __init__(self, latent_dim=20, out_channels=1):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)

        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 7 → 14
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, stride=2, padding=1),  # 14 → 28
            nn.Sigmoid()  # [0, 1]
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 64, 7, 7)
        x_recon = self.deconv_layers(h)
        return x_recon
```

### 전체 VAE

```python
class VAE(nn.Module):
    """Variational Autoencoder (⭐⭐⭐)"""
    def __init__(self, in_channels=1, latent_dim=20):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_dim)
        self.decoder = VAEDecoder(latent_dim, in_channels)
        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def generate(self, num_samples, device):
        """새로운 샘플 생성"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        samples = self.decoder(z)
        return samples

    def reconstruct(self, x):
        """이미지 재구성"""
        with torch.no_grad():
            x_recon, _, _ = self.forward(x)
        return x_recon
```

---

## 5. 학습 루프

```python
def train_vae(model, dataloader, epochs=50, lr=1e-3):
    """VAE 학습 (⭐⭐⭐)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)

            optimizer.zero_grad()

            # Forward
            x_recon, mu, log_var = model(data)

            # Loss
            loss, recon_loss, kl_loss = vae_loss(data, x_recon, mu, log_var)

            # Normalize by batch size
            loss = loss / data.size(0)

            # Backward
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item() / data.size(0)
            total_kl += kl_loss.item() / data.size(0)

        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon / len(dataloader)
        avg_kl = total_kl / len(dataloader)

        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Recon={avg_recon:.4f}, KL={avg_kl:.4f}")

    return model
```

---

## 6. Beta-VAE

### 아이디어

```
ELBO = Reconstruction - beta * KL

beta > 1: KL 항에 더 큰 가중치
    → 잠재 공간이 더 정규화됨
    → Disentangled representations
    → 각 잠재 차원이 독립적인 특징 포착

beta = 1: 일반 VAE
beta < 1: 재구성에 집중
```

### 구현

```python
def beta_vae_loss(x, x_recon, mu, log_var, beta=4.0):
    """Beta-VAE 손실 함수 (⭐⭐⭐)"""
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Beta 가중치
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss
```

### Disentanglement 예시

```
MNIST에서 학습된 Beta-VAE (beta=4):
    z[0]: 숫자의 기울기
    z[1]: 선 두께
    z[2]: 숫자 종류
    ...

각 차원을 독립적으로 조절하면 해당 특징만 변화
```

---

## 7. Latent Space 시각화

### 2D Latent Space

```python
def visualize_latent_space(model, dataloader, device):
    """잠재 공간 시각화 (⭐⭐)"""
    model.eval()

    latents = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            mu, _ = model.encoder(data)
            latents.append(mu.cpu())
            labels.append(label)

    latents = torch.cat(latents, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.colorbar(scatter)
    plt.xlabel('z[0]')
    plt.ylabel('z[1]')
    plt.title('VAE Latent Space')
    plt.savefig('latent_space.png')
    plt.close()
```

### Latent Space 탐색

```python
def explore_latent_dimension(model, dim_idx, range_vals, fixed_z, device):
    """특정 잠재 차원 탐색 (⭐⭐)"""
    model.eval()
    images = []

    with torch.no_grad():
        for val in range_vals:
            z = fixed_z.clone()
            z[0, dim_idx] = val
            img = model.decoder(z.to(device))
            images.append(img.cpu())

    return torch.cat(images, dim=0)
```

### Manifold 생성

```python
def generate_manifold(model, n=20, latent_dim=2, device='cpu'):
    """2D 잠재 공간의 manifold 생성 (⭐⭐⭐)"""
    model.eval()

    # 그리드 생성 (-3, 3) 범위
    grid_x = torch.linspace(-3, 3, n)
    grid_y = torch.linspace(-3, 3, n)

    figure = np.zeros((28 * n, 28 * n))

    with torch.no_grad():
        for i, yi in enumerate(grid_y):
            for j, xi in enumerate(grid_x):
                z = torch.zeros(1, latent_dim)
                z[0, 0] = xi
                z[0, 1] = yi

                x_decoded = model.decoder(z.to(device))
                digit = x_decoded[0, 0].cpu().numpy()

                figure[i * 28:(i + 1) * 28,
                       j * 28:(j + 1) * 28] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.axis('off')
    plt.savefig('vae_manifold.png')
    plt.close()
```

---

## 8. VAE vs GAN 비교

| 특성 | VAE | GAN |
|-----|-----|-----|
| 학습 방식 | 우도 최대화 | 적대적 학습 |
| 손실 함수 | ELBO (명시적) | Min-max (암시적) |
| 학습 안정성 | 안정적 | 불안정 |
| 이미지 품질 | 흐릿한 경향 | 선명함 |
| 잠재 공간 | 구조화됨 | 해석 어려움 |
| Mode Coverage | 좋음 | Mode Collapse 가능 |
| 밀도 추정 | 가능 | 불가 |

### 장단점

```
VAE 장점:
- 명시적 밀도 모델
- 안정적 학습
- 의미 있는 잠재 공간
- 재구성 + 생성 모두 가능

VAE 단점:
- 재구성 손실로 인한 흐릿한 이미지
- KL 항이 잠재 공간 표현력 제한

GAN 장점:
- 선명한 고품질 이미지
- 암시적 밀도 → 더 유연

GAN 단점:
- 학습 불안정
- Mode Collapse
- 평가 어려움
```

---

## 9. 고급 VAE 변형

### Conditional VAE (CVAE)

```python
class CVAE(nn.Module):
    """Conditional VAE (⭐⭐⭐)

    조건(예: 클래스 레이블)을 주어 특정 타입 생성
    """
    def __init__(self, in_channels=1, latent_dim=20, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

        # 조건을 one-hot으로 concat
        self.encoder = CVAEEncoder(in_channels, latent_dim, num_classes)
        self.decoder = CVAEDecoder(latent_dim, in_channels, num_classes)
        self.latent_dim = latent_dim

    def forward(self, x, label):
        # One-hot encoding
        y = F.one_hot(label, self.num_classes).float()

        mu, log_var = self.encoder(x, y)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z, y)

        return x_recon, mu, log_var

    def generate(self, label, num_samples, device):
        """특정 클래스 생성"""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        y = F.one_hot(label, self.num_classes).float().to(device)
        y = y.expand(num_samples, -1)
        return self.decoder(z, y)
```

### VQ-VAE (Vector Quantized VAE)

```python
# VQ-VAE는 연속 잠재 공간 대신 이산 코드북 사용
# 고품질 이미지/오디오 생성에 효과적

class VectorQuantizer(nn.Module):
    """VQ-VAE의 벡터 양자화 (⭐⭐⭐⭐)"""
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 코드북
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        # z: (batch, channels, H, W)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)

        # 가장 가까운 코드북 벡터 찾기
        distances = torch.cdist(z_flat, self.embeddings.weight)
        indices = torch.argmin(distances, dim=1)
        z_q = self.embeddings(indices).view(z.shape[0], z.shape[2], z.shape[3], -1)
        z_q = z_q.permute(0, 3, 1, 2)

        # 손실: 코드북 학습 + commitment loss
        loss = F.mse_loss(z_q.detach(), z) + self.commitment_cost * F.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, loss, indices
```

---

## 10. MNIST VAE 완전 예제

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 설정
latent_dim = 20
batch_size = 128
epochs = 30
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터
transform = transforms.ToTensor()
train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 모델
model = VAE(in_channels=1, latent_dim=latent_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 학습
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        x_recon, mu, log_var = model(data)

        # Loss
        recon_loss = F.binary_cross_entropy(x_recon, data, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = (recon_loss + kl_loss) / data.size(0)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print(f"Epoch {epoch+1}: Loss = {train_loss / len(train_loader):.4f}")

# 생성
model.eval()
with torch.no_grad():
    samples = model.generate(16, device)
    # 저장 또는 시각화...

print("VAE 학습 완료!")
```

---

## 정리

### 핵심 개념

1. **VAE**: 확률적 잠재 공간을 가진 생성 모델
2. **ELBO**: Reconstruction + KL Divergence
3. **Reparameterization**: z = mu + sigma * epsilon
4. **Beta-VAE**: KL 가중치 조절로 disentanglement
5. **잠재 공간**: 연속적, 구조화됨

### 핵심 코드

```python
# Encoder 출력
mu, log_var = encoder(x)

# Reparameterization
std = torch.exp(0.5 * log_var)
z = mu + std * torch.randn_like(std)

# Decoder
x_recon = decoder(z)

# Loss
recon = F.binary_cross_entropy(x_recon, x)
kl = -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp())
loss = recon + kl
```

### 사용 시나리오

| 목적 | 추천 방법 |
|-----|----------|
| 데이터 생성 | VAE 또는 GAN |
| 잠재 공간 분석 | VAE (특히 Beta-VAE) |
| 고품질 이미지 | GAN 또는 VQ-VAE |
| 조건부 생성 | CVAE |
| 압축/재구성 | VAE |

---

## 연습 문제

### 연습 1: KL 발산(KL Divergence) 항 유도

가우시안 인코더 `q(z|x) = N(mu, sigma^2 I)`와 표준 정규 사전 분포 `p(z) = N(0, I)`에 대해 폐쇄형 KL 발산을 유도하세요:

```
KL(q(z|x) || p(z)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
```

1. 정의에서 시작하세요: `KL(q || p) = E_q[log q(z|x) - log p(z)]`.
2. 가우시안 확률 밀도 함수를 대입하세요.
3. 항별로 단순화하여 `vae_loss`에서 사용되는 공식에 도달하세요.
4. 이 폐쇄형 표현이 왜 샘플링을 통한 KL 항 추정의 필요성을 없애는지 설명하세요.

### 연습 2: 재매개변수화 트릭(Reparameterization Trick) 시각화

재매개변수화 트릭이 기울기 흐름(gradient flow)을 가능하게 함을 보여주는 소규모 실험을 구현하세요:
1. 간단한 2D 가우시안 인코더를 만드세요: `mu, log_var = some_network(x)`.
2. 재매개변수화를 사용하여 `z = mu + std * eps`로 샘플링하세요.
3. 간단한 손실 `loss = z.sum()`을 계산하고 `loss.backward()`를 호출하세요.
4. 역전파 후 `mu.grad`와 `log_var.grad`가 None이 아님을 확인하세요.
5. 직접 샘플링 `z ~ N(mu, sigma)`가 이 테스트에서 왜 실패하는지 자신의 말로 설명하세요.

### 연습 3: MNIST에서 VAE 학습 및 잠재 공간 시각화

제공된 `VAE` 구현을 사용하여:
1. `latent_dim=2`로 MNIST에서 30 에폭 학습하세요.
2. 학습 후 `visualize_latent_space`를 사용하여 숫자 클래스별로 색상이 지정된 2D 잠재 공간을 플롯하세요.
3. `generate_manifold`를 사용하여 두 잠재 차원 모두에서 `(-3, 3)` 범위에 걸쳐 20×20 이미지 그리드를 생성하세요.
4. 관찰 내용을 설명하세요: 숫자 클래스가 잘 분리되어 있나요? 잠재 공간이 매끄럽고 연속적인가요?

### 연습 4: Beta-VAE 실험

다른 베타 값으로 `beta_vae_loss`를 사용하도록 학습 루프를 수정하세요:
1. `beta=0.5`, `beta=1.0` (표준 VAE), `beta=4.0`으로 세 가지 모델을 학습하세요.
2. 각 모델의 2D 잠재 공간을 시각화하세요.
3. 재구성 품질 대 잠재 공간 구조를 비교하세요: 베타를 높이면 균형에 어떤 영향을 미치나요?
4. 분리성(disentanglement)을 정성적으로 평가하세요: `beta=4.0`이 각 잠재 차원이 독립적인 시각적 특성(예: 회전, 두께)에 더 명확하게 대응하는 표현을 생성하나요?

### 연습 5: 조건부 VAE(CVAE) 구현

특정 숫자 클래스를 생성하도록 VAE를 확장하세요:
1. 평탄화된 이미지 특성에 원핫(one-hot) 클래스 레이블을 연결하도록 인코더를 수정하세요.
2. 잠재 벡터 `z`에 원핫 레이블을 연결하도록 디코더를 수정하세요.
3. MNIST에서 CVAE를 학습하고 지정된 숫자의 `num_samples`개 이미지를 생성하는 `generate` 메서드를 구현하세요.
4. 각 숫자 클래스(0-9)에 대해 10개의 샘플을 생성하고 10×10 그리드로 표시하세요. 조건부 생성이 올바른 숫자 클래스를 생성하는지 확인하세요.

---

## 다음 단계

[Diffusion Models](./32_Diffusion_Models.md)에서 최신 생성 모델인 Diffusion 모델을 학습합니다.
