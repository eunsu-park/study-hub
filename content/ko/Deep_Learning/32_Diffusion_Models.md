[이전: Variational Autoencoder (VAE)](./31_Impl_VAE.md) | [다음: 확산 모델(DDPM)](./33_Impl_Diffusion.md)

---

# 32. Diffusion Models

## 학습 목표

- Diffusion Process 이론 이해 (Forward/Reverse)
- DDPM (Denoising Diffusion Probabilistic Models) 원리
- Score-based Generative Models 개념
- U-Net 아키텍처 for Diffusion
- Stable Diffusion 핵심 원리
- Classifier-free Guidance
- 간단한 DDPM PyTorch 구현

---

## 1. Diffusion Process 개요

### 핵심 아이디어

```
데이터에 점진적으로 노이즈를 추가 (Forward Process)
    x_0 → x_1 → x_2 → ... → x_T (순수 노이즈)

노이즈를 점진적으로 제거하여 데이터 생성 (Reverse Process)
    x_T → x_{T-1} → ... → x_0 (깨끗한 이미지)

핵심: Reverse Process를 신경망으로 학습
```

### 시각적 이해

```
Forward (노이즈 추가):
[깨끗한 이미지] ──t=0──▶ [약간 노이즈] ──t=500──▶ [더 많은 노이즈] ──t=1000──▶ [완전 노이즈]

Reverse (노이즈 제거):
[완전 노이즈] ──t=1000──▶ [약간 선명] ──t=500──▶ [더 선명] ──t=0──▶ [깨끗한 이미지]
```

---

## 2. DDPM (Denoising Diffusion Probabilistic Models)

### Forward Process (q)

```python
# Forward process: q(x_t | x_{t-1})
# x_t = sqrt(1 - beta_t) * x_{t-1} + sqrt(beta_t) * epsilon

# Closed form (한 번에 x_0에서 x_t로):
# x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

# alpha_t = 1 - beta_t
# alpha_bar_t = prod(alpha_1 * alpha_2 * ... * alpha_t)
```

### 수학적 정의

```python
import torch

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """선형 노이즈 스케줄 (⭐⭐)"""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """코사인 노이즈 스케줄 (더 좋은 성능) (⭐⭐⭐)"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def get_index_from_list(vals, t, x_shape):
    """배치의 각 샘플에 맞는 t 시점의 값 추출"""
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
```

### Forward Diffusion 구현

```python
class DiffusionSchedule:
    """Diffusion 스케줄 관리 (⭐⭐⭐)"""
    def __init__(self, timesteps=1000, beta_schedule='linear'):
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        else:
            betas = cosine_beta_schedule(timesteps)

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # 계산에 필요한 값들
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Posterior 계산용
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward process: x_0에서 x_t 샘플링 (⭐⭐⭐)

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alphas_cumprod_t = get_index_from_list(
            self.sqrt_alphas_cumprod, t, x_0.shape
        )
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            self.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
```

---

## 3. 노이즈 예측 네트워크

### 목표

```
모델이 x_t에서 추가된 노이즈 epsilon을 예측
epsilon_theta(x_t, t) ≈ epsilon

손실 함수:
L = E[||epsilon - epsilon_theta(x_t, t)||^2]
```

### 간단한 U-Net 구조

```python
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """시간 임베딩 (⭐⭐⭐)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """기본 Conv Block"""
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x, t):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(...,) + (None,) * 2]
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SimpleUNet(nn.Module):
    """간단한 U-Net for Diffusion (⭐⭐⭐)"""
    def __init__(self, in_channels=3, out_channels=3, time_dim=256):
        super().__init__()

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.ReLU()
        )

        # Initial projection
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)

        # Downsampling
        self.downs = nn.ModuleList([
            Block(64, 128, time_dim),
            Block(128, 256, time_dim),
            Block(256, 256, time_dim),
        ])

        # Upsampling
        self.ups = nn.ModuleList([
            Block(256, 128, time_dim, up=True),
            Block(128, 64, time_dim, up=True),
            Block(64, 64, time_dim, up=True),
        ])

        # Output
        self.output = nn.Conv2d(64, out_channels, 1)

    def forward(self, x, timestep):
        # Time embedding
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)

        # Downsampling
        residuals = []
        for down in self.downs:
            x = down(x, t)
            residuals.append(x)

        # Upsampling with skip connections
        for up in self.ups:
            residual = residuals.pop()
            x = torch.cat((x, residual), dim=1)
            x = up(x, t)

        return self.output(x)
```

---

## 4. 학습 과정

### 학습 알고리즘 (DDPM)

```
1. x_0 ~ q(x_0): 데이터에서 샘플
2. t ~ Uniform(1, T): 랜덤 타임스텝
3. epsilon ~ N(0, I): 랜덤 노이즈
4. x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
5. Loss = ||epsilon - epsilon_theta(x_t, t)||^2
6. 역전파 및 업데이트
```

### 학습 코드

```python
def train_diffusion(model, schedule, dataloader, epochs=100, lr=1e-4):
    """Diffusion 모델 학습 (⭐⭐⭐)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            batch_size = images.size(0)

            # 랜덤 타임스텝
            t = torch.randint(0, schedule.timesteps, (batch_size,), device=device).long()

            # 노이즈 추가
            noise = torch.randn_like(images)
            x_t = schedule.q_sample(images, t, noise)

            # 노이즈 예측
            noise_pred = model(x_t, t)

            # 손실 계산
            loss = criterion(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
```

---

## 5. 샘플링 (Reverse Process)

### DDPM 샘플링

```python
@torch.no_grad()
def sample_ddpm(model, schedule, shape, device):
    """DDPM 샘플링 (⭐⭐⭐)

    x_T ~ N(0, I)에서 시작하여 x_0 생성
    """
    model.eval()

    # 순수 노이즈에서 시작
    x = torch.randn(shape, device=device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        betas_t = get_index_from_list(schedule.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
            schedule.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = get_index_from_list(
            schedule.sqrt_recip_alphas, t, x.shape
        )

        # 노이즈 예측
        noise_pred = model(x, t)

        # Mean 계산
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )

        if i > 0:
            posterior_variance_t = get_index_from_list(
                schedule.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            x = model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            x = model_mean

    return x
```

### DDIM 샘플링 (더 빠름)

```python
@torch.no_grad()
def sample_ddim(model, schedule, shape, device, num_inference_steps=50, eta=0.0):
    """DDIM 샘플링 (⭐⭐⭐⭐)

    더 적은 스텝으로 빠른 샘플링
    eta=0: 결정론적, eta=1: DDPM과 동일
    """
    model.eval()

    # 스텝 간격
    step_size = schedule.timesteps // num_inference_steps
    timesteps = list(range(0, schedule.timesteps, step_size))
    timesteps = list(reversed(timesteps))

    x = torch.randn(shape, device=device)

    for i, t in enumerate(timesteps):
        t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)

        alpha_cumprod_t = schedule.alphas_cumprod[t]
        alpha_cumprod_prev = schedule.alphas_cumprod[timesteps[i+1]] if i < len(timesteps)-1 else 1.0

        # 노이즈 예측
        noise_pred = model(x, t_tensor)

        # x_0 예측
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # 방향 계산
        sigma = eta * torch.sqrt((1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)) * \
                     torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_prev)

        pred_dir = torch.sqrt(1 - alpha_cumprod_prev - sigma**2) * noise_pred

        # 노이즈 추가 (eta > 0인 경우)
        noise = torch.randn_like(x) if eta > 0 else 0

        x = torch.sqrt(alpha_cumprod_prev) * pred_x0 + pred_dir + sigma * noise

    return x
```

---

## 6. Score-based Models

### Score Function

```
Score = gradient of log probability
s(x) = ∇_x log p(x)

노이즈가 추가된 데이터의 score:
s_theta(x_t, t) ≈ ∇_{x_t} log p(x_t)
```

### DDPM과의 관계

```python
# DDPM에서 노이즈 예측과 score의 관계:
# epsilon_theta(x_t, t) = -sqrt(1 - alpha_bar_t) * s_theta(x_t, t)

# Score 예측 → 노이즈 예측으로 변환 가능
```

---

## 7. Stable Diffusion 원리

### Latent Diffusion

```
이미지 공간이 아닌 잠재 공간에서 diffusion

1. Encoder: 이미지 → 잠재 표현 z
2. Diffusion: z에서 노이즈 추가/제거
3. Decoder: 잠재 표현 → 이미지

장점:
- 계산 효율성 (작은 해상도에서 diffusion)
- 고해상도 이미지 생성 가능
```

### 아키텍처

```
┌──────────────┐
│  Text Prompt │
└──────┬───────┘
       │ CLIP Text Encoder
       ▼
┌──────────────────────────────────────────┐
│              Cross-Attention              │
├──────────────────────────────────────────┤
│                                          │
│  z_T ──▶ U-Net ──▶ z_{T-1} ──▶ ... ──▶ z_0  │
│         (time embedding)                 │
│                                          │
└──────────────────────────────────────────┘
       │ VAE Decoder
       ▼
┌──────────────┐
│    Image     │
└──────────────┘
```

### 조건부 생성 (Cross-Attention)

```python
class CrossAttention(nn.Module):
    """Text-Image Cross Attention (⭐⭐⭐⭐)"""
    def __init__(self, query_dim, context_dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        # x: 이미지 특징 (batch, hw, dim)
        # context: 텍스트 임베딩 (batch, seq_len, context_dim)

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        # Multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
```

---

## 8. Classifier-free Guidance

### 아이디어

```
조건부 생성과 비조건부 생성을 혼합

epsilon_guided = epsilon_uncond + w * (epsilon_cond - epsilon_uncond)

w > 1: 조건을 더 강하게 반영 (더 선명하지만 다양성 감소)
w = 1: 일반 조건부 생성
w < 1: 조건 약화
```

### 구현

```python
def classifier_free_guidance_sample(model, schedule, shape, condition, w=7.5, device='cuda'):
    """Classifier-free Guidance 샘플링 (⭐⭐⭐⭐)"""
    model.eval()

    x = torch.randn(shape, device=device)

    for i in reversed(range(schedule.timesteps)):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)

        # 조건부 예측
        noise_cond = model(x, t, condition)

        # 비조건부 예측 (조건 = None 또는 빈 임베딩)
        noise_uncond = model(x, t, None)

        # Guidance
        noise_pred = noise_uncond + w * (noise_cond - noise_uncond)

        # 샘플링 스텝 (DDPM 또는 DDIM)
        x = sampling_step(x, noise_pred, t, schedule)

    return x
```

### 학습 시 조건 드롭아웃

```python
def train_with_cfg(model, dataloader, drop_prob=0.1):
    """CFG를 위한 학습 (조건 드롭아웃) (⭐⭐⭐)"""
    for images, conditions in dataloader:
        # 일정 확률로 조건을 None으로
        mask = torch.rand(images.size(0)) < drop_prob
        conditions[mask] = None  # 또는 빈 임베딩

        # 일반 학습...
```

---

## 9. 간단한 DDPM 전체 예제

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 설정
image_size = 28
channels = 1
timesteps = 1000
batch_size = 64
epochs = 50
lr = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 데이터
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)  # [0, 1] → [-1, 1]
])

train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# 스케줄
schedule = DiffusionSchedule(timesteps=timesteps, beta_schedule='linear')

# 모델 (간단한 버전)
model = SimpleUNet(in_channels=channels, out_channels=channels).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 학습
for epoch in range(epochs):
    total_loss = 0

    for images, _ in train_loader:
        images = images.to(device)
        batch_size = images.size(0)

        # 랜덤 타임스텝
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # 노이즈 추가 (forward process)
        noise = torch.randn_like(images)
        x_t = schedule.q_sample(images, t, noise)

        # 노이즈 예측
        noise_pred = model(x_t, t)

        # 손실
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")

# 샘플링
model.eval()
with torch.no_grad():
    samples = sample_ddpm(model, schedule, (16, channels, image_size, image_size), device)
    samples = (samples + 1) / 2  # [-1, 1] → [0, 1]

# 시각화
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i, 0].cpu(), cmap='gray')
    ax.axis('off')
plt.savefig('diffusion_samples.png')
print("샘플 저장: diffusion_samples.png")
```

---

## 정리

### 핵심 개념

1. **Forward Process**: 점진적 노이즈 추가 q(x_t|x_0)
2. **Reverse Process**: 점진적 노이즈 제거 p(x_{t-1}|x_t)
3. **DDPM**: 노이즈 예측으로 역과정 학습
4. **DDIM**: 결정론적 샘플링으로 빠른 생성
5. **Latent Diffusion**: 잠재 공간에서 효율적 생성
6. **CFG**: 조건 강도 조절

### 핵심 수식

```
Forward: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

Loss: L = E[||epsilon - epsilon_theta(x_t, t)||^2]

Reverse: x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * epsilon_theta) + sigma_t * z
```

### Diffusion vs GAN vs VAE

| 특성 | Diffusion | GAN | VAE |
|-----|-----------|-----|-----|
| 학습 안정성 | 매우 높음 | 낮음 | 높음 |
| 이미지 품질 | 최고 | 좋음 | 흐림 |
| 샘플링 속도 | 느림 | 빠름 | 빠름 |
| Mode Coverage | 좋음 | Mode Collapse | 좋음 |
| 밀도 추정 | 가능 | 불가 | 가능 |

---

## 연습 문제

### 연습 1: 순방향 확산 과정(Forward Diffusion Process) 시각화

`DiffusionSchedule`과 `q_sample` 메서드를 사용하여:
1. MNIST 이미지 하나를 불러와 타임스텝 `t = 0, 100, 250, 500, 750, 999`에서 순방향 확산을 적용하세요.
2. 6개의 노이즈 이미지를 한 행에 나란히 플롯하세요.
3. 각 타임스텝에서 신호 대 잡음비(SNR, Signal-to-Noise Ratio)를 계산하고 출력하세요: `SNR(t) = alpha_bar_t / (1 - alpha_bar_t)`.
4. SNR과 이미지 인식 가능성의 관계를 설명하세요. 어느 타임스텝에서 원본 이미지가 인식 불가능해지나요?

### 연습 2: 선형 vs 코사인 노이즈 스케줄 비교

`linear_beta_schedule`과 `cosine_beta_schedule`을 사용하여:
1. 두 스케줄의 `beta_t` 대 `t`를 같은 그래프에 플롯하세요.
2. 두 스케줄의 `alpha_bar_t` (누적 곱) 대 `t`를 플롯하세요.
3. 같은 이미지에 두 스케줄을 모두 적용하고 `t = 200, 500, 800`에서 노이즈 진행을 비교하세요.
4. 코사인 스케줄이 일반적으로 선호되는 이유를 설명하세요: 선형 스케줄은 어디서 노이즈 "예산"을 너무 빨리 소모하나요?

### 연습 3: MNIST에서 간단한 DDPM 학습

제공된 `SimpleUNet`, `DiffusionSchedule`, 학습 코드를 사용하여:
1. `timesteps=1000`, `batch_size=64`, `lr=1e-3`으로 50 에폭 학습하세요.
2. 학습 후 `sample_ddpm`을 사용해 16개의 샘플을 생성하고 그리드로 저장하세요.
3. 학습 손실 곡선을 모니터링하고 플롯하세요.
4. 실험: 학습 시 `timesteps=200`만 사용하면 어떻게 되나요? 모델이 여전히 합리적인 샘플을 생성하나요?

### 연습 4: DDIM 샘플링 구현 및 속도 비교

DDPM 모델을 학습한 후:
1. `sample_ddpm` (1000 스텝)으로 16개의 샘플을 생성하고 실제 경과 시간을 측정하세요.
2. `num_inference_steps=50`, `eta=0.0`으로 `sample_ddim`을 사용해 16개의 샘플을 생성하고 시간을 측정하세요.
3. 두 방법의 샘플 품질을 시각적으로 비교하세요.
4. DDIM에서 `eta=0.5`와 `eta=1.0`으로 실험해보세요. eta가 결정론성(determinism)과 샘플 다양성 사이의 균형을 어떻게 제어하는지 설명하세요.

### 연습 5: 조건부 모델을 위한 분류기 없는 안내(Classifier-Free Guidance) 구현

MNIST에서 클래스 조건부 생성을 지원하도록 DDPM을 확장하세요:
1. 선택적 클래스 레이블 `c`를 받도록 `SimpleUNet`을 수정하세요 (`nn.Embedding(10, time_dim)`으로 임베딩하여 시간 임베딩에 더하세요).
2. 학습 중 확률 `p=0.1`로 조건을 무작위로 드롭하세요 (레이블을 null 임베딩 인덱스 예: 인덱스 10으로 교체).
3. 조건부와 비조건부 노이즈 예측을 결합하는 `classifier_free_guidance_sample`을 구현하세요: `noise_guided = noise_uncond + w * (noise_cond - noise_uncond)`.
4. 안내 강도 `w=1.0, 3.0, 7.5`로 각 숫자 클래스 0-9에 대한 샘플을 생성하세요. 안내 강도가 높을수록 샘플 선명도와 클래스 충실도에 어떤 영향을 미치는지 관찰하세요.

---

## 다음 단계

[Attention 메커니즘 심화](./17_Attention_Deep_Dive.md)에서 Attention 메커니즘을 심층적으로 학습합니다.
