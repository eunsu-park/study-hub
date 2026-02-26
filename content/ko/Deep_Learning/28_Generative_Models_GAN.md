[이전: TensorBoard 시각화](./27_TensorBoard.md) | [다음: 생성적 적대 신경망(GAN)](./29_Impl_GAN.md)

---

# 28. 생성 모델 - GAN (Generative Adversarial Networks)

## 학습 목표

- GAN의 기본 원리와 적대적 학습 이해
- Generator와 Discriminator 구조 설계
- 다양한 손실 함수 (Adversarial, Wasserstein, WGAN-GP)
- DCGAN 아키텍처 구현
- 학습 안정화 기법 적용
- StyleGAN 개념 이해

---

## 1. GAN 기초 이론

### 개념

```
GAN = 두 신경망의 경쟁적 학습

Generator (G): 가짜 데이터 생성
    랜덤 노이즈 z → 가짜 이미지

Discriminator (D): 진짜/가짜 판별
    이미지 → 진짜(1) / 가짜(0)

목표: G가 D를 속일 수 있을 만큼 좋은 가짜 생성
```

### 적대적 학습 (Adversarial Training)

```
┌────────────┐     ┌────────────┐
│  Noise z   │────▶│ Generator  │───┬──▶ 가짜 이미지
└────────────┘     └────────────┘   │
                                    │
┌────────────┐                      ▼
│ 진짜 이미지│────────────────────▶ Discriminator ──▶ 진짜/가짜
└────────────┘

D 학습: 진짜=1, 가짜=0 판별 정확도 최대화
G 학습: D가 가짜를 진짜로 판별하도록 유도
```

### Min-Max 게임

```python
# GAN 목적 함수 (민맥스 게임):
#
#   min_G max_D  V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
#
# 기호 정의:
#   D(·)  — 판별자(Discriminator): 이미지를 실제일 확률로 매핑
#   G(·)  — 생성자(Generator): 잠재 노이즈 z를 합성 이미지로 매핑
#   x     — 실제 데이터 분포 p_data에서 추출한 샘플
#   z     — 사전 노이즈 분포(일반적으로 N(0, I))에서 추출한 샘플
#   E_x   — 실제 데이터에 대한 기대값;  E_z — 노이즈에 대한 기대값

# D의 목표: V(D, G) 최대화
#   - D(x) → 1 (실제를 실제로 분류)
#   - D(G(z)) → 0 (가짜를 가짜로 분류)

# G의 목표: V(D, G) 최소화
#   - D(G(z)) → 1 (D가 가짜를 실제로 분류하도록)
```

---

## 2. 기본 GAN 구현

### Generator

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    """간단한 Generator (⭐⭐)"""
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(torch.prod(torch.tensor(img_shape)))),
            nn.Tanh()  # 출력: [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)
```

### Discriminator

```python
class Discriminator(nn.Module):
    """간단한 Discriminator (⭐⭐)"""
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(torch.prod(torch.tensor(img_shape))), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 출력: [0, 1] 확률
        )

    def forward(self, img):
        return self.model(img)
```

### 학습 루프

```python
def train_gan(generator, discriminator, dataloader, epochs=100, latent_dim=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator.to(device)
    discriminator.to(device)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)

            # 레이블
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ==================
            # Discriminator 학습
            # ==================
            optimizer_D.zero_grad()

            # 진짜 이미지
            real_output = discriminator(real_imgs)
            d_loss_real = criterion(real_output, real_labels)

            # 가짜 이미지
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)
            # .detach()는 필수: D의 손실이 G로 역전파되면 안 됩니다.
            # detach 없이는 optimizer_D.step()이 D를 업데이트하면서 동시에
            # G의 생성 품질을 낮추는 방향으로도 작용 — 적대적 게임이 깨집니다.
            fake_output = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_output, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ==================
            # 생성자 학습
            # ==================
            optimizer_G.zero_grad()

            # D를 다시 통과 (이번에는 detach 없이, D의 출력에서
            # G의 매개변수까지 그래디언트가 흐르도록).
            # *실제* 레이블을 사용하는 이유: G는 D가 가짜를 실제로 믿길 원합니다.
            # 이것이 "비포화(Non-saturating)" 손실입니다: log(1 - D(G(z)))를
            # 최소화하는 대신 (D가 확신할 때 포화됨) log(D(G(z)))를
            # 최대화합니다 — 같은 최적점이지만 G가 아직 미숙한 학습 초기에
            # 더 강한 그래디언트를 제공합니다.
            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, real_labels)

            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
```

---

## 3. 손실 함수

### Vanilla GAN Loss (BCE)

```python
# Binary Cross Entropy
criterion = nn.BCELoss()

# D loss
d_loss = criterion(D(real), 1) + criterion(D(G(z)), 0)

# G loss (non-saturating)
g_loss = criterion(D(G(z)), 1)  # -log(D(G(z)))

# G loss (original, saturating)
# g_loss = -criterion(D(G(z)), 0)  # log(1 - D(G(z))) - 잘 안 씀
```

### Wasserstein Loss (WGAN)

```python
def wasserstein_loss(y_pred, y_true):
    """Wasserstein distance (Earth Mover's Distance)"""
    return torch.mean(y_pred * y_true)

# D (Critic) loss - 최대화
d_loss = torch.mean(D(real)) - torch.mean(D(G(z)))
# → 최소화하려면 부호 반전: -D(real) + D(G(z))

# G loss - 최소화
g_loss = -torch.mean(D(G(z)))

# Weight Clipping (WGAN)
for p in discriminator.parameters():
    p.data.clamp_(-0.01, 0.01)
```

### WGAN-GP (Gradient Penalty)

```python
def gradient_penalty(discriminator, real_imgs, fake_imgs, device):
    """Gradient Penalty for WGAN-GP (⭐⭐⭐)"""
    batch_size = real_imgs.size(0)

    # 랜덤 interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
    interpolated.requires_grad_(True)

    # Discriminator 출력
    d_interpolated = discriminator(interpolated)

    # Gradient 계산
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    # Gradient norm
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Penalty: (||grad|| - 1)^2
    penalty = ((gradient_norm - 1) ** 2).mean()

    return penalty

# WGAN-GP Loss
lambda_gp = 10
gp = gradient_penalty(discriminator, real_imgs, fake_imgs, device)
d_loss = -torch.mean(D(real)) + torch.mean(D(G(z))) + lambda_gp * gp
```

### Hinge Loss

```python
# Discriminator loss
d_loss_real = torch.mean(torch.relu(1.0 - D(real)))
d_loss_fake = torch.mean(torch.relu(1.0 + D(G(z))))
d_loss = d_loss_real + d_loss_fake

# Generator loss
g_loss = -torch.mean(D(G(z)))
```

---

## 4. DCGAN 아키텍처

### 핵심 원칙

```
1. Pooling 제거 → Strided Conv (D), Transposed Conv (G)
2. BatchNorm 사용 (G의 출력층, D의 입력층 제외)
3. G에서 ReLU (출력층은 Tanh)
4. D에서 LeakyReLU
```

### DCGAN Generator

```python
class DCGANGenerator(nn.Module):
    """DCGAN Generator (⭐⭐⭐)

    z (100,) → (1024, 4, 4) → (512, 8, 8) → (256, 16, 16) → (128, 32, 32) → (3, 64, 64)
    """
    def __init__(self, latent_dim=100, ngf=64, nc=3):
        super().__init__()

        self.main = nn.Sequential(
            # 입력: z (latent_dim,)
            # 출력: (ngf*8, 4, 4)
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # (ngf*8, 4, 4) → (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # (ngf*4, 8, 8) → (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # (ngf*2, 16, 16) → (ngf, 32, 32)
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # (ngf, 32, 32) → (nc, 64, 64)
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # z: (batch, latent_dim) → (batch, latent_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.main(z)
```

### DCGAN Discriminator

```python
class DCGANDiscriminator(nn.Module):
    """DCGAN Discriminator (⭐⭐⭐)

    (3, 64, 64) → (64, 32, 32) → (128, 16, 16) → (256, 8, 8) → (512, 4, 4) → (1,)
    """
    def __init__(self, nc=3, ndf=64):
        super().__init__()

        self.main = nn.Sequential(
            # (nc, 64, 64) → (ndf, 32, 32)
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf, 32, 32) → (ndf*2, 16, 16)
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2, 16, 16) → (ndf*4, 8, 8)
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4, 8, 8) → (ndf*8, 4, 4)
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8, 4, 4) → (1, 1, 1)
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, img):
        return self.main(img).view(-1, 1)
```

---

## 5. 학습 안정화 기법

### Spectral Normalization

```python
from torch.nn.utils import spectral_norm

class SNDiscriminator(nn.Module):
    """Spectral Normalization 적용 Discriminator (⭐⭐⭐)"""
    def __init__(self, nc=3, ndf=64):
        super().__init__()

        self.main = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False))
        )

    def forward(self, img):
        return self.main(img).view(-1, 1)
```

### Label Smoothing

```python
# One-sided label smoothing
real_labels = torch.ones(batch_size, 1, device=device) * 0.9  # 1.0 → 0.9
fake_labels = torch.zeros(batch_size, 1, device=device)

# 또는 noisy labels
real_labels = 0.7 + 0.3 * torch.rand(batch_size, 1, device=device)  # [0.7, 1.0]
```

### Two Time-Scale Update Rule (TTUR)

```python
# D는 더 높은 학습률
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0004, betas=(0.0, 0.9))
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
```

### Progressive Growing

```python
# 작은 해상도에서 시작해서 점진적으로 키움
# ProGAN (Progressive GAN)

resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# 각 해상도에서 일정 epoch 학습 후 다음 해상도로
# Fade-in: 새 레이어를 점진적으로 추가
```

---

## 6. StyleGAN 개요

### 핵심 아이디어

```
기존 GAN: z → G → 이미지
StyleGAN: z → Mapping Network → w → Synthesis Network → 이미지

Mapping Network: 8층 MLP, z를 "disentangled" w로 변환
AdaIN: w를 사용해 각 레이어의 스타일 주입
```

### Mapping Network

```python
class MappingNetwork(nn.Module):
    """StyleGAN Mapping Network (⭐⭐⭐⭐)"""
    def __init__(self, latent_dim=512, w_dim=512, num_layers=8):
        super().__init__()

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(latent_dim if i == 0 else w_dim, w_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.mapping = nn.Sequential(*layers)

    def forward(self, z):
        return self.mapping(z)
```

### AdaIN (Adaptive Instance Normalization)

```python
class AdaIN(nn.Module):
    """Adaptive Instance Normalization (⭐⭐⭐⭐)"""
    def __init__(self, num_features, w_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features)
        self.style = nn.Linear(w_dim, num_features * 2)  # scale + bias

    def forward(self, x, w):
        # x: (batch, channels, H, W)
        # w: (batch, w_dim)

        normalized = self.norm(x)

        style = self.style(w)  # (batch, channels*2)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * normalized + beta
```

### Style Mixing

```python
# 서로 다른 z에서 w 생성
z1, z2 = torch.randn(2, latent_dim)
w1, w2 = mapping(z1), mapping(z2)

# 특정 레이어까지는 w1, 그 이후는 w2 사용
# → 스타일 혼합 (coarse: w1, fine: w2)
```

---

## 7. 이미지 생성 및 평가

### 샘플 이미지 생성

```python
def generate_samples(generator, latent_dim, num_samples=64):
    generator.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim, device=device)
        fake_imgs = generator(z)
    return fake_imgs

# 시각화
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def show_generated_images(images, nrow=8):
    """생성된 이미지 그리드 표시"""
    # [-1, 1] → [0, 1]
    images = (images + 1) / 2
    grid = vutils.make_grid(images.cpu(), nrow=nrow, normalize=False)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig('generated_samples.png')
    plt.close()
```

### Interpolation

```python
def interpolate_latent(generator, z1, z2, steps=10):
    """잠재 공간 보간 (⭐⭐)"""
    generator.eval()
    images = []

    with torch.no_grad():
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * z1 + alpha * z2
            img = generator(z.unsqueeze(0))
            images.append(img)

    return torch.cat(images, dim=0)

# Spherical interpolation (slerp) - 더 나은 결과
def slerp(z1, z2, alpha):
    """구면 선형 보간"""
    omega = torch.acos((z1 * z2).sum() / (z1.norm() * z2.norm()))
    return (torch.sin((1 - alpha) * omega) / torch.sin(omega)) * z1 + \
           (torch.sin(alpha * omega) / torch.sin(omega)) * z2
```

### FID (Frechet Inception Distance)

```python
# FID 계산 (pytorch-fid 라이브러리 사용)
# pip install pytorch-fid

# from pytorch_fid import fid_score
# fid = fid_score.calculate_fid_given_paths(
#     [real_images_path, fake_images_path],
#     batch_size=50,
#     device=device,
#     dims=2048
# )
# 낮을수록 좋음 (0이 완벽)
```

---

## 8. MNIST GAN 완전 예제

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 하이퍼파라미터
latent_dim = 100
lr = 0.0002
batch_size = 64
epochs = 50

# 데이터
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1]
])

mnist = datasets.MNIST('data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

# 간단한 모델
G = Generator(latent_dim=latent_dim, img_shape=(1, 28, 28)).to(device)
D = Discriminator(img_shape=(1, 28, 28)).to(device)

# 옵티마이저
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# 학습
for epoch in range(epochs):
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)

        # 레이블
        real = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # D 학습
        optimizer_D.zero_grad()
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_imgs = G(z)

        d_loss = criterion(D(real_imgs), real) + criterion(D(fake_imgs.detach()), fake)
        d_loss.backward()
        optimizer_D.step()

        # G 학습
        optimizer_G.zero_grad()
        g_loss = criterion(D(fake_imgs), real)
        g_loss.backward()
        optimizer_G.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: D={d_loss.item():.4f}, G={g_loss.item():.4f}")

        # 샘플 저장
        with torch.no_grad():
            sample = G(torch.randn(16, latent_dim, device=device))
            # 저장 코드...
```

---

## 정리

### 핵심 개념

1. **GAN**: Generator와 Discriminator의 적대적 학습
2. **손실 함수**: BCE, Wasserstein, Gradient Penalty
3. **DCGAN**: Transposed Conv + BatchNorm + LeakyReLU
4. **안정화**: Spectral Norm, TTUR, Progressive Growing
5. **StyleGAN**: Mapping Network + AdaIN으로 스타일 제어

### GAN 학습 팁

```
1. D와 G 균형 유지 (D가 너무 강하면 G가 학습 불가)
2. BatchNorm은 minibatch 전체에 적용 (진짜/가짜 분리하지 않음)
3. Adam beta1=0.5 사용 (momentum 줄임)
4. 학습률: 보통 0.0001 ~ 0.0002
5. 생성 이미지 주기적으로 확인
```

### 모드 붕괴(Mode Collapse)

모드 붕괴는 GAN의 가장 흔한 실패 모드입니다: **G가 전체 데이터 분포를 커버하지 않고 소수의 출력 유형만 생성**하게 됩니다.

**왜 발생하는가**: G가 D를 안정적으로 속이는 단일 출력(또는 소수)을 찾으면, 다양화할 인센티브가 없습니다. D가 결국 이를 알아채고 결정 경계를 이동하지만, G는 단순히 다른 좁은 모드로 점프 — 수렴이 아닌 진동이 발생합니다.

**모드 붕괴의 징후**:
- 다른 노이즈 입력에서 생성된 이미지들이 거의 동일하게 보임
- FID 점수가 개선되다가 정체되거나 진동

**완화 방법**:
- **WGAN/WGAN-GP**: 바서스타인 거리(Wasserstein Distance)는 D가 확신할 때에도 의미 있는 그래디언트를 제공하여, G가 더 많은 모드를 커버하도록 유도
- **스펙트럴 정규화(Spectral Normalization)**: D의 립시츠(Lipschitz) 상수를 제한하여, D가 너무 날카로워져 G를 궁지에 모는 것을 방지
- **다양성 정규화**: 서로 다른 z 입력이 유사한 출력을 생성할 때 G에 페널티 부여

### 손실 함수 비교

| 손실 함수 | 장점 | 단점 |
|----------|------|------|
| BCE | 간단함 | Mode collapse |
| WGAN | 안정적 학습 | Weight clipping |
| WGAN-GP | 매우 안정적 | 계산 비용 |
| Hinge | 간단하고 효과적 | - |

---

## 연습 문제

### 연습 1: 민맥스 게임(Min-Max Game) 설명하기

GAN 학습이 민맥스 게임(min-max game)으로 정식화되는 이유를 자신의 말로 설명하세요:
1. 판별자(Discriminator)가 최대화하는 목표는 무엇이며 그 이유는?
2. 생성자(Generator)가 최소화하는 목표는 무엇이며 그 이유는?
3. 학습 초기에 "포화되지 않는(non-saturating)" 생성자 손실(`-log D(G(z))`)이 원래 공식(`log(1 - D(G(z)))`)보다 잘 동작하는 이유는? 힌트: D가 확신을 가질 때 기울기(gradient)에 어떤 일이 일어나는지 생각해보세요.

### 연습 2: DCGAN 가중치 초기화 구현

DCGAN 논문은 모든 Conv와 BatchNorm 가중치를 `mean=0.0, std=0.02`의 정규 분포로 초기화해야 한다고 명시합니다. `DCGANGenerator`의 `_init_weights` 메서드를 검토하고 답하세요:
1. GAN 안정성을 위해 커스텀 가중치 초기화가 PyTorch 기본 초기화보다 중요한 이유는?
2. `DCGANGenerator`를 수정하여 `nn.init.constant_`를 사용해 `BatchNorm` bias도 `0`으로 초기화하세요. 이 변경 사항이 `DCGANDiscriminator._init_weights`에서 이미 수행하는 것과 일치하는지 확인하세요.

### 연습 3: MNIST에서 Vanilla GAN 학습

제공된 `Generator`, `Discriminator`, 학습 루프 코드를 사용하여 MNIST 데이터셋에서 GAN을 학습하세요:
1. `latent_dim=100`, `batch_size=64`, `epochs=50`, `lr=0.0002`로 설정하세요.
2. `show_generated_images`를 사용하여 10 에폭마다 생성된 16개 이미지의 그리드를 저장하세요.
3. 학습 전반에 걸친 판별자 손실과 생성자 손실 곡선을 플롯하세요.
4. 관찰하고 설명하세요: 둘 중 어느 손실이 수렴하나요? 학습 동작이 적대적 게임에 대해 무엇을 알려주나요?

### 연습 4: WGAN과 Vanilla GAN 비교

학습 루프의 BCE 손실을 Wasserstein 손실로 교체하세요:
1. 판별자 스텝마다 가중치 클리핑(`clamp_(-0.01, 0.01)`)을 사용하는 WGAN 학습 루프를 구현하세요.
2. 판별자(Discriminator)에서 `Sigmoid` 활성화 함수를 제거하세요 (WGAN은 제약 없는 "critic"을 사용합니다).
3. MNIST에서 Vanilla GAN과 WGAN을 각각 20 에폭씩 학습하세요.
4. 생성 이미지 품질과 학습 안정성을 비교하세요. 어느 버전이 더 안정적인 손실 곡선을 보이는지 문서화하세요.

### 연습 5: 잠재 공간 보간(Latent Space Interpolation) 구현 및 평가

학습된 MNIST GAN을 사용하여:
1. 두 개의 랜덤 잠재 벡터 `z1`과 `z2`를 샘플링하세요.
2. 10 스텝으로 `z1`과 `z2` 사이의 선형 보간과 구면 보간(`slerp`) 모두 구현하세요.
3. 두 보간 방법의 이미지 시퀀스를 생성하고 시각화하세요.
4. 구면 보간이 가우시안 잠재 공간에서 선형 보간보다 더 부드러운 전환을 만드는 이유를 설명하세요. 힌트: 랜덤 가우시안 벡터의 기대 노름(norm)과 선형 보간 중간에 어떤 일이 일어나는지 생각해보세요.

---

## 다음 단계

[생성 모델 - VAE (Variational Autoencoder)](./30_Generative_Models_VAE.md)에서 VAE (Variational Autoencoder)를 학습합니다.
