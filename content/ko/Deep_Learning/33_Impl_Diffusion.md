[이전: Diffusion Models](./32_Diffusion_Models.md) | [다음: CLIP과 멀티모달 학습](./34_CLIP_Multimodal.md)

---

# 33. 확산 모델(Diffusion Models, DDPM)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 순방향 확산 과정(Forward Diffusion Process)(가우시안 노이즈를 점진적으로 추가하는 마르코프 체인)을 설명하고, 누적 노이즈 스케줄을 사용하여 q(x_t | x_0)의 폐쇄형 표현식을 유도합니다.
2. 역방향 과정(Reverse Process)을 공식화하고, 각 타임스텝에서 노이즈를 예측하고 제거하도록 신경망을 훈련하는 방법을 설명합니다.
3. ELBO로부터 단순화된 DDPM 학습 목적함수(노이즈 예측 손실)를 유도하고, 이 파라미터화가 실제로 왜 효과적인지 설명합니다.
4. PyTorch에서 U-Net 노이즈 예측기를 사용하여 DDPM 학습 루프와 추론(역방향 확산 샘플링)을 처음부터 구현합니다.
5. 노이즈 스케줄(선형 및 코사인)을 구현하고 비교하며, 스케줄 선택이 생성 품질에 어떤 영향을 미치는지 설명합니다.
6. 샘플 품질, 학습 안정성, 추론 속도, 우도 추정 측면에서 DDPM과 GAN, VAE를 비교합니다.

---

## 개요

디노이징 확산 확률 모델(Denoising Diffusion Probabilistic Models, DDPM)은 점진적인 노이즈 추가 과정을 역전시켜 데이터를 생성하는 강력한 생성 모델입니다. "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)

---

## 수학적 배경

### 1. 순방향 확산 과정(Forward Diffusion Process)

```
목표: 데이터 x₀에 점진적으로 가우시안 노이즈 추가

q(xₜ|xₜ₋₁) = N(xₜ; √(1-βₜ)xₜ₋₁, βₜI)

여기서:
- x₀: 원본 데이터
- xₜ: 타임스텝 t에서의 노이즈가 있는 데이터
- βₜ: 노이즈 스케줄 (β₁, ..., βₜ)
- T: 전체 타임스텝 (일반적으로 1000)

닫힌 형식(Closed form) (αₜ = 1 - βₜ, ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ 사용):
q(xₜ|x₀) = N(xₜ; √ᾱₜ x₀, (1-ᾱₜ)I)

xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε,  ε ~ N(0, I)

t → T일 때: xₜ → N(0, I) (순수 노이즈)
```

### 2. 역방향 확산 과정(Reverse Diffusion Process)

```
목표: 디노이징 p(xₜ₋₁|xₜ) 학습

실제 사후 분포(Intractable):
q(xₜ₋₁|xₜ, x₀) = N(xₜ₋₁; μ̃ₜ(xₜ, x₀), β̃ₜI)

여기서:
μ̃ₜ(xₜ, x₀) = (√ᾱₜ₋₁ βₜ)/(1-ᾱₜ) x₀ + (√αₜ(1-ᾱₜ₋₁))/(1-ᾱₜ) xₜ
β̃ₜ = (1-ᾱₜ₋₁)/(1-ᾱₜ) · βₜ

학습된 역방향 과정:
pθ(xₜ₋₁|xₜ) = N(xₜ₋₁; μθ(xₜ, t), Σθ(xₜ, t))

단순화: 평균 대신 노이즈 ε 예측
εθ(xₜ, t) ≈ ε
```

### 3. 학습 목적 함수(Training Objective)

```
변분 하한(Variational Lower Bound, ELBO):
L = Eₜ,x₀,ε[||ε - εθ(xₜ, t)||²]

여기서:
- t ~ Uniform(1, T)
- x₀ ~ q(x₀)
- ε ~ N(0, I)
- xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε

예측된 노이즈에 대한 단순한 MSE 손실!

┌─────────────────────────────────────────┐
│  학습:                                  │
│  1. x₀, t, ε 샘플링                     │
│  2. xₜ = √ᾱₜ x₀ + √(1-ᾱₜ) ε 생성       │
│  3. ε̂ = εθ(xₜ, t) 예측                 │
│  4. 손실 = ||ε - ε̂||²                  │
└─────────────────────────────────────────┘
```

### 4. 샘플링(생성, Sampling/Generation)

```
xₜ ~ N(0, I)에서 시작

t = T, T-1, ..., 1에 대해:
    z ~ N(0, I) (t > 1일 때), 그렇지 않으면 z = 0

    ε̂ = εθ(xₜ, t)

    xₜ₋₁ = 1/√αₜ (xₜ - (1-αₜ)/√(1-ᾱₜ) ε̂) + σₜz

여기서:
σₜ = √β̃ₜ 또는 √βₜ (분산 스케줄)

최종: x₀가 생성된 샘플
```

---

## DDPM 아키텍처

### 시간 임베딩을 갖는 UNet(UNet with Time Embedding)

```
시간 임베딩(Sinusoidal Positional Encoding):
t (스칼라)
    ↓
PE(t, dim) = [sin(t/10000^(0/d)), cos(t/10000^(0/d)),
              sin(t/10000^(2/d)), cos(t/10000^(2/d)), ...]
    ↓
Linear(dim→4*dim) + SiLU + Linear(4*dim→4*dim)
    ↓
time_emb (공간 차원으로 브로드캐스트)


UNet 구조 (예: 32×32×3 이미지):

입력 xₜ (32×32×3) + time_emb
    ↓
┌─────────────────────────────────────────┐
│  인코더 (다운샘플링)                    │
├─────────────────────────────────────────┤
│ Conv(3→64) + TimeEmb + ResBlock         │ → skip1
│     ↓ Downsample                        │
│ Conv(64→128) + TimeEmb + ResBlock       │ → skip2
│     ↓ Downsample                        │
│ Conv(128→256) + TimeEmb + ResBlock      │ → skip3
│     ↓ Downsample                        │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  병목층(Bottleneck)                     │
│  Conv(256→512) + Attention + ResBlock   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  디코더 (업샘플링)                      │
├─────────────────────────────────────────┤
│     ↑ Upsample + Concat(skip3)          │
│ Conv(512+256→256) + TimeEmb + ResBlock  │
│     ↑ Upsample + Concat(skip2)          │
│ Conv(256+128→128) + TimeEmb + ResBlock  │
│     ↑ Upsample + Concat(skip1)          │
│ Conv(128+64→64) + TimeEmb + ResBlock    │
└─────────────────────────────────────────┘
    ↓
Conv(64→3) + GroupNorm
    ↓
출력 εθ(xₜ, t) (32×32×3)
```

### 시간 임베딩을 갖는 ResBlock(ResBlock with Time Embedding)

```
x, time_emb → ResBlock → out

┌─────────────────────────────────────────┐
│  GroupNorm → SiLU → Conv                │
│       ↓                                 │
│  + time_emb (브로드캐스트)              │
│       ↓                                 │
│  GroupNorm → SiLU → Conv                │
│       ↓                                 │
│  + skip connection (프로젝션 포함)      │
└─────────────────────────────────────────┘
```

---

## 노이즈 스케줄(Noise Schedule)

### 선형 스케줄(Linear Schedule)

```python
# 선형 스케줄 (Ho et al., 2020)
β₁ = 1e-4
βₜ = 0.02
βₜ = linear_interpolate(β₁, βₜ, t/T)

# 효율성을 위한 사전 계산
αₜ = 1 - βₜ
ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ
√ᾱₜ, √(1-ᾱₜ)  # 순방향 과정에서 사용
```

### 코사인 스케줄(개선된 버전, Cosine Schedule - Improved)

```python
# 코사인 스케줄 (Nichol & Dhariwal, 2021)
s = 0.008
f(t) = cos²((t/T + s)/(1 + s) · π/2)
ᾱₜ = f(t) / f(0)
βₜ = 1 - αₜ/αₜ₋₁

# 더 부드러운 노이즈 스케줄, 고해상도에 더 적합
```

---

## 파일 구조

```
13_Diffusion/
├── README.md
├── pytorch_lowlevel/
│   ├── ddpm_mnist.py         # MNIST용 DDPM (28×28)
│   └── ddpm_cifar.py         # CIFAR-10용 DDPM (32×32)
├── paper/
│   ├── ddpm_paper.py         # 전체 DDPM 구현
│   ├── ddim_sampling.py      # DDIM 빠른 샘플링
│   └── cosine_schedule.py    # 개선된 노이즈 스케줄
└── exercises/
    ├── 01_noise_schedule.md  # 노이즈 스케줄 시각화
    └── 02_sampling_steps.md  # DDPM vs DDIM 비교
```

---

## 핵심 개념

### 1. DDPM vs DDIM 샘플링

```
DDPM (Ho et al., 2020):
- 확률적 샘플링(각 단계에서 노이즈 z 추가)
- T 단계 필요 (예: 1000 단계)
- 고품질이지만 느림

DDIM (Song et al., 2020):
- 결정적 샘플링 (z = 0)
- 타임스텝 건너뛰기: 부분집합 사용 [τ₁, τ₂, ..., τₛ]
- 10-50배 빠름 (예: 50 단계)
- 품질 약간 저하

DDIM 업데이트:
xₜ₋₁ = √ᾱₜ₋₁ x̂₀ + √(1-ᾱₜ₋₁) εθ(xₜ, t)

여기서 x̂₀ = (xₜ - √(1-ᾱₜ)εθ(xₜ, t))/√ᾱₜ
```

### 2. 분류기 가이던스(Classifier Guidance)

```
목표: 클래스 y에 조건화된 샘플 생성

조건부 스코어:
∇ₓ log p(xₜ|y) ≈ ∇ₓ log p(xₜ) + s·∇ₓ log p(y|xₜ)
                  ─────────────   ─────────────────
                  무조건부         분류기 그래디언트

가이드된 노이즈 예측:
ε̂ = εθ(xₜ, t) - s·√(1-ᾱₜ)·∇ₓ log pφ(y|xₜ)

s: 가이던스 스케일 (s > 1 → 더 강한 조건화)
```

### 3. 분류기 프리 가이던스(Classifier-Free Guidance)

```
별도의 분류기 불필요!

조건부와 무조건부 모두 처리하도록 모델 학습:
εθ(xₜ, t, c) (확률 p로)
εθ(xₜ, t, ∅) (확률 1-p로) (∅ = 널 클래스)

가이드된 예측:
ε̂ = εθ(xₜ, t, ∅) + w·(εθ(xₜ, t, c) - εθ(xₜ, t, ∅))

w: 가이던스 가중치 (w=0 → 무조건부, w>1 → 더 강함)

사용처: Stable Diffusion, DALL-E 2, Imagen
```

### 4. 학습 팁

```
1. EMA (지수 이동 평균, Exponential Moving Average):
   - θ_ema = 0.9999·θ_ema + 0.0001·θ 유지
   - 샘플링에 θ_ema 사용

2. 점진적 학습(Progressive Training):
   - 작은 해상도로 시작
   - 점진적으로 증가 (8×8 → 16×16 → 32×32)

3. 데이터 증강:
   - 무작위 수평 뒤집기
   - [-1, 1]로 정규화

4. 학습률:
   - MNIST/CIFAR: 2e-4
   - 고해상도: 1e-4

5. 배치 크기:
   - 작은 이미지: 128-256
   - 큰 이미지: 32-64
```

---

## 구현 레벨

### 레벨 2: PyTorch 로우레벨 (pytorch_lowlevel/)
- 순방향/역방향 확산 구현
- 노이즈 스케줄(선형) 구현
- 시간 임베딩이 있는 UNet 구축
- MNIST (28×28) 및 CIFAR-10 (32×32)에서 학습

### 레벨 3: 논문 구현 (paper/)
- 코사인 스케줄을 갖는 전체 DDPM
- DDIM 샘플링 (빠른 추론)
- 분류기 프리 가이던스
- FID/IS 평가 메트릭

---

## 학습 루프

```python
# 의사코드
for epoch in epochs:
    for x0, _ in dataloader:
        # 무작위 타임스텝 샘플링
        t = torch.randint(1, T+1, (batch_size,))

        # 노이즈 샘플링
        noise = torch.randn_like(x0)

        # 순방향 확산: 노이즈가 있는 이미지 생성
        xt = sqrt_alpha_bar[t] * x0 + sqrt_one_minus_alpha_bar[t] * noise

        # 노이즈 예측
        noise_pred = model(xt, t)

        # MSE 손실
        loss = F.mse_loss(noise_pred, noise)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 샘플링 루프

```python
# DDPM 샘플링
x = torch.randn(batch_size, 3, 32, 32)  # 노이즈에서 시작

for t in reversed(range(1, T+1)):
    # 노이즈 예측
    t_batch = torch.full((batch_size,), t)
    noise_pred = model(x, t_batch)

    # 평균 계산
    alpha_t = alpha[t]
    alpha_bar_t = alpha_bar[t]
    mean = (x - (1 - alpha_t) / sqrt(1 - alpha_bar_t) * noise_pred) / sqrt(alpha_t)

    # 노이즈 추가 (마지막 단계 제외)
    if t > 1:
        noise = torch.randn_like(x)
        sigma_t = sqrt(beta[t])
        x = mean + sigma_t * noise
    else:
        x = mean

# x는 생성된 이미지
```

---

## 학습 체크리스트

- [ ] 순방향 확산 닫힌 형식 이해
- [ ] ELBO에서 역방향 확산 유도
- [ ] 노이즈 스케줄 구현 (선형, 코사인)
- [ ] 시간 임베딩이 있는 UNet 구축
- [ ] DDPM vs DDIM 샘플링 이해
- [ ] 분류기 프리 가이던스 구현
- [ ] 평가를 위한 FID 스코어 계산

---

## 참고 문헌

- Ho et al. (2020). "Denoising Diffusion Probabilistic Models"
- Song et al. (2020). "Denoising Diffusion Implicit Models"
- Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models"
- Ho & Salimans (2022). "Classifier-Free Diffusion Guidance"
- [Diffusion Models](./32_Diffusion_Models.md)
