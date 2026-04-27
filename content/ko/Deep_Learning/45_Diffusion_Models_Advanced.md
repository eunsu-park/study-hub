[이전: 테스트 시간 적응](./44_Test_Time_Adaptation.md) | [다음: 상태 공간 모델](./46_State_Space_Models.md)

---

# 45. 확산 모델 심화(Diffusion Models — Advanced Topics)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 분류기 없는 가이던스(Classifier-Free Guidance)와 샘플 품질 대 다양성에 미치는 영향을 설명할 수 있다
2. DDIM 및 생성 단계를 줄이는 기타 가속 샘플링 방법을 설명할 수 있다
3. 잠재 확산(Latent Diffusion) / Stable Diffusion 아키텍처를 전체적으로 이해할 수 있다
4. ControlNet 및 IP-Adapter를 포함한 조건부 생성 기법을 적용할 수 있다
5. 스코어 기반 생성 모델(Score-Based Generative Model)을 SDE 프레임워크와 연결할 수 있다
6. DPM-Solver 등 효율적 샘플러와 미세조정 방법(LoRA, DreamBooth)을 구현할 수 있다
7. 일관성 모델(Consistency Model)과 플로우 매칭(Flow Matching)을 차세대 접근법으로 설명할 수 있다

---

## 목차

1. [분류기 없는 가이던스](#1-분류기-없는-가이던스)
2. [DDIM과 가속 샘플링](#2-ddim과-가속-샘플링)
3. [잠재 확산 모델](#3-잠재-확산-모델)
4. [ControlNet과 조건부 생성](#4-controlnet과-조건부-생성)
5. [스코어 기반 생성 모델 (SDE 관점)](#5-스코어-기반-생성-모델-sde-관점)
6. [DPM-Solver와 효율적 샘플러](#6-dpm-solver와-효율적-샘플러)
7. [미세조정: LoRA, DreamBooth, Textual Inversion](#7-미세조정-lora-dreambooth-textual-inversion)
8. [일관성 모델](#8-일관성-모델)
9. [플로우 매칭](#9-플로우-매칭)
10. [연습문제](#10-연습문제)

## 1. 분류기 없는 가이던스

### 이론: Classifier-Free Guidance

조건부 diffusion은 `x_t`와 조건 `c`(텍스트 프롬프트, 클래스 라벨) 둘 다 주어졌을 때 잡음을 예측:

```
\epsilon_\theta(x_t, t, c)
```

**Classifier guidance** (Dhariwal & Nichol 2021)는 별도 학습된 분류기 `p(c | x_t)`를 사용해 샘플을 조건화 쪽으로 밀음. **Classifier-Free Guidance (CFG)** (Ho & Salimans 2021)는 조건부와 비조건부 사례 둘 다를 처리하는 단일 네트워크를 학습하여 추가 분류기를 회피(학습 중 조건화가 무작위로 null 토큰으로 떨어짐):

```
\epsilon_guided = \epsilon_\theta(x_t, t, \emptyset) + w * (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
```

`w`가 **guidance scale**. `w = 1`이 가이드 안 됨; `w = 0`이 조건 무시; `w > 1`이 조건 방향으로 조건부 예측을 *지나서* 외삽. 더 높은 `w`가 더 충실한("프롬프트에 더 복종하는") 그러나 덜 다양한 샘플을 줌. Stable Diffusion이 일반적으로 `w = 7-10` 사용.

이것이 텍스트-이미지 충실도 뒤의 전체 메커니즘: 학습은 변하지 않지만, 샘플링이 조건부와 비조건부 예측 사이를 외삽.


### 1.1 배경: 분류기 가이던스(Classifier Guidance)

원래의 분류기 가이던스(Dhariwal & Nichol, 2021)에서는 사전 훈련된 분류기 p(y|x_t)가 역방향 과정을 대상 클래스 y 방향으로 유도합니다:

```
가이드된 스코어 = 비조건부 스코어 + s * ∇_{x_t} log p(y | x_t)

여기서 s는 가이던스 스케일
```

**문제점**: 모든 타임스텝에서 노이즈가 있는 입력에 대해 훈련된 별도의 분류기가 필요합니다.

### 1.2 분류기 없는 가이던스(CFG)

Ho & Salimans (2022)는 하나의 모델을 조건부와 비조건부 모드 모두로 훈련하여 외부 분류기를 제거했습니다:

```
훈련 중:
  - 확률 p_uncond (예: 10%)로 조건 신호 c를 드롭
    (널 토큰 ∅으로 대체)
  - 그 외에는 조건 c와 함께 정상 훈련

샘플링 중:
  ε_guided = ε_uncond + w * (ε_cond - ε_uncond)

  여기서 w는 가이던스 스케일 (일반적으로 3-15)
```

```
가이던스 스케일 효과:

w = 1.0: 가이던스 없음 (순수 조건부 모델)
         낮은 품질, 높은 다양성

w = 7.5: 중간 가이던스 (일반적인 기본값)
         품질과 다양성의 좋은 균형

w = 20:  강한 가이던스
         높은 품질, 낮은 다양성 (과포화될 수 있음)
```

### 1.3 PyTorch 구현

```python
import torch
import torch.nn as nn


class CFGDiffusionModel(nn.Module):
    """분류기 없는 가이던스를 지원하는 확산 모델."""

    def __init__(self, base_model, p_uncond=0.1):
        super().__init__()
        self.base_model = base_model  # (x_t, t, cond)를 받는 UNet
        self.p_uncond = p_uncond

    def forward(self, x_t, t, cond):
        """훈련 순전파: 랜덤하게 조건을 드롭."""
        if self.training:
            # 비조건부 훈련을 위해 조건 임베딩을 랜덤하게 0으로 대체
            batch_size = x_t.shape[0]
            mask = torch.rand(batch_size, device=x_t.device) < self.p_uncond
            # 조건 임베딩을 0으로 대체 (널 토큰)
            cond = cond.clone()
            cond[mask] = 0.0
        return self.base_model(x_t, t, cond)

    @torch.no_grad()
    def guided_sample(self, x_t, t, cond, guidance_scale=7.5):
        """분류기 없는 가이던스로 샘플링."""
        # 비조건부 예측
        null_cond = torch.zeros_like(cond)
        eps_uncond = self.base_model(x_t, t, null_cond)

        # 조건부 예측
        eps_cond = self.base_model(x_t, t, cond)

        # 가이드된 예측
        eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        return eps_guided
```

### 1.4 동적 및 재조정 가이던스(Dynamic and Rescaled Guidance)

최신 시스템은 아티팩트를 방지하기 위해 **동적 가이던스**를 사용합니다:

```python
def dynamic_cfg(eps_uncond, eps_cond, guidance_scale, rescale=0.7):
    """과포화 방지를 위한 재조정 CFG (Imagen 스타일)."""
    eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # 색상 과포화 방지를 위한 재조정
    std_guided = eps_guided.std(dim=list(range(1, eps_guided.ndim)), keepdim=True)
    std_cond = eps_cond.std(dim=list(range(1, eps_cond.ndim)), keepdim=True)
    factor = std_cond / (std_guided + 1e-8)
    factor = rescale * factor + (1 - rescale)

    return eps_guided * factor
```

---

## 2. DDIM과 가속 샘플링

### 이론: 더 빠른 샘플링

DDPM의 1000 스텝은 필요한 것보다 훨씬 많음. 여러 방법이 이를 감소:

**DDIM** (Song et al. 2020): 결정적 ODE 극한이 있는 비-마르코프 과정으로 역방향 과정 재정식화. 같은 잡음 예측기, 하지만 각 스텝이 결정적 궤적 따라 훨씬 큰 점프 취함. 50-100 스텝이 DDPM 비교 품질을 줌.

**DPM-Solver / DPM-Solver++** (Lu et al. 2022): 같은 역방향 diffusion ODE의 고차 ODE 해법. 10-20 스텝이면 충분. 대부분 프로덕션 diffusion 모델의 기본값.

**Consistency Models** (Song et al. 2023): 임의의 `(x_t, t)`를 `x_0`로 직접 매핑하는 네트워크 학습, 1-스텝 또는 적은-스텝 샘플링 허용. 50-스텝 DDIM보다 약간 낮은 품질이지만 50배 빠름.

패턴: 같은 학습된 모델, 다른 추론 알고리즘. 이 분리가 GAN(생성기와 샘플러가 분리 불가) 대비 diffusion의 가장 큰 실용적 이점 중 하나.


### 2.1 DDPM 샘플링의 문제점

DDPM은 생성에 T 단계(일반적으로 1000)가 필요하여 매우 느립니다:

```
DDPM: x_T → x_{T-1} → x_{T-2} → ... → x_1 → x_0   (1000 NFE)

NFE = 함수 평가 횟수(Function Evaluations, 신경망 순전파 횟수)
각 단계마다 UNet 순전파 1회 필요 (512x512에서 A100 기준 ~0.1초)
총 소요: 이미지 1장당 ~100초
```

### 2.2 DDIM: 잡음 제거 확산 암시적 모델(Denoising Diffusion Implicit Models)

Song et al. (2021)은 DDPM의 순방향 과정을 **비마르코프(non-Markovian)** 과정으로 일반화하여 더 적은 단계로 결정론적 샘플링을 가능하게 했습니다:

```
DDIM 업데이트 규칙:

x_{t-1} = √(ᾱ_{t-1}) * predicted_x0
         + √(1 - ᾱ_{t-1} - σ²_t) * predicted_direction
         + σ_t * noise

여기서:
  predicted_x0 = (x_t - √(1 - ᾱ_t) * ε_θ(x_t, t)) / √(ᾱ_t)
  predicted_direction = ε_θ(x_t, t)
  σ_t = 0이면 결정론적 샘플링 (DDIM)
  σ_t = √(β̃_t)이면 확률적 샘플링 (DDPM)
```

### 2.3 DDIM 구현

```python
import torch
import numpy as np


class DDIMSampler:
    """설정 가능한 단계 수를 가진 DDIM 샘플러."""

    def __init__(self, model, num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.model = model
        self.num_train_timesteps = num_train_timesteps

        # 노이즈 스케줄 사전 계산
        betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)

    def get_timestep_subsequence(self, num_inference_steps):
        """훈련 스케줄에서 균등 간격의 타임스텝을 선택."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
        return torch.from_numpy(timesteps).flip(0)  # 역순: T -> 0

    @torch.no_grad()
    def sample(self, shape, num_inference_steps=50, eta=0.0, cond=None,
               guidance_scale=7.5):
        """
        DDIM을 사용한 샘플 생성.

        Args:
            shape: (B, C, H, W) 출력 형태
            num_inference_steps: 잡음 제거 단계 수 (예: 20-50)
            eta: 0.0 = 결정론적 DDIM, 1.0 = DDPM과 동등
            cond: 조건 신호 (텍스트 임베딩, 클래스 레이블 등)
            guidance_scale: CFG 스케일
        """
        device = next(self.model.parameters()).device
        timesteps = self.get_timestep_subsequence(num_inference_steps).to(device)

        # 순수 노이즈에서 시작
        x_t = torch.randn(shape, device=device)

        for i, t in enumerate(timesteps):
            t_batch = t.expand(shape[0])

            # 모델 예측 (선택적 CFG 포함)
            if cond is not None and guidance_scale > 1.0:
                eps_pred = self.model.guided_sample(
                    x_t, t_batch, cond, guidance_scale
                )
            else:
                eps_pred = self.model(x_t, t_batch, cond)

            # DDIM 업데이트
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else torch.tensor(1.0)
            )

            # x_0 예측
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(-1, 1)  # 안정성을 위한 클리핑

            # 확률성을 위한 시그마 계산
            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                * (1 - alpha_bar_t / alpha_bar_prev)
            )

            # x_t 방향
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma_t**2) * eps_pred

            # DDIM 단계
            x_t = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt

            if sigma_t > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + sigma_t * noise

        return x_t
```

### 2.4 단계 수 대 품질

```
단계:   품질 (FID↓):   시간 (512x512, A100):
1000    ~3.2              ~100초   (DDPM 기준선)
 200    ~3.5              ~20초
  50    ~4.0              ~5초     (DDIM 최적점)
  20    ~5.5              ~2초
  10    ~12.0             ~1초     (눈에 띄는 품질 저하)
   1    ~50+              ~0.1초   (일관성 증류 필요)
```

---

## 3. 잠재 확산 모델

### 이론: Latent Diffusion / Stable Diffusion

전체 이미지 해상도(예: 512x512x3 = 786k 차원)에서 diffusion하는 것은 계산적으로 극단. **Latent Diffusion** (Rombach et al. 2022)은 문제를 인수분해:

1. 이미지를 작은 잠재 공간으로 압축하는 **VAE** 학습 (예: 64x64x4).
2. 픽셀 공간이 아닌 **잠재 공간에서 diffusion** 학습.
3. 생성 시: 잠재 샘플, 그 다음 VAE를 통해 픽셀로 디코딩.

잠재 공간이 지각적으로 무관한 디테일(고주파 텍스처)을 버리지만 의미 구조를 유지. 잠재 공간의 diffusion이 픽셀 해상도보다 8-32배 빠르며, 비교 가능한 시각 품질. 이것이 *실용적* 고해상도 텍스트-이미지 생성을 가능하게 한 것 — Stable Diffusion은 본질적으로 "latent diffusion + CLIP 텍스트 조건화 + CFG."


### 3.1 동기: 픽셀 공간은 비용이 비싸다

고해상도 이미지에서 픽셀 공간의 확산을 실행하는 것은 계산적으로 매우 비쌉니다:

```
픽셀 공간 (256×256×3):  196,608 차원
잠재 공간 (32×32×4):    4,096 차원     (48배 압축)

픽셀 공간 (512×512×3):  786,432 차원
잠재 공간 (64×64×4):    16,384 차원    (48배 압축)
```

### 3.2 Stable Diffusion 아키텍처

```
Stable Diffusion 파이프라인:

텍스트 프롬프트 ──► CLIP 텍스트 인코더 ──► 텍스트 임베딩 (77×768)
                                            │
                                            ▼
랜덤 노이즈 ──► ┌──────────────────────────────────┐
  (64×64×4)      │  U-Net (잠재 공간에서)            │
                 │  - 텍스트 조건을 위한 교차 어텐션    │
                 │  - 공간을 위한 자기 어텐션           │
                 │  - 특징을 위한 ResNet 블록          │ × N 잡음제거 단계
                 └──────────────────────────────────┘
                                            │
                                            ▼
                 잡음 제거된 잠재 (64×64×4)
                                            │
                                            ▼
                 VAE 디코더 ──► 이미지 (512×512×3)
```

### 3.3 핵심 구성 요소

```python
class LatentDiffusionModel(nn.Module):
    """간략화된 잠재 확산 모델(LDM) 아키텍처."""

    def __init__(self, vae, unet, text_encoder, tokenizer, scheduler):
        super().__init__()
        self.vae = vae              # 사전 훈련된 VAE (인코더 + 디코더)
        self.unet = unet            # 잠재 공간의 조건부 U-Net
        self.text_encoder = text_encoder  # CLIP 텍스트 인코더
        self.tokenizer = tokenizer
        self.scheduler = scheduler  # DDIM, DPM-Solver 등
        self.vae_scale_factor = 0.18215  # VAE 잠재의 스케일링 팩터

    def encode_prompt(self, prompt):
        """텍스트 프롬프트를 임베딩으로 인코딩."""
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt"
        )
        text_embeddings = self.text_encoder(tokens.input_ids)[0]
        return text_embeddings

    def encode_image(self, image):
        """VAE를 통해 이미지를 잠재 공간으로 인코딩."""
        latents = self.vae.encode(image).latent_dist.sample()
        return latents * self.vae_scale_factor

    def decode_latents(self, latents):
        """VAE를 통해 잠재를 픽셀 공간으로 디코딩."""
        latents = latents / self.vae_scale_factor
        image = self.vae.decode(latents).sample
        return image

    @torch.no_grad()
    def generate(self, prompt, num_inference_steps=50, guidance_scale=7.5,
                 height=512, width=512):
        """전체 텍스트-이미지 생성 파이프라인."""
        device = self.unet.device

        # 1. 텍스트 인코딩
        text_emb = self.encode_prompt(prompt).to(device)
        uncond_emb = self.encode_prompt("").to(device)
        text_emb = torch.cat([uncond_emb, text_emb])  # CFG용

        # 2. 잠재 노이즈 초기화
        latents = torch.randn(
            (1, 4, height // 8, width // 8), device=device
        )

        # 3. 스케줄러 설정
        self.scheduler.set_timesteps(num_inference_steps)

        # 4. 잡음 제거 루프
        for t in self.scheduler.timesteps:
            latent_input = torch.cat([latents] * 2)  # CFG용
            noise_pred = self.unet(latent_input, t, encoder_hidden_states=text_emb).sample

            # 분류기 없는 가이던스
            noise_uncond, noise_cond = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # 스케줄러 단계
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5. 잠재를 이미지로 디코딩
        image = self.decode_latents(latents)
        return image
```

### 3.4 교차 어텐션 메커니즘(Cross-Attention)

UNet에서 텍스트 조건화의 핵심:

```python
class CrossAttention(nn.Module):
    """공간 특징과 텍스트 임베딩 간의 교차 어텐션."""

    def __init__(self, query_dim, context_dim=768, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context):
        """
        Args:
            x: 공간 특징 (B, H*W, D)
            context: 텍스트 임베딩 (B, 77, 768)
        """
        B, N, _ = x.shape
        h = self.heads

        q = self.to_q(x).view(B, N, h, -1).transpose(1, 2)
        k = self.to_k(context).view(B, -1, h, k.shape[-1] // h).transpose(1, 2)
        v = self.to_v(context).view(B, -1, h, v.shape[-1] // h).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)
```

---

## 4. ControlNet과 조건부 생성

### 4.1 ControlNet 아키텍처

ControlNet(Zhang et al., 2023)은 사전 훈련된 확산 모델에 공간 조건(엣지, 깊이, 포즈)을 추가합니다:

```
ControlNet 아키텍처:

                    ┌──────────────────┐
입력 조건 ────────► │ 학습 가능한 복사본│
(예: Canny 엣지)    │ UNet 인코더의     │
                    │ (원본 동결        │
                    │  + 제로 컨볼루션) │
                    └────────┬─────────┘
                             │ 잔차(residuals)
                             ▼
                    ┌──────────────────┐
노이즈 잠재 ──────► │ 동결된 원본       │ ──► 잡음 제거된 출력
+ 텍스트 조건       │ UNet              │
                    └──────────────────┘
```

### 4.2 제로 컨볼루션(Zero Convolution)

핵심 혁신: 새로운 연결을 제로 가중치로 초기화하여 훈련이 사전 훈련된 모델에서 정확히 시작되도록 합니다:

```python
class ZeroConv(nn.Module):
    """안정적인 ControlNet 훈련을 위한 제로 초기화 컨볼루션."""

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 1)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return self.conv(x)


class ControlNetBlock(nn.Module):
    """간략화된 ControlNet 블록."""

    def __init__(self, frozen_unet_encoder_block):
        super().__init__()
        # 동결된 인코더 블록의 학습 가능한 복사본
        self.trainable_copy = copy.deepcopy(frozen_unet_encoder_block)
        # 출력을 위한 제로 컨볼루션
        self.zero_conv = ZeroConv(self.trainable_copy.out_channels)

    def forward(self, control_input):
        h = self.trainable_copy(control_input)
        return self.zero_conv(h)
```

### 4.3 조건 유형

```
조건 유형           입력 예시              용도
─────────────────────────────────────────────────────
Canny 엣지         엣지 맵               구조 보존
깊이 맵(Depth Map)  MiDaS 깊이           장면 레이아웃
OpenPose           골격 키포인트          인체 포즈 제어
세그멘테이션        의미론적 마스크        영역 기반 제어
스크리블(Scribble)  사용자 그림           스케치→이미지
노말 맵            표면 법선             3D 인식 생성
IP-Adapter         참조 이미지           스타일/정체성 전이
T2I-Adapter        경량 조건             효율적 조건화
```

---

## 5. 스코어 기반 생성 모델 (SDE 관점)

### 이론: Score SDE, ControlNet, LoRA

**Score SDE 관점** (Song et al. 2021)은 DDPM/DDIM/DPM-Solver를 하나의 확률 미분 방정식 프레임워크 하에 통합. 역방향 과정은 같은 SDE; 다른 샘플러는 다른 수치 해법.

**ControlNet** (Zhang et al. 2023)은 추가 입력(에지 맵, 자세, 깊이)에 조건화된 diffusion U-Net의 encoder 분기 복사본 추가. 복사본은 원래 encoder에서 초기화되고 0-초기화된 `1x1` conv를 통해 decoder에 연결(초기 학습이 원본과 정확히 같이 동작하도록). 생성의 미세한 공간 제어 허용.

**LoRA** (Hu et al. 2021, 2023년 diffusion에 적용): 모든 파라미터 파인튜닝 대신, `A in R^{d x r}, B in R^{r x d}`이고 `r << d`인 저순위 업데이트 `\Delta W = A B` 학습. A와 B의 작은 `(d * r)` 파라미터만 학습; 나머지는 동결. 빠른 개인화 허용(예: 100 이미지에서 특정 주제로 Stable Diffusion 파인튜닝).


### 5.1 통합 프레임워크

Song et al. (2021)은 DDPM과 스코어 매칭 모두 확률 미분 방정식(SDE)으로 통합할 수 있음을 보였습니다:

```
순방향 SDE (노이즈 추가):
  dx = f(x, t) dt + g(t) dw

  여기서:
    f(x, t) = 드리프트 계수
    g(t)    = 확산 계수
    dw      = 위너 과정 (브라운 운동)

역방향 SDE (노이즈 제거):
  dx = [f(x, t) - g(t)² ∇_x log p_t(x)] dt + g(t) dw̄

  여기서:
    ∇_x log p_t(x) = 스코어 함수 (네트워크가 학습하는 것)
    dw̄ = 역방향 위너 과정
```

### 5.2 두 가지 정규적 SDE

```
분산 폭발(Variance Exploding, VE-SDE):     SMLD / NCSN에 해당
  f(x, t) = 0
  g(t) = σ(t) * √(2 log(σ_max/σ_min))

분산 보존(Variance Preserving, VP-SDE):     DDPM에 해당
  f(x, t) = -½ β(t) x
  g(t) = √β(t)
```

### 5.3 확률 흐름 ODE(Probability Flow ODE)

핵심 통찰: 역방향 SDE에는 **결정론적** ODE 대응물이 있습니다(노이즈 항 없음):

```
확률 흐름 ODE:
  dx/dt = f(x, t) - ½ g(t)² ∇_x log p_t(x)

장점:
  - 결정론적: 같은 노이즈 → 같은 이미지 (편집에 유용)
  - 정확한 우도 계산 가능
  - 빠른 ODE 솔버 사용 가능 (오일러-마루야마뿐 아닌)
```

```python
from scipy.integrate import solve_ivp


def probability_flow_ode(score_model, x_T, t_start=1.0, t_end=0.0,
                          beta_min=0.1, beta_max=20.0):
    """결정론적 샘플링을 위한 확률 흐름 ODE 풀이."""

    def drift_fn(t, x_flat):
        x = torch.tensor(x_flat, dtype=torch.float32).reshape(1, *shape)
        t_tensor = torch.tensor([t], dtype=torch.float32)

        beta_t = beta_min + t * (beta_max - beta_min)
        with torch.no_grad():
            score = score_model(x, t_tensor)

        drift = -0.5 * beta_t * x - 0.5 * beta_t * score
        return drift.flatten().numpy()

    shape = x_T.shape[1:]
    solution = solve_ivp(
        drift_fn,
        t_span=(t_start, t_end),
        y0=x_T.flatten().numpy(),
        method='RK45',
        rtol=1e-5, atol=1e-5
    )
    return torch.tensor(solution.y[:, -1]).reshape(1, *shape)
```

---

## 6. DPM-Solver와 효율적 샘플러

### 6.1 빠른 샘플러 개요

```
샘플러          좋은 품질 위한 단계   유형            핵심 아이디어
──────────────────────────────────────────────────────────────────
DDPM             1000                확률적          원래 SDE 이산화
DDIM             50                  결정론적        비마르코프 단계 건너뛰기
PNDM             50                  결정론적        의사 수치 방법
DPM-Solver       20                  결정론적        ODE의 정확한 풀이
DPM-Solver++     15-20               양쪽 모두       다단계 + 임계처리
UniPC            10-15               결정론적        통합 예측-교정기
Euler Ancestral  25-30               확률적          오일러 방법 + 노이즈 주입
```

### 6.2 DPM-Solver: 정확한 확산 ODE 솔버

Lu et al. (2022)은 변수 변환 공식을 사용하여 확산 ODE의 **정확한** 풀이를 도출했습니다:

```python
class DPMSolverSecondOrder:
    """간략화된 DPM-Solver-2 (2차 솔버)."""

    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def lambda_t(self, t):
        """로그 신호-대-잡음 비."""
        alpha_bar = self.alphas_cumprod[t]
        return 0.5 * torch.log(alpha_bar / (1 - alpha_bar))

    def predict_x0(self, x_t, t, eps_pred):
        """노이즈 예측에서 x_0를 예측."""
        alpha_bar = self.alphas_cumprod[t]
        return (x_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)

    @torch.no_grad()
    def step(self, x_t, t, t_prev, t_mid=None):
        """DPM-Solver-2 한 단계 (2차)."""
        eps_t = self.model(x_t, t)
        x0_pred = self.predict_x0(x_t, t, eps_t)

        if t_mid is not None:
            # 2차: 중간점 사용
            lambda_t = self.lambda_t(t)
            lambda_mid = self.lambda_t(t_mid)
            lambda_prev = self.lambda_t(t_prev)

            h = lambda_prev - lambda_t
            h_mid = lambda_mid - lambda_t
            r = h_mid / h

            # 중간점에서의 1차 추정
            alpha_mid = self.alphas_cumprod[t_mid]
            sigma_mid = torch.sqrt(1 - alpha_mid)
            x_mid = (
                torch.sqrt(alpha_mid / self.alphas_cumprod[t]) * x_t
                - sigma_mid * (torch.exp(-h_mid) - 1) * eps_t
            )

            # 2차 보정
            eps_mid = self.model(x_mid, t_mid)
            alpha_prev = self.alphas_cumprod[t_prev]
            sigma_prev = torch.sqrt(1 - alpha_prev)

            x_prev = (
                torch.sqrt(alpha_prev / self.alphas_cumprod[t]) * x_t
                - sigma_prev * (torch.exp(-h) - 1) * eps_t
                - sigma_prev * (0.5 / r) * (torch.exp(-h) - 1) * (eps_mid - eps_t)
            )
            return x_prev
        else:
            # 1차 폴백
            alpha_prev = self.alphas_cumprod[t_prev]
            sigma_prev = torch.sqrt(1 - alpha_prev)
            h = self.lambda_t(t_prev) - self.lambda_t(t)
            x_prev = (
                torch.sqrt(alpha_prev / self.alphas_cumprod[t]) * x_t
                - sigma_prev * (torch.exp(-h) - 1) * eps_t
            )
            return x_prev
```

---

## 7. 미세조정: LoRA, DreamBooth, Textual Inversion

### 7.1 LoRA (Low-Rank Adaptation, 저랭크 적응)

LoRA는 동결된 어텐션 레이어에 학습 가능한 저랭크 행렬을 주입합니다:

```
원래 가중치:  W ∈ R^{d×d}     (동결)
LoRA 업데이트: ΔW = BA          여기서 B ∈ R^{d×r}, A ∈ R^{r×d}
유효 가중치:  W' = W + α * BA  (α = 스케일링 팩터, r << d)

일반적인 랭크 r = 4-64 (Stable Diffusion UNet에서 d = 320-1280)
파라미터: 2 * d * r vs d * d  →  ~100배 적은 학습 가능 파라미터
```

```python
class LoRALinear(nn.Module):
    """LoRA 적응 선형 레이어."""

    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)

        d_out, d_in = original_linear.weight.shape
        self.lora_A = nn.Parameter(torch.randn(rank, d_in) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        self.scale = alpha / rank

    def forward(self, x):
        original_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return original_out + self.scale * lora_out


def inject_lora(unet, rank=4, alpha=1.0, target_modules=("to_q", "to_v")):
    """UNet의 어텐션 레이어에 LoRA 주입."""
    for name, module in unet.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0]
                attr_name = name.rsplit(".", 1)[1]
                parent = dict(unet.named_modules())[parent_name]
                setattr(parent, attr_name, LoRALinear(module, rank, alpha))
    return unet
```

### 7.2 DreamBooth

고유 식별자를 사용하여 3-5장의 이미지로 전체 모델을 미세조정합니다:

```
훈련:
  1. 희귀 토큰 선택: "a photo of [V] dog"
  2. 3-5장의 피사체 이미지로 UNet + 텍스트 인코더 미세조정
  3. 언어 드리프트 방지를 위한 사전 보존 손실(prior preservation loss) 적용

사전 보존 손실:
  L = L_diffusion(피사체 이미지, "[V] dog")
    + λ * L_diffusion(클래스 이미지, "dog")

  여기서 클래스 이미지는 동결된 모델이 생성
```

```python
def dreambooth_training_step(model, vae, noise_scheduler, text_encoder,
                              subject_batch, class_batch,
                              subject_prompt, class_prompt,
                              prior_weight=1.0):
    """사전 보존이 포함된 DreamBooth 훈련 한 단계."""
    # 피사체 손실
    subject_latents = vae.encode(subject_batch).latent_dist.sample() * 0.18215
    noise = torch.randn_like(subject_latents)
    timesteps = torch.randint(0, 1000, (subject_latents.shape[0],), device=subject_latents.device)
    noisy_latents = noise_scheduler.add_noise(subject_latents, noise, timesteps)

    subject_emb = text_encoder(subject_prompt)
    subject_pred = model(noisy_latents, timesteps, subject_emb).sample
    subject_loss = nn.functional.mse_loss(subject_pred, noise)

    # 사전 보존 손실
    class_latents = vae.encode(class_batch).latent_dist.sample() * 0.18215
    class_noise = torch.randn_like(class_latents)
    class_timesteps = torch.randint(0, 1000, (class_latents.shape[0],), device=class_latents.device)
    class_noisy = noise_scheduler.add_noise(class_latents, class_noise, class_timesteps)

    class_emb = text_encoder(class_prompt)
    class_pred = model(class_noisy, class_timesteps, class_emb).sample
    class_loss = nn.functional.mse_loss(class_pred, class_noise)

    return subject_loss + prior_weight * class_loss
```

### 7.3 텍스트 역전(Textual Inversion)

개념을 나타내는 새로운 "단어"(임베딩 벡터)를 학습합니다:

```
접근법:
  1. 전체 모델 동결 (UNet + 텍스트 인코더)
  2. 토큰 [V]에 대한 단일 임베딩 벡터 v*만 최적화
  3. v* ∈ R^{768}  (CLIP 임베딩 차원)
  4. LoRA보다 훨씬 작음: 벡터 하나뿐

훈련: v*에 대해서만 L_diffusion을 최소화
장점: 매우 작은 모델 크기 (~3KB), 어떤 프롬프트와도 조합 가능
한계: DreamBooth/LoRA보다 표현력이 낮음
```

### 7.4 비교

```
방법                학습 가능 파라미터   훈련 데이터    품질    모델 크기
───────────────────────────────────────────────────────────────────────
Textual Inversion   ~768 (벡터 1개)     3-5장         ★★★      ~3KB
LoRA                ~1-10M              다양          ★★★★     ~10-100MB
DreamBooth          ~860M (전체 UNet)    3-5장         ★★★★★    ~2-4GB
DreamBooth+LoRA     ~1-10M              3-5장         ★★★★½    ~10-100MB
```

---

## 8. 일관성 모델

### 8.1 동기

Song et al. (2023)은 ODE 궤적의 어떤 점이든 원점(x_0)으로 직접 매핑하는 **일관성 모델(Consistency Model)**을 제안했습니다:

```
표준 확산 (다단계):
  x_T → x_{T-1} → ... → x_1 → x_0     (여러 단계)

일관성 모델 (단일 단계):
  x_T ─────────────────────────► x_0    (한 단계!)
  x_{T/2} ─────────────────────► x_0    (같은 x_0!)

핵심 속성 (자기 일관성):
  f(x_t, t) = f(x_{t'}, t')  같은 ODE 궤적 위의 임의의 t, t'에 대해
```

### 8.2 훈련 접근법

```
1. 일관성 증류(Consistency Distillation, CD):
   - 사전 훈련된 확산 모델로 시작
   - 자기 일관성 속성을 만족하도록 일관성 모델 훈련
   - ODE 솔버로 같은 궤적 위의 (x_t, x_{t-1}) 쌍을 찾음
   - 손실: ||f(x_{t+1}, t+1) - f̂(x_t, t)||²
     여기서 f̂는 f의 EMA (기울기 정지 타겟)

2. 일관성 훈련(Consistency Training, CT):
   - 처음부터 훈련 (사전 훈련된 모델 불필요)
   - 순방향 과정으로 노이즈 쌍 생성
   - 훈련 중 단계 크기를 점진적으로 줄임
```

```python
class ConsistencyModel(nn.Module):
    """간략화된 일관성 모델."""

    def __init__(self, backbone, sigma_min=0.002, sigma_max=80.0):
        super().__init__()
        self.backbone = backbone
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def skip_scaling(self, sigma):
        """출력 매개변수화: c_skip(σ) x + c_out(σ) F(x, σ)."""
        c_skip = self.sigma_min**2 / (sigma**2 + self.sigma_min**2)
        c_out = sigma * self.sigma_min / torch.sqrt(sigma**2 + self.sigma_min**2)
        return c_skip, c_out

    def forward(self, x, sigma):
        """노이즈 입력을 깨끗한 출력으로 직접 매핑."""
        c_skip, c_out = self.skip_scaling(sigma)
        F = self.backbone(x, sigma)
        return c_skip * x + c_out * F

    @torch.no_grad()
    def single_step_generate(self, z, sigma=80.0):
        """노이즈 z에서 한 단계로 생성."""
        sigma_t = torch.full((z.shape[0],), sigma, device=z.device)
        return self.forward(z, sigma_t)

    @torch.no_grad()
    def multi_step_generate(self, z, sigmas):
        """향상된 품질을 위한 다단계 생성."""
        x = z
        for i, sigma in enumerate(sigmas):
            sigma_t = torch.full((z.shape[0],), sigma, device=z.device)
            x = self.forward(x, sigma_t)
            if i < len(sigmas) - 1:
                # 다음 시그마 수준에서 노이즈 재추가
                noise = torch.randn_like(x)
                x = x + sigmas[i + 1] * noise
        return x
```

---

## 9. 플로우 매칭

### 9.1 핵심 아이디어

플로우 매칭(Lipman et al., 2023)은 스코어 매칭보다 더 간단하고 안정적인 대안을 제공합니다. 스코어 함수를 학습하는 대신, 노이즈를 데이터로 직선 경로를 따라 변환하는 **속도장(velocity field)**을 학습합니다:

```
스코어 매칭 (확산):
  학습: ∇_x log p_t(x)        (스코어 함수)
  SDE:  dx = [f - g² ∇log p] dt + g dw

플로우 매칭:
  학습: v_t(x)                 (속도장)
  ODE:  dx/dt = v_t(x)        (확률적 항 없음!)
  경로: x_t = (1-t) * x_0 + t * x_1    (데이터에서 노이즈로의 직선)
```

### 9.2 조건부 플로우 매칭(Conditional Flow Matching, CFM)

```python
class FlowMatchingModel(nn.Module):
    """최적 운송 경로를 사용한 플로우 매칭."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # 속도 v(x_t, t)를 예측

    def forward(self, x_t, t):
        return self.backbone(x_t, t)

    def compute_loss(self, x_0, x_1=None):
        """
        조건부 플로우 매칭 손실.
        x_0: 데이터 샘플
        x_1: 노이즈 샘플 (None이면 N(0,I)에서 샘플링)
        """
        if x_1 is None:
            x_1 = torch.randn_like(x_0)

        # 랜덤 시간 샘플링
        t = torch.rand(x_0.shape[0], 1, 1, 1, device=x_0.device)

        # 직선 경로를 따른 보간 (최적 운송)
        x_t = (1 - t) * x_0 + t * x_1

        # 타겟 속도: 데이터에서 노이즈 방향
        target_v = x_1 - x_0

        # 예측된 속도
        pred_v = self.forward(x_t, t.squeeze())

        # 단순 MSE 손실
        return nn.functional.mse_loss(pred_v, target_v)

    @torch.no_grad()
    def generate(self, z, num_steps=50):
        """t=1에서 t=0까지 속도장을 적분하여 샘플 생성."""
        dt = -1.0 / num_steps
        x = z

        for i in range(num_steps):
            t = 1.0 - i / num_steps
            t_batch = torch.full((z.shape[0],), t, device=z.device)
            v = self.forward(x, t_batch)
            x = x + v * dt  # 오일러 적분

        return x
```

### 9.3 플로우 매칭의 장점

```
속성              스코어 매칭              플로우 매칭
───────────────────────────────────────────────────────────────
훈련 타겟        스코어 (∇ log p)         속도 (v)
훈련 안정성      불안정할 수 있음          더 안정적
경로 기하학      곡선 (SDE)               직선 (ODE)
우도            ODE 변환 통해             직접 ODE
샘플링          SDE 또는 ODE             ODE만
단계 효율성      20-50 단계 필요           10-20 단계면 충분한 경우가 많음
사용 모델        DDPM, Stable Diffusion   Stable Diffusion 3, Flux
```

### 9.4 정류 흐름(Rectified Flow)

정류 흐름(Liu et al., 2023)은 더 빠른 샘플링을 위해 반복적으로 흐름 경로를 직선화합니다:

```
라운드 1: 플로우 매칭 모델 훈련 → (x_0, x_1) 쌍 생성
라운드 2: "리플로우" — 쌍에 대해 재훈련 → 더 직선적인 경로
라운드 3: 리플로우 반복 → 더욱 직선적

k 라운드의 리플로우 후, 경로가 거의 직선에 가까움
→ 매우 적은 오일러 단계(1-5)로 샘플링 가능
```

---

## 10. 연습문제

### 연습문제 1: 분류기 없는 가이던스 탐구

간단한 2D 분포에서 CFG 실험을 구현하세요:

```python
"""
연습문제 1: 2D 가우시안 혼합에서 분류기 없는 가이던스 구현.

과제:
1. 4개의 다른 클래스(4개 가우시안 클러스터)에 대한
   2D 점을 생성하는 조건부 모델 생성
2. CFG 훈련 구현 (랜덤 조건 드롭아웃)
3. 다양한 가이던스 스케일(w=1, 3, 7, 15)로 샘플 생성
4. 생성된 분포를 플로팅하고 가이던스가 품질 대
   다양성에 미치는 영향 관찰

시작 코드:
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class SimpleDiffusion2D(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=256, p_uncond=0.1):
        super().__init__()
        self.p_uncond = p_uncond
        self.class_embed = nn.Embedding(num_classes + 1, 64)  # +1 널 클래스용
        self.null_class = num_classes

        self.net = nn.Sequential(
            nn.Linear(2 + 64 + 1, hidden_dim),  # x, class_emb, t
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_t, t, class_label):
        # TODO: 훈련 중 랜덤 비조건부 드롭아웃 구현
        pass

    def guided_predict(self, x_t, t, class_label, guidance_scale):
        # TODO: CFG 샘플링 구현
        pass

# TODO: 모델을 훈련하고 다양한 가이던스 스케일에서 결과 시각화
```

### 연습문제 2: DDIM 샘플러

DDIM 샘플링을 구현하고 다양한 단계 수에서 품질을 비교하세요:

```python
"""
연습문제 2: 다양한 단계 수의 DDIM 샘플링.

과제:
1. 사전 훈련된 DDPM 모델이 주어졌을 때 DDIM 샘플링 구현
2. 10, 20, 50, 100, 1000 단계로 이미지 생성
3. 생성 품질 계산 및 비교 (시각적 검사)
4. 각 단계 수의 실시간 시간 측정
5. 결정론적(η=0) 및 확률적(η=1) 모드 모두 구현

예상 관찰:
- 50 단계가 1000과 거의 동일한 품질
- η=0은 결정론적 출력 (같은 노이즈 → 같은 이미지)
- η=1은 확률성 추가 (같은 노이즈 → 다른 이미지)
"""

def ddim_sample(model, alphas_cumprod, shape, num_steps, eta=0.0):
    """
    DDIM 샘플링 구현.

    Args:
        model: 훈련된 노이즈 예측 모델 ε_θ
        alphas_cumprod: (1-β)의 누적곱
        shape: 출력 형태 (B, C, H, W)
        num_steps: 잡음 제거 단계 수
        eta: 확률성 파라미터 (0=결정론적, 1=DDPM)

    Returns:
        생성된 샘플
    """
    # TODO: DDIM 샘플링 구현
    # 1. 타임스텝 부분 수열 계산
    # 2. 랜덤 노이즈에서 시작
    # 3. 각 타임스텝에 대해:
    #    a. 노이즈 예측
    #    b. x_0 예측
    #    c. eta에 기반한 시그마 계산
    #    d. DDIM 업데이트 단계
    pass
```

### 연습문제 3: LoRA 미세조정

처음부터 LoRA를 구현하고 작은 UNet에 적용하세요:

```python
"""
연습문제 3: LoRA를 구현하고 확산 모델을 미세조정.

과제:
1. LoRALinear 모듈을 처음부터 구현
2. 모든 어텐션 레이어에 LoRA를 주입하는 함수 작성
3. 파라미터 수 계산 및 비교 (원본 vs LoRA)
4. 작은 데이터셋(예: 10장)에 LoRA 가중치 훈련
5. LoRA 파라미터만 기울기를 가지는지 확인

답해야 할 질문:
- 랭크 r은 품질과 훈련 속도에 어떤 영향을 미치는가?
- 스케일링 팩터 α의 효과는?
- 어떤 레이어가 LoRA에서 가장 혜택을 받는가 (Q, K, V, 또는 출력)?
"""

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1.0):
        super().__init__()
        # TODO: LoRA 래퍼 구현
        pass

    def forward(self, x):
        # TODO: LoRA를 포함한 순전파 구현
        pass

def count_trainable_params(model):
    """requires_grad=True인 파라미터 수 계산."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# TODO: LoRA 주입 및 파라미터 수 확인
```

### 연습문제 4: 플로우 매칭

2D 데이터셋에서 플로우 매칭을 구현하세요:

```python
"""
연습문제 4: Swiss Roll 데이터셋에서 플로우 매칭.

과제:
1. Swiss Roll 데이터를 타겟 분포로 생성
2. 조건부 플로우 매칭 훈련 구현
3. 속도장 네트워크 훈련
4. t=1에서 t=0까지 ODE를 적분하여 샘플 생성
5. 다양한 단계 수에서 오일러 vs RK4 적분 비교

보너스:
- 정류 흐름 구현 (1 라운드 리플로우)
- 리플로우 전후의 경로 직선성 비교
"""
import torch
from sklearn.datasets import make_swiss_roll

class VelocityField(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 2D 점 + 시간
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, t):
        t = t.unsqueeze(-1) if t.dim() == 1 else t
        inp = torch.cat([x, t], dim=-1)
        return self.net(inp)

def flow_matching_loss(model, x_data):
    """
    조건부 플로우 매칭 손실 계산.
    # TODO: 구현
    """
    pass

def generate_euler(model, num_samples, num_steps=100):
    """
    학습된 속도장의 오일러 적분으로 생성.
    # TODO: 구현
    """
    pass

# TODO: 훈련 및 생성
```

---

**이전**: [테스트 시간 적응](./44_Test_Time_Adaptation.md) | **다음**: [상태 공간 모델](./46_State_Space_Models.md)

---

*레슨 45 끝*
