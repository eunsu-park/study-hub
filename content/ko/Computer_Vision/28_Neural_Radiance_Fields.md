[이전: 파놉틱 세그멘테이션](./27_Panoptic_Segmentation.md)

---

# 28. Neural Radiance Fields (NeRF)

## 학습 목표

이 수업을 완료하면 다음을 수행할 수 있습니다:

1. NeRF 표현 설명: 3D 좌표를 색상과 밀도로 매핑
2. 미분 가능한 레이 마칭을 사용한 볼류메트릭 렌더링 구현
3. 위치 인코딩을 사용한 원본 NeRF 아키텍처 구축
4. 100배 빠른 학습을 위한 Instant-NGP와 해시 인코딩 설명
5. 새로운 시점 합성과 3D 재구성에 NeRF 적용

---

## 목차

참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 암묵적 신경 장면 표현으로서의 NeRF, volume rendering 적분, positional encoding의 역할, 그리고 해시 인코딩된 특징을 통한 Instant-NGP 가속을 다룹니다.

1. [신경 장면 표현](#1-신경-장면-표현)
2. [NeRF 기초](#2-nerf-기초)
3. [볼류메트릭 렌더링](#3-볼류메트릭-렌더링)
4. [NeRF 구현](#4-nerf-구현)
5. [Instant-NGP와 속도 개선](#5-instant-ngp와-속도-개선)
6. [학습 파이프라인](#6-학습-파이프라인)
7. [응용 및 확장](#7-응용-및-확장)
8. [연습문제](#8-연습문제)

---

## 이론과 원리

Neural Radiance Fields(NeRF, Mildenhall 등, 2020)는 3D 장면을 **메시, 포인트 클라우드, voxel 격자가 아니라 신경망으로** 표현합니다 — 3D 좌표와 시점 방향을 색상과 밀도로 매핑하는 신경망. Volume rendering이 그 다음 이 암묵적 표현을 2D 이미지로 변환. 훈련: 렌더링된 이미지를 장면의 실제 사진에 대해 감독. 결과: 장면당 ~50 사진에서 photorealistic novel view synthesis.

핵심 아이디어가 놀라운 이유는 단순하기 때문: <1M 파라미터의 MLP가 디테일한 3D 장면을 충실히 표현할 수 있음. 이 섹션이 왜 그것이 작동하는지 설명.

이 섹션은 다음을 다룹니다:

- **(A) 암묵적 vs 명시적 장면 표현** — 기하 원시 대신 신경망으로 3D를 표현할 때 무엇이 변하는가.
- **(B) NeRF MLP** — `5D 입력 → 4D 출력` 매핑.
- **(C) Volume rendering 적분** — 광선과 밀도/색상 필드에서 픽셀을 렌더링하는 방법.
- **(D) Positional encoding** — 그것 없이 MLP가 고주파 디테일을 학습할 수 없는 이유.
- **(E) 훈련 목표와 뷰 일관성** — photometric 손실과 그것이 작동하는 이유.
- **(F) Instant-NGP** — NeRF를 100× 빠르게 만드는 hash encoding 트릭.

### A. 암묵적 vs 명시적 장면 표현

**명시적**: 장면이 기하 원시 — 메시(정점 + 삼각형), 포인트 클라우드, voxel 격자 — 로 저장. "이 점에 무엇이 있는가?"를 조회로 질의.

**암묵적**: 장면이 함수 `f: ℝ³ → 속성`으로 저장. **함수를 평가**해 질의. `f`가 신경망이면, 장면이 네트워크의 가중치에 저장.

트레이드오프:

- **메모리**: 명시적은 해상도와 함께 성장(고해상도 octree voxel은 폭발); 암묵적은 해상도와 무관하게 고정 파라미터 수.
- **렌더링**: 명시적은 표준 그래픽(rasterization)으로 렌더링; 암묵적은 각 광선을 따라 함수를 여러 번 샘플링 필요.
- **편집**: 명시적은 직접(정점 이동); 암묵적은 어려움(어느 가중치가 이 영역을 제어?).
- **Photorealism**: volume rendering의 암묵적은 명시적 메시가 어려워하는 흐릿한 현상(머리카락, 안개, 반투명) 표현 가능.

NeRF가 암묵적 + volume rendering을 고른 이유는 그 조합이 photorealism을 가장 잘 처리하기 때문.

### B. NeRF MLP

NeRF 함수:

```
F_Θ : (x, y, z, θ, φ)  →  (r, g, b, σ)

입력:  3D 위치 (x, y, z) + 시점 방향 (θ, φ) 단위 벡터로     [5D]
출력: RGB 색상 (r, g, b) + 밀도 σ ≥ 0                      [4D]
```

MLP는 ~8 hidden layer × 256 unit. 두 설계 선택:

- **시점 의존 색상**: 시점 방향이 밀도가 계산된 후 마지막 layer에만 입력됨. 이는 **시점 의존 색상**(specular 반사, Fresnel 등)을 강제하면서 **밀도를 순수 기하적**으로 유지(객체가 시점과 무관하게 거기 있거나 없음).
- **장면 전체에 공유**: 하나의 MLP가 전체 장면을 표현. 네트워크가 ~1M 가중치에 모든 기하와 외관을 학습해야 함.

### C. Volume Rendering 적분

광선 `r(t) = o + t · d`(원점 `o`, 방향 `d`, 파라미터 `t ≥ 0`)이 주어지면, 광선의 색(대응 픽셀에 들어가는 것):

```
C(r) = ∫_{t_n}^{t_f}  T(t) · σ(r(t)) · c(r(t), d) dt

T(t) = exp( -∫_{t_n}^t σ(r(s)) ds )        투과율: 광선이 막히지 않고 t에 도달할 확률
```

직관:

- `σ(x)`가 점 `x`에서의 **밀도** — 높은 밀도 = 광선이 그 점에서 막힐 가능성 높음.
- `c(x, d)`가 방향 `d`에서 봤을 때 `x`에서의 **색상**.
- `T(t)`가 **광선의 빛이 깊이 `t`까지 도달할 만큼 살아남는 양** — 카메라에서 1로 시작해 더 밀집된 영역을 통과하면서 단조 감소.
- 적분은 "광선을 따라 합산, 광선이 아직 흡수되지 않았을 확률로 가중".

구현을 위해 이산화: 광선을 따라 `N` 점 샘플링, 각각에서 MLP 평가, 누적 곱으로 투과율 계산, 합산. 모든 것이 미분 가능 — 기울기가 렌더링 적분을 통해 MLP 가중치로 흐름.

### D. Positional Encoding: 고주파 문제 해결

3D 좌표를 직접 받는 바닐라 MLP는 **흐릿한** 렌더링을 생성 — 자연 장면의 고주파 디테일을 표현할 수 없음. 해결: 각 좌표를 **positional encoding**으로 변환:

```
γ(x) = (sin(2⁰·π·x), cos(2⁰·π·x), sin(2¹·π·x), cos(2¹·π·x), ..., sin(2^{L-1}·π·x), cos(2^{L-1}·π·x))
```

(위치는 L ≈ 10, 방향은 L ≈ 4.)

작동 이유: neural tangent kernel 분석은 MLP가 강한 **저주파 편향**을 가짐을 보여줌 — 자연스럽게 매끄러운 함수를 맞춤. Positional encoding은 고주파 성분을 입력 특징으로 명시적으로 주입, MLP가 이들을 선형으로 결합해 고주파 출력 생성 가능.

이것이 3D 딥러닝에서 가장 중요한 "단순한 트릭" 결과 중 하나 — positional encoding 없이는 NeRF가 작동하지 않음. 같은 트릭이 transformer(rotary / sinusoidal position embedding)에 비슷한 이유로 등장.

### E. 훈련: Photometric 감독

유일한 감독 신호는 **렌더링된 색상이 사진과 일치해야 함**:

```
L = Σ_pixels  ‖ C_rendered(r) - C_real(pixel) ‖²
```

각 훈련 이미지에 대해:

1. 알려진 카메라 내부 파라미터 + 포즈로 각 픽셀을 통과하는 광선 계산.
2. 광선을 따라 점 샘플링.
3. 각 점에서 NeRF MLP 평가.
4. Volume-render해 예측 픽셀 색상 획득.
5. 실제 픽셀 색상에 대해 L2 손실 계산.
6. MLP 가중치를 갱신하기 위해 역전파.

**왜 이것이 3D 기하를 학습하는가?** MLP가 **모든 뷰를 동시에 설명**해야 하기 때문. 단일 뷰는 많은 다른 밀도 분포로 설명될 수 있음 — 하지만 모든 뷰와 일관된 것은 올바른 3D 기하뿐. 다중 뷰 일관성이 밀도 필드가 실제 장면 기하와 일치하도록 강제하는 암묵적 감독.

훈련은 알려진 **카메라 포즈**가 필요. 일반적으로 NeRF 훈련 전에 COLMAP(SFM, §22.F)으로 계산.

### F. Instant-NGP: 해시 인코딩된 특징

원본 NeRF는 느림: 단일 장면에 GPU에서 ~1일, MLP가 volume rendering을 위해 픽셀당 수백 번 질의되어야 하기 때문. Instant-NGP(Müller 등, 2022)가 한 큰 아이디어로 **100× 가속** 달성:

무거운 positional encoding + 깊은 MLP를 다음으로 대체:

- **다중 해상도 hash 격자**: 서로 다른 해상도의 3D 격자 집합. 각 격자 셀이 작은 특징 벡터(예: 2 차원) 저장. 격자가 공간 해싱으로 인덱싱되어, 격자 해상도와 무관하게 메모리가 유계 유지.
- **작은 MLP**: 모든 격자 레벨의 연결된 특징을 받아 색상 + 밀도를 생성하는 2-3 layer MLP.

작동 이유: hash 격자가 대부분의 장면 정보를 명시적으로 특징으로 저장, MLP의 작업이 사소해짐. 훈련이 시간 대신 몇 분으로 단축. 렌더링도 많은 경우 실시간으로 충분히 빨라짐.

Instant-NGP가 NeRF를 실용적으로 만듦. 후속(Nerfacto, ZipNeRF, Mip-NeRF 360) 모두 hash encoding 아이디어 위에 구축.

### 이론에서 아래 함수들로

이전 레슨과 달리, NeRF는 OpenCV에서 구현되지 않음. 표준 도구:

- **NerfStudio**: Nerfacto, Instant-NGP, 많은 변형의 Python 프레임워크. 일반 사용에 최선.
- **Nvidia Instant-NGP**: 원본 빠른 C++/CUDA 구현.
- **3D Gaussian Splatting**: (29 레슨) — 2023년 NeRF를 최고 수준으로 대체한 완전히 다른 접근.

NeRF 훈련 파이프라인:
1. 많은 시점에서 사진 캡처.
2. 카메라 포즈 추정을 위해 COLMAP 실행.
3. 포즈 주석된 이미지로 NeRF 훈련.
4. 훈련된 MLP를 새 카메라 광선에서 평가해 새 뷰 렌더링.

---

## 1. 신경 장면 표현

### 1.1 명시적 vs 암시적 표현

```
명시적 표현:
  포인트 클라우드: 3D 점들의 집합
  메시: 꼭짓점 + 면
  복셀: 값의 3D 격자
  + 빠른 렌더링 (래스터라이제이션)
  - 고정 해상도, 큰 메모리

암시적 표현 (NeRF):
  신경망: f(x, y, z, θ, φ) → (r, g, b, σ)
  입력: 3D 위치 + 시선 방향
  출력: 해당 지점의 색상 + 밀도
  + 연속적 (무한 해상도)
  + 컴팩트 (네트워크 가중치만 필요)
  - 느린 렌더링 (네트워크를 많이 쿼리해야 함)
```

---

## 2. NeRF 기초

### 2.1 NeRF 방정식

```
NeRF는 장면을 연속적인 볼류메트릭 함수로 표현:

  F: (x, y, z, θ, φ) → (r, g, b, σ)

  (x, y, z): 공간의 3D 위치
  (θ, φ):    시선 방향 (정반사와 같은 시점 의존적 효과용)
  (r, g, b): 이 지점의 색상
  σ:         밀도 (이 지점이 얼마나 불투명한가?)

  높은 σ → 단단한 표면
  낮은 σ → 빈 공간 / 투명

픽셀을 렌더링하려면:
  1. 카메라에서 픽셀을 통과하는 레이 투사
  2. 레이를 따라 N개의 점 샘플링
  3. 각 점에서 NeRF 쿼리 → (색상, 밀도) 획득
  4. 볼류메트릭 렌더링 방정식을 사용하여 색상 누적
```

### 2.2 위치 인코딩

```python
import torch
import torch.nn as nn
import numpy as np


def positional_encoding(x, L=10):
    """
    정현파 함수를 사용하여 좌표를 고차원 공간으로 매핑.

    γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ...,
            sin(2^(L-1)πp), cos(2^(L-1)πp)]

    네트워크가 고주파 세부 사항을 학습하는 데 도움.
    PE 없이는 네트워크가 부드럽고/흐릿한 표현을 학습하는 경향.
    """
    encodings = [x]
    for i in range(L):
        freq = 2.0 ** i * np.pi
        encodings.append(torch.sin(freq * x))
        encodings.append(torch.cos(freq * x))
    return torch.cat(encodings, dim=-1)
    # 입력 차원 d → 출력 차원 d(1 + 2L)
```

---

## 3. 볼류메트릭 렌더링

### 3.1 렌더링 방정식

```
레이 r(t) = o + td를 따른 볼륨 렌더링:

  C(r) = ∫[t_near to t_far] T(t) · σ(r(t)) · c(r(t), d) dt

  여기서:
  T(t) = exp(-∫[t_near to t] σ(r(s)) ds)  (투과율)
  σ = 밀도, c = 색상, d = 방향

  T(t) = 레이가 t_near에서 t까지 아무것도 부딪히지 않고 이동할 확률

이산 근사 (구적법):
  C(r) ≈ Σᵢ Tᵢ · αᵢ · cᵢ

  여기서:
  αᵢ = 1 - exp(-σᵢ · δᵢ)           (구간 i의 불투명도)
  Tᵢ = Π_{j<i} (1 - αⱼ)            (누적 투과율)
  δᵢ = tᵢ₊₁ - tᵢ                   (샘플 간 거리)
```

### 3.2 볼륨 렌더링 구현

```python
def render_rays(network, rays_o, rays_d, near=2.0, far=6.0,
                n_samples=64, n_importance=64):
    """
    레이 배치에 대한 색상 렌더링.

    Args:
        network: NeRF 모델
        rays_o: (N, 3) 레이 원점
        rays_d: (N, 3) 레이 방향
        near/far: 샘플링 범위
        n_samples: 조밀 샘플 수
        n_importance: 세밀 (중요도) 샘플 수
    Returns:
        rgb: (N, 3) 렌더링된 색상
        depth: (N,) 추정 깊이
    """
    N = rays_o.shape[0]
    device = rays_o.device

    # 1. 레이를 따라 점 샘플링 (층화 샘플링)
    t_vals = torch.linspace(near, far, n_samples, device=device)
    # 학습 중 정규화를 위한 노이즈 추가
    noise = torch.rand(N, n_samples, device=device) * (far - near) / n_samples
    t_vals = t_vals.unsqueeze(0) + noise

    # 레이를 따른 3D 위치
    points = rays_o.unsqueeze(1) + t_vals.unsqueeze(2) * rays_d.unsqueeze(1)
    # points 형상: (N, n_samples, 3)

    # 2. 네트워크에 색상과 밀도 쿼리
    dirs = rays_d.unsqueeze(1).expand_as(points)

    # 위치 인코딩
    encoded_points = positional_encoding(points.reshape(-1, 3), L=10)
    encoded_dirs = positional_encoding(dirs.reshape(-1, 3), L=4)

    raw = network(encoded_points, encoded_dirs)
    raw = raw.reshape(N, n_samples, 4)  # (rgb=3, sigma=1)

    rgb_raw = torch.sigmoid(raw[..., :3])   # [0, 1] 범위의 색상
    sigma = torch.relu(raw[..., 3])         # 밀도 >= 0

    # 3. 볼륨 렌더링
    deltas = t_vals[:, 1:] - t_vals[:, :-1]
    deltas = torch.cat([deltas, torch.full((N, 1), 1e10, device=device)], dim=1)

    alpha = 1 - torch.exp(-sigma * deltas)  # 불투명도

    # 투과율: (1 - alpha)의 누적 곱
    transmittance = torch.cumprod(
        torch.cat([torch.ones(N, 1, device=device), 1 - alpha + 1e-10], dim=1),
        dim=1
    )[:, :-1]

    weights = transmittance * alpha  # (N, n_samples)

    # 색상의 가중 합
    rgb = (weights.unsqueeze(-1) * rgb_raw).sum(dim=1)  # (N, 3)

    # 깊이 추정
    depth = (weights * t_vals).sum(dim=1)  # (N,)

    return rgb, depth, weights
```

---

## 4. NeRF 구현

### 4.1 NeRF 네트워크 아키텍처

```python
class NeRF(nn.Module):
    """원본 NeRF 아키텍처."""

    def __init__(self, pos_dim=63, dir_dim=27, hidden_dim=256, n_layers=8):
        super().__init__()
        # pos_dim = 3 + 3*2*10 = 63 (위치 + L=10의 PE)
        # dir_dim = 3 + 3*2*4 = 27 (방향 + L=4의 PE)

        # 위치 인코딩 레이어
        layers = [nn.Linear(pos_dim, hidden_dim), nn.ReLU()]
        for i in range(1, n_layers):
            if i == 4:
                # 레이어 4에서 스킵 연결
                layers.append(nn.Linear(hidden_dim + pos_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.pos_layers = nn.ModuleList([l for l in layers if isinstance(l, nn.Linear)])

        # 밀도 출력 (시점 독립적)
        self.sigma_layer = nn.Linear(hidden_dim, 1)

        # 색상 출력 (시점 의존적)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dir_layer = nn.Linear(hidden_dim + dir_dim, hidden_dim // 2)
        self.rgb_layer = nn.Linear(hidden_dim // 2, 3)

    def forward(self, pos_encoded, dir_encoded):
        """
        Args:
            pos_encoded: (N, pos_dim) 위치 인코딩된 3D 위치
            dir_encoded: (N, dir_dim) 위치 인코딩된 시선 방향
        Returns:
            output: (N, 4) [r, g, b, sigma]
        """
        x = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            if i == 4:
                x = torch.cat([x, pos_encoded], dim=-1)
            x = torch.relu(layer(x))

        # 밀도 (방향 의존성 없음)
        sigma = self.sigma_layer(x)

        # 색상 (시선 방향에 의존)
        features = self.feature_layer(x)
        x = torch.cat([features, dir_encoded], dim=-1)
        x = torch.relu(self.dir_layer(x))
        rgb = self.rgb_layer(x)

        return torch.cat([rgb, sigma], dim=-1)
```

---

## 5. Instant-NGP와 속도 개선

### 5.1 해시 인코딩

```
Instant-NGP (Mueller et al., 2022):
  정현파 위치 인코딩을 다중 해상도 해시 테이블로 대체.

  학습 시간: 시간 단위 → 분 단위 (100배 속도 향상!)

  다중 해상도 해시 격자:
  ┌─────────┐ ┌───────────┐ ┌─────────────────┐
  │ 조밀     │ │ 중간       │ │ 세밀             │
  │ 16×16×16 │ │ 32×32×32  │ │ 512×512×512      │
  │ 격자     │ │ 격자       │ │ 격자 (해시됨!)    │
  └─────────┘ └───────────┘ └─────────────────┘
       │            │              │
       └────────────┼──────────────┘
                    ▼
              연결 → 작은 MLP → (rgb, sigma)

  핵심 통찰: 해시 충돌은 네트워크가 처리!
  같은 해시 항목을 공유하는 서로 다른 위치의 그래디언트가
  일반적인 장면 기하에 대해 올바르게 평균화됨.
```

---

## 6. 학습 파이프라인

### 6.1 NeRF 학습

```python
def train_nerf(model, images, poses, intrinsics, n_iterations=200000,
               lr=5e-4, batch_size=4096):
    """포즈가 있는 이미지 세트로 NeRF 학습."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    H, W = images.shape[1:3]
    n_images = len(images)

    for iteration in range(n_iterations):
        # 랜덤 이미지와 랜덤 픽셀 샘플링
        img_idx = np.random.randint(n_images)
        target_img = images[img_idx]
        pose = poses[img_idx]

        # 랜덤 픽셀 좌표 샘플링
        pixel_indices = np.random.choice(H * W, batch_size, replace=False)
        pixel_y = pixel_indices // W
        pixel_x = pixel_indices % W

        # 선택된 픽셀에 대한 레이 생성
        rays_o, rays_d = get_rays(H, W, intrinsics, pose, pixel_y, pixel_x)

        # 목표 색상
        target_rgb = target_img[pixel_y, pixel_x]

        # 렌더링
        pred_rgb, depth, weights = render_rays(model, rays_o, rays_d)

        # MSE 손실
        loss = ((pred_rgb - target_rgb) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (iteration + 1) % 10000 == 0:
            psnr = -10 * torch.log10(loss)
            print(f"Iter {iteration+1}: Loss={loss.item():.6f}, PSNR={psnr.item():.2f}")

    return model
```

---

## 7. 응용 및 확장

### 7.1 NeRF 응용 분야

```
Neural Radiance Fields의 응용:

1. 새로운 시점 합성:
   학습 데이터에 없는 임의의 시점에서 장면 렌더링.

2. 3D 재구성:
   밀도 필드에 대한 Marching Cubes를 통해 메시 추출.

3. 가상 현실:
   사진에서 몰입감 있는 3D 환경 생성.

4. 디지털 트윈:
   시뮬레이션을 위한 실제 위치 재구성.

5. 전자 상거래:
   모든 각도에서의 제품 시각화.

6. 문화유산 보존:
   문화 유물과 건축물의 디지털화.

확장:
  - Mip-NeRF: 콘 트레이싱을 통한 안티앨리어싱 렌더링
  - NeRF-W: 다양한 조명과 일시적 객체 처리
  - Dynamic NeRF: 시간에 따라 변하는 장면 모델링
  - NeRF in the Wild: 광도 변화에 강건
```

---

## 8. 연습문제

### 연습문제 1: 최소 NeRF

최소 NeRF를 처음부터 구축:
1. 위치 인코딩 구현 (위치에 L=10, 방향에 L=4)
2. 레이어 4에서 스킵 연결이 있는 8층 MLP 구축
3. 층화 레이 샘플링과 볼륨 렌더링 구현
4. 간단한 합성 장면에서 학습 (예: Blender Lego)
5. 새로운 시점 렌더링 및 PSNR 측정

### 연습문제 2: 레이 마칭 시각화

레이 마칭 과정 시각화:
1. 장면 위에 레이를 따라 샘플링된 점 표시
2. 여러 레이를 따른 밀도 및 색상 값 플롯
3. 투과율과 가중치 함수 시각화
4. 레이를 따라 불투명도가 누적되는 과정 표시
5. 조밀 vs 세밀 샘플링 비교 (계층적 샘플링)

### 연습문제 3: 계층적 샘플링

NeRF의 2단계 샘플링 구현:
1. 64개의 균일 샘플을 가진 조밀 네트워크
2. 64개의 중요도 샘플을 가진 세밀 네트워크 (조밀 가중치 기반)
3. 품질 비교: 조밀만 vs 조밀+세밀
4. 중요도 샘플이 집중되는 위치 시각화
5. 측정: 계층적 샘플링으로 인한 PSNR 개선

### 연습문제 4: 깊이와 법선 추출

학습된 NeRF에서 기하 정보 추출:
1. 여러 시점에서 깊이 맵 렌더링
2. 깊이 기울기로부터 표면 법선 계산
3. 밀도 필드에 대한 Marching Cubes를 사용하여 메시 추출
4. 추출된 메시와 정답 비교 (가능한 경우)
5. 식별: NeRF가 기하학적으로 어려움을 겪는 곳은?

### 연습문제 5: Instant-NGP 스타일 인코딩

해시 인코딩 가속 구현:
1. 다중 해상도 해시 격자 구축 (2-3 레벨)
2. 격자 조회를 위한 삼선형 보간 구현
3. 정현파 PE를 해시 인코딩으로 대체
4. 학습 속도 비교: 정현파 PE vs 해시 인코딩
5. 품질 측정: 동일 학습 시간에서의 PSNR 비교

---

*28강 끝*
