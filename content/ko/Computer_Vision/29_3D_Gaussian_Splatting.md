[이전: Neural Radiance Fields](./28_Neural_Radiance_Fields.md)

---

# 29. 3D Gaussian Splatting

## 학습 목표

이 수업을 완료하면 다음을 수행할 수 있습니다:

1. NeRF의 명시적 대안으로서의 3D Gaussian Splatting 설명
2. 가우시안 프리미티브 표현 (위치, 공분산, 색상, 불투명도) 서술
3. 가우시안 프리미티브를 위한 미분 가능한 래스터라이제이션 구현
4. 품질, 속도, 메모리 측면에서 Gaussian Splatting과 NeRF 비교
5. 실시간 새로운 시점 합성에 3D Gaussian Splatting 적용

---

## 목차

1. [왜 Gaussian Splatting인가?](#1-왜-gaussian-splatting인가)
2. [3D 가우시안 표현](#2-3d-가우시안-표현)
3. [미분 가능한 래스터라이제이션](#3-미분-가능한-래스터라이제이션)
4. [적응적 밀도 제어](#4-적응적-밀도-제어)
5. [학습 파이프라인](#5-학습-파이프라인)
6. [실시간 렌더링](#6-실시간-렌더링)
7. [NeRF와의 비교](#7-nerf와의-비교)
8. [연습문제](#8-연습문제)

---

## 1. 왜 Gaussian Splatting인가?

### 이론: 명시적 원시가 렌더링에서 이김

NeRF의 병목은 광선당 MLP를 수백 번 질의하는 것. 작은 MLP의 Instant-NGP조차 모든 픽셀의 광선을 따라 모든 샘플 점에 대해 평가해야 함.

Gaussian Splatting은 추론에서 네트워크를 완전히 제거. 각 장면 요소가 파라미터를 가진 3D 가우시안으로 **명시적으로 저장**. 렌더링은 단지 정렬된 알파 블렌드 — 게임 엔진이 수십 년 동안 해온 같은 연산, 단지 삼각형 대신 가우시안.

### 1.1 NeRF vs Gaussian Splatting

```
NeRF (암시적):
  장면 = 신경망 f(x,y,z,θ,φ) → (rgb, σ)
  렌더링: 레이 마칭 (레이당 네트워크 ~100회 쿼리)
  속도: 1080p에서 ~1 FPS
  학습: 수시간

3D Gaussian Splatting (명시적):
  장면 = 3D 가우시안의 집합 {μᵢ, Σᵢ, cᵢ, αᵢ}
  렌더링: 래스터라이제이션 (투영 + 정렬 + 알파 블렌딩)
  속도: 1080p에서 ~100+ FPS (실시간!)
  학습: ~10-30분

핵심 통찰: 미분 가능한 래스터라이제이션을 사용한 점 기반 렌더링.
각 "스플랫"은 투영되고 블렌딩되는 3D 가우시안.
```

---

## 2. 3D 가우시안 표현

### 이론: 3D 가우시안 원시

각 가우시안이 가지는 파라미터:

- **위치** `μ ∈ ℝ³`: 공간 내 위치.
- **공분산** `Σ ∈ ℝ³×³`: 모양 — 방향과 축별 스케일(비등방 blob).
- **색상**(또는 시점 의존 색상을 위한 spherical harmonics, §E).
- **불투명도** `α ∈ [0, 1]`.

`Σ`를 학습 가능하게 유지하면서 유효(positive semi-definite)하도록, `Σ = R · S · Sᵀ · Rᵀ`로 매개변수화 — `R`은 회전(쿼터니언에서), `S = diag(s_x, s_y, s_z)`는 축별 스케일. 쿼터니언 + 3 스케일 = 모양에 가우시안당 7 숫자.

장면은 일반적으로 **수백만** 개 가우시안 포함.

### 이론: 시점 의존 색상을 위한 Spherical Harmonics

가우시안당 단순 RGB 색상은 시점에 따라 변하는 specular 반사를 포착할 수 없음. NeRF는 시점 방향을 받는 MLP 사용. Gaussian Splatting은 대신 **spherical harmonics**(SH) — 구면에서의 Fourier series 유사물 — 사용:

```
color(direction) = Σ_l  Σ_m  c_{lm} · Y_l^m(direction)
```

각 가우시안이 어떤 차수까지 SH 계수 저장(보통 차수 3, 즉 채널당 16 계수 × 3 채널 = 가우시안당 48 숫자). 렌더 시점에 시점 방향에서 SH 기저 평가하고 저장된 계수와 결합.

SH는 명시적, 고정 계수로 **연속 방향 의존 색상** 제공 — 렌더 시점에 신경망 질의 없음.

### 2.1 가우시안 파라미터

```python
import torch
import torch.nn as nn
import numpy as np


class GaussianModel(nn.Module):
    """장면을 나타내는 3D 가우시안의 집합."""

    def __init__(self, n_points=100000):
        super().__init__()
        # 각 가우시안은 다음을 가짐:
        self.positions = nn.Parameter(torch.randn(n_points, 3) * 0.5)
        self.scales = nn.Parameter(torch.ones(n_points, 3) * -3.0)  # log 스케일
        self.rotations = nn.Parameter(torch.zeros(n_points, 4))  # 쿼터니언
        self.rotations.data[:, 0] = 1.0  # 항등 회전
        self.opacities = nn.Parameter(torch.zeros(n_points, 1))  # 로짓
        self.sh_coeffs = nn.Parameter(torch.zeros(n_points, 48))  # 구면 조화 함수

    @property
    def get_scales(self):
        return torch.exp(self.scales)

    @property
    def get_opacities(self):
        return torch.sigmoid(self.opacities)

    def get_covariance(self):
        """스케일과 회전으로부터 3D 공분산 행렬 계산."""
        S = torch.diag_embed(self.get_scales)  # (N, 3, 3)
        R = self._quaternion_to_matrix(self.rotations)  # (N, 3, 3)
        # Σ = R · S · Sᵀ · Rᵀ
        L = R @ S
        return L @ L.transpose(-1, -2)

    def _quaternion_to_matrix(self, q):
        """쿼터니언을 3x3 회전 행렬로 변환."""
        q = torch.nn.functional.normalize(q, dim=-1)
        w, x, y, z = q.unbind(-1)

        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
            2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
            2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
        ], dim=-1).reshape(-1, 3, 3)

        return R

    def get_colors(self, viewdir=None):
        """구면 조화 함수에서 색상 가져오기."""
        # 0차 (상수): 기본 색상만
        base_color = self.sh_coeffs[:, :3]  # RGB
        return torch.sigmoid(base_color)
```

---

## 3. 미분 가능한 래스터라이제이션

### 이론: Splatting: 미분 가능한 래스터라이제이션

가우시안을 2D 이미지로 렌더링:

1. 카메라를 통해 **가우시안 투영**: 원근 투영 아래 3D 가우시안이 이미지 평면의 2D 가우시안이 됨(국소 선형 근사를 통해 근사적으로).
2. **영향 받는 픽셀 계산**: 투영된 평균 주변의 픽셀 타일.
3. 각 픽셀에서 **가우시안 밀도 평가**: 픽셀별 가중치 `w_i = exp(-0.5 · (p - μ_2D)ᵀ · Σ_2D⁻¹ · (p - μ_2D))`.
4. **불투명도와 색상 곱하기**: 픽셀별 기여 = `α · color · w_i`.

여러 겹친 가우시안의 경우, 깊이로 정렬하고 **알파 블렌드**(앞에서 뒤로):

```
final_color(pixel) = Σ_i  T_i · α_i · w_i · c_i
T_i = Π_{j < i}  (1 - α_j · w_j)        이전 가우시안을 통한 투과율
```

이것이 **NeRF와 같은 volume rendering 적분**, 단지 점 샘플 대신 가우시안에 걸쳐 이산화. 결정적으로, 미분 가능 — 기울기가 알파 블렌딩을 통해 흐르며 훈련 중 가우시안 파라미터를 갱신.

구현은 GPU의 **타일 기반 래스터라이제이션** 사용: 이미지를 16×16 타일로 분할, 타일당 가우시안 정렬, 각 타일을 병렬로 렌더링. 총 처리량: 현대 GPU에서 초당 수천만 가우시안.

### 3.1 투영 및 스플래팅

```
렌더링 파이프라인:

1. 3D 가우시안을 2D로 투영:
   μ_2D = K · [R|t] · μ_3D        (카메라 투영)
   Σ_2D = J · W · Σ_3D · Wᵀ · Jᵀ  (공분산 투영)

   여기서 J = 투영의 야코비안, W = 월드-카메라 변환

2. 깊이 기준으로 가우시안 정렬 (앞에서 뒤로)

3. 각 픽셀에 대해 겹치는 가우시안을 알파 합성:
   C(pixel) = Σᵢ cᵢ · αᵢ · Tᵢ

   αᵢ = 불투명도 × 픽셀에서의 가우시안 값
   Tᵢ = Π_{j<i} (1 - αⱼ)  (투과율)

   NeRF 볼류메트릭 렌더링과 동일하지만, 정렬된 프리미티브 사용!
```

### 3.2 간소화된 래스터라이저

```python
def render_gaussians(gaussians, camera, H, W):
    """
    간소화된 Gaussian Splatting 렌더러.
    실제 구현은 타일 기반 래스터라이제이션에 CUDA 사용.
    """
    # 1. 2D로 투영
    positions_3d = gaussians.positions
    means_2d, depths = project_points(positions_3d, camera)

    # 2. 2D 공분산 계산
    cov_3d = gaussians.get_covariance()
    cov_2d = project_covariance(cov_3d, camera, positions_3d)

    # 3. 깊이 기준 정렬
    sorted_indices = depths.argsort()

    # 4. 래스터라이즈 (알파 합성)
    image = torch.zeros(H, W, 3, device=positions_3d.device)
    accumulated_alpha = torch.zeros(H, W, 1, device=positions_3d.device)

    colors = gaussians.get_colors()
    opacities = gaussians.get_opacities

    for idx in sorted_indices:
        if accumulated_alpha.max() > 0.99:
            break  # 조기 종료

        mu = means_2d[idx]  # 2D 중심
        cov = cov_2d[idx]   # 2x2 공분산
        color = colors[idx]  # RGB
        opacity = opacities[idx]

        # 각 픽셀에서 가우시안 평가
        alpha = evaluate_2d_gaussian(mu, cov, opacity, H, W)

        # 알파 합성
        weight = alpha * (1 - accumulated_alpha)
        image += weight * color.unsqueeze(0).unsqueeze(0)
        accumulated_alpha += weight

    return image


def evaluate_2d_gaussian(mean, cov, opacity, H, W):
    """각 픽셀 위치에서 2D 가우시안 평가."""
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack([x.float(), y.float()], dim=-1)  # (H, W, 2)

    diff = coords - mean  # (H, W, 2)
    cov_inv = torch.inverse(cov)  # (2, 2)

    # 마할라노비스 거리
    exponent = -0.5 * torch.sum(diff @ cov_inv * diff, dim=-1)

    # 가우시안 값 * 불투명도
    alpha = opacity * torch.exp(exponent)
    return alpha.unsqueeze(-1)  # (H, W, 1)
```

---

## 4. 적응적 밀도 제어

### 이론: 적응적 밀도 제어

희소 SfM 포인트 클라우드에서 시작, 옵티마이저는 장면에 디테일이 있는 곳에 가우시안을 추가하고 도움이 되지 않는 곳에서 제거해야 함. 이것이 **적응적 밀도 제어**:

- **밀집화**: 가우시안이 큰 위치 기울기를 가지면(이미지 손실이 강하게 이동을 원함), **복제 또는 분할**. 복제는 중복; 분할은 큰 가우시안을 두 작은 것으로 대체.
- **가지치기**: 훈련 중 가우시안의 불투명도가 임계값 아래로 떨어지면 제거.

이 동적 관리가 가우시안 수를 장면 복잡도에 적응시킴. 매끄러운 배경은 적은 큰 가우시안 얻음; 세밀한 디테일(잎사귀, 머리카락)은 많은 작은 것 얻음. 최종 수는 보통 장면에 따라 1-5백만 가우시안.

### 4.1 가우시안 증식 및 가지치기

```
학습 중 가우시안 수를 적응적으로 조정:

밀집화 (가우시안 추가):
  - 복제: 그래디언트가 크고 가우시안이 작으면 → 근처에 복제
  - 분할: 그래디언트가 크고 가우시안이 크면 → 둘로 분할

가지치기 (가우시안 제거):
  - 불투명도 < 임계값인 가우시안 제거 (거의 투명)
  - 너무 큰 가우시안 제거 (너무 넓은 영역 차지)
  - 주기적으로 불투명도를 리셋하여 가지치기 촉진

이를 통해 필요한 곳 (복잡한 영역)에
적응적으로 디테일을 할당할 수 있음.

일반적인 진행:
  초기: 100K 포인트 (SfM에서)
  밀집화 후: 2-5M 가우시안
  가지치기 후: 1-3M 가우시안
```

---

## 5. 학습 파이프라인

### 5.1 완전한 학습 루프

```python
def train_gaussian_splatting(model, images, cameras, n_iterations=30000,
                              lr_position=0.00016, lr_other=0.0025):
    """3D Gaussian Splatting 모델 학습."""
    optimizer = torch.optim.Adam([
        {'params': [model.positions], 'lr': lr_position},
        {'params': [model.scales], 'lr': lr_other},
        {'params': [model.rotations], 'lr': lr_other * 0.1},
        {'params': [model.opacities], 'lr': 0.05},
        {'params': [model.sh_coeffs], 'lr': lr_other * 0.5},
    ])

    n_images = len(images)

    for iteration in range(n_iterations):
        # 랜덤 시점
        idx = np.random.randint(n_images)
        gt_image = images[idx]
        camera = cameras[idx]
        H, W = gt_image.shape[:2]

        # 렌더링
        rendered = render_gaussians(model, camera, H, W)

        # L1 + SSIM 손실
        l1_loss = torch.abs(rendered - gt_image).mean()
        ssim_loss = 1 - compute_ssim(rendered, gt_image)
        loss = 0.8 * l1_loss + 0.2 * ssim_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 적응적 밀도 제어
        if iteration > 500 and iteration % 100 == 0:
            densify_and_prune(model, iteration)

        if (iteration + 1) % 1000 == 0:
            psnr = -10 * torch.log10(((rendered - gt_image) ** 2).mean())
            n_gaussians = model.positions.shape[0]
            print(f"Iter {iteration+1}: PSNR={psnr:.2f}, "
                  f"Gaussians={n_gaussians:,}")
```

---

## 6. 실시간 렌더링

### 6.1 타일 기반 래스터라이제이션

```
실시간 렌더링은 타일 기반 CUDA 래스터라이제이션 사용:

1. 화면을 16×16 픽셀 타일로 분할
2. 각 가우시안에 대해 겹치는 타일 결정
3. 각 타일에서 겹치는 가우시안을 깊이 기준으로 정렬
4. GPU에서 타일별 병렬 알파 합성

달성 가능:
  - 1080p에서 100+ FPS
  - 4K에서 30+ FPS
  - NeRF보다 수 자릿수 더 빠름
```

---

## 7. NeRF와의 비교

### 이론: NeRF와의 비교

| 속성 | NeRF (Instant-NGP) | 3D Gaussian Splatting |
|------|---------------------|----------------------|
| 품질 (PSNR) | ~32-34 dB | ~32-34 dB |
| 훈련 시간 | ~분 | ~분 |
| 렌더링 속도 (1080p) | ~10 FPS | **~100+ FPS** |
| 메모리 (저장) | ~50 MB | ~500 MB - 2 GB |
| 편집성 | 어려움 | 더 쉬움(원시 이동/재색상) |
| Specularity | 예 (MLP) | 예 (SH) |
| 기하 품질 | 좋음 | 좋지만 blob 같음 |
| 앤티앨리어싱 | 내장 | 주의 필요 |

큰 트레이드오프: 3DGS가 명시적 원시 수백만을 저장하므로 더 많은 **메모리** 사용, 하지만 추론 루프에 신경망이 없으므로 훨씬 더 빠르게 **렌더링**. 같은 장면을 여러 번 렌더링하는 응용(VR, 게이밍, 텔레프레전스)에는 3DGS가 지배적. 컴팩트 표현이 필요한 응용(NeRF mobile, 자산 배포)에는 NeRF가 여전히 이점.

3DGS가 빠르게 새로운 시점 합성의 선호 접근이 됨. 변형(비디오용 4D-GS, 애니메이션 장면용 deformable-GS, 야외용 large-scale GS)이 더 확장.

### 7.1 특징 비교

```
특징                | NeRF           | 3DGS
--------------------|----------------|------------------
표현                | 암시적 (MLP)   | 명시적 (포인트)
학습 시간           | 수시간          | 10-30분
렌더링 속도         | ~1 FPS         | 100+ FPS
품질 (PSNR)         | ~33 dB         | ~33 dB (비슷)
메모리 (모델)       | ~5 MB          | ~50-500 MB
편집 가능성         | 어려움          | 쉬움 (포인트 이동/삭제)
시점 외삽           | 부족            | 부족
동적 장면           | 확장 필요       | 확장 필요
```

---

## 8. 연습문제

### 연습문제 1: 2D Gaussian Splatting

3D로 이동하기 전에 2D 가우시안으로 시작:
1. 1000개의 색상 가우시안을 사용하여 2D 이미지 표현
2. 대상 이미지에 맞게 위치, 스케일, 회전, 색상 최적화
3. 미분 가능한 알파 합성 구현
4. 시각화: 가우시안이 이미지를 표현하기 위해 어떻게 분포되는지
5. 애니메이션: 무작위에서 수렴까지의 최적화 과정 표시

### 연습문제 2: 3D 가우시안 초기화

포인트 클라우드에서 가우시안 초기화:
1. COLMAP 또는 SfM을 사용하여 이미지에서 희소 3D 포인트 획득
2. 포인트 클라우드에서 가우시안 파라미터 초기화
3. 초기 포인트 클라우드 vs 최적화된 가우시안 시각화
4. 비교: 무작위 초기화 vs SfM 초기화
5. 측정: 초기화가 최종 품질에 미치는 영향

### 연습문제 3: 밀도 제어 분석

적응적 밀도 제어 연구:
1. 밀집화/가지치기 있이와 없이 학습
2. 학습 반복 횟수에 따른 가우시안 수 기록
3. 시각화: 가우시안이 추가되는 곳 (복잡한 영역)?
4. 시각화: 가우시안이 제거되는 곳 (빈 공간)?
5. 품질 vs 가우시안 수 트레이드오프 측정

### 연습문제 4: 품질 비교

3DGS와 NeRF 비교:
1. 동일 장면에서 두 방법 학습 (Mip-NeRF 360 데이터셋)
2. 측정: 두 방법의 PSNR, SSIM, LPIPS
3. 학습 시간과 렌더링 속도 비교
4. 세밀한 디테일 확대: 각 방법이 뛰어난 곳은?
5. 어려운 장면 테스트: 반사, 얇은 구조물

### 연습문제 5: 장면 편집

3DGS의 편집 가능성 시연:
1. 장면에서 3DGS 학습
2. 구현: 영역 내 가우시안 선택 및 삭제
3. 구현: 가우시안 그룹 이동 (객체 이동)
4. 구현: 선택된 가우시안의 색상 변경
5. 새로운 시점에서 편집된 장면 렌더링

---

*29강 끝*
