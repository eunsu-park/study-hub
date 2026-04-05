# 05. 셰이딩 모델

[← 이전: 04. 래스터화](04_Rasterization.md) | [다음: 06. 텍스처 매핑 →](06_Texture_Mapping.md)

---

## 학습 목표

1. 경험적 셰이딩의 구성 요소인 주변광(Ambient), 확산(Diffuse, Lambert), 반사광(Specular, Phong)을 이해한다
2. 반사 하이라이트를 위한 블린-퐁(Blinn-Phong) 수정 모델을 유도하고 구현한다
3. 플랫(Flat), 고로(Gouraud), 퐁(Phong) 셰이딩 보간 방식의 차이를 구분한다
4. 물리 기반 렌더링(PBR, Physically Based Rendering)의 물리적 근거를 이해한다
5. GGX 법선 분포 함수, 슐릭(Schlick) 프레넬, 스미스(Smith) 기하 함수를 사용하는 쿡-토런스(Cook-Torrance) BRDF를 구현한다
6. 현대 게임 엔진에서 사용되는 메탈릭-러프니스(Metallic-Roughness) 워크플로를 설명한다
7. 구(Sphere)에서 퐁 셰이딩과 PBR 셰이딩의 결과를 비교한다
8. 표면 특성(거칠기, 금속성)이 외관에 미치는 영향에 대한 직관을 기른다

---

## 왜 중요한가

셰이딩(Shading)은 3D 물체를 *실제처럼* 보이게 만드는 요소다. 셰이딩이 없으면 구(Sphere)는 그냥 납작한 원에 불과하다. 플라스틱 공, 크롬 구, 거친 돌의 차이는 전적으로 표면이 빛과 어떻게 상호작용하는지에 의해 결정된다. 1970년대부터 가르쳐 온 고전적인 퐁(Phong) 모델부터 현대 게임 엔진에서 사용되는 물리 기반 렌더링(PBR)까지, 셰이딩 모델을 이해하는 것은 설득력 있는 디지털 이미지를 만드는 데 필수적이다. PBR은 실제 물리학에 근거하여 어떤 조명 조건에서도 일관되고 예측 가능한 결과를 생성하기 때문에 업계 표준이 되었다.

---

## 1. 빛과 표면의 상호작용

빛이 표면에 닿으면 세 가지 현상이 일어날 수 있다:
- **흡수(Absorption)**: 빛 에너지가 열로 변환된다 (표면이 더 어둡게 보임)
- **반사(Reflection)**: 빛이 표면에서 튕겨 나간다
- **투과(Transmission)**: 빛이 통과한다 (투명도, 굴절)

불투명한 표면에서는 반사에 집중하며, 반사는 두 가지 요소로 구성된다:

- **확산 반사(Diffuse Reflection)**: 빛이 모든 방향으로 동일하게 산란된다 (분필, 고무 같은 무광 표면)
- **정반사(Specular Reflection)**: 빛이 특정 방향으로 반사된다 (금속, 유리 같은 광택 표면)

```
         Incoming      Specular        Incoming      Diffuse
          light        reflection       light       reflection
            \         /                  \         / | \
             \       /                    \       /  |  \
              \     /                      \     /   |   \
    ───────────●───────────     ───────────●───────────
              surface                     surface
```

---

## 2. 퐁 반사 모델

**퐁 모델**(Bui Tuong Phong, 1975)은 세 가지 덧셈 성분을 사용하여 표면 외관을 근사하는 경험적 모델이다.

### 2.1 주변광 성분(Ambient Component)

주변광은 환경에서 오는 간접 조명을 나타낸다 — 이 지점에 도달하기 전에 여러 표면에서 반사된 빛이다. 이는 상수로 근사된다:

$$I_{\text{ambient}} = k_a \cdot I_a$$

여기서:
- $k_a$: 주변광 반사율 계수 (재질 속성, $[0, 1]$)
- $I_a$: 주변광 강도

> **한계**: 실제 간접 조명은 장면 전체에서 다양하게 변한다. 전역 조명(Global Illumination) 기법(라디오시티, 경로 추적)이 이를 정확히 모델링하지만, 주변광은 값싼 근사치다.

### 2.2 확산 성분(Diffuse Component) — 람베르트 법칙

완전히 확산되는(람베르트, Lambertian) 표면은 입사광을 모든 방향으로 동일하게 산란시킨다. 인지되는 밝기는 표면 법선(Normal) $\mathbf{N}$과 빛 방향 $\mathbf{L}$ 사이의 각도에만 의존한다:

$$I_{\text{diffuse}} = k_d \cdot I_l \cdot \max(\mathbf{N} \cdot \mathbf{L}, 0)$$

여기서:
- $k_d$: 확산 반사율 계수 (재질 색상, 일반적으로 RGB 튜플)
- $I_l$: 광원 강도 (색상)
- $\mathbf{N}$: 단위 표면 법선
- $\mathbf{L}$: 표면 점에서 광원을 향하는 단위 벡터
- $\max(\ldots, 0)$: 음수 기여를 방지하기 위한 클램프 (빛과 반대 방향을 보는 표면)

**물리적 해석**: $\mathbf{N} \cdot \mathbf{L} = \cos\theta_i$ (여기서 $\theta_i$는 입사각). $\theta_i = 0$일 때(빛이 수직으로 닿을 때) 표면은 최대 조명을 받는다. $\theta_i = 90°$일 때 표면은 빛에 수직이 되어 조명을 전혀 받지 못한다.

### 2.3 반사광 성분(Specular Component) — 퐁

정반사는 광택 표면에 밝은 하이라이트를 만든다. 퐁 모델에서 반사광 강도는 반사 벡터 $\mathbf{R}$과 시선 방향 $\mathbf{V}$ 사이의 각도에 의존한다:

$$I_{\text{specular}} = k_s \cdot I_l \cdot \max(\mathbf{R} \cdot \mathbf{V}, 0)^n$$

여기서:
- $k_s$: 반사광 반사율 계수
- $\mathbf{R}$: $\mathbf{N}$에 대한 $\mathbf{L}$의 반사: $\mathbf{R} = 2(\mathbf{N} \cdot \mathbf{L})\mathbf{N} - \mathbf{L}$
- $\mathbf{V}$: 표면 점에서 카메라를 향하는 단위 벡터
- $n$: 광택 지수(높을수록 더 날카롭고 작은 하이라이트)

| 광택도 $n$ | 표면 외관 |
|------------|----------|
| 1-10 | 매우 거칠고 넓은 하이라이트 (고무) |
| 30-50 | 보통 광택 (플라스틱) |
| 100-500 | 매우 광택 있는 표면 (연마된 금속) |

### 2.4 완전한 퐁 모델

$$I = k_a I_a + \sum_{\text{lights}} \left[ k_d I_l (\mathbf{N} \cdot \mathbf{L}) + k_s I_l (\mathbf{R} \cdot \mathbf{V})^n \right]$$

```python
import numpy as np

def normalize(v):
    """Normalize a vector to unit length."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

def reflect(incident, normal):
    """
    Compute the reflection of a vector about a normal.

    The reflection formula: R = 2(N.L)N - L
    This mirrors the incident direction across the surface normal.
    """
    return 2.0 * np.dot(normal, incident) * normal - incident

def phong_shading(point, normal, view_pos, light_pos, light_color,
                  ka=0.1, kd=0.7, ks=0.3, shininess=32.0,
                  ambient_color=None, object_color=None):
    """
    Compute Phong shading at a surface point.

    Parameters:
        point: surface position (3D)
        normal: surface normal (unit vector)
        view_pos: camera position (3D)
        light_pos: light position (3D)
        light_color: light RGB color
        ka, kd, ks: ambient, diffuse, specular coefficients
        shininess: specular exponent (higher = sharper highlight)
        ambient_color: color of ambient light (defaults to light_color)
        object_color: diffuse material color (defaults to white)

    Returns:
        RGB color (numpy array)
    """
    if ambient_color is None:
        ambient_color = np.array([1.0, 1.0, 1.0])
    if object_color is None:
        object_color = np.array([1.0, 1.0, 1.0])

    N = normalize(normal)
    L = normalize(light_pos - point)     # Direction to light
    V = normalize(view_pos - point)      # Direction to camera
    R = reflect(L, N)                    # Reflection of light direction

    # Ambient: constant base illumination
    ambient = ka * ambient_color * object_color

    # Diffuse: Lambert's cosine law
    diff_factor = max(np.dot(N, L), 0.0)
    diffuse = kd * diff_factor * light_color * object_color

    # Specular: shiny highlight
    spec_factor = max(np.dot(R, V), 0.0) ** shininess
    specular = ks * spec_factor * light_color

    return np.clip(ambient + diffuse + specular, 0.0, 1.0)
```

---

## 3. 블린-퐁 모델

Jim Blinn(1977)은 반사 벡터 $\mathbf{R}$ 대신 **하프웨이 벡터(Halfway Vector)** $\mathbf{H}$를 사용하는 수정안을 제안했다:

$$\mathbf{H} = \text{normalize}(\mathbf{L} + \mathbf{V})$$

반사광 항은 다음과 같이 된다:

$$I_{\text{specular}} = k_s \cdot I_l \cdot \max(\mathbf{N} \cdot \mathbf{H}, 0)^n$$

### 3.1 블린-퐁을 사용하는 이유

1. **효율성**: $\mathbf{H}$ 계산이 $\mathbf{R}$ 계산보다 저렴하다 (덧셈 + 정규화 vs. 세 번의 곱셈)
2. **더 나은 물리적 근거**: 하프웨이 벡터 모델은 마이크로패싯(Microfacet) 이론(PBR의 기반)을 더 잘 근사한다
3. **안정성**: 방향 광원(Directional Light, $\mathbf{L}$이 상수인 경우)에서 $\mathbf{H}$는 $\mathbf{V}$에만 의존한다
4. **업계 표준**: OpenGL의 고정 기능 파이프라인이 블린-퐁을 사용했다

> **참고**: 블린-퐁은 동일한 지수 $n$에 대해 퐁보다 약간 더 넓은 하이라이트를 생성한다. 퐁의 외관에 맞추려면 대략 $n_{\text{Blinn}} \approx 4 \times n_{\text{Phong}}$을 사용한다.

```python
def blinn_phong_shading(point, normal, view_pos, light_pos, light_color,
                         ka=0.1, kd=0.7, ks=0.3, shininess=64.0,
                         object_color=None):
    """
    Blinn-Phong shading model.

    The key difference from Phong: we use the halfway vector H instead
    of the reflection vector R. This is both faster to compute and
    more physically plausible.
    """
    if object_color is None:
        object_color = np.array([1.0, 1.0, 1.0])

    N = normalize(normal)
    L = normalize(light_pos - point)
    V = normalize(view_pos - point)

    # The halfway vector: the "average" of light and view directions
    # Physically, N.H measures how well the surface is oriented to
    # reflect light from L toward V
    H = normalize(L + V)

    # Ambient
    ambient = ka * object_color

    # Diffuse (same as Phong)
    diff = max(np.dot(N, L), 0.0)
    diffuse = kd * diff * light_color * object_color

    # Specular: N.H instead of R.V
    spec = max(np.dot(N, H), 0.0) ** shininess
    specular = ks * spec * light_color

    return np.clip(ambient + diffuse + specular, 0.0, 1.0)
```

---

## 4. 셰이딩 보간 방식

셰이딩 모델은 특정 지점에서 색상을 계산한다. 그런데 정확히 어디서 평가해야 할까 — 삼각형마다, 꼭짓점마다, 아니면 픽셀마다?

### 4.1 플랫 셰이딩(Flat Shading)

면 법선(Face Normal)을 사용하여 **삼각형마다 한 번** 조명을 계산한다. 삼각형 전체가 동일한 색상을 갖는다.

- **장점**: 빠르고 매우 저렴하다
- **단점**: 각진 외관 — 개별 삼각형이 뚜렷하게 보인다
- **사용처**: 로우폴리(Low-poly) 아트 스타일, 극도로 성능이 제한된 시나리오

### 4.2 고로 셰이딩(Gouraud Shading)

꼭짓점 법선(Vertex Normal)을 사용하여 **각 꼭짓점에서** 조명을 계산한 후, 무게중심 좌표를 사용하여 삼각형 전체에 **색상을 보간**한다.

- **장점**: 확산 조명에서 부드러운 외관, 꼭짓점마다만 조명을 계산하면 된다
- **단점**: 반사광 하이라이트가 꼭짓점 사이에 있을 경우 놓치거나 왜곡될 수 있다
- **사용처**: 초기 GPU (고정 기능 시대), 매우 단순한 셰이딩

### 4.3 퐁 셰이딩(Phong Shading) — 보간 방식

삼각형 전체에 **법선을 보간**한 후, 보간된 법선을 사용하여 **각 픽셀(프래그먼트)에서** 조명을 계산한다.

> **명칭 정리**: "퐁 셰이딩"은 *반사 모델*(섹션 2)과 이 *보간 방식* 모두를 가리킬 수 있다. 이 둘은 같은 논문에서 소개되었지만 독립적인 개념이다. 예를 들어, 블린-퐁 반사 모델에 퐁 보간 방식을 사용할 수 있다.

- **장점**: 정확한 반사광 하이라이트, 거친 메시에서도 부드러운 외관
- **단점**: 비용이 더 높다 (꼭짓점이 아닌 픽셀마다 조명 계산)
- **사용처**: 모든 현대 렌더링 (프래그먼트 셰이더가 픽셀당 조명 계산)

```python
def shade_triangle_flat(v0, v1, v2, light_pos, view_pos, light_color):
    """
    Flat shading: compute one color for the entire triangle.

    The face normal is the cross product of two edges.
    """
    edge1 = v1['pos'] - v0['pos']
    edge2 = v2['pos'] - v0['pos']
    face_normal = normalize(np.cross(edge1, edge2))
    centroid = (v0['pos'] + v1['pos'] + v2['pos']) / 3.0

    color = blinn_phong_shading(centroid, face_normal, view_pos,
                                 light_pos, light_color)
    return color  # Same color for every pixel in the triangle


def shade_triangle_gouraud(alpha, beta, gamma,
                            color_v0, color_v1, color_v2):
    """
    Gouraud shading: interpolate pre-computed vertex colors.

    Lighting is computed ONCE per vertex (before rasterization).
    During rasterization, we just interpolate the resulting colors.
    """
    return alpha * color_v0 + beta * color_v1 + gamma * color_v2


def shade_triangle_phong_interp(alpha, beta, gamma,
                                 normal_v0, normal_v1, normal_v2,
                                 frag_pos, view_pos, light_pos, light_color):
    """
    Phong interpolation: interpolate normals, compute lighting per pixel.

    This is more expensive than Gouraud but produces much better
    specular highlights. All modern GPUs do this in the fragment shader.
    """
    # Interpolate normal (must re-normalize after interpolation!)
    # Why re-normalize? Linear interpolation of unit vectors does NOT
    # produce a unit vector in general (the result is slightly shorter).
    interp_normal = alpha * normal_v0 + beta * normal_v1 + gamma * normal_v2
    interp_normal = normalize(interp_normal)

    return blinn_phong_shading(frag_pos, interp_normal, view_pos,
                                light_pos, light_color)
```

### 4.4 시각적 비교

```
Flat Shading:           Gouraud Shading:        Phong Interpolation:
┌───┐                   ┌───────┐               ┌───────────┐
│   │ Each face         │ ░░▓▓▓ │ Colors        │  ░░░▓▓▓   │ Normals
│   │ = one color       │ ░░▓▓▓ │ smoothly      │ ░░▓▓▓▓▓░  │ smoothly
├───┤                   │ ░░░▓▓ │ interpolated  │ ░░▓▓●▓▓░  │ interpolated,
│   │ Visible           │ ░░░░▓ │               │ ░░▓▓▓▓▓░  │ lighting per
│   │ facets            └───────┘               │  ░░░▓▓▓   │ pixel -- sharp
└───┘                   Highlight may           └───────────┘ specular dot!
                        be missed!
```

---

## 5. 물리 기반 렌더링(PBR)

### 5.1 동기

퐁/블린-퐁 모델은 수십 년간 잘 사용되었지만 근본적인 한계가 있다:
- 파라미터($k_a$, $k_d$, $k_s$, $n$)가 물리적으로 의미가 없다
- 재질이 에너지를 보존하지 않는다 (반사광이 입사광을 초과할 수 있음)
- 금속, 거친 표면, 또는 프레넬(Fresnel) 효과를 정확히 표현할 수 없다
- 아티스트가 조명 설정마다 파라미터를 다시 조정해야 한다

**PBR**은 셰이딩 모델을 물리학에 근거하여 이러한 문제를 해결한다:
- **에너지 보존(Energy Conservation)**: 반사광은 절대 입사광을 초과하지 않는다
- **프레넬 효과(Fresnel Effect)**: 모든 재질은 스침각에서 더 반사적이 된다
- **마이크로패싯 이론(Microfacet Theory)**: 표면 거칠기를 미세한 거울들의 통계적 분포로 모델링한다
- **두 가지 직관적 파라미터**: 거칠기(Roughness)와 금속성(Metallicness)

### 5.2 렌더링 방정식

PBR은 **렌더링 방정식**(Kajiya, 1986)에 기반한다:

$$L_o(\mathbf{p}, \omega_o) = L_e(\mathbf{p}, \omega_o) + \int_\Omega f_r(\mathbf{p}, \omega_i, \omega_o) \cdot L_i(\mathbf{p}, \omega_i) \cdot (\omega_i \cdot \mathbf{n}) \, d\omega_i$$

여기서:
- $L_o$: 나가는 복사휘도(Radiance) (우리가 보는 것)
- $L_e$: 방출 복사휘도 (광원의 경우)
- $f_r$: **BRDF**(양방향 반사율 분포 함수, Bidirectional Reflectance Distribution Function)
- $L_i$: 방향 $\omega_i$에서 오는 입사 복사휘도
- $\omega_i \cdot \mathbf{n} = \cos\theta_i$: 람베르트 코사인 인수

실시간 렌더링에서는 이산 광원을 합산하여 적분을 근사한다:

$$L_o \approx \sum_{\text{lights}} f_r(\omega_i, \omega_o) \cdot L_i \cdot (\mathbf{N} \cdot \mathbf{L})$$

### 5.3 쿡-토런스(Cook-Torrance) BRDF

표준 실시간 PBR BRDF는 반사율을 확산 및 반사광으로 분리한다:

$$f_r = k_d \cdot f_{\text{Lambert}} + k_s \cdot f_{\text{Cook-Torrance}}$$

**확산 항**(람베르트):

$$f_{\text{Lambert}} = \frac{\text{albedo}}{\pi}$$

$\frac{1}{\pi}$ 인수는 에너지 보존을 보장한다 (람베르트 표면은 빛을 반구 입체각 $2\pi$에 분산시키고, $\cos\theta$ 가중치가 $\pi$로 적분됨).

**반사광 항**(쿡-토런스):

$$f_{\text{Cook-Torrance}} = \frac{D(\mathbf{H}) \cdot F(\mathbf{V}, \mathbf{H}) \cdot G(\mathbf{L}, \mathbf{V}, \mathbf{H})}{4 \cdot (\mathbf{N} \cdot \mathbf{L}) \cdot (\mathbf{N} \cdot \mathbf{V})}$$

여기서:
- $D$: **법선 분포 함수(NDF, Normal Distribution Function)** — 얼마나 많은 마이크로패싯이 $\mathbf{H}$ 방향을 향하는지
- $F$: **프레넬 함수(Fresnel Function)** — 시야각에 따라 변하는 반사율
- $G$: **기하 함수(Geometry Function)** — 마이크로패싯의 자기 그림자(Self-Shadowing)와 마스킹(Masking)
- 분모는 정규화 인수다

### 5.4 법선 분포 함수: GGX / Trowbridge-Reitz

GGX 분포(Trowbridge-Reitz라고도 불림)는 마이크로패싯 방향의 통계적 분포를 모델링한다:

$$D_{\text{GGX}}(\mathbf{H}) = \frac{\alpha^2}{\pi \left[(\mathbf{N} \cdot \mathbf{H})^2 (\alpha^2 - 1) + 1\right]^2}$$

여기서 $\alpha = \text{roughness}^2$ (거칠기는 아티스트가 다루는 파라미터로 $[0, 1]$ 범위).

- $\alpha \rightarrow 0$ (완전히 매끄러울 때): $D$는 날카로운 스파이크가 된다 (거울)
- $\alpha \rightarrow 1$ (완전히 거칠 때): $D$는 넓은 돔 형태가 된다 (무광)

```python
def distribution_ggx(n_dot_h, roughness):
    """
    GGX / Trowbridge-Reitz Normal Distribution Function.

    Models the probability that a microfacet's normal aligns with
    the halfway vector H. Rougher surfaces have more spread-out
    microfacet orientations, producing wider specular highlights.

    The alpha = roughness^2 remapping provides a more perceptually
    linear roughness slider for artists.
    """
    alpha = roughness * roughness
    a2 = alpha * alpha
    n_dot_h2 = n_dot_h * n_dot_h

    denom = n_dot_h2 * (a2 - 1.0) + 1.0
    denom = np.pi * denom * denom

    return a2 / max(denom, 1e-10)
```

### 5.5 프레넬 방정식: 슐릭 근사(Schlick's Approximation)

**프레넬 효과(Fresnel Effect)**는 스침각에서 표면 반사율이 증가하는 현상을 설명한다. 호수를 생각해 보자: 바로 아래를 보면 물이 투명하게 보이지만, 얕은 각도에서는 표면이 거울처럼 된다.

정확한 프레넬 방정식은 복잡하다. 슐릭 근사는 정확하고 빠르다:

$$F(\mathbf{V}, \mathbf{H}) = F_0 + (1 - F_0)(1 - \mathbf{V} \cdot \mathbf{H})^5$$

여기서 $F_0$는 **기본 반사율(Base Reflectance)** — 표면을 정면으로 보았을 때의 반사율:
- **유전체(Dielectrics)** (비금속): $F_0 \approx 0.04$ (정면에서 4% 반사)
- **금속(Metals)**: $F_0$ = 금속의 색상 (금: $(1.0, 0.765, 0.336)$, 철: $(0.56, 0.57, 0.58)$)

```python
def fresnel_schlick(cos_theta, f0):
    """
    Schlick's approximation of the Fresnel equation.

    At normal incidence (cos_theta = 1): returns F0 (base reflectance)
    At grazing angle (cos_theta -> 0): returns 1.0 (fully reflective)

    This captures the universal physical phenomenon where all
    surfaces become more reflective when viewed at shallow angles.
    """
    return f0 + (1.0 - f0) * (1.0 - cos_theta) ** 5
```

### 5.6 기하 함수: Smith GGX

기하 함수는 마이크로패싯의 **자기 그림자(Self-Shadowing)** (다른 마이크로패싯에 의해 차단된 입사광)와 **마스킹(Masking)** (뷰어에 도달하기 전에 차단된 반사광)을 고려한다.

스미스(Smith) 공식은 이 두 가지를 독립적인 항으로 분리한다:

$$G(\mathbf{L}, \mathbf{V}) = G_1(\mathbf{L}) \cdot G_1(\mathbf{V})$$

슐릭-GGX 근사를 사용하면:

$$G_1(\mathbf{X}) = \frac{\mathbf{N} \cdot \mathbf{X}}{(\mathbf{N} \cdot \mathbf{X})(1 - k) + k}$$

여기서:
- 직접 조명의 경우: $k = \frac{(\alpha + 1)^2}{8}$
- IBL(이미지 기반 조명, Image-Based Lighting)의 경우: $k = \frac{\alpha^2}{2}$

```python
def geometry_schlick_ggx(n_dot_v, roughness):
    """
    Schlick-GGX geometry function (one direction).

    Models self-occlusion of microfacets: on rough surfaces,
    microscopic peaks block light from reaching or leaving
    nearby valleys.
    """
    r = roughness + 1.0
    k = (r * r) / 8.0  # k for direct lighting

    return n_dot_v / (n_dot_v * (1.0 - k) + k)

def geometry_smith(n_dot_v, n_dot_l, roughness):
    """
    Smith's geometry function: combines shadowing and masking.

    G(L,V) = G1(L) * G1(V)
    Both the incoming light direction and outgoing view direction
    can be partially blocked by microfacet geometry.
    """
    ggx_v = geometry_schlick_ggx(max(n_dot_v, 0.0), roughness)
    ggx_l = geometry_schlick_ggx(max(n_dot_l, 0.0), roughness)
    return ggx_v * ggx_l
```

### 5.7 메탈릭-러프니스(Metallic-Roughness) 워크플로

현대 PBR은 두 가지 직관적인 파라미터를 사용한다:

| 파라미터 | 범위 | 효과 |
|---------|------|------|
| **거칠기(Roughness)** | $[0, 1]$ | 0 = 거울처럼 매끄러움, 1 = 완전한 무광 |
| **금속성(Metallic)** | $[0, 1]$ | 0 = 유전체 (플라스틱, 나무), 1 = 금속 (금, 철) |

금속성 파라미터는 다음을 결정한다:
- $F_0$: `lerp(0.04, albedo, metallic)` — 금속은 색상을 기본 반사율로 사용
- 확산 기여: `albedo * (1 - metallic)` — 금속은 확산 성분이 없다

```python
def pbr_shading(point, normal, view_pos, light_pos, light_color,
                albedo, metallic, roughness):
    """
    Complete PBR shading using the Cook-Torrance BRDF.

    This is the shading model used by Unity, Unreal Engine, Godot,
    and virtually every modern rendering engine.

    Parameters:
        point: surface position
        normal: surface normal (unit)
        view_pos: camera position
        light_pos: light position
        light_color: light RGB color/intensity
        albedo: base color of the surface (RGB)
        metallic: 0 = dielectric, 1 = metal
        roughness: 0 = mirror, 1 = matte
    """
    N = normalize(normal)
    V = normalize(view_pos - point)
    L = normalize(light_pos - point)
    H = normalize(V + L)

    # Dot products (clamped to non-negative)
    n_dot_l = max(np.dot(N, L), 0.0)
    n_dot_v = max(np.dot(N, V), 0.0)
    n_dot_h = max(np.dot(N, H), 0.0)
    v_dot_h = max(np.dot(V, H), 0.001)

    # Base reflectance: dielectrics use 0.04, metals use albedo
    f0 = np.full(3, 0.04)
    f0 = f0 * (1.0 - metallic) + albedo * metallic

    # Cook-Torrance specular BRDF components
    D = distribution_ggx(n_dot_h, roughness)
    F = fresnel_schlick(v_dot_h, f0)
    G = geometry_smith(n_dot_v, n_dot_l, roughness)

    # Specular contribution
    numerator = D * F * G
    denominator = 4.0 * n_dot_v * n_dot_l + 0.0001  # Prevent division by zero
    specular = numerator / denominator

    # Energy conservation: what's not reflected is refracted (diffuse)
    # Metals have no diffuse component (all light is reflected)
    ks = F  # Specular coefficient = Fresnel
    kd = (1.0 - ks) * (1.0 - metallic)

    # Diffuse: Lambertian (divided by pi for energy conservation)
    diffuse = kd * albedo / np.pi

    # Final color: (diffuse + specular) * light * cos_theta
    lo = (diffuse + specular) * light_color * n_dot_l

    # Tone mapping (Reinhard) -- compress HDR to displayable range
    lo = lo / (lo + 1.0)

    # Gamma correction (linear to sRGB)
    lo = np.power(np.clip(lo, 0.0, 1.0), 1.0 / 2.2)

    return lo
```

---

## 6. 구(Sphere)에서 퐁과 PBR 비교

```python
"""
Render a sphere with both Phong and PBR shading for comparison.

The sphere is rendered analytically (ray-sphere intersection) rather
than via rasterization, to focus purely on the shading computation.
"""

import numpy as np

def render_sphere_comparison(width=400, height=200):
    """
    Render a sphere with Phong (left half) and PBR (right half).
    """
    image = np.zeros((height, width, 3), dtype=float)

    # Scene setup
    sphere_center = np.array([0.0, 0.0, -3.0])
    sphere_radius = 1.0
    light_pos = np.array([2.0, 3.0, 0.0])
    light_color = np.array([1.0, 1.0, 1.0]) * 3.0  # Bright white
    view_pos = np.array([0.0, 0.0, 0.0])

    # PBR material: slightly rough, non-metallic
    albedo = np.array([0.8, 0.2, 0.2])  # Red
    metallic = 0.0
    roughness = 0.4

    aspect = width / height

    for y in range(height):
        for x in range(width):
            # Ray from camera through pixel
            # Map pixel to [-1, 1] range
            u = (2.0 * x / width - 1.0) * aspect
            v = 1.0 - 2.0 * y / height
            ray_dir = normalize(np.array([u, v, -1.0]))

            # Ray-sphere intersection
            oc = view_pos - sphere_center
            a = np.dot(ray_dir, ray_dir)
            b = 2.0 * np.dot(oc, ray_dir)
            c = np.dot(oc, oc) - sphere_radius ** 2
            discriminant = b * b - 4 * a * c

            if discriminant < 0:
                image[y, x] = [0.05, 0.05, 0.08]  # Background
                continue

            t = (-b - np.sqrt(discriminant)) / (2 * a)
            if t < 0:
                image[y, x] = [0.05, 0.05, 0.08]
                continue

            # Hit point and normal
            hit_point = view_pos + t * ray_dir
            normal = normalize(hit_point - sphere_center)

            # Left half: Phong shading
            if x < width // 2:
                color = phong_shading(
                    hit_point, normal, view_pos, light_pos,
                    light_color, ka=0.1, kd=0.7, ks=0.5,
                    shininess=64.0, object_color=albedo
                )
            # Right half: PBR shading
            else:
                color = pbr_shading(
                    hit_point, normal, view_pos, light_pos,
                    light_color, albedo=albedo,
                    metallic=metallic, roughness=roughness
                )

            image[y, x] = color

    return image


def render_material_grid(width=600, height=400):
    """
    Render a grid of spheres with varying metallic and roughness values.

    This demonstrates how PBR's two parameters intuitively control appearance:
    - Rows: roughness from 0.1 (top, smooth) to 0.9 (bottom, rough)
    - Columns: metallic from 0 (left, dielectric) to 1 (right, metal)
    """
    image = np.zeros((height, width, 3), dtype=float)

    rows, cols = 5, 5
    sphere_radius = 0.35
    light_pos = np.array([5.0, 5.0, 5.0])
    light_color = np.array([1.0, 1.0, 1.0]) * 5.0
    view_pos = np.array([0.0, 0.0, 0.0])
    albedo = np.array([0.9, 0.6, 0.2])  # Gold-like base color

    for row in range(rows):
        for col in range(cols):
            roughness = 0.1 + 0.8 * row / (rows - 1)
            metallic = col / (cols - 1)

            # Center of this sphere on screen
            cx = (col + 0.5) / cols * width
            cy = (row + 0.5) / rows * height
            pix_radius = min(width / cols, height / rows) * 0.4

            # Render this sphere
            for dy in range(int(-pix_radius), int(pix_radius) + 1):
                for dx in range(int(-pix_radius), int(pix_radius) + 1):
                    px = int(cx + dx)
                    py = int(cy + dy)
                    if not (0 <= px < width and 0 <= py < height):
                        continue

                    # Map pixel offset to sphere surface
                    nx = dx / pix_radius
                    ny = -dy / pix_radius
                    r2 = nx * nx + ny * ny
                    if r2 > 1.0:
                        continue

                    nz = np.sqrt(1.0 - r2)
                    normal = np.array([nx, ny, nz])
                    point = np.array([nx, ny, nz - 3.0])

                    color = pbr_shading(
                        point, normal, view_pos, light_pos,
                        light_color, albedo=albedo,
                        metallic=metallic, roughness=roughness
                    )
                    image[py, px] = color

    return image


if __name__ == "__main__":
    print("Rendering Phong vs PBR comparison...")
    comparison = render_sphere_comparison()
    print(f"Comparison image shape: {comparison.shape}")

    print("Rendering material grid (roughness x metallic)...")
    grid = render_material_grid()
    print(f"Grid image shape: {grid.shape}")

    try:
        from PIL import Image
        img1 = Image.fromarray((comparison * 255).astype(np.uint8))
        img1.save('phong_vs_pbr.png')
        print("Saved phong_vs_pbr.png")

        img2 = Image.fromarray((grid * 255).astype(np.uint8))
        img2.save('material_grid.png')
        print("Saved material_grid.png")
    except ImportError:
        print("Install Pillow for image output: pip install Pillow")
```

---

## 7. 톤 매핑(Tone Mapping)과 감마 보정(Gamma Correction)

### 7.1 톤 매핑이 필요한 이유

PBR은 **선형, 고동적 범위(HDR, High Dynamic Range)** 공간에서 동작한다. 빛의 강도는 임의로 클 수 있다 (태양은 촛불보다 수백만 배 밝다). 하지만 디스플레이는 $[0, 1]$ 범위의 값만 표시할 수 있다. **톤 매핑(Tone Mapping)**은 HDR 값을 표시 가능한 범위로 압축한다.

**라인하르트(Reinhard) 톤 매핑** (단순):

$$L_{\text{display}} = \frac{L}{L + 1}$$

**ACES 필름 톤 매핑** (업계 표준):

$$f(x) = \frac{x(2.51x + 0.03)}{x(2.43x + 0.59) + 0.14}$$

### 7.2 감마 보정

디스플레이는 비선형적이다: 감마 곡선($\text{output} = \text{input}^{2.2}$)을 적용한다. 이를 보정하기 위해 색상을 디스플레이에 보내기 전에 역감마를 적용한다:

$$C_{\text{sRGB}} = C_{\text{linear}}^{1/2.2}$$

**PBR 계산은 반드시 선형 공간에서 수행해야 한다.** sRGB로 저장된 텍스처는 입력 시 선형으로 변환해야 하고, 결과는 출력 시 다시 sRGB로 변환해야 한다.

---

## 8. 다중 광원과 광원 유형

### 8.1 빛 기여 합산

다중 광원의 경우, 단순히 기여를 합산한다 (렌더링 방정식은 입사 복사휘도에 대해 선형):

$$L_o = \sum_{i=1}^{n} f_r(\omega_i, \omega_o) \cdot L_i \cdot (\mathbf{N} \cdot \mathbf{L}_i)$$

### 8.2 광원 유형

| 광원 유형 | 설명 | 감쇠(Attenuation) |
|----------|------|------------------|
| **방향 광원(Directional)** | 무한히 먼 곳 (태양) | 없음 (상수) |
| **점 광원(Point)** | 모든 방향으로 방출 | $\frac{1}{d^2}$ (역제곱 법칙) |
| **스팟 광원(Spot)** | 각도 제한이 있는 점 광원 | $\frac{1}{d^2} \cdot \text{cone\_factor}$ |
| **면적 광원(Area)** | 유한한 표면 (현실적) | 복잡함 (샘플링 또는 LTC 필요) |

```python
def point_light_attenuation(light_pos, frag_pos):
    """
    Inverse-square attenuation for point lights.

    This follows the physical law: light intensity decreases with
    the square of distance. A light twice as far away is 1/4 as bright.
    """
    distance = np.linalg.norm(light_pos - frag_pos)
    return 1.0 / (distance * distance + 0.0001)
```

---

## 요약

| 모델 | 유형 | 파라미터 | 에너지 보존? | 사용처 |
|------|------|---------|------------|--------|
| **Phong** | 경험적 | $k_a, k_d, k_s, n$ | 아니오 | 학습, 단순 애플리케이션 |
| **Blinn-Phong** | 경험적 | 동일, $\mathbf{H}$ 사용 | 아니오 | 레거시 엔진 |
| **Cook-Torrance PBR** | 물리적 | albedo, roughness, metallic | 예 | 모든 현대 엔진 |

| 보간 방식 | 조명 계산 위치 | 품질 | 비용 |
|---------|-------------|------|------|
| **Flat** | 면마다 한 번 | 각진 외관 | 가장 저렴 |
| **Gouraud** | 꼭짓점마다 | 부드러움 (반사광 놓칠 수 있음) | 보통 |
| **Phong 보간** | 픽셀마다 | 최상 (정확한 반사광) | 가장 비쌈 |

**핵심 정리**:
- 퐁 모델의 세 성분(주변광 + 확산 + 반사광)은 핵심적인 시각 요소를 포착한다
- 블린-퐁의 하프웨이 벡터는 더 효율적이면서도 물리적으로 더 타당하다
- PBR (쿡-토런스)은 세 가지 함수와 마이크로패싯 이론을 사용한다: D (분포), F (프레넬), G (기하)
- 메탈릭-러프니스 워크플로는 아티스트에게 모든 재질을 위한 두 가지 직관적인 파라미터를 제공한다
- 모든 PBR 계산은 적절한 톤 매핑과 감마 보정을 적용한 선형 색상 공간에서 수행해야 한다
- 퐁 보간(픽셀당 법선)은 모든 현대 픽셀당 조명의 표준이다

---

## 연습 문제

1. **퐁 성분 분리**: 구를 세 번 렌더링하여 주변광, 확산, 반사광 성분을 각각 따로 표시하라. 그런 다음 합산 결과를 표시하라. 광택 지수 $n$을 변경하면 각 성분이 어떻게 달라지는가?

2. **블린 vs 퐁 반사광 비교**: 동일한 광택도로 퐁 반사광과 블린-퐁 반사광으로 구를 렌더링하라. 하이라이트 크기의 차이를 관찰하라. 어떤 블린-퐁 지수가 퐁 지수 64에 대략적으로 일치하는가?

3. **셰이딩 비교**: 저폴리곤 구(20면)를 플랫, 고로, 퐁 보간으로 렌더링하라. 몇 개의 폴리곤에서 고로 셰이딩된 구가 허용 가능하게 부드럽게 보이기 시작하는가?

4. **PBR 재질 탐색**: 거칠기(0.0~1.0, 행)와 금속성(0.0~1.0, 열)을 변화시킨 5×5 구 격자를 만들어라. 금색 알베도를 사용하라. 완전히 금속적이고 완전히 매끄러운 구가 거울처럼 보이는 반면, 완전히 금속적이고 완전히 거친 구는 왜 브러시드 메탈처럼 보이는지 설명하라.

5. **프레넬 효과**: $\cos\theta$가 0에서 1까지 변할 때 $F_0 = 0.04$ (유전체)와 $F_0 = 0.9$ (금속)에 대한 슐릭 프레넬 함수를 그래프로 그려라. 유전체 표면이 50% 반사적이 되는 각도는 어디인가?

6. **에너지 보존**: $k_d = 0.8$, $k_s = 0.5$인 퐁 모델에서 총 반사광이 입사광을 초과할 수 있음을 보여라 (에너지 보존 위반). 그런 다음 PBR 모델에는 이 문제가 없음을 보여라.

---

## 더 읽을거리

1. Phong, B.T. "Illumination for Computer Generated Pictures" (1975) — 원본 논문
2. Burley, B. "Physically Based Shading at Disney" (SIGGRAPH 2012) — 현대 PBR의 기초
3. [Learn OpenGL - PBR Theory](https://learnopengl.com/PBR/Theory) — 우수한 실용 PBR 튜토리얼
4. [Filament PBR Documentation](https://google.github.io/filament/Filament.html) — 구글의 포괄적인 PBR 참고 자료
5. Hoffman, N. "Background: Physics and Math of Shading" in *Real-Time Rendering* (4th ed.), Ch. 9
