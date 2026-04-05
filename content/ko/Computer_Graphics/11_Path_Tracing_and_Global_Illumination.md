# 11. 패스 트레이싱과 전역 조명

[← 이전: 10. 레이 트레이싱 기초](10_Ray_Tracing_Basics.md) | [다음: 12. 애니메이션과 골격 시스템 →](12_Animation_and_Skeletal_Systems.md)

---

## 학습 목표

1. 광 전달(light transport)의 근본 방정식인 렌더링 방정식(Kajiya 1986)을 이해한다
2. 몬테 카를로 적분(Monte Carlo integration)을 적용하여 무작위 샘플링으로 적분을 추정한다
3. 씬(scene)을 통과하는 빛의 무작위 보행으로서 단순 패스 트레이싱(naive path tracing)을 구현한다
4. 분산(variance)을 줄이기 위해 중요도 샘플링(importance sampling, 코사인 가중치, BRDF 비례)을 사용한다
5. 러시안 룰렛(Russian roulette)을 편향 없는 경로 종료 전략으로 설명한다
6. 더 빠른 수렴을 위한 다음 이벤트 추정(next event estimation, 직접 광 샘플링)을 구현한다
7. 다중 중요도 샘플링(multiple importance sampling, MIS)과 그 장점을 이해한다
8. 디노이징(denoising)을 실용적 필수 요소로 인식하고 현재 접근법을 살펴본다

---

## 왜 중요한가

휘티드 방식 레이 트레이싱(Whitted-style ray tracing, L10)은 거울 반사와 선명한 그림자를 아름답게 처리하지만, 실세계는 **부드러운** 효과들로 가득하다. 붉은 벽에서 흰 천장으로 번지는 색 번짐(color bleeding), 코너를 밝게 만드는 확산 상호 반사(diffuse inter-reflections), 유리를 통해 집중된 빛으로 생기는 코스틱(caustics) 등이 그것이다. 이 효과들은 빛이 *모든* 표면에서 *모든* 방향으로 튀면서 발생하는데, 이를 **전역 조명(global illumination)**이라고 한다.

패스 트레이싱(path tracing)은 이러한 완전한 광 전달을 충실하게 시뮬레이션하는 알고리즘이다. 영화에 쓰이는 주요 오프라인 렌더러(Arnold, RenderMan, Cycles, Manuka)의 기반이며, 실시간 엔진에서도 점점 더 많이 등장하고 있다. 사진처럼 현실적인 CG 이미지를 볼 때, 패스 트레이싱이 만들어냈을 가능성이 매우 높다.

---

## 1. 렌더링 방정식(Rendering Equation)

### 1.1 수식

James Kajiya는 1986년 **렌더링 방정식**을 발표했다. 이 방정식은 점 $\mathbf{p}$에서 방향 $\omega_o$로 나가는 총 복사 휘도(radiance) $L_o$를 설명한다:

$$L_o(\mathbf{p}, \omega_o) = L_e(\mathbf{p}, \omega_o) + \int_{\Omega} f_r(\mathbf{p}, \omega_i, \omega_o) \, L_i(\mathbf{p}, \omega_i) \, (\omega_i \cdot \mathbf{n}) \, d\omega_i$$

여기서:
- $L_e(\mathbf{p}, \omega_o)$는 **방출(emitted)** 복사 휘도 (광원에서만 0이 아님)
- $f_r(\mathbf{p}, \omega_i, \omega_o)$는 $\mathbf{p}$에서의 **BRDF(Bidirectional Reflectance Distribution Function)**
- $L_i(\mathbf{p}, \omega_i)$는 방향 $\omega_i$에서 들어오는 **입사(incoming)** 복사 휘도
- $\omega_i \cdot \mathbf{n} = \cos\theta_i$는 코사인 단축(foreshortening) 항
- $\Omega$는 $\mathbf{p}$에서의 표면 위 반구(hemisphere) 방향 집합

**핵심 통찰**: 한 점의 $L_i$는 다른 점의 $L_o$에 의존하므로, 이는 재귀적인 적분 방정식이 된다. 일반적인 씬에서 닫힌 형태의 해(closed-form solution)는 존재하지 않으므로, 수치적으로 풀어야 한다.

### 1.2 각 항의 이해

**방출(Emission)** $L_e$: 광원이 복사 휘도를 방출한다. 비방출(non-emissive) 표면에서는 $L_e = 0$.

**BRDF** $f_r$: $\omega_i$에서 들어온 빛이 $\omega_o$ 방향으로 산란되는 방식을 설명한다. 예시:
- **램버시안(Lambertian, 확산)**: $f_r = \frac{\rho}{\pi}$ (여기서 $\rho$는 알베도)
- **완벽한 거울**: 디랙 델타 함수 — 모든 빛이 반사 방향으로 진행
- **광택(Glossy)**: 반사 방향 주변에 집중 (예: Cook-Torrance 마이크로패싯 모델)

**코사인 항** $\cos\theta_i$: 그레이징 각도(grazing angle)로 들어오는 빛은 더 넓은 면적을 비추므로 기여가 줄어든다.

### 1.3 왜 그냥 적분하지 않는가?

적분은 반구의 방향에 대한 것이며, 각 방향에 대해 광선을 추적하여 무엇이 보이는지 찾아야 $L_i$를 알 수 있다. 그런데 $L_i$는 다음 표면에서의 동일한 적분으로 정의된다. 이 무한 재귀가 전역 조명을 어렵게 만드는 이유다. 패스 트레이싱은 **몬테 카를로 추정(Monte Carlo estimation)**을 통해 이를 처리한다.

---

## 2. 몬테 카를로 적분(Monte Carlo Integration)

### 2.1 기본 개념

적분 $I = \int_a^b f(x)\,dx$를 추정하기 위해, $[a, b]$에서 균일하게 $N$개의 무작위 샘플 $x_k$를 뽑는다:

$$I \approx \frac{b - a}{N} \sum_{k=1}^{N} f(x_k)$$

이 **몬테 카를로 추정량(Monte Carlo estimator)**은 $N \to \infty$일 때 참 적분으로 수렴하며, 오차는 $\frac{1}{\sqrt{N}}$에 비례한다. 정확도를 두 배로 높이려면 샘플을 네 배 늘려야 한다.

### 2.2 PDF를 사용한 일반 추정량

확률 밀도 함수(probability density function) $p(x)$에서 뽑은 샘플(반드시 균일하지 않아도 됨):

$$I = \int f(x)\,dx \approx \frac{1}{N} \sum_{k=1}^{N} \frac{f(x_k)}{p(x_k)}$$

이는 $f(x) \neq 0$인 곳에서 $p(x) > 0$이면 어떤 $p(x)$에도 유효하다. 핵심은 **분산(variance)**(노이즈)을 줄이기 위해 $p$를 잘 선택하는 것이다.

### 2.3 분산과 수렴

몬테 카를로 추정량의 분산은:

$$\text{Var}\left[\frac{f(X)}{p(X)}\right] = \int \left(\frac{f(x)}{p(x)} - I\right)^2 p(x)\,dx$$

$p(x) \propto f(x)$일 때 분산이 최소화된다. 이것이 **중요도 샘플링(importance sampling)**의 원리다. $p$가 $f$와 완벽히 일치하면 분산이 0이 된다(샘플 하나면 충분!).

실제로는 알 수 없는 피적분 함수에 $p$를 완벽하게 맞출 수 없지만, 피적분 함수의 해석적 부분(예: 코사인 항 또는 BRDF)을 이용하여 샘플링을 유도할 수 있다.

---

## 3. 단순 패스 트레이싱(Naive Path Tracing)

### 3.1 알고리즘

가장 단순한 패스 트레이서는 씬을 통과하는 광선을 무작위로 튀기면서 렌더링 방정식을 추정한다:

```
function path_trace(ray, depth):
    if depth > MAX_DEPTH:
        return BLACK

    hit = find_nearest_intersection(ray)
    if no hit:
        return BACKGROUND

    p, n, material = hit

    // 방출: 광원에 닿으면 수집
    color = material.emission

    // 반구 위의 무작위 방향 샘플링
    wi = random_hemisphere_direction(n)

    // 렌더링 방정식 추정량 계산
    // f(x)/p(x) 여기서 f = BRDF * Li * cos(theta), p = 1/(2*pi)
    cos_theta = dot(wi, n)
    brdf = material.brdf(wi, wo)

    // 재귀: 튀긴 광선 추적
    Li = path_trace(Ray(p + epsilon*n, wi), depth + 1)

    color += 2 * pi * brdf * Li * cos_theta

    return color
```

알베도 $\rho$인 램버시안 표면에서 BRDF는 $\frac{\rho}{\pi}$다. 균일 반구 샘플링($p(\omega) = \frac{1}{2\pi}$) 적용 시:

$$\text{color} \approx L_e + \frac{f_r \cdot L_i \cdot \cos\theta}{p(\omega)} = L_e + \frac{\frac{\rho}{\pi} \cdot L_i \cdot \cos\theta}{\frac{1}{2\pi}} = L_e + 2\rho \cdot L_i \cdot \cos\theta$$

### 3.2 노이즈 문제

단순 패스 트레이싱은 동작하지만 수렴이 느리다. 각 바운스마다 무작위 방향 하나가 선택되는데, 이 방향들 중 많은 수가 씬의 어두운 부분을 가리켜 정보를 거의 주지 않는다. 결과적으로 이미지가 매우 **노이즈**가 심해, 픽셀당 수백 또는 수천 개의 샘플이 있어야 깨끗해진다.

---

## 4. 중요도 샘플링(Importance Sampling)

### 4.1 코사인 가중 반구 샘플링(Cosine-Weighted Hemisphere Sampling)

렌더링 방정식에 $\cos\theta$ 인수가 포함되어 있으므로, 이를 샘플링 PDF에 반영할 수 있다. 반구를 균일하게 샘플링하는 대신, $\cos\theta$에 비례하게 샘플링한다:

$$p(\omega) = \frac{\cos\theta}{\pi}$$

두 개의 균일 난수 $(\xi_1, \xi_2) \in [0, 1)^2$로 생성한다:

$$\phi = 2\pi\xi_1, \quad \theta = \arccos\sqrt{1 - \xi_2}$$

또는 원판-반구 매핑(disk-to-hemisphere mapping)을 사용하면:

$$x = \cos\phi\sqrt{\xi_2}, \quad y = \sin\phi\sqrt{\xi_2}, \quad z = \sqrt{1 - \xi_2}$$

$z$ 성분이 표면 법선(normal)과 정렬된다.

이 PDF를 사용하면 램버시안 표면의 몬테 카를로 추정량이 단순해진다:

$$\frac{f_r \cdot L_i \cdot \cos\theta}{p(\omega)} = \frac{\frac{\rho}{\pi} \cdot L_i \cdot \cos\theta}{\frac{\cos\theta}{\pi}} = \rho \cdot L_i$$

코사인 항이 소거되어 분산이 크게 줄어든다.

### 4.2 BRDF 비례 샘플링(BRDF-Proportional Sampling)

광택(glossy) BRDF(예: GGX 마이크로패싯)에 대해서는 BRDF 자체에 비례하는 방향을 샘플링한다. GGX 분포 함수는 알려진 해석적 샘플링 공식이 있으므로, BRDF의 무거운 로브(lobe)가 더 자주 샘플링된다.

### 4.3 비교

| 샘플링 전략 | PDF $p(\omega)$ | 분산 |
|------------|-----------------|------|
| 균일 반구 | $1/(2\pi)$ | 높음 |
| 코사인 가중 | $\cos\theta / \pi$ | 낮음 (추정량의 코사인 소거) |
| BRDF 비례 | $\propto f_r \cdot \cos\theta$ | BRDF 항에서 가장 낮음 |

---

## 5. 러시안 룰렛(Russian Roulette)

### 5.1 동기

패스 트레이싱 경로는 무한히 튈 수 있다. 고정된 최대 깊이를 설정하면 **편향(bias)**이 생긴다(더 깊은 바운스의 에너지를 놓침). **러시안 룰렛**은 편향 없는 종료를 제공한다:

각 바운스에서 확률 $q$로 경로를 종료한다. 경로가 살아남으면(확률 $1 - q$), 기여를 $\frac{1}{1 - q}$로 증폭한다:

$$L \approx \begin{cases} \frac{L_{\text{bounce}}}{1 - q} & \text{확률 } 1 - q \text{ 로} \\ 0 & \text{확률 } q \text{ 로} \end{cases}$$

기댓값은:

$$E[L] = (1 - q) \cdot \frac{L_{\text{bounce}}}{1 - q} + q \cdot 0 = L_{\text{bounce}}$$

이것은 **편향 없음(unbiased)**: 평균적으로 올바른 답을 얻는다. 기여가 적은 경로(예: 어두운 표면에 닿은 후)는 더 공격적으로 종료할 수 있다.

### 5.2 $q$ 선택

일반적인 휴리스틱: 생존 확률을 표면 알베도(또는 최대 BRDF 응답)에 비례하게 설정:

$$q_{\text{terminate}} = 1 - \min(\max(\rho_r, \rho_g, \rho_b), 0.95)$$

밝은 표면은 거의 항상 계속 튀고(상당한 에너지를 전달), 어두운 표면은 일찍 종료된다.

---

## 6. 다음 이벤트 추정(Next Event Estimation, 직접 광 샘플링)

### 6.1 문제

단순 패스 트레이싱에서 빛을 수집하는 유일한 방법은 무작위로 튀긴 광선이 우연히 광원에 닿는 경우다. 작은 광원의 경우 이는 극히 드물어 이미지에 노이즈가 심하다.

### 6.2 해결책

각 교차점에서 광원을 명시적으로 샘플링한다:

1. 광원 위의 점 $\mathbf{q}$ 선택 (예: 면적 광원 위의 무작위 점)
2. $\mathbf{p}$에서 $\mathbf{q}$로 **그림자 광선(shadow ray)** 투사
3. 막히지 않으면(unoccluded) 직접 조명(direct illumination) 기여 추가

이를 **다음 이벤트 추정(Next Event Estimation, NEE)** 또는 **직접 광 샘플링(direct light sampling)**이라고 한다. 면적 $A$인 광원의 기여는:

$$L_{\text{direct}} \approx \frac{f_r(\omega_i, \omega_o) \cdot L_e(\mathbf{q}) \cdot \cos\theta_i \cdot \cos\theta_q}{|\mathbf{p} - \mathbf{q}|^2 \cdot p(\mathbf{q})}$$

여기서 $\cos\theta_q$는 광원의 방향을 고려하고, 균일 샘플링의 경우 $p(\mathbf{q}) = 1/A$다.

**중요**: NEE를 사용할 때, 간접 바운스는 광원에서 직접 방출(emission)을 **수집하지 않아야 한다** (이중 계산 방지). 바운스는 간접 조명만 처리한다.

---

## 7. 다중 중요도 샘플링(Multiple Importance Sampling, MIS)

### 7.1 문제

BRDF 샘플링은 재질이 광택(좁은 로브)일 때 좋지만 작은 광원에는 나쁘다. 광 샘플링은 작은 광원에는 좋지만 선명한 정반사(specular) 표면에는 나쁘다. 어떤 전략도 범용적으로 최선이 아니다.

### 7.2 MIS 해결책

Veach와 Guibas(1995)는 **균형 휴리스틱(balance heuristic)** 가중치로 여러 샘플링 전략을 결합하는 방법을 제시했다:

$$F \approx \sum_{k=1}^{N_{\text{BRDF}}} \frac{w_{\text{BRDF}}(\omega_k) \cdot f_r \cdot L_i \cdot \cos\theta}{p_{\text{BRDF}}(\omega_k)} + \sum_{k=1}^{N_{\text{light}}} \frac{w_{\text{light}}(\omega_k) \cdot f_r \cdot L_i \cdot \cos\theta}{p_{\text{light}}(\omega_k)}$$

BRDF 샘플링에 대한 **균형 휴리스틱** 가중치는:

$$w_{\text{BRDF}}(\omega) = \frac{p_{\text{BRDF}}(\omega)}{p_{\text{BRDF}}(\omega) + p_{\text{light}}(\omega)}$$

MIS는 증명 가능하게 최적에 가깝다. 각 특정 샘플에 대해 더 나은 전략을 자동으로 선호한다. 사실상 모든 프로덕션 패스 트레이서에서 사용된다.

### 7.3 거듭제곱 휴리스틱(Power Heuristic)

실제로는 **거듭제곱 휴리스틱**(지수 $\beta = 2$)이 더 나은 성능을 보인다:

$$w_{\text{BRDF}}(\omega) = \frac{p_{\text{BRDF}}(\omega)^2}{p_{\text{BRDF}}(\omega)^2 + p_{\text{light}}(\omega)^2}$$

---

## 8. 수렴과 노이즈

### 8.1 오차 분석

몬테 카를로 패스 트레이싱은 픽셀당 샘플 수 $N$에 대해 $O(1/\sqrt{N})$으로 수렴한다. 이는 다음을 의미한다:

| 픽셀당 샘플(spp) | 상대 노이즈 |
|-----------------|------------|
| 1 | 1.00 |
| 4 | 0.50 |
| 16 | 0.25 |
| 64 | 0.125 |
| 256 | 0.0625 |
| 1024 | 0.03125 |

노이즈를 절반으로 줄이려면 샘플이 $4$배 필요하다. 프로덕션 렌더에서는 보통 128~4096 spp를 사용한다.

### 8.2 노이즈의 원인

- **낮은 확률 경로**: 코스틱(caustics, 유리를 통해 집중된 빛)은 빛이 드문 경로를 통해 카메라에 도달하므로 악명 높게 어렵다
- **작은 광원**: 작고 밝은 광원은 높은 분산을 유발한다 (닿을 때는 큰 값, 그렇지 않으면 0)
- **광택 상호 반사**: 광택 표면 간의 여러 바운스가 분산을 복합적으로 증가시킨다

### 8.3 분산 감소 기법 요약

| 기법 | 메커니즘 |
|------|---------|
| 중요도 샘플링 | 샘플링 PDF를 피적분 함수에 맞춤 |
| 러시안 룰렛 | 편향 없는 경로 종료 |
| 다음 이벤트 추정 | 각 바운스에서 광원 명시적 샘플링 |
| MIS | 여러 전략을 최적으로 결합 |
| 층화 샘플링(Stratified sampling) | 샘플 공간을 층(strata)으로 분할 |
| 준 몬테 카를로(Quasi-Monte Carlo) | 저-불일치 수열 (Halton, Sobol) |

---

## 9. 디노이징(Denoising)

### 9.1 왜 디노이징이 필요한가?

위의 모든 분산 감소 기법을 적용해도, 낮은 샘플 수에서 패스 트레이싱된 이미지는 노이즈가 심하다. **디노이징**은 노이즈 있는 입력으로부터 깨끗한 이미지를 복원하는 필터링으로, 낮은 샘플 수에서도 프로덕션 품질의 결과를 가능하게 한다.

### 9.2 고전적 접근법

**양방향 필터(Bilateral filter)**: 인접 픽셀을 평균 내되, 공간적 거리와 색상 유사성 모두로 가중치를 준다. 노이즈를 스무딩하면서 에지(edge)를 보존한다:

$$\hat{I}(\mathbf{p}) = \frac{1}{W} \sum_{\mathbf{q}} I(\mathbf{q}) \cdot G_{\sigma_s}(\|\mathbf{p} - \mathbf{q}\|) \cdot G_{\sigma_r}(\|I(\mathbf{p}) - I(\mathbf{q})\|)$$

**A-trous 웨이블릿 필터**: 증가하는 팽창(dilation)을 사용한 다중 스케일 양방향 필터. 실시간 디노이저(SVGF)에서 사용된다.

### 9.3 머신러닝 디노이저

최신 디노이저는 (노이즈, 레퍼런스) 이미지 쌍으로 훈련된 신경망을 사용한다:

- **NVIDIA OptiX AI Denoiser**: 훈련된 CNN; 노이즈 있는 색상, 알베도, 법선 버퍼를 입력으로 받음
- **Intel Open Image Denoise (OIDN)**: 오픈소스 ML 디노이저; GPU 레이 트레이싱 없이 동작
- **SVGF** (Spatiotemporal Variance-Guided Filtering): 실시간 디노이징을 위한 시간적 재투영(temporal reprojection)과 픽셀별 분산 추정 사용

이 디노이저들은 필터링을 유도하기 위해 **보조 버퍼(auxiliary buffers)**(법선, 알베도, 깊이)를 사용한다:
- 노이즈 없는 법선은 디노이저에게 기하학적 에지 위치를 알려줌
- 알베도는 텍스처 디테일을 조명 노이즈에서 분리
- 깊이는 전경과 배경 구분에 도움

### 9.4 프로덕션에서의 디노이징

필름 렌더러(Arnold, Manuka)는 64~256 spp로 렌더링하고 ML 디노이징을 적용하여 수 시간의 컴퓨팅 시간을 절약한다. 실시간 레이 트레이싱(1~4 spp)은 시간적 누적(temporal accumulation)과 ML 디노이징에 더욱 의존한다.

---

## 10. Python 구현: 몬테 카를로 패스 트레이서

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

# --- Utilities ---

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def random_cosine_hemisphere(normal):
    """
    Sample a direction on the hemisphere proportional to cos(theta).
    Uses Malley's method: sample uniform disk, then project to hemisphere.
    This ensures more samples near the normal (where cos(theta) is large),
    reducing variance compared to uniform hemisphere sampling.
    """
    # Build local coordinate frame around the normal
    if abs(normal[0]) > 0.9:
        tangent = normalize(np.cross(np.array([0, 1, 0]), normal))
    else:
        tangent = normalize(np.cross(np.array([1, 0, 0]), normal))
    bitangent = np.cross(normal, tangent)

    # Random point on unit disk
    xi1, xi2 = np.random.random(), np.random.random()
    phi = 2.0 * np.pi * xi1
    r = np.sqrt(xi2)  # Why sqrt: ensures uniform distribution on disk
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    z = np.sqrt(max(0.0, 1.0 - xi2))  # Project up to hemisphere

    # Transform to world space
    return x * tangent + y * bitangent + z * normal


# --- Scene Objects ---

@dataclass
class PTMaterial:
    albedo: np.ndarray          # Diffuse color
    emission: np.ndarray = None  # Emissive color (for lights)

    def __post_init__(self):
        if self.emission is None:
            self.emission = np.zeros(3)


@dataclass
class PTSphere:
    center: np.ndarray
    radius: float
    material: PTMaterial

    def intersect(self, origin, direction):
        oc = origin - self.center
        a = np.dot(direction, direction)
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return float('inf'), None
        sq = np.sqrt(disc)
        t = (-b - sq) / (2 * a)
        if t < 1e-4:
            t = (-b + sq) / (2 * a)
        if t < 1e-4:
            return float('inf'), None
        hit = origin + t * direction
        n = (hit - self.center) / self.radius
        return t, n


@dataclass
class PTPlane:
    point: np.ndarray
    normal: np.ndarray
    material: PTMaterial

    def intersect(self, origin, direction):
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-8:
            return float('inf'), None
        t = np.dot(self.point - origin, self.normal) / denom
        if t < 1e-4:
            return float('inf'), None
        return t, self.normal.copy()


# --- Path Tracer ---

class PathTracer:
    """
    Monte Carlo path tracer with:
    - Cosine-weighted importance sampling
    - Russian roulette for unbiased termination
    - Next Event Estimation (direct light sampling)
    """

    def __init__(self, objects, max_depth=10):
        self.objects = objects
        self.max_depth = max_depth
        # Identify emissive objects for direct light sampling
        self.lights = [obj for obj in objects
                       if np.any(obj.material.emission > 0)]

    def find_nearest(self, origin, direction):
        nearest_t = float('inf')
        nearest_obj = None
        nearest_normal = None
        for obj in self.objects:
            t, n = obj.intersect(origin, direction)
            if t < nearest_t:
                nearest_t = t
                nearest_obj = obj
                nearest_normal = n
        return nearest_t, nearest_obj, nearest_normal

    def sample_light(self, hit_point, normal):
        """
        Next Event Estimation: sample a point on a light source and
        compute the direct illumination contribution.
        """
        if not self.lights:
            return np.zeros(3)

        # Randomly pick one light (uniform selection)
        light = self.lights[np.random.randint(len(self.lights))]

        # Sample a random point on the light sphere
        # Why random direction: approximates area light sampling
        rand_dir = normalize(np.random.randn(3))
        light_point = light.center + light.radius * rand_dir

        # Direction from hit point to light sample
        to_light = light_point - hit_point
        dist2 = np.dot(to_light, to_light)
        dist = np.sqrt(dist2)
        light_dir = to_light / dist

        # Check if light is above the surface
        cos_theta = np.dot(normal, light_dir)
        if cos_theta <= 0:
            return np.zeros(3)

        # Cosine at the light surface
        light_normal = normalize(light_point - light.center)
        cos_light = abs(np.dot(light_normal, -light_dir))

        # Shadow test
        shadow_t, shadow_obj, _ = self.find_nearest(
            hit_point + 1e-4 * normal, light_dir
        )
        if shadow_obj is not light or shadow_t > dist + 0.01:
            if shadow_t < dist - 0.01:
                return np.zeros(3)  # Occluded

        # Light area (sphere surface = 4*pi*r^2)
        light_area = 4.0 * np.pi * light.radius ** 2

        # Contribution: Le * BRDF * cos_theta * cos_light * area / dist^2
        # BRDF for Lambertian = albedo / pi
        # Multiplied by number of lights for unbiased estimator
        brdf = 1.0 / np.pi
        contribution = (light.material.emission * brdf * cos_theta
                       * cos_light * light_area / dist2)

        return contribution * len(self.lights)

    def trace(self, origin, direction, depth=0):
        """
        Trace a single path through the scene.
        Returns the estimated radiance along this path.
        """
        t, obj, normal = self.find_nearest(origin, direction)
        if obj is None:
            return np.zeros(3)  # No hit -> black background

        hit_point = origin + t * direction

        # Ensure normal faces the incoming ray
        if np.dot(normal, direction) > 0:
            normal = -normal

        # Collect emission (only on first bounce to avoid double-counting with NEE)
        if depth == 0:
            radiance = obj.material.emission.copy()
        else:
            radiance = np.zeros(3)

        # Russian roulette termination
        albedo = obj.material.albedo
        # Why max component: ensures bright channels survive with high probability
        survival_prob = min(max(albedo[0], albedo[1], albedo[2]), 0.95)
        if depth > 2:
            if np.random.random() > survival_prob:
                return radiance
        else:
            survival_prob = 1.0

        # Direct illumination via Next Event Estimation
        direct = self.sample_light(hit_point, normal)
        radiance += albedo * direct

        # Indirect illumination: sample a bounce direction
        bounce_dir = random_cosine_hemisphere(normal)

        # Why we don't include cos/pi here: cosine-weighted sampling cancels them
        # For Lambertian: (albedo/pi) * Li * cos(theta) / (cos(theta)/pi) = albedo * Li
        bounce_radiance = self.trace(
            hit_point + 1e-4 * normal, bounce_dir, depth + 1
        )

        # Add indirect contribution, compensated by Russian roulette survival
        radiance += albedo * bounce_radiance / survival_prob

        return radiance

    def render(self, width, height, spp=32, fov_deg=60.0,
               eye=np.array([0.0, 1.0, 4.0]),
               target=np.array([0.0, 0.5, 0.0])):
        """Render the scene with multiple samples per pixel."""
        image = np.zeros((height, width, 3))
        fov = np.radians(fov_deg)
        aspect = width / height

        forward = normalize(target - eye)
        right = normalize(np.cross(forward, np.array([0, 1, 0])))
        up = np.cross(right, forward)
        half_h = np.tan(fov / 2)
        half_w = half_h * aspect

        for j in range(height):
            for i in range(width):
                pixel_color = np.zeros(3)

                for s in range(spp):
                    # Jittered sampling: random offset within the pixel
                    u = (2 * (i + np.random.random()) / width - 1) * half_w
                    v = (1 - 2 * (j + np.random.random()) / height) * half_h
                    direction = normalize(forward + u * right + v * up)

                    pixel_color += self.trace(eye, direction)

                image[j, i] = pixel_color / spp

            if (j + 1) % max(1, height // 10) == 0:
                print(f"  Row {j+1}/{height} ({100*(j+1)//height}%)")

        return image


# --- Build Cornell-Box-Like Scene ---

white   = PTMaterial(albedo=np.array([0.73, 0.73, 0.73]))
red     = PTMaterial(albedo=np.array([0.65, 0.05, 0.05]))
green   = PTMaterial(albedo=np.array([0.12, 0.45, 0.15]))
light_m = PTMaterial(albedo=np.array([0.0, 0.0, 0.0]),
                     emission=np.array([15.0, 15.0, 15.0]))

objects = [
    # Floor
    PTPlane(np.array([0, 0, 0]), np.array([0, 1, 0]), white),
    # Ceiling
    PTPlane(np.array([0, 3, 0]), np.array([0, -1, 0]), white),
    # Back wall
    PTPlane(np.array([0, 0, -2]), np.array([0, 0, 1]), white),
    # Left wall (red)
    PTPlane(np.array([-2, 0, 0]), np.array([1, 0, 0]), red),
    # Right wall (green)
    PTPlane(np.array([2, 0, 0]), np.array([-1, 0, 0]), green),
    # Spheres
    PTSphere(np.array([-0.7, 0.5, -0.5]), 0.5, white),
    PTSphere(np.array([0.7, 0.8, 0.2]), 0.8, white),
    # Light source (small sphere on ceiling)
    PTSphere(np.array([0, 2.8, -0.5]), 0.3, light_m),
]

# --- Render ---
pt = PathTracer(objects, max_depth=8)
print("Path tracing 200x150 @ 64 spp...")
image = pt.render(200, 150, spp=64,
                  eye=np.array([0.0, 1.5, 5.0]),
                  target=np.array([0.0, 1.0, 0.0]))

# Tone mapping: simple Reinhard + gamma correction
# Why tone mapping: path tracer output is HDR; must compress for display
image = image / (1.0 + image)          # Reinhard tone mapping
image = np.power(np.clip(image, 0, 1), 1.0 / 2.2)  # Gamma correction

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7.5))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Path Tracer (64 spp, cosine sampling + NEE + Russian roulette)')
    plt.tight_layout()
    plt.savefig('path_trace_result.png', dpi=150)
    plt.close()
    print("Saved path_trace_result.png")
except ImportError:
    print("Install matplotlib to save the image")
```

---

## 11. 고급 주제 (간략 개요)

### 11.1 양방향 패스 트레이싱(Bidirectional Path Tracing, BDPT)

카메라와 광원 양쪽에서 경로를 추적한 후 연결한다. 빛이 복잡한 경로를 따르는 씬(예: 작은 창문으로 채광되는 실내 씬)을 효율적으로 처리한다.

### 11.2 메트로폴리스 광 전달(Metropolis Light Transport, MLT)

마르코프 체인 몬테 카를로(Markov Chain Monte Carlo, MCMC)를 사용하여 경로 공간을 탐색한다. 어려운 광 경로(코스틱, 열쇠구멍을 통과하는 빛)를 찾고 활용하는 데 탁월하다.

### 11.3 광자 매핑(Photon Mapping)

1패스: 광원에서 광자(photon)를 추적하여 공간 해시에 저장. 2패스: 카메라에 보이는 교차점에서 주변 광자 수집. 코스틱과 체적 효과(volumetric effects)에 적합.

### 11.4 스펙트럼 렌더링(Spectral Rendering)

RGB 대신 개별 파장으로 빛을 시뮬레이션한다. 정확한 분산(dispersion, 프리즘을 통한 무지개), 형광(fluorescence), 편광(polarization) 효과에 필요하다.

---

## 요약

| 개념 | 핵심 아이디어 |
|------|-------------|
| 렌더링 방정식 | $L_o = L_e + \int f_r L_i \cos\theta\,d\omega$ — 광 전달의 완전한 설명 |
| 몬테 카를로 | 무작위 샘플링으로 적분 추정; 오차 $\propto 1/\sqrt{N}$ |
| 패스 트레이싱 | 무작위 보행: 씬을 통해 광선을 튀김; 각 경로가 렌더링 방정식 샘플링 |
| 중요도 샘플링 | 피적분 함수가 큰 곳에서 샘플링; 확산에는 코사인 가중, 광택에는 BRDF |
| 러시안 룰렛 | 편향 없는 경로 종료; $1/(1-q)$로 보상 |
| 다음 이벤트 추정 | 각 바운스에서 광원 명시적 샘플링; 노이즈를 극적으로 감소 |
| MIS | 여러 전략을 최적으로 결합; 균형 휴리스틱 |
| 디노이징 | 노이즈 있는 저샘플 이미지 필터링; ML 디노이저는 보조 버퍼 사용 |

## 연습문제

1. **몬테 카를로 파이 추정**: 단위 정사각형에 점을 무작위로 던져 내접 원 안에 떨어지는 수를 세어 $\pi$를 추정한다. 샘플 수 대비 수렴을 플롯하고 $1/\sqrt{N}$ 비율을 검증한다.

2. **중요도 샘플링 비교**: (a) 균일 샘플링과 (b) $p(x) = 11x^{10}$인 중요도 샘플링을 사용하여 $\int_0^1 x^{10}\,dx$를 추정한다. 1000개 샘플 후 분산을 비교한다.

3. **노이즈 vs. spp**: 코넬 박스 씬을 1, 4, 16, 64, 256 spp로 렌더링한다. 4096 spp 레퍼런스 대비 픽셀별 RMSE를 측정하고 $1/\sqrt{N}$ 수렴을 검증한다.

4. **NEE 소거 분석**: 64 spp에서 다음 이벤트 추정 유무로 씬을 렌더링한다. 노이즈 수준을 비교한다. 광원이 작을수록 NEE가 더 도움이 되는 이유를 설명한다.

5. **러시안 룰렛**: 러시안 룰렛 없이 (고정 최대 깊이 = 5) 버전을 구현하고 러시안 룰렛 버전과 밝기를 비교한다. 어느 것이 더 정확한가?

6. **간단한 디노이저**: 양방향 필터를 후처리로 구현한다. 16 spp 렌더에 적용하고 원시 64 spp 출력과 비교한다. 공간 및 범위 시그마 파라미터를 실험한다.

## 더 읽을거리

- Kajiya, J. "The Rendering Equation." *SIGGRAPH*, 1986. (모든 것의 시작이 된 기초 논문)
- Veach, E. "Robust Monte Carlo Methods for Light Transport Simulation." PhD thesis, Stanford, 1997. (MIS, BDPT, MLT를 도입한 렌더링 분야에서 가장 영향력 있는 박사 논문)
- Pharr, M., Jakob, W., Humphreys, G. *Physically Based Rendering*, 4th ed. MIT Press, 2023. (패스 트레이싱 구현의 레퍼런스)
- Schied, C. et al. "Spatiotemporal Variance-Guided Filtering: Real-Time Reconstruction for Path-Traced Global Illumination." *HPG*, 2017. (SVGF 디노이저)
- Kulla, C. and Conty, A. "Revisiting Physically Based Shading at Imageworks." *SIGGRAPH Course*, 2017. (Sony Pictures에서의 프로덕션 패스 트레이싱)
