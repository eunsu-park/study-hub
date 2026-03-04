# 다변량 정규분포

**이전**: [확률변수의 변환](./08_Transformations_of_Random_Variables.md) | **다음**: [수렴 개념](./10_Convergence_Concepts.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 다변량 정규분포 $\mathbf{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$의 밀도 작성하기
2. 공분산 행렬 $\boldsymbol{\Sigma}$의 역할과 성질 설명하기
3. 결합 다변량 정규분포로부터 주변 분포와 조건부 분포 유도하기
4. 다변량 정규 벡터에 선형 변환 적용하기
5. 마할라노비스 거리(Mahalanobis distance)를 계산하고 해석하기
6. 다변량 정규분포와 카이제곱 분포의 연결 설명하기
7. 이변량 정규 경우와 등고선 기하학 기술하기
8. 촐레스키 분해(Cholesky decomposition)를 통해 다변량 정규 표본 생성하기

---

다변량 정규분포(Multivariate Normal, MVN)는 다변량 통계학의 기초이며, 선형 회귀, 판별 분석, 주성분 분석 등 수많은 방법의 근간입니다. 익숙한 종 모양 곡선을 $p$ 차원으로 확장하며, 공분산 행렬이 변수 간의 전체 종속 구조를 포착합니다.

---

## 1. 정의와 밀도

### 1.1 밀도 공식

확률 벡터 $\mathbf{X} = (X_1, X_2, \ldots, X_p)^T$가 **$p$-변량 정규분포** $\mathbf{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$를 따르면, 결합 PDF는:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{p/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

여기서:

- $\boldsymbol{\mu} \in \mathbb{R}^p$는 **평균 벡터**
- $\boldsymbol{\Sigma} \in \mathbb{R}^{p \times p}$는 **공분산 행렬** (대칭, 양정치)
- $|\boldsymbol{\Sigma}|$는 $\boldsymbol{\Sigma}$의 행렬식

### 1.2 MGF를 통한 동치 특성화

$$M_{\mathbf{X}}(\mathbf{t}) = E[e^{\mathbf{t}^T \mathbf{X}}] = \exp\!\left(\mathbf{t}^T \boldsymbol{\mu} + \frac{1}{2}\mathbf{t}^T \boldsymbol{\Sigma}\, \mathbf{t}\right)$$

이 MGF는 MVN 분포를 유일하게 결정합니다.

### 1.3 대안적 정의

$\mathbf{X}$가 다변량 정규인 것은, 모든 $\mathbf{a} \in \mathbb{R}^p$에 대해 모든 선형 결합 $\mathbf{a}^T \mathbf{X} = a_1 X_1 + \cdots + a_p X_p$가 (일변량) 정규인 것과 동치입니다.

---

## 2. 공분산 행렬 $\boldsymbol{\Sigma}$

### 2.1 구조

$$\boldsymbol{\Sigma} = \begin{pmatrix} \sigma_1^2 & \sigma_{12} & \cdots & \sigma_{1p} \\ \sigma_{21} & \sigma_2^2 & \cdots & \sigma_{2p} \\ \vdots & \vdots & \ddots & \vdots \\ \sigma_{p1} & \sigma_{p2} & \cdots & \sigma_p^2 \end{pmatrix}$$

여기서 $\sigma_{ij} = \text{Cov}(X_i, X_j)$이고 $\sigma_{ii} = \text{Var}(X_i)$입니다.

### 2.2 양반정치 성질

$\boldsymbol{\Sigma}$는 **양반정치**(Positive Semi-Definite, PSD)이어야 합니다: 모든 $\mathbf{a} \in \mathbb{R}^p$에 대해,

$$\mathbf{a}^T \boldsymbol{\Sigma}\, \mathbf{a} = \text{Var}(\mathbf{a}^T \mathbf{X}) \ge 0$$

$\boldsymbol{\Sigma}$가 **양정치**(positive definite, 모든 고유값이 순양수)이면, 밀도가 존재하고 분포가 비퇴화입니다.

### 2.3 고유분해

$\boldsymbol{\Sigma}$가 대칭 PSD이므로, 스펙트럼 분해(spectral decomposition)를 갖습니다:

$$\boldsymbol{\Sigma} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^T$$

여기서 $\mathbf{Q}$는 직교 행렬 (열이 고유벡터)이고 $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_p)$ ($\lambda_i \ge 0$)입니다.

고유벡터는 분포의 **주축**(principal axes)을 정의하고, 고유값은 각 축을 따른 분산을 나타냅니다.

### 2.4 상관 행렬

상관 행렬(correlation matrix) $\mathbf{R}$은 표준화된 공분산입니다:

$$R_{ij} = \frac{\sigma_{ij}}{\sigma_i \sigma_j}, \quad \mathbf{R} = \mathbf{D}^{-1} \boldsymbol{\Sigma}\, \mathbf{D}^{-1}$$

여기서 $\mathbf{D} = \text{diag}(\sigma_1, \ldots, \sigma_p)$입니다.

---

## 3. 주변 분포

### 3.1 주변분포는 정규

$\mathbf{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$이면, $\mathbf{X}$의 임의의 부분 벡터도 다변량 정규입니다.

$\mathbf{X}$, $\boldsymbol{\mu}$, $\boldsymbol{\Sigma}$를 두 그룹으로 분할합니다:

$$\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}, \quad \boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$$

그러면:

$$\mathbf{X}_1 \sim N(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_{11}), \qquad \mathbf{X}_2 \sim N(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_{22})$$

### 3.2 중요한 주의 사항

역은 성립하지 **않습니다**: 개별적으로 정규인 주변분포를 갖는다고 결합 다변량 정규가 보장되지 않습니다. MVN은 모든 선형 결합이 정규일 것을 요구합니다.

---

## 4. 조건부 분포

### 4.1 조건부 정규 공식

$\mathbf{X}_2 = \mathbf{x}_2$가 주어졌을 때 $\mathbf{X}_1$의 조건부 분포는:

$$\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2 \sim N\!\left(\boldsymbol{\mu}_{1|2},\, \boldsymbol{\Sigma}_{1|2}\right)$$

여기서:

$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

$$\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$$

### 4.2 핵심 관찰

- 조건부 평균은 조건 변수 $\mathbf{x}_2$의 **선형 함수**입니다.
- 조건부 공분산 $\boldsymbol{\Sigma}_{1|2}$는 $\mathbf{x}_2$에 **의존하지 않습니다**; 어떤 값을 관측하든 항상 동일합니다.
- 행렬 $\boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}$는 **회귀 계수** 행렬의 역할을 합니다.

### 4.3 스칼라 예제 (이변량 경우)

$p = 2$이고 상관 $\rho$인 경우:

$$X_1 \mid X_2 = x_2 \sim N\!\left(\mu_1 + \rho\frac{\sigma_1}{\sigma_2}(x_2 - \mu_2),\; \sigma_1^2(1 - \rho^2)\right)$$

조건부 분산은 $(1 - \rho^2)$ 인수만큼 감소합니다: 상관이 강할수록 $X_2$가 $X_1$에 대해 더 많은 정보를 제공합니다.

---

## 5. 선형 변환

### 5.1 주요 결과

$\mathbf{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$이고 $\mathbf{A}$가 $q \times p$ 행렬, $\mathbf{b} \in \mathbb{R}^q$이면:

$$\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b} \sim N_q(\mathbf{A}\boldsymbol{\mu} + \mathbf{b},\; \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

### 5.2 귀결

- **표준화**: $\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{X} - \boldsymbol{\mu}) \sim N_p(\mathbf{0}, \mathbf{I}_p)$
- **비상관화**: 고유분해 $\boldsymbol{\Sigma} = \mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^T$를 사용하여, $\mathbf{Y} = \mathbf{Q}^T(\mathbf{X} - \boldsymbol{\mu})$로 정의하면, $\mathbf{Y} \sim N_p(\mathbf{0}, \boldsymbol{\Lambda})$이 되어 $\mathbf{Y}$의 성분이 독립입니다.
- **사영**: 임의의 $\mathbf{a}^T \mathbf{X}$는 평균 $\mathbf{a}^T\boldsymbol{\mu}$, 분산 $\mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a}$인 일변량 정규입니다.

---

## 6. 마할라노비스 거리

### 6.1 정의

점 $\mathbf{x}$에서 분포 $N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$까지의 **마할라노비스 거리**(Mahalanobis distance)는:

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})}$$

### 6.2 해석

- 일변량 $z$-점수 $|x - \mu|/\sigma$를 다차원으로 일반화합니다.
- 변수 간의 상관과 각 축에 따른 서로 다른 분산을 고려합니다.
- MVN 밀도의 같은 등고선 위의 점들은 같은 마할라노비스 거리를 갖습니다.
- $\boldsymbol{\Sigma} = \mathbf{I}$이면, 마할라노비스 거리는 $\boldsymbol{\mu}$로부터의 유클리드 거리로 환원됩니다.

### 6.3 활용 사례

- 다변량 데이터에서의 이상값 탐지
- 분류 (예: 이차 판별 분석)
- 다변량 평균에 대한 가설 검정

---

## 7. 카이제곱과의 연결

### 7.1 마할라노비스 거리의 제곱

$\mathbf{X} \sim N_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$이면, 마할라노비스 거리의 제곱:

$$D_M^2 = (\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \sim \chi^2(p)$$

**증명 스케치**: $\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2}(\mathbf{X} - \boldsymbol{\mu}) \sim N_p(\mathbf{0}, \mathbf{I})$로 놓으면, $D_M^2 = \mathbf{Z}^T\mathbf{Z} = \sum_{i=1}^p Z_i^2$로, $p$개의 독립 표준 정규의 제곱합이므로 $\chi^2(p)$입니다.

### 7.2 응용: 신뢰 타원체

집합 $\{\mathbf{x} : (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu}) \le c^2\}$는 $P(\chi^2(p) \le c^2)$와 같은 확률 질량을 포함하는 타원체입니다. $p = 2$의 경우, $c^2 = 5.991$이 95% 신뢰 타원을 제공합니다.

---

## 8. 이변량 정규: 기하학과 등고선

### 8.1 이변량 경우

$p = 2$이고 매개변수 $\boldsymbol{\mu} = (\mu_1, \mu_2)^T$, 분산 $\sigma_1^2, \sigma_2^2$, 상관 $\rho$인 경우:

$$f(x_1, x_2) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-\rho^2}} \exp\!\left(-\frac{Q}{2(1-\rho^2)}\right)$$

여기서:

$$Q = \frac{(x_1-\mu_1)^2}{\sigma_1^2} - \frac{2\rho(x_1-\mu_1)(x_2-\mu_2)}{\sigma_1\sigma_2} + \frac{(x_2-\mu_2)^2}{\sigma_2^2}$$

### 8.2 등고선 기하학

등밀도 등고선은 $\boldsymbol{\mu}$를 중심으로 한 **타원**입니다:

- $\rho = 0$: 좌표축에 정렬된 타원
- $\rho > 0$: 양의 대각선 방향으로 기울어진 타원
- $\rho < 0$: 음의 대각선 방향으로 기울어진 타원
- $\sigma_1 = \sigma_2$이고 $\rho = 0$: 등고선이 원

타원의 주축은 $\boldsymbol{\Sigma}$의 고유벡터에 의해 결정되고, 그 길이는 고유값의 제곱근에 비례합니다.

### 8.3 독립성 vs. 비상관

MVN에서 (그리고 **오직** MVN에서만), 비상관은 독립을 함의합니다:

$$\rho = 0 \iff X_1 \perp X_2 \quad \text{(결합 정규 변수에 대해)}$$

이것은 임의의 결합 분포에서는 성립하지 않는 특수한 성질입니다.

---

## 9. 주성분 분석과의 연결

### 9.1 간략한 개요

**주성분 분석**(Principal Component Analysis, PCA)은 $\boldsymbol{\Sigma}$의 고유분해와 직접 연결됩니다.

- $k$번째 주성분은 $Y_k = \mathbf{q}_k^T(\mathbf{X} - \boldsymbol{\mu})$이며, $\mathbf{q}_k$는 $\boldsymbol{\Sigma}$의 $k$번째 고유벡터입니다.
- $\text{Var}(Y_k) = \lambda_k$ ($k$번째 고유값).
- MVN 하에서, 주성분 $Y_1, Y_2, \ldots, Y_p$는 상호 독립입니다.
- PCA는 데이터에서 최대 분산의 방향을 찾으며, 이는 MVN 밀도 등고선의 주축에 해당합니다.

MVN과 PCA 사이의 이 연결이 다변량 분석에서 정규성 가정이 빈번하게 등장하는 이유입니다.

---

## 10. Python 예제

### 10.1 촐레스키 분해를 통한 이변량 정규 표본 생성

촐레스키 분해(Cholesky decomposition)는 양정치 행렬을 $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$로 인수분해합니다. 여기서 $\mathbf{L}$은 하삼각 행렬입니다. $\mathbf{X} \sim N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$를 표집하려면:

1. $\mathbf{Z} \sim N(\mathbf{0}, \mathbf{I})$를 생성
2. $\mathbf{X} = \boldsymbol{\mu} + \mathbf{L}\mathbf{Z}$를 계산

```python
import random
import math

def cholesky_2x2(sigma):
    """Cholesky decomposition for a 2x2 positive definite matrix.
    sigma = [[a, b], [b, d]] -> L such that L @ L^T = sigma.
    """
    a, b = sigma[0][0], sigma[0][1]
    d = sigma[1][1]
    l11 = math.sqrt(a)
    l21 = b / l11
    l22 = math.sqrt(d - l21 ** 2)
    return [[l11, 0.0], [l21, l22]]

def sample_bivariate_normal(mu, sigma, n, seed=42):
    """Generate n samples from N(mu, sigma) using Cholesky."""
    random.seed(seed)
    L = cholesky_2x2(sigma)
    samples = []
    for _ in range(n):
        z1 = random.gauss(0, 1)
        z2 = random.gauss(0, 1)
        x1 = mu[0] + L[0][0] * z1 + L[0][1] * z2
        x2 = mu[1] + L[1][0] * z1 + L[1][1] * z2
        samples.append((x1, x2))
    return samples

# Parameters
mu = [2.0, 5.0]
sigma = [[4.0, 3.0],    # Var(X1)=4, Cov=3
         [3.0, 9.0]]    # Var(X2)=9
rho_true = 3.0 / (2.0 * 3.0)  # 0.5

samples = sample_bivariate_normal(mu, sigma, n=50_000)

# Verify sample statistics
x1 = [s[0] for s in samples]
x2 = [s[1] for s in samples]

mean1 = sum(x1) / len(x1)
mean2 = sum(x2) / len(x2)
var1 = sum((v - mean1) ** 2 for v in x1) / (len(x1) - 1)
var2 = sum((v - mean2) ** 2 for v in x2) / (len(x2) - 1)
cov12 = sum((x1[i] - mean1) * (x2[i] - mean2)
            for i in range(len(x1))) / (len(x1) - 1)

print("Bivariate Normal via Cholesky:")
print(f"  E[X1] = {mean1:.4f}  (true: {mu[0]})")
print(f"  E[X2] = {mean2:.4f}  (true: {mu[1]})")
print(f"  Var(X1) = {var1:.4f}  (true: {sigma[0][0]})")
print(f"  Var(X2) = {var2:.4f}  (true: {sigma[1][1]})")
print(f"  Cov(X1,X2) = {cov12:.4f}  (true: {sigma[0][1]})")
print(f"  Corr = {cov12 / math.sqrt(var1 * var2):.4f}  (true: {rho_true:.4f})")
```

### 10.2 조건부 분포 검증

```python
import random
import math

random.seed(100)
mu = [0.0, 0.0]
sigma = [[1.0, 0.6], [0.6, 1.0]]  # rho = 0.6
n = 200_000

samples = sample_bivariate_normal(mu, sigma, n, seed=100)

# Condition on X2 being near 1.5
x2_target = 1.5
tol = 0.05
conditional_x1 = [s[0] for s in samples
                   if abs(s[1] - x2_target) < tol]

# Theoretical: X1 | X2=1.5 ~ N(rho * 1.5, 1 - rho^2)
rho = 0.6
cond_mean_theory = rho * x2_target
cond_var_theory = 1 - rho ** 2

cond_mean_sample = sum(conditional_x1) / len(conditional_x1)
cond_var_sample = (sum((x - cond_mean_sample) ** 2 for x in conditional_x1)
                   / (len(conditional_x1) - 1))

print(f"\nConditional X1 | X2 ~ {x2_target} (n = {len(conditional_x1)}):")
print(f"  Sample mean:     {cond_mean_sample:.4f}  (theory: {cond_mean_theory:.4f})")
print(f"  Sample variance: {cond_var_sample:.4f}  (theory: {cond_var_theory:.4f})")
```

### 10.3 마할라노비스 거리와 카이제곱 확인

```python
import random
import math

random.seed(200)
mu = [1.0, 3.0]
sigma = [[2.0, 1.0], [1.0, 4.0]]
n = 100_000

samples = sample_bivariate_normal(mu, sigma, n, seed=200)

# Compute inverse of 2x2 sigma
det_s = sigma[0][0] * sigma[1][1] - sigma[0][1] * sigma[1][0]
inv_s = [[sigma[1][1] / det_s, -sigma[0][1] / det_s],
         [-sigma[1][0] / det_s, sigma[0][0] / det_s]]

# Compute squared Mahalanobis distances
d2_samples = []
for x1, x2 in samples:
    dx = [x1 - mu[0], x2 - mu[1]]
    d2 = (dx[0] * (inv_s[0][0] * dx[0] + inv_s[0][1] * dx[1])
          + dx[1] * (inv_s[1][0] * dx[0] + inv_s[1][1] * dx[1]))
    d2_samples.append(d2)

# Should follow chi-squared(2): mean=2, variance=4
d2_mean = sum(d2_samples) / n
d2_var = sum((d - d2_mean) ** 2 for d in d2_samples) / (n - 1)

print(f"\nSquared Mahalanobis distance (should be Chi-sq(2)):")
print(f"  Mean: {d2_mean:.4f}  (theoretical: 2)")
print(f"  Var:  {d2_var:.4f}  (theoretical: 4)")

# Fraction within 95% confidence ellipse
# chi2(2) at 95% is 5.991
within_95 = sum(1 for d in d2_samples if d <= 5.991) / n
print(f"  Within 95% ellipse: {within_95:.4f}  (theoretical: 0.95)")
```

### 10.4 선형 변환 검증

```python
import random
import math

random.seed(300)
mu = [1.0, 2.0]
sigma = [[4.0, 1.0], [1.0, 2.0]]
n = 100_000

samples = sample_bivariate_normal(mu, sigma, n, seed=300)

# Apply A = [[2, 1], [0, 3]], b = [1, -1]
# Y = AX + b ~ N(A*mu + b, A*Sigma*A^T)
A = [[2, 1], [0, 3]]
b = [1.0, -1.0]

y_samples = []
for x1, x2 in samples:
    y1 = A[0][0] * x1 + A[0][1] * x2 + b[0]
    y2 = A[1][0] * x1 + A[1][1] * x2 + b[1]
    y_samples.append((y1, y2))

# Theoretical: E[Y] = A*mu + b
ey1 = A[0][0] * mu[0] + A[0][1] * mu[1] + b[0]
ey2 = A[1][0] * mu[0] + A[1][1] * mu[1] + b[1]

# A*Sigma*A^T
# First: A*Sigma
AS = [[A[0][0]*sigma[0][0]+A[0][1]*sigma[1][0],
       A[0][0]*sigma[0][1]+A[0][1]*sigma[1][1]],
      [A[1][0]*sigma[0][0]+A[1][1]*sigma[1][0],
       A[1][0]*sigma[0][1]+A[1][1]*sigma[1][1]]]
# Then: (A*Sigma)*A^T
ASAT = [[AS[0][0]*A[0][0]+AS[0][1]*A[0][1],
         AS[0][0]*A[1][0]+AS[0][1]*A[1][1]],
        [AS[1][0]*A[0][0]+AS[1][1]*A[0][1],
         AS[1][0]*A[1][0]+AS[1][1]*A[1][1]]]

y1_vals = [y[0] for y in y_samples]
y2_vals = [y[1] for y in y_samples]
my1 = sum(y1_vals) / n
my2 = sum(y2_vals) / n

print(f"\nLinear transformation Y = AX + b:")
print(f"  E[Y1] = {my1:.4f}  (theory: {ey1:.4f})")
print(f"  E[Y2] = {my2:.4f}  (theory: {ey2:.4f})")
print(f"  Var(Y1) = {sum((v-my1)**2 for v in y1_vals)/(n-1):.4f}  "
      f"(theory: {ASAT[0][0]:.4f})")
print(f"  Var(Y2) = {sum((v-my2)**2 for v in y2_vals)/(n-1):.4f}  "
      f"(theory: {ASAT[1][1]:.4f})")
```

---

## 핵심 요약

1. **다변량 정규분포**는 평균 벡터 $\boldsymbol{\mu}$와 공분산 행렬 $\boldsymbol{\Sigma}$로 완전히 명세됩니다. 지수부의 이차 형식은 마할라노비스 거리의 제곱입니다.
2. 임의의 부분 벡터의 **주변 분포**도 다변량 정규입니다. **조건부 분포**는 조건 변수에 대해 평균이 선형인 정규분포입니다.
3. **선형 변환**은 정규성을 보존합니다: $\mathbf{X}$가 MVN이면 $\mathbf{A}\mathbf{X} + \mathbf{b}$도 MVN입니다.
4. **마할라노비스 거리**는 $z$-점수를 다차원으로 일반화하며, $\chi^2(p)$와 연결됩니다.
5. 결합 정규 변수에서 **비상관은 독립을 함의합니다** -- 이것은 정규 분포족에 고유한 성질입니다.
6. **촐레스키 분해**는 독립 표준 정규로부터 MVN 표본을 효율적으로 생성하는 알고리즘을 제공합니다.
7. $\boldsymbol{\Sigma}$의 고유분해는 주축을 드러내며, PCA와 연결됩니다.

---

*다음 레슨: [수렴 개념](./10_Convergence_Concepts.md)*
