# 확률변수의 변환

**이전**: [연속 분포족](./07_Continuous_Distribution_Families.md) | **다음**: [다변량 정규분포](./09_Multivariate_Normal_Distribution.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. CDF 기법을 사용하여 $Y = g(X)$의 분포 유도하기
2. 단조 변환에 대한 변수 변환 공식(야코비안 방법) 적용하기
3. 함수 대응을 통해 이산 확률변수 변환하기
4. 두 확률변수의 합, 곱, 비의 분포 구하기
5. 독립 확률변수의 합에 대한 합성곱(convolution) 공식 사용하기
6. MGF 기법을 적용하여 합의 분포 식별하기
7. 순서 통계량의 PDF 유도하기
8. Box-Muller 변환 설명 및 구현하기

---

확률변수 $X$의 분포가 알려져 있을 때, 변환된 변수 $Y = g(X)$의 분포가 자주 필요합니다. 이 레슨에서는 단일 변수 변환부터 다변수 확률변수의 함수와 순서 통계량(order statistics)까지, 이러한 분포를 구하는 핵심 기법을 발전시킵니다.

---

## 1. 단일 확률변수의 함수

### 1.1 문제

$X$가 알려진 PDF $f_X(x)$를 가지고 $Y = g(X)$로 정의합니다. $f_Y(y)$는 무엇인가?

모든 함수 $g$에 대해 작동하는 단일 공식은 없습니다. 접근 방식은 $g$가 단조인지, 구간별 단조인지, 더 복잡한지에 따라 달라집니다. 두 가지 일반적 기법을 소개합니다: CDF 방법과 PDF(야코비안) 방법.

---

## 2. CDF 기법

### 2.1 일반 절차

CDF 기법은 **임의의** 가측 함수 $g$에 대해 작동합니다. 아이디어는 직관적입니다:

1. $F_Y(y) = P(Y \le y) = P(g(X) \le y)$를 씁니다.
2. 사건 $\{g(X) \le y\}$를 $X$에 관해 표현합니다 (즉, 집합 $A_y = \{x : g(x) \le y\}$를 구합니다).
3. $F_Y(y) = P(X \in A_y) = \int_{A_y} f_X(x)\,dx$를 계산합니다.
4. 미분합니다: $f_Y(y) = F_Y'(y)$.

### 2.2 예제: $Y = X^2$ ($X \sim N(0, 1)$)

$y > 0$에 대해:

$$F_Y(y) = P(X^2 \le y) = P(-\sqrt{y} \le X \le \sqrt{y}) = \Phi(\sqrt{y}) - \Phi(-\sqrt{y}) = 2\Phi(\sqrt{y}) - 1$$

미분하면:

$$f_Y(y) = 2\phi(\sqrt{y}) \cdot \frac{1}{2\sqrt{y}} = \frac{1}{\sqrt{y}} \phi(\sqrt{y}) = \frac{1}{\sqrt{2\pi y}}\, e^{-y/2}$$

이것은 $\chi^2(1)$의 PDF로서, 표준 정규의 제곱이 자유도 1인 카이제곱임을 확인합니다.

### 2.3 예제: $Y = -\ln(X)$ ($X \sim \text{Uniform}(0, 1)$)

$y \ge 0$에 대해:

$$F_Y(y) = P(-\ln X \le y) = P(X \ge e^{-y}) = 1 - e^{-y}$$

이것은 $\text{Exp}(1)$의 CDF입니다. 이것이 지수 확률변수를 생성하기 위한 역변환 방법입니다.

---

## 3. 야코비안을 이용한 PDF 기법 (단조 변환)

### 3.1 일대일 (단조) 경우

$g$가 순단조이고 미분 가능하며 역함수가 $x = g^{-1}(y)$이면:

$$f_Y(y) = f_X\!\left(g^{-1}(y)\right) \cdot \left|\frac{dx}{dy}\right|$$

인수 $\left|\frac{dx}{dy}\right| = \left|\frac{d}{dy}g^{-1}(y)\right|$는 역변환의 **야코비안**(Jacobian)입니다. 절댓값은 $g$가 증가이든 감소이든 밀도가 비음이 되도록 보장합니다.

### 3.2 유도

순증가 $g$에 대한 CDF 기법에서 시작합니다:

$$F_Y(y) = P(g(X) \le y) = P(X \le g^{-1}(y)) = F_X(g^{-1}(y))$$

연쇄 법칙으로 미분하면:

$$f_Y(y) = f_X(g^{-1}(y)) \cdot \frac{d}{dy}g^{-1}(y)$$

순감소 $g$의 경우, 부등호가 역전되어 음의 부호가 나오지만 절댓값이 이를 흡수합니다.

### 3.3 예제: 로그정규분포

$X \sim N(\mu, \sigma^2)$이고 $Y = e^X$이면, $X = \ln Y$이고 $dx/dy = 1/y$. $y > 0$에 대해:

$$f_Y(y) = \frac{1}{y\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(\ln y - \mu)^2}{2\sigma^2}\right)$$

이것이 **로그정규**(log-normal) 분포입니다.

### 3.4 구간별 단조 함수

$g$가 전역적으로 단조가 아니지만 단조인 구간들로 분할될 수 있으면, 각 가지(branch)에 대해 합산합니다:

$$f_Y(y) = \sum_{i} f_X(x_i) \cdot \left|\frac{dx_i}{dy}\right|$$

여기서 $x_1, x_2, \ldots$는 $g(x) = y$의 모든 해입니다.

---

## 4. 이산 변수의 변환

이산 확률변수 $X$가 PMF $p_X(x)$를 갖고 함수 $Y = g(X)$에 대해:

$$p_Y(y) = P(Y = y) = \sum_{x:\, g(x) = y} p_X(x)$$

같은 $y$에 대응되는 $X$의 모든 값을 모아 그 확률을 합산하면 됩니다.

**예제**: $X \sim \text{Binomial}(n, p)$이고 $Y = n - X$ (실패 횟수)이면, $Y \sim \text{Binomial}(n, 1-p)$.

---

## 5. 두 확률변수의 함수

### 5.1 합: $Z = X + Y$

결합 PDF $f_{X,Y}(x, y)$가 주어지면, 변환 $(X, Y) \to (Z, W)$를 $Z = X + Y$, $W = Y$ (보조)로 정의합니다. 야코비안은:

$$\frac{\partial(x, y)}{\partial(z, w)} = \begin{vmatrix} 1 & -1 \\ 0 & 1 \end{vmatrix} = 1$$

따라서 $f_{Z,W}(z, w) = f_{X,Y}(z - w,\, w)$이고, $w$에 대해 주변화하면:

$$f_Z(z) = \int_{-\infty}^{\infty} f_{X,Y}(z - w,\, w)\, dw$$

### 5.2 곱: $Z = XY$

$Z = XY$, $W = Y$로 치환합니다. 그러면 $X = Z/W$이고 $|J| = 1/|w|$:

$$f_Z(z) = \int_{-\infty}^{\infty} f_{X,Y}\!\left(\frac{z}{w},\, w\right) \frac{1}{|w|}\, dw$$

### 5.3 비: $Z = X/Y$

$Z = X/Y$, $W = Y$로 치환합니다. 그러면 $X = ZW$이고 $|J| = |w|$:

$$f_Z(z) = \int_{-\infty}^{\infty} f_{X,Y}(zw,\, w)\, |w|\, dw$$

이 기법은 정규분포와 카이제곱으로부터 $t$-분포와 $F$-분포를 유도하는 데 사용됩니다.

---

## 6. 독립 확률변수의 합에 대한 합성곱

### 6.1 합성곱 공식

$X$와 $Y$가 **독립**이면, $f_{X,Y}(x, y) = f_X(x) f_Y(y)$이고, $Z = X + Y$의 PDF는 **합성곱**(convolution)으로 단순화됩니다:

$$f_Z(z) = (f_X * f_Y)(z) = \int_{-\infty}^{\infty} f_X(z - y)\, f_Y(y)\, dy$$

### 6.2 예제: 두 독립 지수분포의 합

$X \sim \text{Exp}(\lambda)$이고 $Y \sim \text{Exp}(\lambda)$가 독립이라 합니다. $z > 0$에 대해:

$$f_Z(z) = \int_0^z \lambda e^{-\lambda(z-y)} \cdot \lambda e^{-\lambda y}\, dy = \lambda^2 e^{-\lambda z} \int_0^z dy = \lambda^2 z\, e^{-\lambda z}$$

이것은 $\text{Gamma}(2, \lambda)$ (즉, $\text{Erlang}(2, \lambda)$)로, 감마분포족의 가법 성질을 확인합니다.

### 6.3 이산 합성곱

독립 이산 확률변수에 대해:

$$p_Z(z) = \sum_k p_X(k)\, p_Y(z - k)$$

---

## 7. 합에 대한 MGF 기법

### 7.1 핵심 성질

$X$와 $Y$가 독립이면, $Z = X + Y$의 MGF는:

$$M_Z(t) = E[e^{tZ}] = E[e^{t(X+Y)}] = E[e^{tX}] \cdot E[e^{tY}] = M_X(t) \cdot M_Y(t)$$

MGF가 분포를 유일하게 결정하므로 (0의 근방에서 존재할 때), MGF의 곱을 인식하여 $f_Z$를 식별할 수 있습니다.

### 7.2 예제: 독립 정규분포의 합

$X \sim N(\mu_1, \sigma_1^2)$이고 $Y \sim N(\mu_2, \sigma_2^2)$가 독립이면:

$$M_Z(t) = \exp\!\left(\mu_1 t + \frac{\sigma_1^2 t^2}{2}\right) \cdot \exp\!\left(\mu_2 t + \frac{\sigma_2^2 t^2}{2}\right) = \exp\!\left((\mu_1+\mu_2)t + \frac{(\sigma_1^2+\sigma_2^2)t^2}{2}\right)$$

이것은 $N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$의 MGF로, 정규분포족이 합에 대해 닫혀있음을 확인합니다.

### 7.3 MGF와 합성곱 중 언제 어떤 것을 선호하는가

- **MGF 기법**: 결과 MGF가 인식 가능한 형태일 때 최적. 빠르고 우아함.
- **합성곱**: MGF가 존재하지 않거나 결과 MGF를 쉽게 인식할 수 없을 때 필요.

---

## 8. 순서 통계량

### 8.1 설정

$X_1, X_2, \ldots, X_n$이 CDF $F(x)$와 PDF $f(x)$를 갖는 i.i.d.일 때, **순서 통계량**(order statistics)은:

$$X_{(1)} \le X_{(2)} \le \cdots \le X_{(n)}$$

여기서 $X_{(k)}$는 $k$번째로 작은 값입니다.

### 8.2 최솟값 $X_{(1)}$의 분포

$$P(X_{(1)} > x) = P(\text{모든 } X_i > x) = [1 - F(x)]^n$$

$$f_{X_{(1)}}(x) = n[1 - F(x)]^{n-1} f(x)$$

### 8.3 최댓값 $X_{(n)}$의 분포

$$F_{X_{(n)}}(x) = P(\text{모든 } X_i \le x) = [F(x)]^n$$

$$f_{X_{(n)}}(x) = n[F(x)]^{n-1} f(x)$$

### 8.4 일반적인 $k$번째 순서 통계량

$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} [F(x)]^{k-1} [1-F(x)]^{n-k} f(x)$$

**해석**: $n$개의 값 중 정확히 $k-1$개가 $x$ 아래, 하나가 $x$에, $n-k$개가 $x$ 위에 있습니다. 조합적 인수는 배열의 수를 셉니다.

### 8.5 특수한 경우: 균등분포의 순서 통계량

$X_i \sim \text{Uniform}(0,1)$이면, $X_{(k)} \sim \text{Beta}(k, n - k + 1)$. 이 우아한 연결은 비모수 통계학(nonparametric statistics)의 기초입니다.

---

## 9. Box-Muller 변환

### 9.1 동기

균등 확률변수로부터 정규 확률변수를 어떻게 생성할 수 있을까요? Box-Muller 변환은 정확한 방법을 제공합니다.

### 9.2 변환

$U_1, U_2 \sim \text{Uniform}(0,1)$이 독립이라 합니다. 다음을 정의합니다:

$$Z_1 = \sqrt{-2\ln U_1}\, \cos(2\pi U_2)$$
$$Z_2 = \sqrt{-2\ln U_1}\, \sin(2\pi U_2)$$

그러면 $Z_1$과 $Z_2$는 **독립인** $N(0,1)$ 확률변수입니다.

### 9.3 왜 작동하는가

이 변환은 2차원에서의 변수 변환 기법으로 검증할 수 있습니다. 극좌표 $(R, \Theta)$에서 $R = \sqrt{-2\ln U_1}$이고 $\Theta = 2\pi U_2$이면:

- $R^2 = -2\ln U_1 \sim \text{Exp}(1/2)$이며 이는 $\chi^2(2)$와 같습니다
- $\Theta \sim \text{Uniform}(0, 2\pi)$
- $R$과 $\Theta$는 독립

$(Z_1, Z_2)$의 결합 밀도는 두 독립 표준 정규 밀도의 곱으로 인수분해됩니다.

---

## 10. Python 예제

### 10.1 CDF 기법: 표준 정규의 $Y = X^2$ 검증

```python
import random
import math

random.seed(42)
n = 200_000
z_samples = [random.gauss(0, 1) for _ in range(n)]
y_samples = [z ** 2 for z in z_samples]

# Chi-squared(1) has mean=1 and variance=2
mean_y = sum(y_samples) / n
var_y = sum((y - mean_y) ** 2 for y in y_samples) / (n - 1)

print("Y = Z^2 where Z ~ N(0,1):")
print(f"  Sample mean:     {mean_y:.4f}  (theoretical: 1.0)")
print(f"  Sample variance: {var_y:.4f}  (theoretical: 2.0)")
```

### 10.2 역변환: 균등분포로부터 지수분포 생성

```python
import random
import math

random.seed(7)
lam = 2.5
n = 100_000

# Inverse-transform: X = -ln(U) / lambda
uniform_samples = [random.random() for _ in range(n)]
exp_samples = [-math.log(u) / lam for u in uniform_samples]

mean_est = sum(exp_samples) / n
var_est = sum((x - mean_est) ** 2 for x in exp_samples) / (n - 1)

print(f"Exp({lam}) via inverse transform:")
print(f"  Sample mean:     {mean_est:.4f}  (theoretical: {1/lam:.4f})")
print(f"  Sample variance: {var_est:.4f}  (theoretical: {1/lam**2:.4f})")
```

### 10.3 합성곱: 두 독립 균등분포의 합

```python
import random

random.seed(314)
n = 100_000

# Z = U1 + U2, each Uniform(0,1)
# The result is a triangular distribution on [0, 2]
z_samples = [random.random() + random.random() for _ in range(n)]

mean_z = sum(z_samples) / n
var_z = sum((z - mean_z) ** 2 for z in z_samples) / (n - 1)

print("Z = U1 + U2 (triangular distribution):")
print(f"  Sample mean:     {mean_z:.4f}  (theoretical: 1.0)")
print(f"  Sample variance: {var_z:.4f}  (theoretical: {1/6:.4f})")

# Histogram approximation using text
bins = [0] * 20
for z in z_samples:
    idx = min(int(z * 10), 19)
    bins[idx] += 1

print("\nApproximate shape (triangular):")
for i, count in enumerate(bins):
    bar = '#' * (count * 80 // max(bins))
    print(f"  [{i*0.1:4.1f}-{(i+1)*0.1:4.1f}): {bar}")
```

### 10.4 Box-Muller 변환 구현

```python
import random
import math

random.seed(2024)
n = 100_000

z1_samples = []
z2_samples = []

for _ in range(n // 2):
    u1 = random.random()
    u2 = random.random()
    r = math.sqrt(-2.0 * math.log(u1))
    theta = 2.0 * math.pi * u2
    z1_samples.append(r * math.cos(theta))
    z2_samples.append(r * math.sin(theta))

all_z = z1_samples + z2_samples

mean_z = sum(all_z) / len(all_z)
var_z = sum((z - mean_z) ** 2 for z in all_z) / (len(all_z) - 1)

print("Box-Muller generated N(0,1) samples:")
print(f"  Sample mean:     {mean_z:.4f}  (theoretical: 0.0)")
print(f"  Sample variance: {var_z:.4f}  (theoretical: 1.0)")

# Verify independence: correlation between z1 and z2
n_pairs = len(z1_samples)
mean1 = sum(z1_samples) / n_pairs
mean2 = sum(z2_samples) / n_pairs
cov = sum((z1_samples[i] - mean1) * (z2_samples[i] - mean2)
          for i in range(n_pairs)) / (n_pairs - 1)
std1 = math.sqrt(sum((z - mean1) ** 2 for z in z1_samples) / (n_pairs - 1))
std2 = math.sqrt(sum((z - mean2) ** 2 for z in z2_samples) / (n_pairs - 1))
corr = cov / (std1 * std2)

print(f"  Correlation(Z1, Z2): {corr:.4f}  (theoretical: 0.0)")
```

### 10.5 순서 통계량: 균등 표본의 최솟값과 최댓값

```python
import random

random.seed(99)
n_sim = 50_000
sample_size = 10

min_samples = []
max_samples = []

for _ in range(n_sim):
    data = [random.random() for _ in range(sample_size)]
    min_samples.append(min(data))
    max_samples.append(max(data))

# X_(1) ~ Beta(1, n) => mean = 1/(n+1)
# X_(n) ~ Beta(n, 1) => mean = n/(n+1)
min_mean = sum(min_samples) / n_sim
max_mean = sum(max_samples) / n_sim

k = sample_size
print(f"Order statistics for Uniform(0,1), sample size = {k}:")
print(f"  E[X_(1)]:  {min_mean:.4f}  (theoretical: {1/(k+1):.4f})")
print(f"  E[X_({k})]: {max_mean:.4f}  (theoretical: {k/(k+1):.4f})")
```

---

## 핵심 요약

1. **CDF 기법**은 가장 일반적인 접근법입니다: $F_Y(y) = P(g(X) \le y)$를 계산하고 미분합니다. 임의의 변환에 대해 작동합니다.
2. **단조** 변환에 대해, 야코비안 공식 $f_Y(y) = f_X(g^{-1}(y)) \cdot |dx/dy|$가 직접적인 지름길을 제공합니다.
3. **합성곱** 공식은 독립 확률변수의 합의 PDF를 제공합니다. **MGF 기법**은 MGF를 인식할 수 있을 때 더 우아한 경로를 제공하는 경우가 많습니다.
4. **순서 통계량**은 명시적 밀도 공식을 갖습니다. 균등분포의 경우, 순서 통계량은 베타분포를 따릅니다.
5. **Box-Muller 변환**은 두 독립 균등분포를 두 독립 정규분포로 변환하며, 역CDF 아이디어와 극좌표 분해를 결합합니다.
6. 올바른 기법의 선택은 문제에 따라 달라집니다: 일반적 함수에는 CDF, 단조 사상에는 야코비안, 알려진 생성함수를 갖는 합에는 MGF를 사용합니다.

---

*다음 레슨: [다변량 정규분포](./09_Multivariate_Normal_Distribution.md)*
