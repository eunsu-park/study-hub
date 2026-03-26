# 01. 무한급수와 수렴 (Infinite Series and Convergence)

## 학습 목표

- **급수의 수렴과 발산**을 정의하고, 부분합(partial sum)을 통해 수렴을 판별할 수 있다
- **비교 판정법, 비율 판정법, 근 판정법, 적분 판정법, 교대급수 판정법** 등 주요 수렴 판정법을 적용할 수 있다
- **멱급수 (power series)** 의 수렴 반경을 구하고, 항별 미분/적분을 수행할 수 있다
- **테일러 급수와 매클로린 급수**를 이용하여 함수를 근사하고, 물리학 문제에 적용할 수 있다
- **점근 급수 (asymptotic series)** 와 **스털링 근사**의 개념을 이해하고 물리학적 응용에 활용할 수 있다

---

## 1. 급수의 기본 개념

### 1.1 수열과 급수

**수열 (sequence)** 은 자연수에서 실수(또는 복소수)로의 함수입니다:

$$a_1, a_2, a_3, \ldots, a_n, \ldots$$

수열 $\{a_n\}$이 특정 값 $L$에 수렴한다는 것은:

$$\lim_{n \to \infty} a_n = L$$

즉, $n$이 충분히 클 때 $a_n$이 $L$에 임의로 가까워진다는 의미입니다.

**급수 (series)** 는 수열의 항들의 합입니다:

$$S = \sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots$$

급수가 수렴하기 위한 **필요조건**: $\lim_{n \to \infty} a_n = 0$

> **주의**: 이것은 필요조건일 뿐 충분조건이 아닙니다. 조화급수 $\sum(1/n)$은 $a_n \to 0$이지만 발산합니다.

### 1.2 부분합과 수렴

**부분합 (partial sum)** $S_N$은 급수의 처음 $N$개 항의 합입니다:

$$S_N = \sum_{n=1}^{N} a_n = a_1 + a_2 + \cdots + a_N$$

급수의 **수렴 (convergence)**: 부분합의 수열 $\{S_N\}$이 유한한 값 $S$에 수렴하면, 급수 $\sum a_n = S$로 수렴한다고 합니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 예제: 기하급수 (geometric series)의 부분합
# sum_{n=0}^{inf} r^n = 1/(1-r)  (|r| < 1)

def geometric_partial_sums(r, N_max):
    """기하급수의 부분합을 계산합니다."""
    n_values = np.arange(N_max + 1)
    terms = r ** n_values
    partial_sums = np.cumsum(terms)
    return n_values, partial_sums

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 수렴하는 경우: |r| < 1
r_conv = 0.5
n_vals, S_n = geometric_partial_sums(r_conv, 20)
exact = 1 / (1 - r_conv)
axes[0].plot(n_vals, S_n, 'bo-', markersize=4, label=f'S_N (r={r_conv})')
axes[0].axhline(y=exact, color='r', linestyle='--', label=f'S = {exact:.4f}')
axes[0].set_xlabel('N')
axes[0].set_ylabel('S_N')
axes[0].set_title(f'수렴하는 기하급수 (r = {r_conv})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 조화급수 vs p-급수 비교
N_max = 100
n = np.arange(1, N_max + 1)
harmonic = np.cumsum(1.0 / n)           # p = 1 (발산)
p2_series = np.cumsum(1.0 / n**2)       # p = 2 (수렴)
p3_series = np.cumsum(1.0 / n**3)       # p = 3 (수렴)

axes[1].plot(n, harmonic, 'r-', label='p=1 (조화급수, 발산)')
axes[1].plot(n, p2_series, 'b-', label=f'p=2 (수렴, pi^2/6 = {np.pi**2/6:.4f})')
axes[1].plot(n, p3_series, 'g-', label='p=3 (수렴)')
axes[1].axhline(y=np.pi**2/6, color='b', linestyle='--', alpha=0.5)
axes[1].set_xlabel('N')
axes[1].set_ylabel('S_N')
axes[1].set_title('p-급수 비교: sum(1/n^p)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('series_convergence.png', dpi=150, bbox_inches='tight')
plt.show()
```

**핵심 급수 정리**:

| 급수 | 수렴 조건 | 합 (수렴 시) |
|------|-----------|-------------|
| 기하급수 $\sum r^n$ | $|r| < 1$ | $1/(1-r)$ |
| p-급수 $\sum 1/n^p$ | $p > 1$ | $\zeta(p)$ |
| 조화급수 $\sum 1/n$ | 발산 | - |

---

## 2. 수렴 판정법

### 2.1 비교 판정법 (Comparison Test)

$0 \leq a_n \leq b_n$ (충분히 큰 $n$에 대해) 이면:

- $\sum b_n$ 수렴 $\Rightarrow$ $\sum a_n$ 수렴
- $\sum a_n$ 발산 $\Rightarrow$ $\sum b_n$ 발산

**극한 비교 판정법 (Limit Comparison Test)**: $a_n > 0$, $b_n > 0$이고

$$\lim_{n \to \infty} \frac{a_n}{b_n} = L \quad (0 < L < \infty)$$

이면 $\sum a_n$과 $\sum b_n$은 동시에 수렴하거나 동시에 발산합니다.

### 2.2 비율 판정법 (Ratio Test)

$$\rho = \lim_{n \to \infty} \left|\frac{a_{n+1}}{a_n}\right|$$

- $\rho < 1$ 이면 **절대 수렴 (absolutely convergent)**
- $\rho > 1$ 이면 **발산**
- $\rho = 1$ 이면 **판정 불능** (다른 판정법 사용)

> **물리학 팁**: 비율 판정법은 팩토리얼이나 지수 함수가 포함된 급수에 특히 유용합니다.

### 2.3 근 판정법 (Root Test)

$$\rho = \lim_{n \to \infty} |a_n|^{1/n}$$

- $\rho < 1$ 이면 절대 수렴
- $\rho > 1$ 이면 발산
- $\rho = 1$ 이면 판정 불능

### 2.4 적분 판정법 (Integral Test)

$f(x)$가 $[1, \infty)$에서 양수, 연속, 감소 함수이고 $f(n) = a_n$이면:

$$\sum_{n=1}^{\infty} a_n \text{ 과 } \int_1^{\infty} f(x) \, dx \text{ 는 동시에 수렴 또는 발산}$$

**예제**: p-급수의 수렴 조건 유도

$$\int_1^{\infty} \frac{1}{x^p} \, dx = \left[\frac{x^{1-p}}{1-p}\right]_1^{\infty}$$

$p > 1$이면 적분이 수렴하므로 $\sum(1/n^p)$도 수렴합니다.

### 2.5 교대급수 판정법 (Alternating Series Test)

교대급수 $\sum (-1)^n b_n$ ($b_n > 0$)이 다음 조건을 만족하면 수렴합니다:

1. $b_{n+1} \leq b_n$ (단조감소)
2. $\lim_{n \to \infty} b_n = 0$

교대급수의 오차 한계: $|S - S_N| \leq b_{N+1}$ (다음 항의 절대값 이하)

```python
import numpy as np
import sympy as sp

# 수렴 판정법 자동 적용 도구
def test_convergence(a_n_func, name="급수"):
    """주어진 급수에 대해 비율 판정법과 근 판정법을 적용합니다."""
    n = sp.Symbol('n', positive=True, integer=True)
    a_n = a_n_func(n)

    print(f"=== {name}: sum a_n, a_n = {a_n} ===\n")

    # 비율 판정법
    ratio = sp.simplify(a_n_func(n + 1) / a_n)
    rho_ratio = sp.limit(sp.Abs(ratio), n, sp.oo)
    print(f"비율 판정법: |a_(n+1)/a_n| -> {rho_ratio}")
    if rho_ratio < 1:
        print("  => 절대 수렴\n")
    elif rho_ratio > 1:
        print("  => 발산\n")
    else:
        print("  => 판정 불능\n")

    # 근 판정법
    root = sp.Abs(a_n) ** (sp.Rational(1, n))
    rho_root = sp.limit(root, n, sp.oo)
    print(f"근 판정법: |a_n|^(1/n) -> {rho_root}")
    if rho_root < 1:
        print("  => 절대 수렴\n")
    elif rho_root > 1:
        print("  => 발산\n")
    else:
        print("  => 판정 불능\n")

# 테스트 예제들
n = sp.Symbol('n', positive=True, integer=True)

# 1) sum n! / n^n  (수렴)
test_convergence(lambda n: sp.factorial(n) / n**n, "n!/n^n")

# 2) sum 1/n^2  (수렴)
test_convergence(lambda n: 1 / n**2, "1/n^2")

# 3) sum n^2 / 2^n  (수렴)
test_convergence(lambda n: n**2 / 2**n, "n^2/2^n")
```

```python
import numpy as np
import matplotlib.pyplot as plt

# 교대급수 시각화: ln(2) = sum_{n=1}^{inf} (-1)^{n+1} / n
N = 30
n = np.arange(1, N + 1)
terms = (-1.0) ** (n + 1) / n
partial_sums = np.cumsum(terms)

plt.figure(figsize=(10, 5))
plt.plot(n, partial_sums, 'bo-', markersize=5, label='부분합 S_N')
plt.axhline(y=np.log(2), color='r', linestyle='--',
            label=f'ln(2) = {np.log(2):.6f}')

# 오차 범위 표시
for i in range(len(n)):
    error_bound = 1.0 / (n[i] + 1)
    plt.plot([n[i], n[i]],
             [partial_sums[i] - error_bound, partial_sums[i] + error_bound],
             'g-', alpha=0.3, linewidth=2)

plt.xlabel('N')
plt.ylabel('S_N')
plt.title('교대급수: ln(2) = 1 - 1/2 + 1/3 - 1/4 + ...')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alternating_series.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 3. 멱급수 (Power Series)

### 3.1 수렴 반경

**멱급수 (power series)** 는 다음과 같은 형태의 급수입니다:

$$f(x) = \sum_{n=0}^{\infty} c_n (x - a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots$$

여기서 $a$는 **전개 중심 (center)**, $c_n$은 **계수 (coefficients)** 입니다.

**수렴 반경 (radius of convergence)** $R$:

$$R = \lim_{n \to \infty} \left|\frac{c_n}{c_{n+1}}\right| \quad \text{또는} \quad R = \frac{1}{\lim_{n \to \infty} |c_n|^{1/n}}$$

- $|x - a| < R$ 이면 절대 수렴
- $|x - a| > R$ 이면 발산
- $|x - a| = R$ 이면 별도 조사 필요

수렴 구간 (interval of convergence):

```
      발산        수렴         발산
  -------|========|========|--------->  x
        a-R       a       a+R
         <---- R ---->
```

### 3.2 항별 미분과 적분

수렴 구간 내에서, 멱급수는 **항별 미분 (term-by-term differentiation)** 과 **항별 적분 (term-by-term integration)** 이 가능합니다:

$$f'(x) = \sum_{n=1}^{\infty} n \cdot c_n \cdot (x-a)^{n-1}$$

$$\int f(x) \, dx = C + \sum_{n=0}^{\infty} \frac{c_n \cdot (x-a)^{n+1}}{n+1}$$

미분과 적분 후에도 **수렴 반경은 동일**합니다 (경계점 수렴 여부는 달라질 수 있음).

```python
import sympy as sp

x = sp.Symbol('x')

# 멱급수의 수렴 반경 계산 예제
print("=== 멱급수의 수렴 반경 ===\n")

# 1) sum x^n / n!  (지수함수)
n = sp.Symbol('n', positive=True, integer=True)
c_n = 1 / sp.factorial(n)
c_n1 = 1 / sp.factorial(n + 1)
R1 = sp.limit(sp.Abs(c_n / c_n1), n, sp.oo)
print(f"sum x^n/n!: R = {R1}  (모든 x에서 수렴)")

# 2) sum n * x^n  (R = 1)
c_n = n
c_n1 = n + 1
R2 = sp.limit(sp.Abs(c_n / c_n1), n, sp.oo)
print(f"sum n*x^n: R = {R2}")

# 3) sum n! * x^n  (R = 0, x=0에서만 수렴)
c_n = sp.factorial(n)
c_n1 = sp.factorial(n + 1)
R3 = sp.limit(sp.Abs(c_n / c_n1), n, sp.oo)
print(f"sum n!*x^n: R = {R3}  (x=0에서만 수렴)")

# 항별 미분 시각화
print("\n=== 항별 미분: d/dx [1/(1-x)] = 1/(1-x)^2 ===")
# 1/(1-x) = sum x^n  =>  미분하면  1/(1-x)^2 = sum n*x^{n-1}
x_vals = np.linspace(-0.9, 0.9, 200)

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 원래 함수와 급수
for N in [3, 5, 10, 20]:
    series_sum = sum(x_vals**k for k in range(N))
    axes[0].plot(x_vals, series_sum, alpha=0.7, label=f'N={N}')

axes[0].plot(x_vals, 1/(1-x_vals), 'k--', linewidth=2, label='1/(1-x)')
axes[0].set_ylim(-5, 15)
axes[0].set_title('1/(1-x) = sum x^n')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 미분한 함수와 급수
for N in [3, 5, 10, 20]:
    deriv_sum = sum(k * x_vals**(k-1) for k in range(1, N))
    axes[1].plot(x_vals, deriv_sum, alpha=0.7, label=f'N={N}')

axes[1].plot(x_vals, 1/(1-x_vals)**2, 'k--', linewidth=2, label='1/(1-x)^2')
axes[1].set_ylim(-5, 30)
axes[1].set_title('1/(1-x)^2 = sum n*x^{n-1}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('power_series_differentiation.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 4. 테일러 급수와 매클로린 급수

### 4.1 테일러 전개

함수 $f(x)$가 $x = a$ 근방에서 무한 번 미분가능하면, **테일러 급수 (Taylor series)** 로 전개할 수 있습니다:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!} (x - a)^n$$

$$= f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots$$

$a = 0$인 경우를 **매클로린 급수 (Maclaurin series)** 라고 합니다.

**테일러 나머지 (remainder)** $R_N$: $N$차까지 전개한 후의 오차

$$R_N(x) = \frac{f^{(N+1)}(c)}{(N+1)!} (x-a)^{N+1} \quad \text{(라그랑주 나머지, } a < c < x\text{)}$$

### 4.2 주요 함수의 급수 표현

물리학에서 가장 빈번하게 사용되는 급수 전개:

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots \quad \text{(모든 } x\text{)}$$

$$\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots \quad \text{(모든 } x\text{)}$$

$$\cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots \quad \text{(모든 } x\text{)}$$

$$\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots \quad (-1 < x \leq 1)$$

$$(1+x)^p = 1 + px + \frac{p(p-1)}{2!} x^2 + \cdots \quad (|x| < 1, \text{ 이항급수})$$

$$\frac{1}{1-x} = 1 + x + x^2 + x^3 + \cdots \quad (|x| < 1)$$

$$\arctan(x) = x - \frac{x^3}{3} + \frac{x^5}{5} - \frac{x^7}{7} + \cdots \quad (|x| \leq 1)$$

```python
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# SymPy를 이용한 테일러 전개
x = sp.Symbol('x')

functions = {
    'e^x': sp.exp(x),
    'sin(x)': sp.sin(x),
    'cos(x)': sp.cos(x),
    'ln(1+x)': sp.ln(1 + x),
}

print("=== 주요 함수의 테일러 급수 (a=0, 6차까지) ===\n")
for name, func in functions.items():
    taylor = sp.series(func, x, 0, n=7)
    print(f"{name} = {taylor}\n")

# 테일러 급수의 수렴 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 500)

# sin(x)의 테일러 근사
ax = axes[0, 0]
ax.plot(x_vals, np.sin(x_vals), 'k-', linewidth=2, label='sin(x)')
for N in [1, 3, 5, 7, 9]:
    approx = sum((-1)**k * x_vals**(2*k+1) / np.math.factorial(2*k+1)
                 for k in range(N))
    ax.plot(x_vals, approx, alpha=0.7, label=f'{2*N-1}차')
ax.set_ylim(-2, 2)
ax.set_title('sin(x)의 테일러 근사')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# cos(x)의 테일러 근사
ax = axes[0, 1]
ax.plot(x_vals, np.cos(x_vals), 'k-', linewidth=2, label='cos(x)')
for N in [1, 2, 4, 6, 8]:
    approx = sum((-1)**k * x_vals**(2*k) / np.math.factorial(2*k)
                 for k in range(N))
    ax.plot(x_vals, approx, alpha=0.7, label=f'{2*N-2}차')
ax.set_ylim(-2, 2)
ax.set_title('cos(x)의 테일러 근사')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# e^x의 테일러 근사
x_exp = np.linspace(-3, 3, 300)
ax = axes[1, 0]
ax.plot(x_exp, np.exp(x_exp), 'k-', linewidth=2, label='e^x')
for N in [1, 2, 4, 6, 10]:
    approx = sum(x_exp**k / np.math.factorial(k) for k in range(N + 1))
    ax.plot(x_exp, approx, alpha=0.7, label=f'{N}차')
ax.set_ylim(-5, 20)
ax.set_title('e^x의 테일러 근사')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ln(1+x)의 테일러 근사
x_ln = np.linspace(-0.95, 2, 300)
ax = axes[1, 1]
ax.plot(x_ln, np.log(1 + x_ln), 'k-', linewidth=2, label='ln(1+x)')
for N in [1, 3, 5, 10, 20]:
    x_series = np.linspace(-0.95, 0.95, 300)  # 수렴 구간 내
    approx = sum((-1)**(k+1) * x_series**k / k for k in range(1, N + 1))
    ax.plot(x_series, approx, alpha=0.7, label=f'{N}차')
ax.axvline(x=1, color='gray', linestyle=':', alpha=0.5, label='x=1 (경계)')
ax.axvline(x=-1, color='gray', linestyle=':', alpha=0.5, label='x=-1 (경계)')
ax.set_ylim(-4, 2)
ax.set_title('ln(1+x)의 테일러 근사')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('taylor_series_visualization.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.3 급수를 이용한 근사

물리학에서 급수 전개의 가장 중요한 응용은 **소량 근사 (small-quantity approximation)** 입니다.

$\epsilon \ll 1$ 일 때:

$$(1 + \epsilon)^p \approx 1 + p\epsilon + \frac{p(p-1)}{2} \epsilon^2 + \cdots$$

$$e^\epsilon \approx 1 + \epsilon + \frac{\epsilon^2}{2} + \cdots$$

$$\sin(\epsilon) \approx \epsilon - \frac{\epsilon^3}{6} + \cdots$$

$$\cos(\epsilon) \approx 1 - \frac{\epsilon^2}{2} + \cdots$$

$$\tan(\epsilon) \approx \epsilon + \frac{\epsilon^3}{3} + \cdots$$

$$\frac{1}{1+\epsilon} \approx 1 - \epsilon + \epsilon^2 - \cdots$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 소량 근사의 정확도 비교
epsilon = np.linspace(0, 1, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# sqrt(1 + epsilon) = (1+epsilon)^{1/2} 의 근사
exact = np.sqrt(1 + epsilon)
order1 = 1 + epsilon / 2                       # 1차 근사
order2 = 1 + epsilon / 2 - epsilon**2 / 8      # 2차 근사
order3 = 1 + epsilon/2 - epsilon**2/8 + epsilon**3/16  # 3차

axes[0].plot(epsilon, exact, 'k-', linewidth=2, label='정확값')
axes[0].plot(epsilon, order1, 'b--', label='1차: 1 + e/2')
axes[0].plot(epsilon, order2, 'r--', label='2차: 1 + e/2 - e^2/8')
axes[0].plot(epsilon, order3, 'g--', label='3차')
axes[0].set_xlabel('epsilon')
axes[0].set_ylabel('sqrt(1 + epsilon)')
axes[0].set_title('이항급수 근사: (1+e)^{1/2}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 상대 오차
rel_err1 = np.abs(order1 - exact) / exact * 100
rel_err2 = np.abs(order2 - exact) / exact * 100
rel_err3 = np.abs(order3 - exact) / exact * 100

axes[1].semilogy(epsilon, rel_err1, 'b-', label='1차 근사')
axes[1].semilogy(epsilon, rel_err2, 'r-', label='2차 근사')
axes[1].semilogy(epsilon, rel_err3, 'g-', label='3차 근사')
axes[1].set_xlabel('epsilon')
axes[1].set_ylabel('상대 오차 (%)')
axes[1].set_title('근사 차수에 따른 상대 오차')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('approximation_error.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 5. 점근 급수 (Asymptotic Series)

### 5.1 발산하지만 유용한 급수

**점근 급수 (asymptotic series)** 는 무한히 더해 나가면 발산하지만, 유한한 항까지의 부분합이 함수의 훌륭한 근사를 제공하는 급수입니다.

함수 $f(x)$에 대한 점근 전개:

$$f(x) \sim \sum_{n=0}^{\infty} \frac{a_n}{x^n} \quad (x \to \infty)$$

이것은 $N$개 항까지의 부분합에 대해 다음이 성립한다는 의미입니다:

$$\lim_{x \to \infty} x^N \left[f(x) - \sum_{n=0}^{N} \frac{a_n}{x^n}\right] = 0 \quad \text{(각 } N\text{에 대해)}$$

> **핵심**: 급수 전체는 발산하지만, **최적 항 수 (optimal truncation)** 에서 끊으면 매우 좋은 근사를 얻습니다.
> 일반적으로 항들의 크기가 감소하다가 다시 증가하기 시작하는 지점에서 끊는 것이 최적입니다.

### 5.2 스털링 근사

**스털링 근사 (Stirling's approximation)** 는 팩토리얼의 점근 근사입니다:

$$n! \sim \sqrt{2\pi n} \left(\frac{n}{e}\right)^n \quad (n \to \infty)$$

$$\ln(n!) \sim n\ln(n) - n + \frac{1}{2}\ln(2\pi n)$$

이 근사는 통계역학에서 볼츠만 엔트로피 계산, 조합론적 문제 등에 핵심적으로 사용됩니다.

더 정밀한 스털링 급수:

$$n! \sim \sqrt{2\pi n} \left(\frac{n}{e}\right)^n \left(1 + \frac{1}{12n} + \frac{1}{288n^2} - \frac{139}{51840n^3} + \cdots\right)$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, factorial

# 스털링 근사의 정확도
n_vals = np.arange(1, 51)

# 정확한 ln(n!)
exact_lnfact = np.array([np.sum(np.log(np.arange(1, n+1))) for n in n_vals])

# 스털링 근사 (여러 차수)
stirling_0 = n_vals * np.log(n_vals) - n_vals  # 가장 간단한 형태
stirling_1 = stirling_0 + 0.5 * np.log(2 * np.pi * n_vals)  # 1차 보정
stirling_2 = stirling_1 + 1 / (12 * n_vals)  # 2차 보정

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 절대값 비교
axes[0].plot(n_vals, exact_lnfact, 'k-', linewidth=2, label='ln(n!) 정확값')
axes[0].plot(n_vals, stirling_0, 'b--', label='n*ln(n) - n')
axes[0].plot(n_vals, stirling_1, 'r--', label='+ (1/2)ln(2*pi*n)')
axes[0].set_xlabel('n')
axes[0].set_ylabel('ln(n!)')
axes[0].set_title('스털링 근사')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 상대 오차
rel_err0 = np.abs(stirling_0 - exact_lnfact) / exact_lnfact * 100
rel_err1 = np.abs(stirling_1 - exact_lnfact) / exact_lnfact * 100
rel_err2 = np.abs(stirling_2 - exact_lnfact) / exact_lnfact * 100

axes[1].semilogy(n_vals, rel_err0, 'b-', label='0차: n*ln(n)-n')
axes[1].semilogy(n_vals, rel_err1, 'r-', label='1차: + ln(sqrt(2*pi*n))')
axes[1].semilogy(n_vals, rel_err2, 'g-', label='2차: + 1/(12n)')
axes[1].set_xlabel('n')
axes[1].set_ylabel('상대 오차 (%)')
axes[1].set_title('스털링 근사의 상대 오차')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stirling_approximation.png', dpi=150, bbox_inches='tight')
plt.show()

# 통계역학 응용: 볼츠만 엔트로피
print("=== 통계역학 응용: 이상기체 엔트로피 ===\n")
print("N개 입자가 W개 미시상태를 가질 때:")
print("S = k_B * ln(W)")
print("스털링 근사를 사용하면:")
print("ln(N!) ~ N*ln(N) - N")
print()
print("예: N = 10^23 (아보가드로 수 규모)")
N = 1e23
print(f"ln(N!) ~ N*ln(N) - N = {N * np.log(N) - N:.4e}")
print("직접 계산은 불가능하지만, 스털링 근사로 쉽게 계산 가능!")
```

---

## 6. 물리학에서의 응용

### 6.1 진자의 주기 (급수 전개)

단진자의 정확한 주기는 완전 타원 적분으로 주어집니다:

$$T = 4\sqrt{\frac{L}{g}} K\left(\sin\frac{\theta_0}{2}\right)$$

여기서 $K(k)$는 제1종 완전 타원 적분입니다. 이를 급수로 전개하면:

$$T = 2\pi\sqrt{\frac{L}{g}} \left[1 + \frac{1}{4}\sin^2\frac{\theta_0}{2} + \frac{9}{64}\sin^4\frac{\theta_0}{2} + \cdots\right]$$

$$\approx T_0 \left[1 + \frac{\theta_0^2}{16} + \frac{11\theta_0^4}{3072} + \cdots\right]$$

여기서 $T_0 = 2\pi\sqrt{L/g}$는 소각 근사에서의 주기입니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import ellipk

# 단진자의 주기: 정확값 vs 급수 근사
g = 9.81  # m/s^2
L = 1.0   # m
T0 = 2 * np.pi * np.sqrt(L / g)  # 소각 근사 주기

theta0 = np.linspace(0.01, np.pi * 0.95, 200)  # 초기 각도 (rad)
k = np.sin(theta0 / 2)

# 정확한 주기 (완전 타원 적분)
T_exact = 4 * np.sqrt(L / g) * ellipk(k**2)

# 급수 근사 (소각 근사부터 고차항까지)
T_approx0 = T0 * np.ones_like(theta0)                          # 0차 (소각)
T_approx2 = T0 * (1 + (1/4) * k**2)                           # 2차
T_approx4 = T0 * (1 + (1/4)*k**2 + (9/64)*k**4)               # 4차
T_approx6 = T0 * (1 + (1/4)*k**2 + (9/64)*k**4 + (25/256)*k**6)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 주기 비교
theta_deg = np.degrees(theta0)
axes[0].plot(theta_deg, T_exact / T0, 'k-', linewidth=2, label='정확값')
axes[0].plot(theta_deg, T_approx0 / T0, 'b--', label='0차 (소각근사)')
axes[0].plot(theta_deg, T_approx2 / T0, 'r--', label='2차 보정')
axes[0].plot(theta_deg, T_approx4 / T0, 'g--', label='4차 보정')
axes[0].plot(theta_deg, T_approx6 / T0, 'm--', label='6차 보정')
axes[0].set_xlabel('초기 각도 (도)')
axes[0].set_ylabel('T / T_0')
axes[0].set_title('단진자 주기: 급수 전개 근사')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 상대 오차
axes[1].semilogy(theta_deg, np.abs(T_approx0 - T_exact)/T_exact * 100,
                 'b-', label='0차 (소각)')
axes[1].semilogy(theta_deg, np.abs(T_approx2 - T_exact)/T_exact * 100,
                 'r-', label='2차')
axes[1].semilogy(theta_deg, np.abs(T_approx4 - T_exact)/T_exact * 100,
                 'g-', label='4차')
axes[1].semilogy(theta_deg, np.abs(T_approx6 - T_exact)/T_exact * 100,
                 'm-', label='6차')
axes[1].set_xlabel('초기 각도 (도)')
axes[1].set_ylabel('상대 오차 (%)')
axes[1].set_title('급수 근사의 상대 오차')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pendulum_period.png', dpi=150, bbox_inches='tight')
plt.show()

# 수치 결과
print("=== 단진자 주기 비교 (L=1m) ===\n")
print(f"{'각도':>6s}  {'정확값(s)':>10s}  {'소각근사(s)':>12s}  {'오차(%)':>8s}")
print("-" * 45)
for angle_deg in [5, 10, 30, 45, 60, 90, 120, 150]:
    angle_rad = np.radians(angle_deg)
    k_val = np.sin(angle_rad / 2)
    T_ex = 4 * np.sqrt(L / g) * ellipk(k_val**2)
    err = (T0 - T_ex) / T_ex * 100
    print(f"{angle_deg:>5d}   {T_ex:>10.6f}  {T0:>12.6f}  {err:>8.3f}")
```

### 6.2 상대론적 에너지의 비상대론적 근사

아인슈타인의 상대론적 에너지-운동량 관계:

$$E = \gamma mc^2 = \frac{mc^2}{\sqrt{1 - v^2/c^2}}$$

운동 에너지 $K = E - mc^2$를 급수 전개하면 ($\beta = v/c \ll 1$):

$$\gamma = \frac{1}{\sqrt{1 - \beta^2}} = 1 + \frac{1}{2}\beta^2 + \frac{3}{8}\beta^4 + \frac{5}{16}\beta^6 + \cdots$$

$$K = mc^2(\gamma - 1)$$

$$= \frac{1}{2}mv^2 + \frac{3}{8}\frac{mv^4}{c^2} + \frac{5}{16}\frac{mv^6}{c^4} + \cdots$$

여기서 첫 번째 항이 뉴턴 역학, 두 번째 항이 상대론적 보정(1차)입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 상대론적 운동 에너지의 급수 근사
beta = np.linspace(0.001, 0.99, 500)  # v/c

# 정확한 상대론적 운동 에너지: K/(mc^2) = gamma - 1
gamma = 1 / np.sqrt(1 - beta**2)
K_exact = gamma - 1  # mc^2 단위

# 급수 근사 (이항급수 전개)
K_order2 = 0.5 * beta**2                                         # 뉴턴 역학
K_order4 = 0.5 * beta**2 + (3/8) * beta**4                      # 1차 보정
K_order6 = 0.5 * beta**2 + (3/8) * beta**4 + (5/16) * beta**6  # 2차 보정

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 운동 에너지 비교
axes[0].plot(beta, K_exact, 'k-', linewidth=2, label='상대론적 (정확)')
axes[0].plot(beta, K_order2, 'b--', label='뉴턴: (1/2)mv^2')
axes[0].plot(beta, K_order4, 'r--', label='1차 보정')
axes[0].plot(beta, K_order6, 'g--', label='2차 보정')
axes[0].set_xlabel('beta = v/c')
axes[0].set_ylabel('K / (mc^2)')
axes[0].set_title('운동 에너지: 상대론 vs 뉴턴')
axes[0].set_ylim(0, 5)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 뉴턴 역학의 상대 오차
beta_low = np.linspace(0.001, 0.5, 300)
gamma_low = 1 / np.sqrt(1 - beta_low**2)
K_exact_low = gamma_low - 1
K_newton = 0.5 * beta_low**2
K_corr1 = 0.5 * beta_low**2 + (3/8) * beta_low**4

rel_err_newton = np.abs(K_newton - K_exact_low) / K_exact_low * 100
rel_err_corr1 = np.abs(K_corr1 - K_exact_low) / K_exact_low * 100

axes[1].semilogy(beta_low, rel_err_newton, 'b-', label='뉴턴 역학 오차')
axes[1].semilogy(beta_low, rel_err_corr1, 'r-', label='1차 보정 후 오차')
axes[1].axhline(y=1, color='gray', linestyle=':', alpha=0.5, label='1% 오차선')
axes[1].set_xlabel('beta = v/c')
axes[1].set_ylabel('상대 오차 (%)')
axes[1].set_title('뉴턴 역학의 유효 범위')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('relativistic_energy.png', dpi=150, bbox_inches='tight')
plt.show()

# 결과 출력
print("=== 뉴턴 역학이 유효한 속도 범위 ===\n")
print(f"{'v/c':>6s}  {'상대론적 K':>12s}  {'뉴턴 K':>10s}  {'오차(%)':>8s}")
print("-" * 42)
for b in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    g = 1 / np.sqrt(1 - b**2)
    K_rel = g - 1
    K_new = 0.5 * b**2
    err = abs(K_new - K_rel) / K_rel * 100
    print(f"{b:>6.2f}  {K_rel:>12.6f}  {K_new:>10.6f}  {err:>8.3f}")
```

### 6.3 전기 다중극 전개

점전하 $q$가 원점에서 거리 $d$인 곳에 있을 때, 원점에서 멀리 떨어진 점($r \gg d$)에서의 전위를 급수로 전개합니다.

```
    관측점 P(r, theta)
    *
    |\
    | \  r
    |  \
    |   \
    |theta\
    |------*--- 전하 q (거리 d, z축 위)
    |
    원점 O
```

관측점까지의 거리:

$$\frac{1}{|\mathbf{r} - \mathbf{d}|} = \frac{1}{r} \sum_{l=0}^{\infty} \left(\frac{d}{r}\right)^l P_l(\cos\theta)$$

여기서 $P_l$은 르장드르 다항식 (Legendre polynomials)입니다.

전기 쌍극자 (dipole)의 경우 ($+q$와 $-q$가 거리 $d$만큼 떨어져 있을 때):

$$V(r, \theta) \approx \frac{1}{4\pi\epsilon_0} \frac{p\cos\theta}{r^2} \quad (r \gg d)$$

여기서 $p = qd$는 쌍극자 모멘트 (dipole moment)입니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import legendre

# 다중극 전개 (Multipole Expansion) 시각화
# 전하 q가 z축 위 d = 1에 있을 때, 1/|r - d*z_hat|의 급수 전개

def exact_potential_ratio(r, theta, d=1.0):
    """정확한 전위 비율: r / |r - d*z_hat| (1/r 제외한 부분)"""
    # |r - d*z_hat|^2 = r^2 - 2*r*d*cos(theta) + d^2
    dist = np.sqrt(r**2 - 2*r*d*np.cos(theta) + d**2)
    return r / dist

def multipole_approx(r, theta, d=1.0, L_max=5):
    """다중극 전개: sum_{l=0}^{L_max} (d/r)^l * P_l(cos(theta))"""
    result = np.zeros_like(r)
    cos_theta = np.cos(theta)
    for l in range(L_max + 1):
        Pl = legendre(l)
        result += (d / r)**l * Pl(cos_theta)
    return result

# r/d 비율에 따른 다중극 전개의 정확도
r_values = np.linspace(1.5, 10, 200)
theta = np.pi / 4  # 45도

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

exact = exact_potential_ratio(r_values, theta)

for L_max in [0, 1, 2, 5, 10]:
    approx = multipole_approx(r_values, theta, L_max=L_max)
    axes[0].plot(r_values, approx, label=f'L_max = {L_max}')

axes[0].plot(r_values, exact, 'k--', linewidth=2, label='정확값')
axes[0].set_xlabel('r/d')
axes[0].set_ylabel('r * V(r) / (kq)')
axes[0].set_title(f'다중극 전개 (theta = 45도)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 쌍극자 전위 패턴 (2D)
r_grid = np.linspace(0.5, 5, 200)
theta_grid = np.linspace(0, 2*np.pi, 200)
R, Theta = np.meshgrid(r_grid, theta_grid)
X = R * np.sin(Theta)
Z = R * np.cos(Theta)

# 쌍극자 전위: V ~ cos(theta)/r^2 (단위 생략)
V_dipole = np.cos(Theta) / R**2

axes[1].contourf(X, Z, V_dipole, levels=np.linspace(-2, 2, 41),
                 cmap='RdBu_r', extend='both')
axes[1].set_xlabel('x')
axes[1].set_ylabel('z')
axes[1].set_title('전기 쌍극자 전위 패턴')
axes[1].set_aspect('equal')
axes[1].set_xlim(-3, 3)
axes[1].set_ylim(-3, 3)
# 전하 위치 표시
axes[1].plot(0, 0.1, 'r+', markersize=15, markeredgewidth=2)
axes[1].plot(0, -0.1, 'b_', markersize=15, markeredgewidth=2)

plt.tight_layout()
plt.savefig('multipole_expansion.png', dpi=150, bbox_inches='tight')
plt.show()

# 르장드르 다항식의 처음 몇 개
print("=== 르장드르 다항식 P_l(x) ===\n")
import sympy as sp
x_sym = sp.Symbol('x')
for l in range(6):
    Pl = sp.legendre(l, x_sym)
    print(f"P_{l}(x) = {sp.expand(Pl)}")
```

---

## 연습 문제

### 문제 1. 수렴 판정 (기본)

다음 급수의 수렴/발산을 판정하시오. 사용한 판정법을 명시하시오.

(a) $\sum_{n=1}^{\infty} \frac{n^2}{3^n}$

(b) $\sum_{n=2}^{\infty} \frac{1}{n \ln^2(n)}$

(c) $\sum_{n=1}^{\infty} \frac{(-1)^n n}{n^2 + 1}$

**힌트**: (a) 비율 판정법, (b) 적분 판정법, (c) 교대급수 판정법

### 문제 2. 멱급수의 수렴 반경 (기본)

다음 멱급수의 수렴 반경 $R$을 구하시오.

(a) $\sum_{n=0}^{\infty} \frac{n!}{(2n)!} x^n$

(b) $\sum_{n=1}^{\infty} \frac{(-1)^{n+1}}{n \cdot 3^n} x^n$

### 문제 3. 테일러 급수 근사 (중급)

함수 $f(x) = \sqrt{1 + x}$를 $x = 0$에서 3차까지 테일러 전개하고, 다음을 구하시오.

(a) $\sqrt{1.1}$의 근사값 ($x = 0.1$ 대입)

(b) 정확값과 비교하여 상대 오차를 계산하시오.

(c) Python으로 1차, 2차, 3차 근사의 오차를 비교하는 그래프를 그리시오.

### 문제 4. 물리학 응용 (중급)

진자의 최대 각도가 $\theta_0 = 30$도일 때:

(a) 주기의 정확한 값 (타원 적분)을 계산하시오.

(b) 소각 근사의 주기와 비교하여 오차를 구하시오.

(c) 2차 보정항까지 포함했을 때의 오차를 구하시오.

### 문제 5. 스털링 근사 (중급)

(a) 스털링 근사를 이용하여 $100!$의 상용로그 ($\log_{10}$)를 구하시오.

(b) Python의 정확한 계산과 비교하시오.

(c) 이상기체 $N$개 분자의 배치 경우의 수 $W = N! / (n_1! n_2! \cdots n_k!)$에서, 스털링 근사가 왜 필수적인지 설명하시오.

### 문제 6. 종합 응용 (고급)

전자의 속도가 $v = 0.1c$일 때:

(a) 상대론적 운동 에너지의 정확한 값을 $mc^2$ 단위로 구하시오.

(b) 뉴턴 역학의 운동 에너지와 비교하여 상대 오차를 구하시오.

(c) 1차 상대론적 보정항 $(3/8)mv^4/c^2$을 포함하면 오차가 얼마나 줄어드는지 계산하시오.

(d) 뉴턴 역학의 오차가 1% 미만이 되려면 $v/c$가 얼마 이하여야 하는지 구하시오.

```python
# 연습 문제 풀이 도우미
import numpy as np
import sympy as sp

# 문제 1(a) 검증
n = sp.Symbol('n', positive=True, integer=True)
a_n = n**2 / 3**n
ratio = sp.simplify(sp.Abs((n+1)**2 / 3**(n+1)) / (n**2 / 3**n))
rho = sp.limit(ratio, n, sp.oo)
print(f"문제 1(a): 비율 = {rho} < 1 => 수렴")

# 문제 4 검증
from scipy.special import ellipk
L, g = 1.0, 9.81
theta0 = np.radians(30)
T0 = 2 * np.pi * np.sqrt(L / g)
k = np.sin(theta0 / 2)
T_exact = 4 * np.sqrt(L / g) * ellipk(k**2)
print(f"\n문제 4: T_exact = {T_exact:.6f}s, T_0 = {T0:.6f}s")
print(f"소각 근사 오차 = {abs(T0 - T_exact)/T_exact * 100:.4f}%")

T_corrected = T0 * (1 + (1/4) * k**2)
print(f"2차 보정 후 오차 = {abs(T_corrected - T_exact)/T_exact * 100:.6f}%")

# 문제 5 검증
import math
stirling_log10_100 = (100 * np.log(100) - 100 + 0.5 * np.log(200 * np.pi)) / np.log(10)
exact_log10_100 = np.log10(float(math.factorial(100)))
print(f"\n문제 5: log10(100!) 스털링 = {stirling_log10_100:.4f}")
print(f"         log10(100!) 정확값 = {exact_log10_100:.4f}")
print(f"         오차 = {abs(stirling_log10_100 - exact_log10_100):.6f}")
```

---

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 1. Wiley.
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 5. Academic Press.
3. **Riley, K. F. et al.** (2006). *Mathematical Methods for Physics and Engineering*, Chapter 4. Cambridge University Press.

### 온라인 자료
1. **MIT OCW 18.01SC**: Single Variable Calculus - Sequences and Series
2. **Paul's Online Math Notes**: Series & Sequences (https://tutorial.math.lamar.edu/)
3. **3Blue1Brown**: Taylor series 시각화

### Python 도구
- `sympy.series()`: 심볼릭 테일러 급수 계산
- `scipy.special.ellipk()`: 완전 타원 적분
- `scipy.special.factorial()`: 팩토리얼 계산
- `numpy.cumsum()`: 부분합 계산

---

## 다음 레슨

[02. 복소수 (Complex Numbers)](02_Complex_Numbers.md)에서는 복소수의 대수적 연산, 극좌표와 지수 표현, 드모아브르 정리, 그리고 물리학에서 필수적인 오일러 공식을 다룹니다. 급수에서 배운 테일러 전개를 이용하여 $e^{ix} = \cos(x) + i\sin(x)$를 유도하게 됩니다.
