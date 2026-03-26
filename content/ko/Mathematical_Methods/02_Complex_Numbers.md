# 02. 복소수 (Complex Numbers)

> **Boas Chapter 2** — 물리과학에서 복소수는 단순한 수학적 도구를 넘어, 파동 현상, 교류 회로, 양자역학 등 핵심 물리학의 언어입니다.

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

- **복소수의 대수적, 기하학적 표현**을 자유자재로 변환할 수 있다 (직교좌표, 극좌표, 지수 형식)
- **오일러 공식과 드모아브르 정리**를 이용하여 삼각함수 항등식을 유도하고 거듭제곱근을 구할 수 있다
- **복소 함수**(지수, 삼각, 쌍곡, 로그)의 성질을 이해하고 계산할 수 있다
- **물리학 응용**: 교류 회로의 임피던스 계산, 파동의 복소 표현, 양자역학적 파동함수를 다룰 수 있다
- **복소수를 이용한 2D 변환**의 기하학적 의미를 파악하고, 등각사상(conformal mapping)의 기본 원리를 이해한다

---

## 1. 복소수의 기본

### 1.1 허수 단위와 복소수 정의

실수(real number)만으로는 $x^2 + 1 = 0$과 같은 방정식을 풀 수 없습니다. **허수 단위**(imaginary unit) $i$를 다음과 같이 정의합니다:

$$
i^2 = -1, \quad i = \sqrt{-1}
$$

**복소수**(complex number) $z$는 두 실수 $a$, $b$로 구성됩니다:

$$
z = a + bi
$$

- $a = \text{Re}(z)$: **실수부**(real part)
- $b = \text{Im}(z)$: **허수부**(imaginary part)

> **참고**: 공학에서는 전류 $i$와의 혼동을 피하기 위해 허수 단위를 $j$로 표기합니다. Python도 `j`를 사용합니다.

**복소 켤레**(complex conjugate):

$$
\bar{z} = z^* = a - bi
$$

**절댓값**(modulus, absolute value):

$$
|z| = \sqrt{a^2 + b^2} = \sqrt{z \cdot \bar{z}}
$$

```python
import numpy as np

# 복소수 기본 연산
z1 = 3 + 4j
z2 = 1 - 2j

print(f"z1 = {z1}")
print(f"실수부: {z1.real}, 허수부: {z1.imag}")
print(f"켤레: {z1.conjugate()}")
print(f"|z1| = {abs(z1):.4f}")  # sqrt(9 + 16) = 5.0

# 유용한 성질
print(f"\nz1 * conj(z1) = {z1 * z1.conjugate()}")  # |z1|^2 = 25
print(f"|z1|^2 = {abs(z1)**2}")
```

### 1.2 복소 평면 (아르강 다이어그램)

복소수 $z = a + bi$는 2차원 평면 위의 점 $(a, b)$로 표현됩니다. 이 평면을 **복소 평면**(complex plane) 또는 **아르강 다이어그램**(Argand diagram)이라 합니다.

- **가로축**: 실수축 (real axis)
- **세로축**: 허수축 (imaginary axis)
- **점 $z$까지의 거리**: $|z|$ (원점으로부터의 거리)
- **실수축과의 각도**: $\theta = \arg(z)$ (편각, argument)

```python
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# 복소수 배열
points = {
    r'$3+4i$': 3+4j,
    r'$-2+3i$': -2+3j,
    r'$-1-2i$': -1-2j,
    r'$4-i$': 4-1j,
    r'$2i$': 2j,
    r'$-3$': -3+0j,
}

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

for (label, z), color in zip(points.items(), colors):
    # 점 표시
    ax.plot(z.real, z.imag, 'o', color=color, markersize=10, zorder=5)
    ax.annotate(label, (z.real, z.imag), textcoords="offset points",
                xytext=(10, 10), fontsize=12, color=color)
    # 원점에서 화살표
    ax.annotate('', xy=(z.real, z.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# 축 설정
ax.axhline(y=0, color='k', linewidth=0.8)
ax.axvline(x=0, color='k', linewidth=0.8)
ax.set_xlim(-5, 6)
ax.set_ylim(-4, 6)
ax.set_xlabel('Re(z)', fontsize=13)
ax.set_ylabel('Im(z)', fontsize=13)
ax.set_title('복소 평면 (Argand Diagram)', fontsize=15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complex_plane.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 1.3 복소수의 사칙연산

두 복소수 $z_1 = a + bi$, $z_2 = c + di$에 대해:

**덧셈/뺄셈**: 실수부와 허수부를 각각 연산

$$
z_1 \pm z_2 = (a \pm c) + (b \pm d)i
$$

**곱셈**: $i^2 = -1$을 이용하여 전개

$$
z_1 \cdot z_2 = (ac - bd) + (ad + bc)i
$$

**나눗셈**: 분모를 켤레로 유리화(rationalization)

$$
\frac{z_1}{z_2} = \frac{z_1 \cdot \bar{z_2}}{|z_2|^2} = \frac{(ac + bd) + (bc - ad)i}{c^2 + d^2}
$$

```python
import numpy as np

z1 = 3 + 4j
z2 = 1 - 2j

print("=== 복소수 사칙연산 ===")
print(f"z1 + z2 = {z1 + z2}")          # (4+2j)
print(f"z1 - z2 = {z1 - z2}")          # (2+6j)
print(f"z1 * z2 = {z1 * z2}")          # (11-2j) = (3+8)+(4-6)i
print(f"z1 / z2 = {z1 / z2}")          # (-1+2j)

# 나눗셈 검증: 수동 계산
numerator = z1 * z2.conjugate()
denominator = abs(z2)**2
print(f"\n나눗셈 수동 검증:")
print(f"  z1 * conj(z2) = {numerator}")
print(f"  |z2|^2 = {denominator}")
print(f"  결과 = {numerator / denominator}")
```

---

## 2. 극좌표 표현과 지수 표현

### 2.1 극좌표 형식 (r, $\theta$)

복소수 $z = a + bi$를 극좌표로 표현하면:

$$
z = r(\cos\theta + i\sin\theta)
$$

여기서:
- $r = |z| = \sqrt{a^2 + b^2}$ : 절댓값 (modulus)
- $\theta = \arg(z) = \arctan\left(\frac{b}{a}\right)$ : 편각 (argument)

**역변환**:

$$
a = r\cos\theta, \quad b = r\sin\theta
$$

> **주의**: $\arctan(b/a)$만으로는 사분면을 구별할 수 없으므로, 프로그래밍에서는 `np.arctan2(b, a)` 또는 `np.angle(z)`를 사용합니다.

### 2.2 오일러 공식 (Euler's Formula)

수학에서 가장 아름다운 공식 중 하나인 **오일러 공식**:

$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

이를 이용하면 복소수를 **지수 형식**(exponential form)으로 표현할 수 있습니다:

$$
z = re^{i\theta}
$$

**오일러 등식** ($\theta = \pi$ 대입):

$$
e^{i\pi} + 1 = 0
$$

이 등식은 수학의 다섯 가지 기본 상수($e$, $i$, $\pi$, $1$, $0$)를 하나의 식으로 연결합니다.

**증명 (테일러 급수 이용)**:

$$
e^{i\theta} = \sum_{n=0}^{\infty} \frac{(i\theta)^n}{n!}
= \underbrace{\left(1 - \frac{\theta^2}{2!} + \frac{\theta^4}{4!} - \cdots\right)}_{\cos\theta}
+ i\underbrace{\left(\theta - \frac{\theta^3}{3!} + \frac{\theta^5}{5!} - \cdots\right)}_{\sin\theta}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# 오일러 공식 시각화: e^{i*theta}는 단위원 위의 점
theta = np.linspace(0, 2*np.pi, 300)
z = np.exp(1j * theta)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) 단위원 위의 e^{i*theta}
ax = axes[0]
ax.plot(z.real, z.imag, 'b-', linewidth=2)

# 특별한 각도 표시
special_angles = [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2, np.pi,
                  3*np.pi/2]
labels = [r'$1$', r'$e^{i\pi/6}$', r'$e^{i\pi/4}$', r'$e^{i\pi/3}$',
          r'$e^{i\pi/2}=i$', r'$e^{i\pi}=-1$', r'$e^{i3\pi/2}=-i$']

for angle, label in zip(special_angles, labels):
    w = np.exp(1j * angle)
    ax.plot(w.real, w.imag, 'ro', markersize=8, zorder=5)
    offset = (15 * np.cos(angle), 15 * np.sin(angle))
    ax.annotate(label, (w.real, w.imag), textcoords="offset points",
                xytext=offset, fontsize=10, ha='center')

ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_xlim(-1.6, 1.6)
ax.set_ylim(-1.6, 1.6)
ax.set_aspect('equal')
ax.set_title(r'$e^{i\theta}$: 단위원 위의 복소수', fontsize=14)
ax.set_xlabel('Re', fontsize=12)
ax.set_ylabel('Im', fontsize=12)
ax.grid(True, alpha=0.3)

# (b) 오일러 공식의 테일러 급수 수렴
ax2 = axes[1]
theta_val = np.pi / 3  # 60도

n_terms_list = range(1, 12)
partial_real = []
partial_imag = []
cumsum = 0 + 0j

for n in range(20):
    cumsum += (1j * theta_val)**n / np.math.factorial(n)
    if n + 1 <= 11:
        partial_real.append(cumsum.real)
        partial_imag.append(cumsum.imag)

exact = np.exp(1j * theta_val)
ax2.axhline(y=exact.real, color='blue', linestyle='--', alpha=0.5,
            label=f'cos(pi/3) = {exact.real:.4f}')
ax2.axhline(y=exact.imag, color='red', linestyle='--', alpha=0.5,
            label=f'sin(pi/3) = {exact.imag:.4f}')
ax2.plot(range(1, 12), partial_real, 'bo-', label='부분합 실수부')
ax2.plot(range(1, 12), partial_imag, 'rs-', label='부분합 허수부')
ax2.set_xlabel('항의 수', fontsize=12)
ax2.set_ylabel('값', fontsize=12)
ax2.set_title(r'$e^{i\pi/3}$ 테일러 급수의 수렴', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('euler_formula.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 2.3 지수 형식과 곱셈/나눗셈

지수 형식은 곱셈과 나눗셈을 극도로 간단하게 만듭니다.

$z_1 = r_1 e^{i\theta_1}$, $z_2 = r_2 e^{i\theta_2}$ 일 때:

**곱셈**: 절댓값은 곱하고, 편각은 더한다

$$
z_1 \cdot z_2 = r_1 r_2 \, e^{i(\theta_1 + \theta_2)}
$$

**나눗셈**: 절댓값은 나누고, 편각은 뺀다

$$
\frac{z_1}{z_2} = \frac{r_1}{r_2} \, e^{i(\theta_1 - \theta_2)}
$$

**거듭제곱**: 절댓값은 거듭제곱, 편각은 배수

$$
z^n = r^n e^{in\theta}
$$

```python
import numpy as np

# 극좌표/지수 형식 변환
z = 1 + 1j * np.sqrt(3)  # = 2 * e^{i*pi/3}

r = abs(z)
theta = np.angle(z)  # 라디안

print(f"z = {z}")
print(f"|z| = {r:.4f}")
print(f"arg(z) = {theta:.4f} rad = {np.degrees(theta):.1f}°")
print(f"지수 형식: {r:.4f} * exp(i * {theta:.4f})")

# 곱셈 예시
z1 = 2 * np.exp(1j * np.pi/4)   # r=2, theta=45°
z2 = 3 * np.exp(1j * np.pi/6)   # r=3, theta=30°
z_product = z1 * z2

print(f"\n=== 곱셈 ===")
print(f"z1 = 2*exp(i*pi/4), z2 = 3*exp(i*pi/6)")
print(f"z1*z2 = {z_product:.4f}")
print(f"|z1*z2| = {abs(z_product):.4f} (= 2*3 = 6)")
print(f"arg(z1*z2) = {np.degrees(np.angle(z_product)):.1f}° (= 45+30 = 75°)")
```

---

## 3. 드모아브르 정리와 거듭제곱근

### 3.1 드모아브르 정리 (De Moivre's Theorem)

오일러 공식에서 직접 유도되는 중요한 정리입니다:

$$
(\cos\theta + i\sin\theta)^n = \cos(n\theta) + i\sin(n\theta)
$$

이 정리는 다음과 같은 강력한 응용을 가집니다:
- **다중각 공식**(multiple angle formulas) 유도
- **삼각함수 항등식** 증명
- **거듭제곱근** 계산

**예시**: $\cos(3\theta)$를 $\cos\theta$로 표현

드모아브르 정리에서 $n = 3$:

$$
\cos(3\theta) + i\sin(3\theta) = (\cos\theta + i\sin\theta)^3
$$

우변을 이항정리로 전개하고 실수부를 비교하면:

$$
\cos(3\theta) = \cos^3\theta - 3\cos\theta\sin^2\theta = 4\cos^3\theta - 3\cos\theta
$$

```python
import sympy as sp

# SymPy로 드모아브르 정리를 이용한 다중각 공식 유도
theta = sp.Symbol('theta', real=True)

# (cos(theta) + i*sin(theta))^n 을 전개하여 cos(n*theta), sin(n*theta) 유도
for n in [2, 3, 4]:
    expr = sp.expand((sp.cos(theta) + sp.I * sp.sin(theta))**n)
    real_part = sp.re(expr)
    imag_part = sp.im(expr)

    # sin^2 = 1 - cos^2 으로 치환하여 cos만으로 표현
    real_simplified = sp.simplify(
        real_part.rewrite(sp.cos).expand(trig=True)
    )
    imag_simplified = sp.simplify(
        imag_part.rewrite(sp.sin).expand(trig=True)
    )

    print(f"=== n = {n} ===")
    print(f"  cos({n}*theta) = {sp.trigsimp(real_part)}")
    print(f"  sin({n}*theta) = {sp.trigsimp(imag_part)}")
    print()

# 수치 검증
import numpy as np
t = np.pi / 7
for n in [2, 3, 4]:
    lhs = np.cos(n * t)
    rhs = (np.exp(1j * t)**n).real
    print(f"cos({n}*pi/7): 직접 계산={lhs:.10f}, 드모아브르={rhs:.10f}, 차이={abs(lhs-rhs):.2e}")
```

### 3.2 n제곱근

복소수 $w$의 **$n$제곱근**, 즉 $z^n = w$를 만족하는 $z$를 구합니다.

$w = Re^{i\Phi}$로 놓으면 ($R = |w|$, $\Phi = \arg(w)$):

$$
z_k = R^{1/n} \exp\left(i\frac{\Phi + 2\pi k}{n}\right), \quad k = 0, 1, 2, \ldots, n-1
$$

핵심: **$n$개의 서로 다른 근**이 존재하며, 이들은 반지름 $R^{1/n}$인 원 위에 **균등 배치**됩니다.

```python
import numpy as np
import matplotlib.pyplot as plt

def nth_roots(w, n):
    """복소수 w의 n제곱근을 모두 구한다."""
    R = abs(w)
    Phi = np.angle(w)
    roots = []
    for k in range(n):
        z_k = R**(1/n) * np.exp(1j * (Phi + 2*np.pi*k) / n)
        roots.append(z_k)
    return np.array(roots)

# 예시: (1+i)의 세제곱근
w = 1 + 1j
n = 3
roots = nth_roots(w, n)

print(f"w = {w} 의 {n}제곱근:")
for k, root in enumerate(roots):
    print(f"  z_{k} = {root:.6f}")
    print(f"       = {abs(root):.4f} * exp(i * {np.degrees(np.angle(root)):.2f}°)")
    # 검증
    print(f"       검증: z_{k}^{n} = {root**n:.6f}")
    print()

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
test_cases = [
    (8+0j, 3, r'$\sqrt[3]{8}$'),
    (1+1j, 4, r'$\sqrt[4]{1+i}$'),
    (-1+0j, 5, r'$\sqrt[5]{-1}$'),
]

for ax, (w, n, title) in zip(axes, test_cases):
    roots = nth_roots(w, n)
    R = abs(w)**(1/n)

    # 원 그리기
    circle_theta = np.linspace(0, 2*np.pi, 200)
    ax.plot(R*np.cos(circle_theta), R*np.sin(circle_theta),
            'b--', alpha=0.3, linewidth=1)

    # 근 표시
    for k, z_k in enumerate(roots):
        ax.plot(z_k.real, z_k.imag, 'ro', markersize=10, zorder=5)
        ax.annotate(f'$z_{k}$', (z_k.real, z_k.imag),
                    textcoords="offset points", xytext=(8, 8), fontsize=11)

    # 정다각형 연결선
    polygon = np.append(roots, roots[0])
    ax.plot(polygon.real, polygon.imag, 'r-', alpha=0.4, linewidth=1)

    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

plt.suptitle('복소수의 n제곱근: 원 위에 균등 배치', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('nth_roots.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 3.3 1의 n제곱근 (Roots of Unity)

$w = 1$인 특수한 경우: $z^n = 1$의 근을 **1의 n제곱근**(n-th roots of unity)이라 합니다.

$$
\omega_k = e^{2\pi i k / n}, \quad k = 0, 1, \ldots, n-1
$$

**원시 n제곱근**(primitive root): $\omega = e^{2\pi i / n}$으로 놓으면

$$
\omega_k = \omega^k, \quad k = 0, 1, \ldots, n-1
$$

**핵심 성질**:

1. $\omega^n = 1$
2. $1 + \omega + \omega^2 + \cdots + \omega^{n-1} = 0$ (근의 합은 0)
3. $\omega_j \cdot \omega_k = \omega_{(j+k) \bmod n}$ (군 구조)

> **응용**: 1의 n제곱근은 **이산 푸리에 변환**(DFT)의 핵심이며, 신호 처리에서 필수적입니다.

```python
import numpy as np

# 1의 n제곱근의 성질 검증
for n in [3, 4, 6, 8]:
    omega = np.exp(2j * np.pi / n)
    roots = np.array([omega**k for k in range(n)])

    print(f"=== 1의 {n}제곱근 ===")
    print(f"  원시근 omega = exp(2*pi*i/{n}) = {omega:.6f}")
    print(f"  omega^{n} = {omega**n:.6f} (= 1 검증)")
    print(f"  근의 합 = {roots.sum():.6f} (= 0 검증)")
    print(f"  근의 곱 = {np.prod(roots):.6f}")
    print()
```

---

## 4. 복소 함수

### 4.1 복소 지수함수

복소수 $z = x + iy$에 대해 복소 지수함수는:

$$
e^z = e^{x+iy} = e^x(\cos y + i\sin y)
$$

**주요 성질**:
- $|e^z| = e^x = e^{\text{Re}(z)}$ (절댓값은 실수부에만 의존)
- $\arg(e^z) = y = \text{Im}(z)$
- $e^z$는 **주기 $2\pi i$**: $e^{z + 2\pi i} = e^z$

```python
import numpy as np
import matplotlib.pyplot as plt

# 복소 지수함수 시각화: e^z의 상 (image)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# (a) 직선 x = const 의 상: 원점 중심 원
ax = axes[0]
y = np.linspace(0, 2*np.pi, 200)
for x_val in [-1, -0.5, 0, 0.5, 1]:
    w = np.exp(x_val + 1j*y)
    ax.plot(w.real, w.imag, label=f'x={x_val}')

ax.set_aspect('equal')
ax.set_title(r'$e^{x+iy}$: 수직선 $x=\mathrm{const}$ 의 상', fontsize=13)
ax.set_xlabel('Re', fontsize=12)
ax.set_ylabel('Im', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# (b) 직선 y = const 의 상: 원점에서 뻗어나가는 반직선
ax2 = axes[1]
x = np.linspace(-2, 2, 200)
for y_val in np.linspace(0, 2*np.pi, 9)[:-1]:
    w = np.exp(x + 1j*y_val)
    ax2.plot(w.real, w.imag, label=f'y={y_val:.2f}')

ax2.set_xlim(-8, 8)
ax2.set_ylim(-8, 8)
ax2.set_aspect('equal')
ax2.set_title(r'$e^{x+iy}$: 수평선 $y=\mathrm{const}$ 의 상', fontsize=13)
ax2.set_xlabel('Re', fontsize=12)
ax2.set_ylabel('Im', fontsize=12)
ax2.legend(fontsize=9, ncol=2)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complex_exp.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 4.2 복소 삼각함수와 쌍곡함수

오일러 공식을 이용하면 삼각함수와 쌍곡함수를 지수함수로 정의할 수 있습니다.

**복소 삼각함수**:

$$
\cos z = \frac{e^{iz} + e^{-iz}}{2}, \quad \sin z = \frac{e^{iz} - e^{-iz}}{2i}
$$

**복소 쌍곡함수** (hyperbolic functions):

$$
\cosh z = \frac{e^z + e^{-z}}{2}, \quad \sinh z = \frac{e^z - e^{-z}}{2}
$$

**삼각함수와 쌍곡함수의 관계**:

$$
\cos(iz) = \cosh z, \quad \sin(iz) = i\sinh z
$$

$$
\cosh(iz) = \cos z, \quad \sinh(iz) = i\sin z
$$

> **중요**: 실수에서는 $|\sin x| \leq 1$, $|\cos x| \leq 1$이지만, 복소수에서는 이 제한이 사라집니다. 예를 들어 $\cos(i) = \cosh(1) \approx 1.543$입니다.

```python
import numpy as np

# 복소 삼각함수/쌍곡함수의 관계 검증
z_values = [1+1j, 2-0.5j, 0+2j, np.pi/4+0j]

print("=== 복소 삼각함수 ===")
for z in z_values:
    # cos(z)를 지수 정의로 직접 계산
    cos_euler = (np.exp(1j*z) + np.exp(-1j*z)) / 2
    cos_numpy = np.cos(z)
    print(f"z = {z:.4f}")
    print(f"  cos(z) [오일러] = {cos_euler:.6f}")
    print(f"  cos(z) [NumPy]  = {cos_numpy:.6f}")
    print(f"  차이 = {abs(cos_euler - cos_numpy):.2e}")
    print()

# cos(iz) = cosh(z) 검증
z = 1.5 + 0.7j
print("=== 관계식 검증 ===")
print(f"cos(iz) = {np.cos(1j*z):.8f}")
print(f"cosh(z) = {np.cosh(z):.8f}")
print(f"sin(iz) = {np.sin(1j*z):.8f}")
print(f"i*sinh(z) = {1j*np.sinh(z):.8f}")
```

### 4.3 복소 로그함수

복소수 $z = re^{i\theta}$의 로그:

$$
\ln z = \ln r + i\theta = \ln|z| + i\arg(z)
$$

**다가함수** (multi-valued function): 편각 $\theta$는 $2\pi$의 정수배만큼 자유도가 있으므로:

$$
\text{Ln}(z) = \ln|z| + i(\theta + 2n\pi), \quad n \in \mathbb{Z}
$$

- **주값**(principal value): $-\pi < \theta \leq \pi$로 제한한 값을 $\text{Log}(z)$로 표기합니다.
- **분지**(branch): 다가함수를 단가함수로 만들기 위해 편각의 범위를 제한하는 것
- **분지점**(branch point): $z = 0$ (로그가 정의되지 않는 점)
- **분지 절단**(branch cut): 통상적으로 음의 실수축

```python
import numpy as np

# 복소 로그의 다가성
z = -1 + 0j

print("=== ln(-1)의 다가성 ===")
print(f"주값: Log(-1) = {np.log(-1+0j)}")  # i*pi

for n in range(-2, 3):
    val = np.log(abs(z)) + 1j * (np.pi + 2*np.pi*n)
    # 검증: e^val = z?
    check = np.exp(val)
    print(f"  n={n:+d}: ln(-1) = {val:.6f}, exp(ln(-1)) = {check:.6f}")

# 복소 로그의 성질 검증
z1 = 2 + 3j
z2 = 1 - 1j
print(f"\n=== 로그 성질 (주값 기준) ===")
print(f"Log(z1*z2) = {np.log(z1*z2):.6f}")
print(f"Log(z1) + Log(z2) = {np.log(z1) + np.log(z2):.6f}")
print("주의: 주값에서는 Log(z1*z2) != Log(z1) + Log(z2)일 수 있음")
print(f"차이 = {abs(np.log(z1*z2) - np.log(z1) - np.log(z2)):.6e}")
```

---

## 5. 물리학에서의 응용

### 5.1 교류 회로 (임피던스)

교류(AC) 회로 해석에서 복소수는 필수적입니다. 전압/전류를 복소 지수로 표현하면 미분방정식이 대수 방정식으로 변환됩니다.

**복소 임피던스**(impedance) $Z$:

| 소자 | 임피던스 | 위상 |
|------|----------|------|
| 저항 $R$ | $Z_R = R$ | $0$ |
| 인덕터 $L$ | $Z_L = i\omega L$ | $+90°$ |
| 커패시터 $C$ | $Z_C = \frac{1}{i\omega C} = -\frac{i}{\omega C}$ | $-90°$ |

**RLC 직렬 회로**의 총 임피던스:

$$
Z = R + i\left(\omega L - \frac{1}{\omega C}\right)
$$

- $|Z|$: 임피던스의 크기 (전압 진폭 / 전류 진폭)
- $\arg(Z)$: 전압과 전류 사이의 위상차
- **공진** (resonance): $\omega L = 1/(\omega C)$일 때, $Z = R$ (순수 저항)

$$
\omega_0 = \frac{1}{\sqrt{LC}} \quad \text{(공진 주파수)}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# RLC 직렬 회로의 임피던스 분석
R = 100      # Ohm
L = 0.1      # Henry
C = 1e-6     # Farad

omega_0 = 1 / np.sqrt(L * C)  # 공진 각주파수
f_0 = omega_0 / (2 * np.pi)   # 공진 주파수

print(f"공진 주파수: f_0 = {f_0:.1f} Hz")
print(f"공진 각주파수: omega_0 = {omega_0:.1f} rad/s")

# 주파수 범위
omega = np.linspace(100, 20000, 2000)
Z = R + 1j * (omega * L - 1/(omega * C))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) |Z| vs omega
ax = axes[0, 0]
ax.semilogy(omega, np.abs(Z), 'b-', linewidth=2)
ax.axvline(x=omega_0, color='r', linestyle='--', alpha=0.7,
           label=f'$\\omega_0$ = {omega_0:.0f} rad/s')
ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
ax.set_ylabel(r'$|Z|$ ($\Omega$)', fontsize=12)
ax.set_title('임피던스 크기', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (b) arg(Z) vs omega
ax = axes[0, 1]
ax.plot(omega, np.degrees(np.angle(Z)), 'g-', linewidth=2)
ax.axvline(x=omega_0, color='r', linestyle='--', alpha=0.7)
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
ax.set_ylabel(r'$\arg(Z)$ (degrees)', fontsize=12)
ax.set_title('위상각', fontsize=13)
ax.grid(True, alpha=0.3)

# (c) 전류 응답 (V_0 = 1V)
V0 = 1.0  # 전압 진폭
I = V0 / Z

ax = axes[1, 0]
ax.plot(omega, np.abs(I) * 1000, 'm-', linewidth=2)
ax.axvline(x=omega_0, color='r', linestyle='--', alpha=0.7,
           label=f'공진: I_max = {1000*V0/R:.1f} mA')
ax.set_xlabel(r'$\omega$ (rad/s)', fontsize=12)
ax.set_ylabel(r'$|I|$ (mA)', fontsize=12)
ax.set_title('전류 응답 (공진 곡선)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# (d) 임피던스의 복소 평면 궤적
ax = axes[1, 1]
ax.plot(Z.real, Z.imag, 'b-', linewidth=2)
ax.plot(R, 0, 'ro', markersize=10, zorder=5, label=f'공진점 ($\\omega_0$)')
ax.set_xlabel(r'Re$(Z)$ ($\Omega$)', fontsize=12)
ax.set_ylabel(r'Im$(Z)$ ($\Omega$)', fontsize=12)
ax.set_title('임피던스 궤적 (Nyquist plot)', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.suptitle(f'RLC 직렬 회로 (R={R}Ω, L={L*1000}mH, C={C*1e6}μF)',
             fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig('rlc_circuit.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.2 파동의 복소 표현

일차원 진행파(traveling wave)는 복소수로 간결하게 표현됩니다:

$$
\psi(x, t) = A e^{i(kx - \omega t)}
$$

여기서:
- $A$: 복소 진폭 (크기와 초기 위상을 포함)
- $k$: 파수 (wave number), $k = 2\pi/\lambda$
- $\omega$: 각진동수, $\omega = 2\pi f$

실제 물리량은 **실수부**입니다:

$$
\text{Re}[\psi] = |A| \cos(kx - \omega t + \phi)
$$

여기서 $\phi = \arg(A)$.

**복소 표현의 장점**:
1. 미분이 간단: $\frac{\partial \psi}{\partial t} = -i\omega\psi$
2. 중첩(superposition)과 간섭(interference) 계산이 용이
3. 위상 관계가 명확

```python
import numpy as np
import matplotlib.pyplot as plt

# 두 파동의 중첩 (간섭)
x = np.linspace(0, 10, 500)
t = 0

# 파동 1: A1 = 1, k1 = 2, omega1 = 3
A1, k1, omega1 = 1.0, 2.0, 3.0
psi1 = A1 * np.exp(1j * (k1*x - omega1*t))

# 파동 2: A2 = 0.8, k2 = 2.2, omega2 = 3.1 (약간 다른 파수/진동수)
A2, k2, omega2 = 0.8, 2.2, 3.1
psi2 = A2 * np.exp(1j * (k2*x - omega2*t))

# 중첩
psi_total = psi1 + psi2

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

axes[0].plot(x, psi1.real, 'b-', linewidth=1.5, label='파동 1')
axes[0].plot(x, np.abs(psi1)*np.ones_like(x), 'b--', alpha=0.3)
axes[0].set_ylabel('Re(ψ₁)', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, psi2.real, 'r-', linewidth=1.5, label='파동 2')
axes[1].plot(x, np.abs(psi2)*np.ones_like(x), 'r--', alpha=0.3)
axes[1].set_ylabel('Re(ψ₂)', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

axes[2].plot(x, psi_total.real, 'purple', linewidth=1.5, label='중첩 (ψ₁ + ψ₂)')
axes[2].plot(x, np.abs(psi_total), 'k--', alpha=0.5, label='포락선 |ψ|')
axes[2].plot(x, -np.abs(psi_total), 'k--', alpha=0.5)
axes[2].set_ylabel('Re(ψ₁ + ψ₂)', fontsize=12)
axes[2].set_xlabel('x', fontsize=12)
axes[2].legend(fontsize=11)
axes[2].grid(True, alpha=0.3)

plt.suptitle('파동의 복소 표현과 맥놀이(Beat) 현상', fontsize=14)
plt.tight_layout()
plt.savefig('wave_superposition.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.3 양자역학의 파동함수

양자역학에서 입자의 상태는 **복소 파동함수** $\Psi(x, t)$로 기술됩니다.

**자유 입자의 파동함수**:

$$
\Psi(x, t) = A \exp\left[i\left(kx - \frac{\hbar k^2}{2m}t\right)\right]
$$

**확률 밀도**: $|\Psi(x,t)|^2 = \Psi^* \Psi$ (관측 가능한 물리량은 항상 실수)

**슈뢰딩거 방정식**:

$$
i\hbar \frac{\partial \Psi}{\partial t} = -\frac{\hbar^2}{2m}\frac{\partial^2 \Psi}{\partial x^2} + V(x)\Psi
$$

> **본질적 복소성**: 슈뢰딩거 방정식에는 $i$가 명시적으로 나타납니다. 양자역학에서 복소수는 단순한 편의가 아니라 **물리의 본질**입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 가우시안 파동 패킷의 시간 진화
hbar = 1.0  # 자연 단위계
m = 1.0
sigma_0 = 1.0   # 초기 파동 패킷 폭
k_0 = 5.0       # 평균 파수 (운동량)

x = np.linspace(-10, 20, 1000)

def gaussian_wavepacket(x, t, sigma_0, k_0, m, hbar):
    """가우시안 파동 패킷의 해석적 해."""
    sigma_t = sigma_0 * np.sqrt(1 + (hbar*t/(2*m*sigma_0**2))**2)
    phase_factor = hbar * t / (2 * m * sigma_0**2)

    psi = (2*np.pi*sigma_0**2)**(-0.25) * (sigma_0 / sigma_t) * np.exp(
        -(x - hbar*k_0*t/m)**2 / (4*sigma_0*sigma_t) *
        (sigma_0/sigma_t) *
        np.exp(-1j * np.arctan(phase_factor)/2)
    ) * np.exp(1j * (k_0*x - hbar*k_0**2*t/(2*m)))

    # 정규화
    norm = np.sqrt(np.trapz(np.abs(psi)**2, x))
    return psi / norm if norm > 0 else psi

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

times = [0, 1, 2, 3, 4]
colors = plt.cm.viridis(np.linspace(0, 0.9, len(times)))

for t, color in zip(times, colors):
    psi = gaussian_wavepacket(x, t, sigma_0, k_0, m, hbar)
    prob = np.abs(psi)**2

    axes[0].plot(x, psi.real, color=color, linewidth=1.5,
                 label=f't = {t}', alpha=0.8)
    axes[1].plot(x, prob, color=color, linewidth=1.5,
                 label=f't = {t}', alpha=0.8)

axes[0].set_ylabel(r'Re($\Psi$)', fontsize=13)
axes[0].set_title('가우시안 파동 패킷의 시간 진화', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('x', fontsize=13)
axes[1].set_ylabel(r'$|\Psi|^2$', fontsize=13)
axes[1].set_title('확률 밀도 (파동 패킷 퍼짐)', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('wavepacket.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 5.4 삼각함수 항등식의 유도

오일러 공식을 이용하면 복잡한 삼각함수 항등식을 체계적으로 유도할 수 있습니다.

**방법**: $e^{i\theta} = \cos\theta + i\sin\theta$를 이용하여 지수법칙으로 전개한 뒤, 실수부와 허수부를 비교합니다.

**예시 1**: 덧셈정리

$$
e^{i(\alpha + \beta)} = e^{i\alpha} \cdot e^{i\beta}
$$

좌변: $\cos(\alpha+\beta) + i\sin(\alpha+\beta)$

우변: $(\cos\alpha + i\sin\alpha)(\cos\beta + i\sin\beta)$

실수부 비교: $\cos(\alpha+\beta) = \cos\alpha\cos\beta - \sin\alpha\sin\beta$

허수부 비교: $\sin(\alpha+\beta) = \sin\alpha\cos\beta + \cos\alpha\sin\beta$

**예시 2**: $\cos^n\theta$를 다중각의 코사인으로 표현

$$
\cos\theta = \frac{e^{i\theta} + e^{-i\theta}}{2}
$$

이므로:

$$
\cos^n\theta = \frac{1}{2^n}(e^{i\theta} + e^{-i\theta})^n
$$

이항정리로 전개하면 다중각 공식을 얻습니다.

```python
import sympy as sp

theta, alpha, beta = sp.symbols('theta alpha beta', real=True)

# 오일러 공식을 이용한 삼각함수 항등식 유도
print("=== 덧셈정리 유도 ===")
lhs = sp.exp(sp.I * (alpha + beta))
rhs = sp.exp(sp.I * alpha) * sp.exp(sp.I * beta)

# rhs를 전개
rhs_expanded = sp.expand(rhs, complex=True)
rhs_trig = sp.expand((sp.cos(alpha) + sp.I*sp.sin(alpha)) *
                      (sp.cos(beta) + sp.I*sp.sin(beta)))

print(f"실수부: cos(a+b) = {sp.re(rhs_trig)}")
print(f"허수부: sin(a+b) = {sp.im(rhs_trig)}")

# cos^n(theta)를 다중각으로 표현
print("\n=== cos^n(theta) 전개 ===")
for n in [2, 3, 4]:
    # cos(theta) = (e^{it} + e^{-it}) / 2
    t = sp.Symbol('t')
    expr = ((sp.exp(sp.I*t) + sp.exp(-sp.I*t)) / 2)**n
    expanded = sp.expand(expr)
    # e^{ikt} + e^{-ikt} = 2*cos(kt) 이용
    result = sp.simplify(sp.trigsimp(expanded.rewrite(sp.cos)))
    print(f"  cos^{n}(theta) = {result.subs(t, theta)}")

# 수치 검증
import numpy as np
theta_val = np.pi / 5
print(f"\n=== 수치 검증 (theta = pi/5) ===")
print(f"cos^3(theta) = {np.cos(theta_val)**3:.8f}")
print(f"(3*cos(theta) + cos(3*theta))/4 = "
      f"{(3*np.cos(theta_val) + np.cos(3*theta_val))/4:.8f}")
```

---

## 6. 복소수와 2D 변환

### 6.1 회전과 스케일링

복소수의 곱셈은 기하학적으로 **회전**(rotation)과 **스케일링**(scaling)에 대응합니다.

$w = e^{i\phi}$를 곱하면 각도 $\phi$만큼 **반시계 방향 회전**:

$$
z' = e^{i\phi} z
$$

일반적으로 $w = re^{i\phi}$를 곱하면:
- $r$배 스케일링
- $\phi$만큼 회전

**등각사상**(conformal mapping): 복소 해석함수에 의한 변환은 **각도를 보존**합니다. 이 성질이 유체역학, 전자기학에서 핵심적입니다.

```python
import numpy as np
import matplotlib.pyplot as plt

# 복소수 곱셈에 의한 회전과 스케일링
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 원래 도형: 삼각형
triangle = np.array([1+0j, 0+1j, -0.5-0.5j, 1+0j])

# (a) 순수 회전: e^{i*pi/4} 곱
angle = np.pi / 4
w_rotate = np.exp(1j * angle)

ax = axes[0]
ax.plot(triangle.real, triangle.imag, 'b-o', linewidth=2,
        markersize=8, label='원본')
rotated = w_rotate * triangle
ax.plot(rotated.real, rotated.imag, 'r-o', linewidth=2,
        markersize=8, label=f'회전 ({np.degrees(angle):.0f}°)')
ax.set_title(r'회전: $z \mapsto e^{i\pi/4} z$', fontsize=13)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 2)

# (b) 스케일링 + 회전: (1+i)*z = sqrt(2)*e^{i*pi/4}*z
w_scale_rotate = 1 + 1j

ax = axes[1]
ax.plot(triangle.real, triangle.imag, 'b-o', linewidth=2,
        markersize=8, label='원본')
transformed = w_scale_rotate * triangle
ax.plot(transformed.real, transformed.imag, 'r-o', linewidth=2,
        markersize=8, label=r'$(1+i) \cdot z$')
ax.set_title(r'스케일링+회전: $z \mapsto (1+i)z$', fontsize=13)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(-3, 3)
ax.set_ylim(-2, 3)

# (c) z^2 변환 (비선형, 등각)
theta = np.linspace(0, 2*np.pi, 200)
r_vals = [0.5, 0.75, 1.0]

ax = axes[2]
for r in r_vals:
    z_circle = r * np.exp(1j * theta)
    w_mapped = z_circle**2
    ax.plot(z_circle.real, z_circle.imag, 'b-', alpha=0.5, linewidth=1)
    ax.plot(w_mapped.real, w_mapped.imag, 'r-', alpha=0.7, linewidth=1.5)

ax.plot([], [], 'b-', label='원본 (원)', linewidth=2)
ax.plot([], [], 'r-', label=r'$z^2$ 변환', linewidth=2)
ax.set_title(r'비선형 등각사상: $z \mapsto z^2$', fontsize=13)
ax.set_aspect('equal')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complex_transformations.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.2 주코프스키 변환 (항공역학)

**주코프스키 변환**(Joukowski/Zhukovsky transform)은 항공역학에서 날개 단면(airfoil)의 유동을 분석하는 데 사용되는 대표적인 등각사상입니다:

$$
w = z + \frac{c^2}{z}
$$

- 원( $z$-평면)을 날개형 단면($w$-평면)으로 변환
- 원의 중심을 약간 이동시키면 다양한 날개 형상을 생성
- 유체의 속도장과 압력 분포를 해석적으로 계산 가능

```python
import numpy as np
import matplotlib.pyplot as plt

def joukowski(z, c=1.0):
    """주코프스키 변환: w = z + c^2/z"""
    return z + c**2 / z

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

theta = np.linspace(0, 2*np.pi, 500)
c = 1.0

# 다양한 원의 중심 이동에 따른 날개형 변화
offsets = [
    (0.0, 0.0, '원 (중심 원점)'),      # 완벽한 원 -> 직선
    (-0.1, 0.1, '약간 이동한 원'),       # 비대칭 날개형
    (-0.15, 0.15, '더 이동한 원'),       # 두꺼운 날개형
]

for ax, (dx, dy, title) in zip(axes, offsets):
    # z-평면: 이동된 원
    R = np.sqrt((c - dx)**2 + dy**2) + 0.02  # c를 포함하는 원
    z = (dx + dy*1j) + R * np.exp(1j * theta)

    # w-평면: 주코프스키 변환
    w = joukowski(z, c)

    # z-평면과 w-평면을 함께 표시
    ax.plot(z.real, z.imag, 'b-', linewidth=1.5, alpha=0.5,
            label=r'$z$-평면 (원)')
    ax.plot(w.real, w.imag, 'r-', linewidth=2.5,
            label=r'$w$-평면 (날개형)')

    # 특이점 표시
    ax.plot(c, 0, 'ko', markersize=6)
    ax.plot(-c, 0, 'ko', markersize=6)

    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-2.5, 2.5)

plt.suptitle(r'주코프스키 변환: $w = z + c^2/z$ (항공역학 응용)',
             fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('joukowski.png', dpi=150, bbox_inches='tight')
plt.show()
```

**유동장 시각화** (streamlines):

```python
import numpy as np
import matplotlib.pyplot as plt

# 주코프스키 날개 주위의 유동장
c = 1.0
dx, dy = -0.1, 0.08
R = np.sqrt((c - dx)**2 + dy**2) + 0.01
center = dx + dy*1j

# 복소 포텐셜: 일양류 + 원기둥 주위 유동 + 순환
U_inf = 1.0  # 자유류 속도
Gamma = 4 * np.pi * U_inf * R * np.sin(
    np.arctan2(dy, c - dx) + np.arcsin(Gamma_approx := 0.1)
) if False else 2.5  # 쿠타 조건에 의한 순환

# 격자 생성 (z-평면)
x_grid = np.linspace(-3, 4, 400)
y_grid = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x_grid, y_grid)
z_grid = X + 1j*Y

# 원 내부 마스킹
mask = np.abs(z_grid - center) < R

# 복소 속도 (z-평면에서)
# w(z) = U*(z - center) + U*R^2/(z - center) + i*Gamma/(2*pi)*log(z - center)
zeta = z_grid - center
F = U_inf * zeta + U_inf * R**2 / zeta - 1j*Gamma/(2*np.pi)*np.log(zeta)

# 유선 = Im(F) = const
psi = F.imag
psi[mask] = np.nan

# w-평면으로 변환
w_grid = joukowski(z_grid, c)
w_grid[mask] = np.nan

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# z-평면 유동
ax = axes[0]
levels = np.linspace(-4, 4, 40)
ax.contour(X, Y, psi, levels=levels, colors='steelblue', linewidths=0.7)
circle_plot = center + R * np.exp(1j * np.linspace(0, 2*np.pi, 200))
ax.fill(circle_plot.real, circle_plot.imag, color='lightgray', zorder=3)
ax.plot(circle_plot.real, circle_plot.imag, 'k-', linewidth=2, zorder=4)
ax.set_title('z-평면: 원기둥 주위 유동', fontsize=13)
ax.set_aspect('equal')
ax.set_xlim(-3, 4)
ax.set_ylim(-3, 3)

# w-평면 유동 (날개 주위)
ax = axes[1]
# 날개형 경계
theta_wing = np.linspace(0, 2*np.pi, 500)
z_wing = center + R * np.exp(1j * theta_wing)
w_wing = joukowski(z_wing, c)

ax.contour(w_grid.real, w_grid.imag, psi, levels=levels,
           colors='steelblue', linewidths=0.7)
ax.fill(w_wing.real, w_wing.imag, color='lightgray', zorder=3)
ax.plot(w_wing.real, w_wing.imag, 'k-', linewidth=2, zorder=4)
ax.set_title('w-평면: 날개형 주위 유동 (주코프스키 변환)', fontsize=13)
ax.set_aspect('equal')
ax.set_xlim(-3.5, 4.5)
ax.set_ylim(-3, 3)

plt.tight_layout()
plt.savefig('joukowski_flow.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 연습 문제

### 문제 1: 극좌표 변환

다음 복소수를 극좌표 형식 $re^{i\theta}$로 표현하시오 ($-\pi < \theta \leq \pi$):

(a) $z = -1 + i$
(b) $z = -3 - 3\sqrt{3}\,i$
(c) $z = 5i$

### 문제 2: 드모아브르 정리 응용

드모아브르 정리를 이용하여 $\sin(4\theta)$를 $\sin\theta$와 $\cos\theta$로 표현하시오.

### 문제 3: 거듭제곱근

$z^4 = -16$의 모든 근을 구하시오. 복소 평면에 도시하고, 결과를 $a + bi$ 형태로 표현하시오.

### 문제 4: 복소 로그

다음을 계산하시오 (주값):

(a) $\ln(-e)$
(b) $\ln(1 + i)$
(c) $i^i = e^{i \ln i}$

### 문제 5: 교류 회로 분석

$R = 50\,\Omega$, $L = 20\,\text{mH}$, $C = 10\,\mu\text{F}$인 RLC 직렬 회로에 $V(t) = 10\cos(\omega t)$ (V)가 인가될 때:

(a) 공진 주파수 $f_0$를 구하시오.
(b) $f = 500\,\text{Hz}$에서의 임피던스 $Z$의 크기와 위상을 구하시오.
(c) 공진 시 전류의 최대값을 구하시오.

### 문제 6: 등각사상

주코프스키 변환 $w = z + 1/z$에서:

(a) $|z| = 2$인 원이 $w$-평면에서 어떤 곡선으로 변환되는지 매개변수 표현을 구하시오.
(b) 이 곡선이 타원임을 보이고, 장축과 단축의 길이를 구하시오.

---

<details>
<summary><strong>연습 문제 풀이 (클릭하여 펼치기)</strong></summary>

```python
import numpy as np

# === 문제 1 풀이 ===
print("=== 문제 1: 극좌표 변환 ===\n")

problems_1 = {
    '(a) -1 + i': -1 + 1j,
    '(b) -3 - 3*sqrt(3)*i': -3 - 3*np.sqrt(3)*1j,
    '(c) 5i': 5j,
}

for label, z in problems_1.items():
    r = abs(z)
    theta = np.angle(z)
    print(f"{label}")
    print(f"  z = {z}")
    print(f"  r = {r:.6f}, theta = {theta:.6f} rad = {np.degrees(theta):.2f}°")
    print(f"  극좌표: {r:.4f} * exp(i * {theta:.4f})")
    print()

# === 문제 3 풀이 ===
print("=== 문제 3: z^4 = -16 ===\n")
w = -16 + 0j
n = 4
R = abs(w)**(1/n)   # 16^(1/4) = 2
Phi = np.angle(w)    # pi

for k in range(n):
    z_k = R * np.exp(1j * (Phi + 2*np.pi*k) / n)
    print(f"z_{k} = {z_k.real:+.6f} {z_k.imag:+.6f}i")
    print(f"     = {R:.4f} * exp(i * {np.degrees((Phi + 2*np.pi*k)/n):.1f}°)")
    print(f"     검증: z^4 = {z_k**4:.6f}")
    print()

# === 문제 4 풀이 ===
print("=== 문제 4: 복소 로그 ===\n")

# (a) ln(-e)
z_a = -np.e + 0j
print(f"(a) ln(-e) = {np.log(z_a):.6f}")
print(f"    = ln(e) + i*pi = 1 + i*pi = {1 + 1j*np.pi:.6f}\n")

# (b) ln(1+i)
z_b = 1 + 1j
print(f"(b) ln(1+i) = {np.log(z_b):.6f}")
print(f"    = ln(sqrt(2)) + i*pi/4 = {np.log(np.sqrt(2)) + 1j*np.pi/4:.6f}\n")

# (c) i^i
z_c = 1j ** 1j
print(f"(c) i^i = {z_c:.10f}")
print(f"    = exp(i * ln(i)) = exp(i * i*pi/2) = exp(-pi/2)")
print(f"    = {np.exp(-np.pi/2):.10f}")

# === 문제 5 풀이 ===
print("\n=== 문제 5: RLC 회로 ===\n")
R = 50
L = 20e-3
C = 10e-6

# (a)
f_0 = 1 / (2*np.pi*np.sqrt(L*C))
print(f"(a) 공진 주파수: f_0 = {f_0:.2f} Hz")

# (b)
f = 500
omega = 2 * np.pi * f
Z = R + 1j*(omega*L - 1/(omega*C))
print(f"(b) f = {f} Hz에서:")
print(f"    Z = {Z:.4f}")
print(f"    |Z| = {abs(Z):.4f} Ohm")
print(f"    arg(Z) = {np.degrees(np.angle(Z)):.2f}°")

# (c)
V0 = 10
I_max = V0 / R
print(f"(c) 공진 시 I_max = V0/R = {I_max:.4f} A = {I_max*1000:.1f} mA")
```

</details>

---

## 참고 자료

### 교재
1. **Boas, M. L.** (2005). *Mathematical Methods in the Physical Sciences*, 3rd ed., Chapter 2. Wiley.
2. **Arfken, G. B., Weber, H. J., & Harris, F. E.** (2012). *Mathematical Methods for Physicists*, 7th ed., Chapter 6. Academic Press.
3. **Needham, T.** (1997). *Visual Complex Analysis*. Oxford University Press. — 복소수의 기하학적 이해에 탁월한 교재

### 온라인 자료
1. **MIT OCW 18.04**: Complex Variables with Applications
2. **3Blue1Brown**: "What is Euler's Formula?" (YouTube) — 오일러 공식의 직관적 이해
3. **Better Explained**: *An Intuitive Guide to Imaginary Numbers*

### 연관 레슨
- [01. 무한급수](01_Infinite_Series.md): 테일러 급수 (오일러 공식 증명에 사용)
- [14. 복소해석](14_Complex_Analysis.md): 해석함수, 코시 정리, 유수 정리 (본 레슨의 심화)
- [07. 푸리에 급수](07_Fourier_Series.md): 1의 n제곱근과 DFT의 연결

---

## 다음 레슨

[03. 선형대수 (Linear Algebra)](03_Linear_Algebra.md)에서는 행렬, 행렬식, 고유값 문제, 이차형식 등 물리학과 공학에 필수적인 선형대수를 다룹니다.
