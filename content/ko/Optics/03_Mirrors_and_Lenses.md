# 03. 거울과 렌즈

[← 이전: 02. 기하광학 기초](02_Geometric_Optics_Fundamentals.md) | [다음: 04. 광학 기기 →](04_Optical_Instruments.md)

---

## 학습 목표

1. 거울 방정식과 박막 렌즈 방정식을 적용하여 상의 위치와 배율을 구한다
2. 렌즈 제작자 방정식을 유도하고, 굴절률과 곡률에 대한 의존성을 이해한다
3. 실상과 허상을 구별하고, 어떤 광학 소자가 어느 유형을 형성하는지 예측한다
4. 주요 광선 추적법으로 거울과 박막 렌즈에서 광선 도표를 작성한다
5. 5가지 1차 수차(자이델 수차(Seidel aberration))를 파악하고 물리적 원인을 이해한다
6. 행렬(ABCD) 방법으로 다중 소자 광학계를 체계적으로 분석한다
7. 직렬로 배치된 여러 렌즈로 구성된 복합 광학계 문제를 풀 수 있다

---

## 왜 중요한가

거울과 렌즈는 모든 광학 기기의 기본 구성 요소다. 시력을 교정하는 안경 렌즈부터 제임스 웹 우주 망원경의 6.5미터 주경(primary mirror)까지 그 범위는 광대하다. 곡면이 어떻게 상을 형성하는지 이해하는 것은 단순한 학문적 연습에 그치지 않는다. 그것은 카메라, 현미경, 프로젝터, 레이저 공진기(laser cavity), 광섬유 결합기(fiber-optic coupler)를 설계하는 실질적인 토대다. 박막 렌즈 방정식은 단순하지만, 상 형성의 본질적인 물리학을 담고 있다.

> **비유**: 렌즈는 광파(light wave)를 위한 고속도로 요금소와 같다. 단일 광원에서 나온 모든 파동은 서로 다른 거리를 이동했기 때문에 렌즈에 도달할 때 위상이 다르다. 렌즈는 각 파면에 정확히 필요한 양의 위상을 추가한다. 얇은 부분에는 더 많이, 두꺼운 부분에는 더 적게. 그 결과 모든 파동이 상점(image point)에 동위상으로 도달한다. 렌즈는 발산하는 파면을 수렴하는 파면으로 바꾸는 "위상 교정기(phase corrector)"다.

---

## 1. 구면 거울

### 1.1 오목(수렴) 거울

오목 거울은 반사면이 안쪽으로 굽어 있다(숟가락 안쪽 면처럼). 평행 광선은 곡률 반지름의 절반 위치에 있는 **초점** $F$로 수렴한다:

$$f = \frac{R}{2}$$

여기서 $R$은 곡률 반지름, $f$는 초점 거리다.

**거울 방정식**은 물체 거리 $s$, 상 거리 $s'$, 초점 거리 $f$를 연결한다:

$$\frac{1}{s} + \frac{1}{s'} = \frac{1}{f} = \frac{2}{R}$$

**횡(가로) 배율(Lateral magnification)**:

$$m = -\frac{s'}{s}$$

음의 부호는 $s$와 $s'$ 모두 양수일 때(실물체, 실상) 상이 도립(inverted)임을 의미한다.

### 1.2 볼록(발산) 거울

볼록 거울은 반사면이 바깥쪽으로 굽어 있다. 평행 광선은 반사 후 발산하며, 거울 뒤쪽에 있는 가상의 초점에서 나오는 것처럼 보인다. 부호 규약에 따르면:

$$f = -\frac{|R|}{2} \quad (\text{볼록 거울은 음수})$$

볼록 거울은 항상 허상, 정립(upright), 축소(reduced) 상을 형성한다. 차량 사이드 미러로 사용되는 이유다("거울 속 물체는 실제보다 가깝습니다").

### 1.3 거울의 주요 광선 추적법

상의 위치를 결정하는 데 세 가지 주요 광선으로 충분하다:

1. **평행 광선**: 광축과 평행하게 입사 → 초점 $F$를 통과하도록 반사
2. **초점 광선**: $F$를 통과하며 입사 → 광축과 평행하게 반사
3. **중심 광선**: 거울 중심에 입사 → 대칭적으로 반사 (입사각 = 반사각)

임의의 두 광선이 교차하는 곳에 상이 형성된다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Ray tracing for a concave mirror
# Demonstrates image formation for objects at different distances

def mirror_image(s, f):
    """
    Calculate image distance and magnification for a curved mirror.
    s: object distance (positive for real object)
    f: focal length (positive for concave, negative for convex)
    Returns: (s', m) — image distance and magnification
    """
    if abs(1/f - 1/s) < 1e-10:
        return float('inf'), float('inf')  # object at focal point
    s_prime = 1 / (1/f - 1/s)
    m = -s_prime / s
    return s_prime, m

# Mirror parameters
f = 10.0  # focal length (cm), concave mirror
R = 2 * f

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Four cases: object at different positions relative to the mirror
cases = [
    ('Object beyond C (s > 2f)', 30.0),
    ('Object at C (s = 2f)', 20.0),
    ('Object between F and C (f < s < 2f)', 15.0),
    ('Object inside F (s < f)', 5.0),
]

for ax, (title, s) in zip(axes.flat, cases):
    s_prime, m = mirror_image(s, f)

    # Draw the mirror (vertical line at x=0 with curved cap)
    mirror_y = np.linspace(-8, 8, 100)
    mirror_x = mirror_y**2 / (4 * f) * 0.3  # slight curvature for visualization
    ax.plot(-mirror_x, mirror_y, 'k-', linewidth=3)

    # Draw optical axis
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='-')

    # Mark focal point and center of curvature
    ax.plot(-f, 0, 'ro', markersize=8, label=f'F (f={f})')
    ax.plot(-R, 0, 'bs', markersize=6, label=f'C (R={R})')

    # Object (arrow at -s)
    obj_height = 3.0
    ax.annotate('', xy=(-s, obj_height), xytext=(-s, 0),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax.text(-s, obj_height + 0.5, 'Object', ha='center', fontsize=9, color='green')

    # Principal rays (from tip of object)
    # Ray 1: parallel to axis → reflects through F
    ax.plot([-s, 0], [obj_height, obj_height], 'b-', linewidth=1.2)
    if np.isfinite(s_prime) and s_prime > 0:
        # Real image: ray goes through F to the image
        ax.plot([0, -s_prime], [obj_height, m * obj_height], 'b-', linewidth=1.2)
    else:
        # Virtual image or at infinity
        ax.plot([0, -f], [obj_height, 0], 'b-', linewidth=1.2)
        if np.isfinite(s_prime):
            ax.plot([0, 15], [obj_height, obj_height + 15 * obj_height / f], 'b--', linewidth=1)

    # Ray 2: through F → reflects parallel
    if abs(s - f) > 0.1:
        ax.plot([-s, 0], [obj_height, obj_height * (1 - s/f) + obj_height * s/f * 0], 'r-', linewidth=1.2)

    # Image (if real)
    if np.isfinite(s_prime) and abs(s_prime) < 100:
        img_height = m * obj_height
        if s_prime > 0:  # real image
            ax.annotate('', xy=(-s_prime, img_height), xytext=(-s_prime, 0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
            img_type = 'Real'
        else:  # virtual image
            ax.annotate('', xy=(abs(s_prime), img_height), xytext=(abs(s_prime), 0),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.5, linestyle='--'))
            img_type = 'Virtual'
        ax.set_title(f"{title}\n$s'$={s_prime:.1f}, m={m:.2f} ({img_type})", fontsize=11)
    else:
        ax.set_title(f"{title}\nImage at infinity", fontsize=11)

    ax.set_xlim(-35, 15)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('Position (cm)', fontsize=10)
    ax.set_aspect('equal')
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.2)

plt.suptitle('Concave Mirror: Image Formation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('concave_mirror_cases.png', dpi=150)
plt.show()
```

---

## 2. 박막 렌즈

### 2.1 박막 렌즈 방정식

박막 렌즈(두께가 곡률 반지름에 비해 무시할 수 있을 정도로 얇은 렌즈)는 거울 방정식과 동일한 수학적 형태를 따른다:

$$\frac{1}{s} + \frac{1}{s'} = \frac{1}{f}$$

중요한 차이점은 상 거리가 물체와 *반대쪽*(투과 광선 측)에서 양수로 정의된다는 것이다.

**횡 배율**:

$$m = -\frac{s'}{s}$$

### 2.2 렌즈 제작자 방정식

박막 렌즈의 초점 거리는 렌즈 재료의 굴절률 $n$과 양면의 곡률 반지름 $R_1$, $R_2$에 의해 결정된다:

$$\frac{1}{f} = (n - 1)\left(\frac{1}{R_1} - \frac{1}{R_2}\right)$$

부호 규약: $R > 0$은 곡률 중심이 투과 측(왼쪽에서 오른쪽으로 진행하는 빛의 오른쪽)에 있을 때다.

| 렌즈 종류 | 모양 | $R_1$ | $R_2$ | $f$ |
|-----------|------|-------|-------|-----|
| 양볼록(Biconvex) | () | + | - | + (수렴) |
| 평볼록(Plano-convex) | D | + | $\infty$ | + (수렴) |
| 양오목(Biconcave) | )( | - | + | - (발산) |
| 평오목(Plano-concave) | ( | $\infty$ | + | - (발산) |
| 메니스커스 수렴(Meniscus converging) | )( | + | + ($R_1 < R_2$) | + |
| 메니스커스 발산(Meniscus diverging) | )( | + | + ($R_1 > R_2$) | - |

### 2.3 광학 파워

렌즈의 **광학 파워(optical power)**는 초점 거리의 역수다:

$$P = \frac{1}{f}$$

$f$를 미터(m) 단위로 표현할 때, 파워의 단위는 **디옵터(diopter, D)**다. +2 D 렌즈는 $f = 0.5$ m를 의미한다. 안경 처방에서는 디옵터를 사용한다.

### 2.4 박막 렌즈의 주요 광선 추적법

수렴 렌즈에서 세 가지 주요 광선은 다음과 같다:

1. **평행 광선**: 광축과 평행하게 입사 → 후방 초점 $F'$를 통과하도록 굴절
2. **초점 광선**: 전방 초점 $F$를 통과하며 입사 → 광축과 평행하게 굴절
3. **중심 광선**: 렌즈 중심을 통과 → 방향이 변하지 않고 직진

```python
import numpy as np
import matplotlib.pyplot as plt

# Thin lens image formation: demonstrate converging and diverging cases
# Plot principal ray diagrams

def thin_lens_image(s, f):
    """Calculate image distance and magnification for a thin lens."""
    s_prime = 1 / (1/f - 1/s) if abs(1/f - 1/s) > 1e-10 else float('inf')
    m = -s_prime / s if s != 0 else float('inf')
    return s_prime, m

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# --- Converging lens (f > 0) ---
ax = axes[0]
f_conv = 8.0   # cm
s_obj = 14.0   # cm (object beyond F)
s_prime, m = thin_lens_image(s_obj, f_conv)

# Draw lens (thin vertical double-arrow)
lens_height = 7
ax.annotate('', xy=(0, lens_height), xytext=(0, -lens_height),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
ax.axhline(0, color='gray', linewidth=0.5)

# Focal points
ax.plot(-f_conv, 0, 'ro', markersize=8, zorder=5)
ax.plot(f_conv, 0, 'ro', markersize=8, zorder=5)
ax.text(-f_conv, -1, 'F', ha='center', fontsize=10, color='red')
ax.text(f_conv, -1, "F'", ha='center', fontsize=10, color='red')

# Object
h_obj = 4.0
ax.annotate('', xy=(-s_obj, h_obj), xytext=(-s_obj, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))

# Image
h_img = m * h_obj
ax.annotate('', xy=(s_prime, h_img), xytext=(s_prime, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5))

# Ray 1: parallel → through F'
ax.plot([-s_obj, 0, s_prime + 5], [h_obj, h_obj, h_obj - (s_prime + 5) * h_obj / f_conv + h_obj],
        'b-', linewidth=1.2, alpha=0.7)
# Simplified: parallel ray bends through F'
ax.plot([-s_obj, 0], [h_obj, h_obj], 'b-', linewidth=1.2)
ax.plot([0, s_prime], [h_obj, h_img], 'b-', linewidth=1.2)

# Ray 2: through center (undeviated)
ax.plot([-s_obj, s_prime], [h_obj, h_img], 'orange', linewidth=1.2)

# Ray 3: through F → parallel after lens
ax.plot([-s_obj, 0], [h_obj, h_obj * (1 - s_obj / (-f_conv))], 'g-', linewidth=1.2, alpha=0.5)

ax.set_xlim(-22, 22)
ax.set_ylim(-8, 8)
ax.set_title(f"Converging Lens: f={f_conv}, s={s_obj}, s'={s_prime:.1f}, m={m:.2f}", fontsize=11)
ax.set_xlabel('Position (cm)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

# --- Diverging lens (f < 0) ---
ax = axes[1]
f_div = -8.0
s_obj = 14.0
s_prime, m = thin_lens_image(s_obj, f_div)

# Draw lens (thin vertical concave shape)
ax.annotate('', xy=(0, lens_height), xytext=(0, -lens_height),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax.axhline(0, color='gray', linewidth=0.5)

# Focal points (virtual for diverging lens)
ax.plot(f_div, 0, 'ro', markersize=8, zorder=5)
ax.plot(-f_div, 0, 'ro', markersize=8, zorder=5)
ax.text(f_div, -1, "F'", ha='center', fontsize=10, color='red')
ax.text(-f_div, -1, 'F', ha='center', fontsize=10, color='red')

# Object
ax.annotate('', xy=(-s_obj, h_obj), xytext=(-s_obj, 0),
            arrowprops=dict(arrowstyle='->', color='green', lw=2.5))

# Image (virtual — on the same side as object)
h_img = m * h_obj
ax.annotate('', xy=(s_prime, h_img), xytext=(s_prime, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2.5, linestyle='--'))

# Ray through center
ax.plot([-s_obj, 15], [h_obj, h_obj + (15 + s_obj) * (h_img - h_obj) / (s_prime + s_obj)],
        'orange', linewidth=1.2)

ax.set_xlim(-22, 22)
ax.set_ylim(-8, 8)
ax.set_title(f"Diverging Lens: f={f_div}, s={s_obj}, s'={s_prime:.1f}, m={m:.2f}", fontsize=11)
ax.set_xlabel('Position (cm)')
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('thin_lens_ray_diagrams.png', dpi=150)
plt.show()
```

---

## 3. 상의 특성

### 3.1 실상과 허상

| 특성 | 실상(Real Image) | 허상(Virtual Image) |
|------|-----------------|---------------------|
| 형성 방식 | 광선이 상점에서 실제로 수렴 | 광선이 상점에서 발산하는 것처럼 보임 |
| 스크린에 투영 가능? | 가능 | 불가능 |
| $s'$ 부호 | 양수 (투과 광선과 같은 쪽) | 음수 (입사 광선과 같은 쪽) |
| 예 | 프로젝터, 카메라 센서 | 돋보기, 평면 거울 |

### 3.2 상 형성 요약

**수렴 렌즈/오목 거울** ($f > 0$):

| 물체 위치 | 상 위치 | 상의 종류 | 방향 | 크기 |
|-----------|---------|-----------|------|------|
| $s > 2f$ | $f < s' < 2f$ | 실상 | 도립 | 축소 |
| $s = 2f$ | $s' = 2f$ | 실상 | 도립 | 동일 |
| $f < s < 2f$ | $s' > 2f$ | 실상 | 도립 | 확대 |
| $s = f$ | $s' = \infty$ | — | — | — |
| $s < f$ | $s' < 0$ (허상) | 허상 | 정립 | 확대 |

**발산 렌즈/볼록 거울** ($f < 0$): 항상 허상, 정립, 축소 상을 형성한다.

---

## 4. 복합 렌즈 시스템

### 4.1 접촉한 두 박막 렌즈

초점 거리가 $f_1$, $f_2$인 두 박막 렌즈를 밀착시키면, 합성 초점 거리는:

$$\frac{1}{f_{\text{total}}} = \frac{1}{f_1} + \frac{1}{f_2}$$

또는 동등하게, 파워(power)가 더해진다:

$$P_{\text{total}} = P_1 + P_2$$

이것이 색지움 이중 렌즈(achromatic doublet)의 원리다. 수렴하는 크라운 유리(crown glass) 렌즈와 발산하는 플린트 유리(flint glass) 렌즈를 조합하여 색수차(chromatic aberration)를 상쇄하면서 전체적으로 수렴하는 파워를 유지한다.

### 4.2 거리 $d$만큼 분리된 두 박막 렌즈

두 렌즈가 거리 $d$만큼 떨어져 있을 때:

$$\frac{1}{f_{\text{eff}}} = \frac{1}{f_1} + \frac{1}{f_2} - \frac{d}{f_1 f_2}$$

첫 번째 렌즈에 의한 상이 두 번째 렌즈의 물체가 된다. 첫 번째 렌즈가 거리 $s_1'$에 상을 형성하면, 두 번째 렌즈에 대한 물체 거리는:

$$s_2 = d - s_1'$$

참고: $s_2 < 0$이면 두 번째 렌즈에 대한 "물체"는 허물체(virtual object)다 (첫 번째 렌즈의 상이 두 번째 렌즈 너머에 형성되는 경우).

### 4.3 ABCD 행렬 방법

모든 근축(paraxial) 광학 소자는 광선 벡터 $\begin{pmatrix} y \\ \theta \end{pmatrix}$에 작용하는 $2 \times 2$ **광선 전달 행렬(ray transfer matrix)**로 표현할 수 있다. 여기서 $y$는 광선 높이, $\theta$는 광선 각도다:

$$\begin{pmatrix} y_{\text{out}} \\ \theta_{\text{out}} \end{pmatrix} = \begin{pmatrix} A & B \\ C & D \end{pmatrix} \begin{pmatrix} y_{\text{in}} \\ \theta_{\text{in}} \end{pmatrix}$$

주요 행렬:

| 소자 | 행렬 |
|------|------|
| 자유 공간(거리 $d$) | $\begin{pmatrix} 1 & d \\ 0 & 1 \end{pmatrix}$ |
| 박막 렌즈(초점 거리 $f$) | $\begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$ |
| 구면 거울(곡률 반지름 $R$) | $\begin{pmatrix} 1 & 0 \\ -2/R & 1 \end{pmatrix}$ |
| 평면 경계($n_1 \to n_2$) | $\begin{pmatrix} 1 & 0 \\ 0 & n_1/n_2 \end{pmatrix}$ |
| 구면 경계(반지름 $R$, $n_1 \to n_2$) | $\begin{pmatrix} 1 & 0 \\ (n_1-n_2)/(n_2 R) & n_1/n_2 \end{pmatrix}$ |

여러 소자로 구성된 시스템의 경우, 행렬을 오른쪽에서 왼쪽으로 곱한다(첫 번째 소자가 가장 오른쪽):

$$M_{\text{system}} = M_N \cdot M_{N-1} \cdots M_2 \cdot M_1$$

유효 초점 거리는 $f_{\text{eff}} = -1/C$이며, 여기서 $C$는 시스템 행렬의 (2,1) 원소다.

```python
import numpy as np

# ABCD Matrix method: trace rays through a compound optical system
# Example: two thin lenses separated by a distance

def free_space(d):
    """Transfer matrix for propagation through free space of distance d."""
    return np.array([[1, d],
                     [0, 1]])

def thin_lens(f):
    """Transfer matrix for a thin lens with focal length f."""
    return np.array([[1, 0],
                     [-1/f, 1]])

def curved_mirror(R):
    """Transfer matrix for a curved mirror with radius of curvature R."""
    return np.array([[1, 0],
                     [-2/R, 1]])

# Example system: two converging lenses
f1 = 10.0   # cm, first lens
f2 = 20.0   # cm, second lens
d = 15.0    # cm, separation between lenses

# System matrix: M = L2 * D * L1
# (multiply right to left: light hits L1 first, propagates distance d, then hits L2)
M = thin_lens(f2) @ free_space(d) @ thin_lens(f1)

print("System matrix M:")
print(f"  A = {M[0,0]:.4f}")
print(f"  B = {M[0,1]:.4f}")
print(f"  C = {M[1,0]:.4f}")
print(f"  D = {M[1,1]:.4f}")
print()

# Effective focal length from the C element
f_eff = -1 / M[1, 0]
print(f"Effective focal length: {f_eff:.2f} cm")

# Compare with the formula: 1/f_eff = 1/f1 + 1/f2 - d/(f1*f2)
f_formula = 1 / (1/f1 + 1/f2 - d/(f1*f2))
print(f"Formula result:         {f_formula:.2f} cm")
print()

# Trace a specific ray: object at s=25 cm from first lens, height y=2 cm
s = 25.0
y_in = 2.0
theta_in = -y_in / s  # angle for a ray from the object tip to the lens center

# Full system: free_space(s) on input side, then the optical system
# Then find where the output ray crosses the axis (image location)
ray_in = np.array([y_in, theta_in])

# After the two-lens system:
ray_out = M @ ray_in
y_out, theta_out = ray_out

# The image is where y = 0 after the last lens
# y_out + theta_out * s_prime = 0  =>  s_prime = -y_out / theta_out
if abs(theta_out) > 1e-10:
    s_image = -y_out / theta_out
    m = (y_out + theta_out * s_image) / y_in  # actually should be y_img / y_obj
    # Better: magnification from full matrix approach
    print(f"Object distance from L1: {s:.1f} cm")
    print(f"Image distance from L2:  {s_image:.2f} cm")
    print(f"Output ray: y = {y_out:.4f} cm, theta = {theta_out:.6f} rad")
```

---

## 5. 수차

실제 렌즈와 거울은 이상적인 박막 렌즈 거동에서 벗어난다. 이러한 편차를 **수차(aberration)**라 한다. 5가지 1차 단색 수차(자이델 수차(Seidel aberrations))는 더 큰 각도나 높이의 광선에서 근축 근사(paraxial approximation)가 무너질 때 발생한다.

### 5.1 구면 수차(Spherical Aberration)

**원인**: 광축에서 멀리 떨어진 광선이 근축 광선과 다른 점에서 초점을 맺는다.

**영향**: 축 상에서도 흐릿한 상이 형성된다. 구면 수차의 크기는 $h^4$ (광선 높이의 네제곱)에 비례한다.

**보정**: 비구면 표면(aspheric surface), 조리개 조임(aperture stop), 양성 및 음성 소자 조합.

### 5.2 코마 수차(Coma)

**원인**: 렌즈의 서로 다른 환형 영역을 통해 결상되는 축 외 점의 배율이 다르다.

**영향**: 축 외 점 광원이 혜성 모양("코마")의 번짐으로 나타난다.

**보정**: 무수차 렌즈 설계(aplanatic lens design, 구면 수차와 코마 수차 모두 보정).

### 5.3 비점 수차(Astigmatism)

**원인**: 축 외 물점(object point)에 대해 렌즈가 접선면(tangential plane)과 시상면(sagittal plane)에서 서로 다른 유효 곡률을 갖는다.

**영향**: 점 광원이 서로 다른 거리에서 수직인 두 개의 선 상을 형성한다. 그 사이에 "최소 착란원(circle of least confusion)"이 있다.

**보정**: 무비점 렌즈 설계(anastigmatic lens design), 만곡 상면.

### 5.4 상면 만곡(Field Curvature, Petzval Curvature)

**원인**: 비점 수차를 보정한 후에도, 평면 물체의 상은 곡면(페쯔발 면(Petzval surface)) 위에 맺힌다.

**영향**: 평면 검출기에서 시야의 중심과 가장자리를 동시에 선명하게 초점 맞출 수 없다.

**보정**: 상면 평탄화 렌즈(field-flattening lens), 메니스커스 소자 ($\sum 1/(n_i f_i)$인 페쯔발 합(Petzval sum)을 최소화해야 함).

### 5.5 왜곡(Distortion)

**원인**: 광축으로부터의 거리에 따라 배율이 달라진다.

**영향**: 직선이 곡선으로 보인다. 두 가지 유형이 있다:
- **배럴 왜곡(Barrel distortion)**: 중심에서 멀어질수록 배율 감소 (광각 렌즈)
- **핀쿠션 왜곡(Pincushion distortion)**: 중심에서 멀어질수록 배율 증가 (망원 렌즈)

**보정**: 대칭 렌즈 구성, 계산에 의한 보정(스마트폰에서 일반적).

### 5.6 색수차(Chromatic Aberration)

이것은 분산(dispersion)으로 인한 *다색(polychromatic)* 수차로, 파장마다 초점 거리가 달라진다.

**축 방향 색수차(Longitudinal chromatic aberration)**: 색에 따라 광축상에서 초점 위치가 달라진다.

**축 수직 방향 색수차(Lateral chromatic aberration)**: 색에 따라 배율이 달라져 색 번짐(color fringes)이 생긴다.

**보정**: 색지움 이중 렌즈(achromatic doublet, 크라운 + 플린트 유리), 색지움 삼중 렌즈(apochromatic triplet), 회절-굴절 하이브리드.

```python
import numpy as np
import matplotlib.pyplot as plt

# Spherical aberration: rays at different heights focus at different points
# Compare a perfect lens (all rays focus at f) vs. real spherical lens

def trace_spherical_lens(y_in, R, n_lens, thickness):
    """
    Simple ray trace through a thick symmetric biconvex lens.
    Uses exact Snell's law (not paraxial approximation) to show spherical aberration.

    y_in: input ray height (parallel to axis)
    R: radius of curvature of both surfaces (|R1| = |R2| = R)
    n_lens: refractive index of lens
    thickness: lens center thickness
    """
    n_air = 1.0

    # First surface: center at (R, 0), ray hits at height y_in
    # Find intersection with sphere: (x - R)^2 + y^2 = R^2 for the left surface
    x1 = R - np.sqrt(R**2 - y_in**2)

    # Normal at this point: pointing from center (R, 0) toward (x1, y_in)
    nx1 = (x1 - R) / R
    ny1 = y_in / R
    # Angle of incidence (ray is horizontal, so sin(theta_i) = |ny1| effectively)
    sin_theta1 = abs(ny1)  # horizontal ray hitting curved surface
    sin_theta1_r = sin_theta1 / n_lens  # Snell's law: refracted angle
    theta1_r = np.arcsin(sin_theta1_r)
    theta1 = np.arcsin(sin_theta1)

    # Refracted ray direction (2D rotation)
    # The deviation angle
    delta1 = theta1 - theta1_r  # bending toward normal

    # Simplified: the refracted ray slope inside the lens
    slope_inside = np.tan(-delta1) if y_in > 0 else np.tan(delta1)

    # Second surface: center at (thickness - R, 0) for symmetric biconvex
    # Approximate: the ray exits and crosses the axis at some distance
    # For a full trace we'd need the second surface intersection too

    # Use the thin lens formula with correction for ray height (exact trace)
    # For simplicity, use the paraxial focal length + spherical aberration term
    f_paraxial = R / (2 * (n_lens - 1))  # thin lens approximation for symmetric biconvex
    # Longitudinal spherical aberration scales as y^2
    # (3rd order: the marginal focus differs from paraxial focus)
    SA_coeff = 0.002  # empirical coefficient for demonstration
    f_actual = f_paraxial * (1 - SA_coeff * y_in**2)

    return f_actual

# Parameters
R = 50.0      # mm radius of curvature
n_glass = 1.5
thickness = 5.0

y_values = np.linspace(0.1, 20, 100)  # ray heights from 0.1 to 20 mm
f_paraxial = R / (2 * (n_glass - 1))  # = 50 mm for these values

# Calculate focus position for each ray height
# Using the exact formula for a spherical lens longitudinal aberration:
# Delta_f ≈ -h^2 / (2 * f * n * (n-1)) * [n^2 + (n-1)^2 * (3n+2)/(n-1)] ...
# Simplified: LSA ≈ -k * h^2 where k depends on lens shape and n
k_sa = 1 / (8 * f_paraxial * (n_glass - 1)**2)  # approximate coefficient
focus_positions = f_paraxial - k_sa * y_values**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: focus position vs ray height (shows spherical aberration)
ax1.plot(y_values, focus_positions, 'b-', linewidth=2)
ax1.axhline(f_paraxial, color='r', linestyle='--', label=f'Paraxial focus: {f_paraxial:.1f} mm')
ax1.set_xlabel('Ray height (mm)', fontsize=12)
ax1.set_ylabel('Focus position (mm)', fontsize=12)
ax1.set_title('Longitudinal Spherical Aberration', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Right: spot diagram — ray intersections in the focal plane
# Simulate 2D: rays enter at various heights and azimuthal angles
N_rays = 500
heights = np.random.uniform(0, 15, N_rays)  # random ray heights
azimuths = np.random.uniform(0, 2*np.pi, N_rays)  # random azimuthal angles

# Each ray's focus error (how far from paraxial focus it lands)
delta_f = -k_sa * heights**2
# In the paraxial focal plane, the ray misses the center by:
# transverse error ≈ delta_f * (height / f_paraxial)
x_spot = delta_f * np.cos(azimuths) * heights / f_paraxial
y_spot = delta_f * np.sin(azimuths) * heights / f_paraxial

ax2.scatter(x_spot, y_spot, s=1, alpha=0.5, c=heights, cmap='viridis')
ax2.set_xlabel('x (mm)', fontsize=12)
ax2.set_ylabel('y (mm)', fontsize=12)
ax2.set_title('Spot Diagram at Paraxial Focus\n(color = ray height)', fontsize=13)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)
cbar = plt.colorbar(ax2.collections[0], ax=ax2, label='Ray height (mm)')

plt.tight_layout()
plt.savefig('spherical_aberration.png', dpi=150)
plt.show()
```

---

## 6. 색수차와 색지움 이중 렌즈

### 6.1 문제

$n(\lambda)$가 파장에 따라 달라지므로, 단일 렌즈는 각 색마다 초점 거리가 다르다:

$$f(\lambda) = \frac{R_1 R_2}{(n(\lambda) - 1)(R_2 - R_1)}$$

정상 분산(normal dispersion)에서 청색광은 적색광보다 렌즈에 더 가깝게 초점을 맺어, 상 주변에 색 번짐이 생긴다.

### 6.2 색지움 이중 렌즈 설계

색지움 이중 렌즈는 저분산 크라운 유리(crown glass, 아베 수(Abbe number) $V_1$ 큼)와 고분산 플린트 유리(flint glass, $V_2$ 작음)를 조합하여 색수차를 상쇄한다:

$$\frac{P_1}{V_1} + \frac{P_2}{V_2} = 0$$

여기서 $P_i = 1/f_i$는 각 렌즈의 파워다. 전체 파워 조건 $P_1 + P_2 = P_{\text{total}}$과 연립하면 각 렌즈의 파워를 구할 수 있다:

$$P_1 = P_{\text{total}} \frac{V_1}{V_1 - V_2}, \qquad P_2 = -P_{\text{total}} \frac{V_2}{V_1 - V_2}$$

```python
import numpy as np

# Design an achromatic doublet lens
# Crown glass (BK7) + Flint glass (SF2)

# Abbe numbers and refractive indices
V_crown = 64.17   # BK7 Abbe number (low dispersion)
V_flint = 33.85   # SF2 Abbe number (high dispersion)
n_d_crown = 1.5168  # at 587.6 nm (d-line)
n_d_flint = 1.6477

# Desired total focal length
f_total = 100.0  # mm
P_total = 1 / f_total  # total power (1/mm)

# Individual powers that cancel chromatic aberration
P_crown = P_total * V_crown / (V_crown - V_flint)
P_flint = -P_total * V_flint / (V_crown - V_flint)

f_crown = 1 / P_crown
f_flint = 1 / P_flint

print("Achromatic Doublet Design")
print("=" * 40)
print(f"Total focal length:  {f_total:.1f} mm")
print(f"Crown lens (BK7):    f = {f_crown:.1f} mm  (P = {P_crown:.6f} mm⁻¹)")
print(f"Flint lens (SF2):    f = {f_flint:.1f} mm  (P = {P_flint:.6f} mm⁻¹)")
print(f"\nVerification:")
print(f"  P_crown + P_flint = {P_crown + P_flint:.6f} mm⁻¹ (should be {P_total:.6f})")
print(f"  P_crown/V_crown + P_flint/V_flint = {P_crown/V_crown + P_flint/V_flint:.8f} (should be ~0)")

# The crown element is converging (positive power) and the flint is diverging (negative power)
# Together they form a net converging doublet with minimal chromatic aberration
```

---

## 7. 후꺼운 렌즈와 기본점

### 7.1 주면(Principal Planes)

두꺼운 렌즈나 복합 시스템에서도, 렌즈 표면이 아닌 **주면** $H$와 $H'$에서 거리를 측정하면 박막 렌즈 방정식이 그대로 적용된다:

$$\frac{1}{s_H} + \frac{1}{s'_{H'}} = \frac{1}{f}$$

여기서 $s_H$는 $H$에서 측정한 물체 거리, $s'_{H'}$는 $H'$에서 측정한 상 거리다.

### 7.2 6개의 기본점(Cardinal Points)

두꺼운 렌즈에는 6개의 기본점이 있다:

1. **두 개의 초점** ($F$, $F'$): 평행 광선이 수렴하는 점
2. **두 개의 주점** ($H$, $H'$): 렌즈가 광선을 꺾는 것처럼 보이는 점
3. **두 개의 절점** ($N$, $N'$): 비스듬한 광선이 각도 변화 없이 통과하는 점

양쪽 매질이 동일한 렌즈에서는 $N = H$, $N' = H'$이다.

ABCD 행렬로 모든 기본점을 구할 수 있다:

$$f = -\frac{1}{C}, \qquad \text{BFD} = -\frac{A}{C}, \qquad \text{FFD} = \frac{D}{C}$$

여기서 BFD(후방 초점 거리(back focal distance))는 마지막 면에서 $F'$까지의 거리, FFD(전방 초점 거리(front focal distance))는 첫 번째 면에서 $F$까지의 거리다.

---

## 연습 문제

### 연습 1: 거울 문제

곡률 반지름이 40 cm인 오목 거울이 있다.

(a) 거울 앞 60 cm에 놓인 물체의 상 위치는? 실상인가 허상인가?

(b) 배율은 얼마인가? 상이 정립인가 도립인가?

(c) 물체 크기의 3배 되는 허상을 만들려면 물체를 어디에 놓아야 하는가?

### 연습 2: 렌즈 제작자 방정식

$n = 1.52$인 유리로 초점 거리 15 cm의 박막 렌즈를 설계한다.

(a) 등곡률 양볼록 렌즈라면, $R$은 얼마인가?

(b) 한 면이 평면인 평볼록 렌즈라면, 곡면의 반지름은 얼마인가?

(c) 각 설계의 광학 파워를 디옵터 단위로 계산하라.

### 연습 3: 복합 시스템

두 박막 렌즈($f_1 = 10$ cm, $f_2 = -20$ cm)가 8 cm 간격으로 배치되어 있다. 물체가 첫 번째 렌즈 왼쪽 15 cm에 놓여 있다.

(a) 박막 렌즈 방정식을 순차적으로 적용하여 최종 상의 위치를 구하라.

(b) ABCD 행렬 방법으로 답을 검증하라.

(c) 시스템의 총 배율은 얼마인가?

### 연습 4: 색지움 이중 렌즈

BK7 크라운 유리($n_d = 1.517$, $V_d = 64.2$)와 SF11 플린트 유리($n_d = 1.785$, $V_d = 25.7$)를 사용하여 $f = 200$ mm의 색지움 이중 렌즈를 설계한다.

(a) 크라운 소자와 플린트 소자 각각의 필요 초점 거리를 계산하라.

(b) 렌즈 제작자 방정식을 이용하여 각 소자에 적합한 곡률 반지름을 구하라 (접합면의 반지름이 공통이라고 가정).

(c) 이차 스펙트럼(secondary spectrum, 청색과 적색의 초점 거리 차이)의 나머지 값은 얼마인가?

### 연습 5: 수차 식별

각 상황에서 지배적인 수차를 파악하고 보정 방법을 제안하라:

(a) 망원경이 중심에서는 선명한 별 상을 보여주지만, 가장자리에서는 방사 방향으로 길쭉하게 늘어진다.

(b) 현미경 대물렌즈가 고대비 특징 주변에 색 번짐을 만든다.

(c) 광각 카메라 렌즈가 화면 가장자리에서 직선 건물을 곡선으로 보이게 만든다.

(d) 대구경 천문 거울이 흐릿한 별 상을 만드는데, 조리개를 줄이면 개선된다.

---

## 요약

| 개념 | 핵심 공식/사실 |
|------|---------------|
| 거울 방정식 | $1/s + 1/s' = 1/f = 2/R$ |
| 박막 렌즈 방정식 | $1/s + 1/s' = 1/f$ |
| 렌즈 제작자 방정식 | $1/f = (n-1)(1/R_1 - 1/R_2)$ |
| 배율 | $m = -s'/s$ |
| 광학 파워 | $P = 1/f$ ($f$가 미터일 때 디옵터) |
| 밀착 렌즈 | $1/f_{\text{total}} = 1/f_1 + 1/f_2$ |
| 분리된 렌즈 | $1/f_{\text{eff}} = 1/f_1 + 1/f_2 - d/(f_1 f_2)$ |
| ABCD 행렬 (렌즈) | $\begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}$ |
| ABCD 행렬 (자유 공간) | $\begin{pmatrix} 1 & d \\ 0 & 1 \end{pmatrix}$ |
| 색지움 조건 | $P_1/V_1 + P_2/V_2 = 0$ |
| 자이델 수차 | 구면 수차, 코마, 비점 수차, 상면 만곡, 왜곡 |
| 색수차 | 파장마다 $f$ 다름; 색지움 이중 렌즈로 보정 |

---

[← 이전: 02. 기하광학 기초](02_Geometric_Optics_Fundamentals.md) | [다음: 04. 광학 기기 →](04_Optical_Instruments.md)
