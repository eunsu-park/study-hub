# 자기 벡터 퍼텐셜

[← 이전: 04. 정자기학](04_Magnetostatics.md) | [다음: 06. 전자기 유도 →](06_Electromagnetic_Induction.md)

---

## 학습 목표

1. 자기 벡터 퍼텐셜(magnetic vector potential) $\mathbf{A}$를 정의하고, $\mathbf{B} = \nabla \times \mathbf{A}$가 성립함을 보인다
2. 게이지 자유도(gauge freedom)와 게이지 선택의 물리적 의미를 설명한다
3. 쿨롱 게이지(Coulomb gauge)를 적용하여 정자기 계산을 단순화한다
4. 표준 전류 분포(직선 도선, 솔레노이드)의 벡터 퍼텐셜을 계산한다
5. 아하로노프-봄 효과(Aharonov-Bohm effect)와 퍼텐셜의 실재성에 대한 함의를 설명한다
6. 벡터 퍼텐셜의 다중극 전개(multipole expansion)를 전개한다
7. 수치적으로 벡터 퍼텐셜을 계산하고 $\mathbf{B} = \nabla \times \mathbf{A}$를 검증한다

---

정전기학에서 스칼라 퍼텐셜 $V$는 계산의 편의 도구였다 — $\mathbf{E}$가 벡터인 데 반해 $V$는 스칼라이므로 계산이 더 쉬웠다. 자기 벡터 퍼텐셜 $\mathbf{A}$는 정자기학에서 유사한 역할을 하지만, 그 자체로 벡터량이다. 하지만 $\mathbf{A}$는 단순한 수학적 트릭에 그치지 않는다. 양자역학은 $\mathbf{A}$가 직접적인 물리적 의미를 가짐을 보여준다 — 아하로노프-봄 효과에서 퍼텐셜은 $\mathbf{B} = 0$인 영역에서도 하전 입자에 영향을 미친다. 이 레슨에서는 벡터 퍼텐셜 형식론을 전개하고, 게이지 자유도를 탐구하며, $\mathbf{A}$가 자기장의 물리를 어떻게 담아내는지 살펴본다. 그 과정에서 물리학에서 가장 아름다운 결과 중 하나인 아하로노프-봄 효과를 만나게 된다 — 퍼텐셜이 단순한 수학적 편의 도구가 아니라 그 자체로 물리적 실체임을 보여주는 현상이다.

---

## 왜 벡터 퍼텐셜인가?

$\nabla \cdot \mathbf{B} = 0$ (자기 홀극 없음)이므로, 발산 정리에 의해 $\mathbf{B}$는 반드시 어떤 벡터장의 컬(curl)로 표현될 수 있다:

$$\boxed{\mathbf{B} = \nabla \times \mathbf{A}}$$

이는 벡터 해석학의 정리에 의해 보장된다: 발산이 없는 벡터장은 항상 어떤 벡터의 컬로 표현된다.

정전기학과 비교하면:
- $\nabla \times \mathbf{E} = 0 \implies \mathbf{E} = -\nabla V$ (컬이 없음 $\implies$ 스칼라의 기울기)
- $\nabla \cdot \mathbf{B} = 0 \implies \mathbf{B} = \nabla \times \mathbf{A}$ (발산이 없음 $\implies$ 벡터의 컬)

이 결과들의 수학적 토대는 **헬름홀츠 분해(Helmholtz decomposition)**(임의의 벡터장은 컬이 없는 부분과 발산이 없는 부분으로 분해할 수 있다)와 **푸앵카레 보조정리(Poincare lemma)**(닫힌 형식은 국소적으로 완전하다)이다.

### A에 관한 방정식

$\mathbf{B} = \nabla \times \mathbf{A}$를 앙페르 법칙 $\nabla \times \mathbf{B} = \mu_0 \mathbf{J}$에 대입하면:

$$\nabla \times (\nabla \times \mathbf{A}) = \mu_0 \mathbf{J}$$

벡터 항등식 $\nabla \times (\nabla \times \mathbf{A}) = \nabla(\nabla \cdot \mathbf{A}) - \nabla^2 \mathbf{A}$를 이용하면:

$$\nabla(\nabla \cdot \mathbf{A}) - \nabla^2 \mathbf{A} = \mu_0 \mathbf{J}$$

이는 우리가 원하는 것보다 훨씬 복잡한 형태이지만, 여기서 게이지 자유도가 구원자로 등장한다.

---

## 게이지 자유도

$\mathbf{A}$가 올바른 $\mathbf{B}$를 주면, 임의의 스칼라 함수 $\lambda$에 대해 $\mathbf{A}' = \mathbf{A} + \nabla \lambda$도 마찬가지다. 왜냐하면:

$$\nabla \times \mathbf{A}' = \nabla \times \mathbf{A} + \nabla \times (\nabla \lambda) = \nabla \times \mathbf{A} = \mathbf{B}$$

(기울기의 컬은 항상 영이기 때문이다.)

이는 $\mathbf{A}$가 유일하지 않음을 의미한다 — $\mathbf{A}$는 임의의 스칼라 기울기 항만큼 자유도를 가진다. 이 모호성을 **게이지 자유도(gauge freedom)**라 하고, 특정한 $\lambda$를 선택하는 것을 **게이지 고정(gauge fixing)**이라 한다.

> **유추**: 게이지 자유도는 고도의 영점을 선택하는 것과 같다. 해수면을 기준으로 하든 주방 바닥을 기준으로 하든, 고도의 차이(물리적으로 의미 있는 양)는 동일하다. 절대적인 "퍼텐셜 높이"는 모호하지만, 상대적인 높이차($\mathbf{B} = \nabla \times \mathbf{A}$에 해당하는 것)는 명확하다.

### 쿨롱 게이지

정자기학에서 가장 일반적인 선택은 **쿨롱 게이지(Coulomb gauge)**이다:

$$\nabla \cdot \mathbf{A} = 0$$

이 선택으로 $\mathbf{A}$에 관한 방정식이 아름답게 단순화된다:

$$-\nabla^2 \mathbf{A} = \mu_0 \mathbf{J}$$

이는 포아송 방정식(Poisson's equation)의 세 묶음이다 — $\mathbf{A}$의 각 성분에 하나씩! 이 방정식의 해는 ($V = \frac{1}{4\pi\epsilon_0}\int \frac{\rho}{r}\,d\tau'$와의 유추에 의해):

$$\boxed{\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi} \int \frac{\mathbf{J}(\mathbf{r}')}{|\mathbf{r} - \mathbf{r}'|} \, d\tau'}$$

선전류(line current)의 경우:

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0 I}{4\pi} \int \frac{d\mathbf{l}'}{|\mathbf{r} - \mathbf{r}'|}$$

비오-사바르 법칙(Biot-Savart)과 비교하면:
- 비오-사바르: 외적과 $1/r^2$ — 계산이 어려움
- 벡터 퍼텐셜 공식: 외적 없이 $1/r$ — 적분이 쉬움

---

## 무한 직선 도선의 벡터 퍼텐셜

z축을 따라 전류 $I$가 흐르는 무한 직선 도선의 경우:

$$\mathbf{A} = -\frac{\mu_0 I}{2\pi} \ln(s/s_0) \, \hat{z}$$

여기서 $s$는 도선으로부터의 수직 거리이고, $s_0$는 기준 거리(퍼텐셜의 임의 상수에 해당)이다.

검증: $\mathbf{B} = \nabla \times \mathbf{A}$. $\mathbf{A} = A_z(s)\hat{z}$로 놓고 원통 좌표계에서:

$$\nabla \times \mathbf{A} = -\frac{\partial A_z}{\partial s}\hat{\phi} = \frac{\mu_0 I}{2\pi s}\hat{\phi}$$

이는 무한 직선 도선의 올바른 자기장이다.

### 원형 루프의 벡터 퍼텐셜 (축 위에서)

반지름 $R$인 원형 루프에 전류 $I$가 흐를 때, 축 위($\rho = 0$, 임의의 $z$)에서의 벡터 퍼텐셜은 대칭성에 의해 사라진다: $\mathbf{A}_{\text{axis}} = 0$. 루프 양쪽의 기여가 상쇄되기 때문이다.

그러나 축에서 약간 벗어나면 벡터 퍼텐셜은 0이 아니며 $\phi$ 성분만 가진다. $\rho \ll R$ (축 근방)일 때:

$$A_\phi \approx \frac{\mu_0 I R^2 \rho}{4(R^2 + z^2)^{3/2}}$$

축에서 멀리 벗어난 완전한 표현식은 타원 적분(elliptic integral)을 포함한다 — 퍼텐셜을 해석적보다 수치적으로 계산하는 것이 더 쉬운 상황 중 하나이다.

### A와 자기 선속의 관계

벡터 퍼텐셜을 자기 선속(magnetic flux)과 연결하는 아름다운 등식이 있다:

$$\Phi_B = \int_S \mathbf{B} \cdot d\mathbf{a} = \int_S (\nabla \times \mathbf{A}) \cdot d\mathbf{a} = \oint_C \mathbf{A} \cdot d\mathbf{l}$$

임의의 곡면을 통과하는 자기 선속은 그 경계에 대한 $\mathbf{A}$의 선적분과 같다. 이는 벡터 퍼텐셜에 적용된 스토크스 정리(Stokes' theorem)이며, 선속을 알 때 $\mathbf{A}$를 계산하는 실용적인 방법을 제공한다(아래 솔레노이드 예시 참고).

---

## 솔레노이드의 벡터 퍼텐셜

반지름 $R$, 단위 길이당 $n$회 감긴 무한 솔레노이드에 전류 $I$가 흐를 때, 내부에서 $\mathbf{B} = \mu_0 n I \hat{z}$이고 외부에서 $\mathbf{B} = 0$이다.

대칭성에 의해 $\mathbf{A} = A_\phi(s)\hat{\phi}$ (z축 주위를 순환)이다. $\oint \mathbf{A} \cdot d\mathbf{l} = \int \mathbf{B} \cdot d\mathbf{a}$를 이용하여 구한다 (스토크스 정리에 의해 $\int (\nabla \times \mathbf{A}) \cdot d\mathbf{a} = \oint \mathbf{A} \cdot d\mathbf{l}$):

**내부** ($s < R$):
$$A_\phi (2\pi s) = \mu_0 n I (\pi s^2) \implies A_\phi = \frac{\mu_0 n I}{2} s$$

**외부** ($s > R$):
$$A_\phi (2\pi s) = \mu_0 n I (\pi R^2) \implies A_\phi = \frac{\mu_0 n I R^2}{2s}$$

주목할 점: 솔레노이드 외부에서는 $\mathbf{B} = 0$이지만 $\mathbf{A} \neq 0$이다! 자기장이 없는 곳에서도 벡터 퍼텐셜은 존재한다. 이 놀라운 사실 — 장이 없는 곳에서도 퍼텐셜이 존재할 수 있다는 것 — 은 아하로노프-봄 효과를 통해 심오한 물리적 결과를 낳는다(아래에서 논의).

솔레노이드 외부에서 $A_\phi \propto 1/s$ 형태는 자기 홀극 끈(magnetic monopole string)의 벡터 퍼텐셜과 동일하다 — 이론 물리학의 다양한 맥락에서 나타나는 위상학적 특성이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Vector potential of a solenoid: A_φ as a function of distance from axis
# Why important: A ≠ 0 outside the solenoid even though B = 0 there

mu_0 = 4 * np.pi * 1e-7
n = 1000     # turns per meter
I = 1.0      # current (A)
R = 0.05     # solenoid radius (5 cm)

s = np.linspace(0.001, 0.15, 500)

# A_φ: linear inside, 1/s outside
A_phi = np.where(
    s < R,
    mu_0 * n * I * s / 2,           # inside: grows linearly
    mu_0 * n * I * R**2 / (2 * s)   # outside: falls as 1/s
)

# B_z: uniform inside, zero outside
B_z = np.where(s < R, mu_0 * n * I, 0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vector potential A_φ
axes[0].plot(s * 100, A_phi * 1e6, 'b-', linewidth=2)
axes[0].axvline(x=R*100, color='gray', linestyle='--', label=f'R = {R*100:.0f} cm')
axes[0].set_xlabel('s (cm)')
axes[0].set_ylabel(r'$A_\phi$ ($\mu$T·m)')
axes[0].set_title(r'Vector Potential $A_\phi(s)$')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].annotate('A ≠ 0 here!\n(but B = 0)', xy=(R*100 + 2, A_phi[300]*1e6),
                fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))

# Magnetic field B_z
axes[1].plot(s * 100, B_z * 1e3, 'r-', linewidth=2)
axes[1].axvline(x=R*100, color='gray', linestyle='--', label=f'R = {R*100:.0f} cm')
axes[1].set_xlabel('s (cm)')
axes[1].set_ylabel('$B_z$ (mT)')
axes[1].set_title(r'Magnetic Field $B_z(s)$')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Solenoid: A is nonzero where B is zero!', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('solenoid_vector_potential.png', dpi=150)
plt.show()

# Verify B = curl A numerically
# In cylindrical coordinates with A = A_φ(s) φ̂:
# B_z = (1/s) d(sA_φ)/ds
ds = s[1] - s[0]
sA = s * A_phi
# Why numerical derivative: to verify B = curl A
B_numerical = np.gradient(sA, ds) / s

print("Verification: B_z from curl(A)")
print(f"  Analytic B inside:  {mu_0 * n * I * 1e3:.4f} mT")
idx_inside = len(s) // 5   # point well inside
print(f"  Numerical B inside: {B_numerical[idx_inside] * 1e3:.4f} mT")
idx_outside = 4 * len(s) // 5  # point well outside
print(f"  Numerical B outside: {B_numerical[idx_outside] * 1e3:.6f} mT (should be ~0)")
```

---

## 아하로노프-봄 효과

아하로노프-봄 효과(Aharonov-Bohm effect, 1959)는 양자역학에서 가장 심오한 결과 중 하나이다. $\mathbf{B} = 0$인 영역에서도 벡터 퍼텐셜 $\mathbf{A}$가 직접적인 물리적 결과를 낳음을 보여준다.

### 실험 설정

전자 빔이 두 경로로 분리되어 솔레노이드 양쪽을 지나 다시 합쳐진다. 솔레노이드는 완전히 차폐되어 있어 전자는 어떤 자기장도 경험하지 않는다(솔레노이드 외부에서 $\mathbf{B} = 0$).

### 관측 결과

전자가 지나는 어느 곳에서나 $\mathbf{B} = 0$임에도 불구하고, 솔레노이드의 전류가 바뀌면 간섭 무늬가 이동한다. 두 경로 사이의 위상차는:

$$\Delta\phi = \frac{e}{\hbar}\oint \mathbf{A} \cdot d\mathbf{l} = \frac{e}{\hbar}\Phi_B$$

여기서 $\Phi_B = \int \mathbf{B} \cdot d\mathbf{a}$는 솔레노이드를 통과하는 자기 선속이다.

### 의의

- 고전 물리학에서 $\mathbf{A}$는 단순한 수학적 편의 도구에 불과하며, "실재"하는 것은 $\mathbf{B}$뿐이다
- 양자역학에서 $\mathbf{A}$는 전자 파동함수의 위상에 직접 영향을 미친다
- 이는 **퍼텐셜이 장(field)보다 더 근본적임**을 시사한다
- 이 효과는 실험적으로 확인되었다 (토노무라 외, 1986: 외부에서 $\mathbf{B} = 0$을 보장하기 위해 초전도 차폐물로 덮은 토로이달 자석 사용)

### 베리 위상과의 연결

아하로노프-봄 위상은 **기하학적 위상(geometric phase)**, 즉 베리 위상(Berry phase)의 예이다 — 해밀토니안의 매개변수가 닫힌 경로를 따라 순환할 때 양자 상태가 획득하는 위상이다. AB 위상은 국소적인 장의 값이 아니라, 경로의 위상학적 성질(솔레노이드를 감싸는지 여부)에 의존한다. 이 위상학적 특성 때문에 어떤 국소 장 효과로도 설명할 수 없으며, 본질적으로 비국소적이다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Aharonov-Bohm effect: interference pattern shift with enclosed flux
# Why simulate: visualizing the fringe shift makes the abstract effect concrete

hbar = 1.055e-34      # reduced Planck constant (J·s)
e = 1.6e-19            # electron charge (C)
lambda_dB = 1e-10      # de Broglie wavelength (1 Å, typical for electrons)
k = 2 * np.pi / lambda_dB

# Screen position (angular coordinate)
theta = np.linspace(-0.01, 0.01, 1000)  # small angles (radians)

# Baseline: path length difference → standard double-slit pattern
d_slit = 1e-6     # slit separation (1 μm)
delta_path = d_slit * np.sin(theta)  # path length difference
# Why k*delta_path: the phase difference from geometry alone
phase_geom = k * delta_path

# Aharonov-Bohm phase: additional phase from vector potential
# Φ_B = magnetic flux through solenoid (in units of flux quantum Φ₀ = h/e)
phi_0 = 2 * np.pi * hbar / e  # flux quantum

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for idx, flux_ratio in enumerate([0, 0.25, 0.5, 1.0]):
    ax = axes[idx // 2, idx % 2]

    # AB phase shift
    phase_AB = 2 * np.pi * flux_ratio  # in radians

    # Interference: I ∝ cos²((δ_geom + δ_AB)/2)
    # Why cosine squared: this is two-beam interference
    I_pattern = np.cos(0.5 * (phase_geom + phase_AB))**2

    ax.plot(theta * 1e3, I_pattern, 'b-', linewidth=1.5)
    ax.set_xlabel('θ (mrad)')
    ax.set_ylabel('Intensity (arb. units)')
    ax.set_title(f'Φ/Φ₀ = {flux_ratio:.2f}  (AB phase = {phase_AB/np.pi:.1f}π)')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)

    # Mark the central fringe position
    central_shift = -phase_AB / k / d_slit  # shift in angle
    ax.axvline(x=central_shift * 1e3, color='red', linestyle='--', alpha=0.7)

plt.suptitle('Aharonov-Bohm Effect: Interference Pattern vs. Enclosed Flux',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('aharonov_bohm.png', dpi=150)
plt.show()
```

---

## 벡터 퍼텐셜의 다중극 전개

정전기 퍼텐셜을 다중극(단극, 쌍극, 사중극, ...)으로 전개하듯, 벡터 퍼텐셜도 마찬가지로 전개할 수 있다.

국소화된 전류 분포에 대해, 멀리 떨어진 거리($r \gg r'$)에서 벡터 퍼텐셜은:

$$\mathbf{A}(\mathbf{r}) = \frac{\mu_0}{4\pi} \frac{1}{r} \int \mathbf{J}(\mathbf{r}') \, d\tau' + \frac{\mu_0}{4\pi} \frac{1}{r^2} \int (\hat{r} \cdot \mathbf{r}') \mathbf{J}(\mathbf{r}') \, d\tau' + \cdots$$

### 단극항(Monopole Term)

$$\mathbf{A}_{\text{mono}} = \frac{\mu_0}{4\pi r} \int \mathbf{J} \, d\tau' = 0$$

단극항은 정자기학에서 **항상 사라진다**. 이는 $\nabla \cdot \mathbf{B} = 0$과 깊이 연결되어 있다 — 자기 홀극이 없으므로, 퍼텐셜에 단극 기여가 없다.

전류 루프의 경우: $\int I \, d\mathbf{l}' = I \oint d\mathbf{l}' = 0$ (닫힌 루프에 대한 적분은 0이다).

이는 정전기학과 근본적으로 다르다. 정전기학에서는 단극항 $V_{\text{mono}} = Q/(4\pi\epsilon_0 r)$가 먼 거리에서 지배적인 기여를 한다. 자기 단극항의 부재는 쌍극항이 장거리에서 주도적인 기여임을 의미한다 — 이것이 자기장이 ($1/r^3$) 전기장($1/r^2$)보다 더 빠르게 감소하는 이유이다 (중성 전하 분포로부터).

### 쌍극항(Dipole Term)

$$\mathbf{A}_{\text{dip}} = \frac{\mu_0}{4\pi} \frac{\mathbf{m} \times \hat{r}}{r^2}$$

여기서 $\mathbf{m} = I \int d\mathbf{a}' = I\mathbf{A}$는 자기 쌍극자 모멘트(magnetic dipole moment)이다.

이것이 먼 거리에서 주도적인 항이다. 컬을 취하면:

$$\mathbf{B}_{\text{dip}} = \nabla \times \mathbf{A}_{\text{dip}} = \frac{\mu_0}{4\pi r^3}[3(\mathbf{m}\cdot\hat{r})\hat{r} - \mathbf{m}]$$

이는 레슨 4에서 이미 만난 결과이다.

### 고차 다중극

쌍극항 다음은 **자기 사중극(magnetic quadrupole)**이다. 그 벡터 퍼텐셜은 $1/r^3$으로, 장은 $1/r^4$로 감소한다. 사중극 모멘트 텐서는 2차 텐서로, 전류 분포 기하학의 다음 단계 세부 정보를 담는다.

원자·핵 물리학에서 자기 쌍극자 모멘트는 놀라운 정밀도로 측정된다 — 전자의 이상 자기 모멘트는 $10^{12}$분의 1 수준에서 양자 전기역학(QED) 예측과 일치한다. 더 높은 자기 다중극 모멘트는 핵의 내부 구조를 드러낸다: 핵 자기 사중극 모멘트가 0이 아니라는 것은, 예를 들어 해당 핵이 구형 대칭에서 벗어나 있음을 의미한다.

### 비교: 전기 다중극 vs. 자기 다중극

| 차수 | 전기 퍼텐셜 | 자기 퍼텐셜 | 사라지는가? |
|---|---|---|---|
| 단극 ($l=0$) | $\sim Q/r$ | 항상 0 | 예 (항상, 홀극 없음) |
| 쌍극 ($l=1$) | $\sim p\cos\theta/r^2$ | $\sim m\sin\theta/r^2$ | $\mathbf{p}=0$ 또는 $\mathbf{m}=0$인 경우만 |
| 사중극 ($l=2$) | $\sim 1/r^3$ | $\sim 1/r^3$ | 기하학적 구조에 따라 다름 |

> **유추**: 다중극 전개는 복잡한 형태를 멀리서 묘사하는 것과 같다. 먼 거리에서 보면 어떤 전류 루프도 자기 쌍극자처럼 보인다 — 마치 어떤 전하 분포도 점전하처럼 보이듯이. 고차 다중극은 가까이에서만 중요한 세밀한 구조를 담는다.

---

## 게이지 변환 심화

$\mathbf{A} \to \mathbf{A} + \nabla\lambda$ 변환의 자유는 단순한 불편함이 아니라 전자기학의 심오한 대칭성이다.

### 다양한 게이지

| 게이지 | 조건 | 최적 용도 |
|---|---|---|
| 쿨롱 게이지(Coulomb gauge) | $\nabla \cdot \mathbf{A} = 0$ | 정자기학, 복사 |
| 로렌츠 게이지(Lorenz gauge) | $\nabla \cdot \mathbf{A} = -\mu_0\epsilon_0 \frac{\partial V}{\partial t}$ | 상대론적 문제 |
| 축 게이지(Axial gauge) | $A_z = 0$ | 일부 격자 계산 |
| 시간 게이지(Temporal gauge) | $V = 0$ | 양자장론 |
| 바일 게이지(Weyl gauge) | $V = 0$ | 정준 양자화 |

### 게이지 불변성

물리적 관측량(힘, 에너지, 간섭 무늬)은 반드시 게이지 불변(gauge-invariant)이어야 한다. 예를 들어:
- $\mathbf{B} = \nabla \times \mathbf{A}$는 게이지 불변이다 (기울기의 컬은 0이므로)
- 닫힌 루프에 대한 AB 위상 $\oint \mathbf{A} \cdot d\mathbf{l}$은 게이지 불변이다 (닫힌 경로에서 $\nabla\lambda$를 적분하면 스토크스 정리에 의해 0이 되므로)
- 자기 선속 $\Phi = \oint \mathbf{A}\cdot d\mathbf{l}$은 게이지 불변이다 (같은 이유)
- $\mathbf{E}$와 $\mathbf{B}$만으로 유도된 양(에너지 밀도, 포인팅 벡터, 힘)은 자동으로 게이지 불변이다

### 게이지 자유도가 중요한 이유

한 번 게이지를 고정하면 되지 않겠느냐고 물을 수 있다. 그렇지 않은 이유는 실용적이다:

1. **게이지마다 다른 문제를 단순화한다.** 쿨롱 게이지는 정자기학에, 로렌츠 게이지는 복사 문제에, 축 게이지는 특정 격자 계산에 이상적이다.
2. **게이지 불변성이 이론을 제약한다.** 새로운 이론을 구성할 때, 결과가 게이지 불변이어야 한다는 요구는 강력한 검증 수단이 된다 — 게이지 선택에 의존하는 결과는 반드시 잘못된 것이다.
3. **게이지 대칭성은 이론을 생성한다.** 양자장론에서 게이지 불변성을 근본 원리로 삼으면 전자기역학 전체가 처음부터 유도된다.

게이지 불변성의 요구는 현대 물리학에서 가장 강력한 원리 중 하나로, 입자물리학의 표준 모형(Standard Model)의 토대를 이룬다.

### 고전에서 양자 게이지 이론으로

양자역학에서 게이지 변환은 파동함수로 확장된다:

$$\psi \to \psi' = \psi \, e^{iq\lambda/\hbar}$$

국소 게이지 변환($\lambda$가 위치와 시간에 의존)에 대해 물리가 불변이어야 한다는 요구 — 이것이 양자장론에서 전자기학 자체를 만들어내는 원리이다. 자유 입자에서 시작하여 국소 게이지 불변성을 요구하면, 게이지장 $A_\mu$를 반드시 도입해야 하고 — 이것이 바로 전자기 사중극 퍼텐셜이다! 이것이 전자기학이 존재하는 심오한 이유이다.

같은 논리를 더 복잡한 대칭군에 적용하면 약력과 강한 핵력이 유도된다 — 표준 모형 전체가 게이지 불변성으로부터 나온다.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate gauge freedom: two different A fields giving the same B
# Why: seeing gauge invariance in action builds confidence in the formalism

# Setup: uniform B = B_0 z_hat in a region
B_0 = 1e-3   # 1 mT

# Grid
x = np.linspace(-1, 1, 20)
y = np.linspace(-1, 1, 20)
X, Y = np.meshgrid(x, y)

# Gauge 1: symmetric gauge A = (B₀/2)(-y x̂ + x ŷ)
# Why symmetric: treats x and y equivalently
Ax_1 = -B_0 * Y / 2
Ay_1 = B_0 * X / 2

# Gauge 2: Landau gauge A = B₀ x ŷ  (all in the y-component)
# This is related to gauge 1 by λ = B₀xy/2
Ax_2 = np.zeros_like(X)
Ay_2 = B_0 * X

# Both should give the same B_z = ∂Ay/∂x - ∂Ax/∂y
dx = x[1] - x[0]
dy = y[1] - y[0]

# Numerical curl for gauge 1
dAy1_dx = np.gradient(Ay_1, dx, axis=1)
dAx1_dy = np.gradient(Ax_1, dy, axis=0)
Bz_1 = dAy1_dx - dAx1_dy

# Numerical curl for gauge 2
dAy2_dx = np.gradient(Ay_2, dx, axis=1)
dAx2_dy = np.gradient(Ax_2, dy, axis=0)
Bz_2 = dAy2_dx - dAx2_dy

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Gauge 1: symmetric gauge
axes[0].quiver(X, Y, Ax_1, Ay_1, color='blue', alpha=0.7)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title('Gauge 1 (Symmetric): A = B₀/2 (-y, x, 0)')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# Gauge 2: Landau gauge
axes[1].quiver(X, Y, Ax_2, Ay_2, color='red', alpha=0.7)
axes[1].set_xlabel('x')
axes[1].set_ylabel('y')
axes[1].set_title('Gauge 2 (Landau): A = B₀(0, x, 0)')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# Both give the same B
axes[2].plot(x, Bz_1[10, :] * 1e3, 'bo-', label='Gauge 1', markersize=4)
axes[2].plot(x, Bz_2[10, :] * 1e3, 'r^-', label='Gauge 2', markersize=4)
axes[2].axhline(y=B_0 * 1e3, color='green', linestyle='--', label=f'B₀ = {B_0*1e3} mT')
axes[2].set_xlabel('x')
axes[2].set_ylabel('B_z (mT)')
axes[2].set_title('Same B from Different A')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle('Gauge Freedom: Different A, Same B', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('gauge_freedom.png', dpi=150)
plt.show()

print(f"B_z from Gauge 1: mean = {np.mean(Bz_1)*1e3:.4f} mT, std = {np.std(Bz_1)*1e3:.6f} mT")
print(f"B_z from Gauge 2: mean = {np.mean(Bz_2)*1e3:.4f} mT, std = {np.std(Bz_2)*1e3:.6f} mT")
print(f"Max difference:    {np.max(np.abs(Bz_1 - Bz_2))*1e3:.6f} mT")
```

---

## 요약

| 개념 | 핵심 방정식 |
|---|---|
| 벡터 퍼텐셜 | $\mathbf{B} = \nabla \times \mathbf{A}$ |
| 게이지 자유도 | $\mathbf{A}' = \mathbf{A} + \nabla\lambda$는 같은 $\mathbf{B}$를 준다 |
| 쿨롱 게이지 | $\nabla \cdot \mathbf{A} = 0$ |
| 포아송 형태 | $\nabla^2 \mathbf{A} = -\mu_0\mathbf{J}$ (쿨롱 게이지에서) |
| 해(solution) | $\mathbf{A} = \frac{\mu_0}{4\pi}\int \frac{\mathbf{J}}{|\mathbf{r}-\mathbf{r}'|}\,d\tau'$ |
| 무한 직선 도선 | $A_z = -\frac{\mu_0 I}{2\pi}\ln(s/s_0)$ |
| 솔레노이드 (내부) | $A_\phi = \frac{\mu_0 n I}{2}s$ |
| 솔레노이드 (외부) | $A_\phi = \frac{\mu_0 n I R^2}{2s}$ |
| 단극항 | $\mathbf{A}_{\text{mono}} = 0$ (항상) |
| 쌍극항 | $\mathbf{A}_{\text{dip}} = \frac{\mu_0}{4\pi}\frac{\mathbf{m}\times\hat{r}}{r^2}$ |
| AB 위상 | $\Delta\phi = \frac{e}{\hbar}\oint\mathbf{A}\cdot d\mathbf{l}$ |

---

## 연습 문제

### 연습 문제 1: 유한 솔레노이드의 벡터 퍼텐셜
길이 $L$, 반지름 $R$인 유한 솔레노이드의 $\mathbf{A}$를 $N$개의 원형 전류 루프로 모델링하여 수치적으로 계산하라. 내부와 외부에서 무한 솔레노이드 공식과 비교하라. 어디서 근사가 무너지는가?

### 연습 문제 2: 쿨롱 게이지 검증
비오-사바르 유사 공식으로 수치적으로 계산한 원형 전류 루프의 벡터 퍼텐셜에 대해, 여러 점에서 $\nabla \cdot \mathbf{A} = 0$이 성립하는지 검증하라.

### 연습 문제 3: 게이지 변환
대칭 게이지 $\mathbf{A}_1 = \frac{B_0}{2}(-y\hat{x} + x\hat{y})$에서 시작하여, 이를 란다우 게이지(Landau gauge) $\mathbf{A}_2 = B_0 x\hat{y}$로 변환하는 게이지 함수 $\lambda$를 구하라. 답을 검증하라.

### 연습 문제 4: 자기 사중극
반대 방향 전류가 흐르는 두 원형 루프(자기 사중극)의 벡터 퍼텐셜을 계산하라. $\mathbf{A}$가 거리에 따라 어떻게 감소하는가? 쌍극자의 경우와 비교하라.

### 연습 문제 5: AB 효과와 선속 양자화
초전도체에서 루프를 통과하는 자기 선속은 $\Phi_0 = h/(2e)$의 정수배로 양자화된다 (인자 2는 초전도 운반자가 쿠퍼 쌍(Cooper pair)이기 때문). SI 단위로 $\Phi_0$를 계산하라. 넓이 $1\,\text{mm}^2$인 초전도 링에서 단일 선속 양자에 해당하는 최대 $\mathbf{B}$는 얼마인가? 선속 양자 $\Phi_0 \approx 2.07 \times 10^{-15}$ Wb는 초전도 양자 소자(SQUID)에서 중심적 역할을 하는 기본 상수이며, 자기장 측정의 감도 한계를 정의한다.

### 연습 문제 6: 전류 시트의 수치적 벡터 퍼텐셜
$xy$-평면 위의 무한 전류 시트가 면전류 밀도(surface current density) $\mathbf{K} = K_0\hat{x}$를 가진다 ($x$-방향으로 단위 길이당 전류). 벡터 퍼텐셜은 $\mathbf{A} = -\frac{\mu_0 K_0}{2}|z|\hat{x}$이다. $\nabla \times \mathbf{A}$를 계산하여 이것이 올바른 $\mathbf{B}$ (시트 위아래에서 반대 방향을 가리키는 균일한 자기장)를 주는지 확인하라.

---

[← 이전: 04. 정자기학](04_Magnetostatics.md) | [다음: 06. 전자기 유도 →](06_Electromagnetic_Induction.md)
