# 14. 상대론적 전자기학

[← 이전: 13. 복사와 안테나](13_Radiation_and_Antennas.md) | [다음: 15. 다중극 전개 →](15_Multipole_Expansion.md)

## 학습 목표

1. 전기장과 자기장에 로렌츠 변환(Lorentz transformation)을 적용하고, 두 장이 섞이는 현상을 이해한다
2. 전자기 장 텐서(field tensor) $F^{\mu\nu}$와 그 쌍대(dual)를 구성한다
3. 4-벡터와 텐서를 이용하여 맥스웰 방정식을 명시적으로 공변(covariant)한 형태로 쓴다
4. 공변 언어에서 4-퍼텐셜 $A^\mu$, 4-전류 $J^\mu$, 게이지 불변성(gauge invariance)을 이해한다
5. 공변 형식으로부터 로렌츠 힘 법칙을 유도한다
6. 자기가 전기정역학의 상대론적 보정으로 자연스럽게 나타나는 방식을 이해한다
7. 전자기학의 라그랑지안 형식화(Lagrangian formulation)와 응력-에너지 텐서(stress-energy tensor)를 파악한다

전기와 자기는 두 가지 별개의 힘이 아니다 — 이들은 단일 실체인 전자기장(electromagnetic field)의 서로 다른 단면으로, 그 모습은 관측자의 기준계에 따라 달라진다. 특수 상대성이론은 이 통일성을 놀라운 우아함으로 드러낸다. 한 기준계에서 순수한 전기장은 운동하는 기준계에서 보면 자기 성분을 얻는다. 전자기 장 텐서 $F^{\mu\nu}$는 $\mathbf{E}$와 $\mathbf{B}$의 6개 성분 모두를 하나의 기하학적 대상으로 묶어내며, 맥스웰의 네 방정식은 단 두 개의 텐서 방정식으로 압축된다. 이 레슨은 상대론이 전자기학에 선택적으로 추가되는 것이 아니라 그 구조 깊숙이 내재되어 있음을 보여준다.

> **유추**: 깃대의 그림자를 상상해 보자. 남쪽에서 보면 그림자는 북쪽을 향하며 일정한 길이를 가진다. 동쪽으로 걸어가면 그림자는 이제 북서쪽을 향하고 길이도 달라 보인다. 그림자 자체는 변하지 않았다 — 단지 관점이 바뀌었을 뿐이다. 마찬가지로 $\mathbf{E}$와 $\mathbf{B}$는 전자기 장 텐서가 특정 기준계에 투영된 "그림자"와 같다. 서로 다른 관측자는 서로 다른 $\mathbf{E}$와 $\mathbf{B}$를 보지만, 그 밑에 있는 $F^{\mu\nu}$는 동일한 기하학적 대상이다.

---

## 1. 로렌츠 변환 복습

### 1.1 시공간 좌표

특수 상대성이론에서 사건(event)은 4-위치(4-position)로 기술된다:

$$x^\mu = (ct, x, y, z) = (x^0, x^1, x^2, x^3)$$

계량(metric) 부호는 $(+, -, -, -)$를 사용한다 (입자물리학 관례). 민코프스키 계량(Minkowski metric)은:

$$\eta_{\mu\nu} = \text{diag}(+1, -1, -1, -1)$$

### 1.2 로렌츠 부스트

속도 $v$로 $x$축 방향으로의 부스트(boost)에 대해:

$$\Lambda^\mu_{\ \nu} = \begin{pmatrix} \gamma & -\gamma\beta & 0 & 0 \\ -\gamma\beta & \gamma & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}$$

여기서 $\beta = v/c$이고 $\gamma = 1/\sqrt{1-\beta^2}$이다.

### 1.3 4-벡터

**4-벡터(4-vector)** $A^\mu$는 $A'^\mu = \Lambda^\mu_{\ \nu} A^\nu$로 변환된다. 주요 예:

- **4-속도(4-velocity)**: $u^\mu = \gamma(c, \mathbf{v})$
- **4-운동량(4-momentum)**: $p^\mu = m u^\mu = (\gamma mc, \gamma m\mathbf{v}) = (E/c, \mathbf{p})$
- **4-전류 밀도(4-current density)**: $J^\mu = (c\rho, \mathbf{J})$
- **4-퍼텐셜(4-potential)**: $A^\mu = (V/c, \mathbf{A})$

---

## 2. 전기장과 자기장의 변환

### 2.1 장 변환 규칙

$\mathbf{v} = v\hat{x}$로 로렌츠 부스트를 할 때, 장은 다음과 같이 변환된다:

**부스트 방향($x$축)에 평행한 성분:**

$$E'_x = E_x, \quad B'_x = B_x$$

**부스트 방향에 수직인 성분:**

$$\boxed{E'_y = \gamma(E_y - vB_z), \quad E'_z = \gamma(E_z + vB_y)}$$

$$\boxed{B'_y = \gamma\left(B_y + \frac{v}{c^2}E_z\right), \quad B'_z = \gamma\left(B_z - \frac{v}{c^2}E_y\right)}$$

일반적인 부스트 $\boldsymbol{\beta} = \mathbf{v}/c$에 대한 간결한 표기:

$$\mathbf{E}' = \gamma(\mathbf{E} + \boldsymbol{\beta} \times c\mathbf{B}) - (\gamma - 1)(\mathbf{E} \cdot \hat{\beta})\hat{\beta}$$

$$\mathbf{B}' = \gamma\left(\mathbf{B} - \frac{\boldsymbol{\beta}}{c} \times \mathbf{E}\right) - (\gamma - 1)(\mathbf{B} \cdot \hat{\beta})\hat{\beta}$$

### 2.2 핵심 결론

1. **순수한 전기장은 운동하는 기준계에서 자기 성분을 얻는다** (그 반대도 마찬가지)
2. 한 기준계에서 $\mathbf{E} = 0$이면, $\mathbf{E}' = \gamma(\boldsymbol{\beta} \times c\mathbf{B})$ — 자기장을 통과하여 운동하면 전기장이 생성된다
3. 두 **로렌츠 불변량(Lorentz invariant)**은 모든 기준계에서 보존된다:

$$\boxed{\mathbf{E} \cdot \mathbf{B} = \text{불변}, \quad E^2 - c^2 B^2 = \text{불변}}$$

이 불변량들은 장의 성격을 결정한다: $E^2 > c^2 B^2$이면 $\mathbf{B} = 0$인 기준계가 존재하고; $E^2 < c^2 B^2$이면 $\mathbf{E} = 0$인 기준계가 존재한다.

```python
import numpy as np
import matplotlib.pyplot as plt

def lorentz_transform_fields(E, B, beta_vec):
    """
    Transform E and B fields under a Lorentz boost.

    Parameters:
        E        : electric field [Ex, Ey, Ez] (V/m)
        B        : magnetic field [Bx, By, Bz] (T)
        beta_vec : velocity/c [bx, by, bz]

    Returns:
        E', B' in the boosted frame

    Why this matters: the transformation reveals that E and B are not
    independent — they are components of a single entity (the field tensor)
    that mix under changes of reference frame.
    """
    c = 3e8
    E = np.array(E, dtype=float)
    B = np.array(B, dtype=float)
    beta = np.array(beta_vec, dtype=float)
    beta_mag = np.linalg.norm(beta)

    if beta_mag < 1e-15:
        return E.copy(), B.copy()

    gamma = 1.0 / np.sqrt(1 - beta_mag**2)
    beta_hat = beta / beta_mag

    # Parallel and perpendicular components
    E_par = np.dot(E, beta_hat) * beta_hat
    E_perp = E - E_par
    B_par = np.dot(B, beta_hat) * beta_hat
    B_perp = B - B_par

    # Transform
    E_prime = E_par + gamma * (E_perp + c * np.cross(beta, B))
    B_prime = B_par + gamma * (B_perp - np.cross(beta, E) / c)

    return E_prime, B_prime

# Example: Pure magnetic field in lab frame
B_lab = np.array([0, 0, 1.0])     # 1 T in z-direction
E_lab = np.array([0, 0, 0])       # no electric field

# Observe from frame moving at v = 0.5c in x-direction
beta = np.array([0.5, 0, 0])

E_prime, B_prime = lorentz_transform_fields(E_lab, B_lab, beta)

print("Lab frame:")
print(f"  E = {E_lab} V/m")
print(f"  B = {B_lab} T")
print(f"\nMoving frame (v = 0.5c in x-direction):")
print(f"  E' = [{E_prime[0]:.4e}, {E_prime[1]:.4e}, {E_prime[2]:.4e}] V/m")
print(f"  B' = [{B_prime[0]:.4f}, {B_prime[1]:.4f}, {B_prime[2]:.4f}] T")
print(f"\nA magnetic field in the lab becomes electric + magnetic in the moving frame!")
print(f"\nLorentz invariants:")
print(f"  E·B = {np.dot(E_lab, B_lab):.6f} (lab) = {np.dot(E_prime, B_prime):.6f} (moving)")
print(f"  E²-c²B² = {np.dot(E_lab,E_lab) - (3e8)**2 * np.dot(B_lab,B_lab):.4e} (lab)")
print(f"           = {np.dot(E_prime,E_prime) - (3e8)**2 * np.dot(B_prime,B_prime):.4e} (moving)")
```

---

## 3. 전자기 장 텐서

### 3.1 구성

장 텐서(field tensor) $F^{\mu\nu}$는 $\mathbf{E}$와 $\mathbf{B}$의 6개 성분을 모두 담는 반대칭 2-계 텐서이다:

$$F^{\mu\nu} = \begin{pmatrix} 0 & -E_x/c & -E_y/c & -E_z/c \\ E_x/c & 0 & -B_z & B_y \\ E_y/c & B_z & 0 & -B_x \\ E_z/c & -B_y & B_x & 0 \end{pmatrix}$$

텐서는 다음과 같이 변환된다:

$$F'^{\mu\nu} = \Lambda^\mu_{\ \alpha} \Lambda^\nu_{\ \beta} F^{\alpha\beta}$$

이것은 자동으로 2절의 장 변환 규칙을 재현한다.

### 3.2 쌍대 텐서

**쌍대 텐서(dual tensor)** $\tilde{F}^{\mu\nu}$는 $\mathbf{E} \to c\mathbf{B}$, $\mathbf{B} \to -\mathbf{E}/c$ 치환으로 얻는다:

$$\tilde{F}^{\mu\nu} = \frac{1}{2}\epsilon^{\mu\nu\alpha\beta}F_{\alpha\beta} = \begin{pmatrix} 0 & -B_x & -B_y & -B_z \\ B_x & 0 & E_z/c & -E_y/c \\ B_y & -E_z/c & 0 & E_x/c \\ B_z & E_y/c & -E_x/c & 0 \end{pmatrix}$$

여기서 $\epsilon^{\mu\nu\alpha\beta}$는 레비-치비타 기호(Levi-Civita symbol)이다.

### 3.3 텐서로부터의 로렌츠 불변량

두 로렌츠 불변량은 우아하게 다음으로 표현된다:

$$F_{\mu\nu}F^{\mu\nu} = 2\left(B^2 - \frac{E^2}{c^2}\right)$$

$$F_{\mu\nu}\tilde{F}^{\mu\nu} = -\frac{4}{c}\mathbf{E} \cdot \mathbf{B}$$

```python
def field_tensor(E, B, c=3e8):
    """
    Construct the electromagnetic field tensor F^{mu,nu}.

    Why the field tensor: it unifies E and B into a single geometric
    object that transforms naturally under Lorentz transformations.
    It makes manifest the relativistic structure of electrodynamics.
    """
    Ex, Ey, Ez = E
    Bx, By, Bz = B

    F = np.array([
        [0,      -Ex/c,  -Ey/c,  -Ez/c],
        [Ex/c,    0,     -Bz,     By   ],
        [Ey/c,    Bz,     0,     -Bx   ],
        [Ez/c,   -By,     Bx,     0    ]
    ])
    return F

def lorentz_boost_matrix(beta_x):
    """4x4 Lorentz boost matrix along x-axis."""
    gamma = 1.0 / np.sqrt(1 - beta_x**2)
    L = np.array([
        [gamma,        -gamma*beta_x, 0, 0],
        [-gamma*beta_x, gamma,         0, 0],
        [0,             0,             1, 0],
        [0,             0,             0, 1]
    ])
    return L

def transform_field_tensor(F, Lambda):
    """Transform F^{mu,nu} under Lorentz transformation."""
    return Lambda @ F @ Lambda.T

# Verify field transformation via the tensor method
E_lab = np.array([0, 0, 0])
B_lab = np.array([0, 0, 1.0])

F_lab = field_tensor(E_lab, B_lab)
Lambda = lorentz_boost_matrix(0.5)
F_prime = transform_field_tensor(F_lab, Lambda)

c = 3e8
print("Field tensor in lab frame:")
print(np.array2string(F_lab, precision=4, suppress_small=True))

print("\nField tensor in boosted frame:")
print(np.array2string(F_prime, precision=4, suppress_small=True))

# Extract E' and B' from F_prime
E_prime_tensor = np.array([F_prime[1,0], F_prime[2,0], F_prime[3,0]]) * c
B_prime_tensor = np.array([F_prime[3,2], F_prime[1,3], F_prime[2,1]])

print(f"\nExtracted fields from F':")
print(f"  E' = {E_prime_tensor}")
print(f"  B' = {B_prime_tensor}")

# Verify invariants
inv1_lab = np.trace(F_lab @ F_lab.T)  # not quite right; need F_{mu nu} F^{mu nu}
# Proper contraction with metric
eta = np.diag([1, -1, -1, -1])
F_lower_lab = eta @ F_lab @ eta
inv1_lab = np.sum(F_lower_lab * F_lab)
F_lower_prime = eta @ F_prime @ eta
inv1_prime = np.sum(F_lower_prime * F_prime)

print(f"\nInvariant F_{{μν}}F^{{μν}}: lab = {inv1_lab:.6f}, boosted = {inv1_prime:.6f}")
```

---

## 4. 공변 맥스웰 방정식

### 4.1 4-퍼텐셜과 4-전류

전자기 퍼텐셜은 4-벡터를 형성한다:

$$A^\mu = \left(\frac{V}{c}, \mathbf{A}\right)$$

장 텐서는 4-퍼텐셜의 회전(curl)이다:

$$F^{\mu\nu} = \partial^\mu A^\nu - \partial^\nu A^\mu$$

소스는 4-전류를 형성한다:

$$J^\mu = (c\rho, \mathbf{J})$$

연속 방정식 $\nabla \cdot \mathbf{J} + \partial\rho/\partial t = 0$은 다음이 된다:

$$\partial_\mu J^\mu = 0$$

### 4.2 공변 형태의 맥스웰 방정식

맥스웰의 네 방정식은 **단 두 개**의 텐서 방정식으로 환원된다:

**비균질 방정식** (가우스 법칙 + 앙페르-맥스웰):

$$\boxed{\partial_\mu F^{\mu\nu} = \mu_0 J^\nu}$$

이 단 하나의 방정식이 담고 있는 것:
- $\nu = 0$: $\nabla \cdot \mathbf{E} = \rho/\epsilon_0$ (가우스 법칙)
- $\nu = 1,2,3$: $\nabla \times \mathbf{B} = \mu_0\mathbf{J} + \mu_0\epsilon_0 \partial\mathbf{E}/\partial t$ (앙페르-맥스웰)

**균질 방정식** (자기 단극자 없음 + 패러데이):

$$\boxed{\partial_\mu \tilde{F}^{\mu\nu} = 0}$$

동등하게, 비앙키 항등식(Bianchi identity)을 이용하면:

$$\partial_\lambda F_{\mu\nu} + \partial_\mu F_{\nu\lambda} + \partial_\nu F_{\lambda\mu} = 0$$

이것이 담고 있는 것:
- $\nabla \cdot \mathbf{B} = 0$
- $\nabla \times \mathbf{E} = -\partial\mathbf{B}/\partial t$ (패러데이 법칙)

### 4.3 게이지 불변성

변환:

$$A^\mu \to A^\mu + \partial^\mu \chi$$

는 $F^{\mu\nu}$를 변화시키지 않는다 (반대칭 도함수가 대칭 부분을 소거하기 때문에). 이것이 **게이지 불변성(gauge invariance)** — 전자기학의 근저에 있는 심오한 대칭이다.

일반적인 게이지 선택:
- **로렌츠 게이지(Lorenz gauge)**: $\partial_\mu A^\mu = 0$ (명시적으로 공변)
- **쿨롱 게이지(Coulomb gauge)**: $\nabla \cdot \mathbf{A} = 0$ (공변하지 않지만 비상대론적 문제에서 편리)

---

## 5. 공변 형태의 로렌츠 힘

### 5.1 공변 운동 방정식

로렌츠 힘 법칙 $\mathbf{F} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$은 다음이 된다:

$$\frac{dp^\mu}{d\tau} = q F^{\mu\nu} u_\nu$$

여기서 $\tau$는 고유 시간(proper time)이고, $u_\nu = \eta_{\nu\alpha} u^\alpha$는 공변 4-속도이다.

이것은 명시적으로 공변하다: 좌변은 4-벡터(4-운동량의 고유 시간 미분)이고, 우변은 2-계 텐서와 4-벡터의 수축이므로 역시 4-벡터이다.

### 5.2 검증

공간 성분 ($\mu = 1, 2, 3$)은:

$$\frac{d(\gamma m\mathbf{v})}{dt} = q(\mathbf{E} + \mathbf{v} \times \mathbf{B})$$

를 주며, 이것이 상대론적 로렌츠 힘 법칙이다. 시간 성분 ($\mu = 0$)은 다음의 일률 방정식을 준다:

$$\frac{dE}{dt} = q\mathbf{v} \cdot \mathbf{E}$$

이것은 오직 전기장만이 전하에 일을 한다는 것을 의미한다 (자기력은 항상 속도에 수직이다).

---

## 6. 자기의 상대론적 기원

### 6.1 사고 실험

전류 $I$가 흐르는 긴 직선 도선과, 도선에 평행하게 속도 $v$로 (전도 전자의 표류 속도와 동일하게) 운동하는 전하 $q$를 생각하자. 실험실 기준계에서 도선은 전기적으로 중성이며, 전하는 오직 자기력만을 받는다.

이제 전하의 정지 기준계로 변환하자. 전하가 정지해 있으므로 자기력은 없을 수 없다. 그러나 도선 내 양전하와 음전하 밀도의 로렌츠 수축이 **서로 다르다** (실험실에서 서로 다른 속도를 갖기 때문에). 이것이 알짜 **전기장**을 만들어 정확히 동일한 힘을 제공한다.

### 6.2 정량적 유도

실험실 기준계에서 도선에는:
- 양전하 (이온): 선전하 밀도 $+\lambda$, 정지 상태
- 음전하 (전자): 선전하 밀도 $-\lambda$, 표류 속도 $v_d$

거리 $s$에서의 자기장은 $B = \mu_0 I / (2\pi s)$이며, 시험 전하에 작용하는 힘은:

$$F_{\text{mag}} = qvB = \frac{qv\mu_0 I}{2\pi s}$$

전하의 정지 기준계 (속도 $v$로 운동)에서, 로렌츠 수축은 전하 밀도를 서로 다르게 변형시킨다:

$$\lambda'_+ = \gamma_v \lambda, \quad \lambda'_- = -\frac{\gamma_v}{\gamma_d'}\lambda$$

여기서 $\gamma_d'$는 새 기준계에서의 변형된 전자 속도를 고려한다. 단위 길이당 알짜 전하가 전기장을 만들고, 이로부터의 전기력은 실험실 기준계에서의 자기력과 정확히 일치한다.

이것은 **자기가 전기정역학의 상대론적 효과**임을 보여준다 — 양전하 및 음전하 분포의 서로 다른 로렌츠 수축으로부터 발생하는 것이다.

```python
def magnetism_from_relativity():
    """
    Demonstrate that the magnetic force on a charge moving parallel
    to a current-carrying wire equals the electric force in the
    charge's rest frame due to relativistic length contraction.

    Why this matters: it's one of the most beautiful results in physics.
    The magnetic force — which seems like a fundamentally different
    phenomenon — is nothing but electrostatics viewed from a moving frame.
    """
    c = 3e8
    eps_0 = 8.854e-12
    mu_0 = 4 * np.pi * 1e-7

    # Parameters
    I = 10       # current (A)
    s = 0.01     # distance from wire (m)
    q = 1.6e-19  # test charge (C)

    # Drift velocity of electrons (typically very slow)
    v_d = 1e-4   # m/s (typical for copper wire)

    # Test charge velocity (same as electron drift for simplicity)
    v = v_d

    # Lab frame: magnetic force
    B = mu_0 * I / (2 * np.pi * s)
    F_mag = q * v * B
    print(f"Lab frame (magnetic force):")
    print(f"  B = {B:.6e} T")
    print(f"  F_mag = {F_mag:.6e} N")

    # Now let's show the equivalence for various test charge velocities
    v_test = np.linspace(0.001 * c, 0.9 * c, 100)
    gamma_v = 1.0 / np.sqrt(1 - (v_test / c)**2)

    # Magnetic force in lab frame
    F_magnetic = q * v_test * B

    # Approximate electric force in moving frame
    # The net charge density due to differential Lorentz contraction is:
    # delta_lambda ≈ lambda * v * v_d / c^2 (to first order in v_d/c)
    # This gives E ≈ lambda * v * v_d / (2*pi*eps_0*s*c^2)
    # And F_elec = qE = q*v*mu_0*I/(2*pi*s) = F_mag  (using mu_0 = 1/eps_0*c^2)

    # More precisely, using relativistic velocity addition:
    # lambda_net ≈ gamma_v * lambda * (v*v_d/c^2) for v_d << c
    lambda_line = I / v_d  # linear charge density (C/m)
    delta_lambda = gamma_v * lambda_line * v_test * v_d / c**2
    E_moving = delta_lambda / (2 * np.pi * eps_0 * s)
    F_electric = q * E_moving

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(v_test / c, F_magnetic, 'b-', linewidth=2,
            label='Magnetic force (lab frame)')
    ax.plot(v_test / c, F_electric, 'r--', linewidth=2,
            label='Electric force (moving frame)')
    ax.set_xlabel('Test charge velocity (v/c)')
    ax.set_ylabel('Force (N)')
    ax.set_title('Magnetism as a Relativistic Effect')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add annotation
    ax.text(0.5, F_magnetic[50] * 0.7,
            'The two forces are identical:\nmagnetic force = electric force\nfrom different frames!',
            fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig("magnetism_relativity.png", dpi=150)
    plt.show()

magnetism_from_relativity()
```

---

## 7. 라그랑지안 형식화 (개요)

### 7.1 전자기장 내 하전 입자의 라그랑지안

전자기장 내 하전 입자의 작용(action)은:

$$S = \int \left(-mc \, ds + \frac{q}{c} A_\mu \, dx^\mu\right)$$

여기서 $ds = c \, d\tau = \sqrt{c^2 dt^2 - dx^2 - dy^2 - dz^2}$는 시공간 간격이다.

좌표 시간의 함수로 쓴 라그랑지안은:

$$L = -mc^2\sqrt{1 - v^2/c^2} + q\mathbf{v} \cdot \mathbf{A} - qV$$

오일러-라그랑주 방정식은 로렌츠 힘 법칙을 재현한다.

### 7.2 전자기장의 라그랑지안

자유 전자기장의 작용은:

$$S_{\text{field}} = -\frac{1}{4\mu_0}\int F_{\mu\nu}F^{\mu\nu} \, d^4x$$

소스와의 결합항을 더하면:

$$S_{\text{int}} = -\int A_\mu J^\mu \, d^4x$$

$A_\mu$에 대해 $S_{\text{field}} + S_{\text{int}}$를 변분하면 비균질 맥스웰 방정식이 나온다.

### 7.3 응력-에너지 텐서

전자기장의 에너지와 운동량은 **응력-에너지 텐서(stress-energy tensor)**에 집약된다:

$$T^{\mu\nu} = \frac{1}{\mu_0}\left(F^{\mu\alpha}F^{\nu}_{\ \alpha} - \frac{1}{4}\eta^{\mu\nu}F_{\alpha\beta}F^{\alpha\beta}\right)$$

각 성분의 물리적 의미:
- $T^{00} = \frac{1}{2}\left(\epsilon_0 E^2 + \frac{B^2}{\mu_0}\right)$ — 에너지 밀도
- $T^{0i}/c = \epsilon_0(\mathbf{E} \times \mathbf{B})_i$ — 운동량 밀도 (포인팅 벡터 / $c^2$)
- $T^{ij}$ — 맥스웰 응력 텐서(Maxwell stress tensor) (운동량 플럭스)

에너지-운동량의 보존은 다음으로 표현된다:

$$\partial_\mu T^{\mu\nu} = -F^{\nu\alpha}J_\alpha$$

소스가 없으면 $\partial_\mu T^{\mu\nu} = 0$이다.

---

## 8. 시각화: 장 변환

```python
def visualize_field_transformations():
    """
    Show how E and B fields transform as a function of boost velocity.

    Why visualize: seeing the smooth mixing of E and B fields with
    velocity makes the relativistic nature of electromagnetism tangible.
    """
    beta_range = np.linspace(-0.99, 0.99, 500)
    gamma_range = 1.0 / np.sqrt(1 - beta_range**2)
    c = 3e8

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Case 1: Pure B field (1 T in z-direction)
    B0 = 1.0
    Ey_prime = -gamma_range * beta_range * c * B0
    Bz_prime = gamma_range * B0

    axes[0, 0].plot(beta_range, Ey_prime / 1e8, 'b-', linewidth=2, label="$E'_y$ / 10$^8$")
    axes[0, 0].set_ylabel("$E'_y$ (10$^8$ V/m)")
    axes[0, 0].set_title("Pure B-field ($B_z$ = 1 T): Electric field appears")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(beta_range, Bz_prime, 'r-', linewidth=2, label="$B'_z$")
    axes[0, 1].axhline(y=B0, color='gray', linestyle='--', alpha=0.5, label='Lab value')
    axes[0, 1].set_ylabel("$B'_z$ (T)")
    axes[0, 1].set_title("Pure B-field: Magnetic field strengthens")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Case 2: Pure E field (1 MV/m in y-direction)
    E0 = 1e6
    Ey_prime2 = gamma_range * E0
    Bz_prime2 = -gamma_range * beta_range * E0 / c

    axes[1, 0].plot(beta_range, Ey_prime2 / 1e6, 'b-', linewidth=2, label="$E'_y$ / MV/m")
    axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Lab value')
    axes[1, 0].set_xlabel('$\\beta = v/c$')
    axes[1, 0].set_ylabel("$E'_y$ (MV/m)")
    axes[1, 0].set_title("Pure E-field ($E_y$ = 1 MV/m): Strengthens")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(beta_range, Bz_prime2 * 1e3, 'r-', linewidth=2, label="$B'_z$")
    axes[1, 1].set_xlabel('$\\beta = v/c$')
    axes[1, 1].set_ylabel("$B'_z$ (mT)")
    axes[1, 1].set_title("Pure E-field: Magnetic field appears")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Lorentz Transformation of Electromagnetic Fields', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("field_transformations.png", dpi=150)
    plt.show()

visualize_field_transformations()
```

---

## 요약

| 개념 | 핵심 공식 | 물리적 의미 |
|------|-----------|-------------|
| 장 텐서 | $F^{\mu\nu} = \partial^\mu A^\nu - \partial^\nu A^\mu$ | E와 B를 하나의 대상으로 통합 |
| 공변 맥스웰 (비균질) | $\partial_\mu F^{\mu\nu} = \mu_0 J^\nu$ | 가우스 + 앙페르-맥스웰 |
| 공변 맥스웰 (균질) | $\partial_\mu \tilde{F}^{\mu\nu} = 0$ | 단극자 없음 + 패러데이 |
| 장 변환 | $\mathbf{E}' = \gamma(\mathbf{E} + \boldsymbol{\beta}\times c\mathbf{B}) - (\gamma-1)(\mathbf{E}\cdot\hat{\beta})\hat{\beta}$ | 부스트 하에서 E와 B가 섞인다 |
| 로렌츠 불변량 1 | $\mathbf{E} \cdot \mathbf{B} = \text{불변}$ | E와 B 사이의 각도 보존 |
| 로렌츠 불변량 2 | $E^2 - c^2B^2 = \text{불변}$ | 장의 성격 보존 |
| 공변 힘 | $dp^\mu/d\tau = qF^{\mu\nu}u_\nu$ | 상대론적 로렌츠 힘 |
| 응력-에너지 텐서 | $T^{00} = \frac{1}{2}(\epsilon_0 E^2 + B^2/\mu_0)$ | 장의 에너지 밀도 |

---

## 연습 문제

### 연습 1: 상대론적 전류 루프
반지름 $R$의 원형 전류 루프가 전류 $I$를 흘리며 $xy$평면에 놓여 있다. 전하 $q$가 루프 축 위의 위치 $(0, 0, d)$에 정지해 있다. (a) 실험실 기준계에서 $q$에 작용하는 힘은 무엇인가? (b) 이제 $q$가 속도 $v\hat{x}$로 운동하는 기준계로 부스트하라. 장 변환을 이용하여 전하 위치에서 $\mathbf{E}'$와 $\mathbf{B}'$를 계산하라. (c) 공간 힘 $\mathbf{F}' = q(\mathbf{E}' + \mathbf{v}' \times \mathbf{B}')$이 올바른 결과를 줌을 검증하라.

### 연습 2: 장 텐서 불변량
전자기파 $\mathbf{E} = E_0 \hat{y} \cos(kz - \omega t)$, $\mathbf{B} = (E_0/c) \hat{x} \cos(kz - \omega t)$에 대해, (a) 두 로렌츠 불변량 $\mathbf{E} \cdot \mathbf{B}$와 $E^2 - c^2B^2$를 계산하라. (b) 결과를 해석하라: $\mathbf{E}$만 또는 $\mathbf{B}$만 존재하는 기준계를 찾을 수 있는가? (c) 장 텐서를 구성하고 $F_{\mu\nu}F^{\mu\nu}$를 이용하여 불변량을 검증하라.

### 연습 3: 공변 연속 방정식
$\partial_\mu F^{\mu\nu} = \mu_0 J^\nu$로부터 시작하여, $F^{\mu\nu}$의 반대칭성으로부터 전하 보존 $\partial_\nu J^\nu = 0$이 자동으로 따름을 보여라. 이것은 3차원 표기에서 $\nabla \cdot (\nabla \times \mathbf{B}) = 0$이 $\nabla \cdot \mathbf{J} + \partial\rho/\partial t = 0$을 함의하는 것과 유사하다.

### 연습 4: 평면파의 응력-에너지 텐서
$z$방향으로 전파하는 평면 전자기파에 대해 $T^{\mu\nu}$의 16개 성분을 모두 계산하라. $T^{00} = T^{03}/c = T^{33}$ (에너지 밀도가 운동량 플럭스와 같다)임을 보여라 — 이것이 질량 없는 복사의 특징이다.

---

[← 이전: 13. 복사와 안테나](13_Radiation_and_Antennas.md) | [다음: 15. 다중극 전개 →](15_Multipole_Expansion.md)
