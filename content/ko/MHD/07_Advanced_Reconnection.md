# 7. 고급 재결합

## 학습 목표

이 레슨을 마치면 다음을 할 수 있어야 합니다:

1. 플라스모이드 불안정성과 높은 Lundquist 수에서 빠른 재결합에서의 역할 설명하기
2. 난류 재결합 모델(Lazarian-Vishniac) 이해하기
3. 가이드 자기장이 재결합 동역학에 미치는 영향 분석하기
4. 상대론적 재결합과 그 응용 설명하기
5. 3차원 재결합과 준분리층 이해하기
6. 플라스모이드 불안정성과 3D 재결합 구조의 수치 모델 구현하기

## 1. 플라스모이드 불안정성

### 1.1 높은 S에서 Sweet-Parker의 문제

레슨 5에서 보았듯이, Sweet-Parker 재결합은 큰 Lundquist 수에서 극도로 느린 속도를 예측합니다:

$$M_A = S^{-1/2}$$

$S \sim 10^{14}$인 태양 코로나의 경우:

$$M_A \sim 10^{-7}$$

이것은 너무 느립니다. 그러나 근본적인 문제가 있습니다: **Sweet-Parker 전류 시트 자체가 높은 $S$에서 불안정해집니다**.

### 1.2 플라스모이드 불안정성의 시작

플라스모이드 불안정성(전류 시트의 찢김 불안정성이라고도 함)은 Biskamp(1986)에 의해 처음 식별되었고 Loureiro et al.(2007), Bhattacharjee et al.(2009) 등에 의해 체계적으로 연구되었습니다.

**물리적 메커니즘:**

Sweet-Parker 전류 시트는 길이 $L$과 폭 $\delta \sim L/S$를 가집니다. 큰 $S$에 대해 시트는 매우 길고 얇아집니다. 이러한 구성은 시트를 여러 자기 섬(플라스모이드)으로 나누는 **찢김 모드**에 대해 불안정합니다.

**임계 Lundquist 수:**

선형 안정성 분석은 임계 Lundquist 수를 제공합니다:

$$S_c \sim 10^4$$

$S > S_c$에 대해, Sweet-Parker 시트는 불안정하고 플라스모이드 체인으로 조각납니다.

**성장률:**

가장 빠르게 성장하는 모드는 파수를 가집니다:

$$k_{max} L \sim S^{1/4}$$

그리고 성장률:

$$\gamma \tau_A \sim S^{1/4}$$

여기서 $\tau_A = L/v_A$는 Alfvén 시간입니다. 이것은 저항 확산($\gamma_{resistive} \sim S^{-1}$)보다 훨씬 빠릅니다.

**물리적 그림:**

```
초기: Sweet-Parker 시트
    ════════════════════════  전류 시트(길이 L, 폭 δ)


불안정: 플라스모이드 형성
    ════O════X════O════X════O════  X-점과 O-점
```

불안정성은 **플라스모이드 체인**을 생성합니다: 다중 X-점(재결합 지점)과 O-점(자기 섬).

### 1.3 플라스모이드 지배 재결합

플라스모이드 불안정성이 시작되면 재결합 동역학이 근본적으로 변화합니다.

**더 작은 척도로의 캐스케이드:**

각 플라스모이드 자체가 불안정해질 수 있어(재귀적 불안정성) 더 작은 플라스모이드를 생성합니다. 이것은 **척도의 계층**으로 이어집니다:

- 1차 X-점(원본)
- 2차 플라스모이드(크기 $\sim \delta$)
- 3차 플라스모이드(더 작음), 등

캐스케이드는 작은 척도에서 저항률이 중요해질 때까지 계속됩니다.

**효과적인 재결합률:**

각 개별 X-점은 여전히 국소 Sweet-Parker 속도로 재결합할 수 있지만, **총** 재결합률은 다음 이유로 훨씬 빠릅니다:

1. **다중 X-점**: 많은 재결합 지점이 병렬로 작동
2. **더 짧은 전류 시트**: 각 세그먼트는 길이 $\ell \sim L/N$을 가지며 여기서 $N \sim S^{1/4}$는 플라스모이드 수입니다

각 세그먼트의 효과적인 Lundquist 수는:

$$S_{eff} = \frac{\ell v_A}{\eta} \sim \frac{L v_A}{N \eta} \sim \frac{S}{S^{1/4}} = S^{3/4}$$

각 X-점에서의 국소 재결합률은:

$$M_{A,local} \sim S_{eff}^{-1/2} \sim S^{-3/8}$$

하지만 $N \sim S^{1/4}$개의 X-점이 있으므로, 단위 시간당 재결합되는 총 자속은:

$$M_{A,total} \sim N \cdot M_{A,local} \sim S^{1/4} \cdot S^{-3/8} = S^{-1/8}$$

이것은 Sweet-Parker($S^{-1/2}$)보다 **훨씬 빠릅니다**!

**점근적 행동:**

$S \to \infty$의 극한에서, 캐스케이드가 운동학적 척도까지 계속되면, 재결합률은 **$S$와 독립적**이 됩니다:

$$M_A \sim \text{const} \sim 0.01\text{–}0.1$$

이것은 고$S$ 플라즈마에 대한 재결합률 문제를 해결합니다.

### 1.4 수치 시뮬레이션

**2D 저항 MHD 시뮬레이션:**

- Loureiro et al.(2007): 2D MHD에서 플라스모이드 불안정성 시연, $\gamma \propto S^{1/4}$ 확인
- Bhattacharjee et al.(2009): 스케일링 이론 개발
- Huang & Bhattacharjee(2010, 2012): Sweet-Parker에서 플라스모이드 지배 영역으로의 전이 시연
- Uzdensky et al.(2010): 상대론적 재결합에서의 플라스모이드 불안정성

**주요 발견:**

1. $S < 10^4$의 경우: Sweet-Parker 재결합은 안정적
2. $S > 10^4$의 경우: 플라스모이드 불안정성 시작
3. $S \gg 10^6$의 경우: 완전히 발달한 플라스모이드 지배 영역, $M_A \sim 0.01$

**관측 증거:**

- **태양 플레어**: SOHO 및 SDO 이미지의 Supra-arcade downflows(SADs)는 플라스모이드로 해석됩니다
- **자기꼬리**: 우주선에 의해 관측된 지구 방향 및 꼬리 방향으로 이동하는 자속 로프(플라스모이드)
- **토카막**: 톱니파 충돌은 작은 척도 구조의 폭발을 보여줍니다

### 1.5 Python 예제: 플라스모이드 불안정성 성장률

```python
import numpy as np
import matplotlib.pyplot as plt

# Lundquist number range
S = np.logspace(2, 10, 200)

# Critical Lundquist number
S_c = 1e4

# Growth rate scaling
# Below S_c: resistive growth (very slow)
gamma_resistive = 0.01 * S**(-1)

# Above S_c: plasmoid instability
gamma_plasmoid = np.where(S > S_c, 0.1 * S**(1/4), gamma_resistive)

# Alfven time (normalized)
tau_A = 1.0

# Growth rate in units of 1/tau_A
gamma_norm = gamma_plasmoid / tau_A

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Growth rate
ax = axes[0]
ax.loglog(S, gamma_resistive, label='Resistive (no plasmoids): $\\gamma \\propto S^{-1}$',
          linewidth=2, linestyle='--', color='blue')
ax.loglog(S, gamma_plasmoid, label='Plasmoid instability: $\\gamma \\propto S^{1/4}$',
          linewidth=2.5, color='red')
ax.axvline(S_c, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.text(S_c, 1e-4, f'$S_c \\sim {S_c:.0e}$', fontsize=12, rotation=90, va='bottom',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))

ax.set_xlabel('Lundquist number $S$', fontsize=14)
ax.set_ylabel('Growth rate $\\gamma \\tau_A$', fontsize=14)
ax.set_title('Plasmoid Instability Growth Rate', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Panel 2: Reconnection rate scaling
ax = axes[1]

# Sweet-Parker
M_SP = S**(-0.5)

# Plasmoid-mediated
M_plasmoid = np.where(S > S_c, 0.01 * S**(-1/8), M_SP)

# Hall reconnection (constant)
M_Hall = 0.1 * np.ones_like(S)

ax.loglog(S, M_SP, label='Sweet-Parker: $M_A \\propto S^{-1/2}$',
          linewidth=2, linestyle='--', color='blue')
ax.loglog(S, M_plasmoid, label='Plasmoid-mediated: $M_A \\propto S^{-1/8}$',
          linewidth=2.5, color='red')
ax.loglog(S, M_Hall, label='Hall (collisionless): $M_A \\sim 0.1$',
          linewidth=2, linestyle='-.', color='green')

ax.axvline(S_c, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.axhline(0.01, color='gray', linestyle=':', alpha=0.5)
ax.text(1e9, 0.015, 'Typical observed rate', fontsize=11, color='gray')

ax.set_xlabel('Lundquist number $S$', fontsize=14)
ax.set_ylabel('Reconnection rate $M_A$', fontsize=14)
ax.set_title('Reconnection Rate: Sweet-Parker vs Plasmoid-Mediated', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(1e-8, 1)

plt.tight_layout()
plt.savefig('plasmoid_instability_scaling.png', dpi=150)
plt.show()

# Print transition properties
print("Plasmoid Instability Transition")
print("=" * 50)
print(f"Critical Lundquist number: S_c ~ {S_c:.0e}")
print(f"\nAt S = {S_c:.0e}:")
print(f"  Growth rate: γ τ_A ~ {gamma_plasmoid[np.argmin(np.abs(S - S_c))]:.3f}")
print(f"  Reconnection rate: M_A ~ {M_plasmoid[np.argmin(np.abs(S - S_c))]:.2e}")

S_high = 1e8
idx_high = np.argmin(np.abs(S - S_high))
print(f"\nAt S = {S_high:.0e} (solar corona):")
print(f"  Number of plasmoids: N ~ S^(1/4) ~ {S_high**(1/4):.0f}")
print(f"  Growth rate: γ τ_A ~ {gamma_plasmoid[idx_high]:.1f}")
print(f"  Reconnection rate: M_A ~ {M_plasmoid[idx_high]:.3f}")
print(f"  (Compare Sweet-Parker: M_A ~ {M_SP[idx_high]:.2e})")
```

### 1.6 플라스모이드 크기 분포

완전히 발달한 플라스모이드 매개 재결합에서는 대형에서 소형까지 플라스모이드 크기의 분포가 있습니다.

**멱법칙 분포:**

시뮬레이션과 이론 모델은 멱법칙 크기 분포를 제안합니다:

$$N(w) \propto w^{-\alpha}$$

여기서 $N(w)$는 폭 $w$를 가진 플라스모이드의 수이고, $\alpha \sim 1$–2는 영역에 따라 달라집니다.

**가장 큰 플라스모이드:**

가장 큰 플라스모이드는 크기를 가집니다:

$$w_{max} \sim \delta \sim L / S$$

(원래 전류 시트 폭).

**가장 작은 플라스모이드:**

캐스케이드는 저항률(또는 운동학적 효과)이 중요해지는 척도에서 종료됩니다:

$$w_{min} \sim \eta / v_A \sim L / S$$

잠깐, 이것은 $\delta$와 같은 것처럼 보입니다! 핵심은 플라스모이드 지배 영역에서 효과적인 저항률 또는 운동학적 척도가 순진한 추정과 다르다는 것입니다.

실제로 최소 척도는 다음에 의해 설정됩니다:
- 저항 MHD에서: 확산 척도 $\sim \sqrt{\eta t}$
- 운동학적 플라즈마에서: 이온 표피 깊이 $d_i$ 또는 전자 표피 깊이 $d_e$

## 2. 난류 재결합

### 2.1 Lazarian-Vishniac 모델

Lazarian & Vishniac(1999, LV99)은 급진적인 아이디어를 제안했습니다: 플라즈마의 **난류**가 저항률과 독립적으로 빠른 재결합을 가능하게 할 수 있습니다.

**핵심 아이디어:**

난류 플라즈마에서 자기장 선은 **무작위 걷기**(확률적 방황)를 겪습니다. 이것은 재결합 영역을 효과적으로 넓혀 자기장 선이 더 빠르게 확산할 수 있게 합니다.

**모델 설정:**

다음을 가진 난류 매질에서의 재결합을 고려하세요:
- 척도 $l$에서의 난류 속도 $\delta v_l$
- 난류 자기장 섭동 $\delta B_l$
- 배경 재결합 자기장 $B_0$

**자기장 선 방황:**

거리 $L$에 걸쳐 추적된 자기장 선은 무작위 변위를 겪습니다. r.m.s. 변위는:

$$\delta x \sim \frac{\delta B_l}{B_0} l \left( \frac{L}{l} \right)^{1/2}$$

이것은 난류가 충분히 강하면 Sweet-Parker 폭 $\delta \sim L/S$보다 큽니다.

**효과적인 확산 영역:**

난류 방황은 확산 영역의 효과적인 폭을 만듭니다:

$$\delta_{eff} \sim \delta x \gg \delta_{SP}$$

**재결합률:**

재결합률은 다음이 됩니다:

$$M_A \sim \frac{\delta_{eff}}{L} \sim \frac{\delta B}{B_0} \left( \frac{l}{L} \right)^{1/2}$$

난류가 trans-Alfvénic($\delta v_l \sim v_A$, $\delta B_l \sim B_0$를 의미)이면:

$$M_A \sim \left( \frac{l}{L} \right)^{1/2}$$

척도 $l \sim 0.01 L$의 난류의 경우:

$$M_A \sim 0.1$$

**핵심 결과:** 재결합률은 **저항률과 독립적**이며, 난류 특성에만 의존합니다.

### 2.2 난류 재결합의 조건

LV99 모델이 적용되려면:

1. **난류가 존재해야 함**: 기존 난류 또는 자체 생성(재결합 자체가 난류 구동)
2. **강한 난류**: $\delta v_l / v_A \sim 1$(trans-Alfvénic)
3. **큰 척도**: 난류 주입 척도 $l$이 $L$에 비견됨

**응용:**

- **분자 구름**: 별 형성 영역은 강한 초음속 난류를 가짐
- **은하단**: 난류 ICM(클러스터 간 매질)
- **태양풍**: Alfvénic 난류가 어디에나 존재
- **강착원반**: MRI 구동 난류

**논쟁:**

LV99 모델은 논란의 여지가 있습니다. 비평가들은 다음을 주장합니다:
- 난류가 X-점 근처에서 감쇠될 수 있음
- 시뮬레이션은 다른 행동을 보여줌
- 모델은 기존 난류를 가정하지만, 어떻게 발생합니까?

그러나 난류가 재결합을 촉진할 수 있다는 아이디어는 특히 천체물리학 맥락에서 주목을 받고 있습니다.

### 2.3 재결합 구동 난류

재결합 자체는 다음을 통해 난류를 생성할 수 있습니다:

- **플라스모이드 불안정성**: 변동과 흐름 생성
- **Kelvin-Helmholtz 불안정성**: 유출 제트에서
- **스트리밍 불안정성**: 에너지 입자로부터

이 **자체 생성 난류**는 피드백하여 재결합을 강화할 수 있으며, 양의 피드백 루프를 생성합니다.

## 3. 가이드 자기장 재결합

### 3.1 가이드 자기장이란 무엇인가?

지금까지 **반평행 재결합**을 고려했습니다: 재결합 자기장 성분이 반대이며, 전류 방향을 따라 성분이 없습니다.

**가이드 자기장** $B_g$는 **재결합 전류에 평행한** 자기장 성분입니다(2D에서 면외):

```
재결합 자기장:    B_x(시트를 가로질러 반전)
전류 방향:     J_z(면 밖)
가이드 자기장:           B_g = B_z(균일, 면 밖)
```

**정규화된 가이드 자기장:**

$$B_g / B_0$$

여기서 $B_0$는 재결합 자기장 강도입니다.

- $B_g = 0$: 반평행(또는 무가이드) 재결합
- $B_g / B_0 \ll 1$: 약한 가이드 자기장
- $B_g / B_0 \sim 1$: 중간 가이드 자기장
- $B_g / B_0 \gg 1$: 강한 가이드 자기장(성분 재결합)

### 3.2 가이드 자기장의 영향

가이드 자기장은 재결합 동역학에 심오한 영향을 미칩니다:

**1. 대칭성 파괴:**

반평행 재결합은 대칭적입니다(위-아래). 가이드 자기장은 이 대칭성을 깹니다.

**2. 유출 수정:**

유출 속도 방향이 기울어집니다. 반평행 재결합에서 유출은 유입에 수직입니다. 가이드 자기장이 있으면 유출은 비스듬합니다.

**3. 플라스모이드 불안정성 억제:**

강한 가이드 자기장은 플라스모이드 불안정성에 대해 전류 시트를 안정화합니다. 임계 Lundquist 수가 증가합니다:

$$S_c(B_g) \sim S_c(0) \cdot \left( 1 + \frac{B_g^2}{B_0^2} \right)^{3/2}$$

**4. 입자 가속에 영향:**

- **반평행**: 입자는 재결합 전기장에 의해 가속될 수 있습니다(직접 자기장 정렬 가속)
- **가이드 자기장**: 입자는 $B_g$ 주위를 회전하여 가속 메커니즘을 변경합니다(Fermi 반사, 곡률 드리프트)

**5. Hall 자기장 구조 변화:**

무충돌 재결합에서 Hall 사중극 자기장은 가이드 자기장에 의해 수정됩니다. 강한 $B_g$의 경우, Hall 자기장이 억제됩니다.

### 3.3 가이드 자기장 대 재결합률

시뮬레이션은 재결합률이 가이드 자기장 강도에 의존함을 보여줍니다:

$$M_A(B_g) \approx \frac{M_A(0)}{1 + B_g^2 / B_0^2}$$

$B_g \gg B_0$의 경우, 재결합은 매우 느려집니다("성분 재결합").

**물리적 해석:**

가이드 자기장은 자기 장력을 증가시켜 자기장 선을 늘이고 끊기 더 어렵게 만듭니다.

### 3.4 응용

가이드 자기장 재결합은 다음과 관련이 있습니다:

- **태양 코로나**: 코로나 루프는 종종 강한 축 자기장을 가짐
- **자기권계면**: 자기권덮개 자기장은 전류에 평행한 성분을 가짐
- **토카막**: 3D의 재결합은 가이드 자기장을 포함할 수 있음
- **자기꼬리**: 북향 IMF 동안, 자기권계면에 가이드 자기장이 있을 수 있음

## 4. 상대론적 재결합

### 4.1 재결합이 상대론적이 되는 경우

재결합은 다음의 경우 **상대론적**이 됩니다:

1. **자기적으로 지배되는 플라즈마**: 자기화 $\sigma \gg 1$, 여기서:

$$\sigma = \frac{B^2}{\mu_0 \rho c^2} = \frac{v_A^2}{c^2} \cdot \gamma_{th}$$

   $\sigma \gg 1$의 경우, 자기 에너지 밀도가 정지 질량 에너지 밀도를 초과합니다.

2. **상대론적 흐름**: 유출 속도 $v \sim c$, Lorentz 인자 $\Gamma > 1$

3. **상대론적 입자**: 입자 에너지 $\gamma m c^2 \gg m c^2$

**어디에서 발생합니까?**

- **펄서 자기권**: $\sigma \sim 10^4$–$10^7$
- **마그네타**: $\sigma \sim 10^{10}$
- **AGN 제트**: $\sigma \sim 1$–$10$(중간)
- **감마선 폭발**: $\sigma \sim 10$–$100$(또는 더 높음)
- **블랙홀 자기권**: 사건 지평선 근처

### 4.2 상대론적 MHD 방정식

상대론적 MHD 방정식은 **응력-에너지 텐서** $T^{\mu\nu}$와 **전자기 텐서** $F^{\mu\nu}$를 포함합니다.

**에너지-운동량 보존:**

$$\partial_\mu T^{\mu\nu} = 0$$

여기서:

$$T^{\mu\nu} = (\rho c^2 + u + p + b^2) \frac{u^\mu u^\nu}{c^2} + (p + b^2/2) g^{\mu\nu} - b^\mu b^\nu$$

그리고 $b^\mu$는 4-벡터 자기장, $u^\mu$는 4-속도입니다.

**이상 Ohm 법칙:**

플라즈마 정지 프레임에서:

$$E^{\mu} + (u \times B)^\mu = 0$$

**재결합 전기장:**

상대론적 재결합에서 플라즈마 프레임의 전기장은:

$$E' \sim \Gamma v B \sim v_A B$$

여기서 $\Gamma$는 유입의 대량 Lorentz 인자입니다.

### 4.3 상대론적 영역에서의 재결합률

놀랍게도, **상대론적 재결합도 빠르며**, 다음을 가집니다:

$$M_A \equiv \frac{v_{in}}{c} \sim 0.1$$

이것은 비상대론적 무충돌 재결합과 유사합니다!

**왜?**

핵심 물리학은 유사합니다:
- **2-척도 구조**: 이온(또는 쌍)이 표피 깊이 $\sim$ 척도에서 분리
- **전자 척도 소산**: 전자가 작은 척도에서 재결합 제어
- **빠른 속도**: 큰 $S$에 대해 저항률과 독립적

**차이점:**

- 유출은 $v_{out} \sim c$에 도달할 수 있음(Lorentz 인자 $\Gamma_{out} \sim$ 몇에서 10)
- 유출의 자기장 압축은 상대론적 효과에 의해 제한됨
- 입자 가속이 더 효율적(멱법칙 꼬리)

### 4.4 응용: 펄서 바람과 GRBs

**펄서 바람 성운:**

Crab Nebula는 Crab 펄서에 의해 동력을 공급받습니다. 펄서 바람은 다음을 가집니다:

$$\sigma_{wind} \sim 10^4 \text{ (펄서 근처)} \to 0.01\text{–}0.1 \text{ (종결 충격에서)}$$

**시그마 문제:** 자기 에너지가 입자 에너지로 어떻게 변환됩니까?

**답:** 줄무늬 바람(교대 극성)의 재결합이 자기 에너지를 소산합니다.

- 재결합률: $M_A \sim 0.1$
- 입자 가속: 비열적 $\gamma$선 방출
- 플레어: Crab 플레어(2011)는 재결합 이벤트에 기인

**감마선 폭발:**

GRB 제트에서 상대론적 재결합은 다음을 할 수 있습니다:

- 자기 에너지 소산 → 즉발 감마선 방출
- 전자 가속 → 동기복사
- 빠른 변동성 생성 → 플라스모이드 방출

최근 시뮬레이션(Sironi, Spitkovsky, Werner, Uzdensky)은 상대론적 재결합이 GRB 스펙트럼과 일치하는 멱법칙 입자 분포를 효율적으로 생성함을 보여줍니다.

### 4.5 Python 예제: 상대론적 재결합 유출

```python
import numpy as np
import matplotlib.pyplot as plt

# Magnetization parameter
sigma = np.logspace(-1, 4, 100)

# Alfven speed (non-relativistic)
v_A_nonrel = 1  # Normalized to c

# Relativistic Alfven speed
v_A_rel = np.sqrt(sigma / (1 + sigma))  # In units of c

# Outflow speed (approximate, from simulations)
# Non-relativistic: v_out ~ v_A
v_out_nonrel = v_A_nonrel * np.ones_like(sigma)

# Relativistic: v_out ~ c for large sigma
v_out_rel = 0.9 * v_A_rel  # Slightly less than v_A due to compression

# Lorentz factor of outflow
gamma_out_rel = 1 / np.sqrt(1 - v_out_rel**2)

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Panel 1: Outflow velocity
ax = axes[0]
ax.semilogx(sigma, v_A_rel, label='Relativistic Alfvén speed $v_A/c$', linewidth=2.5, color='blue')
ax.semilogx(sigma, v_out_rel, label='Relativistic outflow $v_{out}/c$', linewidth=2.5, linestyle='--', color='red')
ax.axhline(1, color='black', linestyle=':', linewidth=2, alpha=0.7)
ax.text(1e3, 1.05, 'Speed of light', fontsize=12, color='black')

ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.text(1, 0.1, '$\\sigma = 1$', fontsize=12, rotation=90, va='bottom', color='gray')

ax.set_xlabel('Magnetization $\\sigma = B^2/(\\mu_0 \\rho c^2)$', fontsize=14)
ax.set_ylabel('Velocity (units of $c$)', fontsize=14)
ax.set_title('Relativistic Reconnection Outflow Velocity', fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.2)

# Panel 2: Lorentz factor
ax = axes[1]
ax.loglog(sigma, gamma_out_rel, linewidth=2.5, color='purple')
ax.axvline(1, color='gray', linestyle='--', alpha=0.5)
ax.axhline(1, color='black', linestyle=':', alpha=0.7)

# Mark example regimes
ax.axvline(10, color='orange', linestyle=':', alpha=0.7)
ax.text(10, 0.5, 'GRB jets\n$\\sigma \\sim 10$', fontsize=11, ha='center', color='orange',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

ax.axvline(1e6, color='green', linestyle=':', alpha=0.7)
ax.text(1e6, 0.5, 'Pulsar\nwind', fontsize=11, ha='center', color='green',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

ax.set_xlabel('Magnetization $\\sigma$', fontsize=14)
ax.set_ylabel('Outflow Lorentz factor $\\Gamma_{out}$', fontsize=14)
ax.set_title('Outflow Lorentz Factor vs Magnetization', fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 1e2)

plt.tight_layout()
plt.savefig('relativistic_reconnection_outflow.png', dpi=150)
plt.show()

# Print example values
print("Relativistic Reconnection Outflow Properties")
print("=" * 60)
sigma_examples = [0.1, 1, 10, 100, 1000, 1e6]
for sig in sigma_examples:
    v_a = np.sqrt(sig / (1 + sig))
    v_out = 0.9 * v_a
    gamma = 1 / np.sqrt(1 - v_out**2)
    print(f"σ = {sig:>8.1e}:  v_A/c = {v_a:.4f},  v_out/c = {v_out:.4f},  Γ_out = {gamma:>6.2f}")
```

## 5. 3차원 재결합

### 5.1 2D 모델의 한계

지금까지 논의된 모든 모델은 **2D 기하학**을 가정합니다: $x$와 $y$에서의 변화, 하지만 $z$에서의 불변성.

현실에서 재결합은 **3차원적**입니다:

- 자기장은 세 성분 모두를 가짐
- 전류 시트는 범위가 무한하지 않음
- 재결합 영역은 국소화됨

**주요 3D 효과:**

1. **유한 범위**: 재결합 영역은 세 번째 차원에서 유한 길이를 가짐
2. **비스듬한 자기장**: 자기장이 전류 시트에 비스듬할 수 있음
3. **스파인-팬 위상 구조**: 3D 영점은 복잡한 구조를 가짐(단순 X-점이 아님)
4. **자속 튜브 상호작용**: 개별 자속 튜브가 재결합하며, 전체 시트가 아님
5. **준분리층(QSLs)**: 분리선의 일반화

### 5.2 3D의 자기 영점

3D에서 자기 영점은 $\mathbf{B} = 0$인 점입니다. 영점 근처에서 자기장은 선형화될 수 있습니다:

$$\mathbf{B} = \mathbf{M} \cdot \mathbf{r}$$

여기서 $\mathbf{M}$은 Jacobian 행렬입니다. $\mathbf{M}$의 고유값이 영점 유형을 결정합니다.

**3D 영점의 유형:**

1. **방사 영점**: 모든 고유값이 같은 부호를 가짐(소스 또는 싱크) — **불안정**, 무력 자기장에서는 관측되지 않음

2. **나선 영점**: 하나의 실수 고유값, 두 개의 복소수 켤레 — 자기장이 영점 주위로 나선

3. **적절한 영점**: 세 개의 실수 고유값, 두 개는 한 부호(팬 평면), 하나는 반대(스파인) — **가장 일반적**

**스파인-팬 구조:**

```
           스파인(1D)
               |
               |
        팬 평면(2D)
```

**스파인**은 영점을 통과하는 자기장 선입니다(1D). **팬**은 영점을 통과하는 자기장 선의 표면입니다(2D).

**3D 영점에서의 재결합:**

재결합은 팬 평면(분리자 재결합) 또는 스파인을 따라(스파인 재결합)에서 발생합니다.

### 5.3 준분리층(QSLs)

2D에서 **분리선**은 서로 다른 위상 구조의 영역을 분리하는 자기장 선입니다. 3D에서 정확한 분리선은 드뭅니다.

대신, **준분리층(QSLs)**은 자기장 선 연결성이 빠르게 변화하는 얇은 층입니다.

**정의:**

**찌부러짐 인자(squashing factor)** $Q$는 자기장 선 다발이 얼마나 찌부러지는지 측정합니다:

$$Q = \frac{|\nabla_\perp \phi|^2}{|\sin \theta|}$$

여기서 $\phi$는 자기장 선 매핑이고 $\theta$는 자기장 선과 표면 사이의 각도입니다.

높은 $Q$(예: $Q > 2$)는 QSL을 나타냅니다.

**특성:**

- QSLs는 표면입니다(3D 공간에서 2D)
- QSL 내의 자기장 선은 강한 전단을 겪음
- 전류가 QSLs에 집중됨
- 재결합은 우선적으로 QSLs에서 발생함

**응용:**

- **태양 코로나**: 관측된 플레어 리본은 종종 QSLs를 추적
- **토카막**: Edge localized modes(ELMs)는 QSL 동역학을 포함
- **자기권**: 자기권계면 재결합은 본질적으로 3D

### 5.4 슬립-러닝 재결합

3D에서 재결합은 단일 X-점에서 발생할 필요가 없습니다. 대신, 재결합은 자기장 선을 따라 **슬립**할 수 있습니다.

**슬립-러닝 재결합:**

각도로 교차하는 두 자속 튜브를 상상하세요. 재결합은 한 점에서 시작한 다음 교차선(분리자)을 따라 **전파**합니다.

이것을 **슬립-러닝** 또는 **지퍼 재결합**이라고 합니다.

**관측 증거:**

- **태양 분출**: 플레어 리본은 종종 전파하는 밝아짐(슬립 운동)을 보여줌
- **자기 구름**: CME 자속 로프는 점진적 재결합의 신호를 보여줌

### 5.5 Python 예제: 3D 영점 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D magnetic field with a null point
# Example: Spine-fan null

def magnetic_field_null_3d(x, y, z):
    """
    Create a spine-fan null:
    Eigenvalues: (+a, +a, -2a) to satisfy div B = 0
    """
    a = 1.0
    Bx = a * x
    By = a * y
    Bz = -2 * a * z
    return Bx, By, Bz

# Grid
x = np.linspace(-2, 2, 15)
y = np.linspace(-2, 2, 15)
z = np.linspace(-2, 2, 15)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Magnetic field
Bx, By, Bz = magnetic_field_null_3d(X, Y, Z)

# Magnitude
B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)

# Plot
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# Quiver plot (subsample for clarity)
skip = 2
ax.quiver(X[::skip, ::skip, ::skip], Y[::skip, ::skip, ::skip], Z[::skip, ::skip, ::skip],
          Bx[::skip, ::skip, ::skip], By[::skip, ::skip, ::skip], Bz[::skip, ::skip, ::skip],
          length=0.3, normalize=True, color='blue', alpha=0.6)

# Mark the null point
ax.scatter([0], [0], [0], color='red', s=200, marker='o', label='Null point')

# Spine (along z-axis, Bz direction)
z_spine = np.linspace(-2, 2, 50)
x_spine = np.zeros_like(z_spine)
y_spine = np.zeros_like(z_spine)
ax.plot(x_spine, y_spine, z_spine, 'r-', linewidth=3, label='Spine (field line through null)')

# Fan plane (z=0 plane)
theta_fan = np.linspace(0, 2*np.pi, 100)
r_fan = 1.5
x_fan = r_fan * np.cos(theta_fan)
y_fan = r_fan * np.sin(theta_fan)
z_fan = np.zeros_like(theta_fan)
ax.plot(x_fan, y_fan, z_fan, 'g-', linewidth=3, label='Fan (field lines in z=0 plane)')

# Field lines in fan
for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
    r_line = np.linspace(0.1, 1.8, 20)
    x_line = r_line * np.cos(angle)
    y_line = r_line * np.sin(angle)
    z_line = np.zeros_like(r_line)
    ax.plot(x_line, y_line, z_line, 'g--', linewidth=1, alpha=0.5)

ax.set_xlabel('X', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_zlabel('Z', fontsize=13)
ax.set_title('3D Magnetic Null: Spine-Fan Structure', fontsize=16, weight='bold')
ax.legend(fontsize=11)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)

plt.tight_layout()
plt.savefig('3d_null_spine_fan.png', dpi=150)
plt.show()

# Plot field magnitude
fig = plt.figure(figsize=(12, 5))

# XY plane
ax = fig.add_subplot(121)
z_idx = len(z) // 2
contour = ax.contourf(X[:, :, z_idx], Y[:, :, z_idx], B_mag[:, :, z_idx], levels=20, cmap='viridis')
ax.streamplot(X[:, :, z_idx], Y[:, :, z_idx], Bx[:, :, z_idx], By[:, :, z_idx],
              color='white', linewidth=1, density=1.5)
ax.plot(0, 0, 'ro', markersize=12)
plt.colorbar(contour, ax=ax, label='$|\\mathbf{B}|$')
ax.set_xlabel('X', fontsize=13)
ax.set_ylabel('Y', fontsize=13)
ax.set_title('Fan Plane (z=0): Field Magnitude', fontsize=14)
ax.set_aspect('equal')

# XZ plane
ax = fig.add_subplot(122)
y_idx = len(y) // 2
contour = ax.contourf(X[:, y_idx, :], Z[:, y_idx, :], B_mag[:, y_idx, :], levels=20, cmap='plasma')
ax.streamplot(X[:, y_idx, :], Z[:, y_idx, :], Bx[:, y_idx, :], Bz[:, y_idx, :],
              color='white', linewidth=1, density=1.5)
ax.plot(0, 0, 'ro', markersize=12)
plt.colorbar(contour, ax=ax, label='$|\\mathbf{B}|$')
ax.set_xlabel('X', fontsize=13)
ax.set_ylabel('Z', fontsize=13)
ax.set_title('Spine Direction (y=0): Field Magnitude', fontsize=14)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('3d_null_field_magnitude.png', dpi=150)
plt.show()
```

### 5.6 3D 재결합의 관측 신호

**태양 관측:**

- **슬립 자기 재결합**: 플레어 리본이 극성 반전선(PIL)을 따라 슬립 운동을 보여줌으로써, 분리자를 따라 전파하는 재결합을 나타냄
- **원형 리본 플레어**: 3D 영점의 팬을 추적
- **스파인 관련 제트**: 스파인 자기장 선에 국한됨

**자기권 관측:**

- **패치 재결합**: 자기권계면에서의 재결합은 3D로 국소화되며, 자기권계면을 따라 균일하지 않음
- **FTEs(Flux Transfer Events)**: 재결합에 의해 생성된 3D 자속 로프

## 요약

자기 재결합의 고급 주제를 탐구했습니다:

1. **플라스모이드 불안정성**: 높은 Lundquist 수($S > 10^4$)에서 Sweet-Parker 전류 시트는 불안정해져 플라스모이드와 X-점의 체인으로 조각납니다. 성장률은 $\gamma \propto S^{1/4}$로 스케일링됩니다. 플라스모이드 매개 재결합은 훨씬 빠른 속도 $M_A \propto S^{-1/8}$를 제공하며, 매우 큰 $S$에 대해 상수 $\sim 0.01$–$0.1$에 접근합니다. 이것은 천체물리학 플라즈마에 대한 재결합률 문제를 해결합니다.

2. **난류 재결합**: Lazarian-Vishniac 모델은 난류가 확률적 자기장 선 방황을 일으켜 확산 영역을 효과적으로 넓히고 저항률과 독립적으로 빠른 재결합을 가능하게 한다고 가정합니다. 재결합률은 난류 특성에 의존하며, $M_A \sim (l/L)^{1/2}$입니다. 논쟁의 여지가 있지만, 이 모델은 난류 천체물리학 환경과 관련이 있습니다.

3. **가이드 자기장 효과**: 재결합 전류에 평행한 자기장 성분(가이드 자기장)은 대칭성을 깨고, 플라스모이드 불안정성을 억제하며, 재결합률을 감소시킵니다. 강한 가이드 자기장은 느린 "성분 재결합"으로 이어집니다. 가이드 자기장은 또한 입자 가속 메커니즘과 Hall 자기장 구조를 수정합니다.

4. **상대론적 재결합**: 자기적으로 지배되는 플라즈마($\sigma \gg 1$)에서 재결합은 상대론적입니다. 극단적인 조건에도 불구하고, 재결합률은 비상대론적 무충돌 재결합과 유사하게 빠르며, $M_A \sim 0.1$입니다. 유출은 Lorentz 인자 $\Gamma \sim$ 몇에서 10으로 $v \sim c$에 도달할 수 있습니다. 응용은 펄서 바람, 마그네타, GRB 제트, AGN을 포함합니다. 상대론적 재결합은 입자를 비열적 분포로 효율적으로 가속합니다.

5. **3차원 재결합**: 실제 재결합은 3D입니다. 3D 자기 영점은 스파인-팬 구조를 가집니다(단순 X-점이 아님). 재결합은 분리자(분리선의 교차)에서 또는 자기장 선 연결성이 빠르게 변화하는 준분리층(QSLs)에서 발생할 수 있습니다. 슬립-러닝(지퍼) 재결합은 분리자를 따라 전파합니다. 태양 플레어 리본과 자기권 자속 전달 이벤트는 3D 신호를 보여줍니다.

이러한 고급 주제는 재결합이 풍부하고 다중 척도이며 종종 난류적인 현상임을 보여줍니다. 층류 Sweet-Parker에서 플라스모이드 지배, 난류 또는 운동학적 재결합으로의 전이는 보편적으로 관측되는 빠른 속도를 설명합니다.

## 연습 문제

1. **플라스모이드 불안정성 시작**:
   a) $L = 10^9$ m, $v_A = 10^6$ m/s, $\eta = 10^{-4}$ Ω·m인 태양 플레어 전류 시트에 대해 $S$를 계산하세요.
   b) 이것이 임계 $S_c \sim 10^4$보다 위입니까 아래입니까?
   c) 플라스모이드 수를 추정하세요: $N \sim S^{1/4}$.

2. **플라스모이드 성장률**:
   a) 스케일링 $\gamma \tau_A \sim S^{1/4}$를 사용하여, $S = 10^{12}$이고 $\tau_A = 1000$ s일 때 성장률을 계산하세요.
   b) 이것이 저항 확산 시간 $\tau_{diff} \sim L^2 / \eta$와 어떻게 비교됩니까?

3. **플라스모이드 매개 재결합률**:
   a) Sweet-Parker($S^{-1/2}$)와 플라스모이드 매개($S^{-1/8}$) 스케일링에 대해 $S = 10^4$에서 $10^{16}$까지 $M_A$ 대 $S$를 플롯하세요.
   b) 플라스모이드 속도가 Sweet-Parker보다 10배 빠른 $S$는 얼마입니까?

4. **난류 재결합(LV99)**:
   a) 난류 주입 척도 $l = 0.1 L$인 분자 구름에서 재결합률 $M_A \sim (l/L)^{1/2}$를 추정하세요.
   b) $L = 1$ pc이고 $v_A = 1$ km/s이면 재결합 시간은 얼마입니까?
   c) 별 형성 시간 척도(~Myr)와 비교하세요.

5. **가이드 자기장 억제**:
   a) 가이드 자기장이 있는 재결합률은 $M_A(B_g) \approx M_A(0) / (1 + B_g^2/B_0^2)$입니다. $M_A(0) = 0.1$이고 $B_g = B_0$이면 $M_A(B_g)$는 얼마입니까?
   b) $B_g = 3 B_0$의 경우 $M_A(B_g)$는 얼마입니까?
   c) $0 \le B_g/B_0 \le 5$에 대해 $M_A$ 대 $B_g/B_0$를 플롯하세요.

6. **상대론적 Alfvén 속도**:
   a) 상대론적 Alfvén 속도가 $v_A = c \sqrt{\sigma/(1+\sigma)}$임을 보이세요.
   b) $\sigma = 0.1, 1, 10, 100$에 대해 $v_A/c$를 계산하세요.
   c) 어떤 $\sigma$에서 $v_A = 0.9c$입니까?

7. **상대론적 유출 Lorentz 인자**:
   a) 재결합 유출이 $v_{out} = 0.95c$이면 Lorentz 인자 $\Gamma = 1/\sqrt{1 - v^2/c^2}$를 계산하세요.
   b) $\sigma = 10^4$인 펄서 바람에 대해 유출 속도와 Lorentz 인자를 추정하세요.

8. **펄서의 시그마 문제**:
   a) 펄서 바람이 광원통 근처에서 $\sigma = 10^6$을 가집니다. 재결합이 자기 에너지의 99%를 입자 에너지로 변환하면 최종 $\sigma$는 얼마입니까?
   b) 이것이 종결 충격에서 관측된 $\sigma \sim 0.01$을 설명하기에 충분합니까?
   c) 어떤 추가 소산이 필요할 수 있습니까?

9. **3D 영점 고유값**:
   a) 고유값 $(a, a, -2a)$를 가진 스파인-팬 영점에 대해 $\nabla \cdot \mathbf{B} = 0$을 확인하세요.
   b) 자기장 성분 $\mathbf{B} = (ax, ay, -2az)$를 작성하고 자기장 선을 스케치하세요.
   c) 스파인(1D)과 팬(2D) 구조를 식별하세요.

10. **QSL 찌부러짐 인자**:
    a) 찌부러짐 인자 $Q$의 정의를 조사하세요. 높은 $Q$가 강한 전류와 관련된 이유는 무엇입니까?
    b) 태양 플레어 관측에서 플레어 리본은 종종 $Q > 2$인 영역을 추적합니다. 왜 그럴까요?
    c) 3D 자기장으로부터 $Q$를 수치적으로 어떻게 계산하겠습니까?

## 네비게이션

이전: [재결합 응용](./06_Reconnection_Applications.md) | 다음: [MHD 난류](./08_MHD_Turbulence.md)
