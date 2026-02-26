# 13. 양자광학 입문

[← 이전: 12. 비선형 광학](12_Nonlinear_Optics.md) | [다음: 14. 전산 광학 →](14_Computational_Optics.md)

---

이 강좌의 대부분에서 우리는 빛을 고전적인 전자기파로 취급해 왔으며, 이 설명은 레이저, 광섬유, 회절, 심지어 비선형 광학에서도 아름답게 작동합니다. 그러나 빛은 근본적으로 양자역학적이며, 고전 광학으로는 설명할 수 없는 현상들이 존재합니다. 광자 검출기의 낱알 현상(광자 계수기의 개별 클릭 소리), 고전 확률 이론을 위반하는 먼 거리 광자들 사이의 상관관계, 그리고 고전적 진공 한계 아래의 잡음 수준이 그 예입니다.

양자광학(Quantum Optics)은 맥스웰과 양자역학이 만나는 지점입니다. 이는 단일 광자, 얽힌 광자 쌍, 그리고 양자 통신, 양자 컴퓨팅, 양자 감지를 통해 기술을 재편하고 있는 비고전적 빛의 상태를 기술하는 언어를 제공합니다. 이 레슨은 핵심 개념을 이해하고, 주요 실험들을 감상하며, 빠르게 발전하는 양자 정보 분야와의 연결고리를 파악하기에 충분한 입문 내용을 다룹니다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 전자기장의 양자화를 설명하고 포크(Fock) 상태(수 상태)를 기술한다
2. 결맞음 상태(Coherent State)를 정의하고 그것이 포아송(Poissonian) 광자 통계를 갖는 고전적 레이저 빛에 해당함을 보인다
3. 포아송, 아포아송(Sub-Poissonian), 초포아송(Super-Poissonian) 광자 통계와 그 물리적 기원을 구별한다
4. 단일 광자 소스와 양자 간섭의 특징으로서 홍-오우-만델(Hong-Ou-Mandel) 효과를 기술한다
5. 압축 상태(Squeezed State)와 표준 양자 한계를 넘어선 정밀 측정에의 응용을 설명한다
6. 벨 상태(Bell State)를 정의하고 얽힌 광자 쌍의 생성 및 검증 방법을 기술한다
7. BB84 양자 키 분배 프로토콜과 양자 컴퓨팅의 광자 방식을 개략적으로 설명한다

---

## 목차

1. [빛의 양자화](#1-빛의-양자화)
2. [포크 상태 (수 상태)](#2-포크-상태-수-상태)
3. [결맞음 상태](#3-결맞음-상태)
4. [광자 통계](#4-광자-통계)
5. [단일 광자 소스](#5-단일-광자-소스)
6. [홍-오우-만델 효과](#6-홍-오우-만델-효과)
7. [압축 상태](#7-압축-상태)
8. [얽힘과 벨 상태](#8-얽힘과-벨-상태)
9. [양자 키 분배 (BB84)](#9-양자-키-분배-bb84)
10. [광자를 이용한 양자 컴퓨팅](#10-광자를-이용한-양자-컴퓨팅)
11. [Python 예제](#11-python-예제)
12. [요약](#12-요약)
13. [연습문제](#13-연습문제)
14. [참고문헌](#14-참고문헌)

---

## 1. 빛의 양자화

### 1.1 고전에서 양자로

고전 전자기학에서 빛 장(light field)의 에너지는 어떤 값도 가질 수 있으며 연속적입니다. 플랑크(Planck, 1900)와 아인슈타인(Einstein, 1905)은 빛과 물질 사이의 에너지 교환이 이산적인 양자(quanta) $E = h\nu = \hbar\omega$로 일어남을 보였습니다. 그러나 양자광학은 더 나아가 전자기장 자체가 양자화된다는 것을 다룹니다.

### 1.2 양자 조화 진동자

전자기장의 각 모드(파수벡터 $\mathbf{k}$와 편광 $\hat{e}$로 특징지어지는)는 수학적으로 양자 조화 진동자(Quantum Harmonic Oscillator)와 동치입니다. 단일 모드에 대한 해밀토니안(Hamiltonian)은 다음과 같습니다:

$$\hat{H} = \hbar\omega\!\left(\hat{a}^\dagger\hat{a} + \frac{1}{2}\right) = \hbar\omega\!\left(\hat{n} + \frac{1}{2}\right)$$

여기서:
- $\hat{a}^\dagger$는 **생성 연산자(Creation Operator)** (광자 하나 추가)
- $\hat{a}$는 **소멸 연산자(Annihilation Operator)** (광자 하나 제거)
- $\hat{n} = \hat{a}^\dagger\hat{a}$는 **수 연산자(Number Operator)** (광자 수 계산)
- $\frac{1}{2}\hbar\omega$는 **영점 에너지(Zero-Point Energy)** (진공 요동)

교환 관계 $[\hat{a}, \hat{a}^\dagger] = 1$은 빛의 근본적인 양자적 본질을 담고 있습니다.

### 1.3 전기장 연산자

단일 모드에 대한 양자화된 전기장은 다음과 같습니다:

$$\hat{E}(z, t) = \mathcal{E}_0\!\left(\hat{a}\,e^{i(kz - \omega t)} + \hat{a}^\dagger\,e^{-i(kz - \omega t)}\right)$$

여기서 $\mathcal{E}_0 = \sqrt{\hbar\omega/(2\epsilon_0 V)}$는 **광자당 전기장(Electric Field Per Photon)** — 부피 $V$의 모드에서 단일 양자 들뜸(excitation)과 연관된 장의 세기입니다.

$1\,\text{cm}^3$ 공동(cavity)에서 $\lambda = 500\,\text{nm}$의 전형적인 광학 모드의 경우:

$$\mathcal{E}_0 \approx \sqrt{\frac{(1.055 \times 10^{-34})(3.77 \times 10^{15})}{2(8.854 \times 10^{-12})(10^{-6})}} \approx 1.5\,\text{V/m}$$

이것은 극히 작은 장이지만, 전자기 에너지의 기본 양자를 나타내며 현대 검출기로 측정 가능합니다.

> **비유**: 양자화를 파이프를 통해 흐르는 물에 비유해봅시다. 고전적 그림(넓은 강)에서 물은 매끄럽고 연속적으로 흐릅니다. 그러나 분자 수준으로 확대하면 개별 물 분자들이 보입니다. 마찬가지로, 밝은 레이저 빔은 매끄러운 전자기파처럼 보이지만, 낮은 세기에서 또는 충분히 민감한 검출기로는 개별 광자 "분자들"이 보입니다. 생성 연산자와 소멸 연산자는 한 번에 광자 하나를 추가하거나 제거하는 수도꼭지 같은 것입니다.

---

## 2. 포크 상태 (수 상태)

### 2.1 정의

**포크 상태(Fock States)**(또는 수 상태(Number States)) $|n\rangle$은 광자 수 연산자의 고유 상태(eigenstate)입니다:

$$\hat{n}|n\rangle = n|n\rangle, \quad n = 0, 1, 2, 3, \ldots$$

상태 $|n\rangle$은 정확히 $n$개의 광자를 포함합니다. 에너지는 다음과 같습니다:

$$E_n = \hbar\omega\!\left(n + \frac{1}{2}\right)$$

### 2.2 진공 상태

기저 상태 $|0\rangle$은 **진공(Vacuum)** — 광자가 없지만 에너지 $E_0 = \hbar\omega/2$는 영이 아닙니다. 이 진공 에너지는 실제 물리적 결과를 낳습니다:
- **카시미르 효과(Casimir Effect)**: 변형된 진공 모드로 인한 평행한 도체 판 사이의 인력
- **자발 방출(Spontaneous Emission)**: 들뜬 원자가 진공 요동과 결합하여 붕괴
- **램 이동(Lamb Shift)**: 진공장 요동으로 인한 수소의 작은 에너지 이동

### 2.3 생성과 소멸

$$\hat{a}^\dagger|n\rangle = \sqrt{n+1}\,|n+1\rangle \quad (\text{광자 하나 생성})$$

$$\hat{a}|n\rangle = \sqrt{n}\,|n-1\rangle \quad (\text{광자 하나 소멸})$$

$$\hat{a}|0\rangle = 0 \quad (\text{진공에서 광자를 제거할 수 없음})$$

포크 상태는 진공으로부터 만들 수 있습니다: $|n\rangle = \frac{(\hat{a}^\dagger)^n}{\sqrt{n!}}|0\rangle$.

### 2.4 포크 상태의 성질

- **정확한 광자 수**: $\Delta n = 0$ — 광자 수가 정확히 알려짐
- **완전히 불확실한 위상**: 수-위상 불확정성 관계 $\Delta n \cdot \Delta\phi \geq 1/2$에 의해, $\Delta n = 0$인 상태는 위상이 완전히 무작위적
- **비고전적**: $n \geq 1$인 포크 상태는 어떤 고전 파동으로도 기술될 수 없으며 진정한 양자적 상태

### 2.5 준비의 어려움

포크 상태는 준비하기가 극히 어렵습니다. 단일 광자 상태 $|1\rangle$이 가장 일반적으로 달성되며, $n \geq 2$인 상태는 훨씬 더 어렵습니다. 이것은 양자광학에서의 근본적인 도전입니다.

---

## 3. 결맞음 상태

### 3.1 정의

**결맞음 상태(Coherent States)** $|\alpha\rangle$는 글라우버(Glauber, 2005년 노벨상)가 도입한 것으로, 소멸 연산자의 고유 상태입니다:

$$\hat{a}|\alpha\rangle = \alpha|\alpha\rangle$$

여기서 $\alpha$는 복소수입니다. 이는 고전적 전자기파와 가장 가깝게 유사한 양자 상태입니다.

### 3.2 포크 상태로의 전개

$$|\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle$$

$n$개의 광자를 발견할 확률은:

$$P(n) = |\langle n|\alpha\rangle|^2 = \frac{|\alpha|^{2n}}{n!}e^{-|\alpha|^2}$$

이것은 평균 $\bar{n} = |\alpha|^2$를 갖는 **포아송 분포(Poisson Distribution)**입니다.

### 3.3 성질

- **평균 광자 수**: $\langle\hat{n}\rangle = |\alpha|^2$
- **광자 수 분산**: $\text{Var}(\hat{n}) = |\alpha|^2 = \bar{n}$
- **파노 인자(Fano Factor)**: $F = \text{Var}(n)/\bar{n} = 1$ (포아송)
- **최소 불확정성**: $\Delta X_1 \cdot \Delta X_2 = 1/4$ (두 직교 성분에서 동등한 불확정성)
- **비직교성**: $\langle\alpha|\beta\rangle = e^{-|\alpha-\beta|^2/2} \neq 0$ (단, $|\alpha - \beta| \gg 1$이면 근사적으로 직교)

### 3.4 레이저가 결맞음 상태를 생성하는 이유

문턱값 이상의 레이저는 결맞음 상태로 잘 기술되는 빛을 생성합니다. 이득 포화(gain saturation) 메커니즘은 일정한 진폭과 위상을 유지하는 안정화 피드백으로 작용하며, 양자 잡음은 광자 수의 포아송 요동으로 나타납니다. 레이저 물리학(레슨 8)과 양자장론 사이의 이 연결고리는 글라우버의 핵심 통찰 중 하나였습니다.

### 3.5 변위 연산자

결맞음 상태는 변위된 진공입니다:

$$|\alpha\rangle = \hat{D}(\alpha)|0\rangle, \quad \hat{D}(\alpha) = e^{\alpha\hat{a}^\dagger - \alpha^*\hat{a}}$$

위상 공간(광학 직교 성분(quadrature) 평면 $X_1$-$X_2$)에서, 진공 상태는 원점 중심의 원형 불확정성 덩어리입니다. 결맞음 상태는 같은 덩어리가 점 $(\text{Re}(\alpha), \text{Im}(\alpha))$로 변위된 것입니다.

---

## 4. 광자 통계

### 4.1 세 가지 체계

광자 수 분포는 광원의 양자적 특성을 드러냅니다. **파노 인자(Fano Factor)** $F = \text{Var}(n)/\bar{n}$는 다음과 같이 분류합니다:

**포아송(Poissonian)** ($F = 1$): 결맞음 빛(레이저). 광자 도착은 방사성 붕괴처럼 독립적인 무작위 사건입니다.

**초포아송(Super-Poissonian)** ($F > 1$): 열/혼돈(chaotic) 빛(백열전구, LED). 광자들이 "뭉치는(bunch)" 경향이 있어 무리지어 도착합니다. 이것이 핸버리 브라운-트위스(Hanbury Brown-Twiss, HBT) 효과입니다.

**아포아송(Sub-Poissonian)** ($F < 1$): 비고전적 빛(단일 광자 소스, 압축 빛). 광자들이 무작위보다 더 고르게 간격을 두고 있어 "반집속(antibunch)"합니다. 이것은 고전적으로 설명되지 않습니다.

### 4.2 열 빛 통계

온도 $T$에서 단일 모드의 열(흑체) 복사의 경우:

$$P(n) = \frac{\bar{n}^n}{(1+\bar{n})^{n+1}}, \quad \bar{n} = \frac{1}{e^{\hbar\omega/k_BT} - 1}$$

이것은 **보스-아인슈타인 분포(Bose-Einstein Distribution)**입니다. 분산은:

$$\text{Var}(n) = \bar{n}^2 + \bar{n} = \bar{n}(\bar{n} + 1)$$

따라서 $F = \bar{n} + 1 > 1$: 초포아송.

### 4.3 이차 상관 함수

광자 통계의 표준 실험적 척도는 **이차 상관 함수(Second-Order Correlation Function)** $g^{(2)}(\tau)$입니다:

$$g^{(2)}(\tau) = \frac{\langle\hat{a}^\dagger(t)\hat{a}^\dagger(t+\tau)\hat{a}(t+\tau)\hat{a}(t)\rangle}{\langle\hat{a}^\dagger\hat{a}\rangle^2}$$

지연 시간이 0일 때:

| 광원 | $g^{(2)}(0)$ | 통계 |
|--------|---------------|------------|
| 결맞음 빛 (레이저) | 1 | 포아송 |
| 열 빛 | 2 | 초포아송 (뭉침) |
| 단일 광자 | 0 | 아포아송 (반집속) |
| $n$-광자 포크 | $1 - 1/n$ | 아포아송 |

$g^{(2)}(0) < 1$은 어떤 고전 장에서도 불가능 — 양자 빛의 결정적인 특징입니다.

### 4.4 측정: 핸버리 브라운-트위스 구성

```
                    50:50
   소스 ─────── 빔 ─────── 검출기 A
                  분리기
                    │
                    └────────── 검출기 B
                                    │
                          동시 계수기
                          (g⁽²⁾(τ) 측정)
```

시간 지연 $\tau$의 함수로서 동시 검출 사건의 비율을 측정함으로써 $g^{(2)}(\tau)$를 직접 얻습니다.

---

## 5. 단일 광자 소스

### 5.1 단일 광자가 중요한 이유

단일 광자는 양자 정보의 기본 전달자입니다. 양자 키 분배, 양자 컴퓨팅, 양자 네트워킹을 위해, 한 번에 정확히 하나의 광자를 (요구에 따라) 방출하는 소스가 필요합니다.

### 5.2 단일 광자 소스의 종류

**감쇠 레이저(Attenuated Laser)**: 레이저 세기를 $\bar{n} \ll 1$이 될 때까지 낮춥니다. 광자 통계는 포아송으로 유지되어 0개 또는 2개의 광자를 방출할 확률이 항상 존재합니다. 진정한 단일 광자 소스가 아니지만(여전히 $g^{(2)}(0) = 1$), 단순하고 널리 사용됩니다.

**예고 SPDC(Heralded SPDC)**: 자발적 매개 변수 하향 변환(Spontaneous Parametric Down-Conversion)은 광자 쌍을 생성합니다. 한 광자를 검출하면 그 짝의 존재가 "예고"됩니다. 예고된 광자에 대해 $g^{(2)}(0) \approx 0$. 확률적이지만 잘 특성화됩니다.

**양자점(Quantum Dots)**: 인공 원자처럼 작동하는 반도체 나노구조. 펄스 여기(excitation) 하에서 높은 순도($g^{(2)}(0) < 0.01$)로 단일 광자를 방출하며 결정론적일 수 있습니다. 광자 양자 기술의 선도적 플랫폼.

**다이아몬드의 질소-공공(NV) 센터(Nitrogen-Vacancy Centers in Diamond)**: 실온에서 단일 광자를 방출하는 원자 규모의 결함. $g^{(2)}(0) \sim 0.1$. 양자 감지 및 통신에 사용됩니다.

**가둔 이온/원자(Trapped Ions/Atoms)**: 높은 구별불가능성(indistinguishability)을 갖는 우수한 단일 광자 방출기이지만 복잡한 설정이 필요합니다.

### 5.3 성능 지표

- **순도(Purity)**: $g^{(2)}(0) \to 0$ (낮은 다중 광자 확률)
- **구별불가능성(Indistinguishability)**: 연속 방출된 광자들이 동일해야 함 (같은 주파수, 편광, 시간 프로파일). 홍-오우-만델 가시성(visibility)으로 측정
- **밝기(Brightness)**: 높은 수집 및 추출 효율
- **반복률(Repetition Rate)**: 높은 데이터 전송률을 위한 빠른 트리거

---

## 6. 홍-오우-만델 효과

### 6.1 이광자 간섭

홍-오우-만델(Hong-Ou-Mandel, HOM) 효과(1987)는 양자광학의 가장 인상적인 시연 중 하나입니다. 두 개의 **동일한** 단일 광자가 서로 다른 입사구로 50:50 빔 분리기에 들어가면, 이들은 **항상 같은 출사구로 함께 나옵니다** — 각각 다른 출사구로 나오는 경우는 없습니다.

```
   광자 1 ──→ ┌────────┐ ──→ 두 광자 모두 C 또는 D로 이동
                 │  50:50  │
   광자 2 ──→ │   BS    │ ──→ (C와 D로 하나씩 나오는 경우는 없음)
                 └────────┘
```

### 6.2 양자역학적 설명

빔 분리기는 입력 모드 ($\hat{a}_1, \hat{a}_2$)를 출력 모드 ($\hat{a}_3, \hat{a}_4$)로 변환합니다:

$$\hat{a}_3 = \frac{1}{\sqrt{2}}(\hat{a}_1 + i\hat{a}_2), \quad \hat{a}_4 = \frac{1}{\sqrt{2}}(i\hat{a}_1 + \hat{a}_2)$$

입력 상태: 각 입사구에 하나의 광자: $|1\rangle_1|1\rangle_2 = \hat{a}_1^\dagger\hat{a}_2^\dagger|0\rangle$.

출력 계산:

$$\hat{a}_1^\dagger\hat{a}_2^\dagger = \frac{1}{2}(\hat{a}_3^\dagger - i\hat{a}_4^\dagger)(-i\hat{a}_3^\dagger + \hat{a}_4^\dagger)$$

$$= \frac{1}{2}(-i\hat{a}_3^{\dagger2} + \hat{a}_3^\dagger\hat{a}_4^\dagger - i^2\hat{a}_4^\dagger\hat{a}_3^\dagger + (-i)\hat{a}_4^{\dagger2})$$

교차항 $\hat{a}_3^\dagger\hat{a}_4^\dagger$와 $-\hat{a}_4^\dagger\hat{a}_3^\dagger$는 상쇄됩니다(보손 연산자들은 교환 가능하므로: $\hat{a}_3^\dagger\hat{a}_4^\dagger = \hat{a}_4^\dagger\hat{a}_3^\dagger$이고, 부호가 반대). 출력은:

$$\frac{i}{2}(-\hat{a}_3^{\dagger 2} + \hat{a}_4^{\dagger 2})|0\rangle = \frac{i}{\sqrt{2}}(|2,0\rangle - |0,2\rangle)$$

**두 광자 모두 같은 출사구로 나옵니다** — 동시 검출 비율이 0으로 떨어집니다. 이를 **HOM 딥(HOM Dip)**이라 부릅니다.

### 6.3 물리적 해석

이광자 간섭은 같은 결과에 이르는 두 개의 구별불가능한 경로가 있기 때문에 발생합니다: 두 광자 모두 반사되거나, 두 광자 모두 투과되는 경우. 이 두 진폭은 (반사 시 $\pi/2$ 위상 이동으로 인해) 반대 부호를 가지며 완벽하게 상쇄됩니다 — 단, 광자들이 진정으로 동일(구별불가능)할 때만.

### 6.4 HOM 딥

실제로는 두 광자가 상대적인 시간 지연 $\tau$를 갖고 도착합니다. $\tau = 0$(완전한 겹침)일 때 동시 검출 비율이 0으로 떨어집니다. $|\tau|$가 증가함에 따라 광자들이 구별가능해지고 동시 검출 비율이 고전적 수준으로 회복됩니다. 딥의 폭은 광자들의 결맞음 시간(coherence time)에 해당합니다.

**HOM 가시성(HOM Visibility)** $V = (C_{\max} - C_{\min})/C_{\max}$는 광자 쌍의 구별불가능성을 측정합니다. $V = 1$은 완전히 구별불가능함을 의미하며, $V > 0.5$는 고전적 빛으로는 불가능합니다.

---

## 7. 압축 상태

### 7.1 양자 잡음과 불확정성 원리

전자기장은 두 개의 **직교 성분(Quadratures)**(복소 진폭의 실수부와 허수부처럼)으로 분해할 수 있습니다:

$$\hat{X}_1 = \frac{1}{2}(\hat{a} + \hat{a}^\dagger), \quad \hat{X}_2 = \frac{1}{2i}(\hat{a} - \hat{a}^\dagger)$$

하이젠베르크 불확정성 원리는 다음을 요구합니다:

$$\Delta X_1 \cdot \Delta X_2 \geq \frac{1}{4}$$

결맞음 상태(와 진공)의 경우, $\Delta X_1 = \Delta X_2 = 1/2$ — **표준 양자 한계(Standard Quantum Limit, SQL)**입니다.

### 7.2 압축

**압축 상태(Squeezed State)**는 한 직교 성분의 불확정성이 다른 성분의 불확정성이 증가하는 대가로 감소합니다:

$$\Delta X_1 = \frac{1}{2}e^{-r}, \quad \Delta X_2 = \frac{1}{2}e^{+r}$$

여기서 $r > 0$은 압축 매개변수입니다. 불확정성의 곱은 최솟값에 유지됩니다: $\Delta X_1 \cdot \Delta X_2 = 1/4$.

위상 공간에서, 결맞음 상태의 원형 불확정성 덩어리가 타원형으로 "압축"됩니다 — 이름이 여기서 유래합니다.

### 7.3 생성

압축 빛은 매개 변수 하향 변환(레슨 12)을 이용해 생성됩니다 — 구체적으로 문턱값 이하의 축퇴 광학 매개 증폭기(Optical Parametric Amplifier, OPA). $\chi^{(2)}$ 과정은 광자 쌍들을 상관시켜 한 직교 성분의 요동을 감소시킵니다.

최신 압축 기술: SQL 아래 ~15 dB (압축된 직교 성분의 잡음이 진공 잡음 수준의 $10^{-1.5} \approx 3\%$).

### 7.4 응용: 중력파 검출

LIGO(레이저 간섭 중력파 관측소)는 암포트(dark port)에 압축 진공 상태를 주입하여 산탄 잡음(shot noise) 한계를 넘어 감도를 향상시킵니다. 2019년 이후, 압축 빛은 LIGO에서 일상적으로 사용되어 쌍성 중성자별 합병에 대한 검출 범위를 ~50% 향상시켰습니다.

> **비유**: 테이블 위 구슬의 위치를 그림자를 보고 측정한다고 상상해봅시다. 표준 양자 한계는 약간 흐릿한 그림자와 같습니다 — 구슬을 조명하는 광자들의 파동성으로 인해 측정에 항상 어느 정도의 불확정성이 있습니다. 압축은 조명 빔을 재형성하여 수평 방향으로 더 선명한(대신 수직으로 더 흐릿한) 그림자를 만드는 것과 같습니다. 수평 위치만 신경 쓴다면, 압축은 표준 한계가 허용하는 것보다 더 정밀한 측정을 제공합니다.

---

## 8. 얽힘과 벨 상태

### 8.1 양자 얽힘

두 양자 시스템은 결합 시스템의 양자 상태가 개별 상태의 곱으로 쓰일 수 없을 때 **얽혀(Entangled)** 있다고 합니다. 얽힌 광자들의 경우:

$$|\Psi\rangle \neq |\psi_A\rangle \otimes |\psi_B\rangle$$

얽힌 광자들은 어떤 고전 시스템도 만들어낼 수 없는 상관관계를 나타냅니다. 아인슈타인은 이것을 유명하게 "원격 유령 작용(spooky action at a distance)"이라고 불렀습니다.

### 8.2 벨 상태

네 개의 **벨 상태(Bell States)**는 두 큐비트(예: 두 광자의 편광)의 최대 얽힘 상태입니다:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|HH\rangle + |VV\rangle)$$

$$|\Phi^-\rangle = \frac{1}{\sqrt{2}}(|HH\rangle - |VV\rangle)$$

$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|HV\rangle + |VH\rangle)$$

$$|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|HV\rangle - |VH\rangle)$$

여기서 $|H\rangle$과 $|V\rangle$는 수평 및 수직 편광입니다.

### 8.3 SPDC를 통한 생성

Type II SPDC는 자연스럽게 상태 $|\Psi^-\rangle$를 생성합니다: 신호와 공전(idler) 광자는 직교 편광을 가지며, 그 정확한 할당은 측정까지 양자역학적으로 정의되지 않습니다.

### 8.4 벨의 정리와 실험적 검증

벨 부등식(Bell's Inequality, 1964)은 정량적 검증을 제공합니다: 임의의 국소 숨은 변수 이론(local hidden variable theory)은 다음을 만족합니다:

$$|S| \leq 2 \quad (\text{CHSH 부등식})$$

양자역학은 $|S| = 2\sqrt{2} \approx 2.83$까지의 위반을 예측합니다.

아스펙트(Aspect, 1982)의 실험과, 더 최근에는 헨센(Hensen) 등(2015)과 다른 연구자들의 허점 없는 검증(loophole-free tests)이 $|S| > 2$를 결정적으로 입증하여 국소 숨은 변수 이론을 배제했습니다. 아스펙트(Aspect), 클로저(Clauser), 자일링거(Zeilinger)는 이 실험들로 2022년 노벨 물리학상을 수상했습니다.

---

## 9. 양자 키 분배 (BB84)

### 9.1 아이디어

BB84(베넷 & 브라사르(Bennett & Brassard), 1984)는 계산적 복잡도가 아닌 양자역학에 의해 보안이 보장되는, 두 당사자(앨리스와 밥) 사이에 비밀 암호화 키를 분배하는 프로토콜입니다.

### 9.2 프로토콜

1. **앨리스(Alice)**가 두 기저에서 네 가지 편광 상태 중 하나에 무작위로 준비된 단일 광자를 전송합니다:
   - **직교 기저(Rectilinear Basis)** ($+$): $|H\rangle$ = "0", $|V\rangle$ = "1"
   - **대각 기저(Diagonal Basis)** ($\times$): $|D\rangle = |+45°\rangle$ = "0", $|A\rangle = |-45°\rangle$ = "1"

2. **밥(Bob)**이 각 광자에 대해 무작위로 측정 기저($+$ 또는 $\times$)를 선택합니다.

3. **체로 거르기(Sifting)**: 앨리스와 밥이 공개적으로 기저 선택을 비교합니다(결과는 비교하지 않음). 같은 기저를 사용한 비트만 유지합니다(~비트의 50%).

4. **오류 추정(Error Estimation)**: 공유 비트의 일부를 희생하여 오류율을 추정합니다. 오류율이 너무 높으면 도청자(이브(Eve))가 감지됩니다.

5. **프라이버시 증폭(Privacy Amplification)**: 더 짧지만 완전히 비밀인 키를 증류하기 위한 후처리.

### 9.3 보안

이브(Eve)가 광자를 가로채고 측정하면 앨리스가 사용한 기저를 추측해야 합니다. 틀리게 추측하면, 그녀의 측정이 광자 상태를 교란시켜 앨리스와 밥이 감지할 수 있는 오류를 도입합니다. **복제 불가 정리(No-Cloning Theorem)**는 이브가 교란 없이 양자 상태를 복사할 수 없음을 보장합니다.

### 9.4 실용적 양자 키 분배

상업적 양자 키 분배 시스템은 광섬유(중계기 없이 최대 ~100 km)와 자유 공간(2017년 중국의 미키우스(Micius) 위성에 의해 1,200 km 이상에서 위성 양자 키 분배 시연)을 통해 작동합니다.

키 속도: 대도시 거리에서 일반적으로 kbit/s에서 Mbit/s.

---

## 10. 광자를 이용한 양자 컴퓨팅

### 10.1 광자 큐비트

광자는 여러 자유도로 큐비트(qubit)를 인코딩할 수 있습니다:
- **편광(Polarization)**: $|H\rangle$과 $|V\rangle$를 $|0\rangle$과 $|1\rangle$로
- **경로 (이중 궤도(Dual-Rail))**: 모드 $a$ 또는 모드 $b$의 광자
- **시간 빈(Time-Bin)**: 이른 또는 늦은 도착 시간
- **주파수(Frequency)**: 서로 다른 스펙트럼 모드

### 10.2 선형 광학 양자 컴퓨팅 (LOQC)

닐(Knill), 라플람(Laflamme), 밀번(Milburn, 2001)은 다음만을 사용하여 범용 양자 컴퓨팅이 가능함을 보였습니다:
- 단일 광자 소스
- 선형 광학 소자(빔 분리기, 위상 이동기)
- 광자 검출기
- 고전적 피드포워드

핵심 통찰: 광자 검출(비선형 연산)이 **측정 유도 비선형성(Measurement-Induced Nonlinearity)**을 통해 범용 게이트에 필요한 비선형성을 제공합니다.

### 10.3 보손 샘플링

보손 샘플링(Boson Sampling, 아론슨(Aaronson) & 아르키포프(Arkhipov), 2011)은 광자 시스템이 효율적으로 수행할 수 있지만 고전 컴퓨터는 (아마도) 할 수 없는 계산 과제입니다. $m$-모드 간섭계에 $n$개의 동일한 광자를 입력하고 출력 분포를 샘플링합니다. 이것은 양자 계산 우위(quantum computational advantage)를 시연하는 첫 번째 후보 중 하나였습니다.

2020년, 중국의 실험 "지우장(Jiuzhang)"은 76개의 검출된 광자로 보손 샘플링을 시연했습니다 — 고전 슈퍼컴퓨터로는 $10^{10}$년이 걸릴 것으로 추정되는 과제입니다.

### 10.4 광자 양자 우위

광자 큐비트의 장점:
- 실온에서 결어긋남(decoherence) 없음 (광자는 환경과 상호작용하지 않음)
- 양자 통신의 자연적 전달자 (광섬유 네트워크와 호환)
- 고속 동작 (원리적으로 THz 클럭 속도)

단점:
- 광자 손실 (확률적 검출)
- 결정론적 광자-광자 상호작용의 부재
- 초고효율 단일 광자 소스와 검출기 필요

PsiQuantum, Xanadu, QuiX 같은 기업들이 광자 양자 프로세서를 구축하고 있습니다.

---

## 11. Python 예제

### 11.1 광자 통계 비교

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial

def poisson_dist(n, n_mean):
    """Poisson distribution: coherent light (laser)."""
    return n_mean**n * np.exp(-n_mean) / factorial(n, exact=False)

def bose_einstein_dist(n, n_mean):
    """Bose-Einstein distribution: thermal/chaotic light."""
    return n_mean**n / (1 + n_mean)**(n + 1)

def fock_dist(n, n_exact):
    """Fock (number) state: definite photon number."""
    return np.where(n == n_exact, 1.0, 0.0)

# Compare distributions for mean photon number = 5
n_mean = 5
n = np.arange(0, 20)

P_coherent = poisson_dist(n, n_mean)
P_thermal = bose_einstein_dist(n, n_mean)
P_fock = fock_dist(n, n_mean)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Coherent (Poisson)
axes[0].bar(n, P_coherent, color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_title(f'Coherent State |α|² = {n_mean}\n(Poissonian, F = 1)', fontsize=11)
axes[0].set_xlabel('Photon number n')
axes[0].set_ylabel('P(n)')
var_c = n_mean
axes[0].text(0.95, 0.95, f'⟨n⟩ = {n_mean}\nVar = {var_c}\nF = {var_c/n_mean:.1f}',
             transform=axes[0].transAxes, va='top', ha='right',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Thermal (Bose-Einstein)
axes[1].bar(n, P_thermal, color='indianred', alpha=0.8, edgecolor='black')
axes[1].set_title(f'Thermal Light ⟨n⟩ = {n_mean}\n(Super-Poissonian, F > 1)', fontsize=11)
axes[1].set_xlabel('Photon number n')
var_t = n_mean**2 + n_mean
axes[1].text(0.95, 0.95, f'⟨n⟩ = {n_mean}\nVar = {var_t}\nF = {var_t/n_mean:.1f}',
             transform=axes[1].transAxes, va='top', ha='right',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

# Fock state
axes[2].bar(n, P_fock, color='forestgreen', alpha=0.8, edgecolor='black')
axes[2].set_title(f'Fock State |n={n_mean}⟩\n(Sub-Poissonian, F = 0)', fontsize=11)
axes[2].set_xlabel('Photon number n')
axes[2].text(0.95, 0.95, f'⟨n⟩ = {n_mean}\nVar = 0\nF = 0',
             transform=axes[2].transAxes, va='top', ha='right',
             fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow'))

for ax in axes:
    ax.set_xlim(-0.5, 19.5)
    ax.set_ylim(0, max(P_thermal.max(), P_coherent.max(), 1.05) * 1.1)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Photon Number Distributions: Three Types of Light', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('photon_statistics.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.2 홍-오우-만델 딥 시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

def hom_dip(tau, tau_c):
    """
    Simulate the Hong-Ou-Mandel dip.

    The coincidence rate at a 50:50 beam splitter drops to zero
    when two identical photons arrive simultaneously (τ=0).
    The dip width is determined by the photon coherence time τ_c.

    For distinguishable photons (|τ| >> τ_c), the coincidence rate
    is 0.5 (classical: each photon independently chooses an output port).
    For indistinguishable photons (τ → 0), quantum interference causes
    both photons to exit the same port, and coincidences vanish.
    """
    # Gaussian single-photon wavepackets
    # Overlap integral determines the visibility
    visibility = np.exp(-tau**2 / (2 * tau_c**2))

    # Classical coincidence rate = 0.5 (random 50:50 choice for each photon)
    # Quantum coincidence rate drops by the overlap squared
    coincidence = 0.5 * (1 - visibility**2)

    return coincidence

# Photon coherence time (related to bandwidth)
tau_c = 1e-12  # 1 ps (typical for SPDC photons with ~1 nm bandwidth)

tau = np.linspace(-5e-12, 5e-12, 1000)
R_coin = hom_dip(tau, tau_c)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(tau * 1e12, R_coin, 'b-', linewidth=2.5)
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5,
           label='Classical limit (distinguishable photons)')
ax.axhline(y=0.25, color='orange', linestyle=':', linewidth=1.5,
           label='50% visibility (partial indistinguishability)')

# Fill the dip region
ax.fill_between(tau * 1e12, R_coin, 0.5, alpha=0.15, color='blue')

ax.set_xlabel('Relative delay τ (ps)', fontsize=12)
ax.set_ylabel('Coincidence rate (normalized)', fontsize=12)
ax.set_title('Hong-Ou-Mandel Dip', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.02, 0.6)

# Annotate
ax.annotate('Quantum interference:\nboth photons exit\nthe same port',
            xy=(0, 0), xytext=(2, 0.15),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue'))

plt.tight_layout()
plt.savefig('hom_dip.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 11.3 BB84 프로토콜 시뮬레이션

```python
import numpy as np

def bb84_simulation(n_bits, eve_present=False):
    """
    Simulate the BB84 quantum key distribution protocol.

    Alice prepares random polarization states, Bob measures in
    random bases. When their bases match, they share a bit.
    If Eve intercepts and measures (intercept-resend attack),
    she introduces ~25% errors in the sifted key — detectable
    by Alice and Bob through error rate estimation.
    """
    # Step 1: Alice prepares random bits and bases
    alice_bits = np.random.randint(0, 2, n_bits)
    alice_bases = np.random.randint(0, 2, n_bits)  # 0 = rectilinear, 1 = diagonal

    # Step 2: Eve intercepts (if present)
    if eve_present:
        eve_bases = np.random.randint(0, 2, n_bits)
        # Eve measures in her random basis
        # If Eve's basis matches Alice's, she gets the correct bit
        # If not, she gets a random result and disturbs the state
        eve_bits = np.where(
            eve_bases == alice_bases,
            alice_bits,  # Correct measurement
            np.random.randint(0, 2, n_bits)  # Random result
        )
        # Eve re-sends in her basis — this is the "resend" part
        # The state Eve sends may not match Alice's original state
        transmitted_bits = eve_bits
        transmitted_bases = eve_bases  # Eve's measurement collapses to her basis
    else:
        transmitted_bits = alice_bits
        transmitted_bases = alice_bases

    # Step 3: Bob measures in random bases
    bob_bases = np.random.randint(0, 2, n_bits)

    if eve_present:
        # Bob's result depends on whether his basis matches Eve's resend basis
        bob_bits = np.where(
            bob_bases == transmitted_bases,
            transmitted_bits,
            np.random.randint(0, 2, n_bits)
        )
    else:
        bob_bits = np.where(
            bob_bases == alice_bases,
            alice_bits,
            np.random.randint(0, 2, n_bits)
        )

    # Step 4: Sifting — keep only matching bases
    matching_bases = alice_bases == bob_bases
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]

    # Step 5: Error estimation
    n_sifted = len(sifted_alice)
    if n_sifted > 0:
        errors = np.sum(sifted_alice != sifted_bob)
        error_rate = errors / n_sifted
    else:
        error_rate = 0

    return {
        'n_sent': n_bits,
        'n_sifted': n_sifted,
        'sifting_rate': n_sifted / n_bits,
        'error_rate': error_rate,
        'key_bits': sifted_alice[:20]  # First 20 bits of the key
    }

# Run simulations
np.random.seed(42)
n_bits = 10000

print("=" * 60)
print("BB84 Quantum Key Distribution Simulation")
print("=" * 60)

# Without eavesdropper
result_no_eve = bb84_simulation(n_bits, eve_present=False)
print(f"\n--- Without Eavesdropper ---")
print(f"Bits sent: {result_no_eve['n_sent']}")
print(f"Sifted key length: {result_no_eve['n_sifted']} ({result_no_eve['sifting_rate']:.1%})")
print(f"Error rate: {result_no_eve['error_rate']:.4f}")
print(f"First 20 key bits: {result_no_eve['key_bits']}")

# With eavesdropper
result_eve = bb84_simulation(n_bits, eve_present=True)
print(f"\n--- With Eavesdropper (intercept-resend) ---")
print(f"Bits sent: {result_eve['n_sent']}")
print(f"Sifted key length: {result_eve['n_sifted']} ({result_eve['sifting_rate']:.1%})")
print(f"Error rate: {result_eve['error_rate']:.4f}")
print(f"Expected error rate from Eve: ~0.25 (25%)")
print(f"First 20 key bits: {result_eve['key_bits']}")

# Security check
threshold = 0.11  # Typical threshold for BB84
print(f"\n--- Security Decision ---")
print(f"Error threshold: {threshold:.0%}")
for label, result in [("No Eve", result_no_eve), ("With Eve", result_eve)]:
    secure = "SECURE" if result['error_rate'] < threshold else "ABORT (eavesdropper detected!)"
    print(f"  {label}: error = {result['error_rate']:.4f} → {secure}")
```

---

## 12. 요약

| 개념 | 핵심 공식 / 아이디어 |
|---------|--------------------|
| 장 양자화 | $\hat{H} = \hbar\omega(\hat{n} + 1/2)$; 에너지는 이산적 |
| 포크 상태 | $\|n\rangle$: 정확히 $n$개의 광자; $\Delta n = 0$ |
| 결맞음 상태 | $\hat{a}\|\alpha\rangle = \alpha\|\alpha\rangle$; 포아송 통계; 고전적 유사 |
| 광자 통계 | 포아송 ($F=1$, 레이저), 초포아송 ($F>1$, 열 빛), 아포아송 ($F<1$, 양자) |
| $g^{(2)}(0)$ | 1 (결맞음), 2 (열 빛), 0 (단일 광자) |
| 단일 광자 소스 | 양자점, NV 센터, 예고 SPDC |
| HOM 효과 | 두 동일한 광자가 50:50 빔 분리기에서 항상 같은 출사구로 나옴 |
| 압축 상태 | $\Delta X_1 < 1/2$ (SQL 이하), $\Delta X_1 \Delta X_2 = 1/4$ |
| 벨 상태 | 최대 얽힘: $\|\Phi^\pm\rangle$, $\|\Psi^\pm\rangle$ |
| BB84 | 양자 키 분배 프로토콜: 복제 불가 정리에 의한 보안 |
| 광자 양자 컴퓨팅 | 선형 광학 + 단일 광자 + 검출 = 범용 양자 컴퓨팅 |

---

## 13. 연습문제

### 연습문제 1: 결맞음 상태 성질

$\alpha = 3 + 4i$인 결맞음 상태 $|\alpha\rangle$에 대해:

(a) 평균 광자 수 $\bar{n}$을 계산하라.
(b) 광자 수 표준 편차를 계산하라.
(c) 정확히 25개의 광자를 검출할 확률은 얼마인가?
(d) $n = 0$에서 $n = 50$까지의 광자 수 분포를 그려라.

### 연습문제 2: $g^{(2)}$ 측정

어떤 광원이 핸버리 브라운-트위스 구성에서 다음과 같은 검출 기록을 생성합니다: $T = 100\,\text{s}$ 동안 검출기 A는 $N_A = 5000$회 계수, 검출기 B는 $N_B = 4800$회 계수를 기록하며, $\Delta t = 10\,\text{ns}$의 동시 검출 창 내에서 $N_{AB} = 150$회 동시 검출이 기록됩니다.

(a) $g^{(2)}(0) \approx \frac{N_{AB} T}{N_A N_B \Delta t}$를 추정하라.
(b) 이 광원은 고전적인가 양자적인가?
(c) 어떤 종류의 광원이 이 $g^{(2)}(0)$을 생성할 수 있는가?

### 연습문제 3: HOM 가시성

홍-오우-만델 실험에서 광자들은 $\lambda = 810\,\text{nm}$을 중심으로 $\Delta\lambda = 2\,\text{nm}$의 스펙트럼 대역폭을 가집니다.

(a) 결맞음 시간 $\tau_c = \lambda^2/(c\Delta\lambda)$을 계산하라.
(b) 어떤 시간 지연 $\tau$에서 HOM 가시성이 최고값의 $1/e$로 줄어드는가?
(c) 측정된 가시성이 92%라면, 광자 소스의 구별불가능성에 대해 무엇을 시사하는가?

### 연습문제 4: BB84 키 속도

BB84 양자 키 분배 시스템이 다음 매개변수로 1 GHz 펄스 속도로 작동합니다: 소스 효율 0.5, 50 km에 걸친 0.2 dB/km 광섬유 손실, 검출기 효율 10%, 암계수율(dark count rate) 게이트당 $10^{-6}$.

(a) 밥의 검출기에서의 광자 도착률을 계산하라.
(b) 원시 체로 거른 키 속도를 추정하라.
(c) 어떤 거리에서 키 속도가 0으로 떨어지는가 (암계수가 지배할 때)?

### 연습문제 5: 압축과 LIGO

LIGO의 팔 길이는 $L = 4\,\text{km}$이며, $\lambda = 1064\,\text{nm}$에서 순환 전력 $P = 750\,\text{kW}$로 작동합니다.

(a) 1 ms의 측정 시간에 대한 산탄 잡음 한계 변위 감도 $\delta x_{\text{SQL}}$은 얼마인가?
(b) 10 dB의 압축이 주입된다면, 개선된 감도는 얼마인가?
(c) LIGO가 측정한 가장 작은 중력파 변형은 $h \sim 10^{-21}$입니다. 이를 변위 $\delta L = hL/2$로 변환하고 (b)의 답과 비교하라.

---

## 14. 참고문헌

1. Fox, M. (2006). *Quantum Optics: An Introduction*. Oxford University Press. — 훌륭한 입문 교재.
2. Gerry, C. C., & Knight, P. L. (2023). *Introductory Quantum Optics* (2nd ed.). Cambridge University Press.
3. Walls, D. F., & Milburn, G. J. (2008). *Quantum Optics* (2nd ed.). Springer.
4. Glauber, R. J. (1963). "The quantum theory of optical coherence." *Physical Review*, 130, 2529.
5. Hong, C. K., Ou, Z. Y., & Mandel, L. (1987). "Measurement of subpicosecond time intervals between two photons by interference." *Physical Review Letters*, 59, 2044.
6. Bennett, C. H., & Brassard, G. (1984). "Quantum cryptography: Public key distribution and coin tossing." *Proceedings of IEEE International Conference on Computers, Systems and Signal Processing*, 175-179.
7. Aspect, A. (2022). Nobel Lecture: "From Einstein, Bohr and Bell to Quantum Technologies."

---

[← 이전: 12. 비선형 광학](12_Nonlinear_Optics.md) | [다음: 14. 전산 광학 →](14_Computational_Optics.md)
