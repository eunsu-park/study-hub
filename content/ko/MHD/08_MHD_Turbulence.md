# 8. MHD 난류

## 학습 목표

이 레슨을 마치면 다음을 할 수 있어야 합니다:

1. 유체역학 난류와 Kolmogorov K41 이론 복습하기
2. MHD 난류의 Iroshnikov-Kraichnan(IK) 이론 이해하기
3. Goldreich-Sridhar 임계 균형(critical balance) 이론과 비등방 캐스케이드(anisotropic cascade) 설명하기
4. Elsässer 변수와 MHD 난류에서의 역할 다루기
5. 에너지 캐스케이드, 간헐성(intermittency), 그리고 구조 함수(structure functions) 설명하기
6. 태양풍 난류 관측 분석하기
7. MHD 난류 스펙트럼의 수치 모델 구현하기

## 1. 유체역학 난류 복습

### 1.1 난류 문제

난류는 커피 젓기부터 은하 역학까지 자연에서 어디에나 존재합니다. 난류의 특징은 다음과 같습니다:

- **혼란스럽고 불규칙한 운동**: 예측 불가능하며, 초기 조건에 민감함
- **다중 스케일 구조**: 소용돌이 안의 소용돌이(Richardson 캐스케이드)
- **강화된 혼합**: 분자 확산을 훨씬 초과하는 수송
- **에너지 소산**: 작은 스케일에서 운동 에너지가 열로 변환

근본적인 어려움: **Navier-Stokes 방정식**이 비선형이어서, 난류를 해석적으로 다루기 어렵습니다.

$$\frac{\partial \mathbf{v}}{\partial t} + (\mathbf{v} \cdot \nabla) \mathbf{v} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{v} + \mathbf{f}$$

난류는 **에너지 주입 스케일** $L$(가장 큰 소용돌이)부터 **소산 스케일** $\eta$(Kolmogorov 스케일, 점성이 지배하는 곳)까지 거대한 스케일 범위를 포함합니다.

### 1.2 Kolmogorov 1941 (K41) 이론

Kolmogorov(1941)는 차원 분석과 보편성을 기반으로 난류의 통계 이론을 개발했습니다.

**핵심 가정:**

1. **통계적 등방성과 균질성**: 선호되는 방향이나 위치가 없음(국소적으로)
2. **스케일 분리**: $L \gg \eta$ (높은 Reynolds 수 $Re \gg 1$)
3. **관성 범위(Inertial range)**: 스케일 $\eta \ll \ell \ll L$에서 에너지가 소산 없이 전달됨
4. **국소 에너지 전달**: 에너지가 큰 스케일에서 작은 스케일로 캐스케이드

**에너지 캐스케이드:**

에너지는 큰 스케일에서 비율 $\epsilon$(단위 시간당 단위 질량당 에너지)으로 주입됩니다(예: 젓기에 의해). 이 에너지는 소용돌이 분해를 통해 작은 스케일로 **캐스케이드**되며, 결국 Kolmogorov 스케일에서 소산됩니다.

**차원 분석:**

관성 범위에서, 관련된 유일한 매개변수는 에너지 캐스케이드율 $\epsilon$과 스케일 $\ell$입니다. 스케일 $\ell$에서의 속도 변동은:

$$v_\ell \sim (\epsilon \ell)^{1/3}$$

스케일 $\ell$에서의 소용돌이 회전 시간(eddy turnover time)은:

$$\tau_\ell \sim \ell / v_\ell \sim \ell^{2/3} / \epsilon^{1/3}$$

**에너지 스펙트럼:**

단위 파수당 에너지는:

$$E(k) \sim \epsilon^{2/3} k^{-5/3}$$

여기서 $k \sim 1/\ell$은 파수입니다. 이것이 유명한 **Kolmogorov $-5/3$ 스펙트럼**입니다.

**속도 구조 함수:**

$p$차 구조 함수는:

$$S_p(\ell) = \langle |\delta v(\ell)|^p \rangle$$

여기서 $\delta v(\ell) = v(\mathbf{x} + \boldsymbol{\ell}) - v(\mathbf{x})$는 거리 $\ell$에 걸친 속도 증분입니다.

K41의 경우:

$$S_p(\ell) \sim (\epsilon \ell)^{p/3}$$

특히, $S_2(\ell) \sim \epsilon^{2/3} \ell^{2/3}$은 $k^{-5/3}$ 스펙트럼과 일치합니다(Fourier 변환).

### 1.3 K41의 한계

K41은 놀랍게 성공적이지만 한계가 있습니다:

- **등방성 가정**: 실제 난류는 종종 비등방성을 가짐(전단, 회전, 성층)
- **간헐성 무시**: 난류는 자기 유사하지 않음; 극단적 사건이 Gaussian 통계 예측보다 더 흔함
- **국소 캐스케이드**: 비국소 상호작용이 발생할 수 있음
- **결맞는 구조 무시**: 소용돌이, 충격파 등

이러한 한계에도 불구하고, K41은 비교를 위한 기준선을 제공합니다.

### 1.4 Reynolds 수와 Kolmogorov 스케일

**Reynolds 수**는 관성력 대 점성력의 비를 측정합니다:

$$Re = \frac{v L}{\nu}$$

난류의 경우, $Re \gg 1$입니다.

**Kolmogorov 스케일** $\eta$는 점성이 중요해지는 곳입니다:

$$\eta = \left( \frac{\nu^3}{\epsilon} \right)^{1/4}$$

스케일의 비는:

$$\frac{L}{\eta} \sim Re^{3/4}$$

대기 난류의 경우($Re \sim 10^6$), 이는 $L/\eta \sim 10^{4.5} \sim 30,000$를 제공합니다 — 거대한 범위입니다!

## 2. MHD 난류: 초기 이론들

### 2.1 왜 MHD 난류가 다른가?

자기유체역학에서, 자기장은 다음을 도입합니다:

1. **비등방성**: 장의 방향이 선호되는 방향임
2. **Alfvén 파동**: 전파하는 교란(유체역학에는 없음)
3. **감소된 비선형성**: Alfvén 파동 상호작용이 유체역학적 소용돌이 상호작용보다 약함
4. **자기 장력**: 수직 운동을 억제함

이러한 효과들이 난류 캐스케이드를 근본적으로 변화시킵니다.

### 2.2 Iroshnikov-Kraichnan (IK) 이론

Iroshnikov(1963)와 Kraichnan(1965)이 독립적으로 MHD 난류의 첫 번째 이론을 제안했습니다.

**핵심 아이디어:**

난류 소용돌이는 충돌하는 **Alfvén 파동 패킷**으로 이루어져 있습니다. Alfvén 파동은 평균 장 $\mathbf{B}_0$를 따라 Alfvén 속도 $v_A$로 전파합니다. 반대 방향으로 이동하는 파동 패킷이 충돌하여 약하게 상호작용합니다.

**충돌 시간:**

$\mathbf{B}_0$에 수직인 스케일 $\ell_\perp$의 소용돌이는 다음 시간에 걸쳐 상호작용합니다:

$$\tau_{coll} \sim \frac{\ell_\parallel}{v_A}$$

여기서 $\ell_\parallel$은 평행 스케일입니다. **등방성**을 가정하면($\ell_\parallel \sim \ell_\perp \sim \ell$):

$$\tau_{coll} \sim \frac{\ell}{v_A}$$

**캐스케이드 시간:**

소용돌이가 많은 충돌을 겪을 때 에너지가 캐스케이드됩니다. 필요한 충돌 횟수는:

$$N_{coll} \sim \frac{\tau_{eddy}}{\tau_{coll}}$$

여기서 $\tau_{eddy} \sim \ell / v_\ell$은 소용돌이 회전 시간입니다.

에너지는 $N_{coll} \sim 1$ 충돌이 발생했을 때 캐스케이드되지만, MHD에서는 상호작용이 약하므로 많은 충돌이 필요합니다:

$$N_{coll} \sim \left( \frac{v_A}{v_\ell} \right)^2$$

(제곱은 약한 상호작용 강도로부터 옵니다.)

그러면 캐스케이드 시간은:

$$\tau_{cascade} \sim N_{coll} \cdot \tau_{coll} \sim \frac{v_A}{v_\ell^2} \cdot \ell$$

**차원 분석:**

캐스케이드 시간을 소용돌이 회전 시간과 같다고 놓으면(에너지 전달):

$$\frac{\ell}{v_\ell} \sim \frac{v_A \ell}{v_\ell^2}$$

풀면:

$$v_\ell \sim v_A$$

이것은 단지 소용돌이가 Alfvén 속도로 움직인다는 것을 말하며, 그다지 유익하지 않습니다!

**올바른 IK 스케일링:**

에너지 캐스케이드율은:

$$\epsilon \sim \frac{v_\ell^2}{\tau_{cascade}} \sim \frac{v_\ell^4}{v_A \ell}$$

$v_\ell$에 대해 풀면:

$$v_\ell \sim (\epsilon v_A \ell)^{1/4}$$

에너지 스펙트럼은:

$$E(k) \sim (\epsilon v_A)^{1/2} k^{-3/2}$$

이것이 **Iroshnikov-Kraichnan $-3/2$ 스펙트럼**이며, Kolmogorov의 $-5/3$보다 얕습니다.

### 2.3 IK 이론의 문제점

수치 시뮬레이션과 관측은 다음을 보여주었습니다:

1. **IK는 등방성 가정**: 하지만 MHD 난류는 강하게 **비등방적**임($\mathbf{B}_0$를 따라 길게 늘어남)
2. **관측된 스펙트럼**: 종종 $-3/2$보다 $-5/3$에 더 가까움
3. **태양풍**: 관성 범위에서 $k^{-5/3}$를 보임

IK 이론은 좋은 첫 단계였지만 MHD 난류의 본질적인 비등방성을 포착하지 못했습니다.

## 3. Goldreich-Sridhar 임계 균형 이론

### 3.1 MHD 난류의 비등방성

관측과 시뮬레이션은 MHD 난류가 **비등방적**임을 보여주었습니다:

- **수직 캐스케이드**: 소용돌이가 $\mathbf{B}_0$에 $\perp$인 작은 스케일로 캐스케이드
- **평행 연장**: 소용돌이가 $\mathbf{B}_0$를 따라 연장됨

Goldreich & Sridhar(1995, GS95)는 이 비등방성을 통합하는 이론을 제안했습니다.

**핵심 아이디어: 임계 균형(Critical balance)**

각 스케일 $\ell_\perp$(수직 크기)에서, **비선형 캐스케이드 시간**은 **Alfvén 파동 주기**와 비슷합니다:

$$\tau_{nl} \sim \tau_A$$

여기서:

$$\tau_{nl} \sim \frac{\ell_\perp}{v_{\ell_\perp}}$$

는 소용돌이 회전 시간이고:

$$\tau_A \sim \frac{\ell_\parallel}{v_A}$$

는 평행 방향을 따라 Alfvén 파동이 횡단하는 시간입니다.

### 3.2 GS95 스케일링 유도

**임계 균형 조건:**

$$\frac{\ell_\perp}{v_{\ell_\perp}} \sim \frac{\ell_\parallel}{v_A}$$

**$\perp$ 방향의 Kolmogorov형 캐스케이드:**

수직 방향에서 Kolmogorov 캐스케이드를 가정:

$$v_{\ell_\perp} \sim (\epsilon \ell_\perp)^{1/3}$$

**$\ell_\parallel$과 $\ell_\perp$ 관계:**

임계 균형으로부터:

$$\ell_\parallel \sim \frac{v_A \ell_\perp}{v_{\ell_\perp}} \sim \frac{v_A \ell_\perp}{(\epsilon \ell_\perp)^{1/3}}$$

단순화:

$$\ell_\parallel \sim v_A \ell_\perp^{2/3} / \epsilon^{1/3}$$

또는 외부 스케일 $L$로 정규화하면:

$$\frac{\ell_\parallel}{L} \sim \left( \frac{\ell_\perp}{L} \right)^{2/3}$$

(외부 스케일에서 $v_A \sim (\epsilon L)^{1/3}$을 가정하면 일관됩니다).

파수 $k_\parallel \sim 1/\ell_\parallel$, $k_\perp \sim 1/\ell_\perp$ 관점에서:

$$k_\parallel \propto k_\perp^{2/3}$$

**비등방 캐스케이드:**

소용돌이는 작은 $\ell_\perp$으로 캐스케이드됨에 따라 $\mathbf{B}_0$를 따라 점점 더 연장됩니다:

$$\frac{\ell_\parallel}{\ell_\perp} \propto \ell_\perp^{-1/3} \to \infty \quad \text{as } \ell_\perp \to 0$$

작은 스케일에서, 소용돌이는 $\ell_\parallel \gg \ell_\perp$인 리본 모양입니다.

**수직 에너지 스펙트럼:**

수직 방향의 에너지 스펙트럼은:

$$E(k_\perp) \propto k_\perp^{-5/3}$$

Kolmogorov와 같습니다! 캐스케이드는 $\perp$ 방향에서 Kolmogorov형이지만, 고도로 비등방적입니다.

**평행 스펙트럼:**

비등방성 관계 $k_\parallel \propto k_\perp^{2/3}$ 때문에, 평행 스펙트럼이 더 가파릅니다.

### 3.3 물리적 해석

**왜 임계 균형인가?**

$\tau_{nl} \ll \tau_A$이면, Alfvén 파동이 전파할 시간이 있기 전에 소용돌이가 빠르게 캐스케이드 — 캐스케이드가 거의 유체역학적일 것입니다(Kolmogorov).

$\tau_{nl} \gg \tau_A$이면, 소용돌이가 진화하기 전에 Alfvén 파동이 여러 번 전파 — 에너지가 파동에 갇혀서 효과적으로 캐스케이드되지 않습니다.

**임계 균형**은 두 과정이 동등하게 중요한 한계 불안정 상태로, 효율적인 에너지 전달을 허용합니다.

**Alfvén 난류:**

GS95는 난류가 Alfvén 파동(Elsässer 모드)으로 이루어져 있다고 가정하며, 이는 곧 논의할 것입니다.

### 3.4 관측적 지지

**태양풍:**

- 수직 스펙트럼: $E(k_\perp) \propto k_\perp^{-5/3}$ (GS95와 일치)
- 비등방성: 변동이 $\mathbf{B}_0$를 따라 연장됨
- 임계 균형: 관측은 $\tau_{nl} \sim \tau_A$를 시사함

**시뮬레이션:**

수치 MHD 시뮬레이션은 다음을 확인합니다:
- $k_\parallel \propto k_\perp^{2/3}$인 비등방 캐스케이드
- 수직 $k^{-5/3}$ 스펙트럼
- 스케일에 걸쳐 유지되는 임계 균형

GS95는 이제 강한 MHD 난류의 **표준 모델**입니다.

## 4. Elsässer 변수

### 4.1 정의

Elsässer(1950)는 비압축성, 일정 밀도 MHD에 대해 MHD 방정식을 대칭화하는 변수를 도입했습니다.

정의:

$$\mathbf{z}^+ = \mathbf{v} + \frac{\mathbf{B}}{\sqrt{\mu_0 \rho}}$$

$$\mathbf{z}^- = \mathbf{v} - \frac{\mathbf{B}}{\sqrt{\mu_0 \rho}}$$

이들은 **반대 방향으로 전파하는 Alfvén 파동**을 나타냅니다:

- $\mathbf{z}^+$: $+\mathbf{B}_0$ 방향으로 전파하는 Alfvén 파동
- $\mathbf{z}^-$: $-\mathbf{B}_0$ 방향으로 전파하는 Alfvén 파동

**Elsässer 변수로 표현한 속도와 자기장:**

$$\mathbf{v} = \frac{\mathbf{z}^+ + \mathbf{z}^-}{2}$$

$$\frac{\mathbf{B}}{\sqrt{\mu_0 \rho}} = \frac{\mathbf{z}^+ - \mathbf{z}^-}{2}$$

### 4.2 Elsässer 형식의 MHD 방정식

균일 밀도를 가진 비압축성 MHD의 경우, 방정식은:

$$\frac{\partial \mathbf{z}^+}{\partial t} + (\mathbf{z}^- \cdot \nabla) \mathbf{z}^+ = -\nabla P^+ + \nu \nabla^2 \mathbf{z}^+ + \eta \nabla^2 \mathbf{z}^+$$

$$\frac{\partial \mathbf{z}^-}{\partial t} + (\mathbf{z}^+ \cdot \nabla) \mathbf{z}^- = -\nabla P^- + \nu \nabla^2 \mathbf{z}^- + \eta \nabla^2 \mathbf{z}^-$$

$$\nabla \cdot \mathbf{z}^+ = 0, \quad \nabla \cdot \mathbf{z}^- = 0$$

여기서 $P^\pm$는 일반화된 압력입니다.

**핵심 관찰:**

$\mathbf{z}^+$ 방정식의 비선형 항은 $\mathbf{z}^-$를 포함하고, 그 반대도 마찬가지입니다. 이것은 **$\mathbf{z}^+$와 $\mathbf{z}^-$가 서로 상호작용**하며, 자신들끼리는 상호작용하지 않음을 보여줍니다.

물리적으로: 반대 방향으로 전파하는 Alfvén 파동은 충돌하여 상호작용하며; 같은 방향으로 전파하는 파동은 그렇지 않습니다.

### 4.3 균형 대 불균형 난류

**균형 난류(Balanced turbulence):**

$|\mathbf{z}^+| \approx |\mathbf{z}^-|$이면, 난류는 **균형**입니다. 이것은 GS95에서 가정한 경우입니다.

**불균형 난류(Imbalanced turbulence):**

$|\mathbf{z}^+| \neq |\mathbf{z}^-|$이면, 난류는 **불균형**입니다. 예를 들어, $|\mathbf{z}^+| \gg |\mathbf{z}^-|$이면:

- $\mathbf{z}^+$가 에너지를 지배
- $\mathbf{z}^-$는 약한 소수 집단
- 상호작용률이 감소(충돌이 적음)

**태양풍:**

태양풍은 일반적으로 불균형입니다:

$$\frac{E(z^-)}{E(z^+)} \sim 0.2\text{–}0.5$$

이 불균형은 캐스케이드율에 영향을 미치며 다른 스케일링으로 이어질 수 있습니다.

### 4.4 Elsässer 변수의 에너지

총 에너지 밀도는:

$$E = \frac{1}{2} \rho v^2 + \frac{B^2}{2\mu_0} = \frac{\rho}{4} \left( |\mathbf{z}^+|^2 + |\mathbf{z}^-|^2 \right)$$

각 Elsässer 성분의 에너지:

$$E^+ = \frac{\rho}{4} |\mathbf{z}^+|^2, \quad E^- = \frac{\rho}{4} |\mathbf{z}^-|^2$$

균형 난류에서, $E^+ \approx E^-$입니다. 불균형 난류에서, 하나가 지배합니다.

## 5. 에너지 캐스케이드와 간헐성

### 5.1 순방향 대 역방향 캐스케이드

3D 유체역학에서, 에너지는 큰 스케일에서 작은 스케일로 **직접** 캐스케이드됩니다(순방향 캐스케이드).

2D 유체역학에서, 에너지는 작은 스케일에서 큰 스케일로 **역방향** 캐스케이드되고, enstrophy는 순방향 캐스케이드됩니다. 이것은 2D에서 에너지와 enstrophy 둘 다의 보존 때문입니다.

**MHD:**

3D MHD에는 보존량이 있습니다:
- **총 에너지**: $E = E_{kin} + E_{mag}$
- **교차 helicity**: $H_c = \int \mathbf{v} \cdot \mathbf{B} \, dV$
- **자기 helicity**: $H_m = \int \mathbf{A} \cdot \mathbf{B} \, dV$ (특정 경우)

**순방향 캐스케이드:**

대부분의 경우, 에너지는 3D MHD에서 유체역학과 유사하게 **순방향**(큰 스케일에서 작은 스케일로) 캐스케이드됩니다.

**역방향 캐스케이드:**

자기 helicity가 존재하고 보존되면, 에너지가 여전히 순방향 캐스케이드되는 동안 자기 helicity의 **역방향 캐스케이드**가 큰 스케일로 있을 수 있습니다. 이것은 dynamos에서 관련이 있습니다(Lesson 9).

### 5.2 간헐성

**간헐성이란 무엇인가?**

간헐성은 자기 유사 스케일링으로부터의 이탈을 의미합니다. 실제 난류에서:
- 강렬하고 국소화된 구조(전류 시트, 소용돌이 필라멘트)가 존재함
- 소산이 작은 영역에 집중됨
- 구조 함수가 비정상 스케일링을 보임: $S_p(\ell) \propto \ell^{\zeta_p}$이고 $\zeta_p \neq p/3$

**다중 프랙탈 모델:**

소산장은 다중 프랙탈로, 특이점의 스펙트럼으로 특징지어집니다. 다른 영역이 다른 국소 스케일링 지수를 가집니다.

**결과:**

- **비Gaussian 통계**: 속도 증분의 PDF가 확장된 꼬리를 가짐
- **비정상 스케일링**: K41 예측으로부터의 이탈
- **결맞는 구조**: 전류 시트, 자기 플럭스 튜브, 충격파

간헐성은 비등방성과 전류 시트 형성 때문에 MHD에서 유체역학 난류보다 더 두드러집니다.

### 5.3 구조 함수

$p$차 구조 함수는:

$$S_p(\ell) = \langle |\delta z(\ell)|^p \rangle$$

여기서 $\delta z(\ell) = z(\mathbf{x} + \boldsymbol{\ell}) - z(\mathbf{x})$는 Elsässer 변수 증분입니다.

**K41 예측:**

$$S_p(\ell) \propto \ell^{p/3}$$

**간헐적 난류:**

$$S_p(\ell) \propto \ell^{\zeta_p}$$

여기서 $\zeta_p$는 $p/3$로부터 이탈하며, 특히 큰 $p$(드물고 강렬한 사건)에서 그렇습니다.

**측정:**

구조 함수는 우주선 데이터(태양풍) 또는 시뮬레이션 출력으로부터 계산됩니다. 이들은 캐스케이드와 간헐성에 대한 통찰을 제공합니다.

### 5.4 Python 예제: 구조 함수 스케일링

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic turbulent velocity field
# (Simplified: assume a power-law spectrum)

np.random.seed(42)

# Spatial grid
N = 512
L = 1.0
x = np.linspace(0, L, N, endpoint=False)

# Wavenumber
k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
k[0] = 1e-10  # Avoid division by zero

# Power spectrum: E(k) ~ k^{-5/3}
# Navier-Stokes 방정식을 발전시키는 대신 K41 스펙트럼을 직접 지정하는
# 이유: 구조 함수(structure function)에 대한 깨끗한 테스트 케이스를
# 얻기 위해서다; 여기서 목표는 ζ_p를 측정하는 것이지 동역학을
# 시뮬레이션하는 것이 아니다.
P_k = k**(-5/3)
P_k[0] = 0  # Zero mean

# Random phases
# 균일하게 랜덤한 위상(phase)을 부여하면 합성 장이 통계적으로 균질하고
# 등방적이 됨 — K41 이론이 바탕으로 하는 동일한 가정이다 —
# 따라서 우리의 측정에서 ζ_p의 p/3으로부터의 이탈은 물리적 간헐성이
# 아니라 유한 샘플 노이즈에 의한 것이다.
phase = np.exp(2j * np.pi * np.random.rand(N))

# Velocity in Fourier space
v_k = np.sqrt(P_k) * phase

# Velocity in real space
v = np.fft.ifft(v_k).real

# Normalize
v = v / np.std(v)

# Compute structure functions
# 지수 간격의 지연(lag)을 사용하는 이유: 관성 범위(작은 ℓ)와 에너지 포함
# 스케일(큰 ℓ) 모두를 포착하기 위해서다 — 선형 간격이면 자기 유사성이
# 성립하는 관성 범위에 샘플 대부분을 낭비하게 된다.
lags = np.logspace(np.log10(L/N), np.log10(L/4), 30)
orders = [1, 2, 3, 4, 5, 6]
S_p = {p: [] for p in orders}

for lag in lags:
    lag_idx = int(lag / (L/N))
    if lag_idx == 0:
        lag_idx = 1
    delta_v = v[lag_idx:] - v[:-lag_idx]

    for p in orders:
        # p제곱 전에 절댓값을 취하는 것이 필수적인 이유: 그렇지 않으면
        # 홀수 차수 S_p는 대칭에 의해 0이 되고(장의 평균이 0이므로),
        # 속도 증분 PDF에 관한 정보를 전혀 얻을 수 없게 된다.
        S_p[p].append(np.mean(np.abs(delta_v)**p))

# Convert to arrays
for p in orders:
    S_p[p] = np.array(S_p[p])

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Structure functions
ax = axes[0]
colors = plt.cm.viridis(np.linspace(0, 1, len(orders)))
for i, p in enumerate(orders):
    ax.loglog(lags, S_p[p], 'o-', label=f'$S_{p}$', color=colors[i], markersize=5)

# K41 predictions
for i, p in enumerate(orders):
    K41_slope = p / 3
    S_K41 = 0.1 * lags**K41_slope  # Arbitrary normalization
    ax.loglog(lags, S_K41, '--', color=colors[i], alpha=0.5)

ax.set_xlabel('Lag $\\ell$', fontsize=13)
ax.set_ylabel('Structure function $S_p(\\ell)$', fontsize=13)
ax.set_title('Structure Functions (K41 Scaling)', fontsize=15)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Panel 2: Scaling exponents
ax = axes[1]

# Fit power-law to extract exponents
# 로그-로그 공간에서의 선형 피팅이 직접 스케일링 지수 ζ_p를 제공한다;
# 높은 p에서 K41 선 p/3으로부터의 ζ_p 이탈이 간헐성(intermittency)을
# 진단한다 — 희귀한 강한 사건(전류 시트)이 작은 부피를 차지하더라도
# 높은 차수 모멘트에 불균형하게 기여하기 때문이다.
zeta_p = []
for p in orders:
    # Fit log(S_p) vs log(ell)
    coeffs = np.polyfit(np.log10(lags), np.log10(S_p[p]), 1)
    zeta_p.append(coeffs[0])

zeta_K41 = np.array(orders) / 3

ax.plot(orders, zeta_p, 'o-', label='Measured $\\zeta_p$', markersize=8, linewidth=2, color='blue')
ax.plot(orders, zeta_K41, '--', label='K41: $\\zeta_p = p/3$', linewidth=2, color='red')

ax.set_xlabel('Order $p$', fontsize=13)
ax.set_ylabel('Scaling exponent $\\zeta_p$', fontsize=13)
ax.set_title('Scaling Exponents: K41 vs Measured', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('structure_functions_K41.png', dpi=150)
plt.show()

print("Scaling Exponents:")
print(f"{'p':>5} {'ζ_p (measured)':>20} {'ζ_p (K41 = p/3)':>20}")
print("-" * 50)
for p, zeta, zeta_k41 in zip(orders, zeta_p, zeta_K41):
    print(f"{p:>5} {zeta:>20.3f} {zeta_k41:>20.3f}")
```

## 6. 태양풍 난류

### 6.1 난류 실험실로서의 태양풍

**태양풍**은 태양으로부터 흐르는 초음속, 초Alfvén 플라즈마 흐름입니다. 이것은 MHD 난류를 연구하기 위한 이상적인 실험실을 제공합니다:

- **직접 측정**: 우주선(ACE, Wind, Ulysses, PSP, Solar Orbiter)이 $\mathbf{v}$, $\mathbf{B}$, $n$, $T$를 높은 주기로 측정
- **큰 Reynolds 수**: $Re \sim 10^6$, $R_m \sim 10^6$
- **확장된 관성 범위**: 스케일에서 수십 배
- **불균형 난류**: 외부 방향으로 전파하는 파동이 지배

### 6.2 관측된 스펙트럼 영역

태양풍 난류는 여러 스펙트럼 범위를 보입니다:

**1. 에너지 포함 범위** ($f < 10^{-4}$ Hz, $\ell > 10^6$ km):

대규모 구조: 코로나 질량 방출, 흐름 상호작용 영역, 공전 상호작용 영역. 보편적이지 않음.

**2. 관성 범위** ($10^{-4} \text{ Hz} < f < f_{ion}$):

멱법칙 스펙트럼:

$$E(f) \propto f^{-\alpha}$$

$\alpha \approx 5/3$ (GS95 또는 K41과 일치).

이 범위는 주파수에서 2-3 배를 걸칩니다.

**3. 소산 범위** ($f > f_{ion}$):

**이온 회전 주파수** $f_{ion} \sim 0.1\text{–}1$ Hz(1 AU에서)에서, 스펙트럼이 가파르게 됩니다:

$$E(f) \propto f^{-\beta}$$

$\beta \approx 2.5\text{–}3$. 이것은 이온 스케일 운동 물리(gyro-공명, Landau 감쇠)가 중요해지는 곳입니다.

**4. 전자 소산 범위** ($f > f_{electron}$):

훨씬 더 높은 주파수($f \sim 100$ Hz)에서, 전자 스케일 물리가 지배합니다. 최근의 고해상도 데이터(MMS)가 이 영역을 탐구하고 있습니다.

### 6.3 이온 스케일에서의 스펙트럼 분리

이온 스케일에서의 **스펙트럼 분리**는 핵심 특징입니다. 분리 주파수는 이온 회전 반경 또는 이온 관성 길이에 해당합니다:

$$f_{break} \sim \frac{v_{sw}}{2\pi d_i}$$

여기서 $v_{sw}$는 태양풍 속도이고 $d_i = c/\omega_{pi}$는 이온 관성 길이입니다.

**물리적 해석:**

- **$f_{break}$ 이하**: MHD 난류(유체 설명 유효)
- **$f_{break}$ 이상**: 운동 난류(운동 효과: cyclotron 공명, Landau 감쇠)

**가열:**

소산 범위는 난류 에너지가 열로 변환되는 곳입니다. 태양풍은 단열 팽창이 예측하는 것보다 훨씬 더 뜨거운 것으로 관측되며, 난류 가열을 시사합니다.

### 6.4 태양풍의 비등방성

**Taylor 동결 가설**(시간을 공간으로 변환하는 $\mathbf{k} \cdot \mathbf{v}_{sw} = \omega$를 사용)을 사용한 측정은 다음을 보여줍니다:

- **수직 스펙트럼**: $E(k_\perp) \propto k_\perp^{-5/3}$
- **평행 스펙트럼**: 더 가파름(작은 $\ell_\parallel$에서 더 적은 파워)
- **비등방성 관계**: 대략 $k_\parallel \propto k_\perp^{2/3}$, GS95와 일치

그러나 정밀한 측정은 다음 때문에 어렵습니다:
- 단일 지점 측정(대부분의 우주선)
- 공간적 및 시간적 변동을 분리하는 모호함
- 다중 우주선 임무(Cluster, MMS, PSP-Solar Orbiter)가 이를 해결하는 데 도움

### 6.5 Python 예제: 태양풍 스펙트럼

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic solar wind spectrum
# Frequency range
f = np.logspace(-5, 2, 500)  # Hz

# Define spectral regimes
f_inertial = 1e-4  # Start of inertial range
f_ion = 0.5        # Ion gyrofrequency (spectral break)
f_electron = 50    # Electron scales

# Energy-containing range: flat or slightly rising
# 약한 양의 기울기는 대규모 에너지 저장소(태양풍 흐름, CME 구조)를
# 표현하며, 이 에너지가 관성 범위로 주입된다; 난류 자체는 더 높은
# 주파수에 위치한다.
E_energy = np.where(f < f_inertial, 1e2 * (f / f_inertial)**0.5, 0)

# Inertial range: -5/3 slope
# Kolmogorov/GS95의 -5/3 기울기가 약 2 십진배(decade)에 걸쳐 유지되는
# 이유: 이 범위에서 에너지는 소산 없이 전달되기 때문이다 — 대규모
# 소스와 이온 스케일 싱크 사이의 "파이프라인"이다.
E_inertial = np.where((f >= f_inertial) & (f < f_ion),
                      1e2 * (f / f_inertial)**(-5/3), 0)

# Dissipation range (ion scales): -2.8 slope
# 이온 스케일에서 ~-2.8로의 가파른 변화는 운동학적 감쇠(이온 Landau 감쇠,
# cyclotron 공명)의 시작을 반영한다: 파장 λ ~ ρ_i인 파동이 이온과
# 공명하며 에너지를 열로 전달하여, -5/3 관성 범위를 만든 자기 유사
# 캐스케이드를 끊는다.
E_dissipation = np.where((f >= f_ion) & (f < f_electron),
                         1e2 * (f_ion / f_inertial)**(-5/3) * (f / f_ion)**(-2.8), 0)

# Electron dissipation: steeper
# 전자 스케일에서 기울기가 ~-4로 더 가파르게 되는 이유: 전자도 요동을
# 감쇠시키기 시작하기 때문이다; 남은 에너지는 전자 가열로 소산되며,
# 이것이 태양풍에서 전자와 이온이 다르게 가열되는 이유다.
E_electron = np.where(f >= f_electron,
                      1e2 * (f_ion / f_inertial)**(-5/3) * (f_electron / f_ion)**(-2.8) * (f / f_electron)**(-4), 0)

# Total spectrum
E_total = E_energy + E_inertial + E_dissipation + E_electron

# Add noise to make it realistic
# 곱셈적 로그 정규 노이즈는 실제 단일 지점 우주선 측정의 분산을 모방한다:
# 스펙트럼은 앙상블로부터의 잡음 있는 샘플이며, 산란은 신호에 비례한다
# (덧셈적이 아님).
np.random.seed(42)
E_total *= 10**(np.random.normal(0, 0.1, len(f)))

# Plot
fig, ax = plt.subplots(figsize=(12, 7))

ax.loglog(f, E_total, linewidth=2, color='blue', label='Solar wind spectrum (synthetic)')

# Mark regimes
ax.axvline(f_inertial, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.text(f_inertial, 1e-2, 'Inertial range\nstart', fontsize=11, color='green', rotation=90, va='bottom')

ax.axvline(f_ion, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax.text(f_ion, 1e-2, 'Ion gyrofrequency\n(spectral break)', fontsize=11, color='red', rotation=90, va='bottom')

ax.axvline(f_electron, color='purple', linestyle='--', linewidth=2, alpha=0.7)
ax.text(f_electron, 1e-2, 'Electron\nscales', fontsize=11, color='purple', rotation=90, va='bottom')

# Reference slopes
f_ref = np.array([2e-4, 2e-1])
E_53 = 1e1 * (f_ref / f_ref[0])**(-5/3)
E_28 = 1e-1 * (f_ref / f_ref[0])**(-2.8)

ax.loglog(f_ref, E_53, 'k--', linewidth=2, alpha=0.6, label='$f^{-5/3}$ (inertial)')
ax.loglog(f_ref, E_28, 'k:', linewidth=2, alpha=0.6, label='$f^{-2.8}$ (dissipation)')

# Annotations
ax.text(1e-4, 5e1, 'Energy-containing\nrange', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
ax.text(1e-2, 1e-1, 'Inertial range\n(MHD turbulence)', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax.text(5, 1e-5, 'Dissipation range\n(kinetic)', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

ax.set_xlabel('Frequency $f$ (Hz)', fontsize=14)
ax.set_ylabel('Power Spectral Density $E(f)$ (arbitrary units)', fontsize=14)
ax.set_title('Solar Wind Magnetic Field Spectrum', fontsize=16, weight='bold')
ax.legend(fontsize=12, loc='lower left')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(1e-5, 1e2)
ax.set_ylim(1e-6, 1e3)

plt.tight_layout()
plt.savefig('solar_wind_spectrum.png', dpi=150)
plt.show()
```

### 6.6 가열과 소산

태양풍 온도는 단열 팽창이 예측하는 것보다 느리게 감소합니다:

$$T \propto r^{-\gamma}$$

관측된 $\gamma \sim 1$ (단열의 경우 양성자에 대해 $\gamma = 4/3$).

**난류 가열 메커니즘:**

1. **이온 cyclotron 공명**: 이온이 Alfvén/이온-cyclotron 파동과 공명하여 수직 에너지를 얻음
2. **Landau 감쇠**: 파동-입자 상호작용이 파동 에너지를 평행 입자 운동으로 전달
3. **확률론적 가열**: 입자가 난류의 시간 변화 장으로부터 에너지를 얻음
4. **재연결**: 난류로 형성된 전류 시트에서의 소산

어떤 메커니즘이 지배하는지 결정하는 것은 활발한 연구 분야입니다.

## 7. Python 예제: MHD 난류 스펙트럼

### 7.1 스펙트럼 모델 비교

```python
import numpy as np
import matplotlib.pyplot as plt

# Wavenumber range (perpendicular)
k = np.logspace(-1, 2, 200)

# Kolmogorov (hydrodynamic)
# k^{-5/3}은 차원 분석에서 나온다: 관성 범위에서 유일하게 관련된 양은
# ε(에너지 플럭스)와 k이며, E(k) ~ ε^{2/3} k^{-5/3}이 된다.
E_K41 = k**(-5/3)

# Iroshnikov-Kraichnan (MHD, isotropic)
# IK는 등방적 알프벤파(Alfvén wave) 충돌을 가정한다; 각 상호작용은
# (v_ℓ/v_A)² 비율만큼 약화되어 캐스케이드가 느려지고 스펙트럼이
# 얕아진다(k^{-3/2}). IK는 물리를 부분적으로만 맞추고 비등방성의
# 결정적 역할을 무시한다.
E_IK = k**(-3/2)

# Goldreich-Sridhar (MHD, anisotropic, perpendicular)
# GS95는 수직 방향에서 k^{-5/3}을 회복한다: 수직 소용돌이는 Kolmogorov처럼
# 캐스케이드하는 반면 평행 동역학은 알프벤파 전파(임계 균형)에 의해
# 제한되기 때문이다 — K41과의 차이는 수직 기울기가 아닌 비등방성
# (k_∥ ∝ k_⊥^{2/3})에 있다.
E_GS = k**(-5/3)

# Normalize at k=1
# k=1에서 정규화하면 이론 간의 진폭 차이가 아닌 기울기 차이를 플롯이
# 드러낼 수 있다.
E_K41 = E_K41 / E_K41[np.argmin(np.abs(k - 1))]
E_IK = E_IK / E_IK[np.argmin(np.abs(k - 1))]
E_GS = E_GS / E_GS[np.argmin(np.abs(k - 1))]

# Plot
fig, ax = plt.subplots(figsize=(10, 7))

ax.loglog(k, E_K41, linewidth=2.5, label='Kolmogorov (K41): $k^{-5/3}$', color='blue')
ax.loglog(k, E_IK, linewidth=2.5, label='Iroshnikov-Kraichnan (IK): $k^{-3/2}$', color='red')
ax.loglog(k, E_GS, linewidth=2.5, linestyle='--', label='Goldreich-Sridhar (GS95): $k_\\perp^{-5/3}$', color='green')

# Reference lines
k_ref = np.array([1, 10])
ax.loglog(k_ref, 1 * k_ref**(-5/3), 'k:', linewidth=2, alpha=0.5, label='$k^{-5/3}$ reference')
ax.loglog(k_ref, 1.5 * k_ref**(-3/2), 'k--', linewidth=2, alpha=0.5, label='$k^{-3/2}$ reference')

ax.set_xlabel('Wavenumber $k$ (or $k_\\perp$)', fontsize=14)
ax.set_ylabel('Energy spectrum $E(k)$ (normalized)', fontsize=14)
ax.set_title('Comparison of Turbulence Spectral Models', fontsize=16, weight='bold')
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.1, 100)
ax.set_ylim(1e-4, 10)

plt.tight_layout()
plt.savefig('turbulence_spectral_models.png', dpi=150)
plt.show()

# Print spectral indices
print("Spectral Indices:")
print(f"Kolmogorov (K41):           α = -5/3 = {-5/3:.4f}")
print(f"Iroshnikov-Kraichnan (IK):  α = -3/2 = {-3/2:.4f}")
print(f"Goldreich-Sridhar (GS95):   α = -5/3 = {-5/3:.4f} (in k_perp)")
```

### 7.2 비등방성 시각화

```python
import numpy as np
import matplotlib.pyplot as plt

# Perpendicular wavenumber
k_perp = np.logspace(-1, 2, 100)

# Goldreich-Sridhar anisotropy relation
# k_∥ ∝ k_⊥^{2/3}는 임계 균형(critical balance)의 특징이다: 각 수직
# 스케일 ℓ_⊥에서 알프벤 교차 시간 τ_A = ℓ_∥/v_A가 소용돌이 전복 시간
# τ_nl = ℓ_⊥/δv와 같아진다. 이 균형을 위반하는 소용돌이는 즉시
# 캐스케이드하거나(τ_nl < τ_A) 파동적이 되므로(τ_nl > τ_A), 난류는
# 정확히 이 비등방성 곡선에 머물도록 자기 조직화된다.
k_para_GS = k_perp**(2/3)

# Isotropic (IK)
# IK는 k_∥ = k_⊥ (구면 대칭 에너지 분포)를 가정하며, 알프벤파가 B_0을
# 따라 우선적으로 에너지를 운반한다는 사실을 무시한다; 이것이 IK가
# 잘못된 스펙트럼을 예측하는 근본적인 결함이다.
k_para_iso = k_perp

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: k_parallel vs k_perp
ax = axes[0]
ax.loglog(k_perp, k_para_GS, linewidth=2.5, label='GS95: $k_\\parallel \\propto k_\\perp^{2/3}$', color='green')
ax.loglog(k_perp, k_para_iso, linewidth=2.5, linestyle='--', label='Isotropic: $k_\\parallel = k_\\perp$', color='blue')

# Shaded region
ax.fill_between(k_perp, k_para_GS, k_para_iso, alpha=0.3, color='yellow', label='Anisotropic regime')

ax.set_xlabel('$k_\\perp$ (perpendicular wavenumber)', fontsize=13)
ax.set_ylabel('$k_\\parallel$ (parallel wavenumber)', fontsize=13)
ax.set_title('Anisotropy in MHD Turbulence', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')

# Panel 2: Aspect ratio
ax = axes[1]
# k_∥/k_⊥ = k_⊥^{-1/3} → 0 (k_⊥ → ∞일 때), 즉 작은 스케일 소용돌이는
# B_0을 따라 매우 길게 늘어난다(실공간에서 ℓ_∥ ≫ ℓ_⊥); 이 비등방성이
# MHD 난류를 등방적 Navier-Stokes 난류와 근본적으로 다르게 만든다.
aspect_GS = k_para_GS / k_perp  # = k_perp^{-1/3}
aspect_iso = np.ones_like(k_perp)

ax.loglog(k_perp, aspect_GS, linewidth=2.5, label='GS95: $k_\\parallel / k_\\perp \\propto k_\\perp^{-1/3}$', color='green')
ax.loglog(k_perp, aspect_iso, linewidth=2.5, linestyle='--', label='Isotropic: $k_\\parallel / k_\\perp = 1$', color='blue')

ax.set_xlabel('$k_\\perp$ (perpendicular wavenumber)', fontsize=13)
ax.set_ylabel('Aspect ratio $k_\\parallel / k_\\perp$', fontsize=13)
ax.set_title('Eddy Aspect Ratio vs Scale', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3, which='both')

# Annotation
ax.text(5, 0.05, 'Eddies become elongated\nalong $\\mathbf{B}_0$ at small scales', fontsize=12,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

plt.tight_layout()
plt.savefig('mhd_turbulence_anisotropy.png', dpi=150)
plt.show()

# Print aspect ratios at selected scales
print("Aspect Ratio (k_parallel / k_perp) for GS95:")
print(f"{'k_perp':>10} {'k_para':>10} {'Aspect':>10} {'l_para/l_perp':>15}")
print("-" * 50)
for kp in [0.1, 1, 10, 100]:
    kpa = kp**(2/3)
    aspect = kpa / kp
    ell_aspect = kp / kpa  # Invert for real-space aspect ratio
    print(f"{kp:>10.1f} {kpa:>10.3f} {aspect:>10.3f} {ell_aspect:>15.3f}")
```

## 요약

MHD 난류는 풍부하고 복잡한 현상입니다:

1. **Kolmogorov K41 이론**: 난류 이론의 기초로, 등방성 유체역학 난류의 관성 범위에서 $k^{-5/3}$ 에너지 스펙트럼을 예측합니다. 에너지는 큰 스케일에서 작은 스케일로 캐스케이드되어 Kolmogorov 스케일에서 소산됩니다.

2. **Iroshnikov-Kraichnan 이론**: 최초의 MHD 난류 이론으로, 등방성 Alfvén 파동 충돌을 가정합니다. $k^{-3/2}$ 스펙트럼을 예측합니다. 그러나 MHD 난류의 강한 비등방성을 포착하지 못하며 관측으로 뒷받침되지 않습니다.

3. **Goldreich-Sridhar (GS95) 이론**: 강한 MHD 난류의 표준 모델입니다. **임계 균형**을 통해 비등방성을 통합: 각 스케일에서 비선형 캐스케이드 시간이 Alfvén 파동 주기와 같습니다. $k_\parallel \propto k_\perp^{2/3}$ (소용돌이가 $\mathbf{B}_0$를 따라 연장됨)과 $E(k_\perp) \propto k_\perp^{-5/3}$ (수직 방향에서 Kolmogorov형)을 예측합니다. 시뮬레이션과 태양풍 관측에 의해 널리 지지됩니다.

4. **Elsässer 변수**: $\mathbf{z}^+ = \mathbf{v} + \mathbf{B}/\sqrt{\mu_0\rho}$와 $\mathbf{z}^- = \mathbf{v} - \mathbf{B}/\sqrt{\mu_0\rho}$는 반대 방향으로 전파하는 Alfvén 파동을 나타냅니다. MHD 방정식이 Elsässer 형식에서 대칭이 되어, $\mathbf{z}^+$와 $\mathbf{z}^-$가 서로 상호작용함을 명확히 합니다. 균형 난류는 $E^+ \approx E^-$를 가지며; 불균형 난류(예: 태양풍)는 불균등한 에너지를 가집니다.

5. **에너지 캐스케이드와 간헐성**: 에너지는 큰 스케일에서 작은 스케일로 캐스케이드됩니다(순방향 캐스케이드). 간헐성(비자기 유사, 다중 프랙탈 구조)은 강렬하고 국소화된 구조(전류 시트, 소용돌이)와 함께 구조 함수의 비정상 스케일링으로 이어집니다. MHD 난류는 유체역학 난류보다 더 간헐적입니다.

6. **태양풍 난류**: 태양풍은 MHD 난류를 위한 자연 실험실입니다. 관측된 스펙트럼은 $k^{-5/3}$ 관성 범위, 이온 스케일에서의 스펙트럼 분리, 그리고 더 가파른 소산 범위를 보여줍니다. 비등방성과 임계 균형이 확인됩니다. 난류 가열은 관측된 느린 온도 감소를 설명합니다. 최근 임무(PSP, Solar Orbiter, MMS)는 전례 없는 고해상도 데이터를 제공하고 있습니다.

MHD 난류를 이해하는 것은 천체물리학적 및 우주 플라즈마 관측을 해석하고, 난류 가열 및 수송을 모델링하며, dynamos, 재연결, 입자 가속 이론을 발전시키는 데 필수적입니다.

## 연습 문제

1. **Kolmogorov 스케일링**:
   a) 에너지 주입률 $\epsilon = 10^{-3}$ m²/s³과 가장 큰 소용돌이 크기 $L = 1$ m인 난류 흐름의 경우, 스케일 $\ell = 0.01$ m에서의 속도를 추정하십시오.
   b) 이 스케일에서 소용돌이 회전 시간을 계산하십시오.
   c) 운동 점성계수가 $\nu = 10^{-5}$ m²/s인 경우, Kolmogorov 스케일 $\eta = (\nu^3/\epsilon)^{1/4}$를 추정하십시오.

2. **Reynolds 수**:
   a) $L = 1000$ km, $v = 10$ m/s, $\nu = 1.5 \times 10^{-5}$ m²/s인 지구 대기의 경우, Reynolds 수를 계산하십시오.
   b) 비 $L/\eta$를 추정하십시오.
   c) 관성 범위가 몇 배의 스케일을 걸치는지 계산하십시오.

3. **IK 대 K41 스펙트럼**:
   a) $k = 0.1$에서 $100$까지 로그-로그 플롯에 IK($k^{-3/2}$)와 K41($k^{-5/3}$) 둘 다에 대해 $E(k)$ 대 $k$를 그리십시오.
   b) 두 스펙트럼이 2배 차이가 나는 파수 $k$는 무엇입니까($k=1$에서 같다고 가정)?
   c) 2배($k = 1$에서 $100$)에 걸쳐, 어떤 스펙트럼이 더 많은 에너지를 가집니까?

4. **Goldreich-Sridhar 비등방성**:
   a) $k_\perp = 100$ m⁻¹인 경우, GS95에 따라 $k_\parallel$은 무엇입니까($k_\parallel \propto k_\perp^{2/3}$)? 외부 스케일에서 $k_\parallel = k_\perp = 1$이라고 가정하십시오.
   b) 종횡비 $\ell_\parallel / \ell_\perp$는 무엇입니까?
   c) 이 스케일에서 소용돌이의 모양을 스케치하십시오.

5. **Elsässer 변수**:
   a) $\rho = 10^{-12}$ kg/m³인 플라즈마에서 $\mathbf{v} = (1, 0, 0)$ m/s와 $\mathbf{B} = (0, 0.01, 0)$ T가 주어졌을 때, $\mathbf{z}^+$와 $\mathbf{z}^-$를 계산하십시오.
   b) 운동 에너지 $E_{kin} = \frac{1}{2}\rho v^2$와 자기 에너지 $E_{mag} = B^2/(2\mu_0)$를 계산하십시오.
   c) $E_{kin} + E_{mag} = \frac{\rho}{4}(|\mathbf{z}^+|^2 + |\mathbf{z}^-|^2)$임을 확인하십시오.

6. **임계 균형**:
   a) $v_A = 50$ km/s와 난류 주입 스케일 $L = 10^6$ km인 태양풍에서, $L$에서의 속도 변동을 추정하십시오: $v_L \sim v_A$ (임계 균형에 의해).
   b) 스케일 $\ell_\perp = 100$ km에서, Kolmogorov 스케일링을 사용하여 $v_{\ell_\perp}$를 추정하십시오.
   c) 임계 균형으로부터 평행 스케일 $\ell_\parallel$을 계산하십시오.

7. **구조 함수**:
   a) 멱법칙 스펙트럼 $E(k) \propto k^{-5/3}$인 합성 속도장을 생성하십시오.
   b) 다양한 지연 $\ell$에 대해 2차 구조 함수 $S_2(\ell) = \langle |\delta v(\ell)|^2 \rangle$을 계산하십시오.
   c) 멱법칙 $S_2 \propto \ell^{\zeta_2}$를 피팅하고 $\zeta_2$를 K41 예측 $2/3$과 비교하십시오.

8. **태양풍 스펙트럼 분리**:
   a) 1 AU에서, 태양풍은 $n = 10^7$ m⁻³, $B = 5$ nT를 가집니다. 이온 관성 길이 $d_i = c/\omega_{pi}$를 계산하십시오.
   b) 태양풍 속도가 $v_{sw} = 400$ km/s인 경우, Taylor 가설을 사용하여 분리 주파수 $f_{break} = v_{sw}/(2\pi d_i)$를 추정하십시오.
   c) 관측된 분리 주파수 ~0.5 Hz와 비교하십시오.

9. **난류 가열률**:
   a) 태양풍에서 난류 에너지 캐스케이드율이 $\epsilon = 10^{-16}$ erg/g/s인 경우, 초당 양성자당 얼마나 많은 에너지가 소산됩니까?
   b) 이것이 양성자를 가열한다면, 1일에 걸친 온도 증가를 추정하십시오.
   c) 이것이 태양풍의 느린 온도 감소를 설명하기에 충분합니까?

10. **비등방 에너지 스펙트럼**:
    a) 2D $k_\perp$-$k_\parallel$ 평면에서, GS95 난류에 대한 일정한 에너지 밀도 $E(k_\perp, k_\parallel)$의 등고선을 스케치하십시오.
    b) 에너지는 $k_\parallel \propto k_\perp^{2/3}$ 근처에 집중됩니다. $k$-공간에서 이 "임계 균형 표면"을 스케치하십시오.
    c) 이것이 등방성 난류(에너지가 구 $k_\perp^2 + k_\parallel^2 = \text{const}$ 위에 있을 것)와 어떻게 다릅니까?

## 탐색

이전: [Advanced Reconnection](./07_Advanced_Reconnection.md) | 다음: [Dynamo Theory](./09_Dynamo_Theory.md)
