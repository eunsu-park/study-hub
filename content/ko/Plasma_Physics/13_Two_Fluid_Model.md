# 13. Two-Fluid Model

## 학습 목표

- Vlasov 방정식의 속도 공간 모멘트를 취하여 유체 방정식 유도하기
- 닫힘 문제(closure problem)와 다양한 닫힘 근사(등온, 단열, CGL) 이해하기
- 전자 운동량 방정식으로부터 일반화된 Ohm의 법칙 유도하고 각 항의 물리적 의미 분석하기
- Hall 효과와 작은 스케일에서 이온과 자기장의 분리에서의 역할 설명하기
- 입자 표류와 유체 표류의 차이, 특히 반자성 표류 구별하기
- 단일 유체 MHD를 넘어서는 파동 현상을 이해하기 위해 이유체 이론 적용하기

## 1. Vlasov 방정식에서 유체 방정식으로

### 1.1 모멘트 계층

Vlasov 방정식은 입자 종류 $s$에 대한 분포함수 $f_s(\mathbf{r}, \mathbf{v}, t)$의 진화를 기술합니다:

$$\frac{\partial f_s}{\partial t} + \mathbf{v} \cdot \nabla f_s + \frac{q_s}{m_s}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \frac{\partial f_s}{\partial \mathbf{v}} = \left(\frac{\partial f_s}{\partial t}\right)_{\text{coll}}$$

Vlasov 방정식은 플라즈마에 대한 완전한 정보를 포함하지만, 계산적으로 비용이 많이 드는 6차원 편미분 방정식입니다. 많은 응용에서는 전체 분포함수가 필요하지 않으며, 밀도, 유동 속도, 압력과 같은 거시적 양만 필요합니다.

**모멘트 방법**은 Vlasov 방정식을 다른 가중치로 속도 공간에 대해 적분하여 차원을 줄입니다. $n$차 모멘트는 Vlasov 방정식에 $v^n$을 곱하고 적분하여 얻습니다:

$$\int (\text{Vlasov 방정식}) \times (\text{가중함수}) \, d^3v$$

이는 유체 방정식의 계층을 생성하며, 각 방정식은 다음 고차 모멘트를 포함합니다.

### 1.2 0차 모멘트: 연속 방정식

0차 모멘트(가중치 = 1)는 **연속 방정식**을 제공합니다:

$$\int \frac{\partial f_s}{\partial t} d^3v + \int \mathbf{v} \cdot \nabla f_s d^3v + \int \frac{q_s}{m_s}(\mathbf{E} + \mathbf{v} \times \mathbf{B}) \cdot \frac{\partial f_s}{\partial \mathbf{v}} d^3v = 0$$

수밀도는:
$$n_s(\mathbf{r}, t) = \int f_s(\mathbf{r}, \mathbf{v}, t) d^3v$$

첫 번째 항에 대해:
$$\int \frac{\partial f_s}{\partial t} d^3v = \frac{\partial}{\partial t} \int f_s d^3v = \frac{\partial n_s}{\partial t}$$

두 번째 항에 대해, 속도 공간에서 발산 정리를 사용하면:
$$\int \mathbf{v} \cdot \nabla f_s d^3v = \nabla \cdot \int \mathbf{v} f_s d^3v = \nabla \cdot (n_s \mathbf{u}_s)$$

여기서 평균 유동 속도는:
$$\mathbf{u}_s = \frac{1}{n_s} \int \mathbf{v} f_s d^3v$$

세 번째 항인 Lorentz 힘 항은 다음과 같이 소멸합니다:
$$\int \frac{\partial f_s}{\partial \mathbf{v}} d^3v = [f_s]_{v=-\infty}^{v=+\infty} = 0$$

($|\mathbf{v}| \to \infty$일 때 $f_s \to 0$를 가정).

결과는 **연속 방정식**입니다:

$$\boxed{\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{u}_s) = 0}$$

이것은 입자 보존입니다. 이온화/재결합이 존재하면 우변에 소스 항이 나타납니다.

### 1.3 1차 모멘트: 운동량 방정식

1차 모멘트(가중치 = $m_s \mathbf{v}$)는 **운동량 방정식**을 제공합니다. Vlasov 방정식에 $m_s \mathbf{v}$를 곱하고 적분합니다:

운동량 밀도를 정의:
$$\mathbf{p}_s = m_s n_s \mathbf{u}_s = m_s \int \mathbf{v} f_s d^3v$$

고유 속도(열속도)는:
$$\mathbf{w} = \mathbf{v} - \mathbf{u}_s$$

압력 텐서는:
$$\overleftrightarrow{P}_s = m_s \int \mathbf{w} \mathbf{w} f_s d^3v$$

상당한 대수적 계산(부분 적분과 발산 정리 사용) 후, 운동량 방정식은 다음과 같이 됩니다:

$$\boxed{m_s n_s \frac{d \mathbf{u}_s}{dt} = q_s n_s (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) - \nabla \cdot \overleftrightarrow{P}_s + \mathbf{R}_s}$$

여기서 $d/dt = \partial/\partial t + \mathbf{u}_s \cdot \nabla$는 대류 도함수이고, $\mathbf{R}_s$는 다른 종과의 충돌로 인한 운동량 전달입니다.

이것은 유체 요소에 대한 Newton의 제2법칙입니다:
- **좌변**: 질량 × 가속도
- **우변**: Lorentz 힘 + 압력 경사력 + 충돌력

**핵심 포인트**: 이 방정식은 분포함수의 2차 모멘트인 압력 텐서 $\overleftrightarrow{P}_s$라는 새로운 양을 도입합니다.

### 1.4 2차 모멘트: 에너지 방정식

2차 모멘트(가중치 = $\frac{1}{2} m_s v^2$)는 **에너지 방정식**을 제공합니다:

열 에너지 밀도를 정의:
$$\mathcal{E}_s = \frac{1}{2} m_s \int w^2 f_s d^3v$$

등방 압력($\overleftrightarrow{P}_s = p_s \overleftrightarrow{I}$)에 대해:
$$p_s = \frac{1}{3} m_s \int w^2 f_s d^3v = \frac{2}{3} \mathcal{E}_s$$

에너지 방정식은 다음과 같이 됩니다:

$$\frac{\partial \mathcal{E}_s}{\partial t} + \nabla \cdot (\mathcal{E}_s \mathbf{u}_s) = -p_s \nabla \cdot \mathbf{u}_s - \nabla \cdot \mathbf{q}_s + Q_s$$

여기서:
- $\mathbf{q}_s = \frac{1}{2} m_s \int w^2 \mathbf{w} f_s d^3v$는 열유속 벡터(3차 모멘트)
- $Q_s$는 충돌 에너지 전달

$p_s = \frac{2}{3} \mathcal{E}_s$를 사용하면 다음과 같이 다시 쓸 수 있습니다:

$$\frac{3}{2} \frac{d p_s}{dt} + \frac{5}{2} p_s \nabla \cdot \mathbf{u}_s = -\nabla \cdot \mathbf{q}_s + Q_s$$

**닫힘 문제**: 에너지 방정식은 3차 모멘트인 열유속 $\mathbf{q}_s$를 도입합니다. 3차 모멘트를 취하면 4차 모멘트를 포함하는 방정식을 얻게 되며, 계속됩니다. 이 무한 계층은 최고차 모멘트에 대한 가정을 통해 **닫혀야** 합니다.

### 1.5 닫힘 문제

```
모멘트 계층:

0차 모멘트:  ∂n/∂t + ∇·(nu) = 0              (u 도입)
1차 모멘트:  mn(du/dt) = qn(E+u×B) - ∇·P + R  (P 도입)
2차 모멘트:  dp/dt = -p∇·u - ∇·q + Q          (q 도입)
3차 모멘트:  ...                              (다음 모멘트 도입)
...

각 방정식은 다음 고차 모멘트에서 새로운 미지수를 도입합니다.
이것이 닫힘 문제입니다.
```

최고 모멘트와 저차 모멘트 사이의 관계를 가정하여 계층을 **절단**해야 합니다. 일반적인 닫힘:

**1. 등온 닫힘**: 일정한 온도 가정
$$p_s = n_s k_B T_s, \quad T_s = \text{const}$$

이는 열전도가 매우 효율적이어서 온도가 즉시 평형을 이룰 때 유효합니다.

**2. 단열 닫힘**: 열유속이 없고($\mathbf{q}_s = 0$) 단열 진화를 가정
$$\frac{d}{dt}\left( \frac{p_s}{n_s^\gamma} \right) = 0$$

여기서 $\gamma$는 단열 지수입니다(단원자 기체의 경우 $\gamma = 5/3$). 이는 열전도가 무시할 수 있는 빠른 과정에 유효합니다.

**3. CGL 닫힘** (Chew-Goldberger-Low): 무충돌 자화 플라즈마의 경우, 압력은 비등방적입니다:
$$\overleftrightarrow{P}_s = p_{\perp s} \overleftrightarrow{I} + (p_{\parallel s} - p_{\perp s}) \hat{\mathbf{b}} \hat{\mathbf{b}}$$

이중 단열 방정식:
$$\frac{d}{dt}\left( \frac{p_{\perp s}}{n_s B} \right) = 0, \quad \frac{d}{dt}\left( \frac{p_{\parallel s} B^2}{n_s^3} \right) = 0$$

CGL은 Lesson 14에서 논의할 것입니다.

### 1.6 이유체 방정식 요약

각 입자 종(전자 $e$, 이온 $i$)에 대해:

**연속**:
$$\frac{\partial n_s}{\partial t} + \nabla \cdot (n_s \mathbf{u}_s) = 0$$

**운동량**:
$$m_s n_s \frac{d \mathbf{u}_s}{dt} = q_s n_s (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) - \nabla p_s + \mathbf{R}_s$$

(등방 압력 가정)

**에너지** (단열 닫힘):
$$\frac{d}{dt}\left( \frac{p_s}{n_s^\gamma} \right) = 0$$

이들은 **Maxwell 방정식**과 결합됩니다:
$$\nabla \times \mathbf{E} = -\frac{\partial \mathbf{B}}{\partial t}, \quad \nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \epsilon_0 \frac{\partial \mathbf{E}}{\partial t}$$
$$\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}, \quad \nabla \cdot \mathbf{B} = 0$$

여기서 전하와 전류 밀도는:
$$\rho = \sum_s q_s n_s, \quad \mathbf{J} = \sum_s q_s n_s \mathbf{u}_s$$

충돌 항 $\mathbf{R}_s$는 종들을 결합합니다. 전자-이온 충돌의 경우:
$$\mathbf{R}_e = -\mathbf{R}_i = -\frac{m_e n_e}{\tau_{ei}} (\mathbf{u}_e - \mathbf{u}_i)$$

여기서 $\tau_{ei}$는 전자-이온 충돌 시간입니다.

## 2. 일반화된 Ohm의 법칙

### 2.1 전자 운동량 방정식에서 유도

이유체 이론에서 가장 중요한 결과 중 하나는 전기장과 전류를 연결하는 **일반화된 Ohm의 법칙**입니다. 이상적 MHD에서는 간단한 형태를 가집니다:
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0$$

하지만 이것은 심각한 근사입니다. 전자 운동량 방정식에서 완전한 형태를 유도하겠습니다.

시작:
$$m_e n_e \frac{d \mathbf{u}_e}{dt} = -e n_e (\mathbf{E} + \mathbf{u}_e \times \mathbf{B}) - \nabla p_e + \mathbf{R}_e$$

충돌 항은 다음과 같이 쓸 수 있습니다:
$$\mathbf{R}_e = -\frac{m_e n_e}{\tau_{ei}} (\mathbf{u}_e - \mathbf{u}_i) \approx -\frac{m_e n_e \mathbf{u}_e}{\tau_{ei}}$$

(전류를 운반하는 전자에 대해 $u_e \gg u_i$를 가정).

재배열:
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} = \frac{m_e}{e \tau_{ei}} \mathbf{u}_e - \frac{1}{e n_e} \nabla p_e + \frac{m_e}{e n_e} \frac{d \mathbf{u}_e}{dt}$$

이제 모든 것을 **전류 밀도** $\mathbf{J}$와 **질량 중심 속도** $\mathbf{v}$로 표현합니다.

정의:
$$\mathbf{J} = -e n_e \mathbf{u}_e + e n_i \mathbf{u}_i \approx -e n_e (\mathbf{u}_e - \mathbf{u}_i)$$
$$\mathbf{v} = \frac{m_i n_i \mathbf{u}_i + m_e n_e \mathbf{u}_e}{m_i n_i + m_e n_e} \approx \mathbf{u}_i$$

($m_i \gg m_e$와 준중성 $n_e \approx n_i \equiv n$ 사용).

전류 정의로부터:
$$\mathbf{u}_e = \mathbf{u}_i - \frac{\mathbf{J}}{e n} \approx \mathbf{v} - \frac{\mathbf{J}}{e n}$$

재배열된 전자 방정식에 대입:

$$\mathbf{E} + \left( \mathbf{v} - \frac{\mathbf{J}}{en} \right) \times \mathbf{B} = \frac{m_e}{e^2 n \tau_{ei}} \mathbf{J} - \frac{1}{en} \nabla p_e + \frac{m_e}{e^2 n} \frac{d}{dt}\left( -\frac{\mathbf{J}}{e n} \right)$$

외적 간소화:
$$\mathbf{u}_e \times \mathbf{B} = \mathbf{v} \times \mathbf{B} - \frac{\mathbf{J} \times \mathbf{B}}{en}$$

이것이 **일반화된 Ohm의 법칙**을 제공합니다:

$$\boxed{\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J} + \frac{1}{en} \mathbf{J} \times \mathbf{B} - \frac{1}{en} \nabla p_e + \frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}}$$

여기서 **저항률**은:
$$\eta = \frac{m_e}{e^2 n \tau_{ei}}$$

### 2.2 각 항의 물리적 해석

우변의 각 항을 확인해봅시다:

1. **저항 항**: $\eta \mathbf{J}$
   - 전자-이온 충돌에 의한 Ohmic 소산
   - 자기 확산을 야기(저항 MHD)
   - $\eta \sim T_e^{-3/2}$ (온도에 따라 감소)

2. **Hall 항**: $\frac{1}{en} \mathbf{J} \times \mathbf{B}$
   - 자기장으로부터 이온의 분리
   - 이온 skin depth $d_i = c/\omega_{pi}$ 스케일에서 중요
   - 빠른 자기 재결합 가능

3. **전자 압력 항**: $-\frac{1}{en} \nabla p_e$
   - 압력 경사가 E 장 없이도 전류를 구동
   - 급격한 경사 영역에서 중요 (예: 전류 시트)

4. **전자 관성 항**: $\frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}$
   - 전자 skin depth $d_e = c/\omega_{pe}$에서 중요
   - 매우 빠른 현상에 관련 (whistler 파동, 재결합)

### 2.3 스케일 분석: 각 항이 언제 중요한가?

각 항이 언제 중요한지 결정하기 위해 **차수 분석**을 수행합니다.

특성 스케일 정의:
- 길이: $L$
- 속도: $V$
- 자기장: $B_0$
- 밀도: $n_0$
- 전류: $J_0 \sim B_0/(\mu_0 L)$ (Ampère의 법칙에서)

**이상적 MHD 항** (좌변):
$$\mathbf{v} \times \mathbf{B} \sim V B_0$$

**저항 항**:
$$\eta \mathbf{J} \sim \eta \frac{B_0}{\mu_0 L}$$

비율:
$$\frac{\eta J}{\mathbf{v} \times \mathbf{B}} \sim \frac{\eta}{\mu_0 V L} = \frac{1}{R_m}$$

여기서 $R_m = \mu_0 V L / \eta$는 **자기 Reynolds 수**입니다. 저항률은 $R_m \lesssim 1$일 때 중요합니다.

**Hall 항**:
$$\frac{\mathbf{J} \times \mathbf{B}}{en} \sim \frac{B_0^2}{\mu_0 e n_0 L}$$

비율:
$$\frac{J \times B / en}{\mathbf{v} \times \mathbf{B}} \sim \frac{B_0}{\mu_0 e n_0 V L} = \frac{V_A}{V} \frac{d_i}{L}$$

여기서 $d_i = c/\omega_{pi} = \sqrt{m_i / (\mu_0 e^2 n_0)}$는 **이온 skin depth**이고 $V_A = B_0/\sqrt{\mu_0 m_i n_0}$는 Alfvén 속도입니다.

Hall 항은 $L \lesssim d_i$ 또는 이온 스케일에서 $V \lesssim V_A$일 때 중요합니다.

**전자 압력 항**:
$$\frac{\nabla p_e}{en} \sim \frac{k_B T_e}{eL}$$

비율:
$$\frac{\nabla p_e / en}{\mathbf{v} \times \mathbf{B}} \sim \frac{k_B T_e}{e V B_0 L} = \frac{v_{te}^2}{V^2} \frac{\rho_e}{L}$$

여기서 $v_{te} = \sqrt{k_B T_e / m_e}$는 전자 열속도이고 $\rho_e = v_{te}/\omega_{ce}$는 전자 gyroradius입니다.

이 항은 급격한 압력 경사 영역에서 중요합니다.

**전자 관성 항**:
$$\frac{m_e}{e^2 n^2} \frac{dJ}{dt} \sim \frac{m_e}{e^2 n_0^2} \frac{B_0}{\mu_0 L} \frac{V}{L} = \frac{m_e V B_0}{\mu_0 e^2 n_0 L^2}$$

비율:
$$\frac{m_e dJ/dt / (e^2 n^2)}{v \times B} \sim \frac{m_e}{\mu_0 e^2 n_0 L^2} = \frac{d_e^2}{L^2}$$

여기서 $d_e = c/\omega_{pe}$는 **전자 skin depth**입니다.

이 항은 $L \lesssim d_e$일 때 중요합니다.

**요약**:
```
항                   스케일               언제 중요한가
----------------    -----------------   ------------------------
저항                1/R_m               R_m ~ 1 (낮은 T, 작은 L)
Hall                d_i/L               L ~ d_i (이온 스케일)
전자 압력           β_e ρ_e/L           급격한 경사
전자 관성           (d_e/L)^2           L ~ d_e (전자 스케일)

일반적 순서: d_e << ρ_e << d_i << L (MHD)
```

### 2.4 제한 경우

**이상적 MHD** ($R_m \to \infty$, $L \gg d_i$):
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = 0$$

자기장은 유체에 동결됩니다.

**저항 MHD** (저항 항 유지, 다른 항 무시):
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \eta \mathbf{J}$$

자기 재결합을 허용하지만 느림(Sweet-Parker 속도).

**Hall MHD** (Hall 항 유지, 저항/관성 무시):
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \frac{1}{en} \mathbf{J} \times \mathbf{B}$$

빠른 재결합(Petschek 속도), whistler 파동 가능.

**전자 MHD** (Hall + 관성 유지, 저항 무시):
$$\mathbf{E} + \mathbf{v} \times \mathbf{B} = \frac{1}{en} \mathbf{J} \times \mathbf{B} + \frac{m_e}{e^2 n^2} \frac{d \mathbf{J}}{dt}$$

전자 스케일에서 관련 (예: 재결합 확산 영역).

## 3. Hall 효과

### 3.1 Hall 항의 물리

Hall 항 $\frac{1}{en} \mathbf{J} \times \mathbf{B}$는 전자와 이온 운동의 차이에서 발생합니다. 전류가 자기장을 가로질러 흐를 때, 전자와 이온은 다른 Lorentz 힘을 경험하여 전하 분리를 만들고 따라서 **$\mathbf{J}$와 $\mathbf{B}$ 모두에 수직인 전기장**을 만듭니다.

자기장 $\mathbf{B} = B_0 \hat{\mathbf{z}}$에서 전류 $\mathbf{J} = J_x \hat{\mathbf{x}}$를 고려:

$$\mathbf{J} \times \mathbf{B} = J_x B_0 \hat{\mathbf{y}}$$

이것이 전기장을 만듭니다:
$$E_y = \frac{J_x B_0}{en}$$

이것이 **Hall 전기장**입니다.

### 3.2 Hall 매개변수

**Hall 매개변수**는 자기장의 중요성을 정량화합니다:

$$\Omega_s \tau_s = \omega_{cs} \tau_{cs}$$

여기서 $\omega_{cs} = q_s B / m_s$는 cyclotron 주파수이고 $\tau_{cs}$는 충돌 시간입니다.

- $\Omega_s \tau_s \ll 1$일 때: 충돌이 지배적, 입자 궤도가 회전을 완료하기 전에 중단됨 → **비자화**
- $\Omega_s \tau_s \gg 1$일 때: 입자가 충돌 사이에 많은 회전 완료 → **자화**

일반적인 플라즈마에서 전자의 경우, $\Omega_e \tau_e \gg 1$ (강하게 자화됨).
이온의 경우, $\Omega_i \tau_i$는 다양함 (충돌 플라즈마에서 약하게 자화, 고온 핵융합 플라즈마에서 강하게 자화).

### 3.3 자기장으로부터 이온의 분리

이온 skin depth $d_i$보다 큰 스케일에서는 전자와 이온 모두 자기장에 동결됩니다(이상적 MHD). 하지만 $L \lesssim d_i$ 스케일에서는 Hall 항이 중요해지고, **이온이 자기장으로부터 분리**됩니다.

이를 보기 위해 이온과 전자 운동량 방정식을 고려:

**이온**:
$$m_i n \frac{d \mathbf{u}_i}{dt} = e n (\mathbf{E} + \mathbf{u}_i \times \mathbf{B}) - \nabla p_i$$

**전자** (일반화된 Ohm의 법칙에서, Hall 항만 유지):
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} \approx \frac{1}{en} \mathbf{J} \times \mathbf{B}$$

$\mathbf{J} = en(\mathbf{u}_i - \mathbf{u}_e)$ 사용:
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} = \frac{1}{en} en (\mathbf{u}_i - \mathbf{u}_e) \times \mathbf{B}$$

재배열:
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} = (\mathbf{u}_i - \mathbf{u}_e) \times \mathbf{B}$$
$$\mathbf{E} + \mathbf{u}_i \times \mathbf{B} = 0$$

따라서 **전자**는 동결 조건을 만족합니다:
$$\mathbf{E} + \mathbf{u}_e \times \mathbf{B} = 0$$

하지만 **이온**은 그렇지 않습니다! 이온은 전기장을 경험합니다:
$$\mathbf{E} = -\mathbf{u}_i \times \mathbf{B} + (\mathbf{u}_i - \mathbf{u}_e) \times \mathbf{B} = \mathbf{u}_e \times \mathbf{B} \neq -\mathbf{u}_i \times \mathbf{B}$$

이것은 자기장이 이온 유체가 아닌 **전자 유체에 동결**되어 있음을 의미하며, $\sim d_i$ 스케일에서.

### 3.4 Hall MHD 파동

Hall 항을 포함하면 MHD 파동 분산이 수정됩니다. 주요 변화는 고주파수에서 **whistler 파동**의 출현입니다.

Hall MHD 분산 관계(저주파, 소진폭 한계)는 다음을 제공합니다:

**Alfvén/whistler 분기**:
$$\omega = k_\parallel V_A \sqrt{1 + k^2 d_i^2}$$

- $k d_i \ll 1$ (큰 스케일)일 때: $\omega \approx k_\parallel V_A$ (Alfvén 파동)
- $k d_i \gg 1$ (작은 스케일)일 때: $\omega \approx k_\parallel V_A k d_i = k \sqrt{k_\parallel V_A d_i}$ (whistler)

Whistler 파동은:
- **우선회 원편광** (이온 프레임에서)
- **분산적**: 위상 속도가 $k$에 따라 증가
- **이온 운동 없음**: 전자만 반응

아래 Python 코드에서 이 분산 관계를 계산할 것입니다.

## 4. 반자성 표류

### 4.1 입자 표류 vs. 유체 표류

Lesson 3에서 단일 입자 궤도 이론에서 **입자 표류**를 유도했습니다:

$$\mathbf{v}_E = \frac{\mathbf{E} \times \mathbf{B}}{B^2}, \quad \mathbf{v}_{\nabla B} = \frac{m v_\perp^2}{2 q B^3} \mathbf{B} \times \nabla B, \quad \text{등}$$

이들은 개별 입자의 표류입니다.

유체 이론에서는 압력 경사 및 기타 집단 효과에서 발생하는 **유체 표류**가 있습니다. 가장 중요한 것은 **반자성 표류**입니다.

### 4.2 반자성 표류 유도

$\mathbf{B}$에 수직인 압력 경사를 가진 자화 플라즈마에서 운동량 방정식을 고려:

$$m_s n_s \frac{d \mathbf{u}_s}{dt} = q_s n_s (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) - \nabla p_s$$

평형($d\mathbf{u}_s/dt = 0$)에서 전기장이 없을 때($\mathbf{E} = 0$):

$$0 = q_s n_s \mathbf{u}_s \times \mathbf{B} - \nabla p_s$$

$\mathbf{B}$와 외적을 취하면:

$$q_s n_s (\mathbf{u}_s \times \mathbf{B}) \times \mathbf{B} = -\nabla p_s \times \mathbf{B}$$

벡터 항등식 $(\mathbf{A} \times \mathbf{B}) \times \mathbf{C} = \mathbf{B}(\mathbf{A} \cdot \mathbf{C}) - \mathbf{A}(\mathbf{B} \cdot \mathbf{C})$ 사용:

$$q_s n_s [\mathbf{B} (\mathbf{u}_s \cdot \mathbf{B}) - \mathbf{u}_s B^2] = -\nabla p_s \times \mathbf{B}$$

유동이 $\mathbf{B}$에 수직이면(즉, $\mathbf{u}_s \cdot \mathbf{B} = 0$):

$$\mathbf{u}_s = \frac{\nabla p_s \times \mathbf{B}}{q_s n_s B^2} = -\frac{\mathbf{B} \times \nabla p_s}{q_s n_s B^2}$$

이것이 **반자성 표류 속도**입니다:

$$\boxed{\mathbf{v}_{*s} = -\frac{\mathbf{B} \times \nabla p_s}{q_s n_s B^2}}$$

전자($q_e = -e$)에 대해:
$$\mathbf{v}_{*e} = \frac{\mathbf{B} \times \nabla p_e}{e n_e B^2}$$

이온($q_i = +e$)에 대해:
$$\mathbf{v}_{*i} = -\frac{\mathbf{B} \times \nabla p_i}{e n_i B^2}$$

### 4.3 반자성 전류

**반자성 전류**는:

$$\mathbf{J}_* = \sum_s q_s n_s \mathbf{v}_{*s} = -\frac{\mathbf{B} \times \nabla p_e}{B^2} - \frac{\mathbf{B} \times \nabla p_i}{B^2} = \frac{\mathbf{B} \times \nabla p}{B^2}$$

여기서 $p = p_e + p_i$는 총 압력입니다.

이것은 또한 다음과 같이 쓸 수 있습니다:
$$\mathbf{J}_* = -\nabla p \times \frac{\mathbf{B}}{B^2}$$

**핵심 포인트**: 반자성 표류는 **입자 표류가 아닙니다**! 개별 입자 궤도를 풀면 이 표류를 찾을 수 없습니다. 이것은 압력 경사로 인한 **분포함수의 공간 변화**에서 발생합니다.

이를 보기 위해, 반자성 표류 속도가 경사 스케일 길이 $L_p = p / |\nabla p|$에 의존한다는 것을 주목:

$$v_* \sim \frac{p}{q n B L_p} = \frac{k_B T}{q B L_p} \sim \frac{\rho}{L_p} v_{th}$$

여기서 $\rho = v_{th}/\omega_c$는 gyroradius입니다.

무충돌 플라즈마에서, 다른 gyro-궤도의 입자들은 다른 밀도를 가지며, 분포에 대한 평균 시 순 표류를 만듭니다.

### 4.4 물리적 해석: 자화 전류

반자성 전류는 회전하는 입자의 자기 모멘트에서 발생하는 **자화 전류**로 이해될 수 있습니다.

자화는:
$$\mathbf{M} = -n \mu \frac{\mathbf{B}}{B}$$

여기서 $\mu = m v_\perp^2 / (2B)$는 자기 모멘트입니다.

자화 전류는:
$$\mathbf{J}_m = \nabla \times \mathbf{M}$$

$\mathbf{B}$에 수직인 압력 경사에 대해, 이것은 다음을 제공합니다:
$$\mathbf{J}_m = \frac{\mathbf{B} \times \nabla p_\perp}{B^2}$$

이것은 정확히 반자성 전류입니다.

### 4.5 예: 원통형 플라즈마 기둥

다음을 가진 원통형 플라즈마 기둥을 고려:
- 축방향 자기장: $\mathbf{B} = B_0 \hat{\mathbf{z}}$
- 방사형 압력 프로파일: $p(r) = p_0 \left(1 - \frac{r^2}{a^2}\right)$

압력 경사는:
$$\nabla p = \frac{dp}{dr} \hat{\mathbf{r}} = -\frac{2 p_0 r}{a^2} \hat{\mathbf{r}}$$

반자성 전류는:
$$\mathbf{J}_* = \frac{\mathbf{B} \times \nabla p}{B^2} = \frac{B_0 \hat{\mathbf{z}} \times \left( -\frac{2 p_0 r}{a^2} \hat{\mathbf{r}} \right)}{B_0^2} = \frac{2 p_0 r}{B_0 a^2} \hat{\boldsymbol{\theta}}$$

이것은 인가된 장에 반대하는 방위각 전류입니다(반자성).

전자에 대한 반자성 표류 속도는:
$$\mathbf{v}_{*e} = \frac{\mathbf{B} \times \nabla p_e}{e n_e B^2} = \frac{2 k_B T_e r}{e B_0 a^2} \hat{\boldsymbol{\theta}}$$

전자는 $+\hat{\boldsymbol{\theta}}$ 방향으로 표류합니다(위에서 볼 때 반시계 방향).

이온은 반대 방향으로 표류합니다:
$$\mathbf{v}_{*i} = -\frac{2 k_B T_i r}{e B_0 a^2} \hat{\boldsymbol{\theta}}$$

순 전류는 전자와 이온 기여의 합입니다.

## 5. 이유체 파동

### 5.1 Kinetic Alfvén 파동

이온 gyroradius에 접근하는 스케일에서, Alfvén 파동은 운동학적 효과에 의해 수정됩니다. **kinetic Alfvén 파동(KAW)**은 분산 관계를 가집니다:

$$\omega^2 = k_\parallel^2 V_A^2 \left( 1 + k_\perp^2 \rho_s^2 \right)$$

여기서 $\rho_s = c_s / \omega_{ci}$는 **이온 음향 gyroradius**(또는 hybrid gyroradius)이고, $c_s = \sqrt{k_B T_e / m_i}$는 이온 음향 속도입니다.

주요 특징:
- 유한 $k_\perp$가 파동 주파수를 증가시킴
- 전기장이 평행 성분을 가짐: $E_\parallel \neq 0$
- 전자가 $\mathbf{B}$에 평행하게 가속될 수 있음

KAW는 다음에서 중요합니다:
- 오로라 가속
- 태양풍 난류
- 토카막 가장자리 난류

### 5.2 이유체 관점에서의 Whistler 파동

Lesson 10에서 운동 이론에서 whistler 파동을 유도했습니다. 이유체 관점은 다음과 같습니다:

Hall MHD(전자 관성 무시)에서 시작하여, 전자기파의 분산 관계는:

$$\omega = \frac{k_\parallel^2 V_A^2}{\omega_{ci}} \equiv k_\parallel V_A k_\parallel d_i$$

이것이 **whistler 파동**입니다:
- 고주파($\omega \ll \omega_{ce}$, 하지만 $\omega \gg \omega_{ci}$)
- 우선회 편광(전자 회전, 이온 정지)
- 위상 속도가 $k$에 따라 증가(분산적)

Whistler 파동은 다음에서 주요 역할을 합니다:
- 자기 재결합(빠른 유입 가능)
- 복사 벨트 역학(고에너지 전자의 피치각 산란)
- 태양 코로나 가열

### 5.3 이온-사이클로트론 파동

이온 사이클로트론 주파수 근처의 주파수에서 **이온-사이클로트론 파동**(또는 **이온 Bernstein 파동**)이 나타납니다:

$$\omega \approx \omega_{ci} + k_\parallel^2 V_A^2 / \omega_{ci}$$

특징:
- 좌선회 편광(이온 회전, 전자 단열적 반응)
- $\omega = \omega_{ci}$에서 공명 흡수
- 플라즈마 가열에 사용(토카막의 ICRF 가열)

### 5.4 이류 불안정성

두 유체가 상대 흐름 속도 $u_0$를 가질 때, 시스템이 불안정할 수 있습니다. 이온이 정지하고 전자가 속도 $u_0$로 흐르는 경우를 고려:

분산 관계는 다음과 같이 됩니다:
$$\omega^2 - k^2 c_s^2 - \omega_{pe}^2 = 0, \quad \text{(이온 음향)}$$
$$(\omega - k u_0)^2 - \omega_{pe}^2 = 0 \quad \text{(Langmuir, Doppler로 이동)}$$

이 모드들이 결합되면, $u_0 > v_{te}$ (전자 열속도)일 때 **이류 불안정성**을 얻습니다.

성장률:
$$\gamma \sim \frac{\omega_{pe}}{3^{1/3}} \left( \frac{u_0}{v_{te}} \right)^{2/3}$$

이것은 **운동학적 불안정성**이지만, 적절한 닫힘으로 이유체 이론에서 포착될 수 있습니다.

## 6. Python 코드 예제

### 6.1 이유체 vs. 단일 유체 분산 관계

```python
import numpy as np
import matplotlib.pyplot as plt

# Plasma parameters
m_i = 1.67e-27  # proton mass (kg)
m_e = 9.11e-31  # electron mass (kg)
e = 1.6e-19     # elementary charge (C)
c = 3e8         # speed of light (m/s)
mu_0 = 4e-7 * np.pi  # permeability

n = 1e19        # density (m^-3)
B = 0.1         # magnetic field (T)
T_e = 10        # electron temperature (eV)
T_i = 10        # ion temperature (eV)

# Convert temperature to Joules
k_B = 1.38e-23
T_e_J = T_e * e
T_i_J = T_i * e

# Derived quantities
omega_pe = np.sqrt(n * e**2 / (m_e * 8.85e-12))
omega_pi = np.sqrt(n * e**2 / (m_i * 8.85e-12))
omega_ce = e * B / m_e
omega_ci = e * B / m_i

v_A = B / np.sqrt(mu_0 * n * m_i)  # Alfvén speed
c_s = np.sqrt((T_e_J + T_i_J) / m_i)  # ion sound speed
d_i = c / omega_pi  # ion skin depth
d_e = c / omega_pe  # electron skin depth

print("Plasma parameters:")
print(f"  Alfvén speed V_A = {v_A:.2e} m/s = {v_A/c:.2e} c")
print(f"  Ion sound speed c_s = {c_s:.2e} m/s")
print(f"  Ion skin depth d_i = {d_i:.2e} m")
print(f"  Electron skin depth d_e = {d_e:.2e} m")
print(f"  Ion gyrofrequency ω_ci = {omega_ci:.2e} rad/s")
print(f"  Electron gyrofrequency ω_ce = {omega_ce:.2e} rad/s")
print()

# Wavenumber range (parallel to B)
k_min = 1 / (100 * d_i)
k_max = 1 / (0.1 * d_i)
k = np.logspace(np.log10(k_min), np.log10(k_max), 500)

# MHD Alfvén wave (single-fluid)
omega_MHD = k * v_A

# Hall MHD Alfvén/whistler wave (two-fluid)
omega_Hall = k * v_A * np.sqrt(1 + (k * d_i)**2)

# Kinetic Alfvén wave (with finite k_perp)
k_perp = k / 2  # assume oblique propagation
rho_s = c_s / omega_ci  # ion sound gyroradius
omega_KAW = k * v_A * np.sqrt(1 + (k_perp * rho_s)**2)

# Plot dispersion relations
plt.figure(figsize=(10, 6))
plt.loglog(k * d_i, omega_MHD / omega_ci, 'b-', label='MHD Alfvén', linewidth=2)
plt.loglog(k * d_i, omega_Hall / omega_ci, 'r--', label='Hall MHD (whistler)', linewidth=2)
plt.loglog(k * d_i, omega_KAW / omega_ci, 'g-.', label='Kinetic Alfvén', linewidth=2)

plt.axvline(1, color='k', linestyle=':', alpha=0.5, label='$k d_i = 1$')
plt.xlabel(r'$k d_i$ (normalized wavenumber)', fontsize=12)
plt.ylabel(r'$\omega / \omega_{ci}$ (normalized frequency)', fontsize=12)
plt.title('Two-Fluid Dispersion Relations: Alfvén to Whistler Transition', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('two_fluid_dispersion.png', dpi=150)
plt.show()

print("At k d_i = 1:")
idx = np.argmin(np.abs(k * d_i - 1))
print(f"  MHD: ω/ω_ci = {omega_MHD[idx]/omega_ci:.2f}")
print(f"  Hall MHD: ω/ω_ci = {omega_Hall[idx]/omega_ci:.2f}")
print(f"  Kinetic Alfvén: ω/ω_ci = {omega_KAW[idx]/omega_ci:.2f}")
```

### 6.2 일반화된 Ohm의 법칙: 상대적 항 크기

```python
import numpy as np
import matplotlib.pyplot as plt

def ohm_law_terms(n, T_e, B, L, V, eta=None):
    """
    Calculate relative magnitudes of generalized Ohm's law terms.

    Parameters:
    n: density (m^-3)
    T_e: electron temperature (eV)
    B: magnetic field (T)
    L: length scale (m)
    V: flow velocity (m/s)
    eta: resistivity (Ω·m), if None calculate from Spitzer
    """
    e = 1.6e-19
    m_e = 9.11e-31
    m_i = 1.67e-27
    mu_0 = 4e-7 * np.pi
    k_B = 1.38e-23
    c = 3e8

    # Spitzer resistivity (if not provided)
    if eta is None:
        T_e_eV = T_e
        ln_Lambda = 15  # Coulomb logarithm (typical)
        eta = 5.2e-5 * ln_Lambda * T_e_eV**(-3/2)  # Ω·m

    # Current density (from Ampere's law estimate)
    J = B / (mu_0 * L)

    # Characteristic electric field (ideal MHD)
    E_ideal = V * B

    # Generalized Ohm's law terms
    E_resistive = eta * J
    E_Hall = J * B / (e * n)
    E_pressure = k_B * T_e * e / (e * L)  # ∇p_e ~ nkT/L

    omega_pe = np.sqrt(n * e**2 / (m_e * 8.85e-12))
    d_e = c / omega_pe
    E_inertia = (m_e / (e**2 * n**2)) * J * (V / L)

    # Normalize to ideal MHD term
    terms = {
        'Ideal (v×B)': E_ideal,
        'Resistive (ηJ)': E_resistive,
        'Hall (J×B/ne)': E_Hall,
        'Pressure (∇p_e/ne)': E_pressure,
        'Inertia (m_e dJ/dt)': E_inertia
    }

    return {k: v/E_ideal for k, v in terms.items()}, eta

# Parameter scan: vary length scale
L_range = np.logspace(-3, 3, 100)  # 1 mm to 1 km
n = 1e19
T_e = 10
B = 0.1
V = 1e5  # 100 km/s

terms_vs_L = {k: [] for k in ['Ideal (v×B)', 'Resistive (ηJ)',
                               'Hall (J×B/ne)', 'Pressure (∇p_e/ne)',
                               'Inertia (m_e dJ/dt)']}

for L in L_range:
    terms, _ = ohm_law_terms(n, T_e, B, L, V)
    for k, v in terms.items():
        terms_vs_L[k].append(v)

# Plot
plt.figure(figsize=(10, 6))
for key, values in terms_vs_L.items():
    if key != 'Ideal (v×B)':
        plt.loglog(L_range, values, label=key, linewidth=2)

# Mark characteristic scales
d_e = 3e8 / np.sqrt(n * (1.6e-19)**2 / (9.11e-31 * 8.85e-12))
d_i = 3e8 / np.sqrt(n * (1.6e-19)**2 / (1.67e-27 * 8.85e-12))
plt.axvline(d_e, color='r', linestyle=':', alpha=0.7, label=f'$d_e$ = {d_e:.2e} m')
plt.axvline(d_i, color='b', linestyle=':', alpha=0.7, label=f'$d_i$ = {d_i:.2e} m')

plt.xlabel('Length scale L (m)', fontsize=12)
plt.ylabel('Relative magnitude (normalized to v×B)', fontsize=12)
plt.title('Generalized Ohm\'s Law: Term Magnitudes vs. Scale', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('ohm_law_terms.png', dpi=150)
plt.show()

# Print values at specific scales
print("\nRelative term magnitudes:")
print(f"\nAt L = {d_e:.2e} m (electron skin depth):")
terms, _ = ohm_law_terms(n, T_e, B, d_e, V)
for k, v in terms.items():
    print(f"  {k:25s}: {v:.2e}")

print(f"\nAt L = {d_i:.2e} m (ion skin depth):")
terms, _ = ohm_law_terms(n, T_e, B, d_i, V)
for k, v in terms.items():
    print(f"  {k:25s}: {v:.2e}")

print(f"\nAt L = 1 m (macroscopic scale):")
terms, _ = ohm_law_terms(n, T_e, B, 1.0, V)
for k, v in terms.items():
    print(f"  {k:25s}: {v:.2e}")
```

### 6.3 반자성 표류 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def diamagnetic_drift_cylinder():
    """
    Visualize diamagnetic drift in a cylindrical plasma column.
    """
    # Plasma parameters
    a = 0.1  # plasma radius (m)
    B_0 = 1.0  # axial magnetic field (T)
    p_0 = 1e5  # peak pressure (Pa)
    T_e = 10  # electron temperature (eV)
    T_i = 10  # ion temperature (eV)
    n_0 = 1e19  # peak density (m^-3)

    e = 1.6e-19
    k_B = 1.38e-23

    # Radial grid
    r = np.linspace(0, a, 100)

    # Pressure profile (parabolic)
    p = p_0 * (1 - (r/a)**2)
    p_e = p / 2
    p_i = p / 2
    n = n_0 * (1 - (r/a)**2)

    # Pressure gradient
    dp_dr = -2 * p_0 * r / a**2
    dp_e_dr = dp_dr / 2
    dp_i_dr = dp_dr / 2

    # Diamagnetic drift velocities
    v_star_e = -dp_e_dr / (e * n * B_0)  # azimuthal (θ) direction
    v_star_i = dp_i_dr / (e * n * B_0)

    # Diamagnetic current density
    J_theta = -dp_dr / B_0

    # Plot profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Pressure profile
    axes[0, 0].plot(r*100, p/1e3, 'b-', linewidth=2, label='Total')
    axes[0, 0].plot(r*100, p_e/1e3, 'r--', linewidth=2, label='Electron')
    axes[0, 0].plot(r*100, p_i/1e3, 'g--', linewidth=2, label='Ion')
    axes[0, 0].set_xlabel('Radius (cm)', fontsize=11)
    axes[0, 0].set_ylabel('Pressure (kPa)', fontsize=11)
    axes[0, 0].set_title('Pressure Profile', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)

    # Diamagnetic drift velocities
    axes[0, 1].plot(r*100, v_star_e/1e3, 'r-', linewidth=2, label='Electron')
    axes[0, 1].plot(r*100, v_star_i/1e3, 'g-', linewidth=2, label='Ion')
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Radius (cm)', fontsize=11)
    axes[0, 1].set_ylabel('Drift velocity (km/s)', fontsize=11)
    axes[0, 1].set_title('Diamagnetic Drift Velocity (azimuthal)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Diamagnetic current
    axes[1, 0].plot(r*100, J_theta/1e3, 'b-', linewidth=2)
    axes[1, 0].set_xlabel('Radius (cm)', fontsize=11)
    axes[1, 0].set_ylabel('Current density (kA/m²)', fontsize=11)
    axes[1, 0].set_title('Diamagnetic Current Density (azimuthal)', fontsize=12)
    axes[1, 0].grid(alpha=0.3)

    # 2D visualization: top view
    ax = axes[1, 1]
    theta = np.linspace(0, 2*np.pi, 50)
    R, Theta = np.meshgrid(r, theta)
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)

    # Pressure contour
    P_grid = np.outer(np.ones_like(theta), p)
    contour = ax.contourf(X*100, Y*100, P_grid/1e3, levels=20, cmap='hot')
    plt.colorbar(contour, ax=ax, label='Pressure (kPa)')

    # Velocity vectors (sample points)
    n_arrows = 8
    r_arrows = np.linspace(0.2*a, 0.9*a, 5)
    theta_arrows = np.linspace(0, 2*np.pi, n_arrows, endpoint=False)

    for ri in r_arrows:
        for ti in theta_arrows:
            xi = ri * np.cos(ti)
            yi = ri * np.sin(ti)

            # Diamagnetic drift is in theta direction
            # In Cartesian: v_theta = -sin(θ) v_r_hat + cos(θ) v_θ_hat
            idx = np.argmin(np.abs(r - ri))
            v_mag = v_star_e[idx]

            vx = -v_mag * np.sin(ti)
            vy = v_mag * np.cos(ti)

            ax.arrow(xi*100, yi*100, vx*1e-3, vy*1e-3,
                    head_width=0.5, head_length=0.3, fc='cyan', ec='cyan', alpha=0.8)

    ax.set_xlabel('x (cm)', fontsize=11)
    ax.set_ylabel('y (cm)', fontsize=11)
    ax.set_title('Electron Diamagnetic Drift (top view)', fontsize=12)
    ax.set_aspect('equal')
    ax.add_patch(Circle((0, 0), a*100, fill=False, edgecolor='white', linewidth=2))

    plt.tight_layout()
    plt.savefig('diamagnetic_drift.png', dpi=150)
    plt.show()

    # Print values at r = a/2
    idx = np.argmin(np.abs(r - a/2))
    print(f"\nAt r = a/2 = {a/2*100:.1f} cm:")
    print(f"  Pressure: {p[idx]/1e3:.2f} kPa")
    print(f"  Electron drift: {v_star_e[idx]/1e3:.2f} km/s")
    print(f"  Ion drift: {v_star_i[idx]/1e3:.2f} km/s")
    print(f"  Current density: {J_theta[idx]/1e3:.2f} kA/m²")
    print(f"  Drift frequency: {v_star_e[idx]/(a/2):.2e} rad/s")
    print(f"  Compare to ω_ci = {e*B_0/(1.67e-27):.2e} rad/s")

diamagnetic_drift_cylinder()
```

### 6.4 이유체 닫힘 비교

```python
import numpy as np
import matplotlib.pyplot as plt

def compare_closures():
    """
    Compare different closure models: isothermal vs. adiabatic.
    Simulate compression of a plasma element.
    """
    # Initial conditions
    n_0 = 1e19  # m^-3
    T_0 = 10    # eV
    V_0 = 1.0   # m^3

    gamma = 5/3  # adiabatic index

    # Compression ratio
    V = np.linspace(V_0, 0.1*V_0, 100)
    n = n_0 * (V_0 / V)  # density increases as volume decreases

    # Isothermal: T = const
    T_isothermal = np.ones_like(V) * T_0
    p_isothermal = n * T_isothermal

    # Adiabatic: p V^γ = const
    p_adiabatic = n_0 * T_0 * (V_0 / V)**gamma
    T_adiabatic = p_adiabatic / n

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Temperature vs. compression
    axes[0].plot(V/V_0, T_isothermal, 'b-', linewidth=2, label='Isothermal')
    axes[0].plot(V/V_0, T_adiabatic, 'r--', linewidth=2, label='Adiabatic (γ=5/3)')
    axes[0].set_xlabel('V / V₀', fontsize=12)
    axes[0].set_ylabel('Temperature (eV)', fontsize=12)
    axes[0].set_title('Temperature Evolution under Compression', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Pressure vs. density
    axes[1].loglog(n/n_0, p_isothermal/(n_0*T_0), 'b-', linewidth=2, label='Isothermal (p ∝ n)')
    axes[1].loglog(n/n_0, p_adiabatic/(n_0*T_0), 'r--', linewidth=2, label='Adiabatic (p ∝ n^γ)')
    axes[1].set_xlabel('n / n₀', fontsize=12)
    axes[1].set_ylabel('p / (n₀ T₀)', fontsize=12)
    axes[1].set_title('Pressure vs. Density', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig('closure_comparison.png', dpi=150)
    plt.show()

    # At 50% compression
    idx = np.argmin(np.abs(V/V_0 - 0.5))
    print("\nAt 50% compression (V = 0.5 V₀):")
    print(f"  Density: {n[idx]/n_0:.2f} n₀")
    print(f"  Isothermal:")
    print(f"    T = {T_isothermal[idx]:.2f} eV (unchanged)")
    print(f"    p = {p_isothermal[idx]/(n_0*T_0):.2f} (n₀ T₀)")
    print(f"  Adiabatic:")
    print(f"    T = {T_adiabatic[idx]:.2f} eV")
    print(f"    p = {p_adiabatic[idx]/(n_0*T_0):.2f} (n₀ T₀)")
    print(f"  Pressure ratio (adiabatic/isothermal): {p_adiabatic[idx]/p_isothermal[idx]:.2f}")

compare_closures()
```

## 요약

이 수업에서 우리는 Vlasov 방정식의 속도 공간 모멘트를 취하여 이유체 모델을 유도했습니다. 핵심 포인트:

1. **모멘트 계층**: 각 모멘트 방정식은 다음 고차 모멘트를 도입하여 닫힘 문제를 야기합니다.

2. **닫힘 모델**: 등온, 단열, CGL 닫힘은 다른 물리적 가정으로 계층을 절단합니다.

3. **일반화된 Ohm의 법칙**: 완전한 형태는 저항, Hall, 전자 압력, 전자 관성 항을 포함합니다. 각 항은 다른 길이 스케일에서 중요해집니다:
   - 저항: 낮은 $R_m$ (충돌 플라즈마)
   - Hall: $L \sim d_i$ (이온 skin depth)
   - 전자 압력: 급격한 경사
   - 전자 관성: $L \sim d_e$ (전자 skin depth)

4. **Hall 효과**: $\lesssim d_i$ 스케일에서, 이온은 자기장으로부터 분리되지만 전자는 동결된 상태로 남습니다. 이는 빠른 자기 재결합과 whistler 파동 전파를 가능하게 합니다.

5. **반자성 표류**: 압력 경사에서 발생하는 유체 표류로, 단일 입자 표류가 아닙니다. 전류 $\mathbf{J}_* = \mathbf{B} \times \nabla p / B^2$를 만듭니다.

6. **이유체 파동**: Hall MHD는 작은 스케일에서 Alfvén 파동을 whistler 파동으로 수정합니다. Kinetic Alfvén 파동은 유한-$k_\perp$ 효과를 포함합니다.

이유체 모델은 단일 입자 운동 이론과 단일 유체 MHD 사이의 간격을 메웁니다. 이상적 MHD에서 놓치지만 운동 이론의 완전한 복잡성을 필요로 하지 않는 중간 스케일(이온 gyroradius에서 이온 skin depth까지)에서 중요한 물리를 포착합니다.

## 연습 문제

### 문제 1: 모멘트 계산
Vlasov 방정식에서 시작하여, 2차 모멘트(에너지 방정식)을 명시적으로 유도하십시오. 열유속 $\mathbf{q}_s = \frac{1}{2} m_s \int w^2 \mathbf{w} f_s d^3v$가 나타남을 보이십시오. 열유속은 어떤 물리적 과정을 나타냅니까?

### 문제 2: Hall MHD 분산
Hall MHD에서 whistler 파동에 대한 분산 관계를 유도하십시오:
$$\omega = \frac{k_\parallel^2 V_A^2}{\omega_{ci}}$$
Hall 항을 가진 이유체 방정식에서 시작하여, $\omega \ll \omega_{ce}$와 $\omega \gg \omega_{ci}$를 가정하고, 냉 플라즈마 근사($p = 0$)를 사용하십시오.

### 문제 3: 토카막의 반자성 전류
주반경 $R_0 = 3$ m, 부반경 $a = 1$ m인 토카막에서, 전자 압력 프로파일은:
$$p_e(r) = p_0 \left(1 - \frac{r^2}{a^2}\right)^2$$
$p_0 = 5 \times 10^5$ Pa입니다. 토로이달 자기장은 $B_\phi = 5$ T입니다. 계산:
(a) $r = a/2$에서 반자성 전류 밀도.
(b) 반자성 효과로부터의 총 폴로이달 전류(단면에 대해 $J_\theta$ 적분).
(c) 이것을 bootstrap 전류와 비교(유사한 프로파일을 가짐).

### 문제 4: 전류 시트에서의 일반화된 Ohm의 법칙
자기 재결합 전류 시트에서, 길이 스케일은 $L = 10 d_i$이고, 여기서 $d_i = 100$ km는 이온 skin depth입니다. 플라즈마 밀도는 $n = 10^7$ m$^{-3}$ (태양풍), 전자 온도 $T_e = 100$ eV, 자기장 $B = 10$ nT입니다. 다음의 상대적 크기 계산:
(a) 이상적 MHD 항 $\mathbf{v} \times \mathbf{B}$
(b) Hall 항 $\mathbf{J} \times \mathbf{B} / (en)$
(c) 전자 압력 항 $\nabla p_e / (en)$
(d) 전자 관성 항
이 전류 시트에서 어떤 항이 중요합니까?

### 문제 5: 이유체 불안정성
$T_e = T_i$이고 자기장 $\mathbf{B} = B_0 \hat{\mathbf{z}}$에서 밀도 경사 $\nabla n = -n_0 / L_n \hat{\mathbf{x}}$를 가진 이유체 플라즈마를 고려하십시오.
(a) 전자와 이온 반자성 표류 속도를 계산하십시오.
(b) 드리프트 파동 불안정성은 밀도와 전위 섭동 사이의 위상차가 파동 성장을 야기할 때 발생합니다. 연속 방정식과 준중성을 사용하여, 정전 드리프트 파동이 분산 관계를 가짐을 보이십시오:
$$\omega = \frac{k_y k_B T_e}{e B_0 L_n}$$
여기서 $k_y$는 $\mathbf{B}$와 $\nabla n$ 모두에 수직인 파수입니다.
(c) $L_n = 1$ cm, $T_e = 1$ eV, $B_0 = 0.1$ T, $k_y = 100$ m$^{-1}$에 대해, 드리프트 파동 주파수를 계산하십시오.

---

**이전**: [Wave Heating and Instabilities](./12_Wave_Heating_and_Instabilities.md) | **다음**: [From Kinetic to MHD](./14_From_Kinetic_to_MHD.md)
