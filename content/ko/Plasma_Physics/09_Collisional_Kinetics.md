# 9. 충돌 동역학

## 학습 목표

- Boltzmann 충돌 연산자와 그 물리적 의미 이해
- 플라즈마의 Coulomb 충돌에 대한 Fokker-Planck 방정식 유도
- 효율적인 충돌 계산을 위한 Rosenbluth potential 공식화 습득
- 자화 플라즈마에서 Braginskii 수송 이론과 수송 계수 학습
- 토로이달 confinement 장치에서 neoclassical 수송 regime 이해
- 입자 감속 및 저항성과 같은 실제 문제에 충돌 연산자 적용

## 서론

이전 강의에서는 플라즈마를 Vlasov 방정식으로 지배되는 무충돌 시스템으로 다루었습니다. 그러나 실제 플라즈마는 중성 기체보다 훨씬 낮은 비율이지만 충돌을 경험합니다. 충돌은 다음 측면에서 중요합니다:

- 열화(thermalization) 및 평형으로의 접근
- 전기 저항성 및 에너지 소산
- 자기장을 가로지르는 입자, 운동량, 에너지의 수송
- 토로이달 confinement의 neoclassical 효과

플라즈마 충돌 이론의 과제는 전자기적(Coulomb) 상호작용이 **장거리**라는 것입니다. 중성 기체의 hard-sphere 충돌과 달리, 하전 입자는 $1/r^2$ 전기장을 통해 먼 거리에서 상호작용합니다. 이는 큰 각도의 "hard" 충돌보다 **작은 각도 편향**이 우세하게 됩니다.

이 강의에서는 일반 Boltzmann 연산자에서 시작하여 플라즈마에 대한 Fokker-Planck 방정식으로 특수화하고 실용적인 응용을 위한 수송 계수를 유도하여 충돌의 운동 이론을 개발합니다.

## 1. Boltzmann 충돌 연산자

### 1.1 일반 공식화

충돌을 포함한 완전한 운동 방정식은:

$$\frac{\partial f}{\partial t} + \mathbf{v}\cdot\frac{\partial f}{\partial \mathbf{x}} + \mathbf{a}\cdot\frac{\partial f}{\partial \mathbf{v}} = \left(\frac{\partial f}{\partial t}\right)_{\text{coll}}$$

여기서 $\mathbf{a}$는 외부 장으로부터의 가속도입니다. 충돌 항은 이진 충돌로 인한 $f$의 순 변화율을 나타냅니다.

단면적 $\sigma(\mathbf{v}, \mathbf{v}_1; \mathbf{v}', \mathbf{v}_1')$을 가진 이진 충돌에 대해 **Boltzmann 충돌 연산자**는:

$$\left(\frac{\partial f}{\partial t}\right)_{\text{coll}} = \int\int\int \left[f(\mathbf{v}')f(\mathbf{v}_1') - f(\mathbf{v})f(\mathbf{v}_1)\right] |\mathbf{v}-\mathbf{v}_1| \sigma(\Omega)\, d\Omega\, d^3v_1$$

물리적 해석:
- **획득 항**: $f(\mathbf{v}')f(\mathbf{v}_1')$ - $(\mathbf{v}', \mathbf{v}_1')$로부터 속도 $\mathbf{v}$로 산란된 입자
- **손실 항**: $f(\mathbf{v})f(\mathbf{v}_1)$ - 속도 $\mathbf{v}$에서 산란된 입자
- $|\mathbf{v}-\mathbf{v}_1|$ - 상대 속도가 충돌률을 결정
- 모든 충돌 파트너 $\mathbf{v}_1$ 및 산란 각 $\Omega$에 대한 적분

프라임은 충돌 후 속도를 나타냅니다. 보존 법칙의 제약:
- 운동량: $\mathbf{v} + \mathbf{v}_1 = \mathbf{v}' + \mathbf{v}_1'$
- 에너지: $v^2 + v_1^2 = v'^2 + v_1'^2$

### 1.2 충돌 불변량과 보존 법칙

양 $\psi(\mathbf{v})$가 **충돌 불변량**이 되려면:

$$\int \psi(\mathbf{v}) C[f]\, d^3v = 0$$

모든 분포 $f$에 대해 성립하며, 여기서 $C[f]$는 충돌 연산자입니다.

**합 불변량(summational invariants)**은 다음을 만족합니다:
$$\psi(\mathbf{v}) + \psi(\mathbf{v}_1) = \psi(\mathbf{v}') + \psi(\mathbf{v}_1')$$

다섯 가지 합 불변량은:
1. 질량: $\psi = m$ (입자 보존)
2. 운동량: $\psi = m\mathbf{v}$ (세 성분)
3. 에너지: $\psi = \frac{1}{2}mv^2$

이들은 보존 법칙으로 이어집니다:

```
Particle number:   ∫ C[f] d³v = 0
Momentum:          ∫ mv C[f] d³v = 0
Energy:            ∫ (½mv²) C[f] d³v = 0
```

### 1.3 Boltzmann의 H-정리

**H-함수**를 정의합니다:
$$H(t) = \int f(\mathbf{v},t) \ln f(\mathbf{v},t)\, d^3v$$

Boltzmann은 상세 균형을 만족하는 모든 충돌 연산자에 대해 $dH/dt \leq 0$임을 증명했습니다. 이것이 엔트로피 $S = -k_B H$가 항상 증가함을 보여주는 **H-정리**입니다.

증명은 다음에 의존합니다:
$$\frac{dH}{dt} = \int (\ln f + 1) C[f]\, d^3v$$

충돌 적분의 대칭성과 부등식 $\ln x \leq x - 1$을 사용하여, 다음 조건일 때만 등호가 성립하는 $dH/dt \leq 0$을 보입니다:

$$f(\mathbf{v})f(\mathbf{v}_1) = f(\mathbf{v}')f(\mathbf{v}_1')$$

모든 충돌 쌍에 대해. 이 조건은 **Maxwellian 분포**에 의해 만족됩니다:

$$f_M(\mathbf{v}) = n\left(\frac{m}{2\pi k_B T}\right)^{3/2} \exp\left(-\frac{m v^2}{2k_B T}\right)$$

따라서 충돌은 모든 분포를 Maxwellian 평형으로 이끕니다.

### 1.4 충돌 주파수 추정

대략적인 추정을 위해, 차원 분석은 다음을 제공합니다:

$$\nu_{\text{coll}} \sim n \sigma v_{\text{th}}$$

Coulomb 충돌의 경우, 유효 단면적은:
$$\sigma \sim \pi b_{90}^2$$

여기서 $b_{90}$는 90° 편향에 대한 충돌 매개변수입니다:
$$b_{90} = \frac{e^2}{4\pi\epsilon_0 m v_{\text{th}}^2}$$

이는 다음을 제공합니다:
$$\nu_{ei} \sim \frac{n e^4 \ln\Lambda}{4\pi\epsilon_0^2 m_e v_{\text{th}}^3} \sim \frac{n e^4 \ln\Lambda}{4\pi\epsilon_0^2 (k_B T)^{3/2} m_e^{1/2}}$$

**Coulomb 로그** $\ln\Lambda$는 충돌 매개변수에 대한 적분에서 발생합니다:
$$\ln\Lambda = \ln\left(\frac{\lambda_D}{b_{90}}\right) \approx 10-20$$

대부분의 플라즈마에 대해. 전형적인 값들:
- 핵융합 플라즈마 (ITER): $\ln\Lambda \approx 17$
- 태양 코로나: $\ln\Lambda \approx 20$
- 실험실 플라즈마: $\ln\Lambda \approx 10-15$

## 2. 플라즈마에 대한 Fokker-Planck 방정식

### 2.1 작은 각도 산란으로부터의 유도

Coulomb 충돌은 드문 큰 각도 충돌보다 **많은 작은 각도 편향**에 의해 지배됩니다. 시간 $\Delta t$ 동안 많은 약한 산란의 누적 효과를 고려합니다:

$$\Delta \mathbf{v} = \sum_{i=1}^{N} \Delta \mathbf{v}_i$$

여기서 $N \sim n \sigma v \Delta t$는 충돌 횟수입니다.

작은 편향의 경우:
- 평균 편향: $\langle \Delta \mathbf{v} \rangle \sim N \langle \Delta v_i \rangle \propto \Delta t$
- 분산: $\langle (\Delta \mathbf{v})^2 \rangle \sim N \langle (\Delta v_i)^2 \rangle \propto \Delta t$

많은 상관되지 않은 산란이 발생하므로 중심극한정리가 적용됩니다. 분포 함수의 변화는 다음과 같이 전개할 수 있습니다:

$$f(\mathbf{v} + \Delta\mathbf{v}, t + \Delta t) - f(\mathbf{v}, t) = \Delta t \left(\frac{\partial f}{\partial t}\right)_{\text{coll}}$$

왼쪽을 2차까지 전개하면 (분산이 $\Delta t$에서 1차이므로):

$$f(\mathbf{v},t) + \frac{\partial f}{\partial v_i}\langle\Delta v_i\rangle + \frac{1}{2}\frac{\partial^2 f}{\partial v_i \partial v_j}\langle\Delta v_i \Delta v_j\rangle - f(\mathbf{v},t) = \Delta t \left(\frac{\partial f}{\partial t}\right)_{\text{coll}}$$

이는 **Fokker-Planck 충돌 연산자**를 제공합니다:

$$\boxed{\left(\frac{\partial f}{\partial t}\right)_{\text{coll}} = -\frac{\partial}{\partial v_i}\left[f \langle\Delta v_i\rangle\right] + \frac{1}{2}\frac{\partial^2}{\partial v_i \partial v_j}\left[f \langle\Delta v_i \Delta v_j\rangle\right]}$$

(반복되는 인덱스에 대한 Einstein 합 규약이 암시됨.)

두 항은 다음을 나타냅니다:
1. **동역학적 마찰** (항력): $\langle\Delta v_i\rangle$ - 체계적 감속
2. **속도 확산**: $\langle\Delta v_i \Delta v_j\rangle$ - 속도 공간에서 무작위 보행

### 2.2 Rosenbluth Potentials

Coulomb 산란으로부터 $\langle\Delta v_i\rangle$와 $\langle\Delta v_i \Delta v_j\rangle$를 직접 계산하는 것은 번거롭습니다. **Rosenbluth (1957)**는 이들이 두 개의 포텐셜로 표현될 수 있음을 보였습니다.

정의:
$$\boxed{h(\mathbf{v}) = \int \frac{f(\mathbf{v}')}{|\mathbf{v}-\mathbf{v}'|}\, d^3v'}$$

$$\boxed{g(\mathbf{v}) = \int |\mathbf{v}-\mathbf{v}'| f(\mathbf{v}')\, d^3v'}$$

이들이 **Rosenbluth potentials**입니다 (일부 문헌에서는 $H$와 $G$로도 불림).

마찰 및 확산 계수는:

$$\langle\Delta v_i\rangle = -\Gamma \frac{\partial g}{\partial v_i}$$

$$\langle\Delta v_i \Delta v_j\rangle = \Gamma \frac{\partial^2 h}{\partial v_i \partial v_j}$$

여기서 $\Gamma = \frac{e^4 \ln\Lambda}{4\pi\epsilon_0^2 m^2}$는 상수입니다.

Fokker-Planck 연산자는:

$$\boxed{C[f] = \Gamma \frac{\partial}{\partial v_i}\left[f \frac{\partial g}{\partial v_i} + \frac{\partial}{\partial v_j}\left(f \frac{\partial^2 h}{\partial v_i \partial v_j}\right)\right]}$$

이 형태는 계산적으로 효율적입니다: $h$와 $g$를 적분을 통해 한 번 계산한 다음 미분을 평가합니다.

### 2.3 동종 입자 및 이종 입자 충돌

여러 종(전자, 이온)을 가진 플라즈마의 경우 다음을 고려해야 합니다:

**전자-전자 충돌**: $C_{ee}[f_e]$
**이온-이온 충돌**: $C_{ii}[f_i]$
**전자-이온 충돌**: $C_{ei}[f_e]$ 및 $C_{ie}[f_i]$

질량비 $m_i/m_e \approx 1836$ (양성자의 경우)는 전자-이온 충돌을 단순화합니다:
- 전자는 이온에게 에너지를 천천히 잃음 (열화하는 데 많은 충돌 필요)
- 이온은 전자로부터 에너지를 거의 얻지 못하지만 운동량 항력을 경험

종 간 **에너지 교환율**은:

$$\frac{dT_e}{dt} \bigg|_{\text{coll}} = -\nu_{eq}(T_e - T_i)$$

여기서 **평형화 주파수**는:

$$\nu_{eq} = \frac{m_e}{m_i} \nu_{ei} \sim \frac{1}{1836} \nu_{ei}$$

이는 많은 플라즈마에서 전자와 이온 온도가 크게 다를 수 있는 이유를 설명합니다.

### 2.4 Fokker-Planck 연산자의 Landau 형태

대안적 형태는 텐서 구조를 강조합니다. 다음을 정의합니다:

$$\mathbf{A}(\mathbf{v}) = \int \frac{(\mathbf{v}-\mathbf{v}')}{|\mathbf{v}-\mathbf{v}'|^3} f(\mathbf{v}')\, d^3v'$$

$$\overleftrightarrow{B}(\mathbf{v}) = \int \frac{\overleftrightarrow{I} - \hat{\mathbf{v}}\hat{\mathbf{v}}}{|\mathbf{v}-\mathbf{v}'|} f(\mathbf{v}')\, d^3v'$$

여기서 $\overleftrightarrow{I}$는 단위 텐서이고 $\hat{\mathbf{v}} = (\mathbf{v}-\mathbf{v}')/|\mathbf{v}-\mathbf{v}'|$입니다.

그러면:
$$C[f] = \Gamma \nabla_v \cdot \left[f \mathbf{A} + \nabla_v \cdot (f \overleftrightarrow{B})\right]$$

이것이 해석적 계산에 유용한 **Landau 형태**입니다.

## 3. 시험 입자 감속

### 3.1 Maxwellian 배경에서 전자 감속

핵융합 알파 입자, 도주 전자(runaway electrons) 또는 NBI로부터의 빠른 전자가 열 배경 플라즈마에서 감속하는 것을 고려합니다.

열 속도 $v_{\text{th}}$보다 훨씬 큰 속도 $\mathbf{v}$를 가진 시험 입자의 경우, 마찰력은 단순화됩니다:

$$\frac{d\mathbf{v}}{dt} = -\nu_s \frac{\mathbf{v}}{v}$$

여기서 **감속 주파수**는:

$$\nu_s = \frac{n e^4 \ln\Lambda}{4\pi\epsilon_0^2 m_e^2 v^3} \cdot \Phi\left(\frac{v}{v_{\text{th}}}\right)$$

$v \gg v_{\text{th}}$의 경우, $\Phi \approx 1$이므로:

$$\frac{dv}{dt} = -\nu_s \propto -\frac{1}{v^3}$$

풀면:
$$v^4 - v_0^4 = -4\nu_s' t$$

$v_0$에서 $v_{\text{th}}$까지의 감속 시간은:

$$\tau_s = \frac{v_0^3}{4\nu_s(v_0)} \sim \frac{4\pi\epsilon_0^2 m_e^2 v_0^3}{n e^4 \ln\Lambda}$$

핵융합 플라즈마에서 3.5 MeV 알파 입자의 경우:
- $n = 10^{20}$ m$^{-3}$, $T = 10$ keV
- $\tau_s \approx 1$ 초

이는 많은 장치의 confinement 시간보다 훨씬 길므로 알파가 플라즈마를 효과적으로 가열할 수 있습니다.

### 3.2 임계 속도

빠른 입자가 감속할 때, 전자와 이온 모두에 에너지를 전달합니다. **임계 속도** $v_c$는 전자에 대한 항력이 이온에 대한 항력과 같은 곳입니다:

$$\nu_e(v_c) = \nu_i(v_c)$$

$v > v_c$의 경우: 주로 전자 가열
$v < v_c$의 경우: 주로 이온 가열

임계 속도는:

$$v_c \approx v_{\text{th},e} \left(\frac{m_i}{m_e}\right)^{1/3} Z^{2/3}$$

중수소 플라즈마 ($Z=1$)의 경우:
$$v_c \approx 12.2 \, v_{\text{th},e}$$

이는 다음 에너지에 해당합니다:
$$E_c = \frac{1}{2}m_e v_c^2 \approx 14.8 \, T_e$$

$E > E_c$를 가진 입자는 전자를 가열하고, $E < E_c$는 이온을 가열합니다.

### 3.3 도주 전자

전기장 $E$의 존재 하에서 힘의 균형은:

$$eE = m_e \nu_s v$$

높은 속도에서 $\nu_s \propto v^{-3}$이므로 마찰은 속도와 함께 감소합니다. $eE$가 최대 마찰력을 초과하면, 전자는 임의로 높은 에너지로 **도주**합니다.

**Dreicer 장**은:

$$E_D = \frac{n e^3 \ln\Lambda}{4\pi\epsilon_0^2 k_B T_e}$$

$E > E_D$의 경우, 상당한 비율의 전자가 도주합니다. 이는 토카막에서 위험합니다:
- 붕괴(disruption) 동안 $E$가 급증
- 도주 전자가 10-100 MeV로 가속
- 플라즈마 대면 구성요소를 손상시킬 수 있음

완화 전략: 대량 가스 주입, shattered pellet 주입.

## 4. Braginskii 수송 이론

### 4.1 모멘트 접근법

완전한 Fokker-Planck 방정식을 푸는 대신, **속도 모멘트**를 취하여 충돌 항이 있는 유체 방정식을 유도할 수 있습니다.

모멘트 정의:
- 밀도: $n = \int f\, d^3v$
- 유동 속도: $\mathbf{u} = \frac{1}{n}\int \mathbf{v} f\, d^3v$
- 압력 텐서: $\overleftrightarrow{P} = m \int (\mathbf{v}-\mathbf{u})(\mathbf{v}-\mathbf{u}) f\, d^3v$
- 열 플럭스: $\mathbf{q} = \frac{m}{2} \int (\mathbf{v}-\mathbf{u})^2 (\mathbf{v}-\mathbf{u}) f\, d^3v$

운동 방정식의 모멘트를 취하면 위계를 얻습니다:

```
Continuity:        ∂n/∂t + ∇·(nu) = 0
Momentum:          mn(∂u/∂t + u·∇u) = -∇·P + F_coll
Energy:            ∂/∂t(3nT/2) + ∇·q = Q_coll
```

충돌 항 $F_{\text{coll}}$과 $Q_{\text{coll}}$은 $C[f]$의 모멘트를 포함합니다.

**Braginskii (1965)**는 Maxwellian으로부터의 작은 편차를 가정하여 Fokker-Planck 방정식을 섭동적으로 풀어 **수송 계수**를 유도했습니다.

### 4.2 평행 수송

자기장 선을 따라 입자는 자유롭게 흐릅니다 (충돌은 약함). 평행 수송 계수는:

**평행 점성**:
$$\eta_{\parallel} = 0.96 \, n T \tau$$

여기서 $\tau = 1/\nu$는 충돌 시간입니다.

**평행 열전도도**:
$$\kappa_{\parallel,e} = 3.16 \, \frac{n T \tau}{m_e}$$

$$\kappa_{\parallel,i} = 3.9 \, \frac{n T \tau}{m_i}$$

**전기 전도도** (저항성의 역수):
$$\sigma_{\parallel} = \frac{n e^2 \tau}{m_e} = \frac{1.96 \, n e^2 \tau}{m_e}$$

**고전적 저항성**은:
$$\eta_{\text{classical}} = \frac{1}{\sigma_{\parallel}} = \frac{m_e}{1.96 \, n e^2 \tau} \propto T^{-3/2}$$

수치 값:
$$\eta_{\text{classical}} \approx 5.2 \times 10^{-5} \frac{\ln\Lambda}{T_e^{3/2}} \quad (\Omega\cdot\text{m}, \, T_e \text{ in eV})$$

### 4.3 수직 수송

자기장 선을 가로질러, 입자는 충돌을 통해 확산해야 합니다 (Larmor 궤도가 그들을 가둠). 수직 수송은 **훨씬 더 약합니다**.

**수직 열전도도**:
$$\kappa_{\perp,e} = 4.66 \, \frac{n T}{m_e \omega_{ce}^2 \tau}$$

$$\kappa_{\perp,i} = 2.0 \, \frac{n T}{m_i \omega_{ci}^2 \tau}$$

평행 대 수직의 비율은:

$$\frac{\kappa_{\parallel}}{\kappa_{\perp}} \sim (\omega_c \tau)^2$$

핵융합 플라즈마 ($B = 5$ T, $T_e = 10$ keV, $n = 10^{20}$ m$^{-3}$)의 경우:
- $\omega_{ce} \tau \approx 10^6$
- $\kappa_{\parallel}/\kappa_{\perp} \approx 10^{12}$

이 엄청난 비등방성은 열이 거의 전적으로 장 선을 따라 흐른다는 것을 의미합니다.

**수직 점성**:
다양한 응력 성분에 대해 여러 계수 $\eta_0, \eta_1, \eta_2, \eta_3, \eta_4$를 포함합니다. 핵심 결과는 수직 운동량 수송도 $(\omega_c \tau)^{-2}$에 의해 억제된다는 것입니다.

### 4.4 이상 수송

실제 실험에서 관찰된 수송은 종종 고전적 Braginskii 예측보다 **수 자릿수 더 큽니다**. 이것이 다음에 의한 **이상 수송**입니다:

- 난류 (drift waves, ITG, ETG modes)
- 자기장 섭동
- 비-Maxwellian 분포

경험적 스케일링 법칙 (예: ITER H-mode confinement):
$$\tau_E \sim I_p^{0.93} B^{0.15} P^{-0.69} n^{0.41} M^{0.19} R^{1.97} \epsilon^{0.58} \kappa^{0.78}$$

여기서 $I_p$는 플라즈마 전류, $P$는 가열 전력, $M$은 질량, $R$은 주반경, $\epsilon$은 역 aspect ratio, $\kappa$는 elongation입니다.

이상 수송의 이해와 제어는 핵융합 연구의 중심 과제입니다.

## 5. Neoclassical 수송

### 5.1 토로이달 기하학 효과

**토러스** (토카막, stellarator)에서 자기장 강도가 변합니다:
$$B(\theta) = B_0 \left(1 + \epsilon \cos\theta\right)$$

여기서 $\epsilon = r/R$은 역 aspect ratio이고 $\theta$는 poloidal 각입니다.

작은 평행 속도를 가진 입자는 토러스 외부의 저장 영역에 **포획**될 수 있습니다. 그들은 앞뒤로 튀며 전체 poloidal 회로를 완성하지 못합니다.

포획된 입자의 비율은:
$$f_{\text{trapped}} \sim \sqrt{\epsilon}$$

ITER ($\epsilon \sim 0.3$)의 경우: $f_{\text{trapped}} \sim 0.5$ (50% 포획).

### 5.2 바나나 궤도

포획된 입자는 **바나나 궤도**를 실행합니다: 그들의 드리프트 궤도는 poloidal 단면에서 바나나 모양의 영역을 형성합니다.

```
       |  Passing particles:
   ____|____   complete full poloidal circuit
  /    |    \
 /     |     \   Trapped particles:
|      |      |  bounce in "banana" orbit
|    __|__    |  on outer side
 \   /   \   /
  \_/     \_/

     ^
   Low B region
```

바나나 폭은:
$$\Delta_b \sim \rho_L \sqrt{\epsilon}$$

여기서 $\rho_L$은 Larmor 반경입니다.

### 5.3 충돌성 Regime

Neoclassical 수송은 **충돌성** $\nu_* = \nu / \omega_b$에 의존하며, 여기서 $\omega_b$는 bounce 주파수입니다.

**바나나 regime** ($\nu_* \ll 1$): 입자가 충돌 전에 많은 bounce를 완성
- 수송: $D \sim D_{\text{classical}} / \epsilon^{3/2}$
- 고전적 대비 향상: $1/\epsilon^{3/2}$

**고원 regime** ($\nu_* \sim 1$): 충돌 주파수 $\sim$ bounce 주파수
- 수송: $D \sim D_{\text{classical}} / \epsilon$
- 중간 향상

**Pfirsch-Schlüter regime** ($\nu_* \gg 1$): bounce당 많은 충돌
- 수송: $D \sim D_{\text{classical}} \cdot q^2$
- 안전 인자 제곱에 의한 향상

전형적인 토카막 매개변수는 전자를 plateau/banana regime에, 이온을 banana regime에 배치합니다.

### 5.4 Bootstrap 전류

주목할 만한 neoclassical 효과: **자가 생성 전류**가 외부 구동 없이 흐릅니다.

물리적 기원: 포획된 입자는 불균형한 마찰력을 가짐
- 내향 구간: 마찰이 입자를 감속
- 외향 구간: 다른 분포 → 다른 마찰
- 순 운동량 전달 → 전류

**Bootstrap 전류**는:

$$j_{\text{bs}} \sim \frac{n T}{eB_p} \left(\frac{d \ln p}{dr}\right) f_{\text{bs}}(\nu_*)$$

여기서 $B_p$는 poloidal 장이고 $f_{\text{bs}}$는 수치 함수입니다.

ITER 매개변수의 경우:
- Bootstrap 비율: 전체 전류의 $\sim 20-30\%$
- 외부 전류 구동 필요성 감소

고급 시나리오는 $\sim 100\%$ bootstrap 전류를 목표로 합니다 (정상 상태 작동).

## 6. Python 구현

### 6.1 감속을 위한 Fokker-Planck 솔버

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Physical constants
e = 1.602e-19  # C
epsilon_0 = 8.854e-12  # F/m
m_e = 9.109e-31  # kg
k_B = 1.381e-23  # J/K

def slowing_down_frequency(v, n, T_e, ln_Lambda=15):
    """
    Slowing-down frequency for test particle in thermal background.

    Parameters:
    -----------
    v : float
        Test particle velocity (m/s)
    n : float
        Background density (m^-3)
    T_e : float
        Background temperature (eV)
    ln_Lambda : float
        Coulomb logarithm

    Returns:
    --------
    nu_s : float
        Slowing-down frequency (s^-1)
    """
    T_J = T_e * e  # Convert to Joules
    v_th = np.sqrt(2 * T_J / m_e)

    # Gamma는 기본적인 Coulomb 산란 단면적을 인코딩합니다:
    # e^4는 두 전하의 상호작용에서 나오고, ln_Lambda는
    # 90° 편향 반경과 Debye 길이 사이의 모든 충돌 매개변수 기여를 합산합니다.
    Gamma = n * e**4 * ln_Lambda / (4 * np.pi * epsilon_0**2 * m_e**2)

    # 1/v^3 의존성은 Coulomb 항력의 특징입니다: 빠른 입자는
    # 각 장 입자 근처에서 보내는 시간이 적어 유효 단면적이 줄어듭니다.
    # v > 3*v_th 임계값은 배경이 정지 산란 매체처럼 작용하는 영역
    # (시험 입자가 열 입자를 훨씬 앞질러 감)을 식별합니다.
    if v > 3 * v_th:
        nu_s = Gamma / v**3
    else:
        # Chandrasekhar의 phi 함수는 시험 입자보다 느린 배경 입자의 비율을 반영합니다
        # — 같은 방향으로 항력에 기여하는 입자만 포함됩니다.
        # 이 보정은 절반의 배경이 더 빠르게 움직여 항력을 부분적으로 상쇄하는
        # v ~ v_th 근처에서 필수적입니다.
        x = v / v_th
        phi = (np.erf(x) - 2*x*np.exp(-x**2)/np.sqrt(np.pi)) / (2*x**2)
        nu_s = Gamma * phi / v**3

    return nu_s

def dv_dt(v, t, n, T_e, E_field=0):
    """
    Time derivative of velocity including slowing down and electric field.

    Parameters:
    -----------
    v : float
        Current velocity (m/s)
    t : float
        Time (s)
    n : float
        Density (m^-3)
    T_e : float
        Temperature (eV)
    E_field : float
        Electric field (V/m)

    Returns:
    --------
    dvdt : float
        Rate of change of velocity
    """
    if v <= 0:
        return 0.0

    nu_s = slowing_down_frequency(v, n, T_e)

    # 항력(-nu_s * v)과 전기장 가속(eE/m_e) 사이의 경쟁이 도주(runaway)를 결정합니다:
    # E > E_Dreicer(드레이서 전기장)이면 높은 v의 입자에 대해 마찰이 전기장을 이길 수 없습니다.
    # 왜냐하면 nu_s * v ∝ 1/v^2 → v → ∞이 될수록 0으로 줄어들기 때문입니다.
    dvdt = -nu_s * v + e * E_field / m_e

    return dvdt

# Parameters for fusion plasma
n = 1e20  # m^-3
T_e = 10e3  # eV (10 keV)
v_th = np.sqrt(2 * T_e * e / m_e)

# Initial velocity of 3.5 MeV alpha particle
E_alpha = 3.5e6 * e  # Joules
m_alpha = 4 * 1.673e-27  # kg (alpha particle)
v_0 = np.sqrt(2 * E_alpha / m_alpha)

print(f"Thermal velocity: {v_th/1e6:.2f} Mm/s")
print(f"Initial alpha velocity: {v_0/1e6:.2f} Mm/s")
print(f"Velocity ratio v_0/v_th: {v_0/v_th:.1f}")

# Time array
t = np.linspace(0, 2, 1000)  # seconds

# Solve ODE
v_solution = odeint(dv_dt, v_0, t, args=(n, T_e))
v_solution = v_solution.flatten()

# Convert to energy
E_solution = 0.5 * m_alpha * v_solution**2 / e / 1e6  # MeV

# Find slowing-down time (when E drops to thermal energy)
E_thermal = 1.5 * T_e / 1e6  # MeV
idx_thermal = np.argmax(E_solution < E_thermal)
if idx_thermal > 0:
    tau_s = t[idx_thermal]
    print(f"Slowing-down time to thermal energy: {tau_s:.3f} s")

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t, v_solution/1e6, 'b-', linewidth=2, label='Test particle')
ax1.axhline(v_th/1e6, color='r', linestyle='--', label='$v_{th}$')
ax1.set_xlabel('Time (s)', fontsize=12)
ax1.set_ylabel('Velocity (Mm/s)', fontsize=12)
ax1.set_title('Alpha Particle Slowing Down', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.semilogy(t, E_solution, 'b-', linewidth=2, label='Test particle')
ax2.axhline(E_thermal, color='r', linestyle='--', label='Thermal energy')
ax2.set_xlabel('Time (s)', fontsize=12)
ax2.set_ylabel('Energy (MeV)', fontsize=12)
ax2.set_title('Energy vs Time', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('slowing_down.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.2 Braginskii 수송 계수

```python
def braginskii_coefficients(n, T, B, Z=1, A=1, ln_Lambda=15):
    """
    Compute Braginskii transport coefficients.

    Parameters:
    -----------
    n : float or array
        Density (m^-3)
    T : float or array
        Temperature (eV)
    B : float or array
        Magnetic field (T)
    Z : int
        Ion charge number
    A : float
        Ion mass number (in proton masses)
    ln_Lambda : float
        Coulomb logarithm

    Returns:
    --------
    dict with transport coefficients
    """
    # Convert to SI
    T_J = T * e
    m_i = A * 1.673e-27  # kg

    # Braginskii는 Fokker-Planck 충돌 연산자로부터 tau ~ T^(3/2)/n을 보였습니다;
    # T^(3/2) 스케일링은 더 빠른 입자일수록 Coulomb 단면적이 작아진다는 것을 반영합니다.
    tau_e = 12 * np.pi**(3/2) * epsilon_0**2 * m_e**(1/2) * T_J**(3/2) / \
            (n * e**4 * ln_Lambda * np.sqrt(2))
    # 이온 충돌 시간은 sqrt(m_i/m_e)만큼 더 깁니다. 무거운 이온은 더 느리게 움직여
    # 같은 온도에서 더 적게 상호작용하기 때문입니다.
    tau_i = np.sqrt(m_i / m_e) * tau_e

    # Cyclotron frequencies
    omega_ce = e * B / m_e
    omega_ci = Z * e * B / m_i

    # 평행 수송(parallel transport)은 random-walk 평균 자유 경로로 결정됩니다: κ ~ n*T*τ/m ~ v_th^2/ν.
    # Braginskii 수치 계수(3.16, 3.9)는 분포 함수의 비-Maxwellian 보정을 반영하는
    # 완전한 Fokker-Planck 풀이에서 나옵니다.
    kappa_par_e = 3.16 * n * T_J * tau_e / m_e
    kappa_par_i = 3.9 * n * T_J * tau_i / m_i
    eta_par = 0.96 * n * T_J * tau_i
    sigma_par = 1.96 * n * e**2 * tau_e / m_e

    # 수직 전도도(perpendicular conductivity)는 (ω_c τ)^2에 의해 억제됩니다. 입자가
    # 장 선을 가로질러 이동하려면 산란(사이클로트론 궤도 파괴)이 필요하고, 각 이동은
    # Larmor 반경 하나에 불과하여 κ_⊥ ~ nT / (m * ω_c^2 * τ)이 됩니다.
    kappa_perp_e = 4.66 * n * T_J / (m_e * omega_ce**2 * tau_e)
    kappa_perp_i = 2.0 * n * T_J / (m_i * omega_ci**2 * tau_i)
    eta_perp_0 = 0.73 * n * T_J / (omega_ci**2 * tau_i)

    # 비등방성 비율(anisotropy ratio) κ_∥/κ_⊥ ~ (ω_c τ)^2는 자기장이
    # 교차장 열 흐름을 얼마나 강하게 억제하는지를 나타냅니다.
    chi_e = kappa_par_e / kappa_perp_e
    chi_i = kappa_par_i / kappa_perp_i

    return {
        'tau_e': tau_e,
        'tau_i': tau_i,
        'kappa_par_e': kappa_par_e,
        'kappa_par_i': kappa_par_i,
        'kappa_perp_e': kappa_perp_e,
        'kappa_perp_i': kappa_perp_i,
        'eta_par': eta_par,
        'eta_perp_0': eta_perp_0,
        'sigma_par': sigma_par,
        'chi_e': chi_e,
        'chi_i': chi_i,
        'omega_ce_tau': omega_ce * tau_e,
        'omega_ci_tau': omega_ci * tau_i
    }

# ITER-like parameters
n = 1e20  # m^-3
T = np.logspace(2, 4, 100)  # eV, 100 eV to 10 keV
B = 5.3  # T

results = braginskii_coefficients(n, T, B, Z=1, A=2)  # Deuterium

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Thermal conductivity
ax1.loglog(T, results['kappa_par_e'], 'b-', linewidth=2, label='$\\kappa_{\\parallel e}$')
ax1.loglog(T, results['kappa_perp_e'], 'b--', linewidth=2, label='$\\kappa_{\\perp e}$')
ax1.loglog(T, results['kappa_par_i'], 'r-', linewidth=2, label='$\\kappa_{\\parallel i}$')
ax1.loglog(T, results['kappa_perp_i'], 'r--', linewidth=2, label='$\\kappa_{\\perp i}$')
ax1.set_xlabel('Temperature (eV)', fontsize=12)
ax1.set_ylabel('Thermal Conductivity (W/m/K)', fontsize=12)
ax1.set_title('Thermal Conductivity', fontsize=14)
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

# Anisotropy
ax2.loglog(T, results['chi_e'], 'b-', linewidth=2, label='Electrons')
ax2.loglog(T, results['chi_i'], 'r-', linewidth=2, label='Ions')
ax2.set_xlabel('Temperature (eV)', fontsize=12)
ax2.set_ylabel('$\\kappa_{\\parallel} / \\kappa_{\\perp}$', fontsize=12)
ax2.set_title('Thermal Conductivity Anisotropy', fontsize=14)
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

# Resistivity
eta_classical = 1 / results['sigma_par']
ax3.loglog(T, eta_classical, 'b-', linewidth=2)
ax3.set_xlabel('Temperature (eV)', fontsize=12)
ax3.set_ylabel('Resistivity ($\\Omega\\cdot$m)', fontsize=12)
ax3.set_title('Classical Resistivity ($\\eta \\propto T^{-3/2}$)', fontsize=14)
ax3.grid(True, which='both', alpha=0.3)

# Collision time
ax4.loglog(T, results['tau_e']*1e6, 'b-', linewidth=2, label='Electrons')
ax4.loglog(T, results['tau_i']*1e6, 'r-', linewidth=2, label='Ions')
ax4.set_xlabel('Temperature (eV)', fontsize=12)
ax4.set_ylabel('Collision Time ($\\mu$s)', fontsize=12)
ax4.set_title('Collision Time', fontsize=14)
ax4.legend()
ax4.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('braginskii_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()

# Print some typical values
T_typical = 1e4  # 10 keV
idx = np.argmin(np.abs(T - T_typical))
print(f"\nTypical values at T = {T_typical/1e3:.1f} keV, B = {B} T:")
print(f"  Electron collision time: {results['tau_e'][idx]*1e6:.2e} μs")
print(f"  ωce τ: {results['omega_ce_tau'][idx]:.2e}")
print(f"  κ_∥e / κ_⊥e: {results['chi_e'][idx]:.2e}")
print(f"  Classical resistivity: {eta_classical[idx]:.2e} Ω·m")
print(f"  Parallel thermal conductivity (e): {results['kappa_par_e'][idx]:.2e} W/m/K")
```

### 6.3 Neoclassical 수송 Regime

```python
def neoclassical_diffusion(r, R, B_0, n, T, Z=1, A=1):
    """
    Estimate neoclassical diffusion coefficient in different regimes.

    Parameters:
    -----------
    r : float
        Minor radius position (m)
    R : float
        Major radius (m)
    B_0 : float
        Magnetic field on axis (T)
    n : float
        Density (m^-3)
    T : float
        Temperature (eV)
    Z : int
        Ion charge
    A : float
        Ion mass number

    Returns:
    --------
    dict with diffusion coefficients and collisionality
    """
    epsilon = r / R  # Inverse aspect ratio
    m_i = A * 1.673e-27
    T_J = T * e

    # Larmor radius
    v_th = np.sqrt(2 * T_J / m_i)
    omega_ci = Z * e * B_0 / m_i
    rho_L = v_th / omega_ci

    # Collision frequency
    ln_Lambda = 15
    tau_i = 12 * np.pi**(3/2) * epsilon_0**2 * np.sqrt(m_i) * T_J**(3/2) / \
            (n * Z**4 * e**4 * ln_Lambda * np.sqrt(2))
    nu_ii = 1 / tau_i

    # Bounce 주파수(bounce frequency)는 포획된 입자가 바나나 궤도를 순회하는 속도입니다;
    # v_th / (π R √ε)는 포획된 입자가 poloidal 호 ~π R을
    # 평행 속도 v_th * √ε (작은 평행 성분만 살아남음)로 이동하는 데서 유래합니다.
    omega_b = v_th / (np.pi * R * np.sqrt(epsilon))

    # ν* < 1이면 입자가 산란 전에 많은 bounce를 완료한다는 뜻 — banana regime입니다.
    # ν*는 기하학적 효과(bouncing)와 충돌 중 어느 것이 입자 동역학을 지배하는지를 나타냅니다.
    nu_star = nu_ii / omega_b

    # 고전적 확산(classical diffusion)은 step 크기 ρ_L과 step 속도 ν_ii의 random walk입니다;
    # 이것은 토로이달 기하학 보정 이전의 Bohm 자유 경로 추정입니다.
    D_classical = rho_L**2 * nu_ii

    # banana regime에서 유효 step 크기는 banana 폭 ~ ρ_L / √ε으로 Larmor 반경보다 훨씬 크며,
    # D_nc ~ D_classical / ε^(3/2)를 줍니다.
    if nu_star < 0.1:  # Banana regime
        D_nc = D_classical / epsilon**(3/2)
        regime = "Banana"
    elif nu_star < 10:  # Plateau regime
        # plateau regime에서는 궤도 평균화(orbit-averaging)가 고전적 스케일링을 부분적으로 회복합니다;
        # ε^(-1) 향상은 포획된 입자의 비율 ~ √ε와 넓어진 step 크기의 조합에서 옵니다.
        D_nc = D_classical / epsilon
        regime = "Plateau"
    else:  # Pfirsch-Schlüter
        # 높은 충돌성의 PS regime에서 q^2 인자는 입자의 poloidal 순환이
        # 안전 인자(safety factor) q에 의해 유효 교차장 이동을 증폭시키는
        # 환류 전류를 생성하기 때문에 나타납니다.
        q = r * B_0 / (R * (B_0 * r / R))  # Approximate safety factor
        D_nc = D_classical * q**2
        regime = "Pfirsch-Schlüter"

    return {
        'D_classical': D_classical,
        'D_neoclassical': D_nc,
        'nu_star': nu_star,
        'regime': regime,
        'epsilon': epsilon,
        'rho_L': rho_L,
        'omega_b': omega_b,
        'nu_ii': nu_ii
    }

# ITER parameters
R = 6.2  # m
a = 2.0  # m
B_0 = 5.3  # T
n = 1e20  # m^-3

r_array = np.linspace(0.1, a, 50)
T_array = np.array([1e3, 5e3, 10e3, 20e3])  # eV

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

colors = ['blue', 'green', 'orange', 'red']

for i, T in enumerate(T_array):
    D_class = []
    D_nc = []
    nu_star_array = []

    for r in r_array:
        result = neoclassical_diffusion(r, R, B_0, n, T, Z=1, A=2)
        D_class.append(result['D_classical'])
        D_nc.append(result['D_neoclassical'])
        nu_star_array.append(result['nu_star'])

    ax1.semilogy(r_array, D_nc, color=colors[i], linewidth=2,
                 label=f'T = {T/1e3:.0f} keV')
    ax2.loglog(r_array, nu_star_array, color=colors[i], linewidth=2,
               label=f'T = {T/1e3:.0f} keV')
    ax3.semilogy(r_array, np.array(D_nc) / np.array(D_class),
                 color=colors[i], linewidth=2, label=f'T = {T/1e3:.0f} keV')

ax1.set_xlabel('Minor Radius (m)', fontsize=12)
ax1.set_ylabel('Neoclassical Diffusion ($m^2/s$)', fontsize=12)
ax1.set_title('Neoclassical Diffusion Coefficient', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.axhline(1, color='k', linestyle='--', alpha=0.5, label='Regime boundaries')
ax2.axhline(10, color='k', linestyle='--', alpha=0.5)
ax2.text(0.5, 0.05, 'Banana', transform=ax2.transAxes, fontsize=11)
ax2.text(0.5, 0.3, 'Plateau', transform=ax2.transAxes, fontsize=11)
ax2.text(0.5, 0.7, 'Pfirsch-Schlüter', transform=ax2.transAxes, fontsize=11)
ax2.set_xlabel('Minor Radius (m)', fontsize=12)
ax2.set_ylabel('Collisionality $\\nu_*$', fontsize=12)
ax2.set_title('Collisionality Parameter', fontsize=14)
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

ax3.set_xlabel('Minor Radius (m)', fontsize=12)
ax3.set_ylabel('$D_{nc} / D_{classical}$', fontsize=12)
ax3.set_title('Neoclassical Enhancement Factor', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# Bootstrap current coefficient
epsilon_array = r_array / R
L_p = 1.0  # Pressure scale length (m)
beta_p = 0.5  # Poloidal beta

# Bootstrap 전류는 바나나 궤도 드리프트가 비대칭이기 때문에 발생합니다: 외부 구간에서
# 포획된 입자들은 통과 입자가 보상해야 하는 순 반전류(net counter-current)를 운반하여,
# 포획된 입자 비율 √ε에 비례하는 자기 구동 전류를 만듭니다.
# 분모 (1 + ε^2)는 기하학이 포화되는 큰 ε에서 공식을 정규화합니다.
j_bs_coeff = epsilon_array**(1/2) / (1 + epsilon_array**2)

ax4.plot(r_array, j_bs_coeff, 'b-', linewidth=2)
ax4.set_xlabel('Minor Radius (m)', fontsize=12)
ax4.set_ylabel('Bootstrap Current Coefficient', fontsize=12)
ax4.set_title('Bootstrap Current Profile Shape', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('neoclassical_transport.png', dpi=150, bbox_inches='tight')
plt.show()

# Print regime information at r = a/2
r_mid = a / 2
for T in T_array:
    result = neoclassical_diffusion(r_mid, R, B_0, n, T, Z=1, A=2)
    print(f"\nAt T = {T/1e3:.0f} keV, r = {r_mid:.1f} m:")
    print(f"  Regime: {result['regime']}")
    print(f"  ν* = {result['nu_star']:.2f}")
    print(f"  D_nc / D_classical = {result['D_neoclassical']/result['D_classical']:.1f}")
```

## 요약

충돌 동역학은 무충돌 Vlasov 이론과 유체 기술 사이의 간격을 메웁니다:

**Boltzmann 충돌 연산자**:
- 임의의 단면적을 가진 이진 충돌 설명
- 입자, 운동량, 에너지 보존이 내장됨
- H-정리: 엔트로피 증가, Maxwellian 평형으로 구동

**Fokker-Planck 방정식**:
- Coulomb 충돌로 특수화: 많은 작은 각도 편향
- 동역학적 마찰 (항력) 및 속도 확산
- Rosenbluth potentials는 효율적인 계산 접근법 제공

**시험 입자 동역학**:
- 감속: 빠른 입자에 대해 $dv/dt \propto -v^{-3}$
- 임계 속도는 전자 대 이온 가열을 분리
- 전기장이 Dreicer 한계를 초과할 때 도주 전자

**Braginskii 수송**:
- 모멘트 접근법은 충돌 항이 있는 유체 방정식을 산출
- 평행 수송: 장 선을 따라 효율적
- 수직 수송: $(\omega_c \tau)^{-2} \sim 10^{-12}$ 인자로 억제
- 고전적 저항성: $\eta \propto T^{-3/2}$

**Neoclassical 효과**:
- 토로이달 기하학이 바나나 궤도에 포획된 입자를 생성
- 세 regime: Pfirsch-Schlüter (높은 $\nu_*$), plateau, banana (낮은 $\nu_*$)
- 고전적 수송 대비 향상: $q^2$ 또는 $\epsilon^{-3/2}$ 인자
- Bootstrap 전류: 압력 기울기로부터의 자가 생성 전류

**이상 수송**:
- 실제 플라즈마는 고전적 예측의 100배 수송을 보임
- 충돌이 아닌 난류에 의해 발생
- 경험적 스케일링 법칙이 핵융합 반응로 설계를 안내

이러한 충돌 이론은 다음에 필수적입니다:
- 핵융합 장치의 confinement 시간 예측
- 전류 구동 및 가열 메커니즘 이해
- 천체물리 플라즈마 분석 (태양 코로나, 강착 원반)
- 진단 및 제어 시스템 설계

## 연습 문제

### 문제 1: 충돌 주파수
수소 플라즈마가 $n = 10^{19}$ m$^{-3}$, $T_e = T_i = 1$ keV를 가집니다.

(a) $\ln\Lambda = 15$를 사용하여 전자-전자 충돌 주파수 $\nu_{ee}$를 계산하시오.

(b) 전자-이온 충돌 주파수 $\nu_{ei}$를 계산하시오.

(c) 전자와 이온 사이의 에너지 평형화 시간 $\tau_{eq}$를 계산하시오.

(d) 충돌 시간을 플라즈마 주기 $2\pi/\omega_{pe}$ 및 1미터를 가로지르는 전자 열 통과 시간과 비교하시오. 이것이 플라즈마의 충돌성에 대해 무엇을 말합니까?

### 문제 2: 알파 입자 감속
3.5 MeV 알파 입자 (D-T 핵융합으로부터)가 $n = 5 \times 10^{19}$ m$^{-3}$, $T_e = 15$ keV, $T_i = 12$ keV를 가진 플라즈마에서 생성됩니다.

(a) 전자 및 이온 가열을 분리하는 임계 에너지 $E_c$를 계산하시오.

(b) 알파 에너지의 어느 비율이 전자 대 이온으로 갑니까?

(c) 생성 에너지에서 열 에너지까지의 감속 시간을 추정하시오.

(d) 에너지 confinement 시간이 $\tau_E = 3$ s인 경우, 알파가 손실되기 전에 열화될까요? 반응로에서 자체 가열에 대한 함의는 무엇입니까?

### 문제 3: 고전적 대 Neoclassical 수송
$R = 3$ m, $a = 1$ m, $B_0 = 2$ T, $n = 5 \times 10^{19}$ m$^{-3}$, $T_i = 5$ keV를 가진 토카막을 고려하시오.

(a) $r = 0.5$ m에서 중수소 이온에 대한 충돌성 $\nu_*$를 계산하시오.

(b) neoclassical regime (banana, plateau, 또는 Pfirsch-Schlüter)를 식별하시오.

(c) 비율 $D_{nc}/D_{classical}$을 계산하시오.

(d) 이상 수송이 $D_{anomalous} = 1$ m$^2$/s를 주는 경우, 이것이 고전적 및 neoclassical 예측과 어떻게 비교됩니까? 이것이 지배적인 수송 메커니즘에 대해 무엇을 말합니까?

### 문제 4: 수직 대 평행 수송
플라즈마가 $n = 10^{20}$ m$^{-3}$, $T_e = 10$ keV, $B = 5$ T를 가집니다.

(a) 평행 열전도도 $\kappa_{\parallel,e}$를 계산하시오.

(b) 수직 열전도도 $\kappa_{\perp,e}$를 계산하시오.

(c) 장에 수직인 $dT/dx = 10^6$ K/m의 온도 기울기가 존재하는 경우, 교차장 열 플럭스는 무엇입니까?

(d) 장에 평행한 어떤 온도 기울기가 동일한 열 플럭스를 줄까요? 자기 confinement에서 온도 프로파일 제어에 대한 함의를 논의하시오.

### 문제 5: Bootstrap 전류
토카막이 $p(r) = p_0(1 - r^2/a^2)^2$, $p_0 = 5 \times 10^5$ Pa, $a = 2$ m의 압력 프로파일을 가집니다.

(a) $r = 1$ m에서 압력 기울기를 계산하시오.

(b) 근사 공식 $j_{bs} \sim \epsilon^{1/2}/(1+\epsilon^2) \cdot (n T/e B_p)(dp/dr)$를 사용하여 $r = 1$ m에서 bootstrap 전류 밀도를 추정하시오. $R = 6$ m, $B_p = 0.5$ T를 가정하시오.

(c) 전체 플라즈마 전류가 $I_p = 15$ MA인 경우, bootstrap 전류 비율을 추정하시오.

(d) 핵융합 반응로에 대해 높은 bootstrap 비율이 바람직한 이유는 무엇입니까?

---

**이전**: [8. Landau Damping](./08_Landau_Damping.md)
**다음**: [10. Electrostatic Waves](./10_Electrostatic_Waves.md)
