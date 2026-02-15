# 8. Landau 감쇠

## 학습 목표

- 선형화된 Vlasov-Poisson을 사용하여 따뜻한 플라즈마에서 정전기 파동에 대한 분산 관계 유도하기
- Landau 윤곽과 $v = \omega/k$에서 특이점을 처리하는 역할 이해하기
- Landau 감쇠율을 계산하고 플라즈마 매개변수에 대한 의존성 분석하기
- 공명에서 파동-입자 에너지 교환의 물리적 메커니즘 탐구하기
- 역 Landau 감쇠와 bump-on-tail 불안정성 연구하기
- Python을 사용하여 Landau 감쇠와 입자 포획 시뮬레이션하기

## 1. 따뜻한 플라즈마에서의 정전기 파동

### 1.1 선형화된 Vlasov-Poisson 시스템

1D, 자화되지 않은, 정전기 플라즈마를 고려합니다. 평형은:

$$
f = f_0(v), \quad \mathbf{E} = 0
$$

여기서 $f_0(v)$는 평형 분포입니다 (일반적으로 Maxwellian).

작은 섭동의 경우:

$$
f = f_0(v) + f_1(x, v, t), \quad E = E_1(x, t)
$$

$|f_1| \ll f_0$, $|E_1|$는 작습니다.

**선형화된 Vlasov 방정식**:

$$
\frac{\partial f_1}{\partial t} + v\frac{\partial f_1}{\partial x} + \frac{q}{m}E_1\frac{\partial f_0}{\partial v} = 0
$$

**선형화된 Poisson 방정식**:

$$
\frac{\partial E_1}{\partial x} = \frac{1}{\epsilon_0}\sum_s q_s \int f_1^{(s)} \, dv
$$

여기서 합은 종 $s$ (전자, 이온)에 대한 것입니다.

### 1.2 Fourier-Laplace 변환

평면파 해를 가정합니다:

$$
f_1(x, v, t) = \hat{f}_1(v) e^{i(kx - \omega t)}
$$

$$
E_1(x, t) = \hat{E}_1 e^{i(kx - \omega t)}
$$

여기서 $k$는 파수이고 $\omega$는 (복소) 주파수입니다.

선형화된 Vlasov 방정식에 대입하면:

$$
-i\omega\hat{f}_1 + ikv\hat{f}_1 + \frac{q}{m}\hat{E}_1\frac{df_0}{dv} = 0
$$

$\hat{f}_1$에 대해 풀면:

$$
\hat{f}_1(v) = \frac{iq}{m}\frac{\hat{E}_1}{kv - \omega}\frac{df_0}{dv}
$$

### 1.3 Poisson 방정식과 전하 밀도

Poisson으로부터:

$$
ik\hat{E}_1 = \frac{1}{\epsilon_0}\sum_s q_s \int \hat{f}_1^{(s)} dv
$$

$\hat{f}_1$을 대입하면:

$$
ik\hat{E}_1 = \frac{1}{\epsilon_0}\sum_s q_s \int \frac{iq_s}{m_s}\frac{\hat{E}_1}{kv - \omega}\frac{df_0^{(s)}}{dv} dv
$$

$\hat{E}_1$을 소거합니다 (자명하지 않은 해에 대해 $\hat{E}_1 \neq 0$ 가정):

$$
k = \frac{1}{\epsilon_0}\sum_s \frac{q_s^2}{m_s k} \int \frac{1}{v - \omega/k}\frac{df_0^{(s)}}{dv} dv
$$

재정리하면:

$$
1 = \frac{1}{\epsilon_0 k^2}\sum_s \frac{q_s^2}{m_s} \int \frac{1}{v - \omega/k}\frac{df_0^{(s)}}{dv} dv
$$

또는, **유전 함수** $\epsilon(k, \omega)$를 정의하면:

$$
\boxed{\epsilon(k, \omega) = 1 - \sum_s \frac{\omega_{ps}^2}{k^2} \int \frac{\partial f_0^{(s)}/\partial v}{v - \omega/k} dv = 0}
$$

여기서 $\omega_{ps}^2 = n_s q_s^2/(\epsilon_0 m_s)$는 종 $s$에 대한 플라즈마 주파수입니다.

**분산 관계**: $\epsilon(k, \omega) = 0$.

### 1.4 $v = \omega/k$에서의 극점

피적분함수는 $v = v_{\text{ph}} = \omega/k$ (위상 속도)에서 **극점**을 갖습니다. 이 특이점은 주의 깊은 처리가 필요합니다:

- 실수 $\omega$의 경우, 적분은 정의되지 않습니다 (주값 + 유수).
- 올바른 처방은 인과성 (초기 조건을 가진 Laplace 변환)으로부터 나옵니다.

## 2. Landau 윤곽과 해석적 연속

### 2.1 인과성과 Laplace 변환

적절하게, 우리는 초기에 $\text{Im}(\omega) > 0$인 시간에 대한 Laplace 변환을 사용해야 합니다 (지수 감쇠가 수렴을 보장). 그러면 적분은 잘 정의됩니다:

$$
\int \frac{1}{v - \omega/k} dv
$$

$\text{Im}(\omega/k) < 0$일 때 (극점은 속도 공간에서 실축 **아래**에 있음).

$\omega(k)$에 대해 푼 후, 우리는 물리적 해로 해석적 연속을 하며, 이는 $\text{Im}(\omega) < 0$ (감쇠) 또는 $\text{Im}(\omega) > 0$ (성장)을 가질 수 있습니다.

### 2.2 Landau 처방

결과는 **Landau 윤곽**입니다: 속도 공간에서 적분 경로는 $v = \omega/k$의 극점 **아래**로 갑니다.

```
    Complex v-plane

       Im(v)
         ↑
         |
    ─────┼─────────────→ Re(v)
         |        × pole at v = ω/k
         |      (contour goes below)
```

**Plemelj 공식**을 사용하면:

$$
\frac{1}{v - v_0 - i0^+} = \mathcal{P}\frac{1}{v - v_0} + i\pi\delta(v - v_0)
$$

여기서 $\mathcal{P}$는 주값을 나타내고 $\delta$는 Dirac 델타 함수입니다.

따라서:

$$
\int \frac{\partial f_0/\partial v}{v - \omega/k} dv = \mathcal{P}\int \frac{\partial f_0/\partial v}{v - \omega/k} dv + i\pi\frac{\partial f_0}{\partial v}\bigg|_{v = \omega/k}
$$

### 2.3 Landau 처방을 가진 유전 함수

유전 함수는:

$$
\epsilon(k, \omega) = 1 - \sum_s \frac{\omega_{ps}^2}{k^2}\left[\mathcal{P}\int \frac{\partial f_0^{(s)}/\partial v}{v - \omega/k} dv + i\pi\frac{\partial f_0^{(s)}}{\partial v}\bigg|_{v = \omega/k}\right]
$$

$\epsilon = 0$을 설정하면 분산 관계를 얻습니다. $\epsilon$가 복소수이므로, $\omega$는 일반적으로 복소수입니다:

$$
\omega = \omega_r + i\gamma
$$

여기서:
- $\omega_r$: 실수 부분 (진동 주파수)
- $\gamma$: 허수 부분 ($\gamma > 0$이면 성장율, $\gamma < 0$이면 감쇠율)

## 3. 전자 플라즈마 파동의 Landau 감쇠

### 3.1 전자 플라즈마 파동 (Langmuir 파동)

움직이는 전자와 움직이지 않는 이온 ($m_i \to \infty$)을 가진 플라즈마를 고려합니다. 평형 전자 분포는 Maxwellian입니다:

$$
f_0(v) = n_0\sqrt{\frac{m_e}{2\pi k_B T_e}}\exp\left(-\frac{m_e v^2}{2k_BT_e}\right)
$$

도함수는:

$$
\frac{df_0}{dv} = -\frac{m_e v}{k_BT_e}f_0(v)
$$

### 3.2 분산 관계: 실수 부분

$|\gamma| \ll \omega_r$의 경우, 주값 적분에서 $\omega/k \approx \omega_r/k$로 근사할 수 있습니다. $\epsilon = 0$의 실수 부분은:

$$
1 - \frac{\omega_{pe}^2}{k^2}\mathcal{P}\int \frac{df_0/dv}{v - \omega_r/k} dv = 0
$$

부분 적분을 사용하면:

$$
\mathcal{P}\int \frac{df_0/dv}{v - \omega_r/k} dv = -\int f_0(v) \frac{\partial}{\partial v}\left[\mathcal{P}\frac{1}{v - \omega_r/k}\right] dv
$$

Maxwellian 및 $k\lambda_D \ll 1$ (여기서 $\lambda_D = \sqrt{\epsilon_0 k_B T_e/(n_0 e^2)}$는 Debye 길이)에 대해, 결과는:

$$
\boxed{\omega_r^2 \approx \omega_{pe}^2 + 3k^2v_{th,e}^2}
$$

여기서 $v_{th,e} = \sqrt{k_BT_e/m_e}$는 전자 열속도입니다.

이것이 전자 플라즈마 파동 (Langmuir 파동)에 대한 **Bohm-Gross 분산 관계**입니다.

### 3.3 허수 부분: 감쇠율

$\epsilon = 0$의 허수 부분은 감쇠율을 제공합니다. 작은 감쇠 ($|\gamma| \ll \omega_r$)의 경우:

$$
\gamma \approx -\frac{\pi\omega_{pe}^2}{2k^2}\frac{df_0}{dv}\bigg|_{v = \omega_r/k}
$$

Maxwellian의 경우:

$$
\frac{df_0}{dv}\bigg|_{v = \omega_r/k} = -\frac{m_e\omega_r}{k k_B T_e}f_0(\omega_r/k) = -\frac{m_e\omega_r}{k k_B T_e}n_0\sqrt{\frac{m_e}{2\pi k_B T_e}}\exp\left(-\frac{m_e\omega_r^2}{2k^2k_BT_e}\right)
$$

단순화하면:

$$
\gamma = \frac{\pi\omega_{pe}^2}{2k^2} \cdot \frac{m_e\omega_r}{k k_B T_e}n_0\sqrt{\frac{m_e}{2\pi k_B T_e}}\exp\left(-\frac{\omega_r^2}{2k^2v_{th,e}^2}\right)
$$

$\omega_r^2 \approx \omega_{pe}^2(1 + 3k^2\lambda_D^2)$ 및 $k\lambda_D \ll 1$을 사용하면:

$$
\frac{\omega_r^2}{2k^2v_{th,e}^2} \approx \frac{\omega_{pe}^2}{2k^2v_{th,e}^2} = \frac{1}{2k^2\lambda_D^2}
$$

따라서:

$$
\boxed{\gamma \approx -\sqrt{\frac{\pi}{8}}\frac{\omega_{pe}}{(k\lambda_D)^3}\exp\left(-\frac{1}{2k^2\lambda_D^2}\right)}
$$

**주요 특징**:
- $\gamma < 0$: 감쇠 (성장 아님)
- $|\gamma| \propto \exp(-1/(2k^2\lambda_D^2))$: $k\lambda_D \ll 1$에 대해 지수적으로 약함
- $|\gamma|/\omega_r \propto (k\lambda_D)^{-3}\exp(-1/(2k^2\lambda_D^2))$: 전형적인 플라즈마에 대해 매우 작음

### 3.4 유효 조건

Landau 감쇠는 다음 경우 유의미합니다:

$$
k\lambda_D \sim 1
$$

$k\lambda_D \ll 1$의 경우, 감쇠는 지수적으로 약합니다. $k\lambda_D \gg 1$의 경우, 파동은 심하게 감쇠됩니다 (과감쇠).

### 3.5 수치 예제

**예제**: $n_e = 10^{18}$ m$^{-3}$, $T_e = 10$ eV인 실험실 플라즈마.

계산:
- $\omega_{pe} = \sqrt{n_e e^2/(\epsilon_0 m_e)} = 5.64\times 10^{10}$ rad/s
- $\lambda_D = \sqrt{\epsilon_0 k_B T_e/(n_e e^2)} = 2.35\times 10^{-5}$ m
- $k = 10^5$ m$^{-1}$에 대해: $k\lambda_D = 2.35$

그러면:
- $\omega_r \approx \omega_{pe}\sqrt{1 + 3(k\lambda_D)^2} \approx 1.23\omega_{pe} = 6.94\times 10^{10}$ rad/s
- $\gamma/\omega_{pe} \approx -0.09\exp(-0.09) \approx -0.082$
- $|\gamma|/\omega_r \approx 0.067$

파동은 약 15번의 진동에서 감쇠됩니다.

## 4. 물리적 메커니즘: 파동-입자 공명

### 4.1 공명 입자

Landau 감쇠는 공명 입자로부터 발생합니다: 파동의 위상 속도 $v \approx v_{\text{ph}} = \omega/k$로 움직이는 입자들.

이 입자들은 파동을 "서핑"하여, 파동과 에너지를 교환합니다.

```
    Wave electric field

         E(x,t) = E0 sin(kx - ωt)

    Particle at x = x0, v = v_ph:
    - Sees stationary potential (in wave frame)
    - Can gain or lose energy

    Phase space:
         v
         ↑
         |    •   slow particles (v < v_ph)
         |   ••
         |  •••  ← bulk of distribution
         | •••
         |•••──────→ x
         |  ← v_ph (resonance)
         |
       ••|       fast particles (v > v_ph)
        •|

    For Maxwellian: more slow particles than fast
    → Net energy transfer: wave → particles → damping
```

### 4.2 에너지 교환

파동 프레임 ($v_{\text{ph}}$로 움직임)에서, 전기장은 정적입니다. 입자는 다음을 봅니다:

$$
E(x - v_{\text{ph}}t) = E_0\sin(kx - kv_{\text{ph}}t) = E_0\sin(kx - \omega t)
$$

파동 프레임에서 입자 속도가 $v' = v - v_{\text{ph}}$이면:

- $v' > 0$ (입자가 파동보다 빠름): 입자가 포텐셜 언덕을 올라감, 에너지 손실
- $v' < 0$ (입자가 파동보다 느림): 입자가 미끄러져 내려감, 에너지 획득

순 에너지 전달은 $v = v_{\text{ph}}$에서 **분포 함수 기울기**에 달려 있습니다:

$$
\frac{df_0}{dv}\bigg|_{v = v_{\text{ph}}}
$$

Maxwellian의 경우 ($v = 0$에서 단조 감소), 모든 $v > 0$에서 $df_0/dv < 0$입니다. 공명에서 **느린 입자가 빠른 입자보다 많습니다**.

결과: 에너지를 얻는 입자 (느림)가 에너지를 잃는 입자 (빠름)보다 많음 → 파동에서 입자로 순 에너지 전달 → **감쇠**.

### 4.3 서핑 유추

해양 파도의 서퍼를 생각해보세요:

- **느린 서퍼** (파도 마루 뒤): 파도에 의해 가속됨, 에너지 획득
- **빠른 서퍼** (파도 마루 앞): 감속됨, 에너지 손실
- 느린 서퍼가 더 많으면, 파도에서 서퍼로 순 에너지 전달 → 파도 감쇠

### 4.4 감쇠 대 성장

공명에서 $df_0/dv$의 부호가 감쇠 또는 성장을 결정합니다:

$$
\gamma \propto -\frac{df_0}{dv}\bigg|_{v = v_{\text{ph}}}
$$

- $df_0/dv < 0$ (감소하는 분포): $\gamma < 0$ → **감쇠**
- $df_0/dv > 0$ (증가하는 분포): $\gamma > 0$ → **성장** (역 Landau 감쇠)

## 5. 역 Landau 감쇠: Bump-on-Tail 불안정성

### 5.1 비단조 분포

$f_0(v)$가 $df_0/dv > 0$ (양의 기울기)인 영역을 가지면, 그 영역에 $v_{\text{ph}}$를 가진 파동은 성장할 것입니다.

고전적인 예는 **bump-on-tail** 분포입니다:

$$
f_0(v) = f_{\text{core}}(v) + f_{\text{beam}}(v)
$$

여기서:
- Core: $v = 0$을 중심으로 한 Maxwellian
- Beam: $v = v_b > 0$을 중심으로 한 Maxwellian (드리프트 빔)

```
    f(v)
      ↑
      |   Core
      |  /‾‾\___
      | /       \___   Beam
      |/            \_/‾\____
     ─┴──────────────────────→ v
                      v_b

    Between core and beam: df/dv > 0 → unstable
```

### 5.2 성장율

희박한 빔 ($n_b \ll n_c$)의 경우, 성장율은:

$$
\gamma \approx \frac{\pi\omega_{pe}^2}{2k^2}\frac{df_0}{dv}\bigg|_{v = \omega/k}
$$

양의 기울기 영역에서:

$$
\gamma > 0 \quad \Rightarrow \quad \text{성장 (불안정성)}
$$

최대 성장은 $v_{\text{ph}}$가 가장 가파른 양의 기울기와 일치할 때 발생합니다.

### 5.3 준선형 완화

파동이 성장함에 따라, 공명 근처의 입자가 포획되고 (다음 섹션 참조) 분포가 평탄화됩니다:

```
    Initial:  f(v) with bump
              /‾\  ← bump
             /   \_____

    After relaxation:  flattened
              /‾‾‾‾\____
```

공명에서 $df/dv$의 평탄화는 성장율을 감소시킵니다. 결국, 시스템은 공명에서 $df/dv \approx 0$인 **준선형 평탄역**에 도달하고, 성장이 멈춥니다.

이것이 **준선형 완화**입니다: 파동 성장 → 입자 포획 → 분포 평탄화 → 포화.

### 5.4 응용

역 Landau 감쇠 (bump-on-tail 불안정성)는 다음에서 발생합니다:
- 플라즈마의 **전자 빔** (실험실, 우주)
- 태양풍의 **이온 빔**
- **전류 구동 불안정성** (예: 핵융합 플라즈마의 전자 전류)

## 6. 비선형 Landau 감쇠와 입자 포획

### 6.1 입자 포획

파동 진폭이 클 때, $v \approx v_{\text{ph}}$ 근처의 입자가 파동 포텐셜에 **포획**될 수 있습니다.

파동 프레임에서, 포텐셜은:

$$
\Phi(x) = \frac{E_0}{k}\cos(kx)
$$

파동 프레임에서 작은 속도 $v' = v - v_{\text{ph}}$를 가진 입자는 포텐셜 우물을 보고 **바운스 진동**을 실행할 수 있습니다.

바운스 주파수는:

$$
\omega_b = \sqrt{\frac{ekE_0}{m}} = \sqrt{\frac{eE_0 k}{m}}
$$

### 6.2 위상 공간 소용돌이

포획된 입자는 **위상 공간 소용돌이** (고양이 눈 구조)를 형성합니다:

```
    Phase space (x, v)

         v
         ↑
         |        •••
         |      ••   ••   ← separatrix
         |     •  ⊗  •      (trapped particles)
         |      ••   ••
         |        •••
         |──────────────→ x
              λ = 2π/k

    ⊗ = wave fixed point (v = v_ph)
    Particles inside separatrix are trapped
    Particles outside are passing
```

분리면 (포획과 통과 사이의 경계)은 에너지에 해당합니다:

$$
W_{\text{sep}} = e\Phi_0 = \frac{eE_0}{k}
$$

포획된 영역의 속도 폭은:

$$
\Delta v_{\text{trap}} \sim \frac{\omega_b}{k} = \frac{1}{k}\sqrt{\frac{ekE_0}{m}}
$$

### 6.3 BGK 모드와 O'Neil 정리

**BGK (Bernstein-Greene-Kruskal) 모드**는 포획된 입자를 가진 Vlasov-Poisson의 정확한 비선형 해입니다. 이들은 정상 상태 정전기 구조 (전자 홀, 이온 홀)를 나타냅니다.

**O'Neil 정리**: Landau 감쇠는 선형 모드 (다른 위상 속도를 가진 고유 모드)의 위상 혼합으로 볼 수 있습니다. 전기장은 감쇠하지만 섭동된 분포 $f_1$은 지속됩니다 (입자 사이에 재분배됨).

이것은 충돌 감쇠와 근본적으로 다릅니다:
- **충돌**: 에너지가 열로 소산됨 (비가역적)
- **Landau**: 에너지가 입자로 전달됨, 분포에 저장됨 (비선형성 또는 충돌이 작용할 때까지 가역적)

### 6.4 재발과 에코

Landau 감쇠가 가역적이므로, 시스템은 **재발**을 나타낼 수 있습니다: 전기장이 여러 플라즈마 주기 후에 다시 나타납니다. 실제로, 재발은 다음에 의해 파괴됩니다:
- 충돌
- 비선형성 (포획)
- 유한 기하학

**플라즈마 에코**: 두 개의 섭동이 다른 시간에 적용되면, "에코" 신호가 나중 시간에 나타나며, Landau 감쇠의 가역적 특성을 보여줍니다.

## 7. 이온 음향 파동과 Landau 감쇠

### 7.1 이온 음향 파동

이온 음향 파동은 다음을 가진 저주파 정전기 파동입니다:
- 전자가 복원력 제공 (압력을 통해)
- 이온이 관성 제공
- 분산: $\omega/k \approx c_s = \sqrt{k_B T_e/m_i}$ (이온 음속)

분산 관계 (이온과 전자로부터의 Landau 감쇠 포함)는:

$$
\epsilon(k, \omega) = 1 + \frac{1}{k^2\lambda_{De}^2} - \frac{\omega_{pi}^2}{k^2}\int \frac{df_i/dv}{v - \omega/k} dv = 0
$$

여기서 전자 기여는 $1/k^2\lambda_{De}^2$로 근사됩니다 ($\omega/k \ll v_{th,e}$ 가정).

### 7.2 약한 감쇠 조건

이온 Landau 감쇠율은:

$$
\gamma_i \propto -\frac{df_i}{dv}\bigg|_{v = c_s}
$$

약한 감쇠의 경우, $c_s \gg v_{th,i}$ (위상 속도가 이온 열속도보다 훨씬 빠름)가 필요합니다:

$$
\sqrt{\frac{k_B T_e}{m_i}} \gg \sqrt{\frac{k_B T_i}{m_i}} \quad \Rightarrow \quad T_e \gg T_i
$$

따라서, 이온 음향 파동은 **전자가 이온보다 훨씬 뜨거울 때** 낮은 감쇠로 전파됩니다.

$T_e \sim T_i$의 경우, 이온 Landau 감쇠가 강하고, 파동은 심하게 감쇠됩니다.

### 7.3 응용

이온 음향 파동은 다음에서 중요합니다:
- **레이저-플라즈마 상호작용** (유도 Brillouin 산란)
- **관성 가둠 핵융합** (에너지 수송)
- **우주 플라즈마** (태양풍 난류)

## 8. Python 구현

### 8.1 플라즈마 분산 함수 Z(ζ)

플라즈마 분산 함수는 다음과 같이 정의됩니다:

$$
Z(\zeta) = \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty} \frac{e^{-t^2}}{t - \zeta} dt
$$

Landau 윤곽 (극점이 실축 아래)을 가집니다. 이 함수는 플라즈마 운동학 이론에서 자주 나타납니다.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz

def plasma_dispersion_function(zeta):
    """
    Plasma dispersion function Z(zeta)
    Uses Faddeeva function (wofz in scipy)

    Z(zeta) = i*sqrt(pi) * w(zeta)
    where w(z) is the Faddeeva function
    """
    return 1j * np.sqrt(np.pi) * wofz(zeta)

# Plot Z(ζ) for real ζ
zeta_real = np.linspace(-5, 5, 1000)
Z_real = plasma_dispersion_function(zeta_real)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(zeta_real, Z_real.real, 'b-', linewidth=2, label='Re[Z(ζ)]')
ax1.plot(zeta_real, Z_real.imag, 'r-', linewidth=2, label='Im[Z(ζ)]')
ax1.set_xlabel('ζ', fontsize=14)
ax1.set_ylabel('Z(ζ)', fontsize=14)
ax1.set_title('Plasma Dispersion Function', fontsize=16, fontweight='bold')
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linewidth=0.5)
ax1.axvline(x=0, color='k', linewidth=0.5)

# For small ζ: Z(ζ) ≈ i*sqrt(pi)*exp(-ζ^2) - 2ζ (asymptotic)
zeta_small = np.linspace(-2, 2, 100)
Z_approx = 1j*np.sqrt(np.pi)*np.exp(-zeta_small**2) - 2*zeta_small

ax2.plot(zeta_small, np.abs(Z_real[400:600]), 'b-', linewidth=2, label='|Z(ζ)| exact')
ax2.plot(zeta_small, np.abs(Z_approx), 'r--', linewidth=2, label='|Z(ζ)| approx')
ax2.set_xlabel('ζ', fontsize=14)
ax2.set_ylabel('|Z(ζ)|', fontsize=14)
ax2.set_title('Asymptotic Approximation', fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plasma_dispersion_function.png', dpi=150)
print("Saved: plasma_dispersion_function.png")
```

### 8.2 Landau 감쇠율 대 kλ_D

```python
# Constants
e = 1.6e-19
m_e = 9.11e-31
epsilon_0 = 8.85e-12
k_B = 1.38e-23

# Plasma parameters
n_e = 1e18  # m^-3
T_e_eV = 10  # eV
T_e = T_e_eV * e / k_B  # K

# Derived quantities
omega_pe = np.sqrt(n_e * e**2 / (epsilon_0 * m_e))
v_th = np.sqrt(k_B * T_e / m_e)
lambda_D = np.sqrt(epsilon_0 * k_B * T_e / (n_e * e**2))

print(f"Plasma parameters:")
print(f"  n_e = {n_e:.2e} m^-3")
print(f"  T_e = {T_e_eV} eV")
print(f"  ω_pe = {omega_pe:.2e} rad/s")
print(f"  v_th = {v_th:.2e} m/s")
print(f"  λ_D = {lambda_D:.2e} m")

# Range of k*lambda_D
k_lambda_D = np.linspace(0.1, 3, 100)
k_array = k_lambda_D / lambda_D

# Dispersion relation (Bohm-Gross)
omega_r = omega_pe * np.sqrt(1 + 3 * k_lambda_D**2)

# Landau damping rate
gamma = -np.sqrt(np.pi / 8) * (omega_pe / k_lambda_D**3) * np.exp(-1 / (2 * k_lambda_D**2))

# Damping decrement
damping_decrement = -gamma / omega_r

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Dispersion relation
ax = axes[0, 0]
ax.plot(k_lambda_D, omega_r / omega_pe, 'b-', linewidth=2)
ax.axhline(y=1, color='r', linestyle='--', linewidth=1, label='ω_pe (cold plasma)')
ax.set_xlabel('kλ_D', fontsize=12)
ax.set_ylabel('ω_r / ω_pe', fontsize=12)
ax.set_title('Dispersion Relation (Bohm-Gross)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Damping rate
ax = axes[0, 1]
ax.plot(k_lambda_D, np.abs(gamma) / omega_pe, 'r-', linewidth=2)
ax.set_xlabel('kλ_D', fontsize=12)
ax.set_ylabel('|γ| / ω_pe', fontsize=12)
ax.set_title('Landau Damping Rate', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

# Damping decrement
ax = axes[1, 0]
ax.plot(k_lambda_D, damping_decrement, 'g-', linewidth=2)
ax.set_xlabel('kλ_D', fontsize=12)
ax.set_ylabel('|γ| / ω_r', fontsize=12)
ax.set_title('Damping Decrement (per radian)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

# Number of oscillations before e-fold decay
ax = axes[1, 1]
N_osc = omega_r / (2 * np.pi * np.abs(gamma))
ax.plot(k_lambda_D, N_osc, 'm-', linewidth=2)
ax.set_xlabel('kλ_D', fontsize=12)
ax.set_ylabel('N (oscillations)', fontsize=12)
ax.set_title('Number of Oscillations Before e-Fold Decay', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('landau_damping_rate.png', dpi=150)
print("Saved: landau_damping_rate.png")

# Print specific values
print(f"\nLandau damping at kλ_D = 0.3:")
idx = np.argmin(np.abs(k_lambda_D - 0.3))
print(f"  ω_r/ω_pe = {omega_r[idx]/omega_pe:.3f}")
print(f"  |γ|/ω_pe = {np.abs(gamma[idx])/omega_pe:.3e}")
print(f"  |γ|/ω_r = {damping_decrement[idx]:.3e}")
print(f"  N_osc = {N_osc[idx]:.1f}")
```

### 8.3 Landau 감쇠를 가진 Vlasov-Poisson 시뮬레이션

```python
class VlasovPoisson1D:
    """
    1D Vlasov-Poisson solver for Landau damping
    """
    def __init__(self, Nx, Nv, Lx, v_max, n0, T_eV, m, q):
        self.Nx = Nx
        self.Nv = Nv
        self.Lx = Lx
        self.v_max = v_max

        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.v = np.linspace(-v_max, v_max, Nv)
        self.dx = Lx / Nx
        self.dv = 2 * v_max / Nv

        self.n0 = n0
        self.T = T_eV * e / k_B
        self.m = m
        self.q = q

        # Initialize distribution function
        self.f = self._initialize_maxwellian()

    def _initialize_maxwellian(self):
        """Maxwellian distribution"""
        v_th = np.sqrt(k_B * self.T / self.m)
        f = np.zeros((self.Nx, self.Nv))
        for i in range(self.Nx):
            f[i, :] = self.n0 * (self.m / (2 * np.pi * k_B * self.T))**0.5 * \
                     np.exp(-self.m * self.v**2 / (2 * k_B * self.T))
        return f

    def add_perturbation(self, k_mode, amplitude):
        """Add sinusoidal density perturbation"""
        for i in range(self.Nx):
            pert = 1 + amplitude * np.cos(k_mode * self.x[i])
            self.f[i, :] *= pert

    def compute_density(self):
        """Compute density from distribution function"""
        return np.trapz(self.f, self.v, axis=1)

    def compute_electric_field(self):
        """Solve Poisson equation for E-field (periodic BC)"""
        n = self.compute_density()
        rho = self.q * (n - self.n0)  # charge density (background neutrality)

        # Fourier transform
        rho_k = np.fft.fft(rho)
        k_modes = 2 * np.pi * np.fft.fftfreq(self.Nx, self.dx)

        # Poisson: -ε₀ d²φ/dx² = ρ → φ_k = -rho_k / (ε₀ k²)
        phi_k = np.zeros_like(rho_k, dtype=complex)
        phi_k[1:] = -rho_k[1:] / (epsilon_0 * k_modes[1:]**2)
        phi_k[0] = 0  # Set DC component to zero (neutrality)

        # E = -dφ/dx → E_k = i*k*φ_k
        E_k = 1j * k_modes * phi_k

        # Inverse FFT
        E = np.fft.ifft(E_k).real

        return E

    def step(self, dt):
        """Operator splitting: advection in x, then in v"""
        # Step 1: Advection in x (∂f/∂t + v ∂f/∂x = 0)
        f_new = np.zeros_like(self.f)
        for j in range(self.Nv):
            # Upwind scheme
            if self.v[j] > 0:
                for i in range(self.Nx):
                    i_up = (i - 1) % self.Nx
                    f_new[i, j] = self.f[i, j] - self.v[j] * dt / self.dx * \
                                 (self.f[i, j] - self.f[i_up, j])
            else:
                for i in range(self.Nx):
                    i_up = (i + 1) % self.Nx
                    f_new[i, j] = self.f[i, j] - self.v[j] * dt / self.dx * \
                                 (self.f[i_up, j] - self.f[i, j])
        self.f = f_new.copy()

        # Step 2: Acceleration in v (∂f/∂t + a ∂f/∂v = 0)
        E = self.compute_electric_field()
        f_new = np.zeros_like(self.f)
        for i in range(self.Nx):
            a = self.q * E[i] / self.m  # acceleration
            for j in range(self.Nv):
                if a > 0:
                    j_up = max(j - 1, 0)
                    f_new[i, j] = self.f[i, j] - a * dt / self.dv * \
                                 (self.f[i, j] - self.f[i, j_up])
                else:
                    j_up = min(j + 1, self.Nv - 1)
                    f_new[i, j] = self.f[i, j] - a * dt / self.dv * \
                                 (self.f[i, j_up] - self.f[i, j])
        self.f = f_new.copy()

    def run(self, dt, num_steps, save_interval=10):
        """Run simulation"""
        times = []
        E_history = []

        for n in range(num_steps):
            if n % save_interval == 0:
                E = self.compute_electric_field()
                E_max = np.max(np.abs(E))
                E_history.append(E_max)
                times.append(n * dt)
                if n % (num_steps // 10) == 0:
                    print(f"Step {n}/{num_steps}, t = {n*dt:.3e} s, E_max = {E_max:.3e} V/m")

            self.step(dt)

        return np.array(times), np.array(E_history)

# Simulation parameters
Nx = 64
Nv = 128
n0 = 1e18  # m^-3
T_eV = 10  # eV
m = m_e
q = -e

# Domain
lambda_D = np.sqrt(epsilon_0 * k_B * (T_eV * e / k_B) / (n0 * e**2))
k_mode = 0.3 / lambda_D  # kλ_D = 0.3
Lx = 2 * np.pi / k_mode
v_max = 5 * np.sqrt(k_B * (T_eV * e / k_B) / m)

# Initialize solver
print("\n=== Landau Damping Simulation ===")
print(f"Nx = {Nx}, Nv = {Nv}")
print(f"Lx = {Lx:.3e} m, v_max = {v_max:.3e} m/s")
print(f"kλ_D = 0.3")

solver = VlasovPoisson1D(Nx, Nv, Lx, v_max, n0, T_eV, m, q)

# Add perturbation
amplitude = 0.01
solver.add_perturbation(k_mode, amplitude)

# Run simulation
dt = 1e-11  # s (must satisfy CFL condition)
num_steps = 2000
save_interval = 5

times, E_max_history = solver.run(dt, num_steps, save_interval)

# Theoretical damping
omega_pe = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
k_lambda_D_val = 0.3
omega_r = omega_pe * np.sqrt(1 + 3 * k_lambda_D_val**2)
gamma_theory = -np.sqrt(np.pi / 8) * (omega_pe / k_lambda_D_val**3) * \
               np.exp(-1 / (2 * k_lambda_D_val**2))

E_theory = E_max_history[0] * np.exp(gamma_theory * times)

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear plot
ax1.plot(times * omega_pe, E_max_history, 'b-', linewidth=2, label='Simulation')
ax1.plot(times * omega_pe, E_theory, 'r--', linewidth=2, label='Theory')
ax1.set_xlabel('ω_pe t', fontsize=12)
ax1.set_ylabel('E_max (V/m)', fontsize=12)
ax1.set_title('Landau Damping of Electric Field', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Log plot
ax2.semilogy(times * omega_pe, E_max_history, 'b-', linewidth=2, label='Simulation')
ax2.semilogy(times * omega_pe, E_theory, 'r--', linewidth=2, label='Theory')
ax2.set_xlabel('ω_pe t', fontsize=12)
ax2.set_ylabel('E_max (V/m)', fontsize=12)
ax2.set_title('Landau Damping (Log Scale)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('landau_damping_simulation.png', dpi=150)
print("\nSaved: landau_damping_simulation.png")

# Fit damping rate
log_E = np.log(E_max_history)
fit = np.polyfit(times, log_E, 1)
gamma_fit = fit[0]

print(f"\nTheoretical damping rate: γ = {gamma_theory:.3e} rad/s")
print(f"Fitted damping rate: γ = {gamma_fit:.3e} rad/s")
print(f"Relative error: {abs(gamma_fit - gamma_theory)/abs(gamma_theory)*100:.1f}%")
```

### 8.4 입자 포획 시각화

```python
def particle_in_wave(E0, k, m, q, v_ph, num_particles=100, duration=1e-7, dt=1e-10):
    """
    Simulate particles in a static wave (wave frame)
    """
    # Particle initial conditions
    np.random.seed(42)
    x0 = np.random.uniform(0, 2*np.pi/k, num_particles)
    v0 = np.random.normal(v_ph, 1e4, num_particles)  # spread around v_ph

    # Storage
    num_steps = int(duration / dt)
    x_traj = np.zeros((num_particles, num_steps))
    v_traj = np.zeros((num_particles, num_steps))
    x_traj[:, 0] = x0
    v_traj[:, 0] = v0

    # Integrate equations of motion
    for n in range(1, num_steps):
        x = x_traj[:, n-1]
        v = v_traj[:, n-1]

        # Electric field (wave frame: static)
        E = E0 * np.sin(k * x)
        a = q * E / m

        # Velocity Verlet
        v_half = v + 0.5 * a * dt
        x_new = x + v_half * dt
        x_new = x_new % (2 * np.pi / k)  # periodic

        E_new = E0 * np.sin(k * x_new)
        a_new = q * E_new / m
        v_new = v_half + 0.5 * a_new * dt

        x_traj[:, n] = x_new
        v_traj[:, n] = v_new

    return x_traj, v_traj

# Parameters
E0 = 1e3  # V/m (large amplitude)
k = 1e5   # m^-1
v_ph = 1e5  # m/s
omega_b = np.sqrt(e * k * E0 / m_e)

print(f"\n=== Particle Trapping ===")
print(f"E0 = {E0} V/m, k = {k} m^-1")
print(f"v_ph = {v_ph:.2e} m/s")
print(f"Bounce frequency ω_b = {omega_b:.2e} rad/s")
print(f"Bounce period τ_b = {2*np.pi/omega_b:.2e} s")

x_traj, v_traj = particle_in_wave(E0, k, m_e, -e, v_ph, num_particles=50,
                                   duration=2e-7, dt=1e-10)

# Plot phase space
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Initial phase space
ax1.scatter(x_traj[:, 0] * k / (2*np.pi), (v_traj[:, 0] - v_ph) / 1e3,
           c='blue', s=10, alpha=0.6)
ax1.set_xlabel('kx / 2π', fontsize=12)
ax1.set_ylabel('v - v_ph (km/s)', fontsize=12)
ax1.set_title('Initial Phase Space', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)

# Final phase space (with separatrix)
ax2.scatter(x_traj[:, -1] * k / (2*np.pi), (v_traj[:, -1] - v_ph) / 1e3,
           c='red', s=10, alpha=0.6, label='Particles')

# Separatrix
phi_0 = E0 / k
v_sep = np.sqrt(2 * e * phi_0 / m_e)
x_sep = np.linspace(0, 2*np.pi, 100)
v_upper = np.sqrt(2 * e * phi_0 / m_e * (1 + np.cos(x_sep)))
v_lower = -v_upper
ax2.plot(x_sep / (2*np.pi), v_upper / 1e3, 'k-', linewidth=2, label='Separatrix')
ax2.plot(x_sep / (2*np.pi), v_lower / 1e3, 'k-', linewidth=2)

ax2.set_xlabel('kx / 2π', fontsize=12)
ax2.set_ylabel('v - v_ph (km/s)', fontsize=12)
ax2.set_title('Final Phase Space (Trapped Particles)', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig('particle_trapping.png', dpi=150)
print("Saved: particle_trapping.png")
```

## 요약

Landau 감쇠는 플라즈마 물리학에서 가장 심오한 결과 중 하나입니다:

1. **분산 관계**: 선형화된 Vlasov-Poisson은 $v = \omega/k$에서 극점을 가진 $\epsilon(k,\omega) = 0$을 제공합니다.

2. **Landau 윤곽**: 인과성은 적분 경로가 극점 아래로 가도록 요구하며, 다음을 산출합니다:
   $$
   \epsilon = 1 - \sum_s \frac{\omega_{ps}^2}{k^2}\left[\mathcal{P}\int + i\pi\frac{df_0}{dv}\bigg|_{v=\omega/k}\right]
   $$

3. **감쇠율**: Maxwellian의 경우,
   $$
   \gamma \approx -\sqrt{\frac{\pi}{8}}\frac{\omega_{pe}}{(k\lambda_D)^3}\exp\left(-\frac{1}{2k^2\lambda_D^2}\right)
   $$
   $k\lambda_D \ll 1$에 대해 지수적으로 약함.

4. **물리적 메커니즘**: 공명 입자 ($v \approx v_{\text{ph}}$)가 파동과 에너지를 교환합니다. Maxwellian의 경우 (느린 것이 빠른 것보다 많음), 순 에너지 흐름은 파동 → 입자 → 감쇠.

5. **역 Landau 감쇠**: 공명에서 $df_0/dv > 0$ → 성장 (bump-on-tail 불안정성).

6. **비선형 효과**: 큰 진폭 → 입자 포획 → 위상 공간 소용돌이 → 준선형 완화.

7. **이온 음향 파동**: 낮은 감쇠는 $T_e \gg T_i$를 요구합니다.

Landau 감쇠는:
- **무충돌** (엔트로피 증가 없음)
- **가역적** (위상 혼합, 에코)
- **운동학적** (유체 모델은 이를 포착할 수 없음)

Landau 감쇠를 이해하는 것은 다음에 필수적입니다:
- 플라즈마 가열 (예: 파동 흡수)
- 안정성 분석
- 난류 감쇠
- 천체물리학적 플라즈마

## 연습 문제

### 문제 1: Bohm-Gross 분산

$k\lambda_D \ll 1$을 가정하고 감쇠를 무시하여, Maxwellian 분포에 대한 선형화된 Vlasov-Poisson 시스템으로부터 Bohm-Gross 분산 관계 $\omega^2 = \omega_{pe}^2 + 3k^2v_{th}^2$를 유도합니다.

**힌트**: 주값 적분을 사용하고 작은 $k\lambda_D$에 대해 전개합니다.

---

### 문제 2: 다른 kλ_D에서 Landau 감쇠

$n_e = 10^{19}$ m$^{-3}$ 및 $T_e = 100$ eV인 전자 플라즈마에 대해:

(a) $\omega_{pe}$ 및 $\lambda_D$를 계산합니다.

(b) $k\lambda_D = 0.2$, 0.5, 및 1.0에 대해, Landau 감쇠율 $\gamma$ 및 감쇠 감소량 $|\gamma|/\omega_r$를 계산합니다.

(c) 진폭이 $e$ 인자만큼 감쇠하기 전에 파동이 몇 번의 진동을 겪습니까?

(d) $\omega_r$의 분수로 감쇠가 가장 강한 $k\lambda_D$는 무엇입니까?

---

### 문제 3: Bump-on-Tail 불안정성

다음 분포 함수를 고려합니다:

$$
f_0(v) = n_c\sqrt{\frac{m}{2\pi k_BT_c}}\exp\left(-\frac{mv^2}{2k_BT_c}\right) + n_b\sqrt{\frac{m}{2\pi k_BT_b}}\exp\left(-\frac{m(v-v_b)^2}{2k_BT_b}\right)
$$

$n_b = 0.1 n_c$, $T_b = T_c$, 및 $v_b = 3v_{th,c}$ (여기서 $v_{th,c} = \sqrt{k_BT_c/m}$).

(a) $f_0(v)$를 플롯하고 $df_0/dv > 0$인 영역을 식별합니다.

(b) 성장율이 최대인 위상 속도 $v_{\text{ph}}$를 추정합니다.

(c) Landau 공식을 사용하여, $v_{\text{ph}}$에서 성장율을 추정합니다.

(d) 준선형 완화가 시간이 지남에 따라 분포를 어떻게 평탄화할지 논의합니다.

---

### 문제 4: 이온 음향 파동 감쇠

$n = 10^{18}$ m$^{-3}$, $T_e = 1$ keV, 및 $T_i = 100$ eV인 수소 플라즈마에서:

(a) 이온 음속 $c_s = \sqrt{k_BT_e/m_i}$를 계산합니다.

(b) 이온 열속도 $v_{th,i} = \sqrt{k_BT_i/m_i}$를 추정합니다.

(c) $c_s$와 $v_{th,i}$를 비교합니다. 이온 Landau 감쇠는 약합니까 강합니까?

(d) 이온 음향 파동이 약한 감쇠 ($|\gamma|/\omega \ll 1$)로 전파하기 위한 $T_i/T_e$에 대한 조건은 무엇입니까?

---

### 문제 5: 입자 포획과 바운스 주파수

진폭 $E_0 = 10^4$ V/m 및 파수 $k = 10^5$ m$^{-1}$를 가진 파동이 전자 플라즈마에서 전파합니다.

(a) 바운스 주파수 $\omega_b = \sqrt{ekE_0/m_e}$를 계산합니다.

(b) 포획된 영역의 속도 공간 폭을 추정합니다: $\Delta v_{\text{trap}} \sim \omega_b/k$.

(c) $T_e = 10$ eV인 Maxwellian의 경우, 위상 속도 $v_{\text{ph}} = 10^6$ m/s의 $\Delta v_{\text{trap}}$ 내에서 속도를 가진 전자의 비율은 무엇입니까?

(d) 바운스 주파수가 $\omega_b \sim 10^8$ rad/s이고 플라즈마 주파수가 $\omega_{pe} \sim 10^{11}$ rad/s이면, 포획 시간 척도는 플라즈마 진동에 비해 빠릅니까 느립니까?

---

## 내비게이션

- **이전**: [Vlasov 방정식](./07_Vlasov_Equation.md)
- **다음**: [충돌 운동학](./09_Collisional_Kinetics.md)
