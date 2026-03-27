# 12. 파동 가열과 불안정성

## 학습 목표

- 핵융합 플라즈마에서 파동 가열의 물리적 메커니즘 이해
- 속도 공간 불안정성 (빔-플라즈마, bump-on-tail, Weibel) 이론 마스터
- 자화 플라즈마에서 압력 구동 불안정성 (firehose, mirror) 분석
- 레이저-플라즈마 상호작용에서 파라메트릭 불안정성 조건 학습
- 핵융합 및 천체물리학의 실용적 문제에 불안정성 이론 적용
- 다양한 불안정성 메커니즘에 대한 성장률과 안정성 경계 계산

## 소개

플라즈마에서 파동은 두 가지 중요한 역할을 합니다:
1. **가열 및 전류 구동**: 외부 파동이 플라즈마 입자에 에너지를 전달
2. **불안정성**: 파동이 자발적으로 성장하여 플라즈마로부터 자유 에너지를 추출

이 레슨은 두 측면을 모두 다루며, 다음에 초점을 맞춥니다:
- **파동 가열**: RF 파동이 핵융합 플라즈마에 에너지를 침착시키는 방법
- **속도 공간 불안정성**: 비Maxwell 분포로부터 발생
- **압력 구동 불안정성**: 온도 비등방성으로부터 발생
- **파라메트릭 불안정성**: 고출력 레이저 시스템에서 파동-파동 결합

이러한 현상은 다음에 중요합니다:
- 핵융합 반응로 설계 (가열 시스템, 전류 구동)
- 천체물리학적 플라즈마 (태양풍, 펄서 자기권, GRB afterglows)
- 레이저-플라즈마 상호작용 (관성 가둠 핵융합)
- 우주 기상 (복사 벨트, 자기권 역학)

## 1. 핵융합 플라즈마에서의 파동 가열

### 1.1 가열 방법 개요

핵융합 플라즈마는 $T \sim 10-20$ keV ($\sim 100-200$ 백만 K)의 온도가 필요합니다. 세 가지 주요 가열 방법:

**Ohmic 가열**:
- $P = I^2 R$ 여기서 $R \propto T_e^{-3/2}$ (고전 저항률)
- 낮은 $T$에서 효과적, 높은 $T$에서 비효과적
- 토카막에서 $\sim 1-2$ keV로 제한

**중성 빔 주입 (NBI)**:
- 빠른 중성입자 (50-1000 keV) 주입, 이온화, 충돌을 통해 에너지 전달
- 파동 방법이 아니지만 비교를 위해 중요
- ITER에서 빔라인당 10-50 MW 출력

**고주파 (RF) 가열**:
- 안테나 또는 도파관에서 발사된 전자기파
- 세 가지 주파수 범위: ECRH, ICRH, LHCD
- 장점: 국소화된 침착, 전류 구동 능력, 입자 소스 없음

### 1.2 전자 사이클로트론 공명 가열 (ECRH)

**주파수**: $\omega \approx n\omega_{ce}$ 여기서 일반적으로 $n = 1, 2$

**공명 조건**: $\omega = n\omega_{ce}(r)$인 공간 위치에서, 파동 위상이 전자 회전과 일치합니다.

**흡수 메커니즘**:
- 파동과 공명하는 전자 ($\omega - k_\parallel v_\parallel = n\omega_{ce}$)
- $\mathbf{B}$에 수직인 파동 전기장이 회전하는 전자에 일을 함
- 출력 흡수: $P \propto \int d^3v \, \mathbf{j} \cdot \mathbf{E}$

**분산**: 밀도에 따라 X-모드 또는 O-모드 사용
- O-모드: $n_c = \epsilon_0 m_e \omega^2/e^2$에서 차단
- X-모드: 더 높은 차단, 과밀 플라즈마에 더 좋음

**일반 파라미터** (ITER):
- 주파수: 170 GHz ($n=2$에서 $B \sim 5.3$ T의 경우)
- 출력: 총 20 MW (자이로트론)
- 빔 폭: $\sim 5$ cm (고도로 국소화)

**장점**:
- 우수한 국소화 ($\Delta r \sim$ cm)
- 전류 구동 능력 (ECCD)
- 침착 위치의 실시간 제어

**과제**:
- 고주파 자이로트론 필요 (비싼)
- 도파관에서 전송 손실
- 거울 정렬이 중요

### 1.3 이온 사이클로트론 공명 가열 (ICRH)

**주파수**: $\omega \approx n\omega_{ci}$ 여기서 $\omega_{ci} = ZeB/m_i$

**공명 조건**: 이온은 토카막에서 $\omega_{ci} \sim 2\pi \times (30-100)$ MHz로 회전합니다.

**가열 방식**:

**기본 다수 가열**: 주요 이온 종에 대해 $\omega = \omega_{ci}$
- 직접 공명, 하지만 약한 단일 통과 흡수
- 여러 통과가 필요

**2차 고조파**: $\omega = 2\omega_{ci}$
- 기본보다 강한 흡수
- 저자장 장치에서 사용

**소수 가열**: $\omega = \omega_{ci,\text{minority}}$
- 소수 종 도입 (예: D 플라즈마에서 5-10% $^3$He)
- $\omega_{ci}(^3\text{He})$에서 공명, 벌크 중수소는 비공명
- 소수 이온이 고에너지로 가열되어 충돌을 통해 벌크로 전달

**모드 변환**: 빠른 파동이 ion Bernstein 파동 또는 ion cyclotron 파동으로 변환
- 하이브리드 공명 층 근처에서 발생
- 전자를 효율적으로 가열 가능

**일반 파라미터**:
- 주파수: 40-80 MHz
- 출력: 20 MW (ITER)
- 안테나: 용기 벽의 대형 코일

**장점**:
- 잘 확립된 기술
- 이온과 전자 모두 가열 가능
- 중심 가열 가능

**과제**:
- 안테나-플라즈마 상호작용 (불순물, 열점)
- 기생 손실
- ECRH보다 덜 국소화

### 1.4 하부 하이브리드 전류 구동 (LHCD)

**주파수**: $\omega_{ci} \ll \omega \ll \omega_{ce}$ (일반적으로 1-8 GHz)

**목적**: 주로 **전류 구동**용, 가열이 아님
- 비유도 전류 생성
- 토카막에서 정상 상태 작동

**메커니즘**:
- 하부 하이브리드파가 높은 $n_\parallel = k_\parallel c/\omega$로 전파
- 꼬리 전자에 강한 Landau 감쇠 ($v_\parallel \sim \omega/k_\parallel$)
- 비대칭 감쇠가 순 전류 생성

**전류 구동 효율**:
$$\eta_{CD} = \frac{n_{20} I_A R}{P_{\text{MW}}}$$

여기서 $n_{20}$은 $10^{20}$ m$^{-3}$ 단위의 밀도, $I_A$는 MA 단위의 전류, $R$은 m 단위의 주반경, $P$는 MW 단위의 출력입니다.

일반적: LHCD에 대해 $\eta_{CD} \sim 0.2-0.5$.

**접근성**: 하부 하이브리드파가 원하는 위치로 침투해야 함
- 밀도 제한: 파동 차단이 발생하는 $n < n_{\text{access}}$
- 고밀도 코어에서는 침투하지 못할 수 있음

**일반 파라미터**:
- 주파수: 3.7-5 GHz
- 출력: 20 MW (ITER)
- 발사기: 도파관 배열 (그릴)

**장점**:
- 높은 전류 구동 효율
- 축외 전류 프로파일 제어

**과제**:
- 접근성을 위한 밀도 제한
- 스펙트럼 갭 (출력 결합 어려움)
- 고출력에서 비선형 효과

### 1.5 가열 방법 비교

| 방법 | 주파수 | 주요 대상 | 국소화 | 전류 구동 | 출력 (ITER) |
|--------|-----------|-------------|--------------|---------------|--------------|
| ECRH   | 140-170 GHz | 전자 | 우수 | 예 (ECCD) | 20 MW |
| ICRH   | 40-80 MHz | 이온 | 보통 | 약함 | 20 MW |
| LHCD   | 3-8 GHz | 전자 | 좋음 | 우수 | 20 MW |
| NBI    | N/A | 이온 | 나쁨 | 예 | 33 MW |

**시너지**: 방법 결합이 종종 최적
- NBI + ICRH: NBI가 빠른 이온 꼬리 생성, ICRH가 꼬리를 더 가열
- ECRH + LHCD: MHD 제어를 위한 ECRH, 전류 프로파일을 위한 LHCD

## 2. 속도 공간 불안정성

### 2.1 빔-플라즈마 불안정성 (Two-Stream)

밀도 $n_b$와 속도 $v_0$를 가진 **냉각 전자 빔**이 밀도 $n_0$를 가진 배경 플라즈마를 통해 흐르는 것을 고려합니다.

**설정**:
- 빔: $f_b(\mathbf{v}) = n_b \delta(v_x - v_0)\delta(v_y)\delta(v_z)$
- 배경: $f_0(\mathbf{v}) = n_0 \delta(v_x)\delta(v_y)\delta(v_z)$
- 둘 다 냉각 ($T = 0$)

**분산 관계**: 선형화 Vlasov + Poisson으로부터:

$$1 = \frac{\omega_{p0}^2}{\omega^2} + \frac{\omega_{pb}^2}{(\omega - kv_0)^2}$$

여기서 $\omega_{p0}^2 = n_0 e^2/(\epsilon_0 m_e)$이고 $\omega_{pb}^2 = n_b e^2/(\epsilon_0 m_e)$입니다.

**분석**: $\omega = \omega_r + i\gamma$를 가정하고 불안정한 해를 찾습니다 ($\gamma > 0$).

$n_b \ll n_0$의 경우, Langmuir 파동 $\omega \approx \omega_{p0} + \delta\omega$ 주위로 전개:

$$\delta\omega \approx -\frac{\omega_{pb}^2}{2\omega_{p0}} \frac{1}{1 - kv_0/\omega_{p0}}$$

분모가 작을 때, $\delta\omega$가 커집니다. $kv_0 \approx \omega_{p0}$의 경우, 보정이 허수가 됩니다.

**성장률** ($n_b/n_0 \ll 1$의 경우):

$$\boxed{\gamma \approx \omega_{p0} \left(\frac{n_b}{n_0}\right)^{1/3}}$$

불안정성은 다음일 때 **가장 강합니다**:
$$kv_0 \approx \omega_{p0}$$

**물리적 그림**:
```
빔 전자:  ──→  ──→  ──→  ──→
배경:      ·    ·    ·    ·

섭동이 뭉침을 생성:
  ──→ ──→    ──→ ──→     (밀도파)

뭉침이 전기장 강화 → 피드백 → 성장
```

**응용**:
- 플라즈마의 전자 빔 (가속기, 우주)
- 전리층 불안정성
- 입자 가속기에서 초기 문제 야기

### 2.2 Bump-on-Tail 불안정성

더 현실적인 시나리오: Maxwell 배경에 **소수의 빠른 전자**.

**분포**:
$$f(v) = f_M(v) + f_{\text{bump}}(v)$$

여기서 $f_{\text{bump}}$는 $v \sim v_{\text{bump}} > v_{th}$에서 작은 집단입니다.

**불안정성 기준**: 분포는 공명 속도에서 속도 공간에서 **양의 기울기**를 가져야 합니다:

$$\frac{\partial f}{\partial v}\bigg|_{v = \omega/k} > 0$$

이것은 **역 Landau 감쇠**입니다: $v = v_\phi$에서 입자가 파동으로부터 에너지를 받는 대신 파동에 에너지를 전달합니다.

**성장률**: 밀도 $n_b$와 폭 $\Delta v$를 가진 작은 bump의 경우:

$$\gamma \sim \omega_{pe} \left(\frac{n_b}{n_0}\right)^{1/3} \frac{v_{\text{bump}}}{v_{th}}$$

**준선형 이완**: 파동이 성장함에 따라, 다음을 통해 속도 공간에서 입자를 확산시킵니다:

$$\frac{\partial f}{\partial t} = \frac{\partial}{\partial v}\left(D \frac{\partial f}{\partial v}\right)$$

여기서 $D \propto |E_k|^2$는 확산 계수입니다.

결과: **bump가 평평해져** 고원으로:
```
초기:        포화 후:
f(v)            f(v)
  |\              |----\
  | \             |     \
  |  \___         |      \___
  |      \        |          \
  +-------v       +----------v
      v_bump           plateau
```

이 **준선형 고원 형성**은 기본적인 비선형 포화 메커니즘입니다.

**응용**:
- 토카막의 도주 전자
- 태양풍 전자 빔
- 레이저-플라즈마 상호작용

### 2.3 Weibel 불안정성

**Weibel 불안정성**은 **온도 비등방성**으로부터 성장합니다: $T_\perp > T_\parallel$ (또는 그 반대).

**물리적 메커니즘**:
- 비등방성 분포가 전류 요동을 생성
- 전류가 자기장 생성
- 자기장이 비등방성 강화 → 양의 피드백

**설정**: 분포를 고려:
$$f(v_\parallel, v_\perp) = n_0 \left(\frac{m}{2\pi k_B T_\parallel}\right)^{1/2}\left(\frac{m}{2\pi k_B T_\perp}\right) \exp\left(-\frac{mv_\parallel^2}{2k_BT_\parallel} - \frac{mv_\perp^2}{2k_BT_\perp}\right)$$

**분산** ($T_\perp > T_\parallel$의 경우): 순수하게 성장하는 모드 (실수 주파수 없음):

$$\omega = i\gamma$$

**성장률**:

$$\boxed{\gamma_{\text{max}} \approx \omega_{pe} \sqrt{\frac{T_\perp - T_\parallel}{T_\parallel}}}$$

파수의 경우:
$$k_{\text{max}} \approx \frac{\omega_{pe}}{c}\sqrt{\frac{T_\perp}{T_\parallel} - 1}$$

**생성된 자기장**: 불안정성이 강도가 다음인 **소규모 자기장**을 생성합니다:

$$\frac{B^2}{8\pi} \sim n k_B (T_\perp - T_\parallel)$$

**응용**:
- **무충돌 충격**: 천체물리학적 충격 (예: 초신성 잔해, GRB afterglows)에서, Weibel 불안정성이 충격을 매개하는 자기장을 생성
- **레이저-플라즈마 상호작용**: 강한 레이저가 비등방성 전자 분포 생성 → Weibel 불안정성 → 자기장 생성
- **자기권 플라즈마**: 복사 벨트의 비등방성 분포
- **자기장 생성**: Weibel은 우주론에서 씨앗 장에 대한 메커니즘

이 불안정성은 Weibel (1959)에 의해 이론적으로 예측되었고 레이저-플라즈마 실험 (2000년대)에서 실험적으로 확인되었습니다.

## 3. 압력 구동 불안정성

### 3.1 Firehose 불안정성

$p_\parallel > p_\perp$ (평행 압력이 수직 압력을 초과)를 가진 자화 플라즈마에서, **firehose 불안정성**이 발생할 수 있습니다.

**비유**: 압력이 너무 높을 때 꿈틀거리는 가압 정원 호스처럼.

**물리적 메커니즘**:
- 자기장 선이 구부러짐
- 평행 압력이 구부러진 장을 따라 플라즈마를 밀어냄
- 곡률 증가 → 양의 피드백

**안정성 기준**: 비등방성 압력을 가진 MHD로부터:

$$\boxed{p_\parallel - p_\perp < \frac{B^2}{\mu_0}}$$

또는 동등하게:

$$\beta_\parallel - \beta_\perp < 1$$

여기서 $\beta_\parallel = 2\mu_0 p_\parallel/B^2$이고 $\beta_\perp = 2\mu_0 p_\perp/B^2$입니다.

**성장률** (불안정한 경우):

$$\gamma^2 \approx k^2 v_A^2 \left(\frac{p_\parallel - p_\perp}{p_\parallel + p_\perp/2} - \frac{1}{\beta_\parallel}\right)$$

여기서 $v_A = B/\sqrt{\mu_0 \rho}$는 Alfvén 속도입니다.

**최대 성장**:
$$k \sim \frac{1}{L}$$

여기서 $L$은 시스템 크기입니다 (저$k$ 불안정성).

**관측**:
- **태양풍**: 종종 firehose에 대해 한계 안정/불안정
- **Magnetosheath**: 압축된 플라즈마가 안정성 조건을 위반할 수 있음
- **토카막 가장자리**: 빠른 이온 집단이 firehose를 구동할 수 있음

**포화**: 피치각 산란이 비등방성을 이완시켜 불안정성을 억제합니다.

### 3.2 Mirror 불안정성

반대 비등방성, $p_\perp > p_\parallel$은 **mirror 불안정성**을 구동할 수 있습니다.

**물리적 메커니즘**:
- 자기장 강도가 요동: $B = B_0 + B_1$
- 높은 $\mu = mv_\perp^2/(2B)$를 가진 입자가 저$B$ 영역에 갇힘 (자기 거울)
- 저$B$ 영역에서 강화된 $p_\perp$ → $B$가 더 감소 → 피드백

**안정성 기준**:

$$\boxed{\frac{p_\perp}{p_\parallel} < 1 + \frac{1}{\beta_\perp}}$$

또는:

$$\beta_\perp\left(\frac{p_\perp}{p_\parallel} - 1\right) < 1$$

**성장률**: $p_\perp/p_\parallel - 1 = A$ (비등방성)의 경우:

$$\gamma \approx k_\parallel v_A \sqrt{A \beta_\perp}$$

$k_\parallel L \sim 1$의 경우, 여기서 $L$은 스케일 길이입니다.

**특성**:
- **비전파**: $\omega_r = 0$ (순수하게 성장)
- **압축성**: $\delta B_\parallel$과 $\delta n$을 생성
- **비등방성 구조**: $\mathbf{B}$를 따라 길쭉함

**관측**:
- **태양풍**: Mirror 모드 구조 (반상관 $B$와 $n$을 가진 느린 모드 구조)
- **Magnetosheath**: 매우 일반적, 준정상 구조
- **행성 자기권**: 목성, 토성

**포화**: 입자를 가두는 준정적 자기 병을 생성하여 비등방성을 감소시킵니다.

### 3.3 비교: Firehose vs Mirror

| 속성 | Firehose | Mirror |
|----------|----------|--------|
| 비등방성 | $p_\parallel > p_\perp$ | $p_\perp > p_\parallel$ |
| 기준 | $\beta_\parallel - \beta_\perp < 1$ | $\beta_\perp(p_\perp/p_\parallel - 1) < 1$ |
| $\omega_r$ | 유한 (전파) | 0 (비전파) |
| $\delta B$ | 횡방향 | 압축성 |
| 포화 | 피치각 산란 | 자기 포획 |

두 불안정성 모두 압력이 비등방성으로 남을 수 있는 **무충돌 플라즈마**에서 편재합니다 (충돌 시간 $\gg$ 역학 시간).

## 4. 파라메트릭 불안정성

### 4.1 삼파 결합

**파라메트릭 불안정성**은 세 파동의 결합을 포함합니다:
$$\omega_0 = \omega_1 + \omega_2$$
$$\mathbf{k}_0 = \mathbf{k}_1 + \mathbf{k}_2$$

여기서 파동 0 (펌프)이 파동 1과 2 (딸 파동)로 붕괴합니다.

**메커니즘**:
- 펌프 파동이 밀도/속도 진동 생성
- 진동이 플라즈마 응답을 변조
- 변조된 플라즈마가 일치 조건이 만족되면 딸 파동을 증폭할 수 있음

**성장률**: 펌프 진폭에 비례:
$$\gamma \propto \sqrt{\frac{I}{I_c}}$$

여기서 $I$는 펌프 강도이고 $I_c$는 임계값입니다.

### 4.2 Stimulated Raman Scattering (SRS)

**과정**: 전자기파 (펌프) $\to$ EM파 (산란) + Langmuir 파동

**일치 조건**:
$$\omega_0 = \omega_s + \omega_L$$
$$\mathbf{k}_0 = \mathbf{k}_s + \mathbf{k}_L$$

여기서 $\omega_L \approx \omega_{pe}$ (Langmuir 파동)이고 $\omega_s < \omega_0$ (산란 EM파)입니다.

**분산 제약**:
- 펌프: $\omega_0^2 = \omega_{pe}^2 + k_0^2 c^2$
- 산란: $\omega_s^2 = \omega_{pe}^2 + k_s^2 c^2$
- Langmuir: $\omega_L^2 \approx \omega_{pe}^2 + 3k_L^2 v_{th}^2$

**성장률**:

$$\gamma_{SRS} = \frac{k_L v_{osc}}{4} \left(\frac{\omega_0}{\omega_s}\right)^{1/2}$$

여기서 $v_{osc} = eE_0/(m_e\omega_0)$는 펌프 파동에서 진동 속도입니다.

**임계값**: $\gamma > \nu_L$을 요구, 여기서 $\nu_L$은 Landau 감쇠율입니다.

**관련성**: 레이저 핵융합 (ICF)
- 고출력 레이저 ($I \sim 10^{15}$ W/cm$^2$)가 SRS를 구동할 수 있음
- 산란된 빛 손실 → 결합 효율 감소
- Langmuir 파동 가열로부터 열전자 → 타겟 예열 (압축에 나쁨)

**완화**:
- 대역폭: 광대역 레이저가 일관성 감소
- 빔 평활화: 국소 강도 스파이크 감소
- 파장: 더 짧은 파장 (UV)이 더 높은 임계값을 가짐

### 4.3 Stimulated Brillouin Scattering (SBS)

**과정**: EM파 $\to$ EM파 + 이온 음향파

**일치**:
$$\omega_0 = \omega_s + \omega_{ia}$$
$$\mathbf{k}_0 = \mathbf{k}_s + \mathbf{k}_{ia}$$

여기서 $\omega_{ia} = k_{ia} c_s$ (이온 음향파)입니다.

**성장률**:

$$\gamma_{SBS} = \frac{k_{ia} v_{osc}}{4\sqrt{2}} \sqrt{\frac{\omega_0}{\omega_{ia}}}$$

**특성**:
- SRS보다 낮은 임계값 (이온 음향 감쇠가 Landau 감쇠보다 약함)
- **후방산란**: $\mathbf{k}_s \approx -\mathbf{k}_0$에 대해 가장 강함
- 레이저 에너지의 상당 부분을 반사할 수 있음

**관련성**: SBS는 종종 레이저 핵융합에서 **지배적인** 파라메트릭 불안정성입니다.

**완화**: SRS와 유사, 추가로:
- 가스 충전 hohlraum이 SBS 감소
- 다중 이온 종 (이온 음향 감쇠 증가)

### 4.4 관성 가둠 핵융합 (ICF)에 미치는 영향

**National Ignition Facility (NIF)** 및 기타 ICF 실험에서:
- 레이저 출력: $\sim 500$ TW
- 강도: hohlraum에서 $10^{14}-10^{15}$ W/cm$^2$
- SRS와 SBS가 레이저 에너지의 10-50%를 반사할 수 있음

**결과**:
- 타겟에 대한 결합 감소 → 낮은 압축
- SRS로부터 열전자가 연료 예열 → 이득 감소
- 구동에서 비대칭

**최근 진전** (2022-2023): NIF가 다음을 통해 점화 ($Q > 1$)를 달성:
- 개선된 hohlraum 설계
- 더 나은 빔 평활화
- 더 높은 레이저 에너지 (2.05 MJ)
- 파장 디튜닝을 통한 SRS/SBS 완화

## 5. 불안정성 분류

### 5.1 자유 에너지 소스

불안정성은 다음으로부터 자유 에너지를 추출합니다:

1. **속도 공간**: 비Maxwell 분포
   - 빔-플라즈마: 상대적 드리프트
   - Bump-on-tail: 양의 $\partial f/\partial v$
   - Weibel: 온도 비등방성

2. **구성 공간**: 밀도, 온도, 자기장의 기울기
   - 드리프트파 (여기서 다루지 않음)
   - 교환 모드
   - Tearing 모드

3. **전류**: 평행 또는 수직 전류
   - 전류 구동 불안정성
   - Kink 모드

4. **외부 구동**: 외부 파동에 의해 펌핑
   - 파라메트릭 불안정성 (SRS, SBS)

### 5.2 불안정성 요약 표

| 불안정성 | 자유 에너지 | 조건 | 성장률 | 응용 |
|-------------|-------------|-----------|-------------|-------------|
| 빔-플라즈마 | 빔 드리프트 $v_0$ | $kv_0 \sim \omega_{pe}$ | $\omega_{pe}(n_b/n_0)^{1/3}$ | 가속기, 우주 |
| Bump-on-tail | $\partial f/\partial v > 0$ | 공명 입자 | $\omega_{pe}(n_b/n_0)^{1/3}$ | 도주, 태양풍 |
| Weibel | $T_\perp > T_\parallel$ | 비등방성 | $\omega_{pe}\sqrt{(T_\perp-T_\parallel)/T_\parallel}$ | 충격, 레이저 |
| Firehose | $p_\parallel > p_\perp$ | $\beta_\parallel - \beta_\perp > 1$ | $k v_A \sqrt{\Delta p/p}$ | 태양풍 |
| Mirror | $p_\perp > p_\parallel$ | $\beta_\perp(p_\perp/p_\parallel-1) > 1$ | $k_\parallel v_A \sqrt{A\beta_\perp}$ | Magnetosheath |
| SRS | 펌프 레이저 | $I > I_c$ | $(k_L v_{osc}/4)\sqrt{\omega_0/\omega_s}$ | 레이저 핵융합 |
| SBS | 펌프 레이저 | $I > I_c$ | $(k_{ia}v_{osc}/4\sqrt{2})\sqrt{\omega_0/\omega_{ia}}$ | 레이저 핵융합 |

## 6. Python 구현

### 6.1 Two-Stream 불안정성 분산

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def two_stream_dispersion(k, omega_p0, omega_pb, v0):
    """
    Solve two-stream dispersion: 1 = ω_p0²/ω² + ω_pb²/(ω-kv0)²

    Returns complex frequency ω(k).
    """
    def dispersion_eq(omega_complex):
        omega = omega_complex[0] + 1j * omega_complex[1]
        # 분산 관계는 ε = 0으로 표현되며, 두 개의 독립적인 Langmuir 항으로 구성됩니다:
        # 배경 플라즈마는 ω_p0²/ω² (정지 상태)를 기여하고, 빔은
        # 빔 정지계에서 Doppler 이동된 ω_pb²/(ω - kv0)²를 기여합니다.
        eps = 1 - omega_p0**2/omega**2 - omega_pb**2/(omega - k*v0)**2
        return [np.real(eps), np.imag(eps)]

    # (ω_p0, 0) 근처에서 시작하는 이유는 이 불안정성이 배경 Langmuir 파동의
    # 섭동이기 때문입니다; 작은 허수 부분은 근 탐색기를 성장하는 해(Im ω > 0)가
    # 존재하는 상반 평면(upper half-plane)으로 안내합니다.
    omega_guess = [omega_p0, 0.1 * omega_p0]

    sol = fsolve(dispersion_eq, omega_guess)
    return sol[0] + 1j * sol[1]

# Parameters
n0 = 1e19  # m^-3
nb_frac = 0.01  # nb/n0 = 1%
m_e = 9.109e-31  # kg
e = 1.602e-19  # C
epsilon_0 = 8.854e-12  # F/m

omega_p0 = np.sqrt(n0 * e**2 / (epsilon_0 * m_e))
# ω_pb << ω_p0인 이유는 n_b << n_0이기 때문입니다; 이것이 약한 빔 한계(weak-beam limit)로,
# 섭동론적 성장률 γ ~ ω_p0 (n_b/n_0)^(1/3)이 유효한 영역입니다.
omega_pb = np.sqrt(nb_frac * n0 * e**2 / (epsilon_0 * m_e))

# v0는 공명 조건 k v0 ~ ω_p0가 스캔 범위 중간에 오도록 선택합니다;
# 이렇게 하면 최대 성장 영역이 완전히 해상됩니다.
v0 = 2 * omega_p0 * (1e8 / omega_p0)  # Choose v0 ~ ω_p0/k_typical

print(f"Background plasma frequency: ω_p0 = {omega_p0:.2e} rad/s")
print(f"Beam plasma frequency: ω_pb = {omega_pb:.2e} rad/s")
print(f"Beam velocity: v0 = {v0:.2e} m/s")

# k를 ω_p0/v0 근방으로 스캔하면 공명 파수 kv0 = ω_p0 주변에 스캔이 집중됩니다;
# Langmuir 파동의 위상 속도가 빔과 일치하는 이 지점에서 최대 성장이 일어납니다.
k_array = np.linspace(0.5, 3, 100) * omega_p0 / v0

omega_real = []
omega_imag = []

for k in k_array:
    omega = two_stream_dispersion(k, omega_p0, omega_pb, v0)
    omega_real.append(np.real(omega))
    omega_imag.append(np.imag(omega))

omega_real = np.array(omega_real)
omega_imag = np.array(omega_imag)

# Analytical approximation for small nb/n0
gamma_approx = omega_p0 * (nb_frac)**(1/3) * np.ones_like(k_array)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Real frequency
ax1.plot(k_array * v0 / omega_p0, omega_real / omega_p0, 'b-',
         linewidth=2, label='Numerical')
ax1.axhline(1, color='r', linestyle='--', label='$\\omega_{p0}$')
ax1.plot(k_array * v0 / omega_p0, k_array * v0 / omega_p0, 'g--',
         label='$kv_0$')
ax1.set_xlabel('$kv_0 / \\omega_{p0}$', fontsize=13)
ax1.set_ylabel('$\\omega_r / \\omega_{p0}$', fontsize=13)
ax1.set_title('Two-Stream: Real Frequency', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Growth rate
ax2.plot(k_array * v0 / omega_p0, omega_imag / omega_p0, 'b-',
         linewidth=2, label='Numerical')
ax2.plot(k_array * v0 / omega_p0, gamma_approx / omega_p0, 'r--',
         linewidth=1.5, label=f'Approx: $(n_b/n_0)^{{1/3}} = {nb_frac**(1/3):.3f}$')
ax2.set_xlabel('$kv_0 / \\omega_{p0}$', fontsize=13)
ax2.set_ylabel('$\\gamma / \\omega_{p0}$', fontsize=13)
ax2.set_title(f'Two-Stream: Growth Rate ($n_b/n_0 = {nb_frac}$)', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 0.5])

plt.tight_layout()
plt.savefig('two_stream_instability.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.2 Weibel 불안정성 성장률

```python
def weibel_growth_rate(T_perp, T_parallel, n, B0=0):
    """
    Weibel instability growth rate.

    γ_max ≈ ω_pe √[(T_⊥ - T_∥)/T_∥]
    """
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    # 비등방성(anisotropy) A = (T_⊥ - T_∥)/T_∥는 Weibel 문제에서 유일한 무차원
    # 자유 에너지 파라미터입니다; 분포가 등방성에서 얼마나 벗어났는지, 즉
    # 불안정성을 구동하는 데 사용 가능한 에너지가 얼마인지를 나타냅니다.
    anisotropy = (T_perp - T_parallel) / T_parallel

    # γ_max ~ ω_pe √A는 Vlasov-Maxwell 분석에서 나옵니다: 자기 섭동이
    # 비등방성 전류와 결합하고, ω_pe가 생성된 전류 필라멘트에 대한
    # 전자기 응답의 시간 스케일을 설정합니다.
    if anisotropy > 0:
        gamma_max = omega_pe * np.sqrt(anisotropy)
    else:
        # 음의 비등방성(T_∥ > T_⊥)은 Weibel이 아닌 firehose 불안정성을
        # 구동합니다; 해당 영역에서 Weibel 성장이 없음을 나타내기 위해 0을 반환합니다.
        gamma_max = 0

    return gamma_max, omega_pe

# Parameters
n = 1e20  # m^-3
T_parallel = 1e3  # eV
T_perp_array = np.linspace(1e3, 10e3, 100)  # eV

gamma_array = []
for T_perp in T_perp_array:
    gamma, omega_pe = weibel_growth_rate(T_perp, T_parallel, n)
    gamma_array.append(gamma)

gamma_array = np.array(gamma_array)
anisotropy_array = (T_perp_array - T_parallel) / T_parallel

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Growth rate vs anisotropy
ax1.plot(anisotropy_array, gamma_array / omega_pe, 'b-', linewidth=2)
ax1.set_xlabel('Anisotropy $(T_\\perp - T_\\parallel)/T_\\parallel$', fontsize=13)
ax1.set_ylabel('$\\gamma / \\omega_{pe}$', fontsize=13)
ax1.set_title('Weibel Instability Growth Rate', fontsize=14)
ax1.grid(True, alpha=0.3)

# Growth rate vs T_perp
ax2.plot(T_perp_array / 1e3, gamma_array / omega_pe, 'r-', linewidth=2)
ax2.axvline(T_parallel / 1e3, color='k', linestyle='--',
            label=f'$T_\\parallel = {T_parallel/1e3:.0f}$ keV')
ax2.set_xlabel('$T_\\perp$ (keV)', fontsize=13)
ax2.set_ylabel('$\\gamma / \\omega_{pe}$', fontsize=13)
ax2.set_title(f'Growth Rate vs Perpendicular Temperature', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('weibel_instability.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nWeibel instability at T_⊥ = {T_perp_array[-1]/1e3:.0f} keV, T_∥ = {T_parallel/1e3:.0f} keV:")
print(f"  Anisotropy: {anisotropy_array[-1]:.1f}")
print(f"  γ/ω_pe: {gamma_array[-1]/omega_pe:.2f}")
```

### 6.3 Firehose와 Mirror 안정성 경계

```python
def firehose_criterion(beta_parallel, beta_perp):
    """
    Firehose stability: β_∥ - β_⊥ < 1
    Returns True if stable.
    """
    return (beta_parallel - beta_perp) < 1

def mirror_criterion(beta_perp, p_perp, p_parallel):
    """
    Mirror stability: β_⊥(p_⊥/p_∥ - 1) < 1
    Returns True if stable.
    """
    return beta_perp * (p_perp / p_parallel - 1) < 1

# Generate stability diagram
beta_perp_range = np.linspace(0, 5, 200)
beta_parallel_range = np.linspace(0, 5, 200)

Beta_perp, Beta_parallel = np.meshgrid(beta_perp_range, beta_parallel_range)

# Firehose boundary: β_∥ - β_⊥ = 1
firehose_stable = Beta_parallel - Beta_perp < 1

# Mirror boundary: β_⊥(p_⊥/p_∥ - 1) = 1
# → p_⊥/p_∥ = 1 + 1/β_⊥
# Assume isotropic for simplicity in demo (real case needs p_ratio)
# For demo, use β_⊥(β_∥/β_⊥ - 1) < 1 → β_∥ < β_⊥ + 1
mirror_stable = Beta_parallel < Beta_perp + 1

# Combined stability region
stable = firehose_stable & mirror_stable

fig, ax = plt.subplots(figsize=(10, 8))

# Plot stability regions
ax.contourf(Beta_perp, Beta_parallel, stable.astype(int),
            levels=[0, 0.5, 1], colors=['red', 'green'], alpha=0.3)

# Boundaries
beta_line = np.linspace(0, 5, 100)
ax.plot(beta_line, beta_line + 1, 'b-', linewidth=2,
        label='Firehose boundary: $\\beta_\\parallel - \\beta_\\perp = 1$')
ax.plot(beta_line, beta_line - 1, 'r-', linewidth=2,
        label='Mirror boundary (approx)')

# Diagonal
ax.plot(beta_line, beta_line, 'k--', alpha=0.5, label='$\\beta_\\parallel = \\beta_\\perp$')

ax.set_xlabel('$\\beta_\\perp$', fontsize=14)
ax.set_ylabel('$\\beta_\\parallel$', fontsize=14)
ax.set_title('Pressure Anisotropy Stability Diagram', fontsize=15)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 5])
ax.set_ylim([0, 5])

# Annotate regions
ax.text(1, 3.5, 'Firehose\nUnstable', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
ax.text(3.5, 1, 'Mirror\nUnstable', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
ax.text(2, 2, 'Stable', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

plt.tight_layout()
plt.savefig('anisotropy_stability.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 6.4 파라메트릭 불안정성 임계값

```python
def srs_growth_rate(I_laser, n, T_e, lambda_laser=1.053e-6):
    """
    Stimulated Raman Scattering growth rate.

    Parameters:
    -----------
    I_laser : float
        Laser intensity (W/m²)
    n : float
        Density (m^-3)
    T_e : float
        Electron temperature (eV)
    lambda_laser : float
        Laser wavelength (m)

    Returns:
    --------
    gamma_SRS : float
        Growth rate (s^-1)
    """
    c = 3e8
    omega_0 = 2 * np.pi * c / lambda_laser
    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))

    # E_0는 시간 평균 Poynting 선속 I = ε_0 c E_0²/2에서 유도됩니다;
    # 진동 속도(quiver velocity) v_osc = eE_0/(m_e ω_0)는 레이저 장의 진동이
    # 이온에 대해 전자를 얼마나 강하게 변위시키는지를 나타내며,
    # 모든 파라메트릭 불안정성 공식에서 펌프 진폭의 자연스러운 척도입니다.
    E_0 = np.sqrt(2 * I_laser / (c * epsilon_0))
    v_osc = e * E_0 / (m_e * omega_0)

    # 후방산란 기하학(backscatter geometry)에서 ω_s = ω_0 - ω_pe입니다.
    # Langmuir 파동(ω_pe)이 주파수 불일치를 담당하며,
    # 이 근사는 ω_pe << ω_0일 때 유효합니다.
    omega_s = omega_0 - omega_pe  # Approximate

    # k_L = 2ω_0/c는 후방산란 파수입니다: 산란된 EM파가 방향을 역전하므로
    # (k_s ≈ -k_0), Langmuir 파동은 k_L = k_0 + |k_s| ≈ 2k_0을 만족해야 합니다.
    k_L = 2 * omega_0 / c  # Backscatter

    # √(ω_0/ω_s) 인자가 성장을 강화하는 이유는 산란파가 차단 주파수 근처에 있기
    # (ω_s가 ω_pe에 가까움) 때문입니다 — 군속도(group velocity)가 작아져 펌프에서
    # 딸 파동으로의 결맞음 에너지 전달이 더 오래 지속될 수 있습니다.
    gamma_SRS = (k_L * v_osc / 4) * np.sqrt(omega_0 / omega_s)

    return gamma_SRS

# Laser parameters
lambda_laser = 351e-9  # m (UV, 3ω Nd:glass)
I_array = np.logspace(13, 16, 100)  # W/m²
n = 0.1 * 1.1e21  # m^-3 (nc/10 where nc is critical density)
T_e = 3e3  # eV

gamma_array = []
for I in I_array:
    gamma = srs_growth_rate(I, n, T_e, lambda_laser)
    gamma_array.append(gamma)

gamma_array = np.array(gamma_array)

# Landau damping (approximate)
v_th = np.sqrt(2 * T_e * e / m_e)
omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
k_L = 4 * np.pi / lambda_laser
zeta = omega_pe / (k_L * v_th)
gamma_Landau = omega_pe * np.sqrt(np.pi/8) * np.exp(-zeta**2/2) / (k_L**3 * (v_th/omega_pe)**3)

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(I_array / 1e15, gamma_array / omega_pe, 'b-',
          linewidth=2, label='SRS growth rate')
ax.axhline(gamma_Landau / omega_pe, color='r', linestyle='--',
           linewidth=2, label=f'Landau damping: $\\gamma_L/\\omega_{{pe}} = {gamma_Landau/omega_pe:.2e}$')

# Threshold
I_threshold_idx = np.argmin(np.abs(gamma_array - gamma_Landau))
I_threshold = I_array[I_threshold_idx]
ax.axvline(I_threshold / 1e15, color='g', linestyle=':',
           linewidth=2, label=f'Threshold: $I_{{th}} = {I_threshold/1e15:.2f}$ PW/cm²')

ax.set_xlabel('Laser Intensity (PW/cm²)', fontsize=13)
ax.set_ylabel('$\\gamma / \\omega_{pe}$', fontsize=13)
ax.set_title('Stimulated Raman Scattering Growth Rate', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('srs_threshold.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nSRS parameters:")
print(f"  Density: n = {n:.2e} m^-3 (n/n_c = {n/1.1e21:.2f})")
print(f"  Temperature: T_e = {T_e/1e3:.0f} keV")
print(f"  Threshold intensity: I_th = {I_threshold:.2e} W/m² = {I_threshold/1e15:.2f} PW/cm²")
```

## 요약

파동 가열과 불안정성은 플라즈마 물리학의 핵심입니다:

**핵융합에서 파동 가열**:
- **ECRH**: $\omega \approx n\omega_{ce}$, 140-170 GHz, 우수한 국소화, 전류 구동
- **ICRH**: $\omega \approx n\omega_{ci}$, 40-80 MHz, 이온 가열, 소수 방식
- **LHCD**: $\omega_{ci} \ll \omega \ll \omega_{ce}$, 3-8 GHz, 효율적인 전류 구동
- 핵융합 반응로를 위한 다중 방법의 시너지 사용이 최적

**속도 공간 불안정성**:
- **빔-플라즈마**: 냉각 배경에 냉각 빔, $\gamma \sim \omega_{pe}(n_b/n_0)^{1/3}$
- **Bump-on-tail**: 양의 $\partial f/\partial v$가 역 Landau 감쇠 구동, 준선형 고원
- **Weibel**: 온도 비등방성이 자기장 생성, $\gamma \sim \omega_{pe}\sqrt{\Delta T/T}$

**압력 구동 불안정성**:
- **Firehose**: $p_\parallel > p_\perp$, 기준 $\beta_\parallel - \beta_\perp < 1$, 자기장 선 구부림
- **Mirror**: $p_\perp > p_\parallel$, 기준 $\beta_\perp(p_\perp/p_\parallel - 1) < 1$, 자기 병 생성
- 무충돌 플라즈마 (태양풍, 자기권)에서 편재

**파라메트릭 불안정성**:
- **SRS**: EM $\to$ EM + Langmuir, 열전자 생성, 레이저 핵융합 문제
- **SBS**: EM $\to$ EM + 이온 음향, 후방산란, 에너지 손실
- 임계값은 펌프 강도, 감쇠율에 의존
- ICF에서 주요 과제, 대역폭, 평활화를 통한 완화

응용은 핵융합 에너지, 천체물리학, 우주 물리학, 고에너지 밀도 물리학에 걸쳐 있습니다. 불안정성 이해는 플라즈마 성능을 제어하고 최적화하는 데 필수적입니다.

## 연습 문제

### 문제 1: ECRH 시스템 설계
토카막은 축에서 $B_0 = 2.5$ T를 가지고 밀도 프로파일 $n(r) = n_0(1 - r^2/a^2)^2$를 가지며 $n_0 = 8 \times 10^{19}$ m$^{-3}$, $a = 0.5$ m입니다.

(a) 자기 축에서 전자 사이클로트론 주파수 $f_{ce}$를 계산하십시오.

(b) 2차 고조파 ECRH ($\omega = 2\omega_{ce}$)의 경우, 어떤 자이로트론 주파수가 필요합니까?

(c) 이 주파수에서 O-모드 차단 밀도를 계산하십시오. 파동이 코어에 도달할 수 있습니까?

(d) X-모드가 대신 사용되는 경우, 상부 하이브리드 공명 층이 어디에 위치합니까?

### 문제 2: Two-Stream 불안정성
$n_b = 10^{17}$ m$^{-3}$, $v_0 = 10^7$ m/s를 가진 전자 빔이 $n_0 = 10^{19}$ m$^{-3}$를 가진 플라즈마를 통해 전파합니다.

(a) 배경 플라즈마 주파수 $\omega_{p0}$를 계산하십시오.

(b) $\gamma \approx \omega_{p0}(n_b/n_0)^{1/3}$를 사용하여 성장률 $\gamma$를 추정하십시오.

(c) 불안정성이 공명하는 파수 $k$는 무엇입니까 (즉, $kv_0 \approx \omega_{p0}$)?

(d) 파동 진폭이 1000배 성장하는 데 몇 번의 $e$-폴딩 시간이 걸립니까? 빔이 $L/v_0 = 1$ μs에 플라즈마를 통과하는 경우, 이것이 상당한 성장에 충분합니까?

### 문제 3: 레이저 플라즈마에서 Weibel 불안정성
레이저 가열 플라즈마가 $T_\perp = 500$ eV (레이저에 의해 가열), $T_\parallel = 50$ eV (레이저 방향으로 냉각), $n = 10^{21}$ m$^{-3}$를 가집니다.

(a) 비등방성 파라미터 $(T_\perp - T_\parallel)/T_\parallel$를 계산하십시오.

(b) 최대 Weibel 성장률 $\gamma_{\text{max}}$를 추정하십시오.

(c) 생성된 자기장은 $B^2/(8\pi) \sim nk_B(T_\perp - T_\parallel)$로 스케일됩니다. Tesla 단위로 자기장 강도를 추정하십시오.

(d) 이 자기장을 전자 자이로반경 $\rho_L \sim 1/k_{\text{max}}$에 필요한 자기장과 비교하십시오, 여기서 $k_{\text{max}}$는 최대 성장의 파수입니다. 전자가 자체 생성 자기장에 의해 자화됩니까?

### 문제 4: 태양풍 비등방성
1 AU에서 태양풍 관측은 $\beta_\parallel = 0.8$, $\beta_\perp = 1.5$를 보여줍니다.

(a) 플라즈마가 firehose 불안정성에 안정한지 확인하십시오.

(b) 플라즈마가 mirror 불안정성에 안정한지 확인하십시오.

(c) 불안정한 경우, $v_A = 50$ km/s, $L = 10^6$ km에 대해 성장률을 추정하십시오.

(d) 관측된 비등방성이 준정상 상태이며 한계 안정성을 시사합니다. 플라즈마를 안정성 경계 근처에 유지하는 메커니즘을 제안하십시오.

### 문제 5: ICF에서 레이저-플라즈마 불안정성
강도 $I = 3 \times 10^{15}$ W/cm$^2$과 파장 $\lambda = 351$ nm를 가진 레이저가 $n = 0.1 n_c$ (여기서 $n_c$는 임계 밀도)와 $T_e = 3$ keV를 가진 플라즈마를 조명합니다.

(a) 이 파장에 대한 임계 밀도 $n_c$를 계산하십시오.

(b) 레이저 장에서 전자의 진동 속도 $v_{osc}$를 추정하십시오.

(c) $\gamma_{SRS} \approx (k_L v_{osc}/4)\sqrt{\omega_0/\omega_s}$를 사용하여 SRS 성장률을 계산하십시오, 여기서 $k_L \approx 2\omega_0/c$ (후방산란).

(d) Langmuir 파동에 대한 Landau 감쇠율과 비교하십시오. SRS가 임계값 이상입니까? SRS를 줄일 수 있는 전략은 무엇입니까?

---

**이전**: [11. Electromagnetic Waves](./11_Electromagnetic_Waves.md)
**다음**: [13. Two-Fluid Model](./13_Two_Fluid_Model.md)
