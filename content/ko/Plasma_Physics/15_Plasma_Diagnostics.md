# 15. 플라즈마 진단

## 학습 목표

- 밀도, 온도, 플라즈마 전위 측정을 위한 랭뮤어 프로브의 원리 이해
- 비교란 밀도 및 온도 진단을 위한 톰슨 산란(비간섭성 및 간섭성) 설명
- 선적분 및 국소 밀도 측정을 위한 간섭계와 반사계 적용
- 온도, 밀도, 유동 속도, 자기장 측정을 위한 분광학 사용
- 전류, 저장 에너지, 내부 자기장 측정을 위한 자기 진단 이해
- 산업용 응용을 위한 저온 플라즈마 진단 설명

## 1. 왜 플라즈마 진단인가?

### 1.1 플라즈마 측정의 도전 과제

플라즈마는 독특한 측정 도전 과제를 제시합니다:

1. **극한 환경**: 높은 온도(10⁶–10⁸ K), 낮은 밀도(10¹⁴–10²¹ m⁻³), 강한 자기장(1–10 T)

2. **교란에 대한 민감성**: 프로브를 삽입하면 플라즈마를 냉각시키거나 불순물을 유입하거나 평형을 교란할 수 있습니다

3. **직접 접근 불가**: 핵융합 장치에서 플라즈마는 자기장으로 구속되고 진공 용기로 둘러싸여 있습니다

4. **다중 스케일**: mm(난류)부터 미터(평형)까지 현상을 측정해야 합니다

5. **시간 변화**: 플라즈마는 나노초(파동)부터 초(방전 지속 시간)까지 시간 스케일로 진화합니다

**해결책**: 플라즈마 진단은 다음의 조합을 사용합니다:
- **비교란** 기법(광산란, 분광학, 간섭계)
- **최소 교란** 기법(작은 프로브, 가장자리 측정)
- **수동** 진단(자연 방출 관측)
- **능동** 진단(입자/파동 주입 및 반응 관측)

### 1.2 진단 목표

서로 다른 물리 질문은 서로 다른 진단을 요구합니다:

| 물리량 | 중요한 이유 | 진단 |
|----------|----------------|------------|
| $n_e$ | 플라즈마 밀도, 구속 | 간섭계, 톰슨 산란, 랭뮤어 프로브 |
| $T_e$, $T_i$ | 에너지 함량, 구속 | 톰슨 산란, 분광학, CXRS |
| $\mathbf{B}$ | 평형, 안정성 | 자기 코일, MSE, 제만 분리 |
| $I_p$ | 플라즈마 전류, 안정성 | 로고스키 코일 |
| $Z_{eff}$ | 불순물 함량, 복사 | 제동복사, 분광학 |
| $\mathbf{v}$ | 유동, 회전, 수송 | 도플러 분광학, CXRS |
| 요동 | 난류, 불안정성 | 프로브, 반사계, BES |

현대 핵융합 장치는 플라즈마의 완전한 그림을 구축하기 위해 **수십 개**의 진단을 동시에 사용합니다.

## 2. 랭뮤어 프로브

### 2.1 원리

**랭뮤어 프로브**(어빙 랭뮤어 이름, 1920년대)는 플라즈마에 삽입된 간단한 금속 전극입니다. 프로브 전압 $V$를 스윕하고 전류 $I$를 측정하여 $n_e$, $T_e$, 플라즈마 전위 $V_p$를 추론할 수 있습니다.

**장점**:
- 간단하고 저렴
- 직접적인 국소 측정
- 빠른 시간 응답(μs)

**단점**:
- 교란적(국소적으로 플라즈마 가열/냉각)
- 가장자리 플라즈마에서만 작동(중심부에서는 녹음)
- 주의 깊은 해석 필요(차폐 이론)

### 2.2 차폐 형성

프로브가 플라즈마에 삽입되면 **차폐**(디바이 차폐)가 주위에 형성됩니다. 전자는 이온보다 이동성이 높아 초기에 프로브로 흐르며 음전하를 띠게 합니다. 이것은 추가 전자를 밀어내고 이온을 끌어당겨 얇은 비중성 차폐를 가진 준중성 플라즈마를 만듭니다.

차폐 두께는 ~ 몇 디바이 길이입니다:
$$\lambda_D = \sqrt{\frac{\epsilon_0 k_B T_e}{n_e e^2}}$$

일반적으로 $\lambda_D \sim 10$–$100$ μm입니다.

### 2.3 I-V 특성

전류-전압 특성은 세 가지 영역을 가집니다:

```
           I
           ^
           |     전자 포화
         Ie|  ___________________
           | /
           |/
           |   전자 지연
           |       /|
     ------|------/-|----------> V
           |     /  |V_f  V_p
           |    /   |
           |   /    |
           |  /     이온 포화
     -Ii   | /
           |/______________
```

**1. 이온 포화 영역** ($V \ll V_p$):

프로브가 플라즈마에 대해 음으로 바이어스됩니다. 전자는 밀려나고 이온만 프로브에 도달합니다. 이온 전류는:

$$I_{\text{sat},i} = -e n_i u_B A_p$$

여기서 $A_p$는 프로브 면적이고 $u_B$는 **봄 속도**입니다:

$$u_B = \sqrt{\frac{k_B T_e}{m_i}}$$

봄 기준은 안정적인 차폐를 위해 이온이 적어도 이 속도로 차폐에 진입해야 한다고 명시합니다.

**2. 전자 지연 영역** ($V_f < V < V_p$):

전압이 증가함에 따라 일부 전자가 전위 장벽을 극복할 수 있습니다. 전자 전류는 볼츠만 분포를 따릅니다:

$$I_e = I_{e,\text{sat}} \exp\left( \frac{e(V - V_p)}{k_B T_e} \right)$$

총 전류는:
$$I = I_e + I_i \approx I_{e,\text{sat}} \exp\left( \frac{e(V - V_p)}{k_B T_e} \right) - I_{\text{sat},i}$$

**부유 전위** $V_f$에서 총 전류는 0입니다: $I = 0$.

**3. 전자 포화 영역** ($V > V_p$):

프로브는 도달하는 모든 전자를 수집합니다. 작은 프로브(전자 평균 자유 행로보다 훨씬 작은)의 경우, 전류는 다음에서 포화됩니다:

$$I_{e,\text{sat}} = \frac{1}{4} e n_e \bar{v}_e A_p$$

여기서 $\bar{v}_e = \sqrt{8 k_B T_e / (\pi m_e)}$는 평균 전자 속도입니다.

### 2.4 플라즈마 매개변수 추출

I-V 곡선으로부터:

**전자 온도** $T_e$:

전자 지연 영역에서 $\ln(I_e)$ 대 $V$를 플롯합니다. 기울기는:

$$\frac{d \ln I_e}{dV} = \frac{e}{k_B T_e}$$

따라서:
$$T_e = \frac{e}{k_B} \left( \frac{d \ln I_e}{dV} \right)^{-1}$$

**플라즈마 전위** $V_p$:

I-V 곡선의 "무릎"(기울기가 지수에서 평평하게 변하는 곳)이 플라즈마 전위입니다. 더 정확하게는, $V_p$는 이차 도함수 $d^2I/dV^2$가 최대값을 갖는 곳입니다.

**이온 밀도** $n_i$:

이온 포화 전류로부터:
$$n_i = \frac{I_{\text{sat},i}}{e u_B A_p} = \frac{I_{\text{sat},i}}{e A_p} \sqrt{\frac{m_i}{k_B T_e}}$$

**전자 밀도** $n_e$:

준중성으로부터 $n_e \approx n_i$.

### 2.5 복잡성

실제 랭뮤어 프로브는 여러 복잡성에 직면합니다:

1. **자기장**: 자화된 플라즈마에서 차폐는 $\mathbf{B}$를 따라 늘어납니다. 수집 면적은 방향에 따라 $A_\parallel \sim \pi r^2$ (교차장) 또는 $A_\perp \sim 2\pi r L$ (장을 따라)입니다.

2. **흐르는 플라즈마**: 플라즈마가 프로브에 대해 흐르면 I-V 곡선이 왜곡됩니다. **마하 프로브**(반대 방향을 향한 두 프로브)를 사용하여 유동을 측정합니다.

3. **이차 전자 방출**: 에너지가 높은 이온이 프로브에서 전자를 튕겨낼 수 있어 허위 전류를 추가합니다.

4. **차폐 내 충돌**: 차폐가 평균 자유 행로에 비해 두꺼우면 이온-중성 충돌이 발생하여 이온 전류가 감소합니다.

5. **RF 진동**: RF 가열 플라즈마에서 프로브 전위가 RF 주파수로 진동하여 해석이 복잡해집니다.

### 2.6 변형: 이중 및 삼중 프로브

**이중 프로브**: 벽에 대해서가 아니라 서로에 대해 바이어스된 두 개의 부유 프로브. 전원 공급 장치에서 큰 전류를 끌어내는 필요를 피합니다. 흐르는 플라즈마에 사용됩니다.

**삼중 프로브**: 서로 다른 바이어스의 세 프로브를 동시에 측정. 단일 시간 순간에서 $T_e$와 $n_e$를 측정할 수 있습니다(스윕 없음), 빠른 요동에 유용합니다.

## 3. 톰슨 산란

### 3.1 원리

**톰슨 산란**은 자유 전자에 의한 전자기파의 산란입니다. 고출력 레이저가 플라즈마를 통과하고 산란된 빛이 수집되고 분석됩니다.

**장점**:
- 비교란(광자가 플라즈마를 가열하지 않음)
- $n_e$와 $T_e$를 동시에 측정
- 높은 공간 분해능(mm)
- 절대 보정(참조 플라즈마 불필요)

**단점**:
- 비용이 비쌈(고출력 레이저와 민감한 검출기 필요)
- 복잡한 분석
- 제한된 시간 분해능(레이저 반복률)

### 3.2 산란 영역

산란은 매개변수에 따라 달라집니다:

$$\alpha = \frac{1}{k \lambda_D}$$

여기서 $k = |\mathbf{k}_s - \mathbf{k}_i|$는 산란 파동 벡터(산란광자와 입사광자 운동량의 차이)입니다.

**1. 비간섭성 산란** ($\alpha \ll 1$, 또는 $k \lambda_D \gg 1$):

개별 전자로부터의 산란. 전자 밀도 요동은 상관되지 않습니다. 산란 스펙트럼은 전자 속도 분포를 반영합니다.

**2. 간섭성 산란** ($\alpha \gg 1$, 또는 $k \lambda_D \ll 1$):

집단 플라즈마 진동(이온 음향파, 전자 플라즈마파)으로부터의 산란. 전자가 간섭적으로 산란하여 신호가 증폭됩니다.

### 3.3 비간섭성 톰슨 산란

**맥스웰** 전자 분포의 경우, 산란 스펙트럼은:

$$S(\omega) \propto \exp\left( -\frac{(\omega - \omega_0)^2}{2 k^2 v_{te}^2} \right)$$

여기서 $\omega_0$는 레이저 주파수이고 $v_{te} = \sqrt{k_B T_e / m_e}$는 전자 열속도입니다.

스펙트럼은 전자 운동에 의해 **도플러 확장**됩니다:

$$\Delta \omega = k v_{te} = k \sqrt{\frac{k_B T_e}{m_e}}$$

**$T_e$ 측정**:

산란 스펙트럼을 가우스에 피팅합니다. 폭이 $T_e$를 제공합니다:

$$T_e = \frac{m_e (\Delta \omega)^2}{k^2 k_B}$$

**$n_e$ 측정**:

총 산란 전력은:

$$P_s = P_i \sigma_T n_e \Delta V \Delta \Omega$$

여기서:
- $P_i$: 입사 레이저 전력
- $\sigma_T = 6.65 \times 10^{-29}$ m²: 톰슨 단면적
- $n_e$: 전자 밀도
- $\Delta V$: 산란 부피
- $\Delta \Omega$: 수집 광학계의 입체각

$P_s$를 측정하고 $P_i$, $\Delta V$, $\Delta \Omega$를 알면 $n_e$를 추론할 수 있습니다.

### 3.4 간섭성 톰슨 산란(집단 산란)

$k \lambda_D < 1$일 때, 산란은 **집단 요동**으로부터입니다. 산란 스펙트럼은 이온 음향파 주파수에서 피크를 갖습니다:

$$\omega_{ia} \approx k c_s = k \sqrt{\frac{k_B (T_e + T_i)}{m_i}}$$

스펙트럼은 다음을 보여줍니다:
- **중심 피크**: 전자 플라즈마파(랭뮤어 진동)
- **옆 피크**: 이온 음향파(청색 및 적색 편이)

옆 피크 위치에서 $c_s$ → $(T_e + T_i)$를 얻습니다.
피크 폭에서 란다우 감쇠 → $T_i$를 얻습니다.

이를 통해 비간섭성 산란에서 얻기 어려운 **이온 온도** $T_i$의 측정이 가능합니다.

### 3.5 핵융합 장치에서의 톰슨 산란

**ITER 톰슨 산란 시스템**:
- 레이저: Nd:YAG (1064 nm), 6 J/펄스, 20 Hz
- 빔을 따라 ~100개의 공간 지점
- 시간 분해능: 50 ms (50 ms당 한 샷)
- $n_e$ (10¹⁸–10²¹ m⁻³) 및 $T_e$ (0.1–50 keV) 측정

톰슨 산란은 토카막에서 중심부 $n_e$ 및 $T_e$ 프로파일의 **골드 스탠다드**입니다.

## 4. 간섭계와 반사계

### 4.1 마이크로파 간섭계

마이크로파 또는 레이저 빔이 플라즈마를 통과합니다. 위상 편이는 **선적분 밀도**에 비례합니다:

$$\Delta \phi = \frac{2\pi}{\lambda} \int n_e \, dl$$

(약한 파장 의존성을 무시)

더 정확하게는:
$$\Delta \phi = \frac{e^2}{2 \epsilon_0 m_e \omega c} \int n_e \, dl$$

여기서 $\omega$는 빔 주파수입니다.

**$\bar{n}_e L$ 측정**:

위상 편이에서 **선적분 밀도**를 얻습니다:

$$\int n_e \, dl = \frac{2 \epsilon_0 m_e \omega c}{e^2} \Delta \phi$$

밀도 프로파일 $n_e(r)$을 얻기 위해 서로 다른 충격 매개변수에서 **다중 코드**가 필요합니다. 그런 다음 **아벨 역변환**(원통 대칭 가정) 또는 **단층촬영 역변환**(2D)을 사용합니다.

**아벨 역변환**:

원통 대칭 플라즈마의 경우:

$$\int n_e \, dl = 2 \int_r^a n_e(r') \frac{r' \, dr'}{\sqrt{r'^2 - r^2}}$$

여기서 $r$은 충격 매개변수입니다. 역변환:

$$n_e(r) = -\frac{1}{\pi} \int_r^a \frac{d}{dr'} \left( \int n_e \, dl \right) \frac{dr'}{\sqrt{r'^2 - r^2}}$$

### 4.2 반사계

플라즈마를 통과하는 대신, 플라즈마 차단층에서 **반사**되는 마이크로파를 보냅니다:

$$\omega^2 = \omega_{pe}^2 = \frac{n_e e^2}{\epsilon_0 m_e}$$

차단 밀도는:
$$n_c = \frac{\epsilon_0 m_e \omega^2}{e^2} \approx 1.24 \times 10^{10} \, f^2 \quad (\text{m}^{-3}, \, f \text{ in GHz})$$

주파수를 스윕하여 서로 다른 밀도 층을 탐사합니다. 시간 지연은 차단층의 위치를 제공합니다.

**장점**:
- **국소** 측정(특정 밀도에서 반사)
- 빠름(요동, 난류 측정 가능)

**단점**:
- 복잡한 해석(위상 점프, 다중 반사)
- 가장자리/경사 영역으로 제한

**밀도 요동 측정**:

반사계는 **난류** 측정에 탁월합니다. 산란 신호는 밀도 요동으로 인해 변동합니다:

$$\frac{\delta n_e}{n_e} \sim \text{몇 퍼센트}$$

이것은 토카막의 가장자리 난류를 연구하는 데 사용됩니다.

### 4.3 패러데이 회전

자화된 플라즈마에서 선형 편광된 빛의 편광면이 회전합니다:

$$\theta = \frac{e^3}{2 \epsilon_0 m_e^2 \omega^2 c} \int n_e B_\parallel \, dl$$

이것은 $\int n_e B_\parallel \, dl$를 측정하여 자기장에 대한 정보를 제공합니다(간섭계에서 $n_e$를 알 경우).

## 5. 분광학

### 5.1 선 방출

원자와 이온은 전자가 에너지 준위 사이를 전이할 때 특징적인 스펙트럼선을 방출합니다. 이러한 선을 관측하여 다음을 할 수 있습니다:

- **종 식별**: 각 원소는 고유한 선을 가집니다 (예: H$_\alpha$ 656.3 nm, He II 468.6 nm)
- **온도 측정**: 선 강도 비율, 도플러 확장
- **밀도 측정**: 슈타르크 확장, 선 비율
- **유동 속도 측정**: 도플러 편이
- **자기장 측정**: 제만 분리

### 5.2 도플러 확장 → 온도

열운동은 **도플러 확장**을 야기합니다:

$$\Delta \lambda = \lambda_0 \sqrt{\frac{2 k_B T}{m c^2}}$$

여기서 $m$은 이온 질량입니다.

**$T_i$ 측정**:

스펙트럼선을 가우스에 피팅:

$$I(\lambda) = I_0 \exp\left[ -\frac{(\lambda - \lambda_0)^2}{2 (\Delta \lambda)^2} \right]$$

$\Delta \lambda$에서 $T_i$를 추론:

$$T_i = \frac{m c^2}{2 k_B} \left( \frac{\Delta \lambda}{\lambda_0} \right)^2$$

**예**: $T_i = 1$ keV, $\lambda_0 = 529$ nm에서 C⁶⁺ (완전히 벗겨진 탄소)의 경우:

$$\Delta \lambda = 529 \times 10^{-9} \sqrt{\frac{2 \times 1.6 \times 10^{-16}}{12 \times 1.67 \times 10^{-27} \times (3 \times 10^8)^2}} \approx 0.01 \text{ nm}$$

이를 위해서는 고분해능 분광기($R = \lambda / \Delta \lambda \sim 50{,}000$)가 필요합니다.

### 5.3 슈타르크 확장 → 밀도

**슈타르크 확장**은 인근 이온과 전자의 전기장이 에너지 준위를 이동시킬 때 발생합니다. 선 폭은 $n_e^{2/3}$에 비례합니다:

$$\Delta \lambda_{Stark} \propto n_e^{2/3}$$

수소 발머선(H$_\alpha$, H$_\beta$)의 경우 경험적 공식이 존재합니다:

$$\Delta \lambda_{H\alpha} (\text{nm}) \approx 4 \times 10^{-16} n_e^{2/3}$$

**$n_e$ 측정**:

H$_\alpha$ 선 폭을 관측합니다(도플러 및 기기 확장을 뺀 후):

$$n_e = \left( \frac{\Delta \lambda_{H\alpha}}{4 \times 10^{-16}} \right)^{3/2}$$

이것은 가장자리 플라즈마와 저온 플라즈마에서 일반적으로 사용됩니다.

### 5.4 도플러 편이 → 유동 속도

움직이는 플라즈마는 스펙트럼선을 편이시킵니다:

$$\Delta \lambda = \lambda_0 \frac{v_{\parallel}}{c}$$

여기서 $v_\parallel$는 시선 방향을 따른 속도 성분입니다.

**$v$ 측정**:

$$v_{\parallel} = c \frac{\Delta \lambda}{\lambda_0}$$

**예**: 토카막의 토로이달 회전의 경우, C⁶⁺ 방출을 관측합니다. $\lambda_0 = 529$ nm에서 $\Delta \lambda = 0.05$ nm이면:

$$v_{\parallel} = 3 \times 10^8 \times \frac{0.05 \times 10^{-9}}{529 \times 10^{-9}} \approx 28 \text{ km/s}$$

일반적인 토카막 회전 속도는 10–100 km/s입니다.

### 5.5 전하 교환 재결합 분광학(CXRS)

중심부 플라즈마에서 **이온 온도**와 **유동 속도**를 측정하는 뛰어난 기법:

1. **중성 원자** 빔(보통 중수소)을 플라즈마에 주입합니다.
2. 플라즈마의 빠른 이온이 중성 입자와 **전하 교환**을 합니다:
   $$\text{D}^+ + \text{D}^0 \to \text{D}^0 + \text{D}^+$$
   또는 불순물(예: 탄소)의 경우:
   $$\text{C}^{6+} + \text{D}^0 \to \text{C}^{5+} + \text{D}^+$$
3. 새로 형성된 $\text{C}^{5+}$는 들뜬 상태에 있고 빛을 방출합니다(예: 529 nm).
4. 이 빛은 **벌크 이온**의 도플러 편이와 확장을 가지며 $T_i$와 $v_i$를 제공합니다.

**장점**:
- 중심부 $T_i$와 $v_i$ 측정(가장자리 분광학으로는 불가능)
- 높은 공간 분해능(빔 경로를 따라)

**단점**:
- 중성 빔 주입 필요(플라즈마 교란)
- 복잡한 보정

CXRS는 토카막의 **회전 측정**에 필수적이며, 이는 MHD 안정성과 난류에 영향을 미칩니다.

### 5.6 제만 분리 → 자기장

자기장에서 스펙트럼선은 **제만 효과**로 인해 여러 성분으로 분리됩니다:

$$\Delta E = \mu_B g_J m_J B$$

여기서 $\mu_B$는 보어 자기자, $g_J$는 란데 g-인자, $m_J$는 자기 양자수입니다.

파장 분리는:

$$\Delta \lambda = \lambda_0^2 \frac{e B}{4\pi m_e c^2}$$

**예**: $B = 1$ T에서 H$_\alpha$ ($\lambda_0 = 656.3$ nm)의 경우:

$$\Delta \lambda \approx (656.3 \times 10^{-9})^2 \times \frac{1.6 \times 10^{-19} \times 1}{4\pi \times 9.11 \times 10^{-31} \times (3 \times 10^8)^2} \approx 0.014 \text{ nm}$$

이것은 고분해능 분광학으로 검출할 수 있습니다.

**운동 슈타르크 효과(MSE)**:

토카막에서 중성 빔 원자는 로렌츠 변환된 전기장을 봅니다:

$$\mathbf{E}' = -\mathbf{v}_{beam} \times \mathbf{B}$$

이것은 슈타르크 분리를 야기하며, 그 편광은 $\mathbf{B}$의 방향에 따라 달라집니다. 편광각을 측정하여 자기장의 **피치각**을 얻습니다:

$$\tan \theta = \frac{B_\theta}{B_\phi}$$

이것은 암페어 법칙을 통해 **전류 밀도 프로파일**을 제공합니다:

$$J_\phi = \frac{1}{\mu_0} \frac{\partial B_\theta}{\partial r}$$

MSE는 안전 인자 프로파일 $q(r)$ 측정에 중요합니다.

## 6. 자기 진단

### 6.1 로고스키 코일

**로고스키 코일**은 플라즈마 주위에 감긴 토로이달 코일입니다. 암페어 법칙으로부터:

$$\oint \mathbf{B} \cdot d\mathbf{l} = \mu_0 I_{enclosed}$$

코일은 플라즈마 전류의 시간 도함수를 측정합니다:

$$V_{coil} = -\frac{d\Phi}{dt} = -\mu_0 A N \frac{dI_p}{dt}$$

여기서 $N$은 권선 수, $A$는 단면적입니다.

시간에 대해 적분:

$$I_p(t) = -\frac{1}{\mu_0 A N} \int V_{coil} \, dt$$

**정확도**: 로고스키 코일은 **총 플라즈마 전류**를 제공하며, 평형 재구성과 방전 제어에 필수적입니다.

### 6.2 자기 픽업 코일

작은 코일(플럭스 루프)은 국소 자기장을 측정합니다:

$$V_{coil} = -\frac{d\Phi}{dt} = -A \frac{dB}{dt}$$

서로 다른 위치에 많은 코일을 배치하여 2D 폴로이달 자기장 $B_\theta(r, \theta)$를 재구성합니다.

**그라드-샤프라노프 평형 재구성**:

자기 측정과 압력(톰슨 산란에서)을 결합하여 그라드-샤프라노프 방정식을 풉니다:

$$\Delta^* \psi = -\mu_0 r^2 \frac{dp}{d\psi} - F \frac{dF}{d\psi}$$

이것은 자기 플럭스 표면과 $q(r)$을 제공합니다.

### 6.3 반자성 루프

폴로이달 루프는 **반자성 플럭스**를 측정합니다:

$$\Phi_{dia} = \int \mathbf{B}_\theta \cdot d\mathbf{A}$$

반자성 효과는 플라즈마 압력이 존재할 때 $B_\theta$를 감소시킵니다:

$$B_\theta^{vac} - B_\theta^{plasma} \propto p$$

저장 에너지는:

$$W = \frac{3}{2} \int p \, dV \propto \Phi_{dia}$$

이것은 **저장 에너지**의 빠르고 간단한 측정을 제공하며, 핵융합 성능에 중요합니다($Q = P_{fusion} / P_{input} \propto W$).

### 6.4 운동 슈타르크 효과(MSE)

섹션 5.6에서 논의한 바와 같이, MSE는 중성 빔 방출의 슈타르크 분리로부터 **내부 자기장**을 측정합니다. 이것은 플라즈마 내부의 $B(r)$을 측정하는 몇 안 되는 방법 중 하나입니다.

## 7. 저온 및 산업용 플라즈마

### 7.1 글로우 방전

**글로우 방전**은 특징적인 빛(가시광선 방출)을 내는 저압 가스 방전입니다. 다음에 사용됩니다:
- 플라즈마 처리(식각, 증착)
- 조명(네온 사인, 형광등)
- 디스플레이(플라즈마 TV, 현재 사용하지 않음)

**파셴의 법칙**: 파괴 전압 $V_b$(방전을 유지하는 최소 전압)는 곱 $p d$(압력 × 간격 거리)에 따라 달라집니다:

$$V_b = \frac{B p d}{\ln(A p d) - \ln(\ln(1 + 1/\gamma))}$$

여기서 $A$와 $B$는 가스 의존 상수이고, $\gamma$는 이차 방출 계수입니다.

특정 $p d$에서 **최소** 파괴 전압이 있습니다(파셴 최소값).

**예**: 공기의 경우 $p d \approx 0.5$ Torr·cm에서 $V_b \approx 300$ V입니다.

### 7.2 RF 플라즈마

**용량 결합 플라즈마(CCP)**:

RF(일반적으로 13.56 MHz)로 구동되는 두 개의 평행 전극. 이온은 시간 평균 전위에 반응하고 전자는 RF 주파수로 진동합니다.

**유도 결합 플라즈마(ICP)**:

외부 코일의 RF 전류가 시간 변화 자기장을 유도하고, 이것이 방위각 전기장을 유도 → 플라즈마 전류를 구동 → 전자를 가열합니다.

ICP는 CCP보다 높은 밀도($10^{17}$–$10^{18}$ m⁻³)를 달성합니다.

### 7.3 플라즈마 처리 진단

**광학 방출 분광학(OES)**:

방출선을 모니터링하여 다음을 추적:
- **식각 종점**: 기판이 식각되면 하부 층의 방출이 나타남
- **가스 조성**: 불순물 검출, 전구체 분해 모니터링

**랭뮤어 프로브**:

챔버에서 $n_e$, $T_e$ 측정. 스퍼터링을 피하기 위해 **부유 전위**에 있어야 합니다.

**사중극자 질량 분석기(QMS)**:

중성 및 이온 종을 샘플링하고 화학 반응을 식별합니다.

**레이저 유도 형광(LIF)**:

조정 가능한 레이저로 중성 종을 여기하고 형광을 측정 → 특정 종의 밀도와 속도를 얻습니다.

### 7.4 응용

- **반도체 제조**: 플라즈마 식각(이방성, 선택적), PECVD(플라즈마 강화 화학 기상 증착)
- **표면 처리**: 세정, 활성화, 코팅
- **살균**: 저온 플라즈마가 열 없이 박테리아를 죽임
- **조명**: 에너지 효율적(형광등, LED 플라즈마)
- **재료 처리**: 질화, 침탄, 경화

## 8. Python 코드 예제

### 8.1 랭뮤어 프로브 I-V 곡선 피팅

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Generate synthetic I-V data
def langmuir_current(V, n_e, T_e, V_p, V_f, A_p):
    """
    Langmuir probe current model.
    """
    e = 1.6e-19
    k_B = 1.38e-23
    m_e = 9.11e-31
    m_i = 1.67e-27  # proton

    # Bohm velocity
    u_B = np.sqrt(k_B * T_e / m_i)

    # Ion saturation current
    I_sat_i = e * n_e * u_B * A_p

    # Electron current (Boltzmann)
    v_bar_e = np.sqrt(8 * k_B * T_e / (np.pi * m_e))
    I_sat_e = 0.25 * e * n_e * v_bar_e * A_p

    # Total current
    I = np.where(V < V_p,
                 I_sat_e * np.exp(e * (V - V_p) / (k_B * T_e)) - I_sat_i,
                 I_sat_e - I_sat_i)

    return I

# True parameters
n_e_true = 1e16  # m^-3
T_e_true = 3.0   # eV
V_p_true = 10.0  # V
V_f_true = 5.0   # V (not used directly, but implicit)
A_p = 1e-4       # m^2 (1 cm^2)

# Generate data
V = np.linspace(-20, 20, 100)
I_true = langmuir_current(V, n_e_true, T_e_true * 1.6e-19, V_p_true, V_f_true, A_p)
I_noisy = I_true + np.random.normal(0, 0.1e-6, len(V))

# Fit in electron retardation region
mask = (V > 0) & (V < V_p_true)
V_fit = V[mask]
I_fit = I_noisy[mask]

# Take log of electron current (approximate, ignoring ion current)
I_e_approx = I_fit + 1e-6  # shift to avoid log(negative)
ln_I = np.log(np.abs(I_e_approx))

# Linear fit: ln(I) = (e/k_B T_e) * V + const
p = np.polyfit(V_fit, ln_I, 1)
slope = p[0]

e = 1.6e-19
k_B = 1.38e-23
T_e_fit = e / (k_B * slope)

print("Langmuir Probe Analysis:")
print(f"  True T_e = {T_e_true:.2f} eV")
print(f"  Fitted T_e = {T_e_fit:.2f} eV")
print()

# Find V_p (knee of curve)
dI_dV = np.gradient(I_noisy, V)
d2I_dV2 = np.gradient(dI_dV, V)
idx_Vp = np.argmax(d2I_dV2)
V_p_fit = V[idx_Vp]

print(f"  True V_p = {V_p_true:.2f} V")
print(f"  Fitted V_p = {V_p_fit:.2f} V")
print()

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# I-V curve
axes[0].plot(V, I_true * 1e6, 'b-', linewidth=2, label='True')
axes[0].plot(V, I_noisy * 1e6, 'r.', markersize=4, label='Noisy data')
axes[0].axhline(0, color='k', linestyle='--', alpha=0.5)
axes[0].axvline(V_p_true, color='g', linestyle='--', linewidth=2, label=f'V_p = {V_p_true} V')
axes[0].set_xlabel('Probe voltage V (V)', fontsize=12)
axes[0].set_ylabel('Current I (μA)', fontsize=12)
axes[0].set_title('Langmuir Probe I-V Characteristic', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# ln(I) vs V (electron retardation region)
axes[1].plot(V_fit, ln_I, 'bo', markersize=5, label='Data')
axes[1].plot(V_fit, np.polyval(p, V_fit), 'r-', linewidth=2,
             label=f'Fit: slope = {slope:.2f} V⁻¹\nT_e = {T_e_fit:.2f} eV')
axes[1].set_xlabel('Probe voltage V (V)', fontsize=12)
axes[1].set_ylabel('ln(I)', fontsize=12)
axes[1].set_title('Electron Retardation Region (Temperature Fit)', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('langmuir_probe.png', dpi=150)
plt.show()
```

### 8.2 간섭계: 위상 편이로부터 밀도 프로파일

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

def abel_inversion(r_impact, line_integral):
    """
    Abel inversion to get radial profile from line-integrated data.

    Assumes cylindrical symmetry: n(r).
    Given: ∫ n(r) dl for different impact parameters r.
    """
    # Sort by impact parameter
    idx = np.argsort(r_impact)
    r = r_impact[idx]
    L = line_integral[idx]

    # Compute derivative dL/dr
    dL_dr = np.gradient(L, r)

    # Abel inversion: n(r) = -(1/π) ∫_r^a (dL/dr') / sqrt(r'^2 - r^2) dr'
    n_r = np.zeros_like(r)

    for i in range(len(r)):
        ri = r[i]
        # Integrate from ri to r_max
        integrand = dL_dr[i:] / np.sqrt(r[i:]**2 - ri**2 + 1e-10)  # avoid division by zero
        n_r[i] = -1/np.pi * np.trapz(integrand, r[i:])

    return r, n_r

# Synthetic density profile (parabolic)
a = 0.5  # plasma radius (m)
n_0 = 1e20  # peak density (m^-3)

r_true = np.linspace(0, a, 100)
n_true = n_0 * (1 - (r_true / a)**2)**2

# Compute line-integrated density for different chords
N_chords = 20
r_impact = np.linspace(0, 0.9*a, N_chords)
line_integral = np.zeros(N_chords)

for i, r_imp in enumerate(r_impact):
    # Integrate along the chord
    # For cylindrical symmetry: ∫ n dl = 2 ∫_r_imp^a n(r) r dr / sqrt(r^2 - r_imp^2)
    r_chord = np.linspace(r_imp + 1e-6, a, 200)
    integrand = n_0 * (1 - (r_chord/a)**2)**2 * r_chord / np.sqrt(r_chord**2 - r_imp**2)
    line_integral[i] = 2 * np.trapz(integrand, r_chord)

# Add noise
line_integral_noisy = line_integral + np.random.normal(0, 0.02 * n_0 * a, N_chords)

# Abel inversion
r_inverted, n_inverted = abel_inversion(r_impact, line_integral_noisy)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Line-integrated density
axes[0].plot(r_impact * 100, line_integral / (n_0 * a), 'bo', markersize=7, label='True')
axes[0].plot(r_impact * 100, line_integral_noisy / (n_0 * a), 'rx', markersize=7, label='Noisy')
axes[0].set_xlabel('Impact parameter r (cm)', fontsize=12)
axes[0].set_ylabel('Line-integrated density / (n₀ a)', fontsize=12)
axes[0].set_title('Interferometry Measurements', fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(alpha=0.3)

# Density profile
axes[1].plot(r_true * 100, n_true / n_0, 'b-', linewidth=2, label='True profile')
axes[1].plot(r_inverted * 100, np.abs(n_inverted) / n_0, 'r--', linewidth=2, label='Abel inverted')
axes[1].set_xlabel('Radius r (cm)', fontsize=12)
axes[1].set_ylabel('Density n / n₀', fontsize=12)
axes[1].set_title('Reconstructed Density Profile', fontsize=13)
axes[1].legend(fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('interferometry_inversion.png', dpi=150)
plt.show()

print("Interferometry and Abel Inversion:")
print(f"  Number of chords: {N_chords}")
print(f"  Peak density (true): {n_0:.2e} m⁻³")
print(f"  Peak density (inverted): {np.max(np.abs(n_inverted)):.2e} m⁻³")
print(f"  Relative error: {100 * (np.max(np.abs(n_inverted)) - n_0) / n_0:.1f}%")
```

### 8.3 도플러 확장: 스펙트럼선 피팅 → 온도

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def gaussian(x, A, mu, sigma):
    """Gaussian function."""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def doppler_width(T, m, lambda_0):
    """
    Doppler width (FWHM) of spectral line.

    T: temperature (eV)
    m: ion mass (kg)
    lambda_0: rest wavelength (m)
    """
    k_B = 1.38e-23
    c = 3e8
    e = 1.6e-19

    T_J = T * e
    sigma_v = np.sqrt(k_B * T_J / m)  # velocity dispersion
    sigma_lambda = lambda_0 * sigma_v / c  # wavelength dispersion

    return sigma_lambda

# Simulate spectral line (C^6+ at 529 nm)
lambda_0 = 529e-9  # m
m_C = 12 * 1.67e-27  # kg
T_true = 1000  # eV (1 keV)

sigma_true = doppler_width(T_true, m_C, lambda_0)

# Wavelength grid
lambda_grid = np.linspace(lambda_0 - 3*sigma_true, lambda_0 + 3*sigma_true, 200)

# True spectrum
I_true = gaussian(lambda_grid, 1.0, lambda_0, sigma_true)

# Add noise
I_noisy = I_true + np.random.normal(0, 0.02, len(lambda_grid))

# Fit Gaussian
p0 = [1.0, lambda_0, sigma_true * 1.2]  # initial guess
popt, pcov = curve_fit(gaussian, lambda_grid, I_noisy, p0=p0)

A_fit, mu_fit, sigma_fit = popt

# Infer temperature
T_fit = (m_C * c**2 / k_B) * (sigma_fit / lambda_0)**2 / 1.6e-19  # eV

print("Doppler Broadening Analysis:")
print(f"  True temperature: {T_true} eV")
print(f"  Fitted temperature: {T_fit:.1f} eV")
print(f"  True sigma: {sigma_true * 1e12:.3f} pm")
print(f"  Fitted sigma: {sigma_fit * 1e12:.3f} pm")
print()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(lambda_grid * 1e9, I_true, 'b-', linewidth=2, label='True (T = 1000 eV)')
plt.plot(lambda_grid * 1e9, I_noisy, 'r.', markersize=5, label='Noisy data')
plt.plot(lambda_grid * 1e9, gaussian(lambda_grid, *popt), 'g--', linewidth=2,
         label=f'Gaussian fit (T = {T_fit:.0f} eV)')

plt.xlabel('Wavelength λ (nm)', fontsize=12)
plt.ylabel('Intensity (normalized)', fontsize=12)
plt.title('Doppler Broadening of C⁶⁺ Line (529 nm)', fontsize=13)
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('doppler_broadening.png', dpi=150)
plt.show()
```

## 요약

이 레슨에서 주요 플라즈마 진단 기법을 조사했습니다:

1. **랭뮤어 프로브**: I-V 특성을 통한 $n_e$, $T_e$, $V_p$의 간단한 국소 측정. 가장자리 플라즈마에 유용하지만 교란적입니다.

2. **톰슨 산란**: 중심부 $n_e$ 및 $T_e$의 골드 스탠다드. 비간섭성 산란은 전자 분포를 제공하고, 간섭성 산란은 집단 모드를 통해 이온 온도를 제공합니다.

3. **간섭계와 반사계**: 밀도를 위한 마이크로파 진단. 간섭계는 선적분 밀도를 제공하고, 반사계는 국소 밀도와 요동을 제공합니다.

4. **분광학**: 선 방출로부터 풍부한 정보. 도플러 확장 → 온도, 슈타르크 확장 → 밀도, 도플러 편이 → 유동, 제만 분리 → 자기장.

5. **CXRS**: 전하 교환 재결합 분광학은 중성 빔의 불순물 방출을 관측하여 중심부 $T_i$와 회전을 측정합니다.

6. **자기 진단**: 로고스키 코일(총 전류), 픽업 코일(폴로이달 자기장), 반자성 루프(저장 에너지), MSE(내부 자기장).

7. **저온 플라즈마**: 산업용 플라즈마 처리(식각, 증착, 살균)를 위한 랭뮤어 프로브, OES, 질량 분석법.

현대 핵융합 실험은 **통합 진단**을 사용합니다: 여러 기법을 결합하여 플라즈마 상태의 완전한 그림을 구축합니다. 데이터 융합과 베이지안 추론은 서로 다른 측정을 결합하는 새로운 도구입니다.

## 연습 문제

### 문제 1: 가장자리 플라즈마의 랭뮤어 프로브
토카막 가장자리의 랭뮤어 프로브가 다음을 측정합니다:
- 이온 포화 전류: $I_{sat,i} = -5$ mA
- 프로브 면적: $A_p = 2$ mm²
- 전자 온도(기울기에서): $T_e = 20$ eV

계산하세요:
(a) 봄 속도 $u_B$.
(b) 이온 밀도 $n_i$.
(c) 부유 전위 $V_f$ ($T_e = T_i$ 및 단일 전하 이온 가정).

### 문제 2: 톰슨 산란 스펙트럼
톰슨 산란 시스템이 90° 산란각에서 Nd:YAG 레이저($\lambda = 1064$ nm)를 사용합니다. 산란 스펙트럼은 $\Delta \lambda = 2$ nm의 가우스 폭을 보입니다.

(a) 전자 온도 $T_e$를 계산하세요.
(b) 산란 전력이 입사 전력 $P_i = 1$ J/펄스, 산란 부피 $\Delta V = 1$ mm³, 수집 입체각 $\Delta \Omega = 0.01$ sr에 대해 $P_s = 10^{-9}$ W이면, 전자 밀도 $n_e$를 추정하세요.

### 문제 3: 간섭계 아벨 역변환
간섭계가 원통형 플라즈마(반지름 $a = 10$ cm)를 통과하는 5개 코드를 따라 선적분 밀도를 측정합니다:

| 충격 매개변수 $r$ (cm) | 선 적분 $\int n_e \, dl$ (10¹⁸ m⁻²) |
|---------------------------|--------------------------------------------|
| 0                         | 10.0                                       |
| 3                         | 9.5                                        |
| 5                         | 8.0                                        |
| 7                         | 5.0                                        |
| 9                         | 2.0                                        |

(a) 포물선 프로파일 $n_e(r) = n_0 (1 - r^2/a^2)^\alpha$를 가정하고, 데이터에 피팅하여 $n_0$와 $\alpha$를 추정하세요.
(b) 아벨 역변환(수치적 또는 해석적)을 사용하여 $r = 0, 5, 10$ cm에서 $n_e(r)$을 재구성하세요.

### 문제 4: 도플러 분광학
토카막 플라즈마에서 $\lambda_0 = 529.0$ nm의 C⁶⁺ 선이 관측됩니다. 측정된 스펙트럼은:
- 피크 파장: $\lambda_{peak} = 529.05$ nm
- FWHM: $\Delta \lambda_{FWHM} = 0.02$ nm

(a) 토로이달 회전 속도를 계산하세요(시선이 토로이달 방향에 수직이라고 가정).
(b) 도플러 확장으로부터 이온 온도 $T_i$를 계산하세요(탄소 질량 $m = 12$ amu 가정).
(c) $\Delta \lambda_{Stark} = 0.005$ nm의 슈타르크 확장도 있다면, 본질적인 도플러 폭은 얼마입니까?

### 문제 5: 자기 진단
토카막에 플라즈마 전류를 측정하는 로고스키 코일이 있습니다. 코일은 $N = 1000$ 권선, 단면적 $A = 1$ cm²를 가지며, 램프업 단계(1 s 지속) 동안 유도 전압은 $V_{coil} = -50$ mV입니다.

(a) 플라즈마 전류의 변화율 $dI_p / dt$를 계산하세요.
(b) 전류가 0에서 시작한다면, 1 s 후 최종 플라즈마 전류 $I_p$는 얼마입니까($dI_p/dt$가 일정하다고 가정)?
(c) 반자성 루프가 저장 에너지 $W = 10$ MJ를 측정합니다. 플라즈마 부피 $V = 100$ m³에 대해 평균 압력 $\langle p \rangle$을 추정하세요.

---

**이전**: [운동론에서 MHD로](./14_From_Kinetic_to_MHD.md) | **다음**: [프로젝트](./16_Projects.md)
