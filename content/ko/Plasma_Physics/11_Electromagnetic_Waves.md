# 11. 플라즈마에서의 전자기파

## 학습 목표

- 무자화 플라즈마에서 전자기파 전파와 플라즈마 차단을 이해
- 자화 플라즈마를 위한 Stix cold plasma 유전 텐서 마스터
- R-파, L-파, O-모드, X-모드의 분산 관계 유도
- 휘슬러파 분산 및 응용 분석
- CMA (Clemmow-Mullaly-Allis) 다이어그램 구성 및 해석
- Faraday 회전을 적용하여 플라즈마의 자기장 측정

## 소개

정전파(electric field $\mathbf{E}$ 섭동만 포함)와 달리, **전자기파 (EM waves)**는 전기장과 자기장 성분을 모두 가집니다:

$$\mathbf{E}_1 \neq -\nabla\phi, \quad \mathbf{B}_1 \neq 0$$

플라즈마에서 EM 파는 풍부한 물리학을 보여줍니다:
- **차단 (Cutoffs)**: 파동이 전파할 수 없는 주파수 이하 (감쇠)
- **공명 (Resonances)**: 파동 특성이 발산하는 주파수
- **편광 (Polarization)**: 파동 전기장이 선형, 원형 또는 타원형일 수 있음
- **모드 변환 (Mode conversion)**: 한 파동 유형이 다른 유형으로 변환

이러한 파동은 다음에 중요합니다:
- **플라즈마 가열**: ECRH (전자 사이클로트론 공명 가열), ICRH (이온 사이클로트론)
- **진단**: 간섭계, 반사계, 편광계
- **통신**: 전리층 전파, 휘슬러파
- **천체물리학**: 펄서 방출, 태양 전파 폭발

가장 간단한 경우(무자화 플라즈마)부터 시작하여 **Stix 형식론**을 사용하여 완전한 자화 플라즈마 이론을 구축합니다.

## 1. 무자화 플라즈마에서의 EM파

### 1.1 Maxwell 방정식과 파동 방정식

$\propto e^{i(\mathbf{k}\cdot\mathbf{x} - \omega t)}$ 필드에 대한 Maxwell 방정식에서 시작:

$$\mathbf{k} \times \mathbf{E}_1 = \omega \mathbf{B}_1$$

$$\mathbf{k} \times \mathbf{B}_1 = -\frac{\omega}{c^2}(\mathbf{E}_1 + \mathbf{E}_{\text{plasma}})$$

여기서 $\mathbf{E}_{\text{plasma}}$는 플라즈마 전류로부터의 전기장입니다.

냉각 플라즈마의 경우, 플라즈마 전류는:
$$\mathbf{j}_1 = -n_0 e \mathbf{v}_1 = -i\frac{n_0 e^2}{m_e \omega}\mathbf{E}_1 = -i\epsilon_0\omega_{pe}^2/\omega \cdot \mathbf{E}_1$$

이것은 다음을 제공합니다:
$$\mathbf{k} \times \mathbf{B}_1 = -\frac{\omega}{c^2}\left(1 - \frac{\omega_{pe}^2}{\omega^2}\right)\mathbf{E}_1$$

첫 번째 방정식의 $\mathbf{k} \times$를 취하면:
$$\mathbf{k} \times (\mathbf{k} \times \mathbf{E}_1) = \omega \mathbf{k} \times \mathbf{B}_1$$

벡터 항등식 $\mathbf{k} \times (\mathbf{k} \times \mathbf{E}_1) = \mathbf{k}(\mathbf{k}\cdot\mathbf{E}_1) - k^2\mathbf{E}_1$ 사용:

**횡파** ($\mathbf{k}\cdot\mathbf{E}_1 = 0$)의 경우:

$$-k^2 \mathbf{E}_1 = -\frac{\omega^2}{c^2}\left(1 - \frac{\omega_{pe}^2}{\omega^2}\right)\mathbf{E}_1$$

이것은 **분산 관계**를 제공합니다:

$$\boxed{\omega^2 = \omega_{pe}^2 + k^2 c^2}$$

또는 동등하게:

$$\boxed{k^2 c^2 = \omega^2 - \omega_{pe}^2}$$

### 1.2 차단과 굴절률

**차단**: 주파수 $\omega = \omega_{pe}$는 **차단**입니다. $\omega < \omega_{pe}$의 경우:
$$k^2 < 0 \Rightarrow k = i\kappa$$

파동은 **감쇠** (공간에서 지수적으로 감소)가 됩니다:
$$\mathbf{E}_1 \propto e^{-\kappa x} e^{-i\omega t}$$

**침투 깊이** (표피 깊이)는:
$$\delta = \frac{1}{\kappa} = \frac{c}{\sqrt{\omega_{pe}^2 - \omega^2}}$$

$\omega \ll \omega_{pe}$의 경우:
$$\delta \approx \frac{c}{\omega_{pe}}$$

이것이 저주파 전파가 전리층을 통과할 수 없는 이유입니다 ($\omega_{pe} \sim 2\pi \times 10$ MHz).

**굴절률**: $n = kc/\omega$를 정의:

$$\boxed{n^2 = 1 - \frac{\omega_{pe}^2}{\omega^2}}$$

$\omega > \omega_{pe}$의 경우: $n < 1$ (위상 속도 $v_\phi = c/n > c$!)

이것은 **정보**가 군속도로 이동하기 때문에 상대성을 위반하지 않습니다:
$$v_g = \frac{d\omega}{dk} = \frac{k c^2}{\omega} = c \sqrt{1 - \frac{\omega_{pe}^2}{\omega^2}} < c$$

### 1.3 물리적 그림: 진동하는 전자

```
EM파가 플라즈마에 진입:
E-장 → 전자 가속 → 진동 전류
전류 → 2차 E-장 생성 (위상 차이)
순 효과: 파동 전파 수정

낮은 ω (ω < ωpe):  전자가 E-장을 상쇄할 만큼 빠르게 응답
                   → 파동이 전파할 수 없음 (반사)

높은 ω (ω > ωpe): 전자가 충분히 빠르게 응답할 수 없음
                   → 파동 전파 (수정된 c로)
```

### 1.4 전리층 응용

전리층은 높이에 따라 증가하는 밀도 프로파일 $n(h)$를 가집니다:

```
높이 (km)     밀도 (m^-3)      f_pe (MHz)
   100             10^11               0.3
   200             10^12               3
   300             10^13              10
```

1 MHz의 AM 라디오파 ($< 10$ MHz)는 $f = f_{pe}(h)$인 높이에서 반사됩니다. 이것은 수평선 너머 통신을 가능하게 합니다.

100 MHz의 FM 라디오 ($> f_{pe,\max}$)는 전리층을 통과합니다 (가시선만).

## 2. 냉각 자화 플라즈마: Stix 형식론

### 2.1 유전 텐서

$\mathbf{B}_0 = B_0\hat{z}$를 가진 자화 플라즈마에서, 플라즈마 응답은 **비등방성**입니다. 변위는:

$$\mathbf{D} = \epsilon_0 \overleftrightarrow{K} \cdot \mathbf{E}$$

여기서 $\overleftrightarrow{K}$는 **유전 텐서**입니다.

냉각 플라즈마의 경우, 종 $s$에 대한 운동 방정식은:

$$-i\omega m_s \mathbf{v}_s = e_s(\mathbf{E}_1 + \mathbf{v}_s \times \mathbf{B}_0)$$

$\mathbf{v}_s$를 풀고 $\mathbf{j} = \sum_s n_0 e_s \mathbf{v}_s$에 대입하면 $\overleftrightarrow{K}$를 얻습니다.

**Stix 표기법**에서, 텐서는 다음 형태를 가집니다:

$$\overleftrightarrow{K} = \begin{pmatrix} S & -iD & 0 \\ iD & S & 0 \\ 0 & 0 & P \end{pmatrix}$$

여기서:

$$\boxed{S = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2 - \omega_{cs}^2}}$$

$$\boxed{D = \sum_s \frac{\omega_{cs}}{\omega} \frac{\omega_{ps}^2}{\omega^2 - \omega_{cs}^2}}$$

$$\boxed{P = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2}}$$

여기서 $\omega_{cs} = e_s B_0/m_s$는 사이클로트론 주파수입니다 (우리 규약에 따라 전자의 경우 양수, 이온의 경우 음수).

**편리한 조합**:
$$R = S + D$$
$$L = S - D$$

$R$은 **우선형 원형** 편광에 해당하고, $L$은 **좌선형 원형**에 해당합니다.

### 2.2 파동 방정식

파동 방정식은:

$$\mathbf{k} \times (\mathbf{k} \times \mathbf{E}_1) + \frac{\omega^2}{c^2}\overleftrightarrow{K}\cdot\mathbf{E}_1 = 0$$

$\mathbf{k} \times (\mathbf{k} \times \mathbf{E}_1) = \mathbf{k}(\mathbf{k}\cdot\mathbf{E}_1) - k^2\mathbf{E}_1$을 사용하고 **굴절률** $\mathbf{n} = \mathbf{k}c/\omega$를 정의:

$$\mathbf{n} \times (\mathbf{n} \times \mathbf{E}_1) + \overleftrightarrow{K}\cdot\mathbf{E}_1 = 0$$

이것은 텐서 형태의 **Appleton-Hartree 방정식**입니다.

**분산 관계** $D(n, \omega) = 0$은 $\det[\mathbf{n} \times (\mathbf{n} \times \overleftrightarrow{I}) + \overleftrightarrow{K}] = 0$을 요구하는 것에서 나옵니다.

$\mathbf{B}_0$에 대해 각도 $\theta$로 전파하는 경우:

$$A n^4 - B n^2 + C = 0$$

여기서 $A, B, C$는 $S, D, P, \theta$의 복잡한 함수입니다. 두 가지 특별한 경우에 초점을 맞춥니다.

## 3. 평행 전파 ($\mathbf{k} \parallel \mathbf{B}_0$)

### 3.1 원형 편광 모드

$\mathbf{k} = k\hat{z}$ ($\mathbf{B}_0$를 따라)의 경우, $\mathbf{E}_1 = E_x\hat{x} + E_y\hat{y}$를 가진 해를 찾습니다.

파동 방정식은 다음을 제공합니다:
$$-n^2 E_x + S E_x - iD E_y = 0$$
$$-n^2 E_y + iD E_x + S E_y = 0$$

$E_\pm = E_x \pm iE_y$를 결합:

$$(S \mp D - n^2)E_\pm = 0$$

두 모드:
1. **R-파** (우원형): $E_+ \propto e^{ikz}$, $n^2 = R = S + D$
2. **L-파** (좌원형): $E_- \propto e^{ikz}$, $n^2 = L = S - D$

전자-이온 플라즈마의 경우:

$$R = 1 - \frac{\omega_{pe}^2}{\omega(\omega - \omega_{ce})} - \frac{\omega_{pi}^2}{\omega(\omega + \omega_{ci})}$$

$$L = 1 - \frac{\omega_{pe}^2}{\omega(\omega + \omega_{ce})} - \frac{\omega_{pi}^2}{\omega(\omega - \omega_{ci})}$$

### 3.2 전자 사이클로트론 공명

R-파는 $\omega = \omega_{ce}$에서 **공명**을 가집니다:

$$n^2 = R \to \infty \quad \text{as } \omega \to \omega_{ce}$$

공명에서:
- 파동 에너지는 파동과 공명하여 회전하는 전자에 흡수됨
- 파장 $\lambda = 2\pi/k \to 0$ (무한 $k$)
- 군속도 $v_g \to 0$ (에너지가 국소적으로 침착)

이것은 **전자 사이클로트론 공명 가열 (ECRH)**의 기초입니다:
- 주파수: $f = n \times f_{ce}$ (일반적으로 $n=1$ 또는 $2$)
- ITER의 경우: $B \sim 5.3$ T $\Rightarrow f_{ce} \sim 140$ GHz
- X-모드 또는 O-모드로 발사, 사이클로트론 층에서 흡수

### 3.3 이온 사이클로트론 공명

L-파는 $\omega = \omega_{ci}$에서 공명을 가집니다:

$$n^2 = L \to \infty \quad \text{as } \omega \to \omega_{ci}$$

**이온 사이클로트론 공명 가열 (ICRH)**의 경우:
- 주파수: 토카막의 경우 $f \sim 30-100$ MHz
- 이온 종을 선택적으로 가열 가능 (소수 가열)
- 모드 변환과 결합하여 사용

### 3.4 휘슬러파

주파수 범위 $\omega_{ci} \ll \omega \ll \omega_{ce}$에서, R-파는 **휘슬러 모드**가 됩니다.

근사 분산 (이온 무시, $\omega_{ce} \gg \omega$):

$$n^2 = R \approx \frac{\omega_{pe}^2}{\omega(\omega_{ce} - \omega)} \approx \frac{\omega_{pe}^2}{\omega \omega_{ce}}$$

따라서:
$$\boxed{k = \frac{\omega}{c}\sqrt{\frac{\omega_{pe}^2}{\omega \omega_{ce}}}}$$

또는:
$$\omega \approx \frac{k^2 c^2 \omega_{ce}}{\omega_{pe}^2}$$

특성:
- **고분산성**: $\omega \propto k^2$
- **우선형 편광**: $\mathbf{E}_1$이 전자와 같은 방향으로 회전
- **위상 속도**: $v_\phi = \omega/k \propto \omega^{1/2}$ (고주파가 더 빠르게 이동)
- **군속도**: $v_g = d\omega/dk \propto \omega$ (또한 분산성)

**발견**: 제2차 세계대전 중, 번개로부터의 오디오 신호가 무선 수신기에서 감지되어 특징적인 **하강 톤** (떨어지는 "휘슬")을 나타냈습니다:
- 번개는 광대역 EM파를 생성
- 고주파는 지구 자기장 선을 따라 더 빠르게 이동
- 더 일찍 도착 → "휘슬링" 소리

휘슬러는 자기장 선을 따라 한 반구에서 다른 반구로 전파되어 자기권에 대한 정보를 제공합니다.

**응용**:
- 자기권 진단
- 잠수함과의 VLF 통신
- 파동-입자 상호작용 (복사 벨트 역학)

## 4. 수직 전파 ($\mathbf{k} \perp \mathbf{B}_0$)

### 4.1 정상파와 비정상파 모드

$\mathbf{k} = k\hat{x}$, $\mathbf{B}_0 = B_0\hat{z}$의 경우, 두 개의 독립적인 모드가 있습니다:

**정상 모드 (O-모드)**: $\mathbf{E}_1 = E_z\hat{z}$ ($\mathbf{B}_0$에 평행)
- $\mathbf{v} \times \mathbf{B}$ 힘 없음 → 무자화 플라즈마처럼 동작
- 분산: $$\boxed{n^2 = P = 1 - \sum_s \frac{\omega_{ps}^2}{\omega^2}}$$
- $\omega_{pe}$에서 차단 (또는 정확한 처리의 경우 $\omega_R = \sqrt{\omega_{pe}^2 + \omega_{pi}^2}$)

**비정상 모드 (X-모드)**: $\mathbf{E}_1$이 $x$-$y$ 평면에 있음 ($\mathbf{B}_0$에 수직)
- $\mathbf{v} \times \mathbf{B}$ 힘이 운동을 결합
- 분산: $$\boxed{n^2 = \frac{RL}{S} = \frac{(S+D)(S-D)}{S}}$$

더 명시적으로:
$$n^2 = 1 - \frac{\omega_{pe}^2(\omega^2 - \omega_{UH}^2)}{\omega^2(\omega^2 - \omega_{UH}^2 - \omega_{ce}^2)}$$

여기서 $\omega_{UH}^2 = \omega_{pe}^2 + \omega_{ce}^2$는 상부 하이브리드 주파수입니다.

### 4.2 X-모드 차단과 공명

X-모드는 두 개의 **차단** ($n^2 = 0$인 곳)을 가집니다:

$$\omega_R = \frac{1}{2}\left(\omega_{ce} + \sqrt{\omega_{ce}^2 + 4\omega_{pe}^2}\right)$$ (우선형 차단)

$$\omega_L = \frac{1}{2}\left(-\omega_{ce} + \sqrt{\omega_{ce}^2 + 4\omega_{pe}^2}\right)$$ (좌선형 차단)

그리고 하나의 **공명** ($n^2 \to \infty$인 곳):

$$\omega = \omega_{UH} = \sqrt{\omega_{pe}^2 + \omega_{ce}^2}$$ (상부 하이브리드 공명)

상부 하이브리드 층 근처에서, 파장 $\lambda \to 0$이고 파동 에너지가 흡수됩니다 (정전 상부 하이브리드파로 변환).

### 4.3 모드 접근성

O-모드와 X-모드는 고밀도 플라즈마에 대한 접근성이 다릅니다:

**O-모드**:
- $\omega = \omega_{pe}$에서 차단
- $n > n_c$인 경우 전파할 수 없음, 여기서 $\omega_{pe}(n_c) = \omega$
- 밀도로 제한 $n < n_c = \epsilon_0 m_e \omega^2/e^2$

**X-모드**:
- $\omega_R > \omega_{pe}$에서 차단 (O-모드보다 높음)
- 더 높은 밀도에 접근 가능: $n < n_c^{X-mode} > n_c^{O-mode}$
- 과밀 플라즈마 가열에 사용

$\omega = 2\omega_{ce}$에서 ECRH의 경우:
- O-모드 차단: $n_c = 4 n_{ce}$ 여기서 $n_{ce} = \epsilon_0 m_e \omega_{ce}^2/e^2$
- X-모드 차단: 더 높음 (더 높은 밀도 접근 허용)

## 5. CMA 다이어그램

### 5.1 구성

**Clemmow-Mullaly-Allis (CMA) 다이어그램**은 파라미터 공간에서 파동 전파 영역의 지도입니다.

축:
- 수평: $X = \omega_{pe}^2/\omega^2$ (밀도 효과)
- 수직: $Y = \omega_{ce}/\omega$ (자기장 효과)

각 점 $(X, Y)$에 대해, 다이어그램은 어떤 모드 (R, L, O, X)가 전파하거나 감쇠하는지 보여줍니다.

**경계**:
- **차단**: $n^2 = 0$인 곡선 (전파와 감쇠 사이의 경계)
- **공명**: $n^2 \to \infty$인 곡선 (흡수)

주요 곡선:
- O-모드 차단: $X = 1$ (수직선)
- R-파 차단: $R = 0 \Rightarrow X = 1 - Y$
- L-파 차단: $L = 0 \Rightarrow X = 1 + Y$
- 상부 하이브리드: $X = 1 - Y^2$ (수직 전파의 경우)

### 5.2 영역과 모드 특성

```
Y (ω_ce/ω)
   ↑
   |        R, L, O 전파
   |     /
   | R  /   O
   | 차단
   |  /
   |/______________→ X (ω_pe²/ω²)
    1
```

다른 영역:
- **영역 I** ($X < 1 - Y$): 모든 모드 전파
- **영역 II** ($1 - Y < X < 1$): R-파 감쇠, L과 O 전파
- **영역 III** ($X > 1$): O-모드 감쇠

완전한 다이어그램 (이온 효과 및 수직 전파 포함)은 약 10개의 구별되는 영역을 가집니다.

### 5.3 응용

CMA 다이어그램은 다음에 사용됩니다:
- 가열 및 전류 구동 시스템 설계 (적절한 모드 선택)
- 진단 시스템 계획 (반사계, 간섭계)
- 전리층 전파 이해
- 자기권에서 휘슬러 전파 분석

예를 들어, 핵융합 플라즈마에서:
- 고밀도, 고 $B$ → $(X, Y)$가 X-모드가 필요한 영역에 있음
- 저밀도 가장자리 → O-모드 접근 가능

## 6. Faraday 회전

### 6.1 이론

**선형 편광** 파동이 자화 플라즈마를 통해 전파할 때, 편광 평면이 **회전**합니다. 이것이 **Faraday 회전**입니다.

물리적 원리:
- 선형 편광 = 동일한 진폭을 가진 R-파와 L-파의 중첩
- R과 L은 다른 굴절률을 가집니다: $n_R \neq n_L$
- 위상 차이가 누적됩니다: $\Delta\phi = (k_R - k_L) L$
- 편광 평면이 각도 $\theta = \Delta\phi/2$만큼 회전

$\omega \gg \omega_{pe}, \omega_{ce}$의 경우:

$$n_R - n_L \approx \frac{\omega_{pe}^2 \omega_{ce}}{\omega^3}$$

거리 $L$에 걸친 회전 각도는:

$$\boxed{\theta = \frac{\omega_{pe}^2 \omega_{ce}}{2c\omega^2} L = \frac{e^3}{2\epsilon_0 m_e^2 c \omega^2} \int_0^L n_e B_\parallel \, dl}$$

여기서 $B_\parallel$는 광선 경로를 따른 $\mathbf{B}$의 성분입니다.

**회전 측정**은:

$$RM = \frac{e^3}{2\pi \epsilon_0 m_e^2 c} \int n_e B_\parallel \, dl$$

실용 단위로:
$$RM \approx 2.63 \times 10^{-13} \int n_e(\text{cm}^{-3}) B_\parallel(\text{G}) \, dl(\text{pc}) \quad (\text{rad/m}^2)$$

### 6.2 천체물리학적 응용

Faraday 회전은 다음에서 자기장을 측정하는 데 사용됩니다:

**펄서**:
- 다중 주파수 편광 관측으로부터 $RM$ 측정
- 가시선을 따라 $\int n_e B_\parallel dl$ 추론
- 은하 자기장 구조 매핑

**활동 은하핵 (AGN)**:
- 제트는 얽힌 자기장을 가짐
- Faraday 회전이 자기장 강도와 구조를 제공

**은하단 내부 매질**:
- 은하단: $n_e \sim 10^{-3}$ cm$^{-3}$, $L \sim$ Mpc
- 회전 측정으로부터 $B \sim \mu$G 측정

**토카막 진단**:
- 편광계가 $\int n_e B_\parallel dl$ 측정
- 간섭계 ($\int n_e dl$)와 결합하여 $B$ 프로파일 추론 가능

### 6.3 분산과 비편광화

낮은 주파수에서 회전 각도가 더 큽니다: $\theta \propto \omega^{-2}$.

$\Delta\omega$를 가진 광대역 방출의 경우, 다른 주파수가 다른 양만큼 회전하여 **비편광화**를 유발합니다:

$$\Delta\theta \approx 2\theta \frac{\Delta\omega}{\omega}$$

$\Delta\theta \gtrsim \pi/2$인 경우, 순 편광이 뒤섞입니다.

이것은 **비편광화 주파수**를 설정합니다:
$$\omega_{\text{depol}} \sim \left(\frac{e^3 n_e B_\parallel L}{\epsilon_0 m_e^2 c}\right)^{1/2}$$

## 7. Python 구현

### 7.1 무자화 플라즈마에서의 분산

```python
import numpy as np
import matplotlib.pyplot as plt

def unmagnetized_dispersion(k, omega_pe):
    """
    EM wave dispersion in unmagnetized plasma: ω² = ω_pe² + k²c².
    """
    c = 3e8  # m/s
    omega = np.sqrt(omega_pe**2 + k**2 * c**2)
    return omega

# Parameters
n = 1e19  # m^-3
e = 1.602e-19  # C
epsilon_0 = 8.854e-12  # F/m
m_e = 9.109e-31  # kg
c = 3e8  # m/s

omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
f_pe = omega_pe / (2 * np.pi)

print(f"Plasma frequency: f_pe = {f_pe / 1e9:.2f} GHz")
print(f"Cutoff wavelength: λ_c = {c / f_pe:.2f} m")

# Wavenumber
k = np.linspace(0, 5, 1000) * omega_pe / c

# Dispersion
omega = unmagnetized_dispersion(k, omega_pe)

# Refractive index
n_refr = k * c / omega

# Group velocity
v_g = c**2 * k / omega

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Dispersion relation
ax1.plot(k * c / omega_pe, omega / omega_pe, 'b-', linewidth=2, label='Plasma')
ax1.plot(k * c / omega_pe, k * c / omega_pe, 'k--', linewidth=1.5,
         label='Vacuum ($\\omega = kc$)')
ax1.axhline(1, color='r', linestyle=':', linewidth=1.5, label='Cutoff ($\\omega_{pe}$)')
ax1.set_xlabel('$kc/\\omega_{pe}$', fontsize=13)
ax1.set_ylabel('$\\omega/\\omega_{pe}$', fontsize=13)
ax1.set_title('EM Wave Dispersion in Plasma', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 5])
ax1.set_ylim([0, 6])

# Refractive index
ax2.plot(omega / omega_pe, n_refr, 'b-', linewidth=2)
ax2.axhline(1, color='k', linestyle='--', alpha=0.5, label='Vacuum')
ax2.axvline(1, color='r', linestyle=':', linewidth=1.5, label='Cutoff')
ax2.set_xlabel('$\\omega/\\omega_{pe}$', fontsize=13)
ax2.set_ylabel('Refractive Index $n$', fontsize=13)
ax2.set_title('Refractive Index: $n = \\sqrt{1 - \\omega_{pe}^2/\\omega^2}$', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([1, 3])
ax2.set_ylim([0, 1.2])

# Phase velocity
v_phase = omega / k
ax3.plot(omega / omega_pe, v_phase / c, 'b-', linewidth=2, label='$v_\\phi$')
ax3.axhline(1, color='k', linestyle='--', alpha=0.5, label='$c$')
ax3.set_xlabel('$\\omega/\\omega_{pe}$', fontsize=13)
ax3.set_ylabel('$v_\\phi / c$', fontsize=13)
ax3.set_title('Phase Velocity (superluminal!)', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xlim([1, 3])
ax3.set_ylim([0.5, 3])

# Group velocity
ax4.plot(omega / omega_pe, v_g / c, 'r-', linewidth=2, label='$v_g$')
ax4.axhline(1, color='k', linestyle='--', alpha=0.5, label='$c$')
ax4.set_xlabel('$\\omega/\\omega_{pe}$', fontsize=13)
ax4.set_ylabel('$v_g / c$', fontsize=13)
ax4.set_title('Group Velocity (subluminal)', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xlim([1, 3])
ax4.set_ylim([0, 1.2])

plt.tight_layout()
plt.savefig('unmagnetized_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 7.2 Stix 파라미터와 모드 분산

```python
def stix_parameters(omega, n, B, Z=1, A=1):
    """
    Compute Stix parameters S, D, P for cold magnetized plasma.

    Parameters:
    -----------
    omega : float or array
        Wave frequency (rad/s)
    n : float
        Density (m^-3)
    B : float
        Magnetic field (T)
    Z : int
        Ion charge
    A : float
        Ion mass number

    Returns:
    --------
    dict with S, D, P, R, L
    """
    m_i = A * 1.673e-27

    omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
    omega_pi = np.sqrt(n * Z**2 * e**2 / (epsilon_0 * m_i))
    omega_ce = e * B / m_e
    omega_ci = Z * e * B / m_i

    # S, D, P는 차가운 플라즈마 유전 텐서의 세 독립 성분입니다.
    # S (합)는 x와 y 성분을 결합하는 대각 원소입니다; 공명 분모 ω² - ω_cs²는
    # 사이클로트론 주파수 근처에서 발산하여 고유 주파수 근처에서 회전하는 입자의
    # 특이 응답을 반영합니다.
    S = 1 - omega_pe**2 / (omega**2 - omega_ce**2) - \
        omega_pi**2 / (omega**2 - omega_ci**2)

    # D (차이)는 좌/우 대칭을 깨는 비대각 자이로트로픽(gyrotropic) 원소입니다;
    # 자기장이 없으면 (ω_cs → 0) 사라지며 Faraday 회전을 유발합니다.
    D = omega_ce * omega_pe**2 / (omega * (omega**2 - omega_ce**2)) + \
        omega_ci * omega_pi**2 / (omega * (omega**2 - omega_ci**2))

    # P (플라즈마)는 사이클로트론 운동이 관련 없는 B 방향의 응답을 지배합니다;
    # 전자만 있는 경우 비자화 유전체 (1 - ω_pe²/ω²)로 환원됩니다.
    P = 1 - omega_pe**2 / omega**2 - omega_pi**2 / omega**2

    # R과 L은 B를 따른 우원형 및 좌원형 편광에 대한 굴절률의 제곱입니다;
    # 평행 전파에 대한 파동 방정식의 행렬식에서 인수로 나옵니다.
    R = S + D
    L = S - D

    return {
        'S': S, 'D': D, 'P': P, 'R': R, 'L': L,
        'omega_pe': omega_pe, 'omega_pi': omega_pi,
        'omega_ce': omega_ce, 'omega_ci': omega_ci
    }

# Parameters
B = 2.0  # T
n = 5e19  # m^-3

params = stix_parameters(1, n, B, Z=1, A=2)  # Dummy ω for frequencies

f_ce = params['omega_ce'] / (2 * np.pi)
f_ci = params['omega_ci'] / (2 * np.pi)
f_pe = params['omega_pe'] / (2 * np.pi)

print(f"Electron cyclotron frequency: f_ce = {f_ce / 1e9:.2f} GHz")
print(f"Ion cyclotron frequency: f_ci = {f_ci / 1e6:.2f} MHz")
print(f"Electron plasma frequency: f_pe = {f_pe / 1e9:.2f} GHz")

# 0.1-100 GHz를 스캔하면 물리적으로 중요한 모든 공명을 포괄합니다
# (f_ci ~ MHz는 훨씬 아래에, f_ce ~ 56 GHz는 중간에, f_pe ~ GHz는 밀도에 따라 위치).
# 5000점 해상도는 전파와 소산(evanescent) regime 사이 전환을 표시하는
# f_ce 근처 S와 D의 급격한 부호 변화를 해석합니다.
f = np.linspace(0.1, 100, 5000) * 1e9  # Hz
omega = 2 * np.pi * f

params = stix_parameters(omega, n, B, Z=1, A=2)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Stix parameters
ax1.plot(f / 1e9, params['S'], 'b-', linewidth=2, label='S')
ax1.plot(f / 1e9, params['P'], 'r-', linewidth=2, label='P')
ax1.axhline(0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(f_ce / 1e9, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Frequency (GHz)', fontsize=13)
ax1.set_ylabel('Stix Parameter', fontsize=13)
ax1.set_title('S and P Parameters', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 100])
ax1.set_ylim([-5, 5])

ax2.plot(f / 1e9, params['D'], 'g-', linewidth=2)
ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(f_ce / 1e9, color='gray', linestyle=':', alpha=0.5)
ax2.set_xlabel('Frequency (GHz)', fontsize=13)
ax2.set_ylabel('D Parameter', fontsize=13)
ax2.set_title('D Parameter', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 100])

# R and L (refractive index squared for parallel propagation)
R_positive = np.where(params['R'] > 0, params['R'], np.nan)
L_positive = np.where(params['L'] > 0, params['L'], np.nan)

ax3.semilogy(f / 1e9, R_positive, 'b-', linewidth=2, label='R (R-wave)')
ax3.semilogy(f / 1e9, L_positive, 'r-', linewidth=2, label='L (L-wave)')
ax3.axvline(f_ce / 1e9, color='b', linestyle=':', alpha=0.5,
            label='$f_{ce}$ (R resonance)')
ax3.axvline(f_ci / 1e6 / 1e3, color='r', linestyle=':', alpha=0.5,
            label='$f_{ci}$ (L resonance)')
ax3.set_xlabel('Frequency (GHz)', fontsize=13)
ax3.set_ylabel('$n^2$ (R, L modes)', fontsize=13)
ax3.set_title('Parallel Propagation: $n^2 = R, L$', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, which='both', alpha=0.3)
ax3.set_xlim([0, 100])
ax3.set_ylim([0.01, 1000])

# O-mode and X-mode (perpendicular)
n2_O = params['P']
n2_X = params['R'] * params['L'] / params['S']

O_positive = np.where(n2_O > 0, n2_O, np.nan)
X_positive = np.where(n2_X > 0, n2_X, np.nan)

ax4.semilogy(f / 1e9, O_positive, 'b-', linewidth=2, label='O-mode')
ax4.semilogy(f / 1e9, X_positive, 'r-', linewidth=2, label='X-mode')
ax4.axvline(f_pe / 1e9, color='b', linestyle=':', alpha=0.5,
            label='$f_{pe}$ (O cutoff)')
ax4.set_xlabel('Frequency (GHz)', fontsize=13)
ax4.set_ylabel('$n^2$ (O, X modes)', fontsize=13)
ax4.set_title('Perpendicular Propagation: O and X modes', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, which='both', alpha=0.3)
ax4.set_xlim([0, 100])
ax4.set_ylim([0.01, 100])

plt.tight_layout()
plt.savefig('stix_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()
```

### 7.3 CMA 다이어그램

```python
def cma_diagram():
    """
    Generate CMA (Clemmow-Mullaly-Allis) diagram.
    """
    X = np.linspace(0, 3, 500)
    Y = np.linspace(0, 2, 500)
    XX, YY = np.meshgrid(X, Y)

    # Cutoff and resonance curves
    # O-mode cutoff: X = 1
    # R-wave cutoff: R = 0 → X = 1 - Y (for single species, approx)
    # L-wave cutoff: L = 0 → X = 1 + Y
    # Upper hybrid resonance: X = 1 - Y²

    fig, ax = plt.subplots(figsize=(10, 8))

    # Define regions based on cutoffs
    # (Simplified for electrons only)

    # O-mode propagates when X < 1
    O_propagate = XX < 1

    # R-wave propagates when R > 0 (approx X < 1 - Y for Y << 1)
    # More accurately, solve R = 0
    R_cutoff_Y = np.linspace(0, 1.5, 100)
    R_cutoff_X = 1 - R_cutoff_Y

    # L-wave cutoff: X = 1 + Y
    L_cutoff_Y = np.linspace(0, 1.5, 100)
    L_cutoff_X = 1 + L_cutoff_Y

    # Upper hybrid resonance: X = 1 - Y²
    UH_res_Y = np.linspace(0, 1.5, 100)
    UH_res_X = 1 - UH_res_Y**2

    # 각 경계는 파동이 전파에서 소산(evanescent)으로 전환하는 곳(차단)이거나
    # 유한에서 무한 파수로 전환하는 곳(공명)입니다. (X, Y) 공간에서 이를 그리면
    # 밀도나 B가 변할 때 플라즈마가 다이어그램에서 어떤 경로를 통과하는지 보여주어
    # 안테나/발사 설계에 필수적입니다.
    ax.plot([1, 1], [0, 2], 'b-', linewidth=2, label='O-mode cutoff ($X=1$)')
    ax.plot(R_cutoff_X, R_cutoff_Y, 'r-', linewidth=2,
            label='R-wave cutoff ($X=1-Y$)')
    ax.plot(L_cutoff_X, L_cutoff_Y, 'g-', linewidth=2,
            label='L-wave cutoff ($X=1+Y$)')
    # upper hybrid 공명은 O-mode 전파 영역(X < 1) 안에 위치합니다.
    # 이것이 X-mode가 저밀도 측에서 도달할 수 있는 반면 O-mode는 도달할 수 없는 이유입니다.
    ax.plot(UH_res_X, UH_res_Y, 'm--', linewidth=2,
            label='Upper hybrid res ($X=1-Y^2$)')

    # Add shaded regions
    ax.fill_betweenx([0, 2], 0, 1, alpha=0.1, color='blue', label='O propagates')
    ax.fill_between(R_cutoff_X, 0, R_cutoff_Y, alpha=0.1, color='red',
                     label='R evanescent')

    ax.set_xlabel('$X = \\omega_{pe}^2 / \\omega^2$', fontsize=14)
    ax.set_ylabel('$Y = \\omega_{ce} / \\omega$', fontsize=14)
    ax.set_title('CMA Diagram (Simplified, Electrons Only)', fontsize=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 3])
    ax.set_ylim([0, 2])

    # Annotate regions
    ax.text(0.5, 0.5, 'All modes\npropagate', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(1.5, 0.3, 'O-mode\nevanescent', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.3, 1.5, 'R-wave\nevanescent', fontsize=12, ha='center',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    plt.tight_layout()
    plt.savefig('cma_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()

cma_diagram()
```

### 7.4 휘슬러파 분산

```python
def whistler_dispersion(k, omega_pe, omega_ce):
    """
    Whistler wave dispersion: ω ≈ k²c²ω_ce/ω_pe².
    """
    c = 3e8
    # ω ∝ k² 스케일링(이상 분산)은 휘슬러 한계 ω_ci << ω << ω_ce에서
    # R-파 표현 n² = R ≈ ω_pe² / (ω ω_ce)에서 옵니다;
    # kc/ω = n을 재배열하면 ω = k²c²ω_ce / ω_pe²가 됩니다 — 높은 주파수가
    # 더 큰 위상 속도를 가지므로 더 빨리 이동하여 먼저 도달합니다 (하강 톤).
    omega = k**2 * c**2 * omega_ce / omega_pe**2
    # 이온 기여를 R에서 무시합니다. 휘슬러 주파수에서 ω >> ω_ci이므로
    # 이온이 파동에 응답하기에 너무 느려 χ_i ≈ 0이기 때문입니다.
    # 0.5 ω_ce에서의 상한은 휘슬러 근사의 유효성을 강제합니다;
    # ω_ce 근처에서는 공명 발산이 중요하므로 완전한 R 표현을 사용해야 합니다.
    omega = np.minimum(omega, 0.5 * omega_ce)  # Limit to whistler range
    return omega

# 자기권 매개변수는 f_pe와 f_ce를 모두 kHz 범위에 위치시키도록 선택했습니다:
# 이 낮은 밀도(n ~ 10^6 m^-3)에서 f_pe ~ 9 kHz이고, B ~ 50 μT는
# f_ce ~ 1.4 MHz를 줍니다 — 가청 주파수 휘슬러에 대해 ω_ci << ω << ω_ce를 만족합니다.
n = 1e6  # m^-3 (magnetosphere)
B = 5e-5  # T (Earth's magnetic field at magnetosphere)

omega_pe = np.sqrt(n * e**2 / (epsilon_0 * m_e))
omega_ce = e * B / m_e

f_pe = omega_pe / (2 * np.pi)
f_ce = omega_ce / (2 * np.pi)

print(f"Magnetospheric parameters:")
print(f"  f_pe = {f_pe / 1e3:.1f} kHz")
print(f"  f_ce = {f_ce / 1e3:.1f} kHz")

# Wavenumber
k = np.linspace(1e-7, 1e-5, 500)  # m^-1

omega = whistler_dispersion(k, omega_pe, omega_ce)
f = omega / (2 * np.pi)

# Phase and group velocity
v_phase = omega / k
v_group = 2 * omega / k

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Dispersion
ax1.plot(k * 1e6, f / 1e3, 'b-', linewidth=2)
ax1.axhline(f_ce / (2e3), color='r', linestyle='--',
            label='$f_{ce}/2$ (upper limit)')
ax1.set_xlabel('Wavenumber $k$ ($\\mu$m$^{-1}$)', fontsize=13)
ax1.set_ylabel('Frequency (kHz)', fontsize=13)
ax1.set_title('Whistler Dispersion: $\\omega \\propto k^2$', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Phase velocity
ax2.plot(f / 1e3, v_phase / c, 'b-', linewidth=2)
ax2.set_xlabel('Frequency (kHz)', fontsize=13)
ax2.set_ylabel('$v_\\phi / c$', fontsize=13)
ax2.set_title('Phase Velocity (higher f travels faster)', fontsize=14)
ax2.grid(True, alpha=0.3)

# Group velocity
ax3.plot(f / 1e3, v_group / c, 'r-', linewidth=2)
ax3.set_xlabel('Frequency (kHz)', fontsize=13)
ax3.set_ylabel('$v_g / c$', fontsize=13)
ax3.set_title('Group Velocity: $v_g = 2v_\\phi$', fontsize=14)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('whistler_dispersion.png', dpi=150, bbox_inches='tight')
plt.show()

# Simulate whistler arrival times
print("\nWhistler arrival times from lightning (L = 10,000 km):")
L = 1e7  # m
for f_khz in [1, 2, 5, 10]:
    f_val = f_khz * 1e3
    idx = np.argmin(np.abs(f / 1e3 - f_khz))
    t_arrival = L / v_group[idx]
    print(f"  {f_khz} kHz: {t_arrival:.2f} s")
```

## 요약

플라즈마에서 전자기파는 플라즈마 주파수, 사이클로트론 주파수, 파동 주파수의 상호작용에 의해 지배되는 풍부한 동작을 나타냅니다:

**무자화 플라즈마**:
- 분산: $\omega^2 = \omega_{pe}^2 + k^2c^2$
- $\omega_{pe}$에서 차단: $\omega < \omega_{pe}$인 파동은 전파할 수 없음
- 굴절률 $n < 1$: 위상 속도 $> c$, 하지만 군속도 $< c$
- 응용: 전리층 반사, 플라즈마 밀도 진단

**자화 플라즈마 (Stix 형식론)**:
- 성분 $S$, $D$, $P$를 가진 유전 텐서
- 비등방성 응답은 다중 파동 모드를 유도
- 원형 편광에 대한 조합 $R = S + D$, $L = S - D$

**평행 전파**:
- R-파 (우원형): $\omega = \omega_{ce}$에서 공명 → ECRH
- L-파 (좌원형): $\omega = \omega_{ci}$에서 공명 → ICRH
- 휘슬러파 ($\omega_{ci} \ll \omega \ll \omega_{ce}$): 고분산성, $\omega \propto k^2$
- 응용: 자기권 물리학, VLF 통신

**수직 전파**:
- O-모드: $\mathbf{E} \parallel \mathbf{B}$, $\omega_{pe}$에서 차단
- X-모드: $\mathbf{E} \perp \mathbf{B}$, $\omega_R, \omega_L$에서 차단, $\omega_{UH}$에서 공명
- 모드 접근성: X-모드가 O-모드보다 더 높은 밀도에 도달

**CMA 다이어그램**:
- $(X, Y)$ 공간에서 전파 영역 매핑, 여기서 $X = \omega_{pe}^2/\omega^2$, $Y = \omega_{ce}/\omega$
- 차단, 공명, 모드 경계 표시
- 파동 가열 및 진단 설계를 위한 필수 도구

**Faraday 회전**:
- 회전 각도: $\theta \propto \int n_e B_\parallel dl \cdot \omega^{-2}$
- 천체물리학적 플라즈마에서 자기장 측정
- 전류 프로파일 측정을 위한 핵융합 편광계에 사용

이러한 파동 현상은 다음을 가능하게 합니다:
- 플라즈마 가열 (ECRH, ICRH, LHCD)
- 전류 구동 (비유도 작동)
- 진단 (간섭계, 반사계, 편광계)
- 통신 (전리층 전파)
- 천체물리학 관측 (펄서 RM, AGN 제트)

## 연습 문제

### 문제 1: 전리층 반사
전리층은 $n_e = 10^{12}$ m$^{-3}$의 최대 밀도를 가집니다.

(a) 최대에서 플라즈마 주파수 $f_{pe}$를 계산하십시오.

(b) AM 라디오 방송국이 1 MHz로 방송합니다. 신호가 전리층에서 반사되거나 통과합니까?

(c) 신호가 전리층을 통과하는 최소 주파수는 무엇입니까?

(d) 전리층 밀도는 하루 중 시간에 따라 변합니다. 낮 동안 $n_e$가 10배 증가합니다. 이것이 AM 라디오 전파에 어떻게 영향을 줍니까?

### 문제 2: ECRH 시스템 설계
토카막은 축에서 $B_0 = 3.5$ T를 가집니다. 중심 가열을 위한 ECRH 시스템을 설계하고 있습니다.

(a) 전자 사이클로트론 주파수 $f_{ce}$와 진공에서의 파장을 계산하십시오.

(b) 시스템은 2차 고조파 ($\omega = 2\omega_{ce}$)를 사용할 것입니다. 자이로트론이 생성해야 하는 주파수는 무엇입니까?

(c) 밀도 $n = 5 \times 10^{19}$ m$^{-3}$에 대해, $2f_{ce}$에서 O-모드 차단 밀도를 계산하십시오. 파동이 중심에 도달할 수 있습니까?

(d) O-모드가 중심에 도달할 수 없는 경우, X-모드 또는 electron Bernstein wave (EBW)를 대신 사용할 수 있는 방법을 설명하십시오.

### 문제 3: 휘슬러 전파
자기 적도에서의 번개 타격은 필드 선을 따라 반대 반구로 전파하는 광대역 신호를 생성합니다.

(a) $B = 5 \times 10^{-5}$ T, $n = 10^7$ m$^{-3}$에 대해, $f_{ce}$와 $f_{pe}$를 계산하십시오.

(b) 주파수 범위 $f_{ci} \ll f \ll f_{ce}$가 $f \sim 1-10$ kHz에 대해 만족됨을 보이십시오 (수소 이온 가정).

(c) $f = 5$ kHz에서 군속도를 계산하십시오.

(d) 경로 길이가 $L = 10,000$ km인 경우, 5 kHz 성분이 도착하는 데 얼마나 걸립니까? 1 kHz 성분은 어떻습니까? "휘슬링" 소리를 설명하십시오.

### 문제 4: CMA 다이어그램 영역
$f_{pe} = 50$ GHz와 $f_{ce} = 70$ GHz를 가진 플라즈마에 대해, 다음 주파수에서 어떤 모드가 전파할 수 있는지 결정하십시오:

(a) $f = 40$ GHz (X-밴드 레이더)

(b) $f = 75$ GHz (W-밴드)

(c) $f = 120$ GHz (2차 고조파에서 ECRH)

(d) 각 주파수에 대해, $X$와 $Y$를 계산하고, 접근 가능한 모드 (R, L, O, X)를 식별하십시오.

### 문제 5: Faraday 회전 측정
$\lambda = 6$ cm (5 GHz)의 편광파가 $n_e = 10^{18}$ m$^{-3}$, $B_\parallel = 0.1$ T, $L = 1$ m를 가진 플라즈마 슬랩을 통해 전파합니다.

(a) Faraday 회전 각도 $\theta$를 계산하십시오.

(b) 두 파장 ($\lambda_1 = 6$ cm, $\lambda_2 = 3$ cm)에서 측정이 수행되는 경우, 회전 각도의 차이 $\Delta\theta$는 무엇입니까?

(c) $\Delta\theta$로부터 회전 측정 $RM = \theta/\lambda^2$를 유도하십시오.

(d) $RM$만 측정되는 경우 ($n_e$와 $B$가 별도로 측정되지 않음), $B_\parallel$를 결정하기 위해 어떤 추가 측정이 필요합니까?

---

**이전**: [10. Electrostatic Waves](./10_Electrostatic_Waves.md)
**다음**: [12. Wave Heating and Instabilities](./12_Wave_Heating_and_Instabilities.md)
