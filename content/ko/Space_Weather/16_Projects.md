# 프로젝트

## 학습 목표

- Burton 방정식을 적용하여 태양풍 입력으로부터 Dst를 예측하고 LSTM 기반 예측과 비교하기
- 복사대(Radiation Belt) 전자 위상 공간 밀도(Phase Space Density) 진화를 위한 1D 방사 확산 솔버 구현하기
- 다양한 데이터 스트림과 모델을 통합하는 우주 날씨 모니터링 대시보드 구축하기
- 여러 레슨(지수, 예측, 자기권 물리, 복사대)의 지식을 통합하여 활용하기
- 모델 검증, 오차 분석, 시각화를 포함한 과학적 데이터 분석 실습하기

---

## 1. 프로젝트 1: Dst 예측 — Burton 모델 vs LSTM

### 1.1 개요

이 프로젝트는 상류 태양풍 측정으로부터 Dst 지자기 지수를 예측하는 두 가지 근본적으로 다른 접근 방식을 구현합니다: 물리 기반 Burton 방정식과 데이터 기반 LSTM 신경망. 두 방법을 모두 구축하고 비교함으로써, 각 패러다임의 강점과 약점에 대한 직관을 키울 수 있습니다.

### 1.2 Burton 방정식

Burton et al. (1975) 모델은 환전류(Ring Current)를 구동-소산 시스템으로 취급합니다. 압력 보정된 Dst는 다음 방정식에 따라 변화합니다:

$$\frac{d\text{Dst}^*}{dt} = Q(t) - \frac{\text{Dst}^*}{\tau}$$

여기서 주입 함수 $Q$와 회복 시상수(Timescale) $\tau$는 다음과 같습니다:

$$Q(t) = \begin{cases} a \cdot (E_y - E_c) & \text{if } E_y > E_c \\ 0 & \text{if } E_y \leq E_c \end{cases}$$

매개변수:
- $E_y = -v_{sw} \cdot B_z$ (새벽-황혼 방향 대류 전기장(Dawn-Dusk Convection Electric Field), 단위: mV/m. $B_z$는 GSM 좌표계 기준이며, 음수는 남향을 의미)
- $E_c = 0.5$ mV/m (주입 임계 전기장)
- $a = -4.5$ nT/(mV/m $\cdot$ hr) (주입 효율)
- $\tau = 7.7$ hr (환전류 손실에 대한 지수 감쇠 시간)

압력 보정(Pressure Correction)은 관측된 Dst와 환전류 기여분을 연결합니다:

$$\text{Dst}^* = \text{Dst} - b\sqrt{P_{\text{dyn}}} + c$$

여기서 $P_{\text{dyn}} = m_p n v^2$는 태양풍 동압력(Dynamic Pressure), $b \approx 7.26$ nT/$\sqrt{\text{nPa}}$, $c \approx 11$ nT입니다.

### 1.3 단계별 구현

**단계 (a): 데이터 취득**

NASA OMNIWeb(omniweb.gsfc.nasa.gov)에서 2000-2020년 시간별 OMNI 데이터를 획득합니다. 필요한 매개변수:
- 태양풍 속도 $v$ (km/s)
- 양성자 수 밀도(Proton Number Density) $n$ (cm$^{-3}$)
- IMF 크기 $B$ (nT) 및 $B_z$ GSM 성분 (nT)
- 관측된 Dst (nT)

데이터 공백 처리: OMNI는 결측 데이터에 채움 값(Fill Value)(예: 9999.9)을 사용합니다. 3시간 미만의 공백에는 선형 보간을 구현하고, 더 긴 공백 구간은 제외합니다.

**단계 (b): 파생 물리량 계산**

원시 매개변수로부터 다음을 계산합니다:

$$P_{\text{dyn}} = 1.6726 \times 10^{-6} \cdot n \cdot v^2 \quad [\text{nPa}]$$

$$E_y = -v \cdot B_z \times 10^{-3} \quad [\text{mV/m}]$$

$$\text{Dst}^* = \text{Dst} - 7.26\sqrt{P_{\text{dyn}}} + 11 \quad [\text{nT}]$$

$E_y$ 단위 변환에 유의하세요: $v$는 km/s, $B_z$는 nT 단위이므로 $E_y$가 $\mu$V/m으로 계산됩니다. $10^{-3}$을 곱해 mV/m으로 변환합니다. 또는 처음부터 SI 단위를 일관되게 사용할 수 있습니다.

**단계 (c): Burton ODE의 수치 적분**

간단한 오일러(Euler) 방법(시간별 데이터에 충분)이나 4차 룽게-쿠타(Runge-Kutta) 방법(더 정확)으로 ODE를 적분합니다:

**오일러 방법:** 시간 간격 $\Delta t = 1$시간:

$$\text{Dst}^*(t + \Delta t) = \text{Dst}^*(t) + \left[Q(t) - \frac{\text{Dst}^*(t)}{\tau}\right] \Delta t$$

**초기 조건:** 적분 구간 시작 시점의 관측값으로 $\text{Dst}^*(t_0)$를 설정합니다.

**Dst 복원:** 적분 후 역변환:

$$\text{Dst}(t) = \text{Dst}^*(t) + b\sqrt{P_{\text{dyn}}(t)} - c$$

**단계 (d): LSTM 구현**

태양풍 시계열을 전처리합니다:
1. 입력 특성(Feature) 선택: $v$, $n$, $B$, $B_z$, $P_{\text{dyn}}$, $E_y$
2. 각 특성을 평균 0, 분산 1로 정규화 (테스트 데이터에 동일한 스케일러 적용)
3. 슬라이딩 윈도우 생성: 24시간 입력(24 타임스텝 $\times$ 6 특성)을 $t + 1$시간의 Dst에 매핑

아키텍처:
- LSTM 레이어: 64 은닉 유닛 (return sequences = False)
- 드롭아웃(Dropout) 레이어: rate = 0.2
- 덴스(Dense) 레이어: 32 유닛, ReLU 활성화
- 출력 덴스 레이어: 1 유닛 (선형 활성화)

훈련:
- 분할: 2000-2015년 훈련, 2015-2020년 테스트
- 손실 함수: 평균 제곱 오차(Mean Squared Error)
- 옵티마이저: Adam (학습률 = $10^{-3}$)
- 배치 크기: 256
- 에포크: 50-100 (조기 종료: patience = 10, 검증 손실 모니터링)
- 검증: 훈련 데이터의 마지막 10% (시간적 누수(Temporal Leakage) 방지를 위해 무작위 분할 사용 금지)

**단계 (e): 비교**

두 모델 모두 테스트 세트(2015-2020)에서 다음을 계산합니다:
- RMSE (nT)
- 피어슨 상관계수(Pearson Correlation Coefficient) $r$
- 산점도: 예측 vs 관측 Dst (1:1 기준선 포함)
- 시계열 오버레이: 특정 폭풍에 대해 관측 Dst와 두 모델 예측을 함께 표시

**단계 (f): 폭풍 이벤트 분석**

테스트 기간에서 가장 강력한 폭풍 5-10개(Dst < $-100$ nT)를 식별합니다. 각 폭풍에 대해 비교합니다:
- 모델이 Dst 최솟값을 포착하는가? (체계적인 과소 예측은 모델이 폭풍 강도의 최고점을 과소 평가한다는 것을 의미)
- 회복 단계가 얼마나 잘 표현되는가? Burton 모델은 지수 회복을 가정하지만, 실제로는 초기에 빠른 회복 후 느린 꼬리 패턴을 보이는 경우가 많음
- 어떤 모델이 다단계 폭풍(여러 번 하락)을 더 잘 처리하는가?

### 1.4 토론 질문

결과를 분석하면서 다음 질문을 생각해 보세요:

- Burton 모델은 4개의 자유 매개변수를 가지며 특정 물리를 인코딩합니다. LSTM은 수천 개의 매개변수를 갖고 데이터에서 학습합니다. 어떤 상황에서 어느 쪽을 선호하겠습니까?
- 훈련 세트의 어떤 폭풍보다 훨씬 강한 폭풍이 발생하면 LSTM은 어떻게 됩니까? Burton 모델은 어떻게 됩니까?
- 회복 단계 예측을 개선하기 위해 Burton 방정식을 어떻게 수정하겠습니까? (힌트: O'Brien-McPhetridge 공식처럼 $\tau$를 활동 의존적으로 만드는 것을 고려해 보세요.)

---

## 2. 프로젝트 2: 복사대 방사 확산

### 2.1 개요

이 프로젝트는 복사대 내 전자 위상 공간 밀도(PSD, Phase Space Density)에 대한 1D 방사 확산 솔버를 구현합니다. ULF 파동 활동에 의해 구동되는 방사 확산(Radial Diffusion)은 복사대 전자를 가속하고 재분배하는 주요 메커니즘 중 하나입니다. 확산 방정식을 수치적으로 풀어, 지자기 활동(Kp로 매개변수화)의 변화가 복사대의 증강 및 고갈을 어떻게 유발하는지 관찰하게 됩니다.

### 2.2 지배 방정식

첫 번째 및 두 번째 단열 불변량(Adiabatic Invariant)($\mu$, $J$)이 고정된 상태에서 전자 PSD $f$에 대한 방사 확산 방정식:

$$\frac{\partial f}{\partial t} = L^2 \frac{\partial}{\partial L}\left(\frac{D_{LL}}{L^2} \frac{\partial f}{\partial L}\right) - \frac{f}{\tau_{\text{loss}}} + S$$

여기서:
- $f(L, t)$는 $L$-껍질(L-shell) $L$, 시간 $t$에서의 위상 공간 밀도
- $D_{LL}(L, \text{Kp})$는 방사 확산 계수
- $\tau_{\text{loss}}(L, \text{Kp})$는 손실 시상수(파동-입자 산란에 의한 대기 강수)
- $S(L, t)$는 소스 항 (이 프로젝트에서는 $S = 0$으로 설정하여 방사 이동과 손실만 고려)

### 2.3 단계별 구현

**단계 (a): L-격자 설정**

$L$-껍질에 균일한 격자를 생성합니다:
- $L_{\min} = 2.0$ (내부 경계 — 안정적인 내부 벨트)
- $L_{\max} = 7.0$ (외부 경계 — 정지 궤도 근방)
- $N = 100$ 격자점
- $\Delta L = (L_{\max} - L_{\min}) / (N - 1) = 0.0505$

**단계 (b): 방사 확산 계수**

Brautigam & Albert (2000)의 자기 성분 매개변수화를 사용합니다:

$$D_{LL}^M(L, \text{Kp}) = D_0 \times 10^{(0.506 \text{Kp} - 9.325)} \times L^{10}$$

$D_0 = 1$ (단위: day$^{-1}$, SI 계산을 위해 s$^{-1}$로 변환 가능).

강한 $L^{10}$ 의존성은 방사 확산이 큰 $L$에서 압도적으로 지배적이고 내부 벨트에서는 무시 가능하다는 것을 의미합니다. 물리적으로, 이는 ULF 파동 진폭이 지구로부터 거리에 따라 극적으로 증가한다는 사실을 반영합니다.

완전한 모델을 위해 전기 성분도 포함합니다:

$$D_{LL}^E(L, \text{Kp}) = 0.26 \times 10^{(0.217 \text{Kp} - 5.197)} \times L^{8}$$

더 완전한 모델을 위해 $D_{LL} = D_{LL}^M + D_{LL}^E$를 사용하거나, 단순화를 위해 $D_{LL}^M$만 사용합니다.

**단계 (c): 손실 시상수**

$L$과 Kp의 함수로 손실 시상수를 매개변수화합니다. 단순화된 모델:

플라스마 권(Plasmasphere) 내부 (플라스마 권 히스(Plasmaspheric Hiss)가 지배):

$$\tau_{\text{loss}}^{\text{hiss}}(L) = 10 \text{ days} \quad \text{for } L < L_{pp}$$

플라스마 권 외부 (코러스 파동(Chorus Waves)이 지배):

$$\tau_{\text{loss}}^{\text{chorus}}(L, \text{Kp}) = \frac{3}{(\text{Kp} + 1)} \text{ days} \quad \text{for } L \geq L_{pp}$$

플라스마 권계면(Plasmapause) 위치는 활동에 따라 달라집니다:

$$L_{pp} = 5.6 - 0.46 \times \text{Kp}$$

이는 단순하지만 물리적으로 근거 있는 매개변수화입니다: 폭풍 시(높은 Kp) 플라스마 권이 침식되고, 코러스 파동이 안쪽으로 확장되며, 손실 시상수가 짧아집니다.

**단계 (d): 경계 조건**

- **내부 경계** ($L = L_{\min} = 2.0$): 고정(디리클레, Dirichlet). 내부 벨트는 안정적이므로 $f(L_{\min}, t) = f_0$ (조용한 시기 값).
- **외부 경계** ($L = L_{\max} = 7.0$): 시간 변화. 실제로 외부 경계에서의 PSD는 태양풍 구동 과정에 의해 결정됩니다. 이 프로젝트에서는 단순 모델을 사용합니다:

$$f(L_{\max}, t) = f_{\text{quiet}} \times \begin{cases} 5.0 & \text{if Kp} \geq 5 \text{ (enhanced source)} \\ 1.0 & \text{otherwise} \end{cases}$$

**단계 (e): 초기 조건**

조용한 시기 PSD 프로파일로 시작합니다:

$$f_0(L) = f_{\text{peak}} \exp\left[-\frac{(L - L_{\text{peak}})^2}{2\sigma^2}\right]$$

$f_{\text{peak}} = 1.0$ (정규화), $L_{\text{peak}} = 4.5$, $\sigma = 0.8$. 이는 외부 존(Outer Zone)에서 전형적인 조용한 시기 복사대 최대치를 나타냅니다.

**단계 (f): 수치 기법**

안정성을 위해 크랭크-니콜슨(Crank-Nicolson) 방법을 사용합니다. $g = f / L^2$로 정의하여 방정식을 변환합니다:

$$L^2 \frac{\partial (L^2 g)}{\partial t} = L^2 \frac{\partial}{\partial L}\left(\frac{D_{LL}}{L^2} \frac{\partial (L^2 g)}{\partial L}\right) - \frac{L^2 g}{\tau_{\text{loss}}}$$

또는 $f$로 직접 이산화합니다:

$$\frac{f_i^{n+1} - f_i^n}{\Delta t} = \frac{1}{2}\left[\mathcal{D}(f^{n+1}) + \mathcal{D}(f^n)\right] - \frac{1}{2}\left[\frac{f_i^{n+1}}{\tau_i} + \frac{f_i^n}{\tau_i}\right]$$

여기서 $\mathcal{D}(f)$는 중앙 차분(Central Differences)으로 이산화된 공간 확산 연산자입니다. 이는 $f^{n+1}$에 대한 삼대각 시스템을 만들어 토마스(Thomas) 알고리즘으로 풀 수 있습니다.

시간 간격: $\Delta t = 1$시간 (3600 s). 가장 큰 $L$과 가장 높은 Kp에서 확산 CFL 조건 $D_{LL} \Delta t / \Delta L^2 < 1$이 만족되는지 확인하고, 그렇지 않으면 $\Delta t$를 줄이거나 음함수(Implicit) 기법을 사용합니다.

**단계 (g): 실제 폭풍으로 구동**

실제 지자기 폭풍(예: 2003년 할로윈 폭풍, 또는 2015년 성 패트릭의 날 폭풍)에 대한 Kp 시계열을 획득합니다. 폭풍 전 조용한 시기, 폭풍 기간, 회복 기간을 포함하는 구간(예: 총 10-14일)에 대해 시뮬레이션을 실행합니다.

### 2.4 분석

다음 시각화를 생성합니다:
1. **$f(L, t)$ 색상 지도:** $y$축에 $L$, $x$축에 시간, PSD를 색상으로 표현. 이는 복사대 역학을 시각화하는 표준 방법입니다.
2. **주요 시점의 PSD 프로파일:** 폭풍 전 조용한 시기, 폭풍 최대값 시, 회복 수일 후의 $f(L)$. 증강과 고갈이 발생하는 위치를 확인합니다.
3. **고정된 $L$에서의 시간 프로파일:** $L = 4, 5, 6$에서의 $f(t)$ — 서로 다른 방사 거리에서의 진화를 보여줍니다.
4. **Kp 오버레이:** 구동과 반응의 상관 관계를 확인하기 위해 $f$와 함께 Kp를 표시합니다.

### 2.5 물리적 해석

시뮬레이션에서 다음과 같은 결과를 관찰해야 합니다:
- **$L \sim 4-5$에서의 증강:** 증강된 외부 경계로부터의 내향 방사 확산이 전자를 더 낮은 $L$로 이동시키면서 에너지를 부여합니다(더 강한 자기장에서 첫 번째 단열 불변량 보존).
- **$L > 5-6$에서의 고갈:** 폭풍 주상(Main Phase) 동안, 자기권계면(Magnetopause)으로의 외향 방사 이동(입자 손실)과 증강된 코러스 파동 손실이 플럭스 감소를 만듭니다.
- **느린 회복:** 폭풍 이후 방사 확산이 양쪽 경계에서 고갈된 지역을 점진적으로 채웁니다.
- **시상수 분리:** 내부 벨트($L < 3$)는 거의 변화하지 않는 반면(무시 가능한 $D_{LL}$), 외부 벨트($L > 4$)는 수 시간~수 일 내에 극적으로 반응합니다.

---

## 3. 프로젝트 3: 우주 날씨 대시보드

### 3.1 개요

이 프로젝트는 이전 레슨들의 여러 구성 요소를 통합하여 우주 날씨 상태와 간단한 모델 기반 예측을 표시하는 모니터링 대시보드를 구축합니다. 대시보드는 태양풍 데이터를 시각화하고, 파생 물리량을 계산하며, 우주 날씨 상태를 한눈에 파악할 수 있게 합니다.

### 3.2 구성 요소

대시보드는 다섯 개의 주요 패널로 구성됩니다:

**패널 (a): 태양풍 모니터**

OMNI 또는 DSCOVR 데이터에서 태양풍 매개변수의 시계열을 표시합니다:
- 태양풍 속도 $v$ (km/s) — 색상 코딩: 초록(<400), 노랑(400-600), 빨강(>600)
- 양성자 밀도 $n$ (cm$^{-3}$) — 로그 스케일
- IMF 크기 $|B|$ (nT) 및 $B_z$ GSM (nT) — 남향($B_z < 0$)을 빨강으로 강조
- 동압력 $P_{\text{dyn}}$ (nPa)

현재 값을 눈에 띄게 표시하며, 최근 24-48시간 데이터를 보여줍니다.

**패널 (b): 자기권계면 위치**

Shue et al. (1998) 모델을 사용하여 태양 쪽 자기권계면 정지 거리(Standoff Distance)를 계산합니다:

$$r = R_0 \left(\frac{2}{1 + \cos\theta}\right)^\alpha$$

여기서:

$$R_0 = \left(11.4 + K \cdot B_z\right) P_{\text{dyn}}^{-1/6.6}$$

$B_z \geq 0$일 때 $K = 0.013$, $B_z < 0$일 때 $K = 0.140$이며:

$$\alpha = (0.58 - 0.010 \cdot B_z)(1 + 0.010 \cdot P_{\text{dyn}})$$

$R_0$ (단위: $R_E$)와 정지 궤도(6.6 $R_E$)가 표시된 자기권 단면 모식도를 표시합니다. $R_0 < 6.6 R_E$일 경우, "정지 위성이 자기권계면 밖에서 태양풍에 직접 노출됨"이라는 경고를 표시합니다.

**패널 (c): Dst 예측**

현재 태양풍 데이터를 사용하여 Burton 모델을 앞으로 돌려 향후 6시간의 Dst를 예측합니다:

1. 현재 관측된 Dst 값에서 초기화합니다.
2. 예측 기간 동안 태양풍 조건이 현재 값으로 유지된다고 가정합니다(지속성 예측(Persistence Forecast)) 또는 간단한 추세 외삽을 사용합니다.
3. 현재 Dst와 6시간 예측 궤적을 표시합니다.
4. 폭풍 분류에 따라 색상 코딩: 초록(> $-30$), 노랑($-30$ ~ $-50$), 주황($-50$ ~ $-100$), 빨강($-100$ ~ $-250$), 보라(< $-250$).

**패널 (d): 오로라 확률**

Feldstein 관계식을 사용하여 오로라 타원체(Auroral Oval)의 적도 방향 경계를 추정합니다:

$$\Lambda_{\text{eq}} \approx 67° - 2° \times \text{Kp}$$

여기서 $\Lambda_{\text{eq}}$는 적도 방향 경계의 지자기 위도(단위: 도)입니다. 자기장 모델을 사용하여 지리적 위도로 변환합니다.

표시 항목:
- 현재 Kp 값
- 적도 방향 경계 위도
- 오로라 타원체를 근사 표시한 극지방 투영 지도
- 특정 도시의 오로라 가시성: 관측자가 $\Lambda_{\text{eq}}$의 극 방향에 있고 현지 야간인 경우 가시성 표시

**패널 (e): NOAA 척도 표시기**

색상 코딩된 NOAA 우주 날씨 척도 상태를 표시합니다:

- **G-척도 (지자기, Geomagnetic):** 현재 Kp 기반. G0-G5를 적절한 색상과 설명으로 표시.
- **S-척도 (태양 방사선, Solar Radiation):** GOES >10 MeV 양성자 플럭스 기반. S0-S5 표시.
- **R-척도 (전파 장애, Radio Blackout):** GOES X선 플럭스 기반. R0-R5 표시.

각 표시기는 현재 등급, 설명, 주요 영향을 보여줘야 합니다.

### 3.3 구현 옵션

**옵션 A: 정적 대시보드 (matplotlib)**

각 패널에 서브플롯을 사용하여 matplotlib으로 구현합니다. 정기적으로 업데이트되는 정적 이미지를 생성합니다. 이는 가장 단순한 접근 방식으로 물리를 학습하기에 충분합니다.

레이아웃 제안:
- 3행, 2열
- 좌상단: 태양풍 시계열 (4개 서브플롯 스택)
- 우상단: 자기권계면 모식도
- 좌중단: Dst 관측 + 예측
- 우중단: 오로라 타원체 지도
- 하단: NOAA 척도 (3개 색상 바)

**옵션 B: 인터랙티브 대시보드 (Flask + Plotly)**

백엔드에 Flask, 인터랙티브 플롯에 Plotly를 사용하여 웹 애플리케이션을 구축합니다. 이를 통해:
- 시계열에서 확대/축소 및 패닝
- 정확한 값 호버링
- 설정 가능한 간격으로 자동 새로고침
- 모바일 반응형 레이아웃

Flask 백엔드는 데이터(OMNI 파일 또는 데이터베이스에서 로드)를 제공하고 모델을 실행합니다. 프론트엔드는 Plotly.js를 사용하여 플롯을 렌더링합니다.

### 3.4 데이터 소스

개발 및 테스트를 위해, 알려진 폭풍 이벤트(예: 2015년 성 패트릭의 날 폭풍, 2015년 3월 17-22일)에 대한 역사적 OMNI 시간별 데이터를 사용합니다. 이를 통해 모델 출력을 알려진 관측값과 검증할 수 있습니다.

"라이브" 시연을 위해서는 최신 OMNI 데이터를 다운로드하거나 DSCOVR 실시간 태양풍 데이터 피드(SWPC에서 JSON 형식으로 제공)를 사용합니다.

### 3.5 검증

- 테스트 폭풍에 대해 Burton 모델 Dst 예측과 관측된 Dst를 비교합니다.
- Shue 자기권계면 모델이 물리적으로 합리적인 값을 제공하는지 확인합니다: 조용한 시기 $R_0 \approx 10-11 R_E$, 폭풍 시 $R_0 \approx 6-8 R_E$.
- 테스트 이벤트 동안 보고된 오로라 관측과 오로라 타원체 위도를 비교합니다.

---

## 4. 프로젝트 확장

이 확장들은 각 프로젝트의 물리적 깊이를 더하고 복잡도를 높입니다.

### 4.1 프로젝트 1 확장

**확장 1a: Newell 결합 함수(Coupling Function)**

Burton 모델의 단순한 $E_y$ 구동을 Newell et al. (2007) 결합 함수로 대체합니다:

$$\frac{d\Phi_{MP}}{dt} = v^{4/3} B_T^{2/3} \sin^{8/3}(\theta_c/2)$$

여기서 $B_T = \sqrt{B_y^2 + B_z^2}$는 횡방향 IMF, $\theta_c = \arctan(B_y / B_z)$는 IMF 시계 각도(GSM 기준)입니다. 이 함수는 주간 자기권계면에서의 자기 플럭스 개방 속도를 더 잘 나타내며, 자기권 활동의 최고 단일 매개변수 예측 변수로 간주됩니다. 이 구동을 사용했을 때와 원래 $E_y$를 사용했을 때의 Dst 예측을 비교합니다.

**확장 1b: Akasofu 엡실론 매개변수**

Akasofu $\epsilon$ 매개변수와 비교합니다:

$$\epsilon = \frac{4\pi}{\mu_0} v B^2 \sin^4(\theta_c/2) l_0^2$$

여기서 $l_0 \approx 7 R_E$는 경험적 길이 척도입니다. $\epsilon$ 매개변수는 와트 단위를 가지며 태양풍에서 자기권으로의 총 전력 입력을 추정합니다. 전형적인 값: $10^{10}$ W (조용한 시기) ~ $10^{13}$ W (극한 폭풍).

### 4.2 프로젝트 2 확장

**확장 2a: 코러스 파동에 의한 국소 가속**

휘슬러 모드 코러스(Whistler-Mode Chorus) 파동에 의한 전자의 국소 가속을 나타내는 소스 항을 추가합니다. 단순화된 매개변수화:

$$S(L, \text{Kp}) = \begin{cases} S_0 \times (\text{Kp}/5)^2 \times \exp\left[-\frac{(L - L_{\text{acc}})^2}{2 \times 0.5^2}\right] & \text{for } L > L_{pp} \\ 0 & \text{for } L \leq L_{pp} \end{cases}$$

$L_{\text{acc}} = 4.5$이며, $S_0$는 현실적인 가속 시상수(강한 코러스 활동에 대해 ~1일)를 생성하도록 선택합니다. 이는 폭풍 회복 중 $L \sim 4-5$에서 PSD의 국소 최대값을 만들어내며, Van Allen Probes에서 관측된 내부 가속의 특징입니다.

국소 가속 유무에 따른 시뮬레이션을 비교합니다. PSD 프로파일이 국소 최대값(내부 가속의 증거)을 형성합니까, 아니면 단조 증가 프로파일(방사 확산 지배의 증거)을 보입니까?

**확장 2b: 자기권계면 그림자 효과(Magnetopause Shadowing)**

폭풍 중 자기권계면이 낮은 $L$-껍질로 압축될 때, $L > R_0$(자기권계면 거리) 위치의 입자들은 열린 드리프트 경로(Open Drift Paths)에서 손실됩니다. 이를 다음과 같이 구현합니다:

$$\tau_{\text{mp}}(L, t) = \begin{cases} 0.5 \text{ hr} & \text{if } L > R_0(t) \\ \infty & \text{if } L \leq R_0(t) \end{cases}$$

여기서 $R_0(t)$는 관측된 $P_{\text{dyn}}$와 $B_z$로 구동되는 Shue 자기권계면 모델에서 계산됩니다. 파동 구동 손실과 결합합니다: $1/\tau_{\text{total}} = 1/\tau_{\text{wave}} + 1/\tau_{\text{mp}}$.

이는 폭풍 주상에서 관측되는 극적인 외부 존 플럭스 감소를 재현합니다.

### 4.3 프로젝트 3 확장

**확장 3a: 위성 궤도 오버레이**

대시보드에 위성 궤도를 추가합니다:
- **ISS** (고도 ~420 km, 경사각 51.6°): ISS가 남대서양 이상 지역(SAA, South Atlantic Anomaly)을 통과할 때를 표시합니다 — 지리적 축과 자기 축의 편차로 인해 내부 복사대가 저고도로 낮아지는 지역. SAA 경계: 경도 약 0°-60°W, 위도 약 15°-45°S.
- **GPS 위성군** (고도 ~20,200 km, $L \sim 4.2$): GPS 위성이 외부 복사대 중심부에 있어 단일 이벤트 업셋(Single-Event Upsets)이 발생할 수 있는 시기를 표시합니다.
- **정지 위성(GEO)** (고도 ~35,786 km, $L = 6.6$): 폭풍 압축 중 자기권계면 밖에 있을 수 있는 시기를 표시합니다.

**확장 3b: GIC 위험 표시기**

지자기 유도 전류(GIC, Geomagnetically Induced Current) 위험 패널을 추가합니다. GIC 위험은 지상에서의 $|dB/dt|$와 상관 관계가 있으며, 이는 AE 및 Dst 변화율과도 상관 관계가 있습니다. 간단한 프록시:

$$\text{GIC risk} \propto \left|\frac{d\text{Dst}}{dt}\right| + 0.1 \times \text{AE}$$

전력망 운영자 경고를 위한 임계값이 있는 색상 코딩 미터로 표시합니다.

---

## 5. 참고 자료 및 데이터 소스

### 5.1 데이터 포털

| 자료 | URL | 설명 |
|------|-----|------|
| OMNI (NASA/GSFC) | omniweb.gsfc.nasa.gov | 뱃머리 충격파 위치 다중 소스 태양풍 데이터, 1963년-현재 |
| GOES 실시간 | swpc.noaa.gov/products/goes-magnetometer | 정지 궤도에서 실시간 X선, 양성자, 자기장 |
| SuperMAG | supermag.jhuapl.edu | 지상 자력계 네트워크, 300개 이상 관측소 |
| SWPC (NOAA) | swpc.noaa.gov | 운영 예보, 경보, 알림 |
| Van Allen Probes | rbsp-ect.newmexiconsortium.org | 복사대 입자 및 자기장 데이터 (2012-2019) |
| SDO/HMI | jsoc.stanford.edu | 태양 자기도 및 영상 |
| CDAWeb | cdaweb.gsfc.nasa.gov | 다중 임무 우주 물리 데이터 저장소 |
| DSCOVR | swpc.noaa.gov/products/real-time-solar-wind | L1에서 실시간 태양풍 |

### 5.2 Python 패키지

| 패키지 | 목적 |
|--------|------|
| spacepy | 우주 물리 데이터 분석, 좌표 변환, 복사대 모델 |
| sunpy | 태양 데이터 접근 및 분석 |
| geopack | 자기장 모델(Tsyganenko), 좌표 변환 |
| cdflib | NASA CDF 데이터 파일 읽기 |
| astropy | 일반 천문학 유틸리티, 단위, 좌표 |
| heliopy (deprecated) / astrospice | 태양권 물리(Heliophysics) 데이터 다운로드 |
| kamodo | CCMC 모델 출력 접근 및 시각화 |

### 5.3 주요 참고 문헌

- Burton, R. K., McPhetres, R. L., & Russell, C. T. (1975). An empirical relationship between interplanetary conditions and Dst. Journal of Geophysical Research, 80(31), 4204-4214.
- Brautigam, D. H., & Albert, J. M. (2000). Radial diffusion analysis of outer radiation belt electrons during the October 9, 1990, magnetic storm. Journal of Geophysical Research, 105(A1), 291-309.
- Shue, J.-H., et al. (1998). Magnetopause location under extreme solar wind conditions. Journal of Geophysical Research, 103(A8), 17691-17700.
- Newell, P. T., et al. (2007). A nearly universal solar wind-magnetosphere coupling function. Journal of Geophysical Research, 112, A01206.
- Camporeale, E., Wing, S., & Johnson, J. R. (Eds.) (2018). Machine Learning Techniques for Space Weather. Elsevier.

---

## 연습 문제

### 연습 1: Burton 모델 기초 구현

수동으로 구성한 짧은 태양풍 시계열을 사용하여 Burton 모델을 구현하고 검증합니다.

**단계:**

1. 다음 구조를 가진 합성 48시간 태양풍 데이터셋을 생성합니다:
   - 시간 0-11: 조용한 조건 — $v = 400$ km/s, $n = 5$ cm$^{-3}$, $B_z = +2$ nT
   - 시간 12-24: 폭풍 주상(Storm Main Phase) — $v = 600$ km/s, $n = 15$ cm$^{-3}$, $B_z = -10$ nT
   - 시간 25-48: 회복 — $v = 450$ km/s, $n = 8$ cm$^{-3}$, $B_z = +1$ nT

2. 각 시간마다 $P_{\text{dyn}}$과 $E_y$를 계산합니다. 압력 보정(Pressure Correction)을 적용하여 $\text{Dst}^*$를 구합니다.

3. Burton ODE를 $\Delta t = 1$시간 오일러(Euler) 방법으로 수치 적분합니다. Burton의 표준 매개변수($a = -4.5$ nT/(mV/m·hr), $\tau = 7.7$ hr, $E_c = 0.5$ mV/m)를 사용합니다.

4. $\text{Dst}^*$를 Dst로 역변환하고, 태양풍 입력과 함께 전체 시계열을 플롯합니다.

5. 결과를 검증합니다: Dst 최솟값은 남향 $B_z$ 시작 후 몇 시간 뒤에 나타나야 하고, 회복은 ~8시간 시상수로 근사 지수 형태여야 합니다.

**예상 결과:** 명확한 폭풍 시 Dst 감소(대략 $-50$~$-100$ nT 범위)와 지수 회복이 나타나야 하며, 이를 통해 실제 데이터를 사용하기 전에 구현이 물리적으로 올바른지 확인할 수 있습니다.

---

### 연습 2: Dst 예측 모델의 통계적 평가

프로젝트 1을 확장하여 Burton 모델과 지속성 예측(Persistence Forecast)을 체계적으로 통계 비교합니다.

**단계:**

1. OMNIWeb에서 2010-2020년 OMNI 시간별 데이터를 다운로드합니다. 채움 값(Fill Value) 마스크를 사용하여 결측값을 처리합니다.

2. 1.3절에서 설명한 Burton 모델을 구현합니다.

3. 단순 **지속성 예측(Persistence Forecast)**을 구현합니다: $t+1$ 시각의 Dst가 $t$ 시각의 Dst와 같다고 예측합니다(즉, $\hat{D}(t+1) = D(t)$). 이는 예측에서 흔히 사용되는 기준선입니다.

4. 두 모델 모두 2015-2020년 테스트 기간에 대해 다음 기술 점수(Skill Score)를 계산합니다:
   - RMSE
   - 평균 절대 오차(MAE, Mean Absolute Error)
   - 피어슨 상관계수(Pearson Correlation Coefficient) $r$
   - 폭풍 탐지를 위한 하이드케 기술 점수(HSS, Heidke Skill Score): Dst < $-50$ nT를 폭풍으로 정의

5. 두 모델을 관측값과 비교하는 테일러 다이어그램(Taylor Diagram)을 작성합니다. 테일러 다이어그램에서 원점으로부터의 반경 거리는 정규화된 표준 편차, 각도는 상관관계, "관측" 기준점으로부터의 거리는 중심화된 RMSE를 나타냅니다.

6. 폭풍 강도에 따른 성능 변화를 분석합니다. 모든 폭풍 이벤트를 최대 Dst 강도($-50$~$-100$ nT, $-100$~$-200$ nT, < $-200$ nT)로 구분하고, 각 구간에서 RMSE를 별도로 계산합니다.

**토론:** Burton 모델이 지속성 예측보다 우수합니까? 어떤 폭풍 강도 구간에서 Burton 모델이 가장 큰 개선을 보입니까? 짧은 예보 리드 타임에서 단순 지속성이 놀랍도록 경쟁력 있는 이유는 무엇입니까?

---

### 연습 3: 실제 폭풍에 대한 복사대 반응

2003년 할로윈 폭풍(Halloween Storm)(2003년 10월 29-31일)에 대해 프로젝트 2의 방사 확산(Radial Diffusion) 솔버를 실행하고 복사대 역학을 분석합니다.

**단계:**

1. GFZ Potsdam 데이터 서비스(kp.gfz-potsdam.de)에서 2003년 10월 25일 - 11월 5일의 시간별 Kp 값을 다운로드합니다.

2. Brautigam & Albert 확산 계수와 구간별(Piecewise) 손실 시상수 모델을 사용하여 2.3절과 동일하게 솔버를 설정합니다.

3. 전체 12일 기간에 대해 시뮬레이션을 실행합니다. 매 시간마다 전체 $f(L, t)$ 배열을 저장합니다.

4. 다음 4패널 그림을 작성합니다:
   - 패널 1: Kp 시계열 (상단 패널)
   - 패널 2: $f(L, t)$ 색상 지도 (로그 스케일) — `pcolormesh`와 `norm=LogNorm()` 사용
   - 패널 3: 네 시점에서의 $f(L)$ 프로파일: 조용한 시기(10월 25일), 폭풍 시작(10월 29일 0 UT), 폭풍 최대(10월 29일 12 UT), 회복(11월 3일)
   - 패널 4: $L = 4.0$, $L = 5.0$, $L = 6.0$에서의 $f(t)$

5. 시뮬레이션이 폭풍 주상에서 높은 $L$ 값의 플럭스 감소(자기권계면 그림자 효과(Magnetopause Shadowing)의 특징)를 보이는지 확인합니다. 명시적으로 구현하지 않더라도 증강된 손실 항이 어느 정도의 감소를 만들어내야 합니다.

6. 회복 시점의 시뮬레이션된 PSD 프로파일과 조용한 시기 프로파일을 비교합니다. 회복이 완전합니까? 그 이유는 무엇입니까?

**제출물:** 4패널 그림과 관찰된 복사대 역학에 대한 200자 이상의 해석문.

---

### 연습 4: 2015년 성 패트릭의 날 폭풍 우주 날씨 대시보드

2015년 성 패트릭의 날 폭풍(St. Patrick's Day Storm)(2015년 3월 14-22일)에 대한 역사적 OMNI 데이터를 기반으로 정적 matplotlib 대시보드(3.3절 옵션 A)를 구축합니다.

**단계:**

1. OMNIWeb에서 2015년 3월 14-22일 OMNI 시간별 데이터를 다운로드합니다. 필요한 열: $v$, $n$, $B$, $B_y$, $B_z$ (GSM), Dst, Kp.

2. 3.2절에서 설명한 5개 패널을 모두 구현합니다:
   - 패널 a: 4변수 태양풍 시계열($v$, $n$, $|B|$, $B_z$)
   - 패널 b: 이벤트 기간 동안 Shue 자기권계면 이격 거리(Standoff Distance) $R_0(t)$, $6.6 R_E$ (정지 궤도)에 수평선 표시
   - 패널 c: Burton 모델 예측값과 겹쳐 표시된 관측 Dst
   - 패널 d: Feldstein 관계식으로부터 구한 오로라 적도 방향 경계 위도
   - 패널 e: Kp의 계단 함수로 표현된 NOAA G-척도 레벨

3. 폭풍 급시작(SSC, Storm Sudden Commencement)과 Dst 최솟값을 표시하는 수직 점선을 추가합니다.

4. 각 패널에서 임계값이 초과되는 순간에 주석을 달아 표시합니다:
   - 태양풍 속도 > 600 km/s
   - $B_z < -10$ nT가 3시간 이상 지속
   - $R_0 < 8 R_E$
   - 오로라 경계 < 55° 위도

5. 폭풍 기간 동안 Burton 모델 Dst 예측과 관측 Dst 간의 전체 RMSE를 계산합니다.

**심화 목표:** Newell 결합 함수(Coupling Function) $d\Phi_{MP}/dt$와 $E_y$를 비교하는 여섯 번째 패널을 추가합니다. 어느 것이 관측된 Dst 감소와 더 잘 상관되는지 논의합니다.

---

### 연습 5: 결합 함수 비교 (고급)

2000-2020년 전체 OMNI 데이터셋에서 세 가지 태양풍-자기권 결합 함수(Coupling Function)를 Burton 모델의 구동원으로 체계적으로 비교하고, 어느 것이 다양한 폭풍 범주에서 가장 좋은 Dst 예측을 생성하는지 정량화합니다.

**단계:**

1. 세 가지 결합 함수를 모두 구현합니다:
   - $E_y = -v \cdot B_z$ (표준 Burton 구동원, mV/m)
   - Newell 함수: $\frac{d\Phi_{MP}}{dt} = v^{4/3} B_T^{2/3} \sin^{8/3}(\theta_c/2)$
   - Akasofu 엡실론: $\epsilon = \frac{4\pi}{\mu_0} v B^2 \sin^4(\theta_c/2) l_0^2$ ($l_0 = 7 R_E$)

2. 각 결합 함수에 대해, 훈련 기간(2000-2014년) RMSE를 최소화하여 주입 효율 매개변수 $a$ (Burton 매개변수)를 재피팅합니다. $\tau = 7.7$ hr과 $E_c = 0.5$ mV/m는 고정하거나($E_c$도 임계값으로 최적화 가능).

3. 세 가지 피팅된 모델을 2015-2020년 테스트 기간에서 평가합니다. 다음 폭풍 강도 구간별로 RMSE와 $r$을 별도 계산합니다:
   - 약한 폭풍($-30 < \text{Dst} < -50$ nT)
   - 중간 폭풍($-50 < \text{Dst} < -100$ nT)
   - 강한 폭풍(Dst < $-100$ nT)

4. 폭풍 범주와 결합 함수별 RMSE를 나타내는 막대 차트를 작성합니다. 강한 폭풍에서 어떤 결합 함수가 가장 좋은 성능을 보입니까?

5. 모델 간 불일치가 가장 큰 사례를 분석합니다: Newell 기반 예측이 $E_y$ 기반 예측보다 훨씬 좋거나(또는 나쁜) 3-5개 이벤트를 찾습니다. 해당 이벤트에서 태양풍 조건의 특이점은 무엇입니까?

**토론:** Newell 함수는 $E_y$보다 IMF 시계 각도(Clock Angle)에 더 큰 가중치를 부여합니다. 결과를 바탕으로, 어떤 태양풍 조건에서 시계 각도가 Dst 예측에 가장 중요합니까?

---

**이전**: [우주 날씨를 위한 AI/ML](./15_AI_ML_for_Space_Weather.md)
