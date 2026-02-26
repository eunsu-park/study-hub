# 21. 시계열 모형 (Time Series Models)

[이전: 시계열 분석 기초](./20_Time_Series_Basics.md) | [다음: 다변량 분석](./22_Multivariate_Analysis.md)

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. AR(p) 모형의 구조와 정상성(Stationarity) 조건 설명
2. MA(q) 모형이 과거 충격(Shock)의 가중 합으로 시계열을 표현하는 방식 서술
3. ACF/PACF 패턴을 기반으로 AR, MA, ARMA, ARIMA 모형 구별
4. Box-Jenkins 방법론을 적용하여 시계열 모형 식별, 추정, 진단
5. AIC와 BIC 정보 기준으로 후보 모형 비교
6. 롤링(Rolling) 및 고정 호라이즌(Fixed-Horizon) 예측 구현, RMSE와 MAE로 성능 평가
7. 정상성 검정부터 예측까지 ARIMA 모델링 전체 파이프라인 추적
8. 벡터 자기회귀(VAR) 모형을 다변량 시계열에 적용하고, 그랜저 인과성(Granger Causality), 충격 반응 분석(IRF), 분산 분해(FEVD) 수행

---

시계열 데이터는 어디에나 있습니다 — 주가, 센서 측정값, 웹 트래픽 — 그러나 단일 스냅샷만으로는 다음에 무엇이 올지 알 수 없습니다. AR, MA, ARMA, ARIMA 모형은 순차적 관측값에 숨겨진 시간적 구조를 포착하는 체계적인 프레임워크를 제공하며, 미래 값을 예측하고 예측의 불확실성을 정량화할 수 있게 합니다. 시계열 데이터를 다루는 데이터 과학자에게 이러한 모형의 식별과 진단을 위한 Box-Jenkins 방법론 습득은 핵심 기초 역량입니다.

---

## 1. AR 모형 (AutoRegressive Model)

### 1.1 AR(p) 모형 정의

**AR(p) 모형**:
$$Y_t = c + \phi_1 Y_{t-1} + \phi_2 Y_{t-2} + ... + \phi_p Y_{t-p} + \epsilon_t$$

여기서:
- c: 상수 (절편)
- φᵢ: AR 계수
- εₜ: 백색 잡음, εₜ ~ WN(0, σ²)

**정상성 조건**: 특성방정식 1 - φ₁z - φ₂z² - ... - φₚzᵖ = 0의 모든 근이 단위원 밖에 있어야 함

### 1.2 AR(1) 모형

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import arma_generate_sample
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

np.random.seed(42)

def simulate_ar1(phi, n=500, c=0, sigma=1):
    """AR(1) 시뮬레이션: Y_t = c + φ*Y_{t-1} + ε_t"""
    y = np.zeros(n)
    y[0] = np.random.normal(0, sigma / np.sqrt(1 - phi**2))  # 정상 초기값

    for t in range(1, n):
        y[t] = c + phi * y[t-1] + np.random.normal(0, sigma)

    return y

# 다양한 φ 값으로 AR(1) 시뮬레이션
phi_values = [0.9, 0.5, -0.5, -0.9]
fig, axes = plt.subplots(len(phi_values), 3, figsize=(15, 12))

for i, phi in enumerate(phi_values):
    y = simulate_ar1(phi, n=500)

    # 시계열
    axes[i, 0].plot(y[:200])
    axes[i, 0].set_title(f'AR(1), φ = {phi}')
    axes[i, 0].grid(True, alpha=0.3)

    # ACF
    plot_acf(y, lags=20, ax=axes[i, 1], alpha=0.05)
    axes[i, 1].set_title(f'ACF (φ = {phi})')

    # PACF
    plot_pacf(y, lags=20, ax=axes[i, 2], alpha=0.05, method='ywm')
    axes[i, 2].set_title(f'PACF (φ = {phi})')

plt.tight_layout()
plt.show()

# AR(1)의 이론적 ACF
print("AR(1) 이론적 ACF:")
for phi in phi_values:
    acf_theory = [phi**k for k in range(6)]
    print(f"  φ = {phi}: ρ(k) = {acf_theory}")
```

### 1.3 AR(2) 모형

```python
def simulate_ar2(phi1, phi2, n=500, c=0, sigma=1):
    """AR(2) 시뮬레이션"""
    y = np.zeros(n)

    for t in range(2, n):
        y[t] = c + phi1 * y[t-1] + phi2 * y[t-2] + np.random.normal(0, sigma)

    return y

# AR(2) 예시
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 실근 케이스: φ₁ = 0.5, φ₂ = 0.3
y1 = simulate_ar2(0.5, 0.3, n=500)
axes[0, 0].plot(y1[:200])
axes[0, 0].set_title('AR(2): φ₁=0.5, φ₂=0.3 (실근)')
axes[0, 0].grid(True, alpha=0.3)
plot_acf(y1, lags=20, ax=axes[0, 1], alpha=0.05)
plot_pacf(y1, lags=20, ax=axes[0, 2], alpha=0.05, method='ywm')

# 복소근 케이스: φ₁ = 1.0, φ₂ = -0.5 (진동 패턴)
y2 = simulate_ar2(1.0, -0.5, n=500)
axes[1, 0].plot(y2[:200])
axes[1, 0].set_title('AR(2): φ₁=1.0, φ₂=-0.5 (복소근, 진동)')
axes[1, 0].grid(True, alpha=0.3)
plot_acf(y2, lags=20, ax=axes[1, 1], alpha=0.05)
plot_pacf(y2, lags=20, ax=axes[1, 2], alpha=0.05, method='ywm')

plt.tight_layout()
plt.show()

# 정상성 조건 확인
def check_ar2_stationarity(phi1, phi2):
    """AR(2) 정상성 조건 확인"""
    # 조건: |φ₂| < 1, φ₂ + φ₁ < 1, φ₂ - φ₁ < 1
    cond1 = abs(phi2) < 1
    cond2 = phi2 + phi1 < 1
    cond3 = phi2 - phi1 < 1

    print(f"AR(2) φ₁={phi1}, φ₂={phi2}")
    print(f"  |φ₂| < 1: {cond1} ({abs(phi2):.2f})")
    print(f"  φ₂ + φ₁ < 1: {cond2} ({phi2 + phi1:.2f})")
    print(f"  φ₂ - φ₁ < 1: {cond3} ({phi2 - phi1:.2f})")
    print(f"  정상성: {cond1 and cond2 and cond3}")
    return cond1 and cond2 and cond3

check_ar2_stationarity(0.5, 0.3)
check_ar2_stationarity(1.0, -0.5)
check_ar2_stationarity(0.6, 0.5)  # 비정상
```

### 1.4 statsmodels로 AR 추정

```python
# AR(1) 데이터 생성 및 추정
np.random.seed(42)
true_phi = 0.7
y_ar1 = simulate_ar1(true_phi, n=300)

# ARIMA(1,0,0) = AR(1) 적합
model = ARIMA(y_ar1, order=(1, 0, 0))
result = model.fit()

print("=== AR(1) 모형 추정 결과 ===")
print(f"True φ = {true_phi}")
print(result.summary())

# AR(2) 데이터 생성 및 추정
true_phi1, true_phi2 = 0.5, 0.3
y_ar2 = simulate_ar2(true_phi1, true_phi2, n=300)

model_ar2 = ARIMA(y_ar2, order=(2, 0, 0))
result_ar2 = model_ar2.fit()

print("\n=== AR(2) 모형 추정 결과 ===")
print(f"True φ₁ = {true_phi1}, φ₂ = {true_phi2}")
print(result_ar2.summary())
```

---

## 2. MA 모형 (Moving Average Model)

### 2.1 MA(q) 모형 정의

**MA(q) 모형**:
$$Y_t = c + \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + ... + \theta_q \epsilon_{t-q}$$

여기서:
- c: 상수 (평균)
- θᵢ: MA 계수
- εₜ: 백색 잡음

**특징**: MA 모형은 항상 정상 (유한한 과거 충격의 조합)

### 2.2 MA(1) 모형

```python
def simulate_ma(theta_list, n=500, c=0, sigma=1):
    """MA(q) 시뮬레이션"""
    q = len(theta_list)
    epsilon = np.random.normal(0, sigma, n + q)
    y = np.zeros(n)

    for t in range(n):
        y[t] = c + epsilon[t + q]
        for i, theta in enumerate(theta_list):
            y[t] += theta * epsilon[t + q - i - 1]

    return y

# MA(1) 시뮬레이션
theta_values = [0.8, 0.3, -0.3, -0.8]
fig, axes = plt.subplots(len(theta_values), 3, figsize=(15, 12))

for i, theta in enumerate(theta_values):
    y = simulate_ma([theta], n=500)

    axes[i, 0].plot(y[:200])
    axes[i, 0].set_title(f'MA(1), θ = {theta}')
    axes[i, 0].grid(True, alpha=0.3)

    plot_acf(y, lags=20, ax=axes[i, 1], alpha=0.05)
    axes[i, 1].set_title(f'ACF (θ = {theta})')

    plot_pacf(y, lags=20, ax=axes[i, 2], alpha=0.05, method='ywm')
    axes[i, 2].set_title(f'PACF (θ = {theta})')

plt.tight_layout()
plt.show()

# MA(1)의 이론적 ACF
print("MA(1) 이론적 ACF:")
for theta in theta_values:
    rho1 = theta / (1 + theta**2)
    print(f"  θ = {theta}: ρ(1) = {rho1:.3f}, ρ(k>1) = 0")
```

### 2.3 MA(2) 모형

```python
# MA(2) 시뮬레이션
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# MA(2) 예시 1
y1 = simulate_ma([0.5, 0.3], n=500)
axes[0, 0].plot(y1[:200])
axes[0, 0].set_title('MA(2): θ₁=0.5, θ₂=0.3')
axes[0, 0].grid(True, alpha=0.3)
plot_acf(y1, lags=20, ax=axes[0, 1], alpha=0.05)
plot_pacf(y1, lags=20, ax=axes[0, 2], alpha=0.05, method='ywm')

# MA(2) 예시 2
y2 = simulate_ma([1.0, -0.5], n=500)
axes[1, 0].plot(y2[:200])
axes[1, 0].set_title('MA(2): θ₁=1.0, θ₂=-0.5')
axes[1, 0].grid(True, alpha=0.3)
plot_acf(y2, lags=20, ax=axes[1, 1], alpha=0.05)
plot_pacf(y2, lags=20, ax=axes[1, 2], alpha=0.05, method='ywm')

plt.tight_layout()
plt.show()
```

### 2.4 statsmodels로 MA 추정

```python
# MA(1) 데이터 생성 및 추정
np.random.seed(42)
true_theta = 0.6
y_ma1 = simulate_ma([true_theta], n=300)

# ARIMA(0,0,1) = MA(1) 적합
model_ma1 = ARIMA(y_ma1, order=(0, 0, 1))
result_ma1 = model_ma1.fit()

print("=== MA(1) 모형 추정 결과 ===")
print(f"True θ = {true_theta}")
print(result_ma1.summary())

# MA(2) 추정
true_theta1, true_theta2 = 0.5, 0.3
y_ma2 = simulate_ma([true_theta1, true_theta2], n=300)

model_ma2 = ARIMA(y_ma2, order=(0, 0, 2))
result_ma2 = model_ma2.fit()

print("\n=== MA(2) 모형 추정 결과 ===")
print(f"True θ₁ = {true_theta1}, θ₂ = {true_theta2}")
print(result_ma2.summary())
```

---

## 3. ARMA 모형

### 3.1 ARMA(p,q) 정의

**ARMA(p,q) 모형**:
$$Y_t = c + \sum_{i=1}^{p} \phi_i Y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

### 3.2 ARMA(1,1) 예시

```python
# statsmodels의 arma_generate_sample 사용
np.random.seed(42)

# ARMA(1,1): φ=0.7, θ=0.4
ar = [1, -0.7]  # [1, -φ₁]
ma = [1, 0.4]   # [1, θ₁]

y_arma11 = arma_generate_sample(ar, ma, nsample=500, scale=1, burnin=100)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(y_arma11[:200])
axes[0].set_title('ARMA(1,1): φ=0.7, θ=0.4')
axes[0].grid(True, alpha=0.3)

plot_acf(y_arma11, lags=20, ax=axes[1], alpha=0.05)
axes[1].set_title('ACF')

plot_pacf(y_arma11, lags=20, ax=axes[2], alpha=0.05, method='ywm')
axes[2].set_title('PACF')

plt.tight_layout()
plt.show()

# ARMA(1,1) 추정
model_arma = ARIMA(y_arma11, order=(1, 0, 1))
result_arma = model_arma.fit()

print("=== ARMA(1,1) 모형 추정 결과 ===")
print("True: φ=0.7, θ=0.4")
print(result_arma.summary())
```

---

## 4. ARIMA 모형

### 4.1 ARIMA(p,d,q) 정의

**ARIMA(p,d,q)**: Integrated ARMA - 차분을 포함

$$\nabla^d Y_t = c + \sum_{i=1}^{p} \phi_i \nabla^d Y_{t-i} + \epsilon_t + \sum_{j=1}^{q} \theta_j \epsilon_{t-j}$$

여기서 ∇ᵈ는 d차 차분

### 4.2 Box-Jenkins 방법론

```python
def box_jenkins_methodology():
    """Box-Jenkins 방법론 설명"""
    print("""
    =================================================
    Box-Jenkins 방법론 (ARIMA 모델링)
    =================================================

    1단계: 식별 (Identification)
    ───────────────────────────
    - 시계열 플롯으로 패턴 확인
    - 정상성 검정 (ADF, KPSS)
    - 필요시 차분하여 정상화 → d 결정
    - 정상화된 시계열의 ACF/PACF 분석 → p, q 후보 결정

    2단계: 추정 (Estimation)
    ───────────────────────────
    - 후보 모형들의 파라미터 추정 (MLE)
    - 정보 기준 비교 (AIC, BIC)
    - 파라미터 유의성 확인

    3단계: 진단 (Diagnostic Checking)
    ───────────────────────────
    - 잔차 분석: 백색 잡음인지 확인
    - Ljung-Box 검정
    - 잔차의 ACF/PACF
    - 정규성 검정

    4단계: 예측 (Forecasting)
    ───────────────────────────
    - 점 예측
    - 예측 구간
    - 예측 성능 평가
    """)

box_jenkins_methodology()
```

### 4.3 모델 식별: ACF/PACF 패턴

```python
def identify_model_from_acf_pacf():
    """ACF/PACF 패턴 기반 모델 식별 가이드"""
    guide = """
    =================================================
    ACF/PACF 패턴 기반 모델 식별
    =================================================

    패턴 1: ACF 점진적 감소, PACF lag p 후 절단
    → AR(p) 모형

    패턴 2: ACF lag q 후 절단, PACF 점진적 감소
    → MA(q) 모형

    패턴 3: ACF, PACF 모두 점진적 감소
    → ARMA(p,q) 모형

    패턴 4: ACF 매우 느리게 감소 (선형적)
    → 비정상, 차분 필요 (d 증가)

    패턴 5: ACF 계절 주기로 스파이크
    → SARIMA 고려

    실용적 팁:
    - 절단(cut-off): 갑자기 0이 됨
    - 점진적 감소: 지수적으로 또는 진동하며 감소
    - 95% 신뢰대역 밖 = 유의미
    """
    print(guide)

identify_model_from_acf_pacf()
```

### 4.4 ARIMA 실습: 비정상 시계열

```python
# 비정상 시계열 생성: 랜덤 워크 + 노이즈
np.random.seed(42)
n = 300

# ARIMA(1,1,0) 과정: ΔYₜ = 0.5 * ΔYₜ₋₁ + εₜ
# 즉, Yₜ - Yₜ₋₁ = 0.5*(Yₜ₋₁ - Yₜ₋₂) + εₜ
ar = [1, -0.5]
ma = [1]
y_diff = arma_generate_sample(ar, ma, nsample=n, scale=1, burnin=100)
y_arima110 = np.cumsum(y_diff)  # 누적하여 비정상 시계열 생성

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# 원본 (비정상)
axes[0, 0].plot(y_arima110)
axes[0, 0].set_title('원본 (비정상)')
axes[0, 0].grid(True, alpha=0.3)

plot_acf(y_arima110, lags=20, ax=axes[0, 1], alpha=0.05)
axes[0, 1].set_title('원본 ACF (느린 감소)')

plot_pacf(y_arima110, lags=20, ax=axes[0, 2], alpha=0.05, method='ywm')
axes[0, 2].set_title('원본 PACF')

# 1차 차분 후 (정상)
diff1 = np.diff(y_arima110)
axes[1, 0].plot(diff1)
axes[1, 0].set_title('1차 차분 후 (정상)')
axes[1, 0].grid(True, alpha=0.3)

plot_acf(diff1, lags=20, ax=axes[1, 1], alpha=0.05)
axes[1, 1].set_title('차분 ACF')

plot_pacf(diff1, lags=20, ax=axes[1, 2], alpha=0.05, method='ywm')
axes[1, 2].set_title('차분 PACF (AR(1) 패턴)')

plt.tight_layout()
plt.show()

# ARIMA(1,1,0) 추정
from statsmodels.tsa.stattools import adfuller

print("=== 정상성 검정 ===")
print(f"원본 ADF p-value: {adfuller(y_arima110)[1]:.4f}")
print(f"차분 후 ADF p-value: {adfuller(diff1)[1]:.4f}")

model = ARIMA(y_arima110, order=(1, 1, 0))
result = model.fit()

print("\n=== ARIMA(1,1,0) 추정 결과 ===")
print("True: φ=0.5")
print(result.summary())
```

---

## 5. 모델 선택

### 5.1 정보 기준 (AIC, BIC)

```python
def compare_arima_models(y, max_p=3, max_d=2, max_q=3):
    """
    그리드 서치로 최적 ARIMA 모델 선택
    """
    results = []

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                if p == 0 and q == 0:
                    continue  # 의미 없는 모델 제외
                try:
                    model = ARIMA(y, order=(p, d, q))
                    fit = model.fit()
                    results.append({
                        'order': (p, d, q),
                        'aic': fit.aic,
                        'bic': fit.bic,
                        'llf': fit.llf
                    })
                except:
                    pass

    # 결과 정렬
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('aic')

    return results_df

# 데이터 생성 (ARIMA(1,1,1))
np.random.seed(42)
ar = [1, -0.6]
ma = [1, 0.4]
y_diff = arma_generate_sample(ar, ma, nsample=300, scale=1, burnin=100)
y = np.cumsum(y_diff)

# 모델 비교
print("=== ARIMA 모델 비교 (AIC 기준) ===")
comparison = compare_arima_models(y, max_p=3, max_d=2, max_q=3)
print(comparison.head(10))

# 최적 모델
best_order = comparison.iloc[0]['order']
print(f"\n최적 모델: ARIMA{best_order}")
```

### 5.2 auto_arima (pmdarima)

```python
# pmdarima 설치: pip install pmdarima
try:
    from pmdarima import auto_arima

    # 자동 ARIMA
    auto_model = auto_arima(
        y,
        start_p=0, max_p=3,
        start_q=0, max_q=3,
        d=None,  # 자동 결정
        seasonal=False,
        trace=True,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True
    )

    print("\n=== auto_arima 결과 ===")
    print(auto_model.summary())

except ImportError:
    print("pmdarima가 설치되지 않았습니다.")
    print("설치: pip install pmdarima")
```

---

## 6. 모델 진단

### 6.1 잔차 분석

```python
def diagnose_arima_model(result, title=''):
    """
    ARIMA 모델 진단
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    residuals = result.resid

    # 잔차 시계열
    axes[0, 0].plot(residuals)
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_title('잔차 시계열')
    axes[0, 0].grid(True, alpha=0.3)

    # 잔차 히스토그램
    axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    from scipy.stats import norm
    axes[0, 1].plot(x, norm.pdf(x, residuals.mean(), residuals.std()), 'r-', lw=2)
    axes[0, 1].set_title('잔차 분포')
    axes[0, 1].grid(True, alpha=0.3)

    # 잔차 ACF
    plot_acf(residuals, lags=20, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('잔차 ACF')

    # Q-Q Plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')

    plt.suptitle(f'모델 진단: {title}', y=1.02)
    plt.tight_layout()
    plt.show()

    # Ljung-Box 검정
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print("\n=== Ljung-Box 검정 ===")
    print(lb_result)
    print("\np-value > 0.05: 잔차가 백색 잡음 (모델 적합)")

# 진단 실행
model_final = ARIMA(y, order=(1, 1, 1))
result_final = model_final.fit()
diagnose_arima_model(result_final, 'ARIMA(1,1,1)')
```

### 6.2 표준화 잔차

```python
def standardized_residual_analysis(result):
    """표준화 잔차 분석"""
    residuals = result.resid
    std_residuals = residuals / residuals.std()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 표준화 잔차
    axes[0].plot(std_residuals)
    axes[0].axhline(0, color='k', linestyle='-')
    axes[0].axhline(2, color='r', linestyle='--', label='±2σ')
    axes[0].axhline(-2, color='r', linestyle='--')
    axes[0].axhline(3, color='orange', linestyle=':', label='±3σ')
    axes[0].axhline(-3, color='orange', linestyle=':')
    axes[0].set_title('표준화 잔차')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 이상치 비율
    outliers_2sigma = np.sum(np.abs(std_residuals) > 2) / len(std_residuals)
    outliers_3sigma = np.sum(np.abs(std_residuals) > 3) / len(std_residuals)

    text = f"|z| > 2: {outliers_2sigma*100:.1f}% (예상: 4.6%)\n"
    text += f"|z| > 3: {outliers_3sigma*100:.1f}% (예상: 0.3%)"
    axes[1].text(0.1, 0.5, text, transform=axes[1].transAxes, fontsize=12,
                 verticalalignment='center')
    axes[1].axis('off')
    axes[1].set_title('이상치 비율')

    plt.tight_layout()
    plt.show()

standardized_residual_analysis(result_final)
```

---

## 7. 예측 (Forecasting)

### 7.1 점 예측과 구간 예측

```python
# 예측
forecast_steps = 30

# 예측 수행
forecast = result_final.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int(alpha=0.05)

# 시각화
fig, ax = plt.subplots(figsize=(14, 6))

# 원본 데이터 (마지막 100개)
ax.plot(range(len(y)-100, len(y)), y[-100:], 'b-', label='관측값')

# 예측
forecast_index = range(len(y), len(y) + forecast_steps)
ax.plot(forecast_index, forecast_mean, 'r-', label='예측')

# 신뢰구간
ax.fill_between(forecast_index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1],
                color='r', alpha=0.2, label='95% 예측 구간')

ax.axvline(len(y), color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('시간')
ax.set_ylabel('값')
ax.set_title('ARIMA 예측')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 예측 결과
print("=== 예측 결과 (처음 10개) ===")
forecast_df = pd.DataFrame({
    '예측값': forecast_mean[:10],
    '하한': forecast_ci.iloc[:10, 0],
    '상한': forecast_ci.iloc[:10, 1]
})
print(forecast_df)
```

### 7.2 예측 성능 평가

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_forecast(y_train, y_test, order):
    """
    예측 성능 평가

    Parameters:
    -----------
    y_train : array
        훈련 데이터
    y_test : array
        테스트 데이터
    order : tuple
        ARIMA 차수 (p, d, q)
    """
    # 모델 적합
    model = ARIMA(y_train, order=order)
    result = model.fit()

    # 예측
    forecast = result.get_forecast(steps=len(y_test))
    y_pred = forecast.predicted_mean

    # 평가 지표
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"=== 예측 성능 ({order}) ===")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(len(y_train)-50, len(y_train)), y_train[-50:], 'b-', label='훈련')
    ax.plot(range(len(y_train), len(y_train)+len(y_test)), y_test, 'g-', label='실제')
    ax.plot(range(len(y_train), len(y_train)+len(y_test)), y_pred, 'r--', label='예측')
    ax.axvline(len(y_train), color='k', linestyle=':', alpha=0.5)
    ax.legend()
    ax.set_title(f'예측 vs 실제 (ARIMA{order})')
    ax.grid(True, alpha=0.3)
    plt.show()

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'mape': mape}

# 데이터 분할
train_size = int(len(y) * 0.8)
y_train = y[:train_size]
y_test = y[train_size:]

# 평가
evaluate_forecast(y_train, y_test, (1, 1, 1))
```

### 7.3 롤링 예측 (Rolling Forecast)

```python
def rolling_forecast(y, order, initial_train_size, horizon=1):
    """
    롤링 윈도우 예측

    Parameters:
    -----------
    y : array
        전체 시계열
    order : tuple
        ARIMA 차수
    initial_train_size : int
        초기 훈련 데이터 크기
    horizon : int
        예측 호라이즌
    """
    predictions = []
    actuals = []

    for i in range(initial_train_size, len(y) - horizon + 1):
        train = y[:i]
        actual = y[i:i+horizon]

        model = ARIMA(train, order=order)
        result = model.fit()

        pred = result.get_forecast(steps=horizon).predicted_mean.values
        predictions.append(pred[0])
        actuals.append(actual[0])

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # 평가
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)

    print(f"=== 롤링 예측 (ARIMA{order}) ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")

    # 시각화
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actuals, 'b-', label='실제', alpha=0.7)
    ax.plot(predictions, 'r--', label='예측', alpha=0.7)
    ax.set_xlabel('시간')
    ax.set_ylabel('값')
    ax.set_title('롤링 1-step 예측')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

    return predictions, actuals

predictions, actuals = rolling_forecast(y, (1, 1, 1), initial_train_size=200)
```

---

## 8. 실제 데이터 적용

### 8.1 종합 분석 예제

```python
def complete_arima_analysis(y, title=''):
    """
    ARIMA 모델링 전체 파이프라인
    """
    print(f"{'='*60}")
    print(f"시계열 분석: {title}")
    print(f"{'='*60}")

    # 1. 시각화 및 정상성 검정
    print("\n[1단계] 정상성 분석")
    from statsmodels.tsa.stattools import adfuller, kpss

    adf_p = adfuller(y)[1]
    kpss_p = kpss(y, regression='c')[1]

    print(f"ADF p-value: {adf_p:.4f}")
    print(f"KPSS p-value: {kpss_p:.4f}")

    # 차분 결정
    d = 0
    y_diff = y.copy()
    while adf_p > 0.05 and d < 2:
        d += 1
        y_diff = np.diff(y_diff)
        adf_p = adfuller(y_diff)[1]
        print(f"{d}차 차분 후 ADF p-value: {adf_p:.4f}")

    print(f"결정된 d: {d}")

    # 2. ACF/PACF 분석
    print("\n[2단계] ACF/PACF 분석")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(y_diff if d > 0 else y, lags=20, ax=axes[0], alpha=0.05)
    plot_pacf(y_diff if d > 0 else y, lags=20, ax=axes[1], alpha=0.05, method='ywm')
    plt.suptitle(f'ACF/PACF (d={d} 차분 후)')
    plt.tight_layout()
    plt.show()

    # 3. 모델 비교
    print("\n[3단계] 모델 비교 (AIC)")
    results = []
    for p in range(4):
        for q in range(4):
            if p == 0 and q == 0:
                continue
            try:
                model = ARIMA(y, order=(p, d, q))
                fit = model.fit()
                results.append({'p': p, 'd': d, 'q': q, 'aic': fit.aic, 'bic': fit.bic})
            except:
                pass

    results_df = pd.DataFrame(results).sort_values('aic')
    print(results_df.head(5))

    # 최적 모델
    best = results_df.iloc[0]
    best_order = (int(best['p']), int(best['d']), int(best['q']))
    print(f"\n최적 모델: ARIMA{best_order}")

    # 4. 최종 모델 적합 및 진단
    print(f"\n[4단계] 최종 모델 적합: ARIMA{best_order}")
    final_model = ARIMA(y, order=best_order)
    final_result = final_model.fit()
    print(final_result.summary())

    # 진단
    diagnose_arima_model(final_result, f'ARIMA{best_order}')

    # 5. 예측
    print("\n[5단계] 예측")
    forecast = final_result.get_forecast(steps=20)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y[-100:], label='관측값')
    forecast_index = range(len(y), len(y) + 20)
    ax.plot(forecast_index, forecast.predicted_mean, 'r-', label='예측')
    ci = forecast.conf_int()
    ax.fill_between(forecast_index, ci.iloc[:, 0], ci.iloc[:, 1],
                    color='r', alpha=0.2, label='95% CI')
    ax.legend()
    ax.set_title(f'ARIMA{best_order} 예측')
    ax.grid(True, alpha=0.3)
    plt.show()

    return final_result

# 실행 예제
np.random.seed(42)
ar = [1, -0.7, 0.2]
ma = [1, 0.5]
y_sample = arma_generate_sample(ar, ma, nsample=300, scale=1, burnin=100)
y_sample = np.cumsum(y_sample)  # 비정상화

result = complete_arima_analysis(y_sample, '샘플 시계열')
```

---

## 9. 벡터 자기회귀 모형 (Vector Autoregression, VAR)

위에서 다룬 AR, MA, ARMA, ARIMA 모형은 모두 단일 시계열을 다룹니다. 그러나 실제로 경제 및 과학 변수는 함께 움직입니다: GDP와 실업률, 온도와 기압, 업종별 주가 등. **벡터 자기회귀(VAR)** 모형은 AR 프레임워크를 여러 시계열로 동시에 확장하여, 각 변수가 자기 자신의 과거 *그리고* 시스템 내 다른 모든 변수의 과거에 의존할 수 있게 합니다.

### 9.1 VAR(p) 모형 수식

K차원 시계열 **Y**ₜ = (Y₁ₜ, Y₂ₜ, ..., Yₖₜ)'에 대한 VAR(p) 모형:

$$\mathbf{Y}_t = \mathbf{c} + \mathbf{A}_1 \mathbf{Y}_{t-1} + \mathbf{A}_2 \mathbf{Y}_{t-2} + \cdots + \mathbf{A}_p \mathbf{Y}_{t-p} + \mathbf{u}_t$$

여기서:
- **c**: K x 1 절편 벡터(Intercept vector)
- **A**ᵢ: K x K 계수 행렬(Coefficient matrix, 각 시차별 하나)
- **u**ₜ: K x 1 백색 잡음 벡터, E(**u**ₜ) = **0**, 공분산 행렬 Σ

두 변수 Y, X에 대한 이변량 VAR(1)의 경우:

$$\begin{pmatrix} Y_t \\ X_t \end{pmatrix} = \begin{pmatrix} c_1 \\ c_2 \end{pmatrix} + \begin{pmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{pmatrix} \begin{pmatrix} Y_{t-1} \\ X_{t-1} \end{pmatrix} + \begin{pmatrix} u_{1t} \\ u_{2t} \end{pmatrix}$$

비대각 원소(a₁₂, a₂₁)는 **변수 간 교차 동학(Cross-variable dynamics)**을 포착합니다 — 이것이 각 변수에 별도의 AR 모형을 적합하는 것과 비교했을 때 VAR의 핵심 강점입니다.

### 9.2 Python으로 VAR 모형 적합

```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

np.random.seed(42)

# --- 이변량 VAR(2) 과정 시뮬레이션 ---
# 왜 시뮬레이션하는가? 추정기가 참 모수를 복원하는지 검증할 수 있기 때문이다.
n = 500
K = 2
y = np.zeros((n, K))

# 참 계수 행렬
A1 = np.array([[0.5, 0.1],
               [0.2, 0.3]])
A2 = np.array([[-0.2, 0.0],
               [0.1, -0.1]])

for t in range(2, n):
    y[t] = A1 @ y[t-1] + A2 @ y[t-2] + np.random.multivariate_normal(
        [0, 0], [[1.0, 0.3], [0.3, 0.8]]
    )

df_var = pd.DataFrame(y, columns=['GDP_growth', 'Inflation'])

# 정상성 확인 — VAR는 모든 시계열이 정상이어야 한다
for col in df_var.columns:
    adf_p = adfuller(df_var[col])[1]
    print(f"{col}: ADF p-value = {adf_p:.4f} ({'정상' if adf_p < 0.05 else '비정상'})")

# 정보 기준으로 자동 시차 선택
# 왜 정보 기준을 사용하는가? VAR의 모수 수는 K^2 * p로 증가하므로
# (변수 수의 제곱에 비례) 적합도와 과적합 사이의 균형이 특히 중요하다.
model = VAR(df_var)
lag_order = model.select_order(maxlags=8)
print("\n=== 시차 선택 ===")
print(lag_order.summary())

# 선택된 모형 적합
result = model.fit(maxlags=8, ic='aic')
print(f"\n선택된 시차: {result.k_ar}")
print(result.summary())
```

### 9.3 그랜저 인과성 (Granger Causality)

그랜저 인과성(Granger Causality)은 다음을 묻습니다: 변수 X의 과거를 아는 것이 Y 자신의 과거만으로 예측하는 것 이상으로 Y의 예측을 개선하는가? 이것은 *예측적 선행성(Predictive precedence)*에 대한 통계적 검정이며, 진정한 인과관계가 아닙니다.

```python
from statsmodels.tsa.stattools import grangercausalitytests

# 인플레이션이 GDP 성장을 그랜저 인과하는지 검정
# 왜 그랜저 인과가 중요한가: X가 Y를 그랜저 인과하면, X의 시차를
# 모형에 포함할 때 Y의 예측 오차가 유의하게 줄어든다.
# 이는 VAR 시스템에 어떤 변수를 포함할지 결정하는 데 도움을 준다.
print("=== 그랜저 인과: 인플레이션 -> GDP 성장 ===")
gc_result = grangercausalitytests(df_var[['GDP_growth', 'Inflation']], maxlag=4)

print("\n=== 그랜저 인과: GDP 성장 -> 인플레이션 ===")
gc_result = grangercausalitytests(df_var[['Inflation', 'GDP_growth']], maxlag=4)

# 해석: 어떤 시차에서든 p-value < 0.05이면, 두 번째 변수가 첫 번째 변수를
# 그랜저 인과하지 않는다는 귀무가설을 기각한다.
```

### 9.4 충격 반응 함수 (Impulse Response Functions, IRF)

충격 반응 함수(IRF)는 한 변수에 대한 단위 충격이 시간에 따라 시스템을 통해 어떻게 전파되는지 추적합니다. 이는 VAR 결과를 해석하는 데 가장 중요한 도구로, 변수 간 *동적 승수 효과(Dynamic multiplier effects)*를 보여줍니다.

```python
# 충격 반응 함수 계산
# 왜 직교화하는가? 원시 충격은 변수 간에 상관되어 있는 것이 일반적이다.
# 직교화된 IRF는 오차 공분산 행렬의 촐레스키 분해(Cholesky decomposition)를
# 사용하여 한 변수에 대한 "순수한" 충격의 효과를 분리한다.
irf = result.irf(periods=20)

# 신뢰대역과 함께 IRF 그리기
fig = irf.plot(orth=True, impulse='GDP_growth', response='Inflation')
plt.suptitle('직교화 IRF: GDP 성장 충격 -> 인플레이션')
plt.tight_layout()
plt.show()

fig = irf.plot(orth=True, impulse='Inflation', response='GDP_growth')
plt.suptitle('직교화 IRF: 인플레이션 충격 -> GDP 성장')
plt.tight_layout()
plt.show()

# 누적 IRF — 장기적(누적) 효과를 보여줌
fig = irf.plot_cum_effects(orth=True)
plt.suptitle('누적 충격 반응 함수')
plt.tight_layout()
plt.show()
```

### 9.5 예측 오차 분산 분해 (Forecast Error Variance Decomposition, FEVD)

FEVD는 다음에 답합니다: 예측 수평선 h에서, 변수 Y의 예측 불확실성 중 변수 X의 충격에 기인하는 비율은 얼마인가? 이는 시간에 따른 각 충격 원천의 상대적 중요성을 정량화합니다.

```python
# FEVD 계산
fevd = result.fevd(periods=20)
print("=== 예측 오차 분산 분해 ===")
print(fevd.summary())

# FEVD 시각화
fig = fevd.plot()
plt.suptitle('예측 오차 분산 분해')
plt.tight_layout()
plt.show()

# FEVD가 왜 중요한가: 거시경제학에서 FEVD는 "산출 변동의 몇 퍼센트가
# 통화정책 충격 vs 공급 충격에 의한 것인가?"와 같은 질문에 답한다.
# 이는 점추정을 넘어 시스템 내 각 변수의 구조적 중요성을 보여준다.
```

### 9.6 VAR로 예측

```python
# 향후 10기간 예측
forecast_steps = 10
forecast = result.forecast(df_var.values[-result.k_ar:], steps=forecast_steps)

# 신뢰구간과 함께 예측
forecast_ci = result.forecast_interval(
    df_var.values[-result.k_ar:], steps=forecast_steps, alpha=0.05
)
point_forecast, lower_ci, upper_ci = forecast_ci

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, col in enumerate(df_var.columns):
    # 마지막 50개 관측값
    axes[i].plot(range(n-50, n), df_var[col].values[-50:], 'b-', label='관측값')

    # 예측
    fcast_idx = range(n, n + forecast_steps)
    axes[i].plot(fcast_idx, point_forecast[:, i], 'r-', label='예측')
    axes[i].fill_between(fcast_idx, lower_ci[:, i], upper_ci[:, i],
                         color='r', alpha=0.2, label='95% CI')
    axes[i].axvline(n, color='k', linestyle='--', alpha=0.5)
    axes[i].set_title(f'{col} 예측')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 9.7 VAR vs VECM: 공적분 시계열

VAR는 모든 시계열이 정상이라고 가정합니다. 시계열이 비정상이지만 공통 확률적 추세(Stochastic trend)를 공유하는 경우(즉, **공적분(Cointegrated)** 관계), **벡터 오차 수정 모형(VECM, Vector Error Correction Model)**을 사용해야 합니다. VECM은 본질적으로 1차 차분의 VAR에 장기 균형 관계를 포착하는 오차 수정 항(Error correction terms)을 추가한 것입니다.

| 관점 | VAR | VECM |
|--------|-----|------|
| 입력 시계열 | 정상 (또는 차분된) | 비정상이지만 공적분 |
| 장기 관계 | 모델링하지 않음 | 공적분 방정식으로 명시적 모델링 |
| 사용 시점 | 변수가 독립적으로 움직일 때 | 변수가 공통 장기 추세를 공유할 때 |
| 차분 | 먼저 차분 후 VAR 적합 | 차분 불필요 — VECM이 처리 |
| 선행 검정 | 정상성 검정 (ADF/KPSS) | 공적분 순위 검정 (Johansen test) |

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

# 공적분 시계열 시뮬레이션 (Y와 X가 공통 추세를 공유)
np.random.seed(42)
n = 300
trend = np.cumsum(np.random.normal(0, 1, n))  # 공통 확률적 추세
x = trend + np.random.normal(0, 0.5, n)
y_coint = 0.8 * trend + np.random.normal(0, 0.5, n)  # x와 공적분 관계

df_coint = pd.DataFrame({'Y': y_coint, 'X': x})

# 요한센 공적분 검정 (Johansen cointegration test)
# 왜 요한센인가? 엥글-그래인저(Engle-Granger) 검정은 2개 변수로 제한되지만,
# 요한센 검정은 임의의 수의 변수를 다루며 공적분 관계의 수(순위)를 검정한다.
johansen_result = coint_johansen(df_coint, det_order=0, k_ar_diff=2)
print("=== 요한센 공적분 검정 ===")
print(f"Trace 통계량: {johansen_result.lr1}")
print(f"임계값 (90%, 95%, 99%): {johansen_result.cvt}")

# 공적분이 있으면 VECM 적합
vecm = VECM(df_coint, k_ar_diff=2, coint_rank=1)
vecm_result = vecm.fit()
print("\n=== VECM 결과 ===")
print(vecm_result.summary())
```

**경험 법칙**: 수준(levels)에서 VAR를 적합하기 전에 항상 공적분을 검정하세요. 공적분이 발견되면 VECM을 사용하세요 — 차분은 귀중한 장기 정보를 파괴하지만 VECM은 이를 보존합니다.

---

## 10. 연습 문제

### 문제 1: 모델 식별
다음 ACF/PACF 패턴에 해당하는 모델을 식별하세요:
1. ACF: lag 1만 유의 (0.4), 이후 0
   PACF: 지수적 감소

2. ACF: 지수적 감소 (0.9, 0.81, 0.73, ...)
   PACF: lag 1만 유의 (0.9)

### 문제 2: ARIMA 추정
다음 모델에서 데이터를 생성하고 추정하세요:
- ARIMA(2,1,1): φ₁=0.5, φ₂=0.2, θ₁=0.3

### 문제 3: 모델 비교
AIC와 BIC가 다른 모델을 선택할 때 어떻게 결정해야 할까요?

### 문제 4: 예측 평가
훈련-테스트 분할과 롤링 예측의 장단점을 비교하세요.

---

## 11. 핵심 요약

### ARIMA 모델 체계

| 모델 | 형태 | 특징 |
|------|------|------|
| AR(p) | Yₜ = φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ + εₜ | PACF lag p 후 절단 |
| MA(q) | Yₜ = εₜ + θ₁εₜ₋₁ + ... + θₑεₜ₋ₑ | ACF lag q 후 절단 |
| ARMA(p,q) | AR + MA | 둘 다 점진적 감소 |
| ARIMA(p,d,q) | d차 차분 후 ARMA | 비정상 시계열 |

### Box-Jenkins 체크리스트

1. [ ] 시계열 플롯 및 정상성 검정
2. [ ] 필요시 차분 (d 결정)
3. [ ] ACF/PACF 분석 (p, q 후보)
4. [ ] 후보 모델 적합 및 AIC/BIC 비교
5. [ ] 잔차 진단 (백색 잡음, Ljung-Box)
6. [ ] 예측 및 성능 평가

### 정보 기준

- **AIC**: 예측 성능 중시, 모델이 클 수 있음
- **BIC**: 간결성 중시, 작은 모델 선호

### 다음 장 미리보기

12장 **다변량 분석**에서는:
- 주성분 분석 (PCA)
- 요인 분석
- 판별 분석 (LDA, QDA)
- 군집 타당성 검증
