# 10. 시계열 분석 기초 (Time Series Analysis Fundamentals)

## 개요

시계열 데이터는 시간 순서로 수집된 데이터입니다. 이 장에서는 시계열의 구성요소, 정상성 개념, 자기상관 분석, 그리고 시계열 분해 방법을 학습합니다.

---

## 1. 시계열 구성요소

### 1.1 네 가지 구성요소

| 구성요소 | 설명 | 예시 |
|----------|------|------|
| **추세 (Trend)** | 장기적인 증가/감소 패턴 | 인구 증가, 기술 발전 |
| **계절성 (Seasonality)** | 고정 주기로 반복되는 패턴 | 여름 에어컨 판매, 연말 쇼핑 |
| **순환 (Cycle)** | 비고정 주기의 반복 패턴 | 경기 순환 (불규칙적) |
| **잔차 (Residual/Noise)** | 설명되지 않는 무작위 변동 | 측정 오차, 예측 불가 요인 |

### 1.2 시계열 데이터 생성 및 시각화

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

np.random.seed(42)

# 시계열 데이터 생성
n_points = 365 * 3  # 3년 일별 데이터
dates = pd.date_range(start='2021-01-01', periods=n_points, freq='D')
t = np.arange(n_points)

# 구성요소
trend = 0.05 * t  # 선형 추세
seasonal = 10 * np.sin(2 * np.pi * t / 365)  # 연간 계절성
weekly = 3 * np.sin(2 * np.pi * t / 7)  # 주간 패턴
noise = np.random.normal(0, 2, n_points)  # 잡음

# 합성 시계열
y = 100 + trend + seasonal + weekly + noise

# DataFrame 생성
ts_data = pd.DataFrame({
    'date': dates,
    'value': y,
    'trend': 100 + trend,
    'seasonal': seasonal,
    'weekly': weekly,
    'noise': noise
})
ts_data.set_index('date', inplace=True)

# 시각화
fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)

axes[0].plot(ts_data.index, ts_data['value'], 'b-', alpha=0.7)
axes[0].set_ylabel('Value')
axes[0].set_title('합성 시계열')
axes[0].grid(True, alpha=0.3)

axes[1].plot(ts_data.index, ts_data['trend'], 'g-')
axes[1].set_ylabel('Trend')
axes[1].set_title('추세 (Trend)')
axes[1].grid(True, alpha=0.3)

axes[2].plot(ts_data.index, ts_data['seasonal'], 'orange')
axes[2].set_ylabel('Seasonal')
axes[2].set_title('계절성 (Annual)')
axes[2].grid(True, alpha=0.3)

axes[3].plot(ts_data.index, ts_data['weekly'], 'purple')
axes[3].set_ylabel('Weekly')
axes[3].set_title('주간 패턴 (Weekly)')
axes[3].grid(True, alpha=0.3)

axes[4].plot(ts_data.index, ts_data['noise'], 'gray', alpha=0.7)
axes[4].set_ylabel('Noise')
axes[4].set_title('잔차 (Noise)')
axes[4].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 1.3 실제 데이터 예시

```python
# statsmodels 내장 데이터셋
import statsmodels.api as sm
from statsmodels.datasets import co2, sunspots

# CO2 데이터
co2_data = co2.load_pandas().data
co2_data = co2_data.resample('M').mean()  # 월별 평균

# 태양 흑점 데이터
sunspots_data = sunspots.load_pandas().data
sunspots_data.index = pd.date_range(start='1700', periods=len(sunspots_data), freq='Y')

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# CO2: 명확한 추세 + 계절성
axes[0].plot(co2_data.index, co2_data['co2'])
axes[0].set_xlabel('Year')
axes[0].set_ylabel('CO2 (ppm)')
axes[0].set_title('Mauna Loa CO2: 추세 + 계절성')
axes[0].grid(True, alpha=0.3)

# 태양 흑점: 순환 패턴 (약 11년)
axes[1].plot(sunspots_data.index, sunspots_data['SUNACTIVITY'])
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Sunspot Activity')
axes[1].set_title('태양 흑점 활동: 약 11년 주기 순환')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 2. 정상성 (Stationarity)

### 2.1 정상성 개념

**강정상성 (Strict Stationarity)**:
모든 차수의 결합분포가 시간 이동에 불변

**약정상성 (Weak Stationarity)**:
1. 평균이 시간에 상관없이 일정: E[Yₜ] = μ
2. 분산이 시간에 상관없이 일정: Var(Yₜ) = σ²
3. 자기공분산이 시차에만 의존: Cov(Yₜ, Yₜ₊ₕ) = γ(h)

```python
def demonstrate_stationarity():
    """정상성과 비정상성 비교"""
    np.random.seed(42)
    n = 500

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 정상 시계열: 백색 잡음
    white_noise = np.random.normal(0, 1, n)
    axes[0, 0].plot(white_noise)
    axes[0, 0].set_title('정상: 백색 잡음')
    axes[0, 0].axhline(0, color='r', linestyle='--')
    axes[0, 0].set_ylim(-4, 4)

    # 정상 시계열: AR(1) |φ| < 1
    ar1_stationary = np.zeros(n)
    phi = 0.7
    for t in range(1, n):
        ar1_stationary[t] = phi * ar1_stationary[t-1] + np.random.normal(0, 1)
    axes[0, 1].plot(ar1_stationary)
    axes[0, 1].set_title(f'정상: AR(1), φ={phi}')
    axes[0, 1].axhline(0, color='r', linestyle='--')

    # 정상 시계열: MA(1)
    ma1 = np.zeros(n)
    errors = np.random.normal(0, 1, n)
    theta = 0.6
    for t in range(1, n):
        ma1[t] = errors[t] + theta * errors[t-1]
    axes[0, 2].plot(ma1)
    axes[0, 2].set_title(f'정상: MA(1), θ={theta}')
    axes[0, 2].axhline(0, color='r', linestyle='--')

    # 비정상: 선형 추세
    trend_ts = 0.1 * np.arange(n) + np.random.normal(0, 2, n)
    axes[1, 0].plot(trend_ts)
    axes[1, 0].set_title('비정상: 선형 추세')

    # 비정상: 랜덤 워크 (단위근)
    random_walk = np.cumsum(np.random.normal(0, 1, n))
    axes[1, 1].plot(random_walk)
    axes[1, 1].set_title('비정상: 랜덤 워크 (단위근)')

    # 비정상: 이분산
    heteroscedastic = np.zeros(n)
    for t in range(n):
        heteroscedastic[t] = np.random.normal(0, 1 + 0.01 * t)
    axes[1, 2].plot(heteroscedastic)
    axes[1, 2].set_title('비정상: 이분산 (분산 증가)')

    for ax in axes.flatten():
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

demonstrate_stationarity()
```

### 2.2 정상성이 중요한 이유

```python
print("=== 정상성이 중요한 이유 ===")
print()
print("1. 통계적 추론의 기초")
print("   - 표본 통계량(평균, 분산)이 일관된 추정치를 제공")
print("   - 비정상 시계열에서는 표본 평균이 모평균을 추정하지 않음")
print()
print("2. 예측 가능성")
print("   - 정상 시계열: 과거 패턴이 미래에도 반복")
print("   - 비정상 시계열: 장기 예측이 불가능하거나 불안정")
print()
print("3. 모델 적용")
print("   - ARMA 모델은 정상 시계열을 가정")
print("   - 비정상 시계열은 변환 후 모델링 (차분, 로그 등)")
```

---

## 3. 단위근 검정 (Unit Root Tests)

### 3.1 ADF 검정 (Augmented Dickey-Fuller)

**가설**:
- H₀: 단위근 존재 (비정상)
- H₁: 단위근 없음 (정상)

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):
    """
    ADF 단위근 검정 수행 및 결과 출력

    Parameters:
    -----------
    series : array-like
        시계열 데이터
    title : str
        시계열 이름
    """
    result = adfuller(series, autolag='AIC')

    print(f"=== ADF Test: {title} ===")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    print(f"Observations: {result[3]}")
    print("Critical Values:")
    for key, value in result[4].items():
        print(f"   {key}: {value:.4f}")

    if result[1] < 0.05:
        print("결론: p < 0.05 → 단위근 없음 (정상)")
    else:
        print("결론: p >= 0.05 → 단위근 존재 (비정상)")
    print()

    return result[1]  # p-value 반환

# 테스트
np.random.seed(42)

# 정상 시계열
stationary = np.random.normal(0, 1, 500)
adf_test(stationary, '백색 잡음 (정상)')

# 비정상 시계열: 랜덤 워크
random_walk = np.cumsum(np.random.normal(0, 1, 500))
adf_test(random_walk, '랜덤 워크 (비정상)')

# 비정상 시계열: 추세
trend = 0.1 * np.arange(500) + np.random.normal(0, 1, 500)
adf_test(trend, '선형 추세 (비정상)')
```

### 3.2 KPSS 검정

**가설** (ADF와 반대):
- H₀: 정상 (추세 정상)
- H₁: 비정상 (단위근)

```python
from statsmodels.tsa.stattools import kpss

def kpss_test(series, title='', regression='c'):
    """
    KPSS 검정 수행

    Parameters:
    -----------
    regression : str
        'c' - 상수만 (수준 정상성)
        'ct' - 상수 + 추세 (추세 정상성)
    """
    result = kpss(series, regression=regression, nlags='auto')

    print(f"=== KPSS Test: {title} ===")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"Lags Used: {result[2]}")
    print("Critical Values:")
    for key, value in result[3].items():
        print(f"   {key}: {value:.4f}")

    if result[1] < 0.05:
        print("결론: p < 0.05 → 비정상")
    else:
        print("결론: p >= 0.05 → 정상")
    print()

    return result[1]

# ADF와 KPSS 결합 해석
np.random.seed(42)

test_series = {
    '백색 잡음': np.random.normal(0, 1, 500),
    '랜덤 워크': np.cumsum(np.random.normal(0, 1, 500)),
    '추세 + 잡음': 0.1 * np.arange(500) + np.random.normal(0, 1, 500),
}

print("=== ADF + KPSS 결합 해석 ===")
print("ADF p < 0.05 AND KPSS p >= 0.05 → 정상")
print("ADF p >= 0.05 AND KPSS p < 0.05 → 비정상 (차분 필요)")
print("둘 다 p < 0.05 → 추세 정상 (추세 제거 후 정상)")
print("둘 다 p >= 0.05 → 결정 어려움 (추가 분석 필요)")
print()

for name, series in test_series.items():
    print(f"--- {name} ---")
    adf_p = adf_test(series, f'{name} (ADF)')
    kpss_p = kpss_test(series, f'{name} (KPSS)')
    print()
```

---

## 4. 차분 (Differencing)

### 4.1 차분의 개념

**1차 차분**: ∇Yₜ = Yₜ - Yₜ₋₁

**2차 차분**: ∇²Yₜ = ∇(∇Yₜ) = (Yₜ - Yₜ₋₁) - (Yₜ₋₁ - Yₜ₋₂)

**계절 차분**: ∇ₛYₜ = Yₜ - Yₜ₋ₛ (s = 계절 주기)

```python
def demonstrate_differencing(y, title=''):
    """차분의 효과 시각화"""

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    # 원본
    axes[0, 0].plot(y)
    axes[0, 0].set_title(f'원본: {title}')
    axes[0, 0].grid(True, alpha=0.3)

    # 1차 차분
    diff1 = np.diff(y)
    axes[1, 0].plot(diff1)
    axes[1, 0].set_title('1차 차분')
    axes[1, 0].grid(True, alpha=0.3)

    # 2차 차분
    diff2 = np.diff(y, n=2)
    axes[2, 0].plot(diff2)
    axes[2, 0].set_title('2차 차분')
    axes[2, 0].grid(True, alpha=0.3)

    # ADF 검정 결과
    adf_original = adfuller(y)[1]
    adf_diff1 = adfuller(diff1)[1]
    adf_diff2 = adfuller(diff2)[1]

    # 히스토그램
    axes[0, 1].hist(y, bins=30, density=True, alpha=0.7)
    axes[0, 1].set_title(f'원본 분포 (ADF p={adf_original:.4f})')

    axes[1, 1].hist(diff1, bins=30, density=True, alpha=0.7)
    axes[1, 1].set_title(f'1차 차분 분포 (ADF p={adf_diff1:.4f})')

    axes[2, 1].hist(diff2, bins=30, density=True, alpha=0.7)
    axes[2, 1].set_title(f'2차 차분 분포 (ADF p={adf_diff2:.4f})')

    plt.tight_layout()
    plt.show()

    return adf_original, adf_diff1, adf_diff2

# 랜덤 워크
np.random.seed(42)
random_walk = np.cumsum(np.random.normal(0, 1, 500))
demonstrate_differencing(random_walk, '랜덤 워크')

# 2차 트렌드
t = np.arange(500)
quadratic_trend = 0.001 * t**2 + np.random.normal(0, 2, 500)
demonstrate_differencing(quadratic_trend, '2차 트렌드')
```

### 4.2 계절 차분

```python
# 계절성이 있는 데이터
np.random.seed(42)
n = 365 * 3
t = np.arange(n)

# 추세 + 연간 계절성 + 잡음
seasonal_ts = 0.01 * t + 10 * np.sin(2 * np.pi * t / 365) + np.random.normal(0, 1, n)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# 원본
axes[0].plot(seasonal_ts)
axes[0].set_title(f'원본 (ADF p={adfuller(seasonal_ts)[1]:.4f})')
axes[0].grid(True, alpha=0.3)

# 일반 1차 차분
diff1 = np.diff(seasonal_ts)
axes[1].plot(diff1)
axes[1].set_title(f'1차 차분 (ADF p={adfuller(diff1)[1]:.4f})')
axes[1].grid(True, alpha=0.3)

# 계절 차분 (lag=365)
seasonal_diff = seasonal_ts[365:] - seasonal_ts[:-365]
axes[2].plot(seasonal_diff)
axes[2].set_title(f'계절 차분 (lag=365) (ADF p={adfuller(seasonal_diff)[1]:.4f})')
axes[2].grid(True, alpha=0.3)

# 계절 차분 + 1차 차분
both_diff = np.diff(seasonal_diff)
axes[3].plot(both_diff)
axes[3].set_title(f'계절차분 + 1차차분 (ADF p={adfuller(both_diff)[1]:.4f})')
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 5. ACF와 PACF

### 5.1 자기상관함수 (ACF)

**자기공분산**: γ(h) = Cov(Yₜ, Yₜ₊ₕ)

**자기상관**: ρ(h) = γ(h) / γ(0) = Corr(Yₜ, Yₜ₊ₕ)

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf, pacf

def analyze_acf_pacf(y, lags=40, title=''):
    """ACF/PACF 분석"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 시계열
    axes[0, 0].plot(y[:200])  # 처음 200개만
    axes[0, 0].set_title(f'시계열: {title}')
    axes[0, 0].grid(True, alpha=0.3)

    # 히스토그램
    axes[0, 1].hist(y, bins=30, density=True, alpha=0.7)
    axes[0, 1].set_title('분포')
    axes[0, 1].grid(True, alpha=0.3)

    # ACF
    plot_acf(y, lags=lags, ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF (자기상관함수)')

    # PACF
    plot_pacf(y, lags=lags, ax=axes[1, 1], alpha=0.05, method='ywm')
    axes[1, 1].set_title('PACF (편자기상관함수)')

    plt.tight_layout()
    plt.show()

# 다양한 패턴 분석
np.random.seed(42)
n = 500

# 백색 잡음
white_noise = np.random.normal(0, 1, n)
analyze_acf_pacf(white_noise, title='백색 잡음')

# AR(1) 과정
ar1 = np.zeros(n)
phi = 0.8
for t in range(1, n):
    ar1[t] = phi * ar1[t-1] + np.random.normal(0, 1)
analyze_acf_pacf(ar1, title=f'AR(1), φ={phi}')

# MA(1) 과정
ma1 = np.zeros(n)
errors = np.random.normal(0, 1, n)
theta = 0.6
for t in range(1, n):
    ma1[t] = errors[t] + theta * errors[t-1]
analyze_acf_pacf(ma1, title=f'MA(1), θ={theta}')
```

### 5.2 PACF (편자기상관함수)

**개념**: 중간 시차들의 영향을 제거한 후의 상관

```python
def pacf_intuition():
    """PACF 직관적 이해"""

    print("=== PACF 직관 ===")
    print()
    print("ACF(lag 2) = Corr(Yₜ, Yₜ₋₂)")
    print("  → lag 1의 영향이 포함됨")
    print()
    print("PACF(lag 2) = Corr(Yₜ, Yₜ₋₂ | Yₜ₋₁)")
    print("  → lag 1의 영향을 제거한 순수한 lag 2의 영향")
    print()
    print("활용:")
    print("  - AR(p) 모델: PACF가 lag p에서 끊김 (cut-off)")
    print("  - MA(q) 모델: ACF가 lag q에서 끊김 (cut-off)")
    print("  - ARMA: 둘 다 지수적/진동적 감소")

pacf_intuition()
```

### 5.3 모델 식별을 위한 ACF/PACF 패턴

```python
def acf_pacf_patterns():
    """ACF/PACF 패턴과 모델 대응"""

    patterns = """
    | 모델 | ACF 패턴 | PACF 패턴 |
    |------|----------|-----------|
    | AR(p) | 지수적/진동적 감소 | lag p 후 절단 |
    | MA(q) | lag q 후 절단 | 지수적/진동적 감소 |
    | ARMA(p,q) | 지수적/진동적 감소 | 지수적/진동적 감소 |
    | AR(1) φ>0 | 지수적 감소 | lag 1에서만 유의 |
    | AR(1) φ<0 | 교대로 감소 | lag 1에서만 유의 (음수) |
    | AR(2) | 감쇠 사인파 또는 두 지수 혼합 | lag 2에서 절단 |
    | MA(1) θ>0 | lag 1에서만 유의 (음수) | 지수적 감소 |
    | MA(1) θ<0 | lag 1에서만 유의 (양수) | 교대로 감소 |
    """
    print(patterns)

acf_pacf_patterns()

# 시각적 예시
fig, axes = plt.subplots(4, 3, figsize=(15, 12))

np.random.seed(42)
n = 500
errors = np.random.normal(0, 1, n)

# AR(1) φ = 0.8
ar1_pos = np.zeros(n)
for t in range(1, n):
    ar1_pos[t] = 0.8 * ar1_pos[t-1] + errors[t]

# AR(1) φ = -0.8
ar1_neg = np.zeros(n)
for t in range(1, n):
    ar1_neg[t] = -0.8 * ar1_neg[t-1] + errors[t]

# MA(1) θ = 0.8
ma1_pos = np.zeros(n)
for t in range(1, n):
    ma1_pos[t] = errors[t] + 0.8 * errors[t-1]

# AR(2)
ar2 = np.zeros(n)
for t in range(2, n):
    ar2[t] = 0.5 * ar2[t-1] + 0.3 * ar2[t-2] + errors[t]

models = [
    (ar1_pos, 'AR(1) φ=0.8'),
    (ar1_neg, 'AR(1) φ=-0.8'),
    (ma1_pos, 'MA(1) θ=0.8'),
    (ar2, 'AR(2) φ₁=0.5, φ₂=0.3'),
]

for i, (y, title) in enumerate(models):
    axes[i, 0].plot(y[:200])
    axes[i, 0].set_title(title)
    axes[i, 0].grid(True, alpha=0.3)

    plot_acf(y, lags=20, ax=axes[i, 1], alpha=0.05)
    axes[i, 1].set_title(f'{title} - ACF')

    plot_pacf(y, lags=20, ax=axes[i, 2], alpha=0.05, method='ywm')
    axes[i, 2].set_title(f'{title} - PACF')

plt.tight_layout()
plt.show()
```

---

## 6. 시계열 분해 (Decomposition)

### 6.1 가법 모형 vs 승법 모형

**가법 모형 (Additive)**:
$$Y_t = T_t + S_t + R_t$$

**승법 모형 (Multiplicative)**:
$$Y_t = T_t \times S_t \times R_t$$

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# 가법과 승법 비교
np.random.seed(42)
n = 365 * 3
t = np.arange(n)

# 가법 시계열
trend = 100 + 0.1 * t
seasonal_add = 10 * np.sin(2 * np.pi * t / 365)
noise_add = np.random.normal(0, 5, n)
additive_ts = trend + seasonal_add + noise_add

# 승법 시계열 (계절 변동이 수준에 비례)
trend = 100 + 0.1 * t
seasonal_mult = 1 + 0.1 * np.sin(2 * np.pi * t / 365)
noise_mult = np.random.normal(1, 0.05, n)
multiplicative_ts = trend * seasonal_mult * noise_mult

fig, axes = plt.subplots(2, 1, figsize=(14, 6))

dates = pd.date_range(start='2021-01-01', periods=n, freq='D')

axes[0].plot(dates, additive_ts)
axes[0].set_title('가법 모형: 계절 변동이 일정')
axes[0].grid(True, alpha=0.3)

axes[1].plot(dates, multiplicative_ts)
axes[1].set_title('승법 모형: 계절 변동이 수준에 비례')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("모형 선택 기준:")
print("- 계절 변동의 폭이 일정 → 가법")
print("- 계절 변동의 폭이 수준에 비례 → 승법")
print("- 불확실하면 로그 변환 후 가법 모형 시도")
```

### 6.2 고전적 분해 (Classical Decomposition)

```python
# 월별 데이터로 분해 예시
# 항공 승객 데이터 생성 (비행 승객과 유사한 패턴)
np.random.seed(42)
n_months = 144  # 12년
t = np.arange(n_months)

# 승법 패턴의 데이터
trend = 100 + 2 * t + 0.01 * t**2
seasonal = 1 + 0.2 * np.sin(2 * np.pi * t / 12) + 0.1 * np.cos(4 * np.pi * t / 12)
noise = np.random.normal(1, 0.03, n_months)
airline_like = trend * seasonal * noise

dates = pd.date_range(start='2010-01', periods=n_months, freq='M')
ts_series = pd.Series(airline_like, index=dates)

# 가법 분해
decomposition_add = seasonal_decompose(ts_series, model='additive', period=12)

# 승법 분해
decomposition_mult = seasonal_decompose(ts_series, model='multiplicative', period=12)

# 시각화
fig, axes = plt.subplots(4, 2, figsize=(14, 12))

# 가법 분해
axes[0, 0].plot(ts_series)
axes[0, 0].set_title('원본')
axes[0, 0].set_ylabel('값')

axes[1, 0].plot(decomposition_add.trend)
axes[1, 0].set_title('가법 - 추세')
axes[1, 0].set_ylabel('추세')

axes[2, 0].plot(decomposition_add.seasonal)
axes[2, 0].set_title('가법 - 계절성')
axes[2, 0].set_ylabel('계절성')

axes[3, 0].plot(decomposition_add.resid)
axes[3, 0].set_title('가법 - 잔차')
axes[3, 0].set_ylabel('잔차')

# 승법 분해
axes[0, 1].plot(ts_series)
axes[0, 1].set_title('원본')

axes[1, 1].plot(decomposition_mult.trend)
axes[1, 1].set_title('승법 - 추세')

axes[2, 1].plot(decomposition_mult.seasonal)
axes[2, 1].set_title('승법 - 계절성')

axes[3, 1].plot(decomposition_mult.resid)
axes[3, 1].set_title('승법 - 잔차')

for ax in axes.flatten():
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 6.3 STL 분해 (Seasonal and Trend decomposition using Loess)

```python
from statsmodels.tsa.seasonal import STL

# STL 분해 (더 유연함)
stl = STL(ts_series, period=12, robust=True)
result = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(12, 10))

result.observed.plot(ax=axes[0])
axes[0].set_ylabel('Observed')
axes[0].set_title('STL 분해')

result.trend.plot(ax=axes[1])
axes[1].set_ylabel('Trend')

result.seasonal.plot(ax=axes[2])
axes[2].set_ylabel('Seasonal')

result.resid.plot(ax=axes[3])
axes[3].set_ylabel('Residual')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# STL의 장점
print("=== STL 분해의 장점 ===")
print("1. 계절성 패턴이 시간에 따라 변할 수 있음")
print("2. 이상치에 강건한 옵션 (robust=True)")
print("3. 추세 추출의 유연성 조절 가능")
print("4. 비대칭 계절성 패턴 처리 가능")
```

---

## 7. 실제 데이터 분석 예제

### 7.1 종합 분석 함수

```python
def comprehensive_time_series_analysis(series, period=None, title=''):
    """
    시계열 종합 분석

    Parameters:
    -----------
    series : pd.Series
        시계열 데이터 (DatetimeIndex 필요)
    period : int
        계절 주기 (None이면 자동 감지 시도)
    title : str
        제목
    """
    print(f"=== 시계열 종합 분석: {title} ===\n")

    # 기본 통계
    print("1. 기본 통계")
    print(f"   관측치 수: {len(series)}")
    print(f"   평균: {series.mean():.4f}")
    print(f"   표준편차: {series.std():.4f}")
    print(f"   최소/최대: {series.min():.4f} / {series.max():.4f}")
    print(f"   기간: {series.index.min()} ~ {series.index.max()}")
    print()

    # 정상성 검정
    print("2. 정상성 검정")
    adf_result = adfuller(series.dropna())
    print(f"   ADF 통계량: {adf_result[0]:.4f}")
    print(f"   ADF p-value: {adf_result[1]:.4f}")

    kpss_result = kpss(series.dropna(), regression='c')
    print(f"   KPSS 통계량: {kpss_result[0]:.4f}")
    print(f"   KPSS p-value: {kpss_result[1]:.4f}")

    if adf_result[1] < 0.05 and kpss_result[1] >= 0.05:
        print("   결론: 정상 시계열")
        is_stationary = True
    elif adf_result[1] >= 0.05:
        print("   결론: 비정상 (차분 필요)")
        is_stationary = False
    else:
        print("   결론: 추가 분석 필요")
        is_stationary = False
    print()

    # 시각화
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # 원본 시계열
    series.plot(ax=axes[0, 0])
    axes[0, 0].set_title(f'원본 시계열: {title}')
    axes[0, 0].grid(True, alpha=0.3)

    # 분포
    series.hist(bins=30, ax=axes[0, 1], density=True, alpha=0.7)
    axes[0, 1].set_title('분포')
    axes[0, 1].grid(True, alpha=0.3)

    # ACF
    plot_acf(series.dropna(), lags=min(40, len(series)//4), ax=axes[1, 0], alpha=0.05)
    axes[1, 0].set_title('ACF')

    # PACF
    plot_pacf(series.dropna(), lags=min(40, len(series)//4), ax=axes[1, 1],
              alpha=0.05, method='ywm')
    axes[1, 1].set_title('PACF')

    # 차분
    diff_series = series.diff().dropna()
    diff_series.plot(ax=axes[2, 0])
    axes[2, 0].set_title(f'1차 차분 (ADF p={adfuller(diff_series)[1]:.4f})')
    axes[2, 0].grid(True, alpha=0.3)

    # 차분 ACF
    plot_acf(diff_series, lags=min(40, len(diff_series)//4), ax=axes[2, 1], alpha=0.05)
    axes[2, 1].set_title('차분 후 ACF')

    plt.tight_layout()
    plt.show()

    # 분해 (period가 주어진 경우)
    if period is not None and len(series) >= 2 * period:
        print("3. 시계열 분해")
        try:
            stl = STL(series.dropna(), period=period, robust=True)
            result = stl.fit()

            fig, axes = plt.subplots(4, 1, figsize=(12, 10))

            result.observed.plot(ax=axes[0])
            axes[0].set_ylabel('Observed')
            axes[0].set_title('STL 분해')

            result.trend.plot(ax=axes[1])
            axes[1].set_ylabel('Trend')

            result.seasonal.plot(ax=axes[2])
            axes[2].set_ylabel('Seasonal')

            result.resid.plot(ax=axes[3])
            axes[3].set_ylabel('Residual')

            for ax in axes:
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # 계절성 강도
            var_seasonal = result.seasonal.var()
            var_resid = result.resid.var()
            seasonal_strength = max(0, 1 - var_resid / (var_seasonal + var_resid))
            print(f"   계절성 강도: {seasonal_strength:.3f}")

        except Exception as e:
            print(f"   분해 실패: {e}")

    return is_stationary

# 실제 데이터 분석
# CO2 데이터 분석
co2_data = co2.load_pandas().data['co2'].resample('M').mean().dropna()
comprehensive_time_series_analysis(co2_data, period=12, title='Mauna Loa CO2')
```

---

## 8. 연습 문제

### 문제 1: 정상성 판단
다음 시계열이 정상인지 판단하고 이유를 설명하세요:
1. yₜ = 0.5yₜ₋₁ + εₜ
2. yₜ = yₜ₋₁ + εₜ
3. yₜ = t + εₜ
4. yₜ = sin(2πt/12) + εₜ

### 문제 2: 차분 횟수 결정
ADF 검정을 사용하여 다음 데이터를 정상화하는데 필요한 차분 횟수를 결정하세요:
- 랜덤 워크 with drift: yₜ = 0.1 + yₜ₋₁ + εₜ

### 문제 3: ACF/PACF 해석
다음 ACF/PACF 패턴에 해당하는 모델을 식별하세요:
1. ACF: lag 3까지 유의, 이후 0  |  PACF: 지수적 감소
2. ACF: 지수적 감소  |  PACF: lag 2까지 유의, 이후 0

### 문제 4: 시계열 분해
월별 데이터가 다음과 같은 특성을 보일 때:
- 계절 변동의 폭이 점점 커짐
- 장기적 상승 추세

어떤 분해 모형이 적절하고, 어떻게 전처리하면 좋을까요?

---

## 9. 핵심 요약

### 시계열 분석 체크리스트

1. **시각화**: 데이터 패턴 확인 (추세, 계절성, 이상치)
2. **정상성 검정**: ADF + KPSS
3. **변환/차분**: 비정상 → 정상
4. **ACF/PACF 분석**: 모델 식별
5. **분해**: 구성요소 분리 및 이해

### 주요 검정 및 해석

| 검정 | H₀ | 결론 기준 |
|------|-----|----------|
| ADF | 단위근 존재 | p < 0.05 → 정상 |
| KPSS | 정상 | p < 0.05 → 비정상 |

### ACF/PACF 패턴

| 모델 | ACF | PACF |
|------|-----|------|
| AR(p) | 점진적 감소 | lag p 후 절단 |
| MA(q) | lag q 후 절단 | 점진적 감소 |
| ARMA | 점진적 감소 | 점진적 감소 |

### 다음 장 미리보기

11장 **시계열 모형**에서는:
- AR, MA, ARMA 모형
- ARIMA 모형과 차분
- 모델 식별 및 추정
- 예측과 평가
