# 생존 분석(Survival Analysis)

[이전: 인과 추론](./27_Causal_Inference.md) | [다음: 현대 데이터 도구](./29_Modern_Data_Tools.md)

## 개요

생존 분석(Survival Analysis)은 일부 관측값이 중도 절단(censored, 사건이 아직 관측되지 않음)된 시간-사건(time-to-event) 데이터를 모델링합니다. 이 레슨에서는 카플란-마이어 추정량(Kaplan-Meier estimator), 로그 순위 검정(log-rank test), 콕스 비례 위험 모형(Cox proportional hazards model), 모수적 모형(parametric models), 경쟁 위험(competing risks), 그리고 lifelines 라이브러리를 이용한 실전 응용을 다룹니다.

---

## 1. 생존 분석 개념

### 1.1 핵심 정의

```python
"""
주요 개념:

1. 생존 시간(Survival Time, T): 시작 시점에서 사건 발생까지의 시간
   - 의료: 진단 → 사망/재발
   - 비즈니스: 가입 → 이탈
   - 공학: 배포 → 고장

2. 중도 절단(Censoring): 연구 기간 동안 사건이 관측되지 않은 경우
   - 우중도 절단(Right censoring, 가장 흔함): 연구 종료 전에 사건 미발생
   - 좌중도 절단(Left censoring): 관측 시작 전에 사건이 이미 발생
   - 구간 중도 절단(Interval censoring): 두 관측 시점 사이에 사건 발생

   예시 (우중도 절단):
   환자 A: ──────────×  (8개월에 사망)
   환자 B: ────────────── (12개월에 생존 중, 중도 절단)
   환자 C: ──×            (3개월에 사망)
   환자 D: ────────────── (12개월에 생존 중, 중도 절단)

3. 생존 함수(Survival Function): S(t) = P(T > t)
   시간 t를 넘어 생존할 확률.
   S(0) = 1, S(∞) = 0, 단조 감소.

4. 위험 함수(Hazard Function): h(t) = lim_{dt→0} P(t ≤ T < t+dt | T ≥ t) / dt
   시간 t까지 생존했다는 조건 하에 시간 t에서의 순간적 사건 발생률.
   확률이 아님 (1을 초과할 수 있음).

5. 누적 위험(Cumulative Hazard): H(t) = ∫₀ᵗ h(s)ds = -log S(t)
   S(t) = exp(-H(t))
"""
```

---

## 2. 카플란-마이어 추정량(Kaplan-Meier Estimator)

### 2.1 비모수적 생존 곡선(Nonparametric Survival Curve)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

# 임상 시험 시뮬레이션 데이터
np.random.seed(42)
n = 200

# 치료 그룹: 더 긴 생존
treatment_times = np.random.exponential(scale=24, size=n)
treatment_event = np.random.binomial(1, 0.7, n)  # 30% 중도 절단

# 대조 그룹: 더 짧은 생존
control_times = np.random.exponential(scale=16, size=n)
control_event = np.random.binomial(1, 0.7, n)

# 연구 기간(36개월)에 맞게 절단
study_end = 36
treatment_times = np.minimum(treatment_times, study_end)
treatment_event[treatment_times >= study_end] = 0
control_times = np.minimum(control_times, study_end)
control_event[control_times >= study_end] = 0

# 카플란-마이어 적합
kmf_treat = KaplanMeierFitter()
kmf_treat.fit(treatment_times, treatment_event, label="Treatment")

kmf_control = KaplanMeierFitter()
kmf_control.fit(control_times, control_event, label="Control")

# 생존 곡선 플롯
fig, ax = plt.subplots(figsize=(10, 6))
kmf_treat.plot_survival_function(ax=ax, ci_show=True)
kmf_control.plot_survival_function(ax=ax, ci_show=True)
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival Probability")
ax.set_title("Kaplan-Meier Survival Curves")
plt.tight_layout()
plt.show()

# 중앙 생존 시간
print(f"Treatment median survival: {kmf_treat.median_survival_time_:.1f} months")
print(f"Control median survival: {kmf_control.median_survival_time_:.1f} months")

# 특정 시점에서의 요약
for t in [6, 12, 24]:
    s_treat = kmf_treat.predict(t)
    s_control = kmf_control.predict(t)
    print(f"  At {t} months: Treatment S(t)={s_treat:.3f}, Control S(t)={s_control:.3f}")
```

### 2.2 로그 순위 검정(Log-Rank Test)

```python
# 로그 순위 검정: 생존 곡선이 유의미하게 다른가?
result = logrank_test(
    treatment_times, control_times,
    treatment_event, control_event,
)
print(f"\nLog-Rank Test:")
print(f"  Test statistic: {result.test_statistic:.3f}")
print(f"  p-value: {result.p_value:.4f}")
print(f"  Significant (α=0.05): {result.p_value < 0.05}")
```

---

## 3. 콕스 비례 위험 모형(Cox Proportional Hazards Model)

### 3.1 콕스 PH 회귀(Cox PH Regression)

```python
"""
콕스 PH 모형: h(t|X) = h₀(t) * exp(β₁X₁ + β₂X₂ + ...)

  - h₀(t): 기저 위험(baseline hazard, 비모수적으로 비지정)
  - exp(βX): 공변량의 비례적 효과
  - 반모수적(Semi-parametric): 기저 위험 형태에 대한 가정 없음

비례 위험 가정(Proportional Hazards Assumption):
  임의의 두 개체 간 위험비는 시간에 걸쳐 일정하다.
  h(t|X₁) / h(t|X₂) = exp(β(X₁ - X₂))  (t에 의존하지 않음)

해석:
  - exp(β) > 1: 공변량이 위험 증가 (나쁜 예후)
  - exp(β) < 1: 공변량이 위험 감소 (보호 효과)
  - exp(β) = 1: 효과 없음
"""

from lifelines import CoxPHFitter
import numpy as np
import pandas as pd

np.random.seed(42)
N = 500

# 시뮬레이션: 이탈 분석(Churn analysis)
df = pd.DataFrame({
    "tenure": np.random.exponential(12, N).clip(max=36),
    "churned": np.random.binomial(1, 0.6, N),
    "age": np.random.normal(40, 10, N),
    "monthly_charge": np.random.uniform(20, 100, N),
    "contract_type": np.random.choice(["month-to-month", "1-year", "2-year"], N,
                                       p=[0.5, 0.3, 0.2]),
    "tech_support": np.random.binomial(1, 0.4, N),
})

# 요금이 높을수록 → 이탈 높음
df.loc[df.monthly_charge > 70, "churned"] = np.random.binomial(1, 0.8, (df.monthly_charge > 70).sum())
# 계약 기간이 길수록 → 이탈 낮음
df.loc[df.contract_type == "2-year", "tenure"] *= 1.5
df["tenure"] = df["tenure"].clip(max=36)

# 원-핫 인코딩
df_encoded = pd.get_dummies(df, columns=["contract_type"], drop_first=True)

# 콕스 PH 모형 적합
cph = CoxPHFitter()
cph.fit(df_encoded, duration_col="tenure", event_col="churned")

# 결과
cph.print_summary()

# 위험비(Hazard Ratios)
print("\nHazard Ratios (exp(coef)):")
for var, hr in cph.hazard_ratios_.items():
    interpretation = "increases" if hr > 1 else "decreases"
    print(f"  {var}: HR={hr:.3f} → {interpretation} hazard by {abs(hr-1)*100:.1f}%")

# 계수 플롯
cph.plot()
plt.title("Cox PH Coefficients")
plt.tight_layout()
plt.show()
```

### 3.2 비례 위험 가정 검정(Checking the PH Assumption)

```python
# 비례 위험 가정 검정
ph_test = cph.check_assumptions(df_encoded, p_value_threshold=0.05, show_plots=True)

"""
특정 변수에서 PH 가정이 위반된 경우:
  선택지:
  1. 해당 변수로 층화: CoxPHFitter(strata=['variable'])
  2. 시간 가변 계수 포함: 시간과의 교호작용 추가
  3. 모수적 모형(AFT) 사용
"""
```

### 3.3 예측 및 위험 점수(Prediction and Risk Scores)

```python
# 새로운 개체에 대한 생존 함수 예측
new_data = pd.DataFrame({
    "age": [30, 60],
    "monthly_charge": [50, 90],
    "tech_support": [1, 0],
    "contract_type_1-year": [1, 0],
    "contract_type_2-year": [0, 0],
})

# 새로운 개체의 생존 곡선
surv = cph.predict_survival_function(new_data)
fig, ax = plt.subplots(figsize=(10, 6))
surv.plot(ax=ax)
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival Probability")
ax.set_title("Predicted Survival Curves")
ax.legend(["Young, low charge, tech support", "Old, high charge, no support"])
plt.tight_layout()
plt.show()

# 중앙 생존 시간 예측
median_surv = cph.predict_median(new_data)
print(f"Predicted median survival:")
print(f"  Profile 1: {median_surv.iloc[0]:.1f} months")
print(f"  Profile 2: {median_surv.iloc[1]:.1f} months")

# 위험 점수 (부분 위험, partial hazard)
risk = cph.predict_partial_hazard(new_data)
print(f"\nRelative risk scores:")
print(f"  Profile 1: {risk.iloc[0]:.3f}")
print(f"  Profile 2: {risk.iloc[1]:.3f}")
```

---

## 4. 모수적 모형(Parametric Models)

### 4.1 주요 분포(Common Distributions)

```python
"""
모수적 생존 모형 — 생존 시간에 대한 분포를 가정.

| 분포(Distribution)    | 위험 형태(Hazard Shape)       | 사용 사례(Use Case)                  |
|----------------------|------------------------------|--------------------------------------|
| 지수(Exponential)    | 일정 h(t)=λ                  | 무기억성 (실제로는 드묾)              |
| 와이블(Weibull)      | 단조 증가/감소                | 장비 고장, 노화                       |
| 로그 정규(Log-normal) | 비단조 (증가 후 감소)         | 질병 진행                             |
| 로그 로지스틱(Log-logistic) | 비단조                  | 생물학적 시스템                       |
| 곰페르츠(Gompertz)   | 지수적 증가                   | 인간 사망률                           |

와이블(Weibull): h(t) = (ρ/λ)(t/λ)^(ρ-1)
  ρ < 1: 감소하는 위험 (유아 사망률)
  ρ = 1: 일정 (지수 분포)
  ρ > 1: 증가하는 위험 (마모)
"""

from lifelines import (
    WeibullFitter, LogNormalFitter,
    LogLogisticFitter, ExponentialFitter,
    WeibullAFTFitter,
)

# 여러 모수적 모형 적합
fitters = {
    "Exponential": ExponentialFitter(),
    "Weibull": WeibullFitter(),
    "Log-Normal": LogNormalFitter(),
    "Log-Logistic": LogLogisticFitter(),
}

fig, ax = plt.subplots(figsize=(10, 6))
for name, fitter in fitters.items():
    fitter.fit(df["tenure"], df["churned"], label=name)
    fitter.plot_survival_function(ax=ax)
    print(f"{name:15s} AIC={fitter.AIC_:.1f}  BIC={fitter.BIC_:.1f}")

# 비교를 위해 카플란-마이어 추가
kmf = KaplanMeierFitter()
kmf.fit(df["tenure"], df["churned"], label="Kaplan-Meier")
kmf.plot_survival_function(ax=ax, ci_show=False, linestyle="--", color="black")

ax.set_title("Parametric vs Nonparametric Survival")
ax.set_xlabel("Time (months)")
ax.set_ylabel("Survival Probability")
plt.tight_layout()
plt.show()
```

### 4.2 가속 실패 시간 모형(Accelerated Failure Time, AFT)

```python
"""
AFT 모형: log(T) = β₀ + β₁X₁ + ... + σε

  콕스 PH의 대안.
  공변량이 생존 시간을 가속하거나 감속시킴.
  exp(β) > 1 → 실패 지연 (더 긴 생존)
  exp(β) < 1 → 실패 가속 (더 짧은 생존)
"""

 aft = WeibullAFTFitter()
aft.fit(df_encoded, duration_col="tenure", event_col="churned")
aft.print_summary()

print("\nAcceleration Factors:")
for var in aft.params_.index.get_level_values(1).unique():
    if var != "Intercept" and var in aft.params_["lambda_"].index:
        af = np.exp(aft.params_["lambda_"][var])
        effect = "extends" if af > 1 else "shortens"
        print(f"  {var}: AF={af:.3f} → {effect} survival by {abs(af-1)*100:.1f}%")
```

---

## 5. 경쟁 위험(Competing Risks)

### 5.1 누적 발생 함수(Cumulative Incidence Function)

```python
"""
경쟁 위험(Competing Risks): 여러 가능한 사건 (가장 먼저 발생한 것만 중요).

예시: 환자는 다음으로 사망 가능:
  - 심장 질환 (사건 유형 1)
  - 암 (사건 유형 2)
  - 기타 원인 (사건 유형 3)

경쟁 위험이 있을 때 카플란-마이어 추정량은 편향됨.
대신 누적 발생 함수(Cumulative Incidence Function, CIF)를 사용.

CIF_k(t) = P(T ≤ t, event = k)
  = 경쟁 사건을 고려했을 때 시간 t까지 사건 k가 발생할 확률.

모든 CIF의 합 ≤ 임의의 시점에서 1.
"""

# 경쟁 위험 데이터 시뮬레이션
np.random.seed(42)
N = 500

# 각 사건 유형까지의 시간
t1 = np.random.exponential(20, N)  # 심장 질환
t2 = np.random.exponential(25, N)  # 암
t3 = np.random.exponential(40, N)  # 기타

# 관측 시간 = 최솟값
T_obs = np.minimum(np.minimum(t1, t2), t3).clip(max=36)
event_type = np.where(T_obs == t1, 1,
             np.where(T_obs == t2, 2,
             np.where(T_obs == t3, 3, 0)))
# 중도 절단
censored = T_obs >= 36
event_type[censored] = 0

cr_df = pd.DataFrame({
    "time": T_obs,
    "event": event_type,
})

# 각 사건에 대한 원인별(cause-specific) 카플란-마이어
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: 원인별 KM (다른 사건을 중도 절단으로 처리)
for event_id, label in [(1, "Heart Disease"), (2, "Cancer"), (3, "Other")]:
    kmf = KaplanMeierFitter()
    kmf.fit(
        cr_df["time"],
        event_observed=(cr_df["event"] == event_id).astype(int),
        label=label,
    )
    # 1-S(t) = 누적 발생률로 플롯
    ci = 1 - kmf.survival_function_
    ci.plot(ax=axes[0])

axes[0].set_title("Cause-Specific Cumulative Incidence")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Cumulative Incidence")

# 오른쪽: 사건 분포
event_counts = cr_df["event"].value_counts().sort_index()
labels = ["Censored", "Heart Disease", "Cancer", "Other"]
axes[1].bar(labels, [event_counts.get(i, 0) for i in range(4)])
axes[1].set_title("Event Distribution")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()

print(f"Events: {dict(cr_df['event'].value_counts().sort_index())}")
```

---

## 6. 연습 문제

### 연습 1: 고객 이탈 생존 분석

```python
"""
생존 분석을 사용하여 고객 이탈 분석:
1. 통신사 이탈 데이터셋 로드 (또는 5개 이상의 공변량으로 시뮬레이션)
2. 고객 세그먼트별 카플란-마이어 곡선 적합
3. 세그먼트 간 로그 순위 검정 수행
4. 모든 공변량으로 콕스 PH 모형 구축
5. 비례 위험 가정 검정
6. 3개의 고객 프로필에 대한 중앙 생존 시간 예측
7. 이탈의 가장 강력한 예측 변수 파악
"""
```

### 연습 2: 임상 시험 분석

```python
"""
시뮬레이션된 임상 시험 분석:
1. 데이터 생성: 치료군 2개, 환자 300명, 3년 추적 관찰
2. 공변량 포함: 나이, 병기(stage), 바이오마커 수치
3. 카플란-마이어 + 로그 순위 검정으로 군 비교
4. 공변량을 조정한 콕스 PH 모형 적합
5. 와이블 AFT 적합 후 콕스 PH와 비교
6. 12개월 시점의 치료 필요 수(NNT, number-needed-to-treat) 계산
"""
```

---

## 7. 요약

### 핵심 정리

| 개념 | 설명 |
|------|------|
| **중도 절단(Censoring)** | 연구 중 사건 미관측; 분석에서 반드시 처리해야 함 |
| **카플란-마이어(Kaplan-Meier)** | 비모수적 생존 곡선; 공변량 없음 |
| **로그 순위 검정(Log-rank test)** | 그룹 간 생존 비교 |
| **콕스 PH(Cox PH)** | 반모수적 회귀; 위험비(hazard ratios) |
| **와이블 AFT(Weibull AFT)** | 모수적; 위험비 대신 가속 인자(acceleration factors) |
| **경쟁 위험(Competing risks)** | 여러 사건 유형; KM 대신 CIF 사용 |

### 모범 사례

1. **항상 중도 절단 확인** — 중도 절단이 많으면 신뢰할 수 없는 추정치가 나옴
2. **PH 가정 검정** — PH 위반 시 오해를 일으키는 위험비 도출
3. **경쟁 위험에는 CIF 사용** — 사건이 경쟁할 때 카플란-마이어는 편향됨
4. **모수적 적합 비교** — AIC/BIC로 최적 분포 선택
5. **중앙 생존 시간 보고** — 비기술 독자에게 위험비보다 해석이 쉬움

### 다음 단계

- **L29**: 현대 데이터 도구(Modern Data Tools) — 효율적 데이터 처리를 위한 Polars, DuckDB
- **L20** (시계열 기초, Time Series Basics)으로 돌아가 관련 시계열 분석 학습
