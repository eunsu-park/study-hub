# 07. 일반화 선형모형 (Generalized Linear Models)

## 개요

**일반화 선형모형(GLM)**은 종속변수가 정규분포를 따르지 않는 경우에도 적용할 수 있는 회귀분석의 확장입니다. 이진 데이터(로지스틱 회귀), 카운트 데이터(포아송 회귀) 등 다양한 유형의 종속변수를 다룰 수 있습니다.

---

## 1. GLM 프레임워크

### 1.1 GLM의 구성요소

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import links

# GLM의 세 가지 구성요소:
# 1. 랜덤 성분 (Random Component): Y의 확률분포 (지수족)
# 2. 체계적 성분 (Systematic Component): η = β₀ + β₁X₁ + ... + βₚXₚ
# 3. 링크 함수 (Link Function): g(μ) = η, μ = E[Y]

# 주요 GLM 유형
glm_types = pd.DataFrame({
    'Model': ['Linear', 'Logistic', 'Poisson', 'Gamma', 'Negative Binomial'],
    'Distribution': ['Normal', 'Binomial', 'Poisson', 'Gamma', 'Negative Binomial'],
    'Link Function': ['Identity', 'Logit', 'Log', 'Inverse', 'Log'],
    'Y Type': ['Continuous', 'Binary/Proportion', 'Count', 'Positive Continuous', 'Over-dispersed Count']
})

print("GLM 유형:")
print(glm_types.to_string(index=False))
```

### 1.2 링크 함수

```python
# 주요 링크 함수들

def identity_link(mu):
    """항등 링크: η = μ"""
    return mu

def logit_link(mu):
    """로짓 링크: η = log(μ/(1-μ))"""
    return np.log(mu / (1 - mu))

def log_link(mu):
    """로그 링크: η = log(μ)"""
    return np.log(mu)

def inverse_link(mu):
    """역수 링크: η = 1/μ"""
    return 1 / mu

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Identity
mu = np.linspace(-2, 2, 100)
axes[0, 0].plot(mu, identity_link(mu), 'b-', linewidth=2)
axes[0, 0].set_xlabel('μ')
axes[0, 0].set_ylabel('η = g(μ)')
axes[0, 0].set_title('Identity Link: η = μ')
axes[0, 0].grid(alpha=0.3)

# Logit
mu = np.linspace(0.01, 0.99, 100)
axes[0, 1].plot(mu, logit_link(mu), 'r-', linewidth=2)
axes[0, 1].set_xlabel('μ (probability)')
axes[0, 1].set_ylabel('η = logit(μ)')
axes[0, 1].set_title('Logit Link: η = log(μ/(1-μ))')
axes[0, 1].grid(alpha=0.3)

# Log
mu = np.linspace(0.1, 10, 100)
axes[1, 0].plot(mu, log_link(mu), 'g-', linewidth=2)
axes[1, 0].set_xlabel('μ')
axes[1, 0].set_ylabel('η = log(μ)')
axes[1, 0].set_title('Log Link: η = log(μ)')
axes[1, 0].grid(alpha=0.3)

# Inverse logit (sigmoid) - 역함수
eta = np.linspace(-6, 6, 100)
sigmoid = 1 / (1 + np.exp(-eta))
axes[1, 1].plot(eta, sigmoid, 'purple', linewidth=2)
axes[1, 1].set_xlabel('η = Xβ')
axes[1, 1].set_ylabel('μ = P(Y=1)')
axes[1, 1].set_title('Inverse Logit (Sigmoid): μ = 1/(1+e^(-η))')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].axhline(0.5, color='gray', linestyle='--', alpha=0.5)
axes[1, 1].axvline(0, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

### 1.3 지수족 분포 (Exponential Family)

```python
# 지수족 분포의 일반 형태:
# f(y|θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)}
# θ: canonical parameter (자연모수)
# φ: dispersion parameter (산포모수)

# 주요 분포의 평균과 분산
print("지수족 분포의 평균-분산 관계:")
print("-" * 50)
print(f"{'분포':<15} {'평균 (μ)':<20} {'분산 (V(μ))'}")
print("-" * 50)
print(f"{'Normal':<15} {'μ':<20} {'σ²'}")
print(f"{'Binomial':<15} {'nπ':<20} {'nπ(1-π)'}")
print(f"{'Poisson':<15} {'λ':<20} {'λ'}")
print(f"{'Gamma':<15} {'μ':<20} {'μ²/ν'}")
print(f"{'Inverse Gaussian':<15} {'μ':<20} {'μ³/λ'}")
```

---

## 2. 로지스틱 회귀 (Logistic Regression)

### 2.1 이진 로지스틱 회귀

```python
# 데이터 생성: 고객 이탈 예측
np.random.seed(42)
n = 500

# 독립변수
age = np.random.normal(40, 10, n)
tenure = np.random.exponential(24, n)  # 가입기간 (월)
monthly_charge = np.random.normal(65, 20, n)

# 로짓 모형
eta = -5 + 0.05*age - 0.03*tenure + 0.04*monthly_charge
prob = 1 / (1 + np.exp(-eta))
churn = np.random.binomial(1, prob, n)

df_churn = pd.DataFrame({
    'churn': churn,
    'age': age,
    'tenure': tenure,
    'monthly_charge': monthly_charge
})

print("이탈률:", df_churn['churn'].mean())
print("\n데이터 샘플:")
print(df_churn.head())
```

### 2.2 모형 적합

```python
# statsmodels로 로지스틱 회귀
model_logit = smf.glm('churn ~ age + tenure + monthly_charge',
                       data=df_churn,
                       family=sm.families.Binomial()).fit()

print("로지스틱 회귀 결과:")
print(model_logit.summary())

# 또는 logit() 함수 사용
model_logit2 = smf.logit('churn ~ age + tenure + monthly_charge', data=df_churn).fit()
```

### 2.3 계수 해석

```python
# 오즈비 (Odds Ratio) 계산
odds_ratios = np.exp(model_logit.params)
conf_int = np.exp(model_logit.conf_int())

print("오즈비 (Odds Ratios):")
print("-" * 60)
print(f"{'변수':<20} {'OR':<10} {'95% CI':<25} {'p-value'}")
print("-" * 60)

for var in model_logit.params.index:
    or_val = odds_ratios[var]
    ci_low, ci_high = conf_int.loc[var]
    pval = model_logit.pvalues[var]
    print(f"{var:<20} {or_val:<10.4f} [{ci_low:.4f}, {ci_high:.4f}] {pval:<.4f}")

print("\n해석 예시:")
print(f"- tenure가 1개월 증가하면 이탈 오즈가 {(odds_ratios['tenure']-1)*100:.2f}% 변화")
print(f"  (OR = {odds_ratios['tenure']:.4f})")
```

### 2.4 예측과 평가

```python
# 예측 확률
df_churn['pred_prob'] = model_logit.predict(df_churn)
df_churn['pred_class'] = (df_churn['pred_prob'] >= 0.5).astype(int)

# 혼동 행렬
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

cm = confusion_matrix(df_churn['churn'], df_churn['pred_class'])
print("혼동 행렬:")
print(cm)

print("\n분류 보고서:")
print(classification_report(df_churn['churn'], df_churn['pred_class']))

# ROC 곡선
fpr, tpr, thresholds = roc_curve(df_churn['churn'], df_churn['pred_prob'])
auc = roc_auc_score(df_churn['churn'], df_churn['pred_prob'])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ROC 곡선
axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC 곡선')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 예측 확률 분포
axes[1].hist(df_churn[df_churn['churn']==0]['pred_prob'], bins=30, alpha=0.5,
             label='유지 (y=0)', density=True)
axes[1].hist(df_churn[df_churn['churn']==1]['pred_prob'], bins=30, alpha=0.5,
             label='이탈 (y=1)', density=True)
axes[1].set_xlabel('예측 확률')
axes[1].set_ylabel('밀도')
axes[1].set_title('예측 확률 분포')
axes[1].legend()
axes[1].axvline(0.5, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print(f"AUC: {auc:.4f}")
```

### 2.5 다항 로지스틱 회귀 (Multinomial Logistic Regression)

```python
# 3개 이상 범주의 종속변수
np.random.seed(42)
n = 600

# 3가지 선택: A, B, C
X1 = np.random.normal(0, 1, n)
X2 = np.random.normal(0, 1, n)

# 다항 로짓 모형 (기준 범주: C)
eta_A = 0.5 + 1.5*X1 - 0.5*X2
eta_B = -0.5 - 0.5*X1 + 1.5*X2
eta_C = np.zeros(n)  # 기준 범주

# 소프트맥스 확률
exp_eta = np.exp(np.column_stack([eta_A, eta_B, eta_C]))
probs = exp_eta / exp_eta.sum(axis=1, keepdims=True)

# 범주 선택
choice = np.array(['A', 'B', 'C'])[np.array([np.random.choice(3, p=p) for p in probs])]

df_multi = pd.DataFrame({
    'choice': choice,
    'X1': X1,
    'X2': X2
})

print("범주별 빈도:")
print(df_multi['choice'].value_counts())

# statsmodels MNLogit
from statsmodels.discrete.discrete_model import MNLogit

# 범주를 숫자로 변환
df_multi['choice_code'] = df_multi['choice'].map({'A': 0, 'B': 1, 'C': 2})

X_multi = sm.add_constant(df_multi[['X1', 'X2']])
model_mnlogit = MNLogit(df_multi['choice_code'], X_multi).fit()

print("\n다항 로지스틱 회귀 결과:")
print(model_mnlogit.summary())
```

---

## 3. 포아송 회귀 (Poisson Regression)

### 3.1 기본 포아송 회귀

```python
# 데이터 생성: 웹사이트 방문 횟수
np.random.seed(42)
n = 400

# 독립변수
ad_spend = np.random.exponential(100, n)  # 광고비
day_of_week = np.random.randint(0, 7, n)  # 요일
is_weekend = (day_of_week >= 5).astype(int)

# 포아송 모형
log_lambda = 2 + 0.01*ad_spend - 0.5*is_weekend
lambda_param = np.exp(log_lambda)
visits = np.random.poisson(lambda_param)

df_poisson = pd.DataFrame({
    'visits': visits,
    'ad_spend': ad_spend,
    'is_weekend': is_weekend
})

print("방문 횟수 통계:")
print(df_poisson['visits'].describe())

# 포아송 분포 확인
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df_poisson['visits'], bins=30, density=True, alpha=0.7, edgecolor='black')
ax.set_xlabel('방문 횟수')
ax.set_ylabel('밀도')
ax.set_title('방문 횟수 분포')
plt.show()
```

### 3.2 모형 적합

```python
# 포아송 회귀
model_poisson = smf.glm('visits ~ ad_spend + is_weekend',
                         data=df_poisson,
                         family=sm.families.Poisson()).fit()

print("포아송 회귀 결과:")
print(model_poisson.summary())

# 발생비율 (Incidence Rate Ratio)
irr = np.exp(model_poisson.params)
irr_ci = np.exp(model_poisson.conf_int())

print("\n발생비율 (IRR):")
print("-" * 60)
for var in model_poisson.params.index:
    irr_val = irr[var]
    ci_low, ci_high = irr_ci.loc[var]
    print(f"{var}: IRR = {irr_val:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

print("\n해석:")
print(f"- 주말에는 방문 횟수가 평균 {(1-irr['is_weekend'])*100:.1f}% 감소")
print(f"- 광고비 1단위 증가 시 방문 횟수가 {(irr['ad_spend']-1)*100:.2f}% 증가")
```

### 3.3 오프셋 (Offset) 사용

```python
# 오프셋: 노출 시간/공간이 다를 때 사용
# log(λ/t) = β₀ + β₁X  →  log(λ) = log(t) + β₀ + β₁X

np.random.seed(42)
n = 300

# 지역별 범죄 건수 (인구 크기가 다름)
population = np.random.uniform(10000, 100000, n)
poverty_rate = np.random.uniform(5, 30, n)
unemployment = np.random.uniform(3, 15, n)

# 범죄율 = 인구 × exp(β)
log_crime_rate = -6 + 0.05*poverty_rate + 0.03*unemployment
crime_rate = np.exp(log_crime_rate)
crimes = np.random.poisson(population * crime_rate)

df_crime = pd.DataFrame({
    'crimes': crimes,
    'population': population,
    'poverty_rate': poverty_rate,
    'unemployment': unemployment,
    'log_population': np.log(population)
})

# 오프셋 포함 모형
model_offset = smf.glm('crimes ~ poverty_rate + unemployment',
                        data=df_crime,
                        family=sm.families.Poisson(),
                        offset=df_crime['log_population']).fit()

print("오프셋 포함 포아송 회귀:")
print(model_offset.summary())

# 오프셋 없는 모형 (잘못된 모형)
model_no_offset = smf.glm('crimes ~ poverty_rate + unemployment',
                           data=df_crime,
                           family=sm.families.Poisson()).fit()

print("\n오프셋 없는 모형 (비교용):")
print(f"poverty_rate 계수: {model_no_offset.params['poverty_rate']:.4f}")
print(f"오프셋 포함 계수: {model_offset.params['poverty_rate']:.4f} (더 정확)")
```

### 3.4 과대산포 (Overdispersion)

```python
# 포아송 분포 가정: E[Y] = Var(Y) = λ
# 과대산포: Var(Y) > E[Y]

# 과대산포 확인
def check_overdispersion(model):
    """과대산포 검정"""
    pearson_chi2 = model.pearson_chi2
    df_resid = model.df_resid
    dispersion = pearson_chi2 / df_resid

    print(f"Pearson χ²: {pearson_chi2:.2f}")
    print(f"자유도: {df_resid}")
    print(f"과대산포 모수: {dispersion:.4f}")
    print(f"→ {dispersion:.2f} > 1 이면 과대산포 의심")

    return dispersion

print("과대산포 검정:")
print("-" * 40)
dispersion = check_overdispersion(model_poisson)

# 과대산포 데이터 생성
np.random.seed(42)
n = 500

X = np.random.normal(0, 1, n)
# 음이항 분포로 과대산포 데이터 생성
log_mu = 2 + 0.5*X
mu = np.exp(log_mu)
# r=5로 음이항 (분산 > 평균)
Y_overdispersed = np.random.negative_binomial(5, 5/(5+mu), n)

df_over = pd.DataFrame({'Y': Y_overdispersed, 'X': X})

print(f"\n과대산포 데이터:")
print(f"평균: {df_over['Y'].mean():.2f}")
print(f"분산: {df_over['Y'].var():.2f}")
print(f"분산/평균 비율: {df_over['Y'].var()/df_over['Y'].mean():.2f}")

# 포아송 모형 (잘못된 모형)
model_pois_over = smf.glm('Y ~ X', data=df_over, family=sm.families.Poisson()).fit()
print("\n포아송 모형 과대산포:")
check_overdispersion(model_pois_over)
```

### 3.5 음이항 회귀 (Negative Binomial Regression)

```python
# 과대산포 해결: 음이항 회귀
from statsmodels.genmod.families import NegativeBinomial

model_nb = smf.glm('Y ~ X', data=df_over,
                    family=NegativeBinomial(alpha=1)).fit()

print("음이항 회귀 결과:")
print(model_nb.summary())

# 포아송 vs 음이항 비교
print("\n모형 비교:")
print("-" * 40)
print(f"{'모형':<15} {'AIC':<12} {'Deviance':<12}")
print("-" * 40)
print(f"{'Poisson':<15} {model_pois_over.aic:<12.2f} {model_pois_over.deviance:<12.2f}")
print(f"{'Neg Binomial':<15} {model_nb.aic:<12.2f} {model_nb.deviance:<12.2f}")

# 준포아송 (Quasi-Poisson) 회귀
# 분산을 φμ로 모델링 (φ는 추정)
model_quasi = smf.glm('Y ~ X', data=df_over,
                       family=sm.families.Poisson()).fit(scale='X2')  # Pearson χ² 기반

print(f"\n준포아송 회귀 산포모수: {model_quasi.scale:.4f}")
```

---

## 4. 모형 진단

### 4.1 잔차 분석

```python
# GLM 잔차 유형
# 1. Pearson 잔차: (Y - μ̂) / √V(μ̂)
# 2. Deviance 잔차: sign(Y-μ̂) × √d_i
# 3. 표준화 잔차

# 포아송 모형 진단
model = model_poisson

# 잔차 계산
pearson_resid = model.resid_pearson
deviance_resid = model.resid_deviance
fitted = model.fittedvalues

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Pearson 잔차 vs 적합값
axes[0, 0].scatter(fitted, pearson_resid, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('적합값')
axes[0, 0].set_ylabel('Pearson 잔차')
axes[0, 0].set_title('Pearson 잔차 vs 적합값')

# 2. Deviance 잔차 vs 적합값
axes[0, 1].scatter(fitted, deviance_resid, alpha=0.5)
axes[0, 1].axhline(0, color='red', linestyle='--')
axes[0, 1].set_xlabel('적합값')
axes[0, 1].set_ylabel('Deviance 잔차')
axes[0, 1].set_title('Deviance 잔차 vs 적합값')

# 3. 잔차 Q-Q plot
stats.probplot(pearson_resid, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Pearson 잔차 Q-Q Plot')

# 4. 잔차 히스토그램
axes[1, 1].hist(pearson_resid, bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Pearson 잔차')
axes[1, 1].set_ylabel('밀도')
axes[1, 1].set_title('Pearson 잔차 분포')

plt.tight_layout()
plt.show()
```

### 4.2 적합도 검정

```python
# Deviance 검정
deviance = model_poisson.deviance
df = model_poisson.df_resid
p_deviance = 1 - stats.chi2.cdf(deviance, df)

print("적합도 검정:")
print("-" * 40)
print(f"Deviance: {deviance:.2f}")
print(f"자유도: {df}")
print(f"p-value: {p_deviance:.4f}")
print(f"→ p > 0.05이면 모형이 적합")

# Pearson 검정
pearson = model_poisson.pearson_chi2
p_pearson = 1 - stats.chi2.cdf(pearson, df)

print(f"\nPearson χ²: {pearson:.2f}")
print(f"p-value: {p_pearson:.4f}")
```

### 4.3 영향력 진단

```python
# GLM의 영향력 분석
from statsmodels.stats.outliers_influence import GLMInfluence

influence = GLMInfluence(model_poisson)

# Cook's Distance
cooks_d = influence.cooks_distance[0]

# 레버리지 (Hat values)
leverage = influence.hat_matrix_diag

# DFFITS
dffits = influence.dffits[0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Cook's D
n = len(cooks_d)
axes[0].stem(range(n), cooks_d, markerfmt='o', basefmt=' ')
axes[0].axhline(4/n, color='red', linestyle='--', label=f'4/n = {4/n:.4f}')
axes[0].set_xlabel('관측치 인덱스')
axes[0].set_ylabel("Cook's Distance")
axes[0].set_title("Cook's Distance")
axes[0].legend()

# 레버리지
axes[1].hist(leverage, bins=30, alpha=0.7, edgecolor='black')
p = len(model_poisson.params)
axes[1].axvline(2*p/n, color='red', linestyle='--', label=f'2p/n = {2*p/n:.4f}')
axes[1].set_xlabel('레버리지')
axes[1].set_ylabel('빈도')
axes[1].set_title('레버리지 분포')
axes[1].legend()

# DFFITS
axes[2].stem(range(n), dffits, markerfmt='o', basefmt=' ')
threshold = 2 * np.sqrt(p/n)
axes[2].axhline(threshold, color='red', linestyle='--')
axes[2].axhline(-threshold, color='red', linestyle='--', label=f'±2√(p/n)')
axes[2].set_xlabel('관측치 인덱스')
axes[2].set_ylabel('DFFITS')
axes[2].set_title('DFFITS')
axes[2].legend()

plt.tight_layout()
plt.show()

# 영향력 있는 관측치
influential_cooks = np.where(cooks_d > 4/n)[0]
print(f"Cook's D > 4/n인 관측치: {len(influential_cooks)}개")
```

---

## 5. 모형 비교

### 5.1 Deviance 검정

```python
# 중첩 모형 비교: Deviance 차이 검정

# 축소 모형 (Reduced model)
model_reduced = smf.glm('visits ~ ad_spend', data=df_poisson,
                         family=sm.families.Poisson()).fit()

# 완전 모형 (Full model)
model_full = model_poisson

# Deviance 차이 검정
dev_diff = model_reduced.deviance - model_full.deviance
df_diff = model_reduced.df_resid - model_full.df_resid
p_value = 1 - stats.chi2.cdf(dev_diff, df_diff)

print("중첩 모형 비교 (Deviance 검정):")
print("-" * 50)
print(f"축소 모형 Deviance: {model_reduced.deviance:.2f}")
print(f"완전 모형 Deviance: {model_full.deviance:.2f}")
print(f"Deviance 차이: {dev_diff:.2f}")
print(f"자유도 차이: {df_diff}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("→ 완전 모형이 유의하게 더 적합")
else:
    print("→ 축소 모형으로 충분")
```

### 5.2 정보 기준 비교

```python
# AIC, BIC 비교

# 여러 모형 정의
models = {
    'Model 1: ad_spend': smf.glm('visits ~ ad_spend', data=df_poisson,
                                  family=sm.families.Poisson()).fit(),
    'Model 2: is_weekend': smf.glm('visits ~ is_weekend', data=df_poisson,
                                    family=sm.families.Poisson()).fit(),
    'Model 3: both': smf.glm('visits ~ ad_spend + is_weekend', data=df_poisson,
                              family=sm.families.Poisson()).fit(),
}

print("모형 비교 (정보 기준):")
print("-" * 60)
print(f"{'모형':<25} {'AIC':<12} {'BIC':<12} {'Deviance':<12}")
print("-" * 60)

for name, model in models.items():
    print(f"{name:<25} {model.aic:<12.2f} {model.bic:<12.2f} {model.deviance:<12.2f}")

# 최적 모형
best_aic = min(models.items(), key=lambda x: x[1].aic)
best_bic = min(models.items(), key=lambda x: x[1].bic)

print(f"\n최적 모형 (AIC): {best_aic[0]}")
print(f"최적 모형 (BIC): {best_bic[0]}")
```

---

## 6. 실전 예제

### 6.1 보험 청구 건수 예측

```python
# 보험 청구 데이터 시뮬레이션
np.random.seed(42)
n = 1000

# 고객 특성
age = np.random.normal(40, 12, n)
age = np.clip(age, 18, 80)
years_driving = np.clip(age - 18 - np.random.exponential(2, n), 0, None)
previous_claims = np.random.poisson(0.5, n)
vehicle_age = np.random.exponential(5, n)

# 청구 건수 (음이항 분포로 과대산포 시뮬레이션)
log_mu = -1 + 0.02*age - 0.01*years_driving + 0.3*previous_claims + 0.05*vehicle_age
mu = np.exp(log_mu)
claims = np.random.negative_binomial(2, 2/(2+mu), n)

df_insurance = pd.DataFrame({
    'claims': claims,
    'age': age,
    'years_driving': years_driving,
    'previous_claims': previous_claims,
    'vehicle_age': vehicle_age
})

print("보험 청구 데이터:")
print(df_insurance.describe())
print(f"\n청구 건수 평균: {claims.mean():.2f}, 분산: {claims.var():.2f}")
print(f"분산/평균 비율: {claims.var()/claims.mean():.2f} (과대산포)")
```

### 6.2 모형 적합 및 비교

```python
# 포아송 모형
model_pois = smf.glm('claims ~ age + years_driving + previous_claims + vehicle_age',
                      data=df_insurance,
                      family=sm.families.Poisson()).fit()

# 음이항 모형
model_nb = smf.glm('claims ~ age + years_driving + previous_claims + vehicle_age',
                    data=df_insurance,
                    family=NegativeBinomial(alpha=1)).fit()

print("모형 비교:")
print("=" * 70)

# 과대산포 확인
print("\n포아송 모형 과대산포:")
check_overdispersion(model_pois)

# 계수 비교
print("\n계수 비교:")
print("-" * 70)
coef_comparison = pd.DataFrame({
    'Poisson': model_pois.params,
    'Neg Binomial': model_nb.params,
    'Poisson SE': model_pois.bse,
    'NB SE': model_nb.bse
})
print(coef_comparison)

# 정보 기준
print("\n정보 기준:")
print(f"Poisson AIC: {model_pois.aic:.2f}")
print(f"Neg Binomial AIC: {model_nb.aic:.2f}")
print(f"→ 음이항 모형이 더 적합 (낮은 AIC)")
```

### 6.3 최종 모형 해석

```python
# 음이항 모형 해석
print("음이항 회귀 모형 결과:")
print(model_nb.summary())

# 발생비율 (IRR)
irr = np.exp(model_nb.params)
irr_ci = np.exp(model_nb.conf_int())

print("\n발생비율 (Incidence Rate Ratio):")
print("-" * 70)
print(f"{'변수':<20} {'IRR':<10} {'95% CI':<25} {'해석'}")
print("-" * 70)

interpretations = {
    'age': '나이 1세 증가',
    'years_driving': '운전 경력 1년 증가',
    'previous_claims': '이전 청구 1건 증가',
    'vehicle_age': '차량 연식 1년 증가'
}

for var in ['age', 'years_driving', 'previous_claims', 'vehicle_age']:
    irr_val = irr[var]
    ci_low, ci_high = irr_ci.loc[var]
    pct_change = (irr_val - 1) * 100
    direction = "증가" if pct_change > 0 else "감소"
    print(f"{var:<20} {irr_val:<10.4f} [{ci_low:.4f}, {ci_high:.4f}] "
          f"{interpretations[var]} 시 청구 {abs(pct_change):.1f}% {direction}")
```

---

## 연습 문제

### 문제 1: 로지스틱 회귀
다음 데이터로 합격/불합격을 예측하는 로지스틱 회귀 모형을 구축하시오.
```python
np.random.seed(42)
study_hours = np.random.uniform(1, 10, 100)
prep_course = np.random.binomial(1, 0.5, 100)
pass_exam = (0.3*study_hours + 1.5*prep_course + np.random.normal(0, 1, 100) > 3).astype(int)
```

### 문제 2: 포아송 회귀
웹사이트 클릭 수 데이터에 포아송 회귀를 적용하고, 과대산포를 확인하시오.

### 문제 3: 모형 선택
로지스틱 회귀에서 AIC를 기준으로 최적 변수 조합을 선택하시오.

---

## 정리

| GLM 유형 | 분포 | 링크 함수 | Python |
|----------|------|-----------|--------|
| 선형 회귀 | Normal | Identity | `sm.OLS()` |
| 로지스틱 | Binomial | Logit | `smf.logit()` |
| 포아송 | Poisson | Log | `family=Poisson()` |
| 음이항 | Neg Binomial | Log | `family=NegativeBinomial()` |
| 감마 | Gamma | Inverse | `family=Gamma()` |

| 진단 항목 | 방법 | 기준 |
|-----------|------|------|
| 과대산포 | Pearson χ²/df | > 1 |
| 적합도 | Deviance 검정 | p > 0.05 |
| 영향력 | Cook's D | > 4/n |
| 모형 비교 | AIC, BIC | 작을수록 좋음 |
