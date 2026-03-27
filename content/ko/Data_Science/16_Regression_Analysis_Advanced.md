# 16. 회귀분석 심화 (Advanced Regression Analysis)

[이전: 분산분석](./15_ANOVA.md) | [다음: 일반화 선형모형](./17_Generalized_Linear_Models.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. `statsmodels`를 사용하여 다중회귀모형(Multiple Regression)을 적합하고 해석할 수 있으며, 표준화 계수(Standardized Coefficients)도 포함하여 분석할 수 있다
2. 진단 플롯과 공식 검정을 통해 선형성(Linearity), 독립성(Independence), 등분산성(Homoscedasticity), 정규성(Normality)의 네 가지 회귀 가정을 모두 평가할 수 있다
3. 레버리지(Leverage), Cook의 거리(Cook's Distance), DFFITS를 사용하여 영향력 있는 관측치를 식별할 수 있다
4. VIF(분산팽창지수, Variance Inflation Factor)로 다중공선성(Multicollinearity)을 진단하고 변수 제거, 주성분분석(PCA), 릿지 회귀(Ridge Regression) 등의 해결책을 적용할 수 있다
5. 전진선택법(Forward Selection), 후진제거법(Backward Elimination), 단계적 선택법(Stepwise Selection)을 구현할 수 있다
6. 수정 결정계수(Adjusted R-squared), AIC, BIC, 교차검증(Cross-validation)을 사용하여 후보 모형들을 비교할 수 있다
7. 부분회귀 플롯(Partial Regression Plot)과 CCPR 플롯을 활용하여 개별 예측변수의 기여도를 시각적으로 평가할 수 있다

---

단순회귀분석(Simple Regression)은 하나의 변수가 다른 변수를 예측하는지 알려줍니다. 반면 다중회귀분석(Multiple Regression)은 여러 예측변수가 함께 어떻게 작용하는지, 그리고 어떤 경우에 오해를 불러일으킬 수 있는지를 보여줍니다. 공선성(Collinear) 예측변수는 계수 부호를 뒤집을 수 있고, 가정 위반은 p-값을 무효화할 수 있으며, 관련 없는 변수를 포함하면 분산이 증가합니다. 이 레슨은 이러한 함정을 진단하고 신뢰할 수 있는 회귀모형을 체계적으로 구축하는 방법을 가르칩니다.

---

## 1. 다중회귀분석 기초

### 1.1 모형과 추정

```python
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 예시 데이터 생성
np.random.seed(42)
n = 200

# 독립변수
X1 = np.random.normal(50, 10, n)  # 경력(월)
X2 = np.random.normal(70, 15, n)  # 교육점수
X3 = np.random.normal(30, 5, n)   # 나이

# 종속변수 (급여): 선형관계 + 노이즈
Y = 30000 + 500*X1 + 200*X2 + 100*X3 + np.random.normal(0, 5000, n)

# DataFrame 생성
df = pd.DataFrame({
    'salary': Y,
    'experience': X1,
    'education': X2,
    'age': X3
})

# 기본 통계
print("데이터 기술통계:")
print(df.describe())

# 상관행렬
print("\n상관행렬:")
print(df.corr().round(3))
```

### 1.2 statsmodels OLS

```python
# OLS 회귀분석

# 방법 1: formula API (R 스타일)
model_formula = smf.ols('salary ~ experience + education + age', data=df).fit()

# 방법 2: 행렬 API
X = df[['experience', 'education', 'age']]
X = sm.add_constant(X)  # 상수항 추가
y = df['salary']
model_matrix = sm.OLS(y, X).fit()

# 결과 출력
print("회귀분석 결과:")
print(model_formula.summary())

# 주요 통계량 추출
print("\n주요 통계량:")
print(f"R-squared: {model_formula.rsquared:.4f}")
print(f"Adjusted R-squared: {model_formula.rsquared_adj:.4f}")
print(f"F-statistic: {model_formula.fvalue:.2f}")
print(f"F p-value: {model_formula.f_pvalue:.4e}")
print(f"AIC: {model_formula.aic:.2f}")
print(f"BIC: {model_formula.bic:.2f}")
```

### 1.3 회귀계수 해석

```python
# 회귀계수 상세 분석
print("회귀계수 분석:")
print("-" * 70)
print(f"{'변수':<15} {'계수':<12} {'표준오차':<12} {'t-값':<10} {'p-값':<12} {'95% CI'}")
print("-" * 70)

coef_table = pd.DataFrame({
    'coef': model_formula.params,
    'std_err': model_formula.bse,
    't_value': model_formula.tvalues,
    'p_value': model_formula.pvalues,
    'ci_lower': model_formula.conf_int()[0],
    'ci_upper': model_formula.conf_int()[1]
})

for idx, row in coef_table.iterrows():
    ci_str = f"[{row['ci_lower']:.1f}, {row['ci_upper']:.1f}]"
    sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else ""))
    print(f"{idx:<15} {row['coef']:<12.2f} {row['std_err']:<12.2f} {row['t_value']:<10.2f} {row['p_value']:<10.4f}{sig:<2} {ci_str}")

# 표준화 계수 (Beta coefficients)
print("\n표준화 계수 (Beta):")
df_standardized = (df - df.mean()) / df.std()
model_std = smf.ols('salary ~ experience + education + age', data=df_standardized).fit()

for var in ['experience', 'education', 'age']:
    print(f"  {var}: {model_std.params[var]:.4f}")
```

---

## 2. 회귀 가정 검토

### 2.1 가정 개요

```python
# 회귀분석의 가정:
# 1. 선형성 (Linearity): Y와 X의 관계가 선형
# 2. 독립성 (Independence): 잔차가 독립
# 3. 등분산성 (Homoscedasticity): 잔차의 분산이 일정
# 4. 정규성 (Normality): 잔차가 정규분포

# 진단을 위한 값들
fitted = model_formula.fittedvalues
residuals = model_formula.resid
standardized_residuals = model_formula.get_influence().resid_studentized_internal
```

### 2.2 잔차 분석

```python
# 종합 진단 플롯
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 잔차 vs 적합값 (선형성, 등분산성)
axes[0, 0].scatter(fitted, residuals, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('적합값')
axes[0, 0].set_ylabel('잔차')
axes[0, 0].set_title('잔차 vs 적합값 (선형성, 등분산성)')

# Lowess 곡선 추가
from statsmodels.nonparametric.smoothers_lowess import lowess
z = lowess(residuals, fitted, frac=0.3)
axes[0, 0].plot(z[:, 0], z[:, 1], 'g-', linewidth=2, label='Lowess')
axes[0, 0].legend()

# 2. Q-Q Plot (정규성)
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Q-Q Plot (정규성)')

# 3. Scale-Location Plot (등분산성)
sqrt_std_residuals = np.sqrt(np.abs(standardized_residuals))
axes[1, 0].scatter(fitted, sqrt_std_residuals, alpha=0.5)
axes[1, 0].set_xlabel('적합값')
axes[1, 0].set_ylabel('√|표준화 잔차|')
axes[1, 0].set_title('Scale-Location (등분산성)')

z = lowess(sqrt_std_residuals, fitted, frac=0.3)
axes[1, 0].plot(z[:, 0], z[:, 1], 'g-', linewidth=2)

# 4. 잔차 히스토그램 (정규성)
axes[1, 1].hist(residuals, bins=30, density=True, alpha=0.7, edgecolor='black')
xmin, xmax = axes[1, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
axes[1, 1].plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
                'r-', linewidth=2, label='정규분포')
axes[1, 1].set_xlabel('잔차')
axes[1, 1].set_ylabel('밀도')
axes[1, 1].set_title('잔차 히스토그램 (정규성)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

### 2.3 정규성 검정

```python
# 잔차의 정규성 검정

# Shapiro-Wilk 검정
stat_shapiro, p_shapiro = stats.shapiro(residuals)
print(f"Shapiro-Wilk 검정: W = {stat_shapiro:.4f}, p = {p_shapiro:.4f}")

# Kolmogorov-Smirnov 검정
stat_ks, p_ks = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
print(f"Kolmogorov-Smirnov 검정: D = {stat_ks:.4f}, p = {p_ks:.4f}")

# Jarque-Bera 검정
from scipy.stats import jarque_bera
stat_jb, p_jb = jarque_bera(residuals)
print(f"Jarque-Bera 검정: JB = {stat_jb:.4f}, p = {p_jb:.4f}")

# 왜도와 첨도
print(f"\n왜도 (Skewness): {stats.skew(residuals):.4f} (0에 가까우면 대칭)")
print(f"첨도 (Kurtosis): {stats.kurtosis(residuals):.4f} (0에 가까우면 정규분포)")
```

### 2.4 등분산성 검정

```python
# Breusch-Pagan 검정
from statsmodels.stats.diagnostic import het_breuschpagan

bp_stat, bp_pvalue, bp_fstat, bp_f_pvalue = het_breuschpagan(residuals, X)
print(f"Breusch-Pagan 검정:")
print(f"  LM statistic: {bp_stat:.4f}")
print(f"  LM p-value: {bp_pvalue:.4f}")
print(f"  F statistic: {bp_fstat:.4f}")
print(f"  F p-value: {bp_f_pvalue:.4f}")

# White 검정
from statsmodels.stats.diagnostic import het_white

white_stat, white_pvalue, white_fstat, white_f_pvalue = het_white(residuals, X)
print(f"\nWhite 검정:")
print(f"  LM statistic: {white_stat:.4f}")
print(f"  LM p-value: {white_pvalue:.4f}")

# Goldfeld-Quandt 검정
from statsmodels.stats.diagnostic import het_goldfeldquandt

gq_stat, gq_pvalue, gq_alt = het_goldfeldquandt(y, X, alternative='two-sided')
print(f"\nGoldfeld-Quandt 검정:")
print(f"  F statistic: {gq_stat:.4f}")
print(f"  p-value: {gq_pvalue:.4f}")
```

### 2.5 독립성 검정 (자기상관)

```python
# Durbin-Watson 검정
from statsmodels.stats.stattools import durbin_watson

dw_stat = durbin_watson(residuals)
print(f"Durbin-Watson 검정: DW = {dw_stat:.4f}")
print("  DW ≈ 2: 자기상관 없음")
print("  DW < 2: 양의 자기상관")
print("  DW > 2: 음의 자기상관")

# Ljung-Box 검정
from statsmodels.stats.diagnostic import acorr_ljungbox

lb_result = acorr_ljungbox(residuals, lags=[1, 5, 10], return_df=True)
print(f"\nLjung-Box 검정:")
print(lb_result)
```

---

## 3. 진단 플롯 심화

### 3.1 레버리지와 영향력

```python
# 영향력 분석
influence = model_formula.get_influence()

# 레버리지 (Hat values)
leverage = influence.hat_matrix_diag

# Cook's Distance
cooks_d = influence.cooks_distance[0]

# DFFITS
dffits = influence.dffits[0]

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 잔차 vs 레버리지
axes[0, 0].scatter(leverage, standardized_residuals, alpha=0.5)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('레버리지')
axes[0, 0].set_ylabel('표준화 잔차')
axes[0, 0].set_title('잔차 vs 레버리지')

# Cook's D 등고선
n = len(y)
p = len(model_formula.params)
leverage_range = np.linspace(0.001, max(leverage), 100)
for cook_threshold in [0.5, 1]:
    cook_line = np.sqrt(cook_threshold * p * (1 - leverage_range) / leverage_range)
    axes[0, 0].plot(leverage_range, cook_line, 'g--', alpha=0.5)
    axes[0, 0].plot(leverage_range, -cook_line, 'g--', alpha=0.5)

# 2. Cook's Distance
axes[0, 1].stem(range(len(cooks_d)), cooks_d, markerfmt='o', basefmt=' ')
axes[0, 1].axhline(4/n, color='red', linestyle='--', label=f"4/n = {4/n:.4f}")
axes[0, 1].set_xlabel('관측치 인덱스')
axes[0, 1].set_ylabel("Cook's Distance")
axes[0, 1].set_title("Cook's Distance")
axes[0, 1].legend()

# 영향력 있는 관측치 표시
influential = np.where(cooks_d > 4/n)[0]
print(f"영향력 있는 관측치 (Cook's D > 4/n): {influential}")

# 3. 레버리지 분포
axes[1, 0].hist(leverage, bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(2*p/n, color='red', linestyle='--', label=f'2p/n = {2*p/n:.4f}')
axes[1, 0].set_xlabel('레버리지')
axes[1, 0].set_ylabel('밀도')
axes[1, 0].set_title('레버리지 분포')
axes[1, 0].legend()

high_leverage = np.where(leverage > 2*p/n)[0]
print(f"높은 레버리지 관측치: {len(high_leverage)}개")

# 4. DFFITS
axes[1, 1].stem(range(len(dffits)), dffits, markerfmt='o', basefmt=' ')
threshold = 2 * np.sqrt(p/n)
axes[1, 1].axhline(threshold, color='red', linestyle='--')
axes[1, 1].axhline(-threshold, color='red', linestyle='--', label=f'±2√(p/n) = ±{threshold:.4f}')
axes[1, 1].set_xlabel('관측치 인덱스')
axes[1, 1].set_ylabel('DFFITS')
axes[1, 1].set_title('DFFITS')
axes[1, 1].legend()

plt.tight_layout()
plt.show()
```

### 3.2 부분회귀 플롯

```python
# Partial Regression Plots (Added Variable Plots)
fig = sm.graphics.plot_partregress_grid(model_formula)
fig.suptitle('부분회귀 플롯', y=1.02)
plt.tight_layout()
plt.show()

# 개별 부분회귀 플롯
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, var in enumerate(['experience', 'education', 'age']):
    sm.graphics.plot_partregress(
        'salary', var, ['experience', 'education', 'age'],
        data=df, obs_labels=False, ax=axes[i]
    )
    axes[i].set_title(f'salary ~ {var} | 다른 변수들')

plt.tight_layout()
plt.show()
```

### 3.3 성분+잔차 플롯 (CCPR)

```python
# Component-Component plus Residual Plot
fig = sm.graphics.plot_ccpr_grid(model_formula)
fig.suptitle('성분+잔차 플롯 (CCPR)', y=1.02)
plt.tight_layout()
plt.show()
```

---

## 4. 다중공선성 (Multicollinearity)

### 4.1 다중공선성 진단

```python
# 다중공선성이 있는 데이터 생성
np.random.seed(42)
n = 200

X1 = np.random.normal(50, 10, n)
X2 = X1 + np.random.normal(0, 3, n)  # X1과 높은 상관
X3 = np.random.normal(30, 5, n)
Y = 100 + 10*X1 + 5*X2 + 20*X3 + np.random.normal(0, 50, n)

df_multi = pd.DataFrame({
    'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3
})

# 상관행렬
print("상관행렬:")
print(df_multi.corr().round(3))

# 회귀분석
model_multi = smf.ols('Y ~ X1 + X2 + X3', data=df_multi).fit()
print("\n다중공선성이 있는 모형:")
print(model_multi.summary().tables[1])
```

### 4.2 VIF (Variance Inflation Factor)

```python
# VIF 계산
def calculate_vif(df, features):
    """VIF 계산"""
    X = df[features]
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data['feature'] = features
    vif_data['VIF'] = [variance_inflation_factor(X.values, i+1)
                       for i in range(len(features))]
    return vif_data

features = ['X1', 'X2', 'X3']
vif_result = calculate_vif(df_multi, features)

print("VIF (Variance Inflation Factor):")
print(vif_result)
print("\n해석:")
print("  VIF = 1: 다중공선성 없음")
print("  VIF 1-5: 낮은 다중공선성")
print("  VIF 5-10: 중간 다중공선성")
print("  VIF > 10: 높은 다중공선성 (문제)")

# 원본 데이터의 VIF
features_orig = ['experience', 'education', 'age']
vif_orig = calculate_vif(df, features_orig)
print("\n원본 데이터의 VIF:")
print(vif_orig)
```

### 4.3 다중공선성 해결 방법

```python
# 해결 방법 1: 변수 제거
model_reduced = smf.ols('Y ~ X1 + X3', data=df_multi).fit()  # X2 제거

print("변수 제거 후 모형 (X2 제거):")
print(model_reduced.summary().tables[1])

# 해결 방법 2: 주성분 회귀 (PCR)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_multi[['X1', 'X2', 'X3']])

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("\n주성분 분석 결과:")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 설명 분산: {pca.explained_variance_ratio_.cumsum()}")

# PCR 모형
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Y'] = df_multi['Y']

model_pcr = smf.ols('Y ~ PC1 + PC2', data=df_pca).fit()
print("\nPCR 모형:")
print(model_pcr.summary().tables[1])

# 해결 방법 3: Ridge 회귀
from sklearn.linear_model import Ridge

X = df_multi[['X1', 'X2', 'X3']]
y = df_multi['Y']

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

print("\nRidge 회귀 계수:")
for name, coef in zip(['X1', 'X2', 'X3'], ridge.coef_):
    print(f"  {name}: {coef:.4f}")
```

---

## 5. 변수 선택

### 5.1 전진선택, 후진제거, 단계적 선택

```python
# 더 많은 변수가 있는 데이터 생성
np.random.seed(42)
n = 200

# 관련 변수들
X1 = np.random.normal(50, 10, n)  # 중요
X2 = np.random.normal(30, 5, n)   # 중요
X3 = np.random.normal(100, 20, n) # 중요

# 무관한 변수들
X4 = np.random.normal(0, 1, n)
X5 = np.random.normal(0, 1, n)
X6 = np.random.normal(0, 1, n)

Y = 10 + 5*X1 + 3*X2 + 2*X3 + np.random.normal(0, 30, n)

df_select = pd.DataFrame({
    'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3, 'X4': X4, 'X5': X5, 'X6': X6
})

# 전체 모형
model_full = smf.ols('Y ~ X1 + X2 + X3 + X4 + X5 + X6', data=df_select).fit()
print("전체 모형:")
print(model_full.summary().tables[1])
```

```python
def forward_selection(data, response, features, significance_level=0.05):
    """전진선택법"""
    selected = []
    remaining = features.copy()

    while remaining:
        best_pvalue = 1
        best_feature = None

        for feature in remaining:
            formula = f'{response} ~ ' + ' + '.join(selected + [feature])
            model = smf.ols(formula, data=data).fit()
            pvalue = model.pvalues[feature]

            if pvalue < best_pvalue:
                best_pvalue = pvalue
                best_feature = feature

        if best_pvalue < significance_level:
            selected.append(best_feature)
            remaining.remove(best_feature)
            print(f"Added: {best_feature} (p = {best_pvalue:.4f})")
        else:
            break

    return selected

def backward_elimination(data, response, features, significance_level=0.05):
    """후진제거법"""
    selected = features.copy()

    while selected:
        formula = f'{response} ~ ' + ' + '.join(selected)
        model = smf.ols(formula, data=data).fit()

        # 가장 유의하지 않은 변수 찾기
        max_pvalue = 0
        worst_feature = None

        for feature in selected:
            pvalue = model.pvalues[feature]
            if pvalue > max_pvalue:
                max_pvalue = pvalue
                worst_feature = feature

        if max_pvalue > significance_level:
            selected.remove(worst_feature)
            print(f"Removed: {worst_feature} (p = {max_pvalue:.4f})")
        else:
            break

    return selected

# 전진선택
print("전진선택법:")
print("-" * 40)
features = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6']
selected_forward = forward_selection(df_select, 'Y', features)
print(f"선택된 변수: {selected_forward}")

# 후진제거
print("\n후진제거법:")
print("-" * 40)
selected_backward = backward_elimination(df_select, 'Y', features)
print(f"선택된 변수: {selected_backward}")
```

### 5.2 정보 기준 (AIC, BIC)

```python
# 모든 가능한 모형 비교 (작은 변수 집합의 경우)
from itertools import combinations

def all_possible_models(data, response, features):
    """모든 가능한 모형 비교"""
    results = []

    for k in range(1, len(features) + 1):
        for subset in combinations(features, k):
            formula = f'{response} ~ ' + ' + '.join(subset)
            model = smf.ols(formula, data=data).fit()

            results.append({
                'features': subset,
                'n_features': k,
                'R2': model.rsquared,
                'R2_adj': model.rsquared_adj,
                'AIC': model.aic,
                'BIC': model.bic
            })

    return pd.DataFrame(results)

all_models = all_possible_models(df_select, 'Y', features)

# 각 기준별 최적 모형
print("각 기준별 최적 모형:")
print("-" * 60)

best_r2_adj = all_models.loc[all_models['R2_adj'].idxmax()]
best_aic = all_models.loc[all_models['AIC'].idxmin()]
best_bic = all_models.loc[all_models['BIC'].idxmin()]

print(f"최고 Adjusted R²: {best_r2_adj['features']}")
print(f"  R² = {best_r2_adj['R2']:.4f}, Adj R² = {best_r2_adj['R2_adj']:.4f}")

print(f"\n최소 AIC: {best_aic['features']}")
print(f"  AIC = {best_aic['AIC']:.2f}")

print(f"\n최소 BIC: {best_bic['features']}")
print(f"  BIC = {best_bic['BIC']:.2f}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 변수 개수별 최적 모형
for metric, ax, title in zip(['R2_adj', 'AIC', 'BIC'], axes, ['Adjusted R²', 'AIC', 'BIC']):
    for k in range(1, 7):
        subset_models = all_models[all_models['n_features'] == k]
        if metric == 'R2_adj':
            best_val = subset_models[metric].max()
        else:
            best_val = subset_models[metric].min()
        ax.scatter([k], [best_val], s=100)

    ax.set_xlabel('변수 개수')
    ax.set_ylabel(title)
    ax.set_title(f'변수 개수 vs {title}')
    ax.set_xticks(range(1, 7))
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### 5.3 교차검증을 이용한 변수 선택

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# 교차검증으로 최적 모형 찾기
X_all = df_select[features]
y = df_select['Y']

print("교차검증 결과 (5-fold):")
print("-" * 50)

cv_results = []
for k in range(1, len(features) + 1):
    best_cv_score = -np.inf
    best_subset = None

    for subset in combinations(features, k):
        X_subset = df_select[list(subset)]
        model = LinearRegression()
        scores = cross_val_score(model, X_subset, y, cv=5, scoring='r2')
        mean_score = scores.mean()

        if mean_score > best_cv_score:
            best_cv_score = mean_score
            best_subset = subset

    cv_results.append({
        'n_features': k,
        'best_features': best_subset,
        'cv_r2': best_cv_score
    })
    print(f"k={k}: {best_subset}, CV R² = {best_cv_score:.4f}")

# 최적 모형
best_cv = max(cv_results, key=lambda x: x['cv_r2'])
print(f"\n최적 모형: {best_cv['best_features']}")
print(f"CV R² = {best_cv['cv_r2']:.4f}")
```

---

## 6. 실전 예제: 주택 가격 예측

### 6.1 데이터 준비

```python
# 주택 가격 데이터 시뮬레이션
np.random.seed(42)
n = 500

# 특성 생성
size = np.random.normal(1500, 400, n)  # 평방피트
bedrooms = np.random.poisson(3, n) + 1
bathrooms = bedrooms * 0.5 + np.random.normal(0, 0.5, n)
bathrooms = np.clip(bathrooms, 1, 5)
age = np.random.exponential(20, n)
distance = np.random.uniform(1, 30, n)  # 시내까지 거리

# 가격 (비선형 관계 포함)
price = (100000 + 100*size + 20000*bedrooms + 15000*bathrooms
         - 2000*age - 1000*distance
         + 0.02*size*bedrooms  # 상호작용
         + np.random.normal(0, 30000, n))
price = np.maximum(price, 50000)  # 최소 가격

df_house = pd.DataFrame({
    'price': price,
    'size': size,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age,
    'distance': distance
})

print("주택 데이터 기술통계:")
print(df_house.describe())
```

### 6.2 탐색적 분석

```python
# 상관분석
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for i, var in enumerate(['size', 'bedrooms', 'bathrooms', 'age', 'distance']):
    ax = axes[i // 3, i % 3]
    ax.scatter(df_house[var], df_house['price'], alpha=0.3)
    ax.set_xlabel(var)
    ax.set_ylabel('price')

    # 회귀선
    z = np.polyfit(df_house[var], df_house['price'], 1)
    p = np.poly1d(z)
    ax.plot(sorted(df_house[var]), p(sorted(df_house[var])), 'r-')

    # 상관계수
    corr = df_house['price'].corr(df_house[var])
    ax.set_title(f'r = {corr:.3f}')

# 상관행렬 히트맵
axes[1, 2].set_visible(False)
fig.delaxes(axes[1, 2])

plt.tight_layout()
plt.show()

# 상관행렬 히트맵
plt.figure(figsize=(8, 6))
sns.heatmap(df_house.corr(), annot=True, cmap='RdBu_r', center=0,
            fmt='.2f', square=True)
plt.title('상관행렬')
plt.tight_layout()
plt.show()
```

### 6.3 모형 구축 및 진단

```python
# 모형 구축
model_house = smf.ols('price ~ size + bedrooms + bathrooms + age + distance',
                       data=df_house).fit()

print("주택 가격 회귀모형:")
print(model_house.summary())

# 진단
print("\n모형 진단:")
print("-" * 40)

# VIF
features_house = ['size', 'bedrooms', 'bathrooms', 'age', 'distance']
vif_house = calculate_vif(df_house, features_house)
print("\nVIF:")
print(vif_house)

# 정규성 검정
stat, p = stats.shapiro(model_house.resid[:5000])  # Shapiro는 5000개 제한
print(f"\n잔차 정규성 (Shapiro): p = {p:.4f}")

# 등분산성 검정
X_house = sm.add_constant(df_house[features_house])
bp_stat, bp_p, _, _ = het_breuschpagan(model_house.resid, X_house)
print(f"등분산성 (Breusch-Pagan): p = {bp_p:.4f}")
```

### 6.4 모형 개선

```python
# 상호작용 항 추가
model_improved = smf.ols('price ~ size * bedrooms + bathrooms + age + distance',
                          data=df_house).fit()

print("개선된 모형 (상호작용 포함):")
print(model_improved.summary().tables[1])

# AIC/BIC 비교
print(f"\n모형 비교:")
print(f"기본 모형: AIC = {model_house.aic:.2f}, BIC = {model_house.bic:.2f}")
print(f"개선 모형: AIC = {model_improved.aic:.2f}, BIC = {model_improved.bic:.2f}")

# ANOVA로 모형 비교
print("\nANOVA (모형 비교):")
print(sm.stats.anova_lm(model_house, model_improved))
```

---

## 연습 문제

### 문제 1: 회귀 진단
다음 데이터로 회귀분석을 수행하고 가정을 검토하시오.
```python
np.random.seed(42)
X = np.random.normal(0, 1, 100)
Y = 2 + 3*X + np.random.normal(0, 1, 100) * np.abs(X)  # 이분산
```

### 문제 2: 다중공선성
아래 데이터에서 다중공선성을 진단하고 해결하시오.
```python
np.random.seed(42)
X1 = np.random.normal(0, 1, 100)
X2 = 0.9*X1 + np.random.normal(0, 0.1, 100)
X3 = np.random.normal(0, 1, 100)
Y = 1 + 2*X1 + 3*X3 + np.random.normal(0, 1, 100)
```

### 문제 3: 변수 선택
5개의 독립변수 중 최적의 변수 조합을 선택하시오.
- 전진선택, 후진제거, AIC 기준을 각각 적용
- 결과를 비교하시오

---

## 정리

| 진단 항목 | 검정/방법 | Python 함수 |
|-----------|-----------|-------------|
| 정규성 | Shapiro-Wilk | `stats.shapiro()` |
| 등분산성 | Breusch-Pagan | `het_breuschpagan()` |
| 독립성 | Durbin-Watson | `durbin_watson()` |
| 영향력 | Cook's D | `get_influence().cooks_distance` |
| 다중공선성 | VIF | `variance_inflation_factor()` |
| 변수선택 | AIC/BIC | `model.aic`, `model.bic` |
