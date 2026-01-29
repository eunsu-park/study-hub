# 실전 프로젝트 (Real-World Projects)

## 개요

실제 데이터셋을 사용하여 분류와 회귀 문제를 처음부터 끝까지 해결합니다. Kaggle 스타일의 문제 해결 과정과 실무 노하우를 다룹니다.

---

## 1. 머신러닝 프로젝트 워크플로우

### 1.1 전체 프로세스

```python
"""
머신러닝 프로젝트 단계:

1. 문제 정의
   - 비즈니스 목표 이해
   - 성공 지표 정의
   - 분류/회귀/클러스터링 결정

2. 데이터 수집 및 탐색
   - 데이터 로드
   - EDA (탐색적 데이터 분석)
   - 데이터 품질 확인

3. 데이터 전처리
   - 결측치 처리
   - 이상치 처리
   - 특성 엔지니어링
   - 인코딩 및 스케일링

4. 모델링
   - 기준선 모델
   - 모델 선택 및 비교
   - 하이퍼파라미터 튜닝

5. 평가 및 해석
   - 성능 평가
   - 오차 분석
   - 특성 중요도

6. 배포 및 모니터링
   - 모델 저장
   - 예측 API
   - 성능 모니터링
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import warnings
warnings.filterwarnings('ignore')
```

---

## 2. 프로젝트 1: 타이타닉 생존 예측 (분류)

### 2.1 데이터 로드 및 탐색

```python
# 데이터 로드 (실제 Kaggle 데이터 또는 seaborn 내장 데이터)
# df = pd.read_csv('titanic.csv')
df = sns.load_dataset('titanic')

print("=== 데이터 기본 정보 ===")
print(f"데이터 형상: {df.shape}")
print(f"\n컬럼 정보:")
print(df.info())

print(f"\n처음 5행:")
print(df.head())

print(f"\n기술 통계:")
print(df.describe())

print(f"\n타겟 분포:")
print(df['survived'].value_counts(normalize=True))
```

### 2.2 탐색적 데이터 분석 (EDA)

```python
# 결측치 확인
print("=== 결측치 분석 ===")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'결측치 수': missing, '결측치 비율(%)': missing_pct})
print(missing_df[missing_df['결측치 수'] > 0])

# 시각화
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 생존율
sns.countplot(data=df, x='survived', ax=axes[0, 0])
axes[0, 0].set_title('Survival Distribution')

# 성별별 생존율
sns.countplot(data=df, x='sex', hue='survived', ax=axes[0, 1])
axes[0, 1].set_title('Survival by Sex')

# 클래스별 생존율
sns.countplot(data=df, x='pclass', hue='survived', ax=axes[0, 2])
axes[0, 2].set_title('Survival by Class')

# 나이 분포
sns.histplot(data=df, x='age', hue='survived', kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Age Distribution by Survival')

# 요금 분포
sns.histplot(data=df, x='fare', hue='survived', kde=True, ax=axes[1, 1])
axes[1, 1].set_title('Fare Distribution by Survival')

# 승선 항구별 생존율
sns.countplot(data=df, x='embarked', hue='survived', ax=axes[1, 2])
axes[1, 2].set_title('Survival by Embarked')

plt.tight_layout()
plt.show()

# 상관관계
print("\n=== 수치형 변수 상관관계 ===")
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].corr()['survived'].sort_values(ascending=False))
```

### 2.3 데이터 전처리

```python
# 작업용 복사본
df_clean = df.copy()

# 불필요한 컬럼 제거
drop_cols = ['deck', 'embark_town', 'alive', 'who', 'adult_male', 'class']
df_clean = df_clean.drop(columns=drop_cols, errors='ignore')

# 결측치 처리
# 나이: 중간값으로 대체
df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())

# 승선 항구: 최빈값으로 대체
df_clean['embarked'] = df_clean['embarked'].fillna(df_clean['embarked'].mode()[0])

# 특성 엔지니어링
# 가족 크기
df_clean['family_size'] = df_clean['sibsp'] + df_clean['parch'] + 1

# 혼자 여행
df_clean['is_alone'] = (df_clean['family_size'] == 1).astype(int)

# 나이 그룹
df_clean['age_group'] = pd.cut(df_clean['age'],
                                bins=[0, 12, 18, 35, 60, 100],
                                labels=['Child', 'Teen', 'Young', 'Middle', 'Senior'])

# 범주형 인코딩
df_clean['sex'] = LabelEncoder().fit_transform(df_clean['sex'])
df_clean['embarked'] = LabelEncoder().fit_transform(df_clean['embarked'])
df_clean['age_group'] = LabelEncoder().fit_transform(df_clean['age_group'])

# 최종 특성 선택
features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
            'embarked', 'family_size', 'is_alone', 'age_group']
X = df_clean[features]
y = df_clean['survived']

print(f"최종 특성: {features}")
print(f"X 형상: {X.shape}")
```

### 2.4 모델링

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 정의
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

# 모델 비교
print("=== 모델 비교 ===")
results = []

for name, model in models.items():
    # SVM, 로지스틱은 스케일링된 데이터 사용
    if name in ['Logistic Regression']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # 교차 검증
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='accuracy')

    # 학습 및 테스트
    model.fit(X_tr, y_train)
    test_score = model.score(X_te, y_test)

    results.append({
        'Model': name,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test Score': test_score
    })

    print(f"{name}: CV={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f}), Test={test_score:.4f}")

results_df = pd.DataFrame(results)
print(f"\n최고 CV 점수: {results_df.loc[results_df['CV Mean'].idxmax(), 'Model']}")
```

### 2.5 하이퍼파라미터 튜닝

```python
# 최고 성능 모델 튜닝 (예: Random Forest)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("\n=== 하이퍼파라미터 튜닝 결과 ===")
print(f"최적 파라미터: {grid_search.best_params_}")
print(f"최적 CV 점수: {grid_search.best_score_:.4f}")
print(f"테스트 점수: {grid_search.score(X_test, y_test):.4f}")

best_model = grid_search.best_estimator_
```

### 2.6 최종 평가

```python
# 예측
y_pred = best_model.predict(X_test)

# 혼동 행렬
print("=== 최종 평가 ===")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=['Not Survived', 'Survived']))

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 특성 중요도
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

print("\n특성 중요도:")
for i in indices:
    print(f"  {features[i]}: {importances[i]:.4f}")
```

---

## 3. 프로젝트 2: 주택 가격 예측 (회귀)

### 3.1 데이터 로드 및 탐색

```python
from sklearn.datasets import fetch_california_housing

# 캘리포니아 주택 가격 데이터
housing = fetch_california_housing()
df_house = pd.DataFrame(housing.data, columns=housing.feature_names)
df_house['MedHouseVal'] = housing.target

print("=== 주택 가격 데이터 ===")
print(f"데이터 형상: {df_house.shape}")
print(f"\n컬럼: {list(df_house.columns)}")
print(f"\n기술 통계:")
print(df_house.describe())

# 타겟 분포
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(df_house['MedHouseVal'], bins=50, edgecolor='black')
plt.xlabel('Median House Value')
plt.ylabel('Count')
plt.title('Target Distribution')

plt.subplot(1, 2, 2)
plt.hist(np.log1p(df_house['MedHouseVal']), bins=50, edgecolor='black')
plt.xlabel('Log(Median House Value)')
plt.ylabel('Count')
plt.title('Log-Transformed Target')

plt.tight_layout()
plt.show()
```

### 3.2 탐색적 데이터 분석

```python
# 상관관계 히트맵
plt.figure(figsize=(12, 10))
sns.heatmap(df_house.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# 타겟과의 상관관계
print("\n타겟과의 상관관계:")
correlations = df_house.corr()['MedHouseVal'].drop('MedHouseVal').sort_values(ascending=False)
print(correlations)

# 주요 특성과 타겟 관계
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, col in zip(axes.flatten(), housing.feature_names):
    ax.scatter(df_house[col], df_house['MedHouseVal'], alpha=0.1)
    ax.set_xlabel(col)
    ax.set_ylabel('MedHouseVal')
    ax.set_title(f'Corr: {df_house[col].corr(df_house["MedHouseVal"]):.3f}')

plt.tight_layout()
plt.show()
```

### 3.3 데이터 전처리

```python
# 특성과 타겟 분리
X = df_house.drop('MedHouseVal', axis=1)
y = df_house['MedHouseVal']

# 특성 엔지니어링
X_eng = X.copy()

# 방당 인원
X_eng['RoomsPerPerson'] = X_eng['AveRooms'] / X_eng['AveOccup']

# 침실 비율
X_eng['BedroomRatio'] = X_eng['AveBedrms'] / X_eng['AveRooms']

# 인구 밀도 (대략적)
X_eng['PopDensity'] = X_eng['Population'] / X_eng['AveOccup']

# 무한대/NaN 처리
X_eng = X_eng.replace([np.inf, -np.inf], np.nan)
X_eng = X_eng.fillna(X_eng.median())

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.2, random_state=42
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"학습 데이터: {X_train.shape}")
print(f"테스트 데이터: {X_test.shape}")
```

### 3.4 모델링

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# 모델 정의
reg_models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'LightGBM': LGBMRegressor(random_state=42, verbose=-1)
}

# 모델 비교
print("=== 회귀 모델 비교 ===")
reg_results = []

for name, model in reg_models.items():
    # 선형 모델은 스케일링된 데이터
    if name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test

    # 교차 검증
    cv_scores = cross_val_score(model, X_tr, y_train, cv=5, scoring='r2')

    # 학습 및 예측
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    reg_results.append({
        'Model': name,
        'CV R2 Mean': cv_scores.mean(),
        'CV R2 Std': cv_scores.std(),
        'Test RMSE': rmse,
        'Test R2': r2
    })

    print(f"{name}: CV R2={cv_scores.mean():.4f}(+/-{cv_scores.std():.4f}), RMSE={rmse:.4f}, R2={r2:.4f}")

reg_results_df = pd.DataFrame(reg_results)
print(f"\n최고 테스트 R2: {reg_results_df.loc[reg_results_df['Test R2'].idxmax(), 'Model']}")
```

### 3.5 하이퍼파라미터 튜닝

```python
# LightGBM 튜닝
lgbm_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [31, 63, 127]
}

lgbm = LGBMRegressor(random_state=42, verbose=-1)
grid_lgbm = GridSearchCV(lgbm, lgbm_params, cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_lgbm.fit(X_train, y_train)

print("\n=== LightGBM 튜닝 결과 ===")
print(f"최적 파라미터: {grid_lgbm.best_params_}")
print(f"최적 CV R2: {grid_lgbm.best_score_:.4f}")

y_pred_best = grid_lgbm.predict(X_test)
print(f"테스트 RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
print(f"테스트 R2: {r2_score(y_test, y_pred_best):.4f}")
```

### 3.6 최종 평가

```python
# 최종 모델
best_reg = grid_lgbm.best_estimator_
y_pred_final = best_reg.predict(X_test)

# 평가 지표
print("=== 최종 회귀 평가 ===")
print(f"MAE: {mean_absolute_error(y_test, y_pred_final):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_final)):.4f}")
print(f"R2: {r2_score(y_test, y_pred_final):.4f}")

# 예측 vs 실제
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 산점도
axes[0].scatter(y_test, y_pred_final, alpha=0.3)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title('Actual vs Predicted')

# 잔차 분포
residuals = y_test - y_pred_final
axes[1].hist(residuals, bins=50, edgecolor='black')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Count')
axes[1].set_title(f'Residual Distribution\nMean: {residuals.mean():.4f}')

# 잔차 vs 예측값
axes[2].scatter(y_pred_final, residuals, alpha=0.3)
axes[2].axhline(y=0, color='r', linestyle='--')
axes[2].set_xlabel('Predicted')
axes[2].set_ylabel('Residual')
axes[2].set_title('Residuals vs Predicted')

plt.tight_layout()
plt.show()

# 특성 중요도
importances = best_reg.feature_importances_
feature_names = X_eng.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance (LightGBM)')
plt.tight_layout()
plt.show()
```

---

## 4. Kaggle 경진대회 전략

### 4.1 기본 전략

```python
"""
Kaggle 경진대회 전략:

1. 빠른 시작
   - 제공된 baseline 코드 실행
   - 간단한 모델로 첫 제출
   - 리더보드 위치 확인

2. EDA 집중
   - 데이터 이해가 핵심
   - 결측치, 이상치, 분포 파악
   - 타겟과의 관계 분석

3. 특성 엔지니어링
   - 도메인 지식 활용
   - 교차 특성 생성
   - 그룹별 통계량

4. 다양한 모델 시도
   - 선형 모델 → 트리 기반 → 앙상블
   - 하이퍼파라미터 튜닝

5. 앙상블
   - 다른 모델 예측 결합
   - 블렌딩, 스태킹

6. 검증 전략
   - 로컬 CV와 리더보드 점수 일치 확인
   - 과적합 방지
"""
```

### 4.2 교차 검증 전략

```python
from sklearn.model_selection import KFold, StratifiedKFold

def cross_validate_model(model, X, y, n_splits=5, stratified=False, return_preds=False):
    """
    교차 검증 및 OOF 예측 생성
    """
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_func = kf.split(X, y)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_func = kf.split(X)

    scores = []
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(split_func):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        model.fit(X_train_fold, y_train_fold)
        val_pred = model.predict(X_val_fold)

        oof_preds[val_idx] = val_pred
        score = r2_score(y_val_fold, val_pred)
        scores.append(score)

        print(f"Fold {fold+1}: {score:.4f}")

    print(f"Mean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

    if return_preds:
        return oof_preds
    return np.mean(scores)

# 사용 예시
# oof_preds = cross_validate_model(model, X_train, y_train, n_splits=5, return_preds=True)
```

### 4.3 앙상블 기법

```python
def simple_blend(predictions_list, weights=None):
    """간단한 블렌딩"""
    if weights is None:
        weights = [1/len(predictions_list)] * len(predictions_list)

    blended = np.zeros(len(predictions_list[0]))
    for pred, weight in zip(predictions_list, weights):
        blended += weight * pred

    return blended


def stacking_ensemble(models, X_train, y_train, X_test, n_folds=5):
    """스태킹 앙상블"""
    n_models = len(models)
    n_train = len(X_train)
    n_test = len(X_test)

    # OOF 예측
    oof_train = np.zeros((n_train, n_models))
    oof_test = np.zeros((n_test, n_models))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for i, model in enumerate(models):
        print(f"Training model {i+1}/{n_models}")
        oof_test_fold = np.zeros((n_test, n_folds))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            X_val = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_tr = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]

            model.fit(X_tr, y_tr)

            oof_train[val_idx, i] = model.predict(X_val)
            oof_test_fold[:, fold] = model.predict(X_test)

        oof_test[:, i] = oof_test_fold.mean(axis=1)

    return oof_train, oof_test

# 사용 예시
# models = [RandomForestRegressor(), XGBRegressor(), LGBMRegressor()]
# oof_train, oof_test = stacking_ensemble(models, X_train, y_train, X_test)
# meta_model = Ridge()
# meta_model.fit(oof_train, y_train)
# final_preds = meta_model.predict(oof_test)
```

---

## 5. 실전 팁 모음

### 5.1 빠른 실험 템플릿

```python
def quick_experiment(X_train, y_train, X_test, y_test, task='classification'):
    """빠른 모델 비교"""
    if task == 'classification':
        models = {
            'LR': LogisticRegression(max_iter=1000),
            'RF': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGB': XGBClassifier(eval_metric='logloss', random_state=42),
            'LGBM': LGBMClassifier(random_state=42, verbose=-1)
        }
        scoring = 'accuracy'
    else:
        models = {
            'Ridge': Ridge(),
            'RF': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGB': XGBRegressor(random_state=42),
            'LGBM': LGBMRegressor(random_state=42, verbose=-1)
        }
        scoring = 'r2'

    results = {}
    for name, model in models.items():
        cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring).mean()
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        results[name] = {'CV': cv_score, 'Test': test_score}
        print(f"{name}: CV={cv_score:.4f}, Test={test_score:.4f}")

    return results
```

### 5.2 메모리 최적화

```python
def reduce_memory_usage(df):
    """DataFrame 메모리 사용량 감소"""
    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)

            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'메모리: {start_mem:.2f} MB → {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% 감소)')

    return df
```

### 5.3 오류 분석

```python
def analyze_errors(y_true, y_pred, X, feature_names, top_n=10):
    """오류 분석"""
    errors = np.abs(y_true - y_pred)

    # 가장 큰 오류
    top_errors_idx = np.argsort(errors)[-top_n:]

    print(f"=== 상위 {top_n}개 오류 분석 ===")
    for idx in top_errors_idx[::-1]:
        print(f"\n인덱스 {idx}:")
        print(f"  실제: {y_true.iloc[idx] if hasattr(y_true, 'iloc') else y_true[idx]:.4f}")
        print(f"  예측: {y_pred[idx]:.4f}")
        print(f"  오차: {errors[idx]:.4f}")

    # 오류와 특성 상관관계
    X_arr = X.values if hasattr(X, 'values') else X
    error_corr = []
    for i, name in enumerate(feature_names):
        corr = np.corrcoef(errors, X_arr[:, i])[0, 1]
        error_corr.append((name, corr))

    error_corr.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n=== 오류와 특성 상관관계 ===")
    for name, corr in error_corr[:5]:
        print(f"  {name}: {corr:.4f}")

# 사용 예시
# analyze_errors(y_test, y_pred, X_test, feature_names)
```

---

## 연습 문제

### 문제 1: 전체 파이프라인
Iris 데이터로 전체 ML 파이프라인을 구축하세요.

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split

# 풀이
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"CV 점수: {cv_scores.mean():.4f}")

pipeline.fit(X_train, y_train)
print(f"테스트 점수: {pipeline.score(X_test, y_test):.4f}")
```

### 문제 2: 특성 엔지니어링
주어진 데이터에 새로운 특성을 추가하세요.

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50],
    'C': [100, 200, 300, 400, 500]
})

# 풀이: 비율 특성, 합계 특성 추가
df['A_B_ratio'] = df['A'] / df['B']
df['AB_sum'] = df['A'] + df['B']
df['log_C'] = np.log1p(df['C'])
df['A_squared'] = df['A'] ** 2

print(df)
```

### 문제 3: 앙상블
여러 모델을 블렌딩하세요.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 풀이
models = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100, random_state=42)
]

predictions = []
for model in models:
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    predictions.append(pred)

# 평균 블렌딩
blended = np.mean(predictions, axis=0)
blended_labels = (blended > 0.5).astype(int)

print(f"블렌딩 정확도: {accuracy_score(y_test, blended_labels):.4f}")
```

---

## 요약

### 분류 vs 회귀 체크리스트

| 단계 | 분류 | 회귀 |
|------|------|------|
| 평가 지표 | Accuracy, F1, AUC | RMSE, MAE, R2 |
| 타겟 처리 | 인코딩 | 이상치 확인, 로그 변환 |
| 불균형 | SMOTE, 클래스 가중치 | 해당 없음 |
| 오차 분석 | 혼동 행렬 | 잔차 분석 |

### 모델 선택 가이드

| 상황 | 권장 모델 |
|------|-----------|
| 빠른 기준선 | 로지스틱/선형 회귀 |
| 일반적인 성능 | Random Forest |
| 최고 성능 | XGBoost, LightGBM |
| 해석 필요 | 결정 트리, 선형 모델 |
| 대용량 데이터 | LightGBM |

### Kaggle 필수 팁

1. 항상 로컬 CV와 리더보드 점수 비교
2. 과적합 주의 - 퍼블릭 리더보드에 맞추지 말 것
3. 앙상블은 다양한 모델로
4. 특성 엔지니어링이 핵심
5. 커널/노트북 참고하되 이해하고 적용
