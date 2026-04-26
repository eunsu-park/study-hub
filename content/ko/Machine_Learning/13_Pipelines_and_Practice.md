# 파이프라인과 실무 (Pipeline & Practice)

**이전**: [차원 축소](./12_Dimensionality_Reduction.md) | **다음**: [실전 프로젝트](./14_Practical_Projects.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. sklearn 파이프라인(Pipeline)이 데이터 누수(data leakage)를 방지하고 ML 워크플로우를 단순화하는 이유를 설명할 수 있습니다
2. 전처리, 특성 추출, 모델 학습을 단일 객체로 연결하는 파이프라인(pipeline)을 구축할 수 있습니다
3. ColumnTransformer를 적용하여 수치형 및 범주형 특성에 서로 다른 전처리 단계를 적용할 수 있습니다
4. 올바른 하이퍼파라미터 튜닝을 위해 파이프라인(Pipeline)을 교차 검증(cross-validation) 및 GridSearchCV와 통합할 수 있습니다
5. BaseEstimator와 TransformerMixin을 사용하여 커스텀 변환기(custom transformer)를 구현할 수 있습니다
6. joblib을 사용하여 학습된 파이프라인을 버전 메타데이터와 함께 저장하고 불러올 수 있습니다
7. 분류 및 회귀 문제를 위한 재사용 가능한 파이프라인 템플릿(pipeline template)을 설계할 수 있습니다

---

모델 학습 방법을 아는 것은 절반에 불과합니다 -- 프로덕션 환경에서는 전처리부터 예측까지의 전체 체인이 재현 가능하고, 누수가 없으며, 이식 가능해야 합니다. Sklearn의 Pipeline과 ColumnTransformer는 모든 변환 단계를 단일 객체로 캡슐화하여 교차 검증(cross-validated), 직렬화(serialized), 배포(deployed)가 하나의 단위로 이루어질 수 있도록 해결합니다. 이 레슨은 노트북 실험과 프로덕션 수준의 ML 코드 사이의 간극을 연결합니다.

---

## 이론과 원리

scikit-learn 파이프라인은 편의 래퍼 이상입니다 — 전처리 변환의 대수와 모든 단계가 따라야 할 계약(contract)을 형식화합니다. 이 계약을 이해하는 것이 가장 음흉한 ML 버그를 막습니다: 데이터 누수(data leakage), 즉 테스트셋의 정보가 학습 절차를 조용히 오염시키는 것.

### A. 함수 합성으로서의 Estimator/Transformer 인터페이스

모든 scikit-learn 단계는 두 가지 중 하나입니다:
- **변환기(Transformer)**: `.fit(X, y)`와 `.transform(X)`를 구현. 예: `StandardScaler`, `PCA`, `OneHotEncoder`.
- **추정기(Estimator)**: `.fit(X, y)`와 `.predict(X)`(또는 `.predict_proba(X)`)를 구현. 예: `LogisticRegression`, `RandomForestClassifier`.

`Pipeline([(name_1, T_1), ..., (name_n, T_n), (name_final, E)])`는 이들을 추정기처럼 동작하는 단일 객체로 합성합니다:

```
Pipeline.fit(X, y):
    X_1 = T_1.fit_transform(X, y)
    X_2 = T_2.fit_transform(X_1, y)
    ...
    E.fit(X_n, y)

Pipeline.predict(X):
    X_1 = T_1.transform(X)              ← 주의: .transform이지 .fit_transform이 아님
    X_2 = T_2.transform(X_1)
    ...
    return E.predict(X_n)
```

비대칭성이 핵심입니다: `fit` 동안, 각 변환기가 `fit_transform`을 통해 *학습* 데이터에서 매개변수를 학습합니다. `predict` 동안, 각 변환기가 `transform`을 통해 *이미 학습된* 매개변수를 적용합니다. 변환 매개변수는 모델의 일부이지 데이터의 일부가 아닙니다.

### B. 파이프라인이 누수를 막는 이유

고전적인 누수 버그:

```python
# 잘못
scaler = StandardScaler().fit(X)        # 전체 데이터셋에서 평균/표준편차 학습
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
cross_val_score(model, X_train, y_train, cv=5)
```

스케일러가 평균/표준편차를 계산할 때 테스트 데이터를 봤으므로 테스트 정보가 "학습" 파이프라인으로 누설. CV 점수가 위쪽으로 편향되고; 배포 수치는 더 나쁠 것.

파이프라인 버전은 자동으로 올바릅니다:

```python
# 올바름
pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression())])
cross_val_score(pipe, X, y, cv=5)
```

각 CV 폴드 안에서 scikit-learn은 `pipe.fit(X_train_fold, y_train_fold)` 다음 `pipe.predict(X_val_fold)`를 호출. 스케일러는 학습 폴드에서만 `fit`되고; 검증 폴드의 `transform`은 학습 폴드의 평균/표준편차를 사용. 누수 없음. 이것이 파이프라인을 사용하는 *가장 중요한 단일 이유*입니다.

같은 성질이 그리드 탐색으로 확장됩니다: `GridSearchCV(pipe, params)`는 후보 하이퍼파라미터를 적용하여 각 폴드에서 전체 파이프라인(변환기 포함)을 처음부터 재학습. 하이퍼파라미터 `model__C`(이중 밑줄에 주의)는 `model` 단계의 `C`를 설정.

### C. ColumnTransformer: 이질적 전처리

실제 데이터셋은 타입을 섞습니다: 수치 특성은 스케일링이 필요, 범주형 특성은 인코딩 필요, 텍스트 특성은 벡터화 필요. 모든 것에 단일 변환을 적용하면 실패. `ColumnTransformer`는 다른 열을 다른 변환기로 라우팅하고 결과를 연결할 수 있게 해줍니다:

```
ColumnTransformer(
    [('num', StandardScaler(),    ['age', 'income']),
     ('cat', OneHotEncoder(),     ['city', 'occupation']),
     ('txt', TfidfVectorizer(),   'review_text')],
    remainder='drop' | 'passthrough'
)
```

출력은 변환기별 출력의 수평 스택. 전체 `ColumnTransformer` 자체가 변환기이므로 파이프라인 안에 끼워집니다:

```
Pipeline([('preproc', ColumnTransformer(...)), ('model', RandomForestClassifier())])
```

이 단일 객체가 이제 데이터-에서-예측 스택 전체: `pipe.fit(df, y)`가 스케일러 평균, OHE 범주, TF-IDF 어휘를 학습 *그리고* 랜덤 포레스트를 학습. `pipe.predict(new_df)`가 누수 없이 모든 것을 순서대로 적용.

### D. 그 뒤의 합성 대수

수학적으로 파이프라인은 함수 합성 `f_n ∘ ... ∘ f_2 ∘ f_1`이며, 각 `f_i`는 단계 `i-1`의 학습 출력에 적합된 매개변수 `θ_i`를 가진 학습된 함수 `f_i(x; θ_i)`. 단계 `i`의 `fit_transform` 메서드는

```
f_i_fit_transform(X, y):
    θ_i ← argmin_θ  L_i(X, y, θ)              ← 단계 i의 매개변수 학습
    return f_i(X; θ_i)                         ← 방금 학습한 함수 적용
```

대부분의 전처리기에서 `L_i`는 암묵적입니다(예: StandardScaler는 `mean = 0, std = 1`을 만족하도록 `θ = (μ, σ)`를 고름). PCA의 경우 `L_i`는 Lesson 12의 분산 최대화 목적. 요점은 그것들 모두 같은 `fit` / `transform` 분할을 존중한다는 것.

이 대수가 파이프라인을 교차검증, 그리드 탐색, 모델 직렬화(`joblib.dump(pipe, 'model.pkl')`), 배포와 합성할 수 있게 합니다 — 모두가 표준 인터페이스 뒤에 내부 구조가 숨겨진 함수 `pipe.predict(·)`에 대한 연산일 뿐.

### E. 사용자 정의 변환기와 계약

내장 변환기가 맞지 않을 때, `BaseEstimator`와 `TransformerMixin`을 서브클래싱하여 자신의 것을 작성:

```python
from sklearn.base import BaseEstimator, TransformerMixin

class MyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, param=1.0):
        self.param = param

    def fit(self, X, y=None):
        # 학습 데이터에 의존하는 어떤 것이든 학습
        self.learned_ = X.mean()
        return self

    def transform(self, X):
        # 학습된 매개변수 적용
        return X - self.learned_
```

계약은: 학습된 매개변수를 `_`로 끝나는 속성으로 저장, `fit`에서 `self` 반환, `transform`에서 변환된 데이터 반환. 그것을 존중하는 한, 변환기가 어떤 내장처럼 파이프라인, 그리드 탐색, 직렬화에 끼워집니다.

### From Theory to the Code Below

- 섹션 1.2의 `Pipeline([...])` 생성자는 (A)의 함수 합성; 단계별 fit-vs-transform 분할은 (B)의 누수 방지.
- 섹션 2의 `ColumnTransformer`는 (C)의 이질적 라우팅 객체; `remainder` 인자가 라우팅되지 않은 열에 무엇을 할지 결정.
- 섹션 3의 `GridSearchCV(pipe, {'model__C': [...]})`는 이중 밑줄 매개변수 구문 — `step__hyperparameter` — 을 사용해 그리드 탐색이 파이프라인 안으로 도달.
- 섹션 4의 `joblib.dump(pipe, ...)`와 `joblib.load(...)`는 파이프라인의 매개변수가 (D)의 `θ_i` 속성에 모두 저장되어 있기 *때문에* 작동.
- 섹션 5의 사용자 정의 `Transformer` 예제는 (E)에 명시된 계약을 따름.

---

## 1. Pipeline 기초

### 1.1 Pipeline의 필요성

```python
"""
Pipeline 없이 코드 작성 시 문제점:

1. 데이터 누수 (Data Leakage):
   - 테스트 데이터 정보가 학습에 반영
   - 예: 전체 데이터로 스케일링 후 분할

2. 코드 복잡성:
   - 여러 단계를 수동으로 관리
   - 실수 가능성 높음

3. 재현성 문제:
   - 순서 실수
   - 파라미터 불일치

Pipeline 장점:
1. 코드 간소화
2. 데이터 누수 방지
3. 교차 검증과 완벽 통합
4. 하이퍼파라미터 튜닝 용이
5. 모델 저장/배포 편리
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_iris
```

### 1.2 기본 Pipeline 생성

```python
# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Pipeline 생성 (명시적 이름)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
])

# 학습 및 예측
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
score = pipeline.score(X_test, y_test)

print(f"Pipeline 정확도: {score:.4f}")

# make_pipeline (자동 이름)
pipeline_auto = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression()
)

pipeline_auto.fit(X_train, y_train)
print(f"make_pipeline 정확도: {pipeline_auto.score(X_test, y_test):.4f}")
```

### 1.3 Pipeline 단계 접근

```python
# 단계 이름 확인
print("Pipeline 단계:")
for name, step in pipeline.named_steps.items():
    print(f"  {name}: {type(step).__name__}")

# 특정 단계 접근
print(f"\nPCA 설명된 분산: {pipeline.named_steps['pca'].explained_variance_ratio_}")
print(f"로지스틱 회귀 계수 형상: {pipeline.named_steps['classifier'].coef_.shape}")

# 중간 단계 결과 얻기
X_scaled = pipeline.named_steps['scaler'].transform(X_test)
X_pca = pipeline.named_steps['pca'].transform(X_scaled)
print(f"\n스케일링 후 형상: {X_scaled.shape}")
print(f"PCA 후 형상: {X_pca.shape}")
```

---

## 2. ColumnTransformer

### 2.1 다양한 타입의 특성 처리

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

"""
ColumnTransformer:
- 서로 다른 타입의 특성에 다른 전처리 적용
- 수치형: 스케일링
- 범주형: 인코딩
"""

# 샘플 데이터
data = {
    'age': [25, 32, 47, 51, 62],
    'income': [50000, 60000, 80000, 120000, 95000],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master'],
    'purchased': [0, 1, 1, 1, 0]
}
df = pd.DataFrame(data)

X = df.drop('purchased', axis=1)
y = df['purchased']

print("데이터 타입:")
print(X.dtypes)
```

### 2.2 ColumnTransformer 생성

```python
# 특성 분류
numeric_features = ['age', 'income']
categorical_features = ['gender', 'education']

# ColumnTransformer 정의
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'  # 나머지 특성 처리: 'drop', 'passthrough'
)

# 변환
X_transformed = preprocessor.fit_transform(X)

print(f"원본 형상: {X.shape}")
print(f"변환 후 형상: {X_transformed.shape}")

# 변환된 특성 이름
feature_names = (
    numeric_features +
    list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
)
print(f"특성 이름: {feature_names}")
```

### 2.3 Pipeline + ColumnTransformer

```python
from sklearn.ensemble import RandomForestClassifier

# 전체 파이프라인
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 학습 (작은 데이터이므로 전체 사용)
full_pipeline.fit(X, y)

# 예측
new_data = pd.DataFrame({
    'age': [30],
    'income': [70000],
    'gender': ['F'],
    'education': ['Master']
})
prediction = full_pipeline.predict(new_data)
print(f"예측: {prediction[0]}")
```

---

## 3. 복잡한 전처리 파이프라인

### 3.1 결측치 처리 포함

```python
from sklearn.impute import SimpleImputer

# 결측치가 있는 데이터
data_missing = {
    'age': [25, np.nan, 47, 51, 62],
    'income': [50000, 60000, np.nan, 120000, 95000],
    'gender': ['M', 'F', 'M', None, 'M'],
    'education': ['Bachelor', 'Master', 'PhD', 'Bachelor', None],
    'purchased': [0, 1, 1, 1, 0]
}
df_missing = pd.DataFrame(data_missing)
X_missing = df_missing.drop('purchased', axis=1)
y_missing = df_missing['purchased']

# 수치형 파이프라인
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 범주형 파이프라인
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
])

# ColumnTransformer
preprocessor_full = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 전체 파이프라인
complete_pipeline = Pipeline([
    ('preprocessor', preprocessor_full),
    ('classifier', RandomForestClassifier(random_state=42))
])

complete_pipeline.fit(X_missing, y_missing)
print("결측치 포함 파이프라인 학습 완료")
```

### 3.2 특성 선택 포함

```python
from sklearn.feature_selection import SelectKBest, f_classif

# 특성 선택 포함 파이프라인
pipeline_with_selection = Pipeline([
    ('preprocessor', preprocessor_full),
    ('feature_selection', SelectKBest(score_func=f_classif, k='all')),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline_with_selection.fit(X_missing, y_missing)
print("특성 선택 포함 파이프라인 학습 완료")
```

---

## 4. Pipeline과 교차 검증

### 4.1 올바른 교차 검증

```python
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import load_breast_cancer

# 데이터 로드
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 파이프라인 정의
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 교차 검증 (올바른 방법)
# 각 폴드에서 스케일러가 학습 데이터만으로 fit됨
scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

print("교차 검증 결과:")
print(f"  각 폴드: {scores}")
print(f"  평균: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### 4.2 Pipeline 하이퍼파라미터 튜닝

```python
# 파라미터 이름: step__parameter
param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']
}

# Grid Search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X, y)

print("Grid Search 결과:")
print(f"  최적 파라미터: {grid_search.best_params_}")
print(f"  최적 점수: {grid_search.best_score_:.4f}")
```

### 4.3 복잡한 파라미터 그리드

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# 여러 모델 비교 파이프라인
pipeline_multi = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())  # placeholder
])

# 모델별 다른 파라미터
param_grid_multi = [
    {
        'classifier': [LogisticRegression(max_iter=1000)],
        'classifier__C': [0.1, 1, 10]
    },
    {
        'classifier': [RandomForestClassifier(random_state=42)],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 5, 10]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1],
        'classifier__kernel': ['rbf', 'linear']
    }
]

grid_search_multi = GridSearchCV(
    pipeline_multi,
    param_grid_multi,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search_multi.fit(X, y)

print("여러 모델 비교 결과:")
print(f"  최적 모델: {type(grid_search_multi.best_params_['classifier']).__name__}")
print(f"  최적 파라미터: {grid_search_multi.best_params_}")
print(f"  최적 점수: {grid_search_multi.best_score_:.4f}")
```

---

## 5. 모델 저장과 로드

### 5.1 joblib 사용

```python
import joblib

# 최적 모델 학습
best_pipeline = grid_search.best_estimator_

# 모델 저장
joblib.dump(best_pipeline, 'best_model.joblib')
print("모델 저장 완료: best_model.joblib")

# 모델 로드
loaded_model = joblib.load('best_model.joblib')

# 테스트
X_test_sample = X[:5]
predictions = loaded_model.predict(X_test_sample)
print(f"로드된 모델 예측: {predictions}")
```

### 5.2 pickle 사용

```python
import pickle

# pickle 저장
with open('model.pkl', 'wb') as f:
    pickle.dump(best_pipeline, f)

# pickle 로드
with open('model.pkl', 'rb') as f:
    loaded_model_pkl = pickle.load(f)

print("pickle 모델 예측:", loaded_model_pkl.predict(X[:3]))
```

### 5.3 버전 관리

```python
import sklearn
from datetime import datetime

# 메타데이터와 함께 저장
model_metadata = {
    'model': best_pipeline,
    'sklearn_version': sklearn.__version__,
    'training_date': datetime.now().isoformat(),
    'feature_names': list(cancer.feature_names),
    'target_names': list(cancer.target_names),
    'cv_score': grid_search.best_score_
}

joblib.dump(model_metadata, 'model_with_metadata.joblib')

# 로드 및 검증
loaded_metadata = joblib.load('model_with_metadata.joblib')
print(f"학습 날짜: {loaded_metadata['training_date']}")
print(f"sklearn 버전: {loaded_metadata['sklearn_version']}")
print(f"CV 점수: {loaded_metadata['cv_score']:.4f}")
```

---

## 6. FunctionTransformer

### 6.1 커스텀 변환 함수

```python
from sklearn.preprocessing import FunctionTransformer

# 커스텀 변환 함수
def log_transform(X):
    return np.log1p(X)  # log(1 + x)

def add_polynomial_features(X):
    return np.c_[X, X ** 2, X ** 3]

# FunctionTransformer 생성
log_transformer = FunctionTransformer(log_transform, validate=True)
poly_transformer = FunctionTransformer(add_polynomial_features, validate=True)

# 파이프라인에서 사용
pipeline_custom = Pipeline([
    ('log', log_transformer),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 테스트
X_positive = np.abs(X) + 1  # 로그를 위해 양수로 변환
scores = cross_val_score(pipeline_custom, X_positive, y, cv=5)
print(f"커스텀 변환 파이프라인 CV 점수: {scores.mean():.4f}")
```

### 6.2 특성 추가 함수

```python
# 도메인 특정 특성 추가
def create_ratio_features(X):
    """비율 특성 생성"""
    X = np.array(X)
    if X.shape[1] >= 2:
        ratio = (X[:, 0] / (X[:, 1] + 1e-10)).reshape(-1, 1)
        return np.c_[X, ratio]
    return X

ratio_transformer = FunctionTransformer(create_ratio_features)

# 파이프라인
pipeline_ratio = Pipeline([
    ('ratio_features', ratio_transformer),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

scores = cross_val_score(pipeline_ratio, X, y, cv=5)
print(f"비율 특성 추가 CV 점수: {scores.mean():.4f}")
```

---

## 7. 커스텀 Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierRemover(BaseEstimator, TransformerMixin):
    """이상치 제거 트랜스포머"""

    def __init__(self, threshold=3):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y=None):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        X = np.array(X)
        z_scores = np.abs((X - self.mean_) / (self.std_ + 1e-10))
        # 이상치를 경계값으로 대체
        X_clipped = np.where(z_scores > self.threshold,
                             self.mean_ + self.threshold * self.std_ * np.sign(X - self.mean_),
                             X)
        return X_clipped


class FeatureSelector(BaseEstimator, TransformerMixin):
    """특성 선택 트랜스포머"""

    def __init__(self, feature_indices=None):
        self.feature_indices = feature_indices

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        if self.feature_indices is not None:
            return X[:, self.feature_indices]
        return X


# 커스텀 트랜스포머 사용
custom_pipeline = Pipeline([
    ('outlier', OutlierRemover(threshold=3)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000))
])

scores = cross_val_score(custom_pipeline, X, y, cv=5)
print(f"커스텀 트랜스포머 CV 점수: {scores.mean():.4f}")
```

---

## 8. 실전 전처리 템플릿

### 8.1 분류 문제 템플릿

```python
from sklearn.compose import make_column_selector

def create_classification_pipeline(model, numeric_features=None, categorical_features=None):
    """분류 문제용 파이프라인 생성"""

    # 수치형 특성 파이프라인
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 범주형 특성 파이프라인
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # ColumnTransformer
    if numeric_features is None and categorical_features is None:
        # 자동 감지
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
                ('cat', categorical_transformer, make_column_selector(dtype_include=object))
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features or []),
                ('cat', categorical_transformer, categorical_features or [])
            ]
        )

    # 전체 파이프라인
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    return pipeline


# 사용 예시
from sklearn.ensemble import GradientBoostingClassifier

pipeline = create_classification_pipeline(
    GradientBoostingClassifier(random_state=42),
    numeric_features=['age', 'income'],
    categorical_features=['gender', 'education']
)
```

### 8.2 회귀 문제 템플릿

```python
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

def create_regression_pipeline(model, numeric_features=None, categorical_features=None):
    """회귀 문제용 파이프라인 생성"""

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    if numeric_features is None and categorical_features is None:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
                ('cat', categorical_transformer, make_column_selector(dtype_include=object))
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features or []),
                ('cat', categorical_transformer, categorical_features or [])
            ]
        )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline
```

---

## 9. 모델 배포 고려사항

### 9.1 예측 함수 래핑

```python
class ModelWrapper:
    """배포용 모델 래퍼"""

    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.feature_names = None

    def set_feature_names(self, names):
        self.feature_names = names

    def predict(self, input_data):
        """딕셔너리 또는 DataFrame 입력 처리"""
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        if self.feature_names:
            input_data = input_data[self.feature_names]

        return self.model.predict(input_data)

    def predict_proba(self, input_data):
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])

        if self.feature_names:
            input_data = input_data[self.feature_names]

        return self.model.predict_proba(input_data)


# 사용 예시
# wrapper = ModelWrapper('best_model.joblib')
# wrapper.set_feature_names(['age', 'income', 'gender', 'education'])
# prediction = wrapper.predict({'age': 30, 'income': 70000, 'gender': 'M', 'education': 'Bachelor'})
```

### 9.2 입력 검증

```python
def validate_input(data, expected_columns, expected_dtypes=None):
    """입력 데이터 검증"""
    errors = []

    # 필수 컬럼 확인
    missing_cols = set(expected_columns) - set(data.columns)
    if missing_cols:
        errors.append(f"누락된 컬럼: {missing_cols}")

    # 데이터 타입 확인
    if expected_dtypes:
        for col, dtype in expected_dtypes.items():
            if col in data.columns and not np.issubdtype(data[col].dtype, dtype):
                errors.append(f"잘못된 타입 - {col}: {data[col].dtype} (기대: {dtype})")

    # 결측치 확인
    null_counts = data[expected_columns].isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        print(f"경고: 결측치 발견 - {dict(null_cols)}")

    if errors:
        raise ValueError("\n".join(errors))

    return True
```

---

## 10. 실전 체크리스트

```python
"""
ML 프로젝트 체크리스트:

1. 데이터 준비
   [ ] 데이터 로드 및 기본 탐색
   [ ] 타겟 변수 정의
   [ ] 학습/검증/테스트 분할

2. 탐색적 데이터 분석 (EDA)
   [ ] 결측치 확인
   [ ] 이상치 확인
   [ ] 특성 분포 확인
   [ ] 타겟과의 상관관계

3. 전처리 파이프라인
   [ ] 수치형 특성 처리 (스케일링, 결측치)
   [ ] 범주형 특성 처리 (인코딩, 결측치)
   [ ] 특성 선택/생성

4. 모델링
   [ ] 기준선 모델 설정
   [ ] 여러 모델 비교
   [ ] 하이퍼파라미터 튜닝
   [ ] 교차 검증

5. 평가
   [ ] 적절한 평가 지표 선택
   [ ] 과적합/과소적합 확인
   [ ] 오차 분석

6. 배포
   [ ] 모델 저장
   [ ] 입력 검증
   [ ] 예측 함수 래핑
   [ ] 모니터링 계획
"""
```

---

## 연습 문제

### 문제 1: 기본 Pipeline
Iris 데이터에 스케일링 + PCA + 로지스틱 회귀 파이프라인을 만드세요.

```python
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

# 풀이
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', LogisticRegression())
])

scores = cross_val_score(pipeline, X, y, cv=5)
print(f"CV 점수: {scores.mean():.4f}")
```

### 문제 2: ColumnTransformer
수치형과 범주형 특성을 다르게 처리하는 파이프라인을 만드세요.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# 샘플 데이터
data = pd.DataFrame({
    'age': [25, 30, 35, 40],
    'income': [50000, 60000, 70000, 80000],
    'city': ['A', 'B', 'A', 'C']
})

# 풀이
numeric_features = ['age', 'income']
categorical_features = ['city']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(), categorical_features)
])

X_transformed = preprocessor.fit_transform(data)
print(f"변환 후 형상: {X_transformed.shape}")
```

### 문제 3: 모델 저장 및 로드
학습된 파이프라인을 저장하고 로드하세요.

```python
import joblib

# 학습
pipeline.fit(X, y)

# 저장
joblib.dump(pipeline, 'iris_pipeline.joblib')

# 로드
loaded_pipeline = joblib.load('iris_pipeline.joblib')

# 테스트
print(f"로드된 모델 정확도: {loaded_pipeline.score(X, y):.4f}")
```

---

## 요약

| 구성 요소 | 용도 | 예시 |
|-----------|------|------|
| Pipeline | 단계 순차 연결 | 스케일링 → PCA → 모델 |
| ColumnTransformer | 특성별 다른 처리 | 수치형/범주형 분리 |
| FunctionTransformer | 커스텀 함수 | 로그 변환 |
| make_pipeline | 자동 이름 지정 | 간단한 파이프라인 |

### Pipeline 하이퍼파라미터 명명 규칙

```
step_name__parameter_name

예시:
- classifier__C: 분류기의 C 파라미터
- preprocessor__num__scaler__with_mean: 중첩된 파라미터
```

### 모델 저장 비교

| 방법 | 장점 | 단점 |
|------|------|------|
| joblib | 대용량 NumPy 효율적 | sklearn 전용 |
| pickle | 표준 라이브러리 | 대용량 느림 |
| ONNX | 프레임워크 독립적 | 변환 필요 |

### 실무 팁

1. 항상 Pipeline 사용하여 데이터 누수 방지
2. ColumnTransformer로 전처리 명확하게 분리
3. 모델 저장 시 메타데이터 포함
4. 입력 검증 함수 작성
5. 버전 관리 철저히
