# 차원 축소 (Dimensionality Reduction)

**이전**: [클러스터링](./11_Clustering.md) | **다음**: [파이프라인과 실무](./13_Pipelines_and_Practice.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 차원의 저주(curse of dimensionality)를 설명하고 고차원 데이터가 모델 성능을 저하시키는 이유 이해하기
2. 특성 선택(feature selection)과 특성 추출(feature extraction) 접근법 구별하기
3. PCA를 구현하고 주성분(principal components), 설명된 분산 비율(explained variance ratios), 스크리 플롯(scree plots) 해석하기
4. t-SNE와 UMAP을 비선형 시각화에 적용하고 PCA와 동작 방식 비교하기
5. 누적 설명 분산(cumulative explained variance) 임계값을 사용해 PCA 구성 요소 수 결정하기
6. scikit-learn을 이용해 필터(filter), 래퍼(wrapper), 임베디드(embedded) 특성 선택 방법 구현하기
7. 차원 축소가 하위 분류(downstream classification) 성능에 미치는 영향 평가하기

---

실제 데이터셋에는 수십, 수백, 심지어 수천 개의 특성이 포함되는 경우가 많습니다 -- 그러나 유용한 정보의 대부분은 훨씬 적은 수의 방향에 집중되어 있는 경우가 대부분입니다. 차원 축소를 사용하면 그러한 방향을 찾아 노이즈와 중복을 제거하여 모델 학습 속도를 높이고, 시각화를 의미 있게 만들며, 차원의 저주가 알고리즘을 마비시키지 않도록 합니다. 이 기법들을 숙달하는 것은 유전체학부터 컴퓨터 비전(computer vision), NLP 임베딩(NLP embeddings)까지 고차원 데이터를 다루기 위한 전제 조건입니다.

> **벽 위의 그림자.** 복잡한 3D 조각상에 손전등을 비춘다고 상상해 보세요 -- 벽에 비친 그림자는 조각상의 핵심 형태를 담은 2D 표현이지만, 일부 세부 정보는 손실됩니다. PCA도 같은 방식으로 작동합니다. 즉, 고차원 데이터를 더 낮은 차원으로 투영할 때 가장 정보가 풍부한 "그림자"를 만들어내는 "시점(주성분)"을 찾습니다. 첫 번째 주성분(PC1)은 분산을 가장 많이 보존하는 방향 -- 그림자를 가장 잘 알아볼 수 있게 하는 각도 -- 입니다.

---

## 1. 차원 축소의 필요성

### 1.1 차원의 저주 (Curse of Dimensionality)

```python
"""
차원의 저주:
1. 고차원에서 데이터 포인트 간 거리가 비슷해짐
2. 데이터가 희소해짐 (sparse)
3. 모델 학습에 더 많은 데이터 필요
4. 과적합 위험 증가
5. 계산 비용 증가

차원 축소의 목적:
1. 시각화 (2D/3D)
2. 노이즈 제거
3. 계산 효율성
4. 다중공선성 제거
5. 특성 추출
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, load_iris, fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 차원의 저주 데모: 고차원에서 거리 분포
np.random.seed(42)

def distance_distribution(n_dims, n_points=1000):
    """고차원에서 거리 분포 확인"""
    points = np.random.rand(n_points, n_dims)
    # 랜덤 포인트 쌍 간 거리
    idx = np.random.choice(n_points, size=(500, 2), replace=False)
    distances = [np.linalg.norm(points[i] - points[j]) for i, j in idx]
    return distances

# 다양한 차원에서 거리 분포
dims = [2, 10, 100, 1000]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, d in zip(axes, dims):
    distances = distance_distribution(d)
    ax.hist(distances, bins=30, edgecolor='black')
    ax.set_title(f'Dim={d}\nMean={np.mean(distances):.2f}, Std={np.std(distances):.2f}')
    ax.set_xlabel('Distance')

plt.tight_layout()
plt.show()

print("차원이 증가할수록 거리 분포가 좁아짐 → 포인트들이 비슷한 거리에 위치")
```

---

## 2. 주성분 분석 (PCA)

### 2.1 PCA의 원리

```python
"""
PCA (Principal Component Analysis):
- 데이터의 분산을 최대화하는 축(주성분)을 찾음
- 고차원 → 저차원 투영
- 선형 변환

수학적 원리:
1. 데이터 중심화 (평균 0)
2. 공분산 행렬 계산
3. 고유값 분해 (eigendecomposition)
4. 고유값이 큰 순서로 고유벡터(주성분) 선택
5. 선택된 주성분으로 데이터 투영

주성분:
- 첫 번째 주성분: 분산이 가장 큰 방향
- 두 번째 주성분: 첫 번째와 직교하면서 분산이 큰 방향
- n번째 주성분: 이전 주성분들과 직교
"""

from sklearn.decomposition import PCA

# 2D 예시로 PCA 시각화
np.random.seed(42)
X_2d = np.dot(np.random.randn(200, 2), [[2, 1], [1, 2]])

# PCA 적용
pca = PCA(n_components=2)
pca.fit(X_2d)

# 시각화
plt.figure(figsize=(10, 8))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)

# 주성분 방향 (화살표)
mean = pca.mean_
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    end = mean + comp * np.sqrt(var) * 3
    plt.arrow(mean[0], mean[1], end[0]-mean[0], end[1]-mean[1],
              head_width=0.3, head_length=0.2, fc=f'C{i}', ec=f'C{i}',
              linewidth=2, label=f'PC{i+1} (Var: {var:.2f})')

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('PCA: Principal Components')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()

print(f"주성분:\n{pca.components_}")
print(f"설명된 분산: {pca.explained_variance_}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
```

### 2.2 sklearn PCA 사용법

```python
from sklearn.decomposition import PCA

# Iris 데이터
iris = load_iris()
X = iris.data
y = iris.target

# 스케일링 (PCA 전 필수)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 적용 (2차원으로 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"원본 형상: {X.shape}")
print(f"PCA 후 형상: {X_pca.shape}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 설명 분산: {sum(pca.explained_variance_ratio_):.4f}")

# 시각화
plt.figure(figsize=(10, 8))
for i, target_name in enumerate(iris.target_names):
    mask = y == i
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=target_name, alpha=0.7)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.title('PCA: Iris Dataset')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.3 주성분 수 선택

```python
# 전체 주성분으로 PCA
pca_full = PCA()
pca_full.fit(X_scaled)

# 누적 설명 분산
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 개별 분산
axes[0].bar(range(1, len(pca_full.explained_variance_ratio_)+1),
            pca_full.explained_variance_ratio_, edgecolor='black')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Individual Explained Variance')

# 누적 분산
axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'o-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
axes[1].axhline(y=0.99, color='g', linestyle='--', label='99% variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Explained Variance')
axes[1].set_title('Cumulative Explained Variance')
axes[1].legend()

plt.tight_layout()
plt.show()

# 95% 분산을 설명하는 주성분 수
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"95% 분산 설명에 필요한 주성분 수: {n_components_95}")
```

### 2.4 PCA로 분산 비율 지정

```python
# 분산 비율로 주성분 수 자동 결정
pca_95 = PCA(n_components=0.95)  # 95% 분산 설명
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"95% 분산 → {pca_95.n_components_}개 주성분 선택")
print(f"실제 설명된 분산: {sum(pca_95.explained_variance_ratio_):.4f}")

# 다양한 분산 비율
for var_ratio in [0.8, 0.9, 0.95, 0.99]:
    pca_temp = PCA(n_components=var_ratio)
    pca_temp.fit(X_scaled)
    print(f"{var_ratio*100:.0f}% 분산 → {pca_temp.n_components_}개 주성분")
```

### 2.5 PCA 활용: 노이즈 제거

```python
# 숫자 이미지 데이터
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# 노이즈 추가
np.random.seed(42)
X_noisy = X_digits + np.random.normal(0, 4, X_digits.shape)

# PCA로 노이즈 제거 (주요 주성분만 유지)
pca_denoise = PCA(n_components=20)
X_reduced = pca_denoise.fit_transform(X_noisy)
X_denoised = pca_denoise.inverse_transform(X_reduced)

# 시각화
fig, axes = plt.subplots(3, 10, figsize=(15, 5))

for i in range(10):
    # 원본
    axes[0, i].imshow(X_digits[i].reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original')

    # 노이즈
    axes[1, i].imshow(X_noisy[i].reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title('Noisy')

    # 복원
    axes[2, i].imshow(X_denoised[i].reshape(8, 8), cmap='gray')
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title('Denoised')

plt.suptitle('PCA for Noise Reduction')
plt.tight_layout()
plt.show()
```

---

## 3. t-SNE

### 3.1 t-SNE 원리

```python
"""
t-SNE (t-distributed Stochastic Neighbor Embedding):
- 비선형 차원 축소
- 시각화에 주로 사용 (2D/3D)
- 지역 구조 보존에 뛰어남

원리:
1. 고차원에서 점들 간 유사도를 조건부 확률로 계산
2. 저차원에서 t-분포 기반 유사도 정의
3. KL-divergence 최소화로 저차원 좌표 학습

특징:
- 비선형 관계 포착
- 클러스터 분리에 효과적
- 계산 비용 높음
- 새 데이터 변환 불가 (transform 없음)
- 결과 재현성 문제 (random_state 중요)
"""

from sklearn.manifold import TSNE

# t-SNE 적용
tsne = TSNE(
    n_components=2,
    perplexity=30,          # 지역 이웃 크기 (5-50)
    learning_rate='auto',   # 학습률
    n_iter=1000,            # 반복 횟수
    random_state=42
)

# 시간이 오래 걸리므로 일부만 사용
X_sample = X_digits[:500]
y_sample = y_digits[:500]

X_tsne = tsne.fit_transform(X_sample)

# 시각화
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
plt.colorbar(scatter)
plt.title('t-SNE: Digits Dataset')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.show()
```

### 3.2 perplexity 파라미터

```python
# perplexity 효과
perplexities = [5, 30, 50, 100]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, perp in zip(axes, perplexities):
    tsne_temp = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_temp = tsne_temp.fit_transform(X_sample)

    scatter = ax.scatter(X_temp[:, 0], X_temp[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
    ax.set_title(f'perplexity={perp}')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

print("perplexity 가이드:")
print("  - 작은 값 (5-10): 지역 구조에 집중")
print("  - 큰 값 (30-50): 전역 구조 고려")
print("  - 데이터 크기에 따라 조절 필요")
```

### 3.3 PCA vs t-SNE 비교

```python
# 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# 비교 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
axes[0].set_title('PCA')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='tab10', alpha=0.7)
axes[1].set_title('t-SNE')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()

print("PCA: 분산 최대화, 선형, 빠름, 전역 구조")
print("t-SNE: 이웃 보존, 비선형, 느림, 지역 구조")
```

---

## 4. UMAP

```python
"""
UMAP (Uniform Manifold Approximation and Projection):
- t-SNE보다 빠름
- 전역 구조 더 잘 보존
- 새 데이터 변환 가능

# pip install umap-learn
"""

# import umap

# umap_reducer = umap.UMAP(
#     n_neighbors=15,      # 지역 이웃 수
#     min_dist=0.1,        # 포인트 간 최소 거리
#     n_components=2,
#     random_state=42
# )
# X_umap = umap_reducer.fit_transform(X_scaled)

# 설치 없이 설명
print("UMAP 특징:")
print("  - t-SNE보다 빠름")
print("  - 전역 구조 더 잘 보존")
print("  - transform() 지원 (새 데이터 변환)")
print("  - 주요 파라미터: n_neighbors, min_dist")
```

---

## 5. 특성 선택 (Feature Selection)

### 5.1 필터 방법 (Filter Methods)

```python
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile,
    f_classif, mutual_info_classif, chi2
)

"""
필터 방법:
- 모델과 독립적으로 특성 평가
- 빠름, 간단
- 통계적 검정 기반

방법:
1. 분산 기반: VarianceThreshold
2. 상관관계 기반: 타겟과의 상관계수
3. 통계 검정: ANOVA F-value, 카이제곱
4. 정보 이론: 상호 정보량
"""

# 데이터
X, y = load_iris(return_X_y=True)

# ANOVA F-value 기반 특성 선택
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)

print("ANOVA F-value 특성 선택:")
print(f"원본 특성 수: {X.shape[1]}")
print(f"선택된 특성 수: {X_selected.shape[1]}")
print(f"각 특성 점수: {selector.scores_}")
print(f"선택된 특성 인덱스: {selector.get_support(indices=True)}")

# 상호 정보량 기반
selector_mi = SelectKBest(score_func=mutual_info_classif, k=2)
selector_mi.fit(X, y)
print(f"\n상호 정보량 점수: {selector_mi.scores_}")
```

### 5.2 래퍼 방법 (Wrapper Methods)

```python
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression

"""
래퍼 방법:
- 모델 성능 기반 특성 선택
- 정확하지만 느림
- 과적합 위험

방법:
1. RFE (Recursive Feature Elimination)
2. 전진 선택 (Forward Selection)
3. 후진 제거 (Backward Elimination)
"""

# RFE (재귀적 특성 제거)
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=2, step=1)
rfe.fit(X, y)

print("RFE 특성 선택:")
print(f"선택된 특성: {rfe.get_support()}")
print(f"특성 순위: {rfe.ranking_}")

# RFECV (교차 검증 포함)
rfecv = RFECV(estimator=model, cv=5, scoring='accuracy')
rfecv.fit(X, y)

print(f"\nRFECV 최적 특성 수: {rfecv.n_features_}")
print(f"선택된 특성: {rfecv.get_support()}")

# CV 점수 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score'])+1),
         rfecv.cv_results_['mean_test_score'], 'o-')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Score')
plt.title('RFECV: Optimal Number of Features')
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.3 임베디드 방법 (Embedded Methods)

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso

"""
임베디드 방법:
- 모델 학습 과정에서 특성 선택
- 필터와 래퍼의 중간
- L1 정규화, 트리 기반 모델

방법:
1. L1 정규화 (Lasso)
2. 트리 기반 중요도
"""

# Random Forest 중요도 기반
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# 특성 중요도
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [f'Feature {i}' for i in indices])
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Random Forest Feature Importance')
plt.show()

# SelectFromModel
selector = SelectFromModel(rf, threshold='median')
selector.fit(X, y)
X_selected = selector.transform(X)

print(f"Random Forest 기반 선택된 특성 수: {X_selected.shape[1]}")
print(f"선택된 특성: {selector.get_support()}")
```

---

## 6. 분산 기반 특성 선택

```python
from sklearn.feature_selection import VarianceThreshold

# 샘플 데이터 (분산이 다른 특성)
X_var = np.array([
    [0, 0, 1, 100],
    [0, 0, 0, 101],
    [0, 0, 1, 99],
    [0, 0, 0, 100],
    [0, 0, 1, 102]
])

# 분산이 낮은 특성 제거
selector = VarianceThreshold(threshold=0.5)
X_high_var = selector.fit_transform(X_var)

print("분산 기반 특성 선택:")
print(f"각 특성 분산: {selector.variances_}")
print(f"선택된 특성: {selector.get_support()}")
print(f"원본 형상: {X_var.shape}")
print(f"선택 후 형상: {X_high_var.shape}")
```

---

## 7. 상관관계 기반 특성 제거

```python
import pandas as pd

# 샘플 데이터 (상관된 특성 포함)
np.random.seed(42)
n_samples = 100

X_corr = np.column_stack([
    np.random.randn(n_samples),  # 특성 0
    np.random.randn(n_samples),  # 특성 1
    np.random.randn(n_samples),  # 특성 2
])
# 높은 상관관계 특성 추가
X_corr = np.column_stack([X_corr, X_corr[:, 0] + np.random.randn(n_samples) * 0.1])

df = pd.DataFrame(X_corr, columns=['F0', 'F1', 'F2', 'F3'])

# 상관행렬
corr_matrix = df.corr().abs()

# 상관관계 히트맵
plt.figure(figsize=(8, 6))
plt.imshow(corr_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.colorbar(label='Correlation')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.title('Feature Correlation Matrix')

for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                 ha='center', va='center')
plt.show()

# 높은 상관관계 특성 제거 함수
def remove_highly_correlated(df, threshold=0.9):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

df_cleaned, dropped = remove_highly_correlated(df, threshold=0.9)
print(f"제거된 특성: {dropped}")
print(f"남은 특성: {list(df_cleaned.columns)}")
```

---

## 8. 차원 축소 파이프라인

```python
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# 데이터
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PCA + SVM 파이프라인
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=30)),
    ('svm', SVC(kernel='rbf', random_state=42))
])

# 교차 검증
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
print(f"PCA (30) + SVM CV 점수: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 전체 특성 vs PCA
pipeline_full = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42))
])

cv_scores_full = cross_val_score(pipeline_full, X_train, y_train, cv=5)
print(f"전체 특성 + SVM CV 점수: {cv_scores_full.mean():.4f} (+/- {cv_scores_full.std():.4f})")

print(f"\nPCA로 {X.shape[1]} → 30 차원 축소")
```

---

## 9. Incremental PCA (대용량 데이터)

```python
from sklearn.decomposition import IncrementalPCA

"""
Incremental PCA:
- 대용량 데이터에 적합
- 미니배치로 처리
- 메모리 효율적
"""

# 대용량 데이터 시뮬레이션
X_large = np.random.randn(10000, 100)

# 일반 PCA
pca_regular = PCA(n_components=10)
pca_regular.fit(X_large)

# Incremental PCA
ipca = IncrementalPCA(n_components=10, batch_size=500)
ipca.fit(X_large)

print("일반 PCA vs Incremental PCA:")
print(f"설명된 분산 비율 (일반): {sum(pca_regular.explained_variance_ratio_):.4f}")
print(f"설명된 분산 비율 (증분): {sum(ipca.explained_variance_ratio_):.4f}")

# 배치로 처리 (메모리 효율)
ipca_batch = IncrementalPCA(n_components=10)
for batch_start in range(0, len(X_large), 1000):
    batch = X_large[batch_start:batch_start+1000]
    ipca_batch.partial_fit(batch)

print(f"배치 처리 설명된 분산: {sum(ipca_batch.explained_variance_ratio_):.4f}")
```

---

## 10. 차원 축소 알고리즘 비교

```python
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

"""
차원 축소 알고리즘 비교:

1. PCA: 선형, 분산 최대화, 빠름
2. Kernel PCA: 비선형 PCA
3. LDA: 클래스 분리 최대화 (지도 학습)
4. t-SNE: 시각화, 지역 구조
5. UMAP: 시각화, 전역+지역 구조
6. MDS: 거리 보존
7. Isomap: 측지선 거리 보존
"""

# 알고리즘 비교 (작은 데이터셋)
algorithms = {
    'PCA': PCA(n_components=2),
    'Kernel PCA': KernelPCA(n_components=2, kernel='rbf'),
    'LDA': LDA(n_components=2),
    't-SNE': TSNE(n_components=2, random_state=42)
}

# 데이터
X, y = load_iris(return_X_y=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 비교 시각화
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

for ax, (name, algo) in zip(axes, algorithms.items()):
    if name == 'LDA':
        X_reduced = algo.fit_transform(X_scaled, y)
    else:
        X_reduced = algo.fit_transform(X_scaled)

    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', alpha=0.7)
    ax.set_title(name)
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')

plt.tight_layout()
plt.show()
```

---

## 연습 문제

### 문제 1: PCA 적용
Digits 데이터에 PCA를 적용하고 95% 분산을 설명하는 주성분 수를 찾으세요.

```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data

# 풀이
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)

cumsum = np.cumsum(pca.explained_variance_ratio_)
n_95 = np.argmax(cumsum >= 0.95) + 1

print(f"95% 분산에 필요한 주성분 수: {n_95}")
print(f"원본 차원: {X.shape[1]}")
```

### 문제 2: t-SNE 시각화
Digits 데이터를 t-SNE로 시각화하세요.

```python
from sklearn.manifold import TSNE

# 풀이 (시간 단축을 위해 일부만)
X_sample = X[:500]
y_sample = digits.target[:500]

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_sample)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE: Digits')
plt.show()
```

### 문제 3: 특성 선택
Random Forest 중요도 기반으로 상위 20개 특성을 선택하세요.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# 풀이
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, digits.target)

# 상위 20개
selector = SelectFromModel(rf, max_features=20, threshold=-np.inf)
selector.fit(X, digits.target)
X_selected = selector.transform(X)

print(f"선택된 특성 수: {X_selected.shape[1]}")
print(f"선택된 특성 인덱스: {np.where(selector.get_support())[0]}")
```

---

## 요약

| 방법 | 유형 | 특징 | 용도 |
|------|------|------|------|
| PCA | 선형 | 분산 최대화 | 일반적인 차원 축소 |
| Kernel PCA | 비선형 | 커널 트릭 | 비선형 패턴 |
| LDA | 지도 학습 | 클래스 분리 | 분류 전처리 |
| t-SNE | 비선형 | 지역 구조 보존 | 시각화 |
| UMAP | 비선형 | 빠름, 전역 구조 | 시각화 |

### 특성 선택 방법 비교

| 방법 | 유형 | 장점 | 단점 |
|------|------|------|------|
| Filter | 통계 기반 | 빠름 | 특성 간 관계 무시 |
| Wrapper | 모델 기반 | 정확 | 느림, 과적합 |
| Embedded | 학습 중 선택 | 효율적 | 모델 의존적 |

### 차원 축소 선택 가이드

| 상황 | 권장 방법 |
|------|-----------|
| 노이즈 제거, 압축 | PCA |
| 시각화 (2D/3D) | t-SNE, UMAP |
| 분류 전처리 | LDA |
| 비선형 패턴 | Kernel PCA, UMAP |
| 대용량 데이터 | Incremental PCA, TruncatedSVD |
| 특성 해석 필요 | 특성 선택 (Filter/Embedded) |
