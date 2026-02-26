# 22. 다변량 분석 (Multivariate Analysis)

[이전: 시계열 모형](./21_Time_Series_Models.md) | [다음: 비모수 통계](./23_Nonparametric_Statistics.md)

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. 고유값 분해(Eigenvalue Decomposition)와 분산 최대화를 포함한 PCA의 수학적 기반 설명
2. PCA를 처음부터 구현하고 scikit-learn 결과와 검증
3. Kaiser 규칙, 분산 기준, 스크리 플롯(Scree Plot)을 사용하여 적절한 주성분 개수 결정
4. PCA(분산 기반)와 요인 분석(Factor Analysis, 잠재 변수 기반)의 차이를 구분하고 각각의 적합한 사용 상황 파악
5. 요인 해석력 향상을 위한 Varimax 회전 적용
6. 지도 분류를 위한 LDA와 QDA 비교, 클래스 공분산에 대한 가정 포함
7. 내부 지표(실루엣(Silhouette), Calinski-Harabasz, Davies-Bouldin)와 외부 지표(ARI, NMI)를 사용하여 군집 품질 평가

---

실제 데이터셋에는 변수가 하나나 둘만 있는 경우가 거의 없습니다. 수십 개의 특성이 동시에 눈길을 끌 때, 변수들이 서로 어떤 관계를 맺는지 먼저 파악하지 않으면 패턴을 시각화하거나 안정적인 모형을 구축하는 것이 불가능합니다. 다변량 분석(Multivariate Analysis)은 복잡성을 줄이고, 잠재 구조를 발견하며, 관측값을 분류하고, 그룹화를 검증하는 도구를 제공합니다 — 고차원의 노이즈를 실행 가능한 인사이트로 전환하는 것입니다.

---

> **그림자 투영.** 3D 철사 조형물을 램프 앞에 들고 있다고 상상해보세요. 벽에 생기는 그림자는 조형물의 모양을 담은 2D 투영이지만, 어떤 각도는 다른 각도보다 더 많은 정보를 담습니다. 차원 축소(Dimensionality Reduction)(PCA, 요인 분석)도 같은 원리로 작동합니다: 고차원 데이터를 더 적은 차원으로 투영할 때 가장 많은 구조를 보존하는 "관찰 각도"를 찾는 것입니다. 주성분(Principal Components)이 바로 가장 정보가 풍부한 그림자를 만드는 방향입니다.

## 1. 주성분 분석 (PCA)

### 1.1 PCA 개념

**목표**: 고차원 데이터를 저차원으로 투영하면서 분산을 최대한 보존

**주성분**: 데이터의 분산을 최대화하는 직교 방향

**수학적 정의**:
- 첫 번째 주성분: Var(w₁ᵀX)를 최대화하는 단위벡터 w₁
- k번째 주성분: 이전 주성분들과 직교하면서 분산을 최대화

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine, load_breast_cancer

np.random.seed(42)

# 2D 예시로 PCA 직관 이해
n = 200
theta = np.pi / 4
cov = [[3, 2], [2, 2]]
X_2d = np.random.multivariate_normal([0, 0], cov, n)

# PCA 수행
pca_2d = PCA()
X_pca = pca_2d.fit_transform(X_2d)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 원본 데이터
ax = axes[0]
ax.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)
# 주성분 방향 표시
mean = X_2d.mean(axis=0)
for i, (comp, var) in enumerate(zip(pca_2d.components_, pca_2d.explained_variance_)):
    ax.annotate('', xy=mean + 2 * np.sqrt(var) * comp, xytext=mean,
                arrowprops=dict(arrowstyle='->', color=['red', 'blue'][i], lw=2))
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_title('원본 데이터와 주성분 방향')
ax.axis('equal')
ax.grid(True, alpha=0.3)

# 주성분 공간
ax = axes[1]
ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('주성분 공간')
ax.axis('equal')
ax.grid(True, alpha=0.3)

# 설명된 분산
ax = axes[2]
explained_var_ratio = pca_2d.explained_variance_ratio_
ax.bar([1, 2], explained_var_ratio, alpha=0.7)
ax.set_xlabel('주성분')
ax.set_ylabel('설명된 분산 비율')
ax.set_title(f'설명된 분산: PC1={explained_var_ratio[0]:.1%}, PC2={explained_var_ratio[1]:.1%}')
ax.set_xticks([1, 2])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 1.2 PCA 이론

```python
def pca_from_scratch(X, n_components=None):
    """
    PCA 처음부터 구현

    1. 데이터 중심화 (평균 0)
    2. 공분산 행렬 계산
    3. 고유값 분해
    4. 고유벡터 정렬 (큰 고유값 순)
    """
    # 중심화
    X_centered = X - X.mean(axis=0)

    # 공분산 행렬
    n = X.shape[0]
    cov_matrix = (X_centered.T @ X_centered) / (n - 1)

    # 고유값 분해
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 내림차순 정렬
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 설명된 분산 비율
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]
        explained_variance_ratio = explained_variance_ratio[:n_components]

    # 투영
    X_pca = X_centered @ eigenvectors

    return {
        'components': eigenvectors.T,
        'explained_variance': eigenvalues,
        'explained_variance_ratio': explained_variance_ratio,
        'transformed': X_pca
    }

# 검증: sklearn과 비교
X_test = np.random.randn(100, 5)
result_scratch = pca_from_scratch(X_test, n_components=3)
pca_sklearn = PCA(n_components=3).fit(X_test)

print("=== PCA 구현 검증 ===")
print(f"설명된 분산 비율 (scratch): {result_scratch['explained_variance_ratio']}")
print(f"설명된 분산 비율 (sklearn): {pca_sklearn.explained_variance_ratio_}")
print("(부호가 다를 수 있지만 절대값은 동일해야 함)")
```

### 1.3 sklearn을 이용한 PCA

```python
# Iris 데이터셋
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# 결과 분석
print("=== Iris PCA 결과 ===")
print(f"원본 특성 수: {X_iris.shape[1]}")
print(f"설명된 분산 비율: {pca.explained_variance_ratio_}")
print(f"누적 설명된 분산: {np.cumsum(pca.explained_variance_ratio_)}")

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 스크리 플롯
ax = axes[0]
ax.bar(range(1, 5), pca.explained_variance_ratio_, alpha=0.7, label='개별')
ax.plot(range(1, 5), np.cumsum(pca.explained_variance_ratio_), 'ro-', label='누적')
ax.set_xlabel('주성분')
ax.set_ylabel('설명된 분산 비율')
ax.set_title('스크리 플롯')
ax.legend()
ax.set_xticks(range(1, 5))
ax.grid(True, alpha=0.3)

# PC1 vs PC2
ax = axes[1]
for target in np.unique(y_iris):
    mask = y_iris == target
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.7,
               label=iris.target_names[target])
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_title('PCA: PC1 vs PC2')
ax.legend()
ax.grid(True, alpha=0.3)

# 주성분 로딩
ax = axes[2]
loadings = pd.DataFrame(
    pca.components_[:2].T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
loadings.plot(kind='bar', ax=ax, alpha=0.7)
ax.set_ylabel('로딩')
ax.set_title('주성분 로딩')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()
```

### 1.4 주성분 개수 결정

```python
def determine_n_components(X, methods=['kaiser', 'variance', 'elbow']):
    """
    주성분 개수 결정 방법

    1. Kaiser 규칙: 고유값 > 1 (표준화된 데이터)
    2. 분산 기준: 누적 분산 >= 임계값 (보통 85-95%)
    3. 스크리 플롯: 팔꿈치 지점
    """
    pca = PCA()
    pca.fit(X)

    results = {}

    # Kaiser 규칙
    if 'kaiser' in methods:
        n_kaiser = np.sum(pca.explained_variance_ > 1)
        results['kaiser'] = n_kaiser
        print(f"Kaiser 규칙 (고유값 > 1): {n_kaiser}개")

    # 분산 기준
    if 'variance' in methods:
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_80 = np.argmax(cumsum >= 0.80) + 1
        n_90 = np.argmax(cumsum >= 0.90) + 1
        n_95 = np.argmax(cumsum >= 0.95) + 1
        results['variance_80'] = n_80
        results['variance_90'] = n_90
        results['variance_95'] = n_95
        print(f"분산 기준 80%: {n_80}개")
        print(f"분산 기준 90%: {n_90}개")
        print(f"분산 기준 95%: {n_95}개")

    # 스크리 플롯
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(range(1, len(pca.explained_variance_) + 1),
            pca.explained_variance_, 'bo-')
    ax.axhline(1, color='r', linestyle='--', label='Kaiser (eigenvalue=1)')
    ax.set_xlabel('주성분')
    ax.set_ylabel('고유값')
    ax.set_title('스크리 플롯 (고유값)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, len(pca.explained_variance_ratio_) + 1),
            np.cumsum(pca.explained_variance_ratio_), 'go-')
    ax.axhline(0.80, color='orange', linestyle='--', label='80%')
    ax.axhline(0.90, color='r', linestyle='--', label='90%')
    ax.axhline(0.95, color='purple', linestyle='--', label='95%')
    ax.set_xlabel('주성분')
    ax.set_ylabel('누적 설명된 분산')
    ax.set_title('누적 분산')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results

# Wine 데이터셋으로 테스트
wine = load_wine()
X_wine = StandardScaler().fit_transform(wine.data)

print("=== Wine 데이터셋 주성분 개수 결정 ===")
results = determine_n_components(X_wine)
```

### 1.5 바이플롯 (Biplot)

```python
def biplot(X, y, pca, feature_names, target_names, ax=None):
    """
    PCA 바이플롯: 관측치와 변수를 동시에 표시
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    X_pca = pca.transform(X)

    # 스케일링 (관측치와 로딩을 같은 스케일로)
    scale = 1 / np.max(np.abs(X_pca[:, :2]))

    # 관측치 플롯
    for target in np.unique(y):
        mask = y == target
        ax.scatter(X_pca[mask, 0] * scale, X_pca[mask, 1] * scale,
                   alpha=0.5, label=target_names[target], s=30)

    # 로딩 벡터
    loadings = pca.components_[:2].T
    for i, (loading, name) in enumerate(zip(loadings, feature_names)):
        ax.arrow(0, 0, loading[0], loading[1],
                 head_width=0.05, head_length=0.03, fc='red', ec='red')
        ax.text(loading[0] * 1.1, loading[1] * 1.1, name, fontsize=9,
                ha='center', va='center')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('바이플롯 (Biplot)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)

    return ax

# Iris 바이플롯
pca_iris = PCA(n_components=4).fit(X_scaled)
biplot(X_scaled, y_iris, pca_iris, feature_names, iris.target_names)
plt.show()
```

---

## 2. 요인 분석 (Factor Analysis)

### 2.1 요인 분석 vs PCA

| 측면 | PCA | 요인 분석 |
|------|-----|----------|
| **목표** | 분산 최대화 | 잠재 요인 발견 |
| **모형** | 데이터 = 주성분 | 관측변수 = 요인 + 오차 |
| **고유분산** | 없음 | 각 변수별 고유분산 |
| **회전** | 불필요 (직교) | 해석을 위해 회전 |
| **사용** | 차원 축소 | 구조 발견, 설문 분석 |

### 2.2 요인 분석 모형

**모형**:
$$X_i = \mu_i + \lambda_{i1}F_1 + \lambda_{i2}F_2 + ... + \lambda_{im}F_m + \epsilon_i$$

- Fⱼ: 공통 요인 (latent factor)
- λᵢⱼ: 요인 적재량 (factor loading)
- εᵢ: 고유 오차 (unique factor)

```python
from sklearn.decomposition import FactorAnalysis
from scipy.stats import zscore

# 요인 분석 예시
np.random.seed(42)

# 잠재 요인 2개로 데이터 생성
n = 300
F1 = np.random.normal(0, 1, n)  # 요인 1
F2 = np.random.normal(0, 1, n)  # 요인 2

# 관측 변수 6개 (각 요인에 3개씩 로딩)
X1 = 0.8 * F1 + 0.1 * F2 + np.random.normal(0, 0.3, n)
X2 = 0.7 * F1 + 0.2 * F2 + np.random.normal(0, 0.3, n)
X3 = 0.9 * F1 + 0.0 * F2 + np.random.normal(0, 0.3, n)
X4 = 0.1 * F1 + 0.8 * F2 + np.random.normal(0, 0.3, n)
X5 = 0.2 * F1 + 0.7 * F2 + np.random.normal(0, 0.3, n)
X6 = 0.0 * F1 + 0.9 * F2 + np.random.normal(0, 0.3, n)

X_fa = np.column_stack([X1, X2, X3, X4, X5, X6])
X_fa = zscore(X_fa)  # 표준화

# 요인 분석
fa = FactorAnalysis(n_components=2, random_state=42)
F_scores = fa.fit_transform(X_fa)

print("=== 요인 분석 결과 ===")
print("\n요인 적재량 (Factor Loadings):")
loadings_df = pd.DataFrame(
    fa.components_.T,
    columns=['Factor 1', 'Factor 2'],
    index=[f'X{i+1}' for i in range(6)]
)
print(loadings_df.round(3))

print(f"\n고유분산 (Uniqueness):")
print(pd.Series(fa.noise_variance_, index=[f'X{i+1}' for i in range(6)]).round(3))

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 요인 적재량 플롯
ax = axes[0]
loadings_df.plot(kind='bar', ax=ax, alpha=0.7)
ax.set_ylabel('적재량')
ax.set_title('요인 적재량')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.axhline(0, color='k', linewidth=0.5)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 요인 점수
ax = axes[1]
ax.scatter(F_scores[:, 0], F_scores[:, 1], alpha=0.5)
ax.set_xlabel('Factor 1')
ax.set_ylabel('Factor 2')
ax.set_title('요인 점수')
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2.3 요인 회전

```python
def varimax_rotation(loadings, n_iter=100, tol=1e-6):
    """
    Varimax 회전 (직교 회전)
    적재량 분산을 최대화하여 해석 용이성 향상
    """
    p, k = loadings.shape
    rotated = loadings.copy()

    for _ in range(n_iter):
        old_rotated = rotated.copy()

        for i in range(k - 1):
            for j in range(i + 1, k):
                # 2x2 회전
                x = rotated[:, i]
                y = rotated[:, j]

                u = x**2 - y**2
                v = 2 * x * y

                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u**2 - v**2)
                D = 2 * np.sum(u * v)

                phi = 0.25 * np.arctan2(D - 2 * A * B / p,
                                         C - (A**2 - B**2) / p)

                cos_phi = np.cos(phi)
                sin_phi = np.sin(phi)

                rotated[:, i] = x * cos_phi + y * sin_phi
                rotated[:, j] = -x * sin_phi + y * cos_phi

        if np.max(np.abs(rotated - old_rotated)) < tol:
            break

    return rotated

# 회전 전후 비교
loadings_original = fa.components_.T
loadings_rotated = varimax_rotation(loadings_original)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 회전 전
ax = axes[0]
pd.DataFrame(loadings_original, columns=['F1', 'F2'],
             index=[f'X{i+1}' for i in range(6)]).plot(kind='bar', ax=ax, alpha=0.7)
ax.set_title('회전 전')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, alpha=0.3, axis='y')

# 회전 후
ax = axes[1]
pd.DataFrame(loadings_rotated, columns=['F1', 'F2'],
             index=[f'X{i+1}' for i in range(6)]).plot(kind='bar', ax=ax, alpha=0.7)
ax.set_title('Varimax 회전 후')
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print("=== Varimax 회전 후 적재량 ===")
print(pd.DataFrame(loadings_rotated, columns=['Factor 1', 'Factor 2'],
                   index=[f'X{i+1}' for i in range(6)]).round(3))
```

---

## 3. 판별 분석 (Discriminant Analysis)

### 3.1 LDA (Linear Discriminant Analysis)

**목표**: 클래스 간 분리를 최대화하는 선형 결합 찾기

**기준**: 클래스 간 분산 / 클래스 내 분산 최대화

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# Iris 데이터 LDA
lda = LinearDiscriminantAnalysis()
X_lda = lda.fit_transform(X_scaled, y_iris)

print("=== LDA 결과 ===")
print(f"판별함수 개수: {X_lda.shape[1]}")
print(f"설명된 분산 비율: {lda.explained_variance_ratio_}")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LDA 투영
ax = axes[0]
for target in np.unique(y_iris):
    mask = y_iris == target
    ax.scatter(X_lda[mask, 0], X_lda[mask, 1], alpha=0.7,
               label=iris.target_names[target])
ax.set_xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.1%})')
ax.set_title('LDA 투영')
ax.legend()
ax.grid(True, alpha=0.3)

# PCA vs LDA 비교
ax = axes[1]
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
for target in np.unique(y_iris):
    mask = y_iris == target
    ax.scatter(X_pca_2[mask, 0], X_pca_2[mask, 1], alpha=0.7,
               label=iris.target_names[target])
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA 투영 (비교)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 3.2 LDA 분류기

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_iris, test_size=0.3, random_state=42
)

# LDA 분류기
lda_clf = LinearDiscriminantAnalysis()
lda_clf.fit(X_train, y_train)
y_pred_lda = lda_clf.predict(X_test)

print("=== LDA 분류 성능 ===")
print(f"훈련 정확도: {lda_clf.score(X_train, y_train):.4f}")
print(f"테스트 정확도: {lda_clf.score(X_test, y_test):.4f}")
print("\n분류 보고서:")
print(classification_report(y_test, y_pred_lda, target_names=iris.target_names))

# 교차검증
cv_scores = cross_val_score(lda_clf, X_scaled, y_iris, cv=5)
print(f"\n5-fold CV 정확도: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
```

### 3.3 QDA (Quadratic Discriminant Analysis)

```python
# QDA: 각 클래스별 다른 공분산 허용
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)
y_pred_qda = qda.predict(X_test)

print("=== QDA 분류 성능 ===")
print(f"훈련 정확도: {qda.score(X_train, y_train):.4f}")
print(f"테스트 정확도: {qda.score(X_test, y_test):.4f}")

# LDA vs QDA 비교
print("\n=== LDA vs QDA 비교 ===")
comparison = pd.DataFrame({
    'Model': ['LDA', 'QDA'],
    'Train Accuracy': [lda_clf.score(X_train, y_train),
                       qda.score(X_train, y_train)],
    'Test Accuracy': [lda_clf.score(X_test, y_test),
                      qda.score(X_test, y_test)]
})
print(comparison)

print("\nLDA vs QDA 선택 기준:")
print("- LDA: 클래스 간 공분산이 같다고 가정, 더 단순, 작은 데이터에 적합")
print("- QDA: 클래스별 다른 공분산 허용, 더 유연, 큰 데이터에 적합")
```

### 3.4 결정 경계 시각화

```python
def plot_decision_boundary_2d(model, X, y, title='', ax=None):
    """2D 결정 경계 시각화"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    # 그리드 생성
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 예측
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 결정 경계
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.contour(xx, yy, Z, colors='k', linewidths=0.5)

    # 데이터 포인트
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         edgecolors='k', s=50, alpha=0.7)

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax

# 2D로 축소하여 시각화
X_2d = X_scaled[:, :2]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y_iris, test_size=0.3, random_state=42
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LDA 결정 경계
lda_2d = LinearDiscriminantAnalysis()
lda_2d.fit(X_train_2d, y_train_2d)
plot_decision_boundary_2d(lda_2d, X_2d, y_iris, 'LDA 결정 경계', axes[0])

# QDA 결정 경계
qda_2d = QuadraticDiscriminantAnalysis()
qda_2d.fit(X_train_2d, y_train_2d)
plot_decision_boundary_2d(qda_2d, X_2d, y_iris, 'QDA 결정 경계', axes[1])

plt.tight_layout()
plt.show()
```

---

## 4. 군집 타당성 검증 (Cluster Validation)

### 4.1 내부 지표

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score

# 군집 분석 예시
np.random.seed(42)

# 데이터 생성 (3개 군집)
n_samples = 300
X_cluster = np.vstack([
    np.random.normal([0, 0], 0.5, (n_samples//3, 2)),
    np.random.normal([3, 3], 0.5, (n_samples//3, 2)),
    np.random.normal([0, 3], 0.5, (n_samples//3, 2))
])

def evaluate_clustering(X, k_range=range(2, 8)):
    """
    다양한 K에 대해 군집 타당성 평가
    """
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        davies = davies_bouldin_score(X, labels)
        inertia = kmeans.inertia_

        results.append({
            'k': k,
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'inertia': inertia
        })

    return pd.DataFrame(results)

# 평가
eval_results = evaluate_clustering(X_cluster)
print("=== 군집 타당성 지표 ===")
print(eval_results)

# 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 엘보우 플롯 (Inertia)
ax = axes[0, 0]
ax.plot(eval_results['k'], eval_results['inertia'], 'bo-')
ax.set_xlabel('K')
ax.set_ylabel('Inertia')
ax.set_title('엘보우 플롯')
ax.grid(True, alpha=0.3)

# 실루엣 점수
ax = axes[0, 1]
ax.plot(eval_results['k'], eval_results['silhouette'], 'go-')
ax.set_xlabel('K')
ax.set_ylabel('Silhouette Score')
ax.set_title('실루엣 점수 (높을수록 좋음)')
ax.grid(True, alpha=0.3)

# Calinski-Harabasz
ax = axes[1, 0]
ax.plot(eval_results['k'], eval_results['calinski_harabasz'], 'ro-')
ax.set_xlabel('K')
ax.set_ylabel('Calinski-Harabasz Index')
ax.set_title('Calinski-Harabasz (높을수록 좋음)')
ax.grid(True, alpha=0.3)

# Davies-Bouldin
ax = axes[1, 1]
ax.plot(eval_results['k'], eval_results['davies_bouldin'], 'mo-')
ax.set_xlabel('K')
ax.set_ylabel('Davies-Bouldin Index')
ax.set_title('Davies-Bouldin (낮을수록 좋음)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 4.2 실루엣 분석

```python
def silhouette_analysis(X, n_clusters):
    """
    실루엣 분석 시각화
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 실루엣 플롯
    ax = axes[0]
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         alpha=0.7, label=f'Cluster {i}')

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label=f'평균: {silhouette_avg:.3f}')
    ax.set_xlabel('실루엣 계수')
    ax.set_ylabel('군집')
    ax.set_title(f'실루엣 분석 (K={n_clusters})')
    ax.legend()

    # 군집 시각화
    ax = axes[1]
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    for i, c in enumerate(colors):
        ax.scatter(X[labels == i, 0], X[labels == i, 1],
                   color=c, alpha=0.7, label=f'Cluster {i}')
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               marker='x', s=200, linewidths=3, color='red', label='중심')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'K-Means 군집화 (K={n_clusters})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return silhouette_avg

# K=3으로 실루엣 분석
silhouette_analysis(X_cluster, n_clusters=3)

# K=4로 비교
silhouette_analysis(X_cluster, n_clusters=4)
```

### 4.3 외부 지표 (레이블이 있을 때)

```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, homogeneity_completeness_v_measure

# 실제 레이블
true_labels = np.repeat([0, 1, 2], n_samples//3)

# K-means 군집화
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
pred_labels = kmeans.fit_predict(X_cluster)

# 외부 지표
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)
homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, pred_labels)

print("=== 외부 군집 타당성 지표 ===")
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"  - 범위: [-1, 1], 1이 완벽한 일치")
print(f"\nNormalized Mutual Information (NMI): {nmi:.4f}")
print(f"  - 범위: [0, 1], 1이 완벽한 일치")
print(f"\nHomogeneity: {homogeneity:.4f}")
print(f"  - 각 군집이 단일 클래스로 구성되는 정도")
print(f"\nCompleteness: {completeness:.4f}")
print(f"  - 각 클래스가 단일 군집에 할당되는 정도")
print(f"\nV-measure: {v_measure:.4f}")
print(f"  - Homogeneity와 Completeness의 조화평균")
```

---

## 5. 실습 예제

### 5.1 종합 다변량 분석

```python
def comprehensive_multivariate_analysis(X, y, feature_names, target_names):
    """
    종합 다변량 분석 수행
    """
    print("="*60)
    print("종합 다변량 분석")
    print("="*60)

    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. PCA
    print("\n[1] 주성분 분석 (PCA)")
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    print(f"설명된 분산: {pca.explained_variance_ratio_.round(3)}")
    print(f"누적 분산: {np.cumsum(pca.explained_variance_ratio_).round(3)}")

    # 2. LDA
    print("\n[2] 선형 판별 분석 (LDA)")
    lda = LinearDiscriminantAnalysis()
    X_lda = lda.fit_transform(X_scaled, y)
    print(f"LDA 축 수: {X_lda.shape[1]}")
    print(f"설명된 분산: {lda.explained_variance_ratio_.round(3)}")

    # 3. 분류 성능
    print("\n[3] 분류 성능")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )

    models = {
        'LDA': LinearDiscriminantAnalysis(),
        'QDA': QuadraticDiscriminantAnalysis()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"{name}: Train={train_acc:.4f}, Test={test_acc:.4f}")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # PCA
    ax = axes[0, 0]
    for target in np.unique(y):
        mask = y == target
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.7,
                   label=target_names[target])
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.set_title('PCA')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # LDA
    ax = axes[0, 1]
    for target in np.unique(y):
        mask = y == target
        if X_lda.shape[1] >= 2:
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1], alpha=0.7,
                       label=target_names[target])
            ax.set_xlabel('LD1')
            ax.set_ylabel('LD2')
        else:
            ax.scatter(X_lda[mask, 0], np.random.randn(mask.sum())*0.1, alpha=0.7,
                       label=target_names[target])
            ax.set_xlabel('LD1')
            ax.set_ylabel('(jitter)')
    ax.set_title('LDA')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 스크리 플롯
    ax = axes[1, 0]
    ax.bar(range(1, len(pca.explained_variance_ratio_)+1),
           pca.explained_variance_ratio_, alpha=0.7, label='개별')
    ax.plot(range(1, len(pca.explained_variance_ratio_)+1),
            np.cumsum(pca.explained_variance_ratio_), 'ro-', label='누적')
    ax.set_xlabel('주성분')
    ax.set_ylabel('설명된 분산')
    ax.set_title('스크리 플롯')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 로딩
    ax = axes[1, 1]
    loadings_df = pd.DataFrame(
        pca.components_[:2].T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    loadings_df.plot(kind='bar', ax=ax, alpha=0.7)
    ax.set_ylabel('로딩')
    ax.set_title('PC1, PC2 로딩')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

# Wine 데이터로 테스트
comprehensive_multivariate_analysis(wine.data, wine.target,
                                     wine.feature_names, wine.target_names)
```

---

## 6. 연습 문제

### 문제 1: PCA
유방암 데이터셋(load_breast_cancer)에 PCA를 적용하여:
1. 95% 분산을 설명하는데 필요한 주성분 수 결정
2. 처음 2개 주성분으로 시각화
3. PC1에 가장 크게 기여하는 특성 3개 식별

### 문제 2: 요인 분석
6개 변수 데이터셋을 생성하고 (2개의 잠재 요인):
1. 2-요인 모형 적합
2. Varimax 회전 적용
3. 각 요인을 해석

### 문제 3: LDA vs QDA
Wine 데이터셋에서:
1. LDA와 QDA 분류 성능 비교
2. 5-fold 교차검증 수행
3. 결정 경계 시각화 (2D 축소 후)

### 문제 4: 군집 검증
합성 데이터로 K-means 군집화:
1. 엘보우 방법으로 최적 K 결정
2. 실루엣 분석 수행
3. K=2, 3, 4에서 군집 품질 비교

---

## 7. 핵심 요약

### 방법 선택 가이드

| 목적 | 방법 | 특징 |
|------|------|------|
| 차원 축소 (비지도) | PCA | 분산 최대화, 빠름 |
| 구조 발견 | 요인 분석 | 잠재 변수 해석 |
| 차원 축소 (지도) | LDA | 클래스 분리 최대화 |
| 분류 (선형) | LDA | 같은 공분산 가정 |
| 분류 (비선형) | QDA | 다른 공분산 허용 |

### PCA 핵심

- 표준화 후 적용 (변수 스케일 통일)
- 주성분 개수: Kaiser, 분산 기준, 스크리 플롯
- 로딩 해석: 각 변수의 주성분 기여도

### 군집 검증

- 내부 지표: 실루엣, Calinski-Harabasz, Davies-Bouldin
- 외부 지표 (레이블 있을 때): ARI, NMI

### 다음 장 미리보기

13장 **비모수 통계**에서는:
- 비모수 검정이 필요한 상황
- Mann-Whitney U, Wilcoxon, Kruskal-Wallis
- Spearman/Kendall 상관
