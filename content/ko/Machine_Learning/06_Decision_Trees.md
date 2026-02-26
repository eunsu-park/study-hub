# 결정 트리 (Decision Tree)

**이전**: [교차검증과 하이퍼파라미터 튜닝](./05_Cross_Validation_Hyperparameters.md) | **다음**: [앙상블 학습 - 배깅](./07_Ensemble_Bagging.md)

---

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 결정 트리의 구조(루트, 내부 노드, 리프)와 예측이 이루어지는 방식을 설명한다
2. 후보 분할에 대한 엔트로피(entropy), 지니 불순도(Gini impurity), 정보 이득(information gain)을 계산한다
3. CART 알고리즘과 탐욕적 분할 탐색(greedy split-search) 과정을 설명한다
4. 과적합(overfitting)을 제어하기 위해 사전 가지치기(pre-pruning) 하이퍼파라미터(max_depth, min_samples_split, min_samples_leaf)를 적용한다
5. 비용-복잡도 가지치기(cost-complexity pruning, CCP)를 구현하고 교차검증으로 최적 알파(alpha)를 선택한다
6. 학습된 결정 트리에서 특성 중요도(feature importance) 점수를 해석한다
7. 분류 트리(classification tree)와 회귀 트리(regression tree)를 비교하고 각각의 적절한 사용 사례를 식별한다

---

결정 트리는 가장 직관적인 머신러닝 알고리즘 중 하나입니다 — 일련의 예/아니오 질문을 던짐으로써 인간이 의사결정을 하는 방식을 그대로 반영합니다. 이러한 투명성은 이해관계자에게 예측을 설명해야 할 때 매우 유용하며, 랜덤 포레스트(Random Forest)와 그래디언트 부스팅(Gradient Boosting) 같은 강력한 앙상블(ensemble) 방법의 구성 요소로도 활용됩니다.

---

## 1. 결정 트리의 기본 개념

### 1.1 트리 구조

```python
"""
결정 트리 구성 요소:
1. 루트 노드 (Root Node): 첫 번째 분할 지점
2. 내부 노드 (Internal Node): 중간 분할 지점
3. 리프 노드 (Leaf Node): 최종 예측값
4. 분할 (Split): 특성에 따른 데이터 분할
5. 깊이 (Depth): 루트에서 노드까지의 거리

예시: 타이타닉 생존 예측
          [성별]
         /      \
      남성       여성
       |          |
    [나이]      생존
    /    \
  <10   >=10
   |      |
 생존   사망
"""
```

### 1.2 기본 사용법

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.tree import plot_tree, export_text
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# 모델 생성 및 학습
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 예측
y_pred = clf.predict(X_test)
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")

# 트리 구조 출력
print("\n트리 구조:")
print(export_text(clf, feature_names=iris.feature_names))
```

### 1.3 트리 시각화

```python
# 시각화
plt.figure(figsize=(20, 10))
plot_tree(
    clf,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree - Iris Classification')
plt.tight_layout()
plt.show()

# 특성 중요도
print("\n특성 중요도:")
for name, importance in zip(iris.feature_names, clf.feature_importances_):
    print(f"  {name}: {importance:.4f}")
```

---

## 2. 분할 기준 (Split Criteria)

### 2.1 엔트로피 (Entropy)

```python
import numpy as np

def entropy(y):
    """정보 엔트로피(information entropy) 계산.

    엔트로피는 클래스 분포의 무질서/불확실성을 측정함.
    한 클래스만 있는 순수 노드(pure node)는 엔트로피=0.
    클래스가 골고루 섞인 노드는 엔트로피 최대 — 어떤 클래스인지
    전혀 알 수 없으므로 최악의 분할 상태.
    결정 트리는 엔트로피를 가장 많이 감소시키는 분할(정보 이득 최대화)을 선택함.
    """
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)  # 각 클래스에 속하는 샘플 비율
    # log2(p) ≤ 0 이므로 (p ∈ (0,1]) 음수 부호를 붙여 H(y) ≥ 0 으로 만듦 —
    # 엔트로피는 관례상 음수가 아닌 값
    # + 1e-10 은 log(0)을 방지 — 이 노드에 샘플이 없는 클래스(확률=0)가 있으면
    # log(0)=-inf가 되어 계산이 중단됨
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

# 예시
y_pure = [0, 0, 0, 0, 0]  # 순수 노드
y_mixed = [0, 0, 1, 1, 1]  # 혼합 노드
y_balanced = [0, 0, 1, 1]  # 균형 노드

print("엔트로피 예시:")
print(f"  순수 노드: {entropy(y_pure):.4f}")  # 0
print(f"  혼합 노드 [2:3]: {entropy(y_mixed):.4f}")
print(f"  균형 노드 [2:2]: {entropy(y_balanced):.4f}")  # 1 (최대)
```

### 2.2 지니 불순도 (Gini Impurity)

```python
def gini_impurity(y):
    """지니 불순도(Gini impurity) 계산"""
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)  # 각 원소 = 해당 클래스 샘플 비율 (클래스 비율 벡터)
    # 1 - Σp² 는 무작위로 뽑은 두 샘플이 서로 다른 클래스에 속할 확률과 같음 —
    # 순수 노드는 0, 완전히 섞인 노드는 최대 ~0.5
    return 1 - np.sum(probabilities ** 2)

print("\n지니 불순도 예시:")
print(f"  순수 노드: {gini_impurity(y_pure):.4f}")  # 0
print(f"  혼합 노드: {gini_impurity(y_mixed):.4f}")
print(f"  균형 노드: {gini_impurity(y_balanced):.4f}")  # 0.5 (최대)

# 비교: 엔트로피 vs 지니
"""
- Gini: 계산이 빠름, 기본값
- Entropy: 더 균형 잡힌 트리 경향
- 실제로 큰 차이 없음
"""
```

### 2.3 정보 이득 (Information Gain)

```python
def information_gain(parent, left_child, right_child, criterion='gini'):
    """정보 이득 계산"""
    if criterion == 'gini':
        impurity_func = gini_impurity
    else:
        impurity_func = entropy

    # 가중 평균 불순도(weighted impurity): 자식 노드의 불순도를 샘플 수로 가중 평균.
    # 분할이 불균등할 때도 규모를 고려해 분할 품질을 공정하게 측정하기 위함 —
    # 큰 순수 자식 노드는 작은 순수 자식 노드보다 더 가치 있음
    n = len(left_child) + len(right_child)
    n_left, n_right = len(left_child), len(right_child)

    weighted_impurity = (n_left / n) * impurity_func(left_child) + \
                       (n_right / n) * impurity_func(right_child)

    return impurity_func(parent) - weighted_impurity

# 예시: 분할 비교
parent = [0, 0, 0, 1, 1, 1]

# 분할 A: 좋은 분할
left_a = [0, 0, 0]
right_a = [1, 1, 1]

# 분할 B: 나쁜 분할
left_b = [0, 0, 1]
right_b = [0, 1, 1]

print("\n정보 이득 비교:")
print(f"  분할 A (완벽): {information_gain(parent, left_a, right_a):.4f}")
print(f"  분할 B (혼합): {information_gain(parent, left_b, right_b):.4f}")
```

---

## 3. CART 알고리즘

### 3.1 분류 트리 (Classification)

```python
from sklearn.tree import DecisionTreeClassifier

# 다양한 criterion 비교
criteria = ['gini', 'entropy', 'log_loss']

print("분류 트리 - Criterion 비교:")
for criterion in criteria:
    clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  {criterion}: 정확도 = {accuracy:.4f}, 깊이 = {clf.get_depth()}")
```

### 3.2 회귀 트리 (Regression)

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
diabetes = load_diabetes()
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# 회귀 트리 (MSE 기준)
reg = DecisionTreeRegressor(criterion='squared_error', random_state=42)
reg.fit(X_train_r, y_train_r)
y_pred_r = reg.predict(X_test_r)

print("\n회귀 트리 결과:")
print(f"  MSE: {mean_squared_error(y_test_r, y_pred_r):.4f}")
print(f"  R²: {r2_score(y_test_r, y_pred_r):.4f}")

# 다른 criterion
criteria_reg = ['squared_error', 'friedman_mse', 'absolute_error']

print("\n회귀 트리 - Criterion 비교:")
for criterion in criteria_reg:
    reg = DecisionTreeRegressor(criterion=criterion, random_state=42)
    reg.fit(X_train_r, y_train_r)
    y_pred = reg.predict(X_test_r)
    mse = mean_squared_error(y_test_r, y_pred)
    print(f"  {criterion}: MSE = {mse:.4f}")
```

### 3.3 분할 탐색 과정

```python
"""
CART 알고리즘 분할 과정:

1. 모든 특성에 대해:
   - 모든 가능한 분할점 검토
   - 각 분할의 불순도 감소량 계산

2. 최적 분할 선택:
   - 가장 큰 불순도 감소를 주는 (특성, 분할점) 선택

3. 재귀적 분할:
   - 각 자식 노드에 대해 1-2 반복
   - 종료 조건 만족 시 중지

종료 조건:
- 최대 깊이 도달
- 노드 내 샘플 수가 최소 기준 이하
- 순수 노드 도달 (불순도 = 0)
"""

# 분할 과정 시뮬레이션
def find_best_split(X, y, feature_idx):
    """단일 특성에 대한 최적 분할점 찾기"""
    feature = X[:, feature_idx]
    sorted_indices = np.argsort(feature)

    best_gain = -1
    best_threshold = None

    for i in range(1, len(feature)):
        if feature[sorted_indices[i-1]] == feature[sorted_indices[i]]:
            continue

        threshold = (feature[sorted_indices[i-1]] + feature[sorted_indices[i]]) / 2
        left_mask = feature <= threshold

        if np.sum(left_mask) == 0 or np.sum(~left_mask) == 0:
            continue

        gain = information_gain(y, y[left_mask], y[~left_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain

# 테스트
print("\n최적 분할점 탐색:")
for i, name in enumerate(iris.feature_names):
    threshold, gain = find_best_split(iris.data, iris.target, i)
    print(f"  {name}: threshold={threshold:.2f}, gain={gain:.4f}")
```

---

## 4. 가지치기 (Pruning)

### 4.1 사전 가지치기 (Pre-pruning)

```python
# 하이퍼파라미터로 트리 성장 제한
clf_pruned = DecisionTreeClassifier(
    max_depth=3,              # 최대 깊이
    min_samples_split=10,     # 분할에 필요한 최소 샘플 수
    min_samples_leaf=5,       # 리프 노드 최소 샘플 수
    max_features='sqrt',      # 분할 시 고려할 최대 특성 수
    max_leaf_nodes=10,        # 최대 리프 노드 수
    random_state=42
)
clf_pruned.fit(X_train, y_train)

print("사전 가지치기 결과:")
print(f"  깊이: {clf_pruned.get_depth()}")
print(f"  리프 노드 수: {clf_pruned.get_n_leaves()}")
print(f"  정확도: {accuracy_score(y_test, clf_pruned.predict(X_test)):.4f}")
```

### 4.2 사후 가지치기 (Post-pruning) - Cost Complexity Pruning

```python
# CCP (Cost Complexity Pruning)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas
impurities = path.impurities

print("CCP Alpha 경로:")
print(f"  Alpha 값 수: {len(ccp_alphas)}")

# 각 alpha에 대한 트리 생성
clfs = []
for ccp_alpha in ccp_alphas:
    clf_ccp = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    clf_ccp.fit(X_train, y_train)
    clfs.append(clf_ccp)

# 노드 수와 깊이 변화
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
n_leaves = [clf.get_n_leaves() for clf in clfs]
depths = [clf.get_depth() for clf in clfs]

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Alpha vs 정확도
axes[0].plot(ccp_alphas, train_scores, marker='o', label='Train', drawstyle='steps-post')
axes[0].plot(ccp_alphas, test_scores, marker='o', label='Test', drawstyle='steps-post')
axes[0].set_xlabel('Alpha')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Alpha vs Accuracy')
axes[0].legend()

# Alpha vs 리프 노드 수
axes[1].plot(ccp_alphas, n_leaves, marker='o', drawstyle='steps-post')
axes[1].set_xlabel('Alpha')
axes[1].set_ylabel('Number of Leaves')
axes[1].set_title('Alpha vs Number of Leaves')

# Alpha vs 깊이
axes[2].plot(ccp_alphas, depths, marker='o', drawstyle='steps-post')
axes[2].set_xlabel('Alpha')
axes[2].set_ylabel('Depth')
axes[2].set_title('Alpha vs Depth')

plt.tight_layout()
plt.show()

# 최적 alpha 선택 (교차 검증)
from sklearn.model_selection import cross_val_score

cv_scores = []
for ccp_alpha in ccp_alphas:
    clf_ccp = DecisionTreeClassifier(ccp_alpha=ccp_alpha, random_state=42)
    scores = cross_val_score(clf_ccp, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

best_idx = np.argmax(cv_scores)
best_alpha = ccp_alphas[best_idx]
print(f"\n최적 Alpha: {best_alpha:.6f}")
print(f"최적 CV 점수: {cv_scores[best_idx]:.4f}")
```

### 4.3 가지치기 비교

```python
# 가지치기 전/후 비교
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# 가지치기 전
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_train, y_train)

plot_tree(clf_full, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, ax=axes[0], fontsize=8)
axes[0].set_title(f'Full Tree (Depth={clf_full.get_depth()}, Leaves={clf_full.get_n_leaves()})\n'
                  f'Accuracy: {accuracy_score(y_test, clf_full.predict(X_test)):.4f}')

# 가지치기 후
clf_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
clf_pruned.fit(X_train, y_train)

plot_tree(clf_pruned, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, ax=axes[1], fontsize=10)
axes[1].set_title(f'Pruned Tree (Depth={clf_pruned.get_depth()}, Leaves={clf_pruned.get_n_leaves()})\n'
                  f'Accuracy: {accuracy_score(y_test, clf_pruned.predict(X_test)):.4f}')

plt.tight_layout()
plt.show()
```

---

## 5. 결정 경계 시각화

```python
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# 2D 데이터 생성
X_2d, y_2d = make_classification(
    n_samples=200, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# 여러 깊이의 트리 비교
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
depths = [1, 2, 3, 5, 10, None]

for ax, depth in zip(axes.flatten(), depths):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_2d, y_2d)

    # 결정 경계
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, edgecolors='black', cmap='RdYlBu')

    depth_str = depth if depth else 'None'
    ax.set_title(f'Max Depth = {depth_str}\nAccuracy = {clf.score(X_2d, y_2d):.3f}')

plt.tight_layout()
plt.show()
```

---

## 6. 결정 트리의 장단점

### 6.1 장점과 단점

```python
"""
장점:
1. 해석 용이: 시각화하여 의사결정 과정 이해 가능
2. 전처리 최소: 스케일링, 정규화 불필요
3. 비선형 관계: 복잡한 비선형 패턴 학습 가능
4. 다양한 데이터: 수치형, 범주형 모두 처리
5. 빠른 예측: O(log n) 시간 복잡도

단점:
1. 과적합 경향: 깊은 트리는 쉽게 과적합
2. 불안정성: 작은 데이터 변화에 민감
3. 최적화 한계: 전역 최적해 보장 안됨 (Greedy)
4. 외삽 불가: 학습 범위 밖 예측 어려움
5. 편향: 클래스 불균형에 민감
"""
```

### 6.2 불안정성 데모

```python
# 데이터 약간 변경 시 트리 변화
np.random.seed(42)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, ax in enumerate(axes):
    # 약간 다른 랜덤 시드로 데이터 분할
    X_tr, X_te, y_tr, y_te = train_test_split(
        iris.data[:, :2], iris.target, test_size=0.2, random_state=i
    )

    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X_tr, y_tr)

    # 결정 경계
    x_min, x_max = iris.data[:, 0].min() - 0.5, iris.data[:, 0].max() + 0.5
    y_min, y_max = iris.data[:, 1].min() - 0.5, iris.data[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3)
    ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, edgecolors='black')
    ax.set_title(f'Random State = {i}')

plt.suptitle('결정 트리의 불안정성: 데이터 분할에 따른 변화')
plt.tight_layout()
plt.show()
```

---

## 7. 하이퍼파라미터 튜닝

```python
from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Grid Search
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("하이퍼파라미터 튜닝 결과:")
print(f"  최적 파라미터: {grid_search.best_params_}")
print(f"  최적 CV 점수: {grid_search.best_score_:.4f}")
print(f"  테스트 점수: {grid_search.score(X_test, y_test):.4f}")
```

---

## 8. 특성 중요도

```python
# 완전한 트리로 학습
clf = DecisionTreeClassifier(random_state=42)
clf.fit(iris.data, iris.target)

# 특성 중요도
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

# 시각화
plt.figure(figsize=(10, 6))
plt.bar(range(len(importances)), importances[indices])
plt.xticks(range(len(importances)),
           [iris.feature_names[i] for i in indices], rotation=45)
plt.ylabel('Feature Importance')
plt.title('Decision Tree Feature Importance')
plt.tight_layout()
plt.show()

print("\n특성 중요도 순위:")
for i, idx in enumerate(indices):
    print(f"  {i+1}. {iris.feature_names[idx]}: {importances[idx]:.4f}")
```

---

## 연습 문제

### 문제 1: 기본 분류
유방암 데이터로 결정 트리를 학습하고 평가하세요.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 풀이
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=cancer.target_names))
```

### 문제 2: 가지치기
CCP를 사용하여 최적의 alpha를 찾고 가지치기하세요.

```python
# 풀이
from sklearn.model_selection import cross_val_score

# CCP 경로 계산
clf_full = DecisionTreeClassifier(random_state=42)
clf_full.fit(X_train, y_train)
path = clf_full.cost_complexity_pruning_path(X_train, y_train)

# 교차 검증으로 최적 alpha 찾기
best_alpha = 0
best_score = 0
for alpha in path.ccp_alphas[::5]:  # 효율성을 위해 샘플링
    clf = DecisionTreeClassifier(ccp_alpha=alpha, random_state=42)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_alpha = alpha

print(f"최적 Alpha: {best_alpha:.6f}")
print(f"최적 CV 점수: {best_score:.4f}")

clf_pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=42)
clf_pruned.fit(X_train, y_train)
print(f"테스트 정확도: {clf_pruned.score(X_test, y_test):.4f}")
```

### 문제 3: 회귀 트리
당뇨병 데이터로 회귀 트리를 학습하세요.

```python
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# 풀이
reg = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

---

## 요약

| 개념 | 설명 | 용도 |
|------|------|------|
| 엔트로피 | 정보의 불확실성 측정 | 분할 기준 (criterion='entropy') |
| 지니 불순도 | 잘못 분류될 확률 | 분할 기준 (criterion='gini') |
| 정보 이득 | 분할 후 불순도 감소량 | 최적 분할 선택 |
| max_depth | 트리 최대 깊이 | 과적합 방지 |
| min_samples_split | 분할에 필요한 최소 샘플 | 과적합 방지 |
| min_samples_leaf | 리프 노드 최소 샘플 | 과적합 방지 |
| ccp_alpha | 비용-복잡도 가지치기 | 사후 가지치기 |
| feature_importances_ | 특성 중요도 | 특성 선택 |

### 결정 트리 사용 시 체크리스트

1. 과적합 방지를 위해 가지치기 적용
2. 중요 특성 확인으로 해석 가능성 활용
3. 불안정성 해결을 위해 앙상블(Random Forest) 고려
4. 수치형 특성은 스케일링 불필요
5. 범주형 특성은 인코딩 필요 (sklearn 기준)
