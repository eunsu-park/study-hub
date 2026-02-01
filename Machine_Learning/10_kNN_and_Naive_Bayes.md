# k-최근접 이웃(kNN)과 나이브 베이즈

## 개요

k-최근접 이웃(kNN)은 거리 기반 분류 알고리즘이고, 나이브 베이즈는 확률 기반 분류 알고리즘입니다. 두 알고리즘 모두 간단하면서도 효과적인 분류기입니다.

---

## 1. k-최근접 이웃 (k-Nearest Neighbors)

### 1.1 kNN의 기본 개념

```python
"""
kNN 알고리즘:

1. 새로운 데이터 포인트가 들어오면
2. 학습 데이터에서 가장 가까운 k개의 이웃을 찾음
3. k개 이웃의 다수결(분류) 또는 평균(회귀)으로 예측

특징:
- 게으른 학습 (Lazy Learning): 학습 시 모델 생성 안함
- 인스턴스 기반 학습: 모든 학습 데이터 저장
- 비모수적 방법: 데이터 분포 가정 없음
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
```

### 1.2 kNN 시각화

```python
from sklearn.datasets import make_classification

# 2D 데이터 생성
X, y = make_classification(
    n_samples=100, n_features=2, n_redundant=0,
    n_informative=2, n_clusters_per_class=1, random_state=42
)

# kNN 시각화
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
k_values = [1, 5, 15]

for ax, k in zip(axes, k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # 결정 경계
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
    ax.set_title(f'k = {k}\nAccuracy = {knn.score(X, y):.3f}')

plt.tight_layout()
plt.show()
```

### 1.3 기본 사용법

```python
# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# kNN 분류기
knn = KNeighborsClassifier(
    n_neighbors=5,           # k값
    weights='uniform',       # 가중치: 'uniform' 또는 'distance'
    algorithm='auto',        # 알고리즘: 'auto', 'ball_tree', 'kd_tree', 'brute'
    metric='minkowski',      # 거리 측정: 'euclidean', 'manhattan', 'minkowski'
    p=2                      # minkowski p값 (2=euclidean, 1=manhattan)
)

knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("kNN 분류 결과:")
print(f"  정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
```

---

## 2. 거리 측정 방법

### 2.1 주요 거리 메트릭

```python
from scipy.spatial.distance import euclidean, cityblock, minkowski, chebyshev

"""
거리 측정 방법:

1. 유클리드 거리 (Euclidean, L2):
   d = sqrt(Σ(x_i - y_i)²)
   - 가장 일반적으로 사용

2. 맨해튼 거리 (Manhattan, L1):
   d = Σ|x_i - y_i|
   - 직각 좌표계에서의 거리
   - 고차원에서 더 효과적일 수 있음

3. 민코프스키 거리 (Minkowski):
   d = (Σ|x_i - y_i|^p)^(1/p)
   - p=1: 맨해튼, p=2: 유클리드

4. 체비셰프 거리 (Chebyshev, L∞):
   d = max(|x_i - y_i|)
   - 모든 차원에서 최대 차이
"""

# 예시
point1 = np.array([1, 2, 3])
point2 = np.array([4, 5, 6])

print("거리 측정 예시:")
print(f"  유클리드: {euclidean(point1, point2):.4f}")
print(f"  맨해튼: {cityblock(point1, point2):.4f}")
print(f"  민코프스키 (p=3): {minkowski(point1, point2, p=3):.4f}")
print(f"  체비셰프: {chebyshev(point1, point2):.4f}")
```

### 2.2 거리 메트릭 비교

```python
# 거리 메트릭별 성능 비교
metrics = ['euclidean', 'manhattan', 'chebyshev']

print("거리 메트릭별 성능:")
for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"  {metric}: {acc:.4f}")
```

---

## 3. k값 선택

### 3.1 k값에 따른 성능 변화

```python
# k값에 따른 성능 변화
k_range = range(1, 31)
train_scores = []
test_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_scores.append(knn.score(X_train, y_train))
    test_scores.append(knn.score(X_test, y_test))

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(k_range, train_scores, 'o-', label='Train')
plt.plot(k_range, test_scores, 's-', label='Test')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('kNN: k vs Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(k_range[::2])
plt.show()

# 최적 k 찾기
best_k = k_range[np.argmax(test_scores)]
print(f"최적 k: {best_k}")
print(f"최고 테스트 정확도: {max(test_scores):.4f}")
```

### 3.2 교차 검증으로 k 선택

```python
from sklearn.model_selection import cross_val_score

# 교차 검증으로 k 선택
k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# 시각화
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, 'o-')
plt.xlabel('k')
plt.ylabel('Cross-Validation Accuracy')
plt.title('kNN: k Selection with Cross-Validation')
plt.grid(True, alpha=0.3)
plt.xticks(k_range[::2])
plt.show()

best_k = k_range[np.argmax(cv_scores)]
print(f"교차 검증 최적 k: {best_k}")
print(f"최고 CV 정확도: {max(cv_scores):.4f}")
```

---

## 4. 가중 kNN

```python
"""
가중 kNN (Weighted kNN):

1. uniform: 모든 이웃에 동일한 가중치
   - 다수결 투표

2. distance: 거리에 반비례하는 가중치
   - 가까운 이웃에 더 큰 가중치
   - weight = 1 / distance
"""

# 가중치 비교
weights = ['uniform', 'distance']

print("가중치 방식 비교:")
for weight in weights:
    knn = KNeighborsClassifier(n_neighbors=5, weights=weight)
    knn.fit(X_train, y_train)
    acc = knn.score(X_test, y_test)
    print(f"  {weight}: {acc:.4f}")

# 거리 가중 kNN 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, weight in zip(axes, weights):
    knn = KNeighborsClassifier(n_neighbors=15, weights=weight)
    knn.fit(X[:, :2], y)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='black')
    ax.set_title(f'weights = {weight}')

plt.tight_layout()
plt.show()
```

---

## 5. kNN 회귀

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, r2_score

# 데이터 로드
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# 스케일링 (kNN은 거리 기반이므로 필수)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# kNN 회귀
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train_scaled, y_train)
y_pred = knn_reg.predict(X_test_scaled)

print("kNN 회귀 결과:")
print(f"  MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"  R²: {r2_score(y_test, y_pred):.4f}")
```

---

## 6. kNN의 장단점과 최적화

```python
"""
장점:
1. 간단하고 직관적
2. 학습 시간 없음 (게으른 학습)
3. 비모수적: 데이터 분포 가정 불필요
4. 다중 클래스 자연스럽게 처리

단점:
1. 예측 시 느림: O(n*d)
2. 메모리 많이 사용: 모든 데이터 저장
3. 차원의 저주: 고차원에서 성능 저하
4. 스케일링 필수
5. 최적 k 선택 필요

최적화 방법:
1. KD-Tree, Ball-Tree 사용
2. 차원 축소 (PCA 등)
3. 특성 선택
"""

# 알고리즘 비교
from time import time

algorithms = ['brute', 'kd_tree', 'ball_tree']

print("알고리즘별 시간 비교:")
for algo in algorithms:
    knn = KNeighborsClassifier(n_neighbors=5, algorithm=algo)

    # 학습 시간
    start = time()
    knn.fit(X_train, y_train)
    fit_time = time() - start

    # 예측 시간
    start = time()
    knn.predict(X_test)
    pred_time = time() - start

    print(f"  {algo}: fit={fit_time:.4f}s, predict={pred_time:.4f}s")
```

---

## 7. 나이브 베이즈 (Naive Bayes)

### 7.1 베이즈 정리

```python
"""
베이즈 정리:
P(y|X) = P(X|y) * P(y) / P(X)

- P(y|X): 사후 확률 (posterior) - 특성이 주어졌을 때 클래스 확률
- P(X|y): 우도 (likelihood) - 클래스가 주어졌을 때 특성 확률
- P(y): 사전 확률 (prior) - 클래스의 기본 확률
- P(X): 증거 (evidence) - 특성의 확률

나이브 가정 (Naive Assumption):
- 모든 특성이 서로 독립적이라고 가정
- P(X|y) = P(x1|y) * P(x2|y) * ... * P(xn|y)

분류:
y_pred = argmax_y P(y) * Π P(xi|y)
"""

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
```

### 7.2 가우시안 나이브 베이즈

```python
"""
가우시안 나이브 베이즈:
- 연속형 특성에 사용
- 각 특성이 가우시안(정규) 분포를 따른다고 가정
- P(xi|y) = N(xi; μ_y, σ_y)
"""

# 가우시안 NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("가우시안 나이브 베이즈 결과:")
print(f"  정확도: {accuracy_score(y_test, y_pred):.4f}")

# 학습된 파라미터 확인
print(f"\n클래스 사전 확률: {gnb.class_prior_}")
print(f"클래스별 평균 (처음 2개 특성):\n{gnb.theta_[:, :2]}")
print(f"클래스별 분산 (처음 2개 특성):\n{gnb.var_[:, :2]}")
```

### 7.3 확률 예측

```python
# 확률 예측
y_proba = gnb.predict_proba(X_test[:5])

print("확률 예측 (처음 5개):")
print(f"클래스: {gnb.classes_}")
print(y_proba)
print(f"\n예측 클래스: {gnb.predict(X_test[:5])}")
print(f"실제 클래스: {y_test[:5]}")
```

### 7.4 다항 나이브 베이즈 (텍스트 분류)

```python
"""
다항 나이브 베이즈:
- 이산형/카운트 특성에 사용
- 텍스트 분류에 주로 사용 (단어 빈도)
- P(xi|y) = (N_yi + α) / (N_y + αn)
  - N_yi: 클래스 y에서 특성 i의 카운트
  - α: smoothing 파라미터 (Laplace smoothing)
"""

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 뉴스 데이터 로드 (간단한 예시)
categories = ['sci.space', 'rec.sport.baseball', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(
    subset='train',
    categories=categories,
    remove=('headers', 'footers', 'quotes')
)

# 텍스트 벡터화
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
X_news = vectorizer.fit_transform(newsgroups.data)
y_news = newsgroups.target

# 학습/테스트 분할
X_train_news, X_test_news, y_train_news, y_test_news = train_test_split(
    X_news, y_news, test_size=0.2, random_state=42
)

# 다항 나이브 베이즈
mnb = MultinomialNB(alpha=1.0)  # alpha: Laplace smoothing
mnb.fit(X_train_news, y_train_news)

print("다항 나이브 베이즈 (텍스트 분류) 결과:")
print(f"  정확도: {mnb.score(X_test_news, y_test_news):.4f}")

# 각 클래스의 가장 중요한 단어
feature_names = vectorizer.get_feature_names_out()
print("\n각 클래스별 상위 5개 단어:")
for i, category in enumerate(categories):
    top_indices = mnb.feature_log_prob_[i].argsort()[-5:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    print(f"  {category}: {', '.join(top_words)}")
```

### 7.5 베르누이 나이브 베이즈

```python
"""
베르누이 나이브 베이즈:
- 이진 특성에 사용
- 특성의 존재 여부 (0/1)
- 텍스트에서 단어 존재 여부로 사용
"""

# 이진 벡터화
binary_vectorizer = CountVectorizer(max_features=5000, binary=True, stop_words='english')
X_binary = binary_vectorizer.fit_transform(newsgroups.data)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_news, test_size=0.2, random_state=42
)

# 베르누이 나이브 베이즈
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train_bin, y_train_bin)

print("베르누이 나이브 베이즈 결과:")
print(f"  정확도: {bnb.score(X_test_bin, y_test_bin):.4f}")
```

---

## 8. 나이브 베이즈 비교

```python
from sklearn.datasets import load_digits

# 숫자 이미지 데이터
digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(
    digits.data, digits.target, test_size=0.2, random_state=42
)

# 세 가지 나이브 베이즈 비교
models = {
    'Gaussian NB': GaussianNB(),
    'Multinomial NB': MultinomialNB(),
    'Bernoulli NB': BernoulliNB()
}

print("나이브 베이즈 모델 비교:")
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"  {name}: {acc:.4f}")
```

---

## 9. 나이브 베이즈의 장단점

```python
"""
장점:
1. 매우 빠름: 학습 O(n*d), 예측 O(d)
2. 적은 데이터로도 잘 작동
3. 고차원 데이터에 효과적
4. 확률 출력 제공
5. 온라인 학습 가능 (partial_fit)

단점:
1. 나이브 가정: 특성 독립성 가정이 현실에서 위반
2. 상관관계 있는 특성에 약함
3. 연속형 특성: 가우시안 가정이 항상 맞지 않음
4. Zero frequency 문제: smoothing 필요

언제 사용:
- 텍스트 분류 (스팸 필터, 감성 분석)
- 고차원, 적은 데이터
- 빠른 학습/예측 필요시
- 기준선(baseline) 모델
"""
```

---

## 10. 온라인 학습 (Incremental Learning)

```python
# 온라인 학습 (partial_fit)
gnb = GaussianNB()

# 배치 학습 시뮬레이션
batch_size = 50
n_batches = len(X_train) // batch_size

for i in range(n_batches):
    start = i * batch_size
    end = start + batch_size
    X_batch = X_train[start:end]
    y_batch = y_train[start:end]

    # 첫 배치에서 클래스 정의
    if i == 0:
        gnb.partial_fit(X_batch, y_batch, classes=np.unique(y_train))
    else:
        gnb.partial_fit(X_batch, y_batch)

print("온라인 학습 결과:")
print(f"  정확도: {gnb.score(X_test, y_test):.4f}")
```

---

## 11. kNN vs 나이브 베이즈 비교

```python
from sklearn.datasets import load_breast_cancer

# 데이터
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 비교
models = {
    'kNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'kNN (weighted)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'Gaussian NB': GaussianNB()
}

print("kNN vs 나이브 베이즈 비교:")
print("-" * 50)

for name, model in models.items():
    if 'kNN' in name:
        model.fit(X_train_scaled, y_train)
        acc = model.score(X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
    print(f"  {name}: {acc:.4f}")
```

---

## 연습 문제

### 문제 1: 최적 k 찾기
교차 검증으로 Iris 데이터에 최적인 k를 찾으세요.

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

iris = load_iris()
X, y = iris.data, iris.target

# 풀이
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, cv=5)
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"최적 k: {best_k}")
print(f"최고 CV 정확도: {max(cv_scores):.4f}")
```

### 문제 2: 나이브 베이즈 텍스트 분류
간단한 텍스트 분류를 구현하세요.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# 간단한 텍스트 데이터
texts = [
    "I love this movie", "Great film", "Excellent acting",
    "Terrible movie", "Bad film", "Worst movie ever"
]
labels = [1, 1, 1, 0, 0, 0]  # 1: positive, 0: negative

# 풀이
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

mnb = MultinomialNB()
mnb.fit(X, labels)

# 새로운 텍스트 분류
new_texts = ["This is a great movie", "I hate this film"]
X_new = vectorizer.transform(new_texts)
predictions = mnb.predict(X_new)

for text, pred in zip(new_texts, predictions):
    sentiment = "Positive" if pred == 1 else "Negative"
    print(f"'{text}' -> {sentiment}")
```

### 문제 3: 거리 가중 kNN
거리 가중치를 사용하여 kNN 회귀를 수행하세요.

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler

diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.2, random_state=42
)

# 풀이
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train_scaled, y_train)

from sklearn.metrics import r2_score, mean_squared_error
y_pred = knn_reg.predict(X_test_scaled)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
```

---

## 요약

### kNN 요약

| 파라미터 | 설명 | 권장 |
|----------|------|------|
| n_neighbors | 이웃 수 | 교차 검증으로 선택 |
| weights | 가중치 방식 | 'distance' 추천 |
| metric | 거리 측정 | 'euclidean' 기본 |
| algorithm | 탐색 알고리즘 | 'auto' |

### 나이브 베이즈 요약

| 종류 | 특성 타입 | 용도 |
|------|-----------|------|
| GaussianNB | 연속형 (정규 분포) | 일반 분류 |
| MultinomialNB | 카운트/빈도 | 텍스트 분류 |
| BernoulliNB | 이진 (0/1) | 이진 특성 |

### 비교

| 특성 | kNN | 나이브 베이즈 |
|------|-----|---------------|
| 학습 시간 | O(1) | O(n*d) |
| 예측 시간 | O(n*d) | O(d) |
| 메모리 | 높음 | 낮음 |
| 스케일링 | 필수 | 불필요 |
| 고차원 | 약함 | 강함 |
| 해석성 | 직관적 | 확률 기반 |
