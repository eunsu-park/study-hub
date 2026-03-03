# 기호 회귀 (Symbolic Regression)

[← 이전: 23. ML을 위한 A/B 테스팅](23_AB_Testing_for_ML.md) | [다음: 개요 →](00_Overview.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. 기호 회귀가 무엇인지, 매개변수 회귀와 어떻게 다른지 설명한다
2. 후보 수식의 표현 방식으로서 수식 트리(expression tree)를 기술한다
3. 유전 프로그래밍이 수학적 표현 공간을 어떻게 탐색하는지 개요를 설명한다
4. 정확도 대 복잡도의 파레토 프론트를 사용하여 후보 수식을 평가한다
5. PySR과 gplearn을 사용하여 데이터에서 해석 가능한 수식을 발견한다
6. 물리학 및 공학 문제에 기호 회귀를 적용한다
7. 해석 가능성과 일반화 측면에서 기호 회귀와 블랙박스 ML 모델을 비교한다

---

전통적인 회귀 분석은 고정된 수식 형태에 매개변수를 맞춘다: 선형 회귀는 `y = wx + b`를, 다항식 회귀는 `y = Σ wᵢxⁱ`를 가정한다. 사용자가 구조를 선택하면 알고리즘이 숫자를 채운다. 기호 회귀는 이를 뒤집는다: 구조와 매개변수를 동시에 탐색하여 데이터로부터 `y = x₁² + sin(x₂)` 같은 수식을 직접 발견한다. 결과는 수천 개의 매개변수를 가진 블랙박스 모델이 아닌, 간결하고 사람이 읽을 수 있는 공식이다.

---

## 1. 핵심 개념

### 1.1 기호 회귀란?

```python
"""
표준 회귀 vs 기호 회귀

표준 (매개변수) 회귀:
  - 모델 구조를 사용자가 선택: y = w0 + w1*x + w2*x^2
  - 알고리즘이 최적 매개변수를 찾음: w0=1.2, w1=-0.5, w2=3.1
  - 구조 고정, 계수 최적화

기호 회귀:
  - 알고리즘이 구조와 매개변수를 동시에 탐색
  - 입력: 데이터 (X, y)
  - 출력: y = x1^2 + sin(x2)  (자동 발견)
  - 구조 가변, 계수 가변

핵심 장점:
  - 해석 가능한 닫힌 형태의 수식 생성
  - 훈련 분포를 넘어 일반화 가능
  - 발견된 수식이 기저 물리법칙을 드러낼 수 있음
"""
```

### 1.2 수식 트리 (Expression Tree)

모든 수학 표현식은 자연스러운 트리 표현을 가진다:

```python
"""
수식: y = x1^2 + sin(x2)

        [+]
       /   \
     [^]   [sin]
    /   \     |
  [x1]  [2] [x2]

노드:
- 내부 노드: 연산자 (+, -, *, /, ^, sin, cos, exp, log, ...)
- 리프 노드: 변수 (x1, x2, ...) 또는 상수 (2, 3.14, ...)

탐색 공간은 최대 깊이까지의 모든 유효한 수식 트리의 집합이다.
"""

# Expression tree node
class Node:
    def __init__(self, op=None, value=None, left=None, right=None):
        self.op = op          # '+', '-', '*', '/', 'sin', 'cos', ...
        self.value = value    # For leaf nodes: variable name or constant
        self.left = left
        self.right = right

    def evaluate(self, variables):
        """Recursively evaluate the expression tree."""
        if self.value is not None:
            if isinstance(self.value, str):
                return variables[self.value]
            return self.value

        left_val = self.left.evaluate(variables)

        if self.op in ('sin', 'cos', 'exp', 'log', 'sqrt', 'abs'):
            import numpy as np
            return getattr(np, self.op)(left_val)

        right_val = self.right.evaluate(variables)
        if self.op == '+': return left_val + right_val
        if self.op == '-': return left_val - right_val
        if self.op == '*': return left_val * right_val
        if self.op == '/':
            return np.where(np.abs(right_val) > 1e-10,
                            left_val / right_val, 0.0)
        if self.op == '^': return np.power(left_val, right_val)

    def __str__(self):
        if self.value is not None:
            return str(self.value)
        if self.op in ('sin', 'cos', 'exp', 'log', 'sqrt', 'abs'):
            return f"{self.op}({self.left})"
        return f"({self.left} {self.op} {self.right})"

    @property
    def complexity(self):
        """Count total number of nodes."""
        if self.value is not None:
            return 1
        c = 1 + self.left.complexity
        if self.right:
            c += self.right.complexity
        return c
```

---

## 2. 유전 프로그래밍 (Genetic Programming)

### 2.1 알고리즘 개요

```python
"""
기호 회귀를 위한 유전 프로그래밍

1. 초기화: 무작위 수식 트리 집단 생성
2. 평가: 각 트리의 적합도 = f(정확도, 복잡도) 계산
3. 선택: 토너먼트 선택으로 부모 선택
4. 교차: 두 부모 간 서브트리 교환
5. 돌연변이: 자식의 노드를 무작위로 수정
6. 교체: 새 세대가 이전 세대를 대체
7. 반복: 수렴 또는 최대 세대까지

핵심 유전 연산자:

교차 (서브트리 교환):
  부모 A:  [+]           부모 B:  [*]
          / \                    / \
        [x1] [sin]            [x2] [3]
               |
             [x2]

  자식:    [+]           (B의 x2 서브트리가 A의 sin(x2)를 대체)
          / \
        [x1] [x2]

돌연변이 유형:
  - 점 돌연변이: 연산자 변경 (+ → *)
  - 서브트리 돌연변이: 서브트리를 새 무작위 트리로 교체
  - 상수 돌연변이: 수치 상수를 미세 조정
  - 호이스트 돌연변이: 트리를 자신의 서브트리로 교체 (단순화)
"""
```

### 2.2 최소 GP 구현

```python
import numpy as np
import random

BINARY_OPS = ['+', '-', '*', '/']
UNARY_OPS = ['sin', 'cos']
ALL_OPS = BINARY_OPS + UNARY_OPS

def random_tree(variables, max_depth=4, depth=0):
    """Generate a random expression tree."""
    if depth >= max_depth or (depth > 0 and random.random() < 0.3):
        if random.random() < 0.6:
            return Node(value=random.choice(variables))
        else:
            return Node(value=round(random.uniform(-5, 5), 2))

    op = random.choice(ALL_OPS)
    left = random_tree(variables, max_depth, depth + 1)

    if op in UNARY_OPS:
        return Node(op=op, left=left)
    else:
        right = random_tree(variables, max_depth, depth + 1)
        return Node(op=op, left=left, right=right)


def crossover(parent1, parent2):
    """Swap random subtrees between two parents."""
    import copy
    child = copy.deepcopy(parent1)

    def get_nodes(node, parent=None, attr=None):
        result = [(node, parent, attr)]
        if node.left:
            result.extend(get_nodes(node.left, node, 'left'))
        if node.right:
            result.extend(get_nodes(node.right, node, 'right'))
        return result

    nodes1 = get_nodes(child)
    nodes2 = get_nodes(copy.deepcopy(parent2))

    _, p1_parent, p1_attr = random.choice(nodes1[1:]) if len(nodes1) > 1 else nodes1[0]
    donor, _, _ = random.choice(nodes2)

    if p1_parent and p1_attr:
        setattr(p1_parent, p1_attr, donor)

    return child


def mutate(tree, variables, mutation_rate=0.1):
    """Apply point mutation to random nodes."""
    import copy
    tree = copy.deepcopy(tree)

    def _mutate(node):
        if random.random() < mutation_rate:
            if node.value is not None:
                if random.random() < 0.5:
                    node.value = random.choice(variables)
                else:
                    node.value = round(random.uniform(-5, 5), 2)
            elif node.op:
                if node.op in UNARY_OPS:
                    node.op = random.choice(UNARY_OPS)
                else:
                    node.op = random.choice(BINARY_OPS)
        if node.left:
            _mutate(node.left)
        if node.right:
            _mutate(node.right)

    _mutate(tree)
    return tree


def fitness(tree, X, y):
    """RMSE as fitness (lower is better)."""
    try:
        variables = {f'x{i}': X[:, i] for i in range(X.shape[1])}
        y_pred = tree.evaluate(variables)
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            return float('inf')
        return np.sqrt(np.mean((y - y_pred) ** 2))
    except Exception:
        return float('inf')


def symbolic_regression(X, y, pop_size=200, generations=50, max_depth=4):
    """Run genetic programming for symbolic regression."""
    variables = [f'x{i}' for i in range(X.shape[1])]
    population = [random_tree(variables, max_depth) for _ in range(pop_size)]

    best_overall = None
    best_fitness = float('inf')

    for gen in range(generations):
        scores = [(tree, fitness(tree, X, y)) for tree in population]
        scores.sort(key=lambda x: x[1])

        if scores[0][1] < best_fitness:
            best_fitness = scores[0][1]
            best_overall = scores[0][0]

        if gen % 10 == 0:
            print(f"Gen {gen:3d}: best RMSE = {scores[0][1]:.6f}, "
                  f"expr = {scores[0][0]}")

        new_pop = [scores[0][0]]  # Elitism

        while len(new_pop) < pop_size:
            tournament = random.sample(scores, k=5)
            p1 = min(tournament, key=lambda x: x[1])[0]
            tournament = random.sample(scores, k=5)
            p2 = min(tournament, key=lambda x: x[1])[0]

            child = crossover(p1, p2)
            child = mutate(child, variables)

            if child.complexity <= 2 ** (max_depth + 1):
                new_pop.append(child)

        population = new_pop

    return best_overall, best_fitness
```

---

## 3. 파레토 프론트: 정확도 vs 복잡도

### 3.1 다목적 최적화

```python
"""
왜 단순히 오차만 최소화하면 안 되는가?
  → 복잡도 패널티 없이는 GP가 비대한 수식을 생성
  → y = (x + 0.001) * (1/0.001) - x + sin(0) + ... (노이즈 과적합)

파레토 프론트:
  - 플롯: x축 = 복잡도 (노드 수), y축 = 오차 (RMSE)
  - 파레토 최적: 더 단순하면서 동시에 더 정확한 다른 수식이 없음

  오차
  │
  │ ●                          ← 복잡하지만 정확
  │   ●
  │     ●  ● ← 파레토 프론트
  │        ●
  │           ●
  │              ●             ← 단순하지만 부정확
  └──────────────────── 복잡도

파레토 프론트의 "무릎" 지점이 최적 트레이드오프를 제공:
  - 유용할 만큼 충분히 정확
  - 해석 가능할 만큼 충분히 단순
"""

def pareto_front(population, X, y):
    """Extract Pareto-optimal expressions (accuracy vs complexity)."""
    results = []
    for tree in population:
        rmse = fitness(tree, X, y)
        if rmse < float('inf'):
            results.append((tree, rmse, tree.complexity))

    results.sort(key=lambda x: x[2])

    front = []
    best_rmse = float('inf')
    for tree, rmse, comp in results:
        if rmse < best_rmse:
            front.append((tree, rmse, comp))
            best_rmse = rmse

    return front
```

### 3.2 복잡도 측정 방법

| 측정법 | 설명 | 예시 |
|--------|------|------|
| 노드 수 | 수식 트리의 전체 노드 수 | `x + sin(y)` → 4 |
| 트리 깊이 | 최대 깊이 | `x + sin(y)` → 2 |
| 기술 길이 | 수식을 인코딩하는 데 필요한 비트 | MDL 기반 |
| 연산 수 | 연산자 노드의 개수 | `x + sin(y)` → 2 |

---

## 4. 도구: PySR과 gplearn

### 4.1 PySR

```python
"""
PySR (Python Symbolic Regression):
  - Julia의 SymbolicRegression.jl 기반 (고성능)
  - 파레토 프론트 자동 관리
  - 사용자 정의 연산자, 제약조건, 차원 분석 지원
  - pip install pysr
"""
from pysr import PySRRegressor
import numpy as np

# Generate data: y = x0^2 + sin(x1)
np.random.seed(42)
X = np.random.randn(200, 2)
y = X[:, 0]**2 + np.sin(X[:, 1])

# Configure and run
model = PySRRegressor(
    niterations=40,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "square"],
    populations=8,
    population_size=50,
    maxsize=20,           # Max expression complexity
    parsimony=0.0032,     # Complexity penalty
    random_state=42,
)

model.fit(X, y)

# Results: Pareto front of equations
print(model)
# Complexity | Loss       | Equation
# 1          | 1.234      | x0
# 3          | 0.567      | x0^2
# 5          | 0.089      | x0^2 + sin(x1)   ← discovered!

# Best equation
print(f"Best: {model.sympy()}")

# Predict with discovered equation
y_pred = model.predict(X)
```

### 4.2 gplearn

```python
"""
gplearn:
  - 순수 Python, sklearn 호환 API
  - PySR보다 단순하지만 제한적
  - 빠른 실험과 sklearn 파이프라인에 적합
  - pip install gplearn
"""
from gplearn.genetic import SymbolicRegressor

est = SymbolicRegressor(
    population_size=1000,
    generations=20,
    tournament_size=20,
    stopping_criteria=0.01,
    function_set=['add', 'sub', 'mul', 'div', 'sin', 'cos'],
    metric='mse',
    parsimony_coefficient=0.001,
    random_state=42,
    verbose=1,
)

est.fit(X, y)

print(f"Program: {est._program}")
print(f"Fitness: {est._program.fitness_}")

# sklearn pipeline integration
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('sr', SymbolicRegressor(generations=20, random_state=42)),
])
pipe.fit(X, y)
```

---

## 5. 응용

### 5.1 물리 법칙 발견

```python
"""
데이터로부터 물리 법칙 재발견

예시: 뉴턴의 만유인력 법칙
  - 입력: mass1, mass2, distance, measured force
  - 발견: F = G * m1 * m2 / r^2

예시: 케플러의 제3법칙
  - 입력: 공전 주기, 궤도 장반축
  - 발견: T^2 ∝ a^3

예시: 옴의 법칙
  - 입력: 전압, 전류, 저항 측정값
  - 발견: V = I * R

실제 연구 사례:
  - AI Feynman (Udrescu & Tegmark, 2020): 100개 물리 방정식 재발견
  - SINDy (Brunton et al., 2016): 지배 미분방정식 발견
  - PDE-Net: 시뮬레이션 데이터로부터 편미분방정식 학습
"""

# Toy example: discover F = m * a
np.random.seed(42)
n = 500
mass = np.random.uniform(1, 100, n)
acceleration = np.random.uniform(0.1, 10, n)
force = mass * acceleration + np.random.normal(0, 0.5, n)  # Noise

X_physics = np.column_stack([mass, acceleration])

# PySR로 실행
# model = PySRRegressor(
#     niterations=40,
#     binary_operators=["+", "-", "*", "/"],
#     maxsize=10,
# )
# model.fit(X_physics, force)
# Expected output: x0 * x1  (즉, mass * acceleration)
```

### 5.2 기호 회귀를 활용한 특성 공학

```python
"""
기호 회귀를 사용하여 하류 ML을 위한 새로운 특성을 발견:

1. (X, y)에 SR 실행 → 상위 k개 파레토 최적 수식 획득
2. 각 수식을 X에 대해 평가 → 새로운 특성 열
3. 원본 X에 추가 → 강화된 특성 행렬
4. 강화된 특성으로 표준 ML 모델 훈련

기호 회귀의 해석 가능성과 그래디언트 부스팅의 성능을 결합한다.
"""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

X_original = np.random.randn(500, 3)
y = X_original[:, 0]**2 + np.sin(X_original[:, 1]) * X_original[:, 2]

# SR이 발견한 수식이라고 가정:
sr_feature_1 = X_original[:, 0]**2
sr_feature_2 = np.sin(X_original[:, 1])

X_enhanced = np.column_stack([X_original, sr_feature_1, sr_feature_2])

gb_original = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_enhanced = GradientBoostingRegressor(n_estimators=100, random_state=42)

score_orig = cross_val_score(gb_original, X_original, y, cv=5,
                             scoring='neg_mean_squared_error')
score_enh = cross_val_score(gb_enhanced, X_enhanced, y, cv=5,
                            scoring='neg_mean_squared_error')

print(f"Original features MSE:  {-score_orig.mean():.4f}")
print(f"Enhanced features MSE:  {-score_enh.mean():.4f}")
```

---

## 6. 블랙박스 모델과의 비교

### 6.1 기호 회귀를 사용할 때

| 기준 | 기호 회귀 | 블랙박스 ML |
|------|----------|-------------|
| 해석 가능성 | 높음 (닫힌 형태 수식) | 낮음 (SHAP/LIME으로 사후 설명) |
| 외삽 | 종종 양호 (참 법칙 발견 시) | 불량 (보간만 가능) |
| 고차원 데이터 | 약함 (>10 특성은 어려움) | 강함 |
| 대규모 데이터셋 | 느림 (GP 탐색이 비용 큼) | 빠름 (그래디언트 기반) |
| 노이즈 내성 | 보통 | 높음 |
| 도메인 지식 | 연산자 제약 인코딩 가능 | 특성 공학 |
| 출력 | 수학 공식 | 예측 함수 |

### 6.2 한계

```python
"""
기호 회귀의 한계:

1. 확장성: GP 탐색은 O(pop_size * generations * data_size)
   - 실용적 한계: ~10개 입력 특성, ~10k 샘플
   - 더 큰 문제에는 SR을 특성 발견에 사용 후 ML 적용

2. 탐색 공간 폭발:
   - 이진 연산 4개, 단항 연산 2개, 변수 5개, 깊이 5:
   - 가능한 트리 > 10^10
   - 전역 최적 발견 보장 없음

3. 과적합:
   - 복잡한 수식이 노이즈를 기억할 수 있음
   - 파레토 프론트 / 절약성 압력이 필수

4. 수치 불안정성:
   - 0에 가까운 값으로 나누기, exp 오버플로우
   - 보호된 연산자 필요: div(a,b) = a/b if |b|>ε else 0

5. 상수 최적화:
   - GP는 수치 상수 조정에 취약
   - 현대 도구(PySR)는 상수에 경사 하강법 사용
"""
```

---

## 7. 최신 발전

### 7.1 신경망 안내 기호 회귀

```python
"""
신경망과 기호 탐색을 결합한 하이브리드 접근법:

1. AI Feynman (2020):
   - 신경망이 대칭성과 분리 가능성을 식별
   - 기호 회귀 전에 탐색 공간을 축소
   - 파인만 강의의 100개 물리 방정식 재발견

2. Deep Symbolic Regression (Petersen et al., 2021):
   - RNN이 토큰 단위로 수식 트리 생성
   - 강화학습으로 훈련 (보상 = 적합도)
   - 일부 문제 클래스에서 GP보다 빠름

3. Symbolic GPT / E2E Transformers (Kamienny et al., 2022):
   - (데이터, 수식) 쌍으로 트랜스포머 훈련
   - 새 데이터가 주어지면 한 번의 순전파로 수식 예측
   - 반복 탐색보다 수 자릿수 빠름

4. SymbolicRegression.jl (Cranmer, 2023):
   - PySR의 백엔드
   - 다중 집단 진화 탐색
   - 그래디언트 최적화 상수
   - SRBench 벤치마크 최고 성능
"""
```

### 7.2 SINDy: 비선형 동역학의 희소 식별

```python
"""
SINDy (Brunton et al., 2016):
  - 지배 미분방정식 발견: dx/dt = f(x)
  - 후보 항 라이브러리 구축: [1, x, x^2, sin(x), ...]
  - 희소 회귀(LASSO)로 활성 항 선택
  - GP 기반이 아닌: 트리 탐색 대신 희소성 활용

응용:
  - 유체역학: 나비에-스토크스 근사
  - 생물학 시스템: 개체군 데이터로부터 로트카-볼테라
  - 제어 시스템: 데이터 기반 모델 발견
"""
import numpy as np

def sindy(X, X_dot, candidate_library, threshold=0.1):
    """
    Sparse Identification of Nonlinear Dynamics.

    Args:
        X: state measurements (n_samples, n_features)
        X_dot: time derivatives (n_samples, n_features)
        candidate_library: function that builds library from X
        threshold: sparsity threshold for sequential thresholding

    Returns:
        coefficients: sparse coefficient matrix
    """
    Theta = candidate_library(X)

    n_targets = X_dot.shape[1]
    Xi = np.linalg.lstsq(Theta, X_dot, rcond=None)[0]

    for _ in range(10):
        for j in range(n_targets):
            small = np.abs(Xi[:, j]) < threshold
            Xi[small, j] = 0
            big = ~small
            if np.any(big):
                Xi[big, j] = np.linalg.lstsq(
                    Theta[:, big], X_dot[:, j], rcond=None
                )[0]

    return Xi
```

---

## 8. 요약

| 개념 | 설명 |
|------|------|
| 기호 회귀 | 데이터에 맞는 수학적 표현식을 탐색 |
| 수식 트리 | 수식의 트리 표현 (연산자 + 피연산자) |
| 유전 프로그래밍 | 진화적 탐색: 선택, 교차, 돌연변이 |
| 파레토 프론트 | 정확도와 복잡도 간의 트레이드오프 곡선 |
| PySR | 고성능 기호 회귀 (Julia 백엔드) |
| gplearn | sklearn 호환 GP 기호 회귀 |
| SINDy | 미분방정식 발견을 위한 희소 회귀 |
| 신경망 안내 SR | 하이브리드 신경망 + 기호 접근법 (AI Feynman, DSR) |

### 기호 회귀 vs 관련 기법

```
회귀 패밀리
    │
    ├── 매개변수적: 구조 고정, 계수 최적화
    │       ├── 선형 / 다항식 / 로지스틱
    │       └── 신경망 (고정 아키텍처)
    │
    ├── 비매개변수적: 고정 구조 없음, 데이터 기반
    │       ├── k-NN 회귀
    │       ├── 커널 방법
    │       └── 가우시안 프로세스
    │
    └── 기호적: 구조와 계수를 동시에 탐색
            ├── 유전 프로그래밍 (GP)
            ├── SINDy (함수 라이브러리에 대한 희소 회귀)
            └── 신경망 안내 SR (트랜스포머/RL 기반)
```

---

## 참고 자료

- Cranmer, M. (2023). "Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl." *arXiv:2305.01582*
- Udrescu, S. M. & Tegmark, M. (2020). "AI Feynman: A Physics-Inspired Method for Symbolic Regression." *Science Advances*
- Brunton, S. L. et al. (2016). "Discovering Governing Equations from Data by Sparse Identification of Nonlinear Dynamical Systems." *PNAS*
- Petersen, B. K. et al. (2021). "Deep Symbolic Regression." *ICLR 2021*
- Kamienny, P. et al. (2022). "End-to-End Symbolic Regression with Transformers." *NeurIPS 2022*
- [PySR Documentation](https://astroautomata.com/PySR/)
- [gplearn Documentation](https://gplearn.readthedocs.io/)
- [SRBench Benchmark](https://cavalab.org/srbench/)
