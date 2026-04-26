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

## 이론과 원리

기호 회귀는 이 커리큘럼의 다른 모든 알고리즘과 근본적으로 다릅니다: 고정된 식의 매개변수가 아니라 *식의 공간*을 탐색합니다. 두 지배적 접근 — 유전자 프로그래밍과 함수 라이브러리에 대한 희소 선형 회귀(SINDy) — 은 완전히 다른 가정과 트레이드오프로 이 탐색을 완전히 다른 방식으로 풉니다.

### A. 탐색 공간: 표현 트리

식은 트리로 표현 가능:

```
y = x₁² + sin(x₂)

       +
      / \
     ²   sin
     |    |
    x₁   x₂
```

내부 노드는 *연산자*(`+`, `-`, `*`, `/`, `sin`, `exp`, `log`, ...); 리프는 *변수*(`x₁`, `x₂`, ...) 또는 *상수*(`1.0`, `π`, ...). 탐색 공간은 어떤 최대 깊이까지의 모든 유효한 표현 트리 — 지수적으로 크지만 이산.

이는 매개변수 최적화와 근본적으로 다른 문제. 선형 회귀는 `ℝ^p`에서 매끄럽고 볼록한 손실 표면을 가짐; 기호 회귀는 유용한 그래디언트가 없는 조합 탐색 공간을 가짐. 다른 알고리즘 도구가 필요.

### B. 유전자 프로그래밍: 탐색으로서의 진화

**유전자 프로그래밍(GP)**은 표현 트리를 다윈주의 진화 과정의 "개체"로 다룸:

```
1. 무작위 표현 트리의 인구 초기화.
2. 각 트리의 적합도 평가: f(tree) = -error_on_data + λ · complexity_penalty
3. 적합도로 트리를 확률적으로 선택(토너먼트 또는 룰렛).
4. 선택된 트리에 유전 연산자 적용:
     - 교차(Crossover): 두 부모 사이의 부분트리 교환
     - 변이(Mutation): 무작위 부분트리를 새 무작위 부분트리로 교체
     - 재생산(Reproduction): 변경 없이 복사
5. 옛 인구를 자손으로 교체.
6. 수렴 또는 예산 소진까지 단계 2부터 반복.
```

적합도 함수가 알고리즘의 심장. 두 항이 정확도와 간결성 사이의 **파레토 트레이드오프**를 인코딩:

```
fitness = α · (-MSE)  -  β · complexity(tree)
```

복잡도 페널티 없이 GP는 과적합 — 학습 데이터를 외우는 거대한 트리를 찾음. 올바른 `β`로, GP가 *파레토 프론티어*를 찾음: 각 복잡도 수준에서 최선인 트리 집합. 사용자가 나중에 트레이드오프를 선택.

### B.1 GP 강점과 약점

강점:
- 식 형태에 대한 가정 없음(선형, 다항, 삼각, 혼합 — 모두 같은 탐색에).
- 출력이 해석 가능한 닫힌 형태 표현.
- 데이터에서 알려진 물리 법칙을 재발견 가능(케플러 법칙, F = ma).

약점:
- 계산적으로 비쌈(인구 × 세대 × 평가).
- 확률적 — 다른 실행이 다른 식 찾을 수 있음.
- 하이퍼파라미터 많음(인구 크기, 변이율, 최대 깊이, 적합도 가중치).
- 탐색 공간의 국소 최적에 빠질 수 있음.

현대 구현(PySR, gplearn)은 정교한 트릭 — 다중 인구 섬, 적응 변이율, 단순화 규칙 — 을 포함하여 이를 완화.

### C. SINDy: 라이브러리에 대한 희소 회귀

**SINDy**(Sparse Identification of Nonlinear Dynamics, Brunton et al., 2016)는 완전히 다른 접근. 표현 트리를 탐색하는 대신, 후보 함수의 *라이브러리* `Θ(x)`를 만듭니다:

```
Θ(x) = [1, x₁, x₂, x₁², x₁·x₂, x₂², sin(x₁), cos(x₁), exp(x₂), ...]
```

그다음 **희소성 제약**을 가진 선형 회귀를 풉니다:

```
y = Θ(x) · ξ           ‖ξ‖_0가 작아야 함
```

`ξ`의 대부분 계수가 0으로 강제, 따라서 발견된 식은 라이브러리 항 몇 개만 사용. 희소성이 해석성을 줌 — 100항 다항 회귀는 해석 불가; 0이 아닌 항이 3개인 것은 가능.

최적화는 보통 **STLSQ**(Sequentially Thresholded Least Squares) 사용:

```
ξ = y = Θ ξ의 최소제곱 해
루프:
    |ξ_i| < threshold인 ξ의 항목을 0으로 설정
    남은 항목에 대해 최소제곱 재해
수렴까지
```

이는 L0 페널티 회귀의 이산 근사, 정확한 L0보다 빠르고 보통 L1(Lasso)이 주는 것보다 더 희소.

### C.1 SINDy 강점과 약점

강점:
- 빠름: 진화 탐색 대신 단일 선형 회귀.
- 결정적.
- 라이브러리가 시간 도함수, 다항 비선형성, 삼각 강제를 포함할 수 있는 동역학 시스템(`dx/dt = f(x)`)에 우수.
- 희소성 임계값이 유일한 결정적 하이퍼파라미터.

약점:
- 라이브러리가 미리 명시되어야 함; 올바른 항이 라이브러리에 없으면 SINDy가 찾을 수 없음.
- 라이브러리 크기가 입력 차원과 항 복잡도에 조합적으로 자람.
- 진짜 식이 선택된 라이브러리에 *정확히* 희소할 때 가장 잘 작동; 신규 비선형성에는 어려움.

SINDy와 GP는 보완적: 가능한 항에 대한 도메인 지식이 있을 때 SINDy, 그렇지 않을 때 GP.

### D. 파레토 프론티어: 정확도 vs 복잡도

두 방법 모두 파레토 프론티어를 따라 후보 식의 가족을 생성:

```
        높은 복잡도
            |
   오차     |  ●         ← 복잡, 정확
            |    ●
            |      ●
            |        ●   ← 단순, 덜 정확
            |          ●
            +----------- 복잡도
```

"최선" 식은 무엇을 가치 있게 여기는지에 의존. 물리 발견의 경우, 보통 적절히 적합하는 가장 단순한 식을 원함 — 오컴의 면도날. 예측의 경우, 복잡도 예산 내 가장 정확한 것을 골라도 됨. PySR이 전체 프론티어를 반환; 사용자가 선택.

### E. 기호 회귀가 이기는 곳

기호 회귀는 표 형식 ML의 대체가 아닙니다. 그래디언트 부스팅 트리가 여전히 불투명하지만 정확한 예측에 지배적. 기호 회귀는 다음일 때 이김:

- **발견 가능한 구조 존재**: 기저 과정이 정말 컴팩트한 식에 의해 지배됨(물리, 화학, 생물, 공학).
- **해석성이 단단한 제약**: 규제, 과학 출판, 임베디드 시스템 배포.
- **외삽이 중요**: 닫힌 형태 식은 학습 분포 너머로 외삽 가능; 트리 모델은 불가능.
- **컴팩트 배포가 중요**: 진화된 식은 바이트; 트리 앙상블은 메가바이트.

이 중 *어느 것도* 성립하지 않을 때, 그래디언트 부스팅이 탐색 비용의 일부로 기호 회귀를 더 잘 예측할 것. 기호 회귀는 특수 도구이지 범용 기본이 아님.

### F. 검증: 기호 회귀도 같은 통계적 함정을 가짐

다른 탐색 알고리즘에도 불구하고, 기호 회귀는 여전히 ML이며 같은 평가 규율을 따름:

- **학습/테스트 분할**: 어떤 모델에든 그렇듯 필수.
- **교차검증**: 각 적합이 비싸 더 어렵지만 가능(작은 `K`, 폴드 병렬화).
- **다중 비교 효과 주의**: 많은 식을 탐색하고 최선만 보고하는 것은 A/B 테스트의 엿보기와 정확히 같은 통계적 죄. 보고된 테스트 오차가 위쪽으로 편향.
- **파레토 프론티어용 보류된 검증셋**: 검증셋을 사용해 프론티어의 점을 고른 다음, 선택된 식의 오차를 별도 테스트셋에서 보고.

GP가 우아해 보이고 동시에 과적합하는 식을 만들 수 있어 여기에 규율이 더욱 중요 — 기호 형태가 안심을 주지만 테스트 오차가 중요한 것.

### From Theory to the Code Below

- 섹션 2의 표현 트리 시각화와 연산자/단말 집합은 (A)의 탐색 공간 정의.
- 섹션 3의 `gplearn.SymbolicRegressor` 또는 `pysr.PySRRegressor`가 (B)의 유전자 프로그래밍 루프를 실행; `parsimony_coefficient` 매개변수가 적합도 함수의 `β`.
- 섹션 4의 `pysindy.SINDy`가 함수 라이브러리를 만들고 (C)의 STLSQ를 실행; `threshold`가 희소성 매개변수.
- 섹션 5의 파레토 프론티어 그래프와 식 선택이 (D)의 트레이드오프 탐색기.
- 섹션 6의 "기호 회귀를 사용할 때" 안내가 (E)의 틈새 분석에 매핑.

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
