# 레슨 7: 도이치-조자 알고리즘(Deutsch-Jozsa Algorithm)

[<- 이전: 양자 측정](06_Quantum_Measurement.md) | [다음: 그로버 탐색 알고리즘 ->](08_Grovers_Search.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 도이치-조자 약속 문제(Deutsch-Jozsa promise problem)를 정의하고 상수 함수와 균형 함수의 차이를 설명할 수 있다
2. 양자 오라클(quantum oracle) 모델을 설명하고 양자 알고리즘 분석에서 이것이 중요한 이유를 설명할 수 있다
3. 도이치 알고리즘(Deutsch's algorithm, 1-큐비트 버전)을 단계별로 유도하고 구현할 수 있다
4. $n$-큐비트 입력에 대한 완전한 도이치-조자 알고리즘으로 일반화할 수 있다
5. 2-큐비트 오라클을 사용한 알고리즘의 완전한 예제를 단계별로 추적할 수 있다
6. 도이치-조자 알고리즘이 양자 우위(quantum advantage)를 어떻게 보여주는지, 그리고 그 한계는 무엇인지 설명할 수 있다
7. 특정 함수에 대한 오라클 회로를 구성하고 Python으로 전체 알고리즘을 시뮬레이션할 수 있다

---

1992년 데이비드 도이치(David Deutsch)와 리처드 조자(Richard Jozsa)가 제안한 도이치-조자 알고리즘은 양자와 고전 계산 사이의 지수적(exponential) 분리를 최초로 입증한 알고리즘입니다. 이 알고리즘이 푸는 문제는 다소 인위적이지만, 양자 컴퓨팅의 핵심 아이디어들을 소개합니다: 오라클(oracle) 모델, 위상 역반사(phase kickback), 그리고 간섭(interference)과 결합된 양자 병렬성(quantum parallelism)의 힘이 그것입니다.

문제는 단순합니다: $f: \{0,1\}^n \to \{0,1\}$ 인 함수가 *상수(constant)* (모든 입력에 대해 동일한 출력)이거나 *균형(balanced)* (정확히 절반의 입력에 대해 0을 출력하고 나머지 절반에 대해 1을 출력)이라는 보장이 주어졌을 때, 어느 유형인지 판별하는 것입니다. 고전적으로는 최악의 경우 적어도 $2^{n-1} + 1$개의 입력에 대해 $f$를 평가해야 합니다. 양자역학적으로는 정확히 *한 번*의 평가만 필요합니다.

> **비유:** 동전이 공정한 것인지 양면이 앞면인 것인지 판별한다고 상상해 보세요 — 고전적으로는 여러 번 던져야 할 수도 있지만, 양자역학은 단 한 번의 던지기로 확인할 수 있습니다. 동전은 함수 $f$와 같습니다: 항상 같은 답을 내놓거나(상수/양면 앞면) 균형 잡힌 답을 내놓습니다(공정한 동전). 양자 "던지기"는 중첩(superposition)을 통해 모든 가능한 입력을 동시에 질의하며, 간섭(interference)이 단 한 번의 측정으로 답을 명확하게 드러냅니다.

## 목차

1. [약속 문제(The Promise Problem)](#1-약속-문제the-promise-problem)
2. [오라클 모델(The Oracle Model)](#2-오라클-모델the-oracle-model)
3. [도이치 알고리즘 (1 큐비트)](#3-도이치-알고리즘-1-큐비트)
4. [도이치-조자 알고리즘 (n 큐비트)](#4-도이치-조자-알고리즘-n-큐비트)
5. [단계별 예제](#5-단계별-예제)
6. [왜 중요한가](#6-왜-중요한가)
7. [연습 문제](#7-연습-문제)

---

## 1. 약속 문제(The Promise Problem)

### 1.1 문제 정의

**입력**: 다음 중 하나라는 약속이 주어진 함수 $f: \{0,1\}^n \to \{0,1\}$:
- **상수(Constant)**: 모든 $x$에 대해 $f(x) = 0$, 또는 모든 $x$에 대해 $f(x) = 1$
- **균형(Balanced)**: 정확히 $2^{n-1}$개의 입력에 대해 $f(x) = 0$이고, 나머지 $2^{n-1}$개의 입력에 대해 $f(x) = 1$

**출력**: $f$가 상수인지 균형인지 판별.

### 1.2 고전적 복잡도

고전적으로는 다른 입력들에 대해 $f$를 평가하면서 답을 결정해야 합니다:
- **최선의 경우**: 운 좋게 2번의 질의 후 서로 다른 두 출력을 발견합니다. 답: 균형.
- **최악의 경우**: $2^{n-1} + 1$개의 입력을 평가합니다. 모두 같은 출력을 낸다면, 비둘기집 원리(pigeonhole principle)에 의해 $f$는 균형일 수 없으므로 상수임을 알 수 있습니다.
- **평균적인 경우**: 무작위화(randomization)를 사용하면, 높은 확률로 $O(1)$번의 질의로 충분합니다(하지만 확실하지는 않습니다).

핵심: *결정론적(deterministic)* 고전 계산의 경우, 최악의 경우 $2^{n-1} + 1$번의 질의가 필요하며 — 이는 $n$에 대해 지수적입니다.

### 1.3 양자 복잡도

도이치-조자 알고리즘은 $f$에 대해 정확히 **한 번**의 질의로 확실하게 답을 결정합니다. 이는 결정론적 고전 계산에 비해 지수적 속도 향상입니다.

참고: 이 속도 향상은 *결정론적* 고전 알고리즘에 비해서만 해당됩니다. 무작위화된 고전 알고리즘은 높은 확률로 $O(1)$번의 질의로 이를 풀 수 있습니다. 따라서 도이치-조자 알고리즘은 유계 오류(bounded-error) 알고리즘이 아닌 정확(exact, 오류 없음) 알고리즘에 대한 지수적 격차를 보여줍니다. 개념 증명(proof of concept)으로서 여전히 중요합니다.

```python
import numpy as np

# 고전적 도이치-조자: 고전적 질의 복잡도 시연

def classical_deutsch_jozsa(f, n):
    """
    도이치-조자 문제에 대한 고전 알고리즘.

    최악의 경우 왜 2^{n-1} + 1번의 질의가 필요한가?
    처음 2^{n-1}번의 평가가 모두 같은 값을 줄 수 있는데,
    이는 상수 함수와 균형 함수 모두와 일치합니다. 오직
    (2^{n-1} + 1)번째 질의만이 구별할 수 있습니다: 그것도 같은
    값을 준다면, f는 상수임에 틀림없습니다 (균형 함수는 정의상
    2^{n-1}개 이상의 동일한 출력을 가질 수 없으므로).
    """
    first_value = f(0)
    queries = 1

    for x in range(1, 2**n):
        if f(x) != first_value:
            return "balanced", queries + 1
        queries += 1
        # 조기 종료: 절반 + 1개를 확인했고 모두 같다면 상수
        if queries > 2**(n-1):
            return "constant", queries

    return "constant", queries

# n=3 예제 함수
n = 3

# 상수 함수: 모든 x에 대해 f(x) = 0
f_constant = lambda x: 0

# 균형 함수: f(x) = x의 패리티 (2의 거듭제곱 도메인에서 균형)
f_balanced = lambda x: bin(x).count('1') % 2

# 최악의 경우 상수: 모든 출력이 0
result, queries = classical_deutsch_jozsa(f_constant, n)
print(f"=== Classical Deutsch-Jozsa (n={n}, 2^n={2**n} inputs) ===\n")
print(f"Constant f(x)=0: result={result}, queries={queries}")
print(f"  (Worst case: had to query {queries} times out of {2**n})")

result, queries = classical_deutsch_jozsa(f_balanced, n)
print(f"Balanced f(x)=parity(x): result={result}, queries={queries}")
print(f"  (Got lucky: found disagreement after {queries} queries)")

# 최악의 경우가 2^{n-1}+1인 이유
print(f"\n  Theoretical worst case: 2^{n-1}+1 = {2**(n-1)+1} queries")
print(f"  Total possible inputs: 2^{n} = {2**n}")
```

---

## 2. 오라클 모델(The Oracle Model)

### 2.1 오라클이란?

양자 컴퓨팅에서 함수 $f$는 **양자 오라클(quantum oracle)** (또는 "블랙 박스"라고도 함)을 통해 접근합니다. 오라클은 $f$를 가역적(reversible)인 방식으로 인코딩하는 유니터리(unitary) 연산자 $U_f$입니다.

표준 오라클은 $n$-큐비트 입력 레지스터 $|x\rangle$와 1-큐비트 출력 레지스터 $|y\rangle$를 받습니다:

$$U_f|x\rangle|y\rangle = |x\rangle|y \oplus f(x)\rangle$$

여기서 $\oplus$는 모듈로 2 덧셈(XOR)입니다.

### 2.2 왜 XOR인가?

XOR 구성은 오라클이 *가역적*(따라서 유니터리)임을 보장합니다:
- $U_f U_f |x\rangle|y\rangle = |x\rangle|y \oplus f(x) \oplus f(x)\rangle = |x\rangle|y\rangle$
- 따라서 $U_f^{-1} = U_f$ (오라클은 자기 자신의 역)

### 2.3 위상 오라클(Phase Oracle)과 위상 역반사(Phase Kickback)

양자 알고리즘의 핵심 기법은 오라클을 사용하여 비트를 뒤집는 것이 아닌 *위상(phase)*을 부여하는 것입니다. 출력 큐비트를 $|-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}$로 설정합니다:

$$U_f|x\rangle|-\rangle = |x\rangle|f(x) \oplus -\rangle$$

그런데 $b \in \{0, 1\}$에 대해 $|b \oplus -\rangle = (-1)^b|-\rangle$이므로 (직접 확인해 보세요!):

$$U_f|x\rangle|-\rangle = (-1)^{f(x)}|x\rangle|-\rangle$$

오라클이 $(-1)^{f(x)}$의 위상을 입력 레지스터에 "역반사(kicked back)"시켰으며, 출력 큐비트는 $|-\rangle$ 상태로 변하지 않습니다. 이 **위상 역반사(phase kickback)** 기법은 오라클을 비트 뒤집기 연산에서 위상 연산으로 변환하며, 간섭(interference)에 필수적입니다.

```python
import numpy as np

# 오라클 구성

def build_oracle_matrix(f, n):
    """
    n+1 큐비트에 대한 오라클 U_f의 유니터리 행렬 구성.
    U_f|x>|y> = |x>|y XOR f(x)>

    왜 전체 행렬을 구성하는가? 작은 n의 경우, 이것이 오라클이
    무엇을 하는지 이해하는 가장 명확한 방법입니다. 행렬의 각 열은
    각 기저 상태에 무슨 일이 일어나는지 보여줍니다.
    """
    dim = 2**(n+1)  # n개의 입력 큐비트 + 1개의 출력 큐비트
    U = np.zeros((dim, dim), dtype=complex)

    for x in range(2**n):
        for y in range(2):
            # 입력 기저 상태: |x>|y>
            # 상태 벡터에서의 인덱스: x * 2 + y
            in_idx = x * 2 + y

            # 출력: |x>|y XOR f(x)>
            out_y = y ^ f(x)
            out_idx = x * 2 + out_y

            U[out_idx, in_idx] = 1

    return U

def build_phase_oracle(f, n):
    """
    위상 오라클 구성: U_f|x> = (-1)^f(x) |x>
    위상 역반사 후의 오라클 (출력 큐비트 추적 소거).

    왜 별도의 함수인가? 도이치-조자 알고리즘에서 출력 큐비트는
    |-> 상태를 유지하며 무시할 수 있습니다. 입력 레지스터에
    대한 효과적인 연산은 그냥 위상의 대각 행렬입니다.
    """
    dim = 2**n
    U = np.zeros((dim, dim), dtype=complex)

    for x in range(dim):
        U[x, x] = (-1)**f(x)

    return U

# 예제: n=2, 균형 함수 f(x) = x0 (최하위 비트)
n = 2
f_balanced = lambda x: x & 1  # f(00)=0, f(01)=1, f(10)=0, f(11)=1

print("=== Oracle for f(x) = x0 (LSB), n=2 ===\n")

# 전체 오라클
U_f = build_oracle_matrix(f_balanced, n)
print(f"Full oracle U_f ({2**(n+1)}x{2**(n+1)} matrix):")
print(np.real(U_f).astype(int))

# 위상 오라클
U_phase = build_phase_oracle(f_balanced, n)
print(f"\nPhase oracle ({2**n}x{2**n} diagonal matrix):")
print(f"Diagonal: {np.diag(U_phase).real}")
print("(-1)^f(x) for x = 00, 01, 10, 11:", [(-1)**f_balanced(x) for x in range(4)])

# 검증: 위상 역반사
print("\n=== Phase Kickback Demonstration ===\n")
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)

for x in range(2**n):
    # |x>|-> 준비
    ket_x = np.zeros(2**n, dtype=complex)
    ket_x[x] = 1
    input_state = np.kron(ket_x, ket_minus)

    # 오라클 적용
    output_state = U_f @ input_state

    # 예상 결과: (-1)^f(x) |x>|->
    expected = (-1)**f_balanced(x) * input_state

    match = np.allclose(output_state, expected)
    print(f"|{format(x, f'0{n}b')}>|->  ->  (-1)^{f_balanced(x)} |{format(x, f'0{n}b')}>|->  "
          f"{'OK' if match else 'FAIL'}")
```

---

## 3. 도이치 알고리즘 (1 큐비트)

### 3.1 가장 단순한 경우

도이치 알고리즘(1985)은 $n = 1$인 버전입니다. 함수 $f: \{0, 1\} \to \{0, 1\}$에는 네 가지 가능성이 있습니다:

| 함수 | $f(0)$ | $f(1)$ | 유형 |
|------|--------|--------|------|
| $f_1$ | 0 | 0 | 상수 |
| $f_2$ | 1 | 1 | 상수 |
| $f_3$ | 0 | 1 | 균형 |
| $f_4$ | 1 | 0 | 균형 |

고전적으로는 유형을 판별하기 위해 두 입력 모두에 대해 $f$를 평가해야 합니다. 도이치 알고리즘은 한 번의 질의로 이를 수행합니다.

### 3.2 회로

```
q0: ─[H]─────[U_f]─[H]─[M]─
              │
q1: ─[X]─[H]─[U_f]─────────
```

1. 시작: $|01\rangle$ (q0 = $|0\rangle$, X 게이트 후 q1 = $|1\rangle$)
2. 두 큐비트 모두에 H 적용
3. 오라클 $U_f$ 적용
4. q0에 H 적용
5. q0 측정

### 3.3 단계별 유도

**단계 0**: $|0\rangle|1\rangle$ 초기화

**단계 1**: 두 큐비트 모두에 H 적용:

$$|0\rangle|1\rangle \xrightarrow{H \otimes H} |+\rangle|-\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes \frac{1}{\sqrt{2}}(|0\rangle - |1\rangle)$$

**단계 2**: $U_f$ 적용 (위상 역반사 사용):

$$\frac{1}{\sqrt{2}}\left[(-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle\right] \otimes |-\rangle$$

**단계 3**: q0에 집중 ($|-\rangle$에 머무는 q1은 무시):

$$\frac{1}{\sqrt{2}}\left[(-1)^{f(0)}|0\rangle + (-1)^{f(1)}|1\rangle\right]$$

$(-1)^{f(0)}$을 인수분해:

$$(-1)^{f(0)} \cdot \frac{1}{\sqrt{2}}\left[|0\rangle + (-1)^{f(0) \oplus f(1)}|1\rangle\right]$$

**단계 4**: q0에 H 적용:

$$(-1)^{f(0)} \cdot H\left[\frac{1}{\sqrt{2}}\left(|0\rangle + (-1)^{f(0) \oplus f(1)}|1\rangle\right)\right]$$

$H|+\rangle = |0\rangle$와 $H|-\rangle = |1\rangle$을 이용:

- $f(0) \oplus f(1) = 0$이면 (상수): $H|+\rangle = |0\rangle$
- $f(0) \oplus f(1) = 1$이면 (균형): $H|-\rangle = |1\rangle$

**단계 5**: q0 측정:
- **결과 0**: $f$는 상수
- **결과 1**: $f$는 균형

한 번의 질의. 100% 정확. 항상.

```python
import numpy as np

# 도이치 알고리즘 구현

def deutsch_algorithm(f):
    """
    n=1 도이치-조자 문제에 대한 도이치 알고리즘.

    왜 작동하는가? 핵심 통찰은 위상 역반사 + 간섭입니다.
    1. 아다마르(Hadamard)가 중첩을 생성: |0>과 |1>을 동시에 질의
    2. 위상 역반사: 오라클이 (-1)^f(x) 위상을 부여
    3. 두 번째 아다마르: 두 진폭 사이의 간섭
       - f(0)=f(1)이면 (상수): 진폭이 보강 간섭 -> |0>
       - f(0)!=f(1)이면 (균형): 진폭이 상쇄 -> |1>
    """
    # 단계 1: 초기 상태 |01>
    # 두 큐비트 모두에 H 적용 후: |+>|->
    # 위상 역반사로 오라클 적용 후: 큐비트 0의 상태는
    # ((-1)^f(0)|0> + (-1)^f(1)|1>) / sqrt(2)
    # 큐비트 0에 H 적용: 측정

    # 직접 계산
    state_q0 = np.array([(-1)**f(0), (-1)**f(1)], dtype=complex) / np.sqrt(2)

    # 아다마르 적용
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    final = H @ state_q0

    # 측정: |0>과 |1>의 확률
    p0 = abs(final[0])**2
    p1 = abs(final[1])**2

    measurement = 0 if p0 > 0.5 else 1
    result = "constant" if measurement == 0 else "balanced"

    return result, p0, p1

# 가능한 네 가지 함수 모두 테스트
print("=== Deutsch's Algorithm ===\n")
functions = {
    "f1: f(0)=0, f(1)=0": lambda x: 0,
    "f2: f(0)=1, f(1)=1": lambda x: 1,
    "f3: f(0)=0, f(1)=1": lambda x: x,
    "f4: f(0)=1, f(1)=0": lambda x: 1 - x,
}

for name, f in functions.items():
    result, p0, p1 = deutsch_algorithm(f)
    print(f"{name}")
    print(f"  P(0) = {p0:.4f}, P(1) = {p1:.4f}")
    print(f"  Result: {result}")
    expected = "constant" if f(0) == f(1) else "balanced"
    print(f"  Correct? {result == expected}")
    print()

# 전체 회로 시뮬레이션 (보조 큐비트 포함)
print("=== Full Circuit Simulation ===\n")

def deutsch_full_simulation(f):
    """완전한 2-큐비트 회로 시뮬레이션."""
    # 게이트
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)

    # n=1에 대한 오라클 행렬
    U_f = np.zeros((4, 4), dtype=complex)
    for x in range(2):
        for y in range(2):
            in_idx = x * 2 + y
            out_idx = x * 2 + (y ^ f(x))
            U_f[out_idx, in_idx] = 1

    # 초기 상태: |0>|0>
    state = np.array([1, 0, 0, 0], dtype=complex)

    # 큐비트 1(보조)에 X 적용: |0>|0> -> |0>|1>
    state = np.kron(I2, X) @ state

    # 두 큐비트 모두에 H 적용
    state = np.kron(H, H) @ state

    # 오라클 적용
    state = U_f @ state

    # 큐비트 0(입력)에 H 적용
    state = np.kron(H, I2) @ state

    # 큐비트 0 측정
    # P(q0=0) = |<00|state>|^2 + |<01|state>|^2
    p_q0_0 = abs(state[0])**2 + abs(state[1])**2
    p_q0_1 = abs(state[2])**2 + abs(state[3])**2

    return "constant" if p_q0_0 > 0.5 else "balanced"

# 전체 시뮬레이션이 단순화된 버전과 일치하는지 검증
for name, f in functions.items():
    result = deutsch_full_simulation(f)
    expected = "constant" if f(0) == f(1) else "balanced"
    print(f"{name}: {result} ({'CORRECT' if result == expected else 'WRONG'})")
```

---

## 4. 도이치-조자 알고리즘 (n 큐비트)

### 4.1 회로

```
         ┌───┐     ┌─────┐ ┌───┐ ┌───┐
q0:   |0>┤ H ├─────┤     ├─┤ H ├─┤ M ├
         ├───┤     │     │ ├───┤ ├───┤
q1:   |0>┤ H ├─────┤     ├─┤ H ├─┤ M ├
         ├───┤     │ U_f │ ├───┤ ├───┤
...       ...      │     │  ...   ...
         ├───┤     │     │ ├───┤ ├───┤
q_{n-1}|0>┤ H ├───┤     ├─┤ H ├─┤ M ├
         ├───┤     │     │ └───┘ └───┘
ancilla|1>┤ H ├───┤     ├──────────────
         └───┘     └─────┘
```

### 4.2 수학적 유도

**단계 1**: $|0\rangle^{\otimes n}|1\rangle$로 시작

**단계 2**: $H^{\otimes (n+1)}$ 적용:

$$\frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} |x\rangle \otimes |-\rangle$$

**단계 3**: $U_f$ 적용 (위상 역반사와 함께):

$$\frac{1}{\sqrt{2^n}} \sum_{x=0}^{2^n-1} (-1)^{f(x)} |x\rangle \otimes |-\rangle$$

**단계 4**: 입력 레지스터에 $H^{\otimes n}$ 적용. 항등식 사용:

$$H^{\otimes n}|x\rangle = \frac{1}{\sqrt{2^n}} \sum_{z=0}^{2^n-1} (-1)^{x \cdot z}|z\rangle$$

여기서 $x \cdot z = \bigoplus_i x_i z_i$는 모듈로 2에서의 비트별 내적(bitwise inner product):

$$\frac{1}{2^n} \sum_{z=0}^{2^n-1} \left[\sum_{x=0}^{2^n-1} (-1)^{f(x) + x \cdot z}\right] |z\rangle \otimes |-\rangle$$

**단계 5**: 입력 레지스터 측정. 결과 $|0\rangle^{\otimes n}$의 확률:

$$P(0^n) = \left|\frac{1}{2^n} \sum_{x=0}^{2^n-1} (-1)^{f(x)}\right|^2$$

**핵심 통찰**:
- $f$가 **상수**이면: 모든 $(-1)^{f(x)}$ 항이 같은 부호를 가집니다. 합은 $\pm 2^n$입니다. 따라서 $P(0^n) = 1$.
- $f$가 **균형**이면: 절반의 항은 $+1$이고 절반은 $-1$입니다. 합은 0입니다. 따라서 $P(0^n) = 0$.

**결과**: 상수이면 확률 1로 $|0\rangle^{\otimes n}$을 측정하고, 균형이면 확률 0으로 측정합니다. 한 번의 질의. 결정론적.

### 4.3 간섭의 힘

알고리즘은 *보강(constructive)*과 *상쇄(destructive)* 간섭 때문에 작동합니다:
- 상수 함수의 경우, 모든 항이 $|0^n\rangle$에서 보강 간섭
- 균형 함수의 경우, 항들이 $|0^n\rangle$에서 완벽하게 상쇄

이것이 양자 알고리즘의 근본 패턴입니다: 잘못된 답은 상쇄 간섭하고 올바른 답은 보강 간섭하도록 계산을 배열하는 것입니다.

```python
import numpy as np

def deutsch_jozsa(f, n):
    """
    전체 도이치-조자 알고리즘 시뮬레이션.

    왜 이러한 구조인가? 알고리즘은 세 단계로 구성됩니다:
    1. 중첩(SUPERPOSITION): H 게이트가 입력을 2^n개의 모든 입력의
       균등 중첩 상태로 만들기 — "모든 입력을 동시에 질의"
    2. 위상 표시(PHASE MARKING): 오라클이 각 입력 x에 (-1)^f(x) 위상을 표시
    3. 간섭(INTERFERENCE): H 게이트가 보강/상쇄 간섭을 일으켜
       상수 함수의 경우 진폭을 |0...0>에 집중시킴

    그런 다음 측정이 두 경우를 결정론적으로 구별합니다.
    """
    dim = 2**n

    # 단계 1: 균등 중첩 생성
    # |psi_0> = H^n |0...0> = (1/sqrt(2^n)) sum_x |x>
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    # 단계 2: 위상 오라클 적용
    # |psi_1> = (1/sqrt(2^n)) sum_x (-1)^f(x) |x>
    for x in range(dim):
        state[x] *= (-1)**f(x)

    # 단계 3: H^n 적용
    # H^n 행렬 구성
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    H_n = np.array([[1]], dtype=complex)
    for _ in range(n):
        H_n = np.kron(H_n, H)

    state = H_n @ state

    # 단계 4: 측정
    # |0...0>의 확률
    p_zero = abs(state[0])**2

    # 모든 확률
    probs = np.abs(state)**2

    return {
        'p_all_zeros': p_zero,
        'result': 'constant' if p_zero > 0.5 else 'balanced',
        'state': state,
        'probabilities': probs,
    }

# 다양한 함수로 테스트
print("=== Deutsch-Jozsa Algorithm ===\n")

for n in [2, 3, 4]:
    print(f"--- n = {n} ({2**n} inputs) ---\n")

    # 상수 함수: f(x) = 0
    f_const = lambda x: 0
    result = deutsch_jozsa(f_const, n)
    print(f"  f(x) = 0 (constant):")
    print(f"    P(0...0) = {result['p_all_zeros']:.6f}")
    print(f"    Result: {result['result']}")

    # 상수 함수: f(x) = 1
    f_const1 = lambda x: 1
    result = deutsch_jozsa(f_const1, n)
    print(f"  f(x) = 1 (constant):")
    print(f"    P(0...0) = {result['p_all_zeros']:.6f}")
    print(f"    Result: {result['result']}")

    # 균형 함수: f(x) = x의 최상위 비트
    f_balanced = lambda x, n=n: (x >> (n-1)) & 1
    result = deutsch_jozsa(f_balanced, n)
    print(f"  f(x) = MSB(x) (balanced):")
    print(f"    P(0...0) = {result['p_all_zeros']:.6f}")
    print(f"    Result: {result['result']}")

    # 균형 함수: f(x) = x의 패리티
    f_parity = lambda x: bin(x).count('1') % 2
    result = deutsch_jozsa(f_parity, n)
    print(f"  f(x) = parity(x) (balanced):")
    print(f"    P(0...0) = {result['p_all_zeros']:.6f}")
    print(f"    Result: {result['result']}")
    print()
```

---

## 5. 단계별 예제

$n = 2$이고 균형 함수 $f(x) = x_0$(최하위 비트)인 경우로 알고리즘을 추적해 봅시다.

### 5.1 함수

| $x$ (이진) | $x$ (십진) | $f(x) = x_0$ |
|:---:|:---:|:---:|
| 00 | 0 | 0 |
| 01 | 1 | 1 |
| 10 | 2 | 0 |
| 11 | 3 | 1 |

이 함수는 균형입니다: $f$는 $\{00, 10\}$에 대해 0을 출력하고 $\{01, 11\}$에 대해 1을 출력합니다.

### 5.2 상세한 상태 변화

**단계 1**: 중첩:

$$|\psi_0\rangle = H^{\otimes 2}|00\rangle = \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$$

**단계 2**: 오라클이 위상을 적용:

$$|\psi_1\rangle = \frac{1}{2}\left[(-1)^0|00\rangle + (-1)^1|01\rangle + (-1)^0|10\rangle + (-1)^1|11\rangle\right]$$

$$= \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$$

**단계 3**: $H^{\otimes 2}$ 적용:

$H^{\otimes 2}$의 매핑을 떠올리면:
- $|00\rangle \to \frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$
- $|01\rangle \to \frac{1}{2}(|00\rangle - |01\rangle + |10\rangle - |11\rangle)$
- $|10\rangle \to \frac{1}{2}(|00\rangle + |01\rangle - |10\rangle - |11\rangle)$
- $|11\rangle \to \frac{1}{2}(|00\rangle - |01\rangle - |10\rangle + |11\rangle)$

$H^{\otimes 2}|\psi_1\rangle$ 계산:

$$= \frac{1}{2}\left[\frac{1}{2}(+,+,+,+) - \frac{1}{2}(+,-,+,-) + \frac{1}{2}(+,+,-,-) - \frac{1}{2}(+,-,-,+)\right]$$

여기서 $(a,b,c,d)$는 $a|00\rangle + b|01\rangle + c|10\rangle + d|11\rangle$을 의미합니다.

$|00\rangle$의 계수: $\frac{1}{4}(1 - 1 + 1 - 1) = 0$
$|01\rangle$의 계수: $\frac{1}{4}(1 + 1 + 1 + 1) = 1$
$|10\rangle$의 계수: $\frac{1}{4}(1 - 1 - 1 + 1) = 0$
$|11\rangle$의 계수: $\frac{1}{4}(1 + 1 - 1 - 1) = 0$

**결과**: $|\psi_2\rangle = |01\rangle$

$P(00) = 0$ — 모두 0이 아니므로 함수는 균형입니다. 정확!

```python
import numpy as np

# n=2에 대한 상세 단계별 추적

n = 2
f = lambda x: x & 1  # f(x) = x0 (LSB)

print("=== Step-by-Step Trace: n=2, f(x) = x0 ===\n")

# 표준 게이트
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
H2 = np.kron(H, H)

# 단계 1: 중첩
state = np.array([1, 0, 0, 0], dtype=complex)  # |00>
state = H2 @ state
print("Step 1: H tensor H on |00>")
print(f"  |psi_0> = {state}")
print(f"  = (1/2)(|00> + |01> + |10> + |11>)\n")

# 단계 2: 오라클
print("Step 2: Apply phase oracle f(x) = x0")
print("  Phases: ", end="")
for x in range(4):
    phase = (-1)**f(x)
    state[x] *= phase
    print(f"(-1)^{f(x)}={phase:+d} for |{format(x, '02b')}>", end="  ")
print()
print(f"  |psi_1> = {state}")
print(f"  = (1/2)(|00> - |01> + |10> - |11>)\n")

# 단계 3: H 텐서 H
state = H2 @ state
print("Step 3: Apply H tensor H")
print(f"  |psi_2> = {np.round(state, 6)}")
print()

# 결과 표시
for i in range(4):
    label = format(i, '02b')
    if abs(state[i]) > 1e-10:
        print(f"  |{label}>: amplitude = {state[i]:.4f}, P = {abs(state[i])**2:.4f}")
    else:
        print(f"  |{label}>: amplitude = {state[i]:.4f}, P = {abs(state[i])**2:.4f} (zero!)")

p_00 = abs(state[0])**2
print(f"\nP(00) = {p_00:.6f}")
print(f"Result: {'constant' if p_00 > 0.5 else 'balanced'}")
print(f"Expected: balanced")
print(f"\nThe amplitude at |00> is ZERO because destructive interference")
print(f"perfectly canceled it. This is the quantum advantage!")
```

---

## 6. 왜 중요한가

### 6.1 역사적 의의

도이치-조자 알고리즘(그리고 그 선구자인 도이치 알고리즘)은 양자 컴퓨터가 *일부* 문제를 고전 컴퓨터보다 지수적으로 적은 질의로 풀 수 있음을 최초로 입증했습니다. 이는 양자 계산이 단순한 이론적 호기심이 아닌 진정으로 다른 계산 모델임을 증명했습니다.

### 6.2 개념적 기여

이 알고리즘은 이후 거의 모든 양자 알고리즘에서 사용되는 몇 가지 아이디어를 소개했습니다:

| 개념 | 설명 | 사용 위치 |
|------|------|---------|
| **오라클 모델(Oracle model)** | 함수에 대한 블랙박스 접근 | 그로버, 쇼어 등 다수 |
| **위상 역반사(Phase kickback)** | 비트 오라클을 위상 오라클로 변환 | 그로버, QPE, 쇼어 |
| **양자 병렬성(Quantum parallelism)** | 중첩이 모든 입력을 질의 | 모든 양자 알고리즘 |
| **간섭(Interference)** | 올바른 답에 대해 보강 | 그로버, 쇼어, VQE |
| **아다마르 샌드위치(Hadamard sandwich)** | $H^n \to \text{오라클} \to H^n$ | 번스타인-바지라니, 사이먼 |

### 6.3 한계

- 문제가 인위적입니다 ($f$가 상수 또는 균형이라는 "약속"이 강력함)
- 무작위화된 고전 알고리즘으로 효율적으로 풀 수 있음 (BPP 관점)
- 실용적인 응용이 없음 (상수 함수와 균형 함수를 구별할 필요가 실제로는 없음)
- 장점은 유계 오류가 아닌 정확(오류 없음) 대비이며, 실용적인 속도 향상이 아님

### 6.4 다음에 오는 것

도이치-조자의 아이디어는 직접적으로 다음으로 이어집니다:
- **번스타인-바지라니 알고리즘(Bernstein-Vazirani algorithm)**: $f(x) = s \cdot x$에서 숨겨진 문자열 $s$를 한 번의 질의로 찾음
- **사이먼 알고리즘(Simon's algorithm)**: 지수적 속도 향상으로 함수의 주기를 찾음
- **그로버 탐색(Grover's search)** ([레슨 8](08_Grovers_Search.md)): 비구조적 탐색에서 이차 속도 향상
- **쇼어 알고리즘(Shor's algorithm)**: 인수분해에서 지수적 속도 향상 (사이먼의 아이디어 기반)

```python
import numpy as np

# 고전적 질의 복잡도와 양자 질의 복잡도 비교

print("=== Query Complexity Comparison ===\n")
print(f"{'n':>5} {'2^n':>10} {'Classical (det)':>18} {'Classical (rand)':>18} {'Quantum':>10}")
print("-" * 65)

for n in range(1, 16):
    dim = 2**n
    classical_det = 2**(n-1) + 1
    classical_rand = 2  # 높은 확률로 O(1)번의 질의
    quantum = 1

    print(f"{n:>5} {dim:>10,} {classical_det:>18,} {classical_rand:>18} {quantum:>10}")

print(f"\nThe quantum advantage is EXPONENTIAL over deterministic classical,")
print(f"but only CONSTANT over randomized classical.")
print(f"\nThis illustrates an important nuance: quantum speedups depend on")
print(f"what classical model you compare against.")

# 보너스: 번스타인-바지라니 (밀접하게 관련된 알고리즘)
print("\n\n=== Bonus: Bernstein-Vazirani Algorithm ===\n")
print("Problem: Find hidden string s where f(x) = s . x (mod 2)")
print("Classical: n queries. Quantum: 1 query.\n")

def bernstein_vazirani(f, n):
    """
    번스타인-바지라니 알고리즘.

    왜 도이치-조자와 관련이 있는가? 정확히 같은 회로를 사용합니다!
    유일한 차이는 함수 클래스: f(x) = s.x (숨겨진 문자열 s와의
    비트별 내적). 동일한 H-오라클-H 구조가 한 번에 s를 직접 드러냅니다.
    """
    dim = 2**n
    # 도이치-조자와 같은 회로!
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for x in range(dim):
        state[x] *= (-1)**f(x)
    H_n = np.array([[1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    for _ in range(n):
        H_n = np.kron(H_n, H)
    state = H_n @ state
    # 상태는 이제 확률 1로 |s>
    s = np.argmax(np.abs(state)**2)
    return format(s, f'0{n}b')

# 테스트
n = 5
secret = 0b10110  # 숨겨진 문자열 "10110"

def f_bv(x, s=secret):
    """f(x) = s . x (비트별 내적 mod 2)."""
    return bin(x & s).count('1') % 2

result = bernstein_vazirani(f_bv, n)
print(f"Hidden string s = {format(secret, f'0{n}b')}")
print(f"Algorithm found: {result}")
print(f"Correct? {result == format(secret, f'0{n}b')}")
print(f"Queries used: 1 (classical would need {n})")
```

---

## 7. 연습 문제

### 연습 1: 오라클 구성

$n = 2$ 큐비트에 대한 다음 함수들의 위상 오라클 행렬을 구성하세요:

a) 모든 $x$에 대해 $f(x) = 1$ (상수)
b) $f(x) = x_1$ (최상위 비트)
c) $f(x) = x_0 \oplus x_1$ (비트의 XOR)

각각에 대해 도이치-조자 알고리즘을 실행하고 출력을 검증하세요.

### 연습 2: 스케일링 분석

a) $n = 1$부터 $n = 15$까지 도이치-조자 알고리즘을 구현하세요.
b) 각 $n$에 대해, 상수 함수에 대해 $P(0^n) = 1$이고 균형 함수에 대해 $P(0^n) = 0$임을 검증하세요.
c) 각 $n$에 대한 고전 시뮬레이션 시간을 측정하세요. 어느 $n$에서 시뮬레이션이 느려지기 시작하나요? (실제 양자 하드웨어가 필요한 이유를 보여줍니다!)

### 연습 3: 약속되지 않은 함수

상수도 균형도 아닌 함수에 도이치-조자 알고리즘을 실행하면 어떻게 되나요? 예를 들어, $x < 2^{n-1}/3$이면 $f(x) = 1$이고 그렇지 않으면 $f(x) = 0$인 경우.

a) $P(0^n)$을 해석적으로 계산하세요.
b) 시뮬레이션으로 검증하세요.
c) 출력에서 여전히 유용한 정보를 추출할 수 있나요?

### 연습 4: 전체 회로 구현

[레슨 4](04_Quantum_Circuits.md)의 `QuantumCircuit` 시뮬레이터를 사용하여 $n = 3$에 대한 완전한 도이치-조자 회로(보조 큐비트 포함)를 구현하세요. 최소 두 개의 상수 함수와 두 개의 균형 함수로 실행하고, 결과가 이론과 일치하는지 검증하세요.

### 연습 5: 번스타인-바지라니 확장

a) $n = 8$에 대한 번스타인-바지라니 알고리즘을 구현하고 숨겨진 문자열을 올바르게 찾는지 검증하세요.
b) 알고리즘을 수정하여 미지의 비트 $b$에 대해 $f(x) = s \cdot x \oplus b$인 경우를 처리할 수 있나요? 이것이 출력에 어떤 영향을 미치나요?
c) 다양한 $n$에 대해 양자 질의 수(1)와 고전 질의 수($n$)를 비교하세요. 실용적인 관점에서 이 격차가 도이치-조자 격차보다 덜 중요한 이유는 무엇인가요?

---

[<- 이전: 양자 측정](06_Quantum_Measurement.md) | [다음: 그로버 탐색 알고리즘 ->](08_Grovers_Search.md)
