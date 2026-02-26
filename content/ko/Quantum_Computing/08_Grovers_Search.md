# 레슨 8: 그로버 탐색 알고리즘(Grover's Search Algorithm)

[<- 이전: 도이치-조자 알고리즘](07_Deutsch_Jozsa_Algorithm.md) | [다음: 양자 푸리에 변환 ->](09_Quantum_Fourier_Transform.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 비구조적 탐색(unstructured search) 문제를 정의하고 $O(N)$ 질의의 고전적 하한(classical lower bound)을 설명할 수 있다
2. 그로버의 양자 한계 $O(\sqrt{N})$ 질의를 기술하고 이 이차 속도 향상이 증명 가능하게 최적인 이유를 설명할 수 있다
3. 특정 목표 원소에 대한 그로버 오라클(위상 표시)을 구성할 수 있다
4. 확산 연산자(평균에 대한 반전(inversion about the mean))를 유도하고 구현할 수 있다
5. 그로버 알고리즘의 기하학적 해석을 2차원 부분 공간에서의 회전으로 설명할 수 있다
6. 최적 반복 횟수 $\lfloor \pi/4 \cdot \sqrt{N} \rfloor$를 계산하고 다중 해 경우를 처리할 수 있다
7. Python으로 완전한 그로버 알고리즘을 구현하고 진폭 변화를 시각화할 수 있다

---

1996년 Lov Grover가 발표한 그로버 탐색 알고리즘은 정렬되지 않은 데이터베이스 탐색에 이차 속도 향상을 제공합니다. 쇼어 알고리즘의 지수적 속도 향상에 비하면 소박해 보일 수 있지만, 그로버의 결과는 *일반성* 측면에서 놀랍습니다: 어떤 조건을 만족하는 항목을 찾는 탐색 문제로 공식화할 수 있는 모든 문제에 적용됩니다. 더욱이, 이 이차 속도 향상은 증명 가능하게 최적입니다 — 어떤 양자 알고리즘도 비구조적 탐색에서 더 잘할 수 없습니다.

이 알고리즘은 단순한 데이터베이스 탐색을 훨씬 넘어서는 광범위한 응용을 가집니다. 모든 NP 문제(충족 가능성(satisfiability), 그래프 채색, 최적화)는 탐색 문제로 표현될 수 있으며, 그로버 알고리즘은 이 모든 것에 제곱근 속도 향상을 제공합니다. $N$개의 후보가 있는 문제의 경우, 고전적 브루트 포스는 $O(N)$번의 확인이 필요하지만 그로버 알고리즘은 $O(\sqrt{N})$번만 필요합니다.

> **비유:** 그로버 알고리즘은 군중이 파도타기를 하는 것과 같습니다 — 각 반복은 목표의 진폭을 증폭시키는데, 마치 파도 주기마다 목표 사람을 조금씩 더 높이 들어올려 군중 속에서 눈에 띄게 만드는 것과 같습니다. 오라클은 목표 사람을 "표시"하고, 확산 연산자(diffusion operator)는 그 표시를 군중 전체로 전파시켜 목표를 점점 더 눈에 띄게 만듭니다.

## 목차

1. [비구조적 탐색 문제](#1-비구조적-탐색-문제)
2. [그로버 오라클](#2-그로버-오라클)
3. [확산 연산자](#3-확산-연산자)
4. [알고리즘](#4-알고리즘)
5. [기하학적 해석](#5-기하학적-해석)
6. [최적 반복 횟수](#6-최적-반복-횟수)
7. [다중 해](#7-다중-해)
8. [진폭 변화 시각화](#8-진폭-변화-시각화)
9. [연습 문제](#9-연습-문제)

---

## 1. 비구조적 탐색 문제

### 1.1 문제 정의

**주어진 것**: $N = 2^n$에서 $f: \{0, 1, \ldots, N-1\} \to \{0, 1\}$인 함수. $f(w) = 1$이고 $x \neq w$인 모든 $x$에 대해 $f(x) = 0$인 고유한 "목표" 원소 $w$가 존재한다는 약속이 주어집니다.

**목표**: 가능한 한 적은 횟수의 $f$ 평가를 사용하여 $w$를 찾는 것.

이것을 $N$개의 항목이 있는 전화번호부에서 무작위 순서로 특정 이름을 찾는 것으로 생각해 보세요. 알파벳 순서와 같은 구조를 사용할 수 없습니다 — 항목을 하나씩 확인해야 합니다.

### 1.2 고전적 하한

고전적으로는 항목을 하나씩 확인하는 것이 최선입니다:
- **예상 질의 수**: $N/2$ (평균적으로 목표를 찾기 전에 절반의 항목을 확인)
- **최악의 경우**: $N$번의 질의 (목표가 마지막으로 확인하는 항목)
- **하한**: $\Omega(N)$번의 질의가 필요함 (적대자 논증(adversary argument)으로 증명 가능)

### 1.3 양자 하한

그로버 알고리즘은 다음을 사용하여 높은 확률로 목표를 찾습니다:

$$O(\sqrt{N})$$

$f$에 대한 질의 수. 구체적으로, 약 $\frac{\pi}{4}\sqrt{N}$번의 질의면 충분합니다.

이것이 최적임이 증명되었습니다: **BBBV 정리**(Bennett, Bernstein, Brassard, Vazirani)는 비구조적 탐색을 위한 어떤 양자 알고리즘도 $\Omega(\sqrt{N})$번의 질의가 필요함을 보여줍니다.

### 1.4 속도 향상 요약

| 방법 | 질의 수 | $N = 10^6$일 때 | $N = 10^{12}$일 때 |
|------|---------|----------------|-------------------|
| 고전 | $O(N)$ | $\sim 500{,}000$ | $\sim 500{,}000{,}000{,}000$ |
| 그로버 | $O(\sqrt{N})$ | $\sim 785$ | $\sim 785{,}000$ |
| 속도 향상 | $\sqrt{N}$ | $\sim 637\times$ | $\sim 637{,}000\times$ |

```python
import numpy as np

# 고전적 탐색 대 양자 탐색 비교

print("=== Unstructured Search: Classical vs Quantum ===\n")
print(f"{'N':>15} {'Classical':>15} {'Grover':>15} {'Speedup':>12}")
print("-" * 60)

for n in [5, 10, 15, 20, 25, 30, 40, 50]:
    N = 2**n
    classical = N // 2
    grover = int(np.pi / 4 * np.sqrt(N))
    speedup = classical / grover if grover > 0 else float('inf')

    def format_num(x):
        if x < 1e6:
            return f"{x:,}"
        elif x < 1e9:
            return f"{x/1e6:.1f}M"
        elif x < 1e12:
            return f"{x/1e9:.1f}B"
        else:
            return f"{x:.2e}"

    print(f"2^{n:>2} = {format_num(N):>10}  {format_num(classical):>14} "
          f"{format_num(grover):>14} {speedup:>11.0f}x")

print(f"\nThe quadratic speedup means Grover converts 'infeasible' into 'feasible'")
print(f"for many practical problem sizes.")
```

---

## 2. 그로버 오라클

### 2.1 위상 오라클(Phase Oracle)

그로버 오라클(Grover oracle)은 목표 원소의 위상을 뒤집어 표시합니다:

$$O_f|x\rangle = (-1)^{f(x)}|x\rangle = \begin{cases} -|x\rangle & \text{if } x = w \text{ (목표)} \\ |x\rangle & \text{otherwise} \end{cases}$$

행렬 형태로는:

$$O_f = I - 2|w\rangle\langle w|$$

오라클은 반사(reflection)입니다: $|w\rangle$에 직교하는 초평면(hyperplane)에 대해 상태 벡터를 반사시킵니다.

### 2.2 오라클 구성

실제로 오라클은 $f(x)$를 평가하고 위상 역반사(phase kickback)를 적용하는 양자 회로로 구현됩니다 (도이치-조자 알고리즘 [레슨 7](07_Deutsch_Jozsa_Algorithm.md)과 정확히 동일). 핵심 요구 사항은 $f$를 양자적으로 (중첩 상태에서) 평가할 수 있어야 한다는 것입니다. 어떤 $x$가 $f(x) = 1$을 만족하는지 모르더라도 마찬가지입니다.

예를 들어, SAT 공식의 해를 찾는 경우, 오라클 회로는 입력 $|x\rangle$에 대해 공식을 평가하고 공식이 만족되면 위상을 뒤집습니다.

### 2.3 오라클 복잡도

오라클은 내부 회로 복잡도에 관계없이 $f$에 대한 *단일 질의*로 계산됩니다. 이것이 질의 복잡도(query complexity)의 표준 관례입니다. 물론 실제로는 오라클 회로 자체의 게이트 비용이 총 실행 시간에 기여합니다.

```python
import numpy as np

def grover_oracle(n, targets):
    """
    목표 상태에 -1 위상을 표시하는 그로버 오라클 구성.

    O_f = I - 2 * sum_w |w><w|

    왜 이 형태인가? 오라클은 목표 상태들에 직교하는 부분 공간에
    대한 반사(REFLECTION)입니다. 기하학적으로, 목표 방향을 따라
    상태의 성분을 뒤집습니다.

    매개변수:
        n: 큐비트 수
        targets: 목표 인덱스 목록 (표시할 상태들)
    """
    dim = 2**n
    O = np.eye(dim, dtype=complex)
    for w in targets:
        O[w, w] = -1  # 목표 상태의 위상 뒤집기
    return O

# 예제: 8개 원소 탐색 (n=3), 목표 = |101> = 5
n = 3
target = 5

print(f"=== Grover Oracle for n={n}, target=|{format(target, f'0{n}b')}> ===\n")

oracle = grover_oracle(n, [target])

# 균등 중첩에 적용
dim = 2**n
uniform = np.ones(dim, dtype=complex) / np.sqrt(dim)

print("Before oracle (uniform superposition):")
for i in range(dim):
    print(f"  |{format(i, f'0{n}b')}>: amplitude = {uniform[i]:.4f}")

after_oracle = oracle @ uniform

print("\nAfter oracle (target flipped):")
for i in range(dim):
    marker = " <-- FLIPPED" if i == target else ""
    print(f"  |{format(i, f'0{n}b')}>: amplitude = {after_oracle[i]:.4f}{marker}")

print(f"\nOnly the target state |{format(target, f'0{n}b')}> had its sign flipped.")
print("All probabilities remain the same! The oracle works through PHASE,")
print("which will be exploited by the diffusion operator.")
```

---

## 3. 확산 연산자

### 3.1 정의

**확산 연산자(diffusion operator)** (그로버 확산 또는 "평균에 대한 반전(inversion about the mean)"이라고도 함)는:

$$D = 2|s\rangle\langle s| - I$$

여기서 $|s\rangle = H^{\otimes n}|0\rangle^{\otimes n} = \frac{1}{\sqrt{N}} \sum_{x=0}^{N-1} |x\rangle$은 균등 중첩 상태입니다.

### 3.2 무엇을 하는가?

확산 연산자는 상태 벡터를 균등 중첩 $|s\rangle$에 대해 반사시킵니다. 동등하게, "평균에 대한 반전"을 수행합니다: 각 진폭 $\alpha_x$에 대해:

$$\alpha_x \to 2\bar{\alpha} - \alpha_x$$

여기서 $\bar{\alpha} = \frac{1}{N}\sum_x \alpha_x$는 평균 진폭입니다.

### 3.3 왜 평균에 대한 반전인가?

오라클이 목표의 진폭을 $+a$에서 $-a$로 뒤집은 후, 평균 진폭이 *약간 감소*합니다. 그러면 확산 연산자가:
1. 새로운 (약간 감소된) 평균을 계산
2. 각 진폭을 이 평균에 대해 반사

목표 ($-a$로 뒤집어진)는 평균 *위*의 값으로 반사되고, 다른 모든 진폭들($+a$)은 평균 *아래*의 값으로 반사됩니다. 순 효과: 목표의 진폭은 증가하고 나머지는 감소합니다.

### 3.4 회로 구현

확산 연산자는 다음과 같이 구현할 수 있습니다:

$$D = H^{\otimes n} (2|0\rangle\langle 0| - I) H^{\otimes n}$$

중간 연산자 $2|0\rangle\langle 0| - I$는 $|0^n\rangle$을 제외한 모든 상태의 위상을 뒤집습니다. 다중 제어 Z 게이트(multi-controlled Z gate)를 사용하여 구성할 수 있습니다.

```python
import numpy as np

def diffusion_operator(n):
    """
    그로버 확산 연산자 D = 2|s><s| - I 구성.

    왜 이 특정한 형태인가? 확산 연산자는 균등 중첩 |s>에 대한
    반사(REFLECTION)입니다. 오라클(목표의 직교 여공간에 대한 반사)과
    결합하면, 목표 상태를 향한 회전(ROTATION)을 만듭니다. 각 반복은
    고정된 각도만큼 회전하여 목표의 진폭이 증가합니다.

    동등하게, D = H^n (2|0><0| - I) H^n이며, 이는 회로로
    구현하는 방법을 보여줍니다: H 게이트, |0...0>에 위상 뒤집기, H 게이트.
    """
    dim = 2**n
    # |s> = 균등 중첩
    s = np.ones(dim, dtype=complex) / np.sqrt(dim)
    # D = 2|s><s| - I
    D = 2 * np.outer(s, s.conj()) - np.eye(dim, dtype=complex)
    return D

# 평균에 대한 반전 시연
n = 3
dim = 2**n
target = 5

print(f"=== Diffusion Operator: Inversion About the Mean ===\n")
print(f"n={n}, target=|{format(target, f'0{n}b')}>\n")

# 균등 중첩에서 시작, 오라클 적용
oracle = grover_oracle(n, [target])
D = diffusion_operator(n)

state = np.ones(dim, dtype=complex) / np.sqrt(dim)
print("Step 0 (uniform):")
mean_amp = np.mean(state.real)
print(f"  Mean amplitude: {mean_amp:.4f}")
print(f"  Target amplitude: {state[target]:.4f}")

# 오라클 적용
state = oracle @ state
print("\nAfter oracle:")
mean_amp = np.mean(state.real)
print(f"  Mean amplitude: {mean_amp:.4f}")
print(f"  Target amplitude: {state[target]:.4f}")
print(f"  Other amplitudes: {state[0]:.4f}")

# 확산 적용
state = D @ state
print("\nAfter diffusion (one Grover iteration):")
mean_amp = np.mean(state.real)
print(f"  Mean amplitude: {mean_amp:.4f}")
print(f"  Target amplitude: {state[target]:.4f}")
print(f"  Other amplitudes: {state[0]:.4f}")
print(f"  P(target) = {abs(state[target])**2:.4f}")
print(f"  P(other) = {abs(state[0])**2:.4f} each")

# 평균에 대한 반전을 명시적으로 표시
print("\n--- Inversion About the Mean (Detailed) ---\n")
# 오라클 후 진폭:
amps = oracle @ (np.ones(dim, dtype=complex) / np.sqrt(dim))
mean = np.mean(amps).real
print(f"After oracle amplitudes: target={amps[target].real:.4f}, "
      f"others={amps[0].real:.4f}")
print(f"Mean = {mean:.4f}")
print(f"Inversion: target -> 2*{mean:.4f} - ({amps[target].real:.4f}) = "
      f"{2*mean - amps[target].real:.4f}")
print(f"Inversion: other  -> 2*{mean:.4f} - ({amps[0].real:.4f}) = "
      f"{2*mean - amps[0].real:.4f}")
```

---

## 4. 알고리즘

### 4.1 그로버 알고리즘

```
1. 초기화: |psi> = H^n |0...0> = (1/sqrt(N)) sum_x |x>
2. k = floor(pi/4 * sqrt(N))번 반복:
   a. 오라클 적용:    O_f |psi>
   b. 확산 적용: D |psi>
3. 모든 큐비트 측정
```

### 4.2 그로버 반복

각 **그로버 반복(Grover iteration)** (또는 "그로버 단계"라고도 함)은 오라클에 이어 확산으로 구성됩니다:

$$G = D \cdot O_f$$

전체 알고리즘은 초기 균등 중첩에 $G$를 정확히 $k$번 적용합니다:

$$|\psi_k\rangle = G^k |s\rangle$$

### 4.3 완전한 구현

```python
import numpy as np

def grovers_algorithm(n, targets, verbose=True):
    """
    완전한 그로버 탐색 알고리즘.

    왜 floor(pi/4 * sqrt(N/M))번 반복하는가? 이것은 상태를
    목표 부분 공간에 가장 가깝게 가져오는 데 필요한 그로버
    회전 수입니다. 더 많은 반복은 지나치게(OVERSHOOT) 되어
    성공 확률을 감소시킵니다. 이것이 고전 탐색과의 중요한
    차이입니다: 그로버에서는 너무 많이 탐색할 수 있습니다!

    매개변수:
        n: 큐비트 수
        targets: 목표 인덱스 목록
        verbose: 단계별 정보 출력

    반환:
        최종 상태, 측정 확률, 성공 확률
    """
    dim = 2**n
    M = len(targets)  # 해의 수

    # 최적 반복 횟수
    k_optimal = int(np.round(np.pi / 4 * np.sqrt(dim / M)))

    # 연산자 구성
    oracle = grover_oracle(n, targets)
    D = diffusion_operator(n)
    G = D @ oracle  # 그로버 반복

    # 균등 중첩 초기화
    state = np.ones(dim, dtype=complex) / np.sqrt(dim)

    if verbose:
        print(f"Grover's Algorithm: n={n}, N={dim}, M={M} solutions")
        print(f"Optimal iterations: k = floor(pi/4 * sqrt({dim}/{M})) = {k_optimal}")
        print(f"Target states: {[format(t, f'0{n}b') for t in targets]}")
        print()

    # 시각화를 위한 진폭 변화 기록
    target_amps = [abs(state[targets[0]])]
    other_amps = [abs(state[0]) if 0 not in targets else abs(state[1])]

    # 그로버 반복 적용
    for i in range(k_optimal):
        state = G @ state
        target_amps.append(abs(state[targets[0]]))
        non_target = 0 if 0 not in targets else 1
        other_amps.append(abs(state[non_target]))

        if verbose and (i < 5 or i == k_optimal - 1):
            p_success = sum(abs(state[t])**2 for t in targets)
            print(f"  Iteration {i+1}: P(target) = {p_success:.6f}")

    # 최종 확률
    probs = np.abs(state)**2
    p_success = sum(probs[t] for t in targets)

    if verbose:
        print(f"\nFinal success probability: {p_success:.6f}")
        print(f"Expected measurement outcome: one of the targets with P = {p_success:.6f}")

    return state, probs, p_success, target_amps, other_amps

# 다양한 크기에 대해 그로버 알고리즘 실행
print("=== Grover's Algorithm Demo ===\n")
for n in [3, 4, 5]:
    target = 2**n - 1  # 마지막 원소 탐색
    state, probs, p_success, _, _ = grovers_algorithm(n, [target])
    print()
```

핵심 루프를 보여주는 간단한 버전:

```python
import numpy as np

# 핵심 그로버 루프 (명확성을 위한 최소 버전)

def grover_search(n, target):
    """
    최소 그로버 알고리즘.

    왜 이렇게 단순한가? 전체 알고리즘은 그냥:
    1. 균등 중첩에서 시작
    2. 반복: 오라클 + 확산
    3. 측정

    수학적 아름다움은 이 두 가지 단순한 반사(reflection)가
    목표의 진폭을 꾸준히 증가시키는 회전으로 합성된다는 것입니다.
    """
    N = 2**n
    k = int(np.pi / 4 * np.sqrt(N))

    # 상태 벡터 (균등하게 시작)
    state = np.ones(N, dtype=complex) / np.sqrt(N)

    # 그로버 반복
    for _ in range(k):
        # 오라클: 목표 위상 뒤집기
        state[target] *= -1

        # 확산: 평균에 대한 반전
        mean = np.mean(state)
        state = 2 * mean - state

    return np.argmax(np.abs(state)**2), np.abs(state)**2

# 테스트
print("=== Minimal Grover Implementation ===\n")
for n in [4, 6, 8, 10]:
    target = 42 % (2**n)  # 임의의 목표
    found, probs = grover_search(n, target)
    print(f"n={n:>2}, N={2**n:>5}, target={target:>4} -> "
          f"found={found:>4}, P(correct)={probs[target]:.4f}, "
          f"correct={found == target}")
```

---

## 5. 기하학적 해석

### 5.1 2차원 부분 공간

핵심 통찰은 그로버 알고리즘이 다음에 의해 생성된 2차원 부분 공간(2-dimensional subspace) 내에서만 동작한다는 것입니다:

$$|w\rangle = \text{목표 상태}$$
$$|w^\perp\rangle = \frac{1}{\sqrt{N-1}} \sum_{x \neq w} |x\rangle = \text{비목표들의 균등 중첩}$$

초기 상태 $|s\rangle$은 다음과 같이 쓸 수 있습니다:

$$|s\rangle = \sin\theta \cdot |w\rangle + \cos\theta \cdot |w^\perp\rangle$$

여기서 $\sin\theta = \frac{1}{\sqrt{N}}$ (균등 중첩에서 목표의 진폭)이고 따라서 큰 $N$에 대해 $\theta = \arcsin(1/\sqrt{N}) \approx 1/\sqrt{N}$.

### 5.2 그로버 반복을 회전으로

각 그로버 반복 $G = D \cdot O_f$은 이 2차원 평면에서 상태 벡터를 $|w\rangle$를 향해 각도 $2\theta$만큼 회전시킵니다:

1. **오라클** ($O_f$): $|w^\perp\rangle$에 대해 반사 ($|w\rangle$ 방향 성분을 뒤집음)
2. **확산** ($D$): $|s\rangle$에 대해 반사

두 반사의 합성은 반사 축 사이의 각도의 두 배만큼의 회전입니다. $|s\rangle$과 $|w^\perp\rangle$ 사이의 각도는 $\theta$이므로 회전 각도는 $2\theta$입니다.

### 5.3 $k$번 반복 후

$k$번 반복 후, 상태는:

$$|s_k\rangle = \sin((2k+1)\theta) |w\rangle + \cos((2k+1)\theta) |w^\perp\rangle$$

성공 확률은:

$$P(\text{목표 찾기}) = \sin^2((2k+1)\theta)$$

$(2k+1)\theta \approx \pi/2$일 때 최대가 되며, $k \approx \frac{\pi}{4\theta} - \frac{1}{2} \approx \frac{\pi}{4}\sqrt{N}$.

```python
import numpy as np

# 그로버 알고리즘의 기하학적 해석

def grover_geometric(N, k_iterations):
    """
    2D 부분 공간에서 그로버 알고리즘의 기하학적 변화를 추적.

    왜 2D인가? 오라클과 확산 연산자는 오직 목표 성분과
    비목표 성분만을 혼합합니다. 다른 어떤 방향으로도 진폭을
    만들지 않습니다. 이것은 전체 알고리즘이 2차원 평면에
    존재하며, 회전으로 기하학적으로 분석하기 쉽다는 것을 의미합니다.
    """
    # 초기 각도
    theta = np.arcsin(1 / np.sqrt(N))

    angles = [(2*0 + 1) * theta]  # |w_perp> 축으로부터의 초기 각도

    for k in range(1, k_iterations + 1):
        angle = (2*k + 1) * theta
        angles.append(angle)

    # 성공 확률
    success_probs = [np.sin(a)**2 for a in angles]
    # {|w>, |w_perp>} 기저에서의 진폭
    target_amps = [np.sin(a) for a in angles]
    other_amps = [np.cos(a) for a in angles]

    return angles, success_probs, target_amps, other_amps

# N=64 (n=6)에 대한 회전 시연
N = 64
n = 6
theta = np.arcsin(1 / np.sqrt(N))
k_opt = int(np.round(np.pi / (4 * theta) - 0.5))

print(f"=== Geometric Interpretation: N={N}, n={n} ===\n")
print(f"Initial angle: theta = arcsin(1/sqrt({N})) = {theta:.4f} rad = {theta*180/np.pi:.2f} deg")
print(f"Rotation per iteration: 2*theta = {2*theta:.4f} rad = {2*theta*180/np.pi:.2f} deg")
print(f"Optimal iterations: k = pi/(4*theta) - 1/2 = {np.pi/(4*theta) - 0.5:.2f} -> {k_opt}\n")

# 변화 추적
angles, probs, t_amps, o_amps = grover_geometric(N, k_opt + 3)

print(f"{'Iteration':>10} {'Angle (deg)':>15} {'P(target)':>12} "
      f"{'amp(target)':>14} {'amp(other)':>14}")
print("-" * 70)

for k in range(len(angles)):
    marker = " <-- optimal" if k == k_opt else ""
    if k > k_opt:
        marker = " (overshooting!)" if probs[k] < probs[k-1] else ""
    print(f"{k:>10} {angles[k]*180/np.pi:>15.2f} {probs[k]:>12.6f} "
          f"{t_amps[k]:>14.6f} {o_amps[k]:>14.6f}{marker}")

print(f"\nKey observations:")
print(f"  1. Each iteration rotates by {2*theta*180/np.pi:.2f} degrees")
print(f"  2. Maximum P(target) at iteration {k_opt} (angle ~90 deg)")
print(f"  3. OVERSHOOTING past the optimal point DECREASES success probability!")
print(f"  4. This is why Grover's algorithm requires knowing when to stop.")
```

---

## 6. 최적 반복 횟수

### 6.1 유도

$P = \sin^2((2k+1)\theta)$를 최대화하려 합니다. 이는 다음 때 최대:

$$(2k+1)\theta = \frac{\pi}{2}$$

$$k = \frac{\pi}{4\theta} - \frac{1}{2} \approx \frac{\pi}{4}\sqrt{N}$$

(큰 $N$에 대해 $\theta \approx 1/\sqrt{N}$이므로).

$k$를 가장 가까운 정수로 반올림합니다. 결과적인 성공 확률은:

$$P_{\max} = \sin^2\left(\frac{\pi}{2} - O(1/\sqrt{N})\right) = 1 - O(1/N)$$

큰 $N$에 대해 이것은 1에 매우 가깝습니다.

### 6.2 과도 반복의 위험

더 많은 반복이 항상 도움이 되는 고전 탐색과 달리, 그로버 알고리즘은 *지나칠(overshoot)* 수 있습니다. 최적 $k$번 반복 후에 추가적인 반복은 $|w\rangle$에서 멀어지는 방향으로 상태를 회전시켜 성공 확률을 감소시킵니다. 이것은 양자 알고리즘의 근본적인 특성입니다.

극단적으로, $\frac{\pi}{2\theta}$번의 반복(최적의 두 배) 후에 상태는 거의 $|w^\perp\rangle$로 다시 회전하며, 성공 확률은 0에 가까워집니다 — 무작위 추측보다 더 나쁩니다!

### 6.3 $N$을 모를 때

일부 응용에서는 해의 수 $M$을 알 수 없습니다. 이를 처리하는 기법들이 있습니다:

1. **지수 탐색(Exponential search)**: $k = 1, 2, 4, 8, \ldots$번의 반복을 시도. 오버헤드는 상수 인자에 불과.
2. **양자 계수(Quantum counting)**: 다음 레슨의 양자 푸리에 변환(quantum Fourier transform)을 사용하여 그로버 실행 전에 $M$을 추정.

```python
import numpy as np

# 최적 반복 횟수와 지나침 시연

print("=== Optimal Iterations and Overshooting ===\n")

for n in [3, 5, 8, 10]:
    N = 2**n
    theta = np.arcsin(1 / np.sqrt(N))
    k_exact = np.pi / (4 * theta) - 0.5
    k_opt = int(np.round(k_exact))

    # 최적 k에서의 성공 확률
    p_opt = np.sin((2*k_opt + 1) * theta)**2

    # 2*k_opt (최적의 두 배)에서의 성공 확률
    k_double = 2 * k_opt
    p_double = np.sin((2*k_double + 1) * theta)**2

    # 최적의 절반에서의 성공 확률
    k_half = max(1, k_opt // 2)
    p_half = np.sin((2*k_half + 1) * theta)**2

    print(f"N = 2^{n} = {N:>6}")
    print(f"  k_optimal = {k_opt} (exact: {k_exact:.2f})")
    print(f"  P(k={k_half:>3}) = {p_half:.6f}  (under-iterating)")
    print(f"  P(k={k_opt:>3}) = {p_opt:.6f}  (optimal)")
    print(f"  P(k={k_double:>3}) = {p_double:.6f}  (over-iterating!)")
    print()

# 전체 확률 진동 표시
print("=== Full Probability Oscillation (N=256, n=8) ===\n")
N = 256
n = 8
theta = np.arcsin(1 / np.sqrt(N))
k_opt = int(np.round(np.pi / (4 * theta) - 0.5))

print(f"{'Iteration':>10} {'P(target)':>12} {'Status':>20}")
print("-" * 45)

for k in range(0, 3*k_opt + 1):
    p = np.sin((2*k + 1) * theta)**2
    if k == k_opt:
        status = "OPTIMAL"
    elif k == 0:
        status = "initial (1/N)"
    elif abs(p - 1/N) < 0.01:
        status = "back to ~1/N"
    elif p < 0.01:
        status = "near ZERO!"
    else:
        status = ""
    print(f"{k:>10} {p:>12.6f} {status:>20}")
```

---

## 7. 다중 해

### 7.1 일반화

$M$개의 목표 원소($M$개의 해)가 있는 경우, 알고리즘은 자연스럽게 일반화됩니다:

$$\sin\theta_M = \sqrt{\frac{M}{N}}$$

최적 반복 횟수는:

$$k_{\text{opt}} = \left\lfloor \frac{\pi}{4} \sqrt{\frac{N}{M}} \right\rfloor$$

성공 확률은 1에 가깝게 유지됩니다.

### 7.2 특수 경우

| $M$ | $k_{\text{opt}}$ | 성공 확률 | 비고 |
|-----|:---:|:---:|------|
| 1 | $\sim \frac{\pi}{4}\sqrt{N}$ | $\sim 1$ | 표준 경우 |
| $N/4$ | $\sim 1$ | $\sim 1$ | 적은 반복 필요 |
| $N/2$ | $\sim 1$ | 정확히 1 | 한 번의 반복으로 충분 |
| $> N/2$ | 0 | $> 1/2$ | 무작위 추측도 이미 작동 |

### 7.3 해의 수를 모를 때

$M$을 알 수 없을 때는 $k_{\text{opt}}$를 직접 계산할 수 없습니다. 지수 탐색 전략이 작동합니다:

1. $\lambda$가 1에서 시작할 때 $\{1, 2, \ldots, \lambda\}$에서 무작위 $k$ 선택
2. $k$번의 반복으로 그로버를 실행하고 측정
3. 측정이 해를 주면 중단
4. 그렇지 않으면 $\lambda \to \min(2\lambda, \sqrt{N})$으로 설정하고 반복

이것은 높은 확률로 총 $O(\sqrt{N/M})$번의 질의로 해를 찾습니다.

```python
import numpy as np

# 다중 해를 가진 그로버 알고리즘

def grovers_multi_solution(n, targets):
    """
    다중 목표 상태를 가진 그로버 알고리즘.

    왜 더 많은 해가 더 적은 반복을 의미하는가? 각 해는
    '목표 부분 공간'에 기여합니다. 더 많은 해는 초기 상태가
    이 부분 공간과 더 큰 겹침을 가짐을 의미하며 (더 큰 theta),
    따라서 도달하는 데 더 적은 회전이 필요합니다.
    """
    N = 2**n
    M = len(targets)
    theta = np.arcsin(np.sqrt(M / N))
    k_opt = max(0, int(np.round(np.pi / (4 * theta) - 0.5)))

    # 연산자 구성
    oracle = grover_oracle(n, targets)
    D = diffusion_operator(n)
    G = D @ oracle

    # 초기화
    state = np.ones(N, dtype=complex) / np.sqrt(N)

    # 반복
    for _ in range(k_opt):
        state = G @ state

    # 성공 확률
    p_success = sum(abs(state[t])**2 for t in targets)

    return k_opt, p_success

# 다양한 해의 수로 테스트
print("=== Multiple Solutions ===\n")
n = 8
N = 2**n
print(f"n={n}, N={N}\n")

print(f"{'M solutions':>12} {'k_opt':>8} {'P(success)':>12} {'Queries saved vs N/2':>22}")
print("-" * 58)

for M in [1, 2, 4, 8, 16, 32, 64, 128]:
    k, p = grovers_multi_solution(n, list(range(M)))
    classical = N // (2 * M)  # M개 목표 중 하나를 찾기 위한 예상 고전 질의 수
    print(f"{M:>12} {k:>8} {p:>12.6f} {f'{classical}/{k} = {classical/max(k,1):.1f}x':>22}")

print(f"\nMore solutions -> fewer iterations (larger theta -> fewer rotations needed)")
```

---

## 8. 진폭 변화 시각화

```python
import numpy as np

# 그로버 알고리즘 진폭 변화의 포괄적 시각화

def grover_visualization(n, target, extra_iterations=3):
    """
    그로버 반복을 통해 진폭이 어떻게 변하는지 시각화.

    왜 진폭을 시각화하는가? 그로버 알고리즘의 핵심 메커니즘은
    반복적인 오라클 + 확산 연산을 통한 목표 진폭의 점진적 증폭입니다.
    반복마다 진폭이 변하는 것을 보면 다음에 대한 직관을 키울 수 있습니다:
    1. 알고리즘이 작동하는 이유 (목표에서 보강 간섭)
    2. 최적 반복 횟수가 있는 이유 (지나침)
    3. 사인파 진동 패턴
    """
    N = 2**n
    theta = np.arcsin(1 / np.sqrt(N))
    k_opt = int(np.round(np.pi / (4 * theta) - 0.5))
    total_iterations = k_opt + extra_iterations

    # 변화 추적
    oracle = grover_oracle(n, [target])
    D = diffusion_operator(n)
    G = D @ oracle

    state = np.ones(N, dtype=complex) / np.sqrt(N)
    history = [state.copy()]

    for k in range(total_iterations):
        state = G @ state
        history.append(state.copy())

    # 진폭 막대 출력 (텍스트 기반 시각화)
    print(f"=== Grover's Algorithm Amplitude Evolution ===")
    print(f"    n={n}, N={N}, target=|{format(target, f'0{n}b')}>, "
          f"k_opt={k_opt}")
    print()

    bar_width = 50

    for k in range(len(history)):
        state = history[k]
        p_target = abs(state[target])**2
        p_other = abs(state[0 if 0 != target else 1])**2

        # 목표를 위한 막대 생성
        target_bar_len = int(p_target * bar_width)
        other_bar_len = int(p_other * bar_width * N)  # 비목표를 보기 위해 스케일 업

        marker = " <-- OPTIMAL" if k == k_opt + 1 else ""
        if k == 0:
            label = "Initial"
        else:
            label = f"Iter {k:>2}"

        target_bar = "#" * target_bar_len + "." * (bar_width - target_bar_len)
        print(f"  {label}: P(target)={p_target:.4f} [{target_bar}]{marker}")

    print()

    # 요약 통계
    print("=== Summary ===\n")
    print(f"{'Iteration':>10} {'P(target)':>12} {'P(each other)':>15} {'Amp(target)':>14}")
    print("-" * 55)
    for k, state in enumerate(history):
        p_t = abs(state[target])**2
        non_target = 0 if 0 != target else 1
        p_o = abs(state[non_target])**2
        a_t = state[target].real
        print(f"{k:>10} {p_t:>12.6f} {p_o:>15.8f} {a_t:>14.6f}")

    return history

# 다양한 크기에 대한 시각화 실행
history = grover_visualization(n=4, target=7, extra_iterations=4)

print("\n\n")
history = grover_visualization(n=6, target=42, extra_iterations=5)

# 사인파 패턴 검증
print("\n\n=== Verifying Sinusoidal Pattern ===\n")
n = 8
N = 2**n
target = 100
theta = np.arcsin(1 / np.sqrt(N))

oracle = grover_oracle(n, [target])
D = diffusion_operator(n)
G = D @ oracle
state = np.ones(N, dtype=complex) / np.sqrt(N)

print(f"n={n}, N={N}, theta={theta:.6f}")
print(f"\n{'Iter':>5} {'Simulated P':>14} {'Theoretical P':>16} {'Match':>8}")
print("-" * 48)

for k in range(int(2 * np.pi / (2 * theta))):
    p_sim = abs(state[target])**2
    p_theory = np.sin((2*k + 1) * theta)**2
    match = np.isclose(p_sim, p_theory, atol=1e-6)
    if k <= 15 or k % 5 == 0:
        print(f"{k:>5} {p_sim:>14.8f} {p_theory:>16.8f} {'YES' if match else 'NO':>8}")
    state = G @ state

print("\nThe simulation perfectly matches the theoretical sin^2 formula!")
```

---

## 9. 연습 문제

### 연습 1: 기본 그로버

$n = 4$ 큐비트 ($N = 16$개 원소)에 대한 그로버 알고리즘을 구현하세요.

a) 목표 $w = 7$을 탐색하세요. 몇 번의 반복이 필요한가요? 최종 성공 확률은 얼마인가요?
b) 0, 1, 2, ..., 10번의 반복에 대해 알고리즘을 실행하고 반복 횟수 대비 성공 확률을 그래프로 나타내세요.
c) 성공 확률이 사인파적으로 진동하는지 검증하세요.

### 연습 2: 오라클 구성

3-큐비트 시스템 ($N = 8$)에 대해 다음에 대한 그로버 오라클을 구성하세요:

a) 단일 목표 $w = 5$ (이진: 101)
b) 두 목표 $w \in \{3, 5\}$
c) 조건으로 정의된 목표: $f(x) = 1$ if $x$에 정확히 두 개의 1 비트가 있는 경우

각각에 대해, 오라클을 직접적인 대각 행렬이 아닌 회로(게이트 시퀀스)로 구현하세요. (힌트: 다중 제어 Z 게이트(multi-controlled Z gates)를 사용하세요.)

### 연습 3: 진폭 증폭

그로버 알고리즘은 **진폭 증폭(amplitude amplification)**이라는 더 일반적인 기법의 특수 경우입니다. 일반 버전에서 초기 상태는 균등할 필요가 없습니다:

a) 상태 $\frac{1}{\sqrt{3}}|0\rangle + \sqrt{\frac{2}{3}}|1\rangle$ (1-큐비트 시스템)에서 시작하여 *이* 상태에 대한 확산 연산자를 적용하세요. 어떤 일이 일어나나요?
b) 일반화: 2-큐비트 시스템에 대해 균등하지 않은 초기 상태를 준비하고 이를 그로버 알고리즘의 "시작 중첩"으로 사용하세요. 알고리즘이 여전히 작동하나요? 목표의 초기 진폭이 필요한 반복 횟수에 어떤 영향을 미치나요?

### 연습 4: SAT에의 응용

불리언 충족 가능성(SAT) 문제: $x = (x_2, x_1, x_0) \in \{0,1\}^3$에서 다음을 만족하는 $x$를 찾으세요:

$$(x_0 \lor x_1) \land (\lnot x_1 \lor x_2) \land (x_0 \lor \lnot x_2)$$

a) 8개의 가능한 입력 모두를 열거하고 어느 것이 공식을 만족하는지 결정하세요.
b) 만족하는 배정(satisfying assignment)을 표시하는 그로버 오라클을 구성하세요.
c) 그로버 알고리즘을 실행하고 만족하는 배정을 찾는지 검증하세요.
d) $M$개의 만족하는 배정이 있다면, 최적의 그로버 반복 횟수는 얼마인가요?

### 연습 5: 노이즈가 있는 그로버 알고리즘

실제 양자 컴퓨터에서는 게이트가 불완전합니다. 노이즈의 효과를 시뮬레이션하세요:

a) 각 그로버 반복 후 각 큐비트에 무작위 위상 오류를 추가하세요: $|x\rangle \to e^{i\epsilon_x}|x\rangle$, 여기서 $\epsilon_x$는 $\mathcal{N}(0, \sigma^2)$에서 추출됩니다.
b) $\sigma = 0, 0.01, 0.05, 0.1, 0.5$에 대해 노이즈가 있는 알고리즘을 실행하고 성공 확률이 어떻게 저하되는지 그래프로 나타내세요.
c) 어느 노이즈 수준에서 고전 탐색에 대한 양자 우위가 사라지나요?

---

[<- 이전: 도이치-조자 알고리즘](07_Deutsch_Jozsa_Algorithm.md) | [다음: 양자 푸리에 변환 ->](09_Quantum_Fourier_Transform.md)
