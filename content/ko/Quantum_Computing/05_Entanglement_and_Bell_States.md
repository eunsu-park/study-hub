# 레슨 5: 얽힘과 벨 상태

[<- 이전: 양자 회로](04_Quantum_Circuits.md) | [다음: 양자 측정 ->](06_Quantum_Measurement.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 수학적 기준을 사용하여 분리 가능한(곱) 상태와 얽힌 양자 상태를 구별
2. 네 가지 벨 상태(Bell State)를 모두 구성하고 식별하며 물리적 의미를 설명
3. 양자 회로에서 아다마르(Hadamard) 게이트와 CNOT 게이트를 사용하여 벨 상태를 구성
4. EPR 역설과 양자역학을 통한 해결 방법을 설명
5. 벨 정리(Bell's Theorem)를 서술하고 CHSH 부등식이 국소 숨은 변수 이론(local hidden variable theory)을 어떻게 배제하는지 설명
6. 얽힘의 단일성(monogamy of entanglement)과 양자 정보에 대한 그 함의를 설명
7. Python에서 얽힌 상태와 벨 부등식 검사를 시뮬레이션

---

얽힘(Entanglement)은 아마도 양자역학에서 가장 심오하게 비고전적인 특성일 것입니다. 두 개 이상의 큐비트가 얽히면, 그 양자 상태들은 고전적인 설명으로는 불가능한 방식으로 상관관계를 가지게 됩니다 -- 한 큐비트를 측정하면 두 입자 사이의 물리적 거리에 관계없이 즉각적으로 다른 큐비트의 상태가 결정됩니다. 아인슈타인은 이것을 유명하게 "유령 같은 원격 작용(spooky action at a distance)"이라 불렀으며, 그를 깊이 불안하게 만들었습니다.

얽힘은 단순한 호기심거리가 아니라 양자 컴퓨팅의 이점을 구동하는 엔진입니다. 얽힘 없이는 양자 컴퓨터가 고전적인 확률적 기계보다 더 강력할 수 없습니다. 지수적 속도 향상을 달성하는 모든 양자 알고리즘은 얽힌 상태를 생성하고 조작함으로써 그렇게 합니다. 이 레슨에서는 가장 기본적인 예시인 벨 상태(Bell States)를 통해 얽힘을 연구하고, 얽힘이 진정으로 비고전적임을 증명하는 이론적 프레임워크를 살펴봅니다.

> **비유:** 얽힌 큐비트는 한 쌍의 마법 주사위와 같습니다 -- 아무리 멀리 떨어져 있어도, 하나가 6을 보이면 다른 하나는 항상 1을 보입니다. 결과들은 고전 물리학으로는 설명할 수 없는 방식으로 상관관계가 있습니다. 분리하기 전에 주사위에 어떤 "숨겨진 지시사항"을 적어 두려 해도, 양자 상관관계는 고전적 상관관계보다 증명 가능하게 더 강합니다.

## 목차

1. [분리 가능한 상태 대 얽힌 상태](#1-분리-가능한-상태-대-얽힌-상태)
2. [네 가지 벨 상태](#2-네-가지-벨-상태)
3. [벨 상태 생성](#3-벨-상태-생성)
4. [EPR 역설](#4-epr-역설)
5. [벨 정리와 CHSH 부등식](#5-벨-정리와-chsh-부등식)
6. [얽힘의 단일성](#6-얽힘의-단일성)
7. [자원으로서의 얽힘](#7-자원으로서의-얽힘)
8. [연습 문제](#8-연습-문제)

---

## 1. 분리 가능한 상태 대 얽힌 상태

### 1.1 정의

두 큐비트 순수 상태 $|\psi\rangle_{AB}$가 단일 큐비트 상태의 텐서 곱으로 쓸 수 있을 때 **분리 가능(separable)**하다(또는 **곱 상태(product state)**라고도 한다)고 합니다:

$$|\psi\rangle_{AB} = |\alpha\rangle_A \otimes |\beta\rangle_B$$

그러한 인수분해가 존재하지 않으면, 상태는 **얽혀 있습니다(entangled)**.

### 1.2 예시: 분리 가능한 상태

$$|+\rangle \otimes |0\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

이것은 인수들을 식별할 수 있으므로 분리 가능합니다: 큐비트 A는 $|+\rangle$ 상태에, 큐비트 B는 $|0\rangle$ 상태에 있습니다. 큐비트 A를 측정해도 큐비트 B에 아무런 영향이 없습니다.

### 1.3 예시: 얽힌 상태

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$$

이것을 $|\alpha\rangle \otimes |\beta\rangle$로 쓸 수 있을까요? 시도해 봅시다:

$$(\alpha_0|0\rangle + \alpha_1|1\rangle) \otimes (\beta_0|0\rangle + \beta_1|1\rangle) = \alpha_0\beta_0|00\rangle + \alpha_0\beta_1|01\rangle + \alpha_1\beta_0|10\rangle + \alpha_1\beta_1|11\rangle$$

이것이 $\frac{1}{\sqrt{2}}(|00\rangle + |11\rangle)$와 같으려면 다음이 필요합니다:
- $\alpha_0\beta_0 = \frac{1}{\sqrt{2}}$ ($|00\rangle$의 계수)
- $\alpha_0\beta_1 = 0$ ($|01\rangle$의 계수)
- $\alpha_1\beta_0 = 0$ ($|10\rangle$의 계수)
- $\alpha_1\beta_1 = \frac{1}{\sqrt{2}}$ ($|11\rangle$의 계수)

두 번째 방정식에서, $\alpha_0 = 0$ 또는 $\beta_1 = 0$ 이어야 합니다. 그러나 $\alpha_0 = 0$이면 첫 번째 방정식이 실패합니다. $\beta_1 = 0$이면 네 번째 방정식이 실패합니다. 해가 없습니다. 이 상태는 얽혀 있습니다.

### 1.4 슈미트 분해 검사

$|\psi\rangle = \sum_{ij} c_{ij}|ij\rangle$로 쓰인 두 큐비트 상태에 대해, 계수들을 행렬 $C$로 배열할 수 있습니다:

$$C = \begin{pmatrix} c_{00} & c_{01} \\ c_{10} & c_{11} \end{pmatrix}$$

$\text{rank}(C) = 1$인 경우에만, 즉 $\det(C) = 0$에 해당할 때에만 상태가 분리 가능합니다.

얽힌 상태의 경우, **슈미트 순위(Schmidt rank)**는 2이며, **슈미트 계수(Schmidt coefficients)**가 얽힘의 정도를 정량화합니다.

```python
import numpy as np

# 분리 가능한 상태 대 얽힌 상태

def check_entanglement(state_2q, label=""):
    """
    2큐비트 상태가 분리 가능한지 얽혀 있는지 판별합니다.

    왜 행렬식을 사용할까요? 2큐비트 상태 |psi> = sum c_ij |ij>는
    계수 행렬 C의 랭크가 1인 경우에만 분리 가능하며, 이는
    det(C) = 0을 의미합니다. 행렬식이 0이 아니면 얽힘을 의미합니다.

    더 정량적인 척도를 위해 슈미트 분해를 사용합니다.
    """
    # 상태 벡터를 2x2 계수 행렬로 재배열
    C = state_2q.reshape(2, 2)

    # 행렬식 검사
    det = np.linalg.det(C)
    is_entangled = not np.isclose(abs(det), 0)

    # SVD를 통한 슈미트 분해
    # 특이값이 슈미트 계수입니다
    U, s, Vh = np.linalg.svd(C)
    schmidt_coeffs = s[s > 1e-10]  # 0이 아닌 슈미트 계수

    # 얽힘 엔트로피: S = -sum(lambda^2 * log(lambda^2))
    probs = schmidt_coeffs**2
    entropy = -np.sum(probs * np.log2(probs + 1e-15))

    print(f"상태: {label}")
    print(f"  계수 행렬:\n    {C[0]}\n    {C[1]}")
    print(f"  |det(C)| = {abs(det):.6f}")
    print(f"  슈미트 계수: {schmidt_coeffs}")
    print(f"  얽힘 엔트로피: {entropy:.4f} bits")
    print(f"  {'얽힘(ENTANGLED)' if is_entangled else '분리 가능(SEPARABLE)'}")
    print()

# 다양한 상태 검사
print("=== 얽힘 분석 ===\n")

# 분리 가능: |+>|0> = (|00> + |10>)/sqrt(2)
check_entanglement(
    np.array([1, 0, 1, 0], dtype=complex) / np.sqrt(2),
    "|+>|0>"
)

# 얽힘: 벨 상태 |Phi+> = (|00> + |11>)/sqrt(2)
check_entanglement(
    np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    "|Phi+> = (|00>+|11>)/sqrt(2)"
)

# 부분 얽힘: sqrt(3/4)|00> + sqrt(1/4)|11>
check_entanglement(
    np.array([np.sqrt(3/4), 0, 0, np.sqrt(1/4)], dtype=complex),
    "sqrt(3/4)|00> + sqrt(1/4)|11>"
)

# 곱 상태: |+>|->
check_entanglement(
    np.kron(
        np.array([1, 1], dtype=complex) / np.sqrt(2),
        np.array([1, -1], dtype=complex) / np.sqrt(2)
    ),
    "|+>|-> (곱 상태)"
)
```

---

## 2. 네 가지 벨 상태

### 2.1 정의

네 가지 **벨 상태(Bell States)**(EPR 쌍 또는 벨 기저라고도 함)는 최대로 얽힌 두 큐비트 상태입니다:

$$|\Phi^+\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) \qquad |\Phi^-\rangle = \frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$$

$$|\Psi^+\rangle = \frac{1}{\sqrt{2}}(|01\rangle + |10\rangle) \qquad |\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$$

### 2.2 속성

네 가지 벨 상태 모두 다음 속성을 공유합니다:

1. **최대 얽힘(Maximally entangled)**: 얽힘 엔트로피는 1 비트입니다(2큐비트 시스템에서의 최대값).
2. **정규 직교(Orthonormal)**: $\langle \Phi^+|\Phi^-\rangle = 0$ 등. 4차원 두 큐비트 공간의 완전한 기저를 형성합니다.
3. **국소 측정이 최대로 무작위**: 어느 한 큐비트만 측정하면 50/50 결과를 얻으며, 상태에 대한 정보를 얻을 수 없습니다.
4. **완전한 상관관계**: 두 큐비트를 같은 기저에서 측정하면 항상 상관된 결과를 얻습니다.

### 2.3 상관관계 패턴

| 벨 상태 | Z 기저 상관관계 | X 기저 상관관계 |
|---------|----------------|----------------|
| $\|\Phi^+\rangle$ | 같음: 둘 다 0 또는 둘 다 1 | 같음: 둘 다 + 또는 둘 다 - |
| $\|\Phi^-\rangle$ | 같음 | 반대 |
| $\|\Psi^+\rangle$ | 반대: 하나는 0, 하나는 1 | 같음 |
| $\|\Psi^-\rangle$ | 반대 | 반대 |

### 2.4 대안적 기저로서의 벨 기저

$\{|00\rangle, |01\rangle, |10\rangle, |11\rangle\}$이 2큐비트의 계산 기저인 것처럼, $\{|\Phi^+\rangle, |\Phi^-\rangle, |\Psi^+\rangle, |\Psi^-\rangle\}$도 동등하게 유효한 기저입니다. 임의의 두 큐비트 상태는 벨 상태의 중첩으로 쓸 수 있습니다. "벨 측정(Bell measurement)"은 이 기저로 투영하며, 양자 텔레포테이션(quantum teleportation)과 초밀도 부호화(dense coding)의 핵심 연산입니다.

```python
import numpy as np

# 네 가지 벨 상태와 그 속성

# 벨 상태 정의
bell_states = {
    "|Phi+>": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    "|Phi->": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
    "|Psi+>": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
    "|Psi->": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
}

# 정규 직교성 검증
print("=== 벨 상태 정규 직교성 ===\n")
names = list(bell_states.keys())
states = list(bell_states.values())
for i in range(4):
    for j in range(4):
        overlap = np.vdot(states[i], states[j])
        if abs(overlap) > 1e-10:
            print(f"  <{names[i]}|{names[j]}> = {overlap:.4f}")

# 상관관계 분석: Z 기저에서 두 큐비트 측정
print("\n=== 상관관계 패턴 (Z 기저, 10000회 시행) ===\n")
np.random.seed(42)

for name, state in bell_states.items():
    probs = np.abs(state)**2  # |00>, |01>, |10>, |11>의 확률
    outcomes = np.random.choice(4, size=10000, p=probs)

    # 상관관계 계산
    same = sum(1 for o in outcomes if o in [0, 3])    # |00> 또는 |11> -> 같음
    diff = sum(1 for o in outcomes if o in [1, 2])     # |01> 또는 |10> -> 다름
    print(f"{name}: 같음={same/100:.1f}%, 다름={diff/100:.1f}%  "
          f"({'상관(correlated)' if same > diff else '반상관(anti-correlated)'})")

# X 기저에서의 상관관계
print("\n=== 상관관계 패턴 (X 기저, 10000회 시행) ===\n")

# X 기저 측정: Z 기저 측정 전 각 큐비트에 H 적용
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
H_H = np.kron(H, H)  # 두 큐비트 모두에 H

for name, state in bell_states.items():
    # X 기저로 변환
    state_x = H_H @ state
    probs = np.abs(state_x)**2
    outcomes = np.random.choice(4, size=10000, p=probs)

    same = sum(1 for o in outcomes if o in [0, 3])
    diff = sum(1 for o in outcomes if o in [1, 2])
    print(f"{name}: 같음={same/100:.1f}%, 다름={diff/100:.1f}%  "
          f"({'상관(correlated)' if same > diff else '반상관(anti-correlated)'})")
```

---

## 3. 벨 상태 생성

### 3.1 벨 회로

네 가지 벨 상태 모두 아다마르 게이트(Hadamard gate)와 CNOT만을 사용하여 계산 기저 상태에서 생성할 수 있습니다:

```
q0: ─[H]─●─
          │
q1: ──────X─
```

대응 관계:

| 입력 | 출력 |
|------|------|
| $\|00\rangle$ | $\|\Phi^+\rangle = \frac{1}{\sqrt{2}}(\|00\rangle + \|11\rangle)$ |
| $\|01\rangle$ | $\|\Psi^+\rangle = \frac{1}{\sqrt{2}}(\|01\rangle + \|10\rangle)$ |
| $\|10\rangle$ | $\|\Phi^-\rangle = \frac{1}{\sqrt{2}}(\|00\rangle - \|11\rangle)$ |
| $\|11\rangle$ | $\|\Psi^-\rangle = \frac{1}{\sqrt{2}}(\|01\rangle - \|10\rangle)$ |

### 3.2 단계별 유도

$|00\rangle$에서 시작합니다:

**단계 1**: 큐비트 0에 H 적용:

$$H \otimes I |00\rangle = \frac{1}{\sqrt{2}}(|0\rangle + |1\rangle) \otimes |0\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)$$

**단계 2**: CNOT 적용 (제어=q0, 목표=q1):

$$\text{CNOT}\left[\frac{1}{\sqrt{2}}(|00\rangle + |10\rangle)\right] = \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle$$

CNOT은 제어 큐비트의 값을 목표에 복사합니다(목표가 $|0\rangle$에서 시작할 때), 얽힘의 특징적인 상관관계를 만들어 냅니다.

### 3.3 역과정: 벨 측정

벨 기저에서 측정하려면 회로를 역으로 실행합니다:

```
q0: ─●─[H]─[M]─
     │
q1: ─X─────[M]─
```

CNOT을 적용한 후 H를 적용하고, 계산 기저에서 측정합니다. 측정 결과가 입력이 어느 벨 상태였는지 알려줍니다.

```python
import numpy as np

# 단계별 벨 상태 생성

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
I2 = np.eye(2, dtype=complex)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

# 큐비트 0에 H, 큐비트 1에 항등 연산
# 규약: |q1 q0>, 따라서 q0에서 H = kron(I, H)
H_I = np.kron(I2, H)

# 벨 회로 유니터리
Bell_circuit = CNOT @ H_I

print("=== 벨 상태 생성 ===\n")

# 네 가지 계산 기저 입력 모두에 적용
inputs = {
    "|00>": np.array([1, 0, 0, 0], dtype=complex),
    "|01>": np.array([0, 1, 0, 0], dtype=complex),
    "|10>": np.array([0, 0, 1, 0], dtype=complex),
    "|11>": np.array([0, 0, 0, 1], dtype=complex),
}

bell_names = {
    (1, 0, 0, 1): "|Phi+>",
    (0, 1, 1, 0): "|Psi+>",
    (1, 0, 0, -1): "|Phi->",
    (0, 1, -1, 0): "|Psi->",
}

for in_name, in_state in inputs.items():
    out_state = Bell_circuit @ in_state

    # 중간 단계: H 이후
    after_H = H_I @ in_state

    print(f"입력: {in_name}")
    print(f"  q0에 H 적용 후: {np.round(after_H, 4)}")
    print(f"  CNOT 적용 후:   {np.round(out_state, 4)}")

    # 어느 벨 상태인지 식별
    rounded = tuple(np.round(out_state * np.sqrt(2)).real.astype(int))
    bell_name = bell_names.get(rounded, "알 수 없음")
    print(f"  = {bell_name}")
    print()

# 벨 회로가 유니터리인지 검증
print(f"벨 회로가 유니터리인가? {np.allclose(Bell_circuit @ Bell_circuit.conj().T, np.eye(4))}")

# 역 벨 회로 (벨 측정용)
Bell_inverse = H_I @ CNOT  # 역순: 먼저 CNOT, 그 다음 H

print(f"\n=== 벨 측정 (역 회로) ===\n")
bell_states_map = {
    "|Phi+>": np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2),
    "|Phi->": np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2),
    "|Psi+>": np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2),
    "|Psi->": np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2),
}

for bell_name, bell_state in bell_states_map.items():
    measured = Bell_inverse @ bell_state
    # 어떤 계산 기저 상태를 얻는지 찾기
    for i in range(4):
        if abs(measured[i]) > 0.9:
            label = format(i, '02b')
            print(f"{bell_name} -> |{label}> (측정 결과: {label})")
```

---

## 4. EPR 역설

### 4.1 아인슈타인, 포돌스키, 로젠 (1935)

1935년 유명한 논문에서, 아인슈타인, 포돌스키, 로젠(EPR)은 양자역학이 *불완전*해야 한다고 주장했습니다. 그들의 논리:

1. **전제**: 시스템을 교란하지 않고 어떤 물리량의 값을 확실하게 예측할 수 있다면, 그 양에 해당하는 "물리적 실재의 요소(element of physical reality)"가 존재합니다.

2. **설정**: 두 입자를 얽힌 상태 $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$으로 준비하고 멀리 보냅니다.

3. **관찰**: $Z$ 기저에서 입자 A를 측정하여 $|0\rangle$을 얻으면, 입자 B를 건드리지 않고도 B가 $|1\rangle$ 상태임을 확실히 알 수 있습니다. 전제에 따르면, B의 $Z$ 값은 실재의 요소입니다.

4. **마찬가지로**: 대신 $X$ 기저에서 A를 측정하여 $|+\rangle$을 얻으면, B가 $|-\rangle$임을 압니다. 따라서 B의 $X$ 값도 실재의 요소입니다.

5. **역설**: $Z$ 값과 $X$ 값 모두 입자 B의 "실재의 요소"이지만, 양자역학에서는 큐비트가 $Z$와 $X$ 모두에 대해 동시에 확정된 값을 가질 수 없다고 합니다(이들은 상보적(complementary) 관측량입니다). 따라서 양자역학은 불완전해야 합니다.

### 4.2 숨은 변수 가설

EPR의 해결책은 **숨은 변수(hidden variables)**가 존재해야 한다는 것이었습니다 -- 양자 상태로 포착되지 않는 추가 정보 -- 가 모든 측정 결과를 미리 결정한다는 것입니다. 입자들은 생성되는 순간부터 이 숨겨진 정보를 가지고 다니며, 마치 미리 쓰인 지시사항을 담은 봉인된 봉투와 같습니다.

### 4.3 해결

30년 동안, EPR의 주장은 과학적이라기보다는 철학적인 것처럼 보였습니다 -- 숨은 변수가 존재하는지 검증할 방법이 없었습니다. 그러다 1964년, 존 벨이 모든 것을 바꾸는 놀라운 정리를 증명했습니다(섹션 5).

---

## 5. 벨 정리와 CHSH 부등식

### 5.1 벨 정리 (비공식적 서술)

**어떤 국소 숨은 변수 이론도 양자역학의 모든 예측을 재현할 수 없습니다.**

이는 다음을 의미합니다: (1) 측정 결과가 숨은 변수에 의해 미리 결정되고 (2) 한 입자의 측정이 다른 입자에 즉각적으로 영향을 미칠 수 없다(국소성(locality))고 가정하면, 측정 상관관계가 만족해야 하는 특정 *부등식*이 있습니다. 양자역학은 이 부등식을 *위반*하고, 실험은 양자 예측을 확인합니다.

### 5.2 CHSH 부등식

클라우저-혼-시모니-홀트(CHSH) 부등식은 벨 정리의 가장 실용적인 형태입니다. 각각 두 가지 측정 설정 중 하나를 선택하는 두 당사자 앨리스와 밥을 고려합니다:

- 앨리스는 $A_0$ 또는 $A_1$을 측정합니다(두 가지 다른 관측량)
- 밥은 $B_0$ 또는 $B_1$을 측정합니다(두 가지 다른 관측량)

각 측정은 $+1$ 또는 $-1$의 결과를 냅니다.

**CHSH 양**을 정의합니다:

$$S = \langle A_0 B_0 \rangle + \langle A_0 B_1 \rangle + \langle A_1 B_0 \rangle - \langle A_1 B_1 \rangle$$

여기서 $\langle A_i B_j \rangle$는 앨리스와 밥 결과의 평균 곱입니다.

### 5.3 고전적 한계

어떤 국소 숨은 변수 이론에 대해서도:

$$|S| \leq 2$$

이것이 **CHSH 부등식**입니다. 입자들이 가질 수 있는 *어떤* 미리 결정된 전략에 대해서도 성립합니다.

### 5.4 양자 위반

양자역학은 최적 측정 선택으로 벨 상태 $|\Phi^+\rangle$에 대해 다음을 예측합니다:

$$S_{\text{quantum}} = 2\sqrt{2} \approx 2.828$$

이는 고전적 한계인 2를 초과하여, 어떤 국소 숨은 변수 이론도 양자 상관관계를 설명할 수 없음을 증명합니다. 최적 측정 설정은:

- 앨리스: $A_0 = Z$, $A_1 = X$
- 밥: $B_0 = \frac{Z + X}{\sqrt{2}}$, $B_1 = \frac{Z - X}{\sqrt{2}}$

(밥은 앨리스의 축에서 45도 회전한 축을 따라 측정합니다.)

### 5.5 치렐손 한계

양자역학으로 달성 가능한 $|S|$의 최대값은 정확히 $2\sqrt{2}$입니다(치렐손 한계(Tsirelson's bound)). 흥미롭게도, 수학적 최대값(물리적 제약 없이)은 4이지만, 고전도 양자도 이에 도달할 수 없습니다.

$$2 < 2\sqrt{2} < 4$$

$$\text{고전} < \text{양자} < \text{대수적 최대값}$$

```python
import numpy as np

# CHSH 부등식 시뮬레이션

def chsh_experiment(state_2q, alice_obs, bob_obs, n_trials=100000):
    """
    CHSH 실험을 시뮬레이션합니다.

    왜 이것을 시뮬레이션할까요? CHSH 부등식은 양자역학과 고전 숨은 변수
    이론을 구별하는 가장 중요한 검사입니다. 수치적으로 시뮬레이션하면
    양자 위반을 검증할 수 있습니다.

    alice_obs, bob_obs: 각각 2개의 2x2 관측량 행렬 목록
    각 관측량은 고유값 +1과 -1을 가집니다.
    """
    results = {}

    for i, A in enumerate(alice_obs):
        for j, B in enumerate(bob_obs):
            # 결합 관측량 A 텐서 B
            AB = np.kron(A, B)

            # 기댓값: <psi|A(x)B|psi>
            # 왜 이 공식을 사용할까요? 순수 상태에서 관측량의 기댓값은
            # <psi|O|psi>입니다. 두 국소 관측량의 곱에 대해 O = A 텐서 B입니다.
            expectation = np.real(np.vdot(state_2q, AB @ state_2q))
            results[(i, j)] = expectation

    # CHSH 양: S = <A0 B0> + <A0 B1> + <A1 B0> - <A1 B1>
    S = results[(0,0)] + results[(0,1)] + results[(1,0)] - results[(1,1)]
    return S, results

# 파울리 행렬
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# 벨 상태 |Phi+>
phi_plus = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

# === 고전적 한계: 앨리스와 밥이 같은 기저에서 측정 ===
print("=== 정렬된 측정을 사용한 CHSH ===\n")

alice_obs = [Z, X]
bob_obs = [Z, X]

S, results = chsh_experiment(phi_plus, alice_obs, bob_obs)
print("앨리스: {Z, X}, 밥: {Z, X}")
for (i,j), val in results.items():
    print(f"  <A{i} B{j}> = {val:.4f}")
print(f"  S = {S:.4f}")
print(f"  |S| = {abs(S):.4f} (고전적 한계: 2)")

# === 양자 최적: 밥의 축이 45도 회전 ===
print("\n=== 최적 측정을 사용한 CHSH (치렐손 한계) ===\n")

# 밥의 최적 관측량: (Z+X)/sqrt(2) 및 (Z-X)/sqrt(2)
B0 = (Z + X) / np.sqrt(2)
B1 = (Z - X) / np.sqrt(2)

alice_obs = [Z, X]
bob_obs = [B0, B1]

S, results = chsh_experiment(phi_plus, alice_obs, bob_obs)
print("앨리스: {Z, X}, 밥: {(Z+X)/sqrt(2), (Z-X)/sqrt(2)}")
for (i,j), val in results.items():
    print(f"  <A{i} B{j}> = {val:.4f}")
print(f"  S = {S:.4f}")
print(f"  |S| = {abs(S):.4f}")
print(f"  고전적 한계: 2.0000")
print(f"  치렐손 한계: {2*np.sqrt(2):.4f}")
print(f"\n  위반: |S| = {abs(S):.4f} > 2.0000!")
print(f"  어떤 국소 숨은 변수 이론도 이 결과를 만들어낼 수 없습니다.")

# === CHSH 게임의 몬테카를로 시뮬레이션 ===
print("\n\n=== 몬테카를로 CHSH 게임 ===\n")

def chsh_game_quantum(state_2q, alice_obs, bob_obs, n_rounds=10000):
    """
    실제 측정 결과를 시뮬레이션합니다(기댓값만이 아닌).

    왜 실제 측정을 시뮬레이션할까요? 위의 기댓값 계산은 정확한 양자
    예측을 제공합니다. 이 시뮬레이션은 개별 측정 결과가 무작위이지만
    그 상관관계가 고전적 한계를 위반함을 보여줍니다.
    """
    np.random.seed(42)
    total_S_contribution = 0

    for _ in range(n_rounds):
        # 측정 설정 무작위 선택
        i = np.random.randint(2)
        j = np.random.randint(2)

        A = alice_obs[i]
        B = bob_obs[j]

        # 결합 측정
        AB = np.kron(A, B)
        eig_vals, eig_vecs = np.linalg.eigh(AB)

        # 각 결합 결과의 확률
        probs = np.abs(eig_vecs.conj().T @ state_2q)**2
        outcome_idx = np.random.choice(4, p=probs)
        outcome_value = eig_vals[outcome_idx]

        # CHSH 부호: (0,0), (0,1), (1,0)에 대해 +1, (1,1)에 대해 -1
        sign = -1 if (i == 1 and j == 1) else 1
        total_S_contribution += sign * outcome_value

    # 각 설정 쌍은 ~n_rounds/4 번 선택됩니다
    # S = sign * outcome의 합 / (n_rounds/4)
    S_estimated = total_S_contribution / (n_rounds / 4)
    return S_estimated

S_mc = chsh_game_quantum(phi_plus, [Z, X], [B0, B1], n_rounds=40000)
print(f"몬테카를로 S 추정값 (40000회): {S_mc:.4f}")
print(f"정확한 양자값: {2*np.sqrt(2):.4f}")
```

---

## 6. 얽힘의 단일성

### 6.1 원리

**얽힘의 단일성(Monogamy of entanglement)**: 큐비트 A가 큐비트 B와 최대로 얽혀 있으면, A는 다른 어떤 큐비트 C와도 *전혀* 얽힐 수 없습니다.

더 정확하게는, 큐비트 A, B, C에 대해 얽힘은 다음을 만족합니다:

$$E(A:B) + E(A:C) \leq E(A:BC)$$

여기서 $E$는 얽힘의 척도입니다. $E(A:B)$가 최대이면, $E(A:C)$를 위해 남겨진 것이 없습니다.

### 6.2 함의

단일성은 심오한 결과를 가집니다:

1. **양자 암호학**: 앨리스와 밥이 최대로 얽힌 쌍을 공유하면, 도청자 이브(Eve)는 어느 쪽과도 얽힐 수 없습니다. 이것이 양자 키 분배(quantum key distribution) 보안의 기반입니다.

2. **복제 불가 원리(No-cloning)**: 단일성은 복제 불가 정리와 밀접하게 관련됩니다. A의 B와의 얽힘을 복제할 수 있다면 단일성을 위반하게 됩니다.

3. **얽힘 분배**: 다중 큐비트 시스템에서 얽힘은 신중하게 "공유"되어야 하는 유한한 자원입니다.

### 6.3 고전적 상관관계와의 대비

고전적 상관관계는 단일적이지 않습니다. 앨리스의 동전이 밥의 것과 상관관계가 있는 경우(예: 항상 같은 면이 나오는 경우), 앨리스의 동전은 찰리의 동전과도 완벽하게 상관관계가 있을 수 있습니다. 단일성은 순수하게 양자적인 현상입니다.

```python
import numpy as np

# 단일성 시연: GHZ 상태 대 W 상태

# GHZ 상태: (|000> + |111>)/sqrt(2)
# 큐비트 0은 큐비트 1, 2와 집합적으로 최대 상관관계이지만,
# 큐비트 1과 단독으로는 최대 얽힘이 아닙니다
ghz = np.zeros(8, dtype=complex)
ghz[0] = 1/np.sqrt(2)  # |000>
ghz[7] = 1/np.sqrt(2)  # |111>

# W 상태: (|001> + |010> + |100>)/sqrt(3)
# 얽힘이 모든 쌍에 걸쳐 더 "고르게" 분포됩니다
w = np.zeros(8, dtype=complex)
w[1] = 1/np.sqrt(3)  # |001>
w[2] = 1/np.sqrt(3)  # |010>
w[4] = 1/np.sqrt(3)  # |100>

def reduced_density_matrix(state_3q, trace_out_qubit):
    """
    3큐비트 상태에서 하나의 큐비트를 추적 제거하여 축소 밀도 행렬을 계산합니다.

    왜 축소 밀도 행렬이 필요할까요? 다중 큐비트 시스템이 있고 부분 시스템만
    기술하고 싶을 때, 축소 밀도 행렬이 필요합니다. 얽힌 상태의 경우,
    축소 밀도 행렬은 혼합 상태(mixed state)이며, 혼합 정도가 얽힘을 정량화합니다.
    """
    rho = np.outer(state_3q, state_3q.conj())  # 전체 8x8 밀도 행렬
    rho_reshaped = rho.reshape(2, 2, 2, 2, 2, 2)

    # 지정된 큐비트 추적 제거
    # 큐비트 순서: q2, q1, q0 (q0이 최하위 비트)
    if trace_out_qubit == 0:
        # q0 추적 제거: q0 인덱스에 대해 합산
        reduced = np.trace(rho_reshaped, axis1=2, axis2=5)  # 4x4
    elif trace_out_qubit == 1:
        reduced = np.trace(rho_reshaped, axis1=1, axis2=4)  # 4x4
    else:  # trace_out_qubit == 2
        reduced = np.trace(rho_reshaped, axis1=0, axis2=3)  # 4x4

    return reduced.reshape(4, 4)

def von_neumann_entropy(rho):
    """폰 노이만 엔트로피 S = -Tr(rho * log2(rho))를 계산합니다."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # 0 제거
    return -np.sum(eigenvalues * np.log2(eigenvalues))

print("=== 얽힘의 단일성 ===\n")
print("GHZ 상태와 W 상태 비교:\n")

for name, state in [("GHZ", ghz), ("W", w)]:
    print(f"--- {name} 상태 ---")
    for q in range(3):
        rho_pair = reduced_density_matrix(state, q)
        S = von_neumann_entropy(rho_pair)
        other_qubits = [i for i in range(3) if i != q]
        print(f"  q{q} 추적 제거: S(q{other_qubits[0]},q{other_qubits[1]}) = {S:.4f} bits")
    print()

print("GHZ: 큐비트들은 집합적으로 최대 상관관계이지만 각 쌍은")
print("     제한된 쌍별 얽힘을 가집니다(단일성 제약).")
print("W:   얽힘이 쌍들 사이에 더 고르게 분포됩니다.")
```

---

## 7. 자원으로서의 얽힘

### 7.1 양자 컴퓨팅에서 얽힘이 중요한 이유

얽힘은 단지 호기심스러운 물리적 현상이 아닙니다 -- 그것은 *계산 자원*입니다:

1. **지수적 상태 공간**: $n$큐비트 얽힌 상태는 별도의 큐비트 기술로는 효율적으로 설명할 수 없는 $2^n$차원 공간에 존재합니다.

2. **양자 속도 향상**: 알려진 모든 지수적 양자 속도 향상은 얽힘을 필요로 합니다. 얽힘 없이는 양자 회로가 고전적으로 효율적으로 시뮬레이션될 수 있습니다(클리포드 회로의 고트스먼-크닐(Gottesman-Knill) 정리, 또는 분리 가능한 상태의 텐서 곱 구조).

3. **양자 통신**: 얽힘은 텔레포테이션(얽힘 + 고전 통신을 사용하여 양자 상태 전송)과 초밀도 부호화(1큐비트 + 얽힘을 사용하여 2개의 고전 비트 전송)를 가능하게 합니다.

### 7.2 알고리즘에서의 얽힘

앞으로 공부할 양자 알고리즘에서:

- **도이치-조자(Deutsch-Jozsa)** ([레슨 7](07_Deutsch_Jozsa_Algorithm.md)): 오라클이 쿼리 레지스터와 보조 큐비트 사이에 얽힘을 생성하여, 상수 함수와 균형 함수를 구별하는 간섭을 가능하게 합니다.

- **그로버 탐색(Grover's search)** ([레슨 8](08_Grovers_Search.md)): 오라클과 확산 연산자가 다중 큐비트 얽힘을 생성하고 조작하여 올바른 답의 진폭을 증폭합니다.

- **쇼어 알고리즘(Shor's algorithm)** (후속 레슨): 모듈식 지수 함수의 입력 레지스터와 출력 레지스터 사이의 얽힘이 양자 푸리에 변환이 주기를 추출할 수 있게 합니다.

### 7.3 얽힘의 양

이분(bipartite) 상태 $|\psi\rangle_{AB}$의 **얽힘 엔트로피(entanglement entropy)**는 어느 쪽 축소 밀도 행렬의 폰 노이만 엔트로피(von Neumann entropy)입니다:

$$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A)$$

두 큐비트의 최대 얽힌 상태의 경우, $S = 1$ 비트입니다. 두 $d$차원 시스템의 최대 얽힌 상태의 경우, $S = \log_2 d$ 비트입니다.

---

## 8. 연습 문제

### 연습 1: 벨 상태 식별

다음 각 상태에 대해, 어느 벨 상태인지(또는 벨 상태가 아닌지) 판별하세요:

a) $\frac{1}{\sqrt{2}}(|00\rangle - |11\rangle)$
b) $\frac{1}{\sqrt{2}}(|01\rangle + |10\rangle)$
c) $\frac{1}{\sqrt{2}}(|00\rangle + i|11\rangle)$
d) $\frac{1}{\sqrt{2}}(|+0\rangle + |-1\rangle)$ (먼저 계산 기저로 전개하세요.)

### 연습 2: 얽힘 탐지

각 상태가 분리 가능한지 얽혀 있는지 판별하세요. 얽힌 상태에 대해 얽힘 엔트로피를 계산하세요.

a) $\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle + |11\rangle)$
b) $\frac{1}{2}(|00\rangle + |01\rangle + |10\rangle - |11\rangle)$
c) $\frac{1}{\sqrt{2}}|00\rangle + \frac{1}{2}|01\rangle + \frac{1}{2}|11\rangle$

### 연습 3: CHSH 계산

벨 상태 $|\Psi^-\rangle = \frac{1}{\sqrt{2}}(|01\rangle - |10\rangle)$에 대해:

a) $\langle Z \otimes Z \rangle$ (두 큐비트 모두에서 $Z$의 기댓값)를 계산하세요.
b) $\langle X \otimes X \rangle$를 계산하세요.
c) 이 상태에 대해 CHSH 양 $S$를 최대화하는 앨리스와 밥의 측정 설정을 찾으세요.

### 연습 4: 3큐비트 얽힘

a) 양자 회로를 사용하여 GHZ 상태 $\frac{1}{\sqrt{2}}(|000\rangle + |111\rangle)$를 생성하세요(게이트를 명시하세요).
b) W 상태 $\frac{1}{\sqrt{3}}(|001\rangle + |010\rangle + |100\rangle)$를 생성하세요. 이것은 더 어렵습니다 -- 제어 회전이 필요합니다.
c) 두 상태를 계산 기저에서 각각 10,000번 측정하고 결과 분포를 비교하세요.

### 연습 5: 얽힘 교환

상태 $|\Phi^+\rangle_{12} \otimes |\Phi^+\rangle_{34}$ (두 개의 독립적인 벨 쌍)를 고려하세요. 큐비트 2와 3에 대해 벨 측정을 수행하면, 큐비트 1과 4는 어떻게 될까요? 큐비트 1과 4가 직접 상호작용한 적이 없음에도 불구하고 얽히게 됨을 보이세요. 이것이 **얽힘 교환(entanglement swapping)**입니다. (힌트: 큐비트 2, 3에 대한 벨 기저로 전체 4큐비트 상태를 다시 쓰세요.)

---

[<- 이전: 양자 회로](04_Quantum_Circuits.md) | [다음: 양자 측정 ->](06_Quantum_Measurement.md)
