# 레슨 11: 양자 오류 정정(Quantum Error Correction)

[← 이전: 쇼어의 인수분해 알고리즘](10_Shors_Algorithm.md) | [다음: 양자 텔레포테이션과 통신 →](12_Quantum_Teleportation.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 양자 오류 정정(Quantum Error Correction)이 고전적 오류 정정과 근본적으로 다른 이유를 설명할 수 있다
2. 복제 불가 정리(No-Cloning Theorem)와 그것이 오류 정정에 주는 시사점을 설명할 수 있다
3. 3-큐비트 비트 반전 코드(Bit-Flip Code)와 위상 반전 코드(Phase-Flip Code)를 구성할 수 있다
4. 임의의 단일 큐비트 오류를 정정하는 쇼어의 9-큐비트 코드를 설명할 수 있다
5. 안정자 형식론(Stabilizer Formalism)을 사용해 양자 오류 정정 코드를 기술할 수 있다
6. CSS(Calderbank-Shor-Steane) 코드와 표면 코드(Surface Code)를 개괄할 수 있다
7. 오류 허용 임계값 정리(Fault-Tolerant Threshold Theorem)와 그 의의를 진술할 수 있다

---

모든 계산은 오류를 만든다. 고전 컴퓨터는 이를 중복성(Redundancy)으로 처리한다 — 각 비트를 세 번 저장하고 다수결로 복원한다. 양자 컴퓨터는 훨씬 더 어려운 도전에 직면한다: 복제 불가 정리는 양자 상태를 복사하는 것을 금지하고, 연속적인 오류(작은 각도의 회전)는 정정하기 위해 무한한 정밀도가 필요해 보이며, 측정은 보호하려는 정보 자체를 파괴한다. 수십 년간 많은 물리학자들은 양자 오류 정정이 불가능하다고 믿었다.

양자 오류 정정이 *가능하다*는 발견 — 쇼어(1995)와 스테인(1996)에 의해 — 은 양자 정보 이론에서 가장 놀라운 결과 중 하나였다. 이 모든 장애물에도 불구하고, 영리한 인코딩이 정보가 무엇인지 알지 못하면서도 양자 정보를 노이즈로부터 보호할 수 있음을 보였다. 이 돌파구는 오류 허용 양자 컴퓨팅의 전체 사업을 뒷받침한다: 이것 없이는 어떤 양자 알고리즘도 유용할 만큼 충분히 오래 실행될 수 없다.

> **비유:** 양자 오류 정정은 합창단에 메시지를 인코딩하는 것과 같습니다. 한 가수가 음이 틀리면, 다른 가수들이 원래 음표를 알지 못하면서도 오류를 감지하고 정정할 수 있습니다. 합창단원들은 부르는 멜로디를 결코 알지 못합니다 — 그들은 오직 자신들의 음표 사이의 *관계*(화음)만 압니다. 이 관계(신드롬)를 확인함으로써, 노래가 계속되는 동안 특정 가수의 실수를 찾아내고 고칠 수 있습니다.

## 목차

1. [QEC가 다른 이유](#1-qec가-다른-이유)
2. [복제 불가 정리](#2-복제-불가-정리)
3. [3-큐비트 비트 반전 코드](#3-3-큐비트-비트-반전-코드)
4. [3-큐비트 위상 반전 코드](#4-3-큐비트-위상-반전-코드)
5. [쇼어의 9-큐비트 코드](#5-쇼어의-9-큐비트-코드)
6. [안정자 형식론](#6-안정자-형식론)
7. [CSS 코드](#7-css-코드)
8. [표면 코드](#8-표면-코드)
9. [오류 허용 임계값 정리](#9-오류-허용-임계값-정리)
10. [Python 구현](#10-python-구현)
11. [연습 문제](#11-연습-문제)

---

## 1. QEC가 다른 이유

### 1.1 고전적 오류 정정 복습

고전적 오류 정정은 간단하다: 비트 $b \in \{0, 1\}$를 보호하기 위해 세 개의 복사본으로 인코딩한다:

$$0 \to 000, \quad 1 \to 111$$

한 비트가 반전되면(예: $000 \to 010$), 다수결이 원래 값을 복원한다. 이 **반복 코드(Repetition Code)**는 임의의 단일 비트 반전을 정정할 수 있다.

### 1.2 세 가지 양자 장애물

양자 오류 정정은 고전적 유사물이 없는 세 가지 과제에 직면한다:

**장애물 1: 복제 불가 정리(No-Cloning Theorem)**

미지의 양자 상태 $|\psi\rangle$를 복사할 수 없다. 따라서 $|\psi\rangle$를 $|\psi\rangle|\psi\rangle|\psi\rangle$으로 인코딩하는 단순한 접근법은 불가능하다. 근본적으로 다른 중복성 개념이 필요하다.

**장애물 2: 연속적인 오류(Continuous Errors)**

고전적 비트는 오직 반전만 가능하다($0 \leftrightarrow 1$). 그러나 큐비트는 임의의 유니타리 회전을 겪을 수 있다 — 연속적인 오류의 집합. 작은 회전 $R_x(\epsilon)$은 $|0\rangle$을 $|1\rangle$ 방향으로 약간 이동시킨다:

$$R_x(\epsilon)|0\rangle = \cos(\epsilon/2)|0\rangle - i\sin(\epsilon/2)|1\rangle$$

이산 코드로 어떻게 연속적인 오류의 범위를 정정할 수 있을까?

**장애물 3: 측정이 정보를 파괴한다(Measurement Destroys Information)**

고전적 오류 정정에서는 인코딩된 비트를 자유롭게 검사할 수 있다. 양자 상태를 측정하면 붕괴(Collapse)가 일어나 보호하려는 정보를 잠재적으로 파괴한다.

### 1.3 해결책

놀랍게도, 세 가지 장애물 모두 우아한 해결책이 있다:

1. **복제 불가**: $|\psi\rangle$를 복사하는 대신, 여러 큐비트에 걸쳐 *얽힘(Entangle)*을 만든다. 정보는 개별 큐비트가 아닌 상관관계(Correlation)에 인코딩된다.

2. **연속적인 오류**: 임의의 단일 큐비트 오류는 항등 연산자 $I$와 세 파울리 행렬 $X$, $Y$, $Z$의 선형 결합으로 쓸 수 있다. 이 네 가지 이산 오류를 정정하면 양자역학의 선형성에 의해 자동으로 모든 연속 오류가 정정된다.

3. **측정**: 인코딩된 정보를 드러내지 않으면서 오류를 감지하는 *신드롬 측정(Syndrome Measurement)*을 수행한다. 이 측정은 오류를 이산 파울리 오류 중 하나로 사영(Project)하며, 그런 다음 정정을 적용한다.

---

## 2. 복제 불가 정리

### 2.1 진술

**복제 불가 정리(No-Cloning Theorem)**: 임의의 양자 상태를 복제할 수 있는 유니타리 연산 $U$는 존재하지 않는다:

$$U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle \quad \text{for all } |\psi\rangle$$

### 2.2 증명 개요

그러한 $U$가 존재한다고 가정하자. 두 상태 $|\psi\rangle$와 $|\phi\rangle$에 대해:

$$U|\psi\rangle|0\rangle = |\psi\rangle|\psi\rangle$$
$$U|\phi\rangle|0\rangle = |\phi\rangle|\phi\rangle$$

양변의 내적을 취하면:

$$\langle\psi|\phi\rangle \cdot \langle 0|0\rangle = \langle\psi|\phi\rangle^2$$
$$\langle\psi|\phi\rangle = \langle\psi|\phi\rangle^2$$

이는 $\langle\psi|\phi\rangle = 0$ 또는 $\langle\psi|\phi\rangle = 1$일 때만 만족된다. 따라서 $U$는 동일하거나 직교하는 상태만 복제할 수 있다 — 임의의 상태는 불가능하다. 모순.

### 2.3 오류 정정에 대한 시사점

복제가 불가능하므로 고전적 반복 전략을 사용할 수 없다. 대신, 양자 코드는 하나의 논리적 큐비트를 여러 물리적 큐비트의 *얽힘 상태(Entangled State)*에 인코딩한다. 인코딩은 다음을 매핑한다:

$$\alpha|0\rangle + \beta|1\rangle \to \alpha|0_L\rangle + \beta|1_L\rangle$$

여기서 $|0_L\rangle$과 $|1_L\rangle$은 다중 큐비트 *코드워드(Codeword)*이며, 계수 $\alpha, \beta$는 복사되지 않으면서 보존된다.

---

## 3. 3-큐비트 비트 반전 코드

### 3.1 인코딩

비트 반전 코드(Bit-Flip Code)는 하나의 논리적 큐비트를 세 개의 물리적 큐비트로 인코딩한다:

$$|0\rangle \to |0_L\rangle = |000\rangle$$
$$|1\rangle \to |1_L\rangle = |111\rangle$$

일반 상태 $\alpha|0\rangle + \beta|1\rangle$은 다음으로 인코딩된다:

$$|\psi_L\rangle = \alpha|000\rangle + \beta|111\rangle$$

참고: 이것은 $\alpha|0\rangle + \beta|1\rangle$의 세 복사본이 아니다(그것은 $(\alpha|0\rangle + \beta|1\rangle)^{\otimes 3}$이 될 것이다). 이것은 *얽힘 상태(Entangled State)*다 — $\alpha$와 $\beta$에 대한 정보는 세 큐비트 사이의 상관관계에 저장된다.

### 3.2 오류 모델

큐비트 $i$에 대한 비트 반전 오류는 해당 큐비트에 파울리 $X$ 연산자를 적용한다:

$$X_1: |000\rangle \to |100\rangle, \quad |111\rangle \to |011\rangle$$
$$X_2: |000\rangle \to |010\rangle, \quad |111\rangle \to |101\rangle$$
$$X_3: |000\rangle \to |001\rangle, \quad |111\rangle \to |110\rangle$$

### 3.3 신드롬 측정

인코딩된 정보를 측정하지 않고 오류를 감지하기 위해 **패리티 검사(Parity Check)**를 측정한다:

- $Z_1 Z_2$: 큐비트 1과 2가 같은 값인지(+1) 다른 값인지(-1) 측정
- $Z_2 Z_3$: 큐비트 2와 3이 같은 값인지(+1) 다른 값인지(-1) 측정

**신드롬(Syndrome)**은 측정 결과 쌍 $(s_1, s_2)$이다:

| 오류 | 상태 ($|000\rangle$으로부터) | $Z_1 Z_2$ | $Z_2 Z_3$ | 신드롬 |
|-------|--------------------------|-----------|-----------|----------|
| 없음 | $|000\rangle$ | +1 | +1 | (0, 0) |
| $X_1$ | $|100\rangle$ | -1 | +1 | (1, 0) |
| $X_2$ | $|010\rangle$ | -1 | -1 | (1, 1) |
| $X_3$ | $|001\rangle$ | +1 | -1 | (0, 1) |

각 오류는 고유한 신드롬을 생성하여 명확한 식별 및 정정이 가능하다. 중요하게도, 이러한 측정은 상태가 $|000\rangle$인지 $|111\rangle$인지(또는 중첩인지)를 드러내지 않는다 — 오직 *패리티 정보*만 드러낸다.

### 3.4 신드롬 측정이 양자 정보를 보존하는 이유

핵심 통찰은 $Z_1 Z_2$가 $|000\rangle$과 $|111\rangle$ 모두에 대해 고유값 $+1$을 가진다는 것이다:

$$Z_1 Z_2 |000\rangle = (+1)(+1)|000\rangle = +|000\rangle$$
$$Z_1 Z_2 |111\rangle = (-1)(-1)|111\rangle = +|111\rangle$$

따라서 중첩 $\alpha|000\rangle + \beta|111\rangle$은 고유값 $+1$을 가진 $Z_1 Z_2$의 고유상태(Eigenstate)다. $Z_1 Z_2$를 측정해도 이 중첩이 흐트러지지 않는다 — 오직 오류가 패리티 관계를 깼는지만 알려준다.

### 3.5 한계

비트 반전 코드는 오직 $X$(비트 반전) 오류만 정정한다. $Z$(위상 반전) 오류는 정정할 수 없다:

$$Z_1(\alpha|000\rangle + \beta|111\rangle) = \alpha|000\rangle - \beta|111\rangle$$

이 위상 오류는 $|0_L\rangle$과 $|1_L\rangle$ 사이의 상대적 위상을 바꾸며, $Z_1 Z_2$와 $Z_2 Z_3$ 신드롬 측정으로는 감지할 수 없다.

---

## 4. 3-큐비트 위상 반전 코드

### 4.1 위상 반전 오류

위상 반전 오류는 파울리 $Z$ 연산자를 적용한다: $Z|0\rangle = |0\rangle$, $Z|1\rangle = -|1\rangle$.

위상 반전을 정정하기 위해 아다마르(Hadamard, X) 기저로 인코딩한다:

$$|0\rangle \to |0_L\rangle = |{+}{+}{+}\rangle$$
$$|1\rangle \to |1_L\rangle = |{-}{-}{-}\rangle$$

여기서 $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$이고 $|-\rangle = (|0\rangle - |1\rangle)/\sqrt{2}$이다.

### 4.2 작동 원리

큐비트 $i$에 대한 위상 반전은 $|+\rangle \to |-\rangle$으로 매핑하고 그 반대도 마찬가지다. $\{|+\rangle, |-\rangle\}$ 기저에서 $Z$ 오류는 비트 반전처럼 작동한다! 따라서 동일한 신드롬 측정 전략을 사용할 수 있지만 X 기저에서 사용한다.

신드롬 측정은 $X_1 X_2$와 $X_2 X_3$이며, X 기저에서 패리티를 확인한다.

### 4.3 한계

위상 반전 코드는 $Z$ 오류를 정정하지만 $X$ 오류는 정정하지 못한다 — 비트 반전 코드와 상호 보완적인 문제다.

---

## 5. 쇼어의 9-큐비트 코드

### 5.1 핵심 아이디어

쇼어의 탁월한 통찰은 두 코드를 결합하는 것이었다: 먼저 위상 반전에 대해 인코딩하고(외부 코드), 그런 다음 그 코드의 각 큐비트를 비트 반전에 대해 인코딩한다(내부 코드). 결과는 임의의 단일 큐비트 오류를 정정하는 9-큐비트 코드다.

### 5.2 인코딩

**1단계 (위상 반전 인코딩)**: 논리적 큐비트를 인코딩:

$$|0\rangle \to |{+}{+}{+}\rangle, \quad |1\rangle \to |{-}{-}{-}\rangle$$

**2단계 (각 큐비트에 비트 반전 인코딩)**: 각 $|+\rangle$와 $|-\rangle$을 아다마르 기저의 3-큐비트 반복 코드로 인코딩:

$$|+\rangle \to \frac{1}{\sqrt{2}}(|000\rangle + |111\rangle) \equiv |+_L\rangle$$
$$|-\rangle \to \frac{1}{\sqrt{2}}(|000\rangle - |111\rangle) \equiv |-_L\rangle$$

**완전한 인코딩**:

$$|0\rangle \to |0_L\rangle = \frac{1}{2\sqrt{2}} (|000\rangle + |111\rangle)(|000\rangle + |111\rangle)(|000\rangle + |111\rangle)$$

$$|1\rangle \to |1_L\rangle = \frac{1}{2\sqrt{2}} (|000\rangle - |111\rangle)(|000\rangle - |111\rangle)(|000\rangle - |111\rangle)$$

### 5.3 오류 정정

**비트 반전 오류** (임의의 큐비트 $i$에 $X_i$): 각 3-큐비트 블록 내에서 $Z_i Z_j$ 측정을 사용하는 패리티 검사로 감지된다. 3-큐비트 비트 반전 코드와 동일하며, 각 블록에 독립적으로 적용된다.

**위상 반전 오류** (임의의 큐비트 $i$에 $Z_i$): 블록 내 임의의 큐비트에 대한 $Z$ 오류는 그 블록에서 $|111\rangle$의 부호를 반전시켜, 효과적으로 $|+_L\rangle \leftrightarrow |-_L\rangle$을 변환한다. $X_1 X_2 X_3 X_4 X_5 X_6$와 $X_4 X_5 X_6 X_7 X_8 X_9$를 측정하여(블록 쌍별 비교) 감지된다.

**$Y$ 오류**: $Y = iXZ$이므로, $Y$ 오류는 비트 반전과 위상 반전을 모두 일으키며, 위의 신드롬 측정으로 모두 감지되고 정정된다.

### 5.4 연속 오류가 처리되는 이유

임의의 단일 큐비트 오류 $E$는 다음과 같이 분해될 수 있다:

$$E = \alpha_I I + \alpha_X X + \alpha_Y Y + \alpha_Z Z$$

신드롬 측정 후, 오류는 $\{I, X, Y, Z\}$ 중 하나로 *사영(Projected)*된다. 신드롬은 어떤 파울리 오류가 발생했는지 알려주고, 해당 정정을 적용한다. 양자역학의 선형성은 연속 오류에도 이것이 작동하도록 보장한다 — 측정이 오류를 "디지털화"한다.

### 5.5 매개변수

쇼어의 9-큐비트 코드: $[[9, 1, 3]]$

- **물리적 큐비트 9개**가 **논리적 큐비트 1개**를 인코딩
- **거리 3(Distance 3)**: 임의의 단일 큐비트 오류를 정정할 수 있음 ($\leq 1$개 큐비트에 작용하는 임의의 오류)
- **8개의 신드롬 측정** (3개 블록당 2개, 블록 간 2개)으로 오류를 고유하게 식별

---

## 6. 안정자 형식론

### 6.1 동기

양자 코드가 커질수록, 코드워드로 기술하는 것이 번거로워진다. **안정자 형식론(Stabilizer Formalism)**은 양자 오류 정정 코드를 기술하기 위한 간결하고 강력한 프레임워크를 제공한다.

### 6.2 파울리 군

**$n$-큐비트 파울리 군(Pauli Group)** $\mathcal{P}_n$은 파울리 행렬 $\{I, X, Y, Z\}$의 $n$중 텐서곱 전체로 구성되며, 위상 $\{\pm 1, \pm i\}$가 곱해진다:

$$\mathcal{P}_n = \{\pm 1, \pm i\} \times \{I, X, Y, Z\}^{\otimes n}$$

예를 들어, $X_1 Z_3 = X \otimes I \otimes Z$는 $\mathcal{P}_3$의 원소다.

### 6.3 안정자 코드

**안정자 코드(Stabilizer Code)** $C$는 아벨 부분군(Abelian Subgroup) $\mathcal{S} \subset \mathcal{P}_n$ (**안정자 군(Stabilizer Group)**)으로 정의된다:

$$C = \{|\psi\rangle : S|\psi\rangle = |\psi\rangle \text{ for all } S \in \mathcal{S}\}$$

코드 공간은 모든 안정자 연산자의 동시적인 $+1$ 고유 공간(Eigenspace)이다.

**핵심 특성**: 안정자 군은 $n - k$개의 독립적인 생성원(Generator)으로 생성되며, $k$는 인코딩된 논리적 큐비트의 수다. 이 생성원을 측정하면 신드롬이 나온다.

### 6.4 안정자 언어로 본 비트 반전 코드

3-큐비트 비트 반전 코드의 경우:

- **안정자 생성원**: $S_1 = Z_1 Z_2$, $S_2 = Z_2 Z_3$
- **코드 공간**: $Z_1 Z_2 = +1$이고 $Z_2 Z_3 = +1$인 상태, 즉 $|000\rangle$과 $|111\rangle$
- **논리 연산자**: $\bar{X} = X_1 X_2 X_3$ (논리적 비트 반전), $\bar{Z} = Z_1$ (논리적 위상 반전)

코드는 $[[3, 1, 1]]$이다: 물리적 큐비트 3개, 논리적 큐비트 1개, 거리 1(위상 반전 오류를 정정할 수 없음).

### 6.5 안정자 언어로 본 쇼어의 코드

9-큐비트 쇼어 코드는 8개의 안정자 생성원을 가진다:

| 생성원 | 큐비트 | 감지 |
|-----------|--------|---------|
| $Z_1 Z_2$ | 블록 1 | 블록 1의 비트 반전 |
| $Z_2 Z_3$ | 블록 1 | 블록 1의 비트 반전 |
| $Z_4 Z_5$ | 블록 2 | 블록 2의 비트 반전 |
| $Z_5 Z_6$ | 블록 2 | 블록 2의 비트 반전 |
| $Z_7 Z_8$ | 블록 3 | 블록 3의 비트 반전 |
| $Z_8 Z_9$ | 블록 3 | 블록 3의 비트 반전 |
| $X_1 X_2 X_3 X_4 X_5 X_6$ | 블록 1-2 | 블록 간 위상 반전 |
| $X_4 X_5 X_6 X_7 X_8 X_9$ | 블록 2-3 | 블록 간 위상 반전 |

매개변수: $[[9, 1, 3]]$ — 물리적 9개, 논리적 1개, 거리 3.

### 6.6 안정자로 오류 감지

오류 $E$는 $\{E, S\} = 0$ (반교환, Anti-commute)이면 안정자 $S$에 의해 감지된다:

$$SE|\psi\rangle = -ES|\psi\rangle = -E|\psi\rangle$$

따라서 오류 $E$가 발생하면 $S$를 측정할 때 $-1$이 나와, 신드롬에서 오류를 표시한다.

---

## 7. CSS 코드

### 7.1 구성

**Calderbank-Shor-Steane(CSS) 코드**는 $C_2 \subset C_1$인 두 고전적 선형 코드 $C_1$과 $C_2$로부터 구성되는 양자 코드 계열이다:

- $C_1$: $[n, k_1, d_1]$ 고전적 코드 (비트 반전 정정)
- $C_2 \subset C_1$: $[n, k_2, d_2]$ 고전적 코드 (이중 코드를 통한 위상 반전 정정)

결과 양자 코드는 $[[n, k_1 - k_2, \min(d_1, d_2^\perp)]]$이다.

### 7.2 스테인 코드

가장 유명한 CSS 코드는 **스테인(Steane) $[[7, 1, 3]]$ 코드**로, 고전적 해밍(Hamming) $[7, 4, 3]$ 코드로부터 구성된다. 7개의 물리적 큐비트로 1개의 논리적 큐비트를 거리 3으로 인코딩한다(임의의 단일 큐비트 오류 정정).

**안정자 생성원**:

| 생성원 | 유형 |
|-----------|------|
| $X_1 X_3 X_5 X_7$ | X형 |
| $X_2 X_3 X_6 X_7$ | X형 |
| $X_4 X_5 X_6 X_7$ | X형 |
| $Z_1 Z_3 Z_5 Z_7$ | Z형 |
| $Z_2 Z_3 Z_6 Z_7$ | Z형 |
| $Z_4 Z_5 Z_6 Z_7$ | Z형 |

**쇼어 코드 대비 장점**: 스테인 코드는 7개의 큐비트만 사용하며(쇼어의 9개 대비), 가로방향 게이트(Transversal Gate) 구현을 단순화하는 대칭 구조를 가진다.

### 7.3 CSS 코드 특성

- X형 안정자는 Z 오류를 감지하고(반대도 마찬가지)
- 신드롬 디코딩은 $C_1$과 $C_2^\perp$의 고전적 디코딩으로 환원됨
- 가로방향 CNOT은 자동으로 오류 허용적(Fault-Tolerant)

---

## 8. 표면 코드

### 8.1 표면 코드가 중요한 이유

**표면 코드(Surface Code)**는 다음과 같은 이유로 근미래 양자 컴퓨터에서 가장 유망한 오류 정정 코드로 널리 인정받는다:

1. **국소 연산만 사용**: 모든 안정자 측정이 2D 격자에서 최근접 이웃 큐비트만 포함
2. **높은 임계값**: 오류 임계값이 게이트당 약 1% — 알려진 코드 중 가장 높음
3. **간단한 신드롬 추출**: 가중치-4 측정만 사용 (검사당 큐비트 4개)

### 8.2 구조

$L \times L$ 격자의 표면 코드 사용:

- $n = 2L^2 - 2L + 1$개의 물리적 큐비트 (모서리에 데이터 큐비트)
- $L^2 - L$개의 X형 안정자 (플래킷 연산자, Plaquette Operator)
- $L^2 - L$개의 Z형 안정자 (꼭짓점 연산자, Vertex Operator)
- $k = 1$개의 논리적 큐비트 인코딩
- 코드 거리 $d = L$

```
      Z──Z──Z
      │  │  │
   ───●──●──●───
      │  │  │
      X──X──X
      │  │  │
   ───●──●──●───
      │  │  │
      Z──Z──Z

  ● = 데이터 큐비트
  Z = Z-안정자 (꼭짓점)
  X = X-안정자 (플래킷)
```

### 8.3 오류 정정 절차

1. 모든 안정자를 반복적으로 측정 (측정 오류 처리를 위해)
2. 결함(Defect) 식별: $-1$인 안정자 결과
3. **디코더(Decoder)** 사용 (예: 최소 가중치 완전 매칭, Minimum-Weight Perfect Matching)으로 결함 쌍 짓기
4. 쌍짓기에 기반하여 정정 적용

### 8.4 오버헤드

표면 코드로 오류율 $p$에서 오류를 정정하려면:

- 필요한 코드 거리: 목표 논리적 오류율 $\epsilon$에 대해 $d \sim O(\log(1/\epsilon))$
- 논리적 큐비트당 물리적 큐비트: $\sim 2d^2$
- $p = 10^{-3}$이고 $\epsilon = 10^{-10}$인 경우: $d \approx 17$, 논리적 큐비트당 $\sim 578$개의 물리적 큐비트 필요

---

## 9. 오류 허용 임계값 정리

### 9.1 문제

오류 정정이 있더라도, 정정 회로 자체가 오류를 도입할 수 있다! 신드롬 측정에 사용된 결함 있는 CNOT 게이트는 한 큐비트에서 다른 큐비트로 오류를 전파할 수 있다. 이것은 표면적으로 순환적인 문제를 만든다: 양자 회로를 실행하려면 오류 정정이 필요하지만, 오류 정정 회로 자체도 오류를 가진다.

### 9.2 임계값 정리

**정리** (Aharonov & Ben-Or, 1997; Knill, Laflamme & Zurek, 1998): 게이트당 오류율이 임계값 $p_{\text{th}}$ 아래에 있으면, 임의의 작은 논리적 오류율로 임의의 긴 양자 계산을 수행할 수 있다.

더 정확하게, 크기 $L$의 양자 회로를 논리적 오류율 $\epsilon$으로 시뮬레이션하려면:

$$O\left(L \cdot \text{polylog}(L/\epsilon)\right)$$

개의 물리적 게이트가 필요하며, $p < p_{\text{th}}$를 만족해야 한다.

### 9.3 임계값 값

| 코드 계열 | 임계값 $p_{\text{th}}$ | 참고 |
|-------------|-------------------------|-------|
| 연접 코드(Concatenated Codes) | $\sim 10^{-5}$ ~ $10^{-4}$ | 초기 증명 |
| 표면 코드 | $\sim 10^{-2}$ (1%) | 가장 실용적 |
| 컬러 코드(Color Codes) | $\sim 3 \times 10^{-3}$ | 가로방향 게이트 |

현재 최첨단 큐비트 오류율: $\sim 10^{-3}$ ~ $10^{-2}$로, 표면 코드 임계값 바로 근처에 있다. 지속적으로 임계값 이하의 오류율을 달성하는 것이 양자 컴퓨팅의 주요 공학적 과제 중 하나다.

### 9.4 시사점

임계값 정리는 오류 허용 양자 컴퓨팅의 *존재 증명(Existence Proof)*이다. 원칙적으로 양자 컴퓨터를 신뢰할 수 있게 만들 수 있다고 말한다. 실질적인 질문은 오버헤드(논리적 큐비트당 수천 개의 물리적 큐비트)가 기술적으로 실현 가능할 만큼 충분히 줄어들 수 있는지 여부다. 이것이 이 분야의 중심 과제로 남아있다.

---

## 10. Python 구현

### 10.1 3-큐비트 비트 반전 코드

```python
import numpy as np

def encode_bit_flip(alpha, beta):
    """Encode a logical qubit |ψ⟩ = α|0⟩ + β|1⟩ into the 3-qubit bit-flip code.

    Why this encoding? The bit-flip code maps |0⟩→|000⟩ and |1⟩→|111⟩.
    By linearity, α|0⟩+β|1⟩ → α|000⟩+β|111⟩. This is an entangled state,
    NOT three copies of |ψ⟩ — consistent with the no-cloning theorem.
    """
    # 3-qubit state: 8-dimensional vector
    # Basis order: |000⟩, |001⟩, |010⟩, |011⟩, |100⟩, |101⟩, |110⟩, |111⟩
    state = np.zeros(8, dtype=complex)
    state[0b000] = alpha  # α|000⟩
    state[0b111] = beta   # β|111⟩
    return state

def apply_error(state, error_type, qubit):
    """Apply a Pauli error to a specific qubit in a 3-qubit state.

    Why use Pauli errors? Any single-qubit error can be decomposed as
    a linear combination of I, X, Y, Z. By correcting these four discrete
    errors, we automatically correct all continuous errors.
    """
    n = 3
    N = 8

    # Pauli matrices
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    # Build the n-qubit error operator
    ops = [I, I, I]
    ops[qubit] = paulis[error_type]

    error_op = np.kron(np.kron(ops[0], ops[1]), ops[2])
    return error_op @ state

def measure_syndrome_bit_flip(state):
    """Measure the bit-flip syndrome without collapsing the logical information.

    Why these specific measurements? Z_1⊗Z_2 checks if qubits 1 and 2
    have the same parity. Z_2⊗Z_3 checks qubits 2 and 3. Together, they
    uniquely identify which qubit (if any) was flipped, without revealing
    whether the logical state is |0_L⟩ or |1_L⟩.
    """
    n = 3
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    # Syndrome operator Z1Z2 ⊗ I3
    Z1Z2 = np.kron(np.kron(Z, Z), I)
    # Syndrome operator I1 ⊗ Z2Z3
    Z2Z3 = np.kron(np.kron(I, Z), Z)

    # Expectation values give syndrome bits
    s1 = np.real(state.conj() @ Z1Z2 @ state)
    s2 = np.real(state.conj() @ Z2Z3 @ state)

    # Convert to syndrome: +1 → 0 (no error), -1 → 1 (error detected)
    syndrome = (int(round(-s1 + 1) / 2), int(round(-s2 + 1) / 2))

    return syndrome

def correct_bit_flip(state, syndrome):
    """Apply correction based on the syndrome.

    The syndrome uniquely identifies the error:
    (0,0) → no error, (1,0) → flip qubit 1, (1,1) → flip qubit 2, (0,1) → flip qubit 3
    """
    correction_map = {
        (0, 0): None,
        (1, 0): 0,  # Error on qubit 0
        (1, 1): 1,  # Error on qubit 1
        (0, 1): 2,  # Error on qubit 2
    }

    qubit = correction_map[syndrome]
    if qubit is not None:
        state = apply_error(state, 'X', qubit)  # Apply X to correct the flip
    return state

def decode_bit_flip(state):
    """Decode the 3-qubit state back to a single logical qubit.

    Returns the amplitudes (α, β) of the logical state α|0⟩ + β|1⟩.
    """
    alpha = state[0b000]
    beta = state[0b111]
    norm = np.sqrt(abs(alpha)**2 + abs(beta)**2)
    return alpha/norm, beta/norm

# Demonstration: encode, introduce error, detect, correct, decode
print("=" * 60)
print("3-Qubit Bit-Flip Code Demonstration")
print("=" * 60)

# Encode |ψ⟩ = (|0⟩ + i|1⟩)/√2
alpha, beta = 1/np.sqrt(2), 1j/np.sqrt(2)
print(f"\nOriginal state: α = {alpha:.4f}, β = {beta:.4f}")

state = encode_bit_flip(alpha, beta)
print(f"Encoded state (3 qubits): {state.round(4)}")

# Introduce a bit-flip error on qubit 1 (middle qubit)
error_qubit = 1
state_with_error = apply_error(state, 'X', error_qubit)
print(f"\nAfter X error on qubit {error_qubit}: {state_with_error.round(4)}")

# Measure syndrome
syndrome = measure_syndrome_bit_flip(state_with_error)
print(f"Syndrome: {syndrome}")
syndrome_meaning = {(0,0): "no error", (1,0): "qubit 0", (1,1): "qubit 1", (0,1): "qubit 2"}
print(f"Diagnosis: {syndrome_meaning[syndrome]}")

# Correct the error
corrected = correct_bit_flip(state_with_error, syndrome)
print(f"\nCorrected state: {corrected.round(4)}")

# Decode
alpha_out, beta_out = decode_bit_flip(corrected)
print(f"Decoded: α = {alpha_out:.4f}, β = {beta_out:.4f}")
print(f"Fidelity: {abs(alpha * alpha_out.conj() + beta * beta_out.conj())**2:.6f}")
```

### 10.2 쇼어의 9-큐비트 코드

```python
import numpy as np

def encode_shor_9qubit(alpha, beta):
    """Encode a logical qubit using Shor's 9-qubit code.

    The encoding is:
    |0_L⟩ = (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩) / 2√2
    |1_L⟩ = (|000⟩-|111⟩)(|000⟩-|111⟩)(|000⟩-|111⟩) / 2√2

    Why two layers? The outer code (3-qubit phase-flip) handles Z errors,
    and the inner code (3-qubit bit-flip per block) handles X errors.
    Together, they correct any single-qubit Pauli error.
    """
    N = 2**9  # 512-dimensional Hilbert space

    # Build |+_L⟩ = (|000⟩ + |111⟩)/√2 for each block
    plus_block = np.zeros(8, dtype=complex)
    plus_block[0b000] = 1/np.sqrt(2)
    plus_block[0b111] = 1/np.sqrt(2)

    # Build |-_L⟩ = (|000⟩ - |111⟩)/√2 for each block
    minus_block = np.zeros(8, dtype=complex)
    minus_block[0b000] = 1/np.sqrt(2)
    minus_block[0b111] = -1/np.sqrt(2)

    # |0_L⟩ = |+_L⟩ ⊗ |+_L⟩ ⊗ |+_L⟩
    zero_L = np.kron(np.kron(plus_block, plus_block), plus_block)

    # |1_L⟩ = |-_L⟩ ⊗ |-_L⟩ ⊗ |-_L⟩
    one_L = np.kron(np.kron(minus_block, minus_block), minus_block)

    return alpha * zero_L + beta * one_L

def apply_9qubit_error(state, error_type, qubit):
    """Apply a Pauli error to a specific qubit in the 9-qubit code."""
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    paulis = {'I': I, 'X': X, 'Y': Y, 'Z': Z}

    ops = [I] * 9
    ops[qubit] = paulis[error_type]

    # Build 9-qubit operator via tensor product
    error_op = ops[0]
    for i in range(1, 9):
        error_op = np.kron(error_op, ops[i])

    return error_op @ state

def measure_shor_syndrome(state):
    """Measure all 8 syndrome operators for Shor's 9-qubit code.

    Stabilizers:
    - Z1Z2, Z2Z3 (block 1 bit-flip)
    - Z4Z5, Z5Z6 (block 2 bit-flip)
    - Z7Z8, Z8Z9 (block 3 bit-flip)
    - X1X2X3X4X5X6 (blocks 1-2 phase comparison)
    - X4X5X6X7X8X9 (blocks 2-3 phase comparison)
    """
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def build_9qubit_op(op_list):
        """Build 9-qubit operator from list of single-qubit ops."""
        result = op_list[0]
        for i in range(1, 9):
            result = np.kron(result, op_list[i])
        return result

    def measure_stabilizer(state, ops):
        """Measure expectation value of a stabilizer."""
        op = build_9qubit_op(ops)
        return int(round(np.real(state.conj() @ op @ state)))

    # Bit-flip syndromes (Z-type)
    bit_syndromes = []
    for block_start in [0, 3, 6]:
        for pair in [(0, 1), (1, 2)]:
            ops = [I] * 9
            ops[block_start + pair[0]] = Z
            ops[block_start + pair[1]] = Z
            val = measure_stabilizer(state, ops)
            bit_syndromes.append(val)

    # Phase-flip syndromes (X-type)
    phase_syndromes = []
    for block_pair in [(0, 3), (3, 6)]:
        ops = [I] * 9
        for i in range(block_pair[0], block_pair[0] + 3):
            ops[i] = X
        for i in range(block_pair[1], block_pair[1] + 3):
            ops[i] = X
        val = measure_stabilizer(state, ops)
        phase_syndromes.append(val)

    return bit_syndromes, phase_syndromes

def correct_shor_code(state, bit_syndromes, phase_syndromes):
    """Correct errors based on Shor code syndromes.

    Why separate bit-flip and phase-flip correction? Shor's code has a
    hierarchical structure: bit flips are corrected within each block of 3,
    then phase flips are corrected between blocks.
    """
    # Bit-flip correction (within each block)
    for block in range(3):
        s1 = bit_syndromes[2 * block]
        s2 = bit_syndromes[2 * block + 1]

        if s1 == -1 and s2 == 1:
            qubit = block * 3 + 0
            state = apply_9qubit_error(state, 'X', qubit)
        elif s1 == -1 and s2 == -1:
            qubit = block * 3 + 1
            state = apply_9qubit_error(state, 'X', qubit)
        elif s1 == 1 and s2 == -1:
            qubit = block * 3 + 2
            state = apply_9qubit_error(state, 'X', qubit)

    # Phase-flip correction (between blocks)
    p1, p2 = phase_syndromes
    if p1 == -1 and p2 == 1:
        # Phase error in block 1
        state = apply_9qubit_error(state, 'Z', 0)
    elif p1 == -1 and p2 == -1:
        # Phase error in block 2
        state = apply_9qubit_error(state, 'Z', 3)
    elif p1 == 1 and p2 == -1:
        # Phase error in block 3
        state = apply_9qubit_error(state, 'Z', 6)

    return state

# Test Shor's 9-qubit code with all error types
print("=" * 60)
print("Shor's 9-Qubit Code: Correcting X, Y, Z Errors")
print("=" * 60)

alpha, beta = np.sqrt(0.3), np.sqrt(0.7) * np.exp(1j * np.pi / 4)
print(f"\nLogical state: α={alpha:.4f}, β={beta:.4f}")

for error_type in ['X', 'Y', 'Z']:
    for error_qubit in [0, 4, 8]:  # Test one qubit in each block
        state = encode_shor_9qubit(alpha, beta)
        state_err = apply_9qubit_error(state, error_type, error_qubit)

        bit_syn, phase_syn = measure_shor_syndrome(state_err)
        corrected = correct_shor_code(state_err, bit_syn, phase_syn)

        # Check fidelity
        fidelity = abs(np.dot(state.conj(), corrected))**2
        print(f"  {error_type} on q{error_qubit}: "
              f"bit_syn={['+' if s==1 else '-' for s in bit_syn]}, "
              f"phase_syn={['+' if s==1 else '-' for s in phase_syn]}, "
              f"fidelity={fidelity:.6f}")
```

### 10.3 물리적 오류율 대 논리적 오류율

```python
import numpy as np

def simulate_error_correction_threshold(physical_error_rate, code_distance,
                                         n_trials=10000):
    """Simulate error correction to demonstrate the threshold effect.

    For a distance-d code, logical errors occur when ⌈d/2⌉ or more physical
    qubits have errors. Below threshold, increasing d exponentially suppresses
    the logical error rate. Above threshold, increasing d makes things worse.
    """
    n_qubits = code_distance  # Simplified: n = d for repetition code
    t = (code_distance - 1) // 2  # Number of correctable errors

    logical_errors = 0
    for _ in range(n_trials):
        # Each qubit has independent error with probability p
        errors = np.random.random(n_qubits) < physical_error_rate
        n_errors = np.sum(errors)

        # Code fails if more than t errors occur
        if n_errors > t:
            logical_errors += 1

    return logical_errors / n_trials

print("=" * 60)
print("Error Correction Threshold Demonstration")
print("=" * 60)
print("\nLogical error rate vs physical error rate for various code distances:")
print(f"{'p_phys':>8} {'d=3':>10} {'d=5':>10} {'d=7':>10} {'d=9':>10} {'d=11':>10}")

for p in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3]:
    rates = []
    for d in [3, 5, 7, 9, 11]:
        rate = simulate_error_correction_threshold(p, d, n_trials=50000)
        rates.append(rate)
    print(f"{p:8.3f} {rates[0]:10.6f} {rates[1]:10.6f} {rates[2]:10.6f} "
          f"{rates[3]:10.6f} {rates[4]:10.6f}")

print("\nNote: Below the threshold (~0.11 for repetition code),")
print("increasing d dramatically reduces logical error rate.")
print("Above threshold, increasing d makes things WORSE.")
```

---

## 11. 연습 문제

### 연습 1: 비트 반전 코드 직접 계산

인코딩된 상태 $|\psi_L\rangle = \frac{1}{\sqrt{3}}|000\rangle + \sqrt{\frac{2}{3}}|111\rangle$ (비표준 중첩)를 고려하자.

(a) 큐비트 2(중간 큐비트)에 비트 반전 오류($X$)를 적용하라. 결과 상태는?
(b) $\langle \psi'|Z_1 Z_2|\psi'\rangle$와 $\langle \psi'|Z_2 Z_3|\psi'\rangle$를 평가하여 신드롬을 계산하라.
(c) 신드롬에 기반하여 어떤 정정을 적용해야 하는가?
(d) 정정된 상태가 원래 인코딩된 상태와 같음을 검증하라.

### 연습 2: 위상 반전 코드

3-큐비트 위상 반전 코드를 구현하라:
(a) 아다마르 게이트와 CNOT 게이트로 인코딩 회로를 작성하라.
(b) 신드롬 연산자($X_1 X_2$와 $X_2 X_3$)를 정의하라.
(c) $|+\rangle$ 상태를 인코딩하고, 큐비트 2에 $Z$ 오류를 적용한 후 정정하는 시뮬레이션을 구현하라.
(d) 이 코드가 $X$ 오류에 실패함을 보여라.

### 연습 3: 오류 이산화

3-큐비트 비트 반전 코드의 큐비트 1에 연속 회전 오류 $R_x(\epsilon) = \cos(\epsilon/2)I - i\sin(\epsilon/2)X$를 고려하자.
(a) 인코딩된 상태 $\alpha|000\rangle + \beta|111\rangle$에 $R_x(\epsilon)$을 적용하라.
(b) 신드롬 측정 확률을 계산하라.
(c) 신드롬 측정 후 상태가 오류 없는 부분공간 또는 단일 비트 반전 부분공간으로 사영됨을 보여라 — 연속 오류가 "디지털화"되었다.
(d) $\epsilon$의 함수로 성공적인 정정의 확률을 계산하라.

### 연습 4: 스테인 코드 구현

스테인 $[[7, 1, 3]]$ 코드를 구현하라:
(a) 스테인 코드 안정자 생성원을 찾아 인코딩을 구현하라.
(b) 6개의 안정자 생성원 모두 코드 공간에서 고유값 $+1$을 가짐을 검증하라.
(c) 7개의 큐비트 각각에 단일 큐비트 $X$, $Y$, $Z$ 오류를 시뮬레이션하고 신드롬이 각 오류를 고유하게 식별함을 검증하라.

### 연습 5: 임계값 추정

몬테카를로 시뮬레이션을 사용하여:
(a) $d = 3, 5, 7, 9, 11$의 반복 코드에 대해 $p \in [0.001, 0.5]$에서 물리적 오류율의 함수로 논리적 오류율을 그려라.
(b) 모든 곡선이 교차하는 교차점으로 임계값 $p_{\text{th}}$를 추정하라.
(c) i.i.d. 비트 반전 노이즈에 대한 무한 반복 코드의 이론적 임계값 $p_{\text{th}} = 0.5$와 비교하라.
(d) 탈분극화(Depolarizing) 노이즈 모델(각각 확률 $p/3$로 $X$, $Y$, $Z$ 오류가 발생)에서는 어떻게 변하는가?

---

[← 이전: 쇼어의 인수분해 알고리즘](10_Shors_Algorithm.md) | [다음: 양자 텔레포테이션과 통신 →](12_Quantum_Teleportation.md)
