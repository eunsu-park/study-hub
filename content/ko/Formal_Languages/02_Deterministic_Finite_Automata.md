# 레슨 2: 결정적 유한 오토마타(Deterministic Finite Automata, DFA)

**이전**: [형식 언어 입문](./01_Introduction_to_Formal_Languages.md) | **다음**: [비결정적 유한 오토마타](./03_Nondeterministic_Finite_Automata.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. DFA를 5-튜플(5-tuple)로 형식적으로 정의한다
2. 전이 함수(transition function)를 사용하여 입력 문자열에 대한 계산을 추적한다
3. 주어진 언어에 대한 DFA를 설계한다
4. DFA 구성의 정확성을 증명한다
5. 확장 전이 함수(extended transition function)를 적용한다

---

DFA는 촘스키 계층(Chomsky hierarchy)에서 가장 단순한 계산 모델이지만, 컴파일러의 어휘 분석기(lexical analyzer)부터 텍스트 편집기의 패턴 매칭(pattern matching)까지 가장 널리 사용되는 도구들을 뒷받침한다. DFA를 이해하는 것은 이후 배울 모든 오토마톤의 기초를 쌓는 일이다.

## 1. 비형식적 설명

**결정적 유한 오토마타(Deterministic Finite Automaton, DFA)**는 가장 단순한 계산 모델이다. 입력 문자열을 한 번에 하나의 기호씩 읽으며, 유한한 수의 상태 사이를 전이하고, 최종적으로 입력을 수리(accept)하거나 거부(reject)한다.

핵심 특성:
- **결정적(Deterministic)**: 각 상태-기호 쌍에 대해 정확히 하나의 전이만 존재
- **유한(Finite)**: 제한된 메모리(유한한 수의 상태)
- 현재 상태 이외의 보조 저장소 없음

> **비유 -- 회전식 개찰구(Turnstile).** DFA는 지하철 회전식 개찰구와 같다. 고정된 상태 집합(잠김, 열림)이 있고, 한 번에 하나의 입력(동전, 밀기)을 읽으며, 결정론적으로 전이한다 -- 같은 상태에서 같은 입력은 항상 같은 결과를 낸다. 추측도 없고 되돌아가기도 없으며, 토큰을 수리하거나 잠긴 채로 유지하는 단순한 메커니즘이다.

## 2. 형식적 정의

DFA는 5-튜플 $M = (Q, \Sigma, \delta, q_0, F)$로 정의되며, 각 구성 요소는 다음과 같다:

| 구성 요소 | 설명 |
|-----------|------|
| $Q$ | 유한한 **상태(states)** 집합 |
| $\Sigma$ | 유한한 **입력 알파벳(input alphabet)** |
| $\delta: Q \times \Sigma \to Q$ | **전이 함수(transition function)** |
| $q_0 \in Q$ | **시작 상태(start state)** |
| $F \subseteq Q$ | **수리(수락) 상태(accept/final states)** 집합 |

전이 함수 $\delta$는 **전체(total)** 함수이다: 모든 상태-기호 쌍에 대해 반드시 정의되어야 한다.

## 3. 상태 다이어그램

DFA는 방향 그래프(directed graph)로 표현되는 경우가 많다:
- 노드(node)는 상태를 나타낸다
- 기호로 레이블된 간선(edge)은 전이를 나타낸다
- 시작 상태는 외부에서 들어오는 화살표로 표시한다
- 수리 상태는 이중 원(double circle)으로 그린다

**예시:** $\{0, 1\}$ 위에서 1의 개수가 짝수인 문자열을 수리하는 DFA.

```
States: Q = {q_even, q_odd}
Alphabet: Σ = {0, 1}
Start: q_even
Accept: F = {q_even}

Transitions:
  δ(q_even, 0) = q_even
  δ(q_even, 1) = q_odd
  δ(q_odd, 0) = q_odd
  δ(q_odd, 1) = q_even
```

## 4. 확장 전이 함수

전이 함수 $\delta$는 기호 하나를 처리한다. 이를 문자열 처리로 확장하면:

**정의.** **확장 전이 함수(extended transition function)** $\hat{\delta}: Q \times \Sigma^* \to Q$는 다음과 같이 재귀적으로 정의된다:

$$\hat{\delta}(q, \varepsilon) = q$$
$$\hat{\delta}(q, wa) = \delta(\hat{\delta}(q, w), a) \quad \text{for } w \in \Sigma^*, a \in \Sigma$$

직관적으로, $\hat{\delta}(q, w)$는 상태 $q$에서 시작하여 문자열 $w$를 처리한 후 도달하는 상태이다.

## 5. DFA가 인식하는 언어

DFA $M$이 **인식(recognize)**하는(또는 수리하는) **언어(language)**는:

$$L(M) = \{w \in \Sigma^* \mid \hat{\delta}(q_0, w) \in F\}$$

어떤 DFA $M$에 대해 $L = L(M)$이 성립하면 언어 $L$은 **정규 언어(regular language)**이다.

## 6. DFA 설계 예시

### 6.1 "01"로 끝나는 문자열

**언어:** $L = \{w \in \{0,1\}^* \mid w \text{ ends with } 01\}$

**접근법:** 마지막으로 읽은 두 기호를 추적한다.

```
States: {q_start, q_0, q_01}
Start: q_start
Accept: {q_01}

δ(q_start, 0) = q_0      δ(q_start, 1) = q_start
δ(q_0, 0) = q_0           δ(q_0, 1) = q_01
δ(q_01, 0) = q_0          δ(q_01, 1) = q_start
```

### 6.2 "ab"를 부분 문자열로 포함하는 문자열

**언어:** $L = \{w \in \{a,b\}^* \mid ab \text{ is a substring of } w\}$

```
States: {q0, q_a, q_ab}
Start: q0
Accept: {q_ab}

δ(q0, a) = q_a      δ(q0, b) = q0
δ(q_a, a) = q_a     δ(q_a, b) = q_ab
δ(q_ab, a) = q_ab   δ(q_ab, b) = q_ab
```

상태 `q_ab`는 **트랩 상태(trap state)**(또는 **흡수 상태(absorbing state)**)이다 — 한번 진입하면 입력에 관계없이 DFA는 그 상태에 머문다.

### 6.3 3으로 나누어지는 이진수

**언어:** $L = \{w \in \{0,1\}^* \mid w \text{ represents a number divisible by 3}\}$

**핵심 통찰:** 현재까지 읽은 수의 값이 $n \pmod{3}$일 때, 비트 $b$를 읽으면 값은 $2n + b \pmod{3}$이 된다.

```
States: {r0, r1, r2}  (remainder mod 3)
Start: r0
Accept: {r0}

δ(r0, 0) = r0   δ(r0, 1) = r1
δ(r1, 0) = r2   δ(r1, 1) = r0
δ(r2, 0) = r1   δ(r2, 1) = r2
```

## 7. DFA 곱 구성

DFA $M_1$과 $M_2$가 주어졌을 때, $L(M_1) \cap L(M_2)$와 $L(M_1) \cup L(M_2)$에 대한 DFA를 구성할 수 있다.

$M_1 = (Q_1, \Sigma, \delta_1, q_{01}, F_1)$과 $M_2 = (Q_2, \Sigma, \delta_2, q_{02}, F_2)$에 대한 **곱 구성(product construction)**:

$M = (Q_1 \times Q_2, \Sigma, \delta, (q_{01}, q_{02}), F)$를 다음과 같이 정의한다:
- $\delta((p, q), a) = (\delta_1(p, a), \delta_2(q, a))$
- **교집합(intersection)**: $F = F_1 \times F_2$
- **합집합(union)**: $F = (F_1 \times Q_2) \cup (Q_1 \times F_2)$

**정리.** $L_1$과 $L_2$가 정규 언어이면, $L_1 \cap L_2$와 $L_1 \cup L_2$도 정규 언어이다.

## 8. 보체(Complement)

언어 $L$에 대한 DFA $M = (Q, \Sigma, \delta, q_0, F)$가 주어졌을 때, $\overline{L}$에 대한 DFA는:

$$M' = (Q, \Sigma, \delta, q_0, Q \setminus F)$$

단순히 수리 상태와 비수리 상태를 교환하면 된다!

**정리.** $L$이 정규 언어이면, $\overline{L}$도 정규 언어이다.

참고: 이 구성은 **완전한(complete)** DFA(전체 전이 함수를 갖는)를 필요로 한다. NFA에서 시작하는 경우 먼저 DFA로 변환해야 한다.

## 9. DFA 정확성 증명

DFA $M$이 언어 $L$을 올바르게 인식함을 증명하려면 $L(M) = L$임을 보인다:

**방법 — 루프 불변식(Loop Invariant):** 각 상태가 나타내는 것에 대한 주장을 정의한 후, 입력 길이에 대한 귀납법으로 증명한다.

**예시:** "1의 개수가 짝수" DFA가 올바름을 증명하기.

**주장:** 문자열 $w$를 읽은 후, $w$가 짝수 개의 1을 포함하는 경우 그리고 그 경우에만 DFA는 상태 $q_{even}$에 있다.

*$|w|$에 대한 귀납법으로 증명:*

**기저:** $|w| = 0$, 즉 $w = \varepsilon$. DFA는 $q_{even}$(시작 상태)에 있고, $\varepsilon$은 1이 0개(짝수)이다. ✓

**귀납 단계:** 문자열 $w$에 대해 주장이 성립한다고 가정한다. $a \in \{0, 1\}$일 때 $wa$를 고려한다.

- 경우 $a = 0$: 홀짝성(parity)이 바뀌지 않으며, DFA는 같은 상태에 머문다. ✓
- 경우 $a = 1$: 홀짝성이 뒤집히며, DFA는 $q_{even}$과 $q_{odd}$ 사이를 전이한다. ✓

$F = \{q_{even}\}$이므로, DFA는 $w$에 1이 짝수 개 있을 때 그리고 그때에만 $w$를 수리한다. $\square$

## 10. DFA의 한계

DFA는 **유한한 메모리**(오직 현재 상태만)를 갖는다. 따라서 다음을 수행할 수 없다:

- 무한정 세기(예: $n$개의 a와 $n$개의 b를 대응시키는 것 불가)
- 전체 입력 기억
- 스택 또는 기타 보조 저장소가 필요한 문제 해결

예를 들어, $L = \{a^n b^n \mid n \geq 0\}$을 인식하는 DFA는 존재하지 않는다. 레슨 5에서 펌핑 보조 정리(pumping lemma)를 사용하여 이를 형식적으로 증명할 것이다.

## 연습 문제

### 연습 1: DFA 구성

$\Sigma = \{0, 1\}$ 위의 각 언어에 대해 DFA(5개 구성 요소 모두 명시)를 설계하라:

1. $L = \{w \mid w \text{ contains the substring } 110\}$
2. $L = \{w \mid |w| \text{ is divisible by 3}\}$
3. $L = \{w \mid w \text{ does not contain two consecutive 1s}\}$

### 연습 2: 곱 구성

주어진 조건:
- $M_1$은 0의 개수가 짝수인 문자열을 수리한다
- $M_2$는 1의 개수가 홀수인 문자열을 수리한다

$L(M_1) \cap L(M_2)$에 대한 곱 DFA를 구성하라. 상태는 몇 개인가?

### 연습 3: 정확성 증명

6.3절의 "3으로 나누어지는" DFA가 3으로 나누어지는 이진수를 올바르게 인식함을 귀납법으로 증명하라. (힌트: 상태 $r_i$가 입력값이 $\equiv i \pmod{3}$임을 나타냄을 증명하라.)

---

*이전: [레슨 1 — 형식 언어 개론](01_Introduction_to_Formal_Languages.md) | 다음: [레슨 3 — 비결정적 유한 오토마타](03_Nondeterministic_Finite_Automata.md)*
