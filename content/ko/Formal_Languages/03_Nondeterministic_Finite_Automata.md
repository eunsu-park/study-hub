# 레슨 3: 비결정적 유한 오토마타(Nondeterministic Finite Automata, NFA)

**이전**: [결정적 유한 오토마타](./02_Deterministic_Finite_Automata.md) | **다음**: [정규 표현식](./04_Regular_Expressions.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 엡실론 전이(epsilon-transition)를 포함하여 NFA를 형식적으로 정의한다
2. 가능성의 트리(tree of possibilities)로 NFA 계산을 추적한다
3. 부분 집합 구성(subset construction)을 사용하여 NFA를 동등한 DFA로 변환한다
4. NFA와 DFA가 같은 언어 클래스를 인식함을 증명한다
5. 비결정성(nondeterminism)을 설계 도구로 활용한다

## 1. 동기

결정적 기계(deterministic machine)는 각 단계에서 정확히 하나의 선택을 한다. **비결정적(nondeterministic)** 기계는 동시에 여러 선택을 할 수 있다 — 가능한 모든 경로를 병렬로 탐색한다.

비결정성은 유한 오토마타의 계산 능력을 증가시키지 않지만(NFA는 정확히 정규 언어를 수리한다), 오토마타 설계를 훨씬 단순하게 만들 수 있다.

## 2. 형식적 정의

**NFA**는 5-튜플 $N = (Q, \Sigma, \delta, q_0, F)$로 정의되며, 각 구성 요소는 다음과 같다:

| 구성 요소 | 설명 |
|-----------|------|
| $Q$ | 유한한 상태 집합 |
| $\Sigma$ | 유한한 입력 알파벳 |
| $\delta: Q \times (\Sigma \cup \{\varepsilon\}) \to \mathcal{P}(Q)$ | 전이 함수 |
| $q_0 \in Q$ | 시작 상태 |
| $F \subseteq Q$ | 수리 상태 집합 |

DFA와의 핵심 차이점:
- $\delta$는 상태들의 **집합(set)**을 반환한다(공집합일 수도 있음): $\delta(q, a) \subseteq Q$
- $\delta$는 $\Sigma \cup \{\varepsilon\}$ 위에서 정의되므로, 입력을 소비하지 않는 **엡실론 전이(epsilon-transition)**가 허용된다

## 3. NFA 계산

NFA는 수리 상태로 이어지는 전이 시퀀스가 **적어도 하나 존재하면** 문자열 $w$를 **수리(accept)**한다. 모든 경로가 비수리 상태로 끝나거나 소멸(이용 가능한 전이가 없는 상태에 도달)하는 경우에만 $w$를 **거부(reject)**한다.

가능성의 트리로 생각할 수 있다:
- 각 가지(branch)는 하나의 선택을 나타낸다
- **어떤** 가지든 수리 상태에 도달하면 해당 문자열은 수리된다
- "소멸" 가지(이용 가능한 전이 없음)는 단순히 종료된다

## 4. 엡실론-폐포(Epsilon-Closure)

상태 $q$의 **엡실론-폐포(epsilon-closure)** $\text{ECLOSE}(q)$는 $q$에서 0개 이상의 $\varepsilon$-전이를 따라 도달 가능한 상태들의 집합이다:

$$\text{ECLOSE}(q) = \{q\} \cup \{p \mid p \text{ is reachable from } q \text{ via } \varepsilon\text{-transitions}\}$$

상태 집합 $S$에 대해: $\text{ECLOSE}(S) = \bigcup_{q \in S} \text{ECLOSE}(q)$

## 5. NFA의 확장 전이 함수

$$\hat{\delta}(q, \varepsilon) = \text{ECLOSE}(q)$$
$$\hat{\delta}(q, wa) = \text{ECLOSE}\left(\bigcup_{p \in \hat{\delta}(q, w)} \delta(p, a)\right)$$

NFA의 언어: $L(N) = \{w \in \Sigma^* \mid \hat{\delta}(q_0, w) \cap F \neq \emptyset\}$

## 6. NFA 설계 예시

### 6.1 "01"로 끝나는 문자열

```
States: {q0, q1, q2}
Start: q0
Accept: {q2}

δ(q0, 0) = {q0, q1}   δ(q0, 1) = {q0}
δ(q1, 0) = ∅           δ(q1, 1) = {q2}
δ(q2, 0) = ∅           δ(q2, 1) = ∅
```

이 3-상태 NFA를 레슨 2의 3-상태 DFA와 비교해보자. NFA는 접미사 "01"이 시작되는 위치를 "추측(guess)"한다.

### 6.2 "aba" 또는 "bab"를 포함하는 문자열

NFA는 어떤 패턴을 찾을지 비결정적으로 추측할 수 있다:

```
States: {q0, qa1, qa2, qa3, qb1, qb2, qb3}
Start: q0
Accept: {qa3, qb3}

From q0: ε-transitions to qa1 and qb1 (start both searches)
  also: δ(q0, a) = {q0}, δ(q0, b) = {q0}  (stay in q0)

qa1 --a--> qa2 --b--> qa3(accept) --a--> qa3 --b--> qa3
qb1 --b--> qb2 --a--> qb3(accept) --a--> qb3 --b--> qb3
```

DFA에서는 두 패턴을 동시에 처리하기 위해 신중한 상태 추적이 필요하다. NFA 설계가 더 직관적이다.

## 7. DFA와 NFA의 동등성

**정리.** 어떤 NFA에 의해 수리되는 언어는 어떤 DFA에 의해서도 수리되며, 그 역도 성립한다.

*증명 방향: NFA → DFA* (부분 집합 구성(Subset Construction))

NFA $N = (Q, \Sigma, \delta, q_0, F)$가 주어졌을 때, DFA $D = (Q', \Sigma, \delta', q_0', F')$를 다음과 같이 구성한다:

- $Q' = \mathcal{P}(Q)$ — 각 DFA 상태는 NFA 상태들의 **집합**
- $q_0' = \text{ECLOSE}(q_0)$
- $\delta'(S, a) = \text{ECLOSE}\left(\bigcup_{q \in S} \delta(q, a)\right)$
- $F' = \{S \in Q' \mid S \cap F \neq \emptyset\}$

*증명 방향: DFA → NFA*는 자명하다 — 모든 DFA는 이미 NFA이다(단원소 전이 집합이며 $\varepsilon$-전이 없음).

### 7.1 부분 집합 구성 예시

"01"로 끝나는 문자열에 대한 NFA를 고려한다:

```
NFA states: {q0, q1, q2}
Start: q0, Accept: {q2}
```

부분 집합 구성:

| DFA 상태 | 0 입력 시 | 1 입력 시 |
|-----------|-----------|-----------|
| $\{q_0\}$ | $\{q_0, q_1\}$ | $\{q_0\}$ |
| $\{q_0, q_1\}$ | $\{q_0, q_1\}$ | $\{q_0, q_2\}$ |
| $\{q_0, q_2\}$ | $\{q_0, q_1\}$ | $\{q_0\}$ |

수리 DFA 상태: $\{q_0, q_2\}$ ($q_2 \in F$를 포함).

결과는 3-상태 DFA로, 이전에 설계한 것과 일치한다.

### 7.2 지수적 상태 폭발(Exponential Blowup)

부분 집합 구성은 최대 $2^{|Q|}$개의 DFA 상태를 생성할 수 있다. 이 폭발이 불가피한 경우도 있다:

**예시:** 언어 $L_n = \{w \in \{0,1\}^* \mid \text{the } n\text{th-from-last symbol is } 1\}$

- NFA는 $n + 1$개의 상태만 필요하다(위치를 추측)
- 임의의 DFA는 최소 $2^n$개의 상태를 필요로 한다($n$개 기호의 윈도우를 추적해야 함)

이 지수적 격차는 비결정성이 능력을 증가시키지 않더라도 실질적인 **간결성(succinctness)**을 제공함을 보여준다.

## 8. 연접(Concatenation)에 대한 폐쇄

NFA는 폐쇄 성질을 쉽게 증명할 수 있게 한다.

**정리.** $L_1$과 $L_2$가 정규 언어이면, $L_1 \cdot L_2$도 정규 언어이다.

*증명.* $L_1$에 대한 NFA $N_1$과 $L_2$에 대한 NFA $N_2$가 주어졌을 때:
1. $N_1$의 모든 수리 상태에서 $N_2$의 시작 상태로 $\varepsilon$-전이를 추가한다
2. $N_1$의 수리 상태들을 비수리 상태로 만든다
3. 시작 상태는 $N_1$의 시작 상태이고, 수리 상태는 $N_2$의 수리 상태이다

결과 NFA는 $L_1$과 $L_2$ 사이에서 입력을 어디서 분리할지 "추측"한다. $\square$

## 9. 클리네 별(Kleene Star)에 대한 폐쇄

**정리.** $L$이 정규 언어이면, $L^*$도 정규 언어이다.

*증명.* $L$에 대한 NFA $N$이 주어졌을 때:
1. 새로운 시작 상태 $q_s$를 추가한다(수리 상태이기도 하며, $\varepsilon \in L^*$를 위해)
2. $q_s$에서 원래 시작 상태로 $\varepsilon$-전이를 추가한다
3. $N$의 모든 수리 상태에서 원래 시작 상태로 $\varepsilon$-전이를 추가한다

NFA는 $N$을 임의의 횟수만큼 반복할 수 있다. $\square$

## 10. 증명 기법으로서의 비결정성

비결정성은 물리적으로 구현 가능하지 않지만, 다음 용도로 매우 유용하다:
- **설계 도구(design tool)**: 올바르게 "추측"하는 단순한 오토마타를 구성
- **증명 기법(proof technique)**: 오토마타를 결합하여 폐쇄 성질 증명
- **이론적 개념(theoretical concept)**: 복잡도 이론의 핵심(P vs NP)

DFA-NFA 동등성은 편의에 따라 비결정성을 자유롭게 사용할 수 있음을 의미하며, (더 클 수 있는) 결정적 동등본이 항상 존재함을 보장한다.

## 연습 문제

### 연습 1: NFA 설계

다음 언어에 대한 NFA를 설계하라(그림으로 그리거나 형식적으로 명시):

1. $L = \{w \in \{a,b\}^* \mid w \text{ contains } aab \text{ as a substring}\}$
2. $L = \{w \in \{0,1\}^* \mid \text{the third symbol from the end is } 0\}$
3. $L = \{w \in \{a,b\}^* \mid |w| \equiv 0 \pmod{2} \text{ or } w \text{ ends with } ab\}$

### 연습 2: 부분 집합 구성

다음 NFA에 대해 부분 집합 구성을 수행하라:

```
States: {q0, q1, q2}, Start: q0, Accept: {q2}
δ(q0, a) = {q0, q1}   δ(q0, b) = {q0}
δ(q1, a) = ∅           δ(q1, b) = {q2}
δ(q2, a) = ∅           δ(q2, b) = ∅
```

도달 가능한 모든 DFA 상태와 그 전이를 나열하라.

### 연습 3: 연접 구성

$L_1 = \{a^n \mid n \geq 1\}$과 $L_2 = \{b^n \mid n \geq 1\}$에 대한 NFA가 주어졌을 때, 8절의 연접 구성을 사용하여 $L_1 \cdot L_2$에 대한 NFA를 구성하라.

---

*이전: [레슨 2 — 결정적 유한 오토마타](02_Deterministic_Finite_Automata.md) | 다음: [레슨 4 — 정규 표현식](04_Regular_Expressions.md)*
