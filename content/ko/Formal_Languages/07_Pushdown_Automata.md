# 레슨 7: 푸시다운 오토마타(Pushdown Automata)

**이전**: [문맥 자유 문법](./06_Context_Free_Grammars.md) | **다음**: [문맥 자유 언어의 성질](./08_Properties_of_Context_Free_Languages.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 푸시다운 오토마타(pushdown automata)를 형식적으로 정의한다
2. 유한 오토마타를 확장하는 데 있어 스택(stack)의 역할을 이해한다
3. CFG와 PDA 사이를 변환한다
4. 최종 상태에 의한 수리(acceptance by final state)와 빈 스택에 의한 수리(acceptance by empty stack)를 구별한다
5. 결정론적 PDA(deterministic PDA)와 비결정론적 PDA를 비교하고 그 한계를 파악한다

## 1. 동기(Motivation)

유한 오토마타(finite automata)는 현재 상태 이외의 기억 수단이 없기 때문에 $\{0^n 1^n \mid n \geq 0\}$을 인식할 수 없다. **푸시다운 오토마타(pushdown automaton, PDA)**는 유한 오토마타에 **스택(stack)** — 한계가 없는 LIFO(Last-In, First-Out) 자료 구조 — 을 추가하여 확장한 것이다.

**비유:** 구내식당의 쟁반 디스펜서를 떠올려 보자 -- 쟁반을 맨 위에 올려놓거나(push) 맨 위의 쟁반만 꺼낼 수(pop) 있고, 아래에 있는 쟁반은 볼 수도 꺼낼 수도 없다. PDA의 스택은 이와 정확히 같은 방식으로 동작한다: 나중에 들어온 것이 먼저 나간다(Last-In, First-Out). 이 단순한 메커니즘은 놀랍도록 강력하여, 기계가 여는 괄호를 몇 개 보았는지 "기억"하고 닫는 괄호와 대응시킬 수 있게 해 주는데, 이것이 바로 유한 오토마타가 할 수 없는 것이다.

스택은 문맥 자유 언어(context-free language)를 인식하기에 정확히 적합한 계산 능력을 제공한다.

## 2. 형식적 정의

**푸시다운 오토마타(pushdown automaton)**는 6-튜플 $P = (Q, \Sigma, \Gamma, \delta, q_0, F)$로 정의된다:

| 구성 요소 | 설명 |
|-----------|------|
| $Q$ | 상태의 유한 집합 |
| $\Sigma$ | 유한 입력 알파벳 |
| $\Gamma$ | 유한 **스택 알파벳(stack alphabet)** |
| $\delta: Q \times (\Sigma \cup \{\varepsilon\}) \times (\Gamma \cup \{\varepsilon\}) \to \mathcal{P}(Q \times (\Gamma \cup \{\varepsilon\}))$ | 전이 함수 |
| $q_0 \in Q$ | 시작 상태 |
| $F \subseteq Q$ | 수리 상태 집합 |

전이 $\delta(q, a, X) \ni (p, Y)$의 의미: 상태 $q$에서 입력 $a$ (또는 $\varepsilon$)를 읽고, 스택 최상단에 $X$가 있을 때 (또는 스택을 읽지 않으면 $\varepsilon$), 상태 $p$로 이동하고 $X$를 $Y$로 교체한다 (또는 $Y$를 푸시하거나 $X$를 팝).

### 2.1 스택 연산(Stack Operations)

전이 함수는 스택 연산을 다음과 같이 인코딩한다:

| 전이 | 스택 효과 |
|------|----------|
| $\delta(q, a, \varepsilon) \ni (p, X)$ | $X$를 **푸시(push)** (스택 읽기 없음) |
| $\delta(q, a, X) \ni (p, \varepsilon)$ | $X$를 **팝(pop)** |
| $\delta(q, a, X) \ni (p, Y)$ | 최상단 $X$를 $Y$로 **교체(replace)** |
| $\delta(q, a, X) \ni (p, YZ)$ | $X$를 $YZ$로 **교체** ($Y$가 최상단) |
| $\delta(q, a, \varepsilon) \ni (p, \varepsilon)$ | 스택 변화 없음 |

참고: PDA는 본질적으로 **비결정론적(nondeterministic)** — $\delta$는 집합을 반환한다.

## 3. 구성과 계산(Configuration and Computation)

PDA의 **구성(configuration)** (또는 순간 기술(instantaneous description))은 다음 세 값으로 이루어진 트리플 $(q, w, \gamma)$이다:
- $q$: 현재 상태
- $w$: 남은 입력
- $\gamma$: 스택의 내용 (왼쪽이 최상단)

**도출(yields)** 관계 $\vdash$는 한 계산 단계를 기술한다:
$$(q, aw, X\gamma) \vdash (p, w, Y\gamma) \quad \text{if } (p, Y) \in \delta(q, a, X)$$

$\vdash^*$는 반사적 전이 폐포(reflexive transitive closure)를 나타낸다.

## 4. 수리 방식(Acceptance Modes)

### 4.1 최종 상태에 의한 수리(Acceptance by Final State)

어떤 $q \in F$와 임의의 $\gamma$에 대해 $(q_0, w, Z_0) \vdash^* (q, \varepsilon, \gamma)$가 성립하면 PDA가 $w$를 수리한다.

$$L(P) = \{w \mid (q_0, w, Z_0) \vdash^* (q, \varepsilon, \gamma), q \in F\}$$

(여기서 $Z_0$는 초기 스택 기호이며, 정의의 일부로 포함되는 경우도 있다.)

### 4.2 빈 스택에 의한 수리(Acceptance by Empty Stack)

임의의 상태 $q$에 대해 $(q_0, w, Z_0) \vdash^* (q, \varepsilon, \varepsilon)$가 성립하면 PDA가 $w$를 수리한다.

$$N(P) = \{w \mid (q_0, w, Z_0) \vdash^* (q, \varepsilon, \varepsilon)\}$$

### 4.3 수리 방식의 동등성(Equivalence of Acceptance Modes)

**정리.** 최종 상태로 수리하는 PDA $P_1$에 대해, $N(P_2) = L(P_1)$을 만족하며 빈 스택으로 수리하는 PDA $P_2$가 존재한다. 역방향도 성립한다.

변환에는 표준 구성이 사용된다:
- **최종 상태 → 빈 스택**: 수리 상태에 도달한 후 스택의 모든 내용을 팝
- **빈 스택 → 최종 상태**: 바닥 마커(bottom marker)를 사용하여 빈 스택을 감지하고 수리 상태로 전이

## 5. PDA 설계 예제(PDA Design Examples)

### 5.1 $L = \{0^n 1^n \mid n \geq 0\}$

**전략:** 각 0에 대해 기호를 푸시하고, 각 1에 대해 팝한다. 전체 입력을 읽은 후 스택이 비어있으면 수리.

```
States: {q0, q1, q2}
Start: q0, Accept: {q2}
Stack alphabet: {Z, X}  (Z = bottom marker)

δ(q0, ε, ε) = {(q1, Z)}     -- push bottom marker
δ(q1, 0, ε) = {(q1, X)}     -- push X for each 0
δ(q1, 1, X) = {(q2, ε)}     -- start popping
δ(q2, 1, X) = {(q2, ε)}     -- keep popping
δ(q2, ε, Z) = {(q2, ε)}     -- pop marker, accept by empty stack
```

### 5.2 $L = \{ww^R \mid w \in \{a,b\}^*\}$ (짝수 길이 팰린드롬(Even-Length Palindromes))

**전략:** 전반부를 스택에 푸시한 후, 후반부와 스택을 비교하여 매칭.

```
States: {q_push, q_pop, q_accept}
Start: q_push, Accept: {q_accept}

-- Push phase: nondeterministically guess the middle
δ(q_push, a, ε) = {(q_push, a)}
δ(q_push, b, ε) = {(q_push, b)}
δ(q_push, ε, ε) = {(q_pop, ε)}    -- guess: we're at the middle

-- Pop phase: match input with stack
δ(q_pop, a, a) = {(q_pop, ε)}
δ(q_pop, b, b) = {(q_pop, ε)}
δ(q_pop, ε, Z) = {(q_accept, ε)}  -- stack empty = matched
```

비결정론(nondeterminism)이 필수적이다: PDA는 중간 지점을 추측한다.

### 5.3 $L = \{a^i b^j c^k \mid i = j \text{ or } j = k\}$

PDA는 비결정론적으로 어느 등식을 검증할지 선택한다:

- **분기 1:** $a$를 푸시하고, $b$와 매칭한 후, $c$는 건너뜀
- **분기 2:** $a$를 건너뛰고, $b$를 푸시한 후, $c$와 매칭

## 6. PDA와 CFG의 동등성(Equivalence of PDA and CFG)

**정리.** 어떤 언어가 문맥 자유(context-free)인 것은 그 언어가 어떤 PDA에 의해 수리되는 것과 동치이다.

### 6.1 CFG → PDA

CFG $G = (V, \Sigma, R, S)$가 주어지면, 최좌 유도(leftmost derivation)를 시뮬레이션하는 PDA를 구성한다:

1. $S$를 스택에 푸시
2. 반복:
   - 스택 최상단이 변수 $A$이면: 비결정론적으로 규칙 $A \to \alpha$를 선택하고 $A$를 $\alpha$로 교체
   - 스택 최상단이 단말 기호 $a$이면: 입력에서 $a$를 읽고 팝

이 PDA는 단 3개의 상태(시작, 루프, 수리)를 가지며, 모든 작업을 스택에서 수행한다.

**실행 추적 예시:** 문법 $S \to aSb \mid \varepsilon$을 변환하여 입력 `aabb`에 대한 PDA의 동작을 추적해 보자. PDA는 스택에 $S$를 넣고 시작하며, 위의 규칙을 적용한다.

| 단계 | 상태 | 남은 입력 | 스택 (왼쪽이 최상단) | 동작 |
|------|------|----------|---------------------|------|
| 0 | start | `aabb` | (비어 있음) | $S$ 푸시, loop 상태로 이동 |
| 1 | loop | `aabb` | $S$ | $S$를 $aSb$로 교체 (규칙 $S \to aSb$ 선택) |
| 2 | loop | `aabb` | $a\,S\,b$ | 최상단이 단말 $a$; 입력의 $a$와 매칭, 팝 |
| 3 | loop | `abb` | $S\,b$ | $S$를 $aSb$로 교체 (규칙 $S \to aSb$ 선택) |
| 4 | loop | `abb` | $a\,S\,b\,b$ | 최상단이 단말 $a$; 입력의 $a$와 매칭, 팝 |
| 5 | loop | `bb` | $S\,b\,b$ | $S$를 $\varepsilon$으로 교체 (규칙 $S \to \varepsilon$ 선택), $S$ 팝 |
| 6 | loop | `bb` | $b\,b$ | 최상단이 단말 $b$; 입력의 $b$와 매칭, 팝 |
| 7 | loop | `b` | $b$ | 최상단이 단말 $b$; 입력의 $b$와 매칭, 팝 |
| 8 | loop | (비어 있음) | (비어 있음) | 입력 소진, 스택 비어 있음 -- 수리! |

각 단계에서 PDA는 문법 규칙을 사용하여 변수를 확장하거나, 단말 기호를 다음 입력 기호와 매칭한다. 비결정론(nondeterminism)은 어떤 문법 규칙을 적용할지 선택하는 부분(1단계 vs. 5단계)에 있다.

### 6.2 PDA → CFG

PDA $P$가 주어지면, 변수가 트리플 $[q, A, p]$에 대응하는 CFG를 구성한다. 이 트리플은 "상태 $q$에서 시작하여 스택 최상단에 $A$가 있을 때, 상태 $p$에 도달하면서 $A$가 팝된다"를 나타낸다.

구성 방법은 더 복잡하지만 체계적이다:
- 푸시하는 각 전이에 대해: 푸시와 최종 팝을 연결하는 규칙을 생성
- 각 상태 쌍에 대해: 그 사이의 계산을 나타내는 변수를 생성

## 7. 결정론적 푸시다운 오토마타(Deterministic Pushdown Automata)

PDA가 **결정론적(deterministic, DPDA)**이라는 것은 모든 구성에 대해 적용 가능한 전이가 최대 하나임을 의미한다:

모든 $q \in Q$, $a \in \Sigma$, $X \in \Gamma$에 대해:
- $|\delta(q, a, X)| + |\delta(q, \varepsilon, X)| + |\delta(q, a, \varepsilon)| + |\delta(q, \varepsilon, \varepsilon)| \leq 1$

(각 상태, 입력, 스택 조합에 대해 적용 가능한 전이가 최대 하나.)

### 7.1 DPDA의 계산 능력(DPDA Power)

**중요:** DPDA는 비결정론적 PDA보다 **진정으로 약하다(strictly weaker)**!

- DPDA는 정확히 **결정론적 문맥 자유 언어(deterministic context-free language, DCFL)**를 인식한다
- DCFL $\subsetneq$ CFL
- 예제: $\{ww^R \mid w \in \{a,b\}^*\}$는 CFL이지만 DCFL이 아니다
- 예제: $\{wcw^R \mid w \in \{a,b\}^*\}$는 DCFL이다 (중간 마커 $c$가 추측의 필요성을 제거)

이는 DFA와 NFA의 계산 능력이 동등한 유한 오토마타와 대조된다.

### 7.2 DCFL의 성질(Properties of DCFL)

- 여집합(complement)에 대해 닫혀 있다 (일반 CFL과 달리!)
- 합집합(union), 교집합(intersection), 연결(concatenation), 클리니 스타(Kleene star)에 대해서는 닫혀 있지 **않다**
- 모든 DCFL은 비모호(unambiguous) 문법을 가진다
- DCFL은 LR(k) 파싱 가능 — 실용적인 파서 생성기(parser generator)의 기반

## 8. PDA 변형(PDA Variants)

| 변형 | 설명 | 계산 능력 |
|------|------|----------|
| 표준 PDA | 위에서 정의한 것 | CFL |
| 두 스택 PDA(Two-stack PDA) | 두 개의 독립적인 스택 | 튜링 기계와 동등! |
| 큐를 사용하는 PDA(PDA with queue) | 스택을 큐로 교체 | 튜링 기계와 동등! |
| 카운터 오토마타(Counter automaton) | $\{Z, X\}$에 대한 스택만 사용 | DFA와 PDA 사이의 엄격한 중간 |

두 번째 스택을 추가하거나 스택을 큐로 교체하면 모델이 튜링 완전(Turing-complete)이 된다 — 이는 무제한 메모리의 계산 능력에 대한 힌트이다.

## 연습 문제

### 연습 1: PDA 설계

다음 언어에 대한 PDA를 설계하라:

1. $L = \{a^n b^{2n} \mid n \geq 0\}$
2. $L = \{w \in \{a,b\}^* \mid |w|_a = |w|_b\}$
3. $L = \{a^i b^j c^k \mid i + k = j\}$

### 연습 2: CFG에서 PDA로

6.1절의 구성을 사용하여 다음 문법을 PDA로 변환하라:

$$S \to aSb \mid \varepsilon$$

입력 $aabb$에 대한 PDA의 계산 과정을 추적하라.

### 연습 3: 결정론성(Determinism)

다음 언어들 중 어떤 것이 결정론적 CFL인가? 이유를 설명하라.

1. $\{a^n b^n \mid n \geq 0\}$
2. $\{ww^R \mid w \in \{a,b\}^*\}$
3. $\{a^n b^n \mid n \geq 0\} \cup \{a^n b^{2n} \mid n \geq 0\}$

---

*이전: [레슨 6 — 문맥 자유 문법](06_Context_Free_Grammars.md) | 다음: [레슨 8 — 문맥 자유 언어의 성질](08_Properties_of_Context_Free_Languages.md)*
