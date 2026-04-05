# 레슨 11: 환원 가능성(Reducibility)

**이전**: [결정 가능성](./10_Decidability.md) | **다음**: [계산 복잡도](./12_Computational_Complexity.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 언어 간의 사상 환원(mapping reduction)을 정의하고 적용한다
2. 환원을 이용하여 새로운 문제의 결정 불가능성을 증명한다
3. 라이스 정리(Rice's theorem)를 서술하고 적용한다
4. 포스트 대응 문제(Post Correspondence Problem)와 그 의의를 이해한다
5. 튜링 환원(Turing reduction)과 사상 환원(mapping reduction)을 구별한다

## 1. 환원(Reduction)의 개념

**환원(reduction)**은 한 문제를 다른 문제로 변환한다. 문제 $A$가 문제 $B$로 환원된다면:
- $B$에 대한 해결책이 $A$에 대한 해결책을 제공한다
- $A$가 어렵다면(결정 불가능하다면), $B$도 반드시 어렵다

환원은 결정 불가능성을 증명하는 주요 도구이다: 이미 알려진 결정 불가능한 문제를 대상 문제로 환원한다.

## 2. 사상 환원(Mapping Reductions)

### 2.1 정의

함수 $f: \Sigma^* \to \Sigma^*$가 **계산 가능(computable)**하다는 것은, 입력 $w$에 대해 테이프에 $f(w)$를 남기고 정지하는 TM이 존재한다는 의미이다.

언어 $A$가 언어 $B$로 **사상 환원(mapping reducible)**된다는 것은, $A \leq_m B$로 표기하며, 다음을 만족하는 계산 가능한 함수 $f$가 존재한다는 의미이다:

$$w \in A \iff f(w) \in B$$

함수 $f$를 **환원(reduction)**이라고 한다.

### 2.2 성질

**정리.** $A \leq_m B$이고 $B$가 결정 가능하면, $A$도 결정 가능하다.

*증명.* $B$의 결정기 $D_B$가 주어지면, $A$의 결정기를 구성한다: 입력 $w$에 대해 $f(w)$를 계산하고 $f(w)$에 대해 $D_B$를 실행한다. $\square$

**대우(Contrapositive).** $A$가 결정 불가능하고 $A \leq_m B$이면, $B$도 결정 불가능하다.

인식 가능성에 대해서도 유사하게 성립한다:
- $A \leq_m B$이고 $B$가 인식 가능하면, $A$도 인식 가능하다
- $A$가 인식 불가능하고 $A \leq_m B$이면, $B$도 인식 불가능하다

## 3. 환원 예제들(Reduction Examples)

### 3.1 $HALT_{TM}$은 결정 불가능하다

$HALT_{TM} = \{\langle M, w \rangle \mid M \text{이 입력 } w \text{에서 정지함}\}$

**환원:** $A_{TM} \leq_m HALT_{TM}$

$f(\langle M, w \rangle) = \langle M', w \rangle$으로 정의하며, $M'$은 다음과 같다:

> 입력 $x$에 대한 $M'$:
> 1. $x$에 대해 $M$을 시뮬레이션한다
> 2. $M$이 수락하면 수락한다
> 3. $M$이 거부하면 무한 루프에 진입한다

이제:
- $\langle M, w \rangle \in A_{TM}$ ($M$이 $w$를 수락) $\iff$ $M'$이 $w$에서 정지(수락) $\iff$ $\langle M', w \rangle \in HALT_{TM}$
- $\langle M, w \rangle \notin A_{TM}$ $\iff$ $M'$이 루프($M$이 루프하는 경우) 또는 루프($M$이 거부하는 경우) $\iff$ $\langle M', w \rangle \notin HALT_{TM}$ $\square$

### 3.2 $E_{TM}$은 결정 불가능하다

$E_{TM} = \{\langle M \rangle \mid L(M) = \emptyset\}$

**환원:** $A_{TM} \leq_m \overline{E_{TM}}$

$f(\langle M, w \rangle) = \langle M' \rangle$으로 정의하며, $M'$은 다음과 같다:

> 입력 $x$에 대한 $M'$:
> 1. $x$를 무시한다
> 2. $w$에 대해 $M$을 시뮬레이션한다
> 3. $M$이 $w$를 수락하면 수락한다

이제:
- $\langle M, w \rangle \in A_{TM}$ $\iff$ $L(M') = \Sigma^*$ $\iff$ $\langle M' \rangle \notin E_{TM}$ $\iff$ $\langle M' \rangle \in \overline{E_{TM}}$

따라서 $A_{TM} \leq_m \overline{E_{TM}}$이므로 $\overline{E_{TM}}$은 결정 불가능하고, 따라서 $E_{TM}$도 결정 불가능하다. $\square$

### 3.3 $EQ_{TM}$은 결정 불가능하다

$EQ_{TM} = \{\langle M_1, M_2 \rangle \mid L(M_1) = L(M_2)\}$

**환원:** $E_{TM} \leq_m EQ_{TM}$

$f(\langle M \rangle) = \langle M, M_\emptyset \rangle$으로 정의하며, $M_\emptyset$은 모든 입력을 거부한다.

$\langle M \rangle \in E_{TM} \iff L(M) = \emptyset = L(M_\emptyset) \iff \langle M, M_\emptyset \rangle \in EQ_{TM}$ $\square$

### 3.4 $REGULAR_{TM}$은 결정 불가능하다

$REGULAR_{TM} = \{\langle M \rangle \mid L(M) \text{이 정규 언어이다}\}$

**환원:** $A_{TM} \leq_m REGULAR_{TM}$ (실제로는 그 여에 대한 환원이거나, 라이스 정리를 사용한다 — 4절 참조).

## 4. 라이스 정리(Rice's Theorem)

라이스 정리는 많은 결정 불가능성 문제를 한 번에 해결하는 강력한 메타 결과이다.

### 4.1 언어의 성질(Property of Languages)

튜링 인식 가능 언어의 **성질(property)**이란 튜링 인식 가능 언어들의 집합 $P$이다. TM $M$이 "성질 $P$를 가진다"는 것은 $L(M) \in P$를 의미한다.

성질이 **비자명(nontrivial)**하다는 것은:
- 어떤 TM이 그 성질을 가진다 (즉, $P \neq \emptyset$)
- 어떤 TM은 그 성질을 갖지 않는다 (즉, $P$가 모든 RE 언어들의 집합이 아니다)

### 4.2 정리 서술

**정리 (라이스 정리).** $P$가 튜링 인식 가능 언어들의 어떤 비자명한 성질이라면, 언어

$$L_P = \{\langle M \rangle \mid L(M) \in P\}$$

는 결정 불가능하다.

### 4.3 증명

일반성을 잃지 않고(WLOG) $\emptyset \notin P$라고 가정한다 (그렇지 않으면 $\overline{P}$를 고려한다).

$P$가 비자명하므로, $L(T) \in P$인 TM $T$가 존재한다.

$A_{TM}$을 $L_P$로 환원한다: $\langle M, w \rangle$이 주어지면 $M'$을 구성한다:

> 입력 $x$에 대한 $M'$:
> 1. $w$에 대해 $M$을 시뮬레이션한다
> 2. $M$이 $w$를 수락하면, $x$에 대해 $T$를 시뮬레이션하고 그 답을 출력한다
> 3. $M$이 $w$를 거부하거나 루프에 빠지면, 영원히 루프한다

이제:
- $M$이 $w$를 수락하는 경우: $L(M') = L(T) \in P$이므로 $\langle M' \rangle \in L_P$
- $M$이 $w$를 수락하지 않는 경우: $L(M') = \emptyset \notin P$이므로 $\langle M' \rangle \notin L_P$

따라서 $\langle M, w \rangle \in A_{TM} \iff \langle M' \rangle \in L_P$이며, $A_{TM} \leq_m L_P$가 성립한다. $\square$

**쉬운 말로 요약:** 라이스 정리(Rice's theorem)는 **다른 프로그램이 무엇을 계산하는지에 대한 "흥미로운" 질문에 신뢰성 있게 답하는 프로그램을 작성할 수 없다**고 말한다. 여기서 "흥미로운"이란 프로그램의 소스 코드가 아닌 입출력 동작(그 언어)에 관한 질문이며, 답이 자명하게 항상-예 또는 항상-아니오가 아닌 것을 뜻한다. 예를 들어, 임의의 소스 코드를 보고 그것이 공 함수를 계산하는지, `"hello"`를 출력하는 경우가 있는지, 정규 언어(regular language)를 인식하는지 등을 신뢰성 있게 판정하는 알고리즘은 존재하지 않는다. 프로그램이 **어떻게 생겼는지**가 아니라 **무엇을 하는지**에 관심을 가지는 순간, 결정 가능성(decidability)은 사라진다.

### 4.4 라이스 정리의 응용

다음은 모두 결정 불가능하다 (비자명한 의미론적 성질):

| 언어 | 성질 |
|----------|----------|
| $\{\langle M \rangle \mid L(M) = \emptyset\}$ | 언어가 공집합이다 |
| $\{\langle M \rangle \mid L(M) = \Sigma^*\}$ | 언어가 전체 언어이다 |
| $\{\langle M \rangle \mid L(M) \text{이 정규 언어이다}\}$ | 언어가 정규적이다 |
| $\{\langle M \rangle \mid L(M) \text{이 문맥 자유이다}\}$ | 언어가 문맥 자유이다 |
| $\{\langle M \rangle \mid L(M) \text{이 유한하다}\}$ | 언어가 유한하다 |
| $\{\langle M \rangle \mid |L(M)| = 42\}$ | 언어가 정확히 42개의 문자열을 포함한다 |
| $\{\langle M \rangle \mid L(M) \text{이 } \varepsilon \text{를 포함한다}\}$ | 언어가 ε를 포함한다 |

**라이스 정리가 적용되지 않는 경우:**
- TM의 성질(언어가 아닌): 예) "$M$이 100개 미만의 상태를 가지는가?" — 이는 결정 가능하다
- 자명한 성질: 예) "$M$이 어떤 RE 언어를 수락하는가?" — 항상 그렇다, 자명하게 결정 가능하다

## 5. 포스트 대응 문제(The Post Correspondence Problem)

### 5.1 정의

**포스트 대응 문제(Post Correspondence Problem, PCP)**의 인스턴스(instance)는 **도미노(dominos)**의 모음이다: $(t_1, b_1), (t_2, b_2), \ldots, (t_k, b_k)$ 쌍들로 구성되며 $t_i, b_i \in \Sigma^+$이다.

**매칭(match)**은 인덱스 수열 $i_1, i_2, \ldots, i_n$ (반복 허용)으로서 다음을 만족한다:

$$t_{i_1} t_{i_2} \cdots t_{i_n} = b_{i_1} b_{i_2} \cdots b_{i_n}$$

**결정 문제(decision problem)**: 주어진 인스턴스에 매칭이 존재하는가?

### 5.2 예제

도미노: $(a, ab)$, $(b, a)$, $(ab, b)$

매칭: 인덱스 1, 2, 1, 3:
- 상단: $a \cdot b \cdot a \cdot ab = abaab$
- 하단: $ab \cdot a \cdot ab \cdot b = abaab$ ✓

### 5.3 결정 불가능성

**정리.** PCP는 결정 불가능하다.

*증명 개요:* $A_{TM}$으로부터 환원한다. TM의 계산을 PCP 인스턴스로 인코딩하는데, TM이 수락하는 경우에만 매칭이 존재하도록 한다. 도미노들은 전이 함수(transition function)와 초기/수락 구성(configuration)을 인코딩한다.

### 5.4 응용

PCP의 결정 불가능성은 다음의 결정 불가능성을 증명하는 데 사용된다:
- CFG의 모호성(Ambiguity of CFGs)
- CFG의 동치성(Equivalence of CFGs)
- 두 문맥 자유 언어의 교집합이 공집합인지 여부
- 형식 검증(formal verification)의 다양한 문제들

## 6. 튜링 환원(Turing Reductions)

### 6.1 정의

언어 $A$가 $B$로 **튜링 환원(Turing reducible)**된다는 것은, $A \leq_T B$로 표기하며, $B$를 오라클(oracle)로 사용하여 $A$를 결정하는 오라클 TM $M^B$가 존재한다는 의미이다.

**오라클 TM(oracle TM)** $M^B$는 한 단계에서 "$w \in B$인가?"를 질의하고 정확한 답을 얻을 수 있다.

### 6.2 비교

| | 사상 환원 ($\leq_m$) | 튜링 환원 ($\leq_T$) |
|---|---|---|
| 메커니즘 | 계산 가능 함수 | 오라클 질의 |
| $B$에 대한 질의 | 정확히 한 번, 마지막에 | 임의의 횟수, 어디서든 |
| 강도 | 더 강하다 (더 제한적) | 더 약하다 (더 일반적) |
| $A \leq B$이면 | $A \leq_m B \implies A \leq_T B$ | 역은 성립하지 않음 |

### 6.3 예제

$A_{TM} \leq_T HALT_{TM}$: 정지 오라클이 있으면 수락 여부를 결정할 수 있다 — 먼저 $M$이 $w$에서 정지하는지 묻고, 정지한다면 시뮬레이션한다.

$\overline{A_{TM}} \leq_T A_{TM}$: $A_{TM}$ 오라클이 있으면 그 답을 부정하면 된다.

참고: $\overline{A_{TM}} \leq_m A_{TM}$은 성립하지 **않는다** (사상 환원은 인식 가능성을 보존한다).

## 연습 문제

### 연습 1: 사상 환원(Mapping Reductions)

다음을 증명하기 위해 사상 환원을 구성하라:

1. $A_{TM} \leq_m HALT_{TM}$ ($M'$의 명시적 구성을 제시하라)
2. $A_{TM} \leq_m \{\langle M \rangle \mid L(M) \text{이 무한하다}\}$

### 연습 2: 라이스 정리(Rice's Theorem)

각 언어에 대해 라이스 정리가 적용되는지 판단하고 결정 가능성을 결론지어라:

1. $\{\langle M \rangle \mid M \text{이 최대 5개의 상태를 가진다}\}$
2. $\{\langle M \rangle \mid L(M) \text{이 짝수 길이 문자열만 포함한다}\}$
3. $\{\langle M \rangle \mid M \text{이 100 단계 이내에 모든 입력에서 정지한다}\}$

### 연습 3: PCP

다음 PCP 인스턴스에 대한 매칭을 찾거나, 매칭이 존재하지 않음을 논증하라:

도미노: $(ab, a)$, $(b, ab)$, $(a, b)$

---

*이전: [레슨 10 — 결정 가능성](10_Decidability.md) | 다음: [레슨 12 — 계산 복잡도](12_Computational_Complexity.md)*
