# 레슨 5: 정규 언어의 성질(Properties of Regular Languages)

**이전**: [정규 표현식](./04_Regular_Expressions.md) | **다음**: [문맥 자유 문법](./06_Context_Free_Grammars.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 정규 언어에 대한 펌핑 보조정리(pumping lemma)를 서술하고 적용한다
2. 특정 언어가 정규 언어가 아님을 증명한다
3. 정규 언어의 폐쇄 성질(closure properties)을 열거한다
4. 마이힐-너로드 정리(Myhill-Nerode theorem)를 서술하고 적용한다
5. 정규 언어에 대한 결정 가능 문제(decidable problems)를 이해한다

---

정규 언어가 *할 수 있는* 것을 아는 것은 절반에 불과하다. 정규 언어의 한계 — 즉 *표현할 수 없는* 것 — 를 이해하는 것도 똑같이 중요하다. 펌핑 보조정리와 마이힐-너로드 정리(Myhill-Nerode theorem)는 이 경계를 엄밀하게 그을 수 있는 도구를 제공하며, 폐쇄 성질(closure properties)은 기존 정규 언어로부터 새로운 정규 언어를 구성하게 해 준다. 이 결과들은 언어 인식 문제에 적합한 형식 체계를 선택하는 데 필수적이다.

## 1. 펌핑 보조정리(The Pumping Lemma)

펌핑 보조정리는 정규성(regularity)에 대한 필요 조건을 제공한다. 이는 언어가 정규 언어가 **아님**을 증명하는 주요 도구이다.

**정리 (정규 언어에 대한 펌핑 보조정리(Pumping Lemma for Regular Languages)).** $L$이 정규 언어이면, 펌핑 길이(pumping length) $p \geq 1$인 상수가 존재하여, $|w| \geq p$인 모든 문자열 $w \in L$에 대해 $w = xyz$로 쓸 수 있으며 다음이 성립한다:

1. $|y| \geq 1$ ("펌프"가 공이 아님)
2. $|xy| \leq p$ (펌프는 처음 $p$개의 기호 내에 존재함)
3. 모든 $i \geq 0$에 대해, $xy^iz \in L$ ($y$를 펌핑해도 소속성이 유지됨)

**직관:** $p$개의 상태를 가진 DFA에서, 길이 $\geq p$인 문자열은 어떤 상태를 반드시 재방문한다(비둘기집 원리(pigeonhole principle)). 같은 상태를 두 번 방문하는 구간에 해당하는 부분 문자열은 "펌핑"될 수 있다 — 즉, 임의의 횟수만큼 반복할 수 있다.

### 1.1 펌핑 보조정리 증명

$p = |Q|$개의 상태를 가진 $L$에 대한 DFA $M$을 생각하자. $n \geq p$인 $w = a_1 a_2 \cdots a_n$을 고려한다.

계산은 상태 $q_0, q_1, \ldots, q_n$을 거치며, 여기서 $q_i = \hat{\delta}(q_0, a_1 \cdots a_i)$이다.

처음 $p + 1$개의 상태 $q_0, q_1, \ldots, q_p$ 중에서, 비둘기집 원리에 의해 어떤 상태가 반복되어야 한다: $0 \leq i < j \leq p$인 $q_i = q_j$가 존재한다.

다음과 같이 설정한다:
- $x = a_1 \cdots a_i$ (반복 상태 이전)
- $y = a_{i+1} \cdots a_j$ (루프)
- $z = a_{j+1} \cdots a_n$ (루프 이후)

그러면 $|y| = j - i \geq 1$, $|xy| = j \leq p$이며, $y$를 임의의 횟수만큼 펌핑해도 루프가 같은 상태로 돌아오므로 $L$에 속한다. $\square$

## 2. 펌핑 보조정리 활용

펌핑 보조정리는 **귀류법(proof by contradiction)**으로 활용된다: $L$이 정규 언어라 가정하고 모순을 도출한다.

### 2.1 예시: $L = \{0^n 1^n \mid n \geq 0\}$은 정규 언어가 아니다

*증명.* $L$이 펌핑 길이 $p$를 가진 정규 언어라 가정한다. $|w| = 2p \geq p$인 $w = 0^p 1^p \in L$을 고려한다.

펌핑 보조정리에 의해, $|y| \geq 1$, $|xy| \leq p$를 만족하는 $w = xyz$가 존재한다.

$|xy| \leq p$이므로, $x$와 $y$는 모두 0으로만 구성된다. $k \geq 1$로 $y = 0^k$라 쓰자.

펌핑 다운: $xy^0z = xz = 0^{p-k}1^p$. $k \geq 1$이므로 0의 개수가 1의 개수보다 적어 $xz \notin L$이다. 모순. $\square$

### 2.2 예시: $L = \{ww \mid w \in \{0,1\}^*\}$은 정규 언어가 아니다

*증명.* $L$이 펌핑 길이 $p$를 가진 정규 언어라 가정한다. $w = 0^p 1 0^p 1 \in L$을 고려한다.

펌핑 보조정리에 의해, $|xy| \leq p$를 만족하는 $w = xyz$가 존재하므로 $y = 0^k$($k \geq 1$, 첫 번째 0 블록 내)이다.

그러면 $xy^2z = 0^{p+k} 1 0^p 1$이 된다. 이것이 $L$에 속하려면 어떤 $u$에 대해 $uu$이어야 한다. 그런데 $|xy^2z| = 2p + k + 2$이므로 $|u| = p + k/2 + 1$이 되어야 하는데, $k$가 홀수이면 정수가 아니다. $k$가 짝수인 경우에도 분할이 동일한 두 절반을 만들지 못한다. 모순. $\square$

### 2.3 예시: $L = \{0^{n^2} \mid n \geq 0\}$은 정규 언어가 아니다

*증명.* 펌핑 길이 $p$를 가진 정규 언어라 가정한다. $w = 0^{p^2}$을 선택한다.

$1 \leq |y| \leq p$인 $w = xyz$를 쓰면, $|xy^2z| = p^2 + |y|$이다.

$1 \leq |y| \leq p$이므로: $p^2 < p^2 + |y| \leq p^2 + p < (p+1)^2$.

따라서 $|xy^2z|$는 연속한 두 완전제곱수 사이에 있어 완전제곱수가 아니다. 즉 $xy^2z \notin L$이다. 모순. $\square$

## 3. 폐쇄 성질(Closure Properties)

정규 언어는 많은 연산에 대해 닫혀(closed) 있다:

### 3.1 불리언 연산(Boolean Operations)

| 연산 | 닫혀 있음? | 증명 기법 |
|-----------|---------|-----------------|
| 합집합(Union) $L_1 \cup L_2$ | 예 | NFA 구성 또는 곱 DFA(product DFA) |
| 교집합(Intersection) $L_1 \cap L_2$ | 예 | 곱 DFA(Product DFA) |
| 여집합(Complement) $\overline{L}$ | 예 | DFA에서 수락/거부 상태 교환 |
| 차집합(Difference) $L_1 \setminus L_2$ | 예 | $L_1 \cap \overline{L_2}$ |
| 대칭 차(Symmetric diff.) $L_1 \triangle L_2$ | 예 | $(L_1 \setminus L_2) \cup (L_2 \setminus L_1)$ |

### 3.2 정규 연산(Regular Operations)

| 연산 | 닫혀 있음? | 증명 기법 |
|-----------|---------|-----------------|
| 연결(Concatenation) $L_1 \cdot L_2$ | 예 | NFA 구성 |
| 클리니 스타(Kleene star) $L^*$ | 예 | NFA 구성 |
| 역전(Reversal) $L^R$ | 예 | 모든 전이 역방향, 시작/수락 교환 |

### 3.3 기타 연산(Other Operations)

| 연산 | 닫혀 있음? | 증명 기법 |
|-----------|---------|-----------------|
| 동형사상(Homomorphism) $h(L)$ | 예 | NFA의 각 기호 치환 |
| 역 동형사상(Inverse homomorphism) $h^{-1}(L)$ | 예 | 각 입력 $a$에 대해 $h(a)$로 DFA 시뮬레이션 |
| 몫(Quotient) $L_1 / L_2$ | 예 | 수락 상태 변경 |

### 3.4 증명에서의 폐쇄 성질 활용

폐쇄 성질을 사용하면 펌핑 보조정리 없이도 비정규성을 증명할 수 있다.

**예시:** $L = \{0^n 1^m \mid n \neq m\}$이 정규 언어가 아님을 보여라.

*증명.* $L$이 정규 언어라 가정한다. 정규 언어는 여집합에 닫혀 있으므로 $\overline{L}$도 정규 언어이다. $\overline{L} = \{0^n 1^n \mid n \geq 0\} \cup \{w \mid w \text{가 } 0^*1^* \text{ 꼴이 아님}\}$으로 쓸 수 있다.

$\{w \mid w \text{가 } 0^*1^* \text{ 꼴}\} = 0^*1^*$은 정규 언어이다. 따라서 $\overline{L} \cap 0^*1^* = \{0^n 1^n \mid n \geq 0\}$도 정규 언어이어야 한다. 하지만 이미 이것이 정규 언어가 아님을 증명했다. 모순. $\square$

## 4. 마이힐-너로드 정리(The Myhill-Nerode Theorem)

마이힐-너로드 정리는 정규성에 대한 **완전한 특성화(exact characterization)**를 제공한다(필요 조건만 제공하는 펌핑 보조정리와 달리).

### 4.1 구별 가능성(Distinguishability)

언어 $L \subseteq \Sigma^*$에 대해, 두 문자열 $x, y \in \Sigma^*$가 **$L$-구별 가능(L-distinguishable)**하다는 것은 $xz, yz$ 중 정확히 하나만 $L$에 속하는 문자열 $z \in \Sigma^*$가 존재함을 의미한다.

그러한 $z$가 없으면 $x$와 $y$는 **$L$-구별 불가능(L-indistinguishable)**하며, $x \equiv_L y$로 표기한다.

### 4.2 동치 관계 $\equiv_L$

$\equiv_L$은 $\Sigma^*$ 위에서 동치 관계(reflexive, symmetric, transitive)이다. 이는 $\Sigma^*$를 동치류(equivalence classes)로 분할한다.

### 4.3 정리

**정리 (마이힐-너로드(Myhill-Nerode)).** 다음은 서로 동치이다:
1. $L$은 정규 언어이다
2. $L$은 유한 지표(finite index)를 가진 우불변(right-invariant) 동치 관계의 일부 동치류들의 합집합이다
3. $\equiv_L$의 지표(index)가 유한하다(동치류의 수가 유한하다)

더 나아가, $\equiv_L$의 동치류 수는 $L$에 대한 **최소 DFA(minimum DFA)**의 상태 수와 같다.

### 4.4 예시: 마이힐-너로드로 보는 $\{0^n 1^n\}$

$L = \{0^n 1^n \mid n \geq 0\}$을 고려하자. $i \neq j$인 임의의 경우, 문자열 $0^i$와 $0^j$는 $L$-구별 가능하다: $z = 1^i$를 사용하면 $0^i 1^i \in L$이지만 $0^j 1^i \notin L$이다.

$\{0^i \mid i \geq 0\}$의 모든 원소가 쌍으로 구별 가능하므로, $\equiv_L$은 무한히 많은 동치류를 가진다. 마이힐-너로드 정리에 의해 $L$은 정규 언어가 아니다.

### 4.5 예시: 최소 DFA 크기

$L = \{w \in \{0,1\}^* \mid w \text{가 } 01\text{로 끝남}\}$에 대한 마이힐-너로드 동치류는:
- $[\varepsilon]$: 0이나 01로 끝나지 않는 문자열 ($01$을 뒤에 붙여 구별)
- $[0]$: 0으로 끝나지만 01로 끝나지 않는 문자열 ($1$을 뒤에 붙여 구별)
- $[01]$: 01로 끝나는 문자열

세 개의 동치류 → 최소 DFA의 상태 수는 3.

## 5. 정규 언어에 대한 결정 문제(Decision Problems for Regular Languages)

정규 언어에 대해 다음 문제들은 모두 **결정 가능(decidable)**하다:

| 문제 | 질문 | 결정 가능? | 방법 |
|---------|----------|------------|--------|
| 소속성(Membership) | $w \in L$인가? | 예 | DFA로 $w$ 시뮬레이션 |
| 공집합(Emptiness) | $L = \emptyset$인가? | 예 | 수락 상태로의 도달 가능성 확인 |
| 유한성(Finiteness) | $L$이 유한한가? | 예 | 수락 상태로 가는 경로의 사이클 확인 |
| 동치(Equivalence) | $L_1 = L_2$인가? | 예 | 두 DFA 최소화 후 동형사상 확인 |
| 포함(Containment) | $L_1 \subseteq L_2$인가? | 예 | $L_1 \cap \overline{L_2} = \emptyset$ 확인 |
| 보편성(Universality) | $L = \Sigma^*$인가? | 예 | $\overline{L} = \emptyset$ 확인 |

## 연습 문제

### 연습 1: 펌핑 보조정리 증명

다음 언어들이 정규 언어가 아님을 증명하라:

1. $L = \{a^n b^{2n} \mid n \geq 0\}$
2. $L = \{w \in \{0,1\}^* \mid w \text{에서 0과 1의 개수가 같음}\}$
3. $L = \{a^p \mid p \text{는 소수}\}$

### 연습 2: 폐쇄 성질

펌핑 보조정리를 사용하지 않고 폐쇄 성질을 사용하여 증명하라:

1. $L \cap a^*b^* = \{a^nb^n\}$이라는 사실을 이용하여 $L = \{w \in \{a,b\}^* \mid |w|_a = |w|_b\}$가 정규 언어가 아님을 보여라
2. $L$이 정규 언어이면, $\text{HALF}(L) = \{x \mid \exists y, |x| = |y|, xy \in L\}$도 정규 언어임을 보여라

### 연습 3: 마이힐-너로드

$L = \{w \in \{a,b\}^* \mid |w|_a \text{가 짝수}\}$에 대한 마이힐-너로드 동치류를 구하라. 최소 DFA의 상태 수는 얼마인가?

---

*이전: [레슨 4 — 정규 표현식(Regular Expressions)](04_Regular_Expressions.md) | 다음: [레슨 6 — 문맥 자유 문법(Context-Free Grammars)](06_Context_Free_Grammars.md)*
