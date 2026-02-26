# 레슨 6: 문맥 자유 문법(Context-Free Grammars)

**이전**: [정규 언어의 성질](./05_Properties_of_Regular_Languages.md) | **다음**: [푸시다운 오토마타](./07_Pushdown_Automata.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 문맥 자유 문법(context-free grammar)을 형식적으로 정의한다
2. 주어진 언어에 대한 문법을 구성한다
3. 유도(derivation), 파스 트리(parse tree), 모호성(ambiguity)을 분석한다
4. 문법을 촘스키 정규형(Chomsky Normal Form)과 그라이바흐 정규형(Greibach Normal Form)으로 변환한다
5. CYK 파싱 알고리즘을 적용한다

---

문맥 자유 문법은 프로그래밍 언어 설계의 근간이다. 컴파일러가 소스 코드를 파싱할 때마다 CFG(또는 그 변형)를 사용하여 구문이 유효한지 판단하고, 코드 생성을 이끄는 파스 트리(parse tree)를 구축한다. CFG는 괄호 매칭(matching parentheses)이나 중첩 함수 호출처럼 정규 언어가 표현할 수 없는 중첩·재귀 구조를 표현할 수 있다.

## 1. 형식적 정의

**문맥 자유 문법(context-free grammar, CFG)**은 4-튜플 $G = (V, \Sigma, R, S)$로 정의된다:

| 구성 요소 | 설명 |
|-----------|------|
| $V$ | **변수(variable)**(비단말 기호) 의 유한 집합 |
| $\Sigma$ | **단말 기호(terminal)**의 유한 집합 ($V$와 서로소) |
| $R$ | **생성 규칙(production rule)**의 유한 집합. 각 규칙의 형태는 $A \to \alpha$이며, $A \in V$, $\alpha \in (V \cup \Sigma)^*$ |
| $S \in V$ | **시작 변수(start variable)** |

핵심 제약 조건: 각 규칙의 좌변은 반드시 **단일** 변수여야 한다. 이것이 "문맥 자유"라는 이름의 의미이다 — 변수는 주변 문맥에 관계없이 대체될 수 있다.

## 2. 유도(Derivations)

**유도(derivation)**는 규칙 적용의 순서로서, 시작 변수를 단말 기호열로 변환하는 과정이다.

**표기법:**
- $\alpha \Rightarrow \beta$: 한 단계 유도 (규칙 하나로 $\alpha$에서 $\beta$ 도출)
- $\alpha \Rightarrow^* \beta$: 0회 이상의 유도 단계

**문법의 언어**: $L(G) = \{w \in \Sigma^* \mid S \Rightarrow^* w\}$

### 2.1 예제

$\{0^n 1^n \mid n \geq 0\}$에 대한 문법 $G_1$:

$$S \to 0S1 \mid \varepsilon$$

$000111$의 유도:
$$S \Rightarrow 0S1 \Rightarrow 00S11 \Rightarrow 000S111 \Rightarrow 000111$$

### 2.2 최좌 유도와 최우 유도(Leftmost and Rightmost Derivations)

- **최좌 유도(leftmost derivation)**: 항상 가장 왼쪽 변수를 먼저 전개
- **최우 유도(rightmost derivation)**: 항상 가장 오른쪽 변수를 먼저 전개

**예제:** 문법 $S \to AB$, $A \to aA \mid a$, $B \to bB \mid b$

$aabb$의 최좌 유도:
$$S \Rightarrow AB \Rightarrow aAB \Rightarrow aaB \Rightarrow aabB \Rightarrow aabb$$

$aabb$의 최우 유도:
$$S \Rightarrow AB \Rightarrow AbB \Rightarrow Abb \Rightarrow aAbb \Rightarrow aabb$$

## 3. 파스 트리(Parse Trees)

**파스 트리(parse tree)**는 유도를 그래프로 표현한 것이다:
- 루트는 시작 변수
- 내부 노드는 변수
- 잎 노드는 단말 기호 또는 $\varepsilon$
- 내부 노드 $A$의 자식이 $X_1, \ldots, X_k$일 때, 규칙 $A \to X_1 \cdots X_k$가 존재한다

파스 트리의 **수확(yield)**은 잎 노드를 왼쪽에서 오른쪽으로 읽어서 얻은 문자열이다.

하나의 문자열이 여러 개의 최좌 유도를 가질 수 있으며, 이는 각기 다른 파스 트리에 대응한다.

## 4. 모호성(Ambiguity)

문법 $G$가 **모호(ambiguous)**하다는 것은 $L(G)$의 어떤 문자열 $w$에 대해 두 개 이상의 서로 다른 파스 트리가 존재하는 것이다 (동등하게, 두 개 이상의 최좌 유도가 존재하는 것).

### 4.1 예제: 모호한 문법

$$E \to E + E \mid E \times E \mid (E) \mid a$$

문자열 $a + a \times a$는 두 가지 파스 트리를 가진다:
1. $(a + a) \times a$ — 덧셈 먼저
2. $a + (a \times a)$ — 곱셈 먼저

### 4.2 모호성 해소

문법 구조를 통해 우선순위(precedence)와 결합 법칙(associativity)을 도입한다:

$$E \to E + T \mid T$$
$$T \to T \times F \mid F$$
$$F \to (E) \mid a$$

이 문법은 **비모호(unambiguous)**하며, 표준 우선순위에 따라 $\times$가 $+$보다 먼저 처리된다.

### 4.3 본질적 모호성(Inherent Ambiguity)

**문맥 자유 언어(context-free language)**가 **본질적으로 모호(inherently ambiguous)**하다는 것은 해당 언어를 생성하는 모든 문법이 모호함을 의미한다.

**예제:** $L = \{a^i b^j c^k \mid i = j \text{ or } j = k\}$는 본질적으로 모호하다.

모든 CFL이 본질적으로 모호한 것은 아니다 — 실용적인 언어의 대부분은 비모호한 문법을 가진다.

## 5. 문법 변환(Grammar Transformations)

### 5.1 $\varepsilon$-생성 규칙 제거

$A \to \varepsilon$ 형태의 규칙을 **$\varepsilon$-생성 규칙($\varepsilon$-production)**이라 한다. 이를 제거하는 방법:

1. 모든 **소거 가능(nullable)** 변수 (즉, $\varepsilon$으로 유도될 수 있는 변수)를 찾는다
2. 각 규칙 $A \to X_1 X_2 \cdots X_k$에 대해, 소거 가능한 변수가 있는/없는 모든 조합의 규칙을 추가한다
3. 모든 $\varepsilon$-생성 규칙을 제거한다 (단, $\varepsilon \in L$인 경우 $S \to \varepsilon$는 유지 가능; 이때 $S$는 어떤 규칙의 우변에도 나타나지 않아야 한다)

### 5.2 단위 생성 규칙 제거(Eliminating Unit Productions)

**단위 생성 규칙(unit production)**은 $A \to B$ ($B \in V$) 형태의 규칙이다.

1. 단위 쌍(unit pair) 관계를 계산한다: 단위 생성 규칙만으로 $A \Rightarrow^* B$가 성립하는 모든 쌍 $(A, B)$
2. 각 단위 쌍 $(A, B)$와 비단위 생성 규칙 $B \to \alpha$에 대해, $A \to \alpha$를 추가한다
3. 모든 단위 생성 규칙을 제거한다

### 5.3 쓸모없는 기호 제거(Eliminating Useless Symbols)

기호가 **유용(useful)**하다는 것은 $w \in \Sigma^*$에 대한 어떤 유도 $S \Rightarrow^* w$에 해당 기호가 등장함을 의미한다.

1. **생성 불가능(non-generating)** 변수 (어떤 단말 기호열도 유도할 수 없는 변수)를 제거한다
2. **도달 불가능(unreachable)** 기호 ($S$에서 도달할 수 없는 기호)를 제거한다

순서가 중요하다: 먼저 생성 가능 여부를 확인한 후, 도달 가능성을 확인한다.

## 6. 촘스키 정규형(Chomsky Normal Form, CNF)

CFG가 **촘스키 정규형(Chomsky Normal Form)**에 있다는 것은 모든 규칙이 다음 형태 중 하나임을 의미한다:
- $A \to BC$ (두 변수)
- $A \to a$ (단일 단말 기호)
- $S \to \varepsilon$ ($\varepsilon \in L$인 경우에만, $S$는 어떤 우변에도 나타나지 않음)

**정리.** 모든 CFL은 CNF 형태의 문법을 가진다.

**변환 알고리즘:**
1. $\varepsilon$-생성 규칙 제거 ($S \to \varepsilon$ 제외)
2. 단위 생성 규칙 제거
3. 긴 규칙에서 단말 기호 교체: $A \to aBC$에 대해 $X_a \to a$를 추가하고 $A \to X_a BC$로 변환
4. 긴 규칙 분리: $A \to BCD$에 대해 $A \to BX$, $X \to CD$를 추가

### 6.1 CNF 예제

원본: $S \to aSb \mid ab$

3단계: 단말 기호 교체
- $X_a \to a$, $X_b \to b$
- $S \to X_a S X_b \mid X_a X_b$

4단계: 긴 규칙 $S \to X_a S X_b$ 분리
- $S \to X_a Y$, $Y \to S X_b$
- $S \to X_a X_b$

CNF 결과:
$$S \to X_a Y \mid X_a X_b, \quad Y \to S X_b, \quad X_a \to a, \quad X_b \to b$$

## 7. 그라이바흐 정규형(Greibach Normal Form, GNF)

CFG가 **그라이바흐 정규형(Greibach Normal Form)**에 있다는 것은 모든 규칙이 다음 형태임을 의미한다:
- $A \to a B_1 B_2 \cdots B_k$ (단말 기호 뒤에 0개 이상의 변수, $k \geq 0$)
- $S \to \varepsilon$ (필요한 경우)

**정리.** 모든 CFL은 GNF 형태의 문법을 가진다.

GNF가 유용한 이유:
- 각 유도 단계에서 정확히 하나의 입력 기호를 소비한다
- 유도의 길이가 문자열의 길이와 같다 (본문에 $\varepsilon$-생성 규칙 없음)
- $\varepsilon$-전이 없는 PDA에 직접 대응된다 ($\varepsilon$ 수리 제외)

## 8. CYK 알고리즘(The CYK Algorithm)

**코크-영거-카사미(Cocke-Younger-Kasami, CYK)** 알고리즘은 CNF 형태의 문법 $G$에 대해 $w \in L(G)$를 판별한다. 동적 프로그래밍(dynamic programming)을 사용한다.

**입력:** CNF 형태의 문법 $G$, 문자열 $w = a_1 a_2 \cdots a_n$

**테이블:** $T[i][j]$ = $a_i a_{i+1} \cdots a_j$를 유도할 수 있는 변수들의 집합

**알고리즘:**
```
// 기저 사례: 길이 1인 부분 문자열
for i = 1 to n:
    T[i][i] = {A | A → a_i is a rule}

// 귀납 사례: 길이 2, 3, ..., n인 부분 문자열
for length = 2 to n:
    for i = 1 to n - length + 1:
        j = i + length - 1
        T[i][j] = ∅
        for k = i to j - 1:
            for each rule A → BC:
                if B ∈ T[i][k] and C ∈ T[k+1][j]:
                    T[i][j] = T[i][j] ∪ {A}

// 시작 변수가 전체 문자열을 유도하면 수리
accept if S ∈ T[1][n]
```

**시간 복잡도(time complexity):** $O(n^3 \cdot |G|)$

**공간 복잡도(space complexity):** $O(n^2)$

### 8.1 CYK 예제

CNF 형태의 문법: $S \to AB$, $A \to a$, $B \to b$, $S \to a$ (언어 $\{a, ab\}$에 대한 단순화 예제).

더 복잡한 문법의 경우, CYK 테이블은 체계적으로 상향식으로 채워진다 — 예제 코드를 참고하라.

## 연습 문제

### 연습 1: 문법 설계

다음 언어에 대한 CFG를 작성하라:

1. $L = \{a^i b^j \mid i \geq j \geq 0\}$
2. $L = \{w \in \{a,b\}^* \mid |w|_a = |w|_b\}$
3. $L = \{a^i b^j c^k \mid i + k = j\}$

### 연습 2: CNF 변환

다음 문법을 촘스키 정규형(Chomsky Normal Form)으로 변환하라:

$$S \to aAb \mid B$$
$$A \to aA \mid a$$
$$B \to bB \mid \varepsilon$$

각 변환 단계를 보여라.

### 연습 3: CYK 파싱

다음 문법이 생성하는 언어에 $w = aabb$가 포함되는지 CYK 알고리즘을 적용하여 확인하라:

$$S \to AB \mid BC, \quad A \to BA \mid a, \quad B \to CC \mid b, \quad C \to AB \mid a$$

완전한 CYK 테이블을 채워라.

---

*이전: [레슨 5 — 정규 언어의 성질](05_Properties_of_Regular_Languages.md) | 다음: [레슨 7 — 푸시다운 오토마타](07_Pushdown_Automata.md)*
