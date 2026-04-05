# 레슨 1: 형식 언어 입문(Introduction to Formal Languages)

**다음**: [결정적 유한 오토마타](./02_Deterministic_Finite_Automata.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 알파벳(alphabet), 문자열(string), 형식 언어(formal language)를 수학적으로 정의한다
2. 문자열과 언어에 대한 기본 연산을 수행한다
3. 컴퓨터 과학에서 형식 언어의 역할을 이해한다
4. 언어를 명세하는 다양한 방법을 구별한다

## 1. 동기

모든 프로그래밍 언어, 통신 프로토콜, 데이터 형식은 **형식 언어(formal language)** — 어떤 알파벳 위의 문자열 집합을 정밀하게 정의한 것 — 라는 개념 위에 구축됩니다. 형식 언어를 이해하면 다음과 같은 근본적 질문에 답할 수 있습니다:

- 무엇을 계산할 수 있는가? (계산 가능성(Computability))
- 얼마나 효율적으로 계산할 수 있는가? (복잡도(Complexity))
- 유효한 프로그램을 어떻게 인식하는가? (파싱(Parsing))
- 기계적 추론의 한계는 무엇인가? (결정 불가능성(Undecidability))

이 토픽은 이러한 질문들을 위한 수학적 프레임워크를 제공합니다.

## 2. 알파벳과 문자열

### 2.1 알파벳(Alphabet)

**알파벳(alphabet)** $\Sigma$는 기호들의 유한하고 비어 있지 않은 집합입니다.

**예시:**
- 이진 알파벳: $\Sigma = \{0, 1\}$
- 라틴 알파벳: $\Sigma = \{a, b, c, \ldots, z\}$
- ASCII: $\Sigma = \{$ 모든 128개 ASCII 문자 $\}$
- DNA: $\Sigma = \{A, C, G, T\}$

### 2.2 문자열(String, Word)

알파벳 $\Sigma$ 위의 **문자열(string)** (또는 **단어(word)**)은 $\Sigma$의 기호들로 이루어진 유한한 수열입니다.

- **빈 문자열(empty string)**은 $\varepsilon$ (때로는 $\lambda$)으로 표기합니다.
- 문자열 $w$의 **길이(length)**는 $|w|$로 표기합니다.
  - $|\varepsilon| = 0$
  - $|abba| = 4$
- $\Sigma^*$는 $\varepsilon$을 포함한 $\Sigma$ 위의 **모든 문자열**의 집합을 나타냅니다.
- $\Sigma^+$는 모든 **비어 있지 않은** 문자열을 나타냅니다: $\Sigma^+ = \Sigma^* \setminus \{\varepsilon\}$.

### 2.3 문자열 연산

**연접(Concatenation):** $x = a_1 a_2 \cdots a_m$이고 $y = b_1 b_2 \cdots b_n$이면, $xy = a_1 a_2 \cdots a_m b_1 b_2 \cdots b_n$입니다.

성질:
- $|xy| = |x| + |y|$
- $x \varepsilon = \varepsilon x = x$ (항등원(identity))
- $(xy)z = x(yz)$ (결합법칙(associativity))
- 일반적으로 $xy \neq yx$ (교환법칙 불성립)

**거듭제곱(Power):** $w^0 = \varepsilon$, $n \geq 0$에 대해 $w^{n+1} = w^n w$.

**역전(Reversal):** $w^R$은 $w$를 거꾸로 쓴 것입니다. $(abc)^R = cba$, $\varepsilon^R = \varepsilon$.

**부분 문자열, 접두사, 접미사:**
- 어떤 $x, y$에 대해 $w = xvy$이면 $v$는 $w$의 **부분 문자열(substring)**입니다.
- 어떤 $y$에 대해 $w = vy$이면 $v$는 $w$의 **접두사(prefix)**입니다.
- 어떤 $x$에 대해 $w = xv$이면 $v$는 $w$의 **접미사(suffix)**입니다.

## 3. 형식 언어

### 3.1 정의

알파벳 $\Sigma$ 위의 **형식 언어(formal language)** $L$은 $\Sigma^*$의 임의의 부분집합입니다:

$$L \subseteq \Sigma^*$$

언어는 유한할 수도 있고 무한할 수도 있습니다.

**예시** ($\Sigma = \{0, 1\}$ 위에서):
- $L_1 = \{0, 01, 011, 0111, \ldots\} = \{0 1^n \mid n \geq 0\}$
- $L_2 = \{w \in \{0,1\}^* \mid w = w^R\}$ (이진 회문(binary palindromes))
- $L_3 = \emptyset$ (빈 언어 — 문자열을 하나도 포함하지 않음)
- $L_4 = \{\varepsilon\}$ (빈 문자열만 정확히 하나 포함)

참고: $\emptyset \neq \{\varepsilon\}$ — 빈 언어와 빈 문자열만을 포함하는 언어는 다릅니다.

### 3.2 언어 연산

언어는 집합이므로 모든 집합 연산을 지원합니다:

**합집합(Union):** $L_1 \cup L_2 = \{w \mid w \in L_1 \text{ or } w \in L_2\}$

**교집합(Intersection):** $L_1 \cap L_2 = \{w \mid w \in L_1 \text{ and } w \in L_2\}$

**여집합(Complement):** $\overline{L} = \Sigma^* \setminus L = \{w \in \Sigma^* \mid w \notin L\}$

**차집합(Difference):** $L_1 \setminus L_2 = L_1 \cap \overline{L_2}$

또한, 언어에는 연접과 클로저 연산이 있습니다:

**연접(Concatenation):** $L_1 \cdot L_2 = \{xy \mid x \in L_1, y \in L_2\}$

**거듭제곱(Power):** $L^0 = \{\varepsilon\}$, $L^{n+1} = L^n \cdot L$

**클리니 스타(Kleene star, closure):** $L^* = \bigcup_{n=0}^{\infty} L^n = L^0 \cup L^1 \cup L^2 \cup \cdots$

**양의 클로저(Positive closure):** $L^+ = \bigcup_{n=1}^{\infty} L^n = L \cdot L^*$

### 3.3 언어 연산 예시

$L_1 = \{a, ab\}$이고 $L_2 = \{b, ba\}$라 하면:

- $L_1 \cup L_2 = \{a, ab, b, ba\}$
- $L_1 \cdot L_2 = \{ab, aba, abb, abba\}$
- $L_1^* = \{\varepsilon, a, ab, aa, aab, aba, abab, \ldots\}$

## 4. 언어 명세 방법

언어는 여러 방식으로 명세할 수 있습니다:

### 4.1 열거(Enumeration)

모든 문자열을 나열합니다 (유한 언어에만 실용적):

$$L = \{cat, dog, fish\}$$

### 4.2 집합 조건 표기법(Set-Builder Notation)

조건자(predicate)로 기술합니다:

$$L = \{a^n b^n \mid n \geq 0\} = \{\varepsilon, ab, aabb, aaabbb, \ldots\}$$

### 4.3 문법(Grammars)

언어 안의 모든 문자열을 생성하는 생성 규칙(production rules)의 집합입니다. 문법의 종류(정규(regular), 문맥 자유(context-free), 문맥 민감(context-sensitive), 무제한(unrestricted))에 따라 다른 언어 클래스가 정의됩니다.

### 4.4 오토마타(Automata)

언어 안의 문자열만 정확히 수리하는 계산 모델입니다. 오토마타의 종류(DFA, PDA, TM)에 따라 다른 언어 클래스를 인식합니다.

### 4.5 정규 표현식(Regular Expressions)

패턴을 기술하는 대수적 표기법입니다. 표현력 면에서 유한 오토마타와 동치입니다.

## 5. 촘스키 계층(The Chomsky Hierarchy) (미리 보기)

Noam Chomsky는 문법(및 그에 대응하는 언어)을 네 가지 타입으로 분류했습니다:

| 타입 | 문법 | 오토마톤 | 언어 클래스 | 예시 |
|------|------|---------|-----------|------|
| 3 | 정규(Regular) | 유한 오토마톤(Finite automaton) | 정규(Regular) | $a^*b^*$ |
| 2 | 문맥 자유(Context-free) | 푸시다운 오토마톤(Pushdown automaton) | 문맥 자유(Context-free) | $a^n b^n$ |
| 1 | 문맥 민감(Context-sensitive) | 선형 경계 오토마톤(Linear bounded automaton) | 문맥 민감(Context-sensitive) | $a^n b^n c^n$ |
| 0 | 무제한(Unrestricted) | 튜링 기계(Turing machine) | 재귀적 열거 가능(Recursively enumerable) | 정지 문제(Halting problem) |

각 타입은 다음 타입에 진부분집합으로 포함됩니다: 정규(Regular) $\subsetneq$ 문맥 자유(Context-free) $\subsetneq$ 문맥 민감(Context-sensitive) $\subsetneq$ 재귀적 열거 가능(Recursively enumerable).

이 토픽 전반에 걸쳐 각 레벨을 자세히 학습합니다.

## 6. 언어에 대한 결정 문제(Decision Problems)

언어 $L$이 주어졌을 때, 근본적인 질문들은 다음과 같습니다:

- **원소 판별(Membership):** $w \in L$인가? (언어의 문자열을 인식할 수 있는가?)
- **공집합 판별(Emptiness):** $L = \emptyset$인가?
- **유한성 판별(Finiteness):** $L$은 유한한가?
- **동치 판별(Equivalence):** $L_1, L_2$가 주어졌을 때, $L_1 = L_2$인가?
- **포함 판별(Containment):** $L_1 \subseteq L_2$인가?

이 질문들이 **결정 가능(decidable)**한지는 언어 클래스에 따라 다르며, 이는 이 토픽의 핵심 주제입니다.

## 연습 문제

### 연습 1: 문자열 연산

$\Sigma = \{a, b\}$, $x = abba$, $y = bab$라 하자.

1. $xy$, $yx$, $x^R$, $y^R$을 계산하라.
2. $x$의 모든 접두사를 나열하라.
3. $y$의 모든 부분 문자열을 나열하라 (중복 없이).

### 연습 2: 언어 연산

$\Sigma = \{a, b\}$ 위에서 $L_1 = \{a, bb\}$이고 $L_2 = \{b, ab\}$라 하자.

1. $L_1 \cdot L_2$를 계산하라.
2. $L_1^2 = L_1 \cdot L_1$을 계산하라.
3. $L_1^*$에서 길이 $\leq 3$인 문자열을 나열하라.

### 연습 3: 언어 분류

각 언어를 분류하고 비형식적으로 근거를 제시하라 (나중에 형식적으로 증명할 것임):

1. $L = \{w \in \{0,1\}^* \mid w \text{ has an even number of 0s}\}$
2. $L = \{0^n 1^n \mid n \geq 0\}$
3. $L = \{0^n 1^n 2^n \mid n \geq 0\}$
4. 모든 유효한 C 프로그램의 집합

---

*다음: [레슨 2 — 결정적 유한 오토마타(Deterministic Finite Automata)](02_Deterministic_Finite_Automata.md)*
