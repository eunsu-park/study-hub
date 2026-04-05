# 레슨 4: 정규 표현식(Regular Expressions)

**이전**: [비결정적 유한 오토마타](./03_Nondeterministic_Finite_Automata.md) | **다음**: [정규 언어의 성질](./05_Properties_of_Regular_Languages.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 정규 표현식(regular expression)을 구문과 의미론을 갖춘 형식적 정의로 설명한다
2. 정규 표현식과 유한 오토마타(finite automata) 사이를 상호 변환한다
3. 대수적 법칙(algebraic laws)을 적용하여 정규 표현식 동치를 판별한다
4. 이론적 정규 표현식과 실용적인 정규식 엔진의 차이를 구별한다

---

터미널에서 `grep 'error.*timeout' server.log`를 입력해 본 적이 있거나, 셸(shell)에서 `*.txt`로 파일을 매칭해 본 적이 있다면, 이미 정규 표현식(regular expression)의 실용적 후손을 사용한 것이다. 여기서 다루는 이론적 정규 표현식은 편의 기능을 제거하고 정확한 질문을 던진다: **선택, 연결, 반복이라는 세 가지 연산만으로 무엇을 기술할 수 있는가?** 그 답은 정확히 정규 언어(regular language), 즉 유한 오토마타(finite automata)가 인식하는 것과 동일한 언어 클래스이다.

정규 표현식은 유한 오토마타의 대수적 대응물이다. 오토마타가 언어를 절차적으로(기계로서) 기술하는 반면, 정규 표현식은 선언적으로(패턴으로서) 기술한다. 모든 프로그래머는 `grep`, 텍스트 편집기, 입력 유효성 검사기 같은 도구에서 정규식을 접하지만, 이론적 토대는 정규식이 표현할 수 있는 패턴과 없는 패턴을 정확히 설명해 준다.

## 1. 형식적 정의

알파벳 $\Sigma$ 위에서의 **정규 표현식(regular expression)**은 다음과 같이 재귀적으로 정의된다.

**기저 사례(Base cases):**
- $\emptyset$은 공 언어(empty language) $\emptyset$을 나타내는 정규 표현식이다
- $\varepsilon$은 $\{\varepsilon\}$을 나타내는 정규 표현식이다
- 각 $a \in \Sigma$에 대해, 기호 $a$는 $\{a\}$를 나타내는 정규 표현식이다

**재귀 사례(Recursive cases):** $R$과 $S$가 정규 표현식이면:
- $(R \cup S)$ (합집합(union))은 $L(R) \cup L(S)$를 나타낸다
- $(RS)$ (연결(concatenation))은 $L(R) \cdot L(S)$를 나타낸다
- $(R^*)$ (클리니 스타(Kleene star))은 $L(R)^*$를 나타낸다

**클리니 스타(Kleene star)의 직관:** `*`는 "0번 이상 반복"을 의미한다. 셸 글로브(shell glob)의 `*` 와일드카드와 비슷하게 작동한다: `*.txt`는 임의의 파일 이름(빈 접두어 포함) 뒤에 `.txt`가 오는 것에 매칭된다. 마찬가지로 $a^*$는 빈 문자열, `a`, `aa`, `aaa` 등 -- 반복이 전혀 없는 경우를 포함하여 임의의 횟수의 반복에 매칭된다.

**우선순위(Precedence)** (높음에서 낮음 순): 스타 > 연결 > 합집합.

따라서 $ab^* \cup c$는 $(a(b^*)) \cup c$를 의미하며, $a(b^*\cup c)$나 $(ab)^* \cup c$가 아니다.

## 2. 예시

$\Sigma = \{0, 1\}$ 위에서:

| 정규 표현식 | 언어 |
|-------|---------|
| $0^*$ | $\{\varepsilon, 0, 00, 000, \ldots\}$ |
| $(0 \cup 1)^*$ | $\Sigma^*$ (모든 문자열) |
| $0(0 \cup 1)^*$ | 0으로 시작하는 문자열 |
| $(0 \cup 1)^* 1$ | 1로 끝나는 문자열 |
| $(0 \cup 1)^* 01 (0 \cup 1)^*$ | 부분 문자열로 01을 포함하는 문자열 |
| $1^* (01^+)^*$ | 모든 0 뒤에 적어도 하나의 1이 오는 문자열 |
| $(01 \cup 10)^*$ | $\{\varepsilon, 01, 10, 0101, 0110, 1001, 1010, \ldots\}$ |

## 3. 유한 오토마타와의 동치

**정리 (클리니 정리(Kleene's Theorem)).** 언어가 정규 언어(regular language)인 것과 정규 표현식으로 기술될 수 있는 것은 동치이다.

이를 위해 두 방향의 증명이 필요하다:
1. 정규 표현식 → NFA (톰슨 구성법(Thompson's construction)을 통한 구성적 변환)
2. DFA → 정규 표현식 (상태 제거법(state elimination)을 통한 구성적 변환)

### 3.1 정규 표현식에서 NFA로: 톰슨 구성법(Thompson's Construction)

각 정규 표현식 구성 요소에 대해, 시작 상태 하나와 수락 상태 하나를 가진 소형 NFA 단편(fragment)을 구성한다:

**기저 사례(Base cases):**

- $\varepsilon$: `→(start)--ε-->(accept)`
- $a$: `→(start)--a-->(accept)`

**합집합(Union)** $R \cup S$:
- $N_R$과 $N_S$의 시작 상태로 $\varepsilon$-전이(ε-transition)를 가지는 새 시작 상태 추가
- $N_R$과 $N_S$의 수락 상태들에서 새 수락 상태로의 $\varepsilon$-전이 추가

**연결(Concatenation)** $RS$:
- $N_R$의 수락 상태를 $\varepsilon$으로 $N_S$의 시작 상태에 연결
- $N_R$의 시작 상태가 새 시작 상태; $N_S$의 수락 상태가 새 수락 상태

**스타(Star)** $R^*$:
- $N_R$의 시작 상태로 $\varepsilon$을 가지는 새 시작 상태(동시에 수락 상태) 추가
- $N_R$의 수락 상태에서 $N_R$의 시작 상태로의 $\varepsilon$ 역방향 전이 추가
- $N_R$의 수락 상태에서 새 수락 상태로의 $\varepsilon$ 전이도 추가

각 단계에서 최대 2개의 상태가 추가되므로, 길이 $n$의 정규 표현식에 대해 NFA는 $O(n)$개의 상태를 가진다.

### 3.2 DFA에서 정규 표현식으로: 상태 제거법(State Elimination)

$n$개의 상태를 가진 DFA(또는 NFA)가 주어지면, 전이를 정규 표현식으로 표기하면서 상태를 하나씩 제거할 수 있다:

**알고리즘:**
1. 원래 시작 상태로 $\varepsilon$-전이를 가지는 새 시작 상태 $q_s$ 추가
2. 모든 원래 수락 상태에서 $\varepsilon$-전이를 받는 새 수락 상태 $q_a$ 추가
3. $q_s$와 $q_a$만 남을 때까지 반복:
   - 제거할 상태 $q$ 선택 ($q_s$나 $q_a$는 제외)
   - $p \to q$와 $q \to r$ 전이를 가지는 각 쌍 $(p, r)$에 대해:
     - $q$가 레이블 $S$인 자기 루프(self-loop)를 가지면, $p$에서 $r$로의 새 레이블은 $R_1 S^* R_2 \cup R_3$이다. 여기서 $R_1$은 $p \to q$ 레이블, $R_2$는 $q \to r$ 레이블, $R_3$은 기존 $p \to r$ 레이블이다.
4. 남은 단일 전이 $q_s \to q_a$의 레이블이 정규 표현식이다

### 3.3 예시: 상태 제거법

"1의 개수가 짝수인" 언어에 대한 DFA를 상태 $q_e$ (시작, 수락)와 $q_o$로 구성한다고 하자:

$q_s$와 $q_a$ 추가 후:
- $q_s \xrightarrow{\varepsilon} q_e$
- $q_e \xrightarrow{0} q_e$, $q_e \xrightarrow{1} q_o$
- $q_o \xrightarrow{0} q_o$, $q_o \xrightarrow{1} q_e$
- $q_e \xrightarrow{\varepsilon} q_a$

$q_o$ 제거: $q_o$는 자기 루프 $0$을 가지며 $q_e \xrightarrow{1} q_o \xrightarrow{1} q_e$ 전이를 가진다.

새 전이 $q_e \to q_e$: $1 \cdot 0^* \cdot 1$ ($q_o$로 가서, 0으로 루프하고, 돌아옴).

기존 자기 루프와 합산: $q_e \to q_e$는 이제 $0 \cup 1 0^* 1$.

$q_e$ 제거: 결과는 $(0 \cup 1 0^* 1)^*$.

따라서 정규 표현식은 $(0 \cup 10^*1)^*$ — 1이 쌍으로 나타나는 문자열(사이에 임의의 0 허용).

## 4. 대수적 법칙(Algebraic Laws)

정규 표현식은 다음과 같은 대수적 항등식을 만족한다:

### 4.1 합집합 법칙(Union Laws)
- $R \cup S = S \cup R$ (교환법칙(commutativity))
- $R \cup (S \cup T) = (R \cup S) \cup T$ (결합법칙(associativity))
- $R \cup R = R$ (멱등법칙(idempotence))
- $R \cup \emptyset = R$ (항등원(identity))

### 4.2 연결 법칙(Concatenation Laws)
- $R(ST) = (RS)T$ (결합법칙(associativity))
- $R\varepsilon = \varepsilon R = R$ (항등원(identity))
- $R\emptyset = \emptyset R = \emptyset$ (소멸법칙(annihilation))

### 4.3 분배 법칙(Distribution)
- $R(S \cup T) = RS \cup RT$ (좌분배(left distribution))
- $(R \cup S)T = RT \cup ST$ (우분배(right distribution))

### 4.4 스타 법칙(Star Laws)
- $\emptyset^* = \varepsilon$
- $\varepsilon^* = \varepsilon$
- $R^* = R^* R^* = (R^*)^*$
- $R^* = \varepsilon \cup R R^*$
- $R^* = \varepsilon \cup R^* R$
- $(R \cup S)^* = (R^* S^*)^*$

### 4.5 아던의 보조정리(Arden's Lemma)

$\varepsilon \notin L(A)$인 경우 $X = AX \cup B$이면, $X = A^*B$이다.

이 보조정리는 정규 표현식에 대한 연립방정식을 풀 때 유용하다(예: 상태 방정식을 풀어 DFA를 정규 표현식으로 변환할 때).

## 5. 이론적 정규 표현식 vs. 실용적 정규 표현식

**이론적(Theoretical)** 정규 표현식은 $\cup$, 연결, $*$, $\varepsilon$, $\emptyset$, 그리고 알파벳 기호만 사용한다.

**실용적(Practical)** 정규식 엔진(PCRE, Python `re` 등)은 다음을 추가로 지원한다:

| 기능 | 이론 | 실용 |
|---------|--------|----------|
| 문자 클래스(Character classes) | — | `[a-z]`, `\d`, `\w` |
| 수량자(Quantifiers) | $R^*$만 | `R+`, `R?`, `R{n,m}` |
| 앵커(Anchors) | — | `^`, `$`, `\b` |
| 역참조(Backreferences) | — | `\1`, `(?: ...)` |
| 전방/후방 탐색(Lookahead/behind) | — | `(?=...)`, `(?<=...)` |

**중요:** 역참조(backreferences)는 정규 언어를 **벗어난다**! 패턴 `(a+)\1`은 $\{a^n a^n \mid n \geq 1\}$에 매칭되는데, 이는 정규 언어가 아니다. 실용적인 정규식 엔진은 이론적 정규 표현식보다 강력하지만, 선형 시간 매칭을 보장하지 못한다.

## 6. 정규 표현식 복잡도(Regular Expression Complexity)

| 연산 | 시간 복잡도 |
|-----------|----------------|
| 톰슨 구성법(Thompson's construction) (정규식 → NFA) | $O(n)$ 상태/전이 |
| 부분집합 구성법(Subset construction) (NFA → DFA) | 최악의 경우 $O(2^n)$ |
| DFA 시뮬레이션 | 입력 길이 $m$에 대해 $O(m)$ |
| NFA 시뮬레이션 (온더플라이 부분집합) | 정규식 크기 $n$, 입력 $m$에 대해 $O(nm)$ |

NFA 시뮬레이션 방식은 지수적인 DFA 구성 과정을 피하면서도 다항 시간 매칭을 제공한다. 이는 지수 시간이 걸릴 수 있는 역추적(backtracking) 엔진과 대조된다.

## 연습 문제

### 연습 1: 정규 표현식 설계

$\Sigma = \{0, 1\}$ 위에서 다음 언어에 대한 정규 표현식을 작성하라:

1. 같은 기호로 시작하고 끝나는 길이 2 이상의 문자열
2. 부분 문자열 "11"을 포함하지 않는 문자열
3. 인접한 모든 0 쌍이 1보다 앞에 나타나는 문자열

### 연습 2: 정규 표현식에서 NFA로

톰슨 구성법(Thompson's construction)을 적용하여 $(ab \cup a)^*b$에 대한 NFA를 구성하라. 각 단계에서 중간 NFA 단편(fragment)을 보여라.

### 연습 3: 상태 제거법

상태 제거법(state elimination)을 사용하여 다음 DFA를 정규 표현식으로 변환하라:

```
States: {q0, q1}, Start: q0, Accept: {q1}
δ(q0, a) = q0    δ(q0, b) = q1
δ(q1, a) = q0    δ(q1, b) = q1
```

### 연습 4: 대수적 간소화

대수적 법칙을 사용하여 다음을 간소화하라:

1. $(a^* b^*)^* $
2. $\emptyset^* \cup a$
3. $(a \cup b)^* a (a \cup b)^*$

---

*이전: [레슨 3 — 비결정적 유한 오토마타(Nondeterministic Finite Automata)](03_Nondeterministic_Finite_Automata.md) | 다음: [레슨 5 — 정규 언어의 성질(Properties of Regular Languages)](05_Properties_of_Regular_Languages.md)*
