# 레슨 13: 촘스키 위계(The Chomsky Hierarchy)

**이전**: [계산 복잡도](./12_Computational_Complexity.md) | **다음**: [고급 주제와 응용](./14_Advanced_Topics_and_Applications.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 촘스키 위계(Chomsky hierarchy)의 네 단계를 상세히 설명한다
2. 문맥 의존 문법(context-sensitive grammar)과 선형 제한 오토마톤(linear bounded automaton)을 정의한다
3. 모든 단계에 걸쳐 폐쇄 성질(closure properties), 결정 가능성(decidability), 표현력을 비교한다
4. 언어를 위계의 적절한 단계로 분류한다

---

촘스키 위계는 우리가 학습한 모든 내용의 조직화 틀이다. 형식 언어를 네 개의 중첩된 클래스 — 정규(regular), 문맥 자유(context-free), 문맥 의존(context-sensitive), 재귀 열거 가능(recursively enumerable) — 로 배열하며, 각 클래스에는 대응하는 문법 유형과 오토마톤 모델이 있다. 이 레슨은 개별 결과들을 하나의 통합된 그림으로 종합하여, 각 계산 능력 수준이 표현할 수 있는 것과 없는 것을 보여 준다.

## 1. 개요

**촘스키 계층**은 문법(과 그에 대응하는 언어)을 네 가지 유형으로 분류하며, 각 단계는 이전 단계보다 엄격히 더 강력하다:

$$\text{Type 3 (Regular)} \subsetneq \text{Type 2 (Context-Free)} \subsetneq \text{Type 1 (Context-Sensitive)} \subsetneq \text{Type 0 (Unrestricted)}$$

| 유형 | 문법 | 오토마톤 | 규칙 형태 |
|------|---------|-----------|-----------|
| 3 | 정규(Regular) | DFA/NFA | $A \to aB$ 또는 $A \to a$ |
| 2 | 문맥 자유(Context-free) | PDA | $A \to \alpha$ |
| 1 | 문맥 의존(Context-sensitive) | LBA | $\alpha A \beta \to \alpha \gamma \beta$, $|\gamma| \geq 1$ |
| 0 | 무제한(Unrestricted) | TM | $\alpha \to \beta$ (제한 없음) |

## 2. 유형 3: 정규 문법(Regular Grammars)

### 2.1 정의

**정규 문법(regular grammar)**(또는 **우선형 문법(right-linear grammar)**)은 규칙이 다음으로 제한된다:
- $A \to aB$ (터미널 뒤에 변수)
- $A \to a$ (단일 터미널)
- $A \to \varepsilon$ (필요한 경우 시작 변수에 한해 허용)

**좌선형 문법(left-linear grammar)**은 $A \to Ba$와 $A \to a$를 사용한다. 두 방식 모두 정확히 정규 언어를 생성한다.

### 2.2 유한 오토마톤과의 동치성

정규 문법, DFA, NFA, 정규 표현식은 모두 동일한 클래스를 정의한다:

$$\text{Regular grammars} = \text{DFA} = \text{NFA} = \text{Regex}$$

### 2.3 성질 요약

- 폐쇄(Closed under): 합집합, 교집합, 여집합, 연접(concatenation), 클리니 스타(Kleene star), 역전(reversal), 준동형(homomorphism)
- 결정 가능(Decidable): 멤버십, 공집합 여부, 동치성, 포함 관계, 유한성
- 표현 불가: 계수($a^nb^n$), 중첩(nesting), 교차 직렬 의존성(cross-serial dependencies)

## 3. 유형 2: 문맥 자유 문법(Context-Free Grammars)

### 3.1 정의

**문맥 자유 문법(context-free grammar)**은 다음 형태의 규칙을 가진다:
$$A \to \alpha \quad \text{where } A \in V, \alpha \in (V \cup \Sigma)^*$$

변수 $A$는 **주변 문맥에 관계없이** $\alpha$로 교체될 수 있다.

### 3.2 PDA와의 동치성

$$\text{CFG} = \text{PDA (nondeterministic)}$$

주의: PDA의 경우 DPDA $\subsetneq$ NPDA이며, 유한 오토마톤에서 DFA = NFA인 것과 다르다.

### 3.3 성질 요약

- 폐쇄(Closed under): 합집합, 연접, 클리니 스타, 역전, 준동형, 정규 언어와의 교집합
- **비폐쇄(NOT closed under)**: 교집합, 여집합
- 결정 가능(Decidable): 멤버십(CYK, $O(n^3)$), 공집합 여부, 유한성
- **결정 불가(Undecidable)**: 동치성, 전체성(universality), 모호성(ambiguity), 정규성(regularity)
- 표현 불가: 교차 직렬 의존성($a^nb^nc^n$), 복사 언어(copy language, $ww$)

## 4. 유형 1: 문맥 의존 문법(Context-Sensitive Grammars)

### 4.1 정의

**문맥 의존 문법(CSG, context-sensitive grammar)**은 다음 형태의 규칙을 가진다:
$$\alpha A \beta \to \alpha \gamma \beta$$

여기서 $A \in V$, $\alpha, \beta \in (V \cup \Sigma)^*$, $\gamma \in (V \cup \Sigma)^+$이다.

핵심 제약 조건:
- 변수 $A$는 $\alpha$와 $\beta$라는 **문맥(context)** 안에서만 $\gamma$로 교체된다
- $|\gamma| \geq 1$: 규칙이 문장 형태(sentential form)를 짧게 만들 수 없다(비축소(non-contracting))
- 예외: $S$가 어떤 규칙의 오른쪽에도 등장하지 않는 경우 $S \to \varepsilon$는 허용된다

동치 표현으로 **단조 문법(monotone grammars)**을 사용한다: $\alpha \to \beta$ (단, $|\alpha| \leq |\beta|$).

### 4.2 문맥 의존 언어(CSL, Context-Sensitive Languages)

**예시:**
- $\{a^n b^n c^n \mid n \geq 1\}$ — 고전적인 비-CFL
- $\{ww \mid w \in \{a,b\}^*\}$ — 복사 언어(copy language)
- $\{a^{n^2} \mid n \geq 1\}$ — 완전 제곱수
- $\{a^{2^n} \mid n \geq 0\}$ — 2의 거듭제곱
- $\{a^p \mid p \text{ is prime}\}$ — 소수

### 4.3 $\{a^n b^n c^n\}$에 대한 문법

```
S → aSBC | aBC
CB → BC          (context-sensitive: reorder B and C)
aB → ab
bB → bb
bC → bc
cC → cc
```

규칙 $CB \to BC$는 $B$를 $C$ 앞으로 "정렬"하는데, 이는 문맥 의존성을 필요로 한다.

### 4.4 선형 제한 오토마톤(LBA, Linear Bounded Automata)

**선형 제한 오토마톤**은 입력이 차지하는 테이프 셀만 사용하도록 제한된(일정한 오버헤드 포함) 비결정론적 TM이다.

형식적으로: LBA는 헤드가 입력 경계(특별한 끝 마커로 표시) 밖으로 이동하지 않는 NTM이다.

**정리.** 언어가 문맥 의존인 것과 LBA가 수용하는 것은 동치이다.

### 4.5 CSL의 성질

- 폐쇄(Closed under): 합집합, 교집합, 여집합, 연접, 클리니 스타
- 여집합에 대한 폐쇄는 주요 미해결 문제였으나, 이머만-셀레프세니(Immerman-Szelepcsényi)에 의해 1987-88년에 해결되었다
- 결정 가능(Decidable): 멤버십(LBA 시뮬레이션으로 $O(n)$ 공간)
- **결정 불가(Undecidable)**: 공집합 여부, 동치성, 포함 관계

### 4.6 LBA 문제(The LBA Problem)

**미해결 문제(부분 해결):** LBA에 대한 공집합 문제(emptiness problem)는 결정 가능한가?

이는 여전히 미해결 문제이다. 표준 표현에서 $E_{LBA}$는 결정 불가임이 알려져 있다.

## 5. 유형 0: 무제한 문법(Unrestricted Grammars)

### 5.1 정의

**무제한 문법(unrestricted grammar)**(또는 **구조 문법(phrase-structure grammar)**)은 다음 형태의 규칙을 가진다:
$$\alpha \to \beta$$

여기서 $\alpha \in (V \cup \Sigma)^* V (V \cup \Sigma)^*$(최소 하나의 변수를 포함해야 함)이고 $\beta \in (V \cup \Sigma)^*$이다.

$|\alpha|$와 $|\beta|$ 사이의 관계에 대한 제약이 없다 — 규칙이 문자열을 짧게 만들 수 있다.

### 5.2 튜링 기계와의 동치성

$$\text{Unrestricted grammars} = \text{Turing machines (recognizers)}$$

무제한 문법이 생성하는 언어들은 정확히 **재귀 열거 가능(recursively enumerable)**(튜링 인식 가능) 언어들이다.

### 5.3 성질

- 폐쇄(Closed under): 합집합, 교집합, 연접, 클리니 스타
- **비폐쇄(NOT closed under)**: 여집합(여집합에 대해 닫혀 있으면 모든 RE 언어가 결정 가능해진다)
- **결정 불가(Undecidable)**: 멤버십, 공집합 여부, 동치성, 모든 것

## 6. 종합 비교

### 6.1 폐쇄 성질

| 연산 | 정규(Regular) | CF | CS | RE |
|-----------|---------|-----|-----|-----|
| 합집합(Union) | ✓ | ✓ | ✓ | ✓ |
| 교집합(Intersection) | ✓ | ✗ | ✓ | ✓ |
| 여집합(Complement) | ✓ | ✗ | ✓ | ✗ |
| 연접(Concatenation) | ✓ | ✓ | ✓ | ✓ |
| 클리니 스타(Kleene star) | ✓ | ✓ | ✓ | ✓ |
| 준동형(Homomorphism) | ✓ | ✓ | ✗ | ✓ |
| 역준동형(Inverse homomorphism) | ✓ | ✓ | ✓ | ✓ |
| 정규 언어와의 교집합(Intersection with regular) | ✓ | ✓ | ✓ | ✓ |

### 6.2 결정 문제

| 문제 | 정규(Regular) | CF | CS | RE |
|---------|---------|-----|-----|-----|
| 멤버십(Membership) | ✓ | ✓ | ✓ | ✗ |
| 공집합 여부(Emptiness) | ✓ | ✓ | ✗ | ✗ |
| 유한성(Finiteness) | ✓ | ✓ | ✗ | ✗ |
| 동치성(Equivalence) | ✓ | ✗ | ✗ | ✗ |
| 전체성(Universality) | ✓ | ✗ | ✗ | ✗ |
| 포함 관계(Containment) | ✓ | ✗ | ✗ | ✗ |

### 6.3 유형별 언어 예시

| 언어 | 최소 유형 |
|----------|--------------|
| $a^*b^*$ | 정규(Regular, Type 3) |
| $a^n b^n$ | 문맥 자유(Context-free, Type 2) |
| $a^n b^n c^n$ | 문맥 의존(Context-sensitive, Type 1) |
| $A_{TM}$ | 재귀 열거 가능(Recursively enumerable, Type 0) |
| $\overline{A_{TM}}$ | 어떤 유형에도 속하지 않음 |

## 7. 계층 너머

### 7.1 결정 가능(Recursive) 언어

CS와 RE 사이에 중요한 클래스가 위치한다:

$$\text{CS} \subsetneq \text{Decidable} \subsetneq \text{RE}$$

결정 가능 언어는 항상 정지하는 TM에 의해 인식된다. 일부 결정 가능 언어는 문맥 의존이 아니다(예: 초선형(superlinear) 공간을 필요로 하는 언어).

### 7.2 계층 바깥의 언어들

임의의 알파벳 위에는 셀 수 없이 많은 언어가 있지만, TM은 셀 수 있는 수만큼만 존재한다. 따라서 **대부분의** 언어는 재귀 열거 가능하지도 않다.

**기수(cardinality) 논증:**
- $|\{0,1\}^*|$은 가산 집합(countable)
- $|\mathcal{P}(\{0,1\}^*)|$ (모든 언어의 집합)는 비가산 집합(uncountable)
- $|\{\text{TMs}\}|$은 가산 집합
- 따라서 대부분의 언어는 TM 인식기를 가지지 않는다

## 연습 문제

### 연습 1: 분류

각 언어를 가장 작은 촘스키 계층 단계로 분류하라:

1. $\{a^n b^m \mid n > m\}$
2. $\{a^n b^n c^n d^n \mid n \geq 0\}$
3. $\{w \in \{a,b\}^* \mid |w|_a = |w|_b\}$
4. $\{a^p \mid p \text{ is prime}\}$
5. $\{\langle M \rangle \mid M \text{ halts on } \varepsilon\}$

### 연습 2: 문맥 의존 문법

$\{a^n b^n c^n \mid n \geq 1\}$에 대한 문맥 의존 문법을 작성하라. $aabbcc$의 유도(derivation) 과정을 추적하라.

### 연습 3: 폐쇄 성질

1. 문법을 구성하여 문맥 의존 언어가 합집합에 대해 닫혀 있음을 보여라.
2. CSL의 여집합에 대한 폐쇄가 (DFA에서 상태를 뒤집으면 되는 정규 언어와 달리) 왜 자명하지 않은지 설명하라.

---

*이전: [레슨 12 — 계산 복잡도(Computational Complexity)](12_Computational_Complexity.md) | 다음: [레슨 14 — 고급 주제와 응용(Advanced Topics and Applications)](14_Advanced_Topics_and_Applications.md)*
