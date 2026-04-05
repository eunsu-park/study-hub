# 레슨 12: 계산 복잡도(Computational Complexity)

**이전**: [환원 가능성](./11_Reducibility.md) | **다음**: [촘스키 위계](./13_The_Chomsky_Hierarchy.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 시간 및 공간 복잡도 클래스를 정의한다
2. P, NP, coNP, PSPACE 클래스의 특성을 이해한다
3. 쿡-레빈 정리(Cook-Levin theorem)와 그 의의를 설명한다
4. NP-완전(NP-complete) 문제들 사이의 다항 시간 환원(polynomial-time reduction)을 수행한다
5. P vs NP 문제와 그 함의를 명확히 설명한다

---

결정 가능성이 문제를 아예 풀 수 있는지를 묻는다면, 복잡도 이론은 *얼마나 효율적으로* 풀 수 있는지를 묻는다. 다항 시간 알고리즘과 지수 시간 알고리즘의 차이는, 몇 초 안에 끝나는 프로그램과 우주의 수명을 넘어서는 프로그램의 차이이다. P vs NP 문제 — 해(解)를 빠르게 검증할 수 있는 모든 문제를 빠르게 풀 수도 있는가? — 는 컴퓨터 과학에서 가장 중요한 미해결 문제로, 그 답에는 100만 달러의 상금이 걸려 있다.

## 1. 복잡도 측정

### 1.1 시간 복잡도(Time Complexity)

결정기(decider) $M$의 **시간 복잡도**는 함수 $t: \mathbb{N} \to \mathbb{N}$으로, $t(n)$은 길이 $n$인 임의의 입력에 대해 $M$이 수행하는 최대 단계 수이다.

어떤 상수 $c$와 충분히 큰 모든 $n$에 대해 $t(n) \leq c \cdot f(n)$이 성립하면, $M$이 $O(f(n))$ 시간에 실행된다고 말한다.

### 1.2 공간 복잡도(Space Complexity)

결정기 $M$의 **공간 복잡도**는 $s(n)$으로, 길이 $n$인 임의의 입력에 대해 $M$이 사용하는 최대 테이프 셀 수이다.

### 1.3 복잡도 클래스(Complexity Classes)

**복잡도 클래스**는 자원 한계로 정의되는 언어들의 집합이다:

$$\text{TIME}(f(n)) = \{L \mid L \text{ is decided by a TM in } O(f(n)) \text{ time}\}$$
$$\text{NTIME}(f(n)) = \{L \mid L \text{ is decided by an NTM in } O(f(n)) \text{ time}\}$$
$$\text{SPACE}(f(n)) = \{L \mid L \text{ is decided by a TM using } O(f(n)) \text{ space}\}$$

## 2. 클래스 P

$$\mathbf{P} = \bigcup_{k \geq 0} \text{TIME}(n^k)$$

$\mathbf{P}$는 결정론적 튜링 기계(deterministic TM)가 **다항 시간(polynomial time)**에 결정할 수 있는 언어들의 클래스이다.

$\mathbf{P}$는 "효율적으로 풀 수 있는" 문제들의 클래스로 간주된다.

### 2.1 P에 속하는 예시

| 문제 | 시간 | 알고리즘 |
|---------|------|-----------|
| DFA 수용(acceptance) | $O(n)$ | DFA 시뮬레이션 |
| CFG 멤버십(membership) | $O(n^3)$ | CYK 알고리즘 |
| 그래프 연결성(connectivity) | $O(n + m)$ | BFS/DFS |
| 최단 경로(Shortest path) | $O(n^2)$ | 다익스트라(Dijkstra) |
| 2-SAT | $O(n)$ | 함의 그래프 SCC |
| 소수 판별(Primality testing) | $O(n^6)$ | AKS 알고리즘 |
| 선형 프로그래밍(Linear programming) | $O(n^3)$ | 내점법(Interior point methods) |

### 2.2 P의 강건성(Robustness of P)

$\mathbf{P}$는 다음 어느 경우를 사용하더라도 동일하다:
- 단일 테이프 또는 다중 테이프 TM (다항 감속만 발생)
- "합리적인" 임의의 결정론적 계산 모델

이는 **확장된 처치-튜링 명제(Extended Church-Turing thesis)**를 뒷받침한다: 어떤 "합리적인" 계산 모델도 다른 모델을 최대 다항 오버헤드로 시뮬레이션할 수 있다.

## 3. 클래스 NP

$$\mathbf{NP} = \bigcup_{k \geq 0} \text{NTIME}(n^k)$$

$\mathbf{NP}$는 **비결정론적(nondeterministic)** TM이 다항 시간에 결정할 수 있는 언어들의 클래스이다.

### 3.1 검증기(Verifier) 특성화

동치 조건으로, $L \in \mathbf{NP}$인 것과 다항 시간 **검증기(verifier)** $V$가 존재하는 것은 동치이다:

$$L = \{w \mid \exists c \in \{0,1\}^{p(|w|)}, V(w, c) = \text{accept}\}$$

여기서 $c$는 다항 길이의 **인증서(certificate)**(또는 증인(witness))이고, $V$는 다항 시간에 실행된다.

**직관:** NP 문제는 해를 찾기는 어렵지만 검증하기는 쉬운 문제들이다.

### 3.2 NP에 속하는 예시

| 문제 | 인증서 | 검증 |
|---------|-------------|-------------|
| SAT | 만족 배정(Satisfying assignment) | 공식 평가 |
| 해밀턴 경로(Hamiltonian Path) | 경로 목록 | 모든 정점 방문 여부 확인 |
| 클리크(Clique) | $k$개 정점의 집합 | 모든 쌍이 연결되는지 확인 |
| 부분 집합 합(Subset Sum) | 숫자들의 부분 집합 | 합산 후 비교 |
| 그래프 채색(Graph Coloring) | 색상 배정 | 인접한 정점이 같은 색인지 확인 |
| 외판원 문제(Traveling Salesman, 결정형) | 순회 경로 | 간선 가중치 합산 후 한계와 비교 |

### 3.3 P ⊆ NP

모든 결정론적 TM은 (자명한) 비결정론적 TM이다. 따라서 $\mathbf{P} \subseteq \mathbf{NP}$이다.

$\mathbf{P} \stackrel{?}{=} \mathbf{NP}$라는 질문은 이론 컴퓨터 과학에서 가장 중요한 미해결 문제이다.

## 4. NP-완전성(NP-Completeness)

### 4.1 다항 시간 환원(Polynomial-Time Reductions)

언어 $A$가 $B$로 **다항 시간 환원 가능(polynomial-time reducible)**하다는 것은 $A \leq_P B$로 표기하며, 다항 시간에 계산 가능한 함수 $f$가 존재하여 $w \in A \iff f(w) \in B$가 성립하는 것을 의미한다.

### 4.2 정의

언어 $B$가 **NP-완전(NP-complete)**하려면:
1. $B \in \mathbf{NP}$
2. 모든 $A \in \mathbf{NP}$에 대해 $A \leq_P B$가 성립

조건 2만 만족하는 언어는 **NP-난해(NP-hard)**이다.

**의의:** NP-완전 문제 중 하나라도 다항 시간 알고리즘을 가지면 $\mathbf{P} = \mathbf{NP}$이다. NP-완전 문제 중 하나라도 P에 속하지 않으면 $\mathbf{P} \neq \mathbf{NP}$이다.

### 4.3 쿡-레빈 정리(The Cook-Levin Theorem)

**정리(쿡-레빈).** SAT는 NP-완전이다.

**SAT(불리언 만족 가능성(Boolean Satisfiability)):** 불리언 공식 $\phi$가 주어졌을 때, $\phi$를 참으로 만드는 진리값 배정이 존재하는가?

*증명 아이디어:* 시간 $n^k$에 실행되는 NTM $M$과 입력 $w$가 주어졌을 때, $M$이 $w$를 수용하는 것과 동치인 만족 가능한 불리언 공식 $\phi$를 구성한다. 이 공식은 다음을 인코딩한다:
- 계산 태블로(computation tableau)(각 시간 단계에서의 상태, 테이프 내용, 헤드 위치)에 대한 변수들
- 유효한 시작 구성을 보장하는 절(clauses)
- 각 단계가 전이 함수를 따름을 보장하는 절
- 수용 구성에 도달함을 보장하는 절

이 구성은 $|w|$에 대해 다항 시간이므로 SAT가 NP-난해임이 증명된다. SAT는 NP에 속한다(만족 배정이 인증서이다). $\square$

## 5. 고전적 NP-완전 문제들

### 5.1 환원 연쇄(Reductions Chain)

```
SAT
 └── 3-SAT (restrict clauses to 3 literals)
      ├── CLIQUE (k-clique in graph)
      ├── VERTEX COVER (k vertices covering all edges)
      ├── INDEPENDENT SET
      ├── 3-COLORING
      └── SUBSET SUM
           └── PARTITION
                └── BIN PACKING
```

### 5.2 3-SAT

**3-SAT:** SAT를 3-CNF(각 절이 정확히 3개의 리터럴로 구성)로 제한한 것.

**정리.** 3-SAT는 NP-완전이다.

*SAT에서의 환원:* 보조 변수를 사용하여 긴 절을 여러 개의 3-리터럴 절로 교체한다.

### 5.3 CLIQUE

**CLIQUE:** 그래프 $G$와 정수 $k$가 주어졌을 때, $G$에 $k$개 정점으로 이루어진 완전 부분 그래프가 존재하는가?

**3-SAT에서의 환원:** 각 절마다 리터럴당 하나의 노드를 생성한다. 모순되는 경우($x_i$와 $\neg x_i$)를 제외하고 서로 다른 절의 노드들을 연결한다. 만족 배정은 $k$ = 절의 수인 $k$-클리크에 대응된다.

### 5.4 VERTEX COVER

**VERTEX COVER:** 그래프 $G$와 정수 $k$가 주어졌을 때, 모든 간선에 닿는 $k$개 정점의 집합이 존재하는가?

**CLIQUE에서의 환원:** $G$에 $k$-클리크가 존재하는 것과 $\overline{G}$에 크기 $|V| - k$의 정점 커버가 존재하는 것은 동치이다.

### 5.5 SUBSET SUM

**SUBSET SUM:** 정수 $a_1, \ldots, a_n$과 목표값 $t$가 주어졌을 때, 합이 $t$가 되는 부분 집합이 존재하는가?

**3-SAT에서의 환원:** 변수와 절을 신중히 선택한 정수들로 인코딩하며, 각 자릿수 위치가 변수 및 절에 대응된다.

## 6. coNP

$$\mathbf{coNP} = \{L \mid \overline{L} \in \mathbf{NP}\}$$

coNP는 **여(complement)**가 NP에 속하는 언어들의 클래스이다. 동치 표현으로, 효율적으로 검증 가능한 **반례(disproof)**를 가지는 문제들의 클래스이다.

**예시:**
- TAUTOLOGY(공식이 항상 참인가?) — coNP에 속함
- UNSAT(공식이 만족 불가능한가?) — coNP-완전

**관계:** $\mathbf{P} \subseteq \mathbf{NP} \cap \mathbf{coNP}$. $\mathbf{NP} = \mathbf{coNP}$인지 여부는 미해결이다.

## 7. 공간 복잡도 클래스

### 7.1 PSPACE

$$\mathbf{PSPACE} = \bigcup_{k \geq 0} \text{SPACE}(n^k)$$

### 7.2 주요 관계

$$\mathbf{P} \subseteq \mathbf{NP} \subseteq \mathbf{PSPACE} \subseteq \mathbf{EXPTIME}$$

**새비치 정리(Savitch's Theorem):** $\text{NSPACE}(f(n)) \subseteq \text{SPACE}(f(n)^2)$

따라서 $\mathbf{NPSPACE} = \mathbf{PSPACE}$이다(다항 공간에서는 비결정론성이 도움이 되지 않는다).

### 7.3 PSPACE-완전 문제들

- **TQBF**(참 양화 불리언 공식(True Quantified Boolean Formula)): $\exists x_1 \forall x_2 \exists x_3 \cdots \phi(x_1, \ldots, x_n)$
- **일반화된 지리(Generalized geography)**, **일반화된 체스(generalized chess)** 등

## 8. 복잡도 클래스 요약

```
EXPTIME
└── PSPACE = NPSPACE (Savitch)
    └── NP ∪ coNP
        ├── NP
        │   └── P
        └── coNP
            └── P
```

$\mathbf{P} \neq \mathbf{EXPTIME}$이므로 적어도 하나의 포함 관계는 진부분집합 관계여야 하지만, 어느 것인지는 알 수 없다!

## 연습 문제

### 연습 1: 복잡도 분류

각 문제에 대해 가장 잘 알려진 복잡도 클래스(P, NP-완전, 또는 PSPACE-완전)를 파악하라:

1. 비가중 그래프에서의 최단 경로
2. 3-CNF 형태의 불리언 공식의 만족 가능성 판별
3. QBF(양화 불리언 공식(quantified Boolean formula))의 참 여부 판별
4. 그래프의 2-채색(2-coloring)

### 연습 2: 다항 환원

INDEPENDENT SET $\leq_P$ VERTEX COVER임을 보여라. (힌트: $S$가 독립 집합인 것과 $V \setminus S$가 정점 커버인 것은 동치이다.)

### 연습 3: NP 멤버십

각 문제가 NP에 속함을 인증서와 다항 시간 검증기를 서술하여 보여라:

1. **COMPOSITE**: $n$이 주어졌을 때, $n$이 합성수(소수가 아닌 수)인가?
2. **그래프 동형(GRAPH ISOMORPHISM)**: $G_1, G_2$가 주어졌을 때, 두 그래프가 동형인가?
3. **집합 커버(SET COVER)**: 집합 $S_1, \ldots, S_m$과 정수 $k$가 주어졌을 때, $k$개의 집합으로 모든 원소를 커버할 수 있는가?

---

*이전: [레슨 11 — 환원 가능성](11_Reducibility.md) | 다음: [레슨 13 — 촘스키 계층(The Chomsky Hierarchy)](13_The_Chomsky_Hierarchy.md)*
