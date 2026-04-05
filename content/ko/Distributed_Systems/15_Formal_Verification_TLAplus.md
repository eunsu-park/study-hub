# Lesson 15: TLA+를 이용한 형식 검증

[Overview](./00_Overview.md) | [이전: 분산 조정 프리미티브](./14_Distributed_Coordination_Primitives.md) | [다음: 캡스톤 — 분산 KV 스토어 구축](./16_Capstone_Building_Distributed_KV_Store.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있게 됩니다:

1. 분산 시스템의 정확성을 위해 테스트를 넘어 형식 검증이 필요한 이유를 설명한다
2. 변수, 연산자, 시간 논리, prime 표기법을 사용하여 TLA+ 명세를 작성한다
3. 동시성 프로토콜에 대한 safety(불변식) 및 liveness(시간적) 속성을 명세한다
4. PlusCal을 사용하여 의사코드 수준의 알고리즘을 검증 가능한 TLA+ 명세로 변환한다
5. TLC model checker를 적용하여 프로토콜 속성을 철저히 검증하고 반례를 해석한다

---

## 목차

1. [왜 형식 검증인가](#1-왜-형식-검증인가)
2. [TLA+ 개요](#2-tla-개요)
3. [TLA+ 언어 기초](#3-tla-언어-기초)
4. [간단한 명세 작성: 상호 배제](#4-간단한-명세-작성-상호-배제)
5. [PlusCal: TLA+를 위한 알고리즘 언어](#5-pluscal-tla를-위한-알고리즘-언어)
6. [TLC Model Checker](#6-tlc-model-checker)
7. [사례 연구: 단순화된 Raft Leader Election의 TLA+ 명세](#7-사례-연구-단순화된-raft-leader-election의-tla-명세)
8. [TLA+로 합의 프로토콜 모델링하기](#8-tla로-합의-프로토콜-모델링하기)
9. [AWS의 TLA+ 활용](#9-aws의-tla-활용)
10. [실용적 검증 워크플로우](#10-실용적-검증-워크플로우)
11. [기타 형식 방법론](#11-기타-형식-방법론)
12. [구현: TLA+ 명세 검증기와 PlusCal 번역기](#12-구현-tla-명세-검증기와-pluscal-번역기)
13. [요약 및 추가 읽을거리](#13-요약-및-추가-읽을거리)

---

## 1. 왜 형식 검증인가

### 1.1 테스트의 한계

분산 시스템 테스트는 근본적으로 제한적입니다:

```
Testing:     Explores a tiny fraction of the state space
             ┌─────────────────────────────────┐
             │         All possible states      │
             │    ┌─────────────┐               │
             │    │             │               │
             │    │  Tested     │  Untested     │
             │    │  states     │  states       │
             │    │  (tiny)     │  (vast)       │
             │    └─────────────┘               │
             └─────────────────────────────────┘

Verification: Exhaustively checks ALL reachable states
             ┌─────────────────────────────────┐
             │         All possible states      │
             │  ┌───────────────────────────┐   │
             │  │                           │   │
             │  │  Verified                 │   │ Unreachable
             │  │  (all reachable states)   │   │ states
             │  │                           │   │
             │  └───────────────────────────┘   │
             └─────────────────────────────────┘
```

**테스트가 분산 시스템의 버그를 놓치는 이유**:

| 버그 유형 | 테스트가 놓치는 이유 | 예시 |
|----------|----------------------|---------|
| 경쟁 조건 | 특정 인터리빙이 필요 | 같은 term에 두 leader가 선출됨 |
| 장애 복구 에지 케이스 | 특정 장애 타이밍이 필요 | leader 교체 중 데이터 손실 |
| Liveness 위반 | 장시간 실행에서만 나타날 수 있음 | 프로세스가 영원히 기아 상태 |
| 다중 노드 상호작용 | 순서의 조합 폭발 | 3개 노드, 5개 메시지 = 120개 이상의 순서 |

### 1.2 형식 검증이 제공하는 것

형식 검증은 시스템의 **모든 가능한 동작**에 대한 속성을 증명합니다:

- **Safety**: "나쁜 일은 절대 발생하지 않음" (예: term당 최대 하나의 leader)
- **Liveness**: "좋은 일이 결국 발생함" (예: leader가 결국 선출됨)

```
                Testing              Formal Verification
Finds bugs?       Yes                   Yes
Proves absence?   No                    Yes (within the model)
Scales to         Large systems         Small models (~10-20 nodes)
production code?  Yes                   No (verifies the design, not code)
Human effort      Write test cases      Write specification
Automation        High                  High (model checking)
```

### 1.3 분산 시스템의 하이젠버그

분산 시스템에서 가장 위험한 버그는 정확한 타이밍과 메시지 순서에 의존하는 **비결정적** 장애입니다:

```
미묘한 split-brain 버그를 보여주는 타임라인:

Node A (리더, term 1):        ──[append]──────[commit]──────►
                                     │                    ▲
                                     ▼                    │ (지연된 ack)
네트워크 파티션:              ═══════════════════════════════
                                     │                    │
Node B (리더로 선출, term 2): ──[elect]──[append']──[commit']──►
                                               │
                                               ▼
Node C:                       ──────────────[accept B]──────────►

버그: Node A가 인덱스 3의 항목(term 1)을 커밋하는 동시에
     Node B가 인덱스 3에 다른 항목(term 2)을 커밋합니다.
     파티션 + 정확한 타이밍이 필요하여 재현이 어렵습니다.
```

이러한 버그의 공통적 특성:

| 속성 | 테스트 | 형식 검증 |
|---|---|---|
| 커버리지 | 경로를 무작위로 샘플링 | **모든** 도달 가능 상태를 탐색 |
| 재현성 | 어려움 — 타이밍 의존적 | 결정적 반례 |
| 확신 | 확률적 | 수학적 증명 (모델 범위 내) |
| 확장성 | 테스트 수에 선형 | 상태 변수에 지수적 |
| 노력 | 초기 비용 낮음, 유지보수 높음 | 초기 비용 높음, 유지보수 낮음 |

### 1.4 테스트를 통과한 실제 장애 사례

**Amazon DynamoDB (2015):** 메타데이터 손상 버그가 6개월간의 테스트를 통과했고, 코디네이터가 스토리지 노드가 아직 오래된 멤버십 뷰를 유지하고 있는 상태에서 재구성을 완료할 수 있는 엣지 케이스를 TLA+ 형식 명세가 발견한 후에야 확인되었습니다.

**Apache ZooKeeper (CVE-2017-5637):** 리더 선출 프로토콜의 버그로 인해 4개의 특정 네트워크 이벤트 시퀀스에서 두 개의 리더가 공존할 수 있었습니다. 무작위 테스트는 그 정확한 시퀀스를 한 번도 생성하지 못했습니다.

**CockroachDB (2019):** 트랜잭션 직렬화 이상 현상은 3개의 동시 트랜잭션과 특정 충돌 패턴을 필요로 했습니다. 모든 단위 및 통합 테스트를 통과한 후 Jepsen 테스트를 통해 발견되었습니다.

### 1.5 형식 검증의 위치

형식 검증은 테스트를 대체하지 않습니다 — 다른 수준에서 작동합니다:

```
                   ┌─────────────────────────────┐
                   │     명세 (TLA+)              │  ← 형식 검증이
                   │   "알고리즘이 정확하다"        │     이 수준을 검증
                   └──────────────┬──────────────┘
                                  │ 정제 간극
                   ┌──────────────▼──────────────┐
                   │   구현 (코드)                 │  ← 테스트가
                   │   "코드가 명세와 일치한다"     │     이 수준을 검증
                   └──────────────┬──────────────┘
                                  │ 환경 간극
                   ┌──────────────▼──────────────┐
                   │   배포 (프로덕션)             │  ← 카오스 엔지니어링이
                   │   "실제 장애 하에서 작동한다"  │     이 수준을 검증
                   └─────────────────────────────┘
```

검증 스택:
- **모델 체킹 (TLA+/TLC):** 알고리즘이 정확함을 증명
- **단위/통합 테스트:** 코드가 알고리즘을 구현함을 검증
- **Jepsen/카오스 테스트:** 실제 장애 모드에서의 동작을 검증

```
Design Phase         ──▶  Write TLA+ specification
                          ┌──────────────────────┐
                          │ TLC model checker     │
                          │ finds design bugs     │
                          │ BEFORE any code       │
                          └──────┬───────────────┘
                                 │
Implementation Phase ──▶  Write code based on verified spec
                          ┌──────────────────────┐
                          │ Unit tests, property  │
                          │ tests verify code     │
                          │ matches spec          │
                          └──────┬───────────────┘
                                 │
Production Phase     ──▶  Monitor invariants at runtime
                          ┌──────────────────────┐
                          │ Jepsen, chaos testing │
                          │ verify deployment     │
                          └──────────────────────┘
```

---

## 2. TLA+ 개요

### 2.1 TLA+란?

TLA+ (Temporal Logic of Actions)는 Leslie Lamport가 만든 형식 명세 언어입니다. 시스템을 실행 가능한 프로그램이 아닌 수학적 객체 — 상태, 전이, 속성 — 로 기술합니다.

**핵심 통찰**: TLA+는 시스템이 *어떻게* 작동하는지(구현)가 아닌 시스템이 *무엇*을 하는지(동작)를 기술합니다. 이 추상화 수준이 바로 설계 버그를 찾는 데 강력한 이유입니다.

### 2.2 핵심 개념

**상태(State)**: 모든 변수에 값을 할당한 것. 상태는 한 시점에서 시스템의 스냅샷입니다.

```
Example state of a mutual exclusion system:
  pc = [p1 ↦ "waiting", p2 ↦ "critical"]
  lock = "p2"
```

**액션(Action)**: 상태와 그 후속 상태 사이의 관계. 액션은 시스템의 한 단계를 기술합니다.

```
Example action: Process p1 enters the critical section
  Precondition: pc[p1] = "waiting" ∧ lock = "free"
  Effect: pc'[p1] = "critical" ∧ lock' = "p1"
  (primed variables refer to the next state)
```

**동작(Behavior)**: 무한한 상태 시퀀스. 동작은 시스템의 한 가능한 실행을 나타냅니다.

```
s0 → s1 → s2 → s3 → s4 → ...

Example behavior:
  [pc = [p1↦"idle", p2↦"idle"], lock = "free"]
  → [pc = [p1↦"waiting", p2↦"idle"], lock = "free"]
  → [pc = [p1↦"critical", p2↦"idle"], lock = "p1"]
  → [pc = [p1↦"idle", p2↦"idle"], lock = "free"]
  → ...
```

**명세(Specification)**: 모든 유효한 동작의 집합을 기술하는 공식:

```
Spec ≡ Init ∧ □[Next]_vars

Where:
  Init    = initial state predicate
  Next    = disjunction of all possible actions
  □       = "always" (in every state of the behavior)
  [A]_v   = A ∨ (v' = v)  (action A or stuttering)
  vars    = tuple of all variables
```

### 2.3 왜 Stuttering인가?

`[Next]_vars` 표기법은 **stuttering step** — 아무것도 변하지 않는 단계 — 을 허용합니다. 이는 조합 가능성(composability)을 위해 필수적입니다: 두 명세를 결합할 때, 한 시스템의 액션은 다른 시스템에게 stuttering처럼 보입니다.

```
System A: s0 ─a1─▶ s1 ─a2─▶ s2 ─a3─▶ s3
System B: t0 ─────────────── b1 ──▶ t1 ────── b2 ──▶ t2

Combined: (s0,t0) ─a1─▶ (s1,t0) ─a2─▶ (s2,t0) ─b1─▶ (s2,t1) ─a3─▶ (s3,t1) ─b2─▶ (s3,t2)

B sees stuttering during a1, a2, a3
A sees stuttering during b1, b2
```

---

## 3. TLA+ 언어 기초

### 3.1 변수와 상수

```tla
---- MODULE SimpleKV ----
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Keys, Values, Nodes   \* Model parameters (set at check time)
VARIABLES store, pending        \* State variables (change during execution)

vars == <<store, pending>>      \* Tuple of all variables (for stuttering)
====
```

- `CONSTANTS`: 주어진 모델 체크에 대해 고정된 매개변수. 예: `Keys = {"k1", "k2"}`.
- `VARIABLES`: 가변 상태. 각 상태는 모든 변수에 값을 할당합니다.
- `vars`: 모든 변수를 그룹화하는 편의 튜플.

### 3.2 데이터 구조

TLA+는 집합, 함수, 레코드, 시퀀스, 튜플에 대한 내장 지원을 가지고 있습니다:

```tla
\* Sets
S == {1, 2, 3}
T == {x \in S : x > 1}           \* Filter: {2, 3}
U == {x * 2 : x \in S}           \* Map: {2, 4, 6}

\* Functions (total mappings from domain to range)
f == [x \in {"a", "b"} |-> 0]    \* f["a"] = 0, f["b"] = 0
g == [f EXCEPT !["a"] = 1]       \* g["a"] = 1, g["b"] = 0 (functional update)

\* Records (functions with string domains)
r == [name |-> "Alice", age |-> 30]
r.name                              \* "Alice"

\* Sequences (functions with domain 1..n)
seq == <<1, 2, 3>>
Len(seq)                            \* 3
Append(seq, 4)                      \* <<1, 2, 3, 4>>
Head(seq)                           \* 1
Tail(seq)                           \* <<2, 3>>

\* Tuples (just sequences used as pairs/triples)
pair == <<x, y>>
```

### 3.3 액션과 Priming

**액션**은 primed된 변수와 unprimed된 변수를 포함하는 불리언 공식입니다:
- Unprimed `x`는 현재 상태를 참조
- Primed `x'`는 다음 상태를 참조

```tla
\* Action: increment counter
Increment ==
    /\ counter < MAX          \* Precondition (current state)
    /\ counter' = counter + 1 \* Effect (next state)
    /\ UNCHANGED other_var    \* other_var does not change

\* UNCHANGED x is shorthand for x' = x
```

### 3.4 ENABLED와 공정성(Fairness)

```tla
\* ENABLED A is TRUE if action A can be taken in the current state
\* (i.e., there exists a next state satisfying A)

\* Weak fairness: if A is continuously enabled, it is eventually taken
WF_vars(A) == □(□ENABLED <A>_vars => ◇<A>_vars)

\* Strong fairness: if A is repeatedly enabled, it is eventually taken
SF_vars(A) == □(□◇ENABLED <A>_vars => ◇<A>_vars)
```

**언제 어떤 것을 사용하나**:
- Weak fairness (WF): 액션이 *지속적으로* 활성화되어야 함. 한번 활성화되면 활성 상태를 유지하는 액션에 사용 (예: "프로세스가 항상 진행할 수 있음").
- Strong fairness (SF): 액션이 *반복적으로* 활성화되지만 중간에 비활성화될 수 있음. 다른 것과 경쟁하는 액션에 사용 (예: 잠금 획득).

### 3.5 시간 연산자

| 연산자 | TLA+ 구문 | 의미 |
|----------|-------------|---------|
| Always | `[]P` 또는 `□P` | P가 모든 동작의 모든 상태에서 참 |
| Eventually | `<>P` 또는 `◇P` | P가 모든 동작의 적어도 하나의 상태에서 참 |
| Leads-to | `P ~> Q` | P가 참이 될 때마다 Q가 결국 참이 됨 |
| Always-eventually | `[]<>P` | P가 무한히 자주 참 |
| Eventually-always | `<>[]P` | P가 결국 참이 되고 참으로 유지됨 |

**중요한 조합**:

```
Safety:    □(¬bad_state)          "Bad things never happen"
Liveness:  ◇(good_state)          "Good things eventually happen"
Progress:  request ~> response     "Every request gets a response"
Fairness:  □◇(enabled => taken)   "Enabled actions eventually execute"
```

---

## 4. 간단한 명세 작성: 상호 배제

### 4.1 문제

두 프로세스가 임계 영역에 진입하려 합니다. 어느 시점에서든 최대 하나만 임계 영역에 있어야 합니다.

### 4.2 TLA+ 명세

```tla
---- MODULE MutualExclusion ----
EXTENDS Integers

\* Two processes
CONSTANTS Procs
ASSUME Procs = {0, 1}

VARIABLES pc, turn, flag

vars == <<pc, turn, flag>>

\* Process states: "idle", "want", "wait", "critical"

Init ==
    /\ pc = [p \in Procs |-> "idle"]
    /\ turn = 0
    /\ flag = [p \in Procs |-> FALSE]

\* ----- Actions -----

\* Process p wants to enter critical section
Want(p) ==
    /\ pc[p] = "idle"
    /\ pc' = [pc EXCEPT ![p] = "want"]
    /\ flag' = [flag EXCEPT ![p] = TRUE]
    /\ UNCHANGED turn

\* Process p sets turn and waits (Peterson's algorithm)
Wait(p) ==
    /\ pc[p] = "want"
    /\ turn' = 1 - p                    \* Give priority to other process
    /\ pc' = [pc EXCEPT ![p] = "wait"]
    /\ UNCHANGED flag

\* Process p enters critical section
Enter(p) ==
    /\ pc[p] = "wait"
    /\ \/ flag[1-p] = FALSE             \* Other process not interested
       \/ turn = p                       \* It's our turn
    /\ pc' = [pc EXCEPT ![p] = "critical"]
    /\ UNCHANGED <<turn, flag>>

\* Process p exits critical section
Exit(p) ==
    /\ pc[p] = "critical"
    /\ flag' = [flag EXCEPT ![p] = FALSE]
    /\ pc' = [pc EXCEPT ![p] = "idle"]
    /\ UNCHANGED turn

\* ----- Specification -----

Next ==
    \E p \in Procs :
        \/ Want(p)
        \/ Wait(p)
        \/ Enter(p)
        \/ Exit(p)

Spec == Init /\ [][Next]_vars
        /\ WF_vars(Want(0)) /\ WF_vars(Wait(0))
        /\ WF_vars(Enter(0)) /\ WF_vars(Exit(0))
        /\ WF_vars(Want(1)) /\ WF_vars(Wait(1))
        /\ WF_vars(Enter(1)) /\ WF_vars(Exit(1))

\* ----- Properties to check -----

\* Safety: mutual exclusion
MutualExclusion ==
    ~(pc[0] = "critical" /\ pc[1] = "critical")

\* Liveness: no starvation (if a process wants the CS, it eventually gets it)
NoStarvation ==
    /\ (pc[0] = "want" ~> pc[0] = "critical")
    /\ (pc[1] = "want" ~> pc[1] = "critical")

\* Type invariant (helps TLC check faster)
TypeOK ==
    /\ pc \in [Procs -> {"idle", "want", "wait", "critical"}]
    /\ turn \in Procs
    /\ flag \in [Procs -> BOOLEAN]

====
```

### 4.3 상태 공간 시각화

```
The TLC model checker explores all reachable states:

Initial state:
  pc=[0↦idle, 1↦idle], turn=0, flag=[0↦F, 1↦F]

From here, two actions are possible:
  Want(0) or Want(1)

Want(0) leads to:
  pc=[0↦want, 1↦idle], turn=0, flag=[0↦T, 1↦F]

  From here: Wait(0) or Want(1)
  ... and so on

State graph (partial):

  (idle,idle) ──Want(0)──▶ (want,idle)
       │                       │
   Want(1)                 Wait(0)
       │                       │
       ▼                       ▼
  (idle,want) ──────▶    (wait,idle) ──Enter(0)──▶ (critical,idle)
                              │                         │
                          Want(1)                    Exit(0)
                              │                         │
                              ▼                         ▼
                        (wait,want) ──────▶      (idle,idle)

TLC explores ALL paths through this graph, checking
MutualExclusion at every state.
```

### 4.4 TLC가 검사하는 것

위의 상호 배제 명세에 대해:

| 속성 | 유형 | TLC가 수행하는 것 |
|----------|------|---------------|
| TypeOK | 불변식 | 도달 가능한 모든 상태에서 검사 |
| MutualExclusion | 불변식 | 도달 가능한 모든 상태에서 검사 |
| NoStarvation | 시간적 (liveness) | 모든 가능한 무한 동작에서 검사 |

어떤 속성이 위반되면, TLC는 **반례(counterexample)**를 출력합니다: 위반으로 이어지는 특정 상태 시퀀스입니다.

---

## 5. PlusCal: TLA+를 위한 알고리즘 언어

### 5.1 왜 PlusCal인가?

TLA+는 수학적 논리입니다 — 강력하지만 프로그래머에게 낯섭니다. PlusCal은 `pcal` 번역기에 의해 자동으로 TLA+로 변환되는 의사코드 같은 구문을 제공합니다.

```
PlusCal code                    Translated TLA+
(readable pseudocode)    ────▶  (mathematical spec)
                                     │
                                     ▼
                                TLC model checker
```

### 5.2 PlusCal 구문

```tla
---- MODULE PetersonPlusCal ----
EXTENDS Integers

(*--algorithm Peterson
variables
    flag = [i \in {0, 1} |-> FALSE],
    turn = 0;

process proc \in {0, 1}
begin
    P1: \* Label: want to enter
        flag[self] := TRUE;
    P2: \* Give priority to other
        turn := 1 - self;
    P3: \* Wait until safe to enter
        await flag[1 - self] = FALSE \/ turn = self;
    CS: \* Critical section
        skip; \* (do critical work here)
    P4: \* Exit critical section
        flag[self] := FALSE;
    P5: \* Return to beginning
        goto P1;
end process;

end algorithm; *)
====
```

### 5.3 PlusCal 핵심 기능

**레이블**: 원자성의 세분도를 정의합니다. 두 레이블 사이의 모든 것은 원자적으로 실행됩니다. 이는 동시성을 올바르게 모델링하는 데 매우 중요합니다.

```
\* WRONG: too much atomicity
begin
    FullyCritical:   \* Everything is one atomic step — no concurrency!
        flag[self] := TRUE;
        turn := 1 - self;
        await ...;
        skip;
        flag[self] := FALSE;

\* RIGHT: each step is a separate atomic action
begin
    SetFlag:    flag[self] := TRUE;
    SetTurn:    turn := 1 - self;
    Wait:       await flag[1-self] = FALSE \/ turn = self;
    CritSec:    skip;
    ClearFlag:  flag[self] := FALSE;
```

**await**: 조건이 참이 될 때까지 차단합니다. 바쁜 대기와 달리, 조건이 유지되는 상태만 탐색하도록 TLC에 알려줍니다.

**with**: 비결정적 선택.

```
\* Choose a random element from a set
with node \in Nodes do
    Send(node, message);
end with;
```

**either/or**: 비결정적 분기.

```
either
    \* Path 1: message is delivered
    Deliver(msg);
or
    \* Path 2: message is lost
    skip;
or
    \* Path 3: message is delayed
    Enqueue(msg, delayed_queue);
end either;
```

### 5.4 비결정성 모델링

비결정성은 분산 시스템 모델링에 필수적입니다:
1. 메시지 전달 순서가 비결정적
2. 프로세스 스케줄링이 비결정적
3. 장애가 아무 때나 발생할 수 있음

PlusCal은 모든 가능한 비결정적 선택을 모델링하고, TLC는 그것들을 모두 탐색합니다:

```tla
(*--algorithm NonDetExample
variables msgs = {};

process sender = "S"
begin
    Send:
        msgs := msgs \union {<<"hello">>, <<"world">>};
end process;

process receiver = "R"
begin
    Recv:
        either
            \* Receive in order
            with m \in msgs do
                msgs := msgs \ {m};
            end with;
        or
            \* Network failure: lose a message
            with m \in msgs do
                msgs := msgs \ {m};
            end with;
        end either;
end process;

end algorithm; *)
```

---

## 6. TLC Model Checker

### 6.1 TLC 작동 원리

TLC는 **명시적 상태 모델 검사(explicit state model checking)**를 수행합니다: 도달 가능한 모든 상태와 모든 가능한 전이를 탐색하여 완전한 상태 그래프를 구성합니다.

```
Algorithm:
  1. Compute all initial states satisfying Init
  2. For each unexplored state:
     a. Check all invariants
     b. Compute all successor states via Next
     c. Add new states to the exploration queue
  3. For liveness: check temporal properties on the state graph
     (find cycles that violate liveness using Tarjan's algorithm)

BFS exploration ensures shortest counterexample.
```

### 6.2 상태 공간 크기

상태 수는 변수 수와 그 도메인에 따라 기하급수적으로 증가합니다:

```
Example: Mutual exclusion with 2 processes
  pc: 4 states each = 4² = 16
  turn: 2 states = 2
  flag: 2 states each = 2² = 4
  Total: 16 × 2 × 4 = 128 states (manageable)

Example: Raft with 3 nodes, 2 terms, log size 2
  state: 3 values × 3 nodes = 27
  currentTerm: 2 values × 3 nodes = 8
  votedFor: 4 values × 3 nodes = 64
  log: (entries)³ = hundreds
  Total: millions of states (still manageable for TLC)

Example: Raft with 5 nodes, 3 terms, log size 5
  Total: billions of states (may take hours/days)
```

### 6.3 TLC 설정

TLC는 `.cfg` 파일이나 TLA+ Toolbox를 통해 설정합니다:

```
\* ModelChecking.cfg

SPECIFICATION Spec

\* Set concrete values for CONSTANTS
CONSTANTS
    Procs = {0, 1}

\* Check these invariants at every state
INVARIANT TypeOK
INVARIANT MutualExclusion

\* Check these temporal properties
PROPERTY NoStarvation

\* Constraint: limit state space (optional)
CONSTRAINT
    turn \in 0..1

\* Symmetry: {0, 1} are interchangeable (reduces state space by 2x)
SYMMETRY Permutations(Procs)
```

### 6.4 대칭 집합

대칭 축소는 상태 공간을 극적으로 줄일 수 있습니다. 모델 값의 집합이 대칭적(교환 가능)이면, TLC는 하나의 대표 순서만 탐색하면 됩니다.

```
Without symmetry: nodes = {n1, n2, n3}
  State (n1=leader, n2=follower, n3=follower)
  State (n2=leader, n1=follower, n3=follower)
  State (n3=leader, n1=follower, n2=follower)
  → 3 distinct states

With symmetry: nodes is a symmetry set
  All three states are equivalent (one representative explored)
  → 1 state checked instead of 3

For n nodes: up to n! reduction factor
```

### 6.5 TLC 출력 해석

```
TLC output for a successful check:

  Model checking completed. No error has been found.
    Estimates of the probability that TLC did not check all
    reachable states because two distinct states had the
    same fingerprint:
      calculated: 1.2E-15

    2847 states generated, 1423 distinct states found, 0 states left on queue.
    The depth of the complete state graph search is 23.

TLC output for an invariant violation:

  Error: Invariant MutualExclusion is violated.

  Error: The following behavior constitutes a counterexample:

  State 1: <Initial predicate>
    /\ pc = [0 |-> "idle", 1 |-> "idle"]
    /\ flag = [0 |-> FALSE, 1 |-> FALSE]
    /\ turn = 0

  State 2: <Want(0)>
    /\ pc = [0 |-> "want", 1 |-> "idle"]
    /\ flag = [0 |-> TRUE, 1 |-> FALSE]
    /\ turn = 0

  State 3: ...
  (sequence continues to the violating state)
```

### 6.6 TLC 실용 팁

| 팁 | 근거 |
|-----|-----------|
| 작은 상수로 시작 | 스케일링 전에 명세 로직을 디버그 |
| TypeOK 불변식 추가 | 많은 버그를 일찍 잡음; TLC 속도도 높임 |
| 대칭 집합 사용 | 상태 공간을 n!만큼 줄일 수 있음 |
| 상태 제약 추가 | 무한 변수를 제한 (예: 로그 길이) |
| liveness 전에 불변식 검사 | 불변식 검사가 훨씬 빠름 |
| 여러 worker 사용 | TLC가 멀티스레드 탐색을 지원 |
| `-coverage`로 프로파일링 | 어떤 액션이 얼마나 자주 수행되는지 보여줌 |

---

## 7. 사례 연구: 단순화된 Raft Leader Election의 TLA+ 명세

### 7.1 명세 대상

다음을 포함하는 Raft leader election의 단순화된 버전입니다:
- N개의 서버, 각각 {Follower, Candidate, Leader} 상태
- Term (election을 위한 논리적 클럭)
- 투표 요청과 부여
- Safety 속성: term당 최대 하나의 leader

로그 복제는 생략하여 election 메커니즘에 집중합니다.

### 7.2 전체 TLA+ 명세

```tla
---- MODULE RaftLeaderElection ----
\*
\* Simplified Raft Leader Election
\*
\* Specifies the election protocol from the Raft paper.
\* Safety property: at most one leader per term.
\* Liveness property: a leader is eventually elected (with fairness).
\*

EXTENDS Integers, FiniteSets

CONSTANTS
    Server,       \* Set of server IDs, e.g., {s1, s2, s3}
    MaxTerm       \* Maximum term number (bounds the state space)

VARIABLES
    currentTerm,  \* currentTerm[s]: latest term server s has seen
    votedFor,     \* votedFor[s]: candidate that server s voted for in current term (or Nil)
    state,        \* state[s]: Follower, Candidate, or Leader
    votesGranted  \* votesGranted[s]: set of servers that granted vote to s

vars == <<currentTerm, votedFor, state, votesGranted>>

Nil == "nil"

\* ----- Helper Definitions -----

\* A quorum is a majority of servers
Quorum == {Q \in SUBSET Server : Cardinality(Q) * 2 > Cardinality(Server)}

\* ----- Initial State -----

Init ==
    /\ currentTerm = [s \in Server |-> 0]
    /\ votedFor    = [s \in Server |-> Nil]
    /\ state       = [s \in Server |-> "Follower"]
    /\ votesGranted = [s \in Server |-> {}]

\* ----- Actions -----

\* Server s times out and starts an election
Timeout(s) ==
    /\ state[s] \in {"Follower", "Candidate"}
    /\ currentTerm[s] < MaxTerm        \* Bound for model checking
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[s] + 1]
    /\ votedFor'    = [votedFor EXCEPT ![s] = s]   \* Vote for self
    /\ state'       = [state EXCEPT ![s] = "Candidate"]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = {s}]  \* Self-vote

\* Server s (candidate) requests vote from server t
\* Server t grants vote if:
\*   1. t's term <= s's term
\*   2. t hasn't voted in s's term (or already voted for s)
RequestVote(s, t) ==
    /\ state[s] = "Candidate"
    /\ s # t
    /\ currentTerm[s] >= currentTerm[t]
    /\ \/ votedFor[t] = Nil
       \/ votedFor[t] = s
       \/ currentTerm[s] > currentTerm[t]
    \* Update t's state
    /\ currentTerm' = [currentTerm EXCEPT ![t] = currentTerm[s]]
    /\ votedFor'    = [votedFor EXCEPT ![t] = s]
    /\ votesGranted' = [votesGranted EXCEPT ![s] = votesGranted[s] \union {t}]
    /\ state'       = [state EXCEPT ![t] = IF currentTerm[s] > currentTerm[t]
                                            THEN "Follower"
                                            ELSE state[t]]

\* Server s becomes leader (has received votes from a quorum)
BecomeLeader(s) ==
    /\ state[s] = "Candidate"
    /\ votesGranted[s] \in Quorum
    /\ state' = [state EXCEPT ![s] = "Leader"]
    /\ UNCHANGED <<currentTerm, votedFor, votesGranted>>

\* Server s discovers a higher term and steps down
StepDown(s, t) ==
    /\ currentTerm[t] > currentTerm[s]
    /\ state[s] \in {"Candidate", "Leader"}
    /\ currentTerm' = [currentTerm EXCEPT ![s] = currentTerm[t]]
    /\ state'       = [state EXCEPT ![s] = "Follower"]
    /\ votedFor'    = [votedFor EXCEPT ![s] = Nil]
    /\ UNCHANGED votesGranted

\* ----- Next State Relation -----

Next ==
    \/ \E s \in Server : Timeout(s)
    \/ \E s, t \in Server : RequestVote(s, t)
    \/ \E s \in Server : BecomeLeader(s)
    \/ \E s, t \in Server : StepDown(s, t)

\* ----- Specification -----

Spec == Init /\ [][Next]_vars
        /\ \A s \in Server : WF_vars(BecomeLeader(s))

\* ----- Safety Properties -----

\* At most one leader per term
AtMostOneLeaderPerTerm ==
    \A s, t \in Server :
        (s # t /\ state[s] = "Leader" /\ state[t] = "Leader")
        => currentTerm[s] # currentTerm[t]

\* Type invariant
TypeOK ==
    /\ currentTerm \in [Server -> 0..MaxTerm]
    /\ votedFor \in [Server -> Server \union {Nil}]
    /\ state \in [Server -> {"Follower", "Candidate", "Leader"}]
    /\ votesGranted \in [Server -> SUBSET Server]

\* A server's votesGranted only contains servers that voted for it
VotesCorrect ==
    \A s \in Server :
        \A t \in votesGranted[s] :
            votedFor[t] = s \/ currentTerm[t] # currentTerm[s]

\* ----- Liveness Properties -----

\* A leader is eventually elected (requires fairness)
EventuallyLeader ==
    <>(\E s \in Server : state[s] = "Leader")

====
```

### 7.3 명세 워크스루

**상태 변수와 역할**:

```
currentTerm[s]:
  ┌─────┐ Timeout  ┌─────┐ Timeout  ┌─────┐
  │  0  │─────────▶│  1  │─────────▶│  2  │──▶ ...
  └─────┘          └─────┘          └─────┘
  Monotonically increasing. Never decreases.

votedFor[s]:
  In each term, a server votes for at most one candidate.
  Reset to Nil when entering a new term via StepDown.

state[s]:
  Follower ──Timeout──▶ Candidate ──BecomeLeader──▶ Leader
     ▲                      │                          │
     │                      │ StepDown                 │ StepDown
     └──────────────────────┘──────────────────────────┘

votesGranted[s]:
  Set of servers that voted for s in the current election.
  Reset when s starts a new election (Timeout).
```

**핵심 불변식: AtMostOneLeaderPerTerm**:

이것은 Raft의 가장 중요한 safety 속성입니다. 증명 스케치는 다음과 같습니다:

```
1. A server becomes leader only after receiving votes from a quorum
2. Each server votes for at most one candidate per term (votedFor)
3. Any two quorums must overlap (majority property)
4. Therefore, two candidates in the same term cannot both get quorums
5. Therefore, at most one leader per term □
```

### 7.4 이 명세에 대한 TLC 실행

```
Configuration:
  Server = {s1, s2, s3}   (3-node cluster)
  MaxTerm = 3              (bound term space)

TLC results:
  States generated: ~45,000
  Distinct states: ~12,000
  Time: ~2 seconds
  Properties checked: TypeOK ✓, AtMostOneLeaderPerTerm ✓, EventuallyLeader ✓

With Server = {s1, s2, s3, s4, s5} and MaxTerm = 2:
  States generated: ~2,000,000
  Distinct states: ~400,000
  Time: ~30 seconds
  Properties checked: All pass ✓
```

---

## 8. TLA+로 합의 프로토콜 모델링하기

### 8.1 TLA+로 표현한 Two-Phase Commit

Two-Phase Commit (2PC)는 모델링하기 가장 간단한 합의 유사 프로토콜입니다:

```tla
---- MODULE TwoPhaseCommit ----
EXTENDS Integers, FiniteSets

CONSTANTS Participants

VARIABLES
    tmState,        \* 트랜잭션 관리자 상태: "init", "committed", "aborted"
    tmPrepared,     \* "prepared"로 응답한 참가자 집합
    pmState         \* 함수: 참가자 -> "working", "prepared", "committed", "aborted"

vars == <<tmState, tmPrepared, pmState>>

Init ==
    /\ tmState = "init"
    /\ tmPrepared = {}
    /\ pmState = [p \in Participants |-> "working"]

\* --- 참가자 액션 ---

\* 참가자가 prepare에 투표
Prepare(p) ==
    /\ pmState[p] = "working"
    /\ pmState' = [pmState EXCEPT ![p] = "prepared"]
    /\ UNCHANGED <<tmState, tmPrepared>>

\* 참가자가 자발적으로 중단 (장애/타임아웃 시뮬레이션)
ParticipantAbort(p) ==
    /\ pmState[p] = "working"
    /\ pmState' = [pmState EXCEPT ![p] = "aborted"]
    /\ UNCHANGED <<tmState, tmPrepared>>

\* --- 트랜잭션 관리자 액션 ---

\* TM이 prepare 투표를 수신
ReceivePrepare(p) ==
    /\ tmState = "init"
    /\ pmState[p] = "prepared"
    /\ tmPrepared' = tmPrepared \cup {p}
    /\ UNCHANGED <<tmState, pmState>>

\* TM이 커밋을 결정 (모든 참가자가 prepared)
TMCommit ==
    /\ tmState = "init"
    /\ tmPrepared = Participants
    /\ tmState' = "committed"
    /\ pmState' = [p \in Participants |-> "committed"]
    /\ UNCHANGED tmPrepared

\* TM이 중단을 결정
TMAbort ==
    /\ tmState = "init"
    /\ tmState' = "aborted"
    /\ pmState' = [p \in Participants |->
                    IF pmState[p] = "working" THEN "aborted"
                    ELSE pmState[p]]
    /\ UNCHANGED tmPrepared

Next ==
    \/ \E p \in Participants : Prepare(p)
    \/ \E p \in Participants : ParticipantAbort(p)
    \/ \E p \in Participants : ReceivePrepare(p)
    \/ TMCommit
    \/ TMAbort

\* --- 속성 ---

\* Safety: 어떤 참가자가 커밋했으면, 어떤 참가자도 중단하지 않음 (역도 마찬가지)
Consistency ==
    \A p1, p2 \in Participants :
        ~ (pmState[p1] = "committed" /\ pmState[p2] = "aborted")

Spec == Init /\ [][Next]_vars

====
```

### 8.2 TLA+로 표현한 단일 결정 Paxos

Paxos를 모델링하려면 프로토콜의 메시지 전달 특성을 포착해야 합니다:

```tla
---- MODULE Paxos ----
EXTENDS Integers, FiniteSets

CONSTANTS Acceptors, Values, Quorums, MaxBallot

ASSUME QuorumAssumption ==
    /\ \A Q \in Quorums : Q \subseteq Acceptors
    /\ \A Q1, Q2 \in Quorums : Q1 \cap Q2 /= {}

VARIABLES
    maxBal,      \* maxBal[a]: acceptor a가 본 가장 높은 ballot
    maxVBal,     \* maxVBal[a]: 가장 높게 수락된 제안의 ballot
    maxVal,      \* maxVal[a]: 가장 높게 수락된 제안의 값
    msgs         \* 전송된 모든 메시지의 집합

vars == <<maxBal, maxVBal, maxVal, msgs>>

Init ==
    /\ maxBal = [a \in Acceptors |-> -1]
    /\ maxVBal = [a \in Acceptors |-> -1]
    /\ maxVal = [a \in Acceptors |-> "none"]
    /\ msgs = {}

\* Phase 1a: 제안자가 Prepare(ballot) 전송
Phase1a(b) ==
    /\ b <= MaxBallot
    /\ msgs' = msgs \cup {[type |-> "1a", bal |-> b]}
    /\ UNCHANGED <<maxBal, maxVBal, maxVal>>

\* Phase 1b: Acceptor가 Promise로 응답
Phase1b(a) ==
    /\ \E m \in msgs :
        /\ m.type = "1a"
        /\ m.bal > maxBal[a]
        /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
        /\ msgs' = msgs \cup {[type |-> "1b",
                                bal |-> m.bal,
                                acc |-> a,
                                mbal |-> maxVBal[a],
                                mval |-> maxVal[a]]}
    /\ UNCHANGED <<maxVBal, maxVal>>

\* Phase 2a: 제안자가 선택된 값으로 Accept 전송
Phase2a(b, v) ==
    /\ \E Q \in Quorums :
        LET promises == {m \in msgs : m.type = "1b" /\ m.bal = b}
            promisers == {m.acc : m \in promises}
        IN
        /\ Q \subseteq promisers
        /\ \/ \A m \in promises : m.mbal = -1    \* 이전에 수락된 값 없음
              /\ v \in Values                      \* 임의의 값 선택 가능
           \/ LET maxPromise ==                    \* 가장 높게 수락된 값 사용
                    CHOOSE m \in promises :
                        \A m2 \in promises : m.mbal >= m2.mbal
              IN v = maxPromise.mval
    /\ msgs' = msgs \cup {[type |-> "2a", bal |-> b, val |-> v]}
    /\ UNCHANGED <<maxBal, maxVBal, maxVal>>

\* Phase 2b: Acceptor가 제안을 수락
Phase2b(a) ==
    /\ \E m \in msgs :
        /\ m.type = "2a"
        /\ m.bal >= maxBal[a]
        /\ maxBal' = [maxBal EXCEPT ![a] = m.bal]
        /\ maxVBal' = [maxVBal EXCEPT ![a] = m.bal]
        /\ maxVal' = [maxVal EXCEPT ![a] = m.val]
        /\ msgs' = msgs \cup {[type |-> "2b",
                                bal |-> m.bal,
                                acc |-> a,
                                val |-> m.val]}

Next ==
    \/ \E b \in 0..MaxBallot : Phase1a(b)
    \/ \E a \in Acceptors : Phase1b(a)
    \/ \E b \in 0..MaxBallot, v \in Values : Phase2a(b, v)
    \/ \E a \in Acceptors : Phase2b(a)

\* --- Safety: 합의 ---
\* 값이 선택(쿼럼에 의해 수락)되면, 다른 값은 선택되지 않음
ChosenValues ==
    {v \in Values :
        \E Q \in Quorums :
            \A a \in Q : \E m \in msgs :
                /\ m.type = "2b"
                /\ m.acc = a
                /\ m.val = v}

Agreement == Cardinality(ChosenValues) <= 1

Spec == Init /\ [][Next]_vars

====
```

### 8.3 TLA+로 표현한 Raft 합의 핵심

Raft 모델링의 핵심 도전은 **네트워크 비결정성**으로 **로그 복제**를 포착하는 것입니다:

```tla
---- MODULE RaftConsensus ----
\* 핵심 safety 속성에 집중한 단순화된 Raft
EXTENDS Integers, Sequences, FiniteSets

CONSTANTS Nodes, Values, MaxTerm, MaxLogLen

VARIABLES
    currentTerm,    \* [Nodes -> Nat]
    state,          \* [Nodes -> {"follower", "candidate", "leader"}]
    votedFor,       \* [Nodes -> Nodes \cup {"none"}]
    log,            \* [Nodes -> Seq([term: Nat, val: Values])]
    commitIndex,    \* [Nodes -> Nat]
    messages        \* 전송 중인 메시지 집합

vars == <<currentTerm, state, votedFor, log, commitIndex, messages>>

\* --- 핵심 Safety 속성 ---
\* 선출 안전성: term당 최대 하나의 리더
ElectionSafety ==
    \A n1, n2 \in Nodes :
        /\ state[n1] = "leader"
        /\ state[n2] = "leader"
        /\ currentTerm[n1] = currentTerm[n2]
        => n1 = n2

\* 상태 기계 안전성: 노드가 인덱스 i에서 항목을 적용했으면,
\* 다른 노드는 인덱스 i에서 다른 항목을 적용하지 않음
StateMachineSafety ==
    \A n1, n2 \in Nodes :
        \A i \in 1..Min(commitIndex[n1], commitIndex[n2]) :
            log[n1][i] = log[n2][i]

====
```

### 8.4 네트워크 비결정성 모델링

분산 시스템 모델링의 핵심적 측면은 네트워크가 오작동할 수 있는 모든 방식을 포착하는 것입니다:

```tla
\* --- 네트워크 모델 ---

\* 메시지 전달 (비결정적 선택)
DeliverMessage ==
    /\ \E m \in messages :
        /\ HandleMessage(m)           \* 메시지 처리
        /\ messages' = messages \ {m}  \* 네트워크에서 제거

\* 메시지 삭제 (패킷 손실 모델링)
DropMessage ==
    /\ \E m \in messages :
        /\ messages' = messages \ {m}
        /\ UNCHANGED <<currentTerm, state, votedFor, log, commitIndex>>

\* 네트워크 파티션: 모든 메시지에 DropMessage를 허용함으로써 암시적으로 모델링
\* 메시지 순서 변경: 집합(순서 없음)에서 선택하므로 자동으로 처리됨
```

주요 명세의 상태 공간 크기:

```
명세                          | 노드 | 상태 수    | 시간
──────────────────────────────┼──────┼────────────┼──────────
Two-Phase Commit              | 3    | 12,408     | <1초
Two-Phase Commit              | 5    | 524,288    | 3초
Paxos (단일 결정)              | 3    | 1.2M       | 8초
Paxos (단일 결정)              | 5    | 847M       | 4시간
Raft Leader Election          | 3    | 58,904     | 2초
Raft (전체, 로그 2개)          | 3    | 38M        | 15분
Raft (전체, 로그 3개)          | 5    | >10B       | 수일
```

---

## 9. AWS의 TLA+ 활용

### 9.1 개요

Amazon Web Services는 2011년부터 핵심 인프라 구성 요소의 정확성을 검증하기 위해 TLA+를 사용해 왔습니다. 이 관행은 Chris Newcombe가 도입했으며 2014년 논문에 문서화되었습니다.

### 9.2 TLA+로 검증된 시스템

| AWS 서비스 | 구성 요소 | 버그 발견 | 영향 |
|-------------|-----------|------------|--------|
| **S3** | 객체 스토리지 장애 허용 | 예 | 데이터 손실 버그 방지 |
| **DynamoDB** | 복제 및 장애 복구 | 예 | 장애 복구의 미묘한 경쟁 조건 발견 |
| **EBS** | 블록 스토리지 복제 | 예 | 데이터 손상 경로 발견 |
| **내부 잠금 관리자** | 분산 잠금 프로토콜 | 예 | 여러 에지 케이스 |

### 9.3 사례 연구: DynamoDB 복제 버그

Newcombe 등의 논문에서:

```
The Bug:
  - DynamoDB replication protocol had a subtle race condition
  - During failover, if two events occurred in a specific order:
    1. Primary fails
    2. Secondary receives a late replication message
  - A small window existed where data could be lost

  This bug was NEVER found by:
    - Code review (multiple engineers reviewed)
    - Unit tests
    - Integration tests
    - Load tests running for months
    - Production traffic

  TLA+ found it in minutes.

Why testing missed it:
  - Required exact interleaving of 3 events across 3 nodes
  - Probability of occurring in any single test run: ~10^-6
  - Would take centuries of random testing to find reliably
```

### 9.4 TLA+가 AWS 개발에 적합한 방법

```
AWS Engineering Workflow:

1. Design Phase
   └── Engineer writes informal design document
   └── Engineer writes TLA+ specification
   └── Run TLC to find design bugs ◀── CATCHES MOST BUGS
   └── Fix spec, re-run TLC
   └── Iterate until all properties pass

2. Implementation Phase
   └── Implement based on verified design
   └── Write unit and integration tests
   └── Reference spec for edge cases

3. Review Phase
   └── Code review includes spec review
   └── Reviewers check spec matches implementation
   └── Any design change → update spec first, re-verify

4. Maintenance Phase
   └── Spec serves as precise documentation
   └── New engineers read spec to understand design
   └── Design changes start with spec modification
```

### 9.5 AWS 경험으로부터의 교훈

Chris Newcombe의 보고서에서:

| 교훈 | 세부 사항 |
|--------|--------|
| 명세는 작음 | TLA+ 명세는 일반적으로 200-800줄 (vs 코드 100K+ 줄) |
| 엔지니어가 2-3주에 TLA+를 배움 | 대부분의 AWS 엔지니어가 접근 가능하다고 봄 |
| ROI가 매우 높음 | 하나의 명세가 2일 만에 수개월의 테스트가 놓친 2개의 버그를 발견 |
| 명세가 설계를 개선함 | 명세를 작성하면 종종 설계가 단순화됨 |
| 명세는 문서화임 | 복잡한 프로토콜에 대한 최고의 문서화 |
| 모든 코드에 적용하지 않음 | 복잡하고 동시적이며 장애 허용적인 구성 요소에만 가치가 있음 |

### 9.6 TLA+가 하지 않는 것

이해해야 할 중요한 한계:

```
TLA+ verifies:                    TLA+ does NOT verify:
  ✓ Protocol design                 ✗ Implementation correctness
  ✓ Safety properties               ✗ Performance characteristics
  ✓ Liveness properties             ✗ Memory safety
  ✓ Edge cases in state machines    ✗ Security properties
  ✓ Concurrent interactions         ✗ Actual code (only the model)

The gap between spec and code:
  TLA+ spec says: "in action A, variable x gets value v"
  Code might have: off-by-one error implementing A
  TLA+ won't catch that.

Solution: use property-based testing to bridge the gap
  - Generate random test inputs
  - Check that code behavior matches spec behavior
  - E.g., Jepsen checks real databases against consistency specs
```

---

## 10. 실용적 검증 워크플로우

### 10.1 Amazon의 TLA+ 경험

Amazon Web Services는 2014년 랜드마크 논문("Use of Formal Methods at Amazon Web Services", Newcombe 외)을 발표했습니다:

```
AWS에서 TLA+를 채택한 팀:
┌────────────────────┬──────────────────────────────────────────┐
│ 서비스              │ 검증한 내용                               │
├────────────────────┼──────────────────────────────────────────┤
│ S3                 │ 내결함성 복제 프로토콜                      │
│ DynamoDB           │ 복제 및 그룹 멤버십                        │
│ EBS                │ 볼륨 관리 상태 기계                        │
│ 내부 Lock 서비스     │ 분산 잠금 정확성                           │
│ CloudFormation     │ 스택 업데이트 오케스트레이션                  │
└────────────────────┴──────────────────────────────────────────┘

주요 발견:
  - TLA+가 5개 시스템에서 7개의 심각한 버그를 발견
  - 2개의 버그는 이미 프로덕션에 있는 시스템에서 발견
  - 모든 버그는 기존 테스트가 놓친 미묘한 이벤트 조합과 관련
  - 이전에 형식 방법론 경험이 없는 엔지니어들이 TLA+를 2-3주 만에 학습
  - 일반적인 명세: 500-2000줄의 TLA+
  - 평균 검증 시간: 모델 검사에 1-5분
```

### 10.2 형식 검증을 사용해야 할 때

모든 시스템에 형식 검증이 필요한 것은 아닙니다. 다음과 같은 경우에 사용합니다:

```
높은 가치                           낮은 가치
─────────────────────────────────────────────────────
✓ 합의 프로토콜                     ✗ CRUD 애플리케이션
✓ 분산 잠금 관리자                   ✗ 단일 노드 프로그램
✓ 복제 상태 기계                    ✗ 상태 없는 서비스
✓ 멤버십/재구성                     ✗ 단순 요청-응답
✓ 트랜잭션 프로토콜                  ✗ UI 로직
✓ 캐시 일관성 프로토콜               ✗ 배치 처리

결정 기준:
  1. 알고리즘이 동시적이거나 분산적인가? → 예: TLA+ 고려
  2. safety-critical 불변식이 있는가? → 예: 강력히 고려
  3. 상태 공간이 테스트로 감당하기엔 너무 큰가? → 예: 형식 검증
  4. 프로토콜이 프로덕션에서 사용될 것인가? → 예: 강력히 고려
  5. 팀이 2-4주를 투자할 의향이 있는가? → 예: 실행
```

### 10.3 명세 주도 개발 워크플로우

```
단계 1: 영어 산문 설명 작성
  └── "노드들이 과반수 투표로 리더를 선출한다.
       노드는 term당 최대 하나의 후보에게 투표한다.
       과반수를 얻은 후보가 리더가 된다."

단계 2: 상태 변수 식별
  └── currentTerm, state, votedFor, votesGranted

단계 3: Safety 속성을 먼저 작성
  └── OneLeaderPerTerm, TypeOK

단계 4: 명세 작성
  └── Init, 액션 (StartElection, GrantVote, BecomeLeader), Next

단계 5: TLC 실행 — 실패를 예상
  └── 명세의 버그 수정 (가장 흔한 것: 누락된 UNCHANGED,
       잘못된 액션 가드, 불충분하게 지정된 타입)

단계 6: 깨끗해질 때까지 반복
  └── 모든 불변식 통과, 데드락 없음

단계 7: Liveness 속성 추가
  └── EventualLeader, ElectionProgress + 공정성 조건

단계 8: 명세를 코드로 변환
  └── 명세를 구현을 위한 설계 문서로 사용
       각 액션 → 하나의 함수/메서드
       각 변수 → 하나의 데이터 구조

단계 9: 코드를 명세 동작에 대해 테스트
  └── 추적 검증: 코드를 실행하고 상태 전이를 기록하여
       명세의 허용된 동작과 일치하는지 검증
```

### 10.4 공통 명세 패턴

많은 분산 시스템 명세에서 나타나는 패턴:

```tla
\* --- 패턴 1: 네트워크 집합을 이용한 메시지 전달 ---
VARIABLE messages
Send(m) == messages' = messages \cup {m}
Receive(m) == /\ m \in messages
              /\ messages' = messages \ {m}

\* --- 패턴 2: 쿼럼 교차 ---
CONSTANTS Quorums
ASSUME \A Q1, Q2 \in Quorums : Q1 \cap Q2 /= {}

\* --- 패턴 3: Epoch/term 기반 추론 ---
StepDown(n, term) ==
    /\ term > currentTerm[n]
    /\ currentTerm' = [currentTerm EXCEPT ![n] = term]
    /\ state' = [state EXCEPT ![n] = "follower"]
    /\ votedFor' = [votedFor EXCEPT ![n] = "none"]

\* --- 패턴 4: 비결정적 장애 ---
NodeCrash(n) ==
    /\ state' = [state EXCEPT ![n] = "crashed"]
    /\ UNCHANGED <<persistent_vars>>  \* 영속적 상태는 유지

NodeRecover(n) ==
    /\ state[n] = "crashed"
    /\ state' = [state EXCEPT ![n] = "follower"]
    /\ UNCHANGED <<persistent_vars>>

\* --- 패턴 5: 유계 모델 검사를 위한 상태 제약 ---
StateConstraint ==
    /\ \A n \in Nodes : currentTerm[n] <= MaxTerm
    /\ Cardinality(messages) <= MaxMessages
    /\ \A n \in Nodes : Len(log[n]) <= MaxLogLen
```

### 10.5 검증의 함정과 모범 사례

명세 작성 시 흔한 실수:

```
함정 1: UNCHANGED 누락
  잘못됨:  Action(n) == /\ x' = x + 1
  올바름:  Action(n) == /\ x' = x + 1
                       /\ UNCHANGED <<y, z>>
  TLC 오류: "Variable y has no next-state value"

함정 2: 모델의 과도한 제약
  잘못됨:  메시지가 FIFO 순서로 전달됨
  올바름:  메시지가 어떤 순서로든 전달될 수 있음
  이유:    실제 네트워크는 패킷 순서를 바꿈. 명세는 프로토콜이
          작동해야 하는 가장 약한 가정을 모델링해야 함.

함정 3: 타입의 불충분한 제약
  잘못됨:  VARIABLE x  (타입 제약 없음)
  올바름:  TypeOK == x \in 0..MaxVal
  이유:    TLC가 무한 값으로 상태를 탐색하여 종료하지 않을 수 있음.

함정 4: 구현 세부사항 모델링
  잘못됨:  TCP 핸드셰이크, 직렬화 형식, 타임아웃 명세
  올바름:  메시지 전달, 비결정적 장애로 추상화
  이유:    명세는 구현이 아닌 프로토콜을 모델링해야 함.

함정 5: liveness를 위한 공정성 무시
  잘못됨:  Spec == Init /\ [][Next]_vars  (liveness 검사 시)
  올바름:  Spec == Init /\ [][Next]_vars /\ Fairness
  이유:    공정성 없이는 TLC가 활성화된 액션이 절대 실행되지 않는
          사소한 반례를 찾음.
```

### 10.6 명세에서 프로덕션까지

TLA+를 엔지니어링 팀에 통합하기 위한 실용적 워크플로우:

```
단계 1: 학습 (1-2주)
  ├── Lamport의 "Specifying Systems" 1-6장 학습
  ├── 잘 알려진 알고리즘(mutex, producer-consumer)의 명세 작성
  └── TLA+ Toolbox IDE를 사용하여 즉각적 피드백 획득

단계 2: 명세 (프로토콜당 1-2주)
  ├── 영어 설계 문서에서 시작
  ├── TypeOK 불변식을 먼저 작성 (대부분의 변수 오류를 포착)
  ├── 액션을 하나씩 추가하며 각 추가 후 검사
  └── 빠른 반복을 위해 작은 상수 사용 (3노드, 2값)

단계 3: 검증 (수일에서 수주)
  ├── 작은 상수로 TLC 실행, 모든 위반 수정
  ├── 확신을 쌓기 위해 점진적으로 상수 증가
  ├── safety 속성이 통과한 후 liveness 속성 추가
  └── 모든 TLC 반례와 적용한 수정 사항을 문서화

단계 4: 구현 (진행 중)
  ├── 명세를 권위 있는 설계 문서로 사용
  ├── 각 TLA+ 액션을 코드 함수에 매핑
  ├── 코드 추적을 명세 동작에 대해 검증하는 테스트 작성
  └── 프로토콜이 변경되면 명세를 재검증

단계 5: 유지보수 (진행 중)
  ├── 명세를 코드와 같은 저장소에 보관
  ├── 프로토콜을 변경하기 전에 명세를 먼저 업데이트
  ├── CI에서 TLC를 실행하여 회귀 검사
  └── 새 팀원 온보딩 시 명세를 먼저 읽도록 안내
```

### 10.7 도구 생태계

```
┌──────────────────┬───────────────────────────────────────────┐
│ 도구              │ 목적                                      │
├──────────────────┼───────────────────────────────────────────┤
│ TLA+ Toolbox     │ 구문 강조, 모델 설정 및 TLC 통합을 갖춘 IDE │
├──────────────────┼───────────────────────────────────────────┤
│ TLC              │ 명시적 상태 모델 체커 (철저 또는 시뮬레이션)  │
├──────────────────┼───────────────────────────────────────────┤
│ TLAPS            │ TLA+ 증명 시스템 — 수학적 증명을 위한       │
│                  │ 대화형 정리 증명기                          │
├──────────────────┼───────────────────────────────────────────┤
│ PlusCal          │ TLA+로 컴파일되는 알고리즘 언어              │
├──────────────────┼───────────────────────────────────────────┤
│ Apalache         │ SMT 솔버를 사용하는 기호적 모델 체커        │
│                  │ — TLC보다 큰 상태 공간을 처리               │
├──────────────────┼───────────────────────────────────────────┤
│ TLA+ VSCode      │ TLA+ 편집 및 기본 TLC 통합을 위한          │
│                  │ VSCode 확장                                │
├──────────────────┼───────────────────────────────────────────┤
│ Alloy            │ 대안: SAT 기반 분석을 갖춘 관계형 모델링 언어 │
├──────────────────┼───────────────────────────────────────────┤
│ Ivy              │ 대안: 결정 가능한 논리를 갖춘               │
│                  │ 분산 프로토콜 검증 언어                     │
├──────────────────┼───────────────────────────────────────────┤
│ P language       │ 대안: 체계적 테스팅을 갖춘                  │
│                  │ Microsoft의 상태 기계 언어                  │
└──────────────────┴───────────────────────────────────────────┘
```

---

## 11. 기타 형식 방법론

### 11.1 형식 방법론 비교

| 도구 | 패러다임 | 자동화 | 학습 곡선 | 최적 용도 |
|------|----------|-----------|----------------|----------|
| **TLA+** | 시간 논리 | 모델 검사 (TLC) | 중간 | 분산 프로토콜, 동시성 시스템 |
| **Alloy** | 관계 논리 | SAT 기반 모델 탐색 | 낮음-중간 | 데이터 모델, API, 구조적 속성 |
| **Coq** | 의존 타입 | 대화형 정리 증명 | 매우 높음 | 수학적 증명, 검증된 컴파일러 |
| **Isabelle/HOL** | 고차 논리 | 대화형 정리 증명 | 높음 | 수학적 증명, OS 검증 |
| **Spin/Promela** | 프로세스 대수 | 모델 검사 | 중간 | 통신 프로토콜 |
| **CBMC** | C 의미론 | 유한 모델 검사 | 낮음 | C 코드 검증 |
| **Dafny** | Hoare 논리 | 자동화된 검증 | 중간 | 검증된 프로그램 |

### 11.2 Alloy

Alloy (Daniel Jackson, MIT)는 관계 논리와 SAT 풀이를 사용하여 반례를 찾습니다:

```alloy
// Example: simple file system
sig File {}
sig Directory {
    contents: set (File + Directory)
}
sig Root extends Directory {}

// No cycles in directory structure
fact NoCycles {
    no d: Directory | d in d.^contents
}

// Root contains everything (directly or indirectly)
fact RootContainsAll {
    all f: File | f in Root.^contents
}

// Check: can we have an empty root?
check NoEmptyRoot {
    some Root.contents
} for 5   // Check with up to 5 objects of each type
```

**Alloy vs TLA+**:
- Alloy: 구조적 속성(데이터 모델, 불변식)에 더 적합
- TLA+: 동작 속성(프로토콜, 동시성 시스템)에 더 적합

### 11.3 Coq

Coq는 대화형 정리 증명기입니다. Coq 커널이 기계적으로 검증하는 증명을 작성합니다. 모델 검사보다 훨씬 많은 노력이 필요하지만, 증명은 모든 매개변수 값에 대해 유효합니다 (작은 인스턴스뿐만 아니라).

```coq
(* Example: proving commutativity of addition *)
Theorem plus_comm : forall n m : nat, n + m = m + n.
Proof.
  intros n m.
  induction n.
  - simpl. rewrite <- plus_n_O. reflexivity.
  - simpl. rewrite IHn. rewrite plus_n_Sm. reflexivity.
Qed.
```

**주목할 만한 Coq 프로젝트**:
- **CompCert**: 완전히 검증된 C 컴파일러
- **sel4**: 검증된 마이크로커널 (Isabelle/HOL 사용, 유사한 접근)
- **Verdi**: Coq에서 검증된 분산 시스템을 작성하기 위한 프레임워크
- **IronFleet**: 검증된 분산 시스템 구현 (Dafny)

### 11.4 언제 어떤 것을 사용하나

```
Decision tree:

Is the property about structure (data model, API)?
  YES → Alloy
  NO → Continue

Is the property about concurrent/distributed behavior?
  YES → TLA+ or Spin
  NO → Continue

Do you need a proof for ALL input sizes (not just small ones)?
  YES → Coq/Isabelle/Dafny
  NO → TLA+ with model checking

Do you want to verify actual running code (not just a model)?
  YES → Dafny, CBMC, or runtime verification
  NO → TLA+
```

---

## 12. 구현: TLA+ 명세 검증기와 PlusCal 번역기

### 12.1 TLA+ 명세 구조 검증기

```python
"""
TLA+ Specification Structure Validator

Validates the structure of TLA+ specifications by checking for
required sections, matching delimiters, and common mistakes.

This is NOT a full TLA+ parser — it validates structural conventions
to catch common authoring errors before running TLC.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class Severity(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationIssue:
    """A single validation finding."""
    severity: Severity
    line: int
    message: str
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Complete validation result for a TLA+ spec."""
    filename: str
    issues: List[ValidationIssue] = field(default_factory=list)
    module_name: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(i.severity == Severity.ERROR for i in self.issues)

    def summary(self) -> str:
        errors = sum(1 for i in self.issues if i.severity == Severity.ERROR)
        warnings = sum(1 for i in self.issues if i.severity == Severity.WARNING)
        infos = sum(1 for i in self.issues if i.severity == Severity.INFO)
        return (
            f"Module: {self.module_name or '(unknown)'}\n"
            f"Variables: {', '.join(self.variables) or '(none)'}\n"
            f"Constants: {', '.join(self.constants) or '(none)'}\n"
            f"Actions: {', '.join(self.actions) or '(none)'}\n"
            f"Properties: {', '.join(self.properties) or '(none)'}\n"
            f"Issues: {errors} errors, {warnings} warnings, {infos} info"
        )


class TLAPlusValidator:
    """
    Validates TLA+ specification structure.

    Checks for:
      - Module header and footer
      - EXTENDS declarations
      - VARIABLES and CONSTANTS
      - Init and Next definitions
      - Matching delimiters
      - Common mistakes
    """

    def __init__(self):
        self._patterns = {
            "module_start": re.compile(
                r"^-{4,}\s*MODULE\s+(\w+)\s*-{4,}$"
            ),
            "module_end": re.compile(r"^={4,}$"),
            "extends": re.compile(r"^EXTENDS\s+(.+)$"),
            "variables": re.compile(r"^VARIABLES?\s+(.+)$"),
            "constants": re.compile(r"^CONSTANTS?\s+(.+)$"),
            "definition": re.compile(r"^(\w+)\s*=="),
            "action_def": re.compile(r"^(\w+)\s*\(.*\)\s*=="),
            "invariant_check": re.compile(
                r"(TypeOK|Invariant|Safety|Mutex|Mutual)"
            ),
            "temporal_prop": re.compile(
                r"(Liveness|Eventually|NoStarvation|Progress|Fairness)"
            ),
            "pluscal_start": re.compile(r"\(\*--algorithm"),
            "pluscal_end": re.compile(r"end\s+algorithm;\s*\*\)"),
        }

    def validate(self, spec_text: str, filename: str = "spec.tla") -> ValidationResult:
        """
        Validate a TLA+ specification.

        Args:
            spec_text: The complete TLA+ specification text
            filename: Name of the file (for reporting)

        Returns:
            ValidationResult with findings
        """
        result = ValidationResult(filename=filename)
        lines = spec_text.split("\n")

        self._check_module_structure(lines, result)
        self._check_variables_and_constants(lines, result)
        self._check_init_next(lines, result)
        self._check_delimiters(lines, result)
        self._extract_definitions(lines, result)
        self._check_common_mistakes(lines, result)
        self._check_pluscal(lines, result)

        return result

    def _check_module_structure(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for proper module header and footer."""
        found_start = False
        found_end = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            match = self._patterns["module_start"].match(stripped)
            if match:
                found_start = True
                result.module_name = match.group(1)

            if self._patterns["module_end"].match(stripped):
                found_end = True

        if not found_start:
            result.issues.append(ValidationIssue(
                Severity.ERROR, 0,
                "Missing module header. Expected: ---- MODULE Name ----",
                "Add '---- MODULE YourModuleName ----' at the top"
            ))

        if not found_end:
            result.issues.append(ValidationIssue(
                Severity.ERROR, len(lines),
                "Missing module footer. Expected: ====",
                "Add '====' at the end of the file"
            ))

    def _check_variables_and_constants(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for VARIABLES and CONSTANTS declarations."""
        found_vars = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            match = self._patterns["variables"].match(stripped)
            if match:
                found_vars = True
                var_str = match.group(1).rstrip(",")
                var_names = [v.strip().rstrip(",") for v in var_str.split(",")]
                result.variables.extend(
                    v for v in var_names if v and not v.startswith("\\*")
                )

            match = self._patterns["constants"].match(stripped)
            if match:
                const_str = match.group(1).rstrip(",")
                const_names = [c.strip().rstrip(",") for c in const_str.split(",")]
                result.constants.extend(
                    c for c in const_names if c and not c.startswith("\\*")
                )

        if not found_vars:
            result.issues.append(ValidationIssue(
                Severity.WARNING, 0,
                "No VARIABLES declaration found",
                "Add 'VARIABLES var1, var2, ...' to declare state variables"
            ))

    def _check_init_next(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for Init and Next definitions."""
        has_init = False
        has_next = False
        has_spec = False

        for i, line in enumerate(lines):
            stripped = line.strip()
            if re.match(r"^Init\s*==", stripped):
                has_init = True
            if re.match(r"^Next\s*==", stripped):
                has_next = True
            if re.match(r"^Spec\s*==", stripped):
                has_spec = True

        if not has_init:
            result.issues.append(ValidationIssue(
                Severity.ERROR, 0,
                "No Init definition found",
                "Define 'Init == ...' specifying the initial state predicate"
            ))

        if not has_next:
            result.issues.append(ValidationIssue(
                Severity.ERROR, 0,
                "No Next definition found",
                "Define 'Next == ...' specifying the next-state relation"
            ))

        if not has_spec:
            result.issues.append(ValidationIssue(
                Severity.WARNING, 0,
                "No Spec definition found",
                "Define 'Spec == Init /\\ [][Next]_vars' as the full specification"
            ))

    def _check_delimiters(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for matching delimiters."""
        stack = []
        openers = {"(": ")", "[": "]", "{": "}", "<<": ">>"}
        closers = {")", "]", "}", ">>"}

        for i, line in enumerate(lines):
            stripped = line.split("\\*")[0]

            j = 0
            while j < len(stripped):
                if stripped[j:j+2] == "<<":
                    stack.append(("<<", i + 1))
                    j += 2
                    continue
                if stripped[j:j+2] == ">>":
                    if stack and stack[-1][0] == "<<":
                        stack.pop()
                    else:
                        result.issues.append(ValidationIssue(
                            Severity.ERROR, i + 1,
                            f"Unmatched '>>' at line {i + 1}"
                        ))
                    j += 2
                    continue

                ch = stripped[j]
                if ch in ("(", "[", "{"):
                    stack.append((ch, i + 1))
                elif ch in closers:
                    expected_opener = {")": "(", "]": "[", "}": "{"}[ch]
                    if stack and stack[-1][0] == expected_opener:
                        stack.pop()
                    elif stack:
                        result.issues.append(ValidationIssue(
                            Severity.ERROR, i + 1,
                            f"Mismatched delimiter: expected "
                            f"'{openers[stack[-1][0]]}' but found '{ch}'"
                        ))
                    else:
                        result.issues.append(ValidationIssue(
                            Severity.ERROR, i + 1,
                            f"Unmatched closing delimiter '{ch}'"
                        ))
                j += 1

        for opener, line_num in stack:
            result.issues.append(ValidationIssue(
                Severity.ERROR, line_num,
                f"Unclosed delimiter '{opener}' opened at line {line_num}"
            ))

    def _extract_definitions(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Extract action and property definitions."""
        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("\\*"):
                continue

            match = self._patterns["action_def"].match(stripped)
            if match:
                name = match.group(1)
                if name not in ("Init", "Next", "Spec"):
                    result.actions.append(name)
                continue

            match = self._patterns["definition"].match(stripped)
            if match:
                name = match.group(1)
                if name in ("Init", "Next", "Spec", "vars"):
                    continue
                if self._patterns["invariant_check"].search(name):
                    result.properties.append(name)
                elif self._patterns["temporal_prop"].search(name):
                    result.properties.append(name)

    def _check_common_mistakes(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for common TLA+ authoring mistakes."""
        for i, line in enumerate(lines):
            stripped = line.strip()

            if re.match(r"^\w+\s*=[^=]", stripped):
                if not stripped.startswith("CONSTANT"):
                    result.issues.append(ValidationIssue(
                        Severity.WARNING, i + 1,
                        "Possible '=' instead of '==' in definition",
                        "TLA+ uses '==' for definitions and '=' for equality"
                    ))

            if "'" in stripped and "UNCHANGED" not in stripped:
                primed_vars = re.findall(r"(\w+)'", stripped)
                if primed_vars and len(primed_vars) < len(result.variables):
                    if "/\\" in stripped or "\\/" in stripped:
                        pass

            if "[]" in stripped and "<>" in stripped:
                if "[]<>" not in stripped and "<>[]" not in stripped:
                    result.issues.append(ValidationIssue(
                        Severity.INFO, i + 1,
                        "Line contains both [] and <> — verify the nesting is correct"
                    ))

    def _check_pluscal(
        self, lines: List[str], result: ValidationResult
    ) -> None:
        """Check for PlusCal algorithm blocks."""
        pcal_start = None
        pcal_end = None

        for i, line in enumerate(lines):
            if self._patterns["pluscal_start"].search(line):
                pcal_start = i + 1
            if self._patterns["pluscal_end"].search(line):
                pcal_end = i + 1

        if pcal_start and not pcal_end:
            result.issues.append(ValidationIssue(
                Severity.ERROR, pcal_start,
                "PlusCal algorithm started but never ended",
                "Add 'end algorithm; *)' to close the PlusCal block"
            ))
        elif not pcal_start and pcal_end:
            result.issues.append(ValidationIssue(
                Severity.ERROR, pcal_end,
                "PlusCal end found without matching start",
                "Add '(*--algorithm Name' before the algorithm body"
            ))
        elif pcal_start and pcal_end:
            result.issues.append(ValidationIssue(
                Severity.INFO, pcal_start,
                f"PlusCal algorithm found (lines {pcal_start}-{pcal_end}). "
                f"Run pcal translator before TLC."
            ))
```

### 12.2 PlusCal-to-Pseudocode 번역기

```python
"""
PlusCal to Pseudocode Translator

Translates PlusCal algorithm blocks into readable pseudocode.
Useful for documentation and design review by engineers
who are not familiar with TLA+/PlusCal syntax.
"""


class PlusCalToPseudocode:
    """
    Translates PlusCal algorithm syntax to readable pseudocode.

    Handles:
      - Variable declarations
      - Process declarations
      - Labels and statements
      - Control flow (if/then/else, while, either/or)
      - await → wait until
      - with → for each / choose
    """

    def __init__(self):
        self._indent_level = 0
        self._output_lines: List[str] = []

    def translate(self, pluscal_text: str) -> str:
        """
        Translate PlusCal text to pseudocode.

        Args:
            pluscal_text: PlusCal algorithm text (without TLA+ wrapper)

        Returns:
            Human-readable pseudocode string
        """
        self._indent_level = 0
        self._output_lines = []

        lines = pluscal_text.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            i = self._process_line(line, lines, i)
            i += 1

        return "\n".join(self._output_lines)

    def _emit(self, text: str) -> None:
        """Emit a line of pseudocode with current indentation."""
        indent = "    " * self._indent_level
        self._output_lines.append(f"{indent}{text}")

    def _process_line(self, line: str, all_lines: List[str], idx: int) -> int:
        """Process a single PlusCal line. Returns the (possibly advanced) index."""

        if not line or line.startswith("\\*"):
            return idx

        match = re.match(r"\(\*--algorithm\s+(\w+)", line)
        if match:
            self._emit(f"ALGORITHM {match.group(1)}")
            self._emit("=" * 40)
            return idx

        if line.startswith("end algorithm"):
            self._emit("=" * 40)
            self._emit("END ALGORITHM")
            return idx

        if line.startswith("variables"):
            self._emit("SHARED VARIABLES:")
            self._indent_level += 1
            return idx

        if line.startswith("variable "):
            var_decl = line[len("variable "):].rstrip(";").rstrip(",")
            self._emit(f"- {self._translate_expr(var_decl)}")
            return idx

        match = re.match(r"process\s+(\w+)\s*(?:\\in|=)\s*(.+)", line)
        if match:
            proc_name = match.group(1)
            proc_set = self._translate_expr(match.group(2))
            self._indent_level = 0
            self._emit("")
            self._emit(f"PROCESS {proc_name} in {proc_set}:")
            self._indent_level = 1
            return idx

        if line == "end process;":
            self._indent_level = 0
            self._emit("END PROCESS")
            return idx

        if line == "begin":
            return idx

        match = re.match(r"^(\w+):\s*(.*)", line)
        if match and not line.startswith("if") and not line.startswith("while"):
            label = match.group(1)
            rest = match.group(2).strip()
            self._emit(f"[{label}]")
            if rest:
                self._indent_level += 1
                self._emit(self._translate_statement(rest))
                self._indent_level -= 1
            return idx

        if line.startswith("if ") or line.startswith("if("):
            cond = line[3:].rstrip(" then").strip()
            self._emit(f"IF {self._translate_expr(cond)} THEN")
            self._indent_level += 1
            return idx

        if line.startswith("elsif ") or line.startswith("elif "):
            cond = line.split(" ", 1)[1].rstrip(" then").strip()
            self._indent_level -= 1
            self._emit(f"ELSE IF {self._translate_expr(cond)} THEN")
            self._indent_level += 1
            return idx

        if line == "else":
            self._indent_level -= 1
            self._emit("ELSE")
            self._indent_level += 1
            return idx

        if line == "end if;" or line == "end if":
            self._indent_level -= 1
            self._emit("END IF")
            return idx

        if line.startswith("while "):
            cond = line[6:].rstrip(" do").strip()
            self._emit(f"WHILE {self._translate_expr(cond)} DO")
            self._indent_level += 1
            return idx

        if line == "end while;" or line == "end while":
            self._indent_level -= 1
            self._emit("END WHILE")
            return idx

        if line == "either":
            self._emit("NON-DETERMINISTICALLY CHOOSE:")
            self._indent_level += 1
            self._emit("OPTION 1:")
            self._indent_level += 1
            return idx

        if line == "or":
            self._indent_level -= 1
            option_num = sum(
                1 for l in self._output_lines if "OPTION" in l
            ) + 1
            self._emit(f"OPTION {option_num}:")
            self._indent_level += 1
            return idx

        if line == "end either;" or line == "end either":
            self._indent_level -= 2
            self._emit("END CHOOSE")
            return idx

        self._emit(self._translate_statement(line))
        return idx

    def _translate_statement(self, stmt: str) -> str:
        """Translate a single PlusCal statement to pseudocode."""
        stmt = stmt.rstrip(";").strip()

        if stmt.startswith("await "):
            cond = stmt[6:]
            return f"WAIT UNTIL {self._translate_expr(cond)}"

        if ":=" in stmt:
            parts = stmt.split(":=", 1)
            lhs = parts[0].strip()
            rhs = self._translate_expr(parts[1].strip())
            return f"{lhs} <- {rhs}"

        if stmt == "skip":
            return "// do nothing"

        if stmt.startswith("goto "):
            return f"GOTO [{stmt[5:]}]"

        if stmt.startswith("assert "):
            return f"ASSERT {self._translate_expr(stmt[7:])}"

        if stmt.startswith("print "):
            return f"PRINT {stmt[6:]}"

        return stmt

    def _translate_expr(self, expr: str) -> str:
        """Translate TLA+ expression syntax to more readable form."""
        replacements = [
            (r"\\in", "in"),
            (r"\\/", "OR"),
            (r"/\\", "AND"),
            (r"\\union", "UNION"),
            (r"\\subseteq", "SUBSET OF"),
            (r"\\cup", "UNION"),
            (r"\\cap", "INTERSECT"),
            (r"~", "NOT"),
            (r"\\lnot", "NOT"),
            (r"\\E", "EXISTS"),
            (r"\\A", "FOR ALL"),
            (r"\\X", "CROSS"),
            (r"EXCEPT !\[", "with ["),
            (r"\|->", "=>"),
        ]

        result = expr
        for pattern, replacement in replacements:
            result = result.replace(pattern, replacement)

        return result.strip()


# --- Demo and Testing ---

def demo_validator():
    """Demonstrate the TLA+ spec validator."""
    spec = """
---- MODULE SimpleCounter ----
EXTENDS Integers

CONSTANTS MaxValue

VARIABLES counter

vars == <<counter>>

Init ==
    /\\ counter = 0

Increment ==
    /\\ counter < MaxValue
    /\\ counter' = counter + 1

Decrement ==
    /\\ counter > 0
    /\\ counter' = counter - 1

Next ==
    \\/ Increment
    \\/ Decrement

Spec == Init /\\ [][Next]_vars /\\ WF_vars(Next)

TypeOK == counter \\in 0..MaxValue

EventuallyMax == <>(counter = MaxValue)

====
"""
    validator = TLAPlusValidator()
    result = validator.validate(spec, "SimpleCounter.tla")

    print("=== TLA+ Spec Validation ===\n")
    print(result.summary())
    print()

    for issue in result.issues:
        print(f"  [{issue.severity.value}] Line {issue.line}: {issue.message}")
        if issue.suggestion:
            print(f"    Suggestion: {issue.suggestion}")

    print(f"\n  Valid: {result.is_valid}")


def demo_translator():
    """Demonstrate the PlusCal to pseudocode translator."""
    pluscal = """(*--algorithm Peterson
variables
    flag = [i \\in {0, 1} |-> FALSE],
    turn = 0;

process proc \\in {0, 1}
begin
    P1: flag[self] := TRUE;
    P2: turn := 1 - self;
    P3: await flag[1 - self] = FALSE \\/ turn = self;
    CS: skip;
    P4: flag[self] := FALSE;
    P5: goto P1;
end process;

end algorithm; *)"""

    translator = PlusCalToPseudocode()
    pseudocode = translator.translate(pluscal)

    print("=== PlusCal to Pseudocode Translation ===\n")
    print("--- PlusCal Input ---")
    print(pluscal)
    print("\n--- Pseudocode Output ---")
    print(pseudocode)


if __name__ == "__main__":
    demo_validator()
    print("\n" + "=" * 60 + "\n")
    demo_translator()
```

---

## 13. 요약 및 추가 읽을거리

### 핵심 요점

| 개념 | 핵심 통찰 |
|---------|-------------|
| 테스트 vs 검증 | 테스트는 버그를 찾고; 검증은 부재를 증명 (모델 내에서) |
| TLA+ | 시스템이 *무엇*을 하는지 기술, *어떻게*가 아님 — 설계 버그를 찾는 데 강력 |
| 상태와 액션 | 시스템 = 초기 상태 + 다음 상태 관계; 동작 = 무한 상태 시퀀스 |
| Safety 속성 | 도달 가능한 모든 상태에서 검사되는 불변식 (□P) |
| Liveness 속성 | 모든 동작에서 검사되는 시간적 속성 (◇P, P ~> Q) |
| PlusCal | TLA+로 컴파일되는 의사코드 같은 구문; 더 쉬운 진입점 |
| TLC model checker | 도달 가능한 모든 상태를 철저히 탐색; 최단 반례를 찾음 |
| AWS 경험 | TLA+가 수개월의 테스트가 놓친 중요한 버그를 발견; ROI가 매우 높음 |
| 실용적 범위 | 설계를 검증 (코드가 아님); 속성 기반 테스트로 격차를 메움 |

### 필수 읽을거리

1. **Lamport (2002)** — "Specifying Systems: The TLA+ Language and Tools for Hardware and Software Engineers" (결정적 교과서)
2. **Newcombe et al. (2015)** — "How Amazon Web Services Uses Formal Methods" (CACM)
3. **Lamport (2009)** — "The PlusCal Algorithm Language" (PlusCal 튜토리얼)
4. **Lamport** — "Learn TLA+" 웹사이트 (https://learntla.com)
5. **Wayne (2018)** — "Practical TLA+: Planning Driven Development" (서적)

### 다른 레슨과의 연결

- **Lesson 3 (FLP)**: FLP 불가능성을 TLA+에서 표현하고 탐색할 수 있음
- **Lesson 5 (Paxos)**: Lamport의 원래 Paxos 명세가 TLA+로 작성됨
- **Lesson 6 (Raft)**: Raft 논문에 Diego Ongaro가 작성한 TLA+ 명세가 포함됨
- **Lesson 16 (캡스톤)**: 우리의 KV 스토어 설계를 구현 전에 TLA+로 검증할 수 있음

---

[다음: 캡스톤 — 분산 KV 스토어 구축](./16_Capstone_Building_Distributed_KV_Store.md)
