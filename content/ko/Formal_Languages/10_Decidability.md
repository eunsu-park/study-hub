# 레슨 10: 결정 가능성(Decidability)

**이전**: [튜링 기계](./09_Turing_Machines.md) | **다음**: [환원 가능성](./11_Reducibility.md)

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 결정 가능(decidable) 언어와 결정 불가능(undecidable) 언어를 구별한다
2. 대각선 논법(diagonalization)을 이용하여 정지 문제(halting problem)의 결정 불가능성을 증명한다
3. 대각선 논법을 새로운 문제에 적용한다
4. 정규 언어와 문맥 자유 언어에 대한 결정 가능한 문제들을 파악한다
5. 결정 가능성(decidability)과 인식 가능성(recognizability)의 관계를 이해한다

---

결정 가능성은 컴퓨터 과학에서 가장 선명한 경계선을 그린다: 알고리즘이 풀 수 있는 문제와 어떤 알고리즘도 결코 풀 수 없는 문제 사이의 경계이다. 정지 문제 — 프로그램이 일반적으로 실행을 마칠지 여부를 결정할 수 있는가? — 는 결정 불가능한 것으로 밝혀졌다. 이는 현재 기술의 한계가 아니라, 과거·현재·미래의 모든 컴퓨터에 적용되는 수학적 불가능성이다.

## 1. 결정 가능한 언어(Decidable Languages)

언어 $L$이 **결정 가능(decidable)**하다는 것은, 모든 입력에 대해 정지하면서 원소 여부를 올바르게 수락/거부하는 TM이 존재한다는 의미이다. 이러한 TM을 **결정기(decider)**라고 한다.

**핵심 구분:**
- 결정 가능(Decidable) = 항상 정지한다 (수락 또는 거부)
- 인식 가능(Recognizable) = 원소에 대해서는 정지하고 수락하지만, 비원소에 대해서는 무한 루프에 빠질 수 있다

## 2. 정규 언어에 대한 결정 가능한 문제들

**정리.** 다음 문제들은 결정 가능하다:

### 2.1 $A_{DFA}$: DFA 수락 문제(DFA Acceptance)

$A_{DFA} = \{\langle B, w \rangle \mid B \text{는 } w \text{를 수락하는 DFA}\}$

*결정기:* $w$에 대해 $B$를 시뮬레이션한다. DFA는 $|w|$ 단계 후 항상 정지하므로 종료가 보장된다.

### 2.2 $E_{DFA}$: DFA 공집합 문제(DFA Emptiness)

$E_{DFA} = \{\langle A \rangle \mid A \text{는 DFA이고 } L(A) = \emptyset\}$

*결정기:* 시작 상태에서 수락 상태(accept state)에 도달 가능한지 그래프 도달 가능성(graph reachability)을 확인한다.

### 2.3 $EQ_{DFA}$: DFA 동치 문제(DFA Equivalence)

$EQ_{DFA} = \{\langle A, B \rangle \mid A, B \text{는 DFA이고 } L(A) = L(B)\}$

*결정기:* 대칭 차집합 $L(A) \triangle L(B) = (L(A) \cap \overline{L(B)}) \cup (\overline{L(A)} \cap L(B))$에 대한 DFA를 구성하고 공집합 여부를 확인한다.

## 3. 문맥 자유 언어에 대한 결정 가능한 문제들

### 3.1 $A_{CFG}$: CFG 원소 포함 문제(CFG Membership)

$A_{CFG} = \{\langle G, w \rangle \mid G \text{는 } w \text{를 생성하는 CFG}\}$

*결정기:* $G$를 촘스키 정규형(CNF)으로 변환하고 CYK 알고리즘을 실행한다.

### 3.2 $E_{CFG}$: CFG 공집합 문제(CFG Emptiness)

$E_{CFG} = \{\langle G \rangle \mid G \text{는 CFG이고 } L(G) = \emptyset\}$

*결정기:* 터미널(terminal)을 "생성 가능"으로 표시한다. 이미 표시된 기호들로만 이루어진 규칙을 가진 변수(variable)를 반복적으로 표시한다. $S$가 한 번도 표시되지 않으면 수락한다.

### 3.3 $EQ_{CFG}$는 어떨까?

$EQ_{CFG} = \{\langle G, H \rangle \mid L(G) = L(H)\}$는 **결정 불가능**하다!

이것이 정규 언어와 문맥 자유 언어의 핵심적인 차이점이다.

## 4. 정지 문제(The Halting Problem)

### 4.1 $A_{TM}$: TM 수락 문제(TM Acceptance)

$A_{TM} = \{\langle M, w \rangle \mid M \text{은 } w \text{를 수락하는 TM}\}$

**정리.** $A_{TM}$은 튜링 인식 가능(Turing-recognizable)하지만 결정 가능하지 않다.

*인식 가능성:* 범용 TM(universal TM) $U$가 $A_{TM}$을 인식한다 — $U$는 $w$에 대해 $M$을 시뮬레이션하고 $M$이 수락하면 수락한다. 하지만 $M$이 루프에 빠지면 $U$도 루프에 빠진다.

### 4.2 결정 불가능성 증명 (대각선 논법)

**정리.** $A_{TM}$은 결정 불가능하다.

*귀류법(proof by contradiction).* $H$가 $A_{TM}$의 결정기라고 가정한다:

$$H(\langle M, w \rangle) = \begin{cases} \text{수락} & \text{if } M \text{이 } w \text{를 수락하면} \\ \text{거부} & \text{if } M \text{이 } w \text{를 수락하지 않으면} \end{cases}$$

새로운 TM $D$를 구성한다:

> 입력 $\langle M \rangle$에 대한 $D$:
> 1. $\langle M, \langle M \rangle \rangle$에 대해 $H$를 실행한다
> 2. $H$가 수락하면 **거부**한다. $H$가 거부하면 **수락**한다.

이제 입력 $\langle D \rangle$에 대해 $D$를 실행하는 경우를 고려한다:
- $D$가 $\langle D \rangle$를 수락하면, $H$가 $\langle D, \langle D \rangle \rangle$를 수락했으므로 $D$는 $\langle D \rangle$를 수락한다. 그런데 $D$의 구성에 따르면 거부해야 한다. **모순.**
- $D$가 $\langle D \rangle$를 거부하면, $H$가 $\langle D, \langle D \rangle \rangle$를 거부했으므로 $D$는 $\langle D \rangle$를 수락하지 않는다. 그런데 $D$의 구성에 따르면 수락해야 한다. **모순.**

두 경우 모두 모순이 발생하므로 $H$는 존재할 수 없다. $\square$

### 4.3 대각선 논법(The Diagonalization Argument)

이 증명은 **대각선 논법(diagonalization)**을 사용하며, 이는 칸토어(Cantor)가 실수의 비가산성(uncountability of the reals)을 증명할 때 사용한 기법과 동일하다:

| | $\langle M_1 \rangle$ | $\langle M_2 \rangle$ | $\langle M_3 \rangle$ | $\cdots$ |
|---|---|---|---|---|
| $M_1$ | 수락 | 거부 | 수락 | $\cdots$ |
| $M_2$ | 거부 | 거부 | 수락 | $\cdots$ |
| $M_3$ | 수락 | 수락 | 거부 | $\cdots$ |
| $\vdots$ | | | | $\ddots$ |
| $D$ | **거부** | **수락** | **수락** | $\cdots$ |

$D$는 대각선 항목들을 뒤집으므로, $D$는 어떤 행에도 나타날 수 없다 — $D$는 입력 $\langle M_i \rangle$에 대해 모든 $M_i$와 다르다.

## 5. 결과들(Consequences)

### 5.1 $A_{TM}$의 여(Complement)

$\overline{A_{TM}} = \{\langle M, w \rangle \mid M \text{이 } w \text{를 수락하지 않음}\}$

**정리.** $\overline{A_{TM}}$은 튜링 인식 불가능(not Turing-recognizable)하다.

*증명.* 만약 $A_{TM}$과 $\overline{A_{TM}}$ 모두 인식 가능하다면, 두 인식기를 병렬로 실행하여 $A_{TM}$의 결정기를 구성할 수 있다. $A_{TM}$이 결정 불가능하므로 $\overline{A_{TM}}$은 인식 가능할 수 없다. $\square$

### 5.2 인식 가능(Recognizable)과 공동 인식 가능(Co-Recognizable)

**정리.** $L$이 결정 가능한 것은 $L$과 $\overline{L}$ 모두 튜링 인식 가능인 것과 동치이다.

*증명.*
- ($\Rightarrow$) $L$이 결정 가능하면, $L$의 결정기는 $L$을 인식하며, 그 출력을 뒤집으면 $\overline{L}$의 결정기(따라서 인식기)가 된다.
- ($\Leftarrow$) $L$의 인식기 $M_1$과 $\overline{L}$의 인식기 $M_2$가 주어질 때, 두 인식기를 병렬로 실행한다. 모든 입력에 대해 둘 중 하나는 반드시 수락한다. $M_1$이 먼저 수락하면 수락하고, $M_2$가 먼저 수락하면 거부한다. $\square$

## 6. 더 많은 결정 불가능한 문제들(More Undecidable Problems)

### 6.1 $HALT_{TM}$

$HALT_{TM} = \{\langle M, w \rangle \mid M \text{이 } w \text{에서 정지함}\}$

**결정 불가능.** $A_{TM}$으로부터 환원(reducible) 가능하다: 정지 문제를 결정할 수 있다면, 먼저 $M$이 $w$에서 정지하는지 확인하고 이어서 시뮬레이션하여 수락 여부를 판단함으로써 수락 문제도 결정할 수 있다.

### 6.2 $E_{TM}$: TM 공집합 문제(TM Emptiness)

$E_{TM} = \{\langle M \rangle \mid L(M) = \emptyset\}$

**결정 불가능.** $A_{TM}$을 결정하기 위해 $\langle M, w \rangle$가 주어지면, 입력을 무시하고 $w$에 대해 $M$을 시뮬레이션하는 $M'$을 구성한다. 그러면 $L(M') = \emptyset$인 것은 $M$이 $w$를 수락하지 않는 것과 동치이다.

### 6.3 $EQ_{TM}$: TM 동치 문제(TM Equivalence)

$EQ_{TM} = \{\langle M_1, M_2 \rangle \mid L(M_1) = L(M_2)\}$

**결정 불가능.** $E_{TM}$으로부터 환원 가능하다: $L(M) = L(M_\emptyset)$인가? 여기서 $M_\emptyset$은 모든 것을 거부한다.

## 7. 결정 가능성의 지형도(The Landscape of Decidability)

```
모든 언어 (비가산 무한개)
├── 튜링 인식 불가능 (예: A_TM의 여)
├── 튜링 인식 가능하지만 결정 불가능 (예: A_TM)
└── 결정 가능
    ├── 문맥 자유 언어(Context-free)
    │   └── 정규 언어(Regular)
    └── 비문맥 자유 결정 가능 언어 (예: {a^n b^n c^n})
```

## 연습 문제

### 연습 1: 결정 가능성 증명

다음 언어들이 결정 가능함을 증명하라:

1. $\{\langle M \rangle \mid M \text{은 길이} \leq 5 \text{인 어떤 문자열을 수락하는 DFA}\}$
2. $\{\langle G \rangle \mid G \text{는 CFG이고 } L(G) \text{가 무한하다}\}$
3. $\{\langle M \rangle \mid M \text{은 적어도 하나의 회문(palindrome)을 수락하는 DFA}\}$

### 연습 2: 결정 불가능성

$L = \{\langle M \rangle \mid M \text{은 } \varepsilon \text{를 수락하는 TM}\}$이 결정 불가능함을 보여라. (힌트: $A_{TM}$으로부터 환원하라.)

### 연습 3: 인식 가능성 분류

각 언어를 결정 가능, 인식 가능(하지만 결정 불가능), 또는 인식 불가능으로 분류하라:

1. $\{\langle M, w \rangle \mid M \text{이 1000 단계 이내에 } w \text{에서 정지함}\}$
2. $\{\langle M \rangle \mid M \text{이 어떤 문자열을 수락함}\}$
3. $\{\langle M \rangle \mid L(M) = \Sigma^*\}$

---

*이전: [레슨 9 — 튜링 기계](09_Turing_Machines.md) | 다음: [레슨 11 — 환원 가능성](11_Reducibility.md)*
