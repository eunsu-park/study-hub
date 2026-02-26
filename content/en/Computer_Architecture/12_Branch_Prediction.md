# Branch Prediction

**Previous**: [11_Pipelining.md](./11_Pipelining.md) | **Next**: [13_Superscalar_Out_of_Order.md](./13_Superscalar_Out_of_Order.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why branches disrupt pipelining
2. Distinguish static from dynamic branch prediction
3. Implement a 1-bit and 2-bit branch predictor
4. Describe the Branch Target Buffer (BTB) and its role
5. Calculate the performance impact of branch misprediction
6. Explain speculative execution and its relationship to prediction

---

Modern CPUs can execute billions of instructions per second, but a single mispredicted branch can waste dozens of cycles of work. Branch prediction -- the CPU's attempt to guess the outcome of a branch before it is resolved -- is critical to keeping the pipeline full. Understanding prediction explains why loop performance differs from branchy code, and why Spectre-class vulnerabilities exist.

## Table of Contents

1. [Control Hazards and Branch Problem](#1-control-hazards-and-branch-problem)
2. [Static Branch Prediction](#2-static-branch-prediction)
3. [Dynamic Branch Prediction](#3-dynamic-branch-prediction)
4. [Branch Target Buffer](#4-branch-target-buffer)
5. [Speculative Execution](#5-speculative-execution)
6. [Practice Problems](#6-practice-problems)

---

## 1. Control Hazards and Branch Problem

### Pipeline Bubbles from Branches

```
beq $t0, $t1, target
add $s0, $s1, $s2     # Should this execute?
sub $s3, $s4, $s5     # Should this execute?
...
target:
or  $s6, $s7, $s8

Time:    1    2    3    4    5    6    7
beq:    IF   ID  [EX]  MEM  WB
                  ↑
             Branch decision
add:         IF   ID   ← May need flush
sub:              IF   ← May need flush
```

### Branch Penalty

```
Penalty based on decision timing:

┌──────────────────┬────────────────┬───────────┐
│   Decision Stage │    Penalty     │   Note    │
├──────────────────┼────────────────┼───────────┤
│   ID stage       │   1 cycle      │ Early MIPS│
│   EX stage       │   2 cycles     │ Typical   │
│   MEM stage      │   3 cycles     │ Complex   │
└──────────────────┴────────────────┴───────────┘

Modern processors: 10-20 cycle pipelines
→ Branch prediction essential!
```

### Necessity of Branch Prediction

```
Branch frequency: ~20% (1 in every 5 instructions)

Always stall without prediction:
CPI = 1 + 0.2 × 3 = 1.6  (assuming 3 cycle penalty)

90% accurate prediction:
CPI = 1 + 0.2 × 0.1 × 3 = 1.06

Performance improvement: 1.6/1.06 = 1.5x
```

---

## 2. Static Branch Prediction

### 2.1 Always Not Taken

```
Strategy: Predict all branches are not taken

Pros: Simple implementation
Cons: Poor performance on loops

for (i = 0; i < 100; i++)  // 99 times taken, 1 time not taken
                           // Prediction accuracy: 1%
```

### 2.2 Always Taken

```
Strategy: Predict all branches are taken

Pros: Good for loops
Cons: Requires branch target address calculation

for (i = 0; i < 100; i++)  // Prediction accuracy: 99%
```

### 2.3 BTFN (Backward Taken, Forward Not Taken)

```
Strategy:
- Backward branches (loops): Predict Taken
- Forward branches (if statements): Predict Not Taken

┌─────────────────────────────────────────┐
│     Prediction by Branch Direction       │
│                                         │
│  PC=100:  beq label    (label=80)       │
│           ↑                             │
│           Backward branch → Predict Taken│
│                                         │
│  PC=100:  beq label    (label=120)      │
│           ↑                             │
│           Forward branch → Predict Not Taken│
└─────────────────────────────────────────┘

Loop end jumping back to start: Backward branch
if statement jumping to else: Forward branch
```

---

## 3. Dynamic Branch Prediction

### 3.1 1-bit Predictor

```
┌─────────────────────────────────────────┐
│           1-bit Predictor               │
├─────────────────────────────────────────┤
│  State: Taken(1) or Not Taken(0)        │
│                                         │
│  Operation:                             │
│  - Predict based on current state       │
│  - Flip state if wrong                  │
└─────────────────────────────────────────┘

State transition:
     Wrong        Wrong
  ┌────────┐   ┌────────┐
  │        │   │        │
  ▼        │   ▼        │
┌────┐     │ ┌────┐     │
│ NT │◀────┴─│ T  │◀────┘
└────┘  Right └────┘  Right

Problem: Always wrong twice at loop start and end
```

### Limitations of 1-bit Predictor

```
100 loop iterations:
T T T ... T T N  (99 T, 1 N)
                 ↑ Loop exit

Prediction:   T T T ... T T T N
Actual:       T T T ... T T N ←
                      ↑ Wrong (predicted T, actual N)

Next loop start:
Prediction:   N ← Wrong (predicted N, actual T)

2 mispredictions per 100 iterations → 98% accuracy
```

### 3.2 2-bit Predictor

```
┌─────────────────────────────────────────────────┐
│              2-bit Saturating Counter           │
├─────────────────────────────────────────────────┤
│  4 states:                                      │
│  - 00: Strongly Not Taken                       │
│  - 01: Weakly Not Taken                         │
│  - 10: Weakly Taken                             │
│  - 11: Strongly Taken                           │
└─────────────────────────────────────────────────┘

State transition diagram:
        N              N              N
   ┌────────┐     ┌────────┐     ┌────────┐
   │        ▼     │        ▼     │        ▼
┌──┴──┐  ┌──┴──┐  ┌──┴──┐  ┌──┴──┐
│ SNT │  │ WNT │  │ WT  │  │ ST  │
│ 00  │  │ 01  │  │ 10  │  │ 11  │
└──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘
   │        ▲     │        ▲     │        ▲
   └────────┘     └────────┘     └────────┘
        T              T              T

Prediction: Taken if upper bit is 1, Not Taken if 0
```

### 2-bit Predictor Operation Example

```
100 loop iterations:
Actual:   T T T ... T T N T T T ... (next loop)

State:  ST ST ST ... ST WT ST ST ST ...
Pred:    T  T  T ...  T  T  T  T  T ...
Correct: ✓  ✓  ✓ ... ✓  ✗  ✓  ✓  ✓ ...

Only 1 misprediction! 1 failure per loop
99%+ accuracy
```

### 3.3 Branch History Table (BHT)

```
┌─────────────────────────────────────────────────┐
│         Branch History Table (BHT)              │
├─────────────────────────────────────────────────┤
│                                                 │
│   PC[9:2] ───▶ ┌─────────┐                     │
│                │ Index   │                     │
│                │  256    │                     │
│                │ entries │                     │
│                └────┬────┘                     │
│                     │                          │
│                     ▼                          │
│              2-bit counter                     │
│                                                │
└─────────────────────────────────────────────────┘

Uses part of PC as index
- Each entry stores a 2-bit counter
- Multiple branches can map to same index (aliasing)
```

### 3.4 Correlating Predictor

```
Idea: Other branch outcomes affect current branch

Example code:
if (a == 2)     // B1
    a = 0;
if (b == 2)     // B2
    b = 0;
if (a != b)     // B3: Correlated with B1, B2 outcomes!
    ...

(m, n) predictor:
- m: Record last m branch outcomes (Global History)
- n: n-bit counter

┌──────────────────────────────────────────┐
│         (2, 2) Correlating Predictor     │
│                                          │
│  Global History: 2 bits (00, 01, 10, 11) │
│  Per history: 2-bit counter              │
│                                          │
│  Total of 4 2-bit counters               │
└──────────────────────────────────────────┘
```

### 3.5 Tournament Predictor

```
┌────────────────────────────────────────────────┐
│           Tournament Predictor                  │
│                                                 │
│         ┌─────────────┐                        │
│  PC ───▶│  Selector   │ (2-bit counter)        │
│         └──────┬──────┘                        │
│                │                               │
│         ┌──────┴──────┐                        │
│         ▼             ▼                        │
│  ┌────────────┐ ┌────────────┐                 │
│  │   Local    │ │   Global   │                 │
│  │ Predictor  │ │ Predictor  │                 │
│  └──────┬─────┘ └─────┬──────┘                 │
│         │             │                        │
│         └──────┬──────┘                        │
│                ▼                               │
│           MUX (select)                         │
│                │                               │
│                ▼                               │
│            Prediction                          │
└────────────────────────────────────────────────┘

- Local: Uses individual branch history
- Global: Uses overall branch history
- Selector: Learns which predictor is more accurate
```

---

## 4. Branch Target Buffer

### BTB (Branch Target Buffer)

```
Stores whether instruction is a branch + target address

┌─────────────────────────────────────────────────┐
│              Branch Target Buffer               │
├───────────┬──────────┬─────────────────────────┤
│    Tag    │  Target  │  Prediction (optional)  │
├───────────┼──────────┼─────────────────────────┤
│ 0x1000... │  0x2000  │        ST               │
│ 0x1100... │  0x1500  │        WT               │
│ 0x2000... │  0x1800  │        SNT              │
│    ...    │   ...    │        ...              │
└───────────┴──────────┴─────────────────────────┘

BTB lookup in IF stage:
- Hit + Taken prediction: Jump to target immediately
- Miss: Normal execution, update BTB later
```

### BTB Operation Flow

```
┌─────────────────────────────────────────────────┐
│                IF Stage                          │
│                                                 │
│   PC ───▶ BTB Lookup                            │
│              │                                  │
│              ▼                                  │
│         ┌────────┐                             │
│         │  Hit?  │                             │
│         └────┬───┘                             │
│              │                                  │
│      No      │      Yes                        │
│      ▼       │       ▼                         │
│   PC + 4     │   Taken prediction?             │
│              │       │                          │
│              │   Yes │  No                     │
│              │    ▼  │   ▼                     │
│              │  Target  PC + 4                 │
│              │   addr                          │
└─────────────────────────────────────────────────┘
```

---

## 5. Speculative Execution

### Concept

```
Execute instructions on predicted path before branch result confirmed

┌─────────────────────────────────────────────────┐
│           Speculative Execution                 │
│                                                 │
│   beq prediction: Taken                         │
│      │                                          │
│      ▼                                          │
│   Execute target instructions speculatively     │
│      │                                          │
│      ▼                                          │
│   Branch result confirmed                       │
│      │                                          │
│   ┌──┴──┐                                      │
│   │     │                                      │
│ Correct Wrong                                   │
│ prediction prediction                           │
│   │     │                                      │
│   ▼     ▼                                      │
│ Commit Discard                                  │
│ result (flush)                                  │
└─────────────────────────────────────────────────┘
```

### Misprediction Recovery

```
On misprediction:
1. Pipeline flush (remove speculative instructions)
2. Restore register state
3. Re-execute on correct path

Recovery cost:
- Proportional to pipeline depth
- Modern processors: 10-20 cycle penalty
```

### Impact of Misprediction Rate

```
CPI = CPI_base + (branch ratio) × (misprediction rate) × (penalty)

Example:
- CPI_base = 1
- Branch ratio = 20%
- Penalty = 15 cycles

┌────────────────┬──────────────────┐
│ Misprediction  │      CPI         │
│    Rate        │                  │
├────────────────┼──────────────────┤
│     10%        │ 1 + 0.2×0.1×15   │
│                │ = 1.30           │
├────────────────┼──────────────────┤
│      5%        │ 1 + 0.2×0.05×15  │
│                │ = 1.15           │
├────────────────┼──────────────────┤
│      2%        │ 1 + 0.2×0.02×15  │
│                │ = 1.06           │
├────────────────┼──────────────────┤
│      1%        │ 1 + 0.2×0.01×15  │
│                │ = 1.03           │
└────────────────┴──────────────────┘
```

---

## 6. Practice Problems

### Basic Problems

1. What are the 3 static branch prediction methods?

2. What are the 4 states of a 2-bit predictor?

3. For the sequence below, what are the predictions of a 1-bit predictor (initial state T)?
   `T T N T T N T T`

### Analysis Problems

4. How does a 2-bit predictor (initial ST) behave for this loop?
```c
for (int i = 0; i < 4; i++) {
    // loop body
}
```

5. With 25% branch ratio, 10 cycle penalty, and 95% prediction accuracy, what is the CPI?

### Advanced Problems

6. Explain why tournament predictors are better than single predictors.

7. Compare branch handling with and without BTB.

<details>
<summary>Answers</summary>

1. Always Not Taken, Always Taken, BTFN (Backward Taken, Forward Not Taken)

2. Strongly Not Taken (00), Weakly Not Taken (01), Weakly Taken (10), Strongly Taken (11)

3.
```
Actual: T T N T T N T T
Pred:   T T T N T T N T
State:  T T N T T N T T
Correct: ✓ ✓ ✗ ✗ ✓ ✗ ✗ ✓  (4/8 = 50%)
```

4.
```
Iterations: T T T N (4 iterations)
State: ST ST ST WT WT
Pred:  T  T  T  T  T
Correct: ✓  ✓  ✓  ✗
```
1 failure in 4 = 75% accuracy

5. CPI = 1 + 0.25 × 0.05 × 10 = 1.125

6. Tournament predictors use both Local and Global predictors, selectively choosing the more accurate one for each branch. Some branches have important local patterns while others have important global correlations.

7.
- Without BTB: Branch type and target discovered only at ID or EX stage, penalty incurred
- With BTB: On BTB hit in IF stage, can jump to target immediately, minimizing penalty

</details>

---

## Next Steps

- [13_Superscalar_Out_of_Order.md](./13_Superscalar_Out_of_Order.md) - ILP and Out-of-Order Execution

---

## References

- Computer Architecture: A Quantitative Approach, Chapter 3 (Hennessy & Patterson)
- [Branch Prediction Competition](https://www.jilp.org/cbp2016/)
- [Agner Fog's Microarchitecture Guide](https://www.agner.org/optimize/)
