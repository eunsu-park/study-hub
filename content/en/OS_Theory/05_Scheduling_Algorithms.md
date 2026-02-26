# Scheduling Algorithms

**Previous**: [CPU Scheduling Basics](./04_CPU_Scheduling_Basics.md) | **Next**: [Advanced Scheduling](./06_Advanced_Scheduling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement FCFS scheduling and identify the convoy effect that degrades its performance
2. Calculate average waiting time and turnaround time for SJF scheduling using Gantt charts
3. Compare SJF (non-preemptive) with SRTF (preemptive) and explain when preemption improves performance
4. Explain priority scheduling and analyze the starvation problem and how aging resolves it
5. Analyze Round Robin performance as a function of time quantum size, identifying the tradeoffs between responsiveness and context switch overhead
6. Evaluate which scheduling algorithm is most appropriate for a given workload type (batch, interactive, real-time)

---

There is no single "best" scheduling algorithm -- each makes different tradeoffs between fairness, throughput, and response time. Understanding these algorithms helps you reason about why your program might be slow even when the CPU is not fully loaded. In this lesson, we study the operating principles of FCFS, SJF, SRTF, Priority, and Round Robin, and learn to calculate average waiting time and turnaround time through Gantt charts.

## Table of Contents

1. [FCFS (First-Come, First-Served)](#1-fcfs-first-come-first-served)
2. [SJF (Shortest Job First)](#2-sjf-shortest-job-first)
3. [SRTF (Shortest Remaining Time First)](#3-srtf-shortest-remaining-time-first)
4. [Priority Scheduling](#4-priority-scheduling)
5. [Round Robin](#5-round-robin)
6. [Algorithm Comparison](#6-algorithm-comparison)
7. [Practice Problems](#7-practice-problems)

---

## 1. FCFS (First-Come, First-Served)

### Concept

```
FCFS = First arrived process gets served first
     = Simplest scheduling algorithm
     = Non-preemptive

┌─────────────────────────────────────────────────────────┐
│                      FCFS Operation                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Ready Queue (FIFO):                                   │
│  ┌───┐   ┌───┐   ┌───┐   ┌───┐                         │
│  │P1 │ → │P2 │ → │P3 │ → │P4 │ →  CPU                  │
│  └───┘   └───┘   └───┘   └───┘                         │
│  First   Second   Third   Fourth                        │
│  arrived arrived  arrived arrived                       │
│                                                         │
│  Execution Order: P1 → P2 → P3 → P4                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Example 1: Basic

```
Input:
┌──────────┬──────────────┬──────────────┐
│ Process  │ Arrival Time │  Burst Time  │
├──────────┼──────────────┼──────────────┤
│    P1    │      0       │     24       │
│    P2    │      0       │      3       │
│    P3    │      0       │      3       │
└──────────┴──────────────┴──────────────┘

Gantt Chart (Order: P1 → P2 → P3):
┌────────────────────────────────────┬──────┬──────┐
│               P1                   │  P2  │  P3  │
└────────────────────────────────────┴──────┴──────┘
0                                   24     27     30

Why this order? FCFS is strictly arrival-order. All three arrive
at time 0, so they execute in submission order: P1 first.
P2 and P3 (burst=3 each) must wait 24ms for P1 to finish --
this is the convoy effect in action. If P2 and P3 ran first,
the average waiting time would drop from 17 to just 3.

Calculation:
┌──────────┬──────────────┬─────────────────────┐
│ Process  │ Waiting Time │ Turnaround Time     │
├──────────┼──────────────┼─────────────────────┤
│    P1    │ 0            │ 24 - 0 = 24         │
│    P2    │ 24           │ 27 - 0 = 27         │
│    P3    │ 27           │ 30 - 0 = 30         │
├──────────┼──────────────┼─────────────────────┤
│  Average │ (0+24+27)/3 = 17 │ (24+27+30)/3 = 27│
└──────────┴──────────────┴─────────────────────┘
```

### Convoy Effect

```
┌─────────────────────────────────────────────────────────┐
│                    Convoy Effect                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  When a long process arrives first:                    │
│  ┌────────────────────────┬──┬──┬──┐                    │
│  │       P1 (long job)     │P2│P3│P4│                    │
│  └────────────────────────┴──┴──┴──┘                    │
│  0                        100                           │
│                                                         │
│  Short processes wait for the long process             │
│  → Average waiting time increases significantly        │
│                                                         │
│  If the order was reversed (P2, P3, P4, P1):          │
│  ┌──┬──┬──┬────────────────────────┐                    │
│  │P2│P3│P4│       P1 (long job)     │                    │
│  └──┴──┴──┴────────────────────────┘                    │
│  0  3  6  9                       109                   │
│                                                         │
│  Short jobs complete quickly                           │
│  → Average waiting time decreases                      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### FCFS Characteristics

```
┌──────────┬─────────────────────────────────────────────┐
│   Pros   │ • Very simple implementation (FIFO queue)   │
│          │ • No starvation                             │
│          │ • Fair (first-come, first-served order)     │
├──────────┼─────────────────────────────────────────────┤
│   Cons   │ • Convoy Effect possible                    │
│          │ • Average waiting time can be long          │
│          │ • Disadvantageous for I/O-bound processes   │
├──────────┼─────────────────────────────────────────────┤
│ Suitable │ • When all processes have similar burst time│
│ for      │ • Batch systems                             │
└──────────┴─────────────────────────────────────────────┘
```

**Why FCFS is simple but dangerous**: FCFS requires only a FIFO queue -- no comparisons, no predictions, no priority management. This makes it trivially correct and starvation-free. But this simplicity comes at a cost: it completely ignores burst length. A single long process can block dozens of short ones, causing the convoy effect. This is why FCFS is acceptable only when all processes have similar burst times (where ordering doesn't matter much) or in batch systems where response time is not a concern.

---

## 2. SJF (Shortest Job First)

### Concept

```
SJF = Execute process with shortest burst time first
    = Provides optimal average waiting time (non-preemptive)
    = Non-preemptive

┌─────────────────────────────────────────────────────────┐
│                      SJF Operation                      │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Select shortest burst time from Ready Queue:          │
│                                                         │
│  ┌───┐   ┌───┐   ┌───┐   ┌───┐                         │
│  │P1 │   │P2 │   │P3 │   │P4 │                         │
│  │24 │   │ 3 │   │ 3 │   │ 5 │  ← Burst Time           │
│  └───┘   └───┘   └───┘   └───┘                         │
│                                                         │
│  Selection Order: P2(3) → P3(3) → P4(5) → P1(24)      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Example 2: Non-preemptive SJF

```
Input:
┌──────────┬──────────────┬──────────────┐
│ Process  │ Arrival Time │  Burst Time  │
├──────────┼──────────────┼──────────────┤
│    P1    │      0       │      6       │
│    P2    │      2       │      8       │
│    P3    │      4       │      7       │
│    P4    │      5       │      3       │
└──────────┴──────────────┴──────────────┘

Gantt Chart:
Time 0: Only P1 arrived → Execute P1 (no choice -- it's the only one)
Time 6: P1 complete. P2(8), P3(7), P4(3) are waiting.
        SJF picks the shortest burst → P4(3) wins.
        Why? Finishing P4 first frees up the CPU soonest,
        minimizing how long P2 and P3 continue to wait.
Time 9: P4 complete. P2(8) and P3(7) remain.
        SJF picks P3(7) over P2(8) -- shorter burst first.
Time 16: P3 complete, only P2 remains → execute P2
Time 24: P2 complete

┌──────────┬─────┬──────────┬────────────┐
│    P1    │ P4  │    P3    │     P2     │
└──────────┴─────┴──────────┴────────────┘
0          6     9         16          24

Calculation:
┌──────────┬───────────────────────┬─────────────────────┐
│ Process  │    Waiting Time       │  Turnaround Time    │
├──────────┼───────────────────────┼─────────────────────┤
│    P1    │ 0 - 0 = 0            │ 6 - 0 = 6           │
│    P2    │ 16 - 2 = 14          │ 24 - 2 = 22         │
│    P3    │ 9 - 4 = 5            │ 16 - 4 = 12         │
│    P4    │ 6 - 5 = 1            │ 9 - 5 = 4           │
├──────────┼───────────────────────┼─────────────────────┤
│  Average │ (0+14+5+1)/4 = 5     │ (6+22+12+4)/4 = 11  │
└──────────┴───────────────────────┴─────────────────────┘
```

### Burst Time Prediction Problem

```
┌─────────────────────────────────────────────────────────┐
│          SJF Burst Time Prediction Problem              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Problem: How to know next CPU burst length?           │
│                                                         │
│  Solution: Predict based on history (exponential avg)  │
│                                                         │
│  τ(n+1) = α * t(n) + (1 - α) * τ(n)                   │
│                                                         │
│  Where:                                                 │
│  τ(n+1) = Next CPU burst prediction                    │
│  t(n)   = Actual n-th CPU burst value                 │
│  τ(n)   = Previous prediction                          │
│  α      = 0 ≤ α ≤ 1 (weight)                          │
│                                                         │
│  Example with α = 0.5:                                 │
│  ┌───────┬────────┬────────┬────────┐                  │
│  │  n    │ t(n)   │ τ(n)   │ τ(n+1) │                  │
│  ├───────┼────────┼────────┼────────┤                  │
│  │  0    │   -    │  10    │   -    │                  │
│  │  1    │   6    │  10    │   8    │                  │
│  │  2    │   4    │   8    │   6    │                  │
│  │  3    │   6    │   6    │   6    │                  │
│  └───────┴────────┴────────┴────────┘                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Why SJF is optimal for average waiting time but impractical**: SJF provably minimizes average waiting time among all non-preemptive algorithms. The intuition is simple: by running short jobs first, you reduce the cumulative waiting of all jobs behind them (the same reason express lanes at grocery stores reduce average wait). The catch is that the OS cannot know a process's burst time in advance -- it must predict it. Exponential averaging works reasonably well for repetitive workloads (e.g., database queries that repeat similar operations), but for unpredictable workloads the predictions can be wildly off, negating SJF's theoretical advantage. Furthermore, SJF can starve long processes indefinitely if short ones keep arriving.

---

## 3. SRTF (Shortest Remaining Time First)

### Concept

```
SRTF = Preemptive version of SJF
     = Select process with shortest remaining time
     = Preemption possible when new process arrives

┌─────────────────────────────────────────────────────────┐
│                      SRTF Operation                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Time 0: P1(6) executing                               │
│  Time 2: P2(8) arrives                                 │
│     P1 remaining time = 4                              │
│     P2 remaining time = 8                              │
│     → P1(4) < P2(8) → Continue P1                      │
│                                                         │
│  Time 4: P3(7) arrives                                 │
│     P1 remaining time = 2                              │
│     P2 remaining time = 8                              │
│     P3 remaining time = 7                              │
│     → P1(2) shortest → Continue P1                     │
│                                                         │
│  Time 5: P4(3) arrives                                 │
│     P1 remaining time = 1                              │
│     → P1(1) shortest → Continue P1                     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Example 3: SRTF

```
Input:
┌──────────┬──────────────┬──────────────┐
│ Process  │ Arrival Time │  Burst Time  │
├──────────┼──────────────┼──────────────┤
│    P1    │      0       │      8       │
│    P2    │      1       │      4       │
│    P3    │      2       │      9       │
│    P4    │      3       │      5       │
└──────────┴──────────────┴──────────────┘

Gantt Chart Analysis:
Time 0: P1(8) starts -- only process available
Time 1: P2(4) arrives. P1 remaining=7, P2 remaining=4.
        P2(4) < P1(7) → Preempt P1 and switch to P2!
        Why preempt? P2 will finish sooner, reducing total
        accumulated waiting for all other processes.
Time 2: P3(9) arrives. P2 remaining=3 is still shortest
        (3 < 7 < 9) → Continue P2. No benefit to switching.
Time 3: P4(5) arrives. P2 remaining=2 still shortest
        (2 < 5 < 7 < 9) → Continue P2. Almost done.
Time 5: P2 completes. Now compare: P4(5) < P1(7) < P3(9)
        → Select P4. Shorter remaining time = less total wait.
Time 10: P4 completes. P1(7) < P3(9) → Select P1.
Time 17: P1 completes. Only P3(9) remains → Select P3.
         P3 waited the longest because it always had the
         longest remaining time -- this is the starvation risk.
Time 26: P3 completes

┌──┬────────┬──────────┬──────────────┬──────────────────┐
│P1│   P2   │    P4    │      P1      │        P3        │
└──┴────────┴──────────┴──────────────┴──────────────────┘
0  1        5         10             17                 26

Calculation:
┌──────────┬──────────────────────────────┬────────────────────┐
│ Process  │      Waiting Time            │  Turnaround Time   │
├──────────┼──────────────────────────────┼────────────────────┤
│    P1    │ (10-1)-(0) = 9              │ 17 - 0 = 17        │
│    P2    │ 0 (executes after preemption)│ 5 - 1 = 4          │
│    P3    │ 17 - 2 = 15                 │ 26 - 2 = 24        │
│    P4    │ 5 - 3 = 2                   │ 10 - 3 = 7         │
├──────────┼──────────────────────────────┼────────────────────┤
│  Average │ (9+0+15+2)/4 = 6.5          │ (17+4+24+7)/4 = 13 │
└──────────┴──────────────────────────────┴────────────────────┘
```

### SRTF Characteristics

```
┌──────────┬─────────────────────────────────────────────┐
│   Pros   │ • Theoretically optimal average waiting time│
│          │ • Priority to short processes               │
├──────────┼─────────────────────────────────────────────┤
│   Cons   │ • Burst time prediction required            │
│          │ • Long processes can starve                 │
│          │ • Context switch overhead                   │
├──────────┼─────────────────────────────────────────────┤
│Starvation│ If short processes keep arriving,           │
│          │ long processes may never execute            │
└──────────┴─────────────────────────────────────────────┘
```

---

## 4. Priority Scheduling

### Concept

```
Priority Scheduling = Execute higher priority process first
                    = Can be preemptive or non-preemptive
                    = Lower number = Higher priority (typically)

┌─────────────────────────────────────────────────────────┐
│                  Priority Scheduling                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Compare priorities in Ready Queue:                    │
│                                                         │
│  ┌───┐   ┌───┐   ┌───┐   ┌───┐                         │
│  │P1 │   │P2 │   │P3 │   │P4 │                         │
│  │ 3 │   │ 1 │   │ 4 │   │ 2 │  ← Priority             │
│  └───┘   └───┘   └───┘   └───┘                         │
│                                                         │
│  Selection Order: P2(1) → P4(2) → P1(3) → P3(4)       │
│                                                         │
│  Priority determination criteria:                      │
│  • Internal: Time limits, memory requirements, I/O:CPU │
│  • External: User importance, cost, political factors  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Example 4: Priority (Non-preemptive)

```
Input:
┌──────────┬──────────────┬──────────────┬──────────┐
│ Process  │ Arrival Time │  Burst Time  │ Priority │
├──────────┼──────────────┼──────────────┼──────────┤
│    P1    │      0       │     10       │    3     │
│    P2    │      0       │      1       │    1     │
│    P3    │      0       │      2       │    4     │
│    P4    │      0       │      1       │    5     │
│    P5    │      0       │      5       │    2     │
└──────────┴──────────────┴──────────────┴──────────┘
(Lower number = Higher priority)

Gantt Chart:
Priority order: P2(1) → P5(2) → P1(3) → P3(4) → P4(5)

┌──┬───────┬────────────┬────┬──┐
│P2│  P5   │     P1     │ P3 │P4│
└──┴───────┴────────────┴────┴──┘
0  1       6           16   18 19

Calculation:
┌──────────┬──────────────┬─────────────────────┐
│ Process  │ Waiting Time │  Turnaround Time    │
├──────────┼──────────────┼─────────────────────┤
│    P1    │ 6            │ 16                  │
│    P2    │ 0            │ 1                   │
│    P3    │ 16           │ 18                  │
│    P4    │ 18           │ 19                  │
│    P5    │ 1            │ 6                   │
├──────────┼──────────────┼─────────────────────┤
│  Average │ 8.2          │ 12                  │
└──────────┴──────────────┴─────────────────────┘
```

### Starvation and Aging

```
┌─────────────────────────────────────────────────────────┐
│              Starvation Problem                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Problem Scenario:                                     │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Ready Queue:                                      │ │
│  │  P(low priority) ──────────────────────────────▶   │ │
│  │        ↑ High priority processes keep arriving    │ │
│  │  P(high), P(high), P(high), P(high), ...          │ │
│  │                                                    │ │
│  │  Low priority process never executes              │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
│  Solution: Aging                                        │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Gradually increase priority with waiting time    │ │
│  │                                                    │ │
│  │  Time 0: P priority = 100 (low)                   │ │
│  │  Time 10: P priority = 90                         │ │
│  │  Time 20: P priority = 80                         │ │
│  │  ...                                              │ │
│  │  Time 100: P priority = 0 (highest) → Eventually │ │
│  │            executes                               │ │
│  │                                                    │ │
│  │  → Solves starvation problem                      │ │
│  └────────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Round Robin

### Concept

```
Round Robin (RR) = Algorithm for time-sharing systems
                 = Allocate equal time (Time Quantum) to each
                 = Preemptive
                 = FCFS + Preemption

┌─────────────────────────────────────────────────────────┐
│                   Round Robin Operation                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Time Quantum = 4ms                                    │
│                                                         │
│  Ready Queue (circular):                               │
│  ┌───┐   ┌───┐   ┌───┐                                 │
│  │P1 │ → │P2 │ → │P3 │ ─┐                              │
│  └───┘   └───┘   └───┘   │                             │
│    ↑                     │                              │
│    └─────────────────────┘                              │
│                                                         │
│  Operation:                                             │
│  1. P1 executes for 4ms                                │
│  2. Time quantum expires → P1 goes to end of queue     │
│  3. P2 executes for 4ms                                │
│  4. Repeat...                                          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Example 5: Round Robin

```
Input:
┌──────────┬──────────────┬──────────────┐
│ Process  │ Arrival Time │  Burst Time  │
├──────────┼──────────────┼──────────────┤
│    P1    │      0       │     24       │
│    P2    │      0       │      3       │
│    P3    │      0       │      3       │
└──────────┴──────────────┴──────────────┘

Time Quantum = 4ms

Gantt Chart:
Time 0: P1 executes (4ms)
Time 4: P1 interrupted (remaining=20), P2 executes (completes in 3ms)
Time 7: P2 completes, P3 executes (completes in 3ms)
Time 10: P3 completes, P1 executes (4ms)
Time 14: P1 interrupted (remaining=16), no other processes, P1 executes
Time 18: P1 interrupted (remaining=12), P1 executes
Time 22: P1 interrupted (remaining=8), P1 executes
Time 26: P1 interrupted (remaining=4), P1 executes
Time 30: P1 completes

┌────┬───┬───┬────┬────┬────┬────┬────┐
│ P1 │P2 │P3 │ P1 │ P1 │ P1 │ P1 │ P1 │
└────┴───┴───┴────┴────┴────┴────┴────┘
0    4   7  10   14   18   22   26   30

Calculation:
┌──────────┬──────────────┬───────────────────────────┐
│ Process  │ Waiting Time │  Turnaround Time          │
├──────────┼──────────────┼───────────────────────────┤
│    P1    │ 30-24 = 6    │ 30 - 0 = 30               │
│    P2    │ 4 - 0 = 4    │ 7 - 0 = 7                 │
│    P3    │ 7 - 0 = 7    │ 10 - 0 = 10               │
├──────────┼──────────────┼───────────────────────────┤
│  Average │ (6+4+7)/3 = 5.67│ (30+7+10)/3 = 15.67    │
└──────────┴──────────────┴───────────────────────────┘
```

### Time Quantum Impact

```
┌─────────────────────────────────────────────────────────┐
│              Time Quantum Impact                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  If Time Quantum is very large:                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  → Becomes identical to FCFS                    │    │
│  │  → Response time increases                      │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  If Time Quantum is very small:                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  → Context switch overhead increases            │    │
│  │  → CPU time wasted                              │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  Recommended Time Quantum:                             │
│  • 80% of average CPU bursts complete within TQ       │
│  • Typically 10~100ms                                  │
│  • At least 10x context switch time                   │
│                                                         │
│  Visualization:                                         │
│                                                         │
│  q=∞: │P1(entire)              │P2(entire)  │P3...     │
│       └────────────────────────┴───────────┴──...      │
│                                                         │
│  q=1: │P│P│P│P│P│P│...│ ← Too many context switches    │
│       └─┴─┴─┴─┴─┴─┴...┘                                │
│                                                         │
│  Appropriate q: │──P1──│──P2──│──P1──│──P3──│...│      │
│                └──────┴──────┴──────┴──────┴...┘       │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Why time quantum choice matters -- the Goldilocks problem**: The time quantum is a critical tuning parameter because it directly controls the balance between responsiveness and overhead:

- **Too small** (e.g., q=1ms with 5us context switch): The CPU spends ~0.5% of every quantum just switching contexts. With many processes, this overhead compounds -- the system spends more time *managing* work than *doing* work. In the extreme (q approaching context switch time), useful throughput drops toward zero.
- **Too large** (e.g., q=10 seconds): Each process monopolizes the CPU for a long stretch, and other processes must wait. This degenerates into FCFS behavior, defeating the purpose of Round Robin. Interactive users would perceive the system as unresponsive.
- **Just right** (typically 10-100ms): The rule of thumb is that about 80% of CPU bursts should complete within one quantum. This means most processes finish without being preempted (avoiding unnecessary context switches), while long-running processes still get interrupted often enough to keep response time low.

---

## 6. Algorithm Comparison

### Comprehensive Comparison Table

```
┌─────────────┬────────┬─────────┬─────────┬─────────────┬──────────┐
│  Algorithm  │Preempt │Starvation│Avg Wait │Implementation│Features  │
├─────────────┼────────┼─────────┼─────────┼─────────────┼──────────┤
│ FCFS        │   X    │   X     │  Long   │ Very Simple │ Convoy   │
│             │        │         │         │             │ Effect   │
├─────────────┼────────┼─────────┼─────────┼─────────────┼──────────┤
│ SJF         │   X    │   O     │ Optimal │ Predict Time│ Possible │
│             │        │         │         │             │Starvation│
├─────────────┼────────┼─────────┼─────────┼─────────────┼──────────┤
│ SRTF        │   O    │   O     │ Optimal │ Predict Time│ Possible │
│             │        │         │         │             │Starvation│
├─────────────┼────────┼─────────┼─────────┼─────────────┼──────────┤
│ Priority    │  O/X   │   O     │ Varies  │Priority Mgmt│ Need     │
│             │        │         │         │             │ Aging    │
├─────────────┼────────┼─────────┼─────────┼─────────────┼──────────┤
│ RR          │   O    │   X     │ Medium  │ Simple      │TQ Choice │
│             │        │         │         │             │Important │
└─────────────┴────────┴─────────┴─────────┴─────────────┴──────────┘
```

### Same Data Comparison Example

```
Input:
┌──────────┬──────────────┬──────────────┐
│ Process  │ Arrival Time │  Burst Time  │
├──────────┼──────────────┼──────────────┤
│    P1    │      0       │      7       │
│    P2    │      2       │      4       │
│    P3    │      4       │      1       │
│    P4    │      5       │      4       │
└──────────┴──────────────┴──────────────┘

FCFS:
┌────────────────┬─────────┬──┬─────────┐
│       P1       │    P2   │P3│   P4    │
└────────────────┴─────────┴──┴─────────┘
0                7        11 12        16
Average Waiting Time: (0 + 5 + 7 + 7) / 4 = 4.75
→ P1 runs first simply because it arrived first. P2-P4 all
  wait behind P1's full 7ms burst -- classic convoy effect.

SJF (non-preemptive):
┌────────────────┬──┬─────────┬─────────┐
│       P1       │P3│    P4   │   P2    │
└────────────────┴──┴─────────┴─────────┘
0                7  8        12        16
Average Waiting Time: (0 + 10 + 3 + 3) / 4 = 4.00
→ P1 runs first (only arrival at time 0). After P1,
  SJF picks P3(1) then P4(4) then P2(4). Shorter jobs
  go first, reducing total accumulated waiting.

SRTF:
┌────┬─────────┬──┬─────────┬────────┐
│ P1 │   P2    │P3│   P4    │   P1   │
└────┴─────────┴──┴─────────┴────────┘
0    2         6  7        11       16
Average Waiting Time: (9 + 0 + 2 + 2) / 4 = 3.25
→ P1 starts but gets preempted at time 2 when P2(4)
  arrives with a shorter remaining time than P1(5).
  Preemption lets short jobs finish sooner, pulling
  the average wait down further than non-preemptive SJF.

RR (q=2):
┌────┬────┬────┬──┬────┬────┬────┐
│ P1 │ P2 │ P1 │P3│ P4 │ P2 │ P1 │
└────┴────┴────┴──┴────┴────┴────┘
0    2    4    6  7    9   11   14   16
Average Waiting Time: Calculation needed...
→ Everyone gets a fair 2ms slice. No starvation,
  but more context switches than needed -- this is
  the price of fairness over optimality.

Conclusion: SRTF provides shortest average waiting time
```

---

## 7. Practice Problems

### Problem 1: FCFS

Apply FCFS scheduling to the following processes and calculate average waiting time.

| Process | Arrival Time | Burst Time |
|---------|-------------|-----------|
| P1 | 0 | 5 |
| P2 | 1 | 3 |
| P3 | 2 | 8 |
| P4 | 3 | 6 |

<details>
<summary>View Answer</summary>

**Gantt Chart:**
```
┌───────┬─────┬──────────┬────────┐
│  P1   │ P2  │    P3    │   P4   │
└───────┴─────┴──────────┴────────┘
0       5     8         16       22
```

**Calculation:**
- P1 waiting: 0
- P2 waiting: 5 - 1 = 4
- P3 waiting: 8 - 2 = 6
- P4 waiting: 16 - 3 = 13

**Average Waiting Time:** (0 + 4 + 6 + 13) / 4 = **5.75**

</details>

### Problem 2: SJF (non-preemptive)

Apply non-preemptive SJF to the same data from Problem 1.

<details>
<summary>View Answer</summary>

**Gantt Chart:**
```
Time 0: Execute P1(5) (only P1 arrived)
Time 5: P1 complete, select P2 from P2(3), P3(8), P4(6)
Time 8: P2 complete, select P4 from P3(8), P4(6)
Time 14: P4 complete, select P3
Time 22: P3 complete

┌───────┬─────┬────────┬──────────┐
│  P1   │ P2  │   P4   │    P3    │
└───────┴─────┴────────┴──────────┘
0       5     8       14        22
```

**Calculation:**
- P1 waiting: 0
- P2 waiting: 5 - 1 = 4
- P3 waiting: 14 - 2 = 12
- P4 waiting: 8 - 3 = 5

**Average Waiting Time:** (0 + 4 + 12 + 5) / 4 = **5.25**

</details>

### Problem 3: Round Robin

Apply RR (Time Quantum = 2) to the same data from Problem 1.

<details>
<summary>View Answer</summary>

**Gantt Chart:**
```
Time 0-2: P1(remaining 3)
Time 2-4: P2(remaining 1) - P2 arrived at time 1
Time 4-6: P3(remaining 6) - P3 arrived at time 2
Time 6-8: P4(remaining 4) - P4 arrived at time 3
Time 8-10: P1(remaining 1)
Time 10-11: P2 completes
Time 11-13: P3(remaining 4)
Time 13-15: P4(remaining 2)
Time 15-16: P1 completes
Time 16-18: P3(remaining 2)
Time 18-20: P4 completes
Time 20-22: P3 completes

┌──┬──┬──┬──┬──┬─┬──┬──┬─┬──┬──┬──┐
│P1│P2│P3│P4│P1│P2│P3│P4│P1│P3│P4│P3│
└──┴──┴──┴──┴──┴─┴──┴──┴─┴──┴──┴──┘
0  2  4  6  8 10 11 13 15 16 18 20 22
```

**Waiting Time Calculation:**
- P1: 16 - 0 - 5 = 11 (turnaround 16, burst 5)
- P2: 11 - 1 - 3 = 7 (turnaround 10, arrival 1, burst 3)
- P3: 22 - 2 - 8 = 12
- P4: 20 - 3 - 6 = 11

**Average Waiting Time:** (11 + 7 + 12 + 11) / 4 = **10.25**

</details>

### Problem 4: Priority (Preemptive)

Apply preemptive Priority scheduling to the following data.
(Lower number = Higher priority)

| Process | Arrival Time | Burst Time | Priority |
|---------|-------------|-----------|---------|
| P1 | 0 | 4 | 2 |
| P2 | 1 | 3 | 1 |
| P3 | 2 | 2 | 3 |
| P4 | 3 | 5 | 4 |

<details>
<summary>View Answer</summary>

**Gantt Chart:**
```
Time 0: P1(priority 2) executes
Time 1: P2(priority 1) arrives → Preempt P1, execute P2
Time 4: P2 completes, select P1(priority 2) from P1, P3(priority 3), P4(priority 4)
Time 7: P1 completes, select P3 from P3 and P4
Time 9: P3 completes, select P4
Time 14: P4 completes

┌─┬─────┬─────┬────┬────────────┐
│P1│  P2 │ P1  │ P3 │     P4     │
└─┴─────┴─────┴────┴────────────┘
0 1     4     7    9           14
```

**Waiting Time:**
- P1: 4-1 = 3 (preempted at time 1, resumed at time 4)
- P2: 0
- P3: 7 - 2 = 5
- P4: 9 - 3 = 6

**Average Waiting Time:** (3 + 0 + 5 + 6) / 4 = **3.5**

</details>

### Problem 5: Algorithm Selection

Select the most appropriate scheduling algorithm for each situation and explain why.

1. Maximize throughput in a batch processing system
2. Minimize response time in an interactive system
3. Process all processes fairly

<details>
<summary>View Answer</summary>

1. **SJF (or SRTF)**
   - Minimize average waiting time by processing short jobs first
   - Maximize throughput
   - Burst time prediction is often possible in batch systems

2. **Round Robin**
   - Quickly allocate CPU time to all processes
   - Short response time
   - Appropriate Time Quantum selection is important

3. **Round Robin** or **FCFS**
   - Provide equal opportunity to all processes
   - FCFS: Fair processing in arrival order
   - RR: Equal time allocation to each process
   - No starvation

</details>

---

## Hands-On Exercises

### Exercise 1: Scheduling Algorithm Comparison

Run `examples/OS_Theory/05_scheduling_sim.py` and analyze the output.

**Tasks:**
1. Add a new process set: P1(0,10), P2(2,5), P3(4,3), P4(6,1), P5(8,7). Run all four algorithms and rank them by average waiting time
2. Modify `round_robin()` to accept different quantum values (1, 2, 4, 8). Plot how average waiting time changes with quantum size
3. For SJF, the OS doesn't know burst times in advance. Implement exponential averaging prediction: `τ_{n+1} = α × t_n + (1-α) × τ_n` with α=0.5

### Exercise 2: Gantt Chart Visualization

Enhance the scheduling simulator with a graphical Gantt chart:

**Tasks:**
1. Using matplotlib (or ASCII art), create a color-coded Gantt chart where each process has a unique color
2. Add a timeline showing key events (arrivals, completions, preemptions)
3. Display per-process metrics below the chart: wait time, turnaround time, response time

### Exercise 3: Starvation Demonstration

Design an experiment that demonstrates starvation in SJF and Priority scheduling:

**Tasks:**
1. Create a workload where a long process never gets to run under SJF because short processes keep arriving
2. Implement aging: increase the priority of waiting processes by 1 every 10 time units
3. Show that aging eliminates starvation by comparing the maximum waiting time with and without aging

---

## Exercises

### Exercise 1: FCFS vs SJF Comparison

Given the following process set, compute all metrics for both **FCFS** and **non-preemptive SJF**:

| Process | Arrival Time | CPU Burst |
|---------|-------------|-----------|
| P1 | 0 | 8 |
| P2 | 1 | 3 |
| P3 | 2 | 6 |
| P4 | 3 | 1 |
| P5 | 4 | 4 |

For each algorithm:
1. Draw the Gantt chart
2. Calculate waiting time, turnaround time, and response time for each process
3. Calculate the average waiting time and average turnaround time
4. Identify whether the convoy effect occurs in FCFS. If so, which process is the cause?

### Exercise 2: SRTF Analysis

Use the same process set from Exercise 1 and apply **SRTF (Shortest Remaining Time First)**.

1. Draw the Gantt chart, marking each preemption point
2. Compare the average waiting time of SRTF vs non-preemptive SJF. Which is lower and why?
3. At time t=3, what is the remaining time of the currently running process? Does preemption occur? Show your calculation.
4. Name one disadvantage of SRTF that does not apply to non-preemptive SJF.

### Exercise 3: Round Robin Time Quantum Trade-offs

Apply **Round Robin** to the following process set with time quantum q=2, then again with q=5.

| Process | Arrival Time | CPU Burst |
|---------|-------------|-----------|
| P1 | 0 | 6 |
| P2 | 1 | 4 |
| P3 | 2 | 8 |
| P4 | 3 | 3 |

1. Draw the Gantt chart for q=2 and q=5
2. Calculate average waiting time for each
3. Count the number of context switches for each
4. Fill in the table and explain the trade-off:

| Quantum | Avg Waiting Time | Context Switches | Behaves more like |
|---------|-----------------|------------------|-------------------|
| q=2 | | | FCFS or SRTF? |
| q=5 | | | FCFS or SRTF? |

### Exercise 4: Starvation and Aging

A system uses non-preemptive **priority scheduling** (lower number = higher priority). Processes arrive continuously. At t=0, the ready queue contains:

| Process | CPU Burst | Priority |
|---------|-----------|----------|
| P1 | 10 | 5 (lowest) |
| P2 | 2 | 1 |
| P3 | 3 | 2 |
| P4 | 5 | 3 |

New short processes with priority 1 arrive every 4 time units. Without aging:

1. What happens to P1? At what time, if ever, does it get CPU access?
2. Implement an aging policy: every 8 time units that a process waits, its priority improves by 1. With this policy, at what time does P1's priority reach 1?
3. Describe a real-world situation where starvation in scheduling is a critical problem.

### Exercise 5: Algorithm Selection

For each system described below, choose the **most appropriate** scheduling algorithm from {FCFS, SJF, SRTF, Priority, Round Robin}. Justify your choice by referencing the specific strengths and weaknesses of each algorithm.

1. A print server that processes documents in the order they are submitted, where simplicity is valued over optimization
2. A real-time control system where time-critical tasks must run before lower-priority housekeeping tasks
3. A shared university computing cluster where interactive users expect quick feedback within a few seconds
4. A batch processing system with accurate historical CPU burst data where overall throughput is the only goal
5. An OS kernel's process scheduler that must balance both interactive and background processes fairly

---

## Next Steps

- [06_Advanced_Scheduling.md](./06_Advanced_Scheduling.md) - MLFQ, Multiprocessor Scheduling, Real-time Scheduling

---

## References

- [OSTEP - Scheduling: The Multi-Level Feedback Queue](https://pages.cs.wisc.edu/~remzi/OSTEP/cpu-sched-mlfq.pdf)
- [Operating System Concepts - Chapter 5](https://www.os-book.com/)
- [Process Scheduling Simulator](https://github.com/topics/process-scheduling-simulator)
