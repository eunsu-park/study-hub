# Pipelining

**Previous**: [10_Assembly_Language_Basics.md](./10_Assembly_Language_Basics.md) | **Next**: [12_Branch_Prediction.md](./12_Branch_Prediction.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how pipelining increases instruction throughput without reducing latency
2. Describe the classic 5-stage pipeline (IF, ID, EX, MEM, WB)
3. Identify data hazards, control hazards, and structural hazards
4. Explain forwarding (bypassing) as a data hazard solution
5. Explain branch prediction as a control hazard mitigation
6. Calculate pipeline speedup and analyze why it is less than the ideal N times

---

Pipelining is the single most impactful technique in processor design. Without it, a CPU would execute one instruction at a time, wasting most of its hardware. Modern processors pipeline dozens of stages, achieving throughput close to one instruction per clock cycle -- but hazards (data dependencies, branches) constantly threaten this ideal. Understanding pipelining explains why instruction order matters for performance.

## Table of Contents

1. [Pipelining Concept](#1-pipelining-concept)
2. [5-Stage Pipeline](#2-5-stage-pipeline)
3. [Pipeline Performance](#3-pipeline-performance)
4. [Pipeline Hazards](#4-pipeline-hazards)
5. [Hazard Resolution Techniques](#5-hazard-resolution-techniques)
6. [Practice Problems](#6-practice-problems)

---

> Pipelining is like a car assembly line. Instead of one worker building an entire car from start to finish (sequential execution), the assembly line has specialized stations -- one installs the engine, the next adds wheels, the next paints the body. Each station works on a different car simultaneously. The time to build one car does not change, but the factory produces a finished car much more frequently. CPU pipelining applies the same principle: different stages of the processor work on different instructions at the same time.

## 1. Pipelining Concept

### Basic Idea

```
Non-pipelined (Sequential Execution):
┌─────┐     ┌─────┐     ┌─────┐
│ I1  │────▶│ I2  │────▶│ I3  │
└─────┘     └─────┘     └─────┘
  5ns         5ns         5ns      Total 15ns

Pipelined (Parallel Execution):
Time:  1ns   2ns   3ns   4ns   5ns   6ns   7ns
I1:   [IF]─[ID]─[EX]─[MEM]─[WB]
I2:        [IF]─[ID]─[EX]─[MEM]─[WB]
I3:             [IF]─[ID]─[EX]─[MEM]─[WB]
                                     Total 7ns
```

### Laundry Analogy

```
Non-pipelined:
Wash1 ──▶ Dry1 ──▶ Wash2 ──▶ Dry2 ──▶ Wash3 ──▶ Dry3

Pipelined:
Time:   1    2    3    4    5    6
Wash1   ■
Dry1        ■
Wash2       ■
Dry2             ■
Wash3            ■
Dry3                  ■

3x faster!
```

---

## 2. 5-Stage Pipeline

### MIPS 5-Stage Pipeline

```
┌────────────────────────────────────────────────────────────┐
│                    5-Stage Pipeline                         │
├───────┬───────┬───────┬───────┬───────┐                    │
│  IF   │  ID   │  EX   │  MEM  │  WB   │                    │
│(Fetch)│(Decode│(Exec) │(Memory│(Write │                    │
│       │       │       │Access)│ Back) │                    │
└───────┴───────┴───────┴───────┴───────┘                    │
└────────────────────────────────────────────────────────────┘
```

### Stage Descriptions

| Stage | Name | Operation |
|-------|------|-----------|
| IF | Instruction Fetch | Fetch instruction from memory at PC |
| ID | Instruction Decode | Decode instruction, read registers |
| EX | Execute | Perform ALU operation, address calculation |
| MEM | Memory Access | Read/write memory (load/store) |
| WB | Write Back | Store result to register |

### Pipeline Registers

```
┌─────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌─────────┐
│ IF  │──▶│ IF/ID   │──▶│ ID/EX   │──▶│ EX/MEM   │──▶│ MEM/WB  │
└─────┘   │Register │   │Register │   │Register  │   │Register │
          └─────────┘   └─────────┘   └──────────┘   └─────────┘

Pipeline Registers: Temporarily store data between stages
- Data is passed to the next stage every clock cycle
```

### Pipeline Usage by Instruction Type

```
Stage usage by instruction type:

R-type (add, sub):
IF ─▶ ID ─▶ EX ─▶ --- ─▶ WB

Load (lw):
IF ─▶ ID ─▶ EX ─▶ MEM ─▶ WB

Store (sw):
IF ─▶ ID ─▶ EX ─▶ MEM ─▶ ---

Branch (beq):
IF ─▶ ID ─▶ EX ─▶ --- ─▶ ---
```

---

## 3. Pipeline Performance

### Ideal Speedup

```
Speedup = Number of pipeline stages (ideal)

5-stage pipeline → Maximum 5x speedup

In practice:
- Pipeline fill/drain time
- Stalls due to hazards
- Stage imbalance
```

### Throughput Calculation

```
Throughput = Number of instructions / Time

Non-pipelined:
- 1 instruction / 5 cycles

Pipelined (ideal):
- 1 instruction / 1 cycle (full pipeline state)

CPI (Cycles Per Instruction):
- Ideal: CPI = 1
- Actual: CPI = 1 + stall cycles
```

### Example: Executing 100 Instructions

```
Non-pipelined:
Time = 100 × 5 = 500 cycles

5-stage pipeline:
Time = 5 + (100 - 1) = 104 cycles
      ↑first instr.  ↑remaining instructions

Speedup = 500 / 104 ≈ 4.8x
```

---

## 4. Pipeline Hazards

### Hazard Types

```
┌─────────────────────────────────────────────────────────┐
│                   Pipeline Hazards                       │
├─────────────────┬─────────────────┬─────────────────────┤
│   Structural    │      Data       │      Control        │
│    Hazard       │     Hazard      │      Hazard         │
├─────────────────┼─────────────────┼─────────────────────┤
│ Hardware        │ Problems due    │ Problems due        │
│ resource        │ to data         │ to branch           │
│ conflicts       │ dependencies    │ instructions        │
└─────────────────┴─────────────────┴─────────────────────┘
```

### 4.1 Structural Hazards

```
Problem: Trying to use the same hardware resource simultaneously

Example: Single memory usage
Cycle 4:
- Instruction 1: MEM stage (data memory access)
- Instruction 4: IF stage (instruction memory access)

    I1: IF─ID─EX─MEM─WB
    I4:          IF ← Conflict!

Solution: Harvard architecture (separate instruction/data memory)
```

### 4.2 Data Hazards

```
Three types:

1. RAW (Read After Write) - Most common
   # RAW hazard: sub reads $s0 before add's WB stage writes it back
   # This is the most frequent hazard because most instructions consume a previous result
   add $s0, $t0, $t1    # Write to $s0
   sub $t2, $s0, $t3    # Read $s0 ← Not written yet!

2. WAR (Write After Read)
   # WAR hazard: rare in in-order pipelines because reads (ID) always precede writes (WB)
   # Becomes a real problem only in out-of-order execution (see Lesson 13)
   sub $t2, $s0, $t3    # Read $s0
   add $s0, $t0, $t1    # Write to $s0

3. WAW (Write After Write)
   # WAW hazard: only possible in pipelines that allow multiple writes in flight
   # In a simple 5-stage pipeline, instructions write back in order, so WAW cannot occur
   add $s0, $t0, $t1    # Write to $s0
   sub $s0, $t2, $t3    # Write to $s0
```

### RAW Hazard Example

```
add $s0, $t0, $t1
sub $t2, $s0, $t3

# Why this is a problem: add computes $s0 in EX (cycle 3) but only writes it to
# the register file in WB (cycle 5). Meanwhile sub needs $s0 as an ALU input in
# its EX stage (cycle 4) — one cycle before the value is available in the register file.
Time:    1    2    3    4    5    6    7
add:    IF   ID   EX  MEM  [WB] ← $s0 written
sub:         IF   ID  [EX] ← $s0 needed!
                       ↑
                   Problem occurs!

$s0 is written in cycle 5,
but sub needs $s0 in cycle 4
```

### 4.3 Control Hazards

```
# Why control hazards exist: the branch outcome is not known until EX (cycle 3),
# but the pipeline has already fetched the next 2 instructions (add, sub) assuming
# no branch. If the branch IS taken, those 2 instructions are wrong — they must be
# flushed, wasting 2 cycles. This is the "branch penalty."
beq $t0, $t1, target    # Branch decision
add $t2, $t3, $t4       # Should this execute?
sub $t5, $t6, $t7       # Should this execute?

Time:    1    2    3    4    5
beq:    IF   ID  [EX] ← Branch decided
add:         IF   ID   ← Wrong instruction?
sub:              IF   ← Wrong instruction?

Next instructions already entered pipeline before branch decision
```

---

## 5. Hazard Resolution Techniques

### 5.1 Stalling

```
Pause pipeline by inserting bubbles (NOPs)

add $s0, $t0, $t1
sub $t2, $s0, $t3

Time:    1    2    3    4    5    6    7    8    9
add:    IF   ID   EX  MEM  WB
        --- stall ---
        --- stall ---
sub:              IF   ID   EX  MEM  WB

2 cycle stall occurs → Performance degradation
```

### 5.2 Forwarding/Bypassing

```
Pass ALU result directly to next instruction

add $s0, $t0, $t1
sub $t2, $s0, $t3

Time:    1    2    3    4    5    6
add:    IF   ID  [EX] MEM  WB
                  │
                  └─────▶ Forwarding
sub:         IF   ID  [EX] MEM  WB
                       ↑
                    Use $s0 value

Execution possible without stalls!
```

### Forwarding Paths

```
┌─────────────────────────────────────────────────────────┐
│                    Forwarding Unit                       │
│                                                         │
│   EX/MEM.ALUResult ───────────────┐                     │
│                                   ▼                     │
│   MEM/WB.ALUResult ─────────────▶ MUX ──▶ ALU input     │
│                                   ▲                     │
│   ID/EX.RegisterRs ───────────────┘                     │
└─────────────────────────────────────────────────────────┘

Forwarding conditions:
1. EX/MEM.RegisterRd == ID/EX.RegisterRs
   # Why EX/MEM: the producing instruction just finished EX — its ALU result sits
   # in the EX/MEM pipeline register. Forward it to the consuming instruction's
   # ALU input before EX begins, avoiding a 1-cycle stall.
2. MEM/WB.RegisterRd == ID/EX.RegisterRs
   # Why MEM/WB: the producer is one stage further along (finished MEM).
   # If condition 1 didn't match (e.g., an instruction in between), this
   # catches the 2-cycle-old result — still faster than waiting for WB.
```

### 5.3 Load-Use Hazard

```
Cases that cannot be resolved by forwarding:

# Why forwarding fails here: lw produces its data at the END of MEM stage,
# but add needs it at the START of its EX stage — which is the same cycle.
# The data literally does not exist yet when the ALU needs it, so a 1-cycle
# stall is unavoidable. This is why compilers try to schedule an unrelated
# instruction between a load and its consumer ("load delay slot scheduling").
lw  $s0, 0($t0)     # Load from memory
add $t2, $s0, $t3   # Use immediately

Time:    1    2    3    4    5    6    7
lw:     IF   ID   EX  [MEM] WB
                       │
                       └───▶ Data available
add:         IF   ID  stall [EX] MEM  WB
                   ↑
              Need data but not available yet

1 cycle stall mandatory (Load-Use Stall)
```

### 5.4 Branch Prediction

```
Static Prediction:
- Always Not Taken: Predict no branch
- Always Taken: Predict branch taken
- BTFN: Backward Taken, Forward Not Taken

Dynamic Prediction:
- Predict based on branch history
- Covered in detail in next lesson
```

### 5.5 Delayed Branch

```
Place always-executed instruction in slot after branch instruction

# Why delayed branch works: the instruction after beq ALWAYS executes (the pipeline
# has already fetched it before the branch resolves). Instead of wasting that slot
# with a NOP, the compiler fills it with a useful instruction that is needed regardless
# of branch outcome — turning a penalty cycle into productive work.
beq $t0, $t1, target
add $t2, $t3, $t4    # Delay slot (always executed)
...
target:
sub $t5, $t6, $t7

Compiler places branch-independent instruction in delay slot
```

---

## 6. Practice Problems

### Basic Problems

1. What are the names and roles of each stage in a 5-stage pipeline?

2. How many cycles are needed to execute 100 instructions in a 5-stage pipeline? (Assume no hazards)

3. Which of the following is NOT a type of data hazard?
   - (a) RAW
   - (b) RAR
   - (c) WAR
   - (d) WAW

### Hazard Analysis

4. Find the data hazards in the following code:
```assembly
add $s0, $t0, $t1   # Produces $s0
sub $s1, $s0, $t2   # Reads $s0 (1 instr after add) and produces $s1
and $s2, $s0, $s1   # Reads $s0 (2 instr after add) and $s1 (1 instr after sub)
# Hint: check every producer-consumer pair — which reads happen before the write-back?
```

5. Which hazard cannot be resolved by forwarding?
```assembly
lw  $s0, 0($t0)     # Data available after MEM stage
add $t1, $s0, $t2   # Needs $s0 at start of EX — same cycle MEM finishes
# Hint: can the MEM output be forwarded in time for the next instruction's EX input?
```

### Performance Calculation

6. If 30% of 1000 instructions are branches, branch misprediction rate is 20%, and misprediction penalty is 3 cycles, what is the CPI?

<details>
<summary>Answers</summary>

1. IF (Fetch), ID (Decode), EX (Execute), MEM (Memory Access), WB (Write Back)

2. 5 + (100 - 1) = 104 cycles

3. (b) RAR - Read After Read is not a hazard

4.
- add → sub: RAW on $s0
- add → and: RAW on $s0
- sub → and: RAW on $s1

5. Load-Use hazard. Data is only available after MEM stage of lw, so 1 cycle stall is required

6.
- Number of branch instructions: 1000 × 0.3 = 300
- Mispredictions: 300 × 0.2 = 60
- Penalty cycles: 60 × 3 = 180
- CPI = 1 + 180/1000 = 1.18

</details>

---

## Next Steps

- [12_Branch_Prediction.md](./12_Branch_Prediction.md) - Dynamic Branch Prediction Techniques

---

## References

- Computer Organization and Design, Chapter 4 (Patterson & Hennessy)
- [Pipeline Visualization](https://www.youtube.com/watch?v=eVRdfl4zxfI)
- [MIPS Pipeline Simulator](http://www.cs.umd.edu/~meesh/411/mips-pipe/)
