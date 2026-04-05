# LLVM Backend

**Previous**: [20. LLVM IR Introduction](./20_LLVM_IR_Introduction.md) | **Next**: [22. JIT Compilation](./22_JIT_Compilation.md)

---

The LLVM backend transforms LLVM IR into machine code for a specific target architecture. This process involves multiple stages: instruction selection (translating IR into target instructions), register allocation, instruction scheduling, and final code emission. Understanding the backend is crucial for writing custom compiler passes, adding new target architectures, or diagnosing performance issues at the machine code level.

This lesson covers the LLVM pass pipeline, how to write custom optimization passes, the SelectionDAG framework, instruction selection algorithms, and backend code generation.

**Difficulty**: ⭐⭐⭐⭐⭐

**Prerequisites**: [20. LLVM IR Introduction](./20_LLVM_IR_Introduction.md), [11. Code Generation](./11_Code_Generation.md), [19. Register Allocation](./19_Register_Allocation.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe the LLVM pass pipeline from IR to machine code
2. Distinguish between analysis passes, transformation passes, and machine passes
3. Write a custom LLVM IR optimization pass in C++
4. Explain how SelectionDAG converts IR to target instructions
5. Understand instruction selection via pattern matching and tiling
6. Trace the backend pipeline: SelectionDAG -> MachineInstr -> MCInst -> object code

---

## Table of Contents

1. [The LLVM Pass Pipeline](#1-the-llvm-pass-pipeline)
2. [Writing Custom Passes](#2-writing-custom-passes)
3. [The New Pass Manager](#3-the-new-pass-manager)
4. [SelectionDAG Overview](#4-selectiondag-overview)
5. [Instruction Selection](#5-instruction-selection)
6. [Register Allocation in LLVM](#6-register-allocation-in-llvm)
7. [Instruction Scheduling](#7-instruction-scheduling)
8. [Machine Code Emission](#8-machine-code-emission)
9. [TableGen and Target Description](#9-tablegen-and-target-description)
10. [Summary](#10-summary)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. The LLVM Pass Pipeline

### 1.1 Overview

LLVM compiles code through a series of **passes**, each performing a specific analysis or transformation:

```
LLVM IR  -->  [IR Passes]  -->  [Codegen]  -->  Machine Code

IR Passes:                    Codegen Pipeline:
- Constant propagation        - Instruction Selection (SelectionDAG/GlobalISel)
- Dead code elimination       - Register Allocation
- Loop unrolling             - Instruction Scheduling
- Inlining                   - Frame Lowering
- GVN                        - MC Emission
- LICM
- Vectorization
```

### 1.2 Pass Types

| Pass Type | Operates On | Example |
|-----------|-------------|---------|
| **ModulePass** | Entire module | Inlining, LTO |
| **FunctionPass** | Single function | Mem2Reg, ADCE |
| **LoopPass** | Single loop | LICM, Loop Unroll |
| **BasicBlockPass** | Single block | Instruction combining |
| **MachineFunctionPass** | Machine function | Register allocation |

### 1.3 Viewing the Pipeline

```bash
# Show all passes that run at -O2
clang -O2 -mllvm -debug-pass=Structure example.c -c 2>&1 | head -50

# With the new pass manager
clang -O2 -mllvm -print-pipeline-passes example.c -c

# Dump IR after each pass
clang -O2 -mllvm -print-after-all example.c -c
```

---

## 2. Writing Custom Passes

### 2.1 A Simple Function Pass (Legacy)

```cpp
// CountInstructions.cpp - A pass that counts instructions per function
#include "llvm/IR/Function.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct CountInstructions : public FunctionPass {
    static char ID;
    CountInstructions() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
        unsigned count = 0;
        for (auto &BB : F) {
            count += BB.size();
        }
        errs() << "Function " << F.getName()
               << " has " << count << " instructions\n";
        return false;  // We didn't modify the function
    }
};
}

char CountInstructions::ID = 0;
static RegisterPass<CountInstructions> X(
    "count-inst", "Count instructions in each function");
```

### 2.2 Building and Running

```bash
# Build as a shared library
clang++ -shared -fPIC -o CountInst.so CountInstructions.cpp \
    $(llvm-config --cxxflags --ldflags --libs core)

# Run the pass
opt -load ./CountInst.so -count-inst < input.bc > /dev/null
```

### 2.3 A Transformation Pass

```cpp
// ConstantMultiplyToShift.cpp
// Transform: x * (power of 2) -> x << log2(power)
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"
#include "llvm/Pass.h"

using namespace llvm;

namespace {
struct MulToShift : public FunctionPass {
    static char ID;
    MulToShift() : FunctionPass(ID) {}

    bool runOnFunction(Function &F) override {
        bool changed = false;
        for (auto &BB : F) {
            for (auto it = BB.begin(); it != BB.end(); ) {
                Instruction *I = &*it++;

                // Look for: %r = mul %x, <constant power of 2>
                if (auto *Mul = dyn_cast<BinaryOperator>(I)) {
                    if (Mul->getOpcode() != Instruction::Mul) continue;

                    auto *C = dyn_cast<ConstantInt>(Mul->getOperand(1));
                    if (!C) continue;

                    APInt val = C->getValue();
                    if (!val.isPowerOf2()) continue;

                    // Replace with shift
                    unsigned shift_amt = val.logBase2();
                    auto *ShiftConst = ConstantInt::get(C->getType(), shift_amt);
                    auto *Shift = BinaryOperator::Create(
                        Instruction::Shl, Mul->getOperand(0), ShiftConst,
                        Mul->getName(), Mul);
                    Mul->replaceAllUsesWith(Shift);
                    Mul->eraseFromParent();
                    changed = true;
                }
            }
        }
        return changed;
    }
};
}

char MulToShift::ID = 0;
static RegisterPass<MulToShift> Y("mul-to-shift",
    "Replace multiply by power of 2 with shift");
```

---

## 3. The New Pass Manager

### 3.1 Overview

LLVM's New Pass Manager (NPM) replaces the legacy pass manager with better support for analysis caching and pass ordering:

```cpp
// NewPM style pass
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

struct CountInstructionsPass : public PassInfoMixin<CountInstructionsPass> {
    PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
        unsigned count = 0;
        for (auto &BB : F) {
            count += BB.size();
        }
        errs() << "Function " << F.getName()
               << ": " << count << " instructions\n";
        return PreservedAnalyses::all();
    }
};

// Plugin registration
extern "C" ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
    return {
        LLVM_PLUGIN_API_VERSION, "CountInstructions", "0.1",
        [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                    if (Name == "count-inst") {
                        FPM.addPass(CountInstructionsPass());
                        return true;
                    }
                    return false;
                });
        }
    };
}
```

### 3.2 Running with New PM

```bash
# Run with new pass manager
opt --load-pass-plugin=./CountInst.so --passes="count-inst" input.ll -S

# Combine with built-in passes
opt --load-pass-plugin=./CountInst.so \
    --passes="default<O2>,count-inst" input.ll -S
```

---

## 4. SelectionDAG Overview

### 4.1 From IR to Machine Instructions

The SelectionDAG is LLVM's primary instruction selection framework. It represents a basic block as a directed acyclic graph:

```
LLVM IR:                     SelectionDAG:
  %a = load i32, ptr %p         [load]
  %b = add i32 %a, 1              |
  store i32 %b, ptr %q        [add, 1]
                                   |
                               [store]
```

### 4.2 SelectionDAG Phases

```
1. Build initial DAG (from LLVM IR)
       |
2. Optimize DAG (combine/simplify)
       |
3. Legalize types (convert unsupported types)
       |
4. Legalize operations (convert unsupported ops)
       |
5. Select instructions (pattern match to target instructions)
       |
6. Schedule instructions (determine order)
       |
7. Emit MachineInstrs
```

### 4.3 Viewing the DAG

```bash
# Dump SelectionDAG before and after selection
llc -view-dag-combine1-dags example.ll    # After first combine
llc -view-isel-dags example.ll            # Before selection
llc -view-sched-dags example.ll           # After selection
```

---

## 5. Instruction Selection

### 5.1 Pattern Matching

LLVM uses TableGen patterns to match DAG nodes to target instructions:

```
// In a .td (TableGen) file for x86:
// Pattern: add i32 -> ADD32rr
def ADD32rr : I<0x01, ...> {
    let Pattern = [(set GR32:$dst, (add GR32:$src1, GR32:$src2))];
}

// Pattern: add i32 with immediate -> ADD32ri
def ADD32ri : I<0x81, ...> {
    let Pattern = [(set GR32:$dst, (add GR32:$src1, imm:$src2))];
}

// Complex pattern: multiply-add -> LEA
// lea eax, [ebx + ecx*4 + 8]
def LEA32r : I<0x8D, ...> {
    let Pattern = [(set GR32:$dst,
                     (add (add GR32:$base, (shl GR32:$index, (i8 2))),
                          (i32 8)))];
}
```

### 5.2 Legalization

Targets may not support all types and operations. Legalization expands, promotes, or custom-lowers them:

```
Target: x86-32 (no native i64 arithmetic)

i64 add --> expand to:
  add_lo = add i32 %a_lo, %b_lo
  carry  = carry from add_lo
  add_hi = add i32 %a_hi, %b_hi
  add_hi = add i32 add_hi, carry

i8 add --> promote to:
  extend i8 to i32
  add i32
  truncate i32 to i8
```

### 5.3 GlobalISel (Alternative Pipeline)

GlobalISel is LLVM's newer instruction selection framework, operating at the function level rather than per-basic-block:

```
LLVM IR --> IRTranslator --> Legalizer --> RegBankSelect --> InstructionSelect
                  |              |              |                    |
            Generic MIR    Legal MIR    RegBank MIR         Target MIR
```

Advantages over SelectionDAG:
- Faster compilation (no DAG construction)
- Better cross-block optimization opportunities
- Simpler to extend for new targets

---

## 6. Register Allocation in LLVM

### 6.1 LLVM's Register Allocators

LLVM provides multiple register allocation strategies:

```bash
# Select register allocator
llc -regalloc=greedy example.ll    # Default: greedy (Briggs-like)
llc -regalloc=basic example.ll     # Simple baseline
llc -regalloc=fast example.ll      # Linear scan (for -O0)
llc -regalloc=pbqp example.ll      # PBQP-based (experimental)
```

### 6.2 The Greedy Allocator

The default **greedy** allocator is a priority-based, region-splitting allocator:

1. Sort live ranges by spill weight (priority)
2. For each live range, try to assign a register
3. If no register available, try to evict a lower-priority range
4. If eviction fails, split the live range and retry
5. As last resort, spill to stack

---

## 7. Instruction Scheduling

### 7.1 Purpose

Instruction scheduling reorders instructions to:
- Hide latencies (e.g., issue independent instructions while waiting for a load)
- Reduce register pressure
- Exploit instruction-level parallelism

### 7.2 List Scheduling

LLVM uses list scheduling with various heuristics:

```
# Before scheduling:
1: load r1, [addr1]    ; latency 4
2: add r2, r1, 1       ; depends on 1
3: load r3, [addr2]    ; independent
4: add r4, r3, 2       ; depends on 3

# After scheduling (interleave to hide latencies):
1: load r1, [addr1]    ; cycle 0
3: load r3, [addr2]    ; cycle 1 (independent, issued during load 1)
2: add r2, r1, 1       ; cycle 4 (load 1 complete)
4: add r4, r3, 2       ; cycle 5 (load 3 complete)
```

---

## 8. Machine Code Emission

### 8.1 MC Layer

The **MC (Machine Code)** layer is LLVM's low-level code representation:

```
MachineInstr (high-level machine instruction)
    ↓
MCInst (abstract machine instruction)
    ↓
MCCodeEmitter (binary encoding)
    ↓
Object file (.o) or Assembly (.s)
```

### 8.2 Output Formats

```bash
# Assembly output
llc example.ll -o example.s

# Object file
llc -filetype=obj example.ll -o example.o

# Show encoding details
llc -show-mc-encoding example.ll
```

---

## 9. TableGen and Target Description

### 9.1 What is TableGen?

TableGen is a domain-specific language used to describe target architectures:

```
// Define a register class
def GR32 : RegisterClass<"X86", [i32], 32,
    (add EAX, ECX, EDX, EBX, ESI, EDI, EBP, ESP)>;

// Define an instruction
def ADDrr : I<0x01, MRMDestReg,
    (outs GR32:$dst), (ins GR32:$src1, GR32:$src2),
    "add\t$dst, $src2",
    [(set GR32:$dst, (add GR32:$src1, GR32:$src2))]>;
```

### 9.2 Target Description Files

A target backend consists of several `.td` files:

| File | Describes |
|------|-----------|
| `*RegisterInfo.td` | Registers and register classes |
| `*InstrInfo.td` | Instruction definitions |
| `*InstrFormats.td` | Instruction encoding formats |
| `*CallingConv.td` | Calling conventions |
| `*Schedule.td` | Instruction latencies |

---

## 10. Summary

- LLVM uses a **pass pipeline** that transforms IR through analysis and optimization passes to machine code
- The **New Pass Manager** provides better analysis caching and composability
- **SelectionDAG** converts IR basic blocks into target machine instructions via pattern matching
- **GlobalISel** is the newer, faster alternative to SelectionDAG
- **Register allocation** in LLVM defaults to the greedy allocator with live range splitting
- **Instruction scheduling** hides latencies and exploits ILP
- **TableGen** describes target architectures declaratively
- The **MC layer** handles final code emission to assembly or object files

---

## 11. Exercises

1. **Write a pass**: Implement an LLVM pass that counts the number of phi nodes in each function.

2. **Transformation pass**: Write a pass that replaces `x / 2` with `x >> 1` for unsigned integers.

3. **Inspect SelectionDAG**: Compile a small C function and dump the SelectionDAG at each stage. Identify pattern matches.

4. **Compare allocators**: Compile a function with both `-regalloc=greedy` and `-regalloc=fast` and compare spill counts.

5. **Read TableGen**: Find the x86 TableGen definition for the `MOV` instruction and trace how it maps IR to machine code.

---

## 12. References

1. LLVM Writing an LLVM Pass: https://llvm.org/docs/WritingAnLLVMPass.html
2. LLVM New Pass Manager: https://llvm.org/docs/WritingAnLLVMNewPMPass.html
3. LLVM Code Generator: https://llvm.org/docs/CodeGenerator.html
4. LLVM SelectionDAG: https://llvm.org/docs/SelectionDAG.html
5. LLVM TableGen: https://llvm.org/docs/TableGen/

---

**Previous**: [20. LLVM IR Introduction](./20_LLVM_IR_Introduction.md) | **Next**: [22. JIT Compilation](./22_JIT_Compilation.md)
