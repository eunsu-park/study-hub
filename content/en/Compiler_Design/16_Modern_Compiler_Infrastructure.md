# Modern Compiler Infrastructure

**Previous**: [15. Interpreters and Virtual Machines](./15_Interpreters_and_Virtual_Machines.md)

---

Modern compilers are not monolithic programs built from scratch. They are assembled from reusable infrastructure -- intermediate representations, optimization passes, code generators, and tooling frameworks -- that can be shared across many languages. The LLVM project exemplifies this philosophy: dozens of languages (C, C++, Rust, Swift, Julia, Zig, and more) share the same optimizer and code generators.

This lesson explores the infrastructure that powers modern compilers: LLVM's architecture and IR, MLIR's multi-level approach, GCC's internals, domain-specific language (DSL) design, compiler construction tools, and advanced compilation techniques like PGO and LTO.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [09. Intermediate Representations](./09_Intermediate_Representations.md), [11. Code Generation](./11_Code_Generation.md), [12. Optimization -- Local and Global](./12_Optimization_Local_and_Global.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Describe LLVM's modular architecture and pass pipeline
2. Read and write LLVM IR, including types, instructions, and SSA form
3. Understand how to write an LLVM optimization pass
4. Explain MLIR's multi-level IR philosophy and dialects
5. Compare LLVM and GCC internal representations (GIMPLE, RTL)
6. Design and implement domain-specific languages (DSLs)
7. Use compiler construction tools (ANTLR, Tree-sitter)
8. Understand the Language Server Protocol (LSP)
9. Explain incremental compilation strategies
10. Apply profile-guided optimization (PGO) and link-time optimization (LTO)
11. Reason about compiler verification approaches

---

## Table of Contents

1. [LLVM Overview](#1-llvm-overview)
2. [LLVM IR in Detail](#2-llvm-ir-in-detail)
3. [Writing LLVM Passes](#3-writing-llvm-passes)
4. [MLIR: Multi-Level IR](#4-mlir-multi-level-ir)
5. [GCC Internals](#5-gcc-internals)
6. [Domain-Specific Languages](#6-domain-specific-languages)
7. [Compiler Construction Tools](#7-compiler-construction-tools)
8. [Language Server Protocol](#8-language-server-protocol)
9. [Incremental Compilation](#9-incremental-compilation)
10. [Profile-Guided Optimization](#10-profile-guided-optimization)
11. [Link-Time Optimization](#11-link-time-optimization)
12. [Compiler Verification](#12-compiler-verification)
13. [Summary](#13-summary)
14. [Exercises](#14-exercises)
15. [References](#15-references)

---

## 1. LLVM Overview

### 1.1 What is LLVM?

**LLVM** (originally "Low Level Virtual Machine," now just a name) is a collection of modular compiler and toolchain technologies. Its core idea is a well-defined intermediate representation (LLVM IR) that decouples language-specific frontends from target-specific backends.

```
Language Frontends:              LLVM Core:              Target Backends:
┌───────┐                                                ┌──────────┐
│ Clang │───┐                                        ┌──▶│  x86-64  │
│ (C/C++)│   │    ┌──────────┐   ┌──────────┐        │   └──────────┘
└───────┘   │    │          │   │          │        │   ┌──────────┐
┌───────┐   ├───▶│ LLVM IR  │──▶│Optimizer │──▶─────┼──▶│  ARM64   │
│ Rust  │───┤    │          │   │ (Passes) │        │   └──────────┘
│(rustc)│   │    └──────────┘   └──────────┘        │   ┌──────────┐
└───────┘   │                                        ├──▶│  RISC-V  │
┌───────┐   │                                        │   └──────────┘
│ Swift │───┤                                        │   ┌──────────┐
│       │   │                                        └──▶│  WASM    │
└───────┘   │                                            └──────────┘
┌───────┐   │
│ Julia │───┘
│       │
└───────┘
```

### 1.2 LLVM Architecture

LLVM consists of several key components:

| Component | Purpose |
|-----------|---------|
| **LLVM Core** | IR definition, optimization passes, code generation |
| **Clang** | C/C++/Objective-C frontend |
| **LLDB** | Debugger |
| **libc++** | C++ standard library |
| **compiler-rt** | Runtime support (sanitizers, profiling) |
| **LLD** | Linker |
| **MLIR** | Multi-Level IR framework |
| **Polly** | Polyhedral optimization pass |

### 1.3 The Three-Phase Design

```
Phase 1: Frontend          Phase 2: Optimizer       Phase 3: Backend
┌─────────────────┐       ┌──────────────────┐     ┌─────────────────┐
│                 │       │                  │     │                 │
│ Source Code     │       │ LLVM IR          │     │ Machine Code    │
│     │           │       │     │            │     │     │           │
│     ▼           │       │     ▼            │     │     ▼           │
│  Lexing         │       │ Analysis Passes  │     │ Instruction     │
│  Parsing        │       │ Transform Passes │     │ Selection       │
│  Semantic       │──────▶│                  │────▶│ Register Alloc  │
│  Analysis       │       │ Optimization     │     │ Instruction     │
│  IR Generation  │       │ Levels:          │     │ Scheduling      │
│                 │       │  -O0, -O1, -O2,  │     │ Code Emission   │
│                 │       │  -O3, -Os, -Oz   │     │                 │
└─────────────────┘       └──────────────────┘     └─────────────────┘
```

The beauty of this design: adding a new language requires only writing a frontend that generates LLVM IR. Adding a new target architecture requires only writing a backend. Both benefit from all existing optimizations.

### 1.4 Compiling with LLVM (Using Clang)

```bash
# Generate LLVM IR from C code
clang -S -emit-llvm hello.c -o hello.ll

# Optimize LLVM IR
opt -O2 hello.ll -o hello_opt.ll

# Compile LLVM IR to assembly
llc hello_opt.ll -o hello.s

# Or compile directly to object file
clang -c hello.c -O2 -o hello.o

# View the optimization pipeline
clang -O2 -mllvm -print-after-all hello.c 2>&1 | head -100
```

---

## 2. LLVM IR in Detail

### 2.1 IR Structure

LLVM IR is a typed, SSA-based intermediate representation. It exists in three isomorphic forms:
- **Human-readable text** (`.ll` files)
- **Dense binary encoding** (bitcode, `.bc` files)
- **In-memory data structures** (C++ objects)

A module (translation unit) contains:

```llvm
; Module-level structure
source_filename = "example.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; Global variables
@global_var = global i32 42, align 4
@hello_str = private constant [12 x i8] c"Hello World\00"

; Function declarations (external)
declare i32 @printf(i8* nocapture readonly, ...)

; Function definitions
define i32 @main() {
entry:
  %x = alloca i32, align 4
  store i32 10, i32* %x, align 4
  %val = load i32, i32* %x, align 4
  %result = add nsw i32 %val, 32
  ret i32 %result
}
```

### 2.2 Type System

LLVM IR has a rich type system:

```llvm
; Integer types
i1        ; 1-bit (boolean)
i8        ; 8-bit (byte/char)
i16       ; 16-bit (short)
i32       ; 32-bit (int)
i64       ; 64-bit (long)
i128      ; 128-bit

; Floating point types
half      ; 16-bit float
float     ; 32-bit float
double    ; 64-bit float

; Pointer type
ptr       ; opaque pointer (LLVM 15+)
i32*      ; typed pointer (legacy)

; Array type
[10 x i32]         ; Array of 10 i32s
[3 x [4 x float]]  ; 3x4 matrix of floats

; Structure types
{ i32, float, i8* }           ; Literal struct
%struct.Point = type { i32, i32 }  ; Named struct

; Vector type (SIMD)
<4 x float>    ; 4-element float vector
<8 x i32>      ; 8-element int vector

; Function type
i32 (i32, i32)      ; Function taking two i32s, returning i32
void (i8*, ...)     ; Variadic function
```

### 2.3 SSA and Instructions

All values in LLVM IR are in **Static Single Assignment (SSA)** form: each variable is defined exactly once. Phi nodes ($\phi$-functions) merge values at control flow join points.

```llvm
; Arithmetic instructions
%sum = add i32 %a, %b          ; Integer addition
%diff = sub i32 %a, %b         ; Integer subtraction
%prod = mul i32 %a, %b         ; Integer multiplication
%quot = sdiv i32 %a, %b        ; Signed integer division
%rem = srem i32 %a, %b         ; Signed remainder

%fsum = fadd double %x, %y     ; Floating-point addition
%fprod = fmul float %x, %y     ; Floating-point multiplication

; nsw/nuw flags: no signed/unsigned wrap (enables optimizations)
%safe_add = add nsw i32 %a, %b

; Comparison
%cmp = icmp eq i32 %a, %b      ; Integer compare (eq, ne, slt, sgt, sle, sge)
%fcmp = fcmp olt double %x, %y ; Float compare (olt, ogt, oeq, ...)

; Bitwise
%and = and i32 %a, %b
%or = or i32 %a, %b
%xor = xor i32 %a, %b
%shl = shl i32 %a, 2           ; Shift left by 2

; Conversion
%ext = sext i32 %a to i64      ; Sign-extend i32 to i64
%trunc = trunc i64 %b to i32   ; Truncate i64 to i32
%fp = sitofp i32 %a to double  ; Signed int to floating point
%int = fptosi double %x to i32 ; Floating point to signed int
%cast = bitcast i32* %p to i8* ; Reinterpret bits (same size)
```

### 2.4 Memory Instructions

```llvm
; Stack allocation
%ptr = alloca i32, align 4         ; Allocate 4 bytes on stack
%arr = alloca [100 x i32], align 16 ; Array on stack

; Load and store
store i32 42, i32* %ptr, align 4    ; *ptr = 42
%val = load i32, i32* %ptr, align 4 ; val = *ptr

; GEP (GetElementPtr) -- address computation without memory access
; This is one of the most important (and confusing) LLVM instructions
%struct.Point = type { i32, i32 }

%p = alloca %struct.Point
; Get pointer to the second field (y):
%y_ptr = getelementptr %struct.Point, %struct.Point* %p, i32 0, i32 1
; First index (i32 0): which struct in an array (0th)
; Second index (i32 1): which field (1 = second field)
```

### 2.5 Control Flow

```llvm
; Unconditional branch
br label %target

; Conditional branch
%cond = icmp slt i32 %i, %n
br i1 %cond, label %loop_body, label %loop_exit

; Phi nodes (SSA merge at join points)
define i32 @abs(i32 %x) {
entry:
  %is_neg = icmp slt i32 %x, 0
  br i1 %is_neg, label %negative, label %done

negative:
  %neg_x = sub i32 0, %x
  br label %done

done:
  ; Phi node: value depends on which predecessor we came from
  %result = phi i32 [ %x, %entry ], [ %neg_x, %negative ]
  ret i32 %result
}

; Switch
switch i32 %val, label %default [
  i32 0, label %case0
  i32 1, label %case1
  i32 2, label %case2
]
```

### 2.6 Function Calls

```llvm
; Direct call
%result = call i32 @add(i32 %a, i32 %b)

; Call with attributes
%result = call i32 @pure_func(i32 %x) nounwind readnone

; Tail call (enables tail call optimization)
%result = tail call i32 @recursive_func(i32 %n)

; Invoke (call that may throw an exception)
%result = invoke i32 @may_throw(i32 %x)
          to label %normal unwind label %exception
```

### 2.7 Complete LLVM IR Example

```llvm
; Factorial function in LLVM IR
define i32 @factorial(i32 %n) {
entry:
  %cmp = icmp sle i32 %n, 1
  br i1 %cmp, label %base_case, label %recursive_case

base_case:
  ret i32 1

recursive_case:
  %n_minus_1 = sub nsw i32 %n, 1
  %sub_result = call i32 @factorial(i32 %n_minus_1)
  %result = mul nsw i32 %n, %sub_result
  ret i32 %result
}

; Iterative version (more optimizable)
define i32 @factorial_iter(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 1, %entry ], [ %next_i, %loop ]
  %acc = phi i32 [ 1, %entry ], [ %next_acc, %loop ]
  %cmp = icmp sle i32 %i, %n
  br i1 %cmp, label %body, label %done

body:
  %next_acc = mul nsw i32 %acc, %i
  %next_i = add nsw i32 %i, 1
  br label %loop

done:
  ret i32 %acc
}
```

### 2.8 Generating LLVM IR from Python

We can generate LLVM IR programmatically using the `llvmlite` library:

```python
# pip install llvmlite

def generate_llvm_ir_example():
    """
    Generate LLVM IR for a simple function using llvmlite.

    Function: int add(int a, int b) { return a + b; }
    """
    try:
        from llvmlite import ir, binding

        # Create module
        module = ir.Module(name='example')
        module.triple = binding.get_default_triple()

        # Define function type: i32 (i32, i32)
        func_type = ir.FunctionType(ir.IntType(32),
                                     [ir.IntType(32), ir.IntType(32)])

        # Create function
        func = ir.Function(module, func_type, name='add')
        func.args[0].name = 'a'
        func.args[1].name = 'b'

        # Create basic block
        block = func.append_basic_block(name='entry')
        builder = ir.IRBuilder(block)

        # Generate instructions
        result = builder.add(func.args[0], func.args[1], name='result')
        builder.ret(result)

        print("Generated LLVM IR:")
        print(str(module))

        # --- More complex example: factorial ---
        fact_type = ir.FunctionType(ir.IntType(32), [ir.IntType(32)])
        fact_func = ir.Function(module, fact_type, name='factorial')
        fact_func.args[0].name = 'n'

        entry = fact_func.append_basic_block('entry')
        loop = fact_func.append_basic_block('loop')
        body = fact_func.append_basic_block('body')
        done = fact_func.append_basic_block('done')

        # Entry block
        builder = ir.IRBuilder(entry)
        builder.branch(loop)

        # Loop header
        builder = ir.IRBuilder(loop)
        i = builder.phi(ir.IntType(32), name='i')
        acc = builder.phi(ir.IntType(32), name='acc')
        i.add_incoming(ir.Constant(ir.IntType(32), 1), entry)
        acc.add_incoming(ir.Constant(ir.IntType(32), 1), entry)

        cmp = builder.icmp_signed('<=', i, fact_func.args[0], name='cmp')
        builder.cbranch(cmp, body, done)

        # Body
        builder = ir.IRBuilder(body)
        next_acc = builder.mul(acc, i, name='next_acc')
        next_i = builder.add(i, ir.Constant(ir.IntType(32), 1), name='next_i')
        i.add_incoming(next_i, body)
        acc.add_incoming(next_acc, body)
        builder.branch(loop)

        # Done
        builder = ir.IRBuilder(done)
        builder.ret(acc)

        print("\nWith factorial:")
        print(str(module))

        return str(module)

    except ImportError:
        print("llvmlite not installed. Install with: pip install llvmlite")
        print("\nHere is what the generated IR would look like:\n")
        print("""; ModuleID = 'example'
target triple = "arm64-apple-macosx14.0.0"

define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}

define i32 @factorial(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [ 1, %entry ], [ %next_i, %body ]
  %acc = phi i32 [ 1, %entry ], [ %next_acc, %body ]
  %cmp = icmp sle i32 %i, %n
  br i1 %cmp, label %body, label %done

body:
  %next_acc = mul i32 %acc, %i
  %next_i = add i32 %i, 1
  br label %loop

done:
  ret i32 %acc
}""")

generate_llvm_ir_example()
```

---

## 3. Writing LLVM Passes

### 3.1 Pass Types

LLVM organizes optimizations as **passes** that transform or analyze IR:

| Pass Type | Scope | Example |
|-----------|-------|---------|
| **Module Pass** | Entire module | Interprocedural analysis |
| **Function Pass** | Single function | Dead code elimination |
| **Loop Pass** | Single loop | Loop unrolling |
| **Basic Block Pass** | Single basic block | Peephole optimization |
| **Analysis Pass** | Read-only | Dominator tree computation |

### 3.2 The Pass Pipeline

LLVM's optimizer runs passes in a carefully ordered pipeline:

```
-O2 Pipeline (simplified):
  1. Simplify CFG
  2. SROA (Scalar Replacement of Aggregates)
  3. Early CSE (Common Subexpression Elimination)
  4. Inlining
  5. Simplify CFG
  6. Instruction Combining
  7. Reassociate
  8. Loop passes:
     a. Loop rotation
     b. LICM (Loop-Invariant Code Motion)
     c. Induction variable simplification
     d. Loop unrolling
  9. GVN (Global Value Numbering)
  10. Dead Code Elimination
  11. Simplify CFG
  12. ... (more passes)
```

### 3.3 Simulating an LLVM Pass in Python

Since writing actual LLVM passes requires C++, we can simulate the concept in Python:

```python
class LLVMIRSimulator:
    """
    Simulate LLVM IR and optimization passes in Python.
    """

    def __init__(self):
        self.functions = {}

    def add_function(self, name, blocks):
        """
        Add a function with basic blocks.
        blocks: dict of block_name -> list of instructions
        Each instruction: (result, opcode, operands...)
        """
        self.functions[name] = blocks

    def dump(self, func_name=None):
        """Print IR in LLVM-like format."""
        funcs = {func_name: self.functions[func_name]} if func_name else self.functions
        for name, blocks in funcs.items():
            print(f"define @{name}() {{")
            for block_name, instructions in blocks.items():
                print(f"{block_name}:")
                for instr in instructions:
                    if instr[1] == 'ret':
                        print(f"  ret {instr[2]}")
                    elif instr[1] == 'br':
                        if len(instr) == 3:
                            print(f"  br label %{instr[2]}")
                        else:
                            print(f"  br i1 {instr[2]}, label %{instr[3]}, label %{instr[4]}")
                    elif instr[1] == 'phi':
                        pairs = ', '.join(f"[ {v}, %{b} ]" for v, b in instr[2])
                        print(f"  {instr[0]} = phi {pairs}")
                    else:
                        ops = ', '.join(str(o) for o in instr[2:])
                        print(f"  {instr[0]} = {instr[1]} {ops}")
            print("}\n")


class OptimizationPass:
    """Base class for optimization passes."""

    def __init__(self, name):
        self.name = name
        self.changes = 0

    def run_on_function(self, func_name, blocks):
        """Override this to implement the pass. Returns modified blocks."""
        return blocks

    def __repr__(self):
        return f"Pass({self.name}, changes={self.changes})"


class ConstantFoldingPass(OptimizationPass):
    """
    Constant folding: evaluate operations on constants at compile time.

    Example: %x = add 3, 4  -->  %x = 7
    """

    def __init__(self):
        super().__init__("Constant Folding")

    def run_on_function(self, func_name, blocks):
        ops = {
            'add': lambda a, b: a + b,
            'sub': lambda a, b: a - b,
            'mul': lambda a, b: a * b,
        }

        new_blocks = {}
        for block_name, instructions in blocks.items():
            new_instructions = []
            for instr in instructions:
                if (len(instr) >= 4 and instr[1] in ops and
                        isinstance(instr[2], (int, float)) and
                        isinstance(instr[3], (int, float))):
                    result = ops[instr[1]](instr[2], instr[3])
                    # Replace with constant
                    new_instructions.append((instr[0], 'const', result))
                    self.changes += 1
                    print(f"  [ConstFold] {instr[0]} = {instr[1]} {instr[2]}, "
                          f"{instr[3]} --> {result}")
                else:
                    new_instructions.append(instr)
            new_blocks[block_name] = new_instructions

        return new_blocks


class DeadCodeEliminationPass(OptimizationPass):
    """
    Dead code elimination: remove instructions whose results are never used.
    """

    def __init__(self):
        super().__init__("Dead Code Elimination")

    def run_on_function(self, func_name, blocks):
        # Collect all used values
        used_values = set()
        for block_name, instructions in blocks.items():
            for instr in instructions:
                # All operands (positions 2+) that are string references
                for operand in instr[2:]:
                    if isinstance(operand, str) and operand.startswith('%'):
                        used_values.add(operand)
                    elif isinstance(operand, list):
                        for item in operand:
                            if isinstance(item, tuple):
                                for elem in item:
                                    if isinstance(elem, str) and elem.startswith('%'):
                                        used_values.add(elem)

        # Remove instructions whose results are not used
        new_blocks = {}
        for block_name, instructions in blocks.items():
            new_instructions = []
            for instr in instructions:
                result = instr[0]
                # Keep terminators and side-effecting instructions
                if instr[1] in ('ret', 'br', 'store', 'call'):
                    new_instructions.append(instr)
                elif result in used_values or result is None:
                    new_instructions.append(instr)
                else:
                    self.changes += 1
                    print(f"  [DCE] Removed: {result} = {instr[1]} ...")
            new_blocks[block_name] = new_instructions

        return new_blocks


class ConstantPropagationPass(OptimizationPass):
    """
    Constant propagation: replace uses of variables known to be constant.
    """

    def __init__(self):
        super().__init__("Constant Propagation")

    def run_on_function(self, func_name, blocks):
        # Find all constant definitions
        constants = {}
        for block_name, instructions in blocks.items():
            for instr in instructions:
                if instr[1] == 'const':
                    constants[instr[0]] = instr[2]

        if not constants:
            return blocks

        # Replace uses of constants
        new_blocks = {}
        for block_name, instructions in blocks.items():
            new_instructions = []
            for instr in instructions:
                new_instr = list(instr)
                for i in range(2, len(new_instr)):
                    if isinstance(new_instr[i], str) and new_instr[i] in constants:
                        old_val = new_instr[i]
                        new_instr[i] = constants[old_val]
                        self.changes += 1
                        print(f"  [ConstProp] {old_val} -> {constants[old_val]}")
                new_instructions.append(tuple(new_instr))
            new_blocks[block_name] = new_instructions

        return new_blocks


def demonstrate_pass_pipeline():
    """Demonstrate an optimization pass pipeline."""
    print("=== LLVM-style Pass Pipeline ===\n")

    ir = LLVMIRSimulator()

    # Function with optimization opportunities
    ir.add_function('compute', {
        'entry': [
            ('%a', 'const', 10),
            ('%b', 'const', 20),
            ('%c', 'add', '%a', '%b'),       # Can be constant folded (after prop)
            ('%d', 'mul', 3, 4),              # Can be constant folded
            ('%e', 'add', '%c', '%d'),        # Can be constant folded
            ('%unused', 'mul', '%a', '%b'),   # Dead code
            (None, 'ret', '%e'),
        ]
    })

    print("Before optimization:")
    ir.dump('compute')

    # Run passes
    passes = [
        ConstantFoldingPass(),
        ConstantPropagationPass(),
        ConstantFoldingPass(),      # Run again after propagation
        DeadCodeEliminationPass(),
    ]

    blocks = ir.functions['compute']
    for p in passes:
        print(f"\nRunning {p.name}:")
        blocks = p.run_on_function('compute', blocks)

    ir.functions['compute'] = blocks

    print("\nAfter optimization:")
    ir.dump('compute')

    print("Pass summary:")
    for p in passes:
        print(f"  {p}")

demonstrate_pass_pipeline()
```

### 3.4 Real LLVM Pass Example (C++ Sketch)

For reference, here is what a real LLVM pass looks like in C++ (New Pass Manager, LLVM 14+):

```cpp
// MyPass.h
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/raw_ostream.h"

struct MyCountPass : public llvm::PassInfoMixin<MyCountPass> {
    llvm::PreservedAnalyses run(llvm::Function &F,
                                 llvm::FunctionAnalysisManager &AM) {
        int count = 0;
        for (auto &BB : F) {
            count += BB.size();
        }
        llvm::errs() << "Function " << F.getName()
                      << " has " << count << " instructions\n";
        return llvm::PreservedAnalyses::all();
    }
};

// Register the pass
// In a plugin:
extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
    return {LLVM_PLUGIN_API_VERSION, "MyPass", LLVM_VERSION_STRING,
            [](PassBuilder &PB) {
                PB.registerPipelineParsingCallback(
                    [](StringRef Name, FunctionPassManager &FPM, ...) {
                        if (Name == "my-count-pass") {
                            FPM.addPass(MyCountPass());
                            return true;
                        }
                        return false;
                    });
            }};
}
```

---

## 4. MLIR: Multi-Level IR

### 4.1 The Problem MLIR Solves

Different domains need different levels of abstraction:

```
High-level:    TensorFlow graph ops (matmul, conv2d, ...)
               ↓
Mid-level:     Affine loops, tensor operations
               ↓
Low-level:     LLVM IR (scalar operations, memory loads/stores)
               ↓
Machine:       Target-specific machine instructions
```

Traditionally, each level has its own IR with its own optimization framework. MLIR provides a **single, extensible framework** for multiple levels.

### 4.2 MLIR Concepts

**Dialects**: MLIR organizes operations into dialects -- namespaced collections of operations, types, and attributes.

```mlir
// Affine dialect (structured loops and memory access)
func.func @matmul(%A: memref<256x256xf32>, %B: memref<256x256xf32>,
                    %C: memref<256x256xf32>) {
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 256 {
        %a = affine.load %A[%i, %k] : memref<256x256xf32>
        %b = affine.load %B[%k, %j] : memref<256x256xf32>
        %c = affine.load %C[%i, %j] : memref<256x256xf32>
        %prod = arith.mulf %a, %b : f32
        %sum = arith.addf %c, %prod : f32
        affine.store %sum, %C[%i, %j] : memref<256x256xf32>
      }
    }
  }
  return
}
```

### 4.3 Key MLIR Dialects

| Dialect | Purpose | Level |
|---------|---------|-------|
| `func` | Functions, calls | High |
| `arith` | Arithmetic operations | Mid |
| `affine` | Affine loops and memory | Mid |
| `linalg` | Linear algebra operations | Mid-High |
| `tensor` | Tensor types and ops | Mid-High |
| `memref` | Memory references | Mid-Low |
| `scf` | Structured control flow (for, if, while) | Mid |
| `cf` | Unstructured control flow (branch, switch) | Low |
| `llvm` | LLVM IR operations | Low |
| `gpu` | GPU operations | Target-specific |

### 4.4 Progressive Lowering

MLIR's key innovation: **progressive lowering** transforms high-level operations into lower-level ones through a series of dialect conversions.

```python
def demonstrate_progressive_lowering():
    """Demonstrate MLIR-style progressive lowering."""
    print("=== Progressive Lowering ===\n")

    levels = [
        {
            'name': 'TensorFlow Dialect',
            'code': '''
  %result = tf.MatMul(%A, %B) : tensor<256x256xf32>
            ''',
            'description': 'High-level: single operation for matrix multiply'
        },
        {
            'name': 'Linalg Dialect',
            'code': '''
  linalg.matmul
    ins(%A, %B : tensor<256x256xf32>, tensor<256x256xf32>)
    outs(%C : tensor<256x256xf32>) -> tensor<256x256xf32>
            ''',
            'description': 'Mid-high: generic linear algebra operation'
        },
        {
            'name': 'Affine Dialect',
            'code': '''
  affine.for %i = 0 to 256 {
    affine.for %j = 0 to 256 {
      affine.for %k = 0 to 256 {
        %a = affine.load %A[%i, %k]
        %b = affine.load %B[%k, %j]
        %prod = arith.mulf %a, %b
        %c = affine.load %C[%i, %j]
        %sum = arith.addf %c, %prod
        affine.store %sum, %C[%i, %j]
      }
    }
  }
            ''',
            'description': 'Mid: explicit loops with affine analysis'
        },
        {
            'name': 'SCF + MemRef Dialect',
            'code': '''
  scf.for %i = 0 to 256 step 1 {
    scf.for %j = 0 to 256 step 1 {
      scf.for %k = 0 to 256 step 1 {
        %addr_a = memref.load %A[%i, %k]
        %addr_b = memref.load %B[%k, %j]
        %prod = arith.mulf %addr_a, %addr_b
        ...
      }
    }
  }
            ''',
            'description': 'Mid-low: structured loops with explicit memory'
        },
        {
            'name': 'LLVM Dialect',
            'code': '''
  llvm.br ^loop_header
  ^loop_header:
    %i = llvm.phi [%zero, ^entry], [%next_i, ^loop_latch]
    %cmp = llvm.icmp "slt" %i, %n
    llvm.cond_br %cmp, ^loop_body, ^loop_exit
  ^loop_body:
    %addr = llvm.getelementptr %base[%i]
    %val = llvm.load %addr
    ...
            ''',
            'description': 'Low: maps directly to LLVM IR'
        },
    ]

    for i, level in enumerate(levels):
        print(f"Level {i + 1}: {level['name']}")
        print(f"  ({level['description']})")
        print(level['code'])
        if i < len(levels) - 1:
            print(f"    {'─' * 40}")
            print(f"    ↓  Lowering pass")
            print(f"    {'─' * 40}\n")

demonstrate_progressive_lowering()
```

---

## 5. GCC Internals

### 5.1 GCC Architecture

GCC (GNU Compiler Collection) uses a different internal architecture than LLVM:

```
GCC Compilation Pipeline:
┌─────────────────────────────────────────────────────────────┐
│                         GCC                                  │
│                                                              │
│  Source ──▶ Frontend ──▶ GENERIC ──▶ GIMPLE ──▶ SSA GIMPLE   │
│             (parser)    (lang-      (simplified  (SSA form   │
│                          specific    3-addr      with phi    │
│                          AST)        code)       nodes)      │
│                                                              │
│  SSA GIMPLE ──▶ Tree SSA Optimizations ──▶ Optimized GIMPLE  │
│                 (SRA, DCE, PRE, SCCP,                        │
│                  loop opts, vectorize)                        │
│                                                              │
│  Optimized GIMPLE ──▶ RTL ──▶ RTL Optimizations ──▶ Assembly │
│                       (Register Transfer    (register alloc, │
│                        Language)             instruction     │
│                                              scheduling)     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 GIMPLE

GIMPLE is GCC's high-level intermediate representation. It is a simplified, three-address form of the AST.

```c
// Source C code:
int sum = 0;
for (int i = 0; i < n; i++) {
    sum += arr[i] * 2;
}

// GIMPLE representation:
sum_1 = 0;
i_2 = 0;
goto <bb 3>;

<bb 2>:
_3 = arr[i_2];
_4 = _3 * 2;
sum_5 = sum_1 + _4;
i_6 = i_2 + 1;

<bb 3>:
# sum_1 = PHI <sum_5(bb2), 0(bb1)>
# i_2 = PHI <i_6(bb2), 0(bb1)>
if (i_2 < n_7)
    goto <bb 2>;
else
    goto <bb 4>;

<bb 4>:
return sum_1;
```

### 5.3 RTL (Register Transfer Language)

RTL is GCC's low-level IR, close to machine instructions:

```
;; RTL for: x = a + b
(set (reg:SI 100)
     (plus:SI (reg:SI 101)
              (reg:SI 102)))

;; RTL for: if (x < 0) goto L1
(set (reg:CC 17)
     (compare:CC (reg:SI 100)
                 (const_int 0)))
(set (pc)
     (if_then_else (lt (reg:CC 17) (const_int 0))
                   (label_ref L1)
                   (pc)))
```

### 5.4 LLVM vs GCC Comparison

```python
def llvm_vs_gcc_comparison():
    """Compare LLVM and GCC architectures."""
    print("=== LLVM vs GCC ===\n")

    comparison = [
        ('Architecture', 'Modular library', 'Monolithic compiler'),
        ('License', 'Apache 2.0', 'GPL v3'),
        ('High-level IR', 'LLVM IR (one level)', 'GENERIC -> GIMPLE (two levels)'),
        ('Low-level IR', 'SelectionDAG -> MachineIR', 'RTL'),
        ('SSA form', 'Core IR is SSA', 'GIMPLE SSA (separate phase)'),
        ('Reusability', 'Library-based (easy to embed)', 'Hard to use as library'),
        ('Frontend API', 'Clean C++ API', 'Plugin API (limited)'),
        ('Targets', '~20 targets', '~50+ targets'),
        ('Diagnostics', 'Excellent (Clang)', 'Good (improved recently)'),
        ('LTO', 'ThinLTO + Full LTO', 'Full LTO'),
        ('Build speed', 'Fast (Clang)', 'Moderate'),
        ('Optimization', 'Strong, especially SIMD', 'Strong, wider target support'),
    ]

    print(f"{'Aspect':<20} {'LLVM':<30} {'GCC':<30}")
    print("-" * 80)
    for aspect, llvm, gcc in comparison:
        print(f"{aspect:<20} {llvm:<30} {gcc:<30}")

llvm_vs_gcc_comparison()
```

---

## 6. Domain-Specific Languages

### 6.1 What is a DSL?

A **domain-specific language** (DSL) is a programming language tailored to a specific problem domain. Unlike general-purpose languages (GPLs), DSLs sacrifice generality for expressiveness and ease of use in their domain.

| Type | Description | Examples |
|------|-------------|---------|
| **External DSL** | Separate language with own parser | SQL, HTML, CSS, regex, Makefile |
| **Internal/Embedded DSL** | Hosted within a GPL | SQLAlchemy (Python), Kotlin DSLs, Scala implicits |

### 6.2 DSL Design Principles

1. **Domain focus**: Express domain concepts directly, not programming concepts
2. **Abstraction**: Hide irrelevant details
3. **Notation**: Match domain experts' notation
4. **Safety**: Prevent errors that don't make sense in the domain
5. **Composition**: Allow combining DSL elements naturally

### 6.3 Building an External DSL

Let us build a simple DSL for defining data processing pipelines:

```python
import re
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Any


# --- AST Nodes ---

@dataclass
class PipelineAST:
    name: str
    stages: List['StageAST']

@dataclass
class StageAST:
    operation: str
    arguments: dict

@dataclass
class FilterAST(StageAST):
    condition: str

@dataclass
class TransformAST(StageAST):
    expression: str

@dataclass
class AggregateAST(StageAST):
    function: str
    column: str


# --- Parser ---

class PipelineDSLParser:
    """
    Parser for a simple data pipeline DSL.

    Syntax:
        pipeline "name" {
            read csv "data.csv"
            filter where age > 18
            transform salary * 1.1 as adjusted_salary
            group by department
            aggregate sum(salary) as total_salary
            write csv "output.csv"
        }
    """

    def __init__(self, source: str):
        self.source = source
        self.lines = [line.strip() for line in source.strip().split('\n')
                      if line.strip() and not line.strip().startswith('#')]
        self.pos = 0

    def parse(self) -> PipelineAST:
        """Parse the entire DSL source."""
        # Parse pipeline header
        header = self.lines[self.pos]
        match = re.match(r'pipeline\s+"([^"]+)"\s*\{', header)
        if not match:
            raise SyntaxError(f"Expected 'pipeline \"name\" {{', got: {header}")

        name = match.group(1)
        self.pos += 1

        # Parse stages
        stages = []
        while self.pos < len(self.lines):
            line = self.lines[self.pos]
            if line == '}':
                break
            stages.append(self.parse_stage(line))
            self.pos += 1

        return PipelineAST(name=name, stages=stages)

    def parse_stage(self, line: str) -> StageAST:
        """Parse a single pipeline stage."""
        parts = line.split(None, 1)
        operation = parts[0]
        rest = parts[1] if len(parts) > 1 else ''

        if operation == 'read':
            fmt_match = re.match(r'(\w+)\s+"([^"]+)"', rest)
            if fmt_match:
                return StageAST(
                    operation='read',
                    arguments={'format': fmt_match.group(1),
                               'path': fmt_match.group(2)}
                )

        elif operation == 'filter':
            return FilterAST(
                operation='filter',
                arguments={},
                condition=rest.replace('where ', '')
            )

        elif operation == 'transform':
            # Parse "expression as new_name"
            match = re.match(r'(.+)\s+as\s+(\w+)', rest)
            if match:
                return TransformAST(
                    operation='transform',
                    arguments={'new_column': match.group(2)},
                    expression=match.group(1)
                )

        elif operation == 'group':
            return StageAST(
                operation='group',
                arguments={'by': rest.replace('by ', '')}
            )

        elif operation == 'aggregate':
            match = re.match(r'(\w+)\((\w+)\)\s+as\s+(\w+)', rest)
            if match:
                return AggregateAST(
                    operation='aggregate',
                    arguments={'as': match.group(3)},
                    function=match.group(1),
                    column=match.group(2)
                )

        elif operation == 'write':
            fmt_match = re.match(r'(\w+)\s+"([^"]+)"', rest)
            if fmt_match:
                return StageAST(
                    operation='write',
                    arguments={'format': fmt_match.group(1),
                               'path': fmt_match.group(2)}
                )

        return StageAST(operation=operation, arguments={'raw': rest})


# --- Code Generator ---

class PythonCodeGenerator:
    """Generate Python code from the pipeline AST."""

    def generate(self, pipeline: PipelineAST) -> str:
        lines = [
            f"# Generated pipeline: {pipeline.name}",
            "import pandas as pd",
            "",
        ]

        for i, stage in enumerate(pipeline.stages):
            if stage.operation == 'read':
                fmt = stage.arguments['format']
                path = stage.arguments['path']
                if fmt == 'csv':
                    lines.append(f"df = pd.read_csv('{path}')")

            elif stage.operation == 'filter':
                lines.append(f"df = df[df.eval('{stage.condition}')]")

            elif stage.operation == 'transform':
                col = stage.arguments['new_column']
                lines.append(f"df['{col}'] = df.eval('{stage.expression}')")

            elif stage.operation == 'group':
                col = stage.arguments['by']
                lines.append(f"df = df.groupby('{col}')")

            elif stage.operation == 'aggregate':
                func = stage.function
                col = stage.column
                alias = stage.arguments['as']
                lines.append(f"df = df.agg({{'{col}': '{func}'}})"
                             f".rename(columns={{'{col}': '{alias}'}})")

            elif stage.operation == 'write':
                fmt = stage.arguments['format']
                path = stage.arguments['path']
                if fmt == 'csv':
                    lines.append(f"df.to_csv('{path}', index=False)")

        return '\n'.join(lines)


def demonstrate_dsl():
    """Demonstrate the pipeline DSL."""
    print("=== Data Pipeline DSL ===\n")

    dsl_source = '''
pipeline "employee_analysis" {
    read csv "employees.csv"
    filter where age > 25
    transform salary * 1.1 as adjusted_salary
    group by department
    aggregate sum(adjusted_salary) as total_adjusted
    write csv "department_totals.csv"
}
'''

    print("DSL Source:")
    print(dsl_source)

    # Parse
    parser = PipelineDSLParser(dsl_source)
    ast = parser.parse()

    print("Parsed AST:")
    print(f"  Pipeline: {ast.name}")
    for stage in ast.stages:
        print(f"  Stage: {stage.operation} {stage.arguments}")

    # Generate Python code
    generator = PythonCodeGenerator()
    python_code = generator.generate(ast)

    print("\nGenerated Python Code:")
    print(python_code)

demonstrate_dsl()
```

### 6.4 Embedded DSLs

An **embedded DSL** (EDSL) uses the host language's syntax to create a domain-specific feel:

```python
class QueryBuilder:
    """
    Embedded DSL for building SQL queries in Python.
    Uses method chaining (fluent interface) for a DSL-like feel.
    """

    def __init__(self):
        self._select_cols = ['*']
        self._from_table = None
        self._where_clauses = []
        self._order_by = []
        self._limit = None
        self._joins = []

    def select(self, *columns):
        self._select_cols = list(columns)
        return self  # Enable chaining

    def from_table(self, table):
        self._from_table = table
        return self

    def where(self, condition):
        self._where_clauses.append(condition)
        return self

    def and_where(self, condition):
        return self.where(condition)

    def join(self, table, on):
        self._joins.append(f"JOIN {table} ON {on}")
        return self

    def left_join(self, table, on):
        self._joins.append(f"LEFT JOIN {table} ON {on}")
        return self

    def order_by(self, column, direction='ASC'):
        self._order_by.append(f"{column} {direction}")
        return self

    def limit(self, n):
        self._limit = n
        return self

    def build(self):
        """Generate the SQL string."""
        parts = [f"SELECT {', '.join(self._select_cols)}"]
        parts.append(f"FROM {self._from_table}")

        for join in self._joins:
            parts.append(join)

        if self._where_clauses:
            parts.append(f"WHERE {' AND '.join(self._where_clauses)}")

        if self._order_by:
            parts.append(f"ORDER BY {', '.join(self._order_by)}")

        if self._limit:
            parts.append(f"LIMIT {self._limit}")

        return '\n'.join(parts)

    def __str__(self):
        return self.build()


def demonstrate_embedded_dsl():
    """Demonstrate the embedded SQL DSL."""
    print("=== Embedded SQL DSL ===\n")

    query = (QueryBuilder()
             .select('e.name', 'e.salary', 'd.name AS department')
             .from_table('employees e')
             .left_join('departments d', 'e.dept_id = d.id')
             .where('e.salary > 50000')
             .and_where('e.active = true')
             .order_by('e.salary', 'DESC')
             .limit(10))

    print("Generated SQL:")
    print(query)

demonstrate_embedded_dsl()
```

---

## 7. Compiler Construction Tools

### 7.1 ANTLR

**ANTLR** (ANother Tool for Language Recognition) is a powerful parser generator that creates parsers from grammar specifications.

```
// ANTLR grammar for a simple expression language
grammar Expr;

// Parser rules
program : statement+ ;

statement : assignment
          | printStmt
          | ifStmt
          | whileStmt
          ;

assignment : ID '=' expr ';' ;
printStmt : 'print' '(' expr ')' ';' ;
ifStmt : 'if' '(' expr ')' block ('else' block)? ;
whileStmt : 'while' '(' expr ')' block ;
block : '{' statement* '}' ;

expr : expr ('*'|'/') expr     # MulDiv
     | expr ('+'|'-') expr     # AddSub
     | expr ('<'|'>'|'==') expr # Compare
     | '(' expr ')'            # Parens
     | ID                      # Identifier
     | INT                     # Integer
     ;

// Lexer rules
ID : [a-zA-Z_][a-zA-Z_0-9]* ;
INT : [0-9]+ ;
WS : [ \t\r\n]+ -> skip ;
COMMENT : '//' ~[\r\n]* -> skip ;
```

ANTLR generates a lexer, parser, and parse tree from this grammar. It supports many target languages (Java, Python, C++, JavaScript, Go, etc.).

### 7.2 Tree-sitter

**Tree-sitter** is a parser generator designed for incremental parsing -- ideal for code editors and language tooling.

Key features:
- **Incremental**: After editing a file, only re-parses the changed portion
- **Error-tolerant**: Produces a valid parse tree even for syntactically incorrect code
- **Fast**: Parses most files in under a millisecond
- **Concrete syntax tree**: Preserves all tokens (whitespace, comments) for exact round-tripping

```python
def demonstrate_tree_sitter_concept():
    """
    Demonstrate Tree-sitter's incremental parsing concept.
    (Actual Tree-sitter requires C bindings; this simulates the behavior.)
    """
    print("=== Tree-sitter Incremental Parsing ===\n")

    class IncrementalParser:
        """Simulated incremental parser."""

        def __init__(self):
            self.tree = None
            self.source = ""

        def parse(self, source):
            """Full parse."""
            self.source = source
            self.tree = self._build_tree(source)
            return self.tree

        def edit(self, start, end, new_text):
            """
            Incremental edit: only re-parse the changed region.

            In real Tree-sitter:
            1. Apply the edit to the old tree
            2. Re-lex only the changed region
            3. Re-parse only affected subtrees
            4. Reuse unchanged subtrees from the old tree
            """
            old_source = self.source
            self.source = old_source[:start] + new_text + old_source[end:]

            # In reality, Tree-sitter would:
            # - Identify which tree nodes are invalidated
            # - Re-parse only those regions
            # - Reuse all other nodes from the old tree

            changed_region = (start, start + len(new_text))
            print(f"  Edit at [{start}:{end}] -> '{new_text}'")
            print(f"  Only re-parsing characters {changed_region}")
            print(f"  Reusing tree nodes outside this range")

            self.tree = self._build_tree(self.source)
            return self.tree

        def _build_tree(self, source):
            """Simplified tree building."""
            return {'source': source, 'type': 'program', 'children': []}

    parser = IncrementalParser()

    # Initial parse
    source = "let x = 10;\nlet y = 20;\nlet z = x + y;"
    print(f"Initial source:\n  {source}\n")
    tree = parser.parse(source)

    # Edit: change "10" to "42"
    print("Edit: change '10' to '42'")
    tree = parser.edit(8, 10, "42")
    print(f"  New source: {parser.source}\n")

    print("Key insight: Tree-sitter re-parses only the changed region,")
    print("not the entire file. For large files, this is orders of magnitude faster.")

demonstrate_tree_sitter_concept()
```

### 7.3 Tool Comparison

```python
def tool_comparison():
    """Compare parser generator tools."""
    tools = [
        ('ANTLR', 'LL(*)', 'Java,Py,C++,JS,Go', 'Full parse tree', 'Language implementation'),
        ('Tree-sitter', 'GLR', 'C (bindings)', 'Incremental CST', 'Editors, tooling'),
        ('Yacc/Bison', 'LALR(1)', 'C, C++', 'Action-based', 'Traditional compilers'),
        ('PEG.js/Pest', 'PEG', 'JS/Rust', 'Full parse tree', 'Simple languages'),
        ('Lark', 'Earley/LALR', 'Python', 'Parse tree', 'Python projects'),
        ('Nom', 'Combinator', 'Rust', 'Custom', 'Binary formats, protocols'),
    ]

    print("=== Parser Generator Comparison ===\n")
    print(f"{'Tool':<15} {'Algorithm':<12} {'Languages':<18} {'Best For':<25}")
    print("-" * 70)
    for name, algo, langs, output, best_for in tools:
        print(f"{name:<15} {algo:<12} {langs:<18} {best_for:<25}")

tool_comparison()
```

---

## 8. Language Server Protocol

### 8.1 What is LSP?

The **Language Server Protocol** (LSP), developed by Microsoft, standardizes the interface between code editors and language-specific tooling. Before LSP, every editor needed a custom plugin for every language (M editors times N languages = M*N implementations). LSP reduces this to M + N.

```
Without LSP:                        With LSP:

  VS Code ──── C plugin              VS Code ──┐
  VS Code ──── Python plugin                    │
  VS Code ──── Rust plugin           Vim ──────┤   LSP    ┌── C server
  Vim ──────── C plugin              Emacs ────┤ Protocol ├── Python server
  Vim ──────── Python plugin         Sublime ──┘          ├── Rust server
  Vim ──────── Rust plugin                                └── ...
  Emacs ────── C plugin
  Emacs ────── Python plugin         M + N implementations
  Emacs ────── Rust plugin           (instead of M * N)
  ...

  M * N implementations
```

### 8.2 LSP Architecture

```
┌──────────────┐         JSON-RPC          ┌──────────────────┐
│              │ ◀─── notifications ──────  │                  │
│    Editor    │ ───── requests ──────────▶ │  Language Server  │
│   (Client)   │ ◀─── responses ─────────  │                  │
│              │                            │  - Parser         │
│  VS Code     │  textDocument/didOpen      │  - Type checker   │
│  Vim         │  textDocument/completion   │  - Diagnostics    │
│  Emacs       │  textDocument/definition   │  - Formatter      │
│  Sublime     │  textDocument/references   │  - Refactoring    │
│              │  textDocument/hover        │                  │
└──────────────┘                            └──────────────────┘
```

### 8.3 LSP Capabilities

```python
def lsp_capabilities():
    """Show key LSP capabilities."""
    print("=== LSP Capabilities ===\n")

    capabilities = [
        ('textDocument/completion', 'Auto-completion suggestions',
         'User types "obj." -> server suggests methods'),
        ('textDocument/hover', 'Type info and docs on hover',
         'Hover over function -> show signature and docstring'),
        ('textDocument/definition', 'Go to definition',
         'Click on function call -> jump to its definition'),
        ('textDocument/references', 'Find all references',
         'Find all places where a symbol is used'),
        ('textDocument/rename', 'Rename symbol',
         'Rename a variable across all files'),
        ('textDocument/diagnostics', 'Error and warning diagnostics',
         'Red underlines for errors, yellow for warnings'),
        ('textDocument/formatting', 'Code formatting',
         'Auto-format according to style rules'),
        ('textDocument/codeAction', 'Quick fixes and refactorings',
         '"Extract method", "Import missing module"'),
        ('textDocument/signatureHelp', 'Function signature help',
         'Show parameter types while typing arguments'),
        ('textDocument/foldingRange', 'Code folding',
         'Collapse/expand functions, classes, regions'),
    ]

    for method, description, example in capabilities:
        print(f"  {method}")
        print(f"    {description}")
        print(f"    Example: {example}\n")

lsp_capabilities()
```

### 8.4 Building a Simple Language Server

```python
import json

class SimpleLSPServer:
    """
    Simplified Language Server Protocol server.
    Demonstrates the core request/response pattern.
    """

    def __init__(self):
        self.documents = {}  # uri -> content
        self.diagnostics = {}

    def handle_request(self, method, params):
        """Handle an LSP request."""
        handler = getattr(self, f'handle_{method.replace("/", "_")}', None)
        if handler:
            return handler(params)
        return {'error': f'Unknown method: {method}'}

    def handle_textDocument_didOpen(self, params):
        """Handle document open notification."""
        uri = params['textDocument']['uri']
        text = params['textDocument']['text']
        self.documents[uri] = text

        # Analyze for diagnostics
        diagnostics = self._analyze(uri, text)
        return {'method': 'textDocument/publishDiagnostics',
                'params': {'uri': uri, 'diagnostics': diagnostics}}

    def handle_textDocument_completion(self, params):
        """Handle completion request."""
        uri = params['textDocument']['uri']
        position = params['position']

        # Simple keyword completion
        keywords = ['if', 'else', 'while', 'for', 'def', 'return',
                     'class', 'import', 'from', 'print']

        text = self.documents.get(uri, '')
        lines = text.split('\n')
        line = lines[position['line']] if position['line'] < len(lines) else ''

        # Get the partial word being typed
        col = position['character']
        partial = ''
        for i in range(col - 1, -1, -1):
            if i < len(line) and line[i].isalnum():
                partial = line[i] + partial
            else:
                break

        # Filter keywords
        items = [
            {'label': kw, 'kind': 14,  # Keyword
             'detail': f'Keyword: {kw}'}
            for kw in keywords if kw.startswith(partial)
        ]

        return {'items': items}

    def handle_textDocument_hover(self, params):
        """Handle hover request."""
        uri = params['textDocument']['uri']
        position = params['position']

        text = self.documents.get(uri, '')
        lines = text.split('\n')

        if position['line'] < len(lines):
            line = lines[position['line']]
            # Simple: return the line content
            return {
                'contents': {
                    'kind': 'markdown',
                    'value': f'```\n{line.strip()}\n```'
                }
            }
        return None

    def _analyze(self, uri, text):
        """Simple static analysis for diagnostics."""
        diagnostics = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            # Check for common issues
            if 'eval(' in line:
                diagnostics.append({
                    'range': {
                        'start': {'line': i, 'character': line.index('eval(')},
                        'end': {'line': i, 'character': line.index('eval(') + 5},
                    },
                    'severity': 2,  # Warning
                    'message': 'Use of eval() is a security risk',
                    'source': 'simple-lsp',
                })

            if len(line) > 120:
                diagnostics.append({
                    'range': {
                        'start': {'line': i, 'character': 120},
                        'end': {'line': i, 'character': len(line)},
                    },
                    'severity': 3,  # Information
                    'message': f'Line exceeds 120 characters ({len(line)})',
                    'source': 'simple-lsp',
                })

        return diagnostics


def demonstrate_lsp():
    """Demonstrate LSP server behavior."""
    print("=== Simple LSP Server Demo ===\n")

    server = SimpleLSPServer()

    # Simulate document open
    result = server.handle_request('textDocument/didOpen', {
        'textDocument': {
            'uri': 'file:///example.py',
            'text': 'x = eval(input())\nprint(x)\n' + 'y = ' + 'a' * 130,
        }
    })
    print("Diagnostics on open:")
    for diag in result['params']['diagnostics']:
        print(f"  Line {diag['range']['start']['line']}: {diag['message']}")

    # Simulate completion
    result = server.handle_request('textDocument/completion', {
        'textDocument': {'uri': 'file:///example.py'},
        'position': {'line': 0, 'character': 3},
    })
    print(f"\nCompletions for partial word:")
    for item in result.get('items', [])[:5]:
        print(f"  {item['label']}")

demonstrate_lsp()
```

---

## 9. Incremental Compilation

### 9.1 The Problem

Large projects take a long time to compile. After changing one file, recompiling everything is wasteful. **Incremental compilation** only recompiles what has changed.

### 9.2 Dependency Tracking

```python
import os
import time
from collections import defaultdict


class IncrementalCompiler:
    """
    Simulates incremental compilation with dependency tracking.
    """

    def __init__(self):
        self.file_timestamps = {}   # file -> last modified time
        self.compiled_cache = {}    # file -> compiled result
        self.dependencies = defaultdict(set)  # file -> set of files it depends on
        self.reverse_deps = defaultdict(set)  # file -> set of files that depend on it
        self.compile_count = 0

    def add_dependency(self, source, depends_on):
        """Record that source depends on depends_on."""
        self.dependencies[source].add(depends_on)
        self.reverse_deps[depends_on].add(source)

    def compile_file(self, filename, content):
        """Compile a single file (simulated)."""
        self.compile_count += 1
        print(f"  Compiling: {filename}")
        # Simulate compilation
        result = f"compiled({filename})"
        self.compiled_cache[filename] = result
        self.file_timestamps[filename] = time.time()
        return result

    def needs_recompilation(self, filename, current_mtime):
        """Check if a file needs recompilation."""
        if filename not in self.compiled_cache:
            return True  # Never compiled

        if current_mtime > self.file_timestamps.get(filename, 0):
            return True  # Source file changed

        # Check if any dependency changed
        for dep in self.dependencies.get(filename, set()):
            dep_mtime = self.file_timestamps.get(dep, 0)
            if dep_mtime > self.file_timestamps.get(filename, 0):
                return True  # Dependency changed

        return False

    def build(self, files_with_mtimes):
        """
        Incremental build: only compile files that need it.

        files_with_mtimes: dict of filename -> (content, mtime)
        """
        self.compile_count = 0

        # Determine what needs recompilation
        to_compile = set()
        for filename, (content, mtime) in files_with_mtimes.items():
            if self.needs_recompilation(filename, mtime):
                to_compile.add(filename)

        # Also recompile reverse dependencies of changed files
        worklist = list(to_compile)
        while worklist:
            f = worklist.pop()
            for rdep in self.reverse_deps.get(f, set()):
                if rdep not in to_compile and rdep in files_with_mtimes:
                    to_compile.add(rdep)
                    worklist.append(rdep)

        if not to_compile:
            print("  Nothing to compile (all up to date)")
            return

        # Topological sort for correct compilation order
        compiled = set()
        def compile_with_deps(f):
            if f in compiled:
                return
            for dep in self.dependencies.get(f, set()):
                if dep in to_compile:
                    compile_with_deps(dep)
            content, mtime = files_with_mtimes[f]
            self.compile_file(f, content)
            compiled.add(f)

        for f in to_compile:
            compile_with_deps(f)

        print(f"  Compiled {self.compile_count} out of {len(files_with_mtimes)} files")


def demonstrate_incremental_compilation():
    """Show incremental compilation behavior."""
    print("=== Incremental Compilation ===\n")

    compiler = IncrementalCompiler()

    # Define dependencies: main.c depends on util.h and math.h
    compiler.add_dependency('main.c', 'util.h')
    compiler.add_dependency('main.c', 'math.h')
    compiler.add_dependency('util.c', 'util.h')
    compiler.add_dependency('math.c', 'math.h')

    now = time.time()

    # Build 1: Full build
    print("Build 1: Initial (full) build")
    files = {
        'util.h': ('header', now),
        'math.h': ('header', now),
        'main.c': ('main code', now),
        'util.c': ('util code', now),
        'math.c': ('math code', now),
    }
    compiler.build(files)

    # Build 2: Nothing changed
    print("\nBuild 2: No changes")
    compiler.build(files)

    # Build 3: Changed util.h (should recompile main.c and util.c)
    print("\nBuild 3: Modified util.h")
    files['util.h'] = ('modified header', now + 1)
    compiler.build(files)

    # Build 4: Changed only math.c
    print("\nBuild 4: Modified only math.c")
    files['math.c'] = ('modified math code', now + 2)
    compiler.build(files)

demonstrate_incremental_compilation()
```

### 9.3 Incremental Compilation in Practice

| System | Strategy |
|--------|----------|
| **Rust (cargo)** | Per-crate compilation, query-based incremental within crates |
| **Go** | Per-package compilation, fast full builds |
| **Java (javac)** | Per-file compilation, dependency tracking |
| **C/C++ (make)** | Per-file compilation, timestamp-based rebuild |
| **TypeScript** | Per-project, in-memory incremental |

---

## 10. Profile-Guided Optimization

### 10.1 What is PGO?

**Profile-Guided Optimization (PGO)** uses runtime profiling data to make better optimization decisions. The compiler first generates instrumented code, runs it on representative inputs to collect profile data, then recompiles using the profile.

```
Phase 1: Instrumented Build
  Source ──▶ Compiler (-fprofile-generate) ──▶ Instrumented Binary

Phase 2: Profile Collection
  Instrumented Binary + Training Input ──▶ Profile Data (.profdata)

Phase 3: Optimized Build
  Source + Profile Data ──▶ Compiler (-fprofile-use) ──▶ Optimized Binary
```

### 10.2 PGO Optimization Decisions

Profile data enables several optimizations:

```python
def pgo_optimizations():
    """Show what PGO enables."""
    print("=== PGO Optimization Decisions ===\n")

    optimizations = [
        {
            'name': 'Branch Prediction Hints',
            'description': 'Mark likely/unlikely branches based on observed frequencies',
            'example': '''
  // Profile says: condition is true 95% of the time
  if (likely(x > 0)) {  // Hot path: optimized layout
      process(x);
  } else {              // Cold path: moved out of line
      handle_error(x);
  }
            ''',
            'impact': 'Better instruction cache utilization, fewer branch mispredictions'
        },
        {
            'name': 'Function Inlining',
            'description': 'Inline hot call sites, don\'t inline cold ones',
            'example': '''
  // Profile says: parse_header called 1M times, parse_trailer called 100 times
  // Inline parse_header (hot), don't inline parse_trailer (cold)
            ''',
            'impact': 'Better code size/speed trade-off'
        },
        {
            'name': 'Basic Block Layout',
            'description': 'Place hot blocks together, cold blocks separately',
            'example': '''
  // Hot path blocks placed sequentially for better i-cache usage
  // Cold blocks (error handling) moved to end of function
            ''',
            'impact': 'Better instruction cache hit rate'
        },
        {
            'name': 'Register Allocation',
            'description': 'Prioritize register allocation for hot paths',
            'example': '''
  // Variables used in hot loops get registers
  // Variables used only in cold paths get stack slots
            ''',
            'impact': 'Fewer memory accesses in hot code'
        },
        {
            'name': 'Virtual Call Devirtualization',
            'description': 'Convert virtual calls to direct calls based on observed types',
            'example': '''
  // Profile: 99% of calls to shape.area() are on Circle objects
  if (shape is Circle) {     // Speculative devirtualization
      circle_area(shape);    // Direct call (inlinable)
  } else {
      shape.area();          // Virtual call fallback
  }
            ''',
            'impact': 'Enables inlining of virtual methods'
        },
    ]

    for opt in optimizations:
        print(f"  {opt['name']}")
        print(f"    {opt['description']}")
        print(f"    Impact: {opt['impact']}")
        print()

pgo_optimizations()
```

### 10.3 Using PGO with Clang/GCC

```bash
# Clang PGO workflow:

# Step 1: Build with instrumentation
clang -fprofile-instr-generate -O2 program.c -o program_instrumented

# Step 2: Run with representative input
./program_instrumented < training_input.txt
# This generates default.profraw

# Step 3: Merge profile data
llvm-profdata merge default.profraw -output=program.profdata

# Step 4: Build with profile data
clang -fprofile-instr-use=program.profdata -O2 program.c -o program_optimized

# Typical speedup: 10-20% for large applications
```

### 10.4 Simulating PGO

```python
def simulate_pgo():
    """Simulate PGO's effect on branch layout."""
    print("=== PGO Simulation ===\n")

    # Simulated function with branches
    import random
    random.seed(42)

    # Generate "execution profile"
    n = 10000
    profile = {
        'branch_A_true': 0,
        'branch_A_false': 0,
        'branch_B_true': 0,
        'branch_B_false': 0,
    }

    for _ in range(n):
        x = random.gauss(100, 20)

        if x > 50:  # Branch A: taken 99% of the time
            profile['branch_A_true'] += 1
        else:
            profile['branch_A_false'] += 1

        if x > 150:  # Branch B: taken ~1% of the time
            profile['branch_B_true'] += 1
        else:
            profile['branch_B_false'] += 1

    print("Profile data collected:")
    for branch, count in profile.items():
        pct = count / n * 100
        print(f"  {branch}: {count} ({pct:.1f}%)")

    print("\nPGO decisions:")
    a_ratio = profile['branch_A_true'] / n
    b_ratio = profile['branch_B_true'] / n

    print(f"  Branch A ({a_ratio*100:.1f}% true):")
    print(f"    -> Predict TRUE, place true-path first (fall-through)")
    print(f"    -> Move false-path out-of-line")

    print(f"  Branch B ({b_ratio*100:.1f}% true):")
    print(f"    -> Predict FALSE, place false-path first (fall-through)")
    print(f"    -> Move true-path out-of-line")

    print("\nCode layout:")
    print("""
  Without PGO:              With PGO:
  func:                     func:
    cmp x, 50                 cmp x, 50
    jle .else_A               jle .cold_A       (rarely taken)
    ; true path A             ; true path A     (fall-through: hot)
    ...                       ...
    jmp .end_A                cmp x, 150
  .else_A:                    jg .cold_B        (rarely taken)
    ; false path A            ; false path B    (fall-through: hot)
    ...                       ...
  .end_A:                     ret
    cmp x, 150
    jle .else_B             ; Cold section (separate cache lines):
    ; true path B           .cold_A:
    ...                       ; false path A
    jmp .end_B                jmp .back_A
  .else_B:                  .cold_B:
    ; false path B            ; true path B
    ...                       jmp .back_B
  .end_B:
    ret
    """)

simulate_pgo()
```

---

## 11. Link-Time Optimization

### 11.1 What is LTO?

**Link-Time Optimization (LTO)** defers some optimizations to link time, when the compiler has visibility across all translation units (source files). This enables:

- Cross-module inlining
- Interprocedural constant propagation
- Dead function elimination across modules
- Whole-program devirtualization

```
Without LTO:
  a.c ──▶ a.o ──┐
  b.c ──▶ b.o ──┤──▶ Linker ──▶ binary
  c.c ──▶ c.o ──┘
  (each .o optimized independently)

With LTO:
  a.c ──▶ a.bc ──┐
  b.c ──▶ b.bc ──┤──▶ LTO Optimizer ──▶ Linker ──▶ binary
  c.c ──▶ c.bc ──┘
  (.bc = LLVM bitcode, optimized together)
```

### 11.2 Full LTO vs ThinLTO

| Aspect | Full LTO | ThinLTO |
|--------|----------|---------|
| **Scope** | Merges all modules into one | Keeps modules separate |
| **Link time** | Slow (single-threaded) | Fast (parallelizable) |
| **Memory** | High (entire program in memory) | Low (summary-based) |
| **Optimization quality** | Best (full visibility) | Nearly as good (95%+) |
| **Incremental** | No (re-optimizes everything) | Yes (per-module) |

```bash
# Full LTO with Clang
clang -flto -O2 a.c b.c c.c -o program

# ThinLTO (recommended for large projects)
clang -flto=thin -O2 a.c b.c c.c -o program
```

### 11.3 What LTO Enables

```python
def lto_example():
    """Show what LTO enables that per-file compilation cannot."""
    print("=== LTO Optimizations ===\n")

    print("Example: Cross-module inlining\n")
    print("  // util.c")
    print("  int square(int x) { return x * x; }")
    print()
    print("  // main.c")
    print("  extern int square(int);")
    print("  int main() { return square(5); }")
    print()
    print("  Without LTO: square() is a function call")
    print("  With LTO:    square(5) is inlined -> return 25")
    print("               Constant folded -> return 25")
    print("               (zero runtime cost!)")

    print("\nExample: Dead code elimination\n")
    print("  // util.c")
    print("  void used_function() { ... }")
    print("  void unused_function() { ... }  // 10,000 lines")
    print()
    print("  Without LTO: both functions in binary (linker can't tell)")
    print("  With LTO:    unused_function() eliminated (smaller binary)")

    print("\nExample: Interprocedural constant propagation\n")
    print("  // config.c")
    print("  int get_mode() { return 3; }  // Always returns 3")
    print()
    print("  // engine.c")
    print("  int mode = get_mode();")
    print("  if (mode == 1) { ... }  // Dead code (mode is always 3)")
    print("  if (mode == 2) { ... }  // Dead code")
    print("  if (mode == 3) { ... }  // Only this branch survives")

lto_example()
```

---

## 12. Compiler Verification

### 12.1 Why Verify Compilers?

Compilers are critical infrastructure: a bug in the compiler can introduce bugs in every program it compiles. Compiler verification ensures that the compiled code is semantically equivalent to the source.

### 12.2 Approaches to Compiler Correctness

| Approach | Description | Example |
|----------|-------------|---------|
| **Testing** | Run test suites, fuzz testing | GCC/LLVM test suites, Csmith |
| **Translation validation** | Verify each compilation instance | Alive2 (for LLVM) |
| **Verified compiler** | Mathematically prove correctness | CompCert |
| **Randomized testing** | Generate random programs, compare outputs | Csmith, YARPGen |

### 12.3 CompCert: A Verified Compiler

**CompCert** is a C compiler mathematically proven correct using the Coq proof assistant. Its correctness theorem states:

> For every source program $S$ and compiled program $C$, if $S$ has defined behavior, then the observable behavior of $C$ is identical to that of $S$.

```python
def compiler_verification_overview():
    """Overview of compiler verification approaches."""
    print("=== Compiler Verification ===\n")

    print("CompCert correctness theorem (informal):")
    print("  For all source programs S with defined behavior:")
    print("  semantics(compile(S)) = semantics(S)")
    print()

    print("Translation validation (Alive2 for LLVM):")
    print("  For each optimization pass applied:")
    print("  Verify: semantics(optimized_IR) ⊆ semantics(original_IR)")
    print("  (optimized code may have fewer behaviors, e.g., removing UB)")
    print()

    print("Fuzzing (Csmith):")
    print("  1. Generate random C programs (avoiding UB)")
    print("  2. Compile with multiple compilers/optimization levels")
    print("  3. Run all binaries -- outputs must match")
    print("  4. Any mismatch indicates a compiler bug")
    print()
    print("  Csmith has found 400+ bugs in GCC and LLVM!")
    print()

    # Simple equivalence checker
    print("Simple translation validation example:")
    print("  Original:   x = a + 0")
    print("  Optimized:  x = a")
    print("  Valid? YES (adding zero is identity)")
    print()
    print("  Original:   x = a * 2")
    print("  Optimized:  x = a << 1")
    print("  Valid? YES (for unsigned; need to check overflow for signed)")
    print()
    print("  Original:   x = a / b")
    print("  Optimized:  x = a >> log2(b)  (when b is power of 2)")
    print("  Valid? Only if a >= 0 and b > 0 (signed division rounds toward zero)")

compiler_verification_overview()
```

### 12.4 Alive2: LLVM IR Verification

**Alive2** automatically verifies LLVM optimization passes by checking that the optimized IR refines the original IR for all possible inputs.

```python
def alive2_example():
    """Simulate Alive2-style verification."""
    print("=== Alive2-style Verification ===\n")

    optimizations = [
        {
            'name': 'Strength reduction: x * 2 -> x + x',
            'original': lambda x: x * 2,
            'optimized': lambda x: x + x,
            'valid': True,
        },
        {
            'name': 'x / 2 -> x >> 1 (signed)',
            'original': lambda x: x // 2 if x >= 0 else -((-x) // 2),
            'optimized': lambda x: x >> 1,
            'valid': False,  # Differs for negative odd numbers!
        },
        {
            'name': 'x + 0 -> x',
            'original': lambda x: x + 0,
            'optimized': lambda x: x,
            'valid': True,
        },
    ]

    for opt in optimizations:
        print(f"Optimization: {opt['name']}")

        # Test with various inputs
        test_values = list(range(-10, 11)) + [127, -128, 0, 1, -1]
        all_match = True
        counterexample = None

        for val in test_values:
            try:
                orig_result = opt['original'](val)
                opt_result = opt['optimized'](val)
                if orig_result != opt_result:
                    all_match = False
                    counterexample = (val, orig_result, opt_result)
                    break
            except Exception:
                pass

        if all_match:
            print(f"  Result: VALID (all {len(test_values)} test values match)")
        else:
            val, orig, optim = counterexample
            print(f"  Result: INVALID!")
            print(f"  Counterexample: x = {val}")
            print(f"    Original:  {orig}")
            print(f"    Optimized: {optim}")
        print()

alive2_example()
```

---

## 13. Summary

Modern compiler infrastructure has evolved from monolithic, language-specific compilers to modular, reusable frameworks:

| Component | Purpose | Key Example |
|-----------|---------|-------------|
| **LLVM IR** | Universal optimization target | Used by Clang, Rust, Swift, Julia |
| **MLIR** | Multi-level IR framework | TensorFlow, hardware compilers |
| **Passes** | Modular optimization | Constant folding, DCE, inlining |
| **DSLs** | Domain-specific expressiveness | SQL, HTML, shader languages |
| **ANTLR** | Parser generation | Language tooling |
| **Tree-sitter** | Incremental parsing | Editor integration |
| **LSP** | Editor-language bridge | VS Code, vim, emacs |
| **PGO** | Runtime-informed optimization | 10-20% speedup |
| **LTO** | Cross-module optimization | Whole-program analysis |
| **Verification** | Compiler correctness | CompCert, Alive2, Csmith |

Key principles:

1. **Modularity**: Separating frontends, optimizers, and backends enables reuse and rapid language development.
2. **IR design matters**: A well-designed IR (like LLVM IR) becomes the foundation for an entire ecosystem.
3. **Multiple levels of abstraction**: MLIR recognizes that different domains need different IR levels.
4. **Tooling is essential**: Modern language development is as much about tooling (LSP, Tree-sitter, formatters) as about the compiler itself.
5. **Optimization is never-ending**: PGO, LTO, and runtime optimization continue to find improvements that static compilation alone cannot.
6. **Correctness is paramount**: As compilers grow more complex, formal verification and automated testing become increasingly important.

---

## 14. Exercises

### Exercise 1: LLVM IR by Hand

Write LLVM IR (textual form) for the following functions:

(a) `int max(int a, int b)` -- returns the larger of two integers using a conditional branch and phi node.

(b) `int sum_array(int* arr, int n)` -- sums all elements of an array using a loop with phi nodes.

(c) `int fibonacci(int n)` -- iterative Fibonacci using a loop.

Verify your IR is syntactically correct by checking it follows SSA form (each `%name` defined exactly once).

### Exercise 2: Optimization Pass

Implement a **strength reduction** pass (in Python, simulating LLVM IR) that:
(a) Replaces `x * 2` with `x + x`.
(b) Replaces `x * power_of_2` with `x << log2(power_of_2)`.
(c) Replaces `x / power_of_2` with `x >> log2(power_of_2)` (unsigned only).

Test on a function that uses these patterns and verify the output is correct.

### Exercise 3: DSL Design and Implementation

Design and implement a DSL for one of the following domains:

(a) **State machines**: A DSL for defining finite state machines with states, transitions, and actions.
(b) **Build systems**: A simplified Makefile-like DSL for defining build rules and dependencies.
(c) **Data validation**: A DSL for defining validation rules on structured data (like JSON Schema but simpler).

Your implementation should include: a parser (can use Python's `re` or a simple recursive descent), an AST, and a code generator that produces runnable Python code.

### Exercise 4: Incremental Compilation

Extend the incremental compiler from Section 9 to:
(a) Track dependencies at the symbol level (not just file level) -- if function `foo` in `a.c` changes but function `bar` in `a.c` doesn't, only recompile files that depend on `foo`.
(b) Handle circular dependencies (detect and report them).
(c) Persist the dependency graph to disk so it survives across invocations.

### Exercise 5: PGO Simulator

Build a PGO simulator that:
(a) Takes a simple program (represented as a CFG with basic blocks and branch probabilities).
(b) Simulates execution with given input to collect branch frequencies.
(c) Uses the profile to reorder basic blocks (hot paths first, cold paths moved to the end).
(d) Calculates the expected improvement in instruction cache hit rate.

### Exercise 6: Translation Validator

Implement a simple translation validator that checks whether two simple expressions are equivalent:
(a) Support integer arithmetic: `+`, `-`, `*`, `/`, `<<`, `>>`.
(b) Support constant folding verification (e.g., `3 + 4` equals `7`).
(c) Support algebraic identities (e.g., `x + 0 = x`, `x * 1 = x`).
(d) Use symbolic execution with concrete test values to find counterexamples.
(e) Report whether the transformation is valid or provide a counterexample.

---

## 15. References

1. Lattner, C. (2002). "LLVM: An Infrastructure for Multi-Level Intermediate Representation." Master's thesis, University of Illinois.
2. Lattner, C., Amini, M., Bondhugula, U., et al. (2021). "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation." *CGO*.
3. LLVM Language Reference Manual. [llvm.org/docs/LangRef.html](https://llvm.org/docs/LangRef.html).
4. Leroy, X. (2009). "Formal Verification of a Realistic Compiler." *Communications of the ACM*, 52(7).
5. Lopes, N. V., Lee, J., Hur, C.-K., Liu, Z., & Regehr, J. (2021). "Alive2: Bounded Translation Validation for LLVM." *PLDI*.
6. Yang, X., Chen, Y., Eide, E., & Regehr, J. (2011). "Finding and Understanding Bugs in C Compilers." *PLDI*.
7. Parr, T. (2013). *The Definitive ANTLR 4 Reference*. Pragmatic Bookshelf.
8. Brunsfeld, M. (2018). "Tree-sitter -- A new parsing system for programming tools." GitHub.
9. Microsoft. "Language Server Protocol Specification." [microsoft.github.io/language-server-protocol](https://microsoft.github.io/language-server-protocol/).
10. Stallman, R. M. (2023). *GCC Internals Manual*. Free Software Foundation.

---

[Previous: 15. Interpreters and Virtual Machines](./15_Interpreters_and_Virtual_Machines.md) | [Overview](./00_Overview.md)
