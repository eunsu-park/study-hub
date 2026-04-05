# JIT Compilation

**Previous**: [21. LLVM Backend](./21_LLVM_Backend.md) | **Next**: [23. Type Inference](./23_Type_Inference.md)

---

Just-In-Time (JIT) compilation bridges the gap between interpretation and ahead-of-time (AOT) compilation. A JIT compiler translates code to native machine instructions at runtime, enabling adaptive optimization based on actual execution behavior. JIT compilation powers languages like Java (HotSpot), JavaScript (V8, SpiderMonkey), Python (PyPy), and Julia, and is the backbone of dynamic optimization systems.

This lesson covers JIT fundamentals, tracing and method JIT architectures, LLVM's JIT infrastructure (ORC), libgccjit, and practical JIT implementation techniques.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**: [15. Interpreters and Virtual Machines](./15_Interpreters_and_Virtual_Machines.md), [20. LLVM IR Introduction](./20_LLVM_IR_Introduction.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the tradeoffs between interpretation, AOT compilation, and JIT compilation
2. Describe method JIT and tracing JIT architectures
3. Implement a simple JIT compiler using LLVM's ORC JIT API
4. Understand tiered compilation and on-stack replacement (OSR)
5. Use libgccjit for lightweight JIT compilation
6. Apply profile-guided deoptimization and speculative optimization

---

## Table of Contents

1. [JIT Fundamentals](#1-jit-fundamentals)
2. [Method JIT](#2-method-jit)
3. [Tracing JIT](#3-tracing-jit)
4. [Tiered Compilation](#4-tiered-compilation)
5. [On-Stack Replacement](#5-on-stack-replacement)
6. [LLVM ORC JIT](#6-llvm-orc-jit)
7. [libgccjit](#7-libgccjit)
8. [Speculative Optimization](#8-speculative-optimization)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. JIT Fundamentals

### 1.1 Compilation Strategies Compared

```
Interpretation:     Read → Decode → Execute  (per instruction, at runtime)
AOT Compilation:    Source → Machine Code     (before execution)
JIT Compilation:    Source → [Interpret] → Machine Code  (during execution)
```

| Strategy | Startup Time | Peak Performance | Memory Use | Adaptivity |
|----------|-------------|-----------------|------------|------------|
| Interpreter | Fast | Slow (10-100x) | Low | None |
| AOT Compiler | Slow (compile) | Fast | Medium | Static only |
| JIT Compiler | Medium | Fast (sometimes faster than AOT) | Higher | Runtime profiles |

### 1.2 Why JIT Can Beat AOT

JIT compilers can:
- **Inline virtual method calls** based on observed receiver types
- **Specialize for constant values** known only at runtime
- **Eliminate dead branches** based on runtime configuration
- **Optimize hot loops** with actual iteration counts

```python
# Example: AOT cannot optimize this, JIT can
config = load_config()  # value only known at runtime

def process(data):
    if config.mode == "fast":      # JIT can constant-fold this
        return fast_path(data)
    else:
        return slow_path(data)
```

### 1.3 JIT Compilation Workflow

```
Source Code
    │
    ▼
┌──────────┐   cold code    ┌────────────┐
│Interpreter│──────────────▶│ Execute     │
│(bytecode) │               │ interpreted │
└──────────┘               └────────────┘
    │ hot code detected
    ▼
┌──────────┐               ┌────────────┐
│   JIT     │──────────────▶│ Execute    │
│ Compiler  │               │ native     │
└──────────┘               └────────────┘
    │ deoptimize (invalid assumption)
    ▼
  back to interpreter
```

---

## 2. Method JIT

### 2.1 How It Works

A **method JIT** (also called "baseline JIT") compiles entire methods/functions when they become hot:

```python
class MethodJIT:
    """
    Simple method-based JIT compiler.
    """
    def __init__(self):
        self.call_counts = {}       # function -> invocation count
        self.compiled_code = {}     # function -> native code pointer
        self.threshold = 1000       # compile after N invocations

    def call_function(self, func, *args):
        """Dispatch: interpret or run compiled code."""
        if func in self.compiled_code:
            return self.compiled_code[func](*args)

        # Count invocations
        self.call_counts[func] = self.call_counts.get(func, 0) + 1

        if self.call_counts[func] >= self.threshold:
            # Hot function! Compile it.
            native = self.compile(func)
            self.compiled_code[func] = native
            return native(*args)

        # Still cold: interpret
        return self.interpret(func, *args)

    def compile(self, func):
        """Compile function to native code."""
        ir = self.generate_ir(func)
        optimized = self.optimize(ir)
        native_code = self.codegen(optimized)
        return native_code
```

### 2.2 Examples in Practice

- **Java HotSpot C1**: Baseline method compiler with simple optimizations
- **V8 Sparkplug**: Fast baseline compiler for JavaScript
- **.NET RyuJIT**: Method-at-a-time JIT for .NET

---

## 3. Tracing JIT

### 3.1 Concept

A **tracing JIT** records the actual execution path through hot loops, then compiles that specific path:

```python
class TracingJIT:
    """
    Tracing JIT compiler.
    Records execution traces through hot loops and compiles them.
    """
    def __init__(self):
        self.loop_counts = {}
        self.traces = {}
        self.threshold = 100

    def execute_loop(self, loop_header, state):
        """Execute a loop, potentially recording/compiling a trace."""
        count = self.loop_counts.get(loop_header, 0) + 1
        self.loop_counts[loop_header] = count

        if loop_header in self.traces:
            return self.execute_trace(self.traces[loop_header], state)

        if count >= self.threshold:
            # Record a trace
            trace = self.record_trace(loop_header, state)
            optimized = self.optimize_trace(trace)
            self.traces[loop_header] = self.compile_trace(optimized)
            return self.execute_trace(self.traces[loop_header], state)

        return self.interpret_loop(loop_header, state)

    def record_trace(self, loop_header, state):
        """Record one iteration of the loop."""
        trace = []
        pc = loop_header
        while True:
            instr = self.fetch(pc)
            trace.append(instr)

            if instr.is_branch:
                # Record which way the branch went (a "guard")
                taken = self.evaluate_branch(instr, state)
                trace.append(Guard(instr.condition, taken))
                pc = instr.target if taken else instr.fallthrough
            else:
                state = self.execute_instr(instr, state)
                pc = pc + 1

            if pc == loop_header:
                break  # Complete loop iteration

        return trace

    def optimize_trace(self, trace):
        """Optimize the linear trace."""
        trace = constant_fold(trace)
        trace = eliminate_redundant_guards(trace)
        trace = dead_code_elimination(trace)
        return trace
```

### 3.2 Trace Trees

When a guard (branch assumption) fails, the tracing JIT can extend the trace with a **side exit** that branches to a new trace:

```
Main trace (loop body):
  ┌──────────────────┐
  │ x = load arr[i]  │
  │ guard(x > 0)─────┼──── Side trace: handle x <= 0
  │ y = sqrt(x)      │
  │ store result[i]  │
  │ i = i + 1        │
  │ guard(i < n)─────┼──── Exit: loop done
  │ jump to top      │
  └──────────────────┘
```

### 3.3 Tracing JIT Examples

- **LuaJIT**: Extremely efficient tracing JIT for Lua
- **PyPy**: Tracing JIT for Python (via RPython meta-tracing)
- **SpiderMonkey (early)**: Firefox's JS engine used tracing (TraceMonkey)

---

## 4. Tiered Compilation

### 4.1 Multi-Tier Architecture

Modern JIT compilers use multiple compilation tiers:

```
Tier 0: Interpreter
  │  (immediate execution, collect profiles)
  ▼
Tier 1: Baseline JIT (fast compile, moderate code quality)
  │  (collect more detailed profiles)
  ▼
Tier 2: Optimizing JIT (slow compile, excellent code quality)
  │  (speculation + deoptimization)
  ▼
Tier 3: Super-optimizing (very slow compile, best code quality)
```

### 4.2 Java HotSpot Tiers

```
Tier 0: Interpreter (with profiling counters)
    │
Tier 1-3: C1 Compiler (baseline, with varying profiling levels)
    │
Tier 4: C2 Compiler (aggressive optimizations, speculative)
```

### 4.3 V8 JavaScript Tiers

```
Tier 0: Ignition (bytecode interpreter)
    │
Tier 1: Sparkplug (fast baseline compiler)
    │
Tier 2: Maglev (mid-tier optimizing compiler)
    │
Tier 3: Turbofan (top-tier optimizing compiler)
```

---

## 5. On-Stack Replacement

### 5.1 The Problem

When a long-running loop becomes hot, we want to switch from interpreted to compiled code **mid-execution** -- without waiting for the function to return.

```python
def long_running():
    total = 0
    for i in range(10_000_000):
        total += compute(i)
        # At iteration 1000, JIT compiles this function.
        # OSR allows switching to compiled code HERE,
        # inside the loop, without restarting.
    return total
```

### 5.2 OSR Entry

On-Stack Replacement (OSR) replaces the interpreter's stack frame with a compiled frame mid-execution:

```
Before OSR (interpreted):          After OSR (compiled):
┌─────────────┐                    ┌─────────────┐
│ i = 1000    │                    │ i = 1000    │
│ total = ... │   ──── OSR ────▶   │ total = ... │
│ interpreter │                    │ native code │
│ frame       │                    │ frame       │
└─────────────┘                    └─────────────┘
```

### 5.3 OSR Implementation

```python
def osr_entry_point(loop_header, interpreter_state):
    """
    Create an OSR entry into compiled code.
    1. Extract live variables from interpreter frame
    2. Create a special compiled version with OSR entry
    3. Jump to the compiled code with the extracted state
    """
    # Extract state from interpreter
    live_vars = extract_live_variables(interpreter_state, loop_header)

    # Compile with OSR entry (skip function prologue, jump to loop)
    if loop_header not in osr_compiled:
        ir = generate_ir_with_osr_entry(loop_header, live_vars)
        osr_compiled[loop_header] = compile_to_native(ir)

    # Transfer execution
    return osr_compiled[loop_header](live_vars)
```

---

## 6. LLVM ORC JIT

### 6.1 ORC Overview

**ORC** (On-Request Compilation) is LLVM's JIT compilation API. It supports lazy compilation, concurrent compilation, and modular JIT layers.

```
┌─────────────────────────────────┐
│        ORC JIT Stack            │
│                                 │
│  ┌──────────────────────────┐   │
│  │  CompileOnDemandLayer    │   │ Lazy compilation
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │  OptimizeLayer           │   │ IR optimization
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │  CompileLayer (IRCompile)│   │ IR -> Object
│  └──────────────────────────┘   │
│  ┌──────────────────────────┐   │
│  │  ObjectLinkingLayer      │   │ Object -> Memory
│  └──────────────────────────┘   │
└─────────────────────────────────┘
```

### 6.2 Simple ORC JIT in C++

```cpp
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"

using namespace llvm;
using namespace llvm::orc;

int main() {
    // Create JIT
    auto JIT = LLJITBuilder().create();
    if (!JIT) {
        errs() << "Failed to create JIT\n";
        return 1;
    }

    // Create a module with a function
    auto Ctx = std::make_unique<LLVMContext>();
    auto M = std::make_unique<Module>("test", *Ctx);

    // Build: int square(int x) { return x * x; }
    auto *FT = FunctionType::get(Type::getInt32Ty(*Ctx),
                                  {Type::getInt32Ty(*Ctx)}, false);
    auto *F = Function::Create(FT, Function::ExternalLinkage, "square", *M);
    auto *BB = BasicBlock::Create(*Ctx, "entry", F);
    IRBuilder<> Builder(BB);
    auto *Arg = F->getArg(0);
    auto *Result = Builder.CreateMul(Arg, Arg);
    Builder.CreateRet(Result);

    // Add module to JIT
    auto TSM = ThreadSafeModule(std::move(M), std::move(Ctx));
    (*JIT)->addIRModule(std::move(TSM));

    // Look up and call the function
    auto Sym = (*JIT)->lookup("square");
    if (!Sym) {
        errs() << "Symbol not found\n";
        return 1;
    }

    auto *SquareFn = Sym->toPtr<int(int)>();
    int result = SquareFn(7);
    outs() << "square(7) = " << result << "\n";  // 49

    return 0;
}
```

### 6.3 JIT with llvmlite (Python)

```python
from llvmlite import ir, binding
import ctypes

def create_jit_engine():
    """Create LLVM JIT execution engine."""
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()

    return target_machine

def jit_compile_and_run(ir_module, func_name, arg_types, ret_type, *args):
    """JIT compile an IR module and call a function."""
    tm = create_jit_engine()

    mod = binding.parse_assembly(str(ir_module))
    mod.verify()

    engine = binding.create_mcjit_compiler(mod, tm)
    engine.finalize_object()

    func_ptr = engine.get_function_address(func_name)

    # Build ctypes signature
    ctype_map = {ir.IntType(32): ctypes.c_int32,
                 ir.IntType(64): ctypes.c_int64,
                 ir.DoubleType(): ctypes.c_double}

    cfunc_type = ctypes.CFUNCTYPE(
        ctype_map[ret_type],
        *[ctype_map[t] for t in arg_types]
    )
    cfunc = cfunc_type(func_ptr)
    return cfunc(*args)
```

---

## 7. libgccjit

### 7.1 Overview

**libgccjit** is GCC's embeddable JIT compilation library. It provides a C API for building and compiling code at runtime using GCC's optimizers and code generators.

### 7.2 Basic Usage

```c
#include <libgccjit.h>
#include <stdio.h>

int main() {
    gcc_jit_context *ctx = gcc_jit_context_acquire();

    // Set optimization level
    gcc_jit_context_set_int_option(ctx,
        GCC_JIT_INT_OPTION_OPTIMIZATION_LEVEL, 2);

    // Create: int square(int x) { return x * x; }
    gcc_jit_type *int_type = gcc_jit_context_get_type(ctx, GCC_JIT_TYPE_INT);

    gcc_jit_param *param_x = gcc_jit_context_new_param(ctx, NULL, int_type, "x");

    gcc_jit_function *func = gcc_jit_context_new_function(
        ctx, NULL, GCC_JIT_FUNCTION_EXPORTED,
        int_type, "square", 1, &param_x, 0);

    gcc_jit_block *block = gcc_jit_function_new_block(func, NULL);

    gcc_jit_rvalue *x_param = gcc_jit_param_as_rvalue(param_x);
    gcc_jit_rvalue *result = gcc_jit_context_new_binary_op(
        ctx, NULL, GCC_JIT_BINARY_OP_MULT, int_type, x_param, x_param);

    gcc_jit_block_end_with_return(block, NULL, result);

    // Compile
    gcc_jit_result *jit_result = gcc_jit_context_compile(ctx);

    // Get function pointer
    typedef int (*square_fn)(int);
    square_fn square = (square_fn)gcc_jit_result_get_code(jit_result, "square");

    printf("square(7) = %d\n", square(7));  // 49

    gcc_jit_result_release(jit_result);
    gcc_jit_context_release(ctx);
    return 0;
}
```

---

## 8. Speculative Optimization

### 8.1 Type Specialization

JIT compilers can speculate on runtime types:

```python
# Python-like pseudocode
def add(a, b):
    return a + b  # Could be int, float, string, list...

# JIT observes: add() always called with (int, int)
# Compiles specialized version:
def add_int_int(a: int, b: int) -> int:
    guard(isinstance(a, int))  # Deoptimize if wrong
    guard(isinstance(b, int))
    return int_add(a, b)       # Fast native integer add
```

### 8.2 Deoptimization

When a speculative assumption fails, the JIT must **deoptimize**: transfer execution back to the interpreter with the correct state:

```python
class Deoptimizer:
    """Handle failed speculative optimizations."""

    def deoptimize(self, compiled_frame, reason):
        """
        Transfer from compiled code back to interpreter.
        1. Extract live variables from compiled frame
        2. Reconstruct interpreter state
        3. Resume interpretation
        """
        live_vars = self.extract_state(compiled_frame)
        interpreter_frame = self.build_interpreter_frame(live_vars)

        # Record deoptimization reason for future compilation
        self.record_failure(reason)

        # Invalidate the compiled code if too many deoptimizations
        if self.deopt_count[compiled_frame.method] > threshold:
            self.invalidate(compiled_frame.method)

        return self.resume_interpreter(interpreter_frame)
```

### 8.3 Inline Caching

Cache the result of dynamic dispatch at each call site:

```
# Monomorphic inline cache (one type)
call site: obj.method()
  cache: [type=String, target=String.method]

  if type(obj) == cache.type:
      call cache.target  # Fast path
  else:
      slow_lookup(obj, "method")  # Slow path, update cache
```

---

## 9. Summary

- **JIT compilation** translates code to native machine instructions at runtime
- **Method JIT** compiles entire functions when they become hot
- **Tracing JIT** records and compiles hot execution paths through loops
- **Tiered compilation** uses multiple compiler tiers for different optimization levels
- **OSR** enables switching between interpreted and compiled code mid-execution
- **LLVM ORC** provides a modular, layered JIT compilation framework
- **libgccjit** offers GCC's optimization pipeline as an embeddable library
- **Speculative optimization** and **deoptimization** enable adaptive, profile-driven performance

---

## 10. Exercises

1. **JIT threshold**: Experiment with different JIT compilation thresholds. At what invocation count does JIT compilation pay for itself?

2. **Simple JIT with llvmlite**: Build a JIT that compiles arithmetic expressions to native code and executes them.

3. **Trace recording**: Implement a simple trace recorder for a bytecode interpreter that records straight-line traces through loops.

4. **OSR simulation**: Write a simulation of on-stack replacement, showing how interpreter state maps to compiled frame state.

5. **Inline cache**: Implement a monomorphic inline cache for a simple dynamic dispatch system.

---

## 11. References

1. Aycock, J. (2003). "A Brief History of Just-In-Time." *ACM Computing Surveys*, 35(2).
2. Bolz, C. F., et al. (2009). "Tracing the Meta-Level: PyPy's Tracing JIT Compiler." *ICOOOLPS*.
3. Pall, M. (2009). "LuaJIT 2.0 - Trace Compiler." http://luajit.org/
4. LLVM ORC JIT APIs: https://llvm.org/docs/ORCv2.html
5. libgccjit documentation: https://gcc.gnu.org/onlinedocs/jit/

---

**Previous**: [21. LLVM Backend](./21_LLVM_Backend.md) | **Next**: [23. Type Inference](./23_Type_Inference.md)
