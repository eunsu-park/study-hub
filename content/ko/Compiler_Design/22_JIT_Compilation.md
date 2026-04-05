# JIT 컴파일(JIT Compilation)

**이전**: [21. LLVM 백엔드](./21_LLVM_Backend.md) | **다음**: [23. 타입 추론](./23_Type_Inference.md)

---

JIT(Just-In-Time) 컴파일은 해석(interpretation)과 사전 컴파일(AOT, Ahead-Of-Time compilation) 사이의 간극을 메웁니다. JIT 컴파일러는 런타임에 코드를 네이티브 기계 명령어로 변환하여, 실제 실행 동작에 기반한 적응적 최적화(adaptive optimization)를 가능하게 합니다. JIT 컴파일은 Java(HotSpot), JavaScript(V8, SpiderMonkey), Python(PyPy), Julia 같은 언어를 구동하며, 동적 최적화 시스템의 중추입니다.

이 레슨은 JIT 기초, 추적(tracing) 및 메서드(method) JIT 아키텍처, LLVM의 JIT 인프라(ORC), libgccjit, 그리고 실용적인 JIT 구현 기법을 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [15. 인터프리터와 가상 머신](./15_Interpreters_and_Virtual_Machines.md), [20. LLVM IR 입문](./20_LLVM_IR_Introduction.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 해석, AOT 컴파일, JIT 컴파일 사이의 트레이드오프를 설명한다
2. 메서드 JIT과 추적 JIT 아키텍처를 기술한다
3. LLVM의 ORC JIT API를 사용하여 간단한 JIT 컴파일러를 구현한다
4. 계층적 컴파일(tiered compilation)과 스택 위 교체(OSR, On-Stack Replacement)를 이해한다
5. 경량 JIT 컴파일을 위해 libgccjit을 사용한다
6. 프로파일 기반 역최적화(deoptimization)와 추측적 최적화(speculative optimization)를 적용한다

---

## 목차

1. [JIT 기초](#1-jit-기초)
2. [메서드 JIT](#2-메서드-jit)
3. [추적 JIT](#3-추적-jit)
4. [계층적 컴파일](#4-계층적-컴파일)
5. [스택 위 교체](#5-스택-위-교체)
6. [LLVM ORC JIT](#6-llvm-orc-jit)
7. [libgccjit](#7-libgccjit)
8. [추측적 최적화](#8-추측적-최적화)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. JIT 기초

### 1.1 컴파일 전략 비교

```
Interpretation:     Read → Decode → Execute  (per instruction, at runtime)
AOT Compilation:    Source → Machine Code     (before execution)
JIT Compilation:    Source → [Interpret] → Machine Code  (during execution)
```

| 전략 | 시작 시간 | 최대 성능 | 메모리 사용 | 적응성 |
|------|----------|----------|------------|--------|
| 인터프리터 | 빠름 | 느림 (10-100x) | 낮음 | 없음 |
| AOT 컴파일러 | 느림 (컴파일) | 빠름 | 중간 | 정적만 |
| JIT 컴파일러 | 중간 | 빠름 (때로 AOT보다 빠름) | 높음 | 런타임 프로파일 |

### 1.2 JIT이 AOT를 이길 수 있는 이유

JIT 컴파일러는 다음을 수행할 수 있습니다:
- 관찰된 수신자 타입에 기반한 **가상 메서드 호출 인라이닝**
- 런타임에만 알 수 있는 **상수 값 특수화**
- 런타임 설정에 기반한 **죽은 분기 제거**
- 실제 반복 횟수에 기반한 **핫 루프 최적화**

```python
# Example: AOT cannot optimize this, JIT can
config = load_config()  # value only known at runtime

def process(data):
    if config.mode == "fast":      # JIT can constant-fold this
        return fast_path(data)
    else:
        return slow_path(data)
```

### 1.3 JIT 컴파일 워크플로우

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

## 2. 메서드 JIT

### 2.1 동작 방식

**메서드 JIT**(또는 "베이스라인 JIT(baseline JIT)")은 메서드/함수가 핫해지면 전체를 컴파일합니다:

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

### 2.2 실제 사례

- **Java HotSpot C1**: 간단한 최적화를 적용하는 베이스라인 메서드 컴파일러
- **V8 Sparkplug**: JavaScript를 위한 빠른 베이스라인 컴파일러
- **.NET RyuJIT**: .NET을 위한 메서드 단위(method-at-a-time) JIT

---

## 3. 추적 JIT

### 3.1 개념

**추적 JIT(tracing JIT)**은 핫 루프를 통한 실제 실행 경로를 기록한 후, 그 특정 경로를 컴파일합니다:

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

### 3.2 추적 트리(Trace Trees)

가드(분기 가정)가 실패하면, 추적 JIT는 새로운 추적으로 분기하는 **부분 출구(side exit)**를 통해 추적을 확장할 수 있습니다:

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

### 3.3 추적 JIT 사례

- **LuaJIT**: Lua를 위한 매우 효율적인 추적 JIT
- **PyPy**: Python을 위한 추적 JIT (RPython 메타 추적 기반)
- **SpiderMonkey (초기)**: Firefox의 JS 엔진이 추적을 사용했음 (TraceMonkey)

---

## 4. 계층적 컴파일

### 4.1 다중 계층 아키텍처

현대의 JIT 컴파일러는 여러 컴파일 계층을 사용합니다:

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

### 4.2 Java HotSpot 계층

```
Tier 0: Interpreter (with profiling counters)
    │
Tier 1-3: C1 Compiler (baseline, with varying profiling levels)
    │
Tier 4: C2 Compiler (aggressive optimizations, speculative)
```

### 4.3 V8 JavaScript 계층

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

## 5. 스택 위 교체

### 5.1 문제

장시간 실행되는 루프가 핫해지면, 함수가 반환될 때까지 기다리지 않고 **실행 도중** 해석 코드에서 컴파일된 코드로 전환하고 싶습니다.

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

### 5.2 OSR 진입

스택 위 교체(OSR, On-Stack Replacement)는 실행 도중 인터프리터의 스택 프레임을 컴파일된 프레임으로 교체합니다:

```
Before OSR (interpreted):          After OSR (compiled):
┌─────────────┐                    ┌─────────────┐
│ i = 1000    │                    │ i = 1000    │
│ total = ... │   ──── OSR ────▶   │ total = ... │
│ interpreter │                    │ native code │
│ frame       │                    │ frame       │
└─────────────┘                    └─────────────┘
```

### 5.3 OSR 구현

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

### 6.1 ORC 개요

**ORC**(On-Request Compilation)는 LLVM의 JIT 컴파일 API입니다. 지연 컴파일(lazy compilation), 동시 컴파일(concurrent compilation), 모듈식 JIT 레이어를 지원합니다.

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

### 6.2 C++를 사용한 간단한 ORC JIT

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

### 6.3 llvmlite를 사용한 JIT (Python)

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

### 7.1 개요

**libgccjit**은 GCC의 내장 가능한 JIT 컴파일 라이브러리입니다. 런타임에 GCC의 옵티마이저와 코드 생성기를 사용하여 코드를 빌드하고 컴파일하기 위한 C API를 제공합니다.

### 7.2 기본 사용법

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

## 8. 추측적 최적화

### 8.1 타입 특수화(Type Specialization)

JIT 컴파일러는 런타임 타입에 대해 추측할 수 있습니다:

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

### 8.2 역최적화(Deoptimization)

추측적 가정이 실패하면, JIT는 올바른 상태로 실행을 인터프리터로 되돌리는 **역최적화**를 수행해야 합니다:

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

### 8.3 인라인 캐싱(Inline Caching)

각 호출 지점에서 동적 디스패치의 결과를 캐시합니다:

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

## 9. 요약

- **JIT 컴파일**은 런타임에 코드를 네이티브 기계 명령어로 변환합니다
- **메서드 JIT**은 핫해진 함수 전체를 컴파일합니다
- **추적 JIT**은 루프를 통한 핫 실행 경로를 기록하고 컴파일합니다
- **계층적 컴파일**은 서로 다른 최적화 수준을 위해 여러 컴파일러 계층을 사용합니다
- **OSR**은 해석 코드와 컴파일된 코드 사이의 실행 중 전환을 가능하게 합니다
- **LLVM ORC**는 모듈식, 계층적 JIT 컴파일 프레임워크를 제공합니다
- **libgccjit**은 GCC의 최적화 파이프라인을 내장 가능한 라이브러리로 제공합니다
- **추측적 최적화**와 **역최적화**는 적응적이고 프로파일 기반의 성능을 가능하게 합니다

---

## 10. 연습 문제

1. **JIT 임계값**: 다양한 JIT 컴파일 임계값을 실험하세요. 몇 번의 호출에서 JIT 컴파일이 비용 대비 이득이 되기 시작합니까?

2. **llvmlite를 사용한 간단한 JIT**: 산술 표현식을 네이티브 코드로 컴파일하고 실행하는 JIT을 구축하세요.

3. **추적 기록**: 바이트코드 인터프리터에서 루프를 통한 직선 추적을 기록하는 간단한 추적 기록기를 구현하세요.

4. **OSR 시뮬레이션**: 인터프리터 상태가 컴파일된 프레임 상태로 어떻게 매핑되는지 보여주는 스택 위 교체 시뮬레이션을 작성하세요.

5. **인라인 캐시**: 간단한 동적 디스패치 시스템에서 단형(monomorphic) 인라인 캐시를 구현하세요.

---

## 11. 참고 자료

1. Aycock, J. (2003). "A Brief History of Just-In-Time." *ACM Computing Surveys*, 35(2).
2. Bolz, C. F., et al. (2009). "Tracing the Meta-Level: PyPy's Tracing JIT Compiler." *ICOOOLPS*.
3. Pall, M. (2009). "LuaJIT 2.0 - Trace Compiler." http://luajit.org/
4. LLVM ORC JIT APIs: https://llvm.org/docs/ORCv2.html
5. libgccjit 문서: https://gcc.gnu.org/onlinedocs/jit/

---

**이전**: [21. LLVM 백엔드](./21_LLVM_Backend.md) | **다음**: [23. 타입 추론](./23_Type_Inference.md)
