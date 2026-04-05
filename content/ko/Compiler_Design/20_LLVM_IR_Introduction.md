# LLVM IR 입문(LLVM IR Introduction)

**이전**: [19. 레지스터 할당](./19_Register_Allocation.md) | **다음**: [21. LLVM 백엔드](./21_LLVM_Backend.md)

---

LLVM IR(중간 표현, Intermediate Representation)은 LLVM 컴파일러 인프라의 공용 언어(lingua franca)입니다. 타입이 있는 SSA 기반 표현으로, 고수준 소스 언어와 저수준 기계 코드 사이에 위치합니다. LLVM IR을 읽고 쓰는 법을 배우는 것은 언어 프론트엔드 구축, 최적화 패스 작성, 현대 컴파일러의 IR 수준 동작 이해에 필수적입니다.

이 레슨에서는 LLVM IR 구문, 타입 시스템, 명령어, 제어 흐름, 함수, 그리고 간단한 언어에서 IR을 생성하는 방법을 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [16. 현대 컴파일러 인프라](./16_Modern_Compiler_Infrastructure.md), [18. SSA 형식](./18_SSA_Form.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. LLVM IR 텍스트 형식을 읽고 이해한다
2. 정수, 부동소수점, 포인터, 배열, 구조체를 포함한 LLVM 타입 시스템을 설명한다
3. 산술, 비교, 변환 명령어를 작성한다
4. 분기, 파이 노드, 함수 호출로 제어 흐름을 구성한다
5. `llvmlite` Python 라이브러리를 사용하여 프로그래밍 방식으로 LLVM IR을 생성한다
6. `lli`와 `llc`를 사용하여 LLVM IR을 컴파일하고 실행한다

---

## 목차

1. [LLVM IR 개요](#1-llvm-ir-개요)
2. [모듈 구조](#2-모듈-구조)
3. [타입 시스템](#3-타입-시스템)
4. [명령어](#4-명령어)
5. [제어 흐름](#5-제어-흐름)
6. [함수](#6-함수)
7. [메모리 연산](#7-메모리-연산)
8. [llvmlite로 IR 생성](#8-llvmlite로-ir-생성)
9. [IR 컴파일 및 실행](#9-ir-컴파일-및-실행)
10. [요약](#10-요약)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. LLVM IR 개요

### 1.1 세 가지 동등한 형태

LLVM IR은 세 가지 동등한 형태로 존재합니다:

| 형태 | 확장자 | 용도 |
|------|--------|------|
| **텍스트** (사람이 읽을 수 있음) | `.ll` | 읽기, 디버깅, 학습 |
| **비트코드** (바이너리) | `.bc` | 저장, 빠른 로딩 |
| **인-메모리** (C++ 객체) | N/A | 패스에 의한 조작 |

다음과 같이 서로 변환할 수 있습니다:

```bash
# C source -> LLVM IR (textual)
clang -S -emit-llvm -O0 hello.c -o hello.ll

# LLVM IR -> bitcode
llvm-as hello.ll -o hello.bc

# Bitcode -> LLVM IR
llvm-dis hello.bc -o hello.ll

# LLVM IR -> native object
llc hello.ll -o hello.o

# Run LLVM IR directly
lli hello.ll
```

### 1.2 핵심 속성

- **SSA 형식**: 모든 가상 레지스터는 정확히 한 번만 할당
- **타입 지정**: 모든 값에 명시적 타입이 있음
- **타겟 독립적**: IR은 이식 가능(타겟 트리플이 백엔드를 지정)
- **무한 레지스터**: 가상 레지스터에 `%name` 또는 `%number` 사용
- **3-주소 코드**: 대부분의 명령어가 두 피연산자를 받아 하나의 결과를 생성

### 1.3 최소 예제

```llvm
; hello.ll - A minimal LLVM IR program
target triple = "x86_64-unknown-linux-gnu"

@.str = private unnamed_addr constant [14 x i8] c"Hello, LLVM!\0A\00"

declare i32 @printf(ptr, ...)

define i32 @main() {
entry:
  %call = call i32 (ptr, ...) @printf(ptr @.str)
  ret i32 0
}
```

---

## 2. 모듈 구조(Module Structure)

### 2.1 최상위 엔티티

LLVM 모듈은 다음을 포함합니다:

```llvm
; Target specification
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Global variables
@global_var = global i32 42
@const_str = constant [6 x i8] c"hello\00"

; Type definitions (named struct types)
%struct.Point = type { i32, i32 }

; Function declarations (external)
declare i32 @puts(ptr)

; Function definitions
define i32 @add(i32 %a, i32 %b) {
entry:
  %result = add i32 %a, %b
  ret i32 %result
}
```

### 2.2 식별자(Identifiers)

```llvm
; Global identifiers start with @
@global_var = global i32 0
@function_name = ...

; Local identifiers start with %
%local_var = add i32 %a, %b
%struct.MyType = type { i32, float }

; Unnamed temporaries use numbers
%0 = add i32 %a, %b
%1 = mul i32 %0, 2
```

### 2.3 주석과 메타데이터(Comments and Metadata)

```llvm
; This is a comment (semicolon to end of line)

; Metadata nodes
!0 = !{!"my_metadata", i32 42}
!dbg = !{!0}

; Instructions can reference metadata
%x = add i32 %a, %b, !dbg !0
```

---

## 3. 타입 시스템(Type System)

### 3.1 정수 타입

```llvm
i1     ; boolean (1 bit)
i8     ; byte
i16    ; short
i32    ; int
i64    ; long
i128   ; 128-bit integer
iN     ; arbitrary width integer (N bits)
```

### 3.2 부동소수점 타입

```llvm
half      ; 16-bit float (IEEE 754)
float     ; 32-bit float
double    ; 64-bit double
fp128     ; 128-bit float (IEEE 754)
x86_fp80  ; 80-bit float (x86 extended)
```

### 3.3 포인터 타입

```llvm
ptr       ; opaque pointer (LLVM 15+)

; Old-style typed pointers (deprecated):
; i32*     ; pointer to i32
; [10 x i8]*  ; pointer to array
```

### 3.4 집합 타입(Aggregate Types)

```llvm
; Arrays: [count x element_type]
[10 x i32]          ; array of 10 i32s
[4 x [3 x float]]   ; 4x3 matrix of floats

; Structs: { type1, type2, ... }
{ i32, float, ptr }              ; struct with 3 fields
{ i32, { i32, i32 } }           ; nested struct

; Packed structs (no padding)
<{ i8, i32 }>

; Named structs
%struct.Point = type { i32, i32 }
```

### 3.5 벡터 타입

```llvm
<4 x i32>       ; vector of 4 i32s (for SIMD)
<8 x float>     ; vector of 8 floats
<2 x double>    ; vector of 2 doubles
```

### 3.6 기타 타입

```llvm
void       ; no value (for function return types)
label      ; basic block label
token      ; used for exception handling
```

---

## 4. 명령어(Instructions)

### 4.1 산술 연산

```llvm
; Integer arithmetic
%sum  = add i32 %a, %b          ; addition
%diff = sub i32 %a, %b          ; subtraction
%prod = mul i32 %a, %b          ; multiplication
%quot = sdiv i32 %a, %b         ; signed division
%rem  = srem i32 %a, %b         ; signed remainder
%uqot = udiv i32 %a, %b        ; unsigned division
%urem = urem i32 %a, %b        ; unsigned remainder

; Flags for undefined behavior optimization
%sum_nsw = add nsw i32 %a, %b   ; no signed wrap
%sum_nuw = add nuw i32 %a, %b   ; no unsigned wrap

; Floating-point arithmetic
%fsum  = fadd float %x, %y
%fdiff = fsub float %x, %y
%fprod = fmul float %x, %y
%fquot = fdiv float %x, %y
%frem  = frem float %x, %y      ; floating-point remainder
```

### 4.2 비트 연산

```llvm
%and = and i32 %a, %b
%or  = or  i32 %a, %b
%xor = xor i32 %a, %b
%shl = shl i32 %a, 2            ; shift left
%lshr = lshr i32 %a, 2          ; logical shift right
%ashr = ashr i32 %a, 2          ; arithmetic shift right
```

### 4.3 비교 연산

```llvm
; Integer comparison (returns i1)
%eq  = icmp eq  i32 %a, %b      ; equal
%ne  = icmp ne  i32 %a, %b      ; not equal
%slt = icmp slt i32 %a, %b      ; signed less than
%sle = icmp sle i32 %a, %b      ; signed less or equal
%sgt = icmp sgt i32 %a, %b      ; signed greater than
%sge = icmp sge i32 %a, %b      ; signed greater or equal
%ult = icmp ult i32 %a, %b      ; unsigned less than

; Floating-point comparison
%feq = fcmp oeq float %x, %y    ; ordered and equal
%flt = fcmp olt float %x, %y    ; ordered and less than
%fun = fcmp uno float %x, %y    ; unordered (either is NaN)
```

### 4.4 변환 연산

```llvm
; Integer conversions
%ext   = sext i8 %val to i32       ; sign extend
%zext  = zext i8 %val to i32       ; zero extend
%trunc = trunc i32 %val to i8      ; truncate

; Float conversions
%fpext  = fpext float %f to double   ; widen
%fptrunc = fptrunc double %d to float ; narrow

; Integer <-> Float
%sitofp = sitofp i32 %i to float    ; signed int to float
%fptosi = fptosi float %f to i32    ; float to signed int

; Pointer conversions
%ptoi = ptrtoint ptr %p to i64      ; pointer to integer
%itop = inttoptr i64 %i to ptr      ; integer to pointer
```

---

## 5. 제어 흐름(Control Flow)

### 5.1 기본 블록과 레이블

```llvm
define i32 @example(i32 %n) {
entry:                              ; label for the entry block
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %positive, label %negative

positive:                           ; label for the positive block
  %result1 = mul i32 %n, 2
  br label %merge

negative:                           ; label for the negative block
  %result2 = sub i32 0, %n
  br label %merge

merge:                              ; label for the merge block
  %result = phi i32 [%result1, %positive], [%result2, %negative]
  ret i32 %result
}
```

### 5.2 분기 명령어(Branch Instructions)

```llvm
; Unconditional branch
br label %target

; Conditional branch
br i1 %condition, label %true_block, label %false_block

; Switch (multi-way branch)
switch i32 %val, label %default [
  i32 0, label %case0
  i32 1, label %case1
  i32 2, label %case2
]
```

### 5.3 파이 노드(Phi Nodes)

```llvm
; Phi selects value based on which predecessor block we came from
%x = phi i32 [%val1, %block1], [%val2, %block2]

; Loop example
define i32 @sum_to_n(i32 %n) {
entry:
  br label %loop

loop:
  %i = phi i32 [0, %entry], [%next_i, %loop]
  %sum = phi i32 [0, %entry], [%next_sum, %loop]
  %next_sum = add i32 %sum, %i
  %next_i = add i32 %i, 1
  %cond = icmp slt i32 %next_i, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret i32 %next_sum
}
```

### 5.4 선택 명령어(Select Instruction)

```llvm
; Ternary operator (no branch needed)
%max = select i1 %cmp, i32 %a, i32 %b
; Equivalent to: max = cmp ? a : b
```

---

## 6. 함수(Functions)

### 6.1 함수 정의

```llvm
; Basic function
define i32 @add(i32 %a, i32 %b) {
  %result = add i32 %a, %b
  ret i32 %result
}

; Function with attributes
define i32 @factorial(i32 %n) nounwind readnone {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %base, label %recurse

base:
  ret i32 1

recurse:
  %n_minus_1 = sub i32 %n, 1
  %sub_result = call i32 @factorial(i32 %n_minus_1)
  %result = mul i32 %n, %sub_result
  ret i32 %result
}
```

### 6.2 호출 규약(Calling Conventions)

```llvm
; Default C calling convention
define i32 @func(i32 %a) {
  ret i32 %a
}

; Fast calling convention
define fastcc i32 @fast_func(i32 %a) {
  ret i32 %a
}

; Tail call optimization
define fastcc i32 @tail_recursive(i32 %n, i32 %acc) {
entry:
  %cmp = icmp eq i32 %n, 0
  br i1 %cmp, label %base, label %recurse

base:
  ret i32 %acc

recurse:
  %n1 = sub i32 %n, 1
  %acc1 = mul i32 %acc, %n
  %result = tail call fastcc i32 @tail_recursive(i32 %n1, i32 %acc1)
  ret i32 %result
}
```

### 6.3 함수 선언(Function Declarations)

```llvm
; External function (defined elsewhere)
declare i32 @printf(ptr, ...)
declare ptr @malloc(i64)
declare void @free(ptr)
```

---

## 7. 메모리 연산(Memory Operations)

### 7.1 스택 할당(Stack Allocation)

```llvm
; alloca allocates on the stack
%ptr = alloca i32                ; allocate space for one i32
%arr = alloca [10 x i32]        ; allocate array of 10 i32s
%aligned = alloca i32, align 16  ; with alignment
```

### 7.2 로드와 저장(Load and Store)

```llvm
; Store a value to memory
store i32 42, ptr %ptr

; Load a value from memory
%val = load i32, ptr %ptr

; Volatile (cannot be optimized away)
store volatile i32 42, ptr %ptr
%val2 = load volatile i32, ptr %ptr
```

### 7.3 GetElementPtr (GEP)

GEP는 메모리에 접근하지 않고 주소를 계산합니다:

```llvm
; Array element access: arr[5]
%arr_ptr = alloca [10 x i32]
%elem_ptr = getelementptr [10 x i32], ptr %arr_ptr, i64 0, i64 5
%elem = load i32, ptr %elem_ptr

; Struct field access: point.y (second field)
%struct.Point = type { i32, i32 }
%point = alloca %struct.Point
%y_ptr = getelementptr %struct.Point, ptr %point, i64 0, i32 1
store i32 42, ptr %y_ptr

; Pointer arithmetic: ptr + offset
%next = getelementptr i32, ptr %base, i64 1
```

---

## 8. llvmlite로 IR 생성

### 8.1 설정

```python
# pip install llvmlite
from llvmlite import ir

# Create a module
module = ir.Module(name="my_module")
module.triple = "x86_64-unknown-linux-gnu"
```

### 8.2 함수 빌드

```python
from llvmlite import ir

def build_add_function():
    """Build: int add(int a, int b) { return a + b; }"""
    module = ir.Module(name="example")

    # Define function type: i32 (i32, i32)
    func_type = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32)])

    # Create the function
    func = ir.Function(module, func_type, name="add")
    func.args[0].name = "a"
    func.args[1].name = "b"

    # Create entry basic block
    block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(block)

    # Build instructions
    result = builder.add(func.args[0], func.args[1], name="result")
    builder.ret(result)

    print(module)
    return module

# Output:
# define i32 @add(i32 %a, i32 %b) {
# entry:
#   %result = add i32 %a, %b
#   ret i32 %result
# }
```

### 8.3 llvmlite로 제어 흐름 구성

```python
def build_max_function():
    """Build: int max(int a, int b) { return a > b ? a : b; }"""
    module = ir.Module(name="example")
    i32 = ir.IntType(32)

    func_type = ir.FunctionType(i32, [i32, i32])
    func = ir.Function(module, func_type, name="max")
    a, b = func.args

    entry = func.append_basic_block("entry")
    then_bb = func.append_basic_block("then")
    else_bb = func.append_basic_block("else")
    merge_bb = func.append_basic_block("merge")

    # Entry block
    builder = ir.IRBuilder(entry)
    cmp = builder.icmp_signed(">", a, b, name="cmp")
    builder.cbranch(cmp, then_bb, else_bb)

    # Then block
    builder = ir.IRBuilder(then_bb)
    builder.branch(merge_bb)

    # Else block
    builder = ir.IRBuilder(else_bb)
    builder.branch(merge_bb)

    # Merge block with phi
    builder = ir.IRBuilder(merge_bb)
    phi = builder.phi(i32, name="result")
    phi.add_incoming(a, then_bb)
    phi.add_incoming(b, else_bb)
    builder.ret(phi)

    print(module)
    return module
```

### 8.4 llvmlite로 루프 구성

```python
def build_factorial():
    """Build iterative factorial."""
    module = ir.Module(name="factorial")
    i32 = ir.IntType(32)

    func_type = ir.FunctionType(i32, [i32])
    func = ir.Function(module, func_type, name="factorial")
    n = func.args[0]
    n.name = "n"

    entry = func.append_basic_block("entry")
    loop = func.append_basic_block("loop")
    exit_bb = func.append_basic_block("exit")

    # Entry
    builder = ir.IRBuilder(entry)
    builder.branch(loop)

    # Loop
    builder = ir.IRBuilder(loop)
    i = builder.phi(i32, name="i")
    acc = builder.phi(i32, name="acc")

    i.add_incoming(ir.Constant(i32, 1), entry)
    acc.add_incoming(ir.Constant(i32, 1), entry)

    new_acc = builder.mul(acc, i, name="new_acc")
    new_i = builder.add(i, ir.Constant(i32, 1), name="new_i")
    cond = builder.icmp_signed("<=", new_i, n, name="cond")

    i.add_incoming(new_i, loop)
    acc.add_incoming(new_acc, loop)

    builder.cbranch(cond, loop, exit_bb)

    # Exit
    builder = ir.IRBuilder(exit_bb)
    builder.ret(new_acc)

    print(module)
    return module
```

---

## 9. IR 컴파일 및 실행

### 9.1 명령줄 도구 사용

```bash
# Write IR to a file
cat > example.ll << 'EOF'
define i32 @main() {
  ret i32 42
}
EOF

# Interpret directly
lli example.ll
echo $?  # prints 42

# Compile to object file
llc -filetype=obj example.ll -o example.o

# Link and run
gcc example.o -o example
./example
echo $?  # prints 42

# Optimize
opt -O2 example.ll -S -o example_opt.ll
```

### 9.2 llvmlite로 JIT 실행

```python
from llvmlite import ir, binding
import ctypes

def jit_execute():
    """Build a function and execute it via JIT."""
    # Initialize LLVM
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmprinter()

    # Build the module (reuse build_add_function)
    module = build_add_function()

    # Parse the IR
    llvm_ir = str(module)
    mod = binding.parse_assembly(llvm_ir)
    mod.verify()

    # Create execution engine
    target = binding.Target.from_default_triple()
    target_machine = target.create_target_machine()
    engine = binding.create_mcjit_compiler(mod, target_machine)

    # Get function pointer
    func_ptr = engine.get_function_address("add")

    # Call via ctypes
    cfunc = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_int32, ctypes.c_int32)(func_ptr)
    result = cfunc(3, 4)
    print(f"add(3, 4) = {result}")  # prints: add(3, 4) = 7
```

---

## 10. 요약

- LLVM IR은 **타입이 있는 SSA 기반** 중간 표현입니다
- 텍스트(`.ll`), 비트코드(`.bc`), 인-메모리의 세 가지 형태로 존재합니다
- **타입 시스템**은 정수, 부동소수점, 포인터, 배열, 구조체, 벡터를 포함합니다
- **명령어**는 산술, 비트, 비교, 변환, 메모리 연산을 포괄합니다
- **제어 흐름**은 기본 블록, 분기, 파이 노드, switch 문을 사용합니다
- **GEP**(getelementptr)는 배열과 구조체의 주소를 계산합니다
- **llvmlite**를 사용하여 Python에서 LLVM IR을 생성하고 JIT 실행할 수 있습니다
- 명령줄 도구(`llc`, `lli`, `opt`)로 IR을 컴파일, 실행, 최적화합니다

---

## 11. 연습 문제

1. **IR 읽기**: C 함수를 `clang -S -emit-llvm`으로 컴파일하고 각 명령어에 주석을 달아보세요.

2. **수동 IR 작성**: 루프를 사용하여 두 정수의 GCD를 계산하는 LLVM IR을 작성하세요.

3. **llvmlite 피보나치**: `llvmlite`를 사용하여 반복적 피보나치 함수의 IR을 생성하고 JIT 실행하세요.

4. **구조체 접근**: 세 개의 필드를 가진 구조체를 생성하고, 값을 저장하고 읽어오는 LLVM IR을 작성하세요.

5. **배열 합계**: 포인터와 길이가 주어진 i32 배열의 모든 요소를 합산하는 함수의 LLVM IR을 작성하세요.

---

## 12. 참고 자료

1. LLVM Language Reference Manual: https://llvm.org/docs/LangRef.html
2. LLVM Tutorial: Implementing a Language with LLVM: https://llvm.org/docs/tutorial/
3. llvmlite documentation: https://llvmlite.readthedocs.io/
4. Lattner, C., Adve, V. (2004). "LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation." *CGO*.
5. LLVM IR Tutorial (Mapping High-Level Constructs to LLVM IR): https://mapping-high-level-constructs-to-llvm-ir.readthedocs.io/

---

**이전**: [19. 레지스터 할당](./19_Register_Allocation.md) | **다음**: [21. LLVM 백엔드](./21_LLVM_Backend.md)
