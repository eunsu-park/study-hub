# 현대 컴파일러 인프라(Modern Compiler Infrastructure)

**이전**: [15. 인터프리터와 가상 머신](./15_Interpreters_and_Virtual_Machines.md)

---

현대 컴파일러는 처음부터 만드는 단일 프로그램이 아닙니다. 재사용 가능한 인프라 -- 중간 표현(Intermediate Representation), 최적화 패스(Optimization Pass), 코드 생성기, 도구 프레임워크 -- 를 조립하여 만들어지며, 이를 여러 언어에서 공유할 수 있습니다. LLVM 프로젝트가 이 철학을 잘 보여줍니다: 수십 개의 언어(C, C++, Rust, Swift, Julia, Zig 등)가 동일한 최적화기와 코드 생성기를 공유합니다.

이 레슨에서는 현대 컴파일러를 구동하는 인프라를 탐구합니다: LLVM의 아키텍처와 IR, MLIR의 다중 레벨 접근 방식, GCC 내부, 도메인 특화 언어(DSL) 설계, 컴파일러 구성 도구, 그리고 PGO와 LTO 같은 고급 컴파일 기법들입니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [09. 중간 표현](./09_Intermediate_Representations.md), [11. 코드 생성](./11_Code_Generation.md), [12. 최적화 -- 지역 및 전역](./12_Optimization_Local_and_Global.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. LLVM의 모듈식 아키텍처와 패스 파이프라인 설명
2. 타입, 명령어, SSA 형식을 포함한 LLVM IR 읽기 및 쓰기
3. LLVM 최적화 패스 작성 방법 이해
4. MLIR의 다중 레벨 IR 철학과 다이얼렉트(Dialect) 설명
5. LLVM과 GCC 내부 표현(GIMPLE, RTL) 비교
6. 도메인 특화 언어(DSL) 설계 및 구현
7. 컴파일러 구성 도구(ANTLR, Tree-sitter) 활용
8. 언어 서버 프로토콜(Language Server Protocol, LSP) 이해
9. 증분 컴파일(Incremental Compilation) 전략 설명
10. 프로파일 기반 최적화(Profile-Guided Optimization, PGO)와 링크 타임 최적화(Link-Time Optimization, LTO) 적용
11. 컴파일러 검증 접근 방식 이해

---

## 목차

1. [LLVM 개요](#1-llvm-개요)
2. [LLVM IR 상세](#2-llvm-ir-상세)
3. [LLVM 패스 작성](#3-llvm-패스-작성)
4. [MLIR: 다중 레벨 IR](#4-mlir-다중-레벨-ir)
5. [GCC 내부](#5-gcc-내부)
6. [도메인 특화 언어](#6-도메인-특화-언어)
7. [컴파일러 구성 도구](#7-컴파일러-구성-도구)
8. [언어 서버 프로토콜](#8-언어-서버-프로토콜)
9. [증분 컴파일](#9-증분-컴파일)
10. [프로파일 기반 최적화](#10-프로파일-기반-최적화)
11. [링크 타임 최적화](#11-링크-타임-최적화)
12. [컴파일러 검증](#12-컴파일러-검증)
13. [요약](#13-요약)
14. [연습 문제](#14-연습-문제)
15. [참고 자료](#15-참고-자료)

---

## 1. LLVM 개요

### 1.1 LLVM이란?

**LLVM**(원래는 "Low Level Virtual Machine"이었으나 이제는 단순히 이름)은 모듈식 컴파일러 및 툴체인 기술의 모음입니다. 핵심 아이디어는 언어별 프론트엔드와 타겟별 백엔드를 분리하는 잘 정의된 중간 표현(LLVM IR)입니다.

```
언어 프론트엔드:              LLVM 핵심:              타겟 백엔드:
┌───────┐                                                ┌──────────┐
│ Clang │───┐                                        ┌──▶│  x86-64  │
│ (C/C++)│   │    ┌──────────┐   ┌──────────┐        │   └──────────┘
└───────┘   │    │          │   │          │        │   ┌──────────┐
┌───────┐   ├───▶│ LLVM IR  │──▶│최적화기  │──▶─────┼──▶│  ARM64   │
│ Rust  │───┤    │          │   │ (패스들) │        │   └──────────┘
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

### 1.2 LLVM 아키텍처

LLVM은 여러 핵심 구성 요소로 이루어집니다:

| 구성 요소 | 목적 |
|-----------|---------|
| **LLVM Core** | IR 정의, 최적화 패스, 코드 생성 |
| **Clang** | C/C++/Objective-C 프론트엔드 |
| **LLDB** | 디버거 |
| **libc++** | C++ 표준 라이브러리 |
| **compiler-rt** | 런타임 지원 (새니타이저, 프로파일링) |
| **LLD** | 링커 |
| **MLIR** | 다중 레벨 IR 프레임워크 |
| **Polly** | 폴리헤드럴 최적화 패스 |

### 1.3 3단계 설계

```
1단계: 프론트엔드        2단계: 최적화기        3단계: 백엔드
┌─────────────────┐       ┌──────────────────┐     ┌─────────────────┐
│                 │       │                  │     │                 │
│ 소스 코드       │       │ LLVM IR          │     │ 기계어 코드     │
│     │           │       │     │            │     │     │           │
│     ▼           │       │     ▼            │     │     ▼           │
│  렉싱           │       │ 분석 패스        │     │ 명령어          │
│  파싱           │       │ 변환 패스        │     │ 선택            │
│  의미           │──────▶│                  │────▶│ 레지스터 할당   │
│  분석           │       │ 최적화           │     │ 명령어          │
│  IR 생성        │       │ 수준:            │     │ 스케줄링        │
│                 │       │  -O0, -O1, -O2,  │     │ 코드 방출       │
│                 │       │  -O3, -Os, -Oz   │     │                 │
└─────────────────┘       └──────────────────┘     └─────────────────┘
```

이 설계의 장점: 새로운 언어를 추가하려면 LLVM IR을 생성하는 프론트엔드만 작성하면 됩니다. 새로운 타겟 아키텍처를 추가하려면 백엔드만 작성하면 됩니다. 양쪽 모두 기존의 모든 최적화 혜택을 누립니다.

### 1.4 LLVM으로 컴파일하기 (Clang 사용)

```bash
# C 코드에서 LLVM IR 생성
clang -S -emit-llvm hello.c -o hello.ll

# LLVM IR 최적화
opt -O2 hello.ll -o hello_opt.ll

# LLVM IR을 어셈블리로 컴파일
llc hello_opt.ll -o hello.s

# 또는 직접 오브젝트 파일로 컴파일
clang -c hello.c -O2 -o hello.o

# 최적화 파이프라인 확인
clang -O2 -mllvm -print-after-all hello.c 2>&1 | head -100
```

---

## 2. LLVM IR 상세

### 2.1 IR 구조

LLVM IR은 타입이 있는 SSA 기반의 중간 표현입니다. 세 가지 동형(Isomorphic) 형태로 존재합니다:
- **사람이 읽을 수 있는 텍스트** (`.ll` 파일)
- **밀집 이진 인코딩** (비트코드, `.bc` 파일)
- **메모리 내 데이터 구조** (C++ 객체)

모듈(번역 단위)의 구성:

```llvm
; 모듈 수준 구조
source_filename = "example.c"
target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-macosx14.0.0"

; 전역 변수
@global_var = global i32 42, align 4
@hello_str = private constant [12 x i8] c"Hello World\00"

; 함수 선언 (외부)
declare i32 @printf(i8* nocapture readonly, ...)

; 함수 정의
define i32 @main() {
entry:
  %x = alloca i32, align 4
  store i32 10, i32* %x, align 4
  %val = load i32, i32* %x, align 4
  %result = add nsw i32 %val, 32
  ret i32 %result
}
```

### 2.2 타입 시스템

LLVM IR은 풍부한 타입 시스템을 갖추고 있습니다:

```llvm
; 정수 타입
i1        ; 1비트 (불리언)
i8        ; 8비트 (바이트/char)
i16       ; 16비트 (short)
i32       ; 32비트 (int)
i64       ; 64비트 (long)
i128      ; 128비트

; 부동소수점 타입
half      ; 16비트 float
float     ; 32비트 float
double    ; 64비트 float

; 포인터 타입
ptr       ; 불투명 포인터 (LLVM 15+)
i32*      ; 타입이 있는 포인터 (레거시)

; 배열 타입
[10 x i32]         ; 10개의 i32 배열
[3 x [4 x float]]  ; 3x4 float 행렬

; 구조체 타입
{ i32, float, i8* }           ; 리터럴 구조체
%struct.Point = type { i32, i32 }  ; 이름 있는 구조체

; 벡터 타입 (SIMD)
<4 x float>    ; 4원소 float 벡터
<8 x i32>      ; 8원소 int 벡터

; 함수 타입
i32 (i32, i32)      ; i32 두 개를 받아 i32를 반환하는 함수
void (i8*, ...)     ; 가변 인자 함수
```

### 2.3 SSA와 명령어

LLVM IR의 모든 값은 **정적 단일 대입(Static Single Assignment, SSA)** 형식입니다: 각 변수는 정확히 한 번만 정의됩니다. 파이 노드($\phi$-함수)는 제어 흐름 합류 지점에서 값을 병합합니다.

```llvm
; 산술 명령어
%sum = add i32 %a, %b          ; 정수 덧셈
%diff = sub i32 %a, %b         ; 정수 뺄셈
%prod = mul i32 %a, %b         ; 정수 곱셈
%quot = sdiv i32 %a, %b        ; 부호 있는 정수 나눗셈
%rem = srem i32 %a, %b         ; 부호 있는 나머지

%fsum = fadd double %x, %y     ; 부동소수점 덧셈
%fprod = fmul float %x, %y     ; 부동소수점 곱셈

; nsw/nuw 플래그: 부호 있는/없는 오버플로우 없음 (최적화 활성화)
%safe_add = add nsw i32 %a, %b

; 비교
%cmp = icmp eq i32 %a, %b      ; 정수 비교 (eq, ne, slt, sgt, sle, sge)
%fcmp = fcmp olt double %x, %y ; 부동소수점 비교 (olt, ogt, oeq, ...)

; 비트 연산
%and = and i32 %a, %b
%or = or i32 %a, %b
%xor = xor i32 %a, %b
%shl = shl i32 %a, 2           ; 2비트 왼쪽 시프트

; 변환
%ext = sext i32 %a to i64      ; i32를 i64로 부호 확장
%trunc = trunc i64 %b to i32   ; i64를 i32로 절단
%fp = sitofp i32 %a to double  ; 부호 있는 정수를 부동소수점으로
%int = fptosi double %x to i32 ; 부동소수점을 부호 있는 정수로
%cast = bitcast i32* %p to i8* ; 비트 재해석 (같은 크기)
```

### 2.4 메모리 명령어

```llvm
; 스택 할당
%ptr = alloca i32, align 4         ; 스택에 4바이트 할당
%arr = alloca [100 x i32], align 16 ; 스택에 배열 할당

; 로드와 스토어
store i32 42, i32* %ptr, align 4    ; *ptr = 42
%val = load i32, i32* %ptr, align 4 ; val = *ptr

; GEP (GetElementPtr) -- 메모리 접근 없이 주소 계산
; LLVM에서 가장 중요하고 (혼란스러운) 명령어 중 하나
%struct.Point = type { i32, i32 }

%p = alloca %struct.Point
; 두 번째 필드(y)에 대한 포인터 얻기:
%y_ptr = getelementptr %struct.Point, %struct.Point* %p, i32 0, i32 1
; 첫 번째 인덱스 (i32 0): 배열에서 몇 번째 구조체인지 (0번째)
; 두 번째 인덱스 (i32 1): 몇 번째 필드인지 (1 = 두 번째 필드)
```

### 2.5 제어 흐름

```llvm
; 무조건 분기
br label %target

; 조건부 분기
%cond = icmp slt i32 %i, %n
br i1 %cond, label %loop_body, label %loop_exit

; 파이 노드 (SSA 합류 지점에서의 병합)
define i32 @abs(i32 %x) {
entry:
  %is_neg = icmp slt i32 %x, 0
  br i1 %is_neg, label %negative, label %done

negative:
  %neg_x = sub i32 0, %x
  br label %done

done:
  ; 파이 노드: 값은 어느 선행 블록에서 왔는지에 따라 달라짐
  %result = phi i32 [ %x, %entry ], [ %neg_x, %negative ]
  ret i32 %result
}

; 스위치
switch i32 %val, label %default [
  i32 0, label %case0
  i32 1, label %case1
  i32 2, label %case2
]
```

### 2.6 함수 호출

```llvm
; 직접 호출
%result = call i32 @add(i32 %a, i32 %b)

; 속성 있는 호출
%result = call i32 @pure_func(i32 %x) nounwind readnone

; 꼬리 호출 (꼬리 호출 최적화 활성화)
%result = tail call i32 @recursive_func(i32 %n)

; 인보크 (예외를 던질 수 있는 호출)
%result = invoke i32 @may_throw(i32 %x)
          to label %normal unwind label %exception
```

### 2.7 완전한 LLVM IR 예제

```llvm
; LLVM IR로 작성한 팩토리얼 함수
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

; 반복 버전 (최적화하기 더 쉬움)
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

### 2.8 Python으로 LLVM IR 생성하기

`llvmlite` 라이브러리를 사용하여 프로그래밍 방식으로 LLVM IR을 생성할 수 있습니다:

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

## 3. LLVM 패스 작성

### 3.1 패스 종류

LLVM은 최적화를 IR을 변환하거나 분석하는 **패스(Pass)**로 구성합니다:

| 패스 종류 | 범위 | 예시 |
|-----------|-------|---------|
| **모듈 패스** | 전체 모듈 | 프로시저 간 분석 |
| **함수 패스** | 단일 함수 | 데드 코드 제거 |
| **루프 패스** | 단일 루프 | 루프 언롤링 |
| **기본 블록 패스** | 단일 기본 블록 | 핍홀 최적화 |
| **분석 패스** | 읽기 전용 | 도미네이터 트리 계산 |

### 3.2 패스 파이프라인

LLVM의 최적화기는 신중하게 순서가 정해진 파이프라인으로 패스를 실행합니다:

```
-O2 파이프라인 (간략):
  1. CFG 단순화
  2. SROA (집합체의 스칼라 대체, Scalar Replacement of Aggregates)
  3. 조기 CSE (공통 부분식 제거, Common Subexpression Elimination)
  4. 인라이닝
  5. CFG 단순화
  6. 명령어 결합
  7. 재결합(Reassociate)
  8. 루프 패스:
     a. 루프 회전
     b. LICM (루프 불변 코드 이동, Loop-Invariant Code Motion)
     c. 귀납 변수 단순화
     d. 루프 언롤링
  9. GVN (전역 값 번호 매기기, Global Value Numbering)
  10. 데드 코드 제거
  11. CFG 단순화
  12. ... (추가 패스)
```

### 3.3 Python으로 LLVM 패스 시뮬레이션

실제 LLVM 패스 작성은 C++이 필요하므로, Python으로 개념을 시뮬레이션할 수 있습니다:

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

### 3.4 실제 LLVM 패스 예제 (C++ 스케치)

참고용으로, 실제 LLVM 패스가 C++에서 어떻게 생겼는지 보여줍니다 (새 패스 매니저, LLVM 14+):

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

## 4. MLIR: 다중 레벨 IR

### 4.1 MLIR이 해결하는 문제

서로 다른 도메인은 서로 다른 수준의 추상화가 필요합니다:

```
고수준:    TensorFlow 그래프 연산 (matmul, conv2d, ...)
               ↓
중간 수준: 어파인 루프(Affine loop), 텐서 연산
               ↓
저수준:    LLVM IR (스칼라 연산, 메모리 로드/스토어)
               ↓
기계 수준: 타겟별 기계어 명령어
```

전통적으로 각 수준은 자체 최적화 프레임워크를 가진 자체 IR을 사용합니다. MLIR은 여러 수준을 위한 **단일하고 확장 가능한 프레임워크**를 제공합니다.

### 4.2 MLIR 개념

**다이얼렉트(Dialect)**: MLIR은 연산을 다이얼렉트로 구성합니다 -- 네임스페이스로 묶인 연산, 타입, 속성의 모음입니다.

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

### 4.3 주요 MLIR 다이얼렉트

| 다이얼렉트 | 목적 | 수준 |
|---------|---------|-------|
| `func` | 함수, 호출 | 고수준 |
| `arith` | 산술 연산 | 중간 |
| `affine` | 어파인 루프와 메모리 | 중간 |
| `linalg` | 선형 대수 연산 | 중간-고수준 |
| `tensor` | 텐서 타입과 연산 | 중간-고수준 |
| `memref` | 메모리 참조 | 중간-저수준 |
| `scf` | 구조적 제어 흐름 (for, if, while) | 중간 |
| `cf` | 비구조적 제어 흐름 (branch, switch) | 저수준 |
| `llvm` | LLVM IR 연산 | 저수준 |
| `gpu` | GPU 연산 | 타겟 특화 |

### 4.4 점진적 하강(Progressive Lowering)

MLIR의 핵심 혁신: **점진적 하강**은 일련의 다이얼렉트 변환을 통해 고수준 연산을 저수준 연산으로 변환합니다.

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

## 5. GCC 내부

### 5.1 GCC 아키텍처

GCC(GNU 컴파일러 모음)는 LLVM과 다른 내부 아키텍처를 사용합니다:

```
GCC 컴파일 파이프라인:
┌─────────────────────────────────────────────────────────────┐
│                         GCC                                  │
│                                                              │
│  소스 ──▶ 프론트엔드 ──▶ GENERIC ──▶ GIMPLE ──▶ SSA GIMPLE   │
│           (파서)       (언어별     (단순화된   (파이 노드가  │
│                         특화 AST)  3주소코드)  있는 SSA)     │
│                                                              │
│  SSA GIMPLE ──▶ Tree SSA 최적화 ──▶ 최적화된 GIMPLE          │
│                 (SRA, DCE, PRE, SCCP,                        │
│                  루프 최적화, 벡터화)                         │
│                                                              │
│  최적화된 GIMPLE ──▶ RTL ──▶ RTL 최적화 ──▶ 어셈블리         │
│                       (레지스터 전송       (레지스터 할당,   │
│                        언어)               명령어            │
│                                            스케줄링)         │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 GIMPLE

GIMPLE은 GCC의 고수준 중간 표현입니다. AST의 단순화된 3주소 형식입니다.

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

### 5.3 RTL (레지스터 전송 언어)

RTL은 기계어 명령어에 가까운 GCC의 저수준 IR입니다:

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

### 5.4 LLVM 대 GCC 비교

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

## 6. 도메인 특화 언어

### 6.1 DSL이란?

**도메인 특화 언어(Domain-Specific Language, DSL)**는 특정 문제 영역에 맞춤화된 프로그래밍 언어입니다. 범용 언어(GPL)와 달리, DSL은 일반성을 포기하는 대신 해당 도메인에서의 표현력과 사용 편의성을 얻습니다.

| 종류 | 설명 | 예시 |
|------|-------------|---------|
| **외부 DSL** | 자체 파서를 가진 별도 언어 | SQL, HTML, CSS, 정규식, Makefile |
| **내부/임베디드 DSL** | GPL 내에서 호스팅됨 | SQLAlchemy (Python), Kotlin DSL, Scala implicits |

### 6.2 DSL 설계 원칙

1. **도메인 집중**: 프로그래밍 개념이 아닌 도메인 개념을 직접 표현
2. **추상화**: 무관한 세부 사항 숨기기
3. **표기법**: 도메인 전문가의 표기법에 맞추기
4. **안전성**: 도메인에서 의미 없는 오류 방지
5. **구성**: DSL 요소들을 자연스럽게 결합 가능하도록

### 6.3 외부 DSL 구축

데이터 처리 파이프라인을 정의하는 간단한 DSL을 만들어 봅니다:

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

### 6.4 임베디드 DSL

**임베디드 DSL(EDSL)**은 호스트 언어의 문법을 사용하여 도메인 특화적인 느낌을 만들어냅니다:

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

## 7. 컴파일러 구성 도구

### 7.1 ANTLR

**ANTLR**(ANother Tool for Language Recognition)은 문법 명세로부터 파서를 생성하는 강력한 파서 생성기입니다.

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

ANTLR은 이 문법으로부터 렉서, 파서, 파스 트리를 생성합니다. 다양한 타겟 언어(Java, Python, C++, JavaScript, Go 등)를 지원합니다.

### 7.2 Tree-sitter

**Tree-sitter**는 증분 파싱(Incremental Parsing)을 위해 설계된 파서 생성기입니다 -- 코드 에디터와 언어 도구에 이상적입니다.

주요 특징:
- **증분적**: 파일 편집 후, 변경된 부분만 재파싱
- **오류 허용**: 문법적으로 올바르지 않은 코드에서도 유효한 파스 트리 생성
- **빠름**: 대부분의 파일을 1밀리초 이내에 파싱
- **구체적 문법 트리**: 정확한 왕복 변환을 위해 모든 토큰(공백, 주석) 보존

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

### 7.3 도구 비교

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

## 8. 언어 서버 프로토콜

### 8.1 LSP란?

Microsoft가 개발한 **언어 서버 프로토콜(Language Server Protocol, LSP)**은 코드 에디터와 언어별 도구 간의 인터페이스를 표준화합니다. LSP 이전에는 모든 에디터가 모든 언어에 대한 커스텀 플러그인이 필요했습니다(M개의 에디터 × N개의 언어 = M×N개의 구현). LSP는 이를 M + N으로 줄입니다.

```
LSP 없이:                        LSP 있을 때:

  VS Code ──── C 플러그인          VS Code ──┐
  VS Code ──── Python 플러그인               │
  VS Code ──── Rust 플러그인       Vim ──────┤   LSP    ┌── C 서버
  Vim ──────── C 플러그인          Emacs ────┤ 프로토콜 ├── Python 서버
  Vim ──────── Python 플러그인     Sublime ──┘          ├── Rust 서버
  Vim ──────── Rust 플러그인                            └── ...
  Emacs ────── C 플러그인
  Emacs ────── Python 플러그인     M + N 구현
  Emacs ────── Rust 플러그인       (M * N 대신)
  ...

  M * N 구현
```

### 8.2 LSP 아키텍처

```
┌──────────────┐         JSON-RPC          ┌──────────────────┐
│              │ ◀─── 알림 ──────────────  │                  │
│    에디터    │ ───── 요청 ─────────────▶ │  언어 서버        │
│   (클라이언트)│ ◀─── 응답 ─────────────  │                  │
│              │                            │  - 파서          │
│  VS Code     │  textDocument/didOpen      │  - 타입 검사기   │
│  Vim         │  textDocument/completion   │  - 진단          │
│  Emacs       │  textDocument/definition   │  - 포매터        │
│  Sublime     │  textDocument/references   │  - 리팩토링      │
│              │  textDocument/hover        │                  │
└──────────────┘                            └──────────────────┘
```

### 8.3 LSP 기능

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

### 8.4 간단한 언어 서버 구축

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

## 9. 증분 컴파일

### 9.1 문제

대규모 프로젝트는 컴파일하는 데 오랜 시간이 걸립니다. 파일 하나를 변경한 후 모든 것을 다시 컴파일하는 것은 낭비입니다. **증분 컴파일(Incremental Compilation)**은 변경된 것만 재컴파일합니다.

### 9.2 의존성 추적

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

### 9.3 실무에서의 증분 컴파일

| 시스템 | 전략 |
|--------|----------|
| **Rust (cargo)** | 크레이트 단위 컴파일, 크레이트 내 쿼리 기반 증분 |
| **Go** | 패키지 단위 컴파일, 빠른 전체 빌드 |
| **Java (javac)** | 파일 단위 컴파일, 의존성 추적 |
| **C/C++ (make)** | 파일 단위 컴파일, 타임스탬프 기반 재빌드 |
| **TypeScript** | 프로젝트 단위, 메모리 내 증분 |

---

## 10. 프로파일 기반 최적화

### 10.1 PGO란?

**프로파일 기반 최적화(Profile-Guided Optimization, PGO)**는 런타임 프로파일링 데이터를 사용하여 더 나은 최적화 결정을 내립니다. 컴파일러는 먼저 계측(Instrumented) 코드를 생성하고, 대표적인 입력으로 실행하여 프로파일 데이터를 수집한 다음, 프로파일을 사용하여 재컴파일합니다.

```
1단계: 계측 빌드
  소스 ──▶ 컴파일러 (-fprofile-generate) ──▶ 계측된 바이너리

2단계: 프로파일 수집
  계측된 바이너리 + 훈련 입력 ──▶ 프로파일 데이터 (.profdata)

3단계: 최적화 빌드
  소스 + 프로파일 데이터 ──▶ 컴파일러 (-fprofile-use) ──▶ 최적화된 바이너리
```

### 10.2 PGO 최적화 결정

프로파일 데이터는 여러 최적화를 가능하게 합니다:

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

### 10.3 Clang/GCC에서 PGO 사용하기

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

### 10.4 PGO 시뮬레이션

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

## 11. 링크 타임 최적화

### 11.1 LTO란?

**링크 타임 최적화(Link-Time Optimization, LTO)**는 일부 최적화를 링크 시간으로 미루는데, 이 때 컴파일러는 모든 번역 단위(소스 파일)에 걸쳐 가시성을 갖습니다. 이를 통해 가능해지는 것들:

- 모듈 간 인라이닝
- 프로시저 간 상수 전파
- 모듈 간 데드 함수 제거
- 전체 프로그램 가상 함수 직접 호출 변환(Devirtualization)

```
LTO 없이:
  a.c ──▶ a.o ──┐
  b.c ──▶ b.o ──┤──▶ 링커 ──▶ 바이너리
  c.c ──▶ c.o ──┘
  (각 .o는 독립적으로 최적화됨)

LTO 있을 때:
  a.c ──▶ a.bc ──┐
  b.c ──▶ b.bc ──┤──▶ LTO 최적화기 ──▶ 링커 ──▶ 바이너리
  c.c ──▶ c.bc ──┘
  (.bc = LLVM 비트코드, 함께 최적화됨)
```

### 11.2 전체 LTO 대 ThinLTO

| 측면 | 전체 LTO | ThinLTO |
|--------|----------|---------|
| **범위** | 모든 모듈을 하나로 병합 | 모듈을 별도로 유지 |
| **링크 시간** | 느림 (단일 스레드) | 빠름 (병렬화 가능) |
| **메모리** | 높음 (전체 프로그램이 메모리에) | 낮음 (요약 기반) |
| **최적화 품질** | 최고 (전체 가시성) | 거의 동등 (95%+) |
| **증분** | 없음 (모든 것 재최적화) | 있음 (모듈별) |

```bash
# Full LTO with Clang
clang -flto -O2 a.c b.c c.c -o program

# ThinLTO (recommended for large projects)
clang -flto=thin -O2 a.c b.c c.c -o program
```

### 11.3 LTO가 가능하게 하는 것들

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

## 12. 컴파일러 검증

### 12.1 컴파일러를 검증해야 하는 이유

컴파일러는 중요한 인프라입니다: 컴파일러의 버그는 컴파일되는 모든 프로그램에 버그를 도입할 수 있습니다. 컴파일러 검증은 컴파일된 코드가 소스와 의미론적으로 동등함을 보장합니다.

### 12.2 컴파일러 정확성 접근 방식

| 접근 방식 | 설명 | 예시 |
|----------|-------------|---------|
| **테스팅** | 테스트 스위트 실행, 퍼즈 테스팅 | GCC/LLVM 테스트 스위트, Csmith |
| **번역 검증** | 각 컴파일 인스턴스 검증 | Alive2 (LLVM용) |
| **검증된 컴파일러** | 수학적으로 정확성 증명 | CompCert |
| **무작위 테스팅** | 무작위 프로그램 생성, 출력 비교 | Csmith, YARPGen |

### 12.3 CompCert: 검증된 컴파일러

**CompCert**는 Coq 증명 보조 도구를 사용하여 수학적으로 올바름이 증명된 C 컴파일러입니다. 정확성 정리는 다음과 같습니다:

> 정의된 동작을 가진 모든 소스 프로그램 $S$와 컴파일된 프로그램 $C$에 대해, $C$의 관찰 가능한 동작은 $S$의 동작과 동일합니다.

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

### 12.4 Alive2: LLVM IR 검증

**Alive2**는 최적화된 IR이 모든 가능한 입력에 대해 원래 IR을 정제(Refine)하는지 확인하여 LLVM 최적화 패스를 자동으로 검증합니다.

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

## 13. 요약

현대 컴파일러 인프라는 단일 언어 특화 컴파일러에서 모듈식 재사용 가능한 프레임워크로 진화했습니다:

| 구성 요소 | 목적 | 주요 예시 |
|-----------|---------|-------------|
| **LLVM IR** | 범용 최적화 타겟 | Clang, Rust, Swift, Julia에서 사용 |
| **MLIR** | 다중 레벨 IR 프레임워크 | TensorFlow, 하드웨어 컴파일러 |
| **패스** | 모듈식 최적화 | 상수 접기, DCE, 인라이닝 |
| **DSL** | 도메인 특화 표현력 | SQL, HTML, 셰이더 언어 |
| **ANTLR** | 파서 생성 | 언어 도구 |
| **Tree-sitter** | 증분 파싱 | 에디터 통합 |
| **LSP** | 에디터-언어 브리지 | VS Code, vim, emacs |
| **PGO** | 런타임 정보 기반 최적화 | 10-20% 속도 향상 |
| **LTO** | 모듈 간 최적화 | 전체 프로그램 분석 |
| **검증** | 컴파일러 정확성 | CompCert, Alive2, Csmith |

핵심 원칙:

1. **모듈성**: 프론트엔드, 최적화기, 백엔드를 분리하면 재사용과 빠른 언어 개발이 가능합니다.
2. **IR 설계의 중요성**: 잘 설계된 IR(LLVM IR 같은)은 전체 생태계의 기반이 됩니다.
3. **다중 추상화 수준**: MLIR은 서로 다른 도메인이 다른 IR 수준을 필요로 한다는 것을 인식합니다.
4. **도구링은 필수**: 현대 언어 개발은 컴파일러 자체만큼이나 도구링(LSP, Tree-sitter, 포매터)에 관한 것입니다.
5. **최적화는 끝이 없음**: PGO, LTO, 런타임 최적화는 정적 컴파일만으로는 찾을 수 없는 개선을 계속 발견합니다.
6. **정확성이 최우선**: 컴파일러가 더 복잡해질수록, 형식적 검증과 자동화된 테스팅이 점점 더 중요해집니다.

---

## 14. 연습 문제

### 연습 1: 직접 LLVM IR 작성

다음 함수들에 대한 LLVM IR(텍스트 형식)을 작성하세요:

(a) `int max(int a, int b)` -- 조건부 분기와 파이 노드를 사용하여 두 정수 중 더 큰 값을 반환합니다.

(b) `int sum_array(int* arr, int n)` -- 파이 노드가 있는 루프를 사용하여 배열의 모든 원소를 합산합니다.

(c) `int fibonacci(int n)` -- 루프를 사용하는 반복적 피보나치입니다.

각 `%name`이 정확히 한 번만 정의되는 SSA 형식을 따르는지 확인하여 IR이 문법적으로 올바른지 검증하세요.

### 연습 2: 최적화 패스

다음을 수행하는 **강도 감소(Strength Reduction)** 패스를 (Python으로, LLVM IR을 시뮬레이션하여) 구현하세요:
(a) `x * 2`를 `x + x`로 대체합니다.
(b) `x * 2의_거듭제곱`을 `x << log2(2의_거듭제곱)`로 대체합니다.
(c) `x / 2의_거듭제곱`을 `x >> log2(2의_거듭제곱)`로 대체합니다 (부호 없는 경우만).

이러한 패턴을 사용하는 함수에서 테스트하고 출력이 올바른지 검증하세요.

### 연습 3: DSL 설계 및 구현

다음 도메인 중 하나에 대한 DSL을 설계하고 구현하세요:

(a) **상태 기계**: 상태, 전이, 액션이 있는 유한 상태 기계를 정의하는 DSL.
(b) **빌드 시스템**: 빌드 규칙과 의존성을 정의하는 단순화된 Makefile 같은 DSL.
(c) **데이터 검증**: 구조화된 데이터(JSON Schema 같지만 더 단순한)에 대한 검증 규칙을 정의하는 DSL.

구현에는 다음을 포함해야 합니다: 파서(Python의 `re`나 단순 재귀 하강 파서 사용 가능), AST, 실행 가능한 Python 코드를 생성하는 코드 생성기.

### 연습 4: 증분 컴파일

9절의 증분 컴파일러를 다음과 같이 확장하세요:
(a) 파일 수준이 아닌 심볼 수준에서 의존성을 추적합니다 -- `a.c`의 함수 `foo`가 변경되어도 `a.c`의 함수 `bar`는 변경되지 않으면, `foo`에 의존하는 파일만 재컴파일합니다.
(b) 순환 의존성을 처리합니다 (감지하고 보고).
(c) 의존성 그래프를 디스크에 저장하여 여러 실행에 걸쳐 지속됩니다.

### 연습 5: PGO 시뮬레이터

다음을 수행하는 PGO 시뮬레이터를 만드세요:
(a) 간단한 프로그램(기본 블록과 분기 확률이 있는 CFG로 표현됨)을 입력으로 받습니다.
(b) 주어진 입력으로 실행을 시뮬레이션하여 분기 빈도를 수집합니다.
(c) 프로파일을 사용하여 기본 블록을 재정렬합니다 (핫 경로 먼저, 콜드 경로는 끝으로).
(d) 명령어 캐시 적중률의 예상 개선을 계산합니다.

### 연습 6: 번역 검증기

두 간단한 식이 동등한지 확인하는 간단한 번역 검증기를 구현하세요:
(a) 정수 산술을 지원합니다: `+`, `-`, `*`, `/`, `<<`, `>>`.
(b) 상수 접기 검증을 지원합니다 (예: `3 + 4`는 `7`과 같음).
(c) 대수적 항등식을 지원합니다 (예: `x + 0 = x`, `x * 1 = x`).
(d) 구체적인 테스트 값으로 기호 실행을 사용하여 반례를 찾습니다.
(e) 변환이 유효한지 보고하거나 반례를 제공합니다.

---

## 15. 참고 자료

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

[이전: 15. 인터프리터와 가상 머신](./15_Interpreters_and_Virtual_Machines.md) | [개요](./00_Overview.md)
