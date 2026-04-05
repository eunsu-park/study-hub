# LLVM 백엔드(LLVM Backend)

**이전**: [20. LLVM IR 입문](./20_LLVM_IR_Introduction.md) | **다음**: [22. JIT 컴파일](./22_JIT_Compilation.md)

---

LLVM 백엔드(backend)는 LLVM IR을 특정 타겟 아키텍처의 기계 코드(machine code)로 변환합니다. 이 과정은 명령어 선택(IR을 타겟 명령어로 변환), 레지스터 할당, 명령어 스케줄링, 최종 코드 방출 등 여러 단계를 포함합니다. 백엔드를 이해하는 것은 커스텀 컴파일러 패스 작성, 새로운 타겟 아키텍처 추가, 기계 코드 수준의 성능 문제 진단에 필수적입니다.

이 레슨에서는 LLVM 패스 파이프라인, 커스텀 최적화 패스 작성법, SelectionDAG 프레임워크, 명령어 선택(instruction selection) 알고리즘, 백엔드 코드 생성을 다룹니다.

**난이도**: ⭐⭐⭐⭐⭐

**선수 지식**: [20. LLVM IR 입문](./20_LLVM_IR_Introduction.md), [11. 코드 생성](./11_Code_Generation.md), [19. 레지스터 할당](./19_Register_Allocation.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. IR에서 기계 코드까지의 LLVM 패스 파이프라인(pass pipeline)을 설명한다
2. 분석 패스(analysis pass), 변환 패스(transformation pass), 기계 패스(machine pass)를 구분한다
3. C++로 커스텀 LLVM IR 최적화 패스를 작성한다
4. SelectionDAG가 IR을 타겟 명령어로 변환하는 방법을 설명한다
5. 패턴 매칭(pattern matching)과 타일링(tiling)을 통한 명령어 선택을 이해한다
6. 백엔드 파이프라인을 추적한다: SelectionDAG -> MachineInstr -> MCInst -> 오브젝트 코드

---

## 목차

1. [LLVM 패스 파이프라인](#1-llvm-패스-파이프라인)
2. [커스텀 패스 작성](#2-커스텀-패스-작성)
3. [새로운 패스 관리자](#3-새로운-패스-관리자)
4. [SelectionDAG 개요](#4-selectiondag-개요)
5. [명령어 선택](#5-명령어-선택)
6. [LLVM의 레지스터 할당](#6-llvm의-레지스터-할당)
7. [명령어 스케줄링](#7-명령어-스케줄링)
8. [기계 코드 방출](#8-기계-코드-방출)
9. [TableGen과 타겟 설명](#9-tablegen과-타겟-설명)
10. [요약](#10-요약)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. LLVM 패스 파이프라인

### 1.1 개요

LLVM은 일련의 **패스(pass)**를 통해 코드를 컴파일하며, 각 패스는 특정 분석이나 변환을 수행합니다:

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

### 1.2 패스 타입

| 패스 타입 | 동작 범위 | 예제 |
|-----------|----------|------|
| **ModulePass** | 전체 모듈 | 인라이닝, LTO |
| **FunctionPass** | 단일 함수 | Mem2Reg, ADCE |
| **LoopPass** | 단일 루프 | LICM, Loop Unroll |
| **BasicBlockPass** | 단일 블록 | Instruction combining |
| **MachineFunctionPass** | 기계 함수 | 레지스터 할당 |

### 1.3 파이프라인 보기

```bash
# Show all passes that run at -O2
clang -O2 -mllvm -debug-pass=Structure example.c -c 2>&1 | head -50

# With the new pass manager
clang -O2 -mllvm -print-pipeline-passes example.c -c

# Dump IR after each pass
clang -O2 -mllvm -print-after-all example.c -c
```

---

## 2. 커스텀 패스 작성(Writing Custom Passes)

### 2.1 간단한 함수 패스(레거시)

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

### 2.2 빌드 및 실행

```bash
# Build as a shared library
clang++ -shared -fPIC -o CountInst.so CountInstructions.cpp \
    $(llvm-config --cxxflags --ldflags --libs core)

# Run the pass
opt -load ./CountInst.so -count-inst < input.bc > /dev/null
```

### 2.3 변환 패스(Transformation Pass)

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

## 3. 새로운 패스 관리자(The New Pass Manager)

### 3.1 개요

LLVM의 새로운 패스 관리자(NPM, New Pass Manager)는 레거시 패스 관리자를 대체하며, 분석 캐싱(analysis caching)과 패스 순서 지정을 더 잘 지원합니다:

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

### 3.2 새로운 PM으로 실행

```bash
# Run with new pass manager
opt --load-pass-plugin=./CountInst.so --passes="count-inst" input.ll -S

# Combine with built-in passes
opt --load-pass-plugin=./CountInst.so \
    --passes="default<O2>,count-inst" input.ll -S
```

---

## 4. SelectionDAG 개요

### 4.1 IR에서 기계 명령어로

SelectionDAG는 LLVM의 주요 명령어 선택 프레임워크입니다. 기본 블록(basic block)을 방향성 비순환 그래프(DAG, Directed Acyclic Graph)로 표현합니다:

```
LLVM IR:                     SelectionDAG:
  %a = load i32, ptr %p         [load]
  %b = add i32 %a, 1              |
  store i32 %b, ptr %q        [add, 1]
                                   |
                               [store]
```

### 4.2 SelectionDAG 단계

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

### 4.3 DAG 보기

```bash
# Dump SelectionDAG before and after selection
llc -view-dag-combine1-dags example.ll    # After first combine
llc -view-isel-dags example.ll            # Before selection
llc -view-sched-dags example.ll           # After selection
```

---

## 5. 명령어 선택(Instruction Selection)

### 5.1 패턴 매칭

LLVM은 TableGen 패턴을 사용하여 DAG 노드를 타겟 명령어에 매칭합니다:

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

### 5.2 합법화(Legalization)

타겟이 모든 타입과 연산을 지원하지 않을 수 있습니다. 합법화(legalization)는 이를 확장(expand), 승격(promote), 또는 커스텀 하강(custom-lower)합니다:

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

### 5.3 GlobalISel (대안 파이프라인)

GlobalISel은 기본 블록 단위가 아닌 함수 수준에서 동작하는 LLVM의 더 새로운 명령어 선택 프레임워크입니다:

```
LLVM IR --> IRTranslator --> Legalizer --> RegBankSelect --> InstructionSelect
                  |              |              |                    |
            Generic MIR    Legal MIR    RegBank MIR         Target MIR
```

SelectionDAG 대비 장점:
- 더 빠른 컴파일(DAG 구성 불필요)
- 더 나은 크로스 블록 최적화 기회
- 새로운 타겟에 대한 확장이 더 단순

---

## 6. LLVM의 레지스터 할당

### 6.1 LLVM의 레지스터 할당기들

LLVM은 여러 레지스터 할당 전략을 제공합니다:

```bash
# Select register allocator
llc -regalloc=greedy example.ll    # Default: greedy (Briggs-like)
llc -regalloc=basic example.ll     # Simple baseline
llc -regalloc=fast example.ll      # Linear scan (for -O0)
llc -regalloc=pbqp example.ll      # PBQP-based (experimental)
```

### 6.2 탐욕적 할당기(Greedy Allocator)

기본 **탐욕적(greedy)** 할당기는 우선순위 기반, 영역 분할(region-splitting) 할당기입니다:

1. 스필 가중치(spill weight, 우선순위)로 생존 범위를 정렬
2. 각 생존 범위에 대해 레지스터 할당 시도
3. 사용 가능한 레지스터가 없으면 우선순위가 낮은 범위의 축출(eviction) 시도
4. 축출이 실패하면 생존 범위를 분할하고 재시도
5. 최후의 수단으로 스택에 스필

---

## 7. 명령어 스케줄링(Instruction Scheduling)

### 7.1 목적

명령어 스케줄링은 명령어를 재배치하여:
- 지연 시간(latency)을 숨김(예: 로드를 기다리는 동안 독립적인 명령어를 발행)
- 레지스터 압력을 줄임
- 명령어 수준 병렬성(ILP, Instruction-Level Parallelism)을 활용

### 7.2 리스트 스케줄링(List Scheduling)

LLVM은 다양한 휴리스틱을 사용한 리스트 스케줄링을 사용합니다:

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

## 8. 기계 코드 방출(Machine Code Emission)

### 8.1 MC 레이어

**MC(Machine Code)** 레이어는 LLVM의 저수준 코드 표현입니다:

```
MachineInstr (high-level machine instruction)
    ↓
MCInst (abstract machine instruction)
    ↓
MCCodeEmitter (binary encoding)
    ↓
Object file (.o) or Assembly (.s)
```

### 8.2 출력 형식

```bash
# Assembly output
llc example.ll -o example.s

# Object file
llc -filetype=obj example.ll -o example.o

# Show encoding details
llc -show-mc-encoding example.ll
```

---

## 9. TableGen과 타겟 설명

### 9.1 TableGen이란?

TableGen은 타겟 아키텍처를 기술하는 데 사용되는 도메인 특화 언어(DSL, Domain-Specific Language)입니다:

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

### 9.2 타겟 설명 파일

타겟 백엔드는 여러 `.td` 파일로 구성됩니다:

| 파일 | 기술 내용 |
|------|----------|
| `*RegisterInfo.td` | 레지스터와 레지스터 클래스 |
| `*InstrInfo.td` | 명령어 정의 |
| `*InstrFormats.td` | 명령어 인코딩 형식 |
| `*CallingConv.td` | 호출 규약 |
| `*Schedule.td` | 명령어 지연 시간 |

---

## 10. 요약

- LLVM은 분석 및 최적화 패스를 통해 IR을 기계 코드로 변환하는 **패스 파이프라인**을 사용합니다
- **새로운 패스 관리자**는 더 나은 분석 캐싱과 합성 가능성(composability)을 제공합니다
- **SelectionDAG**는 패턴 매칭을 통해 IR 기본 블록을 타겟 기계 명령어로 변환합니다
- **GlobalISel**은 SelectionDAG의 더 새롭고 빠른 대안입니다
- LLVM의 **레지스터 할당**은 기본적으로 생존 범위 분할이 있는 탐욕적 할당기를 사용합니다
- **명령어 스케줄링**은 지연 시간을 숨기고 ILP를 활용합니다
- **TableGen**은 타겟 아키텍처를 선언적으로 기술합니다
- **MC 레이어**는 어셈블리 또는 오브젝트 파일로의 최종 코드 방출을 처리합니다

---

## 11. 연습 문제

1. **패스 작성**: 각 함수의 파이 노드(phi node) 수를 세는 LLVM 패스를 구현하세요.

2. **변환 패스**: 부호 없는 정수(unsigned integer)에 대해 `x / 2`를 `x >> 1`로 대체하는 패스를 작성하세요.

3. **SelectionDAG 검사**: 작은 C 함수를 컴파일하고 각 단계에서 SelectionDAG를 덤프하세요. 패턴 매치를 식별하세요.

4. **할당기 비교**: `-regalloc=greedy`와 `-regalloc=fast`로 함수를 컴파일하고 스필 수를 비교하세요.

5. **TableGen 읽기**: x86의 `MOV` 명령어에 대한 TableGen 정의를 찾고, IR에서 기계 코드로의 매핑을 추적하세요.

---

## 12. 참고 자료

1. LLVM Writing an LLVM Pass: https://llvm.org/docs/WritingAnLLVMPass.html
2. LLVM New Pass Manager: https://llvm.org/docs/WritingAnLLVMNewPMPass.html
3. LLVM Code Generator: https://llvm.org/docs/CodeGenerator.html
4. LLVM SelectionDAG: https://llvm.org/docs/SelectionDAG.html
5. LLVM TableGen: https://llvm.org/docs/TableGen/

---

**이전**: [20. LLVM IR 입문](./20_LLVM_IR_Introduction.md) | **다음**: [22. JIT 컴파일](./22_JIT_Compilation.md)
