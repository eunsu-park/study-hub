# 디버그 정보(Debug Information)

**이전**: [25. 링킹과 로딩](./25_Linking_and_Loading.md) | **다음**: [27. 현대 파서 생성기](./27_Modern_Parser_Generators.md)

---

프로그램이 충돌하거나 예상치 못한 동작을 할 때, 개발자는 디버거(debugger)에 의존하여 변수를 검사하고, 중단점(breakpoint)을 설정하며, 소스 코드를 단계별로 실행합니다. 그러나 기계가 실행하는 것은 소스 코드가 아닌 바이너리 명령어입니다. 디버그 정보(debug information)는 기계 코드를 소스 위치, 변수 이름, 타입, 스코프 규칙으로 다시 매핑하여 이 간극을 메웁니다. 컴파일러는 이 메타데이터를 신중하게 생성해야 하며, 그 형식을 이해하는 것은 디버깅 도구, 프로파일러, 충돌 분석 시스템을 구축하는 데 필수적입니다.

이 레슨에서는 DWARF 디버그 형식, 웹 언어용 소스 맵(source map), 컴파일러가 디버그 정보를 생성하는 방법, 그리고 디버거가 이를 사용하는 방법을 다룹니다.

**난이도**: ⭐⭐⭐

**선수 지식**: [11. 코드 생성](./11_Code_Generation.md), [25. 링킹과 로딩](./25_Linking_and_Loading.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 디버그 정보가 필요한 이유와 소스 코드와의 관계를 설명한다
2. DWARF 디버그 정보 형식과 핵심 구조를 기술한다
3. 주소를 소스 위치에 매핑하는 줄 번호 테이블을 이해한다
4. 변수 위치 설명(variable location description)이 어떻게 동작하는지 설명한다
5. LLVM IR에서 디버그 메타데이터를 생성한다
6. JavaScript 및 웹 언어용 소스 맵을 이해한다
7. 디버거가 중단점과 단계 실행에 디버그 정보를 사용하는 방법을 기술한다

---

## 목차

1. [디버그 정보의 필요성](#1-디버그-정보의-필요성)
2. [DWARF 형식 개요](#2-dwarf-형식-개요)
3. [컴파일 단위와 DIE](#3-컴파일-단위와-die)
4. [줄 번호 테이블](#4-줄-번호-테이블)
5. [변수 위치](#5-변수-위치)
6. [LLVM의 디버그 정보](#6-llvm의-디버그-정보)
7. [소스 맵](#7-소스-맵)
8. [디버거 통합](#8-디버거-통합)
9. [디버그 정보와 최적화](#9-디버그-정보와-최적화)
10. [요약](#10-요약)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. 디버그 정보의 필요성

### 1.1 문제

기계 코드에는 변수 이름, 소스 줄, 타입이라는 개념이 없습니다:

```c
// Source code (what the developer sees):
int factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

// Machine code (what executes):
// 0x4005a0: push rbp
// 0x4005a1: mov rbp, rsp
// 0x4005a4: mov [rbp-4], edi
// 0x4005a7: cmp dword [rbp-4], 1
// ...
```

디버그 정보는 다음 질문에 답합니다:
- 주소 `0x4005a7`에 해당하는 소스 파일과 줄은 무엇인가?
- 이 지점에서 변수 `n`의 값은 무엇인가?
- `n`의 타입은 무엇인가?

### 1.2 디버그 정보 생성

```bash
# Compile with debug info
gcc -g program.c -o program           # Full debug info (DWARF)
gcc -g -O2 program.c -o program       # Debug info + optimization
gcc -g1 program.c -o program          # Minimal (line tables only)
gcc -g3 program.c -o program          # Maximum (includes macros)

# Strip debug info
strip --strip-debug program
# or separate it:
objcopy --only-keep-debug program program.debug
objcopy --add-gnu-debuglink=program.debug program
```

### 1.3 디버그 정보 형식

| 형식 | 플랫폼 | 사용처 |
|------|--------|--------|
| **DWARF** | Linux, macOS, BSD | GCC, Clang, GDB, LLDB |
| **PDB** | Windows | MSVC, Visual Studio |
| **CodeView** | Windows | MSVC (레거시) |
| **Stabs** | 레거시 Unix | 구버전 GCC (더 이상 사용되지 않음) |
| **Source Maps** | 웹 | JavaScript, TypeScript |

---

## 2. DWARF 형식 개요

### 2.1 DWARF란?

DWARF(Debugging With Attributed Record Formats)는 ELF 바이너리의 표준 디버그 정보 형식입니다. 오브젝트 파일의 특수 섹션에 저장됩니다:

| 섹션 | 내용 |
|------|------|
| `.debug_info` | 타입, 변수, 함수 설명 (DIE) |
| `.debug_abbrev` | 약어 테이블 (DIE용 템플릿) |
| `.debug_line` | 줄 번호 테이블 (주소 → 소스 줄) |
| `.debug_str` | 디버그 정보용 문자열 테이블 |
| `.debug_loc` | 변수 위치 리스트 |
| `.debug_ranges` | 스코프의 주소 범위 |
| `.debug_frame` | 호출 프레임 정보 (언와인딩용) |

### 2.2 DWARF 검사

```bash
# View all debug info
readelf --debug-dump=info program

# View line number table
readelf --debug-dump=line program

# Using dwarfdump (more detailed)
dwarfdump program

# Using llvm-dwarfdump
llvm-dwarfdump program

# View only specific DIE types
llvm-dwarfdump --name=factorial program
```

---

## 3. 컴파일 단위와 DIE

### 3.1 디버그 정보 항목(Debug Information Entries, DIE)

DWARF는 디버그 정보를 **DIE**(Debug Information Entry)의 트리로 표현합니다. 각 DIE는 **태그**(엔티티 타입)와 **속성**(프로퍼티)을 가집니다:

```
DW_TAG_compile_unit (root)
  DW_AT_name: "main.c"
  DW_AT_language: DW_LANG_C11
  DW_AT_producer: "clang 17.0"
  │
  ├── DW_TAG_subprogram          (function)
  │     DW_AT_name: "factorial"
  │     DW_AT_type: <ref to int>
  │     DW_AT_low_pc: 0x4005a0
  │     DW_AT_high_pc: 0x4005d0
  │     │
  │     ├── DW_TAG_formal_parameter
  │     │     DW_AT_name: "n"
  │     │     DW_AT_type: <ref to int>
  │     │     DW_AT_location: DW_OP_fbreg -20
  │     │
  │     └── DW_TAG_lexical_block   (scope)
  │           DW_AT_low_pc: 0x4005b0
  │           DW_AT_high_pc: 0x4005c0
  │
  ├── DW_TAG_base_type            (type)
  │     DW_AT_name: "int"
  │     DW_AT_encoding: DW_ATE_signed
  │     DW_AT_byte_size: 4
  │
  └── DW_TAG_variable             (global variable)
        DW_AT_name: "global_count"
        DW_AT_type: <ref to int>
        DW_AT_location: DW_OP_addr 0x601000
```

### 3.2 주요 DIE 태그

| 태그 | 설명 |
|------|------|
| `DW_TAG_compile_unit` | 소스 파일 |
| `DW_TAG_subprogram` | 함수 |
| `DW_TAG_variable` | 변수 |
| `DW_TAG_formal_parameter` | 함수 매개변수 |
| `DW_TAG_base_type` | 기본 타입 (int, float 등) |
| `DW_TAG_pointer_type` | 포인터 타입 |
| `DW_TAG_structure_type` | 구조체/클래스 |
| `DW_TAG_array_type` | 배열 타입 |
| `DW_TAG_lexical_block` | 스코프 블록 |
| `DW_TAG_inlined_subroutine` | 인라인된 함수 본문 |

---

## 4. 줄 번호 테이블

### 4.1 목적

줄 번호 테이블(line number table)은 기계 코드 주소를 소스 파일 위치에 매핑합니다:

```
Address         File        Line    Column
0x4005a0        main.c      5       1
0x4005a4        main.c      6       5
0x4005a7        main.c      6       8
0x4005b0        main.c      7       5
```

### 4.2 상태 머신 인코딩

DWARF는 줄 테이블을 가상 상태 머신(virtual state machine)용 연산코드 시퀀스로 인코딩합니다:

```python
class LineNumberStateMachine:
    """DWARF line number state machine."""

    def __init__(self):
        self.address = 0
        self.file = 1
        self.line = 1
        self.column = 0
        self.is_stmt = True
        self.basic_block = False
        self.end_sequence = False

    def execute_opcode(self, opcode, operand=None):
        if opcode == DW_LNS_advance_pc:
            self.address += operand
        elif opcode == DW_LNS_advance_line:
            self.line += operand
        elif opcode == DW_LNS_set_file:
            self.file = operand
        elif opcode == DW_LNS_set_column:
            self.column = operand
        elif opcode == DW_LNS_copy:
            self.emit_row()
        elif opcode == DW_LNE_end_sequence:
            self.end_sequence = True
            self.emit_row()
            self.reset()

    def emit_row(self):
        """Record current state as a line table entry."""
        print(f"0x{self.address:08x}  file={self.file}  "
              f"line={self.line}  col={self.column}")
```

### 4.3 특수 연산코드

컴팩트한 인코딩을 위해, DWARF는 주소와 줄을 하나의 바이트로 동시에 전진시키는 "특수 연산코드(special opcode)"를 사용합니다:

```python
def decode_special_opcode(opcode, opcode_base, line_base, line_range, min_insn_length):
    """
    Special opcodes encode address+line advance in one byte.
    """
    adjusted = opcode - opcode_base
    line_advance = line_base + (adjusted % line_range)
    addr_advance = (adjusted // line_range) * min_insn_length
    return addr_advance, line_advance
```

---

## 5. 변수 위치

### 5.1 위치 문제

최적화(optimization)로 인해 변수의 저장 위치가 실행 중에 변경됩니다:

```c
int x = compute();    // x in register rax
use(x);
// ... more code ...  // x spilled to stack [rbp-8]
use(x);
// ... more code ...  // x optimized away (dead)
```

### 5.2 DWARF 위치 표현식

DWARF는 위치를 설명하기 위해 스택 기반 표현식 언어(stack-based expression language)를 사용합니다:

```
DW_OP_reg0          -- value is in register 0 (rax on x86-64)
DW_OP_breg6 -20     -- value is at [rbp - 20] (rbp is reg 6)
DW_OP_fbreg -8      -- value is at [frame_base - 8]
DW_OP_addr 0x601000 -- value is at absolute address
DW_OP_lit5          -- value is the constant 5

# Complex expression: *(rbp - 16) + 4
DW_OP_breg6 -16     -- push [rbp - 16]
DW_OP_deref          -- dereference
DW_OP_lit4           -- push 4
DW_OP_plus           -- add
```

### 5.3 위치 리스트

변수가 여러 위치 사이를 이동할 때, DWARF는 **위치 리스트(location list)**를 사용합니다:

```
Variable "x":
  [0x4005a0, 0x4005b0): DW_OP_reg0        -- x in rax
  [0x4005b0, 0x4005d0): DW_OP_fbreg -8    -- x on stack
  [0x4005d0, 0x4005e0): <empty>            -- x optimized out
```

---

## 6. LLVM의 디버그 정보

### 6.1 LLVM 디버그 메타데이터

LLVM IR은 메타데이터를 사용하여 디버그 정보를 표현합니다:

```llvm
define i32 @factorial(i32 %n) !dbg !5 {
entry:
  call void @llvm.dbg.value(metadata i32 %n, metadata !10, metadata !DIExpression()), !dbg !11
  %cmp = icmp sle i32 %n, 1, !dbg !12
  ; ...
}

!5 = distinct !DISubprogram(name: "factorial", file: !6, line: 3,
       type: !7, scopeLine: 3, unit: !0)
!6 = !DIFile(filename: "main.c", directory: "/home/user")
!10 = !DILocalVariable(name: "n", arg: 1, scope: !5, file: !6,
        line: 3, type: !9)
!11 = !DILocation(line: 3, column: 20, scope: !5)
!12 = !DILocation(line: 4, column: 8, scope: !5)
```

### 6.2 llvmlite로 디버그 정보 생성하기

```python
from llvmlite import ir

def build_with_debug_info():
    module = ir.Module(name="debug_example")

    # Create debug info
    di_file = module.add_debug_info("DIFile", {
        "filename": "example.c",
        "directory": "/home/user",
    })

    di_compile_unit = module.add_debug_info("DICompileUnit", {
        "language": ir.DIToken("DW_LANG_C"),
        "file": di_file,
        "producer": "my_compiler 1.0",
        "isOptimized": False,
    }, is_distinct=True)

    di_int = module.add_debug_info("DIBasicType", {
        "name": "int",
        "size": 32,
        "encoding": ir.DIToken("DW_ATE_signed"),
    })

    di_func_type = module.add_debug_info("DISubroutineType", {
        "types": module.add_metadata([di_int, di_int]),
    })

    # Create function with debug info
    i32 = ir.IntType(32)
    func = ir.Function(module, ir.FunctionType(i32, [i32]), name="square")

    di_func = module.add_debug_info("DISubprogram", {
        "name": "square",
        "file": di_file,
        "line": 1,
        "type": di_func_type,
        "scopeLine": 1,
        "unit": di_compile_unit,
    }, is_distinct=True)
    func.set_metadata("dbg", di_func)

    block = func.append_basic_block("entry")
    builder = ir.IRBuilder(block)
    builder.debug_metadata = module.add_debug_info("DILocation", {
        "line": 2, "column": 5, "scope": di_func,
    })

    result = builder.mul(func.args[0], func.args[0])
    builder.ret(result)

    return module
```

---

## 7. 소스 맵

### 7.1 소스 맵이란?

소스 맵(Source Map)은 DWARF와 동일한 목적을 제공하지만 웹 언어용입니다. 축소/트랜스파일된 JavaScript를 원본 TypeScript, JSX 또는 기타 소스로 다시 매핑합니다:

```
Original (TypeScript):        Compiled (JavaScript):
function greet(name) {        function a(b){return"Hello "+b}
  return "Hello " + name;
}
```

### 7.2 소스 맵 형식

```json
{
  "version": 3,
  "file": "output.js",
  "sourceRoot": "",
  "sources": ["input.ts"],
  "names": ["greet", "name"],
  "mappings": "AAAA,SAASA,MAAKC,GAAQ,MAAO,SAAWA"
}
```

### 7.3 VLQ 인코딩

소스 맵은 컴팩트한 매핑을 위해 가변 길이 수량(Variable-Length Quantity, VLQ) 인코딩을 사용합니다:

```python
def encode_vlq(value):
    """Encode an integer as a VLQ base64 string."""
    VLQ_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

    # Handle sign
    if value < 0:
        vlq = ((-value) << 1) | 1
    else:
        vlq = value << 1

    result = ""
    while True:
        digit = vlq & 0x1F  # 5 bits
        vlq >>= 5
        if vlq > 0:
            digit |= 0x20   # continuation bit
        result += VLQ_CHARS[digit]
        if vlq == 0:
            break

    return result
```

---

## 8. 디버거 통합

### 8.1 중단점의 동작 원리

```
1. User sets breakpoint at line 10
2. Debugger looks up line 10 in line table → address 0x4005b0
3. Debugger replaces instruction at 0x4005b0 with INT 3 (0xCC)
4. When execution hits 0xCC, CPU raises SIGTRAP
5. Debugger catches SIGTRAP
6. Debugger restores original instruction
7. User inspects variables using DWARF location info
8. Debugger single-steps past the breakpoint, reinserts INT 3
```

### 8.2 단일 단계 실행

줄 수준 단계 실행(line-level stepping)은 줄 테이블을 사용하여 각 소스 줄이 시작되는 위치를 결정합니다:

```python
def step_over_line(debugger, current_addr):
    """Step to the next source line."""
    current_line = debugger.line_table.addr_to_line(current_addr)

    while True:
        debugger.single_step_instruction()
        new_addr = debugger.get_pc()
        new_line = debugger.line_table.addr_to_line(new_addr)
        if new_line != current_line:
            break
```

### 8.3 스택 언와인딩

`.debug_frame` (또는 `.eh_frame`) 섹션은 스택을 언와인딩(unwinding)하기 위한 호출 프레임 정보(call frame information)를 포함합니다:

```
CIE (Common Information Entry):
  Code alignment: 1
  Data alignment: -8
  Return register: 16 (rip)

FDE (Frame Description Entry) for factorial:
  Address range: [0x4005a0, 0x4005d0)
  Instructions:
    DW_CFA_def_cfa: r7 (rsp) offset 8
    DW_CFA_offset: r16 (rip) at CFA-8
    advance 1 to 0x4005a1
    DW_CFA_def_cfa_offset: 16
    DW_CFA_offset: r6 (rbp) at CFA-16
```

---

## 9. 디버그 정보와 최적화

### 9.1 과제

최적화는 코드를 재배치하고, 제거하고, 변환하여 정확한 디버그 정보를 유지하기 어렵게 만듭니다:

| 최적화 | 디버그 영향 |
|--------|------------|
| 죽은 코드 제거(Dead code elimination) | 변수가 "최적화 제거됨" |
| 레지스터 할당(Register allocation) | 변수 위치가 변경됨 |
| 인라이닝(Inlining) | 함수가 "사라지고" 호출자에 나타남 |
| 루프 언롤링(Loop unrolling) | 하나의 소스 줄이 여러 복사본에 매핑 |
| 명령어 재정렬(Instruction reordering) | 비선형 단계 실행 |

### 9.2 전략

- **변환 과정에서 메타데이터 유지**: LLVM은 명령어에 `!dbg` 메타데이터를 전파합니다
- **위치 리스트**: 레지스터와 스택 슬롯 간 변수 이동을 추적합니다
- **인라인된 서브루틴 DIE**: DWARF에 인라이닝 결정을 기록합니다
- **is_stmt 플래그**: 좋은 중단점 후보인 주소를 표시합니다

### 9.3 디버그 수준 트레이드오프

```bash
-g0  # No debug info (smallest binary, no debugging)
-g1  # Line tables only (backtraces work, no variable inspection)
-g   # Full debug info (default: types, variables, scopes)
-g3  # Full + macros (largest, can expand macros in debugger)
```

---

## 10. 요약

- **디버그 정보**는 기계 코드를 디버깅을 위해 소스 코드로 다시 매핑합니다
- **DWARF**는 `.debug_*` ELF 섹션에 저장되는 표준 형식입니다
- **DIE**는 타입, 함수, 변수, 스코프를 기술하는 트리를 형성합니다
- **줄 번호 테이블**은 코드 주소를 소스 파일/줄/열에 매핑합니다
- **위치 표현식**은 변수가 저장된 위치(레지스터, 스택, 메모리)를 기술합니다
- **LLVM**은 IR 메타데이터(`!dbg`, `!DILocation` 등)를 통해 디버그 정보를 생성합니다
- **소스 맵**은 JavaScript/TypeScript에 동등한 기능을 제공합니다
- 최적화는 디버그 정보를 복잡하게 만들며, 위치 리스트와 인라인된 서브루틴 DIE가 이를 돕습니다

---

## 11. 연습 문제

1. **DWARF 검사**: C 프로그램을 `-g`로 컴파일하고 `readelf --debug-dump`와 `llvm-dwarfdump`를 사용하여 DWARF 정보를 탐색하세요.

2. **줄 테이블 워크스루**: 작은 함수에 대해 줄 번호 상태 머신을 추적하고 주소-줄 매핑을 검증하세요.

3. **변수 위치**: `-g -O2`로 컴파일하고 `llvm-dwarfdump --name=<variable>`을 사용하여 함수 전체에서 변수 위치가 어떻게 변화하는지 검사하세요.

4. **소스 맵**: 소스 맵을 생성하는 간단한 트랜스파일러(예: CoffeeScript 유사 → JS)를 작성하세요.

5. **디버그 정보 생성**: 간단한 컴파일러를 확장하여 함수와 변수에 대한 LLVM 디버그 메타데이터를 방출하세요.

---

## 12. 참고 자료

1. DWARF Debugging Standard Committee. *DWARF Debugging Information Format Version 5*. https://dwarfstd.org/
2. Eager, M. J. (2012). "Introduction to the DWARF Debugging Format." https://dwarfstd.org/doc/Debugging%20using%20DWARF-2012.pdf
3. Source Map Specification v3: https://sourcemaps.info/spec.html
4. LLVM Source Level Debugging: https://llvm.org/docs/SourceLevelDebugging.html
5. Eli Bendersky. "How debuggers work." https://eli.thegreenplace.net/tag/debuggers

---

**이전**: [25. 링킹과 로딩](./25_Linking_and_Loading.md) | **다음**: [27. 현대 파서 생성기](./27_Modern_Parser_Generators.md)
