# 링킹과 로딩(Linking and Loading)

**이전**: [24. 클로저 변환](./24_Closure_Conversion.md) | **다음**: [26. 디버그 정보](./26_Debug_Information.md)

---

컴파일이 오브젝트 파일(object file)을 생성한 후, 링커(linker)가 이를 결합하여 실행 파일이나 공유 라이브러리를 만들고, 로더(loader)가 결과를 메모리에 매핑하여 실행합니다. 링킹과 로딩은 컴파일된 오브젝트 코드와 실행 중인 프로그램 사이의 간극을 메웁니다. 이 과정을 이해하는 것은 빌드 오류 진단, 시작 시간 최적화, 공유 라이브러리 관리, 복잡한 소프트웨어 시스템 작업에 필수적입니다.

이 레슨은 오브젝트 파일 형식(ELF), 심볼 해석(symbol resolution), 정적 및 동적 링킹, 재배치(relocation), 위치 독립 코드(PIC), 런타임 로더를 다룹니다.

**난이도**: ⭐⭐⭐⭐

**선수 지식**: [11. 코드 생성](./11_Code_Generation.md), [10. 런타임 환경](./10_Runtime_Environments.md)

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ELF 오브젝트 파일과 실행 파일의 구조를 기술한다
2. 심볼 해석과 링커의 참조 해결 방법을 설명한다
3. 정적 링킹(static linking)과 동적 링킹(dynamic linking)을 구분한다
4. 재배치(relocation)와 위치 독립 코드(PIC, Position-Independent Code)를 이해한다
5. 런타임 동적 링커/로더(ld.so)의 동작 방식을 설명한다
6. 일반적인 링킹 오류와 라이브러리 문제를 진단한다

---

## 목차

1. [오브젝트 파일](#1-오브젝트-파일)
2. [ELF 형식](#2-elf-형식)
3. [심볼과 심볼 테이블](#3-심볼과-심볼-테이블)
4. [정적 링킹](#4-정적-링킹)
5. [재배치](#5-재배치)
6. [동적 링킹](#6-동적-링킹)
7. [위치 독립 코드](#7-위치-독립-코드)
8. [런타임 로더](#8-런타임-로더)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 오브젝트 파일

### 1.1 오브젝트 파일의 종류

| 종류 | 설명 | 확장자 |
|------|------|--------|
| **재배치 가능(Relocatable)** | 컴파일러 출력; 아직 링크되지 않음 | `.o` |
| **실행 파일(Executable)** | 완전히 링크됨; 실행 가능 | (없음 또는 `.exe`) |
| **공유 오브젝트(Shared object)** | 동적 라이브러리 | `.so` (Linux), `.dylib` (macOS) |
| **코어 덤프(Core dump)** | 프로세스 메모리 스냅샷 | `core` |

### 1.2 소스에서 실행 파일까지

```
source.c  -->  [Compiler]  -->  source.o  (relocatable object)
                                    |
lib.c     -->  [Compiler]  -->  lib.o     (relocatable object)
                                    |
                               [Linker]
                                    |
                                    v
                              a.out (executable)
```

### 1.3 섹션(Sections)

오브젝트 파일은 **섹션(section)**으로 나뉩니다:

```
┌──────────────────┐
│   ELF Header     │  Magic number, type, architecture
├──────────────────┤
│ .text            │  Executable machine code
├──────────────────┤
│ .data            │  Initialized global variables
├──────────────────┤
│ .bss             │  Uninitialized global variables (zero-filled)
├──────────────────┤
│ .rodata          │  Read-only data (string literals, constants)
├──────────────────┤
│ .symtab          │  Symbol table
├──────────────────┤
│ .strtab          │  String table (symbol names)
├──────────────────┤
│ .rel.text        │  Relocation entries for .text
├──────────────────┤
│ .rel.data        │  Relocation entries for .data
├──────────────────┤
│ Section headers  │  Metadata about all sections
└──────────────────┘
```

---

## 2. ELF 형식

### 2.1 ELF 헤더

```c
// Simplified ELF header structure
typedef struct {
    unsigned char e_ident[16];  // Magic: 0x7f 'E' 'L' 'F'
    uint16_t e_type;            // ET_REL, ET_EXEC, ET_DYN
    uint16_t e_machine;         // EM_X86_64, EM_AARCH64, etc.
    uint32_t e_version;         // EV_CURRENT
    uint64_t e_entry;           // Entry point virtual address
    uint64_t e_phoff;           // Program header table offset
    uint64_t e_shoff;           // Section header table offset
    uint16_t e_phnum;           // Number of program headers
    uint16_t e_shnum;           // Number of section headers
} Elf64_Ehdr;
```

### 2.2 ELF 파일 검사

```bash
# View ELF header
readelf -h program

# List sections
readelf -S program.o

# View symbol table
readelf -s program.o
# or
nm program.o

# View relocations
readelf -r program.o

# Disassemble .text section
objdump -d program.o

# View program headers (segments)
readelf -l program

# View dynamic section
readelf -d program
```

### 2.3 섹션 vs. 세그먼트

```
Object files have SECTIONS (compile-time view):
  .text, .data, .bss, .rodata, ...

Executables have SEGMENTS (runtime view):
  LOAD segments group sections by permissions:

  Segment 1 (READ + EXECUTE):  .text, .rodata
  Segment 2 (READ + WRITE):    .data, .bss
  Segment 3 (READ):            .dynamic, .dynsym
```

---

## 3. 심볼과 심볼 테이블

### 3.1 심볼이란?

**심볼(symbol)**은 이름이 붙은 엔티티입니다: 함수, 전역 변수, 또는 레이블:

```c
// main.c
extern int global_var;       // undefined symbol (reference)
int local_var = 42;          // defined symbol (global)
static int static_var = 10;  // defined symbol (local -- not exported)

void my_function(void) {}    // defined symbol (global)
static void helper(void) {}  // defined symbol (local)

int main() {
    my_function();
    printf("hello\n");       // undefined symbol (reference)
}
```

### 3.2 심볼 테이블 항목

```c
typedef struct {
    uint32_t st_name;     // Index into string table
    uint8_t  st_info;     // Binding (LOCAL/GLOBAL/WEAK) + Type (FUNC/OBJECT)
    uint8_t  st_other;    // Visibility
    uint16_t st_shndx;    // Section index (or SHN_UNDEF for undefined)
    uint64_t st_value;    // Value (address or offset)
    uint64_t st_size;     // Size of the symbol
} Elf64_Sym;
```

### 3.3 심볼 바인딩(Symbol Binding)

| 바인딩 | 의미 |
|--------|------|
| `LOCAL` | 오브젝트 파일 내에서만 가시적 (예: `static`) |
| `GLOBAL` | 모든 오브젝트 파일에서 가시적 |
| `WEAK` | GLOBAL과 유사하지만 강한(GLOBAL) 심볼로 재정의 가능 |

### 3.4 심볼 해석 예제

```bash
$ nm main.o
                 U global_var      # Undefined
0000000000000000 D local_var       # Defined in .data
0000000000000004 d static_var      # Local (lowercase = local)
0000000000000000 T my_function     # Defined in .text
0000000000000010 t helper          # Local in .text
0000000000000020 T main            # Defined in .text
                 U printf          # Undefined
```

---

## 4. 정적 링킹

### 4.1 링커의 동작 방식

```
Input:  main.o, lib.o, libc.a
Output: executable

Steps:
1. Read all input object files
2. Resolve symbol references:
   - For each undefined symbol, find a definition in another object
3. Perform relocations:
   - Patch addresses now that final layout is known
4. Write the executable with all sections merged
```

### 4.2 심볼 해석 알고리즘

```python
def resolve_symbols(object_files):
    """
    Resolve symbols across multiple object files.
    """
    defined = {}     # symbol name -> (object, section, value)
    undefined = {}   # symbol name -> list of (object, relocation)

    for obj in object_files:
        for sym in obj.symbols:
            if sym.is_defined:
                if sym.name in defined and sym.binding == 'GLOBAL':
                    if defined[sym.name].binding == 'GLOBAL':
                        raise LinkError(f"Multiple definition of {sym.name}")
                    # Weak overridden by global
                defined[sym.name] = sym
            else:
                undefined.setdefault(sym.name, []).append((obj, sym))

    # Check for unresolved symbols
    for name, refs in undefined.items():
        if name not in defined:
            raise LinkError(f"Undefined reference to {name}")

    return defined
```

### 4.3 정적 라이브러리 (.a)

정적 라이브러리(static library)는 오브젝트 파일의 아카이브입니다:

```bash
# Create static library
ar rcs libmath.a add.o multiply.o divide.o

# Link with static library
gcc main.o -L. -lmath -o program

# The linker only pulls in .o files that resolve undefined symbols
```

### 4.4 아카이브 멤버 선택

```python
def link_with_archives(objects, archives):
    """
    Link objects with static library archives.
    Only include archive members that resolve undefined symbols.
    """
    undefined = collect_undefined(objects)

    changed = True
    while changed:
        changed = False
        for archive in archives:
            for member in archive.members:
                # Include member if it defines any undefined symbol
                member_defines = get_defined_symbols(member)
                if member_defines & undefined:
                    objects.append(member)
                    undefined = collect_undefined(objects)
                    changed = True
```

---

## 5. 재배치

### 5.1 재배치가 필요한 이유

컴파일러가 오브젝트 코드를 생성할 때 최종 주소를 알 수 없습니다:

```asm
; In main.o:
call printf          ; Address of printf unknown at compile time
mov eax, [global_var] ; Address of global_var unknown
```

컴파일러는 링커에게 어디를 패치할지 알려주는 **재배치 항목(relocation entry)**을 방출합니다:

```c
typedef struct {
    uint64_t r_offset;   // Where to apply the relocation
    uint64_t r_info;     // Symbol index + relocation type
    int64_t  r_addend;   // Constant addend
} Elf64_Rela;
```

### 5.2 일반적인 재배치 타입 (x86-64)

| 타입 | 의미 | 계산 |
|------|------|------|
| `R_X86_64_64` | 절대 64비트 주소 | S + A |
| `R_X86_64_PC32` | PC 상대 32비트 | S + A - P |
| `R_X86_64_PLT32` | PLT 항목 (함수 호출) | L + A - P |
| `R_X86_64_GOT64` | GOT 항목 | G + A |

여기서 S = 심볼 값, A = 가산값(addend), P = 재배치 위치, L = PLT 항목, G = GOT 오프셋입니다.

### 5.3 재배치 과정

```python
def perform_relocations(sections, symbols, relocations):
    """
    Apply relocations to patch addresses in code and data.
    """
    for reloc in relocations:
        sym = symbols[reloc.symbol_index]
        section = sections[reloc.section]
        offset = reloc.offset
        addend = reloc.addend

        if reloc.type == R_X86_64_PC32:
            # PC-relative: target - current_position
            value = sym.final_address + addend - (section.base + offset)
            section.write_i32(offset, value)

        elif reloc.type == R_X86_64_64:
            # Absolute 64-bit address
            value = sym.final_address + addend
            section.write_i64(offset, value)

        elif reloc.type == R_X86_64_PLT32:
            # PLT entry for function call
            plt_entry = get_plt_entry(sym)
            value = plt_entry + addend - (section.base + offset)
            section.write_i32(offset, value)
```

---

## 6. 동적 링킹

### 6.1 동기

정적 링킹은 모든 라이브러리 코드를 실행 파일에 복사합니다. 동적 링킹(dynamic linking)은 런타임에 라이브러리를 공유합니다:

| 측면 | 정적 링킹 | 동적 링킹 |
|------|----------|----------|
| 실행 파일 크기 | 큼 | 작음 |
| 메모리 공유 | 불가 | 프로세스 간 공유 |
| 업데이트 | 재컴파일 필요 | .so 파일 교체 |
| 시작 시간 | 빠름 | 느림 (심볼 해석) |
| 배포 | 자체 완결 | 라이브러리 설치 필요 |

### 6.2 공유 라이브러리

```bash
# Create shared library
gcc -shared -fPIC -o libmath.so add.c multiply.c

# Link with shared library
gcc main.c -L. -lmath -o program

# Run (library must be findable)
LD_LIBRARY_PATH=. ./program
```

### 6.3 GOT와 PLT

동적 링킹은 두 가지 핵심 데이터 구조를 사용합니다:

**GOT(Global Offset Table, 전역 오프셋 테이블)**: 외부 변수와 함수의 해석된 주소를 저장합니다.

**PLT(Procedure Linkage Table, 프로시저 링키지 테이블)**: 함수 주소를 지연 해석하는 스텁 코드입니다.

```
First call to printf():           Second call to printf():
  call printf@PLT                   call printf@PLT
       |                                 |
  PLT stub:                         PLT stub:
    jmp *GOT[printf]                  jmp *GOT[printf]
       |                                 |
  GOT[printf] = resolver            GOT[printf] = actual printf
       |                                 |
  Dynamic linker resolves            Direct jump to printf
  printf, patches GOT                (no resolver overhead)
       |
  Jump to actual printf
```

---

## 7. 위치 독립 코드

### 7.1 PIC란?

**위치 독립 코드(Position-Independent Code)**는 메모리 어디에 로드되든 올바르게 동작하는 코드입니다. 다른 프로세스가 공유 라이브러리를 다른 주소에 매핑하므로 공유 라이브러리에 필수적입니다.

### 7.2 데이터 접근을 위한 PIC

```asm
; Non-PIC (absolute address):
mov eax, [0x601020]        ; hardcoded address

; PIC (PC-relative via GOT):
mov rax, [rip + global_var@GOTPCREL]  ; GOT-relative
mov eax, [rax]                         ; load actual value
```

### 7.3 함수 호출을 위한 PIC

```asm
; Non-PIC:
call 0x400520              ; absolute address

; PIC (via PLT):
call printf@PLT            ; indirect through PLT/GOT
```

### 7.4 PIC 생성

```bash
# Compile with PIC (required for shared libraries)
gcc -fPIC -c library.c -o library.o

# -fPIC vs -fpic:
# -fPIC: always works (larger GOT allowed)
# -fpic: may use shorter instructions (limited GOT size)
```

---

## 8. 런타임 로더

### 8.1 프로그램 시작 과정

`./program`을 실행하면 커널은:

1. ELF 헤더를 읽습니다
2. 세그먼트를 메모리에 매핑합니다
3. 동적 링커 경로를 찾습니다 (`.interp` 섹션에서)
4. 동적 링커(`ld-linux-x86-64.so.2`)에 제어를 넘깁니다

그런 다음 동적 링커는:

1. 필요한 모든 공유 라이브러리를 로드합니다
2. 심볼을 해석합니다 (즉시 또는 지연)
3. 재배치를 수행합니다
4. 초기화 함수를 호출합니다 (`.init`, 생성자)
5. 프로그램의 진입점으로 제어를 넘깁니다

### 8.2 라이브러리 검색 순서

```bash
# The dynamic linker searches for libraries in this order:
# 1. DT_RPATH in the executable (deprecated)
# 2. LD_LIBRARY_PATH environment variable
# 3. DT_RUNPATH in the executable
# 4. /etc/ld.so.cache (ldconfig cache)
# 5. /lib, /usr/lib (default paths)

# View library dependencies
ldd program

# Trace dynamic linker activity
LD_DEBUG=all ./program
```

### 8.3 dlopen: 런타임 로딩

```c
#include <dlfcn.h>

// Load library at runtime
void *handle = dlopen("./libplugin.so", RTLD_LAZY);
if (!handle) {
    fprintf(stderr, "dlopen: %s\n", dlerror());
    exit(1);
}

// Look up symbol
typedef int (*plugin_fn)(int);
plugin_fn func = (plugin_fn)dlsym(handle, "process");
if (!func) {
    fprintf(stderr, "dlsym: %s\n", dlerror());
    exit(1);
}

// Use the function
int result = func(42);

// Unload
dlclose(handle);
```

---

## 9. 요약

- **오브젝트 파일**은 섹션(.text, .data, .bss)으로 조직된 컴파일된 코드를 포함합니다
- **ELF**는 Linux의 표준 바이너리 형식으로, 헤더가 섹션과 세그먼트를 기술합니다
- **심볼 테이블**은 이름을 주소에 매핑합니다; 링커가 미정의 참조를 해석합니다
- **정적 링킹**은 링크 시에 라이브러리 코드를 실행 파일에 복사합니다
- **재배치**는 최종 레이아웃이 결정된 후 코드의 주소를 패치합니다
- **동적 링킹**은 GOT와 PLT를 사용하여 런타임으로 라이브러리 로딩을 미룹니다
- **위치 독립 코드**(PIC)는 공유 라이브러리가 어떤 주소에든 로드될 수 있게 합니다
- **런타임 로더**(ld.so)는 라이브러리를 매핑하고, 심볼을 해석하며, 프로그램을 시작합니다

---

## 10. 연습 문제

1. **ELF 탐색**: 작은 C 프로그램을 컴파일하고 `readelf`와 `objdump`를 사용하여 ELF 구조를 검사하세요.

2. **심볼 해석**: 교차 참조가 있는 세 개의 오브젝트 파일을 만들고 링커가 이를 어떻게 해석하는지 추적하세요.

3. **정적 vs. 동적**: 같은 프로그램을 정적 링킹과 동적 링킹으로 빌드하세요. 실행 파일 크기와 `ldd` 출력을 비교하세요.

4. **PLT 추적**: `ltrace` 또는 `LD_DEBUG=bindings`를 사용하여 PLT를 통한 지연 심볼 해석을 관찰하세요.

5. **플러그인 시스템**: `dlopen`/`dlsym`을 사용하여 런타임에 공유 라이브러리를 로드하고 그 안의 함수를 호출하는 C 프로그램을 작성하세요.

---

## 11. 참고 자료

1. Levine, J. R. (1999). *Linkers and Loaders*. Morgan Kaufmann.
2. ELF 명세: https://refspecs.linuxfoundation.org/elf/elf.pdf
3. Drepper, U. (2011). "How To Write Shared Libraries." https://www.akkadia.org/drepper/dsohowto.pdf
4. Bryant, R. E., O'Hallaron, D. R. (2015). *Computer Systems: A Programmer's Perspective*. Chapter 7: Linking.
5. System V ABI: https://refspecs.linuxfoundation.org/SystemV/abi/

---

**이전**: [24. 클로저 변환](./24_Closure_Conversion.md) | **다음**: [26. 디버그 정보](./26_Debug_Information.md)
