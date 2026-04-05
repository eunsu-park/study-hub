# 바이너리 기초

**이전**: [08. 클라이언트측 공격](./08_Client_Side_Attacks.md) | **다음**: [10. 버퍼 오버플로우](./10_Buffer_Overflow.md)

---

바이너리 익스플로잇(binary exploitation)을 이해하려면 프로그램이 가장 낮은 수준에서 어떻게 실행되는지에 대한 확실한 기초가 필요하다. 이 레슨에서는 x86/x64 어셈블리 언어, 호출 규약(calling convention), 메모리 레이아웃, 실행 파일의 구조를 다룬다 — 버퍼 오버플로우, 리버스 엔지니어링, 익스플로잇 개발에 필수적인 지식이다.

> **중요**: 바이너리 익스플로잇 기술은 CTF 대회, 승인된 보안 평가, 개인 실습 환경에서만 적용해야 한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. x86/x64 레지스터의 목적과 크기를 이해한다
2. 기본적인 x86 어셈블리 명령어를 읽고 작성한다
3. 스택을 통한 함수 호출을 추적한다
4. cdecl과 System V AMD64 호출 규약을 이해한다
5. 실행 중인 프로세스의 메모리 레이아웃을 매핑한다
6. ELF 헤더와 섹션 테이블을 파싱한다
7. GDB를 사용하여 기본적인 바이너리 분석과 디버깅을 수행한다
8. 컴파일된 코드에서 보안 관련 패턴을 식별한다

---

## 목차

1. [x86 아키텍처 개요](#1-x86-아키텍처-개요)
2. [레지스터와 플래그](#2-레지스터와-플래그)
3. [어셈블리 명령어](#3-어셈블리-명령어)
4. [호출 규약](#4-호출-규약)
5. [스택과 스택 프레임](#5-스택과-스택-프레임)
6. [힙](#6-힙)
7. [ELF 바이너리 형식](#7-elf-바이너리-형식)
8. [PE 바이너리 형식](#8-pe-바이너리-형식)
9. [프로세스 메모리 레이아웃](#9-프로세스-메모리-레이아웃)
10. [바이너리 분석을 위한 GDB 기초](#10-바이너리-분석을-위한-gdb-기초)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. x86 아키텍처 개요

x86 아키텍처(및 64비트 확장인 x86-64/AMD64)는 데스크톱, 서버, 클라우드 인프라에서 지배적인 CPU 아키텍처이다.

### 1.1 주요 특성

- **CISC 아키텍처**: 복합 명령어 집합 컴퓨팅(Complex Instruction Set Computing)
- **리틀 엔디안(Little-endian)**: 최하위 바이트가 가장 낮은 주소에 저장
- **가변 길이 명령어**: 1~15바이트
- **레지스터 기반**: 속도를 위해 CPU 레지스터를 사용하여 연산

### 1.2 워드 크기

| 아키텍처 | 워드 크기 | 주소 공간 | 레지스터 이름 |
|-------------|-----------|---------------|----------------|
| x86 (32비트) | 4바이트 | 4 GB | EAX, EBX, ESP, EIP |
| x86-64 (64비트) | 8바이트 | 256 TB (48비트) | RAX, RBX, RSP, RIP |

---

## 2. 레지스터와 플래그

### 2.1 범용 레지스터 (x86-64)

```
64-bit    32-bit    16-bit    8-bit high   8-bit low
RAX       EAX       AX        AH           AL        — Accumulator
RBX       EBX       BX        BH           BL        — Base
RCX       ECX       CX        CH           CL        — Counter
RDX       EDX       DX        DH           DL        — Data
RSI       ESI       SI        —            SIL       — Source Index
RDI       EDI       DI        —            DIL       — Destination Index
RSP       ESP       SP        —            SPL       — Stack Pointer
RBP       EBP       BP        —            BPL       — Base Pointer
R8-R15    R8D-R15D  R8W-R15W  —            R8B-R15B  — Extended (x64 only)
```

### 2.2 특수 레지스터

- **RIP/EIP**: 명령 포인터(Instruction Pointer) — 다음 명령어의 주소
- **RSP/ESP**: 스택 포인터(Stack Pointer) — 스택의 최상단
- **RBP/EBP**: 베이스 포인터(Base Pointer) — 현재 스택 프레임의 기반
- **RFLAGS/EFLAGS**: 상태 플래그(zero, carry, sign, overflow)

```python
"""
x86 register and memory layout reference module.

Provides educational reference for binary analysis
and exploit development.
"""

from dataclasses import dataclass, field
from enum import Enum


class RegisterSize(Enum):
    BYTE = 1
    WORD = 2
    DWORD = 4
    QWORD = 8


@dataclass
class Register:
    """x86/x64 CPU register."""
    name_64: str
    name_32: str
    name_16: str
    name_8l: str
    purpose: str
    calling_convention_role: str = ""


REGISTERS = [
    Register("RAX", "EAX", "AX", "AL",
             "Accumulator — return values",
             "Return value (both cdecl and SysV)"),
    Register("RBX", "EBX", "BX", "BL",
             "Base — callee-saved",
             "Callee-saved (preserved across calls)"),
    Register("RCX", "ECX", "CX", "CL",
             "Counter — loop counter, 4th arg (Win64)",
             "4th argument (Windows x64)"),
    Register("RDX", "EDX", "DX", "DL",
             "Data — I/O, 3rd arg",
             "3rd argument (SysV), 2nd (Win64)"),
    Register("RSI", "ESI", "SI", "SIL",
             "Source Index — string operations",
             "2nd argument (System V AMD64)"),
    Register("RDI", "EDI", "DI", "DIL",
             "Destination Index — string operations",
             "1st argument (System V AMD64)"),
    Register("RSP", "ESP", "SP", "SPL",
             "Stack Pointer — top of stack",
             "Always points to top of stack"),
    Register("RBP", "EBP", "BP", "BPL",
             "Base Pointer — bottom of stack frame",
             "Frame pointer (optional in optimized code)"),
    Register("RIP", "EIP", "IP", "-",
             "Instruction Pointer — next instruction",
             "Cannot be directly modified by MOV"),
]


@dataclass
class MemoryRegion:
    """A region in process memory layout."""
    name: str
    address_range: str
    purpose: str
    permissions: str
    grows: str = ""


PROCESS_MEMORY_LAYOUT = [
    MemoryRegion("Kernel Space", "0xFFFF... (high)",
                 "OS kernel (not accessible to user)", "---"),
    MemoryRegion("Stack", "High addresses, grows down",
                 "Local variables, return addresses, function args",
                 "rw-", "Grows downward ↓"),
    MemoryRegion("(gap)", "...", "Guard pages between stack and heap", "---"),
    MemoryRegion("Heap", "Above BSS, grows up",
                 "Dynamic allocations (malloc/new)",
                 "rw-", "Grows upward ↑"),
    MemoryRegion("BSS", "After Data segment",
                 "Uninitialized global/static variables", "rw-"),
    MemoryRegion("Data", "After Text segment",
                 "Initialized global/static variables", "rw-"),
    MemoryRegion("Text (Code)", "Low addresses (0x400000 typical)",
                 "Executable code (machine instructions)", "r-x"),
]


def display_memory_layout():
    """Display the process memory layout."""
    print("Process Memory Layout (x86-64 Linux)")
    print("=" * 60)
    print("  High addresses (0xFFFF...)")
    print("  ─────────────────────────────")
    for region in PROCESS_MEMORY_LAYOUT:
        print(f"  │ {region.name:15s} │ {region.permissions} │ {region.grows}")
        print(f"  │ {region.purpose[:50]:50s} │")
        print("  ─────────────────────────────")
    print("  Low addresses (0x0000...)")


def display_registers():
    """Display register reference table."""
    print("\nx86-64 Register Reference")
    print("=" * 70)
    print(f"  {'64-bit':8s} {'32-bit':8s} {'16-bit':8s} {'8-bit':8s} Purpose")
    print("  " + "-" * 66)
    for reg in REGISTERS:
        print(f"  {reg.name_64:8s} {reg.name_32:8s} {reg.name_16:8s} "
              f"{reg.name_8l:8s} {reg.purpose}")


if __name__ == "__main__":
    display_registers()
    print()
    display_memory_layout()
```

---

## 3. 어셈블리 명령어

### 3.1 일반적인 x86 명령어

| 범주 | 명령어 | 설명 | 예제 |
|----------|------------|-------------|---------|
| 데이터 이동 | `mov` | 데이터 복사 | `mov eax, 42` |
| | `push` | 스택에 푸시 | `push ebp` |
| | `pop` | 스택에서 팝 | `pop ebp` |
| | `lea` | 유효 주소 로드 | `lea rax, [rbp-8]` |
| 산술 | `add` | 덧셈 | `add eax, 5` |
| | `sub` | 뺄셈 | `sub esp, 0x20` |
| | `imul` | 부호 있는 곱셈 | `imul eax, ebx` |
| | `inc/dec` | 증가/감소 | `inc ecx` |
| 논리 | `and` | 비트 AND | `and eax, 0xFF` |
| | `or` | 비트 OR | `or eax, 0x1` |
| | `xor` | 비트 XOR | `xor eax, eax` (제로화) |
| | `shr/shl` | 오른쪽/왼쪽 시프트 | `shr eax, 4` |
| 제어 흐름 | `jmp` | 무조건 점프 | `jmp 0x401000` |
| | `je/jne` | 같으면/같지 않으면 점프 | `je label` |
| | `jl/jg` | 작으면/크면 점프 | `jg loop_start` |
| | `call` | 함수 호출 | `call printf` |
| | `ret` | 함수에서 복귀 | `ret` |
| 비교 | `cmp` | 비교 (플래그 설정) | `cmp eax, 0` |
| | `test` | 비트 AND (플래그 설정) | `test eax, eax` |
| 스택 | `enter` | 스택 프레임 생성 | `enter 32, 0` |
| | `leave` | 스택 프레임 해제 | `leave` |

### 3.2 함수 프롤로그와 에필로그

```nasm
; Function prologue — sets up stack frame
push rbp        ; Save caller's base pointer
mov rbp, rsp    ; Set up our base pointer
sub rsp, 0x20   ; Allocate 32 bytes for local variables

; ... function body ...

; Function epilogue — tears down stack frame
leave           ; Equivalent to: mov rsp, rbp; pop rbp
ret             ; Pop return address and jump to it
```

---

## 4. 호출 규약

### 4.1 cdecl (x86 32비트, C 기본값)

- 인수가 오른쪽에서 왼쪽으로 스택에 푸시된다
- 호출자(caller)가 스택을 정리한다
- 반환 값은 EAX에 저장된다

### 4.2 System V AMD64 (x86-64 Linux/macOS)

- 처음 6개 정수 인수: RDI, RSI, RDX, RCX, R8, R9
- 처음 8개 부동소수점 인수: XMM0-XMM7
- 추가 인수는 스택에 전달된다
- 반환 값은 RAX에 저장된다
- 피호출자 저장(callee-saved): RBX, RBP, R12-R15

### 4.3 Microsoft x64 (Windows)

- 처음 4개 인수: RCX, RDX, R8, R9
- 스택에 32바이트 섀도 공간(shadow space) 필요
- 반환 값은 RAX에 저장된다

---

## 5. 스택과 스택 프레임

```
High addresses
┌──────────────────────┐
│ Caller's stack frame  │
├──────────────────────┤
│ Return address        │ ← Saved by CALL instruction
├──────────────────────┤
│ Saved RBP             │ ← Saved by function prologue
├──────────────────────┤ ← RBP points here
│ Local variable 1      │ [rbp - 8]
│ Local variable 2      │ [rbp - 16]
│ ...                   │
├──────────────────────┤ ← RSP points here
│ (future growth ↓)     │
└──────────────────────┘
Low addresses
```

---

## 6. 힙

힙(heap)은 동적 메모리 할당을 제공한다. 주요 개념:

- C에서 `malloc()`/`free()`, C++에서 `new`/`delete`
- 메모리 할당자: ptmalloc2, jemalloc, tcmalloc
- 힙 메타데이터는 할당된 청크(chunk)와 함께 저장된다
- 힙 익스플로잇: use-after-free, 힙 오버플로우, double-free

---

## 7. ELF 바이너리 형식

```python
"""
ELF (Executable and Linkable Format) parser.

Parses ELF headers for binary analysis during
security assessments and CTF challenges.
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ELFHeader:
    """Parsed ELF file header."""
    magic: bytes
    class_: int  # 1=32-bit, 2=64-bit
    endianness: int  # 1=little, 2=big
    elf_type: int  # 2=executable, 3=shared object
    machine: int  # 0x3E=x86-64, 0x03=x86
    entry_point: int
    ph_offset: int  # Program header table offset
    sh_offset: int  # Section header table offset
    ph_count: int
    sh_count: int

    @property
    def is_valid(self) -> bool:
        return self.magic == b'\x7fELF'

    @property
    def arch(self) -> str:
        return "x86-64" if self.class_ == 2 else "x86"

    @property
    def type_name(self) -> str:
        types = {1: "Relocatable", 2: "Executable", 3: "Shared Object", 4: "Core"}
        return types.get(self.elf_type, "Unknown")

    @property
    def is_pie(self) -> bool:
        """Check if binary is Position Independent (PIE)."""
        return self.elf_type == 3  # ET_DYN


@dataclass
class ELFSection:
    """A section in the ELF binary."""
    name: str
    type_: int
    flags: int
    address: int
    offset: int
    size: int

    @property
    def is_executable(self) -> bool:
        return bool(self.flags & 0x4)

    @property
    def is_writable(self) -> bool:
        return bool(self.flags & 0x1)


def parse_elf_header(filepath: str) -> Optional[ELFHeader]:
    """Parse the ELF file header."""
    try:
        with open(filepath, "rb") as f:
            magic = f.read(4)
            if magic != b'\x7fELF':
                return None

            class_ = struct.unpack("B", f.read(1))[0]
            endianness = struct.unpack("B", f.read(1))[0]
            f.read(10)  # Skip padding

            fmt = "<" if endianness == 1 else ">"
            if class_ == 2:  # 64-bit
                elf_type = struct.unpack(f"{fmt}H", f.read(2))[0]
                machine = struct.unpack(f"{fmt}H", f.read(2))[0]
                f.read(4)  # version
                entry = struct.unpack(f"{fmt}Q", f.read(8))[0]
                ph_off = struct.unpack(f"{fmt}Q", f.read(8))[0]
                sh_off = struct.unpack(f"{fmt}Q", f.read(8))[0]
                f.read(4)  # flags
                f.read(2)  # header size
                f.read(2)  # ph entry size
                ph_count = struct.unpack(f"{fmt}H", f.read(2))[0]
                f.read(2)  # sh entry size
                sh_count = struct.unpack(f"{fmt}H", f.read(2))[0]
            else:  # 32-bit
                elf_type = struct.unpack(f"{fmt}H", f.read(2))[0]
                machine = struct.unpack(f"{fmt}H", f.read(2))[0]
                f.read(4)
                entry = struct.unpack(f"{fmt}I", f.read(4))[0]
                ph_off = struct.unpack(f"{fmt}I", f.read(4))[0]
                sh_off = struct.unpack(f"{fmt}I", f.read(4))[0]
                f.read(4)
                f.read(2)
                f.read(2)
                ph_count = struct.unpack(f"{fmt}H", f.read(2))[0]
                f.read(2)
                sh_count = struct.unpack(f"{fmt}H", f.read(2))[0]

            return ELFHeader(
                magic=magic, class_=class_, endianness=endianness,
                elf_type=elf_type, machine=machine, entry_point=entry,
                ph_offset=ph_off, sh_offset=sh_off,
                ph_count=ph_count, sh_count=sh_count,
            )

    except (OSError, struct.error):
        return None


def check_security_features(filepath: str) -> dict:
    """Check binary security features (simplified checksec)."""
    features = {
        "PIE": False,
        "NX": False,
        "Canary": False,
        "RELRO": False,
    }

    header = parse_elf_header(filepath)
    if header:
        features["PIE"] = header.is_pie

    # Full checksec requires reading program headers and symbols
    # Use 'checksec' tool for complete analysis:
    # checksec --file=binary

    return features


if __name__ == "__main__":
    print("ELF Parser Module")
    print("=" * 40)
    print("Usage:")
    print("  header = parse_elf_header('/bin/ls')")
    print("  print(f'Arch: {header.arch}')")
    print("  print(f'Type: {header.type_name}')")
    print("  print(f'Entry: 0x{header.entry_point:x}')")
    print("  print(f'PIE: {header.is_pie}')")
```

---

## 8. PE 바이너리 형식

PE(Portable Executable) 형식은 Windows 실행 파일(.exe, .dll, .sys)에서 사용된다.

### 8.1 PE 구조

```
DOS Header (MZ)
  → PE Signature ("PE\0\0")
    → COFF File Header
      → Optional Header (entry point, image base)
        → Section Headers (.text, .data, .rdata, .rsrc)
          → Section Data
```

---

## 9. 프로세스 메모리 레이아웃

### 9.1 Linux 프로세스 메모리

```
0xFFFFFFFFFFFFFFFF ┌─────────────────┐
                   │  Kernel Space    │ (not accessible to user programs)
0x7FFFFFFFFFFF     ├─────────────────┤
                   │  Stack           │ ↓ grows down
                   │                  │
                   │  (unmapped)      │
                   │                  │
                   │  Heap            │ ↑ grows up
                   ├─────────────────┤
                   │  BSS             │ (uninitialized globals)
                   │  Data            │ (initialized globals)
                   │  Text (Code)     │ (executable instructions)
0x0000000000400000 └─────────────────┘
```

### 9.2 보안 시사점

- **스택**: 버퍼 오버플로우가 반환 주소를 덮어쓸 수 있다
- **힙**: Use-after-free 및 힙 손상 공격이 가능하다
- **텍스트**: 코드 인젝션 대상; NX 비트가 데이터 실행을 방지한다

---

## 10. 바이너리 분석을 위한 GDB 기초

### 10.1 필수 GDB 명령어

```bash
# Start GDB with a binary
gdb ./binary

# Run with arguments
(gdb) run arg1 arg2

# Set breakpoints
(gdb) break main
(gdb) break *0x401234

# Step through code
(gdb) stepi       # Single instruction
(gdb) nexti       # Next instruction (skip calls)
(gdb) continue    # Continue execution

# Examine memory
(gdb) x/20x $rsp  # 20 hex words at stack pointer
(gdb) x/s 0x402000 # String at address
(gdb) x/10i $rip  # 10 instructions at instruction pointer

# Print registers
(gdb) info registers
(gdb) print $rax

# View stack
(gdb) backtrace
(gdb) info frame

# Disassemble
(gdb) disas main
(gdb) disas 0x401000,0x401050
```

### 10.2 GDB 확장 도구

- **pwndbg**: 익스플로잇 개발에 최적 (`pip install pwndbg`)
- **GEF**: GDB Enhanced Features (`pip install gef`)
- **peda**: Python Exploit Development Assistance

---

## 11. 연습 문제

1. **레지스터 실습**: C 프로그램을 작성하고 컴파일한 후, 함수가 호출될 때 GDB를 사용하여 레지스터 값을 추적한다.
2. **스택 분석**: 함수 호출 전후에 브레이크포인트를 설정한다. 저장된 RBP와 반환 주소를 포함하여 스택 프레임을 매핑한다.
3. **ELF 파싱**: ELF 파서를 사용하여 5개의 서로 다른 바이너리를 분석한다. PIE vs non-PIE, stripped vs unstripped를 비교한다.
4. **어셈블리 읽기**: 간단한 함수의 디스어셈블리가 주어지면, 동등한 C 코드를 재구성한다.
5. **메모리 맵**: `/proc/self/maps`를 사용하여 실행 중인 프로세스의 메모리 레이아웃을 검사한다. 모든 영역을 식별한다.
6. **GDB 마스터**: crackmes.one의 "crackme" 챌린지를 GDB만 사용하여 완료한다.

---

## 12. 요약

바이너리 기초는 익스플로잇의 토대를 제공한다:

- **x86/x64 아키텍처**는 레지스터, 스택, 특정 호출 규약을 사용한다
- **레지스터**는 특정 목적을 수행한다 — RSP는 스택, RIP는 실행 흐름을 위한 것이다
- **어셈블리 명령어**는 데이터를 이동하고, 산술을 수행하며, 프로그램 흐름을 제어한다
- **호출 규약**은 함수가 인수를 받고 값을 반환하는 방법을 정의한다
- **메모리 레이아웃**은 코드, 데이터, 힙, 스택을 서로 다른 권한으로 분리한다
- **ELF 형식**은 헤더, 섹션, 세그먼트로 Linux 실행 파일을 구조화한다
- **GDB**는 동적 바이너리 분석의 필수 도구이다

---

## 13. 참고 자료

- Intel x86 Software Developer Manuals: https://www.intel.com/sdm
- x86 Assembly Wikibook: https://en.wikibooks.org/wiki/X86_Assembly
- ELF Specification: https://refspecs.linuxfoundation.org/elf/elf.pdf
- System V AMD64 ABI: https://gitlab.com/x86-psABIs/x86-64-ABI
- GDB Documentation: https://sourceware.org/gdb/documentation/
- pwndbg: https://github.com/pwndbg/pwndbg
