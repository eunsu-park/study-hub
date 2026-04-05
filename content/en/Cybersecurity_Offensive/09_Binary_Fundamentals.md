# Binary Fundamentals

**Previous**: [08. Client-Side Attacks](./08_Client_Side_Attacks.md) | **Next**: [10. Buffer Overflow](./10_Buffer_Overflow.md)

---

Understanding binary exploitation requires a solid foundation in how programs execute at the lowest level. This lesson covers x86/x64 assembly language, calling conventions, memory layout, and the structure of executable files — the essential knowledge needed for buffer overflows, reverse engineering, and exploit development.

> **IMPORTANT**: Binary exploitation skills should only be applied in CTF competitions, authorized security assessments, and personal lab environments.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Understand x86/x64 register purposes and sizes
2. Read and write basic x86 assembly instructions
3. Trace function calls through the stack
4. Understand cdecl and System V AMD64 calling conventions
5. Map the memory layout of a running process
6. Parse ELF headers and section tables
7. Use GDB for basic binary analysis and debugging
8. Identify security-relevant patterns in compiled code

---

## Table of Contents

1. [x86 Architecture Overview](#1-x86-architecture-overview)
2. [Registers and Flags](#2-registers-and-flags)
3. [Assembly Instructions](#3-assembly-instructions)
4. [Calling Conventions](#4-calling-conventions)
5. [The Stack and Stack Frames](#5-the-stack-and-stack-frames)
6. [The Heap](#6-the-heap)
7. [ELF Binary Format](#7-elf-binary-format)
8. [PE Binary Format](#8-pe-binary-format)
9. [Memory Layout of a Process](#9-memory-layout-of-a-process)
10. [GDB Basics for Binary Analysis](#10-gdb-basics-for-binary-analysis)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. x86 Architecture Overview

The x86 architecture (and its 64-bit extension x86-64/AMD64) is the dominant CPU architecture for desktops, servers, and cloud infrastructure.

### 1.1 Key Characteristics

- **CISC architecture**: Complex Instruction Set Computing
- **Little-endian**: Least significant byte stored at lowest address
- **Variable-length instructions**: 1 to 15 bytes
- **Register-based**: Operations use CPU registers for speed

### 1.2 Word Sizes

| Architecture | Word Size | Address Space | Register Names |
|-------------|-----------|---------------|----------------|
| x86 (32-bit) | 4 bytes | 4 GB | EAX, EBX, ESP, EIP |
| x86-64 (64-bit) | 8 bytes | 256 TB (48-bit) | RAX, RBX, RSP, RIP |

---

## 2. Registers and Flags

### 2.1 General Purpose Registers (x86-64)

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

### 2.2 Special Registers

- **RIP/EIP**: Instruction Pointer — address of next instruction
- **RSP/ESP**: Stack Pointer — top of the stack
- **RBP/EBP**: Base Pointer — base of current stack frame
- **RFLAGS/EFLAGS**: Status flags (zero, carry, sign, overflow)

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

## 3. Assembly Instructions

### 3.1 Common x86 Instructions

| Category | Instruction | Description | Example |
|----------|------------|-------------|---------|
| Data Movement | `mov` | Copy data | `mov eax, 42` |
| | `push` | Push onto stack | `push ebp` |
| | `pop` | Pop from stack | `pop ebp` |
| | `lea` | Load effective address | `lea rax, [rbp-8]` |
| Arithmetic | `add` | Addition | `add eax, 5` |
| | `sub` | Subtraction | `sub esp, 0x20` |
| | `imul` | Signed multiply | `imul eax, ebx` |
| | `inc/dec` | Increment/decrement | `inc ecx` |
| Logic | `and` | Bitwise AND | `and eax, 0xFF` |
| | `or` | Bitwise OR | `or eax, 0x1` |
| | `xor` | Bitwise XOR | `xor eax, eax` (zero) |
| | `shr/shl` | Shift right/left | `shr eax, 4` |
| Control Flow | `jmp` | Unconditional jump | `jmp 0x401000` |
| | `je/jne` | Jump if equal/not equal | `je label` |
| | `jl/jg` | Jump if less/greater | `jg loop_start` |
| | `call` | Call function | `call printf` |
| | `ret` | Return from function | `ret` |
| Comparison | `cmp` | Compare (sets flags) | `cmp eax, 0` |
| | `test` | Bitwise AND (sets flags) | `test eax, eax` |
| Stack | `enter` | Create stack frame | `enter 32, 0` |
| | `leave` | Destroy stack frame | `leave` |

### 3.2 Function Prologue and Epilogue

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

## 4. Calling Conventions

### 4.1 cdecl (x86 32-bit, C default)

- Arguments pushed right-to-left on the stack
- Caller cleans up the stack
- Return value in EAX

### 4.2 System V AMD64 (x86-64 Linux/macOS)

- First 6 integer args: RDI, RSI, RDX, RCX, R8, R9
- First 8 float args: XMM0-XMM7
- Additional arguments on the stack
- Return value in RAX
- Callee-saved: RBX, RBP, R12-R15

### 4.3 Microsoft x64 (Windows)

- First 4 args: RCX, RDX, R8, R9
- 32-byte shadow space required on stack
- Return value in RAX

---

## 5. The Stack and Stack Frames

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

## 6. The Heap

The heap provides dynamic memory allocation. Key concepts:

- `malloc()`/`free()` in C, `new`/`delete` in C++
- Memory allocators: ptmalloc2, jemalloc, tcmalloc
- Heap metadata stored alongside allocated chunks
- Heap exploitation: use-after-free, heap overflow, double-free

---

## 7. ELF Binary Format

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

## 8. PE Binary Format

The Portable Executable (PE) format is used by Windows executables (.exe, .dll, .sys).

### 8.1 PE Structure

```
DOS Header (MZ)
  → PE Signature ("PE\0\0")
    → COFF File Header
      → Optional Header (entry point, image base)
        → Section Headers (.text, .data, .rdata, .rsrc)
          → Section Data
```

---

## 9. Memory Layout of a Process

### 9.1 Linux Process Memory

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

### 9.2 Security Implications

- **Stack**: Buffer overflows can overwrite return addresses
- **Heap**: Use-after-free and heap corruption attacks
- **Text**: Code injection targets; NX bit prevents execution of data

---

## 10. GDB Basics for Binary Analysis

### 10.1 Essential GDB Commands

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

### 10.2 GDB Extensions

- **pwndbg**: Best for exploit development (`pip install pwndbg`)
- **GEF**: GDB Enhanced Features (`pip install gef`)
- **peda**: Python Exploit Development Assistance

---

## 11. Exercises

1. **Register Practice**: Write a program in C, compile it, and trace register values through GDB as functions are called.
2. **Stack Analysis**: Set a breakpoint before and after a function call. Map the stack frame including saved RBP and return address.
3. **ELF Parsing**: Use the ELF parser to analyze 5 different binaries. Compare PIE vs non-PIE, stripped vs unstripped.
4. **Assembly Reading**: Given a disassembly of a simple function, reconstruct the equivalent C code.
5. **Memory Map**: Use `/proc/self/maps` to examine the memory layout of a running process. Identify all regions.
6. **GDB Mastery**: Complete the "crackme" challenges on crackmes.one using only GDB.

---

## 12. Summary

Binary fundamentals provide the foundation for exploitation:

- **x86/x64 architecture** uses registers, the stack, and specific calling conventions
- **Registers** serve specific purposes — RSP for the stack, RIP for execution flow
- **Assembly instructions** move data, perform arithmetic, and control program flow
- **Calling conventions** define how functions receive arguments and return values
- **Memory layout** separates code, data, heap, and stack with different permissions
- **ELF format** structures Linux executables with headers, sections, and segments
- **GDB** is the essential tool for dynamic binary analysis

---

## 13. References

- Intel x86 Software Developer Manuals: https://www.intel.com/sdm
- x86 Assembly Wikibook: https://en.wikibooks.org/wiki/X86_Assembly
- ELF Specification: https://refspecs.linuxfoundation.org/elf/elf.pdf
- System V AMD64 ABI: https://gitlab.com/x86-psABIs/x86-64-ABI
- GDB Documentation: https://sourceware.org/gdb/documentation/
- pwndbg: https://github.com/pwndbg/pwndbg
