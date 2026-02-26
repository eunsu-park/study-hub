# Assembly Language Basics

**Previous**: [09_Instruction_Set_Architecture.md](./09_Instruction_Set_Architecture.md) | **Next**: [11_Pipelining.md](./11_Pipelining.md)

**Difficulty**: ⭐⭐⭐

**Prerequisites**: Instruction Set Architecture (ISA), Basic CPU structure

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the relationship between assembly language and machine code
2. Identify common registers and their conventions (general-purpose, stack pointer, frame pointer)
3. Write simple assembly programs using data transfer, arithmetic, and branch instructions
4. Trace the execution of an assembly program by hand
5. Explain how high-level language constructs (if/else, loops, function calls) map to assembly

---

Assembly language is the thinnest layer of abstraction above raw machine code. Learning assembly does not mean you will program in it daily -- but it gives you x-ray vision into what your C, Python, or Java programs actually do on the hardware. When you need to optimize performance, debug a crash dump, or understand a security exploit, assembly knowledge is indispensable.

## Table of Contents

1. [Assembly Language Concepts](#1-assembly-language-concepts)
2. [x86 Assembly Basics](#2-x86-assembly-basics)
3. [ARM Assembly Basics](#3-arm-assembly-basics)
4. [Arithmetic/Logic Instructions](#4-arithmeticlogic-instructions)
5. [Branch Instructions](#5-branch-instructions)
6. [Memory Access Instructions](#6-memory-access-instructions)
7. [Simple Assembly Program Examples](#7-simple-assembly-program-examples)
8. [Practice Problems](#8-practice-problems)

---

## 1. Assembly Language Concepts

### 1.1 What is Assembly Language?

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Programming Language Hierarchy                        │
│                                                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                    High-Level Languages                          │ │
│    │                                                                 │ │
│    │    ┌─────────────────────────────────────────────────────┐     │ │
│    │    │   C, C++, Java, Python, JavaScript, ...            │     │ │
│    │    │                                                     │     │ │
│    │    │   int sum = a + b;                                  │     │ │
│    │    │                                                     │     │ │
│    │    │   Advantages: Readability, Portability, Productivity│     │ │
│    │    └─────────────────────────────────────────────────────┘     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                   │                                     │
│                                   │ Compile                             │
│                                   ▼                                     │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                      Assembly Language                           │ │
│    │                                                                 │ │
│    │    ┌─────────────────────────────────────────────────────┐     │ │
│    │    │   MOV  EAX, [a]       ; Load a into EAX             │     │ │
│    │    │   ADD  EAX, [b]       ; Add b                       │     │ │
│    │    │   MOV  [sum], EAX     ; Store result in sum         │     │ │
│    │    │                                                     │     │ │
│    │    │   Advantages: Direct hardware control, Optimization │     │ │
│    │    │   Disadvantages: No portability, Low productivity   │     │ │
│    │    └─────────────────────────────────────────────────────┘     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                   │                                     │
│                                   │ Assemble                            │
│                                   ▼                                     │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                      Machine Code                                │ │
│    │                                                                 │ │
│    │    ┌─────────────────────────────────────────────────────┐     │ │
│    │    │   A1 00 10 00 00    ; MOV EAX, [0x1000]             │     │ │
│    │    │   03 05 04 10 00 00 ; ADD EAX, [0x1004]             │     │ │
│    │    │   A3 08 10 00 00    ; MOV [0x1008], EAX             │     │ │
│    │    │                                                     │     │ │
│    │    │   Binary code directly executed by CPU              │     │ │
│    │    └─────────────────────────────────────────────────────┘     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Basic Structure of Assembly

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Assembly Instruction Format                         │
│                                                                         │
│    [label:]  instruction  [operand1 [, operand2 [, operand3]]]  [; comment] │
│                                                                         │
│    Example:                                                             │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                 │ │
│    │    loop:                  ; Label (jump target)                 │ │
│    │        mov  eax, ebx      ; eax = ebx                          │ │
│    │        add  eax, 10       ; eax = eax + 10                     │ │
│    │        cmp  eax, 100      ; Compare eax with 100               │ │
│    │        jl   loop          ; Jump to loop if eax < 100          │ │
│    │                                                                 │ │
│    │    end:                                                         │ │
│    │        ret                ; Return                             │ │
│    │                                                                 │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Components:                                                          │
│    ┌───────────┬─────────────────────────────────────────────────────┐ │
│    │ Label     │ Names a memory address (for jumps/data reference)   │ │
│    ├───────────┼─────────────────────────────────────────────────────┤ │
│    │ Instruction│ Operation for CPU to perform (MOV, ADD, JMP, etc.) │ │
│    ├───────────┼─────────────────────────────────────────────────────┤ │
│    │ Operands  │ Operation targets (registers, memory, immediates)   │ │
│    ├───────────┼─────────────────────────────────────────────────────┤ │
│    │ Comment   │ Code explanation (no effect on execution)           │ │
│    └───────────┴─────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Assembler and Assembly Process

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Assembly Process                                │
│                                                                         │
│    Source file (.asm/.s)                                                │
│         │                                                               │
│         ▼                                                               │
│    ┌─────────────┐                                                     │
│    │  Assembler  │  NASM, GAS, MASM, ARMASM, etc.                      │
│    │             │                                                     │
│    └──────┬──────┘                                                     │
│           │                                                             │
│           ▼                                                             │
│    Object file (.o/.obj)                                                │
│         │                                                               │
│         ▼                                                               │
│    ┌─────────────┐                                                     │
│    │   Linker    │  ld, link.exe, etc.                                 │
│    │             │                                                     │
│    └──────┬──────┘                                                     │
│           │                                                             │
│           ▼                                                             │
│    Executable file (.exe, ELF, Mach-O, etc.)                           │
│                                                                         │
│    Major Assemblers:                                                    │
│    ┌───────────────┬──────────────────────────────────────────────┐    │
│    │ NASM          │ Netwide Assembler (x86, cross-platform)      │    │
│    │ GAS           │ GNU Assembler (AT&T syntax)                  │    │
│    │ MASM          │ Microsoft Macro Assembler (Windows)          │    │
│    │ FASM          │ Flat Assembler (x86)                         │    │
│    │ ARMASM/as     │ ARM Assembler                                │    │
│    └───────────────┴──────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.4 Assembly Syntax Styles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Intel Syntax vs AT&T Syntax                         │
│                                                                         │
│    ┌──────────────────────────┬──────────────────────────────────────┐ │
│    │      Intel Syntax        │          AT&T Syntax                 │ │
│    │   (NASM, MASM)           │         (GAS, GCC)                   │ │
│    ├──────────────────────────┼──────────────────────────────────────┤ │
│    │ mov eax, 10              │ movl $10, %eax                       │ │
│    │ (destination, source)    │ (source, destination)                │ │
│    ├──────────────────────────┼──────────────────────────────────────┤ │
│    │ mov eax, ebx             │ movl %ebx, %eax                      │ │
│    ├──────────────────────────┼──────────────────────────────────────┤ │
│    │ mov eax, [ebx]           │ movl (%ebx), %eax                    │ │
│    ├──────────────────────────┼──────────────────────────────────────┤ │
│    │ mov eax, [ebx+ecx*4+10]  │ movl 10(%ebx,%ecx,4), %eax           │ │
│    ├──────────────────────────┼──────────────────────────────────────┤ │
│    │ Immediate: number        │ Immediate: $ prefix                  │ │
│    │ Register: name           │ Register: % prefix                   │ │
│    │ Size: inferred           │ Size: suffix (b/w/l/q)               │ │
│    └──────────────────────────┴──────────────────────────────────────┘ │
│                                                                         │
│    AT&T size suffixes:                                                  │
│    - b (byte): 8-bit                                                    │
│    - w (word): 16-bit                                                   │
│    - l (long): 32-bit                                                   │
│    - q (quad): 64-bit                                                   │
│                                                                         │
│    This document primarily uses Intel syntax.                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. x86 Assembly Basics

### 2.1 x86 Registers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        x86-64 Registers                                  │
│                                                                         │
│    General-Purpose Registers (64-bit):                                  │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                 │ │
│    │  64-bit    32-bit    16-bit    8-bit(H/L)    Purpose            │ │
│    │  ────────────────────────────────────────────────────────────   │ │
│    │  RAX        EAX       AX        AH/AL        Accumulator        │ │
│    │  RBX        EBX       BX        BH/BL        Base               │ │
│    │  RCX        ECX       CX        CH/CL        Counter            │ │
│    │  RDX        EDX       DX        DH/DL        Data               │ │
│    │  RSI        ESI       SI        SIL          Source Index       │ │
│    │  RDI        EDI       DI        DIL          Destination Index  │ │
│    │  RBP        EBP       BP        BPL          Base Pointer       │ │
│    │  RSP        ESP       SP        SPL          Stack Pointer      │ │
│    │  R8         R8D       R8W       R8B          General (x64 new)  │ │
│    │  R9         R9D       R9W       R9B          General (x64 new)  │ │
│    │  R10-R15    R10D-R15D R10W-R15W R10B-R15B   General (x64 new)  │ │
│    │                                                                 │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Register Size Relationship:                                          │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                 │ │
│    │    63                              31              15    7    0 │ │
│    │    ├───────────────────────────────┼───────────────┼────┼─────┤ │ │
│    │    │              RAX              │               │    │     │ │ │
│    │    │                               │      EAX      │    │     │ │ │
│    │    │                               │               │ AX │     │ │ │
│    │    │                               │               │ AH │ AL  │ │ │
│    │    └───────────────────────────────┴───────────────┴────┴─────┘ │ │
│    │                                                                 │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Special Registers:                                                   │
│    ┌───────────┬─────────────────────────────────────────────────────┐ │
│    │ RIP       │ Instruction Pointer (address of next instruction)   │ │
│    │ RFLAGS    │ Status/Condition flags                             │ │
│    └───────────┴─────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Flags Register

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RFLAGS Register                                  │
│                                                                         │
│    Main Status Flags:                                                   │
│    ┌────────┬────────────────────────────────────────────────────────┐ │
│    │ Flag   │                    Description                         │ │
│    ├────────┼────────────────────────────────────────────────────────┤ │
│    │ CF     │ Carry Flag: Carry/borrow in unsigned operations       │ │
│    │        │ Example: 255 + 1 = 0 (CF=1)                           │ │
│    ├────────┼────────────────────────────────────────────────────────┤ │
│    │ ZF     │ Zero Flag: 1 if result is 0                           │ │
│    │        │ Example: 5 - 5 = 0 (ZF=1)                             │ │
│    ├────────┼────────────────────────────────────────────────────────┤ │
│    │ SF     │ Sign Flag: MSB of result (sign bit)                   │ │
│    │        │ Example: -1 (SF=1), 1 (SF=0)                          │ │
│    ├────────┼────────────────────────────────────────────────────────┤ │
│    │ OF     │ Overflow Flag: Overflow in signed operations          │ │
│    │        │ Example: 127 + 1 = -128 (OF=1)                        │ │
│    ├────────┼────────────────────────────────────────────────────────┤ │
│    │ PF     │ Parity Flag: Even number of 1s in lower 8 bits        │ │
│    ├────────┼────────────────────────────────────────────────────────┤ │
│    │ AF     │ Auxiliary Carry: For BCD operations                   │ │
│    └────────┴────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Control/System Flags:                                                │
│    ┌────────┬────────────────────────────────────────────────────────┐ │
│    │ DF     │ Direction Flag: String operation direction (0=inc, 1=dec)│
│    │ IF     │ Interrupt Flag: Enable/disable interrupts             │ │
│    │ TF     │ Trap Flag: Single-step debugging                      │ │
│    └────────┴────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.3 x86 Basic Instructions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        x86 Basic Instructions                            │
│                                                                         │
│    Data Movement:                                                       │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  mov  dst, src     ; dst = src                                  │ │
│    │  mov  eax, 10      ; eax = 10 (immediate)                       │ │
│    │  mov  eax, ebx     ; eax = ebx (register)                       │ │
│    │  mov  eax, [ebx]   ; eax = Memory[ebx] (memory)                 │ │
│    │  mov  [eax], ebx   ; Memory[eax] = ebx                          │ │
│    │                                                                 │ │
│    │  movzx eax, bl     ; Zero extend (8-bit → 32-bit)               │ │
│    │  movsx eax, bl     ; Sign extend                                │ │
│    │                                                                 │ │
│    │  lea  eax, [ebx+ecx*4]  ; Load effective address into eax      │ │
│    │                         ; (no memory access)                    │ │
│    │                                                                 │ │
│    │  xchg eax, ebx     ; Exchange eax and ebx                       │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Stack Operations:                                                    │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  push eax          ; Push to stack (ESP -= 4, [ESP] = eax)      │ │
│    │  pop  ebx          ; Pop from stack (ebx = [ESP], ESP += 4)     │ │
│    │                                                                 │ │
│    │  Stack structure (x86):                                         │ │
│    │  ┌──────────────┐ High address                                  │ │
│    │  │     ...      │                                               │ │
│    │  ├──────────────┤                                               │ │
│    │  │ Previous val │                                               │ │
│    │  ├──────────────┤                                               │ │
│    │  │ PUSHED value │ ◄─── ESP (Stack Pointer)                      │ │
│    │  ├──────────────┤                                               │ │
│    │  │  (empty)     │                                               │ │
│    │  └──────────────┘ Low address                                   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. ARM Assembly Basics

### 3.1 ARM Registers

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ARM Registers (AArch64)                             │
│                                                                         │
│    General-Purpose Registers:                                           │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                 │ │
│    │  64-bit    32-bit   Purpose                                     │ │
│    │  ─────────────────────────────────────────────────              │ │
│    │  X0-X7     W0-W7    Argument passing / Return value             │ │
│    │  X8        W8       Indirect result register                    │ │
│    │  X9-X15    W9-W15   Temporaries (Caller-saved)                  │ │
│    │  X16-X17   W16-W17  Intra-procedure call                        │ │
│    │  X18       W18      Platform register                           │ │
│    │  X19-X28   W19-W28  Callee-saved                                │ │
│    │  X29 (FP)  W29      Frame Pointer                               │ │
│    │  X30 (LR)  W30      Link Register (return address)              │ │
│    │                                                                 │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Special Registers:                                                   │
│    ┌───────────┬─────────────────────────────────────────────────────┐ │
│    │ SP        │ Stack Pointer                                       │ │
│    │ PC        │ Program Counter (limited direct access)             │ │
│    │ XZR/WZR   │ Zero Register (reads as 0, writes discarded)        │ │
│    │ NZCV      │ Condition flags (N, Z, C, V)                        │ │
│    └───────────┴─────────────────────────────────────────────────────┘ │
│                                                                         │
│    ARM32 (Legacy) Registers:                                            │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  R0-R12   : General-purpose registers                           │ │
│    │  R13 (SP) : Stack Pointer                                       │ │
│    │  R14 (LR) : Link Register                                       │ │
│    │  R15 (PC) : Program Counter                                     │ │
│    │  CPSR     : Current Program Status Register                     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 ARM Condition Codes

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ARM Condition Codes                              │
│                                                                         │
│    Most ARM instructions can execute conditionally:                     │
│                                                                         │
│    ┌────────┬────────────────┬──────────────────────────────────────┐  │
│    │ Suffix │   Condition    │              Description              │  │
│    ├────────┼────────────────┼──────────────────────────────────────┤  │
│    │   EQ   │   Z == 1       │ Equal                                │  │
│    │   NE   │   Z == 0       │ Not Equal                            │  │
│    │   CS/HS│   C == 1       │ Carry Set / Unsigned >=              │  │
│    │   CC/LO│   C == 0       │ Carry Clear / Unsigned <             │  │
│    │   MI   │   N == 1       │ Minus (negative)                     │  │
│    │   PL   │   N == 0       │ Plus (positive or zero)              │  │
│    │   VS   │   V == 1       │ Overflow                             │  │
│    │   VC   │   V == 0       │ No Overflow                          │  │
│    │   HI   │   C==1 & Z==0  │ Unsigned >                           │  │
│    │   LS   │   C==0 | Z==1  │ Unsigned <=                          │  │
│    │   GE   │   N == V       │ Signed >=                            │  │
│    │   LT   │   N != V       │ Signed <                             │  │
│    │   GT   │ N==V & Z==0    │ Signed >                             │  │
│    │   LE   │ N!=V | Z==1    │ Signed <=                            │  │
│    │   AL   │   (always)     │ Always (default)                     │  │
│    └────────┴────────────────┴──────────────────────────────────────┘  │
│                                                                         │
│    Example (ARM32):                                                     │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  CMP   R0, R1       ; R0 - R1, set flags                        │ │
│    │  ADDEQ R2, R2, #1   ; If R0 == R1 then R2++                    │ │
│    │  SUBNE R2, R2, #1   ; If R0 != R1 then R2--                    │ │
│    │                                                                 │ │
│    │  ; Conditional execution without if-else!                       │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 ARM Basic Instructions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ARM Basic Instructions (AArch64)                    │
│                                                                         │
│    Data Movement:                                                       │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  MOV  X0, X1         ; X0 = X1                                  │ │
│    │  MOV  X0, #100       ; X0 = 100 (immediate)                     │ │
│    │  MVN  X0, X1         ; X0 = ~X1 (bit inversion)                 │ │
│    │                                                                 │ │
│    │  ; Loading large immediates                                     │ │
│    │  MOVZ X0, #0x1234                  ; X0 = 0x1234                │ │
│    │  MOVK X0, #0x5678, LSL #16         ; X0 = 0x56781234            │ │
│    │                                                                 │ │
│    │  ; PC-relative load                                             │ │
│    │  ADR  X0, label      ; X0 = address of label                    │ │
│    │  ADRP X0, label      ; Page-aligned address                     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Memory Access:                                                       │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Load                                                         │ │
│    │  LDR  X0, [X1]           ; X0 = Memory[X1]                      │ │
│    │  LDR  X0, [X1, #8]       ; X0 = Memory[X1 + 8]                  │ │
│    │  LDR  X0, [X1, X2]       ; X0 = Memory[X1 + X2]                 │ │
│    │  LDR  X0, [X1, X2, LSL #3] ; X0 = Memory[X1 + X2*8]             │ │
│    │                                                                 │ │
│    │  ; Store                                                        │ │
│    │  STR  X0, [X1]           ; Memory[X1] = X0                      │ │
│    │  STR  X0, [X1, #8]!      ; X1 += 8; Memory[X1] = X0 (pre-index) │ │
│    │  STR  X0, [X1], #8       ; Memory[X1] = X0; X1 += 8 (post-idx)  │ │
│    │                                                                 │ │
│    │  ; Size specification                                           │ │
│    │  LDRB W0, [X1]           ; Load byte                            │ │
│    │  LDRH W0, [X1]           ; Load halfword (16-bit)               │ │
│    │  LDRSW X0, [X1]          ; Load sign-extended word              │ │
│    │                                                                 │ │
│    │  ; Multiple registers                                           │ │
│    │  LDP  X0, X1, [X2]       ; Load register pair                   │ │
│    │  STP  X0, X1, [X2]       ; Store register pair                  │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Arithmetic/Logic Instructions

### 4.1 x86 Arithmetic/Logic Instructions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   x86 Arithmetic/Logic Instructions                      │
│                                                                         │
│    Arithmetic Operations:                                               │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Addition                                                     │ │
│    │  add  eax, ebx        ; eax = eax + ebx                         │ │
│    │  add  eax, 10         ; eax = eax + 10                          │ │
│    │  adc  eax, ebx        ; eax = eax + ebx + CF (with carry)       │ │
│    │  inc  eax             ; eax++ (CF not changed)                  │ │
│    │                                                                 │ │
│    │  ; Subtraction                                                  │ │
│    │  sub  eax, ebx        ; eax = eax - ebx                         │ │
│    │  sbb  eax, ebx        ; eax = eax - ebx - CF (with borrow)      │ │
│    │  dec  eax             ; eax-- (CF not changed)                  │ │
│    │  neg  eax             ; eax = -eax (two's complement)           │ │
│    │                                                                 │ │
│    │  ; Multiplication                                               │ │
│    │  mul  ebx             ; EDX:EAX = EAX * EBX (unsigned)          │ │
│    │  imul eax, ebx        ; EAX = EAX * EBX (signed)                │ │
│    │  imul eax, ebx, 10    ; EAX = EBX * 10                          │ │
│    │                                                                 │ │
│    │  ; Division                                                     │ │
│    │  div  ebx             ; EAX = EDX:EAX / EBX, EDX = remainder    │ │
│    │  idiv ebx             ; Signed division                         │ │
│    │                                                                 │ │
│    │  Note: Before div/idiv, EDX must be set                         │ │
│    │  cdq                  ; Sign-extend EAX to EDX:EAX              │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Logic Operations:                                                    │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  and  eax, ebx        ; eax = eax & ebx                         │ │
│    │  or   eax, ebx        ; eax = eax | ebx                         │ │
│    │  xor  eax, ebx        ; eax = eax ^ ebx                         │ │
│    │  not  eax             ; eax = ~eax                              │ │
│    │  test eax, ebx        ; eax & ebx (result discarded, flags only)│ │
│    │                                                                 │ │
│    │  ; Register initialization using XOR (optimization)             │ │
│    │  xor  eax, eax        ; eax = 0 (more efficient than mov eax,0) │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Shift/Rotate:                                                        │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  shl  eax, 2          ; eax <<= 2 (left shift, *4)              │ │
│    │  shr  eax, 1          ; eax >>= 1 (right shift, logical)        │ │
│    │  sar  eax, 1          ; eax >>= 1 (arithmetic shift, sign kept) │ │
│    │  shl  eax, cl         ; Shift by CL register value              │ │
│    │                                                                 │ │
│    │  rol  eax, 4          ; Rotate left by 4 bits                   │ │
│    │  ror  eax, 4          ; Rotate right by 4 bits                  │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 ARM Arithmetic/Logic Instructions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   ARM Arithmetic/Logic Instructions                      │
│                                                                         │
│    Arithmetic Operations:                                               │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; 3-address format: dst = src1 op src2                         │ │
│    │  ADD  X0, X1, X2        ; X0 = X1 + X2                          │ │
│    │  ADD  X0, X1, #100      ; X0 = X1 + 100                         │ │
│    │  ADDS X0, X1, X2        ; X0 = X1 + X2, update flags            │ │
│    │  ADC  X0, X1, X2        ; X0 = X1 + X2 + Carry                  │ │
│    │                                                                 │ │
│    │  SUB  X0, X1, X2        ; X0 = X1 - X2                          │ │
│    │  SUBS X0, X1, X2        ; X0 = X1 - X2, update flags            │ │
│    │  SBC  X0, X1, X2        ; X0 = X1 - X2 - !Carry                 │ │
│    │  NEG  X0, X1            ; X0 = -X1 (= SUB X0, XZR, X1)          │ │
│    │                                                                 │ │
│    │  MUL  X0, X1, X2        ; X0 = X1 * X2                          │ │
│    │  MADD X0, X1, X2, X3    ; X0 = X3 + (X1 * X2)                   │ │
│    │  MSUB X0, X1, X2, X3    ; X0 = X3 - (X1 * X2)                   │ │
│    │                                                                 │ │
│    │  SDIV X0, X1, X2        ; X0 = X1 / X2 (signed)                 │ │
│    │  UDIV X0, X1, X2        ; X0 = X1 / X2 (unsigned)               │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Logic Operations:                                                    │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  AND  X0, X1, X2        ; X0 = X1 & X2                          │ │
│    │  ORR  X0, X1, X2        ; X0 = X1 | X2                          │ │
│    │  EOR  X0, X1, X2        ; X0 = X1 ^ X2                          │ │
│    │  BIC  X0, X1, X2        ; X0 = X1 & ~X2 (bit clear)             │ │
│    │  ORN  X0, X1, X2        ; X0 = X1 | ~X2                         │ │
│    │  TST  X0, X1            ; X0 & X1, set flags only               │ │
│    │                                                                 │ │
│    │  ; Shift operators (can be applied to second operand)           │ │
│    │  ADD  X0, X1, X2, LSL #2  ; X0 = X1 + (X2 << 2)                 │ │
│    │  SUB  X0, X1, X2, LSR #1  ; X0 = X1 - (X2 >> 1)                 │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Shift/Rotate:                                                        │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  LSL  X0, X1, #4        ; X0 = X1 << 4 (logical left)           │ │
│    │  LSR  X0, X1, #4        ; X0 = X1 >> 4 (logical right)          │ │
│    │  ASR  X0, X1, #4        ; X0 = X1 >> 4 (arithmetic right)       │ │
│    │  ROR  X0, X1, #4        ; Rotate right                          │ │
│    │                                                                 │ │
│    │  LSL  X0, X1, X2        ; X0 = X1 << X2 (by register value)     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Branch Instructions

### 5.1 x86 Branch Instructions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        x86 Branch Instructions                           │
│                                                                         │
│    Compare Instructions:                                                │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  cmp  eax, ebx        ; eax - ebx, discard result, set flags    │ │
│    │  test eax, ebx        ; eax & ebx, discard result, set flags    │ │
│    │                                                                 │ │
│    │  Example:                                                       │ │
│    │  cmp  eax, 0          ; Check if eax is 0                       │ │
│    │  test eax, eax        ; Check if eax is 0 (more efficient)      │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Conditional Jumps:                                                   │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Unsigned comparison                                          │ │
│    │  je   label           ; Jump if Equal (ZF=1)                    │ │
│    │  jne  label           ; Jump if Not Equal (ZF=0)                │ │
│    │  ja   label           ; Jump if Above (CF=0, ZF=0)              │ │
│    │  jae  label           ; Jump if Above or Equal (CF=0)           │ │
│    │  jb   label           ; Jump if Below (CF=1)                    │ │
│    │  jbe  label           ; Jump if Below or Equal (CF=1 or ZF=1)   │ │
│    │                                                                 │ │
│    │  ; Signed comparison                                            │ │
│    │  jg   label           ; Jump if Greater (signed)                │ │
│    │  jge  label           ; Jump if Greater or Equal                │ │
│    │  jl   label           ; Jump if Less                            │ │
│    │  jle  label           ; Jump if Less or Equal                   │ │
│    │                                                                 │ │
│    │  ; Flag-based                                                   │ │
│    │  jz   label           ; Jump if Zero (= je)                     │ │
│    │  jnz  label           ; Jump if Not Zero (= jne)                │ │
│    │  js   label           ; Jump if Sign (SF=1, negative)           │ │
│    │  jns  label           ; Jump if Not Sign (SF=0)                 │ │
│    │  jo   label           ; Jump if Overflow                        │ │
│    │  jno  label           ; Jump if No Overflow                     │ │
│    │  jc   label           ; Jump if Carry (= jb)                    │ │
│    │  jnc  label           ; Jump if No Carry (= jae)                │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Unconditional Jump and Call:                                         │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  jmp  label           ; Unconditional jump                      │ │
│    │  jmp  eax             ; Jump to address in register             │ │
│    │  jmp  [eax]           ; Jump to address stored in memory        │ │
│    │                                                                 │ │
│    │  call function        ; Call function (push return addr, jump)  │ │
│    │  ret                  ; Return (pop return address from stack)  │ │
│    │  ret  8               ; Return + clean up stack (pop 8 bytes)   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 ARM Branch Instructions

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ARM Branch Instructions                           │
│                                                                         │
│    Compare Instructions:                                                │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  CMP  X0, X1          ; X0 - X1, set flags only                 │ │
│    │  CMP  X0, #100        ; X0 - 100                                │ │
│    │  CMN  X0, X1          ; X0 + X1, set flags only (Compare Neg.)  │ │
│    │  TST  X0, X1          ; X0 & X1, flags only                     │ │
│    │  TEQ  X0, X1          ; X0 ^ X1, flags only (test equality)     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Conditional Branch (AArch64):                                        │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  B.EQ  label          ; Branch if Equal (Z=1)                   │ │
│    │  B.NE  label          ; Branch if Not Equal (Z=0)               │ │
│    │  B.GT  label          ; Branch if Greater Than (signed)         │ │
│    │  B.GE  label          ; Branch if Greater or Equal              │ │
│    │  B.LT  label          ; Branch if Less Than                     │ │
│    │  B.LE  label          ; Branch if Less or Equal                 │ │
│    │  B.HI  label          ; Branch if Higher (unsigned >)           │ │
│    │  B.HS  label          ; Branch if Higher or Same (unsigned >=)  │ │
│    │  B.LO  label          ; Branch if Lower (unsigned <)            │ │
│    │  B.LS  label          ; Branch if Lower or Same (unsigned <=)   │ │
│    │  B.MI  label          ; Branch if Minus (N=1)                   │ │
│    │  B.PL  label          ; Branch if Plus (N=0)                    │ │
│    │  B.VS  label          ; Branch if Overflow Set                  │ │
│    │  B.VC  label          ; Branch if Overflow Clear                │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Unconditional Branch and Call:                                       │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  B    label           ; Unconditional branch                    │ │
│    │  BR   X0              ; Branch to register                      │ │
│    │                                                                 │ │
│    │  BL   function        ; Branch with Link                        │ │
│    │                       ; X30(LR) = return addr, PC = function    │ │
│    │  BLR  X0              ; Branch with Link to Register            │ │
│    │                                                                 │ │
│    │  RET                  ; Return (= BR X30)                       │ │
│    │  RET  X1              ; Return (to address in X1)               │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Conditional Select (without branching):                              │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Conditional select (no branch penalty)                       │ │
│    │  CSEL  X0, X1, X2, EQ   ; X0 = (EQ) ? X1 : X2                   │ │
│    │  CSINC X0, X1, X2, NE   ; X0 = (NE) ? X1 : X2+1                 │ │
│    │  CSINV X0, X1, X2, LT   ; X0 = (LT) ? X1 : ~X2                  │ │
│    │  CSNEG X0, X1, X2, GE   ; X0 = (GE) ? X1 : -X2                  │ │
│    │                                                                 │ │
│    │  ; Conditional compare                                          │ │
│    │  CCMP  X0, X1, #0, EQ   ; if (EQ) then CMP X0,X1 else flags=0   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Branch Examples

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     if-else Implementation Comparison                    │
│                                                                         │
│    C Code:                                                              │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  if (a > b) {                                                   │ │
│    │      c = a;                                                     │ │
│    │  } else {                                                       │ │
│    │      c = b;                                                     │ │
│    │  }                                                              │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    x86 Assembly:                                                        │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │      mov  eax, [a]        ; eax = a                             │ │
│    │      mov  ebx, [b]        ; ebx = b                             │ │
│    │      cmp  eax, ebx        ; Compare a and b                     │ │
│    │      jle  else_part       ; If a <= b goto else                 │ │
│    │      mov  [c], eax        ; c = a                               │ │
│    │      jmp  end_if                                                │ │
│    │  else_part:                                                     │ │
│    │      mov  [c], ebx        ; c = b                               │ │
│    │  end_if:                                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ARM Assembly (AArch64):                                              │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │      LDR  W0, [X8]        ; W0 = a                              │ │
│    │      LDR  W1, [X9]        ; W1 = b                              │ │
│    │      CMP  W0, W1          ; Compare a and b                     │ │
│    │      CSEL W2, W0, W1, GT  ; W2 = (a>b) ? a : b                  │ │
│    │      STR  W2, [X10]       ; c = W2                              │ │
│    │                                                                 │ │
│    │  ; Or using conditional branch:                                 │ │
│    │      LDR  W0, [X8]                                              │ │
│    │      LDR  W1, [X9]                                              │ │
│    │      CMP  W0, W1                                                │ │
│    │      B.LE else_part                                             │ │
│    │      STR  W0, [X10]       ; c = a                               │ │
│    │      B    end_if                                                │ │
│    │  else_part:                                                     │ │
│    │      STR  W1, [X10]       ; c = b                               │ │
│    │  end_if:                                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Memory Access Instructions

### 6.1 x86 Memory Access

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      x86 Memory Access                                   │
│                                                                         │
│    Addressing Format:                                                   │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  [base + index * scale + displacement]                          │ │
│    │                                                                 │ │
│    │  - base: register (optional)                                    │ │
│    │  - index: register (optional)                                   │ │
│    │  - scale: 1, 2, 4, 8 (multiplier)                               │ │
│    │  - displacement: constant (optional)                            │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Memory Access Examples:                                              │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Direct address                                               │ │
│    │  mov  eax, [0x1000]           ; eax = Memory[0x1000]            │ │
│    │                                                                 │ │
│    │  ; Register indirect                                            │ │
│    │  mov  eax, [ebx]              ; eax = Memory[ebx]               │ │
│    │                                                                 │ │
│    │  ; Displacement                                                 │ │
│    │  mov  eax, [ebx + 8]          ; eax = Memory[ebx + 8]           │ │
│    │                                                                 │ │
│    │  ; Index                                                        │ │
│    │  mov  eax, [ebx + ecx]        ; eax = Memory[ebx + ecx]         │ │
│    │                                                                 │ │
│    │  ; Scaled index (useful for arrays)                             │ │
│    │  mov  eax, [ebx + ecx*4]      ; array[i] (int array)            │ │
│    │                                                                 │ │
│    │  ; Full form                                                    │ │
│    │  mov  eax, [ebx + ecx*4 + 100] ; struct.array[i]                │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Size Specification:                                                  │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  mov  byte  [eax], 0x41       ; Store 1 byte                    │ │
│    │  mov  word  [eax], 0x4142     ; Store 2 bytes                   │ │
│    │  mov  dword [eax], 0x41424344 ; Store 4 bytes                   │ │
│    │  mov  qword [rax], rbx        ; Store 8 bytes (64-bit)          │ │
│    │                                                                 │ │
│    │  ; Size-extending loads                                         │ │
│    │  movzx eax, byte [ebx]        ; Zero extend (8→32 bit)          │ │
│    │  movsx eax, byte [ebx]        ; Sign extend                     │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    String Operations:                                                   │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; ESI = source, EDI = destination, ECX = counter               │ │
│    │  movsb                ; [EDI] = [ESI], ESI++, EDI++             │ │
│    │  rep movsb            ; Repeat ECX times (memcpy)               │ │
│    │  cmpsb                ; Compare [ESI] and [EDI]                 │ │
│    │  scasb                ; Compare AL and [EDI]                    │ │
│    │  stosb                ; [EDI] = AL (memset)                     │ │
│    │  lodsb                ; AL = [ESI]                              │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 ARM Memory Access

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      ARM Memory Access                                   │
│                                                                         │
│    Basic Load/Store:                                                    │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Load                                                         │ │
│    │  LDR   X0, [X1]           ; X0 = Memory[X1] (64-bit)            │ │
│    │  LDR   W0, [X1]           ; W0 = Memory[X1] (32-bit)            │ │
│    │  LDRH  W0, [X1]           ; W0 = Memory[X1] (16-bit, zero ext)  │ │
│    │  LDRB  W0, [X1]           ; W0 = Memory[X1] (8-bit, zero ext)   │ │
│    │  LDRSH W0, [X1]           ; Sign-extended halfword              │ │
│    │  LDRSB W0, [X1]           ; Sign-extended byte                  │ │
│    │                                                                 │ │
│    │  ; Store                                                        │ │
│    │  STR   X0, [X1]           ; Memory[X1] = X0                     │ │
│    │  STRH  W0, [X1]           ; Memory[X1] = lower 16 bits          │ │
│    │  STRB  W0, [X1]           ; Memory[X1] = lower 8 bits           │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Addressing Modes:                                                    │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Offset                                                       │ │
│    │  LDR  X0, [X1, #8]        ; X0 = Memory[X1 + 8]                 │ │
│    │  LDR  X0, [X1, #-8]       ; X0 = Memory[X1 - 8]                 │ │
│    │                                                                 │ │
│    │  ; Register offset                                              │ │
│    │  LDR  X0, [X1, X2]        ; X0 = Memory[X1 + X2]                │ │
│    │  LDR  X0, [X1, X2, LSL #3] ; X0 = Memory[X1 + X2*8]             │ │
│    │                                                                 │ │
│    │  ; Pre-indexed (update base after calculation)                  │ │
│    │  LDR  X0, [X1, #8]!       ; X1 += 8; X0 = Memory[X1]            │ │
│    │                                                                 │ │
│    │  ; Post-indexed (update base after access)                      │ │
│    │  LDR  X0, [X1], #8        ; X0 = Memory[X1]; X1 += 8            │ │
│    │                                                                 │ │
│    │  ; Literal (PC-relative, useful for loading constants)          │ │
│    │  LDR  X0, =0x12345678     ; X0 = 0x12345678 (using literal pool)│ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Multiple Register Load/Store:                                        │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; Register pair                                                │ │
│    │  LDP  X0, X1, [X2]        ; X0=Mem[X2], X1=Mem[X2+8]            │ │
│    │  STP  X0, X1, [X2]        ; Mem[X2]=X0, Mem[X2+8]=X1            │ │
│    │                                                                 │ │
│    │  ; Stack frame setup (common pattern)                           │ │
│    │  STP  X29, X30, [SP, #-16]!  ; Save frame/link registers        │ │
│    │  MOV  X29, SP                ; New frame pointer                │ │
│    │  ...                                                            │ │
│    │  LDP  X29, X30, [SP], #16    ; Restore                          │ │
│    │  RET                                                            │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Simple Assembly Program Examples

### 7.1 Hello World (x86-64 Linux)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Hello World (x86-64 Linux, NASM)                      │
│                                                                         │
│    ; hello.asm                                                          │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  section .data                                                  │ │
│    │      msg     db  "Hello, World!", 10    ; String + newline      │ │
│    │      len     equ $ - msg                 ; String length        │ │
│    │                                                                 │ │
│    │  section .text                                                  │ │
│    │      global _start                                              │ │
│    │                                                                 │ │
│    │  _start:                                                        │ │
│    │      ; write(1, msg, len)                                       │ │
│    │      mov     rax, 1          ; syscall: write                   │ │
│    │      mov     rdi, 1          ; fd: stdout                       │ │
│    │      mov     rsi, msg        ; buffer address                   │ │
│    │      mov     rdx, len        ; length                           │ │
│    │      syscall                                                    │ │
│    │                                                                 │ │
│    │      ; exit(0)                                                  │ │
│    │      mov     rax, 60         ; syscall: exit                    │ │
│    │      xor     rdi, rdi        ; exit code: 0                     │ │
│    │      syscall                                                    │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Build and run:                                                       │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  $ nasm -f elf64 hello.asm -o hello.o                           │ │
│    │  $ ld hello.o -o hello                                          │ │
│    │  $ ./hello                                                      │ │
│    │  Hello, World!                                                  │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Sum of Two Numbers (x86-64)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Sum of Two Numbers (x86-64)                            │
│                                                                         │
│    ; sum.asm - Function callable from C                                 │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  ; int add(int a, int b);                                       │ │
│    │  ; x86-64 calling convention: args in RDI, RSI, RDX, RCX, R8, R9│ │
│    │  ;                            return value in RAX               │ │
│    │                                                                 │ │
│    │  section .text                                                  │ │
│    │      global add                                                 │ │
│    │                                                                 │ │
│    │  add:                                                           │ │
│    │      ; Prologue (optional for simple functions)                 │ │
│    │      mov     eax, edi        ; eax = first argument (a)         │ │
│    │      add     eax, esi        ; eax += second argument (b)       │ │
│    │      ret                     ; result in eax                    │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Calling from C:                                                      │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  // main.c                                                      │ │
│    │  extern int add(int a, int b);                                  │ │
│    │                                                                 │ │
│    │  int main() {                                                   │ │
│    │      int result = add(10, 20);                                  │ │
│    │      printf("10 + 20 = %d\n", result);                          │ │
│    │      return 0;                                                  │ │
│    │  }                                                              │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Build:                                                               │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  $ nasm -f elf64 sum.asm -o sum.o                               │ │
│    │  $ gcc main.c sum.o -o program                                  │ │
│    │  $ ./program                                                    │ │
│    │  10 + 20 = 30                                                   │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Loop (ARM AArch64)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Loop (ARM AArch64)                               │
│                                                                         │
│    C Code:                                                              │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  // Calculate array sum                                         │ │
│    │  int sum_array(int *arr, int n) {                               │ │
│    │      int sum = 0;                                               │ │
│    │      for (int i = 0; i < n; i++) {                              │ │
│    │          sum += arr[i];                                         │ │
│    │      }                                                          │ │
│    │      return sum;                                                │ │
│    │  }                                                              │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ARM Assembly:                                                        │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  // X0 = arr (array pointer), W1 = n (array size)               │ │
│    │  // Returns: W0 = sum                                           │ │
│    │                                                                 │ │
│    │  sum_array:                                                     │ │
│    │      MOV     W2, #0          ; sum = 0                          │ │
│    │      MOV     W3, #0          ; i = 0                            │ │
│    │      CMP     W1, #0          ; Check n == 0                     │ │
│    │      B.LE    done            ; If n <= 0, exit                  │ │
│    │                                                                 │ │
│    │  loop:                                                          │ │
│    │      LDR     W4, [X0, W3, SXTW #2]  ; W4 = arr[i]               │ │
│    │                                     ; (i * 4 as offset)         │ │
│    │      ADD     W2, W2, W4      ; sum += arr[i]                    │ │
│    │      ADD     W3, W3, #1      ; i++                              │ │
│    │      CMP     W3, W1          ; i < n ?                          │ │
│    │      B.LT    loop            ; If true, repeat                  │ │
│    │                                                                 │ │
│    │  done:                                                          │ │
│    │      MOV     W0, W2          ; return value = sum               │ │
│    │      RET                                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Optimized version (using post-increment):                            │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  sum_array_opt:                                                 │ │
│    │      MOV     W2, #0          ; sum = 0                          │ │
│    │      CBZ     W1, done        ; If n == 0, exit immediately      │ │
│    │                                                                 │ │
│    │  loop:                                                          │ │
│    │      LDR     W3, [X0], #4    ; W3 = *arr++                      │ │
│    │      ADD     W2, W2, W3      ; sum += *arr                      │ │
│    │      SUBS    W1, W1, #1      ; n-- (set flags)                  │ │
│    │      B.NE    loop            ; If n != 0, repeat                │ │
│    │                                                                 │ │
│    │  done:                                                          │ │
│    │      MOV     W0, W2                                             │ │
│    │      RET                                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.4 Function Calls and Stack Frames

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Function Calls and Stack Frames                        │
│                                                                         │
│    x86-64 Stack Frame:                                                  │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │                                                                 │ │
│    │    High address                                                 │ │
│    │    ┌─────────────────────┐                                      │ │
│    │    │   Previous frame    │                                      │ │
│    │    ├─────────────────────┤                                      │ │
│    │    │  7th+ arguments     │  (passed via stack)                  │ │
│    │    ├─────────────────────┤                                      │ │
│    │    │   Return address    │  ◄─── Pushed by CALL                 │ │
│    │    ├─────────────────────┤                                      │ │
│    │    │    Saved RBP        │  ◄─── Previous frame pointer         │ │
│    │    ├─────────────────────┤ ◄─── RBP (current frame start)       │ │
│    │    │   Local var 1       │                                      │ │
│    │    ├─────────────────────┤                                      │ │
│    │    │   Local var 2       │                                      │ │
│    │    ├─────────────────────┤                                      │ │
│    │    │  Saved registers    │  (callee-saved)                      │ │
│    │    ├─────────────────────┤ ◄─── RSP (stack top)                 │ │
│    │    │                     │                                      │ │
│    │    Low address                                                  │ │
│    │                                                                 │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    Function Prologue/Epilogue (x86-64):                                 │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  my_function:                                                   │ │
│    │      ; Prologue                                                 │ │
│    │      push    rbp             ; Save previous frame pointer      │ │
│    │      mov     rbp, rsp        ; Set new frame                    │ │
│    │      sub     rsp, 32         ; Allocate local variable space    │ │
│    │      push    rbx             ; Save callee-saved register       │ │
│    │                                                                 │ │
│    │      ; Function body                                            │ │
│    │      ...                                                        │ │
│    │                                                                 │ │
│    │      ; Epilogue                                                 │ │
│    │      pop     rbx             ; Restore registers                │ │
│    │      mov     rsp, rbp        ; Clean up stack                   │ │
│    │      pop     rbp             ; Restore frame pointer            │ │
│    │      ret                                                        │ │
│    │                                                                 │ │
│    │  ; Or using leave instruction                                   │ │
│    │      leave                   ; mov rsp, rbp + pop rbp           │ │
│    │      ret                                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│    ARM AArch64 Function:                                                │
│    ┌─────────────────────────────────────────────────────────────────┐ │
│    │  my_function:                                                   │ │
│    │      ; Prologue                                                 │ │
│    │      STP     X29, X30, [SP, #-32]!  ; Save FP, LR               │ │
│    │      MOV     X29, SP                 ; Set new frame            │ │
│    │      STP     X19, X20, [SP, #16]     ; Save callee-saved        │ │
│    │                                                                 │ │
│    │      ; Function body                                            │ │
│    │      ...                                                        │ │
│    │                                                                 │ │
│    │      ; Epilogue                                                 │ │
│    │      LDP     X19, X20, [SP, #16]     ; Restore registers        │ │
│    │      LDP     X29, X30, [SP], #32     ; Restore FP, LR           │ │
│    │      RET                                                        │ │
│    └─────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Practice Problems

### Basic Problems

1. Explain the relationship between assembly language and machine code.

2. Explain the purpose of the following x86 registers:
   - (a) EAX
   - (b) ESP
   - (c) EIP/RIP

3. List 2 major differences between Intel syntax and AT&T syntax.

### x86 Assembly Problems

4. What is the value of EAX after executing the following x86 assembly code?
   ```asm
   mov  eax, 10
   mov  ebx, 5
   add  eax, ebx
   shl  eax, 1
   ```

5. Convert the following C code to x86-64 assembly:
   ```c
   int abs(int x) {
       if (x < 0)
           return -x;
       return x;
   }
   ```

### ARM Assembly Problems

6. Explain the advantages of conditional execution in ARM.

7. What is the value of X0 after executing the following ARM code?
   ```asm
   MOV  X0, #10
   MOV  X1, #3
   MUL  X0, X0, X1
   SUB  X0, X0, #5
   ```

### Memory Access Problems

8. Explain what each element means in the x86 addressing mode `[EBX + ECX*4 + 100]`.

9. Explain the difference between ARM's pre-indexed and post-indexed addressing with examples.

### Advanced Problems

10. Convert the following C code to x86-64 or ARM AArch64 assembly:
    ```c
    int factorial(int n) {
        int result = 1;
        for (int i = 2; i <= n; i++) {
            result *= i;
        }
        return result;
    }
    ```

<details>
<summary>Answers</summary>

1. Assembly language represents machine code (0s and 1s) using human-readable mnemonics (ADD, MOV, etc.). The assembler performs a 1:1 translation of assembly code to machine code.

2. Register purposes:
   - (a) EAX: Accumulator, stores arithmetic results, function return value
   - (b) ESP: Stack Pointer, points to top of stack
   - (c) EIP/RIP: Instruction Pointer, address of next instruction to execute

3. Intel vs AT&T:
   - Operand order: Intel uses (dst, src), AT&T uses (src, dst)
   - Prefixes: Intel has none, AT&T uses % for registers, $ for immediates

4. Code execution result:
   - mov eax, 10 → eax = 10
   - add eax, ebx → eax = 15
   - shl eax, 1 → eax = 30

5. abs function (x86-64):
   ```asm
   abs:
       mov  eax, edi       ; eax = x
       test eax, eax       ; Check sign of x
       jns  positive       ; If x >= 0, jump
       neg  eax            ; x = -x
   positive:
       ret
   ```

6. ARM conditional execution advantages:
   - Avoids pipeline flush for short branches
   - Reduces code size
   - No branch prediction failure penalty

7. ARM code result:
   - MOV X0, #10 → X0 = 10
   - MUL X0, X0, X1 → X0 = 30
   - SUB X0, X0, #5 → X0 = 25

8. x86 addressing elements:
   - EBX: Base register (array start address)
   - ECX: Index register (array index)
   - 4: Scale (element size, int = 4 bytes)
   - 100: Displacement (offset within struct)

9. ARM indexing:
   - Pre-indexed `LDR X0, [X1, #8]!`: X1 += 8 then load
   - Post-indexed `LDR X0, [X1], #8`: Load then X1 += 8

10. factorial (x86-64):
    ```asm
    factorial:
        mov  eax, 1         ; result = 1
        cmp  edi, 1
        jle  done           ; If n <= 1, exit
        mov  ecx, 2         ; i = 2
    loop:
        imul eax, ecx       ; result *= i
        inc  ecx            ; i++
        cmp  ecx, edi
        jle  loop           ; If i <= n, repeat
    done:
        ret
    ```

</details>

---

## Next Steps

- [11_Pipelining.md](./11_Pipelining.md) - Pipeline stages, hazards, forwarding

---

## References

- [x86 Instruction Set Reference](https://www.felixcloutier.com/x86/)
- [ARM Developer Documentation](https://developer.arm.com/documentation)
- [NASM Documentation](https://www.nasm.us/doc/)
- [Godbolt Compiler Explorer](https://godbolt.org/) - View assembly for various architectures
- Computer Organization and Design (Patterson & Hennessy)
