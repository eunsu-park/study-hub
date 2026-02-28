# RISC-V 아키텍처

**이전**: [병렬 처리와 멀티코어](./18_Parallel_Processing_Multicore.md) | **다음**: [전력 관리](./20_Power_Management.md)

**난이도**: ⭐⭐⭐⭐

**선수 과목**: 명령어 집합 아키텍처(Instruction Set Architecture) (9강), 어셈블리 언어 기초(Assembly Language Basics) (10강)

---

## 학습 목표

이 강의를 마치면 다음을 할 수 있습니다:

1. RISC-V의 설계 철학과 개방형 ISA 모델을 채택한 이유를 설명할 수 있다
2. RV32I 기본 정수 명령어 집합과 레지스터 파일을 기술할 수 있다
3. 표준 확장(M, A, F, D, C, V)의 차이와 각각의 목적을 구분할 수 있다
4. RISC-V의 주소 지정 방식, 명령어 인코딩, 호출 규약을 x86 및 ARM과 비교할 수 있다
5. RISC-V 어셈블리 코드를 분석하고 파이프라인을 통한 명령어 실행 흐름을 추적할 수 있다

---

대부분의 명령어 집합 아키텍처(ISA)는 독점적입니다. x86은 Intel/AMD 소유이고, ARM은 Arm Holdings로부터 라이선스를 취득해야 합니다. RISC-V는 다릅니다. RISC-V는 2010년 UC 버클리에서 설계된 자유롭고 개방된 ISA로, 학문적 프로젝트에서 시작하여 임베디드 시스템, 데이터 센터, 심지어 슈퍼컴퓨터까지 진지하게 경쟁하는 아키텍처로 성장했습니다. RISC-V를 이해하면 수십 년간의 하위 호환성 부담 없이 처음부터 설계된 현대적인 ISA가 어떤 모습인지 알 수 있습니다.

---

## 목차

1. [역사와 설계 철학](#1-역사와-설계-철학)
2. [ISA 개요](#2-isa-개요)
3. [RV32I 기본 정수 명령어 집합](#3-rv32i-기본-정수-명령어-집합)
4. [명령어 인코딩 형식](#4-명령어-인코딩-형식)
5. [표준 확장](#5-표준-확장)
6. [권한 수준과 CSR](#6-권한-수준과-csr)
7. [RISC-V vs x86 vs ARM](#7-risc-v-vs-x86-vs-arm)
8. [RISC-V 생태계](#8-risc-v-생태계)
9. [연습 문제](#9-연습-문제)

---

## 1. 역사와 설계 철학

### 1.1 탄생 배경

RISC-V("risk-five"로 발음)는 2010년 크르스테 아사노비치(Krste Asanović), 데이비드 패터슨(David Patterson, 유명한 Patterson & Hennessy 교재의 공저자), 그리고 UC 버클리의 학생들에 의해 탄생했습니다. "V"는 버클리에서 설계된 다섯 번째 세대의 RISC 아키텍처(RISC-I, RISC-II, SOAR, SPUR 이후)를 의미합니다.

### 1.2 설계 목표

| 목표 | RISC-V의 달성 방식 |
|------|----------------------|
| **개방성과 자유** | ISA 명세가 오픈소스이며, 누구나 라이선스 비용 없이 구현 가능 |
| **모듈화** | 소형 기본 ISA + 선택적 확장; 불필요한 부담 없음 |
| **간결함** | 40년간의 하위 호환성으로 인한 레거시 명령어 없음 |
| **확장성** | 마이크로컨트롤러(RV32E)부터 슈퍼컴퓨터(RV64GC + V)까지 동일한 ISA |
| **안정성** | 기본 ISA는 동결 상태; 확장은 비준 절차를 거침 |

### 1.3 RISC-V 재단 → RISC-V International

RISC-V 재단(현재 지정학적 중립성을 위해 스위스에 기반을 둔 RISC-V International)이 명세를 관리합니다. Google, NVIDIA, Qualcomm, Samsung, Western Digital을 포함하여 3,000개 이상의 회원 기관이 참여하고 있습니다.

---

## 2. ISA 개요

### 2.1 기본 ISA

RISC-V는 여러 기본 정수 ISA를 정의합니다:

| 기본 ISA | 워드 크기 | 레지스터 | 사용 사례 |
|----------|-----------|-----------|----------|
| **RV32I** | 32비트 | 32 × 32비트 | 임베디드, 마이크로컨트롤러 |
| **RV32E** | 32비트 | 16 × 32비트 | 초저전력 임베디드 |
| **RV64I** | 64비트 | 32 × 64비트 | 응용 프로세서, 서버 |
| **RV128I** | 128비트 | 32 × 128비트 | 미래 (초안 명세) |

### 2.2 레지스터 파일

RV32I/RV64I는 32개의 범용 정수 레지스터를 갖습니다:

```
┌──────────┬──────────┬────────────────────────┐
│ Register │ ABI Name │ Description            │
├──────────┼──────────┼────────────────────────┤
│ x0       │ zero     │ Hardwired to 0         │
│ x1       │ ra       │ Return address         │
│ x2       │ sp       │ Stack pointer          │
│ x3       │ gp       │ Global pointer         │
│ x4       │ tp       │ Thread pointer         │
│ x5-x7    │ t0-t2    │ Temporaries            │
│ x8       │ s0/fp    │ Saved register / Frame │
│ x9       │ s1       │ Saved register         │
│ x10-x11  │ a0-a1    │ Arguments / Return val │
│ x12-x17  │ a2-a7    │ Arguments              │
│ x18-x27  │ s2-s11   │ Saved registers        │
│ x28-x31  │ t3-t6    │ Temporaries            │
└──────────┴──────────┴────────────────────────┘
```

핵심 설계 선택: `x0`은 항상 0으로 고정(hardwired)되어 있습니다. 이 덕분에 별도의 "즉시 0 로드" 명령어가 필요 없습니다. 예를 들어, `add x5, x0, x0`은 x5를 0으로 초기화합니다.

### 2.3 프로그램 카운터(Program Counter)

PC는 ARM과 달리 범용 레지스터 파일의 일부가 아닙니다. 분기/점프 명령어에 의해서만 암묵적으로 변경됩니다.

---

## 3. RV32I 기본 정수 명령어 집합

RV32I는 단 **47개의 명령어**만 가집니다. x86(수천 개) 또는 ARM(수백 개)과 비교하면 놀라울 정도로 작습니다.

### 3.1 명령어 분류

| 분류 | 명령어 수 | 예시 |
|----------|-------------|---------|
| **산술** | 10 | `ADD`, `SUB`, `ADDI`, `SLT`, `SLTI` |
| **논리** | 6 | `AND`, `OR`, `XOR`, `ANDI`, `ORI`, `XORI` |
| **시프트** | 6 | `SLL`, `SRL`, `SRA`, `SLLI`, `SRLI`, `SRAI` |
| **로드** | 5 | `LB`, `LH`, `LW`, `LBU`, `LHU` |
| **스토어** | 3 | `SB`, `SH`, `SW` |
| **분기** | 6 | `BEQ`, `BNE`, `BLT`, `BGE`, `BLTU`, `BGEU` |
| **점프** | 2 | `JAL`, `JALR` |
| **상위 즉시값** | 2 | `LUI`, `AUIPC` |
| **시스템** | 7 | `ECALL`, `EBREAK`, `FENCE`, CSR 명령어 |

### 3.2 어셈블리 예시

```asm
# --- 기본 산술 ---
addi  x5, x0, 42       # x5 = 0 + 42 = 42
addi  x6, x0, 17       # x6 = 17
add   x7, x5, x6       # x7 = 42 + 17 = 59
sub   x8, x5, x6       # x8 = 42 - 17 = 25

# --- 메모리 접근 ---
sw    x7, 0(x2)        # Store x7 to address [sp + 0]
lw    x9, 0(x2)        # Load word from [sp + 0] into x9

# --- 조건부 분기 ---
beq   x5, x6, equal    # If x5 == x6, jump to 'equal'
blt   x5, x6, less     # If x5 < x6 (signed), jump to 'less'

# --- 함수 호출 ---
jal   ra, my_function  # Jump to my_function, save return addr in ra
# ... my_function returns with:
jalr  x0, ra, 0        # Jump to address in ra (return)
```

### 3.3 기본 ISA에 곱셈 없음

x86/ARM과 달리, 기본 RV32I에는 곱셈 또는 나눗셈 명령어가 없습니다. 이들은 **M 확장**에 포함되어 있습니다. 이는 하드웨어 곱셈이 필요 없을 수 있는 가장 단순한 마이크로컨트롤러를 위해 기본 ISA를 최소화하기 위한 설계입니다.

### 3.4 의사 명령어(Pseudo-Instructions)

어셈블러는 자주 쓰이는 패턴을 인식합니다:

| 의사 명령어 | 실제 명령어 | 의미 |
|--------------------|--------------------|---------|
| `li x5, 100` | `addi x5, x0, 100` | 즉시값 로드 |
| `mv x5, x6` | `addi x5, x6, 0` | 레지스터 복사 |
| `nop` | `addi x0, x0, 0` | 아무 동작 없음 |
| `ret` | `jalr x0, ra, 0` | 함수에서 반환 |
| `j label` | `jal x0, label` | 무조건 점프 |
| `beqz x5, label` | `beq x5, x0, label` | 0이면 분기 |
| `not x5, x6` | `xori x5, x6, -1` | 비트 NOT |
| `neg x5, x6` | `sub x5, x0, x6` | 부호 반전 |

---

## 4. 명령어 인코딩 형식

기본 ISA의 모든 RISC-V 명령어는 32비트(고정 길이)입니다. 여섯 가지 인코딩 형식이 있습니다:

```
R-type:  [funct7 | rs2 | rs1 | funct3 | rd  | opcode]   (레지스터-레지스터)
I-type:  [   imm[11:0]  | rs1 | funct3 | rd  | opcode]   (레지스터-즉시값)
S-type:  [imm[11:5]|rs2 | rs1 | funct3 |imm[4:0]|opcode] (스토어)
B-type:  [imm bits | rs2| rs1 | funct3 |imm bits|opcode] (분기)
U-type:  [      imm[31:12]             | rd  | opcode]   (상위 즉시값)
J-type:  [      imm bits               | rd  | opcode]   (점프)
```

### 4.1 R-type 예시: ADD

```
ADD x7, x5, x6
┌─────────┬──────┬──────┬────────┬──────┬─────────┐
│ 0000000 │ 00110│ 00101│  000   │ 00111│ 0110011 │
│ funct7  │ rs2  │ rs1  │ funct3 │  rd  │ opcode  │
└─────────┴──────┴──────┴────────┴──────┴─────────┘
  7 bits    5 bits 5 bits 3 bits  5 bits  7 bits = 32 bits total
```

### 4.2 설계 근거

- **소스 레지스터 위치 고정**: `rs1`과 `rs2`는 모든 형식에서 항상 동일한 비트 위치에 있습니다. 이를 통해 명령어가 완전히 디코딩되기 전에 레지스터 파일 읽기를 시작할 수 있습니다. 파이프라인 최적화에 해당합니다.
- **부호 확장(Sign extension)**: 즉시값은 항상 명령어의 최상위 비트(비트 31)에서 부호 확장됩니다. 하드웨어의 부호 확장 로직이 매우 단순해집니다.

---

## 5. 표준 확장

RISC-V의 모듈성은 확장 시스템에서 비롯됩니다. 확장은 단일 문자로 식별됩니다:

| 확장 | 이름 | 설명 |
|-----------|------|-------------|
| **M** | 정수 곱셈/나눗셈 | `MUL`, `MULH`, `DIV`, `REM` 및 부호 없는 변형 |
| **A** | 원자적 연산(Atomic) | `LR.W/D` (로드 예약), `SC.W/D` (조건부 스토어), 원자적 swap/add/and/or |
| **F** | 단정밀도 부동소수점(Single-Precision Float) | 32개의 IEEE 754 부동소수점 레지스터 (f0-f31), `FADD.S`, `FMUL.S` 등 |
| **D** | 배정밀도 부동소수점(Double-Precision Float) | F 레지스터를 64비트로 확장, `FADD.D`, `FMUL.D` 등 |
| **C** | 압축(Compressed) | 가장 빈번한 명령어를 16비트 인코딩으로 표현 (코드 크기 약 25-30% 감소) |
| **V** | 벡터(Vector) | 확장 가능한 벡터 처리 (AVX/NEON처럼 고정이 아닌 가변 VLEN) |

### 5.1 "G" 약어

**RV32G** = RV32I + M + A + F + D ("범용(general-purpose)" 조합). 대부분의 응용 프로세서는 최소 RV64GC(압축 명령어를 포함한 64비트 범용)를 구현합니다.

### 5.2 C 확장: 압축 명령어

C 확장은 자주 사용되는 명령어에 16비트 인코딩을 추가합니다. 프로세서는 동일한 스트림에서 16비트와 32비트 명령어를 혼합하여 처리합니다:

```asm
# 32비트 명령어 (4바이트)
add   x10, x11, x12     # Full R-type

# 16비트 압축 명령어 (2바이트)
c.add x10, x12          # Same operation, half the size
```

압축 명령어는 코드 크기를 25-30% 줄입니다. ARM의 Thumb-2와 유사한 수준입니다. 절충점: 더 작은 즉시값 범위와 일부 압축 형식에서는 적은 수의 레지스터(x8-x15만 접근 가능)를 사용해야 합니다.

### 5.3 V 확장: 벡터 처리

x86(고정 128/256/512비트 벡터의 SSE/AVX)이나 ARM(고정 128비트의 NEON)과 달리, RISC-V V는 **확장 가능한 벡터(scalable vectors)**를 사용합니다:

```asm
# Set vector length for 32-bit elements
vsetvli t0, a0, e32, m1    # Process up to a0 elements, 32-bit each

# Vector load, add, store
vle32.v v1, (a1)            # Load vector from memory at a1
vle32.v v2, (a2)            # Load vector from memory at a2
vadd.vv v3, v1, v2          # v3 = v1 + v2 (element-wise)
vse32.v v3, (a3)            # Store result to memory at a3
```

하드웨어는 물리적 벡터 레지스터 길이(VLEN)를 기반으로 한 번의 연산에서 처리할 수 있는 원소 수를 결정합니다. VLEN=128 또는 VLEN=1024인 구현에서도 재컴파일 없이 동일한 바이너리가 실행됩니다.

---

## 6. 권한 수준과 CSR

RISC-V는 시스템 소프트웨어를 위해 세 가지 권한 수준을 정의합니다:

```
┌─────────────────────────────────────────┐
│ Level 3: Machine Mode (M-mode)          │ ← 펌웨어 / 부트로더
│   - Highest privilege, always present   │
│   - Direct hardware access              │
├─────────────────────────────────────────┤
│ Level 1: Supervisor Mode (S-mode)       │ ← 운영체제 커널
│   - Virtual memory (Sv39, Sv48)         │
│   - Trap handling                       │
├─────────────────────────────────────────┤
│ Level 0: User Mode (U-mode)             │ ← 응용 프로그램
│   - Restricted access                   │
│   - Uses ecall for system calls         │
└─────────────────────────────────────────┘
```

### 6.1 제어 및 상태 레지스터(Control and Status Registers, CSR)

CSR은 기계 상태를 제어합니다:

| CSR | 목적 |
|-----|---------|
| `mstatus` | 기계 상태 (인터럽트 활성화, 권한 모드) |
| `mtvec` | 기계 트랩 벡터 기본 주소 |
| `mepc` | 기계 예외 프로그램 카운터 |
| `mcause` | 기계 트랩 원인 |
| `mhartid` | 하드웨어 스레드(hart) ID |
| `satp` | 슈퍼바이저 주소 변환 및 보호 |
| `cycle` | 사이클 카운터 (U-mode에서 읽기 전용) |

```asm
# Read cycle counter
csrr  t0, cycle          # t0 = current cycle count

# Enable interrupts
csrsi mstatus, 0x8       # Set MIE bit in mstatus
```

---

## 7. RISC-V vs x86 vs ARM

### 7.1 아키텍처 비교

| 특성 | RISC-V | x86-64 | ARMv8 (AArch64) |
|---------|--------|--------|-----------------|
| **ISA 유형** | RISC, 개방형 | CISC, 독점 | RISC, 독점 |
| **라이선스** | 무료 | Intel/AMD 전용 | 로열티 기반 |
| **명령어 길이** | 32비트 (C 확장 시 16비트) | 가변 (1-15바이트) | 고정 32비트 |
| **범용 레지스터** | 32개 (x0 = 0 고정) | 16개 | 31개 (x0-x30) + xzr |
| **주소 지정 방식** | 베이스 + 오프셋만 | 다양 (복잡) | 베이스 + 오프셋, 일부 복잡 |
| **조건 코드(Condition Codes)** | 없음 (비교-분기 일체) | FLAGS 레지스터 | FLAGS (NZCV) 레지스터 |
| **기본 ISA 곱셈** | 없음 (M 확장) | 있음 | 있음 |
| **SIMD/벡터** | V 확장 (확장 가능) | SSE/AVX (고정 폭) | NEON (128비트) / SVE (확장 가능) |
| **엔디언** | 리틀(기본) | 리틀 | 바이-엔디언(Bi-endian) |
| **권한 수준** | 3단계 (M/S/U) | 4단계 (Ring 0-3) | 4단계 (EL0-EL3) |

### 7.2 조건 플래그 없음

RISC-V는 x86 및 ARM에서 볼 수 있는 FLAGS/조건 코드 레지스터를 없앴습니다. 대신, 분기 명령어가 두 레지스터를 직접 비교합니다:

```asm
# x86: compare then branch (two instructions, implicit state)
cmp   eax, ebx
je    target

# ARM: compare then branch (two instructions, modifies NZCV)
cmp   w0, w1
b.eq  target

# RISC-V: fused compare-and-branch (one instruction, no side effects)
beq   x10, x11, target
```

분기가 플래그 레지스터에 대한 숨겨진 의존성을 갖지 않으므로, 비순서 실행(out-of-order execution)이 단순해집니다.

### 7.3 코드 크기 비교

동일한 "Hello, World" 프로그램에 대해:

| ISA | 바이너리 크기 | 명령어 수 |
|-----|------------|-------------------|
| RV64GC | ~4.2 KB | 15개 |
| AArch64 | ~4.0 KB | 14개 |
| x86-64 | ~4.5 KB | 12개 (가변 길이) |

C(압축) 확장을 사용하면 RISC-V는 ARM Thumb-2에 필적하고 고정 32비트 ARM보다 우수한 코드 밀도를 달성합니다.

---

## 8. RISC-V 생태계

### 8.1 하드웨어

| 분류 | 예시 |
|----------|---------|
| **MCU** | ESP32-C3 (WiFi+BLE), GD32V, CH32V |
| **응용 프로세서** | SiFive U74 (Linux 지원), StarFive JH7110 |
| **HPC** | Sophon SG2042 (64코어), Ventana Veyron |
| **개발 보드** | HiFive Unmatched, VisionFive 2, Milk-V Pioneer |

### 8.2 소프트웨어

- **컴파일러**: GCC 13+, LLVM/Clang 17+ (V 확장 포함 완전 지원)
- **OS**: Linux (4.15부터 메인라인 커널 지원), FreeBSD, Zephyr RTOS, FreeRTOS
- **시뮬레이터**: QEMU, Spike (레퍼런스), gem5
- **툴체인**: `riscv64-unknown-elf-gcc` (베어메탈), `riscv64-linux-gnu-gcc` (Linux)

### 8.3 QEMU를 이용한 시뮬레이션

```bash
# Install RISC-V QEMU (Ubuntu/Debian)
sudo apt install qemu-system-riscv64

# Compile bare-metal RISC-V program
riscv64-unknown-elf-gcc -march=rv64gc -o hello hello.c -nostdlib

# Run in QEMU
qemu-system-riscv64 -machine virt -nographic -bios none -kernel hello
```

---

## 9. 연습 문제

### 문제 1: 명령어 인코딩

다음 명령어를 32비트 이진값으로 인코딩하십시오. 형식 유형(R/I/S/B/U/J)을 식별하고 각 필드를 채우십시오:

```asm
addi x10, x5, -3
```

### 문제 2: 어셈블리 추적

각 명령어 실행 후 레지스터 값을 추적하십시오:

```asm
    addi  x5, x0, 10      # x5 = ?
    addi  x6, x0, 3       # x6 = ?
    add   x7, x5, x6      # x7 = ?
    sub   x8, x5, x6      # x8 = ?
    slli  x9, x6, 2       # x9 = ? (shift left logical by 2)
    and   x10, x5, x6     # x10 = ?
    or    x11, x5, x6     # x11 = ?
    slt   x12, x6, x5     # x12 = ? (set if x6 < x5)
```

### 문제 3: ISA 비교

함수 `int max(int a, int b)`를 다음 각각으로 작성하십시오:
1. RISC-V RV32I 어셈블리 (`bge` / `blt` 사용)
2. x86-64 어셈블리 (`cmp` / `cmovge` 사용)
3. ARMv8 AArch64 어셈블리 (`cmp` / `csel` 사용)

명령어 수를 비교하고, 어떤 ISA 기능(조건 코드, 조건부 이동, 서술)이 구현에 영향을 미치는지 파악하십시오.

### 문제 4: 확장 분석

RISC-V 프로세서에 "RV64IMAFDC"라는 레이블이 붙어 있습니다. 각 구성 요소를 나열하고 다음을 설명하십시오:
1. 각 문자의 의미
2. 각 확장이 추가하는 명령어
3. "RV64GC" 약어의 의미와 이것이 동등한 이유

### 문제 5: 파이프라인 해저드

5단계 파이프라인(IF-ID-EX-MEM-WB)에서 다음 RISC-V 코드를 고려하십시오:

```asm
    lw    x5, 0(x10)       # Load word
    add   x6, x5, x7      # Uses x5 immediately
    sub   x8, x5, x9      # Also uses x5
```

1. 데이터 해저드 유형(RAW/WAR/WAW)을 식별하십시오
2. 포워딩 없이 필요한 스톨(stall) 사이클 수는?
3. 포워딩(EX-EX 및 MEM-EX)이 있을 때 스톨 사이클 수는?
4. 컴파일러가 스톨을 제거할 수 있습니까? 가능하다면 어떻게?

---

*19강 끝*
