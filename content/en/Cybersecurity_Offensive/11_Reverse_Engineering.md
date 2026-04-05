# Reverse Engineering

**Previous**: [10. Buffer Overflow](./10_Buffer_Overflow.md) | **Next**: [12. Privilege Escalation — Linux](./12_Privilege_Escalation_Linux.md)

---

Reverse engineering is the process of analyzing software to understand its functionality without access to source code. In offensive security, reverse engineering is essential for vulnerability research, malware analysis, CTF challenges, and understanding proprietary protocols.

> **IMPORTANT**: Reverse engineering should only be performed in authorized contexts — CTFs, your own software, or with explicit permission.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

1. Use Ghidra for static analysis and decompilation
2. Debug binaries with GDB and extensions (pwndbg/GEF)
3. Analyze control flow to understand program logic
4. Identify and bypass anti-debugging techniques
5. Reverse engineer custom protocols and file formats
6. Patch binaries to modify behavior
7. Solve CTF reverse engineering challenges
8. Apply reverse engineering to vulnerability research

---

## Table of Contents

1. [Static Analysis with Ghidra](#1-static-analysis-with-ghidra)
2. [Dynamic Analysis with GDB](#2-dynamic-analysis-with-gdb)
3. [Disassembly vs Decompilation](#3-disassembly-vs-decompilation)
4. [Control Flow Analysis](#4-control-flow-analysis)
5. [Data Flow Analysis](#5-data-flow-analysis)
6. [Anti-Reversing Techniques](#6-anti-reversing-techniques)
7. [Debugging Stripped Binaries](#7-debugging-stripped-binaries)
8. [Protocol Reverse Engineering](#8-protocol-reverse-engineering)
9. [Patching Binaries](#9-patching-binaries)
10. [CTF Reverse Engineering Challenges](#10-ctf-reverse-engineering-challenges)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. Static Analysis with Ghidra

Ghidra is the NSA's open-source reverse engineering tool, providing disassembly, decompilation, and analysis capabilities.

### 1.1 Ghidra Workflow

```
Import Binary → Auto-Analysis → Review Functions → Decompile
     │                │                │              │
  Set architecture   Type propagation  Rename vars   Annotate
  Define memory map  Cross-references  Add comments  Export
```

### 1.2 Key Ghidra Features

- **Decompiler**: Produces readable C-like pseudocode from assembly
- **Cross-references (XREFs)**: Find where functions/data are used
- **Function graphs**: Visual control flow representation
- **Type system**: Define structs, enums, typedefs for clarity
- **Scripting**: Java/Python scripts for automation (Ghidra Script API)

```python
"""
Binary analysis helper for reverse engineering tasks.

Provides utilities for common RE operations like
string extraction, function identification, and
pattern matching in binary files.
"""

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class StringFound:
    """A string found in a binary file."""
    offset: int
    value: str
    encoding: str
    context: str = ""


@dataclass
class FunctionSignature:
    """A potential function identified by prologue pattern."""
    offset: int
    prologue: bytes
    arch: str


def extract_strings(filepath: str, min_length: int = 4) -> list[StringFound]:
    """
    Extract printable strings from a binary file.

    Similar to the Unix 'strings' command.
    """
    results = []
    try:
        with open(filepath, "rb") as f:
            data = f.read()

        # ASCII strings
        pattern = rb'[\x20-\x7e]{' + str(min_length).encode() + rb',}'
        for match in re.finditer(pattern, data):
            results.append(StringFound(
                offset=match.start(),
                value=match.group().decode('ascii'),
                encoding="ASCII",
            ))

        # UTF-16LE strings (common in Windows binaries)
        utf16_pattern = rb'(?:[\x20-\x7e]\x00){' + str(min_length).encode() + rb',}'
        for match in re.finditer(utf16_pattern, data):
            try:
                value = match.group().decode('utf-16-le')
                results.append(StringFound(
                    offset=match.start(),
                    value=value,
                    encoding="UTF-16LE",
                ))
            except UnicodeDecodeError:
                pass

    except OSError:
        pass

    return results


def find_function_prologues(data: bytes, arch: str = "x64") -> list[FunctionSignature]:
    """
    Find potential function starts by matching common prologue patterns.
    """
    prologues = {
        "x64": [
            b'\x55\x48\x89\xe5',      # push rbp; mov rbp, rsp
            b'\x55\x48\x89\xe5\x48',  # push rbp; mov rbp, rsp; sub rsp, ...
        ],
        "x86": [
            b'\x55\x89\xe5',          # push ebp; mov ebp, esp
            b'\x55\x89\xe5\x83',      # push ebp; mov ebp, esp; sub esp, ...
        ],
    }

    results = []
    for prologue in prologues.get(arch, []):
        offset = 0
        while True:
            idx = data.find(prologue, offset)
            if idx == -1:
                break
            results.append(FunctionSignature(
                offset=idx, prologue=prologue, arch=arch,
            ))
            offset = idx + 1

    return results


def analyze_binary(filepath: str) -> dict:
    """Perform basic binary analysis."""
    results = {
        "strings": [],
        "functions": [],
        "interesting_strings": [],
    }

    strings = extract_strings(filepath)
    results["strings"] = strings

    # Flag interesting strings
    interesting_patterns = [
        r"password", r"secret", r"flag\{", r"admin",
        r"login", r"http://", r"https://", r"/bin/sh",
        r"system", r"execve", r"strcmp", r"key",
    ]
    for s in strings:
        for pattern in interesting_patterns:
            if re.search(pattern, s.value, re.IGNORECASE):
                results["interesting_strings"].append(s)
                break

    return results


if __name__ == "__main__":
    print("Binary Analysis Module")
    print("=" * 40)
    print("Usage:")
    print("  results = analyze_binary('/path/to/binary')")
    print("  for s in results['interesting_strings']:")
    print("      print(f'0x{s.offset:08x}: {s.value}')")
```

---

## 2. Dynamic Analysis with GDB

### 2.1 GDB with pwndbg

```bash
# Install pwndbg
git clone https://github.com/pwndbg/pwndbg && cd pwndbg && ./setup.sh

# Essential pwndbg commands
(gdb) checksec        # Show security features
(gdb) vmmap           # Show memory map
(gdb) heap            # Show heap state
(gdb) search -s "flag"  # Search memory for string
(gdb) cyclic 200      # Generate cyclic pattern
(gdb) cyclic -l 0x6161616b  # Find offset
```

---

## 3. Disassembly vs Decompilation

**Disassembly**: Binary → Assembly (exact, low-level)
**Decompilation**: Binary → C-like pseudocode (approximate, high-level)

Both are needed: disassembly for precision, decompilation for understanding logic quickly.

---

## 4. Control Flow Analysis

Understanding how a program makes decisions is key to reverse engineering:

- **Basic blocks**: Sequences of instructions with one entry and one exit
- **Control flow graphs (CFG)**: Directed graph of basic blocks
- **Loops**: Back-edges in the CFG
- **Conditionals**: Branch instructions (je, jne, jl, jg)

---

## 5. Data Flow Analysis

Track how data moves through the program:

- Where does user input enter?
- How is it transformed or validated?
- Where does it influence control flow?
- Does it reach dangerous functions (strcpy, system, exec)?

---

## 6. Anti-Reversing Techniques

| Technique | Description | Bypass |
|-----------|-------------|--------|
| `ptrace` self-trace | Prevents debugger attachment | Patch ptrace call to NOP |
| Timing checks | Detects debugger slowdown | Skip timing comparison |
| Checksum verification | Detects binary modifications | Patch check or match checksum |
| Obfuscation | Transforms code to confuse analysis | Deobfuscation tools |
| Packing | Compress/encrypt code | Unpack at runtime, dump |
| VM-based protection | Custom bytecode interpreter | Analyze VM dispatcher |

---

## 7. Debugging Stripped Binaries

Stripped binaries have no symbol table (function names removed):

```bash
# Check if stripped
file binary
# binary: ELF 64-bit LSB executable, x86-64, stripped

# In GDB, find main via entry point
(gdb) info files
(gdb) break *0x401000  # Set breakpoint at entry
```

---

## 8. Protocol Reverse Engineering

When analyzing network protocols:

1. Capture traffic with Wireshark
2. Identify message boundaries and structure
3. Correlate with binary analysis (look for `send`/`recv` calls)
4. Build a protocol parser/generator

---

## 9. Patching Binaries

```python
"""
Binary patching utility for CTF challenges.

Modifies specific bytes in a binary to change behavior.
"""

import shutil
from pathlib import Path


def patch_binary(
    filepath: str,
    offset: int,
    original: bytes,
    replacement: bytes,
    output: str = "",
) -> bool:
    """
    Patch bytes at a specific offset in a binary.

    Verifies original bytes match before patching.
    """
    if len(original) != len(replacement):
        print("Error: patch must be same length")
        return False

    if not output:
        output = filepath + ".patched"

    shutil.copy2(filepath, output)

    with open(output, "r+b") as f:
        f.seek(offset)
        current = f.read(len(original))
        if current != original:
            print(f"Error: expected {original.hex()} at offset 0x{offset:x}, "
                  f"found {current.hex()}")
            return False
        f.seek(offset)
        f.write(replacement)

    print(f"Patched {len(replacement)} bytes at offset 0x{offset:x}")
    return True


# Common patches
NOP_X86 = b"\x90"
NOP_X64 = b"\x90"
JMP_SHORT = b"\xeb"  # Short unconditional jump
JE_TO_JNE = {b"\x74": b"\x75"}  # Flip conditional jump
JNE_TO_JE = {b"\x75": b"\x74"}


if __name__ == "__main__":
    print("Binary Patching Module")
    print("=" * 40)
    print("Common patches:")
    print("  NOP sled: Replace instruction with 0x90")
    print("  Flip jump: Change JE (0x74) to JNE (0x75)")
    print("  Force jump: Replace conditional with JMP (0xEB)")
```

---

## 10. CTF Reverse Engineering Challenges

### 10.1 Common Challenge Types

- **Crackme**: Find the correct input (password, key, flag)
- **Keygen**: Generate valid license keys
- **Unpacking**: Unpack a protected binary to reveal code
- **VM**: Reverse a custom virtual machine interpreter
- **Obfuscation**: Deobfuscate transformed code

### 10.2 Solving Strategy

1. Run the binary normally to understand behavior
2. Check strings for clues (`strings binary | grep flag`)
3. Load in Ghidra, find main(), analyze logic
4. Set breakpoints at interesting comparisons
5. Extract the flag/key from the comparison logic

---

## 11. Exercises

1. **Strings Challenge**: Download a crackme from crackmes.one. Find the password using only static analysis.
2. **GDB Debugging**: Trace execution of a simple binary, documenting register values at each function call.
3. **Ghidra Analysis**: Decompile a stripped binary and reconstruct meaningful function names.
4. **Anti-Debug Bypass**: Bypass a ptrace-based anti-debugging protection.
5. **Protocol RE**: Capture network traffic from a simple client-server application and document the protocol.
6. **Binary Patching**: Patch a crackme to always accept any password.

---

## 12. Summary

Reverse engineering reveals the inner workings of compiled software:

- **Static analysis** (Ghidra) examines code without running it
- **Dynamic analysis** (GDB) observes behavior during execution
- **Control flow** analysis reveals program logic and decision points
- **Anti-reversing** techniques attempt to prevent analysis
- **Binary patching** modifies program behavior at the byte level
- **CTF challenges** provide excellent practice for RE skills

---

## 13. References

- Ghidra: https://ghidra-sre.org/
- pwndbg: https://github.com/pwndbg/pwndbg
- GEF: https://github.com/hugsy/gef
- Crackmes.one: https://crackmes.one/
- Practical Reverse Engineering (Dang et al.)
- Reverse Engineering for Beginners: https://beginners.re/
