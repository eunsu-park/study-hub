# 리버스 엔지니어링

**이전**: [10. 버퍼 오버플로우](./10_Buffer_Overflow.md) | **다음**: [12. 권한 상승 — 리눅스](./12_Privilege_Escalation_Linux.md)

---

리버스 엔지니어링(reverse engineering)은 소스 코드에 대한 접근 없이 소프트웨어를 분석하여 그 기능을 이해하는 과정이다. 공격적 보안에서 리버스 엔지니어링은 취약점 연구, 악성코드 분석, CTF 챌린지, 독점 프로토콜 이해에 필수적이다.

> **중요**: 리버스 엔지니어링은 승인된 상황에서만 수행해야 한다 — CTF, 자신의 소프트웨어, 또는 명시적 허가가 있는 경우.

**난이도**: ⭐⭐⭐⭐

## 학습 목표

1. Ghidra를 사용하여 정적 분석과 디컴파일을 수행한다
2. GDB와 확장 도구(pwndbg/GEF)로 바이너리를 디버깅한다
3. 제어 흐름을 분석하여 프로그램 로직을 이해한다
4. 안티디버깅(anti-debugging) 기법을 식별하고 우회한다
5. 커스텀 프로토콜과 파일 형식을 리버스 엔지니어링한다
6. 바이너리를 패칭하여 동작을 수정한다
7. CTF 리버스 엔지니어링 챌린지를 해결한다
8. 리버스 엔지니어링을 취약점 연구에 적용한다

---

## 목차

1. [Ghidra를 이용한 정적 분석](#1-ghidra를-이용한-정적-분석)
2. [GDB를 이용한 동적 분석](#2-gdb를-이용한-동적-분석)
3. [디스어셈블리 vs 디컴파일](#3-디스어셈블리-vs-디컴파일)
4. [제어 흐름 분석](#4-제어-흐름-분석)
5. [데이터 흐름 분석](#5-데이터-흐름-분석)
6. [안티리버싱 기법](#6-안티리버싱-기법)
7. [스트립된 바이너리 디버깅](#7-스트립된-바이너리-디버깅)
8. [프로토콜 리버스 엔지니어링](#8-프로토콜-리버스-엔지니어링)
9. [바이너리 패칭](#9-바이너리-패칭)
10. [CTF 리버스 엔지니어링 챌린지](#10-ctf-리버스-엔지니어링-챌린지)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. Ghidra를 이용한 정적 분석

Ghidra는 NSA의 오픈소스 리버스 엔지니어링 도구로, 디스어셈블리, 디컴파일, 분석 기능을 제공한다.

### 1.1 Ghidra 워크플로우

```
Import Binary → Auto-Analysis → Review Functions → Decompile
     │                │                │              │
  Set architecture   Type propagation  Rename vars   Annotate
  Define memory map  Cross-references  Add comments  Export
```

### 1.2 주요 Ghidra 기능

- **디컴파일러(Decompiler)**: 어셈블리에서 읽기 쉬운 C 유사 의사코드를 생성
- **상호 참조(Cross-references, XREFs)**: 함수/데이터가 사용되는 위치를 찾기
- **함수 그래프**: 시각적 제어 흐름 표현
- **타입 시스템**: 명확성을 위해 구조체, 열거형, typedef 정의
- **스크립팅**: 자동화를 위한 Java/Python 스크립트 (Ghidra Script API)

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

## 2. GDB를 이용한 동적 분석

### 2.1 pwndbg를 사용한 GDB

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

## 3. 디스어셈블리 vs 디컴파일

**디스어셈블리(Disassembly)**: 바이너리 → 어셈블리 (정확, 저수준)
**디컴파일(Decompilation)**: 바이너리 → C 유사 의사코드 (근사, 고수준)

둘 다 필요하다: 정밀함을 위한 디스어셈블리, 로직을 빠르게 이해하기 위한 디컴파일.

---

## 4. 제어 흐름 분석

프로그램이 어떻게 결정을 내리는지 이해하는 것이 리버스 엔지니어링의 핵심이다:

- **기본 블록(Basic block)**: 하나의 진입점과 하나의 출구점을 가진 명령어 시퀀스
- **제어 흐름 그래프(Control Flow Graph, CFG)**: 기본 블록의 방향 그래프
- **루프**: CFG에서의 백엣지(back-edge)
- **조건문**: 분기 명령어 (je, jne, jl, jg)

---

## 5. 데이터 흐름 분석

프로그램을 통해 데이터가 어떻게 이동하는지 추적한다:

- 사용자 입력이 어디에서 들어오는가?
- 어떻게 변환되거나 검증되는가?
- 제어 흐름에 어디서 영향을 미치는가?
- 위험한 함수(strcpy, system, exec)에 도달하는가?

---

## 6. 안티리버싱 기법

| 기법 | 설명 | 우회 방법 |
|-----------|-------------|--------|
| `ptrace` 셀프 트레이스 | 디버거 연결 방지 | ptrace 호출을 NOP으로 패칭 |
| 타이밍 체크 | 디버거 지연 감지 | 타이밍 비교 건너뛰기 |
| 체크섬 검증 | 바이너리 수정 감지 | 체크 패칭 또는 체크섬 일치 |
| 난독화(Obfuscation) | 분석을 혼란시키기 위한 코드 변환 | 난독화 해제 도구 |
| 패킹(Packing) | 코드 압축/암호화 | 런타임에 언패킹, 덤프 |
| VM 기반 보호 | 커스텀 바이트코드 인터프리터 | VM 디스패처 분석 |

---

## 7. 스트립된 바이너리 디버깅

스트립된(stripped) 바이너리는 심볼 테이블이 없다 (함수 이름이 제거됨):

```bash
# Check if stripped
file binary
# binary: ELF 64-bit LSB executable, x86-64, stripped

# In GDB, find main via entry point
(gdb) info files
(gdb) break *0x401000  # Set breakpoint at entry
```

---

## 8. 프로토콜 리버스 엔지니어링

네트워크 프로토콜 분석 시:

1. Wireshark로 트래픽 캡처
2. 메시지 경계와 구조 식별
3. 바이너리 분석과 연관 (`send`/`recv` 호출 찾기)
4. 프로토콜 파서/생성기 구축

---

## 9. 바이너리 패칭

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

## 10. CTF 리버스 엔지니어링 챌린지

### 10.1 일반적인 챌린지 유형

- **Crackme**: 올바른 입력 찾기 (비밀번호, 키, 플래그)
- **Keygen**: 유효한 라이선스 키 생성
- **Unpacking**: 보호된 바이너리를 언패킹하여 코드 노출
- **VM**: 커스텀 가상 머신 인터프리터 리버싱
- **Obfuscation**: 변환된 코드 난독화 해제

### 10.2 풀이 전략

1. 바이너리를 정상적으로 실행하여 동작 이해
2. 문자열에서 단서 확인 (`strings binary | grep flag`)
3. Ghidra에서 로드하고, main()을 찾아 로직 분석
4. 흥미로운 비교문에 브레이크포인트 설정
5. 비교 로직에서 플래그/키 추출

---

## 11. 연습 문제

1. **문자열 챌린지**: crackmes.one에서 크랙미를 다운로드한다. 정적 분석만 사용하여 비밀번호를 찾는다.
2. **GDB 디버깅**: 간단한 바이너리의 실행을 추적하며, 각 함수 호출 시 레지스터 값을 기록한다.
3. **Ghidra 분석**: 스트립된 바이너리를 디컴파일하고 의미 있는 함수 이름을 재구성한다.
4. **안티디버그 우회**: ptrace 기반 안티디버깅 보호를 우회한다.
5. **프로토콜 RE**: 간단한 클라이언트-서버 애플리케이션의 네트워크 트래픽을 캡처하고 프로토콜을 문서화한다.
6. **바이너리 패칭**: 크랙미를 패칭하여 항상 어떤 비밀번호든 수락하도록 만든다.

---

## 12. 요약

리버스 엔지니어링은 컴파일된 소프트웨어의 내부 작동을 드러낸다:

- **정적 분석**(Ghidra)은 실행하지 않고 코드를 검사한다
- **동적 분석**(GDB)은 실행 중 동작을 관찰한다
- **제어 흐름** 분석은 프로그램 로직과 결정 지점을 드러낸다
- **안티리버싱** 기법은 분석을 방지하려고 시도한다
- **바이너리 패칭**은 바이트 수준에서 프로그램 동작을 수정한다
- **CTF 챌린지**는 RE 기술을 위한 훌륭한 실습을 제공한다

---

## 13. 참고 자료

- Ghidra: https://ghidra-sre.org/
- pwndbg: https://github.com/pwndbg/pwndbg
- GEF: https://github.com/hugsy/gef
- Crackmes.one: https://crackmes.one/
- Practical Reverse Engineering (Dang et al.)
- Reverse Engineering for Beginners: https://beginners.re/
