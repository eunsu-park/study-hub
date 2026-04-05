# CTF 방법론

**이전**: [18. 악성코드 분석](./18_Malware_Analysis.md) | **다음**: [20. 레드팀 작전](./20_Red_Team_Operations.md)

---

CTF(Capture-The-Flag) 대회는 공격 보안 기술을 개발하고 연마하는 가장 좋은 방법 중 하나이다. 이 레슨에서는 CTF 카테고리, 필수 도구, 풀이 방법론, 그리고 pwn, web, crypto, forensics, misc 등 모든 주요 카테고리의 챌린지를 접근하는 전략을 다룬다.

> **참고**: CTF 기술은 실제 침투 테스트 및 보안 연구에 직접 적용할 수 있다.

**난이도**: ⭐⭐⭐

## 학습 목표

1. Jeopardy 및 Attack-Defense CTF 형식을 이해한다
2. 바이너리 익스플로잇 챌린지를 체계적으로 접근한다
3. CTF에서 일반적인 웹 익스플로잇 패턴을 풀어본다
4. 암호학 공격 기법을 CTF 챌린지에 적용한다
5. CTF 증거 파일에 대한 포렌식 분석을 수행한다
6. pwntools를 사용하여 효율적으로 익스플로잇을 개발한다
7. 명확하고 교육적인 CTF 풀이(writeup)를 작성한다
8. CTF 팀으로서 효과적으로 조직하고 협업한다

---

## 목차

1. [CTF 대회 형식](#1-ctf-대회-형식)
2. [Pwn (바이너리 익스플로잇) 챌린지](#2-pwn-바이너리-익스플로잇-챌린지)
3. [웹 익스플로잇 챌린지](#3-웹-익스플로잇-챌린지)
4. [암호학 챌린지](#4-암호학-챌린지)
5. [포렌식 챌린지](#5-포렌식-챌린지)
6. [리버스 엔지니어링 챌린지](#6-리버스-엔지니어링-챌린지)
7. [필수 CTF 도구](#7-필수-ctf-도구)
8. [Pwntools 프레임워크](#8-pwntools-프레임워크)
9. [CTF 풀이 방법론](#9-ctf-풀이-방법론)
10. [CTF 팀 구성](#10-ctf-팀-구성)
11. [연습 문제](#11-연습-문제)
12. [요약](#12-요약)
13. [참고 자료](#13-참고-자료)

---

## 1. CTF 대회 형식

### 1.1 Jeopardy

가장 일반적인 형식이다. 여러 카테고리에 걸쳐 챌린지가 출제되며, 각 챌린지에는 플래그(예: `flag{s0m3_t3xt}`)가 있다. 더 어려운 챌린지일수록 더 높은 점수를 받는다.

### 1.2 Attack-Defense

팀이 자신의 서비스를 방어하면서 다른 팀의 서비스를 공격한다. 공격과 방어 기술을 동시에 요구한다.

### 1.3 King of the Hill

대상 머신을 탈취하고 유지한다. 다른 팀이 탈취하려는 동안 접근 권한을 유지해야 한다.

### 1.4 플랫폼

| 플랫폼 | 난이도 | 초점 |
|--------|--------|------|
| picoCTF | 초급 | 모든 카테고리, 교육적 |
| Hack The Box | 중급 | 실제적인 머신 |
| TryHackMe | 초급-중급 | 가이드 학습 |
| CTFtime | 전체 | 대회 일정 |
| OverTheWire | 초급 | 워게임 |
| pwnable.kr | 고급 | 바이너리 익스플로잇 |
| cryptopals | 중급 | 암호학 |

---

## 2. Pwn (바이너리 익스플로잇) 챌린지

### 2.1 접근 방법

```
1. checksec binary          # Check protections (NX, ASLR, PIE, canary)
2. file binary              # Architecture, linked libraries
3. Run and experiment        # Understand normal behavior
4. Disassemble (Ghidra)     # Find vulnerability
5. Determine protection bypass needed
6. Build exploit (pwntools)
7. Test locally, then remote
```

### 2.2 일반적인 패턴

| 보호 기법 | 우회 방법 |
|-----------|-----------|
| 보호 없음 | 직접 셸코드 인젝션 |
| NX만 적용 | ret2libc 또는 ROP 체인 |
| NX + ASLR | libc 주소를 유출한 후 ret2libc |
| NX + ASLR + Canary | 먼저 카나리를 유출한 후 ROP |
| NX + ASLR + PIE | PIE 베이스와 libc를 유출한 후 ROP |
| Full RELRO | GOT가 아닌 스택을 대상으로 함 |

---

## 3. 웹 익스플로잇 챌린지

### 3.1 접근 방법

```
1. Explore the application manually
2. View page source and JavaScript
3. Check robots.txt, .git, backup files
4. Identify technology stack
5. Test for common vulnerabilities (SQLi, XSS, SSTI, SSRF)
6. Check for authentication/authorization flaws
7. Fuzz parameters and endpoints
```

### 3.2 빠른 성과

```bash
# Check for exposed files
curl target/robots.txt
curl target/.git/HEAD
curl target/backup.sql
curl target/.env

# Directory enumeration
gobuster dir -u http://target -w /usr/share/dirb/common.txt

# Technology detection
whatweb target
```

---

## 4. 암호학 챌린지

### 4.1 일반적인 CTF 암호학

| 카테고리 | 공격 기법 |
|----------|-----------|
| 고전 암호 | 시저, 비즈네르, 치환 빈도 분석 |
| RSA | 작은 e, Wiener 공격, Hastad 공격, 공통 모듈러스 |
| AES | ECB 블록 셔플링, CBC 비트 플리핑, 패딩 오라클 |
| XOR | 알려진 평문, 단일 바이트 키 무차별 대입 |
| 해시 | 길이 확장, 충돌 |
| 커스텀 | 알고리즘의 약점 분석 |

```python
"""
CTF cryptography utilities.

Common crypto operations needed for CTF challenges.
"""

import base64
from itertools import cycle


def xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR data with a repeating key."""
    return bytes(a ^ b for a, b in zip(data, cycle(key)))


def single_byte_xor_crack(ciphertext: bytes) -> list[tuple[int, bytes, float]]:
    """Brute-force single-byte XOR encryption."""
    results = []
    for key in range(256):
        plaintext = bytes(b ^ key for b in ciphertext)
        score = sum(1 for b in plaintext if 32 <= b <= 126)  # Printable chars
        results.append((key, plaintext, score / len(plaintext)))
    return sorted(results, key=lambda x: x[2], reverse=True)[:5]


def caesar_cipher(text: str, shift: int) -> str:
    """Apply Caesar cipher with given shift."""
    result = []
    for c in text:
        if c.isalpha():
            base = ord('A') if c.isupper() else ord('a')
            result.append(chr((ord(c) - base + shift) % 26 + base))
        else:
            result.append(c)
    return ''.join(result)


def caesar_brute_force(ciphertext: str) -> list[tuple[int, str]]:
    """Try all 26 Caesar shifts."""
    return [(i, caesar_cipher(ciphertext, i)) for i in range(26)]


# Base encoding/decoding
def decode_multi_base(data: str) -> str:
    """Try multiple base decodings."""
    results = {}
    try:
        results["base64"] = base64.b64decode(data).decode()
    except Exception:
        pass
    try:
        results["base32"] = base64.b32decode(data).decode()
    except Exception:
        pass
    try:
        results["hex"] = bytes.fromhex(data).decode()
    except Exception:
        pass
    return results


if __name__ == "__main__":
    print("CTF Crypto Utilities")
    print("=" * 40)

    # Caesar example
    encrypted = "URYYBJBEYQ"
    print(f"Caesar brute force of '{encrypted}':")
    for shift, plaintext in caesar_brute_force(encrypted):
        if any(w in plaintext.lower() for w in ["the", "flag", "hello", "world"]):
            print(f"  Shift {shift}: {plaintext} ***")
```

---

## 5. 포렌식 챌린지

### 5.1 일반적인 유형

- **파일 카빙(File Carving)**: 이미지/디스크 덤프에서 숨겨진 파일을 추출한다
- **스테가노그래피(Steganography)**: 이미지/오디오에 숨겨진 데이터를 찾는다
- **메모리 포렌식(Memory Forensics)**: RAM 덤프를 분석한다 (Volatility)
- **네트워크 포렌식(Network Forensics)**: 패킷 캡처를 분석한다
- **디스크 포렌식(Disk Forensics)**: 디스크 이미지를 분석한다

### 5.2 필수 명령어

```bash
# File identification
file mystery_file
binwalk mystery_file           # Embedded files
foremost -i mystery_file       # File carving

# Image steganography
steghide info image.jpg
steghide extract -sf image.jpg
zsteg image.png               # PNG/BMP stego
exiftool image.jpg            # EXIF metadata

# Memory forensics (Volatility 3)
vol3 -f memory.dmp windows.info
vol3 -f memory.dmp windows.pslist
vol3 -f memory.dmp windows.filescan
vol3 -f memory.dmp windows.hashdump

# Packet capture
tshark -r capture.pcap -Y "http" -T fields -e http.request.uri
```

---

## 6. 리버스 엔지니어링 챌린지

레슨 11에서 이미 다루었다. CTF RE의 핵심 접근법은 다음과 같다:

1. `file`과 `checksec`으로 바이너리를 확인한다
2. `strings`로 빠른 단서를 찾는다
3. Ghidra에 로드하여 main 함수를 찾는다
4. 비교 로직을 식별한다
5. 비교 또는 알고리즘에서 플래그를 추출한다

---

## 7. 필수 CTF 도구

| 카테고리 | 도구 |
|----------|------|
| Pwn | pwntools, GDB+pwndbg, ROPgadget, one_gadget |
| Web | Burp Suite, ffuf, sqlmap, curl, browser DevTools |
| Crypto | Python + PyCryptodome, RsaCtfTool, SageMath |
| Forensics | Volatility, binwalk, steghide, Wireshark, Autopsy |
| RE | Ghidra, IDA Free, GDB+pwndbg, radare2 |
| Misc | CyberChef, dcode.fr, Python |

---

## 8. Pwntools 프레임워크

```python
"""
Pwntools exploit template for CTF challenges.
"""

# from pwn import *

# Template for a typical pwn challenge:
EXPLOIT_TEMPLATE = '''
from pwn import *

# Configuration
binary_path = "./challenge"
remote_host = "ctf.example.com"
remote_port = 1337

# Setup
elf = ELF(binary_path)
context.binary = elf

# Choose target
if args.REMOTE:
    p = remote(remote_host, remote_port)
else:
    p = process(binary_path)

# Optional: attach GDB
if args.GDB:
    gdb.attach(p, """
        break main
        continue
    """)

# Build payload
offset = 72  # Offset to return address
payload = flat(
    b"A" * offset,
    elf.symbols["win"],  # or p64(address)
)

# Send exploit
p.sendlineafter(b"Input: ", payload)

# Get flag
p.interactive()
'''


if __name__ == "__main__":
    print("Pwntools Exploit Template")
    print("=" * 50)
    print(EXPLOIT_TEMPLATE)
```

---

## 9. CTF 풀이 방법론

좋은 풀이(writeup)에는 다음이 포함된다:

1. **챌린지 설명**: 이름, 카테고리, 점수, 설명
2. **정찰**: 처음에 관찰한 내용
3. **분석**: 취약점을 어떻게 식별했는지
4. **익스플로잇**: 단계별 익스플로잇 개발 과정
5. **플래그**: 획득한 플래그
6. **교훈**: 배운 점, 대안적 접근 방법

---

## 10. CTF 팀 구성

### 10.1 이상적인 팀 구성

- **Pwn 전문가**: 바이너리 익스플로잇
- **Web 전문가**: 웹 애플리케이션 공격
- **Crypto 전문가**: 암호학 챌린지
- **Forensics 전문가**: 파일 분석, 메모리 포렌식
- **제너럴리스트**: 기타 챌린지, 정찰

### 10.2 팀 활동

- CTFtime 챌린지를 활용한 정기 연습 세션
- 공유 지식 기반 및 도구 문서화
- 대회 중 명확한 의사소통
- 대회 후 복기 및 풀이 공유

---

## 11. 연습 문제

1. **초급 CTF**: picoCTF에서 모든 카테고리에 걸쳐 10개의 챌린지를 완료한다.
2. **Pwn 챌린지**: pwnable.kr에서 3개의 바이너리 익스플로잇 챌린지를 풀어본다.
3. **Web 챌린지**: PortSwigger Web Security Academy에서 하나의 주제에 대한 모든 랩을 완료한다.
4. **Crypto 챌린지**: Cryptopals의 처음 8개 챌린지를 완료한다.
5. **포렌식**: Volatility로 메모리 덤프를 분석하고 모든 자격 증명을 추출한다.
6. **실전 CTF**: 다가오는 CTFtime 대회에 팀으로 참가한다.

---

## 12. 요약

CTF 대회는 공격 보안의 훈련장이다:

- **Jeopardy CTF**는 pwn, web, crypto, forensics, RE 전반의 기술을 테스트한다
- **Pwn 챌린지**는 바이너리 익스플로잇을 요구한다 (오버플로우, ROP, 포맷 스트링)
- **Web 챌린지**는 OWASP Top 10과 창의적인 익스플로잇을 테스트한다
- **Crypto 챌린지**는 수학적 추론과 구현 공격을 요구한다
- **포렌식**은 파일 분석, 스테가노그래피, 메모리 포렌식을 결합한다
- **pwntools**는 익스플로잇 개발의 필수 프레임워크이다
- **팀 협업**과 풀이 공유가 학습을 가속한다

---

## 13. 참고 자료

- CTFtime: https://ctftime.org/
- picoCTF: https://picoctf.org/
- pwntools: https://docs.pwntools.com/
- ROP Emporium: https://ropemporium.com/
- Cryptopals: https://cryptopals.com/
- PortSwigger Academy: https://portswigger.net/web-security
- CyberChef: https://gchq.github.io/CyberChef/
