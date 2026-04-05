# CTF Methodology

**Previous**: [18. Malware Analysis](./18_Malware_Analysis.md) | **Next**: [20. Red Team Operations](./20_Red_Team_Operations.md)

---

Capture-The-Flag (CTF) competitions are one of the best ways to develop and sharpen offensive security skills. This lesson covers CTF categories, essential tools, solving methodologies, and strategies for approaching challenges across all major categories: pwn, web, crypto, forensics, and misc.

> **NOTE**: CTF skills directly translate to real-world penetration testing and security research.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

1. Understand Jeopardy and Attack-Defense CTF formats
2. Approach binary exploitation challenges systematically
3. Solve common web exploitation patterns in CTFs
4. Apply cryptographic attack techniques to CTF challenges
5. Perform forensic analysis on CTF evidence files
6. Use pwntools for efficient exploit development
7. Write clear and educational CTF writeups
8. Organize and collaborate effectively as a CTF team

---

## Table of Contents

1. [CTF Competition Formats](#1-ctf-competition-formats)
2. [Pwn (Binary Exploitation) Challenges](#2-pwn-binary-exploitation-challenges)
3. [Web Exploitation Challenges](#3-web-exploitation-challenges)
4. [Cryptography Challenges](#4-cryptography-challenges)
5. [Forensics Challenges](#5-forensics-challenges)
6. [Reverse Engineering Challenges](#6-reverse-engineering-challenges)
7. [Essential CTF Tools](#7-essential-ctf-tools)
8. [Pwntools Framework](#8-pwntools-framework)
9. [CTF Writeup Methodology](#9-ctf-writeup-methodology)
10. [Building a CTF Team](#10-building-a-ctf-team)
11. [Exercises](#11-exercises)
12. [Summary](#12-summary)
13. [References](#13-references)

---

## 1. CTF Competition Formats

### 1.1 Jeopardy

Most common format. Challenges across categories, each with a flag (e.g., `flag{s0m3_t3xt}`). More difficult challenges = more points.

### 1.2 Attack-Defense

Teams defend their own services while attacking others. Requires both offensive and defensive skills simultaneously.

### 1.3 King of the Hill

Capture and hold a target machine. Maintain access while others try to take it.

### 1.4 Platforms

| Platform | Difficulty | Focus |
|----------|-----------|-------|
| picoCTF | Beginner | All categories, educational |
| Hack The Box | Intermediate | Realistic machines |
| TryHackMe | Beginner-Intermediate | Guided learning |
| CTFtime | All | Competition calendar |
| OverTheWire | Beginner | Wargames |
| pwnable.kr | Advanced | Binary exploitation |
| cryptopals | Intermediate | Cryptography |

---

## 2. Pwn (Binary Exploitation) Challenges

### 2.1 Approach

```
1. checksec binary          # Check protections (NX, ASLR, PIE, canary)
2. file binary              # Architecture, linked libraries
3. Run and experiment        # Understand normal behavior
4. Disassemble (Ghidra)     # Find vulnerability
5. Determine protection bypass needed
6. Build exploit (pwntools)
7. Test locally, then remote
```

### 2.2 Common Patterns

| Protection | Bypass |
|-----------|--------|
| No protections | Direct shellcode injection |
| NX only | ret2libc or ROP chain |
| NX + ASLR | Leak libc address, then ret2libc |
| NX + ASLR + Canary | Leak canary first, then ROP |
| NX + ASLR + PIE | Leak PIE base and libc, then ROP |
| Full RELRO | Target stack, not GOT |

---

## 3. Web Exploitation Challenges

### 3.1 Approach

```
1. Explore the application manually
2. View page source and JavaScript
3. Check robots.txt, .git, backup files
4. Identify technology stack
5. Test for common vulnerabilities (SQLi, XSS, SSTI, SSRF)
6. Check for authentication/authorization flaws
7. Fuzz parameters and endpoints
```

### 3.2 Quick Wins

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

## 4. Cryptography Challenges

### 4.1 Common CTF Crypto

| Category | Attacks |
|----------|---------|
| Classical | Caesar, Vigenere, substitution frequency analysis |
| RSA | Small e, Wiener's attack, Hastad's, common modulus |
| AES | ECB block shuffling, CBC bit flipping, padding oracle |
| XOR | Known plaintext, single-byte key brute force |
| Hash | Length extension, collision |
| Custom | Analyze algorithm for weaknesses |

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

## 5. Forensics Challenges

### 5.1 Common Types

- **File carving**: Extract hidden files from images/disk dumps
- **Steganography**: Hidden data in images/audio
- **Memory forensics**: Analyze RAM dumps (Volatility)
- **Network forensics**: Analyze packet captures
- **Disk forensics**: Analyze disk images

### 5.2 Essential Commands

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

## 6. Reverse Engineering Challenges

Already covered in Lesson 11. Key CTF RE approach:

1. `file` and `checksec` the binary
2. `strings` for quick clues
3. Load in Ghidra, find main
4. Identify comparison logic
5. Extract flag from comparison or algorithm

---

## 7. Essential CTF Tools

| Category | Tools |
|----------|-------|
| Pwn | pwntools, GDB+pwndbg, ROPgadget, one_gadget |
| Web | Burp Suite, ffuf, sqlmap, curl, browser DevTools |
| Crypto | Python + PyCryptodome, RsaCtfTool, SageMath |
| Forensics | Volatility, binwalk, steghide, Wireshark, Autopsy |
| RE | Ghidra, IDA Free, GDB+pwndbg, radare2 |
| Misc | CyberChef, dcode.fr, Python |

---

## 8. Pwntools Framework

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

## 9. CTF Writeup Methodology

A good writeup includes:

1. **Challenge description**: Name, category, points, description
2. **Reconnaissance**: What you observed initially
3. **Analysis**: How you identified the vulnerability
4. **Exploitation**: Step-by-step exploit development
5. **Flag**: The captured flag
6. **Lessons learned**: What you learned, alternative approaches

---

## 10. Building a CTF Team

### 10.1 Ideal Team Composition

- **Pwn specialist**: Binary exploitation
- **Web specialist**: Web application attacks
- **Crypto specialist**: Cryptographic challenges
- **Forensics specialist**: File analysis, memory forensics
- **Generalist**: Miscellaneous challenges, reconnaissance

### 10.2 Team Practices

- Regular practice sessions on CTFtime challenges
- Shared knowledge base and tool documentation
- Clear communication during competitions
- Post-competition review and writeup sharing

---

## 11. Exercises

1. **Beginner CTF**: Complete 10 challenges on picoCTF across all categories.
2. **Pwn Challenge**: Solve 3 binary exploitation challenges from pwnable.kr.
3. **Web Challenge**: Complete all PortSwigger Web Security Academy labs for one topic.
4. **Crypto Challenge**: Complete the first 8 Cryptopals challenges.
5. **Forensics**: Analyze a memory dump with Volatility and extract all credentials.
6. **Full CTF**: Participate in an upcoming CTFtime competition with a team.

---

## 12. Summary

CTF competitions are the training ground for offensive security:

- **Jeopardy CTFs** test skills across pwn, web, crypto, forensics, and RE
- **Pwn challenges** require binary exploitation (overflow, ROP, format string)
- **Web challenges** test OWASP Top 10 and creative exploitation
- **Crypto challenges** require mathematical reasoning and implementation attacks
- **Forensics** combines file analysis, steganography, and memory forensics
- **pwntools** is the essential framework for exploit development
- **Team collaboration** and writeups accelerate learning

---

## 13. References

- CTFtime: https://ctftime.org/
- picoCTF: https://picoctf.org/
- pwntools: https://docs.pwntools.com/
- ROP Emporium: https://ropemporium.com/
- Cryptopals: https://cryptopals.com/
- PortSwigger Academy: https://portswigger.net/web-security
- CyberChef: https://gchq.github.io/CyberChef/
