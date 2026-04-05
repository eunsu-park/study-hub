# 해싱과 데이터 무결성

**이전**: [02. 암호학 기초](./02_Cryptography_Basics.md) | **다음**: [04. TLS/SSL과 공개키 인프라](./04_TLS_and_PKI.md)

---

해시 함수는 현대 보안의 핵심 도구입니다. 패스워드 저장, 디지털 서명, 메시지 인증, 블록체인, 소프트웨어 배포, 데이터 중복 제거 등 모든 곳에서 사용됩니다. 이 레슨은 해시 함수의 유용함을 만드는 수학적 속성부터 패스워드 해싱, HMAC, Merkle 트리의 실용적 구현까지 해시 함수를 심도 있게 다룹니다.

**난이도**: ⭐⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 암호학적 해시 함수의 속성 이해
2. SHA-256, SHA-3, BLAKE2, BLAKE3를 사용한 해싱 구현
3. bcrypt, scrypt, Argon2를 사용한 안전한 패스워드 저장
4. 메시지 인증을 위한 HMAC 구성 및 검증
5. Merkle 트리 구축 및 이해
6. 내용 주소 지정 저장소 구현
7. 상수 시간 비교를 사용한 타이밍 공격 인식 및 방지

---

## 목차

1. [해시 함수란 무엇인가?](#1-해시-함수란-무엇인가)
2. [암호학적 해시 속성](#2-암호학적-해시-속성)
3. [해시 함수 조사](#3-해시-함수-조사)
4. [hashlib를 사용한 Python 해싱](#4-hashlib를-사용한-python-해싱)
5. [패스워드 해싱](#5-패스워드-해싱)
6. [HMAC: 메시지 인증](#6-hmac-메시지-인증)
7. [Merkle 트리](#7-merkle-트리)
8. [내용 주소 지정 저장소](#8-내용-주소-지정-저장소)
9. [타이밍 공격과 상수 시간 비교](#9-타이밍-공격과-상수-시간-비교)
10. [해시 함수 공격과 완화](#10-해시-함수-공격과-완화)
11. [연습 문제](#11-연습-문제)
12. [참고 문헌](#12-참고-문헌)

---

## 1. 해시 함수란 무엇인가?

해시 함수는 임의 길이의 입력을 받아 고정 길이의 출력("해시" 또는 "다이제스트")을 생성합니다. **암호학적** 해시 함수는 보안 애플리케이션에 적합하게 만드는 추가적인 보안 속성을 가지고 있습니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       해시 함수                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   입력 (임의 크기)        해시 함수             출력 (고정)           │
│   ──────────────        ──────────────         ──────────────        │
│                                                                      │
│   "Hello"           ──────▶  SHA-256  ──────▶  2cf24dba5fb0...      │
│   (5 bytes)                                    (32 bytes / 256 bits) │
│                                                                      │
│   "Hello World"     ──────▶  SHA-256  ──────▶  a591a6d40bf4...      │
│   (11 bytes)                                   (32 bytes / 256 bits) │
│                                                                      │
│   전쟁과 평화         ──────▶  SHA-256  ──────▶  b28f8b893c45...      │
│   (~3.2 MB)                                    (32 bytes / 256 bits) │
│                                                                      │
│   핵심 통찰: 입력 크기에 관계없이 출력은 항상 동일한 고정 크기입니다.  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.1 비암호학적 vs 암호학적 해시

```
┌─────────────────────────────────────────────────────────────────────┐
│        비암호학적 vs 암호학적 해시 함수                                │
├──────────────────┬──────────────────────┬───────────────────────────┤
│                  │ 비암호학적            │ 암호학적                   │
├──────────────────┼──────────────────────┼───────────────────────────┤
│ 목적             │ 해시 테이블, 체크섬   │ 보안 애플리케이션          │
│ 속도             │ 매우 빠름             │ 의도적으로 느림            │
│ 충돌 저항성       │ 약함 / 없음           │ 강함 (필수)                │
│ 원상 저항성       │ 보장되지 않음         │ 강함 (필수)                │
│ 예시             │ CRC32, MurmurHash,   │ SHA-256, SHA-3, BLAKE2,   │
│                  │ xxHash, FNV          │ BLAKE3                    │
│ 사용 사례         │ 해시 맵, 체크섬,      │ 패스워드, 서명,            │
│                  │ 중복 제거             │ 인증서, HMAC              │
├──────────────────┴──────────────────────┴───────────────────────────┤
│ 규칙: 보안 목적으로 비암호학적 해시를 절대 사용하지 마십시오.         │
│ 속도가 중요하고 보안이 중요하지 않은 경우 암호학적 해시를 절대        │
│ 사용하지 마십시오 (예: 해시 테이블).                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 암호학적 해시 속성

안전한 암호학적 해시 함수는 세 가지 속성을 만족해야 합니다:

```
┌──────────────────────────────────────────────────────────────────────┐
│             암호학적 해시의 세 가지 속성                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 원상 저항성 (단방향)                                              │
│     해시 h가 주어졌을 때, H(m) = h를 만족하는 m을 찾는 것이            │
│     불가능합니다.                                                     │
│                                                                      │
│     해시 ──╳──▶ 원본 입력                                           │
│     (해시를 역으로 계산하여 입력을 얻을 수 없습니다)                   │
│                                                                      │
│  2. 제2 원상 저항성 (약한 충돌 저항성)                                │
│     m₁이 주어졌을 때, H(m₁) = H(m₂)를 만족하는 m₂ ≠ m₁을            │
│     찾는 것이 불가능합니다.                                           │
│                                                                      │
│     "Hello" → abc123    다른 입력 → abc123을 찾을 수 없음            │
│     (동일한 해시를 가진 두 번째 입력을 찾을 수 없습니다)               │
│                                                                      │
│  3. 충돌 저항성 (강한 충돌 저항성)                                    │
│     H(m₁) = H(m₂)를 만족하는 서로 다른 두 메시지 m₁ ≠ m₂를           │
│     찾는 것이 불가능합니다.                                           │
│                                                                      │
│     동일한 해시를 가진 입력 쌍을 찾을 수 없습니다                      │
│     (이것은 제2 원상 저항성보다 엄격하게 강합니다)                     │
│                                                                      │
│  보안 수준 (n비트 해시의 경우):                                       │
│  • 원상:         O(2^n) 작업 (생일 역설이 적용되지 않음)              │
│  • 제2 원상:     O(2^n) 작업                                         │
│  • 충돌:         O(2^(n/2)) 작업 (생일 역설)                         │
│                                                                      │
│  이것이 SHA-256이 128비트 충돌 저항성을 제공하는 이유입니다            │
│  (2^128 ≈ 3.4 × 10^38 연산).                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 눈사태 효과

좋은 해시 함수는 **눈사태 효과**를 나타냅니다: 입력의 작은 변화가 전혀 다른 출력을 생성합니다.

```python
import hashlib

# 눈사태 효과 시연
msg1 = b"Hello, World!"
msg2 = b"Hello, World?"  # '!'를 '?'로 변경
msg3 = b"Hello, World! " # 공백 추가

hash1 = hashlib.sha256(msg1).hexdigest()
hash2 = hashlib.sha256(msg2).hexdigest()
hash3 = hashlib.sha256(msg3).hexdigest()

print("Avalanche Effect Demonstration (SHA-256):")
print(f"  '{msg1.decode()}' → {hash1}")
print(f"  '{msg2.decode()}' → {hash2}")
print(f"  '{msg3.decode()}'→ {hash3}")

# 다른 비트 수 계산
def bit_difference(hex1: str, hex2: str) -> tuple:
    """두 16진수 문자열 간의 다른 비트 수를 계산합니다."""
    b1 = int(hex1, 16)
    b2 = int(hex2, 16)
    xor = b1 ^ b2
    diff_bits = bin(xor).count('1')
    total_bits = len(hex1) * 4
    return diff_bits, total_bits

diff, total = bit_difference(hash1, hash2)
print(f"\n  변경된 비트 (1자 차이): {diff}/{total} "
      f"({diff/total*100:.1f}%)")

diff2, _ = bit_difference(hash1, hash3)
print(f"  변경된 비트 (공백 추가): {diff2}/{total} "
      f"({diff2/total*100:.1f}%)")

# 이상적: 모든 변경에 대해 약 50%의 비트가 다름
print(f"  무작위 예상: ~{total//2} 비트 ({50.0}%)")
```

---

## 3. 해시 함수 조사

### 3.1 해시 함수 비교

```
┌─────────────────────────────────────────────────────────────────────┐
│                해시 함수 비교 표                                      │
├────────────┬────────┬───────────┬──────────┬────────────────────────┤
│ 알고리즘    │ 출력   │ 속도      │ 상태     │ 참고                   │
│            │ (비트) │ (GB/s)*   │          │                        │
├────────────┼────────┼───────────┼──────────┼────────────────────────┤
│ MD5        │ 128    │ ~5.0      │ 손상됨   │ 2004년 충돌 발견       │
│ SHA-1      │ 160    │ ~3.0      │ 손상됨   │ 2017년 SHAttered 공격  │
│ SHA-256    │ 256    │ ~1.5      │ 안전     │ 가장 널리 사용됨       │
│ SHA-512    │ 512    │ ~2.0†     │ 안전     │ 64비트 CPU에서 더 빠름 │
│ SHA-3-256  │ 256    │ ~0.5      │ 안전     │ 다른 구조              │
│ BLAKE2b    │ 256    │ ~3.5      │ 안전     │ SHA-256보다 빠름       │
│ BLAKE2s    │ 256    │ ~2.0      │ 안전     │ 32비트 최적화          │
│ BLAKE3     │ 256    │ ~10.0     │ 안전     │ 병렬화 가능, 최신      │
├────────────┴────────┴───────────┴──────────┴────────────────────────┤
│ * 하드웨어 지원이 있는 최신 x86-64에서의 대략적인 처리량              │
│ † SHA-512는 64비트 프로세서에서 SHA-256보다 빠름                     │
│                                                                      │
│ 권장 사항: 상호 운용성을 위해 SHA-256, 성능에 민감한 애플리케이션을   │
│ 위해 BLAKE2b/BLAKE3                                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 SHA-2 패밀리

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SHA-2 패밀리                                  │
├──────────────┬──────────┬──────────┬──────────┬─────────────────────┤
│ 변형         │ 출력     │ 블록     │ 워드     │ 라운드              │
│              │ (비트)   │ (비트)   │ (비트)   │                     │
├──────────────┼──────────┼──────────┼──────────┼─────────────────────┤
│ SHA-224      │ 224      │ 512      │ 32       │ 64                  │
│ SHA-256      │ 256      │ 512      │ 32       │ 64                  │
│ SHA-384      │ 384      │ 1024     │ 64       │ 80                  │
│ SHA-512      │ 512      │ 1024     │ 64       │ 80                  │
│ SHA-512/256  │ 256      │ 1024     │ 64       │ 80                  │
├──────────────┴──────────┴──────────┴──────────┴─────────────────────┤
│ SHA-256과 SHA-512가 가장 일반적으로 사용됩니다.                      │
│ SHA-512/256은 64비트 프로세서에서 SHA-512의 속도로 SHA-256 출력       │
│ 크기를 제공하며 길이 확장에 취약하지 않습니다.                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 SHA-3 (Keccak)

SHA-3은 SHA-2의 Merkle-Damgard 구조와 근본적으로 다른 **스펀지 구조**를 사용합니다. 이러한 다양성은 가치가 있습니다: SHA-2가 손상되더라도 SHA-3은 안전하게 유지될 가능성이 높습니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│              SHA-3 스펀지 구조 (단순화됨)                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  State = [0...0]  (Keccak의 경우 1600비트)                          │
│                                                                      │
│  흡수 단계:                                                          │
│  ┌───────┐   XOR    ┌───────┐   XOR    ┌───────┐                   │
│  │ msg₁  │──────▶ f ──────▶│ msg₂  │──────▶ f ──────▶ ...        │
│  └───────┘   State  └───────┘   State  └───────┘                   │
│                                                                      │
│  압착 단계:                                                          │
│  ... ──▶ f ──▶ [output₁] ──▶ f ──▶ [output₂] ──▶ ...             │
│                                                                      │
│  f = Keccak 순열 (5개 하위 라운드 × 24 라운드)                       │
│                                                                      │
│  주요 장점: 길이 확장 공격에 취약하지 않음                            │
│  (SHA-256 / SHA-512와 달리)                                         │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.4 BLAKE2와 BLAKE3

```
┌─────────────────────────────────────────────────────────────────────┐
│                   BLAKE2와 BLAKE3                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  BLAKE2 (RFC 7693):                                                 │
│  ├── BLAKE2b: 64비트 플랫폼 최적화 (최대 64바이트)                   │
│  ├── BLAKE2s: 32비트 플랫폼 최적화 (최대 32바이트)                   │
│  ├── 내장 키잉 (HMAC 없이 MAC로 작동 가능)                          │
│  ├── 내장 개인화 및 솔트 지원                                        │
│  ├── 병렬화를 위한 트리 해싱 모드                                    │
│  └── 사용처: Argon2 패스워드 해시, WireGuard, libsodium             │
│                                                                      │
│  BLAKE3 (2020):                                                     │
│  ├── BLAKE2 기반이지만 속도를 위해 재설계됨                          │
│  ├── Merkle 트리 구조 (본질적으로 병렬화 가능)                       │
│  ├── 해시, MAC, KDF, XOF를 위한 단일 알고리즘                       │
│  ├── 256비트 출력 (확장 가능)                                       │
│  ├── 최신 CPU에서 SHA-256보다 약 10배 빠름                          │
│  └── 사용처: Bao (검증된 스트리밍), 많은 새로운 프로젝트             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. hashlib를 사용한 Python 해싱

### 4.1 기본 해싱

```python
import hashlib

message = b"The quick brown fox jumps over the lazy dog"

# SHA-256 (가장 일반적)
sha256 = hashlib.sha256(message).hexdigest()
print(f"SHA-256:   {sha256}")

# SHA-512
sha512 = hashlib.sha512(message).hexdigest()
print(f"SHA-512:   {sha512[:64]}...{sha512[-8:]}")

# SHA-3-256
sha3_256 = hashlib.sha3_256(message).hexdigest()
print(f"SHA-3-256: {sha3_256}")

# BLAKE2b (최대 64바이트까지 가변 출력 크기)
blake2b = hashlib.blake2b(message, digest_size=32).hexdigest()
print(f"BLAKE2b:   {blake2b}")

# BLAKE2s (최대 32바이트까지 가변 출력 크기)
blake2s = hashlib.blake2s(message, digest_size=32).hexdigest()
print(f"BLAKE2s:   {blake2s}")

# 사용 가능한 모든 알고리즘 나열
print(f"\nAvailable: {sorted(hashlib.algorithms_available)[:10]}...")
```

### 4.2 증분 해싱 (스트리밍)

큰 파일의 경우 모든 것을 메모리에 로드하는 대신 데이터를 증분적으로 해시해야 합니다:

```python
import hashlib
from pathlib import Path

def hash_file(filepath: str, algorithm: str = "sha256",
              chunk_size: int = 8192) -> str:
    """모든 것을 메모리에 로드하지 않고 파일을 증분적으로 해시합니다."""
    hasher = hashlib.new(algorithm)
    with open(filepath, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()

# 테스트 파일 생성
test_file = "/tmp/test_hash_file.bin"
with open(test_file, 'wb') as f:
    for i in range(1000):
        f.write(f"Line {i}: some data for hashing\n".encode())

# 해시 계산
sha256_hash = hash_file(test_file, "sha256")
sha3_hash = hash_file(test_file, "sha3_256")
blake2_hash = hash_file(test_file, "blake2b")

print(f"File size: {Path(test_file).stat().st_size:,} bytes")
print(f"SHA-256:   {sha256_hash}")
print(f"SHA-3-256: {sha3_hash}")
print(f"BLAKE2b:   {blake2_hash}")

# 한 번에 모두 읽는 것과 동일 (그러나 메모리 효율적)
data = Path(test_file).read_bytes()
assert hashlib.sha256(data).hexdigest() == sha256_hash
print("\nIncremental hash matches full-read hash: OK")

# 정리
Path(test_file).unlink()
```

### 4.3 키를 사용한 BLAKE2 (키 해싱)

BLAKE2는 키 해싱을 위한 내장 지원이 있어 HMAC 구조 없이 MAC로 사용할 수 있습니다:

```python
import hashlib
import os

# 키가 있는 BLAKE2b (MAC로 작동)
key = os.urandom(32)  # 256비트 키
message = b"Authenticate this message"

# 키 해시
mac = hashlib.blake2b(message, key=key, digest_size=32).hexdigest()
print(f"BLAKE2b keyed hash: {mac}")

# 검증
verification = hashlib.blake2b(message, key=key, digest_size=32).hexdigest()
print(f"Verification match: {mac == verification}")

# 개인화와 함께 (도메인 분리)
mac1 = hashlib.blake2b(
    b"shared data",
    key=key,
    person=b"payment-v1",  # 최대 16바이트
    digest_size=32
).hexdigest()

mac2 = hashlib.blake2b(
    b"shared data",
    key=key,
    person=b"session-v1",  # 다른 도메인
    digest_size=32
).hexdigest()

print(f"\nSame data, different personalization:")
print(f"  payment context: {mac1[:32]}...")
print(f"  session context: {mac2[:32]}...")
print(f"  Same? {mac1 == mac2}")  # False - 다른 도메인
```

### 4.4 BLAKE3

```python
# pip install blake3
import blake3

message = b"BLAKE3 is extremely fast"

# 기본 해시
digest = blake3.blake3(message).hexdigest()
print(f"BLAKE3: {digest}")

# 키 해시 (MAC용)
key = b"0" * 32  # 32바이트 키 필요
keyed = blake3.blake3(message, key=key).hexdigest()
print(f"BLAKE3 keyed: {keyed}")

# 키 유도
derived = blake3.blake3(
    b"input key material",
    derive_key_context="my-app 2026-01-15 encryption key"
).hexdigest()
print(f"BLAKE3 KDF: {derived}")

# 증분 해싱
hasher = blake3.blake3()
hasher.update(b"Hello, ")
hasher.update(b"World!")
print(f"BLAKE3 incremental: {hasher.hexdigest()}")

# 확장 가능한 출력 (XOF) - 임의의 바이트 수 얻기
digest_64 = blake3.blake3(message).hexdigest(length=64)
print(f"BLAKE3 64-byte: {digest_64}")
```

---

## 5. 패스워드 해싱

패스워드 해싱은 범용 해싱과 근본적으로 다릅니다. 패스워드는 엔트로피가 낮으므로 (인간은 예측 가능한 패스워드를 선택합니다) 패스워드 해시를 얻은 공격자는 무차별 대입을 시도할 수 있습니다. 패스워드 해싱 알고리즘은 무차별 대입 공격을 비실용적으로 만들기 위해 **의도적으로 느리게** 설계되었습니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│        범용 해시가 패스워드에 나쁜 이유                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  최신 GPU에서의 SHA-256 속도: 초당 약 100억 해시                      │
│                                                                      │
│  패스워드 "P@ssw0rd":                                                │
│    SHA-256 → d74ff0ee8da3b9806b18c877d...                           │
│    8자 패스워드 무차별 대입 시간: 초에서 분                            │
│                                                                      │
│  bcrypt (cost=12):                                                  │
│    bcrypt → $2b$12$LJ3m4ys3Tdb2vNQhk9Oy...                        │
│    무차별 대입 시간: 수년에서 수세기                                   │
│                                                                      │
│  주요 차이점:                                                         │
│  • 패스워드 해시에는 무작위 솔트가 포함됨 (레인보우 테이블 방지)       │
│  • 패스워드 해시에는 조정 가능한 작업 계수가 있음 (하드웨어에 적응)    │
│  • 패스워드 해시는 큰 메모리가 필요할 수 있음 (GPU 공격 방어)         │
│                                                                      │
│  SHA-256, MD5 또는 다른 범용 해시로 패스워드를 절대 해시하지 마십시오! │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.1 bcrypt

```python
# pip install bcrypt
import bcrypt
import time

password = b"correct-horse-battery-staple"

# 패스워드 해시
# bcrypt는 자동으로 무작위 16바이트 솔트를 생성합니다
# cost=12는 2^12 = 4096 반복을 의미합니다
start = time.time()
hashed = bcrypt.hashpw(password, bcrypt.gensalt(rounds=12))
elapsed = time.time() - start

print(f"Password:  {password.decode()}")
print(f"Hash:      {hashed.decode()}")
print(f"Time:      {elapsed:.3f}s")
print()

# bcrypt 해시의 구조:
# $2b$12$LJ3m4ys3Tdb2vNQhk9OyAeKK3b3eAGQjT5xKp2JFe5cF5NY5U/a2e
# ├──┤├─┤├──────────────────────┤├──────────────────────────────────┤
# 알고 비용       솔트 (22자)            해시 (31자)
#
# $2b = bcrypt 버전
# $12 = 비용 계수 (2^12 반복)

# 패스워드 검증
start = time.time()
is_valid = bcrypt.checkpw(password, hashed)
elapsed = time.time() - start
print(f"Valid password: {is_valid} ({elapsed:.3f}s)")

# 잘못된 패스워드
is_valid = bcrypt.checkpw(b"wrong-password", hashed)
print(f"Wrong password: {is_valid}")

# 동일한 패스워드에 대해서도 각 해시는 고유합니다 (다른 솔트)
hash2 = bcrypt.hashpw(password, bcrypt.gensalt(rounds=12))
print(f"\nHash 1: {hashed.decode()}")
print(f"Hash 2: {hash2.decode()}")
print(f"Same?   {hashed == hash2}")  # False (다른 솔트)
# 그러나 둘 다 동일한 패스워드를 검증합니다
print(f"Both verify 'correct-horse-battery-staple': "
      f"{bcrypt.checkpw(password, hashed) and bcrypt.checkpw(password, hash2)}")
```

### 5.2 scrypt

scrypt는 **메모리 하드**입니다: 비용 매개변수에 비례하여 많은 양의 메모리가 필요하므로 GPU 및 ASIC 공격에 저항력이 있습니다.

```python
import hashlib
import os
import time

password = b"my-secure-password"
salt = os.urandom(16)

# scrypt 매개변수:
# n = CPU/메모리 비용 (2의 거듭제곱이어야 함). 높을수록 = 느리고 더 많은 메모리.
# r = 블록 크기 (8이 표준)
# p = 병렬성 (1이 표준)
# 메모리 사용량 ≈ 128 * n * r 바이트

start = time.time()
derived = hashlib.scrypt(
    password,
    salt=salt,
    n=2**14,    # 16384 반복
    r=8,        # 블록 크기
    p=1,        # 병렬성
    dklen=32    # 출력 256비트
)
elapsed = time.time() - start

print(f"scrypt derived key: {derived.hex()}")
print(f"Salt:              {salt.hex()}")
print(f"Time:              {elapsed:.3f}s")
print(f"Memory used:       ~{128 * 2**14 * 8 / 1024 / 1024:.0f} MB")

# 검증
derived2 = hashlib.scrypt(password, salt=salt, n=2**14, r=8, p=1, dklen=32)
print(f"Verification:      {derived == derived2}")
```

### 5.3 Argon2 (권장)

Argon2는 2015년 Password Hashing Competition에서 우승했으며 현재 권장되는 알고리즘입니다. 세 가지 변형이 있습니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Argon2 변형                                     │
├──────────────┬──────────────────────────────────────────────────────┤
│ 변형         │ 설명                                                 │
├──────────────┼──────────────────────────────────────────────────────┤
│ Argon2d      │ 데이터 의존적 메모리 액세스. 더 빠르고, GPU 저항성이 │
│              │ 더 높지만 사이드 채널 공격에 취약합니다.             │
│              │ 암호화폐 마이닝에 적합, 패스워드에는 부적합합니다.    │
├──────────────┼──────────────────────────────────────────────────────┤
│ Argon2i      │ 데이터 독립적 메모리 액세스. 사이드 채널 공격에      │
│              │ 저항력이 있습니다. 공유 환경에서 패스워드 해싱에     │
│              │ 더 좋습니다.                                         │
├──────────────┼──────────────────────────────────────────────────────┤
│ Argon2id     │ 하이브리드: 첫 번째 패스는 Argon2i, 이후 패스는     │
│              │ Argon2d입니다. 패스워드 해싱에 권장됩니다.           │
│              │ 두 가지의 장점을 모두 가집니다.                      │
└──────────────┴──────────────────────────────────────────────────────┘
```

```python
# pip install argon2-cffi
from argon2 import PasswordHasher, Type
import time

# 권장 매개변수로 해셔 생성
# 서버의 기능에 따라 time_cost와 memory_cost 조정
# 목표: 해시당 약 0.5-1.0초
ph = PasswordHasher(
    time_cost=3,          # 반복 횟수
    memory_cost=65536,    # KB 단위 메모리 (64 MB)
    parallelism=4,        # 스레드 수
    hash_len=32,          # 출력 해시 길이
    salt_len=16,          # 솔트 길이
    type=Type.ID,         # Argon2id (권장)
)

password = "correct-horse-battery-staple"

# 해시
start = time.time()
hashed = ph.hash(password)
elapsed = time.time() - start

print(f"Argon2id hash: {hashed}")
print(f"Time: {elapsed:.3f}s")
print()

# Argon2 해시 문자열의 구조:
# $argon2id$v=19$m=65536,t=3,p=4$c2FsdDEyMzQ1Njc4$aXaShZ7S2X3yBqBvP5WF4w
# ├───────┤├───┤├──────────────┤├──────────────────┤├──────────────────────────┤
#   알고    버전    매개변수            솔트 (b64)           해시 (b64)

# 검증
try:
    is_valid = ph.verify(hashed, password)
    print(f"Valid password: {is_valid}")
except Exception as e:
    print(f"Verification failed: {e}")

# 잘못된 패스워드
try:
    ph.verify(hashed, "wrong-password")
except Exception as e:
    print(f"Wrong password: {type(e).__name__}")

# 재해시가 필요한지 확인 (매개변수가 변경됨)
if ph.check_needs_rehash(hashed):
    print("Hash needs rehash with updated parameters")
else:
    print("Hash parameters are current")
```

### 5.4 패스워드 해싱 비교

```
┌──────────────────────────────────────────────────────────────────────┐
│              패스워드 해싱 알고리즘 비교                               │
├─────────────┬─────────┬───────────────┬──────────┬──────────────────┤
│ 알고리즘     │ 메모리  │ GPU/ASIC      │ 사이드   │ 권장 사항        │
│             │ 하드?   │ 저항성?       │ 채널 안전│                  │
├─────────────┼─────────┼───────────────┼──────────┼──────────────────┤
│ bcrypt      │ 아니오  │ 보통          │ 예       │ 좋음 (성숙함)    │
│ scrypt      │ 예      │ 좋음          │ 부분적   │ 좋음             │
│ Argon2id    │ 예      │ 최고          │ 예       │ 최고 (선호됨)    │
│ PBKDF2      │ 아니오  │ 나쁨          │ 예       │ 레거시만         │
├─────────────┴─────────┴───────────────┴──────────┴──────────────────┤
│                                                                      │
│ 권장 매개변수 (2025+, 서버에서 약 0.5초 목표):                        │
│ • Argon2id: m=64MB, t=3, p=4                                       │
│ • bcrypt:   cost=12에서 14                                          │
│ • scrypt:   N=2^15, r=8, p=1                                       │
│ • PBKDF2:   SHA-256으로 600,000+ 반복                               │
│                                                                      │
│ OWASP 권장: Argon2id 첫 번째, bcrypt 두 번째.                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.5 완전한 패스워드 저장 시스템

```python
from argon2 import PasswordHasher, Type, exceptions
from dataclasses import dataclass, field
from typing import Optional, Dict
import secrets
import time

@dataclass
class UserRecord:
    username: str
    password_hash: str
    created_at: float
    last_login: Optional[float] = None
    failed_attempts: int = 0
    locked_until: Optional[float] = None

class PasswordStore:
    """Argon2id를 사용한 프로덕션 품질 패스워드 저장소."""

    MAX_ATTEMPTS = 5
    LOCKOUT_SECONDS = 300  # 5분
    MIN_PASSWORD_LENGTH = 12

    def __init__(self):
        self.hasher = PasswordHasher(
            time_cost=3,
            memory_cost=65536,
            parallelism=4,
            hash_len=32,
            salt_len=16,
            type=Type.ID,
        )
        self.users: Dict[str, UserRecord] = {}

    def _validate_password_strength(self, password: str) -> list:
        """기본 강도 요구 사항에 대해 패스워드를 확인합니다."""
        issues = []
        if len(password) < self.MIN_PASSWORD_LENGTH:
            issues.append(f"Password must be at least {self.MIN_PASSWORD_LENGTH} chars")
        if password.lower() == password:
            issues.append("Password must contain uppercase letters")
        if not any(c.isdigit() for c in password):
            issues.append("Password must contain digits")

        # 일반적인 패스워드에 대해 확인 (축약된 목록)
        common = {"password", "123456", "qwerty", "admin", "letmein"}
        if password.lower() in common:
            issues.append("Password is too common")

        return issues

    def register(self, username: str, password: str) -> dict:
        """새 사용자를 등록합니다."""
        if username in self.users:
            return {"success": False, "error": "Username already exists"}

        issues = self._validate_password_strength(password)
        if issues:
            return {"success": False, "error": "Weak password", "issues": issues}

        hashed = self.hasher.hash(password)
        self.users[username] = UserRecord(
            username=username,
            password_hash=hashed,
            created_at=time.time(),
        )
        return {"success": True, "message": f"User {username} registered"}

    def authenticate(self, username: str, password: str) -> dict:
        """사용자를 인증합니다."""
        if username not in self.users:
            # 타이밍 기반 사용자 열거를 방지하기 위해 더미 해시 수행
            self.hasher.hash("dummy-to-waste-time")
            return {"success": False, "error": "Invalid credentials"}

        user = self.users[username]

        # 잠금 확인
        if user.locked_until and time.time() < user.locked_until:
            remaining = int(user.locked_until - time.time())
            return {"success": False, "error": f"Account locked. Try in {remaining}s"}

        try:
            self.hasher.verify(user.password_hash, password)

            # 재해시가 필요한지 확인 (매개변수 업그레이드됨)
            if self.hasher.check_needs_rehash(user.password_hash):
                user.password_hash = self.hasher.hash(password)

            user.failed_attempts = 0
            user.locked_until = None
            user.last_login = time.time()
            return {"success": True, "message": f"Welcome, {username}!"}

        except exceptions.VerifyMismatchError:
            user.failed_attempts += 1
            if user.failed_attempts >= self.MAX_ATTEMPTS:
                user.locked_until = time.time() + self.LOCKOUT_SECONDS
            return {
                "success": False,
                "error": "Invalid credentials",
                "attempts_remaining": max(0, self.MAX_ATTEMPTS - user.failed_attempts)
            }

# 사용
store = PasswordStore()

# 등록
print(store.register("alice", "short"))  # 실패 - 너무 짧음
print(store.register("alice", "MySecureP@ss123"))  # 성공
print(store.register("alice", "AnotherPassword1"))  # 실패 - 이미 존재

# 인증
print(store.authenticate("alice", "MySecureP@ss123"))  # 성공
print(store.authenticate("alice", "wrong-password"))    # 실패
print(store.authenticate("bob", "anything"))            # 사용자를 찾을 수 없음
```

---

## 6. HMAC: 메시지 인증

HMAC (Hash-based Message Authentication Code)은 무결성과 진위성을 모두 제공합니다. 비밀 키를 해시 함수와 결합하여 키를 아는 사람만 생성하고 검증할 수 있는 태그를 생성합니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HMAC 구조                                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  HMAC(K, m) = H((K' ⊕ opad) || H((K' ⊕ ipad) || m))               │
│                                                                      │
│  여기서:                                                             │
│  K  = 비밀 키                                                        │
│  K' = 블록 크기로 패딩/해시된 키                                      │
│  opad = 블록 크기까지 반복된 0x5c                                     │
│  ipad = 블록 크기까지 반복된 0x36                                     │
│  H  = 해시 함수 (SHA-256 등)                                         │
│  || = 연결                                                           │
│  ⊕  = XOR                                                           │
│                                                                      │
│  단계별:                                                             │
│  1. 키 > 블록 크기인 경우: K' = H(K)                                 │
│  2. 그렇지 않으면: K' = 0으로 패딩된 K                                │
│  3. 내부 해시: H((K' ⊕ ipad) || 메시지)                             │
│  4. 외부 해시: H((K' ⊕ opad) || inner_hash)                         │
│                                                                      │
│  왜 단순히 H(key || message)가 아닌가?                               │
│  → Merkle-Damgard 해시(SHA-256)로 길이 확장 공격에 취약합니다.       │
│    HMAC의 중첩 구조는 이를 방지합니다.                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.1 Python에서 HMAC

```python
import hmac
import hashlib
import os
import time

# 비밀 키 생성
key = os.urandom(32)  # 256비트 키

# HMAC 생성
message = b"Transfer $1000 to account 12345"
mac = hmac.new(key, message, hashlib.sha256).hexdigest()
print(f"Message: {message.decode()}")
print(f"HMAC:    {mac}")

# HMAC 검증 (상수 시간 비교)
received_mac = mac
is_valid = hmac.compare_digest(
    mac,
    hmac.new(key, message, hashlib.sha256).hexdigest()
)
print(f"Valid:   {is_valid}")

# 변조된 메시지
tampered = b"Transfer $9999 to account 12345"
is_valid = hmac.compare_digest(
    mac,
    hmac.new(key, tampered, hashlib.sha256).hexdigest()
)
print(f"Tampered valid: {is_valid}")  # False
```

### 6.2 실용: API 요청 서명

```python
import hmac
import hashlib
import json
import time
import os
from urllib.parse import urlencode

class APIRequestSigner:
    """
    HMAC-SHA256을 사용하여 API 요청에 서명합니다.
    AWS Signature V4, Stripe 웹훅 서명 등과 유사합니다.
    """

    def __init__(self, api_key: str, api_secret: bytes):
        self.api_key = api_key
        self.api_secret = api_secret

    def sign_request(self, method: str, path: str,
                      body: dict = None, timestamp: float = None) -> dict:
        """API 요청에 서명하고 헤더를 반환합니다."""
        timestamp = timestamp or time.time()
        ts_str = str(int(timestamp))

        # 정규 요청 문자열 구축
        body_str = json.dumps(body, sort_keys=True, separators=(',', ':')) if body else ""
        canonical = f"{method}\n{path}\n{ts_str}\n{body_str}"

        # HMAC 계산
        signature = hmac.new(
            self.api_secret,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        return {
            "X-API-Key": self.api_key,
            "X-Timestamp": ts_str,
            "X-Signature": signature,
        }

    def verify_request(self, method: str, path: str,
                        headers: dict, body: dict = None,
                        max_age_seconds: int = 300) -> dict:
        """서명된 API 요청을 검증합니다."""
        api_key = headers.get("X-API-Key", "")
        timestamp = headers.get("X-Timestamp", "")
        signature = headers.get("X-Signature", "")

        # API 키 확인
        if api_key != self.api_key:
            return {"valid": False, "error": "Invalid API key"}

        # 타임스탬프 확인 (재생 공격 방지)
        try:
            ts = int(timestamp)
            age = abs(time.time() - ts)
            if age > max_age_seconds:
                return {"valid": False, "error": f"Request too old ({age:.0f}s)"}
        except ValueError:
            return {"valid": False, "error": "Invalid timestamp"}

        # 서명 재계산 및 검증
        body_str = json.dumps(body, sort_keys=True, separators=(',', ':')) if body else ""
        canonical = f"{method}\n{path}\n{timestamp}\n{body_str}"

        expected = hmac.new(
            self.api_secret,
            canonical.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return {"valid": False, "error": "Invalid signature"}

        return {"valid": True, "api_key": api_key}

# 사용
api_key = "ak_live_abc123"
api_secret = os.urandom(32)

signer = APIRequestSigner(api_key, api_secret)

# 요청 서명
headers = signer.sign_request(
    method="POST",
    path="/api/v1/transfers",
    body={"amount": 1000, "currency": "USD", "to": "acct_xyz"}
)
print("Signed Request Headers:")
for k, v in headers.items():
    print(f"  {k}: {v}")

# 요청 검증
result = signer.verify_request(
    method="POST",
    path="/api/v1/transfers",
    headers=headers,
    body={"amount": 1000, "currency": "USD", "to": "acct_xyz"}
)
print(f"\nVerification: {result}")

# 변조된 본문
result = signer.verify_request(
    method="POST",
    path="/api/v1/transfers",
    headers=headers,
    body={"amount": 9999, "currency": "USD", "to": "acct_xyz"}
)
print(f"Tampered body: {result}")
```

### 6.3 웹훅 서명 검증

```python
import hmac
import hashlib

def verify_webhook_signature(
    payload: bytes,
    signature_header: str,
    secret: bytes,
    tolerance_seconds: int = 300,
) -> bool:
    """
    웹훅 서명을 검증합니다 (Stripe 스타일).
    헤더 형식: t=<timestamp>,v1=<signature>
    """
    # 헤더 파싱
    elements = {}
    for part in signature_header.split(","):
        key, _, value = part.partition("=")
        elements[key.strip()] = value.strip()

    timestamp = elements.get("t", "")
    received_sig = elements.get("v1", "")

    if not timestamp or not received_sig:
        return False

    # 타임스탬프 신선도 확인
    import time
    try:
        ts = int(timestamp)
        if abs(time.time() - ts) > tolerance_seconds:
            return False
    except ValueError:
        return False

    # 예상 서명 계산
    signed_payload = f"{timestamp}.".encode() + payload
    expected_sig = hmac.new(
        secret, signed_payload, hashlib.sha256
    ).hexdigest()

    # 상수 시간 비교
    return hmac.compare_digest(expected_sig, received_sig)

# 웹훅 시뮬레이션
import time
webhook_secret = b"whsec_test_secret_key_123"
payload = b'{"event":"payment.success","amount":5000}'
ts = str(int(time.time()))
sig = hmac.new(
    webhook_secret,
    f"{ts}.".encode() + payload,
    hashlib.sha256
).hexdigest()

header = f"t={ts},v1={sig}"

is_valid = verify_webhook_signature(payload, header, webhook_secret)
print(f"Webhook signature valid: {is_valid}")

# 변조된 페이로드
is_valid = verify_webhook_signature(
    b'{"event":"payment.success","amount":50000}',  # 금액 변경
    header,
    webhook_secret
)
print(f"Tampered webhook valid: {is_valid}")
```

---

## 7. Merkle 트리

Merkle 트리는 각 리프 노드가 데이터 블록의 해시이고 각 내부 노드가 자식의 해시인 이진 해시 트리입니다. 루트 해시 (Merkle 루트)는 모든 데이터를 요약합니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                       Merkle 트리                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                        루트 해시                                      │
│                      H(H12 || H34)                                  │
│                       /        \                                     │
│                      /          \                                    │
│                   H12            H34                                 │
│                H(H1||H2)      H(H3||H4)                             │
│                /      \        /      \                              │
│              H1       H2     H3       H4                            │
│            H(D1)    H(D2)  H(D3)   H(D4)                           │
│              |        |      |        |                              │
│            Data1   Data2   Data3   Data4                            │
│                                                                      │
│  속성:                                                               │
│  • 루트 해시는 모든 데이터 블록이 변경되면 변경됩니다                 │
│  • O(log n) 해시로 단일 블록을 검증할 수 있습니다 (Merkle 증명)      │
│  • 사용처: Git, Bitcoin, Certificate Transparency, IPFS            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.1 Merkle 트리 구현

```python
import hashlib
from typing import List, Optional
from dataclasses import dataclass

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

@dataclass
class MerkleNode:
    hash: bytes
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None

class MerkleTree:
    """증명 생성/검증이 포함된 완전한 Merkle 트리 구현."""

    def __init__(self, data_blocks: List[bytes]):
        if not data_blocks:
            raise ValueError("Cannot create Merkle tree from empty data")

        self.leaves = [MerkleNode(hash=sha256(block)) for block in data_blocks]
        self.root = self._build_tree(self.leaves)

    def _build_tree(self, nodes: List[MerkleNode]) -> MerkleNode:
        """트리를 상향식으로 구축합니다."""
        if len(nodes) == 1:
            return nodes[0]

        # 홀수 개의 노드인 경우 마지막 노드를 복제합니다
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])

        parent_nodes = []
        for i in range(0, len(nodes), 2):
            combined = nodes[i].hash + nodes[i + 1].hash
            parent = MerkleNode(
                hash=sha256(combined),
                left=nodes[i],
                right=nodes[i + 1],
            )
            parent_nodes.append(parent)

        return self._build_tree(parent_nodes)

    @property
    def root_hash(self) -> str:
        return self.root.hash.hex()

    def get_proof(self, index: int) -> List[tuple]:
        """
        주어진 인덱스의 리프에 대한 Merkle 증명을 생성합니다.
        위치가 'left' 또는 'right'인 (hash, position) 튜플 목록을 반환합니다.
        """
        if index < 0 or index >= len(self.leaves):
            raise IndexError(f"Index {index} out of range")

        proof = []
        nodes = self.leaves[:]

        # 홀수인 경우 마지막 복제
        if len(nodes) % 2 == 1:
            nodes.append(nodes[-1])

        current_index = index

        while len(nodes) > 1:
            next_level = []

            for i in range(0, len(nodes), 2):
                if i == current_index or i + 1 == current_index:
                    # 이것은 우리 노드를 포함하는 쌍입니다
                    if current_index % 2 == 0:
                        # 우리 노드는 왼쪽에 있고, 형제는 오른쪽에 있습니다
                        proof.append((nodes[i + 1].hash, "right"))
                    else:
                        # 우리 노드는 오른쪽에 있고, 형제는 왼쪽에 있습니다
                        proof.append((nodes[i].hash, "left"))

                combined = nodes[i].hash + nodes[i + 1].hash
                parent = MerkleNode(hash=sha256(combined))
                next_level.append(parent)

            current_index = current_index // 2
            nodes = next_level

            if len(nodes) > 1 and len(nodes) % 2 == 1:
                nodes.append(nodes[-1])

        return proof

    @staticmethod
    def verify_proof(data: bytes, proof: List[tuple], root_hash: str) -> bool:
        """Merkle 증명을 검증합니다."""
        current_hash = sha256(data)

        for sibling_hash, position in proof:
            if position == "left":
                combined = sibling_hash + current_hash
            else:
                combined = current_hash + sibling_hash
            current_hash = sha256(combined)

        return current_hash.hex() == root_hash

# Merkle 트리 구축
data_blocks = [
    b"Transaction: Alice -> Bob $100",
    b"Transaction: Bob -> Charlie $50",
    b"Transaction: Charlie -> Dave $25",
    b"Transaction: Dave -> Alice $75",
]

tree = MerkleTree(data_blocks)
print(f"Merkle root: {tree.root_hash}")
print(f"Leaves: {len(tree.leaves)}")

# 인덱스 1의 트랜잭션에 대한 증명 생성
proof = tree.get_proof(1)
print(f"\nProof for block 1 ({len(proof)} nodes):")
for h, pos in proof:
    print(f"  {pos}: {h.hex()[:32]}...")

# 증명 검증
is_valid = MerkleTree.verify_proof(
    data_blocks[1], proof, tree.root_hash
)
print(f"\nProof valid: {is_valid}")

# 변조된 데이터는 실패
is_valid = MerkleTree.verify_proof(
    b"Transaction: Bob -> Charlie $5000",  # 변조됨!
    proof, tree.root_hash
)
print(f"Tampered proof valid: {is_valid}")

# 효율성 표시: 1M 블록 중 1개를 검증하려면 약 20개의 해시만 필요
import math
n_blocks = 1_000_000
proof_size = math.ceil(math.log2(n_blocks))
print(f"\nFor {n_blocks:,} blocks:")
print(f"  Proof size: {proof_size} hashes ({proof_size * 32} bytes)")
print(f"  vs checking all: {n_blocks * 32:,} bytes")
print(f"  Efficiency: {n_blocks * 32 / (proof_size * 32):,.0f}x smaller")
```

---

## 8. 내용 주소 지정 저장소

내용 주소 지정 저장소 (CAS)는 데이터의 해시를 주소/키로 사용합니다. 이것은 Git, IPFS, Docker 레이어 및 많은 중복 제거 시스템의 기반입니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│              내용 주소 지정 저장소 (CAS)                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  기존 저장소:                                                        │
│    filename → data                                                  │
│    /docs/report.pdf → [파일 내용]                                   │
│                                                                      │
│  내용 주소 지정 저장소:                                              │
│    hash(data) → data                                                │
│    sha256:a3f2b8... → [파일 내용]                                   │
│                                                                      │
│  이점:                                                               │
│  • 자동 중복 제거 (동일한 내용 = 동일한 해시 = 한 번만 저장)         │
│  • 내장 무결성 검증 (해시가 주소입니다)                              │
│  • 설계상 불변 (내용을 변경하면 주소가 변경됨)                        │
│  • 캐시 친화적 (주어진 주소에 대해 내용이 절대 변경되지 않음)         │
│                                                                      │
│  사용처:                                                             │
│  • Git (SHA-1/SHA-256로 주소 지정된 blob 객체)                      │
│  • Docker (SHA-256로 주소 지정된 이미지 레이어)                      │
│  • IPFS (multihash로 주소 지정된 블록)                               │
│  • Nix 패키지 매니저                                                 │
│  • 콘텐츠 전송 네트워크                                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.1 CAS 구현

```python
import hashlib
import json
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class CASStats:
    total_objects: int = 0
    total_bytes: int = 0
    deduplicated_bytes: int = 0

class ContentAddressableStore:
    """
    SHA-256을 사용한 간단한 내용 주소 지정 저장소.
    Git이 객체를 저장하는 방법과 유사합니다.
    """

    def __init__(self, store_dir: str):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _hash_content(self, data: bytes) -> str:
        """내용의 SHA-256 해시를 계산합니다."""
        return hashlib.sha256(data).hexdigest()

    def _object_path(self, content_hash: str) -> Path:
        """
        객체의 파일 시스템 경로를 가져옵니다.
        단일 디렉터리에 너무 많은 파일을 피하기 위해
        처음 2자를 디렉터리로 사용합니다 (Git처럼).
        """
        return self.store_dir / content_hash[:2] / content_hash[2:]

    def put(self, data: bytes) -> str:
        """
        데이터를 저장하고 내용 해시를 반환합니다.
        데이터가 이미 존재하는 경우 이것은 no-op입니다 (중복 제거).
        """
        content_hash = self._hash_content(data)
        obj_path = self._object_path(content_hash)

        if not obj_path.exists():
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            obj_path.write_bytes(data)

        return content_hash

    def get(self, content_hash: str) -> Optional[bytes]:
        """내용 해시로 데이터를 검색합니다."""
        obj_path = self._object_path(content_hash)
        if not obj_path.exists():
            return None

        data = obj_path.read_bytes()

        # 읽기 시 무결성 검증
        actual_hash = self._hash_content(data)
        if actual_hash != content_hash:
            raise RuntimeError(
                f"Integrity error! Expected {content_hash}, got {actual_hash}"
            )

        return data

    def exists(self, content_hash: str) -> bool:
        """객체가 존재하는지 확인합니다."""
        return self._object_path(content_hash).exists()

    def delete(self, content_hash: str) -> bool:
        """객체를 삭제합니다 (프로덕션에서 주의해서 사용)."""
        obj_path = self._object_path(content_hash)
        if obj_path.exists():
            obj_path.unlink()
            # 빈 디렉터리 정리
            try:
                obj_path.parent.rmdir()
            except OSError:
                pass
            return True
        return False

    def stats(self) -> CASStats:
        """저장소 통계를 계산합니다."""
        stats = CASStats()
        for subdir in self.store_dir.iterdir():
            if subdir.is_dir():
                for obj_file in subdir.iterdir():
                    stats.total_objects += 1
                    stats.total_bytes += obj_file.stat().st_size
        return stats

# 사용
store = ContentAddressableStore("/tmp/cas_demo")

# 일부 데이터 저장
data1 = b"Hello, content-addressable world!"
data2 = b"Another piece of data"
data3 = b"Hello, content-addressable world!"  # data1의 중복!

hash1 = store.put(data1)
hash2 = store.put(data2)
hash3 = store.put(data3)

print(f"Data 1 hash: {hash1}")
print(f"Data 2 hash: {hash2}")
print(f"Data 3 hash: {hash3}")
print(f"Data 1 == Data 3 hash? {hash1 == hash3}")  # True - 중복 제거!

# 해시로 검색
retrieved = store.get(hash1)
print(f"\nRetrieved: {retrieved.decode()}")
print(f"Integrity: {retrieved == data1}")

# 통계
stats = store.stats()
print(f"\nStore stats:")
print(f"  Objects: {stats.total_objects}")  # 2, 3이 아님 (중복 제거!)
print(f"  Total bytes: {stats.total_bytes}")

# 정리
import shutil
shutil.rmtree("/tmp/cas_demo")
```

### 8.2 Git 스타일 객체 저장소

```python
import hashlib
import zlib
import os
from pathlib import Path

class GitObjectStore:
    """
    단순화된 Git 객체 저장소.
    Git은 객체를 다음과 같이 저장합니다: 헤더 + 내용, zlib로 압축됨.
    헤더 형식: "<type> <size>\0"
    """

    def __init__(self, git_dir: str):
        self.objects_dir = Path(git_dir) / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    def hash_object(self, data: bytes, obj_type: str = "blob") -> str:
        """객체를 해시합니다 ('git hash-object'처럼)."""
        header = f"{obj_type} {len(data)}\0".encode()
        full_content = header + data
        return hashlib.sha1(full_content).hexdigest()  # Git은 SHA-1 사용 (SHA-256으로 전환 중)

    def write_object(self, data: bytes, obj_type: str = "blob") -> str:
        """저장소에 객체를 작성합니다 ('git hash-object -w'처럼)."""
        header = f"{obj_type} {len(data)}\0".encode()
        full_content = header + data
        obj_hash = hashlib.sha1(full_content).hexdigest()

        # 압축하여 저장
        obj_path = self.objects_dir / obj_hash[:2] / obj_hash[2:]
        if not obj_path.exists():
            obj_path.parent.mkdir(parents=True, exist_ok=True)
            compressed = zlib.compress(full_content)
            obj_path.write_bytes(compressed)

        return obj_hash

    def read_object(self, obj_hash: str) -> tuple:
        """객체를 읽습니다 ('git cat-file'처럼)."""
        obj_path = self.objects_dir / obj_hash[:2] / obj_hash[2:]
        if not obj_path.exists():
            raise FileNotFoundError(f"Object {obj_hash} not found")

        compressed = obj_path.read_bytes()
        full_content = zlib.decompress(compressed)

        # 헤더 파싱
        null_pos = full_content.index(b'\0')
        header = full_content[:null_pos].decode()
        obj_type, size = header.split(' ')
        data = full_content[null_pos + 1:]

        assert len(data) == int(size), "Size mismatch"
        return obj_type, data

# 사용
store = GitObjectStore("/tmp/git_demo/.git")

# blob 저장 (파일 내용)
content = b"print('Hello, World!')\n"
blob_hash = store.write_object(content, "blob")
print(f"Blob hash: {blob_hash}")

# 다시 읽기
obj_type, data = store.read_object(blob_hash)
print(f"Type: {obj_type}")
print(f"Content: {data.decode()}", end="")

# 트리 저장 (디렉터리 목록) - 단순화됨
tree_content = f"100644 hello.py\0".encode() + bytes.fromhex(blob_hash)
tree_hash = store.write_object(tree_content, "tree")
print(f"\nTree hash: {tree_hash}")

# 정리
import shutil
shutil.rmtree("/tmp/git_demo")
```

---

## 9. 타이밍 공격과 상수 시간 비교

### 9.1 문제

두 문자열을 문자별로 비교할 때 소요되는 시간은 첫 번째 차이가 발생하는 위치에 따라 달라집니다. 공격자는 응답 시간을 측정하여 비밀 값의 몇 자를 올바르게 추측했는지 확인할 수 있습니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                    문자열 비교에 대한 타이밍 공격                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  비밀 MAC: "a3f2b8c9d1e4"                                           │
│                                                                      │
│  시도 1: "x3f2b8c9d1e4"  → 위치 0에서 실패  → ~100 ns              │
│  시도 2: "a4f2b8c9d1e4"  → 위치 1에서 실패  → ~110 ns              │
│  시도 3: "a3g2b8c9d1e4"  → 위치 2에서 실패  → ~120 ns              │
│                                                                      │
│  공격자는 각 올바른 문자가 약 10 ns를 추가한다는 것을 알아챕니다.     │
│  각 위치의 모든 값을 시도함으로써 전체 MAC을 한 번에 한 문자씩         │
│  복구할 수 있습니다: O(16^n) 대신 O(n × 16).                         │
│                                                                      │
│  32자 16진수 MAC의 경우:                                             │
│  무차별 대입: 16^32 ≈ 3.4 × 10^38 시도                             │
│  타이밍 공격: 32 × 16 = 512 시도 (!)                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.2 취약한 vs 안전한 비교

```python
import hmac
import hashlib
import time
import os

# 취약: 조기 종료 문자열 비교
def insecure_compare(a: str, b: str) -> bool:
    """
    사용하지 마십시오! 타이밍 공격에 취약합니다.
    불일치가 발견되는 즉시 False를 반환합니다.
    """
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            return False  # 조기 종료는 정보를 누출합니다!
    return True

# 안전: 상수 시간 비교
def secure_compare(a: str, b: str) -> bool:
    """
    상수 시간 비교. 문자열이 어디서 (또는 만약) 다른지에 관계없이
    동일한 시간이 걸립니다.
    """
    return hmac.compare_digest(a, b)

# 타이밍 차이 시연
secret_mac = hashlib.sha256(b"secret-key" + b"message").hexdigest()

# 올바른 접두사 길이가 증가하는 테스트 MAC 생성
test_macs = []
for i in range(0, len(secret_mac), 4):
    correct_prefix = secret_mac[:i]
    wrong_suffix = "0" * (len(secret_mac) - i)
    test_macs.append(correct_prefix + wrong_suffix)

print("Timing Attack Demonstration")
print("=" * 60)
print(f"Secret MAC: {secret_mac[:32]}...")
print()

# 안전하지 않은 비교 시간 측정
print("INSECURE comparison (early exit):")
for mac in test_macs[:8]:
    correct_chars = sum(a == b for a, b in zip(mac, secret_mac))
    times = []
    for _ in range(10000):
        start = time.perf_counter_ns()
        insecure_compare(mac, secret_mac)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  {correct_chars:2d} correct chars → avg {avg:6.0f} ns")

print()

# 안전한 비교 시간 측정
print("SECURE comparison (constant-time):")
for mac in test_macs[:8]:
    correct_chars = sum(a == b for a, b in zip(mac, secret_mac))
    times = []
    for _ in range(10000):
        start = time.perf_counter_ns()
        secure_compare(mac, secret_mac)
        elapsed = time.perf_counter_ns() - start
        times.append(elapsed)
    avg = sum(times) / len(times)
    print(f"  {correct_chars:2d} correct chars → avg {avg:6.0f} ns")

print()
print("Note: Secure comparison time should be roughly constant")
print("regardless of how many characters match.")
```

### 9.3 타이밍 안전 코드를 위한 규칙

```
┌─────────────────────────────────────────────────────────────────────┐
│               상수 시간 작업을 위한 규칙                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 비밀 (MAC, 토큰, 패스워드)을 비교할 때 ==를 절대 사용하지 마십시오│
│     사용: hmac.compare_digest() 또는 secrets.compare_digest()       │
│                                                                      │
│  2. 비밀 데이터를 기반으로 조기에 반환하지 마십시오                   │
│     나쁨:  if mac[i] != expected[i]: return False                   │
│     좋음: result |= (mac[i] ^ expected[i])   # 차이 누적            │
│                                                                      │
│  3. 비밀 데이터에 대해 분기하지 마십시오                              │
│     나쁨:  if secret_key[0] == 'a': ...                             │
│     좋음: 상수 시간 선택 / 마스킹 사용                               │
│                                                                      │
│  4. 비밀 값으로 배열을 인덱싱하지 마십시오                            │
│     나쁨:  table[secret_byte]   # 캐시 타이밍 누출                  │
│     좋음: 상수 시간 룩업 테이블 사용                                 │
│                                                                      │
│  5. 복호화하기 전에 항상 MAC을 검증하십시오                          │
│     이것은 패딩 오라클 공격을 방지합니다 (Encrypt-then-MAC)         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 10. 해시 함수 공격과 완화

### 10.1 알려진 공격

```
┌─────────────────────────────────────────────────────────────────────┐
│                해시 함수 공격 요약                                    │
├────────────────────┬────────────────────────────────────────────────┤
│ 공격               │ 설명 및 상태                                   │
├────────────────────┼────────────────────────────────────────────────┤
│ 생일 공격          │ O(2^(n/2))로 모든 충돌 찾기                    │
│                    │ SHA-256: 2^128 (안전), MD5: 2^64 (손상됨)     │
├────────────────────┼────────────────────────────────────────────────┤
│ 길이 확장          │ H(m)이 주어지면 m을 모르고                     │
│                    │ H(m||pad||suffix)를 계산합니다.                │
│                    │ SHA-256, SHA-512에 영향을 미칩니다.            │
│                    │ 해당 없음: SHA-3, BLAKE2, HMAC, SHA-512/256   │
├────────────────────┼────────────────────────────────────────────────┤
│ 레인보우 테이블    │ 사전 계산된 해시-패스워드 룩업 테이블.         │
│                    │ 패스워드 솔팅으로 방어됩니다.                  │
├────────────────────┼────────────────────────────────────────────────┤
│ 사전 공격          │ 해시에 대해 일반적인 패스워드를 시도합니다.    │
│                    │ 느린 패스워드 해시(Argon2)로 완화됩니다.       │
├────────────────────┼────────────────────────────────────────────────┤
│ GPU 무차별 대입    │ GPU는 초당 수십억 개의 해시를 계산할 수        │
│                    │ 있습니다. 메모리 하드 해시(Argon2id)로         │
│                    │ 완화됩니다.                                    │
├────────────────────┼────────────────────────────────────────────────┤
│ SHAttered (2017)   │ Google이 발견한 실용적인 SHA-1 충돌.          │
│                    │ 동일한 SHA-1이지만 다른 내용의 두 PDF         │
│                    │ SHA-1은 이제 보안에 손상된 것으로 간주됩니다.  │
├────────────────────┼────────────────────────────────────────────────┤
│ 다중 충돌          │ 동일한 해시를 가진 많은 메시지를 찾습니다.     │
│                    │ 반복 해시에 대해 예상보다 쉽습니다.            │
│                    │ SHA-3의 스펀지 구조는 더 저항력이 있습니다.    │
└────────────────────┴────────────────────────────────────────────────┘
```

### 10.2 길이 확장 공격

```python
# 길이 확장 공격 시연
# 이것이 H(key || message) 대신 HMAC을 사용해야 하는 이유입니다
import hashlib
import struct

def sha256_pad(message_len: int) -> bytes:
    """주어진 길이의 메시지에 대한 SHA-256 패딩을 계산합니다."""
    bit_len = message_len * 8
    # 패딩: 1비트, 그 다음 0, 그 다음 64비트 길이
    padding = b'\x80'
    padding += b'\x00' * ((56 - (message_len + 1) % 64) % 64)
    padding += struct.pack('>Q', bit_len)
    return padding

# 취약점: H(secret || message)는 안전한 MAC이 아닙니다
#
# 공격자가 다음을 알고 있는 경우:
#   - H(secret || message)  (MAC)
#   - len(secret)           (또는 추측할 수 있음)
#   - message               (원본 메시지)
#
# 비밀을 모르고도 H(secret || message || padding || extension)을
# 계산할 수 있습니다!
#
# 이것은 SHA-256이 블록 단위로 데이터를 처리하고
# (secret || message || padding) 처리 후의 해시 상태가
# 정확히 공개 해시 값이기 때문입니다. 공격자는 거기서부터
# 해싱을 재개할 수 있습니다.

print("Length-Extension Attack Concept:")
print("=" * 60)
print()
print("VULNERABLE construction: MAC = SHA-256(secret || message)")
print("An attacker who knows MAC and len(secret) can extend the message.")
print()
print("SAFE alternatives:")
print("  1. HMAC-SHA256(key, message)     — HMAC construction")
print("  2. SHA-3-256(key || message)      — SHA-3 is not vulnerable")
print("  3. BLAKE2b(message, key=key)      — Built-in keying")
print("  4. SHA-512/256(key || message)    — Truncated hash")
```

---

## 11. 연습 문제

### 연습 1: 해시 탐색기 (초급)

다음을 수행하는 Python 프로그램을 작성하십시오:
1. 파일 이름을 입력으로 받습니다
2. SHA-256, SHA-3-256, BLAKE2b-256, BLAKE3 해시를 계산합니다
3. 모든 해시와 각각에 소요된 시간을 표시합니다
4. 100 MB 파일에서 각 알고리즘의 속도를 비교합니다
5. 해시를 비교하여 파일이 수정되지 않았는지 확인합니다

### 연습 2: 패스워드 크래커 방어 (중급)

패스워드 크래킹 시뮬레이션을 구현하십시오:
1. 다음을 사용하여 해시된 패스워드가 있는 1000개의 "사용자 계정" 목록을 만듭니다:
   a. 일반 SHA-256 (솔트 없음)
   b. 고유한 솔트가 있는 SHA-256
   c. bcrypt (cost=10)
   d. Argon2id (기본 매개변수)
2. 10,000개의 일반적인 패스워드 사전을 사용하여 모든 패스워드 크래킹을 시도합니다
3. 각 해싱 방법에 필요한 시간을 측정하고 비교합니다
4. 크래킹된 것과 크래킹되지 않은 것의 백분율을 보여주는 보고서를 생성합니다

### 연습 3: Merkle 트리 파일 검증기 (중급)

Merkle 트리를 사용하여 파일 무결성 검증 시스템을 구축하십시오:
1. 디렉터리가 주어지면 모든 파일에 대한 Merkle 트리를 계산합니다 (경로별로 정렬)
2. Merkle 루트와 트리 구조를 저장합니다
3. 나중에 O(log n) 해시만 다시 계산하여 개별 파일을 검증합니다
4. 독립적으로 검증할 수 있는 Merkle 증명을 생성합니다
5. 파일 추가 및 삭제를 효율적으로 처리합니다

### 연습 4: HMAC 기반 API 인증 (중급)

완전한 API 인증 시스템을 구현하십시오:
1. 서버는 각 클라이언트에게 API 키 + 비밀을 발급합니다
2. 클라이언트는 HMAC-SHA256으로 각 요청에 서명합니다:
   - 포함: 메서드, 경로, 타임스탬프, 본문 해시, nonce
3. 서버 검증: 유효한 서명, 신선한 타임스탬프 (5분 이내), 사용되지 않은 nonce
4. nonce 캐시를 사용하여 재생 공격 방지 구현
5. 클라이언트와 서버 간의 시계 차이 처리

### 연습 5: 내용 주소 지정 파일 동기화 (고급)

단순화된 파일 동기화 시스템을 구축하십시오 (내용 주소 지정이 있는 rsync처럼):
1. 양쪽 모두 파일의 CAS를 유지합니다
2. 동기화하려면 내용 해시 목록만 교환합니다
3. 상대방이 누락한 블록만 전송합니다
4. 수신된 모든 블록의 무결성을 검증합니다
5. Merkle 트리를 사용하여 큰 파일의 어느 부분이 다른지 효율적으로 감지합니다

### 연습 6: 타이밍 공격 실험실 (고급)

제어된 타이밍 공격을 구축하십시오:
1. 안전하지 않은 비교 (`==`)를 사용하여 API 토큰을 검증하는 서버를 만듭니다
2. 응답 시간을 측정하여 토큰을 한 번에 한 문자씩 추측하는 클라이언트를 작성합니다
3. 공격이 전체 토큰을 복구한다는 것을 시연합니다
4. `hmac.compare_digest()`를 사용하도록 서버를 수정합니다
5. 공격이 더 이상 작동하지 않음을 보여줍니다
6. 다른 타이밍 사이드 채널 (캐시 타이밍, 전력 분석) 논의

---

## 12. 참고 문헌

- Aumasson, J.P. (2017). *Serious Cryptography*. No Starch Press.
- NIST FIPS 180-4: Secure Hash Standard (SHA-2)
- NIST FIPS 202: SHA-3 Standard
- RFC 7693: The BLAKE2 Cryptographic Hash
- BLAKE3 Specification - https://github.com/BLAKE3-team/BLAKE3-specs
- RFC 2104: HMAC: Keyed-Hashing for Message Authentication
- OWASP Password Storage Cheat Sheet - https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html
- Argon2 Reference - https://github.com/P-H-C/phc-winner-argon2
- Merkle, R.C. (1979). "A Certified Digital Signature"

---

**이전**: [02. 암호학 기초](./02_Cryptography_Basics.md) | **다음**: [04. TLS/SSL과 공개키 인프라](./04_TLS_and_PKI.md)
