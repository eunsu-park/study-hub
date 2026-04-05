# 암호학 이론 (Cryptography Theory)

## 개요

이 토픽은 수론적 기초부터 고전적·포스트양자 알고리즘, 영지식 증명과 동형 암호 같은 고급 프로토콜까지 현대 암호학을 엄밀하게 다룹니다. Security 토픽이 *응용* 암호학(TLS, JWT, 해싱 등 라이브러리 사용)을 다루는 반면, 이 토픽은 그 도구들이 *왜* 작동하는지 — 수학적 구조, 보안 증명, 설계 원칙을 설명합니다.

## 선수지식

- 모듈러 산술 기초 (최대공약수, 모듈러 역원)
- Python 중급 수준 (함수, 클래스)
- 수학 표기법에 대한 친숙함 (증명, 군론은 도움이 되지만 필수는 아님)
- 권장: Security L02 (암호학 기초) — 실습적 맥락 파악

## 레슨 구성

### 기초 (Foundations)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|--------|--------|-----------|------|
| [01_Number_Theory_Foundations.md](./01_Number_Theory_Foundations.md) | ⭐⭐ | 모듈러 산술, GCD, 오일러 정리, CRT, 소수 판별 | 이후 모든 레슨의 수학 선수지식 |
| [02_Symmetric_Ciphers.md](./02_Symmetric_Ciphers.md) | ⭐⭐ | AES 내부 구조 (SubBytes, ShiftRows, MixColumns, AddRoundKey), DES, Feistel 네트워크 | 블록 암호 구성 |
| [03_Block_Cipher_Modes.md](./03_Block_Cipher_Modes.md) | ⭐⭐⭐ | ECB, CBC, CTR, GCM, CCM, 패딩 오라클 | 인증 암호화 |
| [04_Hash_Functions.md](./04_Hash_Functions.md) | ⭐⭐⭐ | Merkle-Damgård, SHA-256 내부, 생일 공격, HMAC, SHA-3/Keccak | 충돌 저항성 |

### 공개키 암호 (Public-Key Cryptography)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|--------|--------|-----------|------|
| [05_RSA_Cryptosystem.md](./05_RSA_Cryptosystem.md) | ⭐⭐⭐ | RSA 키 생성, 암호화/복호화 증명, OAEP, CCA 보안, 인수분해 공격 | 정확성의 수학적 증명 |
| [06_Elliptic_Curve_Cryptography.md](./06_Elliptic_Curve_Cryptography.md) | ⭐⭐⭐⭐ | 유한체 위의 타원곡선, 점 덧셈, 스칼라 곱셈, ECDLP, Curve25519 | 기하학적 직관 + 대수 |
| [07_Digital_Signatures.md](./07_Digital_Signatures.md) | ⭐⭐⭐ | RSA-PSS, ECDSA, EdDSA, Schnorr 서명, 보안 모델 | 부인 방지 증명 |
| [08_Key_Exchange.md](./08_Key_Exchange.md) | ⭐⭐⭐ | Diffie-Hellman, ECDH, 인증된 키 교환, 전방향 비밀성, Signal 프로토콜 | 중간자 공격 방지 |

### 인프라 (Infrastructure)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|--------|--------|-----------|------|
| [09_PKI_and_Certificates.md](./09_PKI_and_Certificates.md) | ⭐⭐⭐ | X.509, 인증서 체인, CRL/OCSP, Certificate Transparency, 신뢰의 웹 | 신뢰 모델 비교 |

### 포스트양자 암호 (Post-Quantum Cryptography)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|--------|--------|-----------|------|
| [10_Lattice_Based_Cryptography.md](./10_Lattice_Based_Cryptography.md) | ⭐⭐⭐⭐ | 격자, SVP/CVP, LWE, NTRU, CRYSTALS-Kyber, CRYSTALS-Dilithium | NIST PQC 표준 |
| [11_Post_Quantum_Cryptography.md](./11_Post_Quantum_Cryptography.md) | ⭐⭐⭐⭐ | 코드 기반(McEliece), 해시 기반(SPHINCS+), 아이소제니 기반, NIST PQC 일정 | 마이그레이션 전략 |

### 고급 주제 (Advanced Topics)

| 파일명 | 난이도 | 핵심 주제 | 비고 |
|--------|--------|-----------|------|
| [12_Zero_Knowledge_Proofs.md](./12_Zero_Knowledge_Proofs.md) | ⭐⭐⭐⭐ | 대화형 증명, Schnorr ZKP, zk-SNARKs, zk-STARKs, 응용 | 블록체인과 프라이버시 |
| [13_Homomorphic_Encryption.md](./13_Homomorphic_Encryption.md) | ⭐⭐⭐⭐ | 부분/준/완전 동형 암호, BFV/CKKS, 노이즈 성장, 부트스트래핑 | 프라이버시 보존 연산 |
| [14_Applied_Cryptographic_Protocols.md](./14_Applied_Cryptographic_Protocols.md) | ⭐⭐⭐⭐ | TLS 1.3 핸드셰이크, Signal/Double Ratchet, MPC, 비밀 분산, 임계값 서명 | 실제 프로토콜 분석 |

## 학습 경로

```
기초 (L01-L04)              공개키 (L05-L08)            인프라 (L09)
       │                            │                         │
       ▼                            ▼                         ▼
  수론 기초                  RSA / ECC / 서명           PKI / 인증서
  대칭 암호                  키 교환                    신뢰 모델
  모드 / 해싱                전방향 비밀성
       │                            │                         │
       └────────────────────────────┴─────────────────────────┘
                                    │
                                    ▼
                      포스트양자 (L10-L11)
                      격자 / 코드 / 해시 기반
                                    │
                                    ▼
                      고급 (L12-L14)
                      ZKP, 동형 암호, 프로토콜
```

## 관련 토픽

| 토픽 | 관계 |
|------|------|
| Security | Security는 *응용* 암호학(라이브러리 사용, TLS 설정)을, 이 토픽은 *이론*(알고리즘이 작동하는 이유)을 다룸 |
| Math_for_AI | 선형대수, 확률론 기초 공유; 이 토픽은 수론, 유한체를 추가 |
| Formal_Languages | 계산 복잡도(P, NP)가 암호학적 난해성 가정의 기반 |
| Networking | TLS 핸드셰이크(L14)가 Networking L10/L12와 연결 |

## 예제 코드

이 토픽의 예제 코드는 `examples/Cryptography_Theory/`에 있습니다.

## 총 구성

- **14 레슨** (기초 4 + 공개키 4 + 인프라 1 + 포스트양자 2 + 고급 3)
- **난이도 범위**: ⭐⭐ ~ ⭐⭐⭐⭐
- **언어**: Python (주)
- **주요 라이브러리**: 내장(이론용), cryptography(검증용), pycryptodome(저수준)
