# TLS/SSL과 공개 키 인프라

**이전**: [03. 해싱과 데이터 무결성](./03_Hashing_and_Integrity.md) | **다음**: [05. 인증 시스템](./05_Authentication.md)

---

전송 계층 보안(TLS)은 HTTPS를 가능하게 하는 프로토콜입니다. 이것은 사실상 모든 웹 트래픽, 이메일, 메시징, VPN, API 통신을 보호합니다. 이 레슨은 TLS 1.3, 인증서 신뢰 모델(PKI), 실용적인 OpenSSL 및 Python 사용, 상호 TLS, 보안을 약화시키는 일반적인 잘못된 구성에 대한 깊이 있는 기술적 안내를 제공합니다.

**난이도**: ⭐⭐⭐⭐

## 학습 목표(Learning Objectives)

이 레슨을 완료하면 다음을 할 수 있습니다:

1. TLS 1.3 핸드셰이크를 단계별로 완전히 이해하기
2. 인증서 체인과 신뢰 웹 설명하기
3. OpenSSL과 Python을 사용하여 X.509 인증서 작업하기
4. 인증 기관과 인증서 발급 이해하기
5. ACME 프로토콜로 Let's Encrypt 구성하기
6. 인증서 피닝과 상호 TLS(mTLS) 구현하기
7. 개발용 자체 서명 인증서 생성하기
8. 일반적인 TLS 잘못된 구성 식별 및 수정하기
9. 표준 도구를 사용하여 TLS 구성 테스트하기

---

## 목차

1. [TLS 개요와 역사](#1-tls-개요와-역사)
2. [TLS 1.3 핸드셰이크 상세](#2-tls-13-핸드셰이크-상세)
3. [암호화 스위트](#3-암호화-스위트)
4. [X.509 인증서](#4-x509-인증서)
5. [인증서 체인과 신뢰](#5-인증서-체인과-신뢰)
6. [인증 기관(CA)](#6-인증-기관ca)
7. [Let's Encrypt와 ACME](#7-lets-encrypt와-acme)
8. [인증서 피닝](#8-인증서-피닝)
9. [상호 TLS(mTLS)](#9-상호-tlsmtls)
10. [Python TLS 프로그래밍](#10-python-tls-프로그래밍)
11. [OpenSSL로 인증서 생성하기](#11-openssl로-인증서-생성하기)
12. [일반적인 잘못된 구성](#12-일반적인-잘못된-구성)
13. [TLS 구성 테스트](#13-tls-구성-테스트)
14. [연습 문제](#14-연습-문제)
15. [참고 자료](#15-참고-자료)

---

## 1. TLS 개요와 역사

### 1.1 프로토콜 진화

```
┌─────────────────────────────────────────────────────────────────────┐
│                  TLS/SSL 버전 역사                                    │
├──────────┬──────┬───────────┬────────────────────────────────────────┤
│ 프로토콜  │ 연도 │ 상태      │ 참고사항                               │
├──────────┼──────┼───────────┼────────────────────────────────────────┤
│ SSL 1.0  │ 1994 │ 미출시    │ 출시되지 않음 (심각한 결함)            │
│ SSL 2.0  │ 1995 │ 폐기됨    │ 완전히 손상됨, 모든 곳에서 비활성화   │
│ SSL 3.0  │ 1996 │ 폐기됨    │ POODLE 공격 (2014), 손상됨             │
│ TLS 1.0  │ 1999 │ 폐기됨    │ BEAST 공격 (2011), 2020년 수명 종료   │
│ TLS 1.1  │ 2006 │ 폐기됨    │ 알려진 공격 없음, 하지만 약한 암호화  │
│ TLS 1.2  │ 2008 │ 지원됨    │ 여전히 널리 사용됨, 올바르게 구성    │
│          │      │           │ 시 안전함                              │
│ TLS 1.3  │ 2018 │ 권장됨    │ 주요 재설계: 더 빠르고, 더 안전함    │
│          │      │           │ 모든 약한 알고리즘 제거               │
├──────────┴──────┴───────────┴────────────────────────────────────────┤
│                                                                      │
│ 현재 권장 사항 (2025+):                                             │
│ • TLS 1.3 활성화 (권장) 및 TLS 1.2 (대체)                          │
│ • TLS 1.2 미만의 모든 버전 비활성화                                │
│ • 강력한 암호화 스위트만 사용                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 스택에서 TLS의 위치

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TLS를 포함한 프로토콜 스택                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  애플리케이션 계층                                           │    │
│  │  HTTP, SMTP, IMAP, FTP, MQTT, gRPC, WebSocket              │    │
│  └────────────────────────────┬────────────────────────────────┘    │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐    │
│  │  TLS (전송 계층 보안)                                        │    │
│  │  ┌─────────────┐  ┌───────────────┐  ┌─────────────────┐   │    │
│  │  │  핸드셰이크  │  │ 레코드        │  │  알림           │   │    │
│  │  │  프로토콜    │  │ 프로토콜      │  │  프로토콜       │   │    │
│  │  └─────────────┘  └───────────────┘  └─────────────────┘   │    │
│  └────────────────────────────┬────────────────────────────────┘    │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐    │
│  │  전송 계층 (TCP)                                             │    │
│  │  포트 443 (HTTPS), 465 (SMTPS), 993 (IMAPS), 등            │    │
│  └────────────────────────────┬────────────────────────────────┘    │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐    │
│  │  인터넷 계층 (IP)                                            │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 TLS가 제공하는 것

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TLS 보안 속성                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 기밀성                                                          │
│     ├── 전송 중 데이터 암호화 (AES-GCM, ChaCha20-Poly1305)         │
│     └── 도청자는 암호화된 데이터만 볼 수 있음                       │
│                                                                      │
│  2. 무결성                                                          │
│     ├── AEAD 암호화는 모든 레코드를 인증함                         │
│     └── 모든 변조가 즉시 감지됨                                    │
│                                                                      │
│  3. 인증                                                            │
│     ├── 서버는 X.509 인증서를 통해 신원을 증명함                   │
│     ├── 선택사항: 클라이언트 인증 (mTLS)                           │
│     └── 중간자 공격 방지                                           │
│                                                                      │
│  4. 전방향 안전성 (TLS 1.3에서 필수)                               │
│     ├── 임시 키 교환 (ECDHE)                                       │
│     └── 장기 개인 키가 나중에 손상되더라도 과거 세션은             │
│         복호화할 수 없음                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. TLS 1.3 핸드셰이크 상세

TLS 1.3는 TLS 1.2에 비해 핸드셰이크를 크게 단순화하여 2번의 왕복에서 1번(또는 0-RTT로 0번)으로 줄였습니다.

### 2.1 전체 핸드셰이크 (1-RTT)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TLS 1.3 전체 핸드셰이크 (1-RTT)                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   클라이언트                                      서버               │
│   ────────                                        ────               │
│                                                                      │
│   ┌─ ClientHello ───────────────────────────────▶                   │
│   │  • 지원되는 TLS 버전 (1.3)                                      │
│   │  • 암호화 스위트 (예: TLS_AES_256_GCM_SHA384)                  │
│   │  • 키 공유 (X25519 공개 키)  ←── 1.3의 새로운 기능             │
│   │  • 지원되는 그룹 (X25519, P-256)                               │
│   │  • 서명 알고리즘 (Ed25519, ECDSA-P256-SHA256)                  │
│   │  • 난수 (32 바이트)                                            │
│   │  • SNI (서버 이름 표시)                                        │
│   └──────────────────────────────────────────────                   │
│                                                                      │
│           ┌─ ServerHello ◀──────────────────────────                │
│           │  • 선택된 TLS 버전 (1.3)                                │
│           │  • 선택된 암호화 스위트                                 │
│           │  • 서버 키 공유 (X25519 공개 키)                        │
│           │  • 난수 (32 바이트)                                     │
│           └──────────────────────────────────────                   │
│                                                                      │
│           ┌─ {EncryptedExtensions} ◀──────────────── (암호화됨)    │
│           │  • 서버 확장                                            │
│           └──────────────────────────────────────                   │
│                                                                      │
│           ┌─ {Certificate} ◀──────────────────────── (암호화됨)    │
│           │  • 서버의 X.509 인증서 체인                            │
│           └──────────────────────────────────────                   │
│                                                                      │
│           ┌─ {CertificateVerify} ◀────────────────── (암호화됨)    │
│           │  • 핸드셰이크 전사본에 대한 디지털 서명                │
│           │  • 서버가 개인 키를 소유하고 있음을 증명               │
│           └──────────────────────────────────────                   │
│                                                                      │
│           ┌─ {Finished} ◀──────────────────────────── (암호화됨)   │
│           │  • 전체 핸드셰이크 전사본에 대한 MAC                   │
│           └──────────────────────────────────────                   │
│                                                                      │
│   ┌─ {Finished} ──────────────────────────────────▶ (암호화됨)     │
│   │  • 클라이언트의 핸드셰이크 전사본에 대한 MAC                   │
│   └──────────────────────────────────────────────                   │
│                                                                      │
│   ═══════════════════════════════════════════════════                │
│   애플리케이션 데이터가 양방향으로 흐름 (암호화됨)                  │
│   ═══════════════════════════════════════════════════                │
│                                                                      │
│   핵심 인사이트: TLS 1.3에서 키 교환은 첫 번째 메시지                │
│   (ClientHello가 키 공유 포함)에서 발생하므로, ServerHello 이후의    │
│   모든 후속 메시지가 암호화됩니다.                                   │
│                                                                      │
│   TLS 1.2와 비교: 2-RTT 핸드셰이크, 인증서가 평문으로 전송됨,       │
│   필수 전방향 안전성 없음.                                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 TLS 1.3의 키 파생

```
┌──────────────────────────────────────────────────────────────────────┐
│              TLS 1.3 키 스케줄 (단순화)                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ECDHE 공유 비밀 (X25519 키 교환에서)                               │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────┐                                                │
│  │  HKDF-Extract   │ ← IKM = ECDHE 공유 비밀                       │
│  │  (Early Secret) │    Salt = 0                                    │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  HKDF-Extract   │ ← IKM = ECDHE 공유 비밀                       │
│  │(Handshake Secret)│   Salt = early secret에서 파생                │
│  └────────┬────────┘                                                │
│           │                                                         │
│      ┌────┴────┐                                                    │
│      │         │                                                    │
│      ▼         ▼                                                    │
│  ┌────────┐ ┌────────┐                                              │
│  │Client  │ │Server  │ ← HKDF-Expand-Label                         │
│  │Handshk │ │Handshk │   (handshake secret + 전사본 해시에서        │
│  │Traffic │ │Traffic │    트래픽 키 파생)                           │
│  │Key/IV  │ │Key/IV  │                                              │
│  └────────┘ └────────┘                                              │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  HKDF-Extract   │ ← IKM = 0                                     │
│  │ (Master Secret) │    Salt = handshake secret에서 파생            │
│  └────────┬────────┘                                                │
│           │                                                         │
│      ┌────┴────┐                                                    │
│      │         │                                                    │
│      ▼         ▼                                                    │
│  ┌────────┐ ┌────────┐                                              │
│  │Client  │ │Server  │ ← HKDF-Expand-Label                         │
│  │App     │ │App     │   (애플리케이션 트래픽 키 파생)              │
│  │Traffic │ │Traffic │                                              │
│  │Key/IV  │ │Key/IV  │                                              │
│  └────────┘ └────────┘                                              │
│                                                                      │
│  각 방향은 자체 키와 IV를 가짐.                                     │
│  키는 공유 비밀과 핸드셰이크 전사본에서 파생되어,                   │
│  이 특정 세션에 바인딩됨.                                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.3 TLS 1.3 vs TLS 1.2

```
┌─────────────────────────────────────────────────────────────────────┐
│              TLS 1.3 vs TLS 1.2 개선 사항                           │
├─────────────────────┬─────────────────────┬─────────────────────────┤
│ 기능                │ TLS 1.2             │ TLS 1.3                 │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ 핸드셰이크 RTT      │ 2 왕복              │ 1 왕복                  │
│ 0-RTT 재개          │ 없음                │ 있음 (선택사항)         │
│ 전방향 안전성       │ 선택사항 (ECDHE)    │ 필수 (항상)             │
│ RSA 키 교환         │ 지원됨              │ 제거됨 (FS 없음)        │
│ 정적 DH             │ 지원됨              │ 제거됨 (FS 없음)        │
│ CBC 모드 암호화     │ 지원됨              │ 제거됨 (패딩 오라클)    │
│ RC4                 │ 지원됨 (약함)       │ 제거됨 (손상됨)         │
│ 압축                │ 지원됨              │ 제거됨 (CRIME 공격)     │
│ 재협상              │ 지원됨              │ 제거됨 (복잡성)         │
│ 인증서 암호화?      │ 아니오 (평문)       │ 예 (암호화됨)           │
│ 암호화 스위트       │ 300+ (많은 약한 것) │ 5 (모두 강함)           │
│ Change Cipher Spec  │ 별도 메시지         │ 제거됨                  │
│ 키 파생             │ PRF (TLS 1.2 PRF)   │ HKDF (더 강함)          │
├─────────────────────┴─────────────────────┴─────────────────────────┤
│ TLS 1.3는 더 간단하고, 더 빠르며, 알려진 약한 구성이 없습니다.     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 0-RTT (제로 라운드 트립 타임) 재개

```
┌──────────────────────────────────────────────────────────────────────┐
│                 TLS 1.3 0-RTT 재개                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  전체 핸드셰이크 후 서버는 세션 티켓을 보냅니다.                    │
│  재연결 시 클라이언트는 즉시 데이터를 보낼 수 있습니다:              │
│                                                                      │
│   클라이언트                                      서버               │
│   ────────                                        ────               │
│                                                                      │
│   ┌─ ClientHello + EarlyData ────────────────────▶                  │
│   │  • 세션 티켓 (이전 연결에서)                                    │
│   │  • 0-RTT 애플리케이션 데이터 (PSK로 암호화됨)                  │
│   └──────────────────────────────────────────────                   │
│                                                                      │
│   ⚠ 경고: 0-RTT 데이터는 재생 안전하지 않습니다!                   │
│   공격자는 ClientHello + EarlyData를 재생할 수 있습니다.            │
│                                                                      │
│   0-RTT 규칙:                                                       │
│   • 멱등성 요청에만 사용 (GET, POST 아님)                          │
│   • 서버는 재생 보호를 구현해야 함                                 │
│   • 0-RTT 데이터 크기 제한                                         │
│   • 많은 배포가 0-RTT를 완전히 비활성화함                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. 암호화 스위트

### 3.1 TLS 1.3 암호화 스위트

TLS 1.3에는 5개의 암호화 스위트만 있으며, 모두 AEAD입니다:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  TLS 1.3 암호화 스위트                               │
├─────────────────────────────────────────┬───────────────────────────┤
│ 암호화 스위트                           │ 참고사항                  │
├─────────────────────────────────────────┼───────────────────────────┤
│ TLS_AES_256_GCM_SHA384                  │ 강력함, 널리 지원됨       │
│ TLS_AES_128_GCM_SHA256                  │ 기본값, HW로 매우 빠름   │
│ TLS_CHACHA20_POLY1305_SHA256            │ 모바일/AES-NI 없을 때 최적│
│ TLS_AES_128_CCM_SHA256                  │ 제약된 장치용             │
│ TLS_AES_128_CCM_8_SHA256                │ 짧은 태그, IoT 전용       │
├─────────────────────────────────────────┴───────────────────────────┤
│                                                                      │
│ TLS 1.3에서 암호화 스위트는 다음만 지정합니다:                     │
│  • AEAD 알고리즘 (AES-GCM, ChaCha20-Poly1305, AES-CCM)            │
│  • HKDF용 해시 (SHA-256, SHA-384)                                  │
│                                                                      │
│ 키 교환 및 서명 알고리즘은 확장을 통해 별도로 협상됩니다            │
│ (supported_groups, signature_algorithms).                           │
│                                                                      │
│ 권장 우선순위:                                                      │
│  1. TLS_AES_256_GCM_SHA384                                         │
│  2. TLS_CHACHA20_POLY1305_SHA256                                   │
│  3. TLS_AES_128_GCM_SHA256                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 TLS 1.2 암호화 스위트 이름 이해하기

레거시 호환성을 위해 TLS 1.2 암호화 스위트 명명 이해가 중요합니다:

```
TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
│    │      │        │   │    │    │
│    │      │        │   │    │    └── PRF 해시 (SHA-384)
│    │      │        │   │    └─────── AEAD 모드 (GCM)
│    │      │        │   └──────────── 키 크기 (256비트)
│    │      │        └──────────────── 암호화 (AES)
│    │      └───────────────────────── 인증 (RSA 인증서)
│    └──────────────────────────────── 키 교환 (ECDHE = 전방향 안전성)
└───────────────────────────────────── 프로토콜 (TLS)

좋은 TLS 1.2 암호화 스위트 (모두 전방향 안전성을 위한 ECDHE 포함):
  TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
  TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
  TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
  TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
  TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256

나쁜 TLS 1.2 암호화 스위트 (이것들을 비활성화하세요):
  TLS_RSA_WITH_AES_256_CBC_SHA256        (전방향 안전성 없음, CBC)
  TLS_RSA_WITH_3DES_EDE_CBC_SHA          (FS 없음, 3DES, CBC)
  TLS_RSA_WITH_RC4_128_SHA               (FS 없음, RC4 손상됨)
  TLS_DHE_RSA_WITH_AES_256_CBC_SHA       (CBC, 약한 DH 가능)
```

---

## 4. X.509 인증서

X.509 인증서는 공개 키를 신원(도메인 이름, 조직)에 바인딩합니다. 이것은 TLS 인증의 기초입니다.

### 4.1 인증서 구조

```
┌──────────────────────────────────────────────────────────────────────┐
│                    X.509 v3 인증서 구조                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  tbsCertificate (서명 대상):                                        │
│  ├── Version: v3 (가장 일반적)                                      │
│  ├── Serial Number: CA의 고유 식별자                                │
│  ├── Signature Algorithm: 예: SHA256withRSA, Ed25519               │
│  ├── Issuer: 이 인증서에 서명한 CA의 DN                             │
│  │   └── CN=Let's Encrypt Authority X3, O=Let's Encrypt, C=US     │
│  ├── Validity:                                                      │
│  │   ├── Not Before: 2026-01-01 00:00:00 UTC                      │
│  │   └── Not After:  2026-03-31 23:59:59 UTC                      │
│  ├── Subject: 인증서 소유자의 DN                                    │
│  │   └── CN=www.example.com                                        │
│  ├── Subject Public Key Info:                                       │
│  │   ├── Algorithm: RSA (2048비트) 또는 ECDSA (P-256)             │
│  │   └── Public Key: [실제 공개 키 바이트]                        │
│  └── Extensions (v3):                                               │
│      ├── Subject Alternative Names (SAN):                           │
│      │   ├── DNS: example.com                                      │
│      │   ├── DNS: www.example.com                                  │
│      │   └── DNS: api.example.com                                  │
│      ├── Key Usage:                                                 │
│      │   ├── Digital Signature                                     │
│      │   └── Key Encipherment                                      │
│      ├── Extended Key Usage:                                        │
│      │   └── TLS Web Server Authentication                         │
│      ├── Basic Constraints:                                         │
│      │   └── CA: FALSE (이것은 리프/최종 엔티티 인증서)            │
│      ├── Authority Key Identifier: [CA의 키 ID]                    │
│      ├── CRL Distribution Points: http://crl.example.com/ca.crl   │
│      └── Authority Information Access:                              │
│          ├── OCSP: http://ocsp.example.com                         │
│          └── CA Issuers: http://crt.example.com/ca.crt             │
│                                                                      │
│  Signature Algorithm: SHA256withRSA                                  │
│  Signature Value: [tbsCertificate에 대한 CA의 디지털 서명]          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Python에서 인증서 읽기

```python
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes
import ssl
import socket
from datetime import datetime, timezone

def get_server_certificate(hostname: str, port: int = 443) -> x509.Certificate:
    """서버의 TLS 인증서를 검색하고 파싱합니다."""
    # 연결하고 DER 형식으로 인증서 가져오기
    context = ssl.create_default_context()
    with socket.create_connection((hostname, port), timeout=10) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            der_cert = ssock.getpeercert(binary_form=True)
    return x509.load_der_x509_certificate(der_cert)

def print_certificate_info(cert: x509.Certificate):
    """상세한 인증서 정보를 표시합니다."""
    print("=" * 70)
    print("X.509 인증서 상세 정보")
    print("=" * 70)

    # Subject
    print(f"\nSubject:")
    for attr in cert.subject:
        print(f"  {attr.oid._name}: {attr.value}")

    # Issuer
    print(f"\nIssuer:")
    for attr in cert.issuer:
        print(f"  {attr.oid._name}: {attr.value}")

    # Validity
    now = datetime.now(timezone.utc)
    print(f"\nValidity:")
    print(f"  Not Before: {cert.not_valid_before_utc}")
    print(f"  Not After:  {cert.not_valid_after_utc}")
    days_remaining = (cert.not_valid_after_utc - now).days
    status = "VALID" if cert.not_valid_before_utc <= now <= cert.not_valid_after_utc else "EXPIRED"
    print(f"  Status:     {status} ({days_remaining}일 남음)")

    # Serial and Version
    print(f"\nSerial Number: {cert.serial_number}")
    print(f"Version:       v{cert.version.value + 1}")

    # Public Key
    pub_key = cert.public_key()
    key_type = type(pub_key).__name__
    print(f"\nPublic Key:")
    print(f"  Algorithm: {key_type}")
    if hasattr(pub_key, 'key_size'):
        print(f"  Key Size:  {pub_key.key_size} bits")

    # Signature
    print(f"\nSignature Algorithm: {cert.signature_algorithm_oid._name}")

    # Fingerprints
    print(f"\nFingerprints:")
    print(f"  SHA-256: {cert.fingerprint(hashes.SHA256()).hex()}")
    print(f"  SHA-1:   {cert.fingerprint(hashes.SHA1()).hex()}")

    # Extensions
    print(f"\nExtensions:")
    try:
        san = cert.extensions.get_extension_for_oid(ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        names = san.value.get_attributes_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
        dns_names = san.value.get_attributes_for_oid(x509.DNSName) if hasattr(san.value, 'get_attributes_for_oid') else []
        # SAN의 대체 접근 방식
        for name in san.value:
            print(f"  SAN: {name.value}")
    except x509.ExtensionNotFound:
        print("  SAN 확장 없음")

    try:
        basic = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
        print(f"  Basic Constraints: CA={basic.value.ca}")
    except x509.ExtensionNotFound:
        pass

# 예제: 실제 인증서 검사
try:
    cert = get_server_certificate("www.google.com")
    print_certificate_info(cert)
except Exception as e:
    print(f"인증서를 가져올 수 없습니다: {e}")
    print("(네트워크 접근 필요)")
```

---

## 5. 인증서 체인과 신뢰

### 5.1 신뢰 체인

```
┌──────────────────────────────────────────────────────────────────────┐
│                    인증서 신뢰 체인                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  ROOT CA 인증서 (자체 서명)                              │       │
│  │  ├── Issuer: "Root CA"                                    │       │
│  │  ├── Subject: "Root CA"  (Issuer == Subject → 자체 서명) │       │
│  │  ├── Valid: 20-30년                                       │       │
│  │  ├── OS/브라우저 신뢰 저장소에 저장됨                     │       │
│  │  └── 서명: 중간 CA 인증서                                 │       │
│  └────────────────────────┬─────────────────────────────────┘       │
│                           │ 서명                                    │
│                           ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  중간 CA 인증서                                           │       │
│  │  ├── Issuer: "Root CA"                                    │       │
│  │  ├── Subject: "Intermediate CA"                           │       │
│  │  ├── Valid: 5-10년                                        │       │
│  │  ├── Basic Constraints: CA=TRUE                           │       │
│  │  └── 서명: 최종 엔티티(리프) 인증서                       │       │
│  └────────────────────────┬─────────────────────────────────┘       │
│                           │ 서명                                    │
│                           ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  최종 엔티티(리프) 인증서                                 │       │
│  │  ├── Issuer: "Intermediate CA"                            │       │
│  │  ├── Subject: "www.example.com"                           │       │
│  │  ├── Valid: 90일 (Let's Encrypt) ~ 1년                    │       │
│  │  ├── Basic Constraints: CA=FALSE                          │       │
│  │  └── 이것이 서버의 인증서입니다                           │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  검증 프로세스:                                                     │
│  1. 서버로부터 리프 인증서 + 중간 인증서 수신                       │
│  2. 중간 인증서의 공개 키를 사용하여 리프 인증서 서명 검증          │
│  3. 루트의 공개 키를 사용하여 중간 인증서 서명 검증                 │
│  4. 루트 인증서가 로컬 신뢰 저장소에 있는지 확인                    │
│  5. 모든 인증서가 유효 기간 내에 있는지 확인                        │
│  6. 폐기 상태 확인 (CRL 또는 OCSP)                                 │
│  7. 리프 인증서의 SAN이 요청된 호스트명과 일치하는지 확인           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Python에서 인증서 체인 검증

```python
from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.x509.oid import NameOID
from datetime import datetime, timedelta, timezone
import ipaddress

def create_ca_certificate(
    subject_name: str,
    key_size: int = 4096,
    valid_days: int = 3650,
) -> tuple:
    """자체 서명된 CA 인증서를 생성합니다."""
    # CA 키 쌍 생성
    ca_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Demo CA"),
        x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
    ])

    now = datetime.now(timezone.utc)

    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=valid_days))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=1),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_cert_sign=True,
                crl_sign=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return ca_key, ca_cert

def create_server_certificate(
    ca_key,
    ca_cert: x509.Certificate,
    hostname: str,
    valid_days: int = 90,
) -> tuple:
    """CA가 서명한 서버 인증서를 생성합니다."""
    # 서버 키 쌍 생성
    server_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    now = datetime.now(timezone.utc)

    server_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, hostname),
        ]))
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=valid_days))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(hostname),
                x509.DNSName(f"*.{hostname}"),
            ]),
            critical=False,
        )
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return server_key, server_cert

def verify_certificate_chain(
    leaf_cert: x509.Certificate,
    intermediate_certs: list,
    trusted_roots: list,
    hostname: str = None,
) -> dict:
    """
    인증서 체인을 수동으로 검증합니다.
    프로덕션에서는 이를 자동으로 처리하는 ssl.SSLContext를 사용하세요.
    """
    results = {
        "valid": True,
        "checks": [],
        "errors": [],
    }

    now = datetime.now(timezone.utc)

    # 체인 구축: 리프 → 중간 → 루트
    chain = [leaf_cert] + intermediate_certs

    for i, cert in enumerate(chain):
        cert_name = "Leaf" if i == 0 else f"Intermediate {i}"

        # 유효 기간 확인
        if cert.not_valid_before_utc > now:
            results["errors"].append(f"{cert_name}: 아직 유효하지 않음")
            results["valid"] = False
        elif cert.not_valid_after_utc < now:
            results["errors"].append(f"{cert_name}: 만료됨")
            results["valid"] = False
        else:
            days_left = (cert.not_valid_after_utc - now).days
            results["checks"].append(f"{cert_name}: 유효함 ({days_left}일 남음)")

        # 서명 검증 (발급자가 이 인증서에 서명함)
        if i < len(chain) - 1:
            # 체인의 다음 인증서가 발급자여야 함
            issuer_cert = chain[i + 1]
        else:
            # 마지막 인증서는 신뢰할 수 있는 루트에 의해 서명되어야 함
            issuer_cert = None
            for root in trusted_roots:
                if root.subject == cert.issuer:
                    issuer_cert = root
                    break

            if issuer_cert is None:
                results["errors"].append(f"{cert_name}: 신뢰 저장소에서 발급자를 찾을 수 없음")
                results["valid"] = False
                continue

        # 디지털 서명 검증
        try:
            issuer_pub = issuer_cert.public_key()
            issuer_pub.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm,
            )
            results["checks"].append(f"{cert_name}: 서명 유효 ({issuer_cert.subject.rfc4514_string()}가 서명함)")
        except Exception as e:
            results["errors"].append(f"{cert_name}: 서명 검증 실패: {e}")
            results["valid"] = False

    # SAN에 대한 호스트명 확인
    if hostname and results["valid"]:
        try:
            san = leaf_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            dns_names = san.value.get_attributes_for_oid(x509.DNSName) if hasattr(san.value, 'get_attributes_for_oid') else []
            names = [name.value for name in san.value]
            if hostname in names or any(
                name.startswith("*.") and hostname.endswith(name[1:])
                for name in names
            ):
                results["checks"].append(f"호스트명 '{hostname}'이 SAN과 일치함")
            else:
                results["errors"].append(f"호스트명 '{hostname}'이 SAN에 없음: {names}")
                results["valid"] = False
        except x509.ExtensionNotFound:
            results["errors"].append("SAN 확장을 찾을 수 없음")
            results["valid"] = False

    return results

# 완전한 인증서 체인 생성
print("인증서 체인 생성")
print("=" * 60)

# 1. 루트 CA 생성
ca_key, ca_cert = create_ca_certificate("Demo Root CA")
print(f"Root CA: {ca_cert.subject.rfc4514_string()}")

# 2. 서버 인증서 생성
server_key, server_cert = create_server_certificate(
    ca_key, ca_cert, "example.com"
)
print(f"Server:  {server_cert.subject.rfc4514_string()}")

# 3. 체인 검증
result = verify_certificate_chain(
    leaf_cert=server_cert,
    intermediate_certs=[],  # 직접 CA 서명 (중간 없음)
    trusted_roots=[ca_cert],
    hostname="example.com",
)

print(f"\n체인 검증: {'유효' if result['valid'] else '무효'}")
for check in result["checks"]:
    print(f"  [OK] {check}")
for error in result["errors"]:
    print(f"  [!!] {error}")

# 잘못된 호스트명으로 테스트
result2 = verify_certificate_chain(
    leaf_cert=server_cert,
    intermediate_certs=[],
    trusted_roots=[ca_cert],
    hostname="evil.com",
)
print(f"\n잘못된 호스트명: {'유효' if result2['valid'] else '무효'}")
for error in result2["errors"]:
    print(f"  [!!] {error}")
```

---

## 6. 인증 기관(CA)

### 6.1 CA 작동 방식

```
┌──────────────────────────────────────────────────────────────────────┐
│              인증 기관 프로세스                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 도메인 소유자가 키 쌍과 CSR 생성                                 │
│     ┌─────────────┐                                                 │
│     │ Private Key  │ (비밀 유지!)                                    │
│     │ Public Key   │ → CSR에 포함됨                                  │
│     │ CSR          │ → CA에 전송됨                                   │
│     └─────────────┘                                                 │
│            │                                                        │
│            ▼                                                        │
│  2. CA가 도메인 소유권 검증                                         │
│     ┌─────────────────────────────────────────────┐                │
│     │  DV (도메인 검증):                           │                │
│     │    HTTP 챌린지, DNS 챌린지, 또는 이메일      │                │
│     │                                               │                │
│     │  OV (조직 검증):                             │                │
│     │    DV + 조직 신원 확인                       │                │
│     │                                               │                │
│     │  EV (확장 검증):                             │                │
│     │    OV + 광범위한 법적/물리적 검증            │                │
│     └─────────────────────────────────────────────┘                │
│            │                                                        │
│            ▼                                                        │
│  3. CA가 인증서에 서명                                              │
│     ┌─────────────┐                                                 │
│     │ CA가 X.509  │                                                 │
│     │ 인증서 생성 │ → CA의 개인 키로 서명됨                         │
│     │ (소유자의   │                                                 │
│     │ 공개 키 포함)│                                                 │
│     └─────────────┘                                                 │
│            │                                                        │
│            ▼                                                        │
│  4. 도메인 소유자가 서버에 인증서 설치                              │
│     TLS 핸드셰이크 중 서버가 인증서 체인 전송                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 인증서 폐기

```
┌─────────────────────────────────────────────────────────────────────┐
│              인증서 폐기 방법                                        │
├──────────────────┬──────────────────────────────────────────────────┤
│ 방법             │ 설명                                             │
├──────────────────┼──────────────────────────────────────────────────┤
│ CRL              │ 인증서 폐기 목록                                 │
│ (RFC 5280)       │ • CA가 폐기된 인증서의 서명된 목록을 게시       │
│                  │ • 클라이언트가 목록을 다운로드하고 확인         │
│                  │ • 문제: 목록이 커지고, 오래된 데이터            │
├──────────────────┼──────────────────────────────────────────────────┤
│ OCSP             │ 온라인 인증서 상태 프로토콜                     │
│ (RFC 6960)       │ • 클라이언트가 CA에게 질문: "이 인증서 폐기됨?"│
│                  │ • 실시간 응답                                   │
│                  │ • 문제: 개인정보 (CA가 방문 사이트 볼 수 있음),│
│                  │   가용성 의존성                                 │
├──────────────────┼──────────────────────────────────────────────────┤
│ OCSP Stapling    │ 서버가 OCSP 응답을 가져와서 TLS 핸드셰이크에   │
│ (RFC 6066)       │ 첨부                                             │
│                  │ • 클라이언트가 CA에 연결하지 않고 검증         │
│                  │ • 모범 사례! 서버에서 구성하세요               │
│                  │ • 응답은 CA가 서명하고, 시간 제한이 있음       │
├──────────────────┼──────────────────────────────────────────────────┤
│ CRLite           │ 압축된 폐기 데이터를 브라우저로 푸시            │
│ (Firefox)        │ • 완전한 폐기 정보, 네트워크 불필요            │
│                  │ • 매우 효율적 (모든 폐기에 대해 < 1 MB)        │
├──────────────────┼──────────────────────────────────────────────────┤
│ 단기 인증서      │ 며칠만 유효한 인증서                            │
│                  │ • 폐기 불필요 (폐기가 효력을 발휘하기 전에     │
│                  │   인증서가 만료됨)                              │
│                  │ • Let's Encrypt 인증서: 90일                    │
└──────────────────┴──────────────────────────────────────────────────┘
```

### 6.3 인증서 투명성(CT)

```
┌─────────────────────────────────────────────────────────────────────┐
│              인증서 투명성 (RFC 6962)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  문제: 불량하거나 손상된 CA가 모든 도메인에 대해 사기성 인증서를    │
│  발급할 수 있습니다.                                                │
│                                                                      │
│  솔루션: 모든 인증서가 공개적으로 로그되어야 합니다.                │
│                                                                      │
│  ┌────────┐  인증서 발급  ┌─────┐  제출   ┌───────┐                │
│  │ 도메인 │ ◀───────────  │ CA  │ ───────▶│  CT   │                │
│  │ 소유자 │               │     │         │  로그 │                │
│  └────────┘               └─────┘         └───┬───┘                │
│                                               │                     │
│  ┌────────┐  TLS의 SCT  ┌─────┐  확인      │                     │
│  │브라우저│ ◀────────── │서버 │            │                     │
│  │        │─────────────▶│     │◀───────────┘                     │
│  │        │   SCT 검증   └─────┘  SCT 반환                        │
│  └────┬───┘                                                       │
│       │                                                            │
│       ▼                                                            │
│  ┌────────────┐                                                    │
│  │  모니터링  │  무단 인증서에 대한 CT 로그 감시                   │
│  │  서비스    │  도메인 소유자가 불량 인증서 감지 가능             │
│  └────────────┘                                                    │
│                                                                      │
│  SCT = 서명된 인증서 타임스탬프 (로깅 증명)                         │
│  Chrome/Firefox는 모든 공개적으로 신뢰할 수 있는 인증서에 대해      │
│  CT를 요구합니다.                                                   │
│                                                                      │
│  CT 로그 검색: https://crt.sh/                                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Let's Encrypt와 ACME

### 7.1 ACME 프로토콜 개요

```
┌──────────────────────────────────────────────────────────────────────┐
│         ACME 프로토콜 (자동 인증서 관리)                             │
│                       RFC 8555                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   클라이언트 (certbot)                      Let's Encrypt            │
│   ────────────────                          ──────────────            │
│                                                                      │
│   1. 계정 생성                                                      │
│      POST /acme/new-acct ──────────────────▶                        │
│                           ◀────────────────── 계정 URL              │
│                                                                      │
│   2. 주문 제출                                                      │
│      POST /acme/new-order ─────────────────▶                        │
│      { identifiers: ["example.com"] }                               │
│                           ◀────────────────── 인증 URL들            │
│                                                                      │
│   3. 챌린지 완료 (도메인 소유권 증명)                               │
│                                                                      │
│      HTTP-01 챌린지:                                                │
│      PUT /.well-known/acme-challenge/{token}                        │
│      → Let's Encrypt가 이 URL을 GET하여 검증                       │
│                                                                      │
│      DNS-01 챌린지:                                                 │
│      TXT 레코드 생성: _acme-challenge.example.com                   │
│      → Let's Encrypt가 DNS를 쿼리하여 검증                          │
│                                                                      │
│   4. 주문 완료 (CSR 제출)                                           │
│      POST /acme/finalize ──────────────────▶                        │
│      { csr: "MIIBkTCB..." }                                        │
│                           ◀────────────────── 인증서 URL            │
│                                                                      │
│   5. 인증서 다운로드                                                │
│      GET /acme/cert/abc123 ─────────────────▶                       │
│                           ◀────────────────── PEM 인증서 체인       │
│                                                                      │
│   6. 자동 갱신 (만료 전, 일반적으로 60일에)                         │
│      2-5단계를 자동으로 반복                                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 Certbot 명령어

```bash
# certbot 설치
# Ubuntu/Debian
sudo apt install certbot python3-certbot-nginx

# macOS
brew install certbot

# 인증서 획득 (독립 실행 모드 - 웹 서버를 임시로 중지)
sudo certbot certonly --standalone -d example.com -d www.example.com

# Nginx 플러그인으로 획득 (다운타임 없음)
sudo certbot --nginx -d example.com -d www.example.com

# DNS 챌린지로 획득 (와일드카드용)
sudo certbot certonly --manual --preferred-challenges dns -d "*.example.com"

# 갱신 테스트
sudo certbot renew --dry-run

# 실제 갱신 (일반적으로 cron/systemd 타이머를 통해)
sudo certbot renew

# 인증서 보기
sudo certbot certificates

# 인증서 폐기
sudo certbot revoke --cert-path /etc/letsencrypt/live/example.com/cert.pem
```

### 7.3 인증서 파일

```
/etc/letsencrypt/live/example.com/
├── cert.pem       # 도메인의 인증서 (리프만)
├── chain.pem      # 중간 CA 인증서
├── fullchain.pem  # cert.pem + chain.pem (대부분의 서버가 필요로 함)
├── privkey.pem    # 개인 키 (비밀 유지!)
└── README

# Nginx 구성
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # 현대적인 TLS 구성
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305;
    ssl_prefer_server_ciphers off;

    # HSTS
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload" always;

    # OCSP Stapling
    ssl_stapling on;
    ssl_stapling_verify on;
    ssl_trusted_certificate /etc/letsencrypt/live/example.com/chain.pem;
}
```

---

## 8. 인증서 피닝

인증서 피닝은 주어진 도메인에 대해 허용되는 인증서를 제한하여, CA가 손상되더라도 공격을 방지합니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│              인증서 피닝 전략                                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 공개 키 피닝 (권장)                                             │
│     공개 키의 해시를 고정합니다 (전체 인증서가 아님).               │
│     인증서 갱신 시에도 유지됩니다 (키는 동일하게 유지).             │
│                                                                      │
│  2. 인증서 피닝                                                     │
│     전체 인증서의 해시를 고정합니다.                                │
│     인증서가 갱신될 때 핀을 업데이트해야 합니다.                    │
│                                                                      │
│  3. 중간 CA 피닝                                                    │
│     중간 CA의 공개 키를 고정합니다.                                 │
│     리프 피닝보다 더 유연합니다.                                    │
│                                                                      │
│  중요한 경고:                                                       │
│  • HPKP (HTTP 공개 키 피닝)는 브라우저에서 폐기되었습니다          │
│    (도메인을 영구적으로 손상시키기 너무 쉬움)                       │
│  • 주로 모바일 앱과 API 클라이언트에서 피닝 사용                   │
│  • 항상 백업 핀을 포함하세요                                       │
│  • 웹 브라우저의 경우 이제 피닝 대신 인증서 투명성이 선호됩니다    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.1 Python에서 인증서 피닝

```python
import ssl
import socket
import hashlib
from cryptography import x509
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat
)

class PinnedHTTPSConnection:
    """인증서 피닝을 사용하는 HTTPS 연결."""

    def __init__(self, hostname: str, port: int = 443,
                 pinned_hashes: list = None):
        """
        Args:
            hostname: 서버 호스트명
            port: 서버 포트
            pinned_hashes: 허용되는 공개 키의 SHA-256 해시 목록
                           (base64로 인코딩됨)
        """
        self.hostname = hostname
        self.port = port
        self.pinned_hashes = pinned_hashes or []

    def _get_pubkey_hash(self, cert_der: bytes) -> str:
        """인증서의 공개 키의 SHA-256 해시를 계산합니다."""
        cert = x509.load_der_x509_certificate(cert_der)
        pub_key_bytes = cert.public_key().public_bytes(
            Encoding.DER,
            PublicFormat.SubjectPublicKeyInfo,
        )
        return hashlib.sha256(pub_key_bytes).hexdigest()

    def connect(self) -> dict:
        """인증서 핀 검증으로 연결합니다."""
        context = ssl.create_default_context()

        with socket.create_connection(
            (self.hostname, self.port), timeout=10
        ) as sock:
            with context.wrap_socket(sock, server_hostname=self.hostname) as ssock:
                # 인증서 체인 가져오기
                cert_der = ssock.getpeercert(binary_form=True)
                cert_info = ssock.getpeercert()

                # 공개 키 해시 계산
                actual_hash = self._get_pubkey_hash(cert_der)

                # 핀 확인
                pin_valid = True
                if self.pinned_hashes:
                    pin_valid = actual_hash in self.pinned_hashes

                return {
                    "hostname": self.hostname,
                    "protocol": ssock.version(),
                    "cipher": ssock.cipher(),
                    "cert_subject": dict(x[0] for x in cert_info["subject"]),
                    "cert_issuer": dict(x[0] for x in cert_info["issuer"]),
                    "pubkey_hash": actual_hash,
                    "pin_valid": pin_valid,
                    "pinned_hashes": self.pinned_hashes,
                }

# 먼저 서버의 핀 찾기
try:
    discovery = PinnedHTTPSConnection("www.google.com")
    result = discovery.connect()
    print("인증서 핀 찾기:")
    print(f"  호스트:       {result['hostname']}")
    print(f"  프로토콜:     {result['protocol']}")
    print(f"  암호화:       {result['cipher'][0]}")
    print(f"  Subject:      {result['cert_subject']}")
    print(f"  Pubkey Hash:  {result['pubkey_hash']}")

    # 이제 핀으로 연결
    pinned = PinnedHTTPSConnection(
        "www.google.com",
        pinned_hashes=[result["pubkey_hash"]]
    )
    pinned_result = pinned.connect()
    print(f"\n  핀 유효:      {pinned_result['pin_valid']}")

    # 잘못된 핀
    wrong = PinnedHTTPSConnection(
        "www.google.com",
        pinned_hashes=["0000000000000000000000000000000000000000000000000000000000000000"]
    )
    wrong_result = wrong.connect()
    print(f"  잘못된 핀:    {wrong_result['pin_valid']}")

except Exception as e:
    print(f"네트워크 오류 (오프라인 환경에서 예상됨): {e}")
```

---

## 9. 상호 TLS(mTLS)

표준 TLS에서는 서버만 인증서를 제시합니다. 상호 TLS에서는 클라이언트와 서버 모두 인증서를 제시하고 서로를 검증합니다. 이것은 서비스 간 통신, API 인증, 제로 트러스트 아키텍처에 사용됩니다.

```
┌──────────────────────────────────────────────────────────────────────┐
│              표준 TLS vs 상호 TLS                                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  표준 TLS (단방향):                                                 │
│                                                                      │
│  클라이언트 ──── "당신은 누구입니까?" ──── 서버                     │
│         ◀─── [서버 인증서] ───                                      │
│         ──── "좋아요, 신뢰합니다" ────▶                             │
│                                                                      │
│  서버만 인증됩니다.                                                 │
│  클라이언트 신원은 다른 수단으로 검증됩니다 (JWT, API 키 등)       │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  상호 TLS (양방향):                                                 │
│                                                                      │
│  클라이언트 ──── "당신은 누구입니까?" ──── 서버                     │
│         ◀─── [서버 인증서] ───                                      │
│         ◀─── "당신은 누구입니까?" ─────────                         │
│         ──── [클라이언트 인증서] ────▶                              │
│         ◀─── "좋아요, 신뢰합니다" ──────                            │
│         ──── "좋아요, 저도 신뢰합니다" ──▶                          │
│                                                                      │
│  클라이언트와 서버 모두 인증됩니다.                                 │
│  비밀번호, 토큰, API 키가 필요 없습니다.                            │
│                                                                      │
│  사용 사례:                                                         │
│  • 마이크로서비스 간 (Istio, Linkerd 서비스 메시)                  │
│  • 파트너/기계용 API 인증                                          │
│  • IoT 장치 인증                                                   │
│  • 제로 트러스트 네트워크                                          │
│  • 데이터베이스 연결                                               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.1 mTLS 구현

```python
import ssl
import socket
import threading
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from datetime import datetime, timedelta, timezone

def create_mtls_certificates(base_dir: str) -> dict:
    """mTLS용 CA, 서버, 클라이언트 인증서를 생성합니다."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)

    # 1. CA 생성
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "mTLS Demo CA"),
        ]))
        .issuer_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "mTLS Demo CA"),
        ]))
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=365))
        .add_extension(
            x509.BasicConstraints(ca=True, path_length=0), critical=True
        )
        .sign(ca_key, hashes.SHA256())
    )

    # 2. 서버 인증서 생성
    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ]))
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=90))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            ]), critical=False
        )
        .add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.SERVER_AUTH,
            ]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    # 3. 클라이언트 인증서 생성
    client_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    client_cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, "api-client-1"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "My App"),
        ]))
        .issuer_name(ca_cert.subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + timedelta(days=90))
        .add_extension(
            x509.ExtendedKeyUsage([
                x509.oid.ExtendedKeyUsageOID.CLIENT_AUTH,
            ]), critical=False
        )
        .sign(ca_key, hashes.SHA256())
    )

    # 모든 인증서와 키 작성
    files = {}
    for name, key, cert in [
        ("ca", ca_key, ca_cert),
        ("server", server_key, server_cert),
        ("client", client_key, client_cert),
    ]:
        cert_path = base / f"{name}_cert.pem"
        key_path = base / f"{name}_key.pem"

        cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

        files[f"{name}_cert"] = str(cert_path)
        files[f"{name}_key"] = str(key_path)

    return files

import ipaddress

# 인증서 생성
certs = create_mtls_certificates("/tmp/mtls_demo")
print("생성된 인증서:")
for name, path in certs.items():
    print(f"  {name}: {path}")

def create_mtls_server_context(certs: dict) -> ssl.SSLContext:
    """mTLS 서버용 SSL 컨텍스트를 생성합니다."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # 서버 인증서와 키 로드
    ctx.load_cert_chain(
        certfile=certs["server_cert"],
        keyfile=certs["server_key"],
    )

    # 클라이언트 인증서 요구 (이것이 mTLS로 만듭니다)
    ctx.verify_mode = ssl.CERT_REQUIRED

    # 클라이언트 인증서에 대해 우리 CA만 신뢰
    ctx.load_verify_locations(cafile=certs["ca_cert"])

    return ctx

def create_mtls_client_context(certs: dict) -> ssl.SSLContext:
    """mTLS 클라이언트용 SSL 컨텍스트를 생성합니다."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # 클라이언트 인증서와 키 로드
    ctx.load_cert_chain(
        certfile=certs["client_cert"],
        keyfile=certs["client_key"],
    )

    # 서버 인증서 검증을 위해 우리 CA 신뢰
    ctx.load_verify_locations(cafile=certs["ca_cert"])

    return ctx

# 간단한 mTLS 에코 서버
def mtls_server(certs: dict, port: int):
    """간단한 mTLS 서버를 실행합니다."""
    ctx = create_mtls_server_context(certs)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", port))
        sock.listen(1)

        with ctx.wrap_socket(sock, server_side=True) as ssock:
            conn, addr = ssock.accept()
            with conn:
                # 클라이언트 인증서 정보 가져오기
                client_cert = conn.getpeercert()
                client_cn = dict(x[0] for x in client_cert["subject"])["commonName"]
                print(f"  [서버] 클라이언트 인증됨: {client_cn}")

                # 수신한 데이터 에코
                data = conn.recv(1024)
                conn.sendall(f"안녕하세요 {client_cn}! 당신이 말한 것: {data.decode()}".encode())

# mTLS 테스트
port = 8443
server_thread = threading.Thread(target=mtls_server, args=(certs, port))
server_thread.daemon = True
server_thread.start()

import time
time.sleep(0.5)  # 서버 시작 대기

# 클라이언트로 연결
client_ctx = create_mtls_client_context(certs)
with socket.create_connection(("localhost", port)) as sock:
    with client_ctx.wrap_socket(sock, server_hostname="localhost") as ssock:
        print(f"  [클라이언트] {ssock.version()}로 연결됨")
        print(f"  [클라이언트] 서버 인증서: {dict(x[0] for x in ssock.getpeercert()['subject'])}")

        ssock.sendall(b"mTLS 클라이언트에서 안녕하세요!")
        response = ssock.recv(1024)
        print(f"  [클라이언트] 응답: {response.decode()}")

# 정리
import shutil
shutil.rmtree("/tmp/mtls_demo")
```

---

## 10. Python TLS 프로그래밍

### 10.1 안전한 HTTPS 클라이언트

```python
import ssl
import urllib.request

def make_secure_request(url: str) -> dict:
    """보안 모범 사례를 사용하여 HTTPS 요청을 합니다."""
    # 안전한 SSL 컨텍스트 생성
    ctx = ssl.create_default_context()

    # 최소 TLS 버전 강제
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # 압축 비활성화 (CRIME 공격 방지)
    ctx.options |= ssl.OP_NO_COMPRESSION

    # 강력한 암호화 스위트 설정 (TLS 1.2 대체용)
    ctx.set_ciphers(
        "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20"
    )

    # 요청 생성
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
        return {
            "status": response.status,
            "headers": dict(response.headers),
            "url": response.url,
        }

# 예제
try:
    result = make_secure_request("https://www.example.com")
    print(f"Status: {result['status']}")
    print(f"URL: {result['url']}")
except Exception as e:
    print(f"Error: {e}")
```

### 10.2 Python TLS 서버

```python
import ssl
import socket
import threading

def create_tls_server(certfile: str, keyfile: str,
                       host: str = "localhost", port: int = 8443):
    """현대적인 구성으로 TLS 서버를 생성합니다."""
    # SSL 컨텍스트 생성
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # 인증서와 키 로드
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)

    # 강력한 암호화
    ctx.set_ciphers(
        "ECDHE+AESGCM:ECDHE+CHACHA20"
    )

    # 보안 옵션
    ctx.options |= ssl.OP_NO_COMPRESSION    # CRIME 방지
    ctx.options |= ssl.OP_SINGLE_DH_USE     # 새로운 DH 파라미터
    ctx.options |= ssl.OP_SINGLE_ECDH_USE   # 새로운 ECDH 파라미터

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(5)
        print(f"TLS 서버가 {host}:{port}에서 수신 대기 중")

        with ctx.wrap_socket(sock, server_side=True) as ssock:
            while True:
                conn, addr = ssock.accept()
                print(f"{addr}로부터 연결")
                print(f"  프로토콜: {conn.version()}")
                print(f"  암호화:   {conn.cipher()}")

                # 연결 처리
                data = conn.recv(1024)
                if data:
                    conn.sendall(b"HTTP/1.1 200 OK\r\n"
                                b"Content-Type: text/plain\r\n\r\n"
                                b"Hello, TLS!\n")
                conn.close()
```

---

## 11. OpenSSL로 인증서 생성하기

### 11.1 자체 서명 인증서 (개발용)

```bash
# 개발용 자체 서명 인증서 생성
# (프로덕션에서는 사용하지 마세요!)

# 한 줄로: 한 명령으로 키 + 인증서 생성
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 \
    -nodes -keyout server.key -out server.crt \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

# 인증서 확인
openssl x509 -in server.crt -text -noout

# 키 확인
openssl rsa -in server.key -check -noout
```

### 11.2 CA + 서버 인증서 (전체 프로세스)

```bash
# 1단계: CA 개인 키 생성
openssl genrsa -aes256 -out ca.key 4096
# CA 키의 암호 입력 (기억하세요!)

# 2단계: CA 인증서 생성 (자체 서명)
openssl req -new -x509 -sha256 -days 3650 -key ca.key -out ca.crt \
    -subj "/C=US/O=My Organization/CN=My Root CA"

# 3단계: 서버 개인 키 생성 (자동화를 위한 암호 없음)
openssl genrsa -out server.key 2048

# 4단계: 인증서 서명 요청(CSR) 생성
openssl req -new -sha256 -key server.key -out server.csr \
    -subj "/CN=myserver.example.com"

# 5단계: SAN (Subject Alternative Names)을 위한 구성 파일 생성
cat > server_ext.cnf << 'EOF'
authorityKeyIdentifier=keyid,issuer
basicConstraints=CA:FALSE
keyUsage=digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=@alt_names

[alt_names]
DNS.1 = myserver.example.com
DNS.2 = *.myserver.example.com
IP.1 = 192.168.1.100
EOF

# 6단계: CA 키로 CSR 서명
openssl x509 -req -sha256 -days 365 \
    -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out server.crt -extfile server_ext.cnf

# 7단계: 인증서 검증
openssl verify -CAfile ca.crt server.crt

# 8단계: 인증서 상세 정보 보기
openssl x509 -in server.crt -text -noout | head -30

# 9단계: fullchain 생성 (서버 인증서 + CA 인증서)
cat server.crt ca.crt > fullchain.crt
```

### 11.3 ECDSA 인증서 (더 작고, 더 빠름)

```bash
# ECDSA 키 생성 (P-256 곡선)
openssl ecparam -genkey -name prime256v1 -out server_ec.key

# 또는 Ed25519 (OpenSSL 버전이 지원하는 경우)
openssl genpkey -algorithm Ed25519 -out server_ed25519.key

# ECDSA 키로 CSR 생성
openssl req -new -sha256 -key server_ec.key -out server_ec.csr \
    -subj "/CN=myserver.example.com"

# 자체 서명 ECDSA 인증서
openssl req -x509 -sha256 -days 365 -key server_ec.key -out server_ec.crt \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

# 크기 비교
echo "RSA-2048 인증서 크기:"
wc -c server.crt
echo "ECDSA P-256 인증서 크기:"
wc -c server_ec.crt
```

### 11.4 mTLS용 클라이언트 인증서

```bash
# 클라이언트 키 생성
openssl genrsa -out client.key 2048

# 클라이언트 CSR 생성
openssl req -new -sha256 -key client.key -out client.csr \
    -subj "/CN=api-client-1/O=My App"

# CA로 서명 (참고: extendedKeyUsage = clientAuth)
cat > client_ext.cnf << 'EOF'
basicConstraints=CA:FALSE
keyUsage=digitalSignature
extendedKeyUsage=clientAuth
EOF

openssl x509 -req -sha256 -days 365 \
    -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out client.crt -extfile client_ext.cnf

# PKCS12 번들 생성 (브라우저/앱으로 가져오기용)
openssl pkcs12 -export -out client.p12 \
    -inkey client.key -in client.crt -certfile ca.crt

# 클라이언트 인증서 검증
openssl verify -CAfile ca.crt client.crt
```

---

## 12. 일반적인 잘못된 구성

```
┌──────────────────────────────────────────────────────────────────────┐
│         일반적인 TLS 잘못된 구성과 수정 방법                         │
├───┬──────────────────────────────┬──────────────────────────────────┤
│ # │ 잘못된 구성                  │ 수정 방법                        │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 1 │ 레거시 TLS 버전 활성화       │ SSL 2/3, TLS 1.0/1.1 비활성화   │
│   │ (TLS 1.0, SSL 3.0)          │ min_version = TLSv1.2           │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 2 │ 약한 암호화 스위트           │ ECDHE를 사용하는 AEAD만 사용    │
│   │ (RC4, 3DES, CBC, 정적 RSA)  │ 모든 non-FS 스위트 비활성화     │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 3 │ 프로덕션에서 자체 서명 인증서│ Let's Encrypt 사용 (무료!)      │
│   │                              │ 자체 서명은 개발/테스트용만     │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 4 │ 만료된 인증서                │ certbot으로 자동 갱신           │
│   │                              │ 외부 도구로 모니터링            │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 5 │ 중간 인증서 누락             │ 항상 전체 체인 전송             │
│   │ (불완전한 체인)              │ (fullchain.pem, cert.pem 아님)  │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 6 │ HSTS 헤더 없음               │ Strict-Transport-Security 추가  │
│   │                              │ 긴 max-age와 함께               │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 7 │ HTTP에서 HTTPS로 리다이렉트  │ 301 영구 리다이렉트 사용        │
│   │ 누락                         │ HSTS preload 목록 등록          │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 8 │ 개인 키 권한이 너무 느슨함   │ chmod 600으로 키 파일 설정      │
│   │ (world-readable)             │ root 또는 www-data만 소유       │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 9 │ TLS 압축 활성화              │ 비활성화 (CRIME 공격 방지)      │
│   │                              │ ssl_options |= OP_NO_COMPRESSION│
├───┼──────────────────────────────┼──────────────────────────────────┤
│10 │ OCSP 스테이플링 없음         │ 서버 구성에서 OCSP 스테이플링   │
│   │                              │ 활성화 (성능 향상)              │
├───┼──────────────────────────────┼──────────────────────────────────┤
│11 │ 공개 사이트에 와일드카드     │ 특정 도메인으로 SAN 사용        │
│   │ 인증서 사용                  │ 와일드카드는 내부용만           │
├───┼──────────────────────────────┼──────────────────────────────────┤
│12 │ 백업 핀 없는 인증서 피닝     │ 항상 백업 핀 포함               │
│   │                              │ 또는 피닝 대신 CT 사용          │
└───┴──────────────────────────────┴──────────────────────────────────┘
```

### 12.1 보안 헤더 체크리스트

```python
# HTTPS 사이트에 권장되는 보안 헤더

SECURITY_HEADERS = {
    # 2년간 HTTPS 강제, 하위 도메인 포함
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",

    # MIME 타입 스니핑 방지
    "X-Content-Type-Options": "nosniff",

    # 클릭재킹 방지
    "X-Frame-Options": "DENY",

    # XSS 필터 활성화 (레거시 브라우저)
    "X-XSS-Protection": "0",  # "0"이 이제 권장됨 (CSP가 더 나음)

    # 콘텐츠 보안 정책
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'",

    # 리퍼러 정책
    "Referrer-Policy": "strict-origin-when-cross-origin",

    # 권한 정책 (이전 Feature-Policy)
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

# Flask 예제
from flask import Flask, make_response

app = Flask(__name__)

@app.after_request
def add_security_headers(response):
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    return response
```

---

## 13. TLS 구성 테스트

### 13.1 명령줄 도구

```bash
# OpenSSL s_client로 TLS 연결 테스트
openssl s_client -connect example.com:443 -servername example.com

# 전체 인증서 체인 표시
openssl s_client -connect example.com:443 -servername example.com -showcerts

# 특정 TLS 버전 테스트
openssl s_client -connect example.com:443 -tls1_3
openssl s_client -connect example.com:443 -tls1_2

# 인증서 만료 확인
echo | openssl s_client -connect example.com:443 -servername example.com 2>/dev/null \
    | openssl x509 -noout -dates

# 지원되는 암호화 스위트 확인
nmap --script ssl-enum-ciphers -p 443 example.com

# testssl.sh로 테스트 (종합 스캐너)
# git clone https://github.com/drwetter/testssl.sh.git
./testssl.sh example.com

# sslyze로 테스트
pip install sslyze
sslyze example.com
```

### 13.2 Python TLS 스캐너

```python
import ssl
import socket
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class TLSScanResult:
    hostname: str
    port: int
    supported_protocols: List[str] = field(default_factory=list)
    certificate_info: dict = field(default_factory=dict)
    cipher_suite: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

def scan_tls(hostname: str, port: int = 443) -> TLSScanResult:
    """서버의 TLS 구성을 스캔합니다."""
    result = TLSScanResult(hostname=hostname, port=port)

    # 지원되는 프로토콜 테스트
    protocols = {
        "TLSv1.0": ssl.TLSVersion.TLSv1,
        "TLSv1.1": ssl.TLSVersion.TLSv1_1,
        "TLSv1.2": ssl.TLSVersion.TLSv1_2,
        "TLSv1.3": ssl.TLSVersion.TLSv1_3,
    }

    for name, version in protocols.items():
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.minimum_version = version
            ctx.maximum_version = version
            ctx.check_hostname = True
            ctx.verify_mode = ssl.CERT_REQUIRED
            ctx.load_default_certs()

            with socket.create_connection((hostname, port), timeout=5) as sock:
                with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                    result.supported_protocols.append(name)
                    if name == "TLSv1.0":
                        result.issues.append("TLS 1.0 지원됨 (폐기됨)")
                    elif name == "TLSv1.1":
                        result.issues.append("TLS 1.1 지원됨 (폐기됨)")
        except (ssl.SSLError, ConnectionRefusedError, OSError):
            pass

    # 사용 가능한 최상의 프로토콜로 인증서 및 암호화 정보 가져오기
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                # 프로토콜 및 암호화
                result.cipher_suite = ssock.cipher()[0]

                # 인증서 정보
                cert = ssock.getpeercert()
                result.certificate_info = {
                    "subject": dict(x[0] for x in cert.get("subject", ())),
                    "issuer": dict(x[0] for x in cert.get("issuer", ())),
                    "notAfter": cert.get("notAfter", ""),
                    "notBefore": cert.get("notBefore", ""),
                    "serialNumber": cert.get("serialNumber", ""),
                    "version": cert.get("version", ""),
                    "san": [
                        x[1] for x in cert.get("subjectAltName", ())
                    ],
                }

                # 만료 확인
                from datetime import datetime
                not_after = ssl.cert_time_to_seconds(cert["notAfter"])
                import time
                days_left = (not_after - time.time()) / 86400
                if days_left < 0:
                    result.issues.append(f"인증서 만료됨 {abs(days_left):.0f}일 전!")
                elif days_left < 30:
                    result.issues.append(f"인증서가 {days_left:.0f}일 후에 만료됨!")

    except Exception as e:
        result.issues.append(f"연결 오류: {e}")

    # 권장 사항 생성
    if "TLSv1.3" not in result.supported_protocols:
        result.recommendations.append("더 나은 보안과 성능을 위해 TLS 1.3 활성화")
    if "TLSv1.0" in result.supported_protocols:
        result.recommendations.append("TLS 1.0 비활성화 (2020년부터 폐기됨)")
    if "TLSv1.1" in result.supported_protocols:
        result.recommendations.append("TLS 1.1 비활성화 (2020년부터 폐기됨)")
    if not result.issues:
        result.recommendations.append("구성이 양호합니다!")

    return result

def print_scan_result(result: TLSScanResult):
    """TLS 스캔 결과를 보기 좋게 출력합니다."""
    print(f"\nTLS 스캔: {result.hostname}:{result.port}")
    print("=" * 60)

    print(f"\n지원되는 프로토콜:")
    for proto in result.supported_protocols:
        status = "[!!]" if "1.0" in proto or "1.1" in proto else "[OK]"
        print(f"  {status} {proto}")

    if result.cipher_suite:
        print(f"\n협상된 암호화: {result.cipher_suite}")

    if result.certificate_info:
        ci = result.certificate_info
        print(f"\n인증서:")
        print(f"  Subject: {ci.get('subject', {}).get('commonName', 'N/A')}")
        print(f"  Issuer:  {ci.get('issuer', {}).get('commonName', 'N/A')}")
        print(f"  Valid:   {ci.get('notBefore', 'N/A')} to {ci.get('notAfter', 'N/A')}")
        if ci.get("san"):
            print(f"  SANs:    {', '.join(ci['san'][:5])}")

    if result.issues:
        print(f"\n문제점 ({len(result.issues)}):")
        for issue in result.issues:
            print(f"  [!!] {issue}")

    if result.recommendations:
        print(f"\n권장 사항:")
        for rec in result.recommendations:
            print(f"  --> {rec}")

# 스캔 실행
try:
    result = scan_tls("www.google.com")
    print_scan_result(result)
except Exception as e:
    print(f"스캔 실패 (네트워크 필요): {e}")
```

### 13.3 지속적인 인증서 모니터링

```python
import ssl
import socket
import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List

@dataclass
class CertMonitorEntry:
    hostname: str
    port: int = 443
    warning_days: int = 30   # 이보다 적은 일수 남으면 경고
    critical_days: int = 7   # 이보다 적은 일수 남으면 치명적

class CertificateMonitor:
    """여러 도메인의 인증서 만료를 모니터링합니다."""

    def __init__(self, entries: List[CertMonitorEntry]):
        self.entries = entries

    def check_certificate(self, entry: CertMonitorEntry) -> dict:
        """단일 인증서의 만료를 확인합니다."""
        try:
            ctx = ssl.create_default_context()
            with socket.create_connection(
                (entry.hostname, entry.port), timeout=10
            ) as sock:
                with ctx.wrap_socket(
                    sock, server_hostname=entry.hostname
                ) as ssock:
                    cert = ssock.getpeercert()
                    not_after = ssl.cert_time_to_seconds(cert["notAfter"])
                    days_left = (not_after - time.time()) / 86400

                    if days_left < 0:
                        status = "EXPIRED"
                    elif days_left < entry.critical_days:
                        status = "CRITICAL"
                    elif days_left < entry.warning_days:
                        status = "WARNING"
                    else:
                        status = "OK"

                    return {
                        "hostname": entry.hostname,
                        "status": status,
                        "days_remaining": round(days_left, 1),
                        "expires": cert["notAfter"],
                        "issuer": dict(x[0] for x in cert["issuer"]).get(
                            "commonName", "Unknown"
                        ),
                        "protocol": ssock.version(),
                    }
        except Exception as e:
            return {
                "hostname": entry.hostname,
                "status": "ERROR",
                "error": str(e),
            }

    def check_all(self) -> list:
        """모든 모니터링되는 인증서를 확인합니다."""
        results = []
        for entry in self.entries:
            result = self.check_certificate(entry)
            results.append(result)
        return results

    def print_report(self, results: list):
        """형식화된 모니터링 보고서를 출력합니다."""
        print(f"\n인증서 모니터링 보고서")
        print(f"생성 시각: {datetime.now(timezone.utc).isoformat()}")
        print("=" * 70)

        status_icons = {
            "OK": "[OK]",
            "WARNING": "[!!]",
            "CRITICAL": "[XX]",
            "EXPIRED": "[XX]",
            "ERROR": "[??]",
        }

        for r in sorted(results, key=lambda x: x.get("days_remaining", -999)):
            icon = status_icons.get(r["status"], "[??]")
            if "days_remaining" in r:
                print(f"  {icon} {r['hostname']:30s} "
                      f"{r['days_remaining']:6.0f} 일  "
                      f"({r['status']})")
            else:
                print(f"  {icon} {r['hostname']:30s} "
                      f"{'N/A':>6s}       "
                      f"({r.get('error', r['status'])})")

# 사용법
monitor = CertificateMonitor([
    CertMonitorEntry("www.google.com"),
    CertMonitorEntry("github.com"),
    CertMonitorEntry("expired.badssl.com"),  # 테스트용 만료된 인증서
])

try:
    results = monitor.check_all()
    monitor.print_report(results)
except Exception as e:
    print(f"모니터링에는 네트워크 액세스가 필요합니다: {e}")
```

---

## 14. 연습 문제

### 연습 문제 1: TLS 핸드셰이크 추적 (초급)

`openssl s_client`를 사용하여 세 개의 다른 웹사이트에 연결하고 다음 질문에 답하세요:
1. 어떤 TLS 버전이 협상되었나요?
2. 어떤 암호화 스위트가 선택되었나요?
3. 체인에 몇 개의 인증서가 있나요?
4. 루트 CA는 무엇인가요?
5. 리프 인증서는 언제 만료되나요?

```bash
# 템플릿 명령어:
echo | openssl s_client -connect <host>:443 -servername <host> 2>/dev/null
```

google.com, github.com, letsencrypt.org에 대한 결과를 비교하세요.

### 연습 문제 2: 인증서 생성 실습 (중급)

Python의 `cryptography` 라이브러리를 사용하여:
1. 4096비트 RSA 키로 루트 CA 생성 (10년 유효)
2. 루트가 서명한 중간 CA 생성 (5년 유효)
3. 중간 CA가 서명한 서버 인증서 생성 (90일 유효)
4. 모든 인증서와 키를 PEM 파일로 작성
5. 전체 체인을 프로그래밍 방식으로 검증
6. `openssl verify`로 체인 테스트

### 연습 문제 3: mTLS 서비스 (중급)

mTLS로 보호되는 간단한 HTTP API 구축:
1. CA, 서버, 두 개의 클라이언트 인증서 생성
2. 클라이언트 인증서를 요구하는 Flask/HTTP 서버 작성
3. 인증서에서 클라이언트의 Common Name 추출
4. 클라이언트 CN 기반 권한 부여 구현
5. 인증서를 제시하는 클라이언트 작성
6. 유효한 클라이언트 인증서 없는 연결이 거부됨을 보여주기

### 연습 문제 4: TLS 구성 감사기 (고급)

다음을 확인하는 종합 TLS 스캐너 구축:
1. 지원되는 TLS 버전 (TLS 1.0/1.1을 폐기된 것으로 플래그)
2. 지원되는 암호화 스위트 (약한 것 플래그)
3. 인증서 체인 완전성
4. 인증서 만료 날짜
5. HSTS 헤더 존재 및 max-age
6. OCSP 스테이플링 상태
7. 키 크기 (RSA < 2048 또는 ECDSA < 256 플래그)
8. 전방향 안전성 지원
9. SSL Labs와 유사한 점수 출력 (A+에서 F까지)

### 연습 문제 5: 인증서 모니터링 시스템 (고급)

프로덕션 품질의 인증서 모니터링 시스템 구축:
1. YAML/JSON 구성 파일에서 도메인 목록 수락
2. 일정에 따라 인증서 만료 확인 (cron 호환)
3. 다음과 같은 경우 알림 전송 (이메일/Slack/webhook):
   - 30일 내 만료 (경고)
   - 7일 내 만료 (치명적)
   - 이미 만료됨 (긴급)
4. 인증서 변경 추적 (발급자 변경, 키 변경)
5. 추세 분석을 위해 SQLite에 이력 저장
6. HTML 보고서 생성

### 연습 문제 6: 단순화된 TLS 구현 (교육용)

TLS 핸드셰이크의 단순화된 버전 구축 (교육용, 프로덕션용 아님):
1. 클라이언트 전송: 지원되는 암호화 스위트, 난수 nonce, X25519 공개 키
2. 서버 응답: 선택된 암호화, 난수 nonce, X25519 공개 키, 인증서
3. 둘 다 X25519를 사용하여 공유 비밀 파생
4. 둘 다 HKDF를 사용하여 암호화 키 파생
5. 서버 전송: 핸드셰이크 전사본에 대한 서명
6. 클라이언트가 인증서의 공개 키를 사용하여 서명 검증
7. 둘 다 AES-GCM을 사용하여 암호화된 메시지 교환

이 연습 문제는 실제 프로토콜의 전체 복잡성 없이 TLS 1.3의 핵심 메커니즘을 시연합니다.

---

## 15. 참고 자료

- Rescorla, E. (2018). *The Transport Layer Security (TLS) Protocol Version 1.3* (RFC 8446).
- Mozilla Server Side TLS - https://wiki.mozilla.org/Security/Server_Side_TLS
- SSL Labs Best Practices - https://github.com/ssllabs/research/wiki/SSL-and-TLS-Deployment-Best-Practices
- Let's Encrypt Documentation - https://letsencrypt.org/docs/
- RFC 8555: Automatic Certificate Management Environment (ACME)
- RFC 6125: Representation and Verification of Domain-Based Application Service Identity
- Bulletproof TLS and PKI (Ivan Ristic, 2022) - https://www.feistyduck.com/books/bulletproof-tls-and-pki/
- Certificate Transparency - https://certificate.transparency.dev/
- testssl.sh - https://testssl.sh/
- Python `cryptography` library - https://cryptography.io/
- BadSSL.com - https://badssl.com/ (다양한 TLS 잘못된 구성 테스트)

---

**이전**: [03. 해싱과 데이터 무결성](./03_Hashing_and_Integrity.md) | **다음**: [05. 인증 시스템](./05_Authentication.md)
