# Security

## 개요

이 토픽은 웹 보안, 암호학, 그리고 안전한 소프트웨어 개발 관행을 다룹니다. CIA Triad와 Threat Modeling 같은 기본 개념부터 안전한 API와 취약점 스캐너를 구축하는 실습 프로젝트까지, 이 레슨들은 애플리케이션 보안에 대한 포괄적인 기초를 제공합니다.

## 선수 지식

- Python 중급 수준 (함수, 클래스, 데코레이터)
- HTTP와 웹 개발에 대한 기본 이해
- 커맨드라인 도구 사용 경험
- 기본 네트워킹 개념 (TCP/IP, DNS)

## 학습 계획

### 기초

| 파일명 | 난이도 | 주요 주제 | 비고 |
|----------|------------|------------|-------|
| [01_Security_Fundamentals.md](./01_Security_Fundamentals.md) | ⭐ | CIA Triad, Threat Modeling, STRIDE, Defense in Depth | 개념적 기초 |
| [02_Cryptography_Basics.md](./02_Cryptography_Basics.md) | ⭐⭐ | AES, RSA, ECDSA, Key Exchange, Digital Signatures | Python `cryptography` 라이브러리 |
| [03_Hashing_and_Integrity.md](./03_Hashing_and_Integrity.md) | ⭐⭐ | SHA-256, bcrypt, Argon2, HMAC, Merkle Trees | 패스워드 해싱 모범 사례 |
| [04_TLS_and_PKI.md](./04_TLS_and_PKI.md) | ⭐⭐ | TLS 1.3, X.509, Certificate Chains, mTLS, Let's Encrypt | OpenSSL 실습 예제 |

### 인증과 인가

| 파일명 | 난이도 | 주요 주제 | 비고 |
|----------|------------|------------|-------|
| [05_Authentication.md](./05_Authentication.md) | ⭐⭐⭐ | OAuth 2.0, JWT, TOTP/MFA, Session Management | PyJWT, pyotp 예제 |
| [06_Authorization.md](./06_Authorization.md) | ⭐⭐⭐ | RBAC, ABAC, ACL, Policy Engines, IDOR Prevention | Flask 미들웨어 예제 |

### 웹 보안

| 파일명 | 난이도 | 주요 주제 | 비고 |
|----------|------------|------------|-------|
| [07_OWASP_Top10.md](./07_OWASP_Top10.md) | ⭐⭐⭐ | OWASP Top 10 (2021), Vulnerable vs Fixed Code | 포괄적인 참고 자료 |
| [08_Injection_Attacks.md](./08_Injection_Attacks.md) | ⭐⭐⭐ | SQL Injection, XSS, CSRF, Command Injection, SSTI | 공격/방어 쌍 |
| [09_Web_Security_Headers.md](./09_Web_Security_Headers.md) | ⭐⭐⭐ | CSP, HSTS, CORS, SRI, Permissions-Policy | 헤더 설정 |
| [10_API_Security.md](./10_API_Security.md) | ⭐⭐⭐ | Rate Limiting, CORS, Input Validation, API Gateway | Flask 예제 |

### 운영과 인프라

| 파일명 | 난이도 | 주요 주제 | 비고 |
|----------|------------|------------|-------|
| [11_Secrets_Management.md](./11_Secrets_Management.md) | ⭐⭐⭐ | Vault, .env, Git Secrets Scanning, Secret Rotation | 12-factor app |
| [12_Container_Security.md](./12_Container_Security.md) | ⭐⭐⭐⭐ | Docker Security, K8s RBAC, Image Scanning, SBOM | Trivy, cosign |

### 테스트와 대응

| 파일명 | 난이도 | 주요 주제 | 비고 |
|----------|------------|------------|-------|
| [13_Security_Testing.md](./13_Security_Testing.md) | ⭐⭐⭐⭐ | SAST, DAST, SCA, Fuzzing, Penetration Testing | Bandit, Semgrep, ZAP |
| [14_Incident_Response.md](./14_Incident_Response.md) | ⭐⭐⭐⭐ | NIST IR Framework, Forensics, Log Analysis, SIEM | Playbook 템플릿 |

### 프로젝트

| 파일명 | 난이도 | 주요 주제 | 비고 |
|----------|------------|------------|-------|
| [15_Project_Secure_API.md](./15_Project_Secure_API.md) | ⭐⭐⭐⭐ | Secure FastAPI, Argon2, JWT, RBAC, Rate Limiting | 전체 프로젝트 |
| [16_Project_Vulnerability_Scanner.md](./16_Project_Vulnerability_Scanner.md) | ⭐⭐⭐⭐ | Port Scanner, Header Checker, TLS Analyzer, CVE Lookup | 전체 프로젝트 |

## 권장 학습 경로

```
기초 (L01-L04)              인증 (L05-L06)              웹 보안 (L07-L10)
       │                          │                            │
       ▼                          ▼                            ▼
  CIA Triad              OAuth 2.0 / JWT            OWASP Top 10, XSS, SQLi
  암호학 기초             RBAC / ABAC                CSP, CORS, Rate Limiting
  TLS / PKI              Session Mgmt               API Security
       │                          │                            │
       └──────────────────────────┴────────────────────────────┘
                                  │
                                  ▼
                    운영 (L11-L12)
                    Secrets, Containers
                                  │
                                  ▼
                    테스트 & 대응 (L13-L14)
                    SAST/DAST, Incident Response
                                  │
                                  ▼
                    프로젝트 (L15-L16)
                    Secure API, Vuln Scanner
```

## 예제 코드

이 토픽의 예제 코드는 `examples/Security/`에서 확인할 수 있습니다.

## 총계

- **16개 레슨** (기초 4개 + 인증 2개 + 웹 4개 + 운영 2개 + 테스트 2개 + 프로젝트 2개)
- **난이도 범위**: ⭐ ~ ⭐⭐⭐⭐
- **언어**: Python (주), Bash (보조)
- **주요 라이브러리**: cryptography, PyJWT, pyotp, bcrypt, FastAPI, Bandit, Semgrep
