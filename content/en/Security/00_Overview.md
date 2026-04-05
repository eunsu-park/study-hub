# Security

## Overview

This topic covers web security, cryptography, and secure software development practices. From fundamental concepts like the CIA Triad and threat modeling to hands-on projects building secure APIs and vulnerability scanners, these lessons provide a comprehensive foundation in application security.

## Prerequisites

- Python intermediate level (functions, classes, decorators)
- Basic understanding of HTTP and web development
- Familiarity with command-line tools
- Basic networking concepts (TCP/IP, DNS)

## Lesson Plan

### Foundations

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [01_Security_Fundamentals.md](./01_Security_Fundamentals.md) | ⭐⭐ | CIA Triad, Threat Modeling, STRIDE, Defense in Depth | Conceptual foundation |
| [02_Cryptography_Basics.md](./02_Cryptography_Basics.md) | ⭐⭐⭐ | AES, RSA, ECDSA, Key Exchange, Digital Signatures | Python `cryptography` library |
| [03_Hashing_and_Integrity.md](./03_Hashing_and_Integrity.md) | ⭐⭐⭐ | SHA-256, bcrypt, Argon2, HMAC, Merkle Trees | Password hashing best practices |
| [04_TLS_and_PKI.md](./04_TLS_and_PKI.md) | ⭐⭐⭐⭐ | TLS 1.3, X.509, Certificate Chains, mTLS, Let's Encrypt | OpenSSL practical examples |

### Authentication and Authorization

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [05_Authentication.md](./05_Authentication.md) | ⭐⭐⭐ | OAuth 2.0, JWT, TOTP/MFA, Session Management | PyJWT, pyotp examples |
| [06_Authorization.md](./06_Authorization.md) | ⭐⭐⭐ | RBAC, ABAC, ACL, Policy Engines, IDOR Prevention | Flask middleware examples |

### Web Security

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [07_OWASP_Top10.md](./07_OWASP_Top10.md) | ⭐⭐⭐ | OWASP Top 10 (2021), Vulnerable vs Fixed Code | Comprehensive reference |
| [08_Injection_Attacks.md](./08_Injection_Attacks.md) | ⭐⭐⭐ | SQL Injection, XSS, CSRF, Command Injection, SSTI | Attack/defense pairs |
| [09_Web_Security_Headers.md](./09_Web_Security_Headers.md) | ⭐⭐⭐ | CSP, HSTS, CORS, SRI, Permissions-Policy | Header configuration |
| [10_API_Security.md](./10_API_Security.md) | ⭐⭐⭐ | Rate Limiting, CORS, Input Validation, API Gateway | Flask examples |

### Operations and Infrastructure

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [11_Secrets_Management.md](./11_Secrets_Management.md) | ⭐⭐⭐ | Vault, .env, Git Secrets Scanning, Secret Rotation | 12-factor app |
| [12_Container_Security.md](./12_Container_Security.md) | ⭐⭐⭐⭐ | Docker Security, K8s RBAC, Image Scanning, SBOM | Trivy, cosign |

### Testing and Response

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [13_Security_Testing.md](./13_Security_Testing.md) | ⭐⭐⭐⭐ | SAST, DAST, SCA, Fuzzing, Penetration Testing | Bandit, Semgrep, ZAP |
| [14_Incident_Response.md](./14_Incident_Response.md) | ⭐⭐⭐⭐ | NIST IR Framework, Forensics, Log Analysis, SIEM | Playbook templates |

### Projects

| Filename | Difficulty | Key Topics | Notes |
|----------|------------|------------|-------|
| [15_Project_Secure_API.md](./15_Project_Secure_API.md) | ⭐⭐⭐⭐ | Secure FastAPI REST API, Argon2, JWT, RBAC, Rate Limiting | Full project |
| [16_Project_Vulnerability_Scanner.md](./16_Project_Vulnerability_Scanner.md) | ⭐⭐⭐⭐ | Port Scanner, Header Checker, TLS Analyzer, CVE Lookup | Full project |

## Recommended Learning Path

```
Fundamentals (L01-L04)     Authentication (L05-L06)     Web Security (L07-L10)
       │                          │                            │
       ▼                          ▼                            ▼
  CIA Triad              OAuth 2.0 / JWT            OWASP Top 10, XSS, SQLi
  Crypto basics          RBAC / ABAC                CSP, CORS, Rate Limiting
  TLS / PKI              Session Mgmt               API Security
       │                          │                            │
       └──────────────────────────┴────────────────────────────┘
                                  │
                                  ▼
                    Operations (L11-L12)
                    Secrets, Containers
                                  │
                                  ▼
                    Testing & Response (L13-L14)
                    SAST/DAST, Incident Response
                                  │
                                  ▼
                    Projects (L15-L16)
                    Secure API, Vuln Scanner
```

## Example Code

Example code for this topic is available in `examples/Security/`.

## Total

- **16 lessons** (4 foundations + 2 auth + 4 web + 2 ops + 2 testing + 2 projects)
- **Difficulty range**: ⭐ to ⭐⭐⭐⭐
- **Languages**: Python (primary), Bash (supplementary)
- **Key libraries**: cryptography, PyJWT, pyotp, bcrypt, Flask, FastAPI, Bandit, Semgrep
