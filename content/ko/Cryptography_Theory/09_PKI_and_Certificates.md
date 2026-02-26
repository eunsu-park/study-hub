# 레슨 9: PKI와 인증서

**이전**: [키 교환](./08_Key_Exchange.md) | **다음**: [격자 기반 암호](./10_Lattice_Based_Cryptography.md)

---

## 학습 목표

이 레슨을 마치면 다음을 할 수 있습니다:

1. 키 교환만으로는 신뢰 모델 없이 불충분한 이유를 설명할 수 있다
2. X.509 인증서 구조를 이해하고 프로그래밍으로 필드를 파싱할 수 있다
3. 최종 엔티티에서 중간 CA를 거쳐 루트 CA까지 인증서 체인을 추적할 수 있다
4. 계층형 PKI(X.509)와 신뢰의 웹(PGP) 모델을 비교할 수 있다
5. 인증서 폐지 메커니즘(CRL, OCSP, CT 로그)과 그 트레이드오프를 평가할 수 있다
6. Let's Encrypt와 ACME 프로토콜이 HTTPS를 대중화한 방식을 설명할 수 있다
7. 인증서 핀닝과 DANE/TLSA를 보조적 신뢰 메커니즘으로 분석할 수 있다

---

레슨 8에서 Diffie-Hellman을 통해 두 당사자가 안전하지 않은 채널 위에서 공유 비밀을 합의하는 방법을 배웠습니다. 그러나 DH만으로는 모든 통신에서 가장 근본적인 질문에 답할 수 없습니다: **"내가 정말 내가 생각하는 상대와 통신하고 있는가?"** 인증 없이는 공격자가 어느 쪽 당사자든 가장할 수 있습니다(중간자 공격). 공개 키 기반 구조(Public Key Infrastructure, PKI)는 신뢰 앵커로부터 인터넷의 모든 서버, 장치, 사용자까지 신뢰 체인을 형성함으로써 이 질문에 답합니다. 모든 HTTPS 연결, 모든 코드 서명 인증서, 모든 암호화된 이메일이 PKI에 의존하기 때문에 PKI를 이해하는 것은 필수입니다.

## 목차

1. [신뢰 문제](#1-신뢰-문제)
2. [X.509 인증서 구조](#2-x509-인증서-구조)
3. [인증서 체인과 신뢰 계층](#3-인증서-체인과-신뢰-계층)
4. [인증서 폐지](#4-인증서-폐지)
5. [인증서 투명성](#5-인증서-투명성)
6. [신뢰의 웹 vs. 계층형 PKI](#6-신뢰의-웹-vs-계층형-pki)
7. [Let's Encrypt와 ACME](#7-lets-encrypt와-acme)
8. [인증서 핀닝과 DANE](#8-인증서-핀닝과-dane)
9. [요약](#9-요약)
10. [연습 문제](#10-연습-문제)

---

## 1. 신뢰 문제

### 1.1 부트스트랩 딜레마(Bootstrap Dilemma)

앨리스가 `bank.com`과 안전하게 통신하려 한다고 가정합시다. ECDHE(레슨 8)를 사용해 공유 키를 수립할 수 있지만, 먼저 수신한 공개 키가 공격자가 아닌 실제 `bank.com`에 속하는지 확인해야 합니다.

핵심 질문은: **인터넷을 통해 수신한 공개 키를 어떻게 신뢰할 수 있는가?**

> **비유:** PKI는 여권 시스템과 같습니다. 여권(인증서)은 정부(인증 기관)가 발급합니다. 외국 국경(서버에 연결)에서 여권을 제시하면, 상대국(브라우저)은 발급 정부를 신뢰하기 때문에 여권을 신뢰합니다. 여권의 보안 기능(디지털 서명)을 확인하고, 효력이 취소되지 않았는지 검사하고, 사진이 얼굴과 일치하는지(도메인 이름이 인증서와 일치) 확인합니다.

### 1.2 역사적 해결책

| 접근 방식 | 작동 방식 | 한계 |
|----------|----------|------|
| 물리적 만남 | 직접 만나 키 교환 | 확장성 없음 |
| 신뢰 디렉토리 | 중앙 서버가 키 공개 | 단일 실패 지점 |
| 신뢰의 웹(Web of Trust) | 사용자들이 서로 보증 | 역량 보장 없음; 확장성 없음 |
| **계층형 PKI** | 신뢰할 수 있는 루트 CA가 신원 인증 | 루트 CA 집합을 신뢰해야 함 |

인터넷은 X.509 인증서를 사용하는 **계층형 PKI**를 주요 모델로 채택했습니다.

---

## 2. X.509 인증서 구조

### 2.1 X.509 인증서란?

X.509 인증서는 공개 키를 신원(예: 도메인 이름)에 바인딩하는 서명된 데이터 구조입니다. 인증 기관(Certificate Authority, CA)은 자신의 개인 키로 인증서에 서명함으로써 이 바인딩을 보증합니다.

### 2.2 인증서 필드 (v3)

```
Certificate:
    Version:             v3 (most common today)
    Serial Number:       Unique identifier assigned by the CA
    Signature Algorithm: e.g., sha256WithRSAEncryption, ecdsa-with-SHA384
    Issuer:              DN of the CA that signed this certificate
    Validity:
        Not Before:      Start of validity period
        Not After:       End of validity period
    Subject:             DN of the certificate holder
    Subject Public Key:  The public key being certified
    Extensions:
        Basic Constraints:      CA:TRUE or CA:FALSE
        Key Usage:              digitalSignature, keyEncipherment, etc.
        Subject Alt Name (SAN): DNS names, IP addresses
        Authority Key ID:       Links to issuer's key
        CRL Distribution Points: Where to check revocation
        Authority Info Access:   OCSP responder URL
    Signature:           CA's signature over all fields above
```

### 2.3 Python으로 인증서 파싱

```python
"""
Parse and inspect an X.509 certificate.

Why use the cryptography library instead of openssl CLI?
  Programmatic access allows automated validation, integration with
  application logic, and cross-platform compatibility.
"""

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes
import ssl
import socket
from datetime import datetime, timezone


def fetch_certificate(hostname: str, port: int = 443) -> x509.Certificate:
    """
    Fetch the TLS certificate from a live server.

    Why create a raw SSL context? We want to inspect the certificate
    regardless of whether it's valid — for educational purposes.
    """
    context = ssl.create_default_context()
    with socket.create_connection((hostname, port)) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            der_cert = ssock.getpeercert(binary_form=True)
    return x509.load_der_x509_certificate(der_cert)


def inspect_certificate(cert: x509.Certificate) -> None:
    """Display key fields of an X.509 certificate."""

    print("=== X.509 Certificate ===")
    print(f"Version:    {cert.version.name}")
    print(f"Serial:     {cert.serial_number}")

    # Subject — who this certificate is for
    subject = cert.subject
    cn = subject.get_attributes_for_oid(NameOID.COMMON_NAME)
    print(f"Subject CN: {cn[0].value if cn else 'N/A'}")

    # Issuer — who signed this certificate
    issuer = cert.issuer
    issuer_cn = issuer.get_attributes_for_oid(NameOID.COMMON_NAME)
    print(f"Issuer CN:  {issuer_cn[0].value if issuer_cn else 'N/A'}")

    # Validity period
    print(f"Not Before: {cert.not_valid_before_utc}")
    print(f"Not After:  {cert.not_valid_after_utc}")

    # Check if currently valid
    now = datetime.now(timezone.utc)
    is_valid = cert.not_valid_before_utc <= now <= cert.not_valid_after_utc
    print(f"Currently valid: {is_valid}")

    # Subject Alternative Names (SAN) — the domains this cert covers
    # Why SAN matters: Modern browsers ignore the CN field and rely
    # entirely on SAN for domain validation.
    try:
        san = cert.extensions.get_extension_for_oid(
            ExtensionOID.SUBJECT_ALTERNATIVE_NAME
        )
        names = san.value.get_values_for_type(x509.DNSName)
        print(f"SAN (DNS):  {', '.join(names[:5])}")
        if len(names) > 5:
            print(f"            ... and {len(names) - 5} more")
    except x509.ExtensionNotFound:
        print("SAN: Not present")

    # Signature algorithm
    print(f"Sig Algorithm: {cert.signature_algorithm_oid.dotted_string}")

    # Certificate fingerprint
    fingerprint = cert.fingerprint(hashes.SHA256())
    print(f"SHA256 Fingerprint: {fingerprint.hex()[:32]}...")

    # Public key info
    pub_key = cert.public_key()
    print(f"Public Key Type: {type(pub_key).__name__}")
    print(f"Key Size: {pub_key.key_size} bits")


if __name__ == "__main__":
    # Fetch and inspect a real certificate
    cert = fetch_certificate("www.google.com")
    inspect_certificate(cert)
```

### 2.4 자체 서명 vs. CA 서명

| 속성 | 자체 서명 | CA 서명 |
|------|----------|---------|
| 발급자 == 주체? | 예 | 아니오 |
| 브라우저 신뢰? | 아니오 (수동 추가 필요) | 예 (CA가 신뢰 저장소에 있으면) |
| 비용 | 무료 | 무료(Let's Encrypt)~고가(EV) |
| 사용 사례 | 개발, 내부 시스템 | 공개 서비스 |

---

## 3. 인증서 체인과 신뢰 계층

### 3.1 신뢰 체인(Chain of Trust)

브라우저와 운영체제는 **신뢰 저장소(trust store)**를 내장하고 있습니다 — 암묵적으로 신뢰되는 약 100~150개의 루트 CA 인증서 집합입니다. 루트 CA는 최종 엔티티 인증서에 직접 서명하는 경우가 드뭅니다. 대신 계층 구조는 다음과 같습니다:

```
Root CA (in trust store, 20+ year validity)
  └── Intermediate CA (signed by Root, 5-10 year validity)
        └── End-Entity Certificate (signed by Intermediate, 90 days - 1 year)
```

### 3.2 왜 중간 CA가 필요한가?

- **보안**: 루트 CA 개인 키는 HSM(Hardware Security Module)에서 오프라인으로 보관됩니다. 일상적인 서명에는 중간 CA 키만 사용됩니다.
- **폐지**: 중간 CA가 침해되더라도 해당 CA만 폐지하면 됩니다 — 루트와 다른 중간 CA는 신뢰를 유지합니다.
- **유연성**: 다른 중간 CA가 서로 다른 목적(DV, OV, EV 인증서)을 담당할 수 있습니다.

### 3.3 인증서 검증 알고리즘

브라우저가 인증서 체인을 수신하면 다음을 수행합니다:

1. **체인 구성**: 최종 엔티티 인증서에서 시작하여 발급자(Issuer) 필드를 따라가며 신뢰 저장소의 루트 CA까지 인증서를 찾습니다.
2. **서명 검증**: 각 인증서의 서명이 발급자의 공개 키로 유효해야 합니다.
3. **유효 기간 확인**: 체인의 모든 인증서가 유효 기간 내에 있어야 합니다.
4. **폐지 확인**: 체인의 어떤 인증서도 폐지되지 않았는지 확인합니다(CRL 또는 OCSP 사용).
5. **제약 확인**: 중간 CA는 기본 제약(Basic Constraints)에 `CA:TRUE`가 있어야 합니다. 경로 길이 제약을 준수합니다.
6. **이름 확인**: 최종 엔티티 인증서의 SAN에 요청된 도메인이 포함되어야 합니다.

```python
"""
Certificate chain verification (conceptual demonstration).

Why is chain building non-trivial? In practice, servers may send
certificates out of order, omit intermediates, or include unnecessary
certificates. Robust chain building requires handling all these cases.
"""

from cryptography import x509
from cryptography.x509.oid import ExtensionOID
from cryptography.hazmat.primitives.asymmetric import padding, ec
from cryptography.exceptions import InvalidSignature
from datetime import datetime, timezone


def verify_chain(chain: list[x509.Certificate],
                 trust_store: list[x509.Certificate]) -> bool:
    """
    Verify a certificate chain against a trust store.

    Parameters:
        chain: [end_entity, intermediate1, intermediate2, ...]
        trust_store: list of trusted root CA certificates
    """
    if not chain:
        return False

    now = datetime.now(timezone.utc)

    for i, cert in enumerate(chain):
        # Step 1: Check validity period
        if not (cert.not_valid_before_utc <= now <= cert.not_valid_after_utc):
            print(f"  Certificate {i} has expired or is not yet valid")
            return False

        # Step 2: Find the issuer (next cert in chain or trust store)
        if i + 1 < len(chain):
            issuer_cert = chain[i + 1]
        else:
            # Look in trust store
            issuer_cert = _find_issuer_in_store(cert, trust_store)
            if issuer_cert is None:
                print(f"  Certificate {i} issuer not found in trust store")
                return False

        # Step 3: Verify signature
        # Why try/except instead of boolean return?
        # The cryptography library raises exceptions for invalid
        # signatures rather than returning False, following the
        # principle that signature failure is exceptional.
        try:
            issuer_pub = issuer_cert.public_key()
            issuer_pub.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                # Padding depends on key type (RSA vs EC)
                _get_verification_params(issuer_pub, cert),
            )
            print(f"  Certificate {i}: signature valid")
        except (InvalidSignature, TypeError) as e:
            print(f"  Certificate {i}: signature INVALID ({e})")
            return False

        # Step 4: Check CA constraint for non-leaf certs
        if i > 0:
            try:
                bc = cert.extensions.get_extension_for_oid(
                    ExtensionOID.BASIC_CONSTRAINTS
                )
                if not bc.value.ca:
                    print(f"  Certificate {i}: not a CA but used as one!")
                    return False
            except x509.ExtensionNotFound:
                print(f"  Certificate {i}: missing Basic Constraints")
                return False

    print("  Chain verification: PASSED")
    return True


def _find_issuer_in_store(cert, trust_store):
    """Find the issuer of cert in the trust store."""
    for root in trust_store:
        if root.subject == cert.issuer:
            return root
    return None


def _get_verification_params(pub_key, cert):
    """Return appropriate verification parameters based on key type."""
    from cryptography.hazmat.primitives.asymmetric import rsa, ec
    if isinstance(pub_key, rsa.RSAPublicKey):
        return padding.PKCS1v15()
    # For EC keys, the signature algorithm is embedded in the cert
    return ec.ECDSA(cert.signature_hash_algorithm)
```

### 3.4 검증 수준별 인증서 유형

| 유형 | 검증 | 표시 | 비용 | 소요 시간 |
|------|------|------|------|---------|
| **DV** (도메인 검증) | 도메인 제어권 증명 | 자물쇠 아이콘 | 무료~$50 | 수 분 |
| **OV** (조직 검증) | 법적 실체 확인 | 인증서에 조직명 | $100~$500 | 수 일 |
| **EV** (확장 검증) | 광범위한 심사 | 녹색 바 (폐지됨) | $500~$2000 | 수 주 |

> **참고**: 주요 브라우저는 더 이상 EV를 DV와 다르게 표시하지 않습니다. 연구 결과 사용자들이 녹색 바를 인지하거나 이해하지 못한다는 것이 밝혀졌고, 대부분의 조직에서 EV의 추가 비용을 정당화하기 어렵습니다.

---

## 4. 인증서 폐지

### 4.1 왜 인증서를 폐지하는가?

개인 키가 침해된 경우, 자연 만료 이전에 인증서를 폐지해야 합니다. 일반적인 이유:
- 키 침해 (서버 해킹)
- CA 침해 (DigiNotar 사건, 2011년)
- 도메인 소유권 변경
- 오류로 발급된 인증서

### 4.2 인증서 폐지 목록(Certificate Revocation List, CRL)

CRL은 CA가 발행하는 폐지된 인증서 일련번호의 서명된 목록입니다.

**CRL의 문제점**:
- **크기**: 인기 있는 CA의 CRL은 메가바이트 크기로, 시간이 갈수록 커집니다
- **지연**: 클라이언트가 전체 CRL을 주기적으로 다운로드해야 합니다
- **최신성**: CRL에는 "다음 업데이트" 필드가 있어서 업데이트 사이에 폐지된 인증서가 여전히 수락될 수 있습니다

### 4.3 온라인 인증서 상태 프로토콜(Online Certificate Status Protocol, OCSP)

OCSP는 인증서별로 실시간 폐지 확인을 가능하게 합니다:

```
Client → OCSP Responder: "Is cert with serial 12345 revoked?"
OCSP Responder → Client: "Good" / "Revoked" / "Unknown" (signed response)
```

**CRL 대비 개선점**:
- 특정 인증서만 확인 (대량 다운로드 불필요)
- 더 신속한 응답

**OCSP의 문제점**:
- **프라이버시**: CA가 방문하는 모든 사이트를 알 수 있습니다 (OCSP 쿼리가 도메인을 노출)
- **가용성**: OCSP 응답자가 다운되면 클라이언트는 어떻게 해야 하나?
  - **강경 실패(Hard fail)**: 인증서 거부 (안전하지만 취약)
  - **연성 실패(Soft fail)**: 인증서 수락 (가용하지만 안전하지 않음 — OCSP를 차단할 수 있는 공격자가 폐지를 우회할 수 있음)

### 4.4 OCSP 스테이플링(OCSP Stapling)

서버가 직접 OCSP 응답을 가져와 TLS 핸드셰이크에 "스테이플"합니다:

```
Server → Client: Certificate + Signed OCSP Response (fresh, from CA)
```

**장점**:
- 프라이버시 노출 없음 (클라이언트가 CA에 연결하지 않음)
- 가용성 우려 없음 (OCSP 응답이 서버에 캐시됨)
- OCSP 응답이 CA에 의해 서명되므로 서버가 위조할 수 없음

### 4.5 단기 인증서

Let's Encrypt 인증서의 유효 기간은 90일에 불과하며 자동 갱신이 가능합니다. 이는 폐지 기간을 줄여줍니다 — 키가 침해되더라도 인증서가 곧 만료됩니다. 일부 제안에서는 24~48시간 동안만 유효한 인증서를 옹호하여 폐지 자체의 필요성을 없애자고 합니다.

---

## 5. 인증서 투명성

### 5.1 CT가 해결하는 문제

2011년, DigiNotar라는 CA가 침해되어 공격자들이 `*.google.com`에 대한 사기성 인증서를 발급했습니다. Google은 Chrome 브라우저가 Google 인증서에 대한 핀을 하드코딩하고 있었기 때문에 이를 발견했습니다. 이 우연한 확인이 없었다면, 사기성 인증서는 감지되지 않았을 것입니다.

**인증서 투명성(Certificate Transparency, CT)**은 모든 인증서가 공개적으로 로깅되어 도메인 소유자가 무단 발급을 감지할 수 있도록 보장합니다.

### 5.2 CT 동작 방식

```
1. CA issues certificate
2. CA submits certificate to CT logs (append-only, publicly auditable)
3. CT log returns a Signed Certificate Timestamp (SCT)
4. Certificate is embedded with SCTs (or they're delivered via TLS/OCSP)
5. Browser checks that the certificate has valid SCTs from multiple logs
```

### 5.3 CT 로그의 핵심 속성

- **추가 전용(Append-only)**: 인증서를 추가할 수 있지만 제거할 수 없습니다 (블록체인과 유사한 머클 트리로 강제)
- **공개 감사 가능**: 누구나 로그를 쿼리하고 일관성을 검증할 수 있습니다
- **다중 로그**: 인증서는 최소 2~3개의 독립적인 로그에 기록되어야 합니다

### 5.4 모니터링

도메인 소유자는 자신의 도메인에 발급된 인증서를 CT 로그에서 모니터링할 수 있습니다:

```python
"""
Query Certificate Transparency logs for a domain.

Why monitor CT logs? If a CA mistakenly or maliciously issues a
certificate for your domain, CT logs are the fastest way to detect it.
Services like crt.sh aggregate CT log data and make it searchable.
"""

import urllib.request
import json


def search_ct_logs(domain: str, limit: int = 10) -> list[dict]:
    """
    Search crt.sh (a CT log aggregator) for certificates
    issued to a given domain.
    """
    url = f"https://crt.sh/?q={domain}&output=json"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "CT-Monitor/1.0")
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())

        # Show recent certificates
        for entry in data[:limit]:
            print(f"  ID: {entry.get('id')}")
            print(f"  Issuer: {entry.get('issuer_name', 'N/A')}")
            print(f"  Not Before: {entry.get('not_before', 'N/A')}")
            print(f"  Not After: {entry.get('not_after', 'N/A')}")
            print(f"  Common Name: {entry.get('common_name', 'N/A')}")
            print()
        return data[:limit]
    except Exception as e:
        print(f"CT log query failed: {e}")
        return []


if __name__ == "__main__":
    # Example: check certificates issued for a domain
    print("=== CT Log Search: example.com ===")
    search_ct_logs("example.com", limit=5)
```

---

## 6. 신뢰의 웹 vs. 계층형 PKI

### 6.1 PGP의 신뢰의 웹(Web of Trust)

PGP(Pretty Good Privacy)는 분산형 신뢰 모델을 사용합니다:

- 사용자들이 서로의 공개 키에 서명합니다(키 서명)
- 신뢰는 전이적입니다: 앨리스가 밥을 신뢰하고, 밥이 캐럴의 키에 서명했다면, 앨리스는 캐럴의 키에 어느 정도 확신을 가집니다
- 중앙 기관 없음; 신뢰 결정은 개인이 내립니다

### 6.2 비교

| 측면 | 계층형 PKI (X.509) | 신뢰의 웹 (PGP) |
|------|-------------------|----------------|
| **신뢰 앵커** | 루트 CA (중앙화) | 개인 사용자 (분산화) |
| **확장성** | 우수 (브라우저에 루트 인증서 내장) | 불량 (수동 키 서명 필요) |
| **폐지** | CRL/OCSP 인프라 | 키 폐지 인증서 (수동) |
| **실패 모드** | CA 침해가 수백만 명에게 영향 | 한 사용자의 실수가 연락처에만 영향 |
| **사용 사례** | 웹 (TLS), 코드 서명 | 이메일 암호화, 파일 서명 |
| **사용자 부담** | 없음 (사용자에게 투명) | 높음 (신뢰 결정을 직접 평가해야 함) |

### 6.3 실제 신뢰 실패 사례

**PKI 실패:**
- DigiNotar (2011): CA 침해로 Google에 대한 사기성 인증서 발급
- Symantec (2017): 체계적인 무단 인증서 발급; 주요 브라우저 모두 신뢰 제거
- 카자흐스탄 (2020): 정부가 모든 HTTPS 트래픽을 가로채기 위해 국가 루트 CA 설치 시도

**신뢰의 웹 실패:**
- 키 서명 파티가 대규모로는 비현실적
- 신뢰 그래프가 사회적 연결을 노출 (메타데이터 유출)
- 대부분의 사용자가 지문을 확인하지 않음

### 6.4 현대적 대안

- **TOFU (Trust On First Use, 최초 사용 시 신뢰)**: 첫 접촉 시 키를 수락하고 변경 시 경고 (SSH 모델)
- **키 투명성(Key Transparency)**: CT 스타일 로그와 종단간 암호화 키 관리를 결합한 Google의 제안
- **분산 신원(Decentralized Identity, DIDs)**: 검증 가능한 자격 증명을 사용한 자기 주권 신원을 위한 W3C 표준

---

## 7. Let's Encrypt와 ACME

### 7.1 Let's Encrypt 이전의 세계

2015년 이전에는 HTTPS 인증서가 연간 $50~$300의 비용이 들었고, 수동 검증이 필요했으며, 1~3년 후 만료되었습니다. 그 결과 웹 트래픽의 약 40%만이 암호화되었습니다.

### 7.2 Let's Encrypt의 혁명

Let's Encrypt(2015년 출시)는 다음을 제공합니다:
- **무료** DV 인증서
- ACME 프로토콜을 통한 **자동화** 발급 및 갱신
- **90일 유효 기간** (자동화를 촉진하고 침해 기간 제한)
- **오픈**: 모든 인증서가 CT에 로깅됨

2024년까지 Let's Encrypt는 40억 개 이상의 인증서를 발급했으며 모든 웹 페이지의 약 60%를 암호화합니다.

### 7.3 ACME 프로토콜 (RFC 8555)

ACME(Automatic Certificate Management Environment)는 전체 인증서 수명 주기를 자동화합니다:

```
1. Client → CA: "I want a cert for example.com" (new order)
2. CA → Client: "Prove you control example.com" (challenge)
3. Client proves control via:
   - HTTP-01: Place a file at http://example.com/.well-known/acme-challenge/<token>
   - DNS-01: Create a TXT record _acme-challenge.example.com
   - TLS-ALPN-01: Respond on port 443 with a specific self-signed cert
4. CA verifies the challenge
5. Client → CA: Submit CSR (Certificate Signing Request)
6. CA → Client: Signed certificate
```

```python
"""
Simplified ACME workflow demonstration.

Why HTTP-01 is the most common challenge type:
  It requires only a running web server on port 80, no DNS access.
  DNS-01 is needed for wildcard certificates (*.example.com).
"""

import hashlib
import json
import base64


def simulate_acme_challenge():
    """Demonstrate the ACME HTTP-01 challenge concept."""

    # Step 1: Generate account key (in practice, an RSA or EC key pair)
    # The account key proves ownership of the ACME account
    account_key_thumbprint = hashlib.sha256(
        b"simulated-jwk-thumbprint"
    ).hexdigest()[:32]

    # Step 2: CA provides a token
    token = base64.urlsafe_b64encode(
        hashlib.sha256(b"random-challenge-bytes").digest()
    ).decode().rstrip("=")

    # Step 3: Client constructs the key authorization
    # Why token + thumbprint? The token proves the CA issued this challenge,
    # and the thumbprint proves the ACME account owner placed the response.
    key_authorization = f"{token}.{account_key_thumbprint}"

    print("=== ACME HTTP-01 Challenge ===")
    print(f"Token: {token}")
    print(f"Key Authorization: {key_authorization}")
    print(f"\nThe client must serve this at:")
    print(f"  http://example.com/.well-known/acme-challenge/{token}")
    print(f"\nResponse body: {key_authorization}")
    print(f"\nThe CA will verify by making an HTTP GET to that URL.")

    return token, key_authorization


if __name__ == "__main__":
    simulate_acme_challenge()
```

---

## 8. 인증서 핀닝과 DANE

### 8.1 인증서 핀닝(Certificate Pinning)

PKI가 있더라도 약 100개 이상의 신뢰할 수 있는 루트 CA 중 어느 것이든 어떤 도메인에 대한 인증서를 발급할 수 있습니다. 인증서 **핀닝**은 특정 도메인에 유효한 CA(또는 특정 인증서/키)를 제한합니다.

**핀닝 유형:**
- **공개 키 핀닝(HPKP, HTTP Public Key Pinning)**: 사이트에 유효한 공개 키를 지정하는 HTTP 헤더. 잘못된 구성이 사용자를 영구적으로 차단할 수 있어 2018년에 **폐지**되었습니다.
- **내장 핀(Built-in pins)**: 브라우저가 고가치 도메인(Google, Firefox 업데이트)에 대한 핀을 하드코딩
- **모바일 앱 핀닝**: 앱이 예상 서버 인증서/키를 내장하고 다른 모든 것을 거부

### 8.2 DANE (DNS-Based Authentication of Named Entities)

DANE는 DNSSEC 서명된 DNS 레코드를 사용하여 도메인에 유효한 인증서나 CA를 지정합니다:

```
_443._tcp.example.com. IN TLSA 3 1 1 <sha256-hash-of-public-key>
```

**TLSA 레코드 필드:**
- **Usage (사용)**: 0~3 (CA 제약, 서비스 인증서 제약, 신뢰 앵커 어서션, 도메인 발급 인증서)
- **Selector (선택자)**: 0 (전체 인증서) 또는 1 (공개 키만)
- **Matching type (매칭 유형)**: 0 (정확), 1 (SHA-256), 2 (SHA-512)

**DANE의 장점:**
- 도메인 소유자가 신뢰를 제어 (어떤 CA에도 의존하지 않음)
- CA가 침해되더라도 동작

**한계:**
- DNSSEC 필요 (보편적으로 배포되지 않음)
- 브라우저의 채택이 느림 (DANE는 이메일/SMTP에서 더 많이 사용됨)

### 8.3 Expect-CT 헤더

`Expect-CT` HTTP 헤더는 브라우저에 해당 도메인에 대한 인증서 투명성을 요구하도록 지시합니다:

```
Expect-CT: max-age=86400, enforce, report-uri="https://example.com/ct-report"
```

이는 CT 로깅 없이 사기성 인증서가 발급될 경우 브라우저가 거부하도록 보장합니다.

---

## 9. 요약

| 개념 | 핵심 내용 |
|------|---------|
| 신뢰 문제 | 키 교환에는 인증이 필요하며; PKI가 이를 제공함 |
| X.509 인증서 | CA 서명을 통해 공개 키를 신원에 바인딩 |
| 인증서 체인 | 루트 CA → 중간 CA → 최종 엔티티 (심층 방어) |
| CRL/OCSP | 폐지는 PKI에서 가장 어려운 문제; OCSP 스테이플링이 최선의 해결책 |
| 인증서 투명성 | 공개적, 추가 전용 로그가 사기성 인증서를 감지 |
| 신뢰의 웹 | 분산형 대안; 웹에서는 확장성 없음 |
| Let's Encrypt/ACME | 무료, 자동화된 인증서; 웹 보안 변혁 |
| 핀닝/DANE | CA 신뢰를 제한하는 보조 메커니즘 |

---

## 10. 연습 문제

### 연습 1: 인증서 검사 (코딩)

다음을 수행하는 Python 스크립트를 작성하세요:
1. 임의의 HTTPS 웹사이트(예: `github.com`)에서 인증서 체인을 가져오기
2. 체인의 각 인증서에 대해 주체, 발급자, 유효 기간, SAN 출력
3. 체인의 각 서명 검증
4. 체인이 신뢰할 수 있는 루트 CA로 연결되는지 보고

### 연습 2: 자체 서명 CA (코딩)

`cryptography` 라이브러리를 사용하여 다음을 생성하세요:
1. 루트 CA 키 쌍과 자체 서명 인증서
2. 중간 CA 키 쌍과 인증서 (루트가 서명)
3. `test.example.com`에 대한 최종 엔티티 인증서 (중간 CA가 서명)
4. 완전한 체인을 프로그래밍으로 검증

### 연습 3: CT 로그 분석 (개념 + 코딩)

소유한 도메인(또는 `github.com` 같은 잘 알려진 도메인)에 대해 `crt.sh`를 쿼리하세요:
1. 지난 1년간 발급된 인증서는 몇 개인가?
2. 어떤 CA가 발급했는가?
3. 예기치 않은 인증서(잠재적 오발급)가 있는가?
4. 예기치 않은 CA에서 새 인증서가 나타나면 경고하는 모니터링 스크립트 작성

### 연습 4: 폐지 트레이드오프 (개념)

다음 차원에서 CRL, OCSP, OCSP 스테이플링, 단기 인증서를 비교하세요:
- 클라이언트 대역폭 비용
- 클라이언트 지연 영향
- 프라이버시 영향
- 폐지 확인을 차단할 수 있는 공격자에 대한 보안
- 서버 측 복잡성

비교 표를 작성하고 다음에 대한 최선의 접근 방식을 추천하세요: (a) 고트래픽 웹 서비스, (b) IoT 장치, (c) 모바일 앱.

### 연습 5: PKI 설계 과제 (심화)

다음이 필요한 50,000명의 사용자를 가진 대학교를 위한 PKI를 설계하세요:
- 이메일 암호화 (S/MIME)
- VPN 인증
- 문서 서명
- Wi-Fi (802.1X) 인증

설계에서 다음을 다루어야 합니다:
- 인증서 계층 (CA가 몇 개? 어떤 신뢰 수준?)
- 키 보관 (개인 키는 어디에? HSM? 스마트 카드? 소프트웨어?)
- 폐지 전략
- 인증서 수명 주기 (발급, 갱신, 폐지)
- 재해 복구 (루트 CA가 침해되면?)
