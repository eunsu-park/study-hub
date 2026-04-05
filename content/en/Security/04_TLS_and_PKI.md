# TLS/SSL and Public Key Infrastructure

**Previous**: [03. Hashing and Data Integrity](./03_Hashing_and_Integrity.md) | **Next**: [05. Authentication Systems](./05_Authentication.md)

---

Transport Layer Security (TLS) is the protocol that makes HTTPS possible. It secures virtually all web traffic, email, messaging, VPN, and API communication. This lesson provides a deep technical walkthrough of TLS 1.3, the certificate trust model (PKI), practical OpenSSL and Python usage, mutual TLS, and common misconfigurations that undermine security.

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the complete TLS 1.3 handshake step by step
2. Explain certificate chains and the web of trust
3. Work with X.509 certificates using OpenSSL and Python
4. Understand Certificate Authorities and certificate issuance
5. Configure Let's Encrypt with the ACME protocol
6. Implement certificate pinning and mutual TLS (mTLS)
7. Create self-signed certificates for development
8. Identify and fix common TLS misconfigurations
9. Test TLS configurations using standard tools

---

## Table of Contents

1. [TLS Overview and History](#1-tls-overview-and-history)
2. [TLS 1.3 Handshake in Detail](#2-tls-13-handshake-in-detail)
3. [Cipher Suites](#3-cipher-suites)
4. [X.509 Certificates](#4-x509-certificates)
5. [Certificate Chains and Trust](#5-certificate-chains-and-trust)
6. [Certificate Authorities (CA)](#6-certificate-authorities-ca)
7. [Let's Encrypt and ACME](#7-lets-encrypt-and-acme)
8. [Certificate Pinning](#8-certificate-pinning)
9. [Mutual TLS (mTLS)](#9-mutual-tls-mtls)
10. [Python TLS Programming](#10-python-tls-programming)
11. [Creating Certificates with OpenSSL](#11-creating-certificates-with-openssl)
12. [Common Misconfigurations](#12-common-misconfigurations)
13. [Testing TLS Configurations](#13-testing-tls-configurations)
14. [Exercises](#14-exercises)
15. [References](#15-references)

---

## 1. TLS Overview and History

### 1.1 Protocol Evolution

```
┌─────────────────────────────────────────────────────────────────────┐
│                  TLS/SSL Version History                              │
├──────────┬──────┬───────────┬────────────────────────────────────────┤
│ Protocol │ Year │ Status    │ Notes                                  │
├──────────┼──────┼───────────┼────────────────────────────────────────┤
│ SSL 1.0  │ 1994 │ NEVER REL.│ Never released (serious flaws)        │
│ SSL 2.0  │ 1995 │ DEPRECATED│ Completely broken, disable everywhere │
│ SSL 3.0  │ 1996 │ DEPRECATED│ POODLE attack (2014), broken          │
│ TLS 1.0  │ 1999 │ DEPRECATED│ BEAST attack (2011), end-of-life 2020│
│ TLS 1.1  │ 2006 │ DEPRECATED│ No known attacks, but weak ciphers   │
│ TLS 1.2  │ 2008 │ SUPPORTED │ Still widely used, secure when       │
│          │      │           │ properly configured                   │
│ TLS 1.3  │ 2018 │ PREFERRED │ Major redesign: faster, more secure  │
│          │      │           │ Removed all weak algorithms           │
├──────────┴──────┴───────────┴────────────────────────────────────────┤
│                                                                      │
│ CURRENT RECOMMENDATION (2025+):                                     │
│ • Enable TLS 1.3 (preferred) and TLS 1.2 (fallback)               │
│ • Disable everything below TLS 1.2                                  │
│ • Use only strong cipher suites                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Where TLS Sits in the Stack

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Protocol Stack with TLS                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Application Layer                                           │    │
│  │  HTTP, SMTP, IMAP, FTP, MQTT, gRPC, WebSocket              │    │
│  └────────────────────────────┬────────────────────────────────┘    │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐    │
│  │  TLS (Transport Layer Security)                              │    │
│  │  ┌─────────────┐  ┌───────────────┐  ┌─────────────────┐   │    │
│  │  │  Handshake  │  │ Record       │  │  Alert          │   │    │
│  │  │  Protocol   │  │ Protocol     │  │  Protocol       │   │    │
│  │  └─────────────┘  └───────────────┘  └─────────────────┘   │    │
│  └────────────────────────────┬────────────────────────────────┘    │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐    │
│  │  Transport Layer (TCP)                                       │    │
│  │  Port 443 (HTTPS), 465 (SMTPS), 993 (IMAPS), etc.          │    │
│  └────────────────────────────┬────────────────────────────────┘    │
│                               │                                     │
│  ┌────────────────────────────▼────────────────────────────────┐    │
│  │  Internet Layer (IP)                                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 What TLS Provides

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TLS Security Properties                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. CONFIDENTIALITY                                                 │
│     ├── Data encrypted in transit (AES-GCM, ChaCha20-Poly1305)     │
│     └── Eavesdroppers see only encrypted gibberish                  │
│                                                                      │
│  2. INTEGRITY                                                       │
│     ├── AEAD ciphers authenticate every record                      │
│     └── Any tampering detected immediately                          │
│                                                                      │
│  3. AUTHENTICATION                                                  │
│     ├── Server proves identity via X.509 certificate                │
│     ├── Optional: client authentication (mTLS)                      │
│     └── Prevents man-in-the-middle attacks                          │
│                                                                      │
│  4. FORWARD SECRECY (TLS 1.3 mandatory)                            │
│     ├── Ephemeral key exchange (ECDHE)                              │
│     └── Past sessions cannot be decrypted even if long-term         │
│         private key is later compromised                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. TLS 1.3 Handshake in Detail

TLS 1.3 dramatically simplified the handshake compared to TLS 1.2, reducing it from 2 round trips to 1 (or 0 with 0-RTT).

### 2.1 Full Handshake (1-RTT)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    TLS 1.3 Full Handshake (1-RTT)                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Client                                          Server             │
│   ──────                                          ──────             │
│                                                                      │
│   ┌─ ClientHello ───────────────────────────────▶                   │
│   │  • Supported TLS versions (1.3)                                 │
│   │  • Cipher suites (e.g., TLS_AES_256_GCM_SHA384)               │
│   │  • Key shares (X25519 public key)  ←── NEW in 1.3             │
│   │  • Supported groups (X25519, P-256)                            │
│   │  • Signature algorithms (Ed25519, ECDSA-P256-SHA256)           │
│   │  • Random (32 bytes)                                           │
│   │  • SNI (Server Name Indication)                                │
│   └──────────────────────────────────────────────                   │
│                                                                      │
│           ┌─ ServerHello ◀──────────────────────────                │
│           │  • Selected TLS version (1.3)                           │
│           │  • Selected cipher suite                                │
│           │  • Server key share (X25519 public key)                 │
│           │  • Random (32 bytes)                                    │
│           └──────────────────────────────────────                   │
│                                                                      │
│           ┌─ {EncryptedExtensions} ◀──────────────── (encrypted)   │
│           │  • Server extensions                                    │
│           └──────────────────────────────────────                   │
│                                                                      │
│           ┌─ {Certificate} ◀──────────────────────── (encrypted)   │
│           │  • Server's X.509 certificate chain                    │
│           └──────────────────────────────────────                   │
│                                                                      │
│           ┌─ {CertificateVerify} ◀────────────────── (encrypted)   │
│           │  • Digital signature over handshake transcript          │
│           │  • Proves server owns the private key                  │
│           └──────────────────────────────────────                   │
│                                                                      │
│           ┌─ {Finished} ◀──────────────────────────── (encrypted)  │
│           │  • MAC over entire handshake transcript                │
│           └──────────────────────────────────────                   │
│                                                                      │
│   ┌─ {Finished} ──────────────────────────────────▶ (encrypted)    │
│   │  • Client's MAC over handshake transcript                      │
│   └──────────────────────────────────────────────                   │
│                                                                      │
│   ═══════════════════════════════════════════════════                │
│   Application Data flows in both directions (encrypted)             │
│   ═══════════════════════════════════════════════════                │
│                                                                      │
│   Key insight: In TLS 1.3, the key exchange happens in the FIRST   │
│   message (ClientHello includes key shares), so all subsequent      │
│   messages after ServerHello are encrypted.                         │
│                                                                      │
│   Compare to TLS 1.2: 2-RTT handshake, certificate sent in        │
│   plaintext, no mandatory forward secrecy.                          │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Derivation in TLS 1.3

```
┌──────────────────────────────────────────────────────────────────────┐
│              TLS 1.3 Key Schedule (Simplified)                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ECDHE shared secret (from X25519 key exchange)                     │
│            │                                                        │
│            ▼                                                        │
│  ┌─────────────────┐                                                │
│  │  HKDF-Extract   │ ← IKM = ECDHE shared secret                  │
│  │  (Early Secret) │    Salt = 0                                    │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  HKDF-Extract   │ ← IKM = ECDHE shared secret                  │
│  │(Handshake Secret)│   Salt = derived from early secret            │
│  └────────┬────────┘                                                │
│           │                                                         │
│      ┌────┴────┐                                                    │
│      │         │                                                    │
│      ▼         ▼                                                    │
│  ┌────────┐ ┌────────┐                                              │
│  │Client  │ │Server  │ ← HKDF-Expand-Label                        │
│  │Handshk │ │Handshk │   (derive traffic keys from                 │
│  │Traffic │ │Traffic │    handshake secret + transcript hash)       │
│  │Key/IV  │ │Key/IV  │                                              │
│  └────────┘ └────────┘                                              │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │  HKDF-Extract   │ ← IKM = 0                                    │
│  │ (Master Secret) │    Salt = derived from handshake secret       │
│  └────────┬────────┘                                                │
│           │                                                         │
│      ┌────┴────┐                                                    │
│      │         │                                                    │
│      ▼         ▼                                                    │
│  ┌────────┐ ┌────────┐                                              │
│  │Client  │ │Server  │ ← HKDF-Expand-Label                        │
│  │App     │ │App     │   (derive application traffic keys)         │
│  │Traffic │ │Traffic │                                              │
│  │Key/IV  │ │Key/IV  │                                              │
│  └────────┘ └────────┘                                              │
│                                                                      │
│  Each direction gets its OWN key and IV.                            │
│  Keys are derived from the shared secret AND the handshake          │
│  transcript, binding them to this specific session.                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.3 TLS 1.3 vs TLS 1.2

```
┌─────────────────────────────────────────────────────────────────────┐
│              TLS 1.3 vs TLS 1.2 Improvements                        │
├─────────────────────┬─────────────────────┬─────────────────────────┤
│ Feature             │ TLS 1.2             │ TLS 1.3                 │
├─────────────────────┼─────────────────────┼─────────────────────────┤
│ Handshake RTT       │ 2 round trips       │ 1 round trip            │
│ 0-RTT resumption    │ No                  │ Yes (optional)          │
│ Forward secrecy     │ Optional (ECDHE)    │ Mandatory (always)      │
│ RSA key exchange    │ Supported           │ Removed (no FS)         │
│ Static DH           │ Supported           │ Removed (no FS)         │
│ CBC mode ciphers    │ Supported           │ Removed (padding oracle)│
│ RC4                 │ Supported (weak)    │ Removed (broken)        │
│ Compression         │ Supported           │ Removed (CRIME attack)  │
│ Renegotiation       │ Supported           │ Removed (complexity)    │
│ Encrypt cert?       │ No (plaintext)      │ Yes (encrypted)         │
│ Cipher suites       │ 300+ (many weak)    │ 5 (all strong)          │
│ Change Cipher Spec  │ Separate message    │ Removed                 │
│ Key derivation      │ PRF (TLS 1.2 PRF)  │ HKDF (stronger)         │
├─────────────────────┴─────────────────────┴─────────────────────────┤
│ TLS 1.3 is simpler, faster, and has no known weak configurations.  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.4 0-RTT (Zero Round Trip Time) Resumption

```
┌──────────────────────────────────────────────────────────────────────┐
│                 TLS 1.3 0-RTT Resumption                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  After a full handshake, server sends a session ticket.             │
│  On reconnection, client can send data immediately:                 │
│                                                                      │
│   Client                                          Server            │
│   ──────                                          ──────            │
│                                                                      │
│   ┌─ ClientHello + EarlyData ────────────────────▶                  │
│   │  • Session ticket (from previous connection)                    │
│   │  • 0-RTT application data (encrypted with PSK)                 │
│   └──────────────────────────────────────────────                   │
│                                                                      │
│   ⚠ WARNING: 0-RTT data is NOT replay-safe!                        │
│   An attacker can replay the ClientHello + EarlyData.              │
│                                                                      │
│   Rules for 0-RTT:                                                  │
│   • ONLY use for idempotent requests (GET, not POST)               │
│   • Server must implement replay protection                         │
│   • Limit 0-RTT data size                                          │
│   • Many deployments disable 0-RTT entirely                        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Cipher Suites

### 3.1 TLS 1.3 Cipher Suites

TLS 1.3 has only 5 cipher suites, all of which are AEAD:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  TLS 1.3 Cipher Suites                               │
├─────────────────────────────────────────┬───────────────────────────┤
│ Cipher Suite                            │ Notes                     │
├─────────────────────────────────────────┼───────────────────────────┤
│ TLS_AES_256_GCM_SHA384                  │ Strong, widely supported  │
│ TLS_AES_128_GCM_SHA256                  │ Default, very fast w/ HW │
│ TLS_CHACHA20_POLY1305_SHA256            │ Best for mobile/no AES-NI│
│ TLS_AES_128_CCM_SHA256                  │ For constrained devices  │
│ TLS_AES_128_CCM_8_SHA256                │ Short tag, IoT only      │
├─────────────────────────────────────────┴───────────────────────────┤
│                                                                      │
│ In TLS 1.3, the cipher suite ONLY specifies:                       │
│  • AEAD algorithm (AES-GCM, ChaCha20-Poly1305, AES-CCM)           │
│  • Hash for HKDF (SHA-256, SHA-384)                                │
│                                                                      │
│ Key exchange and signature algorithms are negotiated separately    │
│ via extensions (supported_groups, signature_algorithms).            │
│                                                                      │
│ Recommended priority:                                               │
│  1. TLS_AES_256_GCM_SHA384                                        │
│  2. TLS_CHACHA20_POLY1305_SHA256                                   │
│  3. TLS_AES_128_GCM_SHA256                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Understanding TLS 1.2 Cipher Suite Names

For legacy compatibility, understanding TLS 1.2 cipher suite naming is important:

```
TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
│    │      │        │   │    │    │
│    │      │        │   │    │    └── PRF hash (SHA-384)
│    │      │        │   │    └─────── AEAD mode (GCM)
│    │      │        │   └──────────── Key size (256-bit)
│    │      │        └──────────────── Encryption (AES)
│    │      └───────────────────────── Authentication (RSA cert)
│    └──────────────────────────────── Key Exchange (ECDHE = forward secrecy)
└───────────────────────────────────── Protocol (TLS)

GOOD TLS 1.2 cipher suites (all have ECDHE for forward secrecy):
  TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
  TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
  TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256
  TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256
  TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256

BAD TLS 1.2 cipher suites (disable these):
  TLS_RSA_WITH_AES_256_CBC_SHA256        (no forward secrecy, CBC)
  TLS_RSA_WITH_3DES_EDE_CBC_SHA          (no FS, 3DES, CBC)
  TLS_RSA_WITH_RC4_128_SHA               (no FS, RC4 broken)
  TLS_DHE_RSA_WITH_AES_256_CBC_SHA       (CBC, weak DH possible)
```

---

## 4. X.509 Certificates

An X.509 certificate binds a public key to an identity (domain name, organization). It is the foundation of TLS authentication.

### 4.1 Certificate Structure

```
┌──────────────────────────────────────────────────────────────────────┐
│                    X.509 v3 Certificate Structure                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  tbsCertificate (To Be Signed):                                     │
│  ├── Version: v3 (most common)                                      │
│  ├── Serial Number: unique identifier from CA                       │
│  ├── Signature Algorithm: e.g., SHA256withRSA, Ed25519              │
│  ├── Issuer: DN of the CA that signed this cert                     │
│  │   └── CN=Let's Encrypt Authority X3, O=Let's Encrypt, C=US     │
│  ├── Validity:                                                      │
│  │   ├── Not Before: 2026-01-01 00:00:00 UTC                      │
│  │   └── Not After:  2026-03-31 23:59:59 UTC                      │
│  ├── Subject: DN of the certificate holder                          │
│  │   └── CN=www.example.com                                        │
│  ├── Subject Public Key Info:                                       │
│  │   ├── Algorithm: RSA (2048-bit) or ECDSA (P-256)               │
│  │   └── Public Key: [actual public key bytes]                     │
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
│      │   └── CA: FALSE (this is a leaf/end-entity cert)            │
│      ├── Authority Key Identifier: [CA's key ID]                   │
│      ├── CRL Distribution Points: http://crl.example.com/ca.crl   │
│      └── Authority Information Access:                              │
│          ├── OCSP: http://ocsp.example.com                         │
│          └── CA Issuers: http://crt.example.com/ca.crt             │
│                                                                      │
│  Signature Algorithm: SHA256withRSA                                  │
│  Signature Value: [CA's digital signature over tbsCertificate]      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Reading Certificates in Python

```python
from cryptography import x509
from cryptography.x509.oid import NameOID, ExtensionOID
from cryptography.hazmat.primitives import hashes
import ssl
import socket
from datetime import datetime, timezone

def get_server_certificate(hostname: str, port: int = 443) -> x509.Certificate:
    """Retrieve and parse a server's TLS certificate."""
    # Connect and get certificate in DER format
    context = ssl.create_default_context()
    with socket.create_connection((hostname, port), timeout=10) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            der_cert = ssock.getpeercert(binary_form=True)
    return x509.load_der_x509_certificate(der_cert)

def print_certificate_info(cert: x509.Certificate):
    """Display detailed certificate information."""
    print("=" * 70)
    print("X.509 CERTIFICATE DETAILS")
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
    print(f"  Status:     {status} ({days_remaining} days remaining)")

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
        # Alternative approach for SAN
        for name in san.value:
            print(f"  SAN: {name.value}")
    except x509.ExtensionNotFound:
        print("  No SAN extension")

    try:
        basic = cert.extensions.get_extension_for_oid(ExtensionOID.BASIC_CONSTRAINTS)
        print(f"  Basic Constraints: CA={basic.value.ca}")
    except x509.ExtensionNotFound:
        pass

# Example: Inspect a real certificate
try:
    cert = get_server_certificate("www.google.com")
    print_certificate_info(cert)
except Exception as e:
    print(f"Could not fetch certificate: {e}")
    print("(This requires network access)")
```

---

## 5. Certificate Chains and Trust

### 5.1 Chain of Trust

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Certificate Chain of Trust                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  ROOT CA Certificate (Self-signed)                        │       │
│  │  ├── Issuer: "Root CA"                                    │       │
│  │  ├── Subject: "Root CA"  (Issuer == Subject → self-signed)│       │
│  │  ├── Valid: 20-30 years                                   │       │
│  │  ├── Stored in OS/browser trust store                     │       │
│  │  └── Signs: Intermediate CA certificate                   │       │
│  └────────────────────────┬─────────────────────────────────┘       │
│                           │ signs                                   │
│                           ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  INTERMEDIATE CA Certificate                              │       │
│  │  ├── Issuer: "Root CA"                                    │       │
│  │  ├── Subject: "Intermediate CA"                           │       │
│  │  ├── Valid: 5-10 years                                    │       │
│  │  ├── Basic Constraints: CA=TRUE                           │       │
│  │  └── Signs: End-entity (leaf) certificates               │       │
│  └────────────────────────┬─────────────────────────────────┘       │
│                           │ signs                                   │
│                           ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  END-ENTITY (LEAF) Certificate                            │       │
│  │  ├── Issuer: "Intermediate CA"                            │       │
│  │  ├── Subject: "www.example.com"                           │       │
│  │  ├── Valid: 90 days (Let's Encrypt) to 1 year             │       │
│  │  ├── Basic Constraints: CA=FALSE                          │       │
│  │  └── This is your server's certificate                   │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
│  VERIFICATION PROCESS:                                              │
│  1. Receive leaf cert + intermediate cert from server               │
│  2. Verify leaf cert signature using intermediate's public key      │
│  3. Verify intermediate cert signature using root's public key      │
│  4. Check root cert is in the local trust store                    │
│  5. Check all certs are within validity period                     │
│  6. Check revocation status (CRL or OCSP)                          │
│  7. Check leaf cert's SAN matches the requested hostname           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 Certificate Chain Verification in Python

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
    """Create a self-signed CA certificate."""
    # Generate CA key pair
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
    """Create a server certificate signed by the CA."""
    # Generate server key pair
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
    Verify a certificate chain manually.
    In production, use ssl.SSLContext which handles this automatically.
    """
    results = {
        "valid": True,
        "checks": [],
        "errors": [],
    }

    now = datetime.now(timezone.utc)

    # Build the chain: leaf → intermediates → root
    chain = [leaf_cert] + intermediate_certs

    for i, cert in enumerate(chain):
        cert_name = "Leaf" if i == 0 else f"Intermediate {i}"

        # Check validity period
        if cert.not_valid_before_utc > now:
            results["errors"].append(f"{cert_name}: Not yet valid")
            results["valid"] = False
        elif cert.not_valid_after_utc < now:
            results["errors"].append(f"{cert_name}: Expired")
            results["valid"] = False
        else:
            days_left = (cert.not_valid_after_utc - now).days
            results["checks"].append(f"{cert_name}: Valid ({days_left} days remaining)")

        # Verify signature (issuer signed this cert)
        if i < len(chain) - 1:
            # Next cert in chain should be the issuer
            issuer_cert = chain[i + 1]
        else:
            # Last cert should be signed by a trusted root
            issuer_cert = None
            for root in trusted_roots:
                if root.subject == cert.issuer:
                    issuer_cert = root
                    break

            if issuer_cert is None:
                results["errors"].append(f"{cert_name}: Issuer not found in trust store")
                results["valid"] = False
                continue

        # Verify digital signature
        try:
            issuer_pub = issuer_cert.public_key()
            issuer_pub.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cert.signature_hash_algorithm,
            )
            results["checks"].append(f"{cert_name}: Signature valid (signed by {issuer_cert.subject.rfc4514_string()})")
        except Exception as e:
            results["errors"].append(f"{cert_name}: Signature verification failed: {e}")
            results["valid"] = False

    # Check hostname against SAN
    if hostname and results["valid"]:
        try:
            san = leaf_cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            dns_names = san.value.get_attributes_for_oid(x509.DNSName) if hasattr(san.value, 'get_attributes_for_oid') else []
            names = [name.value for name in san.value]
            if hostname in names or any(
                name.startswith("*.") and hostname.endswith(name[1:])
                for name in names
            ):
                results["checks"].append(f"Hostname '{hostname}' matches SAN")
            else:
                results["errors"].append(f"Hostname '{hostname}' not in SAN: {names}")
                results["valid"] = False
        except x509.ExtensionNotFound:
            results["errors"].append("No SAN extension found")
            results["valid"] = False

    return results

# Create a complete certificate chain
print("Creating Certificate Chain")
print("=" * 60)

# 1. Create Root CA
ca_key, ca_cert = create_ca_certificate("Demo Root CA")
print(f"Root CA: {ca_cert.subject.rfc4514_string()}")

# 2. Create server certificate
server_key, server_cert = create_server_certificate(
    ca_key, ca_cert, "example.com"
)
print(f"Server:  {server_cert.subject.rfc4514_string()}")

# 3. Verify the chain
result = verify_certificate_chain(
    leaf_cert=server_cert,
    intermediate_certs=[],  # Direct CA signing (no intermediate)
    trusted_roots=[ca_cert],
    hostname="example.com",
)

print(f"\nChain Verification: {'VALID' if result['valid'] else 'INVALID'}")
for check in result["checks"]:
    print(f"  [OK] {check}")
for error in result["errors"]:
    print(f"  [!!] {error}")

# Test with wrong hostname
result2 = verify_certificate_chain(
    leaf_cert=server_cert,
    intermediate_certs=[],
    trusted_roots=[ca_cert],
    hostname="evil.com",
)
print(f"\nWrong hostname: {'VALID' if result2['valid'] else 'INVALID'}")
for error in result2["errors"]:
    print(f"  [!!] {error}")
```

---

## 6. Certificate Authorities (CA)

### 6.1 How CAs Work

```
┌──────────────────────────────────────────────────────────────────────┐
│              Certificate Authority Process                            │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. DOMAIN OWNER generates key pair and CSR                         │
│     ┌─────────────┐                                                 │
│     │ Private Key  │ (keep secret!)                                 │
│     │ Public Key   │ → embedded in CSR                              │
│     │ CSR          │ → sent to CA                                   │
│     └─────────────┘                                                 │
│            │                                                        │
│            ▼                                                        │
│  2. CA VALIDATES domain ownership                                   │
│     ┌─────────────────────────────────────────────┐                │
│     │  DV (Domain Validation):                     │                │
│     │    HTTP challenge, DNS challenge, or email    │                │
│     │                                               │                │
│     │  OV (Organization Validation):                │                │
│     │    DV + verify organization identity          │                │
│     │                                               │                │
│     │  EV (Extended Validation):                    │                │
│     │    OV + extensive legal/physical verification │                │
│     └─────────────────────────────────────────────┘                │
│            │                                                        │
│            ▼                                                        │
│  3. CA SIGNS the certificate                                        │
│     ┌─────────────┐                                                 │
│     │ CA creates   │                                                │
│     │ X.509 cert  │ → signed with CA's private key                 │
│     │ with owner's│                                                 │
│     │ public key  │                                                 │
│     └─────────────┘                                                 │
│            │                                                        │
│            ▼                                                        │
│  4. DOMAIN OWNER installs cert on server                           │
│     Server sends cert chain during TLS handshake                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 Certificate Revocation

```
┌─────────────────────────────────────────────────────────────────────┐
│              Certificate Revocation Methods                          │
├──────────────────┬──────────────────────────────────────────────────┤
│ Method           │ Description                                      │
├──────────────────┼──────────────────────────────────────────────────┤
│ CRL              │ Certificate Revocation List                      │
│ (RFC 5280)       │ • CA publishes a signed list of revoked certs   │
│                  │ • Clients download and check the list            │
│                  │ • Problem: lists grow large, stale data          │
├──────────────────┼──────────────────────────────────────────────────┤
│ OCSP             │ Online Certificate Status Protocol               │
│ (RFC 6960)       │ • Client asks CA: "Is this cert revoked?"       │
│                  │ • Real-time response                             │
│                  │ • Problem: privacy (CA sees which sites you      │
│                  │   visit), availability dependency                 │
├──────────────────┼──────────────────────────────────────────────────┤
│ OCSP Stapling    │ Server fetches OCSP response, attaches it to    │
│ (RFC 6066)       │ TLS handshake                                   │
│                  │ • Client verifies without contacting CA          │
│                  │ • Best practice! Configure on your server        │
│                  │ • Response is signed by CA, time-limited         │
├──────────────────┼──────────────────────────────────────────────────┤
│ CRLite           │ Compressed revocation data pushed to browser     │
│ (Firefox)        │ • Complete revocation info, no network needed   │
│                  │ • Very efficient (< 1 MB for all revocations)   │
├──────────────────┼──────────────────────────────────────────────────┤
│ Short-lived      │ Certificates valid for only days                 │
│ certificates     │ • No revocation needed (cert expires before     │
│                  │   revocation would take effect)                  │
│                  │ • Let's Encrypt certs: 90 days                  │
└──────────────────┴──────────────────────────────────────────────────┘
```

### 6.3 Certificate Transparency (CT)

```
┌─────────────────────────────────────────────────────────────────────┐
│              Certificate Transparency (RFC 6962)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Problem: A rogue or compromised CA could issue fraudulent          │
│  certificates for any domain.                                       │
│                                                                      │
│  Solution: ALL certificates must be publicly logged.                │
│                                                                      │
│  ┌────────┐  issue cert  ┌─────┐  submit  ┌───────┐               │
│  │ Domain │ ◀──────────── │ CA  │ ────────▶│  CT   │               │
│  │ Owner  │              │     │          │  Log  │               │
│  └────────┘              └─────┘          └───┬───┘               │
│                                               │                     │
│  ┌────────┐  SCT in TLS  ┌─────┐  check    │                     │
│  │Browser │ ◀──────────── │Servr│          │                     │
│  │        │──────────────▶│     │◀─────────┘                     │
│  │        │   verify SCT  └─────┘  return SCT                    │
│  └────┬───┘                                                       │
│       │                                                            │
│       ▼                                                            │
│  ┌────────────┐                                                    │
│  │  Monitor   │  Watches CT logs for unauthorized certificates    │
│  │  Service   │  Domain owners can detect rogue certs             │
│  └────────────┘                                                    │
│                                                                      │
│  SCT = Signed Certificate Timestamp (proof of logging)             │
│  Chrome/Firefox REQUIRE CT for all publicly trusted certificates.  │
│                                                                      │
│  Search CT logs: https://crt.sh/                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. Let's Encrypt and ACME

### 7.1 ACME Protocol Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│         ACME Protocol (Automatic Certificate Management)             │
│                       RFC 8555                                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Client (certbot)                          Let's Encrypt            │
│   ────────────────                          ──────────────            │
│                                                                      │
│   1. Create Account                                                 │
│      POST /acme/new-acct ──────────────────▶                        │
│                           ◀────────────────── Account URL            │
│                                                                      │
│   2. Submit Order                                                   │
│      POST /acme/new-order ─────────────────▶                        │
│      { identifiers: ["example.com"] }                               │
│                           ◀────────────────── Authorization URLs     │
│                                                                      │
│   3. Complete Challenges (prove domain ownership)                   │
│                                                                      │
│      HTTP-01 Challenge:                                             │
│      PUT /.well-known/acme-challenge/{token}                        │
│      → Let's Encrypt GETs this URL to verify                       │
│                                                                      │
│      DNS-01 Challenge:                                              │
│      Create TXT record: _acme-challenge.example.com                 │
│      → Let's Encrypt queries DNS to verify                          │
│                                                                      │
│   4. Finalize Order (submit CSR)                                    │
│      POST /acme/finalize ──────────────────▶                        │
│      { csr: "MIIBkTCB..." }                                        │
│                           ◀────────────────── Certificate URL        │
│                                                                      │
│   5. Download Certificate                                           │
│      GET /acme/cert/abc123 ─────────────────▶                       │
│                           ◀────────────────── PEM certificate chain  │
│                                                                      │
│   6. Auto-renewal (before expiration, typically at 60 days)         │
│      Repeat steps 2-5 automatically                                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 7.2 Certbot Commands

```bash
# Install certbot
# Ubuntu/Debian
sudo apt install certbot python3-certbot-nginx

# macOS
brew install certbot

# Obtain a certificate (standalone mode - stops web server temporarily)
sudo certbot certonly --standalone -d example.com -d www.example.com

# Obtain with Nginx plugin (no downtime)
sudo certbot --nginx -d example.com -d www.example.com

# Obtain with DNS challenge (for wildcards)
sudo certbot certonly --manual --preferred-challenges dns -d "*.example.com"

# Test renewal
sudo certbot renew --dry-run

# Actual renewal (usually via cron/systemd timer)
sudo certbot renew

# View certificates
sudo certbot certificates

# Revoke a certificate
sudo certbot revoke --cert-path /etc/letsencrypt/live/example.com/cert.pem
```

### 7.3 Certificate Files

```
/etc/letsencrypt/live/example.com/
├── cert.pem       # Your domain's certificate (leaf only)
├── chain.pem      # Intermediate CA certificate(s)
├── fullchain.pem  # cert.pem + chain.pem (what most servers need)
├── privkey.pem    # Your private key (KEEP SECRET!)
└── README

# Nginx configuration
server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;

    # Modern TLS configuration
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

## 8. Certificate Pinning

Certificate pinning restricts which certificates are accepted for a given domain, preventing attacks even if a CA is compromised.

```
┌──────────────────────────────────────────────────────────────────────┐
│              Certificate Pinning Strategies                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. PUBLIC KEY PINNING (recommended)                                │
│     Pin the hash of the public key (not the whole cert).           │
│     Survives certificate renewal (key stays the same).             │
│                                                                      │
│  2. CERTIFICATE PINNING                                             │
│     Pin the hash of the entire certificate.                        │
│     Must update pins when certificate is renewed.                  │
│                                                                      │
│  3. INTERMEDIATE CA PINNING                                         │
│     Pin the intermediate CA's public key.                          │
│     More flexible than leaf pinning.                               │
│                                                                      │
│  IMPORTANT WARNINGS:                                                │
│  • HPKP (HTTP Public Key Pinning) is DEPRECATED in browsers       │
│    (too easy to brick a domain permanently)                         │
│  • Use pinning primarily in mobile apps and API clients            │
│  • ALWAYS include a backup pin                                     │
│  • Certificate Transparency is now preferred over pinning          │
│    for web browsers                                                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 8.1 Certificate Pinning in Python

```python
import ssl
import socket
import hashlib
from cryptography import x509
from cryptography.hazmat.primitives.serialization import (
    Encoding, PublicFormat
)

class PinnedHTTPSConnection:
    """HTTPS connection with certificate pinning."""

    def __init__(self, hostname: str, port: int = 443,
                 pinned_hashes: list = None):
        """
        Args:
            hostname: Server hostname
            port: Server port
            pinned_hashes: List of SHA-256 hashes of accepted
                           public keys (base64-encoded)
        """
        self.hostname = hostname
        self.port = port
        self.pinned_hashes = pinned_hashes or []

    def _get_pubkey_hash(self, cert_der: bytes) -> str:
        """Compute SHA-256 hash of certificate's public key."""
        cert = x509.load_der_x509_certificate(cert_der)
        pub_key_bytes = cert.public_key().public_bytes(
            Encoding.DER,
            PublicFormat.SubjectPublicKeyInfo,
        )
        return hashlib.sha256(pub_key_bytes).hexdigest()

    def connect(self) -> dict:
        """Connect with certificate pin verification."""
        context = ssl.create_default_context()

        with socket.create_connection(
            (self.hostname, self.port), timeout=10
        ) as sock:
            with context.wrap_socket(sock, server_hostname=self.hostname) as ssock:
                # Get the certificate chain
                cert_der = ssock.getpeercert(binary_form=True)
                cert_info = ssock.getpeercert()

                # Compute public key hash
                actual_hash = self._get_pubkey_hash(cert_der)

                # Check pin
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

# First, discover the pin for a server
try:
    discovery = PinnedHTTPSConnection("www.google.com")
    result = discovery.connect()
    print("Certificate Pin Discovery:")
    print(f"  Host:         {result['hostname']}")
    print(f"  Protocol:     {result['protocol']}")
    print(f"  Cipher:       {result['cipher'][0]}")
    print(f"  Subject:      {result['cert_subject']}")
    print(f"  Pubkey Hash:  {result['pubkey_hash']}")

    # Now connect with the pin
    pinned = PinnedHTTPSConnection(
        "www.google.com",
        pinned_hashes=[result["pubkey_hash"]]
    )
    pinned_result = pinned.connect()
    print(f"\n  Pin valid:    {pinned_result['pin_valid']}")

    # Wrong pin
    wrong = PinnedHTTPSConnection(
        "www.google.com",
        pinned_hashes=["0000000000000000000000000000000000000000000000000000000000000000"]
    )
    wrong_result = wrong.connect()
    print(f"  Wrong pin:    {wrong_result['pin_valid']}")

except Exception as e:
    print(f"Network error (expected in offline environments): {e}")
```

---

## 9. Mutual TLS (mTLS)

In standard TLS, only the server presents a certificate. In mutual TLS, both the client and server present certificates and verify each other. This is used for service-to-service communication, API authentication, and zero-trust architectures.

```
┌──────────────────────────────────────────────────────────────────────┐
│              Standard TLS vs Mutual TLS                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  STANDARD TLS (one-way):                                            │
│                                                                      │
│  Client ──── "Who are you?" ──── Server                             │
│         ◀─── [Server Certificate] ───                               │
│         ──── "OK, I trust you" ────▶                                │
│                                                                      │
│  Only the SERVER is authenticated.                                  │
│  Client identity is verified by other means (JWT, API key, etc.)   │
│                                                                      │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                      │
│  MUTUAL TLS (two-way):                                              │
│                                                                      │
│  Client ──── "Who are you?" ──── Server                             │
│         ◀─── [Server Certificate] ───                               │
│         ◀─── "Who are YOU?" ─────────                               │
│         ──── [Client Certificate] ────▶                             │
│         ◀─── "OK, I trust you" ──────                               │
│         ──── "OK, I trust you too" ──▶                              │
│                                                                      │
│  BOTH client and server are authenticated.                          │
│  No passwords, tokens, or API keys needed.                         │
│                                                                      │
│  Use cases:                                                         │
│  • Microservice-to-microservice (Istio, Linkerd service mesh)      │
│  • API authentication for partners/machines                        │
│  • IoT device authentication                                       │
│  • Zero-trust networks                                             │
│  • Database connections                                             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 9.1 mTLS Implementation

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
    """Create CA, server, and client certificates for mTLS."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)

    # 1. Create CA
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

    # 2. Create Server Certificate
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

    # 3. Create Client Certificate
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

    # Write all certificates and keys
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

# Create certificates
certs = create_mtls_certificates("/tmp/mtls_demo")
print("Generated certificates:")
for name, path in certs.items():
    print(f"  {name}: {path}")

def create_mtls_server_context(certs: dict) -> ssl.SSLContext:
    """Create an SSL context for a mTLS server."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Load server certificate and key
    ctx.load_cert_chain(
        certfile=certs["server_cert"],
        keyfile=certs["server_key"],
    )

    # Require client certificate (this is what makes it mTLS)
    ctx.verify_mode = ssl.CERT_REQUIRED

    # Trust only our CA for client certificates
    ctx.load_verify_locations(cafile=certs["ca_cert"])

    return ctx

def create_mtls_client_context(certs: dict) -> ssl.SSLContext:
    """Create an SSL context for a mTLS client."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Load client certificate and key
    ctx.load_cert_chain(
        certfile=certs["client_cert"],
        keyfile=certs["client_key"],
    )

    # Trust our CA for server certificate verification
    ctx.load_verify_locations(cafile=certs["ca_cert"])

    return ctx

# Simple mTLS echo server
def mtls_server(certs: dict, port: int):
    """Run a simple mTLS server."""
    ctx = create_mtls_server_context(certs)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("localhost", port))
        sock.listen(1)

        with ctx.wrap_socket(sock, server_side=True) as ssock:
            conn, addr = ssock.accept()
            with conn:
                # Get client certificate info
                client_cert = conn.getpeercert()
                client_cn = dict(x[0] for x in client_cert["subject"])["commonName"]
                print(f"  [Server] Client authenticated: {client_cn}")

                # Echo received data
                data = conn.recv(1024)
                conn.sendall(f"Hello {client_cn}! You said: {data.decode()}".encode())

# Test mTLS
port = 8443
server_thread = threading.Thread(target=mtls_server, args=(certs, port))
server_thread.daemon = True
server_thread.start()

import time
time.sleep(0.5)  # Wait for server to start

# Connect as client
client_ctx = create_mtls_client_context(certs)
with socket.create_connection(("localhost", port)) as sock:
    with client_ctx.wrap_socket(sock, server_hostname="localhost") as ssock:
        print(f"  [Client] Connected with {ssock.version()}")
        print(f"  [Client] Server cert: {dict(x[0] for x in ssock.getpeercert()['subject'])}")

        ssock.sendall(b"Hello from mTLS client!")
        response = ssock.recv(1024)
        print(f"  [Client] Response: {response.decode()}")

# Clean up
import shutil
shutil.rmtree("/tmp/mtls_demo")
```

---

## 10. Python TLS Programming

### 10.1 Secure HTTPS Client

```python
import ssl
import urllib.request

def make_secure_request(url: str) -> dict:
    """Make an HTTPS request with security best practices."""
    # Create a secure SSL context
    ctx = ssl.create_default_context()

    # Enforce minimum TLS version
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Disable compression (prevents CRIME attack)
    ctx.options |= ssl.OP_NO_COMPRESSION

    # Set strong cipher suites (for TLS 1.2 fallback)
    ctx.set_ciphers(
        "ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20"
    )

    # Make request
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
        return {
            "status": response.status,
            "headers": dict(response.headers),
            "url": response.url,
        }

# Example
try:
    result = make_secure_request("https://www.example.com")
    print(f"Status: {result['status']}")
    print(f"URL: {result['url']}")
except Exception as e:
    print(f"Error: {e}")
```

### 10.2 TLS Server with Python

```python
import ssl
import socket
import threading

def create_tls_server(certfile: str, keyfile: str,
                       host: str = "localhost", port: int = 8443):
    """Create a TLS server with modern configuration."""
    # Create SSL context
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2

    # Load certificate and key
    ctx.load_cert_chain(certfile=certfile, keyfile=keyfile)

    # Strong ciphers
    ctx.set_ciphers(
        "ECDHE+AESGCM:ECDHE+CHACHA20"
    )

    # Security options
    ctx.options |= ssl.OP_NO_COMPRESSION    # Prevent CRIME
    ctx.options |= ssl.OP_SINGLE_DH_USE     # Fresh DH params
    ctx.options |= ssl.OP_SINGLE_ECDH_USE   # Fresh ECDH params

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.listen(5)
        print(f"TLS server listening on {host}:{port}")

        with ctx.wrap_socket(sock, server_side=True) as ssock:
            while True:
                conn, addr = ssock.accept()
                print(f"Connection from {addr}")
                print(f"  Protocol: {conn.version()}")
                print(f"  Cipher:   {conn.cipher()}")

                # Handle connection
                data = conn.recv(1024)
                if data:
                    conn.sendall(b"HTTP/1.1 200 OK\r\n"
                                b"Content-Type: text/plain\r\n\r\n"
                                b"Hello, TLS!\n")
                conn.close()
```

---

## 11. Creating Certificates with OpenSSL

### 11.1 Self-Signed Certificate (Development)

```bash
# Generate a self-signed certificate for development
# (DO NOT use in production!)

# One-liner: generate key + cert in one command
openssl req -x509 -newkey rsa:4096 -sha256 -days 365 \
    -nodes -keyout server.key -out server.crt \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

# Verify the certificate
openssl x509 -in server.crt -text -noout

# Check the key
openssl rsa -in server.key -check -noout
```

### 11.2 CA + Server Certificate (Full Process)

```bash
# Step 1: Create CA private key
openssl genrsa -aes256 -out ca.key 4096
# Enter passphrase for CA key (remember this!)

# Step 2: Create CA certificate (self-signed)
openssl req -new -x509 -sha256 -days 3650 -key ca.key -out ca.crt \
    -subj "/C=US/O=My Organization/CN=My Root CA"

# Step 3: Create server private key (no passphrase for automation)
openssl genrsa -out server.key 2048

# Step 4: Create Certificate Signing Request (CSR)
openssl req -new -sha256 -key server.key -out server.csr \
    -subj "/CN=myserver.example.com"

# Step 5: Create a config file for SAN (Subject Alternative Names)
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

# Step 6: Sign the CSR with the CA key
openssl x509 -req -sha256 -days 365 \
    -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out server.crt -extfile server_ext.cnf

# Step 7: Verify the certificate
openssl verify -CAfile ca.crt server.crt

# Step 8: View certificate details
openssl x509 -in server.crt -text -noout | head -30

# Step 9: Create fullchain (server cert + CA cert)
cat server.crt ca.crt > fullchain.crt
```

### 11.3 ECDSA Certificates (Smaller, Faster)

```bash
# Generate ECDSA key (P-256 curve)
openssl ecparam -genkey -name prime256v1 -out server_ec.key

# Or Ed25519 (if supported by your OpenSSL version)
openssl genpkey -algorithm Ed25519 -out server_ed25519.key

# Create CSR with ECDSA key
openssl req -new -sha256 -key server_ec.key -out server_ec.csr \
    -subj "/CN=myserver.example.com"

# Self-signed ECDSA certificate
openssl req -x509 -sha256 -days 365 -key server_ec.key -out server_ec.crt \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

# Compare sizes
echo "RSA-2048 cert size:"
wc -c server.crt
echo "ECDSA P-256 cert size:"
wc -c server_ec.crt
```

### 11.4 Client Certificate for mTLS

```bash
# Generate client key
openssl genrsa -out client.key 2048

# Create client CSR
openssl req -new -sha256 -key client.key -out client.csr \
    -subj "/CN=api-client-1/O=My App"

# Sign with CA (note: extendedKeyUsage = clientAuth)
cat > client_ext.cnf << 'EOF'
basicConstraints=CA:FALSE
keyUsage=digitalSignature
extendedKeyUsage=clientAuth
EOF

openssl x509 -req -sha256 -days 365 \
    -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out client.crt -extfile client_ext.cnf

# Create PKCS12 bundle (for importing into browsers/apps)
openssl pkcs12 -export -out client.p12 \
    -inkey client.key -in client.crt -certfile ca.crt

# Verify client cert
openssl verify -CAfile ca.crt client.crt
```

---

## 12. Common Misconfigurations

```
┌──────────────────────────────────────────────────────────────────────┐
│         Common TLS Misconfigurations and How to Fix Them             │
├───┬──────────────────────────────┬──────────────────────────────────┤
│ # │ Misconfiguration             │ Fix                              │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 1 │ Legacy TLS versions enabled  │ Disable SSL 2/3, TLS 1.0/1.1   │
│   │ (TLS 1.0, SSL 3.0)          │ min_version = TLSv1.2           │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 2 │ Weak cipher suites           │ Use only AEAD ciphers with ECDHE│
│   │ (RC4, 3DES, CBC without      │ Disable all non-FS suites       │
│   │  AEAD, static RSA)          │                                  │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 3 │ Self-signed cert in prod     │ Use Let's Encrypt (free!)       │
│   │                              │ Self-signed only for dev/testing│
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 4 │ Expired certificate          │ Auto-renew with certbot         │
│   │                              │ Monitor with external tools     │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 5 │ Missing intermediate cert    │ Always send full chain          │
│   │ (incomplete chain)           │ (fullchain.pem, not cert.pem)   │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 6 │ No HSTS header               │ Add Strict-Transport-Security   │
│   │                              │ with long max-age               │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 7 │ HTTP to HTTPS redirect       │ Use 301 permanent redirect      │
│   │ missing                      │ Register for HSTS preload list  │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 8 │ Private key too permissive   │ chmod 600 for key files         │
│   │ (world-readable)             │ Own by root or www-data only    │
├───┼──────────────────────────────┼──────────────────────────────────┤
│ 9 │ TLS compression enabled      │ Disable (prevents CRIME attack) │
│   │                              │ ssl_options |= OP_NO_COMPRESSION│
├───┼──────────────────────────────┼──────────────────────────────────┤
│10 │ No OCSP stapling             │ Enable OCSP stapling in server  │
│   │                              │ config (improves performance)   │
├───┼──────────────────────────────┼──────────────────────────────────┤
│11 │ Wildcard cert on public site │ Use SAN with specific domains   │
│   │                              │ Wildcards only for internal use │
├───┼──────────────────────────────┼──────────────────────────────────┤
│12 │ Certificate pinning without  │ ALWAYS include backup pins      │
│   │ backup pins                  │ Or use CT instead of pinning    │
└───┴──────────────────────────────┴──────────────────────────────────┘
```

### 12.1 Security Headers Checklist

```python
# Recommended security headers for HTTPS sites

SECURITY_HEADERS = {
    # Force HTTPS for 2 years, include subdomains
    "Strict-Transport-Security": "max-age=63072000; includeSubDomains; preload",

    # Prevent MIME type sniffing
    "X-Content-Type-Options": "nosniff",

    # Prevent clickjacking
    "X-Frame-Options": "DENY",

    # Enable XSS filter (legacy browsers)
    "X-XSS-Protection": "0",  # "0" is now recommended (CSP is better)

    # Content Security Policy
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'",

    # Referrer Policy
    "Referrer-Policy": "strict-origin-when-cross-origin",

    # Permissions Policy (formerly Feature-Policy)
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

# Flask example
from flask import Flask, make_response

app = Flask(__name__)

@app.after_request
def add_security_headers(response):
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    return response
```

---

## 13. Testing TLS Configurations

### 13.1 Command-Line Tools

```bash
# Test TLS connection with OpenSSL s_client
openssl s_client -connect example.com:443 -servername example.com

# Show full certificate chain
openssl s_client -connect example.com:443 -servername example.com -showcerts

# Test specific TLS version
openssl s_client -connect example.com:443 -tls1_3
openssl s_client -connect example.com:443 -tls1_2

# Check certificate expiration
echo | openssl s_client -connect example.com:443 -servername example.com 2>/dev/null \
    | openssl x509 -noout -dates

# Check supported cipher suites
nmap --script ssl-enum-ciphers -p 443 example.com

# Test with testssl.sh (comprehensive scanner)
# git clone https://github.com/drwetter/testssl.sh.git
./testssl.sh example.com

# Test with sslyze
pip install sslyze
sslyze example.com
```

### 13.2 Python TLS Scanner

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
    """Scan a server's TLS configuration."""
    result = TLSScanResult(hostname=hostname, port=port)

    # Test supported protocols
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
                        result.issues.append("TLS 1.0 supported (deprecated)")
                    elif name == "TLSv1.1":
                        result.issues.append("TLS 1.1 supported (deprecated)")
        except (ssl.SSLError, ConnectionRefusedError, OSError):
            pass

    # Get certificate and cipher info with best available protocol
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, port), timeout=5) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                # Protocol and cipher
                result.cipher_suite = ssock.cipher()[0]

                # Certificate info
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

                # Check expiration
                from datetime import datetime
                not_after = ssl.cert_time_to_seconds(cert["notAfter"])
                import time
                days_left = (not_after - time.time()) / 86400
                if days_left < 0:
                    result.issues.append(f"Certificate EXPIRED {abs(days_left):.0f} days ago!")
                elif days_left < 30:
                    result.issues.append(f"Certificate expires in {days_left:.0f} days!")

    except Exception as e:
        result.issues.append(f"Connection error: {e}")

    # Generate recommendations
    if "TLSv1.3" not in result.supported_protocols:
        result.recommendations.append("Enable TLS 1.3 for better security and performance")
    if "TLSv1.0" in result.supported_protocols:
        result.recommendations.append("Disable TLS 1.0 (deprecated since 2020)")
    if "TLSv1.1" in result.supported_protocols:
        result.recommendations.append("Disable TLS 1.1 (deprecated since 2020)")
    if not result.issues:
        result.recommendations.append("Configuration looks good!")

    return result

def print_scan_result(result: TLSScanResult):
    """Pretty-print a TLS scan result."""
    print(f"\nTLS Scan: {result.hostname}:{result.port}")
    print("=" * 60)

    print(f"\nSupported Protocols:")
    for proto in result.supported_protocols:
        status = "[!!]" if "1.0" in proto or "1.1" in proto else "[OK]"
        print(f"  {status} {proto}")

    if result.cipher_suite:
        print(f"\nNegotiated Cipher: {result.cipher_suite}")

    if result.certificate_info:
        ci = result.certificate_info
        print(f"\nCertificate:")
        print(f"  Subject: {ci.get('subject', {}).get('commonName', 'N/A')}")
        print(f"  Issuer:  {ci.get('issuer', {}).get('commonName', 'N/A')}")
        print(f"  Valid:   {ci.get('notBefore', 'N/A')} to {ci.get('notAfter', 'N/A')}")
        if ci.get("san"):
            print(f"  SANs:    {', '.join(ci['san'][:5])}")

    if result.issues:
        print(f"\nIssues ({len(result.issues)}):")
        for issue in result.issues:
            print(f"  [!!] {issue}")

    if result.recommendations:
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  --> {rec}")

# Run scan
try:
    result = scan_tls("www.google.com")
    print_scan_result(result)
except Exception as e:
    print(f"Scan failed (network required): {e}")
```

### 13.3 Continuous Certificate Monitoring

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
    warning_days: int = 30   # Warn if less than this many days left
    critical_days: int = 7   # Critical if less than this many days

class CertificateMonitor:
    """Monitor certificate expiration across multiple domains."""

    def __init__(self, entries: List[CertMonitorEntry]):
        self.entries = entries

    def check_certificate(self, entry: CertMonitorEntry) -> dict:
        """Check a single certificate's expiration."""
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
        """Check all monitored certificates."""
        results = []
        for entry in self.entries:
            result = self.check_certificate(entry)
            results.append(result)
        return results

    def print_report(self, results: list):
        """Print a formatted monitoring report."""
        print(f"\nCertificate Monitoring Report")
        print(f"Generated: {datetime.now(timezone.utc).isoformat()}")
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
                      f"{r['days_remaining']:6.0f} days  "
                      f"({r['status']})")
            else:
                print(f"  {icon} {r['hostname']:30s} "
                      f"{'N/A':>6s}       "
                      f"({r.get('error', r['status'])})")

# Usage
monitor = CertificateMonitor([
    CertMonitorEntry("www.google.com"),
    CertMonitorEntry("github.com"),
    CertMonitorEntry("expired.badssl.com"),  # Known expired cert for testing
])

try:
    results = monitor.check_all()
    monitor.print_report(results)
except Exception as e:
    print(f"Monitoring requires network access: {e}")
```

---

## 14. Exercises

### Exercise 1: TLS Handshake Trace (Beginner)

Using `openssl s_client`, connect to three different websites and answer:
1. What TLS version was negotiated?
2. What cipher suite was selected?
3. How many certificates are in the chain?
4. What is the root CA?
5. When does the leaf certificate expire?

```bash
# Template command:
echo | openssl s_client -connect <host>:443 -servername <host> 2>/dev/null
```

Compare the results for: google.com, github.com, and letsencrypt.org.

### Exercise 2: Certificate Generation Lab (Intermediate)

Using Python's `cryptography` library:
1. Create a Root CA with a 4096-bit RSA key (valid for 10 years)
2. Create an Intermediate CA signed by the Root (valid for 5 years)
3. Create a server certificate signed by the Intermediate (valid for 90 days)
4. Write all certificates and keys to PEM files
5. Verify the complete chain programmatically
6. Test the chain with `openssl verify`

### Exercise 3: mTLS Service (Intermediate)

Build a simple HTTP API protected by mTLS:
1. Create CA, server, and two client certificates
2. Write a Flask/HTTP server that requires client certificates
3. Extract the client's Common Name from the certificate
4. Implement authorization based on the client CN
5. Write a client that presents its certificate
6. Show that connections without a valid client cert are rejected

### Exercise 4: TLS Configuration Auditor (Advanced)

Build a comprehensive TLS scanner that checks:
1. Supported TLS versions (flag TLS 1.0/1.1 as deprecated)
2. Supported cipher suites (flag weak ones)
3. Certificate chain completeness
4. Certificate expiration date
5. HSTS header presence and max-age
6. OCSP stapling status
7. Key size (flag RSA < 2048 or ECDSA < 256)
8. Forward secrecy support
9. Output a score (A+ to F) similar to SSL Labs

### Exercise 5: Certificate Monitoring System (Advanced)

Build a production-quality certificate monitoring system:
1. Accept a list of domains from a YAML/JSON config file
2. Check certificate expiration on a schedule (cron-compatible)
3. Send alerts (email/Slack/webhook) when certificates are:
   - Expiring within 30 days (warning)
   - Expiring within 7 days (critical)
   - Already expired (emergency)
4. Track certificate changes (issuer changed, key changed)
5. Store history in SQLite for trend analysis
6. Generate an HTML report

### Exercise 6: Implement Simplified TLS (Educational)

Build a simplified version of the TLS handshake (educational, not for production):
1. Client sends: supported cipher suites, random nonce, X25519 public key
2. Server responds: selected cipher, random nonce, X25519 public key, certificate
3. Both derive shared secret using X25519
4. Both derive encryption keys using HKDF
5. Server sends: signature over the handshake transcript
6. Client verifies the signature using the certificate's public key
7. Both exchange encrypted messages using AES-GCM

This exercise demonstrates the core mechanics of TLS 1.3 without the full complexity of the real protocol.

---

## References

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
- BadSSL.com - https://badssl.com/ (test various TLS misconfigurations)

---

**Previous**: [03. Hashing and Data Integrity](./03_Hashing_and_Integrity.md) | **Next**: [05. Authentication Systems](./05_Authentication.md)
