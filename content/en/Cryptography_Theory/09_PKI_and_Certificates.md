# Lesson 9: PKI and Certificates

**Previous**: [Key Exchange](./08_Key_Exchange.md) | **Next**: [Lattice-Based Cryptography](./10_Lattice_Based_Cryptography.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why key exchange alone is insufficient without a trust model
2. Describe the X.509 certificate structure and parse its fields programmatically
3. Trace a certificate chain from end-entity through intermediate CAs to a root CA
4. Compare hierarchical PKI (X.509) with the Web of Trust (PGP) model
5. Evaluate certificate revocation mechanisms (CRL, OCSP, CT logs) and their trade-offs
6. Explain how Let's Encrypt and the ACME protocol democratized HTTPS
7. Analyze certificate pinning and DANE/TLSA as supplementary trust mechanisms

---

In Lesson 8, we learned how Diffie-Hellman enables two parties to agree on a shared secret over an insecure channel. But DH alone cannot answer the most fundamental question in any communication: **"Am I actually talking to who I think I'm talking to?"** Without authentication, an attacker can impersonate either party (the man-in-the-middle attack). Public Key Infrastructure (PKI) provides the answer by creating a chain of trust from a trusted authority down to every server, device, and user on the internet. Understanding PKI is essential because every HTTPS connection, every code-signing certificate, and every encrypted email relies on it.

## Table of Contents

1. [The Trust Problem](#1-the-trust-problem)
2. [X.509 Certificate Structure](#2-x509-certificate-structure)
3. [Certificate Chains and Trust Hierarchies](#3-certificate-chains-and-trust-hierarchies)
4. [Certificate Revocation](#4-certificate-revocation)
5. [Certificate Transparency](#5-certificate-transparency)
6. [Web of Trust vs. Hierarchical PKI](#6-web-of-trust-vs-hierarchical-pki)
7. [Let's Encrypt and ACME](#7-lets-encrypt-and-acme)
8. [Certificate Pinning and DANE](#8-certificate-pinning-and-dane)
9. [Summary](#9-summary)
10. [Exercises](#10-exercises)

---

## 1. The Trust Problem

### 1.1 The Bootstrap Dilemma

Suppose Alice wants to communicate securely with `bank.com`. She can use ECDHE (Lesson 8) to establish a shared key, but she must first verify that the public key she received actually belongs to `bank.com` and not to an attacker.

The core question is: **How do you trust a public key you received over the internet?**

> **Analogy:** PKI is like a passport system. Your passport (certificate) is issued by your government (Certificate Authority). When you present it at a foreign border (connect to a server), the foreign country (your browser) trusts it because they trust the issuing government. They verify the passport's security features (digital signature), check that it hasn't been revoked, and confirm your photo matches your face (domain name matches the certificate).

### 1.2 Historical Solutions

| Approach | How It Works | Limitation |
|----------|-------------|------------|
| Physical meeting | Exchange keys in person | Doesn't scale |
| Trusted directory | Central server publishes keys | Single point of failure/trust |
| Web of Trust | Users vouch for each other | No guarantee of competence; doesn't scale |
| **Hierarchical PKI** | Trusted root CAs certify identities | Requires trusting a set of root CAs |

The internet settled on **hierarchical PKI** with X.509 certificates as the dominant model.

---

## 2. X.509 Certificate Structure

### 2.1 What Is an X.509 Certificate?

An X.509 certificate is a signed data structure that binds a public key to an identity (such as a domain name). The Certificate Authority (CA) vouches for this binding by signing the certificate with its own private key.

### 2.2 Certificate Fields (v3)

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

### 2.3 Parsing Certificates in Python

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

### 2.4 Self-Signed vs. CA-Signed

| Property | Self-Signed | CA-Signed |
|----------|-------------|-----------|
| Issuer == Subject? | Yes | No |
| Trusted by browsers? | No (must add manually) | Yes (if CA is in trust store) |
| Cost | Free | Free (Let's Encrypt) to expensive (EV) |
| Use case | Development, internal | Public-facing services |

---

## 3. Certificate Chains and Trust Hierarchies

### 3.1 The Chain of Trust

Browsers and operating systems ship with a **trust store** — a set of ~100-150 root CA certificates that are implicitly trusted. These root CAs rarely sign end-entity certificates directly. Instead, the hierarchy looks like:

```
Root CA (in trust store, 20+ year validity)
  └── Intermediate CA (signed by Root, 5-10 year validity)
        └── End-Entity Certificate (signed by Intermediate, 90 days - 1 year)
```

### 3.2 Why Intermediate CAs?

- **Security**: Root CA private keys are kept offline in HSMs (Hardware Security Modules). Only intermediate keys are used for day-to-day signing.
- **Revocation**: If an intermediate CA is compromised, only it is revoked — the root and other intermediates remain trusted.
- **Flexibility**: Different intermediates can serve different purposes (DV, OV, EV certificates).

### 3.3 Certificate Validation Algorithm

When a browser receives a certificate chain, it performs:

1. **Build the chain**: Starting from the end-entity cert, follow the Issuer field to find the next cert, up to a root CA in the trust store.
2. **Verify signatures**: Each certificate's signature must be valid under the issuer's public key.
3. **Check validity dates**: Every certificate in the chain must be within its validity period.
4. **Check revocation**: Verify that no certificate in the chain has been revoked (via CRL or OCSP).
5. **Check constraints**: Intermediate CAs must have `CA:TRUE` in Basic Constraints. Path length constraints are respected.
6. **Check name**: The end-entity certificate's SAN must include the requested domain.

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

### 3.4 Certificate Types by Validation Level

| Type | Validation | Indicator | Cost | Time |
|------|-----------|-----------|------|------|
| **DV** (Domain Validated) | Prove domain control | Lock icon | Free-$50 | Minutes |
| **OV** (Organization Validated) | Verify legal entity | Organization in cert | $100-$500 | Days |
| **EV** (Extended Validation) | Extensive vetting | Green bar (deprecated) | $500-$2000 | Weeks |

> **Note**: Major browsers no longer display EV differently from DV certificates. Research showed users did not notice or understand the green bar, making EV's added cost difficult to justify for most organizations.

---

## 4. Certificate Revocation

### 4.1 Why Revoke Certificates?

If a private key is compromised, the certificate must be revoked before its natural expiration. Common reasons:
- Key compromise (server hacked)
- CA compromise (DigiNotar incident, 2011)
- Domain ownership change
- Certificate issued in error

### 4.2 Certificate Revocation Lists (CRLs)

A CRL is a signed list of revoked certificate serial numbers published by the CA.

**Problems with CRLs**:
- **Size**: A popular CA's CRL can be megabytes, growing over time
- **Latency**: Clients must download the full CRL periodically
- **Freshness**: CRLs have a "next update" field; between updates, a revoked cert may still be accepted

### 4.3 Online Certificate Status Protocol (OCSP)

OCSP allows real-time, per-certificate revocation checking:

```
Client → OCSP Responder: "Is cert with serial 12345 revoked?"
OCSP Responder → Client: "Good" / "Revoked" / "Unknown" (signed response)
```

**Improvements over CRL**:
- Only checks the specific certificate (no bulk download)
- More timely responses

**Problems with OCSP**:
- **Privacy**: The CA knows every site you visit (OCSP query reveals the domain)
- **Availability**: If the OCSP responder is down, what should the client do?
  - **Hard fail**: Reject the certificate (secure but brittle)
  - **Soft fail**: Accept the certificate (available but insecure — an attacker who can block OCSP can bypass revocation)

### 4.4 OCSP Stapling

The server itself fetches the OCSP response and "staples" it to the TLS handshake:

```
Server → Client: Certificate + Signed OCSP Response (fresh, from CA)
```

**Advantages**:
- No privacy leak (client never contacts the CA)
- No availability concern (OCSP response is cached by the server)
- The OCSP response is signed by the CA, so the server cannot forge it

### 4.5 Short-Lived Certificates

Let's Encrypt certificates are valid for only 90 days (and can be renewed automatically). This reduces the revocation window — if a key is compromised, the certificate expires soon anyway. Some proposals advocate for certificates valid only 24-48 hours, potentially eliminating the need for revocation entirely.

---

## 5. Certificate Transparency

### 5.1 The Problem CT Solves

In 2011, a CA called DigiNotar was compromised, and the attackers issued fraudulent certificates for `*.google.com`. Google discovered this only because the Chrome browser had hardcoded pins for Google certificates. Without that lucky check, the fraudulent certificates would have been undetectable.

**Certificate Transparency (CT)** ensures that every certificate is publicly logged so that domain owners can detect unauthorized issuance.

### 5.2 How CT Works

```
1. CA issues certificate
2. CA submits certificate to CT logs (append-only, publicly auditable)
3. CT log returns a Signed Certificate Timestamp (SCT)
4. Certificate is embedded with SCTs (or they're delivered via TLS/OCSP)
5. Browser checks that the certificate has valid SCTs from multiple logs
```

### 5.3 Key Properties of CT Logs

- **Append-only**: Certificates can be added but never removed (enforced via Merkle trees, similar to blockchain)
- **Publicly auditable**: Anyone can query the log and verify its consistency
- **Multiple logs**: Certificates must be logged in at least 2-3 independent logs

### 5.4 Monitoring

Domain owners can monitor CT logs for certificates issued to their domains:

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

## 6. Web of Trust vs. Hierarchical PKI

### 6.1 PGP's Web of Trust

PGP (Pretty Good Privacy) uses a decentralized trust model:

- Users sign each other's public keys (key signing)
- Trust is transitive: if Alice trusts Bob, and Bob signed Carol's key, Alice has some confidence in Carol's key
- No central authority; trust decisions are made by individuals

### 6.2 Comparison

| Aspect | Hierarchical PKI (X.509) | Web of Trust (PGP) |
|--------|--------------------------|---------------------|
| **Trust anchor** | Root CAs (centralized) | Individual users (decentralized) |
| **Scalability** | Excellent (browsers ship root certs) | Poor (requires manual key signing) |
| **Revocation** | CRL/OCSP infrastructure | Key revocation certificates (manual) |
| **Failure mode** | CA compromise affects millions | One user's mistake affects their contacts |
| **Use case** | Web (TLS), code signing | Email encryption, file signing |
| **User burden** | None (transparent to users) | High (must evaluate trust decisions) |

### 6.3 Real-World Trust Failures

**PKI failures:**
- DigiNotar (2011): CA compromised, fraudulent Google certificates issued
- Symantec (2017): Systematic issuance of unauthorized certificates; trust removed by all major browsers
- Kazakhstan (2020): Government attempted to install a state root CA to intercept all HTTPS traffic

**Web of Trust failures:**
- Key signing parties are impractical at scale
- Trust graph reveals social connections (metadata leak)
- Most users never verify fingerprints

### 6.4 Modern Alternatives

- **TOFU (Trust On First Use)**: Accept the key on first contact, alert on changes (SSH model)
- **Key Transparency**: Google's proposal combining CT-style logs with end-to-end encryption key management
- **Decentralized Identity (DIDs)**: W3C standard for self-sovereign identity using verifiable credentials

---

## 7. Let's Encrypt and ACME

### 7.1 The Pre-Let's Encrypt World

Before 2015, HTTPS certificates cost $50-$300/year, required manual validation, and expired after 1-3 years. As a result, only ~40% of web traffic was encrypted.

### 7.2 Let's Encrypt's Revolution

Let's Encrypt (launched 2015) provides:
- **Free** DV certificates
- **Automated** issuance and renewal via the ACME protocol
- **90-day validity** (encourages automation, limits compromise window)
- **Open**: All certificates logged in CT

By 2024, Let's Encrypt had issued over 4 billion certificates and encrypts ~60% of all web pages.

### 7.3 ACME Protocol (RFC 8555)

ACME (Automatic Certificate Management Environment) automates the entire certificate lifecycle:

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

## 8. Certificate Pinning and DANE

### 8.1 Certificate Pinning

Even with PKI, any of the ~100+ trusted root CAs could issue a certificate for any domain. Certificate **pinning** restricts which CAs (or which specific certificates/keys) are valid for a given domain.

**Types of pinning:**
- **Public Key Pinning (HPKP)**: HTTP header specifying which public keys are valid for a site. **Deprecated** in 2018 because misconfiguration could permanently lock users out.
- **Built-in pins**: Browsers hardcode pins for high-value domains (Google, Firefox updates)
- **Mobile app pinning**: Apps embed the expected server certificate/key and reject all others

### 8.2 DANE (DNS-Based Authentication of Named Entities)

DANE uses DNSSEC-signed DNS records to specify which certificates or CAs are valid for a domain:

```
_443._tcp.example.com. IN TLSA 3 1 1 <sha256-hash-of-public-key>
```

**TLSA record fields:**
- **Usage**: 0-3 (CA constraint, service certificate constraint, trust anchor assertion, domain-issued certificate)
- **Selector**: 0 (full cert) or 1 (public key only)
- **Matching type**: 0 (exact), 1 (SHA-256), 2 (SHA-512)

**Advantages of DANE:**
- Domain owner controls trust (not dependent on any CA)
- Works even if a CA is compromised

**Limitations:**
- Requires DNSSEC (not universally deployed)
- Browsers have been slow to adopt (DANE is more popular for email/SMTP)

### 8.3 Expect-CT Header

The `Expect-CT` HTTP header instructs browsers to require Certificate Transparency for a domain:

```
Expect-CT: max-age=86400, enforce, report-uri="https://example.com/ct-report"
```

This ensures that if a fraudulent certificate is issued without CT logging, the browser will reject it.

---

## 9. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Trust problem | Key exchange needs authentication; PKI provides it |
| X.509 certificates | Bind public keys to identities via CA signatures |
| Certificate chains | Root CA → Intermediate CA → End-entity (defense in depth) |
| CRL/OCSP | Revocation is the hardest problem in PKI; OCSP stapling is the best solution |
| Certificate Transparency | Public, append-only logs detect fraudulent certificates |
| Web of Trust | Decentralized alternative; doesn't scale for the web |
| Let's Encrypt/ACME | Free, automated certificates; transformed web security |
| Pinning/DANE | Supplementary mechanisms to limit CA trust |

---

## 10. Exercises

### Exercise 1: Certificate Inspection (Coding)

Write a Python script that:
1. Fetches the certificate chain from any HTTPS website (e.g., `github.com`)
2. Prints the subject, issuer, validity dates, and SAN for each certificate in the chain
3. Verifies each signature in the chain
4. Reports whether the chain leads to a trusted root CA

### Exercise 2: Self-Signed CA (Coding)

Using the `cryptography` library, create:
1. A root CA key pair and self-signed certificate
2. An intermediate CA key pair and certificate (signed by the root)
3. An end-entity certificate for `test.example.com` (signed by the intermediate)
4. Verify the complete chain programmatically

### Exercise 3: CT Log Analysis (Conceptual + Coding)

Query `crt.sh` for a domain you own (or a well-known domain like `github.com`):
1. How many certificates have been issued in the past year?
2. Which CAs issued them?
3. Are there any unexpected certificates (potential misissuance)?
4. Write a monitoring script that alerts if a new certificate appears from an unexpected CA.

### Exercise 4: Revocation Trade-offs (Conceptual)

Compare CRL, OCSP, OCSP stapling, and short-lived certificates across these dimensions:
- Client bandwidth cost
- Client latency impact
- Privacy implications
- Security against an attacker who can block revocation checks
- Server-side complexity

Create a comparison table and recommend the best approach for: (a) a high-traffic web service, (b) an IoT device, (c) a mobile app.

### Exercise 5: PKI Design Challenge (Challenging)

Design a PKI for a university with 50,000 users who need:
- Email encryption (S/MIME)
- VPN authentication
- Document signing
- Wi-Fi (802.1X) authentication

Your design should address:
- Certificate hierarchy (how many CAs? What trust levels?)
- Key storage (where are private keys kept? HSMs? Smart cards? Software?)
- Revocation strategy
- Certificate lifecycle (issuance, renewal, revocation)
- Disaster recovery (what if the root CA is compromised?)
