"""
Public Key Infrastructure and Certificates
===========================================
Simplified X.509-like certificate structure, certificate chain
verification, and self-signed certificate creation.
"""

from __future__ import annotations
import hashlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Import RSA primitives (with fallback)
try:
    from importlib import import_module
    _rsa = import_module("05_rsa")
    rsa_keygen = _rsa.rsa_keygen
    rsa_sign = _rsa.rsa_sign
    rsa_verify = _rsa.rsa_verify
except (ImportError, ModuleNotFoundError):
    # Minimal fallback
    def _mod_pow(b, e, m):
        r = 1; b %= m
        while e > 0:
            if e & 1: r = r * b % m
            e >>= 1; b = b * b % m
        return r

    def _ext_gcd(a, b):
        o, r = a, b; os, s = 1, 0
        while r:
            q = o // r; o, r = r, o - q * r; os, s = s, os - q * s
        return o, os

    def _mod_inv(a, m):
        g, x = _ext_gcd(a % m, m); return x % m

    def _miller_rabin(n, k=20):
        if n < 4: return n >= 2
        if n % 2 == 0: return False
        s, d = 0, n-1
        while d % 2 == 0: d //= 2; s += 1
        for _ in range(k):
            a = random.randrange(2, n-1)
            x = _mod_pow(a, d, n)
            if x in (1, n-1): continue
            for _ in range(s-1):
                x = _mod_pow(x, 2, n)
                if x == n-1: break
            else: return False
        return True

    def _gen_prime(bits=256):
        while True:
            c = random.getrandbits(bits) | (1 << (bits-1)) | 1
            if _miller_rabin(c): return c

    import math
    def rsa_keygen(bits=512):
        p, q = _gen_prime(bits//2), _gen_prime(bits//2)
        while p == q: q = _gen_prime(bits//2)
        n = p * q
        ln = (p-1)*(q-1) // math.gcd(p-1,q-1)
        e = 65537; d = _mod_inv(e, ln)
        return (n, e), (n, d)

    def rsa_sign(msg, priv):
        n, d = priv
        h = int.from_bytes(hashlib.sha256(msg).digest(), "big") % n
        return _mod_pow(h, d, n)

    def rsa_verify(msg, sig, pub):
        n, e = pub
        h = int.from_bytes(hashlib.sha256(msg).digest(), "big") % n
        return _mod_pow(sig, e, n) == h


# ---------------------------------------------------------------------------
# Certificate Structure
# ---------------------------------------------------------------------------

# Why: X.509 certificates bind a public key to an identity (domain name,
# organization). The binding is certified by a trusted third party (the CA)
# via a digital signature. Without PKI, public key cryptography is
# vulnerable to man-in-the-middle attacks — you'd have no way to know
# if a public key truly belongs to "google.com" or to an attacker.

@dataclass
class Certificate:
    """Simplified X.509-like certificate."""
    version: int = 3
    serial_number: int = 0
    subject: str = ""           # e.g., "CN=example.com"
    issuer: str = ""            # e.g., "CN=My CA"
    not_before: str = ""        # validity start
    not_after: str = ""         # validity end
    public_key: tuple[int, int] = (0, 0)  # (n, e)
    is_ca: bool = False         # can this cert sign other certs?
    # Why: path_length limits how deep the chain can go. A root CA with
    # path_length=1 can only sign intermediate CAs, not end-entity certs.
    # This constrains trust delegation and limits blast radius of compromise.
    path_length: int | None = None
    signature: int = 0          # signed by issuer's private key
    issuer_public_key: tuple[int, int] = (0, 0)  # for verification

    def to_bytes(self) -> bytes:
        """Serialize certificate fields for signing (excluding signature)."""
        # Why: We serialize all fields EXCEPT the signature itself.
        # The CA signs this serialized form; the verifier re-serializes
        # and checks the signature against the CA's public key.
        data = (
            f"v{self.version}|SN:{self.serial_number}|"
            f"S:{self.subject}|I:{self.issuer}|"
            f"NB:{self.not_before}|NA:{self.not_after}|"
            f"PK:{self.public_key[0]}:{self.public_key[1]}|"
            f"CA:{self.is_ca}|PL:{self.path_length}"
        )
        return data.encode()

    def __str__(self) -> str:
        status = "CA" if self.is_ca else "End-Entity"
        return (f"Certificate(subject='{self.subject}', "
                f"issuer='{self.issuer}', type={status})")


# ---------------------------------------------------------------------------
# Certificate Authority
# ---------------------------------------------------------------------------

# Why: A CA is an entity trusted to vouch for the binding between public
# keys and identities. The root CA's certificate is self-signed (it signs
# its own certificate). Trust in the root CA is established out-of-band
# (e.g., pre-installed in your OS/browser).
class CertificateAuthority:
    """Simplified Certificate Authority that can issue and sign certificates."""

    _serial_counter = 1

    def __init__(self, name: str, key_bits: int = 512,
                 parent: CertificateAuthority | None = None) -> None:
        self.name = name
        self.public_key, self.private_key = rsa_keygen(key_bits)
        self.parent = parent
        self.certificate: Certificate | None = None

    def create_self_signed_cert(self, validity_days: int = 3650) -> Certificate:
        """Create a self-signed root CA certificate."""
        now = datetime.now()
        cert = Certificate(
            serial_number=self._next_serial(),
            subject=f"CN={self.name}",
            issuer=f"CN={self.name}",  # self-signed: issuer == subject
            not_before=now.isoformat(),
            not_after=(now + timedelta(days=validity_days)).isoformat(),
            public_key=self.public_key,
            is_ca=True,
            path_length=2,
            issuer_public_key=self.public_key,
        )
        # Why: Self-signed means the CA signs its own certificate.
        # This is the trust anchor — it must be distributed securely.
        cert.signature = rsa_sign(cert.to_bytes(), self.private_key)
        self.certificate = cert
        return cert

    def issue_certificate(
        self,
        subject: str,
        subject_public_key: tuple[int, int],
        is_ca: bool = False,
        path_length: int | None = None,
        validity_days: int = 365,
    ) -> Certificate:
        """Issue a certificate signed by this CA."""
        now = datetime.now()
        cert = Certificate(
            serial_number=self._next_serial(),
            subject=f"CN={subject}",
            issuer=f"CN={self.name}",
            not_before=now.isoformat(),
            not_after=(now + timedelta(days=validity_days)).isoformat(),
            public_key=subject_public_key,
            is_ca=is_ca,
            path_length=path_length,
            issuer_public_key=self.public_key,
        )
        cert.signature = rsa_sign(cert.to_bytes(), self.private_key)
        return cert

    @classmethod
    def _next_serial(cls) -> int:
        sn = cls._serial_counter
        cls._serial_counter += 1
        return sn


# ---------------------------------------------------------------------------
# Certificate Chain Verification
# ---------------------------------------------------------------------------

# Why: In practice, your browser doesn't trust every website's certificate
# directly. Instead, it trusts a small set of root CAs. The website
# presents a chain: [site cert] -> [intermediate CA] -> [root CA].
# Verification walks the chain from leaf to root, checking each signature.
def verify_certificate(cert: Certificate) -> bool:
    """Verify a single certificate's signature against its issuer's public key."""
    return rsa_verify(cert.to_bytes(), cert.signature, cert.issuer_public_key)


def verify_chain(chain: list[Certificate]) -> tuple[bool, str]:
    """Verify a certificate chain from leaf (index 0) to root (last index).

    Returns (is_valid, reason).
    """
    if not chain:
        return False, "Empty chain"

    # Why: We verify from leaf to root. Each cert in the chain must be
    # signed by the next cert's private key (verified with its public key).
    for i, cert in enumerate(chain):
        # Verify signature
        if not verify_certificate(cert):
            return False, f"Invalid signature on cert {i}: {cert.subject}"

        # Check CA flag for intermediate certs
        if i < len(chain) - 1:
            # The issuer cert is the next in the chain
            issuer_cert = chain[i + 1]
            if not issuer_cert.is_ca:
                return False, f"Issuer {issuer_cert.subject} is not a CA"

            # Why: Path length constraint prevents unauthorized sub-CAs.
            # If a CA has path_length=0, it can only sign end-entity certs,
            # not other CAs. This limits the depth of trust delegation.
            if issuer_cert.path_length is not None:
                remaining_depth = len(chain) - i - 2
                if remaining_depth > issuer_cert.path_length:
                    return False, (
                        f"Path length exceeded for {issuer_cert.subject}"
                    )

    # Check self-signed root
    root = chain[-1]
    if root.subject != root.issuer:
        return False, "Root certificate is not self-signed"

    return True, "Chain is valid"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  Public Key Infrastructure & Certificates")
    print("=" * 65)

    # Create Root CA
    print("\n[1. Create Root CA]")
    root_ca = CertificateAuthority("GlobalTrust Root CA", key_bits=512)
    root_cert = root_ca.create_self_signed_cert()
    print(f"  {root_cert}")
    print(f"  Self-signed: {root_cert.subject == root_cert.issuer}")
    print(f"  Signature valid: {verify_certificate(root_cert)}")

    # Create Intermediate CA
    print("\n[2. Create Intermediate CA]")
    inter_ca = CertificateAuthority("SecureSign Intermediate CA",
                                     key_bits=512, parent=root_ca)
    inter_cert = root_ca.issue_certificate(
        "SecureSign Intermediate CA",
        inter_ca.public_key,
        is_ca=True,
        path_length=0,  # can only sign end-entity certs
    )
    inter_ca.certificate = inter_cert
    print(f"  {inter_cert}")
    print(f"  Signed by: {inter_cert.issuer}")
    print(f"  Signature valid: {verify_certificate(inter_cert)}")

    # Create End-Entity Certificate (website)
    print("\n[3. Issue Website Certificate]")
    site_pub, site_priv = rsa_keygen(512)
    site_cert = inter_ca.issue_certificate(
        "www.example.com",
        site_pub,
        is_ca=False,
    )
    print(f"  {site_cert}")
    print(f"  Signed by: {site_cert.issuer}")
    print(f"  Signature valid: {verify_certificate(site_cert)}")

    # Verify the full chain
    print("\n[4. Verify Certificate Chain]")
    chain = [site_cert, inter_cert, root_cert]
    print(f"  Chain: {' -> '.join(c.subject.replace('CN=', '') for c in chain)}")

    valid, reason = verify_chain(chain)
    print(f"  Valid: {valid}")
    print(f"  Reason: {reason}")

    # Demonstrate chain failure: tampered certificate
    print("\n[5. Tampered Certificate Detection]")
    tampered_cert = Certificate(
        serial_number=site_cert.serial_number,
        subject="CN=www.evil.com",  # changed subject!
        issuer=site_cert.issuer,
        not_before=site_cert.not_before,
        not_after=site_cert.not_after,
        public_key=site_cert.public_key,
        signature=site_cert.signature,  # reusing original signature
        issuer_public_key=site_cert.issuer_public_key,
    )
    tampered_chain = [tampered_cert, inter_cert, root_cert]
    valid_t, reason_t = verify_chain(tampered_chain)
    print(f"  Tampered subject: '{tampered_cert.subject}'")
    print(f"  Valid: {valid_t}")
    print(f"  Reason: {reason_t}")

    # Demonstrate missing intermediate
    print("\n[6. Incomplete Chain Detection]")
    short_chain = [site_cert, root_cert]
    valid_s, reason_s = verify_chain(short_chain)
    print(f"  Chain without intermediate:")
    print(f"  Valid: {valid_s}")
    print(f"  Reason: {reason_s}")

    # Summary
    print(f"\n[PKI Trust Model Summary]")
    print(f"  Root CA (self-signed, pre-installed in OS/browser)")
    print(f"    |")
    print(f"    +-- Intermediate CA (signed by Root)")
    print(f"          |")
    print(f"          +-- www.example.com (signed by Intermediate)")
    print(f"")
    print(f"  Your browser verifies the chain bottom-up,")
    print(f"  checking each signature against the issuer's public key.")

    print(f"\n{'=' * 65}")
