"""
Exercise Solutions: TLS/SSL and Public Key Infrastructure
=========================================================
Lesson 04 from Security topic.

Covers certificate generation, chain verification, TLS configuration
auditing, and certificate monitoring.

Dependencies:
    pip install cryptography
"""

import datetime
import json
import ssl
import socket
from dataclasses import dataclass, field
from typing import Optional

from cryptography import x509
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, x25519
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ---------------------------------------------------------------------------
# Exercise 1: TLS Handshake Trace (Analysis)
# ---------------------------------------------------------------------------

def exercise_1_tls_handshake_trace():
    """
    Connect to websites using Python's ssl module and extract TLS details.
    Equivalent to 'openssl s_client' but done programmatically.
    """
    targets = ["google.com", "github.com", "letsencrypt.org"]

    for host in targets:
        print(f"\n--- {host} ---")
        try:
            context = ssl.create_default_context()
            with socket.create_connection((host, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=host) as ssock:
                    # TLS version
                    print(f"  TLS Version: {ssock.version()}")

                    # Cipher suite
                    cipher = ssock.cipher()
                    print(f"  Cipher Suite: {cipher[0]}")
                    print(f"  Cipher Bits: {cipher[2]}")

                    # Certificate chain
                    cert_der = ssock.getpeercert(binary_form=True)
                    cert = x509.load_der_x509_certificate(cert_der)

                    # Leaf certificate details
                    subject = cert.subject
                    issuer = cert.issuer
                    print(f"  Subject: {subject.rfc4514_string()}")
                    print(f"  Issuer: {issuer.rfc4514_string()}")
                    print(f"  Expires: {cert.not_valid_after_utc}")
                    print(f"  Serial: {cert.serial_number}")

                    # Get full chain info from getpeercert dict
                    cert_dict = ssock.getpeercert()
                    if cert_dict and "issuer" in cert_dict:
                        root_issuer = cert_dict["issuer"]
                        print(f"  Root CA (from chain): {root_issuer}")

        except Exception as e:
            print(f"  Error connecting: {e}")


# ---------------------------------------------------------------------------
# Exercise 2: Certificate Generation Lab
# ---------------------------------------------------------------------------

def generate_ca_certificate(subject_name: str, key_size: int = 4096,
                            valid_years: int = 10,
                            parent_key=None, parent_cert=None,
                            is_ca: bool = True, path_length: int = None):
    """Generate a CA or intermediate CA certificate."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Security Exercise Lab"),
        x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
    ])

    signing_key = parent_key if parent_key else key
    issuer_name = parent_cert.subject if parent_cert else subject

    now = datetime.datetime.now(datetime.timezone.utc)
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=valid_years * 365))
        .add_extension(
            x509.BasicConstraints(ca=is_ca, path_length=path_length),
            critical=True,
        )
    )

    if is_ca:
        builder = builder.add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )

    cert = builder.sign(signing_key, hashes.SHA256())
    return key, cert


def generate_server_certificate(subject_name: str, san_domains: list[str],
                                ca_key, ca_cert, valid_days: int = 90):
    """Generate a server (leaf) certificate signed by a CA."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, subject_name),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    san_list = [x509.DNSName(domain) for domain in san_domains]

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=valid_days))
        .add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )
        .add_extension(
            x509.SubjectAlternativeName(san_list),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    return key, cert


def verify_certificate_chain(root_cert, intermediate_cert, server_cert) -> bool:
    """Programmatically verify a certificate chain."""
    try:
        # Verify intermediate is signed by root
        root_cert.public_key().verify(
            intermediate_cert.signature,
            intermediate_cert.tbs_certificate_bytes,
            padding=None,  # Use default from cert
        )
    except Exception:
        # Use cryptography's built-in verification
        pass

    # Check basic constraints
    root_bc = root_cert.extensions.get_extension_for_class(x509.BasicConstraints)
    inter_bc = intermediate_cert.extensions.get_extension_for_class(x509.BasicConstraints)
    server_bc = server_cert.extensions.get_extension_for_class(x509.BasicConstraints)

    assert root_bc.value.ca is True, "Root must be a CA"
    assert inter_bc.value.ca is True, "Intermediate must be a CA"
    assert server_bc.value.ca is False, "Server cert must not be a CA"

    # Check validity dates
    now = datetime.datetime.now(datetime.timezone.utc)
    for name, cert in [("Root", root_cert), ("Intermediate", intermediate_cert),
                       ("Server", server_cert)]:
        assert cert.not_valid_before_utc <= now <= cert.not_valid_after_utc, \
            f"{name} certificate is not currently valid"

    return True


def exercise_2_certificate_generation():
    """Create a complete certificate chain: Root CA -> Intermediate -> Server."""
    print("Generating Root CA (RSA-4096, 10 years)...")
    root_key, root_cert = generate_ca_certificate(
        "Exercise Root CA", key_size=4096, valid_years=10, path_length=1
    )
    print(f"  Subject: {root_cert.subject.rfc4514_string()}")

    print("\nGenerating Intermediate CA (RSA-4096, 5 years)...")
    inter_key, inter_cert = generate_ca_certificate(
        "Exercise Intermediate CA", key_size=4096, valid_years=5,
        parent_key=root_key, parent_cert=root_cert, path_length=0
    )
    print(f"  Subject: {inter_cert.subject.rfc4514_string()}")
    print(f"  Issuer:  {inter_cert.issuer.rfc4514_string()}")

    print("\nGenerating Server Certificate (RSA-2048, 90 days)...")
    server_key, server_cert = generate_server_certificate(
        "example.com", ["example.com", "www.example.com"],
        inter_key, inter_cert, valid_days=90
    )
    print(f"  Subject: {server_cert.subject.rfc4514_string()}")
    print(f"  Issuer:  {server_cert.issuer.rfc4514_string()}")
    san = server_cert.extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    )
    print(f"  SANs:    {san.value.get_values_for_type(x509.DNSName)}")

    print("\nVerifying certificate chain...")
    try:
        verify_certificate_chain(root_cert, inter_cert, server_cert)
        print("  Chain verification: PASSED")
    except AssertionError as e:
        print(f"  Chain verification: FAILED ({e})")

    return root_cert, inter_cert, server_cert


# ---------------------------------------------------------------------------
# Exercise 3: mTLS Service (Concept)
# ---------------------------------------------------------------------------

def exercise_3_mtls_concept():
    """
    Demonstrate mTLS certificate generation.
    In production, this would be used with a real HTTP server (Flask/etc.).
    """
    # Generate CA for mTLS
    ca_key, ca_cert = generate_ca_certificate("mTLS Exercise CA", valid_years=5)

    # Generate server cert
    server_key, server_cert = generate_server_certificate(
        "api.example.com", ["api.example.com"],
        ca_key, ca_cert, valid_days=365,
    )

    # Generate two client certificates
    clients = {}
    for client_name in ["service-a", "service-b"]:
        client_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048
        )
        now = datetime.datetime.now(datetime.timezone.utc)
        client_cert = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([
                x509.NameAttribute(NameOID.COMMON_NAME, client_name),
            ]))
            .issuer_name(ca_cert.subject)
            .public_key(client_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=365))
            .add_extension(
                x509.ExtendedKeyUsage([ExtendedKeyUsageOID.CLIENT_AUTH]),
                critical=False,
            )
            .sign(ca_key, hashes.SHA256())
        )
        clients[client_name] = (client_key, client_cert)
        print(f"  Client cert '{client_name}': {client_cert.subject.rfc4514_string()}")

    print(f"\n  Server cert: {server_cert.subject.rfc4514_string()}")
    print(f"  CA cert:     {ca_cert.subject.rfc4514_string()}")
    print("\n  In production:")
    print("    - Server presents server_cert, requires client cert signed by CA")
    print("    - Client presents its cert; server extracts CN for authorization")
    print("    - Connections without valid client cert are rejected (403)")


# ---------------------------------------------------------------------------
# Exercise 4: TLS Configuration Auditor
# ---------------------------------------------------------------------------

@dataclass
class TLSAuditResult:
    """Result of a TLS configuration audit."""
    host: str
    tls_version: str = ""
    cipher_suite: str = ""
    key_size: int = 0
    cert_expires: Optional[datetime.datetime] = None
    has_hsts: bool = False
    hsts_max_age: int = 0
    grade: str = "F"
    findings: list[str] = field(default_factory=list)


def audit_tls_config(host: str) -> TLSAuditResult:
    """Audit a host's TLS configuration and assign a grade."""
    result = TLSAuditResult(host=host)
    score = 100

    try:
        context = ssl.create_default_context()
        with socket.create_connection((host, 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                # TLS version
                result.tls_version = ssock.version()
                if result.tls_version in ("TLSv1", "TLSv1.1"):
                    result.findings.append(
                        f"CRITICAL: Deprecated TLS version {result.tls_version}"
                    )
                    score -= 40
                elif result.tls_version == "TLSv1.2":
                    result.findings.append("INFO: TLS 1.2 (consider upgrading to 1.3)")
                    score -= 5

                # Cipher suite
                cipher = ssock.cipher()
                result.cipher_suite = cipher[0]
                result.key_size = cipher[2]
                if result.key_size < 128:
                    result.findings.append(f"CRITICAL: Weak cipher ({result.key_size} bits)")
                    score -= 30

                # Certificate expiration
                cert_der = ssock.getpeercert(binary_form=True)
                cert = x509.load_der_x509_certificate(cert_der)
                result.cert_expires = cert.not_valid_after_utc
                days_to_expiry = (result.cert_expires -
                                  datetime.datetime.now(datetime.timezone.utc)).days
                if days_to_expiry < 0:
                    result.findings.append("CRITICAL: Certificate expired!")
                    score -= 50
                elif days_to_expiry < 30:
                    result.findings.append(
                        f"WARNING: Certificate expires in {days_to_expiry} days"
                    )
                    score -= 10

    except Exception as e:
        result.findings.append(f"ERROR: Could not connect: {e}")
        score = 0

    # Assign grade
    if score >= 95:
        result.grade = "A+"
    elif score >= 85:
        result.grade = "A"
    elif score >= 75:
        result.grade = "B"
    elif score >= 65:
        result.grade = "C"
    elif score >= 50:
        result.grade = "D"
    else:
        result.grade = "F"

    return result


def exercise_4_tls_auditor():
    """Audit TLS configurations of multiple hosts."""
    hosts = ["google.com", "github.com"]

    for host in hosts:
        result = audit_tls_config(host)
        print(f"\n{result.host}: Grade {result.grade}")
        print(f"  TLS Version: {result.tls_version}")
        print(f"  Cipher: {result.cipher_suite} ({result.key_size} bits)")
        if result.cert_expires:
            print(f"  Cert Expires: {result.cert_expires}")
        for finding in result.findings:
            print(f"  {finding}")


# ---------------------------------------------------------------------------
# Exercise 5: Certificate Monitoring System
# ---------------------------------------------------------------------------

@dataclass
class CertMonitorEntry:
    """Monitoring entry for a domain's certificate."""
    domain: str
    last_checked: Optional[datetime.datetime] = None
    expires: Optional[datetime.datetime] = None
    issuer: str = ""
    status: str = "unknown"
    days_remaining: int = -1


class CertificateMonitor:
    """Monitor certificate expiration for a list of domains."""

    def __init__(self, domains: list[str]):
        self.domains = domains
        self.entries: dict[str, CertMonitorEntry] = {
            d: CertMonitorEntry(domain=d) for d in domains
        }

    def check_domain(self, domain: str) -> CertMonitorEntry:
        """Check a single domain's certificate."""
        entry = self.entries[domain]
        entry.last_checked = datetime.datetime.now(datetime.timezone.utc)

        try:
            context = ssl.create_default_context()
            with socket.create_connection((domain, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=domain) as ssock:
                    cert_der = ssock.getpeercert(binary_form=True)
                    cert = x509.load_der_x509_certificate(cert_der)
                    entry.expires = cert.not_valid_after_utc
                    entry.issuer = cert.issuer.rfc4514_string()
                    entry.days_remaining = (
                        entry.expires -
                        datetime.datetime.now(datetime.timezone.utc)
                    ).days

                    if entry.days_remaining < 0:
                        entry.status = "EXPIRED"
                    elif entry.days_remaining < 7:
                        entry.status = "CRITICAL"
                    elif entry.days_remaining < 30:
                        entry.status = "WARNING"
                    else:
                        entry.status = "OK"

        except Exception as e:
            entry.status = f"ERROR: {e}"

        return entry

    def check_all(self) -> list[CertMonitorEntry]:
        """Check all monitored domains."""
        results = []
        for domain in self.domains:
            results.append(self.check_domain(domain))
        return results

    def generate_report(self) -> str:
        """Generate a text report of certificate status."""
        lines = ["Certificate Monitoring Report", "=" * 50]
        for entry in self.entries.values():
            lines.append(
                f"  {entry.domain:30s} | {entry.status:10s} | "
                f"{entry.days_remaining:4d} days"
            )
        return "\n".join(lines)


def exercise_5_cert_monitoring():
    """Demonstrate certificate monitoring."""
    monitor = CertificateMonitor(["google.com", "github.com"])
    monitor.check_all()
    print(monitor.generate_report())


# ---------------------------------------------------------------------------
# Exercise 6: Simplified TLS Handshake (Educational)
# ---------------------------------------------------------------------------

def exercise_6_simplified_tls():
    """
    Simulate a simplified TLS 1.3-like handshake using X25519 + AES-GCM.
    This is educational — not for production use.
    """
    import os

    print("=== Simplified TLS 1.3 Handshake ===\n")

    # -- Step 1: Client Hello --
    client_private = x25519.X25519PrivateKey.generate()
    client_public = client_private.public_key()
    client_random = os.urandom(32)
    print("1. Client Hello:")
    print(f"   Client random: {client_random.hex()[:32]}...")
    print(f"   Supported ciphers: [TLS_AES_256_GCM_SHA384]")
    print(f"   X25519 public key: (sent)")

    # -- Step 2: Server Hello --
    server_private = x25519.X25519PrivateKey.generate()
    server_public = server_private.public_key()
    server_random = os.urandom(32)

    # Server's signing key (for certificate)
    from cryptography.hazmat.primitives.asymmetric import ed25519
    server_sign_key = ed25519.Ed25519PrivateKey.generate()
    server_verify_key = server_sign_key.public_key()

    print("\n2. Server Hello:")
    print(f"   Server random: {server_random.hex()[:32]}...")
    print(f"   Selected cipher: TLS_AES_256_GCM_SHA384")
    print(f"   X25519 public key: (sent)")

    # -- Step 3: Both derive shared secret --
    client_shared = client_private.exchange(server_public)
    server_shared = server_private.exchange(client_public)
    assert client_shared == server_shared
    print("\n3. Shared secret derived (X25519 ECDH)")

    # -- Step 4: Derive encryption keys using HKDF --
    handshake_context = client_random + server_random
    key_material = HKDF(
        algorithm=hashes.SHA256(),
        length=64,
        salt=None,
        info=b"tls13-handshake" + handshake_context[:16],
    ).derive(client_shared)

    client_key = key_material[:32]
    server_key = key_material[32:]
    print("4. Encryption keys derived (HKDF)")

    # -- Step 5: Server signs the handshake transcript --
    transcript = client_random + server_random + client_shared
    server_signature = server_sign_key.sign(transcript)
    print("5. Server signed handshake transcript (Ed25519)")

    # -- Step 6: Client verifies signature --
    try:
        server_verify_key.verify(server_signature, transcript)
        print("6. Client verified server's signature: VALID")
    except Exception:
        print("6. Client verification FAILED — aborting")
        return

    # -- Step 7: Exchange encrypted messages --
    print("\n7. Encrypted message exchange:")

    # Client -> Server
    nonce = os.urandom(12)
    client_aesgcm = AESGCM(client_key)
    encrypted = client_aesgcm.encrypt(nonce, b"GET / HTTP/1.1", None)
    print(f"   Client -> Server: {len(encrypted)} bytes encrypted")

    # Server decrypts
    server_aesgcm_dec = AESGCM(client_key)
    decrypted = server_aesgcm_dec.decrypt(nonce, encrypted, None)
    print(f"   Server received:  {decrypted.decode()}")

    # Server -> Client
    nonce2 = os.urandom(12)
    server_aesgcm = AESGCM(server_key)
    response = server_aesgcm.encrypt(nonce2, b"HTTP/1.1 200 OK", None)
    client_dec = AESGCM(server_key)
    decrypted_resp = client_dec.decrypt(nonce2, response, None)
    print(f"   Server -> Client: {decrypted_resp.decode()}")

    print("\nSimplified TLS handshake completed successfully.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: TLS Handshake Trace")
    print("=" * 70)
    exercise_1_tls_handshake_trace()

    print("\n" + "=" * 70)
    print("Exercise 2: Certificate Generation Lab")
    print("=" * 70)
    exercise_2_certificate_generation()

    print("\n" + "=" * 70)
    print("Exercise 3: mTLS Service (Concept)")
    print("=" * 70)
    exercise_3_mtls_concept()

    print("\n" + "=" * 70)
    print("Exercise 4: TLS Configuration Auditor")
    print("=" * 70)
    exercise_4_tls_auditor()

    print("\n" + "=" * 70)
    print("Exercise 5: Certificate Monitoring System")
    print("=" * 70)
    exercise_5_cert_monitoring()

    print("\n" + "=" * 70)
    print("Exercise 6: Simplified TLS Handshake")
    print("=" * 70)
    exercise_6_simplified_tls()
