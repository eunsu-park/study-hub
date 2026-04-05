"""
Exercises for Lesson 09: PKI and Certificates
Topic: Cryptography_Theory
Solutions to practice problems from the lesson.
"""

import hashlib
import json
import os
import random
import time
from datetime import datetime, timedelta


def exercise_1():
    """Exercise 1: Certificate Inspection (Coding)

    Write a script that:
    1. Fetches the certificate chain from an HTTPS website
    2. Prints subject, issuer, validity, SAN for each cert
    3. Verifies signatures in the chain
    4. Reports if chain leads to a trusted root
    """
    # Simulated certificate chain (no network dependency)
    # In production, use ssl + cryptography library to fetch real certs

    print(f"  Certificate Inspection (Simulated)")
    print(f"  Note: Using simulated certificates to avoid network dependency.")
    print(f"  For real certificate inspection, use the cryptography library.")

    # Simulated certificate chain for github.com
    chain = [
        {
            "subject": "CN=github.com, O=GitHub, Inc.",
            "issuer": "CN=DigiCert SHA2 High Assurance Server CA, O=DigiCert Inc",
            "not_before": "2024-03-07",
            "not_after": "2025-03-08",
            "san": ["github.com", "www.github.com"],
            "sig_algo": "SHA256withECDSA",
            "key_type": "EC",
            "key_size": 256,
            "serial": "0A:B7:C8:D9:E0:F1:23:45",
            "is_ca": False,
        },
        {
            "subject": "CN=DigiCert SHA2 High Assurance Server CA, O=DigiCert Inc",
            "issuer": "CN=DigiCert High Assurance EV Root CA, O=DigiCert Inc",
            "not_before": "2013-10-22",
            "not_after": "2028-10-22",
            "san": [],
            "sig_algo": "SHA256withRSA",
            "key_type": "RSA",
            "key_size": 2048,
            "serial": "04:E1:E7:A4:DC:5C:F2:F3",
            "is_ca": True,
        },
        {
            "subject": "CN=DigiCert High Assurance EV Root CA, O=DigiCert Inc",
            "issuer": "CN=DigiCert High Assurance EV Root CA, O=DigiCert Inc",
            "not_before": "2006-11-10",
            "not_after": "2031-11-10",
            "san": [],
            "sig_algo": "SHA256withRSA",
            "key_type": "RSA",
            "key_size": 4096,
            "serial": "02:AC:5C:26:6A:0B:40:9B",
            "is_ca": True,
        },
    ]

    trust_store_subjects = [
        "CN=DigiCert High Assurance EV Root CA, O=DigiCert Inc",
        "CN=DigiCert Global Root G2, O=DigiCert Inc",
        "CN=ISRG Root X1, O=Internet Security Research Group",
    ]

    now = datetime(2025, 1, 15)

    for i, cert in enumerate(chain):
        role = "End-Entity" if i == 0 else ("Intermediate CA" if i < len(chain) - 1 else "Root CA")
        print(f"\n  Certificate {i} ({role}):")
        print(f"    Subject:    {cert['subject']}")
        print(f"    Issuer:     {cert['issuer']}")
        print(f"    Valid:      {cert['not_before']} to {cert['not_after']}")
        print(f"    Sig Algo:   {cert['sig_algo']}")
        print(f"    Key:        {cert['key_type']} {cert['key_size']}-bit")
        print(f"    Serial:     {cert['serial']}")
        print(f"    Is CA:      {cert['is_ca']}")
        if cert['san']:
            print(f"    SAN:        {', '.join(cert['san'])}")

        # Check validity
        not_before = datetime.strptime(cert['not_before'], "%Y-%m-%d")
        not_after = datetime.strptime(cert['not_after'], "%Y-%m-%d")
        valid = not_before <= now <= not_after
        print(f"    Currently valid (as of {now.date()}): {valid}")

    # Verify chain structure
    print(f"\n  Chain Verification:")
    valid_chain = True

    for i in range(len(chain) - 1):
        issuer_match = chain[i]['issuer'] == chain[i + 1]['subject']
        print(f"    Cert {i} issuer == Cert {i+1} subject: {issuer_match}")
        if not issuer_match:
            valid_chain = False

    # Check if root is self-signed
    root = chain[-1]
    self_signed = root['subject'] == root['issuer']
    print(f"    Root is self-signed: {self_signed}")

    # Check trust store
    root_trusted = root['subject'] in trust_store_subjects
    print(f"    Root in trust store: {root_trusted}")

    # Check CA constraints
    for i in range(1, len(chain)):
        if not chain[i]['is_ca']:
            print(f"    WARNING: Cert {i} is used as CA but CA:FALSE!")
            valid_chain = False

    print(f"\n  Chain valid: {valid_chain and root_trusted}")
    print(f"\n  Production code example (using cryptography library):")
    print(f"    from cryptography import x509")
    print(f"    import ssl, socket")
    print(f"    ctx = ssl.create_default_context()")
    print(f"    with socket.create_connection(('github.com', 443)) as sock:")
    print(f"        with ctx.wrap_socket(sock, server_hostname='github.com') as ssock:")
    print(f"            der = ssock.getpeercert(binary_form=True)")
    print(f"    cert = x509.load_der_x509_certificate(der)")


def exercise_2():
    """Exercise 2: Self-Signed CA (Coding)

    Create a certificate hierarchy using simulated crypto:
    1. Root CA self-signed cert
    2. Intermediate CA cert signed by root
    3. End-entity cert for test.example.com
    4. Verify the chain
    """
    # Simulated RSA key generation and signing
    class SimulatedKey:
        """Simulated cryptographic key for educational purposes."""
        def __init__(self, name):
            self.name = name
            self.private = random.getrandbits(256)
            self.public = hashlib.sha256(str(self.private).encode()).hexdigest()

        def sign(self, data):
            """Simulated signature: HMAC with private key."""
            return hashlib.sha256(
                str(self.private).encode() + data.encode()
            ).hexdigest()

        def verify(self, data, signature):
            """Verify a simulated signature."""
            expected = hashlib.sha256(
                str(self.private).encode() + data.encode()
            ).hexdigest()
            return signature == expected

    class SimulatedCertificate:
        """Simulated X.509 certificate."""
        def __init__(self, subject, issuer_name, public_key, is_ca,
                     san=None, validity_days=365):
            self.subject = subject
            self.issuer_name = issuer_name
            self.public_key = public_key
            self.is_ca = is_ca
            self.san = san or []
            self.serial = random.getrandbits(128)
            self.not_before = datetime.utcnow()
            self.not_after = self.not_before + timedelta(days=validity_days)
            self.signature = None

        def tbs_data(self):
            """The 'to be signed' data."""
            return (f"subject={self.subject}|issuer={self.issuer_name}|"
                    f"pubkey={self.public_key}|ca={self.is_ca}|"
                    f"san={self.san}|serial={self.serial}")

        def sign_with(self, issuer_key):
            """Sign this certificate with the issuer's key."""
            self.signature = issuer_key.sign(self.tbs_data())

        def verify_signature(self, issuer_key):
            """Verify this certificate's signature."""
            return issuer_key.verify(self.tbs_data(), self.signature)

        def __repr__(self):
            return f"Certificate(subject={self.subject}, issuer={self.issuer_name})"

    print(f"  Self-Signed CA Hierarchy")

    # Step 1: Root CA
    root_key = SimulatedKey("Root CA Key")
    root_cert = SimulatedCertificate(
        subject="CN=My Root CA, O=Education",
        issuer_name="CN=My Root CA, O=Education",  # Self-signed
        public_key=root_key.public,
        is_ca=True,
        validity_days=365 * 20  # 20 years
    )
    root_cert.sign_with(root_key)  # Self-signed

    print(f"\n  Step 1: Root CA (self-signed)")
    print(f"    Subject:  {root_cert.subject}")
    print(f"    Issuer:   {root_cert.issuer_name}")
    print(f"    CA:       {root_cert.is_ca}")
    print(f"    Valid:     {root_cert.not_before.date()} to {root_cert.not_after.date()}")
    print(f"    Sig valid: {root_cert.verify_signature(root_key)}")

    # Step 2: Intermediate CA
    inter_key = SimulatedKey("Intermediate CA Key")
    inter_cert = SimulatedCertificate(
        subject="CN=My Intermediate CA, O=Education",
        issuer_name=root_cert.subject,
        public_key=inter_key.public,
        is_ca=True,
        validity_days=365 * 5  # 5 years
    )
    inter_cert.sign_with(root_key)  # Signed by root

    print(f"\n  Step 2: Intermediate CA (signed by Root)")
    print(f"    Subject:  {inter_cert.subject}")
    print(f"    Issuer:   {inter_cert.issuer_name}")
    print(f"    CA:       {inter_cert.is_ca}")
    print(f"    Valid:     {inter_cert.not_before.date()} to {inter_cert.not_after.date()}")
    print(f"    Sig valid: {inter_cert.verify_signature(root_key)}")

    # Step 3: End-entity certificate
    ee_key = SimulatedKey("End-Entity Key")
    ee_cert = SimulatedCertificate(
        subject="CN=test.example.com",
        issuer_name=inter_cert.subject,
        public_key=ee_key.public,
        is_ca=False,
        san=["test.example.com", "www.test.example.com"],
        validity_days=90  # 90 days (like Let's Encrypt)
    )
    ee_cert.sign_with(inter_key)  # Signed by intermediate

    print(f"\n  Step 3: End-Entity Certificate (signed by Intermediate)")
    print(f"    Subject:  {ee_cert.subject}")
    print(f"    Issuer:   {ee_cert.issuer_name}")
    print(f"    SAN:      {', '.join(ee_cert.san)}")
    print(f"    CA:       {ee_cert.is_ca}")
    print(f"    Valid:     {ee_cert.not_before.date()} to {ee_cert.not_after.date()}")
    print(f"    Sig valid: {ee_cert.verify_signature(inter_key)}")

    # Step 4: Verify the complete chain
    print(f"\n  Step 4: Chain Verification")
    chain = [ee_cert, inter_cert, root_cert]
    keys = [None, inter_key, root_key]  # Keys for verification

    all_valid = True
    for i, cert in enumerate(chain):
        # Verify issuer link
        if i + 1 < len(chain):
            issuer_match = cert.issuer_name == chain[i + 1].subject
            sig_valid = cert.verify_signature(keys[i + 1])
        else:
            issuer_match = cert.subject == cert.issuer_name  # Self-signed root
            sig_valid = cert.verify_signature(root_key)

        ca_ok = True
        if i > 0 and not cert.is_ca:
            ca_ok = False

        status = "OK" if (issuer_match and sig_valid and ca_ok) else "FAIL"
        if not (issuer_match and sig_valid and ca_ok):
            all_valid = False

        print(f"    Cert {i}: issuer_match={issuer_match}, "
              f"sig_valid={sig_valid}, ca_ok={ca_ok} -> {status}")

    print(f"\n  Complete chain valid: {all_valid}")


def exercise_3():
    """Exercise 3: CT Log Analysis (Conceptual + Coding)

    Query CT logs for a domain and analyze certificates.
    (Simulated to avoid network dependency)
    """
    print(f"  CT Log Analysis")
    print(f"  Note: Using simulated CT log data to avoid network dependency.")

    # Simulated CT log entries for github.com
    ct_entries = [
        {"id": 1001, "issuer": "DigiCert SHA2 High Assurance Server CA",
         "not_before": "2024-03-07", "not_after": "2025-03-08",
         "common_name": "github.com", "san": "github.com,www.github.com"},
        {"id": 1002, "issuer": "DigiCert SHA2 High Assurance Server CA",
         "not_before": "2023-03-07", "not_after": "2024-03-08",
         "common_name": "github.com", "san": "github.com,www.github.com"},
        {"id": 1003, "issuer": "DigiCert SHA2 High Assurance Server CA",
         "not_before": "2022-03-08", "not_after": "2023-03-08",
         "common_name": "github.com", "san": "github.com,www.github.com"},
        {"id": 1004, "issuer": "Let's Encrypt Authority X3",
         "not_before": "2024-06-01", "not_after": "2024-08-30",
         "common_name": "github.com.phishing-site.evil.com",
         "san": "github.com.phishing-site.evil.com"},
        {"id": 1005, "issuer": "DigiCert SHA2 High Assurance Server CA",
         "not_before": "2021-03-08", "not_after": "2022-03-08",
         "common_name": "github.com", "san": "github.com,www.github.com"},
    ]

    # Part 1: How many certificates in the past year?
    print(f"\n  Part 1: Certificates issued for github.com")
    target_year = 2024
    recent = [e for e in ct_entries
              if e['not_before'].startswith(str(target_year))]
    print(f"    Certificates issued in {target_year}: {len(recent)}")
    for e in recent:
        print(f"      ID {e['id']}: {e['issuer']}, {e['not_before']} to {e['not_after']}")

    # Part 2: Which CAs issued them?
    print(f"\n  Part 2: CAs that issued certificates")
    ca_counts = {}
    for e in ct_entries:
        ca = e['issuer']
        ca_counts[ca] = ca_counts.get(ca, 0) + 1
    for ca, count in sorted(ca_counts.items(), key=lambda x: -x[1]):
        print(f"    {ca}: {count} certificates")

    # Part 3: Unexpected certificates?
    print(f"\n  Part 3: Check for unexpected certificates")
    expected_cas = {"DigiCert SHA2 High Assurance Server CA"}
    for e in ct_entries:
        if e['issuer'] not in expected_cas:
            # Check if it's actually for github.com or a lookalike
            if 'github.com' in e['common_name'] and e['common_name'] != 'github.com':
                print(f"    WARNING: Potential phishing cert!")
                print(f"      ID: {e['id']}")
                print(f"      CN: {e['common_name']}")
                print(f"      Issuer: {e['issuer']}")
            elif e['common_name'] == 'github.com':
                print(f"    ALERT: Unexpected CA for github.com!")
                print(f"      ID: {e['id']}")
                print(f"      Issuer: {e['issuer']}")

    # Part 4: Monitoring script concept
    print(f"\n  Part 4: CT Log Monitoring Script")
    print(f"    Production monitoring script concept:")
    print(f"    1. Query crt.sh API: https://crt.sh/?q=yourdomain.com&output=json")
    print(f"    2. Filter for certificates issued since last check")
    print(f"    3. Check if issuer is in your expected CA list")
    print(f"    4. Alert if unexpected CA or unexpected domain pattern")
    print(f"    5. Store seen certificate IDs to avoid duplicate alerts")
    print(f"    6. Run as a cron job every 15-60 minutes")


def exercise_4():
    """Exercise 4: Revocation Trade-offs (Conceptual)

    Compare CRL, OCSP, OCSP Stapling, and Short-Lived Certificates.
    """
    print(f"  Revocation Mechanism Comparison")

    mechanisms = {
        "CRL": {
            "bandwidth": "High (download full list, can be MBs)",
            "latency": "Medium (periodic download, cached)",
            "privacy": "Good (no per-site queries to CA)",
            "security": "Weak (stale data between updates; attacker can block download -> soft fail)",
            "server_complexity": "None (CA publishes, client downloads)",
        },
        "OCSP": {
            "bandwidth": "Low (per-certificate query)",
            "latency": "High (real-time query to CA per connection)",
            "privacy": "Poor (CA sees every site you visit)",
            "security": "Medium (real-time but soft-fail allows bypass)",
            "server_complexity": "None (CA operates responder)",
        },
        "OCSP Stapling": {
            "bandwidth": "Low (server caches and sends OCSP response)",
            "latency": "None (included in TLS handshake)",
            "privacy": "Excellent (no client-CA communication)",
            "security": "Good (server provides fresh signed response; must-staple pins make it hard-fail)",
            "server_complexity": "Medium (server must periodically fetch OCSP responses)",
        },
        "Short-Lived Certs": {
            "bandwidth": "None (no revocation infrastructure needed)",
            "latency": "None (no revocation checks needed)",
            "privacy": "Excellent (no revocation queries)",
            "security": "Excellent (cert expires quickly; max exposure = cert lifetime)",
            "server_complexity": "High (automated renewal every 24-48 hours)",
        },
    }

    # Print comparison table
    dimensions = ["bandwidth", "latency", "privacy", "security", "server_complexity"]
    header = f"  {'Dimension':<20}"
    for mech in mechanisms:
        header += f" {mech:<20}"
    print(header)
    print(f"  {'-'*20}" + f" {'-'*20}" * len(mechanisms))

    for dim in dimensions:
        row = f"  {dim:<20}"
        for mech in mechanisms:
            val = mechanisms[mech][dim]
            # Truncate for display
            short = val[:18] + ".." if len(val) > 20 else val
            row += f" {short:<20}"
        print(row)

    # Detailed explanations
    for mech, props in mechanisms.items():
        print(f"\n  {mech}:")
        for dim, val in props.items():
            print(f"    {dim}: {val}")

    # Recommendations
    print(f"\n  Recommendations:")
    print(f"    (a) High-traffic web service:")
    print(f"        -> OCSP Stapling + Must-Staple extension")
    print(f"        Rationale: No client-CA traffic, no privacy leak,")
    print(f"        hard-fail with must-staple prevents bypass")
    print(f"")
    print(f"    (b) IoT device:")
    print(f"        -> Short-lived certificates (24-48 hour)")
    print(f"        Rationale: IoT devices have limited connectivity;")
    print(f"        CRL/OCSP adds latency and failure modes;")
    print(f"        short-lived certs need no revocation infrastructure")
    print(f"")
    print(f"    (c) Mobile app:")
    print(f"        -> Certificate pinning + OCSP Stapling")
    print(f"        Rationale: App controls which CAs are trusted;")
    print(f"        pinning limits attack surface; stapling avoids")
    print(f"        OCSP latency and privacy issues")


def exercise_5():
    """Exercise 5: PKI Design Challenge (Challenging)

    Design a PKI for a university with 50,000 users.
    """
    print(f"  PKI Design for University (50,000 users)")
    print(f"  {'='*55}")

    print(f"""
  1. Certificate Hierarchy
  ========================
  Root CA (offline, in HSM)
   |
   +-- Email CA (S/MIME certificates)
   |    |-- User email certs (50,000 users)
   |
   +-- Infrastructure CA
   |    |-- VPN server certs
   |    |-- WiFi RADIUS server certs
   |    |-- Internal web servers
   |
   +-- Document Signing CA
        |-- Faculty/staff signing certs
        |-- Official document certs

  Why 3 intermediate CAs:
  - Separation of concerns: compromise of one CA doesn't affect others
  - Different policies: email certs expire in 1 year, server certs in 90 days
  - Revocation scope: revoking Email CA doesn't break VPN

  2. Key Storage
  ==============
  Root CA:
    - Stored in FIPS 140-2 Level 3 HSM
    - Air-gapped machine, accessed only for intermediate CA signing
    - Ceremony requires 3-of-5 key custodians (Shamir's secret sharing)

  Intermediate CAs:
    - FIPS 140-2 Level 2 HSMs (network-attached for automated signing)
    - Located in university data center with physical access controls

  User Keys (Email, Document Signing):
    - Software keystore for most users (PKCS#12)
    - Smart cards (PIV) for high-security users (administrators, financial officers)
    - Faculty can optionally use hardware tokens (YubiKey)

  Server Keys:
    - Generated on the server, stored in filesystem with restricted permissions
    - Critical servers use HSMs

  3. Revocation Strategy
  ======================
  - OCSP Stapling for all server certificates
  - Short-lived certificates for WiFi (24-hour validity, auto-renewed)
  - CRL for user certificates (updated every 4 hours)
  - Must-Staple extension on all server certificates
  - Automated revocation when:
    * Employee leaves (integrated with HR system)
    * Student graduates
    * Device is decommissioned

  4. Certificate Lifecycle
  ========================
  Issuance:
    - Users: automated via university SSO + identity verification
    - Servers: automated via ACME-like protocol with internal CA
    - Faculty signing: manual approval by department head

  Renewal:
    - Automated for all certificates (ACME/EST protocol)
    - Email reminders at 30, 14, and 7 days before expiration
    - Auto-renewal if user's account is still active

  Validity Periods:
    - Root CA: 20 years
    - Intermediate CAs: 5 years
    - Server certs: 90 days (auto-renewed)
    - User email certs: 1 year
    - WiFi certs: 24 hours
    - Document signing: 2 years

  5. Disaster Recovery
  ====================
  Root CA Compromise:
    - Revoke compromised root (update all trust stores)
    - Issue new root from backup key shares (held by 5 custodians)
    - Re-issue all intermediate CA certificates
    - Re-issue all end-entity certificates (~50,000 user certs)
    - Estimated recovery time: 48-72 hours

  Intermediate CA Compromise:
    - Revoke compromised intermediate
    - Issue new intermediate from root
    - Re-issue only affected certificates (~16,000 for email CA)
    - Estimated recovery time: 4-8 hours

  Prevention:
    - Root CA key ceremony recorded on video
    - Key shares distributed geographically (different buildings)
    - HSM firmware updates require multi-person authorization
    - Annual key ceremony audit
    - CT logging for all issued certificates (internal CT log)
  """)


if __name__ == "__main__":
    print("=== Exercise 1: Certificate Inspection ===")
    exercise_1()

    print("\n=== Exercise 2: Self-Signed CA ===")
    exercise_2()

    print("\n=== Exercise 3: CT Log Analysis ===")
    exercise_3()

    print("\n=== Exercise 4: Revocation Trade-offs ===")
    exercise_4()

    print("\n=== Exercise 5: PKI Design Challenge ===")
    exercise_5()

    print("\nAll exercises completed!")
