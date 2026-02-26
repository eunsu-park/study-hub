# Security Fundamentals and Threat Modeling

**Next**: [02. Cryptography Basics](./02_Cryptography_Basics.md)

---

Security engineering is the discipline of building systems that remain dependable in the face of malice, error, and mischance. Before diving into specific technologies like cryptography or TLS, we need a shared vocabulary for reasoning about threats, risks, and defenses. This lesson establishes the conceptual foundations that every subsequent lesson builds upon.

**Difficulty**: ⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Understand the CIA Triad and extended security properties
2. Apply threat modeling methodologies (STRIDE, DREAD) to real systems
3. Identify attack surfaces and threat vectors in software architectures
4. Design layered defenses using the defense-in-depth principle
5. Apply the principle of least privilege in system design
6. Distinguish security by design from security by obscurity
7. Perform basic risk assessments using standard frameworks
8. Interpret Common Vulnerability Scoring System (CVSS) scores

---

## Table of Contents

1. [The CIA Triad](#1-the-cia-triad)
2. [Extended Security Properties](#2-extended-security-properties)
3. [Threat Modeling with STRIDE](#3-threat-modeling-with-stride)
4. [Risk Scoring with DREAD](#4-risk-scoring-with-dread)
5. [Attack Surfaces and Threat Vectors](#5-attack-surfaces-and-threat-vectors)
6. [Defense in Depth](#6-defense-in-depth)
7. [Principle of Least Privilege](#7-principle-of-least-privilege)
8. [Security by Design vs Security by Obscurity](#8-security-by-design-vs-security-by-obscurity)
9. [Risk Assessment Frameworks](#9-risk-assessment-frameworks)
10. [Common Vulnerability Scoring System (CVSS)](#10-common-vulnerability-scoring-system-cvss)
11. [Putting It All Together: A Worked Example](#11-putting-it-all-together-a-worked-example)
12. [Exercises](#12-exercises)
13. [References](#13-references)

---

## 1. The CIA Triad

The CIA Triad is the most fundamental model in information security. Every security control, every attack, and every risk can be analyzed through these three properties.

```
                         ┌───────────────────────┐
                         │   Confidentiality     │
                         │                       │
                         │   "Who can see it?"    │
                         └───────────┬───────────┘
                                     │
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
          ┌─────────────────┐              ┌─────────────────┐
          │    Integrity    │              │  Availability   │
          │                 │◀────────────▶│                 │
          │ "Can I trust    │              │ "Can I access   │
          │  it?"           │              │  it when I      │
          └─────────────────┘              │  need to?"      │
                                           └─────────────────┘
```

### 1.1 Confidentiality

Confidentiality ensures that information is accessible only to those authorized to have access. A breach of confidentiality means unauthorized disclosure of information.

**Threats to confidentiality:**
- Eavesdropping on network traffic (packet sniffing)
- Unauthorized access to databases
- Social engineering (phishing)
- Shoulder surfing, dumpster diving
- Insider threats

**Controls for confidentiality:**
- Encryption (at rest and in transit)
- Access control lists (ACLs)
- Authentication mechanisms
- Data classification and handling policies
- Physical security controls

```python
# Example: Demonstrating confidentiality through encryption
from cryptography.fernet import Fernet

# Generate a secret key (only authorized parties should have this)
key = Fernet.generate_key()
cipher = Fernet(key)

# Sensitive data
secret_message = b"Patient diagnosis: confidential medical record"

# Encrypt - now only key holders can read this
encrypted = cipher.encrypt(secret_message)
print(f"Encrypted: {encrypted[:50]}...")

# Decrypt - requires the key
decrypted = cipher.decrypt(encrypted)
print(f"Decrypted: {decrypted.decode()}")

# Without the key, the data is meaningless
wrong_key = Fernet.generate_key()
wrong_cipher = Fernet(wrong_key)
try:
    wrong_cipher.decrypt(encrypted)
except Exception as e:
    print(f"Unauthorized access denied: {type(e).__name__}")
```

### 1.2 Integrity

Integrity ensures that information has not been altered in an unauthorized manner. Data should be accurate, consistent, and trustworthy.

**Threats to integrity:**
- Man-in-the-middle (MITM) attacks
- SQL injection modifying database records
- Malware altering files
- Unauthorized configuration changes
- Bit-rot or storage corruption

**Controls for integrity:**
- Cryptographic hash functions (SHA-256)
- Digital signatures
- Message Authentication Codes (MAC/HMAC)
- Version control systems
- Database constraints and transactions

```python
import hashlib
import json

def compute_integrity_hash(data: dict) -> str:
    """Compute a SHA-256 hash of structured data for integrity verification."""
    # Canonical JSON encoding ensures consistent hashing
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()

# Original financial record
record = {
    "account": "ACCT-001",
    "amount": 1500.00,
    "currency": "USD",
    "timestamp": "2026-01-15T10:30:00Z"
}

original_hash = compute_integrity_hash(record)
print(f"Original hash: {original_hash}")

# Verify integrity - unchanged data
verified_hash = compute_integrity_hash(record)
print(f"Verification:  {verified_hash}")
print(f"Integrity OK:  {original_hash == verified_hash}")

# Tampered record - even a tiny change completely changes the hash
tampered = record.copy()
tampered["amount"] = 1500.01  # Changed by $0.01
tampered_hash = compute_integrity_hash(tampered)
print(f"\nTampered hash: {tampered_hash}")
print(f"Integrity OK:  {original_hash == tampered_hash}")  # False
```

### 1.3 Availability

Availability ensures that information and systems are accessible when needed by authorized users.

**Threats to availability:**
- Distributed Denial of Service (DDoS) attacks
- Hardware failures
- Software bugs and crashes
- Natural disasters
- Ransomware

**Controls for availability:**
- Redundancy and failover systems
- Load balancing
- Regular backups and disaster recovery plans
- DDoS mitigation services
- Capacity planning and monitoring

```python
# Example: Simple health check and availability monitor
import time
import random
from dataclasses import dataclass, field
from typing import List

@dataclass
class ServiceHealthCheck:
    """Monitor service availability."""
    name: str
    endpoints: List[str]
    check_history: List[dict] = field(default_factory=list)

    def check_endpoint(self, url: str) -> dict:
        """Simulate checking an endpoint's availability."""
        # In production, this would make actual HTTP requests
        is_up = random.random() > 0.1  # 90% uptime simulation
        latency_ms = random.uniform(10, 500) if is_up else None
        return {
            "url": url,
            "status": "UP" if is_up else "DOWN",
            "latency_ms": round(latency_ms, 2) if latency_ms else None,
            "timestamp": time.time()
        }

    def check_all(self) -> dict:
        """Check all endpoints and compute availability."""
        results = [self.check_endpoint(ep) for ep in self.endpoints]
        up_count = sum(1 for r in results if r["status"] == "UP")
        availability = up_count / len(results) * 100

        summary = {
            "service": self.name,
            "total_endpoints": len(results),
            "up": up_count,
            "down": len(results) - up_count,
            "availability_pct": round(availability, 2),
            "results": results
        }
        self.check_history.append(summary)
        return summary

# Define service endpoints
service = ServiceHealthCheck(
    name="Payment API",
    endpoints=[
        "https://api.example.com/v1/health",
        "https://api-backup.example.com/v1/health",
        "https://api-eu.example.com/v1/health",
    ]
)

# Run health check
result = service.check_all()
print(f"Service: {result['service']}")
print(f"Availability: {result['availability_pct']}%")
for r in result["results"]:
    status_icon = "[OK]" if r["status"] == "UP" else "[!!]"
    latency = f"{r['latency_ms']}ms" if r["latency_ms"] else "N/A"
    print(f"  {status_icon} {r['url']} - {latency}")
```

### 1.4 CIA Trade-offs

In practice, the three properties often tension against each other:

```
┌────────────────────────────────────────────────────────────────────┐
│                      CIA Trade-off Examples                        │
├──────────────────┬─────────────────────────────────────────────────┤
│ Scenario         │ Trade-off                                      │
├──────────────────┼─────────────────────────────────────────────────┤
│ Full-disk        │ Confidentiality ↑  Availability ↓              │
│ encryption       │ (Decryption adds latency; lost key = no data)  │
├──────────────────┼─────────────────────────────────────────────────┤
│ Database         │ Integrity ↑  Availability ↓                    │
│ replication      │ (Synchronous replication slows writes)         │
├──────────────────┼─────────────────────────────────────────────────┤
│ Public API       │ Availability ↑  Confidentiality ↓              │
│ with no auth     │ (Anyone can access; maximum availability)      │
├──────────────────┼─────────────────────────────────────────────────┤
│ Air-gapped       │ Confidentiality ↑  Availability ↓              │
│ network          │ (Very secure but hard to access remotely)      │
└──────────────────┴─────────────────────────────────────────────────┘
```

---

## 2. Extended Security Properties

Beyond the CIA Triad, several additional properties are essential for a complete security model.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Extended Security Properties                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Authentication        "Are you who you claim to be?"               │
│  ├── Something you know (password, PIN)                             │
│  ├── Something you have (token, smart card, phone)                  │
│  └── Something you are  (fingerprint, face, iris)                   │
│                                                                     │
│  Authorization         "What are you allowed to do?"                │
│  ├── Role-Based Access Control (RBAC)                               │
│  ├── Attribute-Based Access Control (ABAC)                          │
│  └── Mandatory Access Control (MAC)                                 │
│                                                                     │
│  Non-repudiation       "Can you deny you did this?"                 │
│  ├── Digital signatures                                             │
│  ├── Audit logs with tamper-evident storage                         │
│  └── Blockchain-based records                                       │
│                                                                     │
│  Accountability        "Can we trace actions to actors?"            │
│  ├── Comprehensive logging                                          │
│  ├── Session tracking                                               │
│  └── Forensic analysis capabilities                                 │
│                                                                     │
│  Privacy               "Is personal data properly protected?"       │
│  ├── Data minimization                                              │
│  ├── Purpose limitation                                             │
│  └── Consent management (GDPR, CCPA)                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 Authentication Factors

Multi-factor authentication (MFA) combines two or more independent factors:

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import hashlib
import secrets
import time

class AuthFactor(Enum):
    KNOWLEDGE = "something_you_know"    # Password, PIN
    POSSESSION = "something_you_have"   # TOTP token, smart card
    INHERENCE = "something_you_are"     # Biometric

@dataclass
class AuthenticationAttempt:
    username: str
    factors_provided: list
    timestamp: float = 0.0
    ip_address: str = ""

    def __post_init__(self):
        self.timestamp = time.time()

class SimpleAuthenticator:
    """Demonstrates multi-factor authentication concepts."""

    def __init__(self):
        self.users = {}
        self.failed_attempts = {}
        self.max_attempts = 5
        self.lockout_duration = 300  # 5 minutes

    def register_user(self, username: str, password: str):
        """Register with salted password hash."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256(
            (salt + password).encode()
        ).hexdigest()
        self.users[username] = {
            "salt": salt,
            "password_hash": password_hash,
            "totp_secret": secrets.token_hex(20),  # For MFA
        }

    def verify_password(self, username: str, password: str) -> bool:
        """Verify knowledge factor (password)."""
        if username not in self.users:
            # Constant-time comparison to prevent user enumeration
            hashlib.sha256(b"dummy_computation").hexdigest()
            return False

        user = self.users[username]
        computed = hashlib.sha256(
            (user["salt"] + password).encode()
        ).hexdigest()
        return secrets.compare_digest(computed, user["password_hash"])

    def is_locked_out(self, username: str) -> bool:
        """Check if account is locked due to failed attempts."""
        if username not in self.failed_attempts:
            return False
        attempts = self.failed_attempts[username]
        if len(attempts) >= self.max_attempts:
            last_attempt = attempts[-1]
            if time.time() - last_attempt < self.lockout_duration:
                return True
            # Lockout expired, reset
            self.failed_attempts[username] = []
        return False

    def authenticate(self, attempt: AuthenticationAttempt) -> dict:
        """Process authentication attempt with multiple factors."""
        if self.is_locked_out(attempt.username):
            return {
                "success": False,
                "reason": "Account locked. Try again later.",
                "factors_verified": []
            }

        factors_verified = []

        for factor in attempt.factors_provided:
            if factor["type"] == AuthFactor.KNOWLEDGE:
                if self.verify_password(attempt.username, factor["value"]):
                    factors_verified.append(AuthFactor.KNOWLEDGE)
            elif factor["type"] == AuthFactor.POSSESSION:
                # In production: verify TOTP code
                factors_verified.append(AuthFactor.POSSESSION)
            elif factor["type"] == AuthFactor.INHERENCE:
                # In production: verify biometric
                factors_verified.append(AuthFactor.INHERENCE)

        success = len(factors_verified) >= 2  # Require at least 2 factors

        if not success:
            self.failed_attempts.setdefault(attempt.username, []).append(
                time.time()
            )

        return {
            "success": success,
            "reason": "MFA verified" if success else "Insufficient factors",
            "factors_verified": [f.value for f in factors_verified]
        }

# Usage
auth = SimpleAuthenticator()
auth.register_user("alice", "correct-horse-battery-staple")

# Single factor - insufficient
attempt_1fa = AuthenticationAttempt(
    username="alice",
    factors_provided=[
        {"type": AuthFactor.KNOWLEDGE, "value": "correct-horse-battery-staple"}
    ],
    ip_address="192.168.1.100"
)
result = auth.authenticate(attempt_1fa)
print(f"1FA Result: {result}")
# {'success': False, 'reason': 'Insufficient factors', ...}

# Two factors - success
attempt_2fa = AuthenticationAttempt(
    username="alice",
    factors_provided=[
        {"type": AuthFactor.KNOWLEDGE, "value": "correct-horse-battery-staple"},
        {"type": AuthFactor.POSSESSION, "value": "123456"}  # TOTP code
    ],
    ip_address="192.168.1.100"
)
result = auth.authenticate(attempt_2fa)
print(f"2FA Result: {result}")
# {'success': True, 'reason': 'MFA verified', ...}
```

---

## 3. Threat Modeling with STRIDE

STRIDE is a threat modeling framework developed at Microsoft. Each letter identifies a category of threat that maps to a specific security property violation.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        STRIDE Threat Model                           │
├────────────────┬──────────────────────┬──────────────────────────────┤
│ Threat         │ Violates             │ Example                      │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ S - Spoofing   │ Authentication       │ Fake login page,             │
│                │                      │ forged JWT token             │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ T - Tampering  │ Integrity            │ Modified API request,        │
│                │                      │ SQL injection                │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ R - Repudiation│ Non-repudiation      │ Denying a transaction,       │
│                │                      │ deleting logs                │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ I - Information│ Confidentiality      │ Data leak, packet            │
│   Disclosure   │                      │ sniffing, error messages     │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ D - Denial of  │ Availability         │ DDoS, resource               │
│   Service      │                      │ exhaustion, crash bugs       │
├────────────────┼──────────────────────┼──────────────────────────────┤
│ E - Elevation  │ Authorization        │ Privilege escalation,        │
│   of Privilege │                      │ path traversal               │
└────────────────┴──────────────────────┴──────────────────────────────┘
```

### 3.1 STRIDE Analysis Process

```
Step 1: Decompose the system
         │
         ▼
Step 2: Create a Data Flow Diagram (DFD)
         │
         ▼
Step 3: Identify threats using STRIDE per element
         │
         ▼
Step 4: Rate and prioritize threats
         │
         ▼
Step 5: Plan mitigations
```

### 3.2 Data Flow Diagram Elements

```
┌─────────────────────────────────────────────────────────────────┐
│                    DFD Element Types                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────┐    External Entity (user, external system)         │
│  │ Entity  │    - Outside your trust boundary                   │
│  └─────────┘    - Source or destination of data                 │
│                                                                 │
│  ┌─────────┐    Process (code that transforms data)             │
│  │(Process)│    - Your application logic                        │
│  └─────────┘    - APIs, services, functions                     │
│                                                                 │
│  ═══════════    Data Store (database, file, cache)              │
│  ║ Store   ║    - Where data persists                           │
│  ═══════════    - Databases, files, queues                      │
│                                                                 │
│  ──────────▶    Data Flow (data moving between elements)        │
│                 - HTTP requests, API calls, file reads           │
│                                                                 │
│  - - - - - -    Trust Boundary                                  │
│  |          |   - Separates different trust levels               │
│  - - - - - -    - Network boundary, process boundary            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Example: Threat Model for a Web Application

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Web Application DFD                                  │
│                                                                      │
│   ┌──────────┐         HTTPS          ┌──────────────┐              │
│   │  Browser │ ──────────────────────▶│  Web Server  │              │
│   │  (User)  │ ◀──────────────────────│  (Nginx)     │              │
│   └──────────┘                        └──────┬───────┘              │
│                                               │                      │
│   - - - - - - - - - - - - - - - - - - - - - -│- - - - - - - - - -  │
│   │         Trust Boundary (DMZ → Internal)   │                  │  │
│   - - - - - - - - - - - - - - - - - - - - - -│- - - - - - - - - -  │
│                                               │                      │
│                                        ┌──────▼───────┐             │
│                                        │  App Server  │             │
│                                        │  (Flask/     │             │
│                                        │   Django)    │             │
│                                        └──────┬───────┘             │
│                                               │                      │
│                               ┌───────────────┼───────────────┐     │
│                               │               │               │     │
│                        ═══════▼═══════ ═══════▼═══════ ┌──────▼──┐  │
│                        ║  PostgreSQL ║ ║   Redis     ║ │ External│  │
│                        ║  (Users,   ║ ║   (Sessions,║ │ Payment │  │
│                        ║   Orders)  ║ ║   Cache)    ║ │ API     │  │
│                        ═══════════════ ═══════════════ └─────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

Now apply STRIDE to each element:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Threat:
    category: str       # S, T, R, I, D, or E
    element: str        # Which DFD element
    description: str    # What could go wrong
    severity: str       # Critical, High, Medium, Low
    mitigation: str     # How to prevent it

def stride_analysis(system_name: str) -> List[Threat]:
    """Example STRIDE analysis for a web application."""
    threats = [
        # Spoofing threats
        Threat(
            category="Spoofing",
            element="Browser → Web Server",
            description="Attacker impersonates a legitimate user with "
                        "stolen credentials or forged session tokens",
            severity="High",
            mitigation="Implement MFA, use short-lived JWTs, "
                       "enforce strong password policies"
        ),
        Threat(
            category="Spoofing",
            element="App Server → Payment API",
            description="Attacker intercepts and replays API calls "
                        "to the payment processor",
            severity="Critical",
            mitigation="Use mutual TLS (mTLS), API key rotation, "
                       "request signing with timestamps"
        ),

        # Tampering threats
        Threat(
            category="Tampering",
            element="Browser → Web Server",
            description="Attacker modifies request parameters to change "
                        "order amounts or access other users' data",
            severity="High",
            mitigation="Server-side validation, HMAC on critical parameters, "
                       "parameterized queries"
        ),
        Threat(
            category="Tampering",
            element="PostgreSQL",
            description="SQL injection allows attacker to modify database "
                        "records directly",
            severity="Critical",
            mitigation="Prepared statements, ORM usage, input sanitization, "
                       "database-level access controls"
        ),

        # Repudiation threats
        Threat(
            category="Repudiation",
            element="App Server",
            description="User denies making a purchase; no audit trail exists",
            severity="Medium",
            mitigation="Comprehensive audit logging with timestamps, "
                       "digital signatures on transactions"
        ),

        # Information Disclosure threats
        Threat(
            category="Information Disclosure",
            element="Web Server",
            description="Verbose error messages reveal stack traces, "
                        "database schema, or internal IPs",
            severity="Medium",
            mitigation="Custom error pages, structured logging "
                       "(not exposed to clients), security headers"
        ),
        Threat(
            category="Information Disclosure",
            element="Redis",
            description="Session data stored in Redis without encryption "
                        "is readable if Redis is compromised",
            severity="High",
            mitigation="Encrypt session data, enable Redis AUTH, "
                       "network isolation for Redis"
        ),

        # Denial of Service threats
        Threat(
            category="Denial of Service",
            element="Web Server",
            description="Volumetric DDoS attack overwhelms the web server",
            severity="High",
            mitigation="CDN/WAF (e.g., Cloudflare), rate limiting, "
                       "auto-scaling infrastructure"
        ),

        # Elevation of Privilege threats
        Threat(
            category="Elevation of Privilege",
            element="App Server",
            description="Regular user accesses admin endpoints due to "
                        "missing authorization checks (IDOR/broken access control)",
            severity="Critical",
            mitigation="RBAC enforcement on every endpoint, "
                       "automated authorization testing"
        ),
    ]
    return threats

# Run analysis
threats = stride_analysis("E-Commerce Web App")
print(f"STRIDE Analysis: {len(threats)} threats identified\n")

# Summary by category
from collections import Counter
by_category = Counter(t.category for t in threats)
for cat, count in sorted(by_category.items()):
    print(f"  {cat}: {count} threats")

print()

# Critical threats
critical = [t for t in threats if t.severity == "Critical"]
print(f"Critical threats ({len(critical)}):")
for t in critical:
    print(f"  [{t.category}] {t.element}")
    print(f"    Risk: {t.description}")
    print(f"    Fix:  {t.mitigation}")
    print()
```

---

## 4. Risk Scoring with DREAD

DREAD provides a numerical scoring system for prioritizing threats. Each factor is rated 1-10.

```
┌───────────────────────────────────────────────────────────────────┐
│                      DREAD Scoring Model                          │
├──────────────────┬────────────────────────────────────────────────┤
│ Factor           │ Question                                      │
├──────────────────┼────────────────────────────────────────────────┤
│ D - Damage       │ How bad would an attack be?                   │
│                  │ 1 = trivial  10 = complete system compromise  │
├──────────────────┼────────────────────────────────────────────────┤
│ R - Reproducibil.│ How easy is it to reproduce?                  │
│                  │ 1 = very hard  10 = always reproducible       │
├──────────────────┼────────────────────────────────────────────────┤
│ E - Exploitab.   │ How much effort to launch the attack?         │
│                  │ 1 = expert + custom tools  10 = browser only  │
├──────────────────┼────────────────────────────────────────────────┤
│ A - Affected     │ How many users are impacted?                  │
│   Users          │ 1 = single user  10 = all users               │
├──────────────────┼────────────────────────────────────────────────┤
│ D - Discoverab.  │ How easy is it to find the vulnerability?     │
│                  │ 1 = requires source code  10 = obvious in URL │
└──────────────────┴────────────────────────────────────────────────┘

DREAD Score = (D + R + E + A + D) / 5

Rating:    1-3 = Low    4-6 = Medium    7-9 = High    10 = Critical
```

```python
from dataclasses import dataclass

@dataclass
class DREADScore:
    """DREAD risk scoring for a vulnerability."""
    vulnerability: str
    damage: int          # 1-10
    reproducibility: int # 1-10
    exploitability: int  # 1-10
    affected_users: int  # 1-10
    discoverability: int # 1-10

    @property
    def score(self) -> float:
        total = (self.damage + self.reproducibility +
                 self.exploitability + self.affected_users +
                 self.discoverability)
        return total / 5

    @property
    def rating(self) -> str:
        s = self.score
        if s >= 9:
            return "CRITICAL"
        elif s >= 7:
            return "HIGH"
        elif s >= 4:
            return "MEDIUM"
        else:
            return "LOW"

    def __str__(self):
        return (
            f"{self.vulnerability}\n"
            f"  D={self.damage} R={self.reproducibility} "
            f"E={self.exploitability} A={self.affected_users} "
            f"D={self.discoverability}\n"
            f"  Score: {self.score:.1f} ({self.rating})"
        )

# Score several vulnerabilities
vulns = [
    DREADScore(
        vulnerability="SQL Injection in login form",
        damage=9, reproducibility=8, exploitability=7,
        affected_users=10, discoverability=8
    ),
    DREADScore(
        vulnerability="Missing rate limiting on password reset",
        damage=5, reproducibility=10, exploitability=9,
        affected_users=6, discoverability=7
    ),
    DREADScore(
        vulnerability="XSS in user profile bio field",
        damage=6, reproducibility=9, exploitability=6,
        affected_users=4, discoverability=5
    ),
    DREADScore(
        vulnerability="Cleartext password in debug log",
        damage=8, reproducibility=3, exploitability=3,
        affected_users=2, discoverability=2
    ),
]

# Sort by score descending
vulns.sort(key=lambda v: v.score, reverse=True)

print("DREAD Vulnerability Ranking")
print("=" * 50)
for i, v in enumerate(vulns, 1):
    print(f"\n#{i}: {v}")
```

---

## 5. Attack Surfaces and Threat Vectors

### 5.1 What Is an Attack Surface?

The attack surface is the sum of all points where an attacker can try to enter or extract data from a system.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Attack Surface Categories                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Network Attack Surface                                             │
│  ├── Open ports and services (SSH, HTTP, SMTP)                      │
│  ├── API endpoints                                                  │
│  ├── DNS configuration                                              │
│  └── Network protocols (TCP, UDP, ICMP)                             │
│                                                                     │
│  Software Attack Surface                                            │
│  ├── Web application inputs (forms, URLs, headers, cookies)         │
│  ├── File upload functionality                                      │
│  ├── Third-party dependencies (npm, pip packages)                   │
│  ├── Database interfaces                                            │
│  └── IPC mechanisms (pipes, sockets, shared memory)                 │
│                                                                     │
│  Physical Attack Surface                                            │
│  ├── USB ports on servers                                           │
│  ├── Physical access to data centers                                │
│  ├── Removable media                                                │
│  └── Hardware implants                                              │
│                                                                     │
│  Human Attack Surface                                               │
│  ├── Phishing targets (employees with access)                       │
│  ├── Social engineering (help desk, support)                        │
│  ├── Insider threats                                                │
│  └── Third-party vendors and contractors                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Attack Surface Enumeration

```python
import json
from dataclasses import dataclass, field
from typing import List, Dict
from enum import Enum

class SurfaceType(Enum):
    NETWORK = "network"
    SOFTWARE = "software"
    PHYSICAL = "physical"
    HUMAN = "human"

class RiskLevel(Enum):
    CRITICAL = 4
    HIGH = 3
    MEDIUM = 2
    LOW = 1

@dataclass
class AttackSurfaceEntry:
    name: str
    surface_type: SurfaceType
    description: str
    risk_level: RiskLevel
    exposed_to: str           # Who can reach this?
    current_controls: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class AttackSurfaceReport:
    system_name: str
    entries: List[AttackSurfaceEntry] = field(default_factory=list)

    def add(self, entry: AttackSurfaceEntry):
        self.entries.append(entry)

    def summary(self) -> Dict:
        by_type = {}
        by_risk = {}
        for entry in self.entries:
            by_type[entry.surface_type.value] = (
                by_type.get(entry.surface_type.value, 0) + 1
            )
            by_risk[entry.risk_level.name] = (
                by_risk.get(entry.risk_level.name, 0) + 1
            )
        return {"by_type": by_type, "by_risk": by_risk, "total": len(self.entries)}

    def print_report(self):
        print(f"Attack Surface Report: {self.system_name}")
        print("=" * 60)

        summary = self.summary()
        print(f"\nTotal attack surface entries: {summary['total']}")
        print(f"By type: {json.dumps(summary['by_type'], indent=2)}")
        print(f"By risk: {json.dumps(summary['by_risk'], indent=2)}")

        # Show critical/high items
        urgent = [e for e in self.entries
                  if e.risk_level.value >= RiskLevel.HIGH.value]
        if urgent:
            print(f"\nHigh/Critical Items ({len(urgent)}):")
            for e in sorted(urgent, key=lambda x: x.risk_level.value, reverse=True):
                print(f"  [{e.risk_level.name}] {e.name} ({e.surface_type.value})")
                print(f"    {e.description}")
                if e.recommendations:
                    print(f"    Recommendation: {e.recommendations[0]}")

# Build an attack surface report
report = AttackSurfaceReport("Corporate Web Portal")

report.add(AttackSurfaceEntry(
    name="Public API (port 443)",
    surface_type=SurfaceType.NETWORK,
    description="REST API serving 50+ endpoints to internet",
    risk_level=RiskLevel.HIGH,
    exposed_to="Internet",
    current_controls=["WAF", "Rate limiting", "JWT auth"],
    recommendations=["Add API gateway with stricter rate limits"]
))

report.add(AttackSurfaceEntry(
    name="Admin panel (/admin)",
    surface_type=SurfaceType.SOFTWARE,
    description="Django admin interface accessible on same domain",
    risk_level=RiskLevel.CRITICAL,
    exposed_to="Internet (by misconfiguration)",
    current_controls=["Username/password"],
    recommendations=["Restrict to VPN only", "Add MFA", "Separate domain"]
))

report.add(AttackSurfaceEntry(
    name="npm dependencies (347 packages)",
    surface_type=SurfaceType.SOFTWARE,
    description="Third-party JavaScript packages in frontend build",
    risk_level=RiskLevel.MEDIUM,
    exposed_to="Supply chain",
    current_controls=["npm audit in CI"],
    recommendations=["Pin dependency versions", "Use lockfile", "Add Snyk/Dependabot"]
))

report.add(AttackSurfaceEntry(
    name="Employee email accounts",
    surface_type=SurfaceType.HUMAN,
    description="150 employees with corporate email, potential phishing targets",
    risk_level=RiskLevel.HIGH,
    exposed_to="Internet (email)",
    current_controls=["Spam filter", "Annual security training"],
    recommendations=["Quarterly phishing simulations", "DMARC/DKIM enforcement"]
))

report.print_report()
```

### 5.3 Common Threat Vectors

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Common Threat Vectors                           │
├──────────────────┬──────────────────────────────────────────────────┤
│ Vector           │ Description and Examples                         │
├──────────────────┼──────────────────────────────────────────────────┤
│ Network-based    │ • Port scanning and exploitation                 │
│                  │ • Man-in-the-middle attacks                      │
│                  │ • DNS spoofing/poisoning                         │
│                  │ • ARP spoofing                                   │
├──────────────────┼──────────────────────────────────────────────────┤
│ Web application  │ • SQL injection (SQLi)                           │
│                  │ • Cross-site scripting (XSS)                     │
│                  │ • Cross-site request forgery (CSRF)              │
│                  │ • Server-side request forgery (SSRF)             │
│                  │ • Insecure direct object references (IDOR)       │
├──────────────────┼──────────────────────────────────────────────────┤
│ Supply chain     │ • Compromised dependencies (SolarWinds, log4j)  │
│                  │ • Typosquatting in package registries            │
│                  │ • Malicious code in build pipelines              │
├──────────────────┼──────────────────────────────────────────────────┤
│ Social engineer. │ • Phishing (email, SMS, voice)                   │
│                  │ • Pretexting and baiting                         │
│                  │ • Tailgating / piggybacking                     │
├──────────────────┼──────────────────────────────────────────────────┤
│ Physical         │ • USB drop attacks                               │
│                  │ • Evil maid attacks                              │
│                  │ • Hardware implants                              │
├──────────────────┼──────────────────────────────────────────────────┤
│ Insider          │ • Disgruntled employees                          │
│                  │ • Accidental data exposure                      │
│                  │ • Credential sharing                            │
└──────────────────┴──────────────────────────────────────────────────┘
```

---

## 6. Defense in Depth

Defense in depth is a strategy that employs multiple layers of security controls throughout an information system. If one layer fails, subsequent layers continue to provide protection.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Defense in Depth Layers                            │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  Layer 7: DATA                                               │    │
│  │  Encryption at rest, DLP, data classification, backups       │    │
│  │  ┌─────────────────────────────────────────────────────┐    │    │
│  │  │  Layer 6: APPLICATION                                │    │    │
│  │  │  Input validation, WAF, secure coding, SAST/DAST     │    │    │
│  │  │  ┌─────────────────────────────────────────────┐    │    │    │
│  │  │  │  Layer 5: HOST                               │    │    │    │
│  │  │  │  OS hardening, AV/EDR, patching, FIM         │    │    │    │
│  │  │  │  ┌─────────────────────────────────────┐    │    │    │    │
│  │  │  │  │  Layer 4: INTERNAL NETWORK           │    │    │    │    │
│  │  │  │  │  Segmentation, VLAN, IDS/IPS, NTA    │    │    │    │    │
│  │  │  │  │  ┌─────────────────────────────┐    │    │    │    │    │
│  │  │  │  │  │  Layer 3: PERIMETER          │    │    │    │    │    │
│  │  │  │  │  │  Firewalls, DMZ, VPN, proxy  │    │    │    │    │    │
│  │  │  │  │  │  ┌─────────────────────┐    │    │    │    │    │    │
│  │  │  │  │  │  │  Layer 2: PHYSICAL   │    │    │    │    │    │    │
│  │  │  │  │  │  │  Guards, locks, CCTV │    │    │    │    │    │    │
│  │  │  │  │  │  │  ┌─────────────┐    │    │    │    │    │    │    │
│  │  │  │  │  │  │  │ Layer 1:    │    │    │    │    │    │    │    │
│  │  │  │  │  │  │  │ POLICIES    │    │    │    │    │    │    │    │
│  │  │  │  │  │  │  │ & TRAINING  │    │    │    │    │    │    │    │
│  │  │  │  │  │  │  └─────────────┘    │    │    │    │    │    │    │
│  │  │  │  │  │  └─────────────────────┘    │    │    │    │    │    │
│  │  │  │  │  └─────────────────────────────┘    │    │    │    │    │
│  │  │  │  └─────────────────────────────────────┘    │    │    │    │
│  │  │  └─────────────────────────────────────────────┘    │    │    │
│  │  └─────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.1 Implementing Layered Security

```python
from dataclasses import dataclass, field
from typing import List, Callable
from enum import Enum
import re

class SecurityAction(Enum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    LOG = "LOG"
    ALERT = "ALERT"

@dataclass
class SecurityEvent:
    source_ip: str
    request_path: str
    method: str
    headers: dict
    body: str
    user: str = ""
    blocked_by: str = ""
    actions_taken: List[str] = field(default_factory=list)

@dataclass
class SecurityLayer:
    """A single layer in the defense-in-depth chain."""
    name: str
    order: int
    check: Callable  # Function that returns (SecurityAction, reason)

class DefenseInDepth:
    """Chain of security layers processed in order."""

    def __init__(self):
        self.layers: List[SecurityLayer] = []

    def add_layer(self, layer: SecurityLayer):
        self.layers.append(layer)
        self.layers.sort(key=lambda l: l.order)

    def process(self, event: SecurityEvent) -> SecurityEvent:
        """Process an event through all security layers."""
        for layer in self.layers:
            action, reason = layer.check(event)
            event.actions_taken.append(f"[{layer.name}] {action.value}: {reason}")

            if action == SecurityAction.BLOCK:
                event.blocked_by = layer.name
                return event

        return event

# Define security layers as check functions

def rate_limiter(event: SecurityEvent):
    """Layer 1: Rate limiting (perimeter)."""
    # Simplified - in production, track request counts per IP
    suspicious_ips = {"10.0.0.99", "192.168.1.200"}
    if event.source_ip in suspicious_ips:
        return SecurityAction.BLOCK, f"IP {event.source_ip} rate limited"
    return SecurityAction.ALLOW, "Rate check passed"

def waf_check(event: SecurityEvent):
    """Layer 2: Web Application Firewall."""
    # Check for common attack patterns
    sqli_patterns = [
        r"(\b(union|select|insert|update|delete|drop)\b.*\b(from|into|table)\b)",
        r"(--|#|/\*)",
        r"(\bor\b\s+\b\d+\b\s*=\s*\b\d+\b)",
    ]
    xss_patterns = [
        r"<script[^>]*>",
        r"javascript:",
        r"on(load|error|click)\s*=",
    ]

    full_input = f"{event.request_path} {event.body}".lower()

    for pattern in sqli_patterns:
        if re.search(pattern, full_input, re.IGNORECASE):
            return SecurityAction.BLOCK, f"SQLi pattern detected: {pattern}"

    for pattern in xss_patterns:
        if re.search(pattern, full_input, re.IGNORECASE):
            return SecurityAction.BLOCK, f"XSS pattern detected: {pattern}"

    return SecurityAction.ALLOW, "WAF check passed"

def auth_check(event: SecurityEvent):
    """Layer 3: Authentication verification."""
    auth_header = event.headers.get("Authorization", "")
    if not auth_header:
        if event.request_path.startswith("/api/"):
            return SecurityAction.BLOCK, "Missing authentication for API endpoint"
    return SecurityAction.ALLOW, "Auth check passed"

def input_validation(event: SecurityEvent):
    """Layer 4: Application-level input validation."""
    if len(event.body) > 10000:
        return SecurityAction.BLOCK, "Request body exceeds maximum size"
    if "../" in event.request_path:
        return SecurityAction.BLOCK, "Path traversal attempt detected"
    return SecurityAction.ALLOW, "Input validation passed"

# Assemble defense layers
defense = DefenseInDepth()
defense.add_layer(SecurityLayer("Rate Limiter", 1, rate_limiter))
defense.add_layer(SecurityLayer("WAF", 2, waf_check))
defense.add_layer(SecurityLayer("Auth Check", 3, auth_check))
defense.add_layer(SecurityLayer("Input Validation", 4, input_validation))

# Test with various events
events = [
    SecurityEvent(
        source_ip="203.0.113.50",
        request_path="/api/users/1",
        method="GET",
        headers={"Authorization": "Bearer eyJ..."},
        body=""
    ),
    SecurityEvent(
        source_ip="203.0.113.51",
        request_path="/api/users",
        method="POST",
        headers={"Authorization": "Bearer eyJ..."},
        body="name=test' OR 1=1 --"
    ),
    SecurityEvent(
        source_ip="203.0.113.52",
        request_path="/api/files/../../../etc/passwd",
        method="GET",
        headers={"Authorization": "Bearer eyJ..."},
        body=""
    ),
]

for i, event in enumerate(events, 1):
    result = defense.process(event)
    status = "BLOCKED" if result.blocked_by else "ALLOWED"
    print(f"\nEvent #{i}: {result.method} {result.request_path} -> {status}")
    for action in result.actions_taken:
        print(f"  {action}")
    if result.blocked_by:
        print(f"  >>> Blocked by: {result.blocked_by}")
```

---

## 7. Principle of Least Privilege

The principle of least privilege (PoLP) states that every module, user, or process should operate using only the minimum set of privileges necessary to complete its task.

```
┌─────────────────────────────────────────────────────────────────────┐
│              Principle of Least Privilege - Examples                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  BAD (Over-privileged):                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Web App Service Account                                     │   │
│  │  ├── DB: root access (can DROP DATABASE)                     │   │
│  │  ├── Files: read/write to entire filesystem                  │   │
│  │  ├── Network: unrestricted outbound                          │   │
│  │  └── Processes: can spawn anything as root                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  GOOD (Least privilege):                                             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Web App Service Account                                     │   │
│  │  ├── DB: SELECT, INSERT on app_db.users, app_db.orders only  │   │
│  │  ├── Files: read-only /app/static, write /app/uploads only   │   │
│  │  ├── Network: outbound only to payment-api.internal:443      │   │
│  │  └── Processes: runs as unprivileged user (uid 1000)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.1 Least Privilege in Database Access

```sql
-- BAD: Application uses root/superuser
-- GRANT ALL PRIVILEGES ON *.* TO 'webapp'@'%';

-- GOOD: Granular permissions per function
-- Read-only user for reporting
CREATE USER 'webapp_reader'@'10.0.0.%' IDENTIFIED BY 'strong_random_pw';
GRANT SELECT ON app_db.users TO 'webapp_reader'@'10.0.0.%';
GRANT SELECT ON app_db.orders TO 'webapp_reader'@'10.0.0.%';
GRANT SELECT ON app_db.products TO 'webapp_reader'@'10.0.0.%';

-- Write user for the application (limited tables and operations)
CREATE USER 'webapp_writer'@'10.0.0.%' IDENTIFIED BY 'another_strong_pw';
GRANT SELECT, INSERT, UPDATE ON app_db.users TO 'webapp_writer'@'10.0.0.%';
GRANT SELECT, INSERT, UPDATE ON app_db.orders TO 'webapp_writer'@'10.0.0.%';
GRANT SELECT ON app_db.products TO 'webapp_writer'@'10.0.0.%';
-- Note: No DELETE, no DROP, no ALTER, no access to other databases

-- Admin user (separate account, not used by the application)
CREATE USER 'db_admin'@'10.0.0.1' IDENTIFIED BY 'very_strong_admin_pw';
GRANT ALL PRIVILEGES ON app_db.* TO 'db_admin'@'10.0.0.1';
-- Only accessible from a single bastion host IP
```

### 7.2 Least Privilege in Code

```python
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

@dataclass
class SandboxedFileAccess:
    """Enforce least privilege for file system access."""
    allowed_read_dirs: list
    allowed_write_dirs: list
    max_file_size: int = 10 * 1024 * 1024  # 10 MB default

    def _is_within_allowed_dirs(self, path: Path, dirs: list) -> bool:
        """Check if a resolved path falls within allowed directories."""
        resolved = path.resolve()
        return any(
            str(resolved).startswith(str(Path(d).resolve()))
            for d in dirs
        )

    def read_file(self, filepath: str) -> Optional[str]:
        """Read a file only if it's within allowed read directories."""
        path = Path(filepath)

        # Prevent path traversal
        if ".." in path.parts:
            raise PermissionError(f"Path traversal not allowed: {filepath}")

        if not self._is_within_allowed_dirs(path, self.allowed_read_dirs):
            raise PermissionError(
                f"Read access denied: {filepath} is outside allowed directories"
            )

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        return path.read_text()

    def write_file(self, filepath: str, content: str) -> bool:
        """Write a file only if it's within allowed write directories."""
        path = Path(filepath)

        if ".." in path.parts:
            raise PermissionError(f"Path traversal not allowed: {filepath}")

        if not self._is_within_allowed_dirs(path, self.allowed_write_dirs):
            raise PermissionError(
                f"Write access denied: {filepath} is outside allowed directories"
            )

        if len(content.encode()) > self.max_file_size:
            raise ValueError(
                f"File size {len(content.encode())} exceeds max {self.max_file_size}"
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return True

# Usage: application can only read from /app/config and write to /app/uploads
sandbox = SandboxedFileAccess(
    allowed_read_dirs=["/app/config", "/app/static"],
    allowed_write_dirs=["/app/uploads"],
    max_file_size=5 * 1024 * 1024  # 5 MB
)

# This would succeed (within allowed read directory)
# content = sandbox.read_file("/app/config/settings.json")

# This would raise PermissionError (outside allowed directories)
try:
    sandbox.read_file("/etc/passwd")
except PermissionError as e:
    print(f"Blocked: {e}")

# This would raise PermissionError (path traversal attempt)
try:
    sandbox.read_file("/app/config/../../etc/shadow")
except PermissionError as e:
    print(f"Blocked: {e}")
```

### 7.3 Least Privilege in Cloud (AWS IAM Example)

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowS3ReadSpecificBucket",
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::my-app-uploads",
                "arn:aws:s3:::my-app-uploads/*"
            ]
        },
        {
            "Sid": "AllowDynamoDBReadWrite",
            "Effect": "Allow",
            "Action": [
                "dynamodb:GetItem",
                "dynamodb:PutItem",
                "dynamodb:Query"
            ],
            "Resource": "arn:aws:dynamodb:us-east-1:123456789:table/users"
        },
        {
            "Sid": "DenyAllElse",
            "Effect": "Deny",
            "Action": "*",
            "NotResource": [
                "arn:aws:s3:::my-app-uploads",
                "arn:aws:s3:::my-app-uploads/*",
                "arn:aws:dynamodb:us-east-1:123456789:table/users"
            ]
        }
    ]
}
```

---

## 8. Security by Design vs Security by Obscurity

### 8.1 Security by Design

Security by design means building security into the system from the start, rather than adding it as an afterthought.

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Security by Design Principles                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Minimize Attack Surface                                          │
│     └── Remove unnecessary features, ports, services                 │
│                                                                      │
│  2. Establish Secure Defaults                                        │
│     └── Default configuration should be the most secure              │
│                                                                      │
│  3. Principle of Least Privilege                                     │
│     └── Grant minimum access needed                                  │
│                                                                      │
│  4. Defense in Depth                                                 │
│     └── Multiple independent layers of security                      │
│                                                                      │
│  5. Fail Securely                                                    │
│     └── Errors should not create security holes                      │
│                                                                      │
│  6. Don't Trust Services                                             │
│     └── Validate all external input and responses                    │
│                                                                      │
│  7. Separation of Duties                                             │
│     └── No single person/process controls everything                 │
│                                                                      │
│  8. Keep Security Simple                                             │
│     └── Complex security is often bypassed or misconfigured          │
│                                                                      │
│  9. Fix Issues Correctly                                             │
│     └── Address root causes, not symptoms                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Kerckhoffs' Principle (1883):** A cryptographic system should be secure even if everything about the system, except the key, is public knowledge. This is the gold standard for security by design.

```python
# Security by Design: Fail-secure examples

def get_user_role(user_id: int, db) -> str:
    """
    Fail-secure: if anything goes wrong, return the most restrictive role.
    Never fail into a privileged state.
    """
    try:
        user = db.get_user(user_id)
        if user and user.role:
            return user.role
    except Exception:
        pass  # Log this in production

    # Fail secure: default to no access
    return "none"


def is_authorized(user_role: str, required_role: str) -> bool:
    """
    Security by design: explicit allowlist, not blocklist.
    """
    role_hierarchy = {
        "admin": 3,
        "editor": 2,
        "viewer": 1,
        "none": 0,
    }

    # Unknown roles get no access (fail secure)
    user_level = role_hierarchy.get(user_role, 0)
    required_level = role_hierarchy.get(required_role, 999)

    return user_level >= required_level


# Secure defaults in configuration
SECURE_DEFAULTS = {
    "session_timeout_minutes": 15,       # Short timeout
    "max_login_attempts": 5,             # Lockout after failures
    "password_min_length": 12,           # Strong passwords
    "require_mfa": True,                 # MFA on by default
    "cors_origins": [],                  # No CORS by default (deny all)
    "debug_mode": False,                 # Never expose debug info
    "tls_min_version": "1.2",           # No legacy TLS
    "cookie_secure": True,               # HTTPS-only cookies
    "cookie_httponly": True,             # No JS access to cookies
    "cookie_samesite": "Strict",         # Prevent CSRF
    "content_security_policy": "default-src 'self'",  # Strict CSP
    "x_frame_options": "DENY",           # Prevent clickjacking
    "hsts_max_age": 31536000,           # 1 year HSTS
}
```

### 8.2 Security by Obscurity (and Why It Fails)

Security by obscurity relies on secrecy of the implementation rather than the soundness of the design. It is not a valid security strategy on its own.

```
┌─────────────────────────────────────────────────────────────────────┐
│          Security by Obscurity - Common (Bad) Examples               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  "Nobody will find our admin panel at /xK9mQ2p"                     │
│    → Trivially discovered by directory brute-forcing, logs, or      │
│      browser history                                                │
│                                                                      │
│  "Our custom encryption algorithm is secret, so it's secure"        │
│    → Reverse-engineered or leaked; no peer review means bugs        │
│                                                                      │
│  "We hide the server version headers"                                │
│    → Fingerprinting still identifies the server software            │
│                                                                      │
│  "Our database port is 54321 instead of 5432"                       │
│    → Port scanners check all 65535 ports in seconds                 │
│                                                                      │
│  "API keys are embedded in minified JavaScript"                     │
│    → Browser devtools expose everything instantly                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
│                                                                      │
│  NOTE: Obscurity can be a SUPPLEMENTARY layer (defense in depth)    │
│  but must NEVER be the PRIMARY security mechanism.                  │
│                                                                      │
│  Examples where obscurity adds marginal value:                      │
│  - Changing default SSH port (reduces noise, not actual attacks)    │
│  - Removing version banners (slows automated scanners slightly)    │
│  - Using non-standard directory names (deters casual browsing)     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. Risk Assessment Frameworks

### 9.1 Qualitative Risk Assessment

```
Risk = Likelihood × Impact

┌─────────────────────────────────────────────────────────────────────┐
│                    Risk Assessment Matrix (5×5)                      │
├──────────┬──────────┬──────────┬──────────┬──────────┬──────────────┤
│          │ Negligible│  Minor   │ Moderate │  Major   │ Catastrophic │
│          │ Impact   │ Impact   │ Impact   │ Impact   │  Impact      │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ Almost   │  Medium  │   High   │   High   │ Critical │  Critical    │
│ Certain  │          │          │          │          │              │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ Likely   │   Low    │  Medium  │   High   │   High   │  Critical    │
│          │          │          │          │          │              │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ Possible │   Low    │  Medium  │  Medium  │   High   │   High       │
│          │          │          │          │          │              │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ Unlikely │   Low    │   Low    │  Medium  │  Medium  │   High       │
│          │          │          │          │          │              │
├──────────┼──────────┼──────────┼──────────┼──────────┼──────────────┤
│ Rare     │   Low    │   Low    │   Low    │  Medium  │  Medium      │
│          │          │          │          │          │              │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────────┘
```

### 9.2 Risk Assessment in Code

```python
from dataclasses import dataclass
from enum import IntEnum
from typing import List

class Likelihood(IntEnum):
    RARE = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    ALMOST_CERTAIN = 5

class Impact(IntEnum):
    NEGLIGIBLE = 1
    MINOR = 2
    MODERATE = 3
    MAJOR = 4
    CATASTROPHIC = 5

@dataclass
class Risk:
    name: str
    description: str
    likelihood: Likelihood
    impact: Impact
    existing_controls: List[str]
    recommended_actions: List[str]

    @property
    def score(self) -> int:
        return self.likelihood.value * self.impact.value

    @property
    def rating(self) -> str:
        s = self.score
        if s >= 20:
            return "CRITICAL"
        elif s >= 12:
            return "HIGH"
        elif s >= 6:
            return "MEDIUM"
        else:
            return "LOW"

    @property
    def response_strategy(self) -> str:
        """Determine risk response based on rating."""
        strategies = {
            "CRITICAL": "Mitigate immediately - stop and fix",
            "HIGH": "Mitigate soon - plan remediation within sprint",
            "MEDIUM": "Accept with monitoring, or mitigate in backlog",
            "LOW": "Accept - document and monitor"
        }
        return strategies[self.rating]

def risk_assessment(risks: List[Risk]) -> None:
    """Perform and display a risk assessment."""
    print("=" * 70)
    print("RISK ASSESSMENT REPORT")
    print("=" * 70)

    # Sort by score (highest risk first)
    sorted_risks = sorted(risks, key=lambda r: r.score, reverse=True)

    for i, risk in enumerate(sorted_risks, 1):
        print(f"\n{'─' * 70}")
        print(f"Risk #{i}: {risk.name}")
        print(f"  Description:  {risk.description}")
        print(f"  Likelihood:   {risk.likelihood.name} ({risk.likelihood.value})")
        print(f"  Impact:       {risk.impact.name} ({risk.impact.value})")
        print(f"  Score:        {risk.score} / 25")
        print(f"  Rating:       {risk.rating}")
        print(f"  Response:     {risk.response_strategy}")
        print(f"  Controls:     {', '.join(risk.existing_controls)}")
        print(f"  Recommended:  {', '.join(risk.recommended_actions)}")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    from collections import Counter
    ratings = Counter(r.rating for r in risks)
    for rating in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
        count = ratings.get(rating, 0)
        bar = "#" * (count * 3)
        print(f"  {rating:10s}: {count:2d} {bar}")

# Example assessment
risks = [
    Risk(
        name="Unpatched web server",
        description="Apache 2.4.49 with known path traversal CVE-2021-41773",
        likelihood=Likelihood.ALMOST_CERTAIN,
        impact=Impact.MAJOR,
        existing_controls=["WAF in front of server"],
        recommended_actions=["Patch immediately to 2.4.52+"]
    ),
    Risk(
        name="No encryption at rest",
        description="Customer PII stored in plaintext in PostgreSQL",
        likelihood=Likelihood.POSSIBLE,
        impact=Impact.CATASTROPHIC,
        existing_controls=["Database behind VPN", "Access logging"],
        recommended_actions=[
            "Enable TDE or application-level encryption",
            "Classify and encrypt PII columns"
        ]
    ),
    Risk(
        name="Weak password policy",
        description="Minimum 6 characters, no complexity requirements",
        likelihood=Likelihood.LIKELY,
        impact=Impact.MODERATE,
        existing_controls=["Account lockout after 10 attempts"],
        recommended_actions=[
            "Require 12+ characters",
            "Check against breached password lists",
            "Implement MFA"
        ]
    ),
    Risk(
        name="Missing CSRF protection",
        description="State-changing forms lack CSRF tokens",
        likelihood=Likelihood.POSSIBLE,
        impact=Impact.MODERATE,
        existing_controls=["SameSite=Lax cookies"],
        recommended_actions=["Add CSRF tokens to all forms", "Use SameSite=Strict"]
    ),
]

risk_assessment(risks)
```

### 9.3 Notable Risk Frameworks

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Risk Assessment Frameworks                        │
├─────────────────┬───────────────────────────────────────────────────┤
│ Framework       │ Description                                       │
├─────────────────┼───────────────────────────────────────────────────┤
│ NIST RMF        │ Risk Management Framework (SP 800-37)             │
│ (SP 800-37)     │ - Categorize → Select → Implement → Assess →     │
│                 │   Authorize → Monitor                             │
│                 │ - US government standard, widely adopted           │
├─────────────────┼───────────────────────────────────────────────────┤
│ ISO 27005       │ Information security risk management              │
│                 │ - Context → Assessment → Treatment → Monitoring   │
│                 │ - International standard, certification available │
├─────────────────┼───────────────────────────────────────────────────┤
│ FAIR            │ Factor Analysis of Information Risk               │
│                 │ - Quantitative: estimates risk in dollar terms    │
│                 │ - Loss Event Frequency × Loss Magnitude           │
├─────────────────┼───────────────────────────────────────────────────┤
│ OCTAVE          │ Operationally Critical Threat, Asset, and        │
│                 │ Vulnerability Evaluation                          │
│                 │ - Self-directed, organization-focused             │
│                 │ - Good for smaller teams                          │
├─────────────────┼───────────────────────────────────────────────────┤
│ OWASP Risk      │ Focused on web application risks                 │
│ Rating          │ - Likelihood × Impact with detailed factors      │
│                 │ - Aligned with OWASP Top 10                      │
└─────────────────┴───────────────────────────────────────────────────┘
```

---

## 10. Common Vulnerability Scoring System (CVSS)

CVSS provides a standardized way to capture the principal characteristics of a vulnerability and produce a numerical score (0-10) reflecting its severity.

### 10.1 CVSS v3.1 / v4.0 Metrics

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CVSS v3.1 Metric Groups                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  BASE METRICS (Intrinsic vulnerability characteristics)              │
│  ├── Attack Vector (AV): Network / Adjacent / Local / Physical      │
│  ├── Attack Complexity (AC): Low / High                              │
│  ├── Privileges Required (PR): None / Low / High                    │
│  ├── User Interaction (UI): None / Required                          │
│  ├── Scope (S): Unchanged / Changed                                  │
│  ├── Confidentiality Impact (C): None / Low / High                  │
│  ├── Integrity Impact (I): None / Low / High                        │
│  └── Availability Impact (A): None / Low / High                     │
│                                                                      │
│  TEMPORAL METRICS (Change over time)                                 │
│  ├── Exploit Code Maturity: Not Defined / Unproven / PoC /          │
│  │                          Functional / High                        │
│  ├── Remediation Level: Not Defined / Official Fix / Temp Fix /     │
│  │                      Workaround / Unavailable                    │
│  └── Report Confidence: Not Defined / Unknown / Reasonable /        │
│                         Confirmed                                   │
│                                                                      │
│  ENVIRONMENTAL METRICS (Organization-specific)                       │
│  ├── Modified Base metrics (customized for your environment)        │
│  └── Security Requirements (CR, IR, AR): Low / Medium / High       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

Score Ranges:
  0.0        = None
  0.1 - 3.9  = Low
  4.0 - 6.9  = Medium
  7.0 - 8.9  = High
  9.0 - 10.0 = Critical
```

### 10.2 CVSS Score Calculator

```python
from dataclasses import dataclass
from typing import Optional
import math

@dataclass
class CVSSv31:
    """
    Simplified CVSS v3.1 Base Score calculator.
    Reference: https://www.first.org/cvss/v3.1/specification-document
    """
    # Attack Vector
    attack_vector: str       # N(etwork), A(djacent), L(ocal), P(hysical)
    # Attack Complexity
    attack_complexity: str   # L(ow), H(igh)
    # Privileges Required
    privileges_required: str # N(one), L(ow), H(igh)
    # User Interaction
    user_interaction: str    # N(one), R(equired)
    # Scope
    scope: str               # U(nchanged), C(hanged)
    # Impact
    confidentiality: str     # N(one), L(ow), H(igh)
    integrity: str           # N(one), L(ow), H(igh)
    availability: str        # N(one), L(ow), H(igh)

    # Metric value lookups
    AV_VALUES = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
    AC_VALUES = {"L": 0.77, "H": 0.44}
    PR_VALUES_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
    PR_VALUES_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.50}
    UI_VALUES = {"N": 0.85, "R": 0.62}
    IMPACT_VALUES = {"N": 0.00, "L": 0.22, "H": 0.56}

    @property
    def vector_string(self) -> str:
        return (f"CVSS:3.1/AV:{self.attack_vector}/AC:{self.attack_complexity}/"
                f"PR:{self.privileges_required}/UI:{self.user_interaction}/"
                f"S:{self.scope}/C:{self.confidentiality}/"
                f"I:{self.integrity}/A:{self.availability}")

    def _exploitability_score(self) -> float:
        av = self.AV_VALUES[self.attack_vector]
        ac = self.AC_VALUES[self.attack_complexity]

        if self.scope == "C":
            pr = self.PR_VALUES_CHANGED[self.privileges_required]
        else:
            pr = self.PR_VALUES_UNCHANGED[self.privileges_required]

        ui = self.UI_VALUES[self.user_interaction]
        return 8.22 * av * ac * pr * ui

    def _impact_score(self) -> float:
        c = self.IMPACT_VALUES[self.confidentiality]
        i = self.IMPACT_VALUES[self.integrity]
        a = self.IMPACT_VALUES[self.availability]

        iss = 1 - ((1 - c) * (1 - i) * (1 - a))

        if self.scope == "U":
            return 6.42 * iss
        else:
            return 7.52 * (iss - 0.029) - 3.25 * ((iss - 0.02) ** 15)

    def base_score(self) -> float:
        impact = self._impact_score()
        exploitability = self._exploitability_score()

        if impact <= 0:
            return 0.0

        if self.scope == "U":
            score = min(impact + exploitability, 10)
        else:
            score = min(1.08 * (impact + exploitability), 10)

        # Round up to nearest 0.1
        return math.ceil(score * 10) / 10

    @property
    def severity(self) -> str:
        score = self.base_score()
        if score == 0.0:
            return "NONE"
        elif score <= 3.9:
            return "LOW"
        elif score <= 6.9:
            return "MEDIUM"
        elif score <= 8.9:
            return "HIGH"
        else:
            return "CRITICAL"

# Example: Log4Shell (CVE-2021-44228)
log4shell = CVSSv31(
    attack_vector="N",      # Network - remotely exploitable
    attack_complexity="L",   # Low - trivial to exploit
    privileges_required="N", # None - no auth needed
    user_interaction="N",    # None - no user action needed
    scope="C",               # Changed - can affect other components
    confidentiality="H",     # High - full read access
    integrity="H",           # High - full write access
    availability="H",        # High - can crash/DoS
)
print(f"Log4Shell (CVE-2021-44228)")
print(f"  Vector: {log4shell.vector_string}")
print(f"  Score:  {log4shell.base_score()}")
print(f"  Rating: {log4shell.severity}")

print()

# Example: A less severe vulnerability (local privilege escalation)
local_vuln = CVSSv31(
    attack_vector="L",      # Local - requires local access
    attack_complexity="H",   # High - specific conditions needed
    privileges_required="L", # Low - needs basic user account
    user_interaction="N",    # None
    scope="U",               # Unchanged
    confidentiality="H",     # High
    integrity="H",           # High
    availability="N",        # None
)
print(f"Local Privilege Escalation")
print(f"  Vector: {local_vuln.vector_string}")
print(f"  Score:  {local_vuln.base_score()}")
print(f"  Rating: {local_vuln.severity}")

print()

# Example: Low severity (reflected XSS requiring user interaction)
xss_vuln = CVSSv31(
    attack_vector="N",
    attack_complexity="L",
    privileges_required="N",
    user_interaction="R",    # Requires user to click a link
    scope="C",               # Can affect other origins
    confidentiality="L",     # Low - can steal some data
    integrity="L",           # Low - can modify some page content
    availability="N",        # None
)
print(f"Reflected XSS")
print(f"  Vector: {xss_vuln.vector_string}")
print(f"  Score:  {xss_vuln.base_score()}")
print(f"  Rating: {xss_vuln.severity}")
```

---

## 11. Putting It All Together: A Worked Example

Let us walk through a complete security analysis for a hypothetical online banking application.

### 11.1 System Description

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Online Banking System                              │
│                                                                      │
│   ┌──────────┐     HTTPS     ┌───────────┐     gRPC     ┌────────┐ │
│   │ Mobile   │ ────────────▶│   API     │ ──────────▶│ Account│ │
│   │ App      │               │  Gateway  │             │ Service│ │
│   └──────────┘               └─────┬─────┘             └────────┘ │
│                                    │                               │
│   ┌──────────┐     HTTPS     ┌─────▼─────┐     gRPC     ┌────────┐ │
│   │ Web      │ ────────────▶│   Auth    │ ──────────▶│Transfer│ │
│   │ Browser  │               │  Service  │             │Service │ │
│   └──────────┘               └─────┬─────┘             └────────┘ │
│                                    │                               │
│   ┌──────────┐                ┌────▼──────┐              ┌────────┐ │
│   │ Admin    │ ── VPN ──────▶│  Admin    │ ── SQL ────▶│  DB    │ │
│   │ Console  │               │  Portal   │              │(PgSQL) │ │
│   └──────────┘               └───────────┘              └────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

### 11.2 STRIDE Threat Summary

| # | Threat | Category | CIA Impact | DREAD Score | Mitigation |
|---|--------|----------|------------|-------------|------------|
| T1 | Stolen session token reused | Spoofing | C | 7.2 | Short-lived JWTs, token binding |
| T2 | Modified transfer amount | Tampering | I | 8.0 | Request signing, server validation |
| T3 | User denies making transfer | Repudiation | Non-rep | 5.4 | Signed audit logs, digital receipts |
| T4 | Account numbers leaked | Info Disc. | C | 7.8 | Encryption at rest, field masking |
| T5 | DDoS on API gateway | DoS | A | 6.2 | CDN, rate limiting, auto-scaling |
| T6 | Admin escalates to super admin | Elev. Priv | Auth | 8.6 | RBAC, approval workflows, MFA |

### 11.3 Defense in Depth Applied

```
Layer 1 - Policies:     Security training, incident response plan
Layer 2 - Physical:     Data center with biometric access
Layer 3 - Perimeter:    CDN/WAF, DDoS protection, VPN for admin
Layer 4 - Network:      VLAN segmentation, mTLS between services
Layer 5 - Host:         Hardened OS, auto-patching, EDR agents
Layer 6 - Application:  Input validation, CSRF tokens, CSP headers
Layer 7 - Data:         AES-256 encryption at rest, TLS 1.3 in transit
```

### 11.4 Least Privilege Applied

```
Mobile App:     Can only call API Gateway (read own account, initiate transfers)
API Gateway:    Can route to Auth and Account services; no direct DB access
Auth Service:   Read-only on users table; write-only on sessions table
Account Svc:    Read/write on accounts table; no access to users
Transfer Svc:   Insert-only on transfers table; read on accounts
Admin Portal:   Accessible only via VPN; requires MFA + manager approval
Database:       No internet access; connections only from service VPC
```

---

## 12. Exercises

### Exercise 1: CIA Analysis (Beginner)

For each scenario below, identify which CIA property is primarily violated and what controls could prevent it:

1. A hospital's electronic health record system goes offline during a ransomware attack.
2. A hacker changes the price of items in an e-commerce database from $99 to $0.01.
3. An employee emails a spreadsheet of customer Social Security numbers to their personal email.
4. DNS records are modified to redirect bank customers to a phishing site.
5. A disgruntled employee deletes the production database backups.

### Exercise 2: STRIDE Threat Model (Intermediate)

Draw a Data Flow Diagram for a food delivery application with these components:
- Customer mobile app
- Restaurant dashboard (web)
- API server
- Payment processor (Stripe)
- Delivery tracking service
- PostgreSQL database

Apply STRIDE to identify at least 8 threats. For each threat, specify:
- STRIDE category
- Affected DFD element
- Severity (use DREAD scoring)
- At least one mitigation

### Exercise 3: CVSS Scoring (Intermediate)

Calculate the CVSS v3.1 base score for these vulnerabilities:

1. A web application allows unauthenticated users to download any file from the server by manipulating a URL parameter (path traversal). No user interaction is required.

2. A desktop application has a buffer overflow that can be triggered by opening a specially crafted file. The attacker needs the victim to open the file, and successful exploitation gives the attacker the same privileges as the victim.

3. An API endpoint requires authentication but fails to check if the authenticated user has permission to access the requested resource (IDOR). Only affects data within the same application.

### Exercise 4: Defense in Depth Design (Advanced)

You are designing a security architecture for a startup that processes credit card payments. They currently have:
- A single web server running both the application and database
- No encryption at rest
- Admin access via SSH with password authentication
- No logging or monitoring
- All employees share one database admin password

Design a 7-layer defense-in-depth strategy. For each layer, specify:
- At least 2 specific controls
- The threats each control mitigates
- Priority (implement first/second/third)

### Exercise 5: Least Privilege Audit (Advanced)

Given the following AWS IAM policy, identify all violations of least privilege and rewrite it to be properly scoped:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "s3:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:*",
                "rds:*"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "iam:*",
            "Resource": "*"
        }
    ]
}
```

Context: This policy is for a web application backend that needs to:
- Read and write files to a single S3 bucket (`my-app-uploads`)
- Read from an RDS PostgreSQL instance (`my-app-db`)
- No EC2 management needed (runs on ECS)
- No IAM management needed

### Exercise 6: Security by Design Review (Advanced)

Review the following Flask code and identify all security issues. Rewrite it following security-by-design principles:

```python
from flask import Flask, request, jsonify
import sqlite3
import hashlib

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    password_hash = hashlib.md5(password.encode()).hexdigest()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password_hash}'"
    cursor.execute(query)
    user = cursor.fetchone()

    if user:
        return jsonify({"status": "success", "user_id": user[0], "role": user[3]})
    else:
        return jsonify({"status": "failed", "error": f"No user found with username {username}"})

@app.route('/admin/delete_user/<user_id>')
def delete_user(user_id):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute(f"DELETE FROM users WHERE id={user_id}")
    conn.commit()
    return jsonify({"status": "deleted"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
```

Identify at least 10 security issues and provide the corrected code.

---

## References

- Anderson, R. (2020). *Security Engineering*, 3rd Edition. Wiley.
- Shostack, A. (2014). *Threat Modeling: Designing for Security*. Wiley.
- OWASP Threat Modeling - https://owasp.org/www-community/Threat_Modeling
- NIST SP 800-30: Guide for Conducting Risk Assessments
- NIST SP 800-37: Risk Management Framework
- FIRST CVSS v3.1 Specification - https://www.first.org/cvss/v3.1/specification-document
- Microsoft STRIDE - https://learn.microsoft.com/en-us/azure/security/develop/threat-modeling-tool-threats

---

## Next Lesson

[02. Cryptography Basics](./02_Cryptography_Basics.md) covers symmetric and asymmetric encryption, key exchange, and digital signatures with practical Python examples.
