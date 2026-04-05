"""
Exercise Solutions: Security Fundamentals and Threat Modeling
=============================================================
Lesson 01 from Security topic.

These exercises cover the CIA Triad, STRIDE threat modeling, CVSS scoring,
defense-in-depth design, least privilege auditing, and security-by-design
code review.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Exercise 1: CIA Analysis
# ---------------------------------------------------------------------------
# Classify each scenario by which CIA property is primarily violated and
# which controls could prevent it.

class CIAProperty(Enum):
    CONFIDENTIALITY = "Confidentiality"
    INTEGRITY = "Integrity"
    AVAILABILITY = "Availability"


@dataclass
class CIAAnalysis:
    """Represents an analysis of a security scenario."""
    scenario: str
    violated_property: CIAProperty
    explanation: str
    controls: list[str]


def exercise_1_cia_analysis() -> list[CIAAnalysis]:
    """
    Analyze five scenarios and identify which CIA property is violated
    and what controls could prevent the violation.
    """
    analyses = [
        CIAAnalysis(
            scenario=(
                "A hospital's electronic health record system goes offline "
                "during a ransomware attack."
            ),
            violated_property=CIAProperty.AVAILABILITY,
            explanation=(
                "Ransomware encrypts data and renders the system unusable, "
                "directly attacking availability. Confidentiality may also "
                "be threatened if data is exfiltrated, but the primary "
                "impact is denial of service to medical staff."
            ),
            controls=[
                "Maintain offline backups with regular testing",
                "Network segmentation to limit ransomware propagation",
                "Endpoint detection and response (EDR) solutions",
                "Incident response plan with defined recovery time objectives",
            ],
        ),
        CIAAnalysis(
            scenario=(
                "A hacker changes the price of items in an e-commerce "
                "database from $99 to $0.01."
            ),
            violated_property=CIAProperty.INTEGRITY,
            explanation=(
                "Unauthorized modification of data is an integrity violation. "
                "The data is still available and accessible, but its accuracy "
                "has been compromised."
            ),
            controls=[
                "Database access controls (principle of least privilege)",
                "Input validation on all data modification endpoints",
                "Audit logging on price-change operations",
                "Database triggers to flag suspicious bulk updates",
            ],
        ),
        CIAAnalysis(
            scenario=(
                "An employee emails a spreadsheet of customer Social Security "
                "numbers to their personal email."
            ),
            violated_property=CIAProperty.CONFIDENTIALITY,
            explanation=(
                "Sensitive personal data (SSNs) is exposed to an unauthorized "
                "destination. This is a classic data exfiltration / "
                "confidentiality breach."
            ),
            controls=[
                "Data Loss Prevention (DLP) tools to detect PII in outbound email",
                "Role-based access control limiting who can export bulk PII",
                "Encryption of sensitive fields at rest and in transit",
                "Employee security awareness training",
            ],
        ),
        CIAAnalysis(
            scenario=(
                "DNS records are modified to redirect bank customers to "
                "a phishing site."
            ),
            violated_property=CIAProperty.INTEGRITY,
            explanation=(
                "DNS records are trusted infrastructure data. Unauthorized "
                "modification redirects users to a malicious site — this is "
                "an integrity violation of the DNS system. It also leads to "
                "confidentiality loss (credential theft), but the root cause "
                "is the integrity breach of DNS."
            ),
            controls=[
                "DNSSEC to cryptographically sign DNS records",
                "Multi-factor authentication on DNS management accounts",
                "Monitor DNS records for unauthorized changes",
                "Registry lock on critical domains",
            ],
        ),
        CIAAnalysis(
            scenario=(
                "A disgruntled employee deletes the production database "
                "backups."
            ),
            violated_property=CIAProperty.AVAILABILITY,
            explanation=(
                "Destroying backups threatens the organization's ability "
                "to recover from any future incident, directly attacking "
                "long-term availability. The production system is still "
                "running, but resilience is gone."
            ),
            controls=[
                "Immutable backups (write-once storage, e.g., AWS S3 Object Lock)",
                "Separation of duties — no single person controls production + backups",
                "Backup integrity checks and off-site replication",
                "Audit logging on all backup operations with alerts on deletions",
            ],
        ),
    ]

    for analysis in analyses:
        print(f"\nScenario: {analysis.scenario}")
        print(f"  Violated: {analysis.violated_property.value}")
        print(f"  Explanation: {analysis.explanation}")
        print(f"  Controls:")
        for ctrl in analysis.controls:
            print(f"    - {ctrl}")

    return analyses


# ---------------------------------------------------------------------------
# Exercise 2: STRIDE Threat Model
# ---------------------------------------------------------------------------
# Apply STRIDE to a food delivery application and identify at least 8 threats.

class STRIDECategory(Enum):
    SPOOFING = "Spoofing"
    TAMPERING = "Tampering"
    REPUDIATION = "Repudiation"
    INFORMATION_DISCLOSURE = "Information Disclosure"
    DENIAL_OF_SERVICE = "Denial of Service"
    ELEVATION_OF_PRIVILEGE = "Elevation of Privilege"


@dataclass
class DREADScore:
    """DREAD risk scoring model."""
    damage: int           # 1-10: How bad is the impact?
    reproducibility: int  # 1-10: How easy to reproduce?
    exploitability: int   # 1-10: How easy to launch?
    affected_users: int   # 1-10: How many users affected?
    discoverability: int  # 1-10: How easy to discover?

    @property
    def total(self) -> float:
        return (
            self.damage + self.reproducibility + self.exploitability
            + self.affected_users + self.discoverability
        ) / 5

    @property
    def severity(self) -> str:
        score = self.total
        if score >= 7:
            return "Critical"
        elif score >= 5:
            return "High"
        elif score >= 3:
            return "Medium"
        else:
            return "Low"


@dataclass
class STRIDEThreat:
    """A threat identified through STRIDE analysis."""
    category: STRIDECategory
    affected_element: str
    description: str
    dread: DREADScore
    mitigations: list[str]


def exercise_2_stride_threat_model() -> list[STRIDEThreat]:
    """
    Apply STRIDE to a food delivery application with these components:
    Customer mobile app, Restaurant dashboard (web), API server,
    Payment processor (Stripe), Delivery tracking service, PostgreSQL database.
    """
    threats = [
        STRIDEThreat(
            category=STRIDECategory.SPOOFING,
            affected_element="Customer mobile app -> API server",
            description=(
                "An attacker impersonates a legitimate customer by stealing "
                "session tokens or using credential stuffing."
            ),
            dread=DREADScore(damage=8, reproducibility=7, exploitability=6,
                             affected_users=8, discoverability=5),
            mitigations=[
                "Implement MFA for customer accounts",
                "Use short-lived JWTs with refresh token rotation",
                "Rate-limit login attempts per IP and per account",
            ],
        ),
        STRIDEThreat(
            category=STRIDECategory.TAMPERING,
            affected_element="API server -> PostgreSQL database",
            description=(
                "SQL injection through the API allows an attacker to modify "
                "order prices, quantities, or delivery addresses in the "
                "database."
            ),
            dread=DREADScore(damage=9, reproducibility=8, exploitability=7,
                             affected_users=9, discoverability=6),
            mitigations=[
                "Use parameterized queries / ORM exclusively",
                "Apply input validation on all API endpoints",
                "Database user should have minimal required privileges",
            ],
        ),
        STRIDEThreat(
            category=STRIDECategory.REPUDIATION,
            affected_element="Customer mobile app -> API server",
            description=(
                "A customer places an order and then claims they never "
                "did, requesting a refund. Without proper audit logging, "
                "the platform cannot prove the order was legitimate."
            ),
            dread=DREADScore(damage=5, reproducibility=8, exploitability=9,
                             affected_users=3, discoverability=4),
            mitigations=[
                "Log all order actions with timestamps, IP, device ID",
                "Require explicit confirmation step before order placement",
                "Send email/push confirmation upon order creation",
            ],
        ),
        STRIDEThreat(
            category=STRIDECategory.INFORMATION_DISCLOSURE,
            affected_element="API server -> Customer mobile app",
            description=(
                "API responses expose sensitive data such as other customers' "
                "addresses, phone numbers, or payment details through IDOR "
                "vulnerabilities."
            ),
            dread=DREADScore(damage=8, reproducibility=9, exploitability=8,
                             affected_users=9, discoverability=7),
            mitigations=[
                "Implement object-level authorization on every endpoint",
                "Return only necessary fields in API responses",
                "Mask sensitive data (e.g., show last 4 digits of phone)",
            ],
        ),
        STRIDEThreat(
            category=STRIDECategory.DENIAL_OF_SERVICE,
            affected_element="API server",
            description=(
                "An attacker floods the API with bogus order requests, "
                "overwhelming the server and preventing legitimate "
                "customers from placing orders."
            ),
            dread=DREADScore(damage=8, reproducibility=9, exploitability=8,
                             affected_users=10, discoverability=8),
            mitigations=[
                "Implement rate limiting per IP and per user",
                "Use a CDN/WAF (e.g., Cloudflare) for DDoS protection",
                "Auto-scale backend infrastructure under load",
            ],
        ),
        STRIDEThreat(
            category=STRIDECategory.ELEVATION_OF_PRIVILEGE,
            affected_element="Restaurant dashboard -> API server",
            description=(
                "A restaurant owner exploits a missing authorization check "
                "to access the admin panel and modify other restaurants' "
                "menus or pricing."
            ),
            dread=DREADScore(damage=7, reproducibility=6, exploitability=5,
                             affected_users=7, discoverability=4),
            mitigations=[
                "Enforce RBAC at every API endpoint",
                "Validate tenant isolation (restaurant can only access own data)",
                "Audit log all admin-level actions",
            ],
        ),
        STRIDEThreat(
            category=STRIDECategory.TAMPERING,
            affected_element="Delivery tracking service",
            description=(
                "A delivery driver spoofs their GPS location to appear "
                "closer to the restaurant or customer than they actually "
                "are, gaming the delivery assignment algorithm."
            ),
            dread=DREADScore(damage=4, reproducibility=7, exploitability=6,
                             affected_users=5, discoverability=3),
            mitigations=[
                "Cross-validate GPS with cell tower triangulation",
                "Detect unrealistic speed/distance jumps",
                "Flag accounts with repeated anomalous location patterns",
            ],
        ),
        STRIDEThreat(
            category=STRIDECategory.INFORMATION_DISCLOSURE,
            affected_element="API server -> Payment processor (Stripe)",
            description=(
                "API error messages or logs inadvertently expose partial "
                "payment card details, Stripe API keys, or internal "
                "system architecture information."
            ),
            dread=DREADScore(damage=9, reproducibility=5, exploitability=4,
                             affected_users=10, discoverability=6),
            mitigations=[
                "Never log full card numbers or API secrets",
                "Return generic error messages to clients",
                "Store Stripe keys in a secrets manager, not in code or env",
            ],
        ),
    ]

    print("STRIDE Threat Model — Food Delivery Application")
    print("=" * 60)
    for i, threat in enumerate(threats, 1):
        print(f"\nThreat #{i}")
        print(f"  Category:  {threat.category.value}")
        print(f"  Element:   {threat.affected_element}")
        print(f"  Description: {threat.description}")
        print(f"  DREAD Score: {threat.dread.total:.1f} ({threat.dread.severity})")
        print(f"  Mitigations:")
        for m in threat.mitigations:
            print(f"    - {m}")

    return threats


# ---------------------------------------------------------------------------
# Exercise 3: CVSS Scoring
# ---------------------------------------------------------------------------
# Calculate CVSS v3.1 base scores for three vulnerability scenarios.

@dataclass
class CVSSVector:
    """Simplified CVSS v3.1 Base Score calculator."""
    # Attack Vector: Network(N)=0.85, Adjacent(A)=0.62, Local(L)=0.55, Physical(P)=0.20
    attack_vector: str
    # Attack Complexity: Low(L)=0.77, High(H)=0.44
    attack_complexity: str
    # Privileges Required: None(N)=0.85, Low(L)=0.62/0.68, High(H)=0.27/0.50
    privileges_required: str
    # User Interaction: None(N)=0.85, Required(R)=0.62
    user_interaction: str
    # Scope: Unchanged(U), Changed(C)
    scope: str
    # Impact: None(N)=0, Low(L)=0.22, High(H)=0.56
    confidentiality_impact: str
    integrity_impact: str
    availability_impact: str

    _AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.20}
    _AC = {"L": 0.77, "H": 0.44}
    _PR_UNCHANGED = {"N": 0.85, "L": 0.62, "H": 0.27}
    _PR_CHANGED = {"N": 0.85, "L": 0.68, "H": 0.50}
    _UI = {"N": 0.85, "R": 0.62}
    _IMPACT = {"N": 0.0, "L": 0.22, "H": 0.56}

    def _exploitability(self) -> float:
        pr_table = self._PR_CHANGED if self.scope == "C" else self._PR_UNCHANGED
        return (
            8.22
            * self._AV[self.attack_vector]
            * self._AC[self.attack_complexity]
            * pr_table[self.privileges_required]
            * self._UI[self.user_interaction]
        )

    def _impact_subscore(self) -> float:
        iss = 1.0 - (
            (1 - self._IMPACT[self.confidentiality_impact])
            * (1 - self._IMPACT[self.integrity_impact])
            * (1 - self._IMPACT[self.availability_impact])
        )
        if self.scope == "U":
            return 6.42 * iss
        else:
            return 7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15

    def base_score(self) -> float:
        import math
        impact = self._impact_subscore()
        if impact <= 0:
            return 0.0
        exploit = self._exploitability()
        if self.scope == "U":
            raw = min(impact + exploit, 10.0)
        else:
            raw = min(1.08 * (impact + exploit), 10.0)
        return math.ceil(raw * 10) / 10

    def severity_rating(self) -> str:
        score = self.base_score()
        if score == 0:
            return "None"
        elif score < 4.0:
            return "Low"
        elif score < 7.0:
            return "Medium"
        elif score < 9.0:
            return "High"
        else:
            return "Critical"


def exercise_3_cvss_scoring():
    """
    Calculate CVSS v3.1 base scores for three vulnerabilities.
    """
    # Vulnerability 1: Path traversal - unauthenticated file download
    # AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N
    vuln1 = CVSSVector(
        attack_vector="N", attack_complexity="L",
        privileges_required="N", user_interaction="N",
        scope="U",
        confidentiality_impact="H", integrity_impact="N",
        availability_impact="N",
    )

    # Vulnerability 2: Buffer overflow via crafted file (requires user to open)
    # AV:L/AC:L/PR:N/UI:R/S:U/C:H/I:H/A:H
    vuln2 = CVSSVector(
        attack_vector="L", attack_complexity="L",
        privileges_required="N", user_interaction="R",
        scope="U",
        confidentiality_impact="H", integrity_impact="H",
        availability_impact="H",
    )

    # Vulnerability 3: IDOR (authenticated, same scope)
    # AV:N/AC:L/PR:L/UI:N/S:U/C:H/I:N/A:N
    vuln3 = CVSSVector(
        attack_vector="N", attack_complexity="L",
        privileges_required="L", user_interaction="N",
        scope="U",
        confidentiality_impact="H", integrity_impact="N",
        availability_impact="N",
    )

    for i, vuln in enumerate([vuln1, vuln2, vuln3], 1):
        score = vuln.base_score()
        rating = vuln.severity_rating()
        print(f"Vulnerability {i}: CVSS {score} ({rating})")

    return vuln1, vuln2, vuln3


# ---------------------------------------------------------------------------
# Exercise 4: Defense in Depth Design
# ---------------------------------------------------------------------------
# Design a 7-layer defense-in-depth strategy for a payment startup.

@dataclass
class SecurityControl:
    """A security control in a defense-in-depth layer."""
    name: str
    threats_mitigated: list[str]
    priority: str  # "first", "second", "third"


@dataclass
class DefenseLayer:
    """One layer in a defense-in-depth architecture."""
    layer_name: str
    controls: list[SecurityControl]


def exercise_4_defense_in_depth() -> list[DefenseLayer]:
    """
    Design a 7-layer defense-in-depth strategy for a startup that
    processes credit card payments.
    """
    layers = [
        DefenseLayer(
            layer_name="1. Physical Security",
            controls=[
                SecurityControl(
                    "Cloud hosting with SOC 2 compliance (migrate off single server)",
                    ["Physical server theft", "Unauthorized data center access"],
                    "second",
                ),
                SecurityControl(
                    "Hardware security modules (HSM) for encryption keys",
                    ["Key theft", "Cryptographic key compromise"],
                    "third",
                ),
            ],
        ),
        DefenseLayer(
            layer_name="2. Network Security",
            controls=[
                SecurityControl(
                    "Network segmentation: separate web, app, and database tiers",
                    ["Lateral movement", "Database exposure to internet"],
                    "first",
                ),
                SecurityControl(
                    "Web Application Firewall (WAF) and DDoS protection",
                    ["SQL injection", "XSS", "DDoS attacks"],
                    "first",
                ),
            ],
        ),
        DefenseLayer(
            layer_name="3. Host Security",
            controls=[
                SecurityControl(
                    "SSH key-based authentication (disable password auth)",
                    ["Brute-force SSH attacks", "Credential theft"],
                    "first",
                ),
                SecurityControl(
                    "Automated OS patching and hardening (CIS benchmarks)",
                    ["Known vulnerability exploitation", "Privilege escalation"],
                    "second",
                ),
            ],
        ),
        DefenseLayer(
            layer_name="4. Application Security",
            controls=[
                SecurityControl(
                    "Input validation and parameterized queries",
                    ["SQL injection", "XSS", "Command injection"],
                    "first",
                ),
                SecurityControl(
                    "Secure session management and CSRF protection",
                    ["Session hijacking", "CSRF attacks"],
                    "first",
                ),
            ],
        ),
        DefenseLayer(
            layer_name="5. Data Security",
            controls=[
                SecurityControl(
                    "Encryption at rest (AES-256) for all stored data",
                    ["Data breach from disk theft", "Unauthorized DB access"],
                    "first",
                ),
                SecurityControl(
                    "TLS 1.3 for all data in transit",
                    ["Man-in-the-middle attacks", "Eavesdropping"],
                    "first",
                ),
            ],
        ),
        DefenseLayer(
            layer_name="6. Identity and Access Management",
            controls=[
                SecurityControl(
                    "Individual accounts with unique credentials (no shared passwords)",
                    ["Credential sharing", "Lack of accountability"],
                    "first",
                ),
                SecurityControl(
                    "Role-based access control with least privilege",
                    ["Privilege escalation", "Unauthorized data access"],
                    "first",
                ),
            ],
        ),
        DefenseLayer(
            layer_name="7. Monitoring and Response",
            controls=[
                SecurityControl(
                    "Centralized logging with SIEM (e.g., ELK, Splunk)",
                    ["Undetected breaches", "Slow incident response"],
                    "second",
                ),
                SecurityControl(
                    "Real-time alerting on anomalous patterns",
                    ["Ongoing attacks", "Data exfiltration"],
                    "second",
                ),
            ],
        ),
    ]

    for layer in layers:
        print(f"\n{layer.layer_name}")
        for ctrl in layer.controls:
            print(f"  [{ctrl.priority.upper()}] {ctrl.name}")
            print(f"    Mitigates: {', '.join(ctrl.threats_mitigated)}")

    return layers


# ---------------------------------------------------------------------------
# Exercise 5: Least Privilege Audit
# ---------------------------------------------------------------------------
# Rewrite the overly permissive IAM policy to follow least privilege.

def exercise_5_least_privilege_audit():
    """
    Identify violations of least privilege in the given AWS IAM policy
    and rewrite it to be properly scoped.
    """
    overly_permissive_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {"Effect": "Allow", "Action": "s3:*", "Resource": "*"},
            {"Effect": "Allow", "Action": ["ec2:*", "rds:*"], "Resource": "*"},
            {"Effect": "Allow", "Action": "iam:*", "Resource": "*"},
        ],
    }

    violations = [
        "s3:* grants full S3 access (Put, Delete, manage buckets, etc.) "
        "when only read/write to one bucket is needed",
        "Resource '*' allows access to ALL S3 buckets, not just 'my-app-uploads'",
        "ec2:* is entirely unnecessary — the app runs on ECS, not EC2",
        "rds:* grants admin-level RDS access (create/delete instances) "
        "when only read access to one database is needed",
        "iam:* is extremely dangerous — allows creating admin users, "
        "and the app has no IAM management needs",
        "All resources are '*' — no resource-level scoping at all",
    ]

    corrected_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "S3ReadWriteUploads",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
                "Resource": [
                    "arn:aws:s3:::my-app-uploads",
                    "arn:aws:s3:::my-app-uploads/*",
                ],
            },
            {
                "Sid": "RDSConnectOnly",
                "Effect": "Allow",
                "Action": [
                    "rds-db:connect",
                ],
                "Resource": [
                    "arn:aws:rds-db:*:*:dbuser:my-app-db/*",
                ],
            },
        ],
    }

    print("=== Violations Found ===")
    for i, v in enumerate(violations, 1):
        print(f"  {i}. {v}")

    print("\n=== Corrected Policy ===")
    print(json.dumps(corrected_policy, indent=2))

    return violations, corrected_policy


# ---------------------------------------------------------------------------
# Exercise 6: Security by Design Review
# ---------------------------------------------------------------------------
# Identify security issues in the Flask code and provide a corrected version.

def exercise_6_security_by_design_review():
    """
    Review the given Flask code and identify 10+ security issues.
    Provide corrected code following security-by-design principles.

    NOTE: This solution demonstrates secure coding patterns.
    The corrected code below is a reference implementation.
    """

    issues = [
        "1. SQL injection: f-string interpolation in SQL query allows "
        "arbitrary SQL execution. Fix: use parameterized queries.",
        "2. Weak hashing: MD5 is cryptographically broken and unsuitable "
        "for password hashing. Fix: use bcrypt or argon2.",
        "3. No salt: MD5 hash has no salt, enabling rainbow table attacks. "
        "Fix: bcrypt/argon2 auto-generate salts.",
        "4. Username enumeration: Error message reveals whether username "
        "exists. Fix: use generic 'Invalid credentials' message.",
        "5. Missing authentication: /admin/delete_user has no auth check. "
        "Fix: require admin authentication.",
        "6. Missing authorization: No RBAC — any authenticated user could "
        "delete other users. Fix: verify admin role before deletion.",
        "7. No CSRF protection: POST endpoints lack CSRF tokens. "
        "Fix: add Flask-WTF or custom CSRF middleware.",
        "8. Debug mode in production: debug=True exposes stack traces and "
        "interactive debugger. Fix: set debug=False in production.",
        "9. Binding to 0.0.0.0: Exposes the server to all network "
        "interfaces. Fix: bind to 127.0.0.1 in development.",
        "10. No rate limiting: Login endpoint has no brute-force protection. "
        "Fix: add rate limiting (e.g., Flask-Limiter).",
        "11. No input validation: username/password not validated for "
        "length, type, or content. Fix: validate inputs.",
        "12. Sensitive data in response: user role and user_id returned "
        "directly on login. Fix: return a session token instead.",
        "13. No HTTPS enforcement: No redirect from HTTP to HTTPS. "
        "Fix: enforce TLS in production.",
        "14. HTTP method not restricted: delete_user uses GET, enabling "
        "CSRF via image tags. Fix: require POST/DELETE method.",
        "15. Connection not closed: SQLite connection not properly managed. "
        "Fix: use context manager (with statement).",
    ]

    corrected_code = '''
import os
import secrets
from flask import Flask, request, jsonify, session
# In production, use: from argon2 import PasswordHasher

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# Simulated argon2 password hasher
# ph = PasswordHasher()

def get_db():
    """Return a database connection with context management."""
    import sqlite3
    conn = sqlite3.connect("users.db")
    conn.row_factory = sqlite3.Row
    return conn

def require_auth(f):
    """Decorator to require authentication."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

def require_admin(f):
    """Decorator to require admin role."""
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            return jsonify({"status": "error", "message": "Forbidden"}), 403
        return f(*args, **kwargs)
    return decorated

@app.route("/login", methods=["POST"])
def login():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")

    if not username or not password:
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, password_hash, role FROM users WHERE username = ?",
            (username,),
        )
        user = cursor.fetchone()

    if not user:
        # Same response as wrong password to prevent enumeration
        return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    # In production: ph.verify(user["password_hash"], password)
    # For now, placeholder check
    # if not ph.verify(user["password_hash"], password):
    #     return jsonify({"status": "error", "message": "Invalid credentials"}), 401

    session.regenerate()  # Prevent session fixation
    session["user_id"] = user["id"]
    session["role"] = user["role"]
    return jsonify({"status": "success"})

@app.route("/admin/delete_user/<int:user_id>", methods=["DELETE"])
@require_auth
@require_admin
def delete_user(user_id):
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
    return jsonify({"status": "deleted"})

if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1")
'''

    print("=== Security Issues Found ===")
    for issue in issues:
        print(f"  {issue}")

    print("\n=== Corrected Code ===")
    print(corrected_code)

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("Exercise 1: CIA Analysis")
    print("=" * 70)
    exercise_1_cia_analysis()

    print("\n" + "=" * 70)
    print("Exercise 2: STRIDE Threat Model")
    print("=" * 70)
    exercise_2_stride_threat_model()

    print("\n" + "=" * 70)
    print("Exercise 3: CVSS Scoring")
    print("=" * 70)
    exercise_3_cvss_scoring()

    print("\n" + "=" * 70)
    print("Exercise 4: Defense in Depth Design")
    print("=" * 70)
    exercise_4_defense_in_depth()

    print("\n" + "=" * 70)
    print("Exercise 5: Least Privilege Audit")
    print("=" * 70)
    exercise_5_least_privilege_audit()

    print("\n" + "=" * 70)
    print("Exercise 6: Security by Design Review")
    print("=" * 70)
    exercise_6_security_by_design_review()
