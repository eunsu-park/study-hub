# 07. OWASP Top 10 (2021)

**Previous**: [06. Authorization and Access Control](06_Authorization.md) | **Next**: [08. Injection Attacks and Prevention](08_Injection_Attacks.md)

---

The OWASP (Open Worldwide Application Security Project) Top 10 is the most widely recognized document for web application security risks. Updated periodically based on real-world vulnerability data, it serves as a standard awareness document for developers and security professionals. The 2021 edition reflects significant shifts in the threat landscape, with three new categories and major reshuffling. This lesson covers each of the ten categories with descriptions, real-world examples, vulnerable code, fixed code, and prevention strategies.

## Learning Objectives

- Understand all ten OWASP Top 10 (2021) categories
- Identify vulnerable code patterns for each category
- Write secure code that prevents each vulnerability class
- Apply the OWASP Top 10 as a security checklist during development and code review
- Recognize real-world impacts of each vulnerability

---

## 1. Overview: The 2021 Top 10

```
┌─────────────────────────────────────────────────────────────────┐
│                  OWASP Top 10 - 2021                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  #   Category                              Change from 2017     │
│  ─── ──────────────────────────────────── ────────────────────  │
│  A01  Broken Access Control                 ↑ from #5           │
│  A02  Cryptographic Failures                ↑ from #3 (renamed) │
│  A03  Injection                             ↓ from #1           │
│  A04  Insecure Design                       ★ NEW               │
│  A05  Security Misconfiguration             ↑ from #6           │
│  A06  Vulnerable/Outdated Components        ↑ from #9 (renamed) │
│  A07  Identification & Auth Failures        ↓ from #2 (renamed) │
│  A08  Software & Data Integrity Failures    ★ NEW               │
│  A09  Security Logging & Monitoring Failures↑ from #10 (renamed)│
│  A10  Server-Side Request Forgery (SSRF)    ★ NEW               │
│                                                                  │
│  Key Trends:                                                     │
│  - Broken Access Control is now #1 (94% of apps tested)         │
│  - Injection dropped from #1 to #3 (frameworks help)            │
│  - Three new categories reflect modern threats                   │
│  - Supply chain and integrity concerns are rising               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. A01: Broken Access Control

### 2.1 Description

Access control enforces policy so that users cannot act outside their intended permissions. Broken access control is the **most common** web application vulnerability, found in 94% of applications tested.

```
┌─────────────────────────────────────────────────────────────────┐
│              A01: Broken Access Control                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Common Weaknesses:                                              │
│  - Bypassing access control by modifying URL/parameters/API     │
│  - Viewing or editing someone else's account (IDOR)             │
│  - Accessing API with missing access controls (POST/PUT/DELETE) │
│  - Elevation of privilege (acting as admin without login)        │
│  - Metadata manipulation (replay JWT, tamper cookies)            │
│  - CORS misconfiguration allowing unauthorized API access       │
│  - Force browsing to unauthenticated/admin pages                │
│                                                                  │
│  Impact: Data theft, unauthorized data modification,            │
│          account takeover, full system compromise                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Vulnerable Code

```python
# VULNERABLE: No access control on admin endpoint
@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
def delete_user(user_id):
    # Anyone who knows this URL can delete users!
    db.delete_user(user_id)
    return jsonify({"status": "deleted"})


# VULNERABLE: IDOR - No ownership check
@app.route('/api/orders/<int:order_id>')
@require_auth
def get_order(order_id):
    order = db.get_order(order_id)
    return jsonify(order)  # User A can see User B's orders
```

### 2.3 Fixed Code

```python
# FIXED: Proper authorization check
@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@require_role('admin')  # Only admins can access
def delete_user(user_id):
    db.delete_user(user_id)
    return jsonify({"status": "deleted"})


# FIXED: Ownership verification
@app.route('/api/orders/<int:order_id>')
@require_auth
def get_order(order_id):
    order = db.get_order(order_id)
    if not order:
        return jsonify({"error": "Not found"}), 404

    # Verify the order belongs to the authenticated user
    if order['user_id'] != g.current_user['id']:
        return jsonify({"error": "Not found"}), 404  # 404, not 403

    return jsonify(order)
```

### 2.4 Prevention

- Deny by default (except for public resources)
- Implement access control mechanisms once, reuse them everywhere
- Enforce record ownership (don't just rely on user-supplied IDs)
- Disable web server directory listing
- Log access control failures and alert administrators
- Rate-limit API access to minimize automated scanning damage
- Invalidate JWT tokens on logout (server-side token blocklist)

---

## 3. A02: Cryptographic Failures

### 3.1 Description

Previously called "Sensitive Data Exposure," this category focuses on failures related to cryptography that often lead to exposure of sensitive data. It covers both the failure to encrypt data that should be protected and the use of weak or obsolete cryptographic algorithms.

```
┌─────────────────────────────────────────────────────────────────┐
│              A02: Cryptographic Failures                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Ask yourself:                                                   │
│  1. Is any data transmitted or stored in cleartext?             │
│  2. Are old/weak cryptographic algorithms or protocols used?    │
│  3. Are default crypto keys used or keys not rotated?           │
│  4. Is encryption not enforced (missing HTTPS redirect)?        │
│  5. Are proper HTTP security headers missing?                   │
│  6. Is the server certificate validated correctly?              │
│  7. Is deprecated hashing used for passwords (MD5, SHA1)?      │
│                                                                  │
│  Sensitive Data Categories:                                      │
│  - Passwords, credit card numbers, health records               │
│  - Personal data, business secrets                               │
│  - Data protected by privacy regulations (GDPR, HIPAA, PCI)    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Vulnerable Code

```python
# VULNERABLE: Storing passwords with MD5
import hashlib

def store_password(password):
    # MD5 is broken for password hashing!
    hash_value = hashlib.md5(password.encode()).hexdigest()
    db.store(hash_value)

# VULNERABLE: Hardcoded encryption key
ENCRYPTION_KEY = "my-secret-key-123"  # Committed to source control!

# VULNERABLE: Using ECB mode (reveals patterns)
from Crypto.Cipher import AES
cipher = AES.new(key, AES.MODE_ECB)  # ECB mode is insecure!
encrypted = cipher.encrypt(data)

# VULNERABLE: HTTP for sensitive data
# No redirect from HTTP to HTTPS
# No HSTS header
```

### 3.3 Fixed Code

```python
# FIXED: Use Argon2 for passwords
from argon2 import PasswordHasher
ph = PasswordHasher()

def store_password(password):
    hash_value = ph.hash(password)  # Argon2id with automatic salting
    db.store(hash_value)

# FIXED: Key from environment, not source code
import os
ENCRYPTION_KEY = os.environ['ENCRYPTION_KEY']  # 256-bit key

# FIXED: Use GCM mode (authenticated encryption)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

key = AESGCM.generate_key(bit_length=256)
aesgcm = AESGCM(key)
nonce = os.urandom(12)  # NEVER reuse nonces
encrypted = aesgcm.encrypt(nonce, data, associated_data)

# FIXED: Enforce HTTPS and add security headers
@app.after_request
def set_security_headers(response):
    response.headers['Strict-Transport-Security'] = \
        'max-age=31536000; includeSubDomains'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response
```

### 3.4 Prevention

- Classify data according to sensitivity; apply controls per classification
- Do not store sensitive data unnecessarily; discard it as soon as possible
- Encrypt all sensitive data at rest and in transit (TLS 1.2+)
- Use strong, standard algorithms (AES-256-GCM, RSA-2048+, Ed25519)
- Use authenticated encryption (GCM, ChaCha20-Poly1305), never ECB
- Store passwords using Argon2id, bcrypt, or scrypt
- Generate keys using cryptographically secure random number generators
- Disable caching for pages containing sensitive data

---

## 4. A03: Injection

### 4.1 Description

Injection flaws occur when untrusted data is sent to an interpreter as part of a command or query. The attacker's hostile data can trick the interpreter into executing unintended commands or accessing data without authorization. SQL injection, NoSQL injection, OS command injection, and LDAP injection are all forms of this vulnerability.

> **Note**: Injection is covered in much greater depth in Lesson 08. This section provides a summary.

```
┌─────────────────────────────────────────────────────────────────┐
│              A03: Injection                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Injection Types:                                                │
│  ├── SQL Injection          (most common)                        │
│  ├── NoSQL Injection        (MongoDB, etc.)                      │
│  ├── Command Injection      (os.system, subprocess)              │
│  ├── LDAP Injection         (directory services)                 │
│  ├── XPath Injection        (XML queries)                        │
│  ├── Template Injection     (Jinja2, Twig SSTI)                 │
│  └── Expression Language    (Spring EL, OGNL)                    │
│                                                                  │
│  Root Cause:                                                     │
│  User input is concatenated into queries/commands                │
│  instead of being parameterized or properly escaped              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Vulnerable vs Fixed Code

```python
# VULNERABLE: SQL Injection
@app.route('/search')
def search():
    query = request.args.get('q')
    # String concatenation = SQL injection!
    results = db.execute(f"SELECT * FROM products WHERE name LIKE '%{query}%'")
    return jsonify(results)

# Attack: /search?q=' OR '1'='1' --


# FIXED: Parameterized query
@app.route('/search')
def search():
    query = request.args.get('q', '')
    results = db.execute(
        "SELECT * FROM products WHERE name LIKE :query",
        {"query": f"%{query}%"}  # Parameter binding
    )
    return jsonify(results)
```

### 4.3 Prevention

- Use parameterized queries / prepared statements (always)
- Use ORM frameworks (SQLAlchemy, Django ORM) which handle parameterization
- Validate and sanitize all input (whitelist validation)
- Escape special characters for the specific interpreter
- Use LIMIT in queries to prevent mass disclosure on injection

---

## 5. A04: Insecure Design

### 5.1 Description

This is a **new category** in 2021 that focuses on risks related to design and architectural flaws. It calls for more use of threat modeling, secure design patterns, and reference architectures. Insecure design cannot be fixed by a perfect implementation - a flawed design is inherently vulnerable.

```
┌─────────────────────────────────────────────────────────────────┐
│              A04: Insecure Design                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Insecure Design ≠ Insecure Implementation                      │
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ Insecure Design  │    │ Insecure Impl.   │                   │
│  │                  │    │                  │                    │
│  │ The blueprint is │    │ The blueprint is │                    │
│  │ flawed. No code  │    │ good but the     │                    │
│  │ fix can help.    │    │ code has bugs.   │                    │
│  │                  │    │                  │                    │
│  │ Example:         │    │ Example:         │                    │
│  │ Password reset   │    │ SQL injection    │                    │
│  │ sends password   │    │ in login form    │                    │
│  │ in plaintext     │    │                  │                    │
│  │ email by design  │    │                  │                    │
│  └──────────────────┘    └──────────────────┘                   │
│                                                                  │
│  Examples of Insecure Design:                                    │
│  - No rate limiting on authentication (brute force possible)    │
│  - Security questions as sole password recovery method          │
│  - No input validation architecture (each dev implements own)   │
│  - Storing secrets in source code by design                     │
│  - No separation between tenant data in multi-tenant system     │
│  - Missing abuse cases in requirements (only happy paths)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Real-World Example: Cinema Booking

```python
# INSECURE DESIGN: Movie ticket booking without rate limiting
# The design allows a bot to book ALL tickets for popular movies

@app.route('/api/book', methods=['POST'])
@require_auth
def book_ticket():
    movie_id = request.json['movie_id']
    seats = request.json['seats']  # No limit on number of seats!

    # No rate limiting per user
    # No maximum seats per transaction
    # No CAPTCHA for high-demand events
    # No fraud detection

    booking = create_booking(g.user.id, movie_id, seats)
    return jsonify(booking)


# SECURE DESIGN: With proper safeguards

@app.route('/api/book', methods=['POST'])
@require_auth
@rate_limit(max_requests=5, per_minutes=1)  # Rate limiting
def book_ticket():
    movie_id = request.json['movie_id']
    seats = request.json['seats']

    # Design-level controls
    MAX_SEATS_PER_BOOKING = 6
    if len(seats) > MAX_SEATS_PER_BOOKING:
        return jsonify({"error": f"Maximum {MAX_SEATS_PER_BOOKING} seats per booking"}), 400

    # Check user hasn't already booked too many seats for this showing
    existing = get_user_bookings(g.user.id, movie_id)
    if len(existing) >= 2:  # Max 2 bookings per user per movie
        return jsonify({"error": "Maximum bookings reached for this movie"}), 400

    # For high-demand events, require additional verification
    movie = get_movie(movie_id)
    if movie.get('high_demand'):
        if not verify_captcha(request.json.get('captcha_token')):
            return jsonify({"error": "CAPTCHA verification required"}), 400

    booking = create_booking(g.user.id, movie_id, seats)
    return jsonify(booking)
```

### 5.3 Prevention

- Use threat modeling for critical authentication, access control, and business logic
- Integrate security language and controls into user stories
- Write unit and integration tests to validate that all critical flows are resistant to abuse
- Design for failure: limit resource consumption per user/session
- Tier your application layers to separate trust boundaries
- Use secure design patterns (e.g., stateless session management, input validation frameworks)

---

## 6. A05: Security Misconfiguration

### 6.1 Description

Security misconfiguration is the most commonly seen issue in practice. It includes improperly configured permissions, unnecessary features enabled, default accounts/passwords unchanged, overly informative error messages, and missing security hardening.

```
┌─────────────────────────────────────────────────────────────────┐
│              A05: Security Misconfiguration                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Common Misconfigurations:                                       │
│  ┌──────────────────────────────────────────────────┐           │
│  │ Issue                    │ Risk                   │           │
│  ├──────────────────────────┼───────────────────────┤           │
│  │ Debug mode in production │ Stack traces exposed   │           │
│  │ Default admin:admin      │ Immediate compromise   │           │
│  │ Directory listing ON     │ Source/config exposure  │           │
│  │ Unnecessary services     │ Increased attack surface│          │
│  │ Verbose error messages   │ Information disclosure  │           │
│  │ Missing security headers │ XSS, clickjacking      │           │
│  │ CORS: Access-Control-    │ Cross-origin attacks    │           │
│  │   Allow-Origin: *        │                         │           │
│  │ Outdated software        │ Known vulnerabilities   │           │
│  │ S3 bucket public         │ Data breach             │           │
│  │ .git folder exposed      │ Source code leak        │           │
│  └──────────────────────────────────────────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Vulnerable Configuration

```python
# VULNERABLE: Flask in debug mode in production
app = Flask(__name__)
app.config['DEBUG'] = True  # Exposes interactive debugger!
app.config['SECRET_KEY'] = 'dev'  # Weak/default secret key

# VULNERABLE: Overly permissive CORS
from flask_cors import CORS
CORS(app, origins="*")  # Any website can make requests!

# VULNERABLE: Detailed error messages
@app.errorhandler(500)
def handle_error(error):
    return jsonify({
        "error": str(error),
        "traceback": traceback.format_exc(),  # Leaks internals!
        "database": app.config['DATABASE_URI'],  # Leaks credentials!
    }), 500
```

### 6.3 Fixed Configuration

```python
import os

app = Flask(__name__)

# FIXED: Environment-based configuration
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['SECRET_KEY'] = os.environ['SECRET_KEY']  # Strong, random

# FIXED: Restrictive CORS
from flask_cors import CORS
CORS(app, origins=[
    "https://myapp.com",
    "https://www.myapp.com",
])

# FIXED: Generic error messages in production
@app.errorhandler(500)
def handle_error(error):
    # Log the full error internally
    app.logger.error(f"Internal error: {error}", exc_info=True)

    # Return generic message to user
    return jsonify({
        "error": "An internal error occurred",
        "reference": generate_error_reference_id(),  # For support tickets
    }), 500

# FIXED: Security headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '0'  # Disable legacy XSS filter
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "frame-ancestors 'none'"
    )
    response.headers['Permissions-Policy'] = (
        'camera=(), microphone=(), geolocation=()'
    )
    if request.is_secure:
        response.headers['Strict-Transport-Security'] = (
            'max-age=31536000; includeSubDomains; preload'
        )
    return response
```

### 6.4 Prevention

- Implement a repeatable hardening process for all environments
- Remove or do not install unnecessary features, frameworks, and components
- Review and update configurations as part of the patch management process
- Use infrastructure as code for consistent, auditable deployments
- Implement automated security configuration verification in CI/CD
- Separate environments (dev, staging, production) with different credentials

---

## 7. A06: Vulnerable and Outdated Components

### 7.1 Description

Applications that use components (libraries, frameworks, OS) with known vulnerabilities may be exploited. This is a supply chain risk that is increasingly important.

```
┌─────────────────────────────────────────────────────────────────┐
│          A06: Vulnerable and Outdated Components                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  You are vulnerable if:                                          │
│  - You don't know the versions of all components used           │
│  - The software is out of support or unpatched                  │
│  - You don't scan for vulnerabilities regularly                 │
│  - You don't fix or upgrade in a timely fashion                 │
│  - Developers don't test compatibility of updated libraries     │
│  - Component configurations are not secured (see A05)            │
│                                                                  │
│  Real-World Impact:                                              │
│  - Log4Shell (CVE-2021-44228): Critical RCE in Log4j           │
│  - Equifax breach (2017): Unpatched Apache Struts               │
│  - Event-Stream (2018): Malicious npm package                   │
│  - ua-parser-js (2021): Supply chain attack on npm              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Dependency Auditing

```bash
# Python: Check for known vulnerabilities
pip install pip-audit
pip-audit                        # Scan installed packages
pip-audit -r requirements.txt    # Scan requirements file

# Python: Alternative with safety
pip install safety
safety check                     # Check installed packages

# Node.js: Built-in audit
npm audit
npm audit fix                    # Auto-fix where possible

# General: OWASP Dependency-Check
# Scans multiple languages (Java, .NET, Python, JS, etc.)
# https://owasp.org/www-project-dependency-check/

# GitHub: Dependabot (automatic PR for vulnerable deps)
# GitLab: Dependency Scanning in CI/CD pipeline

# Pin dependencies with hash verification (Python)
pip install --require-hashes -r requirements.txt
```

```python
# requirements.txt with pinned versions and hashes
# Generate with: pip-compile --generate-hashes requirements.in
flask==3.0.0 \
    --hash=sha256:21128f47e...
werkzeug==3.0.1 \
    --hash=sha256:5a7b12abc...
```

### 7.3 Prevention

- Remove unused dependencies, features, components, files, and documentation
- Continuously inventory component versions using tools (pip-audit, npm audit, OWASP Dependency-Check)
- Monitor sources like CVE and NVD for vulnerability alerts
- Only obtain components from official sources over secure links
- Monitor for unmaintained libraries (no security patches)
- Have a patching plan: test and deploy updates promptly

---

## 8. A07: Identification and Authentication Failures

### 8.1 Description

Confirmation of the user's identity, authentication, and session management is critical to protect against authentication-related attacks. This was previously called "Broken Authentication."

> **Note**: See Lesson 05 for comprehensive authentication coverage.

```
┌─────────────────────────────────────────────────────────────────┐
│          A07: Identification and Authentication Failures          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Weaknesses:                                                     │
│  - Permits brute force or credential stuffing                   │
│  - Allows weak or well-known passwords                          │
│  - Uses weak credential recovery ("What's your pet's name?")   │
│  - Uses plain text or weakly hashed passwords                   │
│  - Missing or ineffective multi-factor authentication           │
│  - Exposes session ID in URL                                    │
│  - Does not rotate session ID after login                       │
│  - Does not properly invalidate sessions on logout              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Vulnerable vs Fixed Code

```python
# VULNERABLE: No brute force protection
@app.route('/login', methods=['POST'])
def login_vulnerable():
    username = request.json['username']
    password = request.json['password']

    user = db.find_user(username)
    if user and check_password(password, user.password_hash):
        session['user_id'] = user.id  # No session regeneration!
        return jsonify({"status": "success"})

    return jsonify({"error": "Invalid credentials"}), 401


# FIXED: With rate limiting, lockout, and session management
from flask_limiter import Limiter

limiter = Limiter(app, default_limits=["200 per day"])

@app.route('/login', methods=['POST'])
@limiter.limit("5 per minute")  # Rate limit login attempts
def login_secure():
    username = request.json['username']
    password = request.json['password']

    # Check account lockout
    if is_account_locked(username):
        return jsonify({"error": "Account temporarily locked"}), 429

    user = db.find_user(username)
    if user and check_password(password, user.password_hash):
        # Reset failed attempts
        reset_failed_attempts(username)

        # Regenerate session (prevent session fixation)
        session.clear()
        session['user_id'] = user.id
        session['created_at'] = time.time()
        session.permanent = True

        return jsonify({"status": "success"})

    # Increment failed attempts
    record_failed_attempt(username)

    # Generic error (don't reveal if username exists)
    return jsonify({"error": "Invalid credentials"}), 401
```

### 8.3 Prevention

- Implement multi-factor authentication (TOTP or FIDO2)
- Use rate limiting and account lockout for login endpoints
- Check passwords against breached password databases
- Use secure password storage (Argon2id, bcrypt)
- Regenerate session IDs after login
- Implement proper session timeout and invalidation

---

## 9. A08: Software and Data Integrity Failures

### 9.1 Description

This **new category** focuses on making assumptions about software updates, critical data, and CI/CD pipelines without verifying integrity. It includes the former "Insecure Deserialization" category.

```
┌─────────────────────────────────────────────────────────────────┐
│          A08: Software and Data Integrity Failures               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Attack Vectors:                                                 │
│                                                                  │
│  1. CI/CD Pipeline Compromise:                                   │
│     Attacker modifies build pipeline to inject malicious code    │
│     ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐              │
│     │ Code │───▶│Build │───▶│ Test │───▶│Deploy│              │
│     │      │    │      │    │      │    │      │              │
│     └──────┘    └──┬───┘    └──────┘    └──────┘              │
│                    │                                             │
│                    ▼ Attacker injects backdoor here              │
│                                                                  │
│  2. Auto-Update Without Verification:                            │
│     App downloads update without verifying digital signature     │
│     Attacker performs MITM to serve malicious update             │
│                                                                  │
│  3. Insecure Deserialization:                                    │
│     App deserializes untrusted data, leading to RCE              │
│     pickle.loads(user_input)  ← Remote code execution!          │
│                                                                  │
│  4. Dependency Confusion:                                        │
│     Attacker publishes malicious package with same name as       │
│     internal package in public registry                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 Vulnerable Code: Insecure Deserialization

```python
import pickle
import yaml

# VULNERABLE: Deserializing untrusted data with pickle
@app.route('/api/import', methods=['POST'])
def import_data_vulnerable():
    data = request.get_data()
    obj = pickle.loads(data)  # RCE! Attacker can execute arbitrary code
    return jsonify({"status": "imported"})

# Attack payload:
# import pickle, os
# class Exploit:
#     def __reduce__(self):
#         return (os.system, ('rm -rf /',))
# pickle.dumps(Exploit())


# VULNERABLE: YAML load (allows arbitrary Python objects)
@app.route('/api/config', methods=['POST'])
def load_config_vulnerable():
    config = yaml.load(request.data)  # Unsafe! Can execute code
    return jsonify(config)


# FIXED: Use safe alternatives
import json

@app.route('/api/import', methods=['POST'])
def import_data_secure():
    # Use JSON instead of pickle for data exchange
    data = request.get_json()
    if not validate_schema(data):  # Validate structure
        return jsonify({"error": "Invalid data format"}), 400
    return jsonify({"status": "imported"})


@app.route('/api/config', methods=['POST'])
def load_config_secure():
    config = yaml.safe_load(request.data)  # safe_load blocks code execution
    return jsonify(config)
```

### 9.3 Prevention

- Use digital signatures or similar to verify software/data integrity
- Ensure libraries and dependencies consume trusted repositories
- Use a software supply chain security tool (SLSA, Sigstore)
- Review CI/CD pipeline for unauthorized access or modifications
- Do not send unsigned or unencrypted serialized data to untrusted clients
- Never use `pickle`, `marshal`, or `yaml.load()` on untrusted data

---

## 10. A09: Security Logging and Monitoring Failures

### 10.1 Description

Without sufficient logging and monitoring, breaches cannot be detected in time. Most successful attacks start with vulnerability probing, and allowing this probing to continue unchecked can increase the likelihood of a successful exploit.

```
┌─────────────────────────────────────────────────────────────────┐
│      A09: Security Logging and Monitoring Failures               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Problems:                                                       │
│  - Login failures not logged                                    │
│  - Warnings and errors generate no or unclear log messages      │
│  - Logs are only stored locally (lost if server compromised)    │
│  - No alerting thresholds or effective escalation               │
│  - Penetration testing and DAST scans don't trigger alerts      │
│  - Application can't detect, escalate, or alert for attacks     │
│  - Log injection: attacker writes fake log entries              │
│                                                                  │
│  Average time to detect a breach: 287 days (IBM 2021)           │
│  Cost of breach detected in <200 days: $3.6M                   │
│  Cost of breach detected in >200 days: $4.9M                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 10.2 Security Logging Implementation

```python
"""
security_logging.py - Comprehensive security event logging
"""
import logging
import json
import time
from flask import Flask, request, g
from datetime import datetime, timezone

app = Flask(__name__)

# ==============================================================
# Security Event Logger
# ==============================================================

class SecurityLogger:
    """Structured security event logging."""

    def __init__(self, app_name: str):
        self.logger = logging.getLogger(f"security.{app_name}")
        self.logger.setLevel(logging.INFO)

        # JSON formatter for structured logging
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def _log_event(self, event_type: str, severity: str, **kwargs):
        """Log a structured security event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "severity": severity,
            "ip_address": request.remote_addr if request else None,
            "user_agent": request.headers.get('User-Agent') if request else None,
            "user_id": getattr(g, 'current_user', {}).get('id'),
            **kwargs
        }
        self.logger.info(json.dumps(event))

    def login_success(self, user_id: str):
        self._log_event("auth.login.success", "INFO", user_id=user_id)

    def login_failure(self, username: str, reason: str):
        self._log_event("auth.login.failure", "WARNING",
                       username=self._sanitize(username),
                       reason=reason)

    def access_denied(self, resource: str, action: str):
        self._log_event("authz.denied", "WARNING",
                       resource=resource, action=action)

    def suspicious_activity(self, description: str, **details):
        self._log_event("security.suspicious", "HIGH",
                       description=description, **details)

    def data_access(self, resource_type: str, resource_id: str,
                    action: str):
        self._log_event("data.access", "INFO",
                       resource_type=resource_type,
                       resource_id=resource_id, action=action)

    def _sanitize(self, value: str) -> str:
        """Sanitize log input to prevent log injection."""
        if not isinstance(value, str):
            return str(value)
        # Remove newlines and control characters
        return value.replace('\n', '\\n').replace('\r', '\\r')


sec_log = SecurityLogger("myapp")


# Usage in routes
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    user = authenticate(username, password)
    if user:
        sec_log.login_success(user.id)
        return jsonify({"status": "success"})
    else:
        sec_log.login_failure(username, "invalid_credentials")
        return jsonify({"error": "Invalid credentials"}), 401


# Monitor for brute force
@app.before_request
def detect_brute_force():
    """Detect and alert on potential brute force attempts."""
    if request.path == '/login' and request.method == 'POST':
        ip = request.remote_addr
        recent_failures = get_recent_login_failures(ip, minutes=5)

        if recent_failures >= 10:
            sec_log.suspicious_activity(
                "Possible brute force attack",
                ip_address=ip,
                failure_count=recent_failures,
                timeframe_minutes=5
            )
            # Optionally: block IP, require CAPTCHA, alert SOC
```

### 10.3 What to Log

| Event | Severity | Details to Include |
|-------|----------|-------------------|
| Login success/failure | INFO/WARNING | Username, IP, timestamp, user agent |
| Authorization failures | WARNING | User, resource, action attempted |
| Input validation failures | WARNING | Endpoint, invalid input type |
| Admin actions | INFO | Admin user, action, target |
| Password changes | INFO | User ID (never the password) |
| Account lockout | WARNING | Username, failure count |
| Data export/download | INFO | User, data type, volume |
| API rate limiting triggered | WARNING | Client, endpoint, rate |
| System errors | ERROR | Error type, stack trace (not to client) |

### 10.4 Prevention

- Log all login, access control, and server-side input validation failures
- Ensure logs have sufficient context for forensic analysis
- Use structured logging (JSON) for machine parsability
- Implement centralized log management (ELK, Splunk, CloudWatch)
- Establish effective monitoring and alerting with escalation
- Create an incident response plan and practice it
- Protect logs from tampering (write-once storage, integrity checks)

---

## 11. A10: Server-Side Request Forgery (SSRF)

### 11.1 Description

SSRF flaws occur when a web application fetches a remote resource without validating the user-supplied URL. This allows an attacker to coerce the application to send a crafted request to an unexpected destination, even when protected by a firewall, VPN, or network ACL. This is a **new category** in 2021.

```
┌─────────────────────────────────────────────────────────────────┐
│              A10: Server-Side Request Forgery (SSRF)             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Normal:                                                         │
│  User ──▶ Server ──▶ https://api.external.com/data              │
│                                                                  │
│  SSRF Attack:                                                    │
│  User ──▶ Server ──▶ http://169.254.169.254/metadata            │
│                       (AWS instance metadata!)                   │
│                                                                  │
│  User ──▶ Server ──▶ http://localhost:6379/                     │
│                       (Internal Redis server!)                   │
│                                                                  │
│  User ──▶ Server ──▶ http://10.0.0.5:8080/admin                │
│                       (Internal admin panel!)                    │
│                                                                  │
│  User ──▶ Server ──▶ file:///etc/passwd                         │
│                       (Local file read!)                         │
│                                                                  │
│  Impact:                                                         │
│  - Access cloud instance metadata (steal credentials)           │
│  - Scan internal network                                         │
│  - Access internal services (Redis, databases, admin panels)    │
│  - Read local files                                              │
│  - Capital One breach (2019): SSRF → metadata → S3 access      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.2 Vulnerable Code

```python
import requests
from urllib.parse import urlparse

# VULNERABLE: Fetching arbitrary URLs
@app.route('/api/fetch-url', methods=['POST'])
def fetch_url_vulnerable():
    url = request.json['url']
    # No validation! User can access internal services
    response = requests.get(url)
    return jsonify({"content": response.text})

# Attack examples:
# {"url": "http://169.254.169.254/latest/meta-data/iam/security-credentials/"}
# {"url": "http://localhost:6379/CONFIG SET dir /tmp"}
# {"url": "file:///etc/passwd"}


# VULNERABLE: Image proxy with no validation
@app.route('/api/proxy-image')
def proxy_image_vulnerable():
    image_url = request.args.get('url')
    response = requests.get(image_url)
    return response.content, 200, {'Content-Type': response.headers.get('Content-Type')}
```

### 11.3 Fixed Code

```python
import ipaddress
import socket
from urllib.parse import urlparse
import requests

# Allowlist of permitted domains
ALLOWED_DOMAINS = {
    "api.example.com",
    "images.example.com",
    "cdn.trusted-partner.com",
}

# Blocked IP ranges (internal networks)
BLOCKED_IP_RANGES = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local (AWS metadata)
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("100.64.0.0/10"),       # Carrier-grade NAT
    ipaddress.ip_network("fd00::/8"),             # IPv6 private
    ipaddress.ip_network("::1/128"),              # IPv6 loopback
]


def is_safe_url(url: str) -> bool:
    """Validate that a URL is safe to fetch."""
    try:
        parsed = urlparse(url)

        # Only allow HTTP(S) schemes
        if parsed.scheme not in ('http', 'https'):
            return False

        hostname = parsed.hostname
        if not hostname:
            return False

        # Option A: Allowlist approach (strongest)
        if hostname not in ALLOWED_DOMAINS:
            return False

        # Option B: Blocklist approach (if allowlist not feasible)
        # Resolve hostname to IP and check against blocked ranges
        resolved_ips = socket.getaddrinfo(hostname, None)
        for family, _, _, _, sockaddr in resolved_ips:
            ip = ipaddress.ip_address(sockaddr[0])
            for blocked_range in BLOCKED_IP_RANGES:
                if ip in blocked_range:
                    return False

        return True

    except (ValueError, socket.gaierror):
        return False


# FIXED: URL validation before fetching
@app.route('/api/fetch-url', methods=['POST'])
def fetch_url_secure():
    url = request.json.get('url', '')

    if not is_safe_url(url):
        return jsonify({"error": "URL not allowed"}), 400

    try:
        response = requests.get(
            url,
            timeout=5,
            allow_redirects=False,  # Prevent redirect to internal services
        )

        # If there's a redirect, validate the destination too
        if response.is_redirect:
            redirect_url = response.headers.get('Location', '')
            if not is_safe_url(redirect_url):
                return jsonify({"error": "Redirect to blocked URL"}), 400

        return jsonify({"content": response.text[:10000]})  # Limit response size

    except requests.RequestException as e:
        return jsonify({"error": "Failed to fetch URL"}), 400
```

### 11.4 Prevention

- Sanitize and validate all client-supplied input URLs
- Enforce URL schema, port, and destination with a positive allow list
- Do not send raw responses to clients
- Disable HTTP redirections
- Use network-level segmentation (firewall rules preventing server-to-internal traffic)
- For cloud environments: use IMDSv2 (requires token) instead of IMDSv1 for instance metadata

---

## 12. Prevention Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│          OWASP Top 10 Prevention Checklist                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  A01 - Broken Access Control:                                    │
│  [ ] Default-deny access control                                │
│  [ ] Resource ownership validation                              │
│  [ ] Rate limiting on sensitive endpoints                       │
│  [ ] CORS properly configured                                   │
│                                                                  │
│  A02 - Cryptographic Failures:                                   │
│  [ ] Data classified by sensitivity                             │
│  [ ] TLS 1.2+ for all data in transit                           │
│  [ ] AES-256-GCM or ChaCha20 for data at rest                  │
│  [ ] Argon2id/bcrypt for passwords                              │
│                                                                  │
│  A03 - Injection:                                                │
│  [ ] Parameterized queries everywhere                           │
│  [ ] ORM used for database access                               │
│  [ ] Input validation (whitelist approach)                       │
│  [ ] Output encoding for context                                │
│                                                                  │
│  A04 - Insecure Design:                                          │
│  [ ] Threat modeling performed                                  │
│  [ ] Abuse cases in requirements                                │
│  [ ] Rate limiting, CAPTCHA for sensitive flows                 │
│  [ ] Security design reviews                                    │
│                                                                  │
│  A05 - Security Misconfiguration:                                │
│  [ ] Hardening checklist for each environment                   │
│  [ ] Debug mode OFF in production                               │
│  [ ] Security headers configured                                │
│  [ ] No default credentials                                     │
│                                                                  │
│  A06 - Vulnerable Components:                                    │
│  [ ] Dependency scanning in CI/CD                               │
│  [ ] Regular updates and patching                               │
│  [ ] Components from trusted sources only                       │
│  [ ] SBOMs (Software Bill of Materials) maintained              │
│                                                                  │
│  A07 - Authentication Failures:                                  │
│  [ ] MFA enabled (especially for admin accounts)                │
│  [ ] Account lockout after failed attempts                      │
│  [ ] Session regeneration after login                           │
│  [ ] Breached password checking                                 │
│                                                                  │
│  A08 - Integrity Failures:                                       │
│  [ ] Digital signatures for updates/deployments                 │
│  [ ] CI/CD pipeline secured with access controls                │
│  [ ] No deserialization of untrusted data                       │
│  [ ] SRI (Subresource Integrity) for external scripts           │
│                                                                  │
│  A09 - Logging & Monitoring:                                     │
│  [ ] Security events logged with sufficient context             │
│  [ ] Centralized log management                                 │
│  [ ] Alerting for suspicious patterns                           │
│  [ ] Incident response plan tested                              │
│                                                                  │
│  A10 - SSRF:                                                     │
│  [ ] URL validation (allowlist preferred)                       │
│  [ ] Network segmentation                                       │
│  [ ] Disabled unnecessary URL schemes                           │
│  [ ] IMDSv2 for cloud metadata                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. Exercises

### Exercise 1: Vulnerability Identification

Review the following Flask application and identify which OWASP Top 10 categories apply to each vulnerability:

```python
"""
Exercise: Identify the OWASP Top 10 category for each numbered issue.
Some lines may have multiple issues.
"""
from flask import Flask, request, jsonify, send_file
import pickle
import os
import sqlite3
import yaml
import requests

app = Flask(__name__)
app.config['DEBUG'] = True                        # Issue 1: ???
app.config['SECRET_KEY'] = 'development'          # Issue 2: ???

@app.route('/api/search')
def search():
    q = request.args.get('q')
    conn = sqlite3.connect('app.db')
    cursor = conn.execute(
        f"SELECT * FROM products WHERE name LIKE '%{q}%'"  # Issue 3: ???
    )
    return jsonify(cursor.fetchall())

@app.route('/api/user/<int:user_id>')
def get_user(user_id):                            # Issue 4: ???
    user = db.get_user(user_id)
    return jsonify(user)

@app.route('/api/import', methods=['POST'])
def import_data():
    data = pickle.loads(request.data)             # Issue 5: ???
    return jsonify({"status": "imported"})

@app.route('/api/fetch', methods=['POST'])
def fetch():
    url = request.json['url']
    resp = requests.get(url)                      # Issue 6: ???
    return resp.text

@app.route('/api/config', methods=['POST'])
def load_config():
    config = yaml.load(request.data)              # Issue 7: ???
    return jsonify(config)

@app.route('/login', methods=['POST'])
def login():
    username = request.json['username']
    password = request.json['password']
    user = db.find_user(username)
    if user and user.password == password:         # Issue 8: ???
        session['user'] = user.id
        return jsonify({"status": "ok"})
    return jsonify({"error": f"User {username} not found or wrong password"}),  401  # Issue 9: ???

@app.errorhandler(500)
def error(e):
    return jsonify({
        "error": str(e),
        "trace": traceback.format_exc()           # Issue 10: ???
    }), 500
```

### Exercise 2: Secure Application Design

Design and implement security controls for a file-sharing application:

```python
"""
Exercise: Implement security controls for each OWASP Top 10 category.
The application allows users to upload, share, and download files.
"""

class SecureFileSharing:
    def upload_file(self, user_id: str, file_data: bytes,
                    filename: str) -> dict:
        """
        Secure file upload.
        Consider: A03 (injection via filename), A04 (file size limits),
                  A05 (file type validation), A08 (integrity check)
        """
        pass

    def share_file(self, owner_id: str, file_id: str,
                   target_user_id: str, permissions: list) -> bool:
        """
        Share a file with another user.
        Consider: A01 (access control), A04 (sharing limits)
        """
        pass

    def download_file(self, user_id: str, file_id: str) -> bytes:
        """
        Download a file.
        Consider: A01 (access control), A09 (logging),
                  A10 (if file references external URL)
        """
        pass

    def fetch_external_file(self, url: str) -> bytes:
        """
        Fetch a file from an external URL.
        Consider: A10 (SSRF), A06 (validate URL library)
        """
        pass
```

### Exercise 3: Security Audit Report

Perform a simulated security audit:

```
Exercise: Given a web application with these characteristics:
- Python/Flask backend
- PostgreSQL database
- JWT authentication
- File upload feature
- Webhook integration (fetches external URLs)
- Uses 15 Python dependencies (not audited in 6 months)
- Logs written to local files only
- No rate limiting
- Running on AWS EC2

For each OWASP Top 10 category:
1. Identify specific risks for this application
2. Rate the risk (Critical/High/Medium/Low)
3. Propose concrete remediation steps
4. Estimate implementation effort

Write your findings as a structured security audit report.
```

### Exercise 4: Fix the Vulnerable Application

Take the code from Exercise 1 and rewrite it with all vulnerabilities fixed. Your fixed version should address every OWASP Top 10 category.

### Exercise 5: OWASP Top 10 Mapping

Map the following real-world breaches to OWASP Top 10 categories:

```
1. Equifax (2017) - Unpatched Apache Struts vulnerability
   → A0?: ___

2. Capital One (2019) - SSRF to access AWS metadata
   → A0?: ___

3. SolarWinds (2020) - Compromised build pipeline
   → A0?: ___

4. Facebook (2019) - 540M user records in unprotected S3
   → A0?: ___

5. Uber (2016) - Hardcoded AWS credentials in GitHub repo
   → A0?: ___

6. British Airways (2018) - Magecart XSS in payment page
   → A0?: ___

7. Marriott (2018) - Breach undetected for 4 years
   → A0?: ___

8. Yahoo (2013-2014) - Weak/no encryption on user data
   → A0?: ___
```

---

## 14. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                OWASP Top 10 (2021) Summary                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  A01: Broken Access Control - #1 threat. Default-deny.          │
│  A02: Cryptographic Failures - Encrypt everything sensitive.    │
│  A03: Injection - Parameterize all queries.                     │
│  A04: Insecure Design - Security starts at design, not code.   │
│  A05: Security Misconfiguration - Harden everything.            │
│  A06: Vulnerable Components - Know and update your deps.        │
│  A07: Auth Failures - MFA, rate limiting, strong passwords.     │
│  A08: Integrity Failures - Sign everything, don't pickle.       │
│  A09: Logging Failures - Log security events, alert, respond.   │
│  A10: SSRF - Validate all URLs, segment networks.               │
│                                                                  │
│  The OWASP Top 10 is a starting point, not an exhaustive list.  │
│  Use it as a minimum baseline for application security.          │
│                                                                  │
│  Resources:                                                      │
│  - https://owasp.org/Top10/                                     │
│  - OWASP Application Security Verification Standard (ASVS)     │
│  - OWASP Testing Guide                                          │
│  - OWASP Cheat Sheet Series                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [06. Authorization and Access Control](06_Authorization.md) | **Next**: [08. Injection Attacks and Prevention](08_Injection_Attacks.md)
