# API 보안

**이전**: [09_Web_Security_Headers.md](./09_Web_Security_Headers.md) | **다음**: [11_Secrets_Management.md](./11_Secrets_Management.md)

---

API는 현대 소프트웨어 시스템의 핵심입니다. 마이크로서비스를 연결하고, 모바일 애플리케이션을 구동하며, 타사 통합을 가능하게 합니다. API는 애플리케이션 로직과 데이터를 직접 노출하기 때문에 공격자의 주요 표적입니다. 단일 엔드포인트의 잘못된 구성으로 수백만 개의 레코드가 노출될 수 있습니다. 이 레슨은 API를 구축, 배포 및 유지하기 위한 필수 보안 관행을 다룹니다 — 인증 및 속도 제한부터 입력 검증 및 CORS 구성까지.

## 학습 목표

- API 키, OAuth 2.0 및 JWT를 사용한 강력한 API 인증 구현
- 남용을 방지하기 위한 속도 제한 전략 설계 및 배포
- 주입 공격을 방지하기 위한 모든 API 입력 검증 및 정제
- cross-origin 접근을 제어하기 위한 CORS 올바르게 구성
- 일반적인 공격 패턴으로부터 GraphQL 엔드포인트 보호
- OpenAPI/Swagger 명세에서 보안 스킴 정의
- 프로덕션 환경을 위한 API 게이트웨이 보안 패턴 적용

---

## 1. API 위협 환경

### 1.1 OWASP API 보안 Top 10 (2023)

```
┌─────────────────────────────────────────────────────────────────────┐
│                 OWASP API 보안 Top 10 (2023)                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  API1  BOLA (Broken Object Level Authorization)                     │
│        다른 사용자의 리소스 접근                                     │
│        GET /api/users/OTHER_USER_ID/orders                           │
│                                                                      │
│  API2  Broken Authentication                                         │
│        약한 인증 메커니즘, 토큰 유출                                 │
│                                                                      │
│  API3  Broken Object Property Level Authorization                   │
│        대량 할당, 민감한 속성 노출                                   │
│                                                                      │
│  API4  Unrestricted Resource Consumption                            │
│        속도 제한 없음, 큰 페이로드, 비용이 많이 드는 쿼리            │
│                                                                      │
│  API5  Broken Function Level Authorization                          │
│        일반 사용자로서 관리 기능 접근                                │
│        POST /api/admin/users/delete                                  │
│                                                                      │
│  API6  Unrestricted Access to Sensitive Business Flows              │
│        비즈니스 기능의 자동화된 남용                                 │
│                                                                      │
│  API7  Server-Side Request Forgery (SSRF)                           │
│        검증 없이 사용자 입력에서 URL 가져오기                        │
│                                                                      │
│  API8  Security Misconfiguration                                    │
│        헤더 누락, 자세한 오류, 기본 자격 증명                        │
│                                                                      │
│  API9  Improper Inventory Management                                │
│        관리되지 않는 API 버전, 섀도우 API                            │
│                                                                      │
│  API10 Unsafe Consumption of APIs                                   │
│        타사 API 응답을 맹목적으로 신뢰                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 API 공격 표면

```
┌─────────────────────────────────────────────────────────────────────┐
│                    API 공격 표면                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │  Client  │───▶│ Network  │───▶│ API      │───▶│ Database │      │
│  │          │    │          │    │ Server   │    │          │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│                                                                      │
│  각 계층의 공격 벡터:                                                │
│                                                                      │
│  Client:     변조된 요청, 도난당한 토큰, 재생 공격                   │
│  Network:    중간자 공격, 도청, DNS 하이재킹                         │
│  API Server: 주입, 인증 손상, BOLA, 대량 할당                       │
│  Database:   SQL 주입, 데이터 유출, 무단 접근                        │
│                                                                      │
│  포괄적인 우려 사항:                                                 │
│  ├── 인증         (당신은 누구인가?)                                │
│  ├── 인가         (무엇에 접근할 수 있나?)                          │
│  ├── 입력 검증    (이 요청은 안전한가?)                             │
│  ├── 속도 제한    (API를 남용하고 있나?)                            │
│  ├── 암호화        (전송 중 데이터가 보호되는가?)                    │
│  └── 로깅/모니터링 (공격을 감지할 수 있나?)                          │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. API 인증 패턴

### 2.1 API 키

```python
"""
API 키 인증 — 단순하지만 제한적.
서버 간 통신 및 사용 추적이 있는 공개 API에 적합.
"""
import secrets
import hashlib
from flask import Flask, request, jsonify, abort
from functools import wraps
from datetime import datetime

app = Flask(__name__)

# ── API 키 생성 ───────────────────────────────────────────
def generate_api_key() -> tuple[str, str]:
    """API 키와 저장용 해시를 생성."""
    # 암호학적으로 안전한 키 생성
    # 접두사는 키 유형과 버전을 식별하는 데 도움
    raw_key = f"sk_live_{secrets.token_urlsafe(32)}"

    # 원시 키는 절대 저장하지 말고 해시만 저장
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    return raw_key, key_hash

# raw_key: sk_live_Abc123...  (클라이언트에 한 번 전송, 저장 안 함)
# key_hash: e3b0c4...         (데이터베이스에 저장)


# ── 시뮬레이션된 키 저장소 (프로덕션에서는 실제 데이터베이스 사용) ────
API_KEYS = {
    # 해시 -> 메타데이터
    hashlib.sha256(b"sk_live_test_key_12345").hexdigest(): {
        "client_id": "client_001",
        "name": "Test Client",
        "rate_limit": 100,       # 분당 요청 수
        "scopes": ["read", "write"],
        "created_at": "2025-01-01",
        "last_used": None,
    }
}


def require_api_key(f):
    """유효한 API 키를 요구하는 데코레이터."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # API 키를 여러 위치에서 확인
        api_key = (
            request.headers.get('X-API-Key') or
            request.headers.get('Authorization', '').replace('Bearer ', '') or
            request.args.get('api_key')  # 덜 안전, 가능하면 피하기
        )

        if not api_key:
            return jsonify({
                "error": "missing_api_key",
                "message": "API 키가 필요합니다. "
                           "X-API-Key 헤더에 전달하세요."
            }), 401

        # 제공된 키를 해시하고 조회
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data = API_KEYS.get(key_hash)

        if not key_data:
            return jsonify({
                "error": "invalid_api_key",
                "message": "제공된 API 키가 유효하지 않습니다."
            }), 401

        # 마지막 사용 타임스탬프 업데이트
        key_data["last_used"] = datetime.utcnow().isoformat()

        # 요청 컨텍스트에 키 메타데이터 저장
        request.api_client = key_data
        return f(*args, **kwargs)

    return decorated


@app.route('/api/data')
@require_api_key
def get_data():
    """API 키가 필요한 보호된 엔드포인트."""
    client = request.api_client
    return jsonify({
        "client": client["name"],
        "data": [1, 2, 3]
    })


# ── API 키 보안 모범 사례 ─────────────────────────────────
"""
1. 전송:
   - 항상 HTTPS 사용
   - 쿼리 매개변수보다 헤더 선호 (쿼리 매개변수는 로그에 나타남)
   - X-API-Key 헤더 또는 Authorization: Bearer <key> 사용

2. 저장:
   - 저장하기 전에 키 해시 (최소 SHA-256)
   - 전체 API 키를 로그에 남기지 말 것
   - UI에는 마지막 4자만 표시: sk_live_****5678

3. 순환:
   - 클라이언트당 여러 활성 키 지원
   - 다운타임 없이 키 순환 허용
   - 키에 만료일 설정

4. 범위 지정:
   - 각 키에 범위/권한 할당
   - 읽기 대 쓰기 작업에 별도 키 사용
   - 테스트 대 프로덕션에 별도 키 사용

5. 제한 사항:
   - API 키는 사용자가 아닌 애플리케이션을 식별
   - 내장 만료가 부족 (토큰과 달리)
   - 요청별로 쉽게 범위를 지정할 수 없음
   - 사용자 컨텍스트 인가에는 OAuth 2.0 사용
"""
```

### 2.2 OAuth 2.0

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OAuth 2.0 Authorization Code 흐름                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 사용자가 "제공자로 로그인" 클릭                                  │
│     Client ──────────────────────────▶ Auth Server                   │
│     GET /authorize?response_type=code                                │
│         &client_id=CLIENT_ID                                         │
│         &redirect_uri=CALLBACK_URL                                   │
│         &scope=read+write                                            │
│         &state=RANDOM_STATE                                          │
│                                                                      │
│  2. 사용자 인증 및 범위 승인                                         │
│     Auth Server ─────────────────────▶ Client Callback               │
│     GET /callback?code=AUTH_CODE&state=RANDOM_STATE                  │
│                                                                      │
│  3. 클라이언트가 코드를 토큰으로 교환                                │
│     Client ──────────────────────────▶ Auth Server                   │
│     POST /token                                                      │
│         grant_type=authorization_code                                │
│         &code=AUTH_CODE                                              │
│         &client_id=CLIENT_ID                                         │
│         &client_secret=CLIENT_SECRET                                 │
│         &redirect_uri=CALLBACK_URL                                   │
│                                                                      │
│  4. 인증 서버가 토큰 반환                                            │
│     Auth Server ─────────────────────▶ Client                        │
│     { "access_token": "...",                                         │
│       "refresh_token": "...",                                        │
│       "token_type": "Bearer",                                        │
│       "expires_in": 3600 }                                           │
│                                                                      │
│  5. 클라이언트가 액세스 토큰 사용                                    │
│     Client ──────────────────────────▶ Resource Server               │
│     Authorization: Bearer ACCESS_TOKEN                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
"""
Flask를 사용한 OAuth 2.0 구현 (서버 측).
"""
import requests
import secrets
from flask import Flask, redirect, request, session, jsonify
from urllib.parse import urlencode

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)

# OAuth 2.0 구성 (예: GitHub)
OAUTH_CONFIG = {
    "client_id": "your_client_id",
    "client_secret": "your_client_secret",
    "authorize_url": "https://github.com/login/oauth/authorize",
    "token_url": "https://github.com/login/oauth/access_token",
    "api_url": "https://api.github.com/user",
    "redirect_uri": "http://localhost:5000/callback",
    "scope": "read:user user:email",
}


@app.route('/login')
def login():
    """OAuth 2.0 Authorization Code 흐름 시작."""
    # CSRF를 방지하기 위한 state 매개변수 생성
    state = secrets.token_urlsafe(32)
    session['oauth_state'] = state

    # 인가 URL 구축
    params = {
        "client_id": OAUTH_CONFIG["client_id"],
        "redirect_uri": OAUTH_CONFIG["redirect_uri"],
        "scope": OAUTH_CONFIG["scope"],
        "state": state,
        "response_type": "code",
    }

    auth_url = f"{OAUTH_CONFIG['authorize_url']}?{urlencode(params)}"
    return redirect(auth_url)


@app.route('/callback')
def callback():
    """OAuth 2.0 콜백 처리."""
    # ── state 매개변수 확인 ───────────────────────────────────
    state = request.args.get('state')
    if state != session.pop('oauth_state', None):
        return jsonify({"error": "유효하지 않은 state 매개변수"}), 400

    # ── 오류 응답 확인 ─────────────────────────────────────
    error = request.args.get('error')
    if error:
        return jsonify({
            "error": error,
            "description": request.args.get('error_description', '')
        }), 400

    # ── 인가 코드를 토큰으로 교환 ───────────────────────────
    code = request.args.get('code')
    token_response = requests.post(
        OAUTH_CONFIG["token_url"],
        data={
            "client_id": OAUTH_CONFIG["client_id"],
            "client_secret": OAUTH_CONFIG["client_secret"],
            "code": code,
            "redirect_uri": OAUTH_CONFIG["redirect_uri"],
            "grant_type": "authorization_code",
        },
        headers={"Accept": "application/json"},
        timeout=10,
    )
    token_data = token_response.json()

    if "error" in token_data:
        return jsonify(token_data), 400

    access_token = token_data["access_token"]

    # ── 사용자 정보 가져오기 ───────────────────────────────────
    user_response = requests.get(
        OAUTH_CONFIG["api_url"],
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        },
        timeout=10,
    )
    user_data = user_response.json()

    # 세션에 저장 (또는 데이터베이스에서 사용자 생성/업데이트)
    session['user'] = {
        "id": user_data["id"],
        "name": user_data.get("name", user_data["login"]),
        "email": user_data.get("email"),
    }

    return redirect('/dashboard')


# ── OAuth 2.0 보안 체크리스트 ────────────────────────────────
"""
1. 항상 state 매개변수 사용 (CSRF 방지)
2. redirect_uri를 정확히 검증 (오픈 리디렉트 방지)
3. 서버 앱에 Authorization Code 흐름 사용 (Implicit 아님)
4. 토큰을 안전하게 저장 (암호화, 서버 측)
5. 공개 클라이언트(SPA, 모바일 앱)에 PKCE 사용
6. 모든 요청에서 토큰 범위 검증
7. 단기 액세스 토큰 + 리프레시 토큰 사용
8. 로그아웃 시 토큰 폐기
"""
```

### 2.3 JWT (JSON Web Tokens)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    JWT 구조                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  eyJhbGci... . eyJzdWIi... . SflKxwRJ...                           │
│  ─────────     ─────────     ──────────                             │
│   Header        Payload       Signature                              │
│                                                                      │
│  Header (base64url):                                                 │
│  {                                                                   │
│    "alg": "RS256",                                                   │
│    "typ": "JWT",                                                     │
│    "kid": "key-id-123"                                               │
│  }                                                                   │
│                                                                      │
│  Payload (base64url):                                                │
│  {                                                                   │
│    "sub": "user_123",        // Subject (사용자 ID)                  │
│    "iss": "api.example.com", // Issuer                               │
│    "aud": "app.example.com", // Audience                             │
│    "exp": 1700000000,        // 만료 시간                            │
│    "iat": 1699996400,        // 발급 시간                            │
│    "nbf": 1699996400,        // Not before                           │
│    "jti": "unique-token-id", // JWT ID (폐기용)                      │
│    "scope": "read write",    // 사용자 정의 클레임                   │
│    "role": "admin"                                                   │
│  }                                                                   │
│                                                                      │
│  Signature:                                                          │
│  RSASHA256(base64url(header) + "." + base64url(payload), key)       │
│                                                                      │
│  중요: Payload는 암호화되지 않음 — base64url 인코딩만 됨            │
│  누구나 읽을 수 있음. JWT에 비밀을 저장하지 말 것.                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

```python
"""
PyJWT를 사용한 JWT 인증 — 안전한 구현.
"""
import jwt
import time
import uuid
from datetime import datetime, timedelta, timezone
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

# ── 키 구성 ────────────────────────────────────────────
# 프로덕션에는 RS256 (비대칭) 사용
# 개인 키는 토큰에 서명; 공개 키는 검증
# 이를 통해 마이크로서비스가 개인 키 없이 검증 가능

# 이 예제에서는 단순화를 위해 HS256 (대칭) 사용
JWT_SECRET = "your-256-bit-secret-change-this"  # 프로덕션에서 환경 변수 사용
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)


# ── 토큰 블랙리스트 (프로덕션에서 Redis 사용) ────────────────────
revoked_tokens = set()


def create_access_token(user_id: str, role: str = "user",
                        scopes: list[str] = None) -> str:
    """단기 액세스 토큰 생성."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iss": "api.example.com",
        "aud": "app.example.com",
        "iat": now,
        "exp": now + JWT_ACCESS_TOKEN_EXPIRES,
        "nbf": now,
        "jti": str(uuid.uuid4()),        # 폐기를 위한 고유 ID
        "type": "access",
        "role": role,
        "scopes": scopes or ["read"],
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """장기 리프레시 토큰 생성."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iss": "api.example.com",
        "iat": now,
        "exp": now + JWT_REFRESH_TOKEN_EXPIRES,
        "jti": str(uuid.uuid4()),
        "type": "refresh",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """JWT 토큰 디코드 및 검증."""
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer="api.example.com",
            audience="app.example.com",
            options={
                "require": ["exp", "iat", "sub", "iss", "aud", "jti"],
                "verify_exp": True,
                "verify_iss": True,
                "verify_aud": True,
                "verify_nbf": True,
            }
        )

        # 토큰이 폐기되었는지 확인
        if payload["jti"] in revoked_tokens:
            raise jwt.InvalidTokenError("토큰이 폐기되었습니다")

        return payload

    except jwt.ExpiredSignatureError:
        raise ValueError("토큰이 만료되었습니다")
    except jwt.InvalidAudienceError:
        raise ValueError("유효하지 않은 토큰 대상")
    except jwt.InvalidIssuerError:
        raise ValueError("유효하지 않은 토큰 발급자")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"유효하지 않은 토큰: {e}")


def require_auth(scopes: list[str] = None):
    """선택적 범위 확인과 함께 JWT 인증을 요구하는 데코레이터."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # Authorization 헤더에서 토큰 추출
            auth_header = request.headers.get('Authorization', '')
            if not auth_header.startswith('Bearer '):
                return jsonify({
                    "error": "missing_token",
                    "message": "Bearer 토큰이 있는 Authorization 헤더 필요"
                }), 401

            token = auth_header.split(' ', 1)[1]

            try:
                payload = decode_token(token)
            except ValueError as e:
                return jsonify({
                    "error": "invalid_token",
                    "message": str(e)
                }), 401

            # 토큰 유형 확인
            if payload.get("type") != "access":
                return jsonify({
                    "error": "wrong_token_type",
                    "message": "액세스 토큰 필요"
                }), 401

            # 범위 확인
            if scopes:
                token_scopes = set(payload.get("scopes", []))
                required_scopes = set(scopes)
                if not required_scopes.issubset(token_scopes):
                    missing = required_scopes - token_scopes
                    return jsonify({
                        "error": "insufficient_scope",
                        "message": f"누락된 범위: {', '.join(missing)}"
                    }), 403

            request.user = payload
            return f(*args, **kwargs)
        return decorated
    return decorator


# ── 라우트 ───────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST'])
def login():
    """인증하고 JWT 토큰 반환."""
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    # 자격 증명 검증 (bcrypt/argon2 해시 비교 사용)
    # 단순화됨 — 자세한 내용은 인증 레슨 참조
    user = authenticate_user(username, password)
    if not user:
        # 사용자 열거 방지를 위한 일관된 타이밍 사용
        return jsonify({
            "error": "invalid_credentials",
            "message": "유효하지 않은 사용자 이름 또는 비밀번호"
        }), 401

    # 토큰 쌍 생성
    access_token = create_access_token(
        user_id=user["id"],
        role=user["role"],
        scopes=user["scopes"]
    )
    refresh_token = create_refresh_token(user_id=user["id"])

    return jsonify({
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "Bearer",
        "expires_in": int(JWT_ACCESS_TOKEN_EXPIRES.total_seconds()),
    })


@app.route('/api/refresh', methods=['POST'])
def refresh():
    """리프레시 토큰을 새 액세스 토큰으로 교환."""
    data = request.get_json()
    refresh_token = data.get('refresh_token')

    if not refresh_token:
        return jsonify({"error": "missing_refresh_token"}), 400

    try:
        # 대상 확인 없이 디코드 (리프레시 토큰은 다를 수 있음)
        payload = jwt.decode(
            refresh_token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            issuer="api.example.com",
            options={"require": ["exp", "sub", "jti", "iss"]}
        )

        if payload.get("type") != "refresh":
            raise ValueError("리프레시 토큰이 아님")

        if payload["jti"] in revoked_tokens:
            raise ValueError("리프레시 토큰이 폐기되었습니다")

    except (jwt.InvalidTokenError, ValueError) as e:
        return jsonify({"error": "invalid_refresh_token",
                        "message": str(e)}), 401

    # 오래된 리프레시 토큰 폐기 (순환)
    revoked_tokens.add(payload["jti"])

    # 새 토큰 쌍 발급
    user_id = payload["sub"]
    # 데이터베이스에서 현재 사용자 역할/범위 조회
    user = get_user_by_id(user_id)

    new_access = create_access_token(
        user_id=user_id,
        role=user["role"],
        scopes=user["scopes"]
    )
    new_refresh = create_refresh_token(user_id=user_id)

    return jsonify({
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "Bearer",
        "expires_in": int(JWT_ACCESS_TOKEN_EXPIRES.total_seconds()),
    })


@app.route('/api/logout', methods=['POST'])
@require_auth()
def logout():
    """현재 액세스 토큰 폐기."""
    revoked_tokens.add(request.user["jti"])
    return jsonify({"message": "성공적으로 로그아웃했습니다"}), 200


@app.route('/api/protected')
@require_auth(scopes=["read"])
def protected_resource():
    """'read' 범위가 필요한 보호된 엔드포인트."""
    return jsonify({
        "user_id": request.user["sub"],
        "message": "이 보호된 리소스에 접근할 수 있습니다"
    })


# 완전성을 위한 플레이스홀더 함수
def authenticate_user(username, password):
    """플레이스홀더 — 적절한 비밀번호 해싱으로 구현."""
    return None

def get_user_by_id(user_id):
    """플레이스홀더 — 데이터베이스 조회로 구현."""
    return {"id": user_id, "role": "user", "scopes": ["read"]}
```

### 2.4 JWT 보안 모범 사례

```python
"""
JWT 보안 모범 사례 및 일반적인 함정.
"""

# ── 함정 1: 'none' 알고리즘 사용 ───────────────────────────
# 공격: 헤더를 {"alg": "none"}으로 변경하고 서명 제거
# 방어: 항상 허용된 알고리즘을 명시적으로 지정
payload = jwt.decode(
    token, key,
    algorithms=["RS256"],  # "none"을 절대 포함하거나 모두 허용하지 말 것
)

# ── 함정 2: HS256과 RS256 혼동 ────────────────────────────
# 공격: 서버가 RS256을 사용하는 경우, 공격자가:
#   1. 공개 키 가져오기 (공개임)
#   2. 공개 키를 비밀로 사용하여 HS256으로 새 토큰 서명
#   3. 서버가 공개 키를 HMAC 비밀로 취급
# 방어: 헤더의 알고리즘을 엄격하게 검증
payload = jwt.decode(
    token, rsa_public_key,
    algorithms=["RS256"],  # 예상된 알고리즘만 허용
)

# ── 함정 3: 만료 누락 ───────────────────────────────────
# exp 클레임이 없는 토큰은 영원히 유효
# 방어: 항상 짧은 만료 설정
payload = {
    "sub": "user_123",
    "exp": datetime.now(timezone.utc) + timedelta(minutes=15),
}

# ── 함정 4: 페이로드에 민감한 데이터 저장 ─────────────────
# JWT 페이로드는 base64url 인코딩되며 암호화되지 않음
# 키 없이도 누구나 디코드 가능
import base64
header, payload_b64, signature = token.split('.')
decoded = base64.urlsafe_b64decode(payload_b64 + '==')
# 전체 페이로드를 이제 읽을 수 있음!

# 절대 포함하지 말 것: 비밀번호, SSN, 신용카드, PII를 JWT에
# 포함할 것만: 사용자 ID, 역할, 범위, 만료

# ── 함정 5: 토큰 폐기 메커니즘 없음 ─────────────────────
# JWT는 자체 포함 — 서버가 무효화할 수 없음
# 해결책:
# 1. 단기 액세스 토큰 (15분) + 리프레시 토큰 순환
# 2. 토큰 블랙리스트 (TTL = 토큰 exp가 있는 Redis)
# 3. 토큰 버전 관리 (DB에 토큰 버전 저장, 각 요청에서 확인)
# 4. 서명 키 변경 (모든 토큰 무효화 — 극단적 옵션)

# ── RS256 설정 (프로덕션 권장) ────────────────────────────
"""
# RSA 키 쌍 생성
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -pubout -out public.pem

# 개인 키: 인증 서버가 토큰에 서명하는 데 사용
# 공개 키: 모든 서비스가 토큰을 검증하는 데 사용
# 공개 키를 자유롭게 공유; 개인 키 보호
"""

from cryptography.hazmat.primitives import serialization

# 키 로드
with open('private.pem', 'rb') as f:
    private_key = serialization.load_pem_private_key(f.read(), password=None)

with open('public.pem', 'rb') as f:
    public_key = serialization.load_pem_public_key(f.read())

# 개인 키로 서명
token = jwt.encode(payload, private_key, algorithm="RS256")

# 공개 키로 검증 (모든 서비스가 할 수 있음)
decoded = jwt.decode(token, public_key, algorithms=["RS256"])
```

---

## 3. 속도 제한

### 3.1 속도 제한 알고리즘

```
┌─────────────────────────────────────────────────────────────────────┐
│                    속도 제한 알고리즘                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 고정 윈도우                                                      │
│     ┌──────────┐┌──────────┐┌──────────┐                            │
│     │ Window 1 ││ Window 2 ││ Window 3 │                            │
│     │ ■■■■■    ││ ■■■      ││ ■■■■■■■  │                           │
│     │ 5/10     ││ 3/10     ││ 7/10     │  (제한: 윈도우당 10)      │
│     └──────────┘└──────────┘└──────────┘                            │
│     장점: 간단.  단점: 윈도우 경계에서 버스트 (2배 제한)             │
│                                                                      │
│  2. 슬라이딩 윈도우 로그                                             │
│     Time: ─────────[====현재 윈도우====]────────                     │
│     각 요청의 정확한 타임스탬프 추적                                 │
│     슬라이딩 윈도우 내 요청 카운트                                   │
│     장점: 정확.  단점: 메모리 집약적 (모든 타임스탬프 저장)          │
│                                                                      │
│  3. 슬라이딩 윈도우 카운터                                           │
│     고정 윈도우 카운트를 가중치 오버랩과 결합                        │
│     weight = (window_size - elapsed) / window_size                   │
│     count = prev_count * weight + current_count                      │
│     장점: 메모리 효율적.  단점: 근사치                               │
│                                                                      │
│  4. 토큰 버킷                                                        │
│     ┌─────────┐                                                      │
│     │ ● ● ● ● │ ← 버킷 (용량: 10 토큰)                             │
│     │ ● ● ●   │                                                     │
│     └─────────┘                                                      │
│         ↑                                                            │
│     리필: 초당 1 토큰                                                │
│     각 요청은 1 토큰 소비                                            │
│     장점: 버스트 허용.  단점: 관리할 상태가 더 많음                  │
│                                                                      │
│  5. 리키 버킷                                                        │
│     요청이 큐(버킷)에 들어감                                         │
│     고정 속도로 처리 (누출 속도)                                     │
│     오버플로는 거부됨                                                │
│     장점: 부드러운 출력 속도.  단점: 용량이 있어도 지연              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 Flask-Limiter를 사용한 Flask 속도 제한

```python
"""
Flask-Limiter를 사용한 속도 제한 구현.
pip install Flask-Limiter
"""
from flask import Flask, jsonify, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

# ── 기본 설정 ──────────────────────────────────────────────────
limiter = Limiter(
    app=app,
    key_func=get_remote_address,       # IP 주소로 속도 제한
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379",  # 분산용 Redis 사용
    # storage_uri="memory://",           # 개발용 인메모리
    strategy="fixed-window-elastic-expiry",
)


# ── 전역 속도 제한 (모든 라우트에 적용) ───────────────────
# 위 default_limits를 통해 이미 설정됨

# ── 라우트별 속도 제한 ────────────────────────────────────────
@app.route('/api/search')
@limiter.limit("10 per minute")
def search():
    """검색 엔드포인트 — 스크래핑 방지를 위한 엄격한 제한."""
    query = request.args.get('q', '')
    return jsonify({"query": query, "results": []})


@app.route('/api/login', methods=['POST'])
@limiter.limit("5 per minute")         # 무차별 대입 방지
def login():
    """공격적인 속도 제한이 있는 로그인 엔드포인트."""
    return jsonify({"message": "login"})


@app.route('/api/data')
@limiter.limit("100 per hour")
@limiter.limit("10 per minute")        # 여러 제한
def get_data():
    """계층화된 속도 제한이 있는 데이터 엔드포인트."""
    return jsonify({"data": []})


# ── API 키 티어에 따른 동적 속도 제한 ────────────────────────
def get_rate_limit_by_tier():
    """클라이언트의 API 티어에 따라 속도 제한 문자열 반환."""
    api_key = request.headers.get('X-API-Key', '')
    # 데이터베이스에서 티어 조회 (단순화됨)
    tiers = {
        "free": "100 per hour",
        "pro": "1000 per hour",
        "enterprise": "10000 per hour",
    }
    tier = get_tier_for_key(api_key)
    return tiers.get(tier, "50 per hour")  # 기본값은 free


@app.route('/api/premium')
@limiter.limit(get_rate_limit_by_tier)
def premium_endpoint():
    """티어 기반 속도 제한이 있는 엔드포인트."""
    return jsonify({"data": "premium"})


# ── 속도 제한 헤더 ──────────────────────────────────────────
# Flask-Limiter가 자동으로 이 헤더들을 추가:
# X-RateLimit-Limit: 100
# X-RateLimit-Remaining: 95
# X-RateLimit-Reset: 1699999999
# Retry-After: 60 (제한 초과 시)

# ── 사용자 정의 오류 핸들러 ────────────────────────────────────────
@app.errorhandler(429)
def ratelimit_handler(e):
    """속도 제한 초과 시 사용자 정의 응답."""
    return jsonify({
        "error": "rate_limit_exceeded",
        "message": "너무 많은 요청. 나중에 다시 시도하세요.",
        "retry_after": e.description,
    }), 429


# 플레이스홀더
def get_tier_for_key(api_key):
    return "free"
```

### 3.3 사용자 정의 토큰 버킷 구현

```python
"""
처음부터 토큰 버킷 속도 제한기 구현.
"""
import time
import threading
from dataclasses import dataclass, field

@dataclass
class TokenBucket:
    """토큰 버킷 속도 제한기.

    Args:
        capacity: 버킷의 최대 토큰 수
        refill_rate: 초당 추가되는 토큰
    """
    capacity: int
    refill_rate: float
    tokens: float = field(init=False)
    last_refill: float = field(init=False)
    lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_refill = time.monotonic()

    def _refill(self):
        """경과 시간에 따라 토큰 추가."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now

    def consume(self, tokens: int = 1) -> bool:
        """토큰 소비 시도. 허용되면 True 반환."""
        with self.lock:
            self._refill()
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def wait_time(self) -> float:
        """최소 1개의 토큰을 사용할 수 있을 때까지의 초 반환."""
        with self.lock:
            self._refill()
            if self.tokens >= 1:
                return 0.0
            return (1 - self.tokens) / self.refill_rate


class RateLimiterStore:
    """클라이언트 키별 속도 제한기 관리."""

    def __init__(self, capacity: int = 100, refill_rate: float = 10.0):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def get_bucket(self, key: str) -> TokenBucket:
        """주어진 키에 대한 토큰 버킷 가져오기 또는 생성."""
        if key not in self._buckets:
            with self._lock:
                if key not in self._buckets:
                    self._buckets[key] = TokenBucket(
                        capacity=self.capacity,
                        refill_rate=self.refill_rate
                    )
        return self._buckets[key]

    def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """주어진 키에 대한 요청이 허용되는지 확인."""
        bucket = self.get_bucket(key)
        return bucket.consume(tokens)


# ── Flask와 함께 사용 ─────────────────────────────────────────────
from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)
rate_limiter = RateLimiterStore(capacity=100, refill_rate=10.0)

def rate_limit(capacity=100, refill_rate=10.0):
    """사용자 정의 속도 제한 데코레이터."""
    store = RateLimiterStore(capacity=capacity, refill_rate=refill_rate)

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            # IP + 엔드포인트를 속도 제한 키로 사용
            client_ip = request.remote_addr
            key = f"{client_ip}:{request.endpoint}"

            bucket = store.get_bucket(key)
            if not bucket.consume():
                wait = bucket.wait_time()
                return jsonify({
                    "error": "rate_limit_exceeded",
                    "retry_after": round(wait, 2)
                }), 429

            response = f(*args, **kwargs)
            return response
        return decorated
    return decorator


@app.route('/api/resource')
@rate_limit(capacity=10, refill_rate=1.0)  # 10 버스트, 초당 1 지속
def get_resource():
    return jsonify({"data": "resource"})
```

---

## 4. 입력 검증 및 정제

### 4.1 검증 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                    입력 검증 계층                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Request ─┬── 계층 1: 스키마 검증                                    │
│           │   (구조, 타입, 필수 필드)                                │
│           │                                                          │
│           ├── 계층 2: 비즈니스 검증                                   │
│           │   (범위, 형식, 일관성)                                   │
│           │                                                          │
│           ├── 계층 3: 정제                                            │
│           │   (공백 제거, 정규화, 인코딩)                            │
│           │                                                          │
│           └── 계층 4: 매개변수화된 작업                              │
│               (SQL 매개변수화, 템플릿 이스케이프)                    │
│                                                                      │
│  원칙: 조기 검증, 빠른 실패, 클라이언트 데이터를 절대 신뢰하지 말 것. │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Marshmallow를 사용한 스키마 검증

```python
"""
Marshmallow 스키마를 사용한 API 입력 검증.
pip install marshmallow
"""
from marshmallow import (
    Schema, fields, validate, validates, validates_schema,
    ValidationError, pre_load, RAISE
)
from flask import Flask, request, jsonify

app = Flask(__name__)


# ── 스키마 정의 ────────────────────────────────────────────
class UserCreateSchema(Schema):
    """새 사용자 생성용 스키마."""

    class Meta:
        # 알 수 없는 필드에 대해 오류 발생 (대량 할당 방지)
        unknown = RAISE

    username = fields.String(
        required=True,
        validate=[
            validate.Length(min=3, max=30),
            validate.Regexp(
                r'^[a-zA-Z0-9_]+$',
                error="사용자 이름은 문자, 숫자, 밑줄만 포함해야 합니다"
            ),
        ]
    )
    email = fields.Email(required=True)
    password = fields.String(
        required=True,
        load_only=True,  # 직렬화된 출력에 절대 포함하지 않음
        validate=validate.Length(min=12, max=128),
    )
    age = fields.Integer(
        validate=validate.Range(min=13, max=150),
        load_default=None,
    )
    role = fields.String(
        validate=validate.OneOf(["user", "moderator"]),
        load_default="user",
        # 참고: "admin"은 API를 통해 허용되지 않음 — 데이터베이스를 통해서만
    )

    @validates('password')
    def validate_password_complexity(self, value):
        """비밀번호 복잡성 요구사항 적용."""
        errors = []
        if not any(c.isupper() for c in value):
            errors.append("최소 하나의 대문자를 포함해야 합니다")
        if not any(c.islower() for c in value):
            errors.append("최소 하나의 소문자를 포함해야 합니다")
        if not any(c.isdigit() for c in value):
            errors.append("최소 하나의 숫자를 포함해야 합니다")
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in value):
            errors.append("최소 하나의 특수 문자를 포함해야 합니다")
        if errors:
            raise ValidationError(errors)

    @pre_load
    def normalize_input(self, data, **kwargs):
        """검증 전에 입력 데이터 정규화."""
        if 'email' in data:
            data['email'] = data['email'].strip().lower()
        if 'username' in data:
            data['username'] = data['username'].strip()
        return data


class SearchQuerySchema(Schema):
    """검색 쿼리용 스키마."""

    class Meta:
        unknown = RAISE

    q = fields.String(
        required=True,
        validate=validate.Length(min=1, max=200),
    )
    page = fields.Integer(
        validate=validate.Range(min=1, max=1000),
        load_default=1,
    )
    per_page = fields.Integer(
        validate=validate.Range(min=1, max=100),
        load_default=20,
    )
    sort = fields.String(
        validate=validate.OneOf(["relevance", "date", "name"]),
        load_default="relevance",
    )
    order = fields.String(
        validate=validate.OneOf(["asc", "desc"]),
        load_default="desc",
    )


# ── 검증 데코레이터 ────────────────────────────────────────
def validate_input(schema_class, location="json"):
    """스키마에 대해 요청 입력을 검증하는 데코레이터."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            schema = schema_class()

            if location == "json":
                data = request.get_json(silent=True)
                if data is None:
                    return jsonify({
                        "error": "invalid_request",
                        "message": "요청 본문은 유효한 JSON이어야 합니다"
                    }), 400
            elif location == "args":
                data = request.args.to_dict()
            elif location == "form":
                data = request.form.to_dict()
            else:
                raise ValueError(f"알 수 없는 위치: {location}")

            try:
                validated = schema.load(data)
            except ValidationError as err:
                return jsonify({
                    "error": "validation_error",
                    "messages": err.messages,
                }), 422

            request.validated_data = validated
            return f(*args, **kwargs)
        return decorated
    return decorator


# ── 검증 데코레이터 사용 ───────────────────────────────────
@app.route('/api/users', methods=['POST'])
@validate_input(UserCreateSchema, location="json")
def create_user():
    """검증된 입력으로 새 사용자 생성."""
    data = request.validated_data
    # 이 시점에서 data는 유효함이 보장됨
    return jsonify({
        "message": "사용자 생성됨",
        "username": data["username"],
        "email": data["email"],
    }), 201


@app.route('/api/search')
@validate_input(SearchQuerySchema, location="args")
def search():
    """검증된 쿼리 매개변수로 검색."""
    data = request.validated_data
    return jsonify({
        "query": data["q"],
        "page": data["page"],
        "per_page": data["per_page"],
    })
```

### 4.3 Pydantic 검증 (대안)

```python
"""
Pydantic v2를 사용한 API 검증.
pip install pydantic
"""
from pydantic import (
    BaseModel, Field, field_validator, model_validator,
    EmailStr, ConfigDict
)
from typing import Optional
import re

class UserCreate(BaseModel):
    """사용자 생성용 Pydantic 모델."""
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra='forbid',  # 알 수 없는 필드 거부
    )

    username: str = Field(
        min_length=3,
        max_length=30,
        pattern=r'^[a-zA-Z0-9_]+$',
    )
    email: EmailStr
    password: str = Field(min_length=12, max_length=128)
    age: Optional[int] = Field(default=None, ge=13, le=150)
    role: str = Field(default="user", pattern=r'^(user|moderator)$')

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not re.search(r'[A-Z]', v):
            raise ValueError('대문자를 포함해야 합니다')
        if not re.search(r'[a-z]', v):
            raise ValueError('소문자를 포함해야 합니다')
        if not re.search(r'\d', v):
            raise ValueError('숫자를 포함해야 합니다')
        if not re.search(r'[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]', v):
            raise ValueError('특수 문자를 포함해야 합니다')
        return v

    @field_validator('email')
    @classmethod
    def normalize_email(cls, v):
        return v.lower()


# ── Flask에서 사용 ───────────────────────────────────────────────
from flask import Flask, request, jsonify
from pydantic import ValidationError as PydanticValidationError

app = Flask(__name__)

@app.route('/api/users', methods=['POST'])
def create_user():
    try:
        user = UserCreate(**request.get_json())
    except PydanticValidationError as e:
        return jsonify({
            "error": "validation_error",
            "details": e.errors(),
        }), 422

    return jsonify({"username": user.username}), 201
```

---

## 5. CORS (Cross-Origin Resource Sharing)

### 5.1 CORS 작동 방식

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CORS 흐름                                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  간단한 요청 (GET, HEAD, 간단한 헤더가 있는 POST):                   │
│                                                                      │
│  Browser ────GET /api/data──────▶ Server                            │
│  (origin: https://app.com)                                          │
│                                                                      │
│  Server ────Response─────────────▶ Browser                           │
│  Access-Control-Allow-Origin: https://app.com                        │
│  ─── 브라우저가 응답 허용 ✓                                         │
│                                                                      │
│  ─────────────────────────────────────────────────────────────       │
│                                                                      │
│  프리플라이트 요청 (PUT, DELETE, 사용자 정의 헤더, JSON):            │
│                                                                      │
│  단계 1: 브라우저가 OPTIONS 프리플라이트 전송                        │
│  Browser ────OPTIONS /api/data───▶ Server                            │
│  Origin: https://app.com                                             │
│  Access-Control-Request-Method: PUT                                  │
│  Access-Control-Request-Headers: Content-Type, Authorization         │
│                                                                      │
│  단계 2: 서버가 허용된 메서드/헤더로 응답                            │
│  Server ────204 No Content───────▶ Browser                           │
│  Access-Control-Allow-Origin: https://app.com                        │
│  Access-Control-Allow-Methods: GET, POST, PUT, DELETE                │
│  Access-Control-Allow-Headers: Content-Type, Authorization           │
│  Access-Control-Max-Age: 86400                                       │
│                                                                      │
│  단계 3: 브라우저가 실제 요청 전송                                   │
│  Browser ────PUT /api/data───────▶ Server                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Flask CORS 구성

```python
"""
Flask에서 CORS 구성.
pip install flask-cors
"""
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)

# ── 옵션 1: 특정 출처 허용 (권장) ──────────────
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "https://app.example.com",
            "https://admin.example.com",
        ],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["X-Request-Id", "X-RateLimit-Remaining"],
        "supports_credentials": True,
        "max_age": 86400,
    }
})

# ── 옵션 2: 다른 라우트에 대한 다른 CORS ────────────────
app2 = Flask(__name__)

# 공개 API: 모든 출처 허용 (자격 증명 없음)
CORS(app2, resources={
    r"/api/public/*": {
        "origins": "*",
        "methods": ["GET"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": False,  # origin: *인 경우 False여야 함
    }
})

# 비공개 API: 자격 증명이 있는 특정 출처
CORS(app2, resources={
    r"/api/private/*": {
        "origins": ["https://app.example.com"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 86400,
    }
})

# ── 옵션 3: 수동 CORS 구현 ────────────────────────────
from flask import Flask, request, make_response

app3 = Flask(__name__)

ALLOWED_ORIGINS = {
    "https://app.example.com",
    "https://admin.example.com",
}

@app3.after_request
def add_cors_headers(response):
    origin = request.headers.get('Origin')

    if origin in ALLOWED_ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Methods'] = (
            'GET, POST, PUT, DELETE, OPTIONS'
        )
        response.headers['Access-Control-Allow-Headers'] = (
            'Content-Type, Authorization, X-Requested-With'
        )
        response.headers['Access-Control-Expose-Headers'] = (
            'X-Request-Id, X-RateLimit-Remaining'
        )
        response.headers['Access-Control-Max-Age'] = '86400'
        # Vary: Origin은 캐시에 응답이 출처에 따라 달라진다고 알림
        response.headers.add('Vary', 'Origin')

    return response

@app3.route('/api/data', methods=['OPTIONS'])
def preflight():
    """CORS 프리플라이트 요청 처리."""
    return make_response('', 204)


# ── CORS 보안 규칙 ─────────────────────────────────────────
"""
1. 자격 증명과 함께 Access-Control-Allow-Origin: *를 절대 사용하지 말 것
   - 실제로 브라우저에서 차단됨
   - 자격 증명이 필요한 경우 특정 출처 나열

2. 확인 없이 Origin 헤더를 Allow-Origin으로 절대 반영하지 말 것
   - 이는 모든 출처를 허용하는 것과 동일
   - 나쁨:  response.headers['ACAO'] = request.headers['Origin']
   - 좋음: 먼저 허용 목록에 대해 확인

3. Access-Control-Allow-Methods를 실제로 필요한 것으로 제한
   - 라우트가 지원하지 않는 경우 DELETE를 허용하지 말 것

4. 프리플라이트 요청을 줄이기 위해 Access-Control-Max-Age 설정
   - 86400 (24시간)이 합리적

5. 사용자 정의 헤더에 Access-Control-Expose-Headers 사용
   - 기본적으로 간단한 헤더만 JavaScript에서 읽을 수 있음

6. ACAO가 요청별로 변경되는 경우 항상 Vary: Origin 추가
   - 캐시 포이즈닝 방지
"""
```

---

## 6. GraphQL 보안

### 6.1 GraphQL 특정 위협

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GraphQL 보안 위협                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. 쿼리 깊이 공격                                                   │
│     query { user { posts { comments { author { posts { ... } } } } }│
│     ──▶ 깊이 중첩 쿼리가 기하급수적 DB 부하 유발                     │
│                                                                      │
│  2. 쿼리 폭 공격                                                     │
│     query { user1: user(id:1) {...} user2: user(id:2) {...} ... }   │
│     ──▶ 많은 별칭이 쿼리 비용을 곱함                                │
│                                                                      │
│  3. 인트로스펙션 남용                                                │
│     query { __schema { types { name fields { name } } } }           │
│     ──▶ 공격자에게 전체 API 스키마 노출                              │
│                                                                      │
│  4. 배치 공격                                                        │
│     [{"query": "..."}, {"query": "..."}, ... x 1000]                │
│     ──▶ 단일 요청에 여러 쿼리                                        │
│                                                                      │
│  5. 변수를 통한 주입                                                 │
│     query ($id: String!) { user(id: $id) { ... } }                  │
│     variables: { "id": "1 OR 1=1" }                                 │
│     ──▶ 검증되지 않은 변수를 통한 SQL 주입                           │
│                                                                      │
│  6. 정보 노출                                                        │
│     내부 세부 정보를 드러내는 자세한 오류 메시지                     │
│     필드 제안: "secretAdminField를 의미하셨나요?"                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 GraphQL 보안 완화

```python
"""
graphene (Python)을 사용한 GraphQL 보안 조치.
pip install graphene flask-graphql
"""

# ── 1. 쿼리 깊이 제한 ─────────────────────────────────────
class DepthAnalyzer:
    """GraphQL 쿼리 깊이 분석 및 제한."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth

    def analyze(self, query_ast, depth: int = 0) -> int:
        """쿼리의 최대 깊이 계산."""
        if depth > self.max_depth:
            raise ValueError(
                f"쿼리 깊이 {depth}가 허용된 최대값({self.max_depth})을 초과합니다"
            )

        max_child_depth = depth
        if hasattr(query_ast, 'selection_set') and query_ast.selection_set:
            for selection in query_ast.selection_set.selections:
                child_depth = self.analyze(selection, depth + 1)
                max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth


# ── 2. 쿼리 비용 분석 ──────────────────────────────────────
class QueryCostAnalyzer:
    """GraphQL 쿼리의 비용 추정."""

    # 필드 유형별 비용 정의
    FIELD_COSTS = {
        "user": 1,
        "posts": 5,       # 리스트 필드, 잠재적으로 비쌈
        "comments": 3,
        "search": 10,     # 전체 텍스트 검색은 비쌈
    }

    def __init__(self, max_cost: int = 1000):
        self.max_cost = max_cost

    def calculate_cost(self, query_ast, multiplier: int = 1) -> int:
        """예상 쿼리 비용 계산."""
        total_cost = 0

        if hasattr(query_ast, 'selection_set') and query_ast.selection_set:
            for selection in query_ast.selection_set.selections:
                field_name = selection.name.value
                field_cost = self.FIELD_COSTS.get(field_name, 1)

                # 비용을 곱하는 페이지네이션 인수 확인
                args = {
                    arg.name.value: arg.value.value
                    for arg in (selection.arguments or [])
                    if hasattr(arg.value, 'value')
                }
                limit = int(args.get('first', args.get('limit', 1)))

                cost = field_cost * multiplier * max(limit, 1)
                total_cost += cost

                # 자식 선택으로 재귀
                total_cost += self.calculate_cost(selection, limit)

        if total_cost > self.max_cost:
            raise ValueError(
                f"쿼리 비용 {total_cost}가 최대값({self.max_cost})을 초과합니다"
            )

        return total_cost


# ── 3. 프로덕션에서 인트로스펙션 비활성화 ──────────────────────
"""
인트로스펙션은 전체 API 스키마를 드러냅니다.
프로덕션에서는 비활성화하세요.
"""
from graphql import GraphQLError

class DisableIntrospection:
    """프로덕션에서 인트로스펙션 쿼리를 비활성화하는 미들웨어."""

    def resolve(self, next, root, info, **kwargs):
        # __schema 및 __type 쿼리 차단
        if info.field_name in ('__schema', '__type'):
            raise GraphQLError("인트로스펙션이 비활성화되었습니다")
        return next(root, info, **kwargs)


# ── 4. 속도 제한 + 배치 제한 ──────────────────────────────
from flask import Flask, request, jsonify

app = Flask(__name__)

MAX_BATCH_SIZE = 5  # 배치 요청당 최대 쿼리 수

@app.before_request
def limit_batch_queries():
    """배치 요청의 쿼리 수 제한."""
    if request.is_json:
        data = request.get_json(silent=True)
        if isinstance(data, list):
            if len(data) > MAX_BATCH_SIZE:
                return jsonify({
                    "error": "batch_limit_exceeded",
                    "message": f"배치당 최대 {MAX_BATCH_SIZE}개 쿼리"
                }), 400


# ── 5. 지속 쿼리 ────────────────────────────────────────
"""
임의의 쿼리를 수락하는 대신, 사전 등록된
쿼리 해시만 수락합니다. 이는 주입 및 쿼리 조작을 방지합니다.
"""
import hashlib

# 사전 등록된 쿼리 (빌드 타임에 생성)
PERSISTED_QUERIES = {
    "abc123": "query { users { id name } }",
    "def456": "query GetUser($id: ID!) { user(id: $id) { id name email } }",
}

@app.route('/graphql', methods=['POST'])
def graphql_endpoint():
    data = request.get_json()

    # 프로덕션에서는 지속 쿼리만 수락
    query_hash = data.get('extensions', {}).get('persistedQuery', {}).get('sha256Hash')

    if query_hash:
        query = PERSISTED_QUERIES.get(query_hash)
        if not query:
            return jsonify({"error": "query_not_found"}), 404
    else:
        # 개발에서는 임의의 쿼리 허용
        # 프로덕션에서는 거부:
        return jsonify({
            "error": "persisted_queries_only",
            "message": "지속 쿼리만 수락됩니다"
        }), 400

    # 쿼리 실행...
    return jsonify({"data": {}})
```

---

## 7. API 게이트웨이 보안

### 7.1 게이트웨이 아키텍처

```
┌─────────────────────────────────────────────────────────────────────┐
│                    API 게이트웨이 보안 아키텍처                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Client                                                              │
│    │                                                                 │
│    ▼                                                                 │
│  ┌────────────────────────────────────────────────┐                  │
│  │              API 게이트웨이                      │                 │
│  │                                                 │                 │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │                 │
│  │  │   TLS    │ │   인증   │ │  속도    │       │                 │
│  │  │  종료    │ │  검증    │ │  제한    │       │                 │
│  │  └──────────┘ └──────────┘ └──────────┘       │                 │
│  │                                                 │                 │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │                 │
│  │  │  입력    │ │  요청    │ │  CORS    │       │                 │
│  │  │  검증    │ │  로깅    │ │  처리    │       │                 │
│  │  └──────────┘ └──────────┘ └──────────┘       │                 │
│  │                                                 │                 │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐       │                 │
│  │  │  WAF     │ │ IP 허용/ │ │  응답    │       │                 │
│  │  │  규칙    │ │ 차단 목록│ │  필터링  │       │                 │
│  │  └──────────┘ └──────────┘ └──────────┘       │                 │
│  └────────────────────────────────────────────────┘                  │
│           │              │              │                             │
│           ▼              ▼              ▼                             │
│    ┌──────────┐   ┌──────────┐   ┌──────────┐                       │
│    │서비스 A  │   │서비스 B  │   │서비스 C  │                       │
│    │(사용자)  │   │(주문)    │   │(검색)    │                       │
│    └──────────┘   └──────────┘   └──────────┘                       │
│                                                                      │
│  중앙 게이트웨이의 보안 이점:                                        │
│  • 인증 적용을 위한 단일 지점                                        │
│  • 서비스 전반에 걸친 일관된 속도 제한                               │
│  • 중앙 집중식 로깅 및 모니터링                                      │
│  • 백엔드 서비스가 TLS를 처리하지 않음                               │
│  • 단순화된 보안 정책 관리                                           │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 요청/응답 필터링

```python
"""
API 게이트웨이 요청/응답 필터링 미들웨어.
"""
from flask import Flask, request, jsonify, g
import re
import uuid
import time
import logging

app = Flask(__name__)
logger = logging.getLogger('api_gateway')


# ── 요청 ID 추적 ─────────────────────────────────────────
@app.before_request
def add_request_id():
    """추적을 위한 고유 요청 ID 추가."""
    g.request_id = request.headers.get(
        'X-Request-Id',
        str(uuid.uuid4())
    )
    g.request_start = time.monotonic()


@app.after_request
def add_response_headers(response):
    """응답에 보안 및 추적 헤더 추가."""
    response.headers['X-Request-Id'] = g.request_id
    # 타이밍 추가
    elapsed = time.monotonic() - g.request_start
    response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
    return response


# ── 요청 크기 제한 ───────────────────────────────────────
MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1 MB
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

@app.before_request
def check_content_length():
    """과도한 크기의 요청 거부."""
    content_length = request.content_length
    if content_length and content_length > MAX_CONTENT_LENGTH:
        return jsonify({
            "error": "payload_too_large",
            "max_bytes": MAX_CONTENT_LENGTH,
        }), 413


# ── IP 허용 목록/차단 목록 ──────────────────────────────────────
BLOCKED_IPS = {"192.168.1.100", "10.0.0.50"}
ADMIN_ALLOWED_IPS = {"10.0.0.1", "10.0.0.2"}

@app.before_request
def check_ip():
    """금지된 IP로부터의 요청 차단."""
    client_ip = request.remote_addr

    if client_ip in BLOCKED_IPS:
        logger.warning(f"금지된 IP에서 요청 차단: {client_ip}")
        return jsonify({"error": "forbidden"}), 403

    # 관리자 라우트는 특정 IP 필요
    if request.path.startswith('/admin/'):
        if client_ip not in ADMIN_ALLOWED_IPS:
            return jsonify({"error": "forbidden"}), 403


# ── 응답 데이터 필터링 ─────────────────────────────────────
SENSITIVE_FIELDS = {'password', 'ssn', 'credit_card', 'secret_key',
                    'token', 'api_key'}

def filter_sensitive_data(data):
    """응답 데이터에서 민감한 필드를 재귀적으로 제거."""
    if isinstance(data, dict):
        return {
            k: filter_sensitive_data(v)
            for k, v in data.items()
            if k.lower() not in SENSITIVE_FIELDS
        }
    elif isinstance(data, list):
        return [filter_sensitive_data(item) for item in data]
    return data


# ── 감사 로깅 ───────────────────────────────────────────────
@app.after_request
def audit_log(response):
    """감사 추적을 위한 모든 API 요청 로그."""
    logger.info(
        "API 요청: method=%s path=%s status=%s "
        "ip=%s user_agent=%s request_id=%s duration=%s",
        request.method,
        request.path,
        response.status_code,
        request.remote_addr,
        request.user_agent.string[:100],
        g.request_id,
        response.headers.get('X-Response-Time', 'N/A'),
    )
    return response
```

---

## 8. OpenAPI 보안 정의

### 8.1 OpenAPI 3.0의 보안 스킴

```yaml
# openapi.yaml - 보안 스킴 정의
openapi: 3.0.3
info:
  title: Secure API
  version: 1.0.0

# ── 보안 스킴 정의 ──────────────────────────────────
components:
  securitySchemes:
    # API 키 인증
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: 서버 간 통신을 위한 API 키

    # JWT Bearer 토큰
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
      description: JWT 액세스 토큰

    # OAuth 2.0
    OAuth2:
      type: oauth2
      flows:
        authorizationCode:
          authorizationUrl: https://auth.example.com/authorize
          tokenUrl: https://auth.example.com/token
          refreshUrl: https://auth.example.com/refresh
          scopes:
            read: 리소스 읽기 접근
            write: 리소스 쓰기 접근
            admin: 관리 접근

    # OpenID Connect
    OpenIdConnect:
      type: openIdConnect
      openIdConnectUrl: https://auth.example.com/.well-known/openid-configuration

# ── 전역 보안 (모든 엔드포인트에 적용) ──────────────────
security:
  - BearerAuth: []

# ── 엔드포인트별 보안 ────────────────────────────────────────
paths:
  /api/public/status:
    get:
      summary: 상태 확인 (인증 불필요)
      security: []  # 재정의: 인증 없음
      responses:
        '200':
          description: OK

  /api/users:
    get:
      summary: 사용자 목록
      security:
        - BearerAuth: []
        - ApiKeyAuth: []  # 대안 인증 (OR)
      responses:
        '200':
          description: 사용자 목록

  /api/admin/settings:
    put:
      summary: 설정 업데이트 (관리자만)
      security:
        - OAuth2: [admin]  # 'admin' 범위 필요
      responses:
        '200':
          description: 설정 업데이트됨

  /api/data:
    post:
      summary: 데이터 생성 (read + write 필요)
      security:
        - OAuth2: [read, write]  # 두 범위 모두 필요
      responses:
        '201':
          description: 데이터 생성됨
```

### 8.2 API 버전 관리 보안

```python
"""
보안을 위한 API 버전 관리 고려 사항.
"""

# ── URL 경로 버전 관리 ──────────────────────────────────────────
# /api/v1/users  (구버전, 알려진 취약점이 있을 수 있음)
# /api/v2/users  (현재, 패치됨)

# ── 헤더 버전 관리 ───────────────────────────────────────────
# Accept: application/vnd.example.v2+json

# ── 버전 관리와 관련된 보안 문제 ───────────────────────────────────────
"""
1. 구 API 버전에는 알려진 취약점이 있을 수 있음
   - 폐기 날짜를 설정하고 적용
   - Deprecation 및 Sunset 헤더 반환

2. 폐기된 버전에 대해 보안 패치를 유지하지 말 것
   - 최신 버전으로 강제 마이그레이션

3. 폐기된 버전의 사용 모니터링
   - 구버전이 여전히 사용 중일 때 경고
"""

from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

API_VERSIONS = {
    "v1": {
        "status": "deprecated",
        "sunset": "2025-06-01",
        "successor": "v2",
    },
    "v2": {
        "status": "current",
        "sunset": None,
        "successor": None,
    },
}

@app.before_request
def check_api_version():
    """폐기된 API 버전 경고 또는 차단."""
    # URL 경로에서 버전 추출
    path_parts = request.path.strip('/').split('/')
    if len(path_parts) >= 2 and path_parts[0] == 'api':
        version = path_parts[1]
        version_info = API_VERSIONS.get(version)

        if not version_info:
            return jsonify({
                "error": "invalid_version",
                "supported": list(API_VERSIONS.keys()),
            }), 400

        if version_info["status"] == "deprecated":
            sunset_date = version_info["sunset"]
            if sunset_date:
                # 일몰 날짜가 지났는지 확인
                if datetime.now() > datetime.fromisoformat(sunset_date):
                    return jsonify({
                        "error": "version_retired",
                        "message": f"API {version}은(는) {sunset_date}에 폐기되었습니다",
                        "upgrade_to": version_info["successor"],
                    }), 410  # 410 Gone


@app.after_request
def add_deprecation_headers(response):
    """응답 헤더에 폐기 경고 추가."""
    path_parts = request.path.strip('/').split('/')
    if len(path_parts) >= 2 and path_parts[0] == 'api':
        version = path_parts[1]
        version_info = API_VERSIONS.get(version, {})

        if version_info.get("status") == "deprecated":
            response.headers['Deprecation'] = 'true'
            if version_info.get("sunset"):
                response.headers['Sunset'] = version_info["sunset"]
            response.headers['Link'] = (
                f'</api/{version_info["successor"]}>; rel="successor-version"'
            )

    return response
```

---

## 9. 요청/응답 암호화

### 9.1 전송 계층 보안

```python
"""
TLS 구성 및 인증서 고정.
"""

# ── TLS를 사용한 Flask (개발) ────────────────────────────────
# 개발용 자체 서명 인증서 생성:
# openssl req -x509 -newkey rsa:4096 -nodes \
#   -out cert.pem -keyout key.pem -days 365

from flask import Flask
app = Flask(__name__)

# TLS로 실행 (개발 전용)
# 프로덕션에서는 TLS가 역방향 프록시(nginx, 로드 밸런서)에서 처리됨
if __name__ == '__main__':
    app.run(
        ssl_context=('cert.pem', 'key.pem'),
        host='0.0.0.0',
        port=443,
    )

# ── 인증서 고정 (API 클라이언트용) ───────────────────────────
import requests
import hashlib
import ssl

def verify_certificate_pin(host: str, expected_pin: str) -> bool:
    """서버 인증서가 예상 핀과 일치하는지 확인."""
    import socket

    context = ssl.create_default_context()
    with socket.create_connection((host, 443)) as sock:
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            cert_der = ssock.getpeercert(True)
            cert_hash = hashlib.sha256(cert_der).hexdigest()
            return cert_hash == expected_pin


# ── 페이로드 수준 암호화 (민감한 필드용) ─────────────────────────
from cryptography.fernet import Fernet

# 키 생성 (코드가 아닌 안전하게 저장)
encryption_key = Fernet.generate_key()
cipher = Fernet(encryption_key)

def encrypt_sensitive_fields(data: dict, fields: list[str]) -> dict:
    """응답 페이로드에서 특정 필드 암호화."""
    result = data.copy()
    for field in fields:
        if field in result:
            value = str(result[field]).encode()
            result[field] = cipher.encrypt(value).decode()
    return result

def decrypt_sensitive_fields(data: dict, fields: list[str]) -> dict:
    """요청 페이로드에서 특정 필드 복호화."""
    result = data.copy()
    for field in fields:
        if field in result:
            value = result[field].encode()
            result[field] = cipher.decrypt(value).decode()
    return result
```

---

## 10. 연습 문제

### 연습 1: 안전한 JWT 인증 서비스

다음을 포함하는 Flask를 사용한 완전한 JWT 인증 서비스 구축:

1. 비밀번호 해싱이 있는 사용자 등록 (argon2 또는 bcrypt)
2. 액세스 + 리프레시 토큰을 반환하는 로그인 엔드포인트
3. 리프레시 토큰 순환이 있는 토큰 리프레시 엔드포인트
4. 토큰 블랙리스트가 있는 로그아웃 엔드포인트 (Redis 또는 인메모리 세트 사용)
5. "admin" 범위가 필요한 보호된 엔드포인트
6. 만료, 잘못된 형식 및 폐기된 토큰에 대한 적절한 오류 처리
7. 로그인 엔드포인트에 대한 속도 제한 (IP당 분당 5회 시도)

### 연습 2: CORS 보안 감사

다음을 수행하는 Python 스크립트 작성:

1. API URL 목록 가져오기
2. 다양한 Origin 헤더로 요청 전송
3. API가 임의의 출처를 반영하는지 테스트 (취약점)
4. 자격 증명이 와일드카드 출처와 함께 허용되는지 확인
5. 일반적인 메서드 및 헤더에 대한 프리플라이트 처리 테스트
6. 각 엔드포인트에 대한 보안 보고서 생성

### 연습 3: GraphQL 보안 미들웨어

다음을 수행하는 GraphQL 보안 미들웨어 구현:

1. 구성 가능한 최대값으로 쿼리 깊이 제한 (기본값: 10)
2. 쿼리 비용 계산 및 비용이 많이 드는 쿼리 거부
3. 프로덕션에서 인트로스펙션 비활성화
4. 배치 쿼리 크기 제한
5. 비용 및 실행 시간과 함께 모든 쿼리 로그
6. 해시 허용 목록이 있는 지속 쿼리 구현

### 연습 4: 슬라이딩 윈도우가 있는 속도 제한기

다음을 수행하는 슬라이딩 윈도우 로그 속도 제한기 구현:

1. Redis를 백업 저장소로 사용 (또는 인메모리 시뮬레이션)
2. 각 요청의 정확한 타임스탬프 추적
3. 구성 가능한 윈도우 지원 (초당, 분당, 시간당)
4. 다른 API 키 티어에 대한 다른 제한 지원
5. 적절한 속도 제한 헤더 반환 (X-RateLimit-Limit, Remaining, Reset)
6. 분산 배포 처리 (여러 서버 인스턴스)

### 연습 5: API 입력 검증 프레임워크

다음을 수행하는 재사용 가능한 검증 프레임워크 구축:

1. 스키마에 대한 JSON 요청 본문 검증
2. 타입 강제를 사용한 쿼리 매개변수 검증
3. 경로 매개변수 검증
4. 중첩 객체 검증 지원
5. 일관된 오류 응답 형식 반환 (RFC 7807)
6. 대량 할당으로부터 보호 (알 수 없는 필드 거부)
7. 문자열 입력 정제 (공백 제거, 유니코드 정규화)

### 연습 6: API 보안 스캐너

다음에 대한 API를 테스트하는 보안 스캐닝 도구 생성:

1. 엔드포인트에서 인증 누락
2. 깨진 객체 수준 인가 (BOLA/IDOR)
3. 속도 제한 누락
4. 내부 세부 정보를 드러내는 자세한 오류 메시지
5. 보안 헤더 누락
6. CORS 잘못된 구성
7. 심각도 등급이 있는 보고서 생성

---

## 요약

### API 보안 체크리스트

| 범주 | 항목 | 우선순위 |
|----------|------|----------|
| 인증 | 사용자에게 OAuth 2.0 또는 JWT 사용 (API 키만 사용하지 말 것) | Critical |
| 인증 | 단기 액세스 토큰 (15분) | Critical |
| 인증 | 리프레시 토큰 순환 | High |
| 인가 | 객체 수준 권한 확인 (BOLA 방지) | Critical |
| 인가 | 모든 요청에서 범위 검증 | Critical |
| 속도 제한 | IP당 및 사용자당 속도 제한 | High |
| 속도 제한 | 인증 엔드포인트에 대한 엄격한 제한 | High |
| 입력 검증 | 모든 입력에 대한 스키마 검증 | Critical |
| 입력 검증 | 알 수 없는 필드 거부 | High |
| CORS | 특정 출처 허용 목록 (자격 증명과 함께 와일드카드 없음) | Critical |
| CORS | Vary: Origin 헤더 | Medium |
| 암호화 | 어디서나 TLS (HTTPS만) | Critical |
| 로깅 | 고유한 요청 ID로 모든 요청 로그 | High |
| 버전 관리 | 일몰 날짜로 구버전 폐기 | Medium |
| 게이트웨이 | 중앙 집중식 인증 및 속도 제한 | Recommended |

### 핵심 요점

1. **심층 방어** — 모든 계층(네트워크, 게이트웨이, 애플리케이션, 데이터베이스)에서 보안 적용
2. **클라이언트 입력을 절대 신뢰하지 말 것** — 서버에서 모든 것을 매번 검증
3. **확립된 라이브러리 사용** — 자체 인증 또는 암호화를 만들지 말 것
4. **모니터링 및 경고** — 모니터링 없는 보안은 불완전함
5. **API 보안 문서화** — 명확성을 위해 OpenAPI 보안 스킴 사용

---

**이전**: [09_Web_Security_Headers.md](./09_Web_Security_Headers.md) | **다음**: [11_Secrets_Management.md](./11_Secrets_Management.md)
