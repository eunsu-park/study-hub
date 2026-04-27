# 15. 인증 패턴(Authentication Patterns)

**이전**: [API 설계 패턴](./14_API_Design_Patterns.md) | **다음**: [프로덕션 배포](./16_Production_Deployment.md)

**난이도**: ⭐⭐⭐

## 학습 목표

- 세션 기반 인증(session-based authentication)과 토큰 기반 인증(token-based authentication)의 차이점과 각각의 트레이드오프(trade-off)를 설명한다
- 백엔드 애플리케이션에서 액세스 토큰(access token)과 리프레시 토큰(refresh token)을 활용한 JWT 인증을 구현한다
- OAuth2 인가 흐름(authorization flow)을 비교하고 클라이언트 유형에 따라 적합한 흐름을 선택한다
- bcrypt 또는 argon2를 사용하여 적절한 설정으로 안전한 비밀번호 해싱(password hashing)을 적용한다
- 다중 인증(Multi-Factor Authentication) 개념과 백엔드 시스템에서의 통합 방법을 설명한다

## 목차

1. [세션 기반 인증](#1-세션-기반-인증)
2. [JWT 인증](#2-jwt-인증)
3. [API 키](#3-api-키)
4. [OAuth2 흐름](#4-oauth2-흐름)
5. [비교: 세션 vs. JWT vs. API 키](#5-비교-세션-vs-jwt-vs-api-키)
6. [비밀번호 해싱](#6-비밀번호-해싱)
7. [다중 인증](#7-다중-인증)
8. [연습 문제](#8-연습-문제)

---

## 1. 세션 기반 인증

세션 기반 인증(session-based authentication)은 사용자 상태를 서버에 저장한다. 로그인 후 서버는 세션 레코드를 생성하고 쿠키(cookie)를 통해 클라이언트에 세션 ID를 전달한다. 이후의 모든 요청에는 이 쿠키가 포함되며, 서버는 세션을 조회하여 사용자를 식별한다.

### 이론: 인증 vs 인가, 그리고 자격 증명 생명주기

두 단어는 서로 바꿔도 될 것 같지만 다른 것을 의미하며, 이를 혼동하면 실제 보안 버그가 생깁니다.

#### A.1 두 가지 구별되는 성질

- **인증(Authentication, AuthN).** "당신은 누구인가?" 요청자가 자신이 주장하는 사람임을 검증. 사용자명 + 비밀번호, 지문, 하드웨어 키, JWT 서명.
- **인가(Authorization, AuthZ).** "당신은 무엇을 할 수 있는가?" 인증된 신원이 주어졌을 때 그가 이 작업을 수행할 수 있는지를 결정. 역할, 권한, ACL, RBAC, ABAC.

올바르게 인증된 사용자도 인가에 의해 거부될 수 있습니다(`403 Forbidden`). 인증되지 않은 요청은 인증에 의해 거부됩니다(`401 Unauthorized`). 상태 코드가 정확히 이를 의미합니다.

피해야 할 실수: 인가 로직을 인증 계층에 넣거나, 그 반대. 흔한 버그 패턴은 "사용자가 유효한 세션을 가지면 어떤 게시물이든 편집할 수 있다" — 세션은 신원을 증명하지만 소유권에 대해서는 아무것도 말하지 않습니다.

#### A.2 토큰 생명주기

모든 자격 증명 — 세션 ID, JWT, API 키, refresh token — 은 같은 생명주기를 가집니다.

1. **발급.** 유효한 자격 증명 요청(로그인, OAuth 콜백, 키 생성)이 자격 증명을 만들고 필요한 서버 측 상태를 저장합니다.
2. **사용.** 자격 증명이 각 요청과 함께 제시됩니다. 서버가 검증합니다.
3. **갱신 / 연장.** 장기 자격 증명이 신선한 단기 것과 교환될 수 있습니다(refresh token에서 access token).
4. **취소.** 로그아웃, 비밀번호 변경, 침해 대응 — 자격 증명이 동작을 멈춰야 합니다.

어려운 단계는 #4입니다. 서버 측 저장소(세션)는 취소를 사소하게 만듭니다: 행을 삭제. 무상태 토큰(JWT)은 이를 어렵게 만듭니다: 토큰은 만료까지 유효한데 *블랙리스트*를 유지하지 않으면 그렇고, 블랙리스트는 서버 측 상태를 다시 도입합니다.

#### A.3 단기 access + 장기 refresh

표준 현대 패턴:

- **Access token** — JWT, 수명 15분 – 1시간. 모든 요청과 함께 보냄.
- **Refresh token** — 서버 측에 저장된 불투명한 무작위 문자열, 수명 7일 – 90일. 새 access token을 얻기 위해 `/auth/refresh`에서만 사용.

Access token이 만료되면 클라이언트는 refresh token을 사용해 새 것을 얻습니다. Refresh token이 취소되면(행 삭제), 사용자는 최대 한 access-token 수명 안에 효과적으로 로그아웃됩니다. 두 세계의 장점: API에는 무상태 요청, 보안에는 상태 기반 취소.

### 동작 방식

```
Client                          Server
  |                               |
  |-- POST /login (credentials) ->|
  |                               |-- Validate credentials
  |                               |-- Create session in store
  |<- Set-Cookie: session_id=abc -|
  |                               |
  |-- GET /profile               ->|
  |   Cookie: session_id=abc       |
  |                               |-- Look up session "abc"
  |                               |-- Return user data
  |<- 200 OK { user data }       -|
```

### FastAPI와 세션 저장소(Redis)

```python
import uuid
import redis
import json
from fastapi import FastAPI, Request, Response, HTTPException

app = FastAPI()
session_store = redis.Redis(host="localhost", port=6379, db=0)

SESSION_TTL = 3600  # 1 hour

@app.post("/login")
async def login(response: Response, credentials: LoginRequest):
    user = await authenticate(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create session
    session_id = str(uuid.uuid4())
    session_data = {"user_id": user.id, "username": user.username}
    session_store.setex(
        f"session:{session_id}",
        SESSION_TTL,
        json.dumps(session_data),
    )

    # Set cookie — HttpOnly prevents JavaScript access
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,     # Not accessible via document.cookie
        secure=True,       # HTTPS only
        samesite="lax",    # CSRF protection
        max_age=SESSION_TTL,
    )
    return {"message": "Logged in"}

@app.get("/profile")
async def get_profile(request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    session_data = session_store.get(f"session:{session_id}")
    if not session_data:
        raise HTTPException(status_code=401, detail="Session expired")

    session = json.loads(session_data)
    user = await get_user_by_id(session["user_id"])
    return {"id": user.id, "username": user.username, "email": user.email}

@app.post("/logout")
async def logout(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if session_id:
        session_store.delete(f"session:{session_id}")
    response.delete_cookie("session_id")
    return {"message": "Logged out"}
```

### 보안 고려사항

- **HttpOnly 쿠키**: XSS(Cross-Site Scripting) 공격으로 세션 ID가 탈취되는 것을 방지한다
- **Secure 플래그**: 쿠키가 HTTPS를 통해서만 전송되도록 보장한다
- **SameSite 속성**: CSRF(Cross-Site Request Forgery) 공격을 완화한다 (`Lax` 또는 `Strict`)
- **세션 만료**: 적절한 TTL을 설정하고 유휴 시간 초과(idle timeout)를 구현한다
- **세션 고정(session fixation)**: 로그인 후 세션 ID를 재생성한다

---

## 2. JWT 인증

JSON Web Token(JWT)은 사용자 클레임(claim)을 담고 있는 자기 완결형(self-contained) 토큰이다. 세션과 달리 서버는 상태를 저장할 필요가 없다 — 토큰 자체에 모든 필요한 정보가 담겨 있으며, 암호화 서명(cryptographic signature)으로 검증된다.

### 이론: JWT 구조와 서명

JSON Web Tokens(RFC 7519)는 무상태 인증의 지배적 토큰 형식입니다. 구조는 위협적으로 보이지만 기계적입니다.

#### B.1 세 부분

JWT는 `base64(header).base64(payload).base64(signature)`입니다.

```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI0MiIsImV4cCI6MTcwMDAwMDAwMH0.k1qV...
└────────── header ──────────┘.└──────── payload ────────┘.└── signature ──┘
```

**Header**는 서명 알고리즘과 토큰 타입을 선언합니다.

```json
{ "alg": "HS256", "typ": "JWT" }
```

**Payload**는 주체에 대한 클레임 집합입니다. 일부 클레임은 표준(`iss`, `sub`, `aud`, `exp`, `iat`, `nbf`)이고, 다른 것은 애플리케이션 특화입니다.

```json
{ "sub": "42", "name": "Alice", "role": "admin", "exp": 1700000000 }
```

**Signature**는 header의 알고리즘과 서버의 서명 키를 사용해 `base64(header) + "." + base64(payload)` 위에서 계산됩니다.

```
signature = HMAC-SHA256(base64(header) + "." + base64(payload), secret)
```

#### B.2 검증 규칙 (그리고 유명한 버그 부류)

JWT를 검증하려면 서버는

1. `.`로 세 부분으로 나눕니다.
2. Header의 알고리즘과 서버의 키로 서명을 다시 계산합니다.
3. 제공된 서명과 비교합니다.
4. 클레임을 검증합니다(주로 `exp`, 때로는 `nbf`, `iss`, `aud`).

버그 부류: header의 `alg` 필드를 신뢰하기. 공격자가 `"alg": "none"`(서명 없음)을 설정할 수 있고, 많은 순진한 라이브러리가 서명되지 않은 토큰을 받아들입니다. 해결책은 검증기에 예상 알고리즘을 *하드코딩*하는 것입니다 — 어떤 알고리즘을 쓸지 header에 절대 묻지 마세요.

관련 버그: 비밀을 공개 RSA 키로 발행하는 HMAC 서명 서비스를 가진 공격자가 `alg: RS256`을 `alg: HS256`으로 바꾸고, 공개 키를 HMAC 비밀처럼 사용해 서명하면, 서버(RS256을 기대했지만 header에서 HS256을 자동 감지하는)가 받아들입니다. 같은 해결책: 알고리즘을 하드코딩하세요.

#### B.3 HMAC vs RSA: 대칭 vs 비대칭

| 알고리즘 | 서명 키 | 검증 키 | 사용 시기 |
|-----------|-------------|---------------|----------|
| HS256 (HMAC-SHA256) | 공유 비밀 | 같은 공유 비밀 | 한 서비스가 자체 토큰을 서명하고 검증 |
| RS256 (RSA-SHA256) | 개인 키 | 공개 키 | 발급자와 검증자가 다른 서비스; 검증자 다수 |

HMAC은 더 단순하고 빠르지만 모든 검증자가 서명 키를 가져야 합니다 — 검증자 침해는 공격자가 새 토큰을 위조할 수 있게 한다는 뜻입니다. RSA는 발급자가 개인 키를 보관하고 공개 키를 발행(종종 `/.well-known/jwks.json`의 JWKS 엔드포인트를 통해)하게 해줍니다. 침해된 검증자는 토큰을 읽을 수 있지만 위조할 수는 없습니다.

규칙: **모놀리스 → HS256 괜찮음; 마이크로서비스나 외부 인증 제공자 → RS256**.

### JWT 구조

JWT는 점(.)으로 구분된 세 개의 base64url 인코딩 파트로 구성된다:

```
eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiI1In0.signature
|---- Header ----|.|---- Payload ----|.|-- Sig --|
```

**헤더(Header)**: 알고리즘과 토큰 유형
```json
{"alg": "HS256", "typ": "JWT"}
```

**페이로드(Payload)**: 클레임 (사용자 데이터 및 메타데이터)
```json
{
    "sub": "5",
    "username": "alice",
    "role": "admin",
    "iat": 1706140800,
    "exp": 1706144400
}
```

**서명(Signature)**: HMAC-SHA256(base64url(header) + "." + base64url(payload), secret)

### 액세스 토큰 + 리프레시 토큰 패턴

수명이 짧은 액세스 토큰(access token)은 토큰이 탈취되더라도 피해 창(damage window)을 최소화한다. 리프레시 토큰(refresh token)은 재인증 없이도 새 액세스 토큰을 발급받을 수 있게 해준다.

```python
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

app = FastAPI()

SECRET_KEY = "your-secret-key-store-in-env"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_token(data: dict, expires_delta: timedelta) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_access_token(user_id: int, username: str) -> str:
    return create_token(
        {"sub": str(user_id), "username": username, "type": "access"},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

def create_refresh_token(user_id: int) -> str:
    return create_token(
        {"sub": str(user_id), "type": "refresh"},
        timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    )

async def get_current_user(token: str = Depends(oauth2_scheme)):
    """JWT에서 현재 사용자를 추출하고 검증하는 의존성(Dependency)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        token_type = payload.get("type")
        if user_id is None or token_type != "access":
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = await get_user_by_id(int(user_id))
    if user is None:
        raise credentials_exception
    return user

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return {
        "access_token": create_access_token(user.id, user.username),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
    }

@app.post("/token/refresh")
async def refresh(refresh_token: str):
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = int(payload["sub"])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = await get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return {
        "access_token": create_access_token(user.id, user.username),
        "token_type": "bearer",
    }

@app.get("/profile")
async def get_profile(current_user=Depends(get_current_user)):
    return {"id": current_user.id, "username": current_user.username}
```

### JWT 보안 모범 사례

- **짧은 만료 시간**: 액세스 토큰은 5~30분 내에 만료되어야 한다
- **리프레시 토큰을 안전하게 저장**: HttpOnly 쿠키나 암호화된 저장소를 사용하며, localStorage에는 절대 저장하지 않는다
- **모든 클레임 검증**: `exp`, `iss`, `aud`, 토큰 유형을 반드시 확인한다
- **분산 시스템에는 비대칭 서명(RS256) 사용**: 여러 서비스가 토큰을 검증하는 환경에서 활용한다
- **토큰 폐기(token revocation) 구현**: 탈취된 토큰에 대한 차단 목록(blocklist)을 유지한다 (예: Redis)
- **페이로드에 민감한 데이터 저장 금지**: JWT는 암호화가 아닌 인코딩이므로 내용이 노출될 수 있다

---

## 3. API 키

API 키(API key)는 특정 사용자가 아닌 호출하는 애플리케이션을 식별하는 길고 무작위적인 문자열이다. 서버 간 통신(server-to-server communication)과 서비스 식별에 가장 적합하다.

### API 키 사용 시점

- **머신-투-머신(Machine-to-machine)** 통신 (백엔드 서비스, CI/CD 파이프라인)
- **공개 데이터 API** — 사용자 신원이 필요 없는 경우 (예: 날씨 데이터)
- **애플리케이션 또는 고객별 사용량 추적 및 과금**

### API 키를 사용하지 말아야 할 시점

- **사용자 인증** — API 키는 사용자 세션을 나타내지 않는다
- **브라우저 기반 클라이언트** — 키가 클라이언트 코드에 노출될 수 있다

### 구현

```python
from fastapi import FastAPI, Security, HTTPException
from fastapi.security import APIKeyHeader

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key")

# In production, store keys in a database with associated metadata
API_KEYS = {
    "sk_live_abc123def456": {"client": "mobile-app", "tier": "pro"},
    "sk_live_xyz789ghi012": {"client": "partner-service", "tier": "enterprise"},
}

async def verify_api_key(api_key: str = Security(api_key_header)):
    key_data = API_KEYS.get(api_key)
    if not key_data:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return key_data

@app.get("/data")
async def get_data(key_data: dict = Security(verify_api_key)):
    return {"client": key_data["client"], "data": "..."}
```

### 보안 관행

- 키 유형을 나타내는 접두사 사용: `sk_live_`, `sk_test_`, `pk_` (공개용)
- 데이터베이스에는 키를 해싱하여 저장 — 평문이 아닌 해시만 보관한다
- 키 로테이션(key rotation) 지원: 클라이언트당 여러 개의 활성 키 허용
- 감사 추적(audit trail)을 위해 키 사용 내역을 로그로 기록
- 만료일을 설정하고 주기적인 로테이션을 요구한다

---

## 4. OAuth2 흐름

OAuth2는 제3자 애플리케이션이 자격 증명(credential)을 노출하지 않고도 사용자 리소스에 접근할 수 있게 해주는 **인가(authorization)** 프레임워크다 (인증(authentication)이 아님). 클라이언트 유형에 따라 다양한 흐름이 적합하다.

### 이론: OAuth 2.0 흐름 + OIDC + SAML

OAuth 2.0은 인증 프로토콜이 *아닙니다*. 인가 프레임워크입니다 — "사용자가 애플리케이션 X에 자기 대신 서비스 Y를 호출할 권리를 부여한다". OAuth 2.0 위의 인증은 OpenID Connect(OIDC)가 제공합니다.

#### C.1 네 가지 역할

- **Resource Owner** — 데이터를 소유한 사용자.
- **Client** — 접근을 요청하는 애플리케이션.
- **Authorization Server** — 토큰의 발급자(Google, Auth0, Okta).
- **Resource Server** — 클라이언트가 호출하려는 API.

#### C.2 PKCE와 함께하는 authorization code 흐름 (현대의 기본값)

PKCE = Proof Key for Code Exchange. 공개 클라이언트(모바일 앱, single-page app)에 대한 코드 인터셉트 공격 부류를 막습니다.

```
1. 클라이언트가 code_verifier(무작위)와 code_challenge = SHA256(code_verifier) 생성
2. 클라이언트가 code_challenge와 함께 사용자를 Authorization Server로 리다이렉트
3. 사용자 로그인; AS가 auth code와 함께 다시 리다이렉트
4. 클라이언트가 (auth code + code_verifier)를 /token 엔드포인트에서 교환
5. AS가 SHA256(code_verifier) == 저장한 code_challenge인지 검증
6. AS가 access_token(과 refresh_token, OIDC라면 id_token)을 반환
```

PKCE의 이유: 모바일 앱에서 리다이렉트는 기기의 다른 악성 앱에 인터셉트될 수 있습니다. PKCE 없이는 악성 앱이 인터셉트한 코드를 토큰으로 교환할 수 있습니다. PKCE와 함께라면 `code_verifier`를 가진 원래 클라이언트만 교환을 완료할 수 있습니다.

`code_verifier`는 클라이언트가 만들고 보유합니다. `code_challenge`는 공개 약속입니다. 고전적 암호학적 commitment scheme입니다.

#### C.3 다른 흐름들 (그리고 대부분이 deprecated인 이유)

| 흐름 | 사용처 | 상태 |
|------|----------|--------|
| Authorization code + PKCE | 모든 상호작용 앱 | 권장 |
| Authorization code (PKCE 없음) | client secret이 있는 서버 측 웹 앱 | 받아들일 만함 |
| Implicit | 백엔드 없는 SPA | Deprecated (code+PKCE 사용) |
| Resource Owner Password Credentials | 극도의 신뢰가 있는 first-party 앱 | Deprecated |
| Client Credentials | 서비스 간 (사용자 없음) | 그 경우에 권장 |
| Device Code | 입력이 제한된 기기(TV, CLI) | 그 경우에 권장 |

지난 10년의 추세: 더 약한 보안의 비용으로 라운드트립이 적은 흐름(Implicit, ROPC)을 deprecate, 안전한 흐름은 유지.

#### C.4 OIDC: OAuth 2.0 위의 인증

OIDC는 OAuth 2.0 응답에 `id_token`을 더합니다. `id_token`은 사용자 신원 클레임(`sub`, `email`, `name`)을 담은 JWT입니다. 이제 갖게 되는 것:

- **Access token** — API 호출용 (OAuth 2.0).
- **ID token** — 사용자가 누구인지 알기 위한 (OIDC).
- **Refresh token** — 새 access token을 얻기 위한 (둘 다).

대부분의 "Sign in with Google/Apple/GitHub" 흐름은 OIDC입니다. `id_token`은 신원을 증명하고, `access_token`은 Google API를 호출하게 해줍니다.

#### C.5 SAML: 더 오래된 XML 기반 대안

SAML 2.0은 OIDC 이전의 엔터프라이즈 SSO 표준입니다. XML 기반, 더 장황하지만, 기업 환경(Okta, Ping, ADFS)에 잘 자리 잡고 있습니다. 개념 모델은 비슷합니다: Identity Provider(IdP)가 사용자를 인증하고 Service Provider(SP)가 검증하는 assertion에 서명합니다.

언제 무엇을 쓸지:

- **그린필드, 현대 앱, 소비자 SSO** → OIDC.
- **레거시 IdP가 있는 엔터프라이즈 SSO** → SAML.

2026년에 둘 다 살아 있습니다. 대부분의 엔터프라이즈 인증 플랫폼은 둘 다 지원합니다.

### 인가 코드 흐름 + PKCE (Authorization Code Flow with PKCE)

가장 안전한 흐름으로, 웹 앱과 모바일 앱에서 사용된다. PKCE(Proof Key for Code Exchange)는 인가 코드 가로채기(authorization code interception)를 방지한다.

```
User          Client App       Auth Server       Resource Server
 |                |                 |                  |
 |-- Click Login ->|                |                  |
 |                |-- Redirect ---->|                  |
 |                |   (+ code_verifier hash)           |
 |<-------------- Auth Page --------|                  |
 |-- Login + Consent -------------->|                  |
 |                |<-- Auth Code ---|                  |
 |                |-- Exchange Code + code_verifier -->|
 |                |<-- Access Token + Refresh Token ---|
 |                |-- GET /resource (Bearer token) --->|
 |                |<-- Resource Data ------------------|
```

### 클라이언트 자격 증명 흐름 (Client Credentials Flow)

사용자가 관여하지 않는 서버 간 통신에 사용된다.

```python
import httpx

async def get_service_token():
    """클라이언트 자격 증명을 교환하여 액세스 토큰을 획득한다."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://auth.example.com/oauth/token",
            data={
                "grant_type": "client_credentials",
                "client_id": "my-service-id",
                "client_secret": "my-service-secret",
                "scope": "read:users",
            },
        )
        token_data = response.json()
        return token_data["access_token"]
```

### 흐름 선택 가이드

| 흐름                                | 클라이언트 유형            | 사용자 관여 여부 | 보안 수준 |
|-------------------------------------|---------------------------|-----------------|----------|
| 인가 코드 + PKCE                    | 웹 앱, 모바일 앱           | 있음            | 높음     |
| 클라이언트 자격 증명                | 서버 간 통신               | 없음            | 높음     |
| 기기 코드(Device Code)              | 스마트 TV, CLI 도구        | 있음            | 중간     |
| 임플리시트(Implicit) (사용 중단됨)  | SPA (레거시)               | 있음            | 낮음     |

> **참고**: OAuth 2.1에서 임플리시트(Implicit) 흐름은 사용 중단(deprecated)되었다. 모든 브라우저 기반 애플리케이션에는 인가 코드 + PKCE를 사용한다.

### FastAPI OAuth2 통합 (GitHub 예제)

```python
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import httpx
import secrets

app = FastAPI()

GITHUB_CLIENT_ID = "your-client-id"
GITHUB_CLIENT_SECRET = "your-client-secret"

# Store state parameters to prevent CSRF
pending_states: dict[str, bool] = {}

@app.get("/auth/github")
async def github_login():
    state = secrets.token_urlsafe(32)
    pending_states[state] = True
    return RedirectResponse(
        f"https://github.com/login/oauth/authorize"
        f"?client_id={GITHUB_CLIENT_ID}"
        f"&scope=user:email"
        f"&state={state}"
    )

@app.get("/auth/github/callback")
async def github_callback(code: str, state: str):
    # Verify state to prevent CSRF
    if state not in pending_states:
        raise HTTPException(status_code=400, detail="Invalid state")
    del pending_states[state]

    # Exchange code for access token
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://github.com/login/oauth/access_token",
            json={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
            },
            headers={"Accept": "application/json"},
        )
        access_token = token_response.json()["access_token"]

        # Fetch user info
        user_response = await client.get(
            "https://api.github.com/user",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        github_user = user_response.json()

    # Create or update local user, issue session/JWT
    user = await get_or_create_user(
        github_id=github_user["id"],
        username=github_user["login"],
        email=github_user.get("email"),
    )
    token = create_access_token(user.id, user.username)
    return {"access_token": token, "token_type": "bearer"}
```

---

## 5. 비교: 세션 vs. JWT vs. API 키

| 항목              | 세션                     | JWT                        | API 키                   |
|-------------------|--------------------------|----------------------------|--------------------------|
| 상태              | 서버 측 (스테이트풀)     | 클라이언트 측 (스테이트리스) | 서버 측 (스테이트풀)    |
| 저장소            | Redis, DB, 메모리        | 쿠키 또는 Authorization 헤더 | 데이터베이스            |
| 확장성            | 공유 저장소 필요         | 수평 확장 가능              | 단순 조회                |
| 폐기              | 쉬움 (세션 삭제)         | 어려움 (차단 목록 필요)     | 쉬움 (키 삭제)           |
| 식별 대상         | 사용자 세션              | 사용자 클레임               | 애플리케이션/클라이언트 |
| 적합한 용도       | 전통적인 웹 앱           | SPA, 모바일, 마이크로서비스 | 서비스 간 통신           |
| CSRF 취약점       | 있음 (쿠키 기반)         | 없음 (Auth 헤더 사용 시)    | 없음                     |
| XSS 취약점        | 낮음 (HttpOnly 쿠키)     | 높음 (localStorage 사용 시) | 낮음 (서버 측)           |

### 적합한 전략 선택

- **서버 렌더링 웹 앱** (Django 템플릿, Jinja2): 세션
- **SPA + API 백엔드**: JWT (액세스 + 리프레시 토큰)
- **모바일 앱**: JWT + 안전한 저장소 (Keychain/Keystore)
- **마이크로서비스 간 통신**: JWT 또는 상호 TLS(mutual TLS)
- **서드파티 API 접근**: API 키
- **서드파티 사용자 로그인**: OAuth2

---

## 6. 비밀번호 해싱

비밀번호를 평문으로 저장해서는 절대 안 된다. 브루트포스(brute-force) 공격에 저항하기 위해 의도적으로 느리게 설계된 전용 비밀번호 해싱 함수를 사용해야 한다.

### bcrypt

가장 널리 사용되는 비밀번호 해싱 알고리즘이다. 연산 비용을 조절하기 위해 설정 가능한 비용 인수(cost factor, 작업 인수)를 사용한다.

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """bcrypt로 비밀번호를 해싱한다 (기본 비용 인수: 12)."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """비밀번호를 해시와 비교하여 검증한다."""
    return pwd_context.verify(plain_password, hashed_password)

# Usage
hashed = hash_password("my-secure-password")
# "$2b$12$LJ3m4ys3Lp..."

assert verify_password("my-secure-password", hashed) is True
assert verify_password("wrong-password", hashed) is False
```

### argon2

비밀번호 해싱 경쟁(Password Hashing Competition, 2015) 우승 알고리즘으로, 새 프로젝트에 권장된다. GPU와 사이드 채널 공격(side-channel attack) 모두에 대한 보호를 제공한다.

```python
from passlib.context import CryptContext

# argon2id is the recommended variant
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

# Same API as bcrypt
hashed = pwd_context.hash("my-secure-password")
is_valid = pwd_context.verify("my-secure-password", hashed)
```

### 비교

| 알고리즘  | GPU 저항성 | 메모리 집약적 | 사이드 채널 저항성 | 성숙도  |
|-----------|-----------|--------------|-------------------|--------|
| bcrypt    | 좋음      | 아니오       | 아니오            | 높음   |
| scrypt    | 좋음      | 예           | 아니오            | 중간   |
| argon2id  | 매우 좋음 | 예           | 예                | 성장 중 |

**권장 사항**: 새 프로젝트에는 **argon2id**를 사용한다. argon2를 사용할 수 없거나 기존 시스템과의 호환성이 필요한 경우 **bcrypt**를 사용한다.

### 비밀번호 정책

```python
import re

def validate_password(password: str) -> list[str]:
    """정책 위반 목록을 반환한다. 빈 목록이면 유효한 비밀번호다."""
    errors = []
    if len(password) < 12:
        errors.append("Password must be at least 12 characters")
    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain an uppercase letter")
    if not re.search(r"[a-z]", password):
        errors.append("Password must contain a lowercase letter")
    if not re.search(r"\d", password):
        errors.append("Password must contain a digit")
    # Check against common password lists in production
    return errors
```

---

## 7. 다중 인증

다중 인증(Multi-Factor Authentication, MFA)은 사용자가 두 가지 이상의 독립적인 인증 요소를 제공하도록 요구한다. 요소는 다음과 같이 분류된다:

- **알고 있는 것**: 비밀번호, PIN
- **갖고 있는 것**: 휴대폰 (TOTP 앱), 하드웨어 키 (YubiKey)
- **본인 자체**: 지문, 얼굴 인식

### TOTP (시간 기반 일회용 비밀번호, Time-Based One-Time Password)

가장 일반적인 MFA 방법이다. 서버와 클라이언트는 비밀 키를 공유하며, 현재 시간을 기반으로 30초마다 변경되는 6자리 코드를 생성한다.

```python
import pyotp
import qrcode
from io import BytesIO

def generate_totp_secret() -> str:
    """사용자를 위한 새 TOTP 시크릿을 생성한다."""
    return pyotp.random_base32()

def get_totp_uri(secret: str, username: str, issuer: str = "MyApp") -> str:
    """QR 코드 생성을 위한 otpauth:// URI를 생성한다."""
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=username, issuer_name=issuer)

def verify_totp(secret: str, code: str) -> bool:
    """TOTP 코드를 검증한다. 시간 오차를 위해 +/- 1 시간 단계(time step)를 허용한다."""
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)

# Enrollment flow
secret = generate_totp_secret()
uri = get_totp_uri(secret, "alice@example.com")
# User scans QR code with authenticator app (Google Authenticator, Authy, etc.)
# Store `secret` in user record (encrypted)

# Verification flow
user_code = "123456"  # from authenticator app
if verify_totp(stored_secret, user_code):
    print("MFA verified")
```

### 로그인 흐름에서의 통합

```python
@app.post("/login")
async def login(credentials: LoginRequest):
    user = await authenticate(credentials.username, credentials.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user.mfa_enabled:
        # Issue a short-lived, limited-scope token for MFA step
        mfa_token = create_token(
            {"sub": str(user.id), "type": "mfa_pending"},
            timedelta(minutes=5),
        )
        return {"mfa_required": True, "mfa_token": mfa_token}

    # No MFA: issue full access token
    return {
        "access_token": create_access_token(user.id, user.username),
        "token_type": "bearer",
    }

@app.post("/login/mfa")
async def verify_mfa(mfa_token: str, code: str):
    try:
        payload = jwt.decode(mfa_token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("type") != "mfa_pending":
            raise HTTPException(status_code=401)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid MFA token")

    user = await get_user_by_id(int(payload["sub"]))
    if not verify_totp(user.totp_secret, code):
        raise HTTPException(status_code=401, detail="Invalid MFA code")

    return {
        "access_token": create_access_token(user.id, user.username),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
    }
```

### 백업 코드(Backup Codes)

MFA 등록 시 사용자가 인증 기기를 분실하는 경우를 대비하여 항상 백업 코드를 제공한다. 8~10개의 일회용 코드를 생성하고 해시하여 데이터베이스에 저장한다.

---

## 8. 연습 문제

### 문제 1: 세션 보안 감사

다음 쿠키 설정을 가진 세션 기반 인증 시스템을 물려받았다:
```python
response.set_cookie(key="session_id", value=session_id, max_age=86400)
```
이 설정의 모든 보안 문제를 식별하고 적절한 보안 설정으로 다시 작성하라. 각 설정이 중요한 이유를 설명하라.

### 문제 2: JWT 리프레시 토큰 로테이션

리프레시 토큰이 사용될 때마다 새 리프레시 토큰이 발급되고 기존 토큰이 무효화되는 리프레시 토큰 로테이션(refresh token rotation) 시스템을 구현하라. 이전에 사용된 리프레시 토큰이 제출되면 (잠재적인 탈취를 나타냄) 전체 토큰 패밀리(token family)를 폐기한다. 토큰 패밀리 추적에 Redis를 사용한다.

### 문제 3: OAuth2 PKCE 구현

모바일 앱을 위한 PKCE를 적용한 인가 코드 흐름(Authorization Code flow with PKCE)을 구현하라. 다음을 작성한다:
1. 코드 검증자(code verifier)와 코드 챌린지(code challenge) 생성
2. 인가 리다이렉트 URL 구성
3. 코드 검증자를 검증하는 토큰 교환 엔드포인트
4. HTTPS를 사용하더라도 PKCE가 필요한 이유를 설명하라

### 문제 4: 비밀번호 마이그레이션

현재 애플리케이션이 MD5(심각한 취약점)로 비밀번호를 저장하고 있다. argon2id로 마이그레이션하는 전략을 설계하고 구현하라:
- 사용자가 즉시 비밀번호를 재설정할 필요가 없도록 한다
- 다음 성공적인 로그인 시 비밀번호를 투명하게 재해싱(rehashing)한다
- 마이그레이션 진행 상황을 추적한다
- 90일 이상 비활성 계정은 강제 재설정한다

### 문제 5: 다중 인증 시스템

FastAPI 애플리케이션을 위한 완전한 MFA 등록 및 검증 흐름을 설계하라. 다음을 포함한다:
- TOTP 시크릿 생성 및 QR 코드 전달
- 백업 코드 생성 (10개, 일회용)
- MFA 단계가 포함된 로그인 흐름
- 인증 기기 분실 시 MFA 복구 방법
- MFA 코드 시도에 대한 속도 제한 (토큰당 최대 5회 시도)

---

## 참고 자료

- [RFC 7519: JSON Web Token (JWT)](https://tools.ietf.org/html/rfc7519)
- [RFC 6749: OAuth 2.0 Authorization Framework](https://tools.ietf.org/html/rfc6749)
- [RFC 7636: PKCE for OAuth](https://tools.ietf.org/html/rfc7636)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
- [Password Hashing Competition](https://www.password-hashing.net/)
- [python-jose Documentation](https://python-jose.readthedocs.io/)
- [passlib Documentation](https://passlib.readthedocs.io/)

---

**이전**: [API 설계 패턴](./14_API_Design_Patterns.md) | **다음**: [프로덕션 배포](./16_Production_Deployment.md)
