# 03. FastAPI 고급(FastAPI Advanced)

**이전**: [FastAPI 기초](./02_FastAPI_Basics.md) | **다음**: [FastAPI 데이터베이스](./04_FastAPI_Database.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- `Depends()`를 사용한 의존성 주입(Dependency Injection) 패턴을 구현하여 엔드포인트 간에 로직을 공유할 수 있다
- OAuth2 패스워드 베어러(password bearer) 방식을 사용한 JWT 기반 인증 흐름을 구축할 수 있다
- 적절한 prefix와 태그 구성을 갖춘 `APIRouter`를 사용하여 모듈형 애플리케이션을 설계할 수 있다
- 실시간 양방향 통신을 위한 WebSocket 엔드포인트를 구현할 수 있다
- 시작 및 종료 시 리소스 관리를 위한 애플리케이션 생명주기 이벤트를 관리할 수 있다

---

## 목차

1. [Depends()를 사용한 의존성 주입](#1-depends를-사용한-의존성-주입)
2. [인증: JWT를 사용한 OAuth2](#2-인증-jwt를-사용한-oauth2)
3. [파일 업로드](#3-파일-업로드)
4. [백그라운드 태스크](#4-백그라운드-태스크)
5. [WebSocket 지원](#5-websocket-지원)
6. [커스텀 미들웨어](#6-커스텀-미들웨어)
7. [모듈형 앱을 위한 APIRouter](#7-모듈형-앱을-위한-apirouter)
8. [생명주기 이벤트](#8-생명주기-이벤트)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. Depends()를 사용한 의존성 주입

의존성 주입(Dependency Injection, DI)은 엔드포인트 간에 재사용 가능한 로직을 공유하기 위한 FastAPI의 메커니즘입니다. 각 핸들러 내부에서 함수를 호출하는 대신, 의존성으로 선언하면 FastAPI가 자동으로 해결합니다.

### 이론: 의존성 주입: 함수 호출이 아니라 그래프

`Depends(callable)`는 "이 파라미터 값을 저 callable을 호출해서 계산하고 결과를 주입하라"고 말하는 FastAPI의 마커입니다. 단순한 멘탈 모델은 "의존성 함수를 호출해 반환값을 전달한다"이지만, 실제 메커니즘은 더 흥미롭습니다 — 캐시, override, yield 의미를 설명해 주는 메커니즘입니다.

#### A.1 의존성 DAG

각 핸들러는 0개 이상의 `Depends(...)` 파라미터를 가집니다. 각 의존성 *자체*가 또 다시 `Depends(...)` 파라미터를 가질 수 있는 callable입니다. 이 참조의 추이 폐쇄(transitive closure)가 방향성 비순환 그래프(DAG)를 이룹니다.

```
get_current_user
    ↓ depends on
get_token_from_header
    ↓ depends on
verify_signing_key
    ↓ depends on
settings
```

요청이 도착하면 FastAPI는

1. DAG를 위상 정렬해서 잎(leaf) 의존성이 그것을 의존하는 노드보다 먼저 해결되도록 합니다.
2. 정렬된 리스트를 따라 가며 각 의존성을 호출하고, `async`라면 await하고, 결과를 보관합니다.
3. 해결된 값들을 핸들러에 전달합니다.

그래프는 시작 시점(레슨 02의 타입 힌트 인트로스펙션 중)에 한 번 구축되어 캐시됩니다. 요청당 작업은 그저 그것을 따라 걷는 것뿐입니다. 요청당 reflection은 없습니다.

#### A.2 요청당 캐싱

기본적으로, 같은 의존성 callable이 한 요청의 DAG에 여러 번 등장하면 *한 번*만 호출됩니다. 결과는 그 요청의 나머지 동안 캐시되어 재사용됩니다. 이것이 "이 라우터의 모든 엔드포인트가 현재 사용자를 필요로 한다" 같은 패턴을 싸게 만듭니다 — `get_current_user`가 추이 의존성으로 다섯 군데에 등장해도 JWT는 한 번만 디코딩됩니다.

캐시 범위는 *한 요청*입니다. 두 요청은 두 개의 독립된 캐시를 갖습니다. 의존성이 등장할 때마다 매번 실행되어야 한다면 `Depends(callable, use_cache=False)`로 기본값을 덮어쓸 수 있습니다(드뭅니다).

#### A.3 `yield` 기반 의존성: 설정과 정리

의존성 함수는 `return` 대신 `yield`를 쓸 수 있습니다.

```python
async def get_db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

`yield` 이전의 모든 것은 의존성 해결 시점에 실행됩니다. yield된 값이 주입됩니다. `yield` *이후*의 모든 것은 응답이 송신된 뒤 — finalizer 단계 — 에 실행됩니다. 이는 `contextlib.contextmanager`와 구조적으로 동일하며, FastAPI는 그렇게 구현했습니다. 해제되어야 하는 모든 자원(DB 세션, 파일 핸들, HTTP 클라이언트, 캐시 연결)에 적합한 패턴입니다.

핸들러가 예외를 던져도 teardown은 실행됩니다 — `try`/`finally`와 비슷합니다. 핸들러가 `HTTPException`을 던지면 의존성의 `finally` 블록이 먼저 실행되고, 그 후 예외가 HTTP 응답으로 변환됩니다.

#### A.4 서브 애플리케이션 override

테스트를 위해 `app.dependency_overrides[real_dep] = fake_dep`을 쓰면 핸들러를 건드리지 않고 의존성을 갈아 끼울 수 있습니다. 해결이 요청 시점에 DAG를 통해 일어나기 때문에 override가 자동으로 픽업됩니다 — 핸들러는 차이를 모릅니다. 이것이 테스트에서 의존성 주입의 보상입니다. import를 monkeypatch하지 않고 가짜 데이터베이스, 가짜 인증, 가짜 시계로 대체할 수 있습니다.

### 기본 의존성

```python
from fastapi import FastAPI, Depends, Query

app = FastAPI()

# 의존성은 단순한 callable (함수 또는 클래스)입니다
async def common_pagination(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=10, ge=1, le=100),
) -> dict[str, int]:
    """재사용 가능한 페이지네이션 로직. 모든 목록 엔드포인트에서
    이러한 쿼리 파라미터를 반복하는 대신, 한 번만 선언합니다."""
    return {"skip": skip, "limit": limit}

@app.get("/users")
async def list_users(pagination: dict = Depends(common_pagination)):
    # pagination = {"skip": 0, "limit": 10}
    return {"users": [], **pagination}

@app.get("/posts")
async def list_posts(pagination: dict = Depends(common_pagination)):
    # 동일한 페이지네이션 로직, 코드 중복 없음
    return {"posts": [], **pagination}
```

### 클래스 기반 의존성

```python
from dataclasses import dataclass

@dataclass
class PaginationParams:
    """클래스 기반 의존성. FastAPI는 함수 의존성과 마찬가지로
    이름으로 일치하는 쿼리 파라미터로 생성자를 호출합니다."""
    skip: int = Query(default=0, ge=0)
    limit: int = Query(default=10, ge=1, le=100)

@app.get("/items")
async def list_items(params: PaginationParams = Depends()):
    # 타입 어노테이션이 있는 파라미터에 인자 없이 Depends()를 사용하면,
    # FastAPI는 어노테이션(PaginationParams)을 의존성으로 사용합니다
    return {"skip": params.skip, "limit": params.limit}
```

### 의존성 체인

의존성은 다른 의존성에 의존할 수 있으며, FastAPI가 순서대로 해결하는 체인을 형성합니다:

```
get_current_user
    └── 의존: get_token_from_header
                    └── 의존: oauth2_scheme (Bearer 토큰 추출)
```

```python
from fastapi import Header, HTTPException

async def get_api_key(x_api_key: str = Header(...)):
    """첫 번째 레벨: 헤더에서 API 키를 추출합니다."""
    return x_api_key

async def verify_api_key(api_key: str = Depends(get_api_key)):
    """두 번째 레벨: 추출된 키를 검증합니다.
    의존성은 체인을 형성합니다 -- FastAPI가 순서대로 해결합니다."""
    valid_keys = {"key-abc-123", "key-def-456"}
    if api_key not in valid_keys:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.get("/protected")
async def protected_route(api_key: str = Depends(verify_api_key)):
    # 이 코드가 실행될 때 API 키는 이미 검증된 상태입니다
    return {"message": "Access granted", "key": api_key}
```

### Yield 의존성 (리소스 정리)

```python
from typing import AsyncGenerator

async def get_db_session() -> AsyncGenerator:
    """Yield 의존성은 yield 전에 설정 코드를 실행하고 그 후에 정리합니다.
    이 패턴은 엔드포인트가 예외를 발생시키더라도
    데이터베이스 세션이 항상 닫히도록 보장합니다."""
    session = AsyncSession()
    try:
        yield session      # 엔드포인트가 여기서 실행됨
    finally:
        await session.close()  # 항상 실행됨 (컨텍스트 매니저와 같음)

@app.get("/data")
async def get_data(db: AsyncSession = Depends(get_db_session)):
    result = await db.execute(select(User))
    return result.scalars().all()
```

---

## 2. 인증: JWT를 사용한 OAuth2

FastAPI는 OAuth2 플로우를 기본 지원합니다. API에서 가장 일반적인 패턴은 **JWT**(JSON Web Tokens)를 사용하는 **패스워드 베어러(password bearer)** 플로우입니다.

### 개요

```
┌──────────┐        ┌──────────────┐        ┌──────────┐
│  클라이언트│        │  FastAPI      │        │  데이터베이스│
│           │        │  서버          │        │          │
│  POST /token       │               │        │          │
│  username+password  │               │        │          │
│ ─────────────────▶ │               │        │          │
│           │        │  자격증명 검증 ──────▶  │          │
│           │        │               │ ◀────  │          │
│           │        │  JWT 생성     │        │          │
│ ◀───────────────── │               │        │          │
│  {access_token}    │               │        │          │
│           │        │               │        │          │
│  GET /users/me     │               │        │          │
│  Authorization:    │               │        │          │
│  Bearer <JWT>      │               │        │          │
│ ─────────────────▶ │               │        │          │
│           │        │  JWT 디코딩   │        │          │
│           │        │  사용자 추출  │        │          │
│ ◀───────────────── │               │        │          │
│  {사용자 데이터}    │               │        │          │
└──────────┘        └──────────────┘        └──────────┘
```

### 구현

```python
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

app = FastAPI()

# --- 설정 ---
# 프로덕션에서는 환경 변수에서 로드하세요, 절대 하드코딩하지 마세요
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# --- 패스워드 해싱 ---
# bcrypt는 의도적으로 느리게 설계되어 무차별 대입 공격을
# 비실용적으로 만들기 때문에 패스워드 해싱에 권장되는 알고리즘입니다
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- OAuth2 스킴 ---
# tokenUrl은 Swagger UI에 로그인 요청을 보낼 위치를 알립니다
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- 모델 ---
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class User(BaseModel):
    username: str
    email: str
    disabled: bool = False

class UserInDB(User):
    hashed_password: str

# --- 가짜 데이터베이스 ---
fake_users_db = {
    "alice": {
        "username": "alice",
        "email": "alice@example.com",
        "hashed_password": pwd_context.hash("secret123"),
        "disabled": False,
    }
}

# --- 헬퍼 함수 ---
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """평문 패스워드를 해시와 비교합니다.
    passlib이 내부적으로 솔트 추출과 비교를 처리합니다."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """만료 시간이 있는 JWT를 생성합니다.
    'sub' (subject) 클레임은 사용자를 식별합니다."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def authenticate_user(username: str, password: str) -> UserInDB | None:
    """자격증명을 검증합니다. 유효하면 사용자를 반환하고, 그렇지 않으면 None을 반환합니다.
    유효한 사용자명을 노출하는 타이밍 공격을 방지하기 위해
    존재하지 않는 사용자에 대해서도 항상 해시 비교를 수행합니다."""
    user_data = fake_users_db.get(username)
    if not user_data:
        return None
    user = UserInDB(**user_data)
    if not verify_password(password, user.hashed_password):
        return None
    return user

# --- 의존성 ---
async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)]
) -> User:
    """JWT를 디코딩하고 현재 사용자를 반환합니다.
    이 의존성은 인증이 필요한 모든 엔드포인트에서 사용됩니다."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user_data = fake_users_db.get(username)
    if user_data is None:
        raise credentials_exception
    return User(**user_data)

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """체인 의존성: 사용자가 비활성화되지 않았는지 확인합니다."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# --- 엔드포인트 ---
@app.post("/token", response_model=Token)
async def login(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    """OAuth2 패스워드 플로우 엔드포인트.
    Swagger UI는 이 URL로 POST하는 로그인 폼을 제공합니다."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me", response_model=User)
async def read_users_me(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """보호된 엔드포인트. 유효한 JWT로만 접근 가능합니다."""
    return current_user
```

---

## 3. 파일 업로드

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from pathlib import Path

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# 최대 파일 크기: 5 MB
MAX_FILE_SIZE = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "application/pdf"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """유효성 검사를 포함한 파일 업로드 처리.
    UploadFile은 스풀드(spooled) 임시 파일을 사용하기 때문에
    bytes보다 선호됩니다 -- 큰 업로드가 메모리를 모두 소비하지 않습니다."""

    # 파일을 읽기 전에 콘텐츠 타입 검증
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file.content_type} not allowed. "
                   f"Allowed: {ALLOWED_TYPES}"
        )

    # 읽고 크기 확인
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large (max 5MB)")

    # 정제된 파일명으로 디스크에 저장
    safe_name = Path(file.filename).name  # 디렉토리 컴포넌트 제거
    save_path = UPLOAD_DIR / safe_name
    save_path.write_bytes(contents)

    return {
        "filename": safe_name,
        "size_bytes": len(contents),
        "content_type": file.content_type,
    }

@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    """단일 요청으로 여러 파일을 받습니다."""
    results = []
    for f in files:
        contents = await f.read()
        results.append({"filename": f.filename, "size": len(contents)})
    return {"uploaded": results}
```

---

## 4. 백그라운드 태스크

백그라운드 태스크는 클라이언트에 응답을 보낸 **후에** 실행됩니다. 이메일 전송이나 감사 로그 작성과 같은 비중요 작업에 이상적입니다.

```python
from fastapi import FastAPI, BackgroundTasks
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

def send_welcome_email(email: str, name: str):
    """이메일 전송을 시뮬레이션합니다. 이것은 응답이 이미
    전송된 후에 실행되므로 클라이언트는 이메일 전달을 기다리지 않습니다."""
    logger.info(f"Sending welcome email to {email} for {name}")
    # 프로덕션에서: SendGrid, SES 등의 이메일 서비스 사용

def write_audit_log(action: str, user_id: int, details: str):
    """감사 로그 파일에 기록합니다. 백그라운드 태스크는
    클라이언트가 I/O 작업을 기다리지 않아야 하므로 로깅에 적합합니다."""
    logger.info(f"AUDIT: {action} by user {user_id}: {details}")

@app.post("/users", status_code=201)
async def create_user(
    name: str,
    email: str,
    background_tasks: BackgroundTasks,
):
    """사용자를 생성하고 사후 작업을 예약합니다.
    응답은 즉시 반환됩니다; 태스크는 백그라운드에서 실행됩니다."""
    user_id = 42  # 시뮬레이션된 DB 삽입

    # 여러 백그라운드 태스크 대기열에 추가 -- 순서대로 실행됨
    background_tasks.add_task(send_welcome_email, email, name)
    background_tasks.add_task(write_audit_log, "CREATE_USER", user_id, f"name={name}")

    # 클라이언트는 즉시 이것을 받고, 이메일/로깅을 기다리지 않습니다
    return {"id": user_id, "name": name, "email": email}
```

**중요**: 백그라운드 태스크는 동일한 프로세스를 공유합니다. 무거운 장시간 실행 작업(비디오 처리, ML 추론)에는 Celery나 Arq와 같은 적절한 태스크 큐를 사용하세요.

---

### 이론: async/await: 이벤트 루프 위의 코루틴

FastAPI 핸들러는 `async def` 또는 일반 `def`일 수 있습니다. 선택은 핸들러가 어떻게 실행되는지, 무엇을 해서는 안 되는지에 직접적인 결과를 가져옵니다.

#### B.1 `async def`가 실제로 사 주는 것

`async def` 함수는 `def` 함수보다 *빠르지 않습니다*. 마법처럼 작업을 병렬화하지도 않습니다. 그것이 제공하는 것은 *중단점(suspension points)*입니다 — 모든 `await`가 함수가 잠시 멈추고 이벤트 루프가 다른 일을 할 수 있게 해주는 지점입니다.

승리는 오직 **I/O 바운드** 워크로드에서 일어납니다. 핸들러가 `await db.execute(...)`를 기다리는 동안, 이벤트 루프는 같은 워커에서 다른 요청을 처리할 수 있습니다. 동기 코드라면 워커는 처음부터 끝까지 블로킹됩니다. 그래서

```
async def: 워커 1개가 동시에 진행 중인 수천 개 요청을 처리할 수 있다
def      : 워커 1개가 한 번에 1개 요청을, 순차적으로 처리한다
```

#### B.2 핵심 규칙: 이벤트 루프를 절대 블로킹하지 말 것

CPU 작업을 하거나 *동기적* 블로킹 I/O 함수를 호출하는 `async def` 핸들러는 전체 이벤트 루프를 얼립니다. 같은 워커의 다른 요청들이 처리되지 않습니다. 증상: 꼬리 지연시간 급증, 알 수 없는 타임아웃, "부하 시 서버가 멈춤".

조심해야 할 블로킹 범인들:

- `await asyncio.sleep(...)` 대신 `time.sleep(...)`
- `await httpx.AsyncClient().get(...)` 대신 `requests.get(...)`
- `await asyncpg.connect(...)`나 `SQLAlchemy` async 엔진 대신 `psycopg2.connect(...)`
- 메인 스레드에서 무거운 NumPy / Pandas / 이미지 처리

대체 불가능한 동기 I/O 라이브러리에 대해서는 핸들러를 `async def`가 아닌 `def`로 선언하세요. FastAPI가 스레드 풀에서 실행시켜 줍니다 — 루프는 자유로 남고, 워커 전체가 아니라 한 스레드만 블로킹됩니다. 비용은 스레드 풀이 유한하다는 것입니다. 무거운 부하에서는 여전히 병목입니다.

#### B.3 혼합 라우트 패턴

실제 FastAPI 앱은 일상적으로 `async def`와 `def` 핸들러를 모두 가집니다. 프레임워크는 둘을 다르게 디스패치합니다.

| 핸들러 | 실행되는 곳 | 위험 |
|---------|---------|------|
| `async def` | 이벤트 루프 직접 | 루프 블로킹이 동시성을 죽임 |
| `def` | `anyio.to_thread.run_sync` 스레드 풀 | 부하 시 스레드 풀 고갈 |

올바른 규칙: 모든 I/O 의존성이 비동기라면 `async def`, 그렇지 않으면 `def`. 한 비동기 핸들러 안에 비동기 I/O와 동기 I/O를 섞는 것이 최악입니다 — 비동기 비용은 치르고 이득은 못 받습니다.

## 5. WebSocket 지원

WebSocket은 단일 TCP 연결을 통해 전이중(full-duplex) 통신을 제공합니다. HTTP의 요청/응답 패턴과 달리, 클라이언트와 서버 모두 언제든지 메시지를 보낼 수 있습니다.

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dataclasses import dataclass, field

app = FastAPI()

@dataclass
class ConnectionManager:
    """활성 WebSocket 연결을 관리합니다.
    프로덕션에서는 다중 서버 지원을 위해 Redis pub/sub을 사용할 것입니다."""
    active_connections: list[WebSocket] = field(default_factory=list)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        """연결된 모든 클라이언트에 메시지를 보냅니다.
        반복 중 수정을 방지하기 위해 리스트의 복사본을 사용합니다."""
        for connection in self.active_connections.copy():
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                self.disconnect(connection)

manager = ConnectionManager()

@app.websocket("/ws/chat/{room_id}")
async def websocket_chat(websocket: WebSocket, room_id: str):
    """채팅룸을 위한 WebSocket 엔드포인트.
    무한 루프는 클라이언트가 연결을 끊을 때까지 연결을 유지합니다."""
    await manager.connect(websocket)
    try:
        await manager.broadcast(f"User joined room {room_id}")
        while True:
            # 클라이언트로부터 메시지 대기
            data = await websocket.receive_text()
            # 연결된 모든 클라이언트에 브로드캐스트
            await manager.broadcast(f"[{room_id}] {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"User left room {room_id}")
```

### 클라이언트 사이드 JavaScript

```javascript
// 테스트를 위한 최소한의 WebSocket 클라이언트
const ws = new WebSocket("ws://localhost:8000/ws/chat/general");

ws.onopen = () => {
    console.log("Connected");
    ws.send("Hello from client!");
};

ws.onmessage = (event) => {
    console.log("Received:", event.data);
};

ws.onclose = () => {
    console.log("Disconnected");
};
```

---

## 6. 커스텀 미들웨어

미들웨어는 모든 요청/응답 사이클을 감쌉니다. 라우트 핸들러 전에 실행되고 응답이 생성된 후에도 실행됩니다.

```python
import time
import uuid
from fastapi import FastAPI, Request, Response

app = FastAPI()

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """분산 추적(distributed tracing)을 위해 모든 요청에 고유 ID를 할당합니다.
    ID는 응답 헤더에 추가되며 서비스 간 로그를 연관시키는 데 사용할 수 있습니다."""
    request_id = str(uuid.uuid4())
    # 핸들러가 접근할 수 있도록 요청 상태에 저장
    request.state.request_id = request_id

    response: Response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    """요청 처리 시간을 측정하고 로깅합니다.
    미들웨어는 선언 역순으로 실행되므로,
    이것은 위의 request_id_middleware를 감쌉니다."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response
```

### 이론: 미들웨어: 양파 모델

미들웨어는 ASGI 서버와 핸들러 사이에 자리합니다. 각 미들웨어는 들어오는 요청을 변환하거나, 나가는 응답을 변환하거나, 둘 다 할 수 있습니다. 동심원 계층으로 합성됩니다 — 요청은 그들을 안쪽으로 통과하고, 응답은 바깥쪽으로 통과합니다.

#### C.1 요청/응답 추적

이 순서로 등록된 세 미들웨어 A, B, C에서 한 요청은 다음과 같이 흐릅니다.

```
        IN                    OUT
client → A → B → C → handler → C → B → A → client
```

A는 요청을 가장 먼저 보고 응답을 가장 마지막에 봅니다. C는 정반대 순서로 봅니다. `add_middleware()` 호출 *순서가 중요*합니다. 그것이 계층화를 제어합니다. 관용적 순서는 바깥에서 안으로: trusted host → CORS → GZip → 세션 → 인증 → request id / 로깅 → 핸들러.

#### C.2 계약: call_next

Starlette/FastAPI 미들웨어는 `request`와 `call_next` 비동기 함수를 받는 callable입니다. 안쪽 계층으로부터 응답을 얻기 위해 `await call_next(request)`를 해야 하고, 그 후 응답을 반환해야 합니다 — 수정될 수도, 원본 그대로일 수도 있습니다.

```python
@app.middleware("http")
async def add_request_id(request, call_next):
    request.state.req_id = uuid.uuid4().hex
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.req_id
    return response
```

`call_next` 이전의 모든 것은 들어가는 길에 실행됩니다. 그 이후의 모든 것은 나오는 길에 실행됩니다. `call_next`를 호출하지 않고 일찍 반환하면 단락(short-circuit)됩니다 — 핸들러가 실행되기 전에 인증되지 않은 요청을 `401`로 거부하는 인증 미들웨어에 유용합니다.

#### C.3 미들웨어가 의존성 주입과 *다른* 이유

둘 다 요청에 데이터를 붙이고 핸들러 전후에 실행될 수 있습니다. 두 가지 결정적인 차이가 있습니다.

1. **세분성.** 미들웨어는 앱에 들어오는 *모든* 요청에 적용됩니다. Depends는 하나 또는 여러 특정 엔드포인트에 적용됩니다. 횡단 관심사(CORS, 로깅)에는 미들웨어를, 엔드포인트별 로직(보호된 라우트에서만 인증)에는 Depends를 선택하세요.
2. **타입 통합.** Depends 결과는 IDE 지원이 완전한 타입 파라미터로 주입됩니다. 미들웨어 결과는 타입 없는 속성으로 `request.state`에 살아 있습니다. 그래서 "현재 사용자"는 보통 미들웨어가 아니라 Depends입니다.

쓸모 있는 어림짐작: HTTP 헤더를 바꾸거나 응답을 단락시키는 것이라면 미들웨어이고, 핸들러가 필요로 하는 타입 값을 만드는 것이라면 Depends입니다.

### 실행 순서

```
요청  ──▶  timing_middleware (진입)
             ──▶  request_id_middleware (진입)
                     ──▶  라우트 핸들러
             ◀──  request_id_middleware (종료)
       ◀──  timing_middleware (종료)  ──▶  응답
```

**먼저** 선언된 미들웨어가 **가장 바깥** 레이어를 감쌉니다. 핸들러는 최종적으로 완전히 처리된 요청을 받습니다.

---

## 7. 모듈형 앱을 위한 APIRouter

애플리케이션이 커지면 모든 것을 하나의 파일에 넣는 것은 유지 불가능해집니다. `APIRouter`를 사용하면 엔드포인트를 모듈로 분리할 수 있습니다.

### 프로젝트 구조

```
app/
├── main.py           # 애플리케이션 팩토리, 라우터 포함
├── routers/
│   ├── __init__.py
│   ├── users.py      # 사용자 관련 엔드포인트
│   ├── posts.py      # 게시물 관련 엔드포인트
│   └── admin.py      # 관리자 전용 엔드포인트
├── models/
│   ├── __init__.py
│   └── schemas.py    # Pydantic 모델
└── dependencies.py   # 공유 의존성
```

### 라우터 정의

```python
# app/routers/users.py
from fastapi import APIRouter, Depends, HTTPException, status
from ..dependencies import get_current_active_user
from ..models.schemas import UserCreate, UserResponse

# prefix와 tags는 이 라우터의 모든 라우트에 적용됩니다
router = APIRouter(
    prefix="/api/users",
    tags=["users"],
    responses={404: {"description": "Not found"}},
)

@router.get("/", response_model=list[UserResponse])
async def list_users():
    """라우터 prefix 덕분에 이것은 GET /api/users/가 됩니다."""
    return []

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    """이것은 GET /api/users/{user_id}가 됩니다."""
    ...

@router.post("/", response_model=UserResponse, status_code=201)
async def create_user(user: UserCreate):
    ...
```

```python
# app/routers/admin.py
from fastapi import APIRouter, Depends
from ..dependencies import get_current_admin_user

# 이 라우터의 모든 라우트는 관리자 인증이 필요합니다
router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_admin_user)],  # 모든 라우트에 적용
)

@router.get("/stats")
async def get_stats():
    """관리자만 접근 가능합니다.
    의존성은 각 엔드포인트가 아닌 라우터에 선언됩니다."""
    return {"total_users": 100, "active_today": 42}
```

### 메인 애플리케이션

```python
# app/main.py
from fastapi import FastAPI
from .routers import users, posts, admin

app = FastAPI(title="My Modular API", version="1.0.0")

# 모든 라우터를 포함. 각 라우터의 prefix와 tags가 보존됩니다.
app.include_router(users.router)
app.include_router(posts.router)
app.include_router(admin.router)

@app.get("/")
async def root():
    return {"message": "API is running"}
```

---

## 8. 생명주기 이벤트

생명주기 이벤트는 애플리케이션이 시작되고 중지될 때 한 번씩 실행되는 설정 및 종료 로직을 처리합니다. 일반적인 용도: 데이터베이스 연결 풀, ML 모델 로딩, 캐시 워밍.

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI

# 시뮬레이션된 리소스
ml_models: dict = {}
db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """lifespan 컨텍스트 매니저는 더 이상 사용되지 않는
    @app.on_event("startup")과 @app.on_event("shutdown") 데코레이터를 대체합니다.
    yield 전의 모든 것은 시작 시 실행되고, yield 후는 종료 시 실행됩니다."""

    # --- 시작(STARTUP) ---
    print("Loading ML model...")
    ml_models["sentiment"] = load_model("sentiment-v2")

    print("Creating database pool...")
    db_pool = await create_pool(
        "postgresql://localhost/mydb",
        min_size=5,
        max_size=20,
    )
    # 의존성이 접근할 수 있도록 app.state에 저장
    app.state.db_pool = db_pool

    print("Application ready!")

    yield  # 여기서 애플리케이션이 실행되어 요청을 처리함

    # --- 종료(SHUTDOWN) ---
    print("Closing database pool...")
    await db_pool.close()

    print("Unloading ML models...")
    ml_models.clear()

    print("Shutdown complete.")

# FastAPI 생성자에 lifespan을 전달
app = FastAPI(lifespan=lifespan)

@app.get("/predict")
async def predict(text: str):
    """시작 중에 초기화된 리소스에 접근합니다."""
    model = ml_models.get("sentiment")
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"sentiment": model.predict(text)}
```

### on_event 대신 lifespan을 사용하는 이유

`@app.on_event()` 데코레이터는 FastAPI 0.93+에서 더 이상 사용되지 않습니다(deprecated). `lifespan` 컨텍스트 매니저가 선호되는 이유:

1. **보장된 정리**: 시작이 실패해도 종료 코드가 실행됨
2. **공유 상태**: 클로저를 통해 시작에서 종료로 변수를 전달 가능
3. **테스트 가능성**: 테스트에서 lifespan을 재정의 가능
4. **명확성**: 설정과 종료를 한 함수에서 함께 보여줌

---

## 9. 연습 문제

### 문제 1: 의존성 주입 체인

세 단계의 의존성 체인을 구축하세요:
1. `get_settings()` -- 구성 로드 (데이터베이스 URL, API 키)
2. `get_db(settings)` -- 설정을 사용하여 데이터베이스 연결 생성
3. `get_user_repo(db)` -- 데이터베이스 연결로 사용자 리포지토리 생성

사용자를 나열, 생성, 삭제하는 세 가지 엔드포인트를 `get_user_repo`를 사용하여 만드세요. `get_settings`를 변경했을 때 체인을 통해 전파되는지 확인하세요.

### 문제 2: 역할 기반 접근 제어(RBAC)

이 레슨의 JWT 인증을 확장하여 **역할**(admin, editor, viewer)을 지원하세요:
- 의존성을 반환하는 `require_role(role: str)` 의존성 팩토리 생성
- `GET /admin/dashboard` -- admin 역할 필요
- `POST /posts` -- editor 또는 admin 역할 필요
- `GET /posts` -- 인증된 사용자면 누구든지

역할은 JWT 페이로드에 저장되어야 합니다.

### 문제 3: 방이 있는 WebSocket 채팅

WebSocket 채팅 예제를 확장하여 다음을 지원하세요:
- 여러 방 (각 방에 자체 연결 리스트)
- 활성 방과 사용자 수를 나열하는 `/rooms` REST 엔드포인트
- 사용자명 식별 (연결 시 쿼리 파라미터로 전달)
- 메시지 기록 (방당 최근 50개 메시지를 메모리에 저장)

### 문제 4: 모듈형 애플리케이션

모놀리식 FastAPI 앱을 모듈로 재구성하세요:
- `routers/auth.py` -- 로그인, 회원가입, 토큰 갱신
- `routers/items.py` -- 아이템 CRUD
- `routers/admin.py` -- 관리자 전용 통계, 사용자 관리
- `dependencies.py` -- 공유 의존성 (인증, DB, 페이지네이션)

각 라우터에는 적절한 prefix, tags, 라우터 레벨 의존성이 있어야 합니다.

### 문제 5: lifespan 리소스 관리

다음을 갖춘 FastAPI 앱을 만들어 보세요:
1. 시작 시: 인메모리 SQLite 데이터베이스 생성, 마이그레이션 실행, 테스트 데이터 시드
2. 종료 시: 통계(처리된 총 요청 수, 가동 시간)를 JSON 파일로 내보내기
3. `app.state`를 사용하여 데이터베이스 연결을 엔드포인트와 공유
4. 모든 요청에서 요청 카운터를 증가시키는 미들웨어 포함

---

## 10. 참고 자료

- [FastAPI 의존성](https://fastapi.tiangolo.com/tutorial/dependencies/)
- [FastAPI 보안 - JWT를 사용한 OAuth2](https://fastapi.tiangolo.com/tutorial/security/oauth2-jwt/)
- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)
- [FastAPI APIRouter](https://fastapi.tiangolo.com/tutorial/bigger-applications/)
- [FastAPI 생명주기 이벤트](https://fastapi.tiangolo.com/advanced/events/)
- [python-jose (JWT 라이브러리)](https://github.com/mpdavis/python-jose)
- [Passlib 문서](https://passlib.readthedocs.io/)

---

**이전**: [FastAPI 기초](./02_FastAPI_Basics.md) | **다음**: [FastAPI 데이터베이스](./04_FastAPI_Database.md)
