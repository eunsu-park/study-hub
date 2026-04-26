# 05. FastAPI 테스트

**이전**: [FastAPI 데이터베이스](./04_FastAPI_Database.md) | **다음**: [Express 기초](./06_Express_Basics.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- FastAPI의 `TestClient`를 사용한 동기 테스트와 `httpx.AsyncClient`를 사용한 비동기 테스트 작성하기
- 재사용 가능한 테스트 설정과 적절한 격리를 위한 pytest 픽스처(fixture)와 `conftest.py` 설계하기
- 트랜잭션 롤백으로 격리된 테스트 데이터베이스를 생성하는 데이터베이스 테스트 픽스처 구현하기
- FastAPI 의존성(dependency)을 오버라이드하여 테스트 중 모의 객체(mock)와 테스트 더블(test double) 주입하기
- 테스트되지 않은 코드 경로를 파악하기 위한 테스트 커버리지(coverage) 리포트 설정하기

---

## 목차

프레임워크 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. TestClient가 ASGI 위에서 in-process로 무엇을 하는지, pytest 픽스처의 네 가지 범위와 그 생명주기 함의가 무엇인지, 그리고 dependency_overrides + 트랜잭션 롤백이 어떻게 일회용 데이터베이스 없이 격리를 주는지를 다룹니다.

1. [동기 테스트를 위한 TestClient](#1-동기-테스트를-위한-testclient)
2. [비동기 테스트를 위한 httpx.AsyncClient](#2-비동기-테스트를-위한-httpxasyncclient)
3. [pytest 픽스처와 conftest.py](#3-pytest-픽스처와-conftestpy)
4. [데이터베이스 픽스처](#4-데이터베이스-픽스처)
5. [dependency_overrides를 활용한 의존성 목킹](#5-dependency_overrides를-활용한-의존성-목킹)
6. [인증 흐름 테스트](#6-인증-흐름-테스트)
7. [커버리지 리포트](#7-커버리지-리포트)
8. [연습 문제](#8-연습-문제)
9. [참고 자료](#9-참고-자료)

---

## 이론과 원리

FastAPI 애플리케이션의 테스트 스위트는 그저 "엔드포인트를 호출하고 JSON을 검사하는 것"이 아닙니다. 빠르고, 격리되고, 정확한 스위트가 되려면 별도로 이해해야 할 세 가지 독립된 메커니즘 위에 얹혀 있습니다.

- **(A) ASGI 트랜스포트를 통한 in-process HTTP** — `TestClient`는 TCP 소켓을 열지 않고, 앱 객체와 직접 ASGI로 대화합니다.
- **(B) 생명주기 위계로서의 픽스처 범위** — `function`, `class`, `module`, `session`이 무엇을 한 번 만들지, 무엇을 매 테스트마다 다시 만들지를 결정합니다.
- **(C) 테스트 격리 전략** — 트랜잭션 롤백, 테스트별 데이터베이스, 의존성 주입이 함께 "이 테스트가 다른 테스트의 데이터를 볼 수 없다"는 계약을 형성합니다.

### A. In-process ASGI 트랜스포트

프로덕션에서 HTTP 요청은 이렇게 흐릅니다: 클라이언트 → 커널 → TCP → 커널 → uvicorn → ASGI → FastAPI → 핸들러. 테스트에서는 그 전체 체인이 과합니다. `TestClient`(그리고 `ASGITransport`가 있는 `httpx.AsyncClient`)는 이를 압축합니다: 클라이언트 → ASGI dict → FastAPI → 핸들러. FastAPI 안의 코드 경로는 동일하고, 소켓은 관여하지 않습니다.

#### A.1 TestClient가 실제로 하는 일

`TestClient(app)`은 네트워크 소켓을 열지 않는 특별한 `WSGITransport`/`ASGITransport`로 구성된 `httpx.Client`입니다. `client.get("/items/")`을 호출하면 트랜스포트는

1. URL, 메서드, 헤더로부터 ASGI `scope` dict를 만듭니다(레슨 02 §A.1과 같은 모양).
2. `await app(scope, receive, send)`를 직접 호출합니다 — `receive`는 요청 본문을 yield하고 `send`는 응답 메시지를 모읍니다.
3. 모인 메시지를 `httpx.Response` 객체로 재조립합니다.

전체 교환이 테스트 프로세스 안, 메모리에서 이루어집니다. 라운드트립 지연시간은 밀리초가 아니라 마이크로초입니다. 할당할 포트도, 리스너 스레드도, TLS 핸드셰이크도 없습니다.

#### A.2 충실도 측면에서 이것이 중요한 이유

같은 FastAPI 애플리케이션 인스턴스가 요청을 처리합니다. 모든 미들웨어, 모든 `Depends`, 모든 직렬화가 실행됩니다. 건너뛰는 것은 *네트워크*뿐입니다. 그래서 in-process ASGI 테스트는 진짜 통합 테스트입니다 — 통과한 테스트는 네트워크/TLS 문제를 제외하면 프로덕션에서도 성공함을 의미합니다.

비용은 테스트가 네트워크 계층에 의존하는 동작을 시험할 수 없다는 것입니다. 재연결 로직, 실제 타임아웃, 프록시 동작을 테스트하려면 진짜 서버가 필요합니다. API 테스트의 99%는 in-process가 답입니다.

#### A.3 동기 vs 비동기 클라이언트: 같은 트랜스포트

`TestClient`는 동기 래퍼이고, 내부적으로는 비동기 앱을 내부 이벤트 루프에서 돌립니다. 픽스처를 await하는 `async def` 핸들러 테스트에는 보통 `httpx.AsyncClient`를 `ASGITransport(app=app)`와 함께 직접 쓰는 것이 좋습니다 — 그렇게 하면 테스트 코드와 핸들러가 같은 이벤트 루프를 공유하고, 픽스처 자원을 자연스럽게 await할 수 있습니다.

선택은 사용성의 문제이지 성능 문제가 아닙니다. 둘 다 같은 ASGI 메커니즘으로 흐릅니다.

### B. 픽스처 범위: 생명주기 위계

pytest 픽스처는 좁은 것부터 넓은 것 순서로 네 개의 내장 범위를 가집니다.

| 범위 | 만들어지는 시점 | 해체되는 시점 | 비용 분산 단위 |
|-------|------------|----------------|---------------------|
| `function`(기본) | 각 테스트 시작 | 각 테스트 종료 | 테스트 1개 |
| `class` | 클래스의 첫 테스트가 필요로 할 때 | 클래스의 마지막 테스트 종료 | 클래스의 모든 테스트 |
| `module` | 파일의 첫 테스트가 필요로 할 때 | 파일의 마지막 테스트 종료 | 파일의 모든 테스트 |
| `session` | 실행의 첫 테스트가 필요로 할 때 | 실행 전체 종료 | 전체 테스트 스위트 |

선택은 **격리**와 **속도** 사이의 트레이드오프입니다.

- 전체 SQLAlchemy 엔진을 `function` 범위로 만드는 것은 정확하지만 매우 느립니다 — 엔진 설정 비용 × N개 테스트를 치릅니다.
- `session` 범위로 만들면 빠르지만, 리셋하지 않으면 테스트 간 상태 누수의 위험이 있습니다.

관용적 패턴: 무거운 읽기 전용 자원은 `session`(엔진, FastAPI 앱 인스턴스). 변경 가능한 테스트별 상태는 `function`(데이터베이스 세션, HTTP 클라이언트, 의존성 override). 이렇게 하면 설정 비용을 최소화하면서 각 테스트의 변경이 이웃에게 보이지 않게 유지됩니다.

#### B.1 픽스처 합성

픽스처는 다른 픽스처를 이름으로 요청할 수 있습니다.

```python
@pytest.fixture(scope="session")
def engine(): ...

@pytest.fixture(scope="function")
def db_session(engine): ...

@pytest.fixture(scope="function")
def client(db_session): ...
```

pytest는 (레슨 03의 FastAPI Depends DAG처럼) DAG를 만들고, 범위를 존중하며(`function`은 자기보다 좁은 범위를 의존할 수 없지만, 더 넓은 범위에 의존하는 것은 괜찮습니다), 빌드 역순으로 해체합니다.

#### B.2 설정/해체를 위한 `yield`

픽스처의 `yield` 패턴은 레슨 03의 Depends 패턴과 일치합니다.

```python
@pytest.fixture
def db_session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    session = SessionLocal(bind=connection)
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
```

yield 이전에 설정, 이후에 해체. 테스트가 실패해도 해체는 실행됩니다. 이것이 §C 테스트 격리의 토대입니다.

### C. 테스트 격리: 세 가지 메커니즘

테스트 스위트에서 가장 유지하기 어려운 성질이 **격리**입니다. 테스트 B가 테스트 A가 한 일을 보아서는 안 됩니다. 세 가지 메커니즘이 함께 작동해 이를 제공합니다.

#### C.1 트랜잭션 롤백

매 테스트마다 데이터베이스를 drop하고 재생성하는 대신, 각 테스트를 끝에서 롤백되는 트랜잭션으로 감쌉니다.

```
BEGIN
  ... 테스트 실행, INSERT, UPDATE ...
ROLLBACK
```

데이터베이스는 시작 상태로 돌아가지만, 테스트마다 `CREATE TABLE` 비용은 치르지 않습니다. SQLAlchemy의 표준 패턴입니다. 세션을 연결에 고정하고, 트랜잭션을 시작하고, 세션을 테스트에 넘겨주고, 끝나면 롤백.

미묘한 점: 테스트 코드가 직접 `session.commit()`을 호출해서는 안 됩니다. 그러면 바깥 트랜잭션이 닫혀 버립니다. 해결책은 **세이브포인트(savepoint)**(`SAVEPOINT` / `ROLLBACK TO SAVEPOINT`) — 중첩 트랜잭션입니다. SQLAlchemy는 `connection.begin_nested()`와 commit 후 새 세이브포인트를 다시 여는 hook으로 이를 제공합니다. 테스트는 정상적인 commit 의미를 보지만, 바깥 트랜잭션은 여전히 롤백됩니다.

#### C.2 페이크를 위한 dependency_overrides

`app.dependency_overrides[get_db] = override_get_db`(레슨 03 §A.4)는 테스트가 어떤 의존성의 가짜 구현을 대체할 수 있게 해줍니다. 데이터베이스 테스트에서는 override가 테스트의 트랜잭션 세션을 반환합니다. 외부 API 테스트에서는 가짜 클라이언트를 반환합니다. 핸들러는 수정되지 않지만 이제 테스트가 통제하는 더블을 받습니다.

override는 `TestClient`가 첫 요청을 보내기 *전에* 설정되어야 하고, 끝난 후에 해체되어야 합니다 — 보통은 픽스처의 setup/teardown에서. 그렇지 않으면 override가 다음 테스트로 누수됩니다.

#### C.3 테스트별이 아니라 워커별 데이터베이스

테스트가 병렬로 실행될 때(`pytest-xdist`), 모든 워커가 한 SQLite 파일이나 한 PostgreSQL 스키마를 공유해서는 안 됩니다 — 서로를 짓밟습니다. 패턴은 **워커당 데이터베이스 1개**입니다. pytest는 `worker_id`("gw0", "gw1", ...)를 노출하고, 데이터베이스 이름에 그것을 접미합니다. 워커 안에서는 테스트가 여전히 직렬화되므로 트랜잭션 롤백이 그대로 작동합니다.

조합 — 워커당 session 범위 엔진, function 범위 트랜잭션 세션, 의존성 override된 핸들러 — 이 올바른 균형을 줍니다. 각 테스트가 격리되고, 스위트는 초 단위로 실행되며, 병렬성은 워커 수에 선형으로 확장됩니다.

### 이론에서 아래 코드로

뒤에 나오는 각 절은 이 틀의 한 조각을 구체화합니다.

- §1 (TestClient)는 §A.1 in-process ASGI 트랜스포트에 동기 래퍼를 입힌 것입니다.
- §2 (httpx.AsyncClient)는 같은 §A.1 트랜스포트를 비동기 테스트에서 사용해 `async def` 픽스처와 루프를 공유합니다.
- §3 (pytest 픽스처와 conftest.py)는 §B의 범위 위계와 DAG 합성을 공유 파일로 조직합니다.
- §4 (데이터베이스 픽스처)는 §B.2 yield 패턴 위에 §C.1 트랜잭션 롤백을 구현합니다.
- §5 (dependency_overrides)는 §C.2 메커니즘입니다 — 핸들러를 바꾸지 않고 페이크를 대체.
- §6 (인증 흐름)은 TestClient를 사용해 "로그인 → 토큰 사용" 시퀀스를 체이닝합니다 — 인증 헤더가 있는 §A 경로입니다.
- §7 (커버리지)는 §A–§C의 테스트 스위트가 실제로 어떤 코드 경로를 시험하는지를 측정하는 메타 도구입니다.

---

## 1. 동기 테스트를 위한 TestClient

FastAPI의 `TestClient`는 `httpx`를 감싸(wrap) 실제 서버를 실행하지 않고도 앱에 요청을 전송합니다. 동기 방식이기 때문에 테스트에서 `async/await`를 사용할 필요가 없습니다.

### 기본 설정

```python
# tests/test_basic.py
from fastapi import FastAPI
from fastapi.testclient import TestClient

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, World!"}

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    return {"user_id": user_id, "name": f"User {user_id}"}

# TestClient는 네트워크 I/O 없이 프로세스 내에서 앱으로 요청을 보냅니다
client = TestClient(app)

def test_root():
    """루트 엔드포인트가 200과 기대하는 메시지를 반환하는지 테스트합니다."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

def test_get_user():
    """경로 파라미터(path parameter) 파싱과 응답 구조를 테스트합니다."""
    response = client.get("/users/42")
    assert response.status_code == 200
    data = response.json()
    assert data["user_id"] == 42
    assert data["name"] == "User 42"

def test_get_user_invalid_id():
    """경로 파라미터 타입 검증이 실패할 때 FastAPI는 422를 반환합니다.
    int가 필요한 곳에 문자열을 보내면 자동 검증이 작동합니다."""
    response = client.get("/users/not-a-number")
    assert response.status_code == 422
    # 오류 응답 본문에는 어떤 필드가 실패했는지 상세 정보가 포함됩니다
    assert "detail" in response.json()
```

### POST 요청 테스트

```python
from pydantic import BaseModel
from fastapi import status

class ItemCreate(BaseModel):
    name: str
    price: float

@app.post("/items", status_code=status.HTTP_201_CREATED)
async def create_item(item: ItemCreate):
    return {"id": 1, **item.model_dump()}

def test_create_item():
    """유효한 데이터로 아이템을 생성하는 테스트입니다."""
    response = client.post(
        "/items",
        json={"name": "Widget", "price": 9.99},  # json=은 Content-Type을 자동으로 설정합니다
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Widget"
    assert data["price"] == 9.99
    assert "id" in data

def test_create_item_missing_field():
    """필수 필드 누락 시 필드 상세 정보가 포함된 422를 반환하는지 테스트합니다."""
    response = client.post("/items", json={"name": "Widget"})
    assert response.status_code == 422
    errors = response.json()["detail"]
    # 누락된 'price' 필드의 오류를 찾습니다
    price_error = next(e for e in errors if "price" in e["loc"])
    assert price_error["type"] == "missing"

def test_create_item_invalid_type():
    """잘못된 타입이 422를 반환하는지 테스트합니다."""
    response = client.post(
        "/items",
        json={"name": "Widget", "price": "not-a-number"},
    )
    assert response.status_code == 422
```

### 헤더 및 쿼리 파라미터 테스트

```python
def test_with_headers():
    """특정 헤더를 필요로 하는 엔드포인트를 테스트합니다."""
    response = client.get(
        "/protected",
        headers={"X-API-Key": "valid-key-123"},
    )
    assert response.status_code == 200

def test_with_query_params():
    """쿼리 파라미터는 params 인수에 딕셔너리로 전달합니다."""
    response = client.get(
        "/users",
        params={"skip": 0, "limit": 5, "role": "admin"},
    )
    assert response.status_code == 200
```

---

## 2. 비동기 테스트를 위한 httpx.AsyncClient

엔드포인트가 비동기 의존성(async database session 등)을 사용하는 경우 비동기 테스트 클라이언트가 필요합니다. `TestClient`도 많은 경우에 작동하지만, `httpx.AsyncClient`는 진정한 비동기 테스트를 지원합니다.

### 설정

```bash
pip install httpx pytest-asyncio
```

### 비동기 테스트 예제

```python
# tests/test_async.py
import pytest
from httpx import ASGITransport, AsyncClient
from app.main import app

# pytest-asyncio는 비동기 테스트 인프라를 제공합니다
# mode="auto"는 모든 async 테스트 함수를 각각 @pytest.mark.asyncio를 달지 않아도
# 자동으로 비동기 테스트로 처리합니다
pytestmark = pytest.mark.anyio

@pytest.fixture
async def async_client():
    """비동기 테스트 클라이언트를 생성합니다.
    ASGITransport는 실제 서버를 시작하거나 네트워크 호출 없이
    ASGI 앱으로 직접 요청을 보냅니다."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

async def test_root(async_client: AsyncClient):
    """비동기 테스트 -- 요청에 await를 사용합니다."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, World!"}

async def test_create_and_get_user(async_client: AsyncClient):
    """생성 후 읽기 전체 워크플로우를 테스트합니다.
    비동기 테스트는 자연스럽게 여러 await 호출을 연결할 수 있습니다."""
    # 생성
    create_response = await async_client.post(
        "/api/users",
        json={"username": "testuser", "email": "test@example.com", "password": "secret123"},
    )
    assert create_response.status_code == 201
    user_id = create_response.json()["id"]

    # 다시 읽기
    get_response = await async_client.get(f"/api/users/{user_id}")
    assert get_response.status_code == 200
    assert get_response.json()["username"] == "testuser"
```

### 어떤 것을 사용해야 할까?

| 시나리오 | `TestClient` 사용 | `AsyncClient` 사용 |
|----------|-----------------|-------------------|
| 단순 엔드포인트 테스트 | 가능 | 가능 |
| 비동기 데이터베이스 작업 | 동작함 (비동기를 감쌈) | 더 좋음 (네이티브 비동기) |
| WebSocket 테스트 | `TestClient`에 내장 지원 | 수동 처리 필요 |
| 속도 | 약간 빠름 | 약간 느림 |
| 디버깅 | 쉬움 (동기 스택) | 어려움 (비동기 스택) |

---

## 3. pytest 픽스처와 conftest.py

픽스처(fixture)는 재사용 가능한 테스트 설정/해제(setup/teardown) 함수입니다. `conftest.py`는 픽스처를 직접 임포트하지 않아도 디렉토리 내 모든 테스트에서 사용 가능하게 합니다.

### 프로젝트 구조

```
tests/
├── conftest.py         # 공유 픽스처 (모든 테스트에서 사용 가능)
├── test_users.py       # 사용자 엔드포인트 테스트
├── test_posts.py       # 게시물 엔드포인트 테스트
└── test_auth.py        # 인증 테스트
```

### conftest.py

```python
# tests/conftest.py
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.main import app
from app.models import Base
from app.dependencies import get_db

# 테스트에는 인메모리(in-memory) SQLite 데이터베이스를 사용합니다
# 빠르고 격리되어 있으며 테스트 실행 시마다 새로 시작됩니다
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def anyio_backend():
    """pytest-asyncio에 asyncio를 사용하도록 지시합니다 (trio 아님)."""
    return "asyncio"

@pytest.fixture(scope="session")
async def test_engine():
    """테스트 세션당 한 번 테스트 데이터베이스 엔진을 생성합니다.
    scope='session'은 테스트마다가 아닌 한 번만 실행됩니다."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)

    # 모델에 정의된 모든 테이블을 생성합니다
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    # 정리: 모든 테이블 삭제
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
async def db_session(test_engine):
    """각 테스트를 위한 새 데이터베이스 세션을 생성합니다.
    테스트 후 롤백되는 트랜잭션을 사용하여
    테스트 간 완전한 격리를 보장합니다."""
    session_factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        # 중첩 트랜잭션(세이브포인트) 시작
        async with session.begin():
            yield session
            # 테스트 후 롤백 -- 테스트 간 데이터가 유지되지 않습니다
            await session.rollback()

@pytest.fixture
async def async_client(db_session: AsyncSession):
    """테스트 데이터베이스가 주입된 비동기 테스트 클라이언트를 생성합니다.
    실제 get_db 의존성을 오버라이드하여 테스트에서
    프로덕션 데이터베이스 대신 테스트 데이터베이스를 사용합니다."""

    async def override_get_db():
        yield db_session

    # 실제 데이터베이스 의존성을 테스트 버전으로 교체합니다
    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client

    # 정리: 오버라이드를 제거하여 다른 테스트에 영향을 주지 않습니다
    app.dependency_overrides.clear()

@pytest.fixture
def sample_user_data() -> dict:
    """재사용 가능한 테스트 데이터. 상수 대신 픽스처를 사용하면
    각 테스트가 새로운 복사본을 받습니다 (딕셔너리는 변경 가능하기 때문)."""
    return {
        "username": "testuser",
        "email": "testuser@example.com",
        "password": "securepassword123",
        "full_name": "Test User",
    }
```

### 픽스처 스코프(Scope)

| 스코프 | 생성 | 소멸 | 사용 사례 |
|-------|---------|-----------|----------|
| `function` (기본값) | 테스트 함수마다 | 테스트 종료 후 | DB 세션, 테스트 데이터 |
| `class` | 테스트 클래스마다 | 클래스 종료 후 | 공유 클래스 설정 |
| `module` | 테스트 모듈마다 | 모듈 종료 후 | 파일당 비용이 큰 설정 |
| `session` | 테스트 실행당 한 번 | 전체 테스트 종료 후 | 엔진, 커넥션 풀 |

---

## 4. 데이터베이스 픽스처

### 전략: 트랜잭션 롤백

데이터베이스 테스트 격리를 위한 가장 효과적인 패턴은 **트랜잭션 롤백**입니다. 각 테스트를 트랜잭션으로 감싸고 이후에 롤백합니다. 테스트마다 테이블을 생성/삭제하는 것보다 훨씬 빠릅니다.

```
테스트 시작
    │
    ▼
┌─────────────────────────┐
│  BEGIN TRANSACTION       │  ◀── 세이브포인트 생성
│                          │
│  INSERT user ...         │  ◀── 테스트가 데이터 생성
│  SELECT * FROM users ... │  ◀── 테스트가 데이터 읽기
│  assert len(users) == 1  │  ◀── 테스트 통과
│                          │
│  ROLLBACK                │  ◀── 모든 변경사항 취소
└─────────────────────────┘
    │
    ▼
테스트 종료 (데이터베이스 변경 없음)
```

### 테스트 데이터 시딩(Seeding)

```python
# tests/conftest.py (계속)
from app.models import User, Post
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@pytest.fixture
async def sample_user(db_session: AsyncSession) -> User:
    """테스트 데이터베이스에 사용자를 생성합니다.
    이 픽스처는 db_session에 의존하므로 사용자는
    테스트 후 자동으로 롤백됩니다."""
    user = User(
        username="alice",
        email="alice@example.com",
        hashed_password=pwd_context.hash("password123"),
        full_name="Alice Johnson",
    )
    db_session.add(user)
    await db_session.flush()  # ID 생성
    await db_session.refresh(user)
    return user

@pytest.fixture
async def sample_posts(db_session: AsyncSession, sample_user: User) -> list[Post]:
    """목록/필터 엔드포인트 테스트를 위해 여러 게시물을 생성합니다.
    외래 키(foreign key) 제약 조건을 만족시키기 위해 sample_user에 의존합니다."""
    posts = [
        Post(title="First Post", content="Content 1", author_id=sample_user.id, is_published=True),
        Post(title="Second Post", content="Content 2", author_id=sample_user.id, is_published=True),
        Post(title="Draft Post", content="Content 3", author_id=sample_user.id, is_published=False),
    ]
    db_session.add_all(posts)
    await db_session.flush()
    for post in posts:
        await db_session.refresh(post)
    return posts
```

### 테스트에서 데이터 픽스처 사용하기

```python
# tests/test_posts.py
import pytest

pytestmark = pytest.mark.anyio

async def test_list_published_posts(async_client, sample_posts):
    """sample_posts 픽스처는 3개의 게시물(2개 공개, 1개 임시저장)을 생성합니다.
    목록 엔드포인트는 공개된 게시물만 반환해야 합니다."""
    response = await async_client.get("/api/posts?published=true")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all(post["is_published"] for post in data)

async def test_get_nonexistent_post(async_client):
    """sample_posts 픽스처 없이는 데이터베이스가 비어 있습니다.
    존재하지 않는 게시물을 요청하면 404를 반환해야 합니다."""
    response = await async_client.get("/api/posts/999")
    assert response.status_code == 404
```

---

## 5. dependency_overrides를 활용한 의존성 목킹

`dependency_overrides`는 테스트 중 의존성을 교체하기 위한 FastAPI의 내장 메커니즘입니다. 의존성 callable을 테스트 더블(test double)로 교체합니다.

### 기본 오버라이드

```python
from app.dependencies import get_current_user
from app.models import User

# 실제 인증을 우회하는 가짜 사용자 생성
mock_user = User(
    id=1,
    username="testadmin",
    email="admin@test.com",
    hashed_password="not-a-real-hash",
    is_active=True,
)

async def mock_get_current_user():
    """JWT 토큰 확인 없이 가짜 사용자를 반환합니다.
    모든 테스트에서 실제 인증을 처리하지 않고도
    보호된 엔드포인트를 테스트할 수 있게 해줍니다."""
    return mock_user

# 테스트 실행 전 오버라이드
app.dependency_overrides[get_current_user] = mock_get_current_user
```

### 픽스처에서 오버라이드 사용하기

```python
# tests/conftest.py
import pytest
from app.main import app
from app.dependencies import get_current_user, get_current_active_user

@pytest.fixture
def authenticated_client(async_client, sample_user):
    """sample_user로 자동 인증되는 클라이언트입니다.
    인증 의존성을 오버라이드하여 보호된 엔드포인트가 작동합니다."""

    async def override_auth():
        return sample_user

    app.dependency_overrides[get_current_active_user] = override_auth
    yield async_client
    # 테스트 후 정리
    del app.dependency_overrides[get_current_active_user]

@pytest.fixture
def admin_client(async_client):
    """관리자 사용자로 인증된 클라이언트입니다."""

    async def override_auth():
        return User(
            id=99,
            username="admin",
            email="admin@test.com",
            hashed_password="",
            is_active=True,
            role="admin",
        )

    app.dependency_overrides[get_current_active_user] = override_auth
    yield async_client
    del app.dependency_overrides[get_current_active_user]
```

### 오버라이드를 사용한 테스트

```python
# tests/test_protected.py
pytestmark = pytest.mark.anyio

async def test_get_my_profile(authenticated_client):
    """목킹된 인증으로 /users/me 엔드포인트를 테스트합니다."""
    response = await authenticated_client.get("/api/users/me")
    assert response.status_code == 200
    assert response.json()["username"] == "alice"

async def test_admin_only_endpoint(admin_client):
    """관리자 오버라이드로 관리자 엔드포인트가 작동하는지 테스트합니다."""
    response = await admin_client.get("/api/admin/stats")
    assert response.status_code == 200

async def test_unauthorized_without_override(async_client):
    """인증 오버라이드 없이 보호된 엔드포인트는 401을 반환합니다."""
    response = await async_client.get("/api/users/me")
    assert response.status_code == 401
```

### 외부 서비스 오버라이드

```python
from app.dependencies import get_email_service

class MockEmailService:
    """이메일을 실제로 보내는 대신 기록만 하는 모의 객체입니다.
    올바른 이메일이 발송되었는지 확인하는 데 유용합니다."""
    def __init__(self):
        self.sent_emails: list[dict] = []

    async def send(self, to: str, subject: str, body: str):
        self.sent_emails.append({"to": to, "subject": subject, "body": body})

@pytest.fixture
def mock_email(async_client):
    """모의 이메일 서비스를 주입하고 검증을 위해 노출합니다."""
    service = MockEmailService()

    async def override():
        return service

    app.dependency_overrides[get_email_service] = override
    yield service
    del app.dependency_overrides[get_email_service]

async def test_registration_sends_welcome_email(async_client, mock_email):
    """사용자 등록 시 환영 이메일이 발송되는지 확인합니다."""
    await async_client.post(
        "/api/users",
        json={"username": "newuser", "email": "new@test.com", "password": "secret123"},
    )
    assert len(mock_email.sent_emails) == 1
    assert mock_email.sent_emails[0]["to"] == "new@test.com"
    assert "welcome" in mock_email.sent_emails[0]["subject"].lower()
```

---

## 6. 인증 흐름 테스트

### 로그인 흐름 엔드-투-엔드(End-to-End) 테스트

```python
# tests/test_auth.py
import pytest

pytestmark = pytest.mark.anyio

async def test_login_success(async_client, sample_user):
    """전체 로그인 흐름 테스트: 자격 증명을 보내고 JWT를 받습니다."""
    # OAuth2 패스워드 흐름은 JSON이 아닌 폼 데이터를 사용합니다
    response = await async_client.post(
        "/token",
        data={  # data=는 JSON이 아닌 폼 인코딩 방식으로 전송합니다
            "username": "alice",
            "password": "password123",
        },
    )
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"

async def test_login_wrong_password(async_client, sample_user):
    """잘못된 비밀번호는 오류 응답 본문이 있는 200이 아닌 401을 반환해야 합니다."""
    response = await async_client.post(
        "/token",
        data={"username": "alice", "password": "wrongpassword"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

async def test_login_nonexistent_user(async_client):
    """존재하지 않는 사용자는 사용자명 열거(enumeration) 공격을 방지하기 위해
    잘못된 비밀번호와 동일한 오류를 반환해야 합니다."""
    response = await async_client.post(
        "/token",
        data={"username": "nonexistent", "password": "anything"},
    )
    assert response.status_code == 401

async def test_protected_endpoint_with_token(async_client, sample_user):
    """전체 통합 테스트: 로그인, 토큰 획득, 보호된 라우트 접근."""
    # 1단계: 로그인
    login_response = await async_client.post(
        "/token",
        data={"username": "alice", "password": "password123"},
    )
    token = login_response.json()["access_token"]

    # 2단계: 토큰으로 보호된 엔드포인트 접근
    response = await async_client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["username"] == "alice"

async def test_expired_token(async_client):
    """만료된 토큰이 올바르게 거부되는지 테스트합니다."""
    # 과거에 만료된 토큰을 생성합니다
    from app.auth import create_access_token
    from datetime import timedelta

    expired_token = create_access_token(
        data={"sub": "alice"},
        expires_delta=timedelta(seconds=-1),  # 이미 만료됨
    )
    response = await async_client.get(
        "/api/users/me",
        headers={"Authorization": f"Bearer {expired_token}"},
    )
    assert response.status_code == 401

async def test_malformed_token(async_client):
    """완전히 유효하지 않은 토큰은 401을 반환해야 합니다."""
    response = await async_client.get(
        "/api/users/me",
        headers={"Authorization": "Bearer not.a.valid.jwt.token"},
    )
    assert response.status_code == 401

async def test_missing_auth_header(async_client):
    """Authorization 헤더가 전혀 없는 경우 401을 반환해야 합니다."""
    response = await async_client.get("/api/users/me")
    assert response.status_code == 401
```

### 인증 요청을 위한 헬퍼 픽스처

```python
# tests/conftest.py
@pytest.fixture
async def auth_headers(async_client, sample_user) -> dict[str, str]:
    """sample_user로 로그인하고 인증 헤더를 반환합니다.
    모킹된 인증이 아닌 실제 JWT 토큰이 필요할 때 이 픽스처를 사용합니다."""
    response = await async_client.post(
        "/token",
        data={"username": "alice", "password": "password123"},
    )
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}

# 테스트에서 사용 예:
async def test_create_post_authenticated(async_client, auth_headers):
    response = await async_client.post(
        "/api/posts",
        json={"title": "My Post", "content": "Hello world"},
        headers=auth_headers,
    )
    assert response.status_code == 201
```

---

## 7. 커버리지 리포트

코드 커버리지(coverage)는 테스트 중 어떤 코드 라인이 실행되는지를 알려줍니다. 테스트되지 않은 경로를 찾는 데 도움이 되지만, 100% 커버리지가 정확성을 보장하지는 않습니다.

### 설정

```bash
pip install pytest-cov
```

### 커버리지로 테스트 실행

```bash
# 커버리지 리포트와 함께 모든 테스트 실행
pytest --cov=app --cov-report=term-missing tests/

# 출력 예시:
# Name                    Stmts   Miss  Cover   Missing
# -------------------------------------------------------
# app/__init__.py             0      0   100%
# app/main.py                15      0   100%
# app/models.py              32      2    94%   45-46
# app/routers/users.py       48      5    90%   72-76
# app/dependencies.py        18      0   100%
# -------------------------------------------------------
# TOTAL                     113      7    94%
```

### pyproject.toml 설정

```toml
# pyproject.toml
[tool.pytest.ini_options]
# 비동기 테스트를 위해 자동으로 asyncio 모드 사용
asyncio_mode = "auto"
# 기본 pytest 인수 -- 항상 커버리지와 함께 실행
addopts = [
    "--cov=app",
    "--cov-report=term-missing",
    "--cov-report=html:htmlcov",  # HTML 리포트 생성
    "-v",                          # 상세 출력
]
# 테스트 파일 탐색 패턴
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]

[tool.coverage.run]
# 측정할 소스 디렉토리
source = ["app"]
# 커버리지가 필요 없는 파일 제외
omit = [
    "app/config.py",       # 설정 -- 암묵적으로 테스트됨
    "alembic/*",           # 마이그레이션 스크립트
    "tests/*",             # 테스트 코드 자체는 측정하지 않음
]

[tool.coverage.report]
# 커버리지가 이 임계값 아래로 떨어지면 실패
fail_under = 85
# 의도적으로 테스트하지 않는 라인
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.",
    "pass",
    "raise NotImplementedError",
]
# 커버되지 않은 라인 표시
show_missing = true
```

### HTML 커버리지 리포트

```bash
# HTML 리포트 생성 및 열기
pytest --cov=app --cov-report=html tests/
open htmlcov/index.html  # macOS
# xdg-open htmlcov/index.html  # Linux
```

HTML 리포트는 커버된 라인을 초록색으로, 누락된 라인을 빨간색으로 강조 표시하여 테스트되지 않은 분기(branch)를 쉽게 찾을 수 있습니다.

### 커버리지 모범 사례

| 실천 방법 | 이유 |
|----------|--------|
| 85~90% 커버리지 목표 | 이 이상은 수익 감소; 100%는 종종 사소한 코드에 노력을 낭비함 |
| 분기 커버리지에 집중 | 라인 커버리지는 테스트되지 않은 `if/else` 분기를 놓칩니다 |
| 오류 경로 커버 | `except` 블록과 `raise HTTPException`은 종종 누락됩니다 |
| 프레임워크 코드는 테스트하지 말 것 | FastAPI가 잘못된 입력에 422를 반환하는지 테스트할 필요 없음 |
| `# pragma: no cover` 절약해서 사용 | 진정으로 테스트할 수 없는 코드(플랫폼별, 디버그 전용)에만 사용 |

---

## 8. 연습 문제

### 문제 1: CRUD 테스트 스위트

`POST /items`, `GET /items`, `GET /items/{id}`, `PUT /items/{id}`, `DELETE /items/{id}` API에 대한 완전한 테스트 스위트를 작성하세요. 포함 내용:
- 각 엔드포인트의 정상 경로(happy path) 테스트
- 엣지 케이스: 필드 누락 생성, 존재하지 않는 아이템 조회, 두 번 삭제
- 데이터 격리: 한 테스트의 데이터가 다른 테스트로 누출되지 않음을 증명

### 문제 2: 데이터베이스 테스트 격리

테스트 격리를 위한 트랜잭션 롤백 패턴을 구현하세요:
1. 엔진, 세션, 클라이언트 픽스처가 있는 `conftest.py` 생성
2. 동일한 사용자명으로 사용자를 생성하는 두 테스트 작성
3. 충돌하지 않음을 증명 (각 테스트의 데이터가 롤백되므로 모두 통과)
4. 각 테스트 시작 시 데이터베이스가 비어있음을 확인하는 테스트 작성

### 문제 3: 의존성 오버라이드 테스트

외부 날씨 API에 의존하는 엔드포인트가 주어졌을 때:

```python
async def get_weather_service():
    return RealWeatherService(api_key="...")

@app.get("/forecast/{city}")
async def get_forecast(city: str, weather=Depends(get_weather_service)):
    return await weather.get_forecast(city)
```

다음을 테스트하는 코드를 작성하세요:
1. `get_weather_service`를 예측 가능한 데이터를 반환하는 모의 객체로 오버라이드
2. 정상 시나리오 테스트 (유효한 도시, 좋은 데이터)
3. 비정상 시나리오 테스트 (모의 객체가 API 오류를 발생시킴)
4. 테스트 중 실제 서비스가 호출되지 않음을 테스트

### 문제 4: 인증 테스트 매트릭스

모든 인증 시나리오를 다루는 테스트 매트릭스를 작성하세요:

| 시나리오 | 예상 상태 코드 |
|----------|----------------|
| 유효한 토큰, 활성 사용자 | 200 |
| 유효한 토큰, 비활성 사용자 | 400 |
| 만료된 토큰 | 401 |
| 잘못된 형식의 토큰 | 401 |
| Authorization 헤더 없음 | 401 |
| 잘못된 토큰 타입 ("Bearer" 대신 "Basic") | 401 |
| 잘못된 시크릿 키로 서명된 토큰 | 401 |

유효하지 않은 토큰 시나리오에 대해 `@pytest.mark.parametrize`를 사용한 파라미터화 테스트를 작성하세요.

### 문제 5: 커버리지 갭 분석

다음과 같이 테스트되지 않은 라인이 있는 FastAPI 앱이 주어졌을 때:

```
app/routers/users.py    48      5    90%   72-76
app/routers/posts.py    65     12    82%   34-39, 88-93
```

1. "Missing" 열이 제공하는 정보를 설명하세요
2. 누락된 라인 중 우선순위를 결정하는 전략을 설명하세요
3. 72~76번 라인을 커버할 테스트 케이스를 작성하세요 (중복 이메일 등록에 대한 오류 처리 블록이라고 가정)
4. 라인을 커버하지 않는 것이 허용 가능한 경우를 설명하세요

---

## 9. 참고 자료

- [FastAPI 테스트 가이드](https://fastapi.tiangolo.com/tutorial/testing/)
- [pytest 공식 문서](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [httpx 공식 문서](https://www.python-httpx.org/)
- [pytest-cov 공식 문서](https://pytest-cov.readthedocs.io/)
- [Coverage.py 설정](https://coverage.readthedocs.io/en/latest/config.html)
- [FastAPI 의존성 오버라이드](https://fastapi.tiangolo.com/advanced/testing-dependencies/)

---

**이전**: [FastAPI 데이터베이스](./04_FastAPI_Database.md) | **다음**: [Express 기초](./06_Express_Basics.md)
