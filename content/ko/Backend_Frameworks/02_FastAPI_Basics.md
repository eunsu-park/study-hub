# 02. FastAPI 기초(FastAPI Basics)

**이전**: [백엔드 웹 기초](./01_Backend_Web_Fundamentals.md) | **다음**: [FastAPI 고급](./03_FastAPI_Advanced.md)

**난이도**: ⭐⭐

---

## 학습 목표

- 타입이 지정된 경로 및 쿼리 파라미터를 사용하여 최소한의 FastAPI 애플리케이션을 구축할 수 있다
- 유효성 검사를 포함한 Pydantic v2 모델을 사용하여 요청 및 응답 스키마를 정의할 수 있다
- FastAPI가 타입 힌트(type hints)에서 OpenAPI 문서를 자동으로 생성하는 방법을 설명할 수 있다
- 프론트엔드 애플리케이션의 교차 출처(cross-origin) 요청을 허용하도록 CORS 미들웨어를 구성할 수 있다
- CRUD 엔드포인트에 적절한 HTTP 상태 코드와 응답 모델을 구현할 수 있다

---

## 목차

프레임워크 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. ASGI가 무엇인지, FastAPI가 왜 Starlette(HTTP 계층)와 Pydantic(데이터 계층)을 분리하는지, 그리고 타입 힌트가 어떻게 스키마 인트로스펙션을 통해 런타임 검증으로 바뀌는지를 다룹니다.

1. [FastAPI란 무엇인가](#1-fastapi란-무엇인가)
2. [설치 및 첫 번째 앱](#2-설치-및-첫-번째-앱)
3. [경로 파라미터와 쿼리 파라미터](#3-경로-파라미터와-쿼리-파라미터)
4. [Pydantic v2를 사용한 요청 본문](#4-pydantic-v2를-사용한-요청-본문)
5. [응답 모델과 상태 코드](#5-응답-모델과-상태-코드)
6. [자동 OpenAPI 문서](#6-자동-openapi-문서)
7. [CORS 미들웨어](#7-cors-미들웨어)
8. [연습 문제](#8-연습-문제)
9. [참고 자료](#9-참고-자료)

---

## 이론과 원리

FastAPI는 *단일* 프레임워크가 아닙니다. 세 개의 독립된 계층 — **ASGI**(와이어 전송), **Starlette**(HTTP 라우팅과 미들웨어), **Pydantic v2**(데이터 검증과 직렬화) — 의 의도된 합성입니다. 각 계층이 무엇을 책임지고, 계층 사이의 경계를 무엇이 가로지르는지 이해하는 것이, FastAPI를 마법으로 다루느냐 확장 가능한 글루로 다루느냐의 차이를 만듭니다.

- **(A) ASGI** — WSGI를 대체하는 비동기 인지 Python 웹 서버 인터페이스.
- **(B) Starlette와 Pydantic의 분리** — HTTP와 데이터 검증이 다른 관심사인 이유.
- **(C) 스키마 원천으로서의 타입 힌트** — Python 어노테이션이 어떻게 JSON Schema, OpenAPI, 런타임 검증기로 바뀌는가.

### A. ASGI: 비동기 서버 인터페이스

ASGI(Asynchronous Server Gateway Interface)는 서버(Uvicorn, Hypercorn, Daphne)와 애플리케이션(FastAPI, Starlette, Django Channels)이 대화하는 계약입니다. WSGI의 정신적 후속이며, 중요한 구조적 변화 세 가지가 있습니다.

#### A.1 단일 ASGI callable

ASGI 앱은 고정된 3-인자 시그니처의 비동기 callable 하나입니다.

```python
async def app(scope, receive, send):
    ...
```

- `scope`은 연결을 기술하는 dict입니다: `{"type": "http", "method": "GET", "path": "/", "headers": [...], ...}`. `type` 필드가 분기자입니다.
- `receive`는 클라이언트로부터 다음 이벤트(요청 본문 청크, WebSocket 메시지, 연결 종료 알림 등)를 가져오기 위해 앱이 `await`하는 비동기 함수입니다.
- `send`는 이벤트(응답 시작, 응답 본문 청크, WebSocket 메시지)를 다시 보내기 위해 앱이 `await`하는 비동기 함수입니다.

전체 프로토콜 — HTTP 요청/응답, WebSocket 생명주기, lifespan 시작/종료 이벤트 — 이 모두 `receive`와 `send`를 흐르는 JSON 모양의 Python dict로 인코딩됩니다. 콜백 등록도, 이벤트 emitter도, 스레딩도 없습니다. 양쪽에서 그저 `await`만 있을 뿐입니다.

#### A.2 ASGI가 WSGI를 대체한 이유

WSGI(PEP 3333)는 설계상 동기였습니다. 애플리케이션 시그니처는 다음과 같았습니다.

```python
def app(environ, start_response):
    start_response("200 OK", [...])
    return [b"Hello"]
```

이로 인해 할 수 없는 일이 세 가지 있었습니다.

1. **WebSocket.** WSGI는 요청/응답만 모델링합니다. WebSocket은 장기 양방향 채널이 필요합니다.
2. **스트리밍 응답.** WSGI는 바이트의 iterable을 반환하지만, 반복은 동기입니다. 느린 upstream을 기다리는 스트리밍 응답은 워커를 블로킹합니다.
3. **`async def` 핸들러.** 코루틴 함수는 바이트가 아니라 코루틴 객체를 반환합니다. WSGI 서버는 이를 구동할 줄 모릅니다.

ASGI는 애플리케이션 자체를 코루틴으로 만들고 메시지 교환을 명시화하여 세 가지 모두를 해결합니다. 서버는 단일 스레드에서 수천 개의 ASGI 앱을 동시에 돌릴 수 있는데, 이는 `await receive()`가 다른 모든 I/O와 마찬가지로 이벤트 루프에 제어권을 양보하기 때문입니다.

#### A.3 HTTP에 대한 ASGI 요청 생명주기

하나의 HTTP 요청에 대해 서버는 보내고 앱은 받습니다.

```
{"type": "http.request", "body": b"...", "more_body": True}
{"type": "http.request", "body": b"...", "more_body": False}
```

앱이 다시 보내는 것은 다음과 같습니다.

```
{"type": "http.response.start", "status": 200, "headers": [...]}
{"type": "http.response.body", "body": b"...", "more_body": True}
{"type": "http.response.body", "body": b"...", "more_body": False}
```

헤더는 첫 본문 청크 이전에 보내야 합니다. 본문은 스트리밍을 위해 여러 청크로 나눌 수 있습니다. 이것이 별도의 서버 협력 없이 `StreamingResponse`와 Server-Sent Events를 가능하게 하는 메커니즘입니다 — 그것들은 루프 안에서 방출되는 평범한 ASGI 바이트 청크일 뿐입니다.

### B. Starlette와 Pydantic의 분리

FastAPI의 가장 중요한 아키텍처 결정은 *직접 하지 않는 것*이 무엇인가입니다. 웹 프레임워크의 두 큰 일 — HTTP 처리와 데이터 검증 — 은 두 개의 다른 라이브러리가 소유합니다. FastAPI는 그들을 엮어 타입 힌트로 노출하는 계층입니다.

#### B.1 Starlette가 책임지는 것

Starlette는 FastAPI 아래의 ASGI 앱입니다. HTTP 모양의 모든 것을 책임집니다.

- **라우팅** — `request.path`를 등록된 URL 패턴과 매칭.
- **미들웨어** — 양파 모양의 요청/응답 변환(CORS, GZip, 세션, 인증).
- **Request와 Response 객체** — 원시 ASGI 메시지에 대한 타입 래퍼.
- **WebSocket 지원** — 양방향 메시지 프로토콜.
- **백그라운드 태스크** — 응답 송신 후 실행되도록 스케줄된 작업.
- **TestClient** — 네트워크를 완전히 건너뛰는 in-process HTTP 클라이언트.

FastAPI 없이 Starlette를 직접 사용할 수도 있습니다. FastAPI는 그 위에 타입 계층을 더할 뿐입니다. 반대로, 모든 FastAPI 앱은 *동시에* Starlette 앱입니다 — `from fastapi import FastAPI; app = FastAPI()`는 `Starlette` 서브클래스를 반환하므로 모든 Starlette 미들웨어와 프리미티브가 그대로 동작합니다.

#### B.2 Pydantic이 책임지는 것

Pydantic v2는 Rust로 작성된 데이터 검증·직렬화 라이브러리입니다(v2 코어 `pydantic-core`가 Rust 크레이트입니다). 다음을 책임집니다.

- **스키마 정의** — 타입 힌트가 있는 Python 클래스 문법으로 데이터 모양 선언.
- **파싱** — 임의 입력(dict, JSON 바이트, query string)을 받아 타입이 있는 Python 객체로 변환.
- **검증** — 스키마와 일치하지 않는 입력을 구조화된 오류 보고와 함께 거부.
- **직렬화** — 타입이 있는 Python 객체를 다시 dict, JSON 등으로 변환.
- **JSON Schema 생성** — 모델을 기술하는 JSON Schema 문서 방출.

Pydantic은 HTTP에 대해 아무것도 모릅니다. 설정 파일, 메시지 큐 페이로드, ML 파이프라인 I/O 등 검증이 필요한 타입 데이터가 있는 어디서나 사용할 수 있습니다.

#### B.3 FastAPI 자체가 책임지는 것

FastAPI는 *글루*입니다. 다음을 받아서

```python
@app.post("/items/")
async def create_item(item: Item, q: int | None = None) -> Item:
    ...
```

시작 시점에 이 함수의 타입 힌트를 `inspect.signature`와 `typing.get_type_hints`로 검사합니다. 그 힌트로부터 다음을 구축합니다.

1. 생성된 래퍼를 호출하는 **Starlette 라우트**.
2. 요청 본문을 파싱하는 `Item`용 **Pydantic 검증기**.
3. 문자열을 `int`로 변환하고 `Optional` 의미를 적용하는 `q`용 **쿼리 파라미터 파서**.
4. 반환된 `Item`을 Pydantic을 통해 다시 JSON으로 변환하는 **응답 직렬화기**.
5. 같은 Pydantic 모델로부터 파생된 요청 본문 스키마와 응답 스키마를 포함하는, 이 라우트용 **OpenAPI 스키마 조각**.

핸들러 자신은 완전히 타입이 있는 `item: Item` 인자만 봅니다. 수동 `request.json()`도, 수동 `dict.get("title")`도, 누락 필드에 대한 수동 오류 처리도 없습니다. 검증, 직렬화, 문서화 모두가 IDE 자동 완성에 쓰이는 *같은* 타입 어노테이션에서 파생됩니다.

### C. 스키마 원천으로서의 타입 힌트

FastAPI의 가장 깊은 설계 아이디어는, 함수의 타입 시그니처가 *실행 가능한 명세*라는 것입니다. 원래 정적 분석 보조(mypy, pyright)였던 Python 타입 힌트가, 검증·문서·의존성 주입을 구동하는 런타임 메타데이터가 됩니다.

#### C.1 인트로스펙션 파이프라인

핸들러를 데코레이트하면

```python
async def get_item(item_id: int, q: str | None = Query(None, max_length=50)) -> ItemOut:
    ...
```

FastAPI는 앱 시작 시 다음 파이프라인을 돌립니다.

1. `inspect.signature(get_item)`이 인자별 `Parameter`를 가진 `Signature` 객체를 가져옵니다.
2. `typing.get_type_hints(get_item, include_extras=True)`가 문자열 어노테이션과 PEP 593 `Annotated` 부가 정보를 해석합니다.
3. 각 파라미터에 대해 FastAPI는 그 타입과 `Param` 마커(`Path`, `Query`, `Header`, `Body`, `Depends`)를 살핍니다.
   - `int`, `str` 같은 원시 스칼라 → 타입 강제 변환이 있는 쿼리/경로 파라미터.
   - Pydantic `BaseModel` → 요청 본문, Pydantic으로 파싱.
   - `Depends(callable)` → 의존성 주입 노드(레슨 03 참조).
4. 반환 어노테이션 `-> ItemOut`은 응답 모델이 됩니다 — 직렬화에 사용되고, OpenAPI에서는 `responses` 스키마로 사용됩니다.
5. 이 모든 것이 라우트에 캐시되므로, 요청당 작업은 그저 "캐시된 검증기와 캐시된 직렬화기를 호출"하는 것뿐입니다.

비용은 시작 시 한 번 치릅니다. 요청당 오버헤드는 Pydantic v2의 Rust 검증기가 지배하며, 손으로 쓴 `if isinstance(...)` 검사와 한 자리수 차이 안에 있습니다.

#### C.2 "컴파일 타임" 보장 — 그리고 그 한계

타입 힌트는 Python에서 진짜로 컴파일 타임이 아닙니다. 런타임 메타데이터입니다. 하지만 FastAPI가 시작 시점에 이를 검사하므로, 잘못된 시그니처는 첫 요청이 아니라 *애플리케이션 부팅* 시점에 실패합니다. 존재하지 않는 의존성을 참조하거나 순환 import가 있는 Pydantic 모델을 사용하면, 앱은 시작을 거부합니다. 이는 한 부류의 오류를 프로덕션에서 CI로 옮기는 효과가 있습니다.

타입 힌트가 잡지 못하는 것:

- **필드 간 검증.** "`end_date`는 `start_date` 이후여야 한다"는 타입이 아니라 Pydantic `@model_validator`가 필요합니다.
- **비즈니스 규칙 위반.** "사용자는 10개 이상의 아이템을 만들 수 없다"는 핸들러 안의 런타임 검사입니다.
- **외부 시스템 계약.** Pydantic은 모양을 검증할 뿐, 데이터베이스에 그 ID의 행이 있는지는 모릅니다.

올바른 멘탈 모델: 타입 힌트는 *구조적* 정확성을 자동으로 다루고, 의미적 정확성은 여전히 여러분의 일입니다.

#### C.3 OpenAPI: 같은 스키마, 두 출력

OpenAPI(예전 Swagger)는 모든 엔드포인트, 파라미터, 응답 모양을 기술하는 JSON 문서입니다. FastAPI는 이를 검증을 구동하는 같은 타입 힌트로부터 즉석에서 생성합니다. 이로써 단일 진실 원천(single source of truth)이 생깁니다. Pydantic 모델을 바꾸는 그 순간, 검증기, 직렬화기, `/docs`의 문서, 생성된 클라이언트 SDK가 모두 함께 갱신됩니다.

이는 Django REST Framework나 Express와 구조적으로 다릅니다. 거기서는 API와 그 문서가 별도의 산출물이며 서로 어긋나기 마련입니다. 어긋남은 절차의 실패가 아니라, 같은 사실에 대한 진실 원천이 둘이라는 자연스러운 결과입니다.

### 이론에서 아래 코드로

뒤에 나오는 각 절은 이 틀의 한 조각을 구체화합니다.

- §1 (FastAPI란 무엇인가)은 §B의 세 계층을 명명합니다.
- §2 (첫 앱)은 §A.1의 ASGI callable에 `@app.get` 데코레이터를 입힌 것입니다.
- §3 (경로/쿼리 파라미터)는 §C.1의 타입 힌트 인트로스펙션을 스칼라 인자에 적용한 것입니다.
- §4 (요청 본문)은 §B.2의 `BaseModel`을 §C.1의 Pydantic 검증기에 끼운 것입니다.
- §5 (응답 모델과 상태 코드)는 §C.1의 반환 어노테이션 절반에, 레슨 01의 §A.2 상태 코드 의미를 더한 것입니다.
- §6 (OpenAPI 문서)은 §C.3에서 기술한 스키마 방출 출력입니다.
- §7 (CORS)는 §B.1의 Starlette 미들웨어 하나를 FastAPI 표면을 통해 구성한 것입니다.

---

## 1. FastAPI란 무엇인가

FastAPI는 세 가지 기반 위에 구축된 현대적인 Python 웹 프레임워크입니다:

1. **타입 힌트(Type hints)** (Python 3.7+): 파라미터에 타입이 지정되어 자동 유효성 검사와 문서화가 가능합니다
2. **Starlette**: 아래에서 HTTP와 WebSocket 연결을 처리하는 ASGI 프레임워크
3. **Pydantic v2**: Python 타입 어노테이션을 사용한 데이터 유효성 검사와 직렬화

```
┌─────────────────────────────────┐
│         사용자 애플리케이션        │
│   (엔드포인트, 비즈니스 로직)     │
├─────────────────────────────────┤
│           FastAPI                │
│   (라우팅, DI, OpenAPI 생성)     │
├─────────────────────────────────┤
│          Starlette               │
│   (ASGI, 미들웨어, 응답)         │
├─────────────────────────────────┤
│      Pydantic v2                 │
│   (유효성 검사, 직렬화)          │
├─────────────────────────────────┤
│    Uvicorn (ASGI 서버)           │
│   (이벤트 루프, HTTP 파싱)        │
└─────────────────────────────────┘
```

### FastAPI를 선택하는 이유

| 기능 | Flask | Django REST | FastAPI |
|---------|-------|-------------|---------|
| 비동기 지원 | 제한적 (확장 필요) | 제한적 | 기본 제공 |
| 자동 유효성 검사 | 없음 | 시리얼라이저 | 타입 힌트 |
| 자동 문서 (OpenAPI) | 없음 (확장 필요) | 있음 (확장 필요) | 내장 |
| 성능 | ~1x | ~1x | ~3-5x |
| 학습 곡선 | 낮음 | 중간 | 낮음-중간 |

---

## 2. 설치 및 첫 번째 앱

### 설치

```bash
# 먼저 가상 환경 생성 -- 프로젝트 의존성을 격리합니다
python -m venv venv
source venv/bin/activate  # Windows의 경우: venv\Scripts\activate

# 모든 선택적 의존성을 포함하여 FastAPI 설치 (uvicorn 등)
pip install "fastapi[standard]"
```

### 최소한의 애플리케이션

```python
# main.py
from fastapi import FastAPI

# 애플리케이션 인스턴스 생성. title과 version은 자동 생성된 문서에 표시됩니다.
app = FastAPI(
    title="My First API",
    version="0.1.0",
    description="A simple API to learn FastAPI basics"
)

@app.get("/")
async def root():
    """루트 엔드포인트. FastAPI는 docstring을
    OpenAPI 문서의 엔드포인트 설명으로 사용합니다."""
    return {"message": "Hello, World!"}

@app.get("/health")
async def health_check():
    """로드 밸런서와 모니터링을 위한 헬스 체크 엔드포인트.
    서비스가 실행 중이면 200을 반환합니다."""
    return {"status": "healthy"}
```

### 서버 실행

```bash
# --reload는 파일 변경을 감지하여 자동으로 재시작합니다
# --reload는 개발 환경에서만 사용 -- 오버헤드가 추가됩니다
uvicorn main:app --reload --port 8000

# 프로덕션: 다중 워커 사용, reload 없음
# uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

시작 후 다음 URL에 접속할 수 있습니다:
- `http://localhost:8000` -- API
- `http://localhost:8000/docs` -- Swagger UI (인터랙티브 문서)
- `http://localhost:8000/redoc` -- ReDoc (대안 문서)

---

## 3. 경로 파라미터와 쿼리 파라미터

### 경로 파라미터(Path Parameters)

경로 파라미터는 URL의 일부로 **필수**입니다. FastAPI는 이를 선언된 타입으로 자동 변환합니다.

```python
from fastapi import FastAPI, Path

app = FastAPI()

@app.get("/users/{user_id}")
async def get_user(
    # Path()는 유효성 검사 제약 조건과 문서 메타데이터를 추가합니다
    user_id: int = Path(
        ...,  # ...은 필수를 의미합니다 (Ellipsis)
        title="User ID",
        description="The unique identifier of the user",
        gt=0,  # greater than 0 (0보다 커야 함)
        examples=[42]
    )
):
    """ID로 사용자를 조회합니다.
    user_id가 유효한 int가 아니면 FastAPI가 자동으로 422를 반환합니다."""
    return {"user_id": user_id, "name": f"User {user_id}"}


# 타입 강제 적용이 있는 다중 경로 파라미터
@app.get("/users/{user_id}/posts/{post_id}")
async def get_user_post(user_id: int, post_id: int):
    return {"user_id": user_id, "post_id": post_id}
```

### Enum을 사용한 미리 정의된 값

```python
from enum import Enum

class UserRole(str, Enum):
    """str을 상속하면 값이 JSON 직렬화 가능해지고
    경로 매칭에서 문자열 비교가 가능해집니다."""
    admin = "admin"
    editor = "editor"
    viewer = "viewer"

@app.get("/users/role/{role}")
async def get_users_by_role(role: UserRole):
    # FastAPI는 role이 enum 값 중 하나인지 검증합니다
    # 유효하지 않은 값은 자동으로 명확한 오류 메시지와 함께 422를 반환합니다
    return {"role": role, "message": f"Listing {role.value} users"}
```

### 쿼리 파라미터(Query Parameters)

쿼리 파라미터는 URL에서 `?` 뒤에 옵니다. 경로에 선언되지 않은 파라미터는 자동으로 쿼리 파라미터로 처리됩니다.

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/users")
async def list_users(
    # 기본값이 있으면 쿼리 파라미터가 선택 사항이 됩니다
    skip: int = Query(
        default=0,
        ge=0,  # >= 0이어야 함
        description="Number of records to skip"
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=100,  # 클라이언트가 너무 많은 레코드를 요청하는 것을 방지
        description="Maximum number of records to return"
    ),
    # 선택적 파라미터는 None을 기본값으로 사용
    role: str | None = Query(
        default=None,
        min_length=2,
        max_length=20,
        description="Filter by user role"
    ),
    # 리스트 쿼리 파라미터: /users?tag=python&tag=api
    tags: list[str] = Query(default=[]),
):
    """페이지네이션과 선택적 필터링으로 사용자를 나열합니다.
    예: /users?skip=0&limit=20&role=admin"""
    result = {"skip": skip, "limit": limit}
    if role:
        result["role_filter"] = role
    if tags:
        result["tag_filter"] = tags
    return result
```

### 경로 vs 쿼리 파라미터 요약

```
GET /users/42/posts?page=2&sort=date
     ├──────┘       ├────┘  ├───────┘
     경로 파라미터   쿼리    쿼리
     (필수)         파라미터 파라미터
                    (선택)   (선택)
```

---

## 4. Pydantic v2를 사용한 요청 본문

`POST`, `PUT`, `PATCH` 요청에서 클라이언트는 요청 본문에 데이터를 전송합니다. FastAPI는 Pydantic 모델을 사용하여 이 데이터를 검증하고 파싱합니다.

### 기본 Pydantic 모델

```python
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class UserCreate(BaseModel):
    """새 사용자를 생성하기 위한 스키마.
    Pydantic v2는 인스턴스 생성 시 모든 필드를 검증하고
    데이터가 스키마와 일치하지 않으면 명확한 오류를 발생시킵니다."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        examples=["Alice Johnson"],
        description="User's full name"
    )
    email: str = Field(
        ...,
        pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",  # 정규식 검증
        examples=["alice@example.com"]
    )
    age: int | None = Field(
        default=None,
        ge=0,
        le=150,
        description="User's age (optional)"
    )

    # Pydantic v2는 v1의 @validator 대신 @field_validator를 사용합니다
    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v: str) -> str:
        """공백을 제거하고 이름이 비어 있지 않은지 확인합니다.
        검증기는 타입 검사 후, 모델이 생성되기 전에 실행됩니다."""
        stripped = v.strip()
        if not stripped:
            raise ValueError("Name cannot be empty or whitespace-only")
        return stripped


class UserResponse(BaseModel):
    """클라이언트에 사용자 데이터를 반환하기 위한 스키마.
    응답에는 id, created_at 같은 서버 생성 필드가 포함되므로
    UserCreate와 별도로 정의합니다."""
    id: int
    name: str
    email: str
    age: int | None = None
    created_at: datetime

    # Pydantic v2는 class Config 대신 model_config를 사용합니다
    model_config = {
        "from_attributes": True  # ORM 객체로부터 생성 허용
    }
```

### 엔드포인트에서 모델 사용

```python
from fastapi import FastAPI, status

app = FastAPI()

# 데모를 위한 인메모리 저장소
users_db: dict[int, dict] = {}
next_id = 1

@app.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED
)
async def create_user(user: UserCreate):
    """새 사용자를 생성합니다.
    FastAPI가 자동으로:
    1. JSON 본문을 UserCreate 인스턴스로 파싱
    2. 모든 필드를 검증 (실패 시 422 반환)
    3. 응답을 UserResponse로 필터링"""
    global next_id
    now = datetime.now()
    user_data = {
        "id": next_id,
        **user.model_dump(),  # Pydantic v2: model_dump()가 dict()를 대체
        "created_at": now
    }
    users_db[next_id] = user_data
    next_id += 1
    return user_data
```

### 중첩 모델

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str = "US"
    zip_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")

class UserWithAddress(BaseModel):
    """Pydantic은 중첩 모델을 재귀적으로 검증합니다.
    address.zip_code가 유효하지 않으면 오류 메시지에
    전체 경로가 포함됩니다: body -> address -> zip_code."""
    name: str
    email: str
    address: Address  # 중첩 모델
    tags: list[str] = []  # 기본값이 있는 문자열 리스트

@app.post("/users-with-address")
async def create_user_with_address(user: UserWithAddress):
    return user
```

### 요청 본문 예제

```json
{
    "name": "Alice Johnson",
    "email": "alice@example.com",
    "address": {
        "street": "123 Main St",
        "city": "Springfield",
        "country": "US",
        "zip_code": "62704"
    },
    "tags": ["admin", "premium"]
}
```

---

## 5. 응답 모델과 상태 코드

### 응답 모델 필터링

응답 모델은 클라이언트에 전송되는 데이터를 제어합니다. 이는 보안에 매우 중요합니다 -- 비밀번호 해시나 내부 필드를 실수로 노출하는 일은 절대 없어야 합니다.

```python
from pydantic import BaseModel, EmailStr

class UserInDB(BaseModel):
    """민감한 필드가 포함된 내부 표현."""
    id: int
    name: str
    email: str
    hashed_password: str  # 절대 노출하면 안 됩니다!
    is_active: bool
    internal_notes: str  # 관리자 전용 필드

class UserPublic(BaseModel):
    """공개 표현 -- 안전한 필드만 포함."""
    id: int
    name: str
    email: str
    is_active: bool

@app.get(
    "/users/{user_id}",
    response_model=UserPublic,  # hashed_password와 internal_notes를 필터링
    response_model_exclude_none=True  # None 값인 필드를 생략
)
async def get_user(user_id: int):
    # 함수가 모든 필드를 반환하더라도, 응답에는 UserPublic의 필드만
    # 나타납니다. 이는 안전망 역할을 합니다.
    user = get_user_from_db(user_id)
    return user
```

### 다중 응답 상태 코드

```python
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "User created successfully"},
        409: {"description": "Email already registered"},
        422: {"description": "Validation error in request body"},
    }
)
async def create_user(user: UserCreate):
    """responses 파라미터를 사용하면 OpenAPI 명세에 모든 가능한 상태 코드가
    문서화되어 API 소비자가 오류 케이스를 이해하는 데 도움이 됩니다."""
    existing = find_user_by_email(user.email)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email {user.email} is already registered"
        )
    return save_user(user)

@app.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT  # 응답에 본문 없음
)
async def delete_user(user_id: int):
    """204 No Content는 성공적인 DELETE의 표준입니다.
    응답에는 본문이 없습니다 -- 상태 코드만 있습니다."""
    user = find_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    remove_user(user_id)
    # 아무것도 반환하지 않음 -- FastAPI가 자동으로 204를 전송
```

---

## 6. 자동 OpenAPI 문서

FastAPI는 타입 힌트, docstring, 메타데이터로부터 OpenAPI 3.1 스키마를 생성합니다. 이 스키마는 두 가지 인터랙티브 문서 UI를 구동합니다.

### Swagger UI (`/docs`)

```
┌──────────────────────────────────────────────┐
│  My First API v0.1.0                          │
│                                               │
│  ▼ users                                      │
│    GET  /users         사용자 목록             │
│    POST /users         새 사용자 생성          │
│    GET  /users/{id}    ID로 사용자 조회        │
│    PUT  /users/{id}    사용자 수정             │
│    DEL  /users/{id}    사용자 삭제             │
│                                               │
│  [Try it out] 버튼으로 실제 요청을 보내고      │
│  응답을 인라인으로 확인할 수 있습니다.          │
└──────────────────────────────────────────────┘
```

### 문서 풍부화

```python
from fastapi import FastAPI, status

app = FastAPI(
    title="User Management API",
    version="1.0.0",
    description="""
    ## 개요
    이 API는 애플리케이션의 사용자를 관리합니다.

    ## 인증
    대부분의 엔드포인트는 Authorization 헤더에 Bearer 토큰이 필요합니다.
    """,
    # 문서 UI에서 태그로 엔드포인트를 그룹화
    openapi_tags=[
        {"name": "users", "description": "사용자 CRUD 연산"},
        {"name": "admin", "description": "관리자 엔드포인트"},
    ]
)

@app.post(
    "/users",
    tags=["users"],  # 문서에서 "users" 아래 그룹화
    summary="새 사용자 생성",  # 엔드포인트 목록에 표시되는 짧은 설명
    description="제공된 정보로 새 사용자 계정을 생성합니다.",
    response_description="새로 생성된 사용자 객체",
    status_code=status.HTTP_201_CREATED,
)
async def create_user(user: UserCreate):
    """summary와 docstring이 모두 제공된 경우,
    summary는 엔드포인트 목록에 사용되고
    docstring은 확장된 상세 뷰에 표시됩니다."""
    ...
```

### OpenAPI 스키마 내보내기

```python
# 프로그래밍 방식으로 스키마 접근
@app.get("/openapi-custom")
async def get_custom_schema():
    """스키마는 JSON이나 YAML로 직렬화 가능한 평범한 dict입니다.
    클라이언트 SDK 생성이나 Postman 가져오기에 유용합니다."""
    return app.openapi()
```

```bash
# 또는 실행 중인 서버에서 직접 가져오기
curl http://localhost:8000/openapi.json | python -m json.tool
```

---

## 7. CORS 미들웨어

**CORS**(Cross-Origin Resource Sharing, 교차 출처 리소스 공유)는 어떤 프론트엔드 도메인이 API를 호출할 수 있는지 제어합니다. CORS 구성 없이는 브라우저가 다른 출처의 요청을 차단합니다.

### 문제

```
프론트엔드: https://myapp.com        백엔드 API: https://api.myapp.com
        │                                      │
        │  fetch("/api/users")                 │
        │ ──────────────────────────────────▶  │
        │                                      │
        │  ✗ 브라우저에 의해 차단됨!            │
        │  "No 'Access-Control-Allow-Origin'"  │
        │ ◀─────────────────────────────────── │
```

### 해결책

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 교차 출처 요청을 허용할 출처 정의
# 개발 환경에서는 모두(*) 허용할 수 있지만, 프로덕션에서는 구체적으로 명시
origins = [
    "http://localhost:3000",     # React 개발 서버
    "http://localhost:5173",     # Vite 개발 서버
    "https://myapp.com",         # 프로덕션 프론트엔드
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # API에 접근할 수 있는 출처
    allow_credentials=True,      # 쿠키/인증 헤더 허용
    allow_methods=["*"],         # 모든 HTTP 메서드 허용
    allow_headers=["*"],         # 모든 헤더 허용
    max_age=600,                 # 프리플라이트 응답을 10분간 캐시
)
```

### CORS 작동 방식 (프리플라이트)

단순하지 않은 요청(예: JSON 본문이 있는 `POST`)의 경우, 브라우저는 먼저 **프리플라이트(preflight)** `OPTIONS` 요청을 보냅니다:

```
브라우저                               서버
  │                                     │
  │  OPTIONS /api/users HTTP/1.1        │  1. 프리플라이트 요청
  │  Origin: https://myapp.com          │
  │  Access-Control-Request-Method: POST│
  │ ──────────────────────────────────▶ │
  │                                     │
  │  HTTP/1.1 204 No Content            │  2. 서버 승인
  │  Access-Control-Allow-Origin: *     │
  │  Access-Control-Allow-Methods: POST │
  │ ◀────────────────────────────────── │
  │                                     │
  │  POST /api/users HTTP/1.1           │  3. 실제 요청
  │  Origin: https://myapp.com          │
  │  Content-Type: application/json     │
  │ ──────────────────────────────────▶ │
  │                                     │
  │  HTTP/1.1 201 Created               │  4. CORS 헤더가 있는 응답
  │  Access-Control-Allow-Origin: *     │
  │ ◀────────────────────────────────── │
```

### 일반적인 CORS 함정

| 문제 | 원인 | 해결책 |
|-------|-------|-----|
| `allow_origins=["*"]`와 자격증명 | 와일드카드 출처와 자격증명은 호환되지 않음 | 특정 출처를 나열 |
| `OPTIONS` 핸들러 누락 | 프레임워크가 프리플라이트를 처리하지 않음 | CORS 미들웨어 사용 (자동으로 처리) |
| `http` vs `https` 불일치 | `http://localhost`는 `https://localhost`가 아님 | 정확한 출처와 일치시킴 |

---

## 8. 연습 문제

### 문제 1: Todo API 구축

다음 엔드포인트를 갖춘 완전한 FastAPI 애플리케이션을 만들어 보세요:
- `POST /todos` -- 새 todo 항목 생성 (title, description, is_completed)
- `GET /todos` -- 모든 todo를 선택적 쿼리 파라미터와 함께 나열: `completed` (bool 필터), `skip`, `limit`
- `GET /todos/{todo_id}` -- 특정 todo 조회
- `PUT /todos/{todo_id}` -- todo 업데이트
- `DELETE /todos/{todo_id}` -- todo 삭제

요구사항:
- 요청/응답에 Pydantic v2 모델 사용
- 적절한 HTTP 상태 코드 사용 (201, 200, 204, 404)
- 인메모리 딕셔너리에 데이터 저장
- 타입 검증 추가 (title은 1-200자)

### 문제 2: Pydantic 모델 설계

전자상거래 제품 카탈로그를 위한 Pydantic v2 모델을 설계하세요:
- `ProductCreate`: name, price (양의 float), category (enum), description (선택), tags (리스트)
- `ProductResponse`: id, created_at, `is_on_sale` 계산 필드 포함
- `ProductUpdate`: 모든 필드를 선택 사항으로 (PATCH 요청용)

적절한 유효성 검사 제약 조건이 있는 `Field()`와 ORM 호환성을 위한 `model_config`를 사용하세요.

### 문제 3: 쿼리 파라미터 검증

다음을 허용하는 `GET /search` 엔드포인트를 만들어 보세요:
- `q`: 필수 검색 쿼리 (최소 2자)
- `category`: 선택적, ["books", "electronics", "clothing"] 중 하나여야 함
- `min_price`와 `max_price`: 선택적 float, min_price는 max_price보다 작아야 함
- `sort_by`: 선택적, 기본값 "relevance", 선택지: ["relevance", "price_asc", "price_desc", "newest"]
- `page`와 `page_size`: 합리적인 기본값과 제한이 있는 페이지네이션

적용된 모든 필터를 보여주는 목업 응답을 반환하세요.

### 문제 4: 오류 응답 표준화

표준화된 오류 응답 형식을 설계하고 커스텀 예외 핸들러를 구현하세요:

```json
{
    "error": {
        "code": "USER_NOT_FOUND",
        "message": "User with ID 42 was not found",
        "details": null,
        "timestamp": "2025-01-15T14:30:00Z"
    }
}
```

최소 세 가지 오류 타입(찾을 수 없음, 유효성 검사 오류, 중복 리소스)에 대해 구현하세요.

### 문제 5: CORS 구성

다음 설정이 있습니다:
- API 서버: `https://api.example.com`
- 웹 앱: `https://app.example.com`
- 모바일 앱: 어떤 출처에서든 요청 가능
- 관리자 패널: `https://admin.example.com` (인증을 위한 쿠키 필요)

다음을 충족하는 CORS 미들웨어 구성을 작성하세요:
1. 세 가지 프론트엔드 모두 허용
2. 관리자 패널의 자격증명 기반 인증 지원
3. API가 사용하는 메서드만으로 제한
4. 커스텀 헤더 허용: `X-API-Key`, `X-Request-Id`

`allow_origins=["*"]`가 여기서 작동하지 않는 이유를 설명하세요.

---

## 9. 참고 자료

- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [Pydantic v2 문서](https://docs.pydantic.dev/latest/)
- [Starlette - ASGI 툴킷](https://www.starlette.io/)
- [Uvicorn - ASGI 서버](https://www.uvicorn.org/)
- [OpenAPI 3.1 명세](https://spec.openapis.org/oas/v3.1.0)
- [MDN CORS 가이드](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS)

---

**이전**: [백엔드 웹 기초](./01_Backend_Web_Fundamentals.md) | **다음**: [FastAPI 고급](./03_FastAPI_Advanced.md)
