# 01. 백엔드 웹 기초(Backend Web Fundamentals)

**이전**: [개요](./00_Overview.md) | **다음**: [FastAPI 기초](./02_FastAPI_Basics.md)

**난이도**: ⭐⭐

---

## 학습 목표

- HTTP 요청/응답 사이클을 메서드, 헤더, 상태 코드, 본문 형식을 포함하여 설명할 수 있다
- REST 원칙을 적용하여 적절한 CRUD 매핑이 포함된 리소스 중심 API를 설계할 수 있다
- WSGI와 ASGI 서버 모델을 비교하고 각각이 적합한 상황을 파악할 수 있다
- 변화하는 서비스에 대한 일관성 있는 API 버전 관리 전략을 설계할 수 있다
- 라우팅부터 응답까지 일반적인 웹 프레임워크에서 요청 생명주기를 추적할 수 있다

---

## 목차

1. [HTTP 요청/응답 사이클](#1-http-요청응답-사이클)
2. [REST 원칙](#2-rest-원칙)
3. [WSGI vs ASGI](#3-wsgi-vs-asgi)
4. [API의 공용어로서의 JSON](#4-api의-공용어로서의-json)
5. [API 버전 관리 전략](#5-api-버전-관리-전략)
6. [웹 프레임워크의 요청 생명주기](#6-웹-프레임워크의-요청-생명주기)
7. [연습 문제](#7-연습-문제)
8. [참고 자료](#8-참고-자료)

---

## 1. HTTP 요청/응답 사이클

클라이언트(브라우저, 모바일 앱, CLI 도구)와 백엔드 서버 간의 모든 상호작용은 **HTTP 요청/응답** 모델을 따릅니다. 클라이언트가 요청을 보내면 서버가 이를 처리하고 응답을 반환합니다.

### HTTP 요청의 구조

```
POST /api/users HTTP/1.1          <-- 요청 라인: 메서드 경로 버전
Host: api.example.com             <-- 헤더 시작
Content-Type: application/json
Authorization: Bearer eyJhbG...
Content-Length: 56
                                  <-- 빈 줄이 헤더와 본문을 구분
{"name": "Alice", "email": "alice@example.com"}   <-- 본문 (선택 사항)
```

### HTTP 메서드

| 메서드 | 목적 | 멱등성(Idempotent) | 안전성(Safe) | 본문 여부 |
|--------|---------|-----------|------|----------|
| `GET` | 리소스 조회 | 예 | 예 | 아니오 |
| `POST` | 새 리소스 생성 | 아니오 | 아니오 | 예 |
| `PUT` | 리소스 전체 교체 | 예 | 아니오 | 예 |
| `PATCH` | 리소스 부분 업데이트 | 아니오 | 아니오 | 예 |
| `DELETE` | 리소스 삭제 | 예 | 아니오 | 선택 사항 |
| `HEAD` | GET과 동일하지만 본문 없음 | 예 | 예 | 아니오 |
| `OPTIONS` | 허용된 메서드 조회 (CORS) | 예 | 예 | 아니오 |

**멱등성(Idempotent)**이란 동일한 요청을 여러 번 호출해도 결과가 같다는 의미입니다. 같은 본문으로 `PUT /users/42`를 반복 호출하면 항상 동일한 상태가 됩니다. 반면 `POST /users`는 매번 중복 데이터를 생성할 수 있습니다.

### HTTP 응답의 구조

```
HTTP/1.1 201 Created             <-- 상태 라인: 버전 코드 사유
Content-Type: application/json
Location: /api/users/42          <-- 새 리소스의 위치
X-Request-Id: abc-123
                                 <-- 빈 줄
{"id": 42, "name": "Alice", "email": "alice@example.com"}
```

### 상태 코드 패밀리

| 범위 | 분류 | 주요 코드 |
|-------|----------|-------------|
| `1xx` | 정보 | `101 Switching Protocols` (WebSocket 업그레이드) |
| `2xx` | 성공 | `200 OK`, `201 Created`, `204 No Content` |
| `3xx` | 리다이렉션 | `301 Moved Permanently`, `304 Not Modified` |
| `4xx` | 클라이언트 오류 | `400 Bad Request`, `401 Unauthorized`, `403 Forbidden`, `404 Not Found`, `422 Unprocessable Entity`, `429 Too Many Requests` |
| `5xx` | 서버 오류 | `500 Internal Server Error`, `502 Bad Gateway`, `503 Service Unavailable` |

유용한 규칙: 클라이언트가 잘못된 요청을 보낸 경우 `4xx`를 반환하고, 서버에서 실패한 경우 `5xx`를 반환합니다. 오류 본문과 함께 `200`을 반환하는 것은 클라이언트와 모니터링 도구를 혼란스럽게 만드므로 절대 해서는 안 됩니다.

### 요청 흐름 다이어그램

```
  클라이언트                          서버
    |                                |
    |  ---- HTTP 요청 -----------> |
    |       메서드 + URL             |
    |       헤더                    |
    |       본문 (선택 사항)          |
    |                                |
    |                          [ 처리   ]
    |                          [ 라우팅  ]
    |                          [ 로직   ]
    |                          [ DB 호출 ]
    |                                |
    |  <--- HTTP 응답 ------------ |
    |       상태 코드               |
    |       헤더                    |
    |       본문 (선택 사항)          |
    |                                |
```

---

## 2. REST 원칙

**REST**(Representational State Transfer, 표현 상태 전달)는 프로토콜이 아닌 아키텍처 스타일입니다. 예측 가능하고 확장 가능하며 사용하기 쉬운 웹 API를 설계하기 위한 지침을 제공합니다.

### 핵심 원칙

1. **행위가 아닌 리소스**: URL은 동사(`/getUser?id=42`)가 아닌 명사(`/users/42`)를 식별합니다
2. **무상태성(Statelessness)**: 각 요청은 처리에 필요한 모든 정보를 담고 있습니다. 서버는 요청 간에 클라이언트 세션 상태를 저장하지 않습니다.
3. **균일한 인터페이스(Uniform interface)**: 모든 리소스에 걸쳐 표준 HTTP 메서드를 일관성 있게 사용합니다.
4. **HATEOAS**(Hypermedia As The Engine Of Application State): 응답에 관련 리소스의 링크가 포함됩니다. 실제로는 많은 API가 이를 생략합니다.

### CRUD 매핑

| 연산 | HTTP 메서드 | URL 패턴 | 응답 코드 |
|-----------|------------|-------------|---------------|
| 생성(Create) | `POST` | `/api/users` | `201 Created` |
| 조회(Read, 목록) | `GET` | `/api/users` | `200 OK` |
| 조회(Read, 상세) | `GET` | `/api/users/42` | `200 OK` |
| 수정(Update, 전체) | `PUT` | `/api/users/42` | `200 OK` |
| 수정(Update, 부분) | `PATCH` | `/api/users/42` | `200 OK` |
| 삭제(Delete) | `DELETE` | `/api/users/42` | `204 No Content` |

### 리소스 명명 규칙

```
# 좋음 - 복수 명사, 계층적 관계
GET  /api/users/42/orders          # 사용자 42에 속한 주문 목록
GET  /api/users/42/orders/7        # 특정 주문
POST /api/users/42/orders          # 사용자 42의 주문 생성

# 나쁨 - URL에 동사 포함, 평면 구조
GET  /api/getUserOrders?userId=42
POST /api/createOrder
```

### 필터링, 정렬, 페이지네이션(Pagination)

```
# 쿼리 파라미터를 사용한 필터링
GET /api/users?role=admin&status=active

# 정렬 (내림차순은 - 접두사 사용)
GET /api/users?sort=-created_at,name

# 페이지네이션 (오프셋 기반)
GET /api/users?page=2&per_page=25

# 페이지네이션 (커서 기반 -- 대용량 데이터셋에 적합)
GET /api/users?cursor=eyJpZCI6NDJ9&limit=25
```

---

## 3. WSGI vs ASGI

Python 웹 서버는 웹 서버와 애플리케이션 사이에 표준 인터페이스가 필요합니다. 두 가지 표준이 존재합니다: **WSGI**(동기) 와 **ASGI**(비동기).

### WSGI (Web Server Gateway Interface)

WSGI는 PEP 3333 (2003)에서 정의되었습니다. 워커 프로세스당 한 번에 하나의 요청을 처리합니다.

```python
# 최소한의 WSGI 애플리케이션
# 각 호출은 응답이 준비될 때까지 블로킹됩니다
def application(environ: dict, start_response):
    """WSGI는 요청 환경(environ)과 응답 시작을 위한 콜백을
    받는 callable을 기대합니다."""
    status = "200 OK"
    headers = [("Content-Type", "text/plain")]
    start_response(status, headers)
    return [b"Hello, WSGI!"]
```

**WSGI 서버**: Gunicorn, uWSGI, mod_wsgi
**WSGI 프레임워크**: Flask, Django (전통 모드)

### ASGI (Asynchronous Server Gateway Interface)

ASGI는 async/await, WebSocket, 장기 연결을 지원하기 위해 도입되었습니다. 워커 하나가 많은 동시 연결을 처리할 수 있습니다.

```python
# 최소한의 ASGI 애플리케이션
# async/await 사용 -- 수천 개의 동시 연결 처리 가능
async def application(scope: dict, receive, send):
    """ASGI는 세 가지 인자를 받는 callable을 사용합니다:
    scope = 연결 메타데이터, receive = 수신 메시지,
    send = 송신 메시지."""
    if scope["type"] == "http":
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"text/plain")],
        })
        await send({
            "type": "http.response.body",
            "body": b"Hello, ASGI!",
        })
```

**ASGI 서버**: Uvicorn, Hypercorn, Daphne
**ASGI 프레임워크**: FastAPI, Starlette, Django (ASGI 모드)

### 비교

```
WSGI (동기)                           ASGI (비동기)
┌──────────────────┐                  ┌──────────────────┐
│  워커 프로세스    │                  │  이벤트 루프       │
│                   │                  │                   │
│  요청 1 ████      │ (블로킹)          │  요청 1 ██  ██    │ (논블로킹)
│  요청 2   ████    │ (대기)           │  요청 2  ██  ██   │ (인터리빙)
│  요청 3     ████  │                  │  요청 3 ██  ██    │
└──────────────────┘                  │  요청 4  ██  ██   │
                                      └──────────────────┘
N개의 동시 요청을 위해 N개의           워커 하나가 많은 연결을
워커가 필요. CPU 집약적 작업에 적합.   처리. I/O 집약적 작업에 적합.
```

### 각각의 사용 시기

| 시나리오 | 권장 사항 |
|----------|---------------|
| 단순 CRUD API | 둘 다 가능; ASGI가 미래 지향적 |
| 고동시성 I/O (DB, 외부 API) | ASGI |
| WebSocket, SSE, 롱 폴링 | ASGI (WSGI는 이를 지원할 수 없음) |
| CPU 집약적 연산 | 다중 워커 WSGI, 또는 스레드 풀을 사용하는 ASGI |
| 레거시 애플리케이션 | Flask/Django로 이미 구축된 경우 WSGI |

---

## 4. API의 공용어로서의 JSON

**JSON**(JavaScript Object Notation)은 웹 API의 지배적인 데이터 형식입니다. 사람이 읽을 수 있고, 언어 독립적이며, JavaScript에서 기본으로 지원됩니다.

### JSON 데이터 타입

```json
{
    "string": "hello",
    "integer": 42,
    "float": 3.14,
    "boolean": true,
    "null_value": null,
    "array": [1, 2, 3],
    "nested_object": {
        "key": "value"
    }
}
```

### Python JSON 처리

```python
import json
from datetime import datetime, date
from decimal import Decimal

# Python의 json 모듈은 기본 타입을 처리하지만, datetime과 Decimal은
# JSON 기본형이 아니므로 커스텀 직렬화가 필요합니다
class APIEncoder(json.JSONEncoder):
    """json.dumps()가 기본으로 처리하지 못하는 타입을 위한 커스텀 인코더."""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()  # ISO 8601이 API 날짜의 표준
        if isinstance(obj, Decimal):
            return float(obj)  # JSON 호환성을 위해 정밀도를 타협
        return super().default(obj)

data = {"created_at": datetime.now(), "price": Decimal("19.99")}
json_string = json.dumps(data, cls=APIEncoder, indent=2)
print(json_string)
# {"created_at": "2025-01-15T14:30:00", "price": 19.99}
```

### 콘텐츠 협상(Content Negotiation)

클라이언트는 `Accept` 헤더를 통해 서버에 원하는 형식을 알립니다. 서버는 선택한 형식을 `Content-Type`으로 응답합니다:

```
# 클라이언트가 JSON 요청
GET /api/users/42 HTTP/1.1
Accept: application/json

# 서버가 JSON으로 응답
HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
```

### JSON의 대안

| 형식 | 장점 | 단점 | 사용 사례 |
|--------|------|------|----------|
| JSON | 범용, 사람이 읽을 수 있음 | 장황함, 스키마 없음 | 일반 API |
| MessagePack | 압축된 바이너리 JSON | 사람이 읽을 수 없음 | 고처리량 내부 통신 |
| Protocol Buffers | 강타입, 빠름 | `.proto` 파일 필요 | gRPC 마이크로서비스 |
| XML | 자기 서술적, 스키마 | 매우 장황함 | 레거시 SOAP, 설정 파일 |

---

## 5. API 버전 관리 전략

API는 진화합니다. 파괴적인 변경(Breaking changes)은 불가피합니다. 버전 관리를 통해 기존 클라이언트를 손상시키지 않고 변경 사항을 도입할 수 있습니다.

### 전략 1: URL 경로 버전 관리

```
GET /api/v1/users/42
GET /api/v2/users/42
```

**장점**: 명시적, 이해하기 쉬움, 라우팅 용이
**단점**: URL 오염, 폐기(deprecation) 어려움
**사용 예**: GitHub, Stripe, Twitter

### 전략 2: 헤더 버전 관리

```
GET /api/users/42
Accept: application/vnd.myapi.v2+json
```

**장점**: 깔끔한 URL, HTTP 시맨틱스 준수
**단점**: 검색 어려움, 브라우저에서 테스트 불편
**사용 예**: GitHub (이 방식도 지원)

### 전략 3: 쿼리 파라미터 버전 관리

```
GET /api/users/42?version=2
```

**장점**: 추가 용이, 선택적 (기본값은 최신 버전)
**단점**: 잊기 쉬움, 쿼리 문자열 오염
**사용 예**: Google APIs, Amazon

### 실용적인 권장 사항

대부분의 프로젝트에서 **URL 경로 버전 관리** (`/api/v1/`)가 최선의 기본 선택입니다. 명시적이고 단순하며 모든 HTTP 클라이언트와 캐싱 레이어에서 작동합니다. 파괴적인 변경이 있을 때만 새 버전을 도입하세요.

```python
# FastAPI 예제: 라우터를 사용한 버전 구성
from fastapi import APIRouter, FastAPI

app = FastAPI()

# 각 버전은 독립적인 로직을 가진 자체 라우터를 가집니다
v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    return {"id": user_id, "name": "Alice"}  # v1: 플랫 응답

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    return {"data": {"id": user_id, "name": "Alice"}, "meta": {"version": 2}}

app.include_router(v1_router)
app.include_router(v2_router)
```

---

## 6. 웹 프레임워크의 요청 생명주기

요청이 도착하면 비즈니스 로직에 도달하기 전에 여러 컴포넌트로 구성된 파이프라인을 통과합니다.

### 파이프라인

```
클라이언트 요청
     │
     ▼
┌─────────────────┐
│  웹 서버         │  Uvicorn / Gunicorn이 원시 HTTP 수신
│  (ASGI/WSGI)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  미들웨어         │  핸들러 전에 실행
│  스택            │  - CORS 헤더
│                   │  - 인증 검사
│                   │  - 요청 로깅
│                   │  - 속도 제한
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  라우터           │  URL 패턴을 핸들러 함수에 매핑
│                   │  /api/users/{id} → get_user(id)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  의존성 주입      │  핸들러가 선언한 의존성 해결
│  (Dependency     │  - 데이터베이스 세션
│   Injection)     │  - 현재 인증된 사용자
│                   │  - 설정 값
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  요청 검증        │  파싱 및 검증:
│  (Validation)    │  - 경로 파라미터 (타입 변환)
│                   │  - 쿼리 파라미터
│                   │  - 요청 본문 (JSON → Pydantic 모델)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  핸들러           │  비즈니스 로직이 실행되는 곳
│  (뷰/엔드포인트)  │  - 데이터베이스 쿼리
│                   │  - 데이터 처리
│                   │  - 응답 객체 반환
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  응답 직렬화      │  반환값을 JSON으로 직렬화
│  (Serialization) │  응답 모델 필터링 적용
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  미들웨어         │  핸들러 후에 실행
│  (후처리)        │  - 응답 헤더 추가
│                   │  - 응답 본문 압축
│                   │  - 응답 상태 + 타이밍 로그
└────────┬────────┘
         │
         ▼
    HTTP 응답
```

### 미들웨어 예제

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """각 요청이 얼마나 걸리는지 측정하여 헤더로 추가합니다.
    모든 엔드포인트를 수정하지 않고 성능 모니터링에 유용합니다."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    # X- 접두사는 커스텀 헤더의 관례입니다
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response
```

### 파이프라인의 오류 처리

어느 단계에서든 예외가 발생하면 프레임워크가 이를 잡아 HTTP 오류 응답으로 변환합니다:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    user = await find_user(user_id)
    if user is None:
        # HTTPException은 파이프라인을 단락시키고
        # 추가 처리 없이 즉시 오류 응답을 반환합니다
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found"
        )
    return user
```

---

## 7. 연습 문제

### 문제 1: REST API 설계

**블로그** 애플리케이션을 위한 URL 구조와 HTTP 메서드를 설계하세요. 다음 리소스를 포함해야 합니다: 게시물(posts), 댓글(comments), 태그(tags). 다음 사항을 포함하세요:
- 게시물에 대한 CRUD 연산
- 게시물 아래 중첩된 댓글
- 게시물에 태그 할당
- 태그로 게시물 필터링
- 게시물 목록 페이지네이션

각 엔드포인트를 `메서드 /경로 -> 상태코드` 형식으로 작성하세요.

### 문제 2: 상태 코드 선택

각 시나리오에 대해 가장 적절한 HTTP 상태 코드를 선택하고 이유를 설명하세요:

1. 사용자가 잘못된 이메일 형식으로 회원가입 폼을 제출한 경우
2. 서버의 데이터베이스 연결 풀이 고갈된 경우
3. 사용자가 `DELETE /api/users/99`를 요청했지만 사용자 99가 존재하지 않는 경우
4. 사용자가 리소스를 요청했지만 JWT 토큰이 만료된 경우
5. `POST` 요청은 성공했지만 생성된 리소스는 나중에 사용 가능할 경우 (비동기 처리)

### 문제 3: WSGI vs ASGI 분석

다음 기능을 가진 채팅 애플리케이션을 구축하고 있다고 가정합니다:
- 사용자 프로필과 채팅 기록을 위한 REST API 제공
- 실시간 메시징을 위한 WebSocket 연결 유지
- 다국어 지원을 위한 외부 번역 API 호출

어떤 서버 모델(WSGI 또는 ASGI)을 선택하겠습니까? 위의 세 가지 요구사항 각각을 다루어 선택 이유를 정당화하세요.

### 문제 4: 미들웨어 파이프라인

다음을 수행하는 Python 함수(의사 미들웨어)를 작성하세요:
1. `X-API-Key` 헤더 확인
2. 키가 없으면 `401 Unauthorized` 반환
3. 키가 유효하지 않으면 (미리 정의된 집합에 없으면) `403 Forbidden` 반환
4. 요청 메서드, 경로, 응답 상태 코드 로깅
5. 모든 응답에 `X-Request-Id` 헤더 (UUID) 추가

### 문제 5: JSON 직렬화 엣지 케이스

다음 Python 데이터가 주어졌을 때, 모든 타입을 올바르게 처리하는 커스텀 JSON 인코더를 작성하세요:
```python
data = {
    "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
    "amount": Decimal("99.95"),
    "created_at": datetime(2025, 6, 15, 10, 30),
    "tags": frozenset({"python", "api"}),
    "metadata": None
}
```

---

## 8. 참고 자료

- [MDN HTTP 레퍼런스](https://developer.mozilla.org/en-US/docs/Web/HTTP)
- [RFC 7231 - HTTP/1.1 시맨틱스와 콘텐츠](https://datatracker.ietf.org/doc/html/rfc7231)
- [Roy Fielding의 REST 논문](https://www.ics.uci.edu/~fielding/pubs/dissertation/rest_arch_style.htm)
- [PEP 3333 - Python 웹 서버 게이트웨이 인터페이스](https://peps.python.org/pep-3333/)
- [ASGI 명세](https://asgi.readthedocs.io/en/latest/specs/main.html)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
- [JSON 명세 (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)

---

**이전**: [개요](./00_Overview.md) | **다음**: [FastAPI 기초](./02_FastAPI_Basics.md)
