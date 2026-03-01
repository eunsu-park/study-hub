# 14. API 설계 패턴

**이전**: [Django 고급](./13_Django_Advanced.md) | **다음**: [인증 패턴](./15_Authentication_Patterns.md)

**난이도**: ⭐⭐⭐

## 학습 목표

- 리소스 지향(resource-oriented) 명명 규칙과 올바른 HTTP 메서드 시맨틱을 사용해 RESTful API를 설계한다
- 확장 가능한 엔드포인트를 위해 페이지네이션(pagination), 필터링(filtering), 정렬(sorting), 필드 선택(field selection)을 구현한다
- API 버전 관리(versioning) 전략을 비교하고, 주어진 프로젝트에 적합한 방식을 선택한다
- RFC 7807 Problem Details를 따르는 일관된 오류 응답(error response)을 구성한다
- 프로덕션 API에 속도 제한(rate limiting) 헤더와 HATEOAS 원칙을 적용한다

## 목차

1. [RESTful 리소스 설계](#1-restful-리소스-설계)
2. [HTTP 메서드와 멱등성](#2-http-메서드와-멱등성)
3. [상태 코드 관례](#3-상태-코드-관례)
4. [페이지네이션 패턴](#4-페이지네이션-패턴)
5. [필터링, 정렬, 필드 선택](#5-필터링-정렬-필드-선택)
6. [API 버전 관리 전략](#6-api-버전-관리-전략)
7. [HATEOAS와 하이퍼미디어](#7-hateoas와-하이퍼미디어)
8. [오류 응답 형식](#8-오류-응답-형식)
9. [속도 제한 헤더](#9-속도-제한-헤더)
10. [연습 문제](#10-연습-문제)

---

## 1. RESTful 리소스 설계

REST API는 리소스를 동사가 아닌 명사로 모델링합니다. URL은 *무엇*을 다루는지 식별하고, HTTP 메서드는 *어떻게* 처리할지를 지정합니다.

**나쁜 예 (동사 지향):**

```
POST /getUsers
POST /createUser
POST /deleteUser/5
```

**좋은 예 (리소스 지향):**

```
GET    /users          # 사용자 목록 조회
POST   /users          # 사용자 생성
GET    /users/5        # 사용자 5 조회
PUT    /users/5        # 사용자 5 교체
PATCH  /users/5        # 사용자 5 부분 업데이트
DELETE /users/5        # 사용자 5 삭제
```

### 명명 규칙

- 컬렉션에는 **복수형 명사** 사용: `/users`, `/posts`, `/comments`
- 관계 표현에는 **중첩 리소스** 사용: `/users/5/posts` (사용자 5의 게시물)
- 중첩은 얕게 유지 (최대 2단계). 그 이상은 쿼리 파라미터나 필터가 있는 최상위 리소스 사용
- 다단어 리소스에는 **kebab-case** 사용: `/blog-posts`, `/order-items`
- URL에 파일 확장자 사용 금지: `/users.json` 대신 `Accept` 헤더 사용

### 서브리소스 vs. 쿼리 파라미터

```
# 서브리소스: 강한 소유 관계
GET /users/5/posts          # 사용자 5에 속한 게시물

# 쿼리 파라미터: 컬렉션 필터링
GET /posts?author_id=5      # 작성자 기준 게시물 필터링
GET /posts?status=published  # 상태 기준 게시물 필터링
```

서브리소스 패턴은 자식이 부모 없이 존재할 수 없음을 의미합니다. 게시물이 사용자 컨텍스트와 독립적으로 존재할 수 있다면 쿼리 파라미터 방식을 선호하세요.

---

## 2. HTTP 메서드와 멱등성

**멱등성(idempotent)**이란 여러 번 호출해도 한 번 호출한 것과 동일한 결과를 생성하는 것을 의미합니다. 이는 재시도 로직과 네트워크 안정성에 중요합니다.

| 메서드  | 용도              | 멱등성 | 안전성 | 요청 본문 |
|--------|------------------|--------|--------|---------|
| GET     | 리소스 조회     | 예     | 예     | 없음     |
| POST    | 리소스 생성     | **아니오** | 아니오 | 있음 |
| PUT     | 리소스 교체     | 예     | 아니오 | 있음     |
| PATCH   | 부분 업데이트   | **아니오*** | 아니오 | 있음 |
| DELETE  | 리소스 삭제     | 예     | 아니오 | 선택적   |
| HEAD    | 헤더만 (본문 없음) | 예   | 예     | 없음     |
| OPTIONS | 지원 메서드 확인 | 예    | 예     | 없음     |

> *PATCH는 패치 문서가 절대값을 지정하는 경우(예: `{"status": "active"}`) 멱등성을 가질 수 있지만, 명세상 멱등성이 보장되지는 않습니다(예: `{"op": "increment", "path": "/count", "value": 1}`).

### PUT vs. PATCH

```python
# PUT: 전체 교체 — 클라이언트가 완전한 리소스를 전송
# 누락된 필드는 null/기본값으로 설정됨
PUT /users/5
{
    "name": "Alice",
    "email": "alice@example.com",
    "bio": "Developer"
}

# PATCH: 부분 업데이트 — 클라이언트가 변경된 필드만 전송
PATCH /users/5
{
    "bio": "Senior Developer"
}
```

---

## 3. 상태 코드 관례

클라이언트가 프로그래밍 방식으로 응답을 처리할 수 있도록 상태 코드를 일관되게 사용하세요.

### 2xx 성공

| 코드 | 의미              | 사용 시점                               |
|------|-----------------|----------------------------------------|
| 200  | OK              | 성공한 GET, PUT, PATCH                  |
| 201  | Created         | 리소스를 생성한 성공적인 POST             |
| 204  | No Content      | 성공한 DELETE (응답 본문 없음)            |

### 4xx 클라이언트 오류

| 코드 | 의미                   | 사용 시점                               |
|------|----------------------|----------------------------------------|
| 400  | Bad Request          | 잘못된 구문, 유효하지 않은 입력           |
| 401  | Unauthorized         | 인증 정보 없음 또는 유효하지 않음         |
| 403  | Forbidden            | 인증됐지만 권한 없음                     |
| 404  | Not Found            | 리소스가 존재하지 않음                   |
| 409  | Conflict             | 중복 리소스, 버전 충돌                   |
| 422  | Unprocessable Entity | 구문은 유효하지만 의미적 오류 (유효성 검사) |
| 429  | Too Many Requests    | 속도 제한 초과                          |

### 5xx 서버 오류

| 코드 | 의미                  | 사용 시점                               |
|------|--------------------|----------------------------------------|
| 500  | Internal Server Error | 예상치 못한 서버 오류                  |
| 502  | Bad Gateway          | 업스트림 서비스가 유효하지 않은 응답 반환 |
| 503  | Service Unavailable  | 서버 과부하 또는 점검 중               |
| 504  | Gateway Timeout      | 업스트림 서비스 타임아웃               |

### FastAPI 예시

```python
from fastapi import FastAPI, HTTPException, status

app = FastAPI()

@app.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(user: UserCreate):
    if await user_exists(user.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User with this email already exists",
        )
    return await save_user(user)

@app.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(user_id: int):
    user = await get_user(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    await remove_user(user_id)
```

---

## 4. 페이지네이션 패턴

목록을 반환하는 모든 엔드포인트는 페이지네이션을 지원해야 합니다. 무제한 쿼리는 서버와 클라이언트 모두의 성능을 저하시킬 수 있습니다.

### 오프셋 기반 페이지네이션(Offset-Based Pagination)

가장 단순한 방식입니다. 클라이언트가 `offset`(또는 `page`)과 `limit`을 지정합니다.

```
GET /posts?offset=20&limit=10
```

```json
{
    "data": [...],
    "pagination": {
        "offset": 20,
        "limit": 10,
        "total": 153
    }
}
```

**장점:** 구현이 쉽고, 임의 페이지로 이동 가능.
**단점:** 큰 오프셋에서 비용이 큼 (데이터베이스가 행을 스캔하고 버려야 함), 페이지 간 데이터 변경 시 결과가 불일치할 수 있음.

### 커서 기반 페이지네이션(Cursor-Based Pagination)

서버가 불투명한 커서(opaque cursor)(일반적으로 base64로 인코딩된 식별자)를 반환하고, 클라이언트가 다음 페이지를 가져오기 위해 이를 전달합니다.

```
GET /posts?limit=10&cursor=eyJpZCI6IDIwfQ==
```

```json
{
    "data": [...],
    "pagination": {
        "next_cursor": "eyJpZCI6IDMwfQ==",
        "has_more": true
    }
}
```

**장점:** 데이터 변경 중에도 일관된 결과, 큰 데이터셋에서 효율적.
**단점:** 임의 페이지로 이동 불가, 커서를 불투명하게 취급해야 함.

### 키셋 페이지네이션(Keyset Pagination)

커서 기반과 유사하지만 불투명한 토큰 대신 명시적인 컬럼 값을 사용합니다.

```
GET /posts?limit=10&created_after=2025-01-15T10:30:00Z&id_after=150
```

**장점:** 투명하고 효율적(인덱스된 컬럼 사용), 스캔 없음.
**단점:** 고유하고 순차적인 정렬 키가 필요하며, 임의 페이지로 이동 불가.

### 비교

| 측면             | 오프셋       | 커서         | 키셋         |
|-----------------|------------|-------------|------------|
| N 페이지로 이동  | 예          | 아니오       | 아니오       |
| 일관된 페이지   | 아니오      | 예           | 예           |
| 대용량 데이터 성능 | 낮음       | 좋음         | 좋음         |
| 구현 복잡도     | 단순        | 보통         | 보통         |
| 적합한 사용처   | 관리자 패널  | 소셜 피드    | 시계열 데이터 |

---

## 5. 필터링, 정렬, 필드 선택

### 필터링

리소스 필드 이름의 쿼리 파라미터를 사용합니다:

```
GET /posts?status=published&author_id=5
GET /posts?created_after=2025-01-01&created_before=2025-06-01
GET /posts?tags=python,fastapi     # IN 쿼리를 위한 쉼표 구분
```

복잡한 필터링에는 구조화된 구문을 고려하세요:

```
GET /posts?filter[status]=published&filter[rating][gte]=4
```

### 정렬

`sort` 파라미터와 필드 이름을 사용합니다. 내림차순은 `-` 접두사를 붙입니다:

```
GET /posts?sort=-created_at          # 최신순
GET /posts?sort=author,-created_at   # 작성자 오름차순, 그 다음 최신순
```

### 필드 선택(Sparse Fieldsets)

클라이언트가 필요한 필드만 요청할 수 있도록 허용하여 페이로드(payload) 크기를 줄입니다:

```
GET /posts?fields=id,title,created_at
GET /users/5?fields=name,email
```

### FastAPI 구현

```python
from fastapi import FastAPI, Query
from typing import Optional

app = FastAPI()

@app.get("/posts")
async def list_posts(
    status: Optional[str] = None,
    author_id: Optional[int] = None,
    sort: str = Query(default="-created_at"),
    fields: Optional[str] = None,
    limit: int = Query(default=20, le=100),
    offset: int = Query(default=0, ge=0),
):
    query = select(Post)

    # 필터 적용
    if status:
        query = query.where(Post.status == status)
    if author_id:
        query = query.where(Post.author_id == author_id)

    # 정렬 적용
    for field in sort.split(","):
        if field.startswith("-"):
            query = query.order_by(getattr(Post, field[1:]).desc())
        else:
            query = query.order_by(getattr(Post, field).asc())

    # 페이지네이션 적용
    query = query.offset(offset).limit(limit)

    results = await db.execute(query)
    posts = results.scalars().all()

    # 필드 선택 적용
    if fields:
        field_list = fields.split(",")
        posts = [
            {k: v for k, v in post.dict().items() if k in field_list}
            for post in posts
        ]

    return {"data": posts, "pagination": {"offset": offset, "limit": limit}}
```

---

## 6. API 버전 관리 전략

API는 발전합니다. 버전 관리는 기존 클라이언트가 계속 작동하는 동안 새 클라이언트가 업데이트된 엔드포인트를 사용할 수 있도록 보장합니다.

| 전략              | 예시                                          | 장점                       | 단점                               |
|-----------------|----------------------------------------------|--------------------------|----------------------------------|
| URL 경로         | `/v1/users`, `/v2/users`                    | 명시적, 라우팅 용이          | URL 오염, 지원 종료 어려움         |
| 쿼리 파라미터    | `/users?version=2`                          | 선택적, 하위 호환           | 누락 가능성, 캐싱 문제             |
| 헤더             | `Accept: application/vnd.api+json;version=2` | 깔끔한 URL                | 숨겨짐, 브라우저 테스트 어려움     |
| 콘텐츠 협상      | `Accept: application/vnd.myapp.v2+json`     | RESTful, 정밀              | 복잡함, 많은 개발자에게 낯섦       |

### 권장 사항

**URL 경로 버전 관리**가 대부분의 팀에 가장 일반적이고 실용적인 선택입니다. 명시적이고 이해하기 쉬우며, 프레임워크 라우터로 구현하기 간단합니다.

```python
# FastAPI: 라우터를 사용한 URL 경로 버전 관리
from fastapi import APIRouter

v1_router = APIRouter(prefix="/v1")
v2_router = APIRouter(prefix="/v2")

@v1_router.get("/users")
async def list_users_v1():
    """이메일 없이 사용자 목록 반환 (레거시)."""
    return [{"id": u.id, "name": u.name} for u in users]

@v2_router.get("/users")
async def list_users_v2():
    """이메일과 아바타를 포함한 사용자 목록 반환."""
    return [
        {"id": u.id, "name": u.name, "email": u.email, "avatar_url": u.avatar}
        for u in users
    ]

app.include_router(v1_router)
app.include_router(v2_router)
```

---

## 7. HATEOAS와 하이퍼미디어

**HATEOAS**(Hypermedia as the Engine of Application State)는 API 응답에 클라이언트가 다음에 수행할 수 있는 작업을 알려주는 링크를 포함하는 것을 의미합니다. 클라이언트는 URL 패턴을 하드코딩하는 대신 이 링크를 통해 API를 탐색합니다.

```json
{
    "id": 42,
    "title": "API Design Patterns",
    "status": "draft",
    "_links": {
        "self": {"href": "/posts/42"},
        "author": {"href": "/users/5"},
        "publish": {"href": "/posts/42/publish", "method": "POST"},
        "comments": {"href": "/posts/42/comments"}
    }
}
```

### HATEOAS를 활용한 페이지네이션

```json
{
    "data": [...],
    "_links": {
        "self": {"href": "/posts?page=3&limit=10"},
        "first": {"href": "/posts?page=1&limit=10"},
        "prev": {"href": "/posts?page=2&limit=10"},
        "next": {"href": "/posts?page=4&limit=10"},
        "last": {"href": "/posts?page=15&limit=10"}
    },
    "_meta": {
        "total": 150,
        "page": 3,
        "limit": 10
    }
}
```

실제로 완전한 HATEOAS는 거의 구현되지 않습니다. 대부분의 프로덕션 API는 실용적인 부분 집합을 채택합니다: 페이지네이션 링크와 리소스별 `self` 링크.

---

## 8. 오류 응답 형식

일관된 오류 응답은 개발자 경험에 매우 중요합니다. **RFC 7807**(HTTP API를 위한 Problem Details)은 표준 형식을 정의합니다.

### RFC 7807 구조

```json
{
    "type": "https://api.example.com/errors/validation",
    "title": "Validation Error",
    "status": 422,
    "detail": "The 'email' field must be a valid email address.",
    "instance": "/users",
    "errors": [
        {
            "field": "email",
            "message": "Not a valid email address",
            "value": "not-an-email"
        }
    ]
}
```

| 필드       | 필수 여부 | 설명                                     |
|-----------|---------|----------------------------------------|
| `type`    | 예      | 오류 유형을 식별하는 URI                    |
| `title`   | 예      | 간단한 사람이 읽을 수 있는 요약              |
| `status`  | 예      | HTTP 상태 코드                           |
| `detail`  | 아니오   | 이번 발생에 대한 사람이 읽을 수 있는 설명     |
| `instance`| 아니오   | 오류를 유발한 특정 요청의 URI              |

### FastAPI 예외 핸들러

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

app = FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })

    return JSONResponse(
        status_code=422,
        content={
            "type": "https://api.example.com/errors/validation",
            "title": "Validation Error",
            "status": 422,
            "detail": f"{len(errors)} validation error(s) in request",
            "instance": str(request.url),
            "errors": errors,
        },
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "type": f"https://api.example.com/errors/{exc.status_code}",
            "title": exc.detail,
            "status": exc.status_code,
            "instance": str(request.url),
        },
    )
```

---

## 9. 속도 제한 헤더

속도 제한(rate limiting)은 API를 남용으로부터 보호하고 공정한 사용을 보장합니다. 표준 헤더는 클라이언트에게 제한을 알려줍니다.

### 표준 헤더

```
HTTP/1.1 200 OK
X-RateLimit-Limit: 100         # 윈도우당 최대 요청 수
X-RateLimit-Remaining: 42      # 윈도우 내 남은 요청 수
X-RateLimit-Reset: 1706140800  # 윈도우 초기화 시각 (Unix 타임스탬프)
Retry-After: 30                # 대기 시간 (초, 429 응답 시)
```

### slowapi를 사용한 FastAPI

```python
from fastapi import FastAPI
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter

@app.get("/posts")
@limiter.limit("100/minute")
async def list_posts(request: Request):
    return await fetch_posts()

@app.get("/search")
@limiter.limit("20/minute")
async def search(request: Request, q: str):
    """검색은 비용이 더 크므로 더 낮은 제한을 가집니다."""
    return await perform_search(q)
```

### 속도 제한 전략

| 전략             | 설명                             | 사용 사례            |
|----------------|--------------------------------|--------------------|
| 고정 윈도우(Fixed window) | N분마다 카운터 초기화        | 단순, 범용          |
| 슬라이딩 윈도우(Sliding window) | 요청 시간 기반 롤링 윈도우  | 더 부드러운 속도 제한 |
| 토큰 버킷(Token bucket) | 일정 속도로 토큰 보충         | 짧은 버스트 허용     |
| 리키 버킷(Leaky bucket) | 일정 속도로 요청 처리         | 엄격한 속도 시행     |

---

## 10. 연습 문제

### 문제 1: 리소스 모델링

다음 엔티티를 포함하는 대학 강좌 관리 시스템의 URL 구조를 설계하세요: 학과(department), 강좌(course), 학기(semester), 수강 신청(enrollment), 성적(grade). 모든 엔드포인트(메서드 + URL)를 나열하고 각각이 반환해야 하는 상태 코드를 명시하세요.

### 문제 2: 페이지네이션 구현

FastAPI와 SQLAlchemy를 사용해 `/comments` 엔드포인트에 커서 기반 페이지네이션을 구현하세요. 커서는 댓글의 `created_at`과 `id` 필드를 인코딩해야 합니다. `next_cursor`와 `has_more`를 포함한 적절한 응답 형식을 구성하세요.

### 문제 3: 오류 핸들러

다음을 충족하는 FastAPI 애플리케이션을 위한 포괄적인 오류 처리 시스템을 만드세요:
- RFC 7807을 준수하는 오류 응답 반환
- 유효성 검사 오류, 미발견 오류, 권한 오류 처리
- 비즈니스 로직 오류를 위한 커스텀 예외 클래스 포함 (예: "InsufficientBalance")
- 적절한 심각도 수준으로 오류 로깅

### 문제 4: 버전 마이그레이션

기존 `/v1/products` 엔드포인트가 `{"name": "Widget", "price": 9.99}`를 반환합니다. v2에서는 `price`를 `price.amount`와 `price.currency`로 분리해야 합니다. v2 응답 형식과 v1 클라이언트가 계속 작동할 수 있는 마이그레이션 전략을 설계하세요. 공유 비즈니스 로직으로 FastAPI에서 해결책을 구현하세요.

### 문제 5: 속도 제한기 설계

JWT를 통해 인증된 사용자별 제한을 지원하며 다음 티어(tier)를 가진 속도 제한 미들웨어를 설계하세요:
- 무료 티어: 분당 60회 요청
- Pro 티어: 분당 600회 요청
- Enterprise 티어: 분당 6,000회 요청

Redis 기반 슬라이딩 윈도우(sliding window) 알고리즘을 사용해 구현하세요. 표준 속도 제한 응답 헤더를 포함하세요.

---

## 참고 자료

- Fielding, R. (2000). *Architectural Styles and the Design of Network-based Software Architectures* (박사 학위 논문). Chapter 5: REST.
- [RFC 7807: HTTP API를 위한 Problem Details](https://tools.ietf.org/html/rfc7807)
- [RFC 6585: 추가 HTTP 상태 코드 (429)](https://tools.ietf.org/html/rfc6585)
- [Microsoft REST API 가이드라인](https://github.com/microsoft/api-guidelines)
- [JSON:API 명세](https://jsonapi.org/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)

---

**이전**: [Django 고급](./13_Django_Advanced.md) | **다음**: [인증 패턴](./15_Authentication_Patterns.md)
