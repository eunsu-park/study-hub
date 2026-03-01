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

## 1. FastAPI란 무엇인가

FastAPI는 세 가지 기반 위에 구축된 현대적인 Python 웹 프레임워크입니다:

1. **타입 힌트(Type hints)** (Python 3.10+): 파라미터에 타입이 지정되어 자동 유효성 검사와 문서화가 가능합니다
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
