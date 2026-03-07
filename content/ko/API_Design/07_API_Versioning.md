# Lesson 7: API Versioning

**이전**: [인증과 인가](06_Authentication_and_Authorization.md) | [개요](00_Overview.md) | **다음**: [오류 처리](08_Error_Handling.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. URL, 헤더, 쿼리 파라미터 버전 관리 전략을 장단점과 함께 비교하기
2. 추가적 변경(additive changes)을 통해 하위 호환성을 유지하는 API 설계하기
3. Sunset 헤더와 마이그레이션 가이드를 활용한 지원 중단(deprecation) 전략 구현하기
4. 베타부터 서비스 종료(end-of-life)까지의 버전 생명주기 정의하기
5. 프로덕션 API에서 호환성 깨짐(breaking changes)을 방지하는 실용적 기법 적용하기
6. API의 대상과 규모에 맞는 올바른 버전 관리 전략 선택하기

---

버전 관리는 기존 소비자를 깨뜨리지 않으면서 API를 발전시키는 기술입니다. 모든 API는 변경됩니다 -- 새 필드, 엔드포인트 이름 변경, 응답 구조 변경. 문제는 버전 관리가 필요한지 여부가 아니라, 얼마나 우아하게 처리할 것인가입니다. 좋은 버전 관리 전략은 소비자와의 계약을 준수하면서도 자유롭게 혁신할 수 있도록 해줍니다.

> **비유:** API 버전 관리는 레스토랑 메뉴와 같습니다. 단골손님을 혼란스럽게 하지 않고 새 메뉴를 추가할 수 있습니다(추가적 변경). 그러나 "스파게티 카르보나라"를 "파스타 #7"로 이름을 바꾸고 레시피도 변경하면 단골손님은 불만을 가질 것입니다. 버전이 붙은 메뉴("2025 봄 메뉴")는 더 큰 변경을 하면서도 고객에게 적응할 시간을 제공합니다.

## 목차
1. [왜 버전 관리가 필요한가?](#왜-버전-관리가-필요한가)
2. [URL Path Versioning](#url-path-versioning)
3. [Header Versioning](#header-versioning)
4. [Query Parameter Versioning](#query-parameter-versioning)
5. [버전 관리 전략 비교](#버전-관리-전략-비교)
6. [하위 호환성](#하위-호환성)
7. [지원 중단 전략](#지원-중단-전략)
8. [버전 생명주기](#버전-생명주기)
9. [연습 문제](#연습-문제)

---

## 왜 버전 관리가 필요한가?

### Breaking vs Non-Breaking 변경

```python
# NON-BREAKING changes (safe to deploy without versioning):
# - Adding a new optional field to a response
# - Adding a new endpoint
# - Adding a new optional query parameter
# - Adding a new HTTP method to an existing resource
# - Making a required field optional
# - Adding a new enum value (if clients are tolerant)

# BREAKING changes (require versioning):
# - Removing a field from a response
# - Renaming a field
# - Changing a field's type (string -> integer)
# - Removing an endpoint
# - Changing the URL structure
# - Making an optional field required
# - Changing the meaning of a status code
# - Changing authentication mechanism
# - Changing pagination format
```

### Breaking Change의 비용

```
Without versioning:
  1. You deploy a breaking change
  2. All clients break simultaneously
  3. Emergency rollback or hotfix
  4. Loss of trust, support tickets, downtime

With versioning:
  1. You deploy v2 alongside v1
  2. v1 clients continue working unchanged
  3. Clients migrate to v2 at their own pace
  4. v1 is deprecated, then sunset after migration
```

---

## URL Path Versioning

버전 번호가 URL 경로의 일부가 됩니다. 가장 일반적인 방식입니다.

### 구현

```python
from fastapi import FastAPI, APIRouter

app = FastAPI()

# Version 1
v1_router = APIRouter(prefix="/api/v1")

@v1_router.get("/users")
async def list_users_v1():
    """V1: Returns users with 'name' as a single field."""
    return {
        "data": [
            {"id": 1, "name": "Alice Smith", "email": "alice@example.com"},
            {"id": 2, "name": "Bob Jones", "email": "bob@example.com"},
        ]
    }

@v1_router.get("/users/{user_id}")
async def get_user_v1(user_id: int):
    return {"id": user_id, "name": "Alice Smith", "email": "alice@example.com"}


# Version 2: Breaking change -- split 'name' into 'first_name' and 'last_name'
v2_router = APIRouter(prefix="/api/v2")

@v2_router.get("/users")
async def list_users_v2():
    """V2: Breaking change -- 'name' split into 'first_name' and 'last_name'."""
    return {
        "data": [
            {"id": 1, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"},
            {"id": 2, "first_name": "Bob", "last_name": "Jones", "email": "bob@example.com"},
        ]
    }

@v2_router.get("/users/{user_id}")
async def get_user_v2(user_id: int):
    return {"id": user_id, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"}


# Mount both versions
app.include_router(v1_router)
app.include_router(v2_router)

# Both work simultaneously:
# GET /api/v1/users  -> {"name": "Alice Smith"}
# GET /api/v2/users  -> {"first_name": "Alice", "last_name": "Smith"}
```

### 조직화된 프로젝트 구조

```
app/
├── main.py
├── v1/
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py      # Pydantic models for v1
│   └── services.py     # Business logic (may be shared)
├── v2/
│   ├── __init__.py
│   ├── router.py
│   ├── schemas.py      # Pydantic models for v2
│   └── services.py
└── shared/
    ├── __init__.py
    ├── database.py      # Shared database access
    ├── auth.py          # Shared authentication
    └── models.py        # Shared ORM models
```

```python
# main.py
from fastapi import FastAPI
from v1.router import router as v1_router
from v2.router import router as v2_router

app = FastAPI(title="My API", version="2.0.0")

app.include_router(v1_router, prefix="/api/v1", tags=["v1"])
app.include_router(v2_router, prefix="/api/v2", tags=["v2"])
```

### Flask에서 URL Versioning

```python
from flask import Flask, Blueprint, jsonify

app = Flask(__name__)

# Version 1
v1 = Blueprint("v1", __name__, url_prefix="/api/v1")

@v1.get("/users")
def list_users_v1():
    return jsonify({
        "data": [{"id": 1, "name": "Alice Smith"}]
    })

# Version 2
v2 = Blueprint("v2", __name__, url_prefix="/api/v2")

@v2.get("/users")
def list_users_v2():
    return jsonify({
        "data": [{"id": 1, "first_name": "Alice", "last_name": "Smith"}]
    })

app.register_blueprint(v1)
app.register_blueprint(v2)
```

---

## Header Versioning

버전이 커스텀 요청 헤더에 지정됩니다. URL은 깔끔하고 안정적으로 유지됩니다.

### 구현

```python
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

@app.get("/api/users")
async def list_users(
    api_version: str = Header(default="1", alias="X-API-Version"),
):
    """
    Header versioning: client specifies version in X-API-Version header.

    Usage:
        GET /api/users
        X-API-Version: 1     -> v1 response format

        GET /api/users
        X-API-Version: 2     -> v2 response format

        GET /api/users        -> defaults to v1 (backward compatible)
    """
    if api_version == "1":
        return {
            "data": [{"id": 1, "name": "Alice Smith", "email": "alice@example.com"}]
        }
    elif api_version == "2":
        return {
            "data": [{"id": 1, "first_name": "Alice", "last_name": "Smith", "email": "alice@example.com"}]
        }
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported API version: {api_version}. Supported: 1, 2"
        )
```

### Accept 헤더 Versioning (Content Negotiation)

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import re

app = FastAPI()

def parse_api_version(accept: str) -> int:
    """
    Parse version from Accept header.
    Example: application/vnd.myapi.v2+json -> 2
    """
    match = re.search(r"application/vnd\.myapi\.v(\d+)\+json", accept)
    if match:
        return int(match.group(1))
    return 1  # default version

@app.get("/api/users")
async def list_users(request: Request):
    """
    Content negotiation versioning via Accept header.

    Usage:
        GET /api/users
        Accept: application/vnd.myapi.v1+json

        GET /api/users
        Accept: application/vnd.myapi.v2+json
    """
    accept = request.headers.get("accept", "")
    version = parse_api_version(accept)

    if version == 1:
        data = [{"id": 1, "name": "Alice Smith"}]
    elif version == 2:
        data = [{"id": 1, "first_name": "Alice", "last_name": "Smith"}]
    else:
        raise HTTPException(status_code=406, detail=f"Unsupported version: v{version}")

    return JSONResponse(
        content={"data": data},
        media_type=f"application/vnd.myapi.v{version}+json",
    )
```

---

## Query Parameter Versioning

버전이 쿼리 파라미터로 전달됩니다.

### 구현

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/api/users")
async def list_users(
    version: int = Query(default=1, alias="v", ge=1, le=2, description="API version"),
):
    """
    Query parameter versioning.

    Usage:
        GET /api/users?v=1    -> v1 format
        GET /api/users?v=2    -> v2 format
        GET /api/users        -> defaults to v1
    """
    if version == 1:
        return {"data": [{"id": 1, "name": "Alice Smith"}]}
    else:
        return {"data": [{"id": 1, "first_name": "Alice", "last_name": "Smith"}]}
```

---

## 버전 관리 전략 비교

| 측면 | URL Path | Header | Query Parameter |
|--------|----------|--------|-----------------|
| 가시성 | `/api/v2/users` -- 매우 명확 | 헤더에 숨겨짐 | `?v=2` -- 명확 |
| 캐시 가능성 | 우수 (다른 URL = 다른 캐시) | `Vary` 헤더 필요 | 양호 (쿼리 스트링이 캐시 키에 포함) |
| 단순성 | 매우 단순 | 보통 | 단순 |
| 탐색 용이성 | 쉬움 (URL 탐색) | 문서 필요 | 쉬움 |
| 링크 공유 | 링크에 버전 포함 | 링크에 버전 미포함 | 링크에 버전 포함 |
| 클라이언트 복잡도 | 낮음 | 중간 (헤더 설정 필요) | 낮음 |
| URL 깔끔함 | 버전으로 인한 복잡 | 깔끔한 URL | 약간 복잡 |
| 라우팅 | 프레임워크 네이티브 | 커스텀 미들웨어 | 프레임워크 네이티브 |
| 사용 기업 | GitHub, Stripe, Twilio | Azure, Google Cloud | Netflix, Amazon |

### 권장 사항

```python
# For most APIs:
# 1. Use URL path versioning (/api/v1/) -- simplest, most common, best tooling support
# 2. Only bump the major version for breaking changes
# 3. Add new features without versioning (additive changes)

# Decision matrix:
def choose_versioning(
    audience: str,       # "public" | "internal" | "partner"
    breaking_frequency: str,  # "rare" | "frequent"
) -> str:
    if audience == "public":
        return "URL path"  # most discoverable, simplest for external devs
    if audience == "internal" and breaking_frequency == "frequent":
        return "header"  # clean URLs, easy to change
    return "URL path"  # safe default
```

---

## 하위 호환성

### 추가적 변경 (Non-Breaking)

```python
from pydantic import BaseModel

# Original response (v1):
class UserV1(BaseModel):
    id: int
    name: str
    email: str

# Enhanced response (still v1 -- backward compatible):
class UserV1Enhanced(BaseModel):
    id: int
    name: str
    email: str
    # New fields added as optional -- existing clients ignore them
    avatar_url: str | None = None
    created_at: str | None = None
    department: str | None = None

# Clients that only read id/name/email continue to work.
# Clients that want the new fields can start using them immediately.
# No version bump needed.
```

### Tolerant Reader 패턴

알 수 없는 필드와 누락된 선택적 필드에 관대한 클라이언트를 설계합니다:

```python
# Server adds a new field "department" to the user response.
# A well-designed client ignores fields it does not recognize.

import httpx

async def get_user(user_id: int) -> dict:
    """Tolerant reader: only extract the fields we need."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.example.com/api/v1/users/{user_id}")
        data = response.json()

    # Extract only what we need -- ignore unknown fields
    return {
        "id": data["id"],
        "name": data["name"],
        "email": data.get("email", ""),  # use .get() for optional fields
    }
    # If the server adds "department", "avatar_url", etc., this code still works.
```

### Breaking Change 방지 전략

```python
# Strategy 1: Add fields, never remove them
# Before: {"id": 1, "name": "Alice"}
# After:  {"id": 1, "name": "Alice", "display_name": "Alice S."}

# Strategy 2: Deprecate fields before removing
# Phase 1: Add new field, keep old field, add deprecation notice
# {"id": 1, "name": "Alice", "first_name": "Alice", "last_name": "Smith"}
# Phase 2: Remove old field in next major version

# Strategy 3: Use nullable types for new required concepts
# Instead of making a new field required (breaking):
# Add it as optional first, then require it in the next version

# Strategy 4: Response evolution with envelope
class UserResponse(BaseModel):
    id: int
    name: str                      # keep for backward compatibility
    first_name: str | None = None  # new field, optional
    last_name: str | None = None   # new field, optional
    email: str

    # Both old and new clients are satisfied:
    # Old client reads "name" -> works
    # New client reads "first_name" + "last_name" -> works
```

---

## 지원 중단 전략

### Sunset Header (RFC 8594)

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import warnings

app = FastAPI()

DEPRECATED_ENDPOINTS = {
    "/api/v1/users": {
        "sunset_date": "2025-06-01",
        "successor": "/api/v2/users",
        "migration_guide": "https://docs.example.com/migration/v1-to-v2",
    }
}

@app.middleware("http")
async def deprecation_middleware(request: Request, call_next):
    """Add deprecation headers to sunset endpoints."""
    response = await call_next(request)
    path = request.url.path

    deprecation_info = DEPRECATED_ENDPOINTS.get(path)
    if deprecation_info:
        response.headers["Deprecation"] = "true"
        response.headers["Sunset"] = deprecation_info["sunset_date"]
        response.headers["Link"] = (
            f'<{deprecation_info["successor"]}>; rel="successor-version", '
            f'<{deprecation_info["migration_guide"]}>; rel="deprecation"'
        )

    return response

@app.get("/api/v1/users")
async def list_users_v1():
    """DEPRECATED: Use /api/v2/users instead. Sunset: 2025-06-01."""
    return JSONResponse(
        content={
            "data": [{"id": 1, "name": "Alice Smith"}],
            "_deprecation": {
                "message": "This endpoint is deprecated. Use /api/v2/users instead.",
                "sunset_date": "2025-06-01",
                "migration_guide": "https://docs.example.com/migration/v1-to-v2",
            }
        },
        headers={
            "Deprecation": "true",
            "Sunset": "Sat, 01 Jun 2025 00:00:00 GMT",
        }
    )
```

### 지원 중단 커뮤니케이션 계획

```
Timeline for deprecating an API version:

Month 0:  Announce deprecation
          - Blog post, changelog, email to registered developers
          - Add Deprecation: true header to all v1 responses
          - Set Sunset date (usually 6-12 months out)

Month 1:  v2 is stable
          - Update all documentation to show v2 as primary
          - Add migration guide with code examples
          - Provide automated migration tools if possible

Month 3:  Usage monitoring
          - Track v1 usage metrics
          - Contact high-volume v1 consumers directly
          - Offer migration support

Month 6:  Sunset warning
          - Return 299 Warning header on v1 responses
          - Log all v1 requests for final outreach
          - Ensure <5% of traffic is still on v1

Month 9:  Hard sunset
          - v1 returns 410 Gone
          - Response body includes migration guide link
          - Keep this 410 response for 6+ months
```

### Sunset 응답

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/v1/{path:path}")
async def sunset_v1(path: str):
    """All v1 endpoints return 410 Gone after sunset date."""
    return JSONResponse(
        content={
            "error": {
                "code": "VERSION_SUNSET",
                "message": "API v1 has been discontinued as of 2025-06-01.",
                "migration_guide": "https://docs.example.com/migration/v1-to-v2",
                "successor": f"/api/v2/{path}",
            }
        },
        status_code=410,
    )
```

---

## 버전 생명주기

### 생명주기 단계

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌────────────┐    ┌──────────┐
│  Beta    │───►│  Stable  │───►│  Deprecated  │───►│  Sunset  │───►│  Removed │
│ (preview)│    │ (current)│    │ (maintenance)│    │  (410)   │    │  (gone)  │
└─────────┘    └──────────┘    └────────────┘    └──────────┘    └──────────┘
  0-3 months    12+ months      3-6 months        3-6 months      permanent
```

### 생명주기 구현

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from enum import Enum
from datetime import date

app = FastAPI()

class VersionStatus(str, Enum):
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"

VERSION_REGISTRY = {
    "v1": {
        "status": VersionStatus.DEPRECATED,
        "released": "2023-01-01",
        "deprecated": "2024-06-01",
        "sunset": "2025-06-01",
        "successor": "v2",
    },
    "v2": {
        "status": VersionStatus.STABLE,
        "released": "2024-06-01",
        "deprecated": None,
        "sunset": None,
        "successor": None,
    },
    "v3": {
        "status": VersionStatus.BETA,
        "released": None,
        "deprecated": None,
        "sunset": None,
        "successor": None,
    },
}

@app.get("/api/versions")
async def list_versions():
    """Discovery endpoint: list all API versions and their status."""
    return {
        "versions": [
            {
                "version": version,
                "status": info["status"].value,
                "released": info["released"],
                "deprecated": info["deprecated"],
                "sunset": info["sunset"],
                "base_url": f"/api/{version}",
            }
            for version, info in VERSION_REGISTRY.items()
        ],
        "current": "v2",
        "latest_beta": "v3",
    }

@app.middleware("http")
async def version_lifecycle_middleware(request: Request, call_next):
    """Enforce version lifecycle policies."""
    path = request.url.path

    # Extract version from path
    for version, info in VERSION_REGISTRY.items():
        if path.startswith(f"/api/{version}/"):
            status = info["status"]

            # Beta: add warning header
            if status == VersionStatus.BETA:
                response = await call_next(request)
                response.headers["X-API-Status"] = "beta"
                response.headers["Warning"] = '199 - "This API version is in beta and may change"'
                return response

            # Deprecated: add sunset headers
            if status == VersionStatus.DEPRECATED:
                response = await call_next(request)
                response.headers["Deprecation"] = "true"
                response.headers["Sunset"] = info["sunset"]
                response.headers["X-API-Status"] = "deprecated"
                successor = info["successor"]
                response.headers["Link"] = f'</api/{successor}/>; rel="successor-version"'
                return response

            # Sunset: return 410
            if status == VersionStatus.SUNSET:
                successor = info["successor"]
                return JSONResponse(
                    status_code=410,
                    content={
                        "error": {
                            "code": "VERSION_SUNSET",
                            "message": f"API {version} was sunset on {info['sunset']}.",
                            "successor": f"/api/{successor}/",
                        }
                    },
                )

            break  # stable version, no special handling

    return await call_next(request)
```

### API를 위한 Semantic Versioning

```python
# Semantic Versioning: MAJOR.MINOR.PATCH
#
# MAJOR: Breaking changes (bump the URL version: /v1/ -> /v2/)
# MINOR: New features, backward compatible (no URL change)
# PATCH: Bug fixes, backward compatible (no URL change)
#
# In the URL, only the MAJOR version appears:
# /api/v1/users   -- covers v1.0.0 through v1.99.99
# /api/v2/users   -- starts with v2.0.0
#
# The full version can be exposed via header or metadata:

@app.get("/api/v2/version")
async def get_version_info():
    return {
        "api_version": "v2",
        "full_version": "2.3.1",
        "release_date": "2025-01-15",
        "changelog": "https://docs.example.com/changelog",
    }
```

---

## 연습 문제

### 연습 문제 1: URL 버전 관리 API

두 가지 API 버전을 가진 FastAPI 애플리케이션을 구축하십시오:
- v1: `GET /api/v1/products`는 `{"name": "...", "price": "..."}`를 반환 (price가 문자열)
- v2: `GET /api/v2/products`는 `{"name": "...", "price_cents": 999}`를 반환 (price가 정수, 센트 단위)
- 두 버전 모두 동일한 데이터베이스/서비스 계층을 공유
- 공유 서비스 모듈과 별도의 라우터 사용

### 연습 문제 2: Header Versioning 미들웨어

헤더 기반 버전 관리를 구현하십시오:
- `X-API-Version` 헤더 수용 (기본값은 최신 안정 버전)
- `Accept: application/vnd.myapp.v2+json` 헤더 수용
- 버전에 따라 올바른 핸들러로 라우팅
- 응답에 적절한 `Content-Type` 반환
- 지원하지 않는 버전에 대해 400 반환

### 연습 문제 3: 지원 중단 시스템

지원 중단 관리 시스템을 구축하십시오:
- Sunset 날짜와 함께 엔드포인트를 지원 중단으로 등록
- `Deprecation`, `Sunset`, `Link` 헤더 자동 추가
- 지원 중단된 엔드포인트 호출 시 경고 로깅
- Sunset 날짜 이후 410 Gone 반환
- `/api/versions` 탐색 엔드포인트 제공

### 연습 문제 4: 하위 호환성 있는 진화

v1 사용자 API를 시작하여 클라이언트를 깨뜨리지 않고 세 단계를 거쳐 발전시키십시오:
1. v1.0: `{"name": "Alice Smith", "email": "alice@example.com"}`
2. v1.1: `avatar_url` 추가 (선택적, 버전 변경 없음)
3. v1.2: `name`과 함께 `first_name`, `last_name` 추가 (버전 변경 없음)
4. v2.0: `name` 제거, `first_name`과 `last_name`만 유지 (버전 변경)

v1.0 클라이언트가 1~3단계를 통해 변경 없이 작동함을 보여주십시오.

### 연습 문제 5: 버전 생명주기 대시보드

다음을 노출하는 엔드포인트를 생성하십시오:
- `GET /api/versions` -- 모든 버전과 상태 목록
- `GET /api/versions/v1/changelog` -- v1의 변경 사항
- `GET /api/versions/v2/breaking-changes` -- v1에서 v2로의 breaking change
- `GET /api/health` -- 현재 버전 정보 포함

---

## 요약

이 레슨에서 다룬 내용:
1. URL path versioning: 단순하고, 탐색 가능하며, 캐시 친화적 (가장 권장)
2. Header versioning: 깔끔한 URL, 문서 필요, content negotiation에 사용
3. Query parameter versioning: 단순하지만 덜 일반적
4. 하위 호환성: 추가적 변경, tolerant reader, 필드 지원 중단
5. 지원 중단 전략: Sunset 헤더, 마이그레이션 가이드, 커뮤니케이션 타임라인
6. 버전 생명주기: beta, stable, deprecated, sunset, removed 단계

---

**이전**: [인증과 인가](06_Authentication_and_Authorization.md) | [개요](00_Overview.md) | **다음**: [오류 처리](08_Error_Handling.md)

**License**: CC BY-NC 4.0
