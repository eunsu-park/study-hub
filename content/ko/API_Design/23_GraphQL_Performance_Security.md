# 23. GraphQL 성능과 보안(GraphQL Performance and Security)

**이전**: [GraphQL 페더레이션](./22_GraphQL_Federation.md) | **다음**: [GraphQL 테스팅과 도구](./24_GraphQL_Testing_and_Tooling.md)

**난이도**: ⭐⭐⭐⭐

---

## 학습 목표

- 쿼리 복잡도를 분석하고 비용 기반 쿼리 제한을 구현할 수 있다
- 리소스 고갈을 방지하기 위해 깊이 제한과 너비 제한을 적용할 수 있다
- 성능과 보안 이점을 위해 영속 쿼리(persisted queries)를 구현할 수 있다
- 여러 수준(클라이언트, CDN, 애플리케이션)에서 GraphQL 캐싱 전략을 설계할 수 있다
- GraphQL의 단일 엔드포인트 모델에 특화된 속도 제한 전략을 적용할 수 있다
- 일반적인 공격 벡터(인젝션, 배칭 남용, 인트로스펙션 유출)로부터 GraphQL API를 보호할 수 있다

---

## 목차

1. [GraphQL 보안 환경](#1-graphql-보안-환경)
2. [쿼리 복잡도 분석](#2-쿼리-복잡도-분석)
3. [깊이와 너비 제한](#3-깊이와-너비-제한)
4. [영속 쿼리](#4-영속-쿼리)
5. [캐싱 전략](#5-캐싱-전략)
6. [속도 제한](#6-속도-제한)
7. [입력 유효성 검사와 인젝션 방지](#7-입력-유효성-검사와-인젝션-방지)
8. [인증과 권한 부여 강화](#8-인증과-권한-부여-강화)
9. [서비스 거부 방어](#9-서비스-거부-방어)
10. [모니터링과 관찰 가능성](#10-모니터링과-관찰-가능성)
11. [연습 문제](#11-연습-문제)
12. [참고 자료](#12-참고-자료)

---

## 1. GraphQL 보안 환경

GraphQL은 클라이언트가 쿼리 구조를 제어하기 때문에 REST에 비해 고유한 보안 과제를 도입합니다.

### 고유한 공격 표면

| 공격 | 설명 | REST 동등물 |
|------|------|------------|
| **깊은 중첩** | 서버 리소스를 고갈시키는 재귀 쿼리 | 직접적 동등물 없음 |
| **넓은 쿼리** | 수백 개 필드를 동시에 선택 | 엔드포인트 설계로 고정 |
| **배치 남용** | 한 요청에 많은 작업 전송 | 엔드포인트별 속도 제한 |
| **인트로스펙션 유출** | 공격자에게 스키마 노출 | API 문서 노출 |
| **별칭 남용** | 별칭으로 비용이 큰 리졸버 증식 | 직접적 동등물 없음 |

### 심층 방어 전략

```
Layer 1: 네트워크       → WAF, TLS, IP 허용 목록
Layer 2: 전송           → 속도 제한, 요청 크기 제한
Layer 3: 쿼리 분석      → 복잡도, 깊이, 너비 제한
Layer 4: 실행           → 타임아웃, 리졸버 수준 인증
Layer 5: 데이터         → 필드 수준 권한 부여, 필터링
Layer 6: 응답           → 오류 정리, 스택 트레이스 제거
```

---

## 2. 쿼리 복잡도 분석

쿼리 복잡도 분석은 각 필드에 비용을 할당하고 임계값을 초과하는 쿼리를 거부합니다.

### 정적 분석(Static Analysis)

예상 리소스 사용량에 따라 필드에 비용을 할당합니다:

```python
import strawberry
from strawberry.extensions import SchemaExtension


# 타입에 비용 어노테이션 적용
@strawberry.type
class User:
    id: strawberry.ID                   # cost: 0 (단순)
    username: str                        # cost: 0 (단순)
    email: str                           # cost: 0 (단순)

    @strawberry.field(
        extensions=[{"cost": 10}]        # cost: 10 (DB 쿼리)
    )
    async def posts(
        self, info, first: int = 10
    ) -> list["Post"]:
        return await load_posts(self.id, first)

    @strawberry.field(extensions=[{"cost": 5}])
    async def followers(
        self, info, first: int = 10
    ) -> list["User"]:
        return await load_followers(self.id, first)
```

### 복잡도 계산기

```python
class QueryComplexityExtension(SchemaExtension):
    """복잡도 임계값을 초과하는 쿼리를 거부합니다."""
    MAX_COMPLEXITY = 1000

    def on_operation(self):
        complexity = self._calculate_complexity(
            self.execution_context.graphql_document,
            self.execution_context.schema,
        )
        if complexity > self.MAX_COMPLEXITY:
            raise ValueError(
                f"쿼리 복잡도 {complexity}가 최대 {self.MAX_COMPLEXITY}을 초과합니다. "
                f"쿼리를 단순화하세요."
            )
        yield

    def _calculate_complexity(self, document, schema, depth=0):
        """재귀적으로 쿼리 복잡도를 계산합니다."""
        total = 0
        for definition in document.definitions:
            for selection in definition.selection_set.selections:
                field_cost = self._get_field_cost(selection, schema)
                multiplier = self._get_list_multiplier(selection)
                child_cost = 0
                if selection.selection_set:
                    child_cost = self._calculate_selections(
                        selection.selection_set.selections,
                        schema,
                        depth + 1,
                    )
                total += field_cost + (multiplier * child_cost)
        return total

    def _get_field_cost(self, selection, schema) -> int:
        """필드의 extensions에서 비용을 가져옵니다."""
        # 기본 비용: 필드당 1
        return 1

    def _get_list_multiplier(self, selection) -> int:
        """'first'/'limit' 인자에서 리스트 크기 배수를 가져옵니다."""
        for arg in selection.arguments:
            if arg.name.value in ("first", "limit"):
                return int(arg.value.value)
        return 1  # 기본값: 리스트 아님


schema = strawberry.Schema(
    query=Query,
    extensions=[QueryComplexityExtension],
)
```

### 비용 계산 예시

```graphql
query {
  users(first: 10) {            # 비용: 1 + 10 * 자식
    username                     # 비용: 0
    posts(first: 5) {            # 비용: 10 + 5 * 자식
      title                      # 비용: 0
      comments(first: 3) {       # 비용: 5 + 3 * 자식
        body                     # 비용: 0
        author {                 # 비용: 1
          username               # 비용: 0
        }
      }
    }
  }
}
# 총: 1 + 10 * (0 + 10 + 5 * (0 + 5 + 3 * (0 + 1 + 0))) = 501
```

---

## 3. 깊이와 너비 제한

### 깊이 제한

깊게 중첩된 쿼리를 방지합니다:

```python
class DepthLimitExtension(SchemaExtension):
    MAX_DEPTH = 10

    def on_operation(self):
        depth = self._calculate_depth(
            self.execution_context.graphql_document
        )
        if depth > self.MAX_DEPTH:
            raise ValueError(
                f"쿼리 깊이 {depth}가 최대 {self.MAX_DEPTH}을 초과합니다"
            )
        yield

    def _calculate_depth(self, document) -> int:
        max_depth = 0
        for definition in document.definitions:
            depth = self._selection_depth(definition.selection_set, 0)
            max_depth = max(max_depth, depth)
        return max_depth

    def _selection_depth(self, selection_set, current_depth) -> int:
        if not selection_set:
            return current_depth
        max_child = current_depth
        for selection in selection_set.selections:
            if hasattr(selection, "selection_set") and selection.selection_set:
                child = self._selection_depth(
                    selection.selection_set, current_depth + 1
                )
                max_child = max(max_child, child)
        return max_child
```

### 너비 제한(Breadth Limiting)

너무 많은 필드를 선택하는 광범위한 쿼리를 방지합니다:

```python
class BreadthLimitExtension(SchemaExtension):
    MAX_BREADTH = 100  # 최대 총 필드 선택 수

    def on_operation(self):
        breadth = self._count_selections(
            self.execution_context.graphql_document
        )
        if breadth > self.MAX_BREADTH:
            raise ValueError(
                f"쿼리가 {breadth}개 필드를 선택하며, "
                f"제한 {self.MAX_BREADTH}을 초과합니다"
            )
        yield

    def _count_selections(self, document) -> int:
        total = 0
        for definition in document.definitions:
            total += self._count_in_set(definition.selection_set)
        return total

    def _count_in_set(self, selection_set) -> int:
        if not selection_set:
            return 0
        count = len(selection_set.selections)
        for selection in selection_set.selections:
            if hasattr(selection, "selection_set") and selection.selection_set:
                count += self._count_in_set(selection.selection_set)
        return count
```

### 별칭 남용 방지

별칭을 사용하면 리졸버 호출이 증폭됩니다:

```graphql
# 악의적 쿼리: user 리졸버를 1000번 호출
query {
  u1: user(id: "1") { username }
  u2: user(id: "2") { username }
  u3: user(id: "3") { username }
  # ... 997개 더
}
```

```python
class AliasLimitExtension(SchemaExtension):
    MAX_ALIASES = 50

    def on_operation(self):
        alias_count = self._count_aliases(
            self.execution_context.graphql_document
        )
        if alias_count > self.MAX_ALIASES:
            raise ValueError(
                f"쿼리가 {alias_count}개의 별칭을 사용하며, "
                f"제한 {self.MAX_ALIASES}을 초과합니다"
            )
        yield

    def _count_aliases(self, document) -> int:
        count = 0
        for definition in document.definitions:
            count += self._count_in_set(definition.selection_set)
        return count

    def _count_in_set(self, selection_set) -> int:
        if not selection_set:
            return 0
        count = sum(
            1 for s in selection_set.selections
            if hasattr(s, "alias") and s.alias
        )
        for selection in selection_set.selections:
            if hasattr(selection, "selection_set") and selection.selection_set:
                count += self._count_in_set(selection.selection_set)
        return count
```

### 권장 제한

| 제한 | 공개 API | 내부 API |
|------|---------|---------|
| 최대 깊이 | 7-10 | 15-20 |
| 최대 너비 | 50-100 | 200-500 |
| 최대 복잡도 | 500-1000 | 5000-10000 |
| 최대 별칭 | 20-50 | 100 |
| 요청 크기 | 10 KB | 100 KB |

---

## 4. 영속 쿼리

### 영속 쿼리란?

전체 쿼리 문자열을 보내는 대신 클라이언트가 해시를 보냅니다. 서버가 해시로 쿼리를 조회합니다.

### 자동 영속 쿼리 (APQ)

```
첫 번째 요청 (캐시 미스):
  Client → { extensions: { persistedQuery: { sha256Hash: "abc123" } } }
  Server → { errors: [{ message: "PersistedQueryNotFound" }] }

  Client → { query: "{ user(id: 1) { name } }", extensions: { persistedQuery: { sha256Hash: "abc123" } } }
  Server → 해시 저장 → { data: { user: { name: "Alice" } } }

후속 요청 (캐시 히트):
  Client → { extensions: { persistedQuery: { sha256Hash: "abc123" } } }
  Server → 해시 조회 → { data: { user: { name: "Alice" } } }
```

### 구현

```python
import hashlib
from functools import lru_cache

# 인메모리 저장소 (프로덕션에서는 Redis 사용)
_query_store: dict[str, str] = {}


class PersistedQueryExtension(SchemaExtension):
    """자동 영속 쿼리(APQ) 지원."""

    def on_operation(self):
        request = self.execution_context
        extensions = request.extensions or {}
        persisted = extensions.get("persistedQuery", {})
        sha256_hash = persisted.get("sha256Hash")

        if sha256_hash:
            if request.query:
                # 클라이언트가 쿼리 + 해시를 보냄: 저장
                computed_hash = hashlib.sha256(
                    request.query.encode()
                ).hexdigest()
                if computed_hash != sha256_hash:
                    raise ValueError("Persisted query hash mismatch")
                _query_store[sha256_hash] = request.query
            else:
                # 클라이언트가 해시만 보냄: 조회
                stored_query = _query_store.get(sha256_hash)
                if not stored_query:
                    raise ValueError("PersistedQueryNotFound")
                request.query = stored_query

        yield
```

### 잠금 영속 쿼리 (빌드 타임)

최대 보안을 위해 사전 등록된 쿼리만 허용합니다:

```python
# 빌드 단계: 클라이언트 코드에서 쿼리 추출
ALLOWED_QUERIES = {
    "abc123": "query GetUser($id: ID!) { user(id: $id) { username email } }",
    "def456": "mutation CreatePost($input: CreatePostInput!) { createPost(input: $input) { post { id } } }",
}


class StrictPersistedQueryExtension(SchemaExtension):
    """사전 등록된 쿼리만 허용 (임의 쿼리 불가)."""

    def on_operation(self):
        extensions = self.execution_context.extensions or {}
        persisted = extensions.get("persistedQuery", {})
        query_id = persisted.get("sha256Hash")

        if not query_id:
            raise ValueError(
                "임의 쿼리는 허용되지 않습니다. 영속 쿼리를 사용하세요."
            )

        stored = ALLOWED_QUERIES.get(query_id)
        if not stored:
            raise ValueError("알 수 없는 영속 쿼리 ID입니다")

        self.execution_context.query = stored
        yield
```

### 장점

| 장점 | 설명 |
|------|------|
| **작은 페이로드** | 해시(64바이트) vs. 전체 쿼리(잠재적으로 KB) |
| **CDN 캐싱** | 쿼리 해시가 있는 GET 요청은 캐시 가능 |
| **보안** | 잠금 모드는 임의 쿼리 실행 방지 |
| **성능** | 알려진 쿼리에 대해 파싱 생략 |

---

## 5. 캐싱 전략

GraphQL 캐싱은 단일 엔드포인트 모델 때문에 REST보다 복잡합니다.

### 캐시 수준

```
클라이언트 캐시 (Apollo Client, urql)
    ↓
CDN / 엣지 캐시 (Cloudflare, Fastly)
    ↓
애플리케이션 캐시 (Redis, 인메모리)
    ↓
DataLoader 캐시 (요청별)
    ↓
데이터베이스 캐시 (쿼리 캐시, 연결 풀)
```

### 클라이언트 측 정규화 캐시(Client-Side Normalized Cache)

```javascript
// Apollo Client 정규화 캐시
import { InMemoryCache } from '@apollo/client';

const cache = new InMemoryCache({
  typePolicies: {
    User: {
      keyFields: ['id'],  // 캐시 키: User:42
    },
    Post: {
      keyFields: ['id'],
      fields: {
        comments: {
          // 페이지네이션된 결과 병합
          keyArgs: false,
          merge(existing = [], incoming) {
            return [...existing, ...incoming];
          },
        },
      },
    },
  },
});
```

### 응답 캐싱

```python
import hashlib
import json
from datetime import timedelta


class ResponseCacheExtension(SchemaExtension):
    """쿼리 해시별로 전체 GraphQL 응답을 캐시합니다."""

    def __init__(self, *, redis_client, default_ttl: int = 60):
        self.redis = redis_client
        self.default_ttl = default_ttl

    def on_operation(self):
        # 쿼리 + 변수로 캐시 키 생성
        query = self.execution_context.query
        variables = self.execution_context.variables or {}
        cache_key = self._build_key(query, variables)

        # 캐시 확인
        cached = self.redis.get(cache_key)
        if cached:
            self.execution_context.result = json.loads(cached)
            return  # 실행 생략

        yield  # 쿼리 실행

        # 결과 저장
        result = self.execution_context.result
        if result and not result.errors:
            self.redis.setex(
                cache_key,
                self.default_ttl,
                json.dumps(result.data),
            )

    def _build_key(self, query: str, variables: dict) -> str:
        raw = json.dumps({"q": query, "v": variables}, sort_keys=True)
        return f"gql:cache:{hashlib.sha256(raw.encode()).hexdigest()}"
```

### Cache-Control 디렉티브

```graphql
# 스키마 수준 캐시 힌트
type Query {
  posts(first: Int): [Post!]! @cacheControl(maxAge: 60)
  me: User @cacheControl(maxAge: 0, scope: PRIVATE)
}

type Post @cacheControl(maxAge: 300) {
  id: ID!
  title: String!
  viewCount: Int! @cacheControl(maxAge: 10)  # 더 휘발성이 높은 필드
}
```

### 영속 쿼리를 활용한 CDN 캐싱

```
# 영속 쿼리는 GET 요청을 가능하게 하여 CDN이 캐시할 수 있습니다:
GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc123"}}&variables={"id":"1"}

# CDN 캐시 키: URL + 쿼리 파라미터
# TTL: Cache-Control 헤더에서 지정
```

---

## 6. 속도 제한

### GraphQL 속도 제한의 과제

REST 속도 제한은 단순합니다: 엔드포인트별로 요청을 제한합니다. GraphQL은 모든 작업을 하나의 엔드포인트로 보내므로 엔드포인트별 제한이 효과적이지 않습니다.

### 비용 기반 속도 제한

```python
class CostBasedRateLimiter:
    """쿼리 복잡도 비용에 기반한 속도 제한."""

    def __init__(self, max_cost_per_minute: int = 1000):
        self.max_cost = max_cost_per_minute
        self.window = timedelta(minutes=1)
        self._usage = defaultdict(list)

    def check(self, client_id: str, query_cost: int) -> bool:
        now = datetime.now()
        cutoff = now - self.window
        self._usage[client_id] = [
            (ts, cost) for ts, cost in self._usage[client_id] if ts > cutoff
        ]
        current_cost = sum(cost for _, cost in self._usage[client_id])
        if current_cost + query_cost > self.max_cost:
            return False
        self._usage[client_id].append((now, query_cost))
        return True
```

### 속도 제한 헤더

```python
class RateLimitExtension(SchemaExtension):
    def on_operation(self):
        client_id = self._get_client_id()
        query_cost = self._calculate_cost()

        if not rate_limiter.check(client_id, query_cost):
            remaining = rate_limiter.remaining(client_id)
            raise ValueError(
                f"속도 제한 초과. 남은 예산: {remaining}. "
                f"쿼리 비용: {query_cost}."
            )

        yield

        # 속도 제한 정보를 extensions에 추가
        result = self.execution_context.result
        if result:
            result.extensions = result.extensions or {}
            result.extensions["rateLimit"] = {
                "cost": query_cost,
                "remaining": rate_limiter.remaining(client_id),
                "resetAt": (datetime.now() + timedelta(minutes=1)).isoformat(),
            }
```

### GitHub 스타일 속도 제한

GitHub의 GraphQL API는 포인트 기반 시스템을 사용합니다:

```json
{
  "data": { "...": "..." },
  "extensions": {
    "rateLimit": {
      "limit": 5000,
      "cost": 12,
      "remaining": 4988,
      "resetAt": "2024-01-01T00:00:00Z",
      "nodeCount": 42
    }
  }
}
```

---

## 7. 입력 유효성 검사와 인젝션 방지

### 쿼리 인젝션

```python
# 취약 — 절대 이렇게 하지 마세요
query = f'{{ user(name: "{user_input}") {{ email }} }}'

# 안전 — 항상 변수를 사용하세요
query = """
    query GetUser($name: String!) {
        user(name: $name) { email }
    }
"""
variables = {"name": user_input}
```

### 입력 유효성 검사

```python
@strawberry.input
class SearchInput:
    query: str
    limit: int = 10

    def __post_init__(self):
        if len(self.query) > 200:
            raise ValueError("검색 쿼리가 너무 깁니다 (최대 200자)")
        if self.limit < 1 or self.limit > 100:
            raise ValueError("limit은 1에서 100 사이여야 합니다")
        # 인젝션 문자 제거
        self.query = self.query.replace("\x00", "")
```

### 요청 크기 제한

```python
from starlette.middleware.base import BaseHTTPMiddleware

MAX_REQUEST_SIZE = 10 * 1024  # 10 KB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_SIZE:
                return JSONResponse(
                    {"error": f"요청이 너무 큽니다 (최대 {MAX_REQUEST_SIZE} 바이트)"},
                    status_code=413,
                )
        return await call_next(request)
```

---

## 8. 인증과 권한 부여 강화

### 토큰 유효성 검사

```python
async def get_context(request: Request) -> RequestContext:
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header else None

    current_user = None
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = await user_repo.find_by_id(payload["sub"])
        except jwt.ExpiredSignatureError:
            pass  # 토큰 만료 — 익명 처리
        except jwt.InvalidTokenError:
            pass  # 유효하지 않은 토큰 — 익명 처리

    return RequestContext(
        current_user=current_user,
        request=request,
    )
```

### 작업 수준 권한 부여

```python
# API 키 인증에 허용된 작업 목록
ALLOWED_OPERATIONS = {
    "api_key": {"GetProduct", "ListProducts", "SearchProducts"},
    "oauth": None,  # 모든 작업 허용
}


class AuthorizationExtension(SchemaExtension):
    def on_operation(self):
        auth_type = self.execution_context.context.auth_type
        op_name = self.execution_context.operation_name

        allowed = ALLOWED_OPERATIONS.get(auth_type)
        if allowed is not None and op_name not in allowed:
            raise PermissionError(
                f"작업 '{op_name}'은 {auth_type} 인증에서 허용되지 않습니다"
            )
        yield
```

### 프로덕션에서 인트로스펙션 비활성화

```python
class DisableIntrospectionExtension(SchemaExtension):
    def on_operation(self):
        query = self.execution_context.query or ""
        if "__schema" in query or "__type" in query:
            if not self.execution_context.context.is_admin:
                raise ValueError("인트로스펙션이 비활성화되어 있습니다")
        yield
```

---

## 9. 서비스 거부 방어

### 작업별 타임아웃

```python
class TimeoutExtension(SchemaExtension):
    TIMEOUT_SECONDS = 30

    def on_operation(self):
        try:
            with asyncio.timeout(self.TIMEOUT_SECONDS):
                yield
        except asyncio.TimeoutError:
            raise ValueError(
                f"쿼리 실행이 {self.TIMEOUT_SECONDS}초 타임아웃을 초과했습니다"
            )
```

### 배치 크기 제한

```python
@app.middleware("http")
async def limit_batch(request: Request, call_next):
    if request.url.path == "/graphql" and request.method == "POST":
        body = await request.body()
        try:
            parsed = json.loads(body)
            if isinstance(parsed, list) and len(parsed) > 10:
                return JSONResponse(
                    {"error": "최대 배치 크기는 10입니다"},
                    status_code=400,
                )
        except json.JSONDecodeError:
            pass
    return await call_next(request)
```

### 통합 보안 구성

```python
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        DepthLimitExtension,          # 최대 깊이: 10
        BreadthLimitExtension,        # 최대 필드: 100
        QueryComplexityExtension,     # 최대 비용: 1000
        AliasLimitExtension,          # 최대 별칭: 50
        TimeoutExtension,             # 최대 시간: 30초
        RateLimitExtension,           # 비용 기반 속도 제한
        DisableIntrospectionExtension,# 프로덕션에서 인트로스펙션 비활성화
        ErrorLoggingExtension,        # 모든 오류 로깅
        TimingExtension,              # 성능 추적
    ],
)
```

---

## 10. 모니터링과 관찰 가능성

### 주요 메트릭

| 메트릭 | 설명 | 알림 임계값 |
|--------|------|-----------|
| `graphql.operation.duration` | 쿼리 실행 시간 | p99 > 5초 |
| `graphql.operation.errors` | 작업별 오류 수 | > 5% 오류율 |
| `graphql.query.complexity` | 쿼리당 비용 | > 80% 제한 |
| `graphql.query.depth` | 쿼리당 중첩 깊이 | > 8 |
| `graphql.resolver.duration` | 리졸버별 타이밍 | p99 > 1초 |
| `graphql.dataloader.batch_size` | DataLoader 배치 크기 | 평균 < 2 (N+1 감지) |
| `graphql.rate_limit.rejections` | 거부된 요청 | > 10/분 |

### 구조화 로깅(Structured Logging)

```python
import structlog

logger = structlog.get_logger()


class ObservabilityExtension(SchemaExtension):
    def on_operation(self):
        start = time.monotonic()
        yield
        duration = time.monotonic() - start

        result = self.execution_context.result
        has_errors = bool(result and result.errors)

        logger.info(
            "graphql.operation",
            operation_name=self.execution_context.operation_name,
            duration_ms=round(duration * 1000, 2),
            has_errors=has_errors,
            error_count=len(result.errors) if result and result.errors else 0,
            client_id=self.execution_context.context.client_id,
        )
```

### Apollo 트레이싱 형식

```json
{
  "extensions": {
    "tracing": {
      "version": 1,
      "startTime": "2024-01-01T00:00:00.000Z",
      "endTime": "2024-01-01T00:00:00.045Z",
      "duration": 45000000,
      "execution": {
        "resolvers": [
          {
            "path": ["user"],
            "parentType": "Query",
            "fieldName": "user",
            "returnType": "User",
            "startOffset": 1000000,
            "duration": 20000000
          },
          {
            "path": ["user", "posts"],
            "parentType": "User",
            "fieldName": "posts",
            "returnType": "[Post!]!",
            "startOffset": 21000000,
            "duration": 15000000
          }
        ]
      }
    }
  }
}
```

---

## 11. 연습 문제

### 연습 1: 복잡도 계산기

완전한 쿼리 복잡도 계산기를 구현하세요:
- 스칼라 필드에 비용 0 할당
- 객체 필드에 비용 1 할당
- 리스트 인자(`first`, `limit`)에 자식 비용 곱하기
- 예상 비용이 있는 알려진 쿼리에 대해 테스트

### 연습 2: 보안 감사

다음 GraphQL 서버 구성을 감사하고 모든 취약점을 식별하세요:

```python
schema = strawberry.Schema(query=Query, mutation=Mutation)
app = FastAPI()
app.include_router(GraphQLRouter(schema), prefix="/graphql")
# 미들웨어 없음, 확장 없음, 인트로스펙션 활성화
```

최소 8개의 보안 문제를 나열하고 수정 방법을 제시하세요.

### 연습 3: 영속 쿼리 저장소

Redis 기반 영속 쿼리 저장소를 구현하세요:
- APQ 프로토콜 지원 (협상 + 캐싱)
- TTL 기반 만료 (24시간)
- 잠금 모드 토글 (알 수 없는 쿼리 거부)

### 연습 4: 속도 제한기

비용 기반 속도 제한기를 구축하세요:
- Redis 기반 슬라이딩 윈도우
- 사용자별 및 API 키별 제한
- 속도 제한 응답 헤더

---

## 12. 참고 자료

### 보안
- [OWASP GraphQL Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/GraphQL_Cheat_Sheet.html)
- [GraphQL Security Best Practices — Escape.tech](https://escape.tech/blog/graphql-security-best-practices/)
- [GraphQL API 보안 방법 — Apollo](https://www.apollographql.com/docs/technotes/TN0021-graph-security/)

### 성능
- [Apollo Persisted Queries](https://www.apollographql.com/docs/apollo-server/performance/apq/)
- [GraphQL Caching Strategies — The Guild](https://the-guild.dev/blog/graphql-response-caching)

---

**License**: CC BY-NC 4.0
