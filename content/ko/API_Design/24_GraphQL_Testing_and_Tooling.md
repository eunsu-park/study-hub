# 24. GraphQL 테스팅과 도구(GraphQL Testing and Tooling)

**이전**: [GraphQL 성능과 보안](./23_GraphQL_Performance_Security.md) | **다음**: [API 캡스톤 — 통합 게이트웨이](./25_API_Capstone_Unified_Gateway.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- Strawberry의 테스트 클라이언트를 사용하여 GraphQL 리졸버의 단위 테스트를 작성할 수 있다
- 테스트 데이터베이스에 대한 전체 쿼리 실행을 검증하는 통합 테스트를 구현할 수 있다
- 스냅샷 테스트를 사용하여 의도하지 않은 스키마 및 응답 변경을 감지할 수 있다
- 코드 생성 도구를 사용하여 GraphQL 스키마에서 타입이 지정된 클라이언트 코드를 생성할 수 있다
- GraphiQL과 Apollo Sandbox를 활용하여 대화형 API 탐색을 수행할 수 있다
- 스키마 린팅, 호환성 깨지는 변경 감지, 계약 테스트가 포함된 CI 파이프라인을 설정할 수 있다

---

## 목차

1. [테스팅 전략 개요](#1-테스팅-전략-개요)
2. [리졸버 단위 테스트](#2-리졸버-단위-테스트)
3. [통합 테스트](#3-통합-테스트)
4. [스냅샷 테스트](#4-스냅샷-테스트)
5. [모킹과 테스트 픽스처](#5-모킹과-테스트-픽스처)
6. [스키마 유효성 검사와 린팅](#6-스키마-유효성-검사와-린팅)
7. [코드 생성](#7-코드-생성)
8. [대화형 도구](#8-대화형-도구)
9. [CI/CD 파이프라인](#9-cicd-파이프라인)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 테스팅 전략 개요

### GraphQL을 위한 테스팅 피라미드

```
          ╱ E2E 테스트  ╲
         ╱ (브라우저 +    ╲
        ╱  GraphQL API)    ╲
       ╱─────────────────────╲
      ╱   통합 테스트           ╲
     ╱  (테스트 데이터베이스로    ╲
    ╱   전체 쿼리 실행)           ╲
   ╱───────────────────────────────╲
  ╱       단위 테스트                ╲
 ╱  (리졸버, 서비스,                  ╲
╱   DataLoader 격리 테스트)            ╲
──────────────────────────────────────────
```

### 각 수준에서 테스트할 항목

| 수준 | 테스트 대상 | 도구 |
|------|-----------|------|
| **단위** | 리졸버 로직, 서비스 메서드, 유효성 검사기 | pytest, mock |
| **통합** | 전체 쿼리 실행, DataLoader 배칭 | Strawberry TestClient, 테스트 DB |
| **스키마** | 호환성 깨지는 변경, 린팅, 폐기 | GraphQL Inspector, Rover |
| **계약** | 클라이언트-서버 일치 | 스냅샷 테스트 |
| **E2E** | API를 통한 전체 사용자 흐름 | httpx, Playwright |

---

## 2. 리졸버 단위 테스트

### Strawberry TestClient로 테스트

```python
# tests/conftest.py
import pytest
import strawberry
from strawberry.test import GraphQLTestClient
from schema import schema
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_context():
    ctx = MagicMock()
    ctx.current_user = MagicMock(id="1", username="testuser", role="USER")
    ctx.db = MagicMock()
    ctx.dataloaders = MagicMock()
    return ctx


@pytest.fixture
def client(mock_context):
    return GraphQLTestClient(schema, context_value=mock_context)
```

### 쿼리 테스트

```python
# tests/test_user_queries.py
import pytest


def test_get_user(client, mock_context):
    """ID로 단일 사용자 조회를 테스트합니다."""
    mock_context.db.users.find_by_id = AsyncMock(return_value={
        "id": "1",
        "username": "alice",
        "email": "alice@example.com",
        "bio": "Engineer",
    })

    result = client.query("""
        query GetUser($id: ID!) {
            user(id: $id) {
                id
                username
                email
                bio
            }
        }
    """, variables={"id": "1"})

    assert result.errors is None
    assert result.data["user"]["username"] == "alice"
    assert result.data["user"]["email"] == "alice@example.com"


def test_get_user_not_found(client, mock_context):
    """존재하지 않는 사용자 조회 시 null 반환을 테스트합니다."""
    mock_context.db.users.find_by_id = AsyncMock(return_value=None)

    result = client.query("""
        query { user(id: "999") { username } }
    """)

    assert result.errors is None
    assert result.data["user"] is None


def test_list_users_with_pagination(client, mock_context):
    """페이지네이션된 사용자 목록을 테스트합니다."""
    mock_context.db.users.find_all = AsyncMock(return_value=[
        {"id": "1", "username": "alice"},
        {"id": "2", "username": "bob"},
    ])

    result = client.query("""
        query { users(limit: 2) { id username } }
    """)

    assert result.errors is None
    assert len(result.data["users"]) == 2
```

### 뮤테이션 테스트

```python
def test_create_post_success(client, mock_context):
    """성공적인 게시글 생성을 테스트합니다."""
    mock_context.db.posts.create = AsyncMock(return_value={
        "id": "101", "title": "Test Post", "status": "DRAFT",
    })

    result = client.query("""
        mutation CreatePost($input: CreatePostInput!) {
            createPost(input: $input) {
                post { id title status }
                userErrors { field message code }
            }
        }
    """, variables={"input": {"title": "Test Post", "content": "Hello World"}})

    assert result.errors is None
    payload = result.data["createPost"]
    assert payload["post"]["title"] == "Test Post"
    assert len(payload["userErrors"]) == 0


def test_create_post_validation_error(client, mock_context):
    """유효하지 않은 입력으로 게시글 생성을 테스트합니다."""
    result = client.query("""
        mutation CreatePost($input: CreatePostInput!) {
            createPost(input: $input) {
                post { id }
                userErrors { field message code }
            }
        }
    """, variables={"input": {"title": "", "content": "short"}})

    payload = result.data["createPost"]
    assert payload["post"] is None
    assert len(payload["userErrors"]) > 0
```

### 권한 부여 테스트

```python
def test_admin_query_as_regular_user(client, mock_context):
    """일반 사용자가 관리자 쿼리에 접근할 수 없음을 테스트합니다."""
    mock_context.current_user.role = "USER"

    result = client.query("""
        query { adminDashboard { totalUsers totalPosts } }
    """)

    assert result.errors is not None
    assert "Admin access required" in result.errors[0].message


def test_admin_query_as_admin(client, mock_context):
    """관리자 사용자가 관리자 쿼리에 접근할 수 있음을 테스트합니다."""
    mock_context.current_user.role = "ADMIN"
    mock_context.db.dashboard.get = AsyncMock(return_value={
        "total_users": 100,
        "total_posts": 500,
    })

    result = client.query("""
        query { adminDashboard { totalUsers totalPosts } }
    """)

    assert result.errors is None
    assert result.data["adminDashboard"]["totalUsers"] == 100
```

---

## 3. 통합 테스트

### 테스트 데이터베이스 설정

```python
# tests/conftest.py
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def engine():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(engine):
    session_factory = async_sessionmaker(engine)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def seeded_db(db_session):
    """샘플 데이터로 테스트 데이터베이스를 채웁니다."""
    users = [
        UserModel(id="1", username="alice", email="alice@test.com"),
        UserModel(id="2", username="bob", email="bob@test.com"),
    ]
    posts = [
        PostModel(id="101", title="First Post", author_id="1"),
        PostModel(id="102", title="Second Post", author_id="1"),
        PostModel(id="103", title="Bob's Post", author_id="2"),
    ]
    db_session.add_all(users + posts)
    await db_session.commit()
    return db_session
```

### 전체 쿼리 통합 테스트

```python
# tests/test_integration.py
@pytest.mark.asyncio
async def test_user_with_posts_integration(seeded_db):
    """통합 테스트: 관련 게시글과 함께 사용자 조회."""
    context = RequestContext(db=seeded_db, current_user=None)
    client = GraphQLTestClient(schema, context_value=context)

    result = client.query("""
        query {
            user(id: "1") {
                username
                posts(first: 10) {
                    title
                }
                postCount
            }
        }
    """)

    assert result.errors is None
    user = result.data["user"]
    assert user["username"] == "alice"
    assert len(user["posts"]) == 2
    assert user["postCount"] == 2


@pytest.mark.asyncio
async def test_dataloader_batching(seeded_db):
    """DataLoader가 쿼리를 배칭하는지 검증합니다 (N+1 없음)."""
    context = RequestContext(db=seeded_db, current_user=None)
    client = GraphQLTestClient(schema, context_value=context)

    # DataLoader 없이는 N+1 문제가 발생하는 쿼리
    result = client.query("""
        query {
            posts(first: 3) {
                title
                author { username }
            }
        }
    """)

    assert result.errors is None
    assert len(result.data["posts"]) == 3
    # 모든 작성자가 해석되어야 함
    for post in result.data["posts"]:
        assert post["author"]["username"] is not None
```

---

## 4. 스냅샷 테스트

### 스키마 스냅샷

```python
def test_schema_snapshot(snapshot):
    """스키마가 예상치 않게 변경되지 않았는지 확인합니다."""
    schema_str = strawberry.printer.print_schema(schema)
    snapshot.assert_match(schema_str, "schema.graphql")
```

### 응답 스냅샷

```python
def test_user_response_snapshot(client, snapshot):
    """사용자 쿼리 응답 형태에 대한 스냅샷 테스트."""
    result = client.query("""
        query { user(id: "1") { id username email bio createdAt } }
    """)
    snapshot.assert_match(
        json.dumps(result.data, indent=2, default=str),
        "user_response.json",
    )
```

### pytest-snapshot 사용법

```bash
# 초기 스냅샷 생성
pytest --snapshot-update

# 테스트 실행 (스냅샷이 다르면 실패)
pytest

# 의도적인 변경 후 검토 및 업데이트
pytest --snapshot-update
```

---

## 5. 모킹과 테스트 픽스처

### 팩토리 픽스처

```python
# tests/factories.py
from dataclasses import dataclass
from datetime import datetime
import itertools

_id_counter = itertools.count(1)


def make_user(**overrides) -> User:
    user_id = str(next(_id_counter))
    defaults = {
        "id": user_id,
        "username": f"user_{user_id}",
        "email": f"user_{user_id}@test.com",
        "bio": None,
        "created_at": datetime(2024, 1, 1),
    }
    defaults.update(overrides)
    return User(**defaults)


def make_post(**overrides) -> Post:
    post_id = str(next(_id_counter))
    defaults = {
        "id": post_id,
        "title": f"Post {post_id}",
        "content": f"Content for post {post_id}",
        "status": PostStatus.DRAFT,
        "author_id": "1",
        "created_at": datetime(2024, 1, 1),
    }
    defaults.update(overrides)
    return Post(**defaults)
```

### 매개변수화된 테스트

```python
@pytest.mark.parametrize("status,expected_count", [
    (None, 3),           # 모든 게시글
    ("PUBLISHED", 2),    # 발행된 것만
    ("DRAFT", 1),        # 초안만
])
def test_posts_filter_by_status(client, mock_context, status, expected_count):
    """상태별 게시글 필터링을 테스트합니다."""
    mock_context.db.posts.find_by_status = AsyncMock(
        return_value=[make_post() for _ in range(expected_count)]
    )

    variables = {}
    if status:
        variables["status"] = status

    result = client.query("""
        query PostsByStatus($status: PostStatus) {
            posts(status: $status) { id title }
        }
    """, variables=variables)

    assert result.errors is None
    assert len(result.data["posts"]) == expected_count
```

---

## 6. 스키마 유효성 검사와 린팅

### GraphQL Inspector

```bash
# 설치
npm install -g @graphql-inspector/cli

# 호환성 깨지는 변경 비교
graphql-inspector diff old-schema.graphql new-schema.graphql

# 출력:
# ✖ Field 'User.email' was removed (BREAKING)
# ✔ Field 'User.phone' was added (NON_BREAKING)
# ⚠ Field 'User.name' was deprecated (DANGEROUS)

# 스키마 유효성 검사
graphql-inspector validate queries/**/*.graphql schema.graphql
```

### 스키마 린팅 규칙

```yaml
# .graphql-inspector.yaml
rules:
  - name: require-description
    severity: warning
    config:
      types: true
      fields: true

  - name: naming-convention
    severity: error
    config:
      types: PascalCase
      fields: camelCase
      enumValues: UPPER_CASE
      inputFields: camelCase

  - name: require-deprecation-reason
    severity: error

  - name: no-unreachable-types
    severity: warning
```

### Rover 스키마 확인

```bash
# 프로덕션 대비 호환성 깨지는 변경 확인
rover subgraph check my-graph@production \
  --name users \
  --schema subgraphs/users/schema.graphql

# 예시 출력:
# 47개 작업에 대해 2개 스키마 변경 비교
# ── FAILURE ──────────────────────────
# BREAKING: User.legacyId 필드 제거
#   영향: GetUserProfile, SearchUsers (12개 클라이언트)
```

---

## 7. 코드 생성

### GraphQL Code Generator

스키마에서 타입이 지정된 클라이언트 코드를 생성합니다:

```bash
npm install -g @graphql-codegen/cli
```

```yaml
# codegen.yml
schema: http://localhost:8000/graphql
documents: src/**/*.graphql
generates:
  src/generated/types.ts:
    plugins:
      - typescript
      - typescript-operations
      - typescript-react-apollo

  src/generated/schema.json:
    plugins:
      - introspection
```

### 생성된 TypeScript 타입

```typescript
// src/generated/types.ts (자동 생성)
export type User = {
  __typename?: 'User';
  id: string;
  username: string;
  email: string;
  bio?: string | null;
  posts: PostConnection;
};

export type GetUserQuery = {
  __typename?: 'Query';
  user?: {
    __typename?: 'User';
    id: string;
    username: string;
    email: string;
  } | null;
};

export type GetUserQueryVariables = {
  id: string;
};

// React 훅 (typescript-react-apollo 플러그인 사용 시)
export function useGetUserQuery(options: QueryHookOptions<GetUserQuery, GetUserQueryVariables>) {
  return useQuery<GetUserQuery, GetUserQueryVariables>(GetUserDocument, options);
}
```

### Python 코드 생성

```bash
# ariadne-codegen으로 Python 타입 생성
pip install ariadne-codegen

# ariadne-codegen.toml
[tool.ariadne-codegen]
schema_path = "schema.graphql"
queries_path = "queries/"
target_package_name = "graphql_client"
```

### 코드 생성의 장점

| 장점 | 설명 |
|------|------|
| 타입 안전성 | 컴파일 타임에 오류 포착 |
| 자동 완성 | IDE가 스키마를 인식 |
| 동기화 | 생성된 코드가 항상 스키마와 일치 |
| 보일러플레이트 감소 | 수동 타입 정의 불필요 |

---

## 8. 대화형 도구

### GraphiQL

내장 GraphQL IDE:

```python
# Strawberry: GraphiQL 활성화
graphql_router = GraphQLRouter(
    schema,
    graphql_ide="graphiql",  # 기본값
)
```

기능:
- 쿼리 자동 완성
- 스키마 문서 브라우저
- 쿼리 히스토리
- 변수 편집기
- 응답 뷰어

### Apollo Sandbox

```python
# GraphiQL 대신 Apollo Sandbox 사용
graphql_router = GraphQLRouter(
    schema,
    graphql_ide="apollo-sandbox",
)
```

기능:
- 모든 GraphiQL 기능 포함
- 작업 컬렉션(Operation Collections)
- 환경 변수
- 사전 실행 스크립트(Pre-flight scripts)
- 응답 비교(Response diffing)

### Postman / Insomnia

두 도구 모두 GraphQL을 지원합니다:

```
1. 새 GraphQL 요청 생성
2. URL을 http://localhost:8000/graphql로 설정
3. 본문 편집기에 쿼리 작성
4. 스키마 인트로스펙션을 통해 자동 완성 작동
5. Variables 탭에서 변수 설정
6. Authorization 헤더 추가
```

### GraphQL Voyager

대화형 스키마 시각화:

```python
# Voyager 라우트 추가
from starlette.responses import HTMLResponse

VOYAGER_HTML = """
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/graphql-voyager/dist/voyager.css" />
</head>
<body>
  <div id="voyager">Loading...</div>
  <script src="https://cdn.jsdelivr.net/npm/graphql-voyager/dist/voyager.standalone.js"></script>
  <script>
    GraphQLVoyager.init(document.getElementById('voyager'), {
      introspection: fetch('/graphql', {
        method: 'post',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: GraphQLVoyager.voyagerIntrospectionQuery,
        }),
      }).then(r => r.json()),
    });
  </script>
</body>
</html>
"""

@app.get("/voyager")
async def voyager():
    return HTMLResponse(VOYAGER_HTML)
```

---

## 9. CI/CD 파이프라인

### GitHub Actions 워크플로우

```yaml
# .github/workflows/graphql-ci.yml
name: GraphQL CI

on:
  pull_request:
    paths:
      - 'src/**'
      - 'schema.graphql'

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4

      - name: Python 설정
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: 의존성 설치
        run: pip install -r requirements.txt -r requirements-test.txt

      - name: 단위 테스트
        run: pytest tests/unit/ -v --cov=src

      - name: 통합 테스트
        run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:test@localhost/test

  schema-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Rover 설치
        run: |
          curl -sSL https://rover.apollo.dev/nix/latest | sh
          echo "$HOME/.rover/bin" >> $GITHUB_PATH

      - name: 스키마 추출
        run: |
          pip install -r requirements.txt
          python -c "
          from schema import schema
          import strawberry
          print(strawberry.printer.print_schema(schema))
          " > current-schema.graphql

      - name: 호환성 깨지는 변경 확인
        run: |
          rover graph check my-graph@production \
            --schema current-schema.graphql || true

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: 스키마 린팅
        run: |
          npx @graphql-inspector/cli validate \
            'src/**/*.graphql' schema.graphql

      - name: Python 린팅
        run: |
          pip install ruff
          ruff check src/ tests/
```

### 스키마 변경 워크플로우

```
1. 개발자가 스키마 수정
2. CI 실행:
   a. 단위 테스트 → 통과/실패
   b. 통합 테스트 → 통과/실패
   c. 스키마 diff → 호환성 깨지는 변경 감지
   d. 스냅샷 테스트 → 응답 형태 검증
   e. 린팅 → 스타일 확인
3. 호환성 깨지는 변경 감지 시:
   a. 머지 차단
   b. 수동 승인 요구
   c. 폐기 기간 요구
4. main 머지 시:
   a. 레지스트리에 스키마 발행
   b. 스테이징 배포
   c. E2E 테스트 실행
5. 릴리스 시:
   a. 프로덕션 배포
   b. API 소비자에게 알림
```

---

## 10. 연습 문제

### 연습 1: 테스트 스위트

블로그 API를 위한 종합 테스트 스위트를 작성하세요:
- 5개 쿼리 테스트 (user, post, list, search, 오류 케이스)
- 3개 뮤테이션 테스트 (create, update, delete + 유효성 검사)
- 2개 권한 부여 테스트 (인증됨 vs. 미인증)
- 1개 DataLoader 통합 테스트

### 연습 2: 스냅샷 테스트

다음에 대해 스냅샷 테스트를 설정하세요:
- 전체 스키마 SDL 출력
- 3개의 다른 쿼리 응답
- 새로운 nullable 필드 추가가 스냅샷을 깨뜨리지 않는지 확인

### 연습 3: 코드 생성

TypeScript React 클라이언트를 위한 GraphQL Code Generator를 설정하세요.

### 연습 4: CI 파이프라인

완전한 GitHub Actions CI 파이프라인을 만드세요.

---

## 11. 참고 자료

### 테스팅
- [Strawberry Testing Documentation](https://strawberry.rocks/docs/general/testing)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)

### 도구
- [GraphiQL](https://github.com/graphql/graphiql)
- [Apollo Sandbox](https://studio.apollographql.com/sandbox)
- [GraphQL Voyager](https://graphql-kit.com/graphql-voyager/)
- [GraphQL Inspector](https://graphql-inspector.com/)

### 코드 생성
- [GraphQL Code Generator](https://the-guild.dev/graphql/codegen)
- [Ariadne Codegen](https://github.com/mirumee/ariadne-codegen)

### CI/CD
- [Rover CI/CD 가이드](https://www.apollographql.com/docs/rover/ci-cd/)
- [GitHub Actions for GraphQL](https://graphql-inspector.com/docs/recipes/github)

---

**License**: CC BY-NC 4.0
