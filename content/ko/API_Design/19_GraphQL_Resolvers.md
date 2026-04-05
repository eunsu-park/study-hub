# 19. GraphQL 리졸버(GraphQL Resolvers)

**이전**: [GraphQL 스키마 설계](./18_GraphQL_Schema_Design.md) | **다음**: [GraphQL 서브스크립션](./20_GraphQL_Subscriptions.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- 리졸버 실행 모델과 GraphQL이 타입 그래프를 순회하는 방식을 설명할 수 있다
- Strawberry를 사용하여 적절한 컨텍스트 및 info 사용과 함께 Python에서 리졸버를 구현할 수 있다
- DataLoader를 사용한 배치 데이터 페칭으로 N+1 문제를 식별하고 해결할 수 있다
- 리졸버 수준의 인증 및 권한 부여 패턴을 적용할 수 있다
- 서비스 레이어와 의존성 주입을 사용하여 유지보수 가능한 리졸버 코드를 구조화할 수 있다
- 리졸버 내에서 필드 수준 오류 처리를 구현할 수 있다

---

## 목차

1. [리졸버 동작 방식](#1-리졸버-동작-방식)
2. [리졸버 구조](#2-리졸버-구조)
3. [Context와 Info 객체](#3-context와-info-객체)
4. [N+1 문제](#4-n1-문제)
5. [DataLoader 패턴](#5-dataloader-패턴)
6. [리졸버 인증과 권한 부여](#6-리졸버-인증과-권한-부여)
7. [리졸버에서의 오류 처리](#7-리졸버에서의-오류-처리)
8. [리졸버 패턴과 아키텍처](#8-리졸버-패턴과-아키텍처)
9. [성능 고려사항](#9-성능-고려사항)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 리졸버 동작 방식

리졸버는 스키마의 단일 필드에 대한 데이터를 채우는 함수입니다. 쿼리가 도착하면 GraphQL 실행 엔진은 깊이 우선, 너비 우선 하이브리드 순회로 각 필드에 대한 리졸버를 호출합니다.

### 실행 모델

```
Query:
{
  user(id: "1") {      # 1. Query.user 리졸버 호출
    username            # 2. User.username 리졸버 호출 (기본)
    posts(first: 3) {   # 3. User.posts 리졸버 호출
      title             # 4. Post.title 리졸버 호출 (기본)
      author {          # 5. Post.author 리졸버 호출
        username        # 6. User.username 리졸버 호출 (기본)
      }
    }
  }
}
```

### 기본 리졸버(Default Resolvers)

속성에 직접 매핑되는 단순 필드의 경우 GraphQL은 **기본 리졸버**(trivial resolver라고도 함)를 사용합니다. 이들은 `obj.field_name` 또는 `obj["field_name"]`을 자동으로 반환합니다.

```python
# You do NOT need to write a resolver for this:
@strawberry.type
class User:
    username: str  # Default resolver returns self.username
    email: str     # Default resolver returns self.email
```

다음이 필요한 필드에만 커스텀 리졸버를 작성합니다:
- 데이터베이스 쿼리
- 계산된 값
- 데이터 변환
- 권한 확인

---

## 2. 리졸버 구조

### Strawberry에서의 기본 리졸버

```python
import strawberry
from typing import Optional


@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: strawberry.ID) -> Optional["User"]:
        """ID로 사용자를 조회합니다."""
        return user_repository.find_by_id(id)

    @strawberry.field
    def users(
        self,
        limit: int = 10,
        offset: int = 0,
    ) -> list["User"]:
        """페이지네이션된 사용자 목록을 조회합니다."""
        return user_repository.find_all(limit=limit, offset=offset)
```

### 객체 타입의 리졸버

```python
@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str

    @strawberry.field
    def posts(
        self,
        info: strawberry.types.Info,
        first: int = 10,
        status: Optional[PostStatus] = None,
    ) -> list["Post"]:
        """사용자의 게시글을 리졸브합니다."""
        filters = {"author_id": self.id}
        if status:
            filters["status"] = status.value
        return post_repository.find_by(filters, limit=first)

    @strawberry.field
    def post_count(self) -> int:
        """계산 필드: 총 게시글 수."""
        return post_repository.count_by_author(self.id)
```

### 리졸버 반환 타입

| 반환값 | GraphQL 효과 |
|--------|-------------|
| 값 | 필드가 해당 값으로 리졸브 |
| `None` | 필드가 `null`로 리졸브 (nullable이어야 함) |
| 예외 | `errors` 배열에 오류 추가; 필드는 `null`로 리졸브 |
| 리스트 | 필드가 리스트로 리졸브 |
| Awaitable | 비동기 리졸버; 실행이 결과를 기다림 |

---

## 3. Context와 Info 객체

### Context 객체

컨텍스트는 단일 요청의 모든 리졸버에서 공유됩니다. 일반적으로 다음을 포함합니다:

```python
from dataclasses import dataclass


@dataclass
class RequestContext:
    db: Database
    current_user: Optional[User]
    request: Request
    dataloaders: "DataLoaders"


async def get_context(request: Request) -> RequestContext:
    """각 GraphQL 요청에 대한 컨텍스트를 빌드합니다."""
    db = get_database()
    token = request.headers.get("Authorization", "").replace("Bearer ", "")
    current_user = await authenticate(token, db) if token else None

    return RequestContext(
        db=db,
        current_user=current_user,
        request=request,
        dataloaders=DataLoaders(db),
    )
```

### 리졸버에서 컨텍스트 사용

```python
@strawberry.type
class Query:
    @strawberry.field
    def me(self, info: strawberry.types.Info) -> Optional["User"]:
        """현재 인증된 사용자를 반환합니다."""
        ctx: RequestContext = info.context
        return ctx.current_user

    @strawberry.field
    def user(self, info: strawberry.types.Info, id: strawberry.ID) -> Optional["User"]:
        ctx: RequestContext = info.context
        return ctx.db.users.find_by_id(id)
```

### Info 객체(The Info Object)

`info` 파라미터는 현재 실행에 대한 메타데이터를 제공합니다:

```python
@strawberry.field
def user(self, info: strawberry.types.Info, id: strawberry.ID) -> Optional["User"]:
    # Access context
    ctx = info.context

    # Check which fields are requested (for optimization)
    selected_fields = [
        selection.name
        for selection in info.selected_fields[0].selections
    ]

    # Only join posts table if posts field is requested
    include_posts = "posts" in selected_fields
    return user_repository.find_by_id(id, include_posts=include_posts)
```

### 룩어헤드 최적화

요청된 필드를 검사하여 데이터베이스 쿼리를 최적화합니다:

```python
def _get_requested_fields(info: strawberry.types.Info) -> set[str]:
    """Extract top-level requested field names."""
    fields = set()
    for selection in info.selected_fields:
        for sub in selection.selections:
            fields.add(sub.name)
    return fields


@strawberry.type
class Query:
    @strawberry.field
    def posts(self, info: strawberry.types.Info, first: int = 10) -> list["Post"]:
        requested = _get_requested_fields(info)
        query = post_repository.query()

        # Only join author if requested
        if "author" in requested:
            query = query.join(User)

        # Only join comments if requested
        if "comments" in requested:
            query = query.join(Comment)

        return query.limit(first).all()
```

---

## 4. N+1 문제

N+1 문제는 GraphQL에서 가장 흔한 성능 함정입니다.

### 발생 원인

```graphql
query {
  posts(first: 10) {     # 1개 쿼리: SELECT * FROM posts LIMIT 10
    title
    author {              # 10개 쿼리: SELECT * FROM users WHERE id = ?
      username            #   (게시글당 하나)
    }
  }
}
```

총: 2개 대신 **11개 쿼리**.

### 발생 메커니즘(Why It Happens)

각 `Post.author` 리졸버가 독립적으로 실행됩니다:

```python
@strawberry.type
class Post:
    author_id: strawberry.ID

    @strawberry.field
    def author(self) -> "User":
        # 게시글마다 한 번씩 실행됨
        return db.query(User).filter(User.id == self.author_id).first()
```

### 규모에서의 문제(The Problem at Scale)

| 게시글 | 배칭 없이 | 배칭으로 |
|--------|----------|---------|
| 10 | 11개 쿼리 | 2개 쿼리 |
| 100 | 101개 쿼리 | 2개 쿼리 |
| 1,000 | 1,001개 쿼리 | 2개 쿼리 |

중첩 쿼리는 더욱 심각합니다:

```graphql
query {
  posts(first: 10) {           # 1 query
    author {                    # 10 queries
      posts(first: 5) {        # 10 queries
        comments(first: 3) {   # 50 queries
          author { username }   # 150 queries
        }
      }
    }
  }
}
# Total: 221 queries!
```

---

## 5. DataLoader 패턴

DataLoader는 데이터베이스 요청을 배칭하고 캐싱하여 N+1 문제를 해결합니다.

### DataLoader 동작 방식

1. **수집**: 단일 실행 틱 동안 DataLoader가 모든 요청된 키를 수집
2. **배치**: 틱 끝에서 모든 키로 배치 함수를 한 번 호출
3. **캐시**: 결과가 요청 기간 동안 캐시됨 (요청별 캐시)

### Strawberry에서의 구현

```python
from strawberry.dataloader import DataLoader


async def load_users(keys: list[str]) -> list[User]:
    """단일 쿼리로 여러 사용자를 로드합니다."""
    users = await db.query(User).filter(User.id.in_(keys)).all()
    # 중요: 키와 같은 순서로 결과 반환
    user_map = {str(u.id): u for u in users}
    return [user_map.get(key) for key in keys]


async def load_posts_by_author(keys: list[str]) -> list[list[Post]]:
    """Load posts grouped by author ID."""
    posts = await db.query(Post).filter(Post.author_id.in_(keys)).all()
    posts_by_author: dict[str, list[Post]] = {}
    for post in posts:
        posts_by_author.setdefault(str(post.author_id), []).append(post)
    return [posts_by_author.get(key, []) for key in keys]


# DataLoader container
@dataclass
class DataLoaders:
    def __init__(self, db: Database):
        self.user_loader = DataLoader(load_fn=load_users)
        self.posts_by_author_loader = DataLoader(load_fn=load_posts_by_author)
```

### 리졸버에서 DataLoader 사용

```python
@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    author_id: strawberry.ID

    @strawberry.field
    async def author(self, info: strawberry.types.Info) -> "User":
        ctx: RequestContext = info.context
        return await ctx.dataloaders.user_loader.load(self.author_id)


@strawberry.type
class User:
    id: strawberry.ID
    username: str

    @strawberry.field
    async def posts(self, info: strawberry.types.Info) -> list["Post"]:
        ctx: RequestContext = info.context
        return await ctx.dataloaders.posts_by_author_loader.load(self.id)
```

### 적용 전후

```
# DataLoader 없이 (N+1)
SELECT * FROM posts LIMIT 10;
SELECT * FROM users WHERE id = 1;
SELECT * FROM users WHERE id = 2;
...
SELECT * FROM users WHERE id = 10;
# 11개 쿼리

# DataLoader 적용 (배칭)
SELECT * FROM posts LIMIT 10;
SELECT * FROM users WHERE id IN (1, 2, 3, ..., 10);
# 2개 쿼리
```

### DataLoader 규칙

| 규칙 | 설명 |
|------|------|
| **요청별 인스턴스** | 각 요청마다 새 DataLoader 인스턴스 생성 |
| **키 순서** | 입력 키와 같은 순서로 결과 반환 |
| **키 개수** | 키당 정확히 하나의 결과 반환 (없는 경우 `None`) |
| **부작용 없음** | 배치 함수는 순수 읽기여야 함 |
| **오류 처리** | 개별 키 실패에 대해 `Error` 객체 반환 |

### 중첩 DataLoader(Nested DataLoaders)

```python
async def load_comment_counts(keys: list[str]) -> list[int]:
    """Batch load comment counts for posts."""
    rows = await db.execute(
        "SELECT post_id, COUNT(*) as cnt FROM comments "
        "WHERE post_id = ANY($1) GROUP BY post_id",
        keys,
    )
    count_map = {str(r["post_id"]): r["cnt"] for r in rows}
    return [count_map.get(key, 0) for key in keys]


@strawberry.type
class Post:
    id: strawberry.ID

    @strawberry.field
    async def comment_count(self, info: strawberry.types.Info) -> int:
        ctx: RequestContext = info.context
        return await ctx.dataloaders.comment_count_loader.load(self.id)
```

---

## 6. 리졸버 인증과 권한 부여

### 컨텍스트를 통한 인증

```python
def require_auth(info: strawberry.types.Info) -> User:
    """인증을 강제하는 헬퍼."""
    ctx: RequestContext = info.context
    if ctx.current_user is None:
        raise PermissionError("Authentication required")
    return ctx.current_user


@strawberry.type
class Query:
    @strawberry.field
    def me(self, info: strawberry.types.Info) -> "User":
        return require_auth(info)
```

### 필드 수준 권한 부여(Field-Level Authorization)

```python
import functools


def require_role(*roles: str):
    """Decorator to enforce role-based access on resolvers."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, info: strawberry.types.Info, *args, **kwargs):
            user = require_auth(info)
            if user.role not in roles:
                raise PermissionError(
                    f"Requires one of: {', '.join(roles)}"
                )
            return func(self, info, *args, **kwargs)
        return wrapper
    return decorator


@strawberry.type
class Query:
    @strawberry.field
    @require_role("ADMIN", "SUPER_ADMIN")
    def admin_dashboard(self, info: strawberry.types.Info) -> "Dashboard":
        return dashboard_service.get_dashboard()
```

### Strawberry 권한

```python
from strawberry.permission import BasePermission
from strawberry.types import Info


class IsAuthenticated(BasePermission):
    message = "사용자가 인증되지 않았습니다"

    def has_permission(self, source, info: Info, **kwargs) -> bool:
        ctx: RequestContext = info.context
        return ctx.current_user is not None


class IsAdmin(BasePermission):
    message = "관리자 접근이 필요합니다"

    def has_permission(self, source, info: Info, **kwargs) -> bool:
        ctx: RequestContext = info.context
        return (
            ctx.current_user is not None
            and ctx.current_user.role == "ADMIN"
        )


@strawberry.type
class Query:
    @strawberry.field(permission_classes=[IsAuthenticated])
    def me(self, info: strawberry.types.Info) -> "User":
        return info.context.current_user

    @strawberry.field(permission_classes=[IsAdmin])
    def all_users(self, info: strawberry.types.Info) -> list["User"]:
        return info.context.db.users.all()
```

### 데이터 수준 권한 부여

뷰어의 권한에 따라 결과를 필터링합니다:

```python
@strawberry.type
class User:
    @strawberry.field
    def email(self, info: strawberry.types.Info) -> Optional[str]:
        """이메일은 본인 또는 관리자에게만 표시됩니다."""
        viewer = info.context.current_user
        if viewer and (viewer.id == self.id or viewer.role == "ADMIN"):
            return self._email
        return None
```

---

## 7. 리졸버에서의 오류 처리

### 최상위 오류 vs. 사용자 오류

| 유형 | 시점 | 방법 |
|------|------|------|
| **GraphQL 오류** | 예상치 못한 실패 (DB 다운, 버그) | 예외로 발생; `errors` 배열에 표시 |
| **사용자 오류** | 예상된 실패 (유효성 검사, 미존재) | 페이로드 `userErrors` 필드에 반환 |

### 예외 기반 오류 처리(Exception-Based Error Handling)

```python
class NotFoundError(Exception):
    def __init__(self, entity: str, id: str):
        self.message = f"{entity} with id '{id}' not found"
        self.extensions = {"code": "NOT_FOUND", "entity": entity, "id": id}


@strawberry.type
class Query:
    @strawberry.field
    def post(self, id: strawberry.ID) -> "Post":
        post = post_repository.find_by_id(id)
        if post is None:
            raise NotFoundError("Post", id)
        return post
```

### 구조화된 오류 페이로드 (뮤테이션에 권장)

```python
@strawberry.type
class UserError:
    field: list[str]
    message: str
    code: str


@strawberry.type
class CreatePostPayload:
    post: Optional["Post"] = None
    user_errors: list[UserError] = strawberry.field(default_factory=list)


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_post(
        self, info: strawberry.types.Info, input: CreatePostInput
    ) -> CreatePostPayload:
        errors = []

        if not input.title.strip():
            errors.append(UserError(
                field=["input", "title"],
                message="제목은 비워둘 수 없습니다",
                code="BLANK",
            ))

        if errors:
            return CreatePostPayload(user_errors=errors)

        post = post_repository.create(
            title=input.title,
            content=input.content,
            author_id=info.context.current_user.id,
        )
        return CreatePostPayload(post=post)
```

---

## 8. 리졸버 패턴과 아키텍처

### 서비스 레이어 패턴

비즈니스 로직을 서비스에 위임하여 리졸버를 얇게 유지합니다:

```python
# services/post_service.py
class PostService:
    def __init__(self, db: Database, current_user: User):
        self.db = db
        self.current_user = current_user

    def create_post(self, input: CreatePostInput) -> CreatePostPayload:
        errors = self._validate(input)
        if errors:
            return CreatePostPayload(user_errors=errors)

        post = Post(
            title=input.title,
            content=input.content,
            author_id=self.current_user.id,
        )
        self.db.posts.save(post)
        return CreatePostPayload(post=post)


# 리졸버: 얇은 래퍼
@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_post(
        self, info: strawberry.types.Info, input: CreatePostInput
    ) -> CreatePostPayload:
        service = PostService(info.context.db, info.context.current_user)
        return service.create_post(input)
```

### 리포지토리 패턴(Repository Pattern)

데이터 접근을 리포지토리 뒤에 추상화합니다:

```python
class UserRepository:
    def __init__(self, db: Database):
        self.db = db

    async def find_by_id(self, id: str) -> Optional[User]:
        return await self.db.query(User).filter(User.id == id).first()

    async def find_by_ids(self, ids: list[str]) -> list[User]:
        return await self.db.query(User).filter(User.id.in_(ids)).all()

    async def find_by_email(self, email: str) -> Optional[User]:
        return await self.db.query(User).filter(User.email == email).first()

    async def save(self, user: User) -> User:
        self.db.add(user)
        await self.db.commit()
        return user
```

### 미들웨어 패턴

횡단 관심사(로깅, 타이밍, 오류 처리)를 적용합니다:

```python
from strawberry.extensions import SchemaExtension
import time
import logging

logger = logging.getLogger("graphql")


class QueryLoggingExtension(SchemaExtension):
    def on_operation(self):
        start = time.monotonic()
        yield
        duration = time.monotonic() - start
        logger.info(
            "GraphQL 작업 완료",
            extra={
                "operation_name": self.execution_context.operation_name,
                "duration_ms": round(duration * 1000, 2),
            },
        )
```

---

## 9. 성능 고려사항

### 리졸버 실행 순서(Resolver Execution Order)

```
Query.posts        →  [Post1, Post2, Post3]
Post1.author       ┐
Post2.author       ├→ DataLoader에 의해 배칭 → SQL 쿼리 1개
Post3.author       ┘
Post1.comments     ┐
Post2.comments     ├→ DataLoader에 의해 배칭 → SQL 쿼리 1개
Post3.comments     ┘
```

### 비동기 리졸버

I/O 바운드 작업에는 비동기 리졸버를 사용합니다:

```python
@strawberry.type
class Query:
    @strawberry.field
    async def user(self, info: strawberry.types.Info, id: strawberry.ID) -> Optional["User"]:
        return await info.context.db.users.find_by_id(id)
```

### 캐싱 전략

| 수준 | 전략 | 구현 |
|------|------|------|
| **요청** | DataLoader 캐시 | DataLoader에 내장 |
| **애플리케이션** | LRU / Redis 캐시 | 인기 쿼리 캐시 |
| **HTTP** | CDN / 리버스 프록시 | 해시별 영속 쿼리 캐시 |

### 데이터베이스 쿼리 최적화(Database Query Optimization)

```python
# Use JOINs based on selected fields
@strawberry.field
async def posts(self, info: strawberry.types.Info, first: int = 10) -> list["Post"]:
    query = select(PostModel).limit(first)

    # Check if author is requested
    requested = get_requested_fields(info)
    if "author" in requested:
        query = query.options(selectinload(PostModel.author))
    if "comments" in requested:
        query = query.options(selectinload(PostModel.comments))

    return await info.context.db.execute(query)
```

---

## 10. 연습 문제

### 연습 1: DataLoader 구현

다음 스키마가 주어졌을 때 N+1 쿼리를 제거하기 위한 DataLoader를 구현하세요:

```graphql
type Query {
  orders(first: Int): [Order!]!
}
type Order {
  id: ID!
  customer: Customer!
  items: [OrderItem!]!
}
type OrderItem {
  product: Product!
  quantity: Int!
}
```

`load_customers`, `load_items_by_order`, `load_products` 배치 함수를 작성하세요.

### 연습 2: 권한 부여 리졸버

다음 조건에서만 이메일을 반환하는 `User.email` 리졸버를 구현하세요:
- 뷰어가 사용자 본인인 경우
- 뷰어가 ADMIN 역할인 경우
- 사용자의 `emailPublic` 설정이 true인 경우

### 연습 3: 오류 처리

포괄적인 오류 처리가 있는 `transferFunds` 뮤테이션을 구현하세요:
- 출발 및 도착 계좌 존재 확인
- 잔액 부족 확인
- 자기 이체 시도 감지
- 각 경우에 대한 구조화된 `UserError` 객체 반환

### 연습 4: 서비스 레이어 리팩토링

"팻 리졸버"를 깔끔한 서비스 레이어로 리팩토링하세요.

---

## 11. 참고 자료

### 라이브러리
- [Strawberry DataLoader 문서](https://strawberry.rocks/docs/guides/dataloaders)
- [graphql-core — Python GraphQL 구현](https://github.com/graphql-python/graphql-core)
- [Facebook DataLoader (원본 JS 구현)](https://github.com/graphql/dataloader)

### 아티클
- "DataLoader — Source Code Walkthrough" by Lee Byron
- "Solving the N+1 Problem for GraphQL through Batching" — Apollo Blog

---

**License**: CC BY-NC 4.0
