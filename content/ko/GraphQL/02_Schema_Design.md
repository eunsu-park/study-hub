# 02. 스키마 설계

**이전**: [GraphQL 기초](./01_GraphQL_Fundamentals.md) | **다음**: [쿼리와 뮤테이션](./03_Queries_and_Mutations.md)

---

스키마(Schema)는 GraphQL 서버와 이를 사용하는 모든 클라이언트 사이의 계약입니다. 잘 설계된 스키마는 직관적이고, 발전 가능하며, 잘못 사용하기 어렵습니다. 잘못 설계된 스키마는 어색한 쿼리, N+1 문제, 고통스러운 마이그레이션으로 이어집니다. 이 레슨에서는 GraphQL 타입 시스템 전반 — 스칼라(Scalar), 객체(Object), 열거형(Enum), 인터페이스(Interface), 유니언(Union), 입력 타입(Input Type) — 과 스키마가 성장해도 유지보수 가능하게 유지하는 설계 원칙을 다룹니다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GraphQL 타입 시스템의 모든 기능 사용: 스칼라, 객체, 열거형, 인터페이스, 유니언, 입력 타입
2. Non-null(`!`)과 List(`[]`) 수식자를 올바르게 적용하고 네 가지 조합을 설명
3. 도메인 특화 데이터(DateTime, JSON, URL)를 위한 커스텀 스칼라 타입 설계
4. 다형성(Polymorphic) 스키마 설계 시 인터페이스와 유니언 중 선택
5. 스키마 설계 모범 사례 적용: 명명 규칙, 기본 nullable, Relay 스타일 커넥션

---

## 목차

1. [타입 시스템 한눈에 보기](#1-타입-시스템-한눈에-보기)
2. [스칼라 타입](#2-스칼라-타입)
3. [객체 타입](#3-객체-타입)
4. [열거형 타입](#4-열거형-타입)
5. [인터페이스 타입](#5-인터페이스-타입)
6. [유니언 타입](#6-유니언-타입)
7. [입력 타입](#7-입력-타입)
8. [Non-null과 List 수식자](#8-non-null과-list-수식자)
9. [스키마 설계 모범 사례](#9-스키마-설계-모범-사례)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 타입 시스템 한눈에 보기

GraphQL의 타입 시스템에는 여섯 가지 명명된 타입 종류가 있습니다:

```
                    ┌─────────────────────────────────────┐
                    │         GraphQL 타입 시스템           │
                    └─────────────────────────────────────┘
                                    │
          ┌─────────────┬───────────┼───────────┬──────────────┐
          │             │           │           │              │
      ┌───▼───┐   ┌────▼────┐ ┌───▼───┐ ┌────▼────┐  ┌─────▼─────┐
      │Scalar │   │ Object  │ │ Enum  │ │Interface│  │   Union   │
      └───────┘   └─────────┘ └───────┘ └─────────┘  └───────────┘
       Int,String   User,Post   Status    Node         SearchResult
       Float,Bool   Comment     Role      Timestamped  Feed
       ID,custom    ...         ...       ...          ...

                           ┌─────▼─────┐
                           │   Input   │
                           └───────────┘
                           CreateUserInput
                           FilterInput
```

또한 다른 타입을 수식하는 두 가지 **래핑 타입(Wrapping Type)**이 있습니다:

- **Non-Null** (`!`): 값이 null이 될 수 없음
- **List** (`[]`): 값이 항목의 배열

## 2. 스칼라 타입

스칼라(Scalar)는 GraphQL 쿼리의 리프 노드(Leaf Node)입니다 — 하위 필드 없이 구체적인 값으로 확인됩니다.

### 2.1 내장 스칼라

```graphql
type Example {
  id: ID!           # 고유 식별자, String으로 직렬화
  name: String!     # UTF-8 문자열
  age: Int          # 32비트 부호 있는 정수 (-2^31 ~ 2^31-1)
  rating: Float     # IEEE 754 배정밀도 부동소수점
  isActive: Boolean # true 또는 false
}
```

### 2.2 커스텀 스칼라

다섯 가지 내장 스칼라만으로는 실제 애플리케이션에 충분하지 않습니다. 커스텀 스칼라는 도메인 특화 의미론과 검증을 추가합니다.

**스키마 정의:**

```graphql
scalar DateTime
scalar JSON
scalar URL
scalar EmailAddress
scalar PositiveInt

type Event {
  id: ID!
  title: String!
  startTime: DateTime!
  endTime: DateTime!
  metadata: JSON
  website: URL
  contactEmail: EmailAddress!
  maxAttendees: PositiveInt!
}
```

**서버 구현 (Apollo Server):**

```javascript
import { GraphQLScalarType, Kind } from 'graphql';

const DateTimeScalar = new GraphQLScalarType({
  name: 'DateTime',
  description: 'ISO 8601 date-time string',

  // 서버 → 클라이언트: 내부 값을 출력으로 직렬화
  serialize(value) {
    if (value instanceof Date) {
      return value.toISOString();
    }
    throw new Error('DateTime must be a Date object');
  },

  // 클라이언트 → 서버: JSON (변수)에서 값 파싱
  parseValue(value) {
    if (typeof value === 'string') {
      const date = new Date(value);
      if (isNaN(date.getTime())) {
        throw new Error(`Invalid DateTime: ${value}`);
      }
      return date;
    }
    throw new Error('DateTime must be a string');
  },

  // 클라이언트 → 서버: 쿼리의 인라인 리터럴에서 값 파싱
  parseLiteral(ast) {
    if (ast.kind === Kind.STRING) {
      const date = new Date(ast.value);
      if (isNaN(date.getTime())) {
        throw new Error(`Invalid DateTime: ${ast.value}`);
      }
      return date;
    }
    throw new Error('DateTime must be a string');
  },
});

const resolvers = {
  DateTime: DateTimeScalar,
  // ... 다른 리졸버들
};
```

**커뮤니티 스칼라 라이브러리:**

공통 스칼라를 직접 구현하는 대신 `graphql-scalars`를 사용하세요:

```bash
npm install graphql-scalars
```

```javascript
import { DateTimeResolver, JSONResolver, URLResolver } from 'graphql-scalars';

const resolvers = {
  DateTime: DateTimeResolver,
  JSON: JSONResolver,
  URL: URLResolver,
  // ... 다른 리졸버들
};
```

이 라이브러리는 40개 이상의 검증된 스칼라 타입을 기본 제공합니다.

## 3. 객체 타입

객체 타입(Object Type)은 GraphQL 스키마의 근간입니다. 필드를 가진 엔티티를 나타냅니다.

### 3.1 기본 객체 타입

```graphql
type User {
  id: ID!
  username: String!
  email: String!
  displayName: String
  bio: String
  avatarUrl: URL
  createdAt: DateTime!
}

type Post {
  id: ID!
  title: String!
  body: String!
  slug: String!
  publishedAt: DateTime
  author: User!          # 관계: Post → User
  comments: [Comment!]!  # 관계: Post → [Comment]
  tags: [Tag!]!
}
```

### 3.2 필드 인수

객체 타입의 모든 필드는 인수를 받을 수 있습니다:

```graphql
type User {
  id: ID!
  username: String!
  posts(
    first: Int = 10       # 기본값
    after: String         # 페이지네이션 커서
    status: PostStatus    # 상태로 필터링
  ): PostConnection!

  # 인수를 가진 계산 필드
  fullName(format: NameFormat = FIRST_LAST): String!
}

enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

enum NameFormat {
  FIRST_LAST
  LAST_FIRST
  USERNAME
}
```

필드 인수는 별도의 쿼리 필드를 만들지 않고도 강력한 필터링과 커스터마이징을 제공합니다.

### 3.3 루트 Query 타입

`Query` 타입은 모든 읽기 작업의 진입점입니다. API의 "목차"라고 생각하면 됩니다:

```graphql
type Query {
  # 단일 엔티티 조회
  user(id: ID!): User
  post(id: ID!): Post
  post_by_slug(slug: String!): Post

  # 컬렉션 쿼리
  users(first: Int = 20, after: String): UserConnection!
  posts(
    first: Int = 20
    after: String
    authorId: ID
    status: PostStatus
    tag: String
  ): PostConnection!

  # 검색
  search(query: String!, types: [SearchType!]): [SearchResult!]!

  # 현재 사용자 (인증 필요)
  me: User
}
```

## 4. 열거형 타입

열거형(Enum)은 허용되는 값의 고정된 집합을 나타냅니다. 자체 문서화가 되며 타입 안전성을 제공합니다.

```graphql
enum Role {
  ADMIN
  MODERATOR
  USER
  GUEST
}

enum OrderStatus {
  PENDING
  CONFIRMED
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}

enum SortOrder {
  ASC
  DESC
}

type User {
  id: ID!
  username: String!
  role: Role!
}

type Query {
  users(
    role: Role
    sortBy: String = "createdAt"
    sortOrder: SortOrder = DESC
  ): [User!]!
}
```

**리졸버 매핑 (데이터베이스 값이 열거형 이름과 다를 때):**

```javascript
const resolvers = {
  // 내부 데이터베이스 값을 GraphQL 열거형 값으로 매핑
  Role: {
    ADMIN: 'admin',
    MODERATOR: 'mod',
    USER: 'user',
    GUEST: 'guest',
  },
  // 리졸버에서 인수 값은 매핑된 값이 됩니다
  Query: {
    users: (_, { role }) => {
      // role은 'admin', 'mod', 'user', 'guest' 중 하나 (매핑된 값)
      return db.users.findAll({ where: role ? { role } : {} });
    },
  },
};
```

## 5. 인터페이스 타입

인터페이스(Interface)는 여러 타입이 구현해야 하는 필드 집합을 정의합니다. 공유 계약을 통해 다양한 타입의 컬렉션을 쿼리할 수 있는 다형성(Polymorphism)을 가능하게 합니다.

### 5.1 인터페이스 정의

```graphql
interface Node {
  id: ID!
}

interface Timestamped {
  createdAt: DateTime!
  updatedAt: DateTime!
}

type User implements Node & Timestamped {
  id: ID!
  createdAt: DateTime!
  updatedAt: DateTime!
  username: String!
  email: String!
}

type Post implements Node & Timestamped {
  id: ID!
  createdAt: DateTime!
  updatedAt: DateTime!
  title: String!
  body: String!
  author: User!
}

type Comment implements Node & Timestamped {
  id: ID!
  createdAt: DateTime!
  updatedAt: DateTime!
  body: String!
  author: User!
}
```

### 5.2 인터페이스 쿼리

필드가 인터페이스 타입을 반환할 때, 공유 필드는 직접 쿼리하고 타입별 필드는 인라인 프래그먼트(Inline Fragment)를 사용합니다:

```graphql
type Query {
  node(id: ID!): Node
  recentActivity(limit: Int = 10): [Timestamped!]!
}
```

```graphql
query {
  node(id: "abc123") {
    id
    ... on User {
      username
      email
    }
    ... on Post {
      title
      body
    }
    ... on Comment {
      body
    }
  }
}
```

### 5.3 인터페이스 리졸버: __resolveType

서버는 객체가 어떤 구체적인 타입인지 알아야 합니다:

```javascript
const resolvers = {
  Node: {
    __resolveType(obj) {
      // 객체의 속성으로 타입 결정
      if (obj.username) return 'User';
      if (obj.title) return 'Post';
      if (obj.body && !obj.title) return 'Comment';
      return null;
    },
  },
};
```

### 5.4 인터페이스를 사용해야 할 때

타입들이 **공통 계약**을 공유하고 균일하게 쿼리하고 싶을 때 인터페이스를 사용합니다:

- 전역 객체 식별을 위한 `Node` 인터페이스 (Relay 패턴)
- 감사(Audit) 필드를 위한 `Timestamped`
- 전문 검색 결과를 위한 `Searchable`
- 타입화된 에러 처리를 위한 `Error` 인터페이스

## 6. 유니언 타입

유니언(Union)은 인터페이스와 유사하지만, 공유 필드가 없습니다. "이 타입들 중 하나"를 나타냅니다.

### 6.1 유니언 정의

```graphql
union SearchResult = User | Post | Comment

union Feed = Post | SharedPost | Ad

type SharedPost {
  id: ID!
  originalPost: Post!
  sharedBy: User!
  comment: String
}

type Ad {
  id: ID!
  title: String!
  imageUrl: URL!
  targetUrl: URL!
  sponsor: String!
}

type Query {
  search(query: String!): [SearchResult!]!
  feed(first: Int = 20, after: String): [Feed!]!
}
```

### 6.2 유니언 쿼리

유니언 타입은 공유 필드가 보장되지 않으므로, 모든 필드에 인라인 프래그먼트를 사용해야 합니다:

```graphql
query SearchQuery {
  search(query: "graphql") {
    ... on User {
      id
      username
      avatarUrl
    }
    ... on Post {
      id
      title
      publishedAt
    }
    ... on Comment {
      id
      body
      author { username }
    }
  }
}
```

`__typename` 메타 필드를 사용해 구체적인 타입을 확인할 수도 있습니다:

```graphql
query {
  search(query: "graphql") {
    __typename
    ... on User { username }
    ... on Post { title }
    ... on Comment { body }
  }
}
```

### 6.3 인터페이스 vs 유니언: 선택 가이드

| 기준 | 인터페이스 | 유니언 |
|------|------------|--------|
| 공유 필드가 있음? | 예, 강제됨 | 아니오 |
| 타입이 구현함? | 예 (`implements`) | 아니오 |
| 나중에 필드 추가 가능? | 모든 구현체 업데이트 필요 | 공유 필드 없음 |
| 사용 사례 | 공통 계약 (Node, Error) | 이질적인 결과 (검색, 피드) |
| 다중 상속? | 예 (`implements A & B`) | 해당 없음 |

**경험칙**: 타입들이 의미 있는 필드를 공유하면 인터페이스를 사용하고, 같은 컨텍스트에 나타나는 근본적으로 다른 타입들이라면 유니언을 사용하세요.

## 7. 입력 타입

입력 타입(Input Type)은 뮤테이션과 복잡한 쿼리의 인수 형태를 정의합니다. 객체 타입의 "쓰기 측" 대응입니다.

### 7.1 입력 타입이 존재하는 이유

객체 타입은 다른 객체 타입을 반환하는 필드(순환 참조, 계산 필드 등)를 가질 수 있어 입력으로 부적합합니다. 입력 타입은 엄격히 트리 구조입니다 — 순환 없음, 계산 필드 없음.

```graphql
# ❌ 객체 타입을 입력으로 사용할 수 없음
type Mutation {
  createUser(user: User!): User!  # 오류: User는 객체 타입
}

# ✅ 인수에는 입력 타입 사용
input CreateUserInput {
  username: String!
  email: String!
  password: String!
  role: Role = USER
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}
```

### 7.2 입력 타입 패턴

```graphql
# 생성 입력: 모든 필수 필드
input CreatePostInput {
  title: String!
  body: String!
  tags: [String!]
  publishNow: Boolean = false
}

# 수정 입력: 모든 필드 선택 사항 (부분 업데이트)
input UpdatePostInput {
  title: String
  body: String
  tags: [String!]
}

# 필터 입력: 유연한 쿼리
input PostFilterInput {
  authorId: ID
  status: PostStatus
  tag: String
  createdAfter: DateTime
  createdBefore: DateTime
  search: String
}

# 페이지네이션 입력
input PaginationInput {
  first: Int = 20
  after: String
}

type Query {
  posts(filter: PostFilterInput, pagination: PaginationInput): PostConnection!
}
```

### 7.3 페이로드 패턴

뮤테이션은 원시 엔티티가 아닌 전용 페이로드(Payload) 타입을 반환해야 합니다. 이 패턴을 사용하면 엔티티, 오류, 메타데이터를 함께 반환할 수 있습니다:

```graphql
type CreateUserPayload {
  user: User
  errors: [UserError!]!
}

type UserError {
  field: String!
  message: String!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}
```

```json
// 성공 응답
{
  "data": {
    "createUser": {
      "user": { "id": "1", "username": "alice" },
      "errors": []
    }
  }
}

// 유효성 검사 오류 응답
{
  "data": {
    "createUser": {
      "user": null,
      "errors": [
        { "field": "email", "message": "Email already in use" }
      ]
    }
  }
}
```

## 8. Non-null과 List 수식자

`!`(Non-null)과 `[]`(List) 수식자는 null 가능 여부와 카디널리티(Cardinality)를 제어합니다. 스키마 설계에서 네 가지 조합을 이해하는 것은 매우 중요합니다.

### 8.1 네 가지 조합

```graphql
type Example {
  a: String         # Nullable 문자열: null 가능
  b: String!        # Non-null 문자열: 절대 null 불가
  c: [String]       # Nullable 문자열의 Nullable 리스트: null, [], ["a", null]
  d: [String]!      # Nullable 문자열의 Non-null 리스트: [], ["a", null]
  e: [String!]      # Non-null 문자열의 Nullable 리스트: null, [], ["a", "b"]
  f: [String!]!     # Non-null 문자열의 Non-null 리스트: [], ["a", "b"]
}
```

**상세 분류:**

| 타입 | `null` 가능? | `null` 항목 포함 가능? | 유효한 값 |
|------|-------------|----------------------|----------|
| `[String]` | 예 | 예 | `null`, `[]`, `["a"]`, `["a", null]` |
| `[String]!` | 아니오 | 예 | `[]`, `["a"]`, `["a", null]` |
| `[String!]` | 예 | 아니오 | `null`, `[]`, `["a", "b"]` |
| `[String!]!` | 아니오 | 아니오 | `[]`, `["a", "b"]` |

### 8.2 Nullable 모범 사례

**기본적으로 Nullable.** 이것이 GraphQL 명세의 권장 사항이며 그만한 이유가 있습니다.

필드를 nullable로 만드는 이유:

1. **오류 복원력**: 하나의 필드에서 리졸버가 실패하면 해당 필드만 `null`이 되고 — 나머지 응답은 여전히 유효합니다. Non-null 필드에서 오류가 발생하면 가장 가까운 nullable 부모로 null이 전파되어 전체 응답이 파괴될 수 있습니다.

2. **진화 가능성**: Non-null 필드를 nullable로 변경하면 해당 필드의 존재에 의존하는 클라이언트에 호환성 파괴(Breaking Change)가 됩니다.

```graphql
# Null 전파 문제
type Query {
  user(id: ID!): User!   # Non-null
}

type User {
  name: String!           # Non-null
  avatar: String!         # Non-null — 이 리졸버가 실패하면...
  posts: [Post!]!         # Non-null
}

# avatar 리졸버가 오류를 발생시키면:
# 1. avatar → null (하지만 String!이라 null 불가)
# 2. User 객체 → null (하지만 Query.user가 User!라 null 불가)
# 3. 전체 data → null
# 결과: { "data": null, "errors": [...] }
```

**Non-null을 사용해야 할 때:**

- `id: ID!` — 엔티티는 항상 ID를 가짐
- 진정으로 필수인 입력 필드
- 명확한 기본값이 있는 Boolean 필드
- 신뢰할 수 있는 소스를 기반으로 하는 필드 (NOT NULL 데이터베이스 컬럼)

## 9. 스키마 설계 모범 사례

### 9.1 명명 규칙

```graphql
# 타입: PascalCase
type UserProfile { ... }

# 필드: camelCase
type User {
  firstName: String!
  lastName: String!
  createdAt: DateTime!
}

# 열거형 값: SCREAMING_SNAKE_CASE
enum OrderStatus {
  IN_PROGRESS
  COMPLETED
  CANCELLED
}

# 입력 타입: PascalCase + Input 접미사
input CreateUserInput { ... }
input UpdateUserInput { ... }
input UserFilterInput { ... }

# 뮤테이션: 동사 + 명사
type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
  updateUser(input: UpdateUserInput!): UpdateUserPayload!
  deleteUser(id: ID!): DeleteUserPayload!
  publishPost(id: ID!): PublishPostPayload!  # setPostPublished가 아님
}
```

### 9.2 설명 추가

SDL은 인트로스펙션과 문서에 표시되는 문자열 설명을 지원합니다:

```graphql
"""
플랫폼에 등록된 사용자.
사용자는 게시글을 작성하고, 댓글을 남기고, 다른 사용자를 팔로우할 수 있습니다.
"""
type User {
  "전역 고유 식별자"
  id: ID!

  "고유 로그인 이름 (3-30자, 영숫자 + 밑줄)"
  username: String!

  "UI에 표시되는 이름. null이면 username으로 대체."
  displayName: String

  "팔로워 수. 캐시되어 약간 오래된 값일 수 있음."
  followerCount: Int!
}
```

### 9.3 페이지네이션: Relay 스타일 커넥션

페이지네이션이 적용된 목록에는 Relay 커넥션 명세가 업계 표준입니다:

```graphql
type Query {
  posts(first: Int, after: String, last: Int, before: String): PostConnection!
}

type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PostEdge {
  node: Post!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
```

```graphql
# 사용 예: 커서 이후 처음 10개 게시글 조회
query {
  posts(first: 10, after: "cursor_abc") {
    edges {
      cursor
      node {
        title
        author { username }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
    totalCount
  }
}
```

### 9.4 엔드포인트가 아닌 그래프로 생각하기

REST는 리소스와 엔드포인트 중심으로 생각합니다. GraphQL은 노드(Node)와 엣지(Edge) 중심으로 생각합니다.

```graphql
# ❌ REST 스타일 사고 (평면적, 단절됨)
type Query {
  user(id: ID!): User
  userPosts(userId: ID!): [Post!]!
  userFollowers(userId: ID!): [User!]!
  postComments(postId: ID!): [Comment!]!
}

# ✅ 그래프 사고 (연결됨)
type Query {
  user(id: ID!): User
  post(id: ID!): Post
}

type User {
  id: ID!
  posts(first: Int = 10): PostConnection!
  followers(first: Int = 10): UserConnection!
  following(first: Int = 10): UserConnection!
}

type Post {
  id: ID!
  author: User!
  comments(first: Int = 10): CommentConnection!
}
```

### 9.5 스키마 진화

스키마는 변합니다. 안전하게 변경하는 방법은 다음과 같습니다:

```graphql
# ✅ 안전한 변경 (호환성 유지)
# - 새로운 nullable 필드 추가
# - 새로운 타입 추가
# - 새로운 열거형 값 추가 (목록 끝에)
# - 새로운 쿼리/뮤테이션 필드 추가
# - 필드 deprecated 처리

type User {
  id: ID!
  name: String!
  email: String! @deprecated(reason: "emailAddress를 사용하세요")
  emailAddress: String!   # 새 필드
  phone: String           # 새 nullable 필드
}

# ❌ 호환성 파괴 변경
# - 필드 삭제
# - nullable 필드를 non-null로 변경
# - 열거형 값 삭제
# - 필드 타입 변경
# - 기존 필드에 필수 인수 추가
```

---

## 10. 연습 문제

### 연습 1: Null 가능성 분석 (초급)

각 필드에 대해 주어진 값이 유효한지 아니면 오류를 발생시킬지 판단하세요:

```graphql
type Article {
  id: ID!
  title: String!
  subtitle: String
  tags: [String!]!
  relatedArticles: [Article]
}
```

1. `id: null`
2. `subtitle: null`
3. `tags: null`
4. `tags: []`
5. `tags: ["graphql", null]`
6. `tags: ["graphql", "api"]`
7. `relatedArticles: null`
8. `relatedArticles: [null]`

### 연습 2: 인터페이스 vs 유니언 (중급)

알림 시스템에는 세 가지 알림 타입이 있습니다:

- **FollowNotification**: 누군가 당신을 팔로우함 (팔로워 사용자 포함)
- **CommentNotification**: 누군가 당신의 게시글에 댓글을 남김 (댓글과 게시글 포함)
- **MentionNotification**: 누군가 당신을 언급함 (언급한 사용자와 문맥 텍스트 포함)

모든 알림이 공유하는 필드: id, createdAt, isRead, recipient

(a) 인터페이스를 사용해서, (b) 유니언을 사용해서 이를 설계하세요. 어떤 접근이 더 나은지, 그 이유는 무엇인지 설명하세요.

### 연습 3: 스키마 설계 (중급)

다음 요소를 포함하는 이커머스 플랫폼용 GraphQL 스키마를 설계하세요:

- 상품: 이름, 가격, 설명, 이미지, 카테고리, 재고 수량
- 카테고리: 중첩 가능 (예: 전자제품 > 휴대폰 > 스마트폰)
- 장바구니: 항목 (상품 + 수량)
- 주문: 항목, 배송 주소, 상태 추적
- 사용자: 주소, 주문 내역, 위시리스트

엔티티 생성/수정을 위한 적절한 입력 타입을 포함하세요. 페이지네이션 목록에는 Relay 스타일 커넥션을 사용하세요.

### 연습 4: 커스텀 스칼라 (고급)

다음 기능을 하는 `Currency` 커스텀 스칼라를 구현하세요:

- `"USD 29.99"` 또는 `"EUR 15.00"` 형식의 값을 허용
- 통화 코드를 화이트리스트(USD, EUR, GBP, JPY, KRW)로 검증
- 내부적으로 `{ code: string, amount: number }`로 저장
- 문자열 형식으로 다시 직렬화

JavaScript로 `GraphQLScalarType` 구현을 작성하세요.

### 연습 5: 스키마 리뷰 (고급)

다음 스키마를 검토하고 모든 설계 문제를 식별하세요. 개선안을 제시하세요.

```graphql
type Query {
  getUser(userId: String!): User!
  getAllUsers: [User]
  getUserPosts(userId: String!): [Post]
  getPost(postId: String!): Post
  search(q: String!): [SearchResult]
}

type User {
  userId: String!
  name: String!
  email: String!
  age: String
  role: String
  Posts: [Post]
}

type Post {
  postId: String!
  Title: String!
  content: String!
  date: String!
  user_id: String!
  comments: [Comment]!
}

type Comment {
  id: Int!
  text: String!
  userId: String!
}

union SearchResult = User | Post

type Mutation {
  createUser(name: String!, email: String!, age: Int, role: String): User!
  updateUser(userId: String!, name: String, email: String, age: Int, role: String): User
  createPost(userId: String!, title: String!, content: String!): Post!
}
```

---

## 11. 참고 자료

- GraphQL 타입 시스템 명세 - https://spec.graphql.org/October2021/#sec-Type-System
- Relay 커넥션 명세 - https://relay.dev/graphql/connections.htm
- GraphQL Scalars 라이브러리 - https://the-guild.dev/graphql/scalars
- Principled GraphQL: 스키마 설계 - https://principledgraphql.com/agility#4-abstract-demand-oriented-schema
- Marc-Andre Giroux, "Production Ready GraphQL" (2020)
- Lee Byron, "Designing a GraphQL Schema" (GraphQL Summit 2018)

---

**이전**: [GraphQL 기초](./01_GraphQL_Fundamentals.md) | **다음**: [쿼리와 뮤테이션](./03_Queries_and_Mutations.md)
