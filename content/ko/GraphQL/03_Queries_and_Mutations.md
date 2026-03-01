# 03. 쿼리와 뮤테이션

**이전**: [스키마 설계](./02_Schema_Design.md) | **다음**: [리졸버](./04_Resolvers.md)

---

레슨 01에서는 쿼리(Query)와 뮤테이션(Mutation)을 개념적으로 소개했습니다. 이 레슨에서는 클라이언트 측 쿼리 언어를 깊이 파고듭니다: 쿼리 구조화 방법, 프래그먼트(Fragment)로 필드 선택 재사용, 별칭(Alias)으로 필드 이름 변경, 디렉티브(Directive)로 실행 제어, 그리고 오류를 우아하게 처리하는 뮤테이션 설계까지 다룹니다. 이러한 패턴을 마스터하는 것은 효율적이고 유지보수 가능한 GraphQL 클라이언트를 구축하는 데 필수적입니다.

**난이도**: ⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 중첩된 선택 집합(Selection Set), 인수, 변수를 사용해 쿼리 작성
2. 프래그먼트(명명 프래그먼트 및 인라인 프래그먼트)를 사용해 쿼리의 중복 제거
3. 별칭(Alias)을 사용해 필드 이름을 변경하고 동일 필드를 다른 인수로 쿼리
4. 내장 디렉티브(`@include`, `@skip`, `@deprecated`)를 사용해 쿼리 동작 제어
5. 입력 타입, 페이로드 타입, 구조화된 오류 처리를 갖춘 뮤테이션 오퍼레이션 설계

---

## 목차

1. [쿼리 구조](#1-쿼리-구조)
2. [인수와 변수](#2-인수와-변수)
3. [프래그먼트](#3-프래그먼트)
4. [인라인 프래그먼트](#4-인라인-프래그먼트)
5. [별칭](#5-별칭)
6. [디렉티브](#6-디렉티브)
7. [뮤테이션 심화](#7-뮤테이션-심화)
8. [오류 처리 패턴](#8-오류-처리-패턴)
9. [쿼리 구성 팁](#9-쿼리-구성-팁)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 쿼리 구조

GraphQL 쿼리는 **선택 집합(Selection Set)**의 트리입니다. 각 선택 집합은 주어진 타입에서 어떤 필드를 가져올지 지정합니다.

### 1.1 기본 쿼리 구조

```graphql
# 오퍼레이션 타입    오퍼레이션 이름 (선택 사항이지만 권장)
query                GetUserProfile {
  # 인수를 가진 루트 필드
  user(id: "1") {
    # 선택 집합: 가져올 필드들
    id
    username
    email
    # 중첩된 선택 집합 (관계 탐색)
    posts {
      id
      title
      publishedAt
      # 더 깊은 중첩
      comments {
        id
        body
        author {
          username
        }
      }
    }
  }
}
```

핵심 규칙:

- **스칼라 필드** (String, Int 등)는 리프 노드 — 하위 선택을 가질 수 없음
- **객체 필드** (User, Post 등)는 반드시 하위 선택 집합을 가져야 함
- 중첩 깊이는 무제한 (단, 깊이 제한에 대해서는 레슨 14 참조)

### 1.2 여러 루트 필드

하나의 쿼리에서 여러 루트 필드를 요청할 수 있습니다. 이들은 병렬로 실행됩니다:

```graphql
query DashboardData {
  me {
    username
    notificationCount
  }
  trending: posts(sortBy: TRENDING, first: 5) {
    title
    author { username }
  }
  recentPosts: posts(sortBy: RECENT, first: 10) {
    title
    publishedAt
  }
}
```

이것이 GraphQL의 강점 중 하나입니다: 클라이언트가 하나의 요청으로 하나의 뷰에 필요한 정확한 데이터를 구성할 수 있습니다.

### 1.3 축약 문법

쿼리에 이름도 변수도 없다면 `query` 키워드를 생략할 수 있습니다:

```graphql
# 축약형 (익명 쿼리)
{
  user(id: "1") {
    name
  }
}

# 이것과 동일:
query {
  user(id: "1") {
    name
  }
}
```

축약형은 GraphiQL에서 탐색할 때는 괜찮지만, 애플리케이션 코드에서는 피해야 합니다. 항상 오퍼레이션에 이름을 붙이세요 — 디버깅, 로깅, 서버 측 메트릭에 도움이 됩니다.

## 2. 인수와 변수

### 2.1 인라인 인수

쿼리에 인수를 직접 하드코딩할 수 있습니다:

```graphql
query {
  user(id: "1") { name }
  posts(first: 5, status: PUBLISHED) { title }
}
```

빠른 탐색에는 유용하지만, 값이 사용자 입력, 상태, 라우트 파라미터에서 오는 실제 애플리케이션에서는 비실용적입니다.

### 2.2 변수

변수(Variable)는 쿼리 구조와 동적 값을 분리합니다:

```graphql
# 변수 선언이 포함된 쿼리 정의
query GetUser($userId: ID!, $postLimit: Int = 10) {
  user(id: $userId) {
    name
    email
    posts(first: $postLimit) {
      title
    }
  }
}
```

```json
// 변수 (별도 JSON 객체로 전송)
{
  "userId": "1",
  "postLimit": 5
}
```

**변수 규칙:**

- `$` 접두사와 타입으로 선언: `$userId: ID!`
- 기본값을 가질 수 있음: `$postLimit: Int = 10`
- 스칼라, 열거형, 입력 타입만 가능 (객체 타입 불가)
- 인수에서 이름으로 참조: `id: $userId`

### 2.3 변수가 중요한 이유

```javascript
// ❌ 문자열 보간 (위험, 캐싱 불가)
const query = `
  query {
    user(id: "${userId}") { name }
  }
`;

// ✅ 변수 (안전, 캐시 가능, 재사용 가능)
const query = `
  query GetUser($userId: ID!) {
    user(id: $userId) { name }
  }
`;
const variables = { userId };
```

변수를 사용하면:

1. **인젝션 공격 방지** — 변수 값은 타입 검사를 거치며 쿼리 문자열에 보간되지 않음
2. **쿼리 캐싱 활성화** — 동일한 쿼리 문자열을 다른 변수로 재사용 가능
3. **Persisted Queries 지원** — 서버가 쿼리 문자열을 저장하고 클라이언트는 해시 + 변수만 전송

### 2.4 변수 타입과 Null 가능성

변수의 타입은 사용되는 인수와 일치해야 하지만, 약간의 유연성이 있습니다:

```graphql
# 스키마
type Query {
  user(id: ID!): User            # id는 non-null
  posts(status: PostStatus): [Post!]!  # status는 nullable
}

# 쿼리: 변수 타입은 일치하거나 더 구체적이어야 함
query($userId: ID!, $status: PostStatus) {
  user(id: $userId) { name }      # ID! 가 ID!와 일치  ✅
  posts(status: $status) { title } # PostStatus가 PostStatus와 일치  ✅
}
```

Non-null 변수(`$x: String!`)는 nullable 인수(`arg: String`)가 예상되는 위치에 사용할 수 있지만, 그 반대는 불가합니다.

## 3. 프래그먼트

프래그먼트(Fragment)는 재사용 가능한 필드 선택 단위입니다. 동일한 필드가 여러 곳에서 필요할 때 중복 문제를 해결합니다.

### 3.1 프래그먼트 정의 및 사용

```graphql
# 프래그먼트 정의
fragment UserBasicInfo on User {
  id
  username
  displayName
  avatarUrl
}

fragment PostSummary on Post {
  id
  title
  publishedAt
  author {
    ...UserBasicInfo
  }
}

# 스프레드 연산자(...)로 프래그먼트 사용
query FeedPage {
  trending: posts(sortBy: TRENDING, first: 5) {
    ...PostSummary
    commentCount
  }
  recent: posts(sortBy: RECENT, first: 10) {
    ...PostSummary
    tags { name }
  }
  me {
    ...UserBasicInfo
    email
    notificationCount
  }
}
```

프래그먼트가 없으면 `UserBasicInfo`가 세 곳에 중복됩니다. 프래그먼트는 쿼리를 DRY(Don't Repeat Yourself)하게 유지하고 유지보수를 쉽게 만듭니다.

### 3.2 프래그먼트 합성

프래그먼트는 다른 프래그먼트를 참조하여 합성 트리를 형성할 수 있습니다:

```graphql
fragment CommentInfo on Comment {
  id
  body
  createdAt
  author {
    ...UserBasicInfo
  }
}

fragment PostDetail on Post {
  ...PostSummary
  body
  tags { name }
  comments(first: 20) {
    ...CommentInfo
  }
}

query PostPage($postId: ID!) {
  post(id: $postId) {
    ...PostDetail
  }
}
```

계층 구조가 만들어집니다: `PostDetail`이 `PostSummary`를 포함하고, `PostSummary`가 `UserBasicInfo`를 포함합니다. 각 프래그먼트는 한 번 정의되어 모든 곳에서 재사용됩니다.

### 3.3 컴포넌트에 동반된 프래그먼트 (클라이언트 패턴)

컴포넌트 기반 프론트엔드(React, Vue)에서는 프래그먼트를 사용하는 컴포넌트와 함께 배치합니다:

```javascript
// UserAvatar.jsx
import { gql } from '@apollo/client';

// 이 컴포넌트는 자신이 필요한 데이터를 정확히 선언합니다
export const USER_AVATAR_FRAGMENT = gql`
  fragment UserAvatar on User {
    id
    username
    avatarUrl
  }
`;

export function UserAvatar({ user }) {
  return <img src={user.avatarUrl} alt={user.username} />;
}

// PostCard.jsx
import { gql } from '@apollo/client';
import { USER_AVATAR_FRAGMENT } from './UserAvatar';

export const POST_CARD_FRAGMENT = gql`
  fragment PostCard on Post {
    id
    title
    publishedAt
    author {
      ...UserAvatar
    }
  }
  ${USER_AVATAR_FRAGMENT}
`;

// FeedPage.jsx — 프래그먼트들을 쿼리로 합성
import { gql, useQuery } from '@apollo/client';
import { POST_CARD_FRAGMENT } from './PostCard';

const FEED_QUERY = gql`
  query FeedPage($first: Int!) {
    posts(first: $first) {
      ...PostCard
    }
  }
  ${POST_CARD_FRAGMENT}
`;
```

이 패턴은 각 컴포넌트가 자신의 데이터 요구사항을 선언하고, 페이지 쿼리가 이를 모두 합성하도록 보장합니다. 컴포넌트의 데이터 요구사항이 변경되면 해당 프래그먼트만 업데이트하면 됩니다.

## 4. 인라인 프래그먼트

인라인 프래그먼트(Inline Fragment)는 두 가지 상황에서 사용됩니다: 다형성 타입(인터페이스/유니언) 쿼리와 필드 그룹에 디렉티브를 적용할 때입니다.

### 4.1 타입 조건

필드가 인터페이스 또는 유니언 타입을 반환할 때, 인라인 프래그먼트로 타입별 필드를 선택합니다:

```graphql
query SearchResults($query: String!) {
  search(query: $query) {
    __typename
    ... on User {
      id
      username
      avatarUrl
    }
    ... on Post {
      id
      title
      publishedAt
      author { username }
    }
    ... on Comment {
      id
      body
      post { title }
    }
  }
}
```

`__typename` 메타 필드는 구체적인 타입 이름(`"User"`, `"Post"`, `"Comment"`)을 반환하며, 클라이언트 측 타입 구별에 유용합니다.

### 4.2 명명 프래그먼트와 인라인 프래그먼트 혼합

```graphql
fragment SearchUser on User {
  id
  username
  avatarUrl
  followerCount
}

query Search($query: String!) {
  search(query: $query) {
    ... on User {
      ...SearchUser
    }
    ... on Post {
      id
      title
      body
    }
  }
}
```

### 4.3 타입 조건 없는 인라인 프래그먼트

인라인 프래그먼트는 디렉티브를 적용하기 위해 필드를 그룹화하는 데도 사용할 수 있습니다:

```graphql
query Profile($userId: ID!, $includeStats: Boolean!) {
  user(id: $userId) {
    username
    bio
    ... @include(if: $includeStats) {
      followerCount
      followingCount
      postCount
    }
  }
}
```

이 패턴은 명명 프래그먼트를 만들지 않고도 필드 그룹을 조건부로 포함합니다.

## 5. 별칭

별칭(Alias)은 응답에서 필드 이름을 변경할 수 있게 합니다. 동일한 필드를 다른 인수로 여러 번 쿼리할 때 필수적입니다.

### 5.1 기본 별칭

```graphql
query {
  # 별칭 없이는 충돌이 발생합니다:
  # 동일한 선택 집합에 두 개의 "user" 필드
  alice: user(id: "1") {
    name
    email
  }
  bob: user(id: "2") {
    name
    email
  }
}
```

응답:

```json
{
  "data": {
    "alice": { "name": "Alice", "email": "alice@example.com" },
    "bob": { "name": "Bob", "email": "bob@example.com" }
  }
}
```

### 5.2 같은 필드, 다른 인수에 별칭 사용

```graphql
query PostsByCategory {
  techPosts: posts(category: "technology", first: 5) {
    title
    publishedAt
  }
  sciencePosts: posts(category: "science", first: 5) {
    title
    publishedAt
  }
  sportsPosts: posts(category: "sports", first: 5) {
    title
    publishedAt
  }
}
```

### 5.3 클라이언트 친화적인 이름을 위한 별칭

스키마 필드 이름이 클라이언트가 기대하는 것과 다를 때:

```graphql
query {
  user(id: "1") {
    userId: id
    userName: username
    profilePic: avatarUrl
  }
}
```

기존 클라이언트 데이터 모델이나 컴포넌트 props에 GraphQL 데이터를 맞출 때 유용합니다.

## 6. 디렉티브

디렉티브(Directive)는 필드나 프래그먼트의 실행 방식을 수정합니다. GraphQL은 세 가지 내장 디렉티브를 지정하며, 서버는 커스텀 디렉티브를 정의할 수 있습니다.

### 6.1 @include

조건이 `true`일 때만 필드를 포함합니다:

```graphql
query GetUser($userId: ID!, $withPosts: Boolean!) {
  user(id: $userId) {
    name
    email
    posts @include(if: $withPosts) {
      title
    }
  }
}
```

```json
// 변수
{ "userId": "1", "withPosts": true }   // → posts 포함
{ "userId": "1", "withPosts": false }  // → posts 제외
```

### 6.2 @skip

`@include`의 반대 — 조건이 `true`이면 필드를 건너뜁니다:

```graphql
query GetUser($userId: ID!, $skipEmail: Boolean!) {
  user(id: $userId) {
    name
    email @skip(if: $skipEmail)
  }
}
```

`@include(if: $x)`와 `@skip(if: $x)`는 논리적으로 반대입니다. 더 자연스럽게 읽히는 것을 사용하세요.

### 6.3 @deprecated (스키마 디렉티브)

`@deprecated`는 쿼리 디렉티브가 아닌 스키마 디렉티브입니다. 스키마에서 필드를 deprecated로 표시합니다:

```graphql
type User {
  id: ID!
  name: String!

  # 마이그레이션 안내가 포함된 deprecated 필드
  email: String! @deprecated(reason: "'emailAddress'를 사용하세요. v3에서 제거됩니다.")
  emailAddress: String!

  # Deprecated 열거형 값
  role: Role!
}

enum Role {
  ADMIN
  USER
  MODERATOR
  SUPER_ADMIN @deprecated(reason: "상승된 권한을 가진 ADMIN을 사용하세요")
}
```

Deprecated 필드는 계속 작동하지만:
- GraphiQL/Apollo Explorer에서 경고 표시
- 문서에서 취소선으로 표시
- 인트로스펙션을 통해 감지 가능 (`isDeprecated` 필드)

### 6.4 커스텀 디렉티브 (미리보기)

서버는 횡단 관심사(Cross-cutting Concern)를 위한 커스텀 디렉티브를 정의할 수 있습니다:

```graphql
# 스키마 정의
directive @auth(requires: Role!) on FIELD_DEFINITION
directive @cacheControl(maxAge: Int!) on FIELD_DEFINITION
directive @rateLimit(max: Int!, window: String!) on FIELD_DEFINITION

type Query {
  publicPosts: [Post!]! @cacheControl(maxAge: 300)

  me: User! @auth(requires: USER)

  adminStats: Stats! @auth(requires: ADMIN) @rateLimit(max: 10, window: "1m")
}
```

커스텀 디렉티브는 레슨 07 (인증)과 레슨 14 (성능)에서 더 자세히 다룹니다.

## 7. 뮤테이션 심화

뮤테이션(Mutation)은 GraphQL의 쓰기 작업입니다. 구문적으로 쿼리와 유사하지만, 고유한 의미론과 설계 패턴을 가집니다.

### 7.1 뮤테이션 구조

```graphql
mutation CreatePost($input: CreatePostInput!) {
  createPost(input: $input) {
    post {
      id
      title
      slug
      publishedAt
    }
    errors {
      field
      message
    }
  }
}
```

```json
{
  "input": {
    "title": "GraphQL Best Practices",
    "body": "Here are some tips for designing GraphQL APIs...",
    "tags": ["graphql", "api-design"],
    "publishNow": true
  }
}
```

### 7.2 순차 실행

병렬로 실행될 수 있는 쿼리 필드와 달리, 뮤테이션 필드는 등장하는 순서대로 **순차적으로** 실행됩니다:

```graphql
mutation {
  # 1단계: 먼저 실행
  createUser(input: { username: "alice", email: "alice@example.com" }) {
    user { id }
  }
  # 2단계: 1단계 완료 후 실행
  createPost(input: { title: "Hello", body: "World", authorId: "new-user-id" }) {
    post { id }
  }
}
```

이 순서 보장은 이후 뮤테이션이 이전 뮤테이션의 부수 효과에 의존할 때 중요합니다. 하지만 첫 번째 뮤테이션의 결과를 두 번째에서 참조할 수는 없습니다 — GraphQL은 필드 간 변수 참조를 지원하지 않습니다. 의존적인 작업의 경우, 별도의 요청을 사용하거나 서버 측에서 체인을 처리하는 단일 뮤테이션을 사용하세요.

### 7.3 Input 패턴

Relay 규칙을 따라 뮤테이션은 단일 `input` 인수를 받습니다:

```graphql
# ❌ 여러 인수 (확장하기 어렵고, 장황함)
type Mutation {
  createUser(
    username: String!
    email: String!
    password: String!
    displayName: String
    bio: String
    avatarUrl: String
  ): User!
}

# ✅ 단일 input 인수 (간결하고, 확장 가능)
input CreateUserInput {
  username: String!
  email: String!
  password: String!
  displayName: String
  bio: String
  avatarUrl: String
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserPayload!
}
```

Input 패턴의 이점:
- **확장 가능**: 입력 타입에 필드를 추가해도 호환성 유지
- **재사용 가능**: 입력 타입을 공유하거나 합성 가능
- **클라이언트 친화적**: 변수가 input에 깔끔하게 매핑됨

### 7.4 일반적인 뮤테이션 패턴

```graphql
# CRUD 작업
type Mutation {
  # 생성
  createPost(input: CreatePostInput!): CreatePostPayload!

  # 수정 (부분 업데이트)
  updatePost(id: ID!, input: UpdatePostInput!): UpdatePostPayload!

  # 삭제
  deletePost(id: ID!): DeletePostPayload!

  # 도메인 특화 작업 (CRUD 아님)
  publishPost(id: ID!): PublishPostPayload!
  archivePost(id: ID!): ArchivePostPayload!
  likePost(id: ID!): LikePostPayload!
  addComment(postId: ID!, input: AddCommentInput!): AddCommentPayload!
}
```

뮤테이션에는 의도를 명확히 전달하도록 `동사 + 명사` 형식으로 이름을 붙이세요. `updateEntity`나 `modifyData` 같은 일반적인 이름은 피하세요.

## 8. 오류 처리 패턴

GraphQL에는 두 계층의 오류가 있으며, 올바른 패턴을 선택하는 것이 클라이언트 개발자 경험에 매우 중요합니다.

### 8.1 최상위 오류 (전송/실행 오류)

응답 최상위의 `errors` 배열에 나타납니다:

```json
{
  "data": null,
  "errors": [
    {
      "message": "Cannot query field 'nonexistent' on type 'User'",
      "locations": [{ "line": 3, "column": 5 }],
      "extensions": {
        "code": "GRAPHQL_VALIDATION_FAILED"
      }
    }
  ]
}
```

최상위 오류에 포함되는 것들:
- 문법 오류 (파싱 실패)
- 검증 오류 (잘못된 쿼리)
- 인가 오류 (로그인 안됨)
- 내부 서버 오류 (처리되지 않은 예외)

### 8.2 페이로드 내 애플리케이션 오류

비즈니스 로직 오류(유효성 검사, 찾지 못함, 권한 거부)에는 페이로드 패턴을 사용합니다:

```graphql
type CreateUserPayload {
  user: User
  errors: [CreateUserError!]!
}

type CreateUserError {
  field: String
  message: String!
  code: CreateUserErrorCode!
}

enum CreateUserErrorCode {
  USERNAME_TAKEN
  EMAIL_TAKEN
  INVALID_EMAIL
  PASSWORD_TOO_WEAK
  RATE_LIMITED
}
```

```json
{
  "data": {
    "createUser": {
      "user": null,
      "errors": [
        {
          "field": "username",
          "message": "Username 'alice' is already taken",
          "code": "USERNAME_TAKEN"
        }
      ]
    }
  }
}
```

### 8.3 유니언 기반 오류 처리

더 타입 안전한 접근 방식으로 유니언을 사용합니다:

```graphql
union CreateUserResult = CreateUserSuccess | ValidationError | NotFoundError

type CreateUserSuccess {
  user: User!
}

type ValidationError {
  field: String!
  message: String!
}

type NotFoundError {
  message: String!
  resourceType: String!
  resourceId: ID!
}

type Mutation {
  createUser(input: CreateUserInput!): CreateUserResult!
}
```

```graphql
mutation {
  createUser(input: { username: "alice", email: "alice@example.com" }) {
    ... on CreateUserSuccess {
      user { id username }
    }
    ... on ValidationError {
      field
      message
    }
  }
}
```

이 패턴은 클라이언트가 오류를 명시적으로 처리하도록 강제합니다. 트레이드오프는 더 장황한 쿼리입니다.

## 9. 쿼리 구성 팁

### 9.1 파일 당 하나의 오퍼레이션

프로덕션 코드베이스에서는 각 오퍼레이션을 자체 파일에 저장하세요:

```
src/
  graphql/
    queries/
      GetUser.graphql
      GetPosts.graphql
      SearchResults.graphql
    mutations/
      CreatePost.graphql
      UpdateUser.graphql
    fragments/
      UserBasicInfo.graphql
      PostSummary.graphql
```

GraphQL 코드 생성기(`graphql-codegen` 등)가 이 파일들을 처리해 타입이 지정된 클라이언트 코드를 생성할 수 있습니다.

### 9.2 명명 규칙

```graphql
# 쿼리: Get + 리소스
query GetUser($id: ID!) { ... }
query GetPosts($filter: PostFilter) { ... }
query SearchContent($query: String!) { ... }

# 뮤테이션: 동사 + 리소스
mutation CreatePost($input: CreatePostInput!) { ... }
mutation UpdateUser($id: ID!, $input: UpdateUserInput!) { ... }
mutation DeleteComment($id: ID!) { ... }
mutation PublishPost($id: ID!) { ... }

# 구독: On + 이벤트
subscription OnCommentAdded($postId: ID!) { ... }
subscription OnUserStatusChanged { ... }
```

### 9.3 쿼리에서의 과잉 응답 방지

GraphQL에서도 과잉 응답은 가능합니다:

```graphql
# ❌ "혹시 모르니" 모두 가져오기
query {
  user(id: "1") {
    id username email displayName bio avatarUrl
    createdAt updatedAt lastLoginAt
    posts { id title body publishedAt tags { name } }
    followers { id username avatarUrl }
    following { id username avatarUrl }
  }
}

# ✅ 현재 뷰에 필요한 것만 가져오기
query UserProfile($userId: ID!) {
  user(id: $userId) {
    username
    displayName
    bio
    avatarUrl
    followerCount
  }
}
```

---

## 10. 연습 문제

### 연습 1: 쿼리 작성 (초급)

이 스키마가 주어졌을 때:

```graphql
type Query {
  user(id: ID!): User
  posts(first: Int, after: String, authorId: ID): PostConnection!
  search(query: String!, types: [SearchType!]): [SearchResult!]!
}

enum SearchType { USER POST COMMENT }
union SearchResult = User | Post | Comment

type User { id: ID!, username: String!, email: String!, posts: [Post!]! }
type Post { id: ID!, title: String!, body: String!, author: User!, comments: [Comment!]! }
type Comment { id: ID!, body: String!, author: User! }

type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
}
type PostEdge { node: Post!, cursor: String! }
type PageInfo { hasNextPage: Boolean!, endCursor: String }
```

적절한 변수를 사용해 다음 쿼리들을 작성하세요:

1. ID로 사용자의 username과 email 조회
2. 제목, 작성자 username, 커서 기반 페이지네이션 정보를 포함한 처음 5개 게시글 조회
3. 모든 타입에서 "graphql"을 검색하여 각 타입에 적합한 필드 반환

### 연습 2: 프래그먼트 리팩토링 (중급)

다음 쿼리에는 중복된 필드가 있습니다. 프래그먼트를 사용해 리팩토링하세요:

```graphql
query {
  recentPosts: posts(sortBy: RECENT, first: 5) {
    id
    title
    publishedAt
    author {
      id
      username
      avatarUrl
    }
    commentCount
  }
  popularPosts: posts(sortBy: POPULAR, first: 5) {
    id
    title
    publishedAt
    author {
      id
      username
      avatarUrl
    }
    likeCount
  }
  me {
    id
    username
    avatarUrl
    drafts: posts(status: DRAFT) {
      id
      title
      publishedAt
      author {
        id
        username
        avatarUrl
      }
    }
  }
}
```

### 연습 3: 뮤테이션 설계 (중급)

"사용자 프로필 업데이트" 뮤테이션을 완전하게 설계하세요. 다음을 포함해야 합니다:

1. 수정 가능한 모든 필드를 가진 입력 타입 (표시 이름, 소개, 아바타 URL, 시간대)
2. 업데이트된 사용자와 잠재적 오류를 포함하는 페이로드 타입
3. 필드 수준 오류 정보를 가진 오류 타입
4. 뮤테이션 필드 정의
5. 변수를 포함한 샘플 뮤테이션 쿼리

### 연습 4: 별칭과 디렉티브 (중급)

다음을 수행하는 단일 쿼리를 작성하세요:

1. 현재 사용자 프로필 조회
2. 기술, 과학, 스포츠 각 카테고리에서 상위 3개 게시글 조회 (별칭 사용)
3. 현재 사용자의 알림 수를 조건부로 포함 (`@include` 사용)
4. 게시글 댓글을 조건부로 건너뜀 (`@skip` 사용)

모든 동적 값에 적절한 변수를 사용하세요.

### 연습 5: 오류 처리 비교 (고급)

소셜 미디어 앱에서 "사용자 팔로우" 뮤테이션이 필요합니다. 다음 두 방식으로 설계하세요:

**방식 A**: 오류 배열을 가진 페이로드 패턴
**방식 B**: 유니언 기반 결과 타입

각 방식에 대해:
- 스키마 타입 정의
- 뮤테이션 쿼리 작성
- 다음 경우의 JSON 응답 표시: (a) 성공, (b) 사용자 없음, (c) 이미 팔로우 중, (d) 자기 자신을 팔로우할 수 없음

두 접근 방식을 비교하고 어느 것을 선택할지, 그 이유는 무엇인지 설명하세요.

---

## 11. 참고 자료

- GraphQL 쿼리 언어 명세 - https://spec.graphql.org/October2021/#sec-Language
- GraphQL 변수 - https://spec.graphql.org/October2021/#sec-Language.Variables
- GraphQL 프래그먼트 - https://spec.graphql.org/October2021/#sec-Language.Fragments
- GraphQL 디렉티브 - https://spec.graphql.org/October2021/#sec-Language.Directives
- Relay 뮤테이션 규칙 - https://relay.dev/docs/guided-tour/updating-data/graphql-mutations/
- Apollo Client 프래그먼트 가이드 - https://www.apollographql.com/docs/react/data/fragments/

---

**이전**: [스키마 설계](./02_Schema_Design.md) | **다음**: [리졸버](./04_Resolvers.md)
