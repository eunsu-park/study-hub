# 01. GraphQL 기초

**이전**: [GraphQL 개요](./00_Overview.md) | **다음**: [스키마 설계](./02_Schema_Design.md)

---

GraphQL은 API를 위한 쿼리 언어이자, 기존 데이터를 사용해 해당 쿼리를 실행하는 런타임(Runtime)입니다. 서버가 각 엔드포인트(Endpoint)에서 반환할 데이터를 결정하는 REST와 달리, GraphQL은 클라이언트가 주도권을 가집니다. 클라이언트는 필요한 것을 정확히 요청하고, 서버는 그 형태 그대로 데이터를 응답합니다. 이 레슨은 스키마(Schema) 작성, 리졸버(Resolver) 구현, 성능 최적화에 앞서 모든 GraphQL 개발자가 반드시 이해해야 할 핵심 개념을 소개합니다.

**난이도**: ⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. GraphQL이 무엇인지 설명하고 REST API 대비 장점을 명확히 제시
2. 세 가지 루트 오퍼레이션(Operation) 타입 식별: 쿼리(Query), 뮤테이션(Mutation), 구독(Subscription)
3. 스키마 정의 언어(SDL, Schema Definition Language)를 사용해 기본 스키마 읽기 및 작성
4. 인트로스펙션(Introspection) 쿼리를 사용해 GraphQL API 탐색
5. GraphQL 실행 모델(파싱, 검증, 실행) 설명

---

## 목차

1. [GraphQL이란?](#1-graphql이란)
2. [GraphQL vs REST](#2-graphql-vs-rest)
3. [핵심 개념](#3-핵심-개념)
4. [스키마 정의 언어 (SDL)](#4-스키마-정의-언어-sdl)
5. [인트로스펙션](#5-인트로스펙션)
6. [실행 모델](#6-실행-모델)
7. [GraphiQL과 Apollo Explorer](#7-graphiql과-apollo-explorer)
8. [첫 번째 GraphQL 서버 만들기](#8-첫-번째-graphql-서버-만들기)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. GraphQL이란?

GraphQL은 Facebook의 모바일 팀이 REST API에서 겪던 문제를 해결하기 위해 2012년에 만들어졌습니다. 예를 들어 뉴스피드(News Feed)는 게시글, 작성자, 댓글, 좋아요, 친구 연결 등 깊이 중첩된 데이터를 필요로 합니다. REST에서는 이 데이터를 가져오려면 여러 번의 요청을 보내거나, 필요 이상의 데이터를 반환하는 비대한 엔드포인트를 사용해야 했습니다.

GraphQL은 세 가지 핵심 혁신을 제공합니다:

1. **타입 시스템(Type System)**: 데이터를 설명하는 스키마(Schema)
2. **쿼리 언어(Query Language)**: 클라이언트가 필요한 것을 정확히 지정할 수 있는 언어
3. **런타임(Runtime)**: 스키마에 대해 쿼리를 검증하고 실행하는 엔진

```
전통적인 REST                         GraphQL
──────────────                         ─────────
GET /users/1           ──→  1 req     query {
GET /users/1/posts     ──→  2 req       user(id: 1) {
GET /posts/5/comments  ──→  3 req         name
                                          posts {
요청 3번, 사용하지 않는                    title
필드 다수 포함                             comments {
                                             body
                                           }
                                         }
                                       }
                                     }
                                     ──→  요청 1번, 정확한 데이터
```

GraphQL 명세(GraphQL Foundation이 Linux Foundation 산하에서 관리)는 언어, 타입 시스템, 실행 의미론을 정의합니다. 전송 방식에 구애받지 않으며(Transport-agnostic) — 대부분의 구현은 HTTP를 사용하지만, GraphQL은 WebSocket, gRPC 또는 다른 프로토콜 위에서도 동작할 수 있습니다.

## 2. GraphQL vs REST

GraphQL이 빛을 발하는 상황(그리고 그렇지 않은 상황)을 이해하려면 REST와 공정하게 비교해야 합니다.

### GraphQL이 유리한 경우

| 문제 | REST | GraphQL |
|------|------|---------|
| **과잉 응답(Over-fetching)** | `/users/1`이 30개 필드 전체 반환 | 클라이언트가 `name`과 `email`만 선택 |
| **부족 응답(Under-fetching)** | 하나의 뷰에 3개 엔드포인트 필요 | 쿼리 하나, 요청 하나 |
| **버전 관리(Versioning)** | `/api/v1/`, `/api/v2/` | `@deprecated`로 스키마 점진적 진화 |
| **문서화(Documentation)** | 별도 작성 필요 (Swagger/OpenAPI) | 인트로스펙션을 통해 내장 제공 |
| **타입 안전성(Type Safety)** | 선택적 (JSON Schema) | 필수 (SDL) |

### REST가 유리한 경우

| 측면 | REST | GraphQL |
|------|------|---------|
| **캐싱(Caching)** | HTTP 캐싱이 자연스럽게 동작 (GET + URL = 캐시 키) | 특별 처리 필요 (POST 본문이 매 요청마다 다름) |
| **파일 업로드(File Uploads)** | Multipart form data가 표준 | 확장 필요 (graphql-upload) |
| **단순성(Simplicity)** | CRUD 리소스에 매우 직관적 | 스키마 설계에 초기 투자 필요 |
| **브라우저 지원(Browser Support)** | URL로 기본 fetch 사용 가능 | 최적 사용을 위해 클라이언트 라이브러리 필요 |
| **속도 제한(Rate Limiting)** | 엔드포인트별로 명확하게 설정 가능 | 요청마다 쿼리 복잡도가 달라짐 |

### 실용적인 관점

대부분의 프로덕션 시스템은 둘 다 사용합니다. REST는 단순한 CRUD와 파일 작업을 처리하고, GraphQL은 복잡한 관계형 데이터 요구를 담당합니다. Facebook, GitHub, Shopify, Airbnb 모두 REST와 GraphQL을 혼합해서 사용합니다.

## 3. 핵심 개념

GraphQL에는 세 가지 루트 오퍼레이션 타입이 있습니다. 모든 GraphQL API는 최소한 `Query` 타입을 정의해야 합니다.

### 3.1 쿼리(Query) — 읽기

쿼리는 데이터를 가져옵니다. REST의 GET 요청과 유사합니다.

```graphql
# 클라이언트가 보내는 쿼리
query GetUser {
  user(id: "1") {
    name
    email
    posts {
      title
      publishedAt
    }
  }
}
```

```json
// 서버가 응답하는 형태
{
  "data": {
    "user": {
      "name": "Alice",
      "email": "alice@example.com",
      "posts": [
        {
          "title": "Intro to GraphQL",
          "publishedAt": "2025-01-15"
        }
      ]
    }
  }
}
```

응답 형태는 쿼리 형태와 정확히 일치합니다. 이것이 GraphQL의 근본 원칙입니다: **요청한 것을 정확히 받는다**.

### 3.2 뮤테이션(Mutation) — 쓰기

뮤테이션은 데이터를 변경합니다. REST의 POST, PUT, PATCH, DELETE와 유사합니다.

```graphql
mutation CreatePost {
  createPost(input: {
    title: "GraphQL Basics"
    body: "GraphQL is a query language..."
    authorId: "1"
  }) {
    id
    title
    createdAt
  }
}
```

뮤테이션은 쿼리와 동일한 요청/응답 패턴을 따르지만, 관례상 부수 효과(Side Effect)를 나타냅니다. 서버는 뮤테이션 필드를 순차적으로 실행합니다(쿼리는 병렬 실행이 가능한 것과 대조됩니다).

### 3.3 구독(Subscription) — 실시간

구독은 실시간 업데이트를 위한 지속적인 연결을 유지합니다.

```graphql
subscription OnNewComment {
  commentAdded(postId: "5") {
    id
    body
    author {
      name
    }
  }
}
```

게시글 5에 새 댓글이 추가되면, 서버는 구독한 모든 클라이언트에 데이터를 푸시합니다. 구독은 일반적으로 WebSocket을 사용합니다.

### 3.4 응답 형식

모든 GraphQL 응답은 일관된 형태를 가집니다:

```json
{
  "data": { ... },      // 결과 (모든 필드에 오류가 발생하면 null)
  "errors": [ ... ],    // 선택 사항: 오류 객체 배열
  "extensions": { ... } // 선택 사항: 메타데이터 (타이밍, 트레이싱)
}
```

이 예측 가능한 구조 덕분에 클라이언트는 항상 데이터와 오류의 위치를 알 수 있습니다.

## 4. 스키마 정의 언어 (SDL)

스키마 정의 언어(SDL, Schema Definition Language)는 GraphQL 스키마를 정의하는 문법입니다. 사람이 읽기 쉬우며, 문서이자 클라이언트와 서버 간의 계약(Contract) 역할을 합니다.

### 4.1 타입 정의

```graphql
# 객체 타입: 엔티티를 나타냄
type User {
  id: ID!                # Null이 될 수 없는 스칼라
  name: String!
  email: String!
  age: Int
  score: Float
  isActive: Boolean!
  posts: [Post!]!        # Null이 될 수 없는 Post의 배열 (배열 내 각 항목도 null 불가)
  createdAt: String
}

type Post {
  id: ID!
  title: String!
  body: String!
  author: User!
  comments: [Comment!]!
  tags: [String!]
}

type Comment {
  id: ID!
  body: String!
  author: User!
  post: Post!
}
```

### 4.2 다섯 가지 내장 스칼라 타입

| 스칼라 | 설명 | 예시 |
|--------|------|------|
| `Int` | 32비트 부호 있는 정수 | `42` |
| `Float` | 배정밀도 부동소수점 | `3.14` |
| `String` | UTF-8 문자 시퀀스 | `"hello"` |
| `Boolean` | `true` 또는 `false` | `true` |
| `ID` | 고유 식별자 (String으로 직렬화) | `"abc123"` |

### 4.3 루트 타입

```graphql
type Query {
  user(id: ID!): User
  users(limit: Int = 10, offset: Int = 0): [User!]!
  post(id: ID!): Post
  searchPosts(term: String!): [Post!]!
}

type Mutation {
  createUser(input: CreateUserInput!): User!
  updateUser(id: ID!, input: UpdateUserInput!): User!
  deleteUser(id: ID!): Boolean!
  createPost(input: CreatePostInput!): Post!
}

type Subscription {
  commentAdded(postId: ID!): Comment!
  userStatusChanged: User!
}
```

### 4.4 입력 타입

입력 타입(Input Type)은 뮤테이션 인수의 형태를 정의합니다:

```graphql
input CreateUserInput {
  name: String!
  email: String!
  age: Int
}

input UpdateUserInput {
  name: String
  email: String
  age: Int
}
```

입력 타입은 `type` 대신 `input` 키워드를 사용하며, 객체 타입을 반환하는 필드를 가질 수 없습니다.

### 4.5 schema 키워드

`schema` 키워드는 루트 타입을 명시적으로 매핑합니다:

```graphql
schema {
  query: Query
  mutation: Mutation
  subscription: Subscription
}
```

대부분의 프레임워크는 타입 이름으로 이를 추론하므로, 직접 작성하는 경우는 드뭅니다.

## 5. 인트로스펙션

GraphQL의 가장 강력한 기능 중 하나는 인트로스펙션(Introspection): 스키마 자체를 쿼리하는 능력입니다. GraphiQL과 Apollo Explorer가 자동으로 문서를 생성하고 자동 완성을 제공할 수 있는 것도 이 덕분입니다.

### 5.1 __schema 쿼리

```graphql
# 스키마의 모든 타입 목록 조회
query IntrospectSchema {
  __schema {
    types {
      name
      kind
      description
    }
    queryType {
      name
    }
    mutationType {
      name
    }
  }
}
```

### 5.2 __type 쿼리

```graphql
# 특정 타입 상세 조회
query IntrospectUser {
  __type(name: "User") {
    name
    kind
    fields {
      name
      type {
        name
        kind
        ofType {
          name
          kind
        }
      }
    }
  }
}
```

응답에는 모든 필드, 해당 타입, null 가능 여부, 인수가 표시됩니다. 이것이 바로 인트로스펙션의 실제 동작입니다.

### 5.3 보안 주의 사항

프로덕션에서는 공개 API에 대해 인트로스펙션을 비활성화해야 합니다. 전체 스키마가 잠재적인 공격자에게 노출될 수 있습니다:

```javascript
// Apollo Server: 프로덕션에서 인트로스펙션 비활성화
const server = new ApolloServer({
  typeDefs,
  resolvers,
  introspection: process.env.NODE_ENV !== 'production',
});
```

## 6. 실행 모델

모든 GraphQL 요청은 세 단계를 거칩니다:

```
                 ┌─────────────┐
   쿼리 문자열   │    파싱     │  → 추상 구문 트리 (AST)
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │    검증     │  → 스키마에 대한 검증
                 └──────┬──────┘
                        │
                 ┌──────▼──────┐
                 │    실행     │  → 리졸버 호출, 응답 구성
                 └─────────────┘
```

### 6.1 파싱(Parse)

쿼리 문자열이 추상 구문 트리(AST, Abstract Syntax Tree)로 파싱됩니다. 문법 오류는 이 단계에서 잡힙니다.

```
"{ user(id: 1) { name } }"

  → Document
    └─ OperationDefinition (query)
       └─ SelectionSet
          └─ Field: user
             ├─ Arguments: [id: 1]
             └─ SelectionSet
                └─ Field: name
```

### 6.2 검증(Validate)

AST가 스키마에 대해 검증됩니다. 이 단계에서는 다음과 같은 오류를 잡습니다:

- 해당 타입에 존재하지 않는 필드 쿼리
- 필수 인수 누락
- 타입 불일치 (Int 기대 위치에 String 전달)
- 불가능한 프래그먼트(Fragment) 타입 조건

```graphql
# 검증 실패 예시: 'age'는 존재할 수 있지만 'nonExistentField'는 존재하지 않음
query {
  user(id: "1") {
    name
    nonExistentField   # ← 검증 오류
  }
}
```

### 6.3 실행(Execute)

실행기(Executor)가 AST를 순회하며 각 필드에 대한 리졸버(Resolver) 함수를 호출합니다. 리졸버는 레슨 04에서 자세히 다루지만, 핵심 동작은 다음과 같습니다:

1. 루트 Query 타입에서 시작
2. 요청된 각 필드의 리졸버를 호출
3. 객체 타입 필드의 경우, 해당 선택 집합으로 재귀 처리
4. 최종 JSON 응답 조립

```
Query.user(id: "1")          → { id: "1", name: "Alice", ... }
  └─ User.name               → "Alice"
  └─ User.posts               → [{ id: "5", title: "...", ... }]
       └─ Post.title          → "Intro to GraphQL"
       └─ Post.comments       → [{ id: "10", body: "...", ... }]
            └─ Comment.body   → "Great post!"
```

## 7. GraphiQL과 Apollo Explorer

GraphQL API는 curl 명령어보다 인터랙티브 도구를 통해 탐색하는 것이 훨씬 좋습니다.

### 7.1 GraphiQL

GraphiQL("그래피컬"로 발음)은 GraphQL의 레퍼런스 IDE입니다. 다음 기능을 제공합니다:

- 스키마 문서 브라우저 (인트로스펙션 기반)
- 문법 강조 및 자동 완성이 지원되는 쿼리 에디터
- 변수 에디터 패널
- 응답 뷰어
- 쿼리 히스토리

대부분의 GraphQL 서버는 전용 엔드포인트에서 GraphiQL을 제공합니다:

```javascript
// Express + graphql-http with GraphiQL
import express from 'express';
import { createHandler } from 'graphql-http/lib/use/express';
import { buildSchema } from 'graphql';

const app = express();

const schema = buildSchema(`
  type Query {
    hello: String
  }
`);

app.all('/graphql', createHandler({
  schema,
  rootValue: {
    hello: () => 'Hello, World!',
  },
}));

// /graphiql 경로에서 GraphiQL HTML 제공
app.get('/graphiql', (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html>
      <head>
        <link rel="stylesheet"
          href="https://unpkg.com/graphiql/graphiql.min.css" />
      </head>
      <body style="margin:0">
        <div id="graphiql" style="height:100vh"></div>
        <script src="https://unpkg.com/graphiql/graphiql.min.js"></script>
        <script>
          GraphiQL.createRoot(document.getElementById('graphiql'))
            .render(React.createElement(GraphiQL, {
              fetcher: GraphiQL.createFetcher({ url: '/graphql' }),
            }));
        </script>
      </body>
    </html>
  `);
});

app.listen(4000);
```

### 7.2 Apollo Explorer

Apollo Explorer는 클라우드 호스팅 GraphQL IDE로 추가 기능을 제공합니다:

- 스키마 변경 추적 및 diff 비교
- 오퍼레이션 컬렉션 및 공유
- 성능 트레이싱(Performance Tracing)
- 목(Mock) 응답

[studio.apollographql.com/sandbox](https://studio.apollographql.com/sandbox)에서 모든 GraphQL 엔드포인트에 연결할 수 있습니다.

## 8. 첫 번째 GraphQL 서버 만들기

모든 개념을 실제로 확인하기 위해 Node.js로 최소한의 GraphQL 서버를 만들어 보겠습니다.

```bash
# 프로젝트 초기화
mkdir graphql-intro && cd graphql-intro
npm init -y
npm install @apollo/server graphql
```

```javascript
// server.js
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';

// 1. 스키마 정의 (SDL)
const typeDefs = `#graphql
  type Book {
    id: ID!
    title: String!
    author: String!
    year: Int
  }

  type Query {
    books: [Book!]!
    book(id: ID!): Book
  }

  type Mutation {
    addBook(title: String!, author: String!, year: Int): Book!
  }
`;

// 2. 인메모리 데이터 저장소
const books = [
  { id: '1', title: 'The Pragmatic Programmer', author: 'David Thomas', year: 1999 },
  { id: '2', title: 'Clean Code', author: 'Robert C. Martin', year: 2008 },
  { id: '3', title: 'Design Patterns', author: 'Gang of Four', year: 1994 },
];

let nextId = 4;

// 3. 리졸버 정의
const resolvers = {
  Query: {
    books: () => books,
    book: (_, { id }) => books.find(b => b.id === id),
  },
  Mutation: {
    addBook: (_, { title, author, year }) => {
      const book = { id: String(nextId++), title, author, year };
      books.push(book);
      return book;
    },
  },
};

// 4. 서버 생성 및 시작
const server = new ApolloServer({ typeDefs, resolvers });
const { url } = await startStandaloneServer(server, { listen: { port: 4000 } });
console.log(`Server ready at ${url}`);
```

실행:

```bash
node server.js
# Server ready at http://localhost:4000/
```

브라우저에서 `http://localhost:4000/`를 열면 Apollo Server 4가 기본으로 Apollo Sandbox를 제공합니다. 다음 쿼리들을 시도해 보세요:

```graphql
# 모든 책 조회
query {
  books {
    title
    author
  }
}

# 단권 조회
query {
  book(id: "2") {
    title
    year
  }
}

# 새 책 추가
mutation {
  addBook(title: "Refactoring", author: "Martin Fowler", year: 1999) {
    id
    title
  }
}
```

이 최소한의 예제는 GraphQL의 전체 흐름을 보여줍니다: 스키마 정의, 리졸버 구현, 쿼리 실행, 뮤테이션 처리.

---

## 9. 연습 문제

### 연습 1: 스키마 읽기 (초급)

다음 스키마를 보고 아래 질문에 답하세요:

```graphql
type Query {
  product(id: ID!): Product
  products(category: String): [Product!]!
}

type Product {
  id: ID!
  name: String!
  price: Float!
  category: String!
  reviews: [Review!]
}

type Review {
  id: ID!
  rating: Int!
  body: String
  reviewer: String!
}
```

1. `product(id: "999")`를 쿼리했을 때 해당 ID의 상품이 없다면 어떻게 됩니까?
2. `products(category: "electronics")`는 빈 배열을 반환할 수 있습니까? `null`을 반환할 수 있습니까?
3. `Product`의 `reviews`가 `null`이 될 수 있습니까? 빈 배열 `[]`이 될 수 있습니까?
4. 개별 `Review`의 `body`가 `null`이 될 수 있습니까? `rating`이 `null`이 될 수 있습니까?

### 연습 2: 스키마 작성 (중급)

음악 스트리밍 서비스를 위한 GraphQL 스키마를 설계하세요. 다음을 포함해야 합니다:

- `Song`: 제목, 아티스트, 재생 시간(초), 앨범, 장르
- `Playlist`: 이름, 설명, 생성자, 곡 목록
- `User`: 사용자 이름, 이메일, 플레이리스트
- 제목으로 곡을 검색하는 쿼리
- 플레이리스트를 생성하는 뮤테이션

적절한 스칼라 타입, 비-null 수식자(Non-null Modifier), 리스트 수식자(List Modifier)를 사용하세요.

### 연습 3: 과잉 응답 식별 (초급)

모바일 앱에서 사용자 이름과 아바타 목록을 표시해야 합니다. REST API는 다음을 반환합니다:

```json
{
  "id": 1, "name": "Alice", "avatar": "/img/alice.jpg",
  "email": "alice@example.com", "phone": "555-0100",
  "address": { "street": "123 Main", "city": "Springfield", "zip": "62704" },
  "preferences": { "theme": "dark", "language": "en", "notifications": true },
  "lastLogin": "2025-01-15T10:30:00Z",
  "createdAt": "2023-06-01T00:00:00Z"
}
```

모바일 앱에 필요한 데이터만 가져오는 GraphQL 쿼리를 작성하세요. 대역폭 절감량을 백분율로 추정해 보세요.

### 연습 4: 인트로스펙션 쿼리 (중급)

다음을 가져오는 인트로스펙션 쿼리를 작성하세요:

1. 연습 1의 `Product` 타입의 모든 필드
2. 각 필드의 타입 이름과 null 가능 여부
3. 루트 `Query` 타입 필드의 인수

### 연습 5: GraphQL 서버 구축 (고급)

섹션 8의 책 서버를 다음과 같이 확장하세요:

1. `Genre` 열거형(Enum) 타입 추가: `FICTION`, `NON_FICTION`, `TECHNICAL`, `BIOGRAPHY`
2. `Book` 타입에 `genre` 필드 추가
3. `booksByGenre(genre: Genre!): [Book!]!` 쿼리 추가
4. `updateBook(id: ID!, title: String, author: String, year: Int): Book` 뮤테이션 추가
5. `deleteBook(id: ID!): Boolean!` 뮤테이션 추가

Apollo Sandbox를 사용해 모든 오퍼레이션을 테스트하세요.

---

## 10. 참고 자료

- GraphQL 명세 (2021년 10월 버전) - https://spec.graphql.org/October2021/
- GraphQL Foundation - https://graphql.org/
- Apollo Server 문서 - https://www.apollographql.com/docs/apollo-server/
- "GraphQL: A Query Language for APIs" (Facebook 원본 블로그 포스트) - https://engineering.fb.com/2015/09/14/core-infra/graphql-a-data-query-language/
- Lee Byron, "Lessons from 4 Years of GraphQL" (GraphQL Summit 2019)
- Principled GraphQL - https://principledgraphql.com/

---

**이전**: [GraphQL 개요](./00_Overview.md) | **다음**: [스키마 설계](./02_Schema_Design.md)
