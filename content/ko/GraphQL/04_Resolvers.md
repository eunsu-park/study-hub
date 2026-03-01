# 04. 리졸버

**이전**: [쿼리와 뮤테이션](./03_Queries_and_Mutations.md) | **다음**: [DataLoader와 N+1](./05_DataLoader_N_plus_1.md)

---

스키마(Schema)가 계약이라면, 리졸버(Resolver)는 구현입니다. GraphQL 스키마의 모든 필드는 해당 필드의 값을 가져오거나 계산하는 방법을 아는 리졸버 함수로 뒷받침됩니다. 리졸버가 작동하는 방식 — 시그니처(Signature), 실행 순서, 상호 관계 — 을 이해하는 것은 올바르고 성능 좋은 GraphQL 서버를 구축하는 데 필수적입니다. 이 레슨에서는 리졸버 메커니즘을 기초부터 깊이 파헤칩니다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 리졸버 함수의 네 가지 인수(parent, args, context, info) 설명
2. 중첩 필드에 대한 리졸버 체인(Resolver Chain) 작동 방식 및 실행 순서 식별
3. 컨텍스트(Context) 객체를 사용해 요청별 리소스(데이터베이스 연결, 인증 토큰) 공유
4. 리졸버 설계 패턴 적용: 얇은 리졸버(Thin Resolver), 서비스 레이어 위임, 필드 수준 리졸버
5. 리졸버 내에서 구조화된 오류 처리 구현

---

## 목차

1. [리졸버란?](#1-리졸버란)
2. [리졸버 시그니처](#2-리졸버-시그니처)
3. [기본 리졸버](#3-기본-리졸버)
4. [리졸버 체인](#4-리졸버-체인)
5. [컨텍스트 객체](#5-컨텍스트-객체)
6. [Info 객체](#6-info-객체)
7. [비동기 리졸버](#7-비동기-리졸버)
8. [리졸버 설계 패턴](#8-리졸버-설계-패턴)
9. [리졸버의 오류 처리](#9-리졸버의-오류-처리)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. 리졸버란?

리졸버(Resolver)는 스키마의 단일 필드에 대한 데이터를 채우는 함수입니다. GraphQL 쿼리가 실행될 때, 서버는 선택 집합의 각 필드에 대한 리졸버를 호출합니다. 이 모든 리졸버 호출의 결과가 응답 JSON으로 조립됩니다.

```
스키마 필드                   리졸버 함수
───────────                   ─────────────────
Query.user(id: ID!)    →    (parent, args, ctx, info) => db.users.findById(args.id)
User.name              →    (parent) => parent.name
User.posts             →    (parent, args, ctx) => db.posts.findByAuthorId(parent.id)
Post.title             →    (parent) => parent.title
Post.author            →    (parent, args, ctx) => db.users.findById(parent.authorId)
```

리졸버 트리는 쿼리 트리를 반영합니다. 다음과 같은 쿼리에 대해:

```graphql
query {
  user(id: "1") {
    name
    posts {
      title
      author {
        name
      }
    }
  }
}
```

실행 순서는:

```
1. Query.user(id: "1")           → User 객체 반환
2.   User.name                   → "Alice" 반환
3.   User.posts                  → [Post, Post, ...] 반환
4.     Post.title (각각)          → "GraphQL 101" 반환
5.     Post.author (각각)         → User 객체 반환
6.       User.name (각각)         → "Alice" 반환
```

## 2. 리졸버 시그니처

모든 리졸버는 네 개의 인수를 받습니다:

```javascript
function resolver(parent, args, context, info) {
  // ...
}
```

### 2.1 parent (root, obj, source라고도 불림)

부모 필드 리졸버의 반환 값입니다. 루트 수준 리졸버(Query, Mutation)의 경우, 서버에 전달된 `rootValue`입니다 (보통 `undefined`).

```javascript
const resolvers = {
  Query: {
    // parent는 rootValue (보통 undefined)
    user: (parent, args, context) => {
      return context.db.users.findById(args.id);
      // 반환: { id: '1', name: 'Alice', email: 'alice@example.com' }
    },
  },
  User: {
    // parent는 Query.user가 반환한 User 객체
    name: (parent) => {
      // parent = { id: '1', name: 'Alice', email: 'alice@example.com' }
      return parent.name; // "Alice"
    },
    posts: (parent, args, context) => {
      // parent.id로 해당 사용자의 게시글을 찾음
      return context.db.posts.findByAuthorId(parent.id);
    },
  },
};
```

### 2.2 args

필드에 전달된 인수를 담은 객체:

```graphql
type Query {
  posts(first: Int = 10, offset: Int = 0, status: PostStatus): [Post!]!
}
```

```javascript
const resolvers = {
  Query: {
    posts: (parent, args, context) => {
      // args = { first: 10, offset: 0, status: 'PUBLISHED' }
      const { first, offset, status } = args;
      return context.db.posts.findAll({
        where: status ? { status } : {},
        limit: first,
        offset,
      });
    },
  },
};
```

### 2.3 context

단일 요청의 모든 리졸버가 사용할 수 있는 공유 객체입니다. 각 요청마다 새롭게 생성되며 일반적으로 다음을 포함합니다:

- 데이터베이스 연결 또는 ORM 인스턴스
- 인증/인가 정보
- DataLoader 인스턴스 (레슨 05)
- 요청별 유틸리티 (로깅, 트레이싱)

```javascript
// 서버 설정: 요청당 컨텍스트 생성
const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => ({
    // 모든 요청마다 새로운 컨텍스트
    db: database,
    user: await authenticateUser(req.headers.authorization),
    loaders: createDataLoaders(database),
  }),
});
```

### 2.4 info

현재 실행 상태에 대한 메타데이터를 담은 `GraphQLResolveInfo` 객체입니다. 가장 복잡한 인수이며 기본적인 리졸버에서는 거의 필요하지 않습니다. 다음을 포함합니다:

- `fieldName`: 현재 필드 이름
- `returnType`: 필드의 GraphQL 타입
- `parentType`: 부모 타입
- `path`: 루트에서 이 필드까지의 경로
- `fieldNodes`: 이 필드의 선택 집합에 대한 AST 노드들
- `schema`: 전체 GraphQL 스키마 객체

`info`는 섹션 6에서 더 자세히 다룹니다.

## 3. 기본 리졸버

GraphQL에는 가장 일반적인 경우를 처리하는 내장 기본 리졸버가 있습니다: 부모 객체에서 속성을 추출하는 것입니다.

```javascript
// 이것이 기본 리졸버가 하는 일입니다:
function defaultResolver(parent, args, context, info) {
  return parent[info.fieldName];
}
```

따라서 단순한 속성 접근에는 리졸버를 작성할 필요가 없습니다:

```javascript
// 스키마
// type User {
//   id: ID!
//   name: String!
//   email: String!
// }

const resolvers = {
  Query: {
    user: (_, { id }, ctx) => ctx.db.users.findById(id),
    // 반환: { id: '1', name: 'Alice', email: 'alice@example.com' }
  },
  // User 리졸버 불필요!
  // 기본 리졸버가 User.id, User.name, User.email 처리
  // 데이터베이스 객체의 속성 이름이 일치하기 때문
};
```

**명시적 리졸버가 필요한 경우:**

```javascript
const resolvers = {
  User: {
    // 1. 계산 필드 (데이터 객체의 직접적인 속성이 아닌 경우)
    fullName: (parent) => `${parent.firstName} ${parent.lastName}`,

    // 2. 속성 이름 불일치 (DB 컬럼이 스키마 필드와 다른 경우)
    avatarUrl: (parent) => parent.avatar_url,

    // 3. 관계 (추가 데이터 조회가 필요한 경우)
    posts: (parent, args, ctx) => ctx.db.posts.findByAuthorId(parent.id),

    // 4. 인가 확인
    email: (parent, args, ctx) => {
      if (ctx.user?.id === parent.id || ctx.user?.role === 'ADMIN') {
        return parent.email;
      }
      return null; // 다른 사용자에게 이메일 숨김
    },
  },
};
```

## 4. 리졸버 체인

리졸버 체인(Resolver Chain)은 GraphQL이 중첩 필드를 해결하는 방식입니다. 각 리졸버는 데이터를 반환하고, 그 데이터는 다음 수준의 리졸버에 대한 `parent` 인수가 됩니다.

### 4.1 단계별 실행

다음 쿼리를 생각해 보세요:

```graphql
query {
  user(id: "1") {
    name
    posts(first: 2) {
      title
      comments(first: 3) {
        body
        author {
          name
        }
      }
    }
  }
}
```

실행 트레이스:

```
단계 1: Query.user({ id: "1" })
  → 반환: { id: "1", name: "Alice", ... }

단계 2: User.name(parent={ id: "1", name: "Alice", ... })
  → 반환: "Alice" (기본 리졸버: parent.name)

단계 3: User.posts(parent={ id: "1", ... }, args={ first: 2 })
  → 반환: [
      { id: "10", title: "Post A", authorId: "1" },
      { id: "11", title: "Post B", authorId: "1" },
    ]

단계 4a: Post.title(parent={ id: "10", title: "Post A", ... })
  → 반환: "Post A"

단계 4b: Post.title(parent={ id: "11", title: "Post B", ... })
  → 반환: "Post B"

단계 5a: Post.comments(parent={ id: "10", ... }, args={ first: 3 })
  → 반환: [
      { id: "100", body: "Great!", authorId: "2" },
      { id: "101", body: "Thanks!", authorId: "3" },
    ]

단계 5b: Post.comments(parent={ id: "11", ... }, args={ first: 3 })
  → 반환: [{ id: "102", body: "Nice post", authorId: "2" }]

단계 6: Comment.author(parent={ id: "100", ..., authorId: "2" })
  → 반환: { id: "2", name: "Bob", ... }

... 각 댓글에 대해 계속
```

### 4.2 트리 구조

```
Query.user ─────────────────────────────────┐
  ├── User.name          → "Alice"          │
  └── User.posts         → [Post, Post]     │
        ├── Post.title   → "Post A"         │
        ├── Post.comments → [Comment, ...]  │
        │     ├── Comment.body → "Great!"   │
        │     └── Comment.author ──────┐    │
        │           └── User.name      │    │
        ├── Post.title   → "Post B"    │    │
        └── Post.comments → [Comment]  │    │
              └── ...                   │    │
                                       │    │
각 화살표는 리졸버 함수를 호출합니다.   │    │
각 리졸버는 부모의 반환 값을 받습니다.  │    │
```

### 4.3 병렬 실행

같은 수준의 형제 필드들은 병렬로 실행될 수 있습니다:

```javascript
// User.name과 User.posts는 선택 집합에서 형제이므로
// 동시에 실행될 수 있습니다
const resolvers = {
  User: {
    name: async (parent) => parent.name,     // 이 두 개는
    posts: async (parent, _, ctx) => {       // 동시에 실행 가능
      return ctx.db.posts.findByAuthorId(parent.id);
    },
  },
};
```

GraphQL 실행 엔진은 형제 비동기 리졸버에 `Promise.all`을 사용합니다. 자식 리졸버는 부모가 완료될 때까지 기다립니다.

## 5. 컨텍스트 객체

컨텍스트(Context)는 GraphQL 리졸버를 위한 의존성 주입(Dependency Injection) 메커니즘입니다. 모든 리졸버가 접근할 수 있는 요청별 상태를 전달합니다.

### 5.1 컨텍스트 구성

```javascript
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { PrismaClient } from '@prisma/client';
import { createDataLoaders } from './dataloaders.js';
import { verifyToken } from './auth.js';

const prisma = new PrismaClient();

const server = new ApolloServer({ typeDefs, resolvers });

const { url } = await startStandaloneServer(server, {
  context: async ({ req }) => {
    // 1. 요청 헤더에서 사용자 인증
    const token = req.headers.authorization?.replace('Bearer ', '');
    let user = null;
    if (token) {
      try {
        user = await verifyToken(token);
      } catch (err) {
        // 토큰이 유효하지 않음; user는 null 유지 (미인증)
      }
    }

    // 2. 요청별 DataLoader 생성 (레슨 05)
    const loaders = createDataLoaders(prisma);

    // 3. 컨텍스트 객체 반환
    return {
      db: prisma,
      user,           // 인증된 사용자 (또는 null)
      loaders,        // DataLoader 인스턴스
      requestId: crypto.randomUUID(),  // 트레이싱/로깅용
    };
  },
});
```

### 5.2 리졸버에서 컨텍스트 사용

```javascript
const resolvers = {
  Query: {
    me: (_, __, context) => {
      // 인증 필요
      if (!context.user) {
        throw new GraphQLError('Not authenticated', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      return context.db.user.findUnique({ where: { id: context.user.id } });
    },
    posts: (_, args, context) => {
      return context.db.post.findMany({
        take: args.first,
        skip: args.offset,
        orderBy: { createdAt: 'desc' },
      });
    },
  },
  User: {
    posts: (parent, _, context) => {
      // 컨텍스트의 DataLoader 사용 (N+1 방지)
      return context.loaders.postsByAuthorId.load(parent.id);
    },
    email: (parent, _, context) => {
      // 인가: 자신이거나 관리자만 이메일 공개
      if (context.user?.id === parent.id || context.user?.role === 'ADMIN') {
        return parent.email;
      }
      return null;
    },
  },
  Post: {
    author: (parent, _, context) => {
      return context.loaders.userById.load(parent.authorId);
    },
  },
};
```

### 5.3 컨텍스트 설계 원칙

1. **요청마다 새롭게 생성**: 변경 가능한 상태를 요청 간에 절대 공유하지 않음
2. **평탄하게 유지**: 깊이 중첩된 컨텍스트 구조 피하기
3. **리졸버에 필요한 것만 포함**: 데이터베이스, 인증, 로더, 로거
4. **타입 지정**: TypeScript에서는 `Context` 인터페이스 정의

```typescript
// types.ts
interface Context {
  db: PrismaClient;
  user: AuthenticatedUser | null;
  loaders: DataLoaders;
  requestId: string;
}
```

## 6. Info 객체

`info` 파라미터(`GraphQLResolveInfo`)는 가장 적게 사용되지만 가장 강력한 리졸버 인수입니다. 전체 실행 컨텍스트를 제공합니다.

### 6.1 주요 속성

```javascript
const resolvers = {
  Query: {
    user: (parent, args, context, info) => {
      console.log(info.fieldName);     // "user"
      console.log(info.returnType);    // User (GraphQL 타입 객체)
      console.log(info.parentType);    // Query (GraphQL 타입 객체)
      console.log(info.path);          // { key: 'user', prev: undefined }

      // info.fieldNodes는 선택 집합의 AST를 담고 있습니다
      // 클라이언트가 정확히 어떤 필드를 요청했는지 알 수 있습니다
      return context.db.users.findById(args.id);
    },
  },
};
```

### 6.2 실제 활용: 필드 수준 최적화

`info` 객체를 통해 어떤 필드가 요청되었는지 검사하여 쿼리 최적화가 가능합니다:

```javascript
import { parseResolveInfo, simplifyParsedResolveInfoFragmentType } from 'graphql-parse-resolve-info';

const resolvers = {
  Query: {
    user: (parent, args, context, info) => {
      // 선택 집합을 파싱하여 어떤 필드가 요청되었는지 확인
      const parsedInfo = parseResolveInfo(info);
      const fields = simplifyParsedResolveInfoFragmentType(parsedInfo, info.returnType);

      // 'posts' 필드가 요청된 경우에만 'posts' 테이블 JOIN
      const includePosts = 'posts' in (fields.fields || {});

      return context.db.user.findUnique({
        where: { id: args.id },
        include: includePosts ? { posts: true } : {},
      });
    },
  },
};
```

### 6.3 Info를 사용해야 할 때

| 사용 사례 | 예시 |
|----------|------|
| SQL SELECT 최적화 | 쿼리된 컬럼만 SELECT |
| JOIN 최적화 | 관계가 요청된 경우에만 관련 테이블 JOIN |
| DataLoader 결정 | 관련 필드가 요청되지 않은 경우 로딩 건너뜀 |
| 쿼리 복잡도 분석 | AST에서 깊이/너비 계산 |
| 캐시 키 생성 | 선택 집합을 캐시 키의 일부로 사용 |

대부분의 애플리케이션에서는 `info`가 필요하지 않습니다. 데이터베이스 쿼리를 최적화하거나 프레임워크 수준의 도구를 구축할 때 관련성이 있습니다.

## 7. 비동기 리졸버

실제로 거의 모든 리졸버는 데이터베이스, 외부 API, 또는 다른 비동기 작업과 상호작용합니다. GraphQL은 비동기 리졸버를 기본적으로 지원합니다.

### 7.1 Promise 반환

```javascript
const resolvers = {
  Query: {
    // 리졸버가 Promise를 반환하면 GraphQL이 자동으로 await 처리
    user: (_, { id }, ctx) => {
      return ctx.db.user.findUnique({ where: { id } });
    },

    // async/await 문법 (동등한 표현)
    posts: async (_, args, ctx) => {
      const posts = await ctx.db.post.findMany({
        take: args.first,
        orderBy: { createdAt: 'desc' },
      });
      return posts;
    },

    // 여러 비동기 작업
    dashboard: async (_, __, ctx) => {
      const [userCount, postCount, commentCount] = await Promise.all([
        ctx.db.user.count(),
        ctx.db.post.count(),
        ctx.db.comment.count(),
      ]);
      return { userCount, postCount, commentCount };
    },
  },
};
```

### 7.2 리졸버 반환 타입

리졸버가 반환할 수 있는 것:

| 반환 값 | 동작 |
|---------|------|
| 스칼라 값 | 직접 사용 |
| 객체 | 자식 리졸버의 `parent`로 전달 |
| 배열 | 각 요소가 개별적으로 해결됨 |
| Promise | await 처리 후 위와 같이 처리 |
| `null` / `undefined` | 필드는 null (non-null 타입이면 오류) |
| 발생된 오류 | 필드는 null, errors 배열에 오류 추가 |

## 8. 리졸버 설계 패턴

### 8.1 얇은 리졸버 (Thin Resolver)

리졸버를 얇게 유지하세요. 비즈니스 로직이 아닌 서비스/데이터 레이어에 위임해야 합니다:

```javascript
// ❌ 뚱뚱한 리졸버: 비즈니스 로직과 데이터 접근이 혼재
const resolvers = {
  Mutation: {
    createPost: async (_, { input }, ctx) => {
      // 유효성 검사
      if (input.title.length < 3) throw new Error('Title too short');
      if (input.title.length > 200) throw new Error('Title too long');
      if (!input.body.trim()) throw new Error('Body is required');

      // 비즈니스 로직
      const slug = input.title.toLowerCase().replace(/\s+/g, '-');
      const existingSlug = await ctx.db.post.findFirst({ where: { slug } });
      const finalSlug = existingSlug ? `${slug}-${Date.now()}` : slug;

      // 데이터 접근
      const post = await ctx.db.post.create({
        data: {
          ...input,
          slug: finalSlug,
          authorId: ctx.user.id,
          publishedAt: input.publishNow ? new Date() : null,
        },
      });

      // 부수 효과
      await ctx.notificationService.notifyFollowers(ctx.user.id, post.id);
      await ctx.searchIndex.indexPost(post);

      return { post, errors: [] };
    },
  },
};

// ✅ 얇은 리졸버: 서비스 레이어에 위임
const resolvers = {
  Mutation: {
    createPost: async (_, { input }, ctx) => {
      return ctx.services.posts.create(input, ctx.user);
    },
  },
};
```

서비스 레이어(`PostService`)가 유효성 검사, 슬러그 생성, 데이터베이스 접근, 알림을 처리합니다. 이 분리는 서비스를 독립적으로 테스트 가능하고 재사용 가능하게 만듭니다.

### 8.2 계산된 데이터를 위한 필드 수준 리졸버

```javascript
const resolvers = {
  Post: {
    // 계산 필드: 읽기 시간
    readingTime: (parent) => {
      const wordsPerMinute = 200;
      const wordCount = parent.body.split(/\s+/).length;
      return Math.ceil(wordCount / wordsPerMinute);
    },

    // 계산 필드: 요약 (처음 200자)
    excerpt: (parent, { length = 200 }) => {
      if (parent.body.length <= length) return parent.body;
      return parent.body.substring(0, length).trimEnd() + '...';
    },

    // 조건부 필드: 인가 확인 필요
    viewCount: (parent, _, ctx) => {
      // 작성자와 관리자만 조회수를 볼 수 있음
      if (ctx.user?.id === parent.authorId || ctx.user?.role === 'ADMIN') {
        return parent.viewCount;
      }
      return null;
    },
  },
};
```

### 8.3 리졸버 맵 구조

```javascript
// 완전한 리졸버 맵 예시
const resolvers = {
  // 루트 타입
  Query: {
    me: (_, __, ctx) => ctx.user ? ctx.services.users.findById(ctx.user.id) : null,
    user: (_, { id }, ctx) => ctx.services.users.findById(id),
    posts: (_, args, ctx) => ctx.services.posts.findMany(args),
    search: (_, { query }, ctx) => ctx.services.search.execute(query),
  },

  Mutation: {
    createUser: (_, { input }, ctx) => ctx.services.users.create(input),
    updateUser: (_, { id, input }, ctx) => ctx.services.users.update(id, input, ctx.user),
    createPost: (_, { input }, ctx) => ctx.services.posts.create(input, ctx.user),
    deletePost: (_, { id }, ctx) => ctx.services.posts.delete(id, ctx.user),
  },

  // 타입 리졸버
  User: {
    posts: (parent, _, ctx) => ctx.loaders.postsByAuthorId.load(parent.id),
    followerCount: (parent, _, ctx) => ctx.loaders.followerCount.load(parent.id),
  },

  Post: {
    author: (parent, _, ctx) => ctx.loaders.userById.load(parent.authorId),
    comments: (parent, _, ctx) => ctx.loaders.commentsByPostId.load(parent.id),
    readingTime: (parent) => Math.ceil(parent.body.split(/\s+/).length / 200),
  },

  Comment: {
    author: (parent, _, ctx) => ctx.loaders.userById.load(parent.authorId),
  },

  // 추상 타입 리졸버
  SearchResult: {
    __resolveType(obj) {
      if (obj.username) return 'User';
      if (obj.title) return 'Post';
      if (obj.body && !obj.title) return 'Comment';
      return null;
    },
  },
};
```

## 9. 리졸버의 오류 처리

### 9.1 오류 발생

리졸버가 오류를 발생시키면, GraphQL이 오류를 잡아 해당 필드를 `null`로 설정하고 응답의 `errors` 배열에 오류를 추가합니다:

```javascript
import { GraphQLError } from 'graphql';

const resolvers = {
  Query: {
    user: async (_, { id }, ctx) => {
      const user = await ctx.db.user.findUnique({ where: { id } });
      if (!user) {
        throw new GraphQLError(`User with ID ${id} not found`, {
          extensions: {
            code: 'NOT_FOUND',
            argumentName: 'id',
          },
        });
      }
      return user;
    },
  },
  Mutation: {
    deletePost: async (_, { id }, ctx) => {
      if (!ctx.user) {
        throw new GraphQLError('You must be logged in', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      const post = await ctx.db.post.findUnique({ where: { id } });
      if (!post) {
        throw new GraphQLError('Post not found', {
          extensions: { code: 'NOT_FOUND' },
        });
      }
      if (post.authorId !== ctx.user.id && ctx.user.role !== 'ADMIN') {
        throw new GraphQLError('Not authorized to delete this post', {
          extensions: { code: 'FORBIDDEN' },
        });
      }
      await ctx.db.post.delete({ where: { id } });
      return { success: true };
    },
  },
};
```

### 9.2 오류 전파와 Null 가능성

레슨 02에서 기억하세요: non-null 필드는 null을 위로 전파합니다. 이것이 오류 처리에 영향을 줍니다:

```graphql
type Query {
  user(id: ID!): User       # Nullable → 오류가 여기에 머묾
}

type User {
  name: String!              # Non-null → 이것이 오류를 발생시키면 User는 null이 됨
  riskyField: String!        # Non-null → 이것이 오류를 발생시키면 User는 null이 됨
  safeField: String          # Nullable → 오류가 여기에 머묾
}
```

```json
// riskyField 리졸버가 오류를 발생시키면:
{
  "data": {
    "user": null          // riskyField가 String!이라 null 전파로 전체 user가 null이 됨
  },
  "errors": [{
    "message": "Something went wrong",
    "path": ["user", "riskyField"]
  }]
}
```

### 9.3 구조화된 오류 처리 패턴

뮤테이션에서는 발생된 예외 대신 오류를 데이터로 반환합니다:

```javascript
const resolvers = {
  Mutation: {
    createUser: async (_, { input }, ctx) => {
      // 유효성 검사
      const errors = [];

      if (input.username.length < 3) {
        errors.push({ field: 'username', message: 'Must be at least 3 characters' });
      }
      const existing = await ctx.db.user.findUnique({
        where: { email: input.email },
      });
      if (existing) {
        errors.push({ field: 'email', message: 'Email already in use' });
      }

      if (errors.length > 0) {
        return { user: null, errors };
      }

      // 생성
      const user = await ctx.db.user.create({ data: input });
      return { user, errors: [] };
    },
  },
};
```

이 접근 방식은:
- 200 HTTP 상태를 반환 (오류는 데이터이지 전송 실패가 아님)
- 단일 응답에서 여러 필드 수준 오류 허용
- 실제 실행 오류를 위해 최상위 `errors` 배열 유지

---

## 10. 연습 문제

### 연습 1: 리졸버 트레이싱 (초급)

이 스키마와 쿼리가 주어졌을 때, 모든 리졸버 호출을 실행 순서대로 나열하세요. 각 리졸버가 `parent`로 받는 인수를 표시하세요.

```graphql
type Query {
  book(id: ID!): Book
}

type Book {
  id: ID!
  title: String!
  author: Author!
  reviews: [Review!]!
}

type Author {
  id: ID!
  name: String!
}

type Review {
  id: ID!
  rating: Int!
  reviewer: String!
}
```

```graphql
query {
  book(id: "1") {
    title
    author {
      name
    }
    reviews {
      rating
      reviewer
    }
  }
}
```

데이터베이스에는:
- Book: `{ id: "1", title: "Clean Code", authorId: "a1" }`
- Author: `{ id: "a1", name: "Robert Martin" }`
- Reviews: `[{ id: "r1", rating: 5, reviewer: "Alice" }, { id: "r2", rating: 4, reviewer: "Bob" }]`

### 연습 2: 리졸버 작성 (중급)

이 스키마에 대한 완전한 리졸버 맵을 작성하세요. 데이터 접근에는 `context.db`를 사용하세요.

```graphql
type Query {
  product(id: ID!): Product
  products(category: String, minPrice: Float, maxPrice: Float): [Product!]!
}

type Product {
  id: ID!
  name: String!
  price: Float!
  category: String!
  inStock: Boolean!
  reviews: [Review!]!
  averageRating: Float       # reviews에서 계산
}

type Review {
  id: ID!
  rating: Int!
  comment: String
  author: User!
}

type User {
  id: ID!
  username: String!
}
```

`context.db`에는 다음 메서드가 있다고 가정합니다:
- `db.products.findById(id)`
- `db.products.findAll({ category, minPrice, maxPrice })`
- `db.reviews.findByProductId(productId)`
- `db.users.findById(userId)`

### 연습 3: 컨텍스트 설계 (중급)

다음이 필요한 블로깅 플랫폼의 컨텍스트 팩토리 함수를 설계하세요:

1. PostgreSQL 데이터베이스 접근 (Prisma 통해)
2. JWT 기반 인증
3. 자주 접근하는 데이터를 위한 Redis 캐시
4. 분산 트레이싱을 위한 요청 ID
5. 속도 제한 상태 (사용자당 분당 요청 수)

Apollo Server의 `startStandaloneServer`를 위한 `context` 함수를 작성하세요.

### 연습 4: 오류 처리 (중급)

다음을 수행하는 `updateProfile` 뮤테이션 리졸버를 작성하세요:

1. 인증 필요 (사용자 없으면 UNAUTHENTICATED 발생)
2. `displayName`이 2~50자 사이인지 유효성 검사
3. `bio`가 최대 500자인지 유효성 검사
4. 유효성 검사 실패 시 필드 수준 오류 반환 (발생된 오류 아님)
5. 성공 시 업데이트된 사용자 반환

구조화된 오류 처리 패턴(user + errors 배열을 가진 payload)을 사용하세요.

### 연습 5: 리졸버 최적화 (고급)

다음 리졸버에는 성능 문제가 있습니다. 모든 문제를 식별하고 재작성하세요:

```javascript
const resolvers = {
  Query: {
    posts: async (_, { first = 10 }, ctx) => {
      const posts = await ctx.db.post.findMany({ take: first });
      return posts;
    },
  },
  Post: {
    author: async (parent, _, ctx) => {
      const author = await ctx.db.user.findUnique({
        where: { id: parent.authorId },
      });
      return author;
    },
    comments: async (parent, _, ctx) => {
      const comments = await ctx.db.comment.findMany({
        where: { postId: parent.id },
      });
      return comments;
    },
    commentCount: async (parent, _, ctx) => {
      const comments = await ctx.db.comment.findMany({
        where: { postId: parent.id },
      });
      return comments.length;
    },
    likeCount: async (parent, _, ctx) => {
      const likes = await ctx.db.like.findMany({
        where: { postId: parent.id },
      });
      return likes.length;
    },
  },
  Comment: {
    author: async (parent, _, ctx) => {
      const author = await ctx.db.user.findUnique({
        where: { id: parent.authorId },
      });
      return author;
    },
  },
};
```

힌트: 이 리졸버에는 N+1 문제, 중복 쿼리, 비효율적인 카운팅이 있습니다. 레슨 05에서 DataLoader를 다루지만, 이미 문제를 식별하고 해결 방안을 제시할 수 있습니다.

---

## 11. 참고 자료

- GraphQL 실행 명세 - https://spec.graphql.org/October2021/#sec-Execution
- Apollo Server 리졸버 - https://www.apollographql.com/docs/apollo-server/data/resolvers/
- graphql-parse-resolve-info - https://github.com/graphile/graphile-engine/tree/master/packages/graphql-parse-resolve-info
- Marc-Andre Giroux, "Production Ready GraphQL" (2020) - Chapter 4: Resolver Design
- GraphQL 오류 처리 - https://www.apollographql.com/docs/apollo-server/data/errors/

---

**이전**: [쿼리와 뮤테이션](./03_Queries_and_Mutations.md) | **다음**: [DataLoader와 N+1](./05_DataLoader_N_plus_1.md)
