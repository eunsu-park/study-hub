# 07. 인증과 인가(Authentication and Authorization)

**이전**: [구독](./06_Subscriptions.md) | **다음**: [Apollo Server](./08_Apollo_Server.md)

---

GraphQL에는 내장된 인증(Authentication) 또는 인가(Authorization) 메커니즘이 없습니다. 이는 의도적인 설계입니다 — GraphQL은 쿼리 언어이지 보안 프레임워크가 아닙니다. 인증(내가 *누구인지* 확인)과 인가(*무엇을 할 수 있는지* 확인)는 컨텍스트 객체, 리졸버 로직, 커스텀 디렉티브를 통해 구현하는 애플리케이션 수준의 관심사입니다. 이 레슨에서는 GraphQL 커뮤니티에서 모범 사례로 자리 잡은 패턴들을 다룹니다.

**난이도**: ⭐⭐⭐

## 학습 목표

이 레슨을 완료하면 다음을 수행할 수 있습니다:

1. HTTP 헤더에서 JWT 토큰을 추출하여 컨텍스트 기반 인증을 구현할 수 있다
2. 리졸버 수준, 타입 수준, 필드 수준의 세분화된 인가를 적용할 수 있다
3. 선언적 접근 제어를 위한 커스텀 `@auth` 스키마 디렉티브를 구축할 수 있다
4. GraphQL API에서 역할 기반 접근 제어(RBAC, Role-Based Access Control)를 구현할 수 있다
5. `connectionParams`를 사용하여 구독 연결을 보안 처리할 수 있다

---

## 목차

1. [인증 vs 인가](#1-인증-vs-인가)
2. [컨텍스트 기반 인증](#2-컨텍스트-기반-인증)
3. [리졸버 수준 인가](#3-리졸버-수준-인가)
4. [필드 수준 권한](#4-필드-수준-권한)
5. [커스텀 @auth 디렉티브](#5-커스텀-auth-디렉티브)
6. [역할 기반 접근 제어 (RBAC)](#6-역할-기반-접근-제어-rbac)
7. [구독에서의 인증](#7-구독에서의-인증)
8. [일반적인 패턴과 함정](#8-일반적인-패턴과-함정)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. 인증 vs 인가

이 두 개념은 종종 혼용되지만 서로 다른 질문을 다룹니다:

```
인증(Authentication, AuthN)            인가(Authorization, AuthZ)
──────────────────────                  ────────────────────
"당신은 누구입니까?"                    "당신은 무엇을 할 수 있습니까?"

신원 확인                               권한 확인
요청당 한 번 수행                        필드/작업마다 수행
JWT, 세션, API 키                       역할, 정책, 소유권
반환: 사용자 객체 또는 null             반환: 허용 또는 거부

예시: JWT 토큰을 확인하여              예시: "관리자만
이 요청이 사용자 #42에게              사용자를 삭제할 수 있다"
속함을 알게 됨
```

GraphQL에서 인증은 일반적으로 컨텍스트 함수에서 한 번 처리되고(리졸버가 실행되기 전), 인가는 각 필드가 해석될 때 리졸버 내부(또는 디렉티브를 통해)에서 처리됩니다.

---

## 2. 컨텍스트 기반 인증

컨텍스트 함수는 모든 요청 전에 실행되며 인증 자격 증명을 추출하고 검증하는 표준 장소입니다. 반환된 컨텍스트 객체는 모든 리졸버에 전달됩니다.

### JWT 인증

```typescript
// src/context.ts
import jwt from 'jsonwebtoken';
import { prisma } from './db';

interface Context {
  currentUser: User | null;
  db: typeof prisma;
}

export async function createContext({ req }): Promise<Context> {
  const context: Context = {
    currentUser: null,
    db: prisma,
  };

  // Extract the token from the Authorization header
  const authHeader = req.headers.authorization;
  if (!authHeader) return context;

  // Expected format: "Bearer <token>"
  const token = authHeader.replace('Bearer ', '');
  if (!token) return context;

  try {
    // Verify and decode the JWT
    const decoded = jwt.verify(token, process.env.JWT_SECRET!) as {
      userId: string;
      role: string;
    };

    // Look up the user in the database
    // Why not just trust the JWT payload? Because the user might have been
    // deleted, banned, or had their role changed since the token was issued.
    const user = await prisma.user.findUnique({
      where: { id: decoded.userId },
    });

    if (user && user.isActive) {
      context.currentUser = user;
    }
  } catch (err) {
    // Invalid token — silently ignore.
    // The user will be treated as unauthenticated.
    // Do NOT throw here — some queries are public.
    console.warn('Invalid JWT:', err.message);
  }

  return context;
}
```

### Apollo Server 통합

```typescript
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { createContext } from './context';

const server = new ApolloServer({ schema });
await server.start();

app.use(
  '/graphql',
  cors(),
  express.json(),
  expressMiddleware(server, {
    // The context function receives the Express request object
    context: createContext,
  })
);
```

이제 모든 리졸버는 `context.currentUser`에 접근할 수 있습니다. 토큰이 유효하지 않거나 없으면 `currentUser`는 `null`이 됩니다 — 일부 작업은 공개적일 수 있으므로 요청이 거부되지는 않습니다.

---

## 3. 리졸버 수준 인가

가장 간단한 인가 패턴: 각 리졸버의 시작 부분에서 권한을 확인합니다.

```typescript
const resolvers = {
  Query: {
    // Public: anyone can query posts
    posts: async (_, __, { db }) => {
      return db.post.findMany({ where: { published: true } });
    },

    // Authenticated: must be logged in
    myProfile: async (_, __, { currentUser, db }) => {
      if (!currentUser) {
        throw new GraphQLError('You must be logged in', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      return db.user.findUnique({ where: { id: currentUser.id } });
    },

    // Authorized: must have admin role
    allUsers: async (_, __, { currentUser, db }) => {
      if (!currentUser) {
        throw new GraphQLError('You must be logged in', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      if (currentUser.role !== 'ADMIN') {
        throw new GraphQLError('Admin access required', {
          extensions: { code: 'FORBIDDEN' },
        });
      }
      return db.user.findMany();
    },
  },

  Mutation: {
    deletePost: async (_, { id }, { currentUser, db }) => {
      if (!currentUser) {
        throw new GraphQLError('You must be logged in', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      const post = await db.post.findUnique({ where: { id } });
      if (!post) {
        throw new GraphQLError('Post not found', {
          extensions: { code: 'NOT_FOUND' },
        });
      }

      // Ownership check: only the author or an admin can delete
      if (post.authorId !== currentUser.id && currentUser.role !== 'ADMIN') {
        throw new GraphQLError('You can only delete your own posts', {
          extensions: { code: 'FORBIDDEN' },
        });
      }

      return db.post.delete({ where: { id } });
    },
  },
};
```

### 인증 헬퍼 추출

반복적인 인증 확인은 헬퍼 함수로 추출할 수 있습니다:

```typescript
// src/auth.ts
import { GraphQLError } from 'graphql';

export function requireAuth(context: Context): User {
  if (!context.currentUser) {
    throw new GraphQLError('You must be logged in', {
      extensions: { code: 'UNAUTHENTICATED' },
    });
  }
  return context.currentUser;
}

export function requireRole(context: Context, ...roles: string[]): User {
  const user = requireAuth(context);
  if (!roles.includes(user.role)) {
    throw new GraphQLError(`Required role: ${roles.join(' or ')}`, {
      extensions: { code: 'FORBIDDEN' },
    });
  }
  return user;
}

export function requireOwnership(
  context: Context,
  resourceOwnerId: string
): User {
  const user = requireAuth(context);
  if (user.id !== resourceOwnerId && user.role !== 'ADMIN') {
    throw new GraphQLError('You do not have permission to access this resource', {
      extensions: { code: 'FORBIDDEN' },
    });
  }
  return user;
}
```

```typescript
// Usage in resolvers — much cleaner
const resolvers = {
  Query: {
    myProfile: async (_, __, ctx) => {
      const user = requireAuth(ctx);
      return ctx.db.user.findUnique({ where: { id: user.id } });
    },
    allUsers: async (_, __, ctx) => {
      requireRole(ctx, 'ADMIN');
      return ctx.db.user.findMany();
    },
  },
  Mutation: {
    deletePost: async (_, { id }, ctx) => {
      const post = await ctx.db.post.findUniqueOrThrow({ where: { id } });
      requireOwnership(ctx, post.authorId);
      return ctx.db.post.delete({ where: { id } });
    },
  },
};
```

---

## 4. 필드 수준 권한

때로는 타입은 보여주되 특정 필드를 숨기고 싶을 때가 있습니다. 예를 들어, 모든 사용자가 `User` 객체를 볼 수 있지만 사용자 본인(또는 관리자)만 `email`이나 `phoneNumber`를 볼 수 있어야 합니다.

```graphql
type User {
  id: ID!
  name: String!
  avatar: String
  email: String          # Only visible to self or admin
  phoneNumber: String    # Only visible to self or admin
  role: Role!            # Only visible to admin
  posts: [Post!]!        # Public
}
```

```typescript
const resolvers = {
  User: {
    // Field resolvers run after the parent resolver.
    // `parent` contains the full User object from the database.
    email: (parent, _, { currentUser }) => {
      // If the viewer is the user themselves, or an admin, show the email.
      // Otherwise, return null (the field appears in the response as null).
      if (currentUser?.id === parent.id || currentUser?.role === 'ADMIN') {
        return parent.email;
      }
      return null;
    },

    phoneNumber: (parent, _, { currentUser }) => {
      if (currentUser?.id === parent.id || currentUser?.role === 'ADMIN') {
        return parent.phoneNumber;
      }
      return null;
    },

    role: (parent, _, { currentUser }) => {
      if (currentUser?.role === 'ADMIN') {
        return parent.role;
      }
      return null;
    },
  },
};
```

**트레이드오프**: 권한 없는 필드에 `null`을 반환하는 것은 간단하지만, 클라이언트가 "사용자에게 전화번호가 없다"와 "전화번호를 볼 권한이 없다"를 구별할 수 없다는 의미입니다. 대안으로 오류를 던질 수 있지만, 이는 일부 GraphQL 클라이언트에서 전체 응답을 중단시킵니다. 중간 방법으로는 유니온 타입을 사용하는 것입니다:

```graphql
union EmailResult = EmailValue | Unauthorized

type EmailValue {
  value: String!
}

type Unauthorized {
  message: String!
}
```

이 방법은 더 명시적이지만 스키마 복잡성이 크게 증가합니다. 실제로는 `null`을 반환하는 것이 가장 일반적인 패턴입니다.

---

## 5. 커스텀 @auth 디렉티브

스키마 디렉티브를 사용하면 스키마 자체에 인가 규칙을 선언하여 리졸버를 깔끔하게 유지할 수 있습니다. 이것이 대규모 API에서 가장 확장성 있는 패턴입니다.

### 스키마 정의

```graphql
# Define the directive
directive @auth(requires: Role = USER) on FIELD_DEFINITION | OBJECT

enum Role {
  ADMIN
  MODERATOR
  USER
  GUEST
}

type Query {
  posts: [Post!]!                          # Public (no directive)
  myProfile: User! @auth                   # Requires USER (default)
  allUsers: [User!]! @auth(requires: ADMIN)  # Requires ADMIN
}

type Mutation {
  createPost(input: PostInput!): Post! @auth
  deleteUser(id: ID!): Boolean! @auth(requires: ADMIN)
}

type AdminDashboard @auth(requires: ADMIN) {
  totalUsers: Int!
  totalPosts: Int!
  revenueThisMonth: Float!
}
```

### 디렉티브 구현 (Apollo Server 4)

Apollo Server 4는 커스텀 디렉티브에 `@graphql-tools/utils`의 `mapSchema` 방식을 사용합니다:

```bash
npm install @graphql-tools/utils @graphql-tools/schema
```

```typescript
// src/directives/auth.ts
import { mapSchema, getDirective, MapperKind } from '@graphql-tools/utils';
import { GraphQLSchema, defaultFieldResolver, GraphQLError } from 'graphql';

const ROLE_HIERARCHY: Record<string, number> = {
  GUEST: 0,
  USER: 1,
  MODERATOR: 2,
  ADMIN: 3,
};

export function authDirectiveTransformer(schema: GraphQLSchema): GraphQLSchema {
  return mapSchema(schema, {
    // Handle @auth on object types
    [MapperKind.OBJECT_TYPE]: (type) => {
      const authDirective = getDirective(schema, type, 'auth')?.[0];
      if (authDirective) {
        // Store the required role on the type's extensions
        (type as any)._requiredRole = authDirective.requires || 'USER';
      }
      return type;
    },

    // Handle @auth on individual fields
    [MapperKind.OBJECT_FIELD]: (fieldConfig, fieldName, typeName) => {
      // Check for field-level directive first, then type-level
      const fieldDirective = getDirective(schema, fieldConfig, 'auth')?.[0];
      const typeDirective = (schema.getType(typeName) as any)?._requiredRole;

      const requiredRole = fieldDirective?.requires || typeDirective;
      if (!requiredRole) return fieldConfig;

      // Wrap the original resolver with an auth check
      const originalResolve = fieldConfig.resolve || defaultFieldResolver;

      fieldConfig.resolve = async (source, args, context, info) => {
        const { currentUser } = context;

        if (!currentUser) {
          throw new GraphQLError('Authentication required', {
            extensions: { code: 'UNAUTHENTICATED' },
          });
        }

        const userLevel = ROLE_HIERARCHY[currentUser.role] ?? 0;
        const requiredLevel = ROLE_HIERARCHY[requiredRole] ?? 0;

        if (userLevel < requiredLevel) {
          throw new GraphQLError(
            `Requires ${requiredRole} role (you have ${currentUser.role})`,
            { extensions: { code: 'FORBIDDEN' } }
          );
        }

        return originalResolve(source, args, context, info);
      };

      return fieldConfig;
    },
  });
}
```

### 디렉티브 적용

```typescript
// src/index.ts
import { makeExecutableSchema } from '@graphql-tools/schema';
import { authDirectiveTransformer } from './directives/auth';

let schema = makeExecutableSchema({ typeDefs, resolvers });
schema = authDirectiveTransformer(schema);

const server = new ApolloServer({ schema });
```

이제 `@auth`로 데코레이션된 모든 필드나 타입은 리졸버가 실행되기 전에 자동으로 사용자의 역할을 확인합니다.

---

## 6. 역할 기반 접근 제어 (RBAC)

RBAC(Role-Based Access Control)는 역할에 권한을 할당하고 사용자에게 역할을 부여합니다. 이는 권한을 사용자에게 직접 할당하는 것보다 더 유지 관리가 용이합니다.

```typescript
// src/permissions.ts

// Define granular permissions
type Permission =
  | 'post:read'
  | 'post:create'
  | 'post:update'
  | 'post:delete'
  | 'user:read'
  | 'user:update'
  | 'user:delete'
  | 'admin:dashboard'
  | 'admin:settings';

// Map roles to permissions
const ROLE_PERMISSIONS: Record<string, Permission[]> = {
  GUEST: ['post:read'],
  USER: ['post:read', 'post:create', 'post:update', 'user:read'],
  MODERATOR: [
    'post:read', 'post:create', 'post:update', 'post:delete',
    'user:read', 'user:update',
  ],
  ADMIN: [
    'post:read', 'post:create', 'post:update', 'post:delete',
    'user:read', 'user:update', 'user:delete',
    'admin:dashboard', 'admin:settings',
  ],
};

export function hasPermission(
  user: User | null,
  permission: Permission
): boolean {
  if (!user) return ROLE_PERMISSIONS['GUEST']?.includes(permission) ?? false;
  return ROLE_PERMISSIONS[user.role]?.includes(permission) ?? false;
}

export function requirePermission(
  context: Context,
  permission: Permission
): void {
  if (!hasPermission(context.currentUser, permission)) {
    throw new GraphQLError(`Missing permission: ${permission}`, {
      extensions: { code: 'FORBIDDEN' },
    });
  }
}
```

```typescript
// Usage in resolvers
const resolvers = {
  Mutation: {
    deletePost: async (_, { id }, ctx) => {
      requirePermission(ctx, 'post:delete');
      return ctx.db.post.delete({ where: { id } });
    },
    updateSettings: async (_, { input }, ctx) => {
      requirePermission(ctx, 'admin:settings');
      return ctx.db.settings.update({ data: input });
    },
  },
};
```

### RBAC와 소유권 결합

대부분의 애플리케이션에서 `USER`는 자신의 게시물을 수정할 수 있지만 다른 사람의 게시물은 수정할 수 없습니다. 이는 역할 기반 권한과 리소스 소유권을 모두 확인해야 합니다:

```typescript
const resolvers = {
  Mutation: {
    updatePost: async (_, { id, input }, ctx) => {
      requirePermission(ctx, 'post:update');

      const post = await ctx.db.post.findUniqueOrThrow({ where: { id } });

      // Users can only update their own posts.
      // Moderators and admins can update any post.
      const canUpdateAny = hasPermission(ctx.currentUser, 'post:delete');
      if (!canUpdateAny && post.authorId !== ctx.currentUser!.id) {
        throw new GraphQLError('You can only update your own posts', {
          extensions: { code: 'FORBIDDEN' },
        });
      }

      return ctx.db.post.update({ where: { id }, data: input });
    },
  },
};
```

---

## 7. 구독에서의 인증

구독은 WebSocket 연결을 사용하는데, 각 메시지마다 HTTP 헤더를 전달하지 않습니다. 대신 WebSocket 핸드셰이크 중에 `connectionParams`를 통해 인증 데이터를 한 번 전송합니다.

### 서버 측

```typescript
import { useServer } from 'graphql-ws/lib/use/ws';
import jwt from 'jsonwebtoken';

useServer(
  {
    schema,
    // onConnect runs once when the WebSocket connection is established.
    // Return false to reject the connection entirely.
    onConnect: async (ctx) => {
      const { authToken } = ctx.connectionParams || {};

      if (!authToken) {
        // Reject unauthenticated subscription connections
        return false;
      }

      try {
        jwt.verify(authToken, process.env.JWT_SECRET!);
      } catch {
        return false;
      }
    },

    // context runs for each subscription operation on this connection.
    context: async (ctx) => {
      const { authToken } = ctx.connectionParams || {};

      if (!authToken) return { currentUser: null };

      try {
        const decoded = jwt.verify(authToken, process.env.JWT_SECRET!) as {
          userId: string;
        };
        const user = await prisma.user.findUnique({
          where: { id: decoded.userId },
        });
        return { currentUser: user, db: prisma };
      } catch {
        return { currentUser: null, db: prisma };
      }
    },
  },
  wsServer
);
```

### 클라이언트 측

```typescript
import { createClient } from 'graphql-ws';

const wsClient = createClient({
  url: 'ws://localhost:4000/graphql',
  connectionParams: () => ({
    // This function is called on every (re)connection.
    // Use a function (not a static object) so the token
    // is always fresh after a reconnect.
    authToken: localStorage.getItem('token'),
  }),
});
```

**중요**: WebSocket 연결이 열려 있는 동안 사용자의 토큰이 만료되면 연결은 그대로 유지됩니다. 이를 처리하기 위한 전략이 필요합니다:

1. 서버에서 합리적인 연결 타임아웃 설정
2. 서버 측에서 주기적으로 토큰을 검증하고 오래된 연결 닫기
3. 새 토큰을 얻었을 때 클라이언트가 연결을 재구성

---

## 8. 일반적인 패턴과 함정

### 패턴: 인가 미들웨어 레이어

대규모 API의 경우, 인가 로직을 비즈니스 로직에서 완전히 분리하는 미들웨어 레이어(예: `graphql-shield`)를 고려하세요:

```typescript
import { shield, rule, and, or } from 'graphql-shield';

const isAuthenticated = rule()((_, __, ctx) => ctx.currentUser !== null);
const isAdmin = rule()((_, __, ctx) => ctx.currentUser?.role === 'ADMIN');
const isOwner = rule()((_, { id }, ctx) => {
  // ... check ownership
  return true;
});

const permissions = shield({
  Query: {
    myProfile: isAuthenticated,
    allUsers: isAdmin,
  },
  Mutation: {
    createPost: isAuthenticated,
    deletePost: or(isAdmin, isOwner),
    deleteUser: isAdmin,
  },
});
```

### 함정: 자식 리졸버에서의 인가

루트 쿼리만 인가하고 자식 리졸버는 인가하지 않는 흔한 실수:

```graphql
# The query requires auth, but what about User.secretField?
query {
  publicPost(id: "123") {
    title
    author {
      secretField  # If author resolver is public, this leaks data
    }
  }
}
```

**해결책**: 민감한 필드에는 도달 경로에 관계없이 필드 수준 인가를 적용하세요. `author` 타입 리졸버는 부모 쿼리와 독립적으로 `secretField`에 대한 권한을 확인해야 합니다.

### 함정: 오류 정보 누출

사용자가 권한이 없을 때 리소스의 존재 여부를 공개하지 마세요:

```typescript
// BAD: Reveals that user #42 exists
throw new GraphQLError('You do not have permission to view user #42');

// GOOD: Does not reveal existence
throw new GraphQLError('Not found', {
  extensions: { code: 'NOT_FOUND' },
});
```

### 함정: 클라이언트 측 인가 신뢰

보안을 위해 UI 요소를 숨기는 것에만 의존하지 마세요. GraphQL 엔드포인트는 HTTP 클라이언트를 가진 누구에게나 접근 가능합니다. 모든 인가는 서버 측에서 강제되어야 합니다.

---

## 9. 연습 문제

### 연습 1: 컨텍스트 인증 (입문)

다음을 수행하는 `createContext` 함수를 작성하세요:

1. `X-API-Key` 헤더에서 API 키 추출
2. 데이터베이스에서 API 키를 조회하여 관련 사용자와 권한 찾기
3. `currentUser`가 있는 컨텍스트 반환 (키가 유효하지 않으면 `null`)
4. API 키가 만료된 경우 처리 (`expiresAt` 필드가 있는 경우)

### 연습 2: 멀티 테넌트 인가 (중급)

다음 조건의 멀티 테넌트 SaaS 애플리케이션을 위한 인가 시스템을 설계하세요:

- 사용자는 하나 이상의 조직에 속함
- 각 사용자는 조직마다 다른 역할을 가짐 (owner, admin, member, viewer)
- 데이터는 조직 범위로 제한됨 — 사용자는 자신이 속하지 않은 조직의 데이터를 볼 수 없어야 함
- 일부 쿼리는 `orgId` 인수를 받고, 일부는 컨텍스트에서 추론

`getProjects(orgId)`와 `deleteProject(projectId)`에 대한 컨텍스트 함수, 인가 헬퍼, 리졸버 예시를 작성하세요.

### 연습 3: 권한이 있는 커스텀 디렉티브 (중급)

역할만이 아닌 세분화된 권한을 지원하도록 `@auth` 디렉티브를 확장하세요:

```graphql
directive @auth(
  requires: Role
  permission: String
) on FIELD_DEFINITION

type Mutation {
  createPost(input: PostInput!): Post! @auth(permission: "post:create")
  publishPost(id: ID!): Post! @auth(permission: "post:publish")
  deletePost(id: ID!): Boolean! @auth(permission: "post:delete")
}
```

역할 계층과 특정 권한 모두를 확인하는 디렉티브 트랜스포머를 구현하세요.

### 연습 4: 토큰 갱신이 있는 구독 인증 (고급)

토큰 만료를 처리하는 완전한 구독 인증 시스템을 구현하세요:

1. 클라이언트가 연결 시 `connectionParams`에 JWT를 전송
2. 서버가 `onConnect`에서 JWT를 검증
3. JWT가 만료되면(예: 15분 후) 서버는 특정 오류 코드와 함께 연결을 닫아야 함
4. 클라이언트가 이 종료를 감지하고, 리프레시 토큰을 사용하여 토큰을 갱신하고 재연결

서버 측 및 클라이언트 측 코드를 모두 작성하세요.

### 연습 5: 역할별 속도 제한 (고급)

사용자의 역할에 따라 다른 제한을 적용하는 속도 제한기를 구현하세요:

| 역할 | 쿼리/분 | 뮤테이션/분 | 구독 (동시) |
|------|--------|-----------|-----------|
| GUEST | 30 | 0 | 0 |
| USER | 100 | 30 | 5 |
| ADMIN | 1000 | 100 | 50 |

인메모리 저장소(예: 슬라이딩 윈도우가 있는 `Map`)를 사용하는 컨텍스트 기반 속도 제한기를 작성하세요. 응답에 속도 제한 헤더(`X-RateLimit-Remaining`, `X-RateLimit-Reset`)를 포함하세요.

---

## 10. 참고 자료

- Apollo Server 인증 문서 — https://www.apollographql.com/docs/apollo-server/security/authentication
- graphql-shield — https://github.com/maticzav/graphql-shield
- GraphQL 인가 패턴 — https://graphql.org/learn/authorization/
- JWT (RFC 7519) — https://datatracker.ietf.org/doc/html/rfc7519
- OWASP API 보안 Top 10 — https://owasp.org/API-Security/

---

**이전**: [구독](./06_Subscriptions.md) | **다음**: [Apollo Server](./08_Apollo_Server.md)
