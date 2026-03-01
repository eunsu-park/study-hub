# 07. Authentication and Authorization

**Previous**: [Subscriptions](./06_Subscriptions.md) | **Next**: [Apollo Server](./08_Apollo_Server.md)

---

GraphQL has no built-in authentication or authorization mechanism. This is by design — GraphQL is a query language, not a security framework. Authentication (verifying *who* you are) and authorization (verifying *what* you can do) are application-level concerns that you implement through the context object, resolver logic, and custom directives. This lesson covers the patterns that have emerged as best practices in the GraphQL community.

**Difficulty**: ⭐⭐⭐

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement context-based authentication by extracting JWT tokens from HTTP headers
2. Apply authorization at resolver-level, type-level, and field-level granularity
3. Build a custom `@auth` schema directive for declarative access control
4. Implement role-based access control (RBAC) in a GraphQL API
5. Secure subscription connections using `connectionParams`

---

## Table of Contents

1. [Authentication vs Authorization](#1-authentication-vs-authorization)
2. [Context-Based Authentication](#2-context-based-authentication)
3. [Resolver-Level Authorization](#3-resolver-level-authorization)
4. [Field-Level Permissions](#4-field-level-permissions)
5. [Custom @auth Directive](#5-custom-auth-directive)
6. [Role-Based Access Control (RBAC)](#6-role-based-access-control-rbac)
7. [Auth in Subscriptions](#7-auth-in-subscriptions)
8. [Common Patterns and Pitfalls](#8-common-patterns-and-pitfalls)
9. [Practice Problems](#9-practice-problems)
10. [References](#10-references)

---

## 1. Authentication vs Authorization

These two concepts are often conflated, but they address different questions:

```
Authentication (AuthN)                Authorization (AuthZ)
────────────────────                  ────────────────────
"Who are you?"                        "What can you do?"

Verifies identity                     Verifies permissions
Happens once per request              Happens per field/operation
JWT, session, API key                 Roles, policies, ownership
Returns: User object or null          Returns: Allow or Deny

Example: Checking a JWT token         Example: "Only admins can
and finding that this request           delete users"
belongs to user #42
```

In GraphQL, authentication typically happens once in the context function (before any resolver runs), while authorization happens inside resolvers (or via directives) as each field is resolved.

---

## 2. Context-Based Authentication

The context function runs before every request and is the standard place to extract and verify authentication credentials. The returned context object is passed to every resolver.

### JWT Authentication

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

### Apollo Server Integration

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

Now every resolver has access to `context.currentUser`. If the token is invalid or missing, `currentUser` is `null` — the request is not rejected, because some operations may be public.

---

## 3. Resolver-Level Authorization

The simplest authorization pattern: check permissions at the beginning of each resolver.

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

### Extracting Auth Helpers

The repetitive auth checks can be extracted into helper functions:

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

## 4. Field-Level Permissions

Sometimes you want to show a type but hide specific fields. For example, all users can see a `User` object, but only the user themselves (or an admin) can see `email` or `phoneNumber`.

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

**Trade-off**: Returning `null` for unauthorized fields is simple but means the client cannot distinguish "the user has no phone number" from "you don't have permission to see the phone number." An alternative is to throw an error, but this aborts the entire response in some GraphQL clients. A middle ground is to use a union type:

```graphql
union EmailResult = EmailValue | Unauthorized

type EmailValue {
  value: String!
}

type Unauthorized {
  message: String!
}
```

This approach is more explicit but significantly increases schema complexity. In practice, returning `null` is the most common pattern.

---

## 5. Custom @auth Directive

Schema directives let you declare authorization rules in the schema itself, keeping resolvers clean. This is the most scalable pattern for large APIs.

### Schema Definition

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

### Directive Implementation (Apollo Server 4)

Apollo Server 4 uses the `@graphql-tools/utils` `mapSchema` approach for custom directives:

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

### Applying the Directive

```typescript
// src/index.ts
import { makeExecutableSchema } from '@graphql-tools/schema';
import { authDirectiveTransformer } from './directives/auth';

let schema = makeExecutableSchema({ typeDefs, resolvers });
schema = authDirectiveTransformer(schema);

const server = new ApolloServer({ schema });
```

Now any field or type decorated with `@auth` automatically checks the user's role before the resolver runs.

---

## 6. Role-Based Access Control (RBAC)

RBAC assigns permissions to roles, and roles to users. This is more maintainable than assigning permissions directly to users.

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

### Combining RBAC with Ownership

In most applications, a `USER` can update their own posts but not others'. This requires checking both role-based permissions and resource ownership:

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

## 7. Auth in Subscriptions

Subscriptions use WebSocket connections, which do not carry HTTP headers on each message. Instead, authentication data is sent once during the WebSocket handshake via `connectionParams`.

### Server-Side

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

### Client-Side

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

**Important**: If the user's token expires while a WebSocket connection is open, the connection remains alive. You need a strategy to handle this:

1. Set a reasonable connection timeout on the server
2. Periodically validate the token server-side and close stale connections
3. Have the client re-establish the connection when it obtains a new token

---

## 8. Common Patterns and Pitfalls

### Pattern: Authorization Middleware Layer

For large APIs, consider a middleware layer (like `graphql-shield`) that separates authorization logic from business logic entirely:

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

### Pitfall: Authorization in Child Resolvers

A common mistake is authorizing only the root query but not the child resolvers:

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

**Solution**: Apply field-level authorization on sensitive fields regardless of how they are reached. The `author` type resolver should check permissions for `secretField` independently of the parent query.

### Pitfall: Error Information Leakage

Do not reveal whether a resource exists when the user lacks permission:

```typescript
// BAD: Reveals that user #42 exists
throw new GraphQLError('You do not have permission to view user #42');

// GOOD: Does not reveal existence
throw new GraphQLError('Not found', {
  extensions: { code: 'NOT_FOUND' },
});
```

### Pitfall: Trusting Client-Side Authorization

Never rely solely on hiding UI elements for security. The GraphQL endpoint is accessible to anyone with an HTTP client. All authorization must be enforced server-side.

---

## 9. Practice Problems

### Exercise 1: Context Authentication (Beginner)

Write a `createContext` function that:

1. Extracts an API key from the `X-API-Key` header
2. Looks up the API key in a database to find the associated user and their permissions
3. Returns a context with `currentUser` (or `null` if the key is invalid)
4. Handles the case where the API key has expired (has an `expiresAt` field)

### Exercise 2: Multi-Tenant Authorization (Intermediate)

Design an authorization system for a multi-tenant SaaS application where:

- Users belong to one or more organizations
- Each user has a different role per organization (owner, admin, member, viewer)
- Data is scoped to organizations — a user must not see data from organizations they do not belong to
- Some queries accept an `orgId` argument; others infer it from context

Write the context function, authorization helpers, and resolver examples for `getProjects(orgId)` and `deleteProject(projectId)`.

### Exercise 3: Custom Directive with Permissions (Intermediate)

Extend the `@auth` directive to support fine-grained permissions instead of just roles:

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

Implement the directive transformer that checks both role hierarchy and specific permissions.

### Exercise 4: Subscription Auth with Token Refresh (Advanced)

Implement a complete subscription authentication system that handles token expiry:

1. Client sends a JWT in `connectionParams` when connecting
2. Server validates the JWT in `onConnect`
3. When the JWT expires (e.g., after 15 minutes), the server should close the connection with a specific error code
4. The client detects this closure, refreshes the token using a refresh token, and reconnects

Write both server-side and client-side code.

### Exercise 5: Rate Limiting by Role (Advanced)

Implement a rate limiter that applies different limits based on the user's role:

| Role | Queries/min | Mutations/min | Subscriptions (concurrent) |
|------|-------------|---------------|---------------------------|
| GUEST | 30 | 0 | 0 |
| USER | 100 | 30 | 5 |
| ADMIN | 1000 | 100 | 50 |

Write a context-based rate limiter using an in-memory store (e.g., `Map` with sliding window). Include the rate limit headers (`X-RateLimit-Remaining`, `X-RateLimit-Reset`) in the response.

---

## 10. References

- Apollo Server Authentication documentation — https://www.apollographql.com/docs/apollo-server/security/authentication
- graphql-shield — https://github.com/maticzav/graphql-shield
- GraphQL Authorization patterns — https://graphql.org/learn/authorization/
- JWT (RFC 7519) — https://datatracker.ietf.org/doc/html/rfc7519
- OWASP API Security Top 10 — https://owasp.org/API-Security/

---

**Previous**: [Subscriptions](./06_Subscriptions.md) | **Next**: [Apollo Server](./08_Apollo_Server.md)
