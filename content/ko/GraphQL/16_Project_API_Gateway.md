# 16. 프로젝트: API 게이트웨이(API Gateway)

**이전**: [REST에서 GraphQL로 마이그레이션](./15_REST_to_GraphQL_Migration.md)

## 학습 목표

이 프로젝트를 완료하면 다음을 할 수 있습니다:

1. 멀티 서비스 이커머스 플랫폼을 위한 연합(Federation) GraphQL API 게이트웨이 아키텍처 설계
2. 서비스 간 엔티티 참조를 포함한 세 개의 독립적인 서브그래프(Subgraph) — 사용자(Users), 상품(Products), 주문(Orders) — 구현
3. Apollo Router 설정으로 서브그래프를 통합 슈퍼그래프(Supergraph)로 합성(Compose)
4. 단일 요청으로 여러 서비스에서 데이터를 가져오는 크로스-서브그래프(Cross-Subgraph) 쿼리 실행
5. 에러 처리, 인증 전파, 배포 등 프로덕션 준비 패턴 적용

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [아키텍처](#2-아키텍처)
3. [서브그래프 1: 사용자 서비스](#3-서브그래프-1-사용자-서비스)
4. [서브그래프 2: 상품 서비스](#4-서브그래프-2-상품-서비스)
5. [서브그래프 3: 주문 서비스](#5-서브그래프-3-주문-서비스)
6. [게이트웨이 설정: Apollo Router](#6-게이트웨이-설정-apollo-router)
7. [크로스-서브그래프 쿼리](#7-크로스-서브그래프-쿼리)
8. [연합 그래프 테스트](#8-연합-그래프-테스트)
9. [배포 고려사항](#9-배포-고려사항)
10. [참고 자료](#10-참고-자료)

**난이도**: ⭐⭐⭐⭐

---

이 프로젝트는 GraphQL 토픽의 모든 내용을 하나의 프로덕션 수준 시스템으로 통합합니다. 세 개의 독립적인 서브그래프 — 사용자(Users), 상품(Products), 주문(Orders) — 으로 구성된 이커머스 플랫폼용 연합(Federation) API 게이트웨이를 구축합니다. 각 서브그래프는 자신의 도메인 데이터를 소유하고 Federation 2 디렉티브(Directive)를 통해 외부에 공개합니다. Apollo Router가 앞단에 위치하여 통합 그래프를 합성하고, 클라이언트는 마치 단일 서버에 쿼리하는 것처럼 사용할 수 있습니다.

프로젝트를 마치면 엔티티 해결(Entity Resolution), 서비스 간 데이터 페칭, 스키마 합성(Schema Composition), 그리고 이 모든 것을 클라이언트에게 투명하게 처리하는 쿼리 플래닝(Query Planning)을 시연하는 동작하는 연합 아키텍처를 갖게 됩니다.

---

## 1. 프로젝트 개요

### 무엇을 만드는가

"ShopGraph"를 위한 API 게이트웨이 — 간소화된 이커머스 플랫폼으로:

- **사용자(Users)**는 회원가입, 로그인, 프로필 관리를 할 수 있습니다
- **상품(Products)**은 카테고리별로 분류되며 사용자의 리뷰를 받습니다
- **주문(Orders)**은 사용자가 생성하며 상품 참조를 포함합니다

### 요구사항

| 요구사항 | 세부사항 |
|----------|---------|
| 서브그래프 3개 | 사용자(포트 4001), 상품(포트 4002), 주문(포트 4003) |
| 게이트웨이 | Apollo Router (포트 4000) |
| 엔티티 타입 | User, Product — 서브그래프 간 해결 |
| 인증 | 게이트웨이에서 서브그래프로 JWT 토큰 전파 |
| 데이터 | 인메모리(In-memory) 저장소 (이 프로젝트는 데이터베이스 불필요) |

### 기술 스택

- **런타임**: Node.js 20+
- **프레임워크**: Apollo Server 4 + @apollo/subgraph
- **게이트웨이**: Apollo Router (Rust 바이너리)
- **합성**: Rover CLI
- **언어**: TypeScript

---

## 2. 아키텍처

```
                           ┌────────────────────────┐
                           │        Clients          │
                           │  (Web, Mobile, Admin)   │
                           └───────────┬────────────┘
                                       │
                                       ▼
                           ┌────────────────────────┐
                           │    Apollo Router        │
                           │    port: 4000           │
                           │                         │
                           │  supergraph.graphql     │
                           │  (composed schema)      │
                           └──┬─────────┬─────────┬──┘
                              │         │         │
                    ┌─────────┘         │         └─────────┐
                    ▼                   ▼                   ▼
           ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
           │    Users      │   │   Products   │   │    Orders    │
           │  Subgraph     │   │  Subgraph    │   │  Subgraph    │
           │  port: 4001   │   │  port: 4002  │   │  port: 4003  │
           │               │   │              │   │              │
           │  User @key    │   │ Product @key │   │  Order @key  │
           │  Auth mutations│   │ Category     │   │  OrderItem   │
           │               │   │ Review       │   │              │
           └───────┬───────┘   └──────┬───────┘   └──────┬───────┘
                   │                  │                   │
                   ▼                  ▼                   ▼
           ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
           │  Users Store  │   │Products Store│   │ Orders Store │
           │  (in-memory)  │   │  (in-memory) │   │  (in-memory) │
           └──────────────┘   └──────────────┘   └──────────────┘
```

### 프로젝트 파일 구조

```
shopgraph/
├── subgraphs/
│   ├── users/
│   │   ├── index.ts
│   │   ├── schema.graphql
│   │   ├── resolvers.ts
│   │   └── data.ts
│   ├── products/
│   │   ├── index.ts
│   │   ├── schema.graphql
│   │   ├── resolvers.ts
│   │   └── data.ts
│   └── orders/
│       ├── index.ts
│       ├── schema.graphql
│       ├── resolvers.ts
│       └── data.ts
├── gateway/
│   ├── router.yaml
│   └── supergraph-config.yaml
├── package.json
├── tsconfig.json
└── README.md
```

---

## 3. 서브그래프 1: 사용자 서비스

사용자 서브그래프는 회원가입, 인증, 프로필 관리 등 사용자 관련 데이터를 모두 담당합니다.

### 스키마

```graphql
# subgraphs/users/schema.graphql
extend schema
  @link(url: "https://specs.apollo.dev/federation/v2.0",
        import: ["@key", "@shareable"])

type Query {
  """Get the currently authenticated user"""
  me: User
  """Look up a user by ID"""
  user(id: ID!): User
  """List all users (admin only)"""
  users: [User!]!
}

type Mutation {
  """Register a new user account"""
  register(input: RegisterInput!): AuthPayload!
  """Authenticate with email and password"""
  login(email: String!, password: String!): AuthPayload!
  """Update the current user's profile"""
  updateProfile(input: UpdateProfileInput!): User!
}

type User @key(fields: "id") {
  id: ID!
  name: String!
  email: String!
  role: UserRole!
  createdAt: String!
}

type AuthPayload {
  token: String!
  user: User!
}

input RegisterInput {
  name: String!
  email: String!
  password: String!
}

input UpdateProfileInput {
  name: String
  email: String
}

enum UserRole {
  CUSTOMER
  ADMIN
}
```

### 데이터 저장소

```typescript
// subgraphs/users/data.ts
export interface UserRecord {
  id: string;
  name: string;
  email: string;
  password: string;  // In production, this would be hashed
  role: 'CUSTOMER' | 'ADMIN';
  createdAt: string;
}

export const users: UserRecord[] = [
  {
    id: 'user-1',
    name: 'Alice Johnson',
    email: 'alice@example.com',
    password: 'password123',
    role: 'CUSTOMER',
    createdAt: '2024-01-15T10:00:00Z',
  },
  {
    id: 'user-2',
    name: 'Bob Smith',
    email: 'bob@example.com',
    password: 'password456',
    role: 'CUSTOMER',
    createdAt: '2024-02-20T14:30:00Z',
  },
  {
    id: 'user-3',
    name: 'Carol Admin',
    email: 'carol@example.com',
    password: 'adminpass',
    role: 'ADMIN',
    createdAt: '2024-01-01T00:00:00Z',
  },
];

let nextId = 4;

export function findUserById(id: string): UserRecord | undefined {
  return users.find((u) => u.id === id);
}

export function findUserByEmail(email: string): UserRecord | undefined {
  return users.find((u) => u.email === email);
}

export function createUser(input: {
  name: string;
  email: string;
  password: string;
}): UserRecord {
  const user: UserRecord = {
    id: `user-${nextId++}`,
    name: input.name,
    email: input.email,
    password: input.password,
    role: 'CUSTOMER',
    createdAt: new Date().toISOString(),
  };
  users.push(user);
  return user;
}
```

### 리졸버(Resolver)

```typescript
// subgraphs/users/resolvers.ts
import { GraphQLError } from 'graphql';
import { findUserById, findUserByEmail, createUser, users } from './data';

// Simple JWT simulation (in production, use jsonwebtoken)
function generateToken(userId: string): string {
  return Buffer.from(JSON.stringify({ userId, exp: Date.now() + 86400000 }))
    .toString('base64');
}

export const resolvers = {
  Query: {
    me: (_, __, { userId }) => {
      if (!userId) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      return findUserById(userId);
    },

    user: (_, { id }) => findUserById(id),

    users: (_, __, { userId }) => {
      const currentUser = findUserById(userId);
      if (!currentUser || currentUser.role !== 'ADMIN') {
        throw new GraphQLError('Admin access required', {
          extensions: { code: 'FORBIDDEN' },
        });
      }
      return users;
    },
  },

  Mutation: {
    register: (_, { input }) => {
      const existing = findUserByEmail(input.email);
      if (existing) {
        throw new GraphQLError('Email already registered', {
          extensions: { code: 'BAD_USER_INPUT' },
        });
      }

      const user = createUser(input);
      return { token: generateToken(user.id), user };
    },

    login: (_, { email, password }) => {
      const user = findUserByEmail(email);
      if (!user || user.password !== password) {
        throw new GraphQLError('Invalid credentials', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      return { token: generateToken(user.id), user };
    },

    updateProfile: (_, { input }, { userId }) => {
      if (!userId) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      const user = findUserById(userId);
      if (!user) throw new GraphQLError('User not found');

      if (input.name) user.name = input.name;
      if (input.email) user.email = input.email;

      return user;
    },
  },

  // Entity resolver: the router calls this when another subgraph
  // references a User entity by its key (id)
  User: {
    __resolveReference: (ref) => findUserById(ref.id),
  },
};
```

### 서버 엔트리 포인트

```typescript
// subgraphs/users/index.ts
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { buildSubgraphSchema } from '@apollo/subgraph';
import { readFileSync } from 'fs';
import { parse } from 'graphql';
import { resolvers } from './resolvers';

const typeDefs = parse(
  readFileSync('./subgraphs/users/schema.graphql', 'utf-8')
);

const server = new ApolloServer({
  schema: buildSubgraphSchema({ typeDefs, resolvers }),
});

startStandaloneServer(server, {
  listen: { port: 4001 },
  context: async ({ req }) => {
    // Extract userId from the Authorization header
    // The router propagates this header from the client
    const token = req.headers.authorization?.replace('Bearer ', '');
    let userId: string | null = null;

    if (token) {
      try {
        const decoded = JSON.parse(Buffer.from(token, 'base64').toString());
        userId = decoded.userId;
      } catch {
        // Invalid token -- proceed without userId
      }
    }

    return { userId };
  },
}).then(({ url }) => {
  console.log(`Users subgraph ready at ${url}`);
});
```

---

## 4. 서브그래프 2: 상품 서비스

상품 서브그래프는 상품 카탈로그, 카테고리, 고객 리뷰를 관리합니다.

### 스키마

```graphql
# subgraphs/products/schema.graphql
extend schema
  @link(url: "https://specs.apollo.dev/federation/v2.0",
        import: ["@key", "@external", "@provides"])

type Query {
  """Get a product by ID"""
  product(id: ID!): Product
  """List all products, optionally filtered by category"""
  products(categoryId: ID): [Product!]!
  """List all categories"""
  categories: [Category!]!
  """Search products by name"""
  searchProducts(query: String!): [Product!]!
}

type Mutation {
  """Add a review to a product (requires authentication)"""
  addReview(input: AddReviewInput!): Review!
}

type Product @key(fields: "id") {
  id: ID!
  name: String!
  description: String!
  price: Float!
  category: Category!
  imageUrl: String!
  inStock: Boolean!
  reviews: [Review!]!
  averageRating: Float
}

type Category {
  id: ID!
  name: String!
  products: [Product!]!
}

type Review @provides(fields: "author { name }") {
  id: ID!
  body: String!
  rating: Int!
  author: User!
  createdAt: String!
}

"""User entity stub -- the full User type lives in the Users subgraph"""
type User @key(fields: "id") {
  id: ID!
  name: String! @external
}

input AddReviewInput {
  productId: ID!
  body: String!
  rating: Int!
}
```

### 데이터 저장소

```typescript
// subgraphs/products/data.ts
export interface ProductRecord {
  id: string;
  name: string;
  description: string;
  price: number;
  categoryId: string;
  imageUrl: string;
  inStock: boolean;
}

export interface CategoryRecord {
  id: string;
  name: string;
}

export interface ReviewRecord {
  id: string;
  productId: string;
  userId: string;
  userName: string;  // Denormalized for @provides
  body: string;
  rating: number;
  createdAt: string;
}

export const categories: CategoryRecord[] = [
  { id: 'cat-1', name: 'Electronics' },
  { id: 'cat-2', name: 'Books' },
  { id: 'cat-3', name: 'Clothing' },
];

export const products: ProductRecord[] = [
  {
    id: 'prod-1',
    name: 'Wireless Headphones',
    description: 'Premium noise-canceling wireless headphones with 30-hour battery life.',
    price: 199.99,
    categoryId: 'cat-1',
    imageUrl: 'https://example.com/images/headphones.jpg',
    inStock: true,
  },
  {
    id: 'prod-2',
    name: 'GraphQL in Action',
    description: 'A comprehensive guide to building APIs with GraphQL.',
    price: 39.99,
    categoryId: 'cat-2',
    imageUrl: 'https://example.com/images/graphql-book.jpg',
    inStock: true,
  },
  {
    id: 'prod-3',
    name: 'Developer T-Shirt',
    description: 'Comfortable cotton t-shirt with "Query Everything" print.',
    price: 24.99,
    categoryId: 'cat-3',
    imageUrl: 'https://example.com/images/tshirt.jpg',
    inStock: false,
  },
  {
    id: 'prod-4',
    name: 'Mechanical Keyboard',
    description: 'Cherry MX Blue switches, RGB backlighting, compact 75% layout.',
    price: 149.99,
    categoryId: 'cat-1',
    imageUrl: 'https://example.com/images/keyboard.jpg',
    inStock: true,
  },
];

export const reviews: ReviewRecord[] = [
  {
    id: 'rev-1',
    productId: 'prod-1',
    userId: 'user-1',
    userName: 'Alice Johnson',
    body: 'Incredible sound quality and the noise canceling is top-notch.',
    rating: 5,
    createdAt: '2024-03-01T12:00:00Z',
  },
  {
    id: 'rev-2',
    productId: 'prod-1',
    userId: 'user-2',
    userName: 'Bob Smith',
    body: 'Good headphones but a bit tight for large heads.',
    rating: 4,
    createdAt: '2024-03-05T15:30:00Z',
  },
  {
    id: 'rev-3',
    productId: 'prod-2',
    userId: 'user-1',
    userName: 'Alice Johnson',
    body: 'Best GraphQL book I have read. Clear examples throughout.',
    rating: 5,
    createdAt: '2024-03-10T09:00:00Z',
  },
];

let nextReviewId = 4;

export function addReview(input: {
  productId: string;
  userId: string;
  userName: string;
  body: string;
  rating: number;
}): ReviewRecord {
  const review: ReviewRecord = {
    id: `rev-${nextReviewId++}`,
    productId: input.productId,
    userId: input.userId,
    userName: input.userName,
    body: input.body,
    rating: input.rating,
    createdAt: new Date().toISOString(),
  };
  reviews.push(review);
  return review;
}
```

### 리졸버(Resolver)

```typescript
// subgraphs/products/resolvers.ts
import { GraphQLError } from 'graphql';
import {
  products,
  categories,
  reviews,
  addReview,
} from './data';

export const resolvers = {
  Query: {
    product: (_, { id }) => products.find((p) => p.id === id),

    products: (_, { categoryId }) => {
      if (categoryId) {
        return products.filter((p) => p.categoryId === categoryId);
      }
      return products;
    },

    categories: () => categories,

    searchProducts: (_, { query }) => {
      const lowerQuery = query.toLowerCase();
      return products.filter(
        (p) =>
          p.name.toLowerCase().includes(lowerQuery) ||
          p.description.toLowerCase().includes(lowerQuery)
      );
    },
  },

  Mutation: {
    addReview: (_, { input }, { userId, userName }) => {
      if (!userId) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      const product = products.find((p) => p.id === input.productId);
      if (!product) {
        throw new GraphQLError('Product not found', {
          extensions: { code: 'BAD_USER_INPUT' },
        });
      }

      if (input.rating < 1 || input.rating > 5) {
        throw new GraphQLError('Rating must be between 1 and 5', {
          extensions: { code: 'BAD_USER_INPUT' },
        });
      }

      return addReview({
        productId: input.productId,
        userId,
        userName: userName || 'Unknown User',
        body: input.body,
        rating: input.rating,
      });
    },
  },

  Product: {
    // Entity resolver for cross-subgraph references
    __resolveReference: (ref) => products.find((p) => p.id === ref.id),

    category: (product) =>
      categories.find((c) => c.id === product.categoryId),

    reviews: (product) =>
      reviews.filter((r) => r.productId === product.id),

    averageRating: (product) => {
      const productReviews = reviews.filter((r) => r.productId === product.id);
      if (productReviews.length === 0) return null;

      const sum = productReviews.reduce((acc, r) => acc + r.rating, 0);
      return Math.round((sum / productReviews.length) * 10) / 10;
    },
  },

  Category: {
    products: (category) =>
      products.filter((p) => p.categoryId === category.id),
  },

  Review: {
    // Return the User entity reference so the router can resolve
    // the full User from the Users subgraph
    author: (review) => ({
      __typename: 'User',
      id: review.userId,
      name: review.userName, // Provided via @provides
    }),
  },
};
```

### 서버 엔트리 포인트

```typescript
// subgraphs/products/index.ts
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { buildSubgraphSchema } from '@apollo/subgraph';
import { readFileSync } from 'fs';
import { parse } from 'graphql';
import { resolvers } from './resolvers';

const typeDefs = parse(
  readFileSync('./subgraphs/products/schema.graphql', 'utf-8')
);

const server = new ApolloServer({
  schema: buildSubgraphSchema({ typeDefs, resolvers }),
});

startStandaloneServer(server, {
  listen: { port: 4002 },
  context: async ({ req }) => {
    const token = req.headers.authorization?.replace('Bearer ', '');
    let userId: string | null = null;
    let userName: string | null = null;

    if (token) {
      try {
        const decoded = JSON.parse(Buffer.from(token, 'base64').toString());
        userId = decoded.userId;
        // In production, you'd look up the user name or pass it in the token
        userName = decoded.userName || null;
      } catch {
        // Invalid token
      }
    }

    return { userId, userName };
  },
}).then(({ url }) => {
  console.log(`Products subgraph ready at ${url}`);
});
```

---

## 5. 서브그래프 3: 주문 서비스

주문 서브그래프는 주문 접수와 주문 내역을 처리합니다. 다른 서브그래프에 있는 사용자(User)와 상품(Product) 엔티티를 참조합니다.

### 스키마

```graphql
# subgraphs/orders/schema.graphql
extend schema
  @link(url: "https://specs.apollo.dev/federation/v2.0",
        import: ["@key", "@external", "@requires"])

type Query {
  """Get an order by ID"""
  order(id: ID!): Order
  """Get all orders for the authenticated user"""
  myOrders: [Order!]!
}

type Mutation {
  """Place a new order"""
  placeOrder(input: PlaceOrderInput!): Order!
  """Cancel an order (only if status is PENDING)"""
  cancelOrder(orderId: ID!): Order!
}

type Order @key(fields: "id") {
  id: ID!
  user: User!
  items: [OrderItem!]!
  status: OrderStatus!
  totalAmount: Float!
  createdAt: String!
}

type OrderItem {
  product: Product!
  quantity: Int!
  unitPrice: Float!
  lineTotal: Float!
}

"""User entity stub -- full type lives in Users subgraph"""
type User @key(fields: "id") {
  id: ID!
  """Orders placed by this user -- extends the User type across subgraphs"""
  orders: [Order!]!
}

"""Product entity stub -- full type lives in Products subgraph"""
type Product @key(fields: "id") {
  id: ID!
}

input PlaceOrderInput {
  items: [OrderItemInput!]!
}

input OrderItemInput {
  productId: ID!
  quantity: Int!
}

enum OrderStatus {
  PENDING
  CONFIRMED
  SHIPPED
  DELIVERED
  CANCELLED
}
```

### 데이터 저장소

```typescript
// subgraphs/orders/data.ts
export interface OrderRecord {
  id: string;
  userId: string;
  items: OrderItemRecord[];
  status: 'PENDING' | 'CONFIRMED' | 'SHIPPED' | 'DELIVERED' | 'CANCELLED';
  totalAmount: number;
  createdAt: string;
}

export interface OrderItemRecord {
  productId: string;
  quantity: number;
  unitPrice: number;
}

// Product price lookup (in production, this would call the products service)
const productPrices: Record<string, number> = {
  'prod-1': 199.99,
  'prod-2': 39.99,
  'prod-3': 24.99,
  'prod-4': 149.99,
};

export const orders: OrderRecord[] = [
  {
    id: 'order-1',
    userId: 'user-1',
    items: [
      { productId: 'prod-1', quantity: 1, unitPrice: 199.99 },
      { productId: 'prod-2', quantity: 2, unitPrice: 39.99 },
    ],
    status: 'DELIVERED',
    totalAmount: 279.97,
    createdAt: '2024-02-15T10:30:00Z',
  },
  {
    id: 'order-2',
    userId: 'user-2',
    items: [
      { productId: 'prod-4', quantity: 1, unitPrice: 149.99 },
    ],
    status: 'SHIPPED',
    totalAmount: 149.99,
    createdAt: '2024-03-01T16:45:00Z',
  },
  {
    id: 'order-3',
    userId: 'user-1',
    items: [
      { productId: 'prod-3', quantity: 3, unitPrice: 24.99 },
      { productId: 'prod-4', quantity: 1, unitPrice: 149.99 },
    ],
    status: 'PENDING',
    totalAmount: 224.96,
    createdAt: '2024-03-15T09:00:00Z',
  },
];

let nextOrderId = 4;

export function createOrder(
  userId: string,
  items: { productId: string; quantity: number }[]
): OrderRecord {
  const orderItems: OrderItemRecord[] = items.map((item) => {
    const unitPrice = productPrices[item.productId];
    if (!unitPrice) {
      throw new Error(`Unknown product: ${item.productId}`);
    }
    return {
      productId: item.productId,
      quantity: item.quantity,
      unitPrice,
    };
  });

  const totalAmount = orderItems.reduce(
    (sum, item) => sum + item.unitPrice * item.quantity,
    0
  );

  const order: OrderRecord = {
    id: `order-${nextOrderId++}`,
    userId,
    items: orderItems,
    status: 'PENDING',
    totalAmount: Math.round(totalAmount * 100) / 100,
    createdAt: new Date().toISOString(),
  };

  orders.push(order);
  return order;
}

export function findOrdersByUserId(userId: string): OrderRecord[] {
  return orders.filter((o) => o.userId === userId);
}
```

### 리졸버(Resolver)

```typescript
// subgraphs/orders/resolvers.ts
import { GraphQLError } from 'graphql';
import { orders, createOrder, findOrdersByUserId } from './data';

export const resolvers = {
  Query: {
    order: (_, { id }, { userId }) => {
      const order = orders.find((o) => o.id === id);
      if (!order) return null;

      // Users can only see their own orders (admins would bypass this)
      if (order.userId !== userId) {
        throw new GraphQLError('Access denied', {
          extensions: { code: 'FORBIDDEN' },
        });
      }

      return order;
    },

    myOrders: (_, __, { userId }) => {
      if (!userId) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }
      return findOrdersByUserId(userId);
    },
  },

  Mutation: {
    placeOrder: (_, { input }, { userId }) => {
      if (!userId) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      if (!input.items || input.items.length === 0) {
        throw new GraphQLError('Order must contain at least one item', {
          extensions: { code: 'BAD_USER_INPUT' },
        });
      }

      return createOrder(userId, input.items);
    },

    cancelOrder: (_, { orderId }, { userId }) => {
      if (!userId) {
        throw new GraphQLError('Authentication required', {
          extensions: { code: 'UNAUTHENTICATED' },
        });
      }

      const order = orders.find((o) => o.id === orderId);
      if (!order) {
        throw new GraphQLError('Order not found', {
          extensions: { code: 'BAD_USER_INPUT' },
        });
      }

      if (order.userId !== userId) {
        throw new GraphQLError('Access denied', {
          extensions: { code: 'FORBIDDEN' },
        });
      }

      if (order.status !== 'PENDING') {
        throw new GraphQLError(
          `Cannot cancel order with status ${order.status}`,
          { extensions: { code: 'BAD_USER_INPUT' } }
        );
      }

      order.status = 'CANCELLED';
      return order;
    },
  },

  Order: {
    __resolveReference: (ref) => orders.find((o) => o.id === ref.id),

    // Return a User entity reference for the router to resolve
    user: (order) => ({ __typename: 'User', id: order.userId }),
  },

  OrderItem: {
    // Return a Product entity reference
    product: (item) => ({ __typename: 'Product', id: item.productId }),

    lineTotal: (item) =>
      Math.round(item.unitPrice * item.quantity * 100) / 100,
  },

  // Extend the User entity with an orders field
  User: {
    orders: (user) => findOrdersByUserId(user.id),
  },
};
```

### 서버 엔트리 포인트

```typescript
// subgraphs/orders/index.ts
import { ApolloServer } from '@apollo/server';
import { startStandaloneServer } from '@apollo/server/standalone';
import { buildSubgraphSchema } from '@apollo/subgraph';
import { readFileSync } from 'fs';
import { parse } from 'graphql';
import { resolvers } from './resolvers';

const typeDefs = parse(
  readFileSync('./subgraphs/orders/schema.graphql', 'utf-8')
);

const server = new ApolloServer({
  schema: buildSubgraphSchema({ typeDefs, resolvers }),
});

startStandaloneServer(server, {
  listen: { port: 4003 },
  context: async ({ req }) => {
    const token = req.headers.authorization?.replace('Bearer ', '');
    let userId: string | null = null;

    if (token) {
      try {
        const decoded = JSON.parse(Buffer.from(token, 'base64').toString());
        userId = decoded.userId;
      } catch {
        // Invalid token
      }
    }

    return { userId };
  },
}).then(({ url }) => {
  console.log(`Orders subgraph ready at ${url}`);
});
```

---

## 6. 게이트웨이 설정: Apollo Router

### 슈퍼그래프 설정

```yaml
# gateway/supergraph-config.yaml
federation_version: =2.0.0
subgraphs:
  users:
    routing_url: http://localhost:4001
    schema:
      file: ../subgraphs/users/schema.graphql
  products:
    routing_url: http://localhost:4002
    schema:
      file: ../subgraphs/products/schema.graphql
  orders:
    routing_url: http://localhost:4003
    schema:
      file: ../subgraphs/orders/schema.graphql
```

### 슈퍼그래프 합성

```bash
# Install Rover CLI
npm install -g @apollo/rover

# Compose the supergraph schema
rover supergraph compose \
  --config gateway/supergraph-config.yaml \
  --output gateway/supergraph.graphql

# Verify composition succeeded
echo "Supergraph schema generated successfully"
```

### 라우터 설정

```yaml
# gateway/router.yaml
supergraph:
  path: ./supergraph.graphql

# Propagate the Authorization header to all subgraphs
headers:
  all:
    request:
      - propagate:
          named: Authorization

# CORS configuration for web clients
cors:
  origins:
    - http://localhost:3000
    - https://shopgraph.example.com
  allow_headers:
    - Content-Type
    - Authorization
    - X-Request-ID

# Traffic shaping
traffic_shaping:
  router:
    timeout: 30s
  all:
    timeout: 10s

# Telemetry (optional -- enable for observability)
telemetry:
  instrumentation:
    spans:
      router:
        attributes:
          http.request.header.x-request-id:
            request_header: x-request-id

# Sandbox explorer in development
sandbox:
  enabled: true
homepage:
  enabled: false

# Listen on port 4000
listen: 0.0.0.0:4000
```

### 게이트웨이 시작

```bash
# Download Apollo Router binary (one-time)
curl -sSL https://router.apollo.dev/download/nix/latest | sh

# Start the router
./router --config gateway/router.yaml

# Output:
# Apollo Router v1.x.x
# Listening on 0.0.0.0:4000
```

### 시작 스크립트

```bash
#!/bin/bash
# start.sh -- Start all services

echo "Starting ShopGraph..."

# Start subgraphs in background
npx tsx subgraphs/users/index.ts &
npx tsx subgraphs/products/index.ts &
npx tsx subgraphs/orders/index.ts &

# Wait for subgraphs to be ready
sleep 3

# Compose supergraph
rover supergraph compose \
  --config gateway/supergraph-config.yaml \
  --output gateway/supergraph.graphql

# Start router
./router --config gateway/router.yaml

echo "ShopGraph is ready at http://localhost:4000"
```

---

## 7. 크로스-서브그래프 쿼리

연합(Federation)의 진정한 강점은 투명한 크로스-서브그래프 해결입니다. 클라이언트는 모든 것이 단일 서버인 것처럼 쿼리를 작성합니다.

### 예제 1: 주문 및 상품 정보를 포함한 사용자 조회

```graphql
# Client sends this to the router at port 4000
query MyOrderHistory {
  me {
    name
    email
    orders {
      id
      status
      totalAmount
      createdAt
      items {
        quantity
        unitPrice
        lineTotal
        product {
          name
          price
          imageUrl
        }
      }
    }
  }
}
```

**라우터의 해결 과정:**

```
Step 1: Fetch from Users subgraph
  → me { id name email }
  → Returns: { id: "user-1", name: "Alice Johnson", email: "alice@example.com" }

Step 2: Fetch from Orders subgraph (using User entity reference)
  → _entities(representations: [{ __typename: "User", id: "user-1" }])
    { ... on User { orders { id status totalAmount createdAt items { quantity unitPrice lineTotal product { __typename id } } } } }
  → Returns: orders with product references

Step 3: Fetch from Products subgraph (using Product entity references)
  → _entities(representations: [{ __typename: "Product", id: "prod-1" }, { __typename: "Product", id: "prod-2" }, ...])
    { ... on Product { name price imageUrl } }
  → Returns: product details

Step 4: Assemble final response
  → Merges all data into the shape requested by the client
```

### 예제 2: 리뷰 및 작성자 정보를 포함한 상품 조회

```graphql
query ProductDetails($productId: ID!) {
  product(id: $productId) {
    name
    description
    price
    inStock
    averageRating
    category {
      name
    }
    reviews {
      body
      rating
      createdAt
      author {
        name
        email    # This comes from the Users subgraph
      }
    }
  }
}
```

**해결 흐름:**

```
Step 1: Products subgraph → product details + reviews + author stubs
  (author.name is provided via @provides, author.email needs Users subgraph)

Step 2: Users subgraph → _entities for review authors to get email
  (The router fetches only the fields not already provided)
```

### 예제 3: 주문 접수 후 전체 내역 조회

```graphql
mutation PlaceNewOrder {
  placeOrder(input: {
    items: [
      { productId: "prod-1", quantity: 1 },
      { productId: "prod-4", quantity: 2 }
    ]
  }) {
    id
    status
    totalAmount
    createdAt
    user {
      name
    }
    items {
      quantity
      unitPrice
      lineTotal
      product {
        name
        category {
          name
        }
      }
    }
  }
}
```

### 예제 4: 관리자 대시보드 (멀티 도메인)

```graphql
query AdminDashboard {
  users {
    id
    name
    role
    orders {
      id
      status
      totalAmount
    }
  }
  categories {
    name
    products {
      name
      price
      averageRating
    }
  }
}
```

이 단일 쿼리는 세 서브그래프 모두에 걸쳐 있으며, 라우터가 최적의 실행 순서를 계획하고 엔티티 참조를 일괄 처리합니다.

---

## 8. 연합 그래프 테스트

### 합성 테스트

배포 전에 서브그래프 스키마들이 오류 없이 합성되는지 검증합니다:

```bash
# Test composition locally
rover supergraph compose \
  --config gateway/supergraph-config.yaml \
  --output /dev/null

# Expected: successfully composed
# If there are errors, Rover will list them with explanations
```

### 서브그래프 단위 테스트

`executeOperation`을 사용하여 각 서브그래프를 독립적으로 테스트합니다:

```typescript
// __tests__/users-subgraph.test.ts
import { ApolloServer } from '@apollo/server';
import { buildSubgraphSchema } from '@apollo/subgraph';
import { describe, it, expect, beforeAll } from 'vitest';

describe('Users Subgraph', () => {
  let server: ApolloServer;

  beforeAll(() => {
    server = new ApolloServer({
      schema: buildSubgraphSchema({ typeDefs, resolvers }),
    });
  });

  it('resolves _entities for User', async () => {
    // This is what the router sends to resolve User entities
    const response = await server.executeOperation({
      query: `
        query ($representations: [_Any!]!) {
          _entities(representations: $representations) {
            ... on User {
              id
              name
              email
            }
          }
        }
      `,
      variables: {
        representations: [
          { __typename: 'User', id: 'user-1' },
          { __typename: 'User', id: 'user-2' },
        ],
      },
    });

    if (response.body.kind === 'single') {
      const entities = response.body.singleResult.data?._entities;
      expect(entities).toHaveLength(2);
      expect(entities[0].name).toBe('Alice Johnson');
      expect(entities[1].name).toBe('Bob Smith');
    }
  });

  it('handles login mutation', async () => {
    const response = await server.executeOperation({
      query: `
        mutation Login($email: String!, $password: String!) {
          login(email: $email, password: $password) {
            token
            user { id name }
          }
        }
      `,
      variables: { email: 'alice@example.com', password: 'password123' },
    });

    if (response.body.kind === 'single') {
      expect(response.body.singleResult.errors).toBeUndefined();
      expect(response.body.singleResult.data?.login.user.name).toBe('Alice Johnson');
      expect(response.body.singleResult.data?.login.token).toBeDefined();
    }
  });
});
```

### 라우터를 통한 엔드-투-엔드(E2E) 테스트

```typescript
// __tests__/e2e/gateway.test.ts
import { describe, it, expect } from 'vitest';

const GATEWAY_URL = 'http://localhost:4000';

describe('Gateway E2E Tests', () => {
  it('executes a cross-subgraph query', async () => {
    const response = await fetch(GATEWAY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: `
          query {
            product(id: "prod-1") {
              name
              price
              reviews {
                body
                rating
                author {
                  name
                  email
                }
              }
            }
          }
        `,
      }),
    });

    const { data, errors } = await response.json();

    expect(errors).toBeUndefined();
    expect(data.product.name).toBe('Wireless Headphones');
    expect(data.product.reviews).toHaveLength(2);
    // author.name comes from Products subgraph (@provides)
    // author.email comes from Users subgraph (entity resolution)
    expect(data.product.reviews[0].author.email).toBe('alice@example.com');
  });

  it('places an order and queries full details', async () => {
    // Step 1: Login to get a token
    const loginResponse = await fetch(GATEWAY_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: `
          mutation {
            login(email: "alice@example.com", password: "password123") {
              token
            }
          }
        `,
      }),
    });

    const { data: loginData } = await loginResponse.json();
    const token = loginData.login.token;

    // Step 2: Place an order with auth
    const orderResponse = await fetch(GATEWAY_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        query: `
          mutation {
            placeOrder(input: {
              items: [{ productId: "prod-2", quantity: 1 }]
            }) {
              id
              status
              totalAmount
              items {
                product { name }
                quantity
              }
            }
          }
        `,
      }),
    });

    const { data: orderData, errors } = await orderResponse.json();

    expect(errors).toBeUndefined();
    expect(orderData.placeOrder.status).toBe('PENDING');
    expect(orderData.placeOrder.totalAmount).toBe(39.99);
    expect(orderData.placeOrder.items[0].product.name).toBe('GraphQL in Action');
  });
});
```

---

## 9. 배포 고려사항

### 컨테이너 설정

```yaml
# docker-compose.yml
version: '3.8'

services:
  users-subgraph:
    build:
      context: .
      dockerfile: Dockerfile.subgraph
    command: npx tsx subgraphs/users/index.ts
    ports:
      - "4001:4001"
    environment:
      - NODE_ENV=production

  products-subgraph:
    build:
      context: .
      dockerfile: Dockerfile.subgraph
    command: npx tsx subgraphs/products/index.ts
    ports:
      - "4002:4002"
    environment:
      - NODE_ENV=production

  orders-subgraph:
    build:
      context: .
      dockerfile: Dockerfile.subgraph
    command: npx tsx subgraphs/orders/index.ts
    ports:
      - "4003:4003"
    environment:
      - NODE_ENV=production

  router:
    image: ghcr.io/apollographql/router:v1.40.0
    ports:
      - "4000:4000"
    volumes:
      - ./gateway/router.yaml:/dist/config/router.yaml
      - ./gateway/supergraph.graphql:/dist/config/supergraph.graphql
    command:
      - --config
      - /dist/config/router.yaml
      - --supergraph
      - /dist/config/supergraph.graphql
    depends_on:
      - users-subgraph
      - products-subgraph
      - orders-subgraph
```

### 프로덕션 체크리스트

| 항목 | 상태 | 비고 |
|------|------|------|
| 인트로스펙션 비활성화 | 필수 | 프로덕션에서 `introspection: false` 설정 |
| 지속 쿼리(Persisted Queries) 활성화 | 권장 | Redis 캐시와 함께 APQ 사용 |
| 쿼리 깊이 제한 설정 | 필수 | 이커머스는 최대 10-12 권장 |
| 복잡도 제한 설정 | 필수 | 1000-5000 예산 |
| 헬스 체크 설정 | 필수 | 각 서브그래프에 `/health` 엔드포인트 노출 |
| 모니터링 설정 | 필수 | Prometheus 메트릭, 분산 트레이싱 |
| CORS 설정 | 필수 | 프로덕션 도메인 허용 목록 지정 |
| HTTPS 활성화 | 필수 | 로드 밸런서에서 TLS 종료 |
| 요청 크기 제한 설정 | 필수 | 최대 본문(Body) 크기 100KB |
| 타임아웃 설정 | 필수 | 라우터 30초, 서브그래프별 10초 |

### 헬스 체크 엔드포인트

```typescript
// Add to each subgraph
import express from 'express';

const app = express();

app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'users-subgraph' });
});

// ... Apollo Server middleware on /graphql
```

### 스키마 변경을 위한 CI/CD 파이프라인

```yaml
# .github/workflows/schema-deploy.yml
name: Deploy Schema
on:
  push:
    branches: [main]
    paths:
      - 'subgraphs/**/schema.graphql'

jobs:
  check-and-publish:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        subgraph: [users, products, orders]
    steps:
      - uses: actions/checkout@v4

      - name: Install Rover
        run: |
          curl -sSL https://rover.apollo.dev/nix/latest | sh
          echo "$HOME/.rover/bin" >> $GITHUB_PATH

      - name: Check for breaking changes
        run: |
          rover subgraph check shopgraph@production \
            --name ${{ matrix.subgraph }} \
            --schema ./subgraphs/${{ matrix.subgraph }}/schema.graphql
        env:
          APOLLO_KEY: ${{ secrets.APOLLO_KEY }}

      - name: Publish schema
        if: success()
        run: |
          rover subgraph publish shopgraph@production \
            --name ${{ matrix.subgraph }} \
            --schema ./subgraphs/${{ matrix.subgraph }}/schema.graphql \
            --routing-url https://${{ matrix.subgraph }}.shopgraph.example.com/graphql
        env:
          APOLLO_KEY: ${{ secrets.APOLLO_KEY }}
```

---

## 10. 참고 자료

- [Apollo Federation 2 문서](https://www.apollographql.com/docs/federation/)
- [Apollo Router 문서](https://www.apollographql.com/docs/router/)
- [Rover CLI 레퍼런스](https://www.apollographql.com/docs/rover/)
- [Apollo Server 4 문서](https://www.apollographql.com/docs/apollo-server/)
- [Federation 명세](https://www.apollographql.com/docs/federation/federation-spec/)
- [Netflix DGS Federation](https://netflix.github.io/dgs/federation/)
- [GraphQL 모범 사례](https://graphql.org/learn/best-practices/)

---

**이전**: [REST에서 GraphQL로 마이그레이션](./15_REST_to_GraphQL_Migration.md)
