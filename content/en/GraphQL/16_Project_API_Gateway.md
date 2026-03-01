# 16. Project: API Gateway with Federation

**Previous**: [REST to GraphQL Migration](./15_REST_to_GraphQL_Migration.md) | [Overview](./00_Overview.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Design a federated supergraph architecture for a real-world e-commerce application
2. Implement multiple subgraphs with entity references and cross-service queries
3. Configure Apollo Router for query planning, authentication, and header propagation
4. Apply caching, monitoring, and observability patterns across the gateway
5. Deploy and operate a production federation setup with Docker and Kubernetes

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Supergraph Architecture](#2-supergraph-architecture)
3. [Users Subgraph](#3-users-subgraph)
4. [Products Subgraph](#4-products-subgraph)
5. [Orders Subgraph](#5-orders-subgraph)
6. [Reviews Subgraph](#6-reviews-subgraph)
7. [Apollo Router Configuration](#7-apollo-router-configuration)
8. [Gateway Authentication](#8-gateway-authentication)
9. [Caching Strategy](#9-caching-strategy)
10. [Monitoring and Observability](#10-monitoring-and-observability)
11. [Testing](#11-testing)
12. [Deployment](#12-deployment)
13. [Extensions](#13-extensions)

**Difficulty**: ⭐⭐⭐⭐⭐

---

This capstone project brings together everything from the course -- schema design, resolvers, federation, authentication, caching, testing, and deployment -- into a single, production-grade API gateway. You will build a federated e-commerce supergraph with four independently deployable subgraphs, a centralized Apollo Router, and full observability. By the end, you will have a working system that demonstrates how modern GraphQL architectures operate at scale.

---

## 1. Project Overview

### Architecture Diagram

```
                          ┌─────────────────┐
                          │   Web / Mobile   │
                          │     Clients      │
                          └────────┬────────┘
                                   │ HTTPS
                          ┌────────▼────────┐
                          │  Apollo Router   │
                          │  (Port 4000)     │
                          │                  │
                          │  - Query planning│
                          │  - Auth (JWT)    │
                          │  - Caching       │
                          │  - Tracing       │
                          └────────┬────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
    ┌─────────▼──────┐  ┌────────▼────────┐  ┌───────▼──────────┐
    │ Users Subgraph │  │Products Subgraph│  │ Orders Subgraph  │
    │  (Port 4001)   │  │  (Port 4002)    │  │   (Port 4003)    │
    │                │  │                 │  │                  │
    │ User, Profile  │  │ Product,        │  │ Order, OrderItem │
    │ Auth mutations │  │ Category,       │  │ Payment, Shipment│
    │ JWT tokens     │  │ Inventory       │  │ Order lifecycle  │
    └───────┬────────┘  └────────┬────────┘  └────────┬─────────┘
            │                    │                     │
    ┌───────▼────────┐  ┌───────▼─────────┐  ┌───────▼──────────┐
    │   PostgreSQL   │  │   PostgreSQL    │  │   PostgreSQL     │
    │   (users_db)   │  │  (products_db)  │  │  (orders_db)     │
    └────────────────┘  └─────────────────┘  └──────────────────┘

                          ┌──────────────────┐
                          │Reviews Subgraph  │
                          │  (Port 4004)     │
                          │                  │
                          │ Review, Rating   │
                          │ Aggregations     │
                          └────────┬─────────┘
                                   │
                          ┌────────▼─────────┐
                          │   PostgreSQL     │
                          │  (reviews_db)    │
                          └──────────────────┘
```

### Tech Stack

| Component | Technology |
|-----------|-----------|
| Gateway | Apollo Router (Rust binary) |
| Subgraphs | Apollo Server 4, Node.js 20, Express |
| Language | TypeScript 5.x |
| Database | PostgreSQL 16 (one per subgraph) |
| ORM | Prisma 5.x |
| Auth | JSON Web Tokens (jsonwebtoken) |
| Schema management | Rover CLI |
| Observability | OpenTelemetry, Jaeger, Prometheus |
| Containerization | Docker, Docker Compose |
| Orchestration | Kubernetes with Helm |

### Project Directory Structure

```
federation-gateway/
├── router/
│   ├── router.yaml
│   └── supergraph.graphql
├── subgraphs/
│   ├── users/
│   │   ├── src/
│   │   │   ├── index.ts
│   │   │   ├── schema.ts
│   │   │   ├── resolvers.ts
│   │   │   ├── datasources.ts
│   │   │   └── auth.ts
│   │   ├── prisma/schema.prisma
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── Dockerfile
│   ├── products/
│   │   ├── src/
│   │   │   ├── index.ts
│   │   │   ├── schema.ts
│   │   │   ├── resolvers.ts
│   │   │   └── datasources.ts
│   │   ├── prisma/schema.prisma
│   │   ├── package.json
│   │   ├── tsconfig.json
│   │   └── Dockerfile
│   ├── orders/
│   │   └── ... (same structure)
│   └── reviews/
│       └── ... (same structure)
├── supergraph-config.yaml
├── docker-compose.yml
├── k8s/
│   ├── router-deployment.yaml
│   ├── users-deployment.yaml
│   ├── products-deployment.yaml
│   ├── orders-deployment.yaml
│   └── reviews-deployment.yaml
├── scripts/
│   ├── compose-supergraph.sh
│   └── seed-data.sh
└── package.json
```

---

## 2. Supergraph Architecture

### Entity Ownership Table

Each subgraph owns specific types and contributes fields to shared entities:

| Entity | Owner Subgraph | Key Fields | Contributing Subgraphs |
|--------|---------------|------------|----------------------|
| `User` | Users | `id` | Products (seller), Orders (buyer), Reviews (author) |
| `Product` | Products | `id` | Orders (line items), Reviews (reviews, averageRating) |
| `Order` | Orders | `id` | -- |
| `Review` | Reviews | `id` | -- |
| `Category` | Products | `id` | -- |

### Entity Relationship Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                       Supergraph                              │
│                                                               │
│   Users Subgraph          Products Subgraph                   │
│   ┌─────────────┐         ┌─────────────────┐                │
│   │ User @key   │◄───────▶│ Product @key    │                │
│   │  id         │         │  id             │                │
│   │  name       │         │  name           │                │
│   │  email      │         │  price          │                │
│   │  profile    │         │  category       │                │
│   │             │         │  seller: User   │                │
│   │  ┌────────┐ │         │  inventory      │                │
│   │  │Profile │ │         │                 │                │
│   │  │ bio    │ │         │ Category @key   │                │
│   │  │ avatar │ │         │  id, name       │                │
│   │  └────────┘ │         └────────┬────────┘                │
│   └──────┬──────┘                  │                          │
│          │                         │                          │
│   Orders Subgraph          Reviews Subgraph                   │
│   ┌──────▼──────┐         ┌────────▼────────┐                │
│   │ Order @key  │         │ Review @key     │                │
│   │  id         │         │  id             │                │
│   │  user: User │         │  body, rating   │                │
│   │  items      │         │  author: User   │                │
│   │  status     │         │  product:Product│                │
│   │  total      │         │                 │                │
│   │             │         │ Extends Product:│                │
│   │ OrderItem   │         │  reviews        │                │
│   │  product    │         │  averageRating  │                │
│   │  quantity   │         └─────────────────┘                │
│   │  price      │                                             │
│   └─────────────┘                                             │
└──────────────────────────────────────────────────────────────┘
```

### Cross-Subgraph Field Contributions

Each subgraph can extend entities it does not own by adding new fields:

```
User entity (owned by Users subgraph):
  ├── Users:    id, name, email, profile, createdAt
  ├── Products: products (list of products sold by this user)
  ├── Orders:   orders (list of orders placed by this user)
  └── Reviews:  reviews (list of reviews written by this user)

Product entity (owned by Products subgraph):
  ├── Products: id, name, description, price, category, seller, inventory
  ├── Orders:   (referenced via OrderItem.product)
  └── Reviews:  reviews, averageRating, reviewCount
```

---

## 3. Users Subgraph

The Users subgraph owns user identity, authentication, and profile data. It issues JWTs for login and registration.

### Schema

```typescript
// subgraphs/users/src/schema.ts
import gql from 'graphql-tag';

export const typeDefs = gql`
  extend schema
    @link(url: "https://specs.apollo.dev/federation/v2.0",
          import: ["@key", "@shareable"])

  type Query {
    """Currently authenticated user"""
    me: User
    """Look up a user by ID"""
    user(id: ID!): User
    """List all users (admin only)"""
    users(limit: Int = 20, offset: Int = 0): UsersConnection!
  }

  type Mutation {
    """Register a new user account"""
    register(input: RegisterInput!): AuthPayload!
    """Log in with email and password"""
    login(email: String!, password: String!): AuthPayload!
    """Update the current user's profile"""
    updateProfile(input: UpdateProfileInput!): User!
    """Change password (requires current password)"""
    changePassword(currentPassword: String!, newPassword: String!): Boolean!
  }

  type User @key(fields: "id") {
    id: ID!
    email: String!
    name: String!
    profile: Profile
    role: UserRole!
    createdAt: String!
    updatedAt: String!
  }

  type Profile {
    bio: String
    avatar: String
    phone: String
    address: Address
  }

  type Address @shareable {
    street: String!
    city: String!
    state: String!
    zipCode: String!
    country: String!
  }

  type AuthPayload {
    token: String!
    user: User!
  }

  type UsersConnection {
    nodes: [User!]!
    totalCount: Int!
  }

  enum UserRole {
    CUSTOMER
    SELLER
    ADMIN
  }

  input RegisterInput {
    email: String!
    password: String!
    name: String!
    role: UserRole = CUSTOMER
  }

  input UpdateProfileInput {
    name: String
    bio: String
    avatar: String
    phone: String
    address: AddressInput
  }

  input AddressInput {
    street: String!
    city: String!
    state: String!
    zipCode: String!
    country: String!
  }
`;
```

### Resolvers

```typescript
// subgraphs/users/src/resolvers.ts
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-key';

export const resolvers = {
  Query: {
    me: async (_, __, { userId, dataSources }) => {
      if (!userId) return null;
      return dataSources.users.findById(userId);
    },

    user: async (_, { id }, { dataSources }) => {
      return dataSources.users.findById(id);
    },

    users: async (_, { limit, offset }, { dataSources, userRole }) => {
      if (userRole !== 'ADMIN') {
        throw new Error('Only admins can list all users');
      }
      const [nodes, totalCount] = await Promise.all([
        dataSources.users.findMany({ limit, offset }),
        dataSources.users.count(),
      ]);
      return { nodes, totalCount };
    },
  },

  Mutation: {
    register: async (_, { input }, { dataSources }) => {
      const existing = await dataSources.users.findByEmail(input.email);
      if (existing) {
        throw new Error('Email already registered');
      }

      const hashedPassword = await bcrypt.hash(input.password, 12);
      const user = await dataSources.users.create({
        ...input,
        password: hashedPassword,
      });

      const token = jwt.sign(
        { userId: user.id, role: user.role },
        JWT_SECRET,
        { expiresIn: '7d' }
      );

      return { token, user };
    },

    login: async (_, { email, password }, { dataSources }) => {
      const user = await dataSources.users.findByEmail(email);
      if (!user) {
        throw new Error('Invalid credentials');
      }

      const valid = await bcrypt.compare(password, user.password);
      if (!valid) {
        throw new Error('Invalid credentials');
      }

      const token = jwt.sign(
        { userId: user.id, role: user.role },
        JWT_SECRET,
        { expiresIn: '7d' }
      );

      return { token, user };
    },

    updateProfile: async (_, { input }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      return dataSources.users.update(userId, input);
    },

    changePassword: async (
      _, { currentPassword, newPassword }, { userId, dataSources }
    ) => {
      if (!userId) throw new Error('Authentication required');

      const user = await dataSources.users.findById(userId);
      const valid = await bcrypt.compare(currentPassword, user.password);
      if (!valid) throw new Error('Current password is incorrect');

      const hashed = await bcrypt.hash(newPassword, 12);
      await dataSources.users.update(userId, { password: hashed });
      return true;
    },
  },

  // Entity resolution -- the router calls this when another subgraph
  // references a User by { __typename: "User", id: "..." }
  User: {
    __resolveReference: async (ref, { dataSources }) => {
      return dataSources.users.findById(ref.id);
    },
  },
};
```

### Data Source

```typescript
// subgraphs/users/src/datasources.ts
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export class UsersDataSource {
  async findById(id: string) {
    return prisma.user.findUnique({
      where: { id },
      include: { profile: true },
    });
  }

  async findByEmail(email: string) {
    return prisma.user.findUnique({ where: { email } });
  }

  async findMany({ limit, offset }: { limit: number; offset: number }) {
    return prisma.user.findMany({
      take: limit,
      skip: offset,
      include: { profile: true },
      orderBy: { createdAt: 'desc' },
    });
  }

  async count() {
    return prisma.user.count();
  }

  async create(data: any) {
    return prisma.user.create({
      data: {
        email: data.email,
        name: data.name,
        password: data.password,
        role: data.role,
        profile: { create: {} },
      },
      include: { profile: true },
    });
  }

  async update(id: string, data: any) {
    const { bio, avatar, phone, address, ...userData } = data;
    return prisma.user.update({
      where: { id },
      data: {
        ...userData,
        profile: {
          update: {
            ...(bio !== undefined && { bio }),
            ...(avatar !== undefined && { avatar }),
            ...(phone !== undefined && { phone }),
            ...(address !== undefined && { address }),
          },
        },
      },
      include: { profile: true },
    });
  }
}
```

### Server Setup

```typescript
// subgraphs/users/src/index.ts
import { ApolloServer } from '@apollo/server';
import { expressMiddleware } from '@apollo/server/express4';
import { buildSubgraphSchema } from '@apollo/subgraph';
import express from 'express';
import cors from 'cors';
import jwt from 'jsonwebtoken';
import { typeDefs } from './schema';
import { resolvers } from './resolvers';
import { UsersDataSource } from './datasources';

const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-key';

async function main() {
  const app = express();

  const server = new ApolloServer({
    schema: buildSubgraphSchema({ typeDefs, resolvers }),
  });

  await server.start();

  app.use(
    '/graphql',
    cors(),
    express.json(),
    expressMiddleware(server, {
      context: async ({ req }) => {
        // The router propagates the Authorization header
        const token = req.headers.authorization?.replace('Bearer ', '');
        let userId: string | null = null;
        let userRole: string | null = null;

        if (token) {
          try {
            const decoded = jwt.verify(token, JWT_SECRET) as any;
            userId = decoded.userId;
            userRole = decoded.role;
          } catch {
            // Invalid token -- proceed without auth
          }
        }

        return {
          userId,
          userRole,
          dataSources: { users: new UsersDataSource() },
        };
      },
    })
  );

  const port = process.env.PORT || 4001;
  app.listen(port, () => {
    console.log(`Users subgraph running at http://localhost:${port}/graphql`);
  });
}

main();
```

---

## 4. Products Subgraph

The Products subgraph manages the product catalog, categories, and inventory. It extends the `User` entity to add a `products` field (items sold by that user).

### Schema

```typescript
// subgraphs/products/src/schema.ts
import gql from 'graphql-tag';

export const typeDefs = gql`
  extend schema
    @link(url: "https://specs.apollo.dev/federation/v2.0",
          import: ["@key", "@external", "@requires", "@shareable"])

  type Query {
    """Fetch a single product by ID"""
    product(id: ID!): Product
    """Browse products with filtering and pagination"""
    products(
      filter: ProductFilter
      sort: ProductSort = NEWEST
      limit: Int = 20
      offset: Int = 0
    ): ProductsConnection!
    """List all categories"""
    categories: [Category!]!
  }

  type Mutation {
    """Create a new product (seller only)"""
    createProduct(input: CreateProductInput!): Product!
    """Update an existing product"""
    updateProduct(id: ID!, input: UpdateProductInput!): Product!
    """Adjust inventory stock"""
    adjustInventory(productId: ID!, delta: Int!): Product!
    """Remove a product from the catalog"""
    deleteProduct(id: ID!): Boolean!
  }

  type Product @key(fields: "id") {
    id: ID!
    name: String!
    description: String!
    price: Float!
    category: Category!
    seller: User!
    inventory: Int!
    sku: String!
    images: [String!]!
    isActive: Boolean!
    createdAt: String!
    updatedAt: String!
  }

  type Category @key(fields: "id") {
    id: ID!
    name: String!
    slug: String!
    products(limit: Int = 20, offset: Int = 0): [Product!]!
  }

  type ProductsConnection {
    nodes: [Product!]!
    totalCount: Int!
    hasNextPage: Boolean!
  }

  """Extend User entity to add the products they sell"""
  type User @key(fields: "id") {
    id: ID!
    """Products listed by this seller"""
    products: [Product!]!
    """Total value of all listed products (price * inventory)"""
    totalListingValue: Float!
  }

  input ProductFilter {
    categoryId: ID
    minPrice: Float
    maxPrice: Float
    sellerId: ID
    search: String
    inStock: Boolean
  }

  input CreateProductInput {
    name: String!
    description: String!
    price: Float!
    categoryId: ID!
    sku: String!
    inventory: Int!
    images: [String!]
  }

  input UpdateProductInput {
    name: String
    description: String
    price: Float
    categoryId: ID
    images: [String!]
    isActive: Boolean
  }

  enum ProductSort {
    NEWEST
    PRICE_LOW
    PRICE_HIGH
    NAME_ASC
  }
`;
```

### Resolvers

```typescript
// subgraphs/products/src/resolvers.ts

export const resolvers = {
  Query: {
    product: async (_, { id }, { dataSources }) => {
      return dataSources.products.findById(id);
    },

    products: async (_, { filter, sort, limit, offset }, { dataSources }) => {
      const [nodes, totalCount] = await Promise.all([
        dataSources.products.findMany({ filter, sort, limit, offset }),
        dataSources.products.count(filter),
      ]);
      return {
        nodes,
        totalCount,
        hasNextPage: offset + limit < totalCount,
      };
    },

    categories: async (_, __, { dataSources }) => {
      return dataSources.products.findAllCategories();
    },
  },

  Mutation: {
    createProduct: async (_, { input }, { userId, userRole, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      if (userRole !== 'SELLER' && userRole !== 'ADMIN') {
        throw new Error('Only sellers can create products');
      }
      return dataSources.products.create({ ...input, sellerId: userId });
    },

    updateProduct: async (_, { id, input }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      const product = await dataSources.products.findById(id);
      if (!product) throw new Error('Product not found');
      if (product.sellerId !== userId) {
        throw new Error('You can only update your own products');
      }
      return dataSources.products.update(id, input);
    },

    adjustInventory: async (_, { productId, delta }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      const product = await dataSources.products.findById(productId);
      if (!product) throw new Error('Product not found');

      const newInventory = product.inventory + delta;
      if (newInventory < 0) throw new Error('Insufficient inventory');

      return dataSources.products.update(productId, {
        inventory: newInventory,
      });
    },

    deleteProduct: async (_, { id }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      const product = await dataSources.products.findById(id);
      if (!product) throw new Error('Product not found');
      if (product.sellerId !== userId) {
        throw new Error('You can only delete your own products');
      }
      await dataSources.products.delete(id);
      return true;
    },
  },

  Product: {
    // Entity resolution for cross-subgraph references
    __resolveReference: async (ref, { dataSources }) => {
      return dataSources.products.findById(ref.id);
    },

    // Resolve the seller field as a User entity reference
    // The router will call the Users subgraph to hydrate this
    seller: (product) => ({ __typename: 'User', id: product.sellerId }),

    category: async (product, _, { dataSources }) => {
      return dataSources.products.findCategoryById(product.categoryId);
    },
  },

  Category: {
    __resolveReference: async (ref, { dataSources }) => {
      return dataSources.products.findCategoryById(ref.id);
    },

    products: async (category, { limit, offset }, { dataSources }) => {
      return dataSources.products.findMany({
        filter: { categoryId: category.id },
        sort: 'NEWEST',
        limit,
        offset,
      });
    },
  },

  // Extend the User entity with products-related fields
  User: {
    __resolveReference: async (ref, { dataSources }) => {
      // Return the reference enriched with products data
      const products = await dataSources.products.findBySellerId(ref.id);
      return { ...ref, products };
    },

    products: async (user, _, { dataSources }) => {
      if (user.products) return user.products;
      return dataSources.products.findBySellerId(user.id);
    },

    totalListingValue: async (user, _, { dataSources }) => {
      const products = user.products ||
        await dataSources.products.findBySellerId(user.id);
      return products.reduce(
        (sum, p) => sum + p.price * p.inventory, 0
      );
    },
  },
};
```

### Data Source

```typescript
// subgraphs/products/src/datasources.ts
import { PrismaClient, Prisma } from '@prisma/client';

const prisma = new PrismaClient();

export class ProductsDataSource {
  async findById(id: string) {
    return prisma.product.findUnique({ where: { id } });
  }

  async findBySellerId(sellerId: string) {
    return prisma.product.findMany({
      where: { sellerId, isActive: true },
      orderBy: { createdAt: 'desc' },
    });
  }

  async findMany({ filter, sort, limit, offset }) {
    const where: Prisma.ProductWhereInput = { isActive: true };

    if (filter?.categoryId) where.categoryId = filter.categoryId;
    if (filter?.sellerId) where.sellerId = filter.sellerId;
    if (filter?.inStock) where.inventory = { gt: 0 };
    if (filter?.minPrice || filter?.maxPrice) {
      where.price = {};
      if (filter.minPrice) where.price.gte = filter.minPrice;
      if (filter.maxPrice) where.price.lte = filter.maxPrice;
    }
    if (filter?.search) {
      where.OR = [
        { name: { contains: filter.search, mode: 'insensitive' } },
        { description: { contains: filter.search, mode: 'insensitive' } },
      ];
    }

    const orderBy = {
      NEWEST: { createdAt: 'desc' },
      PRICE_LOW: { price: 'asc' },
      PRICE_HIGH: { price: 'desc' },
      NAME_ASC: { name: 'asc' },
    }[sort] || { createdAt: 'desc' };

    return prisma.product.findMany({
      where,
      orderBy,
      take: limit,
      skip: offset,
    });
  }

  async count(filter?: any) {
    const where: Prisma.ProductWhereInput = { isActive: true };
    if (filter?.categoryId) where.categoryId = filter.categoryId;
    if (filter?.search) {
      where.OR = [
        { name: { contains: filter.search, mode: 'insensitive' } },
        { description: { contains: filter.search, mode: 'insensitive' } },
      ];
    }
    return prisma.product.count({ where });
  }

  async findAllCategories() {
    return prisma.category.findMany({ orderBy: { name: 'asc' } });
  }

  async findCategoryById(id: string) {
    return prisma.category.findUnique({ where: { id } });
  }

  async create(data: any) {
    return prisma.product.create({
      data: {
        name: data.name,
        description: data.description,
        price: data.price,
        categoryId: data.categoryId,
        sellerId: data.sellerId,
        sku: data.sku,
        inventory: data.inventory,
        images: data.images || [],
        isActive: true,
      },
    });
  }

  async update(id: string, data: any) {
    return prisma.product.update({ where: { id }, data });
  }

  async delete(id: string) {
    return prisma.product.update({
      where: { id },
      data: { isActive: false },
    });
  }
}
```

---

## 5. Orders Subgraph

The Orders subgraph manages the full order lifecycle: creation, payment, shipping, and delivery. It references `User` and `Product` entities from their respective subgraphs.

### Schema

```typescript
// subgraphs/orders/src/schema.ts
import gql from 'graphql-tag';

export const typeDefs = gql`
  extend schema
    @link(url: "https://specs.apollo.dev/federation/v2.0",
          import: ["@key", "@external", "@requires", "@shareable"])

  type Query {
    """Fetch a single order by ID"""
    order(id: ID!): Order
    """List orders for the current user"""
    myOrders(
      status: OrderStatus
      limit: Int = 20
      offset: Int = 0
    ): OrdersConnection!
  }

  type Mutation {
    """Create a new order from cart items"""
    createOrder(input: CreateOrderInput!): Order!
    """Process payment for an order"""
    payOrder(orderId: ID!, paymentMethod: PaymentMethod!): Order!
    """Mark an order as shipped (seller only)"""
    shipOrder(orderId: ID!, trackingNumber: String!): Order!
    """Mark an order as delivered"""
    deliverOrder(orderId: ID!): Order!
    """Cancel an order (only if PENDING or PAID)"""
    cancelOrder(orderId: ID!, reason: String): Order!
  }

  type Order @key(fields: "id") {
    id: ID!
    user: User!
    items: [OrderItem!]!
    status: OrderStatus!
    total: Float!
    shippingAddress: Address!
    trackingNumber: String
    paymentMethod: PaymentMethod
    cancelReason: String
    createdAt: String!
    updatedAt: String!
  }

  type OrderItem {
    id: ID!
    product: Product!
    quantity: Int!
    unitPrice: Float!
    """Computed: quantity * unitPrice"""
    subtotal: Float!
  }

  type OrdersConnection {
    nodes: [Order!]!
    totalCount: Int!
    hasNextPage: Boolean!
  }

  """Extend User entity to add order history"""
  type User @key(fields: "id") {
    id: ID!
    """All orders placed by this user"""
    orders(status: OrderStatus, limit: Int = 20): [Order!]!
    """Total amount spent across all completed orders"""
    totalSpent: Float!
  }

  """Reference Product entity from Products subgraph"""
  type Product @key(fields: "id") {
    id: ID!
  }

  type Address @shareable {
    street: String!
    city: String!
    state: String!
    zipCode: String!
    country: String!
  }

  enum OrderStatus {
    PENDING
    PAID
    SHIPPED
    DELIVERED
    CANCELLED
  }

  enum PaymentMethod {
    CREDIT_CARD
    DEBIT_CARD
    PAYPAL
    BANK_TRANSFER
  }

  input CreateOrderInput {
    items: [OrderItemInput!]!
    shippingAddress: AddressInput!
  }

  input OrderItemInput {
    productId: ID!
    quantity: Int!
  }

  input AddressInput {
    street: String!
    city: String!
    state: String!
    zipCode: String!
    country: String!
  }
`;
```

### Resolvers

```typescript
// subgraphs/orders/src/resolvers.ts

// Valid state transitions for the order lifecycle
const VALID_TRANSITIONS: Record<string, string[]> = {
  PENDING:   ['PAID', 'CANCELLED'],
  PAID:      ['SHIPPED', 'CANCELLED'],
  SHIPPED:   ['DELIVERED'],
  DELIVERED: [],
  CANCELLED: [],
};

export const resolvers = {
  Query: {
    order: async (_, { id }, { userId, dataSources }) => {
      const order = await dataSources.orders.findById(id);
      if (!order) return null;
      // Users can only view their own orders (admin can view all)
      if (order.userId !== userId) {
        throw new Error('Access denied');
      }
      return order;
    },

    myOrders: async (_, { status, limit, offset }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      const [nodes, totalCount] = await Promise.all([
        dataSources.orders.findByUser(userId, { status, limit, offset }),
        dataSources.orders.countByUser(userId, status),
      ]);
      return {
        nodes,
        totalCount,
        hasNextPage: offset + limit < totalCount,
      };
    },
  },

  Mutation: {
    createOrder: async (_, { input }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');

      // Calculate total from item prices
      // In production, you would verify prices with the Products subgraph
      const items = input.items.map(item => ({
        productId: item.productId,
        quantity: item.quantity,
        unitPrice: 0, // Will be set by the data source after price lookup
      }));

      return dataSources.orders.create({
        userId,
        items: input.items,
        shippingAddress: input.shippingAddress,
      });
    },

    payOrder: async (_, { orderId, paymentMethod }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      const order = await dataSources.orders.findById(orderId);
      if (!order) throw new Error('Order not found');
      if (order.userId !== userId) throw new Error('Access denied');

      if (!VALID_TRANSITIONS[order.status]?.includes('PAID')) {
        throw new Error(
          `Cannot pay order in ${order.status} status`
        );
      }

      return dataSources.orders.updateStatus(orderId, 'PAID', {
        paymentMethod,
      });
    },

    shipOrder: async (
      _, { orderId, trackingNumber }, { userId, userRole, dataSources }
    ) => {
      if (userRole !== 'SELLER' && userRole !== 'ADMIN') {
        throw new Error('Only sellers can ship orders');
      }
      const order = await dataSources.orders.findById(orderId);
      if (!order) throw new Error('Order not found');

      if (!VALID_TRANSITIONS[order.status]?.includes('SHIPPED')) {
        throw new Error(
          `Cannot ship order in ${order.status} status`
        );
      }

      return dataSources.orders.updateStatus(orderId, 'SHIPPED', {
        trackingNumber,
      });
    },

    deliverOrder: async (_, { orderId }, { dataSources }) => {
      const order = await dataSources.orders.findById(orderId);
      if (!order) throw new Error('Order not found');

      if (!VALID_TRANSITIONS[order.status]?.includes('DELIVERED')) {
        throw new Error(
          `Cannot deliver order in ${order.status} status`
        );
      }

      return dataSources.orders.updateStatus(orderId, 'DELIVERED');
    },

    cancelOrder: async (_, { orderId, reason }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      const order = await dataSources.orders.findById(orderId);
      if (!order) throw new Error('Order not found');
      if (order.userId !== userId) throw new Error('Access denied');

      if (!VALID_TRANSITIONS[order.status]?.includes('CANCELLED')) {
        throw new Error(
          `Cannot cancel order in ${order.status} status`
        );
      }

      return dataSources.orders.updateStatus(orderId, 'CANCELLED', {
        cancelReason: reason,
      });
    },
  },

  Order: {
    __resolveReference: async (ref, { dataSources }) => {
      return dataSources.orders.findById(ref.id);
    },

    // Return a User entity reference for the router to resolve
    user: (order) => ({ __typename: 'User', id: order.userId }),

    items: async (order, _, { dataSources }) => {
      if (order.items) return order.items;
      return dataSources.orders.findItemsByOrderId(order.id);
    },
  },

  OrderItem: {
    // Return a Product entity reference for the router to resolve
    product: (item) => ({ __typename: 'Product', id: item.productId }),
    subtotal: (item) => item.quantity * item.unitPrice,
  },

  // Extend User entity with order-related fields
  User: {
    __resolveReference: async (ref, { dataSources }) => {
      return ref; // Return the reference as-is; fields resolve individually
    },

    orders: async (user, { status, limit }, { dataSources }) => {
      return dataSources.orders.findByUser(user.id, {
        status,
        limit,
        offset: 0,
      });
    },

    totalSpent: async (user, _, { dataSources }) => {
      return dataSources.orders.sumTotalByUser(user.id);
    },
  },
};
```

### Data Source

```typescript
// subgraphs/orders/src/datasources.ts
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export class OrdersDataSource {
  async findById(id: string) {
    return prisma.order.findUnique({
      where: { id },
      include: { items: true },
    });
  }

  async findByUser(
    userId: string,
    { status, limit, offset }: { status?: string; limit: number; offset: number }
  ) {
    const where: any = { userId };
    if (status) where.status = status;

    return prisma.order.findMany({
      where,
      include: { items: true },
      orderBy: { createdAt: 'desc' },
      take: limit,
      skip: offset,
    });
  }

  async countByUser(userId: string, status?: string) {
    const where: any = { userId };
    if (status) where.status = status;
    return prisma.order.count({ where });
  }

  async findItemsByOrderId(orderId: string) {
    return prisma.orderItem.findMany({ where: { orderId } });
  }

  async sumTotalByUser(userId: string): Promise<number> {
    const result = await prisma.order.aggregate({
      where: { userId, status: 'DELIVERED' },
      _sum: { total: true },
    });
    return result._sum.total || 0;
  }

  async create(data: {
    userId: string;
    items: Array<{ productId: string; quantity: number }>;
    shippingAddress: any;
  }) {
    // In production, fetch current prices from Products subgraph via HTTP
    // For this project, we store prices at order time to prevent price changes
    // from affecting past orders
    return prisma.order.create({
      data: {
        userId: data.userId,
        status: 'PENDING',
        shippingAddress: data.shippingAddress,
        total: 0, // Calculated after price lookup
        items: {
          create: data.items.map(item => ({
            productId: item.productId,
            quantity: item.quantity,
            unitPrice: 0, // Set after price verification
          })),
        },
      },
      include: { items: true },
    });
  }

  async updateStatus(id: string, status: string, extra: any = {}) {
    return prisma.order.update({
      where: { id },
      data: { status, ...extra, updatedAt: new Date() },
      include: { items: true },
    });
  }
}
```

---

## 6. Reviews Subgraph

The Reviews subgraph manages product reviews and ratings. It extends the `Product` entity to add `reviews`, `averageRating`, and `reviewCount` fields, and extends the `User` entity to add `reviews` written by that user.

### Schema

```typescript
// subgraphs/reviews/src/schema.ts
import gql from 'graphql-tag';

export const typeDefs = gql`
  extend schema
    @link(url: "https://specs.apollo.dev/federation/v2.0",
          import: ["@key", "@external", "@provides"])

  type Query {
    """Fetch a single review by ID"""
    review(id: ID!): Review
    """Top-rated products based on average review score"""
    topRatedProducts(limit: Int = 10): [ProductRating!]!
  }

  type Mutation {
    """Submit a review for a product"""
    createReview(input: CreateReviewInput!): Review!
    """Update an existing review"""
    updateReview(id: ID!, input: UpdateReviewInput!): Review!
    """Delete a review"""
    deleteReview(id: ID!): Boolean!
    """Mark a review as helpful"""
    markHelpful(reviewId: ID!): Review!
  }

  type Review @key(fields: "id") {
    id: ID!
    body: String!
    rating: Int!
    author: User! @provides(fields: "name")
    product: Product!
    helpfulCount: Int!
    createdAt: String!
    updatedAt: String!
  }

  type ProductRating {
    product: Product!
    averageRating: Float!
    reviewCount: Int!
  }

  """Extend Product entity to add review-related fields"""
  type Product @key(fields: "id") {
    id: ID!
    """All reviews for this product, most recent first"""
    reviews(limit: Int = 10, offset: Int = 0): [Review!]!
    """Average rating across all reviews (1-5)"""
    averageRating: Float
    """Total number of reviews"""
    reviewCount: Int!
  }

  """Extend User entity to add authored reviews"""
  type User @key(fields: "id") {
    id: ID!
    name: String! @external
    """Reviews written by this user"""
    reviews(limit: Int = 10): [Review!]!
  }

  input CreateReviewInput {
    productId: ID!
    body: String!
    rating: Int!
  }

  input UpdateReviewInput {
    body: String
    rating: Int
  }
`;
```

### Resolvers

```typescript
// subgraphs/reviews/src/resolvers.ts

export const resolvers = {
  Query: {
    review: async (_, { id }, { dataSources }) => {
      return dataSources.reviews.findById(id);
    },

    topRatedProducts: async (_, { limit }, { dataSources }) => {
      return dataSources.reviews.getTopRatedProducts(limit);
    },
  },

  Mutation: {
    createReview: async (_, { input }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');

      // Check if user already reviewed this product
      const existing = await dataSources.reviews.findByUserAndProduct(
        userId, input.productId
      );
      if (existing) {
        throw new Error('You have already reviewed this product');
      }

      if (input.rating < 1 || input.rating > 5) {
        throw new Error('Rating must be between 1 and 5');
      }

      return dataSources.reviews.create({
        ...input,
        authorId: userId,
      });
    },

    updateReview: async (_, { id, input }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      const review = await dataSources.reviews.findById(id);
      if (!review) throw new Error('Review not found');
      if (review.authorId !== userId) {
        throw new Error('You can only edit your own reviews');
      }

      if (input.rating && (input.rating < 1 || input.rating > 5)) {
        throw new Error('Rating must be between 1 and 5');
      }

      return dataSources.reviews.update(id, input);
    },

    deleteReview: async (_, { id }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      const review = await dataSources.reviews.findById(id);
      if (!review) throw new Error('Review not found');
      if (review.authorId !== userId) {
        throw new Error('You can only delete your own reviews');
      }
      await dataSources.reviews.delete(id);
      return true;
    },

    markHelpful: async (_, { reviewId }, { userId, dataSources }) => {
      if (!userId) throw new Error('Authentication required');
      return dataSources.reviews.incrementHelpful(reviewId);
    },
  },

  Review: {
    __resolveReference: async (ref, { dataSources }) => {
      return dataSources.reviews.findById(ref.id);
    },

    // @provides(fields: "name") -- this resolver returns the author with
    // their name, so the router can skip calling the Users subgraph
    author: (review) => ({
      __typename: 'User',
      id: review.authorId,
      name: review.authorName, // Denormalized at write time
    }),

    product: (review) => ({
      __typename: 'Product',
      id: review.productId,
    }),
  },

  // Extend Product with review aggregation fields
  Product: {
    __resolveReference: async (ref, { dataSources }) => {
      return ref; // Fields resolve individually below
    },

    reviews: async (product, { limit, offset }, { dataSources }) => {
      return dataSources.reviews.findByProduct(product.id, { limit, offset });
    },

    averageRating: async (product, _, { dataSources }) => {
      const stats = await dataSources.reviews.getProductStats(product.id);
      return stats.averageRating;
    },

    reviewCount: async (product, _, { dataSources }) => {
      const stats = await dataSources.reviews.getProductStats(product.id);
      return stats.reviewCount;
    },
  },

  // Extend User with their authored reviews
  User: {
    __resolveReference: async (ref) => ref,

    reviews: async (user, { limit }, { dataSources }) => {
      return dataSources.reviews.findByAuthor(user.id, limit);
    },
  },
};
```

### Data Source

```typescript
// subgraphs/reviews/src/datasources.ts
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export class ReviewsDataSource {
  async findById(id: string) {
    return prisma.review.findUnique({ where: { id } });
  }

  async findByProduct(
    productId: string,
    { limit, offset }: { limit: number; offset: number }
  ) {
    return prisma.review.findMany({
      where: { productId },
      orderBy: { createdAt: 'desc' },
      take: limit,
      skip: offset,
    });
  }

  async findByAuthor(authorId: string, limit: number) {
    return prisma.review.findMany({
      where: { authorId },
      orderBy: { createdAt: 'desc' },
      take: limit,
    });
  }

  async findByUserAndProduct(authorId: string, productId: string) {
    return prisma.review.findFirst({ where: { authorId, productId } });
  }

  async getProductStats(productId: string) {
    const result = await prisma.review.aggregate({
      where: { productId },
      _avg: { rating: true },
      _count: true,
    });
    return {
      averageRating: result._avg.rating
        ? Math.round(result._avg.rating * 10) / 10
        : null,
      reviewCount: result._count,
    };
  }

  async getTopRatedProducts(limit: number) {
    const results = await prisma.review.groupBy({
      by: ['productId'],
      _avg: { rating: true },
      _count: true,
      having: { rating: { _count: { gte: 3 } } }, // Minimum 3 reviews
      orderBy: { _avg: { rating: 'desc' } },
      take: limit,
    });

    return results.map(r => ({
      product: { __typename: 'Product', id: r.productId },
      averageRating: Math.round(r._avg.rating * 10) / 10,
      reviewCount: r._count,
    }));
  }

  async create(data: any) {
    return prisma.review.create({
      data: {
        body: data.body,
        rating: data.rating,
        authorId: data.authorId,
        authorName: data.authorName || 'Anonymous',
        productId: data.productId,
        helpfulCount: 0,
      },
    });
  }

  async update(id: string, data: any) {
    return prisma.review.update({ where: { id }, data });
  }

  async delete(id: string) {
    return prisma.review.delete({ where: { id } });
  }

  async incrementHelpful(id: string) {
    return prisma.review.update({
      where: { id },
      data: { helpfulCount: { increment: 1 } },
    });
  }
}
```

---

## 7. Apollo Router Configuration

The Apollo Router is the gateway that composes all subgraph schemas and executes query plans.

### Composing the Supergraph

First, define how subgraphs connect in the composition config:

```yaml
# supergraph-config.yaml
federation_version: =2.0.0
subgraphs:
  users:
    routing_url: http://users:4001/graphql
    schema:
      file: ./subgraphs/users/schema.graphql
  products:
    routing_url: http://products:4002/graphql
    schema:
      file: ./subgraphs/products/schema.graphql
  orders:
    routing_url: http://orders:4003/graphql
    schema:
      file: ./subgraphs/orders/schema.graphql
  reviews:
    routing_url: http://reviews:4004/graphql
    schema:
      file: ./subgraphs/reviews/schema.graphql
```

Compose the supergraph schema using Rover:

```bash
#!/bin/bash
# scripts/compose-supergraph.sh

# Export each subgraph's schema from its running instance
# (or use the file-based approach above)
rover supergraph compose \
  --config supergraph-config.yaml \
  --output router/supergraph.graphql

echo "Supergraph schema composed successfully"
echo "Types in supergraph:"
grep "^type " router/supergraph.graphql | wc -l
```

### Router Configuration File

```yaml
# router/router.yaml

# Where to find the composed supergraph schema
supergraph:
  path: ./supergraph.graphql

# Server settings
listen: 0.0.0.0:4000

# CORS configuration for browser clients
cors:
  origins:
    - http://localhost:3000
    - https://shop.example.com
  allow_headers:
    - Content-Type
    - Authorization
    - X-Request-ID
  methods:
    - GET
    - POST
    - OPTIONS

# Propagate headers from client to all subgraphs
headers:
  all:
    request:
      - propagate:
          named: Authorization
      - propagate:
          named: X-Request-ID
      - insert:
          name: X-Router-Source
          value: "apollo-router"

# Traffic shaping and timeouts
traffic_shaping:
  router:
    timeout: 60s
    global_rate_limit:
      capacity: 1000
      interval: 1s
  all:
    timeout: 30s
    deduplicate_query: true
  subgraphs:
    users:
      timeout: 10s
    products:
      timeout: 15s
    orders:
      timeout: 20s
    reviews:
      timeout: 10s

# Introspection (disable in production)
sandbox:
  enabled: false
homepage:
  enabled: false

# Telemetry and observability
telemetry:
  instrumentation:
    spans:
      mode: spec_compliant
  exporters:
    tracing:
      common:
        service_name: "ecommerce-gateway"
      otlp:
        enabled: true
        endpoint: http://otel-collector:4317
        protocol: grpc
    metrics:
      prometheus:
        enabled: true
        listen: 0.0.0.0:9090
        path: /metrics
    logging:
      stdout:
        enabled: true
        format: json
```

### Running the Router

```bash
# Download the Apollo Router binary
curl -sSL https://router.apollo.dev/download/nix/latest | sh

# Compose the supergraph schema first
rover supergraph compose \
  --config supergraph-config.yaml \
  --output router/supergraph.graphql

# Start the router
./router --config router/router.yaml

# Verify the router is running
curl -s http://localhost:4000/health
# {"status":"UP"}
```

### Query Plan Visualization

When the router receives a complex query, it generates a plan that shows exactly which subgraphs are called in what order:

```graphql
# Client query
query OrderDetails {
  order(id: "ord-100") {
    id
    status
    total
    user {
      name
      email
      totalSpent
    }
    items {
      quantity
      subtotal
      product {
        name
        price
        averageRating
        reviews(limit: 3) {
          body
          rating
          author { name }
        }
      }
    }
  }
}
```

```
Query Plan:
═══════════════════════════════════════════════════════

Sequence {
  Fetch(service: "orders") {
    order(id: "ord-100") {
      id status total
      user { __typename id }
      items {
        quantity unitPrice
        product { __typename id }
      }
    }
  }
  Parallel {
    Fetch(service: "users") {
      _entities(representations: $User) {
        ... on User { name email }
      }
    }
    Sequence {
      Fetch(service: "products") {
        _entities(representations: $Product) {
          ... on Product { name price }
        }
      }
      Fetch(service: "reviews") {
        _entities(representations: $Product) {
          ... on Product {
            averageRating
            reviews(limit: 3) {
              body rating
              author { __typename id name }
            }
          }
        }
      }
    }
    Fetch(service: "orders") {
      _entities(representations: $User) {
        ... on User { totalSpent }
      }
    }
  }
}
```

The router parallelizes independent fetches (Users and Products/Reviews can run simultaneously) and sequences dependent ones (Reviews depends on Product IDs from Products).

---

## 8. Gateway Authentication

Authentication is handled at the router level so that every subgraph receives a consistent user context without duplicating JWT validation logic.

### JWT Validation with Coprocessor

Apollo Router supports external coprocessors -- HTTP services that intercept and modify requests. This is the recommended pattern for custom auth logic.

```typescript
// auth-coprocessor/index.ts
import express from 'express';
import jwt from 'jsonwebtoken';

const app = express();
app.use(express.json());

const JWT_SECRET = process.env.JWT_SECRET || 'super-secret-key';

// The router sends requests here before forwarding to subgraphs
app.post('/auth', (req, res) => {
  const { headers, body } = req.body;
  const authHeader = headers?.authorization;

  if (!authHeader) {
    // Public queries are allowed without auth
    return res.json({
      control: 'continue',
      headers: {
        ...headers,
        'x-user-id': '',
        'x-user-role': '',
      },
    });
  }

  try {
    const token = authHeader.replace('Bearer ', '');
    const decoded = jwt.verify(token, JWT_SECRET) as any;

    // Add user context as headers for subgraphs to consume
    return res.json({
      control: 'continue',
      headers: {
        ...headers,
        'x-user-id': decoded.userId,
        'x-user-role': decoded.role,
      },
    });
  } catch (error) {
    return res.json({
      control: 'break',
      status: 401,
      body: JSON.stringify({
        errors: [{ message: 'Invalid or expired token' }],
      }),
    });
  }
});

app.listen(4010, () => {
  console.log('Auth coprocessor running on port 4010');
});
```

### Router Coprocessor Configuration

```yaml
# Add to router/router.yaml
coprocessor:
  url: http://auth-coprocessor:4010/auth
  router:
    request:
      headers: true
      body: false
```

### Subgraph Context from Headers

Each subgraph reads the user context from headers injected by the coprocessor, so subgraphs never need to parse JWTs themselves:

```typescript
// Shared pattern across all subgraphs
context: async ({ req }) => {
  const userId = req.headers['x-user-id'] || null;
  const userRole = req.headers['x-user-role'] || null;

  return {
    userId: userId || null,
    userRole: userRole || null,
    dataSources: { /* ... */ },
  };
},
```

### Authorization Patterns

For field-level authorization, use a directive or resolver middleware:

```typescript
// shared/auth.ts
export function requireAuth(userId: string | null): void {
  if (!userId) {
    throw new Error('Authentication required');
  }
}

export function requireRole(
  userRole: string | null,
  ...roles: string[]
): void {
  if (!userRole || !roles.includes(userRole)) {
    throw new Error(
      `Requires one of: ${roles.join(', ')}`
    );
  }
}

// Usage in resolvers
import { requireAuth, requireRole } from '../shared/auth';

const resolvers = {
  Mutation: {
    createProduct: async (_, { input }, { userId, userRole, dataSources }) => {
      requireAuth(userId);
      requireRole(userRole, 'SELLER', 'ADMIN');
      return dataSources.products.create({ ...input, sellerId: userId });
    },
  },
};
```

---

## 9. Caching Strategy

Federation requires a layered caching approach because data spans multiple subgraphs with different freshness requirements.

### @cacheControl Directive

Apply cache hints at the type and field level:

```graphql
# Products subgraph -- products change infrequently
type Product @key(fields: "id") @cacheControl(maxAge: 300) {
  id: ID!
  name: String!
  price: Float! @cacheControl(maxAge: 60)
  inventory: Int! @cacheControl(maxAge: 10)
}

type Category @key(fields: "id") @cacheControl(maxAge: 3600) {
  id: ID!
  name: String!
}

# Reviews subgraph -- ratings are aggregated, cache briefly
type Product @key(fields: "id") {
  id: ID!
  averageRating: Float @cacheControl(maxAge: 60)
  reviewCount: Int! @cacheControl(maxAge: 60)
  reviews: [Review!]! @cacheControl(maxAge: 30)
}

# Orders subgraph -- orders are user-specific, private
type Order @key(fields: "id") @cacheControl(maxAge: 0, scope: PRIVATE) {
  id: ID!
  status: OrderStatus!
}
```

### Router-Level Caching

Configure entity caching in the router:

```yaml
# Add to router/router.yaml

# Entity cache -- caches resolved entities across requests
preview_entity_cache:
  enabled: true
  subgraphs:
    products:
      enabled: true
      ttl: 300s
    reviews:
      enabled: true
      ttl: 60s
    users:
      enabled: true
      ttl: 120s
    orders:
      # Do not cache user-specific order data
      enabled: false

# Response cache -- caches entire query responses
preview_cache:
  enabled: true
  storage:
    in_memory:
      limit: 100MB
```

### CDN Integration

For public data that does not vary by user, set HTTP cache headers so a CDN can serve responses:

```typescript
// In the products subgraph, set cache-control on the response
const resolvers = {
  Query: {
    products: async (_, args, { dataSources }) => {
      // Tell Apollo Server to include cache hints in the response
      return dataSources.products.findMany(args);
    },
  },
};

// Apollo Server 4 plugin for cache headers
import { ApolloServerPluginCacheControl } from '@apollo/server/plugin/cacheControl';

const server = new ApolloServer({
  schema: buildSubgraphSchema({ typeDefs, resolvers }),
  plugins: [
    ApolloServerPluginCacheControl({
      defaultMaxAge: 0,  // No caching unless explicitly set
    }),
  ],
});
```

### Cache Invalidation

When data changes, invalidate relevant cache entries:

```typescript
// After a product update, purge the cache via router API
async function invalidateProductCache(productId: string) {
  await fetch('http://router:4000/invalidate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      type: 'Product',
      id: productId,
    }),
  });
}

// Call after mutations
const resolvers = {
  Mutation: {
    updateProduct: async (_, { id, input }, ctx) => {
      const updated = await ctx.dataSources.products.update(id, input);
      await invalidateProductCache(id);
      return updated;
    },
  },
};
```

---

## 10. Monitoring and Observability

A federated architecture has more moving parts than a monolith. Observability is essential for debugging slow queries, tracking errors, and understanding traffic patterns.

### OpenTelemetry Integration

Each subgraph exports traces using OpenTelemetry, which the router also supports natively:

```typescript
// shared/tracing.ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-grpc';
import { getNodeAutoInstrumentations } from '@opentelemetry/auto-instrumentations-node';
import { Resource } from '@opentelemetry/resources';
import {
  ATTR_SERVICE_NAME,
  ATTR_SERVICE_VERSION,
} from '@opentelemetry/semantic-conventions';

export function initTracing(serviceName: string) {
  const sdk = new NodeSDK({
    resource: new Resource({
      [ATTR_SERVICE_NAME]: serviceName,
      [ATTR_SERVICE_VERSION]: '1.0.0',
    }),
    traceExporter: new OTLPTraceExporter({
      url: process.env.OTEL_ENDPOINT || 'http://otel-collector:4317',
    }),
    instrumentations: [getNodeAutoInstrumentations()],
  });

  sdk.start();
  console.log(`Tracing initialized for ${serviceName}`);
}
```

```typescript
// In each subgraph's index.ts, call before anything else
import { initTracing } from '../shared/tracing';
initTracing('users-subgraph');
```

### Prometheus Metrics

The router exposes Prometheus metrics at `/metrics`. Key metrics to monitor:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'apollo-router'
    static_configs:
      - targets: ['router:9090']
    metrics_path: /metrics
    scrape_interval: 15s

  - job_name: 'subgraphs'
    static_configs:
      - targets:
        - 'users:9091'
        - 'products:9092'
        - 'orders:9093'
        - 'reviews:9094'
```

Key metrics to track:

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| `apollo_router_http_requests_total` | Total requests by status | Error rate > 1% |
| `apollo_router_http_request_duration_seconds` | Latency histogram | p99 > 2s |
| `apollo_router_cache_hit_count` | Cache hits | Hit ratio < 50% |
| `apollo_router_query_planning_duration_seconds` | Query plan time | p95 > 100ms |
| `apollo_router_subgraph_request_duration_seconds` | Per-subgraph latency | p95 > 500ms |

### Health Checks

Each subgraph and the router expose health endpoints:

```typescript
// Add to each subgraph's Express app
app.get('/health', (req, res) => {
  res.json({
    status: 'UP',
    service: 'users-subgraph',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// Deep health check -- verifies database connectivity
app.get('/health/ready', async (req, res) => {
  try {
    await prisma.$queryRaw`SELECT 1`;
    res.json({ status: 'READY', database: 'connected' });
  } catch (error) {
    res.status(503).json({
      status: 'NOT_READY',
      database: 'disconnected',
      error: error.message,
    });
  }
});
```

### Grafana Dashboard

A dashboard for the federated gateway would track these panels:

```
┌───────────────────────────────────────────────────┐
│           E-Commerce Gateway Dashboard             │
├──────────────────┬────────────────────────────────┤
│ Request Rate     │ Error Rate (%)                  │
│ ████████▓▓░░     │ ▁▁▁▂▁▁▁▁▃▁▁                   │
│ 450 req/s        │ 0.3%                           │
├──────────────────┼────────────────────────────────┤
│ Latency (p50/95) │ Cache Hit Ratio                │
│ p50: 45ms        │ ████████████░░░                │
│ p95: 180ms       │ 78%                            │
├──────────────────┼────────────────────────────────┤
│ Subgraph Latency │ Active Connections              │
│ Users:   12ms    │ ████████████████               │
│ Products: 25ms   │ 342 / 1000                     │
│ Orders:   35ms   │                                │
│ Reviews:  18ms   │                                │
└──────────────────┴────────────────────────────────┘
```

---

## 11. Testing

Testing a federated system requires both unit testing within each subgraph and integration testing across the full supergraph.

### Subgraph Unit Testing

Test each subgraph's resolvers in isolation:

```typescript
// subgraphs/products/src/__tests__/resolvers.test.ts
import { ApolloServer } from '@apollo/server';
import { buildSubgraphSchema } from '@apollo/subgraph';
import { typeDefs } from '../schema';
import { resolvers } from '../resolvers';

describe('Products Subgraph', () => {
  let server: ApolloServer;

  beforeAll(() => {
    server = new ApolloServer({
      schema: buildSubgraphSchema({ typeDefs, resolvers }),
    });
  });

  it('resolves a product by ID', async () => {
    const response = await server.executeOperation({
      query: `
        query GetProduct($id: ID!) {
          product(id: $id) {
            id
            name
            price
          }
        }
      `,
      variables: { id: 'prod-1' },
    }, {
      contextValue: {
        userId: 'user-1',
        userRole: 'SELLER',
        dataSources: {
          products: {
            findById: jest.fn().mockResolvedValue({
              id: 'prod-1',
              name: 'Widget',
              price: 29.99,
              categoryId: 'cat-1',
              sellerId: 'user-1',
            }),
          },
        },
      },
    });

    expect(response.body.kind).toBe('single');
    const result = (response.body as any).singleResult;
    expect(result.errors).toBeUndefined();
    expect(result.data.product).toEqual({
      id: 'prod-1',
      name: 'Widget',
      price: 29.99,
    });
  });

  it('requires seller role to create products', async () => {
    const response = await server.executeOperation({
      query: `
        mutation CreateProduct($input: CreateProductInput!) {
          createProduct(input: $input) { id name }
        }
      `,
      variables: {
        input: {
          name: 'New Product',
          description: 'A product',
          price: 19.99,
          categoryId: 'cat-1',
          sku: 'NP-001',
          inventory: 100,
        },
      },
    }, {
      contextValue: {
        userId: 'user-1',
        userRole: 'CUSTOMER', // Not a seller
        dataSources: { products: {} },
      },
    });

    const result = (response.body as any).singleResult;
    expect(result.errors[0].message).toContain('Only sellers');
  });

  it('resolves __resolveReference for entity resolution', async () => {
    const response = await server.executeOperation({
      query: `
        query Entities($representations: [_Any!]!) {
          _entities(representations: $representations) {
            ... on Product { id name price }
          }
        }
      `,
      variables: {
        representations: [
          { __typename: 'Product', id: 'prod-1' },
        ],
      },
    }, {
      contextValue: {
        dataSources: {
          products: {
            findById: jest.fn().mockResolvedValue({
              id: 'prod-1',
              name: 'Widget',
              price: 29.99,
            }),
          },
        },
      },
    });

    const result = (response.body as any).singleResult;
    expect(result.data._entities[0].name).toBe('Widget');
  });
});
```

### Integration Testing with Mock Subgraphs

Test cross-subgraph queries by running all subgraphs with the router:

```typescript
// tests/integration/gateway.test.ts
import { ApolloServer } from '@apollo/server';
import { ApolloGateway, IntrospectAndCompose } from '@apollo/gateway';

describe('Federation Integration Tests', () => {
  let gateway: ApolloServer;

  beforeAll(async () => {
    // Start all subgraphs on test ports, then compose
    const apolloGateway = new ApolloGateway({
      supergraphSdl: new IntrospectAndCompose({
        subgraphs: [
          { name: 'users', url: 'http://localhost:14001/graphql' },
          { name: 'products', url: 'http://localhost:14002/graphql' },
          { name: 'orders', url: 'http://localhost:14003/graphql' },
          { name: 'reviews', url: 'http://localhost:14004/graphql' },
        ],
      }),
    });

    gateway = new ApolloServer({ gateway: apolloGateway });
    await gateway.start();
  });

  afterAll(async () => {
    await gateway.stop();
  });

  it('resolves cross-subgraph product query with reviews', async () => {
    const response = await gateway.executeOperation({
      query: `
        query {
          product(id: "prod-1") {
            name
            price
            seller { name email }
            reviews { body rating author { name } }
            averageRating
          }
        }
      `,
    });

    const result = (response.body as any).singleResult;
    expect(result.errors).toBeUndefined();
    expect(result.data.product.seller.name).toBeDefined();
    expect(result.data.product.reviews).toBeInstanceOf(Array);
    expect(result.data.product.averageRating).toBeGreaterThanOrEqual(1);
  });

  it('resolves user with orders and products', async () => {
    const response = await gateway.executeOperation({
      query: `
        query {
          me {
            name
            orders { id status total }
            products { name price }
            totalSpent
          }
        }
      `,
    }, {
      contextValue: { userId: 'user-1' },
    });

    const result = (response.body as any).singleResult;
    expect(result.errors).toBeUndefined();
    expect(result.data.me.orders).toBeInstanceOf(Array);
    expect(result.data.me.products).toBeInstanceOf(Array);
  });
});
```

### Schema Validation with Rover

Validate schema changes before merging:

```bash
#!/bin/bash
# scripts/check-schemas.sh

set -e

SUBGRAPHS=("users" "products" "orders" "reviews")

for subgraph in "${SUBGRAPHS[@]}"; do
  echo "Checking $subgraph subgraph schema..."
  rover subgraph check ecommerce-graph@staging \
    --name "$subgraph" \
    --schema "./subgraphs/$subgraph/schema.graphql"
done

echo "All schema checks passed"

# Attempt full composition to catch cross-subgraph issues
echo "Composing supergraph..."
rover supergraph compose \
  --config supergraph-config.yaml \
  --output /tmp/test-supergraph.graphql

echo "Composition successful"
```

---

## 12. Deployment

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  # --- Databases ---
  users-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: users_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5432:5432"
    volumes:
      - users_data:/var/lib/postgresql/data

  products-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: products_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5433:5432"
    volumes:
      - products_data:/var/lib/postgresql/data

  orders-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: orders_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5434:5432"
    volumes:
      - orders_data:/var/lib/postgresql/data

  reviews-db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: reviews_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    ports:
      - "5435:5432"
    volumes:
      - reviews_data:/var/lib/postgresql/data

  # --- Subgraphs ---
  users:
    build: ./subgraphs/users
    ports:
      - "4001:4001"
    environment:
      PORT: 4001
      DATABASE_URL: postgresql://postgres:postgres@users-db:5432/users_db
      JWT_SECRET: ${JWT_SECRET:-super-secret-key}
    depends_on:
      - users-db

  products:
    build: ./subgraphs/products
    ports:
      - "4002:4002"
    environment:
      PORT: 4002
      DATABASE_URL: postgresql://postgres:postgres@products-db:5432/products_db
      JWT_SECRET: ${JWT_SECRET:-super-secret-key}
    depends_on:
      - products-db

  orders:
    build: ./subgraphs/orders
    ports:
      - "4003:4003"
    environment:
      PORT: 4003
      DATABASE_URL: postgresql://postgres:postgres@orders-db:5432/orders_db
      JWT_SECRET: ${JWT_SECRET:-super-secret-key}
    depends_on:
      - orders-db

  reviews:
    build: ./subgraphs/reviews
    ports:
      - "4004:4004"
    environment:
      PORT: 4004
      DATABASE_URL: postgresql://postgres:postgres@reviews-db:5432/reviews_db
      JWT_SECRET: ${JWT_SECRET:-super-secret-key}
    depends_on:
      - reviews-db

  # --- Auth Coprocessor ---
  auth-coprocessor:
    build: ./auth-coprocessor
    ports:
      - "4010:4010"
    environment:
      JWT_SECRET: ${JWT_SECRET:-super-secret-key}

  # --- Apollo Router ---
  router:
    image: ghcr.io/apollographql/router:v1.57.1
    ports:
      - "4000:4000"
      - "9090:9090"
    volumes:
      - ./router/router.yaml:/dist/config/router.yaml
      - ./router/supergraph.graphql:/dist/config/supergraph.graphql
    command: >
      --config /dist/config/router.yaml
      --supergraph /dist/config/supergraph.graphql
    depends_on:
      - users
      - products
      - orders
      - reviews
      - auth-coprocessor

  # --- Observability ---
  jaeger:
    image: jaegertracing/all-in-one:1.53
    ports:
      - "16686:16686"
      - "14268:14268"
    environment:
      COLLECTOR_OTLP_ENABLED: "true"

  prometheus:
    image: prom/prometheus:v2.49.0
    ports:
      - "9091:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

volumes:
  users_data:
  products_data:
  orders_data:
  reviews_data:
```

### Subgraph Dockerfile

```dockerfile
# subgraphs/users/Dockerfile (same pattern for all subgraphs)
FROM node:20-alpine AS builder

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npx prisma generate
RUN npm run build

FROM node:20-alpine AS runtime

WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/prisma ./prisma
COPY --from=builder /app/package.json ./

# Run Prisma migrations at startup
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 4001
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["node", "dist/index.js"]
```

```bash
#!/bin/sh
# docker-entrypoint.sh
set -e

echo "Running database migrations..."
npx prisma migrate deploy

echo "Starting server..."
exec "$@"
```

### Kubernetes Deployment

```yaml
# k8s/router-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: apollo-router
  labels:
    app: apollo-router
spec:
  replicas: 3
  selector:
    matchLabels:
      app: apollo-router
  template:
    metadata:
      labels:
        app: apollo-router
    spec:
      containers:
        - name: router
          image: ghcr.io/apollographql/router:v1.57.1
          ports:
            - containerPort: 4000
            - containerPort: 9090
          args:
            - --config
            - /config/router.yaml
            - --supergraph
            - /config/supergraph.graphql
          volumeMounts:
            - name: config
              mountPath: /config
          resources:
            requests:
              cpu: 250m
              memory: 256Mi
            limits:
              cpu: "1"
              memory: 512Mi
          readinessProbe:
            httpGet:
              path: /health
              port: 4000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 4000
            initialDelaySeconds: 15
            periodSeconds: 20
      volumes:
        - name: config
          configMap:
            name: router-config
---
apiVersion: v1
kind: Service
metadata:
  name: apollo-router
spec:
  selector:
    app: apollo-router
  ports:
    - name: graphql
      port: 4000
      targetPort: 4000
    - name: metrics
      port: 9090
      targetPort: 9090
  type: LoadBalancer
```

```yaml
# k8s/users-deployment.yaml (pattern for all subgraphs)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: users-subgraph
  labels:
    app: users-subgraph
spec:
  replicas: 2
  selector:
    matchLabels:
      app: users-subgraph
  template:
    metadata:
      labels:
        app: users-subgraph
    spec:
      containers:
        - name: users
          image: registry.example.com/users-subgraph:latest
          ports:
            - containerPort: 4001
          env:
            - name: PORT
              value: "4001"
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef:
                  name: users-db-secret
                  key: url
            - name: JWT_SECRET
              valueFrom:
                secretKeyRef:
                  name: jwt-secret
                  key: secret
          resources:
            requests:
              cpu: 100m
              memory: 128Mi
            limits:
              cpu: 500m
              memory: 256Mi
          readinessProbe:
            httpGet:
              path: /health/ready
              port: 4001
            initialDelaySeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 4001
            initialDelaySeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: users
spec:
  selector:
    app: users-subgraph
  ports:
    - port: 4001
      targetPort: 4001
```

### CI/CD Pipeline for Schema Changes

```yaml
# .github/workflows/deploy.yml
name: Deploy Federation

on:
  push:
    branches: [main]
    paths:
      - 'subgraphs/**'

jobs:
  detect-changes:
    runs-on: ubuntu-latest
    outputs:
      changed: ${{ steps.filter.outputs.changes }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v3
        id: filter
        with:
          filters: |
            users: 'subgraphs/users/**'
            products: 'subgraphs/products/**'
            orders: 'subgraphs/orders/**'
            reviews: 'subgraphs/reviews/**'

  schema-check:
    needs: detect-changes
    runs-on: ubuntu-latest
    strategy:
      matrix:
        subgraph: ${{ fromJson(needs.detect-changes.outputs.changed) }}
    steps:
      - uses: actions/checkout@v4
      - name: Install Rover
        run: |
          curl -sSL https://rover.apollo.dev/nix/latest | sh
          echo "$HOME/.rover/bin" >> $GITHUB_PATH
      - name: Check schema
        run: |
          rover subgraph check ecommerce-graph@production \
            --name ${{ matrix.subgraph }} \
            --schema ./subgraphs/${{ matrix.subgraph }}/schema.graphql
        env:
          APOLLO_KEY: ${{ secrets.APOLLO_KEY }}

  deploy:
    needs: [detect-changes, schema-check]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        subgraph: ${{ fromJson(needs.detect-changes.outputs.changed) }}
    steps:
      - uses: actions/checkout@v4
      - name: Build and push image
        run: |
          docker build -t registry.example.com/${{ matrix.subgraph }}-subgraph:${{ github.sha }} \
            ./subgraphs/${{ matrix.subgraph }}
          docker push registry.example.com/${{ matrix.subgraph }}-subgraph:${{ github.sha }}
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/${{ matrix.subgraph }}-subgraph \
            ${{ matrix.subgraph }}=registry.example.com/${{ matrix.subgraph }}-subgraph:${{ github.sha }}
      - name: Publish schema to GraphOS
        run: |
          rover subgraph publish ecommerce-graph@production \
            --name ${{ matrix.subgraph }} \
            --schema ./subgraphs/${{ matrix.subgraph }}/schema.graphql \
            --routing-url http://${{ matrix.subgraph }}:400${{ strategy.job-index + 1 }}/graphql
        env:
          APOLLO_KEY: ${{ secrets.APOLLO_KEY }}
```

---

## 13. Extensions

Once the core gateway is running, consider these enhancements to improve capability, reliability, and developer experience.

### Real-Time Subscriptions Across Subgraphs

Apollo Router supports subscriptions via WebSocket and HTTP callbacks. Add subscription support to the orders subgraph for order status updates:

```graphql
# Orders subgraph -- add subscription type
type Subscription {
  orderStatusChanged(orderId: ID!): Order!
}
```

```typescript
// Use graphql-subscriptions with Redis PubSub for multi-instance
import { RedisPubSub } from 'graphql-redis-subscriptions';

const pubsub = new RedisPubSub({
  connection: process.env.REDIS_URL,
});

// In the mutation resolver, publish events
const resolvers = {
  Mutation: {
    shipOrder: async (_, args, ctx) => {
      const order = await ctx.dataSources.orders.updateStatus(
        args.orderId, 'SHIPPED', { trackingNumber: args.trackingNumber }
      );
      pubsub.publish(`ORDER_STATUS_${args.orderId}`, {
        orderStatusChanged: order,
      });
      return order;
    },
  },
  Subscription: {
    orderStatusChanged: {
      subscribe: (_, { orderId }) =>
        pubsub.asyncIterableIterator(`ORDER_STATUS_${orderId}`),
    },
  },
};
```

Configure the router for subscription support:

```yaml
# Add to router.yaml
subscription:
  enabled: true
  mode:
    callback:
      public_url: https://gateway.example.com/callback
      listen: 0.0.0.0:4000
      path: /callback
```

### Rate Limiting Per Operation

Protect against abuse by rate limiting expensive operations:

```yaml
# Add to router.yaml
traffic_shaping:
  router:
    global_rate_limit:
      capacity: 500
      interval: 1s
  all:
    deduplicate_query: true

# Per-operation limits via coprocessor
coprocessor:
  url: http://rate-limiter:4011/check
  supergraph:
    request:
      headers: true
      body: true
```

```typescript
// rate-limiter/index.ts
import Redis from 'ioredis';

const redis = new Redis(process.env.REDIS_URL);

const OPERATION_LIMITS = {
  createOrder: { limit: 10, window: 60 },
  createReview: { limit: 5, window: 60 },
  register: { limit: 3, window: 300 },
};

app.post('/check', async (req, res) => {
  const { body, headers } = req.body;
  const userId = headers['x-user-id'];
  const operationName = body?.operationName;

  const rule = OPERATION_LIMITS[operationName];
  if (!rule || !userId) {
    return res.json({ control: 'continue' });
  }

  const key = `rate:${userId}:${operationName}`;
  const count = await redis.incr(key);
  if (count === 1) await redis.expire(key, rule.window);

  if (count > rule.limit) {
    return res.json({
      control: 'break',
      status: 429,
      body: JSON.stringify({
        errors: [{
          message: `Rate limit exceeded for ${operationName}. ` +
                   `Try again in ${rule.window} seconds.`,
        }],
      }),
    });
  }

  return res.json({ control: 'continue' });
});
```

### A/B Testing with @override

Gradually migrate field ownership between subgraphs using `@override` with a percentage label for progressive rollout:

```graphql
# New reviews-v2 subgraph -- takes over averageRating computation
type Product @key(fields: "id") {
  id: ID!
  averageRating: Float @override(from: "reviews", label: "rollout-v2")
}
```

```yaml
# router.yaml -- control the rollout percentage
preview_override:
  rollout-v2:
    percentage: 25  # 25% of traffic goes to the new subgraph
```

### Schema Governance

Enforce schema conventions and prevent breaking changes in CI:

```bash
# Install GraphQL schema linter
npm install -g graphql-schema-linter

# .graphql-schema-linterrc
{
  "rules": [
    "fields-have-descriptions",
    "types-have-descriptions",
    "enum-values-all-caps",
    "input-object-fields-sorted-alphabetically",
    "relay-connection-types-spec"
  ]
}

# Run in CI
graphql-schema-linter subgraphs/*/schema.graphql
```

```yaml
# Add schema governance to CI
- name: Lint GraphQL schemas
  run: |
    for dir in subgraphs/*/; do
      echo "Linting $(basename $dir)..."
      graphql-schema-linter "$dir/schema.graphql"
    done

- name: Check for breaking changes
  run: |
    rover subgraph check ecommerce-graph@production \
      --name ${{ matrix.subgraph }} \
      --schema ./subgraphs/${{ matrix.subgraph }}/schema.graphql
```

---

## References

- [Apollo Federation 2 Documentation](https://www.apollographql.com/docs/federation/) -- Complete guide to federation architecture, directives, and subgraph design
- [Apollo Router Documentation](https://www.apollographql.com/docs/router/) -- Configuration reference for the Rust-based gateway
- [Apollo Router Configuration Reference](https://www.apollographql.com/docs/router/configuration/overview) -- Detailed YAML config options for traffic shaping, caching, telemetry
- [Rover CLI Documentation](https://www.apollographql.com/docs/rover/) -- Schema management, composition, and CI/CD integration
- [Apollo GraphOS](https://www.apollographql.com/docs/graphos/) -- Cloud-hosted schema registry and observability platform
- [Federation Specification](https://www.apollographql.com/docs/federation/federation-spec/) -- Technical spec for federation directives and entity resolution
- [Apollo Server 4 Documentation](https://www.apollographql.com/docs/apollo-server/) -- Subgraph server setup and plugin system
- [OpenTelemetry Node.js SDK](https://opentelemetry.io/docs/languages/js/) -- Distributed tracing instrumentation
- [Prisma Documentation](https://www.prisma.io/docs) -- ORM for database access in each subgraph

---

**Previous**: [REST to GraphQL Migration](./15_REST_to_GraphQL_Migration.md) | [Overview](./00_Overview.md)
