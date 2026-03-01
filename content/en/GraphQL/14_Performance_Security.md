# 14. Performance and Security

**Previous**: [Testing](./13_Testing.md) | **Next**: [REST to GraphQL Migration](./15_REST_to_GraphQL_Migration.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement query depth limiting and cost analysis to prevent resource exhaustion attacks
2. Configure field-level cost directives and per-operation complexity budgets
3. Protect a GraphQL API against common attack vectors including batching abuse, introspection leaks, and injection
4. Set up rate limiting strategies that account for the variable cost of GraphQL operations
5. Build a monitoring pipeline with slow query logging, operation metrics, and alerting

## Table of Contents

1. [Why GraphQL Needs Special Protection](#1-why-graphql-needs-special-protection)
2. [Query Depth Limiting](#2-query-depth-limiting)
3. [Query Complexity Analysis](#3-query-complexity-analysis)
4. [Field-Level Cost Directives](#4-field-level-cost-directives)
5. [Rate Limiting](#5-rate-limiting)
6. [Introspection Control](#6-introspection-control)
7. [Batching Attack Prevention](#7-batching-attack-prevention)
8. [Input Validation and Injection Prevention](#8-input-validation-and-injection-prevention)
9. [Timeout and Resource Limits](#9-timeout-and-resource-limits)
10. [Monitoring and Observability](#10-monitoring-and-observability)
11. [Practice Problems](#11-practice-problems)

**Difficulty**: ⭐⭐⭐⭐

---

GraphQL's power is also its vulnerability. A single query can traverse the entire graph, requesting deeply nested data that triggers thousands of database calls. Unlike REST, where each endpoint has a predictable cost, a GraphQL endpoint's cost depends entirely on what the client asks for. This lesson covers the techniques that let you offer flexible queries while keeping your server safe from abuse -- intentional or accidental.

---

## 1. Why GraphQL Needs Special Protection

### The Difference from REST

```
REST:
  GET /api/users       → Fixed cost: 1 DB query, ~50 rows
  GET /api/users/42    → Fixed cost: 1 DB query, 1 row
  Cost is predictable. Rate limiting by endpoint works.

GraphQL:
  query {
    users {               → 1 query
      posts {             → N queries (one per user)
        comments {        → N*M queries (one per post)
          author {        → N*M*K queries (one per comment)
            posts { ... } → Recursive explosion
          }
        }
      }
    }
  }
  Cost is unpredictable. A single request can be 1 query or 10,000.
```

### Attack Surface

```
┌────────────────────────────────────────────────────┐
│              GraphQL Attack Surface                 │
├────────────────────────────────────────────────────┤
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ Deep Queries │  │   Batch     │  │ Introspect │  │
│  │   Nesting   │  │  Attacks    │  │   Leaks    │  │
│  └─────────────┘  └─────────────┘  └────────────┘  │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  Alias      │  │  Injection  │  │  Resource  │  │
│  │  Flooding   │  │  via Input  │  │ Exhaustion │  │
│  └─────────────┘  └─────────────┘  └────────────┘  │
│                                                      │
└────────────────────────────────────────────────────┘
```

---

## 2. Query Depth Limiting

The simplest defense: reject queries that nest deeper than a threshold.

### Why Depth Matters

```graphql
# Depth 1 -- safe
query { users { name } }

# Depth 3 -- normal
query {
  user(id: "1") {
    posts {
      comments {
        body
      }
    }
  }
}

# Depth 10+ -- likely an attack or a mistake
query {
  user(id: "1") {
    friends {
      friends {
        friends {
          friends {
            friends {
              friends {
                friends { name }
              }
            }
          }
        }
      }
    }
  }
}
```

### Implementation with graphql-depth-limit

```typescript
import depthLimit from 'graphql-depth-limit';
import { ApolloServer } from '@apollo/server';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [
    depthLimit(10), // Reject queries deeper than 10 levels
  ],
});
```

### Custom Depth Limiter

```typescript
import { ValidationContext, ASTVisitor } from 'graphql';

function createDepthLimiter(maxDepth: number): (context: ValidationContext) => ASTVisitor {
  return (context: ValidationContext): ASTVisitor => {
    return {
      // Track depth as we enter/leave selection sets
      Field: {
        enter(node, _key, _parent, path) {
          // Calculate depth from the AST path
          const depth = path.filter(
            (segment) => typeof segment === 'number'
          ).length;

          if (depth > maxDepth) {
            context.reportError(
              new GraphQLError(
                `Query depth ${depth} exceeds maximum allowed depth of ${maxDepth}`,
                { nodes: [node] }
              )
            );
          }
        },
      },
    };
  };
}

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [createDepthLimiter(10)],
});
```

### Depth Limit Recommendations

| API Type | Recommended Max Depth |
|----------|----------------------|
| Simple CRUD | 5-7 |
| Social graph | 7-10 |
| E-commerce | 8-12 |
| Internal/Trusted | 15-20 |

---

## 3. Query Complexity Analysis

Depth limiting is coarse-grained -- a shallow query with 1000 aliases is still dangerous. Complexity analysis assigns a cost to each field and rejects queries that exceed a budget.

### How It Works

```graphql
# Each field has a cost:
# Scalar fields: 0 (free)
# Object fields: 1
# List fields: cost * estimatedSize

query {
  users(first: 100) {        # cost: 1 + (100 * child cost)
    name                     # cost: 0
    posts(first: 50) {       # cost: 100 * (1 + 50 * child cost) = 100 * 51
      title                  # cost: 0
      comments(first: 20) {  # cost: 100 * 50 * (1 + 20 * 1) = 100 * 50 * 21
        body                 # cost: 0
      }
    }
  }
}

# Total: 1 + 100 * (51 + 50 * 21) = 1 + 100 * 1101 = 110,101
# If budget is 10,000 → REJECTED
```

### Implementation with graphql-query-complexity

```typescript
import {
  getComplexity,
  simpleEstimator,
  fieldExtensionsEstimator,
} from 'graphql-query-complexity';
import { ApolloServer } from '@apollo/server';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  plugins: [
    {
      async requestDidStart() {
        return {
          async didResolveOperation({ request, document, schema }) {
            const complexity = getComplexity({
              schema,
              operationName: request.operationName,
              query: document,
              variables: request.variables,
              estimators: [
                // Use @complexity directives from schema (checked first)
                fieldExtensionsEstimator(),
                // Fallback: 1 per field
                simpleEstimator({ defaultComplexity: 1 }),
              ],
            });

            const MAX_COMPLEXITY = 1000;

            if (complexity > MAX_COMPLEXITY) {
              throw new GraphQLError(
                `Query complexity ${complexity} exceeds maximum of ${MAX_COMPLEXITY}`,
                {
                  extensions: {
                    code: 'QUERY_TOO_COMPLEX',
                    complexity,
                    maxComplexity: MAX_COMPLEXITY,
                  },
                }
              );
            }

            // Log complexity for monitoring
            console.log(`Query complexity: ${complexity}`);
          },
        };
      },
    },
  ],
});
```

---

## 4. Field-Level Cost Directives

Instead of a one-size-fits-all estimator, assign specific costs to fields that you know are expensive.

### Schema-Level Cost Annotations

```graphql
directive @complexity(
  value: Int!
  multipliers: [String!]
) on FIELD_DEFINITION

type Query {
  # Simple lookup: low cost
  user(id: ID!): User @complexity(value: 1)

  # Full-text search: expensive
  searchProducts(query: String!, first: Int = 10): [Product!]!
    @complexity(value: 10, multipliers: ["first"])

  # Analytics query: very expensive
  salesReport(from: DateTime!, to: DateTime!): SalesReport
    @complexity(value: 50)
}

type User {
  id: ID!
  name: String!      # Scalar: 0 cost (default)

  # List that requires a join query
  orders(first: Int = 10): [Order!]!
    @complexity(value: 5, multipliers: ["first"])

  # Computed field with external API call
  creditScore: Int @complexity(value: 20)
}
```

### Custom Estimator

```typescript
import { ComplexityEstimator } from 'graphql-query-complexity';

const customEstimator: ComplexityEstimator = ({
  type,
  field,
  args,
  childComplexity,
}) => {
  // Read cost from schema extensions (set via @complexity directive)
  const complexity = field.extensions?.complexity;

  if (complexity) {
    const { value, multipliers } = complexity;

    // Apply multipliers from arguments (e.g., "first", "limit")
    let multiplier = 1;
    if (multipliers) {
      for (const argName of multipliers) {
        if (args[argName]) {
          multiplier *= args[argName];
        }
      }
    }

    // Total cost = base cost + (multiplier * child complexity)
    return value + multiplier * childComplexity;
  }

  // No annotation: return undefined to let the next estimator handle it
  return undefined;
};
```

---

## 5. Rate Limiting

Traditional rate limiting (X requests per minute) does not work well for GraphQL because one request can vary enormously in cost. Instead, rate limit by **complexity budget**.

### Complexity-Based Rate Limiting

```typescript
import { RateLimiterRedis } from 'rate-limiter-flexible';
import Redis from 'ioredis';

const redis = new Redis();

// Each user gets a complexity budget per time window
const complexityLimiter = new RateLimiterRedis({
  storeClient: redis,
  keyPrefix: 'gql_complexity',
  points: 10_000,   // 10,000 complexity points
  duration: 60,     // per 60 seconds
});

// Also limit the raw number of operations
const operationLimiter = new RateLimiterRedis({
  storeClient: redis,
  keyPrefix: 'gql_operations',
  points: 100,      // 100 operations
  duration: 60,     // per 60 seconds
});

const rateLimitPlugin = {
  async requestDidStart() {
    return {
      async didResolveOperation({ request, document, schema, contextValue }) {
        const userId = contextValue.userId || contextValue.ip;

        // Check operation count limit
        try {
          await operationLimiter.consume(userId);
        } catch {
          throw new GraphQLError('Rate limit exceeded: too many requests', {
            extensions: { code: 'RATE_LIMITED' },
          });
        }

        // Check complexity budget
        const complexity = calculateComplexity(schema, document, request.variables);
        try {
          await complexityLimiter.consume(userId, complexity);
        } catch (rateLimiterRes) {
          throw new GraphQLError(
            `Complexity budget exceeded. Retry after ${Math.ceil(
              rateLimiterRes.msBeforeNext / 1000
            )}s`,
            {
              extensions: {
                code: 'COMPLEXITY_BUDGET_EXCEEDED',
                retryAfter: Math.ceil(rateLimiterRes.msBeforeNext / 1000),
              },
            }
          );
        }
      },
    };
  },
};
```

### Rate Limit Response Headers

```typescript
// Inform clients of their remaining budget
const rateLimitHeadersPlugin = {
  async requestDidStart() {
    return {
      async willSendResponse({ response, contextValue }) {
        const userId = contextValue.userId || contextValue.ip;
        const limiterRes = await complexityLimiter.get(userId);

        response.http.headers.set(
          'X-RateLimit-Complexity-Limit', '10000'
        );
        response.http.headers.set(
          'X-RateLimit-Complexity-Remaining',
          String(limiterRes ? limiterRes.remainingPoints : 10000)
        );
        response.http.headers.set(
          'X-RateLimit-Complexity-Reset',
          String(limiterRes ? Math.ceil(limiterRes.msBeforeNext / 1000) : 60)
        );
      },
    };
  },
};
```

---

## 6. Introspection Control

Introspection reveals your entire schema -- types, fields, arguments, and directives. Useful in development, dangerous in production.

### Disable Introspection in Production

```typescript
import { ApolloServer } from '@apollo/server';
import {
  ApolloServerPluginInlineTraceDisabled,
} from '@apollo/server/plugin/disabled';

const server = new ApolloServer({
  typeDefs,
  resolvers,
  introspection: process.env.NODE_ENV !== 'production',
  plugins: [
    // Also disable inline tracing in production
    ...(process.env.NODE_ENV === 'production'
      ? [ApolloServerPluginInlineTraceDisabled()]
      : []),
  ],
});
```

### Selective Introspection (Allow Internal, Block External)

```typescript
// Custom plugin: allow introspection only from internal IPs
const selectiveIntrospectionPlugin = {
  async requestDidStart({ request, contextValue }) {
    return {
      async didResolveOperation({ document }) {
        const isIntrospection = document.definitions.some(
          (def) =>
            def.kind === 'OperationDefinition' &&
            def.selectionSet.selections.some(
              (sel) =>
                sel.kind === 'Field' &&
                (sel.name.value === '__schema' || sel.name.value === '__type')
            )
        );

        if (isIntrospection && !contextValue.isInternalRequest) {
          throw new GraphQLError('Introspection is not allowed', {
            extensions: { code: 'INTROSPECTION_DISABLED' },
          });
        }
      },
    };
  },
};
```

---

## 7. Batching Attack Prevention

GraphQL supports sending multiple operations in a single HTTP request (query batching). Attackers can exploit this to bypass rate limits.

### The Attack

```json
// Single HTTP request with 1000 login attempts
[
  { "query": "mutation { login(email: \"a@b.com\", password: \"pass1\") { token } }" },
  { "query": "mutation { login(email: \"a@b.com\", password: \"pass2\") { token } }" },
  { "query": "mutation { login(email: \"a@b.com\", password: \"pass3\") { token } }" },
  // ... 997 more attempts
]
```

### Alias-Based Attack (Single Query)

```graphql
# Single query with 1000 aliases
query {
  attempt1: login(email: "a@b.com", password: "pass1") { token }
  attempt2: login(email: "a@b.com", password: "pass2") { token }
  attempt3: login(email: "a@b.com", password: "pass3") { token }
  # ... 997 more aliases
}
```

### Prevention

```typescript
// 1. Limit batch size
const server = new ApolloServer({
  typeDefs,
  resolvers,
  allowBatchedHttpRequests: true,  // Allow batching but...
});

// Middleware to limit batch size
app.use('/graphql', (req, res, next) => {
  if (Array.isArray(req.body) && req.body.length > 5) {
    return res.status(400).json({
      error: 'Batch size exceeds maximum of 5 operations',
    });
  }
  next();
});

// 2. Limit aliases per query
import { ASTVisitor, ValidationContext } from 'graphql';

function aliasLimit(maxAliases: number) {
  return (context: ValidationContext): ASTVisitor => {
    let aliasCount = 0;
    return {
      Field(node) {
        if (node.alias) {
          aliasCount++;
          if (aliasCount > maxAliases) {
            context.reportError(
              new GraphQLError(
                `Query uses ${aliasCount} aliases, exceeding the maximum of ${maxAliases}`
              )
            );
          }
        }
      },
    };
  };
}

const server = new ApolloServer({
  typeDefs,
  resolvers,
  validationRules: [aliasLimit(50)],
});
```

---

## 8. Input Validation and Injection Prevention

### Custom Scalars for Validation

```typescript
import { GraphQLScalarType, Kind } from 'graphql';

// Email scalar with validation
const EmailScalar = new GraphQLScalarType({
  name: 'Email',
  description: 'A valid email address',
  serialize(value: string) {
    return value;
  },
  parseValue(value: string) {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(value)) {
      throw new GraphQLError(`Invalid email address: ${value}`);
    }
    return value.toLowerCase();
  },
  parseLiteral(ast) {
    if (ast.kind !== Kind.STRING) {
      throw new GraphQLError('Email must be a string');
    }
    return this.parseValue(ast.value);
  },
});

// Sanitized string that strips HTML/scripts
const SanitizedString = new GraphQLScalarType({
  name: 'SanitizedString',
  description: 'A string with HTML tags stripped',
  parseValue(value: string) {
    // Strip HTML tags to prevent stored XSS
    return value.replace(/<[^>]*>/g, '').trim();
  },
  serialize(value: string) {
    return value;
  },
  parseLiteral(ast) {
    if (ast.kind !== Kind.STRING) {
      throw new GraphQLError('SanitizedString must be a string');
    }
    return this.parseValue(ast.value);
  },
});
```

### SQL Injection Prevention

GraphQL does not inherently prevent SQL injection -- that responsibility falls on the resolver layer.

```typescript
// BAD: String interpolation → SQL injection vulnerable
const resolvers = {
  Query: {
    user: async (_, { id }, { db }) => {
      // NEVER do this
      return db.query(`SELECT * FROM users WHERE id = '${id}'`);
    },
  },
};

// GOOD: Parameterized queries
const resolvers = {
  Query: {
    user: async (_, { id }, { db }) => {
      return db.query('SELECT * FROM users WHERE id = $1', [id]);
    },
  },
};

// GOOD: ORM with built-in parameterization
const resolvers = {
  Query: {
    user: async (_, { id }, { prisma }) => {
      return prisma.user.findUnique({ where: { id } });
    },
  },
};
```

### Input Object Validation

```typescript
import { z } from 'zod';

// Define validation schema with Zod
const CreateUserSchema = z.object({
  name: z.string().min(1).max(100),
  email: z.string().email(),
  password: z.string().min(8).max(72),
  bio: z.string().max(500).optional(),
});

const resolvers = {
  Mutation: {
    createUser: async (_, { input }, { dataSources }) => {
      // Validate input before processing
      const validatedInput = CreateUserSchema.parse(input);
      return dataSources.users.create(validatedInput);
    },
  },
};
```

---

## 9. Timeout and Resource Limits

### Query Execution Timeout

```typescript
// Per-resolver timeout
const withTimeout = (resolver, timeoutMs = 5000) => {
  return async (...args) => {
    const timeoutPromise = new Promise((_, reject) =>
      setTimeout(
        () => reject(new GraphQLError(`Resolver timed out after ${timeoutMs}ms`)),
        timeoutMs
      )
    );

    return Promise.race([resolver(...args), timeoutPromise]);
  };
};

const resolvers = {
  Query: {
    searchProducts: withTimeout(
      async (_, { query }, { dataSources }) => {
        return dataSources.products.search(query);
      },
      3000 // 3-second timeout for search
    ),

    analyticsReport: withTimeout(
      async (_, { dateRange }, { dataSources }) => {
        return dataSources.analytics.generate(dateRange);
      },
      10000 // 10-second timeout for heavy reports
    ),
  },
};
```

### Request Size Limits

```typescript
import express from 'express';

const app = express();

// Limit request body size to prevent huge queries
app.use(express.json({ limit: '100kb' }));

// Custom middleware for additional checks
app.use('/graphql', (req, res, next) => {
  const queryLength = req.body?.query?.length || 0;

  if (queryLength > 10_000) {
    return res.status(413).json({
      errors: [{
        message: 'Query too large',
        extensions: {
          code: 'QUERY_TOO_LARGE',
          maxLength: 10_000,
          actualLength: queryLength,
        },
      }],
    });
  }

  next();
});
```

### Connection Pool Limits

```typescript
// Prevent a single query from consuming all database connections
import { Pool } from 'pg';

const pool = new Pool({
  max: 20,                    // Maximum 20 connections
  connectionTimeoutMillis: 5000, // Wait max 5s for a connection
  idleTimeoutMillis: 30000,     // Close idle connections after 30s
  statement_timeout: 10000,     // Kill queries running > 10s
});
```

---

## 10. Monitoring and Observability

### Slow Query Logging

```typescript
const slowQueryPlugin = {
  async requestDidStart({ request }) {
    const startTime = Date.now();

    return {
      async willSendResponse({ response }) {
        const duration = Date.now() - startTime;
        const SLOW_THRESHOLD = 1000; // 1 second

        if (duration > SLOW_THRESHOLD) {
          console.warn({
            message: 'Slow GraphQL query detected',
            operationName: request.operationName,
            duration: `${duration}ms`,
            query: request.query?.substring(0, 500), // Truncate for logging
            variables: request.variables,
            timestamp: new Date().toISOString(),
          });
        }
      },
    };
  },
};
```

### Operation Metrics with Prometheus

```typescript
import { Counter, Histogram, register } from 'prom-client';

const operationDuration = new Histogram({
  name: 'graphql_operation_duration_seconds',
  help: 'Duration of GraphQL operations',
  labelNames: ['operationName', 'operationType', 'status'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
});

const operationErrors = new Counter({
  name: 'graphql_operation_errors_total',
  help: 'Total GraphQL operation errors',
  labelNames: ['operationName', 'errorCode'],
});

const metricsPlugin = {
  async requestDidStart({ request }) {
    const timer = operationDuration.startTimer();

    return {
      async willSendResponse({ response }) {
        const hasErrors = response.body?.singleResult?.errors?.length > 0;

        timer({
          operationName: request.operationName || 'anonymous',
          operationType: 'query', // Determine from document
          status: hasErrors ? 'error' : 'success',
        });

        if (hasErrors) {
          for (const error of response.body.singleResult.errors) {
            operationErrors.inc({
              operationName: request.operationName || 'anonymous',
              errorCode: error.extensions?.code || 'UNKNOWN',
            });
          }
        }
      },
    };
  },
};

// Expose metrics endpoint
app.get('/metrics', async (req, res) => {
  res.set('Content-Type', register.contentType);
  res.end(await register.metrics());
});
```

### Tracing with OpenTelemetry

```typescript
import { NodeTracerProvider } from '@opentelemetry/sdk-trace-node';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { JaegerExporter } from '@opentelemetry/exporter-jaeger';

const provider = new NodeTracerProvider();
provider.addSpanProcessor(
  new SimpleSpanProcessor(
    new JaegerExporter({ endpoint: 'http://jaeger:14268/api/traces' })
  )
);
provider.register();

// In resolvers, create spans for expensive operations
const tracer = provider.getTracer('graphql-server');

const resolvers = {
  Query: {
    searchProducts: async (_, { query }, { dataSources }) => {
      const span = tracer.startSpan('searchProducts');
      span.setAttribute('search.query', query);

      try {
        const results = await dataSources.products.search(query);
        span.setAttribute('search.resultCount', results.length);
        return results;
      } catch (error) {
        span.recordException(error);
        throw error;
      } finally {
        span.end();
      }
    },
  },
};
```

---

## 11. Practice Problems

### Problem 1: Depth and Complexity Limits
Implement a combined depth + complexity limiter for the following schema. Set appropriate limits and test with queries of varying depth and cost:

```graphql
type Query {
  users(first: Int = 10): [User!]!
  product(id: ID!): Product
}

type User {
  id: ID!
  name: String!
  friends: [User!]!
  orders: [Order!]!
}

type Order {
  id: ID!
  items: [OrderItem!]!
}

type OrderItem {
  product: Product!
  quantity: Int!
}

type Product {
  id: ID!
  name: String!
  reviews: [Review!]!
}

type Review {
  author: User!
  body: String!
}
```

### Problem 2: Rate Limiter
Build a complexity-aware rate limiter that tracks per-user budgets in Redis. Each user gets 5,000 complexity points per minute. Implement the limiter as an Apollo Server plugin and include response headers showing remaining budget.

### Problem 3: Security Audit
Given a GraphQL server with no security measures, write a checklist of all protections it needs and implement the three most critical ones: (a) query depth limit, (b) introspection disabled in production, (c) input validation on all mutation arguments.

### Problem 4: Monitoring Dashboard
Design a monitoring setup using Prometheus and Grafana for a GraphQL server. Define at least 5 metrics to track (operation duration, error rate, complexity distribution, cache hit rate, resolver-level latency). Write the Prometheus metric definitions and a sample Grafana dashboard JSON.

### Problem 5: Alias Flood Defense
An attacker sends this query to your e-commerce API:

```graphql
query {
  a1: product(id: "1") { name price }
  a2: product(id: "2") { name price }
  # ... 998 more aliases
}
```

Implement three layers of defense: (a) alias count limit, (b) complexity analysis that counts aliases, (c) per-IP rate limiting that accounts for alias count. Test each layer independently.

---

## References

- [graphql-depth-limit](https://github.com/stems/graphql-depth-limit)
- [graphql-query-complexity](https://github.com/slicknode/graphql-query-complexity)
- [Apollo Server Security](https://www.apollographql.com/docs/apollo-server/security/authentication/)
- [OWASP GraphQL Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/GraphQL_Cheat_Sheet.html)
- [GraphQL Security Best Practices (The Guild)](https://the-guild.dev/blog/graphql-security)
- [Securing Your GraphQL API (Apollo Blog)](https://www.apollographql.com/blog/graphql/security/)

---

**Previous**: [Testing](./13_Testing.md) | **Next**: [REST to GraphQL Migration](./15_REST_to_GraphQL_Migration.md)
