# 23. GraphQL Performance and Security

**Previous**: [GraphQL Federation](./22_GraphQL_Federation.md) | **Next**: [GraphQL Testing and Tooling](./24_GraphQL_Testing_and_Tooling.md)

**Difficulty**: ⭐⭐⭐⭐

---

## Learning Objectives

- Analyze query complexity and implement cost-based query limiting
- Apply depth limiting and breadth limiting to prevent resource exhaustion
- Implement persisted queries for performance and security benefits
- Design caching strategies for GraphQL at multiple levels (client, CDN, application)
- Apply rate limiting strategies specific to GraphQL's single-endpoint model
- Protect GraphQL APIs against common attack vectors (injection, batching abuse, introspection leaking)

---

## Table of Contents

1. [GraphQL Security Landscape](#1-graphql-security-landscape)
2. [Query Complexity Analysis](#2-query-complexity-analysis)
3. [Depth and Breadth Limiting](#3-depth-and-breadth-limiting)
4. [Persisted Queries](#4-persisted-queries)
5. [Caching Strategies](#5-caching-strategies)
6. [Rate Limiting](#6-rate-limiting)
7. [Input Validation and Injection Prevention](#7-input-validation-and-injection-prevention)
8. [Authentication and Authorization Hardening](#8-authentication-and-authorization-hardening)
9. [Denial of Service Protection](#9-denial-of-service-protection)
10. [Monitoring and Observability](#10-monitoring-and-observability)
11. [Exercises](#11-exercises)
12. [References](#12-references)

---

## 1. GraphQL Security Landscape

GraphQL introduces unique security challenges compared to REST because clients control the query structure.

### Unique Attack Surfaces

| Attack | Description | REST Equivalent |
|--------|-------------|----------------|
| **Deep nesting** | Recursive queries that exhaust server resources | No direct equivalent |
| **Wide queries** | Selecting hundreds of fields simultaneously | Fixed by endpoint design |
| **Batch abuse** | Sending many operations in one request | Rate limiting per endpoint |
| **Introspection leak** | Exposing schema to attackers | API docs exposure |
| **Alias abuse** | Using aliases to multiply expensive resolvers | No direct equivalent |
| **Field suggestion** | Error messages reveal schema structure | Varies |

### Defense-in-Depth Strategy

```
Layer 1: Network         → WAF, TLS, IP allowlisting
Layer 2: Transport       → Rate limiting, request size limits
Layer 3: Query Analysis  → Complexity, depth, breadth limits
Layer 4: Execution       → Timeout, resolver-level auth
Layer 5: Data            → Field-level authorization, filtering
Layer 6: Response        → Error sanitization, no stack traces
```

---

## 2. Query Complexity Analysis

Query complexity analysis assigns a cost to each field and rejects queries that exceed a threshold.

### Static Analysis

Assign costs to fields based on their expected resource usage:

```python
import strawberry
from strawberry.extensions import SchemaExtension


# Cost annotations on types
@strawberry.type
class User:
    id: strawberry.ID                   # cost: 0 (trivial)
    username: str                        # cost: 0 (trivial)
    email: str                           # cost: 0 (trivial)

    @strawberry.field(
        extensions=[{"cost": 10}]        # cost: 10 (DB query)
    )
    async def posts(
        self, info, first: int = 10
    ) -> list["Post"]:
        return await load_posts(self.id, first)

    @strawberry.field(extensions=[{"cost": 5}])
    async def followers(
        self, info, first: int = 10
    ) -> list["User"]:
        return await load_followers(self.id, first)
```

### Complexity Calculator

```python
class QueryComplexityExtension(SchemaExtension):
    """Reject queries exceeding a complexity threshold."""
    MAX_COMPLEXITY = 1000

    def on_operation(self):
        complexity = self._calculate_complexity(
            self.execution_context.graphql_document,
            self.execution_context.schema,
        )
        if complexity > self.MAX_COMPLEXITY:
            raise ValueError(
                f"Query complexity {complexity} exceeds maximum of "
                f"{self.MAX_COMPLEXITY}. Simplify your query."
            )
        yield

    def _calculate_complexity(self, document, schema, depth=0):
        """Recursively calculate query complexity."""
        total = 0
        for definition in document.definitions:
            for selection in definition.selection_set.selections:
                field_cost = self._get_field_cost(selection, schema)
                multiplier = self._get_list_multiplier(selection)
                child_cost = 0
                if selection.selection_set:
                    child_cost = self._calculate_selections(
                        selection.selection_set.selections,
                        schema,
                        depth + 1,
                    )
                total += field_cost + (multiplier * child_cost)
        return total

    def _get_field_cost(self, selection, schema) -> int:
        """Get the cost of a field from its extensions."""
        # Default cost: 1 for each field
        return 1

    def _get_list_multiplier(self, selection) -> int:
        """Get the list size multiplier from 'first'/'limit' arguments."""
        for arg in selection.arguments:
            if arg.name.value in ("first", "limit"):
                return int(arg.value.value)
        return 1  # Default: not a list


schema = strawberry.Schema(
    query=Query,
    extensions=[QueryComplexityExtension],
)
```

### Example Cost Calculation

```graphql
query {
  users(first: 10) {            # cost: 1 + 10 * child
    username                     # cost: 0
    posts(first: 5) {            # cost: 10 + 5 * child
      title                      # cost: 0
      comments(first: 3) {       # cost: 5 + 3 * child
        body                     # cost: 0
        author {                 # cost: 1
          username               # cost: 0
        }
      }
    }
  }
}

# Total: 1 + 10 * (0 + 10 + 5 * (0 + 5 + 3 * (0 + 1 + 0))) = 1 + 10 * (10 + 5 * 8)
#       = 1 + 10 * 50 = 501
```

---

## 3. Depth and Breadth Limiting

### Depth Limiting

Prevent deeply nested queries:

```python
class DepthLimitExtension(SchemaExtension):
    MAX_DEPTH = 10

    def on_operation(self):
        depth = self._calculate_depth(
            self.execution_context.graphql_document
        )
        if depth > self.MAX_DEPTH:
            raise ValueError(
                f"Query depth {depth} exceeds maximum of {self.MAX_DEPTH}"
            )
        yield

    def _calculate_depth(self, document) -> int:
        max_depth = 0
        for definition in document.definitions:
            depth = self._selection_depth(definition.selection_set, 0)
            max_depth = max(max_depth, depth)
        return max_depth

    def _selection_depth(self, selection_set, current_depth) -> int:
        if not selection_set:
            return current_depth
        max_child = current_depth
        for selection in selection_set.selections:
            if hasattr(selection, "selection_set") and selection.selection_set:
                child = self._selection_depth(
                    selection.selection_set, current_depth + 1
                )
                max_child = max(max_child, child)
        return max_child
```

### Breadth Limiting

Prevent wide queries that select too many fields:

```python
class BreadthLimitExtension(SchemaExtension):
    MAX_BREADTH = 100  # Max total field selections

    def on_operation(self):
        breadth = self._count_selections(
            self.execution_context.graphql_document
        )
        if breadth > self.MAX_BREADTH:
            raise ValueError(
                f"Query selects {breadth} fields, exceeding limit of "
                f"{self.MAX_BREADTH}"
            )
        yield

    def _count_selections(self, document) -> int:
        total = 0
        for definition in document.definitions:
            total += self._count_in_set(definition.selection_set)
        return total

    def _count_in_set(self, selection_set) -> int:
        if not selection_set:
            return 0
        count = len(selection_set.selections)
        for selection in selection_set.selections:
            if hasattr(selection, "selection_set") and selection.selection_set:
                count += self._count_in_set(selection.selection_set)
        return count
```

### Alias Abuse Prevention

Aliases can multiply resolver calls:

```graphql
# Malicious: calls user resolver 1000 times
query {
  u1: user(id: "1") { username }
  u2: user(id: "2") { username }
  u3: user(id: "3") { username }
  # ... 997 more aliases
}
```

```python
class AliasLimitExtension(SchemaExtension):
    MAX_ALIASES = 50

    def on_operation(self):
        alias_count = self._count_aliases(
            self.execution_context.graphql_document
        )
        if alias_count > self.MAX_ALIASES:
            raise ValueError(
                f"Query uses {alias_count} aliases, exceeding limit of "
                f"{self.MAX_ALIASES}"
            )
        yield

    def _count_aliases(self, document) -> int:
        count = 0
        for definition in document.definitions:
            count += self._count_in_set(definition.selection_set)
        return count

    def _count_in_set(self, selection_set) -> int:
        if not selection_set:
            return 0
        count = sum(
            1 for s in selection_set.selections
            if hasattr(s, "alias") and s.alias
        )
        for selection in selection_set.selections:
            if hasattr(selection, "selection_set") and selection.selection_set:
                count += self._count_in_set(selection.selection_set)
        return count
```

### Recommended Limits

| Limit | Public API | Internal API |
|-------|-----------|-------------|
| Max depth | 7-10 | 15-20 |
| Max breadth | 50-100 | 200-500 |
| Max complexity | 500-1000 | 5000-10000 |
| Max aliases | 20-50 | 100 |
| Request size | 10 KB | 100 KB |

---

## 4. Persisted Queries

### What Are Persisted Queries?

Instead of sending the full query string, clients send a hash. The server looks up the query by hash.

### Automatic Persisted Queries (APQ)

```
First request (cache miss):
  Client → { query: "...", extensions: { persistedQuery: { sha256Hash: "abc123" } } }
  Server → { errors: [{ message: "PersistedQueryNotFound" }] }

  Client → { query: "{ user(id: 1) { name } }", extensions: { persistedQuery: { sha256Hash: "abc123" } } }
  Server → stores hash → { data: { user: { name: "Alice" } } }

Subsequent requests (cache hit):
  Client → { extensions: { persistedQuery: { sha256Hash: "abc123" } } }
  Server → looks up hash → { data: { user: { name: "Alice" } } }
```

### Implementation

```python
import hashlib
from functools import lru_cache

# In-memory store (use Redis in production)
_query_store: dict[str, str] = {}


class PersistedQueryExtension(SchemaExtension):
    """Support for Automatic Persisted Queries (APQ)."""

    def on_operation(self):
        request = self.execution_context
        extensions = request.extensions or {}
        persisted = extensions.get("persistedQuery", {})
        sha256_hash = persisted.get("sha256Hash")

        if sha256_hash:
            if request.query:
                # Client sent query + hash: store it
                computed_hash = hashlib.sha256(
                    request.query.encode()
                ).hexdigest()
                if computed_hash != sha256_hash:
                    raise ValueError("Persisted query hash mismatch")
                _query_store[sha256_hash] = request.query
            else:
                # Client sent only hash: look it up
                stored_query = _query_store.get(sha256_hash)
                if not stored_query:
                    raise ValueError("PersistedQueryNotFound")
                request.query = stored_query

        yield
```

### Locked Persisted Queries (Build-Time)

For maximum security, only allow pre-registered queries:

```python
# Build step: extract queries from client code
# queries.json
ALLOWED_QUERIES = {
    "abc123": "query GetUser($id: ID!) { user(id: $id) { username email } }",
    "def456": "mutation CreatePost($input: CreatePostInput!) { createPost(input: $input) { post { id } } }",
}


class StrictPersistedQueryExtension(SchemaExtension):
    """Only allow pre-registered queries (no ad-hoc queries)."""

    def on_operation(self):
        extensions = self.execution_context.extensions or {}
        persisted = extensions.get("persistedQuery", {})
        query_id = persisted.get("sha256Hash")

        if not query_id:
            raise ValueError(
                "Ad-hoc queries are not allowed. Use persisted queries."
            )

        stored = ALLOWED_QUERIES.get(query_id)
        if not stored:
            raise ValueError("Unknown persisted query ID")

        self.execution_context.query = stored
        yield
```

### Benefits

| Benefit | Description |
|---------|-------------|
| **Smaller payloads** | Hash (64 bytes) vs. full query (potentially KB) |
| **CDN caching** | GET requests with query hash are cacheable |
| **Security** | Locked mode prevents arbitrary query execution |
| **Performance** | Skip parsing for known queries |

---

## 5. Caching Strategies

GraphQL caching is more complex than REST because of the single-endpoint model.

### Cache Levels

```
Client Cache (Apollo Client, urql)
    ↓
CDN / Edge Cache (Cloudflare, Fastly)
    ↓
Application Cache (Redis, in-memory)
    ↓
DataLoader Cache (per-request)
    ↓
Database Cache (query cache, connection pool)
```

### Client-Side Normalized Cache

```javascript
// Apollo Client normalized cache
import { InMemoryCache } from '@apollo/client';

const cache = new InMemoryCache({
  typePolicies: {
    User: {
      keyFields: ['id'],  // Cache key: User:42
    },
    Post: {
      keyFields: ['id'],
      fields: {
        comments: {
          // Merge paginated results
          keyArgs: false,
          merge(existing = [], incoming) {
            return [...existing, ...incoming];
          },
        },
      },
    },
  },
});
```

### Response Caching

```python
import hashlib
import json
from datetime import timedelta


class ResponseCacheExtension(SchemaExtension):
    """Cache full GraphQL responses by query hash."""

    def __init__(self, *, redis_client, default_ttl: int = 60):
        self.redis = redis_client
        self.default_ttl = default_ttl

    def on_operation(self):
        # Build cache key from query + variables
        query = self.execution_context.query
        variables = self.execution_context.variables or {}
        cache_key = self._build_key(query, variables)

        # Check cache
        cached = self.redis.get(cache_key)
        if cached:
            self.execution_context.result = json.loads(cached)
            return  # Skip execution

        yield  # Execute query

        # Store result
        result = self.execution_context.result
        if result and not result.errors:
            self.redis.setex(
                cache_key,
                self.default_ttl,
                json.dumps(result.data),
            )

    def _build_key(self, query: str, variables: dict) -> str:
        raw = json.dumps({"q": query, "v": variables}, sort_keys=True)
        return f"gql:cache:{hashlib.sha256(raw.encode()).hexdigest()}"
```

### Cache-Control Directives

```graphql
# Schema-level cache hints
type Query {
  posts(first: Int): [Post!]! @cacheControl(maxAge: 60)
  me: User @cacheControl(maxAge: 0, scope: PRIVATE)
}

type Post @cacheControl(maxAge: 300) {
  id: ID!
  title: String!
  viewCount: Int! @cacheControl(maxAge: 10)  # More volatile field
}
```

### CDN Caching with Persisted Queries

```
# Persisted queries enable GET requests, which CDNs can cache:
GET /graphql?extensions={"persistedQuery":{"sha256Hash":"abc123"}}&variables={"id":"1"}

# CDN cache key: URL + query params
# TTL from Cache-Control header
```

---

## 6. Rate Limiting

### Challenges with GraphQL Rate Limiting

REST rate limiting is simple: limit requests per endpoint. GraphQL sends all operations to one endpoint, so per-endpoint limits are ineffective.

### Cost-Based Rate Limiting

```python
from collections import defaultdict
from datetime import datetime, timedelta


class CostBasedRateLimiter:
    """Rate limit based on query complexity cost."""

    def __init__(self, max_cost_per_minute: int = 1000):
        self.max_cost = max_cost_per_minute
        self.window = timedelta(minutes=1)
        self._usage: dict[str, list[tuple[datetime, int]]] = defaultdict(list)

    def check(self, client_id: str, query_cost: int) -> bool:
        """Return True if the request is allowed."""
        now = datetime.now()
        cutoff = now - self.window

        # Clean expired entries
        self._usage[client_id] = [
            (ts, cost) for ts, cost in self._usage[client_id]
            if ts > cutoff
        ]

        # Calculate current usage
        current_cost = sum(cost for _, cost in self._usage[client_id])

        if current_cost + query_cost > self.max_cost:
            return False

        self._usage[client_id].append((now, query_cost))
        return True

    def remaining(self, client_id: str) -> int:
        """Return remaining cost budget."""
        now = datetime.now()
        cutoff = now - self.window
        current = sum(
            cost for ts, cost in self._usage[client_id]
            if ts > cutoff
        )
        return max(0, self.max_cost - current)
```

### Rate Limit Headers

```python
class RateLimitExtension(SchemaExtension):
    def on_operation(self):
        client_id = self._get_client_id()
        query_cost = self._calculate_cost()

        if not rate_limiter.check(client_id, query_cost):
            remaining = rate_limiter.remaining(client_id)
            raise ValueError(
                f"Rate limit exceeded. Remaining budget: {remaining}. "
                f"Query cost: {query_cost}."
            )

        yield

        # Add rate limit info to extensions
        result = self.execution_context.result
        if result:
            result.extensions = result.extensions or {}
            result.extensions["rateLimit"] = {
                "cost": query_cost,
                "remaining": rate_limiter.remaining(client_id),
                "resetAt": (datetime.now() + timedelta(minutes=1)).isoformat(),
            }
```

### GitHub-Style Rate Limiting

GitHub's GraphQL API uses a point-based system:

```json
{
  "data": { "...": "..." },
  "extensions": {
    "rateLimit": {
      "limit": 5000,
      "cost": 12,
      "remaining": 4988,
      "resetAt": "2024-01-01T00:00:00Z",
      "nodeCount": 42
    }
  }
}
```

---

## 7. Input Validation and Injection Prevention

### Query Injection

Unlike SQL injection, GraphQL query injection occurs when user input is interpolated into query strings:

```python
# VULNERABLE — never do this
query = f'{{ user(name: "{user_input}") {{ email }} }}'

# SAFE — always use variables
query = """
    query GetUser($name: String!) {
        user(name: $name) { email }
    }
"""
variables = {"name": user_input}
```

### Input Validation

```python
@strawberry.input
class SearchInput:
    query: str
    limit: int = 10

    def __post_init__(self):
        if len(self.query) > 200:
            raise ValueError("Search query too long (max 200 characters)")
        if self.limit < 1 or self.limit > 100:
            raise ValueError("Limit must be between 1 and 100")
        # Sanitize: remove potential injection characters
        self.query = self.query.replace("\x00", "")
```

### Request Size Limiting

```python
from starlette.middleware.base import BaseHTTPMiddleware

MAX_REQUEST_SIZE = 10 * 1024  # 10 KB


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if request.method == "POST":
            content_length = request.headers.get("content-length")
            if content_length and int(content_length) > MAX_REQUEST_SIZE:
                return JSONResponse(
                    {"error": f"Request too large (max {MAX_REQUEST_SIZE} bytes)"},
                    status_code=413,
                )
        return await call_next(request)
```

---

## 8. Authentication and Authorization Hardening

### Token Validation

```python
async def get_context(request: Request) -> RequestContext:
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header else None

    current_user = None
    if token:
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            current_user = await user_repo.find_by_id(payload["sub"])
        except jwt.ExpiredSignatureError:
            pass  # Token expired — user is anonymous
        except jwt.InvalidTokenError:
            pass  # Invalid token — user is anonymous

    return RequestContext(
        current_user=current_user,
        request=request,
    )
```

### Operation-Level Authorization

```python
# Whitelist operations for API key auth
ALLOWED_OPERATIONS = {
    "api_key": {"GetProduct", "ListProducts", "SearchProducts"},
    "oauth": None,  # All operations allowed
}


class AuthorizationExtension(SchemaExtension):
    def on_operation(self):
        auth_type = self.execution_context.context.auth_type
        op_name = self.execution_context.operation_name

        allowed = ALLOWED_OPERATIONS.get(auth_type)
        if allowed is not None and op_name not in allowed:
            raise PermissionError(
                f"Operation '{op_name}' not allowed for {auth_type} auth"
            )
        yield
```

### Disable Introspection in Production

```python
class DisableIntrospectionExtension(SchemaExtension):
    def on_operation(self):
        query = self.execution_context.query or ""
        if "__schema" in query or "__type" in query:
            if not self.execution_context.context.is_admin:
                raise ValueError("Introspection is disabled")
        yield
```

---

## 9. Denial of Service Protection

### Timeout Per Operation

```python
import asyncio


class TimeoutExtension(SchemaExtension):
    TIMEOUT_SECONDS = 30

    def on_operation(self):
        try:
            with asyncio.timeout(self.TIMEOUT_SECONDS):
                yield
        except asyncio.TimeoutError:
            raise ValueError(
                f"Query execution exceeded {self.TIMEOUT_SECONDS}s timeout"
            )
```

### Batch Size Limiting

```python
@app.middleware("http")
async def limit_batch(request: Request, call_next):
    if request.url.path == "/graphql" and request.method == "POST":
        body = await request.body()
        try:
            parsed = json.loads(body)
            if isinstance(parsed, list) and len(parsed) > 10:
                return JSONResponse(
                    {"error": "Maximum batch size is 10"},
                    status_code=400,
                )
        except json.JSONDecodeError:
            pass
    return await call_next(request)
```

### Combined Security Configuration

```python
schema = strawberry.Schema(
    query=Query,
    mutation=Mutation,
    extensions=[
        DepthLimitExtension,          # Max depth: 10
        BreadthLimitExtension,        # Max fields: 100
        QueryComplexityExtension,     # Max cost: 1000
        AliasLimitExtension,          # Max aliases: 50
        TimeoutExtension,             # Max time: 30s
        RateLimitExtension,           # Cost-based rate limit
        DisableIntrospectionExtension,# No introspection in prod
        ErrorLoggingExtension,        # Log all errors
        TimingExtension,              # Performance tracking
    ],
)
```

---

## 10. Monitoring and Observability

### Key Metrics

| Metric | Description | Alert Threshold |
|--------|-------------|----------------|
| `graphql.operation.duration` | Query execution time | p99 > 5s |
| `graphql.operation.errors` | Error count per operation | > 5% error rate |
| `graphql.query.complexity` | Cost per query | > 80% of limit |
| `graphql.query.depth` | Nesting depth per query | > 8 |
| `graphql.resolver.duration` | Per-resolver timing | p99 > 1s |
| `graphql.dataloader.batch_size` | DataLoader batch sizes | Avg < 2 (N+1 detected) |
| `graphql.rate_limit.rejections` | Rejected requests | > 10/min |

### Structured Logging

```python
import structlog

logger = structlog.get_logger()


class ObservabilityExtension(SchemaExtension):
    def on_operation(self):
        start = time.monotonic()
        yield
        duration = time.monotonic() - start

        result = self.execution_context.result
        has_errors = bool(result and result.errors)

        logger.info(
            "graphql.operation",
            operation_name=self.execution_context.operation_name,
            duration_ms=round(duration * 1000, 2),
            has_errors=has_errors,
            error_count=len(result.errors) if result and result.errors else 0,
            client_id=self.execution_context.context.client_id,
        )
```

### Apollo Tracing Format

```json
{
  "extensions": {
    "tracing": {
      "version": 1,
      "startTime": "2024-01-01T00:00:00.000Z",
      "endTime": "2024-01-01T00:00:00.045Z",
      "duration": 45000000,
      "execution": {
        "resolvers": [
          {
            "path": ["user"],
            "parentType": "Query",
            "fieldName": "user",
            "returnType": "User",
            "startOffset": 1000000,
            "duration": 20000000
          },
          {
            "path": ["user", "posts"],
            "parentType": "User",
            "fieldName": "posts",
            "returnType": "[Post!]!",
            "startOffset": 21000000,
            "duration": 15000000
          }
        ]
      }
    }
  }
}
```

---

## 11. Exercises

### Exercise 1: Complexity Calculator

Implement a complete query complexity calculator that:
- Assigns cost 0 to scalar fields
- Assigns cost 1 to object fields
- Multiplies child costs by list arguments (`first`, `limit`)
- Tests against known queries with expected costs

### Exercise 2: Security Audit

Audit the following GraphQL server configuration and identify all vulnerabilities:

```python
schema = strawberry.Schema(query=Query, mutation=Mutation)
app = FastAPI()
app.include_router(GraphQLRouter(schema), prefix="/graphql")
# No middleware, no extensions, introspection enabled
```

List at least 8 security issues and provide fixes.

### Exercise 3: Persisted Query Store

Implement a Redis-backed persisted query store with:
- APQ protocol support (negotiation + caching)
- TTL-based expiration (24 hours)
- Locked mode toggle (reject unknown queries)
- Metrics: cache hit/miss ratio

### Exercise 4: Rate Limiter

Build a cost-based rate limiter with:
- Redis-backed sliding window (not in-memory)
- Per-user and per-API-key limits
- Rate limit response headers
- Configurable cost multipliers per operation type (query=1x, mutation=2x)

---

## 12. References

### Security
- [OWASP GraphQL Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/GraphQL_Cheat_Sheet.html)
- [GraphQL Security Best Practices — Escape.tech](https://escape.tech/blog/graphql-security-best-practices/)
- [How to Secure a GraphQL API — Apollo](https://www.apollographql.com/docs/technotes/TN0021-graph-security/)

### Performance
- [Apollo Persisted Queries](https://www.apollographql.com/docs/apollo-server/performance/apq/)
- [GraphQL Caching Strategies — The Guild](https://the-guild.dev/blog/graphql-response-caching)

### Monitoring
- [Apollo Studio](https://studio.apollographql.com/) — Operation metrics and tracing
- [GraphQL Inspector — CI/CD Schema Checks](https://graphql-inspector.com/)

---

**License**: CC BY-NC 4.0
