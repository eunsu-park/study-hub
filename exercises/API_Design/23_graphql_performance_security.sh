#!/bin/bash
# Exercises for Lesson 23: GraphQL Performance and Security
# Topic: API_Design
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Complexity Calculator ==="
    cat << 'SOLUTION'
from graphql import parse

def calculate_complexity(query_str: str, cost_map: dict = None) -> int:
    """Calculate the complexity of a GraphQL query.

    cost_map: {"Type.field": cost} e.g. {"User.posts": 10, "Post.comments": 5}
    Scalar fields cost 0, object fields cost 1 by default.
    List arguments (first, limit) multiply child costs.
    """
    if cost_map is None:
        cost_map = {}

    document = parse(query_str)

    def _calc(selections, parent_type="Query", depth=0):
        total = 0
        for sel in selections:
            field_name = sel.name.value
            field_key = f"{parent_type}.{field_name}"

            # Get field cost
            base_cost = cost_map.get(field_key, 1 if sel.selection_set else 0)

            # Get list multiplier from arguments
            multiplier = 1
            for arg in sel.arguments:
                if arg.name.value in ("first", "limit", "last"):
                    multiplier = int(arg.value.value)

            # Calculate child cost
            child_cost = 0
            if sel.selection_set:
                child_cost = _calc(sel.selection_set.selections, field_name, depth + 1)

            total += base_cost + multiplier * child_cost
        return total

    for defn in document.definitions:
        return _calc(defn.selection_set.selections)

# Tests
assert calculate_complexity("{ user(id: 1) { username } }") == 1
assert calculate_complexity(
    "{ users(first: 10) { username } }",
    {"Query.users": 1}
) == 1 + 10 * 0  # 1 (users field) + 10 * 0 (scalar username)

assert calculate_complexity(
    '{ users(first: 10) { posts(first: 5) { title } } }',
    {"Query.users": 1, "User.posts": 10}
) == 1 + 10 * (10 + 5 * 0)  # 101
SOLUTION
}

exercise_2() {
    echo "=== Exercise 2: Security Audit ==="
    cat << 'SOLUTION'
# Vulnerabilities found:

# 1. No depth limiting — allows deeply nested queries to exhaust resources
# Fix: Add DepthLimitExtension (max_depth=10)

# 2. No complexity limiting — allows expensive queries
# Fix: Add QueryComplexityExtension (max_complexity=1000)

# 3. No breadth limiting — allows selection of hundreds of fields
# Fix: Add BreadthLimitExtension (max_fields=100)

# 4. No alias limiting — aliases can multiply expensive resolvers
# Fix: Add AliasLimitExtension (max_aliases=50)

# 5. Introspection enabled — exposes full schema to attackers
# Fix: Disable introspection in production

# 6. No rate limiting — vulnerable to DoS
# Fix: Add cost-based rate limiter

# 7. No request size limit — large payloads can exhaust memory
# Fix: Add RequestSizeLimitMiddleware (max 10KB)

# 8. No CORS configuration — any origin can make requests
# Fix: Add CORSMiddleware with explicit allowed origins

# 9. No timeout — slow queries can hold connections indefinitely
# Fix: Add TimeoutExtension (30s)

# 10. No authentication — all queries run as anonymous
# Fix: Add authentication context getter

# 11. No batch size limit — unbounded batch requests
# Fix: Add batch size middleware (max 10)

# Fixed configuration:
schema = strawberry.Schema(
    query=Query, mutation=Mutation,
    extensions=[
        DepthLimitExtension,
        QueryComplexityExtension,
        BreadthLimitExtension,
        AliasLimitExtension,
        TimeoutExtension,
        RateLimitExtension,
        DisableIntrospectionExtension,
    ],
)
app.add_middleware(CORSMiddleware, allow_origins=["https://app.example.com"])
app.add_middleware(RequestSizeLimitMiddleware)
SOLUTION
}

exercise_3() {
    echo "=== Exercise 3: Persisted Query Store ==="
    cat << 'SOLUTION'
import aioredis, hashlib, json

class RedisPersistedQueryStore:
    TTL = 86400  # 24 hours

    def __init__(self, redis_url, locked_mode=False):
        self.redis_url = redis_url
        self.locked_mode = locked_mode
        self._redis = None
        self.hits = 0
        self.misses = 0

    async def connect(self):
        self._redis = await aioredis.from_url(self.redis_url)

    async def get(self, hash_key: str) -> str | None:
        query = await self._redis.get(f"pq:{hash_key}")
        if query:
            self.hits += 1
            return query.decode()
        self.misses += 1
        return None

    async def put(self, hash_key: str, query: str) -> None:
        if self.locked_mode:
            raise ValueError("Locked mode: cannot register new queries")
        computed = hashlib.sha256(query.encode()).hexdigest()
        if computed != hash_key:
            raise ValueError("Hash mismatch")
        await self._redis.setex(f"pq:{hash_key}", self.TTL, query)

    @property
    def hit_ratio(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
SOLUTION
}

main() { exercise_1; echo ""; exercise_2; echo ""; exercise_3; }
main "$@"
