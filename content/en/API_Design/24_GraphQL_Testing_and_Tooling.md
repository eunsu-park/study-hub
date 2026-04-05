# 24. GraphQL Testing and Tooling

**Previous**: [GraphQL Performance and Security](./23_GraphQL_Performance_Security.md) | **Next**: [API Capstone — Unified Gateway](./25_API_Capstone_Unified_Gateway.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Write unit tests for GraphQL resolvers using Strawberry's test client
- Implement integration tests that validate full query execution against a test database
- Use snapshot testing to detect unintended schema and response changes
- Generate typed client code from GraphQL schemas using code generation tools
- Leverage GraphiQL and Apollo Sandbox for interactive API exploration
- Set up CI pipelines with schema linting, breaking change detection, and contract tests

---

## Table of Contents

1. [Testing Strategy Overview](#1-testing-strategy-overview)
2. [Unit Testing Resolvers](#2-unit-testing-resolvers)
3. [Integration Testing](#3-integration-testing)
4. [Snapshot Testing](#4-snapshot-testing)
5. [Mocking and Test Fixtures](#5-mocking-and-test-fixtures)
6. [Schema Validation and Linting](#6-schema-validation-and-linting)
7. [Code Generation](#7-code-generation)
8. [Interactive Tools](#8-interactive-tools)
9. [CI/CD Pipeline](#9-cicd-pipeline)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Testing Strategy Overview

### The Testing Pyramid for GraphQL

```
          ╱ E2E Tests  ╲
         ╱  (Browser +   ╲
        ╱   GraphQL API)   ╲
       ╱─────────────────────╲
      ╱  Integration Tests     ╲
     ╱  (Full query execution   ╲
    ╱   with test database)      ╲
   ╱───────────────────────────────╲
  ╱       Unit Tests                 ╲
 ╱  (Resolvers, services,            ╲
╱   DataLoaders in isolation)          ╲
──────────────────────────────────────────
```

### What to Test at Each Level

| Level | What to Test | Tools |
|-------|-------------|-------|
| **Unit** | Resolver logic, service methods, validators | pytest, mock |
| **Integration** | Full query execution, DataLoader batching | Strawberry TestClient, test DB |
| **Schema** | Breaking changes, linting, deprecation | GraphQL Inspector, Rover |
| **Contract** | Client-server agreement | snapshot tests |
| **E2E** | Full user flows via API | httpx, Playwright |

---

## 2. Unit Testing Resolvers

### Testing with Strawberry TestClient

```python
# tests/conftest.py
import pytest
import strawberry
from strawberry.test import GraphQLTestClient
from schema import schema
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_context():
    """Create a mock request context."""
    ctx = MagicMock()
    ctx.current_user = MagicMock(id="1", username="testuser", role="USER")
    ctx.db = MagicMock()
    ctx.dataloaders = MagicMock()
    return ctx


@pytest.fixture
def client(mock_context):
    """GraphQL test client with mocked context."""
    return GraphQLTestClient(schema, context_value=mock_context)
```

### Testing Queries

```python
# tests/test_user_queries.py
import pytest


def test_get_user(client, mock_context):
    """Test fetching a single user by ID."""
    mock_context.db.users.find_by_id = AsyncMock(return_value={
        "id": "1",
        "username": "alice",
        "email": "alice@example.com",
        "bio": "Engineer",
    })

    result = client.query("""
        query GetUser($id: ID!) {
            user(id: $id) {
                id
                username
                email
                bio
            }
        }
    """, variables={"id": "1"})

    assert result.errors is None
    assert result.data["user"]["username"] == "alice"
    assert result.data["user"]["email"] == "alice@example.com"


def test_get_user_not_found(client, mock_context):
    """Test fetching a non-existent user returns null."""
    mock_context.db.users.find_by_id = AsyncMock(return_value=None)

    result = client.query("""
        query { user(id: "999") { username } }
    """)

    assert result.errors is None
    assert result.data["user"] is None


def test_list_users_with_pagination(client, mock_context):
    """Test paginated user listing."""
    mock_context.db.users.find_all = AsyncMock(return_value=[
        {"id": "1", "username": "alice"},
        {"id": "2", "username": "bob"},
    ])

    result = client.query("""
        query { users(limit: 2) { id username } }
    """)

    assert result.errors is None
    assert len(result.data["users"]) == 2
```

### Testing Mutations

```python
# tests/test_post_mutations.py
def test_create_post_success(client, mock_context):
    """Test successful post creation."""
    mock_context.db.posts.create = AsyncMock(return_value={
        "id": "101",
        "title": "Test Post",
        "content": "Hello World",
        "status": "DRAFT",
    })

    result = client.query("""
        mutation CreatePost($input: CreatePostInput!) {
            createPost(input: $input) {
                post {
                    id
                    title
                    status
                }
                userErrors {
                    field
                    message
                    code
                }
            }
        }
    """, variables={
        "input": {
            "title": "Test Post",
            "content": "Hello World",
        }
    })

    assert result.errors is None
    payload = result.data["createPost"]
    assert payload["post"]["title"] == "Test Post"
    assert len(payload["userErrors"]) == 0


def test_create_post_validation_error(client, mock_context):
    """Test post creation with invalid input."""
    result = client.query("""
        mutation CreatePost($input: CreatePostInput!) {
            createPost(input: $input) {
                post { id }
                userErrors { field message code }
            }
        }
    """, variables={
        "input": {
            "title": "",           # Empty title
            "content": "short",    # Too short
        }
    })

    assert result.errors is None
    payload = result.data["createPost"]
    assert payload["post"] is None
    assert len(payload["userErrors"]) > 0
    assert any(e["code"] == "BLANK" for e in payload["userErrors"])
```

### Testing Authorization

```python
def test_admin_query_as_regular_user(client, mock_context):
    """Test that non-admin users cannot access admin queries."""
    mock_context.current_user.role = "USER"

    result = client.query("""
        query { adminDashboard { totalUsers totalPosts } }
    """)

    assert result.errors is not None
    assert "Admin access required" in result.errors[0].message


def test_admin_query_as_admin(client, mock_context):
    """Test that admin users can access admin queries."""
    mock_context.current_user.role = "ADMIN"
    mock_context.db.dashboard.get = AsyncMock(return_value={
        "total_users": 100,
        "total_posts": 500,
    })

    result = client.query("""
        query { adminDashboard { totalUsers totalPosts } }
    """)

    assert result.errors is None
    assert result.data["adminDashboard"]["totalUsers"] == 100
```

---

## 3. Integration Testing

### Test Database Setup

```python
# tests/conftest.py
import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def engine():
    engine = create_async_engine(TEST_DATABASE_URL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(engine):
    session_factory = async_sessionmaker(engine)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def seeded_db(db_session):
    """Seed test database with sample data."""
    users = [
        UserModel(id="1", username="alice", email="alice@test.com"),
        UserModel(id="2", username="bob", email="bob@test.com"),
    ]
    posts = [
        PostModel(id="101", title="First Post", author_id="1"),
        PostModel(id="102", title="Second Post", author_id="1"),
        PostModel(id="103", title="Bob's Post", author_id="2"),
    ]
    db_session.add_all(users + posts)
    await db_session.commit()
    return db_session
```

### Full Query Integration Test

```python
# tests/test_integration.py
@pytest.mark.asyncio
async def test_user_with_posts_integration(seeded_db):
    """Integration test: fetch user with related posts."""
    context = RequestContext(db=seeded_db, current_user=None)
    client = GraphQLTestClient(schema, context_value=context)

    result = client.query("""
        query {
            user(id: "1") {
                username
                posts(first: 10) {
                    title
                }
                postCount
            }
        }
    """)

    assert result.errors is None
    user = result.data["user"]
    assert user["username"] == "alice"
    assert len(user["posts"]) == 2
    assert user["postCount"] == 2


@pytest.mark.asyncio
async def test_dataloader_batching(seeded_db):
    """Verify DataLoader batches queries (no N+1)."""
    context = RequestContext(db=seeded_db, current_user=None)
    client = GraphQLTestClient(schema, context_value=context)

    # This query would cause N+1 without DataLoader
    result = client.query("""
        query {
            posts(first: 3) {
                title
                author { username }
            }
        }
    """)

    assert result.errors is None
    assert len(result.data["posts"]) == 3
    # All authors should be resolved
    for post in result.data["posts"]:
        assert post["author"]["username"] is not None
```

---

## 4. Snapshot Testing

Snapshot tests capture the expected output and alert you to changes.

### Schema Snapshot

```python
# tests/test_schema_snapshot.py
import strawberry
from schema import schema


def test_schema_snapshot(snapshot):
    """Ensure schema hasn't changed unexpectedly."""
    schema_str = strawberry.printer.print_schema(schema)
    snapshot.assert_match(schema_str, "schema.graphql")
```

### Response Snapshot

```python
def test_user_response_snapshot(client, snapshot):
    """Snapshot test for user query response shape."""
    result = client.query("""
        query { user(id: "1") { id username email bio createdAt } }
    """)
    snapshot.assert_match(
        json.dumps(result.data, indent=2, default=str),
        "user_response.json",
    )
```

### Using pytest-snapshot

```bash
# Generate initial snapshots
pytest --snapshot-update

# Run tests (fails if snapshots differ)
pytest

# Review and update after intentional changes
pytest --snapshot-update
```

---

## 5. Mocking and Test Fixtures

### Mocking DataLoaders

```python
@pytest.fixture
def mock_dataloaders():
    loaders = MagicMock()
    loaders.user_loader.load = AsyncMock(side_effect=lambda id: {
        "1": User(id="1", username="alice", email="alice@test.com"),
        "2": User(id="2", username="bob", email="bob@test.com"),
    }.get(id))
    loaders.posts_by_author.load = AsyncMock(side_effect=lambda id: {
        "1": [
            Post(id="101", title="Post A"),
            Post(id="102", title="Post B"),
        ],
        "2": [],
    }.get(id, []))
    return loaders
```

### Factory Fixtures

```python
# tests/factories.py
from dataclasses import dataclass
from datetime import datetime
import itertools

_id_counter = itertools.count(1)


def make_user(**overrides) -> User:
    user_id = str(next(_id_counter))
    defaults = {
        "id": user_id,
        "username": f"user_{user_id}",
        "email": f"user_{user_id}@test.com",
        "bio": None,
        "created_at": datetime(2024, 1, 1),
    }
    defaults.update(overrides)
    return User(**defaults)


def make_post(**overrides) -> Post:
    post_id = str(next(_id_counter))
    defaults = {
        "id": post_id,
        "title": f"Post {post_id}",
        "content": f"Content for post {post_id}",
        "status": PostStatus.DRAFT,
        "author_id": "1",
        "created_at": datetime(2024, 1, 1),
    }
    defaults.update(overrides)
    return Post(**defaults)
```

### Parameterized Tests

```python
@pytest.mark.parametrize("status,expected_count", [
    (None, 3),           # All posts
    ("PUBLISHED", 2),    # Only published
    ("DRAFT", 1),        # Only drafts
])
def test_posts_filter_by_status(client, mock_context, status, expected_count):
    """Test filtering posts by status."""
    mock_context.db.posts.find_by_status = AsyncMock(return_value=[
        make_post() for _ in range(expected_count)
    ])

    variables = {}
    if status:
        variables["status"] = status

    result = client.query("""
        query PostsByStatus($status: PostStatus) {
            posts(status: $status) { id title }
        }
    """, variables=variables)

    assert result.errors is None
    assert len(result.data["posts"]) == expected_count
```

---

## 6. Schema Validation and Linting

### GraphQL Inspector

```bash
# Install
npm install -g @graphql-inspector/cli

# Compare schemas for breaking changes
graphql-inspector diff old-schema.graphql new-schema.graphql

# Output:
# ✖ Field 'User.email' was removed (BREAKING)
# ✔ Field 'User.phone' was added (NON_BREAKING)
# ⚠ Field 'User.name' was deprecated (DANGEROUS)

# Validate schema
graphql-inspector validate queries/**/*.graphql schema.graphql
```

### Schema Linting Rules

```yaml
# .graphql-inspector.yaml
rules:
  - name: require-description
    severity: warning
    config:
      types: true
      fields: true

  - name: naming-convention
    severity: error
    config:
      types: PascalCase
      fields: camelCase
      enumValues: UPPER_CASE
      inputFields: camelCase

  - name: require-deprecation-reason
    severity: error

  - name: no-unreachable-types
    severity: warning
```

### Rover Schema Check

```bash
# Check for breaking changes against production
rover subgraph check my-graph@production \
  --name users \
  --schema subgraphs/users/schema.graphql

# Example output:
# Compared 2 schema changes against 47 operations
# ── FAILURE ──────────────────────────
# BREAKING: removing field User.legacyId
#   affected: GetUserProfile, SearchUsers (12 clients)
```

---

## 7. Code Generation

### GraphQL Code Generator

Generate typed client code from your schema:

```bash
npm install -g @graphql-codegen/cli
```

```yaml
# codegen.yml
schema: http://localhost:8000/graphql
documents: src/**/*.graphql
generates:
  src/generated/types.ts:
    plugins:
      - typescript
      - typescript-operations
      - typescript-react-apollo

  src/generated/schema.json:
    plugins:
      - introspection
```

### Generated TypeScript Types

```typescript
// src/generated/types.ts (auto-generated)
export type User = {
  __typename?: 'User';
  id: string;
  username: string;
  email: string;
  bio?: string | null;
  posts: PostConnection;
};

export type GetUserQuery = {
  __typename?: 'Query';
  user?: {
    __typename?: 'User';
    id: string;
    username: string;
    email: string;
  } | null;
};

export type GetUserQueryVariables = {
  id: string;
};

// React hook (if using typescript-react-apollo plugin)
export function useGetUserQuery(options: QueryHookOptions<GetUserQuery, GetUserQueryVariables>) {
  return useQuery<GetUserQuery, GetUserQueryVariables>(GetUserDocument, options);
}
```

### Python Code Generation

```bash
# Generate Python types with ariadne-codegen
pip install ariadne-codegen

# ariadne-codegen.toml
[tool.ariadne-codegen]
schema_path = "schema.graphql"
queries_path = "queries/"
target_package_name = "graphql_client"
```

### Benefits of Code Generation

| Benefit | Description |
|---------|-------------|
| Type safety | Catch errors at compile time |
| Auto-complete | IDE knows the schema |
| Synchronization | Generated code always matches schema |
| Reduced boilerplate | No manual type definitions |

---

## 8. Interactive Tools

### GraphiQL

The built-in GraphQL IDE:

```python
# Strawberry: enable GraphiQL
graphql_router = GraphQLRouter(
    schema,
    graphql_ide="graphiql",  # Default
)
```

Features:
- Auto-complete for queries
- Schema documentation browser
- Query history
- Variable editor
- Response viewer

### Apollo Sandbox

```python
# Use Apollo Sandbox instead of GraphiQL
graphql_router = GraphQLRouter(
    schema,
    graphql_ide="apollo-sandbox",
)
```

Features:
- All GraphiQL features plus:
- Operation collections
- Environment variables
- Pre-flight scripts
- Response diffing

### Postman / Insomnia

Both support GraphQL:

```
1. Create new GraphQL request
2. Set URL to http://localhost:8000/graphql
3. Write query in the body editor
4. Auto-complete works via schema introspection
5. Set variables in the Variables tab
6. Add Authorization header
```

### GraphQL Voyager

Interactive schema visualization:

```python
# Add Voyager route
from starlette.responses import HTMLResponse

VOYAGER_HTML = """
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/graphql-voyager/dist/voyager.css" />
</head>
<body>
  <div id="voyager">Loading...</div>
  <script src="https://cdn.jsdelivr.net/npm/graphql-voyager/dist/voyager.standalone.js"></script>
  <script>
    GraphQLVoyager.init(document.getElementById('voyager'), {
      introspection: fetch('/graphql', {
        method: 'post',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: GraphQLVoyager.voyagerIntrospectionQuery,
        }),
      }).then(r => r.json()),
    });
  </script>
</body>
</html>
"""

@app.get("/voyager")
async def voyager():
    return HTMLResponse(VOYAGER_HTML)
```

---

## 9. CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/graphql-ci.yml
name: GraphQL CI

on:
  pull_request:
    paths:
      - 'src/**'
      - 'schema.graphql'

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_DB: test
          POSTGRES_PASSWORD: test
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt

      - name: Run unit tests
        run: pytest tests/unit/ -v --cov=src

      - name: Run integration tests
        run: pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql+asyncpg://postgres:test@localhost/test

  schema-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install Rover
        run: |
          curl -sSL https://rover.apollo.dev/nix/latest | sh
          echo "$HOME/.rover/bin" >> $GITHUB_PATH

      - name: Export schema
        run: |
          pip install -r requirements.txt
          python -c "
          from schema import schema
          import strawberry
          print(strawberry.printer.print_schema(schema))
          " > current-schema.graphql

      - name: Check for breaking changes
        run: |
          rover graph check my-graph@production \
            --schema current-schema.graphql || true

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Schema lint
        run: |
          npx @graphql-inspector/cli validate \
            'src/**/*.graphql' schema.graphql

      - name: Python lint
        run: |
          pip install ruff
          ruff check src/ tests/
```

### Schema Change Workflow

```
1. Developer modifies schema
2. CI runs:
   a. Unit tests → Pass/Fail
   b. Integration tests → Pass/Fail
   c. Schema diff → Breaking change detection
   d. Snapshot tests → Response shape validation
   e. Lint → Style checks
3. If breaking changes detected:
   a. Block merge
   b. Require manual approval
   c. Require deprecation period
4. On merge to main:
   a. Publish schema to registry
   b. Deploy to staging
   c. Run E2E tests
5. On release:
   a. Deploy to production
   b. Notify API consumers
```

---

## 10. Exercises

### Exercise 1: Test Suite

Write a comprehensive test suite for a blog API with:
- 5 query tests (user, post, list, search, error cases)
- 3 mutation tests (create, update, delete with validation)
- 2 authorization tests (authenticated vs. unauthorized)
- 1 DataLoader integration test

### Exercise 2: Snapshot Testing

Set up snapshot testing for:
- The full schema SDL output
- Three different query responses
- Verify that adding a new nullable field does not break snapshots
- Verify that removing a field does break snapshots

### Exercise 3: Code Generation

Set up GraphQL Code Generator for a TypeScript React client:
- Configure `codegen.yml` to generate types and hooks
- Write 3 `.graphql` operation files
- Generate the code and verify type safety

### Exercise 4: CI Pipeline

Create a complete GitHub Actions CI pipeline that:
- Runs unit and integration tests
- Exports the current schema
- Compares against the production schema for breaking changes
- Lints the schema for naming conventions
- Posts a summary comment on the PR

---

## 11. References

### Testing
- [Strawberry Testing Documentation](https://strawberry.rocks/docs/general/testing)
- [pytest Documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://github.com/pytest-dev/pytest-asyncio)

### Tooling
- [GraphiQL](https://github.com/graphql/graphiql)
- [Apollo Sandbox](https://studio.apollographql.com/sandbox)
- [GraphQL Voyager](https://graphql-kit.com/graphql-voyager/)
- [GraphQL Inspector](https://graphql-inspector.com/)

### Code Generation
- [GraphQL Code Generator](https://the-guild.dev/graphql/codegen)
- [Ariadne Codegen](https://github.com/mirumee/ariadne-codegen)

### CI/CD
- [Rover CI/CD Guide](https://www.apollographql.com/docs/rover/ci-cd/)
- [GitHub Actions for GraphQL](https://graphql-inspector.com/docs/recipes/github)

---

**License**: CC BY-NC 4.0
