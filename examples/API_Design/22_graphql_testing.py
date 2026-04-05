#!/usr/bin/env python3
"""Example: GraphQL Testing

Demonstrates testing patterns for GraphQL APIs:
- Schema introspection tests
- Query execution tests
- Mutation tests with state verification
- Error handling tests
- Snapshot testing for schema changes
- Testing with authenticated context

Related lesson: 22_GraphQL_Testing.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]" pytest
    pytest 22_graphql_testing.py -v
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import pytest
import strawberry
from strawberry.types import Info

# =============================================================================
# SCHEMA UNDER TEST
# =============================================================================

_todos: dict[str, dict] = {}


@strawberry.type
class Todo:
    id: str
    title: str
    done: bool
    owner_id: str


@strawberry.input
class TodoInput:
    title: str


@strawberry.type
class Query:
    @strawberry.field
    def todos(self, owner_id: Optional[str] = None) -> list[Todo]:
        items = list(_todos.values())
        if owner_id:
            items = [t for t in items if t["owner_id"] == owner_id]
        return [Todo(**t) for t in items]

    @strawberry.field
    def todo(self, id: str) -> Optional[Todo]:
        data = _todos.get(id)
        return Todo(**data) if data else None


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_todo(self, input: TodoInput, info: Info) -> Todo:
        user = info.context.get("user_id", "anonymous")
        todo_id = str(uuid4())[:8]
        data = {"id": todo_id, "title": input.title, "done": False, "owner_id": user}
        _todos[todo_id] = data
        return Todo(**data)

    @strawberry.mutation
    def toggle_todo(self, id: str) -> Optional[Todo]:
        data = _todos.get(id)
        if not data:
            return None
        data["done"] = not data["done"]
        return Todo(**data)

    @strawberry.mutation
    def delete_todo(self, id: str) -> bool:
        if id in _todos:
            del _todos[id]
            return True
        return False


schema = strawberry.Schema(query=Query, mutation=Mutation)

# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture(autouse=True)
def reset_db():
    """Isolate each test with a clean database."""
    _todos.clear()
    yield
    _todos.clear()


@pytest.fixture
def context():
    """Default execution context with a test user."""
    return {"user_id": "test-user-1"}


@pytest.fixture
def sample_todo(context):
    """Create a sample todo and return its data."""
    result = schema.execute_sync(
        'mutation { createTodo(input: {title: "Buy milk"}) { id title done ownerId } }',
        context_value=context,
    )
    assert result.errors is None
    return result.data["createTodo"]


# =============================================================================
# SCHEMA INTROSPECTION TESTS
# =============================================================================
# Verify the schema shape has not changed unexpectedly.

class TestSchemaIntrospection:
    def test_query_type_exists(self):
        result = schema.execute_sync("{ __schema { queryType { name } } }")
        assert result.errors is None
        assert result.data["__schema"]["queryType"]["name"] == "Query"

    def test_mutation_type_exists(self):
        result = schema.execute_sync("{ __schema { mutationType { name } } }")
        assert result.errors is None
        assert result.data["__schema"]["mutationType"]["name"] == "Mutation"

    def test_todo_type_fields(self):
        """Ensure the Todo type has all expected fields (contract test)."""
        result = schema.execute_sync("""
            { __type(name: "Todo") {
                fields { name type { name kind } }
            }}
        """)
        assert result.errors is None
        field_names = {f["name"] for f in result.data["__type"]["fields"]}
        assert {"id", "title", "done", "ownerId"}.issubset(field_names)


# =============================================================================
# QUERY TESTS
# =============================================================================

class TestQueries:
    def test_empty_list(self):
        result = schema.execute_sync("{ todos { id } }")
        assert result.errors is None
        assert result.data["todos"] == []

    def test_list_with_data(self, sample_todo):
        result = schema.execute_sync("{ todos { id title } }")
        assert result.errors is None
        assert len(result.data["todos"]) == 1
        assert result.data["todos"][0]["title"] == "Buy milk"

    def test_get_by_id(self, sample_todo):
        todo_id = sample_todo["id"]
        result = schema.execute_sync(
            "query($id: String!) { todo(id: $id) { id title done } }",
            variable_values={"id": todo_id},
        )
        assert result.errors is None
        assert result.data["todo"]["title"] == "Buy milk"

    def test_not_found_returns_null(self):
        result = schema.execute_sync('{ todo(id: "nonexistent") { id } }')
        assert result.errors is None
        assert result.data["todo"] is None

    def test_filter_by_owner(self, context, sample_todo):
        result = schema.execute_sync(
            '{ todos(ownerId: "test-user-1") { id } }',
        )
        assert len(result.data["todos"]) == 1

        result2 = schema.execute_sync('{ todos(ownerId: "other") { id } }')
        assert len(result2.data["todos"]) == 0


# =============================================================================
# MUTATION TESTS
# =============================================================================

class TestMutations:
    def test_create(self, context):
        result = schema.execute_sync(
            'mutation { createTodo(input: {title: "Test"}) { id title done ownerId } }',
            context_value=context,
        )
        assert result.errors is None
        todo = result.data["createTodo"]
        assert todo["title"] == "Test"
        assert todo["done"] is False
        assert todo["ownerId"] == "test-user-1"

    def test_toggle(self, context, sample_todo):
        todo_id = sample_todo["id"]
        # Toggle once -> done=True
        r1 = schema.execute_sync(
            f'mutation {{ toggleTodo(id: "{todo_id}") {{ done }} }}',
        )
        assert r1.data["toggleTodo"]["done"] is True

        # Toggle again -> done=False
        r2 = schema.execute_sync(
            f'mutation {{ toggleTodo(id: "{todo_id}") {{ done }} }}',
        )
        assert r2.data["toggleTodo"]["done"] is False

    def test_delete(self, context, sample_todo):
        todo_id = sample_todo["id"]
        r = schema.execute_sync(f'mutation {{ deleteTodo(id: "{todo_id}") }}')
        assert r.data["deleteTodo"] is True

        # Verify gone
        r2 = schema.execute_sync(f'{{ todo(id: "{todo_id}") {{ id }} }}')
        assert r2.data["todo"] is None

    def test_delete_nonexistent(self):
        r = schema.execute_sync('mutation { deleteTodo(id: "nope") }')
        assert r.data["deleteTodo"] is False


# =============================================================================
# SNAPSHOT TEST — Detect unintended schema changes
# =============================================================================

class TestSchemaSnapshot:
    def test_sdl_has_expected_types(self):
        """Lightweight snapshot: verify key types appear in the SDL."""
        sdl = schema.as_str()
        assert "type Todo" in sdl
        assert "type Query" in sdl
        assert "type Mutation" in sdl
        assert "input TodoInput" in sdl


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
