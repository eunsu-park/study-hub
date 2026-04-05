#!/bin/bash
# Exercises for Lesson 24: GraphQL Testing and Tooling
# Topic: API_Design
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Comprehensive Test Suite ==="
    cat << 'SOLUTION'
import pytest
from strawberry.test import GraphQLTestClient
from unittest.mock import AsyncMock, MagicMock

# --- Query Tests ---

def test_get_user(client, mock_ctx):
    mock_ctx.db.users.find_by_id = AsyncMock(return_value=make_user(username="alice"))
    result = client.query('query { user(id: "1") { username } }')
    assert result.errors is None
    assert result.data["user"]["username"] == "alice"

def test_get_post(client, mock_ctx):
    mock_ctx.db.posts.find_by_id = AsyncMock(return_value=make_post(title="Hello"))
    result = client.query('query { post(id: "101") { title } }')
    assert result.data["post"]["title"] == "Hello"

def test_list_posts(client, mock_ctx):
    mock_ctx.db.posts.find_all = AsyncMock(return_value=[make_post(), make_post()])
    result = client.query("query { posts(first: 10) { id title } }")
    assert len(result.data["posts"]) == 2

def test_search_posts(client, mock_ctx):
    mock_ctx.db.posts.search = AsyncMock(return_value=[make_post(title="GraphQL")])
    result = client.query('query { searchPosts(query: "graph") { title } }')
    assert result.data["searchPosts"][0]["title"] == "GraphQL"

def test_get_nonexistent_user(client, mock_ctx):
    mock_ctx.db.users.find_by_id = AsyncMock(return_value=None)
    result = client.query('query { user(id: "999") { username } }')
    assert result.data["user"] is None

# --- Mutation Tests ---

def test_create_post(client, mock_ctx):
    mock_ctx.db.posts.create = AsyncMock(return_value=make_post(title="New"))
    result = client.query("""
        mutation { createPost(input: {title: "New", content: "Body"}) {
            post { title } userErrors { message }
        }}
    """)
    assert result.data["createPost"]["post"]["title"] == "New"
    assert len(result.data["createPost"]["userErrors"]) == 0

def test_update_post(client, mock_ctx):
    mock_ctx.db.posts.update = AsyncMock(return_value=make_post(title="Updated"))
    result = client.query("""
        mutation { updatePost(id: "101", input: {title: "Updated"}) {
            post { title } userErrors { message }
        }}
    """)
    assert result.data["updatePost"]["post"]["title"] == "Updated"

def test_delete_post_not_found(client, mock_ctx):
    mock_ctx.db.posts.find_by_id = AsyncMock(return_value=None)
    result = client.query('mutation { deletePost(id: "999") { userErrors { code } } }')
    assert any(e["code"] == "NOT_FOUND" for e in result.data["deletePost"]["userErrors"])

# --- Authorization Tests ---

def test_create_post_unauthenticated(client_no_auth):
    result = client_no_auth.query("""
        mutation { createPost(input: {title: "T", content: "C"}) {
            post { id } userErrors { message }
        }}
    """)
    assert result.errors is not None or len(result.data["createPost"]["userErrors"]) > 0

def test_admin_query_authorized(client_admin, mock_ctx):
    mock_ctx.db.dashboard.get = AsyncMock(return_value={"total": 100})
    result = client_admin.query("query { adminDashboard { totalUsers } }")
    assert result.errors is None

# --- DataLoader Integration Test ---

@pytest.mark.asyncio
async def test_dataloader_no_n_plus_1(seeded_db):
    context = RequestContext(db=seeded_db, current_user=None)
    client = GraphQLTestClient(schema, context_value=context)
    result = client.query("""
        query { posts(first: 5) { title author { username } } }
    """)
    assert result.errors is None
    for post in result.data["posts"]:
        assert post["author"]["username"] is not None
SOLUTION
}

exercise_2() {
    echo "=== Exercise 2: Snapshot Testing ==="
    cat << 'SOLUTION'
import strawberry
import json

def test_schema_snapshot(snapshot):
    schema_str = strawberry.printer.print_schema(schema)
    snapshot.assert_match(schema_str, "schema.graphql")

def test_user_response_snapshot(client, snapshot, mock_ctx):
    mock_ctx.db.users.find_by_id = AsyncMock(return_value=make_user())
    result = client.query('query { user(id: "1") { id username email } }')
    snapshot.assert_match(json.dumps(result.data, indent=2), "user_response.json")

def test_posts_response_snapshot(client, snapshot, mock_ctx):
    mock_ctx.db.posts.find_all = AsyncMock(return_value=[make_post()])
    result = client.query("query { posts(first: 1) { id title status } }")
    snapshot.assert_match(json.dumps(result.data, indent=2), "posts_response.json")

def test_error_response_snapshot(client, snapshot, mock_ctx):
    result = client.query("query { unknownField }")
    snapshot.assert_match(json.dumps({"errors": [str(e) for e in result.errors]}), "error_response.json")

# Adding a new nullable field (e.g., "bio: String") should NOT break snapshots
# because the new field won't appear in queries that don't select it.

# Removing a field (e.g., removing "email") WILL break snapshots
# because existing queries that select it will get errors.
SOLUTION
}

main() { exercise_1; echo ""; exercise_2; }
main "$@"
