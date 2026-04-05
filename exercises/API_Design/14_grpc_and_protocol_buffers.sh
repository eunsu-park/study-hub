#!/bin/bash
# Exercises for Lesson 14: GraphQL Fundamentals
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: GraphQL Schema Definition ===
# Problem: Design a GraphQL schema for a blog platform with users, posts,
# and comments. Include queries, mutations, and a subscription.
exercise_1() {
    echo "=== Exercise 1: GraphQL Schema Definition ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# GraphQL Schema Definition Language (SDL)
SCHEMA = """
# Scalars
scalar DateTime

# Enums
enum PostStatus {
  DRAFT
  PUBLISHED
  ARCHIVED
}

enum SortOrder {
  ASC
  DESC
}

# Input types (for mutations)
input CreatePostInput {
  title: String!
  content: String!
  tags: [String!]
}

input UpdatePostInput {
  title: String
  content: String
  status: PostStatus
  tags: [String!]
}

input PostFilterInput {
  status: PostStatus
  authorId: ID
  tag: String
}

# Object types
type User {
  id: ID!
  username: String!
  email: String!
  bio: String
  posts(status: PostStatus): [Post!]!
  postCount: Int!
  createdAt: DateTime!
}

type Post {
  id: ID!
  title: String!
  content: String!
  status: PostStatus!
  author: User!
  comments: [Comment!]!
  commentCount: Int!
  tags: [String!]!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Comment {
  id: ID!
  body: String!
  author: User!
  post: Post!
  createdAt: DateTime!
}

# Pagination (Relay-style connections)
type PostConnection {
  edges: [PostEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type PostEdge {
  node: Post!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}

# Queries
type Query {
  # Single resource lookups
  user(id: ID!): User
  post(id: ID!): Post

  # Collections with filtering and pagination
  posts(
    filter: PostFilterInput
    first: Int = 10
    after: String
    orderBy: SortOrder = DESC
  ): PostConnection!

  # Search
  search(query: String!, limit: Int = 10): [Post!]!
}

# Mutations
type Mutation {
  createPost(input: CreatePostInput!): Post!
  updatePost(id: ID!, input: UpdatePostInput!): Post!
  deletePost(id: ID!): Boolean!
  addComment(postId: ID!, body: String!): Comment!
}

# Subscriptions (real-time)
type Subscription {
  postPublished: Post!
  commentAdded(postId: ID!): Comment!
}
"""
print(SCHEMA)
SOLUTION
}

# === Exercise 2: Strawberry GraphQL in Python ===
# Problem: Implement the blog schema using Strawberry (Python GraphQL library)
# with resolvers and DataLoader for N+1 prevention.
exercise_2() {
    echo "=== Exercise 2: Strawberry GraphQL in Python ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import strawberry
from strawberry.fastapi import GraphQLRouter
from strawberry.dataloader import DataLoader
from fastapi import FastAPI
from typing import Optional
from datetime import datetime, timezone

# --- Data Store ---
users_db = {
    "1": {"id": "1", "username": "alice", "email": "alice@example.com", "bio": "Writer"},
    "2": {"id": "2", "username": "bob", "email": "bob@example.com", "bio": "Developer"},
}

posts_db = [
    {"id": "1", "title": "GraphQL Basics", "content": "Learn GraphQL...",
     "status": "PUBLISHED", "author_id": "1", "tags": ["graphql", "api"],
     "created_at": "2025-06-01T10:00:00Z"},
    {"id": "2", "title": "REST vs GraphQL", "content": "Comparing...",
     "status": "PUBLISHED", "author_id": "1", "tags": ["graphql", "rest"],
     "created_at": "2025-06-02T10:00:00Z"},
    {"id": "3", "title": "Python Tips", "content": "Useful tips...",
     "status": "DRAFT", "author_id": "2", "tags": ["python"],
     "created_at": "2025-06-03T10:00:00Z"},
]


# --- DataLoader (prevents N+1 queries) ---
async def load_users(ids: list[str]) -> list[dict]:
    """Batch load users by ID. Called once per GraphQL request
    regardless of how many User fields are resolved."""
    return [users_db.get(uid, {}) for uid in ids]


# --- Strawberry Types ---
@strawberry.type
class User:
    id: strawberry.ID
    username: str
    email: str
    bio: Optional[str] = None

    @strawberry.field
    async def posts(self, info: strawberry.types.Info) -> list["Post"]:
        user_posts = [p for p in posts_db if p["author_id"] == self.id]
        return [Post(
            id=p["id"], title=p["title"], content=p["content"],
            status=p["status"], author_id=p["author_id"], tags=p["tags"],
        ) for p in user_posts]

    @strawberry.field
    def post_count(self) -> int:
        return len([p for p in posts_db if p["author_id"] == self.id])


@strawberry.type
class Post:
    id: strawberry.ID
    title: str
    content: str
    status: str
    tags: list[str]
    author_id: strawberry.Private[str]  # Not exposed in schema

    @strawberry.field
    async def author(self, info: strawberry.types.Info) -> User:
        """Resolve author via DataLoader — batched, no N+1."""
        user_data = await info.context["user_loader"].load(self.author_id)
        return User(**user_data)


# --- Query ---
@strawberry.type
class Query:
    @strawberry.field
    def post(self, id: strawberry.ID) -> Optional[Post]:
        for p in posts_db:
            if p["id"] == id:
                return Post(
                    id=p["id"], title=p["title"], content=p["content"],
                    status=p["status"], author_id=p["author_id"], tags=p["tags"],
                )
        return None

    @strawberry.field
    def posts(self, status: Optional[str] = None, first: int = 10) -> list[Post]:
        result = posts_db
        if status:
            result = [p for p in result if p["status"] == status]
        return [Post(
            id=p["id"], title=p["title"], content=p["content"],
            status=p["status"], author_id=p["author_id"], tags=p["tags"],
        ) for p in result[:first]]


# --- Setup ---
schema = strawberry.Schema(query=Query)

app = FastAPI()

async def get_context():
    return {"user_loader": DataLoader(load_fn=load_users)}

graphql_app = GraphQLRouter(schema, context_getter=get_context)
app.include_router(graphql_app, prefix="/graphql")

# Query example:
# {
#   posts(status: "PUBLISHED") {
#     title
#     author {
#       username
#     }
#     tags
#   }
# }
SOLUTION
}

# === Exercise 3: GraphQL Query Optimization ===
# Problem: Identify and fix N+1 query problems in a GraphQL resolver,
# and implement query depth limiting.
exercise_3() {
    echo "=== Exercise 3: GraphQL Query Optimization ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Problem: N+1 Query
# When resolving a list of posts, each post.author triggers a separate
# database query. 10 posts = 1 list query + 10 author queries = 11 queries!

# BAD: N+1 queries
# def resolve_author(post):
#     return db.query(User).get(post.author_id)  # Called N times!

# GOOD: DataLoader batches all author lookups into one query
# DataLoader collects all requested IDs during the GraphQL resolve phase,
# then makes a single batch query.

from strawberry.dataloader import DataLoader


async def batch_load_users(user_ids: list[str]) -> list[dict]:
    """One query instead of N queries.
    SQL: SELECT * FROM users WHERE id IN (?, ?, ?, ...)
    """
    # In production, this is a single DB query
    users = {uid: {"id": uid, "name": f"User {uid}"} for uid in user_ids}
    return [users.get(uid, {}) for uid in user_ids]


# Query Depth Limiting — prevent deeply nested queries
# BAD query (could cause exponential resolver calls):
# {
#   user(id: 1) {
#     posts {
#       author {
#         posts {
#           author {
#             posts { ... }  # Infinite nesting!
#           }
#         }
#       }
#     }
#   }
# }

import strawberry
from strawberry.extensions import QueryDepthLimiter

schema = strawberry.Schema(
    query=Query,
    extensions=[
        QueryDepthLimiter(max_depth=5),  # Reject queries deeper than 5 levels
    ],
)

# Query Complexity Analysis
# Assign a cost to each field and reject queries above a threshold

from strawberry.extensions import MaxTokensLimiter

schema = strawberry.Schema(
    query=Query,
    extensions=[
        QueryDepthLimiter(max_depth=5),
        MaxTokensLimiter(max_token_count=1000),
    ],
)

# Best practices for GraphQL performance:
# 1. Use DataLoader for all relationship resolvers
# 2. Set max query depth (5-10 levels)
# 3. Set max query complexity / cost
# 4. Implement timeout per query
# 5. Use persisted queries in production (whitelist allowed queries)
SOLUTION
}

# === Exercise 4: REST to GraphQL Migration ===
# Problem: Wrap existing REST endpoints behind a GraphQL layer,
# demonstrating the BFF (Backend for Frontend) pattern.
exercise_4() {
    echo "=== Exercise 4: REST to GraphQL Migration ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import strawberry
import httpx
from strawberry.fastapi import GraphQLRouter
from fastapi import FastAPI
from typing import Optional

# Existing REST APIs (we do not modify these)
USER_SERVICE = "http://user-service:8001"
ORDER_SERVICE = "http://order-service:8002"
PRODUCT_SERVICE = "http://product-service:8003"


# GraphQL types that map to REST resources
@strawberry.type
class Product:
    id: strawberry.ID
    name: str
    price: float


@strawberry.type
class OrderItem:
    product: Product
    quantity: int
    subtotal: float


@strawberry.type
class Order:
    id: strawberry.ID
    status: str
    total: float
    items: list[OrderItem]


@strawberry.type
class User:
    id: strawberry.ID
    name: str
    email: str

    @strawberry.field
    async def orders(self) -> list[Order]:
        """Fetch orders from REST Order Service."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{ORDER_SERVICE}/api/v1/users/{self.id}/orders")
            data = resp.json()["data"]
            return [Order(
                id=o["id"], status=o["status"], total=o["total"],
                items=[OrderItem(
                    product=Product(id=i["product_id"], name=i["name"], price=i["price"]),
                    quantity=i["quantity"],
                    subtotal=i["subtotal"],
                ) for i in o.get("items", [])],
            ) for o in data]


@strawberry.type
class Query:
    @strawberry.field
    async def user(self, id: strawberry.ID) -> Optional[User]:
        """Fetch user from REST User Service."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{USER_SERVICE}/api/v1/users/{id}")
            if resp.status_code == 404:
                return None
            u = resp.json()
            return User(id=u["id"], name=u["name"], email=u["email"])

    @strawberry.field
    async def product(self, id: strawberry.ID) -> Optional[Product]:
        """Fetch product from REST Product Service."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{PRODUCT_SERVICE}/api/v1/products/{id}")
            if resp.status_code == 404:
                return None
            p = resp.json()
            return Product(id=p["id"], name=p["name"], price=p["price"])


# BFF (Backend for Frontend) pattern:
# The GraphQL layer acts as an aggregation gateway.
# Mobile app can fetch user + orders + products in ONE request:
#
# query {
#   user(id: "123") {
#     name
#     orders {
#       status
#       total
#       items {
#         product { name price }
#         quantity
#       }
#     }
#   }
# }
#
# Without GraphQL, the mobile app would need 3+ REST calls:
# GET /users/123
# GET /users/123/orders
# GET /products/456 (for each order item)

schema = strawberry.Schema(query=Query)
app = FastAPI()
app.include_router(GraphQLRouter(schema), prefix="/graphql")
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 14: GraphQL Fundamentals"
echo "========================================================"
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
