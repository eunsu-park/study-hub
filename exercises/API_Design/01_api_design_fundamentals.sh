#!/bin/bash
# Exercises for Lesson 01: API Design Fundamentals
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: Choosing an API Paradigm ===
# Problem: Given different scenarios, select the most appropriate API paradigm
# (REST, RPC, GraphQL, or Event-Driven) and justify your choice.
exercise_1() {
    echo "=== Exercise 1: Choosing an API Paradigm ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# API Paradigm Selection Guide

# Scenario 1: Public e-commerce product catalog
# Choice: REST
# Why: Resource-oriented (products, categories, orders), cacheable,
#       wide client support (browsers, mobile), self-descriptive.

# Scenario 2: Internal microservice calling a calculation engine
# Choice: RPC (gRPC)
# Why: Action-oriented (calculateShipping, validateAddress),
#       high performance with binary serialization, strong typing,
#       internal-only so no browser constraint.

# Scenario 3: Mobile app dashboard showing data from 5 different services
# Choice: GraphQL
# Why: Client specifies exactly which fields it needs (no over-fetching),
#       single request aggregates multiple data sources,
#       mobile bandwidth is limited so payload efficiency matters.

# Scenario 4: Payment processor notifying merchants of charge events
# Choice: Event-Driven (Webhooks)
# Why: The payment system pushes events as they happen,
#       merchants don't need to poll, real-time notifications,
#       loose coupling between payment system and merchant backend.

# Decision Matrix:
# | Factor             | REST | RPC/gRPC | GraphQL | Events |
# |--------------------|------|----------|---------|--------|
# | Public API         | +++  | -        | ++      | +      |
# | Internal service   | ++   | +++      | +       | ++     |
# | Mobile client      | ++   | +        | +++     | -      |
# | Real-time updates  | +    | +        | ++      | +++    |
# | Cacheability       | +++  | -        | +       | -      |
# | Browser support    | +++  | -        | ++      | +      |
SOLUTION
}

# === Exercise 2: Contract-First API Design ===
# Problem: Design a contract-first API for a library management system.
# Define the resource schema before writing any code.
exercise_2() {
    echo "=== Exercise 2: Contract-First API Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Contract-first means defining the API specification BEFORE implementation.
# This OpenAPI snippet defines the contract for a Library API.

# openapi: "3.1.0"
# info:
#   title: Library Management API
#   version: "1.0.0"
# paths:
#   /api/v1/books:
#     get:
#       summary: List all books
#       parameters:
#         - name: genre
#           in: query
#           schema: { type: string }
#     post:
#       summary: Add a new book
#       requestBody:
#         content:
#           application/json:
#             schema: { $ref: '#/components/schemas/BookCreate' }

# Python implementation matching the contract:
from pydantic import BaseModel, Field
from typing import Optional


class BookCreate(BaseModel):
    """Input schema — matches the OpenAPI contract exactly."""
    title: str = Field(..., min_length=1, max_length=300)
    author: str = Field(..., min_length=1, max_length=200)
    isbn: str = Field(..., pattern=r"^\d{13}$")
    genre: Optional[str] = None
    published_year: Optional[int] = Field(None, ge=1000, le=2100)


class BookResponse(BaseModel):
    """Output schema — server-generated fields added."""
    id: str
    title: str
    author: str
    isbn: str
    genre: Optional[str] = None
    published_year: Optional[int] = None
    available: bool = True
    created_at: str


# Benefits of contract-first:
# 1. Frontend and backend teams can work in parallel
# 2. API consumers know the interface before implementation exists
# 3. SDK generation is possible from day one
# 4. Prevents implementation details from leaking into the API
SOLUTION
}

# === Exercise 3: Design Principles Evaluation ===
# Problem: Evaluate a poorly designed API and improve it using
# Postel's Law, Principle of Least Surprise, and Consistency.
exercise_3() {
    echo "=== Exercise 3: Design Principles Evaluation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from typing import Optional

app = FastAPI()

# --- BAD API DESIGN (before) ---
# Problems:
# 1. Inconsistent naming: /getUsers vs /create_user vs /User-delete
# 2. Violates Least Surprise: POST /getUsers (should be GET)
# 3. Violates Postel's Law: rejects "john" but accepts "John" for role
# 4. Inconsistent response format: sometimes {data:...}, sometimes raw

# --- GOOD API DESIGN (after) ---

# Postel's Law: "Be liberal in what you accept, conservative in what you send"
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str
    role: str = "viewer"

    @field_validator("role", mode="before")
    @classmethod
    def normalize_role(cls, v: str) -> str:
        """Accept any case, normalize to lowercase (Postel's Law)."""
        return v.lower().strip()

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Accept whitespace-padded emails, normalize (Postel's Law)."""
        return v.lower().strip()


# Consistent response envelope (Principle of Consistency)
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    role: str  # Always lowercase in output (conservative output)


# Consistent, predictable URL patterns (Least Surprise)
# All plural nouns, all use standard HTTP methods
@app.get("/api/v1/users")          # List users (not /getUsers)
def list_users():
    return {"data": [], "total": 0}

@app.post("/api/v1/users", status_code=201)  # Create user (not /create_user)
def create_user(body: UserCreate):
    return {"data": UserResponse(id=1, **body.model_dump())}

@app.get("/api/v1/users/{user_id}")  # Get one user
def get_user(user_id: int):
    return {"data": None}

@app.delete("/api/v1/users/{user_id}", status_code=204)  # Delete (not /User-delete)
def delete_user(user_id: int):
    return None
SOLUTION
}

# === Exercise 4: API Comparison Implementation ===
# Problem: Implement the same functionality (user profile lookup) using
# three different paradigms to compare their approaches.
exercise_4() {
    echo "=== Exercise 4: API Comparison Implementation ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# Same feature (user profile) implemented in REST, RPC, and GraphQL styles

# --- REST Style ---
# Resource-oriented: the URL identifies the resource
from fastapi import FastAPI

rest_app = FastAPI()

@rest_app.get("/api/v1/users/{user_id}")
def get_user_rest(user_id: str):
    """REST: Resource is a noun, HTTP method is the verb."""
    return {
        "data": {
            "id": user_id,
            "name": "Alice",
            "email": "alice@example.com",
            "posts_count": 42,
        },
        "links": {
            "self": f"/api/v1/users/{user_id}",
            "posts": f"/api/v1/users/{user_id}/posts",
        },
    }


# --- RPC Style ---
# Action-oriented: the URL describes the operation
@rest_app.post("/api/v1/getUserProfile")
def get_user_rpc(user_id: str):
    """RPC: Function call over HTTP. URL is a verb."""
    return {
        "result": {
            "id": user_id,
            "name": "Alice",
            "email": "alice@example.com",
            "posts_count": 42,
        },
    }


# --- GraphQL Style ---
# Query language: client specifies exactly what it needs
GRAPHQL_QUERY = """
query {
  user(id: "123") {
    name
    email
    # Client only asks for name and email — no posts_count
    # This reduces bandwidth (no over-fetching)
  }
}
"""

GRAPHQL_RESPONSE = {
    "data": {
        "user": {
            "name": "Alice",
            "email": "alice@example.com",
            # posts_count is NOT included because client did not request it
        }
    }
}

# Key differences:
# REST:    Server decides what fields to return
# RPC:     Server decides what fields to return
# GraphQL: Client decides what fields to return
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 01: API Design Fundamentals"
echo "=========================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
