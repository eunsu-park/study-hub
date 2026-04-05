#!/bin/bash
# Exercises for Lesson 17: GraphQL Fundamentals
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: Schema Design ===
# Problem: Design a GraphQL schema (SDL) for an e-commerce platform.
exercise_1() {
    echo "=== Exercise 1: E-Commerce Schema Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# E-Commerce GraphQL Schema

scalar DateTime

enum ProductCategory {
  ELECTRONICS
  CLOTHING
  BOOKS
  HOME
  SPORTS
}

type Product {
  id: ID!
  name: String!
  price: Float!
  category: ProductCategory!
  inStock: Boolean!
  description: String
  createdAt: DateTime!
}

type Category {
  id: ID!
  name: String!
  products(first: Int = 10): [Product!]!
}

type CartItem {
  product: Product!
  quantity: Int!
}

type Cart {
  id: ID!
  items: [CartItem!]!
  total: Float!
  itemCount: Int!
}

type Query {
  product(id: ID!): Product
  products(
    category: ProductCategory
    inStock: Boolean
    minPrice: Float
    maxPrice: Float
    first: Int = 10
    after: String
  ): [Product!]!
  cart(id: ID!): Cart
}

type Mutation {
  addToCart(cartId: ID!, productId: ID!, quantity: Int! = 1): Cart!
  removeFromCart(cartId: ID!, productId: ID!): Cart!
  checkout(cartId: ID!): Order!
}
SOLUTION
}

# === Exercise 2: Query Writing ===
# Problem: Write queries for movie scenarios.
exercise_2() {
    echo "=== Exercise 2: Query Writing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# a) Fetch movie with director and cast
query GetMovie {
  movie(id: "tt0111161") {
    title
    year
    director {
      name
    }
    cast(first: 5) {
      name
      role
    }
  }
}

# b) Search and handle all types
query SearchNolan {
  search(query: "Nolan") {
    __typename
    ... on Movie {
      title
      year
    }
    ... on Actor {
      name
      filmography { title }
    }
    ... on Director {
      name
      movies { title }
    }
  }
}

# c) Two movies with aliases
query TwoMovies {
  shawshank: movie(id: "tt0111161") {
    title
    year
    rating
  }
  godfather: movie(id: "tt0068646") {
    title
    year
    rating
  }
}
SOLUTION
}

# === Exercise 3: Mutation Payloads ===
# Problem: Redesign registerUser with Shopify-style payload pattern.
exercise_3() {
    echo "=== Exercise 3: Mutation Payloads ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
input RegisterUserInput {
  email: String!
  password: String!
  username: String!
}

type RegisterUserPayload {
  user: User
  userErrors: [UserError!]!
}

type UserError {
  field: [String!]!
  message: String!
  code: UserErrorCode!
}

enum UserErrorCode {
  BLANK
  INVALID
  TOO_SHORT
  TOO_LONG
  TAKEN
  INVALID_FORMAT
}

type Mutation {
  registerUser(input: RegisterUserInput!): RegisterUserPayload!
}

# Example response (success):
# {
#   "data": {
#     "registerUser": {
#       "user": { "id": "1", "username": "alice", "email": "alice@example.com" },
#       "userErrors": []
#     }
#   }
# }

# Example response (error):
# {
#   "data": {
#     "registerUser": {
#       "user": null,
#       "userErrors": [
#         { "field": ["input", "email"], "message": "Email already taken", "code": "TAKEN" },
#         { "field": ["input", "password"], "message": "Must be at least 8 characters", "code": "TOO_SHORT" }
#       ]
#     }
#   }
# }
SOLUTION
}

# === Exercise 4: Python Server ===
# Problem: Extend the minimal Strawberry server with Comment type.
exercise_4() {
    echo "=== Exercise 4: Python Server Extension ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import strawberry
from datetime import datetime
from typing import Optional


@strawberry.type
class Comment:
    id: strawberry.ID
    body: str
    author_id: strawberry.ID
    post_id: strawberry.ID
    created_at: datetime


COMMENTS = {}

@strawberry.input
class CreateCommentInput:
    body: str
    post_id: strawberry.ID
    author_id: strawberry.ID


@strawberry.type
class Query:
    @strawberry.field
    def comments_for_post(self, post_id: strawberry.ID) -> list[Comment]:
        return [c for c in COMMENTS.values() if c.post_id == post_id]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_comment(self, input: CreateCommentInput) -> Comment:
        comment_id = str(len(COMMENTS) + 300)
        comment = Comment(
            id=comment_id,
            body=input.body,
            author_id=input.author_id,
            post_id=input.post_id,
            created_at=datetime.now(),
        )
        COMMENTS[comment_id] = comment
        return comment
SOLUTION
}

# Run all exercises
main() {
    exercise_1
    echo ""
    exercise_2
    echo ""
    exercise_3
    echo ""
    exercise_4
}

main "$@"
