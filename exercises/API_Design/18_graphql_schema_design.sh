#!/bin/bash
# Exercises for Lesson 18: GraphQL Schema Design
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: E-Commerce Schema ===
exercise_1() {
    echo "=== Exercise 1: E-Commerce Schema ==="
    cat << 'SOLUTION'
scalar DateTime
scalar Decimal

type Product @key(fields: "id") {
  id: ID!
  name: String!
  description: String!
  variants: [ProductVariant!]!
  category: Category!
  reviews(first: Int, after: String): ReviewConnection!
  averageRating: Float
  createdAt: DateTime!
}

type ProductVariant {
  id: ID!
  sku: String!
  size: String
  color: String
  price: Decimal!
  compareAtPrice: Decimal
  inStock: Boolean!
  inventoryCount: Int!
}

type Category {
  id: ID!
  name: String!
  parent: Category
  children: [Category!]!
  products(first: Int, after: String): ProductConnection!
}

type Cart {
  id: ID!
  items: [CartItem!]!
  subtotal: Decimal!
  tax: Decimal!
  total: Decimal!
  itemCount: Int!
}

type CartItem {
  id: ID!
  variant: ProductVariant!
  quantity: Int!
  lineTotal: Decimal!
}

type Order {
  id: ID!
  status: OrderStatus!
  items: [OrderItem!]!
  total: Decimal!
  createdAt: DateTime!
}

enum OrderStatus {
  PENDING
  CONFIRMED
  SHIPPED
  DELIVERED
  CANCELLED
  REFUNDED
}

type Review {
  id: ID!
  rating: Int!
  comment: String
  author: User!
  product: Product!
  createdAt: DateTime!
}

# Connections for cursor-based pagination
type ProductConnection {
  edges: [ProductEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type ProductEdge {
  node: Product!
  cursor: String!
}

type ReviewConnection {
  edges: [ReviewEdge!]!
  pageInfo: PageInfo!
  totalCount: Int!
}

type ReviewEdge {
  node: Review!
  cursor: String!
}

type PageInfo {
  hasNextPage: Boolean!
  hasPreviousPage: Boolean!
  startCursor: String
  endCursor: String
}
SOLUTION
}

# === Exercise 2: Input Type Design ===
exercise_2() {
    echo "=== Exercise 2: Input Type Design ==="
    cat << 'SOLUTION'
# createReview
input CreateReviewInput {
  productId: ID!
  rating: Int!        # 1-5
  comment: String     # Optional
}

type CreateReviewPayload {
  review: Review
  userErrors: [UserError!]!
}

# updateReview
input UpdateReviewInput {
  rating: Int
  comment: String
}

type UpdateReviewPayload {
  review: Review
  userErrors: [UserError!]!
}

# flagReview
input FlagReviewInput {
  reviewId: ID!
  reason: FlagReason!
  details: String
}

enum FlagReason {
  SPAM
  INAPPROPRIATE
  FAKE
  OFF_TOPIC
  HARASSMENT
}

type FlagReviewPayload {
  review: Review
  userErrors: [UserError!]!
}
SOLUTION
}

# === Exercise 3: Custom Scalars ===
exercise_3() {
    echo "=== Exercise 3: Custom Scalars ==="
    cat << 'SOLUTION'
import strawberry
import re

# Currency scalar (ISO 4217)
VALID_CURRENCIES = {"USD", "EUR", "GBP", "JPY", "KRW", "CNY", "CHF"}

CurrencyScalar = strawberry.scalar(
    str,
    name="Currency",
    description="ISO 4217 currency code",
    serialize=lambda v: str(v),
    parse_value=lambda v: _validate_currency(v),
)

def _validate_currency(value: str) -> str:
    if value.upper() not in VALID_CURRENCIES:
        raise ValueError(f"Invalid currency code: {value}")
    return value.upper()

# Latitude scalar
LatitudeScalar = strawberry.scalar(
    float,
    name="Latitude",
    description="Floating point between -90 and 90",
    serialize=lambda v: float(v),
    parse_value=lambda v: _validate_latitude(v),
)

def _validate_latitude(value: float) -> float:
    if not (-90 <= value <= 90):
        raise ValueError(f"Latitude must be between -90 and 90, got {value}")
    return float(value)

# Longitude scalar
LongitudeScalar = strawberry.scalar(
    float,
    name="Longitude",
    description="Floating point between -180 and 180",
    serialize=lambda v: float(v),
    parse_value=lambda v: _validate_longitude(v),
)

def _validate_longitude(value: float) -> float:
    if not (-180 <= value <= 180):
        raise ValueError(f"Longitude must be between -180 and 180, got {value}")
    return float(value)
SOLUTION
}

# === Exercise 4: Schema Refactoring ===
exercise_4() {
    echo "=== Exercise 4: Schema Refactoring ==="
    cat << 'SOLUTION'
# Problems identified:
# 1. id is Int (should be ID!)
# 2. All fields nullable (most should be non-null)
# 3. "desc" abbreviation (use "description")
# 4. price as Float (use Decimal for money)
# 5. price_currency mixed into Product (separate Money type)
# 6. cat_id/cat_name denormalized (use Category type)
# 7. created as String (use DateTime scalar)
# 8. tags as comma-separated String (use [String!]!)
# 9. snake_case naming (should be camelCase)

# Refactored schema:
scalar DateTime
scalar Decimal

type Product {
  id: ID!
  name: String!
  description: String!
  price: Money!
  category: Category!
  tags: [String!]!
  createdAt: DateTime!
}

type Money {
  amount: Decimal!
  currency: CurrencyCode!
}

enum CurrencyCode {
  USD
  EUR
  GBP
  JPY
}

type Category {
  id: ID!
  name: String!
  products(first: Int, after: String): ProductConnection!
}
SOLUTION
}

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
