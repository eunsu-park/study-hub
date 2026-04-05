#!/bin/bash
# Exercises for Lesson 22: GraphQL Federation
# Topic: API_Design
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Design Subgraph Boundaries ==="
    cat << 'SOLUTION'
# Subgraph: Users
#   Owns: User, Address, Profile
#   Entities: User @key(fields: "id")

# Subgraph: Products
#   Owns: Product, Category, ProductVariant
#   Entities: Product @key(fields: "id")
#   Extends: (none)

# Subgraph: Orders
#   Owns: Order, OrderItem
#   Entities: Order @key(fields: "id")
#   Extends: User (adds `orders` field), Product (adds `orders` field)
#   @requires: OrderItem.lineTotal @requires(fields: "price") from Product

# Subgraph: Reviews
#   Owns: Review
#   Entities: Review @key(fields: "id")
#   Extends: Product (adds `reviews`, `averageRating` fields), User (adds `reviews` field)

# Subgraph: Inventory
#   Owns: InventoryItem, Warehouse
#   Extends: ProductVariant (adds `inStock`, `inventoryCount` fields)
#   @external: ProductVariant.sku

# Subgraph: Payments
#   Owns: Payment, Refund, PaymentMethod
#   Extends: Order (adds `payment` field)
#   @requires: Payment.receiptUrl @requires(fields: "total currency") from Order

# Subgraph: Shipping
#   Owns: Shipment, TrackingEvent
#   Extends: Order (adds `shipment` field)
#   @external: Order.shippingAddress from Orders
#   @requires: Shipment.estimatedDelivery @requires(fields: "shippingAddress") from Order
SOLUTION
}

exercise_2() {
    echo "=== Exercise 2: Implement Two Subgraphs ==="
    cat << 'SOLUTION'
# --- Users Subgraph ---
import strawberry
from strawberry.federation import Schema

USERS_DB = {
    "1": {"id": "1", "username": "alice", "email": "alice@example.com"},
    "2": {"id": "2", "username": "bob", "email": "bob@example.com"},
}

@strawberry.federation.type(keys=["id"])
class User:
    id: strawberry.ID
    username: str
    email: str

    @classmethod
    def resolve_reference(cls, id: strawberry.ID) -> "User":
        data = USERS_DB.get(str(id))
        if data:
            return cls(**data)
        return None

@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: strawberry.ID) -> User | None:
        data = USERS_DB.get(str(id))
        return User(**data) if data else None

    @strawberry.field
    def users(self, limit: int = 10) -> list[User]:
        return [User(**d) for d in list(USERS_DB.values())[:limit]]

users_schema = Schema(query=Query, enable_federation_2=True)


# --- Posts Subgraph ---
POSTS_DB = {
    "101": {"id": "101", "title": "Hello World", "author_id": "1"},
    "102": {"id": "102", "title": "GraphQL Rocks", "author_id": "2"},
}

@strawberry.federation.type(keys=["id"])
class User:
    id: strawberry.ID

    @strawberry.field
    def posts(self) -> list["Post"]:
        return [
            Post(**p) for p in POSTS_DB.values()
            if p["author_id"] == str(self.id)
        ]

@strawberry.federation.type(keys=["id"])
class Post:
    id: strawberry.ID
    title: str
    author_id: strawberry.ID

    @strawberry.field
    def author(self) -> User:
        return User(id=self.author_id)

    @classmethod
    def resolve_reference(cls, id: strawberry.ID) -> "Post":
        data = POSTS_DB.get(str(id))
        return cls(**data) if data else None

@strawberry.type
class PostQuery:
    @strawberry.field
    def post(self, id: strawberry.ID) -> Post | None:
        data = POSTS_DB.get(str(id))
        return Post(**data) if data else None

posts_schema = Schema(query=PostQuery, enable_federation_2=True)
SOLUTION
}

main() { exercise_1; echo ""; exercise_2; }
main "$@"
