#!/usr/bin/env python3
"""Example: GraphQL Federation

Demonstrates Apollo-style GraphQL federation concepts:
- Subgraph schema definition with @key directives
- Entity resolution (__resolveReference)
- Gateway composition (simulated)
- Cross-service entity extension
- Federated queries across service boundaries

Related lesson: 20_GraphQL_Federation.md

Run:
    pip install strawberry-graphql[fastapi] "fastapi[standard]"
    uvicorn 20_graphql_federation:app --reload --port 8000

Note:
    In production, each subgraph runs as a separate service.
    A gateway (Apollo Router, Cosmo Router) composes them.
    This example simulates all subgraphs in one process for clarity.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import strawberry
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from strawberry.federation import Schema
from strawberry.types import Info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("federation")

# =============================================================================
# SUBGRAPH 1: USERS SERVICE — Owns the User entity
# =============================================================================
# In federation, each service "owns" certain entities and exposes them
# with a @key directive so other services can reference them.

USERS_DB = {
    "u1": {"id": "u1", "name": "Alice", "email": "alice@example.com"},
    "u2": {"id": "u2", "name": "Bob", "email": "bob@example.com"},
    "u3": {"id": "u3", "name": "Carol", "email": "carol@example.com"},
}


@strawberry.federation.type(keys=["id"])
class User:
    """User entity — owned by the Users subgraph.

    The @key directive tells the gateway that 'id' uniquely identifies
    this entity across all subgraphs.
    """
    id: str
    name: str
    email: str

    @classmethod
    def resolve_reference(cls, id: str, **kwargs) -> Optional["User"]:
        """Called by the gateway when another subgraph references a User.

        This is the federation equivalent of a foreign key lookup.
        The gateway sends the key fields, and we return the full entity.
        """
        data = USERS_DB.get(id)
        if not data:
            return None
        return User(**data)


# =============================================================================
# SUBGRAPH 2: PRODUCTS SERVICE — Owns the Product entity
# =============================================================================

PRODUCTS_DB = {
    "p1": {"id": "p1", "name": "Laptop", "price": 999.99, "category": "electronics"},
    "p2": {"id": "p2", "name": "Keyboard", "price": 79.99, "category": "electronics"},
    "p3": {"id": "p3", "name": "Notebook", "price": 12.50, "category": "stationery"},
}


@strawberry.federation.type(keys=["id"])
class Product:
    """Product entity — owned by the Products subgraph."""
    id: str
    name: str
    price: float
    category: str

    @classmethod
    def resolve_reference(cls, id: str, **kwargs) -> Optional["Product"]:
        data = PRODUCTS_DB.get(id)
        return Product(**data) if data else None


# =============================================================================
# SUBGRAPH 3: ORDERS SERVICE — Extends User and Product
# =============================================================================
# The Orders service does not own User or Product, but it extends them
# with order-related fields and references them via foreign keys.

ORDERS_DB = {
    "o1": {"id": "o1", "user_id": "u1", "product_id": "p1", "quantity": 1,
            "total": 999.99, "created_at": datetime(2026, 3, 1, tzinfo=timezone.utc)},
    "o2": {"id": "o2", "user_id": "u1", "product_id": "p2", "quantity": 2,
            "total": 159.98, "created_at": datetime(2026, 3, 5, tzinfo=timezone.utc)},
    "o3": {"id": "o3", "user_id": "u2", "product_id": "p3", "quantity": 5,
            "total": 62.50, "created_at": datetime(2026, 3, 10, tzinfo=timezone.utc)},
}


@strawberry.federation.type(keys=["id"])
class Order:
    """Order entity — owned by the Orders subgraph.

    References User and Product via their federation keys.
    The gateway resolves cross-service references automatically.
    """
    id: str
    quantity: int
    total: float
    created_at: datetime
    _user_id: strawberry.Private[str]
    _product_id: strawberry.Private[str]

    @strawberry.field
    def buyer(self) -> User:
        """Returns a User stub — the gateway calls User.resolve_reference."""
        return User(id=self._user_id, name="", email="")

    @strawberry.field
    def product(self) -> Product:
        """Returns a Product stub — the gateway resolves the full entity."""
        return Product(id=self._product_id, name="", price=0, category="")

    @classmethod
    def resolve_reference(cls, id: str, **kwargs) -> Optional["Order"]:
        data = ORDERS_DB.get(id)
        if not data:
            return None
        return Order(
            id=data["id"], quantity=data["quantity"], total=data["total"],
            created_at=data["created_at"],
            _user_id=data["user_id"], _product_id=data["product_id"],
        )


# =============================================================================
# QUERY — Combined (in production, each subgraph has its own Query)
# =============================================================================

@strawberry.type
class Query:
    @strawberry.field
    def user(self, id: str) -> Optional[User]:
        data = USERS_DB.get(id)
        return User(**data) if data else None

    @strawberry.field
    def product(self, id: str) -> Optional[Product]:
        data = PRODUCTS_DB.get(id)
        return Product(**data) if data else None

    @strawberry.field
    def orders(self, user_id: Optional[str] = None) -> list[Order]:
        results = list(ORDERS_DB.values())
        if user_id:
            results = [o for o in results if o["user_id"] == user_id]
        return [
            Order(id=o["id"], quantity=o["quantity"], total=o["total"],
                  created_at=o["created_at"],
                  _user_id=o["user_id"], _product_id=o["product_id"])
            for o in results
        ]

    @strawberry.field
    def order(self, id: str) -> Optional[Order]:
        data = ORDERS_DB.get(id)
        if not data:
            return None
        return Order(
            id=data["id"], quantity=data["quantity"], total=data["total"],
            created_at=data["created_at"],
            _user_id=data["user_id"], _product_id=data["product_id"],
        )


# =============================================================================
# SCHEMA & APP
# =============================================================================
# Use Schema (federation-aware) instead of strawberry.Schema.

schema = Schema(query=Query, enable_federation_2=True)
graphql_app = GraphQLRouter(schema)

app = FastAPI(title="GraphQL Federation Demo")
app.include_router(graphql_app, prefix="/graphql")

# =============================================================================
# EXAMPLE QUERIES
# =============================================================================

EXAMPLE_QUERIES = """
# === Cross-service query: order with buyer and product details ===
query {
  order(id: "o1") {
    id
    quantity
    total
    buyer { name email }
    product { name price category }
    createdAt
  }
}

# === List orders for a user ===
query {
  orders(userId: "u1") {
    id
    total
    product { name }
  }
}
"""

if __name__ == "__main__":
    import uvicorn
    print(EXAMPLE_QUERIES)
    uvicorn.run("20_graphql_federation:app", host="127.0.0.1", port=8000, reload=True)
