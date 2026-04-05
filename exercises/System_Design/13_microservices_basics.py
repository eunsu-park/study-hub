"""
Exercises for Lesson 13: Microservices Basics
Topic: System_Design

Solutions to practice problems from the lesson.
Covers service decomposition, communication method selection,
and monolith-to-microservices migration using Strangler Fig pattern.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


# === Exercise 1: Service Decomposition ===
# Problem: Decompose an online bookstore into microservices.

def exercise_1():
    """Service decomposition for online bookstore."""
    print("Service Decomposition: Online Bookstore:")
    print("=" * 60)

    services = [
        {
            "name": "User Service",
            "bounded_context": "User Management",
            "responsibilities": [
                "User registration and authentication",
                "Profile management",
                "Address book",
            ],
            "database": "PostgreSQL (user_db)",
            "key_entities": ["User", "Address", "AuthCredential"],
        },
        {
            "name": "Catalog Service",
            "bounded_context": "Product Catalog",
            "responsibilities": [
                "Book metadata (title, author, ISBN, description)",
                "Categories and genres",
                "Search and browse",
                "Book reviews and ratings",
            ],
            "database": "PostgreSQL + Elasticsearch (catalog_db)",
            "key_entities": ["Book", "Category", "Review"],
        },
        {
            "name": "Inventory Service",
            "bounded_context": "Inventory Management",
            "responsibilities": [
                "Stock levels per warehouse",
                "Stock reservation",
                "Reorder triggers",
            ],
            "database": "PostgreSQL (inventory_db)",
            "key_entities": ["StockItem", "Warehouse", "Reservation"],
        },
        {
            "name": "Order Service",
            "bounded_context": "Order Processing",
            "responsibilities": [
                "Order creation and lifecycle",
                "Order status tracking",
                "Order history",
            ],
            "database": "PostgreSQL (order_db)",
            "key_entities": ["Order", "OrderItem", "OrderStatus"],
        },
        {
            "name": "Payment Service",
            "bounded_context": "Payment Processing",
            "responsibilities": [
                "Payment method management",
                "Payment processing (integration with Stripe, etc.)",
                "Refund handling",
            ],
            "database": "PostgreSQL (payment_db)",
            "key_entities": ["Payment", "Refund", "PaymentMethod"],
        },
        {
            "name": "Shipping Service",
            "bounded_context": "Shipping & Delivery",
            "responsibilities": [
                "Shipping rate calculation",
                "Carrier integration (UPS, FedEx)",
                "Delivery tracking",
            ],
            "database": "PostgreSQL (shipping_db)",
            "key_entities": ["Shipment", "Carrier", "TrackingEvent"],
        },
        {
            "name": "Notification Service",
            "bounded_context": "Communications",
            "responsibilities": [
                "Email notifications",
                "Push notifications",
                "SMS alerts",
            ],
            "database": "MongoDB (notification_db) + Redis queue",
            "key_entities": ["NotificationTemplate", "NotificationLog"],
        },
    ]

    for svc in services:
        print(f"\n  {svc['name']} (Context: {svc['bounded_context']})")
        print(f"    Database: {svc['database']}")
        print(f"    Responsibilities:")
        for resp in svc['responsibilities']:
            print(f"      - {resp}")
        print(f"    Key Entities: {', '.join(svc['key_entities'])}")

    # Context Mapping
    print("\n  Service Relationships (Context Map):")
    relationships = [
        ("Order Service", "Inventory Service", "Sync (gRPC)",
         "Reserve stock on order creation"),
        ("Order Service", "Payment Service", "Sync (gRPC)",
         "Process payment for order"),
        ("Order Service", "Shipping Service", "Async (Kafka)",
         "Initiate shipping after payment"),
        ("Order Service", "Notification Service", "Async (Kafka)",
         "Send order status updates"),
        ("Catalog Service", "Inventory Service", "Async (Kafka)",
         "Stock level updates for display"),
        ("User Service", "Notification Service", "Async (Kafka)",
         "Welcome email, account changes"),
    ]

    for src, dst, comm, desc in relationships:
        print(f"    {src} -> {dst} [{comm}]: {desc}")


# === Exercise 2: Communication Method Selection ===
# Problem: Choose sync/async for different scenarios.

def exercise_2():
    """Communication method selection for microservices."""
    scenarios = [
        ("Stock check when loading product detail page", "Synchronous (gRPC)",
         "User needs to see stock status immediately on page load. "
         "Blocking call with short timeout."),
        ("Email sending after order completion", "Asynchronous (Kafka event)",
         "User doesn't wait for email. Publish OrderCompleted event, "
         "notification service consumes it."),
        ("Inventory deduction after payment processing", "Asynchronous (Saga)",
         "Part of distributed transaction. Use Saga pattern with "
         "compensating transactions if payment or inventory fails."),
        ("User profile lookup", "Synchronous (gRPC)",
         "Needed for rendering pages. Fast response required. "
         "Cache results for repeated lookups."),
        ("Login event logging", "Asynchronous (Kafka event)",
         "Fire-and-forget. Security audit logs don't need immediate processing. "
         "High volume, no user-facing impact."),
    ]

    print("Communication Method Selection:")
    print("=" * 60)
    for i, (scenario, method, reason) in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario}")
        print(f"   Method: {method}")
        print(f"   Reason: {reason}")


# === Exercise 3: Migration Plan (Strangler Fig) ===
# Problem: Migrate monolith e-commerce to microservices.

@dataclass
class MigrationPhase:
    phase: int
    name: str
    duration: str
    services_extracted: List[str]
    description: str
    risks: List[str]


def exercise_3():
    """Strangler Fig migration plan for monolith to microservices."""
    print("Strangler Fig Migration Plan:")
    print("=" * 60)

    print("\nFirst service to separate: User/Auth Service")
    print("Reasons:")
    print("  1. Clear bounded context (well-defined API boundary)")
    print("  2. Minimal dependencies on other domains")
    print("  3. Enables independent scaling of auth (login spikes)")
    print("  4. Security benefits from isolated auth service")
    print("  5. Reusable across future services (SSO)")

    phases = [
        MigrationPhase(
            1, "Foundation", "2-4 weeks",
            ["API Gateway"],
            "Set up API Gateway as facade in front of monolith. "
            "All traffic routes through gateway. No behavior change.",
            ["Gateway becomes SPOF", "Latency increase from extra hop"]
        ),
        MigrationPhase(
            2, "Extract Auth", "4-6 weeks",
            ["User/Auth Service"],
            "Extract user registration, login, profile management. "
            "Gateway routes /auth/*, /users/* to new service. "
            "Monolith uses shared DB initially, then migrates.",
            ["Data migration complexity", "Session management during transition"]
        ),
        MigrationPhase(
            3, "Extract Catalog", "4-6 weeks",
            ["Catalog Service"],
            "Extract product catalog and search. "
            "Add Elasticsearch for search functionality. "
            "Monolith still handles orders/payments.",
            ["Search index consistency", "Image/asset migration"]
        ),
        MigrationPhase(
            4, "Extract Orders & Payments", "6-8 weeks",
            ["Order Service", "Payment Service"],
            "Most complex phase. Implement Saga pattern for "
            "distributed order processing. Payment service "
            "integrates with payment providers.",
            ["Distributed transactions", "Data consistency", "Rollback scenarios"]
        ),
        MigrationPhase(
            5, "Extract Remaining", "4-6 weeks",
            ["Inventory Service", "Shipping Service", "Notification Service"],
            "Extract remaining services. Decommission monolith. "
            "Full event-driven architecture with Kafka.",
            ["Event ordering", "Monitoring gaps during transition"]
        ),
    ]

    print("\n  Migration Roadmap:")
    print("  " + "=" * 58)
    for phase in phases:
        print(f"\n  Phase {phase.phase}: {phase.name} ({phase.duration})")
        print(f"    Services: {', '.join(phase.services_extracted)}")
        print(f"    Description: {phase.description}")
        print(f"    Risks:")
        for risk in phase.risks:
            print(f"      - {risk}")

    # Visualize the strangler fig pattern
    print("\n  Strangler Fig Progression:")
    print("  " + "-" * 58)
    stages = [
        ("Before", "  [=== Monolith ===========================]"),
        ("Phase 1", "  [GW]-->[=== Monolith =======================]"),
        ("Phase 2", "  [GW]-->[Auth]  [=== Monolith =============]"),
        ("Phase 3", "  [GW]-->[Auth] [Catalog]  [== Monolith ====]"),
        ("Phase 4", "  [GW]-->[Auth] [Catalog] [Order] [Pay] [Mo]"),
        ("Phase 5", "  [GW]-->[Auth] [Catalog] [Order] [Pay] [Ship] [Notif] [Inv]"),
    ]
    for stage, diagram in stages:
        print(f"    {stage:>8}: {diagram}")

    # Total timeline
    print("\n  Estimated total timeline: 20-30 weeks")
    print("  Key principles:")
    print("    - Always keep the system running (no big-bang migration)")
    print("    - Feature flags for gradual traffic shifting")
    print("    - Automated rollback capability at each phase")
    print("    - Comprehensive monitoring before decommissioning monolith")


if __name__ == "__main__":
    print("=" * 60)
    print("=== Exercise 1: Service Decomposition ===")
    print("=" * 60)
    exercise_1()

    print("\n" + "=" * 60)
    print("=== Exercise 2: Communication Method Selection ===")
    print("=" * 60)
    exercise_2()

    print("\n" + "=" * 60)
    print("=== Exercise 3: Strangler Fig Migration ===")
    print("=" * 60)
    exercise_3()

    print("\nAll exercises completed!")
