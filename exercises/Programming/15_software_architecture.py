"""
Exercises for Lesson 15: Software Architecture
Topic: Programming

Solutions to practice problems from the lesson.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Protocol


# === Exercise 1: Choose Architecture for Scenarios ===
# Problem: Recommend architecture for three different scenarios.

def exercise_1():
    """Solution: Architecture recommendations for different scenarios."""

    scenarios = {
        "1. Startup MVP - Task management app (3 devs, 6 months)": {
            "recommendation": "Monolith",
            "justification": [
                "3 developers can't support microservice complexity (deployment, networking, monitoring)",
                "Speed to market is critical for startups; monolith ships faster",
                "Task management doesn't need independent scaling of components",
                "Can always extract services later when product-market fit is proven",
                "Single deployable artifact simplifies CI/CD and debugging",
            ],
            "tech_stack": "Django/Rails/Next.js monolith, single PostgreSQL database",
        },
        "2. E-commerce (100 devs, 1M daily users, scale checkout independently)": {
            "recommendation": "Microservices",
            "justification": [
                "100 developers need team autonomy - microservices enable independent deployment",
                "Checkout MUST scale independently (Black Friday spikes are 10x normal load)",
                "Different services have different scaling profiles (catalog reads vs payment writes)",
                "Team ownership: dedicated teams for catalog, cart, payment, shipping",
                "Fault isolation: checkout failures shouldn't crash product browsing",
            ],
            "tech_stack": "Kubernetes, API Gateway, event bus (Kafka), separate DBs per service",
        },
        "3. Internal data processing tool (monthly batch jobs, 2 devs)": {
            "recommendation": "Serverless / Simple Monolith",
            "justification": [
                "Monthly execution means paying for always-on servers is wasteful",
                "Serverless (AWS Lambda/Step Functions) scales to zero between runs",
                "2 developers can't maintain infrastructure for microservices",
                "Batch processing fits serverless event-driven model naturally",
                "Alternative: single Python script orchestrated by Airflow/cron",
            ],
            "tech_stack": "AWS Lambda + Step Functions, or a single Python script + cron",
        },
    }

    for scenario, info in scenarios.items():
        print(f"  {scenario}")
        print(f"    Recommendation: {info['recommendation']}")
        for point in info["justification"]:
            print(f"      - {point}")
        print(f"    Stack: {info['tech_stack']}")
        print()


# === Exercise 2: Design a Layered Architecture ===
# Problem: Design 3-tier architecture for a blogging platform.

def exercise_2():
    """Solution: Three-layer architecture for a blog platform."""

    layers = {
        "Presentation Layer (UI/API)": {
            "responsibility": "Handle HTTP requests, render responses, input validation",
            "components": [
                "AuthController: login, register, logout endpoints",
                "PostController: CRUD endpoints for blog posts",
                "CommentController: create, delete, list comments",
                "SearchController: full-text search endpoint",
                "Middleware: authentication, rate limiting, CORS",
                "Serializers: convert domain objects to JSON responses",
            ],
        },
        "Business Logic Layer (Domain/Service)": {
            "responsibility": "Core business rules, orchestration, authorization",
            "components": [
                "AuthService: password hashing, JWT generation, permission checks",
                "PostService: create/edit/delete rules, draft/publish workflow",
                "CommentService: moderation rules, spam detection, threading",
                "SearchService: indexing strategy, query parsing, relevance ranking",
                "NotificationService: email on new comment, mention alerts",
            ],
        },
        "Data Access Layer (Persistence)": {
            "responsibility": "Database queries, caching, external API calls",
            "components": [
                "UserRepository: user CRUD, credential storage",
                "PostRepository: post CRUD, tag management, pagination",
                "CommentRepository: comment CRUD, threading queries",
                "SearchIndex: full-text index management (e.g., Elasticsearch adapter)",
                "CacheManager: Redis caching for hot posts and search results",
            ],
        },
    }

    for layer, info in layers.items():
        print(f"  {layer}")
        print(f"    Responsibility: {info['responsibility']}")
        print(f"    Components:")
        for comp in info["components"]:
            print(f"      - {comp}")
        print()

    print("  Key rule: Dependencies flow DOWN only")
    print("    Presentation -> Business Logic -> Data Access")
    print("    Never: Data Access -> Business Logic (use callbacks/events if needed)")


# === Exercise 3: Refactor to Hexagonal Architecture ===
# Problem: Refactor tightly coupled Flask+PostgreSQL code.

def exercise_3():
    """Solution: Hexagonal (Ports & Adapters) architecture refactoring."""

    # 1. Domain Entity - pure business logic, no framework dependencies
    @dataclass
    class User:
        """Domain entity: represents a user in the system."""
        name: str
        email: str
        id: Optional[int] = None

        def validate(self):
            """Business rule: validate user data."""
            if not self.name or len(self.name) < 2:
                raise ValueError("Name must be at least 2 characters")
            if not self.email or "@" not in self.email:
                raise ValueError("Invalid email address")

    # 2. Port (interface) - defines what the domain NEEDS
    class UserRepository(ABC):
        """
        Port: interface for user persistence.
        The domain defines this interface; adapters implement it.
        This is the key insight of hexagonal architecture:
        the domain doesn't know about databases.
        """
        @abstractmethod
        def save(self, user: User) -> User:
            pass

        @abstractmethod
        def find_by_email(self, email: str) -> Optional[User]:
            pass

    # 3. Use Case - orchestrates domain logic
    class CreateUserUseCase:
        """
        Application service: coordinates domain objects and ports.
        Depends only on the Port interface, not on any specific database.
        """
        def __init__(self, user_repo: UserRepository):
            self._repo = user_repo  # Injected dependency (DIP)

        def execute(self, name: str, email: str) -> User:
            user = User(name=name, email=email)
            user.validate()

            # Check for duplicate email
            existing = self._repo.find_by_email(email)
            if existing:
                raise ValueError(f"Email {email} already registered")

            return self._repo.save(user)

    # 4. Adapter (implementation) - connects to specific technology
    class InMemoryUserRepository(UserRepository):
        """
        Adapter: in-memory implementation of UserRepository.
        In production, this would be PostgreSQLUserRepository.
        For tests, this in-memory version avoids database dependencies.
        """
        def __init__(self):
            self._users = {}
            self._next_id = 1

        def save(self, user: User) -> User:
            user.id = self._next_id
            self._next_id += 1
            self._users[user.email] = user
            return user

        def find_by_email(self, email: str) -> Optional[User]:
            return self._users.get(email)

    # Demonstration
    repo = InMemoryUserRepository()
    use_case = CreateUserUseCase(repo)

    # Success case
    user = use_case.execute("Alice", "alice@example.com")
    print(f"  Created: {user}")

    # Duplicate email
    try:
        use_case.execute("Alice2", "alice@example.com")
    except ValueError as e:
        print(f"  Duplicate: {e}")

    # Validation error
    try:
        use_case.execute("", "bad")
    except ValueError as e:
        print(f"  Validation: {e}")

    print("\n  Architecture layers:")
    print("    Domain (User entity) - no dependencies")
    print("    Ports (UserRepository ABC) - defined by domain")
    print("    Use Cases (CreateUserUseCase) - depends on ports only")
    print("    Adapters (InMemoryUserRepository) - implements ports")
    print("    Framework (Flask routes) - calls use cases (not shown)")


# === Exercise 4: Document an Architectural Decision ===
# Problem: Write an ADR for choosing PostgreSQL over MongoDB.

def exercise_4():
    """Solution: Architectural Decision Record (ADR)."""

    adr = """
  ADR-001: Choose PostgreSQL over MongoDB for User Data
  =====================================================

  Status: Accepted
  Date: 2026-02-27
  Deciders: Engineering team

  Context:
    We need a database for storing user profiles, relationships,
    and transactional data (orders, payments). The data is highly
    relational (users have orders, orders have items, items belong
    to categories). We need ACID transactions for payment processing.

  Decision:
    We will use PostgreSQL as our primary database.

  Alternatives Considered:
    1. MongoDB
       - Pros: Flexible schema, horizontal scaling, JSON-native
       - Cons: No ACID transactions across collections (pre-4.0 style),
         relational data requires denormalization (duplication),
         eventual consistency complicates payment logic

    2. PostgreSQL
       - Pros: ACID transactions, strong data integrity (foreign keys),
         excellent for relational data, JSONB for semi-structured needs,
         mature ecosystem (pgAdmin, replication, partitioning)
       - Cons: Vertical scaling is primary model, schema migrations needed

  Rationale:
    - User data is fundamentally relational (users -> orders -> items)
    - Payment processing REQUIRES ACID transactions
    - PostgreSQL's JSONB gives us flexible schema where needed
    - Our team has strong SQL expertise
    - Data integrity is more important than schema flexibility

  Consequences:
    - Need schema migration tooling (Alembic/Flyway)
    - Horizontal scaling via read replicas + connection pooling
    - May add Redis for caching hot paths
    - May add Elasticsearch for full-text search (not PostgreSQL FTS)

  Review Date: 2026-08-27 (6 months)
"""

    print(adr)


# === Exercise 5: Analyze CAP Trade-offs ===
# Problem: Choose CAP priorities for a collaborative document editor.

def exercise_5():
    """Solution: CAP theorem analysis for a collaborative editor."""

    print("  Scenario: Real-time collaborative document editor (like Google Docs)")
    print("  Users must see edits instantly; system must work during partitions.")
    print()

    analysis = {
        "CAP Theorem Recap": [
            "C (Consistency): Every read sees the most recent write",
            "A (Availability): Every request gets a response",
            "P (Partition Tolerance): System works despite network failures",
            "You MUST choose P (network failures are inevitable). So: CP or AP?",
        ],
        "Recommendation: AP (Availability + Partition tolerance)": [
            "Users must ALWAYS be able to edit (availability is critical for UX)",
            "If we chose CP, a network partition would LOCK users out of editing",
            "Locked-out users means lost work and terrible user experience",
            "Temporary inconsistency (seeing stale edits) is acceptable",
        ],
        "How to handle eventual consistency": [
            "Use CRDTs (Conflict-Free Replicated Data Types) for document state",
            "CRDTs mathematically guarantee convergence without coordination",
            "Operational Transformation (OT) as alternative (Google Docs approach)",
            "Each user works on local replica, sync happens in background",
            "Conflicts are auto-resolved by the CRDT/OT algorithm",
        ],
        "Trade-offs accepted": [
            "Users may briefly see different versions during network issues",
            "Edits may appear slightly delayed (not instant) during partitions",
            "Conflict resolution may occasionally produce unexpected merges",
            "System is eventually consistent: all users converge to same state",
        ],
    }

    for section, points in analysis.items():
        print(f"  {section}:")
        for point in points:
            print(f"    - {point}")
        print()


if __name__ == "__main__":
    print("=== Exercise 1: Choose Architecture for Scenarios ===")
    exercise_1()
    print("\n=== Exercise 2: Design a Layered Architecture ===")
    exercise_2()
    print("\n=== Exercise 3: Refactor to Hexagonal Architecture ===")
    exercise_3()
    print("\n=== Exercise 4: Document an Architectural Decision ===")
    exercise_4()
    print("\n=== Exercise 5: Analyze CAP Trade-offs ===")
    exercise_5()
    print("\nAll exercises completed!")
