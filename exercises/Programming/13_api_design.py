"""
Exercises for Lesson 13: API Design
Topic: Programming

Solutions to practice problems from the lesson.
"""
import json


# === Exercise 1: Design a REST API for a Bookstore ===
# Problem: Design endpoints, request/response bodies, status codes, pagination.

def exercise_1():
    """Solution: Complete REST API design for an online bookstore."""

    api_design = {
        "Books": [
            {
                "method": "GET",
                "path": "/api/v1/books",
                "description": "List books with filtering and pagination",
                "query_params": "?genre=fiction&author=tolkien&page=1&per_page=20&sort=title",
                "response_status": 200,
                "response_body": {
                    "data": [{"id": 1, "title": "...", "author": "...", "genre": "...", "price": 29.99}],
                    "pagination": {"page": 1, "per_page": 20, "total": 150, "total_pages": 8},
                },
            },
            {
                "method": "GET",
                "path": "/api/v1/books/:id",
                "description": "Get a single book",
                "response_status": "200 (found) / 404 (not found)",
            },
            {
                "method": "GET",
                "path": "/api/v1/books/search",
                "description": "Full-text search",
                "query_params": "?q=lord+rings&page=1&per_page=20",
                "response_status": 200,
            },
        ],
        "Auth": [
            {
                "method": "POST",
                "path": "/api/v1/auth/register",
                "description": "Create user account",
                "request_body": {"email": "user@example.com", "password": "secure123"},
                "response_status": "201 (created) / 409 (email exists) / 422 (validation error)",
            },
            {
                "method": "POST",
                "path": "/api/v1/auth/login",
                "description": "Authenticate and get JWT token",
                "request_body": {"email": "user@example.com", "password": "secure123"},
                "response_status": "200 (success) / 401 (invalid credentials)",
            },
        ],
        "Cart": [
            {
                "method": "POST",
                "path": "/api/v1/cart/items",
                "description": "Add book to cart",
                "request_body": {"book_id": 1, "quantity": 2},
                "response_status": "201 (added) / 404 (book not found) / 409 (out of stock)",
            },
            {
                "method": "DELETE",
                "path": "/api/v1/cart/items/:book_id",
                "description": "Remove book from cart",
                "response_status": "204 (removed) / 404 (not in cart)",
            },
        ],
        "Orders": [
            {
                "method": "POST",
                "path": "/api/v1/orders",
                "description": "Place order from current cart",
                "request_body": {"shipping_address": "...", "payment_method_id": "pm_123"},
                "response_status": "201 (placed) / 402 (payment failed) / 422 (empty cart)",
            },
            {
                "method": "GET",
                "path": "/api/v1/orders",
                "description": "Order history (authenticated user)",
                "query_params": "?page=1&per_page=10&status=delivered",
                "response_status": 200,
            },
        ],
    }

    for category, endpoints in api_design.items():
        print(f"\n  {category}:")
        for ep in endpoints:
            print(f"    {ep['method']} {ep['path']}")
            print(f"      {ep['description']}")
            print(f"      Status: {ep['response_status']}")

    print("\n  Pagination strategy: Offset-based (page + per_page)")
    print("    - Default: page=1, per_page=20, max per_page=100")
    print("    - Response includes total count and total_pages")
    print("    - For large datasets, consider cursor-based pagination instead")


# === Exercise 2: API Review ===
# Problem: Identify violations in bad API design.

def exercise_2():
    """Solution: Review and fix API design violations."""

    violations = [
        {
            "bad": "GET /getBook?id=123",
            "issues": [
                "Verb in URL ('get') is redundant - HTTP method already says GET",
                "ID should be a path parameter, not query parameter",
            ],
            "fix": "GET /books/123",
        },
        {
            "bad": "POST /createBook?title=NewBook&author=JohnDoe",
            "issues": [
                "Verb in URL ('create') is redundant",
                "Data in query string instead of request body",
                "POST body should be JSON, not URL-encoded",
            ],
            "fix": "POST /books with JSON body {\"title\": \"NewBook\", \"author\": \"JohnDoe\"}",
        },
        {
            "bad": "GET /updateBook?id=123&title=UpdatedTitle",
            "issues": [
                "GET must not modify data (not safe/idempotent for mutations)",
                "Should use PUT or PATCH for updates",
                "Verb in URL again",
            ],
            "fix": "PATCH /books/123 with JSON body {\"title\": \"UpdatedTitle\"}",
        },
        {
            "bad": "DELETE /books/remove/123",
            "issues": [
                "Redundant 'remove' in path - DELETE verb already implies removal",
                "Inconsistent with other endpoints",
            ],
            "fix": "DELETE /books/123",
        },
        {
            "bad": "GET /books?page=1&limit=1000",
            "issues": [
                "limit=1000 is too large, will cause performance issues",
                "Should enforce a maximum page size (e.g., 100)",
            ],
            "fix": "GET /books?page=1&per_page=20 (max 100, default 20)",
        },
        {
            "bad": "POST /users/login returns 200 even if credentials wrong",
            "issues": [
                "200 means success - wrong credentials should return 401 Unauthorized",
                "Client can't distinguish success from failure by status code",
            ],
            "fix": "Return 200 on success with token, 401 on invalid credentials",
        },
    ]

    for v in violations:
        print(f"  Bad:  {v['bad']}")
        for issue in v["issues"]:
            print(f"    Problem: {issue}")
        print(f"  Fix:  {v['fix']}")
        print()


# === Exercise 3: Versioning Strategy ===
# Problem: Add breaking change (split name into first_name/last_name).

def exercise_3():
    """Solution: API versioning strategy for backward-compatible changes."""

    print("  Current: GET /api/v1/users/123 -> {\"name\": \"John Doe\", \"email\": \"...\"}")
    print()

    strategy = {
        "Step 1: Create v2 with new structure": {
            "endpoint": "GET /api/v2/users/123",
            "response": {"first_name": "John", "last_name": "Doe", "email": "john@example.com"},
        },
        "Step 2: Keep v1 running (backward compatible)": {
            "endpoint": "GET /api/v1/users/123",
            "response": {"name": "John Doe", "email": "john@example.com"},
            "note": "v1 internally computes name from first_name + last_name",
        },
        "Step 3: Add deprecation headers to v1": {
            "headers": {
                "Deprecation": "true",
                "Sunset": "2027-01-01",
                "Link": "</api/v2/users>; rel=\"successor-version\"",
            },
        },
        "Step 4: Migration period (6+ months)": {
            "actions": [
                "Document v2 changes in changelog",
                "Provide migration guide",
                "Monitor v1 usage metrics",
                "Notify active v1 consumers",
            ],
        },
        "Step 5: Sunset v1 after deadline": {
            "action": "Return 410 Gone for v1 endpoints with migration instructions",
        },
    }

    for step, details in strategy.items():
        print(f"  {step}:")
        for key, value in details.items():
            print(f"    {key}: {json.dumps(value, indent=6) if isinstance(value, (dict, list)) else value}")
        print()


# === Exercise 4: Error Handling Design ===
# Problem: Design error response format for a banking API.

def exercise_4():
    """Solution: Comprehensive error response format."""

    error_examples = {
        "Validation Error (400)": {
            "error": {
                "type": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": [
                    {"field": "amount", "message": "Must be a positive number", "value": -50},
                    {"field": "account_id", "message": "Required field is missing"},
                ],
                "request_id": "req_abc123",
                "timestamp": "2026-02-27T10:30:00Z",
            }
        },
        "Insufficient Funds (422)": {
            "error": {
                "type": "INSUFFICIENT_FUNDS",
                "message": "Account has insufficient funds for this transaction",
                "details": {
                    "requested": 5000.00,
                    "available": 2350.75,
                    "shortfall": 2649.25,
                },
                "request_id": "req_def456",
                "timestamp": "2026-02-27T10:31:00Z",
            }
        },
        "Account Locked (403)": {
            "error": {
                "type": "ACCOUNT_LOCKED",
                "message": "Account is locked due to suspicious activity",
                "details": {
                    "locked_at": "2026-02-27T09:00:00Z",
                    "reason": "Multiple failed authentication attempts",
                    "unlock_instructions": "Contact support at 1-800-BANK or visit a branch",
                },
                "request_id": "req_ghi789",
            }
        },
        "Rate Limited (429)": {
            "error": {
                "type": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please slow down.",
                "details": {
                    "limit": 100,
                    "window": "60s",
                    "retry_after": 45,
                },
                "request_id": "req_jkl012",
            }
        },
        "Server Error (500)": {
            "error": {
                "type": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. Our team has been notified.",
                "request_id": "req_mno345",
                "timestamp": "2026-02-27T10:32:00Z",
                "support_url": "https://bank.com/support?ref=req_mno345",
            }
        },
    }

    for scenario, response in error_examples.items():
        print(f"  {scenario}:")
        print(f"    {json.dumps(response, indent=4)}")
        print()


# === Exercise 5: Library API Design ===
# Problem: Design a fluent SQL query builder API.

def exercise_5():
    """Solution: Fluent SQL query builder using method chaining."""

    class QueryBuilder:
        """
        Fluent API for building SQL queries via method chaining.

        Each method returns 'self' to enable chaining.
        The build() method produces the final SQL string.
        """

        def __init__(self):
            self._columns = ["*"]
            self._table = ""
            self._conditions = []
            self._joins = []
            self._order = []
            self._limit_val = None
            self._offset_val = None
            self._params = []

        def select(self, *columns):
            """Specify columns to retrieve."""
            self._columns = list(columns) if columns else ["*"]
            return self

        def from_table(self, table):
            """Specify the table to query from."""
            self._table = table
            return self

        def where(self, condition, *params):
            """Add a WHERE condition (AND-combined)."""
            self._conditions.append(("AND", condition))
            self._params.extend(params)
            return self

        def or_where(self, condition, *params):
            """Add an OR condition."""
            self._conditions.append(("OR", condition))
            self._params.extend(params)
            return self

        def join(self, table, on_condition, join_type="INNER"):
            """Add a JOIN clause."""
            self._joins.append(f"{join_type} JOIN {table} ON {on_condition}")
            return self

        def left_join(self, table, on_condition):
            """Add a LEFT JOIN clause."""
            return self.join(table, on_condition, "LEFT")

        def order_by(self, column, direction="ASC"):
            """Add an ORDER BY clause."""
            self._order.append(f"{column} {direction}")
            return self

        def limit(self, count):
            """Limit the number of results."""
            self._limit_val = count
            return self

        def offset(self, count):
            """Skip the first N results."""
            self._offset_val = count
            return self

        def build(self):
            """Generate the SQL query string."""
            parts = [f"SELECT {', '.join(self._columns)}"]
            parts.append(f"FROM {self._table}")

            for join_clause in self._joins:
                parts.append(join_clause)

            if self._conditions:
                # First condition doesn't use the connector
                where_parts = [self._conditions[0][1]]
                for connector, cond in self._conditions[1:]:
                    where_parts.append(f"{connector} {cond}")
                parts.append(f"WHERE {' '.join(where_parts)}")

            if self._order:
                parts.append(f"ORDER BY {', '.join(self._order)}")

            if self._limit_val is not None:
                parts.append(f"LIMIT {self._limit_val}")

            if self._offset_val is not None:
                parts.append(f"OFFSET {self._offset_val}")

            return "\n".join(parts)

    # Demonstrate the fluent API
    print("  Query 1: Simple select with filter")
    q1 = (
        QueryBuilder()
        .select("id", "name", "email")
        .from_table("users")
        .where("age > ?", 18)
        .order_by("name")
        .limit(10)
        .build()
    )
    print(f"  {q1}\n")

    print("  Query 2: Join with multiple conditions")
    q2 = (
        QueryBuilder()
        .select("u.name", "o.total", "o.created_at")
        .from_table("users u")
        .join("orders o", "u.id = o.user_id")
        .where("o.total > ?", 100)
        .where("o.status = ?", "completed")
        .order_by("o.created_at", "DESC")
        .limit(20)
        .offset(40)
        .build()
    )
    print(f"  {q2}\n")

    print("  Query 3: Left join with OR condition")
    q3 = (
        QueryBuilder()
        .select("p.name", "c.name AS category")
        .from_table("products p")
        .left_join("categories c", "p.category_id = c.id")
        .where("p.price < ?", 50)
        .or_where("p.on_sale = ?", True)
        .order_by("p.price")
        .build()
    )
    print(f"  {q3}")


if __name__ == "__main__":
    print("=== Exercise 1: Design a REST API for a Bookstore ===")
    exercise_1()
    print("\n=== Exercise 2: API Review ===")
    exercise_2()
    print("\n=== Exercise 3: Versioning Strategy ===")
    exercise_3()
    print("\n=== Exercise 4: Error Handling Design ===")
    exercise_4()
    print("\n=== Exercise 5: Library API Design ===")
    exercise_5()
    print("\nAll exercises completed!")
