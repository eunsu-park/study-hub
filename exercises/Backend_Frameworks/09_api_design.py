# Exercise: API Design Patterns
# Practice with RESTful design, versioning, pagination, and error handling.

# Exercise 1: Design a RESTful API
# Design endpoints for an e-commerce platform:
# Resources: products, categories, orders, users, reviews
#
# TODO: List all endpoints with HTTP method, path, description
# Example:
#   GET    /api/products              — List products with pagination
#   POST   /api/products              — Create a product (admin only)
#   GET    /api/products/{id}         — Get product details
#   ...
# Include: nested resources, filtering, sorting


# Exercise 2: Implement Cursor Pagination
def cursor_paginate(items: list[dict], cursor: str | None, limit: int = 10) -> dict:
    """Implement cursor-based pagination.

    Args:
        items: List of dicts, each with an 'id' field (sorted by id)
        cursor: Opaque cursor string (base64 encoded id), or None for first page
        limit: Number of items per page

    Returns:
        {
            "data": [...],
            "next_cursor": "...",  # None if no more items
            "has_more": bool,
        }
    """
    # TODO: Implement
    pass


# Test
items = [{"id": i, "name": f"Item {i}"} for i in range(1, 51)]
# page1 = cursor_paginate(items, cursor=None, limit=10)
# page2 = cursor_paginate(items, cursor=page1["next_cursor"], limit=10)


# Exercise 3: Error Response Builder
# Implement RFC 7807 Problem Details format.

def problem_detail(
    status: int,
    title: str,
    detail: str,
    instance: str | None = None,
    errors: list[dict] | None = None,
) -> dict:
    """Build an RFC 7807 Problem Details response.

    Returns: {
        "type": "https://api.example.com/errors/{title-slug}",
        "title": title,
        "status": status,
        "detail": detail,
        "instance": instance,
        "errors": errors  # for validation errors
    }
    """
    # TODO: Implement
    pass


# Exercise 4: Rate Limit Headers
# Implement a function that calculates rate limit headers.

def rate_limit_headers(
    limit: int,
    remaining: int,
    reset_at: float,  # Unix timestamp
) -> dict:
    """Generate standard rate limit headers.

    Returns: {
        "X-RateLimit-Limit": str,
        "X-RateLimit-Remaining": str,
        "X-RateLimit-Reset": str,
        "Retry-After": str  # only if remaining == 0
    }
    """
    # TODO: Implement
    pass


# Exercise 5: API Versioning Strategies
# Compare three versioning approaches and implement URL-based versioning.

def version_router(path: str, handlers: dict) -> callable:
    """Route to the correct handler based on URL version prefix.

    Args:
        path: Request path like "/v2/users/123"
        handlers: {"v1": handler_v1, "v2": handler_v2}

    Returns: The matching handler, or raises ValueError
    """
    # TODO: Implement
    pass


if __name__ == "__main__":
    print("API Design Patterns Exercise")
    print("Implement each function and verify with the test cases.")
