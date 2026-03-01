# Exercise: Backend Web Fundamentals
# Practice HTTP concepts and request/response handling.

# Exercise 1: Parse an HTTP request
# Given a raw HTTP request string, extract method, path, and headers.

def parse_http_request(raw: str) -> dict:
    """Parse a raw HTTP request into components.

    Returns: {"method": str, "path": str, "headers": dict, "body": str}
    """
    # TODO: Implement
    pass


# Test
raw_request = """GET /api/users?page=1 HTTP/1.1
Host: example.com
Accept: application/json
Authorization: Bearer token123

"""

result = parse_http_request(raw_request)
# Expected: {"method": "GET", "path": "/api/users?page=1",
#            "headers": {"Host": "example.com", ...}, "body": ""}


# Exercise 2: Build a simple HTTP response
def build_http_response(status_code: int, body: str, content_type: str = "application/json") -> str:
    """Build a raw HTTP response string."""
    # TODO: Implement
    # Should include: status line, Content-Type, Content-Length, empty line, body
    pass


# Exercise 3: URL Router
# Implement a simple URL router that matches paths to handlers.

class Router:
    def __init__(self):
        self.routes = []

    def add_route(self, method: str, path: str, handler):
        """Register a route. Path can contain params like /users/{id}"""
        # TODO: Implement
        pass

    def match(self, method: str, path: str):
        """Find matching handler and extract path params.

        Returns: (handler, params_dict) or (None, {})
        """
        # TODO: Implement
        pass


# Test
router = Router()
router.add_route("GET", "/users/{id}", lambda params: f"User {params['id']}")
router.add_route("GET", "/users/{id}/posts", lambda params: f"Posts by {params['id']}")

# handler, params = router.match("GET", "/users/42")
# assert params == {"id": "42"}


# Exercise 4: Content Negotiation
def negotiate_content_type(accept_header: str, available: list[str]) -> str | None:
    """Given an Accept header and available types, return the best match.

    Supports quality values: text/html;q=0.9, application/json;q=1.0
    Returns None if no match found.
    """
    # TODO: Implement
    pass


# Test
# negotiate_content_type("text/html;q=0.9, application/json", ["application/json", "text/html"])
# Expected: "application/json" (higher quality)


if __name__ == "__main__":
    print("Run each exercise and verify output.")
    print("All exercises test HTTP fundamentals concepts.")
