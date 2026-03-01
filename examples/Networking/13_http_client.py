"""
HTTP Client

Demonstrates:
- Raw HTTP/1.1 request building
- HTTP methods (GET, POST, PUT, DELETE)
- Headers and body handling
- Response parsing (status, headers, body)
- Chunked transfer encoding

Theory:
- HTTP is a request-response protocol over TCP.
- Request: method + path + version, headers, optional body.
- Response: version + status code + reason, headers, body.
- Headers: key-value pairs (Content-Type, Content-Length, etc.)
- HTTP/1.1 requires Host header; persistent connections by default.

Adapted from Networking Lesson 13.
"""

from dataclasses import dataclass, field
from urllib.parse import urlparse, urlencode


@dataclass
class HTTPRequest:
    method: str
    path: str
    headers: dict[str, str] = field(default_factory=dict)
    body: str = ""
    version: str = "HTTP/1.1"

    # Why: Building the request as a raw string (not using urllib/requests)
    # reveals the wire format that HTTP libraries abstract away. Every HTTP
    # request is just ASCII text over a TCP stream — understanding this is
    # crucial for debugging with tools like tcpdump or Wireshark.
    def build(self) -> str:
        """Build raw HTTP request string."""
        lines = [f"{self.method} {self.path} {self.version}"]

        # Why: Content-Length is mandatory for requests with a body so the
        # server knows exactly how many bytes to read. Without it, the server
        # has no way to determine where the body ends (unlike chunked encoding).
        if self.body and "Content-Length" not in self.headers:
            self.headers["Content-Length"] = str(len(self.body.encode()))

        for key, value in self.headers.items():
            lines.append(f"{key}: {value}")

        # Why: The blank line (\r\n\r\n) is the mandatory delimiter between
        # headers and body per RFC 7230. Parsers rely on this to know when
        # headers end, regardless of whether a body follows.
        lines.append("")  # Empty line separates headers from body
        if self.body:
            lines.append(self.body)
        else:
            lines.append("")

        return "\r\n".join(lines)


@dataclass
class HTTPResponse:
    version: str
    status_code: int
    reason: str
    headers: dict[str, str]
    body: str

    @classmethod
    def parse(cls, raw: str) -> "HTTPResponse":
        """Parse raw HTTP response string."""
        # Why: Splitting on \r\n\r\n with maxsplit=1 correctly handles bodies
        # that themselves contain \r\n\r\n sequences (e.g., nested HTTP in
        # proxy responses). Only the first double-CRLF is the header/body boundary.
        parts = raw.split("\r\n\r\n", 1)
        header_section = parts[0]
        body = parts[1] if len(parts) > 1 else ""

        lines = header_section.split("\r\n")
        status_line = lines[0].split(" ", 2)
        version = status_line[0]
        status_code = int(status_line[1])
        reason = status_line[2] if len(status_line) > 2 else ""

        headers = {}
        for line in lines[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key] = value

        return cls(version, status_code, reason, headers, body)


# ── Simulated HTTP Server ─────────────────────────────────────────────

# Why: An in-memory server avoids real socket I/O, letting us focus on HTTP
# semantics (request routing, status codes, CRUD operations) without network
# setup complexity. The route-matching approach mirrors real frameworks like
# Flask and Express.
class SimpleHTTPServer:
    """In-memory HTTP server for demonstration."""

    def __init__(self):
        self.routes: dict[tuple[str, str], callable] = {}
        self.data: dict[str, dict] = {}  # Simple data store

    def route(self, method: str, path: str):
        def decorator(func):
            self.routes[(method, path)] = func
            return func
        return decorator

    def handle(self, request: HTTPRequest) -> HTTPResponse:
        """Process an HTTP request and return a response."""
        handler = self.routes.get((request.method, request.path))
        if handler:
            return handler(request)

        # Why: Wildcard pattern matching (path/*) enables RESTful resource
        # routing where the ID is part of the URL path (e.g., /api/users/42).
        # Real frameworks use regex or parameterized routes for this.
        for (method, path), handler in self.routes.items():
            if method != request.method:
                continue
            if path.endswith("/*"):
                prefix = path[:-1]
                if request.path.startswith(prefix):
                    return handler(request)

        return HTTPResponse(
            "HTTP/1.1", 404, "Not Found",
            {"Content-Type": "text/plain"},
            "404 Not Found"
        )


# ── Build Demo Server ─────────────────────────────────────────────────

def create_demo_server() -> SimpleHTTPServer:
    server = SimpleHTTPServer()
    server.data = {
        "1": {"id": "1", "name": "Alice", "email": "alice@example.com"},
        "2": {"id": "2", "name": "Bob", "email": "bob@example.com"},
    }

    @server.route("GET", "/api/users")
    def list_users(req):
        import json
        body = json.dumps(list(server.data.values()), indent=2)
        return HTTPResponse("HTTP/1.1", 200, "OK",
                            {"Content-Type": "application/json",
                             "Content-Length": str(len(body))},
                            body)

    @server.route("GET", "/api/users/*")
    def get_user(req):
        import json
        user_id = req.path.split("/")[-1]
        user = server.data.get(user_id)
        if user:
            body = json.dumps(user, indent=2)
            return HTTPResponse("HTTP/1.1", 200, "OK",
                                {"Content-Type": "application/json"},
                                body)
        return HTTPResponse("HTTP/1.1", 404, "Not Found",
                            {"Content-Type": "application/json"},
                            '{"error": "User not found"}')

    @server.route("POST", "/api/users")
    def create_user(req):
        import json
        data = json.loads(req.body)
        new_id = str(max(int(k) for k in server.data) + 1)
        data["id"] = new_id
        server.data[new_id] = data
        body = json.dumps(data, indent=2)
        return HTTPResponse("HTTP/1.1", 201, "Created",
                            {"Content-Type": "application/json",
                             "Location": f"/api/users/{new_id}"},
                            body)

    @server.route("DELETE", "/api/users/*")
    def delete_user(req):
        user_id = req.path.split("/")[-1]
        if user_id in server.data:
            del server.data[user_id]
            return HTTPResponse("HTTP/1.1", 204, "No Content", {}, "")
        return HTTPResponse("HTTP/1.1", 404, "Not Found",
                            {"Content-Type": "application/json"},
                            '{"error": "User not found"}')

    return server


# ── Demos ──────────────────────────────────────────────────────────────

def demo_request_building():
    print("=" * 60)
    print("HTTP REQUEST BUILDING")
    print("=" * 60)

    # GET request
    req = HTTPRequest(
        method="GET",
        path="/api/users",
        headers={
            "Host": "api.example.com",
            "Accept": "application/json",
            "User-Agent": "Python/Demo",
        },
    )
    print(f"\n  GET Request:")
    for line in req.build().split("\r\n"):
        print(f"    {line}")

    # POST request
    import json
    body = json.dumps({"name": "Charlie", "email": "charlie@example.com"})
    req = HTTPRequest(
        method="POST",
        path="/api/users",
        headers={
            "Host": "api.example.com",
            "Content-Type": "application/json",
        },
        body=body,
    )
    print(f"\n  POST Request:")
    for line in req.build().split("\r\n"):
        print(f"    {line}")


def demo_response_parsing():
    print("\n" + "=" * 60)
    print("HTTP RESPONSE PARSING")
    print("=" * 60)

    raw = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: 47\r\n"
        "Cache-Control: max-age=3600\r\n"
        "X-Request-Id: abc-123\r\n"
        "\r\n"
        '{"id": "1", "name": "Alice", "status": "active"}'
    )

    print(f"\n  Raw response:")
    for line in raw.split("\r\n"):
        print(f"    {line}")

    resp = HTTPResponse.parse(raw)
    print(f"\n  Parsed:")
    print(f"    Version:     {resp.version}")
    print(f"    Status:      {resp.status_code} {resp.reason}")
    print(f"    Headers:     {dict(resp.headers)}")
    print(f"    Body:        {resp.body}")


def demo_crud_operations():
    print("\n" + "=" * 60)
    print("HTTP CRUD OPERATIONS")
    print("=" * 60)

    server = create_demo_server()
    import json

    # GET all users
    print(f"\n  1. GET /api/users")
    req = HTTPRequest("GET", "/api/users", {"Host": "localhost"})
    resp = server.handle(req)
    print(f"     → {resp.status_code} {resp.reason}")
    print(f"     Body: {resp.body[:80]}...")

    # GET single user
    print(f"\n  2. GET /api/users/1")
    req = HTTPRequest("GET", "/api/users/1", {"Host": "localhost"})
    resp = server.handle(req)
    print(f"     → {resp.status_code} {resp.reason}")
    print(f"     Body: {resp.body}")

    # POST new user
    print(f"\n  3. POST /api/users")
    body = json.dumps({"name": "Charlie", "email": "charlie@test.com"})
    req = HTTPRequest("POST", "/api/users",
                      {"Host": "localhost", "Content-Type": "application/json"},
                      body=body)
    resp = server.handle(req)
    print(f"     → {resp.status_code} {resp.reason}")
    print(f"     Location: {resp.headers.get('Location')}")
    print(f"     Body: {resp.body}")

    # DELETE user
    print(f"\n  4. DELETE /api/users/2")
    req = HTTPRequest("DELETE", "/api/users/2", {"Host": "localhost"})
    resp = server.handle(req)
    print(f"     → {resp.status_code} {resp.reason}")

    # Verify deletion
    print(f"\n  5. GET /api/users/2 (after delete)")
    req = HTTPRequest("GET", "/api/users/2", {"Host": "localhost"})
    resp = server.handle(req)
    print(f"     → {resp.status_code} {resp.reason}")


def demo_status_codes():
    print("\n" + "=" * 60)
    print("HTTP STATUS CODE REFERENCE")
    print("=" * 60)

    categories = {
        "1xx Informational": [
            (100, "Continue"),
            (101, "Switching Protocols"),
        ],
        "2xx Success": [
            (200, "OK — Standard success"),
            (201, "Created — Resource created (POST)"),
            (204, "No Content — Success, empty body (DELETE)"),
        ],
        "3xx Redirection": [
            (301, "Moved Permanently"),
            (302, "Found (temporary redirect)"),
            (304, "Not Modified (cache valid)"),
        ],
        "4xx Client Error": [
            (400, "Bad Request — Malformed syntax"),
            (401, "Unauthorized — Authentication required"),
            (403, "Forbidden — Authenticated but not allowed"),
            (404, "Not Found"),
            (429, "Too Many Requests — Rate limited"),
        ],
        "5xx Server Error": [
            (500, "Internal Server Error"),
            (502, "Bad Gateway"),
            (503, "Service Unavailable"),
            (504, "Gateway Timeout"),
        ],
    }

    for category, codes in categories.items():
        print(f"\n  {category}:")
        for code, desc in codes:
            print(f"    {code}  {desc}")


if __name__ == "__main__":
    demo_request_building()
    demo_response_parsing()
    demo_crud_operations()
    demo_status_codes()
