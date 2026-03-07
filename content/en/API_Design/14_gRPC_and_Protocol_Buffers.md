# 14. gRPC and Protocol Buffers

**Previous**: [API Gateway Patterns](./13_API_Gateway_Patterns.md) | **Next**: [API Security](./15_API_Security.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Write Protocol Buffer definitions (proto3) for services, messages, and enums
- Implement gRPC services and clients in Python using the grpcio library
- Apply the four gRPC streaming patterns: unary, server streaming, client streaming, and bidirectional
- Compare gRPC and REST for different use cases and select the appropriate protocol
- Use gRPC-Web to call gRPC services from browser-based applications
- Implement proper error handling with gRPC status codes and rich error details

---

## Table of Contents

1. [Why gRPC?](#1-why-grpc)
2. [Protocol Buffers (proto3)](#2-protocol-buffers-proto3)
3. [Service Definition](#3-service-definition)
4. [Python gRPC Implementation](#4-python-grpc-implementation)
5. [Streaming Patterns](#5-streaming-patterns)
6. [gRPC vs. REST](#6-grpc-vs-rest)
7. [gRPC-Web](#7-grpc-web)
8. [Error Handling](#8-error-handling)
9. [Exercises](#9-exercises)
10. [References](#10-references)

---

## 1. Why gRPC?

gRPC (gRPC Remote Procedure Calls) is a high-performance RPC framework created by Google. It uses HTTP/2 for transport, Protocol Buffers for serialization, and provides built-in support for streaming, authentication, and load balancing.

### Key Advantages

| Feature | gRPC | REST/JSON |
|---------|------|-----------|
| Serialization | Protocol Buffers (binary, compact) | JSON (text, verbose) |
| Transport | HTTP/2 (multiplexed, binary) | HTTP/1.1 or HTTP/2 |
| Streaming | Built-in (4 patterns) | Limited (SSE, WebSocket) |
| Code generation | Auto-generated clients and servers | Manual or OpenAPI codegen |
| Type safety | Strongly typed (compiled schema) | Loosely typed (runtime validation) |
| Performance | ~10x faster serialization | Human-readable but slower |
| Browser support | Via gRPC-Web proxy | Native |

### When to Use gRPC

- **Microservice-to-microservice** communication (internal APIs)
- **High-throughput** systems requiring low latency
- **Streaming** data (real-time feeds, log streaming, chat)
- **Polyglot** environments (auto-generate clients in 10+ languages)
- **Mobile clients** on constrained networks (compact binary payloads)

### When to Use REST Instead

- **Public APIs** consumed by third-party developers
- **Browser-first** applications without a gRPC-Web proxy
- **Simple CRUD** operations where REST conventions suffice
- **Debugging** convenience (human-readable JSON, curl-friendly)

---

## 2. Protocol Buffers (proto3)

Protocol Buffers (protobuf) is Google's language-neutral, platform-neutral mechanism for serializing structured data. Proto3 is the current version.

### Basic Message Types

```protobuf
// bookstore.proto
syntax = "proto3";

package bookstore;

// Import well-known types for timestamps and wrappers
import "google/protobuf/timestamp.proto";
import "google/protobuf/wrappers.proto";

// A book in the catalog
message Book {
  int64 id = 1;                    // Field number (not value)
  string title = 2;
  string isbn = 3;
  double price = 4;
  Genre genre = 5;
  int64 author_id = 6;
  google.protobuf.Timestamp created_at = 7;

  // Nested message for edition info
  Edition edition = 8;

  // Repeated field = list
  repeated string tags = 9;
}

// Nested message
message Edition {
  int32 number = 1;
  string publisher = 2;
  int32 year = 3;
}

// Enum type
enum Genre {
  GENRE_UNSPECIFIED = 0;  // proto3 requires 0 as default
  GENRE_FICTION = 1;
  GENRE_NON_FICTION = 2;
  GENRE_SCIENCE = 3;
  GENRE_HISTORY = 4;
  GENRE_BIOGRAPHY = 5;
}
```

### Scalar Types

| Proto Type | Python Type | Description |
|-----------|-------------|-------------|
| `double` | `float` | 64-bit floating point |
| `float` | `float` | 32-bit floating point |
| `int32` | `int` | Signed 32-bit integer |
| `int64` | `int` | Signed 64-bit integer |
| `uint32` | `int` | Unsigned 32-bit integer |
| `bool` | `bool` | Boolean |
| `string` | `str` | UTF-8 string |
| `bytes` | `bytes` | Arbitrary byte data |

### Field Numbers

Field numbers are permanent identifiers. They are encoded in the binary format, so changing them is a breaking change:

```protobuf
message User {
  int64 id = 1;         // Field number 1
  string name = 2;      // Field number 2
  string email = 3;     // Field number 3

  // Field 4 was removed (was: string phone)
  // NEVER reuse field number 4 — old data would be misinterpreted
  reserved 4;
  reserved "phone";

  string address = 5;   // Field number 5
}
```

### Oneof Fields

When only one of several fields should be set:

```protobuf
message SearchRequest {
  // Search by exactly one criterion
  oneof query {
    string title = 1;
    string isbn = 2;
    int64 author_id = 3;
  }

  int32 limit = 10;
  int32 offset = 11;
}
```

### Maps

```protobuf
message BookMetadata {
  // Key-value pairs
  map<string, string> attributes = 1;
  map<string, int32> chapter_pages = 2;
}
```

---

## 3. Service Definition

gRPC services are defined in `.proto` files alongside messages. The service definition specifies the RPC methods, including their request and response types.

```protobuf
// bookstore_service.proto
syntax = "proto3";

package bookstore;

import "bookstore.proto";
import "google/protobuf/empty.proto";

// Request/Response messages for each RPC
message GetBookRequest {
  int64 id = 1;
}

message ListBooksRequest {
  int32 limit = 1;
  int32 offset = 2;
  Genre genre_filter = 3;
  string sort_by = 4;
}

message ListBooksResponse {
  repeated Book books = 1;
  int32 total = 2;
}

message CreateBookRequest {
  string title = 1;
  string isbn = 2;
  double price = 3;
  Genre genre = 4;
  int64 author_id = 5;
  repeated string tags = 6;
}

message UpdateBookRequest {
  int64 id = 1;
  // Use wrapper types for optional fields (distinguish "not set" from "zero")
  google.protobuf.StringValue title = 2;
  google.protobuf.DoubleValue price = 3;
  Genre genre = 4;
}

message DeleteBookRequest {
  int64 id = 1;
}

// The BookService definition
service BookService {
  // Unary RPCs (request → response)
  rpc GetBook(GetBookRequest) returns (Book);
  rpc ListBooks(ListBooksRequest) returns (ListBooksResponse);
  rpc CreateBook(CreateBookRequest) returns (Book);
  rpc UpdateBook(UpdateBookRequest) returns (Book);
  rpc DeleteBook(DeleteBookRequest) returns (google.protobuf.Empty);

  // Server streaming: one request, stream of responses
  rpc WatchBooks(ListBooksRequest) returns (stream Book);

  // Client streaming: stream of requests, one response
  rpc BatchCreateBooks(stream CreateBookRequest) returns (ListBooksResponse);

  // Bidirectional streaming
  rpc BookChat(stream BookQuery) returns (stream BookRecommendation);
}

message BookQuery {
  string question = 1;
}

message BookRecommendation {
  Book book = 1;
  string reason = 2;
}
```

### Code Generation

```bash
# Install the protobuf compiler and Python plugins
pip install grpcio grpcio-tools

# Generate Python code from proto files
python -m grpc_tools.protoc \
    --proto_path=./protos \
    --python_out=./generated \
    --grpc_python_out=./generated \
    ./protos/bookstore.proto \
    ./protos/bookstore_service.proto
```

This generates:
- `bookstore_pb2.py` — Message classes (Book, Genre, etc.)
- `bookstore_service_pb2.py` — Service request/response messages
- `bookstore_service_pb2_grpc.py` — Server and client stubs

---

## 4. Python gRPC Implementation

### Server

```python
# server.py
import grpc
from concurrent import futures
import logging

from generated import bookstore_pb2
from generated import bookstore_service_pb2
from generated import bookstore_service_pb2_grpc

logger = logging.getLogger(__name__)


class BookServicer(bookstore_service_pb2_grpc.BookServiceServicer):
    """Implementation of the BookService gRPC service."""

    def __init__(self):
        # In-memory store for demonstration
        self.books: dict[int, bookstore_pb2.Book] = {}
        self.next_id = 1

    def GetBook(self, request, context):
        """Unary RPC: Get a single book by ID."""
        book = self.books.get(request.id)
        if book is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Book with ID {request.id} not found")
            return bookstore_pb2.Book()
        return book

    def ListBooks(self, request, context):
        """Unary RPC: List books with pagination and filtering."""
        books = list(self.books.values())

        # Apply genre filter
        if request.genre_filter != bookstore_pb2.GENRE_UNSPECIFIED:
            books = [b for b in books if b.genre == request.genre_filter]

        total = len(books)

        # Apply pagination
        offset = request.offset or 0
        limit = request.limit or 20
        books = books[offset:offset + limit]

        return bookstore_service_pb2.ListBooksResponse(
            books=books,
            total=total,
        )

    def CreateBook(self, request, context):
        """Unary RPC: Create a new book."""
        # Validate required fields
        if not request.title:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Title is required")
            return bookstore_pb2.Book()

        if not request.isbn:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("ISBN is required")
            return bookstore_pb2.Book()

        # Check for duplicate ISBN
        for book in self.books.values():
            if book.isbn == request.isbn:
                context.set_code(grpc.StatusCode.ALREADY_EXISTS)
                context.set_details(f"Book with ISBN {request.isbn} already exists")
                return bookstore_pb2.Book()

        book = bookstore_pb2.Book(
            id=self.next_id,
            title=request.title,
            isbn=request.isbn,
            price=request.price,
            genre=request.genre,
            author_id=request.author_id,
            tags=request.tags,
        )
        self.books[self.next_id] = book
        self.next_id += 1

        logger.info(f"Created book: {book.title} (ID: {book.id})")
        return book

    def DeleteBook(self, request, context):
        """Unary RPC: Delete a book."""
        if request.id not in self.books:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Book with ID {request.id} not found")
            return bookstore_pb2.Book()

        del self.books[request.id]
        from google.protobuf.empty_pb2 import Empty
        return Empty()


def serve():
    """Start the gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ("grpc.max_send_message_length", 50 * 1024 * 1024),  # 50MB
            ("grpc.max_receive_message_length", 50 * 1024 * 1024),
        ],
    )

    bookstore_service_pb2_grpc.add_BookServiceServicer_to_server(
        BookServicer(), server
    )

    server.add_insecure_port("[::]:50051")
    server.start()
    logger.info("gRPC server started on port 50051")

    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    serve()
```

### Client

```python
# client.py
import grpc
from generated import bookstore_pb2
from generated import bookstore_service_pb2
from generated import bookstore_service_pb2_grpc


def run():
    """Demonstrate gRPC client operations."""
    # Create a channel to the server
    channel = grpc.insecure_channel("localhost:50051")
    stub = bookstore_service_pb2_grpc.BookServiceStub(channel)

    # Create a book
    print("--- Creating a book ---")
    book = stub.CreateBook(
        bookstore_service_pb2.CreateBookRequest(
            title="API Design Patterns",
            isbn="978-1617295850",
            price=49.99,
            genre=bookstore_pb2.GENRE_NON_FICTION,
            author_id=42,
            tags=["api", "design", "patterns"],
        )
    )
    print(f"Created: {book.title} (ID: {book.id})")

    # Get the book
    print("\n--- Getting book by ID ---")
    retrieved = stub.GetBook(
        bookstore_service_pb2.GetBookRequest(id=book.id)
    )
    print(f"Retrieved: {retrieved.title}, ${retrieved.price}")

    # List all books
    print("\n--- Listing all books ---")
    response = stub.ListBooks(
        bookstore_service_pb2.ListBooksRequest(limit=10)
    )
    for b in response.books:
        print(f"  - {b.title} ({b.isbn})")
    print(f"Total: {response.total}")

    # Handle errors
    print("\n--- Handling not found ---")
    try:
        stub.GetBook(bookstore_service_pb2.GetBookRequest(id=99999))
    except grpc.RpcError as e:
        print(f"Error: {e.code().name} - {e.details()}")

    channel.close()


if __name__ == "__main__":
    run()
```

### Async Client with grpcio

```python
# async_client.py
import asyncio
import grpc.aio
from generated import bookstore_pb2
from generated import bookstore_service_pb2
from generated import bookstore_service_pb2_grpc


async def run():
    """Async gRPC client using grpc.aio."""
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = bookstore_service_pb2_grpc.BookServiceStub(channel)

        # Unary call
        book = await stub.CreateBook(
            bookstore_service_pb2.CreateBookRequest(
                title="Async Book",
                isbn="978-0000000001",
                price=29.99,
                genre=bookstore_pb2.GENRE_SCIENCE,
            )
        )
        print(f"Created: {book.title}")

        # Multiple concurrent calls
        tasks = [
            stub.GetBook(bookstore_service_pb2.GetBookRequest(id=1)),
            stub.ListBooks(bookstore_service_pb2.ListBooksRequest(limit=5)),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, grpc.RpcError):
                print(f"Error: {result.code()}")
            else:
                print(f"Success: {type(result).__name__}")


if __name__ == "__main__":
    asyncio.run(run())
```

---

## 5. Streaming Patterns

gRPC supports four communication patterns:

```
1. Unary:              Client ──request──→ Server
                       Client ←─response── Server

2. Server Streaming:   Client ──request──→ Server
                       Client ←─response── Server
                       Client ←─response── Server
                       Client ←─response── Server

3. Client Streaming:   Client ──request──→ Server
                       Client ──request──→ Server
                       Client ──request──→ Server
                       Client ←─response── Server

4. Bidirectional:      Client ←→ Server (both stream simultaneously)
```

### Server Streaming

The server sends a stream of responses to a single client request. Useful for: real-time feeds, large dataset downloads, progress updates.

```protobuf
// In the service definition
rpc WatchBooks(ListBooksRequest) returns (stream Book);
```

```python
# Server implementation
class BookServicer(bookstore_service_pb2_grpc.BookServiceServicer):

    def WatchBooks(self, request, context):
        """Server streaming: send new books as they are created.

        The client receives a stream of Book messages in real-time.
        """
        last_id = 0
        while context.is_active():
            # Check for new books since last check
            new_books = [
                b for id, b in self.books.items()
                if id > last_id
            ]

            for book in new_books:
                # Apply genre filter if specified
                if (request.genre_filter != bookstore_pb2.GENRE_UNSPECIFIED
                        and book.genre != request.genre_filter):
                    continue
                yield book
                last_id = max(last_id, book.id)

            time.sleep(1)  # Poll interval


# Client usage
def watch_books():
    channel = grpc.insecure_channel("localhost:50051")
    stub = bookstore_service_pb2_grpc.BookServiceStub(channel)

    print("Watching for new books...")
    for book in stub.WatchBooks(
        bookstore_service_pb2.ListBooksRequest(
            genre_filter=bookstore_pb2.GENRE_SCIENCE
        )
    ):
        print(f"New book: {book.title} (${book.price})")
```

### Client Streaming

The client sends a stream of requests and receives a single response. Useful for: batch uploads, file uploads, aggregation queries.

```protobuf
rpc BatchCreateBooks(stream CreateBookRequest) returns (ListBooksResponse);
```

```python
# Server implementation
class BookServicer(bookstore_service_pb2_grpc.BookServiceServicer):

    def BatchCreateBooks(self, request_iterator, context):
        """Client streaming: receive a batch of books to create.

        Returns a summary of all created books.
        """
        created_books = []

        for request in request_iterator:
            book = bookstore_pb2.Book(
                id=self.next_id,
                title=request.title,
                isbn=request.isbn,
                price=request.price,
                genre=request.genre,
            )
            self.books[self.next_id] = book
            created_books.append(book)
            self.next_id += 1

        return bookstore_service_pb2.ListBooksResponse(
            books=created_books,
            total=len(created_books),
        )


# Client usage
def batch_create():
    channel = grpc.insecure_channel("localhost:50051")
    stub = bookstore_service_pb2_grpc.BookServiceStub(channel)

    def book_generator():
        """Generate a stream of book creation requests."""
        books_to_create = [
            ("Python Crash Course", "978-1593279288", 35.99),
            ("Clean Code", "978-0132350884", 39.99),
            ("DDIA", "978-1449373320", 45.99),
        ]
        for title, isbn, price in books_to_create:
            yield bookstore_service_pb2.CreateBookRequest(
                title=title,
                isbn=isbn,
                price=price,
                genre=bookstore_pb2.GENRE_NON_FICTION,
            )

    response = stub.BatchCreateBooks(book_generator())
    print(f"Created {response.total} books")
    for book in response.books:
        print(f"  - {book.title} (ID: {book.id})")
```

### Bidirectional Streaming

Both client and server send streams simultaneously. Useful for: chat, collaborative editing, real-time translation.

```protobuf
rpc BookChat(stream BookQuery) returns (stream BookRecommendation);
```

```python
# Server implementation
class BookServicer(bookstore_service_pb2_grpc.BookServiceServicer):

    def BookChat(self, request_iterator, context):
        """Bidirectional streaming: interactive book recommendation.

        The client sends questions and the server streams back
        book recommendations in real-time.
        """
        for query in request_iterator:
            # Find books matching the query
            matching = [
                b for b in self.books.values()
                if query.question.lower() in b.title.lower()
                or query.question.lower() in [t.lower() for t in b.tags]
            ]

            for book in matching[:3]:
                yield bookstore_service_pb2.BookRecommendation(
                    book=book,
                    reason=f"Matches your query: '{query.question}'",
                )

            if not matching:
                yield bookstore_service_pb2.BookRecommendation(
                    book=bookstore_pb2.Book(title="No matches found"),
                    reason=f"No books match: '{query.question}'",
                )


# Client usage
async def book_chat():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = bookstore_service_pb2_grpc.BookServiceStub(channel)

        async def query_generator():
            queries = ["python", "design patterns", "machine learning"]
            for q in queries:
                yield bookstore_service_pb2.BookQuery(question=q)
                await asyncio.sleep(1)  # Simulate typing delay

        async for recommendation in stub.BookChat(query_generator()):
            print(f"Recommendation: {recommendation.book.title}")
            print(f"  Reason: {recommendation.reason}")
```

---

## 6. gRPC vs. REST

### Performance Comparison

```
Payload size for the same data:
┌─────────────────────────────────────────────┐
│ JSON (REST):  {"id":1,"title":"API Design"} │  = 34 bytes
│ Protobuf:     [binary]                      │  = 18 bytes
└─────────────────────────────────────────────┘
~47% smaller payload with Protocol Buffers

Serialization speed (10,000 messages):
┌──────────────────────────────────┐
│ JSON encode:  12.3 ms            │
│ Protobuf encode:  1.8 ms         │
│                                   │
│ JSON decode:  15.7 ms            │
│ Protobuf decode:  2.1 ms         │
└──────────────────────────────────┘
~7x faster serialization with Protocol Buffers
```

### Feature Comparison

| Feature | REST/JSON | gRPC/Protobuf |
|---------|-----------|---------------|
| Human-readable | Yes | No (binary) |
| Browser-native | Yes | Via gRPC-Web proxy |
| Schema enforcement | Optional (OpenAPI) | Required (proto) |
| Streaming | Limited (SSE, WS) | Built-in (4 patterns) |
| Code generation | Optional | Built-in |
| Caching | HTTP caching (ETags, CDN) | No standard caching |
| Tools (curl, Postman) | Excellent | Limited (grpcurl, Postman) |
| Backward compatibility | Via versioning | Field number stability |
| Error model | HTTP status codes | gRPC status codes + details |

### Decision Matrix

| Use Case | Recommended |
|----------|-------------|
| Public developer API | REST |
| Internal microservice | gRPC |
| Real-time streaming | gRPC |
| Browser SPA | REST (or gRPC-Web) |
| Mobile app | gRPC (smaller payloads) |
| Simple CRUD | REST |
| High-throughput pipeline | gRPC |

---

## 7. gRPC-Web

Browsers cannot use gRPC natively (they lack HTTP/2 trailer support). gRPC-Web solves this by providing a browser-compatible protocol with an Envoy proxy to translate between gRPC-Web and gRPC.

### Architecture

```
Browser (gRPC-Web client)
    │
    │ HTTP/1.1 or HTTP/2 (gRPC-Web protocol)
    ▼
[Envoy Proxy]
    │
    │ HTTP/2 (native gRPC)
    ▼
[gRPC Server]
```

### Envoy Configuration

```yaml
# envoy.yaml
static_resources:
  listeners:
    - name: listener_0
      address:
        socket_address:
          address: 0.0.0.0
          port_value: 8080
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                codec_type: auto
                stat_prefix: ingress_http
                route_config:
                  name: local_route
                  virtual_hosts:
                    - name: local_service
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/"
                          route:
                            cluster: grpc_service
                      cors:
                        allow_origin_string_match:
                          - prefix: "*"
                        allow_methods: GET, PUT, DELETE, POST, OPTIONS
                        allow_headers: content-type, x-grpc-web
                        max_age: "1728000"
                http_filters:
                  - name: envoy.filters.http.grpc_web
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.grpc_web.v3.GrpcWeb
                  - name: envoy.filters.http.cors
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.cors.v3.Cors
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router

  clusters:
    - name: grpc_service
      connect_timeout: 0.25s
      type: logical_dns
      lb_policy: round_robin
      http2_protocol_options: {}
      load_assignment:
        cluster_name: grpc_service
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: grpc-server
                      port_value: 50051
```

### gRPC Gateway (REST ↔ gRPC)

For APIs that need both REST and gRPC access, use a gRPC-Gateway to auto-generate REST endpoints from proto definitions:

```protobuf
// With gRPC-Gateway annotations
import "google/api/annotations.proto";

service BookService {
  rpc GetBook(GetBookRequest) returns (Book) {
    option (google.api.http) = {
      get: "/v1/books/{id}"
    };
  }

  rpc CreateBook(CreateBookRequest) returns (Book) {
    option (google.api.http) = {
      post: "/v1/books"
      body: "*"
    };
  }

  rpc ListBooks(ListBooksRequest) returns (ListBooksResponse) {
    option (google.api.http) = {
      get: "/v1/books"
    };
  }
}
```

This generates a reverse proxy that translates REST calls to gRPC:

```
REST Client:  GET /v1/books/42   →  [Gateway]  →  GetBook(id=42)
gRPC Client:  GetBook(id=42)     →  [Server]   →  Book response
```

---

## 8. Error Handling

gRPC uses its own set of status codes, distinct from HTTP status codes.

### gRPC Status Codes

| Code | Name | HTTP Equivalent | When to Use |
|------|------|----------------|-------------|
| 0 | OK | 200 | Successful |
| 1 | CANCELLED | 499 | Client cancelled the request |
| 2 | UNKNOWN | 500 | Unknown error |
| 3 | INVALID_ARGUMENT | 400 | Bad request data |
| 5 | NOT_FOUND | 404 | Resource does not exist |
| 6 | ALREADY_EXISTS | 409 | Duplicate resource |
| 7 | PERMISSION_DENIED | 403 | Insufficient permissions |
| 8 | RESOURCE_EXHAUSTED | 429 | Rate limit exceeded |
| 12 | UNIMPLEMENTED | 501 | Method not implemented |
| 13 | INTERNAL | 500 | Internal server error |
| 14 | UNAVAILABLE | 503 | Service temporarily unavailable |
| 16 | UNAUTHENTICATED | 401 | Missing or invalid auth |

### Basic Error Handling

```python
# Server: setting error status
class BookServicer(bookstore_service_pb2_grpc.BookServiceServicer):

    def GetBook(self, request, context):
        if request.id <= 0:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Book ID must be positive")
            return bookstore_pb2.Book()

        book = self.books.get(request.id)
        if book is None:
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(f"Book {request.id} not found")
            return bookstore_pb2.Book()

        return book


# Client: handling errors
def get_book_safely(stub, book_id):
    try:
        book = stub.GetBook(
            bookstore_service_pb2.GetBookRequest(id=book_id)
        )
        return book
    except grpc.RpcError as e:
        status_code = e.code()
        details = e.details()

        if status_code == grpc.StatusCode.NOT_FOUND:
            print(f"Book not found: {details}")
        elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
            print(f"Invalid request: {details}")
        elif status_code == grpc.StatusCode.UNAVAILABLE:
            print(f"Service unavailable: {details}")
        else:
            print(f"RPC error: {status_code.name} - {details}")
        return None
```

### Rich Error Details

For structured error information, use the `google.rpc.Status` message with detail annotations:

```python
from google.rpc import status_pb2, error_details_pb2
from google.protobuf import any_pb2
from grpc_status import rpc_status


class BookServicer(bookstore_service_pb2_grpc.BookServiceServicer):

    def CreateBook(self, request, context):
        """Create a book with rich error details on validation failure."""
        errors = []

        if not request.title:
            errors.append(
                error_details_pb2.BadRequest.FieldViolation(
                    field="title",
                    description="Title is required",
                )
            )

        if not request.isbn:
            errors.append(
                error_details_pb2.BadRequest.FieldViolation(
                    field="isbn",
                    description="ISBN is required and must match format 978-XXXXXXXXXX",
                )
            )

        if request.price < 0:
            errors.append(
                error_details_pb2.BadRequest.FieldViolation(
                    field="price",
                    description="Price must be non-negative",
                )
            )

        if errors:
            # Build rich error status
            bad_request = error_details_pb2.BadRequest(
                field_violations=errors
            )

            detail = any_pb2.Any()
            detail.Pack(bad_request)

            rich_status = status_pb2.Status(
                code=grpc.StatusCode.INVALID_ARGUMENT.value[0],
                message=f"{len(errors)} validation error(s)",
                details=[detail],
            )

            context.abort_with_status(rpc_status.to_status(rich_status))

        # ... create the book ...
```

### Interceptors (Middleware for gRPC)

```python
class LoggingInterceptor(grpc.ServerInterceptor):
    """Log all RPC calls with method name, duration, and status."""

    def intercept_service(self, continuation, handler_call_details):
        method = handler_call_details.method
        start = time.perf_counter()

        # Call the actual handler
        response = continuation(handler_call_details)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(f"gRPC {method} completed in {elapsed_ms:.2f}ms")

        return response


# Add interceptor to server
server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=10),
    interceptors=[LoggingInterceptor()],
)
```

---

## 9. Exercises

### Exercise 1: Define a Proto Schema

Write a complete `.proto` file for a **Task Management** system with:

- `Task` message: id, title, description, status (enum: TODO, IN_PROGRESS, DONE), priority (enum: LOW, MEDIUM, HIGH, CRITICAL), assignee_id, created_at, updated_at, tags (repeated)
- `TaskService` with: CreateTask, GetTask, ListTasks (with filtering by status and assignee), UpdateTask, DeleteTask
- Proper use of enums, repeated fields, and timestamps
- Reserved fields for a removed "due_date" field

### Exercise 2: Implement a gRPC Service

Using your proto from Exercise 1, implement the full TaskService in Python:

- Server with in-memory storage
- All five CRUD operations with proper error handling (NOT_FOUND, INVALID_ARGUMENT, ALREADY_EXISTS)
- Client script that demonstrates all operations
- Unit tests for each RPC method

### Exercise 3: Server Streaming Progress

Add a `WatchTaskUpdates` server streaming RPC to the TaskService that:

- Accepts a `WatchRequest` with optional status filter
- Streams `TaskEvent` messages (event_type: CREATED, UPDATED, DELETED; task data)
- Sends a heartbeat message every 10 seconds to keep the connection alive
- Handles client disconnection gracefully

### Exercise 4: Bidirectional Chat

Implement a `TaskDiscussion` bidirectional streaming RPC where:

- The client sends messages about a task (task_id + message text)
- The server broadcasts messages to all connected clients watching that task
- Implement a simple in-memory pub/sub mechanism
- Handle multiple concurrent client connections

### Exercise 5: REST-to-gRPC Proxy

Build a FastAPI application that acts as a REST-to-gRPC proxy:

- `GET /api/books/{id}` → calls `GetBook` gRPC method
- `POST /api/books` → calls `CreateBook` gRPC method
- `GET /api/books` → calls `ListBooks` gRPC method
- Map gRPC status codes to HTTP status codes
- Map protobuf messages to JSON responses
- Include proper error translation (gRPC INVALID_ARGUMENT → HTTP 400, etc.)

---

## 10. References

- [gRPC Official Documentation](https://grpc.io/docs/)
- [Protocol Buffers Language Guide (proto3)](https://protobuf.dev/programming-guides/proto3/)
- [gRPC Python Quickstart](https://grpc.io/docs/languages/python/quickstart/)
- [gRPC-Web Documentation](https://grpc.io/docs/platforms/web/)
- [gRPC Gateway](https://grpc-ecosystem.github.io/grpc-gateway/)
- [gRPC Status Codes](https://grpc.github.io/grpc/core/md_doc_statuscodes.html)
- [Envoy Proxy](https://www.envoyproxy.io/docs/envoy/latest/)
- [Google API Design Guide (gRPC)](https://cloud.google.com/apis/design)

---

**Previous**: [API Gateway Patterns](./13_API_Gateway_Patterns.md) | [Overview](./00_Overview.md) | **Next**: [API Security](./15_API_Security.md)

**License**: CC BY-NC 4.0
