# 14. gRPC와 Protocol Buffers(gRPC and Protocol Buffers)

**이전**: [API Gateway 패턴](./13_API_Gateway_Patterns.md) | **다음**: [API 보안](./15_API_Security.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- 서비스, 메시지, 열거형을 위한 Protocol Buffer 정의(proto3)를 작성할 수 있다
- grpcio 라이브러리를 사용하여 Python으로 gRPC 서비스와 클라이언트를 구현할 수 있다
- gRPC의 네 가지 스트리밍 패턴(단항, 서버 스트리밍, 클라이언트 스트리밍, 양방향)을 적용할 수 있다
- 다양한 사용 사례에 대해 gRPC와 REST를 비교하고 적절한 프로토콜을 선택할 수 있다
- gRPC-Web을 사용하여 브라우저 기반 애플리케이션에서 gRPC 서비스를 호출할 수 있다
- gRPC 상태 코드와 풍부한 오류 상세 정보를 사용하여 적절한 오류 처리를 구현할 수 있다

---

## 목차

1. [왜 gRPC인가?](#1-왜-grpc인가)
2. [Protocol Buffers (proto3)](#2-protocol-buffers-proto3)
3. [서비스 정의](#3-서비스-정의)
4. [Python gRPC 구현](#4-python-grpc-구현)
5. [스트리밍 패턴](#5-스트리밍-패턴)
6. [gRPC vs. REST](#6-grpc-vs-rest)
7. [gRPC-Web](#7-grpc-web)
8. [오류 처리](#8-오류-처리)
9. [연습 문제](#9-연습-문제)
10. [참고 자료](#10-참고-자료)

---

## 1. 왜 gRPC인가?

gRPC(gRPC Remote Procedure Calls)는 Google이 만든 고성능 RPC 프레임워크입니다. 전송에 HTTP/2를, 직렬화에 Protocol Buffers를 사용하며, 스트리밍, 인증, 로드 밸런싱에 대한 내장 지원을 제공합니다.

### 주요 장점

| 기능 | gRPC | REST/JSON |
|------|------|-----------|
| 직렬화 | Protocol Buffers (바이너리, 압축) | JSON (텍스트, 장황) |
| 전송 | HTTP/2 (다중화, 바이너리) | HTTP/1.1 또는 HTTP/2 |
| 스트리밍 | 내장 (4가지 패턴) | 제한적 (SSE, WebSocket) |
| 코드 생성 | 클라이언트와 서버 자동 생성 | 수동 또는 OpenAPI 코드 생성 |
| 타입 안전성 | 강타입 (컴파일된 스키마) | 느슨한 타입 (런타임 검증) |
| 성능 | 약 10배 빠른 직렬화 | 사람이 읽을 수 있지만 느림 |
| 브라우저 지원 | gRPC-Web 프록시를 통해 | 네이티브 |

### gRPC 사용 시기

- **마이크로서비스 간** 통신 (내부 API)
- 낮은 지연 시간이 필요한 **고처리량** 시스템
- **스트리밍** 데이터 (실시간 피드, 로그 스트리밍, 채팅)
- **다중 언어(Polyglot)** 환경 (10개 이상 언어에서 클라이언트 자동 생성)
- 제한된 네트워크의 **모바일 클라이언트** (압축된 바이너리 페이로드)

### REST를 대신 사용해야 할 때

- 서드파티 개발자가 사용하는 **공개 API**
- gRPC-Web 프록시 없는 **브라우저 우선** 애플리케이션
- REST 규약으로 충분한 **단순 CRUD** 작업
- **디버깅** 편의성 (사람이 읽을 수 있는 JSON, curl 친화적)

---

## 2. Protocol Buffers (proto3)

Protocol Buffers(protobuf)는 구조화된 데이터를 직렬화하기 위한 Google의 언어 중립적이고 플랫폼 중립적인 메커니즘입니다. Proto3는 현재 버전입니다.

### 기본 메시지 타입

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

### 스칼라 타입

| Proto 타입 | Python 타입 | 설명 |
|-----------|-------------|------|
| `double` | `float` | 64비트 부동소수점 |
| `float` | `float` | 32비트 부동소수점 |
| `int32` | `int` | 부호 있는 32비트 정수 |
| `int64` | `int` | 부호 있는 64비트 정수 |
| `uint32` | `int` | 부호 없는 32비트 정수 |
| `bool` | `bool` | 불리언 |
| `string` | `str` | UTF-8 문자열 |
| `bytes` | `bytes` | 임의 바이트 데이터 |

### 필드 번호

필드 번호는 영구적인 식별자입니다. 바이너리 형식에 인코딩되므로 변경하면 파괴적인 변경(breaking change)이 됩니다:

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

### Oneof 필드

여러 필드 중 하나만 설정해야 하는 경우:

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

### Map

```protobuf
message BookMetadata {
  // Key-value pairs
  map<string, string> attributes = 1;
  map<string, int32> chapter_pages = 2;
}
```

---

## 3. 서비스 정의

gRPC 서비스는 메시지와 함께 `.proto` 파일에 정의됩니다. 서비스 정의는 요청과 응답 타입을 포함한 RPC 메서드를 지정합니다.

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

### 코드 생성

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

이 명령은 다음을 생성합니다:
- `bookstore_pb2.py` -- 메시지 클래스 (Book, Genre 등)
- `bookstore_service_pb2.py` -- 서비스 요청/응답 메시지
- `bookstore_service_pb2_grpc.py` -- 서버 및 클라이언트 스텁

---

## 4. Python gRPC 구현

### 서버

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

### 클라이언트

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

### 비동기 클라이언트 (grpcio)

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

## 5. 스트리밍 패턴

gRPC는 네 가지 통신 패턴을 지원합니다:

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

### 서버 스트리밍

서버가 단일 클라이언트 요청에 대해 응답 스트림을 전송합니다. 실시간 피드, 대용량 데이터셋 다운로드, 진행 상황 업데이트에 유용합니다.

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

### 클라이언트 스트리밍

클라이언트가 요청 스트림을 전송하고 단일 응답을 수신합니다. 배치 업로드, 파일 업로드, 집계 쿼리에 유용합니다.

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

### 양방향 스트리밍(Bidirectional Streaming)

클라이언트와 서버가 동시에 스트림을 전송합니다. 채팅, 협업 편집, 실시간 번역에 유용합니다.

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

### 성능 비교

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

### 기능 비교

| 기능 | REST/JSON | gRPC/Protobuf |
|------|-----------|---------------|
| 사람이 읽을 수 있음 | 예 | 아니오 (바이너리) |
| 브라우저 네이티브 | 예 | gRPC-Web 프록시를 통해 |
| 스키마 강제 | 선택 (OpenAPI) | 필수 (proto) |
| 스트리밍 | 제한적 (SSE, WS) | 내장 (4가지 패턴) |
| 코드 생성 | 선택적 | 내장 |
| 캐싱 | HTTP 캐싱 (ETags, CDN) | 표준 캐싱 없음 |
| 도구 (curl, Postman) | 우수 | 제한적 (grpcurl, Postman) |
| 하위 호환성 | 버전 관리를 통해 | 필드 번호 안정성 |
| 오류 모델 | HTTP 상태 코드 | gRPC 상태 코드 + 상세 정보 |

### 의사결정 매트릭스

| 사용 사례 | 권장 |
|----------|------|
| 공개 개발자 API | REST |
| 내부 마이크로서비스 | gRPC |
| 실시간 스트리밍 | gRPC |
| 브라우저 SPA | REST (또는 gRPC-Web) |
| 모바일 앱 | gRPC (더 작은 페이로드) |
| 단순 CRUD | REST |
| 고처리량 파이프라인 | gRPC |

---

## 7. gRPC-Web

브라우저는 gRPC를 네이티브로 사용할 수 없습니다 (HTTP/2 트레일러 지원이 부족). gRPC-Web은 Envoy 프록시를 통해 gRPC-Web과 gRPC 사이를 변환하는 브라우저 호환 프로토콜을 제공하여 이 문제를 해결합니다.

### 아키텍처

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

### Envoy 설정

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

### gRPC Gateway (REST <-> gRPC)

REST와 gRPC 모두 접근이 필요한 API의 경우, gRPC-Gateway를 사용하여 proto 정의에서 REST 엔드포인트를 자동 생성할 수 있습니다:

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

이렇게 하면 REST 호출을 gRPC로 변환하는 리버스 프록시가 생성됩니다:

```
REST Client:  GET /v1/books/42   →  [Gateway]  →  GetBook(id=42)
gRPC Client:  GetBook(id=42)     →  [Server]   →  Book response
```

---

## 8. 오류 처리

gRPC는 HTTP 상태 코드와 구별되는 자체 상태 코드 세트를 사용합니다.

### gRPC 상태 코드

| 코드 | 이름 | HTTP 대응 | 사용 시기 |
|------|------|----------|----------|
| 0 | OK | 200 | 성공 |
| 1 | CANCELLED | 499 | 클라이언트가 요청을 취소 |
| 2 | UNKNOWN | 500 | 알 수 없는 오류 |
| 3 | INVALID_ARGUMENT | 400 | 잘못된 요청 데이터 |
| 5 | NOT_FOUND | 404 | 리소스가 존재하지 않음 |
| 6 | ALREADY_EXISTS | 409 | 중복 리소스 |
| 7 | PERMISSION_DENIED | 403 | 권한 부족 |
| 8 | RESOURCE_EXHAUSTED | 429 | 속도 제한 초과 |
| 12 | UNIMPLEMENTED | 501 | 메서드가 구현되지 않음 |
| 13 | INTERNAL | 500 | 내부 서버 오류 |
| 14 | UNAVAILABLE | 503 | 서비스 일시적으로 사용 불가 |
| 16 | UNAUTHENTICATED | 401 | 인증 정보 누락 또는 유효하지 않음 |

### 기본 오류 처리

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

### 풍부한 오류 상세 정보(Rich Error Details)

구조화된 오류 정보를 위해 상세 어노테이션이 포함된 `google.rpc.Status` 메시지를 사용합니다:

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

### 인터셉터 (gRPC의 미들웨어)

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

## 9. 연습 문제

### 문제 1: Proto 스키마 정의

**태스크 관리** 시스템을 위한 완전한 `.proto` 파일을 작성하세요:

- `Task` 메시지: id, title, description, status (열거형: TODO, IN_PROGRESS, DONE), priority (열거형: LOW, MEDIUM, HIGH, CRITICAL), assignee_id, created_at, updated_at, tags (repeated)
- `TaskService`: CreateTask, GetTask, ListTasks (status와 assignee로 필터링), UpdateTask, DeleteTask
- 열거형, repeated 필드, 타임스탬프의 적절한 사용
- 제거된 "due_date" 필드에 대한 reserved 필드

### 문제 2: gRPC 서비스 구현

문제 1의 proto를 사용하여 완전한 TaskService를 Python으로 구현하세요:

- 인메모리 저장소를 사용한 서버
- 적절한 오류 처리 (NOT_FOUND, INVALID_ARGUMENT, ALREADY_EXISTS)를 포함한 다섯 가지 CRUD 연산 모두
- 모든 연산을 시연하는 클라이언트 스크립트
- 각 RPC 메서드에 대한 유닛 테스트

### 문제 3: 서버 스트리밍 진행 상황

TaskService에 `WatchTaskUpdates` 서버 스트리밍 RPC를 추가하세요:

- 선택적 status 필터가 포함된 `WatchRequest` 수락
- `TaskEvent` 메시지 스트리밍 (event_type: CREATED, UPDATED, DELETED; task 데이터)
- 연결을 유지하기 위해 10초마다 하트비트 메시지 전송
- 클라이언트 연결 해제를 우아하게 처리

### 문제 4: 양방향 채팅

`TaskDiscussion` 양방향 스트리밍 RPC를 구현하세요:

- 클라이언트가 태스크에 대한 메시지를 전송 (task_id + 메시지 텍스트)
- 서버가 해당 태스크를 관찰 중인 모든 연결된 클라이언트에게 메시지를 브로드캐스트
- 간단한 인메모리 pub/sub 메커니즘 구현
- 여러 동시 클라이언트 연결 처리

### 문제 5: REST-to-gRPC 프록시

gRPC에 대한 REST-to-gRPC 프록시 역할을 하는 FastAPI 애플리케이션을 구축하세요:

- `GET /api/books/{id}` -> `GetBook` gRPC 메서드 호출
- `POST /api/books` -> `CreateBook` gRPC 메서드 호출
- `GET /api/books` -> `ListBooks` gRPC 메서드 호출
- gRPC 상태 코드를 HTTP 상태 코드로 매핑
- protobuf 메시지를 JSON 응답으로 매핑
- 적절한 오류 변환 포함 (gRPC INVALID_ARGUMENT -> HTTP 400 등)

---

## 10. 참고 자료

- [gRPC Official Documentation](https://grpc.io/docs/)
- [Protocol Buffers Language Guide (proto3)](https://protobuf.dev/programming-guides/proto3/)
- [gRPC Python Quickstart](https://grpc.io/docs/languages/python/quickstart/)
- [gRPC-Web Documentation](https://grpc.io/docs/platforms/web/)
- [gRPC Gateway](https://grpc-ecosystem.github.io/grpc-gateway/)
- [gRPC Status Codes](https://grpc.github.io/grpc/core/md_doc_statuscodes.html)
- [Envoy Proxy](https://www.envoyproxy.io/docs/envoy/latest/)
- [Google API Design Guide (gRPC)](https://cloud.google.com/apis/design)

---

**이전**: [API Gateway 패턴](./13_API_Gateway_Patterns.md) | [개요](./00_Overview.md) | **다음**: [API 보안](./15_API_Security.md)

**License**: CC BY-NC 4.0
