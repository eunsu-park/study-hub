#!/usr/bin/env python3
"""Example: gRPC Service

Demonstrates a gRPC service in Python with:
- Protocol Buffer (proto3) service definition (embedded as docstring)
- gRPC server implementation with servicer classes
- gRPC client usage
- Unary and server-streaming RPCs
- Error handling with gRPC status codes
- Interceptors for logging/auth

Related lesson: 01_API_Design_Fundamentals.md (RPC paradigm comparison)

Setup:
    pip install grpcio grpcio-tools

    # Generate Python code from the proto definition:
    # (The proto is embedded below for reference — save to product.proto first)
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. product.proto

    # Run server
    python 08_grpc_service.py --server

    # Run client
    python 08_grpc_service.py --client
"""

# =============================================================================
# PROTOCOL BUFFER DEFINITION (product.proto)
# =============================================================================
# This is the interface contract — equivalent to OpenAPI in REST.
# Save this as product.proto and compile with protoc to generate code.
#
# Advantages of Protocol Buffers over JSON:
# - Strongly typed (catches errors at compile time)
# - Compact binary format (3-10x smaller than JSON)
# - Backward/forward compatible with field numbering
# - Code generation for multiple languages

PROTO_DEFINITION = """
syntax = "proto3";

package product;

// ProductService provides CRUD operations for products.
// Each RPC method has a clear request/response message.
service ProductService {
  // Unary RPC: single request, single response
  rpc GetProduct(GetProductRequest) returns (ProductResponse);
  rpc CreateProduct(CreateProductRequest) returns (ProductResponse);

  // Server streaming: single request, stream of responses
  // Useful for large result sets, real-time feeds, or export
  rpc ListProducts(ListProductsRequest) returns (stream ProductResponse);

  // Unary RPC for search
  rpc SearchProducts(SearchRequest) returns (SearchResponse);
}

message GetProductRequest {
  string id = 1;  // Field number, not value — used for wire format
}

message CreateProductRequest {
  string name = 1;
  string description = 2;
  double price = 3;
  string category = 4;
  int32 stock = 5;
}

message ListProductsRequest {
  string category = 1;  // Optional filter
  int32 page_size = 2;  // Pagination
}

message SearchRequest {
  string query = 1;
  int32 max_results = 2;
}

message SearchResponse {
  repeated ProductResponse products = 1;
  int32 total_count = 2;
}

message ProductResponse {
  string id = 1;
  string name = 2;
  string description = 3;
  double price = 4;
  string category = 5;
  int32 stock = 6;
  string created_at = 7;
}
"""

import sys
import time
import logging
from concurrent import futures
from datetime import datetime, timezone
from uuid import uuid4

try:
    import grpc
    from grpc import aio
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    print("gRPC not installed. Run: pip install grpcio grpcio-tools")
    print("\nShowing proto definition and pseudocode only.\n")
    print(PROTO_DEFINITION)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SIMULATED GENERATED CODE
# =============================================================================
# Normally, protoc generates these classes. We simulate them here so the
# example runs without the compilation step. In production, always use
# generated code for type safety.

if GRPC_AVAILABLE:

    # --- Data Store ---
    products_db: dict[str, dict] = {}

    # Seed sample data
    for i in range(1, 21):
        pid = str(i)
        products_db[pid] = {
            "id": pid,
            "name": f"Product {i}",
            "description": f"Description for product {i}",
            "price": round(9.99 + i * 5.5, 2),
            "category": ["electronics", "books", "clothing", "food"][i % 4],
            "stock": i * 10,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # SERVER IMPLEMENTATION
    # =========================================================================
    # The servicer class implements the RPCs defined in the proto.
    # Each method receives a request object and a context for metadata/errors.

    class ProductServicer:
        """Implements the ProductService RPC methods.

        This is the server-side logic. gRPC handles serialization,
        transport, and connection management.
        """

        def GetProduct(self, request_id: str, context):
            """Unary RPC: Get a single product by ID.

            gRPC status codes map to HTTP-like semantics:
            - OK (0): Success
            - NOT_FOUND (5): Resource does not exist
            - INVALID_ARGUMENT (3): Bad request
            - INTERNAL (13): Server error
            """
            logger.info(f"GetProduct called with id={request_id}")

            product = products_db.get(request_id)
            if not product:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(f"Product {request_id} not found")
                return None

            return product

        def CreateProduct(self, request_data: dict, context):
            """Unary RPC: Create a new product."""
            logger.info(f"CreateProduct called: {request_data.get('name')}")

            # Validate
            if not request_data.get("name"):
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Product name is required")
                return None

            if request_data.get("price", 0) <= 0:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                context.set_details("Price must be positive")
                return None

            product_id = str(uuid4())
            product = {
                "id": product_id,
                "name": request_data["name"],
                "description": request_data.get("description", ""),
                "price": request_data["price"],
                "category": request_data.get("category", "uncategorized"),
                "stock": request_data.get("stock", 0),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            products_db[product_id] = product
            return product

        def ListProducts(self, request_data: dict, context):
            """Server-streaming RPC: Stream products one at a time.

            Streaming is ideal when:
            - The result set is large (avoid loading all into memory)
            - The client wants to process items as they arrive
            - Real-time data feeds (stock prices, sensor data)

            The yield keyword makes this a generator — gRPC streams each
            yielded item to the client as a separate message.
            """
            logger.info(f"ListProducts called, category={request_data.get('category')}")

            category = request_data.get("category")
            page_size = request_data.get("page_size", 10)
            count = 0

            for product in products_db.values():
                if category and product["category"] != category:
                    continue
                yield product
                count += 1
                if count >= page_size:
                    break

        def SearchProducts(self, query: str, max_results: int, context):
            """Unary RPC: Search products by name/description."""
            logger.info(f"SearchProducts called: query='{query}'")

            results = []
            for product in products_db.values():
                if query.lower() in product["name"].lower() or \
                   query.lower() in product["description"].lower():
                    results.append(product)
                    if len(results) >= max_results:
                        break

            return {"products": results, "total_count": len(results)}


    # =========================================================================
    # INTERCEPTOR — Cross-cutting concerns (logging, auth, metrics)
    # =========================================================================
    # Interceptors are gRPC's equivalent of middleware in REST frameworks.

    class LoggingInterceptor(grpc.ServerInterceptor):
        """Log all incoming RPC calls with timing information.

        Interceptors are chained — you can have auth, logging, and metrics
        interceptors running in sequence.
        """

        def intercept_service(self, continuation, handler_call_details):
            method = handler_call_details.method
            logger.info(f"[gRPC] Incoming call: {method}")
            start = time.monotonic()

            response = continuation(handler_call_details)

            elapsed = (time.monotonic() - start) * 1000
            logger.info(f"[gRPC] {method} completed in {elapsed:.1f}ms")
            return response


    class AuthInterceptor(grpc.ServerInterceptor):
        """Validate API key from gRPC metadata (headers).

        gRPC uses metadata for the same purpose as HTTP headers.
        The key-value pairs are sent alongside each RPC call.
        """

        VALID_API_KEYS = {"dev-key-123", "prod-key-456"}

        def intercept_service(self, continuation, handler_call_details):
            metadata = dict(handler_call_details.invocation_metadata or [])
            api_key = metadata.get("x-api-key")

            if api_key not in self.VALID_API_KEYS:
                logger.warning(f"[gRPC] Unauthorized: invalid API key")
                # In a real interceptor, you would abort the call here
                # For demo purposes, we just log the warning

            return continuation(handler_call_details)


    # =========================================================================
    # SERVER STARTUP
    # =========================================================================

    def serve():
        """Start the gRPC server.

        Key configuration:
        - ThreadPoolExecutor: handles concurrent RPCs
        - Interceptors: logging and auth (applied to all RPCs)
        - Insecure port: for development only; use TLS in production
        """
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=10),
            interceptors=[LoggingInterceptor(), AuthInterceptor()],
        )

        # In production, register the generated servicer:
        # product_pb2_grpc.add_ProductServiceServicer_to_server(
        #     ProductServicer(), server
        # )

        port = 50051
        server.add_insecure_port(f"[::]:{port}")
        server.start()
        logger.info(f"gRPC server started on port {port}")
        logger.info("Press Ctrl+C to stop")

        try:
            server.wait_for_termination()
        except KeyboardInterrupt:
            server.stop(grace=5)
            logger.info("Server stopped")


    # =========================================================================
    # CLIENT EXAMPLE
    # =========================================================================

    def run_client():
        """Demonstrate gRPC client usage.

        The client uses a channel (connection) and a stub (typed client).
        """
        logger.info("Starting gRPC client demo...")
        logger.info("(In production, connect to a running server)")
        logger.info("")

        # Demonstrate the client patterns (not actually connecting)
        print("=" * 60)
        print("gRPC Client Patterns")
        print("=" * 60)

        print("""
# 1. Create a channel (connection to the server)
channel = grpc.insecure_channel('localhost:50051')

# 2. Create a stub (typed client generated from proto)
stub = product_pb2_grpc.ProductServiceStub(channel)

# 3. Unary RPC — simple request/response
response = stub.GetProduct(
    product_pb2.GetProductRequest(id="1"),
    metadata=[("x-api-key", "dev-key-123")],
)
print(f"Product: {response.name}, Price: {response.price}")

# 4. Server streaming — iterate over streamed responses
for product in stub.ListProducts(
    product_pb2.ListProductsRequest(category="electronics", page_size=5)
):
    print(f"  - {product.name}: ${product.price}")

# 5. Error handling
try:
    response = stub.GetProduct(
        product_pb2.GetProductRequest(id="nonexistent")
    )
except grpc.RpcError as e:
    print(f"Error: {e.code()} - {e.details()}")
    # grpc.StatusCode.NOT_FOUND - Product nonexistent not found

# 6. Deadlines (timeouts)
try:
    response = stub.SearchProducts(
        product_pb2.SearchRequest(query="widget", max_results=10),
        timeout=5.0,  # 5-second deadline
    )
except grpc.RpcError as e:
    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
        print("Request timed out")

# 7. Channel lifecycle
channel.close()
# Or use context manager:
# with grpc.insecure_channel('localhost:50051') as channel:
#     stub = product_pb2_grpc.ProductServiceStub(channel)
#     ...
""")

        # Demonstrate local servicer without network
        print("=" * 60)
        print("Local Servicer Demo (no server needed)")
        print("=" * 60)

        servicer = ProductServicer()

        class FakeContext:
            def set_code(self, code): self._code = code
            def set_details(self, details): self._details = details

        ctx = FakeContext()

        # GetProduct
        product = servicer.GetProduct("1", ctx)
        if product:
            print(f"\nGetProduct(1): {product['name']} - ${product['price']}")

        # CreateProduct
        new_product = servicer.CreateProduct(
            {"name": "gRPC Widget", "price": 42.0, "category": "electronics"},
            ctx,
        )
        if new_product:
            print(f"CreateProduct: {new_product['name']} (id={new_product['id'][:8]}...)")

        # ListProducts (streaming)
        print("\nListProducts(category=electronics, page_size=3):")
        for p in servicer.ListProducts({"category": "electronics", "page_size": 3}, ctx):
            print(f"  - {p['name']}: ${p['price']}")

        # SearchProducts
        results = servicer.SearchProducts("Product 1", 5, ctx)
        print(f"\nSearchProducts('Product 1'): {results['total_count']} results")


# =============================================================================
# REST vs gRPC COMPARISON
# =============================================================================

COMPARISON = """
REST vs gRPC — When to Use Each
================================

| Feature         | REST (HTTP/JSON)      | gRPC (HTTP/2 + Protobuf) |
|-----------------|-----------------------|--------------------------|
| Protocol        | HTTP/1.1 or HTTP/2    | HTTP/2 (required)        |
| Serialization   | JSON (text)           | Protobuf (binary)        |
| Contract        | OpenAPI (optional)    | Proto file (required)    |
| Streaming       | SSE, WebSocket        | Native bidirectional     |
| Browser support | Native                | Requires gRPC-Web proxy  |
| Performance     | Good                  | Excellent (3-10x faster) |
| Best for        | Public APIs, web apps | Microservices, real-time |
| Tooling         | Postman, curl, HTTPie | grpcurl, Bloom, Evans    |
"""


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    if not GRPC_AVAILABLE:
        print(COMPARISON)
        sys.exit(0)

    if "--server" in sys.argv:
        serve()
    elif "--client" in sys.argv:
        run_client()
    else:
        print("Usage:")
        print("  python 08_grpc_service.py --server   # Start gRPC server")
        print("  python 08_grpc_service.py --client   # Run client demo")
        print()
        print(COMPARISON)
        print("\nRunning client demo by default...\n")
        run_client()
