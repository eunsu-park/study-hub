# API Design Examples

Example code for the API_Design topic.

| File | Description |
|------|-------------|
| [01_rest_api_design.py](01_rest_api_design.py) | FastAPI REST API with proper resource modeling, status codes, HATEOAS links |
| [02_pagination_patterns.py](02_pagination_patterns.py) | Offset, cursor, and keyset pagination implementations |
| [03_authentication_jwt.py](03_authentication_jwt.py) | JWT authentication with FastAPI: login, token generation, protected endpoints |
| [04_error_handling.py](04_error_handling.py) | RFC 7807 Problem Details error responses, validation error formatting |
| [05_rate_limiting.py](05_rate_limiting.py) | Token bucket and sliding window rate limiters as middleware |
| [06_openapi_spec.py](06_openapi_spec.py) | FastAPI with custom OpenAPI schema, examples, and documentation |
| [07_webhooks.py](07_webhooks.py) | Webhook sender with HMAC signing and receiver with verification |
| [08_grpc_service.py](08_grpc_service.py) | gRPC service definition (proto) and Python implementation |

## Running

```bash
# Install dependencies
pip install "fastapi[standard]" pyjwt passlib[bcrypt] httpx grpcio grpcio-tools

# Run any example as a standalone server
uvicorn examples.API_Design.01_rest_api_design:app --reload --port 8000

# Run a specific example directly
python examples/API_Design/01_rest_api_design.py

# Test endpoints with HTTPie
pip install httpie
http GET localhost:8000/api/v1/books
http POST localhost:8000/api/v1/books title="Clean Code" author="Robert Martin"
```

**License**: CC BY-NC 4.0
