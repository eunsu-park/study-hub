# API_Design Exercises

Practice problem solutions for the API Design topic (16 lessons). Each file corresponds to a lesson and contains working solutions with Python code displayed via heredoc.

## Exercise Files

| # | File | Lesson | Description |
|---|------|--------|-------------|
| 01 | `01_api_design_fundamentals.sh` | API Design Fundamentals | API types, design principles, contract-first vs code-first |
| 02 | `02_rest_architecture.sh` | REST Architecture | REST constraints, Richardson Maturity Model, HATEOAS |
| 03 | `03_url_design_and_naming.sh` | URL Design and Naming | Resource naming, hierarchical URLs, query parameters |
| 04 | `04_request_and_response_design.sh` | Request and Response Design | HTTP methods, status codes, content negotiation |
| 05 | `05_pagination_and_filtering.sh` | Pagination and Filtering | Cursor/offset pagination, filtering, sorting, sparse fieldsets |
| 06 | `06_authentication_and_authorization.sh` | Authentication and Authorization | API keys, OAuth 2.0, JWT, scopes and permissions |
| 07 | `07_api_versioning.sh` | API Versioning | URL/header/query versioning, deprecation, backward compatibility |
| 08 | `08_error_handling.sh` | Error Handling | RFC 7807 Problem Details, error hierarchy, validation errors |
| 09 | `09_rate_limiting_and_throttling.sh` | Rate Limiting and Throttling | Token bucket, sliding window, rate limit headers, retry logic |
| 10 | `10_api_documentation.sh` | API Documentation | OpenAPI 3.1, Swagger UI, Redoc, documentation-driven design |
| 11 | `11_validation_and_serialization.sh` | Validation and Serialization | Pydantic models, request validation, response serialization |
| 12 | `12_hateoas_and_hypermedia.sh` | HATEOAS and Hypermedia | Hypermedia controls, HAL, JSON:API, link relations |
| 13 | `13_webhooks_and_event_driven_apis.sh` | Webhooks and Event-Driven APIs | Webhook design, delivery guarantees, event payloads |
| 14 | `14_graphql_fundamentals.sh` | GraphQL Fundamentals | Schema definition, queries, mutations, subscriptions |
| 15 | `15_api_testing_and_contracts.sh` | API Testing and Contracts | Contract testing, integration tests, mocking, CI pipelines |
| 16 | `16_api_gateway_and_management.sh` | API Gateway and Management | API gateways, developer portals, analytics, lifecycle |

## How to Use

1. Study the lesson in `content/en/API_Design/` or `content/ko/API_Design/`
2. Attempt the exercises at the end of each lesson on your own
3. Run an exercise file to view the solutions: `bash exercises/API_Design/01_api_design_fundamentals.sh`
4. Each exercise function prints its solution as Python code

## File Structure

Each `.sh` file follows this pattern:

```bash
#!/bin/bash
# Exercises for Lesson XX: Title
# Topic: API_Design
# Solutions to practice problems from the lesson.

exercise_1() {
    echo "=== Exercise 1: Title ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
    # Python solution code here
SOLUTION
}

# Run all exercises
exercise_1
exercise_2
```
