# CLAUDE.md

## Team Project: E-Commerce Platform

### Team Conventions

- **Branch naming**: `feature/<ticket-id>-<short-description>`
- **Commit messages**: Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`)
- **PR reviews**: All PRs require at least one approval
- **Deploy**: Merge to `main` triggers CI/CD pipeline

### Architecture

Microservices architecture with the following services:

| Service | Port | Tech |
|---------|------|------|
| api-gateway | 8080 | Node.js/Express |
| user-service | 8081 | Python/FastAPI |
| product-service | 8082 | Python/FastAPI |
| order-service | 8083 | Go |
| notification-service | 8084 | Node.js |

### Shared Libraries

- `libs/common-auth/` — JWT validation middleware (used by all services)
- `libs/event-bus/` — Kafka producer/consumer wrappers
- `libs/proto/` — Protobuf definitions for gRPC

### Development Setup

```bash
# Start all services with Docker Compose
docker compose up -d

# Run tests for a specific service
cd services/user-service && python -m pytest

# Generate protobuf stubs
make proto-gen

# View logs
docker compose logs -f user-service
```

### Database Conventions

- Each service owns its database (no shared databases)
- Use UUID for primary keys
- All tables must have `created_at` and `updated_at` timestamps
- Soft delete with `deleted_at` column (never hard delete user data)

### API Conventions

- REST endpoints follow: `/{version}/{resource}/{id}`
- Use JSON:API response format
- Pagination: cursor-based (not offset)
- Error responses include `error_code`, `message`, and `details`

### Testing Requirements

- Unit test coverage minimum: 80%
- Integration tests for all API endpoints
- E2E tests for critical user flows (checkout, signup)
- Load tests before each release

### Security Rules

- Never log PII (emails, passwords, addresses)
- All secrets in HashiCorp Vault (never in env vars or config files)
- Input validation on all API boundaries
- Rate limiting on all public endpoints
