# Exercise: Production Deployment
# Practice with deployment configuration and patterns.

# Exercise 1: Dockerfile
# Write a multi-stage Dockerfile for a FastAPI application.
# Requirements:
# - Build stage: install dependencies with pip
# - Production stage: slim Python image, non-root user
# - Health check endpoint
# - Proper signal handling (PID 1)
#
# TODO: Write the Dockerfile content as a string

DOCKERFILE = """
# TODO: Write multi-stage Dockerfile
# Stage 1: builder
# Stage 2: production
"""


# Exercise 2: Nginx Reverse Proxy Config
# Write nginx config for:
# - HTTPS termination (self-signed for demo)
# - Proxy to upstream FastAPI on port 8000
# - WebSocket support for /ws
# - Static file serving from /static
# - Security headers
# - Rate limiting

NGINX_CONFIG = """
# TODO: Write nginx configuration
"""


# Exercise 3: Docker Compose
# Write docker-compose.yml for a full stack:
# - FastAPI app (2 replicas)
# - PostgreSQL with persistent volume
# - Redis for caching
# - Nginx as reverse proxy
# - Celery worker
# - Health checks for all services

DOCKER_COMPOSE = """
# TODO: Write docker-compose.yml
"""


# Exercise 4: Gunicorn Configuration
# Create a gunicorn config file for production.

GUNICORN_CONFIG = """
# gunicorn.conf.py
# TODO: Configure:
# - Workers: 2 * CPU + 1
# - Worker class: uvicorn.workers.UvicornWorker
# - Bind address
# - Logging
# - Timeouts
# - Graceful shutdown
"""


# Exercise 5: Health Check Implementation
# Implement comprehensive health check endpoints.

def create_health_checks():
    """Create health check functions that verify:
    - Database connectivity
    - Redis connectivity
    - Disk space
    - Memory usage

    Returns dict with status for each component.
    """
    import os

    async def check_health() -> dict:
        results = {}

        # TODO: Check database (try SELECT 1)
        results["database"] = "unknown"

        # TODO: Check Redis (try PING)
        results["redis"] = "unknown"

        # TODO: Check disk space (warn if < 10% free)
        results["disk"] = "unknown"

        # TODO: Check memory (warn if > 90% used)
        results["memory"] = "unknown"

        overall = "healthy" if all(
            v == "healthy" for v in results.values()
        ) else "unhealthy"

        return {"status": overall, "checks": results}

    return check_health


if __name__ == "__main__":
    print("Production Deployment Exercise")
    print("Write configuration files and verify with Docker.")
