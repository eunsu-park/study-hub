"""
Exercises for Lesson 14: Technical Documentation
Topic: Software_Engineering

Solutions to practice problems from the lesson.
Covers ADR writing, README audit, runbooks, API documentation, documentation rot.
"""


# === Exercise 1: ADR Writing ===

def exercise_1():
    """Write an Architecture Decision Record."""

    print("EXERCISE 1: Architecture Decision Record")
    print("=" * 65)

    print("""
  ADR-007: Use Redis for Application Caching
  ============================================

  Status: Accepted
  Date: 2025-02-15
  Deciders: Engineering Team (Tech Lead, Senior Backend, SRE)

  CONTEXT:
  Our API serves product catalog data that is read 100x more than written.
  Database queries for popular products take 200-500ms under load. Users
  experience slow page loads during peak traffic. We need a caching layer
  to reduce database load and improve response times.

  Current stack: Python FastAPI, PostgreSQL, deployed on AWS ECS.
  Expected cache size: ~500MB (10,000 products with metadata).
  Required features: key-value storage, TTL-based expiration, atomic operations.

  DECISION:
  We will use Redis (AWS ElastiCache for Redis) as the application cache
  for product catalog data, session storage, and rate limiting counters.

  ALTERNATIVES CONSIDERED:

  1. Memcached
     Pros: Simpler, multi-threaded (better per-node throughput for simple gets).
     Cons: No persistence, no data structures (only string values), no pub/sub.
     Rejected because: We need sorted sets for leaderboards, hashes for
     structured product data, and pub/sub for cache invalidation across
     multiple app instances.

  2. Application-level caching (Python functools.lru_cache or cachetools)
     Pros: No infrastructure to manage. Zero network latency.
     Cons: Cache is per-process — not shared across multiple ECS instances.
     Cache is lost on every deployment. No TTL management.
     Rejected because: With 4+ ECS instances, each would have its own
     inconsistent cache. Memory usage would multiply by the number of instances.

  3. PostgreSQL materialized views
     Pros: No new infrastructure. SQL-native.
     Cons: Refreshing materialized views is expensive and blocks writes.
     Does not help with session storage or rate limiting.
     Rejected because: Does not address the full problem (sessions, rate limits).

  CONSEQUENCES:

  Positive:
  - Product API response times expected to drop from 200-500ms to <10ms for
    cached hits (90%+ of reads).
  - Database load reduced by ~80% for catalog queries.
  - Redis data structures (sorted sets, hashes) enable features like
    "trending products" without additional database queries.
  - AWS ElastiCache provides managed failover and monitoring.

  Negative:
  - New infrastructure dependency. Redis downtime affects the application
    (mitigated by circuit breaker: fall back to database on Redis failure).
  - Cache invalidation complexity: must ensure product updates invalidate
    the correct cache keys (mitigated by write-through cache pattern).
  - Additional cost: ~$150/month for ElastiCache (r6g.large).
  - Team must learn Redis operations (monitoring, memory management).

  IMPLEMENTATION NOTES:
  - Use redis-py with connection pooling (max 50 connections).
  - Cache TTL: 5 minutes for product data, 30 minutes for category lists.
  - Cache key format: product:{id}, category:{slug}
  - Monitor: cache hit rate, memory usage, eviction count.
""")


# === Exercise 2: README Audit ===

def exercise_2():
    """Audit a poor README and propose a revised structure."""

    print("EXERCISE 2: README Audit and Revision")
    print("=" * 65)

    print("""
  EXAMPLE POOR README (hypothetical open-source CLI tool):

  ---
  # mytool
  A tool for stuff.
  Run `python main.py` to start.
  ---

  PROBLEMS IDENTIFIED:

  1. No description of what the tool does or who it's for.
     "A tool for stuff" tells the reader nothing.

  2. No installation instructions. What Python version? What dependencies?
     How to install? pip install? clone + setup?

  3. No usage examples. "Run python main.py" — with what arguments?
     What does it output? What does a successful run look like?

  4. No license information. Can I use this in my project?

  5. No contribution guidelines. How do I report bugs? Submit PRs?

  6. No indication of project status. Is this maintained? Alpha? Stable?


  REVISED README OUTLINE:

  # mytool — Automated Log Analyzer for Web Servers

  [![Build Status](badge)](link) [![License: MIT](badge)](link)

  ## What is mytool?
  `mytool` analyzes web server access logs (nginx, Apache) and generates
  reports showing top URLs, error rates, and traffic trends. It processes
  millions of log lines in seconds using streaming parsing.

  ## Quick Start
  ```bash
  pip install mytool
  mytool analyze --input access.log --output report.html
  ```

  ## Installation
  - Requirements: Python 3.10+
  - Install via pip: `pip install mytool`
  - Install from source: `git clone ... && pip install -e .`

  ## Usage
  ### Basic Analysis
  ```bash
  mytool analyze --input access.log
  ```
  ### Filtering by Date Range
  ```bash
  mytool analyze --input access.log --from 2025-01-01 --to 2025-01-31
  ```
  ### Output Formats
  - HTML report: `--format html` (default)
  - JSON: `--format json`
  - CSV: `--format csv`

  ## Configuration
  Configuration file: `~/.mytool/config.yaml`
  List key configuration options with defaults.

  ## Contributing
  See [CONTRIBUTING.md](CONTRIBUTING.md). TL;DR: Fork, branch, PR, tests must pass.

  ## License
  MIT License. See [LICENSE](LICENSE).

  ## Support
  - Issues: GitHub Issues
  - Discussions: GitHub Discussions
""")


# === Exercise 3: Runbook Creation ===

def exercise_3():
    """Runbook for database connection error alert."""

    print("EXERCISE 3: Runbook — Database Connection Error")
    print("=" * 65)

    print("""
  RUNBOOK: Database Connection Pool Exhaustion
  ==============================================
  Alert Name:    db-connection-pool-exhausted
  Severity:      HIGH
  Service:       api-service
  Last Updated:  2025-02-15
  Author:        SRE Team

  DESCRIPTION:
  This alert fires when the database connection pool utilization exceeds
  90% for more than 2 minutes. If all connections are in use, new requests
  will fail with "connection pool exhausted" errors (HTTP 503).

  DIAGNOSIS STEPS:

  1. Check current connection pool status:
     $ kubectl exec -it deploy/api-service -- python -c "
       from app.db import engine; print(engine.pool.status())"

     Expected output: "Pool size: 20  Connections in use: 18  Overflow: 5"
     If "in use" == pool size, the pool is exhausted.

  2. Check for long-running queries:
     $ psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "
       SELECT pid, now() - query_start AS duration, query
       FROM pg_stat_activity
       WHERE state = 'active' AND now() - query_start > interval '10 seconds'
       ORDER BY duration DESC;"

     If you see queries running for minutes, they may be blocking the pool.

  3. Check application error logs:
     $ kubectl logs deploy/api-service --since=10m | grep -i "connection"

     Look for: "QueuePool limit reached," "connection refused," or timeout errors.

  4. Check traffic levels:
     Open Grafana dashboard: https://grafana.internal/d/api-traffic
     Compare current requests/second to baseline. Is traffic abnormally high?

  MITIGATION STEPS (ordered by risk, lowest first):

  Step A (Low risk): Kill long-running queries
     If Step 2 found queries > 60 seconds:
     $ psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "
       SELECT pg_terminate_backend(PID);"
     Replace PID with the process ID from Step 2.
     This frees connections immediately.

  Step B (Low risk): Temporarily increase pool size
     $ kubectl set env deploy/api-service DB_POOL_SIZE=40
     This triggers a rolling restart with larger pool. Wait 2-3 minutes.
     NOTE: This is a temporary fix. The underlying cause must be found.

  Step C (Medium risk): Scale application horizontally
     $ kubectl scale deploy/api-service --replicas=6
     More instances = more total connections. But each instance creates
     its own pool, so total DB connections = replicas × pool_size.
     WARNING: Do not exceed the database's max_connections (default: 100).

  Step D (High risk): Restart the application
     $ kubectl rollout restart deploy/api-service
     Nuclear option. Drops all current requests. Use only if Steps A-C fail.

  ESCALATION CRITERIA:
  - If the alert does not resolve within 15 minutes: page the DBA on-call.
  - If the database itself is unresponsive: page the Infrastructure team.
  - If data loss is suspected: page the Engineering Manager immediately.

  POST-INCIDENT:
  - Revert any temporary changes (pool size, replica count) to baseline.
  - File a postmortem if the outage exceeded 5 minutes.
  - Add a ticket to investigate the root cause (missing index? N+1 query?).
""")


# === Exercise 4: API Documentation (OpenAPI) ===

def exercise_4():
    """OpenAPI specification for a bookmarking API."""

    print("EXERCISE 4: API Documentation — Bookmarking API (OpenAPI)")
    print("=" * 65)

    openapi_spec = """
openapi: "3.0.3"
info:
  title: Bookmarking API
  version: "1.0.0"
  description: API for managing user bookmarks with tag-based organization.

servers:
  - url: https://api.example.com/v1

paths:
  /bookmarks:
    post:
      summary: Create a bookmark
      operationId: createBookmark
      tags: [Bookmarks]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [url, title]
              properties:
                url:
                  type: string
                  format: uri
                  example: "https://example.com/article"
                title:
                  type: string
                  maxLength: 200
                  example: "Interesting Article"
                tags:
                  type: array
                  items:
                    type: string
                  example: ["tech", "python"]
                notes:
                  type: string
                  maxLength: 1000
      responses:
        "201":
          description: Bookmark created
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Bookmark"
              example:
                id: "bm_abc123"
                url: "https://example.com/article"
                title: "Interesting Article"
                tags: ["tech", "python"]
                notes: null
                created_at: "2025-02-15T10:30:00Z"
        "400":
          description: Validation error (missing url or title)
        "409":
          description: Bookmark with this URL already exists

    get:
      summary: List bookmarks (with optional tag filter)
      operationId: listBookmarks
      tags: [Bookmarks]
      parameters:
        - name: tag
          in: query
          required: false
          schema:
            type: string
          description: Filter by tag
          example: "python"
        - name: page
          in: query
          schema:
            type: integer
            default: 1
        - name: per_page
          in: query
          schema:
            type: integer
            default: 20
            maximum: 100
      responses:
        "200":
          description: List of bookmarks
          content:
            application/json:
              schema:
                type: object
                properties:
                  data:
                    type: array
                    items:
                      $ref: "#/components/schemas/Bookmark"
                  total:
                    type: integer
                  page:
                    type: integer
              example:
                data:
                  - id: "bm_abc123"
                    url: "https://example.com/article"
                    title: "Interesting Article"
                    tags: ["tech", "python"]
                    created_at: "2025-02-15T10:30:00Z"
                total: 1
                page: 1

  /bookmarks/{bookmark_id}:
    delete:
      summary: Delete a bookmark
      operationId: deleteBookmark
      tags: [Bookmarks]
      parameters:
        - name: bookmark_id
          in: path
          required: true
          schema:
            type: string
          example: "bm_abc123"
      responses:
        "204":
          description: Bookmark deleted successfully
        "404":
          description: Bookmark not found

components:
  schemas:
    Bookmark:
      type: object
      properties:
        id:
          type: string
        url:
          type: string
          format: uri
        title:
          type: string
        tags:
          type: array
          items:
            type: string
        notes:
          type: string
          nullable: true
        created_at:
          type: string
          format: date-time
"""
    print(openapi_spec)


# === Exercise 5: Documentation Rot Analysis ===

def exercise_5():
    """Identify and analyze documentation rot examples."""

    print("EXERCISE 5: Documentation Rot Analysis")
    print("=" * 65)

    rot_examples = [
        {
            "example": "Setup guide says 'Run npm install' but project migrated to pnpm 6 months ago",
            "what_is_wrong": (
                "The installation command is incorrect. npm install will either fail "
                "(if package-lock.json was removed) or install different dependency "
                "versions than the team uses."
            ),
            "how_reader_is_misled": (
                "A new developer follows the README, gets mysterious build errors, "
                "and spends hours debugging before discovering the tool change. "
                "First impression: 'This team's docs are unreliable.'"
            ),
            "prevention": (
                "Add a CI check that runs the README's setup commands in a fresh "
                "container. If the commands fail, the CI fails. This catches doc rot "
                "at the point of change, not months later."
            ),
        },
        {
            "example": "API docs reference endpoint /api/v1/users but it was renamed to /api/v2/users 3 releases ago",
            "what_is_wrong": (
                "The endpoint URL is wrong. Any developer or integration partner "
                "following the docs will get 404 errors."
            ),
            "how_reader_is_misled": (
                "An external partner building an integration spends a day debugging "
                "404 errors, opens a support ticket, and loses trust in the API's "
                "stability and documentation quality."
            ),
            "prevention": (
                "Generate API docs from the code (OpenAPI auto-generation from route "
                "decorators). The docs are always in sync because they ARE the code. "
                "Alternatively, add a CI step that hits every documented endpoint "
                "and verifies it returns a non-404 response."
            ),
        },
        {
            "example": "Architecture diagram shows a monolithic app but system was decomposed into 5 microservices last year",
            "what_is_wrong": (
                "The architecture overview is fundamentally wrong. It shows a system "
                "that no longer exists."
            ),
            "how_reader_is_misled": (
                "A new team member studies the diagram, forms an incorrect mental model "
                "of the system, and makes design decisions based on a monolith architecture "
                "that does not exist. Their code may not account for network boundaries, "
                "eventual consistency, or service ownership."
            ),
            "prevention": (
                "Use docs-as-code: architecture diagrams are written in a text-based "
                "format (Mermaid, PlantUML, Structurizr DSL) stored in the repo alongside "
                "the code. Include a 'Last verified' date on every diagram. During quarterly "
                "architecture reviews, compare diagrams to running infrastructure. "
                "Consider generating diagrams from service discovery (e.g., Kubernetes "
                "service topology -> auto-generated diagram)."
            ),
        },
    ]

    for i, rot in enumerate(rot_examples, 1):
        print(f"\n  Example {i}: {rot['example']}")
        print(f"\n    What is wrong:")
        print(f"      {rot['what_is_wrong']}")
        print(f"\n    How reader is misled:")
        print(f"      {rot['how_reader_is_misled']}")
        print(f"\n    Process change to prevent recurrence:")
        print(f"      {rot['prevention']}")


if __name__ == "__main__":
    exercises = [
        ("Exercise 1: ADR Writing", exercise_1),
        ("Exercise 2: README Audit", exercise_2),
        ("Exercise 3: Runbook Creation", exercise_3),
        ("Exercise 4: API Documentation", exercise_4),
        ("Exercise 5: Documentation Rot", exercise_5),
    ]
    for title, func in exercises:
        print(f"\n{'=' * 65}")
        print(f"=== {title} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
