"""
Exercises for Lesson 09: Configuration Management
Topic: Software_Engineering

Solutions to practice problems from the lesson.
This lesson has two exercise sections: Practice Exercises (5) and Exercises (5).
Covers CI identification, branching strategies, dependency management, change control.
"""


# =====================================================================
# PRACTICE EXERCISES (Section 12)
# =====================================================================

def practice_exercise_1():
    """Configuration Item identification for a microservice."""

    print("PRACTICE EXERCISE 1: Configuration Item Identification")
    print("=" * 65)

    print("""
  SYSTEM: Python FastAPI microservice + React frontend
  STACK:  PostgreSQL, Redis, AWS (Terraform), OpenAPI docs

  (a) CONFIGURATION ITEMS (15+), by Category:

  SOURCE CODE:
    1. FastAPI application source (Python)
    2. React frontend source (TypeScript/JSX)
    3. Database migration scripts (Alembic)
    4. Unit test suites (pytest, Jest)
    5. Integration test suites

  CONFIGURATION FILES:
    6. pyproject.toml / requirements.txt (Python dependencies)
    7. package.json + package-lock.json (Node dependencies)
    8. Dockerfile (backend) + Dockerfile (frontend)
    9. docker-compose.yml (local development)
    10. .env.example (template for environment variables)

  INFRASTRUCTURE CODE:
    11. Terraform modules (VPC, RDS, ElastiCache, ECS)
    12. CI/CD pipeline definition (.github/workflows/ or Jenkinsfile)

  DOCUMENTATION:
    13. OpenAPI specification (openapi.yaml)
    14. Architecture Decision Records (ADRs)
    15. README.md and developer setup guide

  BUILD/DEPLOYMENT:
    16. Makefile or task runner configuration
    17. Kubernetes manifests or ECS task definitions

  (b) THREE ARTIFACTS THAT ARE NOT CIs:

  1. .env files with actual secrets (passwords, API keys)
     WHY: Secrets must never be in version control. Leaked credentials
     compromise the system. Store in AWS Secrets Manager / HashiCorp Vault.

  2. node_modules/ and __pycache__/ directories
     WHY: Generated from dependency specifications; reproducible from
     lockfiles. Versioning them wastes storage and causes merge conflicts.

  3. Build output artifacts (dist/, *.whl, *.tar.gz)
     WHY: These are derived from source code. They belong in an artifact
     registry (Docker Hub, PyPI, S3), not in Git.

  (c) DOCKER IMAGE TAG NAMING SCHEME:

  Format: <service>:<semver>-<git-sha>-<build-date>
  Example: fastapi-service:1.4.2-a3b2c1d-20250215

  Components:
    - service name:  Identifies which service (in a multi-service repo)
    - semver:        Human-readable release version
    - git-sha:       7-char Git commit hash for exact traceability
    - build-date:    YYYYMMDD for quick age identification

  Additional tags:
    - :latest        Always points to the most recent production build
    - :staging       Current staging deployment
""")


def practice_exercise_2():
    """Branching strategy design and transition plan."""

    print("PRACTICE EXERCISE 2: Branching Strategy")
    print("=" * 65)

    print("""
  CONTEXT: 8 developers, e-commerce platform, biweekly releases
  GOAL: Transition to daily releases within 6 months

  (a) CURRENT STRATEGY: GitFlow (for biweekly releases)

  Branches:
    main          Reflects production. Tagged on each release.
    develop       Integration branch. All feature PRs merge here.
    feature/*     Short-lived feature branches off develop.
    release/*     Cut from develop 2-3 days before release for stabilization.
    hotfix/*      Emergency fixes branched from main, merged back to main + develop.

  Diagram:
    main:    ----o-----------o-----------o--------> (releases)
                  \\          /             /
    release:       \\--R1---/    \\--R2---/
                    \\          /
    develop: ----o---o---o---o---o---o---o---------> (integration)
                /   /       \\   \\
    features: f1  f2        f3   f4

  (b) TRANSITION PLAN TO TRUNK-BASED DEVELOPMENT:

  Phase 1 (Months 1-2): Shorten feature branch lifetime
    - Enforce: no feature branch lives longer than 2 days
    - Require: all PRs must pass CI before merge
    - Establish: automated test suite with >80% coverage

  Phase 2 (Months 3-4): Introduce feature flags
    - Implement feature flag service (LaunchDarkly or Unleash)
    - Developers merge incomplete features behind flags
    - Release weekly (enabled by flags decoupling deploy from release)

  Phase 3 (Months 5-6): Trunk-based development
    - Eliminate develop branch; merge directly to main
    - Use short-lived branches (< 1 day) or commit directly to main
    - Deploy on every green main build (continuous deployment)

  PREREQUISITES FOR TRUNK-BASED:
    1. Comprehensive automated test suite (unit + integration + e2e)
    2. Feature flag infrastructure
    3. Fast CI pipeline (< 10 minutes from push to green)
    4. Automated rollback capability (canary deployments)
    5. Team culture shift: small, frequent commits

  (c) HOTFIX WORKFLOW (During Biweekly Cycle):

  Scenario: Production bug in v2.4.0 while v2.5.0 is in development on develop.

  Step 1: Create hotfix branch from main
    $ git checkout main
    $ git pull origin main
    $ git checkout -b hotfix/fix-payment-timeout

  Step 2: Fix the bug, commit
    $ git add src/payment.py
    $ git commit -m "fix: increase payment gateway timeout to 30s"

  Step 3: Tag and merge to main
    $ git checkout main
    $ git merge --no-ff hotfix/fix-payment-timeout
    $ git tag v2.4.1

  Step 4: Merge hotfix back to develop
    $ git checkout develop
    $ git merge --no-ff hotfix/fix-payment-timeout

  Step 5: Deploy v2.4.1 to production
    $ git push origin main --tags
    # CI/CD deploys tagged releases automatically

  Step 6: Clean up
    $ git branch -d hotfix/fix-payment-timeout
    $ git push origin --delete hotfix/fix-payment-timeout
""")


def practice_exercise_3():
    """Dependency analysis and Dependabot configuration."""

    print("PRACTICE EXERCISE 3: Dependency Analysis")
    print("=" * 65)

    print("""
  (a) RISKS OF boto3 = "*" (ANY VERSION):

  1. Breaking changes: A major version bump (e.g., boto3 2.0) could change
     API signatures, causing runtime errors. With "*", pip will happily
     install the breaking version.

  2. Non-reproducible builds: Two developers running `pip install` on
     different days may get different boto3 versions, leading to
     "works on my machine" issues.

  3. Supply chain risk: An attacker who compromises a new boto3 release
     gets automatic installation on all systems that resolve "*".

  4. Transitive dependency conflicts: A new boto3 version may require
     a different botocore version that conflicts with other dependencies.

  RECOMMENDATION: Pin to a range: boto3 = "^1.28" (allow patches, not major).

  (b) django = ">=4.2,<5.0" vs "4.2.8":

  ADVANTAGES of range (>=4.2,<5.0):
    - Automatically gets security patches (4.2.9, 4.2.10, etc.)
    - Less maintenance burden (no manual version bumps for patches)
    - The lockfile ensures actual reproducibility

  DISADVANTAGES:
    - A new minor version (4.3.0) could introduce subtle behavior changes
    - Less predictable: two environments without a lockfile may differ
    - Harder to debug if a patch introduces a regression

  COMPARISON: Exact pin (4.2.8) gives maximum reproducibility but requires
  manual updates for every security patch — risky if the team forgets.

  RECOMMENDATION: Use the range WITH a lockfile. The range defines the
  acceptable window; the lockfile ensures exact reproducibility.

  (c) DEPENDABOT CONFIGURATION:
""")

    dependabot_yaml = """# .github/dependabot.yml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "09:00"
      timezone: "UTC"

    # Group all patch updates into a single PR
    groups:
      patch-updates:
        update-types:
          - "patch"

    # Require security team review for major updates
    reviewers:
      - "@security-team"

    # Only auto-open PRs for patch and minor updates
    open-pull-requests-limit: 10

    # Labels for filtering
    labels:
      - "dependencies"
      - "automated"

    # Ignore specific packages if needed
    # ignore:
    #   - dependency-name: "boto3"
    #     update-types: ["version-update:semver-major"]
"""
    print(dependabot_yaml)


def practice_exercise_4():
    """Change Request document for session token replacement."""

    print("PRACTICE EXERCISE 4: Change Request — Session Token Replacement")
    print("=" * 65)

    print("""
  (a) CHANGE REQUEST DOCUMENT:

  CR-ID:           CR-2025-018
  Title:           Replace MD5-based session tokens with cryptographically
                   secure random tokens (256-bit, base64-encoded)
  Requestor:       Security Team
  Priority:        HIGH (security vulnerability)
  Category:        Security Enhancement
  Estimated Effort: 4 hours development + 2 hours testing

  Description:
    Current session tokens are generated using MD5 hash of user ID +
    timestamp. MD5 is cryptographically broken — tokens are predictable
    and susceptible to collision attacks. An attacker who knows the user ID
    and approximate login time can brute-force the session token.

    Replace with: `secrets.token_urlsafe(32)` (256-bit CSPRNG).

  Impact Analysis:
    Code:       auth/session.py (token generation function)
    Database:   sessions table — token column length may need to increase
                (MD5 = 32 hex chars; base64(256-bit) = 43 chars)
    Clients:    No change required — tokens are opaque to clients
    Performance: Negligible (CSPRNG is fast)
    Rollback:   All active sessions will be invalidated (users must re-login)
    Risk:       LOW — change is isolated to one function

  (b) CCB QUESTIONS:

  1. Is the MD5 vulnerability currently being exploited, or is this preventive?
  2. Will this change invalidate all existing sessions? (Yes — is downtime acceptable?)
  3. Has the new token format been tested with all client types (mobile, web, API)?
  4. Is the session table column wide enough for the new token format?
  5. What is the rollback plan if the new token generation fails?
  6. Are there any compliance requirements (PCI-DSS) that this change satisfies?

  (c) VERIFICATION STEPS:

  Pre-deployment:
    1. Unit test: new token is 256-bit, URL-safe base64
    2. Unit test: two consecutive tokens are never identical
    3. Integration test: login flow produces valid session with new token
    4. Integration test: old MD5 tokens are rejected after migration
    5. Security test: verify token has sufficient entropy (NIST SP 800-63B)
    6. Load test: token generation under 1000 concurrent logins (no bottleneck)

  Post-deployment:
    7. Smoke test: login on web, mobile, and API clients
    8. Monitor: error rate on /auth/* endpoints for 1 hour
    9. Verify: no MD5-format tokens appearing in new sessions
""")


def practice_exercise_5():
    """Semantic versioning for various changes."""

    print("PRACTICE EXERCISE 5: Semantic Versioning — Release Management")
    print("=" * 65)

    print("  Current version: 2.7.3\n")

    changes = [
        {
            "change": "(a) Bug fix: correct timezone handling in meeting reminders",
            "version": "2.7.4",
            "rule": "PATCH increment. Bug fix with no API change. SemVer: increment Z in X.Y.Z.",
        },
        {
            "change": "(b) New 'dark mode' toggle in user profile settings",
            "version": "2.8.0",
            "rule": "MINOR increment. New user-facing feature added. Does not break existing "
                    "functionality. Reset PATCH to 0.",
        },
        {
            "change": "(c) /users endpoint returns 'user_id' instead of 'id' in JSON",
            "version": "3.0.0",
            "rule": "MAJOR increment. Breaking change to the public API. Existing clients "
                    "that expect 'id' will break. Must bump X and reset Y.Z to 0.",
        },
        {
            "change": "(d) Add Japanese and Korean UI translations, no API changes",
            "version": "2.8.0",
            "rule": "MINOR increment. New functionality (new languages) added in a "
                    "backward-compatible manner. No API contracts changed.",
        },
        {
            "change": "(e) Remove deprecated SOAP API endpoint (announced 12 months ago)",
            "version": "3.0.0",
            "rule": "MAJOR increment. Removing a public API endpoint is a breaking change "
                    "for any consumer still using it. The 12-month deprecation notice does "
                    "not change the SemVer decision — removal = breaking = major bump.",
        },
    ]

    for c in changes:
        print(f"  {c['change']}")
        print(f"    Next version: {c['version']}")
        print(f"    Rule: {c['rule']}")
        print()

    print("  NOTE: If changes (c) and (e) are released together, the version is 3.0.0")
    print("  (not 4.0.0). Multiple breaking changes in one release = one major bump.")


# =====================================================================
# EXERCISES (End of Lesson)
# =====================================================================

def exercise_1():
    """Identify Configuration Items for a FastAPI service."""

    print("EXERCISE 1: Configuration Item Identification — FastAPI Service")
    print("=" * 65)

    print("""
  (a) CONFIGURATION ITEMS (12+), by Category:

  SOURCE CODE:
    1. FastAPI application code (routes, models, services)
    2. Database migration files (Alembic versions/)
    3. Test suites (unit, integration, e2e)

  BUILD ARTIFACTS:
    4. Dockerfile
    5. docker-compose.yml (local dev)
    6. CI/CD pipeline definition (.github/workflows/*.yml)

  DOCUMENTATION:
    7. OpenAPI specification (openapi.yaml / auto-generated)
    8. Architecture Decision Records (docs/adr/)
    9. README and developer setup guide

  CONFIGURATION FILES:
    10. pyproject.toml (dependencies, build config)
    11. poetry.lock (exact dependency versions)
    12. .env.example (template, NOT actual secrets)
    13. Application config (settings.py or config.yaml)

  INFRASTRUCTURE CODE:
    14. Terraform modules (main.tf, variables.tf, outputs.tf)
    15. Terraform state backend configuration

  (b) VERSION CONTROL STRATEGY PER CI:

  Git Repository:
    Items 1-9, 10-13, 14: All managed in Git. These are text-based,
    change frequently, and benefit from diff/merge/history.
    Justification: Git provides line-level tracking, branching, and
    blame — essential for collaborative development.

  Artifact Registry (Docker Hub / ECR):
    Item: Built Docker images
    Justification: Binary artifacts are large and not diffable. Registry
    provides image signing, vulnerability scanning, and pull-based deployment.

  Secrets Manager (AWS Secrets Manager / Vault):
    Items: Database passwords, API keys, TLS certificates
    Justification: Secrets must never appear in Git history. Secrets
    managers provide rotation, access control, and audit logging.

  (c) TWO ARTIFACTS NOT TO VERSION-CONTROL:

  1. .env files with real credentials
     WHY NOT: Leaked secrets in Git history cannot be un-leaked. Even after
     deletion, they persist in Git reflog and clones.
     INSTEAD: Use AWS Secrets Manager. Reference secrets by name in config,
     not by value. Provide .env.example as a template.

  2. Terraform state files (terraform.tfstate)
     WHY NOT: State files contain sensitive data (resource IDs, passwords)
     and are frequently overwritten. Git conflicts in state files can
     corrupt infrastructure tracking.
     INSTEAD: Use a remote backend (S3 + DynamoDB for locking). State is
     managed by Terraform, not by developers.
""")


def exercise_2():
    """Design branching strategy for HR SaaS platform."""

    print("EXERCISE 2: Branching Strategy — HR SaaS Platform")
    print("=" * 65)

    print("""
  (a) MONTHLY RELEASE CADENCE — GitFlow Variant:

  Branches:
    main:      Production code. Tagged at each release (v1.0, v1.1, ...).
    develop:   Integration branch. Feature PRs merge here.
    feature/*: Short-lived (< 1 week). One per feature/ticket.
    release/*: Cut from develop ~1 week before release. Only bug fixes allowed.
    hotfix/*:  Emergency fixes from main.

  Flow:
    feature/* -> develop -> release/* -> main (tagged)
    hotfix/* -> main + develop

  (b) HOTFIX WORKFLOW — Security Vulnerability:

  # 1. Create hotfix branch from main
  $ git checkout main
  $ git pull origin main
  $ git checkout -b hotfix/sec-xss-fix

  # 2. Implement the fix
  $ vim src/sanitizer.py   # fix the XSS vulnerability
  $ python -m pytest tests/test_sanitizer.py  # run relevant tests
  $ git add src/sanitizer.py tests/test_sanitizer.py
  $ git commit -m "fix(security): sanitize HTML input to prevent XSS (CVE-2025-1234)"

  # 3. Merge to main and tag
  $ git checkout main
  $ git merge --no-ff hotfix/sec-xss-fix
  $ git tag -a v1.3.1 -m "Hotfix: XSS vulnerability (CVE-2025-1234)"
  $ git push origin main --tags

  # 4. Merge hotfix to develop (so next release includes the fix)
  $ git checkout develop
  $ git merge --no-ff hotfix/sec-xss-fix
  $ git push origin develop

  # 5. If a release branch exists, merge there too
  $ git checkout release/1.4
  $ git merge --no-ff hotfix/sec-xss-fix
  $ git push origin release/1.4

  # 6. Clean up
  $ git branch -d hotfix/sec-xss-fix
  $ git push origin --delete hotfix/sec-xss-fix

  # 7. Deploy v1.3.1 immediately (CI/CD triggers on tag)

  (c) THREE PREREQUISITES FOR TRUNK-BASED DEVELOPMENT:

  1. Comprehensive automated test suite with >85% coverage and fast
     execution (< 10 minutes). Without this, broken code reaches main
     and blocks everyone.

  2. Feature flag infrastructure to decouple deployment from release.
     Incomplete features are merged behind flags so main is always
     deployable.

  3. Automated deployment pipeline with rollback capability (canary
     deployments, blue-green, or feature flag kill switches). If a bad
     commit reaches production, recovery must be fast and automatic.
""")


def exercise_3():
    """Semantic versioning decisions for a library."""

    print("EXERCISE 3: Semantic Versioning — Library at v3.2.1")
    print("=" * 65)

    print("  Current version: 3.2.1\n")

    changes = [
        ("(a) Bug fix: off-by-one error in date range calculation",
         "3.2.2",
         "PATCH: Bug fix, no API change. Increment Z (3.2.1 -> 3.2.2)."),
        ("(b) New batch_process() method added; existing methods unchanged",
         "3.3.0",
         "MINOR: New functionality added in backward-compatible way. "
         "Increment Y, reset Z (3.2.1 -> 3.3.0)."),
        ("(c) process() return type changes from dict to Result dataclass",
         "4.0.0",
         "MAJOR: Breaking change. Existing callers that unpack the dict "
         "will fail. Increment X, reset Y.Z (3.2.1 -> 4.0.0)."),
        ("(d) Deprecated legacy_process() removed (deprecated since 3.0.0)",
         "4.0.0",
         "MAJOR: Removing a public API function is a breaking change "
         "regardless of deprecation notice. If combined with (c), still 4.0.0."),
        ("(e) Internal algorithm optimization; public API identical",
         "3.2.2",
         "PATCH: No API change. Internal implementation detail. Users are "
         "unaffected. Could also be argued as no version bump needed, but "
         "SemVer convention is to bump PATCH for any change in released code."),
    ]

    for change, version, rule in changes:
        print(f"  {change}")
        print(f"    Next version: {version}")
        print(f"    Rule: {rule}")
        print()


def exercise_4():
    """Dependency risk analysis."""

    print("EXERCISE 4: Dependency Risk Analysis")
    print("=" * 65)

    print("""
  Dependencies:
    python = "^3.11"
    fastapi = ">=0.100,<1.0"
    sqlalchemy = "^2.0"
    pydantic = "*"
    httpx = "^0.25"

  (a) RISK RANKING (Highest to Lowest):

  1. pydantic = "*" — HIGHEST RISK
     No version constraint at all. Pydantic v1 -> v2 was a massive breaking
     change (different API, different validation behavior). With "*", a fresh
     install could silently pick v2 when code expects v1 (or vice versa).

  2. httpx = "^0.25" — HIGH RISK
     Version 0.x in SemVer means "anything can break at any time." The ^
     operator on 0.x allows 0.25 to 0.99 — all of which may contain breaking
     changes. Pre-1.0 libraries should use tight pins.

  3. fastapi = ">=0.100,<1.0" — MODERATE RISK
     Also pre-1.0, but the range explicitly caps at <1.0. FastAPI 0.100 to
     0.199 could still contain breaking changes, but the range is narrower.

  4. sqlalchemy = "^2.0" — LOW-MODERATE RISK
     Post-1.0 library with SemVer. ^2.0 allows 2.0 to 2.99 — only minor/
     patch updates. SQLAlchemy respects SemVer well, so risk is moderate.

  5. python = "^3.11" — LOWEST RISK
     Python itself has very strict backward compatibility. ^3.11 allows
     3.11 to 3.99, which is fine — Python 3.12, 3.13 are backward-compatible.

  (b) PYDANTIC "*" RISK:

  Today: pydantic resolves to 1.10.14.
  In 6 months: pydantic 2.0 is released.

  Risk: A fresh `poetry install` (without lockfile) resolves "*" to 2.0.
  Pydantic v2 has a completely different API:
    - BaseModel.dict() -> model_dump()
    - validator() -> field_validator()
    - Different JSON schema output
  Result: Import errors, validation failures, silent data corruption.

  LOCKFILE MITIGATION:
  poetry.lock pins pydantic to exactly 1.10.14. As long as developers
  run `poetry install` (which reads the lockfile), they get 1.10.14.
  The risk only materializes when:
    - A new developer runs `poetry install` without the lockfile
    - Someone runs `poetry update pydantic` deliberately
    - The CI pipeline doesn't use the lockfile

  FIX: Change to pydantic = "^1.10" (explicit constraint) AND keep lockfile.

  (c) SAFELY UPGRADING sqlalchemy 2.0.x -> 2.1.x:

  Step 1: Create a branch and update the dependency
    $ git checkout -b chore/upgrade-sqlalchemy-2.1
    $ poetry update sqlalchemy
    $ poetry show sqlalchemy  # verify 2.1.x installed
    $ git diff poetry.lock    # review all transitive changes

  Step 2: Run full test suite and fix any issues
    $ pytest tests/ -v
    # Fix any deprecation warnings or behavior changes
    # Check SQLAlchemy 2.1 changelog for known migration steps

  Step 3: Create PR, run CI, merge
    $ git add pyproject.toml poetry.lock
    $ git commit -m "chore: upgrade sqlalchemy to 2.1.x"
    $ git push origin chore/upgrade-sqlalchemy-2.1
    # CI runs full test suite on the PR
    # After review + green CI, merge to main
""")


def exercise_5():
    """Change Request for OAuth 2.0 migration."""

    print("EXERCISE 5: Change Request — OAuth 2.0 Migration")
    print("=" * 65)

    print("""
  CHANGE REQUEST: CR-2025-037
  ============================

  CHANGE DESCRIPTION:
    Migrate production API authentication from HTTP Basic Authentication
    to OAuth 2.0 Bearer tokens. The current Basic Auth sends credentials
    in every request (base64-encoded but not encrypted without TLS),
    violating security best practices.

  BUSINESS JUSTIFICATION:
    - Security team mandate (compliance with SOC 2 Type II)
    - Basic Auth credentials can be intercepted if TLS is misconfigured
    - OAuth 2.0 enables token expiration, scoping, and revocation
    - Required for enterprise customer onboarding (they mandate OAuth)

  AFFECTED CONFIGURATION ITEMS:
    1. Authentication middleware (src/auth/middleware.py)
    2. User model and token storage (src/models/user.py, src/models/token.py)
    3. API client libraries (Python SDK, JS SDK) — breaking change
    4. Developer documentation (docs/authentication.md)
    5. OpenAPI specification (openapi.yaml — security schemes section)
    6. CI/CD pipeline (integration tests need OAuth test tokens)
    7. Third-party integrations (Slack webhook, Stripe callback, monitoring)
    8. Infrastructure: OAuth authorization server (new Terraform module)
    9. Environment configuration (.env: new OAuth client IDs, secrets)

  IMPACT ANALYSIS:
    Schedule:  3 sprints (6 weeks): Sprint 1 = OAuth server + middleware,
               Sprint 2 = client migration support, Sprint 3 = cutover
    Effort:    ~120 person-hours (2 developers × 3 weeks)
    Risk:      HIGH — affects all API consumers. If migration is botched,
               all clients lose access.

  RISK MITIGATION:
    - Run OAuth and Basic Auth in parallel for 4 weeks (dual auth)
    - Provide migration guide and updated SDKs before deprecating Basic
    - Monitor Basic Auth usage to identify stragglers
    - Feature flag to instantly revert to Basic Auth if critical issues arise

  ROLLBACK PLAN:
    1. Feature flag: If OAuth fails, disable flag -> revert to Basic Auth
    2. Database: Old session tokens are preserved for 30 days post-migration
    3. Client SDKs: Previous SDK versions still work with Basic Auth
    4. DNS/Proxy: Route /auth/* to old Basic Auth service if needed

  VERIFICATION AND ACCEPTANCE CRITERIA:
    1. All existing API endpoints work with Bearer tokens
    2. Token refresh flow works without requiring re-authentication
    3. Token expiration is enforced (15-minute access token, 7-day refresh)
    4. Invalid/expired tokens return 401 with clear error message
    5. All 3 third-party integrations tested and confirmed working
    6. Load test: OAuth flow handles 1000 auth/s without degradation
    7. Security pen test: no token leakage, CSRF protection, PKCE for public clients
    8. Zero Basic Auth requests after 4-week parallel period (monitoring confirms)
""")


if __name__ == "__main__":
    print("=" * 65)
    print("=== PRACTICE EXERCISES ===")
    print("=" * 65)

    for i, func in enumerate([practice_exercise_1, practice_exercise_2,
                               practice_exercise_3, practice_exercise_4,
                               practice_exercise_5], 1):
        print(f"\n{'=' * 65}")
        print(f"=== Practice Exercise {i} ===")
        print("=" * 65)
        func()

    print("\n\n" + "=" * 65)
    print("=== EXERCISES (End of Lesson) ===")
    print("=" * 65)

    for i, func in enumerate([exercise_1, exercise_2, exercise_3,
                               exercise_4, exercise_5], 1):
        print(f"\n{'=' * 65}")
        print(f"=== Exercise {i} ===")
        print("=" * 65)
        func()

    print("\nAll exercises completed!")
