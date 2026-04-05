# Lesson 3: CI Fundamentals

**Previous**: [Version Control Workflows](./02_Version_Control_Workflows.md) | **Next**: [GitHub Actions Deep Dive](./04_GitHub_Actions_Deep_Dive.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the principles of continuous integration and why frequent integration reduces risk
2. Design a CI pipeline with build, test, and deploy stages including appropriate quality gates
3. Distinguish between continuous integration, continuous delivery, and continuous deployment
4. Implement artifact management strategies for build outputs
5. Configure build triggers for different scenarios (push, pull request, scheduled, manual)
6. Apply CI best practices including fast feedback, hermetic builds, and test parallelization

---

Continuous Integration (CI) is the practice of automatically building and testing code every time a developer pushes changes. It was pioneered by the Extreme Programming community in the late 1990s and formalized by Martin Fowler in his influential 2006 article. CI is the foundation of the entire DevOps pipeline: without a reliable, fast CI process, continuous delivery and continuous deployment are impossible. This lesson covers the concepts, architecture, and best practices for building effective CI pipelines.

> **Analogy -- Assembly Line Quality Control:** CI is like quality inspection at every station on an assembly line, rather than one big inspection at the end. If a weld is bad, the worker catches it immediately. Without per-station checks, a defective weld propagates through the entire assembly, and the final inspection rejects a nearly complete product -- wasting all downstream work.

## 1. What is Continuous Integration?

Continuous Integration is the practice where developers **frequently merge** their code changes into a shared mainline, and each merge is **verified by an automated build and test suite**.

### Core Principles

```
┌──────────────────────────────────────────────────────────────┐
│                   CI Core Principles                          │
│                                                               │
│  1. Maintain a single source repository                      │
│  2. Automate the build                                       │
│  3. Make the build self-testing                               │
│  4. Everyone commits to mainline every day                   │
│  5. Every commit triggers a build                            │
│  6. Keep the build fast (< 10 minutes ideal)                 │
│  7. Test in a clone of the production environment            │
│  8. Make the latest build artifacts easy to access            │
│  9. Everyone can see what is happening (transparency)         │
│  10. Automate deployment                                      │
│                                                               │
│  -- Martin Fowler, "Continuous Integration" (2006)           │
└──────────────────────────────────────────────────────────────┘
```

### The CI Feedback Loop

```
Developer pushes code
        │
        ▼
┌─────────────────┐
│   CI Server     │
│   detects push  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│     Build       │────▶│     Test        │────▶│   Report        │
│  (compile,      │     │  (unit, lint,   │     │  (pass/fail,    │
│   package)      │     │   integration)  │     │   coverage)     │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                  ┌───────────────┐
                                                  │  Notify dev   │
                                                  │  (Slack, email,│
                                                  │   PR status)   │
                                                  └───────────────┘
         Feedback time: ideally < 10 minutes
```

---

## 2. CI vs CD vs CD

These three terms are often confused but represent distinct practices with increasing levels of automation.

### Continuous Integration (CI)

```
Code ──▶ Build ──▶ Unit Tests ──▶ Integration Tests ──▶ Feedback
                                                         │
                                            Developers fix issues
                                            before merging
```

**Definition**: Automatically build and test every code change.
**Output**: Verified, tested code in the main branch.
**Human action**: Developers must still decide when to release.

### Continuous Delivery (CD)

```
Code ──▶ Build ──▶ Test ──▶ Staging Deploy ──▶ Acceptance Tests
                                                       │
                                               Manual approval
                                                       │
                                                       ▼
                                              Production Deploy
```

**Definition**: Every change is automatically built, tested, and prepared for release. Deployment to production requires **manual approval** (a button click).
**Output**: A release-ready artifact at all times.
**Human action**: Someone clicks "deploy to production."

### Continuous Deployment (CD)

```
Code ──▶ Build ──▶ Test ──▶ Staging Deploy ──▶ Acceptance Tests
                                                       │
                                            Automatic (no human)
                                                       │
                                                       ▼
                                              Production Deploy
```

**Definition**: Every change that passes automated tests is **automatically deployed to production**. No human approval gates.
**Output**: Code in production within minutes of commit.
**Human action**: None (after the initial setup).

### Comparison Table

| Aspect | CI | Continuous Delivery | Continuous Deployment |
|--------|----|--------------------|----------------------|
| Build automation | Yes | Yes | Yes |
| Test automation | Yes | Yes | Yes |
| Deploy to staging | Optional | Automatic | Automatic |
| Deploy to production | Manual | Manual (one click) | Automatic |
| Rollback | Manual | Manual or automatic | Automatic |
| Release frequency | On team's schedule | When team decides | Every successful build |
| Risk per deploy | Varies | Low | Very low |

---

## 3. CI Pipeline Architecture

### Pipeline Stages

A typical CI pipeline consists of sequential stages, each containing one or more parallel jobs.

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Source   │───▶│  Build   │───▶│  Test    │───▶│ Security │───▶│ Artifact │
│          │    │          │    │          │    │  Scan    │    │ Publish  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
   trigger        compile        unit tests     SAST/SCA        docker push
   checkout        package       integration    license check   upload to
   dependencies    lint          coverage                       registry
```

### Stage Details

#### Source Stage

```bash
# Triggered by a code push or pull request
# CI server checks out the code

git clone --depth 1 https://github.com/org/repo.git
cd repo
git checkout $COMMIT_SHA
```

#### Build Stage

```bash
# Install dependencies
pip install -r requirements.txt

# Compile (if applicable)
go build ./...

# Lint code
flake8 src/
eslint src/**/*.js

# Type check
mypy src/
```

#### Test Stage

```bash
# Unit tests (fast, isolated)
pytest tests/unit/ --junitxml=reports/unit.xml

# Integration tests (may need external services)
docker compose up -d postgres redis
pytest tests/integration/ --junitxml=reports/integration.xml
docker compose down

# Code coverage
pytest --cov=src --cov-report=xml:reports/coverage.xml

# Coverage threshold enforcement
coverage report --fail-under=80
```

#### Security Scan Stage

```bash
# Static Application Security Testing (SAST)
bandit -r src/ -f json -o reports/sast.json

# Software Composition Analysis (SCA)
safety check -r requirements.txt --json > reports/sca.json

# Container image scanning
trivy image myapp:latest --format json --output reports/trivy.json
```

#### Artifact Stage

```bash
# Build Docker image
docker build -t myapp:${COMMIT_SHA} .

# Tag with version
docker tag myapp:${COMMIT_SHA} registry.example.com/myapp:${COMMIT_SHA}
docker tag myapp:${COMMIT_SHA} registry.example.com/myapp:latest

# Push to registry
docker push registry.example.com/myapp:${COMMIT_SHA}
docker push registry.example.com/myapp:latest
```

---

## 4. Build Triggers

### Trigger Types

| Trigger | When | Use Case |
|---------|------|----------|
| **Push** | Code pushed to a branch | Run CI on every commit |
| **Pull request** | PR opened/updated | Pre-merge validation |
| **Schedule** | Cron-based timing | Nightly builds, dependency checks |
| **Manual** | User clicks a button | Production deployments, one-off tasks |
| **Tag** | Git tag created | Release builds |
| **API/Webhook** | External event | Cross-repo triggers, ChatOps |

### Trigger Configuration Examples

```yaml
# GitHub Actions trigger examples
on:
  # Run on push to main or develop
  push:
    branches: [main, develop]
    paths-ignore:
      - '*.md'
      - 'docs/**'

  # Run on PRs targeting main
  pull_request:
    branches: [main]

  # Nightly security scan at 2 AM UTC
  schedule:
    - cron: '0 2 * * *'

  # Manual trigger with parameters
  workflow_dispatch:
    inputs:
      environment:
        description: 'Target environment'
        required: true
        type: choice
        options:
          - staging
          - production
```

### Path-Based Triggering

```yaml
# Only run backend CI when backend code changes
# Avoids wasting CI resources on unrelated changes
on:
  push:
    paths:
      - 'src/backend/**'
      - 'tests/backend/**'
      - 'requirements.txt'
      - '.github/workflows/backend-ci.yml'
```

---

## 5. Artifact Management

Artifacts are the outputs of a CI build: compiled binaries, Docker images, test reports, packages.

### Artifact Types

| Artifact | Format | Storage |
|----------|--------|---------|
| Docker images | OCI image layers | Container registry (ECR, GCR, Docker Hub) |
| Python packages | `.whl`, `.tar.gz` | PyPI, private index |
| npm packages | `.tgz` | npm registry, Artifactory |
| Java packages | `.jar`, `.war` | Maven Central, Nexus |
| Binaries | Platform-specific executables | GitHub Releases, S3 |
| Test reports | JUnit XML, HTML | CI server, S3 |
| Coverage reports | XML, HTML | CI server, Codecov |

### Versioning Strategies

```bash
# Semantic Versioning (SemVer): MAJOR.MINOR.PATCH
# MAJOR: breaking changes
# MINOR: new features, backwards compatible
# PATCH: bug fixes, backwards compatible

v1.0.0   # Initial release
v1.1.0   # New feature added
v1.1.1   # Bug fix
v2.0.0   # Breaking change

# Git-based versioning for Docker images
# Use commit SHA for traceability
docker build -t myapp:abc123f .

# CalVer (Calendar Versioning)
# Used by Ubuntu (22.04), pip (24.0)
v2024.03.15

# Build number
build-42
```

### Docker Image Tagging Strategy

```bash
# Tag with multiple identifiers for flexibility
IMAGE=registry.example.com/myapp

# Commit SHA -- exact traceability
docker tag myapp $IMAGE:${GIT_SHA:0:7}

# Branch name -- latest build of a branch
docker tag myapp $IMAGE:${BRANCH_NAME}

# Semantic version (on tagged releases)
docker tag myapp $IMAGE:v1.2.3
docker tag myapp $IMAGE:v1.2
docker tag myapp $IMAGE:v1

# Latest (use with caution -- ambiguous in production)
docker tag myapp $IMAGE:latest
```

---

## 6. CI Best Practices

### Keep Builds Fast

```
Target build times:
  Unit tests:        < 2 minutes
  Full CI pipeline:  < 10 minutes
  Nightly/full:      < 30 minutes

Techniques for speed:
  ✓ Parallelize tests across multiple runners
  ✓ Cache dependencies between builds
  ✓ Use incremental builds (only rebuild what changed)
  ✓ Run slow tests in a separate pipeline (nightly)
  ✓ Use fast base images (Alpine, distroless)
```

### Test Pyramid

```
                    /\
                   /  \          End-to-End Tests
                  / E2E\         (few, slow, expensive)
                 /──────\
                /        \       Integration Tests
               /Integration\     (moderate number)
              /──────────────\
             /                \   Unit Tests
            /    Unit Tests    \  (many, fast, cheap)
           /────────────────────\

  More unit tests, fewer E2E tests.
  Fast feedback at the base, confidence at the top.
```

### Hermetic Builds

A hermetic (isolated) build produces the same output regardless of where or when it runs.

```dockerfile
# Bad: Non-hermetic -- depends on host state
FROM python:3.11
RUN pip install -r requirements.txt  # Versions may change over time
COPY . .

# Good: Hermetic -- pinned versions, reproducible
FROM python:3.11.7-slim@sha256:abc123...  # Pinned image digest
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # requirements.txt has pinned versions
COPY . .
```

```
# requirements.txt -- pin exact versions
flask==3.0.2
requests==2.31.0
SQLAlchemy==2.0.25
```

### Fail Fast

```yaml
# Run linting and type checking first (fastest checks)
# If they fail, don't waste time on slower tests
stages:
  - lint        # 30 seconds
  - build       # 1 minute
  - unit-test   # 2 minutes
  - integration # 5 minutes
  - e2e         # 10 minutes

# Each stage only runs if previous stages passed
```

### CI Pipeline as Code

```yaml
# Store CI configuration in the repository alongside the code
# This ensures:
#   - CI changes are reviewed like code (via PRs)
#   - CI configuration is versioned with the code
#   - Anyone can understand the build process
#   - Reproducible builds across branches

# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install flake8
      - run: flake8 src/

  test:
    needs: lint
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --junitxml=reports/test-results.xml
      - uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: reports/test-results.xml
```

---

## 7. Common CI Patterns

### Matrix Builds

Test across multiple versions, OS, or configurations simultaneously.

```yaml
jobs:
  test:
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
        os: [ubuntu-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pytest tests/
```

### Fan-Out / Fan-In

Run independent jobs in parallel, then aggregate results.

```
              ┌── Unit Tests ─────────┐
              │                        │
Build ───────├── Integration Tests ──├──── Deploy
              │                        │
              └── Security Scan ──────┘
              (fan-out: parallel)    (fan-in: wait for all)
```

### Canary Builds

Deploy to a small subset of production before full rollout.

```
Build ──▶ Test ──▶ Deploy to 5% ──▶ Monitor (30 min)
                                          │
                                    ┌─────┴─────┐
                                    │            │
                               Healthy?     Unhealthy?
                                    │            │
                                    ▼            ▼
                             Deploy to       Rollback
                               100%
```

---

## 8. CI Metrics and Monitoring

### Key CI Metrics

| Metric | Target | Why It Matters |
|--------|--------|---------------|
| **Build duration** | < 10 minutes | Slow builds hurt developer productivity |
| **Build success rate** | > 95% | Flaky builds erode trust in CI |
| **Queue time** | < 2 minutes | Long queues mean not enough runners |
| **Test coverage** | > 80% | Coverage below threshold increases risk |
| **Flaky test rate** | < 1% | Flaky tests waste time and build distrust |
| **MTTR for broken builds** | < 30 minutes | Broken main blocks the whole team |

### Monitoring Build Health

```bash
# Track build times over the last 30 days
# (pseudo-query for a CI analytics platform)
SELECT
  date_trunc('day', started_at) as day,
  avg(duration_seconds) as avg_duration,
  count(*) as build_count,
  count(*) filter (where status = 'success') * 100.0 / count(*) as success_rate
FROM builds
WHERE started_at > now() - interval '30 days'
GROUP BY 1
ORDER BY 1;
```

---

## Exercises

### Exercise 1: Design a CI Pipeline

Design a complete CI pipeline for a Python web application (Flask + PostgreSQL). Specify:
1. All stages and their order
2. What runs in each stage (specific commands)
3. Which stages can run in parallel
4. Quality gates (what conditions fail the build)
5. Artifacts produced
6. Estimated total pipeline duration

### Exercise 2: CI vs CD Decision

Your team currently has CI (automated build and test). The CTO wants to move to continuous deployment (auto-deploy every passing build to production). List:
1. Five prerequisites your team must have before adopting continuous deployment
2. Three risks of adopting continuous deployment too early
3. A phased plan to move from CI to continuous delivery to continuous deployment

### Exercise 3: Flaky Test Triage

Your CI pipeline has a 15% flaky test rate. Developers are ignoring failures and re-running builds until they pass.
1. Explain why this is dangerous
2. Propose a systematic approach to identify and categorize flaky tests
3. Design a quarantine mechanism for flaky tests that does not hide real failures
4. Define metrics to track flaky test reduction over time

### Exercise 4: Build Time Optimization

Your CI pipeline takes 25 minutes. The breakdown is:
- Checkout + install dependencies: 4 minutes
- Lint + type check: 2 minutes
- Unit tests: 5 minutes
- Integration tests: 10 minutes
- Docker build + push: 4 minutes

Propose at least five specific optimizations to get the total time under 10 minutes. For each optimization, estimate the time savings and describe any tradeoffs.

---

**Previous**: [Version Control Workflows](./02_Version_Control_Workflows.md) | [Overview](00_Overview.md) | **Next**: [GitHub Actions Deep Dive](./04_GitHub_Actions_Deep_Dive.md)

**License**: CC BY-NC 4.0
