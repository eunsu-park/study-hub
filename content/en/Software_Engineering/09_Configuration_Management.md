# Lesson 09: Configuration Management

**Previous**: [08. Verification and Validation](./08_Verification_and_Validation.md) | **Next**: [10. Project Management](./10_Project_Management.md)

---

A software system is never finished — it is continuously changed, extended, and deployed across environments. Without a disciplined approach to managing these changes, projects devolve into chaos: teams overwrite each other's work, production runs a version nobody can reproduce, and a hotfix introduces a regression that took three months to find. Software Configuration Management (SCM) is the discipline that prevents this. It gives every artifact a precise identity, controls how changes are made, and makes any version of the system reproducible at any time.

**Difficulty**: ⭐⭐⭐

**Prerequisites**:
- Basic familiarity with version control (Git topic recommended)
- Understanding of the software development lifecycle (Lesson 02)
- General programming experience

## Learning Objectives

After completing this lesson, you will be able to:

1. Define Software Configuration Management and explain its role in software engineering
2. Identify and classify configuration items and establish baselines
3. Describe and compare branching strategies for version control
4. Explain build management and the properties of a reproducible build
5. Apply semantic versioning and design a release process
6. Distinguish the steps in a formal change management process
7. Describe environment management and its relationship to infrastructure as code
8. Manage dependencies safely using lockfiles, pinning, and vulnerability scanning

---

## Table of Contents

1. [What Is Software Configuration Management?](#1-what-is-software-configuration-management)
2. [Configuration Items and Baselines](#2-configuration-items-and-baselines)
3. [Version Control Concepts](#3-version-control-concepts)
4. [Build Management](#4-build-management)
5. [Release Management](#5-release-management)
6. [Change Management](#6-change-management)
7. [Configuration Auditing](#7-configuration-auditing)
8. [Environment Management](#8-environment-management)
9. [Dependency Management](#9-dependency-management)
10. [Tools Overview](#10-tools-overview)
11. [Summary](#11-summary)
12. [Practice Exercises](#12-practice-exercises)
13. [Further Reading](#13-further-reading)

---

## 1. What Is Software Configuration Management?

### 1.1 Definition

**Software Configuration Management (SCM)** is the discipline of identifying, tracking, and controlling changes to all artifacts produced during software development, from requirements documents to production binaries.

The IEEE defines it as:
> "A discipline applying technical and administrative direction and surveillance to identify and document the functional and physical characteristics of a configuration item, control changes to those characteristics, record and report change processing and implementation status, and verify compliance with specified requirements."

While that sounds bureaucratic, SCM solves real, urgent problems:

| Problem Without SCM | SCM Solution |
|---------------------|--------------|
| "Which version is in production?" | Baselines and release tagging |
| "Who broke the build and how?" | Version control history + CI/CD logs |
| "We can't reproduce last month's release" | Locked dependencies, build scripts in VCS |
| "Two developers edited the same file" | Branching and merge workflows |
| "A change in module A broke module B" | Change management + impact analysis |
| "The dev environment behaves differently from prod" | Environment management / IaC |

### 1.2 The Four Core SCM Functions

```
┌─────────────────────────────────────────────────────────────────┐
│  Software Configuration Management                              │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │  Identification  │    │    Control       │                  │
│  │                  │    │                  │                  │
│  │ Name and track   │    │ Manage changes   │                  │
│  │ every artifact   │    │ systematically   │                  │
│  └──────────────────┘    └──────────────────┘                  │
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                  │
│  │   Accounting     │    │    Auditing      │                  │
│  │                  │    │                  │                  │
│  │ Record status of │    │ Verify artifacts │                  │
│  │ all CIs; report  │    │ match baseline;  │                  │
│  │ change history   │    │ confirm process  │                  │
│  └──────────────────┘    └──────────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 SCM vs DevOps

SCM predates DevOps by decades (it originated in defense contracting in the 1950s). DevOps adopts and automates SCM principles:

| SCM Concept | DevOps/Modern Practice |
|-------------|------------------------|
| Configuration identification | Git commits and tags; Docker image digests |
| Version control | Git, GitHub/GitLab |
| Change control | Pull requests, code reviews |
| Build management | CI pipelines (GitHub Actions, Jenkins) |
| Release management | CD pipelines, GitOps |
| Environment management | Infrastructure as Code (Terraform, Ansible) |
| Auditing | Audit logs, SBOM (Software Bill of Materials) |

---

## 2. Configuration Items and Baselines

### 2.1 Configuration Items

A **Configuration Item (CI)** is any artifact that is placed under configuration control — given a unique identity and tracked for changes. CIs are not just source code.

| Category | Examples |
|----------|----------|
| **Source code** | Application code, scripts, test code, database migrations |
| **Build artifacts** | Compiled binaries, container images, packages |
| **Documentation** | Requirements, design documents, user manuals, API specs |
| **Test artifacts** | Test plans, test cases, test data, test scripts |
| **Configuration files** | `application.yaml`, `.env.example`, nginx config, Kubernetes manifests |
| **Third-party components** | Vendored libraries, license files |
| **Infrastructure code** | Terraform `.tf` files, Ansible playbooks, Dockerfiles |
| **Project management** | Project plan, risk register, release notes |

**Selection criteria for CIs**: An artifact should be a CI if:
- Its change needs to be tracked and audited
- Multiple versions of it will exist simultaneously
- Its integrity affects the integrity of the system
- Reverting to a previous state may be needed

### 2.2 CI Naming and Identification

Every CI needs a unique identifier that includes:
- **Name**: descriptive, consistent with naming conventions
- **Type**: source, document, test artifact, etc.
- **Version**: a number or hash that distinguishes revisions
- **Variant**: optional, for platform-specific versions (e.g., `linux-amd64`, `macos-arm64`)

```
Naming scheme example:
  {project}-{component}-{version}-{variant}.{ext}

  myapp-api-1.4.2-linux-amd64.tar.gz
  myapp-api-1.4.2-windows-amd64.zip
  myapp-docs-1.4.2.pdf
```

### 2.3 Baselines

A **baseline** is a formally reviewed and agreed-upon snapshot of one or more CIs that serves as a fixed reference for further development. It is the "official version" at a particular point in time.

| Baseline Type | Established at | Contents |
|---------------|----------------|----------|
| **Functional Baseline** | After system requirements review | Approved requirements specification |
| **Allocated Baseline** | After preliminary design review | Approved architecture and high-level design |
| **Product Baseline** | After final acceptance testing | All source code, docs, tests for the release |
| **Operational Baseline** | During operation | Deployed configuration including all patches |

In modern practice, baselines correspond to Git tags on the main branch:

```bash
# Establish product baseline for v1.4.2
git tag -a v1.4.2 -m "Product baseline: Q1 2024 release

Approved by: Product team 2024-03-28
Change control ticket: CCB-2024-031
Includes: api-service, worker-service, migrations 001-047"

git push origin v1.4.2
```

---

## 3. Version Control Concepts

Version control systems (VCS) are the technical backbone of SCM. This section focuses on strategy; mechanics are covered in the Git topic.

### 3.1 Branching Strategies

A branching strategy defines how a team uses branches to manage concurrent development and releases.

#### Gitflow

Suitable for teams with scheduled releases (monthly/quarterly).

```
main          ─────────────●──────────────────●─────────
                          ↑ v1.0              ↑ v2.0
develop       ──────●──────────────●──────────────────●──
                    ↑              ↑
feature/login ──────●              │
                                   │
feature/pay   ────────────────────●│
                                    │
release/2.0   ───────────────────────●────────●──────────
                                    (test)   (merge)
hotfix/2.0.1  ─────────────────────────────────────●─────
```

**Branches**:
- `main`: production-ready code only; always releasable
- `develop`: integration branch; latest delivered development changes
- `feature/*`: individual features, branched from develop
- `release/*`: release preparation (bug fixes only, no features)
- `hotfix/*`: urgent production fixes, branched from main

#### Trunk-Based Development (TBD)

Suitable for teams with continuous delivery (releasing daily or more).

```
main (trunk)  ──●──●──●──●──●──●──●──●──●──●──●──●──
               ↑  ↑     ↑           ↑
             feat1 feat2 feat3     feat4
              (short-lived branches, < 1–2 days)
```

All developers integrate to main frequently (at least once per day). Feature flags control what is enabled in production, decoupling deployment from release.

#### Choosing a Strategy

| Factor | Gitflow | Trunk-Based |
|--------|---------|-------------|
| Release cadence | Scheduled (weeks/months) | Continuous (daily) |
| Team size | Any | Better for smaller, disciplined teams |
| Test automation | Moderate OK | Requires strong automated test suite |
| Feature flags needed? | No | Yes |
| Long-lived branches | Yes (feature branches) | No (< 2 days) |

### 3.2 Tagging and Release Branches

Tags mark specific commits as significant. Every release should be tagged.

```bash
# Annotated tag (recommended for releases — includes metadata)
git tag -a v1.4.2 -m "Release v1.4.2 - Q1 2024"

# List all tags sorted by version
git tag --sort=version:refname

# Create a release branch for emergency patches on an older version
git checkout -b release/1.3 v1.3.0
git cherry-pick abc1234  # apply hotfix
git tag -a v1.3.1 -m "Hotfix v1.3.1"
```

---

## 4. Build Management

A **build** is the process of transforming source code and resources into a deployable artifact. Build management ensures this process is reliable, reproducible, and automated.

### 4.1 Properties of a Good Build System

| Property | Description | Example |
|----------|-------------|---------|
| **Reproducible** | Same inputs always produce identical outputs | Locked dependencies; no random elements |
| **Automated** | No manual steps required | CI pipeline runs build on every commit |
| **Fast** | Incremental — only rebuilds what changed | Make, Gradle incremental compilation |
| **Idempotent** | Running the build twice produces the same result | No side effects from the build itself |
| **Documented** | Build process is defined in code, not tribal knowledge | `Makefile`, `build.gradle`, `pyproject.toml` in VCS |
| **Verified** | Build includes quality gates | Tests, linting, security scans in the pipeline |

### 4.2 Build Scripts and Tools

```makefile
# Makefile: common build automation for polyglot projects
.PHONY: all clean test lint build docker

# Default target
all: lint test build

# Install dependencies
deps:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Run linter
lint:
	flake8 src/ --max-line-length=100
	mypy src/

# Run tests with coverage
test:
	pytest tests/ -v --cov=src --cov-fail-under=85

# Build production artifact
build:
	python -m build --wheel
	@echo "Build artifact: dist/"

# Build Docker image with git SHA as tag
docker:
	docker build \
	  --build-arg GIT_SHA=$(shell git rev-parse --short HEAD) \
	  --build-arg BUILD_DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ) \
	  -t myapp:$(shell git rev-parse --short HEAD) \
	  -t myapp:latest \
	  .

# Clean generated artifacts
clean:
	rm -rf dist/ build/ *.egg-info .coverage htmlcov/ .mypy_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
```

### 4.3 Reproducible Builds

A **reproducible build** produces bit-for-bit identical output given the same source. This enables:
- Verification that a distributed binary corresponds to published source code
- Detection of supply chain attacks (tampered build environment)
- Reliable caching of build artifacts

```dockerfile
# Dockerfile designed for reproducibility
# 1. Pin the base image to a specific digest, not just a tag
FROM python:3.12.3-slim@sha256:a1e3204e39b5f3e2c3f74bc7fdf14e2ae1dfba9ab8bb1cde0f3a5b1c5e2c2d3f

# 2. Set a fixed timestamp for file metadata
ARG SOURCE_DATE_EPOCH=1711670400

# 3. Pin all system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5=15.6-0+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. Install from lockfile (not just requirements.txt)
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY src/ ./src/

# 5. Embed provenance metadata
ARG GIT_SHA
ARG BUILD_DATE
LABEL org.opencontainers.image.revision=$GIT_SHA \
      org.opencontainers.image.created=$BUILD_DATE \
      org.opencontainers.image.source="https://github.com/org/myapp"

CMD ["python", "-m", "myapp"]
```

### 4.4 Artifact Repository

Build artifacts should be stored in an **artifact repository** — a versioned, searchable store for binaries, packages, and container images.

| Artifact Type | Repository |
|---------------|------------|
| Python packages | PyPI, Artifactory, AWS CodeArtifact |
| Java/JVM packages | Maven Central, Nexus |
| Docker images | Docker Hub, ECR, GCR, GHCR |
| Helm charts | Helm repo, OCI registry |
| Generic binaries | Artifactory, S3 (with versioning) |

Never build from source at deploy time. Build once, store the artifact, deploy the artifact.

---

## 5. Release Management

Release management governs how software versions are packaged, versioned, and delivered to users.

### 5.1 Semantic Versioning (SemVer)

Semantic Versioning (semver.org) provides a universal version number grammar:

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]

Examples:
  1.0.0          Initial stable release
  1.0.1          Patch: backward-compatible bug fix
  1.1.0          Minor: backward-compatible new feature
  2.0.0          Major: breaking change (API incompatible)
  2.1.0-alpha.1  Pre-release: unstable, not for production
  2.1.0-rc.1     Release candidate: feature-complete, final testing
  1.0.0+build.42 Build metadata: informational only, ignored in precedence
```

**Version precedence rules**:
```
1.0.0-alpha < 1.0.0-alpha.1 < 1.0.0-beta < 1.0.0-rc.1 < 1.0.0
```

**When to increment**:
- **PATCH**: you fix a bug without changing any public API
- **MINOR**: you add functionality in a backward-compatible manner
- **MAJOR**: you make incompatible API changes

For libraries, strict SemVer protects downstream consumers. For applications (not libraries), the MAJOR version often tracks product generations or annual releases.

### 5.2 Release Notes

Release notes communicate changes to users. A good release note answers:
- What changed?
- Why does it matter to me (the user)?
- Do I need to do anything?

```markdown
# Release Notes: v2.3.0 (2024-03-28)

## What's New

- **Bulk export**: export up to 10,000 records to CSV in one operation (#1847)
- **Dark mode**: full dark mode support across all screens (#2103)
- **API rate limiting**: new `X-RateLimit-*` response headers (#2251)

## Bug Fixes

- Fixed: session expired users were redirected to blank page (#2289)
- Fixed: decimal amounts rounded incorrectly for JPY and KRW (#2301)

## Breaking Changes

- **API**: `GET /users` no longer returns the `password_hash` field.
  Update any clients that read this field. (#2275)

## Deprecations

- `POST /api/v1/auth/login` is deprecated. Use `POST /api/v2/auth/login`.
  v1 will be removed in v3.0.0 (Q3 2024).

## Upgrade Notes

Run the following migration before upgrading:
```
python manage.py migrate --run-syncdb
```

## Known Issues

- Safari 16.x: dark mode toggle may require a page refresh.
  Workaround: use the keyboard shortcut Cmd+Shift+D.
```

### 5.3 The Release Process

```
Feature freeze
      │
      ▼
Create release branch (release/2.3)
      │
      ▼
Regression testing on release branch
      │
      ├── Bug found? → Fix on release branch → cherry-pick to main
      │
      ▼
Release candidate (v2.3.0-rc.1)
      │
      ▼
Staging deployment + acceptance testing
      │
      ├── Issue found? → Increment rc (v2.3.0-rc.2)
      │
      ▼
Production deployment (phased/blue-green)
      │
      ▼
Tag commit: v2.3.0
      │
      ▼
Merge release branch back to main
      │
      ▼
Publish release notes + notify stakeholders
```

---

## 6. Change Management

Change management ensures that every modification to a baselined CI follows a controlled process. It prevents ad-hoc changes that compromise system integrity.

### 6.1 The Change Request Process

```
1. INITIATION
   Anyone can submit a Change Request (CR) or Request for Change (RFC)
   └── CR includes: description, justification, affected components, urgency

2. IMPACT ANALYSIS
   Technical team assesses:
   ├── Which CIs are affected?
   ├── What is the effort/cost to implement?
   ├── What is the risk?
   └── What is the risk of not doing it?

3. REVIEW AND APPROVAL
   Change Control Board (CCB) reviews:
   ├── Approve → schedule and assign
   ├── Defer → add to backlog with conditions
   ├── Reject → document reason
   └── Return for more information

4. IMPLEMENTATION
   Developer implements the change in a controlled branch.

5. VERIFICATION
   Testing confirms the change works and has no regressions.

6. RELEASE
   Change is deployed and the CR is closed.
   Change log is updated.
```

### 6.2 The Change Control Board (CCB)

The CCB (also called Change Advisory Board, CAB) is the decision-making body for changes. Its composition scales with the organization:

| Organization Size | CCB Members | Meeting Cadence |
|-------------------|-------------|-----------------|
| Small startup | Tech lead + PM | Async (Slack/GitHub) |
| Medium company | Eng. manager, QA lead, PM, Ops | Weekly |
| Enterprise | CTO, VPs of Eng/QA/Ops, Security | Biweekly; emergency board for P1 |
| Regulated (banking, medical) | Above + Compliance officer, Legal | Formal, documented |

In Agile teams, the CCB function is often embedded in sprint planning and pull request reviews rather than conducted as a separate body.

### 6.3 Emergency Changes

For critical production incidents, the normal CCB cycle is too slow. An **emergency change process** allows:
1. Fast-tracked approval (2 approvers instead of full CCB)
2. Implementation and deployment within hours
3. **Post-change documentation**: paperwork completed after the change (not before)
4. Mandatory post-incident review within 48–72 hours

```
Emergency change trigger:
  Production outage | Security breach | Data corruption risk

Fast-track approval (any 2 of): Tech Lead, Engineering Manager, On-call SRE

Rollback plan required before deployment

Post-change within 72 hours:
  - Root cause analysis
  - Full CCB-style documentation
  - Lessons learned
```

### 6.4 Impact Analysis

Before approving a change, the CCB needs to understand what it affects:

```python
# Simplified dependency graph impact analysis
def find_impacted_modules(changed_module: str, dependency_graph: dict) -> set:
    """
    BFS from changed module to find all modules that depend on it.
    dependency_graph[A] = [B, C] means A depends on B and C.
    We want to find all modules that (transitively) depend on changed_module.
    """
    # Reverse the graph: who depends on me?
    reverse_graph = {}
    for module, deps in dependency_graph.items():
        for dep in deps:
            reverse_graph.setdefault(dep, set()).add(module)

    impacted = set()
    queue = [changed_module]
    while queue:
        current = queue.pop(0)
        for dependent in reverse_graph.get(current, []):
            if dependent not in impacted:
                impacted.add(dependent)
                queue.append(dependent)
    return impacted

# Example
graph = {
    "checkout": ["cart", "payment", "inventory"],
    "order_history": ["checkout"],
    "admin_dashboard": ["checkout", "order_history"],
    "notification_service": ["checkout"],
}
print(find_impacted_modules("payment", graph))
# {'checkout', 'order_history', 'admin_dashboard', 'notification_service'}
```

---

## 7. Configuration Auditing

Configuration auditing verifies that what was built and deployed matches what was approved and documented. Two types:

### 7.1 Functional Configuration Audit (FCA)

Verifies that the CI's actual performance matches its requirements. Answers: "Does this build do everything it is supposed to do?"

**Checklist**:
- All planned features are present and functioning
- All known defects from previous audits have been resolved
- Test results meet quality exit criteria
- Release notes accurately describe what changed

### 7.2 Physical Configuration Audit (PCA)

Verifies that the CI physically matches its documented description. Answers: "Is what we're about to release exactly what was approved and tested?"

**Checklist**:
- Version numbers in code match version numbers in build artifacts
- All files listed in the build manifest are present
- Checksums of artifacts match recorded values
- No unauthorized files are included
- Dependencies in the artifact match the approved dependency list

```bash
# Verifying artifact integrity with checksums
sha256sum myapp-2.3.0-linux-amd64.tar.gz > myapp-2.3.0.sha256
cat myapp-2.3.0.sha256
# 7a3b9c1d... myapp-2.3.0-linux-amd64.tar.gz

# Verify (by recipient or auditor)
sha256sum --check myapp-2.3.0.sha256
# myapp-2.3.0-linux-amd64.tar.gz: OK
```

### 7.3 Software Bill of Materials (SBOM)

An SBOM is a complete inventory of all components in a software artifact — first-party code, open-source dependencies, and their licenses. Required for:
- Supply chain security (Executive Order 14028, 2021)
- License compliance
- Vulnerability response (know what you have when a CVE is published)

```bash
# Generate SBOM in SPDX format using syft
syft myapp:2.3.0 -o spdx-json > myapp-2.3.0.sbom.json

# Generate SBOM for Python project
pip install cyclonedx-bom
cyclonedx-py --poetry -o sbom.json
```

---

## 8. Environment Management

Software runs in multiple environments, each serving a different purpose. Managing these environments consistently is essential for reliable delivery.

### 8.1 Standard Environments

```
Developer Workstation
        │
        │ (commits + PR)
        ▼
CI Environment (ephemeral — created per build, destroyed after)
        │
        │ (merged to main)
        ▼
Development / Integration Environment
        │
        │ (scheduled promotion)
        ▼
Staging / Pre-production Environment
        │
        │ (approval gate)
        ▼
Production Environment
```

| Environment | Purpose | Who accesses it | Data |
|-------------|---------|----------------|------|
| **CI** | Automated build and test | CI system | Synthetic test data |
| **Development** | Integration testing; demos | Developers, QA | Anonymized or synthetic |
| **Staging** | Pre-release validation | QA, PMs, stakeholders | Production snapshot (anonymized) |
| **Production** | End users | End users | Real data |

**Environment drift** — staging not behaving the same as production — is one of the most common sources of "works on my machine" problems. Infrastructure as Code (IaC) solves this.

### 8.2 Infrastructure as Code (IaC)

IaC defines infrastructure (servers, databases, networks, load balancers) in declarative configuration files stored in version control. The same code provisions all environments, ensuring consistency.

```hcl
# Terraform: provision a web application stack
# environments/staging/main.tf and environments/prod/main.tf
# share the same module — only inputs differ

module "web_app" {
  source = "../../modules/web_app"

  # Environment-specific values (in terraform.tfvars)
  environment     = var.environment     # "staging" or "prod"
  instance_type   = var.instance_type   # "t3.small" vs "t3.large"
  min_instances   = var.min_instances   # 1 vs 3
  max_instances   = var.max_instances   # 2 vs 10
  db_instance     = var.db_instance     # "db.t3.micro" vs "db.r6g.large"
  enable_backups  = var.enable_backups  # false vs true
  alert_endpoints = var.alert_endpoints # dev team vs ops team + PagerDuty
}
```

```yaml
# Ansible: configure application servers consistently
---
- name: Configure web application servers
  hosts: "{{ target_env }}_web"  # staging_web or prod_web
  become: true
  vars_files:
    - "vars/{{ target_env }}.yml"  # environment-specific vars

  roles:
    - common          # OS baseline: NTP, logging, security patches
    - nginx           # Web server, TLS termination
    - app_deploy      # Deploy application artifact from artifact repository
    - monitoring      # Install Prometheus node exporter

  tasks:
    - name: Ensure application is running
      systemd:
        name: myapp
        state: started
        enabled: true
```

### 8.3 Configuration Files and Secrets

Application configuration varies by environment (database URLs, API keys, feature flags). Three tiers:

```
Tier 1: Non-sensitive, environment-specific config
  → Store in environment-specific config files in VCS
  → Example: application-staging.yaml, application-prod.yaml

Tier 2: Sensitive config (credentials, API keys)
  → NEVER store in VCS
  → Store in secrets management system (Vault, AWS Secrets Manager, GCP Secret Manager)
  → Inject at runtime via environment variables or mounted secrets

Tier 3: Feature flags
  → Store in feature flag service (LaunchDarkly, Unleash, Flagsmith)
  → Change without redeployment
```

```bash
# .env.example (committed to VCS — shows what variables are needed, no values)
DATABASE_URL=postgresql://user:password@host:5432/dbname
REDIS_URL=redis://localhost:6379
STRIPE_API_KEY=sk_...
JWT_SECRET=...
LOG_LEVEL=INFO
FEATURE_FLAG_NEW_UI=false

# .env (NOT committed — each environment fills in real values)
# Retrieved from secrets manager at deployment time:
#   aws secretsmanager get-secret-value --secret-id prod/myapp
```

---

## 9. Dependency Management

Modern software is built on hundreds or thousands of third-party libraries. Managing these dependencies safely is a critical SCM responsibility.

### 9.1 Lockfiles

A **lockfile** records the exact versions of every dependency (including transitive dependencies) resolved at a point in time. It ensures every developer and every CI build uses exactly the same versions.

```
Requirements file (describes intent)    Lockfile (records reality)
──────────────────────────────────      ─────────────────────────────
requests>=2.28.0                        requests==2.31.0
flask>=3.0.0                            flask==3.0.2
                                        werkzeug==3.0.1
                                        click==8.1.7
                                        jinja2==3.1.3
                                        markupsafe==2.1.5
                                        ...47 more transitive deps
```

| Ecosystem | Requirements File | Lockfile |
|-----------|------------------|----------|
| Python (pip) | `requirements.txt` | `requirements.lock` (pip-compile) |
| Python (Poetry) | `pyproject.toml` | `poetry.lock` |
| Node.js (npm) | `package.json` | `package-lock.json` |
| Node.js (Yarn) | `package.json` | `yarn.lock` |
| Ruby | `Gemfile` | `Gemfile.lock` |
| Go | `go.mod` | `go.sum` |
| Rust | `Cargo.toml` | `Cargo.lock` |

**Rule**: Always commit lockfiles for applications. For libraries, committing lockfiles is optional (downstream consumers resolve their own).

### 9.2 Version Pinning

**Pinning** specifies an exact version rather than a range. Critical for:
- Production deployments (predictability)
- Security scanning (can definitively check a specific version against CVE databases)

```python
# Unpinned (risky for production)
requests>=2.28.0
flask~=3.0

# Pinned (safe for production)
requests==2.31.0
flask==3.0.2
```

Pinning strategy: pin in lockfiles, not necessarily in `requirements.txt`. This gives you the intent (range) for library compatibility and the guarantee (exact) for reproducibility.

### 9.3 Dependency Vulnerability Scanning

Even pinned, locked dependencies can become vulnerable when a CVE is published after the lockfile was created.

```bash
# Python: pip-audit scans against OSV and PyPA advisory databases
pip install pip-audit
pip-audit -r requirements.lock

# Output:
# Found 2 known vulnerabilities in 1 package
# Name      Version  ID                    Fix Versions
# ──────────────────────────────────────────────────────
# cryptography 38.0.1  GHSA-jfh8-c2jp-jvq8  39.0.1
# cryptography 38.0.1  GHSA-w7pp-m8wf-vj6r  39.0.1

# Node.js: npm audit
npm audit --audit-level=high

# Docker image scanning
docker scout cves myapp:2.3.0
```

Integrate vulnerability scanning into CI pipelines — fail the build on high-severity vulnerabilities in production dependencies.

### 9.4 Dependency Update Strategy

| Strategy | Description | Risk |
|----------|-------------|------|
| **Manual updates** | Developer decides when to update | Frequently outdated; high accumulated risk |
| **Automated PRs** (Dependabot, Renovate) | Bot opens PR for each available update | Many small PRs; manageable with good tests |
| **Scheduled updates** | Dedicate time each sprint to updating deps | Balanced; predictable workload |
| **Stay on LTS** | Only use Long-Term Support versions | Conservative; miss features but stable |

Recommended: use Dependabot or Renovate with auto-merge for patch updates (when tests pass), manual review for minor and major updates.

---

## 10. Tools Overview

### 10.1 Tools by SCM Function

| Function | Tool | Use case |
|----------|------|----------|
| **Version control** | Git | Universal source control |
| **Code hosting / review** | GitHub, GitLab, Bitbucket | PR workflow, code review, CODEOWNERS |
| **CI/CD** | GitHub Actions, Jenkins, GitLab CI, CircleCI | Automated build, test, deploy pipelines |
| **Artifact storage** | JFrog Artifactory, Nexus, AWS ECR, GitHub Packages | Versioned binary storage |
| **IaC provisioning** | Terraform, Pulumi | Cloud infrastructure as code |
| **Configuration management** | Ansible, Chef, Puppet | Server configuration as code |
| **Container orchestration** | Kubernetes, Docker Compose | Environment consistency |
| **Secrets management** | HashiCorp Vault, AWS Secrets Manager | Secure credential storage |
| **Dependency management** | pip-tools, Poetry, Dependabot, Renovate | Lockfiles, automated updates |
| **Vulnerability scanning** | Snyk, Dependabot, Trivy, pip-audit | CVE detection in deps and images |
| **SBOM generation** | syft, CycloneDX | Software Bill of Materials |
| **Feature flags** | LaunchDarkly, Unleash | Decouple deploy from release |

### 10.2 A Complete SCM Toolchain Example

```
Developer commits code (Git)
           │
           ▼
Pull Request opened (GitHub)
  └── Code review (CODEOWNERS enforces required reviewers)
  └── Status checks must pass:
        ├── Linting (GitHub Actions)
        ├── Unit tests + coverage (GitHub Actions)
        ├── Security scan: Semgrep (GitHub Actions)
        └── Dependency vulnerability check: Dependabot alerts
           │
           ▼ (PR merged to main)
           │
CI Pipeline (GitHub Actions)
  ├── Build Docker image
  ├── Run integration tests
  ├── Push image to ECR (tagged with git SHA)
  └── Generate SBOM (syft)
           │
           ▼
CD Pipeline: deploy to staging
  ├── Terraform apply (if IaC changed)
  ├── Ansible playbook (server config)
  └── Kubernetes rolling update
           │
           ▼
Manual approval gate
           │
           ▼
CD Pipeline: deploy to production (blue/green)
  ├── Deploy new version alongside old
  ├── Shift traffic: 10% → 50% → 100% (over 30 min)
  └── Automatic rollback if error rate exceeds threshold
           │
           ▼
Tag release: v2.3.0 (Git tag)
Publish release notes (GitHub Release)
Archive SBOM to artifact repository
```

---

## 11. Summary

Software Configuration Management is the foundation that makes large-scale software development tractable. Without it, systems become unreliable, releases become unpredictable, and teams lose confidence in their ability to change anything safely.

Key takeaways:

- **Configuration items** are any artifacts that need to be tracked — source code, documents, configuration files, infrastructure code, and build artifacts. Every CI has a unique identity and version.
- **Baselines** are formally approved snapshots that serve as reference points. In modern practice, they correspond to Git tags on the main branch.
- **Branching strategies** — Gitflow for scheduled releases, trunk-based development for continuous delivery — provide the workflow structure for managing concurrent changes.
- **Build management** means automated, reproducible builds from version-controlled build scripts. Build once, store the artifact, deploy the artifact.
- **Semantic versioning** gives version numbers precise meaning. MAJOR.MINOR.PATCH signals compatibility guarantees to consumers.
- **Change management** brings discipline to modifications: every change is requested, analyzed for impact, approved, implemented, verified, and closed.
- **Environment management** using IaC ensures dev, staging, and production environments are consistent and reproducible — eliminating "works on staging but not in prod."
- **Dependency management** requires lockfiles for reproducibility, pinning for predictability, and automated vulnerability scanning to catch security issues promptly.

---

## 12. Practice Exercises

**Exercise 1 — Configuration Item Identification**

You are setting up SCM for a new microservice that:
- Is a Python FastAPI application
- Uses PostgreSQL and Redis
- Is deployed to AWS using Terraform
- Has a companion React frontend
- Has API documentation in OpenAPI format

(a) List at least 15 configuration items for this system, categorized by type.
(b) Identify three artifacts that are *not* CIs (should not be version-controlled) and explain why.
(c) Define a naming scheme for Docker image tags that satisfies SCM identification requirements.

**Exercise 2 — Branching Strategy**

A team of 8 developers is building an e-commerce platform. They currently release every two weeks, but want to move toward daily releases within six months.

(a) Recommend a branching strategy for their current two-week release cycle. Draw a simple diagram showing the key branches.
(b) Describe the transition plan to trunk-based development. What must the team have in place before the switch?
(c) A hotfix is needed for a two-week-old release while work on the next release is underway. Describe the exact Git workflow step by step.

**Exercise 3 — Dependency Analysis**

A Python web application's `pyproject.toml` specifies:

```toml
[tool.poetry.dependencies]
python = "^3.11"
django = ">=4.2,<5.0"
celery = "^5.3"
redis = ">=4.6"
boto3 = "*"
```

(a) What are the risks of using `*` (any version) for boto3?
(b) `django = ">=4.2,<5.0"` is used instead of `django = "4.2.8"`. What are the advantages and disadvantages of this approach?
(c) Write a Dependabot configuration (`dependabot.yml`) that: checks for updates weekly, groups all patch updates together, and requires a review from `@security-team` for major version updates.

**Exercise 4 — Change Control**

Your production system processes financial transactions. A developer proposes the following change: replace the legacy MD5-based session token with a cryptographically secure random token (256-bit, base64-encoded). Estimated implementation time: 4 hours.

(a) Write a complete Change Request document for this change, including impact analysis.
(b) What questions should the Change Control Board ask before approving this change?
(c) Design the verification steps that must pass before this change is deployed to production.

**Exercise 5 — Release Management**

A SaaS product is at version `2.7.3`. Determine the appropriate next version number for each of the following changes, and justify your answer:

(a) A bug fix that corrects timezone handling in meeting reminders.
(b) A new "dark mode" setting that users can toggle in their profile.
(c) The REST API's `/users` endpoint now returns `user_id` instead of `id` in the JSON response.
(d) Addition of support for Japanese and Korean languages in the UI, with no API changes.
(e) Removal of the deprecated SOAP API endpoint that was announced for removal 12 months ago.

---

## 13. Further Reading

- **Books**:
  - *Continuous Delivery* — Jez Humble and David Farley. The definitive guide to automated software delivery including build, test, and deployment pipelines.
  - *The Phoenix Project* — Gene Kim, Kevin Behr, George Spafford. A novel that illustrates DevOps and change management concepts through a compelling narrative.
  - *Software Configuration Management Patterns* — Steve Berczuk and Brad Appleton. Practical patterns for SCM in Agile teams.
  - *Infrastructure as Code* (2nd ed.) — Kief Morris. Comprehensive guide to managing infrastructure with code.

- **Standards**:
  - IEEE Std 828-2012 — IEEE Standard for Configuration Management in Systems and Software Engineering
  - NIST SP 800-128 — Guide for Security-Focused Configuration Management of Information Systems
  - Semantic Versioning 2.0.0 — https://semver.org/

- **Tools Documentation**:
  - Git Reference — https://git-scm.com/doc
  - Terraform Documentation — https://developer.hashicorp.com/terraform/docs
  - Ansible Documentation — https://docs.ansible.com/
  - Dependabot Documentation — https://docs.github.com/en/code-security/dependabot
  - Syft (SBOM generator) — https://github.com/anchore/syft

- **Articles and Specifications**:
  - "Gitflow Workflow" — Atlassian Bitbucket guides
  - "Trunk Based Development" — https://trunkbaseddevelopment.com/
  - Reproducible Builds — https://reproducible-builds.org/
  - NTIA Software Bill of Materials — https://www.ntia.gov/sbom

---

## Exercises

### Exercise 1: Identify Configuration Items

You are setting up SCM for a new Python FastAPI service that uses PostgreSQL, is containerized with Docker, deployed to AWS via Terraform, and has API documentation in OpenAPI format.

(a) List at least 12 configuration items, organized by category (source code, build artifacts, documentation, configuration files, infrastructure code).
(b) For each CI, specify its version control strategy (Git repository, artifact registry, or secrets manager) and justify your choice.
(c) Identify two artifacts that should NOT be version-controlled and explain why. Describe how each should be managed instead.

### Exercise 2: Design a Branching Strategy

A team of six developers builds a SaaS HR platform. They currently release monthly but want to move to weekly releases within three months.

(a) Design a branching strategy appropriate for their current monthly release cadence. Describe each long-lived branch and its purpose.
(b) A production security vulnerability is discovered. Walk through the exact Git commands (branch creation, fix, tag, merge) to create and deploy a hotfix without disrupting the next release in progress.
(c) List three technical prerequisites the team must have in place before safely switching to trunk-based development.

### Exercise 3: Semantic Versioning Decisions

A library is currently at version `3.2.1`. Determine the correct next version number for each change, and explain which SemVer rule applies.

(a) A bug fix corrects an off-by-one error in date range calculation.
(b) A new `batch_process()` method is added to the public API; existing methods are unchanged.
(c) The `process()` method's return type changes from `dict` to a typed `Result` dataclass — existing callers that unpack the dict will break.
(d) A deprecated `legacy_process()` method (deprecated since `3.0.0`) is removed.
(e) An internal implementation detail changes to use a faster algorithm; public API is identical.

### Exercise 4: Analyze Dependency Risk

A Python web application's `pyproject.toml` declares:

```toml
[tool.poetry.dependencies]
python = "^3.11"
fastapi = ">=0.100,<1.0"
sqlalchemy = "^2.0"
pydantic = "*"
httpx = "^0.25"
```

(a) Rank the five dependencies from highest to lowest version-resolution risk. Justify each ranking.
(b) The `pydantic` specifier `"*"` resolves to version `1.10.14` today. Six months later, `pydantic 2.0` is released with a breaking API. What risk does this create, and how does a lockfile mitigate it?
(c) Write the three steps needed to safely upgrade `sqlalchemy` from `2.0.x` to `2.1.x` in a team that uses Poetry, Git, and a CI pipeline with automated tests.

### Exercise 5: Write a Change Request

Your production API currently uses HTTP Basic Authentication. The security team has mandated migration to OAuth 2.0 Bearer tokens. This change affects the authentication middleware, all API clients (internal and external), the developer documentation, and three third-party integrations.

Write a complete Change Request document including:
- Change description and business justification
- Affected configuration items (list at least six)
- Impact analysis (schedule, effort, risk)
- Rollback plan
- Verification and acceptance criteria

---

**Previous**: [08. Verification and Validation](./08_Verification_and_Validation.md) | **Next**: [10. Project Management](./10_Project_Management.md)
